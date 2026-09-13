"""
Smoke test for the glacier_inverse library.

Builds a GlacierProblem from the greenland_coarse config and runs a handful of
cheap consistency checks:

  1. Domain shape is factor-aligned (ny, nx divisible by 2**n_levels).
  2. Each MaternPrior was constructed with the hyperparameters in the config.
  3. GGaPPWhiten/GGaPPMap round-trip is approximate identity in both
     directions, for every field prior.
  4. Initial whitened parameters all live on CUDA with requires_grad=True.
  5. The step scheduler snaps onto required times, extends the horizon, and
     rejects impossible requests (pure CPU checks).
  6. Observations built from the config specs are on CUDA, shaped to the
     domain, and carry acquisition times (from file attrs or the t_end
     fallback).
  7. A short forward run at the coarsest level emits a state snapshot at
     every required observation time, produces finite loss terms, and is
     differentiable end-to-end — including through a non-final snapshot.
  7b. The correlated-noise (MaternNoise) likelihood: an ad-hoc whitened
     SurfaceSpec built against the problem's context yields finite residual
     fields on the level's (coarse) grid, a finite loss and gradient, its
     Matérn member round-trips W(Map(ε)) ≈ ε with unit whitened variance,
     the coarse-level model is the physical model re-discretized with
     nugget/2^L, and the weight == 1 / smoothness contracts are enforced.
  7c. The profiled logit nuisance on the Brier terms: the profile objective
     is finite, never exceeds the plain Brier, is non-increasing under warm
     starts, collapses to the plain Brier as sigma_eps -> 0, passes an
     envelope gradient to z_bed, and exposes the fitted eps field; the
     sigma_p reparametrization reproduces the legacy weight exactly and
     enforces the weight == 1 contract.
  8. The enthalpy SMB backend (smb_model="enthalpy") builds, runs, is
     deterministic (fixed weather realization), and is differentiable w.r.t.
     its two whitened (ny, nx) parameter fields — log_H_atm and the logit
     clear-sky fraction (skipped if the installed glare predates it).

Prints PASS/FAIL per check; exits non-zero on any failure.

Run from the project root:
    python smoke_test.py
"""
from __future__ import annotations

import sys
import traceback

import torch

from ggapp.torch import GGaPPMap, GGaPPWhiten

from glacier_inverse import GlacierProblem, MaternNoise, load_config
from glacier_inverse.forward import year_overlap_weights
from glacier_inverse.scheduling import build_step_sequence, merge_times

# Domain to smoke-test. Override at the call site if you want a different one.
SMOKE_DOMAIN = "domains/greenland_coarse"


_failures = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global _failures
    status = "PASS" if condition else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    if not condition:
        _failures += 1


def header(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    config = load_config(SMOKE_DOMAIN)

    header("Building GlacierProblem")
    try:
        problem = GlacierProblem(config)
    except Exception as e:
        traceback.print_exc()
        print(f"FATAL: failed to construct GlacierProblem: {e}")
        return 2
    print(f"  Built. ny={problem.ny}, nx={problem.nx}, dx={problem.dx}")

    header("1. Domain shape")
    factor = 2 ** config.n_levels
    check("ny is factor-aligned", problem.ny % factor == 0,
          f"ny={problem.ny}, factor={factor}")
    check("nx is factor-aligned", problem.nx % factor == 0,
          f"nx={problem.nx}, factor={factor}")
    check("dx > 0", problem.dx > 0, f"dx={problem.dx}")

    header("2. Prior hyperparameters match config")
    priors = problem.priors

    def _prior_matches(model, expected, label):
        try:
            got_sigma = float(model.mg.parameters.sigma.value)
            got_l = float(model.mg.parameters.l.value)
            got_nu = float(model.mg.parameters.nu.value)
        except AttributeError:
            print(f"  [SKIP] {label}: installed ggapp does not expose "
                  f"mg.parameters.*.value; hyperparameters not introspectable")
            return
        ok = (got_sigma == expected.sigma
              and got_l == expected.l
              and got_nu == float(expected.nu))
        check(f"{label} prior σ={expected.sigma}, ℓ={expected.l}, ν={expected.nu}",
              ok,
              f"got σ={got_sigma}, ℓ={got_l}, ν={got_nu}")

    _prior_matches(priors.bed_model,      config.bed_prior,      "bed")
    _prior_matches(priors.mean_model,     config.mean_prior,     "mean")
    _prior_matches(priors.log_beta_model, config.log_beta_prior, "log_beta")
    _prior_matches(priors.pbias_model,    config.pbias_prior,    "pbias")

    header("3. GGaPP whiten/map round-trip")
    rtol = 5e-2  # multigrid Matern solver is iterative — not exact

    def _round_trip(model, label):
        z = torch.randn(problem.ny, problem.nx, dtype=torch.float32, device="cuda")
        phys = GGaPPMap.apply(model, z)
        z_back = GGaPPWhiten.apply(model, phys)
        rel = (z_back - z).norm() / (z.norm() + 1e-12)
        check(f"{label}: ‖W(M(z)) − z‖ / ‖z‖ < {rtol}", rel.item() < rtol,
              f"relative error = {rel.item():.3e}")

    _round_trip(priors.bed_model,      "bed")
    _round_trip(priors.mean_model,     "mean")
    _round_trip(priors.log_beta_model, "log_beta")
    _round_trip(priors.pbias_model,    "pbias")

    # The optional temperature-bias prior: no hierarchy when disabled; a
    # priors-only build (no IceDynamics) exercises the enabled path cheaply.
    if priors.tbias_model is not None:
        _prior_matches(priors.tbias_model, config.tbias_prior, "tbias")
        _round_trip(priors.tbias_model, "tbias")
    else:
        check("tbias prior absent while disabled",
              priors.tbias_model is None)
        import dataclasses
        from glacier_inverse.priors import GlacierPriors as _GP
        tpriors = _GP(dataclasses.replace(config, tbias_enabled=True),
                      problem.ny, problem.nx, problem.dx)
        _prior_matches(tpriors.tbias_model, config.tbias_prior, "tbias (enabled)")
        _round_trip(tpriors.tbias_model, "tbias (enabled)")
        del tpriors

    # The enthalpy SMB parameter fields (log H_atm, logit clear-sky fraction):
    # Matern members gated on smb_model == "enthalpy"; under ETIM they must be
    # None (the z fields sit inert at 0).
    if priors.h_atm_model is not None:
        _prior_matches(priors.h_atm_model, config.h_atm_prior, "h_atm")
        _round_trip(priors.h_atm_model, "h_atm")
        _prior_matches(priors.cloud_model, config.cloud_prior, "cloud")
        _round_trip(priors.cloud_model, "cloud")
        import dataclasses
        from glacier_inverse.priors import GlacierPriors as _GP
        # bed_conditioning=None: a conditioned config would reopen the input
        # netCDF inside this process (the reopen-after-GC HDF hazard) — the
        # throwaway build only needs to prove the SMB gate, not the bed.
        etpriors = _GP(dataclasses.replace(config, smb_model="temperature_index",
                                           bed_conditioning=None),
                       problem.ny, problem.nx, problem.dx)
        check("h_atm/cloud priors absent under temperature_index",
              etpriors.h_atm_model is None and etpriors.cloud_model is None)
        del etpriors
    else:
        check("h_atm/cloud priors absent under temperature_index",
              priors.h_atm_model is None and priors.cloud_model is None)

    header("4. Initial whitened parameters")
    params = problem.params
    for name, tensor in [
        ("z_bed",      params.z_bed),
        ("z_bed_mean", params.z_bed_mean),
        ("z_log_beta", params.z_log_beta),
        ("z_pbias",    params.z_pbias),
        ("z_tbias",    params.z_tbias),
        ("z_log_mf",   params.z_log_mf),
        ("z_log_rf",   params.z_log_rf),
        ("z_log_H_atm",   params.z_log_H_atm),
        ("z_logit_cloud", params.z_logit_cloud),
    ]:
        check(f"{name} on cuda + requires_grad",
              tensor.is_cuda and tensor.requires_grad,
              f"device={tensor.device}, requires_grad={tensor.requires_grad}")
    check("z_log_H_atm / z_logit_cloud are (ny, nx) fields",
          params.z_log_H_atm.shape == (problem.ny, problem.nx)
          and params.z_logit_cloud.shape == (problem.ny, problem.nx),
          f"shapes {tuple(params.z_log_H_atm.shape)}, "
          f"{tuple(params.z_logit_cloud.shape)}")

    header("5. Step scheduler")
    steps = build_step_sequence(t_start=1012.0, t_end=2012.0, dt_max=20.0)
    check("uniform legacy grid: 50 steps of dt=20 ending at 2012",
          len(steps) == 50 and steps[-1] == (2012.0, 20.0),
          f"n={len(steps)}, last={steps[-1]}")

    req = [2000.0, 2013.0, 2015.0, 2020.0]
    steps = build_step_sequence(t_start=1012.0, t_end=2012.0, dt_max=20.0,
                                required_times=req)
    times = [t for t, _ in steps]
    check("required times snapped exactly; horizon extended past t_end",
          all(r in times for r in req) and times[-1] == 2020.0,
          f"tail={times[-5:]}")
    check("all dt positive and <= dt_max",
          all(0 < dt <= 20.0 + 1e-9 for _, dt in steps))

    try:
        build_step_sequence(t_start=1012.0, t_end=2012.0, dt_max=20.0,
                            required_times=[1000.0])
        check("required time <= t_start raises", False)
    except ValueError:
        check("required time <= t_start raises", True)

    check("merge_times dedupes within eps",
          merge_times([2000.0, 2000.0 + 1e-9], [2013.0], None)
          == (2000.0, 2013.0))

    # Anomaly-integration quadrature weights (pure CPU).
    w = year_overlap_weights(1992.0, 2012.0)
    check("year weights: 20 uniform years over a 20-yr step",
          len(w) == 20 and w[0] == (1992, 0.05) and w[-1] == (2011, 0.05)
          and abs(sum(x for _, x in w) - 1.0) < 1e-12)
    w = year_overlap_weights(2011.5, 2013.0)
    check("year weights: partial-year overlap",
          [y for y, _ in w] == [2011, 2012]
          and abs(w[0][1] - 0.5 / 1.5) < 1e-12
          and abs(w[1][1] - 1.0 / 1.5) < 1e-12)
    sub = {y: wt * 8.0 for y, wt in year_overlap_weights(1995.25, 2003.25)}
    for y, wt in year_overlap_weights(2003.25, 2012.75):
        sub[y] = sub.get(y, 0.0) + wt * 9.5
    whole = dict(year_overlap_weights(1995.25, 2012.75))
    check("year weights: sub-partitions recombine (partition independence)",
          set(sub) == set(whole)
          and all(abs(sub[y] / 17.5 - whole[y]) < 1e-12 for y in whole))

    header("6. Observations")
    obs_names = [o.name for o in problem.observations]
    print(f"  products: {obs_names}")
    check("at least srf/vel/extent/bed present",
          {"srf", "vel", "extent", "bed"} <= set(obs_names))

    dom_shape = (problem.ny, problem.nx)
    domain = problem.domain
    for name, tensor in [("dem", domain.dem),
                         ("rgi_mask", domain.rgi_mask),
                         ("domain_mask", domain.domain_mask)]:
        check(f"domain.{name} shape == {dom_shape} and on cuda",
              tuple(tensor.shape) == dom_shape and tensor.is_cuda,
              f"got shape={tuple(tensor.shape)}, device={tensor.device}")

    srf = problem.get_observation("srf")
    vel = problem.get_observation("vel")
    check("S_obs shape == domain and on cuda",
          tuple(srf.S_obs.shape) == dom_shape and srf.S_obs.is_cuda)
    check("u_obs/v_obs shape == domain and on cuda",
          tuple(vel.u_obs.shape) == dom_shape and vel.u_obs.is_cuda
          and tuple(vel.v_obs.shape) == dom_shape)

    bed = problem.get_observation("bed")
    check("bed_obs is 1-D on cuda",
          bed.bed_obs.ndim == 1 and bed.bed_obs.is_cuda,
          f"shape={tuple(bed.bed_obs.shape)}")
    check("bed_normed_coords is (N, 2) on cuda",
          bed.bed_normed_coords.ndim == 2
          and bed.bed_normed_coords.shape[1] == 2
          and bed.bed_normed_coords.is_cuda,
          f"shape={tuple(bed.bed_normed_coords.shape)}")
    check("bed observation is time-independent", bed.required_times == ())

    snow = problem.get_observation("snow")
    if snow is not None:
        check("snow_label shape == domain, in [0,1], on cuda",
              tuple(snow.snow_label.shape) == dom_shape and snow.snow_label.is_cuda
              and snow.snow_label.min() >= 0.0 and snow.snow_label.max() <= 1.0,
              f"range=[{snow.snow_label.min():.3f}, {snow.snow_label.max():.3f}]")
        check("snow_mask is 0/1 with some valid cells",
              set(snow.snow_mask.unique().tolist()) <= {0.0, 1.0}
              and snow.snow_mask.sum() > 0,
              f"valid cells = {int(snow.snow_mask.sum())}")
    else:
        print("  [SKIP] no snowline product for this domain")

    dhdt = problem.get_observation("dhdt")
    if dhdt is not None:
        mode = ("legacy final-step" if dhdt.legacy_final_step
                else f"two-snapshot [{dhdt.t0}, {dhdt.t1}]")
        print(f"  dhdt mode: {mode}")
    else:
        print("  [SKIP] no dhdt product for this domain")

    required = problem.required_times
    check("every timed observation's epoch is in required_times",
          all(t in required for o in problem.observations
              for t in o.required_times),
          f"required_times={required}")

    header("7. Short forward run at coarsest level")
    level = config.n_levels - 1  # 5 for default n_levels=6
    problem.model.set_top_level(level)
    # Start a few coarse steps before the first observation epoch so the run
    # covers every required time; also request one mid-run snapshot to test
    # non-final-state emission even when all products sit at a single epoch.
    t_first = min(required) if required else config.t_end
    t_start = t_first - 3 * config.dt
    t_probe = t_first - config.dt
    try:
        sim, physical = problem.simulate(
            level=level,
            params=params,
            t_start=t_start,
            record_states_at=list(required) + [t_probe],
        )
    except Exception as e:
        traceback.print_exc()
        check("simulate() runs", False, str(e))
        return 1 if _failures else 0

    check("a state was recorded at every required time",
          all(any(abs(ts - t) < 1e-6 for ts in sim.states) for t in required),
          f"recorded={sorted(sim.states.keys())}")
    check("run reached the horizon",
          sim.final.t >= max([config.t_end] + list(required)) - 1e-6,
          f"final t={sim.final.t}")
    check("sim.H finite", torch.isfinite(sim.H).all().item())
    check("sim.u finite", torch.isfinite(sim.u).all().item())
    check("sim.v finite", torch.isfinite(sim.v).all().item())

    # Gradient through a non-final snapshot only.
    probe = sim.at(t_probe).H_fine.sum()
    probe.backward(retain_graph=True)
    check("gradient reaches z_bed through a NON-final snapshot",
          params.z_bed.grad is not None
          and torch.isfinite(params.z_bed.grad).all().item()
          and params.z_bed.grad.abs().sum() > 0,
          f"|grad|_1 = {params.z_bed.grad.abs().sum().item():.3e}"
          if params.z_bed.grad is not None else "no grad")
    params.z_bed.grad = None

    loss_terms = problem.compute_loss(
        sim=sim, physical=physical, params=params)
    J = loss_terms.J
    check("loss is finite", torch.isfinite(J).item(), f"J = {J.item():.4f}")
    for name, term in loss_terms.data_terms.items():
        check(f"{name} term is finite",
              torch.isfinite(torch.as_tensor(term)).item(),
              f"J_{name} = {float(term):.4f}")
    if snow is not None:
        check("snowline term is active (non-zero)",
              float(loss_terms.data_terms["snow"]) != 0.0)
    if dhdt is not None:
        check("dhdt term is active (non-zero)",
              float(loss_terms.data_terms["dhdt"]) != 0.0)
    # The enthalpy-field priors are 0-d and exactly zero at z = 0 (the field
    # parametrization coincides with the old scalar one at the prior median).
    check("J_prior_h_atm / J_prior_cloud are 0-d and zero at z = 0",
          loss_terms.J_prior_h_atm.dim() == 0
          and loss_terms.J_prior_cloud.dim() == 0
          and float(loss_terms.J_prior_h_atm) == 0.0
          and float(loss_terms.J_prior_cloud) == 0.0,
          f"J_prior_h_atm = {float(loss_terms.J_prior_h_atm):.4g}, "
          f"J_prior_cloud = {float(loss_terms.J_prior_cloud):.4g}")

    J.backward(retain_graph=True)
    check("z_bed received a finite gradient",
          params.z_bed.grad is not None
          and torch.isfinite(params.z_bed.grad).all().item(),
          f"grad norm = {params.z_bed.grad.norm().item():.3e}"
          if params.z_bed.grad is not None else "no grad")
    check("z_log_H_atm received a finite (ny, nx) gradient",
          params.z_log_H_atm.grad is not None
          and params.z_log_H_atm.grad.shape == (problem.ny, problem.nx)
          and torch.isfinite(params.z_log_H_atm.grad).all().item()
          and params.z_log_H_atm.grad.abs().sum() > 0,
          f"|grad|_1 = {params.z_log_H_atm.grad.abs().sum().item():.3e}"
          if params.z_log_H_atm.grad is not None else "no grad")

    header("7b. Correlated-noise (MaternNoise) likelihood")
    from glacier_inverse import Schedule
    from glacier_inverse.observations import (SurfaceSpec,
                                              validate_noise_weights)
    # The registry holds one error model per product: reuse the domain's
    # own surface model when its spec carries one (a fresh registration
    # with different hyperparameters would rightly raise).
    _srf_obs = problem.get_observation("srf")
    noise = (_srf_obs.noise if getattr(_srf_obs, "noise", None) is not None
             else MaternNoise(sigma=10.0, l=1000.0, nu=1))
    # Ad-hoc test models must be resolved by the grid (>= 8 cells per l):
    # the wrangell-era 1 km lengths are sub-grid on a km-scale ice-sheet grid.
    l_test = max(1000.0, 8.0 * float(problem.dx))
    try:
        srf_w = SurfaceSpec(noise=noise, weight=1.0).build(problem.build_ctx)
        check("whitened SurfaceSpec builds against problem.build_ctx", True)
    except Exception as e:
        traceback.print_exc()
        check("whitened SurfaceSpec builds against problem.build_ctx", False, str(e))
        srf_w = None
    if srf_w is not None:
        check("noise member is registered on the priors as noise_srf",
              srf_w.noise_model is priors.noise_models["noise_srf"][1]
              and srf_w.sigma == noise.sigma)
        again = SurfaceSpec(noise=noise, weight=1.0).build(problem.build_ctx)
        check("re-registering the same noise model is idempotent",
              again.noise_model is srf_w.noise_model)
        try:
            SurfaceSpec(noise=MaternNoise(20.0, 1000.0), weight=1.0) \
                .build(problem.build_ctx)
            check("re-registering with different hyperparameters raises", False)
        except ValueError:
            check("re-registering with different hyperparameters raises", True)

        cfg0 = config.at_iteration(0, 0, schedule=False)
        mask = (domain.rgi_mask * domain.domain_mask).to(torch.float32)
        res = srf_w.residuals(sim=sim, physical=physical, config=cfg0,
                              domain=domain, mask=mask, dx=problem.dx)
        # Section 7 ran at the coarsest level: the surface term is evaluated
        # in coarse space, so its residual fields live on that grid.
        f = 2 ** level
        check("residuals() returns raw and whitened fields on the LEVEL's grid",
              set(res) == {"r", "z"}
              and res["z"].shape == (problem.ny // f, problem.nx // f)
              and torch.isfinite(res["z"]).all().item()
              and res["z"] is not res["r"])
        from glacier_inverse.priors import SpectralMaternNoise as _SMN
        nm_L = srf_w.noise_model_at(level)
        check("coarse-level noise model: spectral, nugget scaled by 1/2^L, "
              "same sigma/l/nu",
              isinstance(nm_L, _SMN) and nm_L.hp.nugget == noise.nugget / f
              and (nm_L.hp.sigma, nm_L.hp.l, nm_L.hp.nu) == (noise.sigma, noise.l, noise.nu)
              and nm_L.dx == problem.dx * f)
        check("coarse-level whitening of a coarse white draw has unit variance",
              abs((GGaPPWhiten.apply(nm_L, GGaPPMap.apply(
                  nm_L, torch.randn(problem.ny // f, problem.nx // f,
                                    device="cuda"))) ** 2).mean().item() - 1) < 0.2)
        S_obs_L = srf_w.S_obs_at(level)
        check("S_obs_at(level) is the box-restricted DEM",
              S_obs_L.shape == res["r"].shape
              and abs(S_obs_L.mean().item() - srf_w.S_obs.mean().item()) < 1e-2)
        J_w = srf_w.loss(sim=sim, physical=physical, config=cfg0,
                         domain=domain, mask=mask, dx=problem.dx, weight=1.0)
        check("whitened surface loss is finite", torch.isfinite(J_w).item(),
              f"J_srf_whitened = {J_w.item():.4f}")
        params.z_bed.grad = None
        J_w.backward(retain_graph=True)
        check("whitened loss gives z_bed a finite, non-zero gradient",
              params.z_bed.grad is not None
              and torch.isfinite(params.z_bed.grad).all().item()
              and params.z_bed.grad.abs().sum() > 0,
              f"grad norm = {params.z_bed.grad.norm().item():.3e}"
              if params.z_bed.grad is not None else "no grad")
        params.z_bed.grad = None

        # The member is a Matérn model like any prior: W(Map(ε)) ≈ ε and the
        # whitened image of a correlated draw has unit variance.
        eps = torch.randn(problem.ny, problem.nx, dtype=torch.float32,
                          device="cuda")
        corr = GGaPPMap.apply(srf_w.noise_model, eps)
        z_back = GGaPPWhiten.apply(srf_w.noise_model, corr)
        rel = (z_back - eps).norm() / (eps.norm() + 1e-12)
        check(f"noise member: ‖W(M(ε)) − ε‖ / ‖ε‖ < {rtol}", rel.item() < rtol,
              f"relative error = {rel.item():.3e}")
        m2 = (z_back ** 2).mean().item()
        check("noise member: mean(z²) ≈ 1 for a correlated draw",
              abs(m2 - 1.0) < 0.2, f"mean(z²) = {m2:.3f}")
        # randomized() perturbs by a correlated field with the model's marginal std.
        pert = srf_w.randomized(eps_S=eps).S_obs - srf_w.S_obs
        check("randomized() perturbation has marginal std ≈ noise.sigma",
              abs(pert.std().item() / noise.sigma - 1.0) < 0.25,
              f"std = {pert.std().item():.2f} (sigma = {noise.sigma})")

        # Nugget path: the spectral operator reproduces ggapp's stencil at
        # nugget = 0 and is an exact self-inverse pair with a nugget.
        from glacier_inverse.priors import SpectralMaternNoise
        spec0 = SpectralMaternNoise(noise, problem.ny, problem.nx, problem.dx)
        z_spec = GGaPPWhiten.apply(spec0, corr)
        rel = (z_spec - z_back).norm() / (z_back.norm() + 1e-12)
        check("SpectralMaternNoise(nugget=0) matches the ggapp whitening",
              rel.item() < 1e-4, f"relative error = {rel.item():.3e}")
        noise_n = MaternNoise(sigma=10.0, l=l_test, nu=1, nugget=3.0)
        spec_n = priors.noise_model("srf_nugget_test", noise_n)
        check("noise_model with a nugget returns a SpectralMaternNoise",
              isinstance(spec_n, SpectralMaternNoise))
        spec_h = priors.noise_model("srf_halfnu_test",
                                    MaternNoise(sigma=10.0, l=l_test, nu=0.5))
        z_h = GGaPPWhiten.apply(spec_h, GGaPPMap.apply(spec_h, eps))
        rel = (z_h - eps).norm() / (eps.norm() + 1e-12)
        check("nu = 1/2 (exponential) model: spectral round-trip exact",
              isinstance(spec_h, SpectralMaternNoise) and rel.item() < 1e-4,
              f"relative error = {rel.item():.3e}")
        v_h = (GGaPPMap.apply(spec_h, eps) ** 2).mean().sqrt().item()
        check("nu = 1/2 model: Map(ε) has marginal std ≈ sigma",
              abs(v_h / 10.0 - 1.0) < 0.3, f"std = {v_h:.2f}")
        z_n = GGaPPWhiten.apply(spec_n, GGaPPMap.apply(spec_n, eps))
        rel = (z_n - eps).norm() / (eps.norm() + 1e-12)
        check("nugget model: ‖W(M(ε)) − ε‖ / ‖ε‖ < 1e-4 (exact spectral pair)",
              rel.item() < 1e-4, f"relative error = {rel.item():.3e}")
        white = GGaPPWhiten.apply(spec_n, noise_n.nugget * eps)
        check("nugget model: pure pixel noise at the nugget std whitens to < 1",
              (white ** 2).mean().item() < 1.0,
              f"mean(z²) = {(white ** 2).mean().item():.3f}")
        srf_n = SurfaceSpec(noise=noise_n, weight=1.0)
        try:
            SurfaceSpec(noise=MaternNoise(10.0, l_test, nugget=1.0), weight=1.0) \
                .build(problem.build_ctx)
            check("re-registering srf with a different nugget raises", False)
        except ValueError:
            check("re-registering srf with a different nugget raises", True)
        del srf_n

        import cupy as cp
        import dataclasses
        # Discrepancy component (marginalized Kennedy–O'Hagan model error):
        # C = C_Matern + nugget^2 I + C_disc. Whitening by the sum caps the
        # information about a spatially CONSTANT residual (the product/
        # prediction level) at ~one observation of error sigma_D, while the
        # fine scales keep their weight; the spectral pair stays exactly
        # inverse; sampling from the summed model whitens to unit variance.
        hp_base = MaternNoise(10.0, l_test, nu=0.5, nugget=5.0)
        hp_disc = MaternNoise(10.0, l_test, nu=0.5, nugget=5.0,
                              discrepancy=MaternNoise(20.0, 60000.0, nu=1.0))
        m_base = problem.priors.noise_model("srf_disc_base", hp_base)
        m_disc = problem.priors.noise_model("srf_disc_test", hp_disc)
        check("MaternNoise with discrepancy is not stencil-compatible",
              not hp_disc.stencil_compatible)
        const = cp.ones((problem.ny, problem.nx), dtype=cp.float32)
        info_base = float((cp.asarray(m_base.whiten(const)) ** 2).sum())
        info_disc = float((cp.asarray(m_disc.whiten(const)) ** 2).sum())
        # One observation of a unit offset at sigma_D = 20 per l_D-sized
        # patch -> n_patch * (1/20)^2, n_patch = max(1, 2A/(pi l_D^2)).
        import math as _mm
        _A = problem.ny * problem.nx * float(problem.dx) ** 2
        _n_patch = max(1.0, 2.0 * _A / (_mm.pi * 60000.0 ** 2))
        check("discrepancy caps the constant-mode information near "
              "n_patch (c/sigma_D)^2",
              info_disc < 5.0 * _n_patch * (1.0 / 20.0) ** 2
              and info_base / info_disc > 50.0,
              f"info(const): base={info_base:.4g} disc={info_disc:.4g} "
              f"(1/sigma_D^2 = {(1.0/20.0)**2:.4g}, "
              f"ratio {info_base/info_disc:.3g}x)")
        eps_w = cp.random.standard_normal(
            (problem.ny, problem.nx), dtype=cp.float32)
        rt = float(cp.linalg.norm(m_disc.whiten(m_disc.forward(eps_w)) - eps_w)
                   / cp.linalg.norm(eps_w))
        zsq = float((cp.asarray(m_disc.whiten(m_disc.forward(eps_w))) ** 2).mean())
        check("discrepancy model: exact spectral pair and unit whitened draw",
              rt < 1e-4 and abs(zsq - 1.0) < 0.05,
              f"round-trip rel err = {rt:.3e}, mean(z^2) = {zsq:.3f}")
        same = problem.priors.noise_model("srf_disc_test", hp_disc)
        check("discrepancy model registry is idempotent (nested equality)",
              same is m_disc)
        try:
            problem.priors.noise_model(
                "srf_disc_test", dataclasses.replace(
                    hp_disc, discrepancy=MaternNoise(30.0, 60000.0, nu=1.0)))
            check("re-registering with a different discrepancy raises", False)
        except ValueError:
            check("re-registering with a different discrepancy raises", True)
        try:
            MaternNoise(10.0, 1000.0,
                        discrepancy=MaternNoise(20.0, 60000.0, nugget=1.0))
            check("discrepancy with a nugget raises", False)
        except ValueError:
            check("discrepancy with a nugget raises", True)
        del m_base, m_disc, const, eps_w

        # Contracts.
        try:
            validate_noise_weights([SurfaceSpec(noise=noise, weight=2e-6)
                                    .build(problem.build_ctx)])
            check("weight != 1 on a whitened term raises", False)
        except ValueError:
            check("weight != 1 on a whitened term raises", True)
        try:
            validate_noise_weights([
                SurfaceSpec(noise=noise, weight=Schedule(
                    final=1.0, ramp=lambda i: 0.0 if i < 10 else 1.0))
                .build(problem.build_ctx)])
            check("Schedule(final=1) with a zero ramp passes the contract", True)
        except ValueError as e:
            check("Schedule(final=1) with a zero ramp passes the contract",
                  False, str(e))
    check("MaternNoise: even / fractional nu route to the spectral path",
          not MaternNoise(10.0, 1000.0, nu=2).stencil_compatible
          and not MaternNoise(10.0, 1000.0, nu=0.5).stencil_compatible
          and MaternNoise(10.0, 1000.0, nu=3).stencil_compatible
          and not MaternNoise(10.0, 1000.0, nu=1, nugget=1.0).stencil_compatible)
    try:
        MaternNoise(10.0, 1000.0, nu=0.0)
        check("MaternNoise rejects nu <= 0", False)
    except ValueError:
        check("MaternNoise rejects nu <= 0", True)
    header("7c. Profiled logit nuisance (extent/snowline)")
    from glacier_inverse.observations import ExtentSpec, SnowlineSpec
    nuis_noise = MaternNoise(sigma=3.0, l=2000.0, nu=1)
    ext_plain = ExtentSpec(weight=2e-4, s_H=10.0).build(problem.build_ctx)
    ext_nuis = ExtentSpec(weight=2e-4, s_H=10.0,
                          logit_error=nuis_noise).build(problem.build_ctx)
    kw = dict(sim=sim, physical=physical, config=cfg0, domain=domain,
              mask=mask, dx=problem.dx, weight=2e-4)
    J_plain = ext_plain.loss(**kw)
    J_n1 = ext_nuis.loss(**kw)
    J_n2 = ext_nuis.loss(**kw)
    st = ext_nuis.logit_nuisance.last
    check("profiled extent loss is finite and includes a prior cost >= 0",
          torch.isfinite(J_n1).item() and st["prior"] >= 0.0,
          f"J_plain={J_plain.item():.3f} J_nuis={J_n1.item():.3f} "
          f"prior={st['prior']:.3f} |eps|max={st['eps_absmax']:.2f} cg={st['cg']}")
    check("profile objective <= plain Brier (eps = 0 is feasible)",
          J_n1.item() <= J_plain.item() * (1 + 1e-4))
    check("warm-started second call does not increase the profile",
          J_n2.item() <= J_n1.item() * (1 + 1e-4))
    # The legacy Brier scale is weight*dx^2, so "tiny" must shrink with the
    # cell size for the prior to dominate the profile on a coarse grid.
    ext_tiny = ExtentSpec(weight=2e-4, s_H=10.0,
                          logit_error=MaternNoise(0.01 * min(1.0, 90.0 / float(problem.dx)),
                                                  max(2000.0, 4.0 * float(problem.dx)))) \
        .build(problem.build_ctx)
    J_tiny = ext_tiny.loss(**kw)
    check("sigma_eps -> 0 recovers the plain Brier",
          abs(J_tiny.item() - J_plain.item()) < 0.02 * abs(J_plain.item()) + 1e-3,
          f"J_tiny={J_tiny.item():.3f} vs J_plain={J_plain.item():.3f}")
    params.z_bed.grad = None
    J_n2.backward(retain_graph=True)
    check("profiled extent loss gives z_bed a finite gradient (envelope)",
          params.z_bed.grad is not None
          and torch.isfinite(params.z_bed.grad).all().item(),
          f"grad norm = {params.z_bed.grad.norm().item():.3e}"
          if params.z_bed.grad is not None else "no grad")
    params.z_bed.grad = None
    res_n = ext_nuis.residuals(sim=sim, physical=physical, config=cfg0,
                               domain=domain, mask=mask, dx=problem.dx)
    fL = 2 ** level
    check("residuals() exposes the fitted logit_eps on the level grid",
          set(res_n) == {"logit_eps"}
          and res_n["logit_eps"].shape == (problem.ny // fL, problem.nx // fL))
    if snow is not None:
        snow_nuis = SnowlineSpec(weight=1e-4, s_smb=0.5,
                                 logit_error=MaternNoise(2.0, 3000.0)) \
            .build(problem.build_ctx)
        J_s = snow_nuis.loss(**{**kw, "weight": 1e-4})
        check("profiled snowline loss is finite",
              torch.isfinite(J_s).item(), f"J_snow_nuis={J_s.item():.3f}")
    try:
        LN_bad = MaternNoise(3.0, 2000.0, nugget=1.0)
        ExtentSpec(weight=2e-4, logit_error=LN_bad).build(problem.build_ctx)
        check("LogitNuisance rejects a nugget", False)
    except ValueError:
        check("LogitNuisance rejects a nugget", True)

    # Bounded nuisance (eps_max): ε = eps_max·tanh(u/eps_max). Below the
    # bound it is the same Gaussian model; a full class flip is unreachable,
    # so the constrained profile sits between the unbounded profile and the
    # plain Brier, and |ε| stays strictly inside the bound.
    ext_bnd = ExtentSpec(weight=2e-4, s_H=10.0, logit_error=nuis_noise,
                         eps_max=0.5).build(problem.build_ctx)
    J_b1 = ext_bnd.loss(**kw)
    J_b = ext_bnd.loss(**kw)   # warm-started second call
    st_b = ext_bnd.logit_nuisance.last
    check("bounded nuisance: |eps| <= eps_max and profile between "
          "unbounded and plain",
          st_b["eps_absmax"] <= 0.5 + 1e-6
          and J_b.item() >= J_n2.item() * (1 - 1e-3)
          and J_b.item() <= J_plain.item() * (1 + 1e-4),
          f"J_plain={J_plain.item():.3f} >= J_bounded={J_b.item():.3f} >= "
          f"J_unbounded={J_n2.item():.3f}, |eps|max={st_b['eps_absmax']:.3f}")
    check("bounded nuisance reports saturation where the data over-asks",
          st_b.get("saturated_frac", 0.0) > 0.0,
          f"saturated_frac={st_b.get('saturated_frac'):.3f}")
    ext_inf = ExtentSpec(weight=2e-4, s_H=10.0, logit_error=nuis_noise,
                         eps_max=1e6).build(problem.build_ctx)
    ext_inf.loss(**kw)
    J_inf = ext_inf.loss(**kw)
    # Tolerance covers two independently warm-started nuisance states plus
    # the wrangell avalanche atomicAdd jitter (measured flake at 1.1e-3).
    check("eps_max -> inf recovers the unbounded profile",
          abs(J_inf.item() - J_n2.item()) < 5e-3 * abs(J_n2.item()) + 1e-4,
          f"J_inf={J_inf.item():.4f} vs J_unbounded={J_n2.item():.4f}")
    try:
        ExtentSpec(weight=2e-4, logit_error=nuis_noise, eps_max=0.0) \
            .build(problem.build_ctx)
        check("LogitNuisance rejects eps_max <= 0", False)
    except ValueError:
        check("LogitNuisance rejects eps_max <= 0", True)

    # sigma_p reparametrization: exactly the legacy weight·dx²/s_B² form
    # under sigma_p = s_B/sqrt(2·w·dx²), and weight == 1 by contract.
    import math as _math
    w_leg = 2e-4
    sp_eq = 0.5 / _math.sqrt(2.0 * w_leg * problem.dx ** 2)
    ext_sp = ExtentSpec(weight=1.0, s_H=10.0, sigma_p=sp_eq) \
        .build(problem.build_ctx)
    J_sp = ext_sp.loss(**{**kw, "weight": 1.0})
    check("sigma_p form reproduces the legacy weighted Brier exactly",
          abs(J_sp.item() - J_plain.item()) < 1e-3 * abs(J_plain.item()),
          f"J_sigma_p={J_sp.item():.4f} vs J_legacy={J_plain.item():.4f} "
          f"(sigma_p={sp_eq:.3f})")
    try:
        validate_noise_weights([ExtentSpec(weight=2e-4, sigma_p=0.3)
                                .build(problem.build_ctx)])
        check("sigma_p with weight != 1 raises", False)
    except ValueError:
        check("sigma_p with weight != 1 raises", True)

    try:
        MaternNoise(10.0, 1000.0, nugget=-1.0)
        check("MaternNoise rejects a negative nugget", False)
    except ValueError:
        check("MaternNoise rejects a negative nugget", True)

    header("7e. Fingerprint nuisance (rank-few model-error marginalization)")
    from glacier_inverse.config import FingerprintNuisance
    from glacier_inverse.observations import profile_fingerprints

    # Unit check: with a residual exactly along one fingerprint and a huge
    # Huber threshold (quadratic regime), the profile matches the closed
    # form c* = a|w|^2/(|w|^2 + 1/s^2) and the downdated objective the
    # Sherman-Morrison value.
    wf = torch.randn(64, 64, device="cuda")
    wf = wf / wf.norm() * 30.0            # |w|^2 = 900
    a = 0.7
    z_syn = a * wf
    zd, fprior, c = profile_fingerprints(z_syn, [(wf, 1.0)], nu=1e3)
    c_star = a * 900.0 / 901.0
    F_star = 0.5 * a ** 2 * 900.0 / 901.0
    F_got = float(0.5 * (zd ** 2).sum() + fprior)
    check("fingerprint profile matches the quadratic closed form",
          abs(float(c[0]) - c_star) < 1e-3
          and abs(F_got - F_star) < 1e-3 * (1 + F_star),
          f"c={float(c[0]):.5f} vs {c_star:.5f}; "
          f"J={F_got:.4f} vs {F_star:.4f} (undowndated {0.5*a**2*900:.1f})")

    # Integration: measure real fingerprints on the smoke domain's problem (an
    # ad-hoc whitened surface term; 1 + 2*2 forwards at the coarse level),
    # then check the downdate: J_fp <= J_plain (c = 0 feasible), c-hat
    # finite, envelope gradient reaches z_bed.
    # Same hyperparameters as 7b's registration — the noise-model registry is
    # idempotent per product name and would reject a mismatch.
    srf_fp = SurfaceSpec(noise=noise, weight=1.0).build(problem.build_ctx)
    # refresh_fingerprints keys its targets by obs.name: replace the domain's
    # own "srf" (which may carry a noise model) instead of adding a second.
    _obs_backup = list(problem.observations)
    problem.observations = [o for o in _obs_backup if o.name != "srf"] + [srf_fp]
    try:
        fpn = FingerprintNuisance(params=("log_H_atm", "logit_cloud"),
                                  s=(1.0, 1.0), n_modes=2, fd_step=0.5)
        problem.refresh_fingerprints(params=params, level=level, fp=fpn)
        kw_fp = dict(sim=sim, physical=physical, config=cfg0, domain=domain,
                     mask=mask, dx=problem.dx, weight=1.0)
        J_fp = srf_fp.loss(**kw_fp)
        c_hat = srf_fp.fingerprint_c
        srf_fp.set_fingerprints(None, level)
        J_plain_fp = srf_fp.loss(**kw_fp)
        check("fingerprints installed and profile <= plain whitened loss",
              c_hat is not None and c_hat.shape == (4,)
              and torch.isfinite(c_hat).all().item()
              and J_fp.item() <= J_plain_fp.item() * (1 + 1e-6),
              f"J_fp={J_fp.item():.4f} <= J_plain={J_plain_fp.item():.4f}, "
              f"c-hat={[f'{v:+.3f}' for v in c_hat.tolist()]}")
        params.z_bed.grad = None
        J_fp.backward(retain_graph=True)
        check("fingerprint-downdated loss gives z_bed a finite gradient",
              params.z_bed.grad is not None
              and torch.isfinite(params.z_bed.grad).all().item())
        params.z_bed.grad = None
    finally:
        problem.observations = _obs_backup
    try:
        FingerprintNuisance(params=("log_H_atm",), s=(1.0, 2.0))
        check("FingerprintNuisance rejects mismatched s/params", False)
    except ValueError:
        check("FingerprintNuisance rejects mismatched s/params", True)

    header("7f. Likelihood-side influence control (eta + bounded-influence caps)")
    from glacier_inverse.loss import apply_influence_control
    ls = float(cfg0.loss_scale)
    # Nonzero z so the prior-gradient path is nontrivial.
    with torch.no_grad():
        params.z_log_H_atm.add_(0.3)
    gH0 = params.z_log_H_atm.grad.detach().clone()
    gC0 = params.z_logit_cloud.grad.detach().clone()
    gpH = ls * params.z_log_H_atm.detach()
    gpC = ls * params.z_logit_cloud.detach()
    apply_influence_control(params, eta=0.25, loss_scale=ls)
    errH = (params.z_log_H_atm.grad - (0.25 * (gH0 - gpH) + gpH)).abs().max()
    errC = (params.z_logit_cloud.grad - (0.25 * (gC0 - gpC) + gpC)).abs().max()
    check("eta = 0.25: data component scaled, prior component exact",
          errH.item() < 1e-10 and errC.item() < 1e-10,
          f"max err = {max(errH.item(), errC.item()):.2e}")
    params.z_log_H_atm.grad.copy_(gH0)
    apply_influence_control(params, eta=0.0, loss_scale=ls)
    check("eta = 0 (cut): block gradient collapses to the prior's exactly",
          torch.equal(params.z_log_H_atm.grad, gpH))
    params.z_log_H_atm.grad.copy_(gH0)
    apply_influence_control(params, eta=1.0, loss_scale=ls)
    check("eta = 1 (full Bayes) is a bit-identical no-op",
          torch.equal(params.z_log_H_atm.grad, gH0))

    # Bounded-influence cap: unsaturated regime ~ identity; saturated regime
    # caps the data-score norm at loss_scale*C_z with direction preserved.
    n0 = float((gH0 - gpH).norm())
    C_big = 100.0 * n0 / ls          # x = 0.01, tanh(x)/x ~ 1 - 3e-5
    params.z_log_H_atm.grad.copy_(gH0)
    sat = apply_influence_control(params, loss_scale=ls,
                                  caps={"z_log_H_atm": C_big})
    dev = (params.z_log_H_atm.grad - gH0).norm() / (gH0.norm() + 1e-30)
    check("cap, unsaturated (x = 0.01): essentially full Bayes",
          sat["z_log_H_atm"] < 0.02 and dev.item() < 1e-3,
          f"x = {sat['z_log_H_atm']:.3f}, rel change = {dev.item():.2e}")
    C_small = 0.01 * n0 / ls         # x = 100, deeply saturated
    params.z_log_H_atm.grad.copy_(gH0)
    sat = apply_influence_control(params, loss_scale=ls,
                                  caps={"z_log_H_atm": C_small})
    gd = params.z_log_H_atm.grad - gpH
    cos = float((gd * (gH0 - gpH)).sum()
                / (gd.norm() * (gH0 - gpH).norm() + 1e-30))
    check("cap, saturated (x = 100): |g_data| = loss_scale*C_z, direction kept",
          abs(float(gd.norm()) - ls * C_small) < 1e-6 * ls * C_small
          and cos > 1 - 1e-6 and sat["z_log_H_atm"] > 99.0,
          f"|g_data| = {float(gd.norm()):.3e} vs C~ = {ls*C_small:.3e}, "
          f"cos = {cos:.6f}, x = {sat['z_log_H_atm']:.0f}")
    # Log transfer: non-saturating — |g_data| = C~*log(1+x), direction kept;
    # near-identity when unsaturated (log1p(x)/x = 0.995 at x = 0.01).
    params.z_log_H_atm.grad.copy_(gH0)
    sat = apply_influence_control(params, loss_scale=ls, transfer="log",
                                  caps={"z_log_H_atm": C_big})
    dev = (params.z_log_H_atm.grad - gH0).norm() / (gH0.norm() + 1e-30)
    check("log transfer, unsaturated: essentially full Bayes",
          dev.item() < 6e-3, f"rel change = {dev.item():.2e}")
    params.z_log_H_atm.grad.copy_(gH0)
    sat = apply_influence_control(params, loss_scale=ls, transfer="log",
                                  caps={"z_log_H_atm": C_small})
    gd = params.z_log_H_atm.grad - gpH
    import math as _m
    want = ls * C_small * _m.log1p(sat["z_log_H_atm"])
    check("log transfer, saturated: |g_data| = C~*log(1+x) (non-saturating)",
          abs(float(gd.norm()) - want) < 1e-5 * want
          and want > ls * C_small,   # exceeds the tanh ceiling: not bounded
          f"|g_data| = {float(gd.norm()):.3e} vs C~*log1p(x) = {want:.3e} "
          f"(x = {sat['z_log_H_atm']:.0f}, {_m.log1p(sat['z_log_H_atm']):.1f} e-folds)")
    try:
        apply_influence_control(params, loss_scale=ls, transfer="sigmoid",
                                caps={"z_log_H_atm": 1.0})
        check("unknown influence_transfer raises", False)
    except ValueError:
        check("unknown influence_transfer raises", True)

    # Effective-dof scaling: the joint cap is sqrt(d_eff)*C_z. Resolver:
    # d_eff = max(1, 2A/(pi l^2)) from each parameter prior's correlation
    # area; explicit (C, d) tuples pass through; float == (C, 1) exactly.
    from glacier_inverse.loss import resolve_influence_caps
    rc = resolve_influence_caps(
        {"z_log_H_atm": 2.0, "z_tbias": 2.0, "z_log_mf": 1.0,
         "z_logit_cloud": (3.0, 7.0)},
        config, problem.ny, problem.nx, problem.dx)
    import math as _m
    A_dom = problem.ny * problem.nx * problem.dx ** 2
    d_hatm = max(1.0, 2.0 * A_dom / (_m.pi * config.h_atm_prior.l ** 2))
    d_tb = max(1.0, 2.0 * A_dom / (_m.pi * config.tbias_prior.l ** 2))
    check("resolver: d_eff from prior correlation areas; override and "
          "scalar fallback honored",
          abs(rc["z_log_H_atm"][1] - d_hatm) < 1e-6 * d_hatm
          and abs(rc["z_tbias"][1] - d_tb) < 1e-6 * d_tb
          and rc["z_tbias"][1] >= rc["z_log_H_atm"][1]
          and rc["z_logit_cloud"] == (3.0, 7.0)
          and rc["z_log_mf"][1] == 1.0,
          f"d(h_atm)={rc['z_log_H_atm'][1]:.1f}, d(tbias)={rc['z_tbias'][1]:.0f}")
    params.z_log_H_atm.grad.copy_(gH0)
    apply_influence_control(params, loss_scale=ls,
                            caps={"z_log_H_atm": (C_small, 4.0)})
    n4 = float((params.z_log_H_atm.grad - gpH).norm())
    check("sqrt(d) scaling: (C, d=4) doubles the saturated cap of (C, d=1)",
          abs(n4 - 2.0 * ls * C_small) < 1e-5 * ls * C_small,
          f"|g_data| = {n4:.3e} vs 2*C~ = {2*ls*C_small:.3e}")
    params.z_log_H_atm.grad.copy_(gH0)
    apply_influence_control(params, loss_scale=ls,
                            caps={"z_log_H_atm": (C_small, 1.0)})
    g_tuple = params.z_log_H_atm.grad.detach().clone()
    params.z_log_H_atm.grad.copy_(gH0)
    apply_influence_control(params, loss_scale=ls,
                            caps={"z_log_H_atm": C_small})
    check("float cap value is exactly (C, d=1)",
          torch.equal(params.z_log_H_atm.grad, g_tuple))
    try:
        apply_influence_control(params, eta=1.5, loss_scale=ls)
        check("eta outside [0, 1] raises", False)
    except ValueError:
        check("eta outside [0, 1] raises", True)
    try:
        apply_influence_control(params, loss_scale=ls, caps={"z_nope": 1.0})
        check("unknown influence_cap key raises", False)
    except ValueError:
        check("unknown influence_cap key raises", True)
    with torch.no_grad():
        params.z_log_H_atm.sub_(0.3)
    params.z_log_H_atm.grad = None
    params.z_logit_cloud.grad = None

    header("7d. Checkpoint save/load + scalar->field conversion")
    import tempfile
    from glacier_inverse.io import save_whitened_params, load_whitened_params_into
    with tempfile.TemporaryDirectory() as tdir:
        ckpt = f"{tdir}/torch_vars.p"
        save_whitened_params(problem.params, ckpt,
                             bed_parametrization=problem.priors.bed_parametrization)
        fresh = problem.params.detach_clone()
        fresh.z_log_H_atm += 1.0  # ensure the load actually overwrites
        load_whitened_params_into(fresh, ckpt, priors=problem.priors)
        check("field checkpoint round-trips (z_log_H_atm bit-identical)",
              fresh.z_log_H_atm.shape == (problem.ny, problem.nx)
              and torch.equal(fresh.z_log_H_atm.detach(),
                              problem.params.z_log_H_atm.detach()))

        # Legacy (pre-field) checkpoint: 0-d scalars, no tag. A constant is
        # an eigenfunction of the mirror-Neumann Matern operator, so the
        # whitening step of the conversion is exact; reading it back through
        # Map (iterative multigrid solves) carries the usual round-trip
        # error, so the tolerance matches the section-3 round-trip scale.
        legacy = torch.load(ckpt)
        del legacy["smb_scalar_parametrization"]
        z0 = 0.7
        legacy["log_H_atm"] = torch.tensor(z0, device="cuda")
        legacy["logit_cloud"] = torch.tensor(-0.3, device="cuda")
        torch.save(legacy, ckpt)
        conv = problem.params.detach_clone()
        load_whitened_params_into(conv, ckpt, priors=problem.priors)
        if problem.priors.h_atm_model is not None:
            v = problem.priors.sigma_log_H_atm * z0
            mapped = GGaPPMap.apply(problem.priors.h_atm_model,
                                    conv.z_log_H_atm.detach())
            rel = (mapped - v).abs().max() / abs(v)
            check("scalar->field conversion reproduces the constant offset",
                  conv.z_log_H_atm.shape == (problem.ny, problem.nx)
                  and rel.item() < 2e-2,
                  f"max |Map(z) - v| / |v| = {rel.item():.3e} "
                  f"(v = {v:.4g})")
        del fresh, conv, legacy

    header("8. Enthalpy SMB backend")
    try:
        from glare.enthalpy import EnthalpyModel  # noqa: F401
        enthalpy_available = True
    except ImportError:
        enthalpy_available = False
    if not enthalpy_available:
        print("  [SKIP] installed glare does not provide EnthalpyModel")
    else:
        # Free the temperature-index problem's graph/state before standing up
        # a second full problem on the same GPU.
        del sim, physical, loss_terms, probe, J
        del problem, params, srf, vel, bed, snow, dhdt, domain, priors
        srf_w = again = res = J_w = corr = z_back = pert = None
        spec0 = spec_n = z_spec = z_n = white = spec_h = z_h = None
        ext_plain = ext_nuis = ext_tiny = J_plain = J_n1 = J_n2 = J_tiny = None
        ext_sp = J_sp = None
        res_n = snow_nuis = None
        torch.cuda.empty_cache()

        import dataclasses
        econfig = dataclasses.replace(config, smb_model="enthalpy")
        try:
            eproblem = GlacierProblem(econfig)
        except Exception as e:
            traceback.print_exc()
            check("enthalpy GlacierProblem builds", False, str(e))
            print()
            print(f"FAILED: {_failures} check(s) did not pass")
            return 1
        check("smb model is EnthalpyModel",
              type(eproblem.smb_model).__name__ == "EnthalpyModel")
        check("initial enthalpy smb finite",
              bool(torch.isfinite(torch.tensor(
                  eproblem.smb_model.grid.state.smb.data)).all()))
        has_dif = "monthly_diffuse_potential" in eproblem.gridded_data
        dif = eproblem.insol_dif
        if not has_dif:
            # Optional product, handled gracefully (zeros + warning) like
            # snowline/debris — a data-vintage nag, not a code failure.
            print("  [WARN] no diffuse-sky potential in GLIDE_inputs.nc — "
                  "the diffuse shortwave term is zero; rebuild with "
                  "make_insolation.py --diffuse-only to exercise it")
        check("insol_dif is (12, ny, nx), finite, in [0, 1]",
              tuple(dif.shape) == (12,) + tuple(eproblem.domain.dem.shape)
              and torch.isfinite(dif).all().item()
              and dif.min().item() >= 0.0 and dif.max().item() <= 1.0 + 1e-6,
              f"shape={tuple(dif.shape)} range=[{dif.min().item():.3f}, {dif.max().item():.3f}]")

        eparams = eproblem.params
        elevel = econfig.n_levels - 1
        eproblem.model.set_top_level(elevel)
        erequired = eproblem.required_times
        et_first = min(erequired) if erequired else econfig.t_end
        et_start = et_first - 3 * econfig.dt
        try:
            esim, ephys = eproblem.simulate(level=elevel, params=eparams,
                                            t_start=et_start)
        except Exception as e:
            traceback.print_exc()
            check("enthalpy simulate() runs", False, str(e))
            print()
            print(f"FAILED: {_failures} check(s) did not pass")
            return 1
        check("enthalpy sim.H finite", torch.isfinite(esim.H).all().item())
        check("enthalpy smb_fine finite",
              torch.isfinite(esim.final.smb_fine).all().item())

        # The fixed weather realization: the grid must hold exactly the seeded
        # deviation sequence after a run (an unseeded redraw would differ), and
        # a re-run with the same physical parameters must reproduce smb to well
        # under the O(0.1 m/yr) a redraw would cause. Bitwise equality is only
        # spoiled by the avalanche operator's atomicAdd deposits (~1e-5,
        # order-nondeterministic, pre-existing on the ETIM path too); without
        # the avalanche model the re-run is exact.
        import cupy as _cp
        import numpy as _np
        check("grid.temp_dev holds the fixed seeded realization",
              _np.array_equal(_cp.asnumpy(eproblem.smb_model.grid.temp_dev),
                              eproblem.temp_dev))
        with torch.no_grad():
            esim2 = eproblem.simulate_physical(level=elevel, physical=ephys,
                                               t_start=et_start)
        smb_diff = (esim.final.smb_fine - esim2.final.smb_fine).abs().max()
        exact = torch.equal(esim.final.smb_fine, esim2.final.smb_fine)
        check("re-run smb matches (fixed realization, no redraw)",
              exact or (econfig.use_avalanche_model
                        and smb_diff.item() < 1e-3),
              f"max |Δsmb| = {smb_diff.item():.3e}"
              + ("" if exact else " (avalanche atomicAdd jitter)"))
        del esim2

        eloss = eproblem.compute_loss(sim=esim, physical=ephys, params=eparams)
        eJ = eloss.J
        check("enthalpy loss finite", torch.isfinite(eJ).item(),
              f"J = {eJ.item():.4f}")
        eJ.backward()
        for name, tensor in [("z_log_H_atm", eparams.z_log_H_atm),
                             ("z_logit_cloud", eparams.z_logit_cloud),
                             ("z_bed", eparams.z_bed)]:
            check(f"gradient reaches {name}",
                  tensor.grad is not None
                  and torch.isfinite(tensor.grad).all().item()
                  and tensor.grad.abs().sum() > 0,
                  f"|grad|_1 = {tensor.grad.abs().sum().item():.3e}"
                  if tensor.grad is not None else "no grad")
        # The inactive model's scalar sits at z = 0, so its only gradient path
        # (the prior term) contributes exactly zero — no data-loss leakage.
        check("z_log_mf gradient is exactly zero (inactive model)",
              eparams.z_log_mf.grad is None
              or float(eparams.z_log_mf.grad.abs()) == 0.0,
              f"grad = {float(eparams.z_log_mf.grad):.3e}"
              if eparams.z_log_mf.grad is not None else "no grad")

    header("9. Bed GP conditioning (posterior-as-prior)")
    try:
        from ggapp.conditioning import ConditionedPrior  # noqa: F401
        from ggapp.torch import GGaPPCondition  # noqa: F401
        conditioning_available = True
    except ImportError:
        conditioning_available = False
    if not conditioning_available:
        print("  [SKIP] installed ggapp predates conditioning "
              "(no ggapp.conditioning)")
    else:
        import dataclasses
        import cupy as _cp
        from glacier_inverse.config import BedConditioningConfig
        from glacier_inverse.priors import (
            GlacierPriors, _cropped_inputs, build_bed_conditioning_data)

        # All file IO happens BEFORE the enthalpy-problem teardown: on this
        # HDF5 stack, opening a netCDF after other handles on it have been
        # garbage-collected can segfault (hence also the eager loads in
        # priors._cropped_inputs).
        # Tight PCG tolerances: these checks probe the conditioning ALGEBRA,
        # so the solver must not be the error budget (the round-trip chains
        # condition/Map/whiten/S^{-1} applications, and the default adjoint
        # rtol of 1e-2 alone puts ~10% into the bed-space reconstruction).
        ccfg = dataclasses.replace(
            config, bed_conditioning=BedConditioningConfig(
                enabled=True, pcg_rtol=1e-5, pcg_rtol_adjoint=1e-5))
        gd = _cropped_inputs(
            ccfg, variables=["elevation", "rgi_mask", "domain_mask"])
        cny, cnx = gd.sizes["y"], gd.sizes["x"]
        cdx = (gd.x[1] - gd.x[0]).item()
        cpriors = GlacierPriors(ccfg, cny, cnx, cdx)
        _, D_picks = build_bed_conditioning_data(dataclasses.replace(
            ccfg, bed_conditioning=BedConditioningConfig(
                enabled=True, include_off_ice=False)))
        dem = torch.tensor(gd.elevation.values, dtype=torch.float32,
                           device="cuda")
        off_ice = torch.tensor((gd.rgi_mask.values == 0)
                               | (gd.domain_mask.values == 0)).cuda()

        if enthalpy_available:
            del esim, ephys, eloss, eJ, eparams, eproblem
            torch.cuda.empty_cache()

        cond = cpriors.bed_conditioner
        check("conditioner is built", cond is not None)
        check("pcg knobs reached the layer",
              cond.rtol == ccfg.bed_conditioning.pcg_rtol
              and cond.maxiter == ccfg.bed_conditioning.pcg_maxiter)
        n_pick_cells = int((D_picks > 0).sum())
        check("flightline picks landed on the grid", n_pick_cells > 100,
              f"{n_pick_cells} pick cells")

        # z = 0 must give the kriging posterior mean: ~DEM off-ice, ~picks at
        # pick cells (within a few sigma; the data dominate the flat prior).
        z0 = torch.zeros(cny, cnx, dtype=torch.float32, device="cuda")
        with torch.no_grad():
            bed_mean_krig = cpriors.bed_from_whitened(z0, z0)[0]
        check("PCG converged (cold solve)", cond.last_converged,
              f"{cond.last_iters} iterations")
        off_err = (bed_mean_krig - dem)[off_ice].abs().mean().item()
        check("z=0 ⇒ kriging mean ≈ DEM off-ice",
              off_err < ccfg.bed_conditioning.sigma_dem,
              f"mean |bed − DEM| off-ice = {off_err:.2f} m")
        at_picks = torch.tensor(D_picks > 0).cuda()
        b_full = torch.tensor(cond.data)
        pick_err = (bed_mean_krig - b_full)[at_picks].abs()
        # Gridded bed data (Greenland) enter at their own PER-CELL error
        # (BedMachine errbed spans 10 m to many hundreds of m), so judge the
        # kriging mean per cell in units of that cell's sigma.
        _D = torch.tensor(D_picks).cuda()
        _sig_cell = torch.where(_D > 0, _D, torch.ones_like(_D)).rsqrt()
        _sig_cell = torch.maximum(_sig_cell, torch.tensor(float(ccfg.bed_conditioning.sigma_picks)).cuda())
        pick_err_n = ((bed_mean_krig - b_full).abs() / _sig_cell)[at_picks]
        check("kriging mean honors the picks",
              pick_err_n.mean().item() < 1.0
              and pick_err_n.max().item() < 5.0,
              f"mean/max |bed − b|/sigma at picks = {pick_err_n.mean().item():.2f}"
              f"/{pick_err_n.max().item():.2f} (max |bed − b| {pick_err.max().item():.0f} m)")

        # Conditional round-trip via the checkpoint-conversion algebra
        # (S^{-1} = I + C·D): z -> bed -> z -> bed. Measured in BED space —
        # that is what a converted warm start must reproduce. (The z-space
        # error is amplified by ||S^{-1}|| ~ (sigma_prior/sigma_obs)^2 and is
        # meaninglessly large at solver tolerance.)
        zr = torch.randn(cny, cnx, dtype=torch.float32, device="cuda")
        with torch.no_grad():
            bed_r = cpriors.bed_from_whitened(zr, z0)[0]
        x_rec = cond.latent_from_field(
            _cp.asarray(bed_r), _cp.zeros((cny, cnx), dtype=_cp.float32))
        z_rec = torch.tensor(cpriors.bed_model.whiten(x_rec))
        with torch.no_grad():
            bed_rec = cpriors.bed_from_whitened(z_rec, z0)[0]
        rt_rel = ((bed_rec - bed_r).norm() / bed_r.norm()).item()
        check("conditional round-trip (conversion algebra, bed space) < 5e-2",
              rt_rel < 5e-2, f"relative bed error = {rt_rel:.3e}")

        # Conditional variance: collapsed at data, ~prior sigma far away
        # (6 samples — a loose, cheap check).
        with torch.no_grad():
            draws = torch.stack([
                cpriors.bed_from_whitened(
                    torch.randn(cny, cnx, dtype=torch.float32,
                                device="cuda"), z0)[0]
                for _ in range(6)])
        sd = draws.std(dim=0)
        sd_data = sd[off_ice].mean().item()
        check("posterior sd collapsed at data cells",
              sd_data < 5 * ccfg.bed_conditioning.sigma_dem,
              f"mean sd off-ice = {sd_data:.1f} m (prior "
              f"σ = {ccfg.bed_prior.sigma})")
        from scipy.ndimage import distance_transform_edt
        dist_px = distance_transform_edt(
            _cp.asnumpy(cond.precision) == 0)
        far = torch.tensor(dist_px > 3 * ccfg.bed_prior.l / cdx).cuda()
        if bool(far.any()):
            sd_far = sd[far].mean().item()
            check("posterior sd ≈ prior σ far from data",
                  0.3 * ccfg.bed_prior.sigma < sd_far < 1.8 * ccfg.bed_prior.sigma,
                  f"mean sd far = {sd_far:.1f} m (prior σ = {ccfg.bed_prior.sigma},"
                  f" 6-sample estimate)")
        else:
            print("  [SKIP] no cells farther than 3ℓ from data on this domain")

        # Gradient flows through the conditioning and differs from the
        # unconditional map's gradient.
        zg = torch.randn(cny, cnx, dtype=torch.float32, device="cuda",
                         requires_grad=True)
        wq = torch.randn(cny, cnx, dtype=torch.float32, device="cuda")
        bed_g = cpriors.bed_from_whitened(zg, z0)[0]
        (wq * bed_g).sum().backward()
        g_cond = zg.grad.clone()
        zg2 = zg.detach().clone().requires_grad_()
        (wq * GGaPPMap.apply(cpriors.bed_model, zg2)).sum().backward()
        g_diff = (g_cond - zg2.grad).norm() / zg2.grad.norm()
        check("gradient through conditioning finite & nonzero",
              torch.isfinite(g_cond).all().item()
              and g_cond.abs().sum().item() > 0)
        check("conditioned gradient differs from unconditional",
              g_diff.item() > 1e-3,
              f"relative difference = {g_diff.item():.3e}")

    print()
    if _failures:
        print(f"FAILED: {_failures} check(s) did not pass")
        return 1
    print("OK: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
