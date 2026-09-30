#!/usr/bin/env python
"""
Evaluate a calving sweep (sweep_calving_c.py) with the inversion's OWN
objective, split by glacier basin, and assemble the per-basin margins.

The discrete optimization uses precisely the J of the continuous one: the
observation terms are built by GlacierProblem from the domain config (so
the products, the per-pixel sigmas, the Matern noise models and the
weights are the inversion's), each term's whitened residual comes from its
own `residuals()`, and its per-cell loss -- the summand of `loss()` -- is
summed over the cells of each basin (`model_inputs/calving_basins.nc`,
preprocessing/make_calving_basins.py). The extent term's global logit
nuisance is fitted once per run by `loss()` as in the inversion and its
per-cell Brier split afterwards. Summing the basins (plus the unassigned
cells) reproduces the term's J to round-off; the totals are printed as the
check. Terms: the surface (2008), the velocity (2018), the extent (2015)
and the MEaSUREs / ITS_LIVE dh/dt over 1993-2019 (added here when the
config has it commented out -- the same spec as the inversion's). ATL15
is NOT used: EN4's thermal forcing carries a ~3-year phase error at the
critical fronts (Jakobshavn: the late-2010s cold wave arrives in the early
2020s), so over a 6-year window the forcing error dominates. The snowline
term is left out: it does not respond to the margin.

`--basin-var calving_reach` sums each glacier's J over its calving reach
only (the fjord plus the trough under the ice within max_km of the
terminus, make_calving_basins.py) instead of the whole drainage basin: the
cells a margin change can move, without the interior's constant offset.

The states come from the runs' raw state files at the epochs
(forward_standalone.save_state: staggered surface velocity, unmasked SMB,
active mask, H) rebuilt as ModelState at the run's level -- the terms then
prolong / restrict exactly as in the inversion, so a coarse-level sweep
evaluates too.

Selection (`--assemble`): per basin with a marine terminus, J_i(c) over
the sweep; the response to c is a threshold, so the CENTRE of the best
plateau is taken (the c values within `--plateau` of the range above the
minimum, their median), not the edge; a basin whose J does not respond to
c (range below `--min-range` of its J) takes its REGION's median c over
the responsive basins (global median as the fallback), as do basins
without a marine terminus and the unassigned cells. Written as
`model_inputs/calving_h0_base.nc` (`h0_base`, m, on the full grid; load it
with `ocean_forcing.rho_filename="calving_h0_base.nc"`: OceanForcing adds
h0_base(x) to the margin, h0 = calving_h0 + h0_base + alpha_h dTF) and
`calving_c_basins.csv`. One joint run of the assembled field is the
actual test of the independence approximation (checked 2026-09-29 on v1.1:
the composite reproduces each glacier's sweep state -- Helheim, Petermann,
Kanger, Rink identical -- Jakobshavn excepted, which interacts).

The sweep runs h0 = c alone (forward_standalone's H00 = c), while the
composite adds `calving_h0` to h0_base; so the field is written as
h0_base = c - config.calving_h0 and the composite equals the sweep whatever
the config's baseline (before 2026-09-29 it was c, and a config with
calving_h0 = 10 shifted every front 10 m toward calving -- enough to flip
Helheim, Jakobshavn and Petermann, whose good windows are 10-25 m wide).

BASIN-INTEGRATED TERMS (CPU, from the state files; the whitened per-cell
terms cannot see a smooth trunk-scale error and are exquisitely sensitive
to a front position no model will place exactly): `J_dhdt_int`, the
volume-integrated 1993-2019 rate over a zone, and `J_gate`, the discharge
through the basin's Mankoff gates (calving_basins.csv `gates`) at
`--gate-year` (2018, the velocity epoch) against the SAME integral of the
observations the inversion fits -- the ITS_LIVE mosaic x `thickness_obs`
of the model inputs on the model grid, chain convention (analysis/gates.py)
-- as chi2/2 of the log ratio, sigma `--gate-sigma` (0.15), `--gate-eps`
(0.1 Gt/yr) regularizing the log. Mankoff's D stays validation: only our
own velocity and thickness enter. Basins without gates contribute 0.
Select with e.g. `--select-terms gate dhdt_int`.
FLUX ALONE IS NOT IDENTIFYING (2026-09-29): flux = speed x thickness, and a
filled fjord (Kangerlussuaq at c -12.5: gate thickness 1.94x, speed 0.42x)
matches the observed flux as well as the observed front does. `J_gatef`
(`--select-terms gatef`) penalizes the two factors separately -- the
flux-weighted (weights L |v_obs| H_obs) gate speed and thickness ratios over
the basin's gates, chi2/2 of log(ratio) / `--gate-speed-sigma` (0.15) and
/ `--gate-thick-sigma` (0.2; looser because the spin-up thickens the trunks
1.1-1.3x even with the fronts pinned) -- strictly more information than the
flux, still integrated over the basin's gates and still only our own
observations. SURFACE ADMISSIBILITY (`--admissible-term srf`): per basin
only the c values whose J_srf is within (1 + `--admissible-rel`) of that
basin's minimum (default 9, i.e. 10x) are candidates; the filled-fjord
branch costs 15-50x the minimum there (Kanger 185 vs 6, Helheim 330 vs 10,
Jakobshavn 209 vs 4, Daugaard-Jensen 64 vs 4) while states on the right
branch vary by 1-5x, so the surface term picks the BRANCH and the gate
factors pick the member. Basins without gates fall back to J_srf within
the admissible set.

STAGE 1 AS THE SOURCE OF TRUTH (`--reference-run DIR`, 2026-09-29, the
user's design): stage 1 (the inversion, fronts pinned to the observed
history) has already extracted everything the products hold; stage 2 only
emulates its boundary condition as a function of basin and thermal
forcing. With a reference run -- `forward_standalone.py --stage1`, raw
states at the epochs in {output_dir}/stage1_reference -- every integrated
term compares with THAT run instead of the products: J_gate / J_gatef with
its gate flux, speed and thickness (weights L |v_ref| H_ref), J_dhdt_int
with its reach-integrated 1993-2019 rate, and a new `J_srf_ref`, chi2/2 of
(S - S_ref) / `--srf-ref-sigma` (10 m) at `--srf-year` (2008) over each
basin's cells holding ice in either run -- the admissibility term under
this design (`--admissible-term srf_ref`). The whitened per-cell terms of
the GPU evaluation still refer to the products and are best left out of
the selection. `J_gatefm` (`--select-terms gatefm`, reference only) is
J_gatef summed over `--gate-epochs` (1990 2008 2015 2018), each epoch's
speed / thickness ratios against the reference AT THAT EPOCH (weights from
its flux density): a 2018 snapshot cannot tell a steady front from one
that reached the 2018 state through a collapse inside the window (the CW
overshoot of the first stage-1-referenced composite, 2006-20 -94 vs -53).
Why: against the products the gate term chases a flux the
model only carries while its outlets retreat (0.99 of Mankoff at 2018 and
GrIS 2006-20 -347 vs -239 Gt/yr); against stage 1 the target is a state
the model holds.

Usage:
  python analysis/sweep_calving_eval.py --sweep-root domains/greenland/inverse_v9/sweep_c
  python analysis/sweep_calving_eval.py --sweep-root ... --assemble [--plateau 0.1]
"""
import argparse
import dataclasses
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from calving_screen import load_config  # noqa: E402
from basin_mass_balance import crop_to_factor  # noqa: E402
from glacier_inverse.observations import DhdtSpec  # noqa: E402
from glacier_inverse.observations import MaternNoise  # noqa: E402
from glacier_inverse.forward import ModelState, SimResult  # noqa: E402
from glacier_inverse import observations as om  # noqa: E402

MEASURES_SPEC = DhdtSpec(filename="gridded_dhdt_measures.nc", name="dhdt_measures",
                         noise=MaternNoise(sigma=0.5, l=10000.0, nu=0.5, nugget=0.5),
                         sigma_floor=0.1, sigma_rel=0.25, sigma_rel_km=5.0, weight=1.0, nu=3)
TERM_SPECS = {"srf": "SurfaceSpec", "vel": "VelocitySpec", "extent": "ExtentSpec", "dhdt_measures": "DhdtSpec"}
DEV = "cuda"


from glacier_inverse.loss import _huber as huber  # noqa: E402  the terms' own pseudo-Huber


def build_problem(cfg, terms):
    """GlacierProblem with only the requested observation terms (the config's
    specs; the MEaSUREs dh/dt added when absent)."""
    specs = []
    for s in cfg.observations:
        tn = type(s).__name__
        if tn == "DhdtSpec":
            if getattr(s, "name", None) == "dhdt_measures" and "dhdt_measures" in terms:
                specs.append(s)
            continue
        key = {v: k for k, v in TERM_SPECS.items()}.get(tn)
        if key in terms:
            specs.append(s)
    if "dhdt_measures" in terms and not any(getattr(s, "name", None) == "dhdt_measures" for s in specs):
        specs.append(MEASURES_SPEC)
    cfg2 = dataclasses.replace(cfg, observations=tuple(specs))
    from glacier_inverse.problem import GlacierProblem
    prob = GlacierProblem(cfg2)
    print("observation terms:", [o.name for o in prob.observations], flush=True)
    return prob


def load_state(path, bed_fine, cfg):
    """A ModelState on the run's level from a raw state file."""
    ds = xr.open_dataset(path)
    level = int(ds.attrs["level"])
    T = lambda v: torch.as_tensor(np.asarray(v, dtype=np.float32), device=DEV)
    H, u, v = T(ds.H.values), T(ds.u_s.values), T(ds.v_s.values)
    smb, active = T(ds.smb.values), T(ds.mask.values)
    bed = om._restrict(bed_fine, level) if level > 0 else bed_fine
    st = ModelState(t=float(ds.attrs["time"]), dt_step=1.0, level=level, u=u, v=v,
                    ud=torch.zeros_like(u), vd=torch.zeros_like(v), H=H, active=active,
                    smb_fine=smb, smb_coarse=smb, bed_coarse=bed,
                    flotation_factor=1.0 - cfg.rho_ice / cfg.rho_water, n_glen=float(cfg.n_glen))
    ds.close()
    return st


def to_fine(l, level):
    """Spread a level-L per-cell loss field onto the fine grid, mass-conserving."""
    if level == 0:
        return l
    f = 2 ** level
    return torch.repeat_interleave(torch.repeat_interleave(l, f, dim=0), f, dim=1) / f ** 2


def per_cell_losses(obs, sim, cfg_it, prob, mask, level):
    """The summand of obs.loss(), per fine-grid cell (weighting and loss_scale
    included), and the term's total as loss() reports it."""
    kw = dict(sim=sim, physical=None, config=cfg_it, domain=prob.domain, mask=mask, dx=prob.dx)
    w = obs.weight_at(0, level, schedule=False)
    scale = cfg_it.loss_scale
    if isinstance(obs, om.VelocityObservation):
        r = obs.residuals(**kw)
        z2 = r["z_u"] ** 2 + r["z_v"] ** 2
        l = scale * w * obs.nu ** 2 * (torch.sqrt(1.0 + z2 / obs.nu ** 2) - 1.0)
        return l, float(obs.loss(**kw, weight=w))
    if isinstance(obs, (om.DhdtObservation, om.SurfaceObservation)):
        z = obs.residuals(**kw)["z"]
        l = scale * w * huber(z, obs.nu)
        lvl = level if isinstance(obs, om.SurfaceObservation) else 0
        return to_fine(l, lvl), float(obs.loss(**kw, weight=w))
    if isinstance(obs, om.ExtentObservation):
        J = float(obs.loss(**kw, weight=w))           # fits the global logit nuisance, as the inversion
        state = sim.at(obs.time)
        lvl = state.level
        eps = 0.0
        if obs.logit_nuisance is not None and w > 0.0:
            e = obs.logit_nuisance.eps_at(lvl)
            eps = e if e is not None else 0.0
        mask_L = om._restrict(mask, lvl)
        if obs.two_sided:
            omega, target = om._restrict(prob.domain.domain_mask.to(torch.float32), lvl), mask_L
        else:
            omega, target = mask_L, torch.ones_like(mask_L)
        p = obs._p_g(state.H, state.smb_coarse, state.active, eps)[0]
        l = scale * obs._c_data(lvl, prob.dx, w) * omega * (target - p) ** 2
        return to_fine(l, lvl), J
    raise TypeError(f"no per-cell split for {type(obs).__name__}")


def integrated_dhdt(run_dir, dh, zone, nb, rho_i=917.0, rho_w=1028.0):
    """Per basin the VOLUME-INTEGRATED elevation-change rate over `zone`
    (bool, fine grid) from the run's H at the product's epochs against the
    product's, both in Gt/yr, as a mass rate: on the observed floating cells
    the product's surface rate is converted to a thickness rate by
    rho_w / (rho_w - rho_i). The quantity a displaced or sharper-than-the-
    product response still gets right, unlike the per-cell whitened term."""
    H0 = xr.open_dataset(run_dir / f"state_{dh['t0']:g}.nc").H.values.astype('float64')
    H1 = xr.open_dataset(run_dir / f"state_{dh['t1']:g}.nc").H.values.astype('float64')
    rate_m = (H1 - H0) / (dh['t1'] - dh['t0'])
    obs = np.where(dh['floating'], dh['obs'] * rho_w / (rho_w - rho_i), dh['obs'])
    ok = zone & np.isfinite(obs)
    kgt = dh['dx'] ** 2 * rho_i / 1e12
    b = dh['basin'][ok]
    m = np.bincount(b, weights=rate_m[ok], minlength=nb + 1)[:nb + 1] * kgt
    o = np.bincount(b, weights=obs[ok], minlength=nb + 1)[:nb + 1] * kgt
    return m, o


def evaluate_run(run_dir, prob, cfg, bed_fine, basin_idx, nb, names, region):
    epochs = sorted({t for o in prob.observations for t in o.required_times})
    states = {}
    for t in epochs:
        p = run_dir / f"state_{t:g}.nc"
        if not p.exists():
            raise FileNotFoundError(f"{run_dir}: no state at t = {t:g} (sweep with --state-times covering {epochs})")
        states[t] = load_state(p, bed_fine, cfg)
    level = states[epochs[-1]].level
    sim = SimResult(states=states, final=states[epochs[-1]])
    mask = (prob.domain.rgi_mask * prob.domain.domain_mask).to(torch.float32)
    cfg_it = cfg.at_iteration(0, level, schedule=False)
    table = pd.DataFrame(dict(basin=np.arange(nb + 1), name=list(names) + ["<unassigned>"],
                              region=list(region) + [""]))
    totals = {}
    with torch.no_grad():
        for obs in prob.observations:
            l, J = per_cell_losses(obs, sim, cfg_it, prob, mask, level)
            per = torch.bincount(basin_idx.ravel(), weights=l.ravel().to(torch.float64), minlength=nb + 1).cpu().numpy()
            table[f"J_{obs.name}"] = per
            totals[obs.name] = dict(loss=J, split_sum=float(per.sum()))
    jcols = [c for c in table.columns if c.startswith("J_")]
    table["J_total"] = table[jcols].sum(axis=1)
    return table, totals, level


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain-path", default=str(HERE.parent / "domains" / "greenland"))
    ap.add_argument("--sweep-root", required=True)
    ap.add_argument("--terms", nargs="+", default=["srf", "vel", "extent", "dhdt_measures"], choices=list(TERM_SPECS))
    ap.add_argument("--runs", nargs="*", default=None, help="run dir names to evaluate (default: all in sweep.json)")
    ap.add_argument("--assemble", action="store_true", help="pick c per basin and write the h0_base field")
    ap.add_argument("--basin-var", default="calving_basin", choices=["calving_basin", "calving_reach"],
                    help="cells summed per glacier: the whole drainage basin (+ its fjord), or only the calving reach "
                         "(the fjord + the trough under the ice within max_km of the terminus)")
    ap.add_argument("--select-only", action="store_true", help="skip the evaluation: select from the sweep's existing sweep_eval.csv")
    ap.add_argument("--select-terms", nargs="+", default=None,
                    help="terms summed for the selection (default: all evaluated); e.g. 'extent' alone as a diagnostic")
    ap.add_argument("--plateau-rel", type=float, default=0.1, help="c values with J <= Jmin (1 + rel) + abs form the best plateau; its centre is taken")
    ap.add_argument("--plateau-abs", type=float, default=0.0, help="absolute part of the plateau tolerance (J units)")
    ap.add_argument("--min-range", type=float, default=0.02, help="a basin whose J range over the good runs is below this fraction of its Jmin is unresponsive")
    ap.add_argument("--dv-zone", default="reach", choices=["reach", "low", "basin"],
                    help="cells of the integrated dh/dt term (J_dhdt_int): the calving reach (front_dist >= --dv-reach-km), "
                         "the basin's ice below --dv-elev, or the whole basin")
    ap.add_argument("--dv-reach-km", type=float, default=30.0)
    ap.add_argument("--dv-elev", type=float, default=1500.0)
    ap.add_argument("--dv-rel", type=float, default=0.25, help="relative systematic error of the integrated observed rate")
    ap.add_argument("--dv-floor", type=float, default=0.05, help="absolute floor of that error, Gt/yr")
    ap.add_argument("--min-flux", type=float, default=0.0,
                    help="basins whose observed terminus flux (H |u| summed over terminus cells, calving_basins.csv) is below this "
                         "take their region's median c instead of their own plateau (small fronts fit noise); e.g. 1e5")
    ap.add_argument("--bad-run-factor", type=float, default=5.0,
                    help="a run whose domain total of any term exceeds this multiple of the median over runs, or is non-finite, is excluded")
    ap.add_argument("--out", default=None, help="h0_base file name under model_inputs/ (default calving_h0_base.nc)")
    ap.add_argument("--gates", default=str(HERE.parent / "common_data/dhdt/mankoff/dataverse_files/gates.gpkg"))
    ap.add_argument("--gate-year", type=float, default=2018.0, help="epoch of J_gate (a state file must exist)")
    ap.add_argument("--gate-sigma", type=float, default=0.15, help="std of log(D_model / D_obs) in J_gate")
    ap.add_argument("--gate-eps", type=float, default=0.1, help="Gt/yr added to both fluxes inside the log")
    ap.add_argument("--gate-speed-sigma", type=float, default=0.15, help="J_gatef: std of log gate speed ratio")
    ap.add_argument("--gate-thick-sigma", type=float, default=0.2, help="J_gatef: std of log gate thickness ratio")
    ap.add_argument("--admissible-term", default=None,
                    help="per basin, only c values with J_<term> <= Jmin (1 + --admissible-rel) are candidates (e.g. srf)")
    ap.add_argument("--admissible-rel", type=float, default=9.0)
    ap.add_argument("--reference-run", default=None,
                    help="stage-1 reference run dir (forward_standalone.py --stage1): the integrated terms compare with it, not the products")
    ap.add_argument("--srf-year", type=float, default=2008.0, help="epoch of J_srf_ref")
    ap.add_argument("--gate-epochs", type=float, nargs="+", default=[1990.0, 2008.0, 2015.0, 2018.0],
                    help="epochs of J_gatefm (reference run only; state files must exist in both runs)")
    ap.add_argument("--srf-ref-sigma", type=float, default=10.0, help="m, J_srf_ref")
    a = ap.parse_args()

    domain = Path(a.domain_path)
    cfg = load_config(domain)
    root = Path(a.sweep_root)
    sweep = json.loads((root / "sweep.json").read_text())
    # the union of the manifest and the finished run directories on disk (a
    # partial rerun used to overwrite the manifest with its own runs only)
    cs = {Path(r["dir"]).name: float(r["c"]) for r in sweep["runs"]}
    for d in sorted(root.glob("c[+-]*")):
        if (d / "forward_soln.nc").exists() and d.name not in cs:
            cs[d.name] = float(d.name[1:])
    runs = [root / n for n in sorted(cs, key=lambda n: cs[n]) if (root / n / "forward_soln.nc").exists()]
    if a.runs:
        runs = [r for r in runs if r.name in a.runs]
    print(f"sweep {root}: {len(runs)} runs, alpha_h {sweep['alpha_h']:g}, alpha_q {sweep['alpha_q']:g}, level {sweep['level']}; "
          f"J summed per glacier over '{a.basin_var}'", flush=True)

    binfo = pd.read_csv(domain / "model_inputs" / "calving_basins.csv").set_index("basin")
    bas_full = xr.open_dataset(domain / "model_inputs" / "calving_basins.nc", mask_and_scale=False)
    nb = int(bas_full.attrs["n_basins"])
    names = np.array([binfo.name.get(i, f"basin{i}") for i in range(nb)], dtype=object)
    region = np.array([binfo.region.get(i, "") for i in range(nb)], dtype=object)

    # the integrated dh/dt term needs only the state files and the product (CPU)
    gi = crop_to_factor(xr.open_dataset(domain / "model_inputs" / "GLIDE_inputs.nc"), 2 ** cfg.n_levels)
    dprod = crop_to_factor(xr.open_dataset(domain / "model_inputs" / MEASURES_SPEC.filename), 2 ** cfg.n_levels)
    basc = crop_to_factor(bas_full, 2 ** cfg.n_levels)
    bidx = basc.calving_basin.values.astype(np.int64); bidx = np.where(bidx >= 0, bidx, nb)
    ice_c = gi.rgi_mask.values > 0.5
    if a.dv_zone == "reach":
        zone = ice_c & (basc.front_dist_km.values >= -a.dv_reach_km) & (basc.front_dist_km.values <= 0)
    elif a.dv_zone == "low":
        zone = ice_c & (gi.elevation.values < a.dv_elev)
    else:
        zone = ice_c
    dh = dict(t0=float(dprod.dhdt.attrs["time_start"]), t1=float(dprod.dhdt.attrs["time_end"]), obs=dprod.dhdt.values.astype('float64'),
              floating=(gi.floating_mask.values > 0.5), basin=bidx, dx=float(abs(gi.x[1] - gi.x[0])))
    print(f"integrated dh/dt term over the '{a.dv_zone}' zone ({int(zone.sum())} ice cells), {dh['t0']:g}-{dh['t1']:g}, "
          f"sigma = max({a.dv_floor:g} Gt/yr, {a.dv_rel:g} x |observed|)")

    # the gate term: the observations' own flux through each basin's gates,
    # or with --reference-run the stage-1 run's
    from gates import ChainGates
    G = ChainGates(a.gates, gi.x.values, gi.y.values, gi.vx.values, gi.vy.values)
    gate_ids = [[] if pd.isna(binfo.gates.get(i, np.nan)) else [int(float(g)) for g in str(binfo.gates.get(i)).split()]
                for i in range(nb)]
    ny0 = gi.sizes["y"]

    def fields(run, t):
        """Cell-centred U, V, H, S of a run's raw state at t on the fine grid."""
        S = xr.open_dataset(Path(run) / f"state_{t:g}.nc")
        u, v, H, srf = S.u_s.values, S.v_s.values, S.H.values, S.srf.values
        S.close()
        U, V = 0.5 * (u[:, :-1] + u[:, 1:]), 0.5 * (v[:-1] + v[1:])
        f = ny0 // H.shape[0]
        if f > 1:                                   # a coarse-level run: repeat onto the fine grid
            U, V, H, srf = (np.kron(q, np.ones((f, f))) for q in (U, V, H, srf))
        return U, V, H, srf

    per_basin = lambda df, col: np.array([float(df[col].reindex(ids).fillna(0.0).sum()) if ids else np.nan
                                          for ids in gate_ids] + [np.nan])
    if a.reference_run:
        ref = Path(a.reference_run)
        Ur, Vr, Hr, _ = fields(ref, a.gate_year)
        ref_desc = f"the stage-1 reference {ref}"
    else:
        Ur, Vr, Hr = (np.nan_to_num(gi[k].values) for k in ("vx", "vy", "thickness_obs"))
        ref_desc = "ITS_LIVE x thickness_obs"
    ref_g = G.flux(Ur, Vr, Hr).set_index("gate")
    D_obs = per_basin(ref_g, "D")
    print(f"gate terms at {a.gate_year:g}: {sum(bool(i) for i in gate_ids)} basins with gates, reference flux "
          f"{np.nansum(D_obs):.0f} Gt/yr ({ref_desc}, chain); sigma(log) flux {a.gate_sigma:g}, "
          f"speed {a.gate_speed_sigma:g}, thickness {a.gate_thick_sigma:g}")
    if a.reference_run:
        dV_ref, _ = integrated_dhdt(ref, dh, zone, nb)
        _, _, H_s, S_s = fields(ref, a.srf_year)
        ref_srf = (S_s, H_s)
        print(f"integrated dh/dt and surface ({a.srf_year:g}, sigma {a.srf_ref_sigma:g} m) against the reference too")

    ref_ep = {t: fields(ref, t)[:3] for t in a.gate_epochs} if a.reference_run else {}
    if ref_ep:
        print("multi-epoch gate term against the reference at " + ", ".join(
            f"{t:g} ({np.nansum(per_basin(G.flux(*ref_ep[t]).set_index('gate'), 'D')):.0f} Gt/yr)" for t in a.gate_epochs))

    def factor_z2(U, V, H, R):
        fa = G.factors(U, V, H, *R)
        rs = per_basin(fa, "s_m") / per_basin(fa, "s_o")
        rh = per_basin(fa, "h_m") / per_basin(fa, "h_o")
        z2 = (np.log(np.maximum(rs, 0.01)) / a.gate_speed_sigma) ** 2 + (np.log(np.maximum(rh, 0.01)) / a.gate_thick_sigma) ** 2
        return rs, rh, z2

    def gate_flux(run):
        U, V, H, _ = fields(run, a.gate_year)
        mg = G.flux(U, V, H).set_index("gate")
        rs, rh, _ = factor_z2(U, V, H, (Ur, Vr, Hr))
        return per_basin(mg, "D"), rs, rh

    def add_integrated(table, run):
        m, o = integrated_dhdt(run, dh, zone, nb)
        if a.reference_run:
            o = dV_ref
        sig = np.maximum(a.dv_floor, a.dv_rel * np.abs(o))
        table["dV_model"] = m; table["dV_obs"] = o
        table["J_dhdt_int"] = cfg.loss_scale * 0.5 * ((m - o) / sig) ** 2 * 1e3      # chi2/2 per basin, in J units x 1e3 (one number, not a cell sum)
        Dm, rs, rh = gate_flux(run)
        table["D_gate_model"] = Dm; table["D_gate_obs"] = D_obs
        r = np.log((Dm + a.gate_eps) / (D_obs + a.gate_eps)) / a.gate_sigma
        table["J_gate"] = np.where(np.isfinite(r), cfg.loss_scale * 0.5 * r ** 2 * 1e3, 0.0)   # same units as J_dhdt_int
        table["gate_speed_ratio"] = rs; table["gate_thick_ratio"] = rh
        zs = np.log(np.maximum(rs, 0.01)) / a.gate_speed_sigma
        zh = np.log(np.maximum(rh, 0.01)) / a.gate_thick_sigma
        jf = cfg.loss_scale * 0.5 * (zs ** 2 + zh ** 2) * 1e3
        table["J_gatef"] = np.where(np.isfinite(jf), jf, 0.0)
        if ref_ep:
            tot = np.zeros(nb + 1)
            for t, R in ref_ep.items():
                U, V, H, _ = fields(run, t)
                _, _, z2 = factor_z2(U, V, H, R)
                tot += np.where(np.isfinite(z2), z2, 0.0)
            table["J_gatefm"] = cfg.loss_scale * 0.5 * tot * 1e3
        if a.reference_run:
            _, _, H, S = fields(run, a.srf_year)
            S_r, H_r = ref_srf
            cells = (H > 10.0) | (H_r > 10.0)
            z2 = np.where(cells, ((S - S_r) / a.srf_ref_sigma) ** 2, 0.0)
            table["J_srf_ref"] = cfg.loss_scale * 0.5 * np.bincount(bidx.ravel(), weights=z2.ravel(), minlength=nb + 1)[:nb + 1]
        return table

    if a.select_only:
        long = pd.read_csv(root / "sweep_eval.csv")
        print(f"selection from the existing {root / 'sweep_eval.csv'} ({long.run.nunique()} runs)")
        parts = []
        for run_name, t in long.groupby("run"):
            parts.append(add_integrated(t.copy(), root / run_name))
        # finished runs not yet in the csv (a refinement sweep): the basin-integrated
        # terms alone (per-cell terms need the GPU evaluation; select on gate / dhdt_int)
        new = [r for r in runs if r.name not in set(long.run)]
        for run in new:
            t = pd.DataFrame(dict(basin=np.arange(nb + 1), name=list(names) + ["<unassigned>"], region=list(region) + [""]))
            t = add_integrated(t, run); t["c"] = cs[run.name]; t["run"] = run.name
            parts.append(t)
        if new:
            print(f"added {len(new)} runs not in the csv with the integrated terms only: {[r.name for r in new]}")
        long = pd.concat(parts, ignore_index=True)
        long.to_csv(root / "sweep_eval_integrated.csv", index=False)
    else:
        prob = build_problem(cfg, a.terms)
        nyc, nxc = prob.ny, prob.nx
        bas = crop_to_factor(bas_full, 2 ** cfg.n_levels)
        basin = bas[a.basin_var].values.astype(np.int64)
        assert basin.shape == (nyc, nxc), (basin.shape, (nyc, nxc))
        basin_idx = torch.as_tensor(np.where(basin >= 0, basin, nb), device=DEV)
        phys = xr.open_dataset(Path(sweep["physical_fields"]))
        bed_fine = torch.as_tensor(phys.bed.values.astype(np.float32), device=DEV)
        long = []
        for run in runs:
            table, totals, level = evaluate_run(run, prob, cfg, bed_fine, basin_idx, nb, names, region)
            table.to_csv(run / "J_basins.csv", index=False)
            c = cs[run.name]
            print(f"--- {run.name} (c = {c:+g}, level {level}): " + ", ".join(
                f"{k} {v['loss']:.4f}" + (f" (basin split {v['split_sum']:.4f}; the rest is the logit nuisance's prior, global)"
                                           if k == 'extent' else f" (split {v['split_sum']:.4f})")
                for k, v in totals.items()), flush=True)
            t = add_integrated(table.copy(), run); t["c"] = c; t["run"] = run.name
            long.append(t)
        long = pd.concat(long, ignore_index=True)
        long.to_csv(root / "sweep_eval.csv", index=False)
        print(f"wrote {root / 'sweep_eval.csv'}")

    if not a.assemble:
        return
    # ------------------------------------------------------------ good runs, the selection objective
    jcols = [c for c in long.columns if c.startswith("J_") and c not in ("J_total", "J_dhdt_int", "J_gate", "J_gatef", "J_gatefm", "J_srf_ref")]
    tot = long.groupby("c")[jcols].sum(min_count=1)
    med = tot.median()
    selc = [c for c in jcols if a.select_terms is None or c[2:] in a.select_terms]
    tsel = tot[selc] if selc else tot.iloc[:, :0]
    bad = tot.index[(~np.isfinite(tsel)).any(axis=1) | (tsel > a.bad_run_factor * med[selc]).any(axis=1)] if selc else tot.index[:0]
    if len(bad):
        print("EXCLUDED runs (non-finite or collapsed: a domain total more than "
              f"{a.bad_run_factor:g}x the median over runs): " + ", ".join(f"c {c:+g}" for c in bad))
        print(tot.loc[bad].to_string())
    good = long[~long.c.isin(bad)].copy()
    sel_terms = [f"J_{t}" for t in a.select_terms] if a.select_terms else jcols
    missing = [t for t in sel_terms if t not in good.columns]
    if missing:
        raise SystemExit(f"--select-terms not in the evaluation: {missing}; evaluated: {jcols}")
    good["J_sel"] = good[sel_terms].sum(axis=1)
    print(f"selection objective: {' + '.join(sel_terms)}; plateau: J <= Jmin (1 + {a.plateau_rel:g}) + {a.plateau_abs:g}")
    piv = good.pivot_table(index="basin", columns="c", values="J_sel", dropna=False)
    adm = None
    if a.admissible_term:
        acol = f"J_{a.admissible_term}"
        if acol not in good.columns:
            raise SystemExit(f"--admissible-term {a.admissible_term}: {acol} not evaluated")
        adm = good.pivot_table(index="basin", columns="c", values=acol, dropna=False)
        print(f"admissible c per basin: {acol} <= Jmin (1 + {a.admissible_rel:g})")
    # basins without gates have J_gate(f) = 0 everywhere; with an admissibility
    # term they fall back to it (their plateau is then the admissible set's best)
    gate_sel = any(t in sel_terms for t in ("J_gate", "J_gatef", "J_gatefm"))
    has_gate = np.array([bool(ids) for ids in gate_ids])
    cgrid = np.array(sorted(piv.columns))
    calves = binfo.reindex(range(nb)).calves.fillna(False).astype(bool).values
    flux = binfo.reindex(range(nb)).terminus_flux_m2yr.fillna(0.0).values
    rows = []
    for b in range(nb):
        if b not in piv.index:
            continue
        J = piv.loc[b, cgrid].values.astype(float)
        if adm is not None and b in adm.index:
            A = adm.loc[b, cgrid].values.astype(float)
            if np.isfinite(A).any():
                okA = np.isfinite(A) & (A <= np.nanmin(A) * (1.0 + a.admissible_rel))
                if gate_sel and not has_gate[b]:
                    J = A.copy()                           # no gates: the admissibility term itself
                J = np.where(okA, J, np.inf)
        if not np.isfinite(J).any() or not calves[b]:
            rows.append(dict(basin=b, name=names[b], region=region[b], calves=bool(calves[b]), Jmin=np.nan, Jrange=np.nan, c=np.nan, source="none"))
            continue
        if flux[b] < a.min_flux:
            rows.append(dict(basin=b, name=names[b], region=region[b], calves=True, Jmin=J.min(), Jrange=J.max() - J.min(), c=np.nan, source="small"))
            continue
        fin = np.isfinite(J)
        jmin, jmax = J[fin].min(), J[fin].max()
        rng = jmax - jmin
        # a single admissible value is a decision, not an unresponsive basin
        if fin.sum() > 1 and rng < a.min_range * max(abs(jmin), 1e-12):
            rows.append(dict(basin=b, name=names[b], region=region[b], calves=True, Jmin=jmin, Jrange=rng, c=np.nan, source="unresponsive"))
            continue
        ok = np.flatnonzero(fin & (J <= jmin * (1.0 + a.plateau_rel) + a.plateau_abs))
        # the centre of the plateau: the member nearest its mean c, ties to the argmin
        cm = cgrid[ok].mean(); d = np.abs(cgrid[ok] - cm)
        cand = ok[d <= d.min() + 1e-9]
        c_sel = float(cgrid[cand[np.argmin(J[cand])]])
        rows.append(dict(basin=b, name=names[b], region=region[b], calves=True, Jmin=jmin, Jrange=rng, c=c_sel, source="fit",
                         plateau=" ".join(f"{cgrid[i]:+g}" for i in ok), c_argmin=float(cgrid[int(np.argmin(J))]),
                         J_at_c=float(J[list(cgrid).index(c_sel)])))
    sel = pd.DataFrame(rows).set_index("basin")
    fit = sel[sel.source == "fit"]
    regional = fit.groupby("region").c.median().to_dict()
    global_c = float(fit.c.median()) if len(fit) else 0.0
    for b, r in sel.iterrows():
        if r.source != "fit":
            sel.loc[b, "c"] = regional.get(r.region, global_c)
            sel.loc[b, "source"] = r.source + ":regional" if r.region in regional else r.source + ":global"
    tag = Path(a.out).stem.replace("calving_h0_base", "") if a.out else ""
    csv_path = root / f"calving_c_basins{tag}.csv"
    sel.to_csv(csv_path)
    print(f"\nassembled from runs c = {[float(c) for c in cgrid]}: {len(fit)} basins fitted"
          + (f" (terminus flux >= {a.min_flux:g}; {int((sel.source.str.startswith('small')).sum())} smaller fronts take the regional median)" if a.min_flux else "")
          + ", regional medians " + ", ".join(f"{k} {v:+.0f}" for k, v in sorted(regional.items())) + f", global {global_c:+.0f} m")
    show = fit.sort_values("Jrange", ascending=False).head(30)
    print("most responsive basins (J range over the good runs), their argmin, plateau and chosen c:")
    for b, r in show.iterrows():
        print(f"  {str(r['name'])[:26]:26s} {r.region:9s} Jmin {r.Jmin:9.3f} range {r.Jrange:9.3f}  argmin {r.c_argmin:+5.0f}  plateau [{r.plateau}]  -> c {r.c:+5.0f}")
    key = [n for n in ('NIOGHALVFJERDSFJORDEN', 'ZACHARIAE_ISSTROM', 'PETERMANN_GLETSCHER', 'RYDER_GLETSCHER', 'HUMBOLDT_GLETSCHER', 'JAKOBSHAVN_ISBRAE',
                       'HELHEIMGLETSCHER', 'KANGERLUSSUAQ', 'STORE_GLETSCHER', 'RINK_ISBRAE', 'UPERNAVIK_ISSTROM_N', 'KOGE_BUGT_C', 'DAUGAARD-JENSEN')
           if n in set(sel.name)]
    print("key fronts (dV = integrated dh/dt over the zone at the chosen c, Gt/yr, model vs observed):")
    for n in key:
        r = sel[sel.name == n].iloc[0]
        row = good[(good.name == n) & (good.c == r.c)]
        dv = (f"  dV {row.dV_model.iloc[0]:+.2f} vs {row.dV_obs.iloc[0]:+.2f}"
              + (f"  gate {row.D_gate_model.iloc[0]:.1f} vs {row.D_gate_obs.iloc[0]:.1f} Gt/yr"
                 f" (speed {row.gate_speed_ratio.iloc[0]:.2f}, thickness {row.gate_thick_ratio.iloc[0]:.2f})"
                 if np.isfinite(row.D_gate_obs.iloc[0]) else "")) if len(row) else ""
        print(f"  {n[:26]:26s} c {r.c:+5.0f} ({r.source})" + (f"  Jmin {r.Jmin:.3f} at {r.c_argmin:+g}, plateau [{r.plateau}]" if r.source == 'fit' else '') + dv)

    # ------------------------------------------------------------ the field on the full grid
    full = bas_full
    bfull = full.calving_basin.values.astype(np.int64)
    cvec = np.full(nb, global_c, np.float32)
    for b, r in sel.iterrows():
        cvec[b] = r.c
    # the sweep ran h0 = c alone; the composite adds calving_h0, so subtract it here
    h00 = float(cfg.calving_h0)
    h0_base = (np.where(bfull >= 0, cvec[np.clip(bfull, 0, nb - 1)], global_c) - h00).astype(np.float32)
    out = domain / "model_inputs" / (a.out or "calving_h0_base.nc")
    ds = xr.Dataset(dict(h0_base=(("y", "x"), h0_base), calving_basin=(("y", "x"), bfull.astype(np.int32))),
                    coords=dict(y=full.y.values, x=full.x.values))
    ds.h0_base.attrs.update(units="m", long_name="per-basin calving margin baseline c_i",
                            description=f"h0 = calving_h0 + h0_base + alpha_h dTF (ocean.py); h0_base = c_i - calving_h0 "
                                        f"({h00:g} m at assembly), so the composite's margin is the sweep's c_i; "
                                        f"from sweep_calving_eval.py --assemble")
    ds.attrs.update(source=f"sweep_calving_eval.py {date.today().isoformat()}", sweep_root=str(root),
                    alpha_h=sweep["alpha_h"], alpha_q=sweep["alpha_q"], c_grid=json.dumps([float(c) for c in cgrid]),
                    terms=" ".join(a.terms), select_terms=" ".join(sel_terms), plateau_rel=a.plateau_rel, plateau_abs=a.plateau_abs,
                    min_range=a.min_range, basin_var=a.basin_var, excluded_runs=json.dumps([float(c) for c in bad]), min_flux=a.min_flux,
                    regional_default=json.dumps(regional), global_default=global_c, calving_h0_subtracted=h00,
                    gate_year=a.gate_year, gate_sigma=a.gate_sigma, gate_speed_sigma=a.gate_speed_sigma,
                    gate_thick_sigma=a.gate_thick_sigma, admissible_term=str(a.admissible_term), admissible_rel=a.admissible_rel,
                    reference_run=str(a.reference_run))
    if "spatial_ref" in full:
        ds["spatial_ref"] = full["spatial_ref"]
        ds.h0_base.attrs["grid_mapping"] = "spatial_ref"
    ds.to_netcdf(out, encoding={"h0_base": dict(zlib=True, complevel=4), "calving_basin": dict(zlib=True, complevel=4)})
    print(f"wrote {out} (h0_base over the ice 10/50/90 pct "
          f"{np.percentile(h0_base[full.calving_basin.values >= 0], [10, 50, 90]).round(0)} m) and {csv_path}")


if __name__ == "__main__":
    main()
