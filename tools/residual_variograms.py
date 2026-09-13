"""Empirical variograms of a MAP solution's residual fields (surface, velocity, dh/dt).

    python tools/residual_variograms.py domains/delta [--results-subdir NAME] [--level N]
                                        [--use-saved-fields] [--no-plots] [--out DIR]

Re-runs the forward model from the saved MAP (`level_<N>/torch_vars.p`), forms every
field likelihood's residuals through `Observation.residuals()` — the σ-normalized raw
residual `r` and the whitened residual `z` (identical when the term has no MaternNoise
model) — and, for each of them, computes

  * the isotropic semivariogram  γ(h) = ½ E[(r(x) − r(x+h))²],
  * flow-aligned / cross-flow semivariograms (pairs binned by the angle between the lag
    vector and the local model surface-flow direction; DEM downslope where |U| < 1 m/yr),
  * the same for the within-glacier residual (per-RGI-glacier mean removed),
  * a per-glacier ANOVA (between- vs within-glacier variance), an exponential-model fit
    γ = c0 + c1(1 − e^{−h/a}) and the implied effective observation count A/(2πa²).

The point of the `z` statistics is the post-hoc check of a MaternNoise error model: if
(σ, l, ν) describe the residuals, the whitened residual is white with unit variance —
`std(z) → 1` and `γ_z(dx)/var(z) → 1` (no spatial structure left). The `r` statistics
are what you fit the error model *from* (Matérn l ≈ 2.3 × the exponential range a).

For the velocity product the per-component residuals are supplemented by the speed
residual and a per-glacier least-squares under-read factor η (observed = η·model, the
quantity the surge marginal integrates over).

Outputs (in `--out`, default `<output_dir>/level_<N>`): `residual_variograms.npz`,
`residual_maps.png`, `variograms.png`, plus the printed summary.

`--use-saved-fields` takes bed / β from `inverse_soln.nc` instead of mapping the saved
whitened parameters — for runs whose domain config (prior hyperparameters) has changed
since; the reproduction check (final H vs saved) tells you whether either path is
faithful.
"""
import argparse
import dataclasses
import os
import sys
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glacier_inverse import load_config                       # noqa: E402
from glacier_inverse.io import load_whitened_params_into      # noqa: E402
from glacier_inverse.problem import GlacierProblem            # noqa: E402

LAGS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256]
NDIR = 16
COS_ALONG, COS_ACROSS = np.cos(np.deg2rad(22.5)), np.sin(np.deg2rad(22.5))
REPORT_LAGS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
ANISO_LAGS = (4, 8, 16, 32, 64, 128)


def lag_offsets():
    offsets = {}
    for l in LAGS:
        for k in range(NDIR):
            th = np.pi * k / NDIR
            di, dj = int(round(l * np.sin(th))), int(round(l * np.cos(th)))
            if (di, dj) == (0, 0) or (di, dj) in offsets:
                continue
            offsets[(di, dj)] = l
    return offsets


class Variograms:
    """Variogram machinery bound to one grid (flow direction + glacier labels)."""

    def __init__(self, fr, fc, label, dx):
        self.fr, self.fc, self.label, self.dx = fr, fc, label, dx
        self.ny, self.nx = fr.shape
        self.n_lab = int(label.max().item()) + 2
        self.offsets = lag_offsets()

    def shifted(self, a, di, dj):
        """(a[x], a[x+h]) views over the overlap."""
        ny, nx = self.ny, self.nx
        r0, r1 = max(0, -di), min(ny, ny - di)
        c0, c1 = max(0, -dj), min(nx, nx - dj)
        return a[r0:r1, c0:c1], a[r0 + di:r1 + di, c0 + dj:c1 + dj]

    def variogram(self, r, valid):
        acc = {}   # lag -> [iso_sum, iso_n, along_sum, along_n, across_sum, across_n]
        r = torch.where(valid, r, torch.zeros_like(r))
        for (di, dj), l in self.offsets.items():
            a, b = self.shifted(r, di, dj)
            va, vb = self.shifted(valid, di, dj)
            v = va & vb
            d2 = (a - b) ** 2
            h = np.hypot(di, dj)
            fra = self.shifted(self.fr, di, dj)[0]
            fca = self.shifted(self.fc, di, dj)[0]
            cosang = torch.abs(fra * di + fca * dj) / h
            al = v & (cosang >= COS_ALONG)
            ac = v & (cosang <= COS_ACROSS)
            e = acc.setdefault(l, np.zeros(6))
            e += np.array([(d2 * v).sum().item(), v.sum().item(),
                           (d2 * al).sum().item(), al.sum().item(),
                           (d2 * ac).sum().item(), ac.sum().item()])
        lags = np.array(sorted(acc))
        E = np.array([acc[l] for l in lags])
        with np.errstate(invalid="ignore", divide="ignore"):
            g = 0.5 * E[:, 0] / E[:, 1]
            ga = 0.5 * E[:, 2] / E[:, 3]
            gc = 0.5 * E[:, 4] / E[:, 5]
        return lags * self.dx, g, ga, gc, E[:, 1]

    def glacier_anova(self, r, valid):
        lab = self.label.clone()
        lab[~valid] = -1
        idx = (lab + 1).ravel().long()
        n = torch.zeros(self.n_lab, device="cuda").index_add_(0, idx, valid.ravel().float())
        s = torch.zeros(self.n_lab, device="cuda").index_add_(0, idx, (r * valid).ravel())
        mean = s / n.clamp(min=1)
        within = r - mean[idx].reshape(r.shape)
        keep = n >= 100
        keep[0] = False
        tot = (r[valid] ** 2).mean().item()
        between = ((mean[keep] ** 2 * n[keep]).sum() / n[keep].sum()).item() \
            if keep.any() else 0.0
        return within, dict(n_glaciers=int(keep.sum().item()), mean_sq=tot,
                            between=between, within=(within[valid] ** 2).mean().item(),
                            grand_mean=r[valid].mean().item())


def fit_exp(h, g, n):
    from scipy.optimize import curve_fit
    ok = np.isfinite(g) & (n > 0)

    def f(h, c0, c1, a):
        return c0 + c1 * (1 - np.exp(-h / a))
    try:
        p, _ = curve_fit(f, h[ok], g[ok], p0=[g[ok][0] / 2, g[ok][-1], 2000.0],
                         sigma=1 / np.sqrt(n[ok]),
                         bounds=([0, 0, 50], [np.inf, np.inf, 1e6]))
    except Exception:
        p = [np.nan] * 3
    return tuple(float(x) for x in p)


def analyse(vg, name, r, valid, dx, *, whitened):
    """Statistics + variograms of one residual field; prints a summary block."""
    r = r.detach().float()
    valid = valid & torch.isfinite(r)
    within, an = vg.glacier_anova(r, valid)
    rv = r[valid].cpu().numpy()
    N = rv.size
    A = N * dx * dx
    mad = 1.4826 * np.median(np.abs(rv - np.median(rv)))
    h, g, ga, gc, npair = vg.variogram(r, valid)
    hw, gw, gaw, gcw, _ = vg.variogram(within, valid)
    c0, c1, a = fit_exp(h, g, npair)
    c0w, c1w, aw = fit_exp(hw, gw, npair)
    var = rv.var()

    def range_at(frac, gg):
        hit = gg >= frac * var
        return h[np.argmax(hit)] if hit.any() else np.inf

    nugget = g[0] / var
    aniso = {int(round(hh / dx)): aa / cc for hh, aa, cc in zip(h, ga, gc)
             if np.isfinite(aa) and np.isfinite(cc)}
    out = dict(h=h, g=g, ga=ga, gc=gc, npair=npair, gw=gw, gaw=gaw, gcw=gcw,
               fit=(c0, c1, a), fitw=(c0w, c1w, aw), N=N, var=var, std=rv.std(),
               robust_std=mad, nugget=nugget, matern_l=2.3 * a,
               lag_half=range_at(0.5, g), lag_95=range_at(0.95, g),
               n_eff=A / (2 * np.pi * a ** 2) if a > 0 else np.nan, **an)

    tag = "z (whitened)" if whitened else "r (sigma-normalized)"
    print(f"== {name} / {tag}: N={N} px ({A / 1e6:.0f} km2), "
          f"{an['n_glaciers']} glaciers >= 100 px")
    print(f"   mean {rv.mean():+.3f}  std {rv.std():.3f}  robust std {mad:.3f}  "
          f"frac|.|>1 {(np.abs(rv) > 1).mean():.3f}  frac|.|>3 {(np.abs(rv) > 3).mean():.3f}")
    print(f"   ANOVA: E[.^2] {an['mean_sq']:.3f} = between-glacier {an['between']:.3f} "
          f"+ within {an['within']:.3f}  (between frac "
          f"{an['between'] / an['mean_sq'] if an['mean_sq'] else np.nan:.2f})")
    print("   variogram / var: " + " ".join(
        f"{hh / 1000:.2g}km:{gg / var:.2f}" for hh, gg in zip(h, g)
        if int(round(hh / dx)) in REPORT_LAGS))
    print(f"   nugget gamma(dx)/var {nugget:.2f}; lag at 50% var {out['lag_half'] / 1000:.2f} km, "
          f"95% {out['lag_95'] / 1000:.2f} km")
    print(f"   exp fit: nugget {c0:.3f} sill {c0 + c1:.3f} range a {a / 1000:.2f} km "
          f"-> Matern l ~ 2.3a = {2.3 * a / 1000:.2f} km, N_eff = A/(2 pi a^2) = {out['n_eff']:.0f} "
          f"(N_px = {N})")
    print(f"   within-glacier: var {an['within']:.3f}, exp range {aw / 1000:.2f} km")
    print("   anisotropy gamma_along/gamma_across: " + " ".join(
        f"{k * dx / 1000:.2g}km:{v:.2f}" for k, v in aniso.items() if k in ANISO_LAGS))
    if whitened:
        print(f"   >>> post-hoc: std(z) = {rv.std():.3f} (target 1), "
              f"gamma_z(dx)/var = {nugget:.2f} (target 1)")
    print()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("domain", help="domain directory, e.g. domains/delta")
    ap.add_argument("--results-subdir", default=None,
                    help="override config.results_subdir")
    ap.add_argument("--level", type=int, default=None,
                    help="multigrid level of the saved solution (default config.min_level)")
    ap.add_argument("--use-saved-fields", action="store_true",
                    help="take bed/beta from inverse_soln.nc instead of mapping torch_vars.p")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--out", default=None,
                    help="output directory (default <output_dir>/level_<level>)")
    args = ap.parse_args()

    config = load_config(args.domain)
    if args.results_subdir is not None:
        config = dataclasses.replace(config, results_subdir=args.results_subdir)
    level = config.min_level if args.level is None else args.level
    level_dir = f"{config.output_dir}/level_{level}"
    out_dir = args.out or level_dir
    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.basename(os.path.normpath(args.domain))

    problem = GlacierProblem(config)
    dx = float(problem.dx)
    print(f"[{tag}] results {config.output_dir}  level {level}  dx={dx}")
    params = problem.params
    load_whitened_params_into(params, f"{level_dir}/torch_vars.p", priors=problem.priors)
    problem.model.set_top_level(level)

    import xarray as xr
    H_s = bed_s = beta_s = None
    soln = f"{level_dir}/inverse_soln.nc"
    if os.path.exists(soln):
        ds = xr.open_dataset(soln)
        bed_s, beta_s, H_s = ds.bed.values, ds.beta.values, ds.H.values
        ds.close()

    with torch.no_grad():
        phys = problem.physical_from(params)
        if bed_s is not None:
            print(f"[{tag}] mapped-vs-saved bed rms "
                  f"{np.sqrt(((phys.bed.cpu().numpy() - bed_s) ** 2).mean()):.2f} m, "
                  f"log beta rms "
                  f"{np.sqrt(((phys.log_beta.cpu().numpy() - np.log(beta_s)) ** 2).mean()):.4f}")
        if args.use_saved_fields:
            if bed_s is None:
                raise SystemExit(f"--use-saved-fields needs {soln}")
            phys = dataclasses.replace(
                phys, bed=torch.tensor(bed_s, device="cuda"),
                log_beta=torch.tensor(np.log(beta_s), device="cuda"))
            print(f"[{tag}] using bed/beta from inverse_soln.nc")
        sim = problem.simulate_physical(level=level, physical=phys)
        if H_s is not None:
            H = sim.final.H.detach().cpu().numpy()
            print(f"[{tag}] reproduction check: final H vs saved rms "
                  f"{np.sqrt(((H - H_s) ** 2).mean()):.3f} m, max {np.abs(H - H_s).max():.2f} m")

        dom = problem.domain
        ice = (dom.rgi_mask > 0) & (dom.domain_mask > 0)
        label = dom.rgi_label.clone()
        mask = (dom.rgi_mask * dom.domain_mask).to(torch.float32)
        cfg0 = config.at_iteration(0, 0, schedule=False)
        # Coarse-space terms (surface) return residuals on the level's grid;
        # their masks/labels are the box-restricted ice fraction (> 1/2) and
        # the strided label sample. The fine-grid terms keep the fine masks.
        f = 2 ** level

        def coarse_mask(m):
            return (torch.nn.functional.avg_pool2d(
                m.to(torch.float32)[None, None], f)[0, 0] > 0.5) if f > 1 else m
        ice_L = coarse_mask(ice)
        label_L = label[::f, ::f] if f > 1 else label
        dx_L = dx * f

        # ------------------------------------------------ residual fields
        fields = {}   # name -> (field, valid, whitened?)
        srf = problem.get_observation("srf")
        S_model = sim.at(srf.time).S_fine
        res = srf.residuals(sim=sim, physical=phys, config=cfg0, domain=dom, mask=mask, dx=dx)
        fields["srf_r"] = (res["r"], ice_L, False)
        if srf.noise is not None:
            fields["srf_z"] = (res["z"], ice_L, True)

        vel = problem.get_observation("vel")
        u_mod, v_mod = vel._predicted(sim.at(vel.time))
        speed_mod = torch.sqrt(u_mod ** 2 + v_mod ** 2)
        speed_obs = torch.sqrt(vel.u_obs ** 2 + vel.v_obs ** 2)
        vvalid = ice & (vel.v_mask > 0)
        res = vel.residuals(sim=sim, physical=phys, config=cfg0, domain=dom, mask=mask, dx=dx)
        fields["vel_r_u"] = (res["r_u"], vvalid, False)
        fields["vel_r_v"] = (res["r_v"], vvalid, False)
        fields["vel_speed"] = ((speed_mod - speed_obs) / vel.sigma, vvalid, False)
        if vel.noise is not None:
            fields["vel_z_u"] = (res["z_u"], vvalid, True)
            fields["vel_z_v"] = (res["z_v"], vvalid, True)
        # per-glacier least-squares eta (observed = eta * model), clipped to (0, 1]
        n_lab = int(label.max().item()) + 2
        lab = label.clone()
        lab[~vvalid] = -1
        idx = (lab + 1).ravel().long()
        num = torch.zeros(n_lab, device="cuda").index_add_(
            0, idx, ((vel.u_obs * u_mod + vel.v_obs * v_mod) * vvalid).ravel())
        den = torch.zeros(n_lab, device="cuda").index_add_(
            0, idx, ((u_mod ** 2 + v_mod ** 2) * vvalid).ravel())
        eta = torch.clamp(num / torch.clamp(den, min=1e-6), 0.05, 1.0)
        eta[0] = 1.0
        eta_pix = eta[idx].reshape(u_mod.shape)
        fields["vel_speed_eta"] = ((eta_pix * speed_mod - speed_obs) / vel.sigma, vvalid, False)
        eta_np = eta.cpu().numpy()
        cnt_lab = torch.zeros(n_lab, device="cuda").index_add_(
            0, idx, vvalid.ravel().float()).cpu().numpy()

        dh = problem.get_observation("dhdt")
        if dh is not None:
            res = dh.residuals(sim=sim, physical=phys, config=cfg0, domain=dom, mask=mask, dx=dx)
            dvalid = ice & (dh.dhdt_mask > 0)
            fields["dhdt_r"] = (res["r"], dvalid, False)
            if dh.noise is not None:
                fields["dhdt_z"] = (res["z"], dvalid, True)
            dhdt_raw = dh.model_rate(sim, "fine") - dh.dhdt

        # flow direction in ARRAY coordinates (row, col); y decreases with row index
        fr, fc = -v_mod, u_mod
        slow = speed_mod < 1.0
        dSr = torch.zeros_like(S_model)
        dSc = torch.zeros_like(S_model)
        dSr[1:-1] = (S_model[2:] - S_model[:-2]) / 2
        dSc[:, 1:-1] = (S_model[:, 2:] - S_model[:, :-2]) / 2
        fr = torch.where(slow, -dSr, fr)
        fc = torch.where(slow, -dSc, fc)
        nrm = torch.sqrt(fr ** 2 + fc ** 2).clamp(min=1e-9)
        vg = Variograms(fr / nrm, fc / nrm, label, dx)
        vg_L = (Variograms((fr / nrm)[::f, ::f], (fc / nrm)[::f, ::f], label_L, dx_L)
                if f > 1 else vg)

        print()
        results = {}
        for name, (r, valid, whitened) in fields.items():
            coarse = r.shape != ice.shape
            results[name] = analyse(vg_L if coarse else vg, f"{tag}/{name}", r, valid,
                                    dx_L if coarse else dx, whitened=whitened)

        eta_ok = cnt_lab >= 100
        print(f"per-glacier eta (n>=100 px): n={eta_ok.sum()} "
              f"median {np.median(eta_np[eta_ok]):.2f}  "
              f"frac<0.8 {(eta_np[eta_ok] < 0.8).mean():.2f}  "
              f"frac<0.5 {(eta_np[eta_ok] < 0.5).mean():.2f}")

        np.savez(f"{out_dir}/residual_variograms.npz", dx=dx, eta=eta_np, eta_cnt=cnt_lab,
                 **{f"{k}__{kk}": np.asarray(vv) for k, v in results.items()
                    for kk, vv in v.items()})
        print(f"wrote {out_dir}/residual_variograms.npz")

        if args.no_plots:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        icen = ice.cpu().numpy()
        panels = [("srf_r", (S_model - srf.S_obs).cpu().numpy(), 60, "S_model - S_obs [m] (fine)", icen),
                  ("vel_speed", (speed_mod - speed_obs).cpu().numpy(), 30,
                   "speed model - obs [m/yr]", vvalid.cpu().numpy()),
                  ("vel_speed_eta", (eta_pix * speed_mod - speed_obs).cpu().numpy(), 30,
                   "eta*model - obs [m/yr] (per-glacier eta)", vvalid.cpu().numpy())]
        if dh is not None:
            panels.append(("dhdt_r", dhdt_raw.cpu().numpy(), 1.5, "dhdt model - obs [m/yr]",
                           dvalid.cpu().numpy()))
        for key in ("srf_z", "vel_z_u", "dhdt_z"):
            if key in fields and fields[key][0].shape == ice.shape:
                panels.append((key, fields[key][0].cpu().numpy(), 3.0, f"{key} (whitened)",
                               fields[key][1].cpu().numpy()))
        ncol = 2
        nrow = (len(panels) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(16, 5.5 * nrow), squeeze=False)
        for ax in axes.ravel():
            ax.set_axis_off()
        for ax, (key, arr, vmax, title, m) in zip(axes.ravel(), panels):
            im = ax.imshow(np.where(m, arr, np.nan), vmin=-vmax, vmax=vmax, cmap="RdBu_r")
            ax.set_title(f"{tag}: {title}")
            plt.colorbar(im, ax=ax, shrink=0.7)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/residual_maps.png", dpi=70)
        plt.close(fig)

        names = list(results)
        fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4), squeeze=False)
        for ax, name in zip(axes[0], names):
            rr = results[name]
            ax.semilogx(rr["h"] / 1000, rr["g"] / rr["var"], "k-", label="iso")
            ax.semilogx(rr["h"] / 1000, rr["ga"] / rr["var"], "r--", label="along")
            ax.semilogx(rr["h"] / 1000, rr["gc"] / rr["var"], "b--", label="across")
            ax.axhline(1.0, color="0.6", lw=0.8)
            ax.set_title(f"{name}  std {rr['std']:.2f}  a {rr['fit'][2] / 1000:.2f} km", fontsize=9)
            ax.set_xlabel("lag [km]")
            ax.set_ylim(0, 1.6)
        axes[0][0].set_ylabel("gamma / var")
        axes[0][0].legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/variograms.png", dpi=80)
        print(f"wrote {out_dir}/residual_maps.png, {out_dir}/variograms.png")


if __name__ == "__main__":
    warnings.simplefilter("once")
    main()
