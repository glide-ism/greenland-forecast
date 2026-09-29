#!/usr/bin/env python
"""
Force-balance initialization of the basal traction: a PARTIAL warm-start
checkpoint (only the traction fields) whose beta makes every ice cell slide
at the observed rate from the first iteration. It depends only on the domain
inputs and the config, not on any earlier run; loading it
(`load_whitened_params_into`) keeps the problem's own initialization for every
other parameter (bed from the observed geometry, SMB fields at the prior
median).

Why: the velocity misfit's sensitivity to log beta is d u_s / d ln beta =
-(1/m) u_b, i.e. proportional to the BASAL speed. From a uniform, sticky start
(beta_init 4, m = 1/3, the dimensional N making thick ice 2-3x stickier) the
interior has u_b ~ 0.01 m/yr, so its gradient is ~1e4 too small whatever its
misfit, and the optimizer frees the ice only as a wave from the outlets that
dies where the membrane coupling runs out (probe 2026-09-29, scratch
probe_beta_gradient.py: NEGIS stuck at beta ~3.3 beyond 400 km upstream).
A start with the right amount of sliding everywhere puts every cell on the
live part of the objective.

Per cell, on the domain's fine grid, assuming PLUG FLOW (all of the observed
speed is sliding; deformation is left to the inversion):
  tau_d / (rho_i g) = H |grad S|                     (head units, S smoothed over --smooth-km)
  u_b   = max(u_obs, --min-ub)
  beta0 = tau_d / (X^p K(u_b) u_b)                    glide's drag at that sliding speed:
          K = (u_b^2 + u_reg)^((m-1)/2) (u0 / (sqrt(u_b^2 + u_reg) + u0))^m   (Coulomb when u0 > 0)
          X = xi_f (H + N_floor_H) / N_scale_H or xi_f (normalized), xi_f = clip(1 - d/(r H), 0, 1)
with H = thickness_obs, S = elevation, d = -bed (bed_obs where finite), r = 0.917 (glide's
kernel constant), p = 1. Cells that
are floating, ice-free, unobserved or thinner than --min-H take the prior mean; log beta0
is smoothed over --smooth-log-km (normalized convolution on the valid cells) and clipped
to [--beta-min, config.beta_max].

The field is then whitened under the domain config's log-beta prior(s):
  no log_beta_mean_prior      z_log_beta = W(log beta0 - mu)
  centered (default)          z_log_beta = W(log beta0 - mu), z_log_beta_mean = W_mean(m0),
                              m0 = log beta0 - mu smoothed at log_beta_mean_prior.l / 2
                              (the bed_mean seeding: the prior residual starts small)
  additive                    z_log_beta_mean = W_mean(m0), z_log_beta = W(log beta0 - mu - m0)

Usage:
  python tools/beta_force_balance.py --out domains/greenland/beta_init_force_balance.p
then set inverse.py WARM_START_PATH to the output.
"""
import argparse
import sys
from pathlib import Path

import cupy as cp
import numpy as np
import torch
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from glacier_inverse import load_config                                   # noqa: E402
from glacier_inverse.priors import GlacierPriors, _cropped_inputs          # noqa: E402
from ggapp.torch import GGaPPMap                                           # noqa: E402

R_KERNEL = 0.917          # glide's RHO_I_OVER_RHO_W (common.cu), what xi is computed with


def smooth_valid(a, valid, sigma):
    """Gaussian smoothing of `a` over the `valid` cells only (normalized convolution)."""
    if sigma <= 0:
        return a
    num = ndimage.gaussian_filter(np.where(valid, a, 0.0), sigma)
    den = ndimage.gaussian_filter(valid.astype(float), sigma)
    return np.where(den > 1e-3, num / np.maximum(den, 1e-12), a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--domain-path', default=str(HERE.parent / 'domains' / 'greenland'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--smooth-km', type=float, default=4.0, help='surface smoothing for the driving stress')
    ap.add_argument('--smooth-log-km', type=float, default=2.0, help='smoothing of log beta0 over the valid cells')
    ap.add_argument('--min-ub', type=float, default=1.0, help='u_b floor (m/yr)')
    ap.add_argument('--min-H', type=float, default=20.0)
    ap.add_argument('--beta-min', type=float, default=1e-3)
    a = ap.parse_args()

    cfg = load_config(a.domain_path)
    g = _cropped_inputs(cfg, ['vx', 'vy', 'thickness_obs', 'elevation', 'bed_obs', 'rgi_mask'])
    x = g.x.values
    dx = float(abs(x[1] - x[0]))
    H = np.nan_to_num(g.thickness_obs.values)
    S = np.nan_to_num(g.elevation.values)
    bed = np.where(np.isfinite(g.bed_obs.values), g.bed_obs.values, S - H)
    uobs = np.hypot(g.vx.values, g.vy.values)
    ice = (g.rgi_mask.values > 0.5) & (H > a.min_H)

    Ss = ndimage.gaussian_filter(S, a.smooth_km * 1e3 / dx)
    gy, gx = np.gradient(Ss, dx)
    slope = np.hypot(gx, gy)
    tau_head = H * slope                                              # tau_d / (rho_i g), m
    m, p = float(cfg.sliding_m), 1.0

    Hf = np.maximum(H, 1.0)
    xi_f = np.clip(1.0 - (-bed) / (R_KERNEL * Hf), 0.0, 1.0)
    Hs = getattr(cfg, 'sliding_N_scale_H', None)
    X = xi_f * (H + cfg.sliding_N_floor_H) / Hs if Hs else xi_f
    valid = ice & (xi_f > 0) & np.isfinite(uobs) & (uobs > 0) & (tau_head > 0)

    ub = np.maximum(np.nan_to_num(uobs), a.min_ub)                   # plug flow
    Sreg = ub ** 2 + float(cfg.u_reg)
    u0 = float(getattr(cfg, 'sliding_u0', 0.0) or 0.0)
    R = (u0 / (np.sqrt(Sreg) + u0)) ** m if u0 > 0 else 1.0
    K = Sreg ** ((m - 1) / 2) * R
    with np.errstate(divide='ignore', invalid='ignore'):
        beta0 = tau_head / (np.maximum(X, 1e-6) ** p * K * ub)
    mu = float(cfg.mu_log_beta)
    lb = np.where(valid, np.log(np.clip(beta0, a.beta_min, None)), mu)
    lb = smooth_valid(lb, valid, a.smooth_log_km * 1e3 / dx)
    lb = np.where(valid, lb, mu)
    bmax = getattr(cfg, 'beta_max', None)
    lb = np.clip(lb, np.log(a.beta_min), np.log(bmax) if bmax else None)

    # report
    print(f'valid cells {int(valid.sum())} of {int(ice.sum())} ice; u_b at the --min-ub floor on {np.mean(uobs[valid] < a.min_ub):.1%}')
    print(f"{'obs class':>12s} {'n':>8s} {'beta0 median':>13s} {'5-95 %':>16s} {'X':>6s}")
    for lo, hi in [(0, 5), (5, 15), (15, 30), (30, 100), (100, 300), (300, 1000), (1000, 1e5)]:
        mm = valid & (uobs >= lo) & (uobs < hi)
        if mm.sum() < 50:
            continue
        b = np.exp(lb[mm])
        print(f'{lo:5.0f}-{hi:<6.0f} {mm.sum():8d} {np.median(b):13.3f} {np.percentile(b, 5):7.3f}..{np.percentile(b, 95):7.3f} '
              f'{np.median(X[mm]):6.2f}')

    # whiten under the configured prior(s); a PARTIAL checkpoint (traction only)
    ny, nx = lb.shape
    priors = GlacierPriors(cfg, ny, nx, dx)
    d = {'bed_parametrization': priors.bed_parametrization}
    x0 = cp.asarray(lb - mu, dtype=cp.float32)
    mode = getattr(cfg, 'log_beta_mean_mode', 'centered')
    if priors.log_beta_mean_model is not None:
        m0 = cp.asarray(smooth_valid(lb - mu, np.ones_like(valid), cfg.log_beta_mean_prior.l / 2 / dx), dtype=cp.float32)
        z_mean = priors.log_beta_mean_model.whiten(m0)
        z = priors.log_beta_model.whiten(x0 - m0 if mode == 'additive' else x0)
        d['log_beta_mean'] = torch.as_tensor(z_mean, device='cuda', dtype=torch.float32)
    else:
        z = priors.log_beta_model.whiten(x0)
    d['log_beta'] = torch.as_tensor(z, device='cuda', dtype=torch.float32)
    with torch.no_grad():
        back = priors.log_beta_from_whitened(d['log_beta'], d.get('log_beta_mean'))
    err = float((back - torch.as_tensor(lb, device='cuda')).abs().max())
    prior_cost = 0.5 * float((d['log_beta'] ** 2).sum())
    print(f'log beta round trip max |error| {err:.2e} ({mode if priors.log_beta_mean_model is not None else "single field"}); '
          f'0.5|z_log_beta|^2 = {prior_cost:.3g}'
          + (f', 0.5|z_log_beta_mean|^2 = {0.5 * float((d["log_beta_mean"] ** 2).sum()):.3g}' if 'log_beta_mean' in d else ''))
    d['beta_init_source'] = dict(tool='beta_force_balance.py (plug flow)', smooth_km=a.smooth_km,
                                 smooth_log_km=a.smooth_log_km, min_ub=a.min_ub, u0=u0,
                                 N_scale_H=Hs, N_floor_H=cfg.sliding_N_floor_H, m=m,
                                 log_beta_prior=repr(cfg.log_beta_prior),
                                 log_beta_mean_prior=repr(getattr(cfg, 'log_beta_mean_prior', None)),
                                 mode=mode)
    torch.save(d, a.out)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
