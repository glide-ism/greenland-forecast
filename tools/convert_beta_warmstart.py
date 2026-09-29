#!/usr/bin/env python
"""
Convert a warm-start checkpoint's traction field to a different sliding law,
so an inversion under the new law starts from (approximately) the old one's
drag instead of from a field whose meaning changed.

  tau_b = beta * X * |u|^m * R(|u|)       (glide's drag, library change 19)

  X = xi_f                                 normalized law (sliding_N_scale_H = None)
  X = xi_f (H + N_floor_H) / N_scale_H     floored dimensional N
  R = (u0 / (|u| + u0))^m                  regularized Coulomb (sliding_u0 > 0), else 1

With the old law (normalized, Weertman: the checkpoints before 2026-09-28)
and the new one given here, the drag is kept at a reference state -- the
thickness and basal speed of a forward replay of the SAME checkpoint at
--ref-year (default the velocity epoch 2018) -- by

  log beta_new = log beta_old + p ln(N_scale_H / (H + N_floor_H)) - ln R(|u_b|)

on cells with ice (H > --min-H) in that state; ice-free cells keep their
beta (it carries no calibration there). The whitened field is
z = Whiten(log beta - mu_log_beta) under the domain config's log-beta prior,
exact to float32 (the round trip is checked); every other parameter of the
checkpoint is copied unchanged. Cells whose new beta exceeds the config's
beta_max are reported: the inversion clamps them.

Also re-whitens between log-beta PRIORS (--old-prior): a checkpoint's z is
only meaningful under the prior it was whitened with, so after changing
`log_beta_prior` (e.g. to a SpectralPriorHyperparams) convert with
  python tools/convert_beta_warmstart.py --checkpoint CKPT --old-prior 1,2000,1 --out NEW
(no --ref-run needed when the sliding law is unchanged).

Usage:
  python tools/convert_beta_warmstart.py \
      --checkpoint domains/greenland/inverse_v14/level_0/torch_vars.p \
      --ref-run domains/greenland/inverse_v14/forward_standalone \
      --u0 300 --n-scale 1000 --n-floor 100 \
      --out domains/greenland/inverse_v14/level_0/torch_vars_rc300_nf100.p
"""
import argparse
import re
import sys
from pathlib import Path

import cupy as cp
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from glacier_inverse import load_config                       # noqa: E402
from glacier_inverse.priors import GlacierPriors              # noqa: E402
from ismip_exporter import read_vti                           # noqa: E402


def frame(run_dir: Path, year: float) -> Path:
    pvd = next((run_dir / 'vti').glob('*.pvd'))
    for t, f in re.findall(r'timestep="([\d.]+)"[^>]*file="([^"]+)"', pvd.read_text()):
        if abs(float(t) - year) < 1e-3:
            return pvd.parent / f
    raise SystemExit(f'{run_dir}: no VTI frame at {year}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--domain-path', default=str(HERE.parent / 'domains' / 'greenland'))
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--ref-run', default=None, help='forward_standalone output of the same checkpoint (VTI with H and U_b)')
    ap.add_argument('--ref-year', type=float, default=2018.0)
    ap.add_argument('--u0', type=float, default=0.0, help='new sliding_u0 (m/yr); 0 = Weertman')
    ap.add_argument('--n-scale', type=float, default=None, help='new sliding_N_scale_H (m); omit for the normalized law')
    ap.add_argument('--n-floor', type=float, default=100.0, help='new sliding_N_floor_H (m)')
    ap.add_argument('--min-H', type=float, default=10.0, help='cells thinner than this in the reference keep their beta')
    ap.add_argument('--old-prior', default=None, metavar='SIGMA,L,NU',
                    help="the log-beta prior the checkpoint was WHITENED under, when it differs from the config's "
                         "(e.g. '1,2000,1' for the ggapp Matern of before 2026-09-29); the field is mapped to "
                         "physical with it and re-whitened under the config's prior (exact). Default: the config's")
    ap.add_argument('--old-mu', type=float, default=None, help="the checkpoint's mu_log_beta (default: the config's)")
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    cfg = load_config(a.domain_path)
    d = torch.load(a.checkpoint)
    z_old = d['log_beta'].detach().to('cuda')
    ny, nx = z_old.shape
    from glacier_inverse.priors import _cropped_inputs
    grid = _cropped_inputs(cfg, ['rgi_mask'])
    assert (grid.sizes['y'], grid.sizes['x']) == (ny, nx), (grid.sizes, z_old.shape)
    priors = GlacierPriors(cfg, ny, nx, float(abs(grid.x[1] - grid.x[0])))
    if a.old_prior:
        from glacier_inverse.config import PriorHyperparams
        from glacier_inverse.priors import _field_prior
        s_, l_, n_ = (float(v) for v in a.old_prior.split(','))
        old_model = _field_prior(PriorHyperparams(s_, l_, int(n_)), cfg.n_levels, ny, nx, float(abs(grid.x[1] - grid.x[0])))
        mu_old = priors.mu_log_beta if a.old_mu is None else a.old_mu
        log_beta = mu_old + old_model.forward(cp.asarray(z_old))
        print(f're-whitening from the old prior {a.old_prior} (mu {mu_old:g}) into {cfg.log_beta_prior}')
    else:
        with torch.no_grad():
            log_beta = cp.asarray(priors.log_beta_from_whitened(z_old))

    if a.ref_run is None:                                     # prior change only: no law conversion
        if a.u0 or a.n_scale:
            raise SystemExit('--u0 / --n-scale need --ref-run')
        H = cp.zeros_like(log_beta); ub = cp.zeros_like(log_beta)
    else:
        fr = read_vti(frame(Path(a.ref_run), a.ref_year), ['H', 'U_b'])
        H = cp.asarray(fr['H'], dtype=cp.float32)
        ub = cp.sqrt(cp.asarray(fr['U_b'][..., 0]) ** 2 + cp.asarray(fr['U_b'][..., 1]) ** 2 + float(cfg.u_reg))
    if H.shape != log_beta.shape:
        raise SystemExit(f'reference frame {H.shape} does not match the checkpoint grid {log_beta.shape}')
    p, m = 1.0, float(cfg.sliding_m)                           # glide's sliding.p is 1 (not configured)

    delta = cp.zeros_like(log_beta)
    if a.n_scale:
        delta += p * cp.log(a.n_scale / (cp.maximum(H, 0.0) + a.n_floor))
    if a.u0 > 0:
        delta -= m * cp.log(a.u0 / (ub + a.u0))
    ice = H > a.min_H
    delta = cp.where(ice, delta, 0.0)
    log_beta_new = log_beta + delta

    z_new = priors.log_beta_model.whiten(log_beta_new - priors.mu_log_beta)
    with torch.no_grad():
        back = cp.asarray(priors.log_beta_from_whitened(torch.as_tensor(z_new, device='cuda')))
    err = float(cp.abs(back - log_beta_new).max())
    beta_new = cp.exp(log_beta_new)
    bmax = getattr(cfg, 'beta_max', None)
    if a.ref_run is not None:
        print(f'ice cells {int(ice.sum())}; log-beta change on ice: median {float(cp.median(delta[ice])):+.3f}, '
              f'5-95 % {float(cp.percentile(delta[ice], 5)):+.3f} .. {float(cp.percentile(delta[ice], 95)):+.3f}')
    print(f'round-trip max |error| {err:.2e} (log beta, re-whitened under {cfg.log_beta_prior})')
    if bmax and a.ref_run is not None:
        print(f'beta > beta_max ({bmax:g}) on {int((beta_new[ice] > bmax).sum())} ice cells '
              f'(was {int((cp.exp(log_beta)[ice] > bmax).sum())}); the inversion clamps them')
    d['log_beta'] = torch.as_tensor(z_new, device=d['log_beta'].device, dtype=d['log_beta'].dtype)
    d['converted_sliding_law'] = dict(u0=a.u0, n_scale=a.n_scale, n_floor=a.n_floor, ref_run=str(a.ref_run),
                                      ref_year=a.ref_year, source=str(a.checkpoint), old_prior=a.old_prior,
                                      new_prior=repr(cfg.log_beta_prior))
    torch.save(d, a.out)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
