"""SpectralFieldPrior checks: agreement with ggapp's MaternPrior for a single
mass-1 component, exact round trip / self-adjointness, sample statistics, and
the prior cost of a coherent regional shift for decoupled-mass variants.
Run: python tests/test_spectral_prior.py"""
import sys, math
import numpy as np, cupy as cp
sys.path.insert(0, '.')
from glacier_inverse.config import PriorHyperparams, PriorComponent, SpectralPriorHyperparams
from glacier_inverse.priors import SpectralFieldPrior, _build_matern_prior
import cupyx.scipy.ndimage as ndi

ny, nx, dx = 512, 384, 1000.0
ok = True
def check(name, cond, info=''):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name}  {info}")

# 1. single Matern component vs ggapp (sigma 1, l 2 km, nu 1)
g = _build_matern_prior(PriorHyperparams(1.0, 2000.0, 1), 5, ny, nx, dx)
s = SpectralFieldPrior(SpectralPriorHyperparams((PriorComponent(1.0, 2000.0, 1.0),)), ny, nx, dx)
z = cp.random.standard_normal((ny, nx), dtype=cp.float32)
xg, xs = g.forward(z), s.forward(z)
rel = float(cp.linalg.norm(xg - xs) / cp.linalg.norm(xg))
check('mass-1 component reproduces ggapp MaternPrior.forward', rel < 1e-3, f'rel diff {rel:.2e}')
x = cp.random.standard_normal((ny, nx), dtype=cp.float32)
relw = float(cp.linalg.norm(g.whiten(x) - s.whiten(x)) / cp.linalg.norm(g.whiten(x)))
check('... and whiten', relw < 1e-3, f'rel diff {relw:.2e}')

variants = {
    'Matern 1 / 2 km (current)': SpectralPriorHyperparams((PriorComponent(1.0, 2000.0),)),
    'intrinsic 2 km, sigma_mean 1': SpectralPriorHyperparams((PriorComponent(1.0, 2000.0, mass=0.0),), sigma_mean=1.0),
    'Matern 2 km + Matern 1 / 200 km': SpectralPriorHyperparams((PriorComponent(1.0, 2000.0), PriorComponent(1.0, 200e3))),
    'Matern 2 km + intrinsic 1 / 200 km, sigma_mean 1': SpectralPriorHyperparams(
        (PriorComponent(1.0, 2000.0), PriorComponent(1.0, 200e3, mass=0.0)), sigma_mean=1.0),
}
# a coherent +0.1 shift over a 200 x 150 km block with 5 km smooth edges
blk = cp.zeros((ny, nx), cp.float32); blk[150:350, 100:250] = 1.0
shift = ndi.gaussian_filter(blk, 5.0) * 0.1
bump = cp.zeros((ny, nx), cp.float32); bump[250, 180] = 1.0
bump = ndi.gaussian_filter(bump, 2.0); bump *= 0.1 / float(bump.max())       # a 2-km-scale local feature
for name, hp in variants.items():
    m = SpectralFieldPrior(hp, ny, nx, dx)
    z = cp.random.standard_normal((ny, nx), dtype=cp.float32)
    rt = float(cp.linalg.norm(m.whiten(m.forward(z)) - z) / cp.linalg.norm(z))
    a, b = cp.random.standard_normal((2, ny, nx), dtype=cp.float32)
    adj = abs(float((m.forward(a) * b).sum() - (a * m.forward(b)).sum())) / float(cp.abs(m.forward(a) * b).sum())
    samp = cp.stack([m.sample() for _ in range(20)])
    local_std = float(cp.std(samp - samp.mean(axis=(1, 2), keepdims=True)))
    cost_shift = 0.5 * float((m.whiten(shift) ** 2).sum())
    cost_bump = 0.5 * float((m.whiten(bump) ** 2).sum())
    check(f'{name}: round trip / self-adjoint', rt < 1e-4 and adj < 1e-4, f'({rt:.1e}, {adj:.1e})')
    print(f'      sample std about the mean {local_std:.2f}; prior cost of a +0.1 shift over 200x150 km {cost_shift:10.2f}; '
          f'of a +0.1 2-km bump {cost_bump:7.3f}')
print('ALL PASS' if ok else 'FAILURES')
