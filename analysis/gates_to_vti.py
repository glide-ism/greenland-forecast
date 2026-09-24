"""Mankoff's flux gates as a VTI on the model grid, to overlay on a run's frames.

    python analysis/gates_to_vti.py --run domains/greenland/inverse_v8/forward_standalone --year 2018

Same geometry as the run's frames (index space, origin (0, 0), 1 km spacing,
flip_y, point data on the cropped grid), so it drops into the same ParaView
view. Per cell, zero away from the gates:

  gate_id, region_id (1..7 = NO NE CE SE SW CW NW), gate_width_m (metres of
  gate in the cell, the weight in the flux integral), normal_x / normal_y
  (unit gate normal, oriented downstream by the observed velocity),
  obs_vn (gate-orthogonal observed speed, m/yr), obs_H (BedMachine, m),
  obs_flux (Gt/yr through this cell's share of the gate);
  with --run: mod_vn, mod_H, mod_flux for the frame at --year, and
  flux_ratio = mod_flux / obs_flux (0 where there is no observed flux).

All from basin_mass_balance.load_gates, i.e. the same integral the
--gates analysis uses, so what is coloured here is what is summed there.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from ismip_exporter import read_vti                    # noqa: E402
from glide.io import write_vti                         # noqa: E402
from basin_mass_balance import region_masks, load_gates, REGIONS, RHO_I   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--domain-path', default='domains/greenland')
    ap.add_argument('--gates', default='common_data/dhdt/mankoff/dataverse_files/gates.gpkg')
    ap.add_argument('--run', default=None, help='a run directory with vti/; adds the model arrays')
    ap.add_argument('--year', type=int, default=2018)
    ap.add_argument('--out', default=None, help='default: analysis/output/gates/mankoff_gates[_<run>_<year>].vti')
    a = ap.parse_args()

    masks, dx, gi = region_masks(a.domain_path)
    gates = load_gates(a.gates, gi)
    ny, nx = gi.sizes['y'], gi.sizes['x']
    z = lambda: np.zeros((ny, nx), np.float32)
    gid, reg, width, n_x, n_y = z(), z(), z(), z(), z()
    for gt in gates:
        for iy, ix, w in zip(gt['iy'], gt['ix'], gt['w']):
            gid[iy, ix] = gt['gate']
            reg[iy, ix] = REGIONS.index(gt['region']) + 1
            width[iy, ix] += w
            n_x[iy, ix], n_y[iy, ix] = gt['n']
    on = width > 0
    vn = lambda u, v: np.where(on, np.clip(np.nan_to_num(u) * n_x + np.nan_to_num(v) * n_y, 0.0, None), 0.0).astype(np.float32)
    obs_vn = vn(gi.vx.values, gi.vy.values)
    obs_H = np.where(on, np.nan_to_num(gi.thickness_obs.values), 0.0).astype(np.float32)
    obs_flux = (RHO_I * obs_vn * obs_H * width / 1e12).astype(np.float32)
    data = {'gate_id': gid, 'region_id': reg, 'gate_width_m': width, 'normal_x': n_x, 'normal_y': n_y,
            'obs_vn': obs_vn, 'obs_H': obs_H, 'obs_flux': obs_flux}
    tag = ''
    if a.run:
        pvd = next(Path(a.run, 'vti').glob('*.pvd'))
        items = {round(float(t)): f for t, f in re.findall(r'timestep="([\d.]+)"[^>]*file="([^"]+)"', pvd.read_text())}
        fr = read_vti(pvd.parent / items[a.year], ['H', 'U_s'])
        mod_vn = vn(fr['U_s'][..., 0], fr['U_s'][..., 1])
        mod_H = np.where(on, fr['H'], 0.0).astype(np.float32)
        mod_flux = (RHO_I * mod_vn * mod_H * width / 1e12).astype(np.float32)
        ratio = np.where(obs_flux > 0, mod_flux / np.maximum(obs_flux, 1e-12), 0.0).astype(np.float32)
        data.update(mod_vn=mod_vn, mod_H=mod_H, mod_flux=mod_flux, flux_ratio=ratio)
        tag = f'_{Path(a.run).parent.name}_{Path(a.run).name}_{a.year}'
        print(f'{a.run} @ {a.year}: gate flux model {mod_flux.sum():.1f} vs observed {obs_flux.sum():.1f} Gt/yr '
              f'({mod_flux.sum() / obs_flux.sum():.2f}); cells with obs flux but no model ice: '
              f'{int(((obs_flux > 0) & (mod_H <= 2)).sum())} of {int((obs_flux > 0).sum())}')
    out = Path(a.out) if a.out else HERE / 'output' / 'gates' / f'mankoff_gates{tag}.vti'
    out.parent.mkdir(parents=True, exist_ok=True)
    write_vti(str(out), data, dx=float(dx), flip_y=True, compressor='lz4',
              precision={k: 1e-3 for k in data if k not in ('gate_id', 'region_id')})
    print(f'wrote {out} ({out.stat().st_size / 1e6:.1f} MB): {int(on.sum())} gate cells, {len(gates)} gates')


if __name__ == '__main__':
    main()
