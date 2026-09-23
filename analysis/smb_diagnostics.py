"""SMB diagnostics of forward runs: snowline agreement, elevation bands,
calibrated SMB parameters, and interannual statistics against Mankoff.

    python analysis/smb_diagnostics.py \
        --run v2=domains/greenland/inverse_v2/forward_standalone \
        --run v3=domains/greenland/inverse_v3/forward_standalone \
        [--basin-dir analysis/output/basin_mb_v3]

Per run (yearly VTI frames; the frame at y + 1 carries year y's SMB):

  * snowline: the model's bare-ice area (SMB <= 0) per season 2000-2020 and
    its fraction of seasons with SMB > 0 by 200 m elevation band, against
    gridded_snowline.nc (the label of the inverse's snowline term), on the
    common mask (classified & model ice);
  * mean SMB (m/yr and Gt/yr) in four elevation bands over the same years;
  * calibrated tbias / pbias / f_clear / H_atm by band from the sibling
    physical_fields.nc, and the ice-sheet precipitation after pbias;
  * with --basin-dir (analysis/basin_mass_balance.py's output for the SAME
    run names): ice-sheet SMB mean / std / correlation / regression slope on
    Mankoff, dSMB/dT_jja on the CARRA2 ice-mean JJA series, MB by period.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ismip_exporter import read_vti  # noqa: E402

REGIONS = ['NO', 'NE', 'CE', 'SE', 'SW', 'CW', 'NW']
BANDS = [(0, 800), (800, 1400), (1400, 2000), (2000, 4000)]
N_LEVELS = 6
RHO_I = 917.0


def crop(a, factor=2 ** N_LEVELS):
    ny0, nx0 = a.shape[-2:]
    ny, nx = ny0 // factor * factor, nx0 // factor * factor
    y0, x0 = (ny0 - ny) // 2, (nx0 - nx) // 2
    return a[..., y0:y0 + ny, x0:x0 + nx]


def snowline_and_bands(name, run_dir, lab, lab_y, valid, z, years):
    pvd = next(Path(run_dir, 'vti').glob('*.pvd'))
    items = {round(float(t)): f for t, f in re.findall(r'timestep="([\d.]+)"[^>]*file="([^"]+)"', pvd.read_text())}
    psum = np.zeros(z.shape); ssum = np.zeros(z.shape); n_ice = np.zeros(z.shape)
    bare, prod = [], []
    for yr in years:
        a = read_vti(pvd.parent / items[yr + 1], ['smb', 'mask'])
        ice = a['mask'] < 0.5
        psum += (a['smb'] > 0)
        ssum += np.where(ice, a['smb'], 0.0); n_ice += ice
        bare.append(float(((a['smb'] <= 0) & valid & ice).sum()))
        prod.append(float(((1 - lab_y[yr]) * (valid & ice)).sum()))
    bare, prod = np.array(bare), np.array(prod)
    frac = psum / len(years)
    print(f'\n== {name}: {run_dir}')
    print(f'  bare-ice area {years[0]}-{years[-1]}: model {bare.mean() / 1e3:.0f}k km2 (std {bare.std() / 1e3:.0f}k), product '
          f'{prod.mean() / 1e3:.0f}k (std {prod.std() / 1e3:.0f}k), r {np.corrcoef(bare, prod)[0, 1]:.2f}; '
          + ', '.join(f'{y}: {bare[years.index(y)] / 1e3:.0f}k/{prod[years.index(y)] / 1e3:.0f}k' for y in (2012, 2013, 2019) if y in years))
    edges = np.arange(0, 2600, 200)
    print('  P(snow) model/label by band: ' + ' '.join(
        f'{lo}:{frac[s].mean():.2f}/{lab[s].mean():.2f}' for lo, hi in zip(edges[:-1], edges[1:])
        if (s := valid & (z >= lo) & (z < hi)).any()))
    print(f'  Brier vs label on the mask: {float(((frac - lab)[valid] ** 2).mean()):.4f}, mean bias {float((frac - lab)[valid].mean()):+.3f}')
    mean_smb = np.where(n_ice > 0, ssum / np.maximum(n_ice, 1), np.nan)
    for lo, hi in BANDS:
        s = valid & (z >= lo) & (z < hi) & np.isfinite(mean_smb)
        print(f'  mean SMB {lo}-{hi} m: {mean_smb[s].mean():+.2f} m/yr over {s.sum() / 1e3:.0f}k km2 -> {mean_smb[s].sum() * RHO_I / 1e6:+.0f} Gt/yr')


def parameters(run_dir, z, ice, precip):
    phys = Path(run_dir).parent / 'physical_fields.nc'
    if not phys.exists():
        return
    d = xr.open_dataset(phys)
    for lo, hi in BANDS + [(0, 4000)]:
        s = ice & (z >= lo) & (z < hi)
        print(f'  {lo}-{hi} m: tbias {float(d.tbias.values[s].mean()):+.2f} K, pbias x{float(np.exp(d.log_pbias.values[s]).mean()):.2f}, '
              f'f_clear {float(d.f_clear.values[s].mean()):.2f}, H_atm {float(d.H_atm.values[s].mean()):.1f}')
    print(f'  ice-sheet precip after pbias {float((precip * np.exp(d.log_pbias.values))[ice].sum() * RHO_I / 1e6):.0f} Gt/yr '
          f'(forcing raw {float(precip[ice].sum() * RHO_I / 1e6):.0f})')


def interannual(name, basin_dir, t_jja):
    f = Path(basin_dir) / f'model_{name}.csv'
    if not f.exists():
        return
    m = pd.read_csv(Path(basin_dir) / 'mankoff_annual.csv')
    x = pd.read_csv(f); x['year'] = x.time - 1
    d = m.merge(x, on='year', suffixes=('_obs', ''))
    if t_jja is not None:
        d = d.merge(t_jja, on='year')
    d = d[(d.year >= 1986) & (d.year <= 2025)]
    line = (f'  SMB mean {d.SMB_GrIS.mean():.0f} std {d.SMB_GrIS.std():.0f} (Mankoff {d.SMB_GrIS_obs.mean():.0f}/{d.SMB_GrIS_obs.std():.0f}), '
            f'r {d.SMB_GrIS.corr(d.SMB_GrIS_obs):.2f}, slope on Mankoff {np.polyfit(d.SMB_GrIS_obs, d.SMB_GrIS, 1)[0]:.2f}, '
            f'trend {np.polyfit(d.year, d.SMB_GrIS, 1)[0] * 10:.0f} (Mankoff {np.polyfit(d.year, d.SMB_GrIS_obs, 1)[0] * 10:.0f}) Gt/yr/decade')
    if t_jja is not None:
        line += (f', dSMB/dT_jja {np.polyfit(d.tas_ice_jja, d.SMB_GrIS, 1)[0]:.0f} '
                 f'(Mankoff {np.polyfit(d.tas_ice_jja, d.SMB_GrIS_obs, 1)[0]:.0f}) Gt/yr/K')
    print(line)
    for a, b in ((1986, 2005), (2006, 2025)):
        q = d[(d.year >= a) & (d.year <= b)]
        print(f'  {a}-{b}: MB {q.MB_GrIS.mean():.0f} (Mankoff {q.MB_GrIS_obs.mean():.0f}), SMB {q.SMB_GrIS.mean():.0f} ({q.SMB_GrIS_obs.mean():.0f})')
    print('  regional SMB mean model/Mankoff: ' + ' '.join(f"{r}:{d[f'SMB_{r}'].mean():.0f}/{d[f'SMB_{r}_obs'].mean():.0f}" for r in REGIONS))
    print('  regional SMB std  model/Mankoff: ' + ' '.join(f"{r}:{d[f'SMB_{r}'].std():.0f}/{d[f'SMB_{r}_obs'].std():.0f}" for r in REGIONS))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--domain-path', default='domains/greenland')
    ap.add_argument('--run', action='append', default=[], metavar='NAME=DIR')
    ap.add_argument('--basin-dir', default=None)
    ap.add_argument('--inputs-file', default='GLIDE_inputs.nc',
                    help='the merged inputs the runs were forced with (precip for the pbias totals), e.g. GLIDE_inputs_hybrid.nc')
    ap.add_argument('--years', type=int, nargs=2, default=(2000, 2020))
    ap.add_argument('--tjja-scalars', default='domains/greenland/inverse/projection_CARRA2_ocx_nofb/scalars.csv',
                    help='a forward_projection scalars.csv of a CARRA2-forced run: the ice-mean JJA forcing series')
    a = ap.parse_args()
    inputs = Path(a.domain_path) / 'model_inputs'
    sl = xr.open_dataset(inputs / 'gridded_snowline.nc')
    gi = xr.open_dataset(inputs / a.inputs_file)
    lab = crop(sl.snow_fraction.values); gf = crop(sl.glacier_fraction.values)
    valid = (gf > 0) & np.isfinite(lab)
    z = crop(gi.elevation.values); ice = crop(gi.rgi_mask.values) > 0.5
    precip = crop(gi.monthly_precip.values).mean(0)
    years = list(range(a.years[0], a.years[1] + 1))
    lab_y = {int(y): crop(sl.snow_label.sel(year=y).values) / 100.0 for y in sl.year.values if int(y) in years}
    t_jja = None
    if a.tjja_scalars and Path(a.tjja_scalars).exists():
        sc = pd.read_csv(a.tjja_scalars); sc['year'] = sc.time - 1
        t_jja = sc[['year', 'tas_ice_jja']].dropna()
    for spec in a.run:
        name, run_dir = spec.split('=', 1)
        snowline_and_bands(name, run_dir, lab, lab_y, valid, z, years)
        parameters(run_dir, z, ice, precip)
        if a.basin_dir:
            interannual(name, a.basin_dir, t_jja)


if __name__ == '__main__':
    main()
