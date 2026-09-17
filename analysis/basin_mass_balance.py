"""Modelled vs observed mass balance per Mouginot & Rignot region.

Observations: Mankoff et al. (GEUS) "Greenland ice sheet mass balance from
1840 through next week", `common_data/dhdt/mankoff/MB_region.nc`: daily
MB / SMB / D per region (NO, NE, CE, SE, SW, CW, NW; Gt/day), regional
series from 1986. Summed to calendar years; the annual uncertainty is the
sum of the daily uncertainties (fully correlated within a year, the
product's own convention).

Model: any forward run with time-resolved thickness -- a forward_standalone
output directory (annual VTI frames in `vti/`, the initial state from the
sibling `physical_fields.nc`) or a forward_projection output (`snapshots.nc`,
every SNAPSHOT_EVERY years). Per region r (cells whose `rgi_label` is one of
the 260 Mouginot basins with that SUBREGION1; peripheral glaciers excluded):

  M_r(t)   = sum_r H dx^2 rho_i                (Gt; cells with H > 2 m)
  MB_r     = dM_r / dt between frames           (Gt/yr, plotted at mid-interval)
  SMB_r    = sum_r smb dx^2 rho_i               (over the same ice cells; the SMB the
                                                model computes on ice-free land never
                                                removes mass and is left out)
  D_r      = SMB_r - MB_r                       (discharge + calving + front retreat, as a
                                                residual; the model has no basal melt, so it
                                                is compared with Mankoff's D + BMB)

    python analysis/basin_mass_balance.py \
        --run reference=domains/greenland/inverse_vinther_bedgrad/forward_standalone \
        [--run ssp126_anom=domains/greenland/inverse_vinther_bedgrad/projection_CESM2-WACCM_ssp126]
Writes basin_mb_rates.png, basin_mb_cumulative.png and the annual series
(CSV) to analysis/output/basin_mb/.
"""
import argparse
import re
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
REGIONS = ['NO', 'NE', 'CE', 'SE', 'SW', 'CW', 'NW']
RHO_I = 917.0
H_MIN = 2.0          # m; above the thklim floor
N_LEVELS = 6         # GlacierProblem's crop factor 2**n_levels


# ----------------------------------------------------------------- regions
def crop_to_factor(ds, factor):
    ny0, nx0 = ds.sizes['y'], ds.sizes['x']
    ny, nx = (ny0 // factor) * factor, (nx0 // factor) * factor
    y0, x0 = (ny0 - ny) // 2, (nx0 - nx) // 2
    return ds.isel(y=slice(y0, y0 + ny), x=slice(x0, x0 + nx))


def region_masks(domain_path):
    """(region -> bool (ny, nx) on the cropped model grid, dx)."""
    domain_path = Path(domain_path)
    gi = crop_to_factor(xr.open_dataset(domain_path / 'model_inputs' / 'GLIDE_inputs.nc'), 2 ** N_LEVELS)
    labels = gi.rgi_label.values.astype("float64")          # _FillValue -1 decodes to NaN
    labels = np.where(np.isfinite(labels), labels, -1).astype(int)
    basins = gpd.read_file(HERE.parent / 'common_data' / 'area' / 'basins' / 'Greenland_Basins_PS_v1.4.2.shp')
    names = gi.rgi_id.values.astype(str)
    assert list(names[:len(basins)]) == list(basins.NAME.astype(str)), "rgi_id order differs from the shapefile"
    reg_of_label = np.full(len(names), '', dtype='U2')
    reg_of_label[:len(basins)] = basins.SUBREGION1.astype(str).values
    reg = np.full(labels.shape, '', dtype='U2')
    ok = labels >= 0
    reg[ok] = reg_of_label[labels[ok]]
    masks = {r: reg == r for r in REGIONS}
    dx = float(abs(gi.x[1] - gi.x[0]))
    return masks, dx, gi


# ------------------------------------------------------------------- frames
def _vti_arrays(path, names):
    """Read named Float32 arrays from a glide VTIWriter file (raw appended
    data, UInt32 length headers, y written south-to-north)."""
    with open(path, 'rb') as f:
        head = f.read(1 << 20)
        i = head.index(b'<AppendedData encoding="raw">')
        j = head.index(b'_', i) + 1
        xml = head[:i].decode('utf-8', 'ignore')
        ext = [int(v) for v in re.search(r'WholeExtent="([^"]+)"', xml).group(1).split()]
        nx, ny = ext[1] - ext[0] + 1, ext[3] - ext[2] + 1
        offs = {m.group(1): int(m.group(2)) for m in re.finditer(r'Name="(\w+)"[^>]*format="appended" offset="(\d+)"', xml)}
        out = {}
        for n in names:
            f.seek(j + offs[n])
            nbytes = int(np.frombuffer(f.read(4), np.uint32)[0])
            a = np.frombuffer(f.read(nbytes), np.float32).reshape(ny, nx)
            out[n] = a[::-1].copy()                           # back to the domain's row order
    return out


def frames_vti(run_dir):
    """Yield (time, H, smb) for a forward_standalone run, the initial state first."""
    run_dir = Path(run_dir)
    pvd = next((run_dir / 'vti').glob('*.pvd'))
    items = re.findall(r'timestep="([\d.]+)"[^>]*file="([^"]+)"', pvd.read_text())
    soln = xr.open_dataset(run_dir / 'forward_soln.nc')
    t0 = float(soln.attrs.get('t_start', np.nan))
    phys = run_dir.parent / 'physical_fields.nc'
    if np.isfinite(t0) and phys.exists() and int(soln.attrs.get('level', 0)) == 0:
        H0 = crop_to_factor(xr.open_dataset(phys), 2 ** N_LEVELS).H_init.values
        yield t0, H0, None
    for t, fn in items:
        a = _vti_arrays(pvd.parent / fn, ['H', 'smb'])
        yield float(t), a['H'], a['smb']


def frames_snapshots(run_dir):
    ds = xr.open_dataset(Path(run_dir) / 'snapshots.nc')
    for k in range(ds.sizes['time']):
        yield float(ds.time[k]), ds.H[k].values, ds.smb[k].values


def model_series(run_dir, masks, dx):
    run_dir = Path(run_dir)
    gen = frames_snapshots(run_dir) if (run_dir / 'snapshots.nc').exists() else frames_vti(run_dir)
    rows = []
    for t, H, smb in gen:
        row = {'time': t}
        ice = H > H_MIN
        for r, m in masks.items():
            row[f'M_{r}'] = float((H * (ice & m)).sum()) * dx * dx * RHO_I / 1e12
            row[f'SMB_{r}'] = float((smb * (ice & m)).sum()) * dx * dx * RHO_I / 1e12 if smb is not None else np.nan
        rows.append(row)
        print(f"  {run_dir.name}: t={t:.1f}", end='\r', flush=True)
    print()
    df = pd.DataFrame(rows).set_index('time').sort_index()
    for r in REGIONS:
        df[f'MB_{r}'] = df[f'M_{r}'].diff() / pd.Series(df.index, index=df.index).diff()
    for k in ('M', 'SMB', 'MB'):
        df[f'{k}_GrIS'] = df[[f'{k}_{r}' for r in REGIONS]].sum(axis=1, min_count=1)
    return df


def mankoff_annual(path):
    ds = xr.open_dataset(path)
    yr = ds.time.dt.year
    out = {}
    for v in ('MB', 'SMB', 'D', 'BMB'):
        a = ds[f'{v}_ROI'].groupby(yr).sum(min_count=300)
        e = ds[f'{v}_ROI_err'].groupby(yr).sum(min_count=300)
        for r in REGIONS:
            out[f'{v}_{r}'] = a.sel(region=r).to_series()
            out[f'{v}err_{r}'] = e.sel(region=r).to_series()
        out[f'{v}_GrIS'] = a.sum('region', min_count=7).to_series()
        out[f'{v}err_GrIS'] = np.sqrt((e ** 2).sum('region', min_count=7)).to_series()
    df = pd.DataFrame(out)
    # the model's residual D has no basal melt term to separate: compare it with D + BMB
    for r in REGIONS + ['GrIS']:
        df[f'DB_{r}'] = df[f'D_{r}'] + df[f'BMB_{r}']
    df.index.name = 'year'
    return df[df.index >= 1986]


# ------------------------------------------------------------------- plots
def plot(runs, obs, out_dir, ref_year, t_range):
    panels = REGIONS + ['GrIS']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # rates
    fig, axs = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True)
    for ax, r in zip(axs.ravel(), panels):
        yo = obs.index + 0.5
        ax.fill_between(yo, obs[f'MB_{r}'] - obs[f'MBerr_{r}'], obs[f'MB_{r}'] + obs[f'MBerr_{r}'], color='k', alpha=.15, lw=0)
        ax.plot(yo, obs[f'MB_{r}'], 'k-', lw=1.5, label='Mankoff MB')
        ax.plot(yo, obs[f'SMB_{r}'], 'k:', lw=0.8, alpha=.6, label='Mankoff SMB')
        ax.plot(yo, -obs[f'DB_{r}'], 'k--', lw=0.8, alpha=.6, label='Mankoff -(D+BMB)')
        for (name, df), c in zip(runs.items(), colors):
            df = df[(df.index >= t_range[0] - 1) & (df.index <= t_range[1] + 10)]
            tm = df.index.to_series().rolling(2).mean()            # mid-interval
            ax.plot(tm, df[f'MB_{r}'], '-', color=c, lw=1.5, label=f'{name} MB')
            ax.plot(df.index, df[f'SMB_{r}'], ':', color=c, lw=0.8, alpha=.6, label=f'{name} SMB')
            ax.plot(tm, df[f'MB_{r}'] - df[f'SMB_{r}'].rolling(2).mean(), '--', color=c, lw=0.8, alpha=.6, label=f'{name} -D (MB-SMB)')
        ax.axhline(0, color='grey', lw=.5); ax.set_title(r); ax.grid(alpha=.3); ax.set_xlim(*t_range)
        if r == 'GrIS':
            ax.legend(fontsize=7, ncol=2)
    for ax in axs[1]:
        ax.set_xlabel('year')
    for ax in axs[:, 0]:
        ax.set_ylabel('Gt / yr')
    fig.suptitle('Annual mass balance by Mouginot & Rignot region: Mankoff et al. (black) vs model')
    fig.tight_layout(); fig.savefig(out_dir / 'basin_mb_rates.png', dpi=130); plt.close(fig)

    # cumulative since ref_year
    fig, axs = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True)
    for ax, r in zip(axs.ravel(), panels):
        o = obs[obs.index >= ref_year]
        cum = o[f'MB_{r}'].cumsum(); err = o[f'MBerr_{r}'].cumsum()
        yo = o.index + 1.0
        ax.fill_between(np.r_[ref_year, yo], np.r_[0, cum - err], np.r_[0, cum + err], color='k', alpha=.15, lw=0)
        ax.plot(np.r_[ref_year, yo], np.r_[0, cum], 'k-', lw=1.5, label='Mankoff')
        for (name, df), c in zip(runs.items(), colors):
            d = df[(df.index >= ref_year - 1e-6) & (df.index <= t_range[1])]
            if len(d) == 0:
                continue
            m0 = np.interp(ref_year, df.index, df[f'M_{r}'])
            ax.plot(d.index, d[f'M_{r}'] - m0, '-', color=c, lw=1.5, label=name)
        ax.axhline(0, color='grey', lw=.5); ax.set_title(r); ax.grid(alpha=.3); ax.set_xlim(ref_year, t_range[1])
        if r == 'GrIS':
            ax.legend(fontsize=8)
    for ax in axs[1]:
        ax.set_xlabel('year')
    for ax in axs[:, 0]:
        ax.set_ylabel(f'mass change since {ref_year:g} (Gt)')
    fig.suptitle(f'Cumulative mass change since {ref_year:g} by region')
    fig.tight_layout(); fig.savefig(out_dir / 'basin_mb_cumulative.png', dpi=130); plt.close(fig)


def main(args):
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    masks, dx, gi = region_masks(args.domain_path)
    print("region cells:", {r: int(m.sum()) for r, m in masks.items()})
    obs = mankoff_annual(HERE.parent / 'common_data' / 'dhdt' / 'mankoff' / 'MB_region.nc')
    obs.to_csv(out_dir / 'mankoff_annual.csv')
    runs = {}
    for spec in args.run:
        name, path = spec.split('=', 1)
        print(f"reading {name} from {path}")
        runs[name] = model_series(path, masks, dx)
        runs[name].to_csv(out_dir / f'model_{name}.csv')
    # summary over the overlap
    lo, hi = max(1986, args.t_range[0]), args.t_range[1]
    rows = []
    for name, df in runs.items():
        tm = df.index.to_series().rolling(2).mean()
        for r in REGIONS + ['GrIS']:
            mm = df[f'MB_{r}'][(tm >= lo) & (tm <= hi)]
            oo = obs[f'MB_{r}'][(obs.index + 0.5 >= lo) & (obs.index + 0.5 <= hi)]
            rows.append(dict(run=name, region=r, model_MB=mm.mean(), obs_MB=oo.mean(),
                             model_SMB=df[f'SMB_{r}'][(df.index >= lo) & (df.index <= hi)].mean(),
                             obs_SMB=obs[f'SMB_{r}'][(obs.index >= lo) & (obs.index <= hi)].mean(),
                             obs_D_plus_BMB=obs[f'DB_{r}'][(obs.index >= lo) & (obs.index <= hi)].mean()))
    S = pd.DataFrame(rows)
    S['model_D'] = S.model_SMB - S.model_MB
    print(f"\nmean over {lo:g}-{hi:g} (Gt/yr):")
    print(S.round(1).to_string(index=False))
    S.to_csv(out_dir / 'summary.csv', index=False)
    plot(runs, obs, out_dir, args.ref_year, args.t_range)
    print(f"wrote {out_dir}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--domain-path', default='domains/greenland')
    ap.add_argument('--run', action='append', default=[], metavar='NAME=DIR',
                    help='forward_standalone or forward_projection output directory (repeatable)')
    ap.add_argument('--ref-year', type=float, default=2000.0)
    ap.add_argument('--t-range', type=float, nargs=2, default=(1985, 2027))
    ap.add_argument('--out-dir', default=str(HERE / 'output' / 'basin_mb'))
    a = ap.parse_args()
    if not a.run:
        a.run = ['reference=domains/greenland/inverse_vinther_bedgrad/forward_standalone']
    main(a)
