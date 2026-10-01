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
  Dg_r     = sum over Mankoff's flux gates of   (--gates gates.gpkg: the SAME integral as
             rho (v_s . n)+ H w                 Mankoff's D, with the model's surface velocity
                                                and thickness under the gate pixels; compare
                                                with Mankoff's D, NOT D + BMB)
  MBg_r    = SMB_r - Dg_r                       (the Mankoff-comparable mass balance: their
                                                MB = SMB - D - BMB is the budget upstream of
                                                the gates, and so is this)

The gate integral on the OBSERVED mosaic and BedMachine thickness at the
model grid is printed as the reference: Mankoff integrates 200 m velocity
against 150 m thickness, and at 1 km the narrow fast fjords (CW, SE) lose
some 20 % of their flux, so a model gate number is read against that ceiling
rather than against Mankoff directly. The residual D_r and the gate Dg_r
differ by whatever leaves the region without crossing a gate -- in the v7
state that is a thin marine apron removed by the geometric calving
criterion, some 200-270 Gt/yr -- so the two together locate the loss.

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
sys.path.insert(0, str(HERE.parent))
from ismip_exporter import read_vti  # noqa: E402
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


# -------------------------------------------------------------------- gates
GATE_PIX = 200.0          # m; the raster the gates were drawn on
LAT_TS = 70.0             # EPSG:3413 standard parallel, for the true pixel width


def load_gates(path, gi):
    """Mankoff et al. (2020) flux gates (gates.gpkg, EPSG:3413), prepared for
    the model grid: per gate a unit normal, the model cells under its pixels
    and the metres of gate each carries, and its Mouginot region.

    The gpkg rows are NOT single 200 m pixels. They are column strips of the
    rasterized gate, each polygon holding 1-8 raster pixels (its area over
    200 m^2), and the 2765 rows hold 5890 pixels. Giving every row 200 m
    undercounts the gate length by 2.1x (553 km against a true 1163 km) and
    was the "resolution" shortfall first seen here. So: the gate's true
    length is its extent along the principal axis of its pixel centres
    (plus one pixel), distributed over its raster pixels, and each row gets
    its share. The normal is the perpendicular to that axis, oriented
    downstream by the OBSERVED velocity so the sign never depends on the
    model. The width is divided by the projection's point scale k so it is
    a ground distance, as Mankoff does. Velocity is taken at the model cell
    under each pixel centre."""
    import geopandas as gpd
    g = gpd.read_file(path)
    g['k_pix'] = (g.geometry.area / GATE_PIX ** 2).round().astype(int).clip(lower=1)
    X, Y = gi.x.values, gi.y.values
    dx, dy = float(X[1] - X[0]), float(Y[1] - Y[0])
    vx, vy = gi.vx.values, gi.vy.values
    gates = []
    for gid, sub in g.groupby('gate'):
        P = np.c_[sub.x.values, sub.y.values].astype(float)
        if len(P) > 1:
            C = P - P.mean(0)
            _, _, V = np.linalg.svd(C, full_matrices=False)
            t = V[0]
            L = float((C @ t).max() - (C @ t).min()) + GATE_PIX
        else:
            t, L = np.array([1.0, 0.0]), GATE_PIX
        n = np.array([-t[1], t[0]]); n = n / np.linalg.norm(n)
        ix = np.clip(np.rint((P[:, 0] - X[0]) / dx).astype(int), 0, len(X) - 1)
        iy = np.clip(np.rint((P[:, 1] - Y[0]) / dy).astype(int), 0, len(Y) - 1)
        ov = np.nan_to_num(np.c_[vx[iy, ix], vy[iy, ix]])
        if (ov @ n).sum() < 0:
            n = -n
        k = (1 + np.sin(np.radians(LAT_TS))) / (1 + np.sin(np.radians(sub.mean_lat.values)))
        w = sub.k_pix.values * (L / int(sub.k_pix.sum())) / k
        gates.append(dict(gate=int(gid), region=str(sub.region.iloc[0]), n=n, ix=ix, iy=iy, w=w))
    return gates


def gate_discharge(H, u, v, gates):
    """Gt/yr through the gates per region: rho * (v . n)+ * H * width, summed
    over each gate's pixels -- Mankoff's per-pixel formula with the model's
    fields under the pixels. Outflow only (the clip), as a gate flux is."""
    out = {r: 0.0 for r in REGIONS}
    for gt in gates:
        iy, ix = gt['iy'], gt['ix']
        vn = u[iy, ix] * gt['n'][0] + v[iy, ix] * gt['n'][1]
        f = RHO_I * np.clip(np.nan_to_num(vn), 0.0, None) * np.nan_to_num(H[iy, ix]) * gt['w']
        out[gt['region']] += float(f.sum()) / 1e12
    return out


# ------------------------------------------------------------------- frames
def _vti_arrays(path, names):
    """Named Float32 arrays of a glide VTIWriter file in the model's row order
    (ismip_exporter.read_vti: raw or LZ4/zlib compressed appended layout)."""
    return read_vti(path, names)


def _in(t, t_read):
    return t_read is None or t_read[0] - 1e-6 <= t <= t_read[1] + 1e-6


def frames_vti(run_dir, t_read=None):
    """Yield (time, H, smb, (u_s, v_s)) for a forward_standalone run, the
    initial state first (with smb and velocity None); with `t_read` = (t0, t1)
    only the frames inside it are read."""
    run_dir = Path(run_dir)
    pvd = next((run_dir / 'vti').glob('*.pvd'))
    items = re.findall(r'timestep="([\d.]+)"[^>]*file="([^"]+)"', pvd.read_text())
    meta = next((run_dir / fn for fn in ('forward_soln.nc', 'snapshots.nc', 'final_state.nc') if (run_dir / fn).exists()), None)
    attrs = xr.open_dataset(meta).attrs if meta else {}
    t0 = float(attrs.get('t_start', np.nan))
    phys = run_dir.parent / 'physical_fields.nc'
    if (np.isfinite(t0) and phys.exists() and int(attrs.get('level', 0)) == 0 and float(items[0][0]) > t0
            and _in(t0, t_read)):
        H0 = crop_to_factor(xr.open_dataset(phys), 2 ** N_LEVELS).H_init.values
        yield t0, H0, None, None
    for t, fn in items:
        if not _in(float(t), t_read):
            continue
        a = _vti_arrays(pvd.parent / fn, ['H', 'smb', 'U_s'])
        yield float(t), a['H'], a['smb'], (a['U_s'][..., 0], a['U_s'][..., 1])


def frames_snapshots(run_dir):
    ds = xr.open_dataset(Path(run_dir) / 'snapshots.nc')
    for k in range(ds.sizes['time']):
        yield float(ds.time[k]), ds.H[k].values, ds.smb[k].values, (ds.u_s[k].values, ds.v_s[k].values)


def frames_series_nc(run_dir):
    """Yield (time, H, smb) from tools/vti_to_nc.py's series.nc (preferred:
    yearly, compressed), the initial state first when physical_fields.nc is
    next to the run and the series does not start at t_start."""
    import netCDF4
    run_dir = Path(run_dir)
    nc = netCDF4.Dataset(run_dir / 'series.nc')
    t = np.asarray(nc['model_year'][:], dtype=float)
    t0 = float(nc.getncattr('run_t_start')) if 'run_t_start' in nc.ncattrs() else np.nan
    phys = run_dir.parent / 'physical_fields.nc'
    if np.isfinite(t0) and t[0] > t0 and phys.exists() and int(float(nc.getncattr('run_level'))) == 0:
        H0 = crop_to_factor(xr.open_dataset(phys), 2 ** N_LEVELS).H_init.values
        yield t0, H0, None, None
    for k in range(len(t)):
        yield (float(t[k]), np.asarray(nc['H'][k, :, :]), np.asarray(nc['smb'][k, :, :]),
               (np.asarray(nc['u_s'][k, :, :]), np.asarray(nc['v_s'][k, :, :])))
    nc.close()


def model_series(run_dir, masks, dx, prefer='auto', gates=None, t_read=None):
    """`prefer`: 'auto' takes the finest source available (series.nc, then
    yearly VTI, then snapshots); 'snapshots' forces snapshots.nc, which for a
    projection is every SNAPSHOT_EVERY years instead of every step -- 220
    reads instead of 522, and plenty for a cumulative curve."""
    run_dir = Path(run_dir)
    if prefer == 'snapshots' and (run_dir / 'snapshots.nc').exists():
        gen = frames_snapshots(run_dir)
    elif (run_dir / 'series.nc').exists():
        gen = frames_series_nc(run_dir)
    elif (run_dir / 'snapshots.nc').exists() and not (run_dir / 'vti').exists():
        gen = frames_snapshots(run_dir)
    elif list((run_dir / 'vti').glob('*.pvd')) if (run_dir / 'vti').exists() else False:
        gen = frames_vti(run_dir, t_read)  # yearly frames beat the decadal snapshots
    else:
        gen = frames_snapshots(run_dir)
    rows = []
    shape = next(iter(masks.values())).shape
    for t, H, smb, vel in gen:
        if not _in(t, t_read):
            continue
        if H.shape != shape:
            # a coarse-level run (frames on the run level): repeat onto the
            # level-0 grid, which conserves the area integrals exactly; the
            # gate sampling then reads the coarse cell's value
            f, (ny0, nx0) = shape[0] // H.shape[0], H.shape
            up = lambda a: None if a is None else np.kron(a, np.ones((f, f), a.dtype))[:shape[0] + (a.shape[0] - ny0) * f,
                                                                                    :shape[1] + (a.shape[1] - nx0) * f]
            H, smb = up(H), up(smb)
            vel = None if vel is None else tuple(up(v) for v in vel)
        row = {'time': t}
        ice = H > H_MIN
        for r, m in masks.items():
            row[f'M_{r}'] = float((H * (ice & m)).sum()) * dx * dx * RHO_I / 1e12
            row[f'SMB_{r}'] = float((smb * (ice & m)).sum()) * dx * dx * RHO_I / 1e12 if smb is not None else np.nan
        if gates is not None:
            dg = gate_discharge(H, vel[0], vel[1], gates) if vel is not None else {r: np.nan for r in REGIONS}
            for r in REGIONS:
                row[f'Dg_{r}'] = dg[r]
        rows.append(row)
        print(f"  {run_dir.name}: t={t:.1f}", end='\r', flush=True)
    print()
    df = pd.DataFrame(rows).set_index('time').sort_index()
    for r in REGIONS:
        df[f'MB_{r}'] = df[f'M_{r}'].diff() / pd.Series(df.index, index=df.index).diff()
    for k in ('M', 'SMB', 'MB') + (('Dg',) if gates is not None else ()):
        df[f'{k}_GrIS'] = df[[f'{k}_{r}' for r in REGIONS]].sum(axis=1, min_count=1)
    if gates is not None:
        # the Mankoff-comparable mass balance: SMB minus the GATE flux, i.e.
        # the budget of the ice upstream of the gates, which is what their
        # MB = SMB - D - BMB is. The residual SMB - dM/dt counts everything
        # downstream of the gates as discharge too.
        for r in REGIONS + ['GrIS']:
            df[f'MBg_{r}'] = df[f'SMB_{r}'] - df[f'Dg_{r}']
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
SCEN_COLOR = {'ssp126': 'tab:blue', 'ssp370': 'tab:orange', 'ssp585': 'tab:red', 'ctrl': 'tab:gray'}
# reserved for runs without a scenario (the standalone / OCX reference), kept
# clear of SCEN_COLOR so the reference never collides with a projection
PLAIN_COLOR = ('tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive')


def style_for(name, i, cycle):
    """Colour by scenario and dash by GCM so six projections stay readable;
    a run with no scenario in its name is the reference and gets its own
    colour and a heavier line."""
    for k, c in SCEN_COLOR.items():
        if k in name:
            return c, ('--' if 'MRI' in name.upper() else '-'), 1.4
    return PLAIN_COLOR[i % len(PLAIN_COLOR)], '-', 2.6


def plot(runs, obs, out_dir, ref_year, t_range, ref_run=None):
    panels = REGIONS + ['GrIS']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # rates
    fig, axs = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True)
    for ax, r in zip(axs.ravel(), panels):
        yo = obs.index + 0.5
        ax.fill_between(yo, obs[f'MB_{r}'] - obs[f'MBerr_{r}'], obs[f'MB_{r}'] + obs[f'MBerr_{r}'], color='k', alpha=.15, lw=0)
        ax.plot(yo, obs[f'MB_{r}'], 'k-', lw=1.5, label='Mankoff MB')
        ax.plot(yo, obs[f'SMB_{r}'], 'k:', lw=0.8, alpha=.6, label='Mankoff SMB')
        ax.plot(yo, -obs[f'DB_{r}'], 'k--', lw=0.8, alpha=.4, label='Mankoff -(D+BMB)')
        if any(f'Dg_{r}' in df for df in runs.values()):
            ax.plot(yo, -obs[f'D_{r}'], 'k-.', lw=1.2, label='Mankoff -D (gates)')
        for i, (name, df) in enumerate(runs.items()):
            c, ls, lw = style_for(name, i, colors)
            df = df[(df.index >= t_range[0] - 1) & (df.index <= t_range[1] + 10)]
            tm = df.index.to_series().rolling(2).mean()            # mid-interval
            ax.plot(tm, df[f'MB_{r}'], ls, color=c, lw=lw, label=f'{name} MB')
            ax.plot(df.index, df[f'SMB_{r}'], ':', color=c, lw=0.8, alpha=.6, label=f'{name} SMB')
            ax.plot(tm, df[f'MB_{r}'] - df[f'SMB_{r}'].rolling(2).mean(), '--', color=c, lw=0.8, alpha=.4, label=f'{name} -D residual')
            if f'Dg_{r}' in df:
                ax.plot(df.index, -df[f'Dg_{r}'], '-.', color=c, lw=1.2, label=f'{name} -D gates')
                ax.plot(df.index, df[f'MBg_{r}'], '-', color=c, lw=1.0, alpha=.5, label=f'{name} MB = SMB-Dgates')
        ax.axhline(0, color='grey', lw=.5); ax.set_title(r); ax.grid(alpha=.3); ax.set_xlim(*t_range)
        if r == 'GrIS':
            ax.legend(fontsize=7, ncol=2)
    for ax in axs[1]:
        ax.set_xlabel('year')
    for ax in axs[:, 0]:
        ax.set_ylabel('Gt / yr')
    fig.suptitle('Annual mass balance by Mouginot & Rignot region: Mankoff et al. (black) vs model')
    fig.tight_layout(); fig.savefig(out_dir / 'basin_mb_rates.png', dpi=130); plt.close(fig)

    # cumulative. With `ref_run` every experiment is offset by THAT run's mass
    # at ref_year, so the curves keep their mutual bias instead of each being
    # forced through zero; Mankoff has only rates, so its cumulative is
    # anchored at zero there, i.e. on the reference run.
    fig, axs = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True)
    for ax, r in zip(axs.ravel(), panels):
        o = obs[obs.index >= ref_year]
        cum = o[f'MB_{r}'].cumsum(); err = o[f'MBerr_{r}'].cumsum()
        yo = o.index + 1.0
        ax.fill_between(np.r_[ref_year, yo], np.r_[0, cum - err], np.r_[0, cum + err], color='k', alpha=.15, lw=0)
        ax.plot(np.r_[ref_year, yo], np.r_[0, cum], 'k-', lw=2.0, label='Mankoff', zorder=5)
        m_ref = (np.interp(ref_year, runs[ref_run].index, runs[ref_run][f'M_{r}'])
                 if ref_run is not None else None)
        for i, (name, df) in enumerate(runs.items()):
            c, ls, lw = style_for(name, i, colors)
            d = df[(df.index >= t_range[0] - 1e-6) & (df.index <= t_range[1])]
            if len(d) == 0:
                continue
            m0 = m_ref if m_ref is not None else np.interp(ref_year, df.index, df[f'M_{r}'])
            ax.plot(d.index, d[f'M_{r}'] - m0, ls, color=c, lw=lw, label=name)
        ax.axvline(ref_year, color='grey', lw=.5, ls=':')
        ax.axhline(0, color='grey', lw=.5); ax.set_title(r); ax.grid(alpha=.3)
        ax.set_xlim(t_range[0], t_range[1])
        if r == 'GrIS':
            ax.legend(fontsize=7, ncol=2)
    for ax in axs[1]:
        ax.set_xlabel('year')
    anchor = f'{ref_run} at {ref_year:g}' if ref_run is not None else f'each run at {ref_year:g}'
    for ax in axs[:, 0]:
        ax.set_ylabel(f'mass relative to\n{anchor} (Gt)')
    fig.suptitle(f'Cumulative mass change by region, all experiments relative to {anchor} '
                 f'(Mankoff anchored there too)')
    fig.tight_layout(); fig.savefig(out_dir / 'basin_mb_cumulative.png', dpi=130); plt.close(fig)


def plot_clean(runs, obs, out_dir, t_range):
    """The two fluxes only, both positive: Mankoff SMB and gate discharge with
    their error bands, and each run's SMB and gate discharge. No mass balance,
    no residual -- the readable version of the rates plot for judging a
    calving-parameter change."""
    panels = REGIONS + ['GrIS']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axs = plt.subplots(2, 4, figsize=(17, 7.5), sharex=True)
    yo = obs.index + 0.5
    for ax, r in zip(axs.ravel(), panels):
        for v, ls, lab in (('SMB', '-', 'SMB'), ('D', '--', 'D (gates)')):
            ax.fill_between(yo, obs[f'{v}_{r}'] - obs[f'{v}err_{r}'], obs[f'{v}_{r}'] + obs[f'{v}err_{r}'],
                            color='k', alpha=.12, lw=0)
            ax.plot(yo, obs[f'{v}_{r}'], 'k' + ls, lw=1.6, label=f'Mankoff {lab}')
        for i, (name, df) in enumerate(runs.items()):
            c, _, _ = style_for(name, i, colors)
            d = df[(df.index >= t_range[0] - 1) & (df.index <= t_range[1] + 1)]
            ax.plot(d.index, d[f'SMB_{r}'], '-', color=c, lw=1.4, label=f'{name} SMB')
            if f'Dg_{r}' in d:
                ax.plot(d.index, d[f'Dg_{r}'], '--', color=c, lw=1.4, label=f'{name} D (gates)')
        ax.axhline(0, color='grey', lw=.5); ax.set_title(r); ax.grid(alpha=.3); ax.set_xlim(*t_range)
        if r == 'GrIS':
            ax.legend(fontsize=7, ncol=2)
    for ax in axs[1]:
        ax.set_xlabel('year')
    for ax in axs[:, 0]:
        ax.set_ylabel('Gt / yr')
    fig.suptitle('SMB and discharge through the flux gates by region: Mankoff et al. (black, shaded 1 sigma) vs model')
    fig.tight_layout(); fig.savefig(out_dir / 'basin_mb_clean.png', dpi=130); plt.close(fig)


def main(args):
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    masks, dx, gi = region_masks(args.domain_path)
    print("region cells:", {r: int(m.sum()) for r, m in masks.items()})
    gates, obs_gate = None, {}
    if args.gates:
        gates = load_gates(args.gates, gi)
        # the same integral on the observed mosaic + BedMachine thickness at
        # the model grid: the ceiling a 1 km field can reach against Mankoff's
        # 200 m / 150 m integral, so a model number is read against THIS
        obs_gate = gate_discharge(gi.thickness_obs.values, gi.vx.values, gi.vy.values, gates)
        obs_gate['GrIS'] = sum(obs_gate[r] for r in REGIONS)
        print(f"gates: {len(gates)} from {args.gates}; observed fields at {dx / 1e3:g} km through them: "
              + ', '.join(f'{r} {obs_gate[r]:.0f}' for r in REGIONS + ['GrIS']) + ' Gt/yr')
    obs = mankoff_annual(HERE.parent / 'common_data' / 'dhdt' / 'mankoff' / 'MB_region.nc')
    obs.to_csv(out_dir / 'mankoff_annual.csv')
    runs = {}
    for spec in args.run:
        name, path = spec.split('=', 1)
        print(f"reading {name} from {path}")
        runs[name] = model_series(path, masks, dx, prefer=args.prefer, gates=gates, t_read=args.t_read)
        runs[name].to_csv(out_dir / f'model_{name}.csv')
    for spec in args.run_csv:
        # a series this script wrote earlier (model_<name>.csv), for a run whose
        # frames are gone or too slow to re-read
        name, path = spec.split('=', 1)
        runs[name] = pd.read_csv(path).set_index('time').sort_index()
        print(f"loaded {name} from {path}")
    # summary over the overlap
    lo, hi = max(1986, args.t_range[0]), args.t_range[1]
    rows = []
    for name, df in runs.items():
        tm = df.index.to_series().rolling(2).mean()
        for r in REGIONS + ['GrIS']:
            mm = df[f'MB_{r}'][(tm >= lo) & (tm <= hi)]
            oo = obs[f'MB_{r}'][(obs.index + 0.5 >= lo) & (obs.index + 0.5 <= hi)]
            row = dict(run=name, region=r, model_MB=mm.mean(), obs_MB=oo.mean(),
                       model_SMB=df[f'SMB_{r}'][(df.index >= lo) & (df.index <= hi)].mean(),
                       obs_SMB=obs[f'SMB_{r}'][(obs.index >= lo) & (obs.index <= hi)].mean(),
                       obs_D_plus_BMB=obs[f'DB_{r}'][(obs.index >= lo) & (obs.index <= hi)].mean())
            if f'Dg_{r}' in df:
                w = (df.index >= lo) & (df.index <= hi)
                row.update(model_Dgate=df[f'Dg_{r}'][w].mean(), model_MBgate=df[f'MBg_{r}'][w].mean(),
                           obs_D=obs[f'D_{r}'][(obs.index >= lo) & (obs.index <= hi)].mean(),
                           obs_Dgate_1km=obs_gate.get(r, np.nan))
            rows.append(row)
    S = pd.DataFrame(rows)
    S['model_D'] = S.model_SMB - S.model_MB
    print(f"\nmean over {lo:g}-{hi:g} (Gt/yr):")
    print(S.round(1).to_string(index=False))
    S.to_csv(out_dir / 'summary.csv', index=False)
    if args.ref_run is not None and args.ref_run not in runs:
        raise SystemExit(f'--ref-run {args.ref_run!r} is not one of {list(runs)}')
    plot(runs, obs, out_dir, args.ref_year, args.t_range, ref_run=args.ref_run)
    if gates is not None or any(f'Dg_GrIS' in df for df in runs.values()):
        plot_clean(runs, obs, out_dir, (max(1985.0, args.t_range[0]), min(2027.0, args.t_range[1])))
    print(f"wrote {out_dir}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--domain-path', default='domains/greenland')
    ap.add_argument('--run', action='append', default=[], metavar='NAME=DIR',
                    help='forward_standalone or forward_projection output directory (repeatable)')
    ap.add_argument('--run-csv', action='append', default=[], metavar='NAME=CSV',
                    help='add a series written earlier by this script (model_<name>.csv)')
    ap.add_argument('--ref-year', type=float, default=2000.0)
    ap.add_argument('--t-range', type=float, nargs=2, default=(1985, 2027))
    ap.add_argument('--out-dir', default=str(HERE / 'output' / 'basin_mb'))
    ap.add_argument('--ref-run', default=None,
                    help='name of the run every curve is offset by (its mass at --ref-year), so the '
                         'cumulative panels show the experiments\' mutual bias; default: each run '
                         'referenced to itself')
    ap.add_argument('--gates', default=None, metavar='GPKG',
                    help="Mankoff et al. (2020) gates.gpkg: also compute discharge THROUGH the gates "
                         "per region (Dg_*), the Mankoff-comparable MB = SMB - Dg (MBg_*), and the "
                         "observed-at-model-resolution reference the model number should be read against")
    ap.add_argument('--prefer', default='auto', choices=('auto', 'snapshots'),
                    help="'snapshots' forces snapshots.nc over the yearly VTI series (much faster "
                         "for projections, and enough for cumulative curves)")
    ap.add_argument('--t-read', type=float, nargs=2, default=None, metavar=('T0', 'T1'),
                    help="read only the frames in [T0, T1] (a 1850-2300 projection's yearly VTI series is 450 "
                         "frames; the plots need only --t-range, plus one year before it for the first rate)")
    a = ap.parse_args()
    if not a.run and not a.run_csv:
        a.run = ['reference=domains/greenland/inverse_vinther_bedgrad/forward_standalone']
    main(a)
