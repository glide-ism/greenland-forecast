"""Arctic amplification over Greenland from the RACMO2.3p2-ERA5 1 km record
(1958-2025) relative to the HadCRUT5 global-mean anomaly, and the hybrid
"RACMO before RACMO" product for spin-ups.

Model, per cell x and calendar month m, with G(y) the regressor series:
the HadCRUT annual global anomaly (degC rel. 1961-1990; --regressor
hadcrut) or the Vinther SW-Greenland JJA station anomaly as built by
preprocessing/make_temperature_anomaly.py (--regressor vinther, the
config's forcing series):

    T(x, m, y) = a(x, m) + A(x, m) * G(y) + eps            [amplification A, K/K]
    P(x, m, y) = Pclim(x, m) * (1 + s(x, m) * (G(y) - Gref)) [sensitivity s, 1/K]

so the local anomaly per degree of global warming is A, the number the
config's scalar `alpha_t2m` stands in for. The temperature variable is
`tas` (unclipped 2 m) by default; with --temp-var ts (censored at 0 degC)
A is fitted twice: on all years (`A_all`, biased low where the monthly mean
hits the cap) and on the years in which the cell is below the cap
(`A_unclipped`, needs >= MIN_YEARS such years). For `tas` the two coincide.
The trend ratio (local trend / global trend over the record) is kept as a
cross-check.

Hybrid reconstruction for year y (any year the spliced PAGES2k+HadCRUT
series covers):

    T_hyb = Tclim + A_eff * (G(y) - Gref)        [capped at 0 for ts]
    P_hyb = Pclim * max(1 + s * (G(y) - Gref), 0.1)

with Tclim/Pclim the 1961-1990 RACMO climatology (Gref = mean G over the
same window, ~0 by HadCRUT's baseline) and A_eff = A_unclipped where it is
identified, else A_all, then Gaussian-smoothed over --smooth-km. With
--hybrid-years Y0 Y1 the reconstruction is written year by year into
common_data/climate/RACMO2.3p2-ERA-hybrid/{ts,pr}/ in the RACMO file layout,
so any reader of the RACMO tree can be pointed at it.

Outputs (analysis/output/amplification_{var}/):
    amplification.nc     A_all, A_unclipped, A_eff, se, r2, intercepts,
                         clipped_frac, trend ratios, precip sensitivity,
                         Tclim/Pclim (12, y, x) at 1 km
    regional_series.nc   monthly region / elevation-band mean ts & pr, 1958-2025
    stacks_5km.nc        ts, pr (year, month, y, x) block-averaged to 5 km
    figures *.png, tables *.csv
"""
import argparse
import sys
import time
from pathlib import Path

import warnings
warnings.filterwarnings('ignore', 'Mean of empty slice')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import distance_transform_edt, gaussian_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'preprocessing'))
from racmo_common import (ELEV_BANDS, RACMO_BASE, REGIONS, T_MELT, extent, is_censored, load_grid, load_hadcrut,
                          load_racmo, pr_to_m_ice_per_yr, racmo_path, racmo_years, region_raster)

CLIM_WINDOW = (1961, 1990)
MIN_YEARS = 30
CLIP_EPS = 0.01          # degC below the cap counts as unclipped
COARSEN = 5
SEASONS = {'DJF': [11, 0, 1], 'MAM': [2, 3, 4], 'JJA': [5, 6, 7], 'SON': [8, 9, 10]}
MONTHS = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']


class Regress:
    """Running per-cell sums for T = a + A*G with an optional per-sample mask."""

    def __init__(self, shape, masked):
        self.masked = masked
        z = lambda: np.zeros(shape, dtype='float64')
        self.sT, self.sGT, self.sTT = z(), z(), z()
        if masked:
            self.n, self.sG, self.sGG = z(), z(), z()
        else:
            self.n, self.sG, self.sGG = 0.0, 0.0, 0.0

    def add(self, T, G, mask=None):
        T = T.astype('float64')
        if self.masked:
            w = mask.astype('float64')
            T = np.where(mask, T, 0.0)
            self.n += w; self.sG += w * G; self.sGG += w * G * G
        else:
            T = np.nan_to_num(T)
            self.n += 1; self.sG += G; self.sGG += G * G
        self.sT += T; self.sGT += G * T; self.sTT += T * T

    def solve(self, min_n=3):
        n = self.n
        with np.errstate(divide='ignore', invalid='ignore'):
            sxx = self.sGG - self.sG ** 2 / n
            sxy = self.sGT - self.sG * self.sT / n
            syy = self.sTT - self.sT ** 2 / n
            A = sxy / sxx
            a = (self.sT - A * self.sG) / n
            ss_res = syy - A * sxy
            r2 = 1 - ss_res / syy
            se = np.sqrt(np.maximum(ss_res, 0) / np.maximum(n - 2, 1) / sxx)
        bad = ~(np.asarray(n) >= min_n)
        for f in (A, a, r2, se):
            f[bad] = np.nan
        return A, a, r2, se


def _fill_nearest(a, valid):
    bad = ~np.isfinite(a)
    if not bad.any():
        return a
    idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
    out = a[tuple(idx)]
    return np.where(valid, out, np.nan)


def _map(ax, field, grid, title, vmin, vmax, cmap, mask, label=''):
    im = ax.imshow(np.where(mask, field, np.nan), extent=extent(grid), vmin=vmin, vmax=vmax,
                   cmap=cmap, interpolation='nearest')
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=label)


def accumulate(grid, years, G, out_dir, tvar='tas', regressor_label='HadCRUT5 global'):
    censored = is_censored(tvar)
    ice = grid.rgi_mask.values > 0.5
    z = grid.elevation.values
    regions = region_raster(grid)
    shape = (12,) + ice.shape
    ny, nx = ice.shape
    cy, cx = ny // COARSEN, nx // COARSEN

    rT_all, rT_unc = Regress(shape, False), Regress(shape, True)
    rP = Regress(shape, False)
    trT, trP = Regress(shape, False), Regress(shape, False)        # against calendar year
    n_clip = np.zeros(shape); n_fin = None
    clim_T = np.zeros(shape); clim_P = np.zeros(shape); n_clim = 0
    reg_names = REGIONS + ['GrIS', 'all']
    band_names = [f'{lo}-{hi}' for lo, hi in ELEV_BANDS]
    series_T = np.full((len(years), 12, len(reg_names)), np.nan)
    series_P = np.full_like(series_T, np.nan)
    band_T = np.full((len(years), 12, len(band_names)), np.nan)
    band_P = np.full_like(band_T, np.nan)
    stack_T = np.full((len(years), 12, cy, cx), np.nan, dtype='float32')
    stack_P = np.full_like(stack_T, np.nan)

    t0 = time.time()
    for iy, y in enumerate(years):
        T = load_racmo(tvar, y, grid)
        P = load_racmo('pr', y, grid)
        if n_fin is None:
            fin = np.isfinite(T[0]) & ice
            n_fin = fin
            reg_sel = {name: (fin & ((regions >= 1) & (regions <= 7)) if name == 'GrIS' else
                              fin & (regions >= 1) if name == 'all' else fin & (regions == i + 1))
                       for i, name in enumerate(reg_names)}
            band_sel = {b: fin & (z >= lo) & (z < hi) for b, (lo, hi) in zip(band_names, ELEV_BANDS)}
        g = float(G[y])
        unclipped = (T < -CLIP_EPS) if censored else np.isfinite(T)
        rT_all.add(T, g); rT_unc.add(T, g, unclipped & np.isfinite(T)); rP.add(P, g)
        trT.add(T, float(y)); trP.add(P, float(y))
        n_clip += ~unclipped & np.isfinite(T)
        if CLIM_WINDOW[0] <= y <= CLIM_WINDOW[1]:
            clim_T += np.nan_to_num(T); clim_P += np.nan_to_num(P); n_clim += 1
        for j, name in enumerate(reg_names):
            s = reg_sel[name]
            if s.any():
                series_T[iy, :, j] = T[:, s].mean(1); series_P[iy, :, j] = P[:, s].mean(1)
        for j, b in enumerate(band_names):
            s = band_sel[b]
            if s.any():
                band_T[iy, :, j] = T[:, s].mean(1); band_P[iy, :, j] = P[:, s].mean(1)
        Tc = T[:, :cy * COARSEN, :cx * COARSEN].reshape(12, cy, COARSEN, cx, COARSEN)
        Pc = P[:, :cy * COARSEN, :cx * COARSEN].reshape(12, cy, COARSEN, cx, COARSEN)
        with np.errstate(invalid='ignore'):
            stack_T[iy] = np.nanmean(Tc, axis=(2, 4)); stack_P[iy] = np.nanmean(Pc, axis=(2, 4))
        if iy % 10 == 0 or y == years[-1]:
            print(f'  {y}: {time.time() - t0:.0f} s', flush=True)

    fin = n_fin
    Gv = np.array([G[y] for y in years])
    coords_t = dict(t=np.arange(12), y=grid.y, x=grid.x)
    dims = ('t', 'y', 'x')
    A_all, a_all, r2_all, se_all = rT_all.solve()
    A_unc, a_unc, r2_unc, se_unc = rT_unc.solve(min_n=MIN_YEARS)
    slopeP, interP, r2P, seP = rP.solve()
    trend_T, _, _, _ = trT.solve()
    trend_P, _, _, _ = trP.solve()
    g_trend = np.polyfit(years, Gv, 1)[0]
    clim_T /= n_clim; clim_P /= n_clim
    Gref = float(np.mean([G[y] for y in range(CLIM_WINDOW[0], CLIM_WINDOW[1] + 1)]))
    with np.errstate(divide='ignore', invalid='ignore'):
        s_frac = slopeP / clim_P
        s_frac_ice = np.where(fin[None], s_frac, np.nan)
    clipped_frac = n_clip / len(years)

    # effective amplification: unclipped fit where identified, else all-years fit,
    # nearest-filled within the RACMO footprint, then smoothed
    A_eff = np.where(np.isfinite(A_unc), A_unc, A_all)
    A_eff = np.stack([_fill_nearest(np.where(fin, A_eff[m], np.nan), fin) for m in range(12)])
    ds = xr.Dataset(coords=coords_t)
    ds['spatial_ref'] = grid['spatial_ref']
    for name, arr, attrs in [
            ('A_all', A_all, dict(units='K/K', long_name='amplification, all years (ts censored at 0 C)')),
            ('A_unclipped', A_unc, dict(units='K/K', long_name=f'amplification, years below the cap (>= {MIN_YEARS} yrs)')),
            ('A_eff', A_eff, dict(units='K/K', long_name='A_unclipped where identified else A_all, nearest-filled')),
            ('se_all', se_all, dict(units='K/K')), ('se_unclipped', se_unc, dict(units='K/K')),
            ('r2_all', r2_all, {}), ('r2_unclipped', r2_unc, {}),
            ('intercept_all', a_all, dict(units='degC', long_name='T at G = 0')),
            ('clipped_frac', clipped_frac, dict(long_name='fraction of years with monthly ts at the 0 C cap')),
            ('trend_ratio_T', trend_T / g_trend, dict(units='K/K', long_name='local linear trend / HadCRUT trend')),
            ('precip_slope', slopeP, dict(units='kg m-2 s-1 per K')),
            ('precip_sensitivity', s_frac, dict(units='1/K', long_name='precip slope / Pclim (fraction per K global)')),
            ('precip_r2', r2P, {}),
            ('precip_trend_frac', trend_P / clim_P / g_trend, dict(units='1/K', long_name='precip trend / Pclim / G trend')),
            ('T_clim', clim_T, dict(units='degC', long_name=f'RACMO {tvar} {CLIM_WINDOW[0]}-{CLIM_WINDOW[1]} climatology')),
            ('P_clim', clim_P, dict(units='kg m-2 s-1', long_name=f'RACMO pr {CLIM_WINDOW[0]}-{CLIM_WINDOW[1]} climatology'))]:
        ds[name] = xr.DataArray(np.where(fin[None], arr, np.nan).astype('float32'), dims=dims, coords=coords_t, attrs=attrs)
    ds['racmo_mask'] = xr.DataArray(fin.astype('int8'), dims=('y', 'x'))
    ds['region'] = xr.DataArray(regions, dims=('y', 'x'), attrs=dict(codes=' '.join(f'{i + 1}={r}' for i, r in enumerate(REGIONS))))
    ds.attrs.update(years=f'{years[0]}-{years[-1]}', G_ref=Gref, clim_window=list(CLIM_WINDOW), temp_var=tvar, regressor_label=regressor_label,
                    hadcrut_trend_K_per_yr=float(g_trend), min_years_unclipped=MIN_YEARS)
    enc = {v: dict(zlib=True, complevel=3) for v in ds.data_vars if ds[v].ndim == 3}
    ds.to_netcdf(out_dir / 'amplification.nc', encoding=enc)

    rs = xr.Dataset(coords=dict(year=years, month=np.arange(1, 13), region=reg_names, band=band_names))
    rs['ts'] = (('year', 'month', 'region'), series_T); rs['pr'] = (('year', 'month', 'region'), series_P)
    rs['ts_band'] = (('year', 'month', 'band'), band_T); rs['pr_band'] = (('year', 'month', 'band'), band_P)
    rs['G'] = (('year',), Gv)
    rs.to_netcdf(out_dir / 'regional_series.nc')

    yc = grid.y.values[:cy * COARSEN].reshape(cy, COARSEN).mean(1)
    xc = grid.x.values[:cx * COARSEN].reshape(cx, COARSEN).mean(1)
    st = xr.Dataset(coords=dict(year=years, month=np.arange(1, 13), y=yc, x=xc))
    st['ts'] = (('year', 'month', 'y', 'x'), stack_T); st['pr'] = (('year', 'month', 'y', 'x'), stack_P)
    st.to_netcdf(out_dir / 'stacks_5km.nc', encoding={'ts': dict(zlib=True, complevel=3), 'pr': dict(zlib=True, complevel=3)})
    return ds, rs


def analyse(ds, rs, grid, out_dir, alpha_config):
    fin = ds.racmo_mask.values > 0
    regions = ds.region.values
    years = rs.year.values; Gv = rs.G.values
    A_all, A_unc, A_eff = ds.A_all.values, ds.A_unclipped.values, ds.A_eff.values
    clip = ds.clipped_frac.values
    ann = lambda f: np.nanmean(f, axis=0)
    seas = lambda f, s: np.nanmean(f[SEASONS[s]], axis=0)

    # ---- regional / seasonal table from the regional series (fit on the means)
    rows = []
    for j, name in enumerate(rs.region.values):
        Tm = rs.ts.values[:, :, j]
        row = dict(region=name)
        for s, idx in [('annual', list(range(12)))] + list(SEASONS.items()):
            t = Tm[:, idx].mean(1)
            ok = np.isfinite(t)
            A = np.polyfit(Gv[ok], t[ok], 1)[0]
            tr = np.polyfit(years[ok], t[ok], 1)[0] / ds.attrs['hadcrut_trend_K_per_yr']
            row[f'A_{s}'] = A; row[f'trendratio_{s}'] = tr
            row[f'r_{s}'] = np.corrcoef(Gv[ok], t[ok])[0, 1]
        # split-sample skill: fit on first half, predict second half
        t = Tm.mean(1); h = len(years) // 2
        c = np.polyfit(Gv[:h], t[:h], 1)
        pred = np.polyval(c, Gv[h:])
        row['A_firsthalf'] = c[0]
        row['rmse_2ndhalf_pred'] = float(np.sqrt(np.mean((pred - t[h:]) ** 2)))
        row['rmse_2ndhalf_clim'] = float(np.sqrt(np.mean((t[:h].mean() - t[h:]) ** 2)))
        row['rmse_2ndhalf_alpha_config'] = float(np.sqrt(np.mean((t[:h].mean() + alpha_config * (Gv[h:] - Gv[:h].mean()) - t[h:]) ** 2)))
        Pm = rs.pr.values[:, :, j].mean(1)
        cw = (years >= CLIM_WINDOW[0]) & (years <= CLIM_WINDOW[1])
        row['P_sens_frac_per_K'] = np.polyfit(Gv, Pm, 1)[0] / Pm[cw].mean()
        row['P_clim_m_ice'] = float(pr_to_m_ice_per_yr(Pm[cw].mean()))
        rows.append(row)
    table = pd.DataFrame(rows).set_index('region')
    table.to_csv(out_dir / 'regional_amplification.csv', float_format='%.3f')
    print(table.round(2).to_string())

    # cell-wise areal means of the map products over the ice sheet
    sel = fin & (regions >= 1) & (regions <= 7)
    summary = dict(A_all_annual=float(np.nanmean(ann(A_all)[sel])),
                   A_eff_annual=float(np.nanmean(ann(A_eff)[sel])),
                   A_all_JJA=float(np.nanmean(seas(A_all, 'JJA')[sel])),
                   A_eff_JJA=float(np.nanmean(seas(A_eff, 'JJA')[sel])),
                   A_all_DJF=float(np.nanmean(seas(A_all, 'DJF')[sel])),
                   trend_ratio_annual=float(np.nanmean(ann(ds.trend_ratio_T.values)[sel])),
                   clipped_frac_JJA=float(np.nanmean(seas(clip, 'JJA')[sel])),
                   precip_sens_annual=float(np.nansum(ds.precip_slope.values.mean(0)[sel]) / np.nansum(ds.P_clim.values.mean(0)[sel])),
                   alpha_t2m_config=alpha_config)
    pd.Series(summary).to_csv(out_dir / 'summary.csv', float_format='%.3f')
    print(pd.Series(summary).round(3).to_string())

    # ---- figures
    fig, axs = plt.subplots(2, 4, figsize=(17, 11))
    _map(axs[0, 0], ann(A_all), grid, 'A annual (all years)', 0, 4, 'magma', fin, 'K/K')
    for ax, s in zip(axs[0, 1:], ['DJF', 'JJA', 'SON']):
        _map(ax, seas(A_all, s), grid, f'A {s} (all years)', 0, 4, 'magma', fin, 'K/K')
    _map(axs[1, 0], ann(A_eff), grid, 'A_eff annual (unclipped where identified)', 0, 4, 'magma', fin, 'K/K')
    _map(axs[1, 1], seas(A_eff, 'JJA'), grid, 'A_eff JJA', 0, 4, 'magma', fin, 'K/K')
    _map(axs[1, 2], seas(clip, 'JJA'), grid, 'fraction of years clipped, JJA', 0, 1, 'viridis', fin)
    _map(axs[1, 3], ann(ds.r2_all.values), grid, 'r2 annual-mean fit (all years)', 0, 0.6, 'viridis', fin)
    fig.suptitle(f'Local warming per degree of {ds.attrs.get("regressor_label", "HadCRUT5 global")} warming, RACMO2.3p2-ERA5 {ds.attrs.get("temp_var", "ts")} 1958-2025')
    fig.tight_layout(); fig.savefig(out_dir / 'amplification_maps.png', dpi=130); plt.close(fig)

    fig, axs = plt.subplots(3, 4, figsize=(17, 12))
    for m, ax in enumerate(axs.ravel()):
        _map(ax, A_eff[m], grid, f'A_eff month {m + 1}', 0, 5, 'magma', fin, 'K/K')
    fig.tight_layout(); fig.savefig(out_dir / 'amplification_monthly.png', dpi=110); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(16, 7))
    ps = ds.precip_sensitivity.values
    _map(axs[0], np.nanmean(ps, 0) * 100, grid, 'precip sensitivity, mean of monthly (%/K global)', -20, 20, 'BrBG', fin, '%/K')
    _map(axs[1], ds.precip_trend_frac.values.mean(0) * 100, grid, 'precip trend / Pclim / G trend (%/K)', -20, 20, 'BrBG', fin, '%/K')
    _map(axs[2], pr_to_m_ice_per_yr(ds.P_clim.values.mean(0)), grid, 'P_clim 1961-1990 (m ice/yr)', 0, 3, 'Blues', fin)
    fig.tight_layout(); fig.savefig(out_dir / 'precip_sensitivity_maps.png', dpi=130); plt.close(fig)

    # time series
    fig, axs = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    j = list(rs.region.values).index('GrIS')
    cw = (years >= CLIM_WINDOW[0]) & (years <= CLIM_WINDOW[1])
    t_ann = rs.ts.values[:, :, j].mean(1); t_ann -= t_ann[cw].mean()
    t_jja = rs.ts.values[:, SEASONS['JJA'], j].mean(1); t_jja -= t_jja[cw].mean()
    t_djf = rs.ts.values[:, SEASONS['DJF'], j].mean(1); t_djf -= t_djf[cw].mean()
    axs[0].plot(years, Gv, 'k-', label=ds.attrs.get('regressor_label', 'HadCRUT5 global'))
    axs[0].plot(years, t_ann, 'r-', label=f'GrIS annual t2m anomaly (RACMO {ds.attrs.get("temp_var", "ts")})')
    axs[0].plot(years, t_jja, 'orange', label='GrIS JJA'); axs[0].plot(years, t_djf, 'b-', alpha=0.6, label='GrIS DJF')
    axs[0].plot(years, table.loc['GrIS', 'A_annual'] * (Gv - Gv[cw].mean()), 'r--', label=f'A x G, A={table.loc["GrIS", "A_annual"]:.2f}')
    axs[0].plot(years, alpha_config * (Gv - Gv[cw].mean()), 'g:', label=f'config alpha_t2m={alpha_config}')
    axs[0].axhline(0, color='0.7'); axs[0].legend(fontsize=8, ncol=2); axs[0].set_ylabel('degC rel. 1961-1990'); axs[0].grid(alpha=0.3)
    p_ann = pr_to_m_ice_per_yr(rs.pr.values[:, :, j].mean(1))
    axs[1].plot(years, p_ann, 'b-', label='GrIS mean precip (RACMO)')
    c = np.polyfit(Gv, p_ann, 1); axs[1].plot(years, np.polyval(c, Gv), 'b--', label=f'{100 * c[0] / p_ann[cw].mean():.1f} %/K x G')
    axs[1].legend(fontsize=8); axs[1].set_ylabel('m ice/yr'); axs[1].grid(alpha=0.3); axs[1].set_xlabel('year')
    fig.tight_layout(); fig.savefig(out_dir / 'timeseries_gris.png', dpi=130); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    seas_cols = ['A_annual', 'A_DJF', 'A_MAM', 'A_JJA', 'A_SON']
    table.loc[REGIONS[:7] + ['GrIS'], seas_cols].plot.bar(ax=axs[0], width=0.8)
    axs[0].axhline(alpha_config, color='g', ls=':', label='config alpha_t2m'); axs[0].set_ylabel('K per K global'); axs[0].legend(fontsize=7)
    axs[0].set_title('amplification by region and season (fit on regional means)')
    bands = rs.band.values
    for s, col in [('annual', 'k'), ('DJF', 'b'), ('JJA', 'r')]:
        idx = list(range(12)) if s == 'annual' else SEASONS[s]
        A_b = [np.polyfit(Gv, rs.ts_band.values[:, idx, k].mean(1), 1)[0] for k in range(len(bands))]
        axs[1].plot(bands, A_b, f'{col}-o', label=s)
    axs[1].axhline(alpha_config, color='g', ls=':'); axs[1].set_xlabel('elevation band (m)'); axs[1].set_ylabel('K per K global')
    axs[1].legend(fontsize=8); axs[1].grid(alpha=0.3); axs[1].set_title('amplification by elevation band')
    fig.tight_layout(); fig.savefig(out_dir / 'amplification_regions_bands.png', dpi=130); plt.close(fig)
    return table, summary


def write_hybrid(ds, grid, years, domain_path, out_base, smooth_km):
    """Reconstructed yearly ts/pr files in the RACMO layout."""
    anom = xr.load_dataset(Path(domain_path) / 'model_inputs' / 'temperature_anomaly.nc')
    Gs = pd.Series(anom.temp_anomaly.values, index=anom.time.values.astype(int))
    fin = ds.racmo_mask.values > 0
    dx_km = abs(float(grid.x.values[1] - grid.x.values[0])) / 1e3
    sig = smooth_km / dx_km
    def smooth(f):
        if sig <= 0:
            return f
        w = gaussian_filter(fin.astype('float64'), sig)
        return np.stack([gaussian_filter(np.where(fin, f[m], 0.0), sig) / np.maximum(w, 1e-6) for m in range(12)])
    A = smooth(np.nan_to_num(ds.A_eff.values))
    s = smooth(np.nan_to_num(ds.precip_sensitivity.values))
    Tc, Pc = ds.T_clim.values, ds.P_clim.values
    Gref = ds.attrs['G_ref']
    tvar = ds.attrs.get('temp_var', 'ts')
    tmpl = xr.open_dataset(racmo_path(tvar, racmo_years(tvar)[0]))
    flip = tmpl.y.values[0] < tmpl.y.values[-1]
    for var in (tvar, 'pr'):
        (out_base / var).mkdir(parents=True, exist_ok=True)
    for y in years:
        if y not in Gs.index:
            print(f'  no anomaly for {y}; skipped'); continue
        dG = float(Gs[y]) - Gref
        T = Tc + A * dG
        if is_censored(ds.attrs.get('temp_var', 'ts')):
            T = np.minimum(T, 0.0)
        T = T + T_MELT
        P = Pc * np.maximum(1 + s * dG, 0.1)
        for var, arr, attrs in [(tvar, T, dict(units='K', long_name='hybrid: RACMO 1961-1990 clim + A_eff * global anomaly')),
                                ('pr', P, dict(units='kg m-2 s-1', long_name='hybrid: RACMO 1961-1990 clim * (1 + s * global anomaly)'))]:
            a = np.where(fin[None], arr, np.nan).astype('float32')
            if flip:
                a = a[:, ::-1, :]
            t = pd.to_datetime([f'{y}-{m:02d}-15' for m in range(1, 13)])
            o = xr.Dataset({var: (('time', 'y', 'x'), a, attrs), 'crs': tmpl['crs']},
                           coords=dict(time=t, y=tmpl.y, x=tmpl.x))
            o.attrs.update(title=f'{var} hybrid reconstruction for {y} (global anomaly {Gs[y]:.3f} C, dG {dG:.3f})',
                           source='analysis/arctic_amplification.py', smooth_km=smooth_km)
            o.to_netcdf(out_base / var / f'{var}_GrIS_RACMO2.3p2-ERA_hybrid_1000m_v1_{y}.nc',
                        encoding={var: dict(zlib=True, complevel=3)})
    print(f'wrote hybrid {years[0]}-{years[-1]} to {out_base}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain-path', default='domains/greenland')
    ap.add_argument('--out', default=None)
    ap.add_argument('--temp-var', default='tas', choices=['tas', 'ts'])
    ap.add_argument('--regressor', default='hadcrut', choices=['hadcrut', 'vinther'])
    ap.add_argument('--reuse', action='store_true', help='skip the accumulation pass; reuse amplification.nc')
    ap.add_argument('--hybrid-years', nargs=2, type=int, default=None, metavar=('Y0', 'Y1'))
    ap.add_argument('--hybrid-out', default=str(RACMO_BASE.parent / 'RACMO2.3p2-ERA-hybrid'))
    ap.add_argument('--smooth-km', type=float, default=10.0)
    ap.add_argument('--alpha-config', type=float, default=None, help='the config scalar alpha_t2m, for reference lines')
    a = ap.parse_args()
    suffix = '' if a.regressor == 'hadcrut' else f'_{a.regressor}'
    out_dir = Path(a.out or f'analysis/output/amplification_{a.temp_var}{suffix}'); out_dir.mkdir(parents=True, exist_ok=True)
    grid = load_grid(a.domain_path)
    years = racmo_years(a.temp_var)
    if a.regressor == 'hadcrut':
        G = load_hadcrut()
    else:
        anom = xr.load_dataset(Path(a.domain_path) / 'model_inputs' / 'temperature_anomaly.nc')
        if anom.attrs.get('source') != 'vinther_sw_greenland':
            raise SystemExit('temperature_anomaly.nc is not the Vinther series; run make_temperature_anomaly.py --source vinther')
        G = pd.Series(anom.temp_anomaly.values, index=anom.time.values.astype(int), name='G')
    years = [y for y in years if y in G.index]
    if a.reuse:
        ds = xr.load_dataset(out_dir / 'amplification.nc'); rs = xr.load_dataset(out_dir / 'regional_series.nc')
    else:
        print(f'accumulating {years[0]}-{years[-1]} ({len(years)} years)')
        ds, rs = accumulate(grid, years, G, out_dir, a.temp_var,
                            'HadCRUT5 global' if a.regressor == 'hadcrut' else 'Vinther SW Greenland JJA')
    alpha = a.alpha_config if a.alpha_config is not None else (2.0 if a.regressor == 'hadcrut' else 0.6)
    analyse(ds, rs, grid, out_dir, alpha)
    if a.hybrid_years:
        write_hybrid(ds, grid, list(range(a.hybrid_years[0], a.hybrid_years[1] + 1)), a.domain_path,
                     Path(a.hybrid_out), a.smooth_km)
