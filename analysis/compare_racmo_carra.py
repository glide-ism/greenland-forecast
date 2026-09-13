"""Compare the RACMO2.3p2-ERA5 1 km forcing with the CARRA2 height-level
forcing the model uses, for one calendar year in which both exist.

Temperature: CARRA2 100 m-above-ground air temperature lapse-corrected onto
the DEM (exactly what `make_carra_vars.py` feeds the enthalpy SMB model)
versus RACMO's 2 m air temperature (`tas`; or the 0 degC-clipped `ts` with
--temp-var ts). The CARRA2 field is the "free" air above the boundary
layer, so it is EXPECTED to be warmer than RACMO t2m where a surface
inversion exists (winter interior) and where the melting surface holds the
2 m air near 0 degC (summer margin). With `ts` the comparison is also made
on cells where RACMO is not clipped, and with CARRA2 clipped at 0 degC.

Precipitation: CARRA2 tp (mm/day, median-filtered as in the model) versus
RACMO pr, as m ice eq./yr and as regional totals in Gt/yr.

Usage:
    python analysis/compare_racmo_carra.py --domain-path domains/greenland --year 2000 [--temp-var tas|ts]
Writes figures and a summary table to analysis/output/compare_{year}_{var}/.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'preprocessing'))
from racmo_common import (ELEV_BANDS, REGIONS, band_means, extent, is_censored, load_grid, load_racmo,
                          pr_to_m_ice_per_yr, region_means, region_raster)
import make_carra_vars as mcv

MONTHS = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']


def _map(ax, field, grid, title, vmin=None, vmax=None, cmap='RdBu_r', ice=None, label=''):
    f = np.where(ice, field, np.nan) if ice is not None else field
    im = ax.imshow(f, extent=extent(grid), vmin=vmin, vmax=vmax, cmap=cmap, interpolation='nearest')
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=label)


def main(domain_path, year, out_dir, tvar='tas'):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    grid = load_grid(domain_path)
    ice = grid.rgi_mask.values > 0.5
    z = grid.elevation.values
    regions = region_raster(grid)

    # ---- CARRA2 for this year only, through the production code path
    carra = mcv.build_climate(domain_path, years=[year], write=False,
                              carra_base=HERE.parent / 'common_data' / 'climate' / 'carra2')
    t_c = carra.monthly_t2m.values                      # degC, (12, ny, nx)
    p_c = carra.monthly_precip.values                   # m ice / yr
    lapse = carra.monthly_lapse_rate.values

    # ---- RACMO
    t_r = load_racmo(tvar, year, grid)                  # degC (ts: clipped at 0)
    censored = is_censored(tvar)
    rlabel = 'RACMO t2m (clipped)' if censored else 'RACMO t2m'
    rtitle = 'RACMO2.3p2-ERA5 (2 m, clipped at 0 C)' if censored else 'RACMO2.3p2-ERA5 (2 m)'
    p_r = pr_to_m_ice_per_yr(load_racmo('pr', year, grid))
    have = ice & np.isfinite(t_r[0]) & np.isfinite(t_c[0])
    print(f'ice cells {ice.sum()}, with both products {have.sum()} '
          f'({100 * have.sum() / ice.sum():.1f}% of ice; RACMO covers '
          f'{100 * np.isfinite(t_r[0])[ice].mean():.1f}% of ice cells)')

    unclipped = (t_r < -0.01) if censored else np.isfinite(t_r)
    t_c_clip = np.minimum(t_c, 0.0)
    rows = []
    for m in range(12):
        sel = have
        selu = have & unclipped[m]
        d = t_c[m] - t_r[m]
        dc = t_c_clip[m] - t_r[m]
        rows.append(dict(month=m + 1,
                         clipped_frac=float(1 - unclipped[m][have].mean()),
                         bias_all=float(np.mean(d[sel])), rms_all=float(np.sqrt(np.mean(d[sel] ** 2))),
                         bias_unclipped=float(np.mean(d[selu])), rms_unclipped=float(np.sqrt(np.mean(d[selu] ** 2))),
                         bias_carra_clipped=float(np.mean(dc[sel])), rms_carra_clipped=float(np.sqrt(np.mean(dc[sel] ** 2))),
                         lapse_median_K_per_km=float(np.median(lapse[m][have]) * 1e3),
                         precip_ratio_carra_over_racmo=float(np.sum(p_c[m][sel]) / np.sum(p_r[m][sel]))))
    monthly = pd.DataFrame(rows).set_index('month')
    monthly.to_csv(out_dir / 'monthly_temperature_stats.csv', float_format='%.3f')
    print(monthly.round(2).to_string())

    # annual / seasonal maps
    t_c_ann, t_r_ann = t_c.mean(0), t_r.mean(0)
    d_ann = t_c_ann - t_r_ann
    jja = slice(5, 8); djf = [0, 1, 11]
    d_jja = t_c[jja].mean(0) - t_r[jja].mean(0)
    d_djf = t_c[djf].mean(0) - t_r[djf].mean(0)
    d_jja_clip = t_c_clip[jja].mean(0) - t_r[jja].mean(0)
    p_c_ann, p_r_ann = p_c.mean(0), p_r.mean(0)

    # regional + elevation-band tables
    cell_area_km2 = (float(abs(grid.x.values[1] - grid.x.values[0])) / 1e3) ** 2
    reg_rows = []
    for i, name in enumerate(REGIONS + ['GrIS', 'all'], start=1):
        if name == 'GrIS':
            sel = have & (regions >= 1) & (regions <= 7)
        elif name == 'all':
            sel = have & (regions >= 1)
        else:
            sel = have & (regions == i)
        if not sel.any():
            continue
        gt_c = np.sum(p_c_ann[sel]) * 917 * cell_area_km2 * 1e6 / 1e12
        gt_r = np.sum(p_r_ann[sel]) * 917 * cell_area_km2 * 1e6 / 1e12
        reg_rows.append(dict(region=name, n_cells=int(sel.sum()),
                             T_racmo_ann=float(t_r_ann[sel].mean()), T_carra_ann=float(t_c_ann[sel].mean()),
                             dT_ann=float(d_ann[sel].mean()), dT_djf=float(d_djf[sel].mean()),
                             dT_jja=float(d_jja[sel].mean()), dT_jja_carra_clipped=float(d_jja_clip[sel].mean()),
                             P_racmo_Gt=gt_r, P_carra_Gt=gt_c, P_ratio=gt_c / gt_r,
                             P_racmo_m_ice=float(p_r_ann[sel].mean()), P_carra_m_ice=float(p_c_ann[sel].mean())))
    regional = pd.DataFrame(reg_rows).set_index('region')
    regional.to_csv(out_dir / 'regional_stats.csv', float_format='%.3f')
    print(regional.round(2).to_string())

    band_rows = []
    for lo, hi in ELEV_BANDS:
        sel = have & (z >= lo) & (z < hi)
        band_rows.append(dict(band=f'{lo}-{hi}', n=int(sel.sum()),
                              dT_ann=float(d_ann[sel].mean()), dT_djf=float(d_djf[sel].mean()),
                              dT_jja=float(d_jja[sel].mean()), dT_jja_carra_clipped=float(d_jja_clip[sel].mean()),
                              clipped_frac_jul=float(1 - unclipped[6][sel].mean()),
                              P_ratio=float(p_c_ann[sel].sum() / p_r_ann[sel].sum()),
                              P_racmo_m_ice=float(p_r_ann[sel].mean()), P_carra_m_ice=float(p_c_ann[sel].mean())))
    bands = pd.DataFrame(band_rows).set_index('band')
    bands.to_csv(out_dir / 'elevation_band_stats.csv', float_format='%.3f')
    print(bands.round(2).to_string())

    # ---- figures
    fig, axs = plt.subplots(2, 3, figsize=(13, 12))
    _map(axs[0, 0], t_r_ann, grid, f'{rlabel} annual mean {year}', -30, 5, 'viridis', have, 'degC')
    _map(axs[0, 1], t_c_ann, grid, f'CARRA2 T100 on DEM annual mean {year}', -30, 5, 'viridis', have, 'degC')
    _map(axs[0, 2], d_ann, grid, 'CARRA2 - RACMO, annual', -6, 6, 'RdBu_r', have, 'K')
    _map(axs[1, 0], d_djf, grid, 'CARRA2 - RACMO, DJF', -8, 8, 'RdBu_r', have, 'K')
    _map(axs[1, 1], d_jja, grid, 'CARRA2 - RACMO, JJA', -8, 8, 'RdBu_r', have, 'K')
    _map(axs[1, 2], d_jja_clip, grid, 'min(CARRA2, 0) - RACMO, JJA', -8, 8, 'RdBu_r', have, 'K')
    fig.suptitle(f'Temperature: CARRA2 (100 m, lapse-corrected) vs {rtitle}, {year}')
    fig.tight_layout(); fig.savefig(out_dir / 'temperature_maps.png', dpi=130); plt.close(fig)

    fig, axs = plt.subplots(2, 4, figsize=(16, 8), sharex=True)
    for ax, name in zip(axs.ravel(), REGIONS):
        i = REGIONS.index(name) + 1
        sel = have & (regions == i)
        tr = [t_r[m][sel].mean() for m in range(12)]
        tc = [t_c[m][sel].mean() for m in range(12)]
        tcu = [t_c[m][sel & unclipped[m]].mean() for m in range(12)]
        tru = [t_r[m][sel & unclipped[m]].mean() for m in range(12)]
        ax.plot(range(1, 13), tr, 'k-o', ms=3, label=rlabel)
        ax.plot(range(1, 13), tc, 'r-o', ms=3, label='CARRA2 T100 on DEM')
        ax.plot(range(1, 13), np.array(tc) - np.array(tr), 'b--', label='difference')
        ax.plot(range(1, 13), np.array(tcu) - np.array(tru), 'c:', label='difference, unclipped cells')
        ax.axhline(0, color='0.7', lw=0.5)
        ax.set_title(f'{name} (n={sel.sum()})', fontsize=9); ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTHS)
        ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=7); axs[0, 0].set_ylabel('degC'); axs[1, 0].set_ylabel('degC')
    fig.suptitle(f'Seasonal cycle over ice by region, {year}')
    fig.tight_layout(); fig.savefig(out_dir / 'temperature_seasonal_by_region.png', dpi=130); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 7))
    _map(axs[0], p_r_ann, grid, f'RACMO precip {year}', 0, 3, 'Blues', have, 'm ice/yr')
    _map(axs[1], p_c_ann, grid, f'CARRA2 precip {year}', 0, 3, 'Blues', have, 'm ice/yr')
    _map(axs[2], np.log2(p_c_ann / p_r_ann), grid, 'log2(CARRA2 / RACMO)', -1.5, 1.5, 'RdBu_r', have, 'log2 ratio')
    fig.tight_layout(); fig.savefig(out_dir / 'precip_maps.png', dpi=130); plt.close(fig)

    fig, axs = plt.subplots(2, 4, figsize=(16, 8), sharex=True)
    for ax, name in zip(axs.ravel(), REGIONS):
        i = REGIONS.index(name) + 1
        sel = have & (regions == i)
        pr_ = [p_r[m][sel].mean() for m in range(12)]
        pc_ = [p_c[m][sel].mean() for m in range(12)]
        ax.plot(range(1, 13), pr_, 'k-o', ms=3, label='RACMO')
        ax.plot(range(1, 13), pc_, 'r-o', ms=3, label='CARRA2')
        ax.set_title(f'{name}: {regional.loc[name, "P_racmo_Gt"]:.0f} vs {regional.loc[name, "P_carra_Gt"]:.0f} Gt/yr', fontsize=9)
        ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTHS); ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=8); axs[0, 0].set_ylabel('m ice/yr'); axs[1, 0].set_ylabel('m ice/yr')
    fig.suptitle(f'Precipitation seasonal cycle over ice by region, {year} (titles: annual totals RACMO vs CARRA2)')
    fig.tight_layout(); fig.savefig(out_dir / 'precip_seasonal_by_region.png', dpi=130); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    axs[0].plot(bands.index, bands.dT_djf, 'b-o', label='DJF'); axs[0].plot(bands.index, bands.dT_jja, 'r-o', label='JJA')
    axs[0].plot(bands.index, bands.dT_jja_carra_clipped, 'r--o', label='JJA, CARRA clipped at 0')
    axs[0].plot(bands.index, bands.dT_ann, 'k-o', label='annual'); axs[0].axhline(0, color='0.7')
    axs[0].set_ylabel('CARRA2 - RACMO (K)'); axs[0].set_xlabel('elevation band (m)'); axs[0].legend(fontsize=8); axs[0].grid(alpha=0.3)
    axs[1].plot(bands.index, bands.P_racmo_m_ice, 'k-o', label='RACMO'); axs[1].plot(bands.index, bands.P_carra_m_ice, 'r-o', label='CARRA2')
    axs[1].set_ylabel('precip (m ice/yr)'); axs[1].set_xlabel('elevation band (m)'); axs[1].legend(fontsize=8); axs[1].grid(alpha=0.3)
    axs[2].plot(monthly.index, monthly.bias_all, 'k-o', label='bias, all ice')
    axs[2].plot(monthly.index, monthly.bias_unclipped, 'b-o', label='bias, RACMO unclipped cells')
    axs[2].plot(monthly.index, monthly.bias_carra_clipped, 'g--o', label='bias, CARRA clipped at 0')
    axs[2].plot(monthly.index, monthly.rms_unclipped, 'b:', label='rms, unclipped')
    axs[2].axhline(0, color='0.7'); axs[2].set_xticks(range(1, 13)); axs[2].set_xticklabels(MONTHS)
    axs[2].set_ylabel('K'); axs[2].legend(fontsize=8); axs[2].grid(alpha=0.3)
    ax2 = axs[2].twinx(); ax2.bar(monthly.index, monthly.clipped_frac, color='0.85', zorder=0, width=0.6)
    ax2.set_ylabel('fraction of ice cells clipped (RACMO)', color='0.5'); ax2.set_ylim(0, 1)
    axs[2].set_zorder(1); axs[2].patch.set_visible(False)
    fig.tight_layout(); fig.savefig(out_dir / 'profiles_and_monthly.png', dpi=130); plt.close(fig)

    # scatter: monthly cell values (subsample)
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(have); idx = rng.choice(idx, min(200000, idx.size), replace=False)
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    for m, c in [(0, 'b'), (6, 'r')]:
        axs[0].hexbin(t_r[m].ravel()[idx], t_c[m].ravel()[idx], gridsize=80, bins='log', cmap='Blues' if m == 0 else 'Reds', alpha=0.7)
    axs[0].plot([-45, 10], [-45, 10], 'k--', lw=0.8); axs[0].set_xlabel(f'{rlabel} (degC)'); axs[0].set_ylabel('CARRA2 T100 on DEM (degC)')
    axs[0].set_title('January (blue) and July (red), ice cells')
    axs[1].hexbin(np.log10(p_r_ann.ravel()[idx]), np.log10(p_c_ann.ravel()[idx]), gridsize=80, bins='log', cmap='Greys')
    axs[1].plot([-1.5, 1], [-1.5, 1], 'k--', lw=0.8); axs[1].set_xlabel('log10 RACMO precip (m ice/yr)'); axs[1].set_ylabel('log10 CARRA2 precip')
    axs[1].set_title('annual precipitation')
    fig.tight_layout(); fig.savefig(out_dir / 'scatter.png', dpi=130); plt.close(fig)

    # coverage map: where RACMO is missing over ice
    fig, ax = plt.subplots(figsize=(6, 8))
    cov = np.where(ice, np.where(np.isfinite(t_r[0]), 1.0, 0.0), np.nan)
    _map(ax, cov, grid, 'RACMO coverage of ice cells (1 = present)', 0, 1, 'coolwarm')
    fig.tight_layout(); fig.savefig(out_dir / 'racmo_coverage.png', dpi=130); plt.close(fig)
    print(f'wrote figures and tables to {out_dir}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain-path', default='domains/greenland')
    ap.add_argument('--year', type=int, default=2000)
    ap.add_argument('--out', default=None)
    ap.add_argument('--temp-var', default='tas', choices=['tas', 'ts'])
    a = ap.parse_args()
    main(a.domain_path, a.year, a.out or f'analysis/output/compare_{a.year}_{a.temp_var}', a.temp_var)
