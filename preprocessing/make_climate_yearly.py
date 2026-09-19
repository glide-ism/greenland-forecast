"""
Year-by-year forcing anomalies for the inverse -> model_inputs/gridded_climate_yearly.nc

    python make_climate_yearly.py --domain-path ../domains/greenland

For every year of the CARRA2 record (the per-year kit-layout files that
make_carra_yearly.py writes to ../ismip7_data/CARRA2/ocx/carra2-1000m/, or
`--from-build` to run make_carra_vars.build_climate(years=[y]) directly) the
departure of the monthly fields from the climatology the model holds in
GLIDE_inputs.nc:

    t2m_anom(year, month)     = t2m_year - monthly_t2m            (K, additive)
    precip_ratio(year, month) = (P_year + eps) / (monthly_precip + eps)   (multiplicative)

on the domain grid, stored as int16 codes (0.002 K; ratio 0.002, capped at
65.5 — the cap only bites where the climatological month is nearly dry) in
an UNCOMPRESSED NETCDF4 file chunked per (year, month): the inverse's
YearlyClimate loader reads a year's codes into pinned host memory (or
straight from the page cache) and decodes them on the GPU inside each SMB
checkpoint, so the file's layout is the working format, not an archive
(~9 GB for 40 years at 1 km). `config.yearly_climate_filename` switches the
inverse onto it; the record's mean anomaly over the climatology years is
reported (zero up to the sampling of complete years) so a mismatch between
the yearly files and the climatology shows up here, not in the inversion.
"""
import argparse
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr

KIT_ROOT = Path('../ismip7_data/CARRA2/ocx/carra2-1000m')
SECONDS_PER_YEAR = 31536000.0
ICE_DENSITY = 917.0
EPS_PRECIP = 1e-3            # m/yr; keeps the ratio finite over dry cells
T_SCALE, R_SCALE = 0.002, 0.002
R_MAX = 32767 * R_SCALE


def kit_year(root: Path, year: int):
    """(t2m degC, precip m ice/yr) of one year from the kit-layout files, on
    the domain's (y descending) grid."""
    out = []
    for var, conv in (('tas', lambda a: a - 273.15), ('pr', lambda a: a * SECONDS_PER_YEAR / ICE_DENSITY)):
        p = root / var / f'{var}_GrIS_CARRA2_ocx_carra2-1000m_v1_{year}.nc'
        with xr.open_dataset(p) as d:
            a = d[var].values.astype('float64')
            if d.y.values[0] < d.y.values[-1]:
                a = a[:, ::-1, :]
        out.append(conv(a))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--domain-path', required=True)
    ap.add_argument('--years', type=int, nargs=2, default=(1986, 2025))
    ap.add_argument('--kit-root', default=str(KIT_ROOT))
    ap.add_argument('--from-build', action='store_true', help='rebuild each year with make_carra_vars instead of the kit files')
    ap.add_argument('--out', default=None, help='default <domain>/model_inputs/gridded_climate_yearly.nc')
    a = ap.parse_args()
    dom = Path(a.domain_path)
    out = Path(a.out) if a.out else dom / 'model_inputs' / 'gridded_climate_yearly.nc'
    gi = xr.open_dataset(dom / 'model_inputs' / 'GLIDE_inputs.nc')
    t_clim = gi.monthly_t2m.values.astype('float64')
    p_clim = gi.monthly_precip.values.astype('float64')
    ice = gi.rgi_mask.values > 0.5
    years = list(range(a.years[0], a.years[1] + 1))
    ny, nx = t_clim.shape[1:]

    nc = netCDF4.Dataset(out, 'w', format='NETCDF4')
    nc.createDimension('year', len(years)); nc.createDimension('month', 12)
    nc.createDimension('y', ny); nc.createDimension('x', nx)
    yv = nc.createVariable('year', 'i4', ('year',)); yv[:] = years
    mv = nc.createVariable('month', 'i4', ('month',)); mv[:] = np.arange(1, 13)
    for d in ('y', 'x'):
        c = nc.createVariable(d, 'f8', (d,)); c[:] = gi[d].values; c.units = 'm'
    kw = dict(chunksizes=(1, 1, ny, nx), zlib=False, fill_value=np.int16(-32768))
    tv = nc.createVariable('t2m_anom', 'i2', ('year', 'month', 'y', 'x'), **kw)
    tv.scale_factor, tv.add_offset, tv.units = np.float32(T_SCALE), np.float32(0.0), 'K'
    tv.long_name = 'monthly air temperature minus the monthly climatology in GLIDE_inputs (monthly_t2m)'
    rv = nc.createVariable('precip_ratio', 'i2', ('year', 'month', 'y', 'x'), **kw)
    rv.scale_factor, rv.add_offset, rv.units = np.float32(R_SCALE), np.float32(0.0), '1'
    rv.long_name = f'monthly precipitation over the monthly climatology in GLIDE_inputs (monthly_precip), eps {EPS_PRECIP} m/yr, cap {R_MAX:.1f}'
    for v in (tv, rv):
        v.set_auto_maskandscale(False)
    nc.source = ('make_carra_vars.build_climate per year' if a.from_build else str(Path(a.kit_root).resolve()))
    nc.climatology = str((dom / 'model_inputs' / 'GLIDE_inputs.nc').resolve())
    nc.comment = 'preprocessing/make_climate_yearly.py; consumed by glacier_inverse.yearly_climate'

    sum_t = np.zeros_like(t_clim); sum_r = np.zeros_like(p_clim)
    n_cap = 0
    for k, year in enumerate(years):
        if a.from_build:
            import make_carra_vars as mcv
            ds = mcv.build_climate(str(dom), years=[year], write=False)
            t_y, p_y = ds.monthly_t2m.values.astype('float64'), ds.monthly_precip.values.astype('float64')
        else:
            t_y, p_y = kit_year(Path(a.kit_root), year)
        dt = t_y - t_clim
        r = (p_y + EPS_PRECIP) / (p_clim + EPS_PRECIP)
        n_cap += int(np.sum(r > R_MAX))
        r = np.minimum(r, R_MAX)
        tv[k] = np.round(dt / T_SCALE).astype(np.int16)
        rv[k] = np.round(r / R_SCALE).astype(np.int16)
        sum_t += dt; sum_r += r
        print(f'{year}: ice-mean dT annual {np.nanmean(dt.mean(0)[ice]):+.2f} K, JJA {np.nanmean(dt[5:8].mean(0)[ice]):+.2f} K, '
              f'precip ratio {np.nanmean(p_y.mean(0)[ice]) / np.nanmean(p_clim.mean(0)[ice]):.3f}', flush=True)
        nc.sync()
    mean_t = sum_t / len(years); mean_r = sum_r / len(years)
    nc.record_mean_t2m_anom_ice_K = float(np.nanmean(mean_t[:, ice]))
    nc.record_mean_precip_ratio_ice = float(np.nanmean(mean_r[:, ice]))
    nc.n_ratio_capped = n_cap
    nc.close()
    print(f'wrote {out} ({out.stat().st_size / 1e9:.1f} GB): record-mean anomaly over the ice '
          f'{np.nanmean(mean_t[:, ice]):+.4f} K (max |monthly cell mean| {np.nanmax(np.abs(mean_t[:, ice])):.3f}), '
          f'mean precip ratio {np.nanmean(mean_r[:, ice]):.4f}; {n_cap} cell-months capped at ratio {R_MAX:.1f}')


if __name__ == '__main__':
    main()
