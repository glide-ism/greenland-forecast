"""Hybrid forcing: CARRA2 temperature + RACMO precipitation.

    python make_climate_hybrid.py --domain-path ../domains/greenland

The controlled half of the reanalysis-swap experiment. inverse_v4's SMB
excess over Mankoff is accumulation above 1200 m, and RACMO carries
110 Gt/yr less precipitation than CARRA2 there; swapping the whole forcing
to RACMO also swaps the 100 m temperature for a 2 m one (pinned near 0 degC
over melting ice), which confounds the comparison. Here ONLY the
precipitation changes: the melt physics, the temperature level and glare's
rain/snow split stay on the calibration's CARRA2 field.

    model_inputs/GLIDE_inputs_hybrid.nc          GLIDE_inputs.nc with
                                                 monthly_precip taken from
                                                 GLIDE_inputs_racmo.nc
    model_inputs/gridded_climate_yearly_hybrid.nc
                                                 t2m_anom codes from
                                                 gridded_climate_yearly.nc
                                                 (CARRA2), precip_ratio codes
                                                 from gridded_climate_yearly_
                                                 racmo.nc, the years both have

Both yearly files hold int16 codes relative to THEIR OWN 1986-2025
climatology with the same scale factors, so the hybrid is a copy of codes,
no recomputation. Needs make_racmo_vars.py and make_climate_yearly.py
(--source carra2 and --source racmo) first. Both products are ERA5-bounded,
so their year-to-year variability is shared: the correlation of the
ice-mean annual precipitation ratios is printed as the check on that.
"""
import argparse
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--domain-path', required=True)
    a = ap.parse_args()
    inputs = Path(a.domain_path) / 'model_inputs'

    gi = xr.load_dataset(inputs / 'GLIDE_inputs.nc')
    with xr.open_dataset(inputs / 'GLIDE_inputs_racmo.nc') as gr:
        p_racmo = gr.monthly_precip.values
        p_attrs = dict(gr.monthly_precip.attrs)
        fill = gr.racmo_fill_distance.load()
    merged = gi.copy()
    merged['monthly_precip'] = (gi.monthly_precip.dims, p_racmo)
    merged['monthly_precip'].attrs = p_attrs
    merged['racmo_fill_distance'] = fill
    merged.attrs['climate_source'] = ('hybrid: CARRA2 100 m temperature on the DEM (monthly_t2m) + RACMO2.3p2-ERA5 '
                                      'precipitation (monthly_precip), climatologies 1986-2025')
    merged.to_netcdf(inputs / 'GLIDE_inputs_hybrid.nc')
    ice = gi.rgi_mask.values > 0.5
    print(f'wrote {inputs / "GLIDE_inputs_hybrid.nc"}')

    ct = netCDF4.Dataset(inputs / 'gridded_climate_yearly.nc')
    rp = netCDF4.Dataset(inputs / 'gridded_climate_yearly_racmo.nc')
    for d in (ct, rp):
        for v in ('t2m_anom', 'precip_ratio'):
            d[v].set_auto_maskandscale(False)
    for v in ('t2m_anom', 'precip_ratio'):
        if abs(float(ct[v].scale_factor) - float(rp[v].scale_factor)) > 0:
            raise SystemExit(f'{v}: the two yearly files use different scale factors')
    yc, yr = [int(y) for y in ct['year'][:]], [int(y) for y in rp['year'][:]]
    years = [y for y in yc if y in yr]
    ny, nx = ct.dimensions['y'].size, ct.dimensions['x'].size

    out = inputs / 'gridded_climate_yearly_hybrid.nc'
    nc = netCDF4.Dataset(out, 'w', format='NETCDF4')
    nc.createDimension('year', len(years)); nc.createDimension('month', 12)
    nc.createDimension('y', ny); nc.createDimension('x', nx)
    nc.createVariable('year', 'i4', ('year',))[:] = years
    nc.createVariable('month', 'i4', ('month',))[:] = np.arange(1, 13)
    for d in ('y', 'x'):
        c = nc.createVariable(d, 'f8', (d,)); c[:] = ct[d][:]; c.units = 'm'
    kw = dict(chunksizes=(1, 1, ny, nx), zlib=False, fill_value=np.int16(-32768))
    var = {}
    for v, src in (('t2m_anom', ct), ('precip_ratio', rp)):
        var[v] = nc.createVariable(v, 'i2', ('year', 'month', 'y', 'x'), **kw)
        for k in src[v].ncattrs():
            if k != '_FillValue':
                var[v].setncattr(k, src[v].getncattr(k))
        var[v].set_auto_maskandscale(False)
    nc.source = 't2m_anom: gridded_climate_yearly.nc (CARRA2); precip_ratio: gridded_climate_yearly_racmo.nc (RACMO2.3p2-ERA5)'
    nc.climatology = str((inputs / 'GLIDE_inputs_hybrid.nc').resolve())
    nc.comment = 'preprocessing/make_climate_hybrid.py; consumed by glacier_inverse.yearly_climate'

    sc = float(ct['precip_ratio'].scale_factor)
    r_c, r_r = [], []
    for k, y in enumerate(years):
        var['t2m_anom'][k] = ct['t2m_anom'][yc.index(y)]
        codes_r = rp['precip_ratio'][yr.index(y)]
        var['precip_ratio'][k] = codes_r
        r_r.append(float(codes_r[:, ice].mean()) * sc)
        r_c.append(float(ct['precip_ratio'][yc.index(y)][:, ice].mean()) * sc)
        nc.sync()
        print(f'  {y}', end='', flush=True)
    print()
    nc.close(); ct.close(); rp.close()
    r_c, r_r = np.array(r_c), np.array(r_r)
    print(f'wrote {out} ({out.stat().st_size / 1e9:.1f} GB), years {years[0]}-{years[-1]}')
    print(f'ice-mean annual precip ratio, CARRA2 vs RACMO over {len(years)} years: r = {np.corrcoef(r_c, r_r)[0, 1]:.2f}, '
          f'std {r_c.std():.3f} vs {r_r.std():.3f}; wettest CARRA2 years {[years[i] for i in np.argsort(r_c)[-3:]]}, '
          f'RACMO {[years[i] for i in np.argsort(r_r)[-3:]]}')


if __name__ == '__main__':
    main()
