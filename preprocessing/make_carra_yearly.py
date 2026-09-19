"""
CARRA2 year by year, in the ISMIP7 kit's file layout -> the OCX forcing.

ISMIP7's Observationally Constrained eXperiment (OCX, 1990/1958-2025) asks
for reanalysis-derived forcing. The kit offers RACMO 2 m temperature; this
model is calibrated on CARRA2's 100 m temperature lapse-corrected onto the
DEM (make_carra_vars.py), so the consistent reanalysis forcing is CARRA2
itself, one year at a time, through the same code path as the climatology:

    python make_carra_yearly.py --domain-path ../domains/greenland [--years 1986 2025]

writes, per year, tas (K) and pr (kg m-2 s-1) files named and laid out like
the kit's (monthly, ISMIP grid, y ascending, `time` = mid-month days) under

    ../ismip7_data/CARRA2/ocx/carra2-1000m/{tas,pr}/

so that make_ismip7_forcing.py --gcm CARRA2 --scenario ocx --atm carra2-1000m
builds the catalogue and climatologies and forward_projection.py runs it
unchanged. The climatology of these files IS the model's calibration
climatology (up to the sampling of complete years), so in `raw` mode the run
sees exactly CARRA2's monthly fields plus the calibrated biases, with no
bias correction involved. There is no CARRA2 before 1985-10: the driver's
`--pre-record standalone` keeps forward_standalone's forcing (CARRA2
climatology + the Vinther anomaly index) for the years before the record.
"""
import argparse
from pathlib import Path

import numpy as np
import xarray as xr

import make_carra_vars as mcv

SECONDS_PER_YEAR = 31536000.0
ICE_DENSITY = 917.0
OUT_ROOT = Path('../ismip7_data/CARRA2/ocx/carra2-1000m')
MID_MONTH_DAYS = [14, 45, 73, 104, 134, 165, 195, 226, 257, 287, 318, 348]   # the kit's stamps
MONTH_BOUNDS = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]


def write_year(ds: xr.Dataset, year: int, out_root: Path, gcm='CARRA2', scenario='ocx', product='carra2-1000m'):
    y_desc = ds.y.values.astype('float64')
    flip = y_desc[0] > y_desc[-1]
    y = y_desc[::-1] if flip else y_desc
    fields = {
        'tas': (ds.monthly_t2m.values.astype('float64') + 273.15,
                dict(units='K', standard_name='air_temperature',
                     long_name='Air temperature at 100 m above ground lapse-corrected to the DEM (CARRA2)')),
        'pr': (ds.monthly_precip.values.astype('float64') * ICE_DENSITY / SECONDS_PER_YEAR,
               dict(units='kg m-2 s-1', standard_name='precipitation_flux', long_name='precipitation (CARRA2 tp)')),
    }
    paths = {}
    for var, (a, attrs) in fields.items():
        if flip:
            a = a[:, ::-1, :]
        out = xr.Dataset(coords={'time': ('time', np.array(MID_MONTH_DAYS, dtype='int64')),
                                 'y': ('y', y), 'x': ('x', ds.x.values.astype('float64'))})
        out['time'].attrs = dict(units=f'days since {year}-01-01', calendar='standard', standard_name='time', bounds='time_bnds')
        out['time_bnds'] = (('time', 'nv'), np.array([[MONTH_BOUNDS[k], MONTH_BOUNDS[k + 1]] for k in range(12)], dtype='float64'))
        out[var] = (('time', 'y', 'x'), a.astype('float32'))
        out[var].attrs = attrs
        out.attrs.update(source=ds.attrs.get('climate_source', 'CARRA2'), title=f'{var} on the ISMIP grid, CARRA2 {year}',
                         spatial_resolution='1000 m', comment='make_carra_yearly.py: CARRA2 through make_carra_vars.build_climate(years=[year])')
        d = out_root / var
        d.mkdir(parents=True, exist_ok=True)
        p = d / f'{var}_GrIS_{gcm}_{scenario}_{product}_v1_{year}.nc'
        out.to_netcdf(p, encoding={var: dict(zlib=True, complevel=4)})
        paths[var] = p
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--domain-path', required=True)
    ap.add_argument('--years', type=int, nargs=2, default=(1986, 2025))
    ap.add_argument('--out-root', default=str(OUT_ROOT))
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    out_root = Path(a.out_root)
    for year in range(a.years[0], a.years[1] + 1):
        target = out_root / 'pr' / f'pr_GrIS_CARRA2_ocx_carra2-1000m_v1_{year}.nc'
        if target.exists() and not a.overwrite:
            print(f'{year}: exists, skipped')
            continue
        ds = mcv.build_climate(a.domain_path, years=[year], write=False)
        n = ds.attrs.get('months_per_calendar_month')
        if n is not None and min(n) < 1:
            raise SystemExit(f'{year}: incomplete year in CARRA2 ({n})')
        paths = write_year(ds, year, out_root)
        print(f'{year}: wrote {paths["tas"].name}, {paths["pr"].name}', flush=True)


if __name__ == '__main__':
    main()
