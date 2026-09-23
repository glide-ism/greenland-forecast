"""RACMO2.3p2-ERA5 (ISMIP7 kit, 1 km) as the inverse's climate forcing.

The calibration's forcing is CARRA2 (make_carra_vars.py: 100 m temperature
moved onto the DEM, CARRA tp). This builds the same two inputs from the
kit's RACMO product so the reanalysis can be swapped by FILENAME in the
domain config, with nothing else changed:

    model_inputs/gridded_climate_racmo.nc   monthly_t2m (degC), monthly_precip
                                            (m ice/yr): calendar-month
                                            climatology over --clim-years
                                            (default 1986-2025, the CARRA2
                                            window, so the Vinther index keeps
                                            its zero-mean reference)
    model_inputs/GLIDE_inputs_racmo.nc      GLIDE_inputs.nc with those two
                                            variables replaced (everything
                                            else, incl. the CARRA lapse-rate
                                            diagnostics, untouched)
    -> config: gridded_filename="GLIDE_inputs_racmo.nc"

and make_climate_yearly.py --source racmo writes the year-by-year anomalies
relative to THAT climatology (1958-2025, 68 years).

Source: common_data/climate/RACMO2.3p2-ERA/{tas,pr}/<var>_GrIS_RACMO2.3p2-ERA_
OCX_SDBN1-1000m_v1_<year>.nc, monthly, K and kg m-2 s-1, on the ISMIP 1 km
grid = the domain grid with y ascending (flipped here; coordinates asserted).
`tas` is the 2 m temperature (NOT the 100 m level the CARRA2 forcing uses:
over melting ice it is pinned near 0 degC, so H_atm and tbias will
recalibrate — that is part of what the experiment measures).

Footprint: RACMO covers 100 % of the main ice sheet but 66 % of the
peripheral-glacier ice and 58 % of the domain mask. Outside it the
climatology is CARRA2's field plus the RACMO - CARRA2 offset (precip: ratio)
of the NEAREST covered cell, which keeps CARRA2's DEM-resolved structure and
is continuous at the footprint edge; `racmo_fill_distance` (km) records how
far that was. Yearly anomalies outside the footprint are the nearest covered
cell's (make_climate_yearly.py).
"""
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import ndimage

RACMO_ROOT = Path('../common_data/climate/RACMO2.3p2-ERA')
PATTERN = '{var}_GrIS_RACMO2.3p2-ERA_OCX_SDBN1-1000m_v1_{year}.nc'
SECONDS_PER_YEAR = 31536000.0
ICE_DENSITY = 917.0
EPS_PRECIP = 1e-3            # m/yr, as in make_climate_yearly.py


def racmo_year(root: Path, year: int, x, y):
    """(t2m degC, precip m ice/yr), (12, ny, nx) on the domain grid, NaN
    outside RACMO's footprint."""
    out = []
    for var, conv in (('tas', lambda a: a - 273.15), ('pr', lambda a: a * SECONDS_PER_YEAR / ICE_DENSITY)):
        with xr.open_dataset(Path(root) / var / PATTERN.format(var=var, year=year)) as d:
            a = d[var].values.astype('float32')
            yy = d.y.values
            if yy[0] < yy[-1]:
                a, yy = a[:, ::-1, :], yy[::-1]
            if a.shape[0] != 12 or not (np.allclose(yy, y) and np.allclose(d.x.values, x)):
                raise ValueError(f'RACMO {var} {year}: not 12 months on the domain grid')
        out.append(conv(a))
    return out


def footprint_fill_index(covered: np.ndarray):
    """Indices of the nearest covered cell for every cell, and the distance (cells)."""
    dist, (iy, ix) = ndimage.distance_transform_edt(~covered, return_indices=True)
    return iy, ix, dist


def build_racmo_climate(domain_path: str, root=None, clim_years=(1986, 2025)) -> xr.Dataset:
    dom = Path(domain_path); inputs = dom / 'model_inputs'
    root = Path(root) if root else RACMO_ROOT
    gi = xr.load_dataset(inputs / 'GLIDE_inputs.nc')
    x, y = gi.x.values.astype('float64'), gi.y.values.astype('float64')
    dx = float(abs(x[1] - x[0]))
    years = list(range(clim_years[0], clim_years[1] + 1))
    st = sp = None
    for yr in years:
        t, p = racmo_year(root, yr, x, y)
        st = t.astype('float64') if st is None else st + t
        sp = p.astype('float64') if sp is None else sp + p
        print(f'  {yr}', end='', flush=True)
    print()
    t_r, p_r = st / len(years), sp / len(years)
    covered = np.isfinite(t_r[6]) & np.isfinite(p_r[6])
    iy, ix, dist = footprint_fill_index(covered)

    t_c = gi.monthly_t2m.values.astype('float64'); p_c = gi.monthly_precip.values.astype('float64')
    dT = np.where(covered, t_r - t_c, np.nan)                       # RACMO - CARRA2 on the footprint
    ratio = np.where(covered, (p_r + EPS_PRECIP) / (p_c + EPS_PRECIP), np.nan)
    t_out = np.where(covered, t_r, t_c + dT[:, iy, ix])
    p_out = np.where(covered, p_r, p_c * ratio[:, iy, ix])

    ice = gi.rgi_mask.values > 0.5
    per = gi.rgi_periphery_fraction.values > 0.5 if 'rgi_periphery_fraction' in gi else np.zeros_like(ice)
    main = ice & ~per
    area = lambda m: m.sum() * dx * dx
    gt = lambda f, m: float(f.mean(0)[m].sum() * dx * dx * ICE_DENSITY / 1e12)
    print(f'RACMO footprint: {covered.mean() * 100:.0f} % of the grid, {covered[main].mean() * 100:.1f} % of the main sheet, '
          f'{covered[ice & per].mean() * 100:.0f} % of peripheral ice, {covered[gi.domain_mask.values > 0.5].mean() * 100:.0f} % of the domain mask')
    print(f'main-sheet climatology {years[0]}-{years[-1]}: t2m annual {t_out.mean(0)[main].mean():+.2f} (CARRA2 T100 {t_c.mean(0)[main].mean():+.2f}) degC, '
          f'JJA {t_out[5:8].mean(0)[main].mean():+.2f} ({t_c[5:8].mean(0)[main].mean():+.2f}); '
          f'precip {gt(p_out, main):.0f} (CARRA2 {gt(p_c, main):.0f}) Gt/yr')
    for lo, hi in ((0, 1200), (1200, 2000), (2000, 4000)):
        m = main & (gi.elevation.values >= lo) & (gi.elevation.values < hi)
        print(f'   {lo}-{hi} m: JJA {t_out[5:8].mean(0)[m].mean():+.2f} vs {t_c[5:8].mean(0)[m].mean():+.2f} degC, '
              f'precip {p_out.mean(0)[m].mean():.3f} vs {p_c.mean(0)[m].mean():.3f} m/yr (x{p_out.mean(0)[m].mean() / p_c.mean(0)[m].mean():.2f})')

    src = f'RACMO2.3p2-ERA5 (ISMIP7 kit SDBN1-1000m), calendar-month climatology {years[0]}-{years[-1]}'
    clim = xr.Dataset(coords={'t': gi.t if 't' in gi.coords else np.arange(12), 'y': gi.y, 'x': gi.x})
    clim['monthly_t2m'] = (('t', 'y', 'x'), t_out.astype('float32'))
    clim['monthly_t2m'].attrs = dict(units='Deg C', long_name='Monthly 2 m air temperature (RACMO2.3p2-ERA5)', source=src)
    clim['monthly_precip'] = (('t', 'y', 'x'), p_out.astype('float32'))
    clim['monthly_precip'].attrs = dict(units='m ice equivalent / yr', long_name='Precipitation rate (RACMO2.3p2-ERA5) at monthly time steps', source=src)
    clim['racmo_fill_distance'] = (('y', 'x'), (dist * dx / 1e3).astype('float32'))
    clim['racmo_fill_distance'].attrs = dict(units='km', long_name='distance to the RACMO footprint (0 inside); beyond it: CARRA2 + the nearest '
                                             'covered cell\'s RACMO - CARRA2 offset (precip ratio)')
    clim.attrs.update(climate_source=src, climatology_years=f'{years[0]}-{years[-1]}', forcing_level='2 m')
    if 'spatial_ref' in gi:
        clim['spatial_ref'] = gi['spatial_ref']
    clim.to_netcdf(inputs / 'gridded_climate_racmo.nc')

    merged = gi.copy()
    for v in ('monthly_t2m', 'monthly_precip'):
        merged[v] = (gi[v].dims, clim[v].values)
        merged[v].attrs = dict(clim[v].attrs)
    merged['racmo_fill_distance'] = clim['racmo_fill_distance']
    merged.attrs['climate_source'] = src
    out = inputs / 'GLIDE_inputs_racmo.nc'
    merged.to_netcdf(out)
    print(f'wrote {inputs / "gridded_climate_racmo.nc"} and {out}')
    return clim


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--domain-path', required=True)
    ap.add_argument('--racmo-root', default=str(RACMO_ROOT))
    ap.add_argument('--clim-years', type=int, nargs=2, default=(1986, 2025))
    a = ap.parse_args()
    build_racmo_climate(a.domain_path, a.racmo_root, tuple(a.clim_years))
