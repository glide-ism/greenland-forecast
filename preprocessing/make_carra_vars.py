"""Build the gridded monthly climatology (air temperature, precip) for a
Greenland domain from CARRA2 HEIGHT-LEVEL temperatures.

Why height levels and not 2 m: CARRA2's 2 m temperature over the ice sheet
already reflects the stable boundary layer in which part of the atmospheric
energy has been spent on melting/warming the surface — feeding it to an
energy-balance SMB model double counts that exchange. The 100 m
above-ground temperature is taken as the "free" air the surface exchanges
with (the enthalpy model's H_atm·(T_air − T_s) term), and the second level
(500 m in the downloaded file) gives a LOCAL, monthly lapse rate
Γ = (T_hi − T_lo)/(z_hi − z_lo) used to move that air temperature from the
CARRA2 orography onto the model DEM:

    T_air(z_dem) = T_lo + Γ · (z_dem − z_orog)

(the 100 m offset is deliberately not subtracted — T_lo is the air the
model surface sees). Γ is clipped to [LAPSE_MIN, LAPSE_MAX]; inversions
(Γ > 0, winter interior) are real and kept.

Inputs (CDS CARRA2 pan-Arctic 2.5 km, converted to NetCDF by cfgrib):
    common_data/climate/carra2/1985_2025/t.nc       t(time, heightAboveGround, y, x) [K]
    common_data/climate/carra2/1985_2025/precip.nc  tp(time, y, x) [kg m-2 per day; stamped
                                                    12 UTC on the last day of the PREVIOUS month]
    common_data/climate/carra2/orog.nc              orog(y, x) [m]
The full record (1985-10 .. 2025-12, 483 months) is 32 GB decompressed per
variable, so the files are read lazily, cropped to the finite Greenland box
(2.1 M of 8.2 M cells) and accumulated month by month. The climatology
pools complete calendar years only by default (1986-2025, 40 samples per
month; `years` overrides, e.g. `years=[2000]` for a single-year field).
If the full-record directory is absent the 27-month files in
`climate/carra2/` are used (uneven coverage; see `months_per_calendar_month`).
Precipitation is CARRA's monthly mean daily accumulation (mm/day),
converted to m ice eq. / yr.

Output: {domain_path}/model_inputs/gridded_climate.nc with monthly_t2m
(deg C; keeps the library's variable name), monthly_precip, monthly_lapse_rate
(K/m), carra_orog (m).
"""
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from glare import PanCarraBase
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, median_filter

from domain_grid import grid_from_dem

CARRA_BASE = Path('../common_data/climate/carra2')
FULL_RECORD_DIR = 'full_record'   # resolved in carra_paths(); see there
OROG_PATH = CARRA_BASE / 'orog.nc'

LEVEL_LOW = 100.0            # m above ground: the forcing air temperature
LAPSE_MIN, LAPSE_MAX = -0.012, 0.010   # K/m
FALLBACK_LAPSE = -0.0065
ICE_DENSITY = 917.0
DAYS_PER_YEAR = 365.0
PRECIP_MEDIAN_WINDOW = 3     # native-grid speckle filter (cells); 0/1 disables
READ_BLOCK = 24              # months per read when accumulating


def carra_paths(base: Path = None):
    """(t_path, p_path, orog_path): the full record under
    `<base>/1985_2025/` when present, else the 27-month files in `<base>/`."""
    base = Path(base) if base is not None else CARRA_BASE
    full = base / '1985_2025'
    if (full / 't.nc').exists() and (full / 'precip.nc').exists():
        return full / 't.nc', full / 'precip.nc', base / 'orog.nc'
    return base / 't.nc', base / 'precip.nc', base / 'orog.nc'


def _precip_times(ds_p: xr.Dataset, tdim: str) -> np.ndarray:
    """CARRA2's monthly-mean-of-daily-accumulation product is stamped with
    the forecast base time, 12 UTC on the LAST DAY OF THE PRECEDING MONTH
    (January 2000 carries 1999-12-31T12). Shift by 12 h so the calendar
    month of the stamp is the month the field belongs to. Temperature
    fields are stamped on the first of the month and need no shift."""
    t = ds_p['valid_time'].values if 'valid_time' in ds_p else ds_p[tdim].values
    t = np.asarray(t, dtype='datetime64[ns]')
    if np.all(t.astype('datetime64[h]').astype(int) % 24 == 12):
        t = t + np.timedelta64(12, 'h')
    return t


def open_carra(base: Path = None):
    """Lazily open the CARRA2 files, crop to the finite Greenland box and
    return (carra, ds_t, ds_p, orog, x, y): `carra` is the PanCarraBase
    (grid geometry / projection), ds_t/ds_p are cropped datasets with the
    time coordinate set to the corrected month stamps, `orog` the cropped
    orography array and x, y the cropped native coordinates."""
    t_path, p_path, orog_path = carra_paths(base)
    for p in (t_path, p_path, orog_path):
        if not p.exists():
            raise FileNotFoundError(p)
    carra = PanCarraBase(p_path, t_path, orog_path)
    ds_t, ds_p = carra.temp_dataset, carra.precip_dataset
    tdim = 'time' if 'time' in ds_t['t'].dims else ds_t['t'].dims[0]
    ds_t = ds_t.assign_coords({tdim: ds_t['valid_time'].values}) if 'valid_time' in ds_t else ds_t
    ds_p = ds_p.assign_coords({tdim: _precip_times(ds_p, tdim)})
    probe = ds_t['t'].isel({tdim: 0, 'heightAboveGround': 0}).values
    fin = np.isfinite(probe)
    rows, cols = np.where(fin.any(1))[0], np.where(fin.any(0))[0]
    ys, xs = slice(int(rows[0]), int(rows[-1]) + 1), slice(int(cols[0]), int(cols[-1]) + 1)
    ds_t = ds_t.isel(y=ys, x=xs)
    ds_p = ds_p.isel(y=ys, x=xs)
    orog = carra.orog_dataset['orog'].values[ys, xs]
    return carra, ds_t, ds_p, orog, np.asarray(carra.x)[xs], np.asarray(carra.y)[ys], tdim


def _month_sums(da: xr.DataArray, tdim: str, block: int = READ_BLOCK):
    """Calendar-month mean over every timestep of `da` (time, ...), read in
    blocks so the full record never sits in memory. Returns (12, ...) mean
    (NaN for empty months) and the per-month sample counts."""
    months = da[tdim].dt.month.values
    n = len(months)
    shape = (12,) + da.shape[1:]
    s = np.zeros(shape, dtype='float64')
    cnt = np.zeros(12, dtype=int)
    for i0 in range(0, n, block):
        chunk = da.isel({tdim: slice(i0, i0 + block)}).values.astype('float64')
        for k, m in enumerate(months[i0:i0 + block]):
            s[m - 1] += np.nan_to_num(chunk[k])
            cnt[m - 1] += 1
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = s / cnt.reshape((12,) + (1,) * (s.ndim - 1))
    mean[cnt == 0] = np.nan
    # cells NaN in the probe are NaN everywhere: restore them
    finite = np.isfinite(da.isel({tdim: 0}).values)
    mean = np.where(finite[None], mean, np.nan)
    return mean, cnt


def _interp(x, y, field, qx, qy, method='linear'):
    """Bilinear interpolation of a (y, x) CARRA field at projected query
    points; NaN outside the finite subset."""
    return RegularGridInterpolator((x, y), np.asarray(field).T, method=method,
                                   bounds_error=False, fill_value=np.nan)((qx, qy))


def _fill_nearest(a):
    """Fill NaN cells with the nearest finite value (index-space nearest)."""
    bad = ~np.isfinite(a)
    if not bad.any() or bad.all():
        return a
    idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def _select_years(ds, tdim, years):
    return ds.sel({tdim: ds[tdim].dt.year.isin([int(y) for y in years])})


def build_climate(domain_path: str, year: int = None, reference: str = 'orog',
                  precip_median_window: int = PRECIP_MEDIAN_WINDOW,
                  level_low: float = LEVEL_LOW, years=None,
                  write: bool = True, output_path=None, carra_base=None) -> xr.Dataset:
    """`reference`: 'orog' lapse-corrects from the CARRA orography (default);
    'smoothed_dem' from the DEM low-passed to CARRA's 2.5 km (the
    alaska-forecast convention). `year` is accepted for make_all symmetry and
    ignored. `years` (iterable of calendar years) restricts the pooled months;
    the default is every complete calendar year in the files (all months
    when no year is complete). `write=False` returns the dataset without
    touching disk."""
    domain_path = Path(domain_path)
    dem = xr.load_dataset(domain_path / 'model_inputs' / 'gridded_dem.nc')
    if output_path is None:
        output_path = domain_path / 'model_inputs' / 'gridded_climate.nc'
    grid = grid_from_dem(dem)
    # The composite DEM carries fjord/shelf bathymetry; air over water sits
    # at sea level, so the lapse correction runs from max(z, 0).
    elevation = np.maximum(dem.elevation.values.astype('float64'), 0.0)

    carra, ds_t, ds_p, orog, cx, cy, tdim = open_carra(carra_base)
    levels = ds_t['heightAboveGround'].values.astype(float)
    i_lo = int(np.argmin(np.abs(levels - level_low)))
    i_hi = int(np.argmax(levels)) if len(levels) > 1 else None
    z_lo, z_hi = levels[i_lo], (levels[i_hi] if i_hi is not None else None)
    if i_hi == i_lo:
        i_hi = None
    print(f"CARRA2 levels {levels} m: forcing level {z_lo:g} m"
          + (f", lapse from ({z_lo:g}, {z_hi:g}) m" if i_hi is not None else ", no second level -> fixed lapse"))

    if years is None:
        yr = ds_t[tdim].dt.year.to_series()
        complete = sorted(y for y, n in yr.value_counts().items() if n == 12)
        years = complete if complete else None
    if years is not None:
        years = [int(y) for y in years]
        ds_t, ds_p = _select_years(ds_t, tdim, years), _select_years(ds_p, tdim, years)
        if ds_t.sizes[tdim] == 0 or ds_p.sizes[tdim] == 0:
            raise ValueError(f"CARRA2 files hold no months in years {years}")
    times = np.sort(ds_t[tdim].values)
    print(f"pooling {len(times)} months {str(times[0])[:7]}..{str(times[-1])[:7]}")

    print("Accumulating the native-grid climatology")
    clim_lo, n_per_month = _month_sums(ds_t['t'].isel(heightAboveGround=i_lo), tdim)
    clim_hi = _month_sums(ds_t['t'].isel(heightAboveGround=i_hi), tdim)[0] if i_hi is not None else None
    clim_p, _ = _month_sums(ds_p['tp'], tdim)
    empty = [m + 1 for m in range(12) if n_per_month[m] == 0]
    if empty:
        raise ValueError(f"CARRA2 temperature file has no data for months {empty}")

    grid_x, grid_y = np.meshgrid(dem.x.values.astype('float64'), dem.y.values.astype('float64'))
    qx, qy = carra.transform_to(grid_x, grid_y, grid.crs)

    if reference == 'orog':
        z_ref = _interp(cx, cy, orog, qx, qy)
    elif reference == 'smoothed_dem':
        from scipy.ndimage import gaussian_filter
        z_ref = gaussian_filter(elevation, sigma=2500.0 / grid.resolution, mode='nearest')
    else:
        raise ValueError(reference)

    t2m = np.empty((12,) + elevation.shape, dtype='float32')
    lapse = np.empty_like(t2m)
    precip = np.empty_like(t2m)
    print("Working on temperature fields")
    for m in range(12):
        t_lo = _interp(cx, cy, clim_lo[m], qx, qy)
        if clim_hi is not None:
            t_hi = _interp(cx, cy, clim_hi[m], qx, qy)
            g = np.clip((t_hi - t_lo) / (z_hi - z_lo), LAPSE_MIN, LAPSE_MAX)
        else:
            g = np.full_like(t_lo, FALLBACK_LAPSE)
        g = np.where(np.isfinite(g), g, FALLBACK_LAPSE)
        t2m[m] = (t_lo + g * (elevation - z_ref) - 273.15).astype('float32')
        lapse[m] = g.astype('float32')
    print("Working on precip fields")
    for m in range(12):
        field = clim_p[m]
        if precip_median_window and precip_median_window > 1:
            field = median_filter(np.nan_to_num(field), size=precip_median_window)
            field = np.where(np.isfinite(clim_p[m]), field, np.nan)
        precip[m] = (_interp(cx, cy, field, qx, qy) / ICE_DENSITY * DAYS_PER_YEAR).astype('float32')

    nan_frac = float((~np.isfinite(t2m[0])).mean())
    if nan_frac > 0:
        print(f"WARNING: {100 * nan_frac:.1f}% of the grid lies outside the CARRA2 subset; "
              f"filling with the nearest finite value")
        for m in range(12):
            t2m[m] = _fill_nearest(t2m[m]); precip[m] = _fill_nearest(precip[m]); lapse[m] = _fill_nearest(lapse[m])
        z_ref = _fill_nearest(z_ref)

    months = np.arange(0, 12, dtype=np.float32) / 12
    coords = {"t": months, "y": dem.y, "x": dem.x}
    dims = ['t', 'y', 'x']
    source = (f"CARRA2 pan-Arctic 2.5 km, {len(times)} monthly fields "
              f"{str(times[0])[:7]}..{str(times[-1])[:7]}, calendar-month climatology")
    ds = xr.Dataset(coords=coords)
    ds['spatial_ref'] = dem['spatial_ref']
    ds['monthly_t2m'] = xr.DataArray(t2m, dims=dims, coords=coords, attrs={
        'units': 'Deg C',
        'long_name': f'Monthly air temperature at {z_lo:g} m above ground (CARRA2), '
                     f'lapse-corrected to the DEM with the local {z_lo:g}-{z_hi:g} m lapse rate'
                     if i_hi is not None else f'Monthly {z_lo:g} m air temperature (CARRA2), fixed lapse',
        'source': source, 'forcing_level_m': float(z_lo),
        'lapse_reference': reference})
    ds['monthly_precip'] = xr.DataArray(precip, dims=dims, coords=coords, attrs={
        'units': 'm ice equivalent / yr',
        'long_name': 'Precipitation rate (CARRA2 total precipitation) at monthly time steps',
        'median_filter_window': int(precip_median_window or 0), 'source': source})
    ds['monthly_lapse_rate'] = xr.DataArray(lapse, dims=dims, coords=coords, attrs={
        'units': 'K m-1', 'long_name': 'Local monthly lapse rate between the two CARRA2 height levels',
        'clip': [LAPSE_MIN, LAPSE_MAX]})
    ds['carra_orog'] = xr.DataArray(z_ref.astype('float32'), dims=['y', 'x'],
                                    coords={"y": dem.y, "x": dem.x},
                                    attrs={'units': 'm', 'long_name': 'lapse-correction reference surface'})
    ds.attrs['climate_source'] = source
    ds.attrs['months_per_calendar_month'] = [int(n) for n in n_per_month]
    ds.attrs['climatology_years'] = f"{int(min(years))}-{int(max(years))}" if years else "all months"
    if not write:
        return ds
    ds.to_netcdf(output_path)
    print(f"wrote {output_path}: T {np.nanmin(t2m):.1f}..{np.nanmax(t2m):.1f} C, "
          f"P {np.nanmin(precip):.2f}..{np.nanmax(precip):.2f} m/yr, "
          f"lapse median {np.nanmedian(lapse) * 1000:.2f} K/km")
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-path", type=str, required=True)
    parser.add_argument("--reference", default="orog", choices=["orog", "smoothed_dem"])
    parser.add_argument("--years", type=int, nargs="*", default=None,
                        help="calendar years to pool (default: every complete year)")
    args = parser.parse_args()
    build_climate(args.domain_path, reference=args.reference, years=args.years)
