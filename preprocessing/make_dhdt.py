"""Build the gridded surface elevation-change rate (dH/dt) for a Greenland domain.

Sources (--source):
    atl15     ICESat-2 ATL15 (NSIDC, EPSG:3413, 1 km): a per-pixel weighted
              least-squares trend of the `delta_h` time series over
              [--t0, --t1] (default the full record), with the slope error
              propagated from `delta_h_sigma`. Default.
    itslive_dh ITS_LIVE Greenland ice-sheet + peripheral-glacier elevation
              change (MEaSUREs G1920V01 v1.1, Nilsson & Gardner; 1.92 km,
              monthly 1992-2023, EPSG:3413): the same weighted trend of `dh`
              over [--t0, --t1] with `rms` as the per-epoch error — the long-
              window (e.g. 2000-2020) complement to ATL15.
    gridded   Any NetCDF/GeoTIFF pair giving a rate and its 1-sigma error on a
              georeferenced grid (e.g. the ESA CCI Greenland SEC product, the
              Smith et al. 2020 ICESat/ICESat-2 2003-2019 grids, or a
              CryoSat-2/ICESat-2 merged rate); variable names and the window
              are given on the command line.
    hugonnet  Hugonnet et al. (2021) 1-degree tiles (RGI region 05 for the
              peripheral glaciers), coalesced as in alaska-forecast.

The rate is area-averaged onto the domain grid and written with the
observation window as variable-level attrs; the inverse model compares the
two-snapshot model rate (H(t1) - H(t0)) / (t1 - t0) against it.

Output: {domain_path}/model_inputs/gridded_dhdt.nc
"""
import argparse
from pathlib import Path

import numpy as np
import rasterio
import rasterio.warp
import rioxarray  # noqa: F401
import xarray as xr
from rasterio.enums import Resampling

from domain_grid import grid_from_dem

ATL15_PATH = Path('../common_data/dhdt/atl15/ATL15_GL_0329_01km_005_02.nc')
ITSLIVE_DH_PATH = Path('../common_data/dhdt/measures/Greenland_G1920V01_IceSheetGlacierIceHeight.nc')
ATL15_CRS = "EPSG:3413"
GRIDDED_DEFAULT = dict(path='../common_data/dhdt/cci_sec/greenland_sec.nc',
                       rate_var='dhdt', err_var='dhdt_err', crs=None)
HUGONNET_DIR = Path('../common_data/dhdt/hugonnet/dhdt')
HUGONNET_ERR_DIR = Path('../common_data/dhdt/hugonnet/dhdt_err')
HUGONNET_WINDOW = (2000.0, 2020.0)


def _regrid(da, template):
    out = da.astype('float32').rio.reproject_match(template, resampling=Resampling.average)
    return out.assign_coords(x=template.x, y=template.y).values


def _decimal_years(time_values, epoch=None):
    """Decimal years from decoded datetime64 values, or from raw days
    since `epoch` (files whose time xarray could not decode)."""
    tv = np.asarray(time_values)
    if np.issubdtype(tv.dtype, np.datetime64):
        dates = tv.astype('datetime64[s]')
    else:
        base = np.datetime64(epoch, 'D')
        dates = (base + np.asarray(tv, dtype='float64').astype('timedelta64[D]')).astype('datetime64[s]')
    years = dates.astype('datetime64[Y]').astype(int) + 1970
    y0 = np.array([np.datetime64(f'{y}-01-01') for y in years])
    y1 = np.array([np.datetime64(f'{y + 1}-01-01') for y in years])
    return years + (dates - y0) / (y1 - y0)


def _wls_trend(t, h, s):
    """Per-pixel weighted least-squares slope of h(t) with weights 1/s^2;
    returns (slope, slope_err, n_used) over the leading time axis."""
    w = np.where(np.isfinite(h) & np.isfinite(s) & (s > 0), 1.0 / np.maximum(s, 1e-3) ** 2, 0.0)
    h = np.nan_to_num(h)
    W = w.sum(axis=0)
    tbar = (w * t[:, None, None]).sum(axis=0) / np.where(W > 0, W, np.nan)
    dt = t[:, None, None] - tbar
    Stt = (w * dt ** 2).sum(axis=0)
    slope = (w * dt * h).sum(axis=0) / np.where(Stt > 0, Stt, np.nan)
    # Var(slope) = sum_i (w_i dt_i / Stt)^2 sigma_i^2 = 1/Stt for w = 1/sigma^2
    err = np.sqrt(1.0 / np.where(Stt > 0, Stt, np.nan))
    n = (w > 0).sum(axis=0)
    slope[n < 2] = np.nan
    err[n < 2] = np.nan
    return slope, err, n


def _window_mask(t, t0, t1, what):
    keep = np.ones_like(t, dtype=bool)
    if t0 is not None:
        keep &= t >= t0 - 1e-6
    if t1 is not None:
        keep &= t <= t1 + 1e-6
    if keep.sum() < 2:
        raise ValueError(f"{what} window [{t0}, {t1}] contains < 2 epochs of {t.min():.2f}..{t.max():.2f}")
    return keep


def _atl15_rate(path: Path, t0, t1):
    """Weighted least-squares trend of delta_h over [t0, t1] -> (rate, err,
    window)."""
    ds = xr.open_dataset(path, group='delta_h')
    units = ds['time'].attrs.get('units', 'days since 2018-01-01')
    t = _decimal_years(ds['time'].values, units.split('since')[-1].strip().split(' ')[0])
    keep = _window_mask(t, t0, t1, "ATL15")
    h = ds['delta_h'].values[keep].astype('float64')
    s = ds['delta_h_sigma'].values[keep].astype('float64')
    t = t[keep]
    slope, err, n = _wls_trend(t, h, s)
    template = ds['delta_h'].isel(time=0).drop_vars('time')
    template = template.rio.write_crs(ATL15_CRS, inplace=True)
    return (xr.DataArray(slope.astype('float32'), dims=template.dims, coords=template.coords).rio.write_crs(ATL15_CRS),
            xr.DataArray(err.astype('float32'), dims=template.dims, coords=template.coords).rio.write_crs(ATL15_CRS),
            (float(t.min()), float(t.max())))


def _itslive_dh_rate(path: Path, t0, t1):
    """Weighted trend of the ITS_LIVE monthly `dh` (error `rms`) over
    [t0, t1], read in yearly chunks to bound memory."""
    ds = xr.open_dataset(path)
    t_all = _decimal_years(ds['time'].values, 'days since 1992-01-15')
    keep = _window_mask(t_all, t0, t1, "ITS_LIVE dh")
    idx = np.where(keep)[0]
    h = ds['dh'].isel(time=idx).values.astype('float64')
    s = ds['rms'].isel(time=idx).values.astype('float64')
    t = t_all[keep]
    slope, err, n = _wls_trend(t, h, s)
    if 'mask' in ds:
        off = ds['mask'].values == 0
        slope[off] = np.nan; err[off] = np.nan
    crs = ds['proj'].attrs.get('spatial_ref', "EPSG:3413") if 'proj' in ds else "EPSG:3413"
    template = ds['h_dem'] if 'h_dem' in ds else ds['dh'].isel(time=0).drop_vars('time')
    def da(a):
        out = xr.DataArray(a.astype('float32'), dims=template.dims, coords=template.coords)
        out = out.rio.write_crs(crs, inplace=True)
        if out.y.values[0] < out.y.values[-1]:
            out = out.sortby('y', ascending=False)
        return out
    return da(slope), da(err), (float(t.min()), float(t.max()))


def _hugonnet(grid, template):
    left, bottom, right, top = grid.bounds
    tiles = []
    for tp in sorted(HUGONNET_DIR.glob('*_dhdt.tif')):
        with rasterio.open(tp) as src:
            tl, tb, tr, tt = rasterio.warp.transform_bounds(src.crs, grid.crs.to_wkt(), *src.bounds)
        if tr < left or tl > right or tt < bottom or tb > top:
            continue
        tiles.append(tp)
    if not tiles:
        raise RuntimeError(f"No Hugonnet tiles in {HUGONNET_DIR} intersect the grid")
    def coalesce(paths):
        mosaic = None
        for p in paths:
            t = rioxarray.open_rasterio(p, masked=True).squeeze('band', drop=True)
            m = t.rio.reproject_match(template, resampling=Resampling.bilinear)
            m = m.assign_coords(x=template.x, y=template.y)
            mosaic = m if mosaic is None else mosaic.combine_first(m)
        return mosaic.values
    err_tiles = [HUGONNET_ERR_DIR / (t.name[:-len('_dhdt.tif')] + '_dhdt_err.tif') for t in tiles]
    return coalesce(tiles), coalesce([p for p in err_tiles if p.exists()]), HUGONNET_WINDOW


def build_dhdt(domain_path: str, source: str = 'atl15', t0=None, t1=None,
               gridded: dict = None, name: str = None) -> xr.Dataset:
    """`name` writes gridded_dhdt_<name>.nc instead of gridded_dhdt.nc: a
    second product over another window, consumed by a DhdtSpec(filename=,
    name=) next to the primary one (it is NOT merged into GLIDE_inputs)."""
    domain_path = Path(domain_path)
    dem = xr.load_dataset(domain_path / 'model_inputs' / 'gridded_dem.nc')
    output_path = domain_path / 'model_inputs' / (f'gridded_dhdt_{name}.nc' if name else 'gridded_dhdt.nc')
    grid = grid_from_dem(dem)
    template = grid.template()

    if source == 'atl15':
        rate, err, window = _atl15_rate(ATL15_PATH, t0, t1)
        rate_v, err_v = _regrid(rate, template), _regrid(err, template)
        desc = "ICESat-2 ATL15 delta_h trend {0:.2f}-{1:.2f}"
    elif source == 'itslive_dh':
        rate, err, window = _itslive_dh_rate(ITSLIVE_DH_PATH, t0, t1)
        rate_v, err_v = _regrid(rate, template), _regrid(err, template)
        desc = "ITS_LIVE Greenland elevation change (G1920V01) dh trend {0:.2f}-{1:.2f}"
    elif source == 'gridded':
        g = dict(GRIDDED_DEFAULT, **(gridded or {}))
        path = Path(g['path'])
        if path.suffix == '.nc':
            ds = xr.open_dataset(path)
            rate, err = ds[g['rate_var']], ds[g['err_var']]
            if g.get('crs'):
                rate = rate.rio.write_crs(g['crs']); err = err.rio.write_crs(g['crs'])
        else:
            rate = rioxarray.open_rasterio(path, masked=True).squeeze('band', drop=True)
            err = rioxarray.open_rasterio(g['err_path'], masked=True).squeeze('band', drop=True)
        if t0 is None or t1 is None:
            raise ValueError("--t0/--t1 are required for --source gridded")
        window = (float(t0), float(t1))
        rate_v, err_v = _regrid(rate, template), _regrid(err, template)
        desc = f"gridded rate from {path.name} ({g['rate_var']}/{g['err_var']})"
    elif source == 'hugonnet':
        rate_v, err_v, window = _hugonnet(grid, template)
        desc = "Hugonnet et al. (2021) 2000-2020 tiles"
    else:
        raise ValueError(source)

    # The window the inverse compares (H(t1) - H(t0)) / (t1 - t0) against:
    # the requested [t0, t1] when given (the fit uses the epochs inside it;
    # the slope is the mean rate over that span, and round breakpoints keep
    # the step scheduler from inserting slivers between products), else the
    # kept epochs' extent rounded to 1e-2 yr (ATL15's axis starts 6 h into
    # 2019; unrounded it inserted a 6-hour dynamics step).
    if t0 is not None and t1 is not None and source in ('atl15', 'itslive_dh'):
        window = (float(t0), float(t1))
    window = (float(np.round(window[0], 2)), float(np.round(window[1], 2)))
    desc = desc.format(*window)         # the window the attrs carry
    attrs = dict(time_nominal=0.5 * (window[0] + window[1]),
                 time_start=window[0], time_end=window[1])
    out = xr.Dataset(coords={'y': dem.y, 'x': dem.x})
    out['dhdt'] = (('y', 'x'), rate_v.astype('float32'))
    out['dhdt'].attrs = dict(long_name='Mean rate of surface elevation change',
                             units='m yr-1', source=desc, **attrs)
    out['dhdt_err'] = (('y', 'x'), err_v.astype('float32'))
    out['dhdt_err'].attrs = dict(long_name='Uncertainty (1-sigma) of the elevation-change rate',
                                 units='m yr-1', source=desc, **attrs)
    out['spatial_ref'] = dem['spatial_ref']
    out.attrs['dhdt_source'] = desc
    out.to_netcdf(output_path)
    print(f"wrote {output_path} ({desc}); valid cells {int(np.isfinite(rate_v).sum())}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-path", type=str, required=True)
    parser.add_argument("--source", choices=('atl15', 'itslive_dh', 'gridded', 'hugonnet'), default='atl15')
    parser.add_argument("--t0", type=float, default=None)
    parser.add_argument("--t1", type=float, default=None)
    parser.add_argument("--gridded-path", type=str, default=None)
    parser.add_argument("--rate-var", type=str, default=None)
    parser.add_argument("--err-var", type=str, default=None)
    parser.add_argument("--gridded-crs", type=str, default=None)
    parser.add_argument("--name", type=str, default=None,
                        help="write gridded_dhdt_<name>.nc (a second product over another window)")
    args = parser.parse_args()
    g = {k: v for k, v in dict(path=args.gridded_path, rate_var=args.rate_var,
                               err_var=args.err_var, crs=args.gridded_crs).items()
         if v is not None}
    build_dhdt(args.domain_path, args.source, args.t0, args.t1, g, name=args.name)
