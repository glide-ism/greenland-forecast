"""Build the gridded surface-velocity field for a Greenland domain.

Sources (all EPSG:3413 GeoTIFF/NetCDF; select with --source, paths at the top):
    measures_annual   MEaSUREs annual mosaic NSIDC-0725 v5 (200 m, one Dec-Nov
                      year; default 2015-2016 = the ISMIP7 initial-state epoch),
                      with per-pixel error rasters ex/ey.
    measures_multiyear MEaSUREs multi-year mosaic NSIDC-0670 v1 (250 m,
                      1995-2015 average; what glide's example file used).
    itslive           ITS_LIVE v2 120 m regional composite RGI05A (2014-2022
                      average) with vx_error/vy_error.

Same-CRS products are area-averaged onto the domain grid (the target is
coarser than every source). A source in a different CRS falls back to the
alaska-forecast displacement finite-difference, which keeps the vector
orientation correct under reprojection.

Output: {domain_path}/model_inputs/gridded_velocity.nc with vx, vy, vmask
(|v| > 1 m/yr and finite), vx_err, vy_err (NaN when the source has none).
"""
import argparse
from pathlib import Path

import numpy as np
import pyproj
import rioxarray  # noqa: F401
import xarray as xr
from rasterio.enums import Resampling
from scipy.interpolate import RegularGridInterpolator

from domain_grid import grid_from_dem

VELOCITY_DIR = Path('../common_data/velocity')

SOURCES = {
    "measures_annual": dict(
        vx=VELOCITY_DIR / 'measures_0725' / 'greenland_vel_mosaic200_2015_2016_vx_v05.0.tif',
        vy=VELOCITY_DIR / 'measures_0725' / 'greenland_vel_mosaic200_2015_2016_vy_v05.0.tif',
        vx_err=VELOCITY_DIR / 'measures_0725' / 'greenland_vel_mosaic200_2015_2016_ex_v05.0.tif',
        vy_err=VELOCITY_DIR / 'measures_0725' / 'greenland_vel_mosaic200_2015_2016_ey_v05.0.tif',
        time=dict(time_nominal=2016.0, time_start=2015.92, time_end=2016.92),
        description="MEaSUREs Greenland annual velocity mosaic NSIDC-0725 v5, 2015-12 to 2016-11"),
    "measures_multiyear": dict(
        vx=VELOCITY_DIR / 'measures_0670' / 'greenland_vel_mosaic250_vx_v1.tif',
        vy=VELOCITY_DIR / 'measures_0670' / 'greenland_vel_mosaic250_vy_v1.tif',
        vx_err=VELOCITY_DIR / 'measures_0670' / 'greenland_vel_mosaic250_ex_v1.tif',
        vy_err=VELOCITY_DIR / 'measures_0670' / 'greenland_vel_mosaic250_ey_v1.tif',
        time=dict(time_nominal=2008.0, time_start=1995.9, time_end=2015.8),
        description="MEaSUREs multi-year Greenland velocity mosaic NSIDC-0670 v1, 1995-2015"),
    "itslive": dict(
        # One NetCDF (vx, vy, vx_error, vy_error, mapping) as served by
        # its-live.jpl.nasa.gov / s3://its-live-data/velocity_mosaic/v2/static/
        nc=VELOCITY_DIR / 'itslive' / 'ITS_LIVE_velocity_120m_RGI05A_0000_V02.1.nc',
        vx='vx', vy='vy', vx_err='vx_error', vy_err='vy_error',
        # The file's vx description: "climatological [2014-2024] vx determined
        # by a weighted least squares line fit, described by an offset and
        # slope ... The climatology uses a time-intercept of January 1, 2018."
        # The stored field is the OFFSET, i.e. the velocity at 2018.0 (not the
        # window midpoint); dvx_dt/dvy_dt carry the slope if another epoch is
        # ever wanted (v(t) = v + dv_dt (t - 2018)).
        time=dict(time_nominal=2018.0, time_start=2014.0, time_end=2024.0),
        description="ITS_LIVE V02.1 120 m summary mosaic RGI05A (2014-2024 climatology, created 2025-11)"),
    "glide_example": dict(
        # Fallback: the MEaSUREs 1995-2015 mosaic at 900 m inside glide's
        # example file (no error field).
        nc=Path('../../glide/data/GLIDE_greenland_inputs.h5'),
        vx='vx', vy='vy', vx_err=None, vy_err=None,
        time=dict(time_nominal=2008.0, time_start=1995.9, time_end=2015.8),
        description="MEaSUREs multi-year mosaic NSIDC-0670 via glide's GLIDE_greenland_inputs.h5 (900 m)"),
}
DEFAULT_SOURCE = "itslive"
FD_STEP_YEARS = 10.0
SPEED_MASK_THRESHOLD = 1.0   # m/yr


def _open(path: Path) -> xr.DataArray:
    da = rioxarray.open_rasterio(path, masked=True)
    return da.squeeze('band', drop=True) if 'band' in da.dims else da


def _open_nc(path: Path, var: str) -> xr.DataArray:
    """A (y, x) variable from an EPSG:3413 NetCDF/HDF5 with CRS attached."""
    engine = 'h5netcdf' if path.suffix == '.h5' else None
    ds = xr.open_dataset(path, engine=engine)
    da = ds[var].astype('float32')
    crs = None
    for name in ('mapping', 'Polar_Stereographic', 'proj', 'spatial_ref', 'crs'):
        if name in ds.variables and 'crs_wkt' in ds[name].attrs:
            crs = ds[name].attrs['crs_wkt']; break
        if name in ds.variables and 'spatial_epsg' in ds[name].attrs:
            crs = f"EPSG:{ds[name].attrs['spatial_epsg']}"; break
    # Drop the file's grid_mapping pointer (its target variable is not a
    # coordinate of the extracted array) before attaching the CRS.
    da.attrs.pop('grid_mapping', None)
    da.encoding.pop('grid_mapping', None)
    da = da.rio.write_crs(crs or "EPSG:3413", inplace=True)
    if 'y' in da.coords and da.y.values[0] < da.y.values[-1]:
        da = da.sortby('y', ascending=False)
    return da.rio.write_nodata(np.nan, inplace=True)


def _same_crs_regrid(da, template):
    out = da.astype('float32').rio.reproject_match(template, resampling=Resampling.average)
    return out.assign_coords(x=template.x, y=template.y).values


def _rotated_regrid(vx_da, vy_da, dem_crs, grid_x, grid_y):
    """alaska-forecast's displacement finite difference for a source CRS
    that differs from the project CRS."""
    src_crs = pyproj.CRS(vx_da.rio.crs.to_wkt())
    to_src = pyproj.Transformer.from_crs(dem_crs, src_crs, always_xy=True)
    to_dem = pyproj.Transformer.from_crs(src_crs, dem_crs, always_xy=True)
    gx_s, gy_s = to_src.transform(grid_x, grid_y)
    ys, xs = vx_da.y.values, vx_da.x.values
    flip = ys[0] > ys[-1]
    ys_asc = ys[::-1] if flip else ys
    def interp(da):
        v = np.nan_to_num(da.values[::-1] if flip else da.values)
        return RegularGridInterpolator((ys_asc, xs), v, bounds_error=False,
                                       fill_value=np.nan)((gy_s, gx_s))
    vx_s, vy_s = interp(vx_da), interp(vy_da)
    xp, yp = to_dem.transform(gx_s + FD_STEP_YEARS * vx_s, gy_s + FD_STEP_YEARS * vy_s)
    xm, ym = to_dem.transform(gx_s - FD_STEP_YEARS * vx_s, gy_s - FD_STEP_YEARS * vy_s)
    return ((xp - xm) / (2 * FD_STEP_YEARS)).astype('float32'), \
           ((yp - ym) / (2 * FD_STEP_YEARS)).astype('float32')


def build_velocity(domain_path: str, source: str = DEFAULT_SOURCE,
                   paths: dict = None) -> xr.Dataset:
    """Build the gridded velocity dataset for `domain_path` and write to disk."""
    domain_path = Path(domain_path)
    dem_path = domain_path / 'model_inputs' / 'gridded_dem.nc'
    output_path = domain_path / 'model_inputs' / 'gridded_velocity.nc'
    spec = dict(SOURCES[source])
    if paths:
        spec.update(paths)

    dem_ds = xr.load_dataset(dem_path)
    grid = grid_from_dem(dem_ds)
    template = grid.template()
    dem_crs = grid.crs

    if 'nc' in spec:
        nc = Path(spec['nc'])
        vx_da, vy_da = _open_nc(nc, spec['vx']), _open_nc(nc, spec['vy'])
    else:
        vx_da, vy_da = _open(Path(spec['vx'])), _open(Path(spec['vy']))
    src_crs = pyproj.CRS(vx_da.rio.crs.to_wkt())
    if src_crs == dem_crs:
        pad = 2 * grid.resolution
        vx_da = vx_da.rio.clip_box(grid.xmin - pad, grid.ymin - pad, grid.xmax + pad, grid.ymax + pad)
        vy_da = vy_da.rio.clip_box(grid.xmin - pad, grid.ymin - pad, grid.xmax + pad, grid.ymax + pad)
        vx, vy = _same_crs_regrid(vx_da, template), _same_crs_regrid(vy_da, template)
    else:
        print(f"source CRS {src_crs.to_string()} != project CRS; rotating vectors")
        grid_x, grid_y = np.meshgrid(grid.x, grid.y)
        vx, vy = _rotated_regrid(vx_da, vy_da, dem_crs, grid_x, grid_y)

    errs = {}
    for key in ('vx_err', 'vy_err'):
        p = spec.get(key)
        if 'nc' in spec and p is not None:
            e = _open_nc(Path(spec['nc']), p)
            errs[key] = _same_crs_regrid(
                e.rio.clip_box(grid.xmin - 2 * grid.resolution, grid.ymin - 2 * grid.resolution,
                               grid.xmax + 2 * grid.resolution, grid.ymax + 2 * grid.resolution), template)
        elif p is not None and Path(p).exists():
            e = _open(Path(p))
            if pyproj.CRS(e.rio.crs.to_wkt()) == dem_crs:
                e = e.rio.clip_box(grid.xmin - 2 * grid.resolution, grid.ymin - 2 * grid.resolution,
                                   grid.xmax + 2 * grid.resolution, grid.ymax + 2 * grid.resolution)
                errs[key] = _same_crs_regrid(e, template)
            else:
                errs[key] = np.full(vx.shape, np.nan, dtype='float32')
        else:
            print(f"no {key} raster for source {source!r}; writing NaN")
            errs[key] = np.full(vx.shape, np.nan, dtype='float32')

    speed = np.sqrt(vx ** 2 + vy ** 2)
    vmask = (np.isfinite(speed) & (speed > SPEED_MASK_THRESHOLD)).astype('float32')

    out = xr.Dataset(coords={'y': dem_ds.y, 'x': dem_ds.x})
    out['vx'] = (('y', 'x'), np.nan_to_num(vx).astype('float32'))
    out['vy'] = (('y', 'x'), np.nan_to_num(vy).astype('float32'))
    out['vmask'] = (('y', 'x'), vmask)
    out['vx_err'] = (('y', 'x'), errs['vx_err'])
    out['vy_err'] = (('y', 'x'), errs['vy_err'])
    for name in ('vx', 'vy', 'vmask', 'vx_err', 'vy_err'):
        out[name].attrs.update(spec['time'], source=spec['description'],
                               units='m yr-1' if name != 'vmask' else '1')
    out['spatial_ref'] = dem_ds['spatial_ref']
    out.attrs['velocity_source'] = spec['description']
    out.to_netcdf(output_path)
    print(f"wrote {output_path}: {int(vmask.sum())} moving cells")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-path", type=str, required=True)
    parser.add_argument("--source", choices=sorted(SOURCES), default=DEFAULT_SOURCE)
    args = parser.parse_args()
    build_velocity(args.domain_path, args.source)
