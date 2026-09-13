"""Build the gridded end-of-summer snow/ice mask for a Greenland domain (optional).

The inverse model's snowline term (an ELA proxy) consumes, per cell, the
fraction of the glacierized subarea that retained snow at the end of the melt
season. alaska-forecast fed it a directory of categorical GeoTIFFs (0 no
data, 1 exposed ice, 2 snow). For Greenland the same categorical product can
be derived from MODIS bare-ice classifications (Ryan et al. 2019; MOD10A1 /
MCD43 albedo thresholds) or Sentinel-3 SICE bare-ice albedo — produce one
classified GeoTIFF per year (or one multi-year composite) in any projected
CRS under common_data/snowlines/<window>/ and this builder mosaics, reprojects
and area-averages it onto the domain grid.

The directory name gives the averaging window ('2015-2019-average' ->
time_start 2015, time_end 2019, nominal the midpoint).

Output: {domain_path}/model_inputs/gridded_snowline.nc
"""
import argparse
import re
from pathlib import Path

import numpy as np
import rasterio
import rasterio.warp
import rioxarray  # noqa: F401
import xarray as xr
from rasterio.enums import Resampling
from rasterio.merge import merge as rasterio_merge

from domain_grid import grid_from_dem

SNOWLINE_DIR = Path('../common_data/snowlines/2015-2019-average')
NODATA_CODE, ICE_CODE, SNOW_CODE = 0, 1, 2


def _window(snowline_dir: Path):
    m = re.match(r'^(\d{4})-(\d{4})', snowline_dir.name)
    return (float(m.group(1)), float(m.group(2))) if m else (2015.0, 2019.0)


def _intersecting_tiles(snowline_dir, bounds, target_crs):
    left, bottom, right, top = bounds
    selected = []
    for tile in sorted(snowline_dir.glob('*.tif')):
        with rasterio.open(tile) as src:
            tl, tb, tr, tt = rasterio.warp.transform_bounds(src.crs, target_crs, *src.bounds)
        if tr < left or tl > right or tt < bottom or tb > top:
            continue
        selected.append(tile)
    return selected


def build_snowline(domain_path: str, snowline_dir: str = None) -> xr.Dataset:
    domain_path = Path(domain_path)
    snowline_dir = Path(snowline_dir) if snowline_dir else SNOWLINE_DIR
    dem = xr.load_dataset(domain_path / 'model_inputs' / 'gridded_dem.nc')
    output_path = domain_path / 'model_inputs' / 'gridded_snowline.nc'
    grid = grid_from_dem(dem)
    template = grid.template()

    tiles = _intersecting_tiles(snowline_dir, grid.bounds, grid.crs.to_wkt())
    if not tiles:
        raise RuntimeError(f"No snowline tiles in {snowline_dir} intersect the domain grid.")
    datasets = [rasterio.open(t) for t in tiles]
    try:
        mosaic, transform = rasterio_merge(datasets, nodata=NODATA_CODE)
        mosaic_crs = datasets[0].crs
    finally:
        for d in datasets:
            d.close()
    mosaic = mosaic[0].astype(np.int32)
    n_y, n_x = mosaic.shape
    xs = transform.c + transform.a * (np.arange(n_x) + 0.5)
    ys = transform.f + transform.e * (np.arange(n_y) + 0.5)

    def native(values, dtype):
        da = xr.DataArray(values.astype(dtype), dims=('y', 'x'), coords={'y': ys, 'x': xs})
        da.rio.write_crs(mosaic_crs, inplace=True)
        da.rio.write_nodata(np.nan if dtype == np.float32 else NODATA_CODE, inplace=True)
        return da

    def regrid(da, resampling):
        return da.rio.reproject_match(template, resampling=resampling).assign_coords(
            x=template.x, y=template.y)

    snow_ice = regrid(native(mosaic, np.int32), Resampling.nearest)
    glacier_fraction = regrid(native(mosaic != NODATA_CODE, np.float32), Resampling.average)
    snow_avg = regrid(native(mosaic == SNOW_CODE, np.float32), Resampling.average)
    ice_avg = regrid(native(mosaic == ICE_CODE, np.float32), Resampling.average)
    safe = glacier_fraction.where(glacier_fraction > 0)
    t0, t1 = _window(snowline_dir)
    time_attrs = dict(time_nominal=0.5 * (t0 + t1), time_start=t0, time_end=t1)

    out = xr.Dataset(coords={'y': dem.y, 'x': dem.x})
    out['snow_ice'] = (('y', 'x'), snow_ice.values.astype('int32'))
    out['snow_ice'].attrs = dict(long_name='End-of-summer surface classification',
                                 flag_values=[NODATA_CODE, ICE_CODE, SNOW_CODE],
                                 flag_meanings='no_data ice snow', **time_attrs)
    out['snow_fraction'] = (('y', 'x'), (snow_avg / safe).values.astype('float32'))
    out['ice_fraction_eos'] = (('y', 'x'), (ice_avg / safe).values.astype('float32'))
    out['glacier_fraction'] = (('y', 'x'), glacier_fraction.values.astype('float32'))
    for name in ('snow_fraction', 'ice_fraction_eos', 'glacier_fraction'):
        out[name].attrs.update(time_attrs, units='1')
    out['spatial_ref'] = dem['spatial_ref']
    out.attrs['snowline_source'] = str(snowline_dir)
    out.to_netcdf(output_path)
    print(f"wrote {output_path} from {len(tiles)} tile(s)")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-path", type=str, required=True)
    parser.add_argument("--snowline-dir", type=str, default=None)
    args = parser.parse_args()
    build_snowline(args.domain_path, args.snowline_dir)
