"""
Ocean thermal forcing -> model_inputs/thermal_forcing.nc

Source: the ISMIP7 Greenland ocean forcing (EN4-based TF, Verjans bias
correction and Slater inland mapping; github.com/ehultee/gris-iceocean-process),
one NetCDF per year with MONTHLY `tf(time, y, x)` (degC above the in-situ
freezing point) on the ISMIP 1 km grid (EPSG:3413, y ascending). TF is
defined over the ocean and fjords and extended inland under marine-based
ice by the Slater mapping; it is NaN elsewhere (~40% of the grid).

Output on the domain grid, one record per calendar year:

  tf_mean(time, y, x)  annual mean of the monthly TF
  tf_max(time, y, x)   annual maximum of the monthly TF
  tf_dist(y, x)        distance (km) to the nearest cell holding a NATIVE value
  time                 calendar year (int)

Cells without a native value are filled by nearest neighbour out to
--fill-km (NaN beyond), so a calving front one cell inside the mapped
region still sees a value; consumers use tf_dist to decide how far from
the mapped region the forcing may act (forward_standalone.py: TF_MAX_DIST_KM).
Both annual statistics are kept because a step of several years may want
the mean (integrated melt) or the max (episodic warm-water intrusions).

    python make_thermal_forcing.py --domain-path ../domains/greenland [--fill-km 10]
"""
import argparse
import glob
import re
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from rasterio.enums import Resampling
from scipy import ndimage

from domain_grid import grid_from_dem

TF_DIR = Path('../common_data/ocean/tf')
TF_PATTERN = 'tf_GrIS_*_{year}.nc'
TF_CRS = 'EPSG:3413'


def _year_files(tf_dir: Path, years=None) -> dict:
    files = {}
    for f in sorted(glob.glob(str(tf_dir / TF_PATTERN.format(year='*')))):
        m = re.search(r'_(\d{4})\.nc$', f)
        if m:
            files[int(m.group(1))] = Path(f)
    if years is not None:
        files = {y: f for y, f in files.items() if years[0] <= y <= years[1]}
    if not files:
        raise FileNotFoundError(f"no thermal forcing files {TF_PATTERN} under {tf_dir}")
    return files


def _to_domain(a: np.ndarray, native: xr.DataArray, template: xr.DataArray) -> np.ndarray:
    """Native-grid (y, x) array -> domain grid. Identity (up to a y flip)
    when the native grid IS the domain grid (ismip_greenland preset),
    otherwise a NaN-aware block average via rioxarray."""
    if (native.x.size == template.x.size and native.y.size == template.y.size
            and np.allclose(native.x.values, template.x.values)):
        if np.allclose(native.y.values, template.y.values):
            return a
        if np.allclose(native.y.values[::-1], template.y.values):
            return a[::-1]
    da = xr.DataArray(a.astype('float32'), dims=('y', 'x'),
                      coords={'y': native.y.values, 'x': native.x.values})
    da = da.rio.write_crs(TF_CRS, inplace=True).rio.write_nodata(np.nan, inplace=True)
    out = da.rio.reproject_match(template, resampling=Resampling.average)
    return out.values


def build_thermal_forcing(domain_path: str, fill_km: float = 10.0, years=None,
                          tf_dir=None, output_path=None) -> xr.Dataset:
    domain_path = Path(domain_path)
    tf_dir = Path(tf_dir) if tf_dir else TF_DIR
    output_path = Path(output_path) if output_path else domain_path / 'model_inputs' / 'thermal_forcing.nc'
    files = _year_files(tf_dir, years)

    dem = xr.load_dataset(domain_path / 'model_inputs' / 'gridded_dem.nc')
    grid = grid_from_dem(dem)
    template = grid.template()
    dx_km = float(grid.resolution) / 1000.0
    ny, nx = template.sizes['y'], template.sizes['x']

    # complete calendar years only: the kit's last file can be partial (2026
    # holds Jan-Feb), and a winter-only annual max/mean would bias the hold
    yrs = []
    for y in sorted(files):
        with xr.open_dataset(files[y]) as ds:
            n = ds.sizes['time']
        if n == 12:
            yrs.append(y)
        else:
            print(f"  {y}: {n} months only, skipped (incomplete year)")
    if not yrs:
        raise ValueError("no complete years in the thermal forcing record")
    tf_mean = np.full((len(yrs), ny, nx), np.nan, dtype='float32')
    tf_max = np.full((len(yrs), ny, nx), np.nan, dtype='float32')
    native_mask = None
    fill_idx = None
    tf_dist = None
    for k, y in enumerate(yrs):
        with xr.open_dataset(files[y]) as ds:
            native = ds['tf']
            monthly = native.values.astype('float32')            # (12, y, x), NaN off-product
            with warnings.catch_warnings():             # all-NaN columns off-product
                warnings.simplefilter("ignore", RuntimeWarning)
                mean_n = np.nanmean(monthly, axis=0)
                max_n = np.nanmax(monthly, axis=0)
        mean_d = _to_domain(mean_n, native, template)
        max_d = _to_domain(max_n, native, template)
        finite = np.isfinite(mean_d)
        if native_mask is None or not np.array_equal(finite, native_mask):
            # nearest native cell for every cell (recomputed only when the
            # product's footprint changes, which it does not in the ISMIP kit)
            native_mask = finite
            dist, idx = ndimage.distance_transform_edt(~finite, return_indices=True)
            tf_dist = (dist * dx_km).astype('float32')
            fill_idx = (idx[0], idx[1])
            fill = tf_dist <= fill_km
        tf_mean[k] = np.where(fill, mean_d[fill_idx], np.nan)
        tf_max[k] = np.where(fill, max_d[fill_idx], np.nan)
        print(f"  {y}: native cells {finite.sum()}, filled to {fill_km:g} km: {fill.sum()}, "
              f"margin-area mean TF {np.nanmean(mean_d):.2f} / max {np.nanmean(max_d):.2f} degC")

    out = xr.Dataset(coords={'time': np.array(yrs, dtype='int32'), 'y': dem.y, 'x': dem.x})
    src = ("ISMIP7 Greenland ocean thermal forcing, EN4 with Verjans bias correction and "
           "Slater inland mapping (NASA GSFC), monthly 1 km, annual statistics")
    out['tf_mean'] = (('time', 'y', 'x'), tf_mean)
    out['tf_mean'].attrs = dict(long_name='Annual mean ocean thermal forcing', units='degC', source=src)
    out['tf_max'] = (('time', 'y', 'x'), tf_max)
    out['tf_max'].attrs = dict(long_name='Annual maximum of monthly ocean thermal forcing', units='degC', source=src)
    out['tf_dist'] = (('y', 'x'), tf_dist)
    out['tf_dist'].attrs = dict(long_name='Distance to the nearest cell with a native TF value', units='km')
    out['time'].attrs = dict(long_name='calendar year')
    out.attrs.update(thermal_forcing_source=src, fill_km=float(fill_km),
                     years=f"{yrs[0]}-{yrs[-1]}", tf_dir=str(tf_dir))
    if 'spatial_ref' in dem:
        out['spatial_ref'] = dem['spatial_ref']
    enc = {v: dict(zlib=True, complevel=4, chunksizes=(1, ny, nx)) for v in ('tf_mean', 'tf_max')}
    enc['tf_dist'] = dict(zlib=True, complevel=4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(output_path, encoding=enc)
    print(f"wrote {output_path} ({yrs[0]}-{yrs[-1]}, {len(yrs)} years, "
          f"{output_path.stat().st_size / 1e6:.0f} MB)")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain-path", type=str, required=True)
    parser.add_argument("--fill-km", type=float, default=10.0,
                        help="nearest-neighbour fill distance beyond the native product (km)")
    parser.add_argument("--years", type=int, nargs=2, default=None, metavar=("Y0", "Y1"))
    parser.add_argument("--tf-dir", type=str, default=None)
    args = parser.parse_args()
    build_thermal_forcing(args.domain_path, fill_km=args.fill_km, years=args.years, tf_dir=args.tf_dir)
