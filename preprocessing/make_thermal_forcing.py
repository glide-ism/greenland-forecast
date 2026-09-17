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

Two fill methods put values where the ice needs them (--method):

  marine (default, 2026-09-17): only OPEN-WATER native cells are trusted
    (bathymetry_mask outside the ice mask). Their values are propagated
    layer by layer through the marine domain -- open water plus ice cells
    whose BedMachine bed lies below sea level -- through the whole CONNECTED
    marine domain (a --fill-km cap is optional) as the HARMONIC extension of
    the open-water values (Laplace's equation with the open water as
    Dirichlet data: smooth, seam-free, bounded by the fjord values): a front
    retreating along a trough stays forced, as under the Slater mapping. tf_dist is 0 on every
    reached cell (open water and the marine extension count as "native"
    for OceanForcingConfig.max_dist_km) and NaN elsewhere (land, ice on a
    bed above sea level, marine hollows not connected to the coast);
    tf_path_km holds the propagation distance for diagnostics.
    This discards the product's Slater-mapped under-ice values, which
    carry one sill-depth level of the ocean profile and swing by 1-1.5 K
    from year to year as uniform blocks (the fjord trunk moves 0.3 K),
    and it removes the Voronoi discs, seams and land-bridge leakage of the
    nearest-neighbour fill.
  nearest (the 2026-09 original): every cell within --fill-km of ANY
    native cell copies its Euclidean-nearest native cell; tf_dist is that
    distance.

Consumers use tf_dist to decide how far from the water the forcing may act
(OceanForcingConfig.max_dist_km). Both annual statistics are kept because a
step of several years may want the mean (integrated melt) or the max
(episodic warm-water intrusions).

    python make_thermal_forcing.py --domain-path ../domains/greenland [--fill-km 10] [--method marine|nearest]
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


class MarinePropagator:
    """Harmonic extension of open-water values through the marine domain.

    A breadth-first pass (8-connectivity) from the source cells through
    `marine` finds the reachable cells and their path distance; the values
    on those cells are then the solution of the discrete Laplace equation
    with the source cells as Dirichlet data. The extension is smooth, free
    of fill-front seams, and bounded by the source values (maximum
    principle), so domain extrema stay those of the fjord water. Masks are
    fixed, so the sparse system is factorized once and a year costs one
    solve."""

    def __init__(self, source: np.ndarray, marine: np.ndarray, n_layers: int = None):
        from scipy import sparse
        from scipy.sparse.linalg import splu
        ny, nx = source.shape
        N = ny * nx
        self.shape, self.source = source.shape, source
        filled = source.copy()
        dist = np.full(source.shape, np.nan, dtype='float32')
        dist[source] = 0.0
        k = 0
        kernel = np.ones((3, 3), np.int16)
        while n_layers is None or k < n_layers:
            k += 1
            c = ndimage.convolve(filled.astype(np.int16), kernel, mode='constant')
            new = marine & ~filled & (c > 0)
            if not new.any():
                break
            dist[new] = float(k)
            filled |= new
        self.n_layers, self.dist, self.filled = k - 1, dist, filled
        # Laplace system on the unknown cells U = filled & ~source, 8-neighbour stencil
        U = filled & ~source
        idx_u = np.flatnonzero(U)
        self.idx_u = idx_u
        col_of = np.full(N, -1, dtype=np.int64)
        col_of[idx_u] = np.arange(idx_u.size)
        iy, ix = np.unravel_index(idx_u, source.shape)
        flat_src, flat_u = source.ravel(), U.ravel()
        rows_a, cols_a, rows_b, cols_b = [], [], [], []
        deg = np.zeros(idx_u.size)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                jy, jx = iy + dy, ix + dx
                ok = (jy >= 0) & (jy < ny) & (jx >= 0) & (jx < nx)
                r = np.flatnonzero(ok)
                j = jy[ok] * nx + jx[ok]
                in_u, in_s = flat_u[j], flat_src[j]
                deg += np.bincount(r[in_u | in_s], minlength=idx_u.size)
                rows_a.append(r[in_u]); cols_a.append(col_of[j[in_u]])
                rows_b.append(r[in_s]); cols_b.append(j[in_s])
        rows_a, cols_a = np.concatenate(rows_a), np.concatenate(cols_a)
        rows_b, cols_b = np.concatenate(rows_b), np.concatenate(cols_b)
        n = idx_u.size
        A = (sparse.csc_matrix((deg, (np.arange(n), np.arange(n))), shape=(n, n))
             - sparse.csc_matrix((np.ones(rows_a.size), (rows_a, cols_a)), shape=(n, n)))
        self.B = sparse.csr_matrix((np.ones(rows_b.size), (rows_b, cols_b)), shape=(n, N))
        self.lu = splu(A) if n else None

    def __call__(self, a: np.ndarray) -> np.ndarray:
        """Fill the (y, x) field `a`: open-water native values kept, the
        reachable marine domain filled harmonically, NaN elsewhere."""
        src = self.source & np.isfinite(a)
        v = np.where(src, a, 0.0).astype('float64').ravel()
        out = np.full(v.shape, np.nan, dtype='float32')
        out[src.ravel()] = v[src.ravel()]
        if self.lu is not None:
            out[self.idx_u] = self.lu.solve(self.B @ v)
        return out.reshape(self.shape)


def marine_masks(dem: xr.Dataset):
    """(open_water, marine): bathymetry outside the ice mask; that plus ice
    cells whose BedMachine bed is below sea level."""
    ice = dem['rgi_mask'].values.astype(bool)
    open_water = dem['bathymetry_mask'].values.astype(bool) & ~ice
    bed = dem['bed_obs'].values if 'bed_obs' in dem else dem['elevation'].values
    marine = open_water | (ice & np.isfinite(bed) & (bed < 0.0))
    return open_water, marine


def ice_band_mask(domain_path, band_km: float) -> np.ndarray:
    """Cells within `band_km` of the domain's ice mask (rgi_mask of
    gridded_dem.nc): the only ocean cells whose thermal forcing can ever
    reach a calving front. Used to blank the far ocean in long records."""
    domain_path = Path(domain_path)
    dem = xr.load_dataset(domain_path / 'model_inputs' / 'gridded_dem.nc')
    dx_km = float(grid_from_dem(dem).resolution) / 1000.0
    ice = dem['rgi_mask'].values.astype(bool)
    dist = ndimage.distance_transform_edt(~ice) * dx_km
    return dist <= band_km


def build_thermal_forcing(domain_path: str, fill_km: float = 10.0, years=None,
                          tf_dir=None, output_path=None, files: dict = None,
                          band_km: float = None, source: str = None,
                          extra_attrs: dict = None, method: str = 'marine') -> xr.Dataset:
    """`files` (year -> path) overrides the directory scan (a spliced
    historical + scenario record, make_ismip7_forcing.py); `band_km` blanks
    (NaN) the statistics farther than that from the ice (tf_dist is kept),
    which keeps a 450-year record small; `method` is the fill ('marine' |
    'nearest', see the module docstring). Years are streamed into the file
    one at a time (a 451-year record would need 17 GB dense). Returns the
    written dataset opened lazily."""
    if method not in ('marine', 'nearest'):
        raise ValueError(f"method must be 'marine' or 'nearest', got {method!r}")
    import netCDF4

    domain_path = Path(domain_path)
    tf_dir = Path(tf_dir) if tf_dir else TF_DIR
    output_path = Path(output_path) if output_path else domain_path / 'model_inputs' / 'thermal_forcing.nc'
    files = dict(files) if files is not None else _year_files(tf_dir, years)

    dem = xr.load_dataset(domain_path / 'model_inputs' / 'gridded_dem.nc')
    grid = grid_from_dem(dem)
    template = grid.template()
    dx_km = float(grid.resolution) / 1000.0
    ny, nx = template.sizes['y'], template.sizes['x']
    band = ice_band_mask(domain_path, band_km) if band_km is not None else None
    propagator = None
    if method == 'marine':
        open_water, marine = marine_masks(dem)
        n_layers = None if fill_km is None else int(round(fill_km / dx_km))
        print(f"marine fill: {int(open_water.sum())} open-water cells, marine domain {int(marine.sum())} cells, "
              f"{'unbounded' if n_layers is None else n_layers} layers of {dx_km:g} km", flush=True)
    elif fill_km is None:
        fill_km = 10.0

    # complete calendar years only: the kit's last file can be partial (2026
    # holds Jan-Feb), and a winter-only annual max/mean would bias the hold
    yrs = []
    for y in sorted(files):
        with xr.open_dataset(files[y], decode_times=False) as ds:
            n = ds.sizes['time']
        if n == 12:
            yrs.append(y)
        else:
            print(f"  {y}: {n} months only, skipped (incomplete year)")
    if not yrs:
        raise ValueError("no complete years in the thermal forcing record")

    src = source or ("ISMIP7 Greenland ocean thermal forcing, EN4 with Verjans bias correction and "
                     "Slater inland mapping (NASA GSFC), monthly 1 km, annual statistics")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nc = netCDF4.Dataset(output_path, 'w', format='NETCDF4')
    nc.createDimension('time', len(yrs))
    nc.createDimension('y', ny)
    nc.createDimension('x', nx)
    v_t = nc.createVariable('time', 'i4', ('time',))
    v_t[:] = np.array(yrs, dtype='int32')
    v_t.long_name = 'calendar year'
    v_y = nc.createVariable('y', 'f8', ('y',)); v_y[:] = dem.y.values
    v_x = nc.createVariable('x', 'f8', ('x',)); v_x[:] = dem.x.values
    for c, v in ((dem.y, v_y), (dem.x, v_x)):
        for k, a in c.attrs.items():
            setattr(v, k, a)
    v_mean = nc.createVariable('tf_mean', 'f4', ('time', 'y', 'x'), zlib=True, complevel=4,
                               chunksizes=(1, ny, nx), fill_value=np.float32(np.nan))
    v_mean.long_name = 'Annual mean ocean thermal forcing'; v_mean.units = 'degC'; v_mean.source = src
    v_max = nc.createVariable('tf_max', 'f4', ('time', 'y', 'x'), zlib=True, complevel=4,
                              chunksizes=(1, ny, nx), fill_value=np.float32(np.nan))
    v_max.long_name = 'Annual maximum of monthly ocean thermal forcing'; v_max.units = 'degC'; v_max.source = src
    v_dist = nc.createVariable('tf_dist', 'f4', ('y', 'x'), zlib=True, complevel=4)
    v_dist.long_name = ('0 on open water and the connected marine extension, NaN elsewhere' if method == 'marine' else
                        'Distance to the nearest cell with a native TF value'); v_dist.units = 'km'
    if method == 'marine':
        v_path = nc.createVariable('tf_path_km', 'f4', ('y', 'x'), zlib=True, complevel=4)
        v_path.long_name = 'Propagation distance from open water through the marine domain'; v_path.units = 'km'
    if 'spatial_ref' in dem:
        v_sr = nc.createVariable('spatial_ref', 'i8')
        for k, a in dem['spatial_ref'].attrs.items():
            setattr(v_sr, k, a)
        for v in (v_mean, v_max, v_dist) + ((v_path,) if method == 'marine' else ()):
            v.grid_mapping = 'spatial_ref'
    nc.thermal_forcing_source = src
    nc.fill_km = float(fill_km) if fill_km is not None else -1.0
    nc.fill_method = method
    nc.years = f"{yrs[0]}-{yrs[-1]}"
    nc.tf_dir = str(tf_dir)
    if band_km is not None:
        nc.ice_band_km = float(band_km)
    for k, a in (extra_attrs or {}).items():
        setattr(nc, k, a)

    native_mask = None
    fill_idx = None
    tf_dist = None
    fill = None
    for k, y in enumerate(yrs):
        with xr.open_dataset(files[y], decode_times=False) as ds:
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
            # fill geometry (recomputed only when the product's footprint
            # changes, which it does not in the ISMIP kit)
            native_mask = finite
            if method == 'marine':
                propagator = MarinePropagator(open_water & finite, marine, n_layers)
                tf_path = propagator.dist * dx_km
                fill = propagator.filled
                tf_dist = np.where(fill, 0.0, np.nan).astype('float32')
                v_path[:, :] = tf_path
                print(f"  propagation: {propagator.n_layers} layers, {int(fill.sum() - (open_water & finite).sum())} "
                      f"cells filled beyond open water ({int((fill & ~open_water).sum())} under ice), "
                      f"longest path {np.nanmax(tf_path):.0f} km", flush=True)
            else:
                dist, idx = ndimage.distance_transform_edt(~finite, return_indices=True)
                tf_dist = (dist * dx_km).astype('float32')
                fill_idx = (idx[0], idx[1])
                fill = tf_dist <= fill_km
            if band is not None:
                fill &= band
            v_dist[:, :] = tf_dist
        if method == 'marine':
            mean_f, max_f = propagator(mean_d), propagator(max_d)
        else:
            mean_f, max_f = mean_d[fill_idx], max_d[fill_idx]
        v_mean[k, :, :] = np.where(fill, mean_f, np.nan)
        v_max[k, :, :] = np.where(fill, max_f, np.nan)
        print(f"  {y}: native cells {finite.sum()}, {method} fill{f' to {fill_km:g} km' if fill_km is not None else ''}"
              f"{f' within {band_km:g} km of ice' if band is not None else ''}: {fill.sum()}, "
              f"margin-area mean TF {np.nanmean(mean_d):.2f} / max {np.nanmean(max_d):.2f} degC",
              flush=True)
    nc.close()
    print(f"wrote {output_path} ({yrs[0]}-{yrs[-1]}, {len(yrs)} years, "
          f"{output_path.stat().st_size / 1e6:.0f} MB)")
    return xr.open_dataset(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain-path", type=str, required=True)
    parser.add_argument("--fill-km", type=float, default=None,
                        help="fill reach in km (nearest: default 10; marine: default unbounded)")
    parser.add_argument("--years", type=int, nargs=2, default=None, metavar=("Y0", "Y1"))
    parser.add_argument("--tf-dir", type=str, default=None)
    parser.add_argument("--method", choices=("marine", "nearest"), default="marine")
    parser.add_argument("--output-path", type=str, default=None)
    args = parser.parse_args()
    build_thermal_forcing(args.domain_path, fill_km=args.fill_km, years=args.years, tf_dir=args.tf_dir,
                          method=args.method, output_path=args.output_path)
