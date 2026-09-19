"""Build the gridded end-of-summer snow / bare-ice labels for a Greenland domain (optional).

The inverse's snowline term (an ELA proxy, SnowlineSpec) consumes, per cell,
the fraction of the glacierized subarea that retained snow at the end of the
melt season. alaska-forecast fed it categorical rasters; the Greenland
product is one shapefile per year 2000-2020 in
common_data/snowlines/<year>_snowline.zip: the end-of-summer snowline as
POLYLINES on a 500 m lattice in a polar stereographic CRS (lon_0 = 0,
lat_ts = 60 as the .prj says; verified against the ice mask — 90 % of the
vertices fall on ice at 490-2200 m, lat_ts 70 would put them 400 m too
high). Every part is a CLOSED RING (664 of them in 2012, one 19,464 km
long): they are the boundaries of the snow-covered region exported as
lines, so the sides are recovered exactly from the ring nesting depth —
depth 1 is the snow-covered ice (1.49 M km2 at 1300-3050 m in 2012), depth 0
outside it is bare ice (or off ice), depth 2 bare patches inside the snow,
depth 3 snow inside those: SNOW = ODD DEPTH. (An elevation-based
reconstruction — DEM above the interpolated snowline elevation — was tried
first and mislabels wedges next to the small rings and the low northern
interior; not needed.)

Per year the rings are rasterized with the additive merge on the product's
own 500 m lattice (the domain grid halved, aligned), the four subcells are
averaged to the 1 km cell and divided by the cell's ice fraction (a margin
cell whose ice half is snow is fully snow over its glacierized subarea),
clipped to 1. Peripheral-glacier cells the product never classified
(rgi_periphery_fraction > 0.5, depth 0 in every year, no ring vertex within
`--unclassified-km`) are excluded by glacier_fraction = 0.

Output: {domain_path}/model_inputs/gridded_snowline.nc

    snow_fraction     (y, x)   mean label over the years — the fraction of
                               the 2000-2020 seasons the cell ended with
                               snow, an empirical P(above the ELA), the
                               quantity the observation's sigmoid(SMB /
                               s_smb) models; time_nominal the midpoint
    glacier_fraction  (y, x)   the DEM's ice_fraction, 0 where unclassified
    snow_label        (year, y, x)  int8, per-year label x 100
    n_years           scalar attr; per-year bare-ice areas in the attrs

    python make_snowline.py --domain-path ../domains/greenland [--years 2000 2020]
"""
import argparse
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio.features
import xarray as xr
from rasterio.enums import MergeAlg
from rasterio.transform import from_origin
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

SNOWLINE_DIR = Path('../common_data/snowlines')
PRODUCT_CRS = '+proj=stere +lat_0=90 +lat_ts=60 +lon_0=0 +datum=WGS84 +units=m +no_defs'
SUB = 2                        # subcells per cell side: the product's 500 m lattice on the 1 km grid
UNCLASSIFIED_KM = 5.0


def snowline_files(snowline_dir: Path, years=None):
    files = {}
    for p in sorted(snowline_dir.glob('*_snowline.zip')):
        m = re.match(r'^(\d{4})_snowline\.zip$', p.name)
        if m and (years is None or years[0] <= int(m.group(1)) <= years[1]):
            files[int(m.group(1))] = p
    return files


def rings(path: Path, target_crs):
    """The year's ring polygons in the domain CRS (+ their vertices, (n, 2))."""
    g = gpd.read_file(f'zip://{path}')
    # the .prj's Polar_Stereographic with latitude_of_origin 60 is lat_ts 60
    g = g.set_crs(PRODUCT_CRS, allow_override=True).to_crs(target_crs)
    polys, pts = [], []
    for part in g.geometry.explode(index_parts=False):
        c = np.asarray(part.coords)[:, :2]
        if len(c) < 4:
            continue
        polys.append(Polygon(c))
        pts.append(c)
    return polys, np.concatenate(pts)


def year_fraction(polys, x, y, ice_fraction):
    """Snow fraction of the glacierized subarea per 1 km cell (NaN off ice)."""
    ny, nx = ice_fraction.shape
    dx = float(abs(x[1] - x[0]))
    sub = dx / SUB
    # 500 m lattice covering the same extent; cell (i, j) -> subcells SUB*i.., SUB*j..
    tr = from_origin(x[0] - dx / 2, y[0] + dx / 2, sub, sub)
    depth = rasterio.features.rasterize(((p, 1) for p in polys), out_shape=(ny * SUB, nx * SUB),
                                        transform=tr, merge_alg=MergeAlg.add, dtype='int16')
    snow = (depth % 2 == 1).astype('float32')
    snow_cell = snow.reshape(ny, SUB, nx, SUB).mean(axis=(1, 3))
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = np.where(ice_fraction > 0, np.minimum(snow_cell / np.maximum(ice_fraction, 1.0 / SUB ** 2), 1.0), np.nan)
    return frac.astype('float32'), depth


def build_snowline(domain_path: str, snowline_dir: str = None, years=None,
                   unclassified_km: float = UNCLASSIFIED_KM) -> xr.Dataset:
    domain_path = Path(domain_path)
    snowline_dir = Path(snowline_dir) if snowline_dir else SNOWLINE_DIR
    files = snowline_files(snowline_dir, years)
    if not files:
        raise RuntimeError(f'no <year>_snowline.zip in {snowline_dir}')
    dem = xr.load_dataset(domain_path / 'model_inputs' / 'gridded_dem.nc')
    output_path = domain_path / 'model_inputs' / 'gridded_snowline.nc'
    x, y = dem.x.values.astype('float64'), dem.y.values.astype('float64')
    dx = float(abs(x[1] - x[0]))
    ice_fraction = dem.ice_fraction.values.astype('float32')
    ice = dem.rgi_mask.values > 0.5
    periphery = dem.rgi_periphery_fraction.values > 0.5 if 'rgi_periphery_fraction' in dem else np.zeros_like(ice)
    target_crs = dem.spatial_ref.crs_wkt

    yrs = sorted(files)
    labels = np.full((len(yrs),) + ice.shape, -1, dtype='int8')
    total = np.zeros(ice.shape, 'float64')
    ever_classified = np.zeros(ice.shape, bool)
    all_pts = []
    bare_area = {}
    for i, yr in enumerate(yrs):
        polys, pts = rings(files[yr], target_crs)
        frac, depth = year_fraction(polys, x, y, ice_fraction)
        depth_cell = depth.reshape(ice.shape[0], SUB, ice.shape[1], SUB).max(axis=(1, 3))
        ever_classified |= depth_cell > 0
        labels[i][ice] = np.round(100 * frac[ice]).astype('int8')
        total[ice] += frac[ice]
        all_pts.append(pts)
        bare_area[yr] = float(np.nansum((1 - frac)[ice] * ice_fraction[ice]) * dx * dx / 1e6)
        print(f'{yr}: {len(polys)} rings ({len(pts)} vertices), max depth {int(depth.max())}, '
              f'bare ice {bare_area[yr]:.0f} km2, snow {float(np.nansum(frac[ice] * ice_fraction[ice]) * dx * dx / 1e6):.0f} km2', flush=True)
    n = len(yrs)

    # peripheral glaciers the product never drew a ring on or near: unclassified
    tree = cKDTree(np.concatenate(all_pts))
    cand = periphery & ice & ~ever_classified
    yy, xx = np.nonzero(cand)
    d, _ = tree.query(np.column_stack([x[xx], y[yy]]), workers=-1)
    unclassified = np.zeros(ice.shape, bool)
    unclassified[yy[d > unclassified_km * 1e3], xx[d > unclassified_km * 1e3]] = True
    glacier_fraction = np.where(unclassified, 0.0, ice_fraction).astype('float32')

    t0, t1 = float(yrs[0]), float(yrs[-1])
    time_attrs = dict(time_nominal=0.5 * (t0 + t1), time_start=t0, time_end=t1)
    snow_fraction = np.where(ice & ~unclassified, total / n, np.nan).astype('float32')
    out = xr.Dataset(coords={'y': dem.y, 'x': dem.x, 'year': np.array(yrs, dtype='int32')})
    out['snow_fraction'] = (('y', 'x'), snow_fraction)
    out['snow_fraction'].attrs = dict(long_name=f'fraction of the {yrs[0]}-{yrs[-1]} end-of-summer seasons with snow cover '
                                      'over the glacierized subarea (mean of the yearly labels)', units='1', **time_attrs)
    out['glacier_fraction'] = (('y', 'x'), glacier_fraction)
    out['glacier_fraction'].attrs = dict(long_name='ice fraction of the cell (gridded_dem ice_fraction); 0 where the product never classified the cell',
                                         units='1', **time_attrs)
    out['snow_label'] = (('year', 'y', 'x'), labels)
    out['snow_label'].attrs = dict(long_name='per-year snow fraction of the glacierized subarea x 100 (-1 off ice)', units='%')
    out['spatial_ref'] = dem['spatial_ref']
    for v in ('snow_fraction', 'glacier_fraction', 'snow_label'):
        out[v].attrs['grid_mapping'] = 'spatial_ref'
    out.attrs.update(snowline_source=str(snowline_dir.resolve()), product_crs=PRODUCT_CRS, n_years=n,
                     unclassified_periphery_cells=int(unclassified.sum()), unclassified_km=unclassified_km,
                     method='make_snowline.py: ring nesting depth rasterized at 500 m (snow = odd depth), '
                            'averaged to the cell and divided by ice_fraction',
                     bare_ice_area_km2=' '.join(f'{yr}:{a:.0f}' for yr, a in bare_area.items()))
    out.to_netcdf(output_path, encoding={'snow_label': dict(zlib=True, complevel=4)})
    print(f'wrote {output_path}: {n} years {yrs[0]}-{yrs[-1]}, mean bare-ice area '
          f'{np.mean(list(bare_area.values())):.0f} km2 of {float(np.nansum(ice_fraction[ice])) * dx * dx / 1e6:.0f} km2 ice; '
          f'{int(unclassified.sum())} unclassified peripheral cells excluded')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--domain-path', required=True)
    ap.add_argument('--snowline-dir', default=str(SNOWLINE_DIR))
    ap.add_argument('--years', type=int, nargs=2, default=None)
    ap.add_argument('--unclassified-km', type=float, default=UNCLASSIFIED_KM)
    a = ap.parse_args()
    build_snowline(a.domain_path, a.snowline_dir, a.years, a.unclassified_km)
