"""Build the radar bed-pick point dataset for a Greenland domain (optional).

The Greenland bed is conditioned on the gridded BedMachine product by default
(make_dem.py writes bed_obs / bed_obs_err). This builder is the raw-radar
alternative for domains that want to condition directly on picks: Operation
IceBridge MCoRDS L2 (NSIDC IRMCR2; CSV per flight segment with LAT, LON,
TIME, THICK, ELEVATION, FRAME, SURFACE, BOTTOM, QUALITY) and the identically
formatted pre-IceBridge (BRMCR2) / CReSIS archive files.

Each file is projected into the domain CRS, resampled to a uniform along-track
spacing (the grid resolution), and the points inside the grid with a valid
bottom pick are written to a GeoPackage in the schema alaska-forecast's
BedSpec / bed conditioning consume (x, y, surface, bed, thk, date).

Output: {domain_path}/model_inputs/flightlines.gpkg
"""
import argparse
import datetime
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from scipy.interpolate import interp1d
from shapely.geometry import Point

from domain_grid import grid_from_dem

RADAR_DIRECTORIES = [Path('../common_data/flightlines/irmcr2'),
                     Path('../common_data/flightlines/brmcr2')]
MISSING = -9999.0
MAX_QUALITY = 3      # keep QUALITY <= this (1 = high confidence pick)


def _decimal_year_from_name(path: Path) -> float:
    m = re.search(r'(\d{4})(\d{2})(\d{2})', path.name)
    if not m:
        return np.nan
    d = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    y0 = datetime.datetime(d.year, 1, 1)
    return d.year + (d - y0).days / (datetime.datetime(d.year + 1, 1, 1) - y0).days


def build_flightlines(domain_path: str, radar_dirs=None) -> gpd.GeoDataFrame:
    domain_path = Path(domain_path)
    dem_path = domain_path / 'model_inputs' / 'gridded_dem.nc'
    output_path = domain_path / 'model_inputs' / 'flightlines.gpkg'
    radar_dirs = [Path(p) for p in radar_dirs] if radar_dirs else RADAR_DIRECTORIES

    dem = xr.load_dataset(dem_path)
    grid = grid_from_dem(dem)
    project = pyproj.Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    spacing = grid.resolution

    rows = []
    for directory in radar_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob('*.csv')):
            df = pd.read_csv(path)
            df.columns = [c.strip().upper() for c in df.columns]
            need = {'LAT', 'LON', 'SURFACE', 'BOTTOM'}
            if not need <= set(df.columns):
                continue
            ok = (df['BOTTOM'] != MISSING) & (df['SURFACE'] != MISSING)
            if 'QUALITY' in df:
                ok &= df['QUALITY'] <= MAX_QUALITY
            df = df[ok]
            if len(df) < 2:
                continue
            x, y = project.transform(df['LON'].to_numpy(), df['LAT'].to_numpy())
            inside = (x > grid.xmin) & (x < grid.xmax) & (y > grid.ymin) & (y < grid.ymax)
            if inside.sum() < 2:
                continue
            x, y = x[inside], y[inside]
            srf = df['SURFACE'].to_numpy(float)[inside]
            bot = df['BOTTOM'].to_numpy(float)[inside]
            step = np.hstack(([0.0], np.hypot(np.diff(x), np.diff(y)) + 1e-3))
            s = np.cumsum(step)
            ss = np.arange(s.min(), s.max(), spacing)
            if len(ss) == 0:
                continue
            date = _decimal_year_from_name(path)
            rows.append(pd.DataFrame({
                'x': interp1d(s, x)(ss), 'y': interp1d(s, y)(ss),
                'surface': interp1d(s, srf)(ss), 'bed': interp1d(s, bot)(ss),
                'date': np.full(ss.shape, date)}))
    if not rows:
        print("no radar picks found inside the domain; not writing flightlines.gpkg")
        return gpd.GeoDataFrame({'x': [], 'y': [], 'bed': []}, geometry=[])
    pts = pd.concat(rows, ignore_index=True)
    pts['thk'] = pts['surface'] - pts['bed']
    gdf = gpd.GeoDataFrame(pts, geometry=[Point(xy) for xy in zip(pts.x, pts.y)],
                           crs=grid.crs.to_wkt())
    gdf.to_file(output_path, driver='GPKG')
    print(f"wrote {output_path}: {len(gdf)} picks")
    return gdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-path", type=str, required=True)
    args = parser.parse_args()
    build_flightlines(args.domain_path)
