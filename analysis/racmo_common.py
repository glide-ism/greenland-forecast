"""Shared helpers for the RACMO2.3p2-ERA5 (ISMIP7 1 km forcing) analyses.

The RACMO files (`common_data/climate/RACMO2.3p2-ERA/{ts,pr}/..._{year}.nc`)
are already on the ISMIP 1 km grid that `domains/greenland` uses, but with
`y` ascending; everything here is returned on the model-input orientation
(`y` descending) so arrays line up with `GLIDE_inputs.nc`.

`tas` is RACMO's corrected 2 m air temperature (monthly mean, unclipped);
`ts` is the same field CLIPPED at 273.15 K (ISMIP7's "temperature at top of
ice sheet model"), so its summer ablation-zone values are censored at 0 degC.
`pr` is precipitation mass flux in kg m-2 s-1 (water equivalent).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
RACMO_BASE = ROOT / 'common_data' / 'climate' / 'RACMO2.3p2-ERA'
HADCRUT_PATH = (ROOT / 'common_data' / 'climate' / 'temp_anomaly'
                / 'HadCRUT.5.0.2.0.analysis.summary_series.global.annual.csv')
BASINS_PATH = ROOT / 'common_data' / 'area' / 'basins' / 'Greenland_Basins_PS_v1.4.2.shp'

T_MELT = 273.15
ICE_DENSITY = 917.0
DAYS_PER_YEAR = 365.0
SEC_PER_YEAR = DAYS_PER_YEAR * 86400.0
REGIONS = ['NO', 'NE', 'CE', 'SE', 'SW', 'CW', 'NW', 'periphery']
ELEV_BANDS = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 4000)]


TEMP_VARS = ('tas', 'ts')


def is_censored(var: str) -> bool:
    """True for the 0 degC-capped `ts`; `tas` is unclipped."""
    return var == 'ts'


def racmo_years(var: str = 'tas'):
    return sorted(int(p.stem[-4:]) for p in (RACMO_BASE / var).glob(f'{var}_*.nc'))


def racmo_path(var: str, year: int) -> Path:
    hits = sorted((RACMO_BASE / var).glob(f'{var}_*_{year}.nc'))
    if not hits:
        raise FileNotFoundError(f'no RACMO {var} file for {year}')
    return hits[0]


def load_racmo(var: str, year: int, grid: xr.Dataset) -> np.ndarray:
    """(12, ny, nx) float32 on the model grid orientation. tas/ts in degC,
    pr in kg m-2 s-1."""
    ds = xr.open_dataset(racmo_path(var, year))
    da = ds[var]
    if not (np.allclose(da.x.values, grid.x.values) and
            np.allclose(np.sort(da.y.values), np.sort(grid.y.values))):
        raise ValueError('RACMO grid does not match the domain grid')
    if da.y.values[0] < da.y.values[-1]:
        da = da.isel(y=slice(None, None, -1))
    a = da.values.astype('float32')
    if var in TEMP_VARS:
        a = a - T_MELT
    return a


def pr_to_m_ice_per_yr(pr):
    return pr * SEC_PER_YEAR / ICE_DENSITY


def load_hadcrut() -> pd.Series:
    """Annual global-mean anomaly (degC, rel. 1961-1990), indexed by year."""
    df = pd.read_csv(HADCRUT_PATH)
    return pd.Series(df['Anomaly (deg C)'].values, index=df['Time'].values.astype(int), name='G')


def load_grid(domain_path) -> xr.Dataset:
    p = Path(domain_path) / 'model_inputs' / 'GLIDE_inputs.nc'
    return xr.open_dataset(p)


def region_raster(grid: xr.Dataset) -> np.ndarray:
    """int8 (ny, nx): 1..7 = NO, NE, CE, SE, SW, CW, NW (Mouginot & Rignot
    basins), 8 = ice outside the basins (peripheral glaciers), 0 = no ice."""
    import geopandas as gpd
    from rasterio import features
    from rasterio.transform import from_origin
    g = gpd.read_file(BASINS_PATH).to_crs('EPSG:3413')
    x, y = grid.x.values, grid.y.values
    dx = float(abs(x[1] - x[0]))
    transform = from_origin(x.min() - dx / 2, y.max() + dx / 2, dx, dx)
    code = {r: i + 1 for i, r in enumerate(REGIONS[:7])}
    shapes = [(geom, code[r]) for geom, r in zip(g.geometry, g['SUBREGION1'])]
    ras = features.rasterize(shapes, out_shape=(len(y), len(x)), transform=transform,
                             fill=0, dtype='int16')
    ice = grid.rgi_mask.values > 0.5
    ras = np.where(ice & (ras == 0), 8, ras)
    ras = np.where(ice, ras, 0)
    return ras.astype('int8')


def region_means(field, regions, weights=None):
    """Mean of `field` (ny, nx) over each region code 1..8 plus the whole ice
    sheet ('GrIS' = codes 1..7) and all ice ('all'). NaNs ignored."""
    out = {}
    fin = np.isfinite(field)
    w = np.ones_like(field) if weights is None else weights
    for i, name in enumerate(REGIONS, start=1):
        sel = (regions == i) & fin
        out[name] = float((field[sel] * w[sel]).sum() / w[sel].sum()) if sel.any() else np.nan
    sel = (regions >= 1) & (regions <= 7) & fin
    out['GrIS'] = float((field[sel] * w[sel]).sum() / w[sel].sum())
    sel = (regions >= 1) & fin
    out['all'] = float((field[sel] * w[sel]).sum() / w[sel].sum())
    return out


def band_means(field, elevation, ice):
    out = {}
    fin = np.isfinite(field) & ice
    for lo, hi in ELEV_BANDS:
        sel = fin & (elevation >= lo) & (elevation < hi)
        out[f'{lo}-{hi}'] = float(np.mean(field[sel])) if sel.any() else np.nan
    return out


def extent(grid):
    x, y = grid.x.values / 1e3, grid.y.values / 1e3
    dx = abs(x[1] - x[0]) / 2
    return [x.min() - dx, x.max() + dx, y.min() - dx, y.max() + dx]
