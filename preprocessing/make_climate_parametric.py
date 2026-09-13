"""PLACEHOLDER monthly climatology for a Greenland domain — no reanalysis needed.

For bootstrapping and smoke tests only; the science domains use CARRA
(make_pancarra_vars.py). Writes the same file / variables / units as the
CARRA builder so everything downstream is agnostic:

  * `monthly_t2m` (deg C): the Fausto et al. (2009, J. Glaciol. 55) Greenland
    parameterisation of mean-annual and July 2 m temperature in elevation,
    latitude and longitude, with a cosine seasonal cycle peaking in July;
  * `monthly_precip` (m ice eq. / yr, held constant through the year): where
    a MAR mean SMB is available in gridded_dem.nc (`smb_mar`), the
    precipitation implied by that SMB under a positive-degree-day melt
    estimate from the parametric temperatures (so the SMB model starts near
    MAR's balance); elsewhere a smooth elevation/latitude decay.

Output: {domain_path}/model_inputs/gridded_climate.nc (attrs mark it PLACEHOLDER)
"""
import argparse
from pathlib import Path

import numpy as np
import pyproj
import xarray as xr

from domain_grid import grid_from_dem

DDF_ICE_M_PER_PDD = 0.006     # m ice eq. per positive degree day (placeholder)
DAYS_PER_MONTH = 365.0 / 12.0
P_MIN, P_MAX = 0.05, 4.0      # m ice eq. / yr


def fausto_2009_t2m(z, lat, lon_east):
    """(T_annual, T_july) in deg C. Longitude enters as degrees east
    (negative over Greenland): that sign reproduces Summit's July mean
    (~-12 C) and warmer south-coast values; the opposite sign is ~4 C too
    cold everywhere. Placeholder — see the module docstring."""
    lam = lon_east
    t_ma = 41.83 - 6.309e-3 * z - 0.7189 * lat - 0.0672 * lam
    t_jul = 14.70 - 5.426e-3 * z - 0.1585 * lat - 0.0518 * lam
    return t_ma, t_jul


def build_climate_parametric(domain_path: str) -> xr.Dataset:
    domain_path = Path(domain_path)
    dem = xr.load_dataset(domain_path / 'model_inputs' / 'gridded_dem.nc')
    output_path = domain_path / 'model_inputs' / 'gridded_climate.nc'
    grid = grid_from_dem(dem)

    z = np.maximum(dem.elevation.values.astype('float64'), 0.0)
    gx, gy = np.meshgrid(dem.x.values.astype('float64'), dem.y.values.astype('float64'))
    lon, lat = pyproj.Transformer.from_crs(grid.crs, 'EPSG:4326', always_xy=True).transform(gx, gy)
    t_ma, t_jul = fausto_2009_t2m(z, lat, lon)
    months = np.arange(12)
    phase = np.cos(2.0 * np.pi * (months - 6.5) / 12.0)    # +1 in July, -1 in January
    t2m = (t_ma[None] + (t_jul - t_ma)[None] * phase[:, None, None]).astype('float32')

    pdd = (np.maximum(t2m, 0.0) * DAYS_PER_MONTH).sum(axis=0)
    melt = DDF_ICE_M_PER_PDD * pdd
    if 'smb_mar' in dem:
        smb = dem.smb_mar.values.astype('float64')
        valid = np.isfinite(smb) & (smb > -5.0)
        p_ann = np.where(valid, smb + melt, np.nan)
    else:
        p_ann = np.full(z.shape, np.nan)
    fallback = 1.5 * np.exp(-z / 1500.0) * np.exp(-(lat - 60.0) / 15.0)
    p_ann = np.where(np.isfinite(p_ann), p_ann, fallback)
    p_ann = np.clip(p_ann, P_MIN, P_MAX)
    precip = np.repeat(p_ann[None].astype('float32'), 12, axis=0)

    coords = {"t": (np.arange(12, dtype=np.float32) / 12), "y": dem.y, "x": dem.x}
    dims = ['t', 'y', 'x']
    ds = xr.Dataset(coords=coords)
    ds['spatial_ref'] = dem['spatial_ref']
    ds['monthly_t2m'] = xr.DataArray(t2m, dims=dims, coords=coords, attrs={
        'units': 'Deg C',
        'long_name': 'PLACEHOLDER monthly 2 m temperature, Fausto et al. (2009) parameterisation',
        'source': 'parametric (Fausto 2009); replace with CARRA (make_pancarra_vars.py)'})
    ds['monthly_precip'] = xr.DataArray(precip, dims=dims, coords=coords, attrs={
        'units': 'm ice equivalent / yr',
        'long_name': 'PLACEHOLDER precipitation rate consistent with MAR mean SMB under a PDD melt estimate',
        'source': 'parametric; replace with CARRA (make_pancarra_vars.py)',
        'ddf_m_per_pdd': DDF_ICE_M_PER_PDD})
    ds.attrs['climate_source'] = 'PLACEHOLDER parametric climatology (bootstrap only)'
    ds.to_netcdf(output_path)
    print(f"wrote {output_path} (PLACEHOLDER climatology): T_ann {np.nanmin(t_ma):.1f}..{np.nanmax(t_ma):.1f} C, "
          f"P {p_ann.min():.2f}..{p_ann.max():.2f} m/yr")
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-path", type=str, required=True)
    args = parser.parse_args()
    build_climate_parametric(args.domain_path)
