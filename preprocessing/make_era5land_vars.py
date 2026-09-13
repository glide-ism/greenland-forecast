"""Build the gridded monthly climatology (T2M, precip) for a domain from ERA5-Land.

Companion to make_pancarra_vars.py for source intercomparison: same physics
and units as the CARRA builder (deg C; m ice equivalent / yr; dims (t, y, x)
with t = month/12), but sourced from monthly ERA5-Land reanalysis and written
to its own file with `_era5land`-suffixed variables so both climatologies can
live side by side in the merged dataset.

ERA5-Land (0.1 deg, land + ice sheet) is the optional intercomparison
climatology; the CARRA2 product is the one the model runs on. The Greenland
file must be downloaded separately (CDS; see DATA_MANIFEST.md) — the
alaska-forecast bundle's era5.nc is an Alaska subset.

ERA5-Land ships no orography, so the CARRA lapse correction
(lapse * (z_dem - z_model_orog)) is reproduced with a smoothed-DEM proxy:
the project DEM is low-pass filtered to ~ERA5-Land's grid scale to stand in
for the model surface, t2m is reduced to that surface, then a 6.5 K/km lapse
rate is re-applied to the full-resolution DEM. The same smoothed-DEM surface
anchors the optional elevation-based precip scaling (percent per metre).

Output: {domain_path}/model_inputs/gridded_climate_era5land.nc
"""
import argparse
from pathlib import Path

import numpy as np
import pyproj
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from projection_dictionary import crs


ERA5LAND_PATH = Path('../common_data/climate/era5land/era5land_greenland.nc')

TEMPERATURE_LAPSE_RATE_K_PER_M = -0.0065
PRECIP_ELEVATION_FACTOR_PCT_PER_M = 0.0
ICE_DENSITY = 917.0
WATER_DENSITY = 1000.0
DAYS_PER_YEAR = 365

# ERA5-Land native grid spacing (0.1 deg) in metres; used as the smoothing
# length that approximates the model surface the t2m field "sees".
ERA5LAND_GRID_M = 0.1 * 111320.0


def _interpolate_to_grid(field_lat_lon, lat_asc, lon, lat_q, lon_q, method):
    """Interpolate a (lat, lon) field onto query points given in lat/lon."""
    interpolant = RegularGridInterpolator(
        (lat_asc, lon), field_lat_lon, method=method,
        bounds_error=False, fill_value=None,
    )
    return interpolant((lat_q, lon_q))


def build_climate(domain_path: str, year: int,
                  precip_elevation_factor: float =
                      PRECIP_ELEVATION_FACTOR_PCT_PER_M) -> xr.Dataset:
    """Build the gridded climate dataset for `domain_path`, `year` and write to disk.

    `precip_elevation_factor`: elevation-based precip scaling in % per metre,
    applied as P * max(0, 1 + 0.01*factor*(elevation - z_model)) about the
    smoothed-DEM model surface (same reference as the temperature lapse
    correction). 0 disables it. Overlaps with the inverse's learnable
    depletion ramp (GlacierConfig.precip_lapse_enabled) — don't enable both.
    """
    domain_path = Path(domain_path)
    dem_path = domain_path / 'model_inputs' / 'gridded_dem.nc'
    output_path = domain_path / 'model_inputs' / 'gridded_climate_era5land.nc'

    dem = xr.load_dataset(dem_path)
    elevation = dem.elevation.values.astype('float64')

    era5 = xr.open_dataset(ERA5LAND_PATH)
    era5 = era5.sel(valid_time=era5.valid_time.dt.year == year)
    if era5.sizes['valid_time'] != 12:
        raise ValueError(
            f"Expected 12 monthly ERA5-Land steps for {year}, "
            f"got {era5.sizes['valid_time']}. Available years: "
            f"{sorted(set(xr.open_dataset(ERA5LAND_PATH).valid_time.dt.year.values))}."
        )

    # ERA5-Land latitude is stored descending; RegularGridInterpolator needs
    # strictly increasing axes, so flip onto an ascending latitude axis.
    lat_asc = era5.latitude.values[::-1]
    lon = era5.longitude.values
    t2m = era5.t2m.values[:, ::-1, :]   # (month, lat, lon), K
    tp = era5.tp.values[:, ::-1, :]     # (month, lat, lon), m water / day

    # DEM grid -> ERA5-Land lon/lat query points.
    grid_x, grid_y = np.meshgrid(dem.x.values, dem.y.values)
    to_latlon = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon_q, lat_q = to_latlon.transform(grid_x, grid_y)

    # Smoothed-DEM proxy for the ERA5-Land model surface.
    grid_res_m = float(abs(dem.x.values[1] - dem.x.values[0]))
    sigma_px = ERA5LAND_GRID_M / grid_res_m
    z_model = gaussian_filter(elevation, sigma=sigma_px, mode='nearest')

    print("Working on t2m fields")
    # Match make_pancarra_vars exactly: K-273, then + lapse*(z_dem - z_model).
    t2m_fields = np.stack(
        [
            _interpolate_to_grid(t2m[i], lat_asc, lon, lat_q, lon_q, 'linear')
            - 273
            + TEMPERATURE_LAPSE_RATE_K_PER_M * (elevation - z_model)
            for i in range(12)
        ],
        axis=0,
    ).astype('float32')

    print("Working on precip fields")
    # ERA5-Land monthly tp is mean daily accumulation (m water / day).
    # -> m ice equivalent / yr to match the CARRA output units.
    water_to_ice = WATER_DENSITY / ICE_DENSITY
    precip_fields = np.stack(
        [
            _interpolate_to_grid(tp[i], lat_asc, lon, lat_q, lon_q, 'linear')
            * DAYS_PER_YEAR * water_to_ice
            for i in range(12)
        ],
        axis=0,
    ).astype('float32')

    if precip_elevation_factor:
        print(f"Applying elevation-based precip scaling "
              f"({precip_elevation_factor} %/m about the smoothed-DEM surface)")
        factor = np.maximum(
            1.0 + 0.01 * precip_elevation_factor * (elevation - z_model), 0.0)
        precip_fields = precip_fields * factor[None].astype('float32')

    months = np.arange(0, 12, dtype=np.float32) / 12
    coords = {"t": months, "y": dem.y, "x": dem.x}
    dims = ['t', 'y', 'x']

    t2m_da = xr.DataArray(
        t2m_fields, dims=dims, coords=coords,
        attrs={
            "units": "Deg C",
            "long_name": "Monthly average temperatures derived from ERA5-Land",
        },
    )
    precip_da = xr.DataArray(
        precip_fields, dims=dims, coords=coords,
        attrs={
            "units": "m ice equivalent / yr",
            "long_name": "Precipitation rate derived from ERA5-Land at monthly time steps",
            "precip_elevation_factor_pct_per_m": float(precip_elevation_factor),
        },
    )

    climate_ds = xr.Dataset(coords={"y": dem.y, "x": dem.x})
    climate_ds["spatial_ref"] = dem["spatial_ref"]
    climate_ds["monthly_t2m_era5land"] = t2m_da
    climate_ds["monthly_precip_era5land"] = precip_da
    climate_ds.attrs["climate_source"] = f"ERA5-Land, year {year}"

    climate_ds.to_netcdf(output_path)
    return climate_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-path", type=str, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--precip-elevation-factor", type=float,
                        default=PRECIP_ELEVATION_FACTOR_PCT_PER_M,
                        help="Elevation-based precip scaling in %% per metre, "
                             "P * max(0, 1 + 0.01*factor*(z - z_model)). "
                             "0 disables it.")
    args = parser.parse_args()
    build_climate(args.domain_path, args.year,
                  precip_elevation_factor=args.precip_elevation_factor)
