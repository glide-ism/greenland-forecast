"""Build the gridded monthly solar potential fields for a domain.

Uses gtic's SolarPotential to compute terrain-corrected direct-beam insolation
accounting for slope, aspect, self-shadowing and clear-sky attenuation from the
DEM (the ice surface over the ice sheet — flat-ish, so the terms matter mostly
along the margins and for the peripheral glaciers), decomposes the diurnal cycle into mean/cos/sin Fourier modes per month, and
adds the monthly diffuse-sky potential (isotropic-sky view factor of the tilted,
horizon-limited surface times the monthly-mean positive cos of the solar zenith).

Output: {domain_path}/model_inputs/gridded_insolation.nc

`--diffuse-only` opens an existing gridded_insolation.nc and adds/replaces just
the diffuse potential (36 horizon ray-traces instead of the 8760-position direct
loop), for domains whose direct product already exists.
"""
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from gtic import SolarPotential

from domain_grid import grid_from_dem

# Greenland spans four hours of solar time; gtic anchors the solar geometry
# at one (lat, lon) — the grid centre — in this timezone. The hour-angle
# error at the domain edges is a few degrees for an ice-sheet-wide domain
# (see README: split large domains or extend gtic to per-column longitude).
TIMEZONE = "America/Nuuk"


def _domain_centroid_latlon(domain_path: Path, dem: xr.Dataset) -> tuple[float, float]:
    """(latitude, longitude) of the grid centre in EPSG:4326 degrees."""
    return grid_from_dem(dem).centroid_latlon()


DIFFUSE_N_AZIMUTH = 36


def _diffuse_dataarray(solar: SolarPotential, year: int, dims, coords) -> xr.DataArray:
    """Monthly diffuse-sky potential on the DEM grid (SolarPotential.diffuse_potential_monthly)."""
    dif = solar.diffuse_potential_monthly(year, n_azimuth=DIFFUSE_N_AZIMUTH)
    return xr.DataArray(
        dif.get(), dims=dims, coords=coords,
        attrs={
            "units": ("dimensionless (isotropic-sky diffuse irradiance on the tilted, "
                      "horizon-limited surface relative to the unobstructed horizontal "
                      "diffuse flux at unit cos zenith)"),
            "long_name": ("Monthly diffuse-sky potential: sky-view factor of the tilted, "
                          "horizon-limited surface x monthly-mean max(cos solar zenith, 0)"),
            "n_azimuth": DIFFUSE_N_AZIMUTH,
        },
    )


def _solar_for_domain(domain_path: Path, dem: xr.Dataset) -> tuple:
    latitude, longitude = _domain_centroid_latlon(domain_path, dem)
    grid_resolution = grid_from_dem(dem).resolution
    solar = SolarPotential(
        dem=dem,
        latitude=latitude,
        longitude=longitude,
        grid_resolution=grid_resolution,
        timezone=TIMEZONE,
        clearsky_transmittance=0.7
    )
    return solar, latitude, longitude


def add_diffuse_potential(domain_path: str, year: int) -> xr.Dataset:
    """Add/replace `monthly_diffuse_potential` in an existing gridded_insolation.nc."""
    domain_path = Path(domain_path)
    dem = xr.load_dataset(domain_path / 'model_inputs' / 'gridded_dem.nc')
    output_path = domain_path / 'model_inputs' / 'gridded_insolation.nc'
    insolation_ds = xr.load_dataset(output_path)   # eager load: closed before the rewrite
    solar, _, _ = _solar_for_domain(domain_path, dem)
    months = np.arange(0, 12, dtype=np.float32) / 12
    coords = {"t": months, "y": dem.y, "x": dem.x}
    insolation_ds["monthly_diffuse_potential"] = _diffuse_dataarray(
        solar, year, ["t", "y", "x"], coords)
    insolation_ds.to_netcdf(output_path)
    return insolation_ds


def build_insolation(domain_path: str, year: int) -> xr.Dataset:
    """Build the gridded insolation dataset for `domain_path` and write to disk."""
    domain_path = Path(domain_path)
    dem_path = domain_path / 'model_inputs' / 'gridded_dem.nc'
    output_path = domain_path / 'model_inputs' / 'gridded_insolation.nc'

    dem = xr.load_dataset(dem_path)
    solar, latitude, longitude = _solar_for_domain(domain_path, dem)
    mean, cos_mode, sin_mode = solar.potential_fourier(year)

    months = np.arange(0, 12, dtype=np.float32) / 12
    coords = {"t": months, "y": dem.y, "x": dem.x}
    dims = ["t", "y", "x"]

    mean_da = xr.DataArray(
        mean.get(), dims=dims, coords=coords,
        attrs={
            "units": "dimensionless (intensity relative to continuous orthogonal sunlight)",
            "long_name": "Monthly average daily solar potential (incidence-weighted, shadow-masked)",
        },
    )
    cos_da = xr.DataArray(
        cos_mode.get(), dims=dims, coords=coords,
        attrs={
            "units": "dimensionless (intensity relative to continuous orthogonal sunlight)",
            "long_name": "cos mode of diurnal variability in insolation",
        },
    )
    sin_da = xr.DataArray(
        sin_mode.get(), dims=dims, coords=coords,
        attrs={
            "units": "dimensionless (intensity relative to continuous orthogonal sunlight)",
            "long_name": "sin mode of diurnal variability in insolation",
        },
    )

    # Carry only the projection metadata from the DEM; downstream merge
    # provides the elevation and mask fields.
    insolation_ds = xr.Dataset(coords={"y": dem.y, "x": dem.x})
    insolation_ds["spatial_ref"] = dem["spatial_ref"]
    insolation_ds["monthly_solar_potential_mean"] = mean_da
    insolation_ds["monthly_solar_potential_cos"] = cos_da
    insolation_ds["monthly_solar_potential_sin"] = sin_da
    insolation_ds["monthly_diffuse_potential"] = _diffuse_dataarray(solar, year, dims, coords)
    insolation_ds.attrs["source"] = f"gtic SolarPotential, year {year}"
    insolation_ds.attrs["centroid_lat"] = latitude
    insolation_ds.attrs["centroid_lon"] = longitude
    insolation_ds.attrs["grid_resolution"] = f"{grid_from_dem(dem).resolution} m"

    insolation_ds.to_netcdf(output_path)
    return insolation_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-path", type=str, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--diffuse-only", action="store_true",
                        help="add/replace only monthly_diffuse_potential in the existing file")
    args = parser.parse_args()
    if args.diffuse_only:
        add_diffuse_potential(args.domain_path, args.year)
    else:
        build_insolation(args.domain_path, args.year)
