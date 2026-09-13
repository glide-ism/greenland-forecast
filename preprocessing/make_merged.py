"""Merge per-variable gridded NetCDFs into a single GLIDE input file.

Required: gridded_dem, gridded_velocity, gridded_insolation, gridded_climate.
Optional (merged when present, silently skipped otherwise — the inverse model
drops the corresponding loss term): gridded_snowline, gridded_dhdt,
gridded_climate_era5land, gridded_debris.

Output: {domain_path}/model_inputs/GLIDE_inputs.nc
"""
import argparse
import os
from pathlib import Path

import xarray as xr

REQUIRED = ('gridded_dem', 'gridded_velocity', 'gridded_insolation', 'gridded_climate')
OPTIONAL = ('gridded_snowline', 'gridded_dhdt', 'gridded_climate_era5land', 'gridded_debris')


def build_merged(domain_path: str) -> xr.Dataset:
    domain_path = Path(domain_path)
    inputs_dir = domain_path / 'model_inputs'
    output_path = inputs_dir / 'GLIDE_inputs.nc'

    def load(path):
        ds = xr.load_dataset(path)
        # rioxarray writes the CF grid mapping as a coordinate, the builders
        # that copy it from gridded_dem.nc carry it as a variable: normalise
        # to a coordinate so xr.merge does not have to guess.
        if 'spatial_ref' in ds.data_vars:
            ds = ds.set_coords('spatial_ref')
        return ds

    parts = []
    for name in REQUIRED:
        parts.append(load(inputs_dir / f'{name}.nc'))
    for name in OPTIONAL:
        path = inputs_dir / f'{name}.nc'
        if path.exists():
            parts.append(load(path))
        else:
            print(f"optional product {name} absent; skipping")
    merged = xr.merge(parts, compat='override', combine_attrs='drop_conflicts')

    for var in ('elevation', 'vx', 'rgi_mask', 'snow_fraction', 'dhdt'):
        if var in merged and 'time_nominal' not in merged[var].attrs:
            print(f"Warning: {var} lost its time attrs in the merge; "
                  f"the inverse model will fall back to t_end for it.")

    # Write-then-rename: netCDF4 truncates the target before the HDF5 lock
    # check, so a direct write onto an open GLIDE_inputs.nc leaves 0 bytes.
    tmp_path = output_path.with_suffix('.nc.tmp')
    merged.to_netcdf(tmp_path)
    os.replace(tmp_path, output_path)
    print(f"wrote {output_path}")
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-path", type=str, required=True)
    args = parser.parse_args()
    build_merged(args.domain_path)
