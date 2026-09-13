"""Bootstrap a runnable Greenland domain from glide's example input file.

glide ships `GLIDE_greenland_inputs.h5` (900 m, EPSG:3413): BedMachine v5
bed / surface / thickness, the MEaSUREs multi-year velocity mosaic
(NSIDC-0670) and a MAR 1979-2019 mean SMB. Together with the global
temperature-anomaly record from the alaska-forecast bundle and a PLACEHOLDER
parametric climatology (make_climate_parametric.py; the bundle's CARRA2 file
is an Alaska-only subset) that is enough
to stand up the whole inverse pipeline on real Greenland geometry BEFORE the
full observational bundle (DATA_MANIFEST.md) is in place — no dh/dt, no
snowline, a constant bed error and one basin label, so it is a smoke/
development domain, not a science configuration.

Writes gridded_dem.nc and gridded_velocity.nc in the make_dem/make_velocity
schema (block-averaged by --factor; 2 -> 1800 m), then runs the insolation,
CARRA2 climate, temperature-anomaly and merge builders.

    python preprocessing/bootstrap_from_glide_example.py \
        --domain-path domains/greenland_coarse --factor 2 [--skip-insolation]
"""
import argparse
from pathlib import Path

import numpy as np
import pyproj
import xarray as xr

from domain_grid import DomainGrid, write_grid_metadata
from make_dem import BEDMACHINE_SURFACE_TIME_ATTRS, MASK_TIME_ATTRS
from make_velocity import SOURCES as VELOCITY_SOURCES

H5_PATH = Path('../../glide/data/GLIDE_greenland_inputs.h5')
BED_ERR_M = 100.0     # the h5 carries no errbed; a flat 100 m stands in
RHO_I, RHO_W = 917.0, 1028.0


def _block(a, f, how='mean'):
    ny, nx = a.shape[0] // f * f, a.shape[1] // f * f
    b = a[:ny, :nx].reshape(ny // f, f, nx // f, f)
    return b.mean(axis=(1, 3)) if how == 'mean' else b


def bootstrap(domain_path: str, factor: int = 2, h5_path: str = None,
              skip_insolation: bool = False, year: int = 2015,
              climate: str = "auto") -> None:
    domain_path = Path(domain_path)
    inputs = domain_path / 'model_inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    h5_path = Path(h5_path) if h5_path else H5_PATH
    ds = xr.open_dataset(h5_path, engine='h5netcdf').load()

    f = int(factor)
    x = _block(ds.x.values.astype('float64')[None, :].repeat(f, 0), f)[0]
    y = _block(ds.y.values.astype('float64')[:, None].repeat(f, 1), f)[:, 0]
    res = float(abs(x[1] - x[0]))
    if y[0] < y[-1]:            # north-up convention: y descending
        y = y[::-1]
        flip = True
    else:
        flip = False

    def field(name, how='mean'):
        a = ds[name].values.astype('float64')
        a = np.nan_to_num(a) if name != 'thickness' else np.nan_to_num(a)
        out = _block(a, f, how)
        return out[::-1] if flip else out

    bed, surface, thk = field('bed'), field('surface'), field('thickness')
    smb = field('smb')
    ice_fraction = (_block((np.nan_to_num(ds.thickness.values) > 1.0).astype('float64'), f))
    ice_fraction = ice_fraction[::-1] if flip else ice_fraction
    ice = ice_fraction > 0.5
    # Floating where the hydrostatic base sits above the bed.
    floating = ice & ((surface - thk) > bed + 5.0)
    ocean = (~ice) & (bed < 0.0) & (surface <= 0.5)
    elevation = np.where(ice, surface, bed).astype('float32')

    grid = DomainGrid(crs=pyproj.CRS('EPSG:3413'), resolution=res,
                      xmin=float(x.min() - res / 2), ymin=float(y.min() - res / 2),
                      xmax=float(x.max() + res / 2), ymax=float(y.max() + res / 2))
    print(f"bootstrap grid: {grid.describe()}")
    dem = xr.Dataset(coords={'y': y, 'x': x})
    dem['topography'] = (('y', 'x'), np.where(ocean, np.nan, elevation).astype('float32'))
    dem['bathymetry'] = (('y', 'x'), np.where(ocean, bed, np.nan).astype('float32'))
    dem['elevation'] = (('y', 'x'), elevation)
    dem['bathymetry_mask'] = (('y', 'x'), ocean)
    dem['domain_mask'] = (('y', 'x'), np.ones_like(ice))
    dem['rgi_mask'] = (('y', 'x'), ice)
    dem['ice_fraction'] = (('y', 'x'), ice_fraction.astype('float32'))
    dem['floating_mask'] = (('y', 'x'), floating)
    dem['thickness_obs'] = (('y', 'x'), thk.astype('float32'))
    dem['bed_obs'] = (('y', 'x'), np.where(ice, bed, np.nan).astype('float32'))
    dem['bed_obs_err'] = (('y', 'x'), np.where(ice, BED_ERR_M, np.nan).astype('float32'))
    dem['smb_mar'] = (('y', 'x'), smb.astype('float32'))
    dem['smb_mar'].attrs.update(units='m ice yr-1',
                                source='MAR 1979-2019 mean SMB (from GLIDE_greenland_inputs.h5)')
    for name in ('bathymetry_mask', 'domain_mask', 'rgi_mask', 'floating_mask'):
        dem[name].attrs['_FillValue'] = False
    surf_attrs = dict(BEDMACHINE_SURFACE_TIME_ATTRS,
                      source='BedMachine v5 surface via GLIDE_greenland_inputs.h5')
    dem['elevation'].attrs.update(surf_attrs, units='m')
    dem['topography'].attrs.update(surf_attrs, units='m')
    dem['bed_obs'].attrs.update(units='m', source='BedMachine v5 bed via h5')
    dem['bed_obs_err'].attrs.update(units='m', source=f'flat {BED_ERR_M} m placeholder (no errbed in h5)')
    for name in ('rgi_mask', 'floating_mask', 'ice_fraction', 'thickness_obs'):
        dem[name].attrs.update(MASK_TIME_ATTRS)
    labels = np.where(ice, 0, -1).astype('int32')
    dem['rgi_label'] = (('y', 'x'), labels)
    dem['rgi_label'].attrs.update(MASK_TIME_ATTRS, _FillValue=-1)
    dem['rgi_id'] = xr.DataArray(np.array(['greenland_ice_sheet']).astype('U'), dims=('glacier',),
                                 coords={'glacier': np.arange(1, dtype='int32')})
    dem['surge_type'] = xr.DataArray(np.zeros(1, dtype='int32'), dims=('glacier',),
                                     coords={'glacier': np.arange(1, dtype='int32')})
    dem['surge_type'].attrs.update(MASK_TIME_ATTRS)
    dem = write_grid_metadata(dem, grid)
    dem.attrs.update(title='GLIDE Greenland geometry (bootstrap from glide example h5)',
                     geometry_source=str(h5_path))
    dem['x'] = dem['x'].astype('float32')
    dem['y'] = dem['y'].astype('float32')
    dem.to_netcdf(inputs / 'gridded_dem.nc')
    print(f"wrote gridded_dem.nc: {int(ice.sum())} ice cells, {int(floating.sum())} floating")

    vx, vy = field('vx'), field('vy')
    speed = np.hypot(vx, vy)
    vmask = (speed > 1.0).astype('float32')
    vel = xr.Dataset(coords={'y': dem.y, 'x': dem.x})
    vel['vx'] = (('y', 'x'), vx.astype('float32'))
    vel['vy'] = (('y', 'x'), vy.astype('float32'))
    vel['vmask'] = (('y', 'x'), vmask)
    vel['vx_err'] = (('y', 'x'), np.full(vx.shape, np.nan, dtype='float32'))
    vel['vy_err'] = (('y', 'x'), np.full(vx.shape, np.nan, dtype='float32'))
    spec = VELOCITY_SOURCES['measures_multiyear']
    for name in ('vx', 'vy', 'vmask', 'vx_err', 'vy_err'):
        vel[name].attrs.update(spec['time'], source=spec['description'] + ' via h5')
    vel['spatial_ref'] = dem['spatial_ref']
    vel.to_netcdf(inputs / 'gridded_velocity.nc')
    print(f"wrote gridded_velocity.nc: {int(vmask.sum())} moving cells")

    from make_carra_vars import build_climate as build_climate_carra
    from make_temperature_anomaly import build_temperature_anomaly
    from make_merged import build_merged
    if not skip_insolation:
        from make_insolation import build_insolation
        print("=== insolation ===", flush=True)
        build_insolation(str(domain_path), year=year)
    climate_done = False
    if climate != "parametric":
        print("=== CARRA2 climate ===", flush=True)
        try:
            clim = build_climate_carra(str(domain_path))
            climate_done = bool(np.isfinite(clim.monthly_t2m.values).any())
        except (FileNotFoundError, ValueError) as e:
            print(f"CARRA2 height-level files unavailable ({e}); see DATA_MANIFEST.md #5")
    if not climate_done:
        from make_climate_parametric import build_climate_parametric
        print("=== PLACEHOLDER parametric climate ===", flush=True)
        build_climate_parametric(str(domain_path))
    print("=== temperature anomaly ===", flush=True)
    build_temperature_anomaly(str(domain_path))
    print("=== merge ===", flush=True)
    build_merged(str(domain_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-path", type=str, required=True)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--h5", type=str, default=None)
    parser.add_argument("--year", type=int, default=2015)
    parser.add_argument("--skip-insolation", action="store_true")
    parser.add_argument("--climate", choices=("auto", "carra", "parametric"), default="auto",
                        help="auto: CARRA2 if it covers the grid, else the parametric placeholder")
    args = parser.parse_args()
    bootstrap(args.domain_path, args.factor, args.h5, args.skip_insolation, args.year,
              args.climate)
