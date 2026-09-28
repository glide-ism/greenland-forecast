#!/usr/bin/env python
"""
Push a rebuilt DEM (model_inputs/gridded_dem.nc) into the existing
GLIDE_inputs*.nc files without re-running make_all -- which, after
build_dem, re-runs velocity, dh/dt, snowline, insolation, the CARRA2
climatology, the anomalies and the ocean forcing before it merges.

Only the DEM-derived continuous fields are replaced (`elevation`, the
composite the model seeds its thickness from and the surface observation
targets; `topography`; `bathymetry`; `bed_obs`; `thickness_obs` -- the
masks, fractions, errors and radar-pick averages are untouched), in place
through netCDF4 so the multi-GB files are not rewritten; every file is
copied to <name>.bak first (an existing .bak is kept, so it stays the
ORIGINAL through repeated patches). The grids must match exactly. Use
after `make_dem.py --smooth-sigma-km ...`.

Usage:
  python tools/patch_dem_into_inputs.py --domain-path domains/greenland [--vars elevation topography] [--no-backup]
"""
import argparse
import shutil
from datetime import date
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--domain-path', default='domains/greenland')
    ap.add_argument('--vars', nargs='+', default=['elevation', 'topography', 'bathymetry', 'bed_obs', 'thickness_obs'])
    ap.add_argument('--no-backup', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    mi = Path(a.domain_path) / 'model_inputs'
    dem = xr.open_dataset(mi / 'gridded_dem.nc')
    targets = sorted(p for p in mi.glob('GLIDE_inputs*.nc') if not p.name.endswith('.bak'))
    print(f"DEM {mi / 'gridded_dem.nc'}: elevation smoothing = {dem.elevation.attrs.get('smoothing', 'none')}")
    for t in targets:
        with xr.open_dataset(t) as g:
            if not (np.allclose(g.x.values, dem.x.values) and np.allclose(g.y.values, dem.y.values)):
                print(f"  {t.name}: GRID DIFFERS, skipped"); continue
            diffs = {v: float(np.nanmax(np.abs(g[v].values - dem[v].values))) for v in a.vars if v in g}
        print(f"  {t.name}: max |change| " + ", ".join(f"{v} {d:.0f} m" for v, d in diffs.items()))
        if a.dry_run:
            continue
        if not a.no_backup:
            bak = t.with_suffix(t.suffix + '.bak')
            if not bak.exists():
                shutil.copy2(t, bak); print(f"    backup {bak.name}")
        with netCDF4.Dataset(t, 'r+') as nc:
            for v in a.vars:
                if v not in nc.variables:
                    print(f"    {v}: not in file, skipped"); continue
                nc[v][:] = dem[v].values
                for k, val in dem[v].attrs.items():
                    if isinstance(val, (str, int, float, np.floating, np.integer)):
                        nc[v].setncattr(k, val)
            nc.setncattr('dem_patched', f"{date.today().isoformat()}: {', '.join(a.vars)} from gridded_dem.nc "
                                        f"({dem.elevation.attrs.get('smoothing', 'none')})")
        print(f"    patched {', '.join(v for v in a.vars)}")


if __name__ == '__main__':
    main()
