#!/usr/bin/env python
"""
Front-pin masks with a Little Ice Age spin-up: the TermPicks yearly masks
(preprocessing/make_front_mask.py -> front_mask.nc) with pre-1972 masks
prepended from the GRISHM LIA maximum extent, for `ocean.PinnedFront`
(config `ocean_forcing.pin_front_filename`).

  model_inputs/front_mask_lia.nc   front_mask (time, y, x) uint8 (same layout as front_mask.nc),
                                   lia_mask (y, x), lia_retreat_year (y, x) (NaN outside the LIA-only zone)

GRISHM (Salmani et al., "Nearly half of Greenland's post-Little Ice Age
area loss occurred since 2000"; common_data/area/grishm/) is one polygon of
the maximum observable LIA extent of the ice sheet (trimlines, moraines,
the TermPicks most-extensive fronts, 1980s termini), EPSG:32624, attributed
to ~1850-1900 but ASYNCHRONOUS (earlier in the southwest, as late as ~1920
in parts of the north). It covers the ice sheet only, so peripheral
glaciers come from the TermPicks masks, and floating tongues it may omit
are kept by taking the UNION with the first TermPicks mask throughout.

Pre-1972 extent and its retreat:
  t <= --hold-until          LIA  U  M_1972                                      (the spin-up)
  --hold-until < t < 1972    M_1972 U {LIA-only cells with t < t_c}               (every --ramp-step years)
  t >= 1972                  the TermPicks masks, unchanged
with t_c = hold_until + (1 - phi) (t_first - hold_until) and phi = d_72 / (d_72 + d_LIA)
in the LIA-only zone (d_72: distance to M_1972, d_LIA: distance to outside the LIA
polygon), i.e. each front retreats at a uniform rate from its LIA position to its
1972 position. The distances are Euclidean on the grid (a zone cell nearer another
glacier's 1972 ice across a ridge is timed by that one; the zone is mostly narrow
fjord reaches, where this is immaterial). A pinned mask only PERMITS ice (margin
h0 = pin_h0_inside); whether the model's fronts actually advance to the LIA
position is up to its dynamics. Cells where the LIA polygon lies on land only
permit ice on land, which the pin's margins never act on.

Usage:
  python preprocessing/make_front_mask_lia.py --domain-path domains/greenland [--hold-until 1900] [--ramp-step 5]
"""
import argparse
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from rasterio import features
from rasterio.transform import from_origin
from scipy import ndimage

HERE = Path(__file__).resolve().parent
GRISHM = (HERE.parent / 'common_data' / 'area' / 'grishm' / 'GRISHM - LIA Ice Margin' / 'GRISHM - LIA Ice Margin'
          / 'GRISHM' / 'GRISHM_without_nunataks.shp')


def rasterize_fraction(geom, x, y, sub=4):
    """Area fraction of each (x, y) cell covered by `geom`, from a `sub`-times finer raster."""
    dx = float(x[1] - x[0]); dy = float(y[0] - y[1])           # y descending
    tr = from_origin(float(x[0]) - dx / 2, float(y[0]) + dy / 2, dx / sub, dy / sub)
    fine = features.rasterize([(g, 1) for g in geom], out_shape=(len(y) * sub, len(x) * sub),
                              transform=tr, fill=0, dtype='uint8')
    return fine.reshape(len(y), sub, len(x), sub).mean(axis=(1, 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--domain-path', default=str(HERE.parent / 'domains' / 'greenland'))
    ap.add_argument('--grishm', default=str(GRISHM))
    ap.add_argument('--front-mask', default='front_mask.nc', help='the TermPicks yearly masks (model_inputs/)')
    ap.add_argument('--hold-until', type=float, default=1900.0,
                    help='last year of the LIA extent; the retreat to the first TermPicks mask starts after it')
    ap.add_argument('--ramp-step', type=float, default=5.0, help='years between the masks of the retreat')
    ap.add_argument('--min-fraction', type=float, default=0.5, help='cell counts as LIA ice above this covered fraction')
    ap.add_argument('--out', default='front_mask_lia.nc')
    a = ap.parse_args()

    mi = Path(a.domain_path) / 'model_inputs'
    fm = xr.open_dataset(mi / a.front_mask)
    x, y = fm.x.values, fm.y.values
    years72 = fm.time.values.astype(float)
    M72 = fm.front_mask.isel(time=0).values > 0
    t_first = float(years72[0])
    if not a.hold_until < t_first:
        raise SystemExit(f'--hold-until {a.hold_until} must precede the first TermPicks year {t_first}')

    g = gpd.read_file(a.grishm).to_crs(3413)
    frac = rasterize_fraction(g.geometry.values, x, y)
    lia = frac >= a.min_fraction

    zone = lia & ~M72                                         # ice at the LIA maximum, not in 1972
    d72 = ndimage.distance_transform_edt(~M72)                # cells to the nearest 1972 ice
    dlia = ndimage.distance_transform_edt(lia)                # cells to the nearest non-LIA cell
    phi = np.where(zone, d72 / np.maximum(d72 + dlia, 1e-6), np.nan)
    t_c = a.hold_until + (1.0 - phi) * (t_first - a.hold_until)

    ramp = np.arange(a.hold_until, t_first - 1e-6, a.ramp_step)
    pre = []
    for t in ramp:                                            # the mask of year t: cells not yet lost
        pre.append(M72 | (zone & (t < t_c)) if t > a.hold_until else (M72 | lia))
    years = np.concatenate([ramp, years72])
    masks = np.concatenate([np.stack(pre).astype(np.uint8), fm.front_mask.values.astype(np.uint8)])

    # report: the LIA-only area, overall, marine and per region
    gi = xr.open_dataset(mi / 'GLIDE_inputs.nc')
    assert np.allclose(gi.x.values, x) and np.allclose(gi.y.values, y)
    bed = np.where(np.isfinite(gi.bed_obs.values), gi.bed_obs.values, gi.elevation.values)   # elevation = bed / bathymetry off the ice
    marine = zone & np.nan_to_num(bed < 0)
    dx_km2 = abs(float(x[1] - x[0])) * abs(float(y[1] - y[0])) / 1e6
    print(f'LIA polygon {frac.sum() * dx_km2:,.0f} km2 (GRISHM: {float(g.Area_km2.sum()):,.0f}); '
          f'LIA-only zone {zone.sum() * dx_km2:,.0f} km2, of which below sea level {marine.sum() * dx_km2:,.0f} km2; '
          f'1972 ice outside the LIA polygon {(M72 & ~lia).sum() * dx_km2:,.0f} km2')
    try:
        basins = gpd.read_file(HERE.parent / 'common_data' / 'area' / 'basins' / 'Greenland_Basins_PS_v1.4.2.shp')
        lab = gi.rgi_label.values.astype(float); lab = np.where(np.isfinite(lab), lab, -1).astype(int)
        reg_of = np.full(lab.max() + 2, '', dtype='U2'); reg_of[:len(basins)] = basins.SUBREGION1.astype(str).values
        # zone cells lie outside today's basins: label by the nearest labelled cell
        idx = ndimage.distance_transform_edt(lab < 0, return_distances=False, return_indices=True)
        reg = reg_of[lab[idx[0], idx[1]]]
        rows = [{'region': r, 'zone_km2': (zone & (reg == r)).sum() * dx_km2,
                 'marine_km2': (marine & (reg == r)).sum() * dx_km2} for r in ['NO', 'NE', 'CE', 'SE', 'SW', 'CW', 'NW']]
        print(pd.DataFrame(rows).round(0).to_string(index=False))
    except Exception as e:                                     # the report is optional
        print('regional report skipped:', e)

    ds = xr.Dataset(
        {'front_mask': (('time', 'y', 'x'), masks),
         'lia_mask': (('y', 'x'), lia.astype(np.uint8)),
         'lia_retreat_year': (('y', 'x'), t_c.astype(np.float32))},
        coords={'time': years, 'y': y, 'x': x})
    if 'spatial_ref' in fm:
        ds['spatial_ref'] = fm.spatial_ref
    ds.attrs.update(source=f'make_front_mask_lia.py {date.today().isoformat()}', grishm=str(a.grishm),
                    front_mask=str(mi / a.front_mask), hold_until=a.hold_until, ramp_step=a.ramp_step,
                    min_fraction=a.min_fraction, years=f'{years[0]:g}-{years[-1]:g}',
                    description='t <= hold_until: GRISHM LIA U first TermPicks mask; then a uniform-rate '
                                'distance ramp to the first TermPicks mask; from it the TermPicks masks')
    out = mi / a.out
    ds.to_netcdf(out, encoding={'front_mask': dict(zlib=True, complevel=4, chunksizes=(1, len(y), len(x))),
                                'lia_mask': dict(zlib=True), 'lia_retreat_year': dict(zlib=True)})
    print(f'wrote {out}: {len(years)} masks {years[0]:g}-{years[-1]:g}')


if __name__ == '__main__':
    main()
