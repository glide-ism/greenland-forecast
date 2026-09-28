#!/usr/bin/env python
"""
Calving basins: every marine cell assigned to the glacier basin it belongs
to, for a per-glacier calving margin (h0_i = c_i + alpha dTF_i) and for
per-glacier fit metrics over a sweep of forward runs.

  model_inputs/calving_basins.nc   calving_basin (int32, -1 none), basin_dist_km,
                                   calving_reach (int32), front_dist_km (signed)
  model_inputs/calving_basins.csv  one row per basin: name, region, gates, cell counts

On the ice the basin is the gridded `rgi_label` (the 260 Mouginot & Rignot
basins in shapefile order, then the RGI peripheral glaciers -- the same
names Mankoff's gates carry in `Mouginot_2019`). Over the ocean it is an
INDEX-AWARE FLOOD FILL: a multi-source breadth-first search from every ice
cell through the ice-free cells with bed < 0 (4-connected, so a one-cell
land bridge is not crossed), first arrival wins, i.e. the geodesic Voronoi
of the basins on the marine domain. Euclidean nearest-neighbour would cross
peninsulas; this follows the fjords. Cells farther than `--max-km` along
the water from any ice (open ocean), unconnected hollows and ice-free land
above sea level stay -1: a margin there is inert (nothing floats on land,
and ice cannot reach an unconnected hollow). `basin_dist_km` is the BFS
distance (cells x dx) on the ocean, 0 on the ice.

The same flood runs INWARD: from every terminus cell (ice 4-adjacent to
water) through the ice cells with bed < 0, i.e. along the fjord's trough
under the ice -- the domain the thermal-forcing marine fill covers -- up
to `--max-km`. `calving_reach` labels that reach together with the
flooded water (-1 elsewhere): the cells a change of the margin can move,
per glacier, whereas `calving_basin` on the ice is the whole drainage
basin up to the divide. `front_dist_km` is the signed distance along the
water / the trough from the terminus: positive outward, negative under
the ice, 0 at the terminus cells, NaN off the reach.

Usage:
  python preprocessing/make_calving_basins.py --domain-path domains/greenland [--max-km 60]
"""
import argparse
import json
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

HERE = Path(__file__).resolve().parent
BASINS_SHP = HERE.parent / 'common_data' / 'area' / 'basins' / 'Greenland_Basins_PS_v1.4.2.shp'
GATES = HERE.parent / 'common_data' / 'dhdt' / 'mankoff' / 'dataverse_files' / 'gates.gpkg'


def flood_labels(label, domain, max_rounds):
    """Multi-source BFS: propagate `label` (>= 0 where known) into `domain`
    cells, 4-connected, one ring per round; returns (label, dist in rounds,
    -1 where unreached)."""
    lab = label.copy()
    dist = np.where(lab >= 0, 0, -1).astype(np.int32)
    for r in range(1, max_rounds + 1):
        cand = domain & (lab < 0)
        if not cand.any():
            break
        p = np.pad(lab, 1, constant_values=-1)
        nb = (p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:])          # N S W E
        new = np.full(lab.shape, -1, lab.dtype)
        for n in reversed(nb):                                              # first in N,S,W,E order wins
            new = np.where(n >= 0, n, new)
        take = cand & (new >= 0)
        if not take.any():
            break
        lab[take] = new[take]
        dist[take] = r
    return lab, dist


def flood_primacy(label, domain, seeds_by_rank, max_rounds):
    """Sequential claim: basins in the given order (decreasing flux) each
    flood the still-unclaimed `domain` cells reachable within `max_rounds`
    from their seed cells (4-connected), so a fjord belongs to the largest
    glacier that can reach it. Windowed per basin for speed. Returns
    (label, dist)."""
    lab = label.copy()
    dist = np.where(lab >= 0, 0, -1).astype(np.int32)
    ny, nx = lab.shape
    for b, (iy, ix) in seeds_by_rank:
        y0, y1 = max(iy.min() - max_rounds - 1, 0), min(iy.max() + max_rounds + 2, ny)
        x0, x1 = max(ix.min() - max_rounds - 1, 0), min(ix.max() + max_rounds + 2, nx)
        sub = lab[y0:y1, x0:x1]; dom = domain[y0:y1, x0:x1]; dsub = dist[y0:y1, x0:x1]
        front = np.zeros(sub.shape, bool); front[iy - y0, ix - x0] = True
        claimed = np.zeros(sub.shape, bool)
        for r in range(1, max_rounds + 1):
            p = np.pad(front, 1)
            nxt = (p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]) & dom & (sub < 0) & ~claimed
            if not nxt.any():
                break
            claimed |= nxt; dsub[nxt] = r; front = nxt
        sub[claimed] = b
    return lab, dist


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--domain-path', default=str(HERE.parent / 'domains' / 'greenland'))
    ap.add_argument('--max-km', type=float, default=60.0, help='ocean cells farther along the water than this stay -1')
    ap.add_argument('--primacy', default='flux', choices=['flux', 'nearest'],
                    help="who owns contested water: 'flux' = the largest-flux glacier that can reach it (claimed in "
                         "decreasing terminus flux, each within max_km); 'nearest' = first arrival of the plain flood")
    ap.add_argument('--out', default='calving_basins.nc')
    a = ap.parse_args()
    domain = Path(a.domain_path)
    gi = xr.open_dataset(domain / 'model_inputs' / 'GLIDE_inputs.nc')
    dx = float(abs(gi.x[1] - gi.x[0]))
    ice = gi.rgi_mask.values > 0.5
    lab = gi.rgi_label.values.astype('float64')
    lab = np.where(np.isfinite(lab), lab, -1).astype(np.int32)
    lab[~ice] = -1
    names = gi.rgi_id.values.astype(str)
    bed = gi.bed_obs.values.astype('float64')
    bed = np.where(np.isfinite(bed), bed, gi.elevation.values.astype('float64'))
    ocean = (~ice) & (bed < 0)
    cross = ndimage.generate_binary_structure(2, 1)
    max_rounds = int(np.ceil(a.max_km * 1e3 / dx))
    # Floating tongues first: the Mouginot polygons stop near the grounding
    # line (Petermann's tongue carries no label), and a plain flood hands a
    # tongue to whichever fjord-wall basin touches it first. Each unlabelled
    # floating component takes the label that feeds it: the flux-weighted
    # (H |u|) majority over its labelled grounded 4-neighbours.
    flo = (gi.floating_mask.values > 0.5) & ice
    Hf = np.nan_to_num(gi.thickness_obs.values.astype('float64'))
    spd = np.hypot(np.nan_to_num(gi.vx.values), np.nan_to_num(gi.vy.values))
    comp, ncomp = ndimage.label(flo & (lab < 0), ndimage.generate_binary_structure(2, 1))
    n_tongue_fixed = 0
    for c in range(1, ncomp + 1):
        m = comp == c
        nbh = ndimage.binary_dilation(m, ndimage.generate_binary_structure(2, 1)) & ~m & (lab >= 0)
        if not nbh.any():
            continue
        w = np.bincount(lab[nbh], weights=(Hf * spd)[nbh] + 1.0, minlength=len(names))
        lab[m] = int(np.argmax(w)); n_tongue_fixed += 1
    # remaining ice cells outside every basin polygon (mask / polygon
    # disagreement at the margins) join the nearest basin by the flood
    basin, dist = flood_labels(lab, ice & (lab < 0), max_rounds)
    # then the water: by flux primacy (default) or by first arrival
    if a.primacy == 'nearest':
        basin, dist = flood_labels(basin, ocean, max_rounds)
    else:
        cross4 = ndimage.generate_binary_structure(2, 1)
        term_all = ice & ndimage.binary_dilation(ocean, cross4) & (basin >= 0)
        tflux = np.bincount(basin[term_all], weights=(Hf * spd)[term_all] + 1e-3, minlength=len(names))
        order = [b for b in np.argsort(-tflux) if tflux[b] > 0]
        seeds = []
        for b in order:
            m = term_all & (basin == b)
            iy, ix = np.nonzero(m)
            seeds.append((int(b), (iy, ix)))
        basin, dist = flood_primacy(basin, ocean, seeds, max_rounds)
        print(f"water claimed by flux primacy: {len(seeds)} fronts in decreasing terminus flux")
    dist_km = np.where(dist >= 0, dist * dx / 1e3, np.nan).astype(np.float32)
    # inward: seeds = the terminus cells with their basin, domain = marine ice
    term0 = ice & ndimage.binary_dilation(ocean, ndimage.generate_binary_structure(2, 1))
    seed = np.where(term0 & (basin >= 0), basin, -1).astype(np.int32)
    reach_in, dist_in = flood_labels(seed, ice & (bed < 0) & ~term0, max_rounds)
    reach = np.where(ocean & (basin >= 0), basin, np.where(reach_in >= 0, reach_in, -1)).astype(np.int32)
    front_dist = np.full(ice.shape, np.nan, np.float32)
    front_dist[ocean & (basin >= 0)] = dist_km[ocean & (basin >= 0)]
    front_dist[reach_in >= 0] = -dist_in[reach_in >= 0] * dx / 1e3
    n_ocean = int(ocean.sum()); n_reached = int((ocean & (basin >= 0)).sum())
    n_in = int((reach_in >= 0).sum()); n_marine_ice = int((ice & (bed < 0)).sum())
    print(f"inward reach: {n_in} of {n_marine_ice} marine ice cells within {a.max_km:g} km of a terminus along the trough "
          f"(front_dist 10/50/90 pct {np.percentile(front_dist[reach_in >= 0], [10, 50, 90]).round(0)} km)")
    print(f"{n_tongue_fixed} unlabelled floating components assigned to the basin feeding them")
    print(f"grid {ice.shape}, dx {dx:g} m: {int(ice.sum())} ice cells ({int((ice & (lab < 0)).sum())} outside every basin polygon, "
          f"{int((ice & (basin < 0)).sum())} still unassigned after the flood) in {len(names)} basins; {n_ocean} ice-free cells "
          f"below sea level, {n_reached} reached within {a.max_km:g} km along the water ({n_ocean - n_reached} open ocean / unconnected)")

    # ---------------------------------------------------------------- per-basin table
    shp = gpd.read_file(BASINS_SHP)
    assert list(names[:len(shp)]) == list(shp.NAME.astype(str)), 'rgi_id order differs from the shapefile'
    region = np.array(['periphery'] * len(names), dtype=object); region[:len(shp)] = shp.SUBREGION1.astype(str).values
    gtype = np.array([''] * len(names), dtype=object); gtype[:len(shp)] = shp.GL_TYPE.astype(str).values
    gates = gpd.read_file(GATES)
    gates_by_name = gates.groupby('Mouginot_2019').gate.unique().to_dict()
    term = ice & ndimage.binary_dilation(ocean, cross)
    marine_ice = ice & (bed < 0)
    H = np.nan_to_num(gi.thickness_obs.values.astype('float64'))
    speed = np.hypot(np.nan_to_num(gi.vx.values), np.nan_to_num(gi.vy.values))
    nb = len(names)
    cnt = lambda m: np.bincount(basin[m & (basin >= 0)], minlength=nb)[:nb]
    n_ice, n_mar, n_term, n_oc = cnt(ice), cnt(marine_ice), cnt(term), cnt(ocean & (basin >= 0))
    n_flo = cnt(flo)
    n_reach_in = np.bincount(reach_in[reach_in >= 0], minlength=nb)[:nb]
    tl = term & (basin >= 0)
    flux = np.bincount(basin[tl], weights=(H * speed)[tl], minlength=nb)[:nb]          # m2/yr per m of front, summed
    speed_term = np.bincount(basin[tl], weights=speed[tl], minlength=nb)[:nb] / np.maximum(n_term, 1)
    H_term = np.bincount(basin[tl], weights=H[tl], minlength=nb)[:nb] / np.maximum(n_term, 1)
    rows = pd.DataFrame(dict(basin=np.arange(nb), name=names, region=region, gl_type=gtype, n_ice=n_ice,
                             n_marine_ice=n_mar, n_terminus=n_term, n_floating=n_flo, n_ocean=n_oc, n_reach_under_ice=n_reach_in,
                             terminus_flux_m2yr=flux, terminus_speed=speed_term, terminus_H=H_term,
                             gates=[' '.join(str(int(g)) for g in gates_by_name.get(n, [])) for n in names]))
    rows['calves'] = (rows.n_terminus > 0) | (rows.n_floating > 0)
    tab = rows[rows.n_ice > 0]
    print(f"basins with ice: {len(tab)}; with a marine terminus: {int(tab.calves.sum())} "
          f"({int(tab[tab.calves].region.ne('periphery').sum())} Mouginot + {int(tab[tab.calves].region.eq('periphery').sum())} periphery); "
          f"with Mankoff gates: {int(tab.gates.ne('').sum())}")
    big = tab[tab.calves].sort_values('terminus_flux_m2yr', ascending=False).head(12)
    print("largest marine fronts by H |u| summed over terminus cells:")
    print(big[['name', 'region', 'n_terminus', 'n_ocean', 'terminus_speed', 'terminus_H', 'gates']].to_string(index=False))
    tab.to_csv(domain / 'model_inputs' / Path(a.out).with_suffix('.csv').name, index=False)

    # ---------------------------------------------------------------- write
    ds = xr.Dataset(dict(calving_basin=(('y', 'x'), basin.astype(np.int32)),
                         basin_dist_km=(('y', 'x'), dist_km),
                         calving_reach=(('y', 'x'), reach),
                         front_dist_km=(('y', 'x'), front_dist)),
                    coords=dict(y=gi.y.values, x=gi.x.values))
    ds.calving_reach.attrs.update(long_name='calving reach: the flooded water plus the marine ice within max_km of a terminus along the trough',
                                  description='-1 elsewhere; the cells a margin change can move, per glacier')
    ds.front_dist_km.attrs.update(units='km', long_name='signed distance from the terminus along the water (+) or the trough under the ice (-)')
    ds.calving_basin.attrs.update(long_name='calving basin index (rgi_label on the ice, geodesic flood fill over the ocean)',
                                  description='-1 = none: land above sea level, open ocean beyond max_km, unconnected hollows; '
                                              'names in the CSV / rgi_id of GLIDE_inputs.nc')
    ds.basin_dist_km.attrs.update(units='km', long_name='BFS distance from the labelled ice (0 on labelled ice)')
    ds.attrs.update(source=f"make_calving_basins.py {date.today().isoformat()}", max_km=a.max_km, primacy=a.primacy,
                    n_basins=int(nb), n_reached_ocean=n_reached, connectivity=4,
                    seeds='all ice cells with rgi_label; domain = ice-free cells with bed < 0 (bed_obs on ice, elevation off)')
    if 'spatial_ref' in gi:
        ds['spatial_ref'] = gi['spatial_ref']
        for v in ('calving_basin', 'basin_dist_km', 'calving_reach', 'front_dist_km'):
            ds[v].attrs['grid_mapping'] = 'spatial_ref'
    out = domain / 'model_inputs' / a.out
    ds.to_netcdf(out, encoding={'calving_basin': dict(zlib=True, complevel=4, _FillValue=None), 'basin_dist_km': dict(zlib=True, complevel=4),
                                'calving_reach': dict(zlib=True, complevel=4, _FillValue=None), 'front_dist_km': dict(zlib=True, complevel=4)})
    print(f"wrote {out} and {out.with_suffix('.csv')}")


if __name__ == '__main__':
    main()
