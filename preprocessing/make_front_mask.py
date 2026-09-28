#!/usr/bin/env python
"""
Time-varying front mask from TermPicks v2, for `ocean.PinnedFront` (config
`ocean_forcing.pin_front_filename`): per calendar year the ice extent the
model is pinned to, i.e. the observed terminus positions over the
historical period instead of the one epoch `rgi_mask` carries.

  model_inputs/front_mask.nc   front_mask (time, y, x) uint8; glacier_id (y, x)
  model_inputs/front_mask.csv  per glacier-year: the trace used, cells added / cut

TermPicks (Goliber et al. 2022, v2: 279 glaciers, 39 060 traces, mostly
1972-2020, EPSG:3413 = the model grid) gives terminus traces as polylines
with an integer GlacierID and no names. Each glacier is attached to the
calving basin (`calving_basins.nc`, preprocessing/make_calving_basins.py)
whose TERMINUS cells (ice adjacent to water) are nearest its traces, and
gets a PRIVATE along-fjord coordinate: a BFS from that terminus outward
over the water (positive, km) and inward along its own basin's
sub-sea-level trough (negative), in a window around its traces --
independent of which basin the flux-primacy flood gave the water to (a
small glacier's fjord belongs to its big neighbour there, and measuring
its traces from the neighbour's terminus put phantom fronts 35 km out).
A cell within `--max-dist-km` of a trace is seaward of it when its
coordinate exceeds the median coordinate at the trace's vertices: no
tangents or orientation, immune to trace curvature, consistent for traces
behind the 2015 front (negative coordinate). The
year's mask is the 2015 `rgi_mask` PLUS the landward reach cells (ice may
advance to where the front was) MINUS the seaward ones (no ice beyond the
observed front). Only reach cells are ever touched, so lateral ice on land
stays as the inventory has it. One trace per glacier-year (the one nearest
mid-year; `--min-quality` drops flagged picks), years without one take the
nearest observed year (earlier on ties), the first observation is held
before it (the earliest extent is the closest thing to the pre-satellite
front) and the last after. Glaciers without TermPicks coverage keep the
2015 mask throughout -- the static pin's behaviour.

The loader pins each model step to the mask of the step's END year, so
the state at t1 matches the observation at t1; a 10-yr spin-up step
cannot resolve a retreat inside it either way.

Usage:
  python preprocessing/make_front_mask.py --domain-path domains/greenland [--years 1972 2025] [--max-dist-km 15]
"""
import argparse
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.ops import linemerge

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from make_calving_basins import flood_labels  # noqa: E402
TERMPICKS = HERE.parent / 'common_data' / 'area' / 'termpicks' / 'TermPicks_V2' / 'TermPicks_V2.shp'


def as_line(geom):
    """One LineString per trace: merge a MultiLineString, keep the longest part."""
    if geom.geom_type == 'LineString':
        return geom
    m = linemerge(geom)
    if m.geom_type == 'LineString':
        return m
    return max(m.geoms, key=lambda p: p.length)


def side_of(line, px, py):
    """Signed side of the points (px, py) w.r.t. the polyline (cross product
    of the local tangent with the offset from the nearest point), and the
    distance to the line."""
    pts = shapely.points(px, py)
    s = shapely.line_locate_point(line, pts)
    q = shapely.line_interpolate_point(line, s)
    qx, qy = shapely.get_coordinates(q).T
    L = line.length
    a = shapely.line_interpolate_point(line, np.clip(s - 50.0, 0, L)); b = shapely.line_interpolate_point(line, np.clip(s + 50.0, 0, L))
    ax, ay = shapely.get_coordinates(a).T; bx, by = shapely.get_coordinates(b).T
    tx, ty = bx - ax, by - ay
    cross = tx * (py - qy) - ty * (px - qx)
    dist = np.hypot(px - qx, py - qy)
    return cross, dist


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--domain-path', default=str(HERE.parent / 'domains' / 'greenland'))
    ap.add_argument('--termpicks', default=str(TERMPICKS))
    ap.add_argument('--years', type=int, nargs=2, default=(1972, 2025), help='time axis of the mask (calendar years, inclusive)')
    ap.add_argument('--max-dist-km', type=float, default=15.0, help='reach cells farther than this from the trace are not classified')
    ap.add_argument('--min-quality', default=None, help="keep only traces whose QualFlag is in this set, e.g. '00,05' (default: all)")
    ap.add_argument('--out', default='front_mask.nc')
    a = ap.parse_args()
    domain = Path(a.domain_path)
    gi = xr.open_dataset(domain / 'model_inputs' / 'GLIDE_inputs.nc')
    bas = xr.open_dataset(domain / 'model_inputs' / 'calving_basins.nc', mask_and_scale=False)
    binfo = pd.read_csv(domain / 'model_inputs' / 'calving_basins.csv').set_index('basin')
    X, Y = gi.x.values.astype('float64'), gi.y.values.astype('float64')
    dx, dy = float(X[1] - X[0]), float(Y[1] - Y[0])
    ny, nx = len(Y), len(X)
    base = (gi.rgi_mask.values > 0.5)
    reach = bas.calving_reach.values.astype(np.int64)
    basin = bas.calving_basin.values.astype(np.int64)
    fdist = bas.front_dist_km.values.astype('float64')
    epoch = float(gi.rgi_mask.attrs.get('time_nominal', 2015))

    tp = gpd.read_file(a.termpicks)
    if a.min_quality:
        keep = set(q.strip() for q in a.min_quality.split(','))
        tp = tp[tp.QualFlag.astype(str).isin(keep)]
    tp = tp[np.isfinite(tp.DecDate)].copy()
    tp['line'] = [as_line(g) for g in tp.geometry]
    print(f"TermPicks: {len(tp)} traces, {tp.GlacierID.nunique()} glaciers, {int(tp.Year.min())}-{int(tp.Year.max())}; mask years {a.years[0]}-{a.years[1]}")

    # ------------------------------------------------ glacier -> basin: the nearest TERMINUS cells to its traces
    bed = np.where(np.isfinite(gi.bed_obs.values), gi.bed_obs.values, gi.elevation.values).astype('float64')
    water = (~base) & (bed < 0)
    trough = base & (bed < 0)
    cross4 = ndimage.generate_binary_structure(2, 1)
    term = base & ndimage.binary_dilation(water, cross4) & (basin >= 0)
    ty, tx = np.nonzero(term)
    ttree = cKDTree(np.c_[Y[ty], X[tx]])
    glaciers = {}
    for gid, sub in tp.groupby('GlacierID'):
        cx, cy = np.concatenate([shapely.get_coordinates(l) for l in sub.line]).T
        d, k = ttree.query([float(np.median(cy)), float(np.median(cx))])
        b = int(basin[ty[k], tx[k]])
        glaciers[int(gid)] = dict(basin=b, name=str(binfo.name.get(b, b)), region=str(binfo.region.get(b, '')),
                                  attach_km=float(d) / 1e3, n_traces=len(sub), y0=int(sub.Year.min()), y1=int(sub.Year.max()),
                                  bbox=(cy.min(), cy.max(), cx.min(), cx.max()))
    far = sum(g['attach_km'] > 5 for g in glaciers.values())
    print(f"attached {len(glaciers)} glaciers to {len(set(g['basin'] for g in glaciers.values()))} basins by the nearest terminus "
          f"({far} more than 5 km from any terminus cell)")

    # ------------------------------------------------ per glacier-year: the classified reach cells
    years = np.arange(a.years[0], a.years[1] + 1)
    nt = len(years)
    add = {}      # (gid, year) -> (iy, ix) landward reach cells (may be water: ice allowed)
    cut = {}      # (gid, year) -> (iy, ix) seaward reach cells (no ice)
    rows = []
    gid_map = np.full((ny, nx), -1, np.int16)
    pad = int(np.ceil((a.max_dist_km + 5.0) * 1e3 / dx)) + 1
    for gid, info in glaciers.items():
        b = info['basin']
        y_lo, y_hi, x_lo, x_hi = info['bbox']
        # a private along-fjord coordinate in a window around the traces: BFS from
        # the basin's terminus cells over the water (+) and along its own trough (-)
        i0 = max(int(np.rint((y_hi - Y[0]) / dy)) - pad, 0); i1 = min(int(np.rint((y_lo - Y[0]) / dy)) + pad + 1, ny)
        j0 = max(int(np.rint((x_lo - X[0]) / dx)) - pad, 0); j1 = min(int(np.rint((x_hi - X[0]) / dx)) + pad + 1, nx)
        wterm = term[i0:i1, j0:j1] & (basin[i0:i1, j0:j1] == b)
        if not wterm.any():
            continue
        seed = np.where(wterm, 0, -1).astype(np.int32)
        _, d_out = flood_labels(seed, water[i0:i1, j0:j1], 2 * pad)
        _, d_in = flood_labels(seed, trough[i0:i1, j0:j1] & (basin[i0:i1, j0:j1] == b) & ~wterm, 2 * pad)
        coord = np.full(wterm.shape, np.nan)
        coord[d_out >= 0] = d_out[d_out >= 0] * dx / 1e3
        coord[(d_in > 0)] = -d_in[d_in > 0] * dx / 1e3
        coord[wterm] = 0.0
        wi, wj = np.nonzero(np.isfinite(coord))
        ci, cj = wi + i0, wj + j0
        if len(ci) == 0:
            continue
        cc = coord[wi, wj]
        px, py = X[cj], Y[ci]
        gid_map[ci, cj] = np.where(gid_map[ci, cj] < 0, gid, gid_map[ci, cj])
        ctree = cKDTree(np.c_[py, px])
        sub = tp[tp.GlacierID == gid]
        used = {}
        for y in years:
            s = sub[(sub.DecDate >= y) & (sub.DecDate < y + 1)]
            if len(s) == 0:
                continue
            k = (s.DecDate - (y + 0.5)).abs().idxmin()
            used[int(y)] = k
        for y, k in used.items():
            line = tp.line[k]
            _, dist = side_of(line, px, py)
            near = dist <= a.max_dist_km * 1e3
            if near.sum() < 3:
                continue
            # the trace's position in the private coordinate: the nearest coordinate cells to its vertices
            vx_, vy_ = shapely.get_coordinates(line).T
            d_, k_ = ctree.query(np.c_[vy_, vx_])
            fv = cc[k_[d_ <= 2.0 * dx]]
            if len(fv) == 0:
                fv = cc[k_]
            d_t = float(np.median(fv))
            sea = near & (cc > d_t); land = near & (cc <= d_t)
            add[(gid, y)] = (ci[land], cj[land]); cut[(gid, y)] = (ci[sea], cj[sea])
            rows.append(dict(GlacierID=gid, basin=b, name=info['name'], region=info['region'], year=y, DecDate=float(tp.DecDate[k]),
                             QualFlag=str(tp.QualFlag[k]), Author=str(tp.Author[k]), n_near=int(near.sum()), front_km=d_t,
                             advance_cells=int((land & ~base[ci, cj]).sum()), cut_cells=int((sea & base[ci, cj]).sum())))
    obs = pd.DataFrame(rows)
    print(f"classified {len(obs)} glacier-years; front position (km along the fjord from the 2015 terminus) 10/50/90 pct "
          f"{np.percentile(obs.front_km, [10, 50, 90]).round(1)}; advance cells (water allowed to hold ice) 10/50/90 pct "
          f"{np.percentile(obs.advance_cells, [10, 50, 90]).round(0)}, cut cells (2015 ice removed) {np.percentile(obs.cut_cells, [10, 50, 90]).round(0)}")

    # ------------------------------------------------ the yearly masks: nearest observed year per glacier
    mask = np.zeros((nt, ny, nx), np.uint8)
    filled = []
    for t, y in enumerate(years):
        m = base.copy()
        for gid in glaciers:
            ys = np.array(sorted(yy for (g, yy) in add if g == gid))
            if len(ys) == 0:
                continue
            k = ys[np.argmin(np.abs(ys - y) + 1e-3 * (ys > y))]      # nearest, earlier on ties
            li, lj = add[(gid, int(k))]; si, sj = cut[(gid, int(k))]
            m[li, lj] = True; m[si, sj] = False
            filled.append(dict(GlacierID=gid, year=int(y), trace_year=int(k)))
        mask[t] = m
    filled = pd.DataFrame(filled)
    n_change = np.abs(mask.astype(np.int16) - base.astype(np.int16)[None]).sum(axis=(1, 2))
    print("cells differing from the 2015 mask per year (10-yr samples): " + ", ".join(f"{y}: {n}" for y, n in zip(years[::10], n_change[::10])) + f", {years[-1]}: {n_change[-1]}")

    # ------------------------------------------------ write
    out = domain / 'model_inputs' / a.out
    ds = xr.Dataset(dict(front_mask=(('time', 'y', 'x'), mask), glacier_id=(('y', 'x'), gid_map)),
                    coords=dict(time=years.astype('float64'), y=Y, x=X))
    ds.front_mask.attrs.update(long_name='ice extent the front is pinned to, per calendar year',
                               description=f'rgi_mask ({epoch:g}) + reach cells landward of the TermPicks trace nearest mid-year - reach cells seaward; '
                                           'nearest observed year elsewhere, first / last held outside the record')
    ds.glacier_id.attrs.update(long_name='TermPicks GlacierID whose traces classify this reach cell (-1 none)')
    ds.attrs.update(source=f"make_front_mask.py {date.today().isoformat()}", termpicks=str(a.termpicks), base_epoch=epoch,
                    max_dist_km=a.max_dist_km, min_quality=a.min_quality or 'all', n_glaciers=len(glaciers),
                    n_glacier_years=len(obs), years=f"{years[0]}-{years[-1]}")
    if 'spatial_ref' in gi:
        ds['spatial_ref'] = gi['spatial_ref']
        ds.front_mask.attrs['grid_mapping'] = 'spatial_ref'; ds.glacier_id.attrs['grid_mapping'] = 'spatial_ref'
    ds.to_netcdf(out, encoding={'front_mask': dict(zlib=True, complevel=4, chunksizes=(1, ny, nx)), 'glacier_id': dict(zlib=True)})
    obs.to_csv(out.with_suffix('.csv'), index=False)
    ginfo = pd.DataFrame.from_dict(glaciers, orient='index').drop(columns='bbox'); ginfo.index.name = 'GlacierID'
    ginfo.to_csv(out.with_name(out.stem + '_glaciers.csv'))
    print(f"wrote {out} ({nt} years x {ny} x {nx}), {out.with_suffix('.csv')}, {out.with_name(out.stem + '_glaciers.csv')}")


if __name__ == '__main__':
    main()
