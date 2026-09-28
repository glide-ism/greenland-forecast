#!/usr/bin/env python
"""
Build the critical-anomaly field rho(x) for the calving margins
(model_inputs/calving_rho.nc; config `ocean_forcing.rho_filename`, library
change 16).

Under glide's monotone calving law with q = 0 a floating cell has
F = -h0, so with

    h0(x, t) = alpha_h (dTF(x, t) - rho(x))          (clim_h = 0, calving_h0 = 0)

a front flips exactly when its thermal-forcing anomaly exceeds rho: rho is
the front's distance to threshold in the reference climate, in kelvin,
and alpha_h (the one remaining scalar) the metres of margin per kelvin of
exceedance, i.e. how far a grounded front retreats once flipped. The
observed onsets (analysis/front_onsets.csv) bound rho per documented
front from the forcing record alone (analysis/calving_screen.py's ratio
rule, on the model's own step aggregation of dTF):

    stable front:   rho > max_t dTF                     -> rho = safety * max dTF
    retreat front:  m1 < rho <= m2, m1 / m2 the max dTF before / through
                    the onset window (+- tol)           -> rho = (max(m1, 0) + m2) / 2
    empty window (m2 <= m1: the product's anomaly peaks before the
                    retreat, no rho times it)           -> rho = empty_factor * m1
                    (never flips through the window; flips later if the
                    record exceeds it -- "missed" rather than "early")

Cells get the rho of the documented front they belong to: floating
components by the screen's component assignment (a tongue takes its
front's rho along its whole length), every other cell the nearest
documented front's rho within `--assign-km`, and beyond that the REGIONAL
default (the median rho of the region's documented fronts; the region is
the nearest gate's Mouginot region). Land cells are filled too (rho is
inert where nothing floats). The file is on the full domain grid, like
thermal_forcing.nc; OceanForcing crops it.

THE FIELD IS BOUNDED (`--bound-mask`, default rgi_mask): outside the
observed extent the file's `h0_fixed` holds `--outside-h0` (+250 m, the
pin's value) and OceanForcing uses it verbatim, so marine advance beyond
the mask is cut at the calving timescale while inside it the margin is
alpha_h (dTF - rho). Without the bound (`--bound-mask none`) h0 < 0
everywhere makes floating ice admissible everywhere, every fjord fills
during the spin-up and the advanced ice grounds where a later positive
margin of tens of metres cannot remove it -- the first field did exactly
that (2026-09-24). The bounded field is the front pin with a finite,
time-varying inside depth: it cannot represent an extension beyond the
mask (a pre-retreat front needs a maximal-extent mask, e.g. TermPicks),
but it cannot run away, and re-advance up to the mask is allowed when the
anomaly drops again. Under the bound the model's retreat at a grounded
front is the loss of its 2015 terminus cells, so those (the `term` set)
are the diagnostic set for rho and the verification, not the advance.

The field is calibrated, not predicted: one number per documented front
from one onset per front. What remains testable is the undocumented
fronts on the regional default, the 2015-2026 window, the gate fluxes and
the retreat EXTENT (alpha_h). Verified after building by re-running the
screen's flip machinery with the field (printed table).

Usage:
  python preprocessing/make_calving_rho.py --domain-path domains/greenland
  python preprocessing/make_calving_rho.py --filter ema:5 --min-confidence medium --dry-run
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'analysis'))
sys.path.insert(0, str(HERE.parent))
from calving_screen import (GATES, ONSETS, MIN_TONGUE_CELLS, R_KERNEL, R_FLOAT, Screen, build_fronts,  # noqa: E402
                            classify, filtered_dtf, load_config, load_forcing, load_onsets, ratio_bounds)
from basin_mass_balance import region_masks, load_gates, crop_to_factor, N_LEVELS  # noqa: E402

CONF_RANK = dict(low=0, medium=1, high=2)


def choose_rho(onsets, fronts, m1, m2, m_all, safety, empty_factor, floor):
    """Per documented front: rho and how it was chosen."""
    rows = []
    for i, f in enumerate(fronts.index):
        if f not in onsets.index:
            continue
        o = onsets.loc[f]
        if o.stable:
            rho, how = max(safety * m_all[i], floor), f"stable: {safety:g} x max dTF {m_all[i]:.2f}"
        elif m2[i] > max(m1[i], 0.0):
            lo = max(m1[i], 0.0)
            rho, how = max(0.5 * (lo + m2[i]), floor), f"window ({lo:.2f}, {m2[i]:.2f}]"
        else:
            rho, how = max(empty_factor * max(m1[i], 0.0), floor), f"EMPTY (peak {m1[i]:.2f} before the window): missed by design"
        rows.append(dict(front=f, region=fronts.region[f], rho=float(rho), how=how, onset=o.onset_str,
                         confidence=o.confidence, m1=m1[i], m2=m2[i], m_all=m_all[i], tongue=bool(fronts.tongue_2015[f])))
    return pd.DataFrame(rows).set_index('front')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--domain-path', default=str(HERE.parent / 'domains' / 'greenland'))
    ap.add_argument('--onsets', default=str(ONSETS))
    ap.add_argument('--filter', default='none', help="temporal filter on dTF, as in calving_screen (none | box:N | ema:TAU)")
    ap.add_argument('--tol', type=float, default=3.0, help='years beyond the onset window still on time')
    ap.add_argument('--safety', type=float, default=1.25, help='stable fronts: rho = safety x max dTF over the record')
    ap.add_argument('--empty-factor', type=float, default=1.05, help='empty windows: rho = factor x the pre-window peak')
    ap.add_argument('--floor', type=float, default=0.05, help='minimum rho (K)')
    ap.add_argument('--assign-km', type=float, default=30.0, help='cells within this distance of a documented front take its rho')
    ap.add_argument('--min-confidence', default='low', choices=['low', 'medium', 'high'], help='use onset rows of at least this confidence')
    ap.add_argument('--alpha-check', type=float, default=None, help='alpha_h for the verification (default: config, or 50 if 0)')
    ap.add_argument('--bound-mask', default='rgi_mask', help="gridded boolean variable bounding the field ('none' = unbounded, runs away)")
    ap.add_argument('--outside-h0', type=float, default=250.0, help='fixed margin (m) outside the bound mask')
    ap.add_argument('--out', default='calving_rho.nc', help='file name under model_inputs/')
    ap.add_argument('--dry-run', action='store_true', help='print the table, write nothing')
    a = ap.parse_args()

    domain = Path(a.domain_path)
    cfg = load_config(domain)
    ocfg = cfg.ocean_forcing
    if ocfg.clim_h != 0 or cfg.calving_h0 != 0:
        print(f"WARNING: rho is derived for a static margin of -alpha_h rho alone; the config has clim_h "
              f"{ocfg.clim_h:g} and calving_h0 {cfg.calving_h0:g} -- set both to 0 when using the field")
    print(f"config {cfg.results_subdir}: TF {ocfg.statistic}, ref {ocfg.ref_years}, schedule {cfg.dt_schedule}, "
          f"dt {cfg.dt:g}; filter {a.filter}, tol {a.tol:g}, safety {a.safety:g}")

    masks, dx, gi = region_masks(domain)
    gates = load_gates(GATES, gi)
    names = gpd.read_file(GATES).groupby('gate').Mouginot_2019.first().to_dict()
    gate_names = {g['gate']: str(names.get(g['gate'], f"gate{g['gate']}")) for g in gates}
    fronts, cells = build_fronts(gi, gates, gate_names, cfg.calving_H_c, R_KERNEL, cfg.calving_timescale)
    years, clim, ok, dtf_annual = load_forcing(domain / 'model_inputs' / ocfg.filename, ocfg, cells)
    S = Screen(fronts, cells, years, clim, ok, dtf_annual, cfg, ocfg)
    onsets = load_onsets(a.onsets)
    onsets = onsets[[CONF_RANK.get(str(c), 0) >= CONF_RANK[a.min_confidence] for c in onsets.confidence]]
    onsets = onsets.loc[[f for f in onsets.index if f in fronts.index]]
    fronts['tongue_2015'] = fronts.n_T >= MIN_TONGUE_CELLS
    dtf_stepped = filtered_dtf(dtf_annual, years, a.filter, tuple(cfg.dt_schedule), cfg.dt)
    bounded = a.bound_mask != 'none'
    gset = 'term' if bounded else 'A'
    m1, m2, m_all = ratio_bounds(S, onsets, dtf_stepped, a.tol, gset)
    table = choose_rho(onsets, fronts, m1, m2, m_all, a.safety, a.empty_factor, a.floor)
    regional = table.groupby('region').rho.median().to_dict()
    global_default = float(table.rho.median())

    # ---------------------------------------------------------------- raster (cropped grid)
    ny, nx = gi.rgi_mask.shape
    ice = gi.rgi_mask.values > 0.5
    flo = (gi.floating_mask.values > 0.5) & ice
    dx_km = dx / 1e3
    doc = set(table.index)
    g_iy = np.concatenate([g['iy'] for g in gates]); g_ix = np.concatenate([g['ix'] for g in gates])
    g_front = np.concatenate([[gate_names[g['gate']]] * len(g['iy']) for g in gates])
    g_region = np.concatenate([[g['region']] * len(g['iy']) for g in gates])
    tree_all = cKDTree(np.c_[g_iy, g_ix])
    is_doc = np.array([f in doc for f in g_front])
    tree_doc = cKDTree(np.c_[g_iy[is_doc], g_ix[is_doc]])
    doc_front = g_front[is_doc]
    front_list = list(table.index)
    fidx = {f: i for i, f in enumerate(front_list)}

    Y, X = np.mgrid[0:ny, 0:nx]
    pts = np.c_[Y.ravel(), X.ravel()]
    d_all, k_all = tree_all.query(pts)
    region_cell = g_region[k_all].reshape(ny, nx)
    rho = np.vectorize(lambda r: regional.get(r, global_default))(region_cell).astype('float32')
    source = np.zeros((ny, nx), np.int8)                     # 0 regional default
    front_index = np.full((ny, nx), -1, np.int16)
    d_doc, k_doc = tree_doc.query(pts)
    near = (d_doc * dx_km <= a.assign_km).reshape(ny, nx)
    f_near = doc_front[k_doc].reshape(ny, nx)
    for f in front_list:
        m = near & (f_near == f)
        rho[m] = table.rho[f]; source[m] = 1; front_index[m] = fidx[f]
    # floating components wholesale, as the screen assigns them
    lab, ncomp = ndimage.label(flo, ndimage.generate_binary_structure(2, 1))
    n_comp_doc = 0
    for c in range(1, ncomp + 1):
        iy, ix = np.nonzero(lab == c)
        dd, kk = tree_all.query(np.c_[iy, ix])
        j = int(np.argmin(dd))
        if dd[j] * dx_km > R_FLOAT:
            continue
        f = g_front[kk[j]]
        if f in doc:
            rho[iy, ix] = table.rho[f]; source[iy, ix] = 2; front_index[iy, ix] = fidx[f]; n_comp_doc += 1

    # the bound: h0 fixed outside the observed extent
    fixed = np.full((ny, nx), np.nan, np.float32)
    if bounded:
        if a.bound_mask not in gi:
            raise SystemExit(f"--bound-mask {a.bound_mask!r} is not a variable of the gridded inputs")
        inside = gi[a.bound_mask].values > 0.5
        fixed[~inside] = a.outside_h0

    # ---------------------------------------------------------------- verification with the screen
    S.rho = rho[S.iy, S.ix].astype('float64')
    S.h0_fixed = fixed[S.iy, S.ix].astype('float64')
    alpha = a.alpha_check if a.alpha_check is not None else (ocfg.alpha_h if ocfg.alpha_h > 0 else 50.0)
    # verified in the mode the field is derived for (static margin = -alpha_h rho alone)
    fr, st, _, _ = S.evaluate(ocfg.tf_crit, 0.0, alpha, 0.0, cfg.calving_q, cfg.calving_H_c, 1.0, dtf_stepped, ocfg.alpha_q)
    fl = S.flip_years(fr, st, gset)
    cls = classify(fl, onsets, a.tol)
    table['flip_at_rho'] = fl.flip.reindex(table.index)
    table['adm0'] = fl.adm0.reindex(table.index)
    table['class'] = cls.reindex(table.index)
    print(f"\n{len(table)} documented fronts (confidence >= {a.min_confidence}); regional defaults (K): "
          + ", ".join(f"{r} {v:.2f}" for r, v in sorted(regional.items())) + f"; global {global_default:.2f}")
    print(f"verification with alpha_h {alpha:g}, alpha_q {ocfg.alpha_q:g}/K, clim_h 0, calving_h0 0 (the field's mode; q {cfg.calving_q:g}, H_c {cfg.calving_H_c:g}); "
          f"bound {a.bound_mask} / outside h0 {a.outside_h0:g} m; grounded fronts judged on '{gset}':")
    print("  front                      reg  rho(K)  set  adm0   flip   onset       class      how")
    for f, r in table.sort_values(['region', 'rho']).iterrows():
        fy = '-' if np.isnan(r.flip_at_rho) else str(int(r.flip_at_rho))
        print(f"  {f[:26]:26s} {r.region:3s}  {r.rho:5.2f}   {'T' if r.tongue else gset[0]}   {r.adm0:4.2f}   {fy:5s}  "
              f"{r.onset:10s}  {str(r['class']):9s}  {r.how}")
    n = table['class'].value_counts()
    print("  verdicts: " + ", ".join(f"{k} {v}" for k, v in n.items()))
    print(f"raster: {int((source == 1).sum())} cells on documented fronts within {a.assign_km:g} km, "
          f"{int((source == 2).sum())} on their floating components ({n_comp_doc} components), "
          f"{int((source == 0).sum())} on the regional default; rho over the ice 10/50/90 pct "
          f"{np.percentile(rho[ice], [10, 50, 90]).round(2)} K")
    if a.dry_run:
        print("dry run: nothing written")
        return

    # ---------------------------------------------------------------- write on the full grid
    full = xr.open_dataset(domain / 'model_inputs' / 'GLIDE_inputs.nc')
    ny0, nx0 = full.sizes['y'], full.sizes['x']
    fac = 2 ** N_LEVELS
    nyc, nxc = (ny0 // fac) * fac, (nx0 // fac) * fac
    y0, x0 = (ny0 - nyc) // 2, (nx0 - nxc) // 2
    assert (nyc, nxc) == (ny, nx)

    def embed(arr, fill):
        out = np.full((ny0, nx0), fill, arr.dtype)
        out[y0:y0 + ny, x0:x0 + nx] = arr
        return out
    ds = xr.Dataset(
        dict(calving_rho=(('y', 'x'), embed(rho, np.float32(global_default))),
             h0_fixed=(('y', 'x'), embed(fixed, np.float32(a.outside_h0 if bounded else np.nan))),
             rho_source=(('y', 'x'), embed(source, np.int8(0))),
             front_index=(('y', 'x'), embed(front_index, np.int16(-1)))),
        coords=dict(y=full.y.values, x=full.x.values))
    ds.h0_fixed.attrs.update(units='m', long_name='fixed calving margin where finite (replaces h0)',
                             description=f"{a.outside_h0:g} m outside {a.bound_mask}: the bound on marine advance; NaN inside")
    ds.calving_rho.attrs.update(units='K', long_name='critical thermal-forcing anomaly of the calving margin',
                                description='h0 = calving_h0 + clim_h (TF_clim - tf_crit) + alpha_h (dTF - calving_rho)')
    ds.rho_source.attrs.update(description='0 regional default (median of the region\'s documented fronts), '
                                           '1 documented front within assign_km, 2 its floating component')
    ds.front_index.attrs.update(description='index into the front_names attribute, -1 none')
    ds.attrs.update(source=f"make_calving_rho.py {date.today().isoformat()}", onsets=str(a.onsets), filter=a.filter,
                    tol=a.tol, safety=a.safety, empty_factor=a.empty_factor, floor=a.floor, assign_km=a.assign_km,
                    min_confidence=a.min_confidence, bound_mask=a.bound_mask, outside_h0=a.outside_h0, grounded_set=gset,
                    tf_statistic=ocfg.statistic, ref_years=json.dumps(list(ocfg.ref_years)),
                    dt_schedule=json.dumps([list(map(float, s)) for s in cfg.dt_schedule]), dt=cfg.dt,
                    front_names=json.dumps(front_list), regional_default=json.dumps(regional), global_default=global_default)
    if 'spatial_ref' in full:
        ds['spatial_ref'] = full['spatial_ref']
        for v in ('calving_rho', 'h0_fixed', 'rho_source', 'front_index'):
            ds[v].attrs['grid_mapping'] = 'spatial_ref'
    out = domain / 'model_inputs' / a.out
    ds.to_netcdf(out, encoding={'calving_rho': dict(zlib=True, complevel=4), 'h0_fixed': dict(zlib=True, complevel=4),
                                'rho_source': dict(zlib=True), 'front_index': dict(zlib=True)})
    table.to_csv(out.with_suffix('.csv'))
    print(f"wrote {out} ({ny0} x {nx0}) and {out.with_suffix('.csv')}")


if __name__ == '__main__':
    main()
