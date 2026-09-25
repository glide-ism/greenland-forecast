#!/usr/bin/env python
"""
Offline screen of the calving margins against the observed fronts.

The calving law is GEOMETRIC (glide `common.cu calving_F`): a cell calves
when H < H_calve(depth, q, h0, H_c), and the margins are set per cell from
the thermal forcing record,

    h0(x, t) = calving_h0 + clim_h (TF_clim(x) - tf_crit) + alpha_h dTF(x, t),

so whether the OBSERVED geometry of a front is admissible, and the year it
stops being admissible, are functions of the forcing record and the
parameters alone -- no forward run. This script evaluates them on every
Mankoff flux-gate outlet (grouped by Mouginot name), for a box of
(tf_crit, clim_h, alpha_h), and scores the box against a table of known
retreat onsets (`front_onsets.csv`, literature, approximate, edit it) with
an ASYMMETRIC loss: retreat is irreversible in the model, so an early or
spurious flip costs three late ones.

Per front, three cell sets on the observed (2015) geometry:

  T    observed floating cells thicker than H_c: the tongue. Retained
       where F(H_obs) >= 0; with q = 0 that is exactly h0 < 0.
  A    ocean cells adjacent to the terminus: the model's ADVANCE beyond the
       2015 front. The slab they can carry is limited by the law's implicit
       rate: a cell beyond the front is calved at H / tau until it is
       protected, so it fills to at most H_term u_term tau / dx (the 2018
       mosaic speed), and if that is below H_c it is never protected
       whatever h0 says. Admissible where F(min(H_term, that), depth) >= 0:
       floating advance needs h0 < 0, grounded advance onto a sill HAB > h0.
  term the terminus cells themselves (H > H_c: the calving front, not the
       thin lateral margins): F < 0 there is OVER-RETREAT behind the 2015
       front, wrong before the mask epoch at any front.

Fractions are FLUX-weighted (H |u| per cell), so a front's verdict is
carried by its trunk.

Static (dTF = 0): the tongue census -- fronts with a tongue (2015 mask, or
`tongue_ever` in the table for tongues lost before 2015) need h0_base < 0,
the others h0_base > 0. Transient: the flip year is the first year the
tongue (T; A for fronts whose tongue is already gone) is less than half
admissible, dated to the model's own step aggregation of the forcing
(`--schedule`, 10-yr steps to 1990 then annual, the overlap-weighted mean
per step as ocean.py does).

Usage:
  python analysis/calving_screen.py                      # current config + the default box
  python analysis/calving_screen.py --tf-crit 2 2.5 3 3.5 --clim-h 0 5 10 20 --alpha-h 25 50 100 --dtf-scale 0.5 1 2
  python analysis/calving_screen.py --detail 3.0,5.0,50.0  # per-front, per-year timeline for one combo

Outputs in analysis/output/calving_screen/: census.csv (per front), scores.csv
(per combo), flips_<combo>.csv (per front flip years for the config's combo
and the best-scoring ones), timeline_<combo>.csv for --detail.
"""
import argparse
import importlib.util
import itertools
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage, sparse
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from basin_mass_balance import region_masks, load_gates, crop_to_factor, N_LEVELS  # noqa: E402
from glacier_inverse.scheduling import build_step_sequence  # noqa: E402
from glacier_inverse.ocean import year_overlap_weights  # noqa: E402

GATES = HERE.parent / 'common_data' / 'dhdt' / 'mankoff' / 'dataverse_files' / 'gates.gpkg'
ONSETS = HERE / 'front_onsets.csv'
OUT = HERE / 'output' / 'calving_screen'

# assignment radii (km): a floating component goes to the gate nearest any
# of its cells (Petermann's tongue reaches 50 km beyond its gate), a
# terminus cell to the nearest gate within R_TERM (farther = an ungated outlet)
R_FLOAT = 30.0
R_TERM = 15.0
MIN_TONGUE_CELLS = 5        # cells (km2) of floating ice > H_c that make a front "tongue-bearing"
MASK_EPOCH = 2015           # the observed geometry's nominal year
W_EARLY, W_SPURIOUS, W_LATE, W_MISSED, W_OVER = 3.0, 3.0, 1.0, 1.0, 2.0
# glide's kernels hard-code rho_i / rho_w = 0.917 (common.cu) whatever the
# config's rho_water says (1028 -> 0.892 on the Python side); the law is
# evaluated with the kernel's constant so the screen matches the model.
R_KERNEL = 0.917
V_FLOOR = 10.0              # m/yr floor on the mosaic speed in the flux weights / rate limit


# ------------------------------------------------------------------ the law
def calving_F(H, depth, q, h0, H_c, r):
    """numpy transcription of glide common.cu calving_F (elementwise)."""
    Hg = (depth / r + h0) / (1.0 - q)
    Hs = np.minimum(H_c, Hg)
    gap = np.maximum(depth - r * H, 0.0)
    G = np.maximum(r * (Hg - Hs), 1e-3)
    w = np.maximum(1.0 - gap / G, 0.0)
    return H - (Hs + w * (Hg - Hs))


def load_config(domain_path):
    spec = importlib.util.spec_from_file_location('domain_config', Path(domain_path) / 'config.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CONFIG


# ------------------------------------------------------------- front cells
def build_fronts(gi, gates, gate_names, H_c, r, tau, verbose=True):
    """Diagnostic cells per front. Returns (fronts DataFrame, cells dict)."""
    ice = gi.rgi_mask.values > 0.5
    flo = (gi.floating_mask.values > 0.5) & ice
    H = np.nan_to_num(gi.thickness_obs.values.astype('float64'), nan=0.0)
    # the model's bed: BedMachine on the ice, the DEM composite (bed on land,
    # bathymetry in the fjords: `elevation` off the ice) elsewhere
    bed = gi.bed_obs.values.astype('float64')
    bed = np.where(np.isfinite(bed), bed, gi.elevation.values.astype('float64'))
    bed = np.where(np.isfinite(bed), bed, 0.0)
    depth = -bed
    dx = float(abs(gi.x[1] - gi.x[0]))
    dx_km = dx / 1e3
    speed = np.hypot(np.nan_to_num(gi.vx.values.astype('float64')), np.nan_to_num(gi.vy.values.astype('float64')))
    speed = np.maximum(speed, V_FLOOR)
    cross = ndimage.generate_binary_structure(2, 1)

    ocean = (~ice) & (bed < 0)
    term = ice & ndimage.binary_dilation(ocean, cross) & (H > H_c)
    adv = ocean & ndimage.binary_dilation(term, cross)
    # the slab an advancing front can carry onto each ocean cell: the mean
    # terminus thickness of its 4-neighbours, capped by the law's implicit
    # rate limit H u tau / dx (the sink H / tau against the feeding flux)
    def nmean(field):
        sf = ndimage.convolve(np.where(term, field, 0.0), cross.astype(float), mode='constant')
        nt = ndimage.convolve(term.astype(float), cross.astype(float), mode='constant')
        return np.where(nt > 0, sf / np.maximum(nt, 1), 0.0)
    H_term_mean = nmean(H)
    flux_mean = nmean(H * speed)                       # m2/yr per metre of front
    H_adv = np.minimum(H_term_mean, flux_mean * tau / dx)
    W = H * speed                                       # flux weight per ice cell
    W_adv = flux_mean
    # largest principal surface strain rate (1/yr) from the mosaic, with the
    # off-ice cells filled from the nearest ice cell so the front stencil is
    # not contaminated by zeros
    vx = gi.vx.values.astype('float64'); vy = gi.vy.values.astype('float64')
    good = ice & np.isfinite(vx) & np.isfinite(vy)
    idx = ndimage.distance_transform_edt(~good, return_distances=False, return_indices=True)
    vxf, vyf = vx[idx[0], idx[1]], vy[idx[0], idx[1]]
    dvx_dy, dvx_dx = np.gradient(vxf, dx)
    dvy_dy, dvy_dx = np.gradient(vyf, dx)
    exx, eyy, exy = dvx_dx, dvy_dy, 0.5 * (dvx_dy + dvy_dx)
    strain = 0.5 * (exx + eyy) + np.sqrt((0.5 * (exx - eyy)) ** 2 + exy ** 2)

    # gate pixels -> KD-tree; each gate belongs to a named front
    g_iy = np.concatenate([g['iy'] for g in gates])
    g_ix = np.concatenate([g['ix'] for g in gates])
    g_id = np.concatenate([np.full(len(g['iy']), g['gate']) for g in gates])
    tree = cKDTree(np.c_[g_iy, g_ix])
    front_of_gate = {g['gate']: gate_names[g['gate']] for g in gates}
    region_of_gate = {g['gate']: g['region'] for g in gates}

    def nearest(iy, ix):
        d, k = tree.query(np.c_[iy, ix])
        return d * dx_km, g_id[k]

    cells = {}      # (front, set) -> dict(iy, ix, H, depth, w, strain)
    def add(front, sname, iy, ix, Hc, dc, wc):
        key = (front, sname)
        sc = strain[iy, ix]
        if key in cells:
            c = cells[key]
            c['iy'] = np.r_[c['iy'], iy]; c['ix'] = np.r_[c['ix'], ix]
            c['H'] = np.r_[c['H'], Hc]; c['depth'] = np.r_[c['depth'], dc]; c['w'] = np.r_[c['w'], wc]
            c['strain'] = np.r_[c['strain'], sc]
        else:
            cells[key] = dict(iy=iy, ix=ix, H=Hc, depth=dc, w=wc, strain=sc)

    # floating components -> the gate nearest any of their cells
    lab, ncomp = ndimage.label(flo, cross)
    n_thin = 0
    for c in range(1, ncomp + 1):
        iy, ix = np.nonzero(lab == c)
        d, gid = nearest(iy, ix)
        j = int(np.argmin(d))
        if d[j] > R_FLOAT:
            continue
        thick = H[iy, ix] > H_c
        n_thin += int((~thick).sum())
        if thick.any():
            add(front_of_gate[gid[j]], 'T', iy[thick], ix[thick], H[iy, ix][thick], depth[iy, ix][thick], W[iy, ix][thick])
    # terminus and advance cells -> nearest gate within R_TERM
    n_ungated = 0
    n_rate_limited = int((adv & (H_adv < H_c)).sum())
    for sname, m, Hf, Wf in (('term', term, H, W), ('A', adv, H_adv, W_adv)):
        iy, ix = np.nonzero(m)
        d, gid = nearest(iy, ix)
        ok = d <= R_TERM
        n_ungated += int((~ok).sum()) if sname == 'term' else 0
        for g in np.unique(gid[ok]):
            s = ok & (gid == g)
            add(front_of_gate[g], sname, iy[s], ix[s], Hf[iy[s], ix[s]], depth[iy[s], ix[s]], Wf[iy[s], ix[s]])

    names = sorted({f for f, _ in cells})
    rows = []
    for f in names:
        gids = [g for g, n in front_of_gate.items() if n == f]
        rows.append(dict(front=f, region=region_of_gate[gids[0]], gates=len(gids),
                         n_T=len(cells.get((f, 'T'), {'iy': []})['iy']),
                         n_A=len(cells.get((f, 'A'), {'iy': []})['iy']),
                         n_term=len(cells.get((f, 'term'), {'iy': []})['iy'])))
    fronts = pd.DataFrame(rows).set_index('front')
    if verbose:
        print(f"fronts: {len(fronts)} named outlets from {len(gates)} gates; terminus cells thicker than H_c {int(term.sum())} "
              f"({n_ungated} farther than {R_TERM:g} km from any gate, dropped), advance cells {int(adv.sum())} "
              f"({n_rate_limited} rate-limited below H_c: H u tau / dx < {H_c:g} m), floating cells {int(flo.sum())} "
              f"({n_thin} thinner than H_c = {H_c:g} m, not diagnostic), "
              f"{int(fronts.n_T.ge(MIN_TONGUE_CELLS).sum())} tongue-bearing fronts (>= {MIN_TONGUE_CELLS} cells)")
    return fronts, cells


# ------------------------------------------------------------- the forcing
def load_forcing(tf_path, ocfg, cells):
    """Per diagnostic cell: TF_clim, ok flag, and dTF per record year (annual)."""
    ds = crop_to_factor(xr.open_dataset(tf_path), 2 ** N_LEVELS)
    var = 'tf_max' if ocfg.statistic == 'max' else 'tf_mean'
    years = ds.time.values.astype(int)
    iy = np.concatenate([c['iy'] for c in cells.values()])
    ix = np.concatenate([c['ix'] for c in cells.values()])
    # read the record whole (76 x ny x nx float32, ~1.5 GB) and index once:
    # xarray's pointwise isel on the netCDF backend decomposes into tiny reads
    full = ds[var].values
    stat = full[:, iy, ix].astype('float64')                        # (nt, npt)
    del full
    dist = ds.tf_dist.values[iy, ix].astype('float64')
    y0, y1 = ocfg.ref_years
    sel = (years >= y0) & (years <= y1)
    clim = stat[sel].mean(axis=0)
    ok = np.isfinite(clim) & np.isfinite(dist) & np.isfinite(stat).all(axis=0) & (dist <= ocfg.max_dist_km)
    clim = np.where(ok, clim, 0.0)
    dtf = np.where(ok[None, :], stat - clim[None, :], 0.0)
    dtf = np.where(np.isfinite(dtf), dtf, 0.0)
    return years, clim, ok, dtf


def step_aggregate(years, dtf, schedule, dt_max):
    """dTF per year replaced by the overlap-weighted mean over the model step
    containing that year (the forcing a step actually sees). Years before
    the record see 0 (as ocean.py), so a 1980-1990 step mixes in nothing."""
    steps = build_step_sequence(t_start=float(years[0]), t_end=float(years[-1] + 1), dt_max=dt_max,
                                dt_schedule=schedule)
    out = np.zeros_like(dtf)
    t0 = float(years[0])
    for t1, dt in steps:
        ov = [(y, w) for y, w in year_overlap_weights(t0, t1) if years[0] <= y <= years[-1]]
        if ov:
            idx = [int(np.searchsorted(years, y)) for y, _ in ov]
            agg = sum(w * dtf[i] for (_, w), i in zip(ov, idx)) / sum(w for _, w in ov)
            for y, _ in ov:
                out[int(np.searchsorted(years, y))] = agg
        t0 = t1
    return out


# ------------------------------------------------------------- evaluation
CLASS_W = dict(ok=0.0, early=3.0, spurious=3.0, denied=3.0, late=1.0, missed=1.0, never_adm=1.0, adv=1.0)
W_OVER_T, W_STATIC_T = 2.0, 1.0        # per table front: transient over-retreat before 2015 / seeded front not admissible


class Screen:
    def __init__(self, fronts, cells, years, clim, ok, dtf, cfg, ocfg):
        self.fronts, self.years = fronts, years
        self.cfg, self.ocfg = cfg, ocfg
        self.r = R_KERNEL
        keys = list(cells)
        self.H = np.concatenate([cells[k]['H'] for k in keys])
        self.depth = np.concatenate([cells[k]['depth'] for k in keys])
        self.w = np.concatenate([cells[k]['w'] for k in keys])
        self.strain = np.concatenate([cells[k]['strain'] for k in keys])
        self.clim, self.ok, self.dtf = clim, ok, dtf
        n = len(self.H)
        fi = {f: i for i, f in enumerate(fronts.index)}
        # membership matrices (front x cell), one per set, flux-weighted rows summing to 1
        self.M, self.Mraw = {}, {}
        for sname in ('T', 'A', 'term'):
            rows, cols = [], []
            off = 0
            for k in keys:
                m = len(cells[k]['H'])
                if k[1] == sname:
                    rows += [fi[k[0]]] * m
                    cols += list(range(off, off + m))
                off += m
            M = sparse.csr_matrix((self.w[cols], (rows, cols)), shape=(len(fronts), n))
            wsum = np.asarray(M.sum(axis=1)).ravel()
            self.Mraw[sname] = M
            self.M[sname] = sparse.diags(1.0 / np.maximum(wsum, 1e-9)) @ M
            self.fronts[f'n_{sname}'] = np.asarray((M > 0).sum(axis=1)).ravel().astype(int)
            self.fronts[f'flux_{sname}'] = wsum / 1e6
        # a front's forcing: the flux-weighted mean over ALL its cells
        Mall = sum(self.Mraw.values())
        wsum = np.asarray(Mall.sum(axis=1)).ravel()
        self.Mf = sparse.diags(1.0 / np.maximum(wsum, 1e-9)) @ Mall
        okf = self.Mf @ ok.astype(float)
        self.fronts['tf_clim'] = np.where(okf > 0, (self.Mf @ (clim * ok)) / np.maximum(okf, 1e-9), np.nan)
        self.fronts['forced_frac'] = okf
        hab = self.H - np.maximum(self.depth, 0.0) / self.r
        self.fronts['hab_term'] = np.where(self.fronts.n_term > 0, self.M['term'] @ hab, np.nan)
        self.fronts['strain_term'] = np.where(self.fronts.n_term > 0, self.M['term'] @ self.strain, np.nan)
        self.fronts['strain_T'] = np.where(self.fronts.n_T > 0, self.M['T'] @ self.strain, np.nan)
        self.fronts['H_term'] = np.where(self.fronts.n_term > 0, self.M['term'] @ self.H, np.nan)
        self.fronts['depth_term'] = np.where(self.fronts.n_term > 0, self.M['term'] @ self.depth, np.nan)
        dtf_f = np.asarray(self.Mf @ dtf.T)                                     # (nfront, nt)
        pre = years < 1990
        self.fronts['dtf_max_pre1990'] = dtf_f[:, pre].max(axis=1)
        self.fronts['dtf_max_1990on'] = dtf_f[:, ~pre].max(axis=1)
        self.fronts['dtf_min'] = dtf_f.min(axis=1)

    def h0(self, tf_crit, clim_h, alpha_h, h00, dtf_scale, dtf=None):
        """(nt, ncell) margin field, clipped like ocean.py, and its static part."""
        dtf = self.dtf if dtf is None else dtf
        base = h00 + clim_h * np.where(self.ok, self.clim - tf_crit, 0.0)
        lo, hi = self.ocfg.h0_bounds
        return np.clip(base[None, :] + alpha_h * dtf_scale * dtf, lo, hi), np.clip(base, lo, hi)

    def fractions(self, h0, q, H_c):
        """(nfront, nt) flux-weighted admissible fraction per set for a (nt, ncell)
        h0; 'Af' = the advance cells whose slab would FLOAT, i.e. tongue growth
        (an advance grounded on a sill is a one-cell front shift, not a tongue)."""
        F = calving_F(self.H[None, :], self.depth[None, :], q, h0, H_c, self.r)
        adm = (F >= 0).astype(float)
        out = {s: np.asarray(M @ adm.T) for s, M in self.M.items()}
        floating = (self.depth > self.r * self.H)[None, :]
        out['Af'] = np.asarray(self.M['A'] @ (adm * floating).T)
        return out

    def evaluate(self, tf_crit, clim_h, alpha_h, h00, q, H_c, dtf_scale, dtf):
        h0_t, h0_base = self.h0(tf_crit, clim_h, alpha_h, h00, dtf_scale, dtf)
        fr = self.fractions(h0_t, q, H_c)
        st = self.fractions(h0_base[None, :], q, H_c)
        return fr, {s: v[:, 0] for s, v in st.items()}, h0_t, h0_base

    def flip_years(self, fr, st):
        """Per front: the diagnostic set (T where a tongue exists, else A), its
        static admissibility adm0, the seeded terminus' static admissibility
        term0, the first year the set drops below 1/2 (only if admissible
        statically), and the first year the terminus drops below 1/2 (only if
        admissible statically: a transient retreat BEHIND the 2015 front)."""
        use_T = self.fronts.n_T.values >= MIN_TONGUE_CELLS
        adm0 = np.where(use_T, st['T'], st['A'])
        series = np.where(use_T[:, None], fr['T'], fr['A'])
        n = len(self.fronts)

        def first(below, gate):
            out = np.full(n, np.nan)
            for i in range(n):
                if gate[i] and below[i].any():
                    out[i] = self.years[int(np.argmax(below[i]))]
            return out
        flip = first(series < 0.5, adm0 >= 0.5)
        over = first(fr['term'] < 0.5, st['term'] >= 0.5)
        return pd.DataFrame(dict(set=np.where(use_T, 'T', 'A'), adm0=adm0, adv_float0=st['Af'], term0=st['term'],
                                 flip=flip, over=over), index=self.fronts.index)


def classify(flips, onsets, tol):
    """Per onset-table front, the verdict on the flip year."""
    out = {}
    for f, o in onsets.iterrows():
        if f not in flips.index:
            continue
        r = flips.loc[f]
        y, adm = r.flip, r.adm0
        if o.stable:
            if r['set'] == 'T':
                cls = 'denied' if adm < 0.5 else ('spurious' if np.isfinite(y) else 'ok')
            else:   # stable grounded front: a floating advance (tongue growth) is wrong, a sill advance is not
                cls = 'spurious' if np.isfinite(y) else ('adv' if r.adv_float0 >= 0.5 else 'ok')
        elif adm < 0.5:
            cls = 'never_adm'
        elif np.isnan(y):
            cls = 'missed'
        elif y < o.onset_lo - tol:
            cls = 'early'
        elif y > o.onset_hi + tol:
            cls = 'late'
        else:
            cls = 'ok'
        out[f] = cls
    return pd.Series(out, name='cls')


def score(flips, onsets, tol):
    cls = classify(flips, onsets, tol)
    t = flips.loc[cls.index]
    over = int((t.over < MASK_EPOCH).sum())
    static = int((t.term0 < 0.5).sum())
    n = cls.value_counts()
    s = sum(CLASS_W[c] * n.get(c, 0) for c in CLASS_W) + W_OVER_T * over + W_STATIC_T * static
    counts = {c: int(n.get(c, 0)) for c in CLASS_W}
    counts.update(over_pre2015=over, static_cut=static)
    return s, counts, cls


def alpha_windows(S, tc, ch, h00, q, H_c, ds_, dtf, onsets, tol, alphas):
    """For fixed (tf_crit, clim_h): the alpha_h values at which each table
    front is 'ok', as intervals, plus the score per alpha."""
    cls_by_alpha, scores = [], []
    for ah in alphas:
        fr, st, _, _ = S.evaluate(tc, ch, ah, h00, q, H_c, ds_, dtf)
        fl = S.flip_years(fr, st)
        s, n, cls = score(fl, onsets, tol)
        cls_by_alpha.append(cls)
        scores.append(s)
    C = pd.concat(cls_by_alpha, axis=1)
    C.columns = alphas
    rows = {}
    for f in C.index:
        okv = (C.loc[f] == 'ok').values
        if okv.all():
            rows[f] = 'any'
        elif not okv.any():
            rows[f] = 'none (' + '/'.join(sorted(set(C.loc[f].values))) + ')'
        else:
            iv, start = [], None
            for a, v in zip(alphas, okv):
                if v and start is None:
                    start = a
                if not v and start is not None:
                    iv.append((start, prev)); start = None
                prev = a
            if start is not None:
                iv.append((start, alphas[-1]))
            rows[f] = ' '.join(f"[{a:g},{b:g}]" + ('+' if b == alphas[-1] else '') for a, b in iv)
    return pd.Series(rows, name='alpha_ok'), np.array(scores), C


def load_onsets(path):
    o = pd.read_csv(path).set_index('name')
    o['stable'] = o.onset.astype(str).str.lower().eq('stable')
    o['onset_lo'] = pd.to_numeric(o.onset_lo, errors='coerce')
    o['onset_hi'] = pd.to_numeric(o.onset_hi, errors='coerce')
    o['onset_str'] = np.where(o.stable, 'stable', o.onset_lo.fillna(0).astype(int).astype(str) + '-'
                              + o.onset_hi.fillna(0).astype(int).astype(str))
    return o


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--domain-path', default=str(HERE.parent / 'domains' / 'greenland'))
    ap.add_argument('--tf-crit', type=float, nargs='+', default=[2.0, 2.5, 3.0, 3.5, 4.0])
    ap.add_argument('--clim-h', type=float, nargs='+', default=[0.0, 5.0, 10.0, 20.0])
    ap.add_argument('--alpha-h', type=float, nargs='+', default=[0.0, 25.0, 50.0, 100.0, 150.0])
    ap.add_argument('--alpha-fine', type=float, nargs=3, default=[0.0, 300.0, 5.0], metavar=('LO', 'HI', 'STEP'),
                    help='the alpha_h grid of the per-front feasibility windows')
    ap.add_argument('--h00', type=float, default=None, help='calving_h0 (default: config)')
    ap.add_argument('--q', type=float, default=None, help='calving_q (default: config)')
    ap.add_argument('--H-c', type=float, default=None, help='calving_H_c (default: config)')
    ap.add_argument('--dtf-scale', type=float, nargs='+', default=[1.0],
                    help='multipliers on the TF anomaly, the forcing-uncertainty axis')
    ap.add_argument('--filter', nargs='+', default=['none'],
                    help="temporal filters on the annual dTF before the model's step aggregation: none | box:N "
                         "(N-yr running mean) | ema:TAU (exponential memory, e-folding TAU yr, zero before the record)")
    ap.add_argument('--schedule', default='1850:10,1990:1', help="the model's dt_schedule; 'annual' for none")
    ap.add_argument('--tol', type=float, default=3.0, help='years beyond the onset window still scored ok')
    ap.add_argument('--windows', type=int, default=3, help='print the alpha windows for the config combo + this many best (tf_crit, clim_h)')
    ap.add_argument('--detail', action='append', default=[], help='tf_crit,clim_h,alpha_h[,dtf_scale[,filter]] to dump a timeline')
    ap.add_argument('--top', type=int, default=12)
    ap.add_argument('--out', default=str(OUT))
    a = ap.parse_args()

    cfg = load_config(a.domain_path)
    ocfg = cfg.ocean_forcing
    h00 = cfg.calving_h0 if a.h00 is None else a.h00
    q = cfg.calving_q if a.q is None else a.q
    H_c = cfg.calving_H_c if a.H_c is None else a.H_c
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print(f"config {cfg.results_subdir}: calving_q {q:g}, calving_h0 {h00:g} m, H_c {H_c:g} m, timescale {cfg.calving_timescale:g} yr, "
          f"TF statistic {ocfg.statistic}, ref {ocfg.ref_years}, max_dist {ocfg.max_dist_km:g} km, "
          f"h0 bounds {ocfg.h0_bounds}; config margins tf_crit {ocfg.tf_crit:g} / clim_h {ocfg.clim_h:g} / "
          f"alpha_h {ocfg.alpha_h:g}" + (f" (pin_front={ocfg.pin_front})" if ocfg.pin_front else ''))

    masks, dx, gi = region_masks(a.domain_path)
    gates = load_gates(GATES, gi)
    names = gpd.read_file(GATES).groupby('gate').Mouginot_2019.first().to_dict()
    gate_names = {g['gate']: str(names.get(g['gate'], f"gate{g['gate']}")) for g in gates}
    fronts, cells = build_fronts(gi, gates, gate_names, H_c, R_KERNEL, cfg.calving_timescale)
    years, clim, ok, dtf_annual = load_forcing(Path(a.domain_path) / 'model_inputs' / ocfg.filename, ocfg, cells)
    S = Screen(fronts, cells, years, clim, ok, dtf_annual, cfg, ocfg)
    onsets = load_onsets(ONSETS)
    missing = [f for f in onsets.index if f not in fronts.index]
    if missing:
        print(f"WARNING: onset-table fronts not found among the gates: {missing}")
    onsets = onsets.loc[[f for f in onsets.index if f in fronts.index]]
    fronts['tongue_2015'] = fronts.n_T >= MIN_TONGUE_CELLS
    fronts['tongue_ever'] = fronts.tongue_2015.values | np.array([bool(onsets.tongue_ever.get(f, 0)) for f in fronts.index])
    fronts['onset'] = fronts.index.map(lambda f: onsets.onset_str.get(f, ''))

    # ---------------------------------------------------------- the census
    schedule = () if a.schedule == 'annual' else tuple((float(s.split(':')[0]), float(s.split(':')[1])) for s in a.schedule.split(','))
    dtf_by_smooth = {}
    for sm in a.filter:
        d = dtf_annual
        if sm.startswith('box:'):
            d = ndimage.uniform_filter1d(d, int(sm[4:]), axis=0, mode='nearest')
        elif sm.startswith('ema:'):
            tau = float(sm[4:]); k = 1.0 - np.exp(-1.0 / tau)
            d = np.empty_like(dtf_annual); acc = np.zeros(dtf_annual.shape[1])
            for j in range(len(years)):
                acc = acc + k * (dtf_annual[j] - acc)
                d[j] = acc
        elif sm != 'none':
            raise SystemExit(f"unknown filter {sm!r}")
        dtf_by_smooth[sm] = step_aggregate(years, d, schedule, cfg.dt) if schedule else d
    dtf_f = np.asarray(S.Mf @ dtf_by_smooth[a.filter[0]].T)              # (nfront, nt) the front's stepped anomaly
    m_all = dtf_f.max(axis=1)
    m1 = np.full(len(fronts), np.nan); m2 = np.full(len(fronts), np.nan)
    for i, f in enumerate(fronts.index):
        if f in onsets.index and not onsets.stable[f]:
            lo, hi = onsets.onset_lo[f] - a.tol, onsets.onset_hi[f] + a.tol
            m1[i] = dtf_f[i, years < lo].max() if (years < lo).any() else -np.inf
            m2[i] = dtf_f[i, years <= hi].max()
    fronts['dtf_max'] = m_all; fronts['dtf_max_before_onset'] = m1; fronts['dtf_max_through_onset'] = m2
    print("\n=== STATIC CENSUS ===")
    c = fronts[np.isfinite(fronts.tf_clim)].sort_values('tf_clim')
    tongue = c.tongue_ever.values
    tfs = c.tf_clim.values
    pct = lambda v: np.percentile(v, [10, 50, 90]).round(2) if len(v) else '-'
    print(f"tongue-bearing (2015 or before): n={int(tongue.sum())}, TF_clim 10/50/90 pct {pct(tfs[tongue])};  "
          f"grounded-only: n={int((~tongue).sum())}, {pct(tfs[~tongue])}")
    print("  " + ", ".join(f"{f[:18]} {t:.1f}" for f, t in zip(c.index[tongue], tfs[tongue])))
    # what DOES separate the tongue-bearing fronts? the front's strain rate, thickness, depth
    print("\n  tongue-bearing vs grounded-only fronts, flux-weighted front properties (10/50/90 pct):")
    for col, lab in (('tf_clim', 'TF_clim (degC)'), ('strain_term', 'principal strain rate at the front (1/yr)'),
                     ('H_term', 'front thickness (m)'), ('depth_term', 'front water depth (m)'), ('hab_term', 'front HAB (m)')):
        v = c[col].values
        t_ = v[tongue & np.isfinite(v)]; g_ = v[~tongue & np.isfinite(v)]
        print(f"    {lab:44s} tongues {pct(t_)}   grounded {pct(g_)}")
    st_ = c.strain_term.values
    fin = np.isfinite(st_)
    cand = np.percentile(st_[fin], np.arange(5, 100, 5))
    best = min(((int((tongue & fin & (st_ >= th)).sum()) + int((~tongue & fin & (st_ < th)).sum()), th) for th in cand))
    tfc = np.arange(1.0, 6.01, 0.25)
    best_tf = min(((int((tongue & (tfs >= th)).sum()) + int((~tongue & (tfs < th)).sum()), th) for th in tfc))
    print(f"    best single threshold: strain {best[1]:.4f} 1/yr -> {best[0]} of {int(fin.sum())} fronts wrong;"
          f"  TF_clim {best_tf[1]:.2f} degC -> {best_tf[0]} of {len(tfs)} wrong")
    print("    tongue fronts' strain: " + ", ".join(f"{f[:12]} {v:.4f}" for f, v in zip(c.index[tongue], st_[tongue])))
    rl = c[~tongue]
    rate_ok = rl.flux_A.values > 0
    print(f"  grounded fronts whose advance is NOT rate-limited (could grow a tongue if h0 < 0): "
          f"{int(rate_ok.sum())} of {len(rl)}; their TF_clim 10/50/90 {pct(rl.tf_clim.values[rate_ok])}; "
          f"below the warmest tongue ({tfs[tongue].max():.2f}): {int((rate_ok & (rl.tf_clim.values < tfs[tongue].max())).sum())}")
    print("\n  onset-table fronts (flux-weighted over the front's cells; HAB = terminus height above flotation, m;"
          " dTF = the front's STEPPED anomaly, K).")
    print("  With q = 0 a floating cell has F = -h0, so a tongue (or a floating advance) flips exactly when"
          " alpha_h dTF crosses -h0_base:")
    print("  a stable front needs -h0_base/alpha_h > max dTF over the record; a retreat needs it in"
          " (max dTF before the window, max dTF through the window] -- the REQUIRED RATIO column.")
    print("  front                      reg  TF_clim  HAB_term   n_T  n_A  dTFmax  before  through  required -h0_base/alpha_h   onset")
    for f in onsets.index:
        r = fronts.loc[f]
        if onsets.stable[f]:
            req = f"> {r.dtf_max:.2f}"
        elif r.dtf_max_through_onset <= r.dtf_max_before_onset:
            req = f"EMPTY (peak {r.dtf_max_before_onset:.2f} before the window)"
        else:
            req = f"({r.dtf_max_before_onset:.2f}, {r.dtf_max_through_onset:.2f}]"
        print(f"  {f[:26]:26s} {r.region:3s}  {r.tf_clim:5.2f}   {r.hab_term:6.0f}   {r.n_T:4d} {r.n_A:4d}  {r.dtf_max:+5.2f}  "
              f"{'' if np.isnan(r.dtf_max_before_onset) else f'{r.dtf_max_before_onset:+5.2f}':6s}  "
              f"{'' if np.isnan(r.dtf_max_through_onset) else f'{r.dtf_max_through_onset:+5.2f}':6s}   {req:34s} {r.onset}")
    fronts.to_csv(out / 'census.csv')

    # ---------------------------------------------------------- the box
    combos = list(itertools.product(a.tf_crit, a.clim_h, a.alpha_h, a.dtf_scale, a.filter))
    wdesc = ' '.join(f"{k} {v:g}" for k, v in CLASS_W.items() if v) + f" over<2015 {W_OVER_T:g} static-cut {W_STATIC_T:g}"
    print(f"\n=== FLIP-YEAR BOX: {len(combos)} combos x {len(fronts)} fronts x {len(years)} years "
          f"(schedule {a.schedule}, tol {a.tol:g} yr; weights per table front: {wdesc}) ===")
    results, flips_all = [], {}
    for (tc, ch, ah, ds_, sm) in combos:
        fr, st, h0_t, h0_base = S.evaluate(tc, ch, ah, h00, q, H_c, ds_, dtf_by_smooth[sm])
        fl = S.flip_years(fr, st)
        s, n, cls = score(fl, onsets, a.tol)
        results.append(dict(tf_crit=tc, clim_h=ch, alpha_h=ah, dtf_scale=ds_, filter=sm, score=s, **n,
                            all_flip_pre1990=int((fl.flip < 1990).sum()), all_flip_by2025=int(np.isfinite(fl.flip).sum()),
                            all_static_cut=int((fl.term0 < 0.5).sum()), all_over_pre2015=int((fl.over < MASK_EPOCH).sum()),
                            all_adv_admissible=int((st['A'] >= 0.5).sum())))
        flips_all[(tc, ch, ah, ds_, sm)] = (fl, cls)
    res = pd.DataFrame(results).sort_values(['score', 'early', 'spurious']).reset_index(drop=True)
    res.to_csv(out / 'scores.csv', index=False)
    cols = ['tf_crit', 'clim_h', 'alpha_h', 'dtf_scale', 'filter', 'score', 'ok', 'early', 'spurious', 'denied', 'late',
            'missed', 'never_adm', 'adv', 'over_pre2015', 'static_cut', 'all_flip_pre1990', 'all_flip_by2025',
            'all_static_cut', 'all_adv_admissible']
    print(res[cols].head(a.top).to_string(index=False))
    key_cfg = (ocfg.tf_crit, ocfg.clim_h, ocfg.alpha_h, 1.0, a.filter[0])
    if key_cfg not in flips_all:
        fr, st, _, _ = S.evaluate(*key_cfg[:3], h00, q, H_c, 1.0, dtf_by_smooth[a.filter[0]])
        fl = S.flip_years(fr, st)
        s, n, cls = score(fl, onsets, a.tol)
        flips_all[key_cfg] = (fl, cls)
        print(f"\nconfig combo {key_cfg[:3]}: score {s:g} {n}")
    else:
        rr = res[(res.tf_crit == key_cfg[0]) & (res.clim_h == key_cfg[1]) & (res.alpha_h == key_cfg[2])
                 & (res.dtf_scale == 1.0) & (res['filter'] == a.filter[0])]
        print(f"\nconfig combo {key_cfg[:3]} ranks {int(rr.index[0]) + 1} of {len(res)}:")
        print(rr[cols].to_string(index=False))

    def label(k):
        return f"tc{k[0]:g}_ch{k[1]:g}_ah{k[2]:g}" + (f"_s{k[3]:g}" if k[3] != 1 else '') + (f"_{k[4].replace(':', '')}" if k[4] != 'none' else '')

    def dump(k, title):
        fl, cls = flips_all[k]
        t = fl.join(fronts[['region', 'tf_clim', 'hab_term', 'tongue_ever', 'onset']]).join(cls, how='left')
        t.to_csv(out / f'flips_{label(k)}.csv')
        print(f"\n--- {title} {k[:3]} (dtf x{k[3]:g}, filter {k[4]}): fronts in the onset table ---")
        tt = t[t.cls.notna()].sort_values(['region', 'tf_clim'])
        print("  front                      reg  TF_clim  set  adm0  advF0 term0   flip   over   onset      class")
        for f, rr in tt.iterrows():
            fy = '-' if np.isnan(rr.flip) else str(int(rr.flip))
            oy = '-' if np.isnan(rr.over) else str(int(rr.over))
            print(f"  {f[:26]:26s} {rr.region:3s}  {rr.tf_clim:5.2f}   {rr['set']}   {rr.adm0:4.2f}  {rr.adv_float0:4.2f}  {rr.term0:4.2f}   "
                  f"{fy:5s}  {oy:5s}  {rr.onset:10s} {rr.cls}")
        rest = t[t.cls.isna() & np.isfinite(t.flip)]
        if len(rest):
            print(f"  + {len(rest)} other fronts flip: " + ", ".join(f"{f[:14]} {int(y)}" for f, y in
                  rest.sort_values('flip').flip.items()))

    dump(key_cfg, 'CONFIG')
    for i in range(min(2, len(res))):
        k = tuple(res.loc[i, ['tf_crit', 'clim_h', 'alpha_h', 'dtf_scale', 'filter']].tolist())
        k = (k[0], k[1], k[2], k[3], str(k[4]))
        if k != key_cfg:
            dump(k, f'BEST #{i + 1}')

    if len(a.dtf_scale) > 1:
        g = res.groupby(['tf_crit', 'clim_h', 'alpha_h', 'filter']).score.agg(['max', 'mean']).reset_index()
        g = g.sort_values(['max', 'mean']).head(a.top)
        print(f"\n=== min-max over dtf_scale {a.dtf_scale}: worst-case score per margin combo ===")
        print(g.to_string(index=False))

    # ------------------------------------------------- alpha_h windows per front
    alphas = np.arange(a.alpha_fine[0], a.alpha_fine[1] + 1e-9, a.alpha_fine[2])
    print(f"\n=== ALPHA_H FEASIBILITY WINDOWS (alpha_h {alphas[0]:g}..{alphas[-1]:g} step {a.alpha_fine[2]:g}; dtf x1, filter {a.filter[0]}) ===")
    win_rows = []
    grid = list(itertools.product(sorted(set(a.tf_crit) | {ocfg.tf_crit}), sorted(set(a.clim_h) | {ocfg.clim_h})))
    per_grid = {}
    for tc, ch in grid:
        w, sc, C = alpha_windows(S, tc, ch, h00, q, H_c, 1.0, dtf_by_smooth[a.filter[0]], onsets, a.tol, alphas)
        n_ok = (C == 'ok').sum(axis=0).values
        j = int(np.argmin(sc))
        per_grid[(tc, ch)] = (w, sc, C)
        win_rows.append(dict(tf_crit=tc, clim_h=ch, best_alpha=alphas[j], best_score=sc[j], best_n_ok=int(n_ok[j]),
                             max_n_ok=int(n_ok.max()), alpha_at_max_ok=alphas[int(np.argmax(n_ok))],
                             n_any=int((w == 'any').sum()), n_none=int(w.str.startswith('none').sum())))
    W = pd.DataFrame(win_rows).sort_values(['max_n_ok', 'best_score'], ascending=[False, True])
    W.to_csv(out / 'windows_summary.csv', index=False)
    print("per (tf_crit, clim_h): the most fronts simultaneously 'ok' at any alpha_h (sorted), and the score-minimizing alpha_h"
          " (the score's asymmetry makes alpha_h = 0, nothing ever flips, hard to beat: read max_n_ok)")
    print(W.to_string(index=False))

    def show_windows(tc, ch, title):
        w, sc, C = per_grid[(tc, ch)]
        j = int(np.argmin(sc))
        base = h00 + ch * (fronts.tf_clim - tc)
        print(f"\n--- {title} tf_crit {tc:g}, clim_h {ch:g}: best alpha_h {alphas[j]:g} (score {sc[j]:g}) ---")
        print("  front                      reg  TF_clim  h0_base  HAB_term  set  onset       alpha_h where 'ok'      class@best")
        t = fronts.loc[onsets.index].sort_values(['region', 'tf_clim'])
        for f, r in t.iterrows():
            print(f"  {f[:26]:26s} {r.region:3s}  {r.tf_clim:5.2f}   {base[f]:+6.1f}   {r.hab_term:6.0f}   "
                  f"{'T' if r.tongue_2015 else 'A'}   {r.onset:10s}  {w[f]:24s} {C.loc[f].iloc[j]}")
        C.to_csv(out / f'windows_tc{tc:g}_ch{ch:g}.csv')

    show_windows(ocfg.tf_crit, ocfg.clim_h, 'CONFIG')
    shown = {(ocfg.tf_crit, ocfg.clim_h)}
    for _, r in W.iterrows():
        if len(shown) > a.windows:
            break
        k = (r.tf_crit, r.clim_h)
        if k not in shown:
            show_windows(*k, f'BEST #{len(shown)}')
            shown.add(k)

    for spec in a.detail:
        v = [float(x) for x in spec.split(',')]
        tc, ch, ah = v[:3]
        ds_ = v[3] if len(v) > 3 else 1.0
        sm = str(v[4]) if len(v) > 4 else a.filter[0]
        fr, st, h0_t, h0_base = S.evaluate(tc, ch, ah, h00, q, H_c, ds_, dtf_by_smooth.get(sm, dtf_by_smooth[a.filter[0]]))
        h0_f = np.asarray(S.Mf @ h0_t.T)
        rows = []
        for i, f in enumerate(fronts.index):
            for jj, y in enumerate(years):
                rows.append(dict(front=f, year=int(y), h0_front=h0_f[i, jj], frac_T=fr['T'][i, jj],
                                 frac_A=fr['A'][i, jj], frac_term=fr['term'][i, jj]))
        pd.DataFrame(rows).to_csv(out / f'timeline_{label((tc, ch, ah, ds_, sm))}.csv', index=False)
        print(f"timeline written for {spec}")
    print(f"\noutputs in {out}")


if __name__ == '__main__':
    main()
