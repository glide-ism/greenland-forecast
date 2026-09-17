"""Does a T2m -> T100 correction hold water?

The enthalpy SMB model is forced with CARRA2's 100 m-above-ground air
temperature (the free air above the melt-depleted boundary layer), while
the ISMIP7 kit provides only a 2 m temperature (dEBM2-downscaled CESM2
`tas`). This script tests the proposal to map the kit's tas onto the model's
convention with a relation fitted on CARRA2 itself:

  A. CARRA2 native grid, 1986-2025 monthly: the deficit d = T100 - T2m per
     calendar month over the ice, its dependence on T2m (saturation of T2m
     near 0 degC on melting surfaces) and on elevation, and the per-cell
     interannual regression T100 = a + b T2m (slope b > 1 where T2m
     saturates). Hold-out test: fit on 1986-2005, predict 2006-2025 (a
     warmer period, the direction a projection extrapolates in) with
       (i)   T100 = T2m                       (no correction)
       (ii)  T100 = T2m + d_clim(x, m)        (climatological offset)
       (iii) T100 = a(x, m) + b(x, m) T2m     (per-cell regression)
       (iv)  T100 = T2m + f_m(T2m)            (pooled saturation curve per month)
  B. Domain grid: CARRA2 T2m lapse-corrected onto the DEM exactly like T100
     (the model's production path) against the dEBM2 tas climatology of the
     same window, by month and elevation band: is CARRA2 T2m ~ CESM2 tas,
     i.e. is the whole CARRA2-T100-vs-CESM2 bias the boundary-layer deficit,
     or is there a GCM bias on top?
  C. Interannual variability of the JJA ice-sheet mean: CARRA2 T2m, T100 and
     CESM2 tas.

    python analysis/compare_t2m_t100.py --domain-path domains/greenland \
        --forcing-dir domains/greenland/model_inputs/ismip7/CESM2-WACCM_ssp126
Writes tables and figures to analysis/output/t2m_t100/.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'preprocessing'))
from racmo_common import load_grid
import make_carra_vars as mcv

MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
JJA = [5, 6, 7]
BANDS = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 4000)]
T_BINS = np.arange(-45, 12, 1.0)


def carra_ice_fraction(carra, cx, cy, grid):
    """Ice fraction and mean DEM elevation of every CARRA cell, by binning
    the 1 km domain cells into the CARRA grid."""
    gx, gy = np.meshgrid(grid.x.values.astype('float64'), grid.y.values.astype('float64'))
    qx, qy = carra.transform_to(gx, gy, mcv.grid_from_dem(grid).crs)
    dxc, dyc = float(cx[1] - cx[0]), float(cy[1] - cy[0])
    ix = np.round((qx - cx[0]) / dxc).astype(int)
    iy = np.round((qy - cy[0]) / dyc).astype(int)
    ok = (ix >= 0) & (ix < len(cx)) & (iy >= 0) & (iy < len(cy))
    ice = grid.rgi_mask.values > 0.5
    z = np.maximum(grid.elevation.values.astype('float64'), 0.0)
    n = np.zeros((len(cy), len(cx))); ni = np.zeros_like(n); zs = np.zeros_like(n)
    np.add.at(n, (iy[ok], ix[ok]), 1.0)
    np.add.at(ni, (iy[ok], ix[ok]), ice[ok].astype(float))
    np.add.at(zs, (iy[ok], ix[ok]), z[ok])
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(n > 0, ni / n, 0.0), np.where(n > 0, zs / n, np.nan)


def load_native(base, years, domain_path):
    """(T2m, T100) float32 (n_months, ny, nx) degC on the CARRA box around
    the ice, the calendar months and years of each record, the ice fraction,
    DEM elevation, orography and native coordinates."""
    carra, ds_t, _, orog, cx, cy, tdim = mcv.open_carra(base)
    ds_2 = xr.open_dataset(Path(base) / '1985_2025' / 't2m.nc')
    ds_2 = ds_2.assign_coords({tdim: ds_2['valid_time'].values}) if 'valid_time' in ds_2 else ds_2
    # the same crop as open_carra applied (finite box); recover it from sizes
    probe = ds_t['t'].isel({tdim: 0, 'heightAboveGround': 0})
    full = ds_2['t2m'].isel({tdim: 0}).values
    fin = np.isfinite(full)
    rows, cols = np.where(fin.any(1))[0], np.where(fin.any(0))[0]
    ys, xs = slice(int(rows[0]), int(rows[-1]) + 1), slice(int(cols[0]), int(cols[-1]) + 1)
    ds_2 = ds_2.isel(y=ys, x=xs)
    assert ds_2.sizes['y'] == probe.sizes['y'] and ds_2.sizes['x'] == probe.sizes['x']
    grid = load_grid(domain_path)
    frac, zdem = carra_ice_fraction(carra, cx, cy, grid)
    r, c = np.where(frac > 0)
    ys2 = slice(max(r.min() - 2, 0), r.max() + 3); xs2 = slice(max(c.min() - 2, 0), c.max() + 3)
    frac, zdem, orog = frac[ys2, xs2], zdem[ys2, xs2], orog[ys2, xs2]
    levels = ds_t['heightAboveGround'].values
    i_lo = int(np.argmin(np.abs(levels - mcv.LEVEL_LOW)))
    t100 = ds_t['t'].isel(heightAboveGround=i_lo, y=ys2, x=xs2)
    t2m = ds_2['t2m'].isel(y=ys2, x=xs2)
    t100 = mcv._select_years(t100, tdim, years); t2m = mcv._select_years(t2m, tdim, years)
    assert np.array_equal(t100[tdim].values, t2m[tdim].values)
    months = t100[tdim].dt.month.values - 1
    yrs = t100[tdim].dt.year.values
    n = len(months)
    A = np.empty((n,) + frac.shape, np.float32); B = np.empty_like(A)
    for i0 in range(0, n, 48):
        A[i0:i0 + 48] = t2m.isel({tdim: slice(i0, i0 + 48)}).values - 273.15
        B[i0:i0 + 48] = t100.isel({tdim: slice(i0, i0 + 48)}).values - 273.15
        print(f"  read months {i0}-{min(i0 + 48, n)} of {n}", flush=True)
    return A, B, months, yrs, frac, zdem, orog, cx[xs2], cy[ys2], carra, grid


def fit_cell_regression(T2, T1, months, sel):
    """Per cell and calendar month: mean offset d, slope b, intercept a, r2
    over the records selected by `sel` (boolean over time)."""
    ny, nx = T2.shape[1:]
    d = np.zeros((12, ny, nx)); a = np.zeros_like(d); b = np.zeros_like(d); r2 = np.zeros_like(d)
    for m in range(12):
        k = sel & (months == m)
        x, y = T2[k].astype('float64'), T1[k].astype('float64')
        mx, my = x.mean(0), y.mean(0)
        vx = ((x - mx) ** 2).mean(0); cxy = ((x - mx) * (y - my)).mean(0); vy = ((y - my) ** 2).mean(0)
        with np.errstate(invalid='ignore', divide='ignore'):
            bb = np.where(vx > 1e-6, cxy / vx, 1.0)
            rr = np.where((vx > 1e-6) & (vy > 1e-6), cxy ** 2 / (vx * vy), 0.0)
        b[m], r2[m], d[m] = bb, rr, my - mx
        a[m] = my - bb * mx
    return d, a, b, r2


def pooled_curve(T2, T1, months, sel, w):
    """Mean deficit per (month, T2m bin) over the ice (weights w)."""
    curve = np.full((12, len(T_BINS) - 1), np.nan); cnt = np.zeros((12, len(T_BINS) - 1))
    for m in range(12):
        k = sel & (months == m)
        x = T2[k].ravel(); dd = (T1[k] - T2[k]).ravel(); ww = np.broadcast_to(w, T2[k].shape).ravel()
        good = ww > 0
        idx = np.digitize(x[good], T_BINS) - 1
        inb = (idx >= 0) & (idx < len(T_BINS) - 1)
        s = np.bincount(idx[inb], weights=(dd[good] * ww[good])[inb], minlength=len(T_BINS) - 1)
        c = np.bincount(idx[inb], weights=ww[good][inb], minlength=len(T_BINS) - 1)
        cnt[m] = c
        with np.errstate(invalid='ignore', divide='ignore'):
            curve[m] = np.where(c > 0, s / c, np.nan)
    return curve, cnt


def apply_curve(curve, m, x):
    """Deficit from the pooled curve of month m at temperatures x (linear
    interpolation between bin centres; edges held)."""
    centres = 0.5 * (T_BINS[1:] + T_BINS[:-1])
    ok = np.isfinite(curve[m])
    return np.interp(x, centres[ok], curve[m][ok])


def holdout(T2, T1, months, yrs, w, train, test):
    d, a, b, r2 = fit_cell_regression(T2, T1, months, train)
    curve, _ = pooled_curve(T2, T1, months, train, w)
    rows = []
    for name, sel_m in (('JJA', JJA), ('annual', list(range(12)))):
        err = {k: [] for k in ('none', 'offset', 'regression', 'curve')}
        wts = []
        for i in np.where(test)[0]:
            m = months[i]
            if m not in sel_m:
                continue
            x, y = T2[i].astype('float64'), T1[i].astype('float64')
            preds = {'none': x, 'offset': x + d[m], 'regression': a[m] + b[m] * x, 'curve': x + apply_curve(curve, m, x)}
            for k, p in preds.items():
                err[k].append(p - y)
            wts.append(w)
        W = np.stack(wts); Wsum = W.sum()
        for k, e in err.items():
            E = np.nan_to_num(np.stack(e))
            rows.append(dict(window=name, model=k, bias=float((E * W).sum() / Wsum),
                             rmse=float(np.sqrt((E ** 2 * W).sum() / Wsum)),
                             mae=float((np.abs(E) * W).sum() / Wsum)))
    return pd.DataFrame(rows), d, a, b, r2, curve


def band_table(field12, z, w, bands=BANDS):
    out = {}
    for lo, hi in bands:
        m = (z >= lo) & (z < hi) & (w > 0)
        ww = w[m]
        out[f"{lo}-{hi}"] = [float((field12[k][m] * ww).sum() / ww.sum()) for k in range(12)]
    return pd.DataFrame(out, index=MONTHS).T


def main(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    years = list(range(args.years[0], args.years[1] + 1))
    base = HERE.parent / 'common_data' / 'climate' / 'carra2'
    print("A. loading CARRA2 T2m and T100 on the native grid")
    T2, T1, months, yrs, frac, zdem, orog, cx, cy, carra, grid = load_native(base, years, args.domain_path)
    valid = np.isfinite(T2).all(0) & np.isfinite(T1).all(0)   # CARRA's domain is not a rectangle
    w = np.where(valid, frac, 0.0)                        # ice-fraction weights
    ice = (frac > 0.5) & valid
    print(f"   {int(((frac > 0.5) & ~valid).sum())} ice cells outside the CARRA2 domain dropped")
    print(f"   box {T2.shape[1:]} cells, {ice.sum()} ice cells (fraction > 0.5), {len(yrs)} months")

    allsel = np.ones(len(months), bool)
    d, a, b, r2 = fit_cell_regression(T2, T1, months, allsel)
    zc = np.where(np.isfinite(zdem), zdem, orog)
    tab_d = band_table(d, zc, w); tab_b = band_table(b, zc, w); tab_r2 = band_table(r2, zc, w)
    clim2 = np.stack([T2[months == m].mean(0) for m in range(12)])
    tab_t2 = band_table(clim2, zc, w)
    print("\nT100 - T2m climatological deficit (K) by month and DEM elevation band, ice cells:")
    print(tab_d.round(2).to_string())
    print("\ninterannual slope dT100/dT2m by month and band:")
    print(tab_b.round(2).to_string())
    print("\nr2 of the per-cell regression:")
    print(tab_r2.round(2).to_string())
    for name, t in (('deficit', tab_d), ('slope', tab_b), ('r2', tab_r2), ('t2m_clim', tab_t2)):
        t.to_csv(out / f"A_{name}_by_band.csv")

    curve, cnt = pooled_curve(T2, T1, months, allsel, w)
    centres = 0.5 * (T_BINS[1:] + T_BINS[:-1])
    pd.DataFrame(curve, index=MONTHS, columns=np.round(centres, 1)).to_csv(out / "A_deficit_vs_t2m.csv")
    print("\npooled deficit (K) vs T2m bin, JJA (ice-fraction weighted):")
    sel = (centres >= -20) & (centres <= 8)
    print(pd.DataFrame(curve[JJA][:, sel], index=[MONTHS[m] for m in JJA], columns=np.round(centres[sel], 0)).round(2).to_string())

    train = yrs <= args.split; test = yrs > args.split
    print(f"\nhold-out: fit {years[0]}-{args.split}, test {args.split + 1}-{years[-1]}")
    ho, d_tr, a_tr, b_tr, r2_tr, curve_tr = holdout(T2, T1, months, yrs, w, train, test)
    print(ho.round(3).to_string(index=False))
    ho.to_csv(out / "A_holdout.csv", index=False)
    # warming between the halves, as context for the extrapolation
    jja = np.isin(months, JJA)
    warm2 = np.nan_to_num(T2[test & jja].mean(0) - T2[train & jja].mean(0))
    warm1 = np.nan_to_num(T1[test & jja].mean(0) - T1[train & jja].mean(0))
    print(f"JJA warming test-train, ice mean: T2m {np.average(warm2, weights=w):+.2f} K, T100 {np.average(warm1, weights=w):+.2f} K")

    # ---- figures for A
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    ax = axs[0]
    for lo, hi in BANDS:
        ax.plot(range(12), tab_d.loc[f"{lo}-{hi}"], label=f"{lo}-{hi} m")
    ax.set_xticks(range(12)); ax.set_xticklabels([m[0] for m in MONTHS]); ax.set_ylabel("T100 - T2m (K)")
    ax.set_title("CARRA2 deficit by elevation band"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax = axs[1]
    for m in JJA + [3, 8]:
        ax.plot(centres, curve[m], label=MONTHS[m])
    ax.set_xlabel("T2m (degC)"); ax.set_ylabel("T100 - T2m (K)"); ax.set_xlim(-30, 10)
    ax.set_title("pooled deficit vs T2m (ice)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax = axs[2]
    for lo, hi in BANDS:
        ax.plot(range(12), tab_b.loc[f"{lo}-{hi}"], label=f"{lo}-{hi} m")
    ax.axhline(1, color='k', lw=.5); ax.set_xticks(range(12)); ax.set_xticklabels([m[0] for m in MONTHS])
    ax.set_ylabel("dT100/dT2m"); ax.set_title("interannual slope"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "A_deficit.png", dpi=130); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, f, ttl, kw in ((axs[0], np.where(ice, d[JJA].mean(0), np.nan), "JJA deficit T100-T2m (K)", dict(vmin=0, vmax=6, cmap='magma')),
                           (axs[1], np.where(ice, b[JJA].mean(0), np.nan), "JJA slope dT100/dT2m", dict(vmin=0.5, vmax=2.0, cmap='RdBu_r')),
                           (axs[2], np.where(ice, r2[JJA].mean(0), np.nan), "JJA r2", dict(vmin=0, vmax=1, cmap='viridis'))):
        im = ax.imshow(f, origin='lower' if cy[1] > cy[0] else 'upper', **kw); ax.set_title(ttl); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout(); fig.savefig(out / "A_maps_jja.png", dpi=130); plt.close(fig)

    # ---- B. domain grid: CARRA2 T2m on the DEM vs dEBM2 tas climatology
    print("\nB. CARRA2 T2m lapse-corrected onto the DEM vs dEBM2 tas (same window)")
    fd = Path(args.forcing_dir)
    clim = xr.open_dataset(fd / 'climate.nc')
    tas_c = clim['tas_clim'].values                       # (12, ny, nx) degC, NaN off footprint
    gi = xr.open_dataset(Path(args.domain_path) / 'model_inputs' / 'GLIDE_inputs.nc')
    z = np.maximum(gi.elevation.values.astype('float64'), 0.0)
    lapse = gi.monthly_lapse_rate.values; zref = gi.carra_orog.values
    t100_dem = gi.monthly_t2m.values                      # the model's forcing (T100 on the DEM)
    gx, gy = np.meshgrid(gi.x.values.astype('float64'), gi.y.values.astype('float64'))
    qx, qy = carra.transform_to(gx, gy, mcv.grid_from_dem(gi).crs)
    t2m_dem = np.empty_like(t100_dem); d_dem = np.empty_like(t100_dem)
    for m in range(12):
        t2 = mcv._interp(cx, cy, clim2[m], qx, qy)
        t2m_dem[m] = mcv._fill_nearest((t2 + lapse[m] * (z - zref)).astype('float32'))
        d_dem[m] = mcv._fill_nearest(mcv._interp(cx, cy, d[m], qx, qy).astype('float32'))
    ice_d = (gi.rgi_mask.values > 0.5) & np.isfinite(tas_c[0])
    zi = gi.elevation.values
    rows = []
    for lo, hi in BANDS:
        msk = ice_d & (zi >= lo) & (zi < hi)
        for k in range(12):
            rows.append(dict(band=f"{lo}-{hi}", month=MONTHS[k],
                             carra_t2m=float(t2m_dem[k][msk].mean()), carra_t100=float(t100_dem[k][msk].mean()),
                             cesm_tas=float(tas_c[k][msk].mean()), deficit=float(d_dem[k][msk].mean())))
    B = pd.DataFrame(rows)
    B['t2m_minus_cesm'] = B.carra_t2m - B.cesm_tas
    B['t100_minus_cesm'] = B.carra_t100 - B.cesm_tas
    B.to_csv(out / "B_domain_by_band_month.csv", index=False)
    piv = lambda col: B.pivot(index='band', columns='month', values=col)[MONTHS].loc[[f"{lo}-{hi}" for lo, hi in BANDS]]
    print("\nCARRA2 T2m(DEM) - CESM2 tas (K): the GCM bias left after the boundary-layer deficit")
    print(piv('t2m_minus_cesm').round(2).to_string())
    print("\nCARRA2 T100(DEM) - CESM2 tas (K): the bias the raw run sees")
    print(piv('t100_minus_cesm').round(2).to_string())
    print("\nT100 - T2m deficit interpolated to the DEM (K)")
    print(piv('deficit').round(2).to_string())
    msk = ice_d
    summary = dict(
        annual_t100_minus_cesm=float(np.mean([(t100_dem[k][msk] - tas_c[k][msk]).mean() for k in range(12)])),
        annual_t2m_minus_cesm=float(np.mean([(t2m_dem[k][msk] - tas_c[k][msk]).mean() for k in range(12)])),
        jja_t100_minus_cesm=float(np.mean([(t100_dem[k][msk] - tas_c[k][msk]).mean() for k in JJA])),
        jja_t2m_minus_cesm=float(np.mean([(t2m_dem[k][msk] - tas_c[k][msk]).mean() for k in JJA])),
        jja_deficit=float(np.mean([d_dem[k][msk].mean() for k in JJA])))
    print("\nice-sheet means:", json.dumps({k: round(v, 2) for k, v in summary.items()}))
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, f, ttl in ((axs[0], np.where(ice_d, t100_dem[JJA].mean(0) - tas_c[JJA].mean(0), np.nan), "JJA CARRA2 T100 - CESM2 tas (K)"),
                       (axs[1], np.where(ice_d, t2m_dem[JJA].mean(0) - tas_c[JJA].mean(0), np.nan), "JJA CARRA2 T2m - CESM2 tas (K)"),
                       (axs[2], np.where(ice_d, d_dem[JJA].mean(0), np.nan), "JJA deficit T100 - T2m (K)")):
        im = ax.imshow(f, vmin=-4, vmax=4, cmap='RdBu_r'); ax.set_title(ttl); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout(); fig.savefig(out / "B_maps_jja.png", dpi=130); plt.close(fig)

    # ---- C. interannual variability of the JJA ice-sheet mean
    print("\nC. JJA ice-sheet-mean interannual variability")
    with open(fd / 'catalogue.json') as f:
        cat = json.load(f)
    flip = cat['grid_mapping'] == 'flip_y'
    ice_full = gi.rgi_mask.values > 0.5
    ice_asc = ice_full[::-1] if flip else ice_full
    cesm = {}
    for y in years:
        fp = cat['years'][str(y)]['tas']
        with xr.open_dataset(fp, decode_times=False) as ds:
            a3 = ds['tas'].isel(time=JJA).values
        fin = np.isfinite(a3[0]) & ice_asc
        cesm[y] = float(a3[:, fin].mean() - 273.15)
    cesm = pd.Series(cesm)
    c2 = pd.Series({y: float(np.average(np.nan_to_num(T2[(yrs == y) & jja].mean(0)), weights=w)) for y in years})
    c1 = pd.Series({y: float(np.average(np.nan_to_num(T1[(yrs == y) & jja].mean(0)), weights=w)) for y in years})
    C = pd.DataFrame({'carra_t2m': c2, 'carra_t100': c1, 'cesm_tas': cesm})
    C.to_csv(out / "C_jja_series.csv")
    det = C - C.rolling(11, center=True, min_periods=5).mean()
    print("mean:", C.mean().round(2).to_dict()); print("std:", C.std().round(2).to_dict())
    print("std detrended (11-yr):", det.std().round(2).to_dict())
    print("corr(T2m, T100) CARRA JJA: %.2f; slope dT100/dT2m of ice-mean: %.2f" % (
        C.carra_t2m.corr(C.carra_t100), np.polyfit(C.carra_t2m, C.carra_t100, 1)[0]))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for k in C:
        ax.plot(C.index, C[k], label=k)
    ax.set_ylabel("JJA ice-sheet mean (degC)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "C_jja_series.png", dpi=130); plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain-path", default="domains/greenland")
    ap.add_argument("--forcing-dir", default="domains/greenland/model_inputs/ismip7/CESM2-WACCM_ssp126")
    ap.add_argument("--years", type=int, nargs=2, default=(1986, 2025))
    ap.add_argument("--split", type=int, default=2005, help="last training year of the hold-out test")
    ap.add_argument("--out-dir", default=str(HERE / "output" / "t2m_t100"))
    args = ap.parse_args()
    main(args)
