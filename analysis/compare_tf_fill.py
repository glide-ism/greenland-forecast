"""Side-by-side of the two thermal-forcing fills (make_thermal_forcing.py
--method nearest | marine): one year's annual mean, the 2011-2010 change and
the interannual std around five fronts, plus domain statistics of the
under-ice cells. Usage:
    python analysis/compare_tf_fill.py --nearest domains/greenland/model_inputs/thermal_forcing.nc \
        --marine domains/greenland/model_inputs/thermal_forcing_marine.nc
"""
import argparse
from pathlib import Path
import numpy as np, xarray as xr, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

SITES = {"Jakobshavn": (-180e3, -2280e3), "Helheim": (310e3, -2580e3), "Petermann": (-280e3, -1000e3),
         "79N / Zachariae": (450e3, -1100e3), "Kangerlussuaq": (490e3, -2290e3)}

def main(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    gi = xr.open_dataset(Path(a.domain_path) / "model_inputs" / "GLIDE_inputs.nc")
    ice = gi.rgi_mask.values > 0.5; bed = gi.bed_obs.values; x, y = gi.x.values, gi.y.values
    files = {"nearest": a.nearest, "marine": a.marine}
    F = {k: xr.open_dataset(v) for k, v in files.items()}
    year = a.year; k1 = year + 1
    fields = {}
    for k, ds in F.items():
        tm = ds.tf_mean; ok = ds.tf_dist.values <= 5.0
        fields[k] = dict(mean=np.where(ok, tm.sel(time=year).values, np.nan),
                         diff=np.where(ok, tm.sel(time=k1).values - tm.sel(time=year).values, np.nan),
                         std=np.where(ok, tm.std("time").values, np.nan), ok=ok)
        m = ok & ice
        print(f"{k:8s}: active cells {ok.sum()}, under ice {m.sum()}; under-ice interannual std mean {np.nanmean(fields[k]['std'][m]):.2f} K, "
              f"90th pct {np.nanpercentile(fields[k]['std'][m], 90):.2f}, max {np.nanmax(fields[k]['std'][m]):.2f}; "
              f"mean |{k1}-{year}| {np.nanmean(np.abs(fields[k]['diff'][m])):.2f} K")
    rows = [("mean", f"tf_mean {year} (degC)", dict(vmin=0, vmax=7, cmap="magma")),
            ("diff", f"tf_mean {k1} minus {year} (K)", dict(vmin=-1.5, vmax=1.5, cmap="RdBu_r")),
            ("std", "interannual std (K)", dict(vmin=0, vmax=1.2, cmap="viridis"))]
    fig, axs = plt.subplots(len(rows) * 2, len(SITES), figsize=(4.2 * len(SITES), 3.9 * len(rows) * 2))
    for j, (name, (cx, cy)) in enumerate(SITES.items()):
        ix = slice(np.searchsorted(x, cx - 50e3), np.searchsorted(x, cx + 50e3)); iy = slice(np.searchsorted(-y, -(cy + 50e3)), np.searchsorted(-y, -(cy - 50e3)))
        ext = [x[ix][0] / 1e3, x[ix][-1] / 1e3, y[iy][-1] / 1e3, y[iy][0] / 1e3]
        for r, (key, ttl, kw) in enumerate(rows):
            for m_, meth in enumerate(("nearest", "marine")):
                ax = axs[2 * r + m_, j]
                im = ax.imshow(fields[meth][key][iy, ix], extent=ext, **kw)
                ax.contour(x[ix] / 1e3, y[iy] / 1e3, ice[iy, ix].astype(float), levels=[0.5], colors="cyan", linewidths=0.5)
                ax.contour(x[ix] / 1e3, y[iy] / 1e3, np.nan_to_num(bed[iy, ix], nan=1.0), levels=[0], colors="white", linewidths=0.4)
                ax.set_title(f"{name}\n{meth}: {ttl}" if r == 0 else f"{meth}: {ttl}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
                plt.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle("thermal forcing at the cells OceanForcing uses (tf_dist <= 5 km); cyan: ice mask, white: bed = 0", y=0.995)
    fig.tight_layout(); fig.savefig(out / "tf_fill_comparison.png", dpi=100); print(f"wrote {out / 'tf_fill_comparison.png'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain-path", default="domains/greenland")
    ap.add_argument("--nearest", default="domains/greenland/model_inputs/thermal_forcing_nearest.nc")
    ap.add_argument("--marine", default="domains/greenland/model_inputs/thermal_forcing.nc")
    ap.add_argument("--year", type=int, default=2010)
    ap.add_argument("--out-dir", default="analysis/output/tf_fill")
    main(ap.parse_args())
