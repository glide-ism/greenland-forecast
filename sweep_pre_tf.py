#!/usr/bin/env python
"""
Calibrate the pre-record held thermal forcing (config.OceanForcingConfig
pre_record_scale / pre_record_offset / pre_record_ramp; glacier_inverse/
ocean.py): before EN4 starts (1950) the TF is held at

    TF_pre(x) = max(scale * TF_clim(x) + offset, 0)

and the two coefficients are chosen so the FREE spin-up (no front pin)
reaches the Little Ice Age extent, instead of holding the fronts at it.

`run`: one forward_standalone run per (scale, offset) pair of the grid, from
the config's spin-up start to --t-end (default 1950: only the spin-up is
needed), with the front pins switched off and every other ocean / calving
setting as configured (or overridden here). Only raw states at
--state-times are kept (no VTI unless --vti). Resumable, like
sweep_calving_c.py; {out_root}/sweep.json lists the runs.

`score`: per run and state time, against `lia_mask` of
model_inputs/front_mask_lia.nc (GRISHM, preprocessing/make_front_mask_lia.py)
and the 2015 inventory `rgi_mask`, on MARINE cells (bed < 0) assigned to a
calving basin (model_inputs/calving_basins.nc), per Mouginot region:
  cover   = fraction of the LIA retreat zone (LIA ice, not 2015 ice) holding
            ice thicker than --h-thr (judge on thickness: a cell with a few
            metres of ice in transit is not a front)
  keep    = the same fraction over the 2015 marine ice (a spin-up that loses
            the modern fronts is not an LIA state either)
  over    = km^2 of ice thicker than --h-thr OUTSIDE the LIA extent (advance
            past the LIA maximum)
Written to {out_root}/score.csv (long) and printed as a table.

Usage:
  python sweep_pre_tf.py run --scale 1.0 0.8 0.6 --offset 0 -0.5 -1.0 --level 1
  python sweep_pre_tf.py score --state-time 1900
"""
import argparse
import dataclasses
import itertools
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_STATE_TIMES = (1700.0, 1850.0, 1900.0, 1950.0)


def run_dir(root: Path, s: float, o: float) -> Path:
    return root / f"s{s:.2f}_o{o:+.2f}"


def cmd_run(a):
    import cupy as cp
    import forward_standalone as fs

    out_root = Path(a.out_root or f"{fs.config.output_dir}/sweep_pre_tf")
    out_root.mkdir(parents=True, exist_ok=True)
    stale = (fs.PHYSICAL_PATH.exists() and Path(fs.CHECKPOINT).exists()
             and Path(fs.CHECKPOINT).stat().st_mtime > fs.PHYSICAL_PATH.stat().st_mtime)
    if stale or not fs.PHYSICAL_PATH.exists():
        fs.export_physical_fields(fs.CHECKPOINT, fs.PHYSICAL_PATH)
    pins = {} if (a.keep_pin or a.release_year is not None) else dict(pin_front=None, pin_front_filename=None)
    if a.release_year is not None:
        pins["pin_release_year"] = float(a.release_year)
    if a.index_k is not None:
        pins.update(pre_record_index=a.index, pre_record_index_k=float(a.index_k), pre_record_index_smooth=int(a.index_smooth),
                    pre_record_index_start=a.index_start)
    base = dataclasses.replace(fs.config.ocean_forcing, enabled=True, **pins,
                               pre_record_ramp=float(a.ramp))
    over = {k: v for k, v in dict(alpha_h=a.alpha_h, alpha_q=a.alpha_q).items() if v is not None}
    if a.no_rho:
        over["rho_filename"] = None
    elif a.rho_filename:
        over["rho_filename"] = a.rho_filename
    base = dataclasses.replace(base, **over)
    if a.calving_h0 is not None:
        fs.H00 = float(a.calving_h0)
    if a.t_start is not None:
        fs.T_START = float(a.t_start)
    fs.T_END = float(a.t_end)
    fs.STATE_SAVE_TIMES = tuple(float(t) for t in a.state_times if t <= a.t_end + 1e-6)
    fs.VTI_T_MIN = (None if a.vti_from is None else float(a.vti_from)) if a.vti else 1e9   # no frames by default
    fs.SNAP_TIMES = tuple(sorted(set(tuple(getattr(fs, "SNAP_TIMES", ())) + fs.STATE_SAVE_TIMES)))
    common = dict(index_k=a.index_k, index_start=a.index_start, index_smooth=a.index_smooth, keep_pin=bool(a.keep_pin), release_year=a.release_year, level=a.level, t_start=fs.T_START, t_end=fs.T_END, ramp=a.ramp, state_times=list(fs.STATE_SAVE_TIMES),
                  alpha_h=base.alpha_h, alpha_q=base.alpha_q, rho_filename=base.rho_filename, tf_crit=base.tf_crit,
                  clim_h=base.clim_h, statistic=base.statistic, calving_h0=fs.H00, calving_q=fs.Q0,
                  calving_timescale=fs.config.calving_timescale, calving_H_c=fs.config.calving_H_c,
                  checkpoint=str(fs.CHECKPOINT), config=fs.config.results_subdir)
    man_path = out_root / "sweep.json"
    prev = json.loads(man_path.read_text()).get("runs", []) if man_path.exists() else []
    runs = {(r["scale"], r["offset"]): r for r in prev}
    for s, o in itertools.product(a.scale, a.offset):
        out = run_dir(out_root, s, o)
        if (out / "forward_soln.nc").exists() and not a.force:
            print(f"--- scale {s:g} offset {o:+g}: exists, skipped", flush=True)
            continue
        fs.OCEAN = dataclasses.replace(base, pre_record_scale=float(s), pre_record_offset=float(o))
        t0 = time.time()
        print(f"--- scale {s:g} offset {o:+g} -> {out}", flush=True)
        ctx = fs.setup(level=a.level, out_dir=out)
        fs.run(ctx)
        wall = time.time() - t0
        rec = dict(scale=s, offset=o, dir=str(out), wall_s=wall, finished=datetime.now().isoformat(timespec="seconds"),
                   ocean=ctx.ocean.describe() if ctx.ocean is not None else "none")
        (out / "manifest.json").write_text(json.dumps(dict(**common, **rec), indent=1))
        runs[(s, o)] = rec
        man_path.write_text(json.dumps(dict(**common, runs=list(runs.values())), indent=1))
        print(f"    done in {wall / 60:.1f} min", flush=True)
        del ctx
        cp.get_default_memory_pool().free_all_blocks()


def _coarsen(da: xr.DataArray, f: int, how="mean"):
    if f == 1:
        return da
    c = da.coarsen(y=f, x=f)
    return c.mean() if how == "mean" else c.max()


def cmd_score(a):
    from glacier_inverse import load_config
    from glacier_inverse.priors import _cropped_inputs

    cfg = load_config(a.domain_path)
    out_root = Path(a.out_root or f"{cfg.output_dir}/sweep_pre_tf")
    mi = Path(cfg.base_dir) / "model_inputs"
    g = _cropped_inputs(cfg, ["rgi_mask", "bed_obs", "elevation", "thickness_obs"])
    bed = xr.where(np.isfinite(g.bed_obs), g.bed_obs, g.elevation - g.thickness_obs.fillna(0))
    with xr.open_dataset(mi / "front_mask_lia.nc") as f:
        lia = f.lia_mask.load().sel(x=g.x, y=g.y, method="nearest").astype(float)
    with xr.open_dataset(mi / "calving_basins.nc") as f:
        cb = f.calving_basin.load().sel(x=g.x, y=g.y, method="nearest")
    reg_of = pd.read_csv(mi / "calving_basins.csv").set_index("basin")["region"].to_dict()
    region_full = np.vectorize(lambda b: reg_of.get(int(b), ""), otypes=[object])(cb.values)

    rows = []
    for d in sorted(p for p in out_root.glob("s*_o*") if p.is_dir()):
        man = json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").exists() else None
        if man is None:
            continue
        for st in sorted(d.glob("state_*.nc")):
            with xr.open_dataset(st) as S:
                H = S.H.load()
                lev = int(S.attrs.get("level", 0))
            fct = 2 ** lev
            marine = _coarsen(bed, fct) < 0
            lia_c = _coarsen(lia, fct) > 0.5
            m15 = _coarsen((g.rgi_mask > 0.5).astype(float), fct) > 0.5
            if fct == 1:
                reg = region_full
                assigned = cb.values >= 0
            else:   # the region of the block's first child; assigned if any child is
                reg = region_full[::fct, ::fct][:H.shape[0], :H.shape[1]]
                assigned = _coarsen((cb >= 0).astype(float), fct, "max").values > 0
            h = H.values
            ice = h > a.h_thr
            dA = (float(abs(g.x[1] - g.x[0])) * fct / 1e3) ** 2
            zone = marine.values & assigned
            retreat = zone & lia_c.values & ~m15.values
            modern = zone & m15.values
            outside = zone & ~lia_c.values
            t = float(st.stem.split("_", 1)[1])
            for r in ["GrIS"] + sorted(set(reg_of.values())):
                rm = np.ones_like(zone) if r == "GrIS" else (reg == r)
                nr = int((retreat & rm).sum())
                rows.append(dict(scale=man["scale"], offset=man["offset"], t=t, region=r,
                                 cover=float((ice & retreat & rm).sum()) / nr if nr else np.nan, n_retreat=nr,
                                 keep=float((ice & modern & rm).sum()) / max(int((modern & rm).sum()), 1),
                                 over_km2=float((ice & outside & rm).sum()) * dA))
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"no finished runs with state files under {out_root}")
        return
    df.to_csv(out_root / "score.csv", index=False)
    sel = df[np.isclose(df.t, a.state_time)]
    for metric in ("cover", "keep", "over_km2"):
        tab = sel.pivot_table(index=["scale", "offset"], columns="region", values=metric)
        print(f"\n{metric} at {a.state_time:g} (h > {a.h_thr:g} m, marine cells assigned to a calving basin)")
        print(tab.to_string(float_format=lambda v: f"{v:.2f}" if metric != "over_km2" else f"{v:.0f}"))
    print(f"\nwritten {out_root / 'score.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--scale", type=float, nargs="+", default=[1.0])
    r.add_argument("--offset", type=float, nargs="+", default=[0.0], help="K")
    r.add_argument("--ramp", type=float, default=0.0, help="years before the record over which the hold ramps to 0")
    r.add_argument("--alpha-h", type=float, default=None)
    r.add_argument("--alpha-q", type=float, default=None)
    r.add_argument("--rho-filename", default=None, help="the rho / h0_base field to use instead of the config's")
    r.add_argument("--no-rho", action="store_true", help="drop the config's rho / h0_base field")
    r.add_argument("--level", type=int, default=0)
    r.add_argument("--t-start", type=float, default=None)
    r.add_argument("--t-end", type=float, default=1950.0)
    r.add_argument("--state-times", type=float, nargs="+", default=list(DEFAULT_STATE_TIMES))
    r.add_argument("--vti", action="store_true", help="also write the VTI frames")
    r.add_argument("--vti-from", type=float, default=None, help="with --vti: frames only from this year on")
    r.add_argument("--keep-pin", action="store_true",
                   help="keep the config's front pin (a pinned reference run on the same level / window)")
    r.add_argument("--calving-h0", type=float, default=None,
                   help="baseline margin h00 (m; default config.calving_h0). A per-basin h0_base field from "
                        "sweep_calving_c.py was fitted with h0 = c alone, so run it with 0")
    r.add_argument("--index-k", type=float, default=None,
                   help="add the index-scaled pre-record TF with this k (K/K; config pre_record_index_k)")
    r.add_argument("--index", default="temperature_anomaly.nc")
    r.add_argument("--index-smooth", type=int, default=11)
    r.add_argument("--index-start", type=float, default=None, help="first year of the index term")
    r.add_argument("--release-year", type=float, default=None,
                   help="keep the config's pin until this year, the free TF-driven law after (pin_release_year)")
    r.add_argument("--out-root", default=None, help="default {config.output_dir}/sweep_pre_tf")
    r.add_argument("--force", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--domain-path", default="domains/greenland")
    s.add_argument("--out-root", default=None)
    s.add_argument("--state-time", type=float, default=1900.0)
    s.add_argument("--h-thr", type=float, default=50.0, help="m of ice for a cell to count")
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_score(a)


if __name__ == "__main__":
    main()
