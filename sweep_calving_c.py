#!/usr/bin/env python
"""
Sweep the calving-margin baseline c: one forward_standalone run per value,
with the margin

    h0(x, t) = c + alpha_h dTF(x, t)          (clim_h = 0, no field, alpha fixed)

from the config's spin-up start, keeping ONLY the observational period on
disk (VTI frames from --t-save on, a subset of fields) plus a restart state
at --t-save and the final state. The runs are the material for a
per-glacier fit: the approximation that glaciers calve independently lets
each basin take the c that fit it best across the sweep (h0_i = c_i +
alpha dTF_i), assembled afterwards into a per-basin field
(preprocessing/make_calving_basins.py gives the basins). Nothing before
the measurements is kept: we are not allowed to care what happened then.

Layout: {out_root}/c{+ddd}/ with vti/ (frames >= t_save), state_{t}.nc at
the restart point and the observation epochs (raw fields for the
evaluation), forward_soln.nc, manifest.json; {out_root}/sweep.json lists the runs. A run
whose forward_soln.nc exists is skipped (resumable) unless --force.

Usage:
  python sweep_calving_c.py --c -150 -100 -50 -25 0 25 50 100          # ~10 min each at 1 km
  python sweep_calving_c.py --c -50 0 --level 2 --t-start 1900 --out-root /tmp/x   # smoke
"""
import argparse
import dataclasses
import json
import time
from datetime import datetime
from pathlib import Path

import cupy as cp

import forward_standalone as fs

DEFAULT_FIELDS = ("H", "U_s", "mask", "phi", "h0", "smb", "dhdt")
# restart point + the observation epochs: MEaSUREs dh/dt 1993-2019, BedMachine
# surface 2008, extent 2015, ITS_LIVE velocity 2018 (the products' time_nominal)
DEFAULT_STATE_TIMES = (1990.0, 1993.0, 2008.0, 2015.0, 2018.0, 2019.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c", type=float, nargs="+", required=True, help="baseline margins c (m), one run each")
    ap.add_argument("--alpha", type=float, default=None, help="alpha_h (m/K), fixed for the sweep (default: config)")
    ap.add_argument("--alpha-q", type=float, default=None, help="alpha_q (1/K), fixed (default: config)")
    ap.add_argument("--timescale", type=float, default=None,
                    help="calving removal timescale tau (yr; default: config). The sink removes 1 - exp(-dt/tau) of a cell per "
                         "step with F < 0: tau separates the response to a one-year TF spike from that to a sustained exceedance")
    ap.add_argument("--H-c", type=float, default=None, help="thinnest surviving floating ice (m; default: config)")
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--t-start", type=float, default=None, help="spin-up start (default: config)")
    ap.add_argument("--t-end", type=float, default=None, help="(default: config)")
    ap.add_argument("--t-save", type=float, default=1990.0, help="keep frames from this year on; restart state saved here")
    ap.add_argument("--fields", nargs="+", default=list(DEFAULT_FIELDS), help="VTI fields to keep")
    ap.add_argument("--state-times", type=float, nargs="+", default=list(DEFAULT_STATE_TIMES),
                    help="step ends at which the RAW state is saved (the observation epochs the evaluation needs, + the restart point)")
    ap.add_argument("--out-root", default=None, help="default {config.output_dir}/sweep_c")
    ap.add_argument("--force", action="store_true", help="rerun runs whose forward_soln.nc exists")
    a = ap.parse_args()

    out_root = Path(a.out_root or f"{fs.config.output_dir}/sweep_c")
    out_root.mkdir(parents=True, exist_ok=True)
    # the calibrated fields, re-exported if the checkpoint is newer (as main() does)
    stale = (fs.PHYSICAL_PATH.exists() and Path(fs.CHECKPOINT).exists()
             and Path(fs.CHECKPOINT).stat().st_mtime > fs.PHYSICAL_PATH.stat().st_mtime)
    if stale or not fs.PHYSICAL_PATH.exists():
        fs.export_physical_fields(fs.CHECKPOINT, fs.PHYSICAL_PATH)

    if a.timescale is not None or a.H_c is not None:
        fs.config = dataclasses.replace(fs.config,
                                        calving_timescale=fs.config.calving_timescale if a.timescale is None else float(a.timescale),
                                        calving_H_c=fs.config.calving_H_c if a.H_c is None else float(a.H_c))
    ocean = fs.config.ocean_forcing
    fs.OCEAN = dataclasses.replace(ocean, enabled=True, clim_h=0.0, clim_q=0.0, rho_filename=None, pin_front=None,
                                   pin_front_filename=None, pin_release_year=None,
                                   alpha_h=ocean.alpha_h if a.alpha is None else float(a.alpha),
                                   alpha_q=ocean.alpha_q if a.alpha_q is None else float(a.alpha_q))
    fs.VTI_T_MIN = float(a.t_save)
    fs.VTI_FIELDS = tuple(a.fields)
    fs.STATE_SAVE_TIMES = tuple(sorted(set([float(a.t_save)] + [float(t) for t in a.state_times])))
    if a.t_start is not None:
        fs.T_START = float(a.t_start)
    if a.t_end is not None:
        fs.T_END = float(a.t_end)
    common = dict(alpha_h=fs.OCEAN.alpha_h, alpha_q=fs.OCEAN.alpha_q, calving_q=fs.Q0, level=a.level,
                  calving_timescale=fs.config.calving_timescale, calving_H_c=fs.config.calving_H_c,
                  t_start=fs.T_START, t_end=fs.T_END, dt=fs.DT, dt_schedule=[list(map(float, s)) for s in fs.DT_SCHEDULE],
                  t_save=a.t_save, fields=list(a.fields), state_times=list(fs.STATE_SAVE_TIMES), checkpoint=str(fs.CHECKPOINT),
                  physical_fields=str(fs.PHYSICAL_PATH), config=fs.config.results_subdir,
                  gridded=fs.config.gridded_filename, thermal_forcing=str(fs.THERMAL_PATH))
    print(f"sweep over c = {a.c} with h0 = c + {fs.OCEAN.alpha_h:g} m/K dTF"
          + (f", q = {fs.Q0:g} + {fs.OCEAN.alpha_q:g}/K dTF" if fs.OCEAN.alpha_q else "")
          + f"; tau {fs.config.calving_timescale:g} yr, H_c {fs.config.calving_H_c:g} m"
          + f"; level {a.level}, {fs.T_START:g}-{fs.T_END:g}, frames from {a.t_save:g}: {list(a.fields)}; into {out_root}", flush=True)

    # merge with the existing manifest: a partial rerun (--c -25 --force) must
    # not drop the other runs from sweep.json (it did, 2026-09-25)
    prev = {}
    if (out_root / "sweep.json").exists():
        try:
            for r in json.loads((out_root / "sweep.json").read_text()).get("runs", []):
                prev[float(r["c"])] = r
        except Exception as e:  # noqa: BLE001
            print(f"could not read the existing sweep.json ({e}); starting a new one")
    runs = []

    def write_manifest():
        merged = dict(prev)
        for r in runs:
            merged[float(r["c"])] = r
        allruns = [merged[c] for c in sorted(merged)]
        (out_root / "sweep.json").write_text(json.dumps(dict(**common, runs=allruns), indent=1))
    for c in a.c:
        out = out_root / f"c{c:+05.0f}"
        done = (out / "forward_soln.nc").exists()
        if done and not a.force:
            print(f"--- c = {c:+g}: {out} exists, skipped", flush=True)
            runs.append(dict(c=c, dir=str(out), skipped=True))
            continue
        fs.H00 = float(c)
        t0 = time.time()
        print(f"--- c = {c:+g} m -> {out}", flush=True)
        ctx = fs.setup(level=a.level, out_dir=out)
        fs.run(ctx)
        n_frames = len([p for p in (out / "vti").glob("*.vti") if "_static" not in p.name])
        wall = time.time() - t0
        man = dict(c=c, **common, wall_s=wall, n_frames=n_frames, finished=datetime.now().isoformat(timespec="seconds"),
                   ocean=ctx.ocean.describe() if ctx.ocean is not None else "none")
        (out / "manifest.json").write_text(json.dumps(man, indent=1))
        runs.append(dict(c=c, dir=str(out), wall_s=wall, n_frames=n_frames))
        print(f"    done in {wall / 60:.1f} min, {n_frames} frames", flush=True)
        del ctx
        cp.get_default_memory_pool().free_all_blocks()
        write_manifest()
    write_manifest()
    print(f"sweep manifest: {out_root / 'sweep.json'}")


if __name__ == "__main__":
    main()
