#!/usr/bin/env python
"""
Re-label an ISMIP7 submission tree with one SET COUNTER PER EXPERIMENT.

ismip_exporter.py wrote every experiment under one counter (C001); the
protocol's table assigns a counter per (experiment, ESM):

    C001 historical CESM2-WACCM    C002 historical MRI-ESM2-0
    C003 ssp370     CESM2-WACCM    C004 ssp370     MRI-ESM2-0
    C005 ssp126     CESM2-WACCM    C006 ssp126     MRI-ESM2-0
    C007 ssp585     CESM2-WACCM    C008 ssp585     MRI-ESM2-0
    C009 ctrl       CESM2-WACCM    C010 ctrl       MRI-ESM2-0
    C011 ocx        (any ESM field; ours is CARRA2)

For every .nc under <CORE>/*/ the script parses the file name
<var>_<domain>_<group>_<model>_<member>_<esm>_<forcing>_<experiment>_<set>_<period>.nc,
replaces the set field, MOVES the file to <CORE>/<new set>/ (os.rename: same
filesystem, no data copied), and rewrites the global attribute `set` in place
(netCDF4 append mode; the data are untouched). not_modelled.txt is copied
into every new directory. Dry run by default; --apply executes and writes
<CORE>/set_counter_moves.csv (old path, new path) for an undo.

  python tools/ismip7_set_counters.py --core .../Models/GrIS/UMT/GLIDE/CORE            # dry run
  python tools/ismip7_set_counters.py --core .../Models/GrIS/UMT/GLIDE/CORE --apply
  python tools/ismip7_set_counters.py --core ... --undo                                 # reverse an --apply
"""
import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import netCDF4

COUNTERS = {
    ("historical", "CESM2-WACCM"): "C001", ("historical", "MRI-ESM2-0"): "C002",
    ("ssp370", "CESM2-WACCM"): "C003", ("ssp370", "MRI-ESM2-0"): "C004",
    ("ssp126", "CESM2-WACCM"): "C005", ("ssp126", "MRI-ESM2-0"): "C006",
    ("ssp585", "CESM2-WACCM"): "C007", ("ssp585", "MRI-ESM2-0"): "C008",
    ("ctrl", "CESM2-WACCM"): "C009", ("ctrl", "MRI-ESM2-0"): "C010",
}
OCX_COUNTER = "C011"


def counter_for(exp, esm):
    if exp == "ocx":
        return OCX_COUNTER
    return COUNTERS.get((exp, esm))


def set_attr(path, new):
    with netCDF4.Dataset(path, "a") as d:
        d.setncattr("set", new)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core", required=True, help="the .../Models/GrIS/<group>/<model>/CORE directory")
    ap.add_argument("--apply", action="store_true", help="execute (default: dry run)")
    ap.add_argument("--undo", action="store_true", help="reverse the moves recorded in set_counter_moves.csv")
    a = ap.parse_args()
    core = Path(a.core)
    log = core / "set_counter_moves.csv"

    if a.undo:
        rows = list(csv.DictReader(open(log)))
        for r in rows:
            src, dst = Path(r["new"]), Path(r["old"])
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.rename(src, dst)
            set_attr(dst, r["old_set"])
        log.rename(log.with_suffix(".undone.csv"))
        print(f"undid {len(rows)} moves")
        return

    moves, problems = [], []
    for f in sorted(core.glob("*/*.nc")):
        parts = f.stem.split("_")
        if len(parts) != 10:
            problems.append(f"{f.name}: {len(parts)} name fields, expected 10")
            continue
        esm, exp, old = parts[5], parts[7], parts[8]
        new = counter_for(exp, esm)
        if new is None:
            problems.append(f"{f.name}: no counter for experiment {exp!r} / ESM {esm!r}")
            continue
        parts[8] = new
        dst = core / new / ("_".join(parts) + ".nc")
        if dst == f:
            with netCDF4.Dataset(f) as d:
                if d.getncattr("set") == new:
                    continue                          # already right (name, place and attribute)
        if dst.exists() and dst != f:
            problems.append(f"{f.name}: target {dst} exists")
            continue
        moves.append((f, dst, old, new))
    if problems:
        print("PROBLEMS (nothing done):\n  " + "\n  ".join(problems))
        sys.exit(1)

    by_set = {}
    for _, dst, _, new in moves:
        by_set[new] = by_set.get(new, 0) + 1
    print(f"{len(moves)} files to re-label: " + ", ".join(f"{k} {v}" for k, v in sorted(by_set.items())))
    for src, dst, old, new in moves[:3] + moves[-2:]:
        print(f"  {src.relative_to(core)}\n    -> {dst.relative_to(core)}  (set {old} -> {new})")
    if not a.apply:
        print("dry run; pass --apply")
        return

    notmod = next((p for p in core.glob("*/not_modelled.txt")), None)
    with open(log, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["old", "new", "old_set", "new_set"])
        w.writeheader()
        for src, dst, old, new in moves:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst != src:
                os.rename(src, dst)
            set_attr(dst, new)
            w.writerow(dict(old=str(src), new=str(dst), old_set=old, new_set=new))
            fh.flush()
    for s in by_set:
        if notmod is not None and not (core / s / "not_modelled.txt").exists():
            shutil.copy2(notmod, core / s / "not_modelled.txt")
    print(f"done: {len(moves)} files moved / re-labelled; undo log {log}")
    print("directories:", ", ".join(f"{d.name} ({len(list(d.glob('*.nc')))} files)" for d in sorted(core.iterdir()) if d.is_dir()))


if __name__ == "__main__":
    main()
