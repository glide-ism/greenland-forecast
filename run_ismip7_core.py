#!/usr/bin/env python
"""
Run the ISMIP7 core experiments with forward_projection.py, sequentially on
the one GPU, resumably, and optionally export each to the submission layout.

    historical  x {CESM2-WACCM, MRI-ESM2-0}   config t_start -> 2015
    ssp126, ssp585, ctrl                        branched at 2015 -> 2301
    ssp370                                      branched at 2015 -> 2101
    ocx (CARRA2 year by year)                   config t_start -> 2026

= 11 runs. The historical period is identical for every scenario of a GCM
(the catalogue files and the TF agree over 1850-2014 and every scenario's
anomaly reference is the same one, make_ismip7_forcing.py --clim-scenario),
so each GCM's historical runs ONCE -- from the config's spin-up start with
forward_standalone's forcing before 1850 (--pre-record standalone), then the
GCM's historical record from the ssp126 directory -- and every scenario is a
BRANCH: the historical run directory is copied (VTI frames hard-linked, the
files a continuation rewrites or appends to -- the .pvd, scalars.csv,
snapshots.nc, final_state.nc -- copied) and continued with
`forward_projection.py --continue` on the scenario's forcing (the thermal
state and the elevation-feedback reference come along; velocities restart
from zero, one solve's worth of extra V-cycles). The branch records
`branched_from` in its attrs. OCX is its own run on the INVERSION's forcing
(`--record standalone`: the config's yearly reanalysis fields -- the hybrid
CARRA2 temperature + RACMO precipitation -- over 1986-2025, the Vinther index
before; only the ocean TF comes from model_inputs/ismip7/CARRA2_ocx, the EN4
file). The GCM runs use `--mode anomaly` (default): the calibration's own
climatology + biases + the GCM's departure from its 1986-2025 climatology.

Every run uses domains/greenland/config.py AS IT STANDS (results_subdir,
checkpoint, calving field, thermal settings): the first invocation writes
{root}/manifest.json with that provenance and later ones refuse to mix runs
from a different calibration (--force to override).

Resumable: a job whose scalars.csv reaches its t_end (and has final_state.nc)
is skipped; an unfinished job directory is MOVED aside to
<dir>.unfinished-<timestamp> (not deleted) and redone; a branch is redone from
a fresh copy of its historical run.

Usage:
  python run_ismip7_core.py --dry-run                      # the plan and the commands
  python run_ismip7_core.py                                # all 11, in dependency order
  python run_ismip7_core.py --only historical_CESM2-WACCM ssp585_CESM2-WACCM
  python run_ismip7_core.py --export --resolution 1000     # also write the submission files
  python run_ismip7_core.py --export-only                  # export finished runs, no GPU
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from glacier_inverse import load_config  # noqa: E402

PY = sys.executable
DOMAIN = "domains/greenland"
GCMS = ("CESM2-WACCM", "MRI-ESM2-0")
BRANCHES = {"ssp126": 2301.0, "ssp370": 2101.0, "ssp585": 2301.0, "ctrl": 2301.0}
HIST_FORCING = "ssp126"            # the historical record is read from this scenario's directory
HIST_END = 2015.0                  # nominal year 2014 = the step (2014, 2015]; the branch point
OCX_END = 2026.0                   # CARRA2 1986-2025


def plan():
    jobs = []
    for g in GCMS:
        jobs.append(dict(name=f"historical_{g}", gcm=g, scenario=HIST_FORCING, experiment="historical",
                         t_end=HIST_END, parent=None, pre_record="standalone", record="ismip7"))
        for sc, t_end in BRANCHES.items():
            jobs.append(dict(name=f"{sc}_{g}", gcm=g, scenario=sc, experiment=sc, t_end=t_end,
                             parent=f"historical_{g}", pre_record="standalone", record="ismip7"))
    jobs.append(dict(name="ocx_CARRA2", gcm="CARRA2", scenario="ocx", experiment="ocx", t_end=OCX_END,
                     parent=None, pre_record="standalone", record="standalone"))
    return jobs


def last_time(run_dir: Path):
    p = run_dir / "scalars.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    return float(rows[-1]["time"]) if rows else None


def finished(run_dir: Path, t_end: float) -> bool:
    t = last_time(run_dir)
    return t is not None and t >= t_end - 1e-6 and (run_dir / "final_state.nc").exists()


def move_aside(d: Path):
    if d.exists():
        dst = d.with_name(f"{d.name}.unfinished-{datetime.now():%Y%m%d-%H%M%S}")
        d.rename(dst)
        print(f"  moved the unfinished {d.name} aside to {dst.name}")


def branch_copy(src: Path, dst: Path):
    """dst = a continuation-safe copy of the finished run src: VTI frames and the
    static frame hard-linked (written once, never modified), everything a
    continuation rewrites or appends to copied."""
    (dst / "vti").mkdir(parents=True)
    n = 0
    for f in (src / "vti").iterdir():
        if f.suffix == ".pvd":
            shutil.copy2(f, dst / "vti" / f.name)
        else:
            os.link(f, dst / "vti" / f.name)
            n += 1
    for name in ("scalars.csv", "snapshots.nc", "final_state.nc"):
        shutil.copy2(src / name, dst / name)
    print(f"  branched {src.name} -> {dst.name} ({n} frames hard-linked)")


def provenance(cfg, mode):
    import subprocess as sp
    commit = sp.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(sp.run(["git", "-C", str(HERE), "status", "--porcelain", "--", "glacier_inverse", "forward_projection.py",
                         "forward_standalone.py"], capture_output=True, text=True).stdout.strip())
    oc = cfg.ocean_forcing
    return dict(results_subdir=str(cfg.results_subdir), output_dir=str(cfg.output_dir),
                checkpoint=str(Path(cfg.output_dir) / "level_0" / "torch_vars.p"),
                calving=dict(timescale=float(cfg.calving_timescale), H_c=float(cfg.calving_H_c), q=float(cfg.calving_q),
                             h0=float(cfg.calving_h0), rho_filename=str(getattr(oc, "rho_filename", None)),
                             alpha_h=float(oc.alpha_h), alpha_q=float(oc.alpha_q),
                             pin_front=str(getattr(oc, "pin_front", None)),
                             pin_front_filename=str(getattr(oc, "pin_front_filename", None))),
                A_glen=float(cfg.A_glen), thermal=repr(getattr(cfg, "thermal", None)),
                climate_mode=mode, calibration_climate=f"{cfg.gridded_filename} + yearly {cfg.yearly_climate_filename}",
                git_commit=commit + ("+dirty" if dirty else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="run root (default: {config.output_dir}/ismip7_core)")
    ap.add_argument("--only", nargs="*", default=None, help="job names (their parents run too when unfinished)")
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--mode", default="anomaly", choices=("anomaly", "raw"),
                    help="GCM climate mode: anomaly = the calibration's climatology + the GCM's departure (default); "
                         "raw = the GCM fields + the calibrated biases (a sensitivity)")
    ap.add_argument("--no-elevation-feedback", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the commands, run nothing")
    ap.add_argument("--keep-going", action="store_true", help="after a failure, carry on with jobs that do not depend on it")
    ap.add_argument("--force", action="store_true", help="accept a manifest from a different calibration")
    ap.add_argument("--export", action="store_true", help="export each finished run (ismip_exporter.py)")
    ap.add_argument("--export-only", action="store_true", help="export the finished runs, run nothing")
    ap.add_argument("--resolution", type=float, default=1000.0)
    ap.add_argument("--set-counter", default="C001")
    ap.add_argument("--submission-dir", default=None, help="default: {root}/ISMIP7_submission")
    a = ap.parse_args()

    cfg = load_config(HERE / DOMAIN)
    root = Path(a.root or Path(cfg.output_dir) / "ismip7_core")
    prov = provenance(cfg, a.mode)
    man = root / "manifest.json"
    if man.exists():
        old = json.loads(man.read_text())
        keys = ("results_subdir", "checkpoint", "calving", "A_glen", "thermal", "climate_mode", "calibration_climate")
        diff = [k for k in keys if old.get(k) != prov.get(k)]
        if diff and not a.force:
            sys.exit(f"{man} was written for another configuration ({', '.join(diff)} differ):\n"
                     + "\n".join(f"  {k}: {old.get(k)!r}\n   now {prov.get(k)!r}" for k in diff)
                     + "\nuse another --root, or --force")
    if not a.dry_run and not a.export_only:
        root.mkdir(parents=True, exist_ok=True)
        if not man.exists():
            man.write_text(json.dumps(dict(prov, created=datetime.now().isoformat(timespec="seconds")), indent=1))
    print(f"ISMIP7 core runs in {root}\n  calibration {prov['results_subdir']} (git {prov['git_commit']}), "
          f"calving field {prov['calving']['rho_filename']}, alpha_h {prov['calving']['alpha_h']:g}, "
          f"tau {prov['calving']['timescale']:g}\n  GCM climate mode {a.mode} on {prov['calibration_climate']}; "
          f"OCX on the inversion's forcing\n  thermal {prov['thermal']}")
    if prov["calving"]["pin_front"] not in ("None", "") or prov["calving"]["pin_front_filename"] not in ("None", ""):
        print("  NOTE: the config carries a front pin; forward_projection refuses a pin, so the runs will fail")

    jobs = plan()
    by_name = {j["name"]: j for j in jobs}
    want = set(a.only) if a.only else set(by_name)
    unknown = want - set(by_name)
    if unknown:
        sys.exit(f"unknown jobs {sorted(unknown)}; jobs: {list(by_name)}")
    for n in list(want):                       # parents of wanted jobs
        p = by_name[n]["parent"]
        if p:
            want.add(p)
    logs = root / "logs"
    failed = set()
    for j in jobs:
        if j["name"] not in want:
            continue
        d = root / j["name"]
        status = "done" if finished(d, j["t_end"]) else ("partial" if d.exists() else "todo")
        head = (f"[{j['name']}] {j['gcm']} {j['scenario']} -> {j['t_end']:g}"
                + (f", branched from {j['parent']} at {HIST_END:g}" if j["parent"] else f", from t_start") + f": {status}")
        print(head)
        cmd = [PY, "forward_projection.py", "--gcm", j["gcm"], "--scenario", j["scenario"], "--level", str(a.level),
               "--t-end", f"{j['t_end']:g}", "--out-dir", str(d), "--pre-record", j["pre_record"],
               "--record", j["record"], "--mode", a.mode]
        if a.no_elevation_feedback:
            cmd += ["--no-elevation-feedback"]
        if j["parent"]:
            cmd += ["--continue"]
        if status != "done" and not a.export_only:
            if j["parent"] and j["parent"] in failed:
                print(f"  skipped: {j['parent']} failed")
                failed.add(j["name"])
                continue
            if a.dry_run:
                print("  " + " ".join(cmd))
            else:
                if j["parent"]:
                    parent = root / j["parent"]
                    if not finished(parent, by_name[j["parent"]]["t_end"]):
                        print(f"  skipped: {j['parent']} is not finished")
                        failed.add(j["name"])
                        continue
                    move_aside(d)
                    branch_copy(parent, d)
                else:
                    move_aside(d)
                logs.mkdir(parents=True, exist_ok=True)
                log = logs / f"{j['name']}.log"
                print(f"  running (log {log}) ...", flush=True)
                tic = time.time()
                with open(log, "a") as f:
                    f.write(f"\n### {datetime.now().isoformat(timespec='seconds')}  {' '.join(cmd)}\n")
                    f.flush()
                    rc = subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT).returncode
                ok = rc == 0 and finished(d, j["t_end"])
                print(f"  {'finished' if ok else f'FAILED (exit {rc})'} in {(time.time() - tic) / 60:.1f} min", flush=True)
                if not ok:
                    failed.add(j["name"])
                    if not a.keep_going:
                        sys.exit(f"stopping: see {log} (rerun to resume; --keep-going to carry on past failures)")
                    continue
        if (a.export or a.export_only) and finished(d, j["t_end"]):
            ecmd = [PY, "ismip_exporter.py", "--run-dir", str(d), "--experiment", j["experiment"],
                    "--esm", j["gcm"], "--resolution", f"{a.resolution:g}", "--set-counter", a.set_counter,
                    "--submission-dir", str(Path(a.submission_dir) if a.submission_dir else root / "ISMIP7_submission")]
            if j["experiment"] == "ocx":       # not in the exporter's / checker's experiment table
                ecmd += ["--years", "1986", "2025"]
            if a.dry_run:
                print("  " + " ".join(ecmd))
                continue
            logs.mkdir(parents=True, exist_ok=True)
            elog = logs / f"{j['name']}.export.log"
            with open(elog, "w") as f:
                rc = subprocess.run(ecmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT).returncode
            print(f"  export {'done' if rc == 0 else f'FAILED (exit {rc}), see {elog}'}")
    if failed:
        print(f"\nunfinished: {sorted(failed)}")


if __name__ == "__main__":
    main()
