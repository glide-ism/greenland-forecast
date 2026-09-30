#!/usr/bin/env python
"""
Iterative per-basin calving-margin calibration against the stage-1 reference:
a bracketed bisection on c_i, one FREE-CALVING composite run per iteration,
ONE STEP PER INVOCATION (the GPU run in between is yours to launch).

Why this works: over the uniform c sweep the flux-weighted gate THICKNESS
ratio against stage 1 is strictly monotone (decreasing) in c at every gated
basin (101 of 101 with a stage-1 gate flux > 0.5 Gt/yr on v1.1thermal,
Spearman <= -0.9, no upticks), so the sign of

    s_i = mean over the gate epochs of log( sum_g w H_model / sum_g w H_ref ),
          w = L |u_ref| H_ref (the reference's flux density on the gate chain)

says which way c_i has to move: s > 0 (too thick) -> raise c, s < 0 -> lower
it. Gate SPEED and FLUX are not monotone (a thinning trunk speeds up, then
collapses), so they are not the steering signal; the factorized gate misfit
J_f (speed and thickness logs) only breaks ties at a JUMP, where the
thickness ratio skips over 1 between two c values (the bistable fronts,
Helheim, Silarleq, ...: 18 of 101) and bisection converges onto the jump.
Every basin moves at once in the composite, so each is judged with its
neighbours at THEIR current values -- the state that is deployed -- which is
what the uniform sweep cannot do (shared fjords: 79N / Zachariae).

Bookkeeping per tracked basin (gated, stage-1 gate flux >= --min-ref-flux):
a bracket [lo, hi] with s(lo) > 0 > s(hi), from the uniform sweep at init
and from the composite evaluations afterwards; each step evaluates the
composite at the current c, moves the matching end, and sets c to the
midpoint. A basin is
  converged  |s| <= --tol-s (thickness within exp(tol) of stage 1), or the
             bracket is narrower than --tol-c and the composite has seen
             both signs (then the evaluated c with the lower J_f of the two
             nearest the jump is kept: status 'jump');
  expanded   when the bracket is narrower than --tol-c but the COMPOSITE has
             seen only one sign (the neighbours moved the root outside the
             sweep's bracket): the bracket is extended by --expand m that way;
  edge       no crossing in the sweep (fixed at the base field's value);
  reopened   a converged (not jump) basin whose |s| later exceeds
             --reopen x tol-s gets a fresh --expand bracket on the right side.
Every other basin keeps the base field's value exactly (the new fields are
the base field with the tracked basins' cells overwritten).

Layout ({iter_root}): state.json (all of the above plus every evaluation),
iter_000/h0_base.nc, iter_000/run/ (the forward run), iter_001/h0_base.nc, ...
and history.csv (one row per basin per evaluation).

Usage:
  python calving_iter.py init --sweep-root R/sweep_c --reference-run R/stage1_reference \\
         --base-field calving_h0_base_v1.1thermal_s1m.nc --iter-root R/calving_iter
  # run the printed forward_standalone.py --free-h0 ... command, then
  python calving_iter.py step --iter-root R/calving_iter      # evaluates the last run, writes the next field
  python calving_iter.py status --iter-root R/calving_iter
  python calving_iter.py finalize --iter-root R/calving_iter --out calving_h0_base_v1.1thermal_iter.nc
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "analysis"))
sys.path.insert(0, str(HERE))
from basin_mass_balance import crop_to_factor  # noqa: E402
from gates import ChainGates  # noqa: E402
from glacier_inverse import load_config  # noqa: E402

PY = "~/Source/glide_test_env/bin/python"
KEY = ("JAKOBSHAVN_ISBRAE", "HELHEIMGLETSCHER", "KANGERLUSSUAQ", "RINK_ISBRAE", "NIOGHALVFJERDSFJORDEN",
       "ZACHARIAE_ISSTROM", "PETERMANN_GLETSCHER", "SERMEQ_KUJALLEQ", "SERMEQ_SILARLEQ", "SERMEQ_AVANNARLEQ2",
       "UPERNAVIK_ISSTROM_N", "KAKIVFAAT_SERMIAT", "ALISON_GLETSCHER", "STORE_GLETSCHER", "KOGE_BUGT_C")


# ------------------------------------------------------------------ gate signal
class Gauge:
    """Per-basin gate signals of a run against the reference run."""

    def __init__(self, domain, cfg, gates, reference, epochs, sig_s, sig_h):
        self.domain, self.epochs, self.sig_s, self.sig_h = Path(domain), [float(t) for t in epochs], sig_s, sig_h
        gi = crop_to_factor(xr.open_dataset(self.domain / "model_inputs" / "GLIDE_inputs.nc"), 2 ** cfg.n_levels)
        self.ny0 = gi.sizes["y"]
        self.G = ChainGates(gates, gi.x.values, gi.y.values, gi.vx.values, gi.vy.values)
        binfo = pd.read_csv(self.domain / "model_inputs" / "calving_basins.csv").set_index("basin")
        self.gate_ids = {int(b): [int(float(g)) for g in str(v).split()] for b, v in binfo.gates.items()
                         if not pd.isna(v) and str(v).strip()}
        self.names = binfo.name.to_dict()
        self.regions = binfo.region.to_dict()
        self.ref = {t: self.fields(Path(reference), t) for t in self.epochs}
        self.ref_D = {t: self._per_basin(self.G.flux(*self.ref[t]).set_index("gate"), "D") for t in self.epochs}

    def fields(self, run, t):
        S = xr.open_dataset(Path(run) / f"state_{t:g}.nc")
        u, v, H = S.u_s.values, S.v_s.values, S.H.values
        S.close()
        U, V = 0.5 * (u[:, :-1] + u[:, 1:]), 0.5 * (v[:-1] + v[1:])
        f = self.ny0 // H.shape[0]
        if f > 1:                                       # a coarse-level run: repeat onto the fine grid
            U, V, H = (np.kron(q, np.ones((f, f))) for q in (U, V, H))
        return U, V, H

    def _per_basin(self, df, col):
        return {b: float(df[col].reindex(ids).fillna(0.0).sum()) for b, ids in self.gate_ids.items()}

    def signals(self, run):
        """{basin: dict(s, rs, rh (epoch means of the logs), J_f, D (last epoch))}."""
        acc = {b: dict(ls=[], lh=[], D=np.nan) for b in self.gate_ids}
        for t in self.epochs:
            U, V, H = self.fields(run, t)
            fa = self.G.factors(U, V, H, *self.ref[t])
            sm, so, hm, ho = (self._per_basin(fa, k) for k in ("s_m", "s_o", "h_m", "h_o"))
            D = self._per_basin(self.G.flux(U, V, H).set_index("gate"), "D")
            for b in self.gate_ids:
                if so[b] > 0 and ho[b] > 0:
                    acc[b]["ls"].append(np.log(max(sm[b] / so[b], 1e-3)))
                    acc[b]["lh"].append(np.log(max(hm[b] / ho[b], 1e-3)))
                acc[b]["D"] = D[b]
        out = {}
        for b, a in acc.items():
            if not a["lh"]:
                continue
            ls, lh = np.array(a["ls"]), np.array(a["lh"])
            out[b] = dict(s=float(lh.mean()), rs=float(np.exp(ls.mean())), rh=float(np.exp(lh.mean())),
                          J_f=float(0.5 * np.mean((ls / self.sig_s) ** 2 + (lh / self.sig_h) ** 2)), D=float(a["D"]))
        return out


# ------------------------------------------------------------------ fields
def base_c(field_path, cfg):
    """{basin: c} of an h0_base field (h0_base = c - calving_h0) and the dataset."""
    ds = xr.open_dataset(field_path).load()
    h00 = float(ds.attrs.get("calving_h0_subtracted", cfg.calving_h0))
    return ds, h00


def write_field(base_ds, h00, cvals, out, meta):
    """The base field with the tracked basins' cells set to c - calving_h0."""
    h0 = base_ds.h0_base.values.copy()
    cb = base_ds.calving_basin.values
    for b, c in cvals.items():
        h0[cb == int(b)] = c - h00
    ds = base_ds.copy()
    ds["h0_base"] = (("y", "x"), h0.astype(np.float32), base_ds.h0_base.attrs)
    ds.attrs = dict(base_ds.attrs)
    ds.attrs.update(source=f"calving_iter.py {date.today().isoformat()}", **{k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                                                                            for k, v in meta.items()})
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out, encoding={"h0_base": dict(zlib=True, complevel=4), "calving_basin": dict(zlib=True, complevel=4)})


def run_command(field, run_dir):
    return f"{PY} forward_standalone.py --free-h0 {field} --out-dir {run_dir}"


def done(run_dir, epochs):
    return (run_dir / "forward_soln.nc").exists() and all((run_dir / f"state_{t:g}.nc").exists() for t in epochs)


# ------------------------------------------------------------------ commands
def cmd_init(a):
    cfg = load_config(a.domain_path)
    root = Path(a.iter_root)
    if (root / "state.json").exists() and not a.force:
        sys.exit(f"{root}/state.json exists (use step, or --force to restart)")
    bf = Path(a.base_field)
    bf = bf if bf.exists() else Path(a.domain_path) / "model_inputs" / a.base_field
    base_ds, h00 = base_c(bf, cfg)
    ga = Gauge(a.domain_path, cfg, a.gates, a.reference_run, a.epochs, a.speed_sigma, a.thick_sigma)
    cb = base_ds.calving_basin.values
    # the sweep's signal curves -> the brackets
    sroot = Path(a.sweep_root)
    runs = sorted((d for d in sroot.glob("c[+-]*") if done(d, ga.epochs)), key=lambda d: float(d.name[1:]))
    print(f"sweep {sroot}: {len(runs)} runs with states at {ga.epochs}", flush=True)
    curves = {}
    for d in runs:
        c = float(d.name[1:])
        for b, v in ga.signals(d).items():
            curves.setdefault(b, []).append((c, v["s"]))
        print(f"  c {c:+7.1f} read", flush=True)
    basins, n_edge = {}, 0
    for b, cur in curves.items():
        Dref = ga.ref_D[ga.epochs[-1]].get(b, 0.0)
        if Dref < a.min_ref_flux:
            continue
        cur = sorted(cur)
        cs, ss = np.array([c for c, _ in cur]), np.array([s for _, s in cur])
        cells = cb == b
        c0 = float(np.round(np.median(base_ds.h0_base.values[cells]) + h00, 3)) if cells.any() else np.nan
        pos, neg = np.where(ss > 0)[0], np.where(ss < 0)[0]
        rec = dict(name=ga.names.get(b, f"basin{b}"), region=ga.regions.get(b, ""), D_ref=Dref, c_base=c0,
                   sweep=[[float(c), float(s)] for c, s in cur], evals=[], seen_pos=False, seen_neg=False)
        # first crossing from thick (low c) to thin (high c)
        cross = [i for i in range(len(ss) - 1) if ss[i] > 0 >= ss[i + 1]]
        if not len(pos) or not len(neg) or not cross:
            rec.update(status="edge", lo=None, hi=None, c=c0)
            n_edge += 1
        else:
            i = cross[0]
            rec.update(status="active", lo=float(cs[i]), hi=float(cs[i + 1]), s_lo=float(ss[i]), s_hi=float(ss[i + 1]),
                       c=0.5 * (cs[i] + cs[i + 1]))
        basins[str(b)] = rec
    st = dict(domain=str(a.domain_path), reference_run=str(Path(a.reference_run).resolve()), gates=a.gates,
              base_field=str(bf.resolve()), calving_h0_subtracted=h00, epochs=ga.epochs, min_ref_flux=a.min_ref_flux,
              tol_s=a.tol_s, tol_c=a.tol_c, expand=a.expand, reopen=a.reopen, speed_sigma=a.speed_sigma,
              thick_sigma=a.thick_sigma, iteration=0, basins=basins)
    field = root / "iter_000" / "h0_base.nc"
    write_field(base_ds, h00, {b: r["c"] for b, r in basins.items() if r["status"] == "active"}, field,
                dict(iteration=0, iter_root=str(root.resolve()), base_field=str(bf.resolve())))
    (root / "state.json").write_text(json.dumps(st, indent=1))
    na = sum(r["status"] == "active" for r in basins.values())
    print(f"\n{len(basins)} tracked basins (stage-1 gate flux >= {a.min_ref_flux:g} Gt/yr at {ga.epochs[-1]:g}): "
          f"{na} bracketed, {n_edge} with no crossing in the sweep (held at the base field)")
    show(basins)
    print(f"\nwrote {field}\nnext:\n  {run_command(field, root / 'iter_000' / 'run')}\n  then: {PY} calving_iter.py step --iter-root {root}")


def cmd_step(a):
    root = Path(a.iter_root)
    st = json.loads((root / "state.json").read_text())
    k = st["iteration"]
    run_dir = root / f"iter_{k:03d}" / "run"
    if not done(run_dir, st["epochs"]):
        sys.exit(f"iteration {k}: the run {run_dir} is not finished (forward_soln.nc + state files at {st['epochs']})")
    cfg = load_config(st["domain"])
    at = xr.open_dataset(run_dir / "forward_soln.nc").attrs
    if abs(float(at.get("calving_h0", cfg.calving_h0)) - st["calving_h0_subtracted"]) > 1e-6:
        print(f"WARNING: the run's calving_h0 {at.get('calving_h0')} differs from the {st['calving_h0_subtracted']} the fields "
              f"subtract: every margin is shifted by the difference")
    ga = Gauge(st["domain"], cfg, st["gates"], st["reference_run"], st["epochs"], st["speed_sigma"], st["thick_sigma"])
    sig = ga.signals(run_dir)
    tol_s, tol_c, expand = st["tol_s"], st["tol_c"], st["expand"]
    moved = dict(converged=0, jump=0, expanded=0, reopened=0, active=0)
    for bs, r in st["basins"].items():
        b = int(bs)
        if b not in sig:
            continue
        v = sig[b]
        r["evals"].append(dict(it=k, c=r["c"], **v))
        s = v["s"]
        if r["status"] == "edge":
            continue
        if r["status"] in ("converged", "jump"):
            if r["status"] == "converged" and abs(s) > st["reopen"] * tol_s:
                r.update(status="active", lo=r["c"] if s > 0 else r["c"] - expand, hi=r["c"] + expand if s > 0 else r["c"])
                r["seen_pos"], r["seen_neg"] = s > 0, s < 0
                r["reopened_at"] = k
                r["c"] = 0.5 * (r["lo"] + r["hi"])
                moved["reopened"] += 1
            continue
        r["seen_pos"] |= s > 0
        r["seen_neg"] |= s < 0
        if abs(s) <= tol_s:
            r["status"] = "converged"
            moved["converged"] += 1
            continue
        if s > 0:
            r["lo"], r["s_lo"] = r["c"], s
        else:
            r["hi"], r["s_hi"] = r["c"], s
        if r["hi"] - r["lo"] <= tol_c + 1e-9:
            if r["seen_pos"] and r["seen_neg"]:
                # a jump: keep the better of the last composite evaluations on each side
                ev = [e for e in r["evals"] if e["it"] >= r.get("reopened_at", 0)]
                sides = [max((e for e in ev if e["s"] > 0), key=lambda e: e["c"]),
                         min((e for e in ev if e["s"] < 0), key=lambda e: e["c"])]
                best = min(sides, key=lambda e: e["J_f"])
                r.update(status="jump", c=best["c"])
                moved["jump"] += 1
                continue
            # the composite never crossed: the neighbours moved the root out of the sweep's bracket
            if s > 0:
                r["lo"], r["hi"] = r["c"], r["c"] + expand
            else:
                r["lo"], r["hi"] = r["c"] - expand, r["c"]
            moved["expanded"] += 1
        r["c"] = 0.5 * (r["lo"] + r["hi"])
        moved["active"] += 1
    st["iteration"] = k + 1
    # totals of this run against the reference
    Dm = sum(v["D"] for v in sig.values())
    Dr = sum(ga.ref_D[ga.epochs[-1]].get(b, 0.0) for b in sig)
    print(f"iteration {k}: evaluated {run_dir}; gate flux at {ga.epochs[-1]:g} {Dm:.0f} vs stage 1 {Dr:.0f} Gt/yr "
          f"({Dm / Dr:.3f}); step: " + ", ".join(f"{n} {v}" for n, v in moved.items()))
    show(st["basins"])
    base_ds, h00 = base_c(st["base_field"], cfg)
    field = root / f"iter_{k + 1:03d}" / "h0_base.nc"
    cvals = {b: r["c"] for b, r in st["basins"].items() if r["status"] != "edge"}
    write_field(base_ds, h00, cvals, field, dict(iteration=k + 1, iter_root=str(root.resolve()), base_field=st["base_field"]))
    (root / "state.json").write_text(json.dumps(st, indent=1))
    write_history(root, st)
    n_open = sum(r["status"] == "active" for r in st["basins"].values())
    print(f"\nwrote {field}; {n_open} basins still active")
    if n_open:
        print(f"next:\n  {run_command(field, root / f'iter_{k + 1:03d}' / 'run')}\n  then: {PY} calving_iter.py step --iter-root {root}")
    else:
        print(f"all tracked basins settled: {PY} calving_iter.py finalize --iter-root {root} --out <name>.nc "
              f"(or run {field} once more to see the settled composite)")


def write_history(root, st):
    rows = [dict(basin=int(b), name=r["name"], region=r["region"], status=r["status"], **e)
            for b, r in st["basins"].items() for e in r["evals"]]
    if rows:
        pd.DataFrame(rows).to_csv(root / "history.csv", index=False)


def show(basins, n=None):
    df = pd.DataFrame([dict(basin=int(b), **{k: r.get(k) for k in ("name", "region", "D_ref", "status", "c", "lo", "hi")},
                            s=(r["evals"][-1]["s"] if r["evals"] else np.nan),
                            rs=(r["evals"][-1]["rs"] if r["evals"] else np.nan),
                            rh=(r["evals"][-1]["rh"] if r["evals"] else np.nan)) for b, r in basins.items()])
    print("status: " + ", ".join(f"{k} {v}" for k, v in df.status.value_counts().items()))
    key = df[df.name.isin(KEY)].sort_values("D_ref", ascending=False)
    rest = df[~df.name.isin(KEY)].sort_values("D_ref", ascending=False).head(n or 0)
    with pd.option_context("display.width", 200):
        print(pd.concat([key, rest])[["name", "region", "D_ref", "status", "c", "lo", "hi", "s", "rs", "rh"]]
              .round(dict(D_ref=1, c=2, lo=2, hi=2, s=3, rs=2, rh=2)).to_string(index=False))


def cmd_status(a):
    st = json.loads((Path(a.iter_root) / "state.json").read_text())
    print(f"iteration {st['iteration']} (last evaluated run: iter_{st['iteration'] - 1:03d})")
    show(st["basins"], n=a.n)


def cmd_finalize(a):
    root = Path(a.iter_root)
    st = json.loads((root / "state.json").read_text())
    cfg = load_config(st["domain"])
    base_ds, h00 = base_c(st["base_field"], cfg)
    # the last EVALUATED c of every basin that was evaluated (not an unrun midpoint), else the base
    cvals = {}
    for b, r in st["basins"].items():
        if r["status"] == "edge":
            continue
        if r["status"] == "active" and r["evals"]:
            e = min(r["evals"], key=lambda e: abs(e["s"]))   # the best thickness match seen
            cvals[b] = e["c"]
        else:
            cvals[b] = r["c"]
    out = Path(a.domain_path) / "model_inputs" / a.out
    write_field(base_ds, h00, cvals, out, dict(iteration=st["iteration"], iter_root=str(root.resolve()),
                                               base_field=st["base_field"], finalized=1))
    pd.DataFrame([dict(basin=int(b), name=r["name"], region=r["region"], status=r["status"], c=cvals.get(b, r["c"]))
                  for b, r in st["basins"].items()]).to_csv(root / f"calving_c_{Path(a.out).stem}.csv", index=False)
    print(f"wrote {out} ({len(cvals)} basins from the iteration, the rest from {st['base_field']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--iter-root", required=True)
    p = sub.add_parser("init", parents=[common])
    p.add_argument("--domain-path", default=str(HERE / "domains" / "greenland"))
    p.add_argument("--sweep-root", required=True, help="the uniform c sweep (sweep_calving_c.py) that gives the brackets")
    p.add_argument("--reference-run", required=True, help="the stage-1 reference (forward_standalone.py --stage1)")
    p.add_argument("--base-field", required=True, help="h0_base file (model_inputs name or path); untracked basins keep it")
    p.add_argument("--gates", default=str(HERE / "common_data/dhdt/mankoff/dataverse_files/gates.gpkg"))
    p.add_argument("--epochs", type=float, nargs="+", default=[1990.0, 2008.0, 2015.0, 2018.0])
    p.add_argument("--min-ref-flux", type=float, default=0.5, help="Gt/yr: track basins whose stage-1 gate flux is at least this")
    p.add_argument("--tol-s", type=float, default=0.03, help="converged when |mean log thickness ratio| <= this")
    p.add_argument("--tol-c", type=float, default=1.25, help="m: a bracket this narrow is a jump (or needs expanding)")
    p.add_argument("--expand", type=float, default=10.0, help="m: bracket extension when the composite moved the root")
    p.add_argument("--reopen", type=float, default=3.0, help="reopen a converged basin when |s| exceeds this x tol-s")
    p.add_argument("--speed-sigma", type=float, default=0.15)
    p.add_argument("--thick-sigma", type=float, default=0.2)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("step", parents=[common])
    p = sub.add_parser("status", parents=[common])
    p.add_argument("--n", type=int, default=20, help="also list this many of the largest other basins")
    p = sub.add_parser("finalize", parents=[common])
    p.add_argument("--domain-path", default=str(HERE / "domains" / "greenland"))
    p.add_argument("--out", required=True, help="file name under model_inputs/")
    a = ap.parse_args()
    dict(init=cmd_init, step=cmd_step, status=cmd_status, finalize=cmd_finalize)[a.cmd](a)


if __name__ == "__main__":
    main()
