"""
Convert a run's VTI series to one compressed CF NetCDF, `<run-dir>/series.nc`.

The VTI frames are raw float32 (326 MB per 1 km frame, 140 GB for a
1800-2300 projection) and the velocity and SMB fields do not compress
because the 2/3 of the grid that is ice-free holds solver noise. This
writes every frame's fields as (time, y, x) variables, chunked per time
step, with

  * velocities and SMB zeroed where the active-set mask is 1 (ice-free);
  * values rounded to a physical precision (PRECISION below: 1 cm for H and
    the surface, 0.01 m/yr for velocities, 1 mm/yr for SMB and dh/dt, 1e-4
    for the flotation flags) so the low bits compress;
  * zlib + the shuffle filter.

A 1 km frame goes from 326 MB to about 40 MB; nothing the model resolves is
lost, and the ISMIP exporter, analysis/basin_mass_balance.py and ParaView
(NetCDF CF reader: `time` becomes the animation axis) read the result.
Static fields (bed, beta) are stored once. Coordinates come from the
domain's GLIDE_inputs.nc, cropped as the run was (y north to south, as in
the model), with an EPSG:3413 grid mapping.

    python tools/vti_to_nc.py --run-dir domains/greenland/inverse/projection_CESM2-WACCM_ssp585
    python tools/vti_to_nc.py --run-dir ... --delete-vti      # after the read-back check passes

The read-back check compares a sample of frames against the VTI within the
rounding tolerance before anything is deleted.
"""
import argparse
import re
import sys
import time as _time
from datetime import datetime
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr
from netCDF4 import date2num

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ismip_exporter import read_vti, frame_index  # noqa: E402

TIME_UNITS, TIME_CALENDAR = "days since 1850-01-01", "standard"
PRECISION = {"H": 0.01, "srf": 0.01, "dhdt": 1e-3, "smb": 1e-3, "u": 0.01, "v": 0.01, "u_s": 0.01, "v_s": 0.01,
             "u_b": 0.01, "v_b": 0.01, "q": 1e-4, "h0": 0.01, "tf_anom": 1e-3, "xi": 1e-4, "phi": 1e-4, "psi": 1e-4,
             "mask": 1.0, "bed": 0.01, "beta": 1e-3}
MASK_OFF_ICE = {"u", "v", "u_s", "v_s", "u_b", "v_b", "smb", "dhdt"}
ATTRS = {"H": ("Ice thickness", "m"), "srf": ("Surface elevation", "m"), "dhdt": ("Thickness change rate over the step", "m a-1"),
         "smb": ("Surface mass balance forcing of the step", "m ice a-1"), "mask": ("Active-set mask (1 = ice-free, pinned at thklim)", "1"),
         "xi": ("Flotation fraction (effective pressure)", "1"), "phi": ("Grounded flag", "1"), "psi": ("Calving flag (0 = calving)", "1"),
         "q": ("Calving margin, fraction of thickness", "1"), "h0": ("Calving margin, absolute", "m"),
         "tf_anom": ("Ocean thermal forcing anomaly", "degC"), "u": ("Depth-mean velocity x", "m a-1"), "v": ("Depth-mean velocity y", "m a-1"),
         "u_s": ("Surface velocity x", "m a-1"), "v_s": ("Surface velocity y", "m a-1"), "u_b": ("Basal velocity x", "m a-1"),
         "v_b": ("Basal velocity y", "m a-1"), "bed": ("Bed elevation", "m"), "beta": ("Basal traction coefficient as used", "head per (m a-1)^m")}
VECTORS = {"U": ("u", "v"), "U_s": ("u_s", "v_s"), "U_b": ("u_b", "v_b")}


def vti_arrays(path):
    """Names and component counts of the appended arrays of a VTI file."""
    head = open(path, "rb").read(1 << 20)
    xml = head[:head.index(b"<AppendedData")].decode("utf-8", "ignore")
    return [(m.group(1), int(m.group(2))) for m in re.finditer(r'Name="(\w+)" NumberOfComponents="(\d+)" format="appended"', xml)]


def split(frame):
    """VTI arrays -> scalar fields (vectors split into components)."""
    out = {}
    for n, a in frame.items():
        if n in VECTORS:
            for k, comp in enumerate(VECTORS[n]):
                out[comp] = a[..., k]
        else:
            out[n] = a
    return out


def quantize(name, a, ice):
    if name in MASK_OFF_ICE:
        a = np.where(ice, a, 0.0)
    p = PRECISION.get(name)
    return (np.round(a / p) * p).astype(np.float32) if p else a.astype(np.float32)


def convert(run_dir, domain_path, out_path=None, complevel=4, drop=()):
    run_dir = Path(run_dir)
    out_path = Path(out_path) if out_path else run_dir / "series.nc"
    frames, static_path = frame_index(run_dir)
    names = [n for n, _ in vti_arrays(frames[0][1]) if n != "TimeValue" and n not in drop]
    static_names = [n for n, _ in vti_arrays(static_path) if n not in drop]
    st = split(read_vti(static_path, static_names))
    ny, nx = next(iter(st.values())).shape

    gi = xr.open_dataset(Path(domain_path) / "model_inputs" / "GLIDE_inputs.nc")
    y0, x0 = (gi.sizes["y"] - ny) // 2, (gi.sizes["x"] - nx) // 2
    y, x = gi.y.values[y0:y0 + ny].astype("float64"), gi.x.values[x0:x0 + nx].astype("float64")
    crs_attrs = dict(gi["spatial_ref"].attrs) if "spatial_ref" in gi else {}

    meta = {}
    for fn in ("snapshots.nc", "forward_soln.nc", "final_state.nc"):
        if (run_dir / fn).exists():
            with xr.open_dataset(run_dir / fn) as s:
                meta = {k: str(v) for k, v in s.attrs.items() if k != "crs_wkt"}
            break

    nc = netCDF4.Dataset(out_path, "w", format="NETCDF4")
    nc.createDimension("time", None); nc.createDimension("y", ny); nc.createDimension("x", nx)
    t = nc.createVariable("time", "f8", ("time",))
    t.units, t.calendar, t.axis, t.standard_name, t.long_name = TIME_UNITS, TIME_CALENDAR, "T", "time", "end of the step"
    my = nc.createVariable("model_year", "f8", ("time",)); my.units = "years"; my.long_name = "model time (decimal year)"
    for d, vals in (("y", y), ("x", x)):
        c = nc.createVariable(d, "f8", (d,)); c[:] = vals
        c.units, c.standard_name, c.axis = "m", f"projection_{d}_coordinate", d.upper()
    crs = nc.createVariable("spatial_ref", "i4")
    for k, v in crs_attrs.items():
        crs.setncattr(k, v)
    crs.grid_mapping_name = "polar_stereographic"
    kw = dict(zlib=True, complevel=complevel, shuffle=True)
    var = {}
    scalar_names = [c for m in names for c in (VECTORS[m] if m in VECTORS else (m,))]
    for n in scalar_names:
        var[n] = v = nc.createVariable(n, "f4", ("time", "y", "x"), chunksizes=(1, ny, nx), **kw)
        v.long_name, v.units = ATTRS.get(n, (n, ""))
        v.grid_mapping = "spatial_ref"
        if n in MASK_OFF_ICE:
            v.comment = "zero on ice-free cells (mask = 1)"
        if n in PRECISION:
            v.precision = PRECISION[n]
    for n, a in st.items():
        v = nc.createVariable(n, "f4", ("y", "x"), chunksizes=(ny, nx), **kw)
        v.long_name, v.units = ATTRS.get(n, (n, ""))
        v.grid_mapping = "spatial_ref"
        v[:, :] = quantize(n, a, None)
    for k, v in meta.items():
        nc.setncattr(f"run_{k}", v)
    nc.source_vti = str((run_dir / "vti").resolve())
    nc.history = f"tools/vti_to_nc.py {datetime.utcnow().isoformat()}Z"
    nc.Conventions = "CF-1.7"

    tic = _time.time()
    for k, (tt, path) in enumerate(frames):
        fr = split(read_vti(path, names))
        ice = fr["mask"] < 0.5 if "mask" in fr else None
        for n, a in fr.items():
            var[n][k, :, :] = quantize(n, a, ice)
        yr = int(np.floor(tt)); frac = tt - yr
        t[k] = date2num(datetime(yr, 1, 1), TIME_UNITS, TIME_CALENDAR) + frac * 365.0
        my[k] = tt
        if k % 25 == 0 or k == len(frames) - 1:
            nc.sync()
            print(f"  frame {k + 1}/{len(frames)} t={tt:g} ({_time.time() - tic:.0f} s, {out_path.stat().st_size / 1e9:.1f} GB)", flush=True)
    nc.close()
    vti_bytes = sum(p.stat().st_size for _, p in frames) + static_path.stat().st_size
    print(f"wrote {out_path}: {out_path.stat().st_size / 1e9:.1f} GB from {vti_bytes / 1e9:.1f} GB of VTI "
          f"(x{vti_bytes / out_path.stat().st_size:.1f})")
    return out_path


def verify(run_dir, out_path, n_check=5):
    """Read a sample of frames back and compare with the VTI within tolerance."""
    frames, static_path = frame_index(Path(run_dir))
    ds = netCDF4.Dataset(out_path)
    idx = sorted(set([0, len(frames) - 1] + list(np.linspace(0, len(frames) - 1, n_check).astype(int))))
    worst = 0.0
    for k in idx:
        tt, path = frames[k]
        assert abs(float(ds["model_year"][k]) - tt) < 1e-6, (k, tt, float(ds["model_year"][k]))
        fr = split(read_vti(path, [n for n, _ in vti_arrays(path) if n != "TimeValue"]))
        ice = fr["mask"] < 0.5
        for n, a in fr.items():
            if n not in ds.variables:
                continue
            b = ds[n][k, :, :].filled(np.nan) if hasattr(ds[n][k, :, :], "filled") else ds[n][k, :, :]
            ref = np.where(ice, a, 0.0) if n in MASK_OFF_ICE else a
            tol = 0.5 * PRECISION.get(n, 0.0) + 1e-6 * np.abs(ref).max()
            err = float(np.abs(b - ref).max())
            worst = max(worst, err / max(tol, 1e-30))
            if err > tol:
                raise SystemExit(f"read-back mismatch: frame {k} ({tt:g}) field {n}: max |diff| {err:.3e} > tolerance {tol:.3e}")
    ds.close()
    print(f"read-back check passed on frames {idx} (worst error {worst:.2f} x tolerance)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--domain-path", default="domains/greenland")
    ap.add_argument("--out", default=None, help="default <run-dir>/series.nc")
    ap.add_argument("--complevel", type=int, default=4)
    ap.add_argument("--drop", nargs="*", default=[], help="VTI arrays to leave out (e.g. U_s srf, both derivable)")
    ap.add_argument("--delete-vti", action="store_true", help="remove the vti/ directory after the read-back check passes")
    ap.add_argument("--verify-only", action="store_true", help="only run the read-back check on an existing series.nc")
    a = ap.parse_args()
    out = Path(a.out) if a.out else Path(a.run_dir) / "series.nc"
    if not a.verify_only:
        out = convert(a.run_dir, a.domain_path, out, a.complevel, set(a.drop))
    verify(a.run_dir, out)
    if a.delete_vti:
        import shutil
        vti_dir = Path(a.run_dir) / "vti"
        shutil.rmtree(vti_dir)
        print(f"removed {vti_dir}")
