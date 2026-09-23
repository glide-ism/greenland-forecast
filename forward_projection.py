"""
ISMIP7 projection: the MAP forward of forward_standalone.py, driven by the
ISMIP7 CMIP6-derived forcing instead of the calibration-era forcing.

Same model, same calibrated fields (bed, beta, precip / temperature biases,
enthalpy parameters from {results}/physical_fields.nc), same time loop and
ocean-forced calving margins (glacier_inverse/ocean.py) -- only the three
forcing records are swapped:

    CARRA2 monthly climatology + Vinther anomaly  ->  ISMIP7 <gcm>/<scenario> tas, pr
    EN4 thermal forcing (1950-2025)               ->  ISMIP7 <gcm>/<scenario> tf

as prepared by preprocessing/make_ismip7_forcing.py in
model_inputs/ismip7/<gcm>_<scenario>/ (historical 1850-2014 spliced with the
scenario 2015-2300; the yearly tas / pr files are read from the kit per step
through catalogue.json, the climatologies and the annual TF statistics from
climate.nc / thermal_forcing.nc). One seamless run from T_START to T_END
(2301: the ISMIP7 record ends with nominal year 2300, i.e. the step
(2300, 2301]; the kit's last year is held for it). T_START and DT default to
the INVERSE's own spin-up (config.t_start / config.dt), so the relaxation
handed to the record is the calibrated one:

  * before the record (which starts 1850) the forcing is chosen by
    `--pre-record`. "standalone" delegates to forward_standalone's
    compute_smb, i.e. the inverse's own forcing -- the deep temperature
    index, the index precipitation multiplier and the interannual
    quadrature -- and is what a full spin-up needs. "climatology" holds the
    constant pre-record GCM climatology (tas_pre / pr_pre: 1850-1879
    monthly means) and is refused for spans over 200 yr, since over a
    millennial spin-up it is exactly the flat-hold bias the deep forcing
    work removed. dTF = 0 before the TF record either way. From the first
    record year on, annual steps carry each year's monthly fields (a step
    spanning several years takes their overlap-weighted mean); after the
    last year the last year holds;
  * CLIMATE_MODE "raw": the SMB model sees the ISMIP7 monthly fields as they
    are (2 m temperature from the dEBM2 downscaling, degC; precipitation as
    m ice / yr), nearest-filled onto the 12% of ice cells outside the
    product's footprint, plus the calibrated biases when APPLY_BIASES
    (tbias additive, exp(log_pbias) multiplicative -- calibrated against
    CARRA2, so they are a choice here, not a given);
    CLIMATE_MODE "anomaly": the calibrated CARRA2 climatology + biases, with
    the ISMIP7 departure from its own climatology over the same window
    (tas - tas_clim additive, pr / pr_clim multiplicative). This is the
    bias-corrected option; the raw run shows whether it is needed;
  * the ocean forcing keeps the config's OceanForcingConfig semantics
    (statistic, ref_years, tf_crit, clim_*, alpha_*) on the CESM2 record:
    TF_clim is CESM2's own 1950-79 mean, dTF its departure from it;
  * alpha_t2m / base_anomaly_year / the Vinther series are not used;
  * surface-elevation feedback (ELEVATION_FEEDBACK, on by default): the
    forcing temperature follows the evolving model surface,
    t2m += FEEDBACK_LAPSE(month) * (S_model(t) - S_ref), with S_ref the model
    surface at FEEDBACK_T_REF (2015, the ISMIP projection start: nothing
    changes before it) and the lapse the along-surface gradient of the
    forcing itself (-5.4 to -6.0 K/km). Precipitation does not respond.
    scalars.csv carries dS_ice_mean and dT_feedback_jja.

Outputs in {results}/projection_<gcm>_<scenario>/:
  scalars.csv      one row per step (volume, volume above flotation in mm
                   SLE, ice / grounded / floating area, integrated SMB, the
                   forcing's ice-sheet means, dTF, wall time) -- for a first
                   look and for monitoring a running job
  snapshots.nc     (time, y, x) fields every SNAPSHOT_EVERY years on the run
                   level (H, srf, dhdt, velocities, smb, phi, psi, q, h0, tf_anom)
  final_state.nc   the last step, forward_soln.nc's layout
  vti/             ParaView series every VTI_EVERY years (0 = off; 290 MB
                   per 1 km snapshot)

    python forward_projection.py                       # constants below
    python forward_projection.py --scenario ssp585 --level 1 --t-end 2100

The run level (LEVEL 0 = 1 km, 1 = 2 km, ...) restricts the state as the
inverse does; the SMB is always evaluated on the 1 km grid and restricted.
Only domains whose grid IS the ISMIP grid up to a y flip (ismip_greenland
preset) can read the kit's files directly (make_ismip7_forcing.py records
the mapping; 'regrid' is refused here).
"""
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cupy as cp
import netCDF4
import numpy as np
import xarray as xr
from scipy import ndimage

import forward_standalone as fs
from glacier_inverse.ocean import OceanForcing, year_overlap_weights
from glacier_inverse.scheduling import build_step_sequence

# ----------------------------------------------------------------- settings
GCM, SCENARIO = "CESM2-WACCM", "ssp126"
LEVEL = 0                         # run level (0 = 1 km)
# None = the inverse's own spin-up (config.t_start / config.dt), so a
# projection relaxes exactly as the calibration did before the record takes
# over. T_END 2301: ISMIP's nominal year 2300 is the step (2300, 2301]; 2101
# for ssp370. Override here or with --t-start / --t-end.
T_START, T_END = None, 2301.0
DT = None                         # step before the record; None = config.dt
DT_SCHEDULE = ((1850.0, 1.0),)    # annual steps from the record's first year on
CLIMATE_MODE = "raw"              # "raw" | "anomaly" (see the docstring)
APPLY_BIASES = True               # raw mode: add tbias, multiply by exp(log_pbias)
PR_RATIO_MAX = 5.0                # anomaly mode: cap on pr / pr_clim
SNAPSHOT_EVERY = 10.0             # years between snapshots.nc records (0 = final only)
VTI_EVERY = 1                   # years between VTI frames (0 = off)
ICE_H_MIN = 10.0                  # m; "ice" in the scalar diagnostics
# --- surface-elevation feedback: t2m += FEEDBACK_LAPSE * (S_model(t) - S_ref)
ELEVATION_FEEDBACK = True
_cfg = fs.config
T_START = _cfg.t_start if T_START is None else float(T_START)
DT = _cfg.dt if DT is None else float(DT)
FEEDBACK_T_REF = 2015.0           # S_ref = the MODEL surface at this time (no feedback before
                                  # it, so the calibrated hindcast is untouched; ISMIP's h_ref).
                                  # None -> the observed DEM the climatology sits on (feedback
                                  # acts from the first step, model surface errors included)
# K/m per calendar month (or one scalar): the ALONG-SURFACE gradient of the
# forcing temperature itself, regressed over the ice on elevation plus a
# quadratic horizontal trend (CARRA2 100 m climatology on the DEM, 2026-09-17).
# NOT GLIDE_inputs' monthly_lapse_rate: that is the 100-500 m boundary-layer
# gradient above a FIXED surface (an inversion, > 0, over 2/3 of the ice in
# winter; -2 to -4 K/km in summer), not the response to a moving surface.
FEEDBACK_LAPSE = tuple(v / 1000.0 for v in (-5.60, -5.55, -5.37, -5.39, -5.61, -5.80,
                                            -5.99, -6.05, -5.89, -5.73, -5.78, -5.81))
OUT_DIR = None                    # default: {config.output_dir}/projection_{GCM}_{SCENARIO}
FORCING_DIR = None                # default: {base_dir}/model_inputs/ismip7/{GCM}_{SCENARIO}
OCEAN = None                      # None -> config.ocean_forcing (or dataclasses.replace(...))

config = fs.config
RHO_I, RHO_W = float(config.rho_ice), float(config.rho_water)
KG_PER_MM_SLE = 361.8e12          # 361.8 Gt of ice per mm of sea-level equivalent


# ------------------------------------------------------------ climate record
def _crop_slices(ny0: int, nx0: int, factor: int):
    ny, nx = (ny0 // factor) * factor, (nx0 // factor) * factor
    y0, x0 = (ny0 - ny) // 2, (nx0 - nx) // 2
    return slice(y0, y0 + ny), slice(x0, x0 + nx)


class Ismip7Climate:
    """Monthly tas (degC) / pr (m ice / yr) on the cropped fine grid for any
    step, from the kit's yearly files (catalogue.json) and the preprocessed
    climatologies (climate.nc). Cells outside the product's footprint take
    the nearest native value (index computed once; the footprint is
    constant across the kit and checked on every read)."""

    def __init__(self, forcing_dir: Path, crop_factor: int):
        forcing_dir = Path(forcing_dir)
        with open(forcing_dir / "catalogue.json") as f:
            cat = json.load(f)
        if cat["grid_mapping"] not in ("identity", "flip_y"):
            raise NotImplementedError(f"grid mapping {cat['grid_mapping']!r}: this driver reads the kit's "
                                      f"files directly and needs the ISMIP grid (up to a y flip)")
        self.flip = cat["grid_mapping"] == "flip_y"
        self.files = {int(y): v for y, v in cat["years"].items()}
        self.years = sorted(y for y, v in self.files.items() if v["tas"] and v["pr"])
        self.gcm, self.scenario = cat["gcm"], cat["scenario"]
        self.clim_years, self.pre_years = cat["clim_years"], cat["pre_years"]
        with xr.open_dataset(forcing_dir / "climate.nc") as c:
            self.ny0, self.nx0 = c.sizes["y"], c.sizes["x"]
            self.sl = _crop_slices(self.ny0, self.nx0, crop_factor)
            cc = c.isel(y=self.sl[0], x=self.sl[1])
            footprint = np.isfinite(cc["tas_clim"].isel(t=0).values)
            self.footprint = footprint
            idx = ndimage.distance_transform_edt(~footprint, return_distances=False, return_indices=True)
            self.fill_idx = (idx[0], idx[1])
            self.tas_clim = self._fill(cc["tas_clim"].values.astype(np.float32))
            self.pr_clim = self._fill(cc["pr_clim"].values.astype(np.float32))
            self.tas_pre = self._fill(cc["tas_pre"].values.astype(np.float32))
            self.pr_pre = self._fill(cc["pr_pre"].values.astype(np.float32))
        self._cache = {}          # year -> (tas, pr); the last few years only

    def _fill(self, a: np.ndarray) -> np.ndarray:
        """(12, ny, nx): nearest-native fill of the cells outside the footprint."""
        return a[:, self.fill_idx[0], self.fill_idx[1]]

    def _read(self, path: str, var: str) -> np.ndarray:
        with xr.open_dataset(path, decode_times=False) as ds:
            a = ds[var].values.astype(np.float32)
        if a.shape != (12, self.ny0, self.nx0):
            raise ValueError(f"{path}: shape {a.shape}, expected (12, {self.ny0}, {self.nx0})")
        if self.flip:
            a = a[:, ::-1, :]
        a = a[:, self.sl[0], self.sl[1]]
        if not np.array_equal(np.isfinite(a[0]), self.footprint):
            raise ValueError(f"{path}: footprint differs from climate.nc's")
        return self._fill(a)

    def year(self, y: int):
        """(tas degC, pr m ice / yr) monthly fields of calendar year y."""
        if y not in self._cache:
            if len(self._cache) > 3:
                self._cache.pop(min(self._cache))
            f = self.files[y]
            tas = self._read(f["tas"], "tas") - 273.15
            pr = self._read(f["pr"], "pr") * fs.SECONDS_PER_YEAR / config.rho_ice
            self._cache[y] = (tas, pr)
        return self._cache[y]

    def monthly(self, t0: float, t1: float):
        """Overlap-weighted (12, ny, nx) tas / pr for the step (t0, t1]:
        pre-record climatology before the first year, the last year after
        the last."""
        ya, yb = self.years[0], self.years[-1]
        tas = np.zeros_like(self.tas_pre)
        pr = np.zeros_like(self.pr_pre)
        for y, w in year_overlap_weights(t0, t1):
            if y < ya:
                ty, py = self.tas_pre, self.pr_pre
            else:
                ty, py = self.year(min(y, yb))
            tas += w * ty
            pr += w * py
        return tas, pr

    def describe(self) -> str:
        return (f"ISMIP7 {self.gcm} {self.scenario}: tas / pr {self.years[0]}-{self.years[-1]}, "
                f"pre-record forcing = {self.pre_years[0]}-{self.pre_years[1]} climatology, "
                f"anomaly reference {self.clim_years[0]}-{self.clim_years[1]}, "
                f"{int((~self.footprint).sum())} cells nearest-filled")


# ------------------------------------------------------------------- driver
class ElevationFeedback:
    """Surface-elevation feedback on the forcing temperature,

        dT(x, month, t) = lapse(month) * (S_model(x, t) - S_ref(x)),

    evaluated at the start of each step from the run level's state (so it
    lags the geometry by one step) and prolonged onto the fine SMB grid by
    injection. S_model = max(H + bed, (1 - rho_i/rho_w) H): where ice is
    lost the surface falls to the bed (or to sea level), where it advances
    it rises, so the feedback also acts on newly exposed and newly covered
    cells. S_ref is the model surface captured at the first step starting
    at or after `t_ref`, or (t_ref None) the observed DEM, floored at sea
    level, that the climatology was lapse-corrected onto. Only the air
    temperature responds: precipitation keeps its calibrated pattern, and
    glare reads its own surface field for the (unused) avalanche operator
    only, so that is left alone."""

    def __init__(self, ctx: fs.Run, lapse, t_ref):
        self.ctx, self.t_ref = ctx, t_ref
        lapse = np.atleast_1d(np.asarray(lapse, dtype=np.float32))
        self.lapse = cp.asarray(np.broadcast_to(lapse, (12,)).copy())[:, None, None]      # K/m
        self.factor = 2 ** ctx.level
        self.S_ref = None
        if t_ref is None:
            dem = cp.maximum(cp.asarray(ctx.gd.elevation.values, dtype=cp.float32), 0.0)
            self.S_ref = fs.restrict(dem, ctx.level)
        self.dS_fine = cp.zeros((ctx.gd.sizes["y"], ctx.gd.sizes["x"]), dtype=cp.float32)

    def surface(self) -> cp.ndarray:
        lvl = self.ctx.lvl
        H = lvl.state.H.data
        return cp.maximum(H + lvl.geometry.bed.data, (1.0 - RHO_I / RHO_W) * H)

    def dT(self, t: float) -> cp.ndarray:
        """(12, ny, nx) temperature increment on the fine grid for a step starting at t."""
        if self.S_ref is None:
            if t < self.t_ref - 1e-6:
                return None
            self.S_ref = self.surface().copy()
            print(f"  elevation feedback: reference surface captured at t={t:g}", flush=True)
        dS = self.surface() - self.S_ref
        f = self.factor
        self.dS_fine = cp.repeat(cp.repeat(dS, f, axis=0), f, axis=1) if f > 1 else dS
        return self.lapse * self.dS_fine[None]

    def describe(self) -> str:
        l = cp.asnumpy(self.lapse).ravel() * 1000
        ref = "the observed DEM" if self.t_ref is None else f"the model surface at {self.t_ref:g}"
        return (f"elevation feedback on t2m: lapse {l.min():.2f}..{l.max():.2f} K/km (monthly), "
                f"relative to {ref}")


def make_compute_smb(ctx: fs.Run, climate: Ismip7Climate, mode: str, feedback: Optional[ElevationFeedback] = None,
                     pre_record: str = "climatology"):
    """Replace ctx.compute_smb: the enthalpy SMB model on the fine grid with
    the ISMIP7 monthly forcing of the step (+ the surface-elevation feedback
    when given). Records the forcing's ice-sheet means in ctx.forcing_stats
    for the scalars file. `pre_record`: what steps ending at or before the
    record's first year see -- "climatology" (tas_pre / pr_pre, the
    projections) or "standalone" (forward_standalone's own forcing, the
    CARRA2 climatology + the Vinther anomaly index: the OCX run, whose record
    starts in 1986)."""
    g, smb_model, temp_dev, domain_mask = ctx.smb_grid, ctx.smb_model, ctx.temp_dev, ctx.domain_mask
    standalone_smb = ctx.compute_smb                          # forward_standalone's closure (setup)
    if pre_record not in ("climatology", "standalone"):
        raise ValueError(f"pre_record {pre_record!r}")
    nan_stats = {k: float("nan") for k in ("tas_ice_annual", "tas_ice_jja", "pr_ice_annual", "t2m_model_jja",
                                            "precip_model_annual", "smb_ice_mean", "dS_ice_mean", "dT_feedback_jja")}
    ice = cp.asarray(ctx.gd.rgi_mask.values, dtype=bool)
    tbias, pbias = ctx.tbias, cp.exp(ctx.log_pbias)
    if mode == "anomaly":
        tas_clim = cp.asarray(climate.tas_clim)
        pr_clim = cp.asarray(climate.pr_clim)
        pr_ok = pr_clim > 1e-3
    elif mode != "raw":
        raise ValueError(f"CLIMATE_MODE {mode!r}")

    def compute_smb(t_prev: float, t_next: float) -> cp.ndarray:
        if pre_record == "standalone" and t_next <= climate.years[0] + 1e-6:
            smb = standalone_smb(t_prev, t_next)
            ctx.forcing_stats = dict(nan_stats, smb_ice_mean=float(smb[ice].mean()))
            return smb
        tas_np, pr_np = climate.monthly(t_prev, t_next)
        tas, pr = cp.asarray(tas_np), cp.asarray(pr_np)
        if mode == "raw":
            t2m = tas + tbias if APPLY_BIASES else tas
            precip = pr * pbias if APPLY_BIASES else pr
        else:
            t2m = ctx.t2m_clim + (tas - tas_clim) + tbias
            ratio = cp.where(pr_ok, cp.clip(pr / cp.maximum(pr_clim, 1e-6), 0.0, PR_RATIO_MAX), 1.0)
            precip = ctx.precip_clim * ratio
        dT_fb = feedback.dT(t_prev) if feedback is not None else None
        if dT_fb is not None:
            t2m = t2m + dT_fb
        g.temperature.t2m.set(t2m)
        g.precipitation.precip.set(precip)
        smb_model.forward(temp_deviations=temp_dev)
        smb = g.state.smb.data.mean(axis=0)
        smb[~domain_mask] = -10.0
        ctx.forcing_stats = {
            "tas_ice_annual": float(tas[:, ice].mean()), "tas_ice_jja": float(tas[5:8][:, ice].mean()),
            "pr_ice_annual": float(pr[:, ice].mean()),
            "t2m_model_jja": float(t2m[5:8][:, ice].mean()), "precip_model_annual": float(precip[:, ice].mean()),
            "smb_ice_mean": float(smb[ice].mean()),
            # surface change and the feedback's JJA warming over the ORIGINAL ice mask
            "dS_ice_mean": float(feedback.dS_fine[ice].mean()) if dT_fb is not None else 0.0,
            "dT_feedback_jja": float(dT_fb[5:8][:, ice].mean()) if dT_fb is not None else 0.0}
        return smb

    return compute_smb


def make_ocean_loader(forcing_dir: Path, ocean_cfg, q0: float, h00: float):
    def load() -> Optional[OceanForcing]:
        path = Path(forcing_dir) / "thermal_forcing.nc"
        if not ocean_cfg.enabled:
            print(f"ocean forcing disabled: constant margins q = {q0:g}, h0 = {h00:g} m")
            return None
        if not path.exists():
            print(f"no thermal forcing at {path}: constant margins q = {q0:g}, h0 = {h00:g} m")
            return None
        of = OceanForcing.from_file(path, 2 ** config.n_levels, ocean_cfg, q0=q0, h00=h00, lazy=True)
        print(of.describe())
        return of
    return load


def scalars(ctx: fs.Run, t: float, dt: float, vol_prev: float, wall: float) -> dict:
    lvl = ctx.lvl
    H, bed, phi = lvl.state.H.data, lvl.geometry.bed.data, lvl.state.phi.data
    A = float(lvl.dx) ** 2
    ice = H > ICE_H_MIN
    grounded = ice & (phi > 0.5)
    floating = ice & ~grounded
    vol = float((H * ice).sum()) * A
    haf = cp.maximum(H - (RHO_W / RHO_I) * cp.maximum(-bed, 0.0), 0.0)
    vaf = float((haf * grounded).sum()) * A
    smb_int = float((lvl.forcing.smb.data * ice).sum()) * A * RHO_I / 1e12       # Gt / yr
    row = {"time": t, "dt": dt,
           "volume_km3": vol / 1e9, "vaf_mm_sle": vaf * RHO_I / KG_PER_MM_SLE,
           "area_km2": float(ice.sum()) * A / 1e6, "grounded_km2": float(grounded.sum()) * A / 1e6,
           "floating_km2": float(floating.sum()) * A / 1e6,
           "smb_Gt_yr": smb_int, "dvdt_Gt_yr": (vol - vol_prev) / dt * RHO_I / 1e12 if vol_prev else float("nan")}
    row.update(getattr(ctx, "forcing_stats", {}))
    if ctx.ocean is not None:
        ok = fs.restrict(cp.asarray(ctx.ocean.ok, dtype=cp.float32), ctx.level) > 0.5
        dtf = ctx.tf_anom.data
        row["dtf_mean"] = float(dtf[ok].mean()) if bool(ok.any()) else 0.0
        row["dtf_max"] = float(dtf[ok].max()) if bool(ok.any()) else 0.0
    row["wall_s"] = wall
    return row


class SnapshotWriter:
    """(time, y, x) records on the run level, appended as the run goes."""
    FIELDS = ("H", "srf", "dhdt", "u_s", "v_s", "smb", "phi", "psi", "xi", "q", "h0", "tf_anom")

    def __init__(self, path: Path, ctx: fs.Run, attrs: dict, append: bool = False):
        lvl, gd = ctx.lvl, ctx.gd
        if append:
            self.nc = netCDF4.Dataset(path, "a")
            self.yc, self.xc = np.asarray(self.nc["y"][:]), np.asarray(self.nc["x"][:])
            for k, v in attrs.items():
                setattr(self.nc, k, v)
            self.ctx, self.n = ctx, self.nc.dimensions["time"].size
            return
        ny, nx = gd.sizes["y"], gd.sizes["x"]
        f32 = lambda a: cp.asarray(np.asarray(a), dtype=cp.float32)
        yc = cp.asnumpy(fs.restrict(f32(np.broadcast_to(gd.y.values[:, None], (ny, nx))), ctx.level)[:, 0])
        xc = cp.asnumpy(fs.restrict(f32(np.broadcast_to(gd.x.values[None, :], (ny, nx))), ctx.level)[0, :])
        self.yc, self.xc = yc, xc
        self.nc = netCDF4.Dataset(path, "w", format="NETCDF4")
        self.nc.createDimension("time", None)
        self.nc.createDimension("y", len(yc))
        self.nc.createDimension("x", len(xc))
        self.nc.createVariable("time", "f8", ("time",)).units = "years"
        self.nc.createVariable("y", "f8", ("y",))[:] = yc
        self.nc.createVariable("x", "f8", ("x",))[:] = xc
        ch = (1, len(yc), len(xc))
        for name in self.FIELDS:
            self.nc.createVariable(name, "f4", ("time", "y", "x"), zlib=True, complevel=3, chunksizes=ch)
        for name, arr in (("bed", lvl.geometry.bed.data), ("beta", lvl.sliding.beta.data)):
            self.nc.createVariable(name, "f4", ("y", "x"), zlib=True, complevel=3)[:, :] = cp.asnumpy(arr)
        for k, v in attrs.items():
            setattr(self.nc, k, v)
        self.ctx, self.n = ctx, 0

    def fields(self):
        c, lvl = self.ctx, self.ctx.lvl
        return {"H": lvl.state.H.data, "srf": c.srf.data, "dhdt": c.dhdt.data,
                "u_s": 0.5 * (c.u_s.data[:, 1:] + c.u_s.data[:, :-1]),
                "v_s": 0.5 * (c.v_s.data[1:, :] + c.v_s.data[:-1, :]),
                "smb": lvl.forcing.smb.data, "phi": lvl.state.phi.data, "psi": lvl.state.psi.data,
                "xi": lvl.state.xi.data, "q": lvl.calving.q.data, "h0": lvl.calving.h0.data,
                "tf_anom": c.tf_anom.data}

    def append(self, t: float):
        k = self.n
        self.nc["time"][k] = t
        for name, arr in self.fields().items():
            self.nc[name][k, :, :] = cp.asnumpy(arr)
        self.nc.sync()
        self.n += 1

    def close(self):
        self.nc.close()


def _is_multiple(t: float, every: float, eps: float = 1e-6) -> bool:
    return every > 0 and abs(t / every - round(t / every)) < eps


def resume_state(out_dir: Path, ctx: fs.Run, level: int, feedback: Optional["ElevationFeedback"]):
    """--continue: put the run's last state back on the model (H from
    final_state.nc, full precision; velocities start from zero and the first
    momentum solve costs a few extra V-cycles), seed the VTI writer with the
    existing frames so numbering and the .pvd carry on, and rebuild the
    elevation feedback's reference surface from the VTI frame at
    FEEDBACK_T_REF. Returns (t_resume, vol_prev) from scalars.csv."""
    import re
    fin_path, csv_path = out_dir / "final_state.nc", out_dir / "scalars.csv"
    if not fin_path.exists() or not csv_path.exists():
        raise SystemExit(f"--continue needs {fin_path} and {csv_path} (a run that finished)")
    fin = xr.open_dataset(fin_path)
    if int(fin.attrs.get("level", level)) != level:
        raise SystemExit(f"--continue: the run is on level {fin.attrs.get('level')}, not {level}")
    rows = list(csv.DictReader(open(csv_path)))
    t_resume, vol_prev = float(rows[-1]["time"]), float(rows[-1]["volume_km3"]) * 1e9
    H = cp.asarray(fin.H.values, dtype=cp.float32)
    if H.shape != ctx.lvl.state.H.data.shape:
        raise SystemExit(f"--continue: final_state.nc grid {H.shape} != run level grid {ctx.lvl.state.H.data.shape}")
    ctx.mg.state.H.set(H, start_level=level)
    ctx.mg.state.H_prev.set(H, start_level=level)
    # VTI: continue the numbering and the manifest
    w = ctx.vti_writer
    pvd = Path(w.out_dir) / f"{w.base}.pvd"
    if pvd.exists():
        items = re.findall(r'timestep="([\d.]+)"[^>]*file="([^"]+)"', pvd.read_text())
        w.records = [(float(t), fn) for t, fn in items]
        w._step_idx = max((int(m.group(1)) for fn in w.records for m in [re.search(r"_(\d+)\.vti$", fn[1])] if m), default=-1) + 1
        if w.records and abs(w.records[-1][0] - t_resume) > 1e-6:
            print(f"  WARNING: last VTI frame at t={w.records[-1][0]:g}, scalars end at t={t_resume:g}")
    # elevation feedback reference: the model surface at FEEDBACK_T_REF, from that frame
    if feedback is not None and feedback.S_ref is None and t_resume >= feedback.t_ref - 1e-6:
        from ismip_exporter import read_vti
        hit = [fn for t, fn in w.records if abs(t - feedback.t_ref) < 1e-6]
        if not hit:
            raise SystemExit(f"--continue: no VTI frame at FEEDBACK_T_REF = {feedback.t_ref:g} to rebuild the feedback "
                             f"reference surface from (use --no-elevation-feedback to continue without it)")
        srf = read_vti(Path(w.out_dir) / hit[0], ["srf"])["srf"]
        feedback.S_ref = cp.asarray(srf, dtype=cp.float32)
        print(f"  elevation feedback: reference surface rebuilt from the frame at t={feedback.t_ref:g}")
    print(f"--continue: state at t={t_resume:g} from {fin_path.name}; {len(w.records)} VTI frames, "
          f"{len(rows)} scalar rows, next frame index {w._step_idx}")
    return t_resume, vol_prev


def run(level: int, out_dir: Path, forcing_dir: Path, t_start: float, t_end: float,
        dt: float, dt_schedule, mode: str, ocean_cfg, elevation_feedback: bool = ELEVATION_FEEDBACK,
        continue_run: bool = False, pre_record: str = "climatology") -> None:
    q0, h00 = fs.Q0, fs.H00
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = fs.setup(level=level, out_dir=out_dir,
                   ocean_loader=make_ocean_loader(forcing_dir, ocean_cfg, q0, h00))
    climate = Ismip7Climate(forcing_dir, 2 ** config.n_levels)
    assert climate.tas_clim.shape[1:] == (ctx.gd.sizes["y"], ctx.gd.sizes["x"])
    print(climate.describe())
    print(f"climate mode {mode!r}" + (f", calibrated biases {'applied' if APPLY_BIASES else 'dropped'}"
                                       if mode == "raw" else ""))
    feedback = ElevationFeedback(ctx, FEEDBACK_LAPSE, FEEDBACK_T_REF) if elevation_feedback else None
    print(feedback.describe() if feedback is not None else "elevation feedback OFF: forcing stays on the observed DEM")
    if feedback is not None and FEEDBACK_T_REF is not None and not (t_start - 1e-6 <= FEEDBACK_T_REF <= t_end):
        print(f"  WARNING: FEEDBACK_T_REF {FEEDBACK_T_REF:g} lies outside the run {t_start:g}-{t_end:g}: "
              f"the reference is captured at the first step at or after it, or never")
    # A long pre-record span has to be spun up the way the INVERSE was, or the
    # state handed to the record is not the calibrated one. "standalone"
    # delegates to forward_standalone's compute_smb, which carries the deep
    # temperature series, the index precipitation multiplier and the
    # interannual quadrature; "climatology" holds one 30-yr GCM mean, which
    # over a millennial spin-up is the flat-hold bias the deep forcing work
    # removed. Refused rather than warned: it costs a whole run to discover.
    pre_span = float(climate.years[0]) - t_start
    if pre_record == "climatology" and pre_span > 200.0:
        raise SystemExit(
            f"--pre-record climatology would hold the {climate.pre_years[0]}-{climate.pre_years[1]} "
            f"GCM climatology for {pre_span:.0f} yr before the record, with no anomaly index, "
            f"precipitation response or variance correction -- not how the inversion spun up.\n"
            f"Use --pre-record standalone, or --t-start {climate.years[0] - 200:g} for a short "
            f"pre-industrial relaxation.")
    print(f"before the record ({climate.years[0]}), {pre_span:.0f} yr: "
          + ("forward_standalone's forcing, i.e. the inverse's own (deep temperature index"
             f"{', precip multiplier' if getattr(_cfg, 'alpha_precip', 0.0) else ''}"
             f"{', interannual quadrature' if getattr(_cfg, 'interannual_sigma', None) else ''})"
             if pre_record == "standalone" else f"the {climate.pre_years[0]}-{climate.pre_years[1]} climatology"))
    ctx.compute_smb = make_compute_smb(ctx, climate, mode, feedback, pre_record)
    ctx.forcing_stats = {}
    vol_prev = 0.0
    if continue_run:
        t_resume, vol_prev = resume_state(out_dir, ctx, level, feedback)
        if t_end <= t_resume + 1e-6:
            raise SystemExit(f"--continue: the run already reaches t={t_resume:g}; --t-end {t_end:g} adds nothing")
        t_start = t_resume

    attrs = dict(level=level, t_start=t_start, t_end=t_end, gcm=climate.gcm, scenario=climate.scenario,
                 climate_mode=mode, apply_biases=int(APPLY_BIASES), checkpoint=str(fs.CHECKPOINT),
                 pre_record=pre_record,
                 elevation_feedback=(feedback.describe() if feedback is not None else "off"),
                 crs_wkt=ctx.crs.to_wkt(), climate=climate.describe(),
                 ocean_forcing=(ctx.ocean.describe() if ctx.ocean is not None
                                else f"constant margins q = {q0:g}, h0 = {h00:g} m"))
    if continue_run:
        attrs["continued_from"] = f"t={t_start:g} ({datetime.now().isoformat(timespec='seconds')}); velocity warm start reset"
        snaps = SnapshotWriter(out_dir / "snapshots.nc", ctx, {"continued_from": attrs["continued_from"], "t_end": t_end}, append=True)
    else:
        snaps = SnapshotWriter(out_dir / "snapshots.nc", ctx, attrs)
    seq = build_step_sequence(t_start=t_start, t_end=t_end, dt_max=dt, dt_schedule=dt_schedule)
    ends = [t for t, _ in seq]
    steps = list(zip([t_start] + ends[:-1], ends))
    print(f"{len(steps)} steps {t_start:g}-{t_end:g}: first {steps[0][1] - steps[0][0]:g} yr, "
          f"last {steps[-1][1] - steps[-1][0]:g} yr")

    csv_path = out_dir / "scalars.csv"
    writer, csv_file = None, None
    if continue_run:
        header = next(csv.reader(open(csv_path)))
        csv_file = open(csv_path, "a", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=header, extrasaction="ignore")
    try:
        for k, (t_prev, t_next) in enumerate(steps):
            dt_step = t_next - t_prev
            tic = time.time()
            print(f"Solving forward problem at t={t_prev:.2f} with dt={dt_step:.2f}", flush=True)
            ctx.mg.forcing.smb.set(fs.restrict(ctx.compute_smb(t_prev, t_next), level), start_level=level)
            fs.ocean_forcing(t_prev, dt_step, ctx.mg, level, ctx)
            fs.dynamics_step(ctx, t_prev, dt_step)
            ctx.update_derived(dt_step)
            row = scalars(ctx, t_next, dt_step, vol_prev, time.time() - tic)
            vol_prev = row["volume_km3"] * 1e9
            if writer is None:
                csv_file = open(csv_path, "w", newline="")
                writer = csv.DictWriter(csv_file, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            csv_file.flush()
            print(f"  t={t_next:.1f}: V {row['volume_km3']:.0f} km3, VAF {row['vaf_mm_sle']:.1f} mm SLE, "
                  f"area {row['area_km2']:.0f} km2 (floating {row['floating_km2']:.0f}), SMB {row['smb_Gt_yr']:+.0f} Gt/yr, "
                  f"dV/dt {row['dvdt_Gt_yr']:+.0f} Gt/yr, T_jja {row.get('tas_ice_jja', float('nan')):+.2f} C, "
                  f"{row['wall_s']:.0f} s", flush=True)
            last = k == len(steps) - 1
            if last or _is_multiple(t_next, SNAPSHOT_EVERY):
                snaps.append(t_next)
            if VTI_EVERY > 0 and (last or _is_multiple(t_next, VTI_EVERY)):
                ctx.vti_writer.append(ctx.lvl, time=float(t_next))
                ctx.vti_writer.write_pvd()
    finally:
        snaps.close()
        if csv_file is not None:
            csv_file.close()

    # final state in forward_soln.nc's layout
    final = xr.Dataset(coords={"y": snaps.yc, "x": snaps.xc})
    for name, arr in list(snaps.fields().items()) + [("bed", ctx.lvl.geometry.bed.data),
                                                     ("beta", ctx.lvl.sliding.beta.data),
                                                     ("mask", ctx.lvl.state.mask.data)]:
        final[name] = xr.DataArray(cp.asnumpy(arr), dims=("y", "x"))
    if continue_run:
        old = {k: v for k, v in xr.open_dataset(out_dir / "final_state.nc").attrs.items()}
        old.update(t_end=t_end, continued_from=attrs["continued_from"])
        attrs = old
    final.attrs.update(attrs)
    final.to_netcdf(out_dir / "final_state.nc.tmp")
    (out_dir / "final_state.nc.tmp").replace(out_dir / "final_state.nc")
    print(f"wrote {out_dir / 'final_state.nc'}, {out_dir / 'snapshots.nc'} ({snaps.n} records), {csv_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gcm", default=GCM)
    ap.add_argument("--scenario", default=SCENARIO)
    ap.add_argument("--level", type=int, default=LEVEL)
    ap.add_argument("--t-start", type=float, default=T_START)
    ap.add_argument("--t-end", type=float, default=T_END)
    ap.add_argument("--mode", default=CLIMATE_MODE, choices=("raw", "anomaly"))
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--forcing-dir", default=FORCING_DIR)
    ap.add_argument("--pre-record", default="climatology", choices=("climatology", "standalone"),
                    help="forcing before the record's first year: the pre-record climatology (projections) or "
                         "forward_standalone's CARRA2 climatology + Vinther anomaly (the OCX run)")
    ap.add_argument("--no-elevation-feedback", action="store_true",
                    help="keep the forcing temperature on the observed DEM (the pre-2026-09-17 behaviour)")
    ap.add_argument("--continue", dest="continue_run", action="store_true",
                    help="resume the run in --out-dir from its final state and run on to --t-end "
                         "(appends to scalars.csv, snapshots.nc and the VTI series)")
    ap.add_argument("--export", action="store_true", help="only (re)write physical_fields.nc")
    a = ap.parse_args()
    if a.export or not fs.PHYSICAL_PATH.exists():
        fs.export_physical_fields(fs.CHECKPOINT, fs.PHYSICAL_PATH)
        if a.export:
            return
    forcing_dir = Path(a.forcing_dir or f"{config.base_dir}/model_inputs/ismip7/{a.gcm}_{a.scenario}")
    out_dir = Path(a.out_dir or f"{config.output_dir}/projection_{a.gcm}_{a.scenario}")
    ocean_cfg = config.ocean_forcing if OCEAN is None else OCEAN
    run(a.level, out_dir, forcing_dir, float(a.t_start), float(a.t_end), float(DT), DT_SCHEDULE,
        a.mode, ocean_cfg,
        elevation_feedback=ELEVATION_FEEDBACK and not a.no_elevation_feedback, continue_run=a.continue_run,
        pre_record=a.pre_record)


if __name__ == "__main__":
    main()
