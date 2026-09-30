"""
Standalone forward run of the MAP solution, in the shape of glide's
examples/greenland/greenland_forward.py.

Runs the inverse problem's composite forward (enthalpy SMB -> glide ice
dynamics over the historical window, driven by the calibrated bed, basal
traction, precip bias, temperature bias and enthalpy parameters) with no
autograd, no priors, no observations: plain cupy/numpy state, an explicit
time loop, a VTI writer. Meant as the sandbox for forcing experiments
(thermal forcing / retreat at tidewater fronts) that do not need gradients.

The calibrated fields come from the inverse checkpoint through ONE export
step (needs the glacier_inverse library; see export_physical_fields), cached
in {results}/physical_fields.nc. Everything after that reads only
GLIDE_inputs.nc, temperature_anomaly.nc and that file.

    python forward_standalone.py                # export if needed, then run
    python forward_standalone.py --export       # only refresh the export

Per-step hook: `ocean_forcing(t, dt, mg, level, ctx)` is called before each
dynamics step. Its default applies the config's ocean forcing exactly as the
inverse does (glacier_inverse/ocean.py): the hybrid calving threshold
H - H_f < q H + h0 with

    q (x, t) = Q0  + alpha_q * dTF(x, t)
    h0(x, t) = H00 + alpha_h * dTF(x, t)

and dTF the ISMIP7 thermal-forcing anomaly of the step (OceanForcingConfig:
statistic, reference window, reach, bounds). OCEAN below overrides the
config block for experiments (dataclasses.replace); a disabled block or a
missing thermal_forcing.nc leaves the constant margins in place.

Model setup (geometry, sliding, calving, solver options) follows glide's
examples/greenland/greenland_forward.py after the 2026-09 refactor: signed
depth = -bed, sigmoid_c 1/m, height-above-buoyancy calving
(calving.timescale, calving.q), thklim 1 m, beta capped at BETA_MAX, and the
FAS/Vanka settings of config.forward_solver.
"""
import argparse
import dataclasses
import math
from pathlib import Path
from typing import Optional

import cupy as cp
import numpy as np
import pyproj
import xarray as xr

from glide.model import IceDynamics
from glide.field import Field, GridEntity
from glide.io import VTIWriter
from glare.enthalpy import EnthalpyModel, SECONDS_PER_YEAR, generate_temp_deviations

from glacier_inverse import load_config   # config only: physics constants and paths
from glacier_inverse.ocean import OceanForcing   # TF -> calving margins (shared with the inverse)
from glacier_inverse.scheduling import build_step_sequence   # the inverse's step design

# ----------------------------------------------------------------- settings
DOMAIN = "domains/greenland"
LEVEL = 0                         # multigrid level to run on (0 = finest, 1 km)
CHECKPOINT = None                 # default: {config.output_dir}/level_0/torch_vars.p
OUT_DIR = None                    # default: {config.output_dir}/forward_standalone
#T_START, T_END, DT = 1850, 2100, 1   # default: the config's window
T_START, T_END, DT = None, None, None   # default: the config's window
DT_SCHEDULE = None                # None = config.dt_schedule. Otherwise a tuple of
                                  # (from_time, dt_max) pairs refining the uniform DT
                                  # grid after each time, e.g. ((1850.0, 10.0),) to run
                                  # a long DT spin-up and drop to 10-yr steps at 1850.
                                  # Use it with T_START/DT to append a spin-up here
                                  # without touching the config the inverse ran on.
SNAP_TIMES = ()                   # extra breakpoints (e.g. observation epochs) so the
                                  # step sequence matches the inverse's exactly
RESET_VELOCITY = False            # zero u, v, ud, vd before each momentum solve: with
                                  # the pre-refactor solver settings the warm start
                                  # made the FAS solve diverge on the second 1 km step
BETA_MAX = 20.0                   # cap on the basal traction coefficient (the example
                                  # clips its inverted beta at 20)
# --- VTI frames: ParaView-native LZ4-compressed appended data (glide VTIWriter,
# 2026-09-17), with the fields rounded to physical precision and the velocities
# / SMB / dh/dt zeroed on ice-free cells (active-set mask = 1), where they hold
# solver noise. 326 -> ~75 MB per 1 km frame and the write is faster than the
# raw one. None = the raw layout. tools/vti_compress.py applies the same to
# existing runs; ismip_exporter.read_vti reads both.
VTI_COMPRESSOR = "lz4"
VTI_T_MIN = None                  # write VTI frames only for step ends >= this year (None = all):
                                  # a sweep keeps the observational period and nothing before it
VTI_FIELDS = None                 # subset of the dynamic field names to write (None = all)
STATE_SAVE_TIMES = ()             # step ends at which the raw state is saved to {out_dir}/state_{t}.nc
                                  # (a restart point; the observation epochs for a sweep evaluation)
VTI_PRECISION = {"H": 0.01, "srf": 0.01, "dhdt": 1e-3, "smb": 1e-3, "U": 0.01, "U_s": 0.01, "U_b": 0.01,
                 "q": 1e-4, "h0": 0.01, "tf_anom": 1e-3, "xi": 1e-4, "phi": 1e-4, "psi": 1e-4,
                 "bed": 0.01, "beta": 1e-3, "T_bed": 0.01, "T_mean": 0.01, "omega_w_bed": 1e-5}
VTI_MASKED_FIELDS = ("U", "U_s", "U_b", "smb", "dhdt")
# --- ocean forcing: None -> config.ocean_forcing, or an override such as
# dataclasses.replace(config.ocean_forcing, alpha_q=0.0, alpha_h=50.0)
OCEAN = None
Q0, H00 = None, None              # baseline margins; None -> config.calving_q / calving_h0

config = load_config(DOMAIN)

CHECKPOINT = CHECKPOINT or f"{config.output_dir}/level_0/torch_vars.p"
OUT_DIR = Path(OUT_DIR or f"{config.output_dir}/forward_standalone")
PHYSICAL_PATH = Path(config.output_dir) / "physical_fields.nc"
T_START = config.t_start if T_START is None else T_START
T_END = config.t_end if T_END is None else T_END
DT = config.dt if DT is None else DT
DT_SCHEDULE = config.dt_schedule if DT_SCHEDULE is None else tuple(
    (float(t), float(d)) for t, d in DT_SCHEDULE)
Q0 = config.calving_q if Q0 is None else float(Q0)
H00 = config.calving_h0 if H00 is None else float(H00)
OCEAN = config.ocean_forcing if OCEAN is None else OCEAN
THERMAL_PATH = Path(config.base_dir) / "model_inputs" / OCEAN.filename


# ----------------------------------------------------------- one-time export
def export_physical_fields(checkpoint: str, out_path: Path) -> None:
    """Map the whitened MAP checkpoint to physical fields on the (cropped)
    fine grid and write them to NetCDF. The only place the inverse library
    (priors, GP maps, bed conditioning) is used."""
    import torch
    from glacier_inverse import GlacierProblem
    from glacier_inverse.io import load_whitened_params_into

    problem = GlacierProblem(config)
    load_whitened_params_into(problem.params, checkpoint, priors=problem.priors)
    with torch.no_grad():
        ph = problem.physical_from(problem.params)
        log_pbias = problem.effective_log_pbias(ph)
        H0 = problem._initial_thickness_from_geometry(ph.bed)
    gd = problem.gridded_data
    f32 = lambda t: t.detach().to(torch.float32).cpu().numpy()
    fields = {
        "bed": (f32(ph.bed), "calibrated bed (conditioned)", "m"),
        "bed_mean": (f32(ph.bed_mean), "bed prior mean", "m"),
        "beta": (f32(torch.exp(ph.log_beta)), "basal traction coefficient", "head per (m/yr)^m"),
        "log_pbias": (f32(log_pbias), "log precip multiplier (Matern field minus elevation depletion)", "-"),
        "tbias": (f32(ph.tbias) if ph.tbias is not None else np.zeros(ph.bed.shape, np.float32),
                  "additive air temperature bias", "K"),
        "H_atm": (f32(torch.exp(ph.log_H_atm)), "turbulent exchange coefficient", "W m-2 K-1"),
        "f_clear": (f32(torch.sigmoid(ph.logit_cloud)), "clear-sky fraction", "-"),
        "H_init": (f32(H0), "initial thickness from the observed surface and the calibrated bed", "m"),
    }
    ds = xr.Dataset(coords={"y": gd.y, "x": gd.x})
    ds["spatial_ref"] = gd["spatial_ref"]
    for k, (a, name, units) in fields.items():
        ds[k] = xr.DataArray(a, dims=("y", "x"), attrs={"long_name": name, "units": units})
    ds.attrs.update(checkpoint=str(checkpoint), smb_model=config.smb_model,
                    enthalpy_seed=int(config.enthalpy_seed))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    print(f"wrote {out_path}")
    del problem
    torch.cuda.empty_cache()


# ------------------------------------------------------------------ helpers
def crop_to_factor(ds: xr.Dataset, factor: int) -> xr.Dataset:
    """Same centred crop GlacierProblem applies so ny, nx divide 2^n_levels."""
    ny0, nx0 = ds.sizes["y"], ds.sizes["x"]
    ny, nx = (ny0 // factor) * factor, (nx0 // factor) * factor
    y0, x0 = (ny0 - ny) // 2, (nx0 - nx) // 2
    return ds.isel(y=slice(y0, y0 + ny), x=slice(x0, x0 + nx))


def restrict(a: cp.ndarray, n_times: int) -> cp.ndarray:
    """2x2 block mean, `n_times` times (the inverse's avg-pool restriction)."""
    for _ in range(n_times):
        ny, nx = a.shape
        a = a.reshape(ny // 2, 2, nx // 2, 2).mean(axis=(1, 3))
    return a


def year_overlap_weights(t0: float, t1: float, eps: float = 1e-9) -> list:
    """Fractional overlap of (t0, t1] with each calendar year, weights sum to 1."""
    out, y = [], math.floor(t0 + eps)
    while y < t1 - eps:
        w = min(t1, y + 1.0) - max(t0, y)
        if w > eps:
            out.append((int(y), w))
        y += 1.0
    total = sum(w for _, w in out)
    return [(y, w / total) for y, w in out]


def load_ocean_forcing() -> Optional[OceanForcing]:
    global OCEAN
    rel = getattr(OCEAN, "pin_release_year", None)
    if OCEAN.enabled and rel is not None and (OCEAN.pin_front or getattr(OCEAN, "pin_front_filename", None)):
        # a pin until the release year, the free TF-driven law after (ocean.ReleasedPin)
        from glacier_inverse.ocean import ReleasedPin
        full = OCEAN
        try:
            OCEAN = dataclasses.replace(full, pin_release_year=None)
            pin = load_ocean_forcing()
            OCEAN = dataclasses.replace(full, pin_release_year=None, pin_front=None, pin_front_filename=None)
            free = load_ocean_forcing()
        finally:
            OCEAN = full
        of = ReleasedPin(pin, free, rel)
        print(f"  -> released at {rel:g}")
        return of
    if not OCEAN.enabled:
        print(f"ocean forcing disabled: constant margins q = {Q0:g}, h0 = {H00:g} m")
        return None
    if getattr(OCEAN, "pin_front_filename", None):
        # the TIME-VARYING pin (yearly TermPicks masks, preprocessing/make_front_mask.py)
        from glacier_inverse.ocean import PinnedFront
        of = PinnedFront.from_file(Path(config.base_dir) / "model_inputs" / OCEAN.pin_front_filename,
                                   2 ** config.n_levels, OCEAN, q0=Q0, h00=H00)
        print(of.describe())
        return of
    if OCEAN.pin_front:
        # the same pin the inversion ran with (ocean.PinnedFront); the mask is
        # read from the gridded inputs on the model crop, no TF file
        from glacier_inverse.ocean import PinnedFront
        with xr.open_dataset(f"{config.base_dir}/model_inputs/{config.gridded_filename}") as f:
            gd = crop_to_factor(f[[OCEAN.pin_front]].load(), 2 ** config.n_levels)
        of = PinnedFront.from_gridded(gd, OCEAN, q0=Q0, h00=H00)
        print(of.describe())
        return of
    if not THERMAL_PATH.exists():
        print(f"no thermal forcing at {THERMAL_PATH}: constant margins q = {Q0:g}, h0 = {H00:g} m")
        return None
    of = OceanForcing.from_file(THERMAL_PATH, 2 ** config.n_levels, OCEAN, q0=Q0, h00=H00,
                               freeze_anomaly=getattr(config, "climatology_only", False))
    print(of.describe())
    return of


def ocean_forcing(t: float, dt: float, mg, level: int, ctx: "Run") -> None:
    """Per-step hook, called before the dynamics step: sets the calving
    margins q / h0 for the step (t, t + dt] from the thermal forcing (see
    glacier_inverse/ocean.py) on the run level and its coarse levels. Other
    experiments go here too: mg.calving.timescale.set(...), or edits of
    mg[level].geometry.bed / .depth / mg[level].state.H at retreating fronts
    (pushed down with .set(..., start_level=level))."""
    if ctx.ocean is None:
        return
    q, h0, dtf = ctx.ocean.margins(t, t + dt)
    mg.calving.q.set(restrict(cp.asarray(q), level), start_level=level)
    mg.calving.h0.set(restrict(cp.asarray(h0), level), start_level=level)
    ctx.tf_anom.data[:, :] = restrict(cp.asarray(dtf), level)
    ok = ctx.ocean.ok
    print(f"  ocean forcing: dTF over active cells mean {dtf[ok].mean():+.2f} degC "
          f"(max {dtf[ok].max():+.2f}); q in [{q.min():.2f}, {q.max():.2f}], "
          f"h0 in [{h0.min():.0f}, {h0.max():.0f}] m")


# -------------------------------------------------------------------- setup
class Run:
    """Everything the time loop needs; built by setup(), stepped by run()."""


def setup(level: int = None, out_dir=None, ocean_loader=None) -> Run:
    """Build the model, the SMB model and the writers. `level`, `out_dir`
    default to the module constants; `ocean_loader` (-> OceanForcing or
    None) defaults to load_ocean_forcing (the config's EN4 record). The
    SMB internals are exposed on the returned Run (smb_model, smb_grid,
    temp_dev, domain_mask, t2m_clim, precip_clim, log_pbias, tbias) so a
    driver with another climate forcing (forward_projection.py) can swap
    compute_smb without rebuilding the model."""
    ctx = Run()
    level = LEVEL if level is None else int(level)
    out_dir = Path(OUT_DIR if out_dir is None else out_dir)
    ctx.level, ctx.out_dir = level, out_dir
    ### Load data: gridded inputs (cropped exactly as the inverse), the
    ### calibrated fields and the anomaly record
    with xr.open_dataset(f"{config.base_dir}/model_inputs/{config.gridded_filename}") as f:
        gd = crop_to_factor(f.load(), 2 ** config.n_levels)
    phys = xr.load_dataset(PHYSICAL_PATH)
    assert phys.sizes["y"] == gd.sizes["y"] and phys.sizes["x"] == gd.sizes["x"]
    anom = xr.load_dataset(f"{config.base_dir}/model_inputs/{config.anomaly_filename}")
    t_anom = {int(y): float(v) for y, v in zip(anom.time.values, anom.temp_anomaly.values)}
    year_max = max(t_anom)
    base_anomaly = (0.0 if config.base_anomaly_year is None
                    else t_anom[int(config.base_anomaly_year)])
    # The inverse's year-by-year forcing over the reanalysis record
    # (config.yearly_climate_filename; None or a missing file = the index
    # everywhere). Same loader as the inverse; read from the file per call
    # (no host cache: the driver evaluates each year once).
    # config.climatology_only: reference climate for the whole run (the same
    # switch the inverse honours) - no yearly fields, no anomaly index, so
    # the run measures the model's own relaxation from the initial geometry.
    climatology = getattr(config, "climatology_only", False)
    alpha_t2m_eff = 0.0 if climatology else config.alpha_t2m
    base_anomaly_eff = 0.0 if climatology else base_anomaly
    yearly = None
    if config.yearly_climate_filename is not None and not climatology:
        yc_path = Path(config.base_dir) / "model_inputs" / config.yearly_climate_filename
        if yc_path.exists():
            from glacier_inverse.yearly_climate import YearlyClimate
            yearly = YearlyClimate.from_file(yc_path, 2 ** config.n_levels, cache="none")
            print(yearly.describe())
        else:
            print(f"yearly climate {yc_path} missing: climatology + index anomaly everywhere")
    # The index precipitation multiplier (config.precip_anomaly_filename,
    # preprocessing/make_precip_anomaly.py), exactly as forward.simulate
    # applies it: the weighted mean over the step's INDEX years divided by the
    # base year, INDEX YEARS ONLY - record years already carry their own
    # precip_ratio. Without this a replay would run wetter than the inversion
    # everywhere before the reanalysis record. climatology_only zeroes it, the
    # same way problem.py passes precip_anomaly=None there.
    p_anom = base_precip = p_year_max = None
    alpha_precip_eff = 0.0 if climatology else float(getattr(config, "alpha_precip", 0.0))
    pa_path = Path(config.base_dir) / "model_inputs" / config.precip_anomaly_filename
    if pa_path.exists() and alpha_precip_eff != 0.0:
        pa = xr.load_dataset(pa_path)
        p_anom = {int(y): float(v) for y, v in zip(pa.time.values, pa.precip_anomaly.values)}
        p_year_max = max(p_anom)
        base_precip = p_anom[int(config.base_precip_year)]
        print(f"precip anomaly {pa_path.name}: alpha_precip {alpha_precip_eff:g}, "
              f"base {config.base_precip_year} = {base_precip:.6f}, "
              f"{min(p_anom)}-{p_year_max}")
    # Interannual-variance quadrature over the index years, exactly as
    # forward.simulate applies it (the same nodes, and the same rule that it
    # never touches reanalysis years). Not gated on climatology_only: the
    # variance belongs to the climate, not to the anomaly.
    from glacier_inverse.forward import _hermite_nodes
    _sig = getattr(config, "interannual_sigma", None)
    if _sig:
        _x, _w = _hermite_nodes(int(getattr(config, "interannual_nodes", 3)))
        nodes = (tuple(_sig * xi for xi in _x), _w)
        print(f"interannual quadrature: sigma {_sig:g} K, {len(_x)} nodes "
              f"at {'/'.join(f'{_sig * xi:+.2f}' for xi in _x)} K")
    else:
        nodes = ((0.0,), (1.0,))
    if climatology:
        print("climatology mode: reference climate for every step (no yearly fields, no anomaly index)")

    ny, nx = gd.sizes["y"], gd.sizes["x"]
    dx = float(gd.x[1] - gd.x[0])
    x0, y0 = float(gd.x[0]), float(gd.y[0])
    crs = pyproj.CRS(gd.spatial_ref.crs_wkt)
    f32 = lambda a: cp.asarray(np.asarray(a), dtype=cp.float32)

    ### Ice dynamics (as in glide's example, constants from the domain config)
    model = IceDynamics(n_levels=config.n_levels, ny=ny, nx=nx, dx=dx,
                        x0=x0, y0=y0, crs=crs, stress_scheme=config.stress_scheme)
    mg = model.mg

    bed = f32(phys.bed.values)
    mg.geometry.thklim.set(config.thklim)
    mg.geometry.bed.set(bed)
    mg.geometry.depth.set(-bed)                   # signed: negative on dry land
    mg.geometry.sigmoid_c.set(config.sigmoid_c)

    mg.rheology.B.set(config.B_rate)
    mg.rheology.eps_reg.set(config.eps_reg)
    mg.rheology.n.set(float(config.n_glen))
    mg.rheology.H_reg.set(float(config.H_reg))

    log_beta = cp.log(cp.minimum(f32(phys.beta.values), cp.float32(BETA_MAX)))
    mg.sliding.m.set(config.sliding_m)
    mg.sliding.u_reg.set(config.u_reg)
    mg.sliding.water_drag.set(config.water_drag)
    mg.sliding.u0.set(float(getattr(config, "sliding_u0", 0.0)))
    if getattr(config, "sliding_N_scale_H", None):  # dimensional N (see GlacierConfig.sliding_N_scale_H)
        mg.sliding.N_scale_H.set(float(config.sliding_N_scale_H))
        mg.sliding.N_floor_H.set(float(config.sliding_N_floor_H))

    ### Calving: height-above-buoyancy sink, ice with H < (1 + q) H_f decays
    ### at H / timescale per year (cp.inf disables it)
    mg.calving.timescale.set(config.calving_timescale)
    mg.calving.q.set(Q0)
    mg.calving.h0.set(H00)
    mg.calving.H_c.set(config.calving_H_c)

    ### Multigrid solver parameters (config.forward_solver = the example's)
    fs = config.forward_solver
    model.forward_solver.fas_options.set(
        coarsest_steps=fs.coarsest_steps, pre_steps=fs.pre_steps,
        post_steps=fs.post_steps, finest_steps=fs.finest_steps,
        relative_tolerance=fs.relative_tolerance,
        absolute_tolerance=fs.absolute_tolerance,
        report_norms=True,
        freeze_coarse_calving=getattr(fs, "freeze_coarse_calving", True),
        freeze_coarse_phi=getattr(fs, "freeze_coarse_phi", True),
        trace_file=getattr(fs, "trace_file", None), trace_every=getattr(fs, "trace_every", 25),
        dump_dir=getattr(fs, "dump_dir", None), dump_max=getattr(fs, "dump_max", 5),
        backtrack=getattr(fs, "backtrack", False),
        backtrack_scales=tuple(getattr(fs, "backtrack_scales", (1.0, 0.5, 0.25, 0.0))),
        raise_on_nonfinite=getattr(fs, "raise_on_nonfinite", True),
        cold_start_dt=getattr(fs, "cold_start_dt", 1.0))
    model.forward_solver.vanka_options.omega.set(cp.float32(fs.omega))
    model.forward_solver.vanka_options.newton_options.momentum_damping.set(cp.float32(fs.momentum_damping))
    model.forward_solver.vanka_options.newton_options.step_tolerance.set(cp.float32(fs.step_tolerance))

    ### SMB: glare's enthalpy model on the fine grid, calibrated parameters
    if config.smb_model != "enthalpy":
        raise NotImplementedError("this driver mirrors the enthalpy SMB path only")
    spy = SECONDS_PER_YEAR
    smb_model = EnthalpyModel(ny=ny, nx=nx, nt=12, dx=dx, dt=1.0 / 12, x0=x0, y0=y0,
                              crs=crs, n_substeps=config.enthalpy_n_substeps,
                              materialize_state=config.enthalpy_materialize_state)
    g = smb_model.grid
    t2m = f32(gd.monthly_t2m.values)                                   # (12, ny, nx) degC
    log_pbias = f32(phys.log_pbias.values)
    precip = f32(gd.monthly_precip.values) * cp.exp(log_pbias)
    tbias = f32(phys.tbias.values)
    g.precipitation.precip.set(precip)
    g.radiation.insol_mean.set(f32(gd.monthly_solar_potential_mean.values))
    g.radiation.insol_dif.set(f32(gd.monthly_diffuse_potential.values)
                              if "monthly_diffuse_potential" in gd else 0.0)
    g.geometry.srf.set(f32(gd.elevation.values))
    g.geometry.t_base.set(cp.minimum(t2m.mean(axis=0), 0.0))     # not anomaly-shifted
    g.geometry.debris.set(1.0)
    f_clear = f32(phys.f_clear.values)
    g.thermodynamics.H_atm.set(f32(phys.H_atm.values) * spy)
    g.thermodynamics.H_base0.set(cp.float32(config.H_base0 * spy))
    g.radiation.q_sw_bulk.set(cp.float32(config.q_sw_bulk * spy))
    g.radiation.q_sw_insol.set(f_clear * config.q_sw_clear * spy)
    g.radiation.q_sw_dif.set((f_clear * config.k_diffuse_clear
                              + (1.0 - f_clear) * config.k_diffuse_cloud) * config.q_sw_clear * spy)
    g.radiation.q_lw0.set(cp.float32(config.q_lw0 * spy))
    g.radiation.albedo_snow.set(cp.float32(config.albedo_snow))
    g.radiation.albedo_ice.set(cp.float32(config.albedo_ice))
    g.radiation.M_albedo.set(cp.float32(config.M_albedo))
    # the inverse's fixed weather realization (same seed -> same forward)
    sigma_t2m = float(cp.asnumpy(g.temperature.sigma_t2m.value))
    temp_dev = generate_temp_deviations(12, config.enthalpy_n_substeps, sigma_t2m,
                                        np.random.default_rng(config.enthalpy_seed))
    domain_mask = cp.asarray(gd.domain_mask.values, dtype=bool)

    def smb_index(shift, precip_multiplier: float = 1.0) -> cp.ndarray:
        g.temperature.t2m.set(t2m + shift)
        g.precipitation.precip.set(precip if precip_multiplier == 1.0
                                   else precip * precip_multiplier)
        smb_model.forward(temp_deviations=temp_dev)
        return g.state.smb.data.mean(axis=0)

    def smb_year(year: int) -> cp.ndarray:
        # the reanalysis year + the calibrated biases (forward.YearField)
        g.temperature.t2m.set(t2m + cp.asarray(yearly.t2m_anomaly(year)) + tbias)
        g.precipitation.precip.set(precip * cp.asarray(yearly.precip_ratio(year)))
        smb_model.forward(temp_deviations=temp_dev)
        return g.state.smb.data.mean(axis=0)

    def compute_smb(t_prev: float, t_next: float) -> cp.ndarray:
        """Annual-mean SMB on the fine grid for the step (t_prev, t_next],
        as the inverse builds it (forward.simulate, 'mean_anomaly'): one
        evaluation per reanalysis year the step overlaps (the year's fields
        + biases) and one at the interval-mean index anomaly over the
        remaining years; -10 m/yr outside the domain."""
        weights = year_overlap_weights(t_prev, t_next)
        on_record = [(y, w) for y, w in weights if yearly is not None and yearly.has(y)]
        index = [(y, w) for y, w in weights if not (yearly is not None and yearly.has(y))]
        smb = cp.zeros_like(t2m[0])
        for y, w in on_record:
            smb += w * smb_year(y)
        if index:
            w_index = sum(w for _, w in index)
            a = sum(w * t_anom[min(y, year_max)] for y, w in index) / w_index
            mult = 1.0
            if p_anom is not None:
                p_ratio = sum(w * p_anom[min(y, p_year_max)]
                              for y, w in index) / w_index / base_precip
                mult = 1.0 + alpha_precip_eff * (p_ratio - 1.0)
            shift = alpha_t2m_eff * a - base_anomaly_eff
            # The interannual-variance quadrature (forward._expand_interannual).
            # One scalar term here, as in the inverse's "mean_anomaly", so it
            # carries no spread of its own and takes the full sigma.
            for xi, wi in zip(*nodes):
                smb += w_index * wi * smb_index(shift + xi + tbias, mult)
        smb[~domain_mask] = -10.0
        return smb

    ctx.smb_model, ctx.smb_grid, ctx.temp_dev, ctx.domain_mask = smb_model, g, temp_dev, domain_mask
    ctx.t2m_clim, ctx.precip_clim, ctx.log_pbias, ctx.tbias = t2m, precip, log_pbias, tbias

    ### Initial state on the run level: thickness from the observed surface
    ### and the calibrated bed (init_from_observed_geometry), restricted like
    ### the inverse; bed / beta restricted the same way
    model.set_top_level(level)
    H = restrict(f32(phys.H_init.values), level)
    mg.state.H.set(H, start_level=level)
    mg.state.H_prev.set(H, start_level=level)
    mg.geometry.bed.set(restrict(bed, level), start_level=level)
    mg.sliding.beta.set(cp.exp(restrict(log_beta, level)), start_level=level)
    for f in (mg.state.u, mg.state.v, mg.state.ud, mg.state.vd, mg.state.mask):
        f.set(0.0, start_level=level)

    ### Writers (glide-example style) on the run level
    lvl = mg[level]
    n_glen = float(config.n_glen)
    srf = Field(data=cp.zeros((lvl.ny, lvl.nx), dtype=cp.float32), grid_entity=GridEntity.CELL,
                dx=lvl.dx, grid=lvl, name="srf", units="m", attrs={"long_name": "Surface Elevation"})
    u_s = Field(data=cp.zeros((lvl.ny, lvl.nx + 1), dtype=cp.float32), grid_entity=GridEntity.VERTICAL_FACET,
                dx=lvl.dx, grid=lvl, name="u_s", units="m a^{-1}", attrs={"long_name": "Surface velocity (x)"})
    v_s = Field(data=cp.zeros((lvl.ny + 1, lvl.nx), dtype=cp.float32), grid_entity=GridEntity.HORIZONTAL_FACET,
                dx=lvl.dx, grid=lvl, name="v_s", units="m a^{-1}", attrs={"long_name": "Surface velocity (y)"})
    u_b = Field(data=cp.zeros((lvl.ny, lvl.nx + 1), dtype=cp.float32), grid_entity=GridEntity.VERTICAL_FACET,
                dx=lvl.dx, grid=lvl, name="u_b", units="m a^{-1}", attrs={"long_name": "Basal velocity (x)"})
    v_b = Field(data=cp.zeros((lvl.ny + 1, lvl.nx), dtype=cp.float32), grid_entity=GridEntity.HORIZONTAL_FACET,
                dx=lvl.dx, grid=lvl, name="v_b", units="m a^{-1}", attrs={"long_name": "Basal velocity (y)"})
    dhdt = Field(data=cp.zeros((lvl.ny, lvl.nx), dtype=cp.float32), grid_entity=GridEntity.CELL,
                 dx=lvl.dx, grid=lvl, name="dhdt", units="m a^{-1}", attrs={"long_name": "Thickness change rate"})
    tf_anom = Field(data=cp.zeros((lvl.ny, lvl.nx), dtype=cp.float32), grid_entity=GridEntity.CELL,
                    dx=lvl.dx, grid=lvl, name="tf_anom", units="degC",
                    attrs={"long_name": "Ocean thermal forcing anomaly driving q (0 where inactive)"})
    ctx.ocean = (load_ocean_forcing if ocean_loader is None else ocean_loader)()

    ### Thermomechanical coupling (config.thermal; library change 18): the
    ### same ThermalDriver the inverse uses, spun up in run() before the loop
    ctx.thermal = None
    tcfg = getattr(config, "thermal", None)
    if tcfg is not None:
        from glacier_inverse.thermal import ThermalDriver, surface_temperature_fine
        ctx.thermal = ThermalDriver(model, level, tcfg, rho_i=float(config.rho_ice),
                                    thin_B=float(config.B_rate) if tcfg.thin_ice_isothermal else None)
        ctx.thermal_T_surface = surface_temperature_fine(t2m, tbias if tcfg.surface_T_tbias else None)
        print(f"thermal coupling: {tcfg}")

    rho_ratio = config.rho_ice / config.rho_water

    def update_derived(dt_step):
        # MOLHO: surface velocity u + ud/(n+1), basal velocity u - ud
        u_s.data[:, :] = lvl.state.u.data + lvl.state.ud.data / (n_glen + 1.0)
        v_s.data[:, :] = lvl.state.v.data + lvl.state.vd.data / (n_glen + 1.0)
        u_b.data[:, :] = lvl.state.u.data - lvl.state.ud.data
        v_b.data[:, :] = lvl.state.v.data - lvl.state.vd.data
        srf.data[:, :] = cp.maximum(lvl.state.H.data + lvl.geometry.bed.data,
                                    (1.0 - rho_ratio) * lvl.state.H.data)
        dhdt.data[:, :] = (lvl.state.H.data - lvl.state.H_prev.data) / dt_step

    vti_dir = out_dir / "vti"
    vti_dir.mkdir(parents=True, exist_ok=True)
    vti_writer = VTIWriter(str(vti_dir), base=config.vti_base_name, dx=lvl.dx,
                           compressor=VTI_COMPRESSOR, precision=VTI_PRECISION,
                           mask_field="mask", masked_fields=VTI_MASKED_FIELDS,
                           static_fields={"bed": lvl.geometry.bed, "beta": lvl.sliding.beta},
                           dynamic_fields={k: v for k, v in
                                           {"H": lvl.state.H, "srf": srf, "dhdt": dhdt,
                                            "U": [lvl.state.u, lvl.state.v], "U_s": [u_s, v_s],
                                            "U_b": [u_b, v_b], "smb": lvl.forcing.smb,
                                            "mask": lvl.state.mask, "xi": lvl.state.xi,
                                            "phi": lvl.state.phi, "psi": lvl.state.psi,
                                            "q": lvl.calving.q, "h0": lvl.calving.h0, "tf_anom": tf_anom,
                                            **({"T_bed": lambda: ctx.thermal.fields()["T_bed"],
                                                "T_mean": lambda: ctx.thermal.fields()["T_mean"],
                                                "omega_w_bed": lambda: ctx.thermal.fields()["omega_w_bed"],
                                                "B": lvl.rheology.B} if ctx.thermal is not None else {})}.items()
                                           if VTI_FIELDS is None or k in VTI_FIELDS})
    vti_writer.initialize(lvl)

    ctx.model, ctx.mg, ctx.lvl, ctx.gd, ctx.crs = model, mg, lvl, gd, crs
    ctx.compute_smb, ctx.update_derived, ctx.vti_writer, ctx.vti_dir = compute_smb, update_derived, vti_writer, vti_dir
    ctx.srf, ctx.u_s, ctx.v_s, ctx.dhdt, ctx.tf_anom = srf, u_s, v_s, dhdt, tf_anom
    return ctx


def dynamics_step(ctx: Run, t_prev: float, dt_step: float) -> None:
    """One glide step on the run level with the state handed over exactly as
    the inverse's GlideStep does: H_prev <- H, H <- H_prev, bed / beta reset
    from the run level down so every coarse level of the hierarchy is fresh."""
    mg, lvl, level = ctx.mg, ctx.lvl, ctx.level
    H_prev = lvl.state.H.data.copy()
    mg.state.H_prev.set(H_prev, start_level=level)
    mg.state.H.set(H_prev, start_level=level)
    mg.geometry.bed.set(lvl.geometry.bed.data.copy(), start_level=level)
    mg.sliding.beta.set(lvl.sliding.beta.data.copy(), start_level=level)
    if RESET_VELOCITY:
        for f in (mg.state.u, mg.state.v, mg.state.ud, mg.state.vd):
            f.set(0.0, start_level=level)
    ctx.model.forward(cp.float32(t_prev), cp.float32(dt_step), update_geometry=False)


def save_state(ctx: Run, t: float) -> Path:
    """The model state on the run level at a step end, RAW (nothing masked
    or rounded): H, surface, the staggered surface velocity u_s (ny, nx+1) /
    v_s (ny+1, nx), the step's SMB, the active-set mask and phi. A restart
    point for branches that share the spin-up, and exactly what the
    inversion's observation terms consume (analysis/sweep_calving_eval.py
    rebuilds a ModelState from it), which the VTI frames are not: they mask
    SMB and velocity off the ice and round."""
    lvl, gd, level = ctx.lvl, ctx.gd, ctx.level
    ny, nx = gd.sizes["y"], gd.sizes["x"]
    f32 = lambda a: cp.asarray(np.asarray(a), dtype=cp.float32)
    yc = restrict(f32(np.broadcast_to(gd.y.values[:, None], (ny, nx))), level)[:, 0]
    xc = restrict(f32(np.broadcast_to(gd.x.values[None, :], (ny, nx))), level)[0, :]
    out = xr.Dataset(coords={"y": cp.asnumpy(yc), "x": cp.asnumpy(xc)})
    enc = {}
    for name, arr, dims in [("H", lvl.state.H.data, ("y", "x")), ("srf", ctx.srf.data, ("y", "x")),
                            ("u_s", ctx.u_s.data, ("y", "xs")), ("v_s", ctx.v_s.data, ("ys", "x")),
                            ("smb", lvl.forcing.smb.data, ("y", "x")), ("mask", lvl.state.mask.data, ("y", "x")),
                            ("phi", lvl.state.phi.data, ("y", "x"))]:
        out[name] = xr.DataArray(cp.asnumpy(arr), dims=dims)
        enc[name] = dict(zlib=True, complevel=4)
    out.attrs.update(time=float(t), level=level, n_glen=float(config.n_glen),
                     note="u_s / v_s are the STAGGERED surface velocities (u + ud/(n+1)); smb unmasked; mask = active set")
    path = ctx.out_dir / f"state_{t:g}.nc"
    out.to_netcdf(path, encoding=enc)
    print(f"saved state at t = {t:g} to {path}", flush=True)
    return path


def run(ctx: Run) -> None:
    model, mg, lvl, gd, crs = ctx.model, ctx.mg, ctx.lvl, ctx.gd, ctx.crs
    level, out_dir = ctx.level, ctx.out_dir
    ny, nx = gd.sizes["y"], gd.sizes["x"]
    f32 = lambda a: cp.asarray(np.asarray(a), dtype=cp.float32)
    srf, u_s, v_s, dhdt, vti_writer, vti_dir = ctx.srf, ctx.u_s, ctx.v_s, ctx.dhdt, ctx.vti_writer, ctx.vti_dir
    ### Time loop: the inverse's step design (uniform DT grid, refined by
    ### DT_SCHEDULE = config.dt_schedule unless overridden at the top,
    ### snapped onto SNAP_TIMES = observation epochs)
    seq = build_step_sequence(t_start=float(T_START), t_end=float(T_END), dt_max=float(DT),
                              required_times=[float(t) for t in SNAP_TIMES],
                              dt_schedule=DT_SCHEDULE)
    ends = [t for t, _ in seq]
    steps = list(zip([float(T_START)] + ends[:-1], ends))
    if ctx.thermal is not None:
        H0 = lvl.state.H.data.copy()

        def _momentum_solve():
            mg.forcing.smb.set(0.0, start_level=level)
            lvl.state.H.data[:] = H0
            dynamics_step(ctx, float(T_START), float(ctx.thermal.cfg.spinup_momentum_dt))
        ctx.thermal.spinup(H0=H0, momentum_solve=_momentum_solve, T_surface_fine=ctx.thermal_T_surface)
        lvl.state.H.data[:] = H0
    for t_prev, t_next in steps:
        dt_step = t_next - t_prev
        print(f"Solving forward problem at t={t_prev:.2f} with dt={dt_step:.2f}", flush=True)
        mg.forcing.smb.set(restrict(ctx.compute_smb(t_prev, t_next), level), start_level=level)
        ocean_forcing(t_prev, dt_step, mg, level, ctx)
        if ctx.thermal is not None:
            ctx.thermal.pre_step(lvl.state.H.data)
        dynamics_step(ctx, t_prev, dt_step)
        if ctx.thermal is not None:
            ctx.thermal.post_step(dt_step)
        ctx.update_derived(dt_step)
        if VTI_T_MIN is None or t_next >= float(VTI_T_MIN) - 1e-6:
            vti_writer.append(lvl, time=float(t_next))
            vti_writer.write_pvd()
        if any(abs(t_next - float(ts)) < 1e-6 for ts in STATE_SAVE_TIMES):
            save_state(ctx, t_next)

    ### Final state to NetCDF (cell-centred fields on the run level)
    yc = restrict(f32(np.broadcast_to(gd.y.values[:, None], (ny, nx))), level)[:, 0]
    xc = restrict(f32(np.broadcast_to(gd.x.values[None, :], (ny, nx))), level)[0, :]
    out = xr.Dataset(coords={"y": cp.asnumpy(yc), "x": cp.asnumpy(xc)})
    for name, arr in ([("H", lvl.state.H.data), ("srf", srf.data), ("dhdt", dhdt.data),
                      ("bed", lvl.geometry.bed.data), ("beta", lvl.sliding.beta.data),
                      ("smb", lvl.forcing.smb.data), ("mask", lvl.state.mask.data),
                      ("xi", lvl.state.xi.data), ("phi", lvl.state.phi.data), ("psi", lvl.state.psi.data),
                      ("q", lvl.calving.q.data), ("h0", lvl.calving.h0.data), ("tf_anom", ctx.tf_anom.data),
                      ("u_s", 0.5 * (u_s.data[:, 1:] + u_s.data[:, :-1])),
                      ("v_s", 0.5 * (v_s.data[1:, :] + v_s.data[:-1, :]))]
                      + (list(ctx.thermal.fields().items()) if ctx.thermal is not None else [])):
        out[name] = xr.DataArray(cp.asnumpy(arr), dims=("y", "x"))
    out.attrs.update(level=level, t_start=float(T_START), t_end=float(T_END), dt=float(DT),
                     dt_schedule=repr(DT_SCHEDULE), n_steps=len(steps),
                     calving_timescale=float(config.calving_timescale), calving_H_c=float(config.calving_H_c),
                     calving_q=float(Q0), calving_h0=float(H00),
                     checkpoint=str(CHECKPOINT), crs_wkt=crs.to_wkt(),
                     ocean_forcing=(ctx.ocean.describe() if ctx.ocean is not None
                                    else f"constant margins q = {Q0:g}, h0 = {H00:g} m"),
                     A_glen=float(config.A_glen),
                     sliding_u0=float(lvl.sliding.u0.value),
                     sliding_N_scale_H=float(lvl.sliding.N_scale_H.value),
                     sliding_N_floor_H=float(lvl.sliding.N_floor_H.value),
                     thermal=(repr(ctx.thermal.cfg) if ctx.thermal is not None else "none (isothermal A_glen)"),
                     thermal_spinup=(repr(ctx.thermal.spinup_info) if ctx.thermal is not None else ""))
    out.to_netcdf(out_dir / "forward_soln.nc")
    print(f"wrote {out_dir / 'forward_soln.nc'}; VTI series in {vti_dir}")


def main(export_only: bool = False) -> None:
    # re-export when the checkpoint is newer than the cached fields: an
    # inversion rerun into the same results_subdir otherwise replays the
    # PREVIOUS state without a word (2026-09-20)
    stale = (PHYSICAL_PATH.exists() and Path(CHECKPOINT).exists()
             and Path(CHECKPOINT).stat().st_mtime > PHYSICAL_PATH.stat().st_mtime)
    if stale:
        print(f"{PHYSICAL_PATH.name} is older than {CHECKPOINT}: re-exporting")
    if export_only or stale or not PHYSICAL_PATH.exists():
        export_physical_fields(CHECKPOINT, PHYSICAL_PATH)
        if export_only:
            return
    run(setup())


# stage-1 reference (2026-09-29): the calibrated state with the fronts pinned
# to the observed history, the source of truth the calving selection of
# stage 2 emulates (analysis/sweep_calving_eval.py --reference-run). Raw
# states at the epochs every selection term reads.
STAGE1_STATE_TIMES = (1990.0, 1993.0, 2008.0, 2015.0, 2018.0, 2019.0)
STAGE1_DEFAULT_PIN = "front_mask_lia.nc"


def configure_stage1(pin: Optional[str] = None, vti_from: float = 1980.0) -> Path:
    """Set the module up for the stage-1 reference replay: the front pinned
    for the whole run (the config's yearly pin, else `pin`, else
    front_mask_lia.nc), no release, no h0 field; raw states at
    STAGE1_STATE_TIMES; frames from `vti_from`; into
    {output_dir}/stage1_reference. Returns the output directory."""
    global OCEAN, OUT_DIR, STATE_SAVE_TIMES, VTI_T_MIN, SNAP_TIMES
    pin = pin or getattr(config.ocean_forcing, "pin_front_filename", None) or STAGE1_DEFAULT_PIN
    OCEAN = dataclasses.replace(config.ocean_forcing, enabled=True, pin_front_filename=pin,
                                pin_release_year=None, rho_filename=None)
    OUT_DIR = Path(f"{config.output_dir}/stage1_reference")
    STATE_SAVE_TIMES = tuple(sorted(set(tuple(STATE_SAVE_TIMES) + STAGE1_STATE_TIMES)))
    SNAP_TIMES = tuple(sorted(set(tuple(SNAP_TIMES) + STATE_SAVE_TIMES)))
    VTI_T_MIN = float(vti_from)
    print(f"STAGE-1 REFERENCE: front pinned to {pin} for the whole run, states at {list(STATE_SAVE_TIMES)}, "
          f"frames from {VTI_T_MIN:g}, into {OUT_DIR}")
    return OUT_DIR


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true", help="only (re)write physical_fields.nc")
    ap.add_argument("--stage1", action="store_true",
                    help="the stage-1 reference replay: pinned front, raw states at the observation epochs, "
                         "into {output_dir}/stage1_reference (the reference of sweep_calving_eval.py --reference-run)")
    ap.add_argument("--pin", default=None, help="with --stage1: the yearly front-mask file (default: the config's, else front_mask_lia.nc)")
    args = ap.parse_args()
    if args.stage1:
        configure_stage1(args.pin)
    main(export_only=args.export)
