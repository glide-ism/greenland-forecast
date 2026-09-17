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
T_START, T_END, DT = 1850, 2100, 1   # default: the config's window
SNAP_TIMES = ()                   # extra breakpoints (e.g. observation epochs) so the
                                  # step sequence matches the inverse's exactly
RESET_VELOCITY = False            # zero u, v, ud, vd before each momentum solve: with
                                  # the pre-refactor solver settings the warm start
                                  # made the FAS solve diverge on the second 1 km step
BETA_MAX = 20.0                   # cap on the basal traction coefficient (the example
                                  # clips its inverted beta at 20)
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
    if not OCEAN.enabled:
        print(f"ocean forcing disabled: constant margins q = {Q0:g}, h0 = {H00:g} m")
        return None
    if not THERMAL_PATH.exists():
        print(f"no thermal forcing at {THERMAL_PATH}: constant margins q = {Q0:g}, h0 = {H00:g} m")
        return None
    of = OceanForcing.from_file(THERMAL_PATH, 2 ** config.n_levels, OCEAN, q0=Q0, h00=H00)
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
        report_norms=True)
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

    def compute_smb(t_prev: float, t_next: float) -> cp.ndarray:
        """Annual-mean SMB on the fine grid for the step (t_prev, t_next]:
        t2m shifted by the interval-mean anomaly (config.anomaly_integration
        'mean_anomaly') plus the calibrated bias; -10 m/yr outside the domain."""
        a = sum(w * t_anom[min(y, year_max)] for y, w in year_overlap_weights(t_prev, t_next))
        shift = config.alpha_t2m * a - base_anomaly + tbias
        g.temperature.t2m.set(t2m + shift)
        smb_model.forward(temp_deviations=temp_dev)
        smb = g.state.smb.data.mean(axis=0)
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
                           static_fields={"bed": lvl.geometry.bed, "beta": lvl.sliding.beta},
                           dynamic_fields={"H": lvl.state.H, "srf": srf, "dhdt": dhdt,
                                           "U": [lvl.state.u, lvl.state.v], "U_s": [u_s, v_s],
                                           "U_b": [u_b, v_b], "smb": lvl.forcing.smb,
                                           "mask": lvl.state.mask, "xi": lvl.state.xi,
                                           "phi": lvl.state.phi, "psi": lvl.state.psi,
                                           "q": lvl.calving.q, "h0": lvl.calving.h0, "tf_anom": tf_anom})
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


def run(ctx: Run) -> None:
    model, mg, lvl, gd, crs = ctx.model, ctx.mg, ctx.lvl, ctx.gd, ctx.crs
    level, out_dir = ctx.level, ctx.out_dir
    ny, nx = gd.sizes["y"], gd.sizes["x"]
    f32 = lambda a: cp.asarray(np.asarray(a), dtype=cp.float32)
    srf, u_s, v_s, dhdt, vti_writer, vti_dir = ctx.srf, ctx.u_s, ctx.v_s, ctx.dhdt, ctx.vti_writer, ctx.vti_dir
    ### Time loop: the inverse's step design (uniform DT grid, refined by
    ### config.dt_schedule, snapped onto SNAP_TIMES = observation epochs)
    seq = build_step_sequence(t_start=float(T_START), t_end=float(T_END), dt_max=float(DT),
                              required_times=[float(t) for t in SNAP_TIMES],
                              dt_schedule=config.dt_schedule)
    ends = [t for t, _ in seq]
    steps = list(zip([float(T_START)] + ends[:-1], ends))
    for t_prev, t_next in steps:
        dt_step = t_next - t_prev
        print(f"Solving forward problem at t={t_prev:.2f} with dt={dt_step:.2f}", flush=True)
        mg.forcing.smb.set(restrict(ctx.compute_smb(t_prev, t_next), level), start_level=level)
        ocean_forcing(t_prev, dt_step, mg, level, ctx)
        dynamics_step(ctx, t_prev, dt_step)
        ctx.update_derived(dt_step)
        vti_writer.append(lvl, time=float(t_next))
        vti_writer.write_pvd()

    ### Final state to NetCDF (cell-centred fields on the run level)
    yc = restrict(f32(np.broadcast_to(gd.y.values[:, None], (ny, nx))), level)[:, 0]
    xc = restrict(f32(np.broadcast_to(gd.x.values[None, :], (ny, nx))), level)[0, :]
    out = xr.Dataset(coords={"y": cp.asnumpy(yc), "x": cp.asnumpy(xc)})
    for name, arr in [("H", lvl.state.H.data), ("srf", srf.data), ("dhdt", dhdt.data),
                      ("bed", lvl.geometry.bed.data), ("beta", lvl.sliding.beta.data),
                      ("smb", lvl.forcing.smb.data), ("mask", lvl.state.mask.data),
                      ("xi", lvl.state.xi.data), ("phi", lvl.state.phi.data), ("psi", lvl.state.psi.data),
                      ("q", lvl.calving.q.data), ("h0", lvl.calving.h0.data), ("tf_anom", ctx.tf_anom.data),
                      ("u_s", 0.5 * (u_s.data[:, 1:] + u_s.data[:, :-1])),
                      ("v_s", 0.5 * (v_s.data[1:, :] + v_s.data[:-1, :]))]:
        out[name] = xr.DataArray(cp.asnumpy(arr), dims=("y", "x"))
    out.attrs.update(level=level, t_start=float(T_START), t_end=float(T_END), dt=float(DT),
                     checkpoint=str(CHECKPOINT), crs_wkt=crs.to_wkt(),
                     ocean_forcing=(ctx.ocean.describe() if ctx.ocean is not None
                                    else f"constant margins q = {Q0:g}, h0 = {H00:g} m"))
    out.to_netcdf(out_dir / "forward_soln.nc")
    print(f"wrote {out_dir / 'forward_soln.nc'}; VTI series in {vti_dir}")


def main(export_only: bool = False) -> None:
    if export_only or not PHYSICAL_PATH.exists():
        export_physical_fields(CHECKPOINT, PHYSICAL_PATH)
        if export_only:
            return
    run(setup())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true", help="only (re)write physical_fields.nc")
    main(export_only=ap.parse_args().export)
