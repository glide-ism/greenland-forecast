"""
GlacierProblem: builds the full forward model (IceDynamics + SMB + priors +
observations + whitened parameter tensors) from a GlacierConfig.

A driver script constructs one GlacierProblem and then either:
  * optimizes its `params` against `loss(...)` (inverse, rto),
  * loads RTO samples into its `params` and computes `loss_volume_change(...)`
    for the sensitivity ensemble,
  * inspects its priors only (posterior — use GlacierPriors directly there).

Observations are `Observation` objects (see observations.py): each carries its
data, acquisition time(s), hyperparameters, and misfit. The problem collects
their `required_times` and records model-state snapshots at exactly those
times during every forward run, so each product is compared against the model
at its own epoch. `compute_loss` accepts a substitute observation list so RTO
can pass randomized copies without rebuilding the problem.
"""
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cupy as cp
import geopandas as gpd
import numpy as np
import pyproj
import torch
import xarray as xr

from glide.model import IceDynamics
from glare.model import ImprovedTemperatureIndex
from glare.avalanche import AvalancheOperator
from ggapp.torch import GGaPPMap, GGaPPWhiten

# The enthalpy SMB core is newer than some installed glare versions; import it
# defensively so temperature-index-only setups keep working. Selecting
# smb_model="enthalpy" against an old glare raises a clear error at build time.
try:
    from glare.enthalpy import (
        EnthalpyModel, SECONDS_PER_YEAR, generate_temp_deviations,
    )
except ImportError:
    EnthalpyModel = None
    SECONDS_PER_YEAR = 365 * 24 * 3600

# AvalancheStep (hoisted once-per-iteration redistribution) is newer than some
# installed glare versions; only needed when config.avalanche_hoisted.
try:
    from glare.torch import AvalancheStep
except ImportError:
    AvalancheStep = None

from .config import GlacierConfig, SolverConfig
from .forward import simulate, differentiable_restriction, differentiable_prolongation
from .loss import LossTerms, PriorMeans, compute_prior
from .observations import (
    DomainData, Observation, ObservationBuildContext, default_observation_specs,
    validate_noise_weights,
)
from .priors import GlacierPriors
from .scheduling import merge_times


class WhitenedParameters:
    """The tensors we optimize over: five whitened fields plus the whitened
    scalars (log melt/radiation factors, the precip-depletion tau/z0, and the
    enthalpy-model log H_atm / logit cloud factor). The depletion scalars are
    inert unless precip_lapse_enabled, and the z_tbias field is inert unless
    tbias_enabled; the mf/rf pair and the H_atm/cloud pair are mutually
    exclusive per config.smb_model (every inactive tensor stays at
    z = 0 = prior median and never enters the forward graph)."""

    def __init__(self, z_bed, z_bed_mean, z_log_beta, z_pbias, z_tbias,
                 z_log_mf, z_log_rf, z_tau, z_z0, z_log_H_atm, z_logit_cloud):
        self.z_bed = z_bed
        self.z_bed_mean = z_bed_mean
        self.z_log_beta = z_log_beta
        self.z_pbias = z_pbias
        self.z_tbias = z_tbias
        self.z_log_mf = z_log_mf
        self.z_log_rf = z_log_rf
        self.z_tau = z_tau
        self.z_z0 = z_z0
        self.z_log_H_atm = z_log_H_atm
        self.z_logit_cloud = z_logit_cloud

    def requires_grad_(self) -> "WhitenedParameters":
        for t in (self.z_bed, self.z_bed_mean, self.z_log_beta,
                  self.z_pbias, self.z_tbias, self.z_log_mf, self.z_log_rf,
                  self.z_tau, self.z_z0, self.z_log_H_atm, self.z_logit_cloud):
            t.requires_grad_()
        return self

    def detach_clone(self) -> "WhitenedParameters":
        return WhitenedParameters(
            z_bed=self.z_bed.detach().clone(),
            z_bed_mean=self.z_bed_mean.detach().clone(),
            z_log_beta=self.z_log_beta.detach().clone(),
            z_pbias=self.z_pbias.detach().clone(),
            z_tbias=self.z_tbias.detach().clone(),
            z_log_mf=self.z_log_mf.detach().clone(),
            z_log_rf=self.z_log_rf.detach().clone(),
            z_tau=self.z_tau.detach().clone(),
            z_z0=self.z_z0.detach().clone(),
            z_log_H_atm=self.z_log_H_atm.detach().clone(),
            z_logit_cloud=self.z_logit_cloud.detach().clone(),
        )

    @torch.no_grad()
    def copy_from(self, other: "WhitenedParameters") -> None:
        """In-place reset to `other`'s values. Tensor identities are preserved
        so any optimizer holding references to these tensors keeps working."""
        self.z_bed.copy_(other.z_bed)
        self.z_bed_mean.copy_(other.z_bed_mean)
        self.z_log_beta.copy_(other.z_log_beta)
        self.z_pbias.copy_(other.z_pbias)
        self.z_tbias.copy_(other.z_tbias)
        self.z_log_mf.copy_(other.z_log_mf)
        self.z_log_rf.copy_(other.z_log_rf)
        self.z_tau.copy_(other.z_tau)
        self.z_z0.copy_(other.z_z0)
        self.z_log_H_atm.copy_(other.z_log_H_atm)
        self.z_logit_cloud.copy_(other.z_logit_cloud)


def _apply_solver_settings(solver, cfg: SolverConfig) -> None:
    """Push a SolverConfig onto one of glide's FAS solvers (forward or
    adjoint): the V-cycle schedule and the Vanka/Newton smoother options."""
    solver.fas_options.set(
        coarsest_steps=cfg.coarsest_steps,
        pre_steps=cfg.pre_steps,
        post_steps=cfg.post_steps,
        finest_steps=cfg.finest_steps,
        relative_tolerance=cfg.relative_tolerance,
        absolute_tolerance=cfg.absolute_tolerance,
        report_norms=cfg.report_norms,
    )
    solver.vanka_options.omega.set(cp.float32(cfg.omega))
    solver.vanka_options.newton_options.momentum_damping.set(cp.float32(cfg.momentum_damping))
    solver.vanka_options.newton_options.step_tolerance.set(cp.float32(cfg.step_tolerance))
    if "freeze_coarse_calving" in getattr(solver.fas_options, "options", ()):   # forward solver only
        solver.fas_options.set(freeze_coarse_calving=cfg.freeze_coarse_calving,
                               freeze_coarse_phi=cfg.freeze_coarse_phi,
                               trace_file=cfg.trace_file, trace_every=cfg.trace_every,
                               dump_dir=cfg.dump_dir, dump_max=cfg.dump_max,
                               backtrack=cfg.backtrack, backtrack_scales=tuple(cfg.backtrack_scales))


def _crop_to_factor(gridded_data: xr.Dataset, factor: int) -> xr.Dataset:
    """Crop the gridded dataset so ny and nx are multiples of 2^(n_levels-?)."""
    ny0, nx0 = gridded_data.sizes["y"], gridded_data.sizes["x"]
    ny_target = (ny0 // factor) * factor
    nx_target = (nx0 // factor) * factor
    y_start = (ny0 - ny_target) // 2
    x_start = (nx0 - nx_target) // 2
    return gridded_data.isel(
        y=slice(y_start, y_start + ny_target),
        x=slice(x_start, x_start + nx_target),
    )


def _open_eager(path) -> xr.Dataset:
    """Read a netCDF fully into memory and CLOSE the file before returning.

    A long-lived lazy handle is a double hazard in this environment: (a)
    reopening the same file later in the process, after an earlier handle
    was garbage-collected, corrupts HDF5 state ("NetCDF: HDF error" — this
    was the smoke test's perpetual section-8 failure when it built a second
    GlacierProblem); (b) an open handle in an interactive session makes any
    in-place rewrite of the file truncate it to 0 bytes before the HDF5 lock
    check fails. Everything downstream works from the in-memory Dataset;
    the largest domain file is ~1 GB, so the host-RAM cost is modest.
    """
    with xr.open_dataset(path) as f:
        return f.load()


class GlacierProblem:
    """Full forward model + observations + whitened parameters."""

    def __init__(self, config: GlacierConfig):
        self.config = config
        cfg = config

        inputs_dir = Path(cfg.base_dir) / "model_inputs"
        self.gridded_data = _open_eager(inputs_dir / cfg.gridded_filename)
        self.temperature_anomaly = _open_eager(inputs_dir / cfg.anomaly_filename)
        # Radar bed picks are optional: Greenland domains condition the bed
        # on the gridded BedMachine product instead (bed_obs / bed_obs_err in
        # GLIDE_inputs.nc, see priors.build_bed_conditioning_data), so a
        # missing flightline file just means "no scattered picks".
        flightline_path = inputs_dir / cfg.flightline_filename
        if flightline_path.exists():
            flightlines_df = gpd.read_file(flightline_path)
        else:
            flightlines_df = gpd.GeoDataFrame(
                {"x": [], "y": [], "bed": []}, geometry=[], crs=None)

        self.crs = pyproj.CRS(self.gridded_data.spatial_ref.crs_wkt)
        self.gridded_data = _crop_to_factor(self.gridded_data, 2 ** cfg.n_levels)

        # Optional end-of-summer snowline product (ELA proxy). Built on the
        # same DEM grid as GLIDE_inputs, so the identical centered crop keeps
        # it aligned. Domains without the file simply skip the snowline loss.
        snowline_path = inputs_dir / cfg.snowline_filename
        if snowline_path.exists():
            self.snowline_data = _crop_to_factor(
                _open_eager(snowline_path), 2 ** cfg.n_levels)
        else:
            self.snowline_data = None

        debris_path = inputs_dir / cfg.debris_filename
        if debris_path.exists():
            self.debris_data = _crop_to_factor(
                _open_eager(debris_path), 2 ** cfg.n_levels)
        else:
            self.debris_data = None

        # Optional surface elevation-change rate (Hugonnet dH/dt). Built on the
        # same DEM grid as GLIDE_inputs, so the identical centered crop keeps it
        # aligned. Domains without the file simply skip the dH/dt loss.
        dhdt_path = inputs_dir / cfg.dhdt_filename
        if dhdt_path.exists():
            self.dhdt_data = _crop_to_factor(
                _open_eager(dhdt_path), 2 ** cfg.n_levels)
        else:
            self.dhdt_data = None

        # Optional ocean thermal forcing of the calving margins (q, h0 per
        # step; see ocean.py). Loaded on the same crop; only the annual
        # statistic the config asks for is read.
        self.ocean_forcing = None
        if cfg.ocean_forcing.enabled and (cfg.ocean_forcing.pin_front or getattr(cfg.ocean_forcing, "pin_front_filename", None)):
            # front held at an observed mask: no TF file involved (ocean.py);
            # a yearly mask file (TermPicks) takes precedence over the static variable
            from .ocean import PinnedFront
            if getattr(cfg.ocean_forcing, "pin_front_filename", None):
                self.ocean_forcing = PinnedFront.from_file(
                    Path(cfg.base_dir) / "model_inputs" / cfg.ocean_forcing.pin_front_filename, 2 ** cfg.n_levels,
                    cfg.ocean_forcing, q0=cfg.calving_q, h00=cfg.calving_h0)
            else:
                self.ocean_forcing = PinnedFront.from_gridded(
                    self.gridded_data, cfg.ocean_forcing, q0=cfg.calving_q, h00=cfg.calving_h0)
            print(self.ocean_forcing.describe())
        elif cfg.ocean_forcing.enabled:
            tf_path = inputs_dir / cfg.ocean_forcing.filename
            if tf_path.exists():
                from .ocean import OceanForcing
                self.ocean_forcing = OceanForcing.from_file(
                    tf_path, 2 ** cfg.n_levels, cfg.ocean_forcing,
                    q0=cfg.calving_q, h00=cfg.calving_h0,
                    freeze_anomaly=cfg.climatology_only)
                print(self.ocean_forcing.describe())
            else:
                warnings.warn(f"ocean_forcing enabled but {tf_path} is missing: "
                              f"calving margins stay at calving_q / calving_h0")

        # Optional year-by-year forcing over a reanalysis record (see
        # yearly_climate.py): per-year t2m anomalies / precip ratios on the
        # model's climatology, the same crop, never on the tape.
        self.yearly_climate = None
        if cfg.yearly_climate_filename is not None and not cfg.climatology_only:
            if cfg.base_anomaly_year is not None:
                raise ValueError("yearly_climate_filename requires base_anomaly_year=None: "
                                 "the yearly fields are departures from the climatology window")
            yc_path = inputs_dir / cfg.yearly_climate_filename
            if yc_path.exists():
                from .yearly_climate import YearlyClimate
                self.yearly_climate = YearlyClimate.from_file(
                    yc_path, 2 ** cfg.n_levels, cache=cfg.yearly_climate_cache)
                print(self.yearly_climate.describe())
            else:
                warnings.warn(f"yearly_climate_filename set but {yc_path} is missing: "
                              f"every year uses the climatology + index anomaly")
        if cfg.climatology_only:
            print("climatology mode: the atmosphere is held at the reference climate "
                  "(monthly climatology + tbias / pbias; no yearly fields, no anomaly index) "
                  "and the ocean at its TF climatology (dTF = 0, time-invariant margins)")

        ny, nx = self.gridded_data.sizes["y"], self.gridded_data.sizes["x"]
        dx = (self.gridded_data.x[1] - self.gridded_data.x[0]).item()
        x0 = self.gridded_data.x[0].item()
        y0 = self.gridded_data.y[0].item()
        self.ny, self.nx, self.dx = ny, nx, dx

        self.model = self._build_ice_dynamics(ny, nx, dx, x0, y0)
        self.mg = self.model.mg

        # Set by _build_enthalpy_smb_model when config.avalanche_hoisted: the
        # redistribution operator applied once per simulate_physical call
        # instead of inside every per-step SMB evaluation (see the loud
        # discussion on GlacierConfig.avalanche_hoisted).
        self.avalanche_op = None
        self.smb_model = self._build_smb_model(ny, nx, dx, x0, y0)
        self.mg.forcing.smb.set(self.smb_model.grid.state.smb.data.mean(axis=0))

        self.priors = GlacierPriors(cfg, ny, nx, dx)
        self.domain = self._build_domain_data()
        build_ctx = ObservationBuildContext(
            gridded_data=self.gridded_data,
            snowline_data=self.snowline_data,
            dhdt_data=self.dhdt_data,
            flightlines_df=flightlines_df,
            domain=self.domain,
            ny=ny, nx=nx,
            config=cfg,
            priors=self.priors,
        )
        # Kept for ad-hoc spec builds (smoke test, tools) against this problem.
        self.build_ctx = build_ctx
        specs = cfg.observations if getattr(cfg, "observations", None) \
            else default_observation_specs(cfg)
        self.observations = [obs for spec in specs
                             if (obs := spec.build(build_ctx)) is not None]
        # Whitened (MaternNoise) terms carry weight == 1 by contract.
        validate_noise_weights(self.observations)

        # RTO hook: per-sample perturbed conditioning data (Matheron's rule)
        # consumed by physical_from until reset. Stale PCG warm-start caches
        # across a swap are just initial guesses — always correct.
        self._bed_data_override = None
        if self.priors.bed_conditioner is not None:
            bed_obs = next((o for o in self.observations if o.name == "bed"), None)
            if bed_obs is not None and \
                    bed_obs.weight_at(0, 0, schedule=False) != 0.0:
                warnings.warn(
                    "bed_conditioning is enabled but the BedObservation "
                    "likelihood has nonzero weight — the bed data is counted "
                    "twice (once in the conditioned prior map, once in the "
                    "soft loss). Set BedSpec(weight=0.0) in the domain "
                    "config; the observation is kept only for diagnostics.")

        self.params = self._build_initial_parameters()

        self.base_anomaly = (0.0 if cfg.base_anomaly_year is None else
                             self.temperature_anomaly.sel(
                                 time=cfg.base_anomaly_year).temp_anomaly.item())
        self.alpha_t2m = torch.tensor(cfg.alpha_t2m, device="cuda")

        # Optional ice-core precip anomaly: applied as a multiplicative
        # scaling on the precip field at each time step.
        precip_anomaly_path = inputs_dir / cfg.precip_anomaly_filename
        if precip_anomaly_path.exists():
            self.precip_anomaly = _open_eager(precip_anomaly_path)
            self.base_precip = self.precip_anomaly.sel(
                time=cfg.base_precip_year).precip_anomaly.item()
            self.alpha_precip = torch.tensor(cfg.alpha_precip, device="cuda")
        else:
            self.precip_anomaly = None
            self.base_precip = None
            self.alpha_precip = None

        # Cached fine-grid forcings/state used in every forward call.
        self.t2m = torch.tensor(self.smb_model.grid.temperature.t2m.data,
                                device="cuda")
        self.precip = torch.tensor(self.smb_model.grid.precipitation.precip.data,
                                   device="cuda")
        self.H_prev = torch.tensor(self.mg[0].state.H_prev.data,
                                   device="cuda")

        if cfg.smb_model == "enthalpy":
            self.debris = torch.tensor(
                self.smb_model.grid.geometry.debris.data, device="cuda")
            self.insol_mean = torch.tensor(
                self.smb_model.grid.radiation.insol_mean.data, device="cuda")
            self.insol_dif = torch.tensor(
                self.smb_model.grid.radiation.insol_dif.data, device="cuda")
            self.t_base = torch.tensor(
                self.smb_model.grid.geometry.t_base.data, device="cuda")
            # Fixed (non-inverted) enthalpy constants as 0-dim tensors —
            # EnthalpyStep consumes scalars via .item(); no gradients kept.
            spy = SECONDS_PER_YEAR

            def _const(v):
                return torch.tensor(float(v), dtype=torch.float32, device="cuda")
            self.enthalpy_consts = {
                "H_base0": _const(cfg.H_base0 * spy),
                "q_sw_bulk": _const(cfg.q_sw_bulk * spy),
                "q_lw0": _const(cfg.q_lw0 * spy),
                "albedo_snow": _const(cfg.albedo_snow),
                "albedo_ice": _const(cfg.albedo_ice),
                "M_albedo": _const(cfg.M_albedo),
            }
        else:
            self.debris = torch.tensor(
                self.smb_model.grid.geometry.debris.data, device="cuda")
            self.insol_mean = None
            self.insol_dif = None
            self.t_base = None
            self.enthalpy_consts = None
            self.temp_dev = None
    # ----------------------------------------------------------------- builders

    def _build_ice_dynamics(self, ny, nx, dx, x0, y0) -> IceDynamics:
        cfg = self.config
        model = IceDynamics(n_levels=cfg.n_levels, ny=ny, nx=nx, dx=dx,
                            x0=x0, y0=y0, crs=self.crs,
                            stress_scheme=cfg.stress_scheme)
        mg = model.mg

        mg.state.H.set(cfg.thklim)
        mg.state.H_prev.set(cfg.thklim)

        # Initial bed. Alaska: the DEM composite (the surface over ice — no
        # bed is known a priori, the likelihood has to find it). Greenland:
        # the gridded bed observation (BedMachine `bed_obs`) where it exists,
        # the DEM elsewhere. With the bed prior conditioned on that same
        # gridded data at every ice cell, the unconditional field Map(z_bed)
        # is barely touched by the likelihood, so whatever it is initialized
        # to is (up to the prior) what it stays: seeded from the surface it
        # would sit ~thickness above the bed forever, and bed_mean would
        # converge to a smoothed SURFACE (observed on greenland_coarse).
        self.bed_init = self._initial_bed()
        mg.geometry.thklim.set(cfg.thklim)
        mg.geometry.bed.set(self.bed_init)
        # signed head deficit (water level minus bed): negative on dry land,
        # so thin terrestrial ice is far from flotation, not half-floating
        mg.geometry.depth.set(-self.bed_init)
        mg.geometry.sigmoid_c.set(cfg.sigmoid_c)

        mg.rheology.B.set(cfg.B_rate)
        mg.rheology.eps_reg.set(cfg.eps_reg)
        mg.rheology.n.set(float(cfg.n_glen))
        mg.rheology.H_reg.set(float(cfg.H_reg))
        #mg.rheology.slope_sat.set(float(0.15))

        mg.sliding.beta.set(cfg.beta_init)
        mg.sliding.m.set(cfg.sliding_m)
        mg.sliding.u_reg.set(cfg.u_reg)
        mg.sliding.water_drag.set(cfg.water_drag)

        mg.calving.timescale.set(cfg.calving_timescale)
        mg.calving.q.set(cfg.calving_q)
        mg.calving.h0.set(cfg.calving_h0)
        mg.calving.H_c.set(cfg.calving_H_c)

        _apply_solver_settings(model.forward_solver, cfg.forward_solver)
        _apply_solver_settings(model.adjoint_solver, cfg.adjoint_solver)
        return model

    def _smoothed(self, field: np.ndarray) -> np.ndarray:
        """Gaussian smoothing at half the bed-mean prior's length scale —
        the hierarchical mean's starting point."""
        from cupyx.scipy.ndimage import gaussian_filter as _cp_gauss
        sigma_px = float(self.config.mean_prior.l) / (2.0 * float(self.dx))
        return cp.asnumpy(_cp_gauss(cp.asarray(field, dtype=cp.float32),
                                    sigma_px, mode="nearest"))

    def _initial_bed(self) -> np.ndarray:
        """(ny, nx) float32 initial bed: `bed_obs` where finite (gridded bed
        data, Greenland), the DEM composite elsewhere / when absent (Alaska
        — bit-identical to the historical initialization).

        Under bed_conditioning.gridded_bed_data == "radar" with
        seed_off_track_from_mean, the ice cells WITHOUT a radar pick are
        seeded from the smoothed bed instead of BedMachine's interpolated
        value, so the between-track structure starts from the prior mean."""
        gd = self.gridded_data
        elevation = gd.elevation.values.astype(np.float32)
        if "bed_obs" not in gd:
            return elevation
        bed_obs = gd.bed_obs.values.astype(np.float32)
        use = np.isfinite(bed_obs)
        bed = np.where(use, bed_obs, elevation).astype(np.float32)
        bc = self.config.bed_conditioning
        if (getattr(bc, "gridded_bed_data", "all") == "radar"
                and getattr(bc, "seed_off_track_from_mean", False)
                and "bed_obs_radar" in gd):
            radar = (gd.bed_radar_fraction.values > bc.radar_fraction_min) \
                & np.isfinite(gd.bed_obs_radar.values)
            smooth = self._smoothed(bed)
            bed = np.where(radar, gd.bed_obs_radar.values.astype(np.float32),
                           np.where(use, smooth, bed)).astype(np.float32)
            print(f"[init] bed seeded from radar picks on {int(radar.sum())} cells, "
                  f"the smoothed bed on {int((use & ~radar).sum())} off-track ice "
                  f"cells, DEM elsewhere")
        else:
            print(f"[init] bed seeded from bed_obs on {int(use.sum())} cells, "
                  f"DEM elsewhere")
        return bed

    def _build_smb_model(self, ny, nx, dx, x0, y0):
        cfg = self.config
        if cfg.smb_model == "enthalpy":
            return self._build_enthalpy_smb_model(ny, nx, dx, x0, y0)
        if cfg.smb_model != "temperature_index":
            raise ValueError(
                f"unknown smb_model {cfg.smb_model!r}; "
                "expected 'temperature_index' or 'enthalpy'")

        if cfg.avalanche_hoisted:
            raise ValueError(
                "avalanche_hoisted is only valid with smb_model='enthalpy': "
                "the ETIM snow/rain partition sits upstream of the avalanche "
                "operator and depends on the per-step (anomaly-shifted) t2m, "
                "so the redistribution input genuinely changes every step — "
                "see GlacierConfig.avalanche_hoisted.")

        smb = ImprovedTemperatureIndex(ny=ny, nx=nx, nt=12,
                                       dx=dx, dt=1.0 / 12,
                                       x0=x0, y0=y0, crs=self.crs)

        if cfg.use_avalanche_model:
            smb.avalanche = AvalancheOperator(
                smb.grid, s_crit=cfg.avalanche_s_crit,
                w_trans=cfg.avalanche_w_trans, p=cfg.avalanche_p,
                K=cfg.avalanche_K)

        gd = self.gridded_data
        smb.grid.insolation.insol_mean.set(gd.monthly_solar_potential_mean)
        smb.grid.insolation.insol_cos.set(gd.monthly_solar_potential_cos)
        smb.grid.insolation.insol_sin.set(gd.monthly_solar_potential_sin)
        smb.grid.temperature.t2m.set(gd.monthly_t2m)
        smb.grid.precipitation.precip.set(gd.monthly_precip)
        smb.grid.geometry.srf.set(gd.elevation)
        smb.grid.insolation.rf.set(cfg.mu_rf)
        smb.grid.temperature.mf.set(cfg.mu_mf)
        if self.debris_data is not None:
            smb.grid.geometry.debris.set(
                1.0 - cfg.debris_factor * self.debris_data.debris_fraction)
        else:
            smb.grid.geometry.debris.set(1.0)
        smb.forward()
        return smb

    def _build_enthalpy_smb_model(self, ny, nx, dx, x0, y0):
        """Build glare's EnthalpyModel, initialized at the prior-median
        parameters, with one fixed seeded weather realization (`self.temp_dev`)
        reused by every forward call so the checkpointed objective is
        deterministic."""
        cfg = self.config
        if EnthalpyModel is None:
            raise ImportError(
                "smb_model='enthalpy' requires a glare version providing "
                "glare.enthalpy.EnthalpyModel")
        import inspect
        kwargs = {}
        if "materialize_state" in inspect.signature(EnthalpyModel).parameters:
            # Older glare versions predate the flag; with a current one, the
            # six diagnostic state cubes share one scratch buffer unless the
            # domain asks for them (config.enthalpy_materialize_state).
            kwargs["materialize_state"] = cfg.enthalpy_materialize_state
        smb = EnthalpyModel(ny=ny, nx=nx, nt=12, dx=dx, dt=1.0 / 12,
                            x0=x0, y0=y0, crs=self.crs,
                            n_substeps=cfg.enthalpy_n_substeps, **kwargs)
        if cfg.use_avalanche_model:
            op = AvalancheOperator(
                smb.grid, s_crit=cfg.avalanche_s_crit,
                w_trans=cfg.avalanche_w_trans, p=cfg.avalanche_p,
                K=cfg.avalanche_K)
            if cfg.avalanche_hoisted:
                # Hoisted mode: R is applied once per simulate_physical call
                # (simulate_physical redistributes precip_ up front) and the
                # model itself runs bare — _prepare_effective_precip reduces
                # to a copy and the adjoint's R^T pull-back to identity. Only
                # valid because the enthalpy core takes total precip and the
                # per-step variation is a scalar multiplier — see the loud
                # note on GlacierConfig.avalanche_hoisted.
                if AvalancheStep is None:
                    raise ImportError(
                        "avalanche_hoisted=True requires a glare version "
                        "providing glare.torch.AvalancheStep")
                self.avalanche_op = op
            else:
                smb.avalanche = op

        gd = self.gridded_data
        smb.grid.temperature.t2m.set(gd.monthly_t2m)
        smb.grid.precipitation.precip.set(gd.monthly_precip)
        smb.grid.radiation.insol_mean.set(gd.monthly_solar_potential_mean)
        # Diffuse-sky potential (sky-view factor x mean cos zenith). Inputs
        # built before it existed load as zeros: no diffuse term, loudly.
        if "monthly_diffuse_potential" in gd:
            smb.grid.radiation.insol_dif.set(gd.monthly_diffuse_potential)
        else:
            warnings.warn(
                "GLIDE_inputs.nc has no 'monthly_diffuse_potential': the diffuse "
                "shortwave term is zero for this run. Regenerate with "
                "`python preprocessing/make_insolation.py --domain-path <domain> "
                "--year <year> --diffuse-only` then `python preprocessing/make_merged.py "
                "--domain-path <domain>`.")
            smb.grid.radiation.insol_dif.set(0.0)
        smb.grid.geometry.srf.set(gd.elevation)
        # Same debris attenuation as the ETIM branch: from glare's perspective
        # a field multiplying bare-ice melt fluxes (1 = clean ice).
        if self.debris_data is not None:
            smb.grid.geometry.debris.set(
                1.0 - cfg.debris_factor * self.debris_data.debris_fraction)
        # Static substrate temperature: annual-mean air temperature, capped at
        # the melting point (a permafrost-style proxy; not anomaly-shifted).
        smb.grid.geometry.t_base.set(
            cp.minimum(cp.asarray(gd.monthly_t2m.values).mean(axis=0), 0.0))

        spy = SECONDS_PER_YEAR
        smb.grid.thermodynamics.H_atm.set(cfg.mu_H_atm * spy)
        smb.grid.thermodynamics.H_base0.set(cfg.H_base0 * spy)
        smb.grid.radiation.q_sw_bulk.set(cfg.q_sw_bulk * spy)
        smb.grid.radiation.q_sw_insol.set(cfg.mu_cloud_factor * cfg.q_sw_clear * spy)
        smb.grid.radiation.q_sw_dif.set(
            (cfg.mu_cloud_factor * cfg.k_diffuse_clear
             + (1.0 - cfg.mu_cloud_factor) * cfg.k_diffuse_cloud) * cfg.q_sw_clear * spy)
        smb.grid.radiation.q_lw0.set(cfg.q_lw0 * spy)
        smb.grid.radiation.albedo_snow.set(cfg.albedo_snow)
        smb.grid.radiation.albedo_ice.set(cfg.albedo_ice)
        smb.grid.radiation.M_albedo.set(cfg.M_albedo)

        sigma_t2m = float(cp.asnumpy(smb.grid.temperature.sigma_t2m.value))
        self.temp_dev = generate_temp_deviations(
            12, cfg.enthalpy_n_substeps, sigma_t2m,
            np.random.default_rng(cfg.enthalpy_seed))
        smb.forward(temp_deviations=self.temp_dev)
        return smb

    def _build_domain_data(self) -> DomainData:
        """Shared non-product context (masks, DEM, glacier labels) on the
        cropped grid — everything a loss term may need that is not itself an
        observational product."""
        gd = self.gridded_data
        domain_mask = torch.tensor(gd.domain_mask.values, device="cuda")
        rgi_mask = torch.tensor(gd.rgi_mask.values, device="cuda")
        rgi_label = torch.tensor(gd.rgi_label.values, device="cuda", dtype=torch.long)
        rgi_label[torch.isnan(rgi_label)] = -1
        surge_type = torch.tensor(gd.surge_type.values, device="cuda", dtype=torch.long)
        dem = torch.tensor(gd.elevation.values, dtype=torch.float32, device="cuda")
        return DomainData(
            dem=dem, rgi_mask=rgi_mask, rgi_label=rgi_label,
            surge_type=surge_type, obs_mask=rgi_mask, domain_mask=domain_mask,
        )

    @property
    def required_times(self) -> tuple:
        """Union of every observation's required emission times."""
        return merge_times(*[obs.required_times for obs in self.observations])

    def get_observation(self, name: str) -> Optional[Observation]:
        """The observation with `Observation.name == name`, or None if the
        domain lacks that product."""
        for obs in self.observations:
            if obs.name == name:
                return obs
        return None

    def write_observations(self, output_dir: str,
                           base: str = "observations", level: int = 0) -> None:
        """Dump the observational products to a single-frame PVD.

        Writes the domain context (DEM, masks) plus whatever static fields
        each observation contributes via `diagnostics()` — velocity as a
        vector field, snowline/dhdt rasters, the rasterized flightline picks.
        Call this at the start of each level, pointed at the same directory
        the per-iteration diagnostics go to. `level > 0` writes every field
        box-restricted to that multigrid grid (masks become fractions; NaN
        no-data rasters are averaged over their valid pixels) — the targets
        the coarse-space terms actually see.
        """
        from .io import write_static_vti
        from .observations import _restrict, _restrict_weighted

        def coarsen(arr):
            if level == 0:
                return arr
            arr = torch.as_tensor(arr).to(torch.float32)
            if torch.isnan(arr).any():
                valid = torch.isfinite(arr).to(torch.float32)
                out = _restrict_weighted(arr.nan_to_num(), valid, level)
                return out.masked_fill(_restrict(valid, level) == 0, float("nan"))
            return _restrict(arr, level)

        domain = self.domain
        srf = self.get_observation("srf")
        scalars = {
            "srf_obs": (srf.S_obs if srf is not None
                        else torch.clamp(domain.dem, min=0.0)),
            "dem": domain.dem,
            "rgi_mask": domain.rgi_mask,
            "obs_mask": domain.obs_mask,
            "domain_mask": domain.domain_mask,
        }
        for obs in self.observations:
            scalars.update(obs.diagnostics())
        if self.debris_data is not None:
            scalars["debris_fraction"] = torch.tensor(
                self.debris_data.debris_fraction.values,
                dtype=torch.float32, device="cuda")
        vel = self.get_observation("vel")
        vector_fields = ({"U_obs": (coarsen(vel.u_obs), coarsen(vel.v_obs))}
                         if vel is not None else None)
        write_static_vti(
            self.mg[level], output_dir, base,
            scalar_fields={k: coarsen(v) for k, v in scalars.items()},
            vector_fields=vector_fields,
        )

    def refresh_fingerprints(self, *, params, level: int, fp=None,
                             verbose: bool = True) -> None:
        """Measure the smooth SMB parameters' sensitivity fingerprints at the
        CURRENT state and install them on the whitened surface / dh/dt
        likelihoods (see config.FingerprintNuisance for the model and
        references). Cost: 1 + len(params)·n_modes forwards at `level`, all
        under no_grad.

        Perturbations are taken in WHITENED coordinates along the leading
        orthonormal DCT modes (λ ascending — the parameter prior's KL basis
        to good approximation), so each direction is one prior std of that
        mode and the installed fingerprints w_j = Δz/δ are in
        prior-standardized units: the downdate then caps the data-side
        information at 1/s² per mode against a unit prior curvature.
        The driver calls this at every level start (and every
        `fp.refresh` iterations when > 0); fingerprints are level-native and
        a level mismatch silently disables the downdate, so a term is never
        downdated with a stale grid."""
        fp = self.config.fingerprint_nuisance if fp is None else fp
        if fp is None:
            return
        znames = {"log_H_atm": "z_log_H_atm", "logit_cloud": "z_logit_cloud"}
        members = {"log_H_atm": self.priors.h_atm_model,
                   "logit_cloud": self.priors.cloud_model}
        for p in fp.params:
            if p not in znames:
                raise ValueError(
                    f"FingerprintNuisance param {p!r}: only "
                    f"{tuple(znames)} carry fingerprint support")
            if members[p] is None:
                raise ValueError(
                    f"FingerprintNuisance param {p!r} requires "
                    f"smb_model='enthalpy' (no Matérn member built)")
        targets = [obs for obs in self.observations
                   if (obs.name == "srf" or obs.name.startswith("dhdt"))
                   and getattr(obs, "noise", None) is not None]
        if not targets:
            return
        from scipy.fft import idctn as _idctn
        ny, nx = self.ny, self.nx
        lam = ((2.0 - 2.0 * np.cos(np.pi * np.arange(ny) / ny))[:, None]
               + (2.0 - 2.0 * np.cos(np.pi * np.arange(nx) / nx))[None, :])
        flat = np.argsort(lam.ravel())[:fp.n_modes]
        modes = []
        for f in flat:
            one = np.zeros((ny, nx), dtype=np.float64)
            one[np.unravel_index(f, (ny, nx))] = 1.0
            modes.append(torch.tensor(
                _idctn(one, norm="ortho").astype(np.float32), device="cuda"))

        cfg0 = self.config.at_iteration(0, 0, schedule=False)
        mask = (self.domain.rgi_mask * self.domain.domain_mask).to(torch.float32)
        kw = dict(config=cfg0, domain=self.domain, mask=mask, dx=self.dx)
        if verbose:
            for obs in targets:
                if obs.fingerprint_c is not None:
                    c = ", ".join(f"{v:+.2f}" for v in obs.fingerprint_c.tolist())
                    print(f"[fingerprints] {obs.name}: last c-hat = [{c}]")
        with torch.no_grad():
            sim0, phys0 = self.simulate(level=level, params=params)
            base = {obs.name: obs.residuals(sim=sim0, physical=phys0, **kw)["z"]
                    .detach() for obs in targets}
            del sim0, phys0
            fps = {obs.name: [] for obs in targets}
            for pname, s in zip(fp.params, fp.s):
                for phi in modes:
                    pert = params.detach_clone()
                    getattr(pert, znames[pname]).add_(fp.fd_step * phi)
                    sim, phys = self.simulate(level=level, params=pert)
                    for obs in targets:
                        z = obs.residuals(sim=sim, physical=phys, **kw)["z"]
                        fps[obs.name].append(
                            ((z.detach() - base[obs.name]) / fp.fd_step, s))
                    del sim, phys, pert
        for obs in targets:
            obs.set_fingerprints(fps[obs.name], level)
        if verbose:
            for obs in targets:
                norms = ", ".join(f"{float((w ** 2).sum()):.3g}"
                                  for w, _ in fps[obs.name])
                print(f"[fingerprints] {obs.name} @ level {level}: "
                      f"|w|^2 per direction = [{norms}] "
                      f"(data info per mode, prior-std units; floor 1/s^2)")

    def write_residuals(self, output_dir: str, sim, physical,
                        base: str = "residuals") -> None:
        """Dump every field likelihood's residual diagnostics (`obs.residuals()`:
        σ-normalized `r` and whitened `z`, per component for velocity) on the
        fine grid to a single-frame PVD, keyed `<name>_<field>` (srf_r, srf_z,
        vel_z_u, …, dhdt_z). Call after a level's final simulate. Terms
        evaluated in coarse space at that level (surface) come back on the
        coarse grid and are prolonged (bilinear) onto the fine grid here so
        the dump stays one file."""
        from .forward import differentiable_prolongation
        from .io import write_static_vti

        mask = (self.domain.rgi_mask * self.domain.domain_mask).to(torch.float32)
        config = self.config.at_iteration(0, 0, schedule=False)
        scalars = {}
        for obs in self.observations:
            fields = obs.residuals(
                sim=sim, physical=physical, config=config,
                domain=self.domain, mask=mask, dx=self.dx)
            for k, v in fields.items():
                v = v.detach()
                if v.shape != (self.ny, self.nx):
                    n = int(round(math.log2(self.ny / v.shape[0])))
                    v = differentiable_prolongation(v, n)
                scalars[f"{obs.name}_{k}"] = v
        if not scalars:
            return
        write_static_vti(self.mg[0], output_dir, base, scalar_fields=scalars)

    def _build_initial_parameters(self) -> WhitenedParameters:
        priors = self.priors
        ny, nx = self.ny, self.nx

        log_beta = torch.tensor(cp.log(self.mg[0].sliding.beta.data), device="cuda")
        z_log_beta = GGaPPWhiten.apply(priors.log_beta_model,
                                       log_beta - priors.mu_log_beta)

        bed = torch.tensor(self.mg[0].geometry.bed.data, device="cuda")
        if "bed_obs" in self.gridded_data:
            # Gridded-bed domains: start the hierarchical mean on a smoothed
            # copy of the initial bed (Gaussian at the mean prior's length
            # scale) so the fluctuation field z_bed starts small and the
            # initial bed-prior cost is O(1) instead of O((H/sigma_bed)^2).
            bed_mean = torch.tensor(
                self._smoothed(cp.asnumpy(self.mg[0].geometry.bed.data)), device="cuda")
            print(f"[init] bed_mean seeded from the bed smoothed at "
                  f"{self.config.mean_prior.l / 2000:.0f} km")
        else:
            bed_mean = torch.zeros(ny, nx, dtype=torch.float32, device="cuda")
        # The physical bed is Map(z_bed) ALONE (bed_from_whitened); bed_mean
        # enters only the prior term on (bed_uncond - bed_mean). So z_bed
        # whitens the full initial bed, and a nonzero bed_mean just makes
        # that prior residual small — it must not be subtracted here.
        z_bed = GGaPPWhiten.apply(priors.bed_model, bed)
        z_bed_mean = GGaPPWhiten.apply(priors.mean_model, bed_mean)

        z_pbias = torch.zeros(ny, nx, dtype=torch.float32, device="cuda")
        # Temperature bias field: always allocated (shape-consistent save/load
        # across tasks), but inert at 0 K unless config.tbias_enabled.
        z_tbias = torch.zeros(ny, nx, dtype=torch.float32, device="cuda")

        if self.config.smb_model == "enthalpy":
            # The enthalpy grid carries no mf/rf; the inactive temperature-
            # index scalars sit at their prior median (z = 0).
            z_log_mf = torch.zeros((), dtype=torch.float32, device="cuda")
            z_log_rf = torch.zeros((), dtype=torch.float32, device="cuda")
        else:
            log_mf = torch.tensor(cp.log(self.smb_model.grid.temperature.mf.value), device="cuda")
            log_rf = torch.tensor(cp.log(self.smb_model.grid.insolation.rf.value), device="cuda")
            z_log_mf = (log_mf - priors.mu_log_mf) / priors.sigma_log_mf
            z_log_rf = (log_rf - priors.mu_log_rf) / priors.sigma_log_rf

        # Precip-depletion scalars start at the prior mean (z = 0); inactive
        # ones are inert, but always present so save/load and the prior term
        # stay shape-consistent across tasks.
        z_tau = torch.zeros((), dtype=torch.float32, device="cuda")
        z_z0 = torch.zeros((), dtype=torch.float32, device="cuda")
        # Enthalpy SMB parameter FIELDS (log H_atm, logit clear-sky fraction):
        # always allocated (ny, nx) like z_tbias — inert at 0 (= the prior
        # median field) under the temperature-index backend, where no Matern
        # member exists and they never enter the optimizer.
        z_log_H_atm = torch.zeros(ny, nx, dtype=torch.float32, device="cuda")
        z_logit_cloud = torch.zeros(ny, nx, dtype=torch.float32, device="cuda")

        params = WhitenedParameters(
            z_bed=z_bed, z_bed_mean=z_bed_mean, z_log_beta=z_log_beta,
            z_pbias=z_pbias, z_tbias=z_tbias,
            z_log_mf=z_log_mf, z_log_rf=z_log_rf,
            z_tau=z_tau, z_z0=z_z0,
            z_log_H_atm=z_log_H_atm, z_logit_cloud=z_logit_cloud,
        )
        params.requires_grad_()
        return params

    # -------------------------------------------------------------------- core

    def set_bed_data_override(self, data) -> None:
        """RTO hook: substitute per-sample perturbed bed-conditioning data
        (b + eps, Matheron's rule) for subsequent physical_from calls. Pass
        None to restore the conditioner's stored data. No-op relevance unless
        config.bed_conditioning.enabled."""
        self._bed_data_override = data

    def physical_from(self, params: WhitenedParameters):
        """Map whitened parameters back to physical fields. Returns a namespace
        with (bed, bed_mean, log_beta, pbias, tbias, log_mf, log_rf, tau, z0)."""
        priors = self.priors

        bed, bed_mean, bed_uncond = priors.bed_from_whitened(
            params.z_bed, params.z_bed_mean,
            data_override=self._bed_data_override)
        log_beta = priors.log_beta_from_whitened(params.z_log_beta)
        pbias = GGaPPMap.apply(priors.pbias_model, params.z_pbias)
        # Additive temperature bias (K): mapped only when the term is enabled
        # (no Matern hierarchy exists otherwise); None keeps t2m untouched.
        tbias = (GGaPPMap.apply(priors.tbias_model, params.z_tbias)
                 if priors.tbias_model is not None else None)
        log_mf = priors.mu_log_mf + priors.sigma_log_mf * params.z_log_mf
        log_rf = priors.mu_log_rf + priors.sigma_log_rf * params.z_log_rf
        # Precip-depletion scalars: plain affine de-whitening of a normal prior.
        tau = priors.mu_tau + priors.sigma_tau * params.z_tau
        z0 = priors.mu_z0 + priors.sigma_z0 * params.z_z0
        # Enthalpy-model parameter fields: log H_atm (in W m-2 K-1) and logit
        # of the clear-sky fraction — scalar median + smooth GP fluctuation,
        # the log_beta pattern. Exponentiation/sigmoid happens in
        # simulate_physical; under ETIM no member exists and the affine
        # fallback keeps the (inert, z = 0) fields at the median.
        log_H_atm = priors.mu_log_H_atm + (
            GGaPPMap.apply(priors.h_atm_model, params.z_log_H_atm)
            if priors.h_atm_model is not None
            else priors.sigma_log_H_atm * params.z_log_H_atm)
        logit_cloud = priors.mu_logit_cloud + (
            GGaPPMap.apply(priors.cloud_model, params.z_logit_cloud)
            if priors.cloud_model is not None
            else priors.sigma_logit_cloud * params.z_logit_cloud)
        return PhysicalParameters(
            bed=bed, bed_mean=bed_mean, log_beta=log_beta,
            pbias=pbias, log_mf=log_mf, log_rf=log_rf, tau=tau, z0=z0,
            log_H_atm=log_H_atm, logit_cloud=logit_cloud,
            bed_uncond=bed_uncond, tbias=tbias,
        )

    def effective_log_pbias(self, physical: "PhysicalParameters") -> torch.Tensor:
        """Joint log-precip bias field (fine grid): the Matern pbias field minus
        the optional elevation-dependent depletion ramp. This is the quantity
        that actually multiplies precip in the forward model
        (precip = base * exp(this)); the diagnostic VTI writes it too.

        z is the static observed DEM elevation, so this field is built once per
        forward call (not per step), avoiding the per-step VRAM blowup the precip
        multiplier dodges. With precip_lapse_enabled=False it is exactly the
        Matern pbias field.
        """
        cfg = self.config
        log_pbias = physical.pbias
        if cfg.precip_lapse_enabled:
            w = cfg.precip_lapse_w
            depletion = torch.exp(-physical.tau) * w * torch.nn.functional.softplus(
                (self.domain.dem - physical.z0) / w)
            log_pbias = log_pbias - depletion
        return log_pbias

    def _initial_thickness_from_geometry(self, bed: torch.Tensor) -> torch.Tensor:
        """Seed integration thickness from the observed surface and the current
        bed iterate.

        Inverts the surface model S = max(bed + H, (1 - r) H) for H given the
        observed surface and the *current optimized bed* (`physical.bed`):
        grounded ice gets S_obs - bed, floating ice the hydrostatic value
        S_obs / (1 - r). The bed is detached, so the seeded thickness tracks
        the bed iterate across the inversion but carries no gradient w.r.t. it.
        Restricted to the observed ice extent and floored at `init_H_floor`.
        """
        cfg = self.config
        r = cfg.rho_ice / cfg.rho_water
        bed = bed.detach()                  # current bed iterate, no gradient
        S_obs = torch.clamp(self.domain.dem, min=0.0)   # ice/water surface

        H_grounded = S_obs - bed
        H_float = S_obs / (1.0 - r)
        grounded = bed >= -(r / (1.0 - r)) * S_obs
        H = torch.where(grounded, H_grounded, H_float)

        ice = self.domain.rgi_mask.to(torch.float32)
        H = torch.clamp(H, min=cfg.init_H_floor) * ice \
            + cfg.init_H_floor * (1.0 - ice)
        return H.to(torch.float32)

    def reset_state(self, level: int) -> None:
        mg = self.mg
        mg.state.u.set(0.0, start_level=level)
        mg.state.v.set(0.0, start_level=level)
        mg.state.ud.set(0.0, start_level=level)
        mg.state.vd.set(0.0, start_level=level)
        mg.state.H.set(self.config.thklim, start_level=level)
        mg.state.H_prev.set(self.config.thklim, start_level=level)
        mg.state.mask.set(0.0, start_level=level)
        mg.adjoint.lambda_u.set(0.0, start_level=level)
        mg.adjoint.lambda_v.set(0.0, start_level=level)
        mg.adjoint.lambda_ud.set(0.0, start_level=level)
        mg.adjoint.lambda_vd.set(0.0, start_level=level)
        mg.adjoint.lambda_H.set(0.0, start_level=level)

    def update_depth(self, bed: torch.Tensor) -> None:
        """Blend the new bed-derived (signed) depth into the existing depth
        field. depth = -bed: the head deficit below sea level, negative on
        dry land (glide's flotation excess uses the sign)."""
        depth_new = (-bed).detach()
        w = self.config.depth_blend
        self.mg.geometry.depth.set(
            w * cp.asarray(depth_new) + (1 - w) * self.mg[0].geometry.depth.data
        )

    def simulate_physical(
        self,
        *,
        level: int,
        physical: "PhysicalParameters",
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        record_states_at: Optional[list] = None,
        record_volumes_at: Optional[list] = None,
        time_writer=None,
    ):
        """Run the forward model from physical parameters directly.

        Used by sensitivity.py (which optimizes in physical space to get
        gradients of ΔV w.r.t. bed/pbias/etc., not whitened coordinates).
        """
        from .forward import differentiable_restriction

        cfg = self.config
        if record_states_at is None:
            # Default: emit state at every observation epoch so compute_loss
            # can compare each product against the model at its own time.
            # Pass an explicit empty list to skip snapshots (e.g. projections).
            record_states_at = list(self.required_times)
        self.update_depth(physical.bed)
        self.reset_state(level)

        # Joint log-precip bias (Matern field minus the optional elevation
        # depletion ramp); see effective_log_pbias for why it is static.
        precip_ = self.precip * torch.exp(self.effective_log_pbias(physical))
        # The optional additive temperature bias is NOT pre-added to t2m here
        # (unlike the precip line above, whose retained (12, ny, nx) cost is
        # tolerable for a single field): the (ny, nx) tbias is threaded into
        # the checkpointed SMB fn, which adds it per step alongside the scalar
        # anomaly — the biased t2m is recomputed in backward instead of
        # retained, and only an (ny, nx) gradient leaves each segment (see
        # compute_smb). t_base (enthalpy substrate proxy) deliberately stays
        # unbiased, matching the anomaly's treatment.
        tbias = getattr(physical, "tbias", None)
        if self.avalanche_op is not None:
            # Hoisted avalanche (config.avalanche_hoisted): apply R here, once
            # per forward call, instead of ~150 times inside the per-step SMB
            # (forward steps + checkpoint recomputes + adjoint rebuilds).
            # Exactly equivalent ONLY because the enthalpy core takes total
            # precip and each step scales it by a scalar — R(c*P) = c*R(P).
            precip_ = AvalancheStep.apply(self.avalanche_op, precip_)
        if cfg.smb_model == "enthalpy":
            mf = rf = None
            # Physical scalars in the model's J m-2 yr-1 (K-1) units:
            # H_atm from its log in W m-2 K-1; the direct (q_sw_insol) and
            # diffuse (q_sw_dif) shortwave scales from the SAME clear-sky
            # fraction f = sigmoid(logit_cloud) times the extraterrestrial S0,
            # so both carry gradient to z_logit_cloud.
            H_atm = torch.exp(physical.log_H_atm) * SECONDS_PER_YEAR
            f_clear = torch.sigmoid(physical.logit_cloud)
            q_sw_insol = f_clear * cfg.q_sw_clear * SECONDS_PER_YEAR
            q_sw_dif = ((f_clear * cfg.k_diffuse_clear
                         + (1.0 - f_clear) * cfg.k_diffuse_cloud)
                        * cfg.q_sw_clear * SECONDS_PER_YEAR)
        else:
            mf = torch.exp(physical.log_mf)
            rf = torch.exp(physical.log_rf)
            H_atm = q_sw_insol = q_sw_dif = None

        bed_ = differentiable_restriction(physical.bed, level, method='avg')
        if cfg.init_from_observed_geometry:
            H_prev_full = self._initial_thickness_from_geometry(physical.bed)
        else:
            H_prev_full = self.H_prev
        H_prev_ = differentiable_restriction(H_prev_full, level)
        # config.beta_max caps the FINE-level field before restriction (the
        # order forward_standalone.py uses, so both run the same beta on every
        # level); no gradient passes above the cap, where only the prior acts
        log_beta_full = physical.log_beta
        if cfg.beta_max is not None:
            log_beta_full = torch.clamp(log_beta_full, max=math.log(float(cfg.beta_max)))
        log_beta_ = differentiable_restriction(log_beta_full, level)
        beta_ = torch.exp(log_beta_)

        return simulate(
            model=self.model,
            smb_model=self.smb_model,
            smb_kind=cfg.smb_model,
            level=level,
            t_start=cfg.t_start if t_start is None else t_start,
            t_end=cfg.t_end if t_end is None else t_end,
            dt=cfg.dt,
            bed_=bed_,
            beta_=beta_,
            H_prev_=H_prev_,
            t2m=self.t2m,
            precip_=precip_,
            tbias=tbias,
            debris=self.debris,
            mf=mf,
            rf=rf,
            insol_mean=self.insol_mean,
            insol_dif=self.insol_dif,
            t_base=self.t_base,
            H_atm=H_atm,
            q_sw_insol=q_sw_insol,
            q_sw_dif=q_sw_dif,
            enthalpy_consts=self.enthalpy_consts,
            temp_dev=self.temp_dev,
            anomaly_integration=cfg.anomaly_integration,
            domain_mask=self.domain.domain_mask,
            temperature_anomaly=self.temperature_anomaly,
            # config.climatology_only: reference climate for the whole run —
            # the monthly climatology + tbias / pbias, no yearly fields and no
            # anomaly index (the record is still passed so the step sequence
            # and the year lookups are unchanged; alpha 0 makes every term the
            # climatological one).
            base_anomaly=0.0 if cfg.climatology_only else self.base_anomaly,
            alpha_t2m=0.0 if cfg.climatology_only else self.alpha_t2m,
            precip_anomaly=None if cfg.climatology_only else self.precip_anomaly,
            base_precip=None if cfg.climatology_only else self.base_precip,
            alpha_precip=None if cfg.climatology_only else self.alpha_precip,
            dx_fine=self.dx,
            n_glen=float(cfg.n_glen),
            grad_start_time=cfg.grad_start_time,
            flotation_factor=1.0 - cfg.rho_ice / cfg.rho_water,
            record_states_at=record_states_at,
            record_volumes_at=record_volumes_at,
            time_writer=time_writer,
            ocean_forcing=self.ocean_forcing,
            dt_schedule=cfg.dt_schedule,
            yearly_climate=None if cfg.climatology_only else self.yearly_climate,
            # NOT gated on climatology_only: the missing variance is a property
            # of the climate, not of the anomaly. A drift diagnostic run
            # without it would sit ~90 Gt/yr above the calibration and report
            # that offset as drift.
            interannual_sigma=cfg.interannual_sigma,
            interannual_nodes=cfg.interannual_nodes,
            thermal=self.thermal_driver(level),
            thermal_T_surface=self.thermal_surface_temperature(tbias),
        )

    # ------------------------------------------------------------ thermal
    def thermal_driver(self, level: int):
        """The level's ThermalDriver (built once per level), or None when
        config.thermal is None (isothermal B from A_glen)."""
        tcfg = getattr(self.config, "thermal", None)
        if tcfg is None:
            return None
        if not hasattr(self, "_thermal_drivers"):
            self._thermal_drivers = {}
        if level not in self._thermal_drivers:
            from .thermal import ThermalDriver
            self._thermal_drivers[level] = ThermalDriver(
                self.model, level, tcfg, rho_i=float(self.config.rho_ice),
                thin_B=float(self.config.B_rate) if tcfg.thin_ice_isothermal else None)
        return self._thermal_drivers[level]

    def thermal_surface_temperature(self, tbias=None):
        tcfg = getattr(self.config, "thermal", None)
        if tcfg is None:
            return None
        from .thermal import surface_temperature_fine
        return surface_temperature_fine(
            self.t2m, tbias if tcfg.surface_T_tbias else None)

    def simulate(
        self,
        *,
        level: int,
        params: WhitenedParameters,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        record_states_at: Optional[list] = None,
        record_volumes_at: Optional[list] = None,
        time_writer=None,
    ):
        """Run the forward model from whitened parameters. Returns (sim, physical)."""
        physical = self.physical_from(params)
        sim = self.simulate_physical(
            level=level, physical=physical,
            t_start=t_start, t_end=t_end,
            record_states_at=record_states_at,
            record_volumes_at=record_volumes_at,
            time_writer=time_writer,
        )
        return sim, physical

    def compute_loss(
        self,
        *,
        sim,
        physical: "PhysicalParameters",
        params: WhitenedParameters,
        observations: Optional[list] = None,
        prior_means: Optional[PriorMeans] = None,
        mask: Optional[torch.Tensor] = None,
        iteration: int = 0,
        level: int = 0,
        schedule: bool = False,
    ) -> LossTerms:
        """Assemble all loss terms from a finished simulation.

        Each observation contributes one data term, evaluated against the
        model state at its own required time(s) (the simulation must have been
        run with those times recorded — the default in `simulate`).
        `observations` may be a substitute list (RTO passes randomized copies).

        `iteration`/`level` resolve any scheduled loss weight to its value at
        this optimizer step. `schedule` selects whether continuation ramps are
        honored (the initial inverse solve, `schedule=True`) or each weight
        collapses to its steady-state value (RTO and analysis, the default) —
        see config.Schedule. Constant weights are unaffected either way.
        """
        observations = self.observations if observations is None else observations
        prior_means = prior_means or PriorMeans.zeros()
        if mask is None:
            mask = (self.domain.rgi_mask * self.domain.domain_mask).to(torch.float32)

        config = self.config.at_iteration(iteration, level, schedule=schedule)
        data_terms = {}
        for obs in observations:
            weight = obs.weight_at(iteration, level, schedule=schedule)
            data_terms[obs.name] = obs.loss(
                sim=sim, physical=physical, config=config,
                domain=self.domain, mask=mask, dx=self.dx, weight=weight,
            )
        J_prior_terms = compute_prior(
            config=config,
            priors=self.priors,
            params=params,
            physical_bed=physical.bed,
            physical_bed_mean=physical.bed_mean,
            log_rf=physical.log_rf,
            log_mf=physical.log_mf,
            prior_means=prior_means,
            physical_bed_uncond=getattr(physical, "bed_uncond", None),
        )
        return LossTerms(
            data_terms=data_terms,
            J_prior_bed=J_prior_terms[0],
            J_prior_bed_mean=J_prior_terms[1],
            J_prior_beta=J_prior_terms[2],
            J_prior_pbias=J_prior_terms[3],
            J_prior_tbias=J_prior_terms[4],
            J_prior_h_atm=J_prior_terms[5],
            J_prior_cloud=J_prior_terms[6],
            J_prior_smb=J_prior_terms[7],
        )


@dataclass
class PhysicalParameters:
    bed: torch.Tensor
    bed_mean: torch.Tensor
    log_beta: torch.Tensor
    pbias: torch.Tensor
    log_mf: torch.Tensor
    log_rf: torch.Tensor
    # Precip-depletion scalars. Optional so physical-space callers (sensitivity)
    # that predate the term still construct a valid object; only read when
    # config.precip_lapse_enabled.
    tau: Optional[torch.Tensor] = None
    z0: Optional[torch.Tensor] = None
    # Enthalpy-model scalars; only read when config.smb_model == "enthalpy".
    log_H_atm: Optional[torch.Tensor] = None
    logit_cloud: Optional[torch.Tensor] = None
    # The unconditional bed field Map(z_bed), before the bed-conditioning
    # kriging correction. None (or equal to bed) in the legacy
    # parametrization; the prior term re-whitens THIS field, never the
    # conditioned one. Optional so physical-space callers (sensitivity)
    # still construct a valid object.
    bed_uncond: Optional[torch.Tensor] = None
    # Additive temperature bias field (K); None unless config.tbias_enabled.
    tbias: Optional[torch.Tensor] = None
