"""
Forward simulation helpers shared across all four tasks.

The time-stepping loop in `simulate` runs the SMB → ice-dynamics chain over a
step sequence designed by `scheduling.build_step_sequence` (uniform spinup
grid snapped onto every requested emission time) and returns a `SimResult`
holding a `ModelState` snapshot at each requested time plus the final state.
Used identically by inverse, rto_sample, and sensitivity.
"""
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, Sequence

import cupy as cp
import numpy as np
import torch
from torch.nn.functional import avg_pool2d, max_pool2d, interpolate
from torch.utils.checkpoint import checkpoint

from glare.torch import GlareStep
from glide.torch import GlideStep

# EnthalpyStep is newer than some installed glare versions; the enthalpy SMB
# path raises at problem-build time (problem.py) when it is unavailable.
try:
    from glare.torch import EnthalpyStep
except ImportError:
    EnthalpyStep = None

from .scheduling import build_step_sequence, merge_times

glare_step = GlareStep.apply
glide_step = GlideStep.apply
enthalpy_step = EnthalpyStep.apply if EnthalpyStep is not None else None


def year_overlap_weights(t0: float, t1: float, eps: float = 1e-9) -> list:
    """Fractional overlap of the step (t0, t1] with each calendar year [y, y+1).

    Returns [(year, weight), ...] with the weights summing to one — the exact
    quadrature weights for integrating a piecewise-constant annual signal (the
    anomaly record) over the step. Pure Python floats; the weights of any
    sub-partition of (t0, t1] recombine to these, which is what makes the
    integrated forcing independent of how the scheduler splits time.
    """
    if not t1 > t0:
        raise ValueError(f"empty step ({t0}, {t1}]")
    out = []
    y = math.floor(t0 + eps)
    while y < t1 - eps:
        w = min(t1, y + 1.0) - max(t0, y)
        if w > eps:
            out.append((int(y), w))
        y += 1.0
    total = sum(w for _, w in out)
    return [(y, w / total) for y, w in out]


class YearField:
    """An anomaly term that is a reanalysis YEAR (config.yearly_climate_filename)
    rather than a scalar index anomaly. Only the year travels through the
    checkpointed SMB fn's arguments; the (12, ny, nx) fields are materialized
    from the YearlyClimate loader inside it (and again in the backward
    recompute) so they never sit on the tape — see yearly_climate.py."""
    __slots__ = ("year",)

    def __init__(self, year: int):
        self.year = int(year)

    def __repr__(self):
        return f"YearField({self.year})"


@lru_cache(maxsize=8)
def _hermite_nodes(n: int):
    """Probabilists' Gauss-Hermite rule: (nodes, weights) with sum(w) = 1, so
    sum_i w_i f(x_i) ~ E[f(Z)] for Z ~ N(0, 1), exact for polynomials up to
    degree 2n - 1."""
    x, w = np.polynomial.hermite_e.hermegauss(int(n))
    return tuple(float(v) for v in x), tuple(float(v) for v in w / np.sqrt(2.0 * np.pi))


def _expand_interannual(terms, sigma, n_nodes):
    """Replace each SCALAR anomaly term by a Gauss-Hermite fan, so the step's
    forcing carries total interannual variance `sigma**2` instead of only the
    spread the index terms happen to resolve.

    Why: SMB is concave in temperature (Greenland ~ -160 Gt/yr/K^2 integrated),
    so evaluating it at a step's MEAN anomaly is not its mean over the step --
    E[smb(T)] < smb(E[T]) by ~ curvature sigma^2 / 2. The reanalysis years
    escape this because each is its own term with its own fields, but an index
    year does not: under `mean_anomaly` a 50-yr spin-up step is ONE evaluation
    at the 50-year mean, and before 1784 the index is a multi-decadally
    smoothed core record that carries almost no interannual variance to begin
    with. The result is a spin-up systematically too positive, on the order of
    curvature sigma^2 / 2 ~ 90 Gt/yr for Greenland, while the calibration
    window after 1986 has no such bias -- an inconsistency between the two
    halves of the same run that grows with the spin-up length.

    The correction is the same device `temp_dev` already uses for
    within-month weather: a fixed, deterministic quadrature over the missing
    variance rather than a random draw, so the checkpointed backward
    recomputes the identical forward. Three nodes integrate a Gaussian exactly
    through fifth order, which is ample for a response whose aggregate
    curvature is quadratic; per cell the melt hinge is only approximated, but
    cells far from their threshold contribute no curvature either way.

    `sigma` is the TOTAL interannual std of the forcing temperature in the
    units the terms carry, i.e. after alpha_t2m. The variance the terms
    already carry is subtracted, so the same setting is right for every
    epoch: under `mean_anomaly` one term carries none and the full sigma is
    applied, under `annual` the spread across the step's index years counts
    against it, and a step whose terms already spread wider than `sigma` is
    left alone. YearField terms are never touched -- record years carry real
    weather.
    """
    scalars = [(a, w) for a, w in terms if not isinstance(a, YearField)]
    if not scalars or sigma is None or sigma <= 0 or n_nodes < 2:
        return terms
    wsum = sum(w for _, w in scalars)
    if wsum <= 0:
        return terms
    mean = sum(float(a) * w for a, w in scalars) / wsum
    carried = sum(w * (float(a) - mean) ** 2 for a, w in scalars) / wsum
    resid = sigma * sigma - carried
    if resid <= 0:
        return terms
    s = math.sqrt(resid)
    x, wq = _hermite_nodes(n_nodes)
    out = [t for t in terms if isinstance(t[0], YearField)]
    for a, w in scalars:
        out.extend((a + s * xi, w * wi) for xi, wi in zip(x, wq))
    return tuple(out)


def _term_forcing(a, tbias, base_anomaly, precip_step, precip_, yearly):
    """(temperature shift, precip field) of one anomaly term.

    Scalar term: the index anomaly folded into the (ny, nx) bias (one
    (12, ny, nx) temporary in the caller's add); precip is the step's
    index-scaled field. YearField term: the year's (12, ny, nx) anomaly
    (+ bias) and the year's ratio on the biased climatology — the step's
    index precip multiplier does not apply to record years.
    """
    if isinstance(a, YearField):
        d = yearly.t2m_anomaly(a.year)
        shift = d if tbias is None else d + tbias
        return shift, precip_ * yearly.precip_ratio(a.year)
    shift = (a - base_anomaly) if tbias is None else tbias + (a - base_anomaly)
    return shift, precip_step


def _finish_smb(smb, domain_mask, level, want_fine):
    # Mask, then restrict to the dynamics level *inside* the checkpoint:
    # avg_pool2d saves its input for backward, so restricting outside would
    # retain the full fine-grid smb for every time step of the run (~level-0
    # field x n_steps). Here the fine field dies with the checkpoint's forward;
    # only snapshot steps (want_fine) return it, for ModelState.smb_fine.
    smb = smb.masked_fill(~domain_mask, -10)
    smb_coarse = differentiable_restriction(smb, level)
    if want_fine:
        return smb_coarse, smb
    return smb_coarse


def compute_smb(smb_model, t2m, tbias, anomaly_terms, base_anomaly, precip_,
                precip_multiplier, debris, mf, rf, domain_mask,
                level=0, want_fine=False, yearly=None):
    # `precip_multiplier` is a scalar applied here, inside the checkpoint, so the
    # full (12, ny, nx) scaled precip field is recomputed in backward rather than
    # stored per time step (otherwise ~50 full-grid copies are retained).
    #
    # `tbias` is the optional (ny, nx) additive temperature bias, added here
    # inside the checkpoint for the same VRAM reason: the biased (12, ny, nx)
    # t2m is recomputed in backward instead of retained for the whole run, and
    # the broadcast-add reduces the full-rank g_t2m to an (ny, nx) gradient
    # INSIDE each segment backward, so only the small field accumulates across
    # the ~50 segments (otherwise two extra full-rank tensors are held:
    # t2m + tbias and its gradient accumulation buffer). None means no bias.
    #
    # `anomaly_terms` is a tuple of (anomaly, weight) pairs: the smb source over
    # the step is the weighted mean of the smb at each anomaly (weights sum to
    # 1). One term for the "end"/"mean_anomaly" integration modes; one term per
    # distinct overlapped year for "annual" — the exact interval integral of the
    # forcing, averaging the smb *fields* rather than the anomalies so the melt
    # nonlinearity is respected. The per-year (12, ny, nx) outputs are freed as
    # soon as `.mean(axis=0)` runs, so peak VRAM does not grow with the number
    # of terms.
    #
    # Returns the level-restricted smb, or (coarse, fine) when want_fine — see
    # _finish_smb. The (level=0, want_fine=False) defaults reproduce the
    # historical single-tensor fine-grid return for direct callers.
    precip_step = precip_ * precip_multiplier
    smb = None
    for a, w in anomaly_terms:
        # Fold the scalar anomaly into the (ny, nx) bias first so the sum
        # allocates a single (12, ny, nx) temporary. `yearly` terms (YearField)
        # materialize the year's fields here, inside the checkpoint.
        shift, precip_term = _term_forcing(a, tbias, base_anomaly, precip_step, precip_, yearly)
        s = glare_step(smb_model, t2m + shift, precip_term,
                       mf, rf, debris).mean(axis=0)
        smb = w * s if smb is None else smb + w * s
    return _finish_smb(smb, domain_mask, level, want_fine)


def compute_smb_enthalpy(smb_model, t2m, tbias, anomaly_terms, base_anomaly,
                         precip_, precip_multiplier, insol_mean, insol_dif, t_base,
                         H_atm, H_base0, q_sw_bulk, q_sw_insol, q_sw_dif, q_lw0,
                         albedo_snow, albedo_ice, M_albedo, debris, temp_dev,
                         domain_mask, level=0, want_fine=False, yearly=None):
    # Same contract as compute_smb (annual-mean smb, -10 fill outside the
    # domain, weighted mean over `anomaly_terms`, level restriction inside the
    # checkpoint), with the enthalpy core in place of the temperature index.
    # The precip multiplier and the optional (ny, nx) additive temperature
    # bias `tbias` are applied inside the checkpoint for the same VRAM reason
    # (see compute_smb); `temp_dev` is the fixed weather realization, so the
    # checkpointed re-execution during backprop reproduces the same forward.
    # The air-temperature anomaly and tbias shift t2m only; t_base and debris
    # are static.
    # Note the precip multiplier is deliberately per-step (not per-term): all
    # terms share one effective-precip field, which the glare adjoints
    # re-derive from the raw inputs each backward.
    precip_step = precip_ * precip_multiplier
    # glare's EnthalpyStep takes H_atm / q_sw_insol / q_sw_dif as (ny, nx)
    # parameter FIELDS. The scalars are broadcast here, inside the checkpoint
    # (autograd's expand-backward sums the field gradient onto the 0-d scalar;
    # the (ny, nx) copies are recomputed in backward, not retained per step).
    # A future spatially-varying parameter passes through unchanged: expand is
    # a no-op view on an already-(ny, nx) tensor.
    shape = t_base.shape
    H_atm = H_atm.expand(shape).contiguous()
    q_sw_insol = q_sw_insol.expand(shape).contiguous()
    q_sw_dif = q_sw_dif.expand(shape).contiguous()
    smb = None
    for a, w in anomaly_terms:
        shift, precip_term = _term_forcing(a, tbias, base_anomaly, precip_step, precip_, yearly)
        s = enthalpy_step(smb_model, t2m + shift, precip_term,
                          insol_mean, insol_dif, t_base, H_atm, H_base0, q_sw_bulk,
                          q_sw_insol, q_sw_dif, q_lw0, albedo_snow, albedo_ice,
                          M_albedo, debris, temp_dev).mean(axis=0)
        smb = w * s if smb is None else smb + w * s
    return _finish_smb(smb, domain_mask, level, want_fine)


def _restrict_cupy(a, n_times: int):
    """2x2 block mean of a numpy/cupy (ny, nx) array, `n_times` times, on the
    device; for forcing fields that carry no gradient (the calving margins)."""
    a = cp.asarray(a, dtype=cp.float32)
    for _ in range(n_times):
        ny, nx = a.shape
        a = a.reshape(ny // 2, 2, nx // 2, 2).mean(axis=(1, 3))
    return a


def differentiable_restriction(field: torch.Tensor, n_times: int, method: str = "avg") -> torch.Tensor:
    if method == "avg":
        fn = avg_pool2d
    elif method == 'max':
        fn = max_pool2d
    else:
        raise NotImplementedError(f"restriction method={method!r} not supported")
    for _ in range(n_times):
        field = fn(field[None, :, :], (2, 2))[0]
    return field


def differentiable_prolongation(field: torch.Tensor, n_times: int, grid_entity: str = "cell", mode: str = 'bilinear') -> torch.Tensor:
    for _ in range(n_times):
        if grid_entity == "cell":
            ny_fine, nx_fine = 2 * field.shape[0], 2 * field.shape[1]
        elif grid_entity == "vfacet":
            ny_fine, nx_fine = 2 * field.shape[0], 2 * (field.shape[1] - 1) + 1
        elif grid_entity == "hfacet":
            ny_fine, nx_fine = 2 * (field.shape[0] - 1) + 1, 2 * field.shape[1]
        else:
            raise ValueError(f"unknown grid_entity={grid_entity!r}")
        field = interpolate(field[None, None, :, :], (ny_fine, nx_fine), mode=mode).squeeze()
    return field


class ModelState:
    """Model state emitted at a single time during the forward integration.

    The coarse tensors are references into the autograd graph the stepping loop
    builds anyway, so retaining a snapshot costs essentially no extra VRAM. The
    fine-grid views are prolonged *lazily* on first access (and cached), so a
    snapshot that a loss term never touches never pays for prolongation.
    """

    def __init__(self, *, t: float, dt_step: float, level: int, u, v, ud, vd,
                 H, active, smb_fine, smb_coarse, bed_coarse,
                 flotation_factor: float, n_glen: float):
        self.t = t
        self.dt_step = dt_step   # length of the step that emitted this state
        self.level = level
        # u/v are the depth-averaged velocity components; ud/vd the MOLHO
        # deformational parts (identically zero under stress_scheme="ssa",
        # so everything downstream is scheme-agnostic). Surface velocity —
        # what feature-tracked mosaics observe — is u + ud/(n+1).
        self.u = u
        self.v = v
        self.ud = ud
        self.vd = vd
        self.n_glen = n_glen
        self.H = H
        self.active = active
        # The SMB step runs on the fine grid and is restricted for the
        # dynamics, so the fine field needs no (lazy) prolongation. It is only
        # retained for recorded snapshot steps (None otherwise — the stepping
        # loop keeps just the restricted field to bound VRAM).
        self.smb_fine = smb_fine
        self.smb_coarse = smb_coarse
        self.bed_coarse = bed_coarse
        self.flotation_factor = flotation_factor
        self._cache = {}

    def _lazy(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    @property
    def S_coarse(self):
        return self._lazy("S_coarse", lambda: torch.maximum(
            self.bed_coarse + self.H, self.flotation_factor * self.H))

    @property
    def u_fine(self):
        return self._lazy("u_fine", lambda: differentiable_prolongation(
            self.u, self.level, grid_entity="vfacet"))

    @property
    def v_fine(self):
        return self._lazy("v_fine", lambda: differentiable_prolongation(
            self.v, self.level, grid_entity="hfacet"))

    # Surface velocity u + ud/(n+1): summed at the coarse level so only one
    # prolongation runs (prolongation is linear). Under SSA this equals u/v
    # exactly (ud == 0) but costs one extra add per accessed snapshot.
    @property
    def u_surf_fine(self):
        return self._lazy("u_surf_fine", lambda: differentiable_prolongation(
            self.u + self.ud / (self.n_glen + 1.0), self.level,
            grid_entity="vfacet"))

    @property
    def v_surf_fine(self):
        return self._lazy("v_surf_fine", lambda: differentiable_prolongation(
            self.v + self.vd / (self.n_glen + 1.0), self.level,
            grid_entity="hfacet"))

    # Basal (sliding) velocity u - ud: glide's drag rows act on u_b = u - ud
    # (see glide residuals.cu). Under SSA ud == 0 so this equals u/v exactly.
    # Differenced at the coarse level so only one prolongation runs.
    @property
    def u_base_fine(self):
        return self._lazy("u_base_fine", lambda: differentiable_prolongation(
            self.u - self.ud, self.level, grid_entity="vfacet"))

    @property
    def v_base_fine(self):
        return self._lazy("v_base_fine", lambda: differentiable_prolongation(
            self.v - self.vd, self.level, grid_entity="hfacet"))

    @property
    def H_fine(self):
        return self._lazy("H_fine", lambda: differentiable_prolongation(
            self.H, self.level, grid_entity="cell"))

    @property
    def S_fine(self):
        return self._lazy("S_fine", lambda: differentiable_prolongation(
            self.S_coarse, self.level, grid_entity="cell"))

    @property
    def active_fine(self):
        return self._lazy("active_fine", lambda: differentiable_prolongation(
            self.active, self.level, grid_entity="cell"))


@dataclass
class SimResult:
    """Forward-run output: a state snapshot per requested emission time.

    `states` is keyed by the exact times passed via `record_states_at` (plus
    the final time); `final` is the last emitted state. The accessors below
    delegate to `final`, preserving the historical single-(final-)state API.
    `H_prev`/`H_prev_fine` hold the second-to-last emitted thickness — with
    `final.H_fine` and the last dt they give the model's dH/dt over the final
    step (legacy diagnostic + fallback for untimed dH/dt products).
    """
    states: dict
    final: ModelState
    volumes: dict = field(default_factory=dict)
    H_prev: Optional[torch.Tensor] = None
    # Truncated-backprop diagnostic (grad_start_time runs only): the thickness
    # handed across the no-grad boundary, with requires_grad. After backward,
    # H_boundary.grad is the gradient the truncation discarded.
    H_boundary: Optional[torch.Tensor] = None
    _H_prev_fine: Optional[torch.Tensor] = None

    def at(self, t: float, atol: float = 1e-6) -> ModelState:
        """State snapshot at time `t` (approximate float match)."""
        for tk, state in self.states.items():
            if abs(tk - t) <= atol:
                return state
        raise KeyError(
            f"no recorded model state at t={t}; available times: "
            f"{sorted(self.states.keys())} — was {t} passed via "
            f"record_states_at?"
        )

    @property
    def H_prev_fine(self):
        if self._H_prev_fine is None and self.H_prev is not None:
            self._H_prev_fine = differentiable_prolongation(
                self.H_prev, self.final.level, grid_entity="cell")
        return self._H_prev_fine

    # ------------------------------------------------ final-state delegation
    @property
    def u(self): return self.final.u
    @property
    def v(self): return self.final.v
    @property
    def ud(self): return self.final.ud
    @property
    def vd(self): return self.final.vd
    @property
    def H(self): return self.final.H
    @property
    def active(self): return self.final.active
    @property
    def bed_coarse(self): return self.final.bed_coarse
    @property
    def S_coarse(self): return self.final.S_coarse
    @property
    def u_fine(self): return self.final.u_fine
    @property
    def v_fine(self): return self.final.v_fine
    @property
    def u_surf_fine(self): return self.final.u_surf_fine
    @property
    def v_surf_fine(self): return self.final.v_surf_fine
    @property
    def H_fine(self): return self.final.H_fine
    @property
    def S_fine(self): return self.final.S_fine
    @property
    def active_fine(self): return self.final.active_fine
    @property
    def smb_fine(self): return self.final.smb_fine
    @property
    def smb_coarse(self): return self.final.smb_coarse


def simulate(
    *,
    model,
    smb_model,
    level: int,
    t_start: float,
    t_end: float,
    dt: float,
    bed_,
    beta_,
    H_prev_,
    t2m,
    precip_,
    tbias=None,
    debris=None,
    mf=None,
    rf=None,
    smb_kind: str = "temperature_index",
    insol_mean=None,
    insol_dif=None,
    t_base=None,
    H_atm=None,
    q_sw_insol=None,
    q_sw_dif=None,
    enthalpy_consts: Optional[dict] = None,
    temp_dev=None,
    anomaly_integration: str = "mean_anomaly",
    domain_mask,
    temperature_anomaly,
    base_anomaly: float,
    alpha_t2m: float,
    dx_fine: float,
    precip_anomaly=None,
    base_precip: Optional[float] = None,
    alpha_precip=None,
    n_glen: float = 3.0,
    grad_start_time: Optional[float] = None,
    flotation_factor: float = 0.0,
    record_states_at: Optional[Sequence[float]] = None,
    record_volumes_at: Optional[Sequence[float]] = None,
    time_writer=None,
    ocean_forcing=None,
    dt_schedule=(),
    yearly_climate=None,
    interannual_sigma: Optional[float] = None,
    interannual_nodes: int = 3,
    thermal=None,
    thermal_T_surface=None,
) -> SimResult:
    """Run the forward model on coarse `level` over a snapped step sequence.

    The run covers (t_start, horizon] where the horizon is max(t_end, latest
    requested emission time) — see `scheduling.build_step_sequence`. State
    snapshots are recorded at every time in `record_states_at` (keyed by the
    caller's exact floats) and scalar ice volumes at every time in
    `record_volumes_at`; the final state is always recorded.

    All field inputs (bed_, beta_, H_prev_) are expected at the coarse grid;
    full-grid quantities (t2m, precip_, domain_mask) are restricted internally.

    `tbias` is an optional (ny, nx) additive temperature bias (K), applied to
    t2m per step INSIDE the checkpointed SMB fn (never pre-add it to t2m —
    doing so retains the biased (12, ny, nx) field plus a full-rank gradient
    accumulation buffer for the whole run; see compute_smb).

    `smb_kind` selects the SMB backend. Both consume the static `debris`
    melt-attenuation field; "temperature_index" additionally consumes
    `mf`/`rf`, while "enthalpy" consumes `insol_mean`/`insol_dif`/`t_base`
    (fine-grid forcing; `insol_dif` is the static diffuse-sky potential, no
    gradient), the inverted scalars `H_atm`/`q_sw_insol`/`q_sw_dif`
    (J m-2 yr-1 (K-1); the two shortwave scales derive from one clear-sky fraction),
    the fixed constants in `enthalpy_consts` (H_base0, q_sw_bulk, q_lw0,
    albedo_snow, albedo_ice, M_albedo as 0-dim tensors), and the fixed `(12, n_substeps)`
    weather realization `temp_dev`.

    `flotation_factor` is `1 - rho_i/rho_w`; the surface is lower-bounded by
    the flotation freeboard `flotation_factor * H` for floating ice. The
    default of 0.0 reduces the bound to a sea-level floor (inert for grounded
    ice); callers pass the physical value derived from the config.

    `grad_start_time` enables truncated backpropagation through time: steps
    ending at or before it run under torch.no_grad (identical physics, no
    adjoint solves or retained state in backward), and the thickness handed
    across the boundary is exposed as SimResult.H_boundary so its .grad
    (populated by backward) measures the discarded gradient. Recorded
    state/volume times inside the no-grad window raise. None differentiates
    the whole run. See GlacierConfig.grad_start_time.

    `ocean_forcing` (an ocean.OceanForcing, or None) sets glide's calving
    margins q / h0 for every step before its dynamics solve (restricted to the
    run level, no gradient — the wrapper checkpoints them with the step so the
    adjoint re-solves each step with its own margins).

    `anomaly_integration` selects how the multi-year anomaly signal is
    integrated over each step (see GlacierConfig.anomaly_integration): "end"
    samples at the step's end time (legacy), "mean_anomaly" uses the
    overlap-weighted interval-mean anomaly, "annual" evaluates the SMB once per
    overlapped calendar year and combines with overlap weights (exact interval
    integral of the forcing). The precip-anomaly multiplier follows the same
    weights ("end" keeps its legacy endpoint trapezoid) and is held at its
    step mean in every mode.

    `yearly_climate` (a yearly_climate.YearlyClimate, or None) replaces the
    index anomaly by the reanalysis year's own fields for every calendar year
    it covers: each such year overlapped by a step is one SMB evaluation
    (a YearField term, weight = the overlap), and the step's remaining years
    keep the `anomaly_integration` treatment with their weights renormalized
    among themselves. The index precip multiplier likewise applies only to
    the remaining years. In "end" mode a step whose end year is on record is
    the whole-step evaluation on that year.

    `thermal` (a thermal.ThermalDriver for this level, or None) couples the
    enthalpy model: a frozen-geometry thermal spin-up on the initial state
    (surface temperature `thermal_T_surface`, fine grid, K), then one enthalpy
    step after every dynamics step, each updating glide's B. No gradient flows
    through it (GlideStep checkpoints the B each step used).
    """
    record_states_at = [float(t) for t in (record_states_at or [])]
    record_volumes_at = [float(t) for t in (record_volumes_at or [])]
    steps = build_step_sequence(
        t_start=t_start, t_end=t_end, dt_max=dt, dt_schedule=dt_schedule,
        required_times=merge_times(record_states_at, record_volumes_at),
    )

    if grad_start_time is not None:
        # A misfit evaluated against a no-grad snapshot silently contributes
        # zero gradient - refuse rather than fail quietly.
        bad = [t for t in merge_times(record_states_at, record_volumes_at)
               if t <= grad_start_time + 1e-6]
        if bad:
            raise ValueError(
                f"grad_start_time={grad_start_time} truncates gradients "
                f"through recorded times {bad}; every recorded state/volume "
                "time must lie strictly after it")
        if steps[-1][0] <= grad_start_time + 1e-6:
            raise ValueError(
                f"grad_start_time={grad_start_time} covers the whole run "
                f"(final step ends at {steps[-1][0]}); nothing would be "
                "differentiated")

    states: dict = {}
    volumes: dict = {}
    # Penultimate emitted thickness (coarse). Seeded with the initial H_prev_ so
    # a single-step run degrades to (final - seed)/dt instead of erroring.
    H_penult = H_prev_

    # Clip anomaly lookups to the last year in the dataset (relevant for
    # sensitivity.py-style projections that run beyond the observed record).
    year_max = int(temperature_anomaly.time.max().item())
    p_year_max = (int(precip_anomaly.time.max().item())
                  if precip_anomaly is not None else None)

    def raw_years(index, w_index):
        return [(y, w / w_index) for y, w in index]

    def anomaly_year(t: float, y_max: int) -> int:
        # Round before truncating so 2011.9999999 reads as 2012, matching the
        # intent of grid times that are integers up to float error.
        return int(min(round(t, 9), y_max))

    if anomaly_integration not in ("end", "mean_anomaly", "annual"):
        raise ValueError(
            f"unknown anomaly_integration {anomaly_integration!r}; "
            "expected 'end', 'mean_anomaly', or 'annual'")

    # One-time extraction of the annual records: the weighted modes look up
    # every year a step overlaps, which is too hot a loop for xarray .sel.
    t_anom = {int(y): float(v) for y, v in zip(
        temperature_anomaly.time.values,
        temperature_anomaly.temp_anomaly.values)}
    p_anom = ({int(y): float(v) for y, v in zip(
        precip_anomaly.time.values, precip_anomaly.precip_anomaly.values)}
        if precip_anomaly is not None else None)

    t_prev = float(t_start)
    state = None
    # Truncated backpropagation: H_boundary is the (detached, grad-requiring)
    # thickness handed from the last no-grad step to the first differentiable
    # one. After backward, its .grad is exactly the gradient the truncation
    # discarded - the online check that grad_start_time is early enough.
    H_boundary = None
    prev_no_grad = False
    if thermal is not None:
        def _momentum_solve():
            glide_step(cp.float32(t_start), cp.float32(thermal.cfg.spinup_momentum_dt),
                       model, level, H_prev_.detach(), bed_.detach(), beta_.detach(),
                       torch.zeros_like(H_prev_).detach())
        thermal.spinup(H0=H_prev_, momentum_solve=_momentum_solve,
                       T_surface_fine=thermal_T_surface)
    for t_next, dt_step in steps:
        no_grad_step = (grad_start_time is not None
                        and t_next <= grad_start_time + 1e-6)
        if prev_no_grad and not no_grad_step:
            H_prev_ = H_prev_.detach().requires_grad_()
            H_boundary = H_prev_
        prev_no_grad = no_grad_step
        # Anomaly terms for this step: (anomaly, weight) pairs consumed by the
        # checkpointed smb fn as a weighted mean of smb fields (weights sum
        # to 1). Raw annual values are merged before scaling so years beyond
        # the record (clamped to year_max) collapse into a single evaluation.
        on_record = (lambda y: yearly_climate is not None and yearly_climate.has(y))
        if anomaly_integration == "end":
            y_end = int(round(t_next, 9))
            if on_record(y_end):
                anomaly_terms = ((YearField(y_end), 1.0),)
            else:
                anomaly_terms = ((alpha_t2m * t_anom[min(y_end, year_max)], 1.0),)
            weights = None
        else:
            weights = year_overlap_weights(t_prev, t_next)
            # Record years are their own terms; the index years share the
            # rest of the step's weight, renormalized among themselves so
            # the mean-anomaly term is the mean over THOSE years.
            field_terms = tuple((YearField(y), w) for y, w in weights if on_record(y))
            index = [(y, w) for y, w in weights if not on_record(y)]
            w_index = sum(w for _, w in index)
            raw = [(t_anom[min(y, year_max)], w / w_index) for y, w in index] if index else []
            if not raw:
                anomaly_terms = field_terms
            elif anomaly_integration == "mean_anomaly":
                anomaly_terms = field_terms + (
                    (alpha_t2m * sum(a * w for a, w in raw), w_index),)
            else:  # "annual"
                merged: dict = {}
                for a, w in raw:
                    merged[a] = merged.get(a, 0.0) + w
                anomaly_terms = field_terms + tuple(
                    (alpha_t2m * a, w * w_index) for a, w in merged.items())

        # Restore the interannual variance the scalar index terms do not carry
        # (see _expand_interannual). Record years are untouched, so this acts
        # only on the pre-reanalysis spin-up.
        if interannual_sigma:
            anomaly_terms = _expand_interannual(
                anomaly_terms, interannual_sigma, interannual_nodes)

        if p_anom is not None:
            if anomaly_integration == "end":
                p_ratio = 0.5 * (p_anom[anomaly_year(t_prev, p_year_max)]
                                 + p_anom[anomaly_year(t_next, p_year_max)]) / base_precip
            elif index:
                p_ratio = sum(w * p_anom[min(y, p_year_max)]
                              for y, w in raw_years(index, w_index)) / base_precip
            else:
                p_ratio = 1.0
            precip_multiplier = 1.0 + alpha_precip * (p_ratio - 1.0)
        else:
            precip_multiplier = 1.0

        # The fine-grid smb survives the checkpoint only for steps that emit a
        # recorded snapshot (the final step always does): ModelState.smb_fine
        # feeds the extent/snowline misfits there. Every other step keeps just
        # the level-restricted field the dynamics consume.
        want_fine = (t_next == steps[-1][0]) or any(
            abs(t_next - tt) < 1e-6 for tt in record_states_at)
        if smb_kind == "enthalpy":
            ec = enthalpy_consts
            smb_fn = compute_smb_enthalpy
            smb_args = (smb_model,
                        t2m, tbias, anomaly_terms, base_anomaly, precip_, precip_multiplier,
                        insol_mean, insol_dif, t_base,
                        H_atm, ec["H_base0"], ec["q_sw_bulk"], q_sw_insol, q_sw_dif,
                        ec["q_lw0"],
                        ec["albedo_snow"], ec["albedo_ice"], ec["M_albedo"],
                        debris, temp_dev, domain_mask, level, want_fine, yearly_climate)
        else:
            smb_fn = compute_smb
            smb_args = (smb_model,
                        t2m, tbias, anomaly_terms, base_anomaly, precip_, precip_multiplier,
                        debris,
                        mf, rf, domain_mask, level, want_fine, yearly_climate)

        if ocean_forcing is not None:
            q_f, h0_f, _ = ocean_forcing.margins(t_prev, t_next)
            model.mg.calving.q.set(_restrict_cupy(q_f, level), start_level=level)
            model.mg.calving.h0.set(_restrict_cupy(h0_f, level), start_level=level)

        if thermal is not None:
            thermal.pre_step(H_prev_)
        if no_grad_step:
            # Truncated-backprop spin-up: full physics, no graph - so no
            # checkpoint (it would only warn about grad-free inputs), no
            # adjoint solve in backward, no retained per-step state.
            with torch.no_grad():
                out = smb_fn(*smb_args)
                smb_, smb = out if want_fine else (out, None)
                u, v, ud, vd, H, active = glide_step(
                    cp.float32(t_prev), cp.float32(dt_step),
                    model, level, H_prev_, bed_, beta_, smb_)
        else:
            out = checkpoint(smb_fn, *smb_args, use_reentrant=False)
            smb_, smb = out if want_fine else (out, None)
            u, v, ud, vd, H, active = glide_step(
                cp.float32(t_prev), cp.float32(dt_step),
                model, level, H_prev_, bed_, beta_, smb_)
        if thermal is not None:
            thermal.post_step(dt_step)
        # `H_prev_` here is the thickness emitted by the previous step (the input
        # to this one); capturing it before the reassignment leaves it holding
        # the second-to-last emitted thickness once the loop ends.
        H_penult = H_prev_
        H_prev_ = H
        t_prev = t_next

        state = ModelState(
            t=t_next, dt_step=dt_step, level=level, u=u, v=v, ud=ud, vd=vd,
            H=H, active=active,
            smb_fine=smb, smb_coarse=smb_, bed_coarse=bed_,
            flotation_factor=flotation_factor, n_glen=n_glen,
        )

        # Record snapshots/volumes keyed by the caller's exact requested floats.
        for tt in record_states_at:
            if abs(t_next - tt) < 1e-6 and tt not in states:
                states[tt] = state
        for yr in record_volumes_at:
            if abs(t_next - yr) < 1e-6 and yr not in volumes:
                volumes[yr] = torch.sum(H * (dx_fine * 2 ** level) ** 2)

        if time_writer is not None:
            time_writer.append(model.mg[level], time=float(t_next))
            time_writer.write_pvd()

    states[state.t] = state  # the final state is always available

    return SimResult(
        states=states,
        final=state,
        volumes=volumes,
        H_prev=H_penult,
        H_boundary=H_boundary,
    )
