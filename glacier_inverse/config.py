"""
Configuration dataclasses for glacier inverse problems.

A single GlacierConfig instance captures every numerical knob — physical
hyperparameters, prior hyperparameters, observation noise, solver tolerances,
loss weights, paths — so the four tasks (inverse, rto, posterior, sensitivity)
share the same physical model by construction.
"""
import inspect
from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Union
import numpy as np

# The GlacierConfig fields that may be given as a schedule instead of a constant.
# Per-observation weights are also schedulable, but live on the observation
# specs (see observations.py) and are resolved by Observation.weight_at.
SCHEDULABLE_WEIGHTS = ("loss_scale",)

# Per-parameter learning rates follow the same contract: a constant, or a
# Schedule(final=, ramp=) honored only by the initial MAP solve (inverse.py
# refreshes each optimizer param group's lr from GlacierConfig.learning_rates
# every iteration). RTO warm-starts from the MAP and uses `final`. Setting a
# ramp to 0.0 freezes that parameter for those iterations, e.g.
#   lr_z_log_mf=Schedule(final=0.01, ramp=lambda i: 0.0 if i < 100 else 0.01)
# holds the melt factor for the first 100 iterations of every level.
SCHEDULABLE_LRS = (
    "lr_z_bed", "lr_z_bed_mean", "lr_z_log_beta",
    "lr_z_pbias", "lr_z_tbias", "lr_z_log_mf", "lr_z_log_rf",
    "lr_z_log_H_atm", "lr_z_logit_cloud",
    "lr_z_tau", "lr_z_z0",
)


def _accepts_two_positional(fn: Callable) -> bool:
    """True if `fn` can be called with two positional args, f(i, level); False if
    it only takes one, f(i). Used to support both schedule arities."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (ValueError, TypeError):
        return False  # builtins without an introspectable signature: assume f(i)
    positional = [p for p in params
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    has_var_positional = any(p.kind is p.VAR_POSITIONAL for p in params)
    return has_var_positional or len(positional) >= 2


@dataclass(frozen=True)
class Schedule:
    """A scalar hyperparameter (loss weight or learning rate) that follows a
    continuation ramp during the initial MAP solve only, collapsing to a single
    steady-state value everywhere else.

    `final` is the contract value shared by all four tasks: RTO, posterior, and
    sensitivity always use it, so they target the same objective (and, for
    learning rates, the same optimizer step) the MAP solve converges to. `ramp`
    is a callable f(i) or f(i, level) honored only when a driver opts into
    scheduling (inverse.py); it should asymptote to `final`. Note `i` resets at
    each multigrid level and `level` runs coarsest -> 0 (finest).
    """
    final: float
    ramp: Callable[..., float]

    def at(self, iteration: int, level: int) -> float:
        fn = self.ramp
        return fn(iteration, level) if _accepts_two_positional(fn) else fn(iteration)


# A schedulable scalar. `schedule=` on GlacierConfig.at_iteration selects
# whether continuation is honored (the initial inverse solve) or the steady-state
# value is used (RTO / posterior / sensitivity). Loss weights and learning rates
# share the type and the resolution rule.
Scheduled = Union[float, Schedule, Callable[..., float]]
LossWeight = Scheduled
LearningRate = Scheduled


def resolve_weight(value: Scheduled, iteration: int, level: int, *,
                   schedule: bool, what: str = "loss weight") -> float:
    """Resolve a (possibly scheduled) scalar (loss weight or learning rate) to
    a float.

    * constants pass through unchanged;
    * a Schedule yields its continuation value when `schedule` is True, else its
      steady-state `final`;
    * a bare callable is an inverse-only shorthand: honored when `schedule` is
      True, but rejected otherwise — it declares no steady state, so RTO and the
      analysis tasks cannot use it consistently (wrap it in Schedule(final=...)).
    """
    if isinstance(value, Schedule):
        return value.at(iteration, level) if schedule else value.final
    if callable(value):
        if not schedule:
            raise TypeError(
                f"{what} is a bare schedule callable, but this task does not "
                "use scheduling. Wrap it as Schedule(final=..., ramp=...) so the "
                "steady-state value is defined for RTO / posterior / sensitivity."
            )
        return value(iteration, level) if _accepts_two_positional(value) else value(iteration)
    return value


@dataclass(frozen=True)
class PriorHyperparams:
    sigma: float
    l: float
    nu: int


@dataclass(frozen=True)
class MaternNoise:
    """Correlated (Matérn GP) observation-error model for a field likelihood.

    Attach to `SurfaceSpec`/`VelocitySpec`/`DhdtSpec(noise=...)`. The residual
    field r = model − data is treated as a draw from a zero-mean Matérn GP
    with marginal std `sigma`, correlation length `l` and smoothness `nu`,
    and the misfit is formed on the *whitened* residual z = W r (ggapp's
    `GGaPPWhiten`, a stencil application), so that
        loss = loss_scale · weight · Σ huber(z)
    with NO dx² observation-density factor and `weight == 1` by contract
    (`GlacierProblem` raises otherwise — any up/down-weighting must be a
    change to σ, l or ν, i.e. to the probabilistic model).

    * `sigma`: marginal std of the error in the product's units (m for the
      surface, m/yr per velocity component). For dh/dt it is a dimensionless
      multiplier on the product's own per-pixel error (clamped at
      `sigma_floor`), so σ=1 trusts the reported uncertainty as the marginal.
    * `l`: correlation length in metres, ggapp convention κ = √(8ν)/l (the
      correlation drops to ≈0.1 at distance l for ν = 1; for ν = ½ it is
      exp(−2h/l)). An exponential variogram range a (γ = c0 + c1(1 −
      e^{−h/a})) corresponds to l ≈ 2.3 a for ν = 1 and l = 2a for ν = ½.
    * `nu`: Matérn smoothness, any positive real. ggapp's stencil whitening
      needs a positive ODD integer (α = ν + 1 even; it truncates α//2
      silently otherwise); every other ν — in particular ν = ½, the
      exponential covariance that the residual variograms actually fit — is
      handled exactly by the spectral path (`priors.SpectralMaternNoise`),
      as is any ν with a nugget. Mind the spectral tail: ν = 1 falls as k⁻⁴
      and declares intermediate (glacier-scale) model error impossible, so
      those scales get counted against the nugget alone; ν = ½ (k⁻³) keeps
      real power there. This is NOT the spec's `nu`, which is the
      pseudo-Huber threshold on the whitened residual.
    * `nugget`: std of an additional WHITE (pixel-scale) error component, same
      units as `sigma` — error = Matérn(sigma, l, nu) + N(0, nugget²) per
      pixel. Real products have one (DEM/mosaic pixel noise, the reported
      dh/dt error), and a pure Matérn with a long l declares pixel-scale
      noise impossible: its whitening filter amplifies white noise by
      ~(dx/τ)·‖L‖ (≈45× for l = 9 km at 90 m), so the whitened residual is
      dominated by amplified noise and the term fits noise. With a nugget the
      model is handled as an exact spectral (DCT-II) operator on the Neumann
      grid (`priors.SpectralMaternNoise`) instead of ggapp's stencil — the
      same discretization, diagonalized. Read it off the variogram: nugget² ≈
      the fitted c0 (in the product's units); the post-hoc check is
      `tools/residual_variograms.py` (std(z) → 1, γ_z(dx)/var → 1).

    * `discrepancy`: an optional SECOND Matérn component at the domain/
      synoptic scale — C = C_Matérn(σ, l, ν) + nugget²·I +
      C_Matérn(σ_D, l_D, ν_D) — the exact (conjugate) marginalization of a
      Kennedy–O'Hagan model-discrepancy field δ ~ GP(0, σ_D, l_D): a
      coherent offset between prediction and product, trusted only to σ_D
      per l_D-sized patch. Whitening by the sum is spectrally-shaped
      tempering η(k) = C(k)/(C(k)+C_D(k)): fine scales (which constrain
      bed/β) keep full weight, and only the near-constant modes — where a
      variogram from ONE domain realization carries no information, so the
      base model's k→0 confidence is extrapolation, not inference — are
      discounted. The data's information about the product/prediction LEVEL
      is thereby capped at ~one observation of error σ_D per l_D patch
      instead of N_eff, which is what keeps a domain's few smooth SMB knobs
      from being pinned through the domain-mean channel. The discrepancy
      component itself must have nugget = 0 and no nested discrepancy; its
      presence forces the spectral path.

    Duck-compatible with `PriorHyperparams` (same first three attributes).
    """
    sigma: float
    l: float
    nu: float = 1
    nugget: float = 0.0
    discrepancy: Optional["MaternNoise"] = None

    def __post_init__(self):
        if not (self.sigma > 0.0):
            raise ValueError(f"MaternNoise.sigma must be > 0, got {self.sigma}")
        if not (self.l > 0.0):
            raise ValueError(f"MaternNoise.l must be > 0, got {self.l}")
        if not (self.nugget >= 0.0):
            raise ValueError(f"MaternNoise.nugget must be >= 0, got {self.nugget}")
        if isinstance(self.nu, bool) or not (self.nu > 0.0):
            raise ValueError(f"MaternNoise.nu must be a positive real, got {self.nu!r}")
        if self.discrepancy is not None:
            d = self.discrepancy
            if not isinstance(d, MaternNoise):
                raise ValueError(
                    f"MaternNoise.discrepancy must be a MaternNoise, got "
                    f"{type(d).__name__}")
            if d.nugget != 0.0 or d.discrepancy is not None:
                raise ValueError(
                    "MaternNoise.discrepancy is a single smooth Matérn "
                    "component: it carries no nugget (the base model's "
                    "nugget prices pixel noise) and no nested discrepancy.")

    @property
    def stencil_compatible(self) -> bool:
        """True when ggapp's stencil whitening applies exactly: no nugget,
        no discrepancy component, and ν a positive odd integer. Otherwise
        the spectral path is used."""
        nu = self.nu
        return (self.nugget == 0.0 and self.discrepancy is None
                and float(nu).is_integer()
                and int(nu) >= 1 and int(nu) % 2 == 1)


@dataclass(frozen=True)
class FingerprintNuisance:
    """Rank-few model-error marginalization along the smooth SMB parameters'
    sensitivity fingerprints.

    Motivation: the field likelihoods' Gauss–Newton information about the
    smooth enthalpy parameter fields (log H_atm, logit clear-sky f) is
    10⁴–10⁶× their priors' — but it rides on glacier-scale sensitivity
    structure whose model error is exactly the thing we know exists, so the
    counting is unearned (measured 2026-09: the l_D = 80 km level
    discrepancies discounted 0–9% of it). Fix: admit a nuisance
    `Σ_j c_j·g_j`, `c_j ~ N(0, s²)`, where `g_j = ∂(residual)/∂(mode_j)` are
    the MEASURED fingerprints of the parameter's leading prior modes — "there
    may be model error whose imprint mimics this parameter". Marginalizing
    the Gaussian c (profiled with the pseudo-Huber, envelope gradient — the
    LogitNuisance pattern, k-dimensional instead of field-valued) caps the
    data-side information about mode j at 1/s² in prior-standardized units.

    Since the mode perturbations are taken in WHITENED coordinates (unit
    prior std per mode), `s` reads as "model error up to s× the parameter's
    own prior, mode for mode": s = 1 means the data may at most double the
    prior precision on these modes — the honest few-knobs/few-observations
    regime. Attribution phenomenology per Brynjarsdóttir & O'Hagan (2014);
    span geometry mirrors Plumlee (2017)'s orthogonal discrepancy (sign of
    intent flipped); the downdate algebra is astrophysics template
    marginalization (Rybicki & Press 1992; van Haasteren & Levin 2013).

    Applied to the whitened surface and dh/dt likelihoods (velocity's surge
    marginal needs its own treatment — its per-glacier η already absorbs
    glacier speed levels). Fingerprints are measured by finite differences
    (1 + len(params)·n_modes forwards) from the CURRENT state:
    `refresh = 0`/None re-measures at the start of each multigrid level;
    `refresh = N > 0` additionally every N iterations. A stale fingerprint
    degrades conservatively (less forgiveness, never more). The fitted ĉ
    (per term, in prior-std units) is the audit trail: it reads "this term
    attributes ĉ× the prior's worth of this parameter's pattern to model
    error rather than physics".
    """
    params: tuple = ("log_H_atm", "logit_cloud")
    s: tuple = (1.0, 1.0)      # per-param, in units of the param's prior std
    n_modes: int = 4           # leading prior modes per parameter
    refresh: Optional[int] = 0  # 0/None: each level start; N>0: also every N iters
    fd_step: float = 0.5       # FD step, prior-std units

    def __post_init__(self):
        if len(self.s) != len(self.params):
            raise ValueError(
                f"FingerprintNuisance: len(s)={len(self.s)} must match "
                f"len(params)={len(self.params)}")
        if any(not (v > 0.0) for v in self.s):
            raise ValueError("FingerprintNuisance: every s must be > 0")
        if self.n_modes < 1:
            raise ValueError("FingerprintNuisance: n_modes must be >= 1")
        if not (self.fd_step > 0.0):
            raise ValueError("FingerprintNuisance: fd_step must be > 0")


@dataclass(frozen=True)
class SolverConfig:
    """FAS V-cycle schedule plus the Vanka/Newton smoother options, applied
    to one of glide's solvers (forward or adjoint). Defaults are glide's
    examples/greenland settings (2026-09 refactor): a post-smoothing-heavy
    cycle with no finest-level sweeps, omega 0.5, momentum damping 0.01."""
    coarsest_steps: int = 200
    pre_steps: int = 10
    post_steps: int = 150
    finest_steps: int = 0
    relative_tolerance: float = 1e-2
    absolute_tolerance: float = 10.0
    report_norms: bool = False
    # Vanka smoother: under-relaxation of the patch update and the Newton
    # options inside each patch solve (vanka_options.omega,
    # vanka_options.newton_options.{momentum_damping, step_tolerance}).
    omega: float = 0.5
    momentum_damping: float = 0.01
    step_tolerance: float = 1e-6


@dataclass(frozen=True)
class OceanForcingConfig:
    """Ocean thermal forcing -> calving margins (see glacier_inverse/ocean.py):
    q = calving_q + alpha_q * dTF, h0 = calving_h0 + alpha_h * dTF per step,
    with dTF the step's TF anomaly (annual `statistic` aggregated over the
    years the step overlaps, minus the per-cell mean over `ref_years` or the
    scalar `ref_value`), zero farther than `max_dist_km` from the native
    product and before the record. alpha_q (1/K) shifts the threshold by a
    fraction of the thickness, alpha_h (m/K) by the same distance everywhere;
    both are sweep parameters, not differentiated. Needs
    model_inputs/<filename> (preprocessing/make_thermal_forcing.py); when
    `enabled` and the file is absent the run warns and keeps constant margins."""
    enabled: bool = False
    filename: str = "thermal_forcing.nc"
    statistic: str = "max"          # "max" (annual max of monthly TF) | "mean"
    ref_years: tuple = (1950, 1979)
    ref_value: Optional[float] = None
    max_dist_km: float = 5.0
    alpha_q: float = 0.0            # per K
    alpha_h: float = 0.0            # m per K
    q_bounds: tuple = (-1.0, 1.0)
    h0_bounds: tuple = (-1000.0, 1000.0)


@dataclass(frozen=True)
class BedConditioningConfig:
    """GP-posterior-as-prior for the bed: condition the Matern bed prior on
    the flightline picks and (optionally) bed=DEM over every ice-free /
    out-of-domain pixel, so z_bed parametrizes the CONDITIONAL field
    (Matheron's rule: bed = bed_mean + Map(z_bed) + (Q+D)^-1 D (b - ...)).

    Enabling this CHANGES THE BED PARAMETRIZATION:
      - checkpoints carry a "bed_parametrization" tag and are converted
        exactly on load (io.load_whitened_params_into needs `priors=`);
      - the BedObservation soft likelihood must carry weight 0 (the data is
        already in the map — GlacierProblem warns loudly otherwise);
      - lr_z_bed generally needs per-domain retuning: the conditional map has
        different curvature near data. Start from the legacy value and watch
        the first level.

    The correction is solved matrix-free by flexible PCG (preconditioned with
    the prior multigrid solve); cold iteration counts scale like
    sigma_prior/sigma_obs, so keep the sigmas ~10 m — conditioning at +-10 m
    already pins the bed far harder than the old soft anchor did (do NOT port
    a sigma_dem=1.0 soft-anchor experiment here; it only buys PCG iterations).
    """
    enabled: bool = False
    sigma_picks: float = 10.0    # m, per-pick noise (BedSpec.sigma semantics)
    sigma_dem: float = 10.0      # m, off-ice bed=DEM noise (BedSpec.sigma_dem)
    include_off_ice: bool = True
    # Gridded bed observations (Greenland: BedMachine `bed` + `errbed`,
    # carried into GLIDE_inputs.nc as `bed_obs` / `bed_obs_err` by
    # preprocessing/make_dem.py). Every finite on-ice cell contributes
    # bed = bed_obs at precision 1/(scale * max(errbed, floor))^2 — the
    # per-cell error field replaces the single sigma_picks of the flightline
    # route. Cells with errbed > max_err (None = keep all) are dropped, so a
    # domain can condition only on radar-constrained bed (BedMachine's
    # mass-conservation cells carry errbed ~ 30-100 m, interpolated interior
    # cells several hundred m). Silently inactive when the variables are
    # absent (Alaska-style flightline-only domains).
    use_gridded_bed: bool = True
    gridded_bed_err_floor: float = 10.0   # m
    gridded_bed_err_scale: float = 1.0    # multiplier on errbed
    gridded_bed_max_err: Optional[float] = None   # m; None keeps every cell
    # Which gridded cells count as bed DATA:
    #   "all"   — every BedMachine cell at its errbed (v6: 64% kriging, 22%
    #             mass conservation, 9% IceBoost, 5% interpolation, 4% radar;
    #             errbed is NOT a distance-to-data proxy — the kriged
    #             interior sits at a 30 m floor 100 km from any track, so
    #             gridded_bed_max_err cannot isolate the observed cells).
    #   "radar" — only cells whose BedMachine subcells hold a radar pick
    #             (dataid == 2; `bed_radar_fraction` > radar_fraction_min),
    #             at the pick-averaged bed / errbed (`bed_obs_radar`,
    #             `bed_obs_radar_err`). Everywhere else D = 0: the bed is
    #             the Matern fluctuation about bed_mean, informed only by
    #             the flow likelihood — mass conservation as inference, and
    #             no double counting of the velocity / thinning data that
    #             BedMachine's MC bed was itself built from.
    gridded_bed_data: str = "all"
    radar_fraction_min: float = 0.0       # keep cells with fraction > this
    # "radar" mode only: seed the unconditional bed Map(z_bed) OFF the radar
    # tracks from the smoothed bed (the bed_mean start) instead of from
    # BedMachine's interpolated bed, so between-track structure comes purely
    # from the prior and the flow likelihood rather than from BedMachine's
    # warm start. Radar cells keep their picks either way.
    seed_off_track_from_mean: bool = False
    # Cells where the DEM is exactly 0.0 are void fill at the land-DEM /
    # bathymetry seam (nearshore, fjord heads), not real bed — anchoring them
    # pins fjord bottoms at sea level. Excluded by default: the conditional
    # field there is interpolated from the surrounding anchored pixels, and a
    # spurious sill near a terminus gets corrected by the surface misfit
    # (ice overrides it -> surface too high -> bed pushed down). Real land at
    # exactly 0.0f is essentially nonexistent in float composites.
    exclude_zero_dem: bool = True
    pcg_rtol: float = 1e-4
    # Backward-pass tolerance: a ~1% gradient is ample for SGD, and rough
    # loss cotangents stall float32 PCG well before 1e-4 (burning maxiter
    # every iteration — the dominant cost otherwise).
    pcg_rtol_adjoint: float = 1e-2
    # "shifted" (default): precondition with the diagonally shifted factor
    # (L + d)^-2, d = (tau/dx)sqrt(D) — O(10) iterations for every solve,
    # independent of the sigma ratio and rhs roughness, at one extra
    # multigrid hierarchy of memory. "prior": the original C preconditioner
    # (kept for comparison/fallback; iterations ~ sigma_prior/sigma_obs).
    pcg_preconditioner: str = "shifted"
    # Cold solves at sigma_prior/sigma_obs = 25 land around 250-500 iterations
    # (rough right-hand sides are the worst); warm-started solves during
    # optimization take a handful.
    pcg_maxiter: int = 800
    warm_start: bool = True


@dataclass(frozen=True)
class GlacierConfig:
    base_dir: str

    # Grid / multigrid
    n_levels: int = 6

    # Time stepping. `dt` is the maximum step of the uniform spin-up grid;
    # `dt_schedule` = ((t_from, dt), ...) refines it from t_from onward
    # (equal sub-steps <= dt between consecutive observation epochs; see
    # scheduling.build_step_sequence), e.g. ((1990.0, 3.0),) for the
    # observational period.
    dt: float = 20.0
    dt_schedule: tuple = ()
    t_start: float = 1012.0
    t_end: float = 2012.0

    # Truncated backpropagation through time: steps ending at or before this
    # time run under torch.no_grad - full spin-up physics (J is unchanged),
    # no adjoint solves, no checkpointed state. None (default) differentiates
    # the whole run.
    #
    # CAUTION - the misfit forcing decaying to nothing backward in time does
    # NOT mean the deep-time chain contributes nothing: for time-constant
    # parameters (beta, bed) the per-step contributions accumulate coherently
    # over the glacier response time, and an FD test on delta (level 2,
    # dt=20) confirms the deep content is real gradient signal. The cutoff
    # must precede the first observation epoch by several response times.
    # Measured gradient error on delta (epochs 2000-2020): cutoff 1500 ->
    # 0.5% (beta) / 0.2% (bed) at ~45% backward savings; 1700 -> 7%/3%;
    # 1850 -> 31%/13%; 1950 -> 82%/31%. Rerun that sweep (vary this value,
    # compare grads against None) when adopting it on a new domain.
    #
    # Guards: simulate() raises if any recorded state/volume time falls in
    # the no-grad window, and SimResult.H_boundary.grad (populated by
    # backward) is the adjoint seed of everything discarded - it tracks the
    # error monotonically (delta: |.|~2e-5 at 0.5% error, ~9e-4 at 30%), so
    # watch it drift as the optimizer moves the state. Shared by all
    # gradient-computing tasks (MAP, RTO, sensitivity) so they optimize the
    # same objective approximation.
    grad_start_time: Optional[float] = None
    # Year whose anomaly is subtracted from the series (the climatology's
    # epoch). None: the anomaly file is already referenced to the
    # climatology window (mean zero over it) and no year is subtracted.
    base_anomaly_year: Optional[int] = 2012
    alpha_t2m: float = 2.5
    
    base_precip_year: int = 2012
    alpha_precip: float = 0.0

    # How the multi-year anomaly signal is integrated over each ice-dynamics
    # step. The anomaly record is piecewise constant per calendar year and smb
    # is an exogenous source, so the correct discrete forcing is its exact
    # interval integral, not a point sample:
    #   "end"          — legacy: sample the anomaly at the step's end time and
    #                    hold it for the whole step. Aliases interannual
    #                    variability and makes the forcing depend on how the
    #                    scheduler partitions time; kept for comparison.
    #   "mean_anomaly" — one SMB evaluation per step at the overlap-weighted
    #                    interval-mean anomaly. Fixes aliasing and partition
    #                    dependence; retains a small Jensen bias (smb is
    #                    nonlinear in temperature). The default.
    #   "annual"       — exact: one SMB evaluation per calendar year overlapped
    #                    by the step, combined with overlap weights (years with
    #                    identical clamped anomalies merge into one call). Cost
    #                    scales with simulated years rather than steps.
    # The precip-anomaly multiplier follows the same weights ("end" keeps its
    # legacy endpoint trapezoid) and is held at its step mean in every mode.
    anomaly_integration: str = "mean_anomaly"

    # Field priors (Matern)
    bed_prior:      PriorHyperparams = PriorHyperparams(sigma=500.0,    l=2000.0,  nu=1)
    mean_prior:     PriorHyperparams = PriorHyperparams(sigma=1000.0,   l=10000.0, nu=1)
    log_beta_prior: PriorHyperparams = PriorHyperparams(sigma=1./3.,      l=1000.0,  nu=1)
    pbias_prior:    PriorHyperparams = PriorHyperparams(sigma=0.1,     l=10000.0, nu=1)
    tbias_prior:    PriorHyperparams = PriorHyperparams(sigma=0.1,     l=10000.0, nu=1)
    # Priors for the enthalpy SMB parameters, which are (ny, nx) GP FIELDS
    # (log H_atm and logit clear-sky fraction; medians mu_H_atm /
    # mu_cloud_factor below). sigma is the pointwise marginal std in log /
    # logit space — the same number the old scalar priors used, so at a
    # single point the prior is unchanged; what the field relaxes is the
    # perfect-correlation-across-the-domain assertion of a scalar. l is
    # anchored to the physical decorrelation scale of what the parameter
    # lumps (synoptic cloudiness, orographically organized transfer:
    # ~50-100 km), NOT to the domain size — that is what makes per-parameter
    # curvature, learning rates, and prior information domain-size invariant
    # and the hyperparameters transferable across ranges. nu must be a
    # positive odd int (ggapp stencil). Active only under
    # smb_model="enthalpy" (the Matern members are not built otherwise).
    h_atm_prior:    PriorHyperparams = PriorHyperparams(sigma=0.2,     l=80000.0, nu=1)
    cloud_prior:    PriorHyperparams = PriorHyperparams(sigma=0.25,    l=80000.0, nu=1)
    # Opt-in rank-few model-error marginalization along the smooth SMB
    # parameters' measured sensitivity fingerprints (see FingerprintNuisance).
    fingerprint_nuisance: Optional["FingerprintNuisance"] = None
    # Semi-modular influence eta of the DATA on the enthalpy SMB parameter
    # block (z_log_H_atm, z_logit_cloud) in the MAP solve: the fields see the
    # full posterior; this block sees prior * likelihood^eta GIVEN the fields
    # — a parameter-blocked semi-modular posterior (Carmona & Nicholls 2020;
    # belief-update coherence per Bissiri, Holmes & Walker 2016). eta = 1 is
    # full Bayes (bit-identical, the surgery is skipped); eta = 0 is the cut
    # posterior (SMB block calibrated by the prior alone). Motivation: the
    # melt channels are misspecification-dominated — the measured data pull
    # on these parameters is ~10^2 per prior std against the prior's 1
    # (tools pull table), an exchange rate the wrong model has not earned.
    # Generative shadow: eta equals the saturated-regime marginalization of a
    # theta-mimicking model-error nuisance with prior scale s = 1/sqrt(eta*I).
    # Choose eta so the block's data and prior pulls are comparable:
    # eta ~ 1/(measured pull per prior-std); verify with the pull table at
    # the new equilibrium. Applied as EXACT gradient surgery in inverse.py
    # (the whitened prior gradient is analytic: loss_scale * z), so unlike a
    # learning rate it changes the block's equilibrium, not its speed. The
    # equilibrium is a fixed point of a non-conservative field (same formal
    # status as the warm-started profiled nuisances); the pull-table
    # stationarity test still applies blockwise. RTO must apply the same eta
    # when it is migrated (noted in rto_sample.py).
    smb_data_influence: float = 1.0


    influence_cap: dict = None,
    influence_transfer: str = 'log',

    # Scalar prior mean of the log_beta field: the Matern prior (and its
    # whitened representation) applies to log_beta - mu_log_beta, so the
    # zero-loss state is beta = exp(mu_log_beta) rather than beta = 1.
    # Checkpointed z_log_beta is relative to this mean — changing it shifts
    # the physical field a warm start maps to.
    mu_log_beta: float = np.log(5.0)

    # Optional additive temperature bias field (units: K). A Matern GP field
    # added to the monthly t2m before the anomaly shift — the spatial,
    # time-constant complement of the (uniform, time-varying) anomaly record.
    # Its natural role is absorbing error in the preprocessing lapse
    # correction (fixed 6.5 K/km against a fixed DEM) and the DEM-vs-model-
    # surface mismatch. Unlike pbias it is not logarithmized: it acts
    # additively and may be negative. Gated off by default so already-tuned
    # domains are bit-identical: when disabled, z_tbias stays at 0 (= prior
    # median = 0 K), never enters the optimizer or the forward graph, and no
    # Matern hierarchy is built for it. Caveats when enabling:
    #   * lr_z_tbias is coupled to tbias_prior like every other lr/prior pair;
    #   * t2m becomes a differentiable SMB input, so each checkpoint-segment
    #     backward transiently copies a (12, ny, nx) t2m gradient off the
    #     glare grid;
    #   * the spatially uniform component is degenerate with the anomaly
    #     scaling (alpha_t2m) over the calibration window — the GP prior is
    #     what pins it, exactly as for pbias;
    #   * under the enthalpy backend, tbias shifts t2m ONLY: t_base (the
    #     static substrate proxy) deliberately stays at the unbiased
    #     climatology, matching how the temperature anomaly is treated.
    tbias_enabled: bool = True

    # SMB backend: "temperature_index" (glare's ImprovedTemperatureIndex, the
    # default) or "enthalpy" (glare's EnthalpyModel — an enthalpy formulation of
    # annual snow dynamics; same forcing, same specific SMB, different
    # parameters). Both models' parameters and priors live in this config
    # simultaneously; only the active model's scalars enter the optimizer and
    # the forward graph (the inactive model's whitened z stay at 0 = prior
    # median, contributing exactly zero to the prior loss).
    smb_model: str = "temperature_index"

    # Scalar SMB priors (log-normal). mu_log_* is derived as log(mu_*).
    # These belong to the temperature-index model.
    mu_rf: float = 50.0
    mu_mf: float = 1.0
    sigma_log_rf: float = 0.1
    sigma_log_mf: float = 0.1
    debris_factor: float = 0.5 # Amount by which debris cover reduces melt of bare ice.

    # Enthalpy-model inverted scalars. H_atm (lumped sensible/longwave heat
    # transfer) is inferred as log(H_atm in W m-2 K-1); the shortwave is
    # inferred through logit(f) of the CLEAR-SKY FRACTION f = 1 - cloud
    # fraction in (0, 1): direct q_sw_insol = f * q_sw_clear (the direct-beam
    # potential I already carries the clear-sky tau^airmass attenuation, so
    # q_sw_clear is the extraterrestrial S0) and diffuse
    # q_sw_dif = (f * k_diffuse_clear + (1 - f) * k_diffuse_cloud) * q_sw_clear,
    # both from the same f. Physical values passed to the model are converted
    # to J m-2 yr-1 (K-1) via SECONDS_PER_YEAR.
    mu_H_atm:          float = 15.0   # W m-2 K-1, prior median (of the FIELD)
    # LEGACY scalar sigma: the pointwise prior std now lives in
    # h_atm_prior.sigma. This value is used only (a) to convert pre-field
    # checkpoints (their 0-d z is de-whitened with THIS sigma before being
    # re-whitened into the field parametrization) and (b) as the affine
    # fallback in physical_from when no Matern member exists (ETIM domains,
    # where the field sits inert at 0 anyway).
    sigma_log_H_atm:   float = 0.2
    # Prior median of the clear-sky fraction f. 0.35 ~ interior-Alaska summer
    # cloud fraction 0.65; with k_diffuse_* below it reproduces the station
    # June climatology (horizontal direct ~100, diffuse ~120, global ~220 W m-2
    # at 63 N). Maritime domains sit lower (~0.2).
    mu_cloud_factor:   float = 0.35
    # LEGACY scalar sigma — same status as sigma_log_H_atm (conversion +
    # fallback only); the live pointwise std is cloud_prior.sigma.
    sigma_logit_cloud: float = 0.25

    # Enthalpy-model fixed constants (not inverted; per-second SI units,
    # converted by SECONDS_PER_YEAR where they are fluxes).
    # Extraterrestrial normal irradiance S0. The clear-sky attenuation
    # tau^(airmass * p/p0) lives in the insolation potential (gtic), NOT here.
    q_sw_clear:  float = 1361.0  # W m-2, extraterrestrial (TOA) normal shortwave
    # Diffuse-sky fractions of S0 * cos(zenith) on an unobstructed horizontal
    # surface: clear-sky diffuse (~10-15% of clear-sky global; 0.10 at 1-2 km
    # elevation) and overcast global (all diffuse; ~0.30 of TOA). The monthly
    # diffuse potential I_dif (sky-view factor x mean cos zenith, from
    # make_insolation.py) multiplies q_sw_dif in the balance.
    k_diffuse_clear: float = 0.10
    k_diffuse_cloud: float = 0.30
    q_sw_bulk:   float = 0.0     # W m-2, insolation-independent shortwave
    # Constant flux into the surface that is NOT albedo-scaled and NOT
    # proportional to (T_air - T_s): the dT-independent part of net longwave /
    # latent exchange (clear-sky longwave deficit -(1-eps_a) sigma T^4,
    # evaporation into sub-saturated air). Negative cools; 0 reproduces the
    # original balance. Without it a calibrated H_atm absorbs the offset and
    # comes out too small (flattening the melt-temperature sensitivity), so
    # read mu_H_atm as the dT slope *given* this offset. A-priori interior
    # Alaska summer value ~ -40, maritime ~ -20; a monthly field from CARRA
    # downward longwave is the intended replacement.
    q_lw0:       float = -40.0     # W m-2, constant non-albedo-scaled surface flux
    H_base0:     float = 0.6     # W m-2 K-1, basal conductance at zero snow mass
    albedo_snow: float = 0.9
    albedo_ice:  float = 0.4
    M_albedo:    float = 20.0    # kg m-2, snow-cover albedo transition mass
    # Sub-monthly stochastic temperature deviations: one seeded (12, n_substeps)
    # realization is drawn once at problem build and reused for every time step,
    # iteration, and checkpoint recomputation, keeping the objective (and the
    # checkpointed adjoint) deterministic.
    enthalpy_n_substeps: int = 30
    enthalpy_seed:       int = 0

    # Keep glare's six diagnostic state cubes (M, E, runoff, ice_melt,
    # t_surface, albedo) individually resident. The inversion only consumes
    # smb, so by default they share one scratch buffer (~5 x (12, ny, nx) of
    # VRAM back); flip this on — or call
    # problem.smb_model.grid.materialize_state_fields() + one forward — when
    # the diagnostics are wanted. Ignored by glare versions predating the flag.
    enthalpy_materialize_state: bool = False

    # Elevation-dependent precip depletion (optional). Adds a softplus ramp,
    #   exp(-tau) * w * log(1 + exp((z - z0) / w)),
    # that is *subtracted* from the log-precip bias, reducing precip above the
    # onset elevation z0 to capture high-elevation moisture loss the climate
    # forcing misses. tau (log of the depletion length scale) and z0 are two
    # learnable scalars with normal priors N(mu, sigma) below; w is fixed. The
    # term is gated off by default so already-tuned domains are unaffected until
    # they set precip_lapse_enabled=True (and, since lr is coupled to the prior,
    # retune lr_z_tau / lr_z_z0 if they change the prior widths).
    precip_lapse_enabled: bool = False
    precip_lapse_w: float = 300.0   # m, fixed transition sharpness
    mu_tau:    float = 9.0          # prior mean of tau = log(depletion length scale [m])
    sigma_tau: float = 1.0
    mu_z0:     float = 2000.0       # m, prior mean of depletion-onset elevation
    sigma_z0:  float = 500.0

    # Observations. A tuple of frozen spec objects (SurfaceSpec, VelocitySpec,
    # ExtentSpec, BedSpec, SnowlineSpec, DhdtSpec — see observations.py), each
    # carrying its own noise sigma, loss weight (constant or Schedule), and
    # optional acquisition-time override; acquisition times normally come from
    # variable-level attrs on the input files. None selects the standard six
    # products with library-default hyperparameters. GlacierProblem builds the
    # loaded Observation objects from these specs; products whose input file
    # is absent are skipped gracefully.
    observations: tuple = None

    # Global loss scale. May be a constant, or a Schedule(final=, ramp=) for
    # continuation during the initial inverse solve only. The steady-state
    # `final` is the value RTO / posterior / sensitivity see, so all tasks
    # target the same objective; only inverse.py honors the ramp. Per-product
    # weights follow the same contract but live on the observation specs.
    loss_scale:  LossWeight = 1e-4

    # Ice rheology
    rho_ice:  float = 917.0
    # kg/m^3. Freshwater (1000) for Alaska's proglacial-lake termini; the
    # Greenland domains set seawater (1028) for tidewater/floating termini.
    rho_water: float = 1000.0
    gravity:  float = 9.81
    n_glen:   int   = 3
    A_glen:   float = 1e-16    # Pa^-n s^-1
    eps_reg:  float = 1e-6
    H_reg:    float = 50.0

    # Stress balance approximation: "ssa" (membrane stresses only, the
    # historical physics of this repo) or "molho" (adds vertical shear; the
    # velocity splits into a depth-averaged part u and a deformational part
    # ud, with surface velocity u + ud/(n+1)). SSA is the exact ud == 0
    # restriction of MOLHO, so downstream code is scheme-agnostic: ud/vd are
    # simply zero fields under "ssa". MOLHO costs roughly 2x per solve.
    stress_scheme: str = "ssa"

    # Sliding
    beta_init:   float = 5.0
    sliding_m:   float = 1.0 / 3.0
    u_reg:       float = 1.0
    water_drag:  float = 0.01

    # Calving / geometry (glide's signed-flotation model, 2026-09). Every
    # grounded/floating quantity derives from the flotation excess
    # z = rho_i/rho_w H - depth with depth = -bed (signed: negative on dry
    # land), so the grounded flag phi = sigmoid(sigmoid_c z) is exactly 1/2
    # at flotation and sigmoid_c is in 1/m. Ice calves where H < (1 + q) H_f
    # (calving flag psi = 0), losing thickness at H / timescale per year
    # until the active set pins it at thklim; timescale inf disables it.
    # calving_q < 0 keeps floating ice that is thicker than (1+q) H_f (ice
    # shelves / tongues), > 0 removes a margin above flotation.
    thklim:           float = 1.0     # minimum thickness (m), ice-free cells sit here
    sigmoid_c:        float = 1.0
    calving_timescale: float = 0.5    # years
    # Hybrid threshold: ice calves where H - H_f < calving_q * H + calving_h0
    # (a fraction of the thickness plus an absolute margin in metres); these
    # are the baselines that OceanForcingConfig perturbs with the TF anomaly.
    calving_q:        float = -0.5
    calving_h0:       float = 0.0     # m
    depth_blend:  float = 0.1    # weight on new bed-derived depth vs prior depth
    # Seed the integration from the thickness implied by the observed surface
    # (S_obs - bed, with the hydrostatic value for floating ice) instead of the
    # ice-free state. Matters for tidewater hysteresis. Not differentiated.
    init_from_observed_geometry: bool = False
    init_H_floor: float = 1.0    # minimum/ice-free thickness used when seeding (= thklim)
    use_avalanche_model: bool = False

    # Avalanche-operator hyperparameters (only read when use_avalanche_model).
    # s_crit/w_trans set the deposition sigmoid (degrees), p the MFD flow
    # concentration, K the number of routing passes (Neumann truncation). At
    # K=12 the undeposited residual on St.-Elias-like terrain is ~5e-4 of the
    # slab mass and is deposited in place, so larger K buys very little; the
    # historical hard-coded value was 25.
    avalanche_s_crit: float = 30.0
    avalanche_w_trans: float = 8.0
    avalanche_p: float = 1.5
    avalanche_K: int = 12

    # avalanche_hoisted — READ THIS BEFORE ENABLING.
    #
    # When True, the avalanche redistribution R is applied ONCE per forward
    # call, to the iteration's precip field (via glare's AvalancheStep),
    # instead of inside every per-step SMB evaluation (~150 applications per
    # inversion iteration once checkpoint recomputes and adjoint rebuilds are
    # counted — ~20 s/iteration on a St. Elias-sized grid). This is exactly
    # equivalent — not an approximation — while BOTH of the following hold:
    #
    #   1. smb_model == "enthalpy". The enthalpy core consumes *total* precip
    #      and partitions rain/snow internally, so R acts on a field that the
    #      per-step temperature shift never touches. Under ETIM the snow/rain
    #      partition sits UPSTREAM of R and depends on the (anomaly-shifted)
    #      per-step t2m, so hoisting is invalid there — GlacierProblem raises.
    #
    #   2. The per-step precip variation is a SCALAR multiplier (the annual
    #      precip-anomaly ratio). R is linear, so R(c*P) = c*R(P). If the
    #      precip forcing ever varies per step by more than a scalar multiple
    #      (e.g. monthly anomaly *fields*, storm-track reweighting), hoisting
    #      becomes silently WRONG — no runtime check can catch it. Revisit
    #      this flag whenever the precip pipeline changes.
    avalanche_hoisted: bool = False

    # Solver settings (glide examples/greenland: identical cycles for both
    # solvers; the adjoint variable is small in magnitude, hence its absolute
    # tolerance)
    forward_solver: SolverConfig = field(default_factory=lambda: SolverConfig(
        relative_tolerance=1e-2, absolute_tolerance=10.0, report_norms=False))
    adjoint_solver: SolverConfig = field(default_factory=lambda: SolverConfig(
        relative_tolerance=1e-2, absolute_tolerance=1e-5, report_norms=False))

    # Bed GP conditioning (posterior-as-prior). See BedConditioningConfig.
    bed_conditioning: BedConditioningConfig = field(
        default_factory=BedConditioningConfig)

    # Ocean thermal forcing of the calving margins. See OceanForcingConfig.
    ocean_forcing: OceanForcingConfig = field(default_factory=OceanForcingConfig)

    # Input filenames (relative to base_dir/model_inputs/)
    gridded_filename:    str = "GLIDE_inputs.nc"
    flightline_filename: str = "flightlines.gpkg"  # optional; radar bed picks
    anomaly_filename:    str = "temperature_anomaly.nc"
    precip_anomaly_filename: str = "precip_anomaly.nc"  # optional; multiplicative
    snowline_filename:   str = "gridded_snowline.nc"  # optional; ELA proxy
    debris_filename:   str = "gridded_debris.nc"  # optional; ELA proxy
    dhdt_filename:   str = "gridded_dhdt.nc"  # optional; surface elevation-change rate

    # Diagnostics
    vti_base_name: str = "glacier"

    # Experiment subdirectory (under base_dir). Writers (inverse) write here;
    # readers (posterior, sensitivity, rto continuation) read from here. Edit
    # the domain config to switch experiments rather than editing each driver.
    results_subdir: str = "inverse"

    # Multigrid schedule. max_level is the coarsest grid the solver starts on;
    # min_level is the finest grid it ends on. max_iters[level] is the number
    # of optimizer iterations spent at each level (indexed by the level number,
    # so unused entries below min_level can be 0 or any placeholder).
    min_level: int = 0
    max_level: int = 2
    max_iters: tuple = (20, 50, 500)

    # Per-parameter learning rates. These are tightly coupled to the prior
    # hyperparameters above — in whitened coordinates the natural step is set
    # by the prior curvature, so a domain that changes a prior typically has
    # to retune the corresponding lr. EVERY parameter is optimized by SGD in
    # whitened coordinates (prior-natural gradient: an SGD step of size lr in
    # z is a physical step -lr*C*grad, so updates are C-smoothed and
    # unconstrained directions relax to the prior mean). There is no Adam
    # block — Adam's per-coordinate RMS normalization equalizes step sizes
    # across coordinates, which in whitened coordinates erases exactly the
    # information the whitening encodes and random-walks likelihood-null
    # directions at its noise floor.
    #
    # Each may be a constant or a Schedule(final=, ramp=) (see SCHEDULABLE_LRS
    # at the top of this module): the ramp is a continuation device for the
    # initial MAP solve only — inverse.py refreshes each optimizer group's lr
    # from `learning_rates(i, level, schedule=True)` every iteration, so a ramp
    # of 0.0 freezes that parameter (its optimizer state — SGD momentum —
    # still accumulates, so the first live step is well-conditioned). RTO
    # reads the steady-state `final` — as with the loss weights, `final` is
    # the contract.
    lr_z_bed:      LearningRate = 0.0125
    lr_z_bed_mean: LearningRate = 0.5
    lr_z_log_beta: LearningRate = 4.05

    lr_z_pbias:    LearningRate = 0.001
    lr_z_tbias:    LearningRate = 0.001
    lr_z_log_mf:   LearningRate = 0.01
    lr_z_log_rf:   LearningRate = 0.01
    # Enthalpy-model GP fields (SGD, used in place of z_log_mf/z_log_rf when
    # smb_model == "enthalpy"). The 0.05 Adam-era default was retired with the
    # scalar parametrization: under SGD on a whitened field the data curvature
    # along the smooth modes is large (see CLAUDE.md on the scalar/field
    # curvature), so start small and retune on the level-2 trace.
    lr_z_log_H_atm:   LearningRate = 0.001
    lr_z_logit_cloud: LearningRate = 0.001
    # Elevation-dependent precip depletion scalars (SGD like everything else,
    # only added to the optimizer when precip_lapse_enabled). See the prior
    # widths above.
    lr_z_tau:      LearningRate = 0.01
    lr_z_z0:       LearningRate = 0.01

    def at_iteration(self, iteration: int, level: int = 0, *,
                     schedule: bool = False) -> "GlacierConfig":
        """Return a copy with every schedulable loss weight and learning rate
        resolved to a float at `(iteration, level)`.

        When `schedule` is False (the default, used by RTO / posterior /
        sensitivity) constants pass through and any Schedule collapses to its
        steady-state `final`, so every task targets the same objective. When
        `schedule` is True (the initial inverse solve) Schedule ramps and bare
        callables are evaluated as f(i, level) / f(i). Called per optimizer step
        by GlacierProblem.compute_loss. Per-observation weights follow the same
        contract but are resolved by Observation.weight_at — a domain config may
        set e.g. `SnowlineSpec(weight=Schedule(final=2e-4,
        ramp=lambda i, level: 0.0 if level > 0 else 2e-4))`.
        """
        overrides = {name: resolve_weight(getattr(self, name), iteration, level,
                                          schedule=schedule, what=name)
                     for name in SCHEDULABLE_WEIGHTS}
        overrides.update(self.learning_rates(iteration, level, schedule=schedule))
        return replace(self, **overrides)

    def learning_rates(self, iteration: int = 0, level: int = 0, *,
                       schedule: bool = False) -> dict:
        """Every schedulable learning rate resolved to a float at
        `(iteration, level)`, keyed by field name (`"lr_z_bed"`, ...).

        Same contract as the loss weights: `schedule=True` (inverse.py) honors
        Schedule ramps and bare callables; the default collapses each Schedule
        to its steady-state `final`, which is what RTO warm-starting from the
        MAP should use. Drivers name each optimizer param group after its
        config field and refresh `group["lr"]` from this dict per iteration.
        """
        return {name: resolve_weight(getattr(self, name), iteration, level,
                                     schedule=schedule, what=name)
                for name in SCHEDULABLE_LRS}

    @property
    def output_dir(self) -> str:
        """Absolute path to this domain's active experiment directory."""
        return f"{self.base_dir}/{self.results_subdir}"

    @property
    def B_rate(self) -> float:
        """Computed rate factor used by IceDynamics rheology.B."""
        return self.A_glen ** (-1.0 / self.n_glen) / (self.rho_ice * self.gravity)
