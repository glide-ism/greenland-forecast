"""
Time-stamped observations: data + acquisition time(s) + loss + noise model.

Each observational product is an `Observation` subclass carrying its data
tensors, the calendar time(s) at which the model must emit state for
comparison (`required_times`), its noise/loss hyperparameters, and its misfit
implementation. The scalar loss weight may be a constant or a `Schedule`
(continuation, honored only by the initial MAP solve — see config.Schedule).

Domain configs do not construct Observation objects directly — data loading
needs the cropped model grid, which only exists inside `GlacierProblem`.
Instead they declare lightweight frozen *spec* dataclasses (hyperparameters +
optional time overrides); `GlacierProblem` calls `spec.build(ctx)` with an
`ObservationBuildContext` to produce the loaded Observation, or `None` when
the domain lacks the product (the graceful-skip behavior: the corresponding
loss term simply does not exist).

Timestamps come from the data files: preprocessing writes `time_nominal` /
`time_start` / `time_end` (calendar-year floats) as variable-level attributes.
A spec-level `time` override wins; files without attrs fall back to the
nominal calibration end `config.t_end` with a warning (legacy inputs keep
working unchanged).
"""
import dataclasses
import warnings
from dataclasses import dataclass
from typing import Optional

import cupy as cp
import torch
from torch.nn.functional import grid_sample

from ggapp.torch import GGaPPMap, GGaPPWhiten

from .config import LossWeight, MaternNoise, resolve_weight
from .loss import _huber, marginal_velocity_log_likelihood


# --------------------------------------------------------------------------- #
# Correlated (Matérn GP) observation errors                                   #
# --------------------------------------------------------------------------- #
#
# A field likelihood with `noise=MaternNoise(sigma, l, nu)` on its spec treats
# the residual r = model − data as a draw from a zero-mean Matérn GP and forms
# its misfit on the whitened residual z = W r (ggapp's GGaPPWhiten: one
# application of the (dx/τ)·L^{α/2} stencil — no solve, self-adjoint, cheap).
# The contract is
#     loss = loss_scale · weight · Σ huber(z),   weight ≡ 1,   no dx²
# — the ad-hoc "observation density" weight·dx² of the diagonal terms is gone;
# the error model's σ and l carry all of that information. Whitening is not
# down-weighting: it keeps full weight on fine-scale residual structure (what
# constrains bed/β) and discounts only the smooth directions that a correlated
# error can explain. `validate_noise_weights` enforces the weight contract at
# problem build (continuation ramps may zero a term, but `final` must be 1).
#
# Masks (dhdt_mask, v_mask under mask_unobserved, outlier keep-masks) are
# applied to r BEFORE whitening, so r drops to zero at the mask edge and z
# carries a margin band there (amplitude ~ r_edge·l/(10 dx)); the Huber
# threshold bounds its influence. Inspect the `*_z` fields in residuals.pvd.
#
# Multigrid levels. Surface, extent and snowline are evaluated IN THE MODEL'S
# OWN SPACE at each level: the observation is box-restricted to the level's
# grid and compared with the coarse model field (S_coarse, H, active,
# smb_coarse), the whitening model re-discretized at the coarse dx with the
# white nugget scaled nugget/2^L, and the diagonal/Brier forms carrying dx_L².
# For structure the coarse grid resolves this is the same quadratic form as
# on the fine grid (up to discretization), so the terms' effective
# observation counts are level-independent; what is dropped is only the
# sub-grid content of the observation that a coarse model cannot represent —
# which, compared on the fine grid, was an irreducible floor (delta surface:
# 790 loss units at level 2, 370 at level 1, i.e. essentially the whole
# term). Velocity and dh/dt stay fine-grid (smooth model fields under
# on-ice masks; their floors are a few units). Level 0 is unchanged.
#
# Products with pixel-scale noise need `MaternNoise(nugget=)` (see config.py):
# the whitening then goes through priors.SpectralMaternNoise (exact DCT-II
# form of the same operator plus a white term) — the code here is agnostic,
# `noise_model` just has to provide cupy `whiten`/`forward`.


def _restrict(field: torch.Tensor, level: int) -> torch.Tensor:
    """Box-average a fine (ny, nx) field down `level` times (the model's
    restriction). Masks become fractions; identity at level 0."""
    if level == 0:
        return field
    f = 2 ** level
    return torch.nn.functional.avg_pool2d(
        field.to(torch.float32)[None, None], f)[0, 0]


def _restrict_weighted(field, weight, level):
    """Weight-averaged restriction: sum(w f) / sum(w) per cell (0 where the
    cell carries no weight). For labels defined only where a mask is set."""
    if level == 0:
        return field
    num = _restrict(field * weight, level)
    den = _restrict(weight, level)
    return torch.where(den > 0, num / den.clamp(min=1e-12), torch.zeros_like(num))


def _whiten(member, r: torch.Tensor) -> torch.Tensor:
    """z = W r for a (ny, nx) residual field (GGaPPWhiten needs a contiguous
    float32 tensor — it copies through cupy)."""
    return GGaPPWhiten.apply(member, r.contiguous().to(torch.float32))


def _require_priors(ctx: "ObservationBuildContext", what: str):
    if getattr(ctx, "priors", None) is None:
        raise ValueError(
            f"{what}: a MaternNoise error model needs the GlacierPriors "
            f"collection, but ObservationBuildContext.priors is None (build "
            f"the observation through GlacierProblem, or pass priors=).")
    return ctx.priors


def validate_noise_weights(observations) -> None:
    """Enforce the weight == 1 contract for every noise-modelled term. The
    steady-state (`final`) weight is checked; a Schedule ramp may still hold a
    term at 0 during continuation."""
    for obs in observations:
        modelled = getattr(obs, "noise", None) is not None \
            or getattr(obs, "sigma_p", None) is not None
        if not modelled:
            continue
        w = obs.weight_at(0, 0, schedule=False)
        if w != 1.0:
            raise ValueError(
                f"observation {obs.name!r} carries an explicit error model "
                f"(MaternNoise / sigma_p) but its (final) weight is {w!r}; "
                f"such terms have weight == 1 by contract — express trust in "
                f"the product through the error model's parameters instead.")


@dataclass
class DomainData:
    """Shared, non-product context every loss term may draw on."""
    dem: torch.Tensor           # full DEM (topography + bathymetry)
    rgi_mask: torch.Tensor      # 0/1 ice extent
    rgi_label: torch.Tensor     # per-glacier long id, -1 = unlabeled
    surge_type: torch.Tensor
    obs_mask: torch.Tensor
    domain_mask: torch.Tensor   # 0/1 simulation domain


@dataclass
class ObservationBuildContext:
    """Everything a spec needs to load its product on the cropped grid."""
    gridded_data: object                 # xr.Dataset (GLIDE_inputs, cropped)
    snowline_data: Optional[object]      # xr.Dataset or None
    dhdt_data: Optional[object]          # xr.Dataset or None
    flightlines_df: object               # GeoDataFrame
    domain: DomainData
    ny: int
    nx: int
    config: object                       # GlacierConfig
    priors: object = None                # GlacierPriors (noise-model registry)


def profile_fingerprints(z, fps, nu, iters: int = 2):
    """Profile the rank-few fingerprint nuisance out of a whitened residual.

    `fps` is a list of (w_j, s_j): whitened fingerprint fields (detached) and
    their prior stds. Solves min_c Σ huber(z − Σ c_j w_j, nu) + ½ Σ c_j²/s_j²
    by damped Gauss–Newton in the k-dim c (IRLS weights from the pseudo-Huber;
    k is tiny so each step is a k×k solve). Returns (z_down, prior, c):
    `z_down = z − Σ ĉ_j w_j` with ĉ DETACHED (envelope theorem — the caller's
    gradient flows through z only), `prior = ½ Σ ĉ²/s²` (added to the loss so
    absorption stays visible), and ĉ as a k-vector (prior-std units — the
    audit trail of how much of each parameter's pattern the term attributes
    to model error). See config.FingerprintNuisance for the model.
    """
    k = len(fps)
    W = torch.stack([w for w, _ in fps])            # (k, ny, nx), detached
    s2 = torch.tensor([s ** 2 for _, s in fps], device=z.device)
    with torch.no_grad():
        zd = z.detach()
        c = torch.zeros(k, device=z.device)
        for _ in range(iters):
            t = zd - torch.einsum("k,kij->ij", c, W)
            rho = 1.0 / torch.sqrt(1.0 + (t / nu) ** 2)   # huber'(t)/t
            grad = -torch.einsum("kij,ij->k", W, rho * t) + c / s2
            H = torch.einsum("kij,lij->kl", W, W * rho.unsqueeze(0)) \
                + torch.diag(1.0 / s2)
            c = c - torch.linalg.solve(H, grad)
    z_down = z - torch.einsum("k,kij->ij", c, W)
    prior = 0.5 * (c ** 2 / s2).sum()
    return z_down, prior, c


def _pcg(A, b, M_inv, x0, rtol: float = 1e-3, maxiter: int = 100):
    """Preconditioned conjugate gradients for SPD A (torch, no autograd)."""
    x = x0.clone()
    r = b - A(x)
    z = M_inv(r)
    p = z.clone()
    rz = (r * z).sum()
    b_norm = b.norm() + 1e-30
    it = 0
    for it in range(maxiter):
        if r.norm() <= rtol * b_norm:
            break
        Ap = A(p)
        alpha = rz / ((p * Ap).sum() + 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        z = M_inv(r)
        rz_new = (r * z).sum()
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, it


class LogitNuisance:
    """Profiled GP nuisance on the logit of a Brier term.

    Model: the observed class probability is p(η_model + ε) with a coherent
    logit-error field ε ~ GP(0, Matérn(σ, l, ν)) — margin/snowline
    misplacement that is correlated along the boundary is explained by ε at
    prior cost ~ (L/l)·(δ·|∇η|/σ)² instead of per-pixel Brier cost, while
    fine-scale, incoherent disagreement still pays the full Brier. ε is
    PROFILED inside the loss evaluation (never exposed to the outer
    optimizer): each call runs `inner_steps` damped Gauss–Newton updates of

        min_ε  c·Σ ω (p(η+ε) − t)² + ½ εᵀ C_l⁻¹ ε,

    each solving (C_l⁻¹ + D) ε₊ = D ε − ∇F_d by CG preconditioned with C_l
    (both spectral operators via `SpectralMaternNoise`; the mild data
    strength of a Brier term keeps the preconditioned condition number small,
    so ~10 iterations suffice), warm-started from the last call — a lagged
    iteration that tracks the exact profile ε*(m) as the outer solve
    converges. The returned loss is the profile objective at the detached ε*
    (envelope theorem: its m-gradient needs no differentiation through the
    solve); the ½‖z_ε‖² prior cost is included so the loss trace shows the
    total and absorption stays visible (`last` carries the pieces). Note the
    deliberate omission of the Laplace log-det: its only content beyond
    profiling is an Occam term whose gradient rewards steepening the
    transition — the BCE sharpness pathology the Brier exists to avoid.

    Optional amplitude bound (`eps_max`): a Gaussian ε prices coherence but
    not cause — its cost is linear in area, exactly like the Brier's, so for
    any coherent miss larger than l the contest is area-independent and
    (σ·l vs class-gap·σ_p·dx) decides it for EVERY such feature at once: at
    honest outline-error σ's, whole missing tongues get explained away and
    the envelope gradient on the physics collapses. With `eps_max` set the
    field is saturated, ε = eps_max·tanh(u/eps_max) with u ~ GP(0, Matérn):
    below the bound this is the same Gaussian model (outline-scale coherent
    error priced identically), but a full class flip becomes unreachable —
    p(η+ε) ≤ p(η+eps_max), so a truly missing tongue keeps a floor of Brier
    residual and gradient forever. Read eps_max as the largest CREDIBLE
    outline misplacement in logits (≈ 3·δ_max/W_t, with W_t the
    margin-to-3·s_H transition width); ε̂ pinned at the bound along a feature
    is the diagnostic for a real outline error larger than δ_max. eps_max =
    None (default) is the unbounded model, bit-identical to before.

    Per-level state: the latent u lives on the grid the term is evaluated
    on; a new level warm-starts from the prolonged coarser u. u is not a
    checkpointed parameter — after a warm start it re-converges within a few
    iterations.
    """

    def __init__(self, hp: MaternNoise, inner_steps: int = 2,
                 eps_max: Optional[float] = None):
        if hp.nugget != 0.0:
            raise ValueError(
                "LogitNuisance: the logit-error GP is a smooth model-error "
                "field; a white nugget is meaningless here (the Brier term "
                "itself prices per-pixel fuzz). Use MaternNoise(nugget=0).")
        if eps_max is not None and not eps_max > 0.0:
            raise ValueError(
                f"LogitNuisance: eps_max must be positive (or None for the "
                f"unbounded model), got {eps_max!r}.")
        self.hp = hp
        self.inner_steps = inner_steps
        self.eps_max = eps_max
        self._ops = {}    # level -> SpectralMaternNoise on that grid
        self._u = {}      # level -> latent u (torch, detached); ε = _eps_of(u)
        self.last = {}

    def _eps_of(self, u):
        """The logit-error field for latent u (identity when unbounded)."""
        if self.eps_max is None:
            return u
        return self.eps_max * torch.tanh(u / self.eps_max)

    def _deps_du(self, u):
        """∂ε/∂u = 1 − (ε/eps_max)² (1 when unbounded)."""
        if self.eps_max is None:
            return None
        t = torch.tanh(u / self.eps_max)
        return 1.0 - t * t

    def _op(self, level, ny, nx, dx):
        if level not in self._ops:
            from .priors import SpectralMaternNoise
            self._ops[level] = SpectralMaternNoise(self.hp, ny, nx, dx)
        return self._ops[level]

    @staticmethod
    def _apply2(fn, v):
        """fn∘fn on a torch field through cupy (C_l⁻¹ = whiten², C_l = forward²)."""
        u = cp.asarray(v.detach().contiguous())
        return torch.as_tensor(fn(fn(u)), device=v.device)

    def eps_at(self, level):
        u = self._u.get(level)
        return None if u is None else self._eps_of(u)

    def solve(self, *, level, dx_level, p_g, omega, target, c) -> tuple:
        """Update the latent u and return (ε*, prior_cost) for the current
        model logits.

        `p_g(ε) -> (p, ∂p/∂ε)`; `omega`/`target` the Brier weights/targets on
        the level grid; `c = weight · dx_level² / s_B²` (the resolved Brier
        scale WITHOUT loss_scale, which cancels in the inner problem).
        The Gauss–Newton runs in u (the GP latent): the model Jacobian is
        ∂p/∂u = g·(∂ε/∂u), and the prior term is on u — under the bound this
        is the pushforward prior on ε. All under no_grad; the caller
        re-evaluates the loss at the returned detached ε with the live graph.
        """
        ny, nx = target.shape
        op = self._op(level, ny, nx, dx_level)
        Q = lambda v: self._apply2(op.whiten, v)
        C = lambda v: self._apply2(op.forward, v)
        u = self._u.get(level)
        if u is None:
            coarser = self._u.get(level + 1)
            if coarser is not None:
                from .forward import differentiable_prolongation
                u = differentiable_prolongation(coarser, 1).contiguous()
            else:
                u = torch.zeros(ny, nx, device=target.device)

        def F(v):
            p, _ = p_g(self._eps_of(v))
            return (c * (omega * (p - target) ** 2).sum()
                    + 0.5 * (v * Q(v)).sum())

        cg_total = 0
        for _ in range(self.inner_steps):
            p, g = p_g(self._eps_of(u))
            tprime = self._deps_du(u)
            J = g if tprime is None else g * tprime
            D = 2.0 * c * omega * J * J
            grad_d = 2.0 * c * omega * J * (p - target)
            new, iters = _pcg(lambda v: Q(v) + D * v, D * u - grad_d,
                              C, x0=u)
            cg_total += iters + 1
            # Gauss–Newton can overshoot where the sigmoid (or the amplitude
            # bound) saturates: damp by backtracking on the true profile
            # objective.
            step = new - u
            f0 = F(u)
            accepted = u
            t = 1.0
            for _ in range(4):
                cand = u + t * step
                if F(cand) <= f0 + 1e-6 * abs(f0):
                    accepted = cand
                    break
                t *= 0.5
            u = accepted
        self._u[level] = u
        eps = self._eps_of(u)
        prior = 0.5 * (u * Q(u)).sum()
        self.last = dict(level=level, cg=cg_total,
                         prior=float(prior),
                         eps_absmax=float(eps.abs().max()))
        if self.eps_max is not None:
            # Fraction of the scored region pinned at the bound — the
            # diagnostic for real outline errors larger than the budget.
            scored = omega > 0
            if scored.any():
                self.last["saturated_frac"] = float(
                    (eps.abs()[scored] > 0.95 * self.eps_max)
                    .float().mean())
        return eps, prior


def read_time_attrs(da, *, fallback: float, what: str):
    """(nominal, start, end) acquisition times from variable-level attrs.

    Missing attrs fall back to `fallback` (the nominal calibration end) with a
    one-line warning — legacy model_inputs files keep working, they just pin
    the product to the final epoch as before.
    """
    attrs = getattr(da, "attrs", {})
    nominal = attrs.get("time_nominal")
    start = attrs.get("time_start", nominal)
    end = attrs.get("time_end", nominal)
    if nominal is None:
        warnings.warn(
            f"{what}: no time_nominal/time_start/time_end attrs found; "
            f"assuming t={fallback} (the nominal calibration end). Rebuild "
            f"the product with preprocessing/make_all.py to get real "
            f"acquisition times."
        )
        return float(fallback), float(fallback), float(fallback)
    return float(nominal), float(start), float(end)


# --------------------------------------------------------------------------- #
# Observation classes                                                         #
# --------------------------------------------------------------------------- #

class Observation:
    """Base class: data + required model times + loss + noise model.

    `loss` receives the finished `SimResult` (`sim`) and looks up the model
    state at its own `required_times` via `sim.at(t)`; `GlacierProblem`
    guarantees those times were recorded. `weight` follows the Schedule
    contract: `weight_at` resolves it per optimizer step, honoring
    continuation ramps only when `schedule=True` (the initial MAP solve).

    `randomized(**eps)` returns a perturbed copy for randomize-then-optimize;
    the default passthrough is correct for categorical products. (The RTO
    driver has not yet been migrated to the per-observation API; the hook is
    here so that migration is purely driver-side.)
    """

    name: str = "observation"

    def __init__(self, *, weight: LossWeight):
        self.weight = weight

    # Rank-few fingerprint nuisance (see config.FingerprintNuisance).
    # `GlacierProblem.refresh_fingerprints` measures whitened fingerprints on
    # the current level and installs them here; a whitened loss branch that
    # supports the downdate profiles them out via `profile_fingerprints`.
    # None (the default) is the unmodified likelihood, bit-identical.
    _fingerprints = None
    fingerprint_c = None    # last fitted ĉ (prior-std units), diagnostic

    def set_fingerprints(self, fps, level: int) -> None:
        """Install whitened fingerprints [(w, s), ...] measured at `level`
        (or clear with fps=None). The downdate applies only while the term is
        evaluated on the same level — the driver refreshes at every level
        start, so a mismatch simply disables it."""
        self._fingerprints = None if fps is None else \
            {"level": level, "fps": [(w.detach(), float(s)) for w, s in fps]}
        self.fingerprint_c = None

    def _apply_fingerprints(self, z, level: int, nu):
        """Downdate the whitened residual: returns (z_eff, prior_cost)."""
        fp = self._fingerprints
        if fp is None or fp["level"] != level:
            return z, None
        z_down, prior, c = profile_fingerprints(z, fp["fps"], nu)
        self.fingerprint_c = c
        return z_down, prior

    @property
    def required_times(self) -> tuple:
        return ()

    def weight_at(self, iteration: int, level: int, *, schedule: bool) -> float:
        return resolve_weight(self.weight, iteration, level, schedule=schedule)

    def loss(self, *, sim, physical, config, domain: DomainData,
             mask: torch.Tensor, dx: float, weight: float) -> torch.Tensor:
        raise NotImplementedError

    def residuals(self, *, sim, physical, config, domain: DomainData,
                  mask: torch.Tensor, dx: float) -> dict:
        """Diagnostic residual fields (name -> 2-D tensor) for the field
        likelihoods: `r` = σ-normalized raw residual, `z` = whitened residual
        (`z is r` when the term has no MaternNoise model). Empty for
        categorical / prior-style terms."""
        return {}

    def randomized(self, **eps) -> "Observation":
        return self

    def diagnostics(self) -> dict:
        """Static fields for the observations PVD dump (name -> 2-D tensor)."""
        return {}


class SurfaceObservation(Observation):
    """Surface-elevation misfit (Huber) against the DEM at its epoch.

    With a `noise` model the residual S_model − S_obs is whitened by the
    Matérn member (marginal std `sigma` = noise.sigma) over the full grid —
    off-ice the residual is bed − DEM, exactly as in the diagonal form.

    Evaluated in the model's own space: at multigrid level L the coarse
    surface is compared with the box-restricted DEM (`S_obs_at(L)`) and
    whitened by the error model re-discretized on the coarse grid
    (`noise_model_at(L)`, nugget/2^L); the diagonal form carries dx_L². See
    the module note on multigrid levels.
    """

    name = "srf"

    def __init__(self, *, S_obs, time: float, sigma: float, nu: float,
                 weight: LossWeight, noise: Optional[MaternNoise] = None,
                 noise_model=None, priors=None):
        super().__init__(weight=weight)
        self.S_obs = S_obs
        self.time = time
        self.sigma = sigma          # marginal std under both error models
        self.nu = nu
        self.noise = noise
        self.noise_model = noise_model
        self._priors = priors       # registry for the per-level noise models
        self._S_obs_at = {0: S_obs}

    @property
    def required_times(self):
        return (self.time,)

    def S_obs_at(self, level: int) -> torch.Tensor:
        """The DEM box-restricted to multigrid `level` (cached)."""
        if level not in self._S_obs_at:
            self._S_obs_at[level] = _restrict(self.S_obs, level)
        return self._S_obs_at[level]

    def noise_model_at(self, level: int):
        """The error model discretized on the level's grid (level 0: the
        registered model itself)."""
        if level == 0:
            return self.noise_model
        if self._priors is None:
            raise ValueError("SurfaceObservation: coarse-level whitening needs "
                             "the GlacierPriors registry (build via a spec)")
        return self._priors.noise_model("srf", self.noise, level=level)

    def _raw(self, sim):
        """(S_model − S_obs) on the snapshot's own grid."""
        state = sim.at(self.time)
        return state.S_coarse - self.S_obs_at(state.level), state.level

    def residuals(self, *, sim, physical, config, domain, mask, dx):
        raw, level = self._raw(sim)
        r = raw / self.sigma
        z = _whiten(self.noise_model_at(level), raw) if self.noise is not None else r
        return {"r": r, "z": z}

    def loss(self, *, sim, physical, config, domain, mask, dx, weight):
        raw, level = self._raw(sim)
        if self.noise is not None:
            z = _whiten(self.noise_model_at(level), raw)
            z, fp_prior = self._apply_fingerprints(z, level, self.nu)
            J = config.loss_scale * weight * _huber(z, self.nu).sum()
            if fp_prior is not None:
                J = J + config.loss_scale * weight * fp_prior
            return J
        scale = config.loss_scale * (dx * 2 ** level) ** 2
        r_s = raw / self.sigma
        return scale * weight * _huber(r_s, self.nu).sum()

    def randomized(self, *, eps_S=None):
        eps_S = torch.randn_like(self.S_obs) if eps_S is None else eps_S
        if self.noise is not None:
            perturbation = GGaPPMap.apply(self.noise_model, eps_S.contiguous())
        else:
            perturbation = eps_S * self.sigma
        return SurfaceObservation(
            S_obs=self.S_obs + perturbation, time=self.time,
            sigma=self.sigma, nu=self.nu, weight=self.weight,
            noise=self.noise, noise_model=self.noise_model,
            priors=self._priors)

    def diagnostics(self):
        return {"srf_obs": self.S_obs}


class VelocityObservation(Observation):
    """Velocity misfit at the mosaic's nominal epoch.

    Either a plain pseudo-Huber on the cell-centered residual, or (when
    `surge_biased` is set) the per-glacier marginal likelihood over a velocity
    scaling eta in (0, 1] (observed = eta * model) with Beta(alpha, 1) priors,
    `alpha_surge` for RGI surge_type == 3 and `alpha_nonsurge` otherwise. The
    marginal is formed in the likelihood's own units and tempered by the
    weight afterwards, so the weight expresses trust in the product while the
    alphas express how much the mosaic may under-read the model per glacier
    (alpha=2 is very permissive; hold non-surging glaciers near eta=1 with
    alpha ~ 30-50, or make them permissive too if the mosaic is not trusted
    on slow ice).

    `mask_unobserved` decides what a mosaic pixel with `v_mask == 0` (no
    retrieved motion: |v_obs| <= 1 m/yr in make_velocity.py — every off-ice
    cell and the low-texture accumulation zones) means. False (historical):
    it is an observation of ~zero velocity, so modelled ice flowing there is
    penalized. True: it is undefined and contributes nothing to the misfit.

    With a `noise` model each component's residual is whitened by the Matérn
    member (marginal std `sigma` = noise.sigma per component). The surge
    marginal survives whitening by linearity: inside a glacier
    z(η) = η·W(m·U_mod) − W(m·U_obs), so the marginal is evaluated on the
    whitened stacks with unit sigma (exact away from glacier boundaries,
    where the stencil mixes neighbouring labels). The valid mask m (v_mask
    under `mask_unobserved`, times the `outlier_threshold` keep-mask) is
    applied before whitening in both branches — note the diagonal surge
    branch ignores `outlier_threshold`, the whitened one does not.
    """

    name = "vel"

    def __init__(self, *, u_obs, v_obs, v_mask, time: float, sigma: float,
                 nu: float, surge_biased: bool, weight: LossWeight,
                 outlier_threshold: float, alpha_surge: float = 2.0,
                 alpha_nonsurge: float = 6.0, mask_unobserved: bool = False,
                 noise: Optional[MaternNoise] = None, noise_model=None):
        super().__init__(weight=weight)
        self.alpha_surge = alpha_surge
        self.alpha_nonsurge = alpha_nonsurge
        self.mask_unobserved = mask_unobserved
        self.u_obs = u_obs
        self.v_obs = v_obs
        self.v_mask = v_mask
        self.time = time
        self.sigma = sigma          # marginal std per component, both models
        self.nu = nu
        self.surge_biased = surge_biased
        self.outlier_threshold = outlier_threshold
        self.noise = noise
        self.noise_model = noise_model

    @property
    def required_times(self):
        return (self.time,)

    @staticmethod
    def _predicted(state):
        """Cell-centred modelled surface velocity (u + ud/(n+1) under MOLHO;
        the depth-averaged u under SSA), from the staggered fine fields."""
        u_fine = state.u_surf_fine
        v_fine = state.v_surf_fine
        u_pred = (u_fine[:, 1:] + u_fine[:, :-1]) / 2.0
        v_pred = (v_fine[1:, :] + v_fine[:-1, :]) / 2.0
        return u_pred, v_pred

    def _valid_mask(self) -> torch.Tensor:
        """0/1 float mask of pixels that enter the whitened misfit."""
        m = self.v_mask if self.mask_unobserved else torch.ones_like(self.v_mask)
        if self.outlier_threshold is not None:
            U_obs2 = self.u_obs ** 2 + self.v_obs ** 2
            m = m * (U_obs2 <= self.outlier_threshold ** 2).to(m.dtype)
        return m

    def residuals(self, *, sim, physical, config, domain, mask, dx):
        u_pred, v_pred = self._predicted(sim.at(self.time))
        m = self._valid_mask()
        raw_u = (u_pred - self.u_obs) * m
        raw_v = (v_pred - self.v_obs) * m
        out = {"r_u": raw_u / self.sigma, "r_v": raw_v / self.sigma}
        if self.noise is not None:
            out["z_u"] = _whiten(self.noise_model, raw_u)
            out["z_v"] = _whiten(self.noise_model, raw_v)
        else:
            out["z_u"], out["z_v"] = out["r_u"], out["r_v"]
        return out

    def _loss_whitened(self, *, sim, config, domain, weight):
        from numpy.polynomial.legendre import leggauss

        scale = config.loss_scale
        u_pred, v_pred = self._predicted(sim.at(self.time))
        m = self._valid_mask()
        W = self.noise_model
        if self.surge_biased:
            Z_obs = torch.stack((_whiten(W, m * self.u_obs).ravel(),
                                 _whiten(W, m * self.v_obs).ravel()), dim=1)
            Z_mod = torch.stack((_whiten(W, m * u_pred).ravel(),
                                 _whiten(W, m * v_pred).ravel()), dim=1)
            labels = domain.rgi_label.ravel()
            sigma = torch.ones(Z_obs.shape[0], device='cuda',
                               dtype=torch.float32)
            nodes, weights = leggauss(10)
            eta_nodes = torch.tensor((nodes + 1) / 2, device='cuda',
                                     dtype=torch.float32)
            w_gl = torch.tensor(weights / 2, device='cuda', dtype=torch.float32)

            alpha = torch.where(domain.surge_type == 3,
                                self.alpha_surge, self.alpha_nonsurge).cuda()

            log_w_eff = (torch.log(w_gl)[:, None]
                         + torch.log(alpha)[None, :]
                         + (alpha[None, :] - 1) * torch.log(eta_nodes[:, None]))

            return marginal_velocity_log_likelihood(
                Z_obs, Z_mod, sigma, labels, eta_nodes, log_w_eff,
                self.nu, weight, scale,
            )
        z_u = _whiten(W, (u_pred - self.u_obs) * m)
        z_v = _whiten(W, (v_pred - self.v_obs) * m)
        z2 = z_u ** 2 + z_v ** 2
        return scale * weight * self.nu ** 2 * (
            torch.sqrt(1 + z2 / self.nu ** 2) - 1).sum()

    def loss(self, *, sim, physical, config, domain, mask, dx, weight):
        if self.noise is not None:
            return self._loss_whitened(sim=sim, config=config, domain=domain,
                                       weight=weight)

        from numpy.polynomial.legendre import leggauss

        scale = config.loss_scale * dx ** 2
        state = sim.at(self.time)
        # Feature-tracked mosaics observe the SURFACE velocity, u + ud/(n+1)
        # under MOLHO (identical to the depth-averaged u under SSA, ud == 0).
        u_fine = state.u_surf_fine
        v_fine = state.v_surf_fine
        u_pred = (u_fine[:, 1:] + u_fine[:, :-1]) / 2.0
        v_pred = (v_fine[1:, :] + v_fine[:-1, :]) / 2.0

        if self.surge_biased:
            U_obs = torch.stack((self.u_obs.ravel(), self.v_obs.ravel()), dim=1)
            U_mod = torch.stack((u_pred.ravel(), v_pred.ravel()), dim=1)
            labels = domain.rgi_label.ravel()
            if self.mask_unobserved:
                keep = self.v_mask.ravel() > 0.0
                U_obs, U_mod, labels = U_obs[keep], U_mod[keep], labels[keep]
            sigma = self.sigma * torch.ones(U_obs.shape[0], device='cuda',
                                            dtype=torch.float32)
            nodes, weights = leggauss(10)
            eta_nodes = torch.tensor((nodes + 1) / 2, device='cuda',
                                     dtype=torch.float32)
            w_gl = torch.tensor(weights / 2, device='cuda', dtype=torch.float32)

            alpha = torch.where(domain.surge_type == 3,
                                self.alpha_surge, self.alpha_nonsurge).cuda()

            log_w_eff = (torch.log(w_gl)[:, None]
                         + torch.log(alpha)[None, :]
                         + (alpha[None, :] - 1) * torch.log(eta_nodes[:, None]))

            return marginal_velocity_log_likelihood(
                U_obs, U_mod, sigma, labels, eta_nodes, log_w_eff,
                self.nu, weight, scale,
            )
        else:
            r_u2 = (((u_pred - self.u_obs) ** 2 + (v_pred - self.v_obs) ** 2)
                    / self.sigma ** 2)
            if self.mask_unobserved:
                r_u2 = r_u2 * self.v_mask
            if self.outlier_threshold is not None:
                U_obs2 = self.u_obs**2 + self.v_obs**2
                r_u2[U_obs2 > self.outlier_threshold**2] = 0.0
            return scale * weight * self.nu ** 2 * (
                torch.sqrt(1 + r_u2 / self.nu ** 2) - 1).sum()

    def randomized(self, *, eps_u=None, eps_v=None):
        eps_u = torch.randn_like(self.u_obs) if eps_u is None else eps_u
        eps_v = torch.randn_like(self.v_obs) if eps_v is None else eps_v
        if self.noise is not None:
            du = GGaPPMap.apply(self.noise_model, eps_u.contiguous())
            dv = GGaPPMap.apply(self.noise_model, eps_v.contiguous())
        else:
            du, dv = eps_u * self.sigma, eps_v * self.sigma
        return VelocityObservation(
            u_obs=self.u_obs + du,
            v_obs=self.v_obs + dv,
            v_mask=self.v_mask, time=self.time, sigma=self.sigma, nu=self.nu,
            surge_biased=self.surge_biased, weight=self.weight,
            outlier_threshold=self.outlier_threshold,
            alpha_surge=self.alpha_surge, alpha_nonsurge=self.alpha_nonsurge,
            mask_unobserved=self.mask_unobserved,
            noise=self.noise, noise_model=self.noise_model)


class ExtentObservation(Observation):
    """Glacier-extent misfit (Brier-style) at the inventory's epoch.

    Blends a thickness-derived probability with an SMB-derived one via the
    dynamics' active mask. `smb_dt` is the timescale converting SMB to a
    thickness-equivalent logit (historically `config.dt`); it is an explicit
    hyperparameter so a variable step sequence cannot silently change it.
    The extent labels themselves come through `mask` (rgi_mask ∩ domain, or
    the RTO Bernoulli realization).

    `two_sided=False` (historical) scores only the mask's ice cells — it
    penalizes missing ice where the inventory has it and leaves surplus ice
    to the surface term. `two_sided=True` is the full Brier score on every
    in-domain cell, so modelled ice outside the outline is penalized too.
    """

    name = "extent"

    def __init__(self, *, time: float, s_H: float, smb_dt: float,
                 weight: LossWeight, two_sided: bool = False,
                 logit_nuisance: Optional[LogitNuisance] = None,
                 sigma_p: Optional[float] = None):
        super().__init__(weight=weight)
        self.time = time
        self.s_H = s_H
        self.smb_dt = smb_dt
        self.two_sided = two_sided
        self.logit_nuisance = logit_nuisance
        self.sigma_p = sigma_p

    def _c_data(self, level: int, dx: float, weight: float) -> float:
        """Scale of the per-pixel Brier quadratic. With `sigma_p` (per-pixel
        class-probability noise std, weight == 1 by contract) it is
        4^L/(2σ_p²) — independent pixel errors average under restriction, so
        this is level-consistent by construction and carries no dx² or s_B.
        The legacy form weight·dx_L²/s_B² is identical under
        σ_p = s_B/√(2·weight·dx²)."""
        if self.sigma_p is not None:
            return weight * 4.0 ** level / (2.0 * self.sigma_p ** 2)
        return weight * (dx * 2 ** level) ** 2 / 0.5 ** 2

    @property
    def required_times(self):
        return (self.time,)

    def _p_g(self, H, smb, active, eps):
        """Blended extent probability at logit shift ε, and ∂p/∂ε. ε shifts
        both constituent logits — it is an error on the extent logit itself,
        whichever branch supplies it."""
        s_dyn = torch.sigmoid(H / self.s_H + eps)
        s_smb = torch.sigmoid(self.smb_dt / self.s_H * smb + eps)
        p_dyn = (2.0 * s_dyn - 1.0).clip(min=0.001, max=0.999)
        p_smb = s_smb.clip(min=0.001, max=0.999)
        p = p_dyn * (1 - active) + p_smb * active
        g = (2.0 * s_dyn * (1 - s_dyn) * (1 - active)
             + s_smb * (1 - s_smb) * active)
        return p, g

    def loss(self, *, sim, physical, config, domain, mask, dx, weight):
        # Evaluated on the snapshot's own grid: coarse H / active / smb
        # against the box-averaged (fractional) extent labels. For a
        # cell-constant p, sum_cell (p - y_i)^2 = 4^L [(p - ybar)^2 +
        # ybar (1 - ybar)], so the coarse Brier on the fraction is the fine
        # one minus an irreducible within-cell term; the one-sided form is
        # linear in y and restricts exactly.
        state = sim.at(self.time)
        level = state.level
        H = state.H
        smb = state.smb_coarse
        active = state.active
        mask_L = _restrict(mask, level)
        if self.two_sided:
            omega = _restrict(domain.domain_mask.to(torch.float32), level)
            target = mask_L
        else:
            omega = mask_L
            target = torch.ones_like(mask_L)

        c_data = self._c_data(level, dx, weight)
        eps = 0.0
        prior = None
        if self.logit_nuisance is not None and weight > 0.0:
            with torch.no_grad():
                Hd, smbd, ad = H.detach(), smb.detach(), active.detach()
                eps, prior = self.logit_nuisance.solve(
                    level=level, dx_level=dx * 2 ** level,
                    p_g=lambda e: self._p_g(Hd, smbd, ad, e),
                    omega=omega, target=target, c=c_data)

        p_extent = self._p_g(H, smb, active, eps)[0]
        J = config.loss_scale * c_data * (omega * (target - p_extent) ** 2).sum()
        if prior is not None:
            # Profile objective: the (detached) ε prior cost keeps absorption
            # visible in the loss trace.
            J = J + config.loss_scale * prior
        return J

    def residuals(self, *, sim, physical, config, domain, mask, dx):
        if self.logit_nuisance is None:
            return {}
        eps = self.logit_nuisance.eps_at(sim.at(self.time).level)
        return {} if eps is None else {"logit_eps": eps}


class BedObservation(Observation):
    """Bed misfit: flightline picks + bed-equals-DEM anchor where no ice.

    Time-independent (the bed does not evolve), so `required_times` is empty.
    The off-ice anchor uses the full DEM (topography + bathymetry), not the
    sea-level-clamped surface, so submarine bed is anchored to bathymetry.
    The anchor DEM is owned here (not read from `domain`) so RTO can perturb
    it jointly with the surface observation via a shared eps_S draw.
    """

    name = "bed"

    def __init__(self, *, bed_obs, bed_normed_coords, dem_anchor,
                 sigma: float, sigma_dem: float, nu: float,
                 weight: LossWeight, ny: int, nx: int):
        super().__init__(weight=weight)
        self.bed_obs = bed_obs
        self.bed_normed_coords = bed_normed_coords
        self.dem_anchor = dem_anchor
        self.sigma = sigma
        self.sigma_dem = sigma_dem
        self.nu = nu
        self.ny = ny
        self.nx = nx

    def loss(self, *, sim, physical, config, domain, mask, dx, weight):
        scale = config.loss_scale * dx ** 2
        bed_fine = physical.bed

        r_bed_grid = (bed_fine - self.dem_anchor) / self.sigma_dem * (1 - mask)
        J = _huber(r_bed_grid, self.nu).sum()
        # No scattered picks (no flightline file): only the off-ice anchor.
        if self.bed_obs.numel() > 0:
            bed_at_flightlines = grid_sample(
                bed_fine[None, None, :, :],
                self.bed_normed_coords[None, None, :, :],
                mode="bilinear", align_corners=False,
            ).squeeze()
            r_bed_fl = (bed_at_flightlines - self.bed_obs) / self.sigma
            J = J + _huber(r_bed_fl, self.nu).sum()
        return scale * weight * J

    def randomized(self, *, eps_bed=None, eps_S=None):
        eps_bed = torch.randn_like(self.bed_obs) if eps_bed is None else eps_bed
        eps_S = torch.randn_like(self.dem_anchor) if eps_S is None else eps_S
        # eps_S is shared with the surface observation's draw so the off-ice
        # anchor perturbation matches the surface perturbation, preserving the
        # historical RTO noise bookkeeping.
        return BedObservation(
            bed_obs=self.bed_obs + eps_bed * self.sigma,
            bed_normed_coords=self.bed_normed_coords,
            dem_anchor=self.dem_anchor + eps_S * self.sigma_dem,
            sigma=self.sigma, sigma_dem=self.sigma_dem, nu=self.nu,
            weight=self.weight, ny=self.ny, nx=self.nx)

    def diagnostics(self):
        """Rasterize the scattered picks onto the finest grid (NaN = no data),
        using the same normalized coordinates the loss feeds to grid_sample."""
        cn = self.bed_normed_coords[:, 0]
        rn = self.bed_normed_coords[:, 1]
        col = (((cn + 1.0) * self.nx - 1.0) / 2.0).round().long().clamp_(0, self.nx - 1)
        row = (((rn + 1.0) * self.ny - 1.0) / 2.0).round().long().clamp_(0, self.ny - 1)
        raster = torch.full((self.ny, self.nx), float("nan"),
                            dtype=torch.float32, device="cuda")
        if self.bed_obs.numel() > 0:
            raster[row, col] = self.bed_obs.to(torch.float32)
        return {"bed_obs": raster}


class SnowlineObservation(Observation):
    """Snowline (ELA-proxy) misfit at the composite's nominal epoch.

    The end-of-summer snowline product gives, per cell, the fraction of the
    glacierized subarea that retained snow (`snow_label` in [0, 1]). The model
    produces a logit by scaling the SMB field at the observation time, so
    sigmoid(SMB / s_smb) reads as P(cell is above the ELA). Restricted to
    `snow_mask` (valid, glacierized cells); Brier-scored.

    `two_sided=False` (historical) weights the score by the observed snow
    fraction, so only "snow observed, model bare" is penalized — the model may
    keep snow below the observed snowline for free. `two_sided=True` is the
    full Brier score against the fraction, penalizing both directions.
    """

    name = "snow"

    def __init__(self, *, snow_label, snow_mask, time: float, s_smb: float,
                 weight: LossWeight, two_sided: bool = False,
                 logit_nuisance: Optional[LogitNuisance] = None,
                 sigma_p: Optional[float] = None):
        super().__init__(weight=weight)
        self.logit_nuisance = logit_nuisance
        self.sigma_p = sigma_p
        self.snow_label = snow_label
        self.snow_mask = snow_mask
        self.time = time
        self.s_smb = s_smb
        self.two_sided = two_sided
        self._targets = {0: (snow_mask, snow_label)}

    @property
    def required_times(self):
        return (self.time,)

    def _target_at(self, level: int):
        """(snow_mask, snow_label) box-restricted to `level`: the mask
        becomes the valid fraction, the label its mask-weighted mean."""
        if level not in self._targets:
            m = _restrict(self.snow_mask, level)
            lab = _restrict_weighted(self.snow_label, self.snow_mask, level)
            self._targets[level] = (m, lab)
        return self._targets[level]

    def loss(self, *, sim, physical, config, domain, mask, dx, weight):
        # Evaluated on the snapshot's own grid (see ExtentObservation.loss
        # for why the Brier on box-averaged targets is the right coarse form).
        state = sim.at(self.time)
        level = state.level
        logits = state.smb_coarse / self.s_smb
        snow_mask, snow_label = self._target_at(level)
        if self.two_sided:
            omega, target = snow_mask, snow_label
        else:
            omega, target = snow_mask * snow_label, torch.ones_like(snow_label)

        if self.sigma_p is not None:
            c_data = weight * 4.0 ** level / (2.0 * self.sigma_p ** 2)
        else:
            c_data = weight * (dx * 2 ** level) ** 2 / 0.25 ** 2
        eps = 0.0
        prior = None
        if self.logit_nuisance is not None and weight > 0.0:
            with torch.no_grad():
                ld = logits.detach()

                def p_g(e):
                    s = torch.sigmoid(ld + e)
                    return s, s * (1 - s)
                eps, prior = self.logit_nuisance.solve(
                    level=level, dx_level=dx * 2 ** level,
                    p_g=p_g, omega=omega, target=target, c=c_data)

        y = torch.sigmoid(logits + eps)
        J = config.loss_scale * c_data * (omega * (target - y) ** 2).sum()
        if prior is not None:
            J = J + config.loss_scale * prior
        return J

    def residuals(self, *, sim, physical, config, domain, mask, dx):
        if self.logit_nuisance is None:
            return {}
        eps = self.logit_nuisance.eps_at(sim.at(self.time).level)
        return {} if eps is None else {"logit_eps": eps}

    def diagnostics(self):
        return {"snow_label": self.snow_label, "snow_mask": self.snow_mask}


class DhdtObservation(Observation):
    """Surface elevation-change-rate misfit (Huber) over the product's window.

    The model rate is a true two-snapshot difference over the observation
    window [t0, t1]:

        dHdt_model = (H(t1) - H(t0)) / (t1 - t0),

    compared against the observed rate (Hugonnet), with residuals normalized
    by the per-pixel uncertainty floored at `sigma_floor`. With a `noise`
    model the normalized residual is whitened by a unit-sigma Matérn member,
    and noise.sigma is a multiplier on the per-pixel std (σ=1 takes the
    product's reported error as the marginal).

    `legacy_final_step` reproduces the historical behavior for input files
    without time attrs: the rate over the final emitted step,
    (H_final - H_prev) / dt_final (the *actual* final step length — with a
    snapped step sequence this is not necessarily config.dt).
    """

    name = "dhdt"

    def __init__(self, *, dhdt, dhdt_err, dhdt_mask, t0: float, t1: float,
                 sigma_floor: float, nu: float, weight: LossWeight,
                 legacy_final_step: bool = False,
                 noise: Optional[MaternNoise] = None, noise_model=None,
                 name: Optional[str] = None):
        super().__init__(weight=weight)
        if name is not None:            # a second product (another window) needs its own key
            self.name = name
        self.dhdt = dhdt
        self.dhdt_err = dhdt_err
        self.dhdt_mask = dhdt_mask
        self.t0 = t0
        self.t1 = t1
        self.sigma_floor = sigma_floor
        self.nu = nu
        self.legacy_final_step = legacy_final_step
        self.noise = noise
        self.noise_model = noise_model

    @property
    def required_times(self):
        return () if self.legacy_final_step else (self.t0, self.t1)

    def model_rate(self, sim, space: str = "fine") -> torch.Tensor:
        """The model-side elevation-change rate this observation is compared
        against. Both `loss` and the drivers' dhdt diagnostic call this, so
        the VTI field shows exactly the quantity the misfit penalizes.
        `space` selects the fine (prolonged) or coarse grid."""
        if space not in ("fine", "coarse"):
            raise ValueError(f"space must be 'fine' or 'coarse', got {space!r}")
        if self.legacy_final_step:
            H1 = sim.final.H_fine if space == "fine" else sim.final.H
            H0 = sim.H_prev_fine if space == "fine" else sim.H_prev
            return (H1 - H0) / sim.final.dt_step
        s0, s1 = sim.at(self.t0), sim.at(self.t1)
        if space == "fine":
            return (s1.H_fine - s0.H_fine) / (self.t1 - self.t0)
        return (s1.H - s0.H) / (self.t1 - self.t0)

    def _sigma_pixel(self) -> torch.Tensor:
        """Per-pixel error std: the product's own error clamped at
        `sigma_floor`, times the noise model's multiplier when present (the
        Matérn member itself is registered with unit sigma)."""
        sigma = torch.clamp(self.dhdt_err, min=self.sigma_floor)
        if self.noise is not None:
            sigma = sigma * self.noise.sigma
        return sigma

    def residuals(self, *, sim, physical, config, domain, mask, dx):
        dhdt_model = self.model_rate(sim, "fine")
        r = (dhdt_model - self.dhdt) / self._sigma_pixel() * self.dhdt_mask
        z = _whiten(self.noise_model, r) if self.noise is not None else r
        return {"r": r, "z": z}

    def loss(self, *, sim, physical, config, domain, mask, dx, weight):
        if self.noise is not None:
            dhdt_model = self.model_rate(sim, "fine")
            r = (dhdt_model - self.dhdt) / self._sigma_pixel() * self.dhdt_mask
            z = _whiten(self.noise_model, r)
            z, fp_prior = self._apply_fingerprints(z, sim.final.level, self.nu)
            J = config.loss_scale * weight * _huber(z, self.nu).sum()
            if fp_prior is not None:
                J = J + config.loss_scale * weight * fp_prior
            return J
        scale = config.loss_scale * dx ** 2
        dhdt_model = self.model_rate(sim, "fine")
        sigma = torch.clamp(self.dhdt_err, min=self.sigma_floor)
        r = (dhdt_model - self.dhdt) / sigma * self.dhdt_mask
        return scale * weight * _huber(r, self.nu).sum()

    def randomized(self, *, eps_dhdt=None):
        eps_dhdt = torch.randn_like(self.dhdt) if eps_dhdt is None else eps_dhdt
        sigma = self._sigma_pixel()
        if self.noise is not None:
            # Unit-sigma correlated field, scaled by the per-pixel std.
            eps_dhdt = GGaPPMap.apply(self.noise_model, eps_dhdt.contiguous())
        return DhdtObservation(
            dhdt=self.dhdt + eps_dhdt * sigma * self.dhdt_mask,
            dhdt_err=self.dhdt_err, dhdt_mask=self.dhdt_mask,
            t0=self.t0, t1=self.t1, sigma_floor=self.sigma_floor, nu=self.nu,
            weight=self.weight, legacy_final_step=self.legacy_final_step,
            noise=self.noise, noise_model=self.noise_model)

    def diagnostics(self):
        return {"dhdt": self.dhdt, "dhdt_err": self.dhdt_err,
                "dhdt_mask": self.dhdt_mask}


def build_divide_mask(rgi_label, domain_mask, u_obs, v_obs, v_mask, *,
                      u_conf: float, buffer_px: int) -> torch.Tensor:
    """0/1 float mask of drainage-divide pixels where cross-basin flux is
    penalized.

    Starts from inter-glacier boundaries (4-neighbor pairs of distinct
    non-negative RGI labels), dilates by `buffer_px`, then removes pixels the
    velocity mosaic identifies as confluences — valid observations moving
    faster than `u_conf`. At a true divide the ice barely moves (or feature
    tracking has no data), so slow/no-data boundary pixels are kept; where a
    tributary legitimately crosses its RGI boundary into a trunk, the observed
    speed is high and the penalty is dropped. Device-agnostic and standalone
    so it can be validated against the inputs without building a problem.
    """
    lab = rgi_label
    boundary = torch.zeros_like(lab, dtype=torch.bool)
    for a_sl, b_sl in (
        ((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
        ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
    ):
        a, b = lab[a_sl], lab[b_sl]
        differs = (a >= 0) & (b >= 0) & (a != b)
        boundary[a_sl] |= differs
        boundary[b_sl] |= differs
    if buffer_px > 0:
        boundary = torch.nn.functional.max_pool2d(
            boundary[None, None].float(), 2 * buffer_px + 1,
            stride=1, padding=buffer_px)[0, 0] > 0
    speed = torch.sqrt(u_obs ** 2 + v_obs ** 2)
    confluence = (v_mask > 0) & (speed > u_conf)
    return (boundary & ~confluence & domain_mask.bool()).to(torch.float32)


class DivideFluxObservation(Observation):
    """Cross-basin flux penalty: no significant ice flux across drainage
    divides between distinct RGI glaciers.

    A prior-style pseudo-observation, not a data product: it encodes the
    expert knowledge that inventory drainage divides (Bering/Yahtse, ...) do
    not exchange mass, which the Matern bed prior is too local to express and
    which the Huberized velocity misfit is too forgiving to enforce (a
    fictitious ice stream saturates the robust loss and is written off as an
    outlier). The penalty is deliberately quadratic — large fictitious fluxes
    must stay expensive — on the flux-magnitude excess over a small allowance
    `q0` (legitimate near-divide flux is O(H * a few m/yr)), normalized by
    `q_scale`. `randomized` is the identity: there is no observational noise
    to perturb.
    """

    name = "divide"

    def __init__(self, *, divide_mask, time: float, q0: float, q_scale: float,
                 weight: LossWeight):
        super().__init__(weight=weight)
        self.divide_mask = divide_mask
        self.time = time
        self.q0 = q0
        self.q_scale = q_scale

    @property
    def required_times(self):
        return (self.time,)

    def loss(self, *, sim, physical, config, domain, mask, dx, weight):
        state = sim.at(self.time)
        # Deliberately the depth-AVERAGED velocity (not surface): ice flux is
        # exactly u_bar * H in both stress schemes.
        u = (state.u_fine[:, 1:] + state.u_fine[:, :-1]) / 2.0
        v = (state.v_fine[1:, :] + state.v_fine[:-1, :]) / 2.0
        q = state.H_fine * torch.sqrt(u ** 2 + v ** 2 + 1e-6)
        excess = torch.relu(q - self.q0) / self.q_scale
        return config.loss_scale * weight * dx ** 2 * (
            self.divide_mask * excess ** 2).sum()

    def diagnostics(self):
        return {"divide_mask": self.divide_mask}


class BedSlopeObservation(Observation):
    """Flow-aligned bed-slope penalty: J = sum mask * |u . grad(B) / s_scale|^p.

    A prior-style pseudo-observation (like DivideFluxObservation), not a data
    product: it penalizes the bed for having slopes transverse to the flow —
    riegel-like ramps whose gradient points along the velocity — encoding the
    expectation that the bed under flowing ice varies little along flowlines
    (equivalently, that flow follows bed contours). u . grad(B) is exactly the
    bed-parallel-flow vertical velocity at the base, so the penalty also reads
    as "sliding ice should not be forced up/down bed steps". Whether this is a
    *sensible* prior is an open question — it fights real overdeepenings and
    riegels — hence opt-in with its own weight.

    `velocity` selects which model velocity weights the gradient: "base"
    (u - ud, glide's sliding velocity — the default, and the physically
    natural choice since only sliding ice feels the bed), "surface", or
    "average" (depth-averaged). Under SSA all three coincide.

    The dot product is formed in geographic coordinates: `col_sign`/`row_sign`
    map the raster's index axes onto +x/+y (build() derives them from the
    input file's coordinate ordering, so north-up rasters get row_sign = -1).
    `s0` is a deadband (m/yr of u.grad(B)) below which no penalty accrues;
    `s_scale` normalizes the excess before the exponent `p`; `eps` (m/yr)
    smooths |.| at zero so p <= 1 keeps finite gradients. `randomized` is the
    identity: there is no observational noise to perturb.
    """

    name = "bedslope"

    def __init__(self, *, time: float, p: float, s0: float, s_scale: float,
                 eps: float, velocity: str, col_sign: float, row_sign: float,
                 weight: LossWeight):
        super().__init__(weight=weight)
        if velocity not in ("base", "surface", "average"):
            raise ValueError(
                f"velocity must be 'base', 'surface' or 'average', got "
                f"{velocity!r}")
        self.time = time
        self.p = p
        self.s0 = s0
        self.s_scale = s_scale
        self.eps = eps
        self.velocity = velocity
        self.col_sign = col_sign
        self.row_sign = row_sign

    @property
    def required_times(self):
        return (self.time,)

    def flow_aligned_slope(self, state, bed, dx) -> torch.Tensor:
        """Cell-centered u . grad(B) (m/yr) — the quantity the loss penalizes.
        Exposed so drivers can dump it as a per-iteration diagnostic."""
        if self.velocity == "base":
            u_f, v_f = state.u_base_fine, state.v_base_fine
        elif self.velocity == "surface":
            u_f, v_f = state.u_surf_fine, state.v_surf_fine
        else:
            u_f, v_f = state.u_fine, state.v_fine
        u = (u_f[:, 1:] + u_f[:, :-1]) / 2.0
        v = (v_f[1:, :] + v_f[:-1, :]) / 2.0
        dB_drow, dB_dcol = torch.gradient(bed, spacing=dx)
        return u * self.col_sign * dB_dcol + v * self.row_sign * dB_drow

    def loss(self, *, sim, physical, config, domain, mask, dx, weight):
        state = sim.at(self.time)
        dot = self.flow_aligned_slope(state, physical.bed, dx)
        # One-sided: only ice forced UP bed steps (u.grad(B) > 0) is penalized.
        # Do not write this as sqrt(relu(dot)**2): for 0 < dot < ~3e-23 m/yr
        # (numerically-zero velocities on ice-free cells) the square underflows
        # to exactly 0 in float32 and sqrt's backward divides by zero, seeding
        # NaNs in the bed cotangent (juneau: 1240 cells -> the conditioning
        # PCG rejects the NaN rhs and the bed silently stops updating).
        r = torch.relu(dot) / self.s_scale
        #mag = torch.sqrt(dot ** 2 + self.eps ** 2)
        #r = torch.relu(mag - self.s0) / self.s_scale
        return config.loss_scale * weight * dx ** 2 * (mask * r ** self.p).sum()


# --------------------------------------------------------------------------- #
# Specs: what a domain config declares                                        #
# --------------------------------------------------------------------------- #

def _resolve_time(override: Optional[float], da, *, fallback: float,
                  what: str) -> float:
    if override is not None:
        return float(override)
    nominal, _, _ = read_time_attrs(da, fallback=fallback, what=what)
    return nominal


@dataclass(frozen=True)
class SurfaceSpec:
    weight: LossWeight = 2e-5
    sigma: float = 10.0             # ignored when `noise` is set (noise.sigma)
    nu: float = 1.0                 # pseudo-Huber threshold (NOT noise.nu)
    time: Optional[float] = None    # override; None -> file attr -> t_end
    # Correlated error model: whitened misfit, weight must be 1 (see the
    # module-level note on MaternNoise).
    noise: Optional[MaternNoise] = None

    def build(self, ctx: ObservationBuildContext) -> SurfaceObservation:
        cfg = ctx.config
        time = _resolve_time(self.time, ctx.gridded_data.elevation,
                             fallback=cfg.t_end, what="surface DEM (elevation)")
        sigma, noise_model = self.sigma, None
        if self.noise is not None:
            noise_model = _require_priors(ctx, "SurfaceSpec") \
                .noise_model("srf", self.noise)
            sigma = self.noise.sigma
        return SurfaceObservation(
            S_obs=torch.clamp(ctx.domain.dem, min=0.0), time=time,
            sigma=sigma, nu=self.nu, weight=self.weight,
            noise=self.noise, noise_model=noise_model,
            priors=getattr(ctx, "priors", None))


@dataclass(frozen=True)
class VelocitySpec:
    weight: LossWeight = 2e-5
    sigma: float = 10.0             # ignored when `noise` is set (noise.sigma)
    nu: float = 1.0                 # pseudo-Huber threshold (NOT noise.nu)
    surge_biased: bool = False
    time: Optional[float] = None
    outlier_threshold: Optional[float] = None
    # Beta(alpha, 1) priors on the per-glacier mosaic under-read factor eta
    # (surge_biased only); see VelocityObservation.
    alpha_surge: float = 2.0
    alpha_nonsurge: float = 6.0
    # False (historical): v_mask == 0 pixels are observations of ~0 velocity.
    # True: they are undefined and dropped from the misfit.
    mask_unobserved: bool = False
    # Correlated error model per component: whitened misfit, weight must be 1.
    noise: Optional[MaternNoise] = None

    def build(self, ctx: ObservationBuildContext) -> VelocityObservation:
        cfg = ctx.config
        gd = ctx.gridded_data
        sigma, noise_model = self.sigma, None
        if self.noise is not None:
            noise_model = _require_priors(ctx, "VelocitySpec") \
                .noise_model("vel", self.noise)
            sigma = self.noise.sigma
        domain_mask = ctx.domain.domain_mask
        time = _resolve_time(self.time, gd.vx, fallback=cfg.t_end,
                             what="velocity mosaic (vx)")
        u_obs = torch.tensor(gd.vx.values, dtype=torch.float32, device="cuda") \
            .nan_to_num().masked_fill(~domain_mask, 0.0)
        v_obs = torch.tensor(gd.vy.values, dtype=torch.float32, device="cuda") \
            .nan_to_num().masked_fill(~domain_mask, 0.0)
        v_mask = torch.tensor(gd.vmask.values, dtype=torch.float32, device="cuda") \
            .nan_to_num().masked_fill(~domain_mask, 0.0)
        return VelocityObservation(
            u_obs=u_obs, v_obs=v_obs, v_mask=v_mask, time=time,
            sigma=sigma, nu=self.nu, surge_biased=self.surge_biased,
            weight=self.weight, outlier_threshold=self.outlier_threshold,
            alpha_surge=self.alpha_surge, alpha_nonsurge=self.alpha_nonsurge,
            mask_unobserved=self.mask_unobserved,
            noise=self.noise, noise_model=noise_model)


@dataclass(frozen=True)
class ExtentSpec:
    weight: LossWeight = 2e-4
    s_H: float = 10.0
    smb_dt: Optional[float] = None  # None -> config.dt (the historical factor)
    time: Optional[float] = None
    two_sided: bool = False         # False: penalize missing ice only
    # Profiled GP nuisance on the extent logit (see LogitNuisance): sigma in
    # logit units (a margin misplacement δ is ε ≈ δ·|∇H|/s_H), l the
    # along-margin coherence of the extent error; nugget must be 0.
    logit_error: Optional[MaternNoise] = None
    nuisance_inner_steps: int = 2
    # Amplitude bound on the logit-error field (logits): ε = eps_max·tanh(u/
    # eps_max). Below the bound the Gaussian model is unchanged; full class
    # flips become unreachable, so a coherent missing tongue keeps a Brier
    # floor instead of being explained away (the Gaussian cost is linear in
    # area, like the Brier's, so WITHOUT the bound every coherent miss larger
    # than l is absorbed or fought as one block, area-independently). Read as
    # the largest credible outline misplacement: eps_max ≈ 3·δ_max/W_t with
    # W_t the margin transition width (distance for H to reach ~3·s_H).
    # None = unbounded (the historical model).
    eps_max: Optional[float] = None
    # Per-pixel class-probability noise std (the white component of the error
    # model). Setting it replaces the legacy weight·dx²/s_B² scale with
    # 4^L/(2σ_p²) and pins weight == 1 by contract (GlacierProblem raises);
    # the legacy weight corresponds to σ_p = s_B/√(2·weight·dx²).
    sigma_p: Optional[float] = None

    def build(self, ctx: ObservationBuildContext) -> ExtentObservation:
        cfg = ctx.config
        time = _resolve_time(self.time, ctx.gridded_data.rgi_mask,
                             fallback=cfg.t_end, what="RGI extent (rgi_mask)")
        smb_dt = cfg.dt if self.smb_dt is None else self.smb_dt
        if self.sigma_p is not None and not self.sigma_p > 0.0:
            raise ValueError(f"ExtentSpec.sigma_p must be > 0, got {self.sigma_p}")
        nuis = (LogitNuisance(self.logit_error, self.nuisance_inner_steps,
                              eps_max=self.eps_max)
                if self.logit_error is not None else None)
        return ExtentObservation(time=time, s_H=self.s_H, smb_dt=smb_dt,
                                 weight=self.weight, two_sided=self.two_sided,
                                 logit_nuisance=nuis, sigma_p=self.sigma_p)


@dataclass(frozen=True)
class BedSpec:
    weight: LossWeight = 2e-5
    sigma: float = 10.0
    sigma_dem: float = 10.0   # off-ice anchor noise (historically sigma_s)
    nu: float = 1.0

    def build(self, ctx: ObservationBuildContext) -> BedObservation:
        gd = ctx.gridded_data
        fl = ctx.flightlines_df
        if len(fl) == 0:
            # No radar picks: an empty (0, 3) tensor keeps the off-ice DEM
            # anchor alive and skips the pick term (BedObservation.loss).
            flightlines = torch.zeros((0, 3), dtype=torch.float32, device="cuda")
        else:
            flightlines = torch.tensor(
                fl[["x", "y", "bed"]].values, dtype=torch.float32, device="cuda")
        xmin, xmax = gd.x.min().item(), gd.x.max().item()
        ymin, ymax = gd.y.min().item(), gd.y.max().item()
        col_normed = 2.0 * ((flightlines[:, 0] - xmin) / (xmax - xmin)) - 1
        row_normed = -(2.0 * ((flightlines[:, 1] - ymin) / (ymax - ymin)) - 1)
        bed_normed_coords = torch.stack([col_normed, row_normed], dim=-1)
        return BedObservation(
            bed_obs=flightlines[:, 2], bed_normed_coords=bed_normed_coords,
            dem_anchor=ctx.domain.dem, sigma=self.sigma,
            sigma_dem=self.sigma_dem, nu=self.nu, weight=self.weight,
            ny=ctx.ny, nx=ctx.nx)


@dataclass(frozen=True)
class SnowlineSpec:
    weight: LossWeight = 2e-4
    s_smb: float = 0.2
    time: Optional[float] = None
    two_sided: bool = False         # False: penalize missing snow only
    # Profiled GP nuisance on the SMB logit (coherent ELA-displacement error);
    # sigma in logit units (ELA shift ΔELA·|∂smb/∂z|/s_smb); nugget must be 0.
    logit_error: Optional[MaternNoise] = None
    nuisance_inner_steps: int = 2
    # Amplitude bound on the logit-error field (see ExtentSpec.eps_max): the
    # largest credible coherent ELA misplacement in logits; None = unbounded.
    eps_max: Optional[float] = None
    # Per-pixel snow-fraction noise std; same contract as ExtentSpec.sigma_p
    # (legacy weight ↔ σ_p = 0.25/√(2·weight·dx²)).
    sigma_p: Optional[float] = None

    def build(self, ctx: ObservationBuildContext) -> Optional[SnowlineObservation]:
        sd = ctx.snowline_data
        if sd is None:
            return None

        expected = (ctx.ny, ctx.nx)
        if sd.snow_fraction.shape != expected:
            raise ValueError(
                f"snowline grid {tuple(sd.snow_fraction.shape)} does not match "
                f"the cropped model grid {expected}; rebuild "
                f"{ctx.config.snowline_filename} on the DEM grid."
            )

        cfg = ctx.config
        time = _resolve_time(self.time, sd.snow_fraction, fallback=cfg.t_end,
                             what="snowline composite (snow_fraction)")
        domain_mask = ctx.domain.domain_mask
        snow_label = torch.tensor(
            sd.snow_fraction.values, dtype=torch.float32, device="cuda"
        ).nan_to_num().clamp_(0.0, 1.0)
        glacier_fraction = torch.tensor(
            sd.glacier_fraction.values, dtype=torch.float32, device="cuda"
        ).nan_to_num()
        snow_mask = ((glacier_fraction > 0.0) & domain_mask).to(torch.float32)
        snow_label = snow_label.masked_fill(snow_mask == 0.0, 0.0)
        return SnowlineObservation(
            snow_label=snow_label, snow_mask=snow_mask, time=time,
            s_smb=self.s_smb, weight=self.weight, two_sided=self.two_sided,
            logit_nuisance=(LogitNuisance(self.logit_error,
                                          self.nuisance_inner_steps,
                                          eps_max=self.eps_max)
                            if self.logit_error is not None else None),
            sigma_p=self.sigma_p)


@dataclass(frozen=True)
class DhdtSpec:
    weight: LossWeight = 2e-5
    sigma_floor: float = 0.5
    nu: float = 1.0                 # pseudo-Huber threshold (NOT noise.nu)
    t0: Optional[float] = None    # override; None -> file attrs -> legacy mode
    t1: Optional[float] = None
    # Correlated error model: noise.sigma multiplies the per-pixel product
    # error (the Matérn member has unit sigma); whitened misfit, weight must be 1.
    noise: Optional[MaternNoise] = None
    # A second dh/dt product over another window (e.g. the MEaSUREs/ITS_LIVE
    # 1992-2023 record next to ATL15 2019-2026): `filename` names another
    # gridded_dhdt_*.nc under model_inputs (built by make_dhdt.py --name),
    # loaded here on the same crop; `name` keys its loss term, noise model
    # and residuals (must differ from "dhdt"). None -> config.dhdt_filename.
    filename: Optional[str] = None
    name: Optional[str] = None

    def build(self, ctx: ObservationBuildContext) -> Optional[DhdtObservation]:
        name = self.name or "dhdt"
        if self.filename is None:
            dd = ctx.dhdt_data
            filename = ctx.config.dhdt_filename
        else:
            from pathlib import Path
            from .problem import _crop_to_factor, _open_eager   # lazy: problem imports us
            filename = self.filename
            path = Path(ctx.config.base_dir) / "model_inputs" / filename
            if self.name is None:
                raise ValueError(f"DhdtSpec(filename={filename!r}) needs a distinct `name`")
            dd = (_crop_to_factor(_open_eager(path), 2 ** ctx.config.n_levels)
                  if path.exists() else None)
        if dd is None:
            return None
        noise_model = None
        if self.noise is not None:
            noise_model = _require_priors(ctx, "DhdtSpec").noise_model(
                name, dataclasses.replace(self.noise, sigma=1.0))

        expected = (ctx.ny, ctx.nx)
        if dd.dhdt.shape != expected:
            raise ValueError(
                f"dhdt grid {tuple(dd.dhdt.shape)} does not match the cropped "
                f"model grid {expected}; rebuild {filename} "
                f"on the DEM grid."
            )

        cfg = ctx.config
        attrs = getattr(dd.dhdt, "attrs", {})
        legacy = False
        if self.t0 is not None and self.t1 is not None:
            t0, t1 = float(self.t0), float(self.t1)
        elif "time_start" in attrs and "time_end" in attrs:
            t0, t1 = float(attrs["time_start"]), float(attrs["time_end"])
        else:
            # Legacy input file: no window recorded. Fall back to the
            # historical final-step rate so old model_inputs keep working.
            warnings.warn(
                f"dhdt product ({filename}): no time_start/time_end "
                f"attrs; falling back to the legacy final-step rate "
                f"(H(t_end) - H_prev)/dt. Rebuild with preprocessing/"
                f"make_dhdt.py to compare over the true observation window."
            )
            t0, t1 = cfg.t_end - cfg.dt, cfg.t_end
            legacy = True
        if t1 <= t0:
            raise ValueError(f"dhdt window must have t1 > t0, got [{t0}, {t1}]")

        domain_mask = ctx.domain.domain_mask
        dhdt_raw = torch.tensor(dd.dhdt.values, dtype=torch.float32,
                                device="cuda")
        dhdt_err_raw = torch.tensor(dd.dhdt_err.values, dtype=torch.float32,
                                    device="cuda")
        valid = torch.isfinite(dhdt_raw) & torch.isfinite(dhdt_err_raw)
        dhdt_mask = (valid & domain_mask).to(torch.float32)
        dhdt = dhdt_raw.nan_to_num().masked_fill(dhdt_mask == 0.0, 0.0)
        dhdt_err = dhdt_err_raw.nan_to_num().masked_fill(dhdt_mask == 0.0, 0.0)
        return DhdtObservation(
            dhdt=dhdt, dhdt_err=dhdt_err, dhdt_mask=dhdt_mask, t0=t0, t1=t1,
            sigma_floor=self.sigma_floor, nu=self.nu, weight=self.weight,
            legacy_final_step=legacy,
            noise=self.noise, noise_model=noise_model, name=name)


@dataclass(frozen=True)
class DivideFluxSpec:
    """Opt-in cross-basin flux penalty (see DivideFluxObservation).

    `u_conf` and `buffer_px` shape the divide mask (validate with the
    `divide_mask` field in the observations PVD dump before trusting a run);
    `z_min` optionally restricts the penalty to divides above an elevation,
    for domains where slow quiescent trunks would otherwise keep confluence
    pixels in the mask. `q0`/`q_scale` are in m^2/yr of per-unit-width flux
    (H * speed): the default allowance corresponds to e.g. 300 m of ice
    moving ~15 m/yr.
    """
    weight: LossWeight = 1e-5
    u_conf: float = 30.0            # obs speed (m/yr) marking a confluence
    buffer_px: int = 1
    q0: float = 5e3                 # flux allowance (m^2/yr)
    q_scale: float = 1e4            # residual normalization (m^2/yr)
    z_min: Optional[float] = None   # keep only divides above this elevation
    time: Optional[float] = None    # evaluation epoch; None -> t_end

    def build(self, ctx: ObservationBuildContext) -> Optional["DivideFluxObservation"]:
        cfg = ctx.config
        gd = ctx.gridded_data
        u_obs = torch.tensor(gd.vx.values, dtype=torch.float32,
                             device="cuda").nan_to_num()
        v_obs = torch.tensor(gd.vy.values, dtype=torch.float32,
                             device="cuda").nan_to_num()
        v_mask = torch.tensor(gd.vmask.values, dtype=torch.float32,
                              device="cuda").nan_to_num()
        divide_mask = build_divide_mask(
            ctx.domain.rgi_label, ctx.domain.domain_mask, u_obs, v_obs,
            v_mask, u_conf=self.u_conf, buffer_px=self.buffer_px)
        if self.z_min is not None:
            divide_mask = divide_mask * (ctx.domain.dem > self.z_min)
        if divide_mask.sum() == 0:
            warnings.warn("DivideFluxSpec: divide mask is empty (no "
                          "inter-glacier boundaries survived the filters); "
                          "dropping the term.")
            return None
        time = float(self.time) if self.time is not None else float(cfg.t_end)
        return DivideFluxObservation(
            divide_mask=divide_mask, time=time, q0=self.q0,
            q_scale=self.q_scale, weight=self.weight)


@dataclass(frozen=True)
class BedSlopeSpec:
    """Opt-in flow-aligned bed-slope penalty (see BedSlopeObservation).

    Scaling: `s_scale` sets the m/yr of u.grad(B) that costs O(1) per cell
    (before `weight`); the default corresponds to e.g. 100 m/yr of sliding
    across a 10% bed slope. `s0` is a free allowance below which nothing is
    penalized; `p` is the exponent (p = 2 quadratic, p = 1 an L1-like penalty
    smoothed by `eps`). `velocity` picks base/surface/average model velocity.
    """
    weight: LossWeight = 1e-5
    p: float = 2.0
    s0: float = 0.0                 # deadband on |u.grad(B)| (m/yr)
    s_scale: float = 10.0           # normalization of the excess (m/yr)
    eps: float = 1e-3               # |.| smoothing near zero (m/yr)
    velocity: str = "base"          # "base" | "surface" | "average"
    time: Optional[float] = None    # evaluation epoch; None -> t_end

    def build(self, ctx: ObservationBuildContext) -> "BedSlopeObservation":
        cfg = ctx.config
        gd = ctx.gridded_data
        # Map raster index axes onto geographic +x/+y so the dot product uses
        # the same velocity convention the vx/vy misfit does (north-up files
        # have descending y, hence row_sign = -1).
        col_sign = 1.0 if float(gd.x.values[1] - gd.x.values[0]) > 0 else -1.0
        row_sign = 1.0 if float(gd.y.values[1] - gd.y.values[0]) > 0 else -1.0
        time = float(self.time) if self.time is not None else float(cfg.t_end)
        return BedSlopeObservation(
            time=time, p=self.p, s0=self.s0, s_scale=self.s_scale,
            eps=self.eps, velocity=self.velocity, col_sign=col_sign,
            row_sign=row_sign, weight=self.weight)


def default_observation_specs(config=None) -> tuple:
    """The standard six products with library-default hyperparameters.

    When given a config that still carries the legacy global fields
    (sigma_s, lambda_s, ...), those seed the defaults so pre-migration domain
    configs behave identically; on a migrated config the plain defaults apply.
    """
    g = (lambda name, default: getattr(config, name, default)) if config \
        else (lambda name, default: default)
    # Under bed conditioning the bed data lives in the prior map itself, so
    # the soft likelihood must carry weight 0 (double counting otherwise).
    # The observation object is kept: its diagnostics feed the PVD dump.
    bed_conditioned = bool(getattr(getattr(config, "bed_conditioning", None),
                                   "enabled", False))
    return (
        SurfaceSpec(weight=g("lambda_s", 2e-5), sigma=g("sigma_s", 10.0),
                    nu=g("nu_s", 1.0)),
        VelocitySpec(weight=g("lambda_u", 2e-5), sigma=g("sigma_u", 10.0),
                     nu=g("nu_u", 1.0),
                     surge_biased=g("surge_biased_likelihood", False)),
        ExtentSpec(weight=g("lambda_e", 2e-4), s_H=g("s_H", 10.0)),
        BedSpec(weight=0.0 if bed_conditioned else g("lambda_bed", 2e-5),
                sigma=g("sigma_bed", 10.0),
                sigma_dem=g("sigma_s", 10.0), nu=g("nu_bed", 1.0)),
        SnowlineSpec(weight=g("lambda_snow", 2e-4), s_smb=g("s_smb", 0.2)),
        DhdtSpec(weight=g("lambda_dhdt", 2e-5), sigma_floor=g("sigma_dhdt", 0.5),
                 nu=g("nu_dhdt", 1.0)),
    )
