"""
Loss assembly (data misfit + prior) and RTO mean perturbations.

The per-product data misfits live on the Observation subclasses in
`observations.py`; this module keeps the shared numerical helpers, the prior
terms, and the `LossTerms` container. The same loss is used by deterministic
MAP (zero prior means) and RTO sampling (non-zero whitened-space means).
"""
import math
from dataclasses import dataclass, field
from typing import Optional

import torch

from ggapp.torch import GGaPPWhiten


def _huber(r, nu):
    """Huberized squared residual: sqrt(1 + (r/nu)^2) - 1 times nu^2."""
    return nu ** 2 * (torch.sqrt(1 + (r / nu) ** 2) - 1)


@dataclass
class PriorMeans:
    """Whitened-space prior means.

    For deterministic MAP, leave fields at None (they default to zero). For
    randomize-then-optimize, populate each field with N(0, I) draws shaped
    like the corresponding whitened parameter tensor.
    """
    z_bed:        Optional[torch.Tensor] = None
    z_bed_mean:   Optional[torch.Tensor] = None
    z_log_beta:   Optional[torch.Tensor] = None
    z_log_beta_mean: Optional[torch.Tensor] = None
    z_pbias:      Optional[torch.Tensor] = None
    # Temperature bias field: zero mean until the RTO migration wires it into
    # sample_like (its draw must be appended AFTER the existing draw order so
    # QMC dimension assignment stays stable).
    z_tbias:      Optional[torch.Tensor] = None
    z_log_mf:     Optional[torch.Tensor] = None
    z_log_rf:     Optional[torch.Tensor] = None
    # Precip-depletion scalars. Present so compute_prior can ask for their mean
    # uniformly; left out of sample_like (RTO does not yet perturb them), so they
    # default to a zero mean — i.e. no per-sample perturbation.
    z_tau:        Optional[torch.Tensor] = None
    z_z0:         Optional[torch.Tensor] = None
    # Enthalpy-model scalars: same status as tau/z0 — zero mean until the RTO
    # migration wires them into sample_like (new draws appended AFTER the
    # existing draw order).
    z_log_H_atm:   Optional[torch.Tensor] = None
    z_logit_cloud: Optional[torch.Tensor] = None

    @classmethod
    def zeros(cls) -> "PriorMeans":
        return cls()

    @classmethod
    def sample_like(cls, params) -> "PriorMeans":
        """Draw N(0, I) perturbations with the same shapes as `params`."""
        return cls(
            z_bed=torch.randn_like(params.z_bed),
            z_bed_mean=torch.randn_like(params.z_bed_mean),
            z_log_beta=torch.randn_like(params.z_log_beta),
            z_pbias=torch.randn_like(params.z_pbias),
            z_log_mf=torch.randn_like(params.z_log_mf),
            z_log_rf=torch.randn_like(params.z_log_rf),
        )

    def value(self, name: str, like: torch.Tensor) -> torch.Tensor:
        v = getattr(self, name)
        return torch.zeros_like(like) if v is None else v


# Display labels for the log banner, keyed by Observation.name. Unknown names
# fall back to the raw key so custom observations still log.
_LOG_LABELS = {
    "srf": "Srf", "vel": "U", "extent": "Ext",
    "bed": "Bed", "snow": "Snow", "dhdt": "dHdt", "divide": "Div",
    "bedslope": "BSlope",
}


@dataclass
class LossTerms:
    """All loss terms from one evaluation.

    `data_terms` is keyed by Observation.name — one entry per observation the
    domain actually has (absent products simply have no entry). The historical
    `J_srf` ... `J_dhdt` accessors read from it, returning 0 for a missing
    product, so existing analysis code keeps working.
    """
    data_terms: dict = field(default_factory=dict)
    J_prior_bed:      torch.Tensor = None
    J_prior_bed_mean: torch.Tensor = None
    J_prior_beta:     torch.Tensor = None
    J_prior_pbias:    torch.Tensor = None
    J_prior_tbias:    torch.Tensor = None
    J_prior_h_atm:    torch.Tensor = None
    J_prior_cloud:    torch.Tensor = None
    J_prior_smb:      torch.Tensor = None
    J_prior_beta_mean: torch.Tensor = None

    def _term(self, name: str):
        if name in self.data_terms:
            return self.data_terms[name]
        return torch.zeros((), device=self.J_prior_bed.device) \
            if isinstance(self.J_prior_bed, torch.Tensor) else 0.0

    @property
    def J_srf(self): return self._term("srf")
    @property
    def J_vel(self): return self._term("vel")
    @property
    def J_extent(self): return self._term("extent")
    @property
    def J_bed(self): return self._term("bed")
    @property
    def J_snow(self): return self._term("snow")
    @property
    def J_dhdt(self): return self._term("dhdt")

    @property
    def J_data(self):
        return sum(self.data_terms.values())

    @property
    def J_prior(self):
        return (self.J_prior_bed + self.J_prior_bed_mean + self.J_prior_beta
                + self.J_prior_pbias + self.J_prior_tbias
                + self.J_prior_h_atm + self.J_prior_cloud + self.J_prior_smb
                + (self.J_prior_beta_mean if self.J_prior_beta_mean is not None else 0.0))

    @property
    def J(self):
        return self.J_data + self.J_prior

    def log(self, i: int) -> None:
        bar = "=" * 60
        print(bar)
        print(f"Iteration: {i}, Total Loss: {self.J.item():.2f}, "
              f"Data Loss: {float(self.J_data):.2f}, "
              f"Prior Loss: {self.J_prior.item():.2f}")
        print(", ".join(
            f"{_LOG_LABELS.get(name, name)} Loss: {float(term):.2f}"
            for name, term in self.data_terms.items()))
        beta_mean = (f"Beta Mean Prior: {float(self.J_prior_beta_mean):.2f}, "
                     if self.J_prior_beta_mean is not None and float(self.J_prior_beta_mean) != 0.0 else "")
        print(f"Bed Prior: {float(self.J_prior_bed):.2f}, "
              f"Bed Mean Prior: {float(self.J_prior_bed_mean):.2f}, "
              f"Beta Prior: {float(self.J_prior_beta):.2f}, "
              f"{beta_mean}"
              f"Pbias Prior: {float(self.J_prior_pbias):.2f}, "
              f"Tbias Prior: {float(self.J_prior_tbias):.2f}, "
              f"H_atm Prior: {float(self.J_prior_h_atm):.2f}, "
              f"Cloud Prior: {float(self.J_prior_cloud):.2f}")
        print(bar)


_SMB_BLOCK = ("z_log_H_atm", "z_logit_cloud")


# Whitened-parameter -> prior-hyperparameter field, for effective-dimension
# resolution of the influence caps. Scalars (z_log_mf, ...) have d = 1.
_PRIOR_OF = {
    "z_bed": "bed_prior", "z_bed_mean": "mean_prior",
    "z_log_beta": "log_beta_prior", "z_log_beta_mean": "log_beta_mean_prior",
    "z_pbias": "pbias_prior",
    "z_tbias": "tbias_prior", "z_log_H_atm": "h_atm_prior",
    "z_logit_cloud": "cloud_prior",
}


def resolve_influence_caps(caps, config, ny: int, nx: int, dx: float) -> dict:
    """Resolve {name: C_z | (C_z, d)} to {name: (C_z, d_eff)}.

    C_z is the PER-MODE trust level (prior-stds per effective dof); the cap
    applied to the block's joint score norm is sqrt(d_eff)*C_z — a correctly
    specified likelihood's whitened score across d informed modes scales as
    sqrt(d), so without this a high-d field (tbias) would be rationed
    C_z/sqrt(d) per mode while a scalar got C_z. d_eff is the parameter
    prior's effective dof on the domain, d = max(1, 2*A/(pi*l^2)) (the nu=1
    Matern correlation area A_corr = pi*l^2/2) — computed automatically so a
    larger domain legitimately supports proportionally more structure and
    C_z keeps one meaning across blocks and ranges. Pass an explicit
    (C_z, d) tuple to override (e.g. non-nu=1 priors). NOTE the joint cap
    bounds the TOTAL budget, not concentration: the likelihood may spend
    sqrt(d)*C_z on one mode — inspect the fitted field (t_bias in the VTIs)
    for a single smooth swell vs structure at the prior's l.
    """
    out = {}
    A = ny * nx * dx * dx
    for name, spec in (caps or {}).items():
        if isinstance(spec, (tuple, list)):
            C, d = float(spec[0]), float(spec[1])
        else:
            C = float(spec)
            hp_name = _PRIOR_OF.get(name)
            hp = getattr(config, hp_name) if hp_name else None
            d = max(1.0, 2.0 * A / (math.pi * hp.l ** 2)) if hp is not None \
                else 1.0
        out[name] = (C, d)
    return out


_INFLUENCE_TRANSFERS = {
    # psi(x): score gain at demand x = |g_data|/C~. Near-identity for x << 1;
    # the tail encodes how displacement scales with likelihood demand at
    # equilibrium — tanh: bounded (||z*|| <= C_z); log: C_z per e-fold
    # (||z*|| = C_z*log(1+x), exponential information per sigma, defeasible).
    "tanh": math.tanh,
    "log": math.log1p,
}


def apply_influence_control(params, *, loss_scale: float, eta: float = 1.0,
                            caps: dict = None,
                            transfer: str = "tanh") -> dict:
    """Likelihood-side influence control on whitened parameter blocks —
    gradient surgery after a full backward pass. Two mechanisms, composable:

    * `eta` — semi-modular tempering of the enthalpy SMB block
      (z_log_H_atm, z_logit_cloud): the DATA component of the gradient is
      scaled by eta, the prior component untouched. eta = 1 no-op (full
      Bayes), eta = 0 cut posterior. Fragile in practice: the balancing eta
      scales with the (state-dependent) misspecified pull, so it needs
      per-problem calibration.

    * `caps` — influence-limited (robust, Huber-psi-style) cap, per
      parameter: {z_attr_name: C_z} with C_z in PRIOR-STD units. The block's
      whitened data score is radially transformed,

          g_data  <-  C~ * psi(|g_data| / C~) * g_data/|g_data|,
          C~ = loss_scale * C_z,   psi per `transfer`,

      i.e. adaptive tempering eta(w) = psi(x)/x, x = |g_data|/C~ — near 1
      for plausible demands, engaging only when the likelihood would carry
      the block out of its typical set (the discount read as an implausible
      request of the misspecified model). At stationarity
      g_prior = -C~*psi(x*), so:
        transfer="tanh": ||z*|| <= C_z REGARDLESS of demand (bounded
          influence — magnitude beyond the cap carries nothing; C_z ~ 2-3);
        transfer="log":  ||z*|| = C_z*log(1+x*) — C_z prior-stds PER E-FOLD
          of demand (exponential information per sigma; the flawed-model
          hypothesis is defeasible; C_z is a rate, ~0.2-0.5).
      Both are stated in prior geometry, need no pull-table calibration, and
      transfer across domains unchanged. Direction is preserved (the data
      still says WHERE, just not how far). Cap values may be C_z floats
      (d = 1) or (C_z, d) tuples — pass caps through
      `resolve_influence_caps` so many-dof GP blocks get the sqrt(d_eff)
      budget (C_z stays the per-mode trust level). Lineage:
      bounded/redescending influence functions (Huber), generalized Bayes
      under misspecification (Jewson, Smith & Holmes 2018; beta/gamma-
      divergence posteriors). Pure structural absorbers (log_beta, pbias)
      may stay uncapped; capping a PHYSICALLY MEANINGFUL absorber (tbias —
      to be validated against field observations) turns it into a bounded
      absorber, the eps_max move: honest prior sigma first, cap as the
      misspecification guard.

    In both cases the prior gradient is analytic (loss_scale * z, zero mean
    in the MAP solve) so the data component separates exactly from one
    backward pass. When both are set for a parameter, eta applies first,
    then the cap. The fixed point is that of a non-conservative field (same
    formal status as the warm-started profiled nuisances); the pull-table
    stationarity test applies blockwise.

    Returns {name: x} saturation factors (x = |g_data|/C~) for the capped
    parameters — the audit trail of how implausible the likelihood's current
    request is (x <= 1: essentially full Bayes; x >> 1: pinned near the C_z
    contour).
    """
    if not (0.0 <= eta <= 1.0):
        raise ValueError(
            f"smb_data_influence must be in [0, 1] (1 = full Bayes, "
            f"0 = cut posterior), got {eta!r}")
    caps = caps or {}
    psi = _INFLUENCE_TRANSFERS.get(transfer)
    if psi is None:
        raise ValueError(
            f"influence_transfer must be one of "
            f"{tuple(_INFLUENCE_TRANSFERS)}, got {transfer!r}")
    norm_caps = {}
    for name, spec in caps.items():
        if not hasattr(params, name):
            raise ValueError(
                f"influence_cap key {name!r} is not a whitened parameter "
                f"(expected a WhitenedParameters attribute name like "
                f"'z_log_H_atm')")
        C, d = spec if isinstance(spec, (tuple, list)) else (spec, 1.0)
        if not (C > 0.0) or not (d >= 1.0):
            raise ValueError(
                f"influence_cap[{name!r}] needs C > 0 and d >= 1, "
                f"got C={C!r}, d={d!r}")
        norm_caps[name] = (float(C), float(d))
    caps = norm_caps
    sat = {}
    with torch.no_grad():
        for name in dict.fromkeys(list(_SMB_BLOCK) + list(caps)):
            temper = name in _SMB_BLOCK and eta != 1.0
            C = caps.get(name)
            if not temper and C is None:
                continue    # untouched: keep the gradient bit-identical
            z = getattr(params, name)
            if z.grad is None:
                continue
            g_prior = loss_scale * z.detach()
            g_data = z.grad - g_prior
            if temper:
                g_data = eta * g_data
            if C is not None:
                C_z, d = C
                C_t = loss_scale * C_z * math.sqrt(d)
                n = float(g_data.norm())
                x = n / C_t
                sat[name] = x
                if n > 0.0:
                    g_data = g_data * (C_t * psi(x) / n)
            z.grad.copy_(g_data + g_prior)
    return sat


def marginal_velocity_log_likelihood(
    u_obs,          # (N, 2) observed velocity, flattened raster
    u_mod,          # (N, 2) modeled velocity
    sigma,          # (N,)   per-pixel noise scale
    labels,         # (N,)   long, glacier id per pixel, -1 = unlabeled
    eta_nodes,      # (K,)   quadrature nodes on (0, 1]
    log_w_eff,      # (K, n_glaciers) log(w_k * P(eta_k | glacier)), precomputed once
    nu,             # pseudo-Huber threshold
    lamda,
    scale
):
    """Per-glacier marginal velocity penalty, tempered OUTSIDE the marginal.

    Each labeled glacier carries an unknown factor eta in (0, 1] by which the
    observed mosaic may underestimate the model speed (surge quiescence,
    tracking failure). The likelihood is marginalized over eta in its own
    units — per-pixel pseudo-Huber costs in nats at the stated `sigma` — so
    the pixels of a glacier collectively decide which eta dominates the
    integral, and only the resulting log-marginal is multiplied by the
    tempering weight `scale * lamda`:

        loss = scale*lamda * sum_g [ -log sum_k w_k P(eta_k | g) exp(-c_g(eta_k)) ]

    Tempering the pixel costs *inside* the integral instead (the historical
    form) inflates the effective sigma by 1/sqrt(scale*lamda) (~250x), leaves
    the integrand flat in eta, and collapses the term to the prior average
    E_eta[c(eta)] — a symmetric bowl centred on U_mod = E[eta]/E[eta^2] U_obs,
    which is the opposite of the intended one-sided tolerance.
    """

    K = eta_nodes.shape[0]
    n_glaciers = (labels.max() + 1).item()
    labeled = labels >= 0

    # --- labeled pixels: marginalized likelihood ---
    u_obs_l = u_obs[labeled]                # (M, 2)
    u_mod_l = u_mod[labeled]                # (M, 2)
    sigma_l = sigma[labeled]                # (M,)
    lab     = labels[labeled]               # (M,)

    # residual at each quadrature node: (K, M, 2)
    # eta broadcasts as (K, 1, 1), u_obs_l as (1, M, 2)
    r = u_obs_l.unsqueeze(0) - eta_nodes[:, None, None] * u_mod_l.unsqueeze(0)

    # normalized residual magnitude squared: (K, M)
    r2 = (r / sigma_l[None, :, None]).square().sum(dim=-1)

    # pseudo-Huber negative log-likelihood per pixel per node, in nats
    # (untempered — see the docstring): (K, M)
    phl = (nu ** 2) * (torch.sqrt(1.0 + r2 / nu ** 2) - 1.0)

    # segment sum by glacier label: (K, M) -> (K, n_glaciers)
    per_glacier = torch.zeros(K, n_glaciers, device=u_obs.device, dtype=torch.float32)
    per_glacier.scatter_add_(1, lab.unsqueeze(0).expand(K, -1), phl)

    # Marginalize the per-glacier likelihood over eta: `per_glacier` is a
    # positive penalty (negative log-likelihood), so the marginal penalty is
    #   -log sum_k w_k P(eta_k) exp(-cost_k) = -logsumexp(-cost + log_w_eff).
    # (A `+logsumexp(+cost + log_w_eff)` coincides with this only while
    # per-glacier costs are << 1 nat; at larger costs it tends to the WORST
    # eta instead of the best.) The tempering weight is applied afterwards.
    marginal = -torch.logsumexp(-per_glacier + log_w_eff, dim=0)  # (n_glaciers,)
    ll_labeled = scale * lamda * marginal.sum()

    # --- unlabeled pixels: standard likelihood at eta = 1 ---

    if (~labeled).any():
        r_ul = u_obs[~labeled] - u_mod[~labeled]
        r2_ul = (r_ul / sigma[~labeled, None]).square().sum(dim=-1)
        ll_unlabeled = scale * lamda * (nu ** 2) * (torch.sqrt(1.0 + r2_ul / nu ** 2) - 1.0)
        ll_labeled = ll_labeled + ll_unlabeled.sum()

    return ll_labeled


def compute_prior(
    *,
    config,
    priors,
    params,
    physical_bed: torch.Tensor,
    physical_bed_mean: torch.Tensor,
    log_rf: torch.Tensor,
    log_mf: torch.Tensor,
    prior_means: PriorMeans,
    physical_bed_uncond: Optional[torch.Tensor] = None,
    physical_log_beta: Optional[torch.Tensor] = None,
) -> tuple:
    """Whitened-space Gaussian prior terms: exact negative log-densities
    `loss_scale · ½‖z − mean‖²`, on the same footing as the data terms
    (`_huber(r) ≈ r²/2`). (Before 2026-08 the ½ was missing, i.e. every prior
    was twice as stiff as its stated hyperparameters.)

    physical_bed is the bed produced by the prior map; we recompute its
    whitened representation here (matching the original code) rather than
    threading it through, because GGaPPWhiten/GGaPPMap is not exactly
    self-inverse and the original used this form.
    """
    scale = config.loss_scale

    # The bed_mean hierarchy: re-whiten the UNCONDITIONAL bed fluctuation
    # about the mean field. Under bed conditioning, `physical_bed_uncond`
    # (= Map(z_bed), before the kriging correction) must be used here — the
    # correction is data-driven and must not be penalized by the prior, and
    # the mean stays out of the conditioning map so its curvature (and its
    # tuned learning rate) is identical in both parametrizations.
    base_bed = physical_bed if physical_bed_uncond is None else physical_bed_uncond
    z_bed_recomputed = GGaPPWhiten.apply(priors.bed_model, base_bed - physical_bed_mean)
    J_prior_bed = scale * 0.5 * ((z_bed_recomputed - prior_means.value("z_bed", z_bed_recomputed)) ** 2).sum()
    J_prior_bed_mean = scale * 0.5 * ((params.z_bed_mean - prior_means.value("z_bed_mean", params.z_bed_mean)) ** 2).sum()
    if (getattr(priors, "log_beta_mean_model", None) is not None and not priors.log_beta_mean_additive
            and physical_log_beta is not None):
        # Centered two-field log beta (the bed_mean pattern): the fluctuation
        # about the long-wavelength mean, re-whitened from the physical field.
        m = priors.log_beta_mean_from_whitened(params.z_log_beta_mean)
        z_lb = GGaPPWhiten.apply(priors.log_beta_model, physical_log_beta - priors.mu_log_beta - m)
    else:
        z_lb = params.z_log_beta
    J_prior_beta = scale * 0.5 * ((z_lb - prior_means.value("z_log_beta", z_lb)) ** 2).sum()
    # Long-wavelength log-beta field: exactly zero while it sits at 0 (off).
    zbm = getattr(params, "z_log_beta_mean", None)
    J_prior_beta_mean = (scale * 0.5 * ((zbm - prior_means.value("z_log_beta_mean", zbm)) ** 2).sum()
                         if zbm is not None else torch.zeros((), device=params.z_log_beta.device))
    J_prior_pbias = scale * 0.5 * ((params.z_pbias - prior_means.value("z_pbias", params.z_pbias)) ** 2).sum()
    # Temperature bias: whitened-space term, so it needs no Matern model and
    # is computed unconditionally — exactly zero while the term is disabled
    # (z_tbias stays at 0 and out of the optimizer).
    J_prior_tbias = scale * 0.5 * ((params.z_tbias - prior_means.value("z_tbias", params.z_tbias)) ** 2).sum()
    # Enthalpy SMB parameter fields (log H_atm, logit clear-sky fraction):
    # whitened-space terms like tbias — no Matern model needed here, exactly
    # zero while the fields sit at 0 (ETIM backend, or the prior median).
    J_prior_h_atm = scale * 0.5 * ((params.z_log_H_atm - prior_means.value("z_log_H_atm", params.z_log_H_atm)) ** 2).sum()
    J_prior_cloud = scale * 0.5 * ((params.z_logit_cloud - prior_means.value("z_logit_cloud", params.z_logit_cloud)) ** 2).sum()
    # Standard-normal whitened priors on the remaining scalars (mf/rf and the
    # precip-depletion tau/z0 — each inert at z = 0 when its model/term is
    # disabled). The .sum() is a guard: every term here must stay 0-d.
    J_prior_smb = scale * 0.5 * ((params.z_log_rf - prior_means.value("z_log_rf", params.z_log_rf)) ** 2
                   + (params.z_log_mf - prior_means.value("z_log_mf", params.z_log_mf)) ** 2
                   + (params.z_tau - prior_means.value("z_tau", params.z_tau)) ** 2
                   + (params.z_z0 - prior_means.value("z_z0", params.z_z0)) ** 2).sum()

    return (J_prior_bed, J_prior_bed_mean, J_prior_beta, J_prior_pbias,
            J_prior_tbias, J_prior_h_atm, J_prior_cloud, J_prior_smb,
            J_prior_beta_mean)
