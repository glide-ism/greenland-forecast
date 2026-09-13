"""
Randomize-then-optimize (RTO) posterior sampler.

Builds the GlacierProblem once, then iterates internally to draw N_SAMPLES
posterior samples — each with freshly randomized observations, fresh prior
means in whitened coordinates, and a fresh Bernoulli extent mask. Replaces
the previous shell-driven loop (run_rto.sh).
"""
import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
from scipy.stats import norm, qmc
from torchvision.transforms.functional import gaussian_blur

from glacier_inverse import GlacierProblem, PriorMeans, load_config
from glacier_inverse.forward import differentiable_restriction
from glacier_inverse.io import (
    load_whitened_params_into, make_diagnostic_fields, make_loss_vti_writer,
    make_time_vti_writer, save_whitened_params, update_diagnostic_fields,
)

# Available domains: domains/{greenland,greenland_coarse}
DOMAIN = "domains/greenland"
config = load_config(DOMAIN)


INPUT_PATH = config.output_dir
WARM_START_PATH = f"{INPUT_PATH}/level_3/torch_vars.p"

N_SAMPLES = 100
LEVEL = 3
ITERS_PER_SAMPLE = 100

# Cosine learning-rate decay applied per-sample. Each parameter group's LR
# anneals from its base value down to LR_FINAL_RATIO * base_lr over
# ITERS_PER_SAMPLE iterations, which damps end-of-run oscillation around the
# perturbed MAP. Set LR_FINAL_RATIO = 1.0 to disable; lower it (e.g. 0.001)
# to settle harder.
LR_FINAL_RATIO = 1.0


def cosine_ratio(step: int) -> float:
    """Cosine schedule in ratio space: returns 1.0 at step 0 and
    LR_FINAL_RATIO at step ITERS_PER_SAMPLE."""
    return LR_FINAL_RATIO + 0.5 * (1.0 - LR_FINAL_RATIO) * (
        1.0 + math.cos(math.pi * step / ITERS_PER_SAMPLE)
    )


class NoiseScheme(str, Enum):
    """Randomization strategy for the per-sample noise vector."""
    MC    = "mc"     # i.i.d. torch.randn / torch.rand (status quo)
    SOBOL = "sobol"  # scrambled Sobol points → Gaussian via inverse CDF


# scipy.stats.qmc.Sobol caps out at d=21201. The wrangell noise budget is
# ~20M dims (mostly fields), so we can't QMC everything. The driver below
# *consumes* dims in priority order — scalars first, then flightline bed
# noise, then fields — so the budget goes to the lowest-D / highest-value
# draws. After the budget is exhausted everything falls back to MC.
NOISE_SCHEME = NoiseScheme.MC
SOBOL_DIM    = 21201
SOBOL_SEED   = None  # set to an int for reproducible Sobol scrambling


class NoiseSource:
    """Per-sample Gaussian / uniform noise generator.

    Call `new_sample()` at the start of each RTO sample; subsequent
    `randn_like` / `rand_like` calls consume the same scrambled Sobol point.
    Once the Sobol-d budget is exhausted within a sample, draws fall back
    to standard `torch.randn_like` / `torch.rand_like`.
    """

    def __init__(self, scheme: NoiseScheme, sobol_dim: int = SOBOL_DIM,
                 seed: int = None):
        self.scheme = scheme
        if scheme == NoiseScheme.SOBOL:
            self.sobol_d = min(int(sobol_dim), 21201)
            self.engine = qmc.Sobol(d=self.sobol_d, scramble=True, seed=seed)
        else:
            self.sobol_d = 0
            self.engine = None
        self._u_buffer: np.ndarray = None
        self._cursor = 0

    def new_sample(self) -> None:
        if self.engine is not None:
            self._u_buffer = self.engine.random(n=1)[0]   # shape (sobol_d,)
            self._cursor = 0

    def randn_like(self, t: torch.Tensor) -> torch.Tensor:
        n = t.numel()
        if self.engine is not None and self._cursor + n <= self.sobol_d:
            u = np.clip(self._u_buffer[self._cursor:self._cursor + n], 1e-12, 1 - 1e-12)
            z = norm.ppf(u).astype(np.float32)
            self._cursor += n
            return torch.as_tensor(z, device=t.device).reshape(t.shape)
        return torch.randn_like(t)

    def rand_like(self, t: torch.Tensor) -> torch.Tensor:
        n = t.numel()
        if self.engine is not None and self._cursor + n <= self.sobol_d:
            u = self._u_buffer[self._cursor:self._cursor + n].astype(np.float32)
            self._cursor += n
            return torch.as_tensor(u, device=t.device).reshape(t.shape)
        return torch.rand_like(t)


class Init(str, Enum):
    """Per-parameter starting point for each RTO optimization."""
    MAP             = "map"               # pin to the MAP estimate
    MAP_PLUS_SAMPLE = "map_plus_sample"   # MAP + this sample's prior-mean draw
    ZERO            = "zero"              # start at zero in whitened coords


@dataclass
class InitScheme:
    """Each parameter's choice encodes a different linearization / basin
    assumption for the posterior:

      * MAP keeps a parameter pinned at the deterministic MAP and assumes
        the posterior is well-described by a local Gaussian centered there.
      * MAP_PLUS_SAMPLE perturbs by the prior-mean draw and explores nearby
        modes (the standard RTO assumption).
      * ZERO starts from scratch and is more likely to find structurally
        different modes if the posterior is multi-modal.
    """
    z_bed:      Init = Init.MAP
    z_bed_mean: Init = Init.MAP_PLUS_SAMPLE
    z_log_beta: Init = Init.MAP#_PLUS_SAMPLE
    z_pbias:    Init = Init.MAP_PLUS_SAMPLE
    z_log_mf:   Init = Init.MAP_PLUS_SAMPLE
    z_log_rf:   Init = Init.MAP_PLUS_SAMPLE


# Edit this to change per-parameter initialization for an RTO run.
INIT_SCHEME = InitScheme()


@torch.no_grad()
def initialize_param(name, params, warm_start, prior_means, scheme):
    p = getattr(params, name)
    if scheme == Init.MAP:
        p.copy_(getattr(warm_start, name))
    elif scheme == Init.MAP_PLUS_SAMPLE:
        p.copy_(getattr(warm_start, name) + getattr(prior_means, name))
    elif scheme == Init.ZERO:
        p.zero_()
    else:
        raise ValueError(f"unknown init scheme {scheme!r}")

problem = GlacierProblem(config)
params = problem.params

if WARM_START_PATH is not None:
    load_whitened_params_into(params, WARM_START_PATH, priors=problem.priors)

# Snapshot the warm-start state. Each sample resets params in-place to this
# before perturbing — keeps tensor identities stable so per-sample optimizers
# can be rebuilt without dangling references.
warm_start = params.detach_clone()

problem.model.set_top_level(LEVEL)

noise_source = NoiseSource(NOISE_SCHEME, sobol_dim=SOBOL_DIM, seed=SOBOL_SEED)


def write_loss_vti(diag, vti_writer, sim, physical, observations, i):
    bed_mean_coarse = differentiable_restriction(physical.bed_mean, LEVEL)
    pbias_coarse = differentiable_restriction(physical.pbias, LEVEL)
    pbias_total_coarse = differentiable_restriction(
        problem.effective_log_pbias(physical), LEVEL)
    S_obs_coarse = differentiable_restriction(observations.S_obs, LEVEL)
    dhdt_coarse = (sim.H - sim.H_prev) / config.dt
    update_diagnostic_fields(diag, sim.S_coarse, S_obs_coarse, bed_mean_coarse,
                             pbias_coarse, pbias_total_coarse, dhdt_coarse)
    vti_writer.append(problem.mg[LEVEL], time=i)
    vti_writer.write_pvd()


print("Initialization scheme:")
for f in INIT_SCHEME.__dataclass_fields__:
    print(f"  {f:<10s} = {getattr(INIT_SCHEME, f).value}")
print(f"Noise scheme:  {NOISE_SCHEME.value}"
      + (f"  (Sobol d={noise_source.sobol_d})" if noise_source.engine else ""))

for sample_idx in range(N_SAMPLES):
    output_path = f"{INPUT_PATH}/rto_lr_decay/rto_{sample_idx:04d}/"
    print(f"\n=== Sample {sample_idx + 1}/{N_SAMPLES}  →  rto_{sample_idx} ===")

    # Pull one Sobol point per sample (no-op for MC) and draw noise in
    # priority order so the limited QMC budget hits the lowest-D draws
    # first (scalars → flightline bed → fields → mask).
    noise_source.new_sample()

    eps_z_log_mf   = noise_source.randn_like(params.z_log_mf)
    eps_z_log_rf   = noise_source.randn_like(params.z_log_rf)
    eps_bed_obs    = noise_source.randn_like(problem.observations.bed_obs)
    eps_z_bed      = noise_source.randn_like(params.z_bed)
    eps_z_bed_mean = noise_source.randn_like(params.z_bed_mean)
    eps_z_log_beta = noise_source.randn_like(params.z_log_beta)
    eps_z_pbias    = noise_source.randn_like(params.z_pbias)
    # NOTE: when this script is migrated to the time-stamped-observation API,
    # a tbias-enabled domain also needs an eps_z_tbias draw — appended AFTER
    # the existing draw order (like eps_bed_cond below) so QMC dimension
    # assignment stays stable — plus PriorMeans(z_tbias=...), the init loop,
    # and an SGD group gated on config.tbias_enabled (there is no Adam block
    # anymore — every parameter is SGD in whitened coordinates). Likewise the
    # enthalpy pair z_log_H_atm/z_logit_cloud are now (ny, nx) whitened GP
    # FIELDS (draws are randn_like the field), not scalars. The migration
    # must also apply config.smb_data_influence (semi-modular eta) to that
    # block's data gradient per step — loss.apply_influence_control (eta and influence_cap), as in
    # inverse.py — so samples target the same semi-modular posterior the
    # MAP converged to.
    eps_u_obs      = noise_source.randn_like(problem.observations.u_obs)
    eps_v_obs      = noise_source.randn_like(problem.observations.v_obs)
    eps_S_obs      = noise_source.randn_like(problem.observations.S_obs)
    mask_uniform   = noise_source.rand_like(
        problem.observations.rgi_mask.to(torch.float32))

    # Bed-conditioning data perturbation (Matheron's rule). Appended AFTER the
    # pre-existing draw order so QMC dimension assignment stays stable. The
    # precision-weighted per-cell datum has variance 1/D_ii, and the perturbed
    # data must reach the conditioning MAP (not a loss term) — hence the
    # problem-level override rather than obs.randomized().
    eps_bed_cond = None
    if problem.priors.bed_conditioner is not None:
        cond = problem.priors.bed_conditioner
        b_data = torch.tensor(cond.data)
        D_prec = torch.tensor(cond.precision)
        eps_bed_cond = noise_source.randn_like(b_data)
        observed = D_prec > 0
        sigma_cell = torch.where(observed, D_prec,
                                 torch.ones_like(D_prec)).rsqrt()
        problem.set_bed_data_override(
            b_data + observed.to(b_data.dtype) * sigma_cell * eps_bed_cond)

    observations = problem.observations.randomized(
        sigma_u=config.sigma_u, sigma_s=config.sigma_s, sigma_bed=config.sigma_s,
        eps_u=eps_u_obs, eps_v=eps_v_obs, eps_S=eps_S_obs, eps_bed=eps_bed_obs,
    )
    prior_means = PriorMeans(
        z_bed=eps_z_bed, z_bed_mean=eps_z_bed_mean,
        z_log_beta=eps_z_log_beta, z_pbias=eps_z_pbias,
        z_log_mf=eps_z_log_mf, z_log_rf=eps_z_log_rf,
    )

    for name in ("z_bed", "z_bed_mean", "z_log_beta",
                 "z_pbias", "z_log_mf", "z_log_rf"):
        initialize_param(name, params, warm_start, prior_means,
                         getattr(INIT_SCHEME, name))

    fac = 1.0
    # Steady-state learning rates: any Schedule on an lr_z_* config field is
    # an inverse-only continuation device and collapses to its `final` here
    # (schedule=False), mirroring how the loss weights are treated below.
    lr = config.learning_rates(schedule=False)
    optimizer_sgd = torch.optim.SGD([
        {"params": params.z_bed,      "lr": fac * lr["lr_z_bed"]},
        {"params": params.z_bed_mean, "lr": fac * lr["lr_z_bed_mean"]},
        {"params": params.z_log_beta, "lr": fac * lr["lr_z_log_beta"]},
    ], momentum=0.5)

    optimizer_adam = torch.optim.Adam([
        {"params": params.z_pbias,  "lr": fac * lr["lr_z_pbias"]},
        {"params": params.z_log_mf, "lr": fac * lr["lr_z_log_mf"]},
        {"params": params.z_log_rf, "lr": fac * lr["lr_z_log_rf"]},
    ], betas=(0.5, 0.99))

    # LambdaLR multiplies each param group's base LR by cosine_ratio(step),
    # so groups with very different base LRs all decay to the same fraction
    # of their own initial value.
    scheduler_sgd  = torch.optim.lr_scheduler.LambdaLR(optimizer_sgd,  cosine_ratio)
    scheduler_adam = torch.optim.lr_scheduler.LambdaLR(optimizer_adam, cosine_ratio)

    rgi = observations.rgi_mask.to(torch.float32).unsqueeze(0)
    p_mask = gaussian_blur(rgi, 27).squeeze()
    mask = (mask_uniform < p_mask).to(torch.float32)

    diag = make_diagnostic_fields(problem.mg[LEVEL])
    level_dir = f"{output_path}/level_{LEVEL}/vti"
    vti_writer = make_loss_vti_writer(problem.mg[LEVEL], level_dir,
                                       config.vti_base_name, diag)

    for i in range(ITERS_PER_SAMPLE):
        optimizer_sgd.zero_grad()
        optimizer_adam.zero_grad()

        time_writer = (make_time_vti_writer(problem.mg[LEVEL], level_dir)
                       if i % 20 == 0 else None)

        sim, physical = problem.simulate(
            level=LEVEL, params=params, time_writer=time_writer)
        # No scheduling here: RTO warm-starts from the MAP, so every sample uses
        # the steady-state loss weights (Schedule.final) the inverse solve
        # converged to. compute_loss defaults to schedule=False.
        loss_terms = problem.compute_loss(
            sim=sim, physical=physical, params=params,
            observations=observations, prior_means=prior_means, mask=mask,
        )
        loss_terms.log(i)

        write_loss_vti(diag, vti_writer, sim, physical, observations, i)
        loss_terms.J.backward()

        optimizer_adam.step()
        optimizer_sgd.step()
        scheduler_sgd.step()
        scheduler_adam.step()

    # Save the noise vectors so post-hoc analysis can reproduce / diagnose
    # the realized perturbation per sample.
    noise_extras = {
        "noise_scheme":      NOISE_SCHEME.value,
        "eps_u_obs":         eps_u_obs,
        "eps_v_obs":         eps_v_obs,
        "eps_S_obs":         eps_S_obs,
        "eps_bed_obs":       eps_bed_obs,
        "prior_mean_z_bed":      eps_z_bed,
        "prior_mean_z_bed_mean": eps_z_bed_mean,
        "prior_mean_z_log_beta": eps_z_log_beta,
        "prior_mean_z_pbias":    eps_z_pbias,
        "prior_mean_z_log_mf":   eps_z_log_mf,
        "prior_mean_z_log_rf":   eps_z_log_rf,
        "mask_uniform":      mask_uniform,
        "eps_bed_cond":      eps_bed_cond,
    }
    save_whitened_params(
        params, f"{output_path}/level_{LEVEL}/torch_vars.p",
        extras=noise_extras,
        bed_parametrization=problem.priors.bed_parametrization,
    )
    problem.set_bed_data_override(None)
