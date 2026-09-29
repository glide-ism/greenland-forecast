"""
Posterior sensitivity (information gain) for volume-change predictions.

Loads RTO samples, maps each to physical space, runs the forward model from
t_start through t_end_future, and uses the gradient of ΔV w.r.t. each
physical parameter to compute a Stein-style information-gain metric.
"""
from pathlib import Path

import numpy as np
import torch

import cupy as cp
from ggapp.torch import GGaPPMap
from glide.field import Field, GridEntity

from glacier_inverse import GlacierProblem, load_config
from glacier_inverse.problem import PhysicalParameters
from glacier_inverse.io import make_time_vti_writer

# Available domains: domains/{greenland,greenland_coarse}
DOMAIN = "domains/greenland"
config = load_config(DOMAIN)
if config.tbias_enabled:
    raise NotImplementedError(
        "sensitivity.py hardcodes the five-block flattened parameter layout "
        "(bed, pbias, log_beta, log_mf, log_rf) and would silently project "
        "without the temperature bias — extend collect_rto_samples and the "
        "downstream index arithmetic before running a tbias-enabled domain.")
if config.smb_model == "enthalpy":
    raise NotImplementedError(
        "sensitivity.py's five-block layout has no blocks for the enthalpy "
        "parameter fields (log_H_atm, logit_cloud are (ny,nx) GP fields) and "
        "its PhysicalParameters would hand simulate_physical a None "
        "log_H_atm — extend the layout to seven blocks (5·ny·nx) before "
        "running an enthalpy domain.")
INPUT_PATH = config.output_dir
OUTPUT_PATH = f"{INPUT_PATH}/sens/"
LEVEL = 2

# Project 100 years past the end of the calibration window.
PROJECTION_YEARS = 100
T_START_FUTURE = config.t_end - 1000.0     # 1112 (1000 yrs of spin-up)
T_OBS = config.t_end                       # 2012 (end of calibration)
T_END_FUTURE = config.t_end + PROJECTION_YEARS  # 2112

problem = GlacierProblem(config)
priors = problem.priors
mg = problem.mg
ny, nx, dx = problem.ny, problem.nx, problem.dx


def collect_rto_samples(rto_dir: Path):
    beds, pbiases, log_betas, log_mfs, log_rfs = [], [], [], [], []
    for d in rto_dir.iterdir():
        try:
            data = torch.load(f"{d}/level_2/torch_vars.p")
        except FileNotFoundError:
            continue
        saved = data.get("bed_parametrization", "legacy")
        if saved != priors.bed_parametrization:
            raise ValueError(
                f"{d}: sample saved with bed_parametrization={saved!r} but "
                f"priors are {priors.bed_parametrization!r} — z_bed would be "
                f"misinterpreted. Match config.bed_conditioning to the runs.")
        # Shared whitened->bed map: correct in both bed parametrizations
        # (conditioned checkpoints carry the kriging correction in the
        # fluctuation field; bed_mean stays prior-coupled as in legacy).
        beds.append(priors.bed_from_whitened(
            data["bed"], data["bed_mean"])[0].cpu().detach())
        pbiases.append(GGaPPMap.apply(priors.pbias_model, data["precipitation_bias"]).cpu().detach())
        log_betas.append(priors.log_beta_from_whitened(data["log_beta"], data.get("log_beta_mean")).cpu().detach())
        log_mfs.append((data["log_mf"] * priors.sigma_log_mf + priors.mu_log_mf).cpu().detach())
        log_rfs.append((data["log_rf"] * priors.sigma_log_rf + priors.mu_log_rf).cpu().detach())

    bed_flat      = torch.stack([b.ravel() for b in beds],      axis=-1)
    pbias_flat    = torch.stack([p.ravel() for p in pbiases],   axis=-1)
    log_beta_flat = torch.stack([p.ravel() for p in log_betas], axis=-1)
    log_mf_flat   = torch.stack(log_mfs).reshape(1, -1)
    log_rf_flat   = torch.stack(log_rfs).reshape(1, -1)

    p_flat = torch.vstack((bed_flat, pbias_flat, log_beta_flat, log_mf_flat, log_rf_flat))
    return p_flat


p_flat = collect_rto_samples(Path(f"{INPUT_PATH}/rto_prior_bias/"))
n_samples = p_flat.shape[1]

# Physical-space leaf tensors — we get ∂ΔV/∂(physical param) from these.
bed                 = torch.zeros(ny, nx, device="cuda", requires_grad=True)
bed_mean            = torch.zeros(ny, nx, device="cuda", requires_grad=False)
log_beta            = torch.zeros(ny, nx, device="cuda", requires_grad=True)
precipitation_bias  = torch.zeros(ny, nx, device="cuda", requires_grad=True)
log_mf              = torch.zeros((), device="cuda", requires_grad=True)
log_rf              = torch.zeros((), device="cuda", requires_grad=True)

problem.model.set_top_level(LEVEL)
level_dir = f"{OUTPUT_PATH}/level_{LEVEL}/vti"

deltas = []
qs = []

for s in range(n_samples):
    print(s)

    bed.grad = None
    log_beta.grad = None
    precipitation_bias.grad = None
    log_mf.grad = None
    log_rf.grad = None

    bed.data[:, :]                = p_flat[              :ny * nx,     s].reshape(ny, nx)
    precipitation_bias.data[:, :] = p_flat[ny * nx       :2 * ny * nx, s].reshape(ny, nx)
    log_beta.data[:, :]           = p_flat[2 * ny * nx   :3 * ny * nx, s].reshape(ny, nx)
    log_mf.data.fill_(p_flat[3 * ny * nx,     s].item())
    log_rf.data.fill_(p_flat[3 * ny * nx + 1, s].item())

    physical = PhysicalParameters(
        bed=bed, bed_mean=bed_mean, log_beta=log_beta,
        pbias=precipitation_bias, log_mf=log_mf, log_rf=log_rf,
    )

    time_writer = make_time_vti_writer(problem.mg[LEVEL], level_dir)
    sim = problem.simulate_physical(
        level=LEVEL,
        physical=physical,
        t_start=T_START_FUTURE,
        t_end=T_END_FUTURE,
        record_volumes_at=[T_OBS, T_END_FUTURE],
        time_writer=time_writer,
    )

    V_obs    = sim.volumes[T_OBS]
    V_future = sim.volumes[T_END_FUTURE]
    Delta_V = V_future - V_obs
    Delta_V.backward()

    deltas.append(Delta_V.detach())
    q = torch.hstack((
        bed.grad.cpu().ravel(),
        precipitation_bias.grad.cpu().ravel(),
        log_beta.grad.cpu().ravel(),
        log_mf.grad.cpu().ravel(),
        log_rf.grad.cpu().ravel(),
    ))
    qs.append(q)

# ---------------------------------------------------------------- diagnostics

q = torch.stack(qs, axis=0).mean(axis=0)

L = (p_flat - p_flat.mean(axis=1, keepdims=True)) / np.sqrt(n_samples - 1)
Sigma_diag = (L ** 2).sum(axis=1)
Q_var = (q.T @ L) @ (L.T @ q)

num = (L @ (L.T @ q)) ** 2
num_bed   = num[:nx * ny].reshape(ny, nx)
num_pbias = num[nx * ny:2 * nx * ny].reshape(ny, nx)

denom_bed   = Sigma_diag[:ny * nx].reshape(ny, nx)         + problem.get_observation("srf").sigma ** 2
denom_pbias = Sigma_diag[ny * nx:2 * ny * nx].reshape(ny, nx) + 0.01 ** 2

rho2_bed   = num_bed   / (denom_bed   * Q_var)
rho2_pbias = num_pbias / (denom_pbias * Q_var)
D_kl_bed_stein   = -0.5 * torch.log(1 - rho2_bed)
D_kl_pbias_stein = -0.5 * torch.log(1 - rho2_pbias)

Q = torch.stack(deltas).cpu()
Q_ = Q - Q.mean()
X_ = p_flat - p_flat.mean(axis=1, keepdims=True)
c = (X_ @ Q_) / (n_samples - 1)
v = (X_ * X_).sum(axis=1) / (n_samples - 1)

c_bed   = c[:nx * ny].reshape(ny, nx)
v_bed   = v[:nx * ny].reshape(ny, nx)
c_pbias = c[nx * ny:2 * nx * ny].reshape(ny, nx)
v_pbias = v[nx * ny:2 * nx * ny].reshape(ny, nx)

delta_V_bed   = c_bed   ** 2 / (v_bed   + 10 ** 2)
delta_V_pbias = c_pbias ** 2 / (v_pbias + 0.1 ** 2)

rho2_bed_sample   = delta_V_bed   / ((Q_ ** 2).sum() / (n_samples - 1))
rho2_pbias_sample = delta_V_pbias / ((Q_ ** 2).sum() / (n_samples - 1))
D_kl_bed_sample   = -0.5 * torch.log(1 - rho2_bed_sample)
D_kl_pbias_sample = -0.5 * torch.log(1 - rho2_pbias_sample)


def _to_field(level0_mg, values_2d):
    f = Field(cp.zeros((level0_mg.ny, level0_mg.nx), dtype=cp.float32),
              grid_entity=GridEntity.CELL,
              dx=level0_mg.dx,
              grid=level0_mg)
    f.set(values_2d)
    return f


thk_std = _to_field(mg[0], (L ** 2).sum(axis=1)[:ny * nx].reshape(ny, nx) ** 0.5)
bed_sens = _to_field(mg[0], D_kl_bed_stein).to_dataarray()
precip_sens = _to_field(mg[0], D_kl_pbias_stein).to_dataarray()
