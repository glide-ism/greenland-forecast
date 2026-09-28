"""
Deterministic MAP solve in whitened coordinates.

Thin driver over glacier_inverse. All physical hyperparameters live in
domains/<name>/config.py — per-task knobs (output path, max iters,
learning rates) stay here.
"""
import cupy as cp
import torch
import xarray as xr

from glacier_inverse import GlacierProblem, load_config
from glacier_inverse.config import resolve_weight
from glacier_inverse.forward import differentiable_restriction
from glacier_inverse.loss import apply_influence_control, resolve_influence_caps
from glacier_inverse.io import (
    load_whitened_params_into, make_diagnostic_fields, make_loss_vti_writer,
    make_time_vti_writer, save_whitened_params, update_diagnostic_fields,
)

# Available domains: domains/{greenland,greenland_coarse}
DOMAIN = "domains/greenland"
config = load_config(DOMAIN)

OUTPUT_PATH = config.output_dir
# 2026-09-21: start the tightened-likelihood runs (inverse_v6_*) from the v5
# state. From the prior state the per-pixel velocity term is dominated by
# the 435 k slow-ice cells the prior has 1.8x too fast, and its "more
# friction" update leaks onto the outlets through the beta prior (fast ice
# 0.35 -> 0.22 of observed in one step); from v5 the descent is monotone.
#WARM_START_PATH = None
WARM_START_PATH = f"{DOMAIN}/inverse_v11/level_1/torch_vars.p"  # None = from the prior
#WARM_START_PATH = f"{DOMAIN}/inverse_v9/level_2/torch_vars.p"  # None = from the prior
#WARM_START_PATH = f"{DOMAIN}/inverse/level_1/torch_vars.p"

problem = GlacierProblem(config)
params = problem.params

if WARM_START_PATH is not None:
    load_whitened_params_into(params, WARM_START_PATH, priors=problem.priors)

# The observational products are dumped per level (box-restricted to each
# grid — the targets the coarse-space terms see) at the top of the level loop.

# Each param group is named after its config lr field so the per-iteration
# refresh below can apply scheduled learning rates (Schedule / callable on any
# `lr_z_*` in the domain config). The initial values are the iteration-0,
# coarsest-level ones purely for construction; refresh_learning_rates()
# overwrites them before every step.
lr0 = config.learning_rates(0, config.max_level, schedule=True)

# Every parameter is SGD in whitened coordinates (prior-natural gradient:
# an SGD step in z is a C-preconditioned physical step, so updates carry the
# prior's spectrum and likelihood-null directions relax to the prior mean).
# There is no Adam block — Adam's per-coordinate RMS normalization undoes
# exactly the geometry the whitening encodes.
sgd_groups = []

sgd_groups += [{"params": params.z_bed,      "lr": lr0["lr_z_bed"],      "name": "lr_z_bed"},
    {"params": params.z_bed_mean, "lr": lr0["lr_z_bed_mean"], "name": "lr_z_bed_mean"},
    {"params": params.z_log_beta, "lr": lr0["lr_z_log_beta"], "name": "lr_z_log_beta"},
]

sgd_groups += [{"params": params.z_pbias,  "lr": lr0["lr_z_pbias"], "name": "lr_z_pbias"},]

if config.tbias_enabled:
    # Additive temperature bias field; inert (z = 0) when disabled.
    sgd_groups += [
        {"params": params.z_tbias, "lr": lr0["lr_z_tbias"], "name": "lr_z_tbias"},
    ]
if config.smb_model == "enthalpy":
    # Enthalpy SMB parameter fields (log H_atm, logit clear-sky fraction)
    # replace the temperature-index mf/rf pair — the inactive model's z never
    # enter the forward graph and get no gradient.
    sgd_groups += [
        {"params": params.z_log_H_atm,   "lr": lr0["lr_z_log_H_atm"],   "name": "lr_z_log_H_atm"},
        {"params": params.z_logit_cloud, "lr": lr0["lr_z_logit_cloud"], "name": "lr_z_logit_cloud"},
    ]
else:
    sgd_groups += [
        {"params": params.z_log_mf, "lr": lr0["lr_z_log_mf"], "name": "lr_z_log_mf"},
        {"params": params.z_log_rf, "lr": lr0["lr_z_log_rf"], "name": "lr_z_log_rf"},
    ]
if config.precip_lapse_enabled:
    # Elevation-dependent precip depletion scalars.
    sgd_groups += [
        {"params": params.z_tau, "lr": lr0["lr_z_tau"], "name": "lr_z_tau"},
        {"params": params.z_z0,  "lr": lr0["lr_z_z0"],  "name": "lr_z_z0"},
    ]

optimizer_sgd = torch.optim.SGD(sgd_groups, momentum=0.5)

# Influence caps resolved once: C_z is the per-mode trust level; the joint
# cap is sqrt(d_eff)*C_z with d_eff from each parameter prior's correlation
# area (see loss.resolve_influence_caps).
influence_caps = (resolve_influence_caps(config.influence_cap, config,
                                         problem.ny, problem.nx, problem.dx)
                  if config.influence_cap else None)
if influence_caps:
    print("influence caps (C_z per mode, d_eff):",
          {k[2:]: (v[0], round(v[1], 1)) for k, v in influence_caps.items()})

def refresh_learning_rates(i, level):
    """Resolve every scheduled lr at (i, level) and push it into the matching
    optimizer param group. An lr of 0.0 freezes that parameter for the step
    (optimizer state — SGD momentum — still accumulates, so the step is
    well-conditioned when the schedule switches the lr on). Returns the
    resolved dict so callers can log it."""
    lrs = config.learning_rates(i, level, schedule=True)
    for group in optimizer_sgd.param_groups:
        group["lr"] = lrs[group["name"]]
    return lrs


def write_loss_vti(diag, vti_writer, sim, physical, level, i):
    bed_mean_coarse = differentiable_restriction(physical.bed_mean, level)
    pbias_coarse = differentiable_restriction(physical.pbias, level)
    pbias_total_coarse = differentiable_restriction(
        problem.effective_log_pbias(physical), level)
    tbias_coarse = (differentiable_restriction(physical.tbias, level)
                    if physical.tbias is not None else None)
    # Fitted enthalpy parameter fields in physical units (H_atm in W m-2 K-1,
    # f_clear in (0,1)) — inspect these for compensation of bed/dynamics error
    # masquerading as climatology.
    if config.smb_model == "enthalpy":
        H_atm_coarse = differentiable_restriction(
            torch.exp(physical.log_H_atm), level)
        f_clear_coarse = differentiable_restriction(
            torch.sigmoid(physical.logit_cloud), level)
    else:
        H_atm_coarse = f_clear_coarse = None
    S_obs_coarse = differentiable_restriction(
        torch.clamp(problem.domain.dem, min=0.0), level)
    # The model rate over the same interval the dhdt misfit uses (the
    # observation window in two-snapshot mode, the true final step in legacy
    # mode), so the VTI field is directly comparable to the observed raster.
    # Without a dhdt product, fall back to the final-step rate as a
    # near-steady-state diagnostic.
    dhdt_obs = problem.get_observation("dhdt")
    if dhdt_obs is not None:
        dhdt_coarse = dhdt_obs.model_rate(sim, "coarse")
    else:
        dhdt_coarse = (sim.H - sim.H_prev) / sim.final.dt_step
    update_diagnostic_fields(diag, sim.S_coarse, S_obs_coarse, bed_mean_coarse,
                             pbias_coarse, pbias_total_coarse, dhdt_coarse,
                             tbias_=tbias_coarse,
                             H_atm_=H_atm_coarse, f_clear_=f_clear_coarse)
    vti_writer.append(problem.mg[level], time=i)
    vti_writer.write_pvd()

prev_lrs = None
for level in range(config.max_level, config.min_level - 1, -1):
    problem.model.set_top_level(level)
    diag = make_diagnostic_fields(problem.mg[level])
    level_dir = f"{OUTPUT_PATH}/level_{level}/vti"
    problem.write_observations(level_dir, level=level)
    thermal = problem.thermal_driver(level)     # None unless config.thermal
    vti_writer = make_loss_vti_writer(problem.mg[level], level_dir,
                                       config.vti_base_name, diag, thermal=thermal)

    for i in range(config.max_iters[level]):
        # Rank-few fingerprint nuisance: re-measure the smooth SMB
        # parameters' sensitivity fingerprints from the CURRENT state at
        # every level start, and every `refresh` iterations when > 0
        # (config.FingerprintNuisance; costs 1 + n_params*n_modes forwards).
        fpn = config.fingerprint_nuisance
        if fpn is not None and (i == 0 or (fpn.refresh or 0) > 0
                                and i % fpn.refresh == 0):
            problem.refresh_fingerprints(params=params, level=level)

        optimizer_sgd.zero_grad()
        #optimizer_adam.zero_grad()
        # Scheduled learning rates share the loss-weight continuation contract:
        # ramps are honored here (schedule=True) and nowhere else.
        lrs = refresh_learning_rates(i, level)
        if i == 0 or lrs != prev_lrs:
            print("learning rates:",
                  ", ".join(f"{k[3:]}={v:.3g}" for k, v in lrs.items()))
        prev_lrs = lrs

        # Periodically emit a per-time-step VTI series.
        time_writer = (make_time_vti_writer(problem.mg[level], level_dir, thermal=thermal)
                       if i % 10 == 0 else None)

        sim, physical = problem.simulate(
            level=level, params=params, time_writer=time_writer)
        loss_terms = problem.compute_loss(
            sim=sim, physical=physical, params=params,
            iteration=i, level=level, schedule=True)
        loss_terms.log(i)

        write_loss_vti(diag, vti_writer, sim, physical, level, i)
        
        loss_terms.J.backward()
        # Likelihood-side influence control on whitened blocks: semi-modular
        # eta on the SMB block and/or per-parameter bounded-influence caps
        # (tanh saturation of the data score at C_z prior-stds). Exact — the
        # whitened prior gradient is analytic; no-op at eta = 1 with no caps.
        # See GlacierConfig.smb_data_influence / influence_cap.
        if config.smb_data_influence != 1.0 or config.influence_cap:
            sat = apply_influence_control(
                params, eta=config.smb_data_influence,
                caps=influence_caps,
                transfer=config.influence_transfer,
                loss_scale=resolve_weight(config.loss_scale, i, level,
                                          schedule=True, what="loss_scale"))
            if sat and i % 25 == 0:
                print("influence saturation:",
                      ", ".join(f"{k[2:]}={v:.2g}x" for k, v in sat.items()))
        optimizer_sgd.step()
        #optimizer_adam.step()

        # Drop this iteration's snapshots/graph now: `sim` pins fine-grid
        # tensors per observation epoch, and letting them survive into the
        # next simulate() call doubles the snapshot footprint.
        del sim, physical, loss_terms

        # Return freed blocks to the driver: torch and cupy each hoard their
        # own caching pool, and with the big transients interleaving between
        # the two allocators either can OOM while the other sits on free
        # memory. Costs milliseconds per iteration against a full solve.
        cp.get_default_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()
        #save_whitened_params(params, f"{OUTPUT_PATH}/level_{level}/torch_vars.p", bed_parametrization=problem.priors.bed_parametrization)

    # Final evaluation (no backward) so the multigrid state matches the
    # converged parameters before we save it out. Its residual fields (raw
    # and whitened, per field likelihood) go to level_<n>/vti/residuals.pvd;
    # only the finest level's are statistically meaningful (coarser levels
    # compare prolonged fields).
    sim, physical = problem.simulate(level=level, params=params)
    with torch.no_grad():
        problem.write_residuals(level_dir, sim, physical)
    del sim, physical

    mg_lvl = problem.mg[level]
    ds = xr.merge([
        mg_lvl.state.u.to_dataarray(),
        mg_lvl.state.v.to_dataarray(),
        mg_lvl.state.ud.to_dataarray(),
        mg_lvl.state.vd.to_dataarray(),
        mg_lvl.state.H.to_dataarray(),
        mg_lvl.geometry.bed.to_dataarray(),
        mg_lvl.sliding.beta.to_dataarray(),
        mg_lvl.forcing.smb.to_dataarray(),
    ])
    if thermal is not None:
        # end-of-run basal and depth-averaged temperature (K), on the H grid
        H_da = mg_lvl.state.H.to_dataarray()
        for name, arr in thermal.temperature_fields().items():
            ds[name] = H_da.copy(data=cp.asnumpy(arr)).rename(name).assign_attrs(
                units="K", long_name={"T_bed": "basal ice temperature",
                                      "T_mean": "depth-averaged ice temperature"}[name])
    ds.to_netcdf(f"{OUTPUT_PATH}/level_{level}/inverse_soln.nc")
    save_whitened_params(params, f"{OUTPUT_PATH}/level_{level}/torch_vars.p",
                         bed_parametrization=problem.priors.bed_parametrization)
