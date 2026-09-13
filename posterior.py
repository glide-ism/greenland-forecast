"""
Empirical posterior covariance from RTO samples.

Loads every sample under {INPUT_PATH}/rto/, maps each whitened parameter
back to physical space using the shared prior models, and computes SVD-based
covariance factorizations for bed and precipitation bias.
"""
from pathlib import Path

import numpy as np
import torch

from ggapp.torch import GGaPPMap

from glacier_inverse import load_config
from glacier_inverse.priors import GlacierPriors, domain_shape

# Available domains: domains/{greenland,greenland_coarse}
DOMAIN = "domains/greenland"
config = load_config(DOMAIN)


INPUT_PATH = config.output_dir
MAX_SAMPLES = 100000

ny, nx, dx = domain_shape(config)
priors = GlacierPriors(config, ny, nx, dx)

beds = []
pbiases = []
tbiases = []
log_betas = []
log_mfs = []
log_rfs = []
log_H_atms = []
f_clears = []

for d in Path(f"{INPUT_PATH}/rto_lr_decay/").iterdir():
    try:
        data = torch.load(f"{d}/level_3/torch_vars.p")
        saved = data.get("bed_parametrization", "legacy")
        if saved != priors.bed_parametrization:
            raise ValueError(
                f"{d}: sample saved with bed_parametrization={saved!r} but "
                f"priors are {priors.bed_parametrization!r} — z_bed would be "
                f"misinterpreted. Match config.bed_conditioning to the runs.")
        # The shared whitened->bed map handles both parametrizations (under
        # bed conditioning the fluctuation field carries the kriging
        # correction; bed_mean stays prior-coupled, as in legacy).
        beds.append(priors.bed_from_whitened(
            data["bed"], data["bed_mean"])[0].cpu().detach())
        pbiases.append(GGaPPMap.apply(priors.pbias_model, data["precipitation_bias"]).cpu().detach())
        if priors.tbias_model is not None and "temperature_bias" in data:
            tbiases.append(GGaPPMap.apply(priors.tbias_model, data["temperature_bias"]).cpu().detach())
        # Enthalpy parameter fields — mapped like tbias when both the model
        # and a field-shaped sample exist (0-d entries are pre-field RTO
        # samples; skip them rather than misinterpret).
        if (priors.h_atm_model is not None and "log_H_atm" in data
                and data["log_H_atm"].dim() == 2):
            log_H_atms.append(
                (priors.mu_log_H_atm
                 + GGaPPMap.apply(priors.h_atm_model, data["log_H_atm"])
                 ).cpu().detach())
        if (priors.cloud_model is not None and "logit_cloud" in data
                and data["logit_cloud"].dim() == 2):
            f_clears.append(torch.sigmoid(
                priors.mu_logit_cloud
                + GGaPPMap.apply(priors.cloud_model, data["logit_cloud"])
                ).cpu().detach())
        log_betas.append(priors.log_beta_from_whitened(data["log_beta"]).cpu().detach())
        log_mfs.append(data["log_mf"].cpu().detach() * priors.sigma_log_mf + priors.mu_log_mf)
        log_rfs.append(data["log_rf"].cpu().detach() * priors.sigma_log_rf + priors.mu_log_rf)
    except FileNotFoundError:
        print(d)

bed_samples      = torch.stack(beds[:MAX_SAMPLES],      axis=0)
pbias_samples    = torch.stack(pbiases[:MAX_SAMPLES],   axis=0)
log_beta_samples = torch.stack(log_betas[:MAX_SAMPLES], axis=0)
log_mf_samples   = torch.stack(log_mfs[:MAX_SAMPLES],   axis=0)
log_rf_samples   = torch.stack(log_rfs[:MAX_SAMPLES],   axis=0)


def _centered_factor(samples):
    flat = torch.stack([s.ravel() for s in samples], axis=-1)
    flat = (flat - flat.mean(axis=1, keepdims=True)) / np.sqrt(flat.shape[1] - 1)
    u, s, _ = torch.linalg.svd(flat, full_matrices=False)
    return u*s, s


L_bed, s_bed = _centered_factor(bed_samples)
L_pbias, s_pbias = _centered_factor(pbias_samples)
if tbiases:
    tbias_samples = torch.stack(tbiases[:MAX_SAMPLES], axis=0)
    L_tbias, s_tbias = _centered_factor(tbias_samples)
if log_H_atms:
    log_H_atm_samples = torch.stack(log_H_atms[:MAX_SAMPLES], axis=0)
    L_h_atm, s_h_atm = _centered_factor(log_H_atm_samples)
if f_clears:
    f_clear_samples = torch.stack(f_clears[:MAX_SAMPLES], axis=0)
    L_f_clear, s_f_clear = _centered_factor(f_clear_samples)
