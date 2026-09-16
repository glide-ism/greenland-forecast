"""
Greenland COARSE development domain (1800 m, bootstrapped from glide's example h5).

Whole ice sheet block-averaged from GLIDE_greenland_inputs.h5 by
preprocessing/bootstrap_from_glide_example.py: no dh/dt, no snowline, a flat
100 m bed error and a single basin label. For smoke tests and driver
development only; the science configuration is domains/greenland
(local_data/domain.json). Everything physical lives here so inverse / rto /
posterior / sensitivity share one model; per-task knobs stay in the drivers.

What differs from an Alaska mountain-range domain and why:
  * seawater flotation (rho_water 1028), a height-above-buoyancy calving
    sink and a floating branch: BedMachine mask == 3 cells exist (Petermann,
    79N, Ryder, ...);
  * the ice sheet cannot be grown from ice-free in 1000 years, so the run
    is seeded from the observed geometry (init_from_observed_geometry) and
    integrates a short relaxation/historical window instead of a spin-up;
  * the bed prior is conditioned on the GRIDDED BedMachine bed at its own
    per-cell error (bed_conditioning.use_gridded_bed) — no flightline file;
  * longer prior correlation lengths (1 km cells; sliding and SMB structure
    at 5-50 km) and a colder rate factor;
  * no debris term, no per-glacier surge marginal (drainage basins are the
    labels; surge_type is 0 except RGI-flagged peripheral glaciers).
Learning rates are copied from the tuned Alaska domains and WILL need
retuning on the level-3 trace — they are coupled to the prior hyperparameters.
"""
from pathlib import Path

import numpy as np

from glacier_inverse.config import (
    BedConditioningConfig, GlacierConfig, MaternNoise, OceanForcingConfig,
    PriorHyperparams, Schedule, SolverConfig,
)
from glacier_inverse.observations import (
    BedSpec, DhdtSpec, ExtentSpec, SnowlineSpec, SurfaceSpec, VelocitySpec,
)

_HERE = Path(__file__).parent

CONFIG = GlacierConfig(
    base_dir=str(_HERE),
    vti_base_name="greenland_coarse",
    results_subdir="inverse",
    smb_model="enthalpy",
    anomaly_integration="mean_anomaly",
    stress_scheme="molho",

    # ---- time: historical window ending at the ISMIP7 projection start.
    # Observations: surface ~2015 (ArcticDEM overlay) / 2007 (BedMachine
    # surface), velocity 2015-16, extent ~2015, ATL15 dh/dt 2019-2023 (the
    # horizon auto-extends to the latest observation).
    t_start=1515.0,
    t_end=2015.0,
    dt=20.0,
    # 3-yr steps over the observational period (equal sub-steps between the
    # epochs 1992, 2008, 2015, 2018, 2019, 2026; the coarse dt before 1990)
    dt_schedule=((1990.0, 3.0),),
    grad_start_time=None,      # revisit once a level-3 FD sweep is done
    # temperature_anomaly.nc is the Vinther SW-Greenland JJA series (1784-2013,
    # CARRA2-extended to 2025), referenced to the CARRA2 climatology window,
    # so no base year is subtracted. alpha_t2m scales the coastal anomaly
    # onto the ice sheet: RACMO tas JJA regressed on the Vinther JJA series
    # (1958-2025) gives 0.58 ice-sheet mean, 0.51 CE .. 0.70 SW, r 0.6-0.8
    # (analysis/arctic_amplification.py --regressor vinther).
    base_anomaly_year=None,
    alpha_t2m=0.6,

    n_levels=6,
    max_level=3,
    max_iters=(100, 100, 100, 100),

    init_from_observed_geometry=True,
    init_H_floor=1.0,          # = thklim
    use_avalanche_model=False,

    # ---- ice physics (from glide's Greenland example)
    rho_water=1028.0,
    A_glen=1e-17,
    n_glen=3,
    H_reg=25.0,
    sliding_m=1.0 / 3.0,
    beta_init=2.5,
    mu_log_beta=float(np.log(2.5)),
    # 1e-3, not glide's example 1e-4: with beta_eff = beta*xi vanishing on
    # ice-free/floating cells, water_drag is their only drag, and at 1e-4
    # the ocean cells at fast outlet fronts are nearly dragless (11 km/yr
    # forward velocities, a near-singular momentum block) — the adjoint
    # solve at the velocity epoch stalls at ~0.96/V-cycle. 1e-3 converges in
    # 2 cycles; on grounded ice it is a few % of the Weertman drag (absorbed
    # by the inferred beta), on floating ice ~9 kPa at 1 km/yr.
    water_drag=1e-3,
    # Signed-flotation geometry and height-above-buoyancy calving, as in
    # glide's examples/greenland/greenland_forward.py (2026-09 refactor):
    # depth = -bed, phi = sigmoid(sigmoid_c * (rho_i/rho_w H - depth)) is 1/2
    # exactly at flotation; ice with H < (1 + q) H_f decays at H / timescale
    # per year down to thklim. q = -0.5 removes only ice thinner than half
    # its flotation thickness, so floating tongues thicker than that persist
    # (the example's inverse uses timescale 0.1 with the same q).
    thklim=1.0,
    sigmoid_c=1.0,
    calving_timescale=0.5,
    # baselines of the hybrid threshold H - H_f < q H + h0: q = 0 calves
    # exactly the floating ice before the ocean warms (the standalone
    # experiments' setting); the ocean forcing below perturbs both margins
    calving_q=0.0,
    calving_h0=0.0,
    # floating tongues persist until thinner than H_c (m); the q H + h0
    # criterion above governs grounded fronts (blended by phi). Starting
    # value, not a fit: Petermann / 79N / Ryder fronts are ~100-200 m thick.
    calving_H_c=100.0,

    # ---- ocean thermal forcing of the calving margins (ISMIP7 EN4 TF,
    # model_inputs/thermal_forcing.nc; glacier_inverse/ocean.py). Per step
    #   h0 = calving_h0 + clim_h * (TF_clim - tf_crit) + alpha_h * dTF
    # (q likewise). Under the monotone calving law the sign of the margin at
    # flotation decides whether a tongue is admissible: tf_crit = 3.5 degC
    # puts Petermann (2.0 in the 1950-79 climatology, 2.4 at warmest) and
    # 79N (2.2) below it, Jakobshavn (3.8, cold years 3.0) marginal, and
    # Helheim (6.2) never floating; Kangerlussuaq (4.3) is thermally
    # indistinguishable from Jakobshavn in this product. clim_h = 15 m/K
    # gives baselines of about +40 m Helheim, +5 m Jakobshavn, -22 m
    # Petermann; alpha_h = 50 m/K swings Jakobshavn between -35 and +80 m
    # over its anomaly range. STARTING POINTS for the (clim_h, alpha_h) sweep.
    ocean_forcing=OceanForcingConfig(
        enabled=True, statistic="mean", ref_years=(1950, 1979), max_dist_km=5.0,
        tf_crit=3.5, clim_q=0.0, clim_h=15.0, alpha_q=0.0, alpha_h=50.0),

    # ---- FAS / Vanka settings from the same example, both solvers
    forward_solver=SolverConfig(coarsest_steps=200, pre_steps=10, post_steps=150,
                                finest_steps=0, relative_tolerance=1e-2,
                                absolute_tolerance=10.0, report_norms=False,
                                omega=0.5, momentum_damping=0.01, step_tolerance=1e-6),
    adjoint_solver=SolverConfig(coarsest_steps=200, pre_steps=10, post_steps=150,
                                finest_steps=0, relative_tolerance=1e-2,
                                absolute_tolerance=1e-5, report_norms=False,
                                omega=0.5, momentum_damping=0.01, step_tolerance=1e-6),

    # ---- enthalpy SMB constants: Greenland interior is clearer and colder
    mu_cloud_factor=0.45,
    q_lw0=-35.0,

    # ---- priors (l in metres; 1 km cells)
    bed_prior=PriorHyperparams(sigma=300.0, l=6000.0, nu=1),
    mean_prior=PriorHyperparams(sigma=1000.0, l=30000.0, nu=1),
    log_beta_prior=PriorHyperparams(sigma=1.0 / 3.0, l=12000.0, nu=1),
    pbias_prior=PriorHyperparams(sigma=0.1, l=50000.0, nu=1),
    tbias_prior=PriorHyperparams(sigma=1.0, l=50000.0, nu=1),
    h_atm_prior=PriorHyperparams(sigma=0.2, l=150000.0, nu=1),
    cloud_prior=PriorHyperparams(sigma=0.25, l=150000.0, nu=1),

    influence_cap={"z_log_H_atm": 0.3, "z_logit_cloud": 0.3, "z_tbias": 0.3},
    influence_transfer="log",

    observations=(
        SurfaceSpec(noise=MaternNoise(sigma=20.0, l=6000.0, nu=0.5, nugget=10.0),
                    weight=1.0, nu=3),
        VelocitySpec(noise=MaternNoise(sigma=20.0, l=10000.0, nugget=10.0), weight=1.0,
                     surge_biased=False, mask_unobserved=True, nu=3),
        ExtentSpec(weight=1.0, s_H=10.0, sigma_p=0.3,
                   logit_error=MaternNoise(sigma=0.3, l=5000.0),
                   nuisance_inner_steps=2, eps_max=1.0),
        BedSpec(weight=0.0),     # data lives in the conditioned prior map
        SnowlineSpec(weight=1.0, s_smb=0.5, sigma_p=0.3,
                     logit_error=MaternNoise(sigma=0.3, l=10000.0),
                     nuisance_inner_steps=2, eps_max=1.0),
        DhdtSpec(noise=MaternNoise(sigma=1.0, l=10000.0, nu=0.5, nugget=0.5),
                 weight=1.0),               # ATL15 2019-2026 (gridded_dhdt.nc)
        # MEaSUREs / ITS_LIVE G1920V01 dh trend over 1992-2019: the earlier,
        # non-overlapping window that spans the onset of the tidewater
        # retreats (make_dhdt.py --source itslive_dh --t0 1992 --t1 2019
        # --name measures). Skipped when the file is absent.
        DhdtSpec(filename="gridded_dhdt_measures.nc", name="dhdt_measures",
                 noise=MaternNoise(sigma=1.0, l=10000.0, nu=0.5, nugget=0.5),
                 weight=1.0),
    ),
    loss_scale=1e-3,
    bed_conditioning=BedConditioningConfig(
        enabled=True,
        use_gridded_bed=True,
        gridded_bed_err_floor=50.0,
        # Condition only on cells holding a radar pick (BedMachine dataid ==
        # 2, ~4% of the ice); between tracks the bed is the prior fluctuation
        # about bed_mean, informed by the flow likelihood alone. "all" would
        # condition on BedMachine's kriged / mass-conservation bed everywhere
        # (errbed is not a distance-to-data proxy, so gridded_bed_max_err
        # cannot isolate the observed cells).
        gridded_bed_data="radar",
        radar_fraction_min=0.0,
        seed_off_track_from_mean=False,   # True: start off-track from the smoothed bed
        gridded_bed_max_err=None,
        sigma_picks=50.0,
        sigma_dem=50.0,
        pcg_rtol=1e-3,
        pcg_rtol_adjoint=1e-2),

    lr_z_bed=0.0125,
    lr_z_log_beta=0.0125 * 9 * 9,
    lr_z_pbias=0.05,
    lr_z_tbias=10.0,
    lr_z_log_H_atm=10.0,
    lr_z_logit_cloud=10.0,
)
