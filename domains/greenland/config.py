"""
Greenland ice sheet domain configuration (ISMIP7 initialization).

Whole ice sheet + peripheral glaciers on the ISMIP standard 1 km grid
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

# Reanalysis behind the SMB forcing (2026-09-20). Both are a merged input file
# (monthly climatology 1986-2025 in monthly_t2m / monthly_precip, everything
# else identical) plus a year-by-year anomaly file relative to it:
#   "carra2": CARRA2 100 m temperature on the DEM + CARRA tp, years 1986-2025
#             (make_carra_vars.py, make_climate_yearly.py) - the calibration
#             forcing so far; main-sheet precip 824 Gt/yr;
#   "racmo":  RACMO2.3p2-ERA5 2 m temperature + pr from the ISMIP7 kit, years
#             1958-2025 (make_racmo_vars.py, make_climate_yearly.py --source
#             racmo); 714 Gt/yr, JJA 1.1 K colder. 15.8 GB of int16 codes:
#             yearly_climate_cache="none" reads them from the page cache if
#             pinning that much host RAM is a problem.
#   "hybrid": CARRA2 temperature + RACMO precipitation, years 1986-2025
#             (make_climate_hybrid.py: a copy of the two files' codes). The
#             CONTROLLED experiment: only the precipitation differs from
#             "carra2", so the melt physics, the 100 m temperature level and
#             the rain/snow split are those of inverse_v4 ("racmo" also swaps
#             in a 2 m temperature pinned near 0 degC over melting ice). The
#             products' ice-mean annual precip ratios correlate at r = 0.97
#             (both ERA5-bounded), so the T-P covariance survives the mix.
# The Vinther index covers the years before the record in every case.
CLIMATE = "hybrid"
_CLIMATE_FILES = {
    "carra2": ("GLIDE_inputs.nc", "gridded_climate_yearly.nc"),
    "racmo": ("GLIDE_inputs_racmo.nc", "gridded_climate_yearly_racmo.nc"),
    "hybrid": ("GLIDE_inputs_hybrid.nc", "gridded_climate_yearly_hybrid.nc"),
}
_GRIDDED_FILE, _YEARLY_FILE = _CLIMATE_FILES[CLIMATE]

CONFIG = GlacierConfig(
    base_dir=str(_HERE),
    vti_base_name="greenland",
    gridded_filename=_GRIDDED_FILE,
    results_subdir=f"inverse_v9",
    smb_model="enthalpy",
    anomaly_integration="mean_anomaly",
    #anomaly_filename = "temperature_anomaly_flat.nc",
    # Interannual-variance quadrature over the index years (2026-09-22).
    # 1.05 K is the measured ice-sheet-mean JJA interannual std of the
    # forcing (hybrid yearly file, 1986-2025: JJA 1.055, annual 1.023), which
    # is the right one because the scalar anomaly shifts every month and melt
    # only feels summer. Independently confirmed: with the OCX-implied
    # curvature of -160 Gt/yr/K^2, 0.5 * curv * (1.055^2 - 0.6^2) = -60 Gt/yr
    # against the -62 the OCX experiment measured between the index and the
    # resolved yearly fields at identical mean forcing. A deep step resolves
    # none of it, so the correction there is 0.5 * curv * 1.055^2 = -89 Gt/yr.
    # 3 nodes = 3 SMB evaluations per index step instead of 1; with dt 50 from
    # t_start and 10-yr steps after 1850 that is ~100 extra glare calls for
    # the whole run, against no change to the dynamics solves.
    #interannual_sigma=None,
    interannual_sigma=1.05,
    interannual_nodes=3,
    stress_scheme="molho",

    # ---- time: historical window ending at the ISMIP7 projection start.
    # Observations: surface ~2015 (ArcticDEM overlay) / 2007 (BedMachine
    # surface), velocity 2015-16, extent ~2015, ATL15 dh/dt 2019-2023 (the
    # horizon auto-extends to the latest observation).
    t_start=100.0,
    t_end=2020.0,
    dt=25.0,
    # 3-yr steps over the observational period (equal sub-steps between the
    # epochs 1992, 2008, 2015, 2018, 2019, 2026; the coarse dt before 1990)
    dt_schedule=((1850,10.0),(1990.0, 1.0),),
    grad_start_time=1700,      # revisit once a level-3 FD sweep is done
    # temperature_anomaly.nc is the Vinther SW-Greenland JJA series (1784-2013,
    # CARRA2-extended to 2025), referenced to the CARRA2 climatology window,
    # so no base year is subtracted. alpha_t2m scales the coastal anomaly
    # onto the ice sheet: RACMO tas JJA regressed on the Vinther JJA series
    # (1958-2025) gives 0.58 ice-sheet mean, 0.51 CE .. 0.70 SW, r 0.6-0.8
    # (analysis/arctic_amplification.py --regressor vinther).
    base_anomaly_year=None,
    alpha_t2m=0.6,
    # Precipitation follows the same index (2026-09-22,
    # preprocessing/make_precip_anomaly.py -> precip_anomaly.nc). Before this
    # the pathway was dormant (no file, alpha_precip 0), so accumulation sat
    # at the modern climatology through the whole spin-up while only the
    # temperature half of a colder past was applied - a one-signed bias that
    # grows with the spin-up length. The file holds
    # exp(gamma_ann * dTann_dindex * index) normalized to 1 at
    # `base_precip_year`, with gamma_ann 5 %/K of ice-sheet ANNUAL temperature
    # and dTann_dindex 0.73 (measured over 1986-2025), i.e. 3.65 %/K of index.
    # alpha_precip multiplies gamma to first order, so sweep it here rather
    # than rebuilding: 1.0 = 5 %/K, 0.0 = the old flat accumulation, 1.4 = the
    # Clausius-Clapeyron ceiling. 5 %/K is the modelling convention and the
    # ice-core glacial-interglacial value; the 40-yr reanalysis gives
    # +2.3 +- 1.2 %/K, indistinguishable from zero because interannual
    # Greenland precipitation is circulation-driven, so it neither supports
    # nor excludes it. The multiplier acts on INDEX years only - record years
    # carry their own per-year precip_ratio - so it cannot double-count.
    # Base year 2006: the climatology-window year whose index is closest to
    # zero (-0.054 K, a 0.20 % residual), since the library references this
    # series to a single year and a climatology reference is what is wanted.
    #alpha_precip=0.0,
    alpha_precip=1.0,
    base_precip_year=2006,
    # 2026-09-18: the years CARRA2 covers (1986-2025) are forced by the
    # reanalysis year itself (t2m anomaly + precip ratio on the climatology,
    # preprocessing/make_climate_yearly.py); the Vinther index only acts
    # before 1986. Both dh/dt windows lie inside the record, so the
    # calibration sees the same weather as the OCX hindcast. Set to None
    # for the climatology + index forcing everywhere (the pre-2026-09-18
    # inversions). The fields never sit on the tape (yearly_climate.py).
    yearly_climate_filename=_YEARLY_FILE,
    yearly_climate_cache="ram",
    # Drift / spin-up diagnostic (2026-09-22): True holds the atmosphere at
    # the reference climate for the whole run (monthly climatology + the
    # calibrated tbias / pbias, no yearly fields, no Vinther index), so the
    # dh/dt the model produces is its own relaxation from the initial
    # geometry. The ocean is held at its TF climatology by the same flag
    # (dTF = 0, so the margins keep their climatological geography but no
    # interannual variability) - the thermal forcing drives mass change on
    # the same order as the atmosphere, so both have to be fixed.
    climatology_only=False,

    n_levels=6,
    max_level=0,
    max_iters=(50, 100, 200),

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
    # effectively no-slip above this; also forward_standalone.py's BETA_MAX.
    # Without it beta runs away where xi -> 0 (95 at Humboldt) and the
    # adjoint's effective-pressure term goes stiff (2026-09-17)
    beta_max=20.0,
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
    calving_H_c=100,

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
        tf_crit=5.5, clim_q=0.0, clim_h=15.0, alpha_q=0.0, alpha_h=100.0,
        # Front pin (2026-09-24, library change 15): set pin_front="rgi_mask"
        # to hold the front at BedMachine's 2015 ice mask for the WHOLE run and
        # invert traction over the observed geometry; the TF settings above are
        # then ignored. None = the TF-driven margins.
        pin_front=None),
    # ---- FAS / Vanka settings from the same example, both solvers
    forward_solver=SolverConfig(coarsest_steps=200, pre_steps=10, post_steps=150,
                                finest_steps=0, relative_tolerance=1e-2,
                                absolute_tolerance=10.0, report_norms=True,
                                omega=0.5, momentum_damping=0.1, step_tolerance=1e-6),
    adjoint_solver=SolverConfig(coarsest_steps=200, pre_steps=10, post_steps=150,
                                finest_steps=0, relative_tolerance=1e-2,
                                absolute_tolerance=1e-5, report_norms=False,
                                omega=0.5, momentum_damping=0.01, step_tolerance=1e-6),

    # ---- enthalpy SMB constants: Greenland interior is clearer and colder
    mu_cloud_factor=0.45,
    q_lw0=-35.0,

    # ---- priors (l in metres; 1 km cells)
    bed_prior=PriorHyperparams(sigma=500.0, l=4000.0, nu=1),
    mean_prior=PriorHyperparams(sigma=1000.0, l=30000.0, nu=1),
    log_beta_prior=PriorHyperparams(sigma=1.0, l=8000.0, nu=1),
    pbias_prior=PriorHyperparams(sigma=0.3, l=50000.0, nu=1),
    tbias_prior=PriorHyperparams(sigma=1.0, l=50000.0, nu=1),
    h_atm_prior=PriorHyperparams(sigma=0.2, l=150000.0, nu=1),
    cloud_prior=PriorHyperparams(sigma=0.25, l=150000.0, nu=1),

    # z_pbias 1.0 + lr_z_pbias 5 (2026-09-21): under the tightened velocity /
    # dh/dt terms the interior slows within a few iterations while the SMB
    # block moves over hundreds (prior sigma 0.1 preconditions the pbias
    # step by 0.01); at cap 0.3 / lr 1 pbias in the north did not move at all
    # over 8 level-2 iterations from the v5 state while the 1200-2000 m
    # thickening grew. lr 5 / cap 1.0 is monotone with pbias N -0.0006 per
    # iteration; lr 10 uncapped oscillates (pbias to 0.02 at outlets, the SE
    # swinging 0.79-0.99): the cap is what keeps the outlets' dh/dt misfit
    # from being dumped into precipitation. A curvature-aware optimizer in
    # whitened coordinates would be the real fix.
    influence_cap={"z_log_H_atm": 0.3, "z_logit_cloud": 0.3, "z_tbias": 0.3, "z_pbias": 0.3},
    influence_transfer="log",

    observations=(
        SurfaceSpec(noise=MaternNoise(sigma=15.0, l=4000.0, nu=0.5, nugget=10.0),
                    weight=1.0, nu=3),
        # 2026-09-21: per-pixel errors from the ITS_LIVE vx_err / vy_err
        # rasters, sigma_c = max(5 m/yr, err_c, 0.05 |v|), correlated at 10 km
        # with an equal white part. The scalar 100 m/yr left the interior
        # (7-20 m/yr, product error 0-3) unconstrained, so flux divergence
        # absorbed every accumulation excess (inverse_v5: D rose 1:1 with
        # SMB). The floor is the model's own error on slow ice and covers the
        # mosaic's zero-error pixels; 5 % of speed is its structural error at
        # the outlets. Was: MaternNoise(sigma=100.0, l=10000.0, nugget=100.0).
        VelocitySpec(noise=MaternNoise(sigma=1.0, l=10000.0, nugget=1.0), weight=1.0,
                     per_pixel_error=True, sigma_floor=10.0, sigma_rel=0.05, sigma_rel_km=5.0,
                     surge_biased=False, mask_unobserved=True, nu=3),
        #VelocitySpec(noise=MaternNoise(sigma=50.0, l=10000.0, nugget=50.0), weight=1.0,
        #             surge_biased=False, mask_unobserved=True, nu=1),

        ExtentSpec(weight=1.0, s_H=10.0, sigma_p=0.3,
                   logit_error=MaternNoise(sigma=0.3, l=5000.0),
                   nuisance_inner_steps=2, eps_max=1.0),
        #BedSpec(weight=0.0),     # data lives in the conditioned prior map
        # 2000-2020 end-of-summer snowlines (make_snowline.py): the label is
        # the fraction of seasons with snow, so the model probability is the
        # mean over those 21 years of sigmoid(SMB_year / s_smb) — each year
        # its own step under the yearly CARRA2 forcing (window="file" =
        # the product's time_start..time_end). two_sided: the product
        # classifies both sides on the main sheet, and the one-sided score
        # (penalize missing snow only) overshot — inverse_v2 put snow below
        # the snowline for free (bare ice 130 k vs 187 k km2, SMB 555 Gt/yr,
        # pbias +8 %; 2026-09-19).
        # loss="hinge" (2026-09-20): a snowline says which SIGN the season's
        # SMB had, not how large it was, and the Brier score on
        # sigmoid(SMB / s_smb) cannot saturate on the dry interior's
        # 0.1-0.5 m/yr (inverse_v3: 63 % of the misfit above 2000 m, pbias
        # x1.08 there; s_smb 0.1 cured the interior but not the north). The
        # squared hinge is exactly zero once a cell-season is right by
        # `margin`, quadratic in the violation over s_smb^2 (s_smb = per-cell
        # per-season SMB error std), pseudo-Huber beyond `huber` m/yr, and
        # scores every season against its own label. On the s_smb-0.1 state:
        # <5 % of the sum above 2000 m, no upward pull there. The Brier
        # variant was SnowlineSpec(weight=1.0, s_smb=0.1, sigma_p=0.5,
        # window="file", two_sided=True, logit_error=MaternNoise(sigma=0.5,
        # l=10000.0), nuisance_inner_steps=2, eps_max=1.0).
        SnowlineSpec(weight=1.0, loss="hinge", s_smb=0.35, margin=0.02,
                     huber=1.0, window="file", two_sided=True),
         #sigma_floor 0.1 (2026-09-21; the default 0.5 was set for the
        # ablation zone): times noise.sigma the effective interior floor is
        # 0.05 m/yr - the firn-anomaly uncertainty the model cannot
        # represent - instead of 0.25, so the products' 0.2-2 cm/yr interior
        # precision can see the 5-10 cm/yr thickening the northern
        # accumulation excess produces at 1200-2000 m (inverse_v5).
        # sigma_rel 0.25: at the outlets the model's structural error (retreat
        # timing, 1 km resolution) is a quarter of the observed rate, which
        # keeps their weighting where the old floor had it (0.25 m/yr at a
        # 1 m/yr thinning) while the interior tightens.
        #DhdtSpec(noise=MaternNoise(sigma=0.5, l=10000.0, nu=0.5, nugget=0.5),
        #         sigma_floor=0.1, sigma_rel=0.25, sigma_rel_km=5.0, weight=1.0, nu=3),   # ATL15 2019-2026 (gridded_dhdt.nc)
        # MEaSUREs / ITS_LIVE G1920V01 dh trend over 1992-2019: the earlier,
        # non-overlapping window that spans the onset of the tidewater
        # retreats (make_dhdt.py --source itslive_dh --t0 1992 --t1 2019
        # --name measures). Skipped when the file is absent.
        #DhdtSpec(filename="gridded_dhdt_measures.nc", name="dhdt_measures",
        #         noise=MaternNoise(sigma=0.5, l=10000.0, nu=0.5, nugget=0.5),
        #         sigma_floor=0.1, sigma_rel=0.25, sigma_rel_km=5.0, weight=1.0, nu=3),
    ),
    loss_scale=1e-3,
    bed_conditioning=BedConditioningConfig(
        enabled=True,
        use_gridded_bed=True,
        gridded_bed_err_floor=20.0,
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

    lr_z_bed=0.025,
    lr_z_log_beta=0.25,
    lr_z_pbias=1.0,             # see influence_cap
    lr_z_tbias=1.0,
    lr_z_log_H_atm=1.0,
    lr_z_logit_cloud=1.0,
)
