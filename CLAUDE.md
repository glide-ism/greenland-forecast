# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

`greenland-forecast` is alaska-forecast ported to the Greenland ice sheet:
a thin orchestration layer over the glide-ism GPU libraries (`glide` ice
dynamics, `glare` SMB, `ggapp` GP priors, `gtic` insolation) that calibrates
bed, basal traction, and SMB parameters against time-stamped observations
and hands the state to ISMIP7. It is not an installable package; GPU is
mandatory; `torch` and `cupy` are mixed everywhere.

**The library is alaska-forecast's `glacier_inverse/` verbatim** except
for five additive changes (keep it that way so fixes can be ported back
and forth by diff):

1. `problem.py`: the flightline file is optional (an empty GeoDataFrame
   when absent) — Greenland's bed data is gridded.
2. `priors.build_bed_conditioning_data` + `config.BedConditioningConfig`:
   with `use_gridded_bed` (default True) gridded bed data in
   `GLIDE_inputs.nc` condition the bed prior at precision
   `1/(scale·max(err, floor))²`. `gridded_bed_data="all"` uses every
   BedMachine cell (`bed_obs`/`bed_obs_err`); `"radar"` (the Greenland
   configs' choice) uses only cells whose BedMachine subcells hold a radar
   pick (`dataid == 2` → `bed_radar_fraction`, `bed_obs_radar`,
   `bed_obs_radar_err` from `make_dem.py`), so between tracks the bed is the
   Matérn fluctuation about `bed_mean` informed by the flow likelihood alone
   (BedMachine's kriged/mass-conservation bed is not data, and its MC bed
   was built from the same velocity/thinning data the inversion uses).
   errbed is NOT a distance-to-data proxy (30 m floor 100 km from any
   track), so `gridded_bed_max_err` cannot do this. Off-ice cells stay
   with the DEM anchor. Inactive when the variables are absent.
3. `observations.BedSpec/BedObservation`: tolerate zero picks (only the
   off-ice anchor term remains).
4. `problem.py` initialization, gated on `bed_obs` being present: the initial
   bed is `bed_obs` where finite (DEM elsewhere) instead of the DEM
   composite, and `z_bed_mean` starts on a Gaussian-smoothed copy of that
   bed (sigma = mean_prior.l / 2) instead of zero. Without this the
   unconditional field `Map(z_bed)` — which the dense gridded-bed
   conditioning barely touches — stays at the SURFACE it was seeded from,
   and `bed_mean` converges to a smoothed surface. Note `bed_from_whitened`
   maps `z_bed` alone to the physical bed; the mean acts only in the prior
   residual, so `z_bed` whitens the full bed. In radar mode,
   `seed_off_track_from_mean=True` seeds the off-track ice cells from the
   smoothed bed instead of BedMachine's interpolation (picks kept on track),
   so between-track structure starts from the prior mean; default False
   (BedMachine as a warm start, not a constraint).
5. `config.base_anomaly_year` may be `None` (problem.py then uses
   `base_anomaly = 0`): the Greenland anomaly series is already referenced
   to the CARRA2 climatology window, and a single-year base would inject
   that year's local weather.
6. `observations.DhdtSpec(filename=, name=)`: a second dh/dt product over
   another window, loaded from `model_inputs/<filename>` on the same crop
   (not merged into GLIDE_inputs) and keyed by `name` in the loss / noise
   registry / residuals (`DhdtObservation` takes an instance name; the
   fingerprint targets match `dhdt*`). The Greenland configs pair ATL15
   2019-2026 (`dhdt`) with the MEaSUREs/ITS_LIVE G1920V01 trend over
   1992-2019 (`dhdt_measures`, `gridded_dhdt_measures.nc` from
   `make_dhdt.py --source itslive_dh --t0 1992 --t1 2019 --name measures`,
   or `make_all.py --extra-dhdt itslive_dh:1992:2019:measures`). The VTI
   dhdt diagnostic in inverse.py still shows the primary product only.
   **`make_dhdt.py --method endpoint` (2026-09-17)**: the default `trend`
   is a 1/rms^2-weighted line slope, but the inverse compares it with the
   model's two-snapshot rate (H(t1) - H(t0)) / (t1 - t0), and the two agree
   only for linear change. ITS_LIVE 1992-2019 is flat to 2003 then
   -340 Gt/yr, and the radar-era epochs are down-weighted (weight centroid
   2007), so the trend product integrates to -201 Gt/yr over the ice sheet
   (-230 with periphery) against -146 (-170) for the endpoint rate and
   Mankoff's -162; SW moves most (-35 -> -17, Mankoff -13), CE stays at -27
   against Mankoff's +5. `endpoint` = difference of 1/rms^2-weighted means
   over `--endpoint-window` (2 yr) windows centred on t0/t1, shifted inward
   where they overhang the record; the attrs' window becomes the window
   centres rounded to 0.1 yr (1992-2019 -> 1993.0-2019.0; ATL15 2019-2026
   -> 2020.0-2025.0, where the trend is fine because the loss is near
   linear). **Endpoint is the DEFAULT since 2026-09-17** for the time-series
   sources in both `make_dhdt.py` (`--method`, None -> endpoint; gridded /
   hugonnet rates are used as provided) and `make_all.py` (`--dhdt-method`,
   `--dhdt-endpoint-window`, and a per-product override as a fifth field,
   `--extra-dhdt itslive_dh:1992:2019:measures[:trend]`), so the commands
   above now build endpoint products. `gridded_dhdt_measures.nc` is the
   endpoint product (the old trend is kept as
   `gridded_dhdt_measures_trend.nc`); inversions before that date were fit
   to the trend, and the observation epoch moves from 1992 to 1993. The
   primary `gridded_dhdt.nc` (ATL15) is STILL the 2019-2026 trend: a rebuild
   with the new default turns it into 2020.0-2025.0 (pass `--dhdt-method
   trend` or `--dhdt-endpoint-window 1` to keep more of the span; the trend
   is fine there). Also found: over 2019-2023 ITS_LIVE thins 0.4 m/yr LESS than ATL15
   below 1000 m (-167 vs -259 Gt/yr; the 1.92 km product smooths the outlet
   margins), agreeing to cm/yr above 1500 m.
7. Ocean thermal forcing of the calving margins (`config.OceanForcingConfig`
   + `config.calving_h0`, `ocean.py`, `forward.simulate(ocean_forcing=)`):
   glide's hybrid height-above-buoyancy threshold `H - H_f < q H + h0` gets
   `q = calving_q + alpha_q dTF`, `h0 = calving_h0 + alpha_h dTF` per step
   from `model_inputs/thermal_forcing.nc`, generalized 2026-09-14/17 to
   `q = calving_q + clim_q (TF_clim - tf_crit) + alpha_q dTF` and
   `h0 = calving_h0 + clim_h (TF_clim - tf_crit) + alpha_h dTF` — under the
   monotone calving law the SIGN of the margin at flotation decides whether
   a tongue is admissible, so `tf_crit` (3.5 degC in the configs) is the
   critical thermal forcing: EN4 front climatologies Petermann 2.0, 79N 2.2,
   Jakobshavn 3.8, Kangerlussuaq 4.3, Helheim 6.2 degC; clim_h 15 m/K and
   alpha_h 50 m/K give margins of about -20 m Petermann, +5..+70 m
   Jakobshavn, +40..+80 m Helheim — with TF_clim the
   per-cell `ref_years` (1950-79) mean of the annual-mean or annual-max TF
   (1-7 degC at the fronts, mean 3.8) and dTF the step's departure from it
   (tenths of a degree; zero before the record and farther than
   `max_dist_km` from the native product). The clim coefficients give the
   baseline margins a TF-dependent geography; clim = 0 is the pure anomaly
   model (the tuned one), clim = alpha the pure absolute model, which
   over-levers (fronts either side of the zero crossing go insensitively
   cold or contracted — tried and dropped 2026-09-14). alpha_q (1/K,
   scales with thickness) and alpha_h (m/K, same distance everywhere) are
   SWEEP parameters, not differentiated: psi is a ~1 m-wide switch in
   flotation excess, so dJ/dalpha would be supported on a handful of cells.
   The step's margins carry no gradient; glide's `GlideStep` checkpoints
   q/h0 with each step so the adjoint re-solves with the right margins.
   `GlacierProblem.ocean_forcing` is the loaded forcing (None when disabled
   or the file is absent, with a warning); the same object drives
   `forward_standalone.py`.
8. `config.beta_max` (Optional, default None = the Alaska behaviour): an
   upper bound on the basal traction coefficient, clamped on the fine-level
   log beta before restriction in `problem.simulate`. Greenland sets 20; see
   the level-0 adjoint divergence paragraph below.
9. **Year-by-year forcing over the reanalysis record (2026-09-18)**:
   `config.yearly_climate_filename` (None = the Alaska behaviour; Greenland
   `gridded_climate_yearly.nc`, `preprocessing/make_climate_yearly.py`) +
   `yearly_climate.py` + `forward.YearField`. The file holds per-(year,
   month) `t2m_anom` (K) and `precip_ratio` relative to GLIDE_inputs' own
   climatology as int16 codes (0.002 K / 0.002), UNCOMPRESSED and chunked per
   (year, month) — the working format, 9.3 GB for CARRA2 1986-2025 at 1 km
   (record-mean anomaly 0.0000 K over the ice, max monthly cell mean 0.27 K
   from the per-year lapse correction; 0 ratios capped). `forward.simulate`
   makes every record year a step overlaps its own term `(YearField(y), w)`
   — one glare evaluation on `t2m_clim + anom(y) + tbias`,
   `precip_clim * pbias * ratio(y)` — and folds the remaining years into the
   usual `anomaly_integration` term with their weights renormalized (the
   1980-1990 step is 4 field terms + one mean-anomaly term at weight 0.6;
   with the 1-yr schedule after 1990 every step is one evaluation, so the
   cost is unchanged). The index precip multiplier applies to index years
   only. **Tape budget**: a year is 460 MB float32 at 1 km and a level-0 run
   overlaps 40, so the fields never enter the checkpointed SMB fn as
   tensors; it receives the `YearlyClimate` loader + the year (non-tensor
   args, like the scalar anomalies), decodes the codes on the GPU inside the
   checkpoint, and decodes them again in the backward recompute — nothing
   persists on the tape, the transient is one year's pair (+2.7 GB peak at
   level 2, forward+backward 1.5+1.3 s for 5 steps, gradients checked
   finite). `yearly_climate_cache="ram"` pins the int16 codes in host RAM
   (9.3 GB, one ~20 ms H2D per evaluation); `"none"` reads them from the file
   each time (page cache). Requires `base_anomaly_year=None` (raises
   otherwise). `forward_standalone.compute_smb` follows the same rule (so
   replays and `forward_projection --pre-record standalone` are consistent
   with the inversion; `YearlyClimate` with cache "none" there). Set the
   filename to None for the pre-2026-09-18 behaviour. Motivation: the OCX
   paragraph below.
10. `observations.SnowlineSpec(window=)` (2026-09-18): the model probability
   of the snowline term is the mean over a WINDOW of seasons of
   `sigmoid(SMB_year / s_smb)`, one step per year (the step ending at
   year + 1 carries that year's forcing), instead of the single nominal
   epoch — `window=(y0, y1)` or `"file"` (the product's
   time_start..time_end); None is the historical single-epoch term. The
   Greenland label is the fraction of the 2000-2020 seasons that ended with
   snow (`preprocessing/make_snowline.py`, see the snowline paragraph
   below), so under the yearly forcing the mean of the per-year
   probabilities is the model's fraction of seasons; the single-epoch term
   would compare the 21-year composite with ONE year's weather (2009's).
   The logit nuisance sees the averaged probability and slope. Records 21
   states (each keeps its fine SMB, 19 MB at 1 km); coarse-domain test:
   +0.7 GB, +1.7 s backward at level 2. `greenland_coarse` has the file
   too (smoke test includes the term).
11. `observations.SnowlineSpec(loss="hinge", margin=, huber=)` (2026-09-20;
   default `"brier"` = the Alaska behaviour): a squared hinge on the SIGN of
   each season's SMB instead of a Brier score on sigmoid(SMB / s_smb). Per
   cell i, season t, label y in [0, 1], SMB b: v+ = relu(m - b), v- =
   relu(m + b), l = y rho(v+) + (1 - y) rho(v-), rho(v) = v^2/2 (pseudo-Huber
   delta^2 (sqrt(1 + (v/delta)^2) - 1) with `huber` = delta), J = loss_scale
   weight 4^level / s_smb^2 sum_i omega_i mean_t l. It is the one-sided
   Gaussian: "b_true > 0 observed, error N(0, s_smb^2)" has log-likelihood
   -log Phi(b/s), wrong-side asymptote b^2 / 2 s^2, right side -> 0; the
   hinge keeps the asymptote and makes the right side EXACTLY zero, so a
   cell right by the margin has no loss and no gradient whatever its SMB
   (the logistic could not saturate on the interior's 0.1-0.5 m/yr and
   pulled precipitation up), while a wrong cell feels a gradient linear in
   the violation (Brier's ~ p(1-p) vanishes when badly wrong). Fractional y
   minimizes at b = m (2y - 1): the margin is the SMB half-width of the
   snowline inside a cell. With a window and per-year `snow_label` in the
   file every season is scored against ITS OWN label (uint8 on the device,
   restricted per level with the mask weights). `sigma_p` unused;
   `logit_error` raises (no nuisance in this mode). Greenland: s_smb 0.35
   (per-cell per-season SMB error std), margin 0.05, huber 1.0, two-sided.
   Offline on the s_smb-0.1 state: < 5 % of the sum above 2000 m with a
   slightly NEGATIVE pull there (the NE high-snowline years), 46 % at
   800-1400 m; J_snow ~650 x (0.25/0.35)^2 at that state. Coarse-domain
   test: forward+backward as fast as Brier, +0.7 GB for the 21 labels.
   **`hinge_window` (2026-09-23), how the window's seasons pair with the
   label.** The product is the END-OF-SUMMER SNOWLINE, i.e. the ice-firn
   interface, which survives several melt seasons; the model's b_t is ONE
   season's SMB whose zero contour is that year's ELA. The product is
   therefore a low-pass filter on the model's quantity, which is why the
   model's yearly bare-ice area is ~3x too variable (std 99 k vs 31 k, 2019
   506 k vs 223 k, r 0.38) while its mean snowline and its integrated SMB
   variability are both right - a quantity mismatch, NOT a model defect, and
   not something to chase in firn physics. `"per_season"` (the default, what
   v4-v7 ran) scores each season against its own `snow_label`, i.e. asks for
   a filtered observable at full bandwidth; at the 2019 state that is ~280 k
   cells scored wrong-side, a battle the model cannot win at any parameter
   setting. `"mean_prob"` averages the model as a PROBABILITY over the
   window, mean_t sigmoid(b_t / s_smb) = its fraction of seasons ending with
   snow, and hinges on the residual against the composite `snow_fraction`
   with tolerance `margin_prob` and scale `sigma_p`: the pairing the product
   justifies, and it keeps the saturation the hinge exists for, which the
   Brier window path lacks (interior y = 1, p ~ 0.95, Brier keeps paying
   0.05 and pulling precipitation up; this is exactly 0). `"mean_smb"` is the
   literal "average the ELA over the window" reading and is BROKEN - with a
   fractional y and two_sided both branches are live, so only |mean b| <=
   margin gives zero and every cell in the transition band carries a standing
   pull toward zero mean SMB. Synthetic band (21 seasons, sigma_y 0.5 m/yr),
   loss at the true state / at a 2.5x-FLATTENED SMB gradient: per_season
   0.0013 / 2.98, mean_prob 0.150 / 2.12, mean_smb 0.985 / 0.342 - i.e.
   mean_smb PREFERS the flattened state, which is the ELA-band pathology
   itself. Use `mean_prob`; mean_smb is kept only for a near-binary label.
   Per-year labels are loaded only for `per_season`.
   **`inverse_v4` (hinge, 2026-09-20; `analysis/output/basin_mb_v4/`)**: best
   snowline fit (bare ice 167 k vs 171 k km2, P(snow) within 0.03 of the label
   in every band above 200 m, mean bias +0.002) and best interannual SMB yet
   (std 124 vs 110, slope on Mankoff 1.08, dSMB/dT_jja -83 vs -80, trend -28
   vs -27 Gt/yr/decade, r 0.95); pbias x0.98 EVERYWHERE (precip 846 vs CARRA2
   raw 911) - the interior pull is gone. But SMB mean 453 (Mankoff 337), D 512
   (485), MB -58 (-148), windows -81 / -46 vs the products' -148 / -152: fitting
   the ELA exactly removed the last of the too-large ablation zone (below
   1400 m -85 Gt/yr vs -143 with s_smb 0.1). By elevation band against the
   kit's SDBN1 acabf (CESM2-driven, 2000-2014, main sheet, 4 km; total +361
   vs model +453): below 1200 m -49 vs -66 (+17), 1200-2000 m +138 vs +111
   (+27), above 2000 m +365 vs +315 (+50: 0.41 / 0.34 m ice/yr vs 0.37 /
   0.28). So with the right ELA the residual is mostly ACCUMULATION - CARRA2
   precipitation (+15 % vs RACMO) and no sublimation term - which only pbias
   can remove and only the long-wavelength mass budget can drive: the
   basin-integrated mass-change term is the next step (or a pbias prior mean
   from the RACMO/CARRA2 precip ratio). Also: yearly bare-ice AREA is too
   variable (std 79 k vs 30 k km2, 2019 453 k vs 226 k, r 0.52) although the
   SMB variability is right.

12. `observations.VelocitySpec(per_pixel_error=, sigma_floor=, sigma_rel=)`
   and `DhdtSpec(sigma_rel=)` (2026-09-21; defaults off = the Alaska
   behaviour). Velocity: per-component per-pixel stds from the mosaic's
   `vx_err` / `vy_err`, sigma_c = max(sigma_floor, err_c, sigma_rel |v|),
   the residual normalized by them before whitening and the Matérn member
   registered with unit sigma (`noise.sigma` a multiplier, the nugget in
   units of the per-pixel std — the DhdtObservation convention); the surge
   marginal takes the mean of the two stds per pixel; the RTO draw scales
   the unit member per pixel. dh/dt: sigma = max(sigma_floor, err, sigma_rel
   |dh/dt obs|) (times noise.sigma) in both branches. Greenland:
   `VelocitySpec(noise=MaternNoise(sigma=1, l=10 km, nugget=1),
   per_pixel_error=True, sigma_floor=5, sigma_rel=0.05)` and both DhdtSpecs
   `sigma_floor=0.1, sigma_rel=0.25` (effective interior floor 0.05 m/yr
   with noise.sigma 0.5). Why: the ITS_LIVE summary mosaic reports errors of
   0-3 m/yr in the interior (and 0.00 on most ablation-zone pixels, hence
   the floor), ATL15 0.2-0.6 cm/yr and ITS_LIVE dh 0.4-2 cm/yr; the scalar
   velocity sigma of 100 m/yr and the 0.5 m/yr dh/dt floor made both terms
   blind to the interior — at the converged v5 state the whitened rms above
   2000 m was 0.03 (vel) / 0.13 (dhdt) and at 1200-2000 m 0.14 / 0.60, i.e.
   "perfectly fit", while the model thickens 5-10 cm/yr there against the
   products' -2 (the northern accumulation excess) and flux divergence was
   free to absorb any SMB change (v5: D fell 1:1 with SMB). Under the new
   specs the same state reads 0.5 / 2.1 (vel, > 2000 / 1200-2000 m) and
   0.5 / 1.9 (ATL15) — the excess is now a 2-3 sigma signal over 400 k
   cells; the ablation zone reads 6-9 sigma (the outlets' retreat-timing
   misfit, 0.5-1 m/yr, which the old floor had hidden; Huber nu = 1 bounds
   its pull). Level-2 test 1985-2026: 13 s, 7.4 GB, gradients finite; the
   prior-state loss goes 2100 -> 10000, the v5 state 1560 -> 5160 (vel 440
   -> 2797, dhdt 299 -> 1128). Runs into `inverse_v6_<CLIMATE>`.
   **First v6 attempt from the prior state plunged the velocities and
   thickened the interior (2026-09-21).** Not an overshoot: at the prior
   state (beta_init 2.5) slow ice is 1.8x too fast and 435 k cells with the
   model > 2x too fast on sub-100 m/yr ice carry 75-80 % of the new
   velocity misfit, the 29 k too-slow outlet cells 3 % (under sigma 141 the
   slow ice was invisible at r ~ 0.08 and the 4-sigma outlets dominated, so
   the old runs fixed outlets first). With the Huber saturating at 3 sigma
   every cell pulls alike, the interior wins by count, and its "more
   friction" update leaks onto the outlets through the 8 km beta prior
   (one SGD step: fast ice 0.35 -> 0.22 of observed, slow 1.84 -> 1.53,
   J down; every data term's d/d log beta is negative on both classes).
   Reduced steps go the same way, so it is the objective's direction from
   that state, not the step size. Remedies: (1) WARM-START from the v5
   checkpoint (`inverse.py WARM_START_PATH`) — from there the level-2
   mini-run is stable and monotone (slow 1.25 -> 1.19, fast pinned at 0.39
   by the 4 km grid, which cannot resolve outlets at any state); (2)
   `sigma_rel_km` (both specs, 5 km): the relative term refers to the local
   MAXIMUM of the reference field over that radius, so a feature the model
   places a cell off (or smears on a coarse level) is scored against its
   own magnitude — cuts the smear share only from 75 to 60 %, so it is a
   correction, not the fix. Note the dh/dt Huber is now nu = 3 (user).
   Second finding: under `influence_cap` 0.3 (log transfer) the SMB block
   does not respond to the tightened dh/dt term at all (pbias in the north
   frozen at 1.028 over 6 iterations); without caps it moves (-> 1.004 in 5)
   but the SMB learning rates of 1.0 then oscillate (snow 134 -> 684). The
   caps were sized for the old gradient scale. Level-2 mini-runs from v5
   (8-10 iterations, driver mechanics): cap 1.0 on pbias alone or on all
   four: pbias still frozen (the cap is not the block); no caps + SMB lrs
   x0.2: 0.003 in 8; lr_z_pbias 5 + cap 1.0: monotone, pbias N -0.0006 per
   iteration, the northern 1200-2000 m thickening turns over (+7.2 -> +6.6
   cm/yr by it 9); lr_z_pbias 10 uncapped: oscillates (pbias to 0.02 at
   outlets, SE 0.79-0.99) — the outlets' dh/dt misfit (still 4-9 sigma) is
   what the cap keeps out of precipitation. The pbias block is
   ill-conditioned under the new terms: its interior signal is smooth and
   weak per cell (|dJ/dz_pbias| 7 against |z| 25), its outlet signal local
   and huge, and prior-preconditioned SGD with one lr cannot serve both;
   the log cap suppresses both. CONFIG NOW: `lr_z_pbias=5`, `influence_cap
   z_pbias 1.0`, and `inverse.py WARM_START_PATH` = the v5 level-0
   checkpoint. Expect pbias in the north to reach ~0.8 over a 350-iteration
   run if the rate holds; the real fix is a curvature-aware optimizer in
   whitened coordinates (Gauss-Newton / L-BFGS keep the prior geometry that
   Adam's per-coordinate normalization destroys).
   **`inverse_v6_hybrid` result (warm-started from v5, 2026-09-22;
   `analysis/output/basin_mb_v6/`)**: the interior velocities are now
   matched (slow ice 1.14 -> 1.01 of observed) and SMB came down to 366
   (Mankoff 337; pbias x0.99, precip 729 vs 766 in v5), but discharge fell
   further, 453 -> 411 (Mankoff 485), so MB WORSENED: -44 (v5 -73, Mankoff
   -148); windows -60 / -45 vs the products' -148 / -152. The 1200-2000 m
   thickening persists (+7..+26 cm/yr vs the products). The outlet deficit
   is MISSING ICE, not slow ice: where the model has ice in 2018 the fast
   cells run at 0.85-0.98 of observed (inverse and standalone agree to 1 %),
   but 9 % (300-1000 m/yr), 23 % (1000-3000) and 15 % (> 3000) of the
   observed fast cells are ICE-FREE in the model, 40-50 % of them observed
   floating; NE 309 of 333 fast cells (79N / Zachariae), NW 218/1078, CW
   158/1044 (flux proxy sum |u| H over > 300 m/yr: v1 0.81, v5 0.74, v6
   0.71). Cause, from the v6 replay's h0 / tf_anom frames: the 1990s
   thermal-forcing anomaly (+0.68 K at the NE fronts in 1995) times alpha_h
   (70 m/K) flips the NE margin from -45 to +20 m, the ice fraction there
   falls 69 -> 23 % and NEVER recovers when dTF returns to 0 (2005: h0 -26,
   37 %; 2018: 7 %) - a retreat hysteresis; NW / CW flip after 2005 (dTF
   +0.5-1.1, h0 +24..+84). So the mass-balance gap at the outlets is the
   calving / TF-phase problem, and the early NE collapse moves discharge
   OUT of the observed period. The log-beta prior is NOT the lever: a
   level-0 mini-run with sigma 1.0 (z rescaled to the same physical state;
   the first test reloaded the whitened z unscaled and started 3x off)
   raises the loss (2844 -> 3059 in 6 iterations, the effective beta step
   9x larger) without speeding grounded fast ice. Next: alpha_h / TF
   smoothing sweep with forward_standalone from the v6 state, scored on ice
   presence at the observed fast cells in 2018 and the regional MB.
   **alpha_h 70 -> 100 m/K (same v6 checkpoint, forward_standalone vs
   forward_standalone_v2; `analysis/output/basin_mb_v6_alphah/`)**: GrIS MB
   -45 -> -52 (Mankoff -148), D 411 -> 419; 2006-2025 MB -84 -> -106
   (Mankoff -218), D 399 -> 422 (Mankoff D+BMB 511); windows -60 / -45 ->
   -75 / -50 (products -148 / -152). All of it is CW (-13 -> -19) and NW
   (-20 -> -24); NE gets LESS negative (-1 -> +5): its outlets collapse
   earlier (ice at the observed fast cells 47 % in 1990 vs 78 %) and are
   gone before the windows. NO also loses more ice (63 vs 77 % in 2018) at
   no mass-balance gain. SE / CE / SW do not respond (93-98 % either way).
   The flux proxy stays 0.70-0.73. So alpha_h scales the post-2005 west
   response ~linearly (+7 Gt/yr for +30 m/K) but worsens the early,
   hysteretic north / northeast collapse: a larger coefficient cannot fix
   both, the phase of the 1990s anomaly at NE is the problem.

**Multigrid continuation vs resolution-dependent calving (2026-09-22).**
Measured, same v6 parameters integrated 1700-2026 at each level (the state
is NOT carried across levels - every level re-seeds from the observed
geometry, so only PARAMETERS transfer): 1990-2018 dM/dt -95 (1 km), -81
(2 km), -79 Gt/yr (4 km); observed outlet cells (>= 1000 m/yr) still
carrying ice in 2018: 48 / 51 / 55 %. Coarse levels do calve less - box
averaging shallows the fjord troughs (depth = -bed) and thickens the front
cell, both of which raise H_calve's margin - but the bias is ~16 Gt/yr
(~4 % of SMB) and its SIGN is opposite to the mass-balance deficit: a
coarse level retains MORE ice, so it asks the SMB block for MORE ablation,
not less. SMB itself is resolution-independent (computed on the fine grid,
restricted inside the checkpoint). Where the coarse levels DO damage the
fit is beta at the outlets: over the v6 run (warm-started from v5 L0), the
level-2 stage moved log beta at > 1000 m/yr cells by std 2.13 (a factor of
8) and only 47 % of that change survives the L1 + L0 stages (fast 300-1000:
46 %), while in the interior 84 % survives (std 0.92) - i.e. the two fine
stages spend their budget undoing level 2 at the margins. Cause: at 4 km
the model CANNOT represent outlets (fast-ice ratio pinned at 0.39 of
observed at every state tried), so the velocity residual there is
discretization error and the optimizer converts it into beta. Fix in the
likelihood, not the schedule: make `sigma_rel` (VelocitySpec, DhdtSpec)
LEVEL-DEPENDENT - e.g. 0.05 / 0.25 / 0.5 at L0 / L1 / L2 - so a coarse
level is told it cannot know a 2 km outlet channel; the observation already
sees `state.level` at loss time, so this is a small change. Freezing beta
near the margins at coarse levels is the blunter alternative.

13. `config.climatology_only` (2026-09-22; default False = the Alaska
   behaviour): hold the ATMOSPHERE at the reference climate for the whole
   run - the monthly climatology in the gridded inputs plus the calibrated
   tbias / pbias, with no yearly fields (the 9.3 GB record is not even
   loaded), no temperature-anomaly index (`base_anomaly` and `alpha_t2m`
   forced to 0) and no precipitation anomaly, AND the ocean at its TF
   climatology (`OceanForcing(freeze_anomaly=True)` -> `anomaly()` returns
   0, so q / h0 are the time-invariant baseline + clim_* (TF_clim -
   tf_crit); the climatology is still read, only the interannual departure
   is dropped) - the thermal forcing moves mass through the margins on the
   same order as the atmosphere does through SMB, so a spin-up adequacy
   test has to hold both. Gated in
   `problem.simulate_physical` (one ternary per forcing argument) and in
   `forward_standalone.setup` (`alpha_t2m_eff` / `base_anomaly_eff`, yearly
   loader skipped) and in both `OceanForcing.from_file` call sites, so an
   inversion, a replay and a diagnostic run all see the same forcing;
   `GlacierProblem` and `OceanForcing.describe()` both announce it.
   Purpose: with the
   climate fixed, the model's dh/dt IS its relaxation from the initial
   geometry, which separates the spin-up transient from the forced response
   in the dh/dt misfit (measured drift at the v6 state: +110 Gt/yr over
   1900-1990, against -95 Gt/yr over 1990-2018 - the same order as the
   signal). The intended remedy for a large drift is a LONGER SPIN-UP, not
   a shorter one or a later start: the initial condition is unknown and an
   observed geometry is not in the space of admissible model states, so it
   injects a numerical transient; a model that cannot hindcast 2000 from
   1900 freely has no claim on 2100 from 2000.

14. `config.interannual_sigma` / `interannual_nodes` (2026-09-22; None =
   the Alaska behaviour): Gauss-Hermite quadrature over the interannual
   variance the SCALAR index terms do not carry, in
   `forward._expand_interannual`. smb is concave in temperature, so a step
   evaluated at its MEAN anomaly is not its mean smb; the gap is
   ~ curvature sigma^2 / 2, and Greenland's integrated curvature is
   ~-160 Gt/yr/K^2. **Validated independently**: with sigma_true 1.055 K
   (measured ice-mean JJA interannual std of the hybrid yearly file; annual
   1.023) and the index's effective 0.6 K, 0.5 curv (1.055^2 - 0.6^2) =
   -60 Gt/yr against the -62 the OCX experiment measured between the index
   and the resolved yearly fields at identical MEAN forcing. A deep step
   resolves none of it, so there the correction is 0.5 curv 1.055^2 =
   -89 Gt/yr (the code returns -88.2 on a quadratic). Mechanics: each scalar
   term `(a, w)` becomes `(a + s x_i, w w_i)` over the probabilists'
   Gauss-Hermite rule (3 nodes at 0, +-sqrt(3), weights 2/3, 1/6, 1/6 --
   exact through 5th order), with `s^2 = sigma^2 - <the variance the terms
   already carry>`. That residual is what makes ONE setting right for every
   epoch: under `mean_anomaly` a single term carries no spread and takes the
   full sigma, under `annual` the spread across the step's index years counts
   against it, and a step already spreading wider than sigma is returned
   untouched. `sigma` is in the units the terms carry, i.e. AFTER alpha_t2m.
   YearField terms are never touched, so this acts on exactly the
   pre-reanalysis spin-up. Like `temp_dev` one level up it is a FIXED
   deterministic quadrature, not a random draw, so the checkpointed backward
   reproduces the forward. NOT gated on `climatology_only` (the variance
   belongs to the climate, not the anomaly: a drift diagnostic without it
   would sit ~90 Gt/yr above the calibration and report the offset as
   drift). `forward_standalone.py` applies the identical nodes. Cost is
   `interannual_nodes` glare calls per index term and nothing on the
   dynamics: for the Greenland schedule (dt 50 to 1850, 10 yr to 1990, then
   the record) ~49 index steps, so ~100 extra calls per run. Greenland sets
   `interannual_sigma=1.05`, `interannual_nodes=3`. Verified offline
   (weights, mean and total std exact; record terms preserved; both off
   switches; quadratic recovery exact); NOT yet run on the GPU.

15. `config.OceanForcingConfig.pin_front` (2026-09-24; None = the TF-driven
   margins): hold the calving front at an observed ice mask for the whole
   run. `ocean.PinnedFront` has OceanForcing's surface (`margins`,
   `anomaly`, `describe`, `ok`) and returns time-invariant h0 =
   `pin_h0_inside` (-1000 m) on masked cells and `pin_h0_outside` (+250 m)
   elsewhere, q = calving_q, dTF = 0, through the SAME checkpointed h0 field
   forward.simulate already sets per step -- no glide change, no adjoint
   change, TF file not read. Under the monotone law h0 << 0 never calves
   grounded ice and leaves floating ice to `calving_H_c` alone (2015 tongues
   thicker than 100 m persist, thinner fringes still go); +250 m removes
   ice within 250 m of flotation at the calving timescale (a soft pin:
   marine re-advance is cut, thick grounded land ice outside the mask is
   SMB's problem). Built from the cropped gridded inputs at both sites
   (`problem.py`, `forward_standalone.load_ocean_forcing`);
   `forward_projection` REFUSES a pin. WHY: the velocity term sees the
   misfit at an outlet the model has lost but d misfit / d beta ~ 0 there
   (no ice for beta to act on), so v7 and v8, cold-started at different
   sigmoid_c, converged to the same parameters and both lost NE at the
   gates. Pinning makes beta answerable for observed speeds over observed
   geometry; the calving parameters are then tuned afterwards, forward-only,
   to reproduce those fronts (front position the objective, gate flux the
   validation). The mask carries ONE epoch (rgi_mask: nominal 2015,
   2007-2019), so the spin-up sits at the 2015 extent -- a flat hold, but
   pre-industrial fronts were more extended, so it errs conservative -- and
   the observation window imposes no retreat, which the 2018 velocity and
   surface terms will feel at Jakobshavn (11 km) and Zachariae. A
   time-varying mask (TermPicks, ~280 glaciers from 1972) is the upgrade;
   `from_gridded` takes any boolean variable. Expect beta to run to
   `beta_max` where 1 km cannot resolve an outlet. Config off; enable with
   `pin_front="rgi_mask"` + a warm start.

16. `config.OceanForcingConfig.rho_filename` (2026-09-24; None = the
   behaviour above): a per-cell CRITICAL ANOMALY rho(x) in kelvin from
   `model_inputs/calving_rho.nc` (built by `preprocessing/make_calving_rho.py`)
   entering BOTH anomaly responses,
   `h0 = calving_h0 + clim_h (TF_clim - tf_crit) + alpha_h (dTF - rho)` and
   `q = calving_q + clim_q (...) + alpha_q (dTF - rho)`, plus the file's
   `h0_fixed` (m) which REPLACES h0 where finite (`OceanForcing.from_file
   (rho_path=)`, default next to the TF file; `forward_projection` passes the
   domain's model_inputs; the pin ignores it; a run without the file is
   bit-identical to before). Why a field: under the monotone law with q = 0
   a floating cell has F = -h0, so a front's whole TF response is the one
   ratio -h0_base/alpha_h, and the calving screen showed the observed onsets
   need ratios that are NOT a function of TF_clim (Petermann and Zachariae
   share 2.27 degC and need 0.3 and 1.2 K): no (tf_crit, clim_h, alpha_h)
   times more than 8 of 23 documented fronts. rho is each front's distance
   to threshold in the reference climate; the flip year is unchanged by the
   alphas (for floating ice F = -(dTF - rho)(alpha_q H_f + alpha_h)/(1 - q),
   same sign), which only set the reach into grounded ice (HAB < q H + h0):
   rho = TIMING, alpha_q / alpha_h = EXTENT. USE WITH `clim_h = 0`,
   `calving_h0 = 0` (the builder derives rho for a static margin of
   -alpha rho alone and warns otherwise).
   **THE FIRST FIELD RAN AWAY (user, 2026-09-24): every glacier advanced at
   once and no anomaly brought it back.** With rho > 0 everywhere, h0 < 0
   everywhere, and in this law a negative margin is not "hold the front" but
   "floating ice is admissible": every fjord filled during the spin-up (level
   2 control: 25-38 k km2 of marine ice beyond the 2015 extent from the first
   step, vs 4-6 k bounded), the advanced ice grounded, and a later positive
   margin of tens of metres cannot remove grounded ice. The static screen
   tests the observed geometry and the first advance cell, not 2000 years of
   dynamics; a dynamic test at a coarse level was the step skipped. HENCE THE
   BOUND (`--bound-mask rgi_mask`, default; `--outside-h0 250`): the field is
   the front pin with a finite, time-varying inside depth -- outside the
   observed extent h0_fixed = +250 m (the pin's value) cuts any marine advance
   at the calving timescale, inside it the margin is alpha (dTF - rho); ice
   persists inside in the climatology, goes when dTF > rho, and re-advances
   up to the mask when the anomaly drops (hysteresis only through grounding).
   With the 2015 mask it cannot represent a pre-retreat extension (the pin's
   limitation; a maximal-extent mask from TermPicks would), and under the
   bound a grounded front's retreat is the loss of its 2015 terminus cells,
   so the `term` set is the diagnostic for rho and the verification
   (`calving_screen.py --rho` honours h0_fixed and switches to it).
   **Builder**: per documented front (`analysis/front_onsets.csv`,
   `--min-confidence`) from the diagnostic set's stepped anomaly (the config's
   dt_schedule, `--filter none|box:N|ema:TAU`): stable -> `--safety` (1.25) x
   the record's max dTF, retreat -> the midpoint of (max dTF before the onset
   window, max through it], empty window -> 1.05 x the pre-window peak
   (missed rather than early); cells within `--assign-km` (30) of a
   documented front's gates and its floating component take its rho, every
   other cell the REGIONAL median (CE 0.73, CW 1.55, NE 1.63, NO 0.23, NW
   0.64, SE 0.99, SW 1.95 K); full domain grid; `calving_rho.csv` beside it.
   Built 2026-09-24 (filter none): 79N 2.03, Zachariae 1.24, Petermann 0.33,
   Humboldt 0.19, Jakobshavn 0.69, Helheim 0.46, Kanger 0.46, Store 1.76 K;
   screen: 16 / 18 of 23 on time at alpha_q 0 / 0.3 (alpha_h 50), none early
   or spurious. The unbounded field is kept as `calving_rho_unbounded.nc`.
   **Dynamic tests, 1 km from 1700 (forward_standalone, scratch runs)**:
   bounded, alpha_h 50, alpha_q 0: no runaway (marine ice beyond the 2015
   extent 3-5 k km2 throughout, an apron of ice in transit; total ice area
   flat; coverage of the observed >= 300 m/yr cells 0.997 to 2010, 0.97 by
   2026), the tongues flip on schedule (Petermann 2011-20, Zachariae 2014-20,
   Humboldt 2010-20) and Jakobshavn's terminus collapses 1998-2000 then
   re-advances after 2015, but the fronts the SPIN-UP GROUNDED do not move:
   with nothing calving inside the mask the seeded near-flotation termini
   thicken (Helheim 649 -> 810 m, HAB -17 -> +57; Kanger 491 -> 641, +96;
   Store 185 -> 424, +130; tongues ~2x, 79N 429 -> 700 m -- no sub-shelf melt
   in the model), and alpha_h x exceedance (10-25 m) never reaches them.
   alpha_q 0.3/K (the thickness-scaled response, 0.3 K of exceedance = 70 m
   at Helheim, 15 m at a 170 m front): Helheim retreats 2000 (816 -> 393 m),
   refills by 2005 and goes again 2015 (observed 2001-05 + readvance 2006),
   Jakobshavn 1998-2010, Kanger 2015 (late: the product is cold at its cells
   in 2005), Tracy 2005-10, Alison 2010, Upernavik / Kong Oscar partial;
   Store (HAB +285), Rink, Heilprin, Daugaard-Jensen, 79N, Ryder hold; the
   totals are unchanged. Judge these runs on THICKNESS (a cell with 2 m of ice
   in transit is still "ice" by the mask; the first coverage metric hid every
   retreat). Caveats: (1) coarse levels cannot judge the field -- box
   averaging a +250 / -20 m discontinuity puts a positive margin on every
   straddling front cell (Helheim +37, Store +39, Heilprin +73 m at 4 km
   already in 1850), which the pin's -1000 never suffers; use it at level 0
   (`max_level=0`) or give the restriction a mask-aware rule; (2) the bound's
   sink removes only ice within 250 m of flotation, so shallow fjord heads
   pile up (KNS 150 -> 930 m); (3) Petermann loses its whole tongue (observed
   1/3); (4) it is a calibration, one number per documented front from one
   onset. Config for the next run: `clim_h=0, alpha_h=50, alpha_q=0.3,
   rho_filename='calving_rho.nc'`, `calving_h0=0`, then sweep alpha_q on the
   forward standalone from the pinned beta against the gate fluxes and ATL15.
   (The glide / config density-ratio mismatch is under the calving-screen
   bullet below.)

17. `config.OceanForcingConfig.pin_front_filename` (2026-09-25; None = the
   behaviour above): a TIME-VARYING front pin. `PinnedFront.from_file` reads
   `model_inputs/<file>` with `front_mask(time, y, x)` per calendar year
   (uint8, cropped like GLIDE_inputs, ~0.7 s to load) and `margins(t0, t1)`
   pins each step to the mask of its END year (`mask_at`; the first mask
   held before the record, the last after; one cached h0 per year), the same
   `pin_h0_inside` / `pin_h0_outside` margins as the static pin, dTF = 0,
   no TF file. Takes precedence over `pin_front`; wired in `problem.py`,
   `forward_standalone.load_ocean_forcing` and `forward_projection`'s
   refusal. Purpose: the static pin holds every front at the one inventory
   epoch; this holds them at the OBSERVED terminus over the historical
   period, so the model's fluxes and dh/dt can be judged with the front
   position taken as known -- what the calving law has to reproduce.
   **`preprocessing/make_front_mask.py`** builds it from TermPicks v2
   (`common_data/area/termpicks/`, Goliber et al. 2022: 279 glaciers,
   39 060 traces, mostly 1972-2020, integer GlacierIDs, no names,
   EPSG:3413 = the model grid): each glacier is attached to the calving
   basin whose TERMINUS cells are nearest its traces (`calving_basins.nc`)
   and gets a PRIVATE along-fjord coordinate -- a windowed BFS from that
   terminus over the water (+km) and along its own basin's sub-sea-level
   trough (-km) -- independent of which basin the flux-primacy flood gave
   the water to (a small glacier's fjord belongs to its big neighbour there;
   measuring its traces from the neighbour's terminus put phantom fronts
   35 km out: the first two attempts). A cell within `--max-dist-km` (15) of
   a trace is seaward when its coordinate exceeds the median coordinate at
   the trace's vertices (no tangents or orientation; the plain side-of-line
   test with a correlation-fixed sign was wrong at Jakobshavn 2005). The
   year's mask = the 2015 `rgi_mask` + landward cells (water allowed to
   hold ice where the front was further out) - seaward cells (no ice beyond
   the observed front); only reach cells are touched, so lateral ice on
   land stays as the inventory has it; one trace per glacier-year (nearest
   mid-year; `--min-quality`), other years the nearest observed year, first
   / last held; glaciers without coverage keep the 2015 mask. 55 s.
   `front_mask.csv` (per glacier-year: the trace, `front_km`, cells added /
   cut) and `front_mask_glaciers.csv` (attachment; 11 of 279 more than 5 km
   from any terminus). Histories, km from the inventory front: Jakobshavn
   +12.5 (1975) +13 (1985) +11 (2000) +1 (2005) -2 (2010) -3 (2015);
   Helheim +5 (1985) +4 (2000) -1 (2005) +1 (2010) 0 (2015) -2 (2019);
   Kanger +5 (1985) +4 (2000) +2 (2005) +3 (2010) +2 (2015); Petermann +2
   (2000) +1 (2010) -15 (2015: the 2010 / 2012 calvings, 267 inventory
   cells cut); Zachariae -11 at 2015 (493 cells: BedMachine's mask still
   carries the collapsed tongue). Cells differing from the inventory: 2 800
   (1972) -> 1 300 (2012) -> 2 100 (2022, the post-inventory cuts). Known
   oddities: Unnamed Deception N and Kjer at +8 / +9 km in 2015 (trace vs
   inventory disagreement, 113 / 109 cells), 79N's front jumps -3 (1985)
   -> +10 (2000) between authors. Config: `pin_front_filename=
   "front_mask.nc"` with `enabled=True` (pin_front may stay None).
   **TermPicks-pinned composite (user, 2026-09-25 14:55, tau 1;
   `analysis/output/basin_mb_v9_termpicks`, against the 2015 pin and the
   tau-1 field run).** With the front position taken as known: 1986-2027
   GrIS MB -111 (Mankoff -148), SMB 327 (337), gate D 324 (2015 pin 309,
   tau-1 field 270; reference 391), residual D 438 (462); NE 29.5 (obs
   28.8), NO 24 (25), CE 55 (63), SE 77 (110), SW 21 (15), CW 51 (48), NW
   67 (100). At 2018 the gate flux is 346 / 391 = 0.89, the best snapshot:
   Jakobshavn 1.17 (speed 1.07), Kanger 0.96, Helheim 0.94, Store 1.07,
   Rink 0.99, Petermann 1.10, Daugaard-Jensen 1.21; Zachariae 1.61 (speed
   2.14, thickness 0.78: the imposed collapse), 79N 0.55 (speed 0.49, the
   tongue kept); the deficits are NW 0.70 and SE 0.73 at normal thickness
   with speed 0.76 / 0.70 -- SLOW, the same ceiling the sweep showed at any
   c. THE RESPONSE TO AN IMPOSED RETREAT (the user's observation: velocity
   rose less than expected): Jakobshavn's gate flux 12.8 (2003) -> 18.5
   (2006), speed ratio 0.71 -> 1.07 -> 1.16, +75 % (observed ~ +100 %);
   Helheim 21 -> 24.5 in 2006, +30 % (observed ~ +100 %); Kanger 20 -> 24,
   +24 % (observed ~ +80 %); Zachariae x2 with the 2010-15 cut; Upernavik N
   NOTHING (4.3 -> 4.7 Gt/yr, speed 0.53 -> 0.58) although TermPicks
   imposes its 2005-10 retreat (+4 km -> 0). So the retreat -> speed-up
   mechanism is present but weak, and weakest at the narrow NW / SE
   outlets, where the pinned beta leaves the 2018 speed at 0.6-0.8 even
   with the correct front. Claude's first reading -- a 1 km resolution
   ceiling (3-4-cell trunks) -- was asserted from cell counts, NOT tested;
   the user's alternatives, being tested 2026-09-25: (1) an inconsistency
   between the flux the pseudo-steady pinned margin can support and the
   observed one (the trunks are 1.2-1.4x too thick before the retreats, so
   the response starts from the wrong state), and (2) excessive viscosity
   damping the velocity variations over narrow outlets -- A_glen 1e-17 ->
   1e-16 (slightly damaged temperate ice, reasonable for marine outlets),
   and the inversion rerun with the TermPicks margin (`pin_front_filename`)
   instead of the static 2015 pin so beta is calibrated against the
   observed front history. The pre-retreat speed ratios (0.57 at
   Jakobshavn in 1990, against the 2018 mosaic) are as expected. Regional
   time series: CW 45 -> 52-58 after 2005 (Mankoff 65 -> 90), NE 25 -> 40
   in 2012-14, GrIS 310 (1990) -> 345 (2018), +11 % against Mankoff's +15 %.
   **`inverse_v10` (user, 2026-09-26): the INVERSION rerun with the
   TermPicks margin (`pin_front_filename`), the longer schedule (100 / 100 /
   200), the smoothed DEM, tau 1, calving_h0 10, A_glen 1e-17 -- the 1e-16
   test stays open: with the smoothed DEM it no longer produces the 30 km/yr
   cliffs, but isolated thin floating patches go near-singular in Vanka
   under the weaker viscous coupling (user). Every loss term ended lower
   than in any run before. Replay `inverse_v10/forward_standalone`,
   `analysis/output/basin_mb_v10/` (with v9_termpicks and v9_pinned2015 as
   CSV), per-glacier gates in the scratch `gate_v10.py`. The inversion
   and the replay on disk (09:50) are both at tau = 1.0; the 0.33 replays
   of the same morning gave the same numbers within 1 %.** 1986-2027: GrIS
   MB -89 (v9tp -111, Mankoff -148), SMB 347 (327; 337), gate D 333 (324;
   the old convention -- 419 at 2018 under the chain integral, 0.85 of
   Mankoff's 492), residual D 437 (438; this IS the model's discharge:
   the flux out of the polygons plus the calving sink inside them, vs
   Mankoff's D + BMB 485), SMB - gate D = +14 (v9tp +3, Mankoff -148).
   The MB deficit of ~60 Gt/yr is SMB +10 too high and discharge ~50 too
   low, the latter in SE (0.73 of Mankoff at the gates in 2018), NW
   (0.77), NE (0.84) and CE (0.88) against CW 1.08 and SW 1.07. Note that
   `basin_mass_balance.py`'s cumulative panels are the region's
   INTEGRATED VOLUME (M_r from H, offset at ref_year) against the
   cumulative of Mankoff's MB; the gates enter only the clean rates plot
   (Dg) and the MBg column.
   Regional gate D v10 / v9tp / 1 km reference: NO 22.5 / 24.1 / 24.6, NE
   24.0 / 29.5 / 28.5, CE 59.4 / 55.0 / 56.4, SE 85.1 / 76.9 / 97.9, SW 18.3
   / 21.0 / 13.0, CW 50.4 / 50.7 / 44.7, NW 73.6 / 66.8 / 93.9. NB THE
   REFERENCE MOVED: the observed fields through the gates use
   `thickness_obs`, now the smoothed one, so it is 359 instead of 391
   (see the DEM bullet); v10 is read against 359, v9 against 391. 2018
   snapshot: 348 = 0.97 of the smoothed reference, 0.89 of the unsmoothed
   (v9tp 346, 0.89) -- the same total, a different composition: NE 25
   (v9tp 36: Zachariae 16.6 vs 27.1, now AT the observed 16.7, speed 1.47 x
   thickness 0.67 = the imposed collapse), NO 24 (29), SE 89 (80), NW 79
   (70), CE 62 (59). Per glacier, flux ratio (gate speed ratio): Kanger 1.09
   (0.99; v9tp 0.86), Helheim 1.05 (0.93), Jakobshavn 1.13 (1.18), Petermann
   1.06, Daugaard-Jensen 1.26, Store 1.21, Rink 1.11, Koge Bugt C 1.00
   (0.78), Kong Oscar 1.00 (0.91; was 0.79), Alison 0.92 (0.94; was 0.70),
   Upernavik C 0.88, Ikertivaq M 0.78, Upernavik N 0.79 (0.74; was 0.58),
   Hayes 0.77, Humboldt 0.74, Nordenskiold 0.68, 79N 0.62 (0.54, tongue
   kept), Anorituup 0.41, Kjer 0.39 (unchanged); regions NO 0.99, NE 0.87,
   CE 1.10, SE 0.91 (was 0.73), SW 1.37, CW 1.14, NW 0.84 (was 0.70). Gate
   thickness 1.10-1.15 of the smoothed observed at Kanger / Helheim, 1.25 at
   Daugaard-Jensen: the spin-up still thickens the trunks. Fast-cell speed
   at 2018, median model / observed on covered cells by observed class,
   v10 (v9tp): 300-1000 m/yr 0.92 (0.84), 1000-3000 0.88 (0.77), > 3000
   0.53 (0.47); NW 0.82 / 0.72 / 0.33, SE 0.84 / 0.83 / 0.57, CE 0.91 /
   0.91 / 0.56, CW 0.95 / 1.03 / 0.61 -- the narrow NW / SE outlets gained
   ~0.1, the last few km of every trunk still run at half the observed
   speed. THE RETREAT RESPONSE IS WEAKER THAN v9tp's: Jakobshavn 14.4 (2003)
   -> 17.2 (2006), +19 % (v9tp 12.8 -> 18.5, +45 %; observed ~ +100 %) --
   v10 is already 13.1 in 1990 against v9tp's 10.5, i.e. too fast BEFORE
   the retreat; Kanger 21.6 -> 25.0, +16 %; Zachariae 12.7 (2003) -> 22.2
   (2012), +75 % (v9tp x2), back to 16.6 by 2018; Upernavik N 5.1 -> 5.8,
   +14 %; NW 72 -> 79 (+9 %; Mankoff 90 -> 115), CW 48 -> 52 (+8 %; 65 ->
   90), SE 83 -> 89 (+7 %), GrIS 325 -> 348 (+7 %; Mankoff 437 -> 492,
   +13 %). SMB (`smb_diagnostics.py`, 2000-2019): the best fit yet --
   P(snow) within 0.01-0.03 of the label in EVERY band (v9tp 0.05-0.11 off
   at 1200-1800 m), Brier 0.020 (0.026), bare ice 197 k vs 181 k km2 (v9tp
   220 k), interannual std 122 vs Mankoff 116, slope 0.98, r 0.93, trend
   -60 vs -50 Gt/yr/decade; regional SMB model / Mankoff NO 16 / -4, NE 17
   / 8, CE 59 / 78, SE 109 / 131, SW 10 / 12, CW 68 / 44, NW 68 / 53 (the
   CW-NW-high / SE-CE-low dipole persists, SW is right now). The +20 Gt/yr
   over v9tp sits at 1400-2000 m (+20) and above 2000 m (+18) against -11
   in the ablation zone (0-800 m -84 vs -73); tbias -0.47 K (v9tp -0.34),
   pbias x1.00, precip after pbias 717 (715). So the remaining budget gap
   is discharge, not SMB: 2006-2025 MB -168 vs Mankoff -243, and the
   model's gate flux is 26 below a 1 km reference that itself recovers only
   73 % of Mankoff's 491 -- the fastest trunk cells and the damped
   retreat-to-speed-up response are where it is. **BOTH READINGS WERE
   REVISED THE SAME DAY (user: "why does the reference depart from Mankoff;
   what carries dM/dt vs SMB - D?"), see the two bullets below: the 359 /
   391 reference was a gate-integration convention, not resolution, and
   the dM/dt-vs-budget gap was that convention plus a tau read from the
   wrong place (the model conserves mass).**
   **Gate integral convention (2026-09-26, scratch `gate_fullres.py`,
   `gate_chain_model.py`; `analysis/output/gates/gate_fullres_*.csv`,
   `gate_chain_model_2018.csv`; Mankoff's per-gate `gate.nc` fetched from the
   GEUS dataverse into `common_data/dhdt/mankoff/dataverse_files/`).** The
   NATIVE products through Mankoff's gates -- ITS_LIVE 120 m (2018
   intercept) x BedMachine v6 150 m, bilinear at the 5890 recovered 200 m
   gate pixels -- give 478.5 Gt/yr against Mankoff's 494.5 (2017-19) with
   the RASTER-CHAIN convention (nearest-neighbour walk through the 8-connected
   pixel chain, per-segment normals and true segment lengths, 1330 km of
   gate), 495.5 with a PCA-ordered polyline (zigzags across Jakobshavn's
   40 km, 35 km-wide gate: 144 km of line, +20 Gt/yr there) and 476.6 with
   |v| x 200 m per pixel; per glacier by Mouginot name the chain ratio is
   median 0.98 (10-90 %: 0.85-1.13; Jakobshavn 1.12, Helheim 1.00, Kanger
   1.01, Zachariae 0.98, 79N 0.98, Petermann 0.95), regions 0.91 (CE) to
   1.04 (CW). The 1 km MODEL INPUTS with the same chain integral give 467
   (0.95 of Mankoff; velocity-only swap 0.98, thickness-only 0.98) with the
   unsmoothed DEM and 432 (0.87) with the smoothed one. So resolution costs
   5 %, the DEM smoothing 7.5 %, and the 391 / 359 of
   `basin_mass_balance.py --gates` (and `gate_v10.py`) is the CONVENTION:
   200 m per pixel (the PCA extent equals it, 1163 km vs the true 1330) x
   ONE PCA normal per gate = 403 on the same fields (0.82), i.e. -14 %
   length, -4 % projection at curved gates. `basin_mass_balance.py` still
   carries the old convention -- its Dg_* and obs_Dgate_1km columns, and
   every gate number in this file dated before 2026-09-26, are ~17 % low in
   absolute terms; model / reference RATIOS are unaffected. Under the chain
   convention at 2018: v10 420.6 (0.85 of Mankoff, 0.97 of its own smoothed
   1 km reference), v9tp 417.0 (0.84; 0.89 of the unsmoothed); regions v10 /
   Mankoff NO 0.92, NE 0.77, CE 0.90, SE 0.74, SW 1.09, CW 1.06, NW 0.78;
   v10 series 390 (1990) -> 391 (2000) -> 414 (2006) -> 420 (2012) -> 421
   (2018) against Mankoff 437 -> 432 -> 470 -> 481 -> 492. Per glacier v10 /
   Mankoff: Jakobshavn 1.14, Helheim 1.00, Kanger 1.03, Zachariae 0.88, 79N
   0.58, Petermann 0.98, Koge Bugt C 0.97, Rink 0.95, Store 1.05,
   Daugaard-Jensen 1.07, Ikertivaq M 0.68, Hayes 0.78, Kong Oscar 0.84,
   Nordenskiold 0.52, Koge Bugt S 0.55, Anorituup 0.35, Alison 1.08.
   **Mass budget of the pinned runs -- CLOSED (2026-09-26; scratch
   `exact_v10b.py`, `reach_budget.py`, `closure_unpinned.py`).** A day of
   "the model loses ~300 Gt/yr at the pinned fronts" was Claude's error:
   every budget script hard-coded tau = 1.0 (the config value read early
   that morning), while the replays analysed were run with
   `calving_timescale = 0.33` (the working-tree config; forward_soln.nc
   records it). With the run's own tau the exact thickness stencil
   (flux.cu upwind form on the raw facet velocities of the
   `STATE_SAVE_TIMES` state files, residuals.cu orientation: +v = toward
   decreasing row, vertical upwind term - |v| (H_t - H_b) / 2) balances
   on every cell class: sum over active cells -0.4 Gt/yr, rms 0.5 m/yr,
   flagged cells -0.3 (rms 0.6), and the whole-sheet budget over all cells
   dM/dt +80.2 = SMB 522.8 - sink 432.6 - flux into constrained cells 9.4
   closes to -0.6 Gt/yr (2018 -> 2019); the replay redone at tau = 1.0
   (09:50, the inversion's own value) closes the same way, -0.5 Gt/yr
   (dM/dt +71.5 = SMB 569.5 - sink 491.3 - 6.2 into constrained cells;
   51 706 flagged cells holding 491 Gt = 1.0 yr of sink, against 5 036
   cells and 143 Gt at 0.33), and its basin, gate and SMB numbers are
   identical to the 0.33 replay's within 1 %: tau sets how much ice sits
   in the aprons, not the fluxes or the mass balance, because the front
   position is the pin's. The sink is exactly (1 - psi) H /
   tau with the stored psi (the calving_F transcription reproduces it),
   the apron mass (1 - psi) H = 143 Gt is 0.33 yr of sink, and the reach
   budgets seaward of the gates close: Jakobshavn gate 40.7 in, sink 33.6,
   dM/dt +5.8, closure +0.8; Helheim 26.8 / 27.3 / +1.1 / -1.8; Kanger
   26.9 / 28.0 / -1.5 / +0.2; Store 9.7 / 9.0 / 0 / +0.3; Zachariae 15.4 /
   8.9 / +9.4 / -2.8 (the +/-2 are lateral inflow and the trough-only
   reach). glide conserves mass (the user's smb = 0 / calving-off test:
   1e-5 relative), the solver's |r_H| of 3.15 m/yr after one V-cycle is
   genuine, the unpinned sweep run closes with ITS tau of 1.0, and the
   10x tighter tolerance changed nothing because nothing was wrong. Also
   found on the way and still true: the facet velocities are smooth and
   equal to their cell averages (no staggered-grid mode), the trunks are
   not checkerboarded, and Mankoff's gates are reproduced by the native
   products (previous bullet). LESSONS: read every run parameter from
   the run's own forward_soln.nc attrs, never from a config read earlier
   in the session (the user edits it between runs); a closure that fails
   by a nearly constant amount proportional to a stored field ((1 - psi)
   H here, 3.03x) is a coefficient, not physics; and the first thing to
   check against a solver residual is one's own transcription (the
   vertical upwind sign was also wrong in the first stencil and hid
   behind the tau error). THE RESIDUAL D IS A DISCHARGE: SMB - dM/dt over
   the polygons (437 Gt/yr) = the flux across the polygon boundaries +
   the calving sink inside them, and the model's total front removal at
   2018 (sink 433 + 9 into constrained cells) matches its chain-convention
   gate flux (421) plus the strip's SMB and thinning. The user's original
   question -- why dM/dt-based MB (-90) sits below SMB - D_gate -- was the
   gate CONVENTION (D_gate 333 was 17 % low; ~400 under the chain
   integral gives SMB - D_gate ~ -55) plus SMB and thinning below the
   gates. Which tau the v10 INVERSION itself ran with (config.py is
   uncommitted; HEAD says the older value) is the user's to confirm: a
   replay at 0.33 of a checkpoint inverted at 1.0 is not a faithful
   replay.
   **Why the NW outlets are slow (2026-09-26; scratch `nw_diag.py`,
   v10 tau-1 replay, 2018, cells with observed speed > 1000 m/yr, model
   ice, BedMachine H > 10 m).** It is speed, not thickness: model / obs
   surface speed NW 0.75 (SE 0.83, CE 0.94, CW 1.02), model / BedMachine
   thickness at the gates 1.0-1.08. NW's fast reaches sit ~150 m above
   buoyancy (BedMachine 166) against 414-484 m in SE / CW / CE, so with
   the drag law beta xi^p, p = 1, xi = HAB / H the flotation fraction is
   0.28 at the median (SE 1.00, CW 0.52, CE 0.83), 20 % of the cells have
   xi < 0.05 and 38 % xi < 0.2 (SE 2 / 5 %), and basal drag carries only
   20 % of the driving stress (SE 36 %, CW 34 %, CE 29 %; slow ice 88 %).
   The inversion therefore has almost no purchase on NW through beta
   (d tau_b / d beta = xi |u|^m is 4x weaker than in SE) and left it at
   the prior: beta 1.9 median on the fast cells, 2.4 at the gates (Hayes
   4.6, Kakivfaat 3.7, Upernavik C 3.3; prior mean 2.5) while SE's went
   to 1.2 and CW's to 1.6. With effective drag already lower than
   anywhere else and the ice still 25 % slow, the resistance is MEMBRANE:
   lateral drag against the fjord walls of 3-6-cell channels at a uniform
   A_glen 1e-17 with no thermal or damage softening of the shear margins
   -- the same ceiling SE reaches (0.8) once its beta is lowered; CW's
   wide troughs do not hit it. Two aggravations: the model thins the
   last kilometres of the NW trunks 30-76 m below BedMachine (Hayes,
   Steenstrup, Kakivfaat, Alison, Tracy), which triples the at-flotation
   fraction (18 vs 7 %) and lowers the driving stress 12 % below
   BedMachine's (21 vs 24; SE / CW / CE within 7 %); and the kernel /
   config density-ratio mismatch (0.917 vs 0.892) is 15-20 m of HAB at
   NW depths, second order but not nothing at xi 0.28. Levers that exist
   there: the viscosity (an enhancement factor or an invertible A field
   confined to grounded fast ice, which also sidesteps the thin-floating-
   patch conditioning that made A 1e-16 sketchy), the DEM smoothing at
   the gates (0.87 -> 0.94 of Mankoff for the reference), and the density
   ratio; beta, p or a xi floor cannot speed NW up. Mankoff's own BMB
   (MB_region.nc, Karlsson-based) is 23.1 +- 5.5 Gt/yr 1986-2025 (NE 2.8,
   CE 2.8, SE 4.3, SW 4.4, CW 4.0, NW 3.4, NO 1.4), not 27; the model has
   no basal melt, so it is a genuine ~23 of the ~60 Gt/yr MB gap.
   **`inverse_v11` (user, 2026-09-27): soft ice, A_glen 5e-16 per the user
   (NOT recorded in forward_soln.nc; config.py read 5e-17 after the run),
   glide's base clip at d = 100 m (`TAU_D_CLIP_SCALE`; d = 10 speckled and
   killed convergence a few steps into the inversion), tau 1, dt 25
   spin-up; not fully converged and, in the user's judgement, not usable
   as is (the interior deforms too much, other fields compensate: the
   case for thermomechanical coupling). Replay
   `inverse_v11/forward_standalone` (the directory was `inverse_init_lv`),
   `analysis/output/basin_mb_v11/`, `analysis/output/scratch/
   gate_chain_compare.py`.** Gate flux at 2018, chain integral, v10 -> v11
   as a fraction of Mankoff: GrIS 0.85 -> 0.91 (419 -> 449 of 495), CE 0.88
   -> 0.97, SE 0.73 -> 0.79, NE 0.84 -> 0.90, NO 0.92 -> 0.96, CW 1.08 ->
   1.17 (Jakobshavn 1.29), SW 1.07 -> 1.25, and NW 0.77 -> 0.77 (85.9 ->
   85.6 Gt/yr): every region moved except the one it was run for.
   1986-2027: SMB 347 -> 367, residual D 437 -> 462 (Mankoff D 462, D + BMB
   485), MB -89 -> -95.5 (-148). Speed ratios on covered cells: slow ice
   (< 100 m/yr) 1.10 -> 1.41 (the interior over-deformation), 1000-3000
   m/yr GrIS 0.90 -> 0.97 but NW 0.78 -> 0.79, > 3000 NW 0.45 -> 0.44.
   WHY NW DID NOT MOVE: on the fast cells the inversion answered the
   softer ice by RAISING traction everywhere -- beta NW 1.92 -> 3.76, SE
   1.22 -> 2.05, CE 1.08 -> 1.95, CW 1.61 -> 2.51 -- so basal drag now
   carries 50-65 % of the driving stress (was 20-36 %) and the membrane
   share fell as intended; SE / CE / CW came out 7-9 % faster net, NW
   exactly as slow (0.75) with beta doubled although the model is too slow
   there. So Claude's reading of 2026-09-26 -- viscosity is THE lever in NW
   -- is not borne out: the rheology did act, and the inversion spent the
   freedom on traction. What holds NW back is how beta is determined
   there, not what resists the flow: most likely the too-fast slow ice
   around the narrow outlets (1.33x in NW) pulling beta up through the
   prior's correlation, the v6 mechanism again, and the incomplete
   convergence. NW's driving stress on the fast cells also fell (21 -> 19;
   BedMachine 24).

18. `config.thermal: Optional[ThermalConfig]` (2026-09-27; None = the
   isothermal B from A_glen, the Alaska behaviour): THERMOMECHANICAL
   COUPLING through glide's enthalpy model, `glacier_inverse/thermal.py`
   (`ThermalDriver`, one per run level, built lazily by
   `GlacierProblem.thermal_driver`; the same object in forward_standalone,
   forward_projection not yet wired). Motivation: v11 (uniform soft A) showed
   the interior needs hard ice and the outlets soft; the enhancement-factor
   field is the planned multiplier on top (`ThermalConfig.enhancement`,
   scalar for now). **glide side** (molho tree, uncommitted, on top of the
   user's 9d486d8 + wall clip): `glide/enthalpy.py` + `cuda/enthalpy.cu`
   copied verbatim from Jacob Downs' fork (`../glide-downs/glide` @12ea4cc,
   conservative Aschwanden enthalpy in sigma coordinates, column Newton/Thomas
   + layer Jacobi, implicit Euler, LF advection sharing flux.cu's mass flux)
   with three edits: `rho_i` is a constructor argument (was a hard 910 vs
   917 in the coupling), Paterson-Budd uses the PRESSURE-ADJUSTED temperature
   (was absolute T capped at T_pmp, ~1.5x stiff at a 2 km temperate bed), and
   `get_arrhenius_factor(weighting='shear')` collapses A(sigma) to one B per
   column as A_eff = (n+1) int A (1 - sigma)^n dsigma (the shallow-ice
   deformation weight; MOLHO carries one viscosity per cell; 'mean' = the
   fork's plain mean); `ThermalModel` appended to `model.py` with the DIVA
   branches removed, frictional / strain heating on molho's drag
   beta xi^p (|u|^2 + u_reg)^((m-1)/2) |u| (the fork used beta xi and
   u_reg in m/yr, molho's is (m/yr)^2), B pushed through
   `mg.rheology.B.set(start_level=level)` (the fork wrote the fine level
   only: coarse FAS levels kept a stale B), `thin_B` option, state_dict;
   `GlideStep` (torch.py) now CHECKPOINTS B with the step like q / h0
   (`tests/B_checkpoint_test.py`: step-1 gradient unchanged by a 3x softer
   step 2 to 5e-6; HEAD fails it); `io.py` gained 3-D VTI + callable fields
   inside the compressed writer. Tests: the fork's five enthalpy tests pass,
   grad_test 0.85 % / jvp 0.3 % as before, examples/thermal/coupled_dome runs.
   **Driver**: every forward run starts with a frozen-geometry THERMAL SPIN-UP
   on the initial state (`spinup_outer` = 2 cycles of: momentum solve, 1-kyr
   implicit enthalpy steps to a mean basal change < 1e-3 K, push B; warm-
   started from the previous run's E on the same level), then one enthalpy
   step after every dynamics step; no gradient through the thermal state (B
   is a frozen adjoint input). Surface T = annual-mean t2m + tbias capped at
   0 degC, fixed; Q_geo uniform 0.05 W/m^2 (no product on disk; the fork's
   `examples/greenland/greenland_thermal_forcing.py` builds Martos 2018 /
   SeaRISE maps, copied into glide). Solver tolerance: `absolute_tolerance`
   1e-6 (the fork's 1e-3 left the spin-up smoother-limited: 0.2 K, 2 % in B).
   forward_soln.nc / VTI gain T_bed, T_mean, omega_w_bed, T_pmp_excess_bed,
   B, attrs `thermal`, `thermal_spinup`, `A_glen`. Cost: spin-up 0.5-1.5 s
   (level 2), 7-22 s (1 km); a coupled step is a few ms.
   **First results (v10 physical fields, level 2, 100-2019)**: mean T_bed
   270.5 K, temperate bed 64 % under H > 100 m (52 % under < 30 m/yr ice,
   95-97 % under > 100 m/yr); shear-weighted A_eff median 5e-17 on < 30 m/yr
   ice, 9e-17 at 30-100, 1.2e-16 at 100-1000, 1.4e-16 above (Pa^-3 a^-1) --
   the interior/outlet contrast is there, but the ABSOLUTE interior value is
   NOT hard: the temperate bed dominates the shear weight. Slow ice runs 1.47x
   observed on v10's beta (which was fit at 1e-17; beta ~3.5 there, so the
   interior slides and only a re-inversion can judge the rheology). Level-2
   inversion path: simulate + loss + backward finite in every block.
   **OPEN -- 1 km, 25-yr steps**: with the thermal B the momentum solve is on
   a knife edge. v10: every step 100-275 hits the 10-V-cycle cap at |r_H|
   700-1300 (isothermal 5e-17: converges from the 4th step), 96.5 % of it in
   one 320-cell cluster at x -208, y -3090 km where v10's beta is 0.04 and the
   ice already too thick/fast; v11: NaN in the first step. Outcomes flip with
   start-up details (warm start vs zeroed velocities, the first spin-up solve
   at isothermal vs cold B, thin cells at the isothermal B, 0.75 blends,
   10-cell smoothing -- each converges on one state and diverges on another),
   the blow-up grows from ice-free / thin cells whose own B is unchanged, and
   it is the FINE-level B (thermal on the coarse levels alone converges).
   omega 0.25 or dt 10 do not cure it. At dt 1 (1990-2000, 1 km) and at level
   2 the coupled runs converge in 1-2 V-cycles (level 2 still caps in the
   25-yr era at one trough, x 554, y -1880 km). Needs a solver-side look
   before a full 1 km thermal inversion with the 25-yr spin-up.
   Scratch: `analysis/output/scratch/thermal/` (run_thermal_standalone.py,
   run_variant.py, compare.py, bisect_B.py, grad_check.py, logs).
   **Stabilizers tried and STASHED (user, 2026-09-27: benefit not obvious,
   code harder to read)**: a smooth slope limiter on the facet driving
   stress, tau = H_avg s / (1 + |s / s_max|^p)^(1/p), and an ice-free-only
   linear drag FREE_DRAG / (1 + (H / H_free)^4), both with exact Jacobians
   (jvp / adjoint / grad tests unchanged) and env-selectable defines; saved
   as patches in `analysis/output/scratch/stash_glide_stabilizers/`
   (`git apply`-able against the clip-only stress.cu / HEAD operators.py),
   glide back to the clip (d = 20, the user's value; 100 blew up the thermal
   1 km run, 20 leaves an O(10) |r_H| stall). Sweep findings (thermal, 1 km,
   8 x 25-yr steps, v10 / v11): the limiter at s_max 0.15 + clip finite on
   both states but NaN / stall outcomes non-monotone in every parameter;
   v11's first-step NaN is in ICE-FREE OCEAN cells in front of softened SE
   outlets (water_drag 1e-4), not on a slope; free drag 0.1 without limiter
   was finite on all four runs (v10 |r_H| 4-6 after two steps), the
   remaining NaNs sat in the spin-up's second momentum solve (the jump from
   the cold initial B to the equilibrium B). Logs in `sweep_stab/`,
   `sweep_free/`.
   **Inversion outputs (2026-09-27)**: with config.thermal set, inverse.py
   writes T_bed and T_mean (trapezoid depth average, K) into the loss VTI,
   the periodic time VTI (`io.make_*_vti_writer(thermal=)`) and
   `level_<n>/inverse_soln.nc` (end-of-run state).
   **Calving in the FAS cycle (2026-09-27; glide multigrid.py, uncommitted on
   cfd78a5; SolverConfig fields `freeze_coarse_calving`, `psi_restriction`,
   `truncate_calving_correction`, `truncate_calving_velocity`, `trace_file`,
   `trace_every`).** The user's fix for the 1 km stalls was +150 finest
   sweeps. Tried: (1) coarse calving unfrozen on the restricted psi -- stalls
   return even with the extra sweeps (why it was frozen); (2) the sink-
   consistent restriction 1 - psi_c = sum((1-psi) H) / sum(H) -- unchanged;
   (3) scaling the prolongated coarse H correction by psi -- unchanged frozen,
   NaN unfrozen. The TRACE (`FASCDSolver.trace`: residual split into calving /
   front / constrained / interior cells at start, pre, coarse, post and every
   trace_every finest sweeps) located the stall on 1 km forward runs from the
   v11 L1 checkpoint (finest 0: 13 capped solves t ~ 475-800, NaN at ~800):
   ONE site, x 564 y -1886 km, a 5-7-cell fjord (bed -700..-1170 between
   +1000..+1500 m walls) OUTSIDE the pin, where upstream ice (320-385 m,
   4-5 km/yr) feeds a thin floating apron (psi 0, h0 +250, sink 1/yr) of
   length ~u tau; level 2 sees the fjord as 1-2 cells, level 3 not at all.
   There each coarse correction raised |r_H| ~100 -> 5e3-4e4 and the
   calving-cell MOMENTUM residual to ~1e5, flipped psi on 3-10 k cells, and
   post-smoothing undid it -- a limit cycle; converged solves show a ~10x
   jump that post-smoothing removes. Hence (3b) `truncate_calving_velocity`:
   also scale the u/v/ud/vd corrections on facets touching calving cells
   (facet weight = min psi of its cells). Full 1 km forward 100-2020, frozen
   coarse calving, finest 0: 114 solves / 132 V-cycles, 1 capped (the cold
   first solve), no NaN; vs finest 150 without truncation 120 V-cycles, 0
   capped; final states agree (volume 1e-5, |dH| median 6 mm, p99 0.4 m,
   162 apron cells differ); wall 127 vs 148 s incl. ~20 s thermal spin-up
   and SMB. Not yet tried in the inversion (forward-only change; the adjoint
   cycle is untouched). Scripts: analysis/output/scratch/thermal/
   forward_trace.py, run_inverse_trace.py (inverse.py with config overrides,
   writes under domains/greenland/analysis_scratch/).
   **User, same day: truncation, then `lag_calving_flag` / `lag_flotation`
   (psi, then phi / xi held at the start-of-step state), each looked like a
   fix once and failed under other settings -- luck, not mechanism.**
   **SAVE-AND-REPLAY found it (2026-09-28).** glide `dump.py` +
   `FASCDConfig.dump_dir` / `dump_max` (SolverConfig `dump_dir`): every
   solve ending unconverged / non-finite writes its starting state + all
   solver settings; `load_solve_state(path)` rebuilds a model and
   `model.forward_solver.solve(dt)` repeats it exactly (checked: the omega
   0.25 dump diverges to the same 2.83e12, omega 0.5 converges in 4).
   Solver experiment hooks: `FASCDSolver.correction_hook(l, level)` (edit
   the prolongated z_*) and `.h_prolongation`. Scripts in
   analysis/output/scratch/thermal/: replay.py (options, `corr=velocity|
   thickness|none`, `corr_scale`, `n_levels=1`), zH_anatomy.py,
   replay_backtrack.py, replay_matrix.sh. From the v11 L1 checkpoint with
   the config of 2026-09-28 (finest 0): omega 0.25 and post_steps 50 NaN at
   the FIRST step, unfrozen coarse calving stalls from step 16 and NaNs at
   39; the baseline runs clean. Dissection of those solves: the COARSE-GRID
   THICKNESS CORRECTION does the damage -- smoothing alone, velocity-only
   correction, correction x 0.5 or bilinear H prolongation all converge
   where the full correction diverges; on the first step (frozen coarse
   calving) it puts +2500-2800 m into 2x2 blocks of ICE-FREE OCEAN in front
   of the outlets (no coarse sink: diagonal 1/dt vs the fine 1/dt + rate,
   26x), with unfrozen calving +100-250 m into the thin fast apron of the
   fjord at x 564, y -1886 (a boundary layer of length u tau the 2-4 km
   levels cannot represent). No single fixed remedy works on every dump,
   so the fix is GLOBALIZATION: `FASCDConfig.backtrack` (SolverConfig
   `backtrack`, `backtrack_scales` (1, 0.5, 0.25, 0)): accept a V-cycle
   only if it lowers the finest residual norm, else restore the state and
   redo it with every prolongated coarse correction scaled by the next
   factor (0 = pure smoothing, kept). Replays: every dump converges with
   frozen coarse calving. Full 1 km forward 100-2020, finest 0, backtrack:
   baseline 118 V-cycles / 0 capped / 114 s (1 backtracked cycle); omega
   0.25 141 / 0 / 150 s; post_steps 50 180 / 2 / 106 s; unfrozen 124 / 0 /
   125 s (all three were NaN); omega 0.25 + post 50 finite but 31 capped
   (too little smoothing). Previous fix finest 150: 148 s. Final states
   agree to 1e-5 in volume (median |dH| 2-4 cm) once the spin-up criterion
   below is fixed. Forward cycle only; the adjoint cycle is unchanged.
   **Thermal spin-up stopping test was wrong**: 'per-1-kyr change of mean
   T_bed < 1e-3 K' stopped the second cycle after ONE step (4.6e-4 K) in
   some runs and after 24 in others (depending on the momentum solve
   before it), 269.73 vs 269.57 K, 53 vs 48 % temperate bed, +5 m interior
   thickness by 2020 -- a solver-setting-dependent thermal state. Now
   `spinup_tol_K` (0.01 K) bounds the ESTIMATED REMAINING change,
   dT r / (1 - r) with r the ratio of successive changes, with
   `spinup_min_steps` 3; every run lands at 269.58 K (~5 s more).
   The experiments with no robust effect (psi_restriction,
   truncate_calving_correction / _velocity, lag_calving_flag, lag_flotation)
   were STRIPPED before the commit; kept: freeze_coarse_calving / _phi
   exposure, trace, dump / replay, backtrack. User, same day: with backtrack
   and post_steps 50 the 1 km inversion ran 20 fine-level iterations before
   a NaN (dump_max had been reached by then). The ten dumps it did write (`./dump`,
   solves 6-1338) are all mild: |r| stalls at 10.6-15.3 against
   absolute_tolerance 10, and 63-90 % of the remaining |r_H|^2 sits at ONE
   fjord, x 545-563, y -1871..-1908 km (the Scoresby Sund / Daugaard-Jensen
   system; dump 1291 also x 491 y -2291, 5 km/yr ice): a smooth same-signed
   r_H of -0.3..-0.5 m/yr along a 3-5-cell fast grounded trunk (1-3 km/yr,
   beta ~16 at the pinned front) plus the apron below it. NO mask / psi /
   phi flips between cycles -- slow convergence, not a switching cycle:
   30 V-cycles converge (after 12), post_steps 100 in 2, 150 in 1, finest
   100 in 1; backtracking is irrelevant there. Reading: an along-channel
   transport mode (upwind H at Courant 25-75 couples cells far down a trunk
   narrower than a coarse cell) that local Vanka sweeps move ~1 cell per
   sweep and the coarse grids cannot represent. `dump_max` now keeps the
   MOST RECENT files (older ones deleted), so the solve that ends a run is
   always on disk.
   **The first-step NaNs (2026-09-28): COLD START at large dt.** A NaN at
   the first step after the thermal spin-up (inversion from
   inverse_v14/level_0, and dump 2928) is not the thermal solver (B finite,
   17-63) and not the coarse grid: pure fine smoothing blows up in ONE
   sweep from u = 0 at dt 25 (x 68, y -3216 km: 473 m of grounded ice on
   steep ground beside a calving cell; H -> 3.7e4, vd -> 2.2e6); level 1
   diverges first in the full V-cycle for the same reason. At dt 25 the
   thickness row of a Vanka patch is weak (1/dt = 0.04 against velocity
   couplings ~H/dx), and from zero velocities a few patches take runaway
   Newton steps. Every run starts cold (reset_state and the thermal
   spin-up zero the velocities), which is why all of the day's failures
   were FIRST steps and why they depended on the parameter state. The same
   state solved at dt = 1 converges in 2 V-cycles, and from those
   velocities the dt-25 step in 5. FIX: glide `FASCDConfig.cold_start_dt`
   (SolverConfig `cold_start_dt`, default 1.0): a solve starting from zero
   velocities with dt > cold_start_dt first solves at that dt, keeps the
   velocities, restores H / H_prev / mask / flags, then takes the real step
   with the COLD residual as its relative-tolerance reference (the warm
   start's initial residual is ~13x larger and would loosen the step).
   Also fixed: the backtracking fallback multiplied a NaN coarse correction
   by 0. Replays: every first-step dump converges in 5-6 V-cycles at the
   usual accuracy; full 1 km forward from v14 (post_steps 50, backtrack,
   guard): completes, 115 solves / 343 V-cycles / 166 s, 5 solves capped at
   |r| ~ 12 (floors at the NW trunk x -293 y -1821 and Scoresby Sund; 150
   post-sweeps clear them, more V-cycles do not).
   **ROOT CAUSE of the cold-start blow-up: the shear-row Newton correction
   (2026-09-28; glide stress.cu `get_sigma_vert_dvisc`).** Patch-level
   instrumentation (a debug copy of vanka.cu recording, per Newton iteration,
   the damped J, r, state, LU pivots and step of two target patches;
   scratch `patch_debug.py`, `patch_debug_vcycle.py`, copy in
   `analysis/output/scratch/glide_dbg/`) on the v14 first step: both patches
   START well conditioned (cond 20-30, pivots ~1, steps < 10 m/yr), then within
   2-4 iterations the ud / vd DIAGONALS change sign (-18 -> +27 -> +67 ->
   1e10) while the u / v diagonals stay negative; the patch goes singular
   (cond 5.6e7) and steps 1.7e7. The correction recovers E2 by inverting eta,
   but eta is frozen at the sweep's starting state while the patch Newton
   moves u_d: from a cold start E2 ~ eps_reg, so the correction grows like
   u_d^2 / eps_reg. At any consistent state it is bounded (the facet is part
   of the invariant, E2 >= K_2 u^2 / (2 den)) and can only scale the diagonal
   down to 1/n. FIX: E2 <- max(E2 + K_2 (u_c^2 - u_c0^2) / (2 den),
   K_2 u_c^2 / (2 den)) with u_c0 the facet's sweep-start value -- exact at
   u_c = u_c0 (a plain lower bound also changed normal iterations and put
   grad_test over threshold, 1.4 %). Only the smoother Jacobian changes, not
   the residual. Results: the two patches converge monotonically (|r| 65 ->
   0.13); pure fine smoothing from the cold start is stable; with backtracking
   and cold_start_dt=None every first-step dump (v14, 2928, omega 0.25,
   post 50) converges in 4-5 V-cycles; full 1 km forward from v14 without
   the guard: 114 solves / 350 V-cycles / 178 s, first solve 6 cycles (guard
   run 343 / 166 s; end states differ by 7e-4 in volume, median |dH| 1.4 cm
   -- probably the solves ending unconverged at the floors, not
   confirmed). glide tests: grad 0.31 % (HEAD 0.85 %), jvp / adjoint / ssa
   unchanged. **REVERTED the same day (glide 768879e, user: 'super broken'):
   the incremental safeguard breaks WARM-STARTED solves** -- the thermal
   spin-up's second momentum solve (dt 1, velocities from the first solve,
   new equilibrium B) diverged in its first V-cycle (r_ud 7e9) and ended
   unconverged, where the previous glide converges it in 1 V-cycle; the first
   real step then failed the same way after the cold-start pre-solve. Claude
   missed it because the run summary counted only the solves AFTER the
   spin-up, and the single-level harness (`sweep_track.py`: smoothing against
   gamma = thklim) is NOT the V-cycle's first sweep (FASCD pre-smoothing uses
   the local constraint w_H + phi): the previous glide also 'blows up' in it on
   that dump while its V-cycle converges, so the harness's 'pure smoothing is
   stable' readings are not evidence. After the revert, inverse.py from
   inverse_v14/level_0 (one iteration, forward + backward + final
   evaluation): spin-up solves 2 + 2 V-cycles, 282 solves, 9 capped at the
   floors, no NaN. STANDING: the cold-start sign flip of the ud / vd
   diagonals is real and diagnosed; `cold_start_dt` (1.0) + backtracking is
   what handles it. A correct fix must be tested on warm-started solves and
   through inverse.py, not only cold first steps.

19. `config.sliding_u0`, `sliding_N_scale_H`, `sliding_N_floor_H` (2026-09-28;
   0 / None = the Alaska behaviour): REGULARIZED COULOMB drag with a
   DIMENSIONAL, FLOORED effective pressure. In glide (uncommitted), with
   xi_f = clip(1 - d / (r H), 0, 1) the flotation fraction (d = -bed),
     tau_b = beta X^p |u|^m (u0 / (|u| + u0))^m,
     X = xi_f                                  sliding.N_scale_H = 0 (normalized: N / rho_i g H)
     X = xi_f (H + N_floor_H) / N_scale_H      N_scale_H > 0: N* / (rho_i g N_scale_H),
   N* = xi_f rho_i g (H + H0) -- N itself on thick grounded ice, 0 at
   flotation, bounded below by rho_i g H0 on thin grounded ice (the reason the
   law was normalized in the first place: a bare N = rho_i g H drags thin
   margins to zero). Land sensitivity nu = dlnN*/dlnH = H / (H + H0): ~1 on
   thick ice (what the Coulomb result needs), -> 0 on thin margins. N_scale_H
   is a pure unit scale, degenerate with beta (C = beta / (rho_i g H_s)^p).
   No spatial reference field and no reference year. `sliding.u0` (m/yr) is
   the Coulomb transition: Weertman below, capped at beta X^p u0^m above;
   one device helper (`stress.cu drag_speed_factor`) serves every drag
   evaluation (residual, Vanka, JVP, gradients), and `model.py` frictional
   heating matches. Tests: defaults bit for bit (jvp 0.3 %, grad 0.85 %,
   adjoint < 1e-6); u0 100, H_s 1000, H0 100: jvp 0.3 %, grad 0.12 %, adjoint
   2e-7; greenland_coarse forward + loss + backward finite. forward_soln.nc
   records `sliding_u0`, `sliding_N_scale_H`, `sliding_N_floor_H`.
   WARM START: beta's meaning changes, so convert the checkpoint with
   `tools/convert_beta_warmstart.py` (log beta += p ln(H_s / (H + H0)) -
   ln R(|u_b|) at a forward replay's 2018 state, ice cells only, re-whitened
   exactly under the log-beta prior); from v14 with u0 300 / H_s 1000 / H0 100
   the median change is -0.54 in log beta and 41 k thin, slow margin cells
   land slightly above beta_max 20 (median 22-25). An intermediate version
   (2026-09-28, same day) used a per-cell reference thickness
   (`sliding_N_ref`, `geometry.H_ref`) and a subglacial `water_head` field;
   both were removed.
   WHY (replays from year 100 on v14's beta, `inverse_v14/slide_test/`,
   `analysis/output/basin_mb_{lia,coulomb}/`, scratch `slide_test/`): v14
   loses half the observed ice 1993-2019 (-81 vs ITS_LIVE -153 Gt/yr) and NW
   is near balance in 1986-95 (Mankoff -38): the model's thinning stops within
   ~400-800 m of elevation while the altimetry's reaches 2000 m. Speed-up with
   thinning needs eta nu > 1 (eta = dlnF/dlnN, nu = dlnN/dlnH); glide's
   normalized N has nu = 0 on land, so thinning SLOWS land-based ice. The
   GRISHM LIA front pin (`front_mask_lia.nc`) + regularized Coulomb (u0 300)
   + dimensional N gives NW 1986-95 / 2006-20 -38 / -66 (Mankoff -38 / -70),
   1993-2019 -44 (ITS_LIVE -38.5), the inland profile to 2000 m, SE / NO
   right; overshoots CW (-56 vs -26, gates 142 vs 79) and the total (2018
   gates 586 vs 495; GrIS 1993-2019 -193). Coulomb alone on the normalized
   xi, or the LIA pin alone, gives about half; a smoothed-overburden water
   head adds nothing on top of Coulomb. Claude predicted Coulomb would NOT
   reach land-based ice (amplification only): wrong for the dimensional N,
   where the thickness term is neutral (eta nu ~ 1) and the plastic bed
   raises the slope diffusivity. Replays at full resolution take 15-17 min
   (the plastic bed caps many 25-yr spin-up solves at |r_H| < 150; the
   observation period converges). Not yet inverted: fast-ice beta rises by
   ((|u| + u0) / u0)^m, so `beta_max` may bind at outlets.

20. `config.*_prior` may be a `SpectralPriorHyperparams` (2026-09-29; a
   `PriorHyperparams` = the Alaska behaviour, ggapp's multigrid MaternPrior):
   `glacier_inverse.priors.SpectralFieldPrior`, the field prior as an explicit
   variance spectrum on the domain's orthonormal DCT-II basis (mirror
   boundaries, which diagonalize ggapp's 5-point Neumann Laplacian exactly):
   forward C^{1/2}, whiten C^{-1/2}, both self-adjoint and exact, O(N log N),
   duck-compatible with GGaPPMap / GGaPPWhiten, `sample()` exact. The
   spectrum is a SUM of `PriorComponent(sigma, l, nu, mass)` terms
   (tau/dx)^2 (mass kappa^2 + lambda)^-(nu+1), tau and kappa from the Matern
   (sigma, l, nu): the SPDE with the MASS term decoupled from the derivative
   terms. mass = 1 is the Matern (a single component reproduces ggapp's
   MaternPrior: forward 3e-5, whiten 4e-7 relative); mass = 0 its intrinsic
   counterpart (derivative penalty only; needs `sigma_mean`, the prior std of
   the domain-mean mode). WHY (user, Coulomb single-step tests): interior log
   beta would not move whatever beta_init or m. The adjoint was right (level
   3, interior -0.554 vs FD -0.571), but a +0.1 coherent shift of log beta
   over the slow interior (1 M cells) cost 776 in 0.5|z|^2 under the Matern
   1 / 2 km prior (l was 8 km before commit 3500420: 50) against a
   velocity-misfit change of 0.6 (sigma_floor 10 m/yr): the mass term
   kappa^4 x^2 penalizes amplitude in proportion to AREA, so a short-range
   prior with a fixed mean pins the regional mean of log beta to mu_log_beta
   (the Weertman law only looked fine because mu = ln 2.5 was about right for
   it). The user fixed the run with a longer l and a lower sigma_floor; the
   spectral prior is the structural fix. Production grid, cost of the same
   interior shift / sample std: Matern 2 km 776 / 1.05; + Matern 1 / 200 km
   13 / 1.5; + Matern 1 / 500 km 24 / 1.5; old 8 km 50 / 1.0; + INTRINSIC
   1 / 200 km 13 / 29 -- the alpha = 2 intrinsic spectrum ~ lambda^-2 puts
   enormous variance in the domain-scale modes (pure intrinsic 2 km: std 361
   on a 512 km test domain), so use a PROPER long component, not mass = 0,
   for log beta. 2 km structure costs the same in all of them. A
   checkpoint's z is only meaningful under its own prior:
   `tools/convert_beta_warmstart.py --old-prior 1,2000,1` re-whitens exactly
   (physical log beta preserved to 3e-6). Tests: tests/test_spectral_prior.py;
   smoke_test checks a spectral prior's spec. The bed conditioner (ggapp
   ConditionedPrior) needs a ggapp bed prior: keep bed_prior a
   PriorHyperparams while bed_conditioning is enabled (untested otherwise).
   TWO-FIELD VARIANT (same day; the user found the single summed spectrum
   impractical under SGD: the long component's huge low-mode variance sets
   the stable step for the whole whitened vector, so the 2 km structure
   crawls): `config.log_beta_mean_prior` (None = off) + `lr_z_log_beta_mean`,
   log beta = mu + Map(z_log_beta) + Map_mean(z_log_beta_mean), the second
   field with its own prior term `J_prior_beta_mean`, optimizer group
   (inverse.py), checkpoint key `log_beta_mean` (absent -> zeros) and
   influence-cap mapping; posterior.py / sensitivity.py pass it through;
   rto_sample.py does NOT perturb it yet. Checks (level 3, Greenland): J
   identical with the field at 0; adjoint vs FD through the mean field -130 /
   -122 (the step's nonlinearity, as for the short field); checkpoint round
   trip exact. Step scale per unit lr, first SGD step, physical log beta rms:
   short 1.6, mean (Matern 1 / 200 km) 154 -> lr_z_log_beta_mean ~ 1/100 of
   lr_z_log_beta (default 0.01). CENTERED MODE (same day, user's suggestion
   from the bed field; `config.log_beta_mean_mode`, default "centered",
   "additive" = the form above): log beta = mu + Map(z_log_beta) ALONE and
   J_prior_beta = 0.5 |Whiten(log beta - mu - m)|^2 with m = Map_mean(z_mean)
   (re-whitened from physical.log_beta in loss.compute_prior, exactly as the
   bed's prior term), J_prior_beta_mean = 0.5 |z_mean|^2. A linear change of
   variables of the additive form (same objective, same MAP), but the data
   gradient reaches only z_log_beta, at the short prior's step scale, and m
   is driven by the prior coupling alone. `log_beta_from_whitened` adds m
   only in additive mode; `log_beta_mean_from_whitened` gives m. Warm starts
   carry over unchanged (z_log_beta whitens the full field, z_mean = 0).
   NOT YET TESTED on the GPU (the conditioning comparison was interrupted by
   a machine freeze); the live config runs it. Equivalent alternative, not implemented:
   one spectral field with a per-mode gradient preconditioner
   m_k = sum_i a_i S_i,k / sum_i S_i,k (reproduces the two-field SGD path).

21. **Force-balance traction start, `tools/beta_force_balance.py`
   (2026-09-29; a tool, no library change).** The velocity misfit's
   sensitivity to log beta is d u_s / d ln beta = -(1/m) u_b, i.e.
   proportional to the BASAL speed. From the uniform prior start (beta_init 4,
   m = 1/3, dimensional N with X ~ 2-3 on thick ice) the interior slides at
   ~0.01 m/yr, so its gradient is ~1e4 too small whatever its misfit: the
   user's single-step Coulomb inversions freed the ice as a WAVE from the
   outlets that died where the membrane coupling ran out (scratch
   `probe_beta_gradient.py`, level 1, data-only gradient w.r.t. physical log
   beta: at the prior start NEGIS 125 km upstream ran 0.5 m/yr against 359 at
   |g| 8e-6, the outlets 1e-2; the converged run kept beta ~3.3 beyond 400 km
   upstream, a sharp step where the fit ended; g / u_b roughly constant).
   The tool sets, per cell, tau_d / (rho g) = H |grad S| (S smoothed 4 km),
   u_def = shallow-ice deformation for A_glen / n, u_b = max(u_obs - u_def,
   0.2 u_obs, 1 m/yr), and beta0 from glide's own drag at u_b (m, Coulomb u0,
   dimensional N and floor), log-smoothed 2 km, prior mean off the valid
   ice, clipped to beta_max; whitened under the configured prior(s) (centered:
   the mean field seeded with log beta0 smoothed at l/2; additive: the split
   the other way), every other parameter copied from `--base`. From it
   (level 1): J_data 827 (prior start 3314, converged run 654) and live
   gradients everywhere -- NEGIS upstream |g| 1-7e-3 instead of 1e-6-1e-4,
   the model there at 0.6-0.8 of observed (the local balance ignores the
   margins' share of the stress, so fast ice starts too sticky: outlets
   -5..-8 sigma, which the optimizer fixes fast because their gradient is
   large). The user's alternative, a sign-based optimizer (it fixed a similar
   stall in Antarctica), removes the magnitude disparity instead of the
   plateau; the two combine.
   SIMPLIFIED THE SAME DAY (user): PLUG FLOW (u_b = max(u_obs, --min-ub), no
   deformation term, no A dependence) on the observed thickness / bed, and a
   PARTIAL checkpoint holding only `log_beta` (+ `log_beta_mean`) and the bed
   parametrization tag, so the start depends on the inputs and the config
   alone (`--base` removed). `io.load_whitened_params_into` now tolerates
   partial checkpoints: every absent key (bed, bed_mean, pbias, rf / mf, and
   as before tbias / tau / z0 / H_atm / cloud) keeps the problem's own
   initialization -- NOT z = 0 for the bed, which would be a flat bed at the
   prior mean. Checked: loaded into a fresh GlacierProblem every non-traction
   field is bit-identical to the problem's initialization. Default output
   `domains/greenland/beta_init_force_balance.p`; beta0 medians 1.4-1.8 on
   slow ice, 4.0 at 300-1000 m/yr, 7.5 above (u0 300, H_s 1000, H0 100).

22. **Pre-1950 ocean forcing and the released pin (2026-09-29;
   `OceanForcingConfig` defaults = the dTF = 0 hold, bit-identical).** EN4
   starts in 1950, so the free calving law's spin-up (and any LIA extension
   of the fronts) sees whatever the pre-record ocean is assumed to be. Three
   options, all in `ocean.py`, combinable:
   (a) HOLD `pre_record_scale` / `pre_record_offset` / `pre_record_ramp`:
   TF_pre = max(scale TF_clim + offset, 0), ramped linearly to 0 over the
   `ramp` years before the record;
   (b) INDEX `pre_record_index` (file, e.g. `temperature_anomaly.nc`),
   `pre_record_index_k` (K of TF per K of index), `_smooth` (11 yr centred
   mean), `_start` (first year, None = the whole index): dTF_pre += k (I_s -
   I_ref), I_ref the smoothed index's `ref_years` mean, so it is referenced
   like EN4's dTF; k fitted from EN4 front TF on the 11-yr index over
   1950-2025: 0.19 all fronts, 0.28 NW, 0.31 SE (r 0.85-0.94, ~7 dof);
   (c) RELEASED PIN `pin_release_year` (`ocean.ReleasedPin`, wired in
   problem.py and forward_standalone): the config's pin for steps ending at
   or before that year, the free TF-driven margins after -- the pinned
   extent as an INITIAL CONDITION. A step straddling the record start
   averages / maxes the pre-record years in.
   Why not GCM TF: CESM2 / MRI front TF has no interannual skill against
   EN4 (r 0-0.2, free-running), and 1850-99 minus 1950-79 is -0.32 / -0.04 K
   at NW fronts, -0.6 / -0.3 SE -- the same size as the index route.
   Tool: `sweep_pre_tf.py run` (level, grid of scale x offset, `--index-k`,
   `--index-start`, `--release-year`, `--keep-pin` for a pinned reference,
   `--t-end`, `--vti --vti-from`; pins OFF unless kept / released) and
   `score` (vs GRISHM `lia_mask`: cover of the LIA retreat zone with H > 50 m,
   keep of 2015 marine ice, over_km2 beyond the LIA extent, per region).
   `analysis/basin_mass_balance.py` now reads coarse-level runs (frames
   repeated onto the 1 km grid: area integrals exact, gates sample the coarse
   cell). RESULTS (user's v1.1 isothermal checkpoint, level 1, all with the
   config's `calving_h0_base_tau1.nc` per-basin margin, alpha_h 80, tau 1;
   `inverse_v1.1/sweep_*`, `analysis/output/basin_mb_{pre_tf,release,index,
   index1850}_l1/`; dM/dt Gt/yr, Mankoff NW 1986-95 -38, 2006-20 -70, GrIS
   -74 / -239, 1986-2020 -148):
   * HOLDS: the held extent is reached within ~1600 yr (1700 = 1850 = 1900
     states); cooling raises LIA cover but overshoot beyond the LIA extent
     faster -- GrIS area error (uncovered + overshoot) 8.4 k km2 at (1, 0),
     12.4 k at (1, -0.5), 18 k at (1, -1); in NW too (2.6 / 3.0 / 3.7 k);
     scale hits the warm SE / CE fjords (overshoot 5-14 k km2 against LIA
     zones of 0.3-0.5 k), NO needs the offset, NE sits at cover 0.52 in
     every run. Mechanism: a negative margin makes floating ice admissible,
     so a cooled fjord fills until the margin flips or the fjord ends -- a
     hold picks WHICH fjords fill, not WHERE a front stops. To 2020: offsets
     -0.5 / -1 shed the overshoot inside the record (GrIS 2006-20 -398 /
     -429, CW -113 / -155, 1993-2019 -266 / -281 vs -153); offset 0 (the
     plain free law) -24 / -298, 1986-2020 -144, 1993-2019 -170, NW -2 / -88,
     but 51 k Gt less ice than the pinned run at 1985 and SE / CE gates 52 /
     27 against 81 / 55 pinned.
   * RELEASED PIN (1850 or 1900, identical): the LIA extent is still
     unwinding in the record (GrIS 2006-20 -354 / -357, 1986-2020 -194 /
     -204; CE, CW, SE too negative); NW 1986-95 -11 / -13, i.e. NW's early
     imbalance is NOT a legacy of the LIA extent -- the pinned run gets -18
     only from the imposed 1900-1972 retreat. Initial condition, not physics:
     dropped.
   * INDEX, whole record (GISP2 era included): a RATCHET. The index is warm
     vs 1950-79 through most of the Common Era (+0.3..+0.6 K, 11-yr maxima
     +2..+4 K), each warm excursion strips fronts, the LIA cold rebuilds only
     part (the free law has no restoring force): ice at 1985 vs pinned -72 /
     -85 / -155 k Gt at k 0.25 / 0.5 / 1.0, NW gates 1990 53 / 51 / 46 vs 73.
     Right totals (1993-2019 -152..-165) for the wrong reason (less ice left
     to lose). Do not use the deep index as ocean forcing.
   * INDEX FROM 1850 (`pre_record_index_start=1850`): no ratchet; the
     1920s-40s warm phase is visible (1900-50 volume change -1.9 k Gt at
     k 0 -> -6.3 k at k 1.0) but mostly decays by the 1980s: NW 1986-95 -5 /
     -6 / -11 (k 0.25 / 0.5 / 1.0 = 4x the fit), 2006-20 -87 / -84 / -72;
     GrIS 1986-2020 -148 / -150 / -148; SE, CE, NE, SW unchanged. So
     twentieth-century ocean forcing does not explain NW's 1986-95 imbalance
     under this law; part of the pinned run's shortfall is level (-18 at L1
     vs -23 at L0 for the same pin; v14-beta Coulomb/LIA L0 replays gave
     -38). Usable as the physically motivated pre-record forcing at the
     fitted k; within the error bars like the plain free law.
   * EVERY free variant leaves SE / CE gates at ~52 / 27 against 81 / 55
     pinned: a steady-state front-position property of the per-basin margin
     field (fitted 2026-09-25 against v9's beta), not of the pre-record
     forcing. Re-running the per-basin c sweep on v1.1's beta is the lever.

23. **Two-stage calving calibration: stage 1 is the source of truth
   (2026-09-29, the user's design).** Stage 1 (the inversion, fronts pinned
   to the observed history) has extracted everything the products hold;
   stage 2 only emulates its BOUNDARY CONDITION -- the per-basin calving
   margin c_i of `h0_i = c_i + alpha_h dTF_i` -- as a function of basin and
   thermal forcing, so it can be projected. Pieces:
   `forward_standalone.py --stage1 [--pin FILE]` = the stage-1 reference
   replay (the config's yearly pin, else front_mask_lia.nc, for the whole
   run; no release, no h0 field; raw states at 1990 / 1993 / 2008 / 2015 /
   2018 / 2019; frames from 1980) into `{output_dir}/stage1_reference`.
   `analysis/sweep_calving_eval.py --reference-run DIR` makes every
   BASIN-INTEGRATED term compare with that run instead of the products (CPU,
   state files only; `--select-only` also adds finished runs missing from
   sweep_eval.csv and writes `sweep_eval_integrated.csv`):
   `J_gate` (log flux ratio through the basin's Mankoff gates,
   analysis/gates.py = the chain convention), `J_gatef` (flux-weighted gate
   SPEED and THICKNESS log ratios, sigmas 0.15 / 0.2), `J_gatefm` (J_gatef
   summed over `--gate-epochs` 1990 2008 2015 2018, each against the
   reference at that epoch), `J_dhdt_int` (reach-integrated 1993-2019 change
   vs the reference's) and `J_srf_ref` (chi2/2 of the 2008 surface misfit per
   calving basin over cells with ice in either run). `--admissible-term
   srf_ref --admissible-rel R`: per basin only c with J_srf_ref <= (1 + R) x
   the basin's minimum are candidates (ratio test, so the 10 m sigma is
   irrelevant; R is on a SQUARED quantity: 99 ~ 10x in RMS; the sum runs over
   the whole drainage basin, a reach restriction would sharpen it); basins
   without gates use J_srf_ref itself; a single admissible value is a
   decision, not "unresponsive". The assembler writes h0_base = c -
   config.calving_h0 (before 2026-09-29 raw c: a config baseline of 10 m
   shifted every front toward calving -- enough to flip Helheim, Jakobshavn,
   Petermann, whose good windows are 10-25 m wide). Findings (v1.1
   isothermal, level 0, alpha_h 80, tau 1; `analysis/output/basin_mb_v1.1_
   {calvfield,gatef,s1,s1m,s1m2,fine}/`; dM/dt Gt/yr 1986-95 / 2006-20 /
   1986-2020; stage 1 -59 / -247 / -139, Mankoff -74 / -239 / -148):
   * The composite reproduces each glacier's sweep state (independence
     holds) except where glaciers share a fjord (79N / Zachariae) and at
     Jakobshavn; the user's view: fine for now, a red-black / randomized
     design if it ever matters.
   * Products-referenced selection: the whitened per-cell loss ("previous
     loss", srf vel extent dhdt_int) -> -8 / -226 / -107, gates 0.80 of
     Mankoff; FLUX ALONE is not identifying (a filled fjord, Kanger at c
     -12.5: thickness 1.94x, speed 0.42x, matches the flux); the factorized
     gate term vs the products -> gates 0.99 but -8 / -347 / -167: the model
     only carries the observed flux while its outlets retreat.
   * Stage-1-referenced: gatef + srf_ref (10x) -> +6 / -321 / -149;
     gatefm alone -> +12 / -338 / -150 (the 10x filter blocked the states it
     wanted: Jakobshavn +25 is 20x the minimum); gatefm + dhdt_int (dv-rel
     0.5, floor 0.2) with R 99 -> -23 / -299 / -148, gates 279 -> 366 (stage 1
     317 -> 358); plus the fine grid (2.5 m around Rink and Helheim) -> -33 /
     -300 / -154, 1993-2019 -180 (ITS_LIVE -153), 2018 gates 0.87 of Mankoff
     (stage 1 0.88). CURRENT FIELD: `calving_h0_base_v1.1_s1m2.nc` (re-assembled
     on the fine grid): Rink -2.5 (speed / thickness vs stage 1 1.29 / 1.04;
     was 2.55 / 0.86 at 0), Jakobshavn +27.5, Kanger +27.5, Helheim +37.5.
   * HELHEIM IS BISTABLE: gate thickness vs stage 1 1.42 / 1.41 / 1.19 / 1.39
     / 0.42 / 0.48 at c 25 / 27.5 / 30 / 32.5 / 35 / 37.5 -- a jump between
     32.5 and 35 with a chaotic outlier at 30; stage 1's pinned Helheim is not
     a free-law state at any c. Rink had a narrow window (-2.5).
   * Remaining 2006-20 excess vs stage 1: CW -20, NW -18, SE -13 (Helheim),
     CE -7 Gt/yr: outlets reaching the right 2018 state through too much
     thinning, plus Helheim; 1986-95 is out of reach of any c (the free law
     produces no 20th-century retreat; library change 22). More c resolution
     will not help; a calving law that can hold intermediate fronts or joint
     selection of neighbours would.
   * THERMAL (v1.1thermal, 2026-09-30; sweep at 10 m spacing, its own
     `stage1_reference`; `analysis/output/basin_mb_v1.1thermal_{free,gatefm,
     iter}/`): the coupled stage 1 is -51 / -224 / -123 (SMB +20 over the
     isothermal one, pbias; slow ice 1.27x vs 1.18x). Its free composite with
     the s1m selection: -8 / -311 / -152 (1993-2019 -185), 2018 gates 0.92 of
     Mankoff; excess vs its stage 1 doubled (2006-20: CW -29, NW -29, CE -19,
     NE -13). Checked against the sweep's own states: Jakobshavn / Silarleq
     are SELECTION errors (dhdt_int preferred +20 over the gate's +10; the
     100x surface filter cut Silarleq's -10 at 117x), Rink / Sermeq Kujalleq
     have NO c reproducing stage 1 (2x fast at any matching thickness),
     79N / Upernavik N / Kakivfaat / Sermeq Avannarleq differ between sweep
     and composite (neighbour coupling; 79N follows Zachariae's c in the shared
     fjord), Academy / Ostenfeld / Petermann are out of reach at any c.
     Gate terms alone (`_justflux`): -1 / -330 / -159, 2018 gates 1.13 of
     stage 1 -- fixes Jakobshavn and Silarleq, but with no volume term every
     too-fast front draws down further; and gate-less basins lose their
     fit (the periphery fell back to the GLOBAL 0 instead of +65).
   * ITERATIVE ALTERNATIVE, `calving_iter.py` (init / step / status /
     finalize, one step per invocation; `forward_standalone.py --free-h0
     FIELD --out-dir DIR` is the per-iteration run: pins cleared, the field
     from any path via `RHO_PATH`, raw states at the stage-1 epochs). A
     bracketed bisection per gated basin (stage-1 gate flux >= 0.5 Gt/yr) on
     the flux-weighted gate THICKNESS log ratio vs stage 1, averaged over
     1990 / 2008 / 2015 / 2018 -- strictly monotone in c at 101 of 101 basins
     over the sweep (speed and flux are not: a thinning trunk speeds up, then
     collapses); brackets from the sweep, every basin moves at once in the
     composite (neighbours at their current values), jump / expand / reopen
     rules, untracked basins keep the base field. Mechanically right (5
     iterations: flux-weighted |s| 0.20 -> 0.09) and a FAILURE at basin
     scale: 2006-20 -472 (iter 0) / -457 (iter 4) vs the sweep selection's
     -311, 2018 gates 654 vs stage 1's 443, NO 3.5x and NE 2.6x of Mankoff.
     The epoch MEAN cancels: fronts too thick and slow in 1990 (1.1-1.6x,
     0.4-0.6x) and at thickness ~1 but 1.5-2.8x fast in 2015-18 read as
     matched; the tongues (79N -91, Petermann -19 from the mean-signal
     brackets vs the sweep's -130 / -100) keep gate thickness while running
     6.9x / 5.7x. The minimum over epochs instead of the mean is also
     monotone (89 single crossings) and moves the tongues back; a speed veto
     (2x) breaks monotonicity at 8 basins. The per-epoch pattern is TIMING,
     which one c cannot fix (c sets position, alpha_h the in-record response):
     per-basin (c, alpha_h) from (1990 thickness, 1990 -> 2018 change) is the
     well-posed extension, not built. DECISION (user, 2026-09-30): the sweep
     selection is good enough for ISMIP7; projections run the ISOTHERMAL v1.1
     calibration with `calving_h0_base_v1.1_s1m2.nc`, and OCX is submitted
     FREE (the projections cannot be pinned; a pinned OCX evaluates another
     model, and a pinned state handed to the free law unwinds -- the released
     pin result).

24. `ThermalConfig.couple_rheology` / `surface_T` (2026-09-30; defaults True /
   "climatology" = library change 18): ONE-WAY thermal output for ISMIP7.
   The user's reasoning: the coupling from temperature to rheology is too
   uncertain to be useful (the fits do not distinguish coupled from uncoupled;
   independent models' frozen / thawed bed maps disagree wildly), but the
   model's temperature field is still worth submitting for Tb
   intercomparisons. `couple_rheology=False`: `ThermalDriver` spins up and
   steps the enthalpy model with the run's velocities, strain and basal
   frictional heating, but never pushes B (glide `ThermalModel.
   update_rheology=False`; one spin-up cycle) -- the dynamics are those of the
   uncoupled model BIT FOR BIT (level-2 test: max |dH| 0, B the isothermal
   40.95 everywhere; two-way control: |dH| up to 428 m). Strain heating uses
   the flow's own drag and speeds, so it is consistent with the isothermal
   dynamics. `surface_T="forcing"`: the Dirichlet surface temperature per step
   is the annual mean of the air temperature the step's SMB saw, capped at
   0 degC -- `forward_standalone.compute_smb` records it on
   `ctx.forcing_T_annual` (the reanalysis years' anomaly / the index shift,
   + tbias; the Hermite nodes average to the shift), `forward_projection`'s
   compute_smb does the same with the ISMIP7 field + tbias + the elevation
   feedback; `fs.thermal_spinup(ctx, t0, t1)` spins up on the FIRST step's
   forcing (one extra SMB call) and `fs.thermal_surface_update` sets it per
   step. The inverse's forward.simulate honours couple_rheology but keeps the
   climatology surface. `forward_projection` now carries the thermal model
   too: spin-up before a fresh run, one enthalpy step per dynamics step,
   T_bed / T_mean / T_top in snapshots.nc and the VTI, the enthalpy state
   (`thermal_E`, `thermal_E_surface`, `thermal_Q_geo`) in final_state.nc and
   restored by `--continue` (refused when absent). Cost at 1 km: spin-up 18 s,
   a few ms per step. `ismip_exporter.py` exports `litemptop` (T_top),
   `litempavg` (T_mean), `litempbotgr` (T_bed), `litempbotfl`, `hfgeoubed`
   (the config's uniform Q_geo) when the run's attrs carry a ThermalConfig and
   its frames the T fields (else they stay in not_modelled.txt; `litemp` 3-D
   always does), with the request's fill policies (no_ice / no_grounded_ice /
   no_floating_ice / outside_domain) and an `ice_temperature` attribute that
   states the coupling. `litempbotfl` is NOT the enthalpy model's basal node,
   which is not held at the ocean interface under floating ice (251-264 K):
   it is the in-situ seawater freezing point at the ice base (Jenkins 2011
   liquidus, S 34.5 psu; 270.4-271.3 K). ISMIP7 checker on a 1 km ssp585 test
   export: every naming / numerical / spatial / consistency / attribute test
   passes (only the time window of the 3-yr test fails). Caveat: in
   `CLIMATE_MODE="raw"` the surface temperature steps at the record start
   (1850) from the CARRA2 climatology + index to raw dEBM2 tas + tbias, ~3.5 K
   colder in the annual mean -- the SMB sees the same switch; `litemptop` and
   the near-surface ice show it. Config for the ISMIP7 runs: `thermal=
   ThermalConfig(nz=9, Q_geo=0.042, weighting="mean", thin_ice_isothermal=True,
   couple_rheology=False, surface_T="forcing")` on the isothermal v1.1
   calibration.

Adjoint coverage (reviewed 2026-09-13): the flotation fields phi / xi / psi
are frozen inputs to every glide stencil. The effective-pressure pathway is
now differentiated — the drag Jacobians carry d(beta xi^p)/dH and /dbed via
xi = 1 - depth/(r H) on grounded marine ice (stress.cu), `compute_gradient_bed`
adds the drag term, and `tests/grad_bed_test.py` checks the bed gradient by
finite differences on a marine geometry along long-wavelength bed modes
(1% agreement; the old kernels recovered 46% of the gradient there).
Still neglected: dpsi/dH, dpsi/dbed in the calving sink and dphi/dH,
dphi/dbed in the driving stress (delta-like at sigmoid_c = 1/m: 4 and 214
cells in the band), the active set (identity rows, by construction), the
detached bed in `_initial_thickness_from_geometry`, and q/h0 (sweep
parameters). `depth_blend` is now 1.0 (depth == -bed each forward call).
**dpsi is neglected AGAIN (2026-09-17)**: the exact calving Jacobian added
with the monotone law (2026-09-16) made the 1 km adjoint solve fail to
converge, so `flux.cu` drops `- rate H dpsi/dH` from the sink's d_H and
`grad.cu` drops the dpsi/dbed term of the bed gradient (both commented out
in glide's working tree); the `dpsi_dH` / `dpsi_dbed` fields are still
computed, restricted and checkpointed but no longer consumed by those two.

**Level-0 adjoint divergence, fixed 2026-09-17 (glide `dxi_dH` field +
`config.beta_max`).** With the level-1 warm start the first backward solve
(2020 -> 2026, dt 6) diverged: the fine-level smoother alone converges, but
the LEVEL-1 smoother grew ~1.14x per sweep (150 post-sweeps: 3.6e8),
independent of omega and of x100 momentum damping, with 99% of the residual
in ~100 cells at the Humboldt front (x -364, y -1066 km): a slow floating
slab kept alive by a negative calving margin, beside lightly grounded cells
where beta had run up to 80-100 (192 elsewhere). Cause: the drag Jacobian's
effective-pressure term `beta p xi^(p-1) (1 - xi)/H` was evaluated on the
RESTRICTED (averaged) xi of coarse cells mixing floating and grounded
children (xi 0.02, phi 0.25), where the formula is invalid and maximal
(~beta/H). glide now carries `state.dxi_dH`, written by
`compute_flotation_fraction` ((1 - xi)/H where 0 < xi < 1), read by the
TauBx/TauBy Jacobians, RESTRICTED with xi in `restrict_state` and the adjoint
V-cycle, and checkpointed by `GlideStep`; forward levels that recompute xi
recompute it too, so the forward solve is unchanged. Also fixed a typo in
`TauByStencilDual.get_diffs` (`H_t.d,H_t.d` -> `H_t.d,H_b.d`: the drag JVP
used the upper cell's thickness perturbation for the lower one). glide tests
after: grad_bed 1.1%, grad 0.07%, jvp 0.3%. Result: 38/38 adjoint solves, no
damping restarts, ~65 s per backward pass, with or without the beta cap.
`config.beta_max` (None = unbounded, the library default; Greenland 20)
clamps the FINE-level log beta before restriction in `problem.simulate`, the
same order as `forward_standalone.py`'s `BETA_MAX`, so the inverse, the
standalone driver and the projections run the same field. beta > 20 is
effectively no-slip (the loss moves 2121.75 -> 2122.29 with the cap), and
beta * xi is degenerate as xi -> 0, which is why it ran away; the clamp passes
no gradient above the cap, so only the prior acts there. Diagnostics in
`analysis/output/adjoint_diag/`.

Plus ONE non-additive change forced by glide's 2026-09 refactor (signed
flotation excess + height-above-buoyancy calving; alaska-forecast has to
take the same diff when it moves to that glide): `config.py`/`problem.py`
replace `calving_rate`/`sigmoid_k` with `calving_timescale`/`calving_q`
(`mg.calving.timescale`, `.q`), add `thklim` (also the ice-free thickness
`mg.state.H` starts at; `init_H_floor` should equal it), set
`mg.geometry.depth = -bed` UNCLIPPED (negative on dry land; `update_depth`
too), and `SolverConfig` carries the Vanka options (`omega`,
`momentum_damping`, `step_tolerance`) which `_apply_solver_settings` pushes
onto both solvers. Defaults are the values of glide's
`examples/greenland/greenland_forward.py`.

For everything else — the config-is-the-contract pattern, time-stamped
observation specs, multigrid evaluation, MaternNoise/whitened likelihoods,
logit nuisances, influence caps, bed conditioning, checkpoint conversions,
the four drivers — read alaska-forecast's CLAUDE.md; it is authoritative and
not duplicated here.

## Commands

```bash
# Development domain from glide's example file (no bundle needed; ~3 min)
python preprocessing/bootstrap_from_glide_example.py --domain-path domains/greenland_coarse
python smoke_test.py                                   # PASS/FAIL per check

# Science domain (needs common_data/ per DATA_MANIFEST.md)
python preprocessing/make_all.py --domain-path domains/greenland --year 2015   # velocity: ITS_LIVE summary mosaic (default)
python inverse.py                                      # DOMAIN constant at the top
python export_ismip.py --domain-path domains/greenland --date 2015 --regrid-to 4000
python forward_standalone.py                           # MAP forward, no autograd (see below)
```

`forward_standalone.py` is the sandbox for forcing experiments: it exports
the MAP checkpoint to physical fields once (`{output_dir}/physical_fields.nc`,
the only step that touches priors/GP maps), then runs the composite
enthalpy-SMB -> glide forward in the shape of glide's Greenland example
(explicit time loop, cupy state, VTI writer, `ocean_forcing()` hook per step).
Constants at the top: `DOMAIN`, `LEVEL`, `SNAP_TIMES` (observation epochs; with
the inverse's sequence it reproduces `inverse_soln.nc`), `RESET_VELOCITY`
(zero the velocity warm start before each momentum solve; needed with the
pre-refactor solver settings, where the warm-started second 1 km step
diverged — with the example's omega 0.5 / damping 0.01 / post 150 the 1 km
warm start converges in 1-3 V-cycles per step), `BETA_MAX` (the example's
cap of 20), window overrides. It writes phi/psi/xi, basal velocity, q and
the TF anomaly too. Its `ocean_forcing(t, dt, mg, level, ctx)` hook applies
the parametric ocean forcing `q(x,t) = Q0 + ALPHA_Q * (TF_step(x,t) - TF_ref(x))`
from `model_inputs/thermal_forcing.nc` (`preprocessing/make_thermal_forcing.py`:
annual mean and max of the ISMIP7 EN4-based monthly TF, complete years 1950-2025 (the 2026 file is Jan-Feb only and skipped), nearest-
filled 10 km beyond the native product with `tf_dist` recorded): `TF_AGG`
mean/max over the years a step overlaps, `TF_REF` window (or scalar),
`TF_MAX_DIST_KM`, `Q_BOUNDS`; before the record q = Q0, after it the last
year holds; without the file the hook is a no-op. The driver now takes the
forcing from `config.ocean_forcing` through `glacier_inverse.ocean` (override
with `OCEAN = dataclasses.replace(...)`, baselines `Q0`/`H00`), so a driver
experiment and the inversion see the identical margins. **glide changes
2026-09-11: `calving.q` and the new additive margin `calving.h0` (m) are
CELL FIELDS** (like `sliding.beta`; scalar `.set()` fills them, restricted
by averaging), and `GlideStep` checkpoints both with the step. **2026-09-16: the
calving criterion is the gap-blended monotone law** (`calving.H_c`, config
`calving_H_c`, m; see the calving-law paragraph in the library section).
**Calving law (2026-09-16, monotone form).** The phi-blended pair of
criteria (grounded HAB law where phi -> 1, shelf minimum thickness where
phi -> 0) was NON-MONOTONE in H at tongue roots: a root cell below the
margin calved, thinned across flotation, was handed to the shelf branch
which protected it, refilled and calved again, so the implicit thickness
solve stalled at the V-cycle cap (94-100% of |r_H|^2 at tongue roots;
lagging, relaxation, softening, the exact Jacobian in the Vanka patches,
neighbourhood evaluation alone — all tried and rejected). glide now uses
ONE criterion psi = sigmoid(c r (H - H_calve)) with the critical thickness
blended by the gap between ice base and bed, `common.cu calving_F`:
H_g = (depth/r + h0)/(1 - q), H_s = min(H_c, H_g), gap = max(depth - r H, 0),
w = max(1 - gap / (r (H_g - H_s)), 0), H_calve = H_s + w (H_g - H_s) (the
cap, 2026-09-17, keeps H_c from acting in water shallower than H_c / r). The
linear ramp with that scale is the least-grounded-like blend for which
F = H - H_calve is non-decreasing in H (thinning never reduces calving):
grounded ice F = H - H_g; floating ice down to H_c - h0 has F = -h0 (the
margin's sign decides, no H dependence); thinner floating ice F = H - H_c.
So a positive margin removes floating ice, a negative one lets tongues
exist above H_c; H_c = inf is the old law exactly. Verified: psi(H)
monotone on random fields (0 violations); 2 km, 3-yr steps, H_c 200: 1
V-cycle on every step with the exact-flag |r_H| ~ 0.5 (the old law's
level) while 700-2900 tongue cells persist. (A 3x3 maximum-thickness
windowing of the flag was tried for partially covered front cells and
dropped: not needed for convergence.) The exact calving Jacobian (`state.dpsi_dH`, `state.dpsi_dbed`
from `compute_calving_flag`, central differences of F) WAS in the
residual / JVP / VJP operators and the bed gradient, with the Vanka patches
on the frozen-flag Jacobian; it was REMOVED 2026-09-17 because the 1 km
adjoint solve would not converge with it (psi is a ~1 m-wide switch, so
dpsi/dH is a near-delta), and every operator is on the frozen flag again
(see the adjoint-coverage paragraph). H_c sensitivity (2026-09-17, 2 km, H_c 100 vs 400): the model
keeps ~340 more grounded cells and ~0.05% more volume at H_c = 100, and at
FIXED cells 2-20 km upstream of tongues flows 7% slower and is 3% thicker —
a buttressing/front-geometry effect of thin (100-200 m) floating fringes
that survive on shallow (~-185 m) cold margins, not a numerical artefact
(grounded cells never see H_c in the law; the cap did not change it). H_c
is therefore a physical tuning parameter, "the thinnest floating ice that
survives", with real leverage; ~100-150 m is the observed shelf-front
range.

**Thermal-forcing fill (2026-09-17): `make_thermal_forcing.py --method marine`
(default; `nearest` = the original).** The ISMIP7 TF product's under-ice
values (Slater mapping) carry one sill-depth level of the profile and swing
by 1-1.5 K year to year as uniform blocks while the fjord trunk moves
0.3 K; the old nearest-neighbour fill copied single footprint-edge cells
(35% of them with zero months) into 5-10 km Voronoi discs with seams and
across land bridges. The marine fill trusts only OPEN-WATER native cells
(`bathymetry_mask` outside `rgi_mask`) and extends them HARMONICALLY
(Laplace, 8-neighbour stencil, one `splu` factorization per footprint)
through the connected marine domain (open water + ice cells with
`bed_obs < 0`; 400 k ice cells, the same reach as the mapping). `tf_dist`
is 0 on every reached cell so `max_dist_km` keeps its meaning;
`tf_path_km` holds the propagation distance; land and unconnected hollows
are NaN (baseline margins). Under-ice interannual std becomes the fjord's
(mean 0.42 K, 90th pct 0.43 vs 0.68 before). Native open-water zeros (lakes /
frozen bays in the product) remain. `analysis/compare_tf_fill.py` plots
both. `model_inputs/thermal_forcing.nc` (the inverse's and
forward_standalone's file) carries the marine fill since 2026-09-17; the old
product is kept as `thermal_forcing_nearest.nc`. `make_ismip7_forcing.py
--fill-method` defaults to marine, so the ISMIP7 records need a rebuild to
pick it up (done 2026-09-17 for ssp126 and ssp370).

**ctrl forcing (2026-09-18): `make_ismip7_forcing.py --scenario ctrl`** builds
the ISMIP7 control the way the kit builds its own (the kit's ctrl years are
bit-identical repeats of a climatology): the monthly tas / pr climatology
over `--ctrl-years` (2000-2029) of historical + `--ctrl-from` (ssp126) in OUR
product (dEBM2), written once to `ctrl_climatology.nc` on the native grid in
the kit's units, with every ctrl year of the catalogue pointing at it; the
anomaly reference (1986-2025) uses the real ssp126 years as for every other
run; the ocean is the kit's ctrl tf. Why not the kit's ctrl atmosphere: it
exists only as SDBN1 (CESM, 51 of 286 tas files on disk) / GEMB-SDBN1 (MRI),
neither with a historical to splice against, and SDBN1 is +1.0 K warmer than
dEBM2 in JJA over the ice, which would appear as a step at 2015. The dEBM2
ctrl sits +0.58 K (annual) / x1.04 precip above the 1986-2025 reference.

**OCX forcing (2026-09-18): CARRA2 year by year.** ISMIP7's observationally
constrained experiment wants reanalysis forcing; the consistent choice for a
model calibrated on CARRA2 T100 is CARRA2 itself, not the kit's RACMO 2 m.
`preprocessing/make_carra_yearly.py` runs `make_carra_vars.build_climate
(years=[y])` per year (1986-2025) and writes tas (K) / pr (kg m-2 s-1)
files in the kit's layout to `ismip7_data/CARRA2/ocx/carra2-1000m/`;
`make_ismip7_forcing.py --gcm CARRA2 --scenario ocx --atm carra2-1000m
--pre-years 1986 2015 --skip-ocean` builds `model_inputs/ismip7/CARRA2_ocx/`
(its `thermal_forcing.nc` is a symlink to the EN4 marine-fill file); the
climatology of the yearly files equals the model's `gridded_climate.nc`
to 1e-4 K / 1e-4 m/yr. `forward_projection.py --gcm CARRA2 --scenario ocx
--mode raw --pre-record standalone` feeds the raw fields + calibrated
biases from 1986 and forward_standalone's forcing (CARRA2 climatology +
0.6 x Vinther, no precip variability) before. Result vs the standalone run
(same checkpoint, 1850-2026, no elevation feedback; `analysis/output/
basin_mb_ocx/`, `basin_mass_balance.py` now reads compressed VTI via
`ismip_exporter.read_vti`): discharge identical (447 vs 450 Gt/yr, Mankoff
D+BMB 485), the interannual SMB PATTERN is captured far better (GrIS
correlation with Mankoff SMB 0.58 -> 0.96; per region 0.08-0.64 -> 0.88-0.97,
SE/CW/NW going from no skill to 0.94-0.97), but the AMPLITUDE is 1.6x too
large (GrIS SMB std 181 vs Mankoff 110 Gt/yr; regression slope 1.57;
dSMB/dT_jja -139 vs -80 Gt/yr/K on the same CARRA2 ice-mean JJA series;
2012 -185 vs +87, 2019 -252 vs +96) and the mean drops 60 Gt/yr (1986-2025
SMB 245 vs standalone 307, Mankoff 337; 2006-2025 166 vs 293), so the mass
balance goes from -144 (Mankoff -148) to -200 Gt/yr and the SMB trend
doubles (-56 vs -27 Gt/yr/decade). The mean forcing is identical, so the
mean shift is Jensen's inequality with the model's SMB curvature
(~-160 Gt/yr/K^2 implied) applied to the full 1.06 K interannual std
instead of the index's 0.6 K: the calibrated state is more melt-prone than
the real weather allows, and a too-melt-prone state also has the excess
sensitivity. Since both dh/dt windows (1992-2019, 2019-2026) lie inside the
CARRA2 record, the inversion should be forced with the yearly fields
directly (Vinther only before 1986) so the calibration and OCX see the same
weather — implemented the same day as library change 9
(`yearly_climate_filename`; the next inversion runs on it). Runs: `inverse/projection_CARRA2_ocx` (1800-2026, with
elevation feedback), `projection_CARRA2_ocx_nofb` (1850-2026, without; the
clean comparison). The checker has no OCX row, so its files need the
experiment table extended before submission.

**Snowline product (2026-09-18): `preprocessing/make_snowline.py` ->
`gridded_snowline.nc`.** `common_data/snowlines/<year>_snowline.zip`
(2000-2020) hold the MODIS-era end-of-summer snowline as polylines on a
500 m lattice, polar stereographic lon_0 0 / lat_ts 60 (the .prj's
latitude_of_origin; lat_ts 70 puts the vertices 400 m too high on the DEM).
Every part is a closed ring (664 in 2012, one 19,464 km long), i.e. the
boundary of the snow-covered region: nesting depth 1 = snow (1.49 M km2 at
1300-3050 m in 2012), 0 = bare ice / off ice, 2 = bare patches in the snow,
3 = snow in those — snow = odd depth, rasterized with rasterio's additive
merge on the 500 m lattice (the domain grid halved), averaged to 1 km and
divided by `ice_fraction` (clipped at 1). An elevation-based reconstruction
(DEM above the IDW snowline elevation of the nearest vertices) was tried
first and mislabels wedges beside the small rings and the low northern
interior; dropped. `snow_fraction` = mean of the 21 yearly labels (the
fraction of seasons ending with snow; time attrs 2000/2010/2020),
`glacier_fraction` = ice_fraction, 0 on the 128 k peripheral cells the
product never classified (rgi_periphery_fraction > 0.5, no ring within 5 km
in any year), `snow_label` (year, y, x) int8 %. Bare-ice area 243 k (2002)
to 354 k km2 (2019), 2012 342 k, mean 296 k; the ablation zone is up to
100 km wide in the SW. The Greenland config's SnowlineSpec uses
`window="file"` (library change 10), so the next inversion carries the
term — the run started 2026-09-18 (before the file existed) does not.
**First inversion with it (`inverse_v2`, yearly forcing + ONE-SIDED snowline
term, 2026-09-19; `analysis/output/basin_mb_v2/`)**: the old state's ELA was
~350 m too high against the product (model bare-ice area 291-347 k km2 vs
187 k on the common mask; P(snow) at 1200-1400 m 0.36-0.46 vs label
0.71-0.88), and that oversized ablation zone WAS the excess SMB sensitivity:
with the snowline term the interannual SMB matches Mankoff (std 117 vs 110
Gt/yr, regression slope 1.01, dSMB/dT_jja -74 vs -80, r 0.95, regional stds
within 10-25 %; was std 181 / slope 1.57 / -139). But the MEAN overshoots:
SMB 555 vs 337 Gt/yr, D 590 vs 485, MB -35 vs -148 (2006-2025 -87 vs -218),
bare-ice area 130 k (now too SMALL), P(snow) above the label in every band
over 200 m; tbias -0.5 K (-1.2 K at 800-1400 m), H_atm 14.8 -> 12.7, and
pbias UP 8 % (precip 944 Gt/yr vs CARRA2 raw 911, RCMs ~750). Cause: the
default `two_sided=False` scores only "snow observed, model bare", so with a
fractional label nothing resists snow below the snowline and raising precip
is free. The config is on `two_sided=True` since 2026-09-19 (the product
classifies both sides on the main sheet).
**`inverse_v3` (two-sided, 2026-09-19; `analysis/output/basin_mb_v3/`,
`analysis/smb_diagnostics.py` prints all of this per run)**: the snowline is
matched (bare ice 179 k vs 169 k km2, P(snow) within 0.02-0.07 of the label
in every band, mean bias -0.007) and the interannual SMB stays good (std 132
vs 110, slope 1.15, dSMB/dT_jja -89 vs -80, r 0.96, trend -29 vs -27
Gt/yr/decade). SMB mean 458 (Mankoff 337; v1 307), D 517 (485), MB -58
(-148). The state UNDERFITS ITS OWN dh/dt DATA: integrated over the basins
the products give -148 (ITS_LIVE 1993-2019) and -152 Gt/yr (ATL15
2019-2026), v1 had -179 / -127, v3 has -78 / -50 (v2 -58 / -17); the excess
is thin and interior (1400-2000 m +147 vs v1 +69 Gt/yr, > 2000 m +412 vs
+368; pbias x1.07-1.08 above 1400 m, precip 928 vs CARRA2 raw 911, RCMs
~750; north worst: NO+NE SMB 58 vs Mankoff 8). Why the loss allows it:
90 Gt/yr over the ice sheet is ~0.05-0.1 m/yr of dh/dt, a long-wavelength
offset that a per-pixel sigma 0.5 m/yr likelihood with a 10 km Matern
correlation barely sees, while the snowline term is 1.8 M cells at sigma_p
0.3. v1 only had the right total because its ELA was wrong in the
compensating direction. To do: a basin-integrated mass-change term (the
same dh/dt products summed per Mouginot region and window, sigma ~5-10 Gt/yr
per region for the altimetry's systematic error; `record_volumes_at`
already emits volumes) so the long-wavelength budget is weighted as what it
is. The interior pbias excess was the SNOWLINE TERM ITSELF, not the
geometry: with s_smb 0.5 m/yr the logistic cannot saturate on SMB 0.1-0.5
m/yr (P ~0.7 at label 1.00), so 63 % of the Brier sum and ~3/4 of the upward
pull on SMB came from above 2000 m, where only precipitation can answer.
**Rerun with `s_smb=0.1`, `sigma_p=0.5` (2026-09-20, written INTO
`inverse_v3`; the s_smb 0.5 forward run is kept as
`inverse_v3/forward_standalone_ssmb05` + `physical_fields_ssmb05.nc`,
`analysis/output/basin_mb_v3b/`)**: snowline fit unchanged (bare ice 183 k
vs 169 k, bias -0.009), pbias back to x1.00 above 2000 m (x1.02 overall,
precip 894 vs 928 Gt/yr), SMB 413 (was 458; Mankoff 337), D 495 (517;
485), MB -81 (-58; -148); windows -104 / -70 Gt/yr against the products'
-148 / -152. Interannual: std 135, slope 1.18, dSMB/dT_jja -94 vs -80 (a
little hotter than with s 0.5; 2019 bare ice 371 k vs 224 k). Remaining
excess ~75 Gt/yr: NO+NE SMB 55 vs Mankoff 8, CW+NW +30; the dry north still
sits within ~1 s_smb of zero, so `make_snowline.py --zone-km` (support
restricted to the snowline's neighbourhood; `gridded_snowline_zone20.nc` is
built, 600 k of 1.8 M cells) and the basin-integrated mass-change term are
the next two levers — superseded for the snowline part by the hinge
likelihood (library change 11; the config runs it into `inverse_v4`, the
zone file stays unused: modify the likelihood, not the product).
`forward_standalone.py` now re-exports
`physical_fields.nc` when the checkpoint is newer (it silently replayed
the previous state after a rerun into the same results_subdir).

**Reanalysis switch (2026-09-20): `CLIMATE = "carra2" | "racmo"` at the top
of `domains/greenland/config.py`.** No library change: the two forcings are
twin input files selected by name (`gridded_filename`,
`yearly_climate_filename`), and `results_subdir` is `inverse_v5_<CLIMATE>`.
`preprocessing/make_racmo_vars.py` builds `gridded_climate_racmo.nc` and
`GLIDE_inputs_racmo.nc` (= GLIDE_inputs.nc with `monthly_t2m` /
`monthly_precip` replaced by the RACMO2.3p2-ERA5 1986-2025 climatology - the
CARRA2 window, so the Vinther index keeps its zero-mean reference);
`make_climate_yearly.py --source racmo` builds
`gridded_climate_yearly_racmo.nc` (1958-2025, 68 years, 15.8 GB of int16
codes; anomalies exactly zero-mean over 1986-2025, 0 capped ratios). RACMO is
on the domain grid (y flipped, coordinates asserted) and covers 100 % of the
main sheet but 66 % of the peripheral ice and 58 % of the domain mask:
outside the footprint the climatology is CARRA2's field + the nearest
covered cell's RACMO - CARRA2 offset (precip: ratio; `racmo_fill_distance`
in km), yearly anomalies are the nearest covered cell's. `tas` is the 2 M
temperature, not the 100 m level of the CARRA2 forcing (pinned near 0 degC
over melting ice), so H_atm / tbias will recalibrate - part of what the
experiment measures. Main sheet, 1986-2025: precip 714 vs CARRA2 824 Gt/yr
(x0.93 below 1200 m, x0.86 at 1200-2000 m, x0.85 above), JJA -6.73 vs
-5.59 degC (annual -19.59 vs -16.09): the precip gap alone is the size of
inverse_v4's SMB excess over Mankoff. Level-2 check 1950-2026: fwd + loss +
bwd 14 s, 11.2 GB peak VRAM, the 68-year record pinned in host RAM
(`yearly_climate_cache="none"` reads it from the page cache instead).
`forward_standalone.py` follows `config.gridded_filename`, so the forward
run after the inversion uses the same forcing. Not yet run.
**`CLIMATE = "hybrid"` (the config's setting since 2026-09-20): CARRA2
temperature + RACMO precipitation**, `preprocessing/make_climate_hybrid.py`
-> `GLIDE_inputs_hybrid.nc` (GLIDE_inputs with `monthly_precip` from the
RACMO file) and `gridded_climate_yearly_hybrid.nc` (9.3 GB, 1986-2025: the
CARRA2 file's `t2m_anom` codes + the RACMO file's `precip_ratio` codes, both
relative to their own 1986-2025 climatology with the same scale factors, so
a pure copy). This is the CONTROLLED experiment: against inverse_v4 only the
precipitation differs - melt physics, the 100 m temperature level and
glare's rain/snow split stay on CARRA2, whereas "racmo" also swaps in the
2 m temperature. The mix keeps the T-P covariance because both products
are ERA5-bounded: ice-mean annual precip ratios correlate at r = 0.97 over
the 40 years (std 0.095 vs 0.087). Level-2 check 1980-2026: 12 s, 7.3 GB.
Results go to `inverse_v5_hybrid`; compare with `v4_hinge`.
**Result (`inverse_v5_hybrid`, 2026-09-20; `analysis/output/basin_mb_v5/`)**,
v4 -> v5 (Mankoff): SMB 453 -> 380 (337), D 512 -> 453 (485), MB -58 -> -73
(-148); windows -81 / -46 -> -102 / -60 (products -148 / -152). The melt
side did not move (tbias -0.36 -> -0.41 K, H_atm 13.3 -> 13.2, f_clear 0.41,
pbias x0.99, snowline bias -0.002, bare ice 172 k vs 171 k km2), so the
experiment is clean, and the interannual SMB is the best yet (std 117 vs
110, slope on Mankoff 1.01, dSMB/dT_jja -83 vs -80, r 0.95). Two readings:
(1) ~2/3 of v4's SMB excess over Mankoff (73 of 116 Gt/yr) WAS the choice of
reanalysis; interior SMB above 2000 m 0.32 m/yr (v4 0.41, SDBN1 0.37 / 0.28).
The remaining +43 is a regional dipole, not a uniform offset: NO 17 vs -2,
NE 35 vs 10, SW 33 vs 15, CW 69 vs 47 too high (+88) against SE 104 vs 132
and CE 60 vs 78 too LOW (-46) - RACMO's SE precipitation is below the
three-RCM mean Mankoff uses. (2) The MASS BALANCE barely moved: taking 73
Gt/yr out of the accumulation took 59 out of the discharge (now BELOW
Mankoff's), because the inversion balances D against whatever SMB it is
given - the ice-sheet-wide dh/dt signal is too weakly weighted to hold the
budget (-73 vs -148). So the MB deficit is not a forcing problem: the
basin-integrated mass-change term is what it needs. Also: yearly bare-ice
area still too variable (std 77 k vs 30 k, 2019 425 k vs 226 k, r 0.40).

**ISMIP7 projections (2026-09-16): `forward_projection.py` +
`preprocessing/make_ismip7_forcing.py`.** The projection driver imports
`forward_standalone` (whose `setup(level=, out_dir=, ocean_loader=)` now
exposes the SMB internals on the returned `Run`) and swaps only the three
forcing records for the ISMIP7 kit in `ismip7_data/<gcm>/<scenario>/`:
dEBM2-1000m `tas` (2 m, K) / `pr` (kg m-2 s-1) and ocean-1000m `tf`, one
monthly file per year on the ISMIP grid (y ASCENDING; the domain is y
descending, so `flip_y`). The preprocessing splices historical (1850-2014)
+ scenario (2015-2300) into `model_inputs/ismip7/<gcm>_<scenario>/`:
`catalogue.json` (year -> files; the yearly fields are NOT copied, the
driver reads them from the kit per step), `climate.nc` (monthly
climatologies: `tas_clim`/`pr_clim` over the CARRA2 window 1986-2025 for
the anomaly mode, `tas_pre`/`pr_pre` over 1850-1879 for the constant
pre-record forcing, ice-mean bias attrs vs CARRA2) and `thermal_forcing.nc`
**ANOMALY REFERENCE, fixed 2026-09-23 (`--clim-scenario`, default
`CLIM_REF_SCENARIO` = ssp126 for ssp runs, self otherwise).** `clim_years`
1986-2025 STRADDLES the historical/scenario splice at 2014, so the tail used
to come from whichever scenario was being built -- and the ssps are separate
realizations, differing there by internal variability, not only by forcing.
CESM2-WACCM `tas_clim` ssp126 vs ssp370 differed by -0.137 K in the ice mean
and up to 1 K locally. In anomaly mode
`t2m = t2m_clim + (tas - tas_clim) + tbias`, so a colder reference WARMS the
forcing for the same historical `tas`, and the three scenarios of a GCM
diverged BEFORE 2015 -- ~800 Gt by 1985 in the v7 projections, visible as
the pre-2015 spread in `analysis/output/basin_mb_projections/`. The catalogue
files and the TF are identical over 1850-2014 (checked: max |dTF| 0.0000 K),
so the reference was the only cause. The tail is now taken from ONE scenario
for every scenario of a GCM. Shrinking the window to end at 2014 instead
would NOT work: it has to stay aligned with the model's own CARRA2
climatology or the anomaly picks up the warming between the two windows.
`climate.nc` records `clim_scenario`; `--clim-scenario self` restores the old
behaviour. Products rebuilt 2026-09-23 for CESM2-WACCM / MRI-ESM2-0 ssp370
and ssp585 (ssp126 IS the reference, so it was already correct and its file
is unchanged). Projections run before that date carry the old reference.
(annual mean/max TF for all 451 years in `make_thermal_forcing.py`'s layout,
streamed year by year and blanked > 30 km from the ice; `OceanForcing.from_file(lazy=True)`
reads it per year). The run is seamless 1800-2300: 5-yr steps on the
pre-record climatology, annual steps from 1850 (`DT_SCHEDULE`), last year
held after 2300. `CLIMATE_MODE="raw"` feeds the dEBM2 fields (nearest-filled
onto the 2.7% of ice cells outside its footprint, + tbias / exp(log_pbias)
when `APPLY_BIASES`); `"anomaly"` adds the ISMIP7 departure from its own
1986-2025 climatology to the calibrated CARRA2 climatology (pr as a ratio).
The ocean forcing keeps the config's `OceanForcingConfig` on the CESM2 TF
(TF_clim = CESM2's own 1950-79 mean). Outputs: `scalars.csv` per step
(volume, VAF in mm SLE, areas, SMB in Gt/yr, forcing ice-means, dTF, wall
time), `snapshots.nc` every 10 yr, `final_state.nc`, optional VTI.
`--continue` (2026-09-18) resumes a finished run from `final_state.nc`
(full-precision H; velocities restart from zero, one solve's worth of extra
V-cycles) to the given `--t-end`, appending to scalars.csv / snapshots.nc /
the VTI series (numbering and .pvd carried on) and rebuilding the elevation
feedback's reference surface from the VTI frame at FEEDBACK_T_REF (refuses
without one unless `--no-elevation-feedback`); `continued_from` is recorded
in the attrs. Used 2026-09-18 to extend all six projections (CESM2-WACCM and
MRI-ESM2-0 x ssp126/370/585) from 2300 to 2301 for the ISMIP7 window. Bias
found 2026-09-16 (1986-2025, ice cells in the dEBM2 footprint): dEBM2 tas is
3.5 K colder annual / 1.8 K colder JJA than the CARRA2 100 m forcing and its
precip 12% lower (0.48 vs 0.55 m/yr); the pre-industrial (1850-79) CESM2
climate is another 1.7 K (annual) / 1.5 K (JJA) colder with 10% less precip.

## Greenland-specific conventions

- **Grid**: EPSG:3413 everywhere (`preprocessing/projection_dictionary.py`).
  `preprocessing/domain_grid.py` resolves `local_data/domain.json` — the
  `ismip_greenland` preset puts cell centres on the ISMIP nodes (x −720…960 km,
  y −3450…−570 km) so 1/2/4/8 km nest; `bbox` sub-domains snap onto that
  lattice. Every builder regrids with `rio.reproject_match` onto
  `DomainGrid.template()`; raster edges are half a cell outside the centres.
- **Field names are inherited from Alaska** so the library is untouched:
  `rgi_mask` is the BedMachine ice mask, `rgi_label`/`rgi_id` are Mouginot &
  Rignot drainage basins followed by RGI-7 peripheral glaciers, `surge_type`
  is 0 for basins. New geometry variables: `ice_fraction`, `floating_mask`,
  `thickness_obs`, `bed_obs`, `bed_obs_err`; velocity carries `vx_err`/`vy_err`
  (not yet consumed).
- **Time**: the ice sheet is seeded from observed geometry
  (`init_from_observed_geometry=True`) and integrated over a short historical
  window (1800–2015 nominal, extended to the last observation epoch 2026)
  rather than grown from ice-free over 1000 years. `dt` is the spin-up step;
  `dt_schedule=((1990.0, 3.0),)` refines the observational period to <= 3-yr
  steps, split EQUALLY between consecutive observation epochs (1990, 1992,
  2008, 2015, 2018, 2019, 2026) so no sliver steps appear
  (`scheduling.build_step_sequence(dt_schedule=)`; the driver uses the same
  builder). `grad_start_time` is unset until an FD sweep on level 3 says
  where the adjoint can be truncated.
- **Physics from glide's Greenland example** (post 2026-09 refactor):
  `rho_water=1028`, `A_glen=1e-17`, `beta_init=2.5`, `water_drag=1e-3` (the
  example's 1e-4 leaves ice-free ocean cells dragless and stalled the
  velocity-epoch adjoint solve — the one deliberate deviation), `H_reg=25`,
  `thklim=1`, signed `depth=-bed` with `sigmoid_c=1` (phi is 1/2 exactly at
  flotation), height-above-buoyancy calving `calving_timescale=0.5` yr,
  `calving_q=-0.5` (only ice thinner than half its flotation thickness
  calves, so tongues persist; the example's inverse uses timescale 0.1),
  MOLHO stress scheme, FAS 200/10/150/0 with Vanka omega 0.5, momentum
  damping 0.01, step tolerance 1e-6 on both solvers. Priors have ice-sheet correlation lengths
  (bed 4 km, log β 8 km, pbias/tbias 50 km, SMB fields 150 km). Learning rates
  are copied from the tuned Alaska domains and are **untuned** here.
- **Observation epochs** (variable attrs `time_nominal/start/end`, read by
  the specs; verified 2026-09-11): BedMachine surface takes the file's
  `nominal_year` (2008 in v6; GIMP DEM 2003-2009), mask 2015 (2007-2019; the
  BedMachine metadata does not date it — the Jakobshavn front sits ~11 km
  up-fjord of the 2000 position, so it is post-2008), ITS_LIVE summary
  mosaic **2018.0** (the stored `vx` is the offset of a 2014-2024 line fit
  with a 2018-01-01 intercept, NOT the window midpoint; `dvx_dt` carries the
  slope), dh/dt trend windows: the requested `--t0/--t1` when given, else
  the kept epochs' extent rounded to 1e-2 yr (ATL15's axis starts 6 h into
  2019; unrounded it inserted a 6-hour dynamics step) — ATL15 2019.0-2026.0
  and MEaSUREs G1920V01 1992-2019. Resulting schedule: 20-yr grid to 1980,
  then 1992, 2000, 2008, 2015, 2018, 2019, 2020, 2026. The ArcticDEM overlay (nominal
  2015) is preferred for the surface when present.
- **No debris term, no ETIM path in use, no ERA5-Land by default.**
- **Transient temperature forcing is the Vinther SW-Greenland station series,
  not a scaled global anomaly** (`make_temperature_anomaly.py --source vinther`,
  the default when `climate/temp_anomaly/swgreenlandave.dat` exists): JJA mean
  anomaly 1784-2013, extended to 2025 with the CARRA2 100 m temperature over the
  SW basin (regressed on the stations over 1986-2013: slope 0.77, r 0.83), zero
  mean over the CARRA2 climatology years, gaps interpolated.
  **Before the stations the series is GISP2 (2026-09-22)**, not a constant:
  `--deep-source gisp2` (the default when
  `climate/temp_anomaly/gisp2-temperature2011.txt` is present; `none` restores
  the old flat hold) scales the Kobashi et al. (2011) Summit argon-nitrogen
  temperature onto the station series and crossfades it over 1784-1840. One
  variance-matched slope over the 1784-1950 overlap, both sides low-passed at
  `--deep-smooth` (31 yr): the gas thermometer integrates the firn temperature
  gradient, so its annual increments are 0.085 K against a record std of
  0.98 K and it resolves only multi-decadal variability. Variance matching,
  NOT least squares -- OLS attenuates by r and would return too warm an LIA,
  and it is the forcing AMPLITUDE the model consumes. slope 1.16 (11 yr,
  r 0.58) / 1.12 (31, 0.76) / 1.06 (51, 0.90) / 0.95 (101, 0.97); the decline
  as the filter loosens is residual firn attenuation, and bandwidths leaving
  fewer than three independent samples are refused. The choice barely
  matters: over s = 1.06-1.16 the Common Era index moves only -0.54..-0.46 K
  because the anchor is a measured station mean and the Common Era sits
  ~0.6 K from it. Resulting epoch means (K vs the CARRA2 window):
  Common Era -0.51, Roman (1-500) 0.00, Medieval (900-1200) -0.35, LIA
  (1450-1850) -1.47, LIA core (1600-1800) -1.77; the old flat hold was
  -1.20 K for EVERY pre-1785 year. Blend window = exactly where the stations
  are mostly interpolated (9/16 of 1784-99, 8/20 of 1800-19, 19/20 of
  1820-39 missing; continuous from 1840), so it replaces a 19-year linear
  interpolation rather than discarding observations. The published file
  carries a 2.75 K step at 1730 BCE (plus 1.08 K the next year) with
  everything older ~3 K colder -- a section boundary in the DT integration,
  not climate -- so `DEEP_RECORD_START` drops everything before 1000 BCE and
  a `DEEP_MAX_STEP` check warns about anything that abrupt surviving inside
  the used range. The time axis now runs -1000..2025 (`forward.simulate`
  looks the anomaly up by year in a dict and clips only at the top, so a step
  before the first year would raise rather than saturate). 1840-2025 is
  bit-identical to the pre-GISP2 file. **Impact scales with how deep the
  spin-up goes** (alpha_t2m 0.6, dSMB/dT_jja -83 Gt/yr/K): t_start 1700
  +44 Gt/yr and +2 m of ice-sheet-mean thickness over the spin-up (the old
  hold was too WARM for 1700-1784, the LIA core), t_start 1000 -13 Gt/yr and
  -7 m, t_start 0 -38 Gt/yr and -44 m, t_start -1000 -46 Gt/yr and -82 m. So
  it is inert for the current configuration and first-order exactly when the
  spin-up is lengthened.
  The configs use `base_anomaly_year=None` and `alpha_t2m=0.6`: RACMO
  tas JJA regressed on the Vinther JJA series gives 0.58 ice-sheet mean
  (0.51 CE .. 0.70 SW, r 0.6-0.8, and the fit is stable across halves of
  the record), whereas HadCRUT explains almost nothing of
  Greenland's interannual-to-decadal history (r 0.1-0.4; the 1930s-40s warm
  and 1970s-90s cool periods are absent from a global series). The global
  PAGES2k+HadCRUT splice remains available as `--source global` (then
  `alpha_t2m` ~1.3-2 and a base year are needed).
- **Precipitation follows the same index (2026-09-22)**, where before it did
  not: `make_precip_anomaly.py` was an inert placeholder waiting on an
  accumulation-reconstruction CSV that is not on disk, so no
  `precip_anomaly.nc` existed, `alpha_precip` stayed 0 and
  `forward.simulate` set `precip_multiplier = 1` — accumulation sat at the
  modern climatology through the whole spin-up while only the temperature
  half of a colder past was applied. The script now falls back to scaling the
  index, `R(t) = exp(gamma_ann * dTann_dindex * I(t))` normalized to 1 at
  `base_precip_year`, and the config sets `alpha_precip=1.0`,
  `base_precip_year=2006`. NO library change: this is the Alaska
  `precip_anomaly` pathway, and `_term_forcing` applies the multiplier to
  INDEX years only (record years take `precip_ * yearly.precip_ratio(year)`),
  so it cannot double-count with the reanalysis forcing. `alpha_precip`
  multiplies gamma to first order (1 + a(e^x - 1) ~ e^{ax}), so sweep it from
  the config instead of rebuilding. The CSV path still takes precedence when
  the file appears, with years outside its span filled from the index scaling
  rescaled to the reconstruction's own mean (a dict lookup below the series
  would raise, and a constant would reintroduce the flat-hold bias).
  **gamma is a PRIOR, not a fit**: `--measure` regresses the ice-sheet
  precipitation-weighted annual ratio on temperature over 1986-2025 (hybrid)
  and gets +2.3 +- 1.2 %/K on ice annual T (r 0.31), +1.3 +- 1.4 on the
  index, -0.2 +- 1.2 on JJA — all indistinguishable from zero, because
  interannual Greenland precipitation is circulation-driven, exactly as
  Kapsner et al. (1995) found for Holocene GISP2 accumulation while
  glacial-interglacial accumulation tracks temperature. Fitting that slope
  would silently switch the response off. The default 5 %/K is the modelling
  convention, below the Clausius-Clapeyron ceiling (7.3 %/K at 273 K, 9.6 at
  253 K, a saturation bound not a precipitation sensitivity) and matches the
  3-5 %/K from ice-core accumulation across the glacial transition; the
  40-yr record neither supports nor excludes it. `dTann_dindex` = 0.73
  (measured, r 0.66; JJA 0.92, r 0.80 against `alpha_t2m` 0.6), so 5 %/K of
  ice annual T is 3.65 %/K of index. Epoch multipliers and the ice-sheet
  cost at 814 Gt/yr: Common Era x0.984 (-13 Gt/yr), Roman x1.002 (+2),
  Medieval x0.990 (-9), LIA x0.950 (-41), LIA core x0.940 (-49). Mean over
  the index years (t_start..1985) at 5 %/K: t_start 1700 -43 Gt/yr, 1000
  -29, 100 -17, 0 -15, -1000 -10. Unlike the temperature record this bites
  at SHALLOW starts too, because 1700-1985 sits in the Little Ice Age.
  `forward_standalone.py` applies the same multiplier (`alpha_precip_eff`,
  `smb_index(shift, precip_multiplier)`, the identical weighted-mean-over-
  index-years / base_precip arithmetic, zeroed by `climatology_only`), so
  replays and `forward_projection --pre-record standalone` stay consistent
  with the inversion — without it a replay would run wetter than the
  inversion everywhere before 1986.
- **Climate forcing is CARRA2 at height levels, not 2 m** (`make_carra_vars.py`):
  the 100 m above-ground temperature is the forcing air temperature (the 2 m
  field already carries the melt-depleted boundary layer), moved onto the DEM
  with the LOCAL monthly lapse rate from the 100/500 m pair and CARRA's own
  orography; precipitation is CARRA `tp` (mm/day → m ice/yr; its stamps are
  12 UTC on the last day of the PREVIOUS month and are shifted by 12 h before
  the calendar-month grouping — without that every precip month lands one
  month early). The full record (`climate/carra2/1985_2025/`, 483 months
  1985-10..2025-12, 32 GB decompressed per variable) is read lazily, cropped
  to the finite Greenland box and accumulated in 24-month blocks; the
  climatology pools the complete calendar years 1986-2025
  (`climatology_years` attr; `--years` overrides). The 27-month files in
  `climate/carra2/` are the fallback when the full-record directory is absent. The old 2 m
  `make_pancarra_vars.py` is kept only for Alaska-style files; the parametric
  placeholder (`make_climate_parametric.py`) is the bootstrap fallback only.
- **Data versions on disk** (2026-09-08): BedMachine v6.6 (`nominal_year` 2008
  used for the surface epoch; has `rgi` flag, no mask code 4), RGI 6.0 region
  05 (`RGIId`/`Surging`/`Connect`; Connect 2 stays with the basin), Mouginot &
  Rignot basins v1.4.2, ATL15 v5 (2019-2026), ITS_LIVE Greenland elevation
  change G1920V01 (`--source itslive_dh`, 1992-2023), ITS_LIVE V02.1 120 m
  summary velocity mosaic RGI05A (the default velocity source since
  2026-09-10; `--velocity-source glide_example` = the MEaSUREs 1995-2015
  mosaic inside glide's h5 remains the fallback).
- **`common_data/` is self-contained**: no symlinks to alaska-forecast (a
  shared `pancarra/` tree let the Greenland precip download overwrite the
  Alaska one). CARRA2 lives in `climate/carra2/{t,precip,orog}.nc`, the
  temperature-anomaly inputs are copied into `climate/temp_anomaly/`.
- **RACMO2.3p2-ERA5 1 km (ISMIP7 kit)** sits in `common_data/climate/RACMO2.3p2-ERA/`
  (1958-2025, native ISMIP 1 km grid, `y` ascending; `tas` = 2 m temperature,
  `ts` = the same CLIPPED at 0 degC, `pr` kg m-2 s-1; the analyses use `tas`). It is not a model input; `analysis/`
  holds the intercomparison with CARRA2 (`compare_racmo_carra.py`) and the
  Arctic-amplification / hybrid-reconstruction script
  (`arctic_amplification.py`, outputs in `analysis/output/`, gitignored;
  `--hybrid-years` can write pre-1958 yearly files in the RACMO layout to
  `common_data/climate/RACMO2.3p2-ERA-hybrid/`; ~100 MB per year, not written by default). Findings 2026-09-09: CARRA2
  T100-on-DEM is 4-8 K warmer than RACMO t2m in the winter interior (the
  inversion), 1-2 K in summer; CARRA2 precip is 15% higher over the ice sheet
  (25% in SE); ice-sheet-mean amplification vs HadCRUT5 is ~1.3 K/K annual
  (JJA 1.4, DJF 0.75, SON 1.7; north 1.5-1.8, south ~0.9), below the
  config's scalar `alpha_t2m=2.0`; the regression is weak (r ~0.5 annual)
  and non-stationary (1958-1991 Greenland cooled while the globe warmed).
- **Python environment**: run everything with `~/Source/glide_test_env/bin/python`
  (glide/glare/ggapp/gtic with the diffuse-sky insolation); the `working`
  env's gtic predates `diffuse_potential_monthly`.
- `make_insolation.py` anchors solar geometry at the grid centre in
  `America/Nuuk`; for the whole ice sheet the hour-angle error at the E/W
  margins is a few degrees (gtic takes one lon). Acceptable for now; a
  per-column longitude in gtic is the fix.
- **DEM anti-aliasing (2026-09-25): `make_dem.py --smooth-sigma-km 1.0`
  (default; `make_all.py --dem-smooth-km`).** A_glen 1e-16 (the user's
  soft-ice test) exposed solver aliasing that the stiff rheology had
  smoothed through the physics, at the ice / ice-free boundaries in
  extremely steep terrain. The filter is applied to the NATIVE BedMachine
  `surface`, `bed` and `thickness` (150 m; sigma 6.67 px) BEFORE the
  area-average `reproject_match` to 1 km, and to the ArcticDEM overlay at
  its own 100 m (NaN-aware normalized convolution) when that file is
  present -- NOT to the 1 km composite afterwards. Rationale (user): the
  model resolves nothing below 1 km anyway, so a 1 km filter at native
  resolution is a genuine anti-aliasing filter -- no spurious flotation or
  steep topography is lost that the 1 km grid would not have lost -- and a
  post-resampling filter masked to the ice cannot reach the boundary,
  which is exactly where the problem is (that version, mask-aware with a
  50 m edge cap, was built first the same day and replaced). One linear
  kernel on all three fields keeps S - bed_obs - H = 0 exactly (checked);
  `errbed`, the radar-pick subcell products (`bed_obs_radar*`,
  `bed_radar_fraction`), `dataid` and the mask fractions are not smoothed
  (data stay data; the picks condition the prior at their own values).
  Numbers vs the unsmoothed build, Laplacian rms of the 1 km `elevation`:
  interior (10 cells inside the mask) 14.6 -> 4.5 m, within 2 cells of the
  mask edge 300 -> 133 m; `bed_obs` interior 59 -> 24 m, edge 304 -> 112;
  off-ice `topography` 275 -> 104 m (mountains, as intended). Per-cell
  change on ice: median 0.6 m, 90th pct 17 m, 99th 142 m, max 921 m
  (surface); `bed_obs` median 2.8 / 51 / 191 m. Cells with 0.917 H +
  bed_obs < 0 on the 1 km grid: 5296 -> 5164. ArcticDEM was NOT present at
  `../common_data/dem/arcticdem/` on this machine, so the surface is
  BedMachine's (GIMP, nominal 2008) as before. `--smooth-sigma-km 0`
  restores the old DEM. `tools/patch_dem_into_inputs.py` pushes
  `elevation`, `topography`, `bathymetry`, `bed_obs`, `thickness_obs`
  (its default `--vars` now) from a rebuilt gridded_dem.nc into the
  existing GLIDE_inputs*.nc in place (netCDF4, seconds; a `.bak` copy is
  made once and KEPT, so `.bak` stays the original through repeated
  patches; make_all after build_dem re-runs everything up to the merge) --
  done 2026-09-25 for GLIDE_inputs / _hybrid / _racmo (attr `dem_patched`;
  the `.bak` files are the pre-smoothing originals). Consequences: every
  run from now on seeds H and the initial bed (`bed_obs` where finite,
  library change 4) from the smoothed fields, the surface term targets the
  smoothed surface and the model depth = -bed is the smoothed bathymetry;
  the v9 checkpoints (beta, whitened bed) are unaffected, so a standalone
  replay of v9 differs slightly from the inversion's own state; a
  re-inversion picks the new DEM up everywhere. `make_dem.py`'s default
  BedMachine path is one directory too high for this machine
  (`../common_data/...`); pass `--bedmachine common_data/geometry/
  bedmachine/BedMachineGreenland-v6.nc` (~6 min). **COST (2026-09-26)**:
  the observed 1 km gate-flux reference (`basin_mass_balance.py --gates`,
  mosaic x `thickness_obs` under the gate pixels) falls from 391 to 359
  Gt/yr, because the gates cross exactly the troughs a 1 km filter
  shallows: SE 110 -> 98, CE 63 -> 56, NW 100 -> 94, CW 48 -> 45, SW 15 ->
  13, NO / NE unchanged; gate thickness Helheim x0.92, Store x0.92,
  Daugaard-Jensen x0.89, Anorituup x0.80, Kanger x0.96, Upernavik N x0.96.
  The 1 km reference now recovers 73 % of Mankoff's 491 instead of 80 %;
  read a model gate number against 359 for runs seeded from the smoothed
  DEM (v10 on), against 391 for earlier ones. The velocity term is
  unchanged (the mosaic is not smoothed), so a model matching the observed
  speed over the smoothed trunk carries ~8 % less flux at the gates than
  before; the user's smoothed-DEM v10 trunks are 1.1-1.25x the smoothed
  thickness at the gates anyway (spin-up thickening), so the seeding is
  not the binding constraint. **The last holdout after the smoothing
  (user, 2026-09-26: at A_glen 5e-17 the dt = 25 spin-up stalls in |r_H|
  at O(10), traced to cells (1405, 458-459), Umiammakku fjord at x -254,
  y -1975 km, with ~30 000 m/yr in two cells; dt <= 10 converges, a
  larger water_drag helps).** Anatomy (v10 tau-1 replay, 2020 frame;
  scratch `cell1405.txt`): a 770-950 m trunk in a -450..-580 m trough
  (xi 0.33-0.36, beta 1.2-2.2, surface 320-400 m) against a fjord wall
  whose cell (459) is ice-free rock (rgi 0, inactive, H = thklim, outside
  the pin: h0 +250) at a bed of 848 m (smoothed from 1102; the neighbour
  1372-1750). The driving-stress stencil (`get_tau_dx_jac`) gives that
  rock cell a GHOST surface S = bed + thklim = 849 m, so the wall facet
  458|459 carries H_avg (S_r - S_l) / dx = 385 x 528 / 1000 = +203 in
  glide units, 5x the trunk's own facets (41-44) and pointing INTO the
  trough: rock pushing on ice. The facet drag is the arithmetic mean of
  the two sides, 0.5 (beta xi)_ice + 0.5 (beta)_rock = 0.5 x 2.2 + 0.5 x
  20 (the inversion drove the wall cells' beta to 20-31, the cap: it is
  fighting this push), and a Weertman m = 1/3 balance against 203 gives
  u = (203 / 11)^3 ~ 6 000 m/yr on the facet with only the membrane
  viscosity holding it below that; at A 1e-17 it sits at 450 m/yr, at
  5e-17 the viscosity no longer holds and the facet runs to 30 000, and
  the implicit transport at Courant 750 (dt 25) cannot converge. Census
  of ice / inactive facets ice-sheet-wide: 8 448 x- and 8 806 y-facets
  with a rock ghost surface ABOVE the ice (median step +56 m, 95th pct
  +220), of which 6 carry tau_d > 100 and only THIS one has both a big
  step (528) and a low-drag ice side (xi 0.36; the others sit at xi
  0.74-1.0, beta_eff 8-20 and hold) -- hence the last holdout; the ice
  front-into-water facets have ghost steps ~0 and are fine. **FIXED IN
  THE STENCIL, glide `cuda/stress.cu` (2026-09-26, working tree on top of
  the user's checkpoint commit 9d486d8; uncommitted): rock cannot push
  ice.** In `get_tau_dx_jac` / `get_tau_dy_jac` the part of each cell's
  BASE that stands above the OTHER cell's SURFACE, e = base - S_other, is
  removed from that cell's surface before the step is formed, S_eff = S -
  clip(e), so only the ice standing ON a terrace drives the facet (a wall
  gives H_avg (H_upper + d / 2) / dx instead of H_avg x relief / dx).
  Physics: in the hydrostatic finite-volume balance of a stepped bed the
  rock face's reaction balances the lower column's own pressure, and rock
  above the lower column's surface touches no ice; the old form let the
  bed-slope term act over the full relief. `tau_d_clip` is ONE-SIDED and
  C1 -- exactly 0 for e <= 0, e^2 / 2d to e = d, e - d / 2 beyond, d =
  `TAU_D_CLIP_SCALE` 10 m -- with the exact Jacobian (H and bed) through
  the same function, so residual, Vanka patch, JVP, VJP and the bed
  gradient stay consistent; no mask, no new kernel arguments. Overlapping
  columns, floating ice and calving fronts (bases <= 0 <= surfaces) are
  untouched to the last bit, and a gentle slope is untouched too: the
  clip acts only where a bed rises above the neighbour's ice SURFACE,
  which is the user's cliff-vs-slope distinction made by the geometry
  itself (no-penetration rows were not added for that reason). TWO
  VARIANTS FAILED and should not be retried: (1) keying the clip on the
  ACTIVE-SET mask (the first implementation) makes the driving stress
  jump when the smoother flips a cell, and glide's own `grad_test`
  forward solve stalled at |r| / |r0| 0.9 (HEAD 6.8e-3); (2) a two-sided
  softmin(base, S_other) with a 10 m scale diverged to NaN in the first
  1 km step -- its tail of a few metres is as large as the freeboard of
  thin floating ice and multiplied the front stress of every apron cell.
  A hard `if` (no smoothing) cured the wall but left the cold-start step
  cycling at |r_H| 520-680 with period 3 (cells toggling across the
  switch while the seeded geometry adjusts). TESTS (`analysis/output/
  wall_test/`: `wall_test.py` = four dt = 25 steps from t = 100 at 1 km
  with the config as it stands, A_glen 5e-17, water_drag 1e-4, tau 1,
  tolerances 1e-2 / 10; the baseline ran the same script on a `git
  archive` export of HEAD through PYTHONPATH, the working tree untouched):
  HEAD reproduces the report -- wall facet -36 566 m/yr, cells (1405,
  458-459) at 18 316, the fastest in the domain, V-cycles 4 / 10 / 10 /
  10 with |r_H| 45 / 33 / 25 / 23 and steps 2-4 never meeting the
  tolerance; with the clip the facet is at -4 m/yr, the wall-adjacent
  cells at 9-32, V-cycles 10 / 2 / 2 / 2 with |r_H| 102 / 5.8 / 1.9 / 1.5,
  every step converged (the cold-start step costs 6 more cycles, the
  rest 8 fewer each), and the rest of the domain is unchanged (the
  fastest cells are the same SE outlet at 11.5 km/yr, 14 cells above
  10 km/yr in both). glide's tests: jvp 0.3 %, grad 0.85 % (HEAD 0.64 %,
  the forward solve there is rounding-sensitive at its 40-cycle cap),
  grad_bed 1.19 as at HEAD -- that test ALREADY FAILS at HEAD (the xi
  terms commented out 2026-09-18), not a regression. Not done: the facet
  drag is still the arithmetic mean of the two sides, and the inversion
  has not been rerun -- the wall cells' beta of 20-31 was calibrated
  against the old push and is now holding nothing. **FOOTPRINT OF
  THE CLIP (independent check, 2026-09-26, v10 tau-1 replay, 2018 frame):
  it is NOT confined to walls.** A base stands above the neighbour's
  surface wherever the bed step per cell exceeds the ice thickness, and
  the geometry cannot tell a staircase from a smooth steep slope under
  thin ice -- the user's cliff-vs-slope worry, in the thin-ice limit. The
  clip changes the driving stress on 65 405 of 71 678 ice | inactive-land
  facets (the intended ones; all 36 above 50 go below), on 815 of 4 764
  ice | inactive-water facets (dry coastal cliffs), and on 197 549 of
  3.78 M ICE | ICE facets (5.2 %): thin ice on steep ground, median
  thickness 36 m (90th pct 69), median speed 12 m/yr, where it caps the
  effective surface step at the upper cell's thickness -- new / old
  driving stress 0.26 (thinner cell < 20 m), 0.48 (20-50 m), 0.75
  (50-100 m), 0.88 (100-300 m); 4 343 of those facets run above 300 m/yr.
  For such ice the shallow-ice H grad S was the right driving stress, so
  the periphery and the steep thin margins will flow slower and thicken
  until H approaches the step. ALTERNATIVE TESTED with the same four-step
  script (`analysis/output/scratch/`, glide copies through PYTHONPATH):
  scale the step by the ice-ness of the HIGHER cell relative to the lower,
  w = clamp(H_hi / (0.05 H_lo), 0, 1), exact Jacobian -- V-cycles 10 / 2 /
  2 / 2 with |r_H| 86 / 8.1 / 2.2 / 2.6, wall facet -9 m/yr, and a
  footprint of 9 161 ice | land facets (all 36 above 50) and 1 933 ice |
  ice facets (0.05 %), thin-on-thin slopes untouched; its cost is a
  hard-coded fraction and that the push returns in proportion as a wall
  cell gains ice. Also tested and rejected: using the higher column's
  thickness on non-overlapping facets with a hard geometric switch (cures
  the facet, -6 m/yr, but stalls 9 / 10 / 10 / 10 at |r_H| 11-22). Every
  variant costs the cold-start step (HEAD 4 cycles to |r_H| 45; clip 10
  to 102; ramp 10 to 86). User, 2026-09-26: Vanka omega 0.25 (from
  0.5) restores convergence of that first step with the clip.

- **`ismip_exporter.py` (rewritten 2026-09-17, checked against the official
  ISMIP7 compliance checker 2026-09-18, `../ISM_SimulationChecker`)** exports
  a forward run's YEARLY frames (compressed VTI preferred; series.nc when no
  .pvd) to the submission layout `<root>/Models/GrIS/<group>/<model>/CORE/
  <set>/` (`--group`, default UM -- VERIFY the group id; `--set-counter`,
  default C001: a PARAMETER SET shared by all core experiments, not an
  experiment id -- the cheat sheet's C00x numbering was misread earlier) with
  a `not_modelled.txt` declaring the optional variables we lack. Windows from
  the checker's experiments_ismip7.csv: historical 1850-2014 (start free),
  projections pinned to 2015-2300 (ssp370 2100), so **runs must reach
  t = 2301 (2101)**: nominal year Y is the step (Y, Y+1] and the frame at Y+1
  (forward_projection T_END is 2301 since 2026-09-18; the ssp585 run on disk
  stops at 2300 and fails only that test). What the checker enforces and the
  exporter now does: ST stamped Jan 1 of Y+1, FL Jul 1 with bounds, time in
  days since 1850-01-01 (standard calendar) as FLOAT32; float32 variables,
  netCDF4 default _FillValue; global attrs group/model/contact_name/
  contact_email/crs=EPSG:3413; fill POLICIES per variable (the checker
  compares footprints cell by cell): `forbidden` (lithk, dlithkdt, licalvf,
  ligroundf, lifmassbf, the fractions) = zero, never fill; `no_ice`
  (velocities, strbasemag) = fill exactly where sftgif = 0; `no_grounded_ice`
  / `no_floating_ice` (libmassbfgr / fl) = fill where sftgrf / sftflf = 0;
  `outside_domain` (orog, base, topg, acabf) = ONE shared footprint, the
  computational domain, here ice + ice-free land (bed >= 0; open ocean is
  outside, which also keeps orog >= 0). Consistency: orog = base + lithk,
  base >= topg, base = topg where sftgrf = 1, base > topg where sftflf = 1 --
  so off ice base = orog = bed, all four are coarsened with the same weight,
  and the grounded fraction is RECOMPUTED from the exported geometry
  (sigmoid(c (r H + bed))) because the model's phi lags the final H update in
  ~250 cells per frame. Signs from the request's ranges: licalvf, lifmassbf
  <= 0; ligroundf > 0 (flux of grounded ice); scalar totals are magnitudes
  (scalars are not range-checked). Physics: licalvf = -rho_i (1 - psi) H /
  calving_timescale from the end-of-step frame (the implicit sink exactly);
  lifmassbf = 0 (the ocean forcing acts through the calving margin: the
  calving / frontal-melt split does not exist in this model); ligroundf =
  upwinded H u through grounded | floating-or-ocean faces; strbasemag with the
  BASAL velocity, the per-year xi and the capped beta; rho_water 1028; ice =
  active-set mask < 0.5 (the 1 m thklim floor is not ice); v needs NO sign
  flip (glide's stencils are image oriented). `--resolution` coarsens
  conservatively onto the nested node-centred grids (mass from the 4 km
  lithk / lim = 1.00000). Result 2026-09-18 at 4 km: 0 errors and 0 warnings
  on historical; ssp585 only the missing year 2300 plus a warning on 19 cells
  of acabf below -6e-4 kg m-2 s-1 (-20 m w.e./yr melt). ~4 s per year at
  4 km. Run the checker with `cd ../ISM_SimulationChecker && PYTHONPATH=.
  python -m isschecker --source-path <set dir> --variable-list ismip7`
  (needs cf-units, installed in glide_test_env 2026-09-18).
- **`tools/vti_to_nc.py` (2026-09-17)** converts a run's VTI series to one
  compressed CF NetCDF, `<run-dir>/series.nc` ((time, y, x) per field, vectors
  split into u/v components, bed/beta static, `model_year` + a CF `time`,
  EPSG:3413 grid mapping): velocities and SMB zeroed off the ice (the 2/3 of
  the grid that is ice-free holds solver noise and made those fields
  incompressible), values rounded to physical precision (1 cm, 0.01 m/yr,
  1 mm/yr, 1e-4 for the flags), zlib + shuffle. 326 MB -> ~50 MB per 1 km
  frame (x6.4 with every field kept; ~x9 with `--drop U_s srf`, both
  derivable), ~1.6 s per frame; a read-back check runs before `--delete-vti`
  removes the frames. `ismip_exporter.py` (`FrameSource`) and
  `analysis/basin_mass_balance.py` read `series.nc` when present, else the
  VTI; the exporter's output from the two agrees to rounding. Disk history:
  yearly 1 km VTI is 140 GB per 1800-2300 projection; the previous
  inversion's outputs (~520 GB) were deleted 2026-09-17.
- **`tools/vti_compress.py` (2026-09-17)** is the working-format answer:
  it rewrites a run's VTI frames IN PLACE (masked off-ice velocities / SMB,
  the same rounding as vti_to_nc.py, VTK's compressed appended layout with
  `vtkLZ4DataCompressor`, `header_type="UInt32"`, 32 KiB blocks), so the .pvd
  and the model-year time slider are untouched and ParaView >= 5.5 reads them
  directly, decoding only the ticked arrays at GB/s. 326 -> ~70 MB per 1 km
  frame (x4.6; the LZ4 level is irrelevant, default 1; zlib would be ~50 MB
  but 5x slower to inflate, which is what made series.nc feel slow), ~1 s
  per frame per core, each frame decoded back and compared bit for bit
  before it replaces the original, already-compressed frames skipped.
  `ismip_exporter.read_vti` reads both layouts (LZ4 and zlib). series.nc
  (vti_to_nc.py) remains the archive / CF format. NOT yet verified in
  ParaView itself (no ParaView on this machine) -- open a compressed .pvd
  before converting a run you care about.
- **glide `VTIWriter` compression (2026-09-17)**: `write_vti` / `VTIWriter`
  take `compressor` (None | "lz4" | "zlib"), `compression_level` (0 = LZ4
  fast), `precision` (field -> quantum, rounded on the GPU before the
  transfer), `mask_field` + `masked_fields` (zero the named fields where the
  dynamic field is >= 0.5). `forward_standalone.setup()` turns it on for both
  forward drivers (`VTI_COMPRESSOR`, `VTI_PRECISION`, `VTI_MASKED_FIELDS`):
  1 km frames are ~80 MB and an append takes 0.2 s (the raw 326 MB write
  took longer), decoded frames match the run's final state within the
  quanta. The inverse's writers (glacier_inverse/io.py) still write raw
  frames (library policy: alaska-verbatim).
- `export_ismip.py` uses ISMIP6 names/units (`lithk, orog, topg, xvelmean …`,
  m s⁻¹, kg m⁻² s⁻¹, days since 1850-01-01). Reconcile with the ISMIP7
  variable request (github.com/ismip) before submitting.

- **Discharge through Mankoff's flux gates (2026-09-23,
  `analysis/basin_mass_balance.py --gates
  common_data/dhdt/mankoff/dataverse_files/gates.gpkg`).** Mankoff's D is
  the flux across gates a few km inland of the termini, NOT the residual
  SMB - dM/dt over a basin, and the two differ by everything that leaves the
  basin without crossing a gate. The analysis now computes the SAME integral
  with the model's surface velocity and thickness under the gate pixels
  (`Dg_*`; compare with Mankoff's D, not D + BMB), the Mankoff-comparable
  `MBg_* = SMB - Dg` (their MB is the budget upstream of the gates), and the
  integral on the observed mosaic + BedMachine thickness at the model grid
  as the reference: at 1 km that recovers 391 of Mankoff's 491 Gt/yr (CW
  0.58, SE 0.75 -- the narrow fast fjords), so a model gate number is read
  against 391, not 491. TRAP in the gpkg: its 2765 rows are column strips of
  the rasterized gates holding 5890 pixels (polygon area / 200 m^2 = 1..8),
  not single 200 m pixels; 200 m per row undercounts the gate length 2.1x
  (553 vs 1163 km). v7 at 2018: gate D 283 (sigmoid_c 1.0) / 175 (0.1)
  against 391 observed, i.e. 0.72 / 0.45, while the residual D was 553 /
  492. The gap between residual and gate D (200-270 Gt/yr) is a thin marine
  apron (55-100 m, mostly floating, up to 25 km beyond the basin polygons,
  removed at the calving timescale) -- which is what the GEOMETRIC calving
  criterion does by design: a threshold on geometry, not a rate, handles
  topological change and does not pretend to front-position physics that
  are not established. The v7 state is too SLOW at the gates because its
  traction was calibrated at sigmoid_c 0.1, without tongues, and at 1.0 the
  tongues buttress and lower the surface gradients the optimization tuned
  for; the inversion is being rerun at 1.0 (2026-09-23 overnight), and the
  expectation is gate flux near 391 with matched velocities over the
  BedMachine geometry. Until then the c = 0.1 "7 of 8 basins" result is a
  compensating error: the basin residual landed near Mankoff with the
  outlets at 0.45 of the observed gate flux.

- **Per-glacier calving margin, the sweep route (2026-09-25).** The rho
  field's bound (library change 16) forbids advance past the observed
  extent, which ssp126's AMOC weakening in CESM2-WACCM produces, and the
  law's retreats are reversible anyway; the user's simpler alternative:
  `h0_i = c_i + alpha dTF_i` with c_i per glacier found by a 1-D SWEEP of
  forward runs over a global c, scored per basin under the approximation
  that glaciers calve independently, alpha a single scalar (an undercutting
  process, alike across glaciers). Built: (1)
  `preprocessing/make_calving_basins.py` -> `model_inputs/calving_basins.nc`
  (`calving_basin` int32, -1 none; `basin_dist_km`) + `.csv`: on the ice the
  gridded `rgi_label` (260 Mouginot basins in shapefile order, whose names
  are exactly Mankoff's `Mouginot_2019`, then the RGI periphery); over the
  fjords an INDEX-AWARE FLOOD FILL, a multi-source 4-connected BFS from every
  labelled ice cell through the ice-free cells with bed < 0 (first arrival
  wins = the geodesic Voronoi, following the fjords where Euclidean nearest
  neighbour crosses peninsulas), also over the 67 k ice cells outside every
  polygon (51 k isolated bodies stay -1), `--max-km` 60 beyond which the
  ocean stays -1 (inert). 2 s. 604 basins have a marine terminus (218
  Mouginot + 386 periphery), 173 carry gates; 249 k fjord cells assigned.
  Also in the file: `calving_reach` = the same flood run INWARD from every
  terminus through the marine ice (bed < 0) along the trough, up to
  max_km, together with the flooded water (36 k marine ice cells, median
  16 km in; -1 elsewhere) -- the cells a margin change can move, per
  glacier, without the interior -- and `front_dist_km`, the signed distance
  from the terminus (+ over water, - under ice); the evaluator's
  `--basin-var calving_reach` sums J over it instead of the drainage basin.
  (A saturating colour map makes the main sheet, indices 0-259, look
  unlabelled next to the periphery's 260-18736: it is labelled.)
  (2) `sweep_calving_c.py`: one forward_standalone run per c (`--c ...`,
  `--alpha`, `--alpha-q`, defaults from the config; clim_h 0, no field, no
  pin) from the config's spin-up, keeping ONLY the observational period on
  disk -- driver hooks `VTI_T_MIN` (frames from `--t-save` 1990), `VTI_FIELDS`
  (default H U_s mask phi h0 smb dhdt), `STATE_SAVE_TIMES` (`save_state`: H
  + srf at t_save, a restart point so an alpha sweep can branch from 1950
  without redoing the spin-up, which depends on c alone since dTF = 0 before
  1950) -- into `{output_dir}/sweep_c/c{+ddd}/` (vti/, state_1990.nc,
  forward_soln.nc, manifest.json; sweep.json at the root; resumable), plus
  RAW state files `state_{t}.nc` (H, srf, the STAGGERED surface velocity
  u_s / v_s, unmasked smb, the active mask, phi; `save_state`) at
  `--state-times`, default the restart point 1990 and the observation
  epochs 1993 / 2008 / 2015 / 2018 / 2019 -- the VTI frames mask SMB and
  velocity off the ice and round, so the evaluation reads the state files,
  not the frames. ~2.5 GB and a few minutes per 1 km run (the user's
  standalone runs take ~2 min). (3) `analysis/sweep_calving_eval.py`: the
  discrete optimization uses PRECISELY THE J OF THE CONTINUOUS ONE -- a
  `GlacierProblem` built from the domain config with the surface, velocity
  and extent specs plus the MEaSUREs dh/dt spec (added when the config has
  it commented out, the same parameters), a `ModelState` per epoch rebuilt
  from the state files at the run's level (the terms prolong / restrict as
  in the inversion, so a coarse sweep evaluates too), each term's whitened
  residual from its own `residuals()` and its per-cell summand (the terms'
  `_huber` from `loss.py`) summed per basin (`torch.bincount` on
  `calving_basin`); the extent term's global logit nuisance is fitted once
  by `loss()` and its per-cell Brier split afterwards. Basin sums reproduce
  the srf / vel / dhdt totals to round-off; the extent split misses only the
  nuisance's prior cost (a global scalar). 31 s per sweep at level 2
  including the problem build. NOT used: ATL15 (EN4's TF has a ~3-year
  phase error at the critical fronts -- Jakobshavn's late-2010s cold wave
  arrives in the early 2020s -- so over a 6-year window the forcing error
  dominates; the 27-year MEaSUREs record is the robust target) and the
  snowline (does not respond to the margin). `--assemble`: per basin with a
  marine terminus the CENTRE of the best plateau in c (`--plateau` 0.1 of
  the J range above the minimum; the response is a threshold), unresponsive
  basins (`--min-range`), land-terminating basins and unassigned cells take
  their region's median c; written as `model_inputs/calving_h0_base.nc`
  (`h0_base`, m, full grid) + `{sweep}/calving_c_basins.csv`,
  `{sweep}/sweep_eval.csv` (long: run, basin, term, J), `{run}/J_basins.csv`.
  Loader: `OceanForcing` adds an `h0_base` variable of the `rho_filename`
  file to the margin (`h0 = calving_h0 + h0_base + clim_h (...) + alpha_h
  (dTF - rho)`; `calving_rho` is optional in the file now), so
  `rho_filename="calving_h0_base.nc"` runs the assembled field. First sweep
  agreed 2026-09-25: c in -150..+100 by 25 m, alpha_h 50, alpha_q 0. Then
  one joint run of the assembled field is the test of the independence
  approximation; the alpha sweep can branch from the 1990 states.
  **First sweep (user, 2026-09-25, `inverse_v9/sweep_c`, 11 runs, ~1.3
  min each).** The first assembled field was nonsense -- c = 0 at every
  front (tongues gone at the first positive anomaly, Helheim / Jakobshavn
  never moving) -- for two reasons, both fixed: (1) the c = -25 run is DEAD
  (H at the 1 m floor everywhere, NaN velocity, the whole grid ice-free,
  0.3 min wall: a solver blow-up in the first steps at that c alone; rerun
  it with `--c -25 --force`), and the plateau rule "within 10 % of the J
  RANGE" let that one run inflate every basin's range to thousands, so
  every other c fell inside the plateau and its centre, 0, was chosen
  everywhere. Now: runs with a non-finite term or a domain total more than
  `--bad-run-factor` (5) x the median over runs are EXCLUDED, the plateau
  is relative to the minimum (`--plateau-rel` 0.1, `--plateau-abs`), its
  centre is the member nearest the plateau's mean c (ties to the argmin),
  `--select-only` re-selects from sweep_eval.csv without the GPU, and
  `--select-terms` restricts the objective (the CSV is named after `--out`).
  (2) The Mouginot polygons stop near the grounding lines, so Petermann's
  tongue carried no label and the flood handed it to the fjord-wall basins
  that touched it first (Petermann: 1 floating cell, no terminus, no fjord
  -> "no marine terminus"); `make_calving_basins.py` now gives each
  unlabelled floating component the flux-weighted (H |u|) majority label of
  its grounded 4-neighbours (8 tongues; Petermann 1291 floating cells, 79N
  1625, Zachariae 669) and `calves` includes floating fronts. Re-evaluated
  on the corrected basins (level 0, ~2 min for 11 runs): 79N -100, Zachariae
  -100, Petermann -75 (plateau -100..-50), Ryder -50, Humboldt -100 (flat
  -150..-50), Jakobshavn +25 (plateau +25 / +50), Helheim +50, Kanger +50,
  Store +25, Rink 0, Upernavik 0, Daugaard-Jensen 0, Koge Bugt C +75;
  regional medians NE / NW -62, NO 0, CW +25, SE / CE +50, SW +75; 559
  basins fitted; field in `model_inputs/calving_h0_base.nc`. WHICH TERM
  CARRIES THE INFORMATION: the extent term alone (`--select-terms extent`,
  `calving_c_basins_extent.csv`) picks c <= -75 everywhere -- for any c <= 0
  the 2015 extent is retained and J_extent ~ 0, so it is a one-sided "do
  not over-retreat" constraint that cannot rank the negative values; the
  retreat itself is in the 2008 SURFACE term (Jakobshavn / Helheim: J_srf
  206-210 for c <= -50 vs 4 at +25 / +50 -- the DEM is post-retreat, a front
  that never retreated is far too high) and in the 1993-2019 dh/dt term
  (79N: 173 at c = -50 vs 6.5 at -100 -- removing the tongue makes a
  drawdown the record does not have). Joint run of the field not yet made.
  **Composite run did not reproduce the sweep's solutions at the big
  outlets (user, 2026-09-25, Kanger).** Two causes. (1) OWNERSHIP OF THE
  FJORD: first-arrival flooding gave the water in front of Kanger to a
  one-cell peripheral glacier (84 of the 202 cells within 25 km, flux 9e2
  vs Kanger's 1.6e7), Styrte and Unnamed Kanger E, whose fitted c (+75 /
  +100) then governed Kanger's extended position. `make_calving_basins.py
  --primacy flux` (default; `nearest` = the old rule): the water is claimed
  in DECREASING TERMINUS FLUX, each front taking every unclaimed cell within
  max_km along the water (`flood_primacy`, windowed BFS per front, 8 s for
  602 fronts), so a fjord belongs to the largest glacier that can reach it
  and a peripheral glacier keeps only the cove no bigger one reaches: Kanger
  202 / 202, Helheim 188 / 189, Petermann 483 / 483, Store 246 / 254; 79N
  and Zachariae share their fjord system (786 / 533). (2) THE MANIFEST: a
  partial rerun (`--c -25 --force`) overwrote sweep.json with that
  invocation's runs, so the re-evaluation saw c = -25..+100 only and the
  tongues were forced to c = 0. `sweep_calving_c.py` now merges its manifest
  with the existing one, and the evaluator takes the union of the manifest
  and the finished run directories on disk. Also `--min-flux` (assembler):
  basins whose observed terminus flux is below it take the regional median
  instead of their own plateau (small fronts fit noise). Re-evaluated
  2026-09-25 on the user's rerun sweep, which is at ALPHA_H 100 (all 11
  runs; the joint run must use alpha_h=100), `--min-flux 1e5` (142 fitted,
  462 small fronts on regional medians NO -25, NW -75, NE 0, CW 0, CE +25,
  SE +50, SW +75): 79N -150 (AT THE GRID EDGE: with alpha_h 100 the tongue
  survives the +1.62 K anomalies only for c < -162 -- the ratio rule -- so
  the sweep needs -175 / -200 / -250 for the tongues), Zachariae 0 (the
  1993-2019 dh/dt wants the tongue gone: 8.6 at 0 vs 91 at -100; the
  transition sits between -25 and 0, a finer grid there would time it),
  Petermann -75, Ryder -50, Humboldt -100, Jakobshavn +25, Helheim +50,
  Kanger +50 (plateau +25..+75), Store / Rink / Upernavik / Daugaard-Jensen
  0, Koge Bugt C +100 (edge). Field in `model_inputs/calving_h0_base.nc`.
  **dh/dt term reviewed for the sweep (user: "sieving only on dh/dt selects
  solutions about as far from MEaSUREs as one could get"; alpha_h 50 sweep,
  2026-09-25).** The mechanics are right (the epochs, the states, the split
  reproduces the total); the whitened per-cell term is structurally the
  wrong instrument for ranking c: (1) the basin sum is an INTERIOR term --
  at Kanger the sigma-normalized misfit over the 30-km reach is ~500-650
  against ~75-120 k over the rest of the basin (sigma 0.05 m/yr on
  ~50 000 interior cells vs 0.4 near the front), and that interior part
  varies between runs (73 k -> 121 k) through the spin-up each c produces,
  so the selection followed the interior transient; (2) whitening (Matern
  l 10 km) is a high-pass filter: an unmatched SMOOTH observed signal (the
  1.92 km, 2-yr-endpoint product is smooth by construction) whitens to
  little, a sharp modelled response displaced by a few cells or sharper
  than the product pays twice, so NO RESPONSE is cheapest at every front
  (Helheim c +50 matches the observed reach mean exactly, -0.88 vs -0.89
  m/yr, and still loses 1.91 to 0.96); (3) the model's window-mean dh/dt
  is NON-MONOTONE in c (Kanger reach mean -0.12 for c <= -50, -0.98 at -25,
  -3.25 at 0, -0.57 at +25/+50, -25.8 at +75, -0.44 at +100): c sets the
  flip YEAR, a retreat before 1993 leaves nothing in the window, one at
  +75 is a collapse -- the 25 m grid straddles the transition (-25..0) at
  Kanger, Helheim, Store; (4) a genuine flaw in `DhdtObservation.model_rate`
  (library, not yet fixed): it compares dH/dt with an observed SURFACE rate
  everywhere, but on floating ice dS = (1 - rho_i/rho_w) dH, a factor ~9,
  which biases tongues toward "no thinning" (fix: multiply the model rate
  by (1 - rho_i/rho_w) where the state floats; an additive `surface_rate`
  flag). FIX FOR THE SELECTION: `J_dhdt_int`, the VOLUME-INTEGRATED rate
  per basin over a zone (`--dv-zone reach` = front_dist >= -30 km, default;
  `low` = ice below 1500 m; `basin`), model (H at the epochs) vs product
  (the observed surface rate times rho_w / (rho_w - rho_i) on the observed
  floating cells), as chi2/2 with sigma = max(`--dv-floor` 0.05 Gt/yr,
  `--dv-rel` 0.25 x |observed|) -- the quantity a displaced or
  sharper-than-the-product response still gets right, computed on the CPU
  in both modes and opt-in via `--select-terms ... dhdt_int` (the whitened
  `dhdt_measures` stays evaluated for the record). Alone it selects Helheim
  +50 (dV -0.16 vs -0.16 Gt/yr), Kanger -25 (-0.14 vs -0.23, the transition),
  Jakobshavn 0, Store -50, Upernavik +25, Rink +50, Humboldt +50; for the
  tongues it cannot decide (Petermann: observed -10.8 Gt/yr from the tongue's
  surface lowering x 9.3, model -0.37 kept / -27 removed: NO SUB-SHELF MELT,
  a retained tongue cannot thin, so the plateau spans the grid) -- those are
  decided by the surface / velocity / extent terms. Combined
  `--select-terms srf vel extent dhdt_int --min-flux 1e5` (the field now in
  `model_inputs/calving_h0_base.nc`, alpha_h 50): 79N -25, Zachariae -75,
  Petermann -25, Ryder -50, Humboldt -25, Jakobshavn +50, Helheim +50,
  Kanger +50 (plateau +25 / +50), Store / Rink / Daugaard-Jensen 0,
  Upernavik +25, Koge Bugt C +100; regional medians NE -25, NO / NW 0, CE /
  CW +25, SE +50, SW +75. THE REMAINING TENSION IS THE ALPHA AXIS: at Kanger
  the 2008 surface wants the retreat done by 2008 (c >= +25, i.e. in the
  1990s), the integrated 1993-2019 dh/dt wants it inside the window at the
  observed size (-25, and even then too small), and with alpha_h 50 the
  2004 anomaly buys only 10-25 m of margin -- timing = c / alpha, extent =
  alpha; the secondary alpha sweep (branching from the saved 1990 states;
  forward_standalone still needs an init-from-state option) and a finer c
  grid (-25..+25 by 5) across the transitions are the next steps.
  **Composite runs of the assembled field (`basin_mass_balance.py --gates`,
  `analysis/output/basin_mb_v9_field` at alpha_h 50 and
  `basin_mb_v9_field_a80` at 80, same c field, 2026-09-25).** alpha 50:
  GrIS MB -108 (pinned -71, free-forcing -276, Mankoff -148), SMB 330
  (337), gate D 269 (pinned 309; 391 = the observed fields through the
  gates at 1 km), residual D 438; NE held (-28 vs -19 observed, gate 26 vs
  29: no collapse, no runaway) but the gate fluxes are FLAT everywhere and
  half in SE (52 vs 110), CW declines slightly while Mankoff rises 65 ->
  90: no retreat-driven speed-up anywhere, and the c >= +50 fronts retreat
  during the spin-up. alpha 80 (the user's rerun with the alpha-50 c
  field): 1986-2027 gate D 269 -> 280 only, residual D 438 -> 521, MB -191
  -- the extra loss is ice removed between gate and terminus; the time
  series show the retreats arriving as PULSES: NO 22 -> 45-48 after 2012
  (Mankoff 25-28, the Petermann / Humboldt tongue loss), CE flat then 52 ->
  70-75 after 2014 (Kanger, a decade late), NW 60 -> 78 around 2008 and
  back to 65, CW 40 -> 22 after 2008 while Mankoff rises (Jakobshavn
  retreats, thins at the gate, flux falls), SE unchanged (52). At 2018 the
  gate flux is 312 / 391 = 0.80, the highest snapshot yet, but composed of
  Petermann 3.2x (speed 3.2), Zachariae 2.4x, Kanger 1.4x against Helheim
  0.01 (coverage 1.00, thickness 0.02: retreated PAST its gate), 79N 0.01
  (the gate is on the lost tongue), Jakobshavn 0.29 (thickness 0.44),
  Store 0.45 / Rink 0.93 at thickness 0.5; CW thickness ratio 0.50, SE
  0.82 with speed 0.67, NW 0.84 / 0.85. So a retreat here is a pulse: the
  front settles at HAB > h0, the trunk thins at the gate, and the flux
  falls below its pre-retreat value, whereas the observed CW / NW fluxes
  stay high for 15 years. (Both composites are from the alpha-80 SWEEP,
  the user re-swept -- an earlier note here got that wrong.) **Helheim is
  bistable for a geometric reason** (flowline through its gate, sweep
  states, 2026-09-25): the observed 2015 front sits on a SILL (bed -537 m
  at s = +8 km) in front of an 800-m over-deepening that runs back to the
  gate (-770..-810 m at s = +6..-2) and only shallows at s = -4..-6 (-278 /
  -443 m). Under a constant margin there is no stable front in the trough:
  c = 0 (knife edge, floating F = 0) fills Sermilik to the fjord bend with
  1000-1700 m of GROUNDED ice (HAB 300-1000 m; the trunk 700 m too thick
  12 km upstream, hence J_srf 213-241 for c <= 0), c = +25 removes the
  sill cells (HAB 50-90 m) during the spin-up and the trunk draws down
  dynamically to the pinning point at s = -4 (1990: 116 m at the gate vs
  950 observed; the lower 8 km an apron) -- MISI-type retreat through a
  reverse-sloped trough, not calving at the threshold; the real 2001-05
  retreat stopped and re-advanced inside that trough (lateral drag /
  melange the 1 km, 5-cell-wide fjord does not resolve). The user's finding
  that the CALVING TIMESCALE matters (tau 0.5 -> 1 yr in the composite:
  Helheim's gate keeps 826 m through 2018 instead of ~20 m; a big
  difference, H_c none): the sink removes 1 - exp(-dt/tau) of a cell per
  step with F < 0, so at annual steps tau 0.5 / 1 / 2 take 86 / 63 / 39 %
  and a one-year TF spike no longer amputates a front while a sustained
  exceedance still does (e^-5 at tau 1) -- tau separates spikes from
  trends, which alpha cannot; it is inert in the 10 / 25-yr spin-up steps
  (removal is complete either way) and it slows the traversal of a trough
  without creating a stable position in it. `sweep_calving_c.py
  --timescale`, `--H-c` (recorded in the manifests) and forward_soln.nc
  attrs `calving_timescale` / `calving_H_c` / `calving_q` / `calving_h0`
  keep a sweep and its composite consistent; the composite analysed at
  13:00 was the tau 0.5 run (12:51), the profile above the tau 1 run
  (13:22) -- two runs, not a bug.
  **tau = 1 composite (`basin_mb_v9_field_tau1`, run 14:05, alpha_h 80,
  tau 1; NB it ran with the 11:33 FIELD from the alpha-50 / tau-0.5 sweep
  -- the tau-1 sweep of 13:46 was evaluated at 13:55 but not assembled; its
  own combined selection is `model_inputs/calving_h0_base_tau1.nc` +
  `sweep_c/calving_c_basins_tau1.csv`: Helheim +75, Kanger +75 with dV
  -0.22 vs -0.23 Gt/yr, i.e. timing AND size, Jakobshavn +50, 79N -125,
  Zachariae -75, Petermann -100, Store / Rink 0, Upernavik +25; run it
  with rho_filename='calving_h0_base_tau1.nc').** 1986-2027: GrIS MB -111
  (Mankoff -148), SMB 327, gate D 297 (tau 0.5: 280; 391 ref), residual D
  438; NE -17 (obs -19, the 1990s spike gone), SE 73 at the gates (tau
  0.5: 52), CW 49, NW 65. At 2018 the gate flux is 320 / 391 = 0.82 and,
  unlike tau 0.5, COHERENT: Helheim 0.92 (coverage 1.00, speed 0.98,
  thickness 0.95 -- intact at its gate), Jakobshavn 0.86, Store 1.07, Rink
  1.10, 79N 1.05 (tongue kept, H 1.75), Daugaard-Jensen 1.21; the only
  overshoot Petermann 1.73 (tongue lost), the only collapse Anorituup 0.03;
  Kanger 0.76 (speed 0.63), Zachariae 0.60 (speed 0.34: tongue kept and
  2x thick, the observed 2018 is post-collapse), NW 0.68 and SE 0.68 with
  normal thickness and speed 0.87 / 0.72 -- SLOW, not thin. Time series:
  every region's gate flux is now nearly stationary (NO 22 -> 40 after
  2014 the exception), so what is still missing is the observed 2000-2015
  RISE in CW (65 -> 90), NW (90 -> 115) and SE -- the pulses are damped,
  and with them the speed-ups.
  **Consistent tau-1 composite (`basin_mb_v9_tau1_field`, 14:15, the
  tau-1 field): WORSE at the gates.** GrIS MB -91 (Mankoff -148), gate D
  270 (stale field 297), 2018 gate flux 277 / 391 = 0.71 (was 0.82), SE 54
  (was 73): Helheim at its selected c = +75 is drawn down BEFORE 1995
  (gate thickness ratio 0.28 in 1995, 0.22 in 2018, flux 0.12 of observed)
  and Kanger at +75 never moves (thick 1.33, slow, flux 18-22 all record).
  The decisive table, gate flux ratio model / observed(1 km) at 2018 from
  the sweep's OWN states vs c (-150 .. +100 by 25, tau 1, alpha 80):
  Helheim 0.24 (c <= -25, fjord filled) / 0.48 (0) / 0.69 (+25) / 0.22
  (+50) / 0.12 (+75) -- a ONE-GRID-POINT window; Kanger 0.46-0.56 (<= 0) /
  0.78 / 0.76 / 0.83 (+75) / 0.18 (+100); Jakobshavn 0.45 / 0.79 (0) / 0.87
  (+25) / 0.84 / 0.61; 79N 0.82 (-150) / 0.55 (-100) / 0 beyond; Zachariae
  0.87 / 1.54 (-100) / 0.35 (-50) / 0; Petermann 0.92 (<= -50) / 2.07 (-25)
  / 0; Store 1.16 (0) / 1.10 (+25) / 0.71 (+50); Rink 1.11 (0) / 1.18 (+25);
  Upernavik N 0.61 (0) / 0.73 (+50); regions at their best c: NO 0.87, NE
  1.19, CE 0.84, SE 0.65 (its CEILING at any c: slow, not a calving
  problem), SW 1.0, CW 1.12, NW 0.77; uniform c is best at -25 (0.73). So
  the selection objective chose a dead Helheim (+75) and the gates would
  have chosen +25: the whitened surface / velocity terms are blind to a
  large but SMOOTH trunk-scale error (the high-pass again; J_srf 4.5 ->
  8.2 from +25 to +75 while the gate thickness goes 0.95 -> 0.22), and the
  reach-integrated dV cannot tell "quiet" from "already dead" (a drawn-down
  trunk has dV ~ 0 = the product's small -0.16). The unwhitened, localized
  check the objective lacks is the velocity AND thickness products sampled
  on Mankoff's gate lines -- our observations, not Mankoff's D -- proposed
  as an optional selection term `J_gate` (log flux ratio at 2018); with it
  the per-basin picks would be Helheim +25, Kanger +75, Jakobshavn +25, 79N
  -150, Zachariae -100, Petermann <= -50, Store 0, Rink +25, Upernavik +50,
  ~0.85 of the 1 km reference at 2018. Not implemented pending the user's
  view (it borders the "Mankoff is validation" rule).

- **Calving screen (2026-09-24): `analysis/calving_screen.py` +
  `analysis/front_onsets.csv`.** The calving law is geometric and the
  margins are functions of the TF record and the parameters alone, so
  whether an OBSERVED front is admissible, and the year it stops being
  admissible, need no forward run: the screen evaluates glide's
  `calving_F` (numpy transcription, kernel constant rho_i/rho_w = 0.917 --
  see below) on the 2015 geometry at every Mankoff gate outlet grouped by
  Mouginot name (139 fronts), for a box of (tf_crit, clim_h, alpha_h) x a
  forcing-scale axis (`--dtf-scale`) x temporal filters on dTF (`--filter
  none | box:N | ema:TAU`), with dTF aggregated over the model's own steps
  (`--schedule 1850:10,1990:1`). Three flux-weighted (H |u|) cell sets per
  front: T = observed floating cells thicker than H_c (the tongue; q = 0
  makes F = -h0 there, so a tongue flips exactly when alpha_h dTF crosses
  -h0_base), A = ocean cells adjacent to the terminus carrying the slab the
  law's implicit rate limit allows (min(H_term, H u tau / dx); 566 of 903
  advance cells cannot reach H_c at any h0), term = the terminus cells with
  H > H_c (over-retreat behind the 2015 front). Scored against
  `front_onsets.csv` (23 fronts, literature onsets, approximate -- EDIT IT)
  with an asymmetric loss (early / spurious / denied 3, late / missed /
  never-admissible / advanced 1), and, more usefully, per front: the
  REQUIRED RATIO -h0_base/alpha_h (stable: > max dTF over the record;
  retreat: in (max dTF before the onset window, max dTF through it]) and
  the alpha_h feasibility window per (tf_crit, clim_h). 13 s for 900
  combos. Findings: (1) TF_clim does not classify tongue presence -- the 9
  tongue-bearing fronts span 0.3-4.8 degC and 85 of 129 grounded fronts sit
  below the warmest tongue; the front's principal strain rate (mosaic) is
  no better (northern tongues 0.02-0.07 /yr, but Rink 1.2 and Jakobshavn
  1.8 are tongue-bearing too). (2) The ratio rule is contradictory under any
  TF-linear static margin: Petermann and Zachariae have the SAME TF_clim
  (2.27) but need -h0/alpha in (0.25, 0.41] and (0.95, 1.53]; 79N (1.69)
  needs > 1.62, Humboldt (1.89) <= 0.26 -- the required margin is not
  monotone in TF_clim, so no (tf_crit, clim_h, alpha_h) times more than 8 of
  the 23 fronts (the config's (3, 10, 50) gets 3: 6 early, 4 spurious, 7
  never admissible). Filters shift the ratios but not the contradictions;
  ema:5 opens Steenstrup's window (peak-before-onset under `none`),
  Ostenfeld stays empty (the product's anomaly there peaks in the 1950s-80s).
  (3) The forcing scale is degenerate with alpha_h (the ratio table is
  scale-free), so the forcing-uncertainty axis costs nothing. (4) On the
  1 km grid 35 % of the observed GROUNDED terminus cells thicker than 100 m
  have H below flotation (median HAB 41 m at ice_fraction > 0.9; Kanger
  -280, Jakobshavn -75, Helheim -57 flux-weighted) -- BedMachine's front on
  the model grid floats, so any positive static margin removes the seeded
  fronts of the big warm outlets at t0 (the "missing ice" of v6), and the
  pinned run keeps them only because h0 = -1000 inside the mask.
  Consequence: the static margin must be NEGATIVE at every large outlet
  with a per-front magnitude, i.e. an h0_base FIELD (the ratio table gives
  it, per front, in units of alpha_h), not a function of TF_clim; the
  remaining knob is then alpha_h alone, and tongue growth at the fast
  grounded fronts (Helheim, Store: not rate-limited) is the price -- which
  is the physics the geometric law lacks (a strain-rate term). Outputs in
  `analysis/output/calving_screen/` (census.csv, scores.csv,
  windows_summary.csv, flips_*.csv, windows_tc*_ch*.csv).
  **Density ratio (found 2026-09-24)**: glide's kernels hard-code
  `RHO_I_OVER_RHO_W 0.917` (common.cu, i.e. rho_w = 1000) for phi / xi /
  the calving flag, while the config's `rho_water = 1028` (0.892) is what
  problem.py (`_initial_thickness_from_geometry`, `flotation_factor`),
  forward_standalone, forward_projection and ismip_exporter use: the
  dynamics float ice 3 % thinner than the Python side assumes (18 m of HAB
  at a 600 m grounding line). Not yet reconciled -- glide should take the
  ratio from the config.

- **ISMIP7 core submission (2026-09-30), `run_ismip7_core.py`.** 11 runs,
  `inverse_v1.1/ismip7_core/` (manifest.json records the calibration and
  refuses to mix): historical x {CESM2-WACCM, MRI-ESM2-0} from the config's
  t_start (100) to 2015 (`--pre-record standalone`: the inversion's own
  forcing before 1850, the GCM's historical from the ssp126 directory);
  ssp126 / ssp585 / ctrl to 2301 and ssp370 to 2101 BRANCHED from that
  historical at 2015 (`forward_projection --continue` on a copy: VTI frames
  hard-linked, the appended files copied; thermal state and the elevation-
  feedback reference carried; velocities restart from zero; `branched_from`
  in the attrs); OCX to 2026. Configuration: the ISOTHERMAL v1.1
  calibration, `calving_h0_base_v1.1_s1m2.nc`, alpha_h 80, tau 1, H_c 100,
  A_glen 2e-17, one-way thermal output (library change 24:
  `couple_rheology=False, surface_T="forcing"`, Q_geo 0.042). GCM forcing in
  ANOMALY mode (`forward_projection` default since 2026-09-30): the
  calibration's hybrid climatology (CARRA2 T + RACMO P) + tbias / pbias +
  the GCM's departure from its own 1986-2025 climatology, which for EVERY
  scenario of a GCM is the historical + ssp126 tail (checked: tas_clim /
  pr_clim bit-identical across ssp126 / 370 / 585 / ctrl, and the 1850-2014
  catalogue files identical across scenarios), so the scenarios differ only
  from 2015. Raw mode (`--mode raw`) is kept as a sensitivity: dEBM2 is
  3.5 K colder and 12 % drier than the calibration climatology, which pbias /
  tbias were not fitted against. OCX runs `--record standalone`: every step
  takes forward_standalone's compute_smb (the hybrid yearly fields 1986-2025,
  the Vinther index before) + the elevation feedback through the
  `ctx.t2m_offset` hook; the CARRA2_ocx directory supplies only the ocean TF
  (the EN4 file). Its own tas / pr are CARRA2 in BOTH variables, 15 % wetter
  than what pbias was fitted to -- the first OCX attempt ran on them and was
  dropped. OCX is FREE-calving (the projections cannot be pinned; a pinned
  state handed to the free law unwinds, library change 22).
  **Consistency check, OCX vs the standalone free composite**
  (`analysis/output/basin_mb_v1.1_ocx/`): annual steps from 1850 (standalone:
  10-yr to 1990) change nothing at the fronts -- GrIS 2006-20 -299 vs -300,
  1993-2019 -179 vs -180, gates 287 / 296 / 339 / 363 vs 286 / 293 / 340 / 363
  (1990 / 2000 / 2010 / 2018), every key glacier within 1-2 % at 2018; only
  1986-95 differs (-22 vs -33, 569 Gt less ice at 1986, probably the
  standalone's single 1980-90 frame). The 1950-2025 TF variability does not
  ratchet the fronts under annual stepping.
  **Results** (`analysis/output/basin_mb_core_{cesm,all}/`, all curves and
  Mankoff anchored on OCX at 1985; per-run series in `basin_mb_core_series/`,
  read with `basin_mass_balance.py --t-read 1949 2151`): GCM runs start
  above OCX at 1985 (CESM +4.2 kGt, MRI +1.2 kGt: 1850-1985 GCM anomalies vs
  the Vinther index); GrIS dM/dt 2006-2025 -219..-280 Gt/yr vs Mankoff -223
  (single decades are realization noise: CESM's three scenarios span
  -243..-344 over 2015-25 with near-identical forcing; MRI's 1986-2006 is +18).
  Sea-level contribution from 2015, mm SLE, CESM / MRI: ssp126 80 / 77 (2100),
  131 / 82 (2150); ssp370 114 / 146 (2100); ssp585 144 / 173 (2100), 569 / 577
  (2150); ctrl 38 / 28 (2100) -- ctrl's 120-164 Gt/yr is committed loss under
  a climate 0.58 K above the reference, no further dynamical response.
  THE USER'S READING (keep when writing up): under ssp585 NEGIS (NE) starts a
  marine-ice-sheet retreat right after 2100 (NE -53..-62 kGt by 2150, rate up
  to -1900 Gt/yr) whose RATE is set by the +250 m h0 cap -- the robust
  statement is that NEGIS destabilizes, not how fast; expect small ensemble
  spread to 2100 and enormous after. The totals sit high in ISMIP6 terms
  because Mankoff / IMBIE sit near ISMIP6's 95th historical percentile; they
  are mid-ensemble of the user's earlier PISM ensemble that also matched
  Mankoff.
  **Export** (`ismip_exporter.py`, 1 km, set C001, group UMT, model GLIDE):
  `inverse_v1.1/ismip7_core/ISMIP7_submission/Models/GrIS/UMT/GLIDE/CORE/
  C001/`, 396 files (11 x 36 incl. the five thermal variables), 167 GB,
  ~80 MB and ~3.7 s per model year; OCX exported as experiment `ocx`, years
  1986-2025 (the checker has no OCX row). ISMIP7 checker on CESM historical +
  ssp585: 0 errors, 7 range WARNINGS, all genuine model values on tiny
  fractions: strbasemag up to 1.6 MPa (beta_max cells; limit 1 MPa), topg
  -4047 m (BedMachine, 15 values), xvel* just over 0.0008 m/s (ssp585), and
  acabf down to -6.6e-4 kg m-2 s-1 (-22.7 m ice/yr) at x -211..-202,
  y -3091..-3089 km in ssp585 2154-2165: thin near-sea-level ice in the
  warmest, rain-shadowed corner of Greenland melting out under the elevation
  feedback (user: expected there). FULL CHECK (all 11, 96 min): 360 files of
  the 10 core experiments, 0 errors, 21 warnings of the same four kinds
  (strbasemag in every experiment, topg in historical / ssp126, acabf and
  xvel* in ssp585 only -- up to 0.00086 m/s, 28 values); the single ERROR is
  that the checker's table has no `ocx` experiment, so the OCX files are not
  checked (report: C001/compliance_checker_log.txt). Transfer: Globus from
  campus (Starlink upload 12 Mbps = ~31 h for 167 GB); lossless repacking
  gains 1-3 % (shuffle is already on), GranularBitRound at 4 / 3 significant
  digits would give x1.7 / x2.2 if ISMIP7 accepts quantization (not used). Disk: the 11 runs take 330 GB (1 km yearly LZ4 VTI ~77 MB per
  frame); the root filesystem filled during this work.

## Known gaps / follow-ups

- **Spin-up length: what is actually needed (2026-09-22).** The point of the
  `climatology_only` run is NOT to reach a steady state, it is to find how
  long an integration makes the unknown initial condition stop mattering.
  Linearizing, dH(T) = e^{AT} dH(0) + [int e^{A(T-s)}] B dtheta, so the SMB
  signal in the final surface over the surviving IC error is
  R(T) = tau (e^{T/tau} - 1): linear in T while T << tau, exponential after.
  MEASURED (user, 2000-yr constant-climate run from an optimized state): the
  northern dh/dt halves every 500-700 yr, i.e. tau ~ 700-1000 yr, and faster
  further south. That is NOT the Nye volume time H/adot (11,500 yr for NE),
  which answers a different question (approach to a new equilibrium VOLUME
  after a forcing change). An IC inconsistency is a thickness perturbation at
  FIXED forcing and relaxes diffusively: the fundamental mode is
  L^2/(pi^2 D) with D = 3q/alpha, i.e. H/(3 pi^2 adot) = the Nye time / 29.6
  = 389 yr for NE, and sliding (less slope-sensitive flux, smaller D) lifts
  it to the measured value. Numerical diffusion is NOT the cause: upwind
  u dx/2 and implicit u^2 dt/2 give 1e4-1e5 m2/yr against a physical D of
  4.2e7. Consequence: 2000 yr leaves under 10 % of the IC error anywhere,
  3000 yr under 3 % in the north, and that is inside the window where holding
  the modern reference climate is defensible -- paleo forcing is a refinement,
  not a prerequisite. Cost is ~2.5x the current 326-yr window at a 20-yr
  spin-up step, less with a 50-yr deep step. Watch for outlets that survive a
  326-yr spin-up but not a 2000-yr one: that is information about
  `calving_h0` / `clim_h`, not an artefact to suppress.
- **All three spin-up biases are now addressed (2026-09-22), none yet run.**
  They were one-signed -- every one made the spin-up too positive -- and
  negligible at a 326-yr window but first-order at the 2000-3000 yr the
  relaxation-time measurement calls for. Sizes at t_start 0, as an
  ice-sheet-mean rate and as the steady offset rate x tau with tau ~850 yr:
  the pre-instrumental temperature hold +38 Gt/yr (+21 m, GISP2 deep
  extension), flat precipitation +15 Gt/yr (+8 m, `alpha_precip` +
  `make_precip_anomaly.py`), and the step-mean variance suppression
  +90 Gt/yr (+49 m, library change 14) -- the largest by far. Together ~143
  Gt/yr, or ~78 m of ice-sheet-mean thickness, against the IC error the
  longer spin-up is meant to shed. FIRST THING TO CHECK on the next deep
  run: the three corrections all reduce SMB, so the calibrated pbias / tbias
  should move the other way from v6, and the drift measured under
  `climatology_only` should now be comparable between the spin-up and the
  1986-2025 window instead of offset by ~90 Gt/yr.

- **Projection elevation feedback (done 2026-09-17, temperature only)**:
  `forward_projection.py` adds `FEEDBACK_LAPSE(month) * (S_model(t) - S_ref)`
  to the forcing temperature (`ElevationFeedback`; `ELEVATION_FEEDBACK`,
  `--no-elevation-feedback`). `S_ref` is the MODEL surface at
  `FEEDBACK_T_REF` = 2015 (ISMIP's h_ref; runs are bit-identical before it,
  so the hindcast is untouched), or the observed DEM when None. The lapse is
  the along-surface gradient of the forcing itself, -5.4 to -6.0 K/km by
  month (regression of the CARRA2 100 m climatology on elevation + a
  quadratic horizontal trend over the ice; -4.1 K/km below 1200 m in JJA) —
  NOT `monthly_lapse_rate`, the 100-500 m boundary-layer gradient above a
  fixed surface (an inversion over 2/3 of the ice in winter, -2 to -4 K/km in
  summer). The surface is read at the start of each step (one-step lag) on
  the run level and injected onto the fine SMB grid; lost ice drops the
  surface to the bed, so newly exposed cells warm too. glare reads
  `geometry.srf` for the avalanche operator only, so it is left alone.
  Precipitation does NOT respond (open). Test, SSP370 anomaly mode, 4 km,
  2015-2100: mean surface lowering 21 m, JJA feedback warming 0.12 K over
  the original ice mask, SMB -445 vs -376 Gt/yr in 2096-2100, sea-level
  contribution 100.7 vs 95.0 mm (+6%; +2.6% at 2050). `scalars.csv` carries
  `dS_ice_mean` and `dT_feedback_jja`.
- **CARRA2 2 m vs 100 m (analysis/compare_t2m_t100.py, 2026-09-16)**: the
  T100 - T2m deficit over the ice is 2-4 K in JJA and 7-8 K in winter above
  2000 m. Year to year the ice-mean 2 m and 100 m anomalies move 1:1 in every
  season (slope 1.04-1.07, r 0.94), so the 2 m field is neither amplified nor
  damped interannually. But CARRA2's 2 m record is INHOMOGENEOUS: the JJA
  deficit above 2000 m steps from 5.1 K (1986-99) to 3.2 (2000-09), 2.7
  (2010-19), 1.9 (2020-24), 1.4 (2025) while DJF and all elevations below
  2000 m stay constant to +-0.3 K; summer-only, interior-only steps at 2000
  and 2020 point to observing-system changes in the surface analysis (to be
  checked against the CARRA2 documentation), not physics. Hence CARRA2 T2m is
  3 K colder than RACMO 2 m in the 1990s but 0.2-0.7 K colder after 2020,
  its 1986-2025 JJA trend is doubled (1.12 K/decade vs RACMO 0.38, CARRA
  T100 0.56, CESM tas 0.66), and a 40-yr per-cell regression T100 ~ T2m
  absorbs the step (interior slopes 0.4-0.6 are an artefact). CARRA T100 -
  RACMO 2 m is stable (JJA ice mean 1.25-1.5 K since 2000), so a 2 m -> 100 m
  offset for CESM tas should be built from CARRA T100 minus RACMO tas
  climatologies, not from CARRA's own 2 m; CESM tas then keeps a -0.75 K JJA
  bias vs RACMO 2 m. The calibration (CARRA T100) is unaffected. Outputs in
  analysis/output/t2m_t100/ (gitignored).
- `rto_sample.py` is still on the pre-migration observation API (same as
  alaska); `sensitivity.py` raises on enthalpy/tbias domains (same as alaska).
- `smoke_test.py` section 8 builds a second `GlacierProblem` (large on the
  ice sheet); run it on `greenland_coarse` only.
- Velocity error rasters and per-pixel ATL15 sigma could feed spatially
  varying `MaternNoise` sigmas — the spec API is scalar today.
- Ocean forcing / calving in the INVERSE is a constant `calving_timescale`/`calving_q`
  (the parametric q(TF) lives in `forward_standalone.py` only); the ISMIP7 retreat
  parameterisation is forecast-phase work (see DATA_MANIFEST.md §19).
