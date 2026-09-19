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
classifies both sides on the main sheet; not yet rerun). Expect SMB ~470 Gt/yr at the label's bare-ice area by
interpolation — the remaining ~130 Gt/yr over Mankoff is CARRA2's precip
excess, which then has to come out of pbias against dh/dt.

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
  mean over the CARRA2 climatology years, gaps interpolated, constant before
  1784. The configs use `base_anomaly_year=None` and `alpha_t2m=0.6`: RACMO
  tas JJA regressed on the Vinther JJA series gives 0.58 ice-sheet mean
  (0.51 CE .. 0.70 SW, r 0.6-0.8, and the fit is stable across halves of
  the record), whereas HadCRUT explains almost nothing of
  Greenland's interannual-to-decadal history (r 0.1-0.4; the 1930s-40s warm
  and 1970s-90s cool periods are absent from a global series). The global
  PAGES2k+HadCRUT splice remains available as `--source global` (then
  `alpha_t2m` ~1.3-2 and a base year are needed).
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

## Known gaps / follow-ups

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
