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
7. Ocean thermal forcing of the calving margins (`config.OceanForcingConfig`
   + `config.calving_h0`, `ocean.py`, `forward.simulate(ocean_forcing=)`):
   glide's hybrid height-above-buoyancy threshold `H - H_f < q H + h0` gets
   `q = calving_q + alpha_q dTF`, `h0 = calving_h0 + alpha_h dTF` per step
   from `model_inputs/thermal_forcing.nc` (dTF = the step's annual-mean or
   annual-max TF anomaly vs a reference window, zero before the record and
   farther than `max_dist_km` from the native product). alpha_q (1/K,
   scales with thickness) and alpha_h (m/K, same distance everywhere) are
   SWEEP parameters, not differentiated: psi is a ~1 m-wide switch in
   flotation excess, so dJ/dalpha would be supported on a handful of cells.
   The step's margins carry no gradient; glide's `GlideStep` checkpoints
   q/h0 with each step so the adjoint re-solves with the right margins.
   `GlacierProblem.ocean_forcing` is the loaded forcing (None when disabled
   or the file is absent, with a warning); the same object drives
   `forward_standalone.py`.

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
by averaging; the flag is `psi = sigmoid(c (z - rho_i/rho_w (q H + h0)))`),
and `GlideStep` checkpoints both with the step.

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
- `export_ismip.py` uses ISMIP6 names/units (`lithk, orog, topg, xvelmean …`,
  m s⁻¹, kg m⁻² s⁻¹, days since 1850-01-01). Reconcile with the ISMIP7
  variable request (github.com/ismip) before submitting.

## Known gaps / follow-ups

- `rto_sample.py` is still on the pre-migration observation API (same as
  alaska); `sensitivity.py` raises on enthalpy/tbias domains (same as alaska).
- `smoke_test.py` section 8 builds a second `GlacierProblem` (large on the
  ice sheet); run it on `greenland_coarse` only.
- Velocity error rasters and per-pixel ATL15 sigma could feed spatially
  varying `MaternNoise` sigmas — the spec API is scalar today.
- Ocean forcing / calving in the INVERSE is a constant `calving_timescale`/`calving_q`
  (the parametric q(TF) lives in `forward_standalone.py` only); the ISMIP7 retreat
  parameterisation is forecast-phase work (see DATA_MANIFEST.md §19).
