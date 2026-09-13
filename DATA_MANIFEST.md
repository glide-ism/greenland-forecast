# Observational data manifest — greenland-forecast

Every dataset the preprocessing pipeline (`preprocessing/make_all.py`) and the
ISMIP7 hand-off need, what consumes it, and where it must sit under
`common_data/` (gitignored; the machine-readable twin is `data_manifest.json`).
Everything is regridded onto EPSG:3413 at the domain resolution
(`domains/<name>/local_data/domain.json`; the `greenland` domain is the ISMIP
standard 1 km grid), and every product carries acquisition attrs
(`time_nominal/time_start/time_end`) so the inverse model compares it against
the model state at its own epoch.

Priority: **R** required to build `GLIDE_inputs.nc`; **S** strongly
recommended (a loss term or ISMIP requirement is missing without it); **O**
optional (term dropped gracefully when absent); **F** forecast phase only
(not consumed by the inverse yet).

## Summary

✅ = on disk (2026-09-08 download); ⚠️ = incomplete.

| # | Product | Role (builder → variables → loss term) | Pri. | Local path under `common_data/` |
|---|---------|----------------------------------------|------|-------------------------------|
| 1 | BedMachine Greenland **v6 (v6.6, 2025-11)** (NSIDC IDBMG4) | `make_dem.py` → elevation, bed_obs, bed_obs_err, rgi_mask, floating_mask, thickness_obs, rgi_periphery_fraction → surface term, extent term, bed-prior conditioning, geometry seed | R ✅ | `geometry/bedmachine/BedMachineGreenland-v6.nc` |
| 2 | MEaSUREs annual velocity mosaic NSIDC-0725 v5 (2015-16) | `make_velocity.py --source measures_annual` → vx, vy, vmask, vx_err, vy_err → velocity term | R (one of 2/3/4) | `velocity/measures_0725/greenland_vel_mosaic200_2015_2016_{vx,vy,ex,ey}_v05.0.tif` |
| 3 | MEaSUREs multi-year mosaic NSIDC-0670 v1 (1995-2015) | as 2, `--source measures_multiyear` | alt. | `velocity/measures_0670/greenland_vel_mosaic250_{vx,vy,ex,ey}_v1.tif` |
| 4 | ITS_LIVE V02.1 120 m summary mosaic RGI05A (2014-2024 climatology; file created 2025-11-24), NetCDF with `vx, vy, vx_error, vy_error, count, landice, floatingice, dv_dt`; `vx/vy` are the offset of the line fit at its 2018-01-01 intercept → epoch 2018.0 | as 2, `--source itslive` (default) | R ✅ | `velocity/itslive/ITS_LIVE_velocity_120m_RGI05A_0000_V02.1.nc` (3.9 GB) |
| 4b | glide example file (MEaSUREs 1995-2015 at 900 m) | `--source glide_example` fallback | fallback ✅ | `../../glide/data/GLIDE_greenland_inputs.h5` |
| 5 | CARRA2 pan-Arctic monthly **temperature at 100 m and 500 m above ground**, total precipitation, orography — full record 1985-10..2025-12 (483 months) in `climate/carra2/1985_2025/`, plus the original 27-month files, Greenland box | `make_carra_vars.py` → monthly_t2m (100 m air temperature lapse-corrected with the local 100-500 m lapse rate), monthly_precip, monthly_lapse_rate, carra_orog → SMB forcing | R ✅ | `climate/carra2/1985_2025/{t.nc, precip.nc}`, `climate/carra2/orog.nc` (3.7 GB compressed) |
| 6 | **Vinther et al. (2006) SW Greenland monthly station composite 1784-2013** (tenths of degC, −999 missing) — the transient forcing; PAGES2k global reconstruction + HadCRUT5 annual as the `--source global` alternative | `make_temperature_anomaly.py` (default `--source vinther`) → temperature_anomaly.nc: JJA anomaly, CARRA2 SW-basin extension to 2025, zero-mean over the CARRA2 climatology years | R ✅ | `climate/temp_anomaly/swgreenlandave.dat`; `climate/temp_anomaly/{pages2k_ngeo19_recons.nc, HadCRUT.5.0.2.0.analysis.summary_series.global.annual.csv}` |
| 7 | Terrain (the DEM itself) | `make_insolation.py` (gtic) → direct + diffuse solar potentials | R (derived, no download) | — |
| 8 | Mouginot & Rignot (2019) drainage basins v1.4.2 (260 basins, fields SUBREGION1/NAME/GL_TYPE) | `make_dem.py` → rgi_label, rgi_id → per-basin velocity marginal, divide flux, diagnostics | S ✅ | `area/basins/Greenland_Basins_PS_v1.4.2.{shp,…}` |
| 9 | RGI **6.0** region 05 (NSIDC-0770 v6; 20 261 outlines, `Surging`/`Connect`) | `make_dem.py` → peripheral-glacier labels + surge_type (Connect == 2 glaciers stay with their basin) | S ✅ | `area/rgi/05_rgi60_GreenlandPeriphery.{shp,…}` (RGI 7.0 `RGI2000-v7.0-G-05_*` also accepted) |
| 10 | ICESat-2 ATL15 **v5** Greenland 1 km height change, cycles 03-29 (2019-01 → 2026-01) | `make_dhdt.py --source atl15 [--t0 --t1]` → dhdt, dhdt_err (weighted trend of delta_h) → dh/dt term | S ✅ | `dhdt/atl15/ATL15_GL_0329_01km_005_02.nc` |
| 10b | ITS_LIVE Greenland ice-sheet + peripheral-glacier elevation change, MEaSUREs G1920V01 v1.1 (1.92 km, monthly 1992-2023, `dh`/`rms`/`h_dem`/firn-air anomalies) — used as the SECOND dh/dt product, trend over 1992-2019 (`--name measures` → `gridded_dhdt_measures.nc`, `dhdt_measures` term) | `make_dhdt.py --source itslive_dh --t0 2000 --t1 2020` — the long-window complement | S ✅ | `dhdt/measures/Greenland_G1920V01_IceSheetGlacierIceHeight.nc` |
| 11 | ESA CCI Greenland surface elevation change (radar altimetry, 5 km) | `make_dhdt.py --source gridded` | alt. (superseded by 10b) | `dhdt/cci_sec/greenland_sec.nc` |
| 12 | ArcticDEM mosaic v4.1, 100 m (dated surface) | `make_dem.py` overlay → elevation over ice with 2007-2020 epoch attrs | S | `dem/arcticdem/arcticdem_mosaic_100m_v4.1_dem.tif` |
| 13 | ERA5-Land monthly t2m/tp, Greenland box | `make_era5land_vars.py` → *_era5land companions (intercomparison) | O | `climate/era5land/era5land_greenland.nc` |
| 14 | IceBridge MCoRDS L2 (IRMCR2) + pre-IceBridge (BRMCR2) | `make_bedradar.py` → flightlines.gpkg → radar-pick bed conditioning | O | `flightlines/{irmcr2,brmcr2}/**/*.csv` |
| 15 | End-of-summer snow/bare-ice classification (MODIS-derived) | `make_snowline.py` → snow_fraction → snowline (ELA) term | O | `snowlines/<YYYY-YYYY-average>/*.tif` |
| 16 | Greenland accumulation reconstruction (Box et al. 2013 / NGRIP) | `make_precip_anomaly.py` → precip_anomaly.nc | O | `climate/precip_anomaly/greenland_accumulation.csv` |
| 17 | Hugonnet et al. (2021) dh/dt tiles, RGI 05 | `make_dhdt.py --source hugonnet` (peripheral glaciers) | O | `dhdt/hugonnet/{dhdt,dhdt_err}/*.tif` |
| 18 | MAR v3.12–3.14 / RACMO2.3p2 (1 km) SMB climatologies | validation of the calibrated SMB; ISMIP7 SMB reference | F/S | `smb/{mar,racmo}/…` |
| 19 | ISMIP7 Greenland forcing MIPkit (SMB anomalies, ocean thermal forcing / retreat) | forecast experiments | F | `ismip7/…` (Globus) |
| 19b | ISMIP7 ocean thermal forcing, EN4-based (Verjans bias correction, Slater inland mapping), monthly 1 km 1950-2026 (2026 = Jan-Feb only) | `make_thermal_forcing.py` → thermal_forcing.nc (annual mean/max, `tf_dist`) → `forward_standalone.py` q(TF) | R ✅ | `ocean/tf/tf_GrIS_EN4_OCX_ocean-1000m_v1_{year}.nc` (77 files, 232 MB each) |
| 20 | Terminus positions: TermPicks, MEaSUREs NSIDC-0642 | terminus/extent at epoch; calving validation | F/O | `termini/…` |
| 21 | RACMO2.3p2-ERA5 statistically downscaled 1 km, ISMIP7 forcing kit (`tas` = corrected 2 m air temperature, `ts` = the same clipped at 273.15 K, `pr` kg m⁻² s⁻¹), monthly 1958-2025, native ISMIP 1 km grid | `analysis/compare_racmo_carra.py` (vs CARRA2), `analysis/arctic_amplification.py` → amplification + optional hybrid pre-1958 reconstruction | S ✅ | `climate/RACMO2.3p2-ERA/{tas,ts,pr}/*_GrIS_RACMO2.3p2-ERA_OCX_SDBN1-1000m_v1_{year}.nc` (6.9 GB) |

## Details

### 1. BedMachine Greenland v6 — geometry, bed and bed error ✅ on disk
- Morlighem et al., *IceBridge BedMachine Greenland, Version 6* (file `product_version` v6.6, modified 2025-11-24), NSIDC DAAC, <https://nsidc.org/data/idbmg4/versions/6> (v5 DOI [10.5067/GMEVBWFLWA7X](https://doi.org/10.5067/GMEVBWFLWA7X)).
- 150 m, EPSG:3413, 10218 × 18346, 2.8 GB. Variables: `surface` (GIMP DEM v2.1), `bed`, `thickness` (mass conservation), `errbed`, `mask` (0 ocean, 1 ice-free land, 2 grounded, 3 floating; no code 4 in v6), `source` (0-53 incl. 9 = IceBoost, 10+ = bathymetry surveys), `dataid`, `geoid` (EIGEN-6C4), `rgi` (inside an RGI 7.0 outline). Global attrs: `nominal_year` 2008, `time_coverage` 1970-2019-10, sea-water density 1023.
- Consumed by `make_dem.py`: elevation composite (surface over ice, bed = IBCAO/multibeam bathymetry in fjords), ice mask (`rgi_mask` — the variable name the inverse reads), floating mask, thickness (geometry seed via `init_from_observed_geometry`), `bed_obs`/`bed_obs_err` (all cells) and, from `dataid == 2`, `bed_radar_fraction`/`bed_obs_radar`/`bed_obs_radar_err` (radar-pick cells only: 4% of ice at 150 m, 26% of 1 km cells contain a pick) for the conditioned bed prior (`BedConditioningConfig.gridded_bed_data = "radar"`). v6 `source` on ice: 64% kriging, 22% mass conservation, 9% IceBoost, 5% interpolation; errbed medians 43/86/18/30 m — a 30 m floor even 100 km from any track.
- Epoch: `make_dem.py` tags the surface with the file's `nominal_year` (2008; GIMP DEM v2.1, 2003-2009 imagery) and the mask ~2015 (GIMP mask v2.0 + Mouginot coastline); prefer the ArcticDEM overlay (12) for a dated surface.
- Access: NASA Earthdata login (`earthaccess.download("IDBMG4")` or the NSIDC HTTPS tree).

### 2–4. Surface velocity
- **NSIDC-0725 v5** (Joughin), *MEaSUREs Greenland Annual Ice Sheet Velocity Mosaics from SAR and Landsat*, 200 m, Dec–Nov years 2014-15 … 2022-23, GeoTIFF `vx, vy, ex, ey` (+ `vv`). DOI [10.5067/USBL3Z8KF9C3](https://doi.org/10.5067/USBL3Z8KF9C3). Recommended for the ISMIP7 initial state (use the 2015-2016 mosaic; ~1.5 GB per year).
- **NSIDC-0670 v1** (Joughin et al. 2016), multi-year 1995-2015 mosaic, 250 m, `vx, vy, ex, ey`. DOI [10.5067/QUA5Q9SVMSJG](https://doi.org/10.5067/QUA5Q9SVMSJG). The product in glide's example file; long-window average for a spin-up-style calibration.
- **ITS_LIVE V02.1** summary mosaic `RGI05A_0000` ✅ on disk: 2014-2024 weighted line fit to annual means, `vx/vy` = the value at the 2018-01-01 intercept (stamped `time_nominal` 2018.0, window 2014-2024), `dvx_dt/dvy_dt` the slope, from <https://its-live.jpl.nasa.gov/> (AWS `s3://its-live-data/velocity_mosaic/v2.1/production/`, no login). Also annual mosaics per year.
- All three are EPSG:3413, so `make_velocity.py` area-averages them directly; error rasters are carried as `vx_err`/`vy_err` for a future per-pixel velocity error model (the current `VelocitySpec` uses a scalar/Matérn error).

### 5. CARRA2 pan-Arctic reanalysis — height-level monthly climatology (SMB forcing) ✅ on disk
- Copernicus Arctic Regional Reanalysis 2 (pan-Arctic 2.5 km, 2869×2869 grid; CDS, cfgrib-converted). Files `climate/carra2/t.nc`, `precip.nc`, `orog.nc`, downloaded 2026-09-08 over a Greenland box (15% of the grid finite): **temperature at 100 m and 500 m above ground** (`t(time, heightAboveGround, y, x)`, K, `GRIB_stepType avgd`), **total precipitation** (`tp`, kg m⁻² = mm per day, monthly mean of daily accumulation), and **orography** (`orog`, m) — 27 months: 1985-10..12, 2000-01..12, 2025-01..12. The 2 m field is deliberately not used (see below).
- Why height levels: CARRA2's 2 m temperature over the ice sheet already carries the melt-depleted stable boundary layer; the 100 m level is used as the free-air forcing for the enthalpy model's turbulent exchange, and (T500 − T100)/400 m is the local monthly lapse rate that moves it from CARRA's orography onto the model DEM (`make_carra_vars.py`; lapse clipped to [−12, +10] K/km, inversions kept). The two levels requested were 100 m and 400 m; the file holds 100 and 500 m.
- **Timestamps**: `t.nc` is stamped on the first of its month (00 UTC); `precip.nc` (a monthly mean of daily accumulations, forecast product) is stamped **12 UTC on the last day of the preceding month** (January 2000 = `1999-12-31T12`). `make_carra_vars.py` shifts the precip stamps by 12 h before grouping on calendar month; without that shift every precip month lands in the previous month (fixed 2026-09-09).
- **Coverage gap:** the downloaded box misses the northwest corner of the ISMIP grid — 33 298 ice cells (1.8%, x −653…−383 km, y −1098…−633 km: Inglefield Land / Washington Land and the Ellesmere-facing coast) receive nearest-neighbour climate. Extend the box west/north when re-downloading.
- Only one full year (2000) plus 2025 and a 1985 tail enter the calendar-month climatology; `months_per_calendar_month` in the output records the count. Extending the download to 1991-2025 monthly is the obvious next step (~0.5 GB per variable-year at this subset).
- `make_pancarra_vars.py` reduces to a 12-month calendar climatology, lapse-corrects t2m (6.5 K/km against a 2.5 km-smoothed DEM) and converts precip to m ice eq. yr⁻¹.

### 6. Transient temperature anomaly — required ✅ on disk
- **Vinther et al. (2006, JGR 111:D11105) SW Greenland composite** (`swgreenlandave.dat`, header `SW_GREENLAND 1784-2013`; monthly means in tenths of degC, −999 missing; complete from 1840, 239 missing months before). Coastal stations (Nuuk/Ilulissat/Qaqortoq lineage), 1961-1990 climatology −9.2 degC (Feb) .. 7.1 degC (Jul). `make_temperature_anomaly.py` takes the **JJA mean** as the annual series (winter anomalies are ~2x larger and would dominate a uniformly applied annual mean), extends it 2014-2025 with the CARRA2 100 m temperature averaged over the SW drainage basin (regressed on the stations over 1986-2013, slope 0.77, r 0.83), interpolates gap years, holds the 1784-1813 mean before 1784, and references the series to zero mean over the CARRA2 climatology years so it is applied to the CARRA2 climatology with `base_anomaly_year=None`. Tested against RACMO2.3p2-ERA5 regional JJA anomalies 1958-2013: JJA r 0.6-0.8 and slope 0.51 (CE) .. 0.70 (SW), 0.58 ice-sheet mean (`alpha_t2m=0.6`); HadCRUT5 manages r 0.1-0.6 and misses the 1930s-40s warm and 1970s-90s cool periods entirely.
- PAGES2k (2019) global reconstruction + HadCRUT5 annual (`--source global`): the alaska-forecast splice, kept as the fallback and for the Arctic-amplification analysis.

### 8. Greenland drainage basins (labels)
- Mouginot & Rignot (2019), *Glacier catchments/basins for the Greenland Ice Sheet*, Dryad, DOI [10.7280/D1WT11](https://doi.org/10.7280/D1WT11): 260 basins in 7 regions, shapefile in EPSG:3413. Used as the per-"glacier" label field (`rgi_label`, `rgi_id`) that the velocity marginal, divide-flux term and per-basin diagnostics index; ice-sheet basins get `surge_type = 0`. Also the basis for ISMIP-style regional scalars (IMBIE basins are a coarser alternative: <https://imbie.org/imbie-3/drainage-basins/>).

### 9. RGI 6.0 region 05 — peripheral glaciers ✅ on disk
- RGI Consortium (2017), *Randolph Glacier Inventory 6.0*, NSIDC-0770 v6, DOI [10.7265/4m1f-gd79](https://doi.org/10.7265/4m1f-gd79); `05_rgi60_GreenlandPeriphery.shp` (EPSG:4326, 20 261 outlines; `RGIId`, `Surging` 0-3 with 9 = not assigned, `Connect` 0/1/2, `TermType`). `make_dem.py` labels Connect 0-1 glaciers (Connect 2 = dynamically part of the ice sheet keep their basin label) and maps `Surging` 9 → 0. RGI 7.0 (DOI [10.5067/F6JMOVY5NAVZ](https://doi.org/10.5067/F6JMOVY5NAVZ), fields `rgi_id`/`surge_type`) is accepted by the same builder if dropped in.

### 10–11. Surface elevation change
- **ATL15 v5** ✅ on disk: `ATL15_GL_0329_01km_005_02.nc` (cycles 03-29; quarterly `delta_h` ± `delta_h_sigma` 2019-01-01 → 2026-01-01 relative to the 2020-01-01 datum, 1 km, EPSG:3413; `dhdt_lag1/4/8/12/16/20/24` groups). DOI [10.5067/ATLAS/ATL15.005](https://doi.org/10.5067/ATLAS/ATL15.005); v4 DOI [10.5067/ATLAS/ATL15.004](https://doi.org/10.5067/ATLAS/ATL15.004). `make_dhdt.py --source atl15 [--t0 2019 --t1 2025]` fits a weighted trend per pixel (default: the whole record). Note the formal `delta_h_sigma` is small (the propagated slope error has a median of ~2 mm/yr on the coarse domain), so the dh/dt term's `sigma_floor` (0.5 m/yr) sets the effective per-pixel error; margin cells show rates to −22 m/yr.
- **ITS_LIVE Greenland Ice Sheet and Peripheral Glacier Ice Elevation Change** ✅ on disk (Nilsson & Gardner; MEaSUREs G1920V01 v1.1, created 2026-02-10, DOI [10.5067/ICFVI7DKHZJV](https://doi.org/10.5067/ICFVI7DKHZJV)): `Greenland_G1920V01_IceSheetGlacierIceHeight.nc`, 1.92 km EPSG:3413, 384 monthly epochs 1992-01 → 2023-12, `dh`, `h`, `h_dem`, `rms`, `mask`, GEMB/GSFC firn-air-content anomalies. `make_dhdt.py --source itslive_dh --t0 2000 --t1 2020` gives the Hugonnet-style long-window rate with propagated `rms`.
- **ESA CCI Greenland Ice Sheet SEC** (Simonsen & Sørensen; radar altimetry 1992-present, 5 km, 5-year running windows with error), <http://products.esa-icesheets-cci.org/>. Provides a 2000-2020-style window matching the Alaska Hugonnet term (`--source gridded --t0 … --t1 … --rate-var … --err-var …`). A merged CryoSat-2 + ICESat-2 1992-2023 product is described in Ravinder et al. (ESSD 2026).
- **Hugonnet et al. (2021)** 1° tiles (RGI 05, 2000-2020, UTM) for the peripheral glaciers, as in Alaska.

### 12. ArcticDEM mosaic v4.1 — dated ice surface
- PGC, ArcticDEM v4.1 100 m mosaic (`arcticdem_mosaic_100m_v4.1_dem.tif`, EPSG:3413, ellipsoidal heights; bands/aux rasters `count, mad, mindate, maxdate`), <https://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/mosaic/v4.1/100m/> (also 500 m/1 km; no login; AWS registry <https://registry.opendata.aws/pgc-arcticdem/>). Strips span 2007-2020; the median mosaic is tagged nominal 2015 by `make_dem.py`, which subtracts BedMachine's geoid. Using `mindate/maxdate` for per-pixel epochs is a possible refinement (the observation API takes one epoch per product).

### 13. ERA5-Land (optional intercomparison climatology)
- CDS `reanalysis-era5-land-monthly-means`, variables `2m_temperature`, `total_precipitation`, Greenland box (58-84 N, 75-10 W), years of choice (the builder takes `--year`). Needs a CDS API key; the Alaska bundle's `era5.nc` is Alaska-only.

### 14. Airborne radar bed picks (optional)
- **IRMCR2 v1**, *IceBridge MCoRDS L2 Ice Thickness* (CReSIS; 2009-2019, CSV per segment: LAT, LON, TIME, THICK, ELEVATION, FRAME, SURFACE, BOTTOM, QUALITY), DOI [10.5067/GDQ0CUCVTE2Q](https://doi.org/10.5067/GDQ0CUCVTE2Q); **BRMCR2** (pre-IceBridge 1993-2008) at <https://nsidc.org/data/brmcr2/versions/1>. Tens of GB for all of Greenland; Earthdata login. `make_bedradar.py` writes `flightlines.gpkg`; only needed when conditioning on raw picks instead of BedMachine's gridded bed/errbed (which already integrates them via mass conservation).

### 15. End-of-summer snowline / bare-ice extent (optional ELA term)
- No off-the-shelf categorical product like the Alaska tiles. Sources to derive one: Ryan et al. (2019, Sci. Adv. 5:eaav3738) daily MODIS bare-ice/snow classification 2001-2017 (data on request / supplementary); MODIS MOD10A1 / MCD43A3 albedo thresholds (bare ice < ~0.6); GEUS Sentinel-3 SICE bare-ice albedo (<https://dataverse.geus.dk>). Produce GeoTIFFs coded 0/1/2 (nodata/ice/snow) per year under `snowlines/<YYYY-YYYY-average>/`.
- Point validation: PROMICE/GC-Net AWS ablation and ELA series (GEUS Dataverse).

### 16. Accumulation anomaly (optional multiplicative precip forcing)
- Box et al. (2013, J. Climate 26:3919) *Greenland ice sheet net snow accumulation 1600-2009* (ice-sheet-wide series; data via the paper/author), or single-core annual accumulation (NGRIP/NEEM/GISP2 from NOAA Paleoclimatology). Two-column CSV `year,accum`.

### 18. Regional-climate-model SMB (validation / ISMIP7 reference)
- MAR v3.12-3.14 (Fettweis; ULiège, <https://mar.cnrs.fr/>, Zenodo archives e.g. <https://zenodo.org/records/5024965>), RACMO2.3p2 statistically downscaled to 1 km (Noël et al. 2019; PANGAEA <https://doi.pangaea.de/10.1594/PANGAEA.904428>, Zenodo <https://zenodo.org/records/3367211>; full daily fields on request from UU/IMAU). The 1979-2019 MAR mean already sits in glide's example file (`smb_mar` in the bootstrap domain). Use: compare the calibrated enthalpy-model SMB, and as the reference climatology the ISMIP7 SMB anomalies are defined against.

### 19. ISMIP7 forcing (forecast phase)
- ISMIP7 Greenland MIPkit: CMIP6-derived SMB anomalies (MAR/RACMO-downscaled CESM2-WACCM historical, SSP1-2.6, SSP3-7.0, SSP5-3.4-over, SSP5-7.5 …), ocean thermal forcing and the ISMIP6 retreat parameterisation (Slater et al. 2020), distributed via Globus; protocol at <https://www.ismip.org/research/ismip7>. Projections start 2015 (historical ends 2014); output on the standard ISMIP7 grid; variable request per <https://github.com/ismip>. `export_ismip.py` writes the initial state with ISMIP6 names — reconcile with the final ISMIP7 request.

- **Ocean thermal forcing (EN4 reanalysis-based, ISMIP7 kit)** ✅ on disk 2026-09-11: `ocean/tf/tf_GrIS_EN4_OCX_ocean-1000m_v1_{1950..2026}.nc`, monthly `tf(time, y, x)` in degC above the in-situ freezing point on the ISMIP 1 km grid (`y` ascending; NASA GSFC, code github.com/ehultee/gris-iceocean-process: Verjans bias correction, Slater inland mapping into fjords and under marine-based ice; finite on ~61% of the grid, NaN elsewhere). At the marine ice margin (3255 cells with a below-sea-level non-ice neighbour) the product is defined in the cell for 61% and within 2 km for 99%; margin-mean annual-mean TF ~2.5-3.0 degC (1980s minimum, 2010s maximum, ~0.5 degC apart; monthly max ~1.5 degC higher). `make_thermal_forcing.py` writes annual mean and annual max for every complete year (1950-2025; the partial 2026 file is skipped) plus `tf_dist` (km to the native product), nearest-filled to 10 km; `forward_standalone.py` maps it to glide's calving margin q.

### 20. Terminus positions (optional)
- TermPicks (Goliber et al. 2022, TC 16:3215; 39 060 traces, 278 glaciers, 1916-present; Zenodo) and MEaSUREs NSIDC-0642 annual SAR termini (Joughin et al. 2015). For an extent observation at the exact velocity epoch, or to time-stamp the BedMachine mask locally.

### 21. RACMO2.3p2-ERA5 1 km monthly forcing (ISMIP7 kit) ✅ on disk
- IMAU RACMO2.3p2 forced by ERA5, statistically downscaled to 1 km (Noël), regridded to the ISMIP Greenland 1 km grid for ISMIP7 (D. Dunmire, xesmf; `ts` bilinear, `pr` conservative). Yearly files `climate/RACMO2.3p2-ERA/{tas,ts,pr}/{var}_GrIS_RACMO2.3p2-ERA_OCX_SDBN1-1000m_v1_{year}.nc`, 12 monthly fields each, 1958-2025 (68 years), `y` ascending. Finite on 51% of the grid (Greenland ice sheet, periphery and surrounding land/ocean strip; the Ellesmere ice caps in the BedMachine mask are NaN). `tas` is the bias-corrected 2 m temperature (`t2mcorr`, monthly mean, unclipped — the variable the analyses use); `ts` is the same field **clipped at 273.15 K** (`ncap2 where(>273.15)`), i.e. a thermal boundary condition: in July ~30% of ice cells sit on the cap. `pr` is water-equivalent mass flux.
- Uses: (a) intercomparison with the CARRA2 forcing (`analysis/compare_racmo_carra.py --year 2000`); (b) `analysis/arctic_amplification.py`: per-cell, per-month regression of `tas` and `pr` on the HadCRUT5 global anomaly → local amplification `A(x, m)` (K per K global) and precipitation sensitivity, 1961-1990 climatologies, and a hybrid reconstruction `T = Tclim + A·ΔG`, `P = Pclim·(1 + s·ΔG)` that `--hybrid-years` can write year by year to `climate/RACMO2.3p2-ERA-hybrid/` in the same layout for spin-up years before 1958 (not written by default) (ΔG from the PAGES2k+HadCRUT splice in `temperature_anomaly.nc`).

## Access checklist
- NASA Earthdata account (NSIDC: 1, 2, 3, 9, 10, 14, 20) — `earthaccess.login()` then `earthaccess.download(earthaccess.search_data(short_name=...))`.
- CDS API key (5 if re-downloading CARRA, 13).
- No login: 4 (AWS), 8 (Dryad), 11 (ESA CCI portal), 12 (PGC), 6 (Met Office / NOAA).
- `common_data/` is self-contained (no symlinks into alaska-forecast). The temperature-anomaly files (6) are copies of the Alaska bundle's; CARRA2 (5) sits in `climate/carra2/`.
- Once a curated bundle is assembled, host it like the Alaska one and point `download_common_data.py --manifest` at its `latest.json` (schema: `files: [{filename, url|base_url, sha256, size_bytes}]`).
