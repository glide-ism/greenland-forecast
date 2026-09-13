# greenland-forecast

Time-dependent Bayesian inverse modelling of the Greenland ice sheet with
[glide](https://github.com/glide-ism/glide) / glare / ggapp, as the
initialization step for ISMIP7 experiment submissions. A port of
[alaska-forecast](https://github.com/glide-ism/alaska-forecast) from
mountain-range domains to the ice sheet: the `glacier_inverse` library is the
same code (three additive changes, see `CLAUDE.md`), the preprocessing is
rewritten for Greenland data sources, and `export_ismip.py` writes the
calibrated state in ISMIP variable conventions.

## Setup

```
pip install -r requirements.txt      # pick the cupy wheel for your CUDA
```

GPU is mandatory (RTX-class, ≥ 24 GB for the 1 km ice sheet). If you develop
the glide-ism libraries alongside, install them editable
(`pip install -e ../glide[cuda12]` etc.).

## Data

The observational products are NOT checked in and not yet bundled.
`DATA_MANIFEST.md` lists every dataset (product, version, DOI/URL, epoch,
expected path under `common_data/`, which builder consumes it, priority);
`data_manifest.json` is the machine-readable twin. Once a curated bundle
exists, `python download_common_data.py --manifest <latest.json> --extract`
fetches it as in alaska-forecast.

This repo's `common_data/` is self-contained — no symlinks into
alaska-forecast (a shared directory once let a Greenland download overwrite
an Alaska file). The only inputs common to both projects are the small
global temperature-anomaly files (`climate/temp_anomaly/`), copied. The
bootstrap domain falls back to a placeholder parametric climatology if the
CARRA2 files (`climate/carra2/`) are missing.

## Domains

- `domains/greenland` — the science domain: whole ice sheet + peripheral
  glaciers on the ISMIP standard 1 km grid (EPSG:3413; `local_data/domain.json`
  preset `ismip_greenland`, resolution 1000/2000/4000/8000 all nest).
- `domains/greenland_coarse` — a 1.8 km development domain bootstrapped from
  glide's example file (BedMachine + MEaSUREs 1995-2015 + MAR) without the
  full bundle; runs today:

```
python preprocessing/bootstrap_from_glide_example.py --domain-path domains/greenland_coarse
python smoke_test.py
```

A regional sub-domain is a `domain.json` with `"bbox": [xmin, ymin, xmax, ymax]`
in EPSG:3413 metres (snapped onto the ISMIP lattice) and any resolution.

## Workflow

```
python preprocessing/make_all.py --domain-path domains/greenland --year 2015   # -> model_inputs/GLIDE_inputs.nc
#   options: --velocity-source {itslive,measures_annual,measures_multiyear,glide_example}
#            --dhdt-source {atl15,itslive_dh,gridded,hugonnet} [--dhdt-t0 2000 --dhdt-t1 2020]
python inverse.py            # MAP solve (DOMAIN constant at the top)
python rto_sample.py         # posterior samples (still on the pre-migration API, as in alaska)
python export_ismip.py --domain-path domains/greenland --date 2015 [--regrid-to 4000]
```

See `CLAUDE.md` for the architecture and what changed in the port.

## Analysis

`analysis/` holds standalone diagnostics that are not part of the model input
pipeline: `compare_racmo_carra.py` (RACMO2.3p2-ERA5 vs the CARRA2 forcing for one
year) and `arctic_amplification.py` (local warming per degree of HadCRUT5 global
warming from the 1958-2025 RACMO record, precipitation sensitivity, and the hybrid
pre-1958 RACMO reconstruction for spin-ups). Outputs go to `analysis/output/`.
