"""Run the full preprocessing pipeline for a single Greenland domain.

    geometry (BedMachine + basins) ─┬─ velocity           (MEaSUREs / ITS_LIVE)
                                    ├─ radar picks        (optional, IceBridge)
                                    ├─ dH/dt              (optional, ATL15 / CCI)
                                    ├─ snowline           (optional)
                                    ├─ insolation         (gtic, needs year)
                                    ├─ climate CARRA2     (height-level t + tp, make_carra_vars.py)
                                    └─ climate ERA5-Land  (optional, needs year)
    temperature anomaly  (independent)
    precip anomaly       (independent, optional)
    thermal forcing      (independent, optional; forward_standalone.py only)
    merged = everything present

Run as a script:
    python make_all.py --domain-path ../domains/greenland --year 2015
"""
import argparse
from pathlib import Path

from make_dem import build_dem
from make_bedradar import build_flightlines
from make_velocity import build_velocity, DEFAULT_SOURCE as DEFAULT_VELOCITY
from make_dhdt import build_dhdt
from make_snowline import build_snowline
from make_insolation import build_insolation
from make_carra_vars import build_climate as build_climate_carra
from make_era5land_vars import build_climate as build_climate_era5land, ERA5LAND_PATH
from make_temperature_anomaly import build_temperature_anomaly
from make_precip_anomaly import build_precip_anomaly
from make_thermal_forcing import build_thermal_forcing
from make_merged import build_merged


def _banner(step: str) -> None:
    print(f"\n=== {step} ===", flush=True)


def _optional(step, fn, *args, **kwargs):
    _banner(step)
    try:
        return fn(*args, **kwargs)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"skipped ({type(e).__name__}: {e})")
        return None


def run_all(domain_path: str, year: int, velocity_source: str = DEFAULT_VELOCITY,
            dhdt_source: str = 'atl15', with_radar: bool = False,
            skip_insolation: bool = False, dhdt_window=(None, None),
            extra_dhdt=()) -> None:
    Path(domain_path, 'model_inputs').mkdir(parents=True, exist_ok=True)

    _banner("Geometry (BedMachine)")
    build_dem(domain_path)

    _banner(f"Velocity ({velocity_source})")
    build_velocity(domain_path, source=velocity_source)

    if with_radar:
        _optional("Radar bed picks", build_flightlines, domain_path)

    _optional(f"dH/dt ({dhdt_source})", build_dhdt, domain_path, source=dhdt_source,
              t0=dhdt_window[0], t1=dhdt_window[1])
    for src, t0, t1, nm in extra_dhdt:
        _optional(f"dH/dt extra ({src} {t0}-{t1} -> gridded_dhdt_{nm}.nc)", build_dhdt,
                  domain_path, source=src, t0=t0, t1=t1, name=nm)
    _optional("Snowline", build_snowline, domain_path)

    if not skip_insolation:
        _banner(f"Insolation (year={year})")
        build_insolation(domain_path, year=year)

    _banner("Climate: CARRA2 height-level climatology")
    build_climate_carra(domain_path)

    if ERA5LAND_PATH.exists():
        _optional(f"Climate: ERA5-Land (year={year})", build_climate_era5land,
                  domain_path, year=year)

    _banner("Temperature anomaly")
    build_temperature_anomaly(domain_path)

    _banner("Precip anomaly")
    build_precip_anomaly(domain_path)

    _optional("Ocean thermal forcing (ISMIP7 TF)", build_thermal_forcing, domain_path)

    _banner("Merge")
    build_merged(domain_path)
    _banner("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-path", type=str, required=True)
    parser.add_argument("--year", type=int, default=2015,
                        help="solar-geometry year (the climatology is multi-year)")
    parser.add_argument("--velocity-source", type=str, default=DEFAULT_VELOCITY)
    parser.add_argument("--dhdt-source", type=str, default='atl15')
    parser.add_argument("--with-radar", action="store_true",
                        help="also build flightlines.gpkg from IceBridge MCoRDS L2")
    parser.add_argument("--skip-insolation", action="store_true")
    parser.add_argument("--dhdt-t0", type=float, default=None)
    parser.add_argument("--dhdt-t1", type=float, default=None)
    parser.add_argument("--extra-dhdt", action="append", default=[], metavar="SOURCE:T0:T1:NAME",
                        help="additional dh/dt product over another window, e.g. "
                             "itslive_dh:2000:2019:measures (repeatable)")
    args = parser.parse_args()
    extra = []
    for spec in args.extra_dhdt:
        src, t0, t1, nm = spec.split(":")
        extra.append((src, float(t0), float(t1), nm))
    run_all(args.domain_path, args.year, args.velocity_source, args.dhdt_source,
            args.with_radar, args.skip_insolation, (args.dhdt_t0, args.dhdt_t1), extra)
