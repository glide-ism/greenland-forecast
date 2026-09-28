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
from make_dhdt import build_dhdt, TIME_SERIES_SOURCES
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
            extra_dhdt=(), dhdt_method: str = None,
            dhdt_endpoint_window: float = 2.0, dem_smooth_km: float = 1.0) -> None:
    """`dhdt_method`: 'endpoint' | 'trend' for the primary dh/dt product
    (None = make_dhdt's default: endpoint for the time-series sources).
    `extra_dhdt` entries are (source, t0, t1, name[, method]); an entry
    without a method takes `dhdt_method`."""
    Path(domain_path, 'model_inputs').mkdir(parents=True, exist_ok=True)

    _banner("Geometry (BedMachine)")
    build_dem(domain_path, smooth_sigma_km=dem_smooth_km)

    _banner(f"Velocity ({velocity_source})")
    build_velocity(domain_path, source=velocity_source)

    if with_radar:
        _optional("Radar bed picks", build_flightlines, domain_path)

    def method_for(src, requested):
        # the method applies to the time-series sources only; a gridded /
        # hugonnet rate is used as provided (passing 'endpoint' there would
        # raise, and _optional would silently skip the product)
        return requested if src in TIME_SERIES_SOURCES else None

    meth = method_for(dhdt_source, dhdt_method)
    _optional(f"dH/dt ({dhdt_source}, {meth or 'default method'})", build_dhdt, domain_path,
              source=dhdt_source, t0=dhdt_window[0], t1=dhdt_window[1],
              method=meth, endpoint_window=dhdt_endpoint_window)
    for src, t0, t1, nm, *rest in extra_dhdt:
        meth = method_for(src, rest[0] if rest else dhdt_method)
        _optional(f"dH/dt extra ({src} {t0}-{t1}, {meth or 'default method'} -> gridded_dhdt_{nm}.nc)",
                  build_dhdt, domain_path, source=src, t0=t0, t1=t1, name=nm,
                  method=meth, endpoint_window=dhdt_endpoint_window)
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
    parser.add_argument("--dem-smooth-km", type=float, default=1.0,
                        help="Gaussian smoothing of the native BedMachine / ArcticDEM fields before resampling, sigma in km (0 = none)")
    parser.add_argument("--dhdt-t0", type=float, default=None)
    parser.add_argument("--dhdt-t1", type=float, default=None)
    parser.add_argument("--dhdt-method", choices=('endpoint', 'trend'), default='endpoint',
                        help="rate definition for the atl15 / itslive_dh series (make_dhdt.py): "
                             "endpoint = difference of end-window means, the quantity the model's "
                             "two-snapshot rate represents (default); trend = WLS slope")
    parser.add_argument("--dhdt-endpoint-window", type=float, default=2.0,
                        help="width (yr) of the end windows of the endpoint method")
    parser.add_argument("--extra-dhdt", action="append", default=[], metavar="SOURCE:T0:T1:NAME[:METHOD]",
                        help="additional dh/dt product over another window, e.g. "
                             "itslive_dh:1992:2019:measures (repeatable); METHOD (endpoint | trend) "
                             "overrides --dhdt-method for that product")
    args = parser.parse_args()
    extra = []
    for spec in args.extra_dhdt:
        parts = spec.split(":")
        if len(parts) not in (4, 5) or (len(parts) == 5 and parts[4] not in ('endpoint', 'trend')):
            parser.error(f"--extra-dhdt {spec!r}: expected SOURCE:T0:T1:NAME[:endpoint|trend]")
        extra.append((parts[0], float(parts[1]), float(parts[2]), parts[3], *parts[4:]))
    run_all(args.domain_path, args.year, args.velocity_source, args.dhdt_source,
            args.with_radar, args.skip_insolation, (args.dhdt_t0, args.dhdt_t1), extra,
            dhdt_method=args.dhdt_method, dhdt_endpoint_window=args.dhdt_endpoint_window, dem_smooth_km=args.dem_smooth_km)
