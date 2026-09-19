"""
ISMIP7 projection forcing -> model_inputs/ismip7/<gcm>_<scenario>/

The ISMIP7 Greenland kit (ismip7_data/<gcm>/<scenario>/) holds one NetCDF per
calendar year and variable on the ISMIP 1 km grid (EPSG:3413, y ascending):

  <atm>/tas   monthly 2 m air temperature [K]         (dEBM2 / SDBN1 downscaling,
  <atm>/pr    monthly precipitation [kg m-2 s-1]       NaN off the ice sheet: 46%
                                                       of the grid, 12% of the
                                                       domain's ice cells)
  <ocean>/tf  monthly ocean thermal forcing [degC]     (Verjans bias correction,
                                                       Slater inland mapping)

`historical` runs 1850-2014, the scenarios 2015-2300. This script splices
historical + scenario into ONE seamless record (scenario years win where
both exist) and writes what forward_projection.py needs to run 1800-2300
without touching CARRA2 / Vinther / EN4:

  catalogue.json       year -> {tas, pr, tf} file paths of the splice, the
                       native->domain grid mapping, the window attributes
  climate.nc           monthly climatologies on the domain grid (degC, m ice/yr):
                         tas_clim / pr_clim   over --clim-years (default the
                                              CARRA2 climatology window 1986-2025;
                                              the anomaly mode's reference)
                         tas_pre  / pr_pre    over --pre-years (default 1850-1879;
                                              the constant forcing before the record)
                         fill_dist            km to the nearest native (finite) cell
                       plus ice-sheet-mean comparisons with the CARRA2
                       climatology in the attributes (the bias to look at)
  thermal_forcing.nc   annual mean / max TF per year of the splice, in
                       make_thermal_forcing.py's layout (tf_mean, tf_max,
                       tf_dist), blanked farther than --band-km from the ice
                       so 451 years stay < 1 GB

The yearly tas / pr fields themselves are NOT copied (209 GB): the driver
reads them from the kit per step through catalogue.json.

    python make_ismip7_forcing.py --domain-path ../domains/greenland \\
        --gcm CESM2-WACCM --scenario ssp126 [--atm dEBM2-1000m] [--ocean ocean-1000m]
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import ndimage

from domain_grid import grid_from_dem
from make_thermal_forcing import _to_domain, build_thermal_forcing

ISMIP7_DIR = Path('../ismip7_data')
SECONDS_PER_YEAR = 31536000.0       # glare.enthalpy.SECONDS_PER_YEAR
ICE_DENSITY = 917.0
FILE_PATTERN = '{var}_GrIS_{gcm}_{scenario}_{product}_*_{year}.nc'


def _scan(ismip7_dir: Path, gcm: str, scenario: str, product: str, var: str) -> dict:
    """year -> path for one variable of one scenario."""
    pat = ismip7_dir / gcm / scenario / product / var / FILE_PATTERN.format(
        var=var, gcm=gcm, scenario=scenario, product=product, year='*')
    out = {}
    for f in sorted(glob.glob(str(pat))):
        m = re.search(r'_(\d{4})\.nc$', f)
        if m:
            out[int(m.group(1))] = str(Path(f).resolve())
    return out


def build_catalogue(ismip7_dir, gcm: str, scenario: str, atm: str, ocean: str,
                    historical: str = 'historical') -> dict:
    """Seamless splice: the scenario's years, historical for every earlier
    year. Returns {'years': {year: {var: path}}, 'gaps': {var: [years]}}."""
    ismip7_dir = Path(ismip7_dir)
    per_var = {}
    for var, product in (('tas', atm), ('pr', atm), ('tf', ocean)):
        scen = _scan(ismip7_dir, gcm, scenario, product, var)
        hist = _scan(ismip7_dir, gcm, historical, product, var) if scenario != historical else {}
        first_scen = min(scen) if scen else None
        files = {y: f for y, f in hist.items() if first_scen is None or y < first_scen}
        files.update(scen)
        per_var[var] = files
    years = sorted(set().union(*(set(v) for v in per_var.values())))
    if not years:
        raise FileNotFoundError(f"no ISMIP7 files for {gcm}/{scenario} ({atm}, {ocean}) under {ismip7_dir}")
    cat, gaps = {}, {}
    for y in years:
        cat[y] = {var: per_var[var].get(y) for var in per_var}
    for var in per_var:
        gaps[var] = [y for y in years if per_var[var].get(y) is None]
    return {'years': cat, 'gaps': gaps}


def _open_native(path: str, var: str) -> xr.DataArray:
    return xr.open_dataset(path, decode_times=False)[var]


def _grid_mapping(native: xr.DataArray, template: xr.DataArray) -> str:
    """'identity' | 'flip_y' | 'regrid' (see make_thermal_forcing._to_domain)."""
    if (native.x.size == template.x.size and native.y.size == template.y.size
            and np.allclose(native.x.values, template.x.values)):
        if np.allclose(native.y.values, template.y.values):
            return 'identity'
        if np.allclose(native.y.values[::-1], template.y.values):
            return 'flip_y'
    return 'regrid'


def monthly_climatology(files: dict, var: str, years, template: xr.DataArray,
                        convert) -> np.ndarray:
    """(12, ny, nx) calendar-month mean of `var` over `years` (years missing
    from `files` are skipped with a message), on the domain grid, converted
    by `convert` (K -> degC, kg m-2 s-1 -> m ice / yr)."""
    ny, nx = template.sizes['y'], template.sizes['x']
    acc = np.zeros((12, ny, nx), dtype='float64')
    n = 0
    for y in years:
        f = files.get(y, {}).get(var)
        if f is None:
            print(f"  {var} {y}: no file, skipped in the climatology")
            continue
        da = _open_native(f, var)
        a = da.values.astype('float32')
        if a.shape[0] != 12:
            raise ValueError(f"{f}: {a.shape[0]} months")
        for m in range(12):
            acc[m] += np.nan_to_num(_to_domain(a[m], da, template), nan=0.0)
        n += 1
        if n == 1:
            finite = np.isfinite(_to_domain(a[0], da, template))
        print(f"  {var} {y}: native mean {np.nanmean(a):.4g}", flush=True)
    if n == 0:
        raise ValueError(f"no {var} files in {list(years)[0]}-{list(years)[-1]}")
    clim = convert(acc / n).astype('float32')
    clim[:, ~finite] = np.nan
    return clim


def write_ctrl_climatology(files: dict, years, out_path: Path, gcm: str, atm: str, source: str) -> Path:
    """The ISMIP7 control forcing built the way the kit builds its own ctrl
    files: the monthly climatology of tas and pr over `years` (2000-2029 of
    historical + ssp126), written ONCE on the native grid in the kit's units
    and layout (the kit's ctrl years are bit-identical repeats of such a
    climatology). The catalogue points every ctrl year at this file, so the
    driver reads it like any kit year. Using our own dEBM2 climatology keeps
    the ctrl run in the same downscaling product as the historical and
    scenario runs; the kit's ctrl atmosphere exists only in SDBN1 /
    GEMB-SDBN1, which has no historical to splice against."""
    acc, n, ref = {}, {}, {}
    footprint = None
    for y in years:
        for var in ('tas', 'pr'):
            f = files.get(y, {}).get(var)
            if f is None:
                continue
            da = _open_native(f, var)
            a = da.values.astype('float64')
            if a.shape[0] != 12:
                raise ValueError(f"{f}: {a.shape[0]} months")
            acc[var] = acc.get(var, 0.0) + np.nan_to_num(a)
            n[var] = n.get(var, 0) + 1
            if var not in ref:
                ref[var] = xr.open_dataset(f, decode_times=False)
            if footprint is None:
                footprint = np.isfinite(a[0])
        print(f"  ctrl climatology {y}", flush=True)
    if set(n) != {'tas', 'pr'}:
        raise ValueError(f"no tas/pr files in {years[0]}-{years[-1]} for the ctrl climatology")
    r0 = ref['tas']
    ds = xr.Dataset(coords={'time': r0['time'].values, 'y': r0['y'].values, 'x': r0['x'].values})
    for var in ('tas', 'pr'):
        clim = (acc[var] / n[var]).astype('float32')
        clim[:, ~footprint] = np.nan
        ds[var] = xr.DataArray(clim, dims=('time', 'y', 'x'), attrs=dict(ref[var][var].attrs))
        ds[var].attrs['comment'] = f"monthly climatology over {years[0]}-{years[-1]} of {source} ({n[var]} years), repeated for every ctrl year"
    if 'time_bnds' in r0:
        ds['time_bnds'] = r0['time_bnds']
    ds.attrs.update(title=f"ISMIP7 ctrl forcing for {gcm}: {atm} climatology {years[0]}-{years[-1]}",
                    source=source, years=f"{years[0]}-{years[-1]}")
    ds.to_netcdf(out_path, encoding={v: dict(zlib=True, complevel=4) for v in ('tas', 'pr')})
    print(f"wrote {out_path}")
    return out_path


def _ice_means(field12: np.ndarray, ice: np.ndarray) -> dict:
    """Ice-sheet-mean annual / JJA statistics of a (12, ny, nx) field."""
    ok = ice & np.isfinite(field12[0])
    ann = float(field12[:, ok].mean())
    jja = float(field12[5:8][:, ok].mean())
    return {'annual': ann, 'jja': jja, 'n_cells': int(ok.sum())}


def build_ismip7_forcing(domain_path, gcm: str, scenario: str, atm: str = 'dEBM2-1000m',
                         ocean: str = 'ocean-1000m', ismip7_dir=None, clim_years=(1986, 2025),
                         pre_years=(1850, 1879), band_km: float = 30.0, fill_km: float = 10.0,
                         skip_climate: bool = False, skip_ocean: bool = False,
                         output_dir=None, fill_method: str = 'marine',
                         ctrl_from: str = 'ssp126', ctrl_years=(2000, 2029)) -> Path:
    domain_path = Path(domain_path)
    ismip7_dir = Path(ismip7_dir) if ismip7_dir else ISMIP7_DIR
    out_dir = Path(output_dir) if output_dir else domain_path / 'model_inputs' / 'ismip7' / f'{gcm}_{scenario}'
    out_dir.mkdir(parents=True, exist_ok=True)

    if scenario == 'ctrl':
        # atmosphere: historical + `ctrl_from` (the climatology's source and the
        # anomaly reference, as for every other run); ocean: the kit's ctrl tf
        cat = build_catalogue(ismip7_dir, gcm, ctrl_from, atm, ocean)
        cat_ctrl = build_catalogue(ismip7_dir, gcm, 'ctrl', atm, ocean)
        for y, entry in cat_ctrl['years'].items():
            if entry.get('tf'):
                cat['years'].setdefault(y, {'tas': None, 'pr': None, 'tf': None})['tf'] = entry['tf']
        cat['gaps']['tf'] = [y for y in sorted(cat['years']) if not cat['years'][y]['tf']]
        print(f"ctrl: atmosphere = {atm} climatology {ctrl_years[0]}-{ctrl_years[1]} of historical + {ctrl_from}, "
              f"ocean = the kit's ctrl {ocean} tf")
    else:
        cat = build_catalogue(ismip7_dir, gcm, scenario, atm, ocean)
    years = sorted(cat['years'])
    print(f"{gcm} {scenario}: {years[0]}-{years[-1]} ({len(years)} years); gaps: "
          + ", ".join(f"{v}: {len(g)}" for v, g in cat['gaps'].items()))
    for var, g in cat['gaps'].items():
        if g:
            print(f"  {var} missing for {g[:5]}{' ...' if len(g) > 5 else ''}")

    dem = xr.load_dataset(domain_path / 'model_inputs' / 'gridded_dem.nc')
    grid = grid_from_dem(dem)
    template = grid.template()
    ice = dem['rgi_mask'].values.astype(bool)

    first_tas = next(cat['years'][y]['tas'] for y in years if cat['years'][y]['tas'])
    mapping = _grid_mapping(_open_native(first_tas, 'tas'), template)
    print(f"native -> domain grid: {mapping}")

    to_degc = lambda a: a - 273.15
    to_ice = lambda a: a * SECONDS_PER_YEAR / ICE_DENSITY

    clim_path = out_dir / 'climate.nc'
    if not skip_climate:
        y_clim = [y for y in range(clim_years[0], clim_years[1] + 1)]
        y_pre = [y for y in range(pre_years[0], pre_years[1] + 1)]
        print(f"climatology over {clim_years[0]}-{clim_years[1]}")
        tas_clim = monthly_climatology(cat['years'], 'tas', y_clim, template, to_degc)
        pr_clim = monthly_climatology(cat['years'], 'pr', y_clim, template, to_ice)
        print(f"pre-record climatology over {pre_years[0]}-{pre_years[1]}")
        tas_pre = monthly_climatology(cat['years'], 'tas', y_pre, template, to_degc)
        pr_pre = monthly_climatology(cat['years'], 'pr', y_pre, template, to_ice)
        finite = np.isfinite(tas_clim[0])
        fill_dist = (ndimage.distance_transform_edt(~finite) * grid.resolution / 1000.0).astype('float32')

        months = np.arange(0, 12, dtype=np.float32) / 12
        ds = xr.Dataset(coords={'t': months, 'y': dem.y, 'x': dem.x})
        if 'spatial_ref' in dem:
            ds['spatial_ref'] = dem['spatial_ref']
        dims = ('t', 'y', 'x')
        src = f"ISMIP7 {gcm} {scenario} (+historical) {atm}"
        ds['tas_clim'] = xr.DataArray(tas_clim, dims=dims, attrs=dict(
            units='Deg C', long_name=f'Monthly 2 m air temperature climatology {clim_years[0]}-{clim_years[1]}', source=src))
        ds['pr_clim'] = xr.DataArray(pr_clim, dims=dims, attrs=dict(
            units='m ice equivalent / yr', long_name=f'Monthly precipitation climatology {clim_years[0]}-{clim_years[1]}', source=src))
        ds['tas_pre'] = xr.DataArray(tas_pre, dims=dims, attrs=dict(
            units='Deg C', long_name=f'Monthly 2 m air temperature climatology {pre_years[0]}-{pre_years[1]} (pre-record forcing)', source=src))
        ds['pr_pre'] = xr.DataArray(pr_pre, dims=dims, attrs=dict(
            units='m ice equivalent / yr', long_name=f'Monthly precipitation climatology {pre_years[0]}-{pre_years[1]} (pre-record forcing)', source=src))
        ds['fill_dist'] = xr.DataArray(fill_dist, dims=('y', 'x'), attrs=dict(
            units='km', long_name='Distance to the nearest cell with a native ISMIP7 atmospheric value'))
        ds.attrs.update(gcm=gcm, scenario=scenario, atm_product=atm, grid_mapping=mapping,
                        clim_years=f"{clim_years[0]}-{clim_years[1]}", pre_years=f"{pre_years[0]}-{pre_years[1]}",
                        ice_cells_without_native_value=int((ice & ~finite).sum()),
                        ice_cells=int(ice.sum()))
        # the bias to look at: ice-sheet means against the CARRA2 climatology
        stats = {'ismip7_clim': {'tas': _ice_means(tas_clim, ice), 'pr': _ice_means(pr_clim, ice)},
                 'ismip7_pre': {'tas': _ice_means(tas_pre, ice), 'pr': _ice_means(pr_pre, ice)}}
        carra = domain_path / 'model_inputs' / 'gridded_climate.nc'
        if carra.exists():
            with xr.open_dataset(carra) as c:
                # same cells as the ISMIP7 statistics (its footprint), so the means compare
                t2m = np.where(finite[None], c['monthly_t2m'].values, np.nan)
                p = np.where(finite[None], c['monthly_precip'].values, np.nan)
            stats['carra2'] = {'tas': _ice_means(t2m, ice), 'pr': _ice_means(p, ice),
                               'years': str(c.attrs.get('climatology_years', ''))}
        for k, v in stats.items():
            ds.attrs[f'{k}_tas_annual_degC'] = v['tas']['annual']
            ds.attrs[f'{k}_tas_jja_degC'] = v['tas']['jja']
            ds.attrs[f'{k}_pr_annual_m_yr'] = v['pr']['annual']
            print(f"  {k:12s} ice-sheet mean (ISMIP7 footprint, {v['tas']['n_cells']} cells): "
                  f"T annual {v['tas']['annual']:+.2f} C, JJA {v['tas']['jja']:+.2f} C, P {v['pr']['annual']:.3f} m/yr")
        enc = {v: dict(zlib=True, complevel=4) for v in ('tas_clim', 'pr_clim', 'tas_pre', 'pr_pre', 'fill_dist')}
        ds.to_netcdf(clim_path, encoding=enc)
        print(f"wrote {clim_path} ({clim_path.stat().st_size / 1e6:.0f} MB); "
              f"{(ice & ~finite).sum()} of {ice.sum()} ice cells have no native value (nearest-filled at run time)")

    def write_catalogue():
        cat_out = {'gcm': gcm, 'scenario': scenario, 'atm_product': atm, 'ocean_product': ocean,
                   'ismip7_dir': str(ismip7_dir.resolve()), 'grid_mapping': mapping,
                   'clim_years': list(clim_years), 'pre_years': list(pre_years),
                   'years': {str(y): cat['years'][y] for y in years}, 'gaps': cat['gaps'],
                   'climate': str(clim_path) if clim_path.exists() else None,
                   'thermal_forcing': str(tf_path) if tf_path.exists() else None}
        with open(out_dir / 'catalogue.json', 'w') as f:
            json.dump(cat_out, f, indent=1)
        print(f"wrote {out_dir / 'catalogue.json'}")

    if scenario == 'ctrl':
        # after the climatologies (which use the real 2015-2025 of ctrl_from):
        # every ctrl year reads the repeated climatology instead
        ctrl_tf = _scan(ismip7_dir, gcm, 'ctrl', ocean, 'tf')      # the kit's own ctrl years (2015-)
        first_ctrl = min(ctrl_tf) if ctrl_tf else 2015
        y_ctrl = [y for y in range(ctrl_years[0], ctrl_years[1] + 1)]
        ctrl_file = str(write_ctrl_climatology(cat['years'], y_ctrl, out_dir / 'ctrl_climatology.nc', gcm, atm,
                                               f"{gcm} historical + {ctrl_from} {atm}").resolve())
        for y in years:
            if y >= first_ctrl:
                cat['years'][y]['tas'] = ctrl_file
                cat['years'][y]['pr'] = ctrl_file
        cat['gaps']['tas'] = cat['gaps']['pr'] = [y for y in years if not cat['years'][y]['tas']]
        print(f"ctrl years {first_ctrl}-{years[-1]} read {Path(ctrl_file).name}")

    tf_path = out_dir / 'thermal_forcing.nc'
    write_catalogue()            # the climate part is usable while the TF record streams
    if not skip_ocean:
        tf_files = {y: cat['years'][y]['tf'] for y in years if cat['years'][y]['tf']}
        if tf_files:
            print(f"thermal forcing {min(tf_files)}-{max(tf_files)}, blanked beyond {band_km:g} km from the ice")
            build_thermal_forcing(domain_path, fill_km=(None if fill_method == 'marine' else fill_km),
                                  files=tf_files, output_path=tf_path,
                                  band_km=band_km, method=fill_method,
                                  source=(f"ISMIP7 {gcm} {scenario} (+historical) {ocean} thermal forcing "
                                          f"(Verjans bias correction to EN4 1985-2014, Slater inland mapping), "
                                          f"monthly 1 km, annual statistics"),
                                  extra_attrs=dict(gcm=gcm, scenario=scenario, ocean_product=ocean))
        else:
            print(f"no {ocean}/tf files for {gcm}/{scenario}: no thermal_forcing.nc (constant margins)")

    write_catalogue()
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain-path", required=True)
    ap.add_argument("--gcm", default="CESM2-WACCM")
    ap.add_argument("--scenario", default="ssp126")
    ap.add_argument("--atm", default="dEBM2-1000m", help="atmospheric product directory (tas, pr)")
    ap.add_argument("--ocean", default="ocean-1000m", help="ocean product directory (tf)")
    ap.add_argument("--ismip7-dir", default=None)
    ap.add_argument("--clim-years", type=int, nargs=2, default=(1986, 2025), metavar=("Y0", "Y1"),
                    help="reference window of tas_clim / pr_clim (default: the CARRA2 climatology window)")
    ap.add_argument("--pre-years", type=int, nargs=2, default=(1850, 1879), metavar=("Y0", "Y1"),
                    help="window of the constant pre-record forcing")
    ap.add_argument("--band-km", type=float, default=30.0, help="keep TF within this distance of the ice")
    ap.add_argument("--fill-km", type=float, default=10.0, help="nearest-neighbour TF fill beyond the native product")
    ap.add_argument("--skip-climate", action="store_true")
    ap.add_argument("--skip-ocean", action="store_true")
    ap.add_argument("--fill-method", choices=("marine", "nearest"), default="marine")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--ctrl-from", default="ssp126",
                    help="ctrl only: the scenario whose 2015+ years complete the ctrl climatology and the anomaly reference")
    ap.add_argument("--ctrl-years", type=int, nargs=2, default=(2000, 2029), metavar=("Y0", "Y1"),
                    help="ctrl only: the climatology window (the protocol's 2000-2029)")
    a = ap.parse_args()
    build_ismip7_forcing(a.domain_path, a.gcm, a.scenario, atm=a.atm, ocean=a.ocean, ismip7_dir=a.ismip7_dir,
                         clim_years=tuple(a.clim_years), pre_years=tuple(a.pre_years), band_km=a.band_km,
                         fill_km=a.fill_km, skip_climate=a.skip_climate, skip_ocean=a.skip_ocean,
                         ctrl_from=a.ctrl_from, ctrl_years=tuple(a.ctrl_years),
                         output_dir=a.output_dir, fill_method=a.fill_method)
