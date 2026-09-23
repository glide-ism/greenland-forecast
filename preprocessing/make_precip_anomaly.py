"""Build the precipitation (accumulation) anomaly time series for a domain.

    python make_precip_anomaly.py --domain-path ../domains/greenland
    python make_precip_anomaly.py --domain-path ../domains/greenland --measure

Two sources, in order of preference.

`reconstruction` (used when the CSV is present): alaska-forecast scales precip
by a smoothed Mt. Hunter ice-core accumulation record. For Greenland the
analogue is an ice-sheet-wide accumulation reconstruction: Box et al. (2013,
J. Climate) 1600-2009 net snow accumulation from 86 cores + RACMO2, or a single
long core (NGRIP / NEEM / GISP2 annual accumulation from NOAA
Paleoclimatology). Provide it as a two-column CSV (`year,accum`, any units --
the inverse normalises to `base_precip_year`) at
common_data/climate/precip_anomaly/greenland_accumulation.csv. Years outside
its span are filled from the index scaling below, rescaled to the
reconstruction's own mean over the overlap, because `forward.simulate` looks
the series up by year in a dict and clips only at the top: a spin-up starting
before the reconstruction would otherwise raise.

`index` (the fallback, 2026-09-22): scale the temperature index. Accumulation
was previously held at the modern climatology for the whole pre-reanalysis
spin-up -- `alpha_precip` defaulted to 0 and no file existed, so
`forward.simulate` set `precip_multiplier = 1`. Over a few hundred years that
is minor; over the 2000-3000 yr spin-up the relaxation-time measurement calls
for, it is a one-signed bias, because the Little Ice Age was both colder AND
drier than the reference climate and only the temperature half was applied.

Either way this writes the scalar series the EXISTING library pathway already
consumes, so no library change is involved:

    p_ratio(step)     = <stored>_index-years / stored(base_precip_year)
    precip_multiplier = 1 + alpha_precip (p_ratio - 1)

and `_term_forcing` applies that multiplier ONLY to index years -- record years
take `precip_ * yearly.precip_ratio(year)` from the yearly climate file -- so
it cannot double-count with the reanalysis forcing.

The index series is

    R(t) = exp(gamma_ann * dTann_dindex * I(t)),  renormalized so R(base) = 1

with I the JJA index in `temperature_anomaly.nc`, `gamma_ann` the fractional
precipitation change per K of ice-sheet ANNUAL temperature (the physically
comparable quantity), and `dTann_dindex` the regression of ice-sheet annual
temperature on the index. Set `alpha_precip = 1.0` in the config; because
1 + a(e^x - 1) ~ e^{ax} for small x, `alpha_precip` then acts as a multiplier
on gamma to better than 0.2 % over the Common Era range, so gamma can be swept
from the config without rebuilding.

CHOOSING GAMMA. `--measure` regresses the ice-sheet precipitation-weighted
annual ratio on temperature over the reanalysis record and prints the result.
Over 1986-2025 with the hybrid forcing it is +2.3 +- 1.2 %/K on ice-sheet
annual temperature (r 0.31), +1.3 +- 1.4 on the index, -0.2 +- 1.2 on JJA:
statistically indistinguishable from zero. That is the expected answer and it
is NOT a calibration -- interannual Greenland precipitation is set by
circulation, not thermodynamics, so the thermodynamic signal is buried, exactly
as Kapsner et al. (1995) found for Holocene GISP2 accumulation while
glacial-interglacial accumulation tracks temperature closely. Using the 40-year
slope would be fitting noise and would silently switch the response off.

The default is therefore prior-driven, not fitted: 5 %/K is the long-standing
Greenland ice-sheet modelling convention, sits below the Clausius-Clapeyron
ceiling (7.3 %/K at 273 K, 9.6 at 253 K -- a saturation bound, not a
precipitation sensitivity), and matches the 3-5 %/K implied by ice-core
accumulation across the glacial transition. The 40-year record neither supports
nor excludes it (5 is 2.3 sigma above the measured 2.3). Sweep it with
`alpha_precip`; the printed table gives the mass-budget cost of the whole
0-7 %/K range so the size of the choice stays visible.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter1d

ACCUMULATION_PATH = Path('../common_data/climate/precip_anomaly/greenland_accumulation.csv')
DEFAULT_SMOOTHING_SIGMA = 10.0  # years

# Fractional precipitation change per K of ice-sheet ANNUAL temperature.
GAMMA_ANNUAL = 0.05
# Ice-sheet annual temperature per unit index, regressed over the reanalysis
# record (hybrid forcing 1986-2025: slope 0.73, r 0.66; JJA slope 0.92, r 0.80
# against the config's alpha_t2m of 0.6). `--measure` recomputes it.
DT_ANN_PER_INDEX = 0.73
RHO_I = 917.0
ICE_AREA_M2 = 1.711e12
EPOCHS = (('common_era', 1, 2000), ('roman', 1, 500), ('medieval', 900, 1200),
          ('lia', 1450, 1850), ('lia_core', 1600, 1800))
T_STARTS = (1700, 1000, 100, 0, -1000)
# First year of the yearly climate record: from here on the forcing carries its
# own per-year precip_ratio and the index multiplier does not apply.
REC_START = 1986


def _resolve(path):
    p = Path(path)
    return p if p.is_absolute() else Path(__file__).resolve().parent / p


def ice_sheet_precip_gt(inputs_dir: Path, gridded_filename: str) -> float:
    """Climatological ice-sheet precipitation (Gt/yr) of the forcing file."""
    with xr.open_dataset(inputs_dir / gridded_filename) as gi:
        ice = gi.rgi_mask.values > 0.5
        p = gi.monthly_precip.values.mean(0)
        dx = float(abs(gi.x[1] - gi.x[0]))
    return float(p[ice].sum()) * RHO_I * dx * dx / 1e12


def measure(inputs_dir: Path, gridded_filename: str, yearly_filename: str, index: xr.DataArray):
    """Regress the ice-sheet precipitation-weighted annual ratio and the
    ice-mean temperature anomaly on the index, over the reanalysis record.
    One pass over the (uncompressed, ~9 GB) yearly file."""
    import netCDF4
    with xr.open_dataset(inputs_dir / gridded_filename) as gi:
        ice = gi.rgi_mask.values > 0.5
        w = gi.monthly_precip.values.astype('float64') * ice
    wsum = w.sum()
    d = netCDF4.Dataset(inputs_dir / yearly_filename)
    for v in ('precip_ratio', 't2m_anom'):
        d[v].set_auto_maskandscale(False)
    ps, ts = float(d['precip_ratio'].scale_factor), float(d['t2m_anom'].scale_factor)
    years = [int(y) for y in d['year'][:]]
    P, A, J = [], [], []
    for k, y in enumerate(years):
        tot = ann = jja = 0.0
        for m in range(12):
            tot += float((d['precip_ratio'][k, m].astype('float64') * ps * w[m]).sum())
            v = float(d['t2m_anom'][k, m][ice].mean()) * ts
            ann += v / 12.0
            if m in (5, 6, 7):
                jja += v / 3.0
        P.append(tot / wsum); A.append(ann); J.append(jja)
        print(f'  {y}', end='', flush=True)
    d.close(); print()
    P, A, J = np.array(P), np.array(A), np.array(J)
    I = np.array([float(index.sel(time=y)) for y in years])
    lnP = np.log(P)
    print(f'  ice-sheet precip-weighted annual ratio {years[0]}-{years[-1]}: '
          f'mean {P.mean():.4f}, std {P.std():.4f}')
    for name, x in (('index', I), ('ice annual T', A), ('ice JJA T', J)):
        b, a = np.polyfit(x, lnP, 1)
        se = np.std(lnP - (a + b * x), ddof=2) / (np.std(x) * np.sqrt(len(x) - 1))
        print(f'  dlnP/d({name:12s}) = {b * 100:+6.2f} +- {se * 100:.2f} %/K, '
              f'r {np.corrcoef(x, lnP)[0, 1]:+.2f}'
              + ('' if abs(b) > 2 * se else '   [not significant]'))
    s_ann = float(np.polyfit(I, A, 1)[0])
    print(f'  ice annual T per index {s_ann:.2f} (r {np.corrcoef(I, A)[0, 1]:.2f}); '
          f'ice JJA T per index {float(np.polyfit(I, J, 1)[0]):.2f} '
          f'(r {np.corrcoef(I, J)[0, 1]:.2f})')
    return s_ann


def _index_ratio(index: xr.DataArray, gamma_annual, dt_ann_per_index):
    """exp(gamma * I), un-normalized, on the index's own time axis."""
    return np.exp(gamma_annual * dt_ann_per_index * index.values)


def _pick_base_year(index: xr.DataArray, clim_window, base_year=None):
    """The library references the series to ONE year, so the base year is
    chosen inside the climatology window at the index value closest to zero:
    stored(base) is then 1 while the reference stays the climatology rather
    than one year's weather -- the same reason base_anomaly_year is None."""
    years = index.time.values.astype(int)
    if base_year is None:
        win = (years >= clim_window[0]) & (years <= clim_window[1])
        base_year = int(years[win][np.argmin(np.abs(index.values[win]))])
    return base_year, int(np.where(years == base_year)[0][0])


def build_precip_anomaly(domain_path: str, smoothing_sigma: float = DEFAULT_SMOOTHING_SIGMA,
                         source_path: str = None, gamma_annual: float = GAMMA_ANNUAL,
                         dt_ann_per_index: float = DT_ANN_PER_INDEX, base_year: int = None,
                         gridded_filename: str = 'GLIDE_inputs.nc',
                         yearly_filename: str = None, do_measure: bool = False,
                         write: bool = True, verbose: bool = True):
    """Write model_inputs/precip_anomaly.nc. Returns (DataArray, info)."""
    domain_path = Path(domain_path)
    inputs = domain_path / 'model_inputs'
    anom_path = inputs / 'temperature_anomaly.nc'
    if not anom_path.exists():
        print(f'{anom_path} not found; run make_temperature_anomaly.py first')
        return None, {}
    anom = xr.open_dataset(anom_path)
    index = anom.temp_anomaly
    years = index.time.values.astype(int)
    cw = str(anom.attrs.get('clim_window', '1986-2025')).split('-')
    clim_window = (int(cw[0]), int(cw[-1]))

    if do_measure:
        if yearly_filename is None:
            raise SystemExit('--measure needs --yearly-filename')
        dt_ann_per_index = measure(inputs, gridded_filename, yearly_filename, index)

    R = _index_ratio(index, gamma_annual, dt_ann_per_index)
    src = _resolve(source_path) if source_path else _resolve(ACCUMULATION_PATH)
    kind = 'index'
    n_filled = 0
    if src.exists():
        kind = 'reconstruction'
        df = pd.read_csv(src, comment='#')
        df.columns = [c.strip().lower() for c in df.columns]
        df = df[['year', 'accum']].dropna().sort_values('year')
        ry = df.year.to_numpy().astype(int)
        accum = np.interp(np.arange(ry.min(), ry.max() + 1), ry, df.accum.to_numpy().astype('float64'))
        if smoothing_sigma > 0:
            accum = gaussian_filter1d(accum, sigma=smoothing_sigma, mode='nearest')
        rec = pd.Series(accum, index=np.arange(ry.min(), ry.max() + 1)).reindex(years)
        # Outside the reconstruction, the index scaling rescaled to the
        # reconstruction's own mean over its span: a dict lookup below the
        # series would raise, and holding a constant would reintroduce exactly
        # the flat-hold bias the temperature record just shed.
        span = (years >= ry.min()) & (years <= ry.max())
        scale = float(rec.values[span].mean()) / float(R[span].mean())
        vals = np.where(np.isfinite(rec.values), rec.values, R * scale)
        n_filled = int((~np.isfinite(rec.values)).sum())
    else:
        vals = R
        if verbose:
            print(f'{src} not found; scaling the temperature index instead')

    base_year, b = _pick_base_year(index, clim_window, base_year)
    residual = float(vals[b] / (vals[(years >= clim_window[0]) & (years <= clim_window[1])].mean()) - 1.0)
    stored = vals / vals[b]

    da = xr.DataArray(stored.astype('float64'), coords={'time': years}, dims=['time'],
                      name='precip_anomaly', attrs={
        'units': '1',
        'description': ('precipitation multiplier relative to the climatology, '
                        f'normalized to 1 at {base_year}; consumed by '
                        'forward.simulate with alpha_precip and applied to '
                        'INDEX years only'),
        'kind': kind,
        'gamma_annual_per_K': float(gamma_annual),
        'dTann_per_index': float(dt_ann_per_index),
        'gamma_per_index_K': float(gamma_annual * dt_ann_per_index),
        'base_precip_year': base_year,
        'base_index_K': float(index.values[b]),
        'base_residual_vs_climatology': residual,
        'clim_window': f'{clim_window[0]}-{clim_window[1]}',
        'source': str(src) if kind == 'reconstruction' else 'temperature_anomaly.nc',
        'smoothing_sigma_years': float(smoothing_sigma) if kind == 'reconstruction' else 0.0,
        'n_years_filled_from_index': n_filled,
        'measured': 'yes' if do_measure else 'no',
        'note': ('alpha_precip scales gamma to first order; the interannual '
                 'reanalysis gives +2.3 +- 1.2 %/K (not significant), so the '
                 'index default is prior-driven, not fitted')})
    if write:
        da.to_netcdf(inputs / 'precip_anomaly.nc')
    info = dict(base_year=base_year, residual=residual, kind=kind,
                gamma_per_index=gamma_annual * dt_ann_per_index,
                dt_ann_per_index=dt_ann_per_index, inputs=inputs,
                gridded_filename=gridded_filename, n_filled=n_filled)
    return da, info


def report(da, info, index):
    try:
        tot = ice_sheet_precip_gt(info['inputs'], info['gridded_filename'])
    except (FileNotFoundError, OSError):
        tot = float('nan')
    g, dt = info['gamma_per_index'], info['dt_ann_per_index']
    print(f'source: {info["kind"]}' + (f' ({info["n_filled"]} years filled from the index)'
                                       if info['n_filled'] else ''))
    print(f'gamma {g / dt * 100:.1f} %/K of ice-sheet annual T x {dt:.2f} K per index K '
          f'= {g * 100:.2f} %/K of index')
    print(f'base year {info["base_year"]} (index {float(index.sel(time=info["base_year"])):+.3f} K, '
          f'{info["residual"] * 100:+.3f} % from the climatology mean); '
          f'climatological ice-sheet precipitation {tot:.0f} Gt/yr')
    print('  epoch: index -> precip ratio -> Gt/yr')
    for lab, y0, y1 in EPOCHS:
        i = float(index.sel(time=slice(y0, y1)).mean()); r = float(da.sel(time=slice(y0, y1)).mean())
        print(f'    {lab:12s} {i:+6.2f} K   x{r:.4f}   {(r - 1) * tot:+6.1f} Gt/yr')
    print(f'  mean over the index years (t_start..{REC_START - 1}) across the whole 0-7 %/K range,')
    print('  as Gt/yr and as ice-sheet-mean thickness accumulated over those years:')
    print(f'    {"t_start":>8s} {"index":>7s} ' + ''.join(f'{s:>20s}' for s in ('0 %/K', '2 %/K', '5 %/K', '7 %/K')))
    for t0 in T_STARTS:
        n = REC_START - t0
        i = float(index.sel(time=slice(t0, REC_START - 1)).mean())
        cells = ''
        for gg in (0.0, 0.02, 0.05, 0.07):
            dgt = (np.exp(gg * dt * i) - 1) * tot
            cells += f'{dgt:+8.1f} Gt/yr{dgt * 1e12 / (RHO_I * ICE_AREA_M2) * n:+7.0f} m'
        print(f'    {t0:8d} {i:+7.2f} {cells}')
    print('  The thickness column is rate x years, an UPPER bound: the ice sheet discharges')
    print('  the excess over its relaxation time (~850 yr in the north, less further south),')
    print('  so a deep start settles nearer rate x tau, about -8 m at 5 %/K.')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--domain-path', type=str, required=True)
    p.add_argument('--smoothing-sigma', type=float, default=DEFAULT_SMOOTHING_SIGMA,
                   help='reconstruction source only (default %(default)s yr)')
    p.add_argument('--source', type=str, default=None,
                   help=f'accumulation CSV; default {ACCUMULATION_PATH}')
    p.add_argument('--gamma', type=float, default=GAMMA_ANNUAL * 100,
                   help='%%/K of ice-sheet ANNUAL temperature (default %(default)s)')
    p.add_argument('--dt-ann-per-index', type=float, default=DT_ANN_PER_INDEX,
                   help='ice-sheet annual K per index K (default %(default)s, measured)')
    p.add_argument('--base-year', type=int, default=None,
                   help='default: the climatology-window year whose index is closest to zero')
    p.add_argument('--gridded-filename', default='GLIDE_inputs.nc',
                   help='forcing file, for the Gt/yr totals and the --measure weights')
    p.add_argument('--yearly-filename', default=None, help='for --measure')
    p.add_argument('--measure', action='store_true',
                   help='recompute dTann_dindex and the interannual precip regression '
                        'from the yearly climate file (one pass over ~9 GB)')
    a = p.parse_args()
    da, info = build_precip_anomaly(
        a.domain_path, smoothing_sigma=a.smoothing_sigma, source_path=a.source,
        gamma_annual=a.gamma / 100.0, dt_ann_per_index=a.dt_ann_per_index,
        base_year=a.base_year, gridded_filename=a.gridded_filename,
        yearly_filename=a.yearly_filename, do_measure=a.measure)
    if da is None:
        return
    index = xr.open_dataset(Path(a.domain_path) / 'model_inputs' / 'temperature_anomaly.nc').temp_anomaly
    report(da, info, index)
    print(f'wrote {info["inputs"] / "precip_anomaly.nc"}  '
          f'(set alpha_precip=1.0 and base_precip_year={info["base_year"]} in the config)')


if __name__ == '__main__':
    main()
