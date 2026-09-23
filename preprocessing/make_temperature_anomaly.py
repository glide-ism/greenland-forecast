"""Build the temperature anomaly time series for a domain.

Two sources:

`vinther` (default when the file exists): the Vinther et al. (2006) SW
Greenland monthly station composite (`swgreenlandave.dat`, 1784-2013,
tenths of degC, -999 missing). The annual series is the JJA mean anomaly
(the melt season is what the SMB responds to; winter anomalies are ~2x
larger and would dominate an annual mean applied uniformly to every month).
It is extended past 2013 with the CARRA2 100 m temperature averaged over
the SW drainage basin (JJA, regressed onto Vinther over their overlap) and
referenced so that its mean over the CARRA2 climatology window (the years
`make_carra_vars.py` pools, 1986-2025 with the full record) is zero — so it
is applied with `base_anomaly_year=None` and an `alpha_t2m` that scales the
coastal anomaly onto the ice sheet (~0.5, see
analysis/arctic_amplification.py --regressor vinther). Gaps (mostly
1784-1839) are linearly interpolated.

Before the station record the series is the GISP2 Summit argon-nitrogen
temperature (Kobashi et al. 2011, 2000 BCE - 1993 CE) scaled onto the
station series by a variance-matched slope and crossfaded onto it over
1784-1850 (`--deep-source`, on by default when the file is present; see
`gisp2_deep_series`). `--deep-source none` restores the previous behaviour,
which held the 1784-1813 mean flat back to year 1 -- that is, asserted that
the whole Common Era sat at the Little Ice Age minimum, too cold by about
1.6 K, which over a multi-millennial spin-up is a large and one-signed
surface-mass-balance bias.

`global`: the alaska-forecast splice of the PAGES2k paleo reconstruction
with the HadCRUT instrumental record (PAGES2k bias-corrected over 1850-1900
and cross-faded), a global-mean anomaly to be scaled by an Arctic
amplification factor.

Output: {domain_path}/model_inputs/temperature_anomaly.nc with
`temp_anomaly(time)`; the vinther file also carries
`temp_anomaly_monthly(time, month)` (station anomaly per calendar month,
same reference) for later seasonal use.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


HADCRUT_PATH = '../common_data/climate/temp_anomaly/HadCRUT.5.0.2.0.analysis.summary_series.global.annual.csv'
PAGES_PATH = '../common_data/climate/temp_anomaly/pages2k_ngeo19_recons.nc'
VINTHER_PATH = '../common_data/climate/temp_anomaly/swgreenlandave.dat'
GISP2_PATH = '../common_data/climate/temp_anomaly/gisp2-temperature2011.txt'
BASINS_PATH = '../common_data/area/basins/Greenland_Basins_PS_v1.4.2.shp'

OVERLAP_START = 1850
OVERLAP_END = 1900
JJA = [6, 7, 8]
EARLY_MEAN_WINDOW = (1784, 1813)
EXTEND_BACK_TO = 1

# --- deep (pre-instrumental) extension -----------------------------------
# GISP2 carries a 1-sigma band only through 1950 and is held flat after 1978,
# so the calibration overlap stops at 1950.
DEEP_OVERLAP = (1784, 1950)
DEEP_SMOOTH = 31          # yr; the band the two records share
# Crossfade the core onto the stations across exactly the stretch where the
# station record is mostly interpolated: 9/16 of 1784-99, 8/20 of 1800-19 and
# 19/20 of 1820-39 are missing, and from 1840 it is continuous. So the blend
# does not discard observations, it replaces a 19-year linear interpolation
# with a proxy that has information there.
DEEP_BLEND = (1784, 1840)
# The Kobashi reconstruction integrates firn temperature gradients forward in
# time, and the published file carries a 2.75 K step at 1730 BCE (plus 1.08 K
# the year after) with everything older sitting ~3 K colder: a section
# boundary in that integration, not climate, since the record's own annual
# increments are 0.085 K. Everything before it is dropped. -1000 leaves a
# clear margin and is still 1000 yr deeper than any plausible t_start.
DEEP_RECORD_START = -1000
DEEP_MAX_STEP = 0.5       # K/yr; flag anything this abrupt inside the used range
# Reported so the splice can be checked against what is known about these
# intervals rather than trusted because the pipeline ran.
EPOCHS = (('common_era', 1, 2000), ('roman', 1, 500), ('medieval', 900, 1200),
          ('lia', 1450, 1850), ('lia_core', 1600, 1800))


def _resolve(path):
    p = Path(path)
    return p if p.is_absolute() else Path(__file__).resolve().parent / p


def read_vinther(path=VINTHER_PATH) -> pd.DataFrame:
    """Monthly SW Greenland temperature (degC), index year, columns 1..12."""
    raw = pd.read_csv(_resolve(path), sep=r'\s+', skiprows=1, header=None, index_col=0)
    raw = raw.replace(-999, np.nan) / 10.0
    raw.columns = range(1, 13)
    raw.index = raw.index.astype(int)
    return raw


def read_gisp2(path=GISP2_PATH) -> pd.Series:
    """GISP2 Summit annual surface-snow temperature (degC) indexed by year AD,
    negative years BCE: Kobashi et al. (2011), argon and nitrogen isotopes of
    trapped air, 2000 BCE - 1993 CE. The file is a WDC text report with a
    free-form header, so a line counts as data when its first two fields parse
    as (int, float); the year axis is checked to be contiguous afterwards."""
    rows = []
    for line in _resolve(path).read_text(errors='replace').splitlines():
        f = line.split()
        if len(f) >= 2:
            try:
                rows.append((int(f[0]), float(f[1])))
            except ValueError:
                pass
    s = (pd.DataFrame(rows, columns=['year', 'T'])
         .drop_duplicates('year').set_index('year')['T'].sort_index())
    step = s.index.to_series().diff().dropna()
    if len(s) < 1000 or (step != 1).any():
        raise ValueError(f'{path}: parsed {len(s)} rows with a non-contiguous '
                         'year axis; the file layout is not the expected one')
    return s


def gisp2_deep_series(station_jja: pd.Series, gisp2: pd.Series = None,
                      overlap=DEEP_OVERLAP, smooth=DEEP_SMOOTH,
                      used_to=DEEP_BLEND[1]):
    """Station-equivalent SW Greenland JJA temperature (degC, the raw station
    scale) for the years before the instrumental record, from GISP2.

    GISP2 is a Summit ANNUAL-mean snow temperature at 3200 m and the index is a
    SW-coast JJA station series, so the two differ in site, season and
    elevation. They are tied by one variance-matched slope over their overlap,

        T_stn(t) = <T_stn>_ov + s (T_gisp2(t) - <T_gisp2>_ov),
        s = std(T_stn) / std(T_gisp2),   both low-passed at `smooth` years

    Variance matching rather than least squares: OLS attenuates a
    reconstruction by r, which would return a Little Ice Age that is too warm,
    and what the model consumes is the AMPLITUDE of the forcing, not a
    conditional mean. Both sides are low-passed first because the gas-isotope
    thermometer integrates the firn temperature gradient and so resolves only
    multi-decadal variability: annual increments are 0.09 K against a record
    std of 0.98 K, and the station series keeps only 0.64 of its annual std at
    31 years, so comparing the two unfiltered would inflate s by ~1/0.64.

    The slope runs 1.16 (11 yr, r 0.58), 1.12 (31, r 0.76), 1.06 (51, r 0.90),
    0.95 (101, r 0.97). The decline as the filter loosens is the residual
    firn attenuation showing itself, so the least-attenuated estimates are the
    long ones -- but at 101 years only 1.7 independent samples remain in the
    167-year overlap, so they are also the noisiest. 31 years is the
    compromise, with ~5 samples and r 0.76; `deep_slope_by_smooth` in the
    attrs records the whole curve so the choice stays visible, and anything
    leaving fewer than three independent samples is refused outright.

    Little rests on that choice. The epoch means move linearly with s about
    the anchor, the anchor is a directly measured station mean, and the
    Common Era sits only ~0.6 K from it in the core, so the whole s range
    1.06-1.16 moves the Common Era index from -0.54 to -0.46 K and the Little
    Ice Age core from -1.75 to -1.79. Against a flat hold of -1.20 K for
    every pre-instrumental year, that spread is irrelevant.
    """
    g = read_gisp2() if gisp2 is None else gisp2
    g = g.loc[DEEP_RECORD_START:]
    # Only the years the blend actually consumes are checked; past it the core
    # carries zero weight, and its two >0.5 K steps at 1920-21 are the abrupt
    # early-20th-century Greenland warming, which is climate and is in any
    # case represented by the stations there.
    step = g.loc[:used_to].diff().abs()
    bad = step[step > DEEP_MAX_STEP]
    if len(bad):
        print(f'  WARNING: {len(bad)} year-to-year steps above {DEEP_MAX_STEP} K remain in the '
              f'core record over {DEEP_RECORD_START}-{used_to}, where it is used: ' +
              ', '.join(f'{int(y)} {float(v):+.2f} K' for y, v in bad.items()))
    o = slice(*overlap)
    kw = dict(center=True, min_periods=max(smooth // 2, 1))
    gs = g.rolling(smooth, **kw).mean().loc[o]
    vs = station_jja.rolling(smooth, **kw).mean().loc[o]
    m = gs.notna() & vs.notna()
    if int(m.sum()) < 3 * smooth:
        raise ValueError(f'GISP2/station overlap {overlap} holds only '
                         f'{int(m.sum())} usable years at smooth={smooth}')
    slope = float(vs[m].std() / gs[m].std())
    r = float(np.corrcoef(gs[m], vs[m])[0, 1])
    # Both anchors over exactly the same years: the station series begins at
    # 1785, not 1784 (that year holds only four months and skipna=False nulls
    # it), so slicing each record independently would offset the two means by
    # one year's weather.
    yrs = station_jja.loc[o].dropna().index.intersection(g.loc[o].index)
    g_ov, v_ov = float(g.loc[yrs].mean()), float(station_jja.loc[yrs].mean())
    deep = v_ov + slope * (g.loc[:overlap[1]] - g_ov)
    info = dict(deep_source='gisp2_kobashi2011', deep_slope=slope,
                deep_r=r, deep_overlap=f'{overlap[0]}-{overlap[1]}',
                deep_smooth_yr=int(smooth), deep_n_eff=float(m.sum()) / smooth,
                deep_record=f'{int(g.index.min())}-{int(g.index.max())}',
                deep_slope_by_smooth='; '.join(
                    f'{w}yr {float(vs_.std() / gs_.std()):.2f} (r {float(np.corrcoef(gs_[k], vs_[k])[0, 1]):.2f})'
                    for w in (11, 31, 51, 101)
                    for gs_, vs_ in [(g.rolling(w, center=True, min_periods=max(w // 2, 1)).mean().loc[o],
                                      station_jja.rolling(w, center=True, min_periods=max(w // 2, 1)).mean().loc[o])]
                    for k in [gs_.notna() & vs_.notna()]))
    return deep, info


def carra_sw_series(carra_base=None):
    """Monthly CARRA2 100 m air temperature (degC) averaged over the SW
    drainage basin, on the native grid: DataFrame index year, columns 1..12."""
    import geopandas as gpd
    import shapely
    import make_carra_vars as mcv
    carra, ds_t, ds_p, orog, cx, cy, tdim = mcv.open_carra(carra_base)
    levels = ds_t['heightAboveGround'].values.astype(float)
    i_lo = int(np.argmin(np.abs(levels - mcv.LEVEL_LOW)))
    g = gpd.read_file(_resolve(BASINS_PATH)).to_crs('EPSG:3413')
    sw = g[g['SUBREGION1'] == 'SW'].dissolve().geometry.iloc[0]
    X, Y = np.meshgrid(cx, cy)
    # transform the CARRA cell centres to EPSG:3413 and test containment
    import pyproj
    tr = pyproj.Transformer.from_proj(carra.proj, 'EPSG:3413', always_xy=True)
    ex, ey = tr.transform(X, Y)
    inside = shapely.contains_xy(sw, ex, ey)
    ys, xs = np.where(inside)
    if inside.sum() == 0:
        raise RuntimeError('no CARRA2 cells inside the SW basin')
    da = ds_t['t'].isel(heightAboveGround=i_lo)
    n = da.sizes[tdim]
    series = np.empty(n)
    for i0 in range(0, n, mcv.READ_BLOCK):
        chunk = da.isel({tdim: slice(i0, i0 + mcv.READ_BLOCK)}).values
        series[i0:i0 + chunk.shape[0]] = np.nanmean(chunk[:, ys, xs], axis=1)
    t = pd.DatetimeIndex(da[tdim].values)
    df = pd.DataFrame({'T': series - 273.15, 'year': t.year, 'month': t.month})
    return df.pivot(index='year', columns='month', values='T'), int(inside.sum())


def build_vinther_series(carra_base=None, clim_years=None, verbose=True,
                         deep_source=None, deep_smooth=DEEP_SMOOTH,
                         deep_overlap=DEEP_OVERLAP, deep_blend=DEEP_BLEND):
    """Returns (annual: pd.Series JJA anomaly by year, monthly: DataFrame,
    info: dict). `clim_years` defaults to the complete CARRA2 calendar
    years (what make_carra_vars pools).

    `deep_source` governs the years before the station record: 'gisp2' scales
    the Kobashi et al. (2011) Summit record onto the station series and
    crossfades onto it over `deep_blend`, None holds the 1784-1813 mean flat
    as before. The default is 'gisp2' when the file is present, because the
    flat hold asserts that the whole Common Era sat at the Little Ice Age
    minimum and is wrong by about 1.6 K."""
    v = read_vinther()
    sw, n_cells = carra_sw_series(carra_base)
    if clim_years is None:
        clim_years = [y for y in sw.index if sw.loc[y].notna().all()]
    clim_years = sorted(int(y) for y in clim_years)
    v_jja = v[JJA].mean(axis=1, skipna=False)
    c_jja = sw[JJA].mean(axis=1, skipna=False)
    ov = sorted(set(v_jja.dropna().index) & set(c_jja.dropna().index))
    x, y = c_jja.loc[ov].values, v_jja.loc[ov].values
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    # station-equivalent JJA for the CARRA years beyond the station record
    ext_years = [y for y in c_jja.dropna().index if y > v_jja.dropna().index.max()]
    ext = pd.Series(slope * c_jja.loc[ext_years].values + intercept, index=ext_years)
    annual = pd.concat([v_jja.dropna(), ext]).sort_index()
    annual = annual[~annual.index.duplicated()]
    # None = auto (use the core when the file is there), 'none' = the old
    # flat hold, 'gisp2' = force and fail loudly if the file is missing.
    if deep_source is None:
        deep_source = 'gisp2' if _resolve(GISP2_PATH).exists() else 'none'
    if deep_source not in ('gisp2', 'none'):
        raise ValueError(f'unknown deep_source {deep_source!r}')
    gisp2 = read_gisp2() if deep_source == 'gisp2' else None
    # Run the axis back past any plausible t_start: forward.simulate looks the
    # anomaly up by year in a dict and clips only at the top, so a step before
    # the first year would raise rather than saturate.
    back_to = min(EXTEND_BACK_TO, DEEP_RECORD_START) if gisp2 is not None else EXTEND_BACK_TO
    full_index = pd.RangeIndex(back_to, int(annual.index.max()) + 1)
    annual = annual.reindex(full_index)
    first = int(v_jja.dropna().index.min())
    n_gap = int(annual.loc[first:].isna().sum())
    annual.loc[first:] = annual.loc[first:].interpolate(limit_direction='both')
    early = float(annual.loc[EARLY_MEAN_WINDOW[0]:EARLY_MEAN_WINDOW[1]].mean())
    deep_info = {}
    if gisp2 is None:
        annual.loc[:first - 1] = early
    else:
        deep, deep_info = gisp2_deep_series(annual.loc[first:], gisp2,
                                            overlap=deep_overlap, smooth=deep_smooth)
        # Crossfade core -> stations across `deep_blend` so the join carries
        # neither a step nor a kink; the core alone before it, the stations
        # alone after. The station half is NaN below `first`, so it is zeroed
        # before the weighting rather than multiplied by w = 0.
        b0, b1 = deep_blend
        w = np.clip((annual.index.values - b0) / float(b1 - b0), 0.0, 1.0)
        d = deep.reindex(annual.index).ffill().bfill().values
        a = np.nan_to_num(annual.values.astype(float))
        annual = pd.Series(w * a + (1.0 - w) * d, index=annual.index)
        deep_info['deep_blend'] = f'{b0}-{b1}'
    ref = float(annual.loc[clim_years].mean())
    annual = annual - ref
    # monthly station anomaly with the same reference convention (per month,
    # over the station years inside the climatology window)
    monthly = v.copy()
    win = [y for y in clim_years if y in monthly.index]
    monthly = monthly - monthly.loc[win].mean()
    info = dict(slope=float(slope), intercept=float(intercept), r_overlap=r, n_overlap=len(ov),
                overlap=f'{ov[0]}-{ov[-1]}', extension_years=f'{ext_years[0]}-{ext_years[-1]}' if ext_years else '',
                reference_jja_degC=ref, clim_window=f'{clim_years[0]}-{clim_years[-1]}',
                n_gap_years_interpolated=n_gap, early_constant=early - ref, n_sw_cells=n_cells,
                series_start=int(annual.index.min()), **deep_info)
    if deep_info:
        info['flat_hold_would_be'] = early - ref
        for lab, y0, y1 in EPOCHS:
            info[f'epoch_{lab}'] = float(annual.loc[y0:y1].mean())
    if verbose:
        print(f"Vinther JJA vs CARRA2 SW-basin 100 m JJA over {info['overlap']} (n={len(ov)}): "
              f"slope {slope:.2f}, r {r:.2f}; extended {info['extension_years']}; "
              f"reference {ref:.2f} C over {info['clim_window']}; {n_gap} gap years interpolated")
        if not deep_info:
            print(f"  pre-{first} constant {early - ref:+.2f} C (no deep source)")
        else:
            print(f"  deep extension: GISP2 (Kobashi et al. 2011) {info['deep_record']}, "
                  f"slope {info['deep_slope']:.2f} K/K (r {info['deep_r']:.2f}, "
                  f"{info['deep_n_eff']:.1f} independent {deep_smooth}-yr samples over "
                  f"{info['deep_overlap']}), blended onto the stations over {info['deep_blend']}")
            print(f"  slope vs bandwidth: {info['deep_slope_by_smooth']}")
            print('  epoch means of the index (K vs the CARRA2 window): '
                  + ', '.join(f"{lab} {info[f'epoch_{lab}']:+.2f}" for lab, _, _ in EPOCHS))
            print(f"  the old flat hold would have been {early - ref:+.2f} K for every year "
                  f"before {first}: {info['epoch_common_era'] - (early - ref):+.2f} K too cold")
    return annual, monthly, info


def build_temperature_anomaly(domain_path: str, source: str = None, carra_base=None,
                              deep_source: str = None, deep_smooth: int = DEEP_SMOOTH,
                              out_name: str = None) -> xr.DataArray:
    """Build the anomaly series for `domain_path` and write to disk.

    `out_name` writes beside the default instead of over it, so a variant (a
    flat pre-instrumental hold, say) can be built for `config.anomaly_filename`
    to point at without disturbing the series the next inversion will use."""
    domain_path = Path(domain_path)
    output_path = domain_path / 'model_inputs' / (out_name or 'temperature_anomaly.nc')
    if source is None:
        source = 'vinther' if _resolve(VINTHER_PATH).exists() else 'global'
    if source == 'vinther':
        annual, monthly, info = build_vinther_series(
            carra_base, deep_source=deep_source, deep_smooth=deep_smooth)
        deep = info.get('deep_source')
        ds = xr.Dataset(coords={'time': annual.index.values.astype(int), 'month': np.arange(1, 13)})
        ds['temp_anomaly'] = xr.DataArray(annual.values, dims=['time'], attrs={
            'units': 'degC',
            'description': 'SW Greenland JJA temperature anomaly (Vinther et al. 2006 station composite, '
                           'CARRA2 SW-basin 100 m extension), mean zero over the CARRA2 climatology window'
                           + (f'; before {DEEP_BLEND[1]} the GISP2 Summit Ar-N temperature '
                              '(Kobashi et al. 2011) scaled onto the station series' if deep else
                              '; before the station record the 1784-1813 mean held constant')})
        ds['temp_anomaly_monthly'] = xr.DataArray(
            monthly.reindex(ds.time.values).values, dims=['time', 'month'],
            attrs={'units': 'degC', 'description': 'station anomaly per calendar month, same reference window'})
        ds.attrs.update(source='vinther_sw_greenland', season='JJA', **{k: (v if not isinstance(v, bool) else int(v))
                                                                     for k, v in info.items()})
        ds.to_netcdf(output_path)
        return ds['temp_anomaly']

    hadcrut = pd.read_csv(_resolve(HADCRUT_PATH))
    pages = xr.load_dataset(_resolve(PAGES_PATH))

    pages_years = pages.year.values
    hadcrut_years = hadcrut.Time.values
    pages_anomaly = pages.DA.mean(axis=1)
    hadcrut_anomaly = hadcrut['Anomaly (deg C)'].values

    years = np.arange(
        min(pages_years.min(), hadcrut_years.min()),
        max(pages_years.max(), hadcrut_years.max()) + 1,
    )
    pages_interp = np.interp(years, pages_years, pages_anomaly)
    hadcrut_interp = np.interp(years, hadcrut_years, hadcrut_anomaly)

    # Bias-correct PAGES2k to HadCRUT over their overlap window
    overlap = (years >= OVERLAP_START) & (years <= OVERLAP_END)
    bias = np.mean(hadcrut_interp[overlap] - pages_interp[overlap])
    pages_aligned = pages_interp + bias

    # Linear cross-fade across the overlap so the join is continuous
    weight = np.clip((years - OVERLAP_START) / (OVERLAP_END - OVERLAP_START), 0.0, 1.0)
    spliced = np.where(
        years < OVERLAP_START,
        pages_aligned,
        np.where(
            years > OVERLAP_END,
            hadcrut_interp,
            (1 - weight) * pages_aligned + weight * hadcrut_interp,
        ),
    )

    anomaly = xr.DataArray(
        spliced,
        coords={"time": years},
        dims=["time"],
        name="temp_anomaly",
        attrs={
            "units": "degC",
            "description": "Global mean temperature anomaly (May-Apr year, PAGES2k + HadCRUT splice)",
        },
    )
    anomaly.to_netcdf(output_path)
    return anomaly


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-path", type=str, required=True)
    parser.add_argument("--out-name", type=str, default=None,
                        help="write to model_inputs/<name> instead of temperature_anomaly.nc, "
                             "for building a variant config.anomaly_filename can point at")
    parser.add_argument("--deep-source", choices=["gisp2", "none"], default=None,
                        help="pre-instrumental years (vinther source only): default gisp2 when "
                             "the Kobashi et al. (2011) file is present, 'none' holds the "
                             "1784-1813 mean flat as before")
    parser.add_argument("--deep-smooth", type=int, default=DEEP_SMOOTH,
                        help=f"yr; bandwidth at which the core and the stations are variance "
                             f"matched (default {DEEP_SMOOTH}; the slope is stable over 11-101)")
    parser.add_argument("--source", choices=["vinther", "global"], default=None,
                        help="default: vinther when swgreenlandave.dat exists, else global")
    args = parser.parse_args()
    build_temperature_anomaly(args.domain_path, args.source,
                              deep_source=args.deep_source, deep_smooth=args.deep_smooth,
                              out_name=args.out_name)
