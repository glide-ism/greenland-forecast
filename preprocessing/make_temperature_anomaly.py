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
1784-1839) are linearly interpolated; before 1784 the series holds the
1784-1813 mean back to year 1 so early spin-ups can look it up.

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
BASINS_PATH = '../common_data/area/basins/Greenland_Basins_PS_v1.4.2.shp'

OVERLAP_START = 1850
OVERLAP_END = 1900
JJA = [6, 7, 8]
EARLY_MEAN_WINDOW = (1784, 1813)
EXTEND_BACK_TO = 1


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


def build_vinther_series(carra_base=None, clim_years=None, verbose=True):
    """Returns (annual: pd.Series JJA anomaly by year, monthly: DataFrame,
    info: dict). `clim_years` defaults to the complete CARRA2 calendar
    years (what make_carra_vars pools)."""
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
    full_index = pd.RangeIndex(EXTEND_BACK_TO, int(annual.index.max()) + 1)
    annual = annual.reindex(full_index)
    first = int(v_jja.dropna().index.min())
    n_gap = int(annual.loc[first:].isna().sum())
    annual.loc[first:] = annual.loc[first:].interpolate(limit_direction='both')
    early = float(annual.loc[EARLY_MEAN_WINDOW[0]:EARLY_MEAN_WINDOW[1]].mean())
    annual.loc[:first - 1] = early
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
                n_gap_years_interpolated=n_gap, early_constant=early - ref, n_sw_cells=n_cells)
    if verbose:
        print(f"Vinther JJA vs CARRA2 SW-basin 100 m JJA over {info['overlap']} (n={len(ov)}): "
              f"slope {slope:.2f}, r {r:.2f}; extended {info['extension_years']}; "
              f"reference {ref:.2f} C over {info['clim_window']}; {n_gap} gap years interpolated; "
              f"pre-{first} constant {early - ref:+.2f} C")
    return annual, monthly, info


def build_temperature_anomaly(domain_path: str, source: str = None, carra_base=None) -> xr.DataArray:
    """Build the anomaly series for `domain_path` and write to disk."""
    domain_path = Path(domain_path)
    output_path = domain_path / 'model_inputs' / 'temperature_anomaly.nc'
    if source is None:
        source = 'vinther' if _resolve(VINTHER_PATH).exists() else 'global'
    if source == 'vinther':
        annual, monthly, info = build_vinther_series(carra_base)
        ds = xr.Dataset(coords={'time': annual.index.values.astype(int), 'month': np.arange(1, 13)})
        ds['temp_anomaly'] = xr.DataArray(annual.values, dims=['time'], attrs={
            'units': 'degC',
            'description': 'SW Greenland JJA temperature anomaly (Vinther et al. 2006 station composite, '
                           'CARRA2 SW-basin 100 m extension), mean zero over the CARRA2 climatology window'})
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
    parser.add_argument("--source", choices=["vinther", "global"], default=None,
                        help="default: vinther when swgreenlandave.dat exists, else global")
    args = parser.parse_args()
    build_temperature_anomaly(args.domain_path, args.source)
