"""Build the precipitation (accumulation) anomaly time series for a domain.

alaska-forecast scales precip by a smoothed Mt. Hunter ice-core accumulation
record. For Greenland the analogue is an ice-sheet-wide accumulation
reconstruction: Box et al. (2013, J. Climate) 1600-2009 net snow accumulation
from 86 cores + RACMO2, or a single long core (NGRIP / NEEM / GISP2 annual
accumulation from NOAA Paleoclimatology). Provide it as a two-column CSV
(`year,accum`, any units — the inverse normalises to `base_precip_year`) at
common_data/climate/precip_anomaly/greenland_accumulation.csv. Missing file ->
no product, the multiplicative precip anomaly is simply not applied.

Output: {domain_path}/model_inputs/precip_anomaly.nc
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter1d

ACCUMULATION_PATH = Path('../common_data/climate/precip_anomaly/greenland_accumulation.csv')
DEFAULT_SMOOTHING_SIGMA = 10.0  # years


def build_precip_anomaly(domain_path: str, smoothing_sigma: float = DEFAULT_SMOOTHING_SIGMA,
                         source_path: str = None):
    domain_path = Path(domain_path)
    output_path = domain_path / 'model_inputs' / 'precip_anomaly.nc'
    source_path = Path(source_path) if source_path else ACCUMULATION_PATH
    if not source_path.exists():
        print(f"{source_path} not found; skipping the precip anomaly (optional)")
        return None
    df = pd.read_csv(source_path, comment='#')
    df.columns = [c.strip().lower() for c in df.columns]
    df = df[['year', 'accum']].dropna().sort_values('year')
    years = df.year.to_numpy().astype(int)
    accum = df.accum.to_numpy().astype('float64')
    full = np.arange(years.min(), years.max() + 1)
    accum = np.interp(full, years, accum)     # fill gaps so smoothing is uniform
    if smoothing_sigma > 0:
        accum = gaussian_filter1d(accum, sigma=smoothing_sigma, mode='nearest')
    da = xr.DataArray(accum.astype('float32'), coords={'time': full}, dims=['time'],
                      name='precip_anomaly',
                      attrs={'units': 'source units (normalised at runtime)',
                             'description': 'Greenland accumulation reconstruction, '
                                            'Gaussian-smoothed in time',
                             'source': str(source_path),
                             'smoothing_sigma_years': float(smoothing_sigma)})
    da.to_netcdf(output_path)
    return da


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-path", type=str, required=True)
    parser.add_argument("--smoothing-sigma", type=float, default=DEFAULT_SMOOTHING_SIGMA)
    parser.add_argument("--source", type=str, default=None)
    args = parser.parse_args()
    build_precip_anomaly(args.domain_path, args.smoothing_sigma, args.source)
