import numpy as np
from scipy.signal import fftconvolve


def detrend_dem(z, dx, dy, mask, method="plane"):
    """
    Remove a simple trend from a DEM before covariance estimation.

    method:
        "none"  : no detrending
        "mean"  : subtract mean
        "plane" : subtract least-squares plane
    """
    z = np.asarray(z, dtype=float)

    if method == "none":
        out = z.copy()
        out[~mask] = np.nan
        return out

    if method == "mean":
        out = z - np.nanmean(z[mask])
        out[~mask] = np.nan
        return out

    if method == "plane":
        ny, nx = z.shape
        jj, ii = np.indices(z.shape)

        x = (ii - nx / 2) * dx
        y = (jj - ny / 2) * dy

        A = np.column_stack([
            np.ones(mask.sum()),
            x[mask],
            y[mask],
        ])

        coeffs, *_ = np.linalg.lstsq(A, z[mask], rcond=None)
        trend = coeffs[0] + coeffs[1] * x + coeffs[2] * y

        out = z - trend
        out[~mask] = np.nan
        return out

    raise ValueError(f"Unknown detrending method: {method}")


def radial_empirical_covariance(
    dem,
    dx,
    dy=None,
    bin_width=None,
    max_dist=None,
    detrend="plane",
    min_pairs=1000,
):
    """
    Estimate isotropic empirical covariance C(r) from a gridded DEM.

    Parameters
    ----------
    dem : 2D array
        DEM elevations. NaNs are treated as missing data.
    dx, dy : float
        Grid spacing in x and y directions, in map units.
    bin_width : float
        Width of radial distance bins. Defaults to max(dx, dy).
    max_dist : float
        Maximum lag distance to return. Defaults to half the smaller domain width.
    detrend : {"none", "mean", "plane"}
        Trend removal before covariance estimation.
    min_pairs : int
        Minimum number of valid DEM pairs required for a lag to contribute.

    Returns
    -------
    r : 1D array
        Distance-bin centers.
    cov : 1D array
        Empirical covariance at each distance bin.
    npairs : 1D array
        Effective number of DEM pairs contributing to each bin.
    """

    if dy is None:
        dy = dx
    if bin_width is None:
        bin_width = max(dx, dy)

    z = np.asarray(dem, dtype=float)
    mask = np.isfinite(z)

    if mask.sum() == 0:
        raise ValueError("DEM contains no finite values.")

    z_resid = detrend_dem(z, dx=dx, dy=dy, mask=mask, method=detrend)

    # Fill missing values with zero for convolution.
    z0 = np.where(mask, z_resid, 0.0)
    m = mask.astype(float)

    # Linear cross-correlation, not circular correlation.
    # Shape is (2*ny - 1, 2*nx - 1), with zero lag at [ny-1, nx-1].
    numerator = fftconvolve(z0, z0[::-1, ::-1], mode="full")
    denominator = fftconvolve(m, m[::-1, ::-1], mode="full")

    ny, nx = z.shape

    lag_y = np.arange(-(ny - 1), ny) * dy
    lag_x = np.arange(-(nx - 1), nx) * dx
    lag_x_grid, lag_y_grid = np.meshgrid(lag_x, lag_y)

    distance = np.hypot(lag_x_grid, lag_y_grid)

    if max_dist is None:
        max_dist = 0.5 * min(nx * dx, ny * dy)

    valid = (
        (distance <= max_dist)
        & (denominator >= min_pairs)
        & np.isfinite(numerator)
    )

    bin_index = np.floor(distance[valid] / bin_width).astype(int)
    n_bins = int(np.floor(max_dist / bin_width)) + 1

    # Radial covariance should be pair-weighted:
    # sum all z(x)z(x+h) in bin / sum all valid pairs in bin.
    num_binned = np.bincount(
        bin_index,
        weights=numerator[valid],
        minlength=n_bins,
    )

    den_binned = np.bincount(
        bin_index,
        weights=denominator[valid],
        minlength=n_bins,
    )

    cov = num_binned / den_binned
    r = (np.arange(n_bins) + 0.5) * bin_width

    bad = den_binned == 0
    cov[bad] = np.nan

    return r, cov, den_binned
