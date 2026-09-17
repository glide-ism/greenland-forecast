"""
Ocean thermal forcing -> calving margins.

glide's height-above-buoyancy calving criterion is the hybrid threshold

    ice calves where  H - H_f  <  q(x) H + h0(x)          (calving flag psi = 0)

with q a fraction of the thickness and h0 an absolute margin in metres.
`OceanForcing` drives both margins with the ISMIP7 ocean thermal forcing
(TF, degC above the in-situ freezing point; `model_inputs/thermal_forcing.nc`
from preprocessing/make_thermal_forcing.py: annual mean and annual max of
the monthly product per calendar year, `tf_dist` = distance to the native
product):

    q (x, t) = q0  + clim_q * (TF_clim(x) - tf_crit) + alpha_q * dTF(x, t)   [1/K]
    h0(x, t) = h00 + clim_h * (TF_clim(x) - tf_crit) + alpha_h * dTF(x, t)   [m/K]
    TF_clim(x) = mean of the annual statistic over `ref_years`
    dTF(x, t)  = TF_step(x, t) - TF_clim(x)
    tf_crit    = the critical thermal forcing: under the monotone calving law
                 the sign of the margin at flotation decides whether a tongue
                 is admissible, so fjords colder than tf_crit float, warmer
                 ones calve above flotation

TF_step aggregates the annual statistic (`statistic`: "max" -> annual max,
"mean" -> annual mean) over the calendar years a step (t0, t1] overlaps:
the maximum for "max", the overlap-weighted mean for "mean". The clim
coefficients give the baseline margins a TF-dependent geography (warm
fjords closer to instability, cold fjords negative margins = persisting
tongues); the alphas set the response to warming. clim = 0 recovers the
pure anomaly model, clim = alpha the pure absolute one (which over-levers:
fronts on either side of the zero crossing end up insensitively cold or
contracted). Steps before the record see dTF = 0; after the record the
last year holds. Both terms are zero (margins at their baselines) where
the product is undefined or farther than `max_dist_km` from its native
cells.

The two knobs act differently across glacier sizes: alpha_q shifts the
threshold by a fraction of the local thickness (deep, thick fronts respond
more per K), alpha_h by the same number of metres everywhere (an undercut
length). Both are plain scalars of the config (a sweep, not autodiff:
psi is a switch of width ~1/sigmoid_c in flotation excess, so dJ/dalpha is
supported only on cells inside that band).

Consumers: forward.simulate (per step, before GlideStep, which checkpoints
q/h0 with the step) and forward_standalone.py's ocean_forcing hook.
"""
import math
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

from .config import OceanForcingConfig


def year_overlap_weights(t0: float, t1: float, eps: float = 1e-9) -> list:
    """Fractional overlap of (t0, t1] with each calendar year, weights sum to 1."""
    out, y = [], math.floor(t0 + eps)
    while y < t1 - eps:
        w = min(t1, y + 1.0) - max(t0, y)
        if w > eps:
            out.append((int(y), w))
        y += 1.0
    total = sum(w for _, w in out)
    return [(y, w / total) for y, w in out]


def _crop_to_factor(ds: xr.Dataset, factor: int) -> xr.Dataset:
    """The centred crop GlacierProblem applies so ny, nx divide 2^n_levels."""
    ny0, nx0 = ds.sizes["y"], ds.sizes["x"]
    ny, nx = (ny0 // factor) * factor, (nx0 // factor) * factor
    y0, x0 = (ny0 - ny) // 2, (nx0 - nx) // 2
    return ds.isel(y=slice(y0, y0 + ny), x=slice(x0, x0 + nx))


class _LazyYears:
    """The (nt, ny, nx) annual statistic read from an open NetCDF on demand
    (a 451-year projection record is 8.7 GB dense). Supports the two
    accesses OceanForcing makes: `stat[bool_mask]` and `stat[int_list]`,
    both returning a (k, ny, nx) float32 array, and `.shape`."""

    def __init__(self, da: xr.DataArray):
        self.da = da
        self.shape = tuple(da.shape)

    def __getitem__(self, idx):
        idx = np.asarray(idx)
        if idx.ndim == 0:                       # one year -> (ny, nx), like a dense array
            return self.da.isel(time=int(idx)).values.astype(np.float32)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        return self.da.isel(time=idx).values.astype(np.float32)


class OceanForcing:
    def __init__(self, cfg: OceanForcingConfig, *, years: np.ndarray, stat, dist: np.ndarray,
                 q0: float, h00: float, source: str = ""):
        if cfg.statistic not in ("max", "mean"):
            raise ValueError(f"OceanForcingConfig.statistic must be 'max' or 'mean', got {cfg.statistic!r}")
        self.cfg = cfg
        self.years = np.asarray(years, dtype=int)
        # (nt, ny, nx): a dense float32 array or a _LazyYears reader
        self.stat = stat if isinstance(stat, _LazyYears) else np.asarray(stat, dtype=np.float32)
        self.q0, self.h00 = float(q0), float(h00)
        self.source = source
        y0, y1 = cfg.ref_years
        sel = (self.years >= y0) & (self.years <= y1)
        if not sel.any():
            raise ValueError(f"ocean_forcing.ref_years {cfg.ref_years} outside the "
                             f"record {self.years[0]}-{self.years[-1]}")
        clim = self.stat[sel].mean(axis=0)
        # active where the product is defined (its footprint is constant over the
        # record; the dense case checks every year, the lazy one the climatology)
        finite = np.isfinite(clim) & np.isfinite(dist)
        if isinstance(self.stat, np.ndarray):
            finite &= np.isfinite(self.stat).all(axis=0)
        self.ok = finite & (dist <= cfg.max_dist_km)
        self.clim = np.where(self.ok, clim, 0.0).astype(np.float32)
        self._zero = np.zeros(self.stat.shape[1:], np.float32)

    @classmethod
    def from_file(cls, path, crop_factor: int, cfg: OceanForcingConfig, *,
                  q0: float, h00: float, lazy: bool = False) -> "OceanForcing":
        """Load only the needed annual statistic from thermal_forcing.nc,
        cropped like GLIDE_inputs. Dense (the file is closed before
        returning) or, with `lazy`, read year by year from the file kept
        open (long projection records)."""
        path = Path(path)
        var = "tf_max" if cfg.statistic == "max" else "tf_mean"
        f = xr.open_dataset(path)
        try:
            ds = _crop_to_factor(f, crop_factor)
            years = ds.time.values.astype(int)
            dist = ds.tf_dist.values.astype(np.float32)
            source = str(ds.attrs.get("thermal_forcing_source", path.name))
            stat = _LazyYears(ds[var]) if lazy else ds[var].values.astype(np.float32)
        finally:
            if not lazy:
                f.close()
        of = cls(cfg, years=years, stat=stat, dist=dist, q0=q0, h00=h00, source=source)
        of._file = f if lazy else None
        return of

    @property
    def ny_nx(self):
        return self.stat.shape[1:]

    def describe(self) -> str:
        c = self.cfg
        return (f"ocean forcing: TF {self.years[0]}-{self.years[-1]} (tf_{c.statistic}), "
                f"climatology {c.ref_years}, tf_crit {c.tf_crit:g} degC, {int(self.ok.sum())} active cells within {c.max_dist_km:g} km; "
                f"q = {self.q0:g} + {c.clim_q:g}/K * (TF_clim - tf_crit) + {c.alpha_q:g}/K * dTF in {c.q_bounds}, "
                f"h0 = {self.h00:g} + {c.clim_h:g} m/K * (TF_clim - tf_crit) + {c.alpha_h:g} m/K * dTF in {c.h0_bounds}")

    def anomaly(self, t0: float, t1: float) -> np.ndarray:
        """dTF(x) = TF_step - TF_clim for the step (t0, t1] on the fine grid;
        0 where inactive and before the record."""
        years, stat = self.years, self.stat
        ya, yb = int(years[0]), int(years[-1])
        overlap = [(min(y, yb), w) for y, w in year_overlap_weights(t0, t1) if y >= ya]
        if not overlap:
            return self._zero
        idx = [int(np.searchsorted(years, y)) for y, _ in overlap]
        if self.cfg.statistic == "max":
            agg = stat[idx].max(axis=0)
        else:
            wsum = sum(w for _, w in overlap)
            agg = sum(w * stat[i] for (_, w), i in zip(overlap, idx)) / wsum
        return np.where(self.ok, agg - self.clim, 0.0).astype(np.float32)

    def margins(self, t0: float, t1: float):
        """(q, h0, dTF) fields (float32, fine grid) for the step (t0, t1]:
        baseline + clim * TF_clim + alpha * dTF, clipped to the bounds."""
        c = self.cfg
        dtf = self.anomaly(t0, t1)
        clim_rel = np.where(self.ok, self.clim - c.tf_crit, 0.0).astype(np.float32)
        q = np.clip(self.q0 + c.clim_q * clim_rel + c.alpha_q * dtf, *c.q_bounds).astype(np.float32)
        h0 = np.clip(self.h00 + c.clim_h * clim_rel + c.alpha_h * dtf, *c.h0_bounds).astype(np.float32)
        return q, h0, dtf
