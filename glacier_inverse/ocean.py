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

    q (x, t) = q0  + clim_q * (TF_clim(x) - tf_crit) + alpha_q * (dTF(x, t) - rho(x))  [1/K]
    h0(x, t) = h00 + clim_h * (TF_clim(x) - tf_crit) + alpha_h * (dTF(x, t) - rho(x))  [m/K]
    TF_clim(x) = mean of the annual statistic over `ref_years`
    rho(x)     = the critical-anomaly field (K, config `rho_filename`,
                 preprocessing/make_calving_rho.py; 0 without one): a front
                 flips when its anomaly exceeds rho, so rho is its distance
                 to threshold in the reference climate and alpha_h the
                 metres of margin per kelvin of exceedance. The same file may
                 carry `h0_fixed` (m): where finite it REPLACES h0 -- the
                 builder puts +250 m outside the observed extent so a
                 negative inside margin cannot advance the fronts (with
                 h0 < 0 everywhere, floating ice is admissible everywhere and
                 every fjord fills; found the hard way 2026-09-24)
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
contracted). Steps before the record see the HELD pre-record anomaly
dTF_pre = max(pre_record_scale TF_clim + pre_record_offset, 0) - TF_clim,
ramped to 0 over `pre_record_ramp` years before the record (0 with the
defaults); after the record the last year holds. Both terms are zero (margins at their baselines) where
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


class PinnedFront:
    """Calving margins that hold the front at a given ice mask (config
    `pin_front`), with the same surface as OceanForcing so forward.simulate,
    GlacierProblem and forward_standalone consume it unchanged:

        h0(x) = pin_h0_inside  where mask, else pin_h0_outside   (time-invariant)
        q (x) = calving_q,  dTF(x) = 0

    Inside the mask the monotone law with h0 << 0 never removes grounded ice
    and leaves floating ice to H_c alone; outside, h0 = +250 m removes ice
    within 250 m of flotation at the calving timescale, i.e. any marine
    re-advance. See OceanForcingConfig.pin_front for why."""

    def __init__(self, cfg: OceanForcingConfig, mask, *, q0: float, h00: float, source: str = "",
                 years: Optional[np.ndarray] = None):
        self.cfg = cfg
        if years is not None:                      # (nt, ny, nx) masks, one per calendar year
            self.masks = np.asarray(mask, dtype=bool)
            if self.masks.ndim != 3 or len(years) != self.masks.shape[0]:
                raise ValueError(f"PinnedFront: yearly masks must be (nt, ny, nx) with nt = len(years), got {self.masks.shape}")
            self.mask_years = np.asarray(years, dtype=int)
            mask = self.masks[-1]                  # the static attributes below describe the LAST year
        else:
            self.masks, self.mask_years = None, None
        self.mask = np.asarray(mask, dtype=bool)
        if self.mask.ndim != 2:
            raise ValueError(f"PinnedFront: mask must be (ny, nx), got {self.mask.shape}")
        self.q0, self.h00 = float(q0), float(h00)
        self.source = source
        self.freeze_anomaly = True          # nothing varies: climatology_only is moot
        self.years = np.array([], dtype=int)
        self._file = None
        lo, hi = cfg.h0_bounds
        h_in, h_out = float(cfg.pin_h0_inside), float(cfg.pin_h0_outside)
        if not (lo <= h_in <= hi and lo <= h_out <= hi):
            raise ValueError(f"pin_h0_inside / pin_h0_outside ({h_in:g}, {h_out:g}) must lie in "
                             f"h0_bounds {cfg.h0_bounds}")
        self._h_in, self._h_out = h_in, h_out
        self._h0 = np.where(self.mask, h_in, h_out).astype(np.float32)
        self._q = np.full(self.mask.shape, float(np.clip(self.q0, *cfg.q_bounds)), np.float32)
        self._zero = np.zeros(self.mask.shape, np.float32)
        self.ok = self.mask
        self.clim = self._zero
        self._cache = (None, None)                 # (year index, h0) of the last step served

    @classmethod
    def from_file(cls, path, crop_factor: int, cfg: OceanForcingConfig, *, q0: float, h00: float) -> "PinnedFront":
        """The yearly masks of model_inputs/<pin_front_filename> (`front_mask`
        (time, y, x), preprocessing/make_front_mask.py), cropped like
        GLIDE_inputs."""
        path = Path(path)
        with xr.open_dataset(path) as f:
            ds = _crop_to_factor(f, crop_factor)
            years = ds.time.values.astype(int)
            masks = ds.front_mask.values > 0
            src = f"{path.name} ({ds.attrs.get('source', '')}; {years[0]}-{years[-1]})"
        return cls(cfg, masks, q0=q0, h00=h00, source=src, years=years)

    def mask_at(self, t1: float) -> np.ndarray:
        """The mask a step ending at t1 is pinned to: the END year's, clipped to the record."""
        if self.masks is None:
            return self.mask
        y = int(math.floor(t1 - 1e-6))
        k = int(np.clip(np.searchsorted(self.mask_years, y, side="right") - 1, 0, len(self.mask_years) - 1))
        return self.masks[k]

    @classmethod
    def from_gridded(cls, gridded, cfg: OceanForcingConfig, *, q0: float, h00: float) -> "PinnedFront":
        """From the (already cropped) gridded inputs: the boolean variable
        cfg.pin_front, taken as True where > 0.5."""
        if cfg.pin_front not in gridded:
            raise KeyError(f"ocean_forcing.pin_front={cfg.pin_front!r} is not a variable of the gridded inputs")
        v = gridded[cfg.pin_front]
        src = f"{cfg.pin_front}" + (f" @ {v.attrs['time_nominal']:g}" if "time_nominal" in v.attrs else "")
        return cls(cfg, v.values > 0.5, q0=q0, h00=h00, source=src)

    @property
    def ny_nx(self):
        return self.mask.shape

    def describe(self) -> str:
        c = self.cfg
        if self.masks is not None:
            n = self.masks.sum(axis=(1, 2))
            return (f"ocean forcing: FRONT PINNED to the YEARLY masks of {self.source}: {len(self.mask_years)} years, "
                    f"{int(n.min())}-{int(n.max())} masked cells, each step to its end year's mask (first / last held outside); "
                    f"h0 = {c.pin_h0_inside:g} m inside / {c.pin_h0_outside:g} m outside, q = {self.q0:g}, dTF = 0; "
                    f"the thermal forcing is not read")
        return (f"ocean forcing: FRONT PINNED to {self.source} ({int(self.mask.sum())} masked cells, "
                f"{int((~self.mask).sum())} outside); h0 = {c.pin_h0_inside:g} m inside / "
                f"{c.pin_h0_outside:g} m outside, q = {self.q0:g}, dTF = 0, time-invariant; "
                f"the thermal forcing is not read")

    def anomaly(self, t0: float, t1: float) -> np.ndarray:
        return self._zero

    def margins(self, t0: float, t1: float):
        if self.masks is None:
            return self._q, self._h0, self._zero
        y = int(math.floor(t1 - 1e-6))
        k = int(np.clip(np.searchsorted(self.mask_years, y, side="right") - 1, 0, len(self.mask_years) - 1))
        if self._cache[0] != k:
            self._cache = (k, np.where(self.masks[k], self._h_in, self._h_out).astype(np.float32))
        return self._q, self._cache[1], self._zero


class ReleasedPin:
    """A front pin until `t_release`, the free TF-driven margins after
    (config `pin_release_year`): steps ending at or before t_release take
    pin.margins, later ones free.margins. Same surface as OceanForcing."""

    def __init__(self, pin: "PinnedFront", free: "OceanForcing", t_release: float):
        self.pin, self.free, self.t_release = pin, free, float(t_release)
        self.cfg, self.ok, self.clim, self.years = free.cfg, free.ok, free.clim, free.years
        self.freeze_anomaly = free.freeze_anomaly
        self._file = getattr(free, "_file", None)

    @property
    def ny_nx(self):
        return self.free.ny_nx

    def _src(self, t1: float):
        return self.pin if t1 <= self.t_release + 1e-6 else self.free

    def describe(self) -> str:
        return (f"RELEASED PIN: steps ending <= {self.t_release:g} -> {self.pin.describe()} || "
                f"after -> {self.free.describe()}")

    def anomaly(self, t0: float, t1: float) -> np.ndarray:
        return self._src(t1).anomaly(t0, t1)

    def margins(self, t0: float, t1: float):
        return self._src(t1).margins(t0, t1)


class OceanForcing:
    def __init__(self, cfg: OceanForcingConfig, *, years: np.ndarray, stat, dist: np.ndarray,
                 q0: float, h00: float, source: str = "", freeze_anomaly: bool = False,
                 rho: Optional[np.ndarray] = None, rho_source: str = "",
                 h0_fixed: Optional[np.ndarray] = None, h0_base: Optional[np.ndarray] = None,
                 index: Optional[tuple] = None):
        if cfg.statistic not in ("max", "mean"):
            raise ValueError(f"OceanForcingConfig.statistic must be 'max' or 'mean', got {cfg.statistic!r}")
        self.cfg = cfg
        # critical-anomaly field rho(x) [K]: h0 gets - alpha_h * rho; NaN -> 0
        self.rho = None if rho is None else np.nan_to_num(np.asarray(rho, dtype=np.float32), nan=0.0)
        self.rho_source = rho_source
        # fixed-margin override (m): h0 = h0_fixed where finite (the bound on advance)
        self.h0_fixed = None if h0_fixed is None else np.asarray(h0_fixed, dtype=np.float32)
        # additive static margin (m): the per-basin c_i of h0_i = c_i + alpha dTF_i
        # (sweep_calving_c.py + analysis/sweep_calving_eval.py); NaN -> 0
        self.h0_base = None if h0_base is None else np.nan_to_num(np.asarray(h0_base, dtype=np.float32), nan=0.0)
        self.years = np.asarray(years, dtype=int)
        # (nt, ny, nx): a dense float32 array or a _LazyYears reader
        self.stat = stat if isinstance(stat, _LazyYears) else np.asarray(stat, dtype=np.float32)
        self.q0, self.h00 = float(q0), float(h00)
        self.source = source
        # config.climatology_only: hold the ocean at its reference climate,
        # dTF == 0 for every step, so the margins are the time-invariant
        # q0 + clim_q (TF_clim - tf_crit) / h00 + clim_h (TF_clim - tf_crit).
        # The TF climatology itself is still read and still shapes the
        # margins' geography - only the interannual departure is removed.
        self.freeze_anomaly = bool(freeze_anomaly)
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
        # the held pre-record anomaly (K); None when it is the identity hold
        held = np.maximum(cfg.pre_record_scale * self.clim + cfg.pre_record_offset, 0.0) - self.clim
        held = np.where(self.ok, held, 0.0).astype(np.float32)
        self.pre_held = held if np.any(held != 0.0) else None
        # the pre-record index term: {year: k (I_s - I_ref)} (K), None when off
        self.pre_index, self.index_source = None, ""
        if index is not None and cfg.pre_record_index:
            iy, iv, self.index_source = index
            iy, iv = np.asarray(iy, dtype=int), np.asarray(iv, dtype=float)
            n = max(int(cfg.pre_record_index_smooth), 1)
            num = np.convolve(np.nan_to_num(iv), np.ones(n), "same")
            den = np.convolve(np.isfinite(iv).astype(float), np.ones(n), "same")
            ism = num / np.maximum(den, 1e-9)                  # centred mean, window shrinks at the ends
            sel = (iy >= y0) & (iy <= y1)
            if not sel.any():
                raise ValueError(f"pre_record_index: no index years inside ref_years {cfg.ref_years}")
            iref = ism[sel].mean()
            self.pre_index = {int(y): float(cfg.pre_record_index_k * (v - iref)) for y, v in zip(iy, ism)}
            self._index_first = int(iy.min())

    @classmethod
    def from_file(cls, path, crop_factor: int, cfg: OceanForcingConfig, *,
                  q0: float, h00: float, lazy: bool = False,
                  freeze_anomaly: bool = False, rho_path=None, index_path=None) -> "OceanForcing":
        """Load only the needed annual statistic from thermal_forcing.nc,
        cropped like GLIDE_inputs. Dense (the file is closed before
        returning) or, with `lazy`, read year by year from the file kept
        open (long projection records). With cfg.rho_filename the
        critical-anomaly field is read from `rho_path` (default: next to
        the TF file, i.e. model_inputs/<rho_filename>; a projection whose TF
        lives elsewhere passes the domain's path), cropped the same way."""
        path = Path(path)
        rho, rho_source, h0_fixed, h0_base = None, "", None, None
        if cfg.rho_filename:
            rp = Path(rho_path) if rho_path is not None else path.parent / cfg.rho_filename
            if not rp.exists():
                raise FileNotFoundError(f"ocean_forcing.rho_filename={cfg.rho_filename!r}: {rp} not found "
                                        f"(preprocessing/make_calving_rho.py)")
            with xr.open_dataset(rp) as rf:
                rds = _crop_to_factor(rf, crop_factor)
                rho = rds["calving_rho"].values.astype(np.float32) if "calving_rho" in rds else None
                rho_source = str(rds.attrs.get("source", rp.name))
                if "h0_fixed" in rds:
                    h0_fixed = rds["h0_fixed"].values.astype(np.float32)
                if "h0_base" in rds:
                    h0_base = rds["h0_base"].values.astype(np.float32)
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
        index = None
        if cfg.pre_record_index:
            ip = Path(index_path) if index_path is not None else path.parent / cfg.pre_record_index
            if not ip.exists():
                raise FileNotFoundError(f"ocean_forcing.pre_record_index={cfg.pre_record_index!r}: {ip} not found "
                                        f"(preprocessing/make_temperature_anomaly.py)")
            with xr.open_dataset(ip) as f_i:
                index = (f_i.time.values, f_i.temp_anomaly.values, f"{ip.name} ({f_i.attrs.get('source', '')})")
        of = cls(cfg, years=years, stat=stat, dist=dist, q0=q0, h00=h00, source=source,
                 freeze_anomaly=freeze_anomaly, rho=rho, rho_source=rho_source, h0_fixed=h0_fixed,
                 h0_base=h0_base, index=index)
        of._file = f if lazy else None
        return of

    @property
    def ny_nx(self):
        return self.stat.shape[1:]

    def describe(self) -> str:
        c = self.cfg
        return (f"ocean forcing: TF {self.years[0]}-{self.years[-1]} (tf_{c.statistic}), "
                f"climatology {c.ref_years}, tf_crit {c.tf_crit:g} degC, {int(self.ok.sum())} active cells within {c.max_dist_km:g} km; "
                f"q = {self.q0:g} + {c.clim_q:g}/K * (TF_clim - tf_crit) + {c.alpha_q:g}/K * "
                + ("(dTF - rho)" if self.rho is not None else "dTF") + f" in {c.q_bounds}, "
                f"h0 = {self.h00:g}"
                + (f" + h0_base(x) in [{np.nanmin(self.h0_base):g}, {np.nanmax(self.h0_base):g}] m ({self.rho_source})"
                   if self.h0_base is not None else "")
                + f" + {c.clim_h:g} m/K * (TF_clim - tf_crit) + {c.alpha_h:g} m/K * "
                + (f"(dTF - rho) with rho the critical-anomaly field {self.rho_source} "
                   f"({int((self.rho > 0).sum())} cells > 0, max {float(self.rho.max()):.2f} K"
                   + (f"; h0 FIXED at {np.nanmin(self.h0_fixed):g}..{np.nanmax(self.h0_fixed):g} m on "
                      f"{int(np.isfinite(self.h0_fixed).sum())} cells, the bound on advance"
                      if self.h0_fixed is not None else "") + ")"
                   if self.rho is not None else "dTF")
                + f" in {c.h0_bounds}"
                + (f"; before {self.years[0]}: TF held at max({c.pre_record_scale:g} TF_clim + {c.pre_record_offset:g} K, 0)"
                   f" (dTF {float(self.pre_held[self.ok].min()):+.2f}..{float(self.pre_held[self.ok].max()):+.2f} K"
                   + (f", ramped to 0 over the {c.pre_record_ramp:g} yr before" if c.pre_record_ramp > 0 else "") + ")"
                   if self.pre_held is not None else "")
                + (f"; before {self.years[0]}: + {c.pre_record_index_k:g} K/K x ({c.pre_record_index_smooth}-yr mean of "
                   f"{self.index_source} - its {c.ref_years} mean)"
                   + (f" from {c.pre_record_index_start:g}" if c.pre_record_index_start is not None else "") + ", "
                   f"1850-99 {np.mean([self.pre_index.get(y, 0.0) for y in range(1850, 1900)]):+.2f} K, "
                   f"1920-49 {np.mean([self.pre_index.get(y, 0.0) for y in range(1920, 1950)]):+.2f} K"
                   if self.pre_index is not None else "")
                + ("; CLIMATOLOGY MODE: dTF held at 0, the margins are time-invariant"
                   if self.freeze_anomaly else ""))

    def _pre_field(self, y: int) -> np.ndarray:
        """TF for a calendar year before the record: the climatology + the
        (ramped) hold + the index term (the index's first value held before it)."""
        d = 0.0
        if self.pre_held is not None:
            d = self._pre_weight(y) * self.pre_held
        start = self.cfg.pre_record_index_start
        if self.pre_index is not None and (start is None or y >= start):
            d = d + self.pre_index.get(int(y), self.pre_index.get(self._index_first, 0.0))
        return np.maximum(self.clim + d, 0.0) if np.ndim(d) or d else self.clim

    def _pre_weight(self, y: int) -> float:
        """Weight of the held anomaly for calendar year y < the record start:
        1 before the ramp, linear to 0 at the record start (year centres)."""
        ramp = float(self.cfg.pre_record_ramp)
        if ramp <= 0:
            return 1.0
        return float(min(max((int(self.years[0]) - (y + 0.5)) / ramp, 0.0), 1.0))

    def anomaly(self, t0: float, t1: float) -> np.ndarray:
        """dTF(x) = TF_step - TF_clim for the step (t0, t1] on the fine grid;
        0 where inactive and before the record."""
        if self.freeze_anomaly:
            return self._zero
        years, stat = self.years, self.stat
        ya, yb = int(years[0]), int(years[-1])
        weights = year_overlap_weights(t0, t1)
        if self.pre_held is None and self.pre_index is None:
            overlap = [(min(y, yb), w) for y, w in weights if y >= ya]
            if not overlap:
                return self._zero
            idx = [int(np.searchsorted(years, y)) for y, _ in overlap]
            if self.cfg.statistic == "max":
                agg = stat[idx].max(axis=0)
            else:
                wsum = sum(w for _, w in overlap)
                agg = sum(w * stat[i] for (_, w), i in zip(overlap, idx)) / wsum
            return np.where(self.ok, agg - self.clim, 0.0).astype(np.float32)
        # with a held pre-record anomaly every overlapped year counts: record
        # years by their TF, earlier ones by TF_clim + w(y) * held
        fields = [self._pre_field(y) if y < ya
                  else stat[int(np.searchsorted(years, min(y, yb)))] for y, _ in weights]
        if self.cfg.statistic == "max":
            agg = np.maximum.reduce(fields)
        else:
            agg = sum(w * f for (_, w), f in zip(weights, fields))
        return np.where(self.ok, agg - self.clim, 0.0).astype(np.float32)

    def margins(self, t0: float, t1: float):
        """(q, h0, dTF) fields (float32, fine grid) for the step (t0, t1]:
        baseline + clim * TF_clim + alpha * dTF, clipped to the bounds."""
        c = self.cfg
        dtf = self.anomaly(t0, t1)
        clim_rel = np.where(self.ok, self.clim - c.tf_crit, 0.0).astype(np.float32)
        q = self.q0 + c.clim_q * clim_rel + c.alpha_q * dtf
        h0 = self.h00 + c.clim_h * clim_rel + c.alpha_h * dtf
        if self.h0_base is not None:
            h0 = h0 + self.h0_base
        if self.rho is not None:
            # the field shifts BOTH anomaly responses: alpha_h (dTF - rho) in metres,
            # alpha_q (dTF - rho) as a fraction of the thickness -- the latter is what
            # reaches a front the spin-up has grounded tens of metres above flotation
            h0 = h0 - c.alpha_h * self.rho          # static, and independent of `ok`: the builder sets coverage
            q = q - c.alpha_q * self.rho
        q = np.clip(q, *c.q_bounds).astype(np.float32)
        if self.h0_fixed is not None:
            h0 = np.where(np.isfinite(self.h0_fixed), self.h0_fixed, h0)
        h0 = np.clip(h0, *c.h0_bounds).astype(np.float32)
        return q, h0, dtf
