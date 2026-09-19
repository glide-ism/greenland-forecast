"""
Year-by-year climate forcing for the years a reanalysis record covers.

`GlacierConfig.yearly_climate_filename` names a file in model_inputs with
per-(year, month) departures of the forcing from the climatology the model
holds (`t2m_anom`, K, additive; `precip_ratio`, multiplicative), built by
`preprocessing/make_climate_yearly.py`. For a step overlapping such a year
the SMB is evaluated on `t2m_clim + t2m_anom(year) + tbias` and
`precip_clim * pbias * precip_ratio(year)` — the reanalysis year itself plus
the calibrated biases — instead of the climatology shifted by the scalar
index anomaly (forward.simulate builds one term per record year; years
outside the record keep the index).

Memory: a year is (2, 12, ny, nx) — 460 MB float32 at 1 km — and a level-0
run overlaps ~40 of them, so nothing here may land on the autograd tape.
The checkpointed SMB fn receives THIS OBJECT and a year key (non-tensor
args, like the scalar anomalies), materializes the fields on the GPU inside
the checkpoint, and re-materializes them in the backward recompute; they die
with the checkpoint's forward. The record is stored as the file's int16
codes, either in (pinned) host RAM (`cache="ram"`, ~230 MB per year) or read
per call from the file (`cache="none"`, the page cache does the rest);
decoding to float32 happens on the device.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import netCDF4
import numpy as np
import torch

VARS = ("t2m_anom", "precip_ratio")


class YearlyClimate:
    def __init__(self, path, years, crop, scale, offset, cache: str = "ram"):
        self.path = Path(path)
        self.years = [int(y) for y in years]
        self._index = {y: k for k, y in enumerate(self.years)}
        self._crop = crop                      # (y0, y1, x0, x1) into the file's grid
        self._scale = {v: float(scale[v]) for v in VARS}
        self._offset = {v: float(offset[v]) for v in VARS}
        self.cache = cache
        self._nc: Optional[netCDF4.Dataset] = None
        self._host: Optional[dict] = None
        if cache == "ram":
            self._load_host()
        elif cache != "none":
            raise ValueError(f"yearly_climate_cache={cache!r}: expected 'ram' or 'none'")

    # ------------------------------------------------------------ construction
    @classmethod
    def from_file(cls, path, crop_factor: int, cache: str = "ram") -> "YearlyClimate":
        """Open `path` on the same centred crop the problem applies to the
        gridded inputs (ny, nx to multiples of `crop_factor`)."""
        with netCDF4.Dataset(path) as nc:
            ny0, nx0 = nc.dimensions["y"].size, nc.dimensions["x"].size
            ny, nx = (ny0 // crop_factor) * crop_factor, (nx0 // crop_factor) * crop_factor
            y0, x0 = (ny0 - ny) // 2, (nx0 - nx) // 2
            years = np.asarray(nc["year"][:]).astype(int)
            scale = {v: nc[v].getncattr("scale_factor") for v in VARS}
            offset = {v: (nc[v].getncattr("add_offset") if "add_offset" in nc[v].ncattrs() else 0.0)
                      for v in VARS}
        return cls(path, years, (y0, y0 + ny, x0, x0 + nx), scale, offset, cache)

    def _open(self):
        if self._nc is None:
            self._nc = netCDF4.Dataset(self.path)
            for v in VARS:
                self._nc[v].set_auto_maskandscale(False)      # raw int16 codes
        return self._nc

    def _read_codes(self, var: str, k: int) -> np.ndarray:
        y0, y1, x0, x1 = self._crop
        return np.asarray(self._open()[var][k, :, y0:y1, x0:x1], dtype=np.int16)

    def _load_host(self):
        y0, y1, x0, x1 = self._crop
        n = len(self.years)
        self._host = {}
        for v in VARS:
            t = torch.empty((n, 12, y1 - y0, x1 - x0), dtype=torch.int16)
            try:
                t = t.pin_memory()
            except RuntimeError:                              # no pinnable memory: pageable
                pass
            for k in range(n):
                t[k] = torch.from_numpy(self._read_codes(v, k))
            self._host[v] = t
        self._nc.close()
        self._nc = None

    # ---------------------------------------------------------------- queries
    def has(self, year: int) -> bool:
        return int(year) in self._index

    @property
    def first_year(self) -> int:
        return self.years[0]

    @property
    def last_year(self) -> int:
        return self.years[-1]

    def _field(self, var: str, year: int) -> torch.Tensor:
        k = self._index[int(year)]
        if self._host is not None:
            codes = self._host[var][k].to("cuda", non_blocking=True)
        else:
            codes = torch.from_numpy(self._read_codes(var, k)).to("cuda")
        return codes.to(torch.float32) * self._scale[var] + self._offset[var]

    def t2m_anomaly(self, year: int) -> torch.Tensor:
        """(12, ny, nx) K, additive on the model's monthly t2m climatology."""
        return self._field("t2m_anom", year)

    def precip_ratio(self, year: int) -> torch.Tensor:
        """(12, ny, nx), multiplicative on the model's monthly precip climatology."""
        return self._field("precip_ratio", year)

    def describe(self) -> str:
        ny, nx = self._crop[1] - self._crop[0], self._crop[3] - self._crop[2]
        where = ("pinned host RAM" if self._host is not None and self._host["t2m_anom"].is_pinned()
                 else "host RAM" if self._host is not None else "the file per call")
        mb = 2 * 12 * ny * nx * 2 / 1e6
        return (f"yearly climate: {self.path.name}, {len(self.years)} years "
                f"{self.first_year}-{self.last_year} on the ({ny}, {nx}) crop, int16 codes in {where} "
                f"({mb:.0f} MB per year); fields decoded on the GPU inside each SMB checkpoint")
