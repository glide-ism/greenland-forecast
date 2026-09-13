"""Target model grid for a Greenland domain.

A domain declares its grid in `local_data/domain.json` (preferred) or, as in
alaska-forecast, through `local_data/outline.kml`:

    {"preset": "ismip_greenland", "resolution": 1000}
    {"crs": "EPSG:3413", "bbox": [xmin, ymin, xmax, ymax], "resolution": 500,
     "outline": "outline.kml"}          # optional polygon -> domain_mask

The `ismip_greenland` preset is the ISMIP6/ISMIP7 standard Greenland grid:
EPSG:3413, node coordinates x in [-720, 960] km and y in [-3450, -570] km,
so the 1/2/4/8 km grids nest exactly and the inverse solution can be handed
to the ISMIP output tools without resampling. Cell centres coincide with the
ISMIP nodes; the raster edges lie half a cell outside. `bbox` domains are
snapped outward to the resolution and, when they lie on the ISMIP lattice
(offsets that are multiples of the resolution from -720000/-3450000), also
nest in it.

Every builder calls `DomainGrid.template()` for a georeferenced (y, x)
DataArray to `rio.reproject_match` onto, so all products share one grid.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pyproj
import rasterio.transform
import xarray as xr

from projection_dictionary import crs as PROJECT_CRS

# ISMIP6 / ISMIP7 standard Greenland grid (node coordinates, EPSG:3413).
ISMIP_GREENLAND = dict(x_min=-720000.0, x_max=960000.0,
                       y_min=-3450000.0, y_max=-570000.0)


@dataclass(frozen=True)
class DomainGrid:
    crs: pyproj.CRS
    resolution: float
    # Raster EDGES (not cell centres).
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    outline_path: Optional[Path] = None   # WGS84 polygon file -> domain_mask
    preset: Optional[str] = None

    @property
    def width(self) -> int:
        return int(round((self.xmax - self.xmin) / self.resolution))

    @property
    def height(self) -> int:
        return int(round((self.ymax - self.ymin) / self.resolution))

    @property
    def transform(self):
        return rasterio.transform.from_origin(
            self.xmin, self.ymax, self.resolution, self.resolution)

    @property
    def x(self) -> np.ndarray:
        """Cell-centre x (ascending)."""
        return self.xmin + self.resolution * (np.arange(self.width) + 0.5)

    @property
    def y(self) -> np.ndarray:
        """Cell-centre y (descending, north-up raster convention)."""
        return self.ymax - self.resolution * (np.arange(self.height) + 0.5)

    @property
    def bounds(self):
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    def template(self, name: str = "template") -> xr.DataArray:
        """Zero-filled georeferenced (y, x) float32 DataArray on this grid,
        the `match_data_array` for rioxarray.reproject_match."""
        import rioxarray  # noqa: F401
        da = xr.DataArray(
            np.zeros((self.height, self.width), dtype=np.float32),
            dims=("y", "x"),
            coords={"y": self.y.astype("float64"), "x": self.x.astype("float64")},
            name=name,
        )
        da = da.rio.write_crs(self.crs.to_wkt(), inplace=True)
        da = da.rio.write_transform(self.transform, inplace=True)
        return da

    def centroid_latlon(self) -> tuple:
        """(lat, lon) of the grid centre, for the solar geometry."""
        to_ll = pyproj.Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        lon, lat = to_ll.transform(0.5 * (self.xmin + self.xmax),
                                   0.5 * (self.ymin + self.ymax))
        return float(lat), float(lon)

    def outline_polygon(self):
        """The optional WGS84 domain polygon (shapely), or None."""
        if self.outline_path is None or not self.outline_path.exists():
            return None
        import geopandas
        gdf = geopandas.read_file(self.outline_path)
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(4326)
        return gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") \
            else gdf.geometry.unary_union

    def describe(self) -> str:
        return (f"{self.crs.to_string()} @ {self.resolution:g} m, "
                f"{self.height} x {self.width} cells, x [{self.xmin:.0f}, {self.xmax:.0f}], "
                f"y [{self.ymin:.0f}, {self.ymax:.0f}]"
                + (f" (preset {self.preset})" if self.preset else ""))


def _snap_outward(bbox, res, origin=(0.0, 0.0)):
    ox, oy = origin
    xmin, ymin, xmax, ymax = bbox
    return (ox + math.floor((xmin - ox) / res) * res,
            oy + math.floor((ymin - oy) / res) * res,
            ox + math.ceil((xmax - ox) / res) * res,
            oy + math.ceil((ymax - oy) / res) * res)


def _outline_bbox(outline_path: Path, crs) -> tuple:
    import geopandas
    gdf = geopandas.read_file(outline_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    return tuple(float(v) for v in gdf.to_crs(crs).total_bounds)


def load_domain_grid(domain_path) -> DomainGrid:
    """Resolve the target grid of `domain_path` (see the module docstring)."""
    domain_path = Path(domain_path)
    local = domain_path / "local_data"
    spec_path = local / "domain.json"
    spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}

    crs = pyproj.CRS(spec.get("crs", PROJECT_CRS.to_string()))
    res = float(spec.get("resolution", 1000.0))
    outline = spec.get("outline", "outline.kml")
    outline_path = local / outline if outline else None

    preset = spec.get("preset")
    if preset == "ismip_greenland":
        if crs.to_epsg() != 3413:
            raise ValueError("the ismip_greenland preset is defined on EPSG:3413")
        g = ISMIP_GREENLAND
        for k in ("x_max", "y_max"):
            span = g[k] - g[k.replace("max", "min")]
            if abs(span / res - round(span / res)) > 1e-9:
                raise ValueError(
                    f"resolution {res} m does not tile the ISMIP Greenland "
                    f"grid ({span:.0f} m span); use 1000, 2000, 4000 or 8000")
        # Cell centres on the ISMIP nodes -> edges half a cell outside.
        return DomainGrid(crs=crs, resolution=res,
                          xmin=g["x_min"] - res / 2, ymin=g["y_min"] - res / 2,
                          xmax=g["x_max"] + res / 2, ymax=g["y_max"] + res / 2,
                          outline_path=outline_path if (outline_path and outline_path.exists()) else None,
                          preset=preset)
    if preset is not None:
        raise ValueError(f"unknown domain preset {preset!r}")

    if "bbox" in spec:
        bbox = tuple(float(v) for v in spec["bbox"])
    elif outline_path is not None and outline_path.exists():
        bbox = _outline_bbox(outline_path, crs)
    else:
        raise FileNotFoundError(
            f"{spec_path} needs a 'preset' or 'bbox', or {outline_path} must exist")
    # Snap to the ISMIP lattice (cell EDGES at node - res/2) when on EPSG:3413
    # so sub-domains nest in the standard grids; plain multiples otherwise.
    origin = ((ISMIP_GREENLAND["x_min"] - res / 2, ISMIP_GREENLAND["y_min"] - res / 2)
              if crs.to_epsg() == 3413 else (0.0, 0.0))
    xmin, ymin, xmax, ymax = _snap_outward(bbox, res, origin)
    return DomainGrid(crs=crs, resolution=res, xmin=xmin, ymin=ymin,
                      xmax=xmax, ymax=ymax,
                      outline_path=outline_path if (outline_path and outline_path.exists()) else None)


def grid_from_dem(dem: xr.Dataset) -> DomainGrid:
    """Recover the DomainGrid of an existing gridded_dem.nc (for builders
    that run after make_dem.py)."""
    x = dem.x.values.astype("float64")
    y = dem.y.values.astype("float64")
    res = float(abs(x[1] - x[0]))
    crs = pyproj.CRS(dem.spatial_ref.crs_wkt)
    return DomainGrid(crs=crs, resolution=res,
                      xmin=float(x.min() - res / 2), ymin=float(y.min() - res / 2),
                      xmax=float(x.max() + res / 2), ymax=float(y.max() + res / 2))


def write_grid_metadata(ds: xr.Dataset, grid: DomainGrid) -> xr.Dataset:
    """Attach the CF grid mapping (`spatial_ref`) the inverse model reads the
    CRS from, on a dataset already carrying this grid's y/x coordinates."""
    import rioxarray  # noqa: F401
    ds = ds.rio.write_crs(grid.crs.to_wkt(), inplace=True)
    ds = ds.rio.write_transform(grid.transform, inplace=True)
    ds.attrs["grid_resolution_m"] = grid.resolution
    if grid.preset:
        ds.attrs["grid_preset"] = grid.preset
    return ds
