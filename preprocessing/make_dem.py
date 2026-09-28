"""Build the gridded geometry dataset for a Greenland domain from BedMachine.

Replaces alaska-forecast's Copernicus-DEM + NOAA-bathymetry splice: BedMachine
Greenland (Morlighem et al., NSIDC IDBMG4) already carries surface, bed,
thickness, fjord bathymetry (IBCAO), a class mask and a per-cell bed error on
one 150 m EPSG:3413 grid, so a single conservative regrid onto the domain grid
gives every geometry field the inverse model needs.

Pipeline:
    1. Resolve the target grid (domain_grid.load_domain_grid).
    2. Regrid BedMachine `surface`, `bed`, `thickness`, `errbed` (area
       average) and the class `mask` (fractions per class + mode) onto it.
    3. Compose `elevation` = surface over ice, bed elsewhere (land topography
       and fjord/shelf bathymetry), the ice mask (`rgi_mask` — the field name
       the inverse model reads), floating mask, domain mask, and the gridded
       bed observation (`bed_obs`, `bed_obs_err`) that
       glacier_inverse.priors.build_bed_conditioning_data conditions on.
    4. Optionally overlay a dated surface DEM (ArcticDEM v4.1 mosaic) over
       ice, with its own acquisition attrs.
    5. Rasterize per-basin integer labels (`rgi_label`; Mouginot & Rignot
       2019 drainage basins + RGI 7.0 region 05 peripheral glaciers) with a
       `glacier`-indexed lookup (`rgi_id`) and `surge_type` (0 for ice-sheet
       basins, RGI's attribute for peripheral glaciers).

Inputs (paths at the top; see DATA_MANIFEST.md):
    common_data/geometry/bedmachine/BedMachineGreenland-v5.nc
    common_data/area/basins/Greenland_Basins_PS_v1.4.2.shp        (optional)
    common_data/area/rgi/RGI2000-v7.0-G-05_greenland_periphery.shp (optional)
    common_data/dem/arcticdem/arcticdem_mosaic_100m_v4.1_dem.tif    (optional)

Output: {domain_path}/model_inputs/gridded_dem.nc
"""
import argparse
import warnings
from pathlib import Path

import geopandas
import numpy as np
import rasterio.features
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from rasterio.enums import Resampling

from domain_grid import load_domain_grid, write_grid_metadata

BEDMACHINE_PATH = Path('../common_data/geometry/bedmachine/BedMachineGreenland-v6.nc')
BASINS_PATH = Path('../common_data/area/basins/Greenland_Basins_PS_v1.4.2.shp')
# RGI 6.0 (NSIDC-0770 v6) or 7.0 region 05 — both attribute schemas are handled.
RGI_PATH = Path('../common_data/area/rgi/05_rgi60_GreenlandPeriphery.shp')
ARCTICDEM_PATH = Path('../common_data/dem/arcticdem/arcticdem_mosaic_100m_v4.1_dem.tif')

BEDMACHINE_CRS = "EPSG:3413"
# BedMachine `mask` codes (v5/v6): 0 ocean, 1 ice-free land, 2 grounded, 3
# floating; v5 also had 4 = non-Greenland ice (absent in v6, harmless here).
BM_OCEAN, BM_LAND, BM_GROUNDED, BM_FLOATING, BM_NONGREENLAND = 0, 1, 2, 3, 4

# Acquisition epochs (variable-level attrs read by the inverse model).
# BedMachine's surface is the GIMP DEM v2.1 (2003-2009 imagery); v6 carries
# a `nominal_year` global attribute (2008) which build_dem prefers when
# present. The ArcticDEM overlay (median mosaic of 2007-2020 strips,
# nominal ~2015) gives a better-dated surface when available.
BEDMACHINE_SURFACE_TIME_ATTRS = {'time_nominal': 2008.0, 'time_start': 2003.0,
                                 'time_end': 2009.0}
ARCTICDEM_TIME_ATTRS = {'time_nominal': 2015.0, 'time_start': 2007.0,
                        'time_end': 2020.0}
# The BedMachine ice mask is drawn from the same GIMP imagery / Landsat
# 2015-era classification; it sets the extent-misfit epoch.
MASK_TIME_ATTRS = {'time_nominal': 2015.0, 'time_start': 2007.0,
                   'time_end': 2019.0}

ICE_FRACTION_THRESHOLD = 0.5


def _regrid(da: xr.DataArray, template: xr.DataArray, resampling) -> xr.DataArray:
    if da.rio.crs is None:
        da = da.rio.write_crs(BEDMACHINE_CRS, inplace=True)
    out = da.rio.reproject_match(template, resampling=resampling)
    return out.assign_coords(x=template.x, y=template.y)


def _open_bedmachine(path: Path, grid) -> xr.Dataset:
    """BedMachine subset covering the target grid (with a margin), CRS set."""
    bm = xr.open_dataset(path)
    pad = 4 * grid.resolution + 1000.0
    xs = slice(grid.xmin - pad, grid.xmax + pad)
    # BedMachine y is descending.
    ys = slice(grid.ymax + pad, grid.ymin - pad)
    sub = bm.sel(x=xs, y=ys)
    if sub.sizes["x"] == 0 or sub.sizes["y"] == 0:
        raise ValueError("BedMachine does not intersect the domain grid")
    sub = sub.load()
    # The file's `mapping` grid-mapping variable is a data variable that
    # rioxarray cannot resolve from the per-variable `grid_mapping` attrs;
    # drop both and attach the (known) CRS explicitly to every variable.
    sub = sub.drop_vars('mapping', errors='ignore')
    for v in sub.data_vars:
        sub[v].attrs.pop('grid_mapping', None)
    sub = sub.rio.write_crs(BEDMACHINE_CRS, inplace=True)
    return sub


def _rasterize_labels(gdf, grid, start_label: int):
    """Integer label raster (fill -1) for `gdf` (already in grid CRS),
    labels start_label..start_label+n-1 in row order."""
    shapes = ((geom, start_label + k) for k, geom in enumerate(gdf.geometry)
              if geom is not None and not geom.is_empty)
    return rasterio.features.rasterize(
        shapes, out_shape=(grid.height, grid.width), transform=grid.transform,
        fill=-1, dtype='int32')


def build_dem(domain_path: str, bedmachine_path: str = None,
              basins_path: str = None, rgi_path: str = None,
              arcticdem_path: str = None, surface_source: str = "auto",
              smooth_sigma_km: float = 1.0) -> xr.Dataset:
    """Build the gridded geometry dataset for `domain_path` and write it.

    `surface_source`: "bedmachine", "arcticdem", or "auto" (ArcticDEM overlay
    if the file exists, else BedMachine).

    `smooth_sigma_km`: Gaussian smoothing (sigma in km; 0 = none) applied to
    the NATIVE BedMachine surface, bed and thickness (150 m) and to the
    ArcticDEM overlay (100 m, NaN-aware) BEFORE the area-average resampling
    to the model grid. At native resolution the ice / rock / water
    boundaries in steep topography are resolved, so a 1 km filter there is
    a genuine anti-aliasing filter for what the 1 km grid can carry --
    nothing below 1 km is resolved by the model anyway, so no spurious
    flotation or steep topography is lost that would not have been lost.
    The same linear kernel on surface, bed and thickness keeps S - B = H.
    Why: A_glen 1e-16 (soft ice) exposed solver aliasing at the ice-free /
    ice boundaries of extremely steep terrain that the stiff rheology had
    smoothed through the physics (2026-09-25); a post-resampling, mask-aware
    smoothing of the 1 km surface cannot reach that (the boundary is where
    the mask is). `errbed`, the radar-pick subcell averages and the mask
    fractions are not smoothed.
    """
    domain_path = Path(domain_path)
    output_path = domain_path / 'model_inputs' / 'gridded_dem.nc'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bedmachine_path = Path(bedmachine_path) if bedmachine_path else BEDMACHINE_PATH
    basins_path = Path(basins_path) if basins_path else BASINS_PATH
    rgi_path = Path(rgi_path) if rgi_path else RGI_PATH
    arcticdem_path = Path(arcticdem_path) if arcticdem_path else ARCTICDEM_PATH

    grid = load_domain_grid(domain_path)
    print(f"Target grid: {grid.describe()}")
    template = grid.template()

    bm = _open_bedmachine(bedmachine_path, grid)
    avg = Resampling.average
    surface_time = dict(BEDMACHINE_SURFACE_TIME_ATTRS)
    if 'nominal_year' in bm.attrs:
        surface_time['time_nominal'] = float(bm.attrs['nominal_year'])
    bm_version = bm.attrs.get('product_version', bm.attrs.get('title', 'BedMachine'))

    native_dx = float(abs(bm.x.values[1] - bm.x.values[0]))
    smoothing = 'none'
    if smooth_sigma_km and smooth_sigma_km > 0:
        from scipy import ndimage
        sig = smooth_sigma_km * 1e3 / native_dx
        for v in ('surface', 'bed', 'thickness'):
            arr = bm[v].values.astype('float32')
            bm[v] = bm[v].copy(data=ndimage.gaussian_filter(arr, sig, mode='nearest'))
        smoothing = (f"gaussian sigma {smooth_sigma_km:g} km on the native BedMachine surface / bed / thickness "
                     f"({native_dx:g} m, {sig:.2f} px) before resampling")
        print(f"native BedMachine surface / bed / thickness smoothed: sigma {smooth_sigma_km:g} km = {sig:.2f} px at {native_dx:g} m")
    surface = _regrid(bm.surface.astype('float32'), template, avg)
    bed = _regrid(bm.bed.astype('float32'), template, avg)
    thickness = _regrid(bm.thickness.astype('float32'), template, avg)
    errbed = _regrid(bm.errbed.astype('float32'), template, avg)

    # Radar-constrained subset: BedMachine `dataid == 2` marks the 150 m
    # cells that hold an actual radar pick (~4% of the ice; the rest is
    # kriging / mass conservation / IceBoost). Per model cell: the fraction
    # of radar subcells, and the bed / errbed averaged over those subcells
    # only, so a cell crossed by a track carries the pick, not a blend with
    # its interpolated neighbours. The conditioned bed prior can then use
    # only these (BedConditioningConfig.gridded_bed_data = "radar").
    if 'dataid' in bm:
        radar = (bm['dataid'] == 2).astype('float32').rio.write_crs(BEDMACHINE_CRS, inplace=True)
        radar_frac = _regrid(radar, template, avg).clip(0.0, 1.0).values
        bed_r = _regrid((bm.bed.astype('float32') * radar).rio.write_crs(BEDMACHINE_CRS, inplace=True), template, avg).values
        err_r = _regrid((bm.errbed.astype('float32') * radar).rio.write_crs(BEDMACHINE_CRS, inplace=True), template, avg).values
        with np.errstate(invalid='ignore', divide='ignore'):
            bed_radar = np.where(radar_frac > 0, bed_r / radar_frac, np.nan).astype('float32')
            err_radar = np.where(radar_frac > 0, err_r / radar_frac, np.nan).astype('float32')
    else:
        radar_frac = np.zeros(surface.shape, dtype='float32')
        bed_radar = np.full(surface.shape, np.nan, dtype='float32')
        err_radar = np.full(surface.shape, np.nan, dtype='float32')

    mask_native = bm['mask'].astype('int16')
    frac = {}
    for code, name in ((BM_OCEAN, 'ocean'), (BM_LAND, 'land'),
                       (BM_GROUNDED, 'grounded'), (BM_FLOATING, 'floating'),
                       (BM_NONGREENLAND, 'nongreenland')):
        ind = (mask_native == code).astype('float32')
        ind = ind.rio.write_crs(BEDMACHINE_CRS, inplace=True)
        frac[name] = _regrid(ind, template, avg).clip(0.0, 1.0)
    ice_fraction = (frac['grounded'] + frac['floating']).clip(0.0, 1.0)
    ice = ice_fraction.values > ICE_FRACTION_THRESHOLD
    floating = (frac['floating'].values > frac['grounded'].values) & ice
    ocean = (frac['ocean'].values > 0.5)

    # Composite elevation: ice surface over ice, bed elsewhere (land
    # topography; IBCAO/multibeam bathymetry in fjords and on the shelf).
    elevation = np.where(ice, surface.values, bed.values).astype('float32')
    # Cells the BedMachine raster does not cover (the ISMIP grid reaches
    # west of -653 km and north of -633 km): open ocean at sea level, kept
    # out of the model domain so no term ever sees a NaN.
    uncovered = ~np.isfinite(elevation)
    if uncovered.any():
        print(f"{int(uncovered.sum())} cells outside the BedMachine raster -> sea level, out of domain")
        elevation = np.where(uncovered, 0.0, elevation).astype('float32')
        ocean = ocean | uncovered
        ice = ice & ~uncovered
        floating = floating & ~uncovered

    # Optional dated surface overlay (ArcticDEM v4.1 mosaic, EPSG:3413).
    surface_attrs = dict(surface_time,
                         source=f"BedMachine Greenland {bm_version} surface (GIMP DEM v2.1)")
    use_arctic = surface_source == "arcticdem" or (
        surface_source == "auto" and arcticdem_path.exists())
    if use_arctic:
        if not arcticdem_path.exists():
            raise FileNotFoundError(arcticdem_path)
        adem = rioxarray.open_rasterio(arcticdem_path, masked=True).squeeze('band', drop=True)
        adem = adem.rio.clip_box(grid.xmin - 2000, grid.ymin - 2000,
                                 grid.xmax + 2000, grid.ymax + 2000)
        if smooth_sigma_km and smooth_sigma_km > 0:
            # the same filter at ArcticDEM's native resolution, NaN-aware
            # (normalized convolution over the voids)
            from scipy import ndimage
            adx = float(abs(adem.x.values[1] - adem.x.values[0]))
            sig_a = smooth_sigma_km * 1e3 / adx
            vals = adem.values.astype('float32'); good = np.isfinite(vals)
            num = ndimage.gaussian_filter(np.where(good, vals, 0.0).astype('float32'), sig_a, mode='nearest')
            den = ndimage.gaussian_filter(good.astype('float32'), sig_a, mode='nearest')
            sm = np.where(den > 0.5, num / np.maximum(den, 1e-6), np.nan).astype('float32')
            adem = adem.copy(data=np.where(good, sm, np.nan).astype('float32'))
            smoothing += f"; ArcticDEM overlay smoothed at {adx:g} m ({sig_a:.1f} px, NaN-aware)"
            print(f"ArcticDEM overlay smoothed: sigma {smooth_sigma_km:g} km = {sig_a:.1f} px at {adx:g} m")
            del vals, good, num, den, sm
        adem = _regrid(adem.astype('float32'), template, avg)
        # ArcticDEM heights are ellipsoidal (WGS84); BedMachine is geoid
        # referenced -> subtract BedMachine's geoid if present.
        if 'geoid' in bm:
            geoid = _regrid(bm.geoid.astype('float32'), template, avg).values
            adem = adem - geoid
        good = np.isfinite(adem.values) & ice
        n_fill = int((~good & ice).sum())
        elevation = np.where(good, adem.values, elevation).astype('float32')
        surface_attrs = dict(ARCTICDEM_TIME_ATTRS,
                             source="ArcticDEM v4.1 mosaic over ice (geoid-corrected), "
                                    "BedMachine elsewhere")
        if n_fill:
            print(f"ArcticDEM overlay: {n_fill} ice cells without data keep BedMachine surface")

    surface_attrs['smoothing'] = smoothing

    # Domain mask: everything in the grid except non-Greenland ice
    # (Ellesmere/Canadian Arctic edge), further clipped to an outline if given.
    domain_mask = ~(frac['nongreenland'].values > 0.5) & ~uncovered
    polygon = grid.outline_polygon()
    if polygon is not None:
        clip = template.rio.clip([polygon], crs="EPSG:4326", drop=False).notnull().values
        domain_mask &= clip

    ds = xr.Dataset(coords={"y": template.y, "x": template.x})
    ds['topography'] = (('y', 'x'), np.where(ocean, np.nan, elevation).astype('float32'))
    ds['bathymetry'] = (('y', 'x'), np.where(ocean, np.where(uncovered, 0.0, bed.values), np.nan).astype('float32'))
    ds['elevation'] = (('y', 'x'), elevation)
    ds['bathymetry_mask'] = (('y', 'x'), ocean)
    ds['domain_mask'] = (('y', 'x'), domain_mask)
    ds['rgi_mask'] = (('y', 'x'), ice)
    ds['ice_fraction'] = (('y', 'x'), ice_fraction.values.astype('float32'))
    ds['floating_mask'] = (('y', 'x'), floating)
    ds['thickness_obs'] = (('y', 'x'), thickness.values.astype('float32'))
    ds['bed_obs'] = (('y', 'x'), np.where(ice, bed.values, np.nan).astype('float32'))
    ds['bed_obs_err'] = (('y', 'x'), np.where(ice, errbed.values, np.nan).astype('float32'))
    ds['bed_radar_fraction'] = (('y', 'x'), np.where(ice, radar_frac, 0.0).astype('float32'))
    ds['bed_obs_radar'] = (('y', 'x'), np.where(ice, bed_radar, np.nan).astype('float32'))
    ds['bed_obs_radar_err'] = (('y', 'x'), np.where(ice, err_radar, np.nan).astype('float32'))
    ds['bed_radar_fraction'].attrs.update(units='1', long_name='fraction of BedMachine subcells holding a radar pick (dataid == 2)')
    ds['bed_obs_radar'].attrs.update(units='m', long_name='BedMachine bed averaged over radar subcells only (NaN where none)')
    ds['bed_obs_radar_err'].attrs.update(units='m', long_name='BedMachine errbed averaged over radar subcells only')
    for name in ('bathymetry_mask', 'domain_mask', 'rgi_mask', 'floating_mask'):
        ds[name].attrs['_FillValue'] = False

    ds['elevation'].attrs.update(surface_attrs, units='m',
                                 long_name='surface elevation over ice, bed/bathymetry elsewhere')
    ds['topography'].attrs.update(surface_attrs, units='m')
    ds['bathymetry'].attrs.update(units='m', source=f'BedMachine {bm_version} bed (IBCAO/multibeam)')
    ds['thickness_obs'].attrs.update(units='m', source=f'BedMachine {bm_version} thickness',
                                     **MASK_TIME_ATTRS)
    ds['bed_obs'].attrs.update(units='m', long_name='BedMachine bed elevation (on ice)',
                               source=f'BedMachine {bm_version} bed')
    ds['bed_obs_err'].attrs.update(units='m', long_name='BedMachine bed error (1 sigma, errbed)',
                                   source=f'BedMachine {bm_version} errbed')
    # v6 flags cells inside an RGI 7.0 outline (peripheral glaciers/ice caps).
    if 'rgi' in bm:
        rgi_ind = (bm['rgi'] == 1).astype('float32').rio.write_crs(BEDMACHINE_CRS, inplace=True)
        ds['rgi_periphery_fraction'] = (('y', 'x'), _regrid(rgi_ind, template, avg).clip(0, 1).values.astype('float32'))
        ds['rgi_periphery_fraction'].attrs.update(units='1', long_name='BedMachine v6 RGI-7.0 outline area fraction')
    ds['ice_fraction'].attrs.update(units='1', long_name='BedMachine ice-class area fraction',
                                    **MASK_TIME_ATTRS)
    for name in ('rgi_mask', 'floating_mask'):
        ds[name].attrs.update(MASK_TIME_ATTRS)

    # ----------------------------------------------------------- basin labels
    labels = np.full((grid.height, grid.width), -1, dtype='int32')
    ids, surge = [], []
    if basins_path.exists():
        basins = geopandas.read_file(basins_path, bbox=grid.bounds).to_crs(grid.crs)
        basins = basins.reset_index(drop=True)
        name_col = next((c for c in ('NAME', 'name', 'Name') if c in basins), None)
        lab = _rasterize_labels(basins, grid, 0)
        labels = np.where(lab >= 0, lab, labels)
        ids += [str(v) for v in (basins[name_col] if name_col else basins.index)]
        surge += [0] * len(basins)
        print(f"Rasterized {len(basins)} ice-sheet basins")
    else:
        warnings.warn(f"{basins_path} not found: ice-sheet basins get a single label")
    if rgi_path.exists():
        bbox_ll = geopandas.GeoSeries([__import__('shapely').geometry.box(*grid.bounds)],
                                      crs=grid.crs).to_crs(4326).total_bounds
        rgi = geopandas.read_file(rgi_path, bbox=tuple(bbox_ll)).to_crs(grid.crs)
        # RGI 6.0 (RGIId / Surging 0-3, 9 = unassigned / Connect 0-2) or
        # RGI 7.0 (rgi_id / surge_type). Glaciers dynamically connected to
        # the ice sheet (Connect == 2) stay with their drainage basin label.
        if 'Connect' in rgi:
            rgi = rgi[rgi['Connect'] < 2]
        rgi = rgi.reset_index(drop=True)
        id_col = 'rgi_id' if 'rgi_id' in rgi else 'RGIId'
        surge_col = 'surge_type' if 'surge_type' in rgi else 'Surging'
        lab = _rasterize_labels(rgi, grid, len(ids))
        labels = np.where(lab >= 0, lab, labels)
        ids += [str(v) for v in rgi[id_col]]
        s = rgi[surge_col].fillna(0).astype(int).to_numpy() if surge_col in rgi else np.zeros(len(rgi), int)
        surge += [0 if v == 9 else int(v) for v in s]
        print(f"Rasterized {len(rgi)} RGI peripheral glaciers ({id_col})")
    if not ids:
        ids, surge = ["greenland_ice_sheet"], [0]
        labels = np.where(ice, 0, -1).astype('int32')
    # Ice pixels outside every polygon join the nearest labelled neighbour
    # would be ideal; keep them unlabeled (-1: standard likelihood, no
    # per-glacier marginal) and report how many there are.
    n_unlabeled = int((ice & (labels < 0)).sum())
    if n_unlabeled:
        print(f"{n_unlabeled} ice cells fall outside every basin/glacier polygon (label -1)")
    ds['rgi_label'] = (('y', 'x'), np.where(domain_mask, labels, -1).astype('int32'))
    ds['rgi_label'].attrs['_FillValue'] = -1
    ds['rgi_label'].attrs.update(MASK_TIME_ATTRS, long_name='basin / glacier integer label')
    ds['rgi_id'] = xr.DataArray(np.asarray(ids).astype('U'), dims=('glacier',),
                                coords={'glacier': np.arange(len(ids), dtype='int32')})
    ds['surge_type'] = xr.DataArray(np.asarray(surge, dtype='int32'), dims=('glacier',),
                                    coords={'glacier': np.arange(len(ids), dtype='int32')})
    ds['surge_type'].attrs.update(MASK_TIME_ATTRS)

    ds = write_grid_metadata(ds, grid)
    ds.attrs.update(geometry_source=str(bedmachine_path), title="GLIDE Greenland geometry")
    ds['x'] = ds['x'].astype('float32')
    ds['y'] = ds['y'].astype('float32')
    ds.to_netcdf(output_path)
    print(f"wrote {output_path}: ice cells {int(ice.sum())}, floating {int(floating.sum())}, "
          f"radar-constrained {int((ice & (radar_frac > 0)).sum())}")
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-path", type=str, required=True)
    parser.add_argument("--bedmachine", type=str, default=None)
    parser.add_argument("--basins", type=str, default=None)
    parser.add_argument("--rgi", type=str, default=None)
    parser.add_argument("--arcticdem", type=str, default=None)
    parser.add_argument("--surface-source", choices=("auto", "bedmachine", "arcticdem"),
                        default="auto")
    parser.add_argument("--smooth-sigma-km", type=float, default=1.0,
                        help="Gaussian smoothing of the native BedMachine surface / bed / thickness and the ArcticDEM overlay "
                             "before resampling, sigma in km (0 = none)")
    args = parser.parse_args()
    build_dem(args.domain_path, args.bedmachine, args.basins, args.rgi,
              args.arcticdem, args.surface_source, args.smooth_sigma_km)
