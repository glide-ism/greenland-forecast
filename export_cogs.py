"""
Export a domain's inverse solution as cloud-optimized GeoTIFFs.

Reads level_<n>/inverse_soln.nc (bed, thickness, SMB — already physical and
georeferenced in EPSG:3413) and level_<n>/torch_vars.p (whitened precipitation
bias, de-whitened here through the pbias Matern prior — requires the GPU) and
writes COGs to <output_dir>/cog/:

    <domain>_bed.tif             bed elevation [m]
    <domain>_thickness_2020.tif  ice thickness at the run horizon [m]
    <domain>_smb_2020.tif        final-step surface mass balance [m a^-1]
    <domain>_pbias.tif           multiplicative precip factor exp(pbias) [-]
    <domain>_t2m_<01..12>.tif    monthly mean 2-m air temperature forcing [deg C]
    <domain>_precip_<01..12>.tif monthly precipitation forcing [m ice eq. a^-1]
    <domain>_t2m_annual.tif      annual (12-month) mean of the above
    <domain>_precip_annual.tif   annual (12-month) mean of the above

The monthly climate rasters are the raw model inputs (the climatology the SMB
model is forced with, before the inferred pbias/anomaly corrections), exported
straight from the cropped fine grid.

The run horizon is 2020 for all current domains (the dhdt window end), so the
saved final state is "c. 2020". SMB cells outside the model domain carry a -10
sentinel and are written as nodata. The pbias field is always exported on the
fine (level-0) grid — the whitened parameters are fine-grid regardless of the
level directory they were saved in.

Usage:
    python export_cogs.py --domain-path domains/juneau [--level 0] [--outdir DIR]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import rioxarray  # noqa: F401  (registers the .rio accessor)

from ggapp.torch import GGaPPMap

from glacier_inverse import load_config
from glacier_inverse.priors import _build_matern_prior, _cropped_inputs, domain_shape


def write_cog(values, x, y, crs, path, units, long_name, tags=None):
    da = xr.DataArray(np.asarray(values, dtype=np.float32),
                      coords={"y": np.asarray(y), "x": np.asarray(x)},
                      dims=("y", "x"))
    da = da.rio.write_crs(crs)
    da = da.rio.write_nodata(np.nan)
    da.rio.to_raster(path, driver="COG", compress="DEFLATE",
                     tags={"units": units, "long_name": long_name, **(tags or {})})
    print(f"wrote {path}")


def fine_grid_coords(config, level_dir_parent, min_level):
    """(x, y, crs_wkt) of the fine model grid, preferring the finest saved
    solution's coordinates and falling back to the cropped model inputs."""
    soln = level_dir_parent / f"level_{min_level}" / "inverse_soln.nc"
    if soln.exists() and min_level == 0:
        with xr.open_dataset(soln) as ds:
            return (ds.x_cell.values.copy(), ds.y_cell.values.copy(),
                    ds.bed.attrs["crs_wkt"])
    gd = _cropped_inputs(config, variables=["elevation", "spatial_ref"])
    return gd.x.values, gd.y.values, gd.spatial_ref.attrs["crs_wkt"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--domain-path", required=True,
                        help="domain directory, e.g. domains/juneau")
    parser.add_argument("--level", type=int, default=0,
                        help="multigrid level to export (0 = finest)")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="output directory (default: <output_dir>/cog)")
    parser.add_argument("--results-subdir", default=None,
                        help="override config.results_subdir (e.g. to export a "
                             "past experiment without editing the domain config)")
    args = parser.parse_args()

    config = load_config(args.domain_path)
    domain = Path(args.domain_path).name
    results_dir = (Path(config.base_dir) / args.results_subdir
                   if args.results_subdir else Path(config.output_dir))
    level_dir = results_dir / f"level_{args.level}"
    outdir = args.outdir or results_dir / "cog"
    outdir.mkdir(parents=True, exist_ok=True)

    # --- bed, thickness, smb: already physical and georeferenced in the nc ---
    with xr.open_dataset(level_dir / "inverse_soln.nc") as ds:
        x = ds.x_cell.values.copy()
        y = ds.y_cell.values.copy()
        crs = ds.bed.attrs["crs_wkt"]
        bed = ds.bed.values.copy()
        H = ds.H.values.copy()
        smb = ds.smb.values.copy()

    # Off-domain SMB carries a -10 sentinel. Real ablation-zone SMB can be
    # below -10 m/yr, so mask with the actual domain mask rather than the
    # value; at coarse levels any block touching off-domain cells is a
    # contaminated average, so require the whole block on-domain.
    gd = _cropped_inputs(config, variables=["domain_mask",
                                            "monthly_t2m", "monthly_precip"])
    mask = gd.domain_mask.values.astype(bool)
    if args.level > 0:
        f = 2 ** args.level
        mask = mask.reshape(mask.shape[0] // f, f, mask.shape[1] // f, f).all(axis=(1, 3))
    smb[~mask] = np.nan

    epoch = {"time_nominal": "2020"}
    write_cog(bed, x, y, crs, outdir / f"{domain}_bed.tif",
              "m", "modelled bed elevation")
    write_cog(H, x, y, crs, outdir / f"{domain}_thickness_2020.tif",
              "m", "modelled ice thickness", tags=epoch)
    write_cog(smb, x, y, crs, outdir / f"{domain}_smb_2020.tif",
              "m a^-1", "modelled surface mass balance (final step)", tags=epoch)

    # --- precip bias: de-whiten z through the pbias Matern prior (GPU) ---
    d = torch.load(level_dir / "torch_vars.p", map_location="cpu",
                   weights_only=False)
    z = d["precipitation_bias"].detach()
    ny, nx, dx = domain_shape(config)
    if z.shape != (ny, nx):
        raise ValueError(f"whitened pbias shape {tuple(z.shape)} != fine grid "
                         f"({ny}, {nx}) from model inputs")
    model = _build_matern_prior(config.pbias_prior, config.n_levels, ny, nx, dx)
    pbias = GGaPPMap.apply(model, z.cuda()).cpu().numpy()

    xf, yf, crsf = fine_grid_coords(config, results_dir, 0)
    write_cog(np.exp(pbias), xf, yf, crsf, outdir / f"{domain}_pbias.tif",
              "1", "multiplicative precipitation bias exp(pbias)")

    # --- raw monthly climate forcing: exported as-is from the fine grid ---
    for var, stem, units in ((gd.monthly_t2m, "t2m", "deg C"),
                             (gd.monthly_precip, "precip", "m ice eq. a^-1")):
        tags = {k: str(var.attrs[k]) for k in ("source",) if k in var.attrs}
        for m in range(12):
            write_cog(var.values[m], xf, yf, crsf,
                      outdir / f"{domain}_{stem}_{m + 1:02d}.tif",
                      units, f"{var.attrs.get('long_name', stem)}, month {m + 1:02d}",
                      tags={"month": f"{m + 1:02d}", **tags})
        write_cog(var.values.mean(axis=0), xf, yf, crsf,
                  outdir / f"{domain}_{stem}_annual.tif",
                  units, f"{var.attrs.get('long_name', stem)}, annual mean",
                  tags=tags)


if __name__ == "__main__":
    main()
