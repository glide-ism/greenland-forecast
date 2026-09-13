"""
Export a domain's inverse solution as an ISMIP-style initial-state file.

Reads level_<n>/inverse_soln.nc (u, v, ud, vd, H, bed, beta, smb on the
model grid, physical units, georeferenced) and writes one NetCDF with the
ISMIP6 variable names / units that the ISMIP7 request inherits (Nowicki et
al. 2020; the ISMIP7 variable request splits combined fluxes and adds a few
fields — check the current request in github.com/ismip before submission):

    lithk      land_ice_thickness                        m
    orog       surface_altitude                          m
    topg       bedrock_altitude                          m
    base       base_altitude                             m
    xvelmean / yvelmean   depth-averaged velocity        m s-1
    xvelsurf / yvelsurf   surface velocity (u + ud/(n+1)) m s-1
    xvelbase / yvelbase   basal velocity (u - ud)         m s-1
    strbasemag magnitude of basal drag                   Pa
    acabf      surface mass balance flux                 kg m-2 s-1
    sftgif / sftgrf / sftflf   ice / grounded / floating area fraction  1
    beta       glide basal traction coefficient (extra, model native)

Time is "days since 1850-01-01" at the requested nominal date. When the
domain uses the `ismip_greenland` preset the model grid IS the ISMIP grid at
its resolution; `--regrid-to` block-averages onto a coarser ISMIP resolution
(2/4/8 km) conservatively. Other domains are written on their native grid.

Usage:
    python export_ismip.py --domain-path domains/greenland [--level 0]
                           [--date 2015.0] [--regrid-to 4000] [--out FILE]
"""
import argparse
import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from glacier_inverse import load_config
from glacier_inverse.priors import _cropped_inputs

SECONDS_PER_YEAR = 365.25 * 24 * 3600.0
ISMIP_EPOCH = datetime.datetime(1850, 1, 1)


def _days_since_1850(decimal_year: float) -> float:
    year = int(np.floor(decimal_year))
    y0 = datetime.datetime(year, 1, 1)
    y1 = datetime.datetime(year + 1, 1, 1)
    d = y0 + (y1 - y0) * (decimal_year - year)
    return (d - ISMIP_EPOCH).total_seconds() / 86400.0


def _cell(u_facet, axis):
    """Facet -> cell-centre average along `axis` (u on vertical facets ->
    axis 1, v on horizontal facets -> axis 0)."""
    if axis == 1:
        return 0.5 * (u_facet[:, 1:] + u_facet[:, :-1])
    return 0.5 * (u_facet[1:, :] + u_facet[:-1, :])


def _block(a, f):
    ny, nx = a.shape[0] // f * f, a.shape[1] // f * f
    return a[:ny, :nx].reshape(ny // f, f, nx // f, f).mean(axis=(1, 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--domain-path", required=True)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--date", type=float, default=None,
                    help="nominal decimal year of the state (default config.t_end)")
    ap.add_argument("--regrid-to", type=float, default=None,
                    help="coarser ISMIP resolution in m (block average)")
    ap.add_argument("--results-subdir", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    config = load_config(args.domain_path)
    results = (Path(config.base_dir) / args.results_subdir if args.results_subdir
               else Path(config.output_dir))
    level_dir = results / f"level_{args.level}"
    date = args.date if args.date is not None else float(config.t_end)
    n = float(config.n_glen)
    r = config.rho_ice / config.rho_water

    with xr.open_dataset(level_dir / "inverse_soln.nc") as ds:
        x = ds.x_cell.values.copy()
        y = ds.y_cell.values.copy()
        crs_wkt = ds.bed.attrs.get("crs_wkt", "")
        H = ds.H.values.astype("float64")
        bed = ds.bed.values.astype("float64")
        beta = ds.beta.values.astype("float64")
        smb = ds.smb.values.astype("float64")
        u, v = ds.u.values.astype("float64"), ds.v.values.astype("float64")
        ud, vd = ds.ud.values.astype("float64"), ds.vd.values.astype("float64")

    gd = _cropped_inputs(config, variables=["domain_mask"])
    dom = gd.domain_mask.values.astype(bool)
    f0 = 2 ** args.level
    if f0 > 1:
        dom = dom.reshape(dom.shape[0] // f0, f0, dom.shape[1] // f0, f0).all(axis=(1, 3))

    ice = H > config.init_H_floor + 1e-3
    surface = np.maximum(bed + H, (1.0 - r) * H)
    base = surface - H
    grounded = ice & (base <= bed + 1e-3)
    floating = ice & ~grounded

    um, vm = _cell(u, 1), _cell(v, 0)
    us, vs = _cell(u + ud / (n + 1.0), 1), _cell(v + vd / (n + 1.0), 0)
    ub, vb = _cell(u - ud, 1), _cell(v - vd, 0)
    # glide's drag (stress.cu): beta * (|u_b|^2 + u_reg)^((m-1)/2) * u_b in
    # head units (rho g folded into beta), so |tau_b| in Pa is rho g times that.
    m = float(config.sliding_m)
    ub2 = ub ** 2 + vb ** 2
    tau_b = (config.rho_ice * config.gravity * beta
             * (ub2 + float(config.u_reg)) ** ((m - 1.0) / 2.0) * np.sqrt(ub2))  # Pa

    fields = {
        "lithk": (np.where(ice, H, 0.0), "m", "land_ice_thickness"),
        "orog": (np.where(ice, surface, np.maximum(bed, 0.0)), "m", "surface_altitude"),
        "topg": (bed, "m", "bedrock_altitude"),
        "base": (np.where(ice, base, bed), "m", "base_altitude"),
        "xvelmean": (um / SECONDS_PER_YEAR, "m s-1", "land_ice_vertical_mean_x_velocity"),
        "yvelmean": (vm / SECONDS_PER_YEAR, "m s-1", "land_ice_vertical_mean_y_velocity"),
        "xvelsurf": (us / SECONDS_PER_YEAR, "m s-1", "land_ice_surface_x_velocity"),
        "yvelsurf": (vs / SECONDS_PER_YEAR, "m s-1", "land_ice_surface_y_velocity"),
        "xvelbase": (ub / SECONDS_PER_YEAR, "m s-1", "land_ice_basal_x_velocity"),
        "yvelbase": (vb / SECONDS_PER_YEAR, "m s-1", "land_ice_basal_y_velocity"),
        "strbasemag": (np.where(grounded, tau_b, 0.0), "Pa", "land_ice_basal_drag"),
        "acabf": (np.where(dom, smb * config.rho_ice / SECONDS_PER_YEAR, np.nan),
                  "kg m-2 s-1", "land_ice_surface_specific_mass_balance_flux"),
        "sftgif": (ice.astype(float), "1", "land_ice_area_fraction"),
        "sftgrf": (grounded.astype(float), "1", "grounded_ice_sheet_area_fraction"),
        "sftflf": (floating.astype(float), "1", "floating_ice_shelf_area_fraction"),
        "beta": (beta, "model units", "glide basal traction coefficient"),
    }

    if args.regrid_to:
        dx = float(abs(x[1] - x[0]))
        f = int(round(args.regrid_to / dx))
        if f < 1 or abs(f * dx - args.regrid_to) > 1e-6:
            raise ValueError(f"--regrid-to {args.regrid_to} is not a multiple of dx={dx}")
        if f > 1:
            fields = {k: (_block(np.nan_to_num(a), f), u_, ln) for k, (a, u_, ln) in fields.items()}
            x = _block(x[None, :].repeat(f, 0), f)[0]
            y = _block(y[:, None].repeat(f, 1), f)[:, 0]

    t = np.array([_days_since_1850(date)])
    out = xr.Dataset(coords={"time": ("time", t), "y": y, "x": x})
    out["time"].attrs.update(units="days since 1850-01-01", calendar="standard")
    for name, (a, units, std) in fields.items():
        out[name] = (("time", "y", "x"), a[None].astype("float32"))
        out[name].attrs.update(units=units, standard_name=std)
    out["x"].attrs.update(units="m", standard_name="projection_x_coordinate")
    out["y"].attrs.update(units="m", standard_name="projection_y_coordinate")
    out.attrs.update(title="GLIDE Greenland initial state", crs_wkt=crs_wkt,
                     source=f"{args.domain_path} {results.name} level_{args.level}",
                     nominal_date=date, comment="ISMIP6-style variable names; check "
                     "the ISMIP7 data request before submission")
    out_path = args.out or results / f"ismip_state_{date:.0f}_level{args.level}.nc"
    enc = {k: {"_FillValue": np.float32(np.nan), "zlib": True} for k in fields}
    out.to_netcdf(out_path, encoding=enc)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
