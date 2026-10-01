"""
ISMIP7 export of a forward run (forward_projection.py / forward_standalone.py).

Reads the run's YEARLY frames -- `<run-dir>/series.nc` from tools/vti_to_nc.py
when present, else the VTI series `<run-dir>/vti/*.pvd` (H, smb, dhdt, mask,
phi, psi, xi, the depth-mean velocity U and the basal velocity U_b, plus the
static bed and the beta the run used) -- and writes one NetCDF per variable on
the ISMIP Greenland grid, streaming one model year at a time through every
output file (each 290 MB frame is read once, not once per variable).

    python ismip_exporter.py --run-dir domains/greenland/inverse/projection_CESM2-WACCM_ssp585 \\
        --experiment ssp585                       # nominal 2015-2300 (the run must reach t = 2301)
    python ismip_exporter.py --run-dir ... --experiment historical      # 1850-2014
    python ismip_exporter.py --run-dir ... --experiment ssp585 --years 2015 2020 --resolution 4000

Files land in <submission-dir>/Models/GrIS/<group>/<model>/CORE/<set-counter>/,
the layout the ISMIP7 compliance checker (ISM_SimulationChecker) takes as
--source-path, with a not_modelled.txt declaring the optional variables the
model lacks. One seamless 1800-2301 run serves both the historical and the
scenario export (`--years` selects the window; the defaults follow the
experiment). Verified against the checker 2026-09-18 (4 km, historical
1850-2014 and ssp585): no errors, no warnings except 19 cells of acabf below
-6e-4 kg m-2 s-1 and the missing nominal year 2300 of a run that stopped at
t = 2300.

Time convention (ISMIP): model year Y is the step (Y, Y+1]; its frame carries
time Y+1. State variables (ST) are that frame, stamped YYYY+1-01-01; flux
variables (FL) are the step's rates, stamped mid-year with time_bnds. Years
whose frame is not preceded by a frame one year earlier (the 5-yr steps before
the forcing record) are refused.

What follows the model's physics, and differs from the first version of this
script (written against glide's example zarr and the pre-2026-09 model):

  * CALVING. glide's calving is no longer a front velocity: it is the cell
    sink (1 - psi) H / tau of the monotone gap-blended height-above-buoyancy
    law (common.cu `calving_F`, flux.cu `get_cell_calving_jac`), psi the
    calving flag and tau = config.calving_timescale. The implicit step removes
    exactly (1 - psi) H^{n+1} / tau per year, so
        licalvf = - rho_i (1 - psi) H / tau
    from the end-of-step frame IS the step's calving flux, cell by cell. The
    old u_c = 2000 m/yr face flux is gone, also from ligroundf.
  * FRONTAL MELT. The ocean thermal forcing acts through the calving margins
    (q, h0), not through a separate melt term: every marine loss is calving
    in this model. lifmassbf is therefore 0, and licalvf holds the total;
    the split ISMIP7 asks for does not exist here.
  * ligroundf is the upwinded H u flux through faces between grounded ice
    and floating ice or open water (bed < 0), charged to the grounded cell.
  * strbasemag follows stress.cu: rho_i g (beta xi^p K + water_drag) |u_b|,
    K = (|u_b|^2 + u_reg)^((m-1)/2) (u0 / (|u_b| + u0))^m (the regularized
    Coulomb factor, config.sliding_u0; added 2026-10-01 -- the exports before
    omitted it and overstated fast-ice drag, x2.2 at 3 km/yr for u0 300,
    m 1/3), with the BASAL velocity (MOLHO: u_b = u - u_d), the
    per-year flotation fraction xi (effective pressure) and the beta the run
    actually used (capped at BETA_MAX) -- not the mean velocity, phi of the
    first frame and the uncapped beta.
  * Sea water is config.rho_water (1028), not 1000: ice base, surface of
    floating ice and mass above flotation all depend on it.
  * Ice-free cells hold the thklim floor (1 m), not zero: thickness, mass and
    area use the active-set mask (ice = mask < 0.5), so the floor is not
    exported as a 1 m ice sheet over the whole domain.
  * y-velocity: glide's stencils use image orientation (the previous row is
    "top", v > 0 points to it), so on this north-up grid v is already the
    geographic northward component. No sign change.
  * The model grid is the ISMIP 1 km grid cropped to a multiple of 2^n_levels
    and stored north to south; it is padded back and flipped to ascending y.
    --resolution 2000/4000/8000/16000 coarsens CONSERVATIVELY onto the nested
    node-centred ISMIP grids (a coarse cell is centred on every f-th node:
    weights 1/2, 1, ..., 1, 1/2 across f + 1 fine cells).

Sign convention (the checker's ISMIP7_variable_request.csv ranges): licalvf
and lifmassbf are non-positive (loss), ligroundf is the flux of grounded ice
across the grounding line, positive; the scalar totals are magnitudes
(ranges [0, 1e25]; the checker does not range-check scalars). LOSS_SIGN
applies to the gridded loss fluxes only.
"""
import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr
from netCDF4 import Dataset, date2num
from scipy.ndimage import convolve1d

from glacier_inverse import load_config

# ------------------------------------------------------------------ settings
DOMAIN = "domains/greenland"
SECONDS_PER_YEAR = 31536000.0          # glare / the drivers (365 d)
FILL_VALUE = np.float32(9.96921e36)
LOSS_SIGN = -1.0                        # see the docstring: verify for ISMIP7
SLIDING_P = 1.0                         # glide's effective-pressure exponent (grid.py default)
TIME_UNITS, TIME_CALENDAR = "days since 1850-01-01", "standard"

GROUP, MODEL, REGION = "UMT", "GLIDE", "GrIS"        # file-name fields 3 and 4; group id: VERIFY with ISMIP7
ISM_MEMBER, FORCING_ID = "m001", "f001"
# nominal year windows of experiments_ismip7.csv (the checker's): historical
# starts anywhere in 1850-2014 (the run's first yearly frame here), the
# projections are pinned to 2015-end. The SET COUNTER (C001 ... C011) is one
# per (experiment, ESM) in the ISMIP7 protocol table (tools/ismip7_set_counters.py
# holds it; run_ismip7_core.py passes it), each in its own directory
# Models/GrIS/<group>/<model>/CORE/<set>/ (the checker's --source-path).
EXPERIMENTS = {"historical": (1850, 2014), "ssp370": (2015, 2100), "ssp126": (2015, 2300),
               "ssp585": (2015, 2300), "ctrl": (2015, 2300)}
# non-mandatory variables this model does not represent (-> not_modelled.txt)
NOT_MODELLED = {"hfgeoubed": "no thermal model: the enthalpy model is the surface only",
                "zvelsurf": "vertical velocities are not diagnosed", "zvelbase": "vertical velocities are not diagnosed",
                "litemptop": "isothermal ice", "litempavg": "isothermal ice", "litempbotgr": "isothermal ice",
                "litempbotfl": "isothermal ice", "litemp": "isothermal ice",
                "thdrflf": "the ocean forcing acts on the calving margin, not on a sub-shelf melt",
                "deltag": "no GIA / sea-level model", "refgeoid": "no GIA / sea-level model"}

# name -> (ST | FL, long_name, standard_name, units)
GRID_VARS = {
    "lithk": ("ST", "Ice thickness", "land_ice_thickness", "m"),
    "orog": ("ST", "Surface elevation", "surface_altitude", "m"),
    "topg": ("ST", "Bedrock elevation", "bedrock_altitude", "m"),
    "base": ("ST", "Ice base elevation", "base_altitude", "m"),
    "xvelmean": ("ST", "Mean velocity in x", "land_ice_vertical_mean_x_velocity", "m s-1"),
    "yvelmean": ("ST", "Mean velocity in y", "land_ice_vertical_mean_y_velocity", "m s-1"),
    "xvelsurf": ("ST", "Surface velocity in x", "land_ice_surface_x_velocity", "m s-1"),
    "yvelsurf": ("ST", "Surface velocity in y", "land_ice_surface_y_velocity", "m s-1"),
    "xvelbase": ("ST", "Basal velocity in x", "land_ice_basal_x_velocity", "m s-1"),
    "yvelbase": ("ST", "Basal velocity in y", "land_ice_basal_y_velocity", "m s-1"),
    "strbasemag": ("ST", "Basal drag", "land_ice_basal_drag", "Pa"),
    "sftgif": ("ST", "Land ice area fraction", "land_ice_area_fraction", "1"),
    "sftgrf": ("ST", "Grounded ice sheet area fraction", "grounded_ice_sheet_area_fraction", "1"),
    "sftflf": ("ST", "Floating ice sheet area fraction", "floating_ice_shelf_area_fraction", "1"),
    "acabf": ("FL", "Surface mass balance flux", "land_ice_surface_specific_mass_balance_flux", "kg m-2 s-1"),
    "libmassbfgr": ("FL", "Basal mass balance flux beneath grounded ice", "land_ice_basal_specific_mass_balance_flux", "kg m-2 s-1"),
    "libmassbffl": ("FL", "Basal mass balance flux beneath floating ice", "land_ice_basal_specific_mass_balance_flux", "kg m-2 s-1"),
    "dlithkdt": ("FL", "Ice thickness imbalance", "tendency_of_land_ice_thickness", "m s-1"),
    "licalvf": ("FL", "Calving flux", "land_ice_specific_mass_flux_due_to_calving", "kg m-2 s-1"),
    "lifmassbf": ("FL", "Ice front melt flux", "land_ice_specific_mass_flux_due_to_ice_front_melting", "kg m-2 s-1"),
    "ligroundf": ("FL", "Grounding line flux", "land_ice_specific_mass_flux_at_grounding_line", "kg m-2 s-1"),
}
# exported only when the run carries the enthalpy model's fields (config.thermal;
# frames with T_bed / T_mean / T_top); otherwise listed in not_modelled.txt
THERMAL_VARS = {
    "litemptop": ("ST", "Surface temperature", "temperature_at_top_of_ice_sheet_model", "K"),
    "litempavg": ("ST", "Depth average temperature", "land_ice_temperature", "K"),
    "litempbotgr": ("ST", "Basal temperature beneath grounded ice sheet", "temperature_at_base_of_ice_sheet_model", "K"),
    "litempbotfl": ("ST", "Basal temperature beneath floating ice shelf", "temperature_at_base_of_ice_sheet_model", "K"),
    "hfgeoubed": ("FL", "Geothermal heat flux", "upward_geothermal_heat_flux_in_land_ice", "W m-2"),
}
THERMAL_FRAME_FIELDS = ["T_bed", "T_mean", "T_top"]
SEA_SALINITY = 34.5                                    # psu, litempbotfl
FREEZE_L1, FREEZE_L2, FREEZE_L3 = -5.73e-2, 8.32e-2, -7.53e-4   # K/psu, K, K/m (Jenkins 2011)
SCALAR_VARS = {
    "lim": ("ST", "Total ice mass", "land_ice_mass", "kg"),
    "limnsw": ("ST", "Mass above floatation", "land_ice_mass_not_displacing_sea_water", "kg"),
    "iareagr": ("ST", "Grounded ice area", "grounded_ice_sheet_area", "m2"),
    "iareafl": ("ST", "Floating ice area", "floating_ice_shelf_area", "m2"),
    "tendacabf": ("FL", "Total SMB flux", "tendency_of_land_ice_mass_due_to_surface_mass_balance", "kg s-1"),
    "tendlibmassbfgr": ("FL", "Total BMB flux beneath grounded ice", "tendency_of_land_ice_mass_due_to_basal_mass_balance", "kg s-1"),
    "tendlibmassbffl": ("FL", "Total BMB flux beneath floating ice", "tendency_of_land_ice_mass_due_to_basal_mass_balance", "kg s-1"),
    "tendlicalvf": ("FL", "Total calving flux", "tendency_of_land_ice_mass_due_to_calving", "kg s-1"),
    "tendlifmassbf": ("FL", "Total ice front melting flux", "tendency_of_land_ice_mass_due_to_ice_front_melting", "kg s-1"),
    "tendligroundf": ("FL", "Total grounding line flux", "tendency_of_grounded_ice_mass", "kg s-1"),
}
ZERO_NOTE = {"libmassbfgr": "the model has no basal mass balance", "libmassbffl": "the model has no basal mass balance",
             "lifmassbf": "ocean forcing acts through the calving margin; all marine loss is reported as licalvf"}


# ----------------------------------------------------------------- VTI input
def read_vti(path, names):
    """Named Float32 arrays of a glide VTIWriter file -> (ny, nx) or
    (ny, nx, ncomp), in the model's row order (the writer stores south to
    north). Raw appended data (UInt32 length headers) or VTK's compressed
    appended layout (tools/vti_compress.py: LZ4 or zlib blocks)."""
    with open(path, "rb") as f:
        head = f.read(1 << 20)
        i = head.index(b"<AppendedData")
        j = head.index(b"_", i) + 1
        xml = head[:i].decode("utf-8", "ignore")
        ext = [int(v) for v in re.search(r'WholeExtent="([^"]+)"', xml).group(1).split()]
        nx, ny = ext[1] - ext[0] + 1, ext[3] - ext[2] + 1
        m = re.search(r'header_type="(\w+)"', xml)
        header = {"UInt32": np.uint32, "UInt64": np.uint64}[m.group(1)] if m else np.uint32
        hs = np.dtype(header).itemsize
        m = re.search(r'compressor="(\w+)"', xml)
        codec = None
        if m:
            if m.group(1) == "vtkLZ4DataCompressor":
                import lz4.block
                codec = lambda b, n: lz4.block.decompress(b, uncompressed_size=n)
            elif m.group(1) == "vtkZLibDataCompressor":
                import zlib
                codec = lambda b, n: zlib.decompress(b)
            else:
                raise ValueError(f"{path}: unsupported compressor {m.group(1)}")
        meta = {mm.group(1): (int(mm.group(2)), int(mm.group(3))) for mm in re.finditer(
            r'Name="(\w+)" NumberOfComponents="(\d+)" format="appended" offset="(\d+)"', xml)}
        out = {}
        for n in names:
            ncomp, off = meta[n]
            f.seek(j + off)
            if codec is None:
                nbytes = int(np.frombuffer(f.read(hs), header)[0])
                raw = f.read(nbytes)
            else:
                nblk, bsize, last = (int(v) for v in np.frombuffer(f.read(3 * hs), header))
                sizes = np.frombuffer(f.read(nblk * hs), header)
                raw = b"".join(codec(f.read(int(cs)), bsize if (k < nblk - 1 or last == 0) else last)
                               for k, cs in enumerate(sizes))
            a = np.frombuffer(raw, np.float32)
            a = a.reshape(ny, nx) if ncomp == 1 else a.reshape(ny, nx, ncomp)
            out[n] = a[::-1].copy()
    return out


def frame_index(run_dir):
    pvd = next((Path(run_dir) / "vti").glob("*.pvd"))
    items = re.findall(r'timestep="([\d.]+)"[^>]*file="([^"]+)"', pvd.read_text())
    return [(float(t), pvd.parent / fn) for t, fn in items], next(pvd.parent.glob("*_static.vti"))


class FrameSource:
    """Yearly frames of a run: from `<run-dir>/series.nc` (tools/vti_to_nc.py,
    preferred: velocities and SMB there are zero off the ice and rounded to
    physical precision) or from the VTI series. `times` are the frame times,
    `get(t, names)` returns VTI-named arrays (vectors as (ny, nx, 2)),
    `static()` the bed and beta."""
    VEC = {"U": ("u", "v"), "U_s": ("u_s", "v_s"), "U_b": ("u_b", "v_b")}

    def __init__(self, run_dir):
        run_dir = Path(run_dir)
        self.nc = None
        has_vti = (run_dir / "vti").exists() and any((run_dir / "vti").glob("*.pvd"))
        # the VTI frames (LZ4-compressed since 2026-09-17) are the complete
        # record; series.nc may have been written with fields dropped
        if (run_dir / "series.nc").exists() and not has_vti:
            import netCDF4
            self.nc = netCDF4.Dataset(run_dir / "series.nc")
            self.times = [float(v) for v in self.nc["model_year"][:]]
            self.kind = "series.nc"
        else:
            frames, self.static_path = frame_index(run_dir)
            self.times = [t for t, _ in frames]
            self.paths = {round(t, 6): p for t, p in frames}
            self.kind = "vti"
        self.index = {round(t, 6): k for k, t in enumerate(self.times)}

    def _nc(self, name, k=None):
        v = self.nc[name]
        a = v[k, :, :] if k is not None else v[:, :]
        return np.asarray(a.filled(np.nan) if np.ma.isMaskedArray(a) else a, dtype=np.float32)

    def get(self, t, names):
        k = self.index[round(t, 6)]
        if self.nc is None:
            return read_vti(self.paths[round(t, 6)], names)
        out = {}
        for n in names:
            if n in self.VEC:
                out[n] = np.stack([self._nc(c, k) for c in self.VEC[n]], axis=-1)
            else:
                out[n] = self._nc(n, k)
        return out

    def static(self):
        if self.nc is None:
            return read_vti(self.static_path, ["bed", "beta"])
        return {"bed": self._nc("bed"), "beta": self._nc("beta")}


# ---------------------------------------------------------------- ISMIP grid
class IsmipGrid:
    """Places model arrays (cropped, north to south) on the full ISMIP grid
    (ascending y) and coarsens conservatively to a nested resolution."""

    def __init__(self, domain_path, n_levels, shape, resolution):
        gi = xr.open_dataset(Path(domain_path) / "model_inputs" / "GLIDE_inputs.nc")
        x, y = gi.x.values.astype("float64"), gi.y.values.astype("float64")
        self.dx = float(abs(x[1] - x[0]))
        factor = 2 ** n_levels
        ny, nx = (len(y) // factor) * factor, (len(x) // factor) * factor
        if (ny, nx) != tuple(shape):
            raise ValueError(f"run grid {shape} is not GLIDE_inputs cropped to {factor}: {(ny, nx)}")
        self.y0, self.x0 = (len(y) - ny) // 2, (len(x) - nx) // 2
        self.full = (len(y), len(x))
        self.flip = y[0] > y[-1]
        self.f = int(round(resolution / self.dx))
        if self.f < 1 or abs(self.f * self.dx - resolution) > 1e-6 or (len(x) - 1) % self.f or (len(y) - 1) % self.f:
            raise ValueError(f"resolution {resolution} does not nest in the {self.dx:g} m grid")
        ya = y[::-1] if self.flip else y
        self.x, self.y = x[::self.f], ya[::self.f]
        k = np.ones(self.f + 1); k[0] = k[-1] = 0.5
        self.kernel = k / k.sum()
        self.inside = self._embed(np.ones(shape, np.float32), 0.0)          # 1 inside the model domain

    def _embed(self, a, outside):
        full = np.full(self.full, outside, dtype=np.float32)
        full[self.y0:self.y0 + a.shape[0], self.x0:self.x0 + a.shape[1]] = a
        return full[::-1] if self.flip else full

    def _smooth(self, a):
        if self.f == 1:
            return a
        a = convolve1d(a, self.kernel, axis=0, mode="constant", cval=0.0)
        a = convolve1d(a, self.kernel, axis=1, mode="constant", cval=0.0)
        return a[::self.f, ::self.f]

    def to_ismip(self, a, weight=None, outside=np.nan):
        """Area mean of `a` over each output cell, weighted by `weight`
        (model grid; None = the model domain). Output cells with no weight
        get FILL_VALUE, or `outside` when it is a number (0 for fractions and
        fluxes that are zero by definition off the ice)."""
        w = self.inside if weight is None else self._embed(weight.astype(np.float32), 0.0)
        num = self._smooth(self._embed(np.nan_to_num(a).astype(np.float32), 0.0) * w)
        den = self._smooth(w)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(den > 0.0, num / den, np.nan)      # defined wherever the fraction is
        fill = FILL_VALUE if np.isnan(outside) else np.float32(outside)
        return np.where(np.isfinite(out), out, fill).astype(np.float32)

    def area_fraction(self, a):
        """Fraction of each output cell covered (a in [0, 1] on the model grid);
        0 outside the model domain."""
        return self._smooth(self._embed(a.astype(np.float32), 0.0)).astype(np.float32)


# ------------------------------------------------------------------- physics
class Physics:
    def __init__(self, cfg, bed, beta, dx, Q_geo=None):
        self.Q_geo = Q_geo                  # W m-2, uniform; None = no thermal output
        self.rho_i, self.rho_w, self.g = float(cfg.rho_ice), float(cfg.rho_water), float(cfg.gravity)
        self.m, self.u_reg, self.water_drag = float(cfg.sliding_m), float(cfg.u_reg), float(cfg.water_drag)
        self.u0 = float(getattr(cfg, "sliding_u0", 0.0) or 0.0)       # regularized Coulomb transition (m/yr)
        self.tau_c = float(cfg.calving_timescale)
        self.sigmoid_c = float(cfg.sigmoid_c)
        self.bed, self.beta, self.dx = bed.astype("float64"), beta.astype("float64"), float(dx)
        self.marine = bed < 0.0

    def fields(self, fr):
        """All 2-D fields of one frame on the model grid -> dict name ->
        (array, weight or None, outside value)."""
        r, ri, spy = self.rho_i / self.rho_w, self.rho_i, SECONDS_PER_YEAR
        H = fr["H"].astype("float64")
        ice = fr["mask"] < 0.5                                   # active set: mask = 1 pins the thklim floor
        icef = ice.astype(np.float32)
        psi, xi = fr["psi"].astype("float64"), fr["xi"].astype("float64")
        Hi = np.where(ice, H, 0.0)
        # The grounded fraction is RECOMPUTED from the exported geometry,
        # phi = sigmoid(c (r H + bed)), not taken from the frame: the model's
        # flag lags the final thickness update in a few hundred cells per
        # frame (phi = 1 with the ice just afloat), and the checker requires
        # base = topg wherever sftgrf = 1 and base > topg wherever sftflf = 1.
        z = r * H + self.bed
        phi = 1.0 / (1.0 + np.exp(-np.clip(self.sigmoid_c * z, -60.0, 60.0)))
        # Footprints, by the data request's fill policies (the checker compares
        # them cell by cell): `no_ice` fields are defined exactly where sftgif > 0;
        # `outside_domain` fields (orog, base, topg, acabf) share ONE footprint,
        # the computational domain = ice plus ice-free land (bed >= 0; open
        # ocean is outside, which also keeps orog >= 0), and within it the
        # geometry must satisfy orog = base + lithk, base >= topg, base = topg
        # on wholly grounded cells and base > topg on wholly floating ones --
        # so off the ice base = bed and orog = bed, and every one of these
        # fields is averaged with the same weight when the output is coarsened.
        domain = ice | (self.bed >= 0.0)
        domf = domain.astype(np.float32)
        grf, flf = (phi * ice).astype(np.float32), ((1.0 - phi) * ice).astype(np.float32)
        base = np.where(ice, np.maximum(self.bed, -r * H), self.bed)
        um, vm = fr["U"][..., 0].astype("float64"), fr["U"][..., 1].astype("float64")
        us, vs = fr["U_s"][..., 0].astype("float64"), fr["U_s"][..., 1].astype("float64")
        ub, vb = fr["U_b"][..., 0].astype("float64"), fr["U_b"][..., 1].astype("float64")
        ub2 = ub ** 2 + vb ** 2
        # glide stress.cu: tau_b / (rho_i g) = (beta xi^p K(S) + water_drag) |u_b|, S = |u_b|^2 + u_reg,
        # K = S^((m-1)/2) (u0 / (sqrt(S) + u0))^m (drag_speed_factor; the Coulomb factor only when
        # u0 > 0). xi is glide's state.xi: the flotation fraction, times (H + N_floor_H) / N_scale_H
        # under the dimensional effective pressure (compute_flotation_fraction), as stored in the frame.
        S = ub2 + self.u_reg
        K = S ** ((self.m - 1.0) / 2.0)
        if self.u0 > 0.0:
            K = K * (self.u0 / (np.sqrt(S) + self.u0)) ** self.m
        tau_b = ri * self.g * (self.beta * np.where(xi > 0, xi ** SLIDING_P, 0.0) * K + self.water_drag) * np.sqrt(ub2)
        # licalvf / lifmassbf are non-positive in the request (loss); ligroundf
        # is the flux OF grounded ice across the grounding line, positive
        calv = LOSS_SIGN * ri * (1.0 - psi) * Hi / self.tau_c / spy                  # kg m-2 s-1
        gl = ri * self.grounding_line_flux(Hi, um, vm, ice & (phi >= 0.5), ice) / spy
        zero = np.zeros_like(H)
        out = {
            "lithk": (Hi, domf, 0.0),
            "orog": (base + Hi, domf, np.nan),
            "topg": (self.bed, domf, np.nan),
            "base": (base, domf, np.nan),
            "xvelmean": (um / spy, icef, np.nan), "yvelmean": (vm / spy, icef, np.nan),
            "xvelsurf": (us / spy, icef, np.nan), "yvelsurf": (vs / spy, icef, np.nan),
            "xvelbase": (ub / spy, icef, np.nan), "yvelbase": (vb / spy, icef, np.nan),
            "strbasemag": (tau_b, icef, np.nan),
            "acabf": (fr["smb"].astype("float64") * ri / spy, domf, np.nan),
            "libmassbfgr": (zero, grf, np.nan), "libmassbffl": (zero, flf, np.nan),
            "lifmassbf": (zero, None, 0.0),
            "dlithkdt": (fr["dhdt"].astype("float64") / spy, None, 0.0),
            "licalvf": (calv, None, 0.0),
            "ligroundf": (gl, None, 0.0),
        }
        if self.Q_geo is not None and "T_bed" in fr:
            # the enthalpy model's temperatures (K) on the ice, by the request's
            # fill policies: no_ice / no_grounded_ice / no_floating_ice; the
            # geothermal flux is an outside_domain field like topg
            Tb = fr["T_bed"].astype("float64")
            # Under floating ice the enthalpy model's basal node is not held at
            # the ocean interface (251-264 K where it should be ~271 K), so the
            # floating basal temperature is the in-situ freezing point of
            # seawater at the ice base (Jenkins 2011 liquidus, salinity
            # SEA_SALINITY): T_f = 273.15 + l1 S + l2 + l3 depth.
            T_f = 273.15 + FREEZE_L1 * SEA_SALINITY + FREEZE_L2 + FREEZE_L3 * np.maximum(-base, 0.0)
            out.update({
                "litemptop": (fr["T_top"].astype("float64"), icef, np.nan),
                "litempavg": (fr["T_mean"].astype("float64"), icef, np.nan),
                "litempbotgr": (Tb, grf, np.nan), "litempbotfl": (T_f, flf, np.nan),
                "hfgeoubed": (np.full_like(H, float(self.Q_geo)), domf, np.nan),
            })
        frac = {"sftgif": icef, "sftgrf": (phi * ice).astype(np.float32), "sftflf": ((1.0 - phi) * ice).astype(np.float32)}
        A = self.dx ** 2
        scal = {
            "lim": ri * Hi.sum() * A,
            "limnsw": np.maximum(ri * Hi - self.rho_w * np.maximum(-self.bed, 0.0), 0.0)[ice].sum() * A,
            "iareagr": float((phi * ice).sum()) * A, "iareafl": float(((1.0 - phi) * ice).sum()) * A,
            "tendacabf": float((out["acabf"][0] * ice).sum()) * A,
            "tendlibmassbfgr": 0.0, "tendlibmassbffl": 0.0, "tendlifmassbf": 0.0,
            # the scalar totals are magnitudes in the request (ranges [0, 1e25])
            "tendlicalvf": float(abs(calv.sum())) * A, "tendligroundf": float(gl.sum()) * A,
        }
        return out, frac, scal

    def grounding_line_flux(self, H, u, v, grounded, ice):
        """Upwinded H u outflow (m ice / yr per unit cell area) from grounded
        ice through faces shared with floating ice or open water, charged to
        the grounded cell. Rows run north to south and v > 0 is northward."""
        other = (~grounded) & self.marine                        # floating ice or ocean; not ice-free land
        out = np.zeros_like(H)
        uf = 0.5 * (u[:, :-1] + u[:, 1:])                        # faces between columns j | j + 1
        q = uf * np.where(uf > 0, H[:, :-1], H[:, 1:])
        out[:, :-1] += np.where(grounded[:, :-1] & other[:, 1:], np.maximum(q, 0.0), 0.0)
        out[:, 1:] += np.where(grounded[:, 1:] & other[:, :-1], np.maximum(-q, 0.0), 0.0)
        vf = 0.5 * (v[:-1] + v[1:])                              # faces between rows i (north) | i + 1 (south)
        q = vf * np.where(vf > 0, H[1:], H[:-1])                 # > 0: from the south cell into the north cell
        out[1:] += np.where(grounded[1:] & other[:-1], np.maximum(q, 0.0), 0.0)
        out[:-1] += np.where(grounded[:-1] & other[1:], np.maximum(-q, 0.0), 0.0)
        return out / self.dx


# -------------------------------------------------------------------- output
def _time(year, mode):
    return date2num(datetime(year + 1, 1, 1) if mode == "ST" else datetime(year, 7, 1), TIME_UNITS, TIME_CALENDAR)


class Writer:
    def __init__(self, path, name, meta, years, attrs, grid=None):
        mode, long_name, std, units = meta
        self.nc = nc = Dataset(path, "w", format="NETCDF4")
        for k, v in attrs.items():
            nc.setncattr(k, v)
        nc.title = f"GLIDE export - {name}"
        nc.createDimension("time", None)
        t = nc.createVariable("time", "f4", ("time",))
        t.units, t.calendar, t.axis, t.standard_name = TIME_UNITS, TIME_CALENDAR, "T", "time"
        t[:] = [_time(y, mode) for y in years]
        if mode == "FL":
            nc.createDimension("bnds", 2)
            tb = nc.createVariable("time_bnds", "f4", ("time", "bnds"))
            tb.units, tb.calendar = TIME_UNITS, TIME_CALENDAR
            tb[:, :] = [[date2num(datetime(y, 1, 1), TIME_UNITS, TIME_CALENDAR),
                         date2num(datetime(y + 1, 1, 1), TIME_UNITS, TIME_CALENDAR)] for y in years]
            t.bounds = "time_bnds"
        dims = ("time",)
        kw = {}
        if grid is not None:
            nc.createDimension("y", len(grid.y)); nc.createDimension("x", len(grid.x))
            for d, vals in (("x", grid.x), ("y", grid.y)):
                c = nc.createVariable(d, "f8", (d,))
                c[:] = vals
                c.units, c.standard_name, c.axis = "m", f"projection_{d}_coordinate", d.upper()
            crs = nc.createVariable("crs", "i4")
            crs.grid_mapping_name, crs.epsg_code = "polar_stereographic", "EPSG:3413"
            crs.straight_vertical_longitude_from_pole, crs.latitude_of_projection_origin = -45.0, 90.0
            crs.standard_parallel, crs.false_easting, crs.false_northing = 70.0, 0.0, 0.0
            dims = ("time", "y", "x")
            kw = dict(zlib=True, complevel=4, chunksizes=(1, len(grid.y), len(grid.x)))
        self.var = v = nc.createVariable(name, "f4", dims, fill_value=FILL_VALUE, **kw)
        v.long_name, v.units = long_name, units
        if std:
            v.standard_name = std
        v.cell_methods = "time: point" if mode == "ST" else "time: mean"
        if grid is not None:
            v.grid_mapping = "crs"
        if name in ZERO_NOTE:
            v.comment = "identically zero: " + ZERO_NOTE[name]

    def write(self, k, a):
        if self.var.ndim == 1:
            self.var[k] = np.float32(a)
        else:
            self.var[k, :, :] = a

    def close(self):
        self.nc.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="forward_projection / forward_standalone output directory")
    ap.add_argument("--experiment", required=True, help="historical | ssp126 | ssp370 | ssp585 | ctrl | ...")
    ap.add_argument("--esm", default="CESM2-WACCM")
    ap.add_argument("--years", type=int, nargs=2, default=None, metavar=("Y0", "Y1"),
                    help="first and last model year (default: the core experiment's window, clipped to the run)")
    ap.add_argument("--resolution", type=float, default=1000.0, help="output grid spacing (m): 1000, 2000, 4000, 8000, 16000")
    ap.add_argument("--group", default=GROUP, help="group id in the file name (field 3)")
    ap.add_argument("--set-counter", default="C001", help="set counter (C/E/P + 3 digits), one per experiment and ESM (tools/ismip7_set_counters.py)")
    ap.add_argument("--submission-dir", default=None,
                    help="root of the submission tree; files go to <root>/Models/GrIS/<group>/<model>/CORE/<set>/ "
                         "(default: <run-dir>/../ISMIP7_submission)")
    ap.add_argument("--variables", nargs="*", default=None, help="subset of the variable names")
    a = ap.parse_args()

    cfg = load_config(DOMAIN)
    run_dir = Path(a.run_dir)
    src = FrameSource(run_dir)
    frames = [(t, None) for t in src.times]
    by_time = src.index
    window = EXPERIMENTS.get(a.experiment)
    if a.years:
        y0, y1 = a.years
    elif window:
        y0, y1 = window
        if a.experiment == "historical":            # the modeller's start: the first yearly frame pair
            first = next((int(round(t)) for k, t in enumerate(src.times[:-1])
                          if abs(src.times[k + 1] - t - 1.0) < 1e-6), 1850)
            y0 = max(y0, first)
    else:
        y0, y1 = int(frames[0][0]), int(frames[-1][0]) - 1
    if window and (y1 != window[1] or (a.experiment != "historical" and y0 != window[0])):
        print(f"WARNING: {a.experiment} must cover {window[0]}-{window[1]} ({y0 if a.experiment == 'historical' else window[0]}-{window[1]} "
              f"for this run); {y0}-{y1} will fail the checker's time test")
    y1 = min(y1, int(round(frames[-1][0])) - 1)
    years = list(range(y0, y1 + 1))
    # model year y = the step (y, y + 1]: its frame must exist, and so must the
    # frame one year earlier (or y is the run's first year), else dt != 1 yr
    t_start = int(round(frames[0][0])) - 1
    missing = [y for y in years if round(y + 1.0, 6) not in by_time
               or (round(float(y), 6) not in by_time and y != t_start)]
    if missing:
        raise SystemExit(f"no yearly frame pair for model years {missing[:5]}{' ...' if len(missing) > 5 else ''}: "
                         f"the run must write a VTI frame every year (VTI_EVERY = 1) over the exported window")

    st = src.static()
    grid = IsmipGrid(DOMAIN, cfg.n_levels, st["bed"].shape, a.resolution)

    meta = {}
    for fn in ("snapshots.nc", "forward_soln.nc", "final_state.nc"):
        if (run_dir / fn).exists():
            with xr.open_dataset(run_dir / fn) as s:
                meta = {k: str(v) for k, v in s.attrs.items() if k != "crs_wkt"}
            break
    # thermal output: the run's ThermalConfig (attrs) and the fields in its frames
    thermal_cfg = meta.get("thermal", "")
    Q_geo = None
    if thermal_cfg.startswith("ThermalConfig"):
        m = re.search(r"Q_geo=([-+0-9.eE]+)", thermal_cfg)
        try:
            src.get(src.times[-1], THERMAL_FRAME_FIELDS)
            Q_geo = float(m.group(1)) if m else None
        except Exception as e:                      # frames written without the thermal fields
            print(f"WARNING: the run has {thermal_cfg[:40]}... but its frames lack {THERMAL_FRAME_FIELDS} ({e}); "
                  f"temperatures not exported")
    thermal_on = Q_geo is not None
    one_way = thermal_on and "couple_rheology=False" in thermal_cfg
    grid_vars = dict(GRID_VARS, **(THERMAL_VARS if thermal_on else {}))
    not_modelled = {k: v for k, v in NOT_MODELLED.items() if not (thermal_on and k in THERMAL_VARS)}
    if thermal_on:
        not_modelled["litemp"] = "the 3-D enthalpy field is not written to the yearly frames"
    phys = Physics(cfg, st["bed"], st["beta"], grid.dx, Q_geo=Q_geo)
    attrs = {
        "Conventions": "CF-1.7", "ismip7_version": "7.0", "institution": "University of Montana",
        "source": f"GLIDE ice sheet model (MOLHO), enthalpy SMB model, forced by {a.esm}",
        "contact_name": "Doug Brinkerhoff", "contact_email": "doug.brinkerhoff@mso.umt.edu",
        "model": MODEL, "group": a.group, "grid_type": REGION, "experiment": a.experiment, "esm": a.esm, "set": a.set_counter,
        "crs": "EPSG:3413", "proj_params": "+proj=stere +lon_0=-45 +lat_ts=70 +lat_0=90 +x_0=0 +y_0=0",
        "native_resolution_m": grid.dx, "output_resolution_m": float(a.resolution),
        "calving": (f"cell sink (1 - psi) H / tau, tau = {phys.tau_c:g} a, psi the monotone gap-blended "
                    f"height-above-buoyancy flag (H_c = {float(cfg.calving_H_c):g} m); ocean thermal forcing acts "
                    f"through the calving margins, so lifmassbf = 0 and licalvf holds all marine loss"),
        "flux_sign": "positive = mass gain of the ice sheet" if LOSS_SIGN < 0 else "loss fluxes reported positive",
        "densities": f"rho_ice {phys.rho_i:g}, rho_sea_water {phys.rho_w:g} kg m-3",
        "run_dir": str(run_dir.resolve()),
        "history": f"Generated {datetime.now(timezone.utc).isoformat()} by ismip_exporter.py",
    }
    for k in ("climate_mode", "climate", "ocean_forcing", "elevation_feedback", "checkpoint", "thermal"):
        if k in meta:
            attrs[f"run_{k}"] = meta[k]
    if thermal_on:
        attrs["ice_temperature"] = (
            ("ONE-WAY coupled: the enthalpy model (glide, Aschwanden) is advected and heated (strain, basal friction) "
             "by the model's velocities, but the rheology stays isothermal (A_glen); the temperatures are a diagnostic "
             "of the uncoupled flow. " if one_way else
             "Thermomechanically coupled: B from the enthalpy model (Paterson-Budd, depth-collapsed). ")
            + f"Uniform geothermal flux {Q_geo:g} W m-2; surface temperature = annual-mean forcing air temperature "
              f"capped at 0 degC (or the calibration climatology, see run_thermal: surface_T). litempbotfl is the "
              f"in-situ seawater freezing point at the ice base (S = {SEA_SALINITY:g} psu, Jenkins 2011), not the "
              f"enthalpy model's basal node.")

    root = Path(a.submission_dir) if a.submission_dir else run_dir.parent / "ISMIP7_submission"
    out_dir = root / "Models" / REGION / a.group / MODEL / "CORE" / a.set_counter
    out_dir.mkdir(parents=True, exist_ok=True)
    config_id = a.set_counter
    period = f"{years[0]}-{years[-1]}"
    fname = lambda v: out_dir / (f"{v}_{REGION}_{a.group}_{MODEL}_{ISM_MEMBER}_{a.esm}_{FORCING_ID}_"
                                 f"{a.experiment}_{config_id}_{period}.nc")
    with open(out_dir / "not_modelled.txt", "w") as f:
        f.write("# ISMIP7: non-mandatory variables this model does not represent (read by the compliance checker)\n")
        for v, why in not_modelled.items():
            f.write(f"{v:14s} # {why}\n")
    want = set(a.variables) if a.variables else set(grid_vars) | set(SCALAR_VARS)
    unknown = want - set(grid_vars) - set(SCALAR_VARS)
    if unknown:
        raise SystemExit(f"unknown variables {sorted(unknown)}")
    writers = {v: Writer(fname(v), v, grid_vars[v], years, attrs, grid) for v in grid_vars if v in want}
    writers.update({v: Writer(fname(v), v, SCALAR_VARS[v], years, attrs) for v in SCALAR_VARS if v in want})
    print(f"{a.experiment} (set {config_id}) {period}: {len(years)} years, {len(writers)} variables, "
          f"{a.resolution:g} m grid {len(grid.y)} x {len(grid.x)} from {src.kind} -> {out_dir}")

    names = ["H", "smb", "dhdt", "mask", "phi", "psi", "xi", "U", "U_s", "U_b"] + (THERMAL_FRAME_FIELDS if thermal_on else [])
    try:
        for k, y in enumerate(years):
            out, frac, scal = phys.fields(src.get(y + 1.0, names))
            for v, (arr, w, outside) in out.items():
                if v in writers:
                    writers[v].write(k, grid.to_ismip(arr, w, outside))
            for v, arr in frac.items():
                if v in writers:
                    writers[v].write(k, grid.area_fraction(arr))
            for v, val in scal.items():
                if v in writers:
                    writers[v].write(k, val)
            print(f"  {y}: lim {scal['lim'] / 1e12:.0f} Gt, SMB {scal['tendacabf'] * SECONDS_PER_YEAR / 1e12:+.0f}, "
                  f"calving {scal['tendlicalvf'] * SECONDS_PER_YEAR / 1e12:+.0f}, "
                  f"GL flux {scal['tendligroundf'] * SECONDS_PER_YEAR / 1e12:+.0f} Gt/yr", flush=True)
    finally:
        for w in writers.values():
            w.close()
    print(f"wrote {len(writers)} files to {out_dir}")


if __name__ == "__main__":
    main()
