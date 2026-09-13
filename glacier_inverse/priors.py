"""
GlacierPriors: the prior models (4 Matern fields, plus a 5th for the optional
temperature bias, + scalar SMB priors) and the optional bed GP conditioner
(posterior-as-prior).

Cheap to build — does not construct IceDynamics. Used directly by posterior.py
to map RTO whitened samples back to physical space without standing up the
full forward model. Reused inside GlacierProblem. Everything here (including
the bed-conditioning data) is loadable from the config alone, so the
standalone path stays intact.
"""
import dataclasses
import math
import warnings
from pathlib import Path

import cupy as cp
import numpy as np
import xarray as xr
from scipy.special import gamma as _gamma_fn

from ggapp.model import MaternPrior
from ggapp.torch import GGaPPMap

# PriorCollection (N priors on one shared multigrid/solver — saves a full
# hierarchy of workspace per additional prior) is newer than some installed
# ggapp versions; fall back to one hierarchy per prior when absent.
try:
    from ggapp.collection import PriorCollection
except ImportError:
    PriorCollection = None

from .config import GlacierConfig, MaternNoise, PriorHyperparams


def _build_matern_prior(p: PriorHyperparams, n_levels: int, ny: int, nx: int, dx: float) -> MaternPrior:
    m = MaternPrior(n_levels=n_levels, ny=ny, nx=nx, dx=dx)
    m.mg.parameters.sigma.set(p.sigma)
    m.mg.parameters.l.set(p.l)
    m.mg.parameters.nu.set(p.nu)
    m.forward_solver.fas_options.report_norms.set(False)
    return m


class SpectralMaternNoise:
    """Matérn(ν) + nugget error model as an exact spectral operator.

    ggapp's Matérn operator L = −Δ_h + κ² uses the 5-point Laplacian with
    mirror (Neumann) boundaries, which the orthonormal DCT-II diagonalizes
    exactly (eigenvalues λ_ij = (2 − 2cos(πi/ny) + 2 − 2cos(πj/nx))/dx²), so the
    Matérn covariance C_m = (τ/dx)² L^{−α}, α = ν + 1, has eigen-variance
    S(λ) = (τ/dx)²(κ² + λ)^{−α} in that basis — for ANY real ν, not just the
    odd integers the stencil can realize — and the nugget adds a constant:
    C = C_m + nugget²·I. `whiten` applies C^{−1/2}, `forward` applies C^{+1/2}
    — both self-adjoint, O(N log N) through cupyx.scipy.fft, exact (with
    nugget = 0 and odd ν they reproduce ggapp's stencil whitening to float32
    rounding, ~4e-7 relative). Duck-compatible with a ggapp prior member:
    cupy in / cupy out, so `GGaPPWhiten` / `GGaPPMap` drive it unchanged.
    """

    def __init__(self, hp: MaternNoise, ny: int, nx: int, dx: float):
        import cupyx.scipy.fft as cfft
        self._fft = cfft
        self.hp, self.ny, self.nx, self.dx = hp, ny, nx, dx
        li = (2.0 - 2.0 * np.cos(np.pi * np.arange(ny) / ny)) / dx ** 2
        lj = (2.0 - 2.0 * np.cos(np.pi * np.arange(nx) / nx)) / dx ** 2
        lam = li[:, None] + lj[None, :]

        def matern_var(sigma, l, nu):
            nu, alpha = float(nu), float(nu) + 1.0
            kappa = math.sqrt(8.0 * nu) / l
            tau = math.sqrt(sigma ** 2 * (4.0 * math.pi) * kappa ** (2 * nu)
                            * _gamma_fn(alpha) / _gamma_fn(nu))
            return (tau / dx) ** 2 * (kappa ** 2 + lam) ** (-alpha)

        # Covariances add: base Matérn + white nugget + the optional smooth
        # discrepancy component (the marginalized Kennedy–O'Hagan field) —
        # all diagonal in the shared DCT basis, so the sum is still an exact
        # spectral filter.
        var = matern_var(hp.sigma, hp.l, hp.nu) + hp.nugget ** 2
        if hp.discrepancy is not None:
            d = hp.discrepancy
            var = var + matern_var(d.sigma, d.l, d.nu)
        self._w = cp.asarray(var ** -0.5, dtype=cp.float32)   # C^{-1/2}
        self._m = cp.asarray(var ** 0.5, dtype=cp.float32)    # C^{+1/2}

    def _apply(self, x, f):
        x = cp.asarray(x, dtype=cp.float32)
        X = self._fft.dctn(x, type=2, norm="ortho")
        return self._fft.idctn(X * f, type=2, norm="ortho").astype(cp.float32)

    def whiten(self, x):
        return self._apply(x, self._w)

    def forward(self, z, zero_init=True):
        return self._apply(z, self._m)


def _cropped_inputs(config: GlacierConfig, variables: list = None) -> "xr.Dataset":
    """Open the gridded dataset and apply the same factor-aligned center crop
    the full problem uses. The single home of the crop logic for standalone
    (no-IceDynamics) consumers.

    `variables` selects a subset (an empty list keeps just the coords), which
    is loaded eagerly and the file handle closed before returning — reopening
    a netCDF after other handles on it have been garbage-collected can
    segfault this HDF5 stack, so standalone consumers should never hold a
    lazy handle. `variables=None` returns the lazy full dataset (caller owns
    the handle's lifetime).
    """
    gd = xr.open_dataset(Path(config.base_dir) / "model_inputs" / config.gridded_filename)
    factor = 2 ** config.n_levels
    ny0, nx0 = gd.sizes["y"], gd.sizes["x"]
    ny_target = (ny0 // factor) * factor
    nx_target = (nx0 // factor) * factor
    y_start = (ny0 - ny_target) // 2
    x_start = (nx0 - nx_target) // 2
    cropped = gd.isel(y=slice(y_start, y_start + ny_target),
                      x=slice(x_start, x_start + nx_target))
    if variables is None:
        return cropped
    sub = cropped[variables].load()
    gd.close()
    return sub


def domain_shape(config: GlacierConfig) -> tuple:
    """(ny, nx, dx) of the cropped model grid. Cheap — used by posterior.py to
    build only the priors without standing up IceDynamics."""
    # An empty-variable subset loses the y/x dims from .sizes, so carry one
    # cheap 2-D variable through the eager load-and-close path.
    gd = _cropped_inputs(config, variables=["elevation"])
    ny, nx = gd.sizes["y"], gd.sizes["x"]
    dx = (gd.x[1] - gd.x[0]).item()
    return ny, nx, dx


def _available_variables(config: GlacierConfig) -> set:
    """Names of the data variables in the gridded input file (opened and
    closed eagerly — see _cropped_inputs on why no handle may linger)."""
    path = Path(config.base_dir) / "model_inputs" / config.gridded_filename
    with xr.open_dataset(path) as gd:
        return set(gd.data_vars)


def build_bed_conditioning_data(config: GlacierConfig) -> tuple:
    """Assemble the bed-conditioning data (b, D) on the cropped grid, from the
    config alone. Returns two (ny, nx) float32 numpy arrays: the precision
    field D (zero = unobserved) and the precision-weighted data b.

    Off-ice / out-of-domain pixels contribute bed = DEM (the full DEM
    including bathymetry, matching BedObservation's anchor) at 1/sigma_dem^2.
    Flightline picks are snapped to their nearest grid cell (the picks are
    already 90 m along-track resampled; deliberately NOT the sub-pixel
    grid_sample operator of BedSpec, whose bilinear spreading would make D
    non-diagonal) and scatter-added at 1/sigma_picks^2; coincident data merge
    by precision weighting. Out-of-crop and NaN picks are dropped; a missing
    or empty flightline file just contributes nothing.

    Gridded bed observations (`bed_obs` / `bed_obs_err` in the input file —
    BedMachine bed + errbed for Greenland) contribute every finite on-ice cell
    at precision 1/(scale * max(err, floor))^2 when
    `bed_conditioning.use_gridded_bed` (see BedConditioningConfig); off-ice
    cells are left to the DEM anchor so the two never double count.
    """
    bc = config.bed_conditioning
    variables = ["elevation", "rgi_mask", "domain_mask"]
    available = _available_variables(config)
    use_gridded = (getattr(bc, "use_gridded_bed", False)
                   and {"bed_obs", "bed_obs_err"} <= available)
    mode = getattr(bc, "gridded_bed_data", "all")
    if mode not in ("all", "radar"):
        raise ValueError(f"bed_conditioning.gridded_bed_data must be 'all' or "
                         f"'radar', got {mode!r}")
    radar_vars = {"bed_obs_radar", "bed_obs_radar_err", "bed_radar_fraction"}
    if use_gridded and mode == "radar":
        if not radar_vars <= available:
            raise ValueError(
                "bed_conditioning.gridded_bed_data='radar' needs bed_obs_radar / "
                "bed_obs_radar_err / bed_radar_fraction in the gridded inputs — "
                "rebuild with preprocessing/make_dem.py (BedMachine dataid)")
        variables += sorted(radar_vars)
    elif use_gridded:
        variables += ["bed_obs", "bed_obs_err"]
    gd = _cropped_inputs(config, variables=variables)
    ny, nx = gd.sizes["y"], gd.sizes["x"]
    dem = gd.elevation.values.astype(np.float64)

    D = np.zeros((ny, nx), dtype=np.float64)
    num = np.zeros((ny, nx), dtype=np.float64)   # sum of value/sigma^2

    off = (gd.rgi_mask.values == 0) | (gd.domain_mask.values == 0)
    if use_gridded:
        if mode == "radar":
            b_obs = gd.bed_obs_radar.values.astype(np.float64)
            e_obs = gd.bed_obs_radar_err.values.astype(np.float64) * bc.gridded_bed_err_scale
            frac = gd.bed_radar_fraction.values.astype(np.float64)
            ok = (frac > bc.radar_fraction_min)
        else:
            b_obs = gd.bed_obs.values.astype(np.float64)
            e_obs = gd.bed_obs_err.values.astype(np.float64) * bc.gridded_bed_err_scale
            ok = np.ones(b_obs.shape, dtype=bool)
        ok &= np.isfinite(b_obs) & np.isfinite(e_obs) & (e_obs > 0) & ~off
        if bc.gridded_bed_max_err is not None:
            ok &= e_obs <= bc.gridded_bed_max_err
        e_obs = np.maximum(e_obs, bc.gridded_bed_err_floor)
        w = 1.0 / e_obs ** 2
        D[ok] += w[ok]
        num[ok] += w[ok] * b_obs[ok]
        print(f"[bed conditioning] gridded bed ({mode}): {int(ok.sum())} on-ice cells, "
              f"median error {np.median(e_obs[ok]) if ok.any() else float('nan'):.0f} m")

    if bc.include_off_ice:
        if getattr(bc, "exclude_zero_dem", True):
            zero_fill = dem == 0.0
            n_zero = int((off & zero_fill).sum())
            if n_zero:
                warnings.warn(
                    f"bed conditioning: excluding {n_zero} exactly-zero DEM "
                    f"cells from the off-ice anchor (land/bathymetry void "
                    f"fill, not bed data)")
            off &= ~zero_fill
        w = 1.0 / bc.sigma_dem ** 2
        D[off] += w
        num[off] += w * dem[off]

    fl_path = Path(config.base_dir) / "model_inputs" / config.flightline_filename
    if fl_path.exists():
        import geopandas as gpd
        fl = gpd.read_file(fl_path)
        if len(fl) > 0:
            x = fl["x"].values.astype(np.float64)
            y = fl["y"].values.astype(np.float64)
            bed = fl["bed"].values.astype(np.float64)
            xs = gd.x.values.astype(np.float64)
            ys = gd.y.values.astype(np.float64)
            col = np.rint((x - xs[0]) / (xs[1] - xs[0])).astype(np.int64)
            row = np.rint((y - ys[0]) / (ys[1] - ys[0])).astype(np.int64)
            keep = ((col >= 0) & (col < nx) & (row >= 0) & (row < ny)
                    & np.isfinite(bed))
            n_dropped = int((~keep).sum())
            if n_dropped:
                warnings.warn(f"bed conditioning: dropped {n_dropped} of "
                              f"{len(fl)} flightline picks (outside the "
                              f"cropped grid or NaN bed)")
            w = 1.0 / bc.sigma_picks ** 2
            flat = row[keep] * nx + col[keep]
            np.add.at(D.ravel(), flat, w)
            np.add.at(num.ravel(), flat, w * bed[keep])

    if not D.any():
        warnings.warn("bed conditioning enabled but no data found (no "
                      "flightlines, off-ice anchor disabled) — the "
                      "conditioning map is the identity")
    b = np.divide(num, D, out=np.zeros_like(num), where=D > 0)
    return b.astype(np.float32), D.astype(np.float32)


class GlacierPriors:
    def __init__(self, config: GlacierConfig, ny: int, nx: int, dx: float):
        self.config = config
        self.ny = ny
        self.nx = nx
        self.dx = dx

        tbias_enabled = getattr(config, "tbias_enabled", False)
        enthalpy = getattr(config, "smb_model", "temperature_index") == "enthalpy"
        if PriorCollection is not None:
            # All field priors share one multigrid hierarchy + solver: the
            # SPDE solve is stateless between calls, and members differ only
            # in scalar hyperparameters (re)bound per solve — bit-identical
            # to separate MaternPriors at a fifth of the workspace. The bed
            # conditioner's shifted preconditioner joins the same collection
            # (it owns only its shift-coefficient arrays).
            self.prior_collection = PriorCollection(
                config.n_levels, ny=ny, nx=nx, dx=dx)
            self.prior_collection._prior.forward_solver \
                .fas_options.report_norms.set(False)

            def _add(name, p):
                return self.prior_collection.add(name, p.sigma, p.l, p.nu)
            self.bed_model      = _add("bed",      config.bed_prior)
            self.mean_model     = _add("mean",     config.mean_prior)
            self.log_beta_model = _add("log_beta", config.log_beta_prior)
            self.pbias_model    = _add("pbias",    config.pbias_prior)
            # Optional additive temperature bias (K). Registering a member is
            # ~free (scalars only), but keep the enabled gate so the disabled
            # case stays structurally identical (tbias_model is None).
            self.tbias_model = (_add("tbias", config.tbias_prior)
                                if tbias_enabled else None)
            # Enthalpy SMB parameter FIELDS (log H_atm, logit clear-sky
            # fraction): very smooth GPs (l ~ synoptic scale) replacing the
            # old scalars — same gate pattern as tbias, keyed to the active
            # SMB backend.
            self.h_atm_model = (_add("h_atm", config.h_atm_prior)
                                if enthalpy else None)
            self.cloud_model = (_add("cloud", config.cloud_prior)
                                if enthalpy else None)
        else:
            self.prior_collection = None
            self.bed_model      = _build_matern_prior(config.bed_prior,      config.n_levels, ny, nx, dx)
            self.mean_model     = _build_matern_prior(config.mean_prior,     config.n_levels, ny, nx, dx)
            self.log_beta_model = _build_matern_prior(config.log_beta_prior, config.n_levels, ny, nx, dx)
            self.pbias_model    = _build_matern_prior(config.pbias_prior,    config.n_levels, ny, nx, dx)
            # Optional additive temperature bias (K). No hierarchy is built
            # when the term is disabled — z_tbias sits inert at 0.
            self.tbias_model = (
                _build_matern_prior(config.tbias_prior, config.n_levels, ny, nx, dx)
                if tbias_enabled else None)
            self.h_atm_model = (
                _build_matern_prior(config.h_atm_prior, config.n_levels, ny, nx, dx)
                if enthalpy else None)
            self.cloud_model = (
                _build_matern_prior(config.cloud_prior, config.n_levels, ny, nx, dx)
                if enthalpy else None)

        # Correlated observation-error models (MaternNoise on a spec), keyed
        # by product name; see noise_model(). They join the shared collection
        # (scalars only) or get their own hierarchy under the fallback.
        self.noise_models: dict = {}

        # Optional bed GP conditioning (posterior-as-prior). Built from the
        # config alone so the standalone (posterior.py) path keeps working.
        self.bed_conditioner = None
        bc = getattr(config, "bed_conditioning", None)
        if bc is not None and bc.enabled:
            try:
                from ggapp.conditioning import ConditionedPrior
            except ImportError as e:
                raise ImportError(
                    "config.bed_conditioning.enabled=True requires a ggapp "
                    "with ggapp.conditioning (update the ggapp install)") from e
            b, D = build_bed_conditioning_data(config)
            self.bed_conditioner = ConditionedPrior(
                self.bed_model, b, D,
                rtol=bc.pcg_rtol, maxiter=bc.pcg_maxiter,
                warm_start=bc.warm_start,
                rtol_adjoint=getattr(bc, "pcg_rtol_adjoint", None),
                preconditioner=getattr(bc, "pcg_preconditioner", "shifted"))

        # Scalar mean of the log_beta field prior (the Matern prior acts on
        # log_beta - mu_log_beta).
        self.mu_log_beta = getattr(config, "mu_log_beta", 0.0)

        self.mu_log_rf = float(np.log(config.mu_rf))
        self.mu_log_mf = float(np.log(config.mu_mf))
        self.sigma_log_rf = config.sigma_log_rf
        self.sigma_log_mf = config.sigma_log_mf

        # Enthalpy-model parameter fields: log-normal on H_atm (in W m-2 K-1)
        # and logit-normal on the clear-sky fraction f (direct
        # q_sw_insol = f * S0, diffuse q_sw_dif = (f k_clr + (1 - f) k_cld) * S0
        # from the same f). The mu_* are the (spatially constant) medians of
        # the h_atm_model / cloud_model GP fields; the sigma_* are the LEGACY
        # scalar stds, kept only for checkpoint conversion and the model-less
        # affine fallback in physical_from.
        self.mu_log_H_atm = float(np.log(config.mu_H_atm))
        self.sigma_log_H_atm = config.sigma_log_H_atm
        self.mu_logit_cloud = float(
            np.log(config.mu_cloud_factor / (1.0 - config.mu_cloud_factor)))
        self.sigma_logit_cloud = config.sigma_logit_cloud

        # Elevation-dependent precip depletion scalars (normal priors, directly
        # on tau and z0 — not log-normal; tau is itself a log length scale).
        self.mu_tau = config.mu_tau
        self.sigma_tau = config.sigma_tau
        self.mu_z0 = config.mu_z0
        self.sigma_z0 = config.sigma_z0

    def noise_model(self, name: str, hp: MaternNoise, level: int = 0):
        """The model whitening the residual field of observation `name`:
        a member `noise_<name>` on the shared ggapp collection for a pure
        odd-ν Matérn (nugget == 0), or a `SpectralMaternNoise` otherwise
        (nugget set, or a ν the stencil cannot realize). Idempotent:
        re-registering the same product with equal hyperparameters returns
        the stored model; differing hyperparameters raise, since one product
        cannot carry two error models.

        `level > 0` returns the same physical model discretized on that
        multigrid level's grid (key `noise_<name>_L<level>`, always
        spectral): σ, l, ν carry over unchanged, and the white nugget scales
        as nugget/2^level (box-averaging 4^level pixels), which keeps the
        quadratic form of any resolved residual level-independent."""
        key = f"noise_{name}" if level == 0 else f"noise_{name}_L{level}"
        stored = self.noise_models.get(key)
        if stored is not None:
            hp0, member = stored
            if hp0 != hp:
                raise ValueError(
                    f"noise model {key!r} already registered with {hp0}, "
                    f"cannot re-register with {hp}")
            return member
        if level > 0:
            f = 2 ** level
            hp_L = dataclasses.replace(hp, nugget=hp.nugget / f)
            member = SpectralMaternNoise(hp_L, self.ny // f, self.nx // f,
                                         self.dx * f)
        elif not hp.stencil_compatible:
            member = SpectralMaternNoise(hp, self.ny, self.nx, self.dx)
        elif self.prior_collection is not None:
            member = self.prior_collection.add(key, hp.sigma, hp.l, int(hp.nu))
        else:
            member = _build_matern_prior(hp, self.config.n_levels,
                                         self.ny, self.nx, self.dx)
        self.noise_models[key] = (hp, member)
        return member

    def log_beta_from_whitened(self, z_log_beta):
        """THE whitened -> log_beta map (mu_log_beta + Map(z)), shared by
        problem.physical_from, posterior.py, and sensitivity.py."""
        return self.mu_log_beta + GGaPPMap.apply(self.log_beta_model, z_log_beta)

    @property
    def bed_parametrization(self) -> str:
        return "conditioned" if self.bed_conditioner is not None else "legacy"

    def bed_from_whitened(self, z_bed, z_bed_mean, data_override=None):
        """THE whitened -> physical bed map, shared by problem.physical_from,
        posterior.py, and sensitivity.collect_rto_samples. Returns
        (bed, bed_mean, bed_uncond).

        Legacy parametrization: bed = bed_uncond = Map(z_bed). Conditioned
        parametrization: the unconditional field u0 = Map(z_bed) is corrected
        onto the bed data, bed = u0 + (Q+D)^-1 D (b - u0) (Matheron's rule
        with the fluctuation covariance). In BOTH cases the bed_mean
        hierarchy acts purely through the re-whitened prior term on the
        *unconditional* field (loss.compute_prior) — bed_mean deliberately
        does NOT enter the conditioning map: routing data cotangents through
        the mean model's near-singular Helmholtz backward (kappa ~ 1/10 km)
        overflows float32, and keeping the mean prior-coupled preserves the
        legacy curvature structure that the per-domain learning rates were
        tuned against. `data_override` substitutes b for this call (RTO
        per-sample perturbed data).
        """
        bed_mean = GGaPPMap.apply(self.mean_model, z_bed_mean)
        x = GGaPPMap.apply(self.bed_model, z_bed)
        if self.bed_conditioner is None:
            return x, bed_mean, x
        from ggapp.torch import GGaPPCondition
        return (GGaPPCondition.apply(self.bed_conditioner, x, data_override),
                bed_mean, x)
