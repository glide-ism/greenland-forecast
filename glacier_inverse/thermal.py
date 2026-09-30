"""Thermomechanical coupling: glide's enthalpy model driving the rate factor B.

One `ThermalDriver` per run level wraps glide's `ThermalModel` (glide/model.py,
ported from the DIVA-tree thermal branch) for the three drivers (the inverse's
`forward.simulate`, `forward_standalone`, `forward_projection`):

    driver.spinup(...)        # before the first step: frozen-geometry equilibrium
    for each step:
        driver.pre_step(H_prev)   # snapshot E and the step's starting thickness
        <dynamics step>
        driver.post_step(dt)      # one implicit enthalpy step, B -> every level

The thermal state carries no gradient (B is a frozen input of the adjoint;
glide's GlideStep checkpoints the B each step used). See config.ThermalConfig.
"""
from __future__ import annotations

import time as _time

import cupy as cp
import numpy as np
import torch

SECONDS_PER_YEAR = 365.25 * 86400.0
T0 = 273.15


def _restrict(a, n_times: int):
    a = cp.asarray(a, dtype=cp.float32)
    for _ in range(n_times):
        ny, nx = a.shape
        a = a.reshape(ny // 2, 2, nx // 2, 2).mean(axis=(1, 3))
    return a


def surface_temperature_fine(monthly_t2m, tbias=None) -> cp.ndarray:
    """Annual-mean air temperature (degC, + tbias) capped at 0 degC, in K, fine grid."""
    t = cp.asarray(monthly_t2m, dtype=cp.float32).mean(axis=0)
    if tbias is not None:
        tb = tbias.detach() if isinstance(tbias, torch.Tensor) else tbias
        t = t + cp.asarray(tb, dtype=cp.float32)
    return cp.minimum(t, 0.0) + cp.float32(T0)


class ThermalDriver:
    def __init__(self, model, level: int, cfg, rho_i: float, thin_B: float = None):
        from glide.model import ThermalModel
        self.model = model
        self.level = level
        self.cfg = cfg
        grid = model.mg.levels[level]
        self.grid = grid
        enh = None if float(cfg.enhancement) == 1.0 else cp.float32(cfg.enhancement)
        self.coupled = bool(getattr(cfg, "couple_rheology", True))
        self.tm = ThermalModel(grid, nz=cfg.nz, n_smooth=cfg.n_smooth,
                               update_rheology=self.coupled,
                               frictional_heating=cfg.frictional_heating,
                               strain_heating=cfg.strain_heating,
                               rho_i=float(rho_i), mg=model.mg, level=level,
                               weighting=cfg.weighting, enhancement=enh,
                               thin_B=thin_B)
        sc = self.tm.ops.smoother_config
        sc.n_newton = int(cfg.n_newton)
        sc.lf_c = cp.float32(cfg.lf_c)
        sc.absolute_tolerance = cp.float32(cfg.absolute_tolerance)
        sc.relative_tolerance = cp.float32(1e-7)
        sc.report_norms = False
        self.tm.ops.enthalpy_forcing.h_thin.set(float(cfg.h_thin))
        self._E_last = None            # warm start across runs on this level
        self.spinup_info = {}

    # ------------------------------------------------------------ helpers
    def _T_bed_mean(self, thick):
        T = self.tm.ops.get_temperature()[:, :, 0]
        return T, float(T[thick].mean()) if bool(thick.any()) else float("nan")

    def set_surface(self, T_surface_fine):
        self.T_surface = _restrict(T_surface_fine, self.level)
        self.tm.set_surface_temperature(self.T_surface)

    def _push(self):
        if self.coupled:
            self.tm.push_rheology()

    # ------------------------------------------------------------ spin-up
    def spinup(self, *, H0, momentum_solve, T_surface_fine):
        """Frozen-geometry thermal equilibrium on the initial state.

        H0 is the run level's initial thickness (torch or cupy);
        `momentum_solve()` runs one dynamics solve on the initial state with
        the current B (the caller's step mechanics, zero SMB, a short
        `spinup_momentum_dt`, no grad). For each of `spinup_outer` cycles it
        gives the velocities; the geometry is reset to H0; implicit enthalpy steps
        of `spinup_dt` years run to `spinup_tol_K`; B is pushed. Velocities
        are zeroed afterwards so the run starts as an isothermal one would.
        """
        cfg = self.cfg
        mg = self.model.mg
        lvl = self.level
        tic = _time.time()
        H0c = cp.asarray(H0.detach() if isinstance(H0, torch.Tensor) else H0, dtype=cp.float32).copy()
        self.set_surface(T_surface_fine)
        mg.state.H.set(H0c, start_level=lvl)
        if cfg.warm_start and self._E_last is not None and self._E_last.shape == self.tm.ops.enthalpy_state.E.shape:
            self.tm.ops.enthalpy_state.E[:] = self._E_last
            self.tm.ops.enthalpy_forcing.Q_geo.fill(cp.float32(cfg.Q_geo))
        else:
            self.tm.initialize(T_surface=self.T_surface, T_field=self.T_surface, Q_geo=cfg.Q_geo)
        self._push()                   # B consistent with the starting E before the first solve
        thick = H0c > 100.0
        dt_sec = float(cfg.spinup_dt) * SECONDS_PER_YEAR
        steps = []
        with torch.no_grad():
            # one-way: the velocities never see the thermal B, one cycle is the equilibrium
            for outer in range(int(cfg.spinup_outer) if self.coupled else 1):
                momentum_solve()
                mg.state.H.set(H0c, start_level=lvl)
                mg.state.H_prev.set(H0c, start_level=lvl)
                self.tm.update_rheology = False
                _, Tb_prev = self._T_bed_mean(thick)
                n = 0
                dT = dT_prev = float("inf")
                for n in range(1, int(cfg.spinup_max_steps) + 1):
                    self.tm.pre_momentum()
                    self.tm.step(dt_sec)
                    _, Tb = self._T_bed_mean(thick)
                    dT = abs(Tb - Tb_prev)
                    Tb_prev = Tb
                    # remaining drift of the mean basal temperature, extrapolating
                    # the geometric decay of successive changes (r = dT / dT_prev);
                    # a small single change is NOT convergence when r ~ 1 (a
                    # per-step test stopped after 1 step 0.16 K short, 2026-09-28)
                    r = dT / dT_prev if dT_prev > 0 else 0.0
                    remaining = dT * r / (1.0 - r) if r < 1.0 else float("inf")
                    dT_prev = dT
                    if n >= int(cfg.spinup_min_steps) and (remaining < cfg.spinup_tol_K or dT == 0.0):
                        break
                self.tm.update_rheology = self.coupled
                self._push()
                steps.append((n, dT, Tb_prev))
        # Zero the velocities so the run starts as an isothermal one does. At
        # 1 km with 25-yr steps the first solve under the thermal B is on a
        # knife edge either way (2026-09-27: keeping the spin-up velocities as
        # a warm start, or solving the first spin-up step at the isothermal B,
        # turned v10's stalled-but-finite first step into a NaN).
        for f in (mg.state.u, mg.state.v, mg.state.ud, mg.state.vd):
            f.set(0.0, start_level=lvl)
        mg.state.H.set(H0c, start_level=lvl)
        mg.state.H_prev.set(H0c, start_level=lvl)
        self._E_last = self.tm.ops.enthalpy_state.E.copy()
        T = self.tm.ops.get_temperature()
        temperate = float(((T[:, :, 0] >= self._T_pmp_bed() - 0.1) & thick).sum() / max(int(thick.sum()), 1))
        self.spinup_info = dict(steps=steps, wall_s=time_elapsed(tic), temperate_bed_fraction=temperate,
                                T_bed_mean=steps[-1][2] if steps else float("nan"))
        if cfg.report:
            print(f"thermal spin-up (level {lvl}): "
                  + ", ".join(f"{n} steps (last dT_bed {d:.1e} K, mean T_bed {tb:.2f} K)" for n, d, tb in steps)
                  + f"; temperate bed {100 * temperate:.0f} % of H>100 m; {self.spinup_info['wall_s']:.1f} s")

    def _T_pmp_bed(self):
        from glide.enthalpy import T_MELT, BETA_CC, GRAVITY
        return T_MELT - BETA_CC * self.tm.rho_i * GRAVITY * self.grid.state.H.data

    # ------------------------------------------------------------ coupled steps
    def pre_step(self, H_prev_):
        ops = self.tm.ops
        ops.enthalpy_state.E_prev[:] = ops.enthalpy_state.E
        H = H_prev_.detach() if isinstance(H_prev_, torch.Tensor) else H_prev_
        ops.H_prev[:] = cp.asarray(H, dtype=cp.float32)

    def post_step(self, dt_yr: float):
        with torch.no_grad():
            self.tm.step(float(dt_yr) * SECONDS_PER_YEAR)

    # ------------------------------------------------------------ outputs
    def temperature_fields(self) -> dict:
        """Level-grid basal and depth-averaged temperature (K, cupy); the
        average is the trapezoid rule on the uniform sigma nodes."""
        T = self.tm.ops.get_temperature()
        T_mean = (T[:, :, 1:] + T[:, :, :-1]).sum(axis=2) * (0.5 / (T.shape[2] - 1))
        return {"T_bed": T[:, :, 0], "T_mean": T_mean}

    def fields(self) -> dict:
        """Level-grid diagnostics (cupy): T_bed, T_mean (K), omega_w_bed, B (glide units)."""
        T = self.tm.ops.get_temperature()
        w = self.tm.ops.get_water_content()
        return {**self.temperature_fields(),
                "T_top": T[:, :, -1],
                "omega_w_bed": w[:, :, 0],
                "T_pmp_excess_bed": T[:, :, 0] - self._T_pmp_bed(),
                "B": self.grid.rheology.B.data.copy()}

    def state_dict(self) -> dict:
        return self.tm.state_dict()

    def load_state_dict(self, d):
        self.tm.load_state_dict(d)
        self._push()


def surface_temperature_annual(t_annual) -> cp.ndarray:
    """An annual-mean air temperature (degC, biases already in) capped at 0 degC, in K."""
    return cp.minimum(cp.asarray(t_annual, dtype=cp.float32), 0.0) + cp.float32(T0)


def time_elapsed(tic):
    return _time.time() - tic
