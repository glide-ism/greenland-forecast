"""
Flux through Mankoff et al.'s discharge gates under the RASTER-CHAIN
convention (the one that reproduces Mankoff's 494.5 Gt/yr from the native
ITS_LIVE x BedMachine products to 3 %; CLAUDE.md, "Gate integral
convention"): the gpkg's rasterized gate polygons are expanded to their 200 m
pixels, each gate's pixels are walked nearest-neighbour into a chain, every
chain segment gets its own normal (oriented along the observed flow) and its
true length (polar-stereographic scale corrected), and the flux is
rho_i sum max(v . n, 0) H L over the segment midpoints, sampling the fields
at the nearest model cell.

    G = ChainGates(gpkg, x, y, vx_obs, vy_obs)
    G.flux(U, V, H)            -> DataFrame per gate: gate, region, name, D (Gt/yr)
    G.by_gate_ids(df, ids)     -> the sum of D over a list of gate ids

Shared by analysis/sweep_calving_eval.py (J_gate) and the scratch gate
comparisons.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RHO_I = 917.0
PIX = 200.0
LAT_TS = 70.0


class ChainGates:
    def __init__(self, gpkg, x, y, vx_obs, vy_obs):
        import geopandas as gpd
        from shapely.geometry import Point
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)
        self.dx, self.dy = self.x[1] - self.x[0], self.y[1] - self.y[0]
        g = gpd.read_file(Path(gpkg))
        px = []
        for _, r in g.iterrows():
            x0, y0, x1, y1 = r.geometry.bounds
            poly = r.geometry.buffer(1.0)
            for xx in np.arange(np.floor(x0 / PIX) * PIX + PIX / 2, x1, PIX):
                for yy in np.arange(np.floor(y0 / PIX) * PIX + PIX / 2, y1, PIX):
                    if poly.contains(Point(xx, yy)):
                        px.append((int(r.gate), r.region, r.Mouginot_2019, r.mean_lat, xx, yy))
        P = pd.DataFrame(px, columns=["gate", "region", "name", "lat", "x", "y"])
        ovx, ovy = np.nan_to_num(np.asarray(vx_obs, float)), np.nan_to_num(np.asarray(vy_obs, float))
        self.chains = []
        for gid, s in P.groupby("gate"):
            xs, ys = s.x.values, s.y.values
            C = np.c_[xs, ys] - np.c_[xs, ys].mean(0)
            t = np.linalg.svd(C, full_matrices=False)[2][0] if len(s) > 1 else np.array([1.0, 0.0])
            order = np.argsort(C @ t)
            n_ = len(xs)
            vis = np.zeros(n_, bool)
            ch = [int(order[0])]
            vis[ch[0]] = True
            for _ in range(n_ - 1):
                d2 = (xs - xs[ch[-1]]) ** 2 + (ys - ys[ch[-1]]) ** 2
                d2[vis] = np.inf
                k = int(np.argmin(d2))
                ch.append(k)
                vis[k] = True
            xc, yc = xs[np.array(ch)], ys[np.array(ch)]
            if n_ > 1:
                seg = np.c_[xc[1:] - xc[:-1], yc[1:] - yc[:-1]]
                L = np.hypot(seg[:, 0], seg[:, 1])
                ok = L <= 300.0                         # a jump between pixel clusters is not gate length
                mx, my = 0.5 * (xc[1:] + xc[:-1]), 0.5 * (yc[1:] + yc[:-1])
                nseg = np.c_[-seg[:, 1], seg[:, 0]] / np.maximum(L, 1e-9)[:, None]
            else:
                mx, my, L, ok, nseg = xs, ys, np.array([PIX]), np.array([True]), np.array([[-t[1], t[0]]])
            iy, ix = self._idx(mx, my)
            ref = np.c_[ovx[iy, ix], ovy[iy, ix]]
            sg = np.sign((ref * nseg).sum(1))
            sg[sg == 0] = 1
            nseg = nseg * sg[:, None]
            klat = (1 + np.sin(np.radians(LAT_TS))) / (1 + np.sin(np.radians(s.lat.values[0])))
            self.chains.append((int(gid), s.region.iloc[0], s.name.iloc[0], iy, ix, nseg, L * ok / klat))

    def _idx(self, xs, ys):
        iy = np.clip(np.rint((ys - self.y[0]) / self.dy).astype(int), 0, len(self.y) - 1)
        ix = np.clip(np.rint((xs - self.x[0]) / self.dx).astype(int), 0, len(self.x) - 1)
        return iy, ix

    def flux(self, U, V, H) -> pd.DataFrame:
        """Per-gate discharge (Gt/yr) of the cell-centred velocity (U, V) (m/yr)
        and thickness H (m) on the grid given at construction."""
        rows = []
        for gid, reg, nme, iy, ix, nseg, w in self.chains:
            vn = U[iy, ix] * nseg[:, 0] + V[iy, ix] * nseg[:, 1]
            rows.append(dict(gate=gid, region=reg, name=nme,
                             D=float((RHO_I * np.clip(vn, 0, None) * H[iy, ix] * w).sum() / 1e12)))
        return pd.DataFrame(rows)

    def factors(self, U, V, H, vx_obs, vy_obs, H_obs) -> pd.DataFrame:
        """Per gate the sums behind the FLUX-WEIGHTED speed and thickness
        ratios, with weights w = L |v_obs| H_obs (the observed flux density):
        s_m = sum w |v|, s_o = sum w |v_obs|, h_m = sum w H, h_o = sum w H_obs.
        Speed ratio s_m / s_o, thickness ratio h_m / h_o over any set of gates."""
        osp = np.hypot(np.nan_to_num(vx_obs), np.nan_to_num(vy_obs))
        Ho = np.nan_to_num(H_obs)
        rows = []
        for gid, reg, nme, iy, ix, nseg, w in self.chains:
            ww = w * osp[iy, ix] * Ho[iy, ix]
            rows.append(dict(gate=gid, s_m=float((ww * np.hypot(U[iy, ix], V[iy, ix])).sum()), s_o=float((ww * osp[iy, ix]).sum()),
                             h_m=float((ww * H[iy, ix]).sum()), h_o=float((ww * Ho[iy, ix]).sum())))
        return pd.DataFrame(rows).set_index("gate")

    @staticmethod
    def by_gate_ids(df: pd.DataFrame, ids) -> float:
        return float(df.set_index("gate").D.reindex(list(ids)).fillna(0.0).sum())
