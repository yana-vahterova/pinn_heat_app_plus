from __future__ import annotations
import os, sys
sys.path.append(os.path.dirname(__file__))
import numpy as np
from typing import Callable, Dict, Optional, Tuple
from numerics_thermal_1d import (
    Grid1D, TimeGrid, BC, RobinParams,
    solve_heat_theta, solve_maxwell_cattaneo,
    solve_gn_type_II, solve_gn_type_III_alpha,
)

Array = np.ndarray

def _time_only_from_xt(fn_xt: Optional[Callable]) -> Callable[[float], float]:
    if fn_xt is None:
        return lambda t: 0.0
    def g(t: float) -> float:
        try:
            val = fn_xt(np.array([[0.0]]), np.array([[t]]))
        except TypeError:
            val = fn_xt(np.array([[t]])) if callable(fn_xt) else 0.0
        arr = np.asarray(val)
        return float(arr.reshape(-1)[0])
    return g

def _u0_from_ic(ic_fun: Optional[Callable]):
    if ic_fun is None:
        return lambda xvec: np.zeros_like(xvec)
    def u0(xvec: np.ndarray) -> np.ndarray:
        out = []
        for xx in xvec:
            try:
                val = ic_fun(np.array([[xx]]), np.array([[0.0]]))
            except TypeError:
                val = ic_fun(np.array([[xx]]))
            arr = np.asarray(val)
            out.append(float(arr.reshape(-1)[0]))
        return np.array(out, dtype=float)
    return u0

def _v0_from_ict(ict_fun: Optional[Callable]):
    if ict_fun is None:
        return lambda xvec: np.zeros_like(xvec)
    def v0(xvec: np.ndarray) -> np.ndarray:
        out = []
        for xx in xvec:
            try:
                val = ict_fun(np.array([[xx]]), np.array([[0.0]]))
            except TypeError:
                val = ict_fun(np.array([[xx]]))
            arr = np.asarray(val)
            out.append(float(arr.reshape(-1)[0]))
        return np.array(out, dtype=float)
    return v0

def _source_from_q(q_fun: Optional[Callable]):
    if q_fun is None:
        return None
    def s(xvec: np.ndarray, t: float) -> np.ndarray:
        vals = []
        for xx in xvec:
            vv = q_fun(np.array([[xx]]), np.array([[t]]))
            arr = np.asarray(vv)
            vals.append(float(arr.reshape(-1)[0]))
        return np.array(vals, dtype=float)
    return s

def _source_t_from_q(q_fun: Optional[Callable]):
    if q_fun is None:
        return None
    def st(xvec: np.ndarray, t: float) -> np.ndarray:
        vals = []
        for xx in xvec:
            vv = q_fun(np.array([[xx]]), np.array([[t]]))
            arr = np.asarray(vv)
            vals.append(float(arr.reshape(-1)[0]))
        return np.array(vals, dtype=float)
    return st

def _make_bc(bc_types: Dict[str,str], gL: Callable[[float], float], gR: Callable[[float], float]) -> BC:
    map_type = {"dirichlet":"dirichlet", "neumann":"neumann", "robin":"robin"}
    kl = map_type.get(bc_types.get("left","Dirichlet").lower(), "dirichlet")
    kr = map_type.get(bc_types.get("right","Dirichlet").lower(), "dirichlet")
    rL = RobinParams(alpha=1.0, beta=1.0) if kl=="robin" else None
    rR = RobinParams(alpha=1.0, beta=1.0) if kr=="robin" else None
    return BC(kl, kr, g_left=gL, g_right=gR, robin_left=rL, robin_right=rR)

def solve_numeric(
    model_name: str,
    params: Dict[str, float],
    domain: Dict[str, float],
    bc_types: Dict[str, str],
    ic_fun: Optional[Callable] = None,
    ict_fun: Optional[Callable] = None,
    bc_left_fun: Optional[Callable] = None,
    bc_right_fun: Optional[Callable] = None,
    q_fun: Optional[Callable] = None,
    nx: int = 201, nt: int = 401, device: str = "cpu", method: str = "cn"
) -> Tuple[np.ndarray, Dict]:
    x_min = float(domain.get("x_min", 0.0)); x_max = float(domain.get("x_max", 1.0))
    t_min = float(domain.get("t_min", 0.0)); t_max = float(domain.get("t_max", 1.0))

    grid = Grid1D(L=x_max-x_min, N=nx-1)
    tgrid = TimeGrid(T=t_max-t_min, M=nt-1)

    gL = _time_only_from_xt(bc_left_fun)
    gR = _time_only_from_xt(bc_right_fun)
    bc = _make_bc(bc_types, gL, gR)

    name = model_name.strip()
    info = {"method": method, "bc": dict(bc_types), "params": dict(params), "domain": dict(domain)}

    if name == "Fourier (Classical)":
        alpha = float(params.get("kappa", params.get("alpha", 1.0)))
        theta = 0.5 if method.lower() in ("cn","crank–nicolson","crank-nicolson") else 1.0
        x, t, U = solve_heat_theta(grid, tgrid, alpha, bc,
                                   u0=_u0_from_ic(ic_fun),
                                   source=_source_from_q(q_fun),
                                   theta=theta)
    elif name == "Maxwell–Cattaneo":
        tau = float(params.get("tau_R", params.get("tau", 1.0)))
        alpha = float(params.get("kappa", 1.0))
        x, t, U = solve_maxwell_cattaneo(grid, tgrid, tau, alpha, bc,
                                         u0=_u0_from_ic(ic_fun),
                                         v0=_v0_from_ict(ict_fun),
                                         source=_source_from_q(q_fun),
                                         source_t=None)   # при необходимости можно добавить поле s_t в UI
    elif name == "Green–Naghdi II":
        c_heat = 1.0
        eps = float(params.get("kappa_tilde", params.get("eps", 1.0)))
        x, t, U = solve_gn_type_II(grid, tgrid, c_heat, eps, bc,
                                   u0=_u0_from_ic(ic_fun),
                                   v0=_v0_from_ict(ict_fun),
                                   source_t=_source_t_from_q(q_fun))  # q трактуется как r_t
    elif name == "Green–Naghdi III":
        c_heat = 1.0
        kappa = float(params.get("kappa", 0.0))
        eps = float(params.get("kappa_tilde", 1.0))
        x, t, ALPHA, THETA = solve_gn_type_III_alpha(grid, tgrid, c_heat, kappa, eps, bc,
                                                     alpha0=_u0_from_ic(ic_fun),
                                                     alpha_t0=_v0_from_ict(ict_fun),
                                                     source=_source_from_q(q_fun))  # q трактуется как r
        U = THETA  # температура
    else:
        raise ValueError(f"Unknown model name: {name}")

    if U.shape == (len(t), len(grid.x)):
        U = U.T
    return U, info
