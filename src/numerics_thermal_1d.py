from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Literal, Tuple, Optional, Dict
import numpy as np

Array = np.ndarray
BCType = Literal["dirichlet", "neumann", "robin"]

@dataclass
class Grid1D:
    L: float
    N: int  # number of intervals; nodes = N+1
    def __post_init__(self):
        assert self.N >= 2, "N must be >= 2"
        self.h = self.L / self.N
        self.x = np.linspace(0.0, self.L, self.N + 1)

@dataclass
class TimeGrid:
    T: float
    M: int  # number of steps
    def __post_init__(self):
        assert self.M >= 1
        self.dt = self.T / self.M
        self.t = np.linspace(0.0, self.T, self.M + 1)

@dataclass
class RobinParams:
    alpha: float
    beta: float  # multiplies du/dx

@dataclass
class BC:
    kind_left: BCType
    kind_right: BCType
    g_left: Callable[[float], float]
    g_right: Callable[[float], float]
    robin_left: Optional[RobinParams] = None
    robin_right: Optional[RobinParams] = None

def _virtual_boundary_coeffs_left(bc: BC, h: float, t: float) -> Tuple[float,float,float]:
    if bc.kind_left == "dirichlet":
        g = bc.g_left(t)
        return 0.0, 0.0, g
    elif bc.kind_left == "neumann":
        q = bc.g_left(t)
        # (-3u0 + 4u1 - u2)/(2h) = q  => u0 = (4u1 - u2 - 2h q)/3
        return 4.0/3.0, -1.0/3.0, -(2.0*h/3.0)*q
    elif bc.kind_left == "robin":
        assert bc.robin_left is not None, "robin_left params required"
        p = bc.robin_left
        g = bc.g_left(t)
        den = p.alpha - 3.0*p.beta/(2.0*h)
        if abs(den) < 1e-14:
            raise ZeroDivisionError("Degenerate left Robin parameters: alpha - 3*beta/(2h) ~ 0.")
        # alpha*u0 + beta*(-3u0 + 4u1 - u2)/(2h) = g
        a = - (p.beta/(2.0*h))/den * 4.0
        b = + (p.beta/(2.0*h))/den * 1.0
        c = g/den
        return a, b, c
    else:
        raise ValueError("Unknown left BC type")

def _virtual_boundary_coeffs_right(bc: BC, h: float, t: float) -> Tuple[float,float,float]:
    if bc.kind_right == "dirichlet":
        g = bc.g_right(t)
        return 0.0, 0.0, g
    elif bc.kind_right == "neumann":
        q = bc.g_right(t)
        # (3uN - 4u_{N-1} + u_{N-2})/(2h) = q  => uN = (4u_{N-1} - u_{N-2} + 2h q)/3
        return 4.0/3.0, -1.0/3.0, (2.0*h/3.0)*q
    elif bc.kind_right == "robin":
        assert bc.robin_right is not None, "robin_right params required"
        p = bc.robin_right
        g = bc.g_right(t)
        den = p.alpha + 3.0*p.beta/(2.0*h)
        if abs(den) < 1e-14:
            raise ZeroDivisionError("Degenerate right Robin parameters: alpha + 3*beta/(2h) ~ 0.")
        # alpha*uN + beta*(3uN - 4u_{N-1} + u_{N-2})/(2h) = g
        a = + (p.beta/(2.0*h))/den * 4.0
        b = - (p.beta/(2.0*h))/den * 1.0
        c = g/den
        return a, b, c
    else:
        raise ValueError("Unknown right BC type")

def _assemble_tridiag_Dxx(N: int, h: float, bc: BC, t: float) -> Tuple[Array, Array]:
    m = N - 1  # interior unknowns
    A = np.zeros((m, m))
    rhs = np.zeros(m)

    aL, bL, cL = _virtual_boundary_coeffs_left(bc, h, t)
    aR, bR, cR = _virtual_boundary_coeffs_right(bc, h, t)

    for j in range(m):
        i = j + 1
        coef_center = -2.0 / h**2
        coef_left   =  1.0 / h**2
        coef_right  =  1.0 / h**2

        if i == 1:
            A[j, j]   += coef_center + coef_left * aL
            A[j, j+1] += coef_right  + coef_left * bL
            rhs[j]    += coef_left * cL
        elif i == N-1:
            A[j, j]   += coef_center + coef_right * aR
            A[j, j-1] += coef_left   + coef_right * bR
            rhs[j]    += coef_right * cR
        else:
            A[j, j]   += coef_center
            A[j, j-1] += coef_left
            A[j, j+1] += coef_right

    return A, rhs

def _build_identity(m: int) -> Array:
    return np.eye(m)

def solve_heat_theta(
    grid: Grid1D, tgrid: TimeGrid,
    alpha: float,
    bc: BC,
    u0: Callable[[Array], Array],
    source: Callable[[Array, float], Array] | None = None,
    theta: float = 0.5,
) -> Tuple[Array, Array, Array]:
    N, h = grid.N, grid.h
    m = N - 1
    dt = tgrid.dt
    x = grid.x
    t = tgrid.t

    U = np.zeros((t.size, x.size), dtype=float)
    U[0, :] = u0(x)

    I = _build_identity(m)
    for n in range(tgrid.M):
        tn, tnp1 = t[n], t[n+1]

        Dxx_n, rhs_n = _assemble_tridiag_Dxx(N, h, bc, tn)
        Dxx_np1, rhs_np1 = _assemble_tridiag_Dxx(N, h, bc, tnp1)

        A = I - theta * dt * alpha * Dxx_np1
        B = I + (1.0 - theta) * dt * alpha * Dxx_n

        u_inner_n = U[n, 1:N]

        rhs = B @ u_inner_n + dt * (theta * (source(x[1:N], tnp1) if source else 0.0)
                                    + (1.0 - theta) * (source(x[1:N], tn) if source else 0.0))
        rhs += -(theta * dt * alpha) * rhs_np1 + ((1.0 - theta) * dt * alpha) * rhs_n

        u_inner_np1 = np.linalg.solve(A, rhs)

        U[n+1, 1:N] = u_inner_np1
        if bc.kind_left == "dirichlet":
            U[n+1, 0] = bc.g_left(tnp1)
        else:
            aL, bL, cL = _virtual_boundary_coeffs_left(bc, h, tnp1)
            U[n+1, 0] = aL * U[n+1, 1] + bL * U[n+1, 2] + cL
        if bc.kind_right == "dirichlet":
            U[n+1, N] = bc.g_right(tnp1)
        else:
            aR, bR, cR = _virtual_boundary_coeffs_right(bc, h, tnp1)
            U[n+1, N] = aR * U[n+1, N-1] + bR * U[n+1, N-2] + cR

    return x, t, U

@dataclass
class NewmarkParams:
    gamma: float = 0.5
    beta: float = 0.25

def newmark_linear(
    M: Array, C: Array, K: Array,
    F_func: Callable[[float], Array],
    u0: Array, v0: Array,
    tgrid: TimeGrid,
    params: NewmarkParams = NewmarkParams()
) -> Tuple[Array, Array]:
    dt = tgrid.dt
    t = tgrid.t
    gamma, beta = params.gamma, params.beta
    dof = u0.size

    U = np.zeros((t.size, dof))
    V = np.zeros((t.size, dof))
    A = np.zeros((t.size, dof))

    U[0] = u0
    V[0] = v0
    F0 = F_func(t[0])
    A[0] = np.linalg.solve(M, F0 - C @ V[0] - K @ U[0])

    a0 = 1.0 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0*beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma/(2.0*beta) - 1.0)

    Keff = K + a0 * M + a1 * C

    for n in range(tgrid.M):
        tnp1 = t[n+1]
        Fnp1 = F_func(tnp1)
        rhs = (Fnp1
               + M @ (a0 * U[n] + a2 * V[n] + a3 * A[n])
               + C @ (a1 * U[n] + a4 * V[n] + a5 * A[n]))
        U[n+1] = np.linalg.solve(Keff, rhs)
        A[n+1] = a0 * (U[n+1] - U[n]) - a2 * V[n] - a3 * A[n]
        V[n+1] = V[n] + dt * ((1.0 - gamma) * A[n] + gamma * A[n+1])

    return t, U

def _assemble_operator_with_BC(N: int, h: float, bc: BC, t: float, coef: float) -> Tuple[Array, Array]:
    Dxx, rhs_dxx = _assemble_tridiag_Dxx(N, h, bc, t)
    K = -coef * Dxx
    rhs = -coef * rhs_dxx
    return K, rhs

def _reconstruct_boundaries_from_interior(u_inner: Array, N: int, h: float, bc: BC, t: float) -> Tuple[float, float]:
    if N < 3:
        aL, bL, cL = _virtual_boundary_coeffs_left(bc, h, t)
        aR, bR, cR = _virtual_boundary_coeffs_right(bc, h, t)
        u2 = u_inner[1] if u_inner.size >= 2 else u_inner[0]
        uNm2 = u_inner[-2] if u_inner.size >= 2 else u_inner[-1]
        u0 = aL * u_inner[0] + bL * u2 + cL
        uN = aR * u_inner[-1] + bR * uNm2 + cR
        return u0, uN
    aL, bL, cL = _virtual_boundary_coeffs_left(bc, h, t)
    aR, bR, cR = _virtual_boundary_coeffs_right(bc, h, t)
    u0 = aL * u_inner[0] + bL * u_inner[1] + cL
    uN = aR * u_inner[-1] + bR * u_inner[-2] + cR
    return u0, uN

def solve_maxwell_cattaneo(
    grid: Grid1D, tgrid: TimeGrid,
    tau: float, alpha: float,
    bc: BC,
    u0: Callable[[Array], Array],
    v0: Callable[[Array], Array] | None = None,
    source: Callable[[Array, float], Array] | None = None,
    source_t: Callable[[Array, float], Array] | None = None,
    newmark: NewmarkParams = NewmarkParams()
) -> Tuple[Array, Array, Array]:
    N, h = grid.N, grid.h
    m = N - 1
    x = grid.x
    t = tgrid.t

    M = tau * np.eye(m)
    C = np.eye(m)

    U = np.zeros((t.size, N+1))
    U[0, :] = u0(x)
    V0_full = v0(x) if v0 is not None else np.zeros_like(x)
    u0_inner = U[0, 1:N]
    v0_inner = V0_full[1:N]

    def F_func(tcur: float) -> Array:
        K, rhsK = _assemble_operator_with_BC(N, h, bc, tcur, coef=alpha)
        s_vec = source(x[1:N], tcur) if source is not None else 0.0
        st_vec = source_t(x[1:N], tcur) if source_t is not None else 0.0
        return rhsK + s_vec + tau * st_vec

    K0, _ = _assemble_operator_with_BC(N, h, bc, 0.0, coef=alpha)

    t_out, U_inner = newmark_linear(M, C, K0, F_func, u0_inner, v0_inner, tgrid, newmark)

    for n in range(t.size):
        u_inner = U_inner[n]
        u0_val, uN_val = _reconstruct_boundaries_from_interior(u_inner, N, h, bc, t[n])
        U[n, 0] = u0_val if bc.kind_left != "dirichlet" else bc.g_left(t[n])
        U[n, N] = uN_val if bc.kind_right != "dirichlet" else bc.g_right(t[n])
        U[n, 1:N] = u_inner

    return x, t_out, U

def solve_gn_type_II(
    grid: Grid1D, tgrid: TimeGrid,
    c_heat: float, eps: float,
    bc: BC,
    u0: Callable[[Array], Array],
    v0: Callable[[Array], Array] | None = None,
    source_t: Callable[[Array, float], Array] | None = None,
    newmark: NewmarkParams = NewmarkParams()
) -> Tuple[Array, Array, Array]:
    N, h = grid.N, grid.h
    m = N - 1
    x = grid.x
    t = tgrid.t

    M = c_heat * np.eye(m)
    C = np.zeros((m, m))
    K, _ = _assemble_operator_with_BC(N, h, bc, 0.0, coef=eps)

    U = np.zeros((t.size, N+1))
    U[0, :] = u0(x)
    V0_full = v0(x) if v0 is not None else np.zeros_like(x)
    u0_inner = U[0, 1:N]
    v0_inner = V0_full[1:N]

    def F_func(tcur: float) -> Array:
        _, rhsK = _assemble_operator_with_BC(N, h, bc, tcur, coef=eps)
        st = source_t(x[1:N], tcur) if source_t is not None else 0.0
        return rhsK + st

    t_out, U_inner = newmark_linear(M, C, K, F_func, u0_inner, v0_inner, tgrid, newmark)

    for n in range(t.size):
        u_inner = U_inner[n]
        u0_val, uN_val = _reconstruct_boundaries_from_interior(u_inner, N, h, bc, t[n])
        U[n, 0] = u0_val if bc.kind_left != "dirichlet" else bc.g_left(t[n])
        U[n, N] = uN_val if bc.kind_right != "dirichlet" else bc.g_right(t[n])
        U[n, 1:N] = u_inner

    return x, t_out, U

def solve_gn_type_III_alpha(
    grid: Grid1D, tgrid: TimeGrid,
    c_heat: float, kappa: float, eps: float,
    bc_alpha: BC,
    alpha0: Callable[[Array], Array],
    alpha_t0: Callable[[Array], Array] | None = None,
    source: Callable[[Array, float], Array] | None = None,
    newmark: NewmarkParams = NewmarkParams()
) -> Tuple[Array, Array, Array, Array]:
    N, h = grid.N, grid.h
    m = N - 1
    x = grid.x
    t = tgrid.t
    dt = tgrid.dt

    M = c_heat * np.eye(m)
    K_epsilon, _ = _assemble_operator_with_BC(N, h, bc_alpha, 0.0, coef=eps)
    C_kappa, _      = _assemble_operator_with_BC(N, h, bc_alpha, 0.0, coef=kappa)

    ALPHA = np.zeros((t.size, N+1))
    THETA = np.zeros((t.size, N+1))
    ALPHA[0, :] = alpha0(x)
    V0_full = alpha_t0(x) if alpha_t0 is not None else np.zeros_like(x)
    a0_inner = ALPHA[0, 1:N]
    v0_inner = V0_full[1:N]

    def F_func(tcur: float) -> Array:
        Keps, rhsKeps = _assemble_operator_with_BC(N, h, bc_alpha, tcur, coef=eps)
        Ckap, rhsCk  = _assemble_operator_with_BC(N, h, bc_alpha, tcur, coef=kappa)
        r_vec = source(x[1:N], tcur) if source is not None else 0.0
        solve_gn_type_III_alpha._K_current = Keps
        solve_gn_type_III_alpha._C_current = Ckap
        return r_vec + rhsKeps + rhsCk
    solve_gn_type_III_alpha._K_current = K_epsilon
    solve_gn_type_III_alpha._C_current = C_kappa

    params = newmark
    gamma, beta = params.gamma, params.beta
    t_arr = t
    dof = m
    U_inner = np.zeros((t_arr.size, dof))
    V_inner = np.zeros((t_arr.size, dof))
    A_inner = np.zeros((t_arr.size, dof))

    U_inner[0] = a0_inner
    V_inner[0] = v0_inner

    F0 = F_func(t_arr[0])
    Kcur = solve_gn_type_III_alpha._K_current
    Ccur = solve_gn_type_III_alpha._C_current
    A_inner[0] = np.linalg.solve(M, F0 - Ccur @ V_inner[0] - Kcur @ U_inner[0])

    for n in range(int(t_arr.size-1)):
        dt = tgrid.dt
        a0c = 1.0/(beta*dt*dt)
        a1c = gamma/(beta*dt)
        a2c = 1.0/(beta*dt)
        a3c = 1.0/(2.0*beta) - 1.0
        a4c = gamma/beta - 1.0
        a5c = dt*(gamma/(2.0*beta) - 1.0)

        tnp1 = t_arr[n+1]
        Fnp1 = F_func(tnp1)
        Kcur = solve_gn_type_III_alpha._K_current
        Ccur = solve_gn_type_III_alpha._C_current

        Keff = Kcur + a0c*M + a1c*Ccur
        rhs = (Fnp1
               + M @ (a0c*U_inner[n] + a2c*V_inner[n] + a3c*A_inner[n])
               + Ccur @ (a1c*U_inner[n] + a4c*V_inner[n] + a5c*A_inner[n]))
        U_inner[n+1] = np.linalg.solve(Keff, rhs)
        A_inner[n+1] = a0c*(U_inner[n+1] - U_inner[n]) - a2c*V_inner[n] - a3c*A_inner[n]
        V_inner[n+1] = V_inner[n] + dt*((1.0 - gamma)*A_inner[n] + gamma*A_inner[n+1])

    for n in range(t_arr.size):
        u_inner = U_inner[n]
        v_inner = V_inner[n]
        a0_val, aN_val = _reconstruct_boundaries_from_interior(u_inner, N, h, bc_alpha, t_arr[n])
        ALPHA[n, 0] = a0_val if bc_alpha.kind_left != "dirichlet" else bc_alpha.g_left(t_arr[n])
        ALPHA[n, N] = aN_val if bc_alpha.kind_right != "dirichlet" else bc_alpha.g_right(t_arr[n])
        ALPHA[n, 1:N] = u_inner

        t0_val, tN_val = _reconstruct_boundaries_from_interior(v_inner, N, h, bc_alpha, t_arr[n])
        THETA[n, 0] = t0_val if bc_alpha.kind_left != "dirichlet" else (0.0 if n==0 else (bc_alpha.g_left(t_arr[n]) - bc_alpha.g_left(t_arr[n-1]))/dt)
        THETA[n, N] = tN_val if bc_alpha.kind_right != "dirichlet" else (0.0 if n==0 else (bc_alpha.g_right(t_arr[n]) - bc_alpha.g_right(t_arr[n-1]))/dt)
        THETA[n, 1:N] = v_inner

    return grid.x, t_arr, ALPHA, THETA

# ---- мини-демо для быстрой самопроверки (опционально) ----

def _demo_fourier_dirichlet() -> Dict[str, Array]:
    L=1.0; N=50; T=0.1; M=100
    grid=Grid1D(L,N); time=TimeGrid(T,M)
    alpha=0.5*np.pi**2
    u_exact=lambda x,t: np.sin(np.pi*x)*np.exp(-np.pi**2*alpha*t)
    bc=BC("dirichlet","dirichlet",g_left=lambda t:0.0,g_right=lambda t:0.0)
    x,tarr,U=solve_heat_theta(grid,time,alpha,bc,u0=lambda x:u_exact(x,0.0),source=None,theta=0.5)
    exact= u_exact(x[:,None], tarr[None,:]).T
    err=np.linalg.norm(U-exact, ord=np.inf)
    return {"x":x,"t":tarr,"U":U,"err_inf":np.array([err])}

def _demo_cattaneo_dirichlet() -> Dict[str, Array]:
    L=1.0; N=80; T=0.2; M=200
    grid=Grid1D(L,N); time=TimeGrid(T,M)
    tau=0.05; alpha=1.0
    lam=5.0
    A = (tau*lam**2 - lam + alpha*np.pi**2)/(1.0 - tau*lam)
    u_exact=lambda x,t: np.sin(np.pi*x)*np.exp(-lam*t)
    s=lambda x,t: A * u_exact(x,t)
    st=lambda x,t: -lam*A * u_exact(x,t)
    bc=BC("dirichlet","dirichlet",g_left=lambda t:0.0,g_right=lambda t:0.0)
    x,tarr,U=solve_maxwell_cattaneo(grid,time,tau,alpha,bc,u0=lambda x:u_exact(x,0.0),
                                    v0=lambda x: -lam*u_exact(x,0.0), source=s, source_t=st)
    exact= u_exact(x[:,None], tarr[None,:]).T
    err=np.linalg.norm(U-exact, ord=np.inf)
    return {"x":x,"t":tarr,"U":U,"err_inf":np.array([err])}

def _demo_gn2_dirichlet() -> Dict[str, Array]:
    L=1.0; N=80; T=0.2; M=200
    grid=Grid1D(L,N); time=TimeGrid(T,M)
    c_heat=1.0; eps=1.0
    omega=8.0
    u_exact=lambda x,t: np.sin(np.pi*x)*np.cos(omega*t)
    B = (-c_heat*omega**2 + eps*np.pi**2)/omega
    rt=lambda x,t: B*omega*np.sin(np.pi*x)*np.cos(omega*t)
    bc=BC("dirichlet","dirichlet",g_left=lambda t:0.0,g_right=lambda t:0.0)
    x,tarr,U=solve_gn_type_II(grid,time,c_heat,eps,bc,u0=lambda x:u_exact(x,0.0),
                              v0=lambda x: 0.0*x, source_t=rt)
    exact= u_exact(x[:,None], tarr[None,:]).T
    err=np.linalg.norm(U-exact, ord=np.inf)
    return {"x":x,"t":tarr,"U":U,"err_inf":np.array([err])}

def _demo_gn3_alpha_dirichlet() -> Dict[str, Array]:
    L=1.0; N=80; T=0.2; M=200
    grid=Grid1D(L,N); time=TimeGrid(T,M)
    c_heat=1.0; kappa=0.7; eps=1.3
    omega=6.0
    alpha_exact=lambda x,t: np.sin(np.pi*x)*np.cos(omega*t)
    theta_exact=lambda x,t: -omega*np.sin(np.pi*x)*np.sin(omega*t)
    r=lambda x,t: (-c_heat*omega**2 + eps*np.pi**2)*alpha_exact(x,t) + kappa*np.pi**2*theta_exact(x,t)
    bc_alpha=BC("dirichlet","dirichlet",g_left=lambda t:0.0,g_right=lambda t:0.0)
    x,tarr,ALPHA,THETA=solve_gn_type_III_alpha(grid,time,c_heat,kappa,eps,bc_alpha,
                                               alpha0=lambda x:alpha_exact(x,0.0),
                                               alpha_t0=lambda x:theta_exact(x,0.0),
                                               source=r)
    exactA= alpha_exact(x[:,None], tarr[None,:]).T
    exactT= theta_exact(x[:,None], tarr[None,:]).T
    errA=np.linalg.norm(ALPHA-exactA, ord=np.inf)
    errT=np.linalg.norm(THETA-exactT, ord=np.inf)
    return {"x":x,"t":tarr,"ALPHA":ALPHA,"THETA":THETA,"err_inf_alpha":np.array([errA]),"err_inf_theta":np.array([errT])}
