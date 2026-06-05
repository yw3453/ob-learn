"""Mean-field ODE for the empirical-moment recursion.

For ``N`` sellers, the moment system has dimension ``2N + N(N-1)/2``:

* ``m_i`` -- empirical mean price of seller ``i`` (``N`` variables).
* ``Q_{ii}`` -- empirical second moment of seller ``i``'s price (``N`` variables).
* ``Q_{ij}`` for ``i < j`` -- empirical cross-moments (``N(N-1)/2`` variables).

In the symmetric duopoly case (``N=2``) this collapses to the 5-D system
``(m_1, m_2, Q_{11}, Q_{22}, Q_{12})`` used by the excursion experiment.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .config import DemandParams


def _ab_from_moments(
    m: np.ndarray,
    Q: np.ndarray,
    d: DemandParams,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form misspecified OLS coefficients ``(a_i, b_i)``.

    Returns arrays of shape ``(N,)``.
    """
    G = d.gamma_arr  # (N, N), zero diagonal
    alpha = d.alpha_arr
    beta = d.beta_arr
    V = np.diag(Q) - m**2  # (N,)
    V = np.maximum(V, eps)
    S = Q - np.outer(m, m)  # (N, N)
    R = S / V[:, None]  # R[i, j] = S_{ij} / V_i  (note: own-row variance)
    np.fill_diagonal(R, 0.0)
    sum_gR = np.einsum("ij,ij->i", G, R)  # Σ_{j != i} γ_{i,j} S_{ij} / V_i
    b = beta + sum_gR
    a = alpha + G @ m - m * sum_gR
    return a, b


def greedy_price(
    m: np.ndarray,
    Q: np.ndarray,
    d: DemandParams,
    *,
    project: bool = True,
) -> np.ndarray:
    """Mean-field greedy price ``p_i^g(m, Q) = -a_i / (2 b_i)``.

    Parameters
    ----------
    project : bool
        If ``True``, clip to ``[d.l, d.u]``. The local analysis omits
        projection; set ``project=False`` to match it exactly.
    """
    a, b = _ab_from_moments(m, Q, d)
    p = -a / (2.0 * b)
    if project:
        p = np.clip(p, d.l, d.u)
    return p


@dataclass(frozen=True)
class ODEState:
    m: np.ndarray  # (N,)
    Q: np.ndarray  # (N, N)


def state_to_vector(state: ODEState) -> np.ndarray:
    """Pack ``(m, Q)`` into a flat 1-D vector. Q is stored as upper triangle."""
    N = state.m.size
    iu = np.triu_indices(N)
    return np.concatenate([state.m, state.Q[iu]])


def vector_to_state(v: np.ndarray, N: int) -> ODEState:
    """Inverse of :func:`state_to_vector`."""
    m = v[:N]
    Q = np.zeros((N, N), dtype=np.float64)
    iu = np.triu_indices(N)
    Q[iu] = v[N:]
    Q = Q + Q.T - np.diag(np.diag(Q))
    return ODEState(m=m, Q=Q)


def ode_rhs(
    t: float,
    v: np.ndarray,
    d: DemandParams,
    nu_squared: float,
    *,
    project: bool = True,
) -> np.ndarray:
    """Right-hand side of the mean-field ODE."""
    state = vector_to_state(v, d.N)
    m, Q = state.m, state.Q
    p_g = greedy_price(m, Q, d, project=project)
    dm = p_g - m
    dQ = np.outer(p_g, p_g) - Q
    dQ[np.diag_indices_from(dQ)] += nu_squared
    iu = np.triu_indices(d.N)
    return np.concatenate([dm, dQ[iu]])


@dataclass(frozen=True)
class ODESolution:
    """Container for an ODE integration."""
    t: np.ndarray  # (T,)
    m: np.ndarray  # (T, N)
    Q: np.ndarray  # (T, N, N)
    p_g: np.ndarray  # (T, N) -- mean-field greedy price along trajectory


def integrate(
    d: DemandParams,
    nu_squared: float,
    *,
    m0: np.ndarray,
    Q0: np.ndarray,
    t_max: float,
    n_points: int = 2001,
    project: bool = True,
    method: str = "LSODA",
) -> ODESolution:
    """Integrate the mean-field ODE from initial moments ``(m0, Q0)``.

    Parameters
    ----------
    t_max : float
        Final ODE time. On the relevant time scale ``t = log n`` in discrete
        time, so ``t_max = 10`` covers ``n ~ e^{10} ~ 22000`` periods.
    n_points : int
        Number of dense output points returned (uniform in ``t``).
    project : bool
        Whether to clip ``p_g`` to ``[d.l, d.u]`` along the trajectory. The
        local analysis omits projection; ``project=False`` mirrors it.
    """
    v0 = state_to_vector(ODEState(m=np.asarray(m0, dtype=np.float64), Q=np.asarray(Q0, dtype=np.float64)))
    sol = solve_ivp(
        fun=lambda t, v: ode_rhs(t, v, d, nu_squared, project=project),
        t_span=(0.0, float(t_max)),
        y0=v0,
        method=method,
        t_eval=np.linspace(0.0, float(t_max), int(n_points)),
        rtol=1e-8,
        atol=1e-10,
        max_step=0.05,
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")
    T = sol.t.size
    N = d.N
    iu = np.triu_indices(N)
    m_traj = sol.y[:N, :].T  # (T, N)
    Q_traj = np.zeros((T, N, N))
    for k in range(T):
        Qk = np.zeros((N, N))
        Qk[iu] = sol.y[N:, k]
        Qk = Qk + Qk.T - np.diag(np.diag(Qk))
        Q_traj[k] = Qk
    p_g_traj = np.zeros((T, N))
    for k in range(T):
        p_g_traj[k] = greedy_price(m_traj[k], Q_traj[k], d, project=project)
    return ODESolution(t=sol.t, m=m_traj, Q=Q_traj, p_g=p_g_traj)


def initial_Q_from_prices(prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: ``(m, Q)`` derived from a list of two warm-up price vectors.

    Used to give the discrete-time simulator and ODE consistent initial
    conditions when the discrete simulator's warm-up is two periods long.
    """
    P = np.asarray(prices, dtype=np.float64)  # (k, N)
    m = P.mean(axis=0)
    Q = (P.T @ P) / P.shape[0]
    return m, Q


def make_initial_state(
    *,
    m0: np.ndarray,
    Q_diag: np.ndarray,
    Q_offdiag: float | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(m, Q)`` from the per-component initial values commonly used in
    the baseline plot captions.

    For a duopoly, pass ``m0=[m_1(0), m_2(0)]``, ``Q_diag=[Q_{11}(0), Q_{22}(0)]``
    and ``Q_offdiag=Q_{12}(0)``.
    """
    m = np.asarray(m0, dtype=np.float64)
    N = m.size
    Q = np.diag(np.asarray(Q_diag, dtype=np.float64))
    if Q_offdiag is None:
        return m, Q
    if np.isscalar(Q_offdiag):
        if isinstance(Q_offdiag, (str, bytes, memoryview, complex, np.complexfloating)):
            raise TypeError("Q_offdiag scalar must be numeric")
        Q = Q + (np.ones((N, N)) - np.eye(N)) * float(Q_offdiag)
    else:
        offdiag = np.asarray(Q_offdiag, dtype=np.float64)
        if offdiag.shape != (N, N):
            raise ValueError(f"Q_offdiag has shape {offdiag.shape}; expected ({N},{N})")
        np.fill_diagonal(offdiag, 0.0)
        Q = np.diag(np.diag(Q)) + offdiag
    return m, Q


# ---------------------------------------------------------------------------
# Convenience wrappers used by the experiment scripts
# ---------------------------------------------------------------------------


def integrate_duopoly(
    d: DemandParams,
    nu_squared: float,
    *,
    m1_0: float,
    m2_0: float,
    Q11_0: float,
    Q22_0: float,
    Q12_0: float,
    t_max: float,
    n_points: int = 2001,
    project: bool = True,
) -> ODESolution:
    """Symmetric/asymmetric duopoly 5-D integration matching the baseline notation."""
    if d.N != 2:
        raise ValueError("integrate_duopoly requires N=2")
    m0 = np.array([m1_0, m2_0])
    Q0 = np.array([[Q11_0, Q12_0], [Q12_0, Q22_0]])
    return integrate(d, nu_squared, m0=m0, Q0=Q0, t_max=t_max, n_points=n_points, project=project)


def discrete_time_to_ode_time(n_periods: np.ndarray) -> np.ndarray:
    """Map discrete-time step ``n`` to ODE time ``t = sum_{k <= n} 1/k = log n + O(1)``.

    Uses ``log(n) + Euler-Mascheroni`` for ``n >= 1``.
    """
    n = np.asarray(n_periods, dtype=np.float64)
    return np.log(np.maximum(n, 1.0)) + 0.5772156649


T_FROM_N: Callable[[np.ndarray], np.ndarray] = discrete_time_to_ode_time
