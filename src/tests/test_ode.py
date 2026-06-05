"""Sanity tests for the mean-field ODE module."""

from __future__ import annotations

import numpy as np
import pytest

from src import market, ode
from src.config import DemandParams


def baseline_demand() -> DemandParams:
    return DemandParams.symmetric(N=2, alpha=2.5, beta=-1.0, gamma=0.4, l=0.5, u=2.5, noise_std=0.2)


def test_ab_at_nash_with_collusive_Q_recovers_nash():
    """When the moments are at the ODE equilibrium ``(p^{NE}, p^{NE} p^{NE}^T + nu^2 I)``,
    the greedy price should be exactly ``p^{NE}``."""
    d = baseline_demand()
    p_NE = market.nash_prices(d)
    nu_squared = 0.05
    m = p_NE.copy()
    Q = np.outer(p_NE, p_NE) + nu_squared * np.eye(d.N)
    p_g = ode.greedy_price(m, Q, d, project=False)
    np.testing.assert_allclose(p_g, p_NE, rtol=1e-10, atol=1e-12)


def test_integrate_converges_to_nash():
    """Long-time integration should land on ``(p^{NE}, p^{NE} p^{NE}^T + nu^2 I)``."""
    d = baseline_demand()
    p_NE = market.nash_prices(d)
    nu_squared = 0.05
    # Start away from equilibrium.
    m0 = np.array([0.7, 0.8])
    Q0 = np.array([[0.8, 1.2], [1.2, 0.9]])
    sol = ode.integrate(d, nu_squared, m0=m0, Q0=Q0, t_max=20.0, n_points=2001, project=True)
    np.testing.assert_allclose(sol.m[-1], p_NE, atol=2e-3)
    Q_target = np.outer(p_NE, p_NE) + nu_squared * np.eye(d.N)
    np.testing.assert_allclose(sol.Q[-1], Q_target, atol=2e-3)


def test_state_pack_unpack_roundtrip():
    rng = np.random.default_rng(0)
    N = 3
    m = rng.uniform(size=N)
    Q = rng.normal(size=(N, N))
    Q = (Q + Q.T) / 2  # symmetric
    state = ode.ODEState(m=m, Q=Q)
    v = ode.state_to_vector(state)
    state2 = ode.vector_to_state(v, N)
    np.testing.assert_allclose(state2.m, m)
    np.testing.assert_allclose(state2.Q, Q)


@pytest.mark.parametrize("init_m", [(0.7, 0.8), (1.5, 1.5)])
def test_integrate_duopoly_no_blow_up(init_m):
    d = baseline_demand()
    sol = ode.integrate_duopoly(
        d,
        nu_squared=0.013,
        m1_0=init_m[0],
        m2_0=init_m[1],
        Q11_0=init_m[0] ** 2 + 0.013,
        Q22_0=init_m[1] ** 2 + 0.013,
        Q12_0=init_m[0] * init_m[1],
        t_max=10.0,
        n_points=500,
        project=True,
    )
    assert np.all(np.isfinite(sol.m))
    assert np.all(np.isfinite(sol.Q))
    assert np.all(sol.m >= d.l - 1e-9)
    assert np.all(sol.m <= d.u + 1e-9)


def test_ode_time_to_n():
    n = np.array([1, 10, 100, 1000])
    t = ode.discrete_time_to_ode_time(n)
    assert t[0] == pytest.approx(np.log(1) + 0.5772156649)
    assert t[1] == pytest.approx(np.log(10) + 0.5772156649)
