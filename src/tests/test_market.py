"""Sanity tests for closed-form market quantities."""

from __future__ import annotations

import numpy as np
import pytest

from src import market
from src.config import DemandParams, ProjectionBox


@pytest.fixture
def baseline_duopoly() -> DemandParams:
    return DemandParams.symmetric(N=2, alpha=2.5, beta=-1.0, gamma=0.4, l=0.5, u=2.5, noise_std=0.2)


def test_nash_matches_baseline(baseline_duopoly: DemandParams) -> None:
    p = market.nash_prices(baseline_duopoly)
    assert np.allclose(p, [25 / 16, 25 / 16])  # 1.5625
    # Reference rounds to 1.56 -- check to 2 decimals.
    assert np.allclose(np.round(p, 2), [1.56, 1.56])


def test_collusive_matches_baseline(baseline_duopoly: DemandParams) -> None:
    p = market.collusive_prices(baseline_duopoly)
    # alpha + 2*beta*p + (gamma + gamma)*p = 0  =>  2.5 + 2(-1)p + 2*0.4 p = 0  =>  p = 25/12
    assert np.allclose(p, [25 / 12, 25 / 12])
    assert np.allclose(np.round(p, 2), [2.08, 2.08])


def test_stackelberg_ordering(baseline_duopoly: DemandParams) -> None:
    p_NE = market.nash_prices(baseline_duopoly)
    p_C = market.collusive_prices(baseline_duopoly)
    p1, p2 = market.stackelberg_duopoly(baseline_duopoly)
    # In the symmetric case: p_NE < p_2 < p_1 < p_C.
    assert p_NE[0] < p2 < p1 < p_C[0]


def test_pseudo_true_oblivious_at_NE(baseline_duopoly: DemandParams) -> None:
    p_NE = market.nash_prices(baseline_duopoly)
    target = market.pseudo_true_oblivious(baseline_duopoly, p_NE)
    # At NE, the misspecified greedy phi^{ob}(theta^{*,ob}) should equal p_NE.
    a, b = target[0]
    assert np.isclose(-a / (2 * b), p_NE[0])


def test_gamma_bar_symmetric(baseline_duopoly: DemandParams) -> None:
    assert np.isclose(market.gamma_bar(baseline_duopoly), 0.4)


def test_L_phi_oblivious_box(baseline_duopoly: DemandParams) -> None:
    box = ProjectionBox.from_demand(baseline_duopoly)
    L = market.L_phi_oblivious(box)
    # L is positive and finite.
    assert L > 0 and np.isfinite(L)


def test_per_period_revenue_at_NE_collusive(baseline_duopoly: DemandParams) -> None:
    p_NE = market.nash_prices(baseline_duopoly)
    p_C = market.collusive_prices(baseline_duopoly)
    rev_NE = market.per_period_revenue(baseline_duopoly, p_NE)
    rev_C = market.per_period_revenue(baseline_duopoly, p_C)
    # Collusive revenue should dominate (joint-revenue maximization).
    assert (rev_NE.sum() < rev_C.sum())


def test_predicted_regime_categories(baseline_duopoly: DemandParams) -> None:
    box = ProjectionBox.from_demand(baseline_duopoly)
    info_below = market.predicted_regime(baseline_duopoly, box, nu_squared=1e-3)
    assert info_below["regime"] == "below"
    # For a market with small gamma, the threshold is small enough that
    # large nu^2 puts us in the "fast" regime even with C_M <= 1.
    weak = DemandParams.symmetric(
        N=2, alpha=2.5, beta=-1.0, gamma=0.01, l=0.5, u=2.5, noise_std=0.2
    )
    weak_box = ProjectionBox.from_demand(weak)
    info_fast = market.predicted_regime(weak, weak_box, nu_squared=10.0)
    assert info_fast["regime"] == "fast", info_fast


def test_cm_upper_bound_monotone(baseline_duopoly: DemandParams) -> None:
    cm_small = market.cm_upper_bound(baseline_duopoly, nu_squared=1e-3)
    cm_large = market.cm_upper_bound(baseline_duopoly, nu_squared=10.0)
    assert 0 < cm_small < cm_large
    # Upper-bound is bounded above by 1 (matches V -> infty limit).
    assert market.cm_upper_bound(baseline_duopoly, nu_squared=1e6) <= 1.0 + 1e-6


def test_asymmetric_demand_solvable() -> None:
    d = DemandParams(
        N=2,
        alpha=[2.5, 3.0],
        beta=[-1.0, -1.2],
        gamma=[[0.0, 0.4], [0.5, 0.0]],
        l=0.5,
        u=2.5,
        noise_std=0.2,
    )
    p_NE = market.nash_prices(d)
    p_C = market.collusive_prices(d)
    assert (p_NE > 0).all() and (p_C > p_NE).all()


def test_validation_rejects_bad_dominance() -> None:
    with pytest.raises(ValueError):
        DemandParams.symmetric(N=2, alpha=1.0, beta=-0.3, gamma=0.4, l=0.5, u=2.5, noise_std=0.2)
