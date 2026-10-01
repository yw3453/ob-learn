"""Tests for revenue benchmarks."""

from __future__ import annotations

import numpy as np

from ob_learn import benchmarks
from ob_learn.config import (
    DemandParams,
    ExperimentConfig,
    ExplorationSchedule,
    InformedProjectionBox,
    ProjectionBox,
    SellerSpec,
)
from ob_learn.simulator import run_simulation


def _baseline_demand() -> DemandParams:
    return DemandParams.symmetric(N=2, alpha=2.5, beta=-1.0, gamma=0.4, l=0.5, u=2.5, noise_std=0.2)


def test_per_period_revenue_collusive_dominates_NE() -> None:
    d = _baseline_demand()
    pi = benchmarks.benchmark_per_period_revenues(d)
    assert pi["collusive"].sum() > pi["NE"].sum()
    assert "stackelberg" in pi


def test_learning_regret_zero_when_greedy_is_best_response() -> None:
    """Learning regret vanishes when the greedy price equals the oracle BR.

    Overwrite the recorded greedy prices with the exact best response to the
    greedy rival profile; the per-period and cumulative learning regret must
    then be identically zero (up to floating point).
    """
    d = _baseline_demand()
    sched = ExplorationSchedule(kind="constant", nu=0.4)
    cfg = ExperimentConfig(
        name="test_learn_regret",
        market=d,
        sellers=[SellerSpec(kind="oblivious", exploration=sched) for _ in range(2)],
        oblivious_projection=ProjectionBox.from_demand(d),
        informed_projection=InformedProjectionBox.from_demand(d),
        horizon=200,
        n_seeds=6,
        base_seed=1,
        log_every=1,
    )
    res = run_simulation(cfg)
    # At the Nash profile every seller's greedy price is its own oracle best
    # response (a fixed point), so the learning regret must vanish exactly.
    from ob_learn import market

    p_NE = market.nash_prices(d)
    res.tilde_p[:] = p_NE[None, :, None]
    g = benchmarks.per_period_learning_regret(res, d)
    cum = benchmarks.cumulative_learning_regret(res, d)
    assert np.allclose(g, 0.0, atol=1e-10)
    assert np.allclose(cum, 0.0, atol=1e-8)


def test_learning_regret_sublinear_below_realized() -> None:
    """With persistent exploration, learning regret < realized regret (tax).

    The greedy prices converge to Nash so the learning regret is sublinear,
    while the realized-price regret carries a persistent ``|beta| nu^2`` floor;
    hence the cumulative learning regret must be strictly below the cumulative
    realized regret at the final horizon.
    """
    d = _baseline_demand()
    sched = ExplorationSchedule(kind="constant", nu=0.3)
    cfg = ExperimentConfig(
        name="test_learn_vs_realized",
        market=d,
        sellers=[SellerSpec(kind="oblivious", exploration=sched) for _ in range(2)],
        oblivious_projection=ProjectionBox.from_demand(d),
        informed_projection=InformedProjectionBox.from_demand(d),
        horizon=2_000,
        n_seeds=8,
        base_seed=2,
        log_every=1,
    )
    res = run_simulation(cfg)
    learn = benchmarks.cumulative_learning_regret(res, d).sum(axis=1).mean(axis=1)
    realized = benchmarks.cumulative_regret(res, d).sum(axis=1).mean(axis=1)
    assert learn[-1] < realized[-1]
    assert (learn >= -1e-9).all()


def test_cumulative_revenue_monotone() -> None:
    d = _baseline_demand()
    sched = ExplorationSchedule(kind="constant", nu=0.4)
    cfg = ExperimentConfig(
        name="test_cum",
        market=d,
        sellers=[SellerSpec(kind="oblivious", exploration=sched) for _ in range(2)],
        oblivious_projection=ProjectionBox.from_demand(d),
        informed_projection=InformedProjectionBox.from_demand(d),
        horizon=400,
        n_seeds=8,
        base_seed=0,
        log_every=1,
    )
    res = run_simulation(cfg)
    cum = benchmarks.cumulative_revenue(res)
    # Cumulative should be non-decreasing in time (price * demand can be
    # negative if demand < 0; for our parameters it stays positive in
    # expectation but stochastically may dip slightly. Test only the
    # average across seeds).
    avg = cum.mean(axis=2)
    diffs = np.diff(avg, axis=0)
    # Allow some negative dips but the overwhelming majority should be positive.
    assert (diffs > 0).mean() > 0.9
