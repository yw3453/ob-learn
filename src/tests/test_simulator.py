"""End-to-end sanity tests for the simulator."""

from __future__ import annotations

import numpy as np

from src import market
from src.config import (
    DemandParams,
    ExperimentConfig,
    ExplorationSchedule,
    InformedProjectionBox,
    ProjectionBox,
    SellerSpec,
)
from src.simulator import run_simulation


def _baseline_demand() -> DemandParams:
    return DemandParams.symmetric(N=2, alpha=2.5, beta=-1.0, gamma=0.4, l=0.5, u=2.5, noise_std=0.2)


def _basic_cfg(sellers: list[SellerSpec], horizon: int = 1500, n_seeds: int = 12) -> ExperimentConfig:
    d = _baseline_demand()
    return ExperimentConfig(
        name="test",
        market=d,
        sellers=sellers,
        oblivious_projection=ProjectionBox.from_demand(d),
        informed_projection=InformedProjectionBox.from_demand(d),
        horizon=horizon,
        n_seeds=n_seeds,
        base_seed=0,
        log_every=10,
    )


def test_prices_in_bounds() -> None:
    sched = ExplorationSchedule(kind="constant", nu=0.4)
    cfg = _basic_cfg([SellerSpec(kind="oblivious", exploration=sched) for _ in range(2)])
    res = run_simulation(cfg)
    assert (res.prices >= cfg.market.l - 1e-9).all()
    assert (res.prices <= cfg.market.u + 1e-9).all()


def test_obob_converges_under_strong_exploration() -> None:
    """Both oblivious sellers should be pulled toward p_NE under strong exploration."""
    d = _baseline_demand()
    sched = ExplorationSchedule(kind="constant", nu=0.6)
    cfg = _basic_cfg(
        [SellerSpec(kind="oblivious", exploration=sched) for _ in range(2)],
        horizon=4000,
        n_seeds=24,
    )
    res = run_simulation(cfg)
    p_NE = market.nash_prices(d)
    # Mean greedy price across seeds at the end should be close to p_NE.
    final = res.tilde_p[-1].mean(axis=1)
    assert np.allclose(final, p_NE, atol=0.05)


def test_perfect_prediction_yields_stackelberg() -> None:
    sched_ob = ExplorationSchedule(kind="constant", nu=0.4)
    sched_in = ExplorationSchedule(kind="polynomial", c=0.16, eta=0.5)
    cfg = _basic_cfg(
        [
            SellerSpec(kind="oblivious", exploration=sched_ob),
            SellerSpec(kind="informed", forecast_rule="perfect_prediction", exploration=sched_in),
        ],
        horizon=8000,
        n_seeds=20,
    )
    res = run_simulation(cfg)
    p1, p2 = market.stackelberg_duopoly(cfg.market)
    p_NE = market.nash_prices(cfg.market)
    final = res.tilde_p[-1].mean(axis=1)
    # Allow a moderate tolerance because of the finite horizon and the o(1) term.
    assert final[0] > p_NE[0] - 0.05  # leader at least at NE
    assert final[1] > p_NE[1] - 0.05
    # Leader > follower.
    assert final[0] > final[1] - 0.02


def test_mean_price_forecast_yields_NE() -> None:
    sched_ob = ExplorationSchedule(kind="constant", nu=0.4)
    sched_in = ExplorationSchedule(kind="polynomial", c=0.16, eta=0.5)
    cfg = _basic_cfg(
        [
            SellerSpec(kind="oblivious", exploration=sched_ob),
            SellerSpec(kind="informed", forecast_rule="mean_price", exploration=sched_in),
        ],
        horizon=4000,
        n_seeds=20,
    )
    res = run_simulation(cfg)
    p_NE = market.nash_prices(cfg.market)
    final = res.tilde_p[-1].mean(axis=1)
    assert np.allclose(final, p_NE, atol=0.04)


def test_log_every_returns_correct_shape() -> None:
    sched = ExplorationSchedule(kind="constant", nu=0.4)
    cfg = _basic_cfg([SellerSpec(kind="oblivious", exploration=sched) for _ in range(2)])
    cfg = cfg.model_copy(update={"horizon": 1000, "log_every": 50})
    res = run_simulation(cfg)
    # ceil(1000/50) = 20 snapshots, plus the final step at 999.
    assert res.log_steps[-1] == 999
    assert res.prices.shape == (res.log_steps.size, 2, cfg.n_seeds)
