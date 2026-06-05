"""Tests for revenue benchmarks."""

from __future__ import annotations

import numpy as np

from src import benchmarks
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


def test_per_period_revenue_collusive_dominates_NE() -> None:
    d = _baseline_demand()
    pi = benchmarks.benchmark_per_period_revenues(d)
    assert pi["collusive"].sum() > pi["NE"].sum()
    assert "stackelberg" in pi


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
