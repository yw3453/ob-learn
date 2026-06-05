"""Revenue benchmarks used by experiments 2a-2d.

All quantities are derived from the trajectories in
:class:`src.simulator.SimulationResult` plus the closed-form
per-period revenue formulae in :mod:`src.market`.
"""

from __future__ import annotations

import numpy as np

from . import market
from .config import DemandParams
from .simulator import SimulationResult


def cumulative_revenue(result: SimulationResult) -> np.ndarray:
    """``R_{t, i, s} = sum_{tau <= t} p_{tau, i, s} * d_{tau, i, s}``.

    Returns an array of shape ``(T_log, N, S)``. Note that when
    ``log_every > 1`` the result is undersampled: only the values at the
    stored snapshots are correct cumulative sums of the *stored* per-step
    revenues. For accurate cumulative revenue, use ``log_every == 1``.
    """
    instantaneous = result.prices * result.demands
    return np.cumsum(instantaneous, axis=0)


def average_revenue(result: SimulationResult) -> np.ndarray:
    """Time-averaged ``R_{t, i, s} / (t + 1)`` (where ``t`` is the log index)."""
    cum = cumulative_revenue(result)
    n_log_periods = np.arange(1, result.log_steps.size + 1)[:, None, None]
    return cum / n_log_periods


def benchmark_per_period_revenues(d: DemandParams) -> dict[str, np.ndarray]:
    """Per-period revenue at NE / collusive / Stackelberg (when N=2).

    Returns a dict with keys ``"NE"``, ``"collusive"``, and (if ``N == 2``)
    ``"stackelberg"``. Each value is shape ``(N,)``, the per-seller revenue.
    """
    p_NE = market.nash_prices(d)
    p_C = market.collusive_prices(d)
    out: dict[str, np.ndarray] = {
        "NE": market.per_period_revenue(d, p_NE),
        "collusive": market.per_period_revenue(d, p_C),
    }
    if d.N == 2:
        p_S = np.array(market.stackelberg_duopoly(d))
        out["stackelberg"] = market.per_period_revenue(d, p_S)
    return out


def revenue_summary_statistics(result: SimulationResult) -> dict[str, np.ndarray]:
    """Summarize ``R_T / T`` and the Pi benchmarks across seeds.

    Returns a dict whose values are shape ``(N,)`` (per-seller).
    """
    avg = average_revenue(result)[-1]  # (N, S)
    return {
        "mean": avg.mean(axis=1),
        "p05": np.percentile(avg, 5, axis=1),
        "p25": np.percentile(avg, 25, axis=1),
        "p50": np.percentile(avg, 50, axis=1),
        "p75": np.percentile(avg, 75, axis=1),
        "p95": np.percentile(avg, 95, axis=1),
    }
