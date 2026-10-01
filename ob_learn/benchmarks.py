"""Revenue benchmarks used by experiments 2a-2d.

All quantities are derived from the trajectories in
:class:`ob_learn.simulator.SimulationResult` plus the closed-form
per-period revenue formulae in :mod:`ob_learn.market`.
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


def oracle_best_response_prices(d: DemandParams, prices: np.ndarray) -> np.ndarray:
    """Full-information best-response price for each seller at each step.

    Implements the oracle best response of the paper's regret definition,
    ``phi_i^{in}(theta_i, p_{-i}) = (alpha_i + sum_{j != i} gamma_{i,j} p_j)
    / (-2 beta_i)``, evaluated at the *realized* rival prices. ``prices`` has
    shape ``(T_log, N, S)`` (seller axis in the middle); the return has the
    same shape. The own-price entry drops out because ``gamma`` has a zero
    diagonal, so ``sum_{j != i} gamma_{i,j} p_j = (Gamma_{off} p)_i``.
    """
    alpha = d.alpha_arr[None, :, None]
    beta = d.beta_arr[None, :, None]
    cross = np.einsum("ij,tjs->tis", d.gamma_arr, prices, optimize=True)
    return (alpha + cross) / (-2.0 * beta)


def per_period_regret(result: SimulationResult, d: DemandParams) -> np.ndarray:
    """Instantaneous per-period dynamic regret ``|beta_i| (phi_i - p_{n,i})^2``.

    Equal to the revenue gap ``R_i(phi_i^{in}(theta_i, p_{n,-i}), p_{n,-i}) -
    R_i(p_{n,i}, p_{n,-i})``. Uses the realized posted prices ``result.prices`` (which
    include the exploration perturbation), so a persistently-exploring seller
    retains a per-period regret floor of order ``|beta_i| nu_{n,i}^2``.

    Returns an array of shape ``(T_log, N, S)`` sampled at ``result.log_steps``.
    """
    br = oracle_best_response_prices(d, result.prices)
    neg_beta = -d.beta_arr[None, :, None]  # |beta_i| since beta_i < 0
    return neg_beta * (br - result.prices) ** 2


def per_period_learning_regret(result: SimulationResult, d: DemandParams) -> np.ndarray:
    """Instantaneous *learning* regret ``|beta_i| (phi_i - tilde p_{n,i})^2``.

    Identical to :func:`per_period_regret` but evaluated at the *greedy*
    (exploitation) prices ``tilde p_n`` instead of the realized posted prices.
    This is the per-period increment of the learning component
    ``Delta_i^{le}(T)``.

    Returns an array of shape ``(T_log, N, S)`` sampled at ``result.log_steps``.
    """
    br = oracle_best_response_prices(d, result.tilde_p)
    neg_beta = -d.beta_arr[None, :, None]  # |beta_i| since beta_i < 0
    return neg_beta * (br - result.tilde_p) ** 2


def _trapz_cumulative(g: np.ndarray, log_steps: np.ndarray) -> np.ndarray:
    """Trapezoidal running sum of a per-period quantity ``g`` over the true
    period index ``n = log_steps + 1``.

    ``g`` has shape ``(T_log, ...)``; the leading segment ``[1, n_0]`` is
    approximated by ``g_0 * n_0`` (treating the pre-first-snapshot value as
    roughly constant at ``g_0``), which is negligible for the log--log tail.
    Returns an array of the same shape as ``g``.
    """
    n = np.asarray(log_steps, dtype=np.float64) + 1.0  # (T_log,)
    T_log = n.size
    cum = np.empty_like(g)
    cum[0] = g[0] * n[0]
    for k in range(1, T_log):
        dn = n[k] - n[k - 1]
        cum[k] = cum[k - 1] + 0.5 * (g[k] + g[k - 1]) * dn
    return cum


def cumulative_regret(result: SimulationResult, d: DemandParams) -> np.ndarray:
    """Cumulative dynamic regret ``Delta_i(n) = sum_{m <= n} g_{m,i}``.

    The per-period regret ``g_{m,i}`` is only stored at the sparse
    ``result.log_steps``; we reconstruct the running sum by trapezoidal
    integration of ``g`` against the true period index ``n = log_steps + 1``.
    Returns shape ``(T_log, N, S)``, the cumulative regret at each logged
    horizon.
    """
    return _trapz_cumulative(per_period_regret(result, d), result.log_steps)


def cumulative_learning_regret(result: SimulationResult, d: DemandParams) -> np.ndarray:
    """Cumulative *learning* regret ``Delta_i^{le}(n) = sum_{m <= n} g^{le}_{m,i}``.

    Trapezoidal running sum of :func:`per_period_learning_regret` over the true
    period index. This is the exact quadratic in the greedy prices used to plot
    the learning component of regret in the numerical experiments. Returns shape
    ``(T_log, N, S)``.
    """
    return _trapz_cumulative(per_period_learning_regret(result, d), result.log_steps)


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
