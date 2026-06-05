"""Post-simulation analysis helpers.

All functions take a :class:`src.simulator.SimulationResult` and
optionally other inputs (true parameters, NE prices, etc.) and return either
arrays or pandas DataFrames suitable for direct serialization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import market
from .simulator import SimulationResult

# ---------------------------------------------------------------------------
# MSE curves
# ---------------------------------------------------------------------------


def mse_theta_oblivious(result: SimulationResult) -> np.ndarray:
    """Aggregate MSE ``sum_i E ||theta_hat_{n,i}^{ob} - theta_i^{*, ob}||^2`` per log step.

    Returns an array of shape ``(T_log, S)`` -- per-seed values; aggregate
    across sellers but keep seeds separate so percentiles can be taken.
    """
    if result.theta_ob is None or result.ob_seller_idx.size == 0:
        return np.zeros((result.log_steps.size, result.config.n_seeds))
    d = result.config.market
    p_NE = market.nash_prices(d)
    target = market.pseudo_true_oblivious(d, p_NE)  # (N, 2)
    target_ob = target[result.ob_seller_idx]  # (n_ob, 2)
    # theta_ob has shape (T_log, n_ob, 2, S); broadcast target.
    diff = result.theta_ob - target_ob[None, :, :, None]
    mse = np.sum(diff**2, axis=(1, 2))  # (T_log, S)
    return mse


def mse_theta_informed(result: SimulationResult) -> np.ndarray:
    """Aggregate MSE ``sum_i E ||theta_hat_{n,i}^{in} - theta_i^*||^2`` per log step."""
    if result.theta_in is None or result.in_seller_idx.size == 0:
        return np.zeros((result.log_steps.size, result.config.n_seeds))
    d = result.config.market
    target = market.true_informed_theta(d)  # (N, N+1)
    target_in = target[result.in_seller_idx]  # (n_in, N+1)
    diff = result.theta_in - target_in[None, :, :, None]
    return np.sum(diff**2, axis=(1, 2))  # (T_log, S)


def mse_price(result: SimulationResult) -> np.ndarray:
    """Aggregate MSE ``sum_i E (tilde_p_{n,i} - p_i^{NE})^2`` per log step."""
    p_NE = market.nash_prices(result.config.market)  # (N,)
    diff = result.tilde_p - p_NE[None, :, None]  # (T_log, N, S)
    return np.sum(diff**2, axis=1)  # (T_log, S)


def mse_summary_dataframe(curve: np.ndarray, log_steps: np.ndarray, name: str) -> pd.DataFrame:
    """Tidy ``DataFrame`` with mean and 5/25/50/75/95 percentiles per step."""
    return pd.DataFrame(
        {
            "n": log_steps,
            "metric": name,
            "mean": curve.mean(axis=1),
            "p05": np.percentile(curve, 5, axis=1),
            "p25": np.percentile(curve, 25, axis=1),
            "p50": np.percentile(curve, 50, axis=1),
            "p75": np.percentile(curve, 75, axis=1),
            "p95": np.percentile(curve, 95, axis=1),
        }
    )


# ---------------------------------------------------------------------------
# Log-log rate fit
# ---------------------------------------------------------------------------


def fit_loglog_slope(
    n: np.ndarray,
    y: np.ndarray,
    *,
    tail_fraction: float = 0.5,
    min_y: float = 1e-12,
) -> dict[str, float]:
    """Fit ``log y = a + b log n`` on the last ``tail_fraction`` of the curve.

    Returns ``{"slope": b, "intercept": a, "rss": residual sum of squares,
    "n_used": K}``. Drops entries with ``y <= min_y`` (typically zeros from
    log_every gaps or from the warm-up).
    """
    n_arr = np.asarray(n, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    K = n_arr.size
    start = int(K * (1 - tail_fraction))
    n_tail = n_arr[start:]
    y_tail = y_arr[start:]
    mask = (y_tail > min_y) & np.isfinite(y_tail) & (n_tail > 0)
    if mask.sum() < 3:
        return {"slope": float("nan"), "intercept": float("nan"), "rss": float("nan"), "n_used": int(mask.sum())}
    log_n = np.log(n_tail[mask])
    log_y = np.log(y_tail[mask])
    A = np.vstack([np.ones_like(log_n), log_n]).T
    coef, residuals, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
    rss = float(residuals[0]) if residuals.size else float(np.sum((A @ coef - log_y) ** 2))
    return {"slope": float(coef[1]), "intercept": float(coef[0]), "rss": rss, "n_used": int(mask.sum())}


# ---------------------------------------------------------------------------
# Regime characterization
# ---------------------------------------------------------------------------


def predicted_rates(
    d,
    box,
    nu_squared: float | np.ndarray,
) -> dict[str, float | str | np.ndarray]:
    """Predicted convergence regime + asymptotic rate exponent.

    Returns:

    * ``regime``: ``"fast" | "critical" | "slow" | "below"``.
    * ``rho``: ``threshold / C_M`` (only meaningful in slow / below regimes).
    * ``slope``: predicted log-log slope on MSE curves, i.e. ``-1``,
      ``-1`` (with log correction in the critical case), or
      ``-2 (1 - rho)`` in the slow regime.
    * Plus the gamma_bar / L_phi^{ob} / C_x / threshold / C_M values.
    """
    info = market.predicted_regime(d, box, float(nu_squared))
    regime = info["regime"]
    rho = info["rho"]
    if regime == "fast":
        slope = -1.0
    elif regime == "critical":
        slope = -1.0
    elif regime == "slow":
        slope = -2.0 * (1.0 - float(rho))
    else:
        slope = float("nan")
    return {**info, "slope": slope}


def realized_C_M(result: SimulationResult, *, eps: float = 1e-12) -> np.ndarray:
    """Empirical lower bound on ``C_M``: smallest eigenvalue of the OLS
    second-moment matrix ``X^T X / n`` for each oblivious seller.

    The simulator does not store ``X^T X`` snapshots, so this is reconstructed
    from running price moments. For oblivious sellers,
    ``X^T X / n = [[1, m_i], [m_i, m_i^2 + V_i]]`` where ``V_i = Q_{ii} - m_i^2``.
    The smallest eigenvalue is positive iff ``V_i > 0`` and we return it for
    each ``(t, ob_seller, S)``.

    Returns shape ``(T_log, n_ob, S)``.
    """
    m = result.moments["m"]  # (T_log, N, S)
    Q = result.moments["Q"]  # (T_log, N, N, S)
    ob = result.ob_seller_idx
    if ob.size == 0:
        return np.zeros((result.log_steps.size, 0, result.config.n_seeds))
    m_ob = m[:, ob, :]  # (T_log, n_ob, S)
    Q_diag = np.stack([Q[:, i, i, :] for i in ob], axis=1)  # (T_log, n_ob, S)
    np.maximum(Q_diag - m_ob**2, eps)
    # 2x2 matrix: [[1, m], [m, m^2 + V]]; min eigenvalue solved analytically.
    a = 1.0
    b = m_ob
    c = Q_diag  # = m^2 + V
    tr = a + c
    det = a * c - b**2  # = V
    disc = np.sqrt(np.maximum(tr**2 - 4.0 * det, 0.0))
    lam_min = 0.5 * (tr - disc)
    return lam_min


# ---------------------------------------------------------------------------
# Convenience: collect everything into a tidy long-format DataFrame
# ---------------------------------------------------------------------------


def trajectory_summary(result: SimulationResult) -> pd.DataFrame:
    """Long-format summary DataFrame with every standard metric per log step.

    Columns: ``n, metric, mean, p05, p25, p50, p75, p95``.
    """
    frames = []
    n = result.log_steps
    if result.theta_ob is not None:
        frames.append(mse_summary_dataframe(mse_theta_oblivious(result), n, "mse_theta_ob"))
    if result.theta_in is not None:
        frames.append(mse_summary_dataframe(mse_theta_informed(result), n, "mse_theta_in"))
    frames.append(mse_summary_dataframe(mse_price(result), n, "mse_price"))
    return pd.concat(frames, ignore_index=True)
