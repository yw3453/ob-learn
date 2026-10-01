"""Post-simulation analysis helpers."""

from __future__ import annotations

import numpy as np


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
