"""Vectorized iterated least squares with box projection.

The protocol is plain OLS at every step,
re-fit with one extra observation, then projected onto a per-seller box. We
vectorize over seeds (and over sellers, when they share a regression
dimension) by maintaining ``X^T X`` and ``X^T y`` incrementally; this keeps
the per-step cost ``O(dim^3)`` independent of horizon, and amortizes well
when ``n_seeds`` is large.

Two flavors of regression appear:

* **Oblivious** (``dim = 2``): regressor ``x_{n,i}^{ob} = (1, p_{n,i})``,
  parameters ``(a_i, b_i)``.
* **Informed** (``dim = N + 1``): regressor
  ``x_{n,i}^{in} = (1, p_{n,1}, ..., p_{n,N})``, parameters
  ``(alpha_i, theta_{i,1}, ..., theta_{i,N})`` where ``theta_{i,i} = beta_i``
  and ``theta_{i,k} = gamma_{i,k}`` for ``k != i``.
"""

from __future__ import annotations

import numpy as np


class BatchedIteratedOLS:
    """Vectorized incremental OLS over a ``(n_problems, n_seeds)`` grid.

    Each ``(p, s)`` problem is an independent ``dim``-parameter regression that
    receives one observation per :meth:`update` call. The implementation never
    materializes ``X``; only ``X^T X`` (cumulative second moment) and
    ``X^T y`` (cumulative cross moment) are stored.
    """

    __slots__ = ("XtX", "Xty", "dim", "n_obs", "n_problems", "n_seeds")

    def __init__(self, n_problems: int, n_seeds: int, dim: int) -> None:
        self.n_problems = n_problems
        self.n_seeds = n_seeds
        self.dim = dim
        self.XtX = np.zeros((n_problems, n_seeds, dim, dim), dtype=np.float64)
        self.Xty = np.zeros((n_problems, n_seeds, dim), dtype=np.float64)
        self.n_obs = 0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        """Add one observation per ``(problem, seed)``.

        Parameters
        ----------
        x : array of shape ``(n_problems, n_seeds, dim)``
        y : array of shape ``(n_problems, n_seeds)``
        """
        if x.shape != (self.n_problems, self.n_seeds, self.dim):
            raise ValueError(f"x must have shape {(self.n_problems, self.n_seeds, self.dim)}, got {x.shape}")
        if y.shape != (self.n_problems, self.n_seeds):
            raise ValueError(f"y must have shape {(self.n_problems, self.n_seeds)}, got {y.shape}")
        self.XtX += np.einsum("pnd,pne->pnde", x, x, optimize=True)
        self.Xty += x * y[..., None]
        self.n_obs += 1

    def solve(self, ridge: float = 1e-12) -> np.ndarray:
        """Return ``(X^T X + ridge I)^{-1} X^T y`` for every ``(problem, seed)``.

        A tiny default ridge is added for numerical safety; with sufficient
        warm-up exploration ``X^T X`` is already strictly positive definite
        and the ridge has no observable effect.
        """
        if ridge > 0.0:
            eye = np.eye(self.dim) * ridge
            A = self.XtX + eye
        else:
            A = self.XtX
        # ``np.linalg.solve`` interprets the last two dims of ``b`` as ``(M, K)``;
        # add a trailing singleton so the call corresponds to a 1-D solve.
        return np.linalg.solve(A, self.Xty[..., None])[..., 0]

    def min_eigenvalue(self) -> np.ndarray:
        """Smallest eigenvalue of ``X^T X / n_obs`` for each ``(problem, seed)``.

        Useful as an empirical analogue of the persistent-excitation lower
        bound ``C_M``. Returns shape ``(n_problems, n_seeds)``.
        """
        if self.n_obs == 0:
            return np.zeros((self.n_problems, self.n_seeds))
        # eigvalsh works on batched symmetric matrices.
        eigs = np.linalg.eigvalsh(self.XtX / max(self.n_obs, 1))
        return eigs[..., 0]


def project_box(theta: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Coordinate-wise (Euclidean) projection of ``theta`` onto a box.

    ``theta`` has shape ``(n_problems, n_seeds, dim)``; ``low`` and ``high``
    are broadcastable to that shape (typically ``(n_problems, 1, dim)``).
    Equivalent to :func:`numpy.clip` on each coordinate.
    """
    return np.clip(theta, low, high)


def oblivious_greedy_price(theta: np.ndarray, l: float, u: float) -> np.ndarray:
    """Greedy price ``-a / (2b)`` clipped to ``[l, u]``.

    Parameters
    ----------
    theta : array of shape ``(..., 2)``
        ``theta[..., 0] = a``, ``theta[..., 1] = b`` with ``b < 0``.
    l, u : scalars
        Price bounds from :class:`src.config.DemandParams`.
    """
    a = theta[..., 0]
    b = theta[..., 1]
    p = -a / (2.0 * b)
    return np.clip(p, l, u)


def informed_greedy_price(
    theta: np.ndarray,
    p_forecast: np.ndarray,
    seller_idx: int,
    l: float,
    u: float,
) -> np.ndarray:
    """Greedy informed price given a forecast of competitors' prices.

    Computes ``-(alpha_hat + sum_{j != seller_idx} gamma_hat_{i,j} *
    p_forecast[j]) / (2 beta_hat)`` clipped to ``[l, u]``.

    Parameters
    ----------
    theta : array of shape ``(n_seeds, N + 1)``
        Estimated coefficients for the seller in question.
    p_forecast : array of shape ``(n_seeds, N)``
        Forecasted price vector. Entry ``seller_idx`` is ignored.
    seller_idx : int
        Index of the seller whose price we are computing.
    """
    alpha_hat = theta[..., 0]
    beta_hat = theta[..., seller_idx + 1]
    N = p_forecast.shape[-1]
    cross = np.zeros_like(alpha_hat)
    for k in range(N):
        if k == seller_idx:
            continue
        cross = cross + theta[..., k + 1] * p_forecast[..., k]
    p = -(alpha_hat + cross) / (2.0 * beta_hat)
    return np.clip(p, l, u)
