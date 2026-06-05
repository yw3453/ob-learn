"""Sanity tests for the iterated OLS implementation."""

from __future__ import annotations

import numpy as np
import pytest

from src.estimators import (
    BatchedIteratedOLS,
    informed_greedy_price,
    oblivious_greedy_price,
    project_box,
)


def test_recovery_no_noise() -> None:
    """LS on noiseless data should recover the parameters exactly."""
    rng = np.random.default_rng(0)
    n_obs = 200
    n_problems = 1
    n_seeds = 3
    dim = 2

    ols = BatchedIteratedOLS(n_problems, n_seeds, dim)
    true_theta = np.array([[2.0, -0.5], [3.0, -1.0], [-1.0, 0.7]])  # (n_seeds, dim)
    true_theta = true_theta[None, :, :]  # (n_problems, n_seeds, dim)

    for _ in range(n_obs):
        x = np.empty((n_problems, n_seeds, dim))
        x[..., 0] = 1.0
        x[..., 1] = rng.uniform(-1, 1, size=(n_problems, n_seeds))
        y = (x * true_theta).sum(axis=-1)
        ols.update(x, y)

    theta_hat = ols.solve(ridge=0.0)
    assert np.allclose(theta_hat, true_theta, atol=1e-10)


def test_box_projection() -> None:
    theta = np.array([[[1.5, -0.5], [2.0, -0.1]]])  # (1, 2, 2)
    low = np.array([[[1.0, -1.0]]])
    high = np.array([[[1.8, -0.3]]])
    proj = project_box(theta, low, high)
    assert np.all(proj >= low)
    assert np.all(proj <= high)


def test_oblivious_greedy_price_clipping() -> None:
    theta = np.array([[3.0, -1.0]])  # a=3, b=-1 -> p* = 1.5
    p = oblivious_greedy_price(theta, l=2.0, u=3.0)
    # 1.5 clipped to [2.0, 3.0] -> 2.0
    assert np.allclose(p, 2.0)


def test_informed_greedy_price() -> None:
    # Set theta = (alpha=2.5, beta=-1.0, gamma=0.4) for seller 0
    theta = np.array([[2.5, -1.0, 0.4]])  # (n_seeds=1, N+1=3)
    p_forecast = np.array([[0.0, 1.5625]])  # competitor's price (own pos 0 ignored)
    p = informed_greedy_price(theta, p_forecast, seller_idx=0, l=0.0, u=5.0)
    # phi^{in} = -(alpha + gamma*p_2)/(2 beta) = -(2.5 + 0.4*1.5625)/(-2) = 1.5625
    assert np.allclose(p, 1.5625)


@pytest.mark.parametrize("dim", [2, 3, 5])
def test_min_eigenvalue_positive_after_warmup(dim: int) -> None:
    rng = np.random.default_rng(7)
    ols = BatchedIteratedOLS(1, 4, dim)
    for _ in range(10 * dim):
        x = np.empty((1, 4, dim))
        x[..., 0] = 1.0
        x[..., 1:] = rng.uniform(-1, 1, size=(1, 4, dim - 1))
        y = rng.normal(size=(1, 4))
        ols.update(x, y)
    eigs = ols.min_eigenvalue()
    assert (eigs > 0).all()
