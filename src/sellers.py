"""Forecast-rule functions for informed sellers in mixed markets.

The simulator delegates to these helpers when an informed seller needs a
prediction of competitors' prices for the current step. Each function returns
a forecast vector of shape ``(n_seeds, N)``; the entry corresponding to the
seller itself is irrelevant (it is multiplied by ``gamma_hat[i, i] = beta_i``
in the informed greedy formula and we mask that term explicitly anyway).

The five rules are:

* ``mean_price``        -- forecast competitors with their running mean price.
* ``perfect_prediction``-- forecast competitors with their realized prices.
* ``greedy_component``  -- the "passive vs. active" greedy-component rule.
* ``lag1_autocorr``     -- the lag-1 autocorrelation rule.
* ``oracle_nash``       -- a control: forecast competitors' prices as
                          ``p_j^{NE}``. Useful for sanity-checking the
                          informed seller's behaviour.
"""

from __future__ import annotations

import numpy as np


def forecast_mean_price(running_mean: np.ndarray) -> np.ndarray:
    """``m_{n,j}`` for every seller, broadcast across seeds.

    Parameters
    ----------
    running_mean : array of shape ``(n_seeds, N)``
        The running average price of every seller through step ``n``.
    """
    return running_mean.copy()


def forecast_lag1(last_prices: np.ndarray) -> np.ndarray:
    """Use the previous-step prices ``p_{n,j}`` (lag-1 autocorrelation rule)."""
    return last_prices.copy()


def forecast_perfect_prediction(current_realized: np.ndarray) -> np.ndarray:
    """Use the realized prices of competitors at the current step ``n+1``.

    Requires those competitors to have already chosen their prices: the
    informed seller can perfectly predict the oblivious seller's next price.
    """
    return current_realized.copy()


def forecast_greedy_component(current_tilde: np.ndarray) -> np.ndarray:
    """Use only the *greedy* component ``tilde p_{n+1, j}`` of competitors."""
    return current_tilde.copy()


def forecast_oracle_nash(p_NE: np.ndarray, n_seeds: int) -> np.ndarray:
    """Constant Nash forecast, broadcast to ``(n_seeds, N)``. Sanity-check rule."""
    return np.broadcast_to(p_NE[None, :], (n_seeds, p_NE.shape[0])).copy()
