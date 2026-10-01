"""Vectorized simulator for the estimate--exploit--explore protocol.

Given an :class:`ExperimentConfig`, :func:`run_simulation` runs ``n_seeds``
independent simulations in parallel using NumPy vectorization. The output is a
:class:`SimulationResult` bundling per-step trajectories of prices, demands,
greedy components, dithering draws, OLS estimates, and running moments.

Implementation notes
--------------------
* Sellers are split into oblivious and informed groups; each group has its own
  :class:`ob_learn.estimators.BatchedIteratedOLS` instance shaped
  ``(n_<group>, n_seeds, dim)``. This lets a single ``np.einsum`` /
  ``np.linalg.solve`` update all sellers in the group at once.
* A short warm-up phase fills each design with pure exploration so that
  ``X^T X`` is full rank from the first OLS solve onward.
* Forecast rules are split into *pre-computable* and *current-step deferred*:

  - Pre-computable: ``mean_price`` (uses ``m_{n,j}``).
  - Deferred: ``perfect_prediction`` -- requires the realized
    ``p_{n+1, j}`` of competitors, so the informed seller acts after them
    (Stackelberg-style).

  In each main step we therefore: (1) compute ``tilde p`` for oblivious
  sellers and informed sellers using pre-computable rules, in that order;
  (2) sample dithering and realize prices for those sellers; (3) compute
  ``tilde p`` for deferred informed sellers using realized competitor
  prices; (4) sample dithering and realize their prices; (5) realize
  demand from the true model and update OLS designs and running moments.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from .config import ExperimentConfig
from .estimators import (
    BatchedIteratedOLS,
    informed_greedy_price,
    oblivious_greedy_price,
    project_box,
)
from .exploration import nu_at, sample_demand_noise

_DEFERRED_RULES = frozenset({"perfect_prediction"})


@dataclass
class SimulationResult:
    """Container for a vectorized run's outputs.

    Arrays have one leading dimension over time (``T_log``), then over
    sellers, and then over seeds (``S``). ``T_log = ceil(horizon / log_every)``
    plus one for the final step (``horizon - 1``) when not already on the grid.
    """

    config: ExperimentConfig
    seeds: np.ndarray
    log_steps: np.ndarray
    prices: np.ndarray  # (T_log, N, S)
    demands: np.ndarray  # (T_log, N, S)
    tilde_p: np.ndarray  # (T_log, N, S)
    z: np.ndarray  # (T_log, N, S)
    theta_ob: np.ndarray | None  # (T_log, n_ob, 2, S)
    theta_in: np.ndarray | None  # (T_log, n_in, N+1, S)
    ob_seller_idx: np.ndarray  # (n_ob,)
    in_seller_idx: np.ndarray  # (n_in,)
    moments: dict[str, np.ndarray]  # m, Q, J, r at log_steps
    elapsed_sec: float

    def trajectories_dict(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {
            "log_steps": self.log_steps,
            "prices": self.prices,
            "demands": self.demands,
            "tilde_p": self.tilde_p,
            "z": self.z,
            "ob_seller_idx": self.ob_seller_idx,
            "in_seller_idx": self.in_seller_idx,
            "seeds": self.seeds,
            **{f"moment_{k}": v for k, v in self.moments.items()},
        }
        if self.theta_ob is not None:
            out["theta_ob"] = self.theta_ob
        if self.theta_in is not None:
            out["theta_in"] = self.theta_in
        return out


def run_simulation(
    cfg: ExperimentConfig,
    *,
    logger: Any | None = None,
    progress: bool | None = None,
    progress_desc: str | None = None,
    child_seeds: np.ndarray | None = None,
    compute_moments: bool = True,
) -> SimulationResult:
    """Run a vectorized simulation across ``cfg.n_seeds`` seeds.

    The function does not write to disk; persisting a run is the caller's job
    (typically via :func:`ob_learn.logging_utils.run_directory`).

    Parameters
    ----------
    progress : bool, optional
        If True, show a tqdm progress bar over the ``T`` main-loop steps.
        If None (default), auto-detect: enabled when stderr is a TTY,
        disabled otherwise (so test/CI runs stay clean).
    progress_desc : str, optional
        Description shown on the progress bar; defaults to ``cfg.name``.
    child_seeds : np.ndarray, optional
        Pre-computed per-seed entropy. If supplied, must have shape
        ``(cfg.n_seeds,)``; the simulator skips the default
        ``SeedSequence(cfg.base_seed)`` derivation and uses these as
        the per-seed Generator seeds. Useful for re-running a single
        seed from a larger seed pool without changing the seed-index
        identity.
    compute_moments : bool, default True
        If False, skip the ``_compute_logged_moments`` Python loop and
        return an empty ``moments`` dict. The caller can recompute the
        moments (e.g. the running mean ``m``) more cheaply from
        ``result.prices`` using vectorised numpy. Useful for very long
        single-seed runs where the per-log-step Python loop dominates
        runtime.
    """
    t0 = time.monotonic()
    d = cfg.market
    N = d.N
    S = cfg.n_seeds
    T = cfg.horizon

    ob_seller_idx = np.array(
        [i for i, s in enumerate(cfg.sellers) if s.kind == "oblivious"], dtype=np.int64
    )
    in_seller_idx = np.array(
        [i for i, s in enumerate(cfg.sellers) if s.kind == "informed"], dtype=np.int64
    )
    n_ob = ob_seller_idx.size
    n_in = in_seller_idx.size

    sched = [s.exploration for s in cfg.sellers]
    forecast_rule_per_seller: list[str | None] = [s.forecast_rule for s in cfg.sellers]
    deferred_seller_set = {
        i for i, s in enumerate(cfg.sellers) if s.forecast_rule in _DEFERRED_RULES
    }

    n_warmup = cfg.n_warmup if cfg.n_warmup is not None else max(N + 2, 4)

    # Per-seed RNGs.
    if child_seeds is None:
        seed_seq = np.random.SeedSequence(cfg.base_seed)
        child_seeds = seed_seq.generate_state(S, dtype=np.uint32)
    else:
        child_seeds = np.asarray(child_seeds, dtype=np.uint32)
        if child_seeds.shape != (S,):
            raise ValueError(
                f"child_seeds shape {child_seeds.shape} != (n_seeds={S},)"
            )
    rng_array: list[np.random.Generator] = [np.random.default_rng(int(s)) for s in child_seeds]

    # Logging grid.
    log_idx = np.arange(0, T, cfg.log_every, dtype=np.int64)
    if log_idx.size == 0 or int(log_idx[-1]) != T - 1:
        log_idx = np.append(log_idx, T - 1)
    T_log = int(log_idx.size)
    log_position = -np.ones(T, dtype=np.int64)
    log_position[log_idx] = np.arange(T_log, dtype=np.int64)

    prices_log = np.zeros((T_log, N, S), dtype=np.float64)
    demands_log = np.zeros((T_log, N, S), dtype=np.float64)
    tilde_log = np.zeros((T_log, N, S), dtype=np.float64)
    z_log = np.zeros((T_log, N, S), dtype=np.float64)
    theta_ob_log = np.zeros((T_log, n_ob, 2, S), dtype=np.float64) if n_ob > 0 else None
    theta_in_log = np.zeros((T_log, n_in, N + 1, S), dtype=np.float64) if n_in > 0 else None

    # Estimators.
    ols_ob = BatchedIteratedOLS(n_ob, S, 2) if n_ob > 0 else None
    ols_in = BatchedIteratedOLS(n_in, S, N + 1) if n_in > 0 else None

    # Per-seller projection boxes (broadcastable over seeds).
    ob_low: np.ndarray | None
    ob_high: np.ndarray | None
    if n_ob > 0:
        ob_low = np.stack(
            [
                np.array([cfg.oblivious_projection.a_low[i], cfg.oblivious_projection.b_low[i]])
                for i in ob_seller_idx
            ]
        )[:, None, :]
        ob_high = np.stack(
            [
                np.array([cfg.oblivious_projection.a_high[i], cfg.oblivious_projection.b_high[i]])
                for i in ob_seller_idx
            ]
        )[:, None, :]
    else:
        ob_low = ob_high = None

    in_low: np.ndarray | None
    in_high: np.ndarray | None
    if n_in > 0:
        in_low = np.stack(
            [np.asarray(cfg.informed_projection.low[i], dtype=np.float64) for i in in_seller_idx]
        )[:, None, :]
        in_high = np.stack(
            [np.asarray(cfg.informed_projection.high[i], dtype=np.float64) for i in in_seller_idx]
        )[:, None, :]
    else:
        in_low = in_high = None

    alpha = d.alpha_arr
    beta = d.beta_arr
    G = d.gamma_arr

    # ---- Warm-up: pure exploration to make X^T X full rank --------------
    if cfg.initial_prices is not None:
        ip = np.asarray(cfg.initial_prices, dtype=np.float64)
        if ip.shape != (n_warmup, N):
            raise ValueError(f"initial_prices shape {ip.shape} != (n_warmup={n_warmup}, N={N})")
        warmup_prices = np.broadcast_to(ip[:, :, None], (n_warmup, N, S)).copy()
    else:
        warmup_prices = np.zeros((n_warmup, N, S), dtype=np.float64)
        for w in range(n_warmup):
            for s, rng in enumerate(rng_array):
                warmup_prices[w, :, s] = rng.uniform(d.l, d.u, size=N)

    last_prices = np.zeros((N, S), dtype=np.float64)
    running_sum_p = np.zeros((N, S), dtype=np.float64)
    n_steps_done = 0

    for w in range(n_warmup):
        p = warmup_prices[w]
        eps = _sample_demand_eps(rng_array, d, N, S)
        demand = alpha[:, None] + beta[:, None] * p + G @ p + eps
        _ols_update(ols_ob, ob_seller_idx, p, demand)
        _ols_update_informed(ols_in, in_seller_idx, p, demand, N)
        last_prices[:] = p
        running_sum_p += p
        n_steps_done += 1

    # ---- Main loop -------------------------------------------------------
    if progress is None:
        progress = sys.stderr.isatty()
    desc = progress_desc if progress_desc is not None else cfg.name
    main_iter: Any = range(T)
    if progress:
        main_iter = tqdm(
            main_iter,
            total=T,
            desc=desc,
            leave=False,
            dynamic_ncols=True,
            mininterval=0.5,
            unit="step",
        )
    method = cfg.exploration_method
    ridge_base = max(cfg.ridge_lambda, 1e-12) if cfg.estimator == "ridge" else 1e-12
    ridge_is_decaying = cfg.estimator == "ridge" and cfg.ridge_decay > 0.0
    ridge_val = ridge_base

    # Per-seller exploration knobs (fall back to experiment-level defaults),
    # so sellers can explore asymmetrically.
    eps_c_arr = np.array(
        [s.epsilon_c if s.epsilon_c is not None else cfg.epsilon_c for s in cfg.sellers],
        dtype=np.float64,
    )
    eps_eta_arr = np.array(
        [s.epsilon_eta if s.epsilon_eta is not None else cfg.epsilon_eta for s in cfg.sellers],
        dtype=np.float64,
    )
    for t in main_iter:
        # Shrinking ridge penalty lambda_n = lambda_0 * (n+1)^{-ridge_decay},
        # so the regularization bias vanishes even under decaying exploration.
        if ridge_is_decaying:
            ridge_val = max(cfg.ridge_lambda * (n_steps_done + 1) ** (-cfg.ridge_decay), 1e-12)

        # Solve LS at the current state (uses observations up through n_steps_done).
        if ols_ob is not None:
            assert ob_low is not None and ob_high is not None
            theta_ob = project_box(ols_ob.solve(ridge=ridge_val), ob_low, ob_high)
        else:
            theta_ob = None
        if ols_in is not None:
            assert in_low is not None and in_high is not None
            theta_in = project_box(ols_in.solve(ridge=ridge_val), in_low, in_high)
        else:
            theta_in = None

        # ``tilde_p`` is the *exploitation* (greedy) price built from the
        # projected point estimate; exploration is injected into the realized
        # price below (additive dither, or an epsilon-greedy random price).
        tilde_p = np.zeros((N, S), dtype=np.float64)

        # Step 1a: oblivious greedy (exploitation) component from the point estimate.
        if theta_ob is not None:
            tilde_p[ob_seller_idx] = oblivious_greedy_price(theta_ob, d.l, d.u)

        # Step 1b: informed greedy components under the ``mean_price`` forecast.
        running_mean = running_sum_p / max(n_steps_done, 1)  # (N, S)
        if theta_in is not None:
            for slot, seller_i in enumerate(in_seller_idx):
                seller_idx = int(seller_i)
                rule = forecast_rule_per_seller[seller_idx]
                if rule in _DEFERRED_RULES:
                    continue
                forecast = running_mean.T.copy()  # (S, N)
                tilde_p[seller_idx] = informed_greedy_price(
                    theta_in[slot], forecast, seller_idx=seller_idx, l=d.l, u=d.u
                )

        # Step 2: draw exploration randomness and realize prices for all
        # non-deferred sellers, per the exploration method.
        z = np.zeros((N, S), dtype=np.float64)
        explore_mask: np.ndarray | None = None
        rand_prices: np.ndarray | None = None
        if method == "dither":
            z = _sample_dithering(rng_array, sched, n_steps_done + 1, N, S)
        elif method == "epsilon_greedy":
            n_now = float(n_steps_done + 1)
            eps_per_seller = np.minimum(1.0, eps_c_arr * np.power(n_now, -eps_eta_arr))
            explore_mask = _sample_bernoulli(rng_array, eps_per_seller, N, S)
            rand_prices = _sample_uniform_prices(rng_array, N, S, d.l, d.u)

        # Bind the per-step arrays as defaults so the closure captures the
        # current iteration's values (it is invoked immediately below, but the
        # explicit binding also silences the loop-variable-capture warning).
        def _realize(
            i: int,
            *,
            tilde_p: np.ndarray = tilde_p,
            z: np.ndarray = z,
            explore_mask: np.ndarray | None = explore_mask,
            rand_prices: np.ndarray | None = rand_prices,
        ) -> np.ndarray:
            if method == "epsilon_greedy":
                assert explore_mask is not None and rand_prices is not None
                greedy = np.clip(tilde_p[i], d.l, d.u)
                return np.where(explore_mask[i], rand_prices[i], greedy)
            return np.clip(tilde_p[i] + z[i], d.l, d.u)  # dither (z == 0 if disabled)

        prices_now = np.empty((N, S), dtype=np.float64)
        for i in range(N):
            if i in deferred_seller_set:
                prices_now[i] = np.nan  # filled in step 3
                continue
            prices_now[i] = _realize(i)

        # Step 3: deferred informed sellers (perfect_prediction).
        for slot, seller_i in enumerate(in_seller_idx):
            seller_idx = int(seller_i)
            rule = forecast_rule_per_seller[seller_idx]
            if rule not in _DEFERRED_RULES:
                continue
            assert theta_in is not None
            # Build forecast from realized prices of others; for any other
            # deferred seller use lag-1 (no chicken-and-egg in our experiments).
            forecast = prices_now.T.copy()  # (S, N)
            for j in range(N):
                if j == seller_idx:
                    continue
                if j in deferred_seller_set:
                    forecast[:, j] = last_prices[j]
            tilde_p[seller_idx] = informed_greedy_price(
                theta_in[slot], forecast, seller_idx=seller_idx, l=d.l, u=d.u
            )
            prices_now[seller_idx] = _realize(seller_idx)

        # Record the effective perturbation z = p - tilde_p for logging
        # (exact for dither; captures the realized deviation otherwise).
        if method != "dither":
            z = prices_now - tilde_p

        # Step 4: realize demand and update OLS / running stats.
        eps = _sample_demand_eps(rng_array, d, N, S)
        demand_now = alpha[:, None] + beta[:, None] * prices_now + G @ prices_now + eps
        _ols_update(ols_ob, ob_seller_idx, prices_now, demand_now)
        _ols_update_informed(ols_in, in_seller_idx, prices_now, demand_now, N)

        last_prices[:] = prices_now
        running_sum_p += prices_now
        n_steps_done += 1

        # Step 5: log snapshot.
        li = log_position[t]
        if li >= 0:
            prices_log[li] = prices_now
            demands_log[li] = demand_now
            tilde_log[li] = tilde_p
            z_log[li] = z
            if theta_ob_log is not None and theta_ob is not None:
                theta_ob_log[li] = np.transpose(theta_ob, (0, 2, 1))
            if theta_in_log is not None and theta_in is not None:
                theta_in_log[li] = np.transpose(theta_in, (0, 2, 1))

    if compute_moments:
        moments = _compute_logged_moments(
            prices_log=prices_log, log_idx=log_idx, warmup_prices=warmup_prices
        )
    else:
        moments = {}
    elapsed = time.monotonic() - t0

    if logger is not None:
        logger.info(
            "simulation done: T=%d, N=%d, S=%d, n_ob=%d, n_in=%d, elapsed=%.2fs",
            T,
            N,
            S,
            n_ob,
            n_in,
            elapsed,
        )

    return SimulationResult(
        config=cfg,
        seeds=child_seeds.astype(np.int64),
        log_steps=log_idx,
        prices=prices_log,
        demands=demands_log,
        tilde_p=tilde_log,
        z=z_log,
        theta_ob=theta_ob_log,
        theta_in=theta_in_log,
        ob_seller_idx=ob_seller_idx,
        in_seller_idx=in_seller_idx,
        moments=moments,
        elapsed_sec=elapsed,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ols_update(ols: BatchedIteratedOLS | None, seller_idx: np.ndarray, p: np.ndarray, demand: np.ndarray) -> None:
    """Add one observation per (oblivious seller, seed) to the oblivious OLS."""
    if ols is None or seller_idx.size == 0:
        return
    n_ob = seller_idx.size
    S = p.shape[1]
    x = np.empty((n_ob, S, 2), dtype=np.float64)
    x[..., 0] = 1.0
    x[..., 1] = p[seller_idx]  # (n_ob, S)
    ols.update(x, demand[seller_idx])


def _ols_update_informed(
    ols: BatchedIteratedOLS | None,
    seller_idx: np.ndarray,
    p: np.ndarray,
    demand: np.ndarray,
    N: int,
) -> None:
    """Add one observation per (informed seller, seed) to the informed OLS.

    Each informed seller observes the full price vector; the regressor is
    ``(1, p_1, ..., p_N)`` -- the same row for all informed sellers (only the
    response ``y_i`` differs across sellers).
    """
    if ols is None or seller_idx.size == 0:
        return
    n_in = seller_idx.size
    S = p.shape[1]
    x = np.empty((n_in, S, N + 1), dtype=np.float64)
    x[..., 0] = 1.0
    x[..., 1:] = p.transpose(1, 0)[None, :, :]
    ols.update(x, demand[seller_idx])


def _sample_demand_eps(rng_array, d, N: int, S: int) -> np.ndarray:
    eps = np.empty((N, S), dtype=np.float64)
    for s, rng in enumerate(rng_array):
        eps[:, s] = sample_demand_noise(d.noise_std, rng, (N,))
    return eps


def _sample_dithering(rng_array, sched, n: int, N: int, S: int) -> np.ndarray:
    """Sample ``z_{n,i}`` with shape ``(N, S)``, drawing N values per seed."""
    nu_per_seller = np.array([nu_at(sched[i], n) for i in range(N)], dtype=np.float64)
    z = np.zeros((N, S), dtype=np.float64)
    sqrt3 = float(np.sqrt(3.0))
    for s, rng in enumerate(rng_array):
        for i in range(N):
            if nu_per_seller[i] == 0.0:
                continue
            a = sqrt3 * nu_per_seller[i]
            z[i, s] = rng.uniform(-a, a)
    return z


def _sample_uniform_prices(rng_array, N: int, S: int, l: float, u: float) -> np.ndarray:
    """Per-seed uniform random prices in ``[l, u]`` of shape ``(N, S)`` (epsilon-greedy)."""
    out = np.empty((N, S), dtype=np.float64)
    for s, rng in enumerate(rng_array):
        out[:, s] = rng.uniform(l, u, size=N)
    return out


def _sample_bernoulli(rng_array, eps_per_seller: np.ndarray, N: int, S: int) -> np.ndarray:
    """Per-seed explore/exploit mask of shape ``(N, S)`` with ``P(explore) = eps``."""
    out = np.zeros((N, S), dtype=bool)
    for s, rng in enumerate(rng_array):
        out[:, s] = rng.random(N) < eps_per_seller
    return out


def _compute_logged_moments(
    *,
    prices_log: np.ndarray,
    log_idx: np.ndarray,
    warmup_prices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute ``(m_n, Q_n, J_n, r_n)`` at the snapshot timesteps.

    ``m_n``, ``Q_n`` are the running mean and second-moment matrix. ``J_n`` is
    the cumulative exploration. ``r_n[i, j]`` is the regression-ratio statistic
    ``r_{n, i <- j}`` that drives the omitted-variable channel.

    When ``log_every == 1`` this is exact; for larger ``log_every`` the
    cumulative sums between snapshots are approximated by treating prices as
    constant on each gap (this avoids storing every raw price). Use
    ``log_every == 1`` if exact regression-ratio paths are required.
    """
    T_log, N, S = prices_log.shape
    n_warm = warmup_prices.shape[0]

    sum_p = warmup_prices.sum(axis=0).copy()  # (N, S)
    sum_pp = np.einsum("tis,tjs->ijs", warmup_prices, warmup_prices, optimize=True)

    m = np.zeros((T_log, N, S))
    Q = np.zeros((T_log, N, N, S))
    J = np.zeros((T_log, N, S))
    r = np.zeros((T_log, N, N, S))

    log_idx_total = log_idx + n_warm

    for k in range(T_log):
        sum_p = sum_p + prices_log[k]
        sum_pp = sum_pp + np.einsum("is,js->ijs", prices_log[k], prices_log[k], optimize=True)
        if k > 0:
            gap = int(log_idx[k] - log_idx[k - 1] - 1)
            if gap > 0:
                # Approximate intermediate prices as the average of consecutive snapshots.
                mid = 0.5 * (prices_log[k] + prices_log[k - 1])
                sum_p = sum_p + gap * mid
                sum_pp = sum_pp + gap * np.einsum("is,js->ijs", mid, mid, optimize=True)
        n_total = int(log_idx_total[k]) + 1
        m[k] = sum_p / n_total
        Q[k] = sum_pp / n_total
        J[k] = sum_pp.diagonal(axis1=0, axis2=1).T - n_total * (m[k] ** 2)
        for i in range(N):
            denom = Q[k, i, i] - m[k, i] ** 2
            denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
            for j in range(N):
                if i == j:
                    r[k, i, j] = 1.0
                else:
                    r[k, i, j] = (Q[k, i, j] - m[k, i] * m[k, j]) / denom

    return {"m": m, "Q": Q, "J": J, "r": r}
