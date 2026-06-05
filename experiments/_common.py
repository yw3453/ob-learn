"""Shared helpers for the experiment scripts.

Experiments live as standalone Python scripts under ``experiments/`` and pull
shared configuration defaults from this module. Defaults match the symmetric
duopoly used throughout the experiment suite.
"""

from __future__ import annotations

from src.config import (
    DemandParams,
    ExperimentConfig,
    ExplorationSchedule,
    InformedProjectionBox,
    ProjectionBox,
    SellerSpec,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

BASELINE_PARAMS = dict(N=2, alpha=2.5, beta=-1.0, gamma=0.4, l=0.5, u=2.5, noise_std=0.2)
"""Default symmetric-duopoly demand parameters."""


def baseline_demand() -> DemandParams:
    return DemandParams.symmetric(**BASELINE_PARAMS)


def tight_oblivious_box(d: DemandParams, *, expand: float = 0.5) -> ProjectionBox:
    """Tighter projection box than :meth:`ProjectionBox.from_demand`.

    For experiments that probe the convergence threshold, ``L_phi^{ob}``
    matters and a generous box inflates it. Here we size the box so that each
    seller's pseudo-true ``(a^*, b^*)`` is contained with ``expand`` relative
    margin on each side (default: ``[0.5, 1.5] * a^*`` and so on).
    """
    from src import market as m

    p_NE = m.nash_prices(d)
    target = m.pseudo_true_oblivious(d, p_NE)  # (N, 2)
    a_low = []
    a_high = []
    b_low = []
    b_high = []
    for i in range(d.N):
        a, b = float(target[i, 0]), float(target[i, 1])
        a_low.append(max(1e-6, a * (1 - expand)))
        a_high.append(a * (1 + expand))
        b_low.append(b * (1 + expand))  # more negative
        b_high.append(b * (1 - expand))  # less negative (still negative)
    return ProjectionBox(a_low=a_low, a_high=a_high, b_low=b_low, b_high=b_high)


def tight_informed_box(
    d: DemandParams,
    *,
    beta_abs_min_frac: float = 0.95,
    alpha_margin: float = 2.0,
    gamma_margin: float = 2.0,
    beta_low_margin: float = 2.0,
) -> InformedProjectionBox:
    """Tighter informed projection box that clamps ``|beta_j|`` close to truth.

    The default :meth:`InformedProjectionBox.from_demand` lets each
    informed seller's beta-coordinate run up to ``-1e-6`` (essentially
    zero), which makes ``L_phi^{in,theta} ~ 1/(2 |beta|_min)`` enormous
    via the ``beta_abs_min`` term. For experiments that need
    ``L_phi^{in,theta}`` to be moderate (e.g., the mixed-market stress
    tests where the small-gain condition is on the line), set
    ``beta_abs_min_frac`` close to 1 to clamp ``|beta|`` near the true
    value:

    * ``alpha_j`` coordinate: ``[1e-6, alpha_margin * max(alpha)]``.
    * ``beta_j`` coordinate:
        ``[-beta_low_margin * max|beta|, -beta_abs_min_frac * min|beta|]``;
      the upper end (less negative) directly determines
      ``beta_abs_min_eff = beta_abs_min_frac * min|beta|`` used in the
      small-gain computation.
    * cross ``gamma_{j,k}`` coordinates:
        ``[1e-6, gamma_margin * max(gamma)]``.

    Callers that pass ``beta_abs_min`` to
    :func:`src.market.master_theorem_smallgain` should pass
    ``beta_abs_min_frac * min(|beta_i|)`` to stay consistent.
    """
    N = d.N
    max_alpha = max(d.alpha)
    max_gamma = max(d.gamma[i][j] for i in range(N) for j in range(N) if i != j)
    max_abs_beta = max(-b for b in d.beta)
    min_abs_beta = min(-b for b in d.beta)
    beta_high_neg = -beta_abs_min_frac * min_abs_beta  # less negative endpoint
    beta_low_neg = -beta_low_margin * max_abs_beta
    low = [[0.0] * (N + 1) for _ in range(N)]
    high = [[0.0] * (N + 1) for _ in range(N)]
    for i in range(N):
        low[i][0] = 1e-6
        high[i][0] = alpha_margin * max_alpha
        for k in range(1, N + 1):
            if (k - 1) == i:
                low[i][k] = beta_low_neg
                high[i][k] = beta_high_neg
            else:
                low[i][k] = 1e-6
                high[i][k] = gamma_margin * max_gamma
    return InformedProjectionBox(low=low, high=high)


def make_oblivious_sellers(
    n: int,
    schedule: ExplorationSchedule,
) -> list[SellerSpec]:
    return [SellerSpec(kind="oblivious", exploration=schedule) for _ in range(n)]


def make_mixed_duopoly(
    *,
    oblivious_schedule: ExplorationSchedule,
    informed_schedule: ExplorationSchedule,
    forecast_rule: str,
) -> list[SellerSpec]:
    """Seller 0 oblivious, seller 1 informed with the given forecast rule."""
    return [
        SellerSpec(kind="oblivious", exploration=oblivious_schedule),
        SellerSpec(
            kind="informed",
            forecast_rule=forecast_rule,  # type: ignore[arg-type]
            exploration=informed_schedule,
        ),
    ]


def make_mixed_sellers(
    *,
    n_ob: int,
    n_in: int,
    oblivious_schedule: ExplorationSchedule,
    informed_schedule: ExplorationSchedule,
    forecast_rule: str = "mean_price",
) -> list[SellerSpec]:
    """``N`` = ``n_ob + n_in`` sellers: first ``n_ob`` oblivious, then ``n_in`` informed.

    All oblivious sellers share ``oblivious_schedule``; all informed sellers
    share ``informed_schedule`` and ``forecast_rule``. The convention
    ``ob_idx = list(range(n_ob))``, ``in_idx = list(range(n_ob, n_ob + n_in))``
    is used elsewhere by the mixed-market experiment scripts to slice the
    market into the two groups.
    """
    if n_ob < 0 or n_in < 0:
        raise ValueError(f"n_ob={n_ob}, n_in={n_in}: must be non-negative")
    if n_ob + n_in == 0:
        raise ValueError("need at least one seller")
    sellers: list[SellerSpec] = []
    sellers.extend(
        SellerSpec(kind="oblivious", exploration=oblivious_schedule) for _ in range(n_ob)
    )
    sellers.extend(
        SellerSpec(
            kind="informed",
            forecast_rule=forecast_rule,  # type: ignore[arg-type]
            exploration=informed_schedule,
        )
        for _ in range(n_in)
    )
    return sellers


def base_config(
    *,
    name: str,
    market: DemandParams,
    sellers: list[SellerSpec],
    horizon: int,
    n_seeds: int,
    base_seed: int = 42,
    log_every: int = 1,
    oblivious_box: ProjectionBox | None = None,
    informed_box: InformedProjectionBox | None = None,
) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        market=market,
        sellers=sellers,
        oblivious_projection=oblivious_box or tight_oblivious_box(market),
        informed_projection=informed_box or InformedProjectionBox.from_demand(market),
        horizon=horizon,
        n_seeds=n_seeds,
        base_seed=base_seed,
        log_every=log_every,
    )


def quick_overrides(quick: bool, *, default_T: int, default_S: int) -> tuple[int, int]:
    """Helper for ``--quick`` smoke runs."""
    if quick:
        return min(default_T, 5_000), min(default_S, 16)
    return default_T, default_S


# ---------------------------------------------------------------------------
# Asymmetric / N>2 market factories shared by multiple experiments
# ---------------------------------------------------------------------------


def asymmetric_duopoly() -> DemandParams:
    """Asymmetric duopoly used by the pseudo-equilibria-continuum experiments.

    Sellers differ in vertical demand (``alpha_2 > alpha_1``), price
    sensitivity (``|beta_2| > |beta_1|``), and competition asymmetry
    (``gamma_{2,1} > gamma_{1,2}``).
    """
    return DemandParams(
        N=2,
        alpha=[2.5, 3.0],
        beta=[-1.0, -1.2],
        gamma=[[0.0, 0.4], [0.5, 0.0]],
        l=0.5,
        u=2.5,
        noise_kind="uniform",
        noise_std=0.2,
    )


def symmetric_market(N: int, *, gamma: float = 0.4, noise_std: float = 0.2) -> DemandParams:
    """Symmetric ``N``-seller market with the baseline structural parameters."""
    return DemandParams.symmetric(
        N=N,
        alpha=2.5,
        beta=-1.0,
        gamma=gamma,
        l=0.5,
        u=2.5,
        noise_std=noise_std,
    )


def revenue_duopoly() -> DemandParams:
    """More competitive symmetric duopoly used in the revenue experiments.

    The baseline symmetric duopoly (``\\gamma = 0.4``) has a Nash–collusive
    revenue gap of only ``\\Pi^{C} - \\Pi^{NE} \\approx 0.16``, which makes
    composition effects hard to see. This variant uses ``\\gamma = 0.6``
    and a wider price box ``[0.5, 3.5]`` so

    * ``p^{NE} = 1.786``, ``p^{C} = 3.125``,
    * ``\\Pi^{NE} = 3.19``, ``\\Pi^{C} = 3.91``  (gap $\\approx 0.72$),

    giving each revenue regime a much more visible signature.
    """
    return DemandParams.symmetric(
        N=2,
        alpha=2.5,
        beta=-1.0,
        gamma=0.6,
        l=0.5,
        u=3.5,
        noise_std=0.2,
    )


def asymmetric_market(N: int, *, base_seed: int = 0, noise_std: float = 0.2) -> DemandParams:
    """Asymmetric ``N``-seller market.

    Each seller gets a slightly different ``alpha_i``, ``beta_i`` and
    ``gamma_{i,j}`` so the resulting Nash and collusive prices are
    heterogeneous. We tile the duopoly's ``[α, β, γ]`` heterogeneity
    pattern and rescale the cross-effects by ``1/(N-1)`` to preserve
    diagonal dominance.
    """
    import numpy as np
    rng = np.random.default_rng(base_seed + 7919 * N)
    alpha = 2.5 + 0.4 * rng.standard_normal(N)
    alpha = np.clip(alpha, 1.5, 3.5)
    beta = -1.0 - 0.2 * rng.standard_normal(N)
    beta = np.clip(beta, -1.5, -0.7)
    base_gamma = 0.4 / max(N - 1, 1)
    G = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            G[i, j] = base_gamma * (0.7 + 0.6 * rng.random())
    return DemandParams(
        N=N,
        alpha=alpha.tolist(),
        beta=beta.tolist(),
        gamma=G.tolist(),
        l=0.5,
        u=2.5,
        noise_kind="uniform",
        noise_std=noise_std,
    )
