"""Continuum of pseudo-equilibria in the asymmetric duopoly.

Under decaying exploration the set of reachable pseudo-equilibria forms a
*continuum* parameterised by the regression ratios ``(r_1, r_2)`` solving

.. math::
   (2\\beta_1 + \\gamma_{1,2} r_1) m_1 + \\gamma_{1,2} m_2 &= -\\alpha_1,\\\\
   \\gamma_{2,1} m_1 + (2\\beta_2 + \\gamma_{2,1} r_2) m_2 &= -\\alpha_2.

The admissible ``(r_1, r_2)`` are not the entire ``[-1, 1]^2`` square but
rather

.. math::
   r_1 < -\\beta_1 / \\gamma_{1,2}, \\quad r_2 < -\\beta_2 / \\gamma_{2,1},
   \\quad \\text{and either } r_1 = r_2 = 0
   \\text{ or } 0 < r_1 r_2 \\le 1.

These restrictions reflect realistic values of the empirical regression
ratios and produce a connected pseudo-equilibrium region in price space.

This experiment

1. computes the *theoretical* reachable region in *price* space and in
   *revenue* space using restrictions on ``(r_1, r_2)``; and
2. runs the discrete-time dynamics on the asymmetric duopoly under
   slowly-decaying exploration with a diverse set of warm-up prices, then
   overlays the empirical limit prices and revenues on the theoretical
   regions.
"""

from __future__ import annotations

import _common as C  # type: ignore[import-not-found]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import market
from src.artifact_export import export_figure, export_table
from src.config import (
    ExperimentConfig,
    ExplorationSchedule,
    InformedProjectionBox,
    SellerSpec,
)
from src.logging_utils import run_directory
from src.plotting import SQUARE_FIGSIZE, report_style, smart_legend, square_box
from src.simulator import run_simulation


def _admissible_r(r_grid: np.ndarray, *, ub_r1: float, ub_r2: float) -> np.ndarray:
    """Boolean ``(R, R)`` mask of admissible ``(r_1, r_2)`` pairs.

    Admissibility conditions are ``r_i < ub_i``, plus ``r_1 = r_2 = 0`` *or*
    ``0 < r_1 r_2 <= 1``. We grid this on ``r_grid x r_grid``.

    The product constraint ``r_1 r_2 <= 1`` (combined with the
    same-sign requirement so the product is positive) is the
    *Cauchy-Schwarz* constraint: writing ``r_{i,j} = \\sigma_{ij} /
    \\sigma_i^2``, we have ``r_{1,2} r_{2,1} = \\rho^2`` where
    ``\\rho`` is the correlation, so ``r_1 r_2 \\in [0, 1]`` is
    automatic. The *upper bounds* ``r_i < ub_i = -\\beta_i /
    \\gamma_{i,j}`` are where the misspecified slope ``\\beta + \\gamma
    r`` would flip sign (greedy price ill-defined), so they are the
    relevant economic constraint.

    Importantly the regression slope ``r_i`` is the *unstandardized*
    covariance-to-variance ratio, **not** the correlation: ``r_i`` can
    exceed 1. We therefore use a wide ``r_grid`` that extends up to
    ``max(ub_r1, ub_r2)``; in particular the joint-profit collusive
    price corresponds to ``(r_1, r_2) \\approx (1.18, 0.85)`` for the
    asymmetric duopoly, which a naive ``[-1, 1]^2`` grid would miss.
    """
    r1m, r2m = np.meshgrid(r_grid, r_grid, indexing="ij")
    bounds = (r1m < ub_r1) & (r2m < ub_r2)
    product = r1m * r2m
    same_sign = (product > 0.0) & (product <= 1.0)
    zero = (np.abs(r1m) < 1e-9) & (np.abs(r2m) < 1e-9)
    return bounds & (same_sign | zero)


def _theoretical_region(d, *, r_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sweep admissible ``(r_1, r_2)`` and solve the 2x2 linear system.

    Returns ``(prices, revenues)`` of shape ``(M, 2)`` containing the
    pseudo-equilibrium price pairs and the corresponding per-period
    revenue pairs.

    In addition to the ``r_grid x r_grid`` interior sweep, we densely
    sample the *boundary* ``r_1 r_2 = 1`` (the upper-Cauchy-Schwarz
    edge): this is where the joint-profit collusive price lives, so
    if we sample the boundary only via the discretised interior grid
    we miss the collusive corner. Boundary samples use a finer
    spacing along ``r_1``.
    """
    alpha = d.alpha_arr
    beta = d.beta_arr
    G = d.gamma_arr
    g12 = float(G[0, 1])
    g21 = float(G[1, 0])
    ub_r1 = -float(beta[0]) / g12  # > 0 since beta<0, gamma>0
    ub_r2 = -float(beta[1]) / g21
    mask = _admissible_r(r_grid, ub_r1=ub_r1, ub_r2=ub_r2)

    def _try_pair(r1: float, r2: float) -> tuple[np.ndarray, np.ndarray] | None:
        A = np.array(
            [
                [2.0 * beta[0] + g12 * r1, g12],
                [g21, 2.0 * beta[1] + g21 * r2],
            ],
            dtype=np.float64,
        )
        b = -alpha
        det = np.linalg.det(A)
        if abs(det) < 1e-10:
            return None
        m = np.linalg.solve(A, b)
        if (m < d.l).any() or (m > d.u).any():
            return None
        rev = market.per_period_revenue(d, m)
        return m, rev

    prices: list[np.ndarray] = []
    revenues: list[np.ndarray] = []
    for i, r1 in enumerate(r_grid):
        for j, r2 in enumerate(r_grid):
            if not mask[i, j]:
                continue
            out = _try_pair(float(r1), float(r2))
            if out is None:
                continue
            m, rev = out
            prices.append(m)
            revenues.append(rev)

    # Dense boundary sweep along r_1 r_2 = 1 (joint-profit boundary).
    # Step ~0.001 in r_1, so the boundary is essentially a line in the
    # output scatter and the collusive corner is reached densely.
    eps = 1e-3
    r1_boundary = np.concatenate([
        np.linspace(eps, ub_r1 - eps, 2000),
        np.linspace(-ub_r1 + eps, -eps, 2000),
    ])
    for r1 in r1_boundary:
        r2 = 1.0 / float(r1)
        if r2 >= ub_r2 or r2 <= -ub_r2:
            continue
        if r1 * r2 <= 0.0:
            continue
        out = _try_pair(float(r1), float(r2))
        if out is None:
            continue
        m, rev = out
        prices.append(m)
        revenues.append(rev)

    return np.asarray(prices), np.asarray(revenues)


def _build_warmups(d, *, n_configs: int, base_seed: int) -> list[list[list[float]]]:
    """Generate a diverse set of warm-up price pairs.

    Each entry is a pair of length-2 price vectors. Centres span the
    price box (low/high/diagonal/near-NE/near-collusion). Per-seed jitter
    is injected by the simulator's demand-noise stream.
    """
    rng = np.random.default_rng(base_seed)
    centers = [
        (d.l + 0.2, d.l + 0.2),
        (d.u - 0.2, d.u - 0.2),
        (d.l + 0.2, d.u - 0.2),
        (d.u - 0.2, d.l + 0.2),
        (1.5, 1.5),
        (2.0, 2.0),
    ]
    out: list[list[list[float]]] = []
    for cx, cy in centers:
        for _ in range(max(1, n_configs // len(centers))):
            jitter = rng.uniform(-0.15, 0.15, size=(2, 2))
            p1 = [float(np.clip(cx + jitter[0, 0], d.l + 0.05, d.u - 0.05)),
                  float(np.clip(cy + jitter[0, 1], d.l + 0.05, d.u - 0.05))]
            p2 = [float(np.clip(cx + jitter[1, 0], d.l + 0.05, d.u - 0.05)),
                  float(np.clip(cy + jitter[1, 1], d.l + 0.05, d.u - 0.05))]
            out.append([p1, p2])
    return out


def _plot_region(
    *,
    title_suffix: str,
    theory: np.ndarray,  # (M, 2) theoretical points
    empirical: np.ndarray,  # (K, 2) empirical points
    benchmarks: dict[str, tuple[float, float]],
    xlabel: str,
    ylabel: str,
    pad: float = 0.08,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
):
    """Plot a 2D scatter: theoretical region (shaded) + empirical points + benchmarks.

    If ``xlim`` and ``ylim`` are provided, the axes use those windows
    verbatim; this is the recommended way to make sibling panels (e.g.
    one panel per exploration schedule, same demand) share the same
    geometric view of the theoretical region. If they are ``None``, the
    zoom is computed per-call from the empirical scatter + benchmarks
    with relative padding ``pad``.

    The theoretical-region scatter is ``rasterized=True``: at the dense
    grid sizes needed for visually uniform shading at high-zoom panels,
    a pure vector scatter contains hundreds of thousands of circles and
    blows up the PDF file size. Rasterizing the background keeps the
    file lean while leaving every foreground element (empirical dots,
    benchmark markers, dashed ridge line, axes, legend) as vector.

    The plot uses explicit ``fig.subplots_adjust`` margins instead of
    ``tight_layout``. Combined with ``export_figure(tight_bbox=False)``,
    this guarantees that every PDF produced by this routine has *exactly*
    the same page dimensions regardless of tick-label widths, so LaTeX
    subfigures using the same ``\\linewidth`` render at the same height.
    """
    if xlim is None or ylim is None:
        bench_xy = np.array(list(benchmarks.values()))
        xs = np.concatenate([empirical[:, 0], bench_xy[:, 0]])
        ys = np.concatenate([empirical[:, 1], bench_xy[:, 1]])
        x_lo, x_hi = float(xs.min()), float(xs.max())
        y_lo, y_hi = float(ys.min()), float(ys.max())
        x_pad = pad * (x_hi - x_lo + 1e-9)
        y_pad = pad * (y_hi - y_lo + 1e-9)
        auto_xlim = (x_lo - x_pad, x_hi + x_pad)
        auto_ylim = (y_lo - y_pad, y_hi + y_pad)
        if xlim is None:
            xlim = auto_xlim
        if ylim is None:
            ylim = auto_ylim
    xl, xh = xlim
    yl, yh = ylim

    bench_colors = {
        "NE": "tab:red",
        "C": "tab:green",
        "Stackelberg": "tab:purple",
    }

    with report_style():
        fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
        ax.scatter(
            theory[:, 0], theory[:, 1],
            s=10, color="lightsteelblue", alpha=0.55, edgecolor="none",
            label="theoretical region", rasterized=True,
        )
        if "NE" in benchmarks and "C" in benchmarks:
            ne = benchmarks["NE"]
            cc = benchmarks["C"]
            ax.plot([ne[0], cc[0]], [ne[1], cc[1]],
                    color="tab:gray", linestyle="--", lw=1.2,
                    label=f"$p^{{NE}}$--$p^{{C}}$ {title_suffix}")
        ax.scatter(
            empirical[:, 0], empirical[:, 1],
            s=22, color="tab:blue", alpha=0.75, edgecolor="white", linewidth=0.4,
            label="empirical",
        )
        for name, pt in benchmarks.items():
            ax.scatter([pt[0]], [pt[1]], s=160,
                       color=bench_colors.get(name, "black"),
                       marker="X", edgecolor="white", linewidth=0.6, zorder=10,
                       label=(fr"$p^{{{name}}}$" if name in ("NE", "C") else "Stackelberg"))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xl, xh)
        ax.set_ylim(yl, yh)
        smart_legend(ax, fontsize=11)
        square_box(ax)
        # Explicit margins (no tight_layout) for reproducible page size
        # across calls; see docstring above.
        fig.subplots_adjust(left=0.13, right=0.97, bottom=0.11, top=0.97)
    return fig


def main(
    *,
    horizon: int = 80_000,
    n_seeds: int = 30,
    base_seed: int = 833,
    quick: bool = False,
) -> None:
    horizon, n_seeds = C.quick_overrides(quick, default_T=horizon, default_S=n_seeds)
    d = C.asymmetric_duopoly()
    box_ob = C.tight_oblivious_box(d, expand=0.6)
    p_NE = market.nash_prices(d)
    p_C = market.collusive_prices(d)
    pi_NE = market.per_period_revenue(d, p_NE)
    pi_C = market.per_period_revenue(d, p_C)

    # Theoretical pseudo-equilibrium region under admissibility restrictions on r.
    # ``r_i = cov / var_i`` is the *regression* slope (not the
    # correlation), so ``r_i`` is unbounded above by 1. We grid up to
    # the pseudo-equilibria upper bound ``ub_i = -beta_i / gamma_{i,j}`` (= 2.5 and
    # 2.4 for the asymmetric duopoly) on both axes. The Cauchy-Schwarz
    # constraint ``r_1 r_2 \in [0, 1]`` (same sign, product <= 1) then
    # confines the admissible region; in particular the joint-profit
    # collusive price ``p^C = (2.180, 2.068)`` corresponds to ``(r_1,
    # r_2) \approx (1.18, 0.85)`` and is *on the boundary* of the
    # continuum of pseudo-equilibria (``r_1 r_2 \approx 1``).
    g12 = float(d.gamma_arr[0, 1])
    g21 = float(d.gamma_arr[1, 0])
    r_hi = max(-float(d.beta_arr[0]) / g12, -float(d.beta_arr[1]) / g21)
    # 401 points over [-r_hi, r_hi] for the interior + 2x 2000-point
    # boundary sweep along r_1 r_2 = 1 (the joint-profit
    # Cauchy-Schwarz edge, which is where p^C lives). The
    # _theoretical_region helper takes care of the boundary sweep.
    # Denser interior grid (1201) than the historical 401, so the rendered
    # theoretical-region scatter looks uniformly filled even when a panel
    # is zoomed in tightly near the Nash equilibrium. The scatter is
    # rasterized inside ``_plot_region`` to keep the PDF file size small.
    r_grid = np.linspace(-r_hi, r_hi, 1201)
    theory_prices, theory_revenues = _theoretical_region(d, r_grid=r_grid)

    # Slowly-decaying exploration: eta close to 1 so the system freezes early.
    rep_sched = ExplorationSchedule(kind="polynomial", c=0.3, eta=0.85)
    cfg = ExperimentConfig(
        name="exp_asymmetric_pseudoequilibria_continuum",
        market=d,
        sellers=[SellerSpec(kind="oblivious", exploration=rep_sched) for _ in range(d.N)],
        oblivious_projection=box_ob,
        informed_projection=InformedProjectionBox.from_demand(d),
        horizon=horizon,
        n_seeds=n_seeds,
        base_seed=base_seed,
        log_every=max(1, horizon // 500),
    )

    with run_directory("exp_asymmetric_pseudoequilibria_continuum", cfg) as run:
        run.logger.info(
            "asymmetric: alpha=%s beta=%s gamma=%s",
            d.alpha, d.beta, d.gamma,
        )
        run.logger.info("p_NE=%s p_C=%s", p_NE.tolist(), p_C.tolist())
        run.logger.info(
            "continuum restrictions: r_1 < %.3f, r_2 < %.3f, with r_1 r_2 in {0} or (0, 1]",
            -d.beta[0] / d.gamma[0][1],
            -d.beta[1] / d.gamma[1][0],
        )
        run.logger.info("theoretical region: %d admissible pseudo-equilibria",
                        theory_prices.shape[0])

        # Sweep warm-up configurations to populate the (r_1, r_2) space.
        warmups = _build_warmups(d, n_configs=60, base_seed=base_seed)
        empirical_prices: list[np.ndarray] = []
        empirical_revenues: list[np.ndarray] = []
        rows: list[dict] = []
        for k, warm in enumerate(warmups):
            sub_cfg = ExperimentConfig(
                name=f"warm_{k:03d}",
                market=d,
                sellers=[SellerSpec(kind="oblivious", exploration=rep_sched) for _ in range(d.N)],
                oblivious_projection=box_ob,
                informed_projection=InformedProjectionBox.from_demand(d),
                horizon=horizon,
                n_seeds=n_seeds,
                base_seed=base_seed + 1000 * k,
                log_every=cfg.log_every,
                initial_prices=warm,
                n_warmup=len(warm),
            )
            run.logger.info("warm-up %d / %d: %s", k + 1, len(warmups), warm)
            res = run_simulation(sub_cfg)
            final_m = res.moments["m"][-1]  # (N, S)
            for s in range(n_seeds):
                price_pair = final_m[:, s].copy()
                rev_pair = market.per_period_revenue(d, price_pair)
                empirical_prices.append(price_pair)
                empirical_revenues.append(rev_pair)
                rows.append(
                    dict(
                        warmup_idx=k,
                        seed_idx=s,
                        warm_p1_avg=float(np.mean([row[0] for row in warm])),
                        warm_p2_avg=float(np.mean([row[1] for row in warm])),
                        final_p1=float(price_pair[0]),
                        final_p2=float(price_pair[1]),
                        revenue_1=float(rev_pair[0]),
                        revenue_2=float(rev_pair[1]),
                    )
                )
        emp_prices = np.array(empirical_prices)
        emp_revenues = np.array(empirical_revenues)

        df = pd.DataFrame(rows)
        run.save_summary("asymmetric_pseudoequilibria_seed_points", df)

        # Distance from each empirical point to the NE-C line segment in price space.
        v = p_C - p_NE
        v_norm2 = float(v @ v)
        t = ((emp_prices - p_NE) @ v) / v_norm2  # (K,)
        t = np.clip(t, 0.0, 1.0)
        proj = p_NE[None, :] + t[:, None] * v[None, :]
        off_ridge = np.linalg.norm(emp_prices - proj, axis=1)
        ridge_p95 = float(np.percentile(off_ridge, 95))
        ridge_mean = float(off_ridge.mean())
        # Fraction of empirical points with revenue below NE for either seller.
        below_ne_any = ((emp_revenues[:, 0] < pi_NE[0]) | (emp_revenues[:, 1] < pi_NE[1])).mean()
        below_ne_both = ((emp_revenues[:, 0] < pi_NE[0]) & (emp_revenues[:, 1] < pi_NE[1])).mean()

        run.log_event(
            "asym_continuum_summary",
            n_points=int(emp_prices.shape[0]),
            off_ridge_p95=ridge_p95,
            off_ridge_mean=ridge_mean,
            frac_below_NE_any_seller=float(below_ne_any),
            frac_below_NE_both_sellers=float(below_ne_both),
        )

        summary_df = pd.DataFrame(
            [
                dict(statistic="num empirical points", value=float(emp_prices.shape[0])),
                dict(statistic="std($\\bar p_1$)", value=float(df["final_p1"].std())),
                dict(statistic="std($\\bar p_2$)", value=float(df["final_p2"].std())),
                dict(statistic="mean orthogonal distance from $p^{NE}$--$p^{C}$ segment",
                     value=ridge_mean),
                dict(statistic="95\\%-ile orthogonal distance from $p^{NE}$--$p^{C}$ segment",
                     value=ridge_p95),
                dict(statistic="fraction with either-seller revenue $<\\Pi^{NE}$",
                     value=float(below_ne_any)),
                dict(statistic="fraction with both-seller revenue $<\\Pi^{NE}$",
                     value=float(below_ne_both)),
            ]
        )
        export_table(
            summary_df, "table_asymmetric_pseudoequilibria_continuum_summary",
            caption=(
                "Long-run prices and revenues in the asymmetric duopoly under "
                "$\\nu_n^2 = 0.3 (n+1)^{-0.85}$, started from a spread of warm-up "
                "price pairs. The off-ridge distance measures how far the "
                "empirical limits stray from the $p^{NE}$--$p^{C}$ segment; the "
                "last two rows report the fraction of seeds whose long-run "
                "revenue falls below $\\Pi^{NE}$."
            ),
            floatfmt=".4g",
        )

        # ----- Panel (a): theoretical region + empirical points in PRICE space.
        fig_price = _plot_region(
            title_suffix="segment",
            theory=theory_prices,
            empirical=emp_prices,
            benchmarks={
                "NE": (float(p_NE[0]), float(p_NE[1])),
                "C":  (float(p_C[0]),  float(p_C[1])),
            },
            xlabel=r"$\bar p_1$",
            ylabel=r"$\bar p_2$",
        )
        run.save_figure("asymmetric_pseudoequilibria_region", fig_price, close=False)
        # tight_bbox=False + the explicit subplots_adjust in _plot_region
        # guarantees identical page dimensions across calls; dpi=300 sets
        # the raster resolution of the rasterized theoretical-region scatter.
        export_figure(
            fig_price, "fig_asymmetric_pseudoequilibria_continuum_region",
            strip_title=True, tight_bbox=False, dpi=300,
        )

        # ----- Panel (b): theoretical region + empirical points in REVENUE space.
        fig_rev = _plot_region(
            title_suffix="segment",
            theory=theory_revenues,
            empirical=emp_revenues,
            benchmarks={
                "NE": (float(pi_NE[0]), float(pi_NE[1])),
                "C":  (float(pi_C[0]),  float(pi_C[1])),
            },
            xlabel=r"$\Pi_1(\bar p)$",
            ylabel=r"$\Pi_2(\bar p)$",
        )
        run.save_figure("asymmetric_pseudoequilibria_revenue", fig_rev, close=False)
        export_figure(
            fig_rev, "fig_asymmetric_pseudoequilibria_continuum_revenue",
            strip_title=True, tight_bbox=False, dpi=300,
        )

        run.logger.info(
            "exp_asymmetric_pseudoequilibria_continuum: %d empirical points, mean off-ridge distance = %.4f, "
            "95%%-ile = %.4f, fraction with revenue below Pi_NE: any=%.2f both=%.2f",
            int(emp_prices.shape[0]),
            ridge_mean, ridge_p95, below_ne_any, below_ne_both,
        )


if __name__ == "__main__":
    main()
