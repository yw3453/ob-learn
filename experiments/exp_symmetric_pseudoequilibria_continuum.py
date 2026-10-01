"""Continuum of pseudo-equilibria in the *symmetric* duopoly.

Produces the paper's pseudo-equilibria-continuum figure via
:func:`run_comparison` (``ob-learn run pseudoequilibria-continuum``, or
``--comparison`` when run as a script). Under decaying exploration the set of
reachable pseudo-equilibria forms a *continuum* parameterised by the
regression ratios ``(r_1, r_2)``; we compute the theoretical reachable region
in price and revenue space and overlay the empirical long-run prices from
discrete-time runs started from a diverse spread of warm-up prices.

For the symmetric baseline duopoly (``alpha = 2.5``, ``beta = -1``,
``gamma = 0.4``):

* the theoretical continuum of pseudo-equilibria is symmetric around the diagonal,
* the empirical cloud sits on the ``p_1 = p_2`` diagonal,
* the collusive marker ``p^C = (2.083, 2.083)`` is *on* that diagonal and
  is therefore approximately surrounded by the empirical points.

Outputs of :func:`run_comparison` (exported to ``results/figures/`` and
``results/tables/``):

* ``fig_pseudoequilibria_continuum_{symexplore,asymexplore}_region.pdf`` -- price-space scatter.
* ``fig_pseudoequilibria_continuum_{symexplore,asymexplore}_revenue.pdf`` -- surplus-capture scatter.
* ``table_symmetric_pseudoequilibria_continuum_summary_{symexplore,asymexplore}`` -- empirical summary statistics.
"""

from __future__ import annotations

import _common as C  # type: ignore[import-not-found]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ob_learn import market
from ob_learn.artifact_export import export_figure, export_table
from ob_learn.config import (
    ExperimentConfig,
    ExplorationSchedule,
    InformedProjectionBox,
    SellerSpec,
)
from ob_learn.logging_utils import run_directory
from ob_learn.plotting import SQUARE_FIGSIZE, report_style, smart_legend, square_box
from ob_learn.simulator import run_simulation

# ---------------------------------------------------------------------------
# Theoretical-region helpers.
# ---------------------------------------------------------------------------


def _admissible_r(r_grid: np.ndarray, *, ub_r1: float, ub_r2: float) -> np.ndarray:
    """Boolean ``(R, R)`` mask of admissible ``(r_1, r_2)`` pairs.

    Admissibility conditions are ``r_i < ub_i``, plus ``r_1 = r_2 = 0`` *or*
    ``0 < r_1 r_2 <= 1`` (the Cauchy-Schwarz constraint). The upper bounds
    ``ub_i = -beta_i / gamma_{i,j}`` are where the misspecified slope
    ``beta + gamma r`` flips sign (greedy price ill-defined). ``r_i`` is the
    unstandardized covariance-to-variance ratio and can exceed 1, so the grid
    should extend up to ``max(ub_r1, ub_r2)``.
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
    pseudo-equilibrium price pairs and the corresponding per-period revenue
    pairs. In addition to the interior sweep, the boundary ``r_1 r_2 = 1``
    (where the joint-profit collusive price lives) is sampled densely.
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
    """Generate a diverse set of warm-up price pairs spanning the price box."""
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

    If ``xlim`` / ``ylim`` are provided they are used verbatim (so sibling
    panels share the same geometric view); otherwise the zoom is computed from
    the empirical scatter + benchmarks with relative padding ``pad``. The
    theoretical-region scatter is rasterized to keep the PDF lean, and explicit
    ``subplots_adjust`` margins (with ``export_figure(tight_bbox=False)``)
    guarantee identical page dimensions across calls.
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
        fig.subplots_adjust(left=0.17, right=0.97, bottom=0.11, top=0.97)
    return fig


def main(
    *,
    horizon: int = 80_000,
    n_seeds: int = 30,
    base_seed: int = 833,
    c: float = 0.3,
    eta: float = 0.85,
    eta2: float | None = None,
    c2: float | None = None,
    name_tag: str | None = None,
    quick: bool = False,
    xlim_price: tuple[float, float] | None = None,
    ylim_price: tuple[float, float] | None = None,
    xlim_revenue: tuple[float, float] | None = None,
    ylim_revenue: tuple[float, float] | None = None,
) -> dict:
    """Run the pseudo-equilibria-continuum experiment on the *symmetric* duopoly.

    Exploration is controlled per seller. Seller 0 always uses
    ``nu_n^2 = c (n+1)^{-eta}``; seller 1 uses ``c2 (n+1)^{-eta2}`` (defaulting
    to the same schedule as seller 0, i.e.\\ the *symmetric-exploration* case).
    Setting ``eta2 != eta`` (or ``c2 != c``) yields the *asymmetric-exploration*
    case on the same symmetric demand: one seller eventually accumulates more
    price variance and becomes variance-dominant.

    Optional ``xlim_*`` / ``ylim_*`` arguments override the per-call
    auto-zoom in the plot rendering, so a caller can enforce *shared* axes
    across variants. ``name_tag`` overrides the output-file suffix (used by
    :func:`run_comparison`). Returns a dict with the empirical price- and
    revenue-clouds so a caller can compute a shared axis window across runs.
    """
    horizon, n_seeds = C.quick_overrides(quick, default_T=horizon, default_S=n_seeds)
    d = C.baseline_demand()  # symmetric: alpha=2.5, beta=-1, gamma=0.4
    c2 = c if c2 is None else c2
    eta2 = eta if eta2 is None else eta2
    asymmetric = (eta2 != eta) or (c2 != c)
    seller_scheds = [
        ExplorationSchedule(kind="polynomial", c=c, eta=eta),
        ExplorationSchedule(kind="polynomial", c=c2, eta=eta2),
    ]

    def _make_sellers():
        return [
            SellerSpec(kind="oblivious", exploration=seller_scheds[i])
            for i in range(d.N)
        ]
    box_ob = C.tight_oblivious_box(d, expand=0.6)
    p_NE = market.nash_prices(d)
    p_C = market.collusive_prices(d)
    pi_NE = market.per_period_revenue(d, p_NE)
    pi_C = market.per_period_revenue(d, p_C)

    g12 = float(d.gamma_arr[0, 1])
    g21 = float(d.gamma_arr[1, 0])
    r_hi = max(-float(d.beta_arr[0]) / g12, -float(d.beta_arr[1]) / g21)
    # The theoretical-region scatter is rasterized inside ``_plot_region`` to
    # keep the PDF file size small.
    r_grid = np.linspace(-r_hi, r_hi, 1201)
    theory_prices, theory_revenues = _theoretical_region(d, r_grid=r_grid)

    # Tag outputs so multiple schedule choices coexist in ``results/figures/``.
    # An explicit ``name_tag`` (from ``run_comparison``) wins; otherwise the
    # canonical symmetric run keeps the un-tagged names so default outputs stay
    # stable, and other schedules get an eta-based suffix.
    if name_tag is not None:
        eta_tag = name_tag
    elif asymmetric:
        eta_tag = f"_asym_eta_{eta:g}_{eta2:g}".replace(".", "p")
    elif eta == 0.85 and c == 0.3:
        eta_tag = ""
    else:
        eta_tag = f"_eta_{eta:g}".replace(".", "p")
    exp_name = f"exp_symmetric_pseudoequilibria_continuum{eta_tag}"

    cfg = ExperimentConfig(
        name=exp_name,
        market=d,
        sellers=_make_sellers(),
        oblivious_projection=box_ob,
        informed_projection=InformedProjectionBox.from_demand(d),
        horizon=horizon,
        n_seeds=n_seeds,
        base_seed=base_seed,
        log_every=max(1, horizon // 500),
    )

    with run_directory(exp_name, cfg) as run:
        run.logger.info(
            "symmetric duopoly: alpha=%s beta=%s gamma=%s",
            d.alpha, d.beta, d.gamma,
        )
        run.logger.info(
            "exploration: seller0 nu^2=%g(n+1)^-%g, seller1 nu^2=%g(n+1)^-%g (asymmetric=%s)",
            c, eta, c2, eta2, asymmetric,
        )
        run.logger.info("p_NE=%s p_C=%s", p_NE.tolist(), p_C.tolist())
        run.logger.info(
            "continuum restrictions: r_1 < %.3f, r_2 < %.3f",
            -d.beta[0] / d.gamma[0][1],
            -d.beta[1] / d.gamma[1][0],
        )
        run.logger.info(
            "theoretical region: %d admissible pseudo-equilibria",
            theory_prices.shape[0],
        )

        warmups = _build_warmups(d, n_configs=60, base_seed=base_seed)
        empirical_prices: list[np.ndarray] = []
        empirical_revenues: list[np.ndarray] = []
        rows: list[dict] = []
        for k, warm in enumerate(warmups):
            sub_cfg = ExperimentConfig(
                name=f"warm_{k:03d}",
                market=d,
                sellers=_make_sellers(),
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
            final_m = res.moments["m"][-1]
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
        run.save_summary("symmetric_pseudoequilibria_seed_points", df)

        v = p_C - p_NE
        v_norm2 = float(v @ v)
        t = ((emp_prices - p_NE) @ v) / v_norm2
        t = np.clip(t, 0.0, 1.0)
        proj = p_NE[None, :] + t[:, None] * v[None, :]
        off_ridge = np.linalg.norm(emp_prices - proj, axis=1)
        ridge_p95 = float(np.percentile(off_ridge, 95))
        ridge_mean = float(off_ridge.mean())
        below_ne_any = ((emp_revenues[:, 0] < pi_NE[0]) | (emp_revenues[:, 1] < pi_NE[1])).mean()
        below_ne_both = ((emp_revenues[:, 0] < pi_NE[0]) & (emp_revenues[:, 1] < pi_NE[1])).mean()

        # Coverage of p^C: distance from p^C to the nearest empirical point, and
        # whether the empirical cloud's bounding box contains p^C.
        dist_to_pC = np.linalg.norm(emp_prices - p_C[None, :], axis=1).min()
        bb_lo = emp_prices.min(axis=0)
        bb_hi = emp_prices.max(axis=0)
        pC_in_bbox = bool((bb_lo[0] <= p_C[0] <= bb_hi[0]) and (bb_lo[1] <= p_C[1] <= bb_hi[1]))

        run.log_event(
            "sym_continuum_summary",
            n_points=int(emp_prices.shape[0]),
            off_ridge_p95=ridge_p95,
            off_ridge_mean=ridge_mean,
            frac_below_NE_any_seller=float(below_ne_any),
            frac_below_NE_both_sellers=float(below_ne_both),
            min_dist_to_pC=float(dist_to_pC),
            pC_in_empirical_bbox=pC_in_bbox,
        )

        summary_df = pd.DataFrame(
            [
                dict(statistic="num empirical points", value=float(emp_prices.shape[0])),
                dict(statistic="std($\\bar p_1$)", value=float(df["final_p1"].std())),
                dict(statistic="std($\\bar p_2$)", value=float(df["final_p2"].std())),
                dict(
                    statistic="mean orthogonal distance from $p^{NE}$--$p^{C}$ segment",
                    value=ridge_mean,
                ),
                dict(
                    statistic="95\\%-ile orthogonal distance from $p^{NE}$--$p^{C}$ segment",
                    value=ridge_p95,
                ),
                dict(
                    statistic="fraction with either-seller revenue $<\\Pi^{NE}$",
                    value=float(below_ne_any),
                ),
                dict(
                    statistic="fraction with both-seller revenue $<\\Pi^{NE}$",
                    value=float(below_ne_both),
                ),
                dict(
                    statistic="min Euclidean distance to $p^{C}$",
                    value=float(dist_to_pC),
                ),
            ]
        )
        export_table(
            summary_df, f"table_symmetric_pseudoequilibria_continuum_summary{eta_tag}",
            caption=(
                "Long-run prices and revenues in the *symmetric* duopoly "
                "($\\alpha=2.5$, $\\beta=-1$, $\\gamma=0.4$) under per-seller "
                f"exploration $\\nu_{{n,1}}^2 = {c:g}(n+1)^{{-{eta:g}}}$ and "
                f"$\\nu_{{n,2}}^2 = {c2:g}(n+1)^{{-{eta2:g}}}$ "
                f"({'asymmetric' if asymmetric else 'symmetric'} exploration), "
                "started from a diverse spread of warm-up price pairs."
            ),
            floatfmt=".4g",
        )

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
            xlim=xlim_price, ylim=ylim_price,
        )
        run.save_figure("symmetric_pseudoequilibria_region", fig_price)

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
            xlim=xlim_revenue, ylim=ylim_revenue,
        )
        run.save_figure("symmetric_pseudoequilibria_revenue", fig_rev)

        run.logger.info(
            "exp_symmetric_pseudoequilibria_continuum: %d empirical points, mean off-ridge distance = %.4f, "
            "95%%-ile = %.4f, fraction with revenue below Pi_NE: any=%.2f both=%.2f, "
            "min dist to p^C = %.4f, p^C in bbox: %s",
            int(emp_prices.shape[0]),
            ridge_mean, ridge_p95, below_ne_any, below_ne_both,
            dist_to_pC, pC_in_bbox,
        )

    return {
        "eta": eta,
        "eta2": eta2,
        "asymmetric": asymmetric,
        "eta_tag": eta_tag,
        "emp_prices": emp_prices,
        "emp_revenues": emp_revenues,
        "theory_prices": theory_prices,
        "theory_revenues": theory_revenues,
        "p_NE": p_NE,
        "p_C": p_C,
        "pi_NE": pi_NE,
        "pi_C": pi_C,
        "frac_below_NE_any": float(below_ne_any),
        "frac_below_NE_both": float(below_ne_both),
        "frac_both_above_NE": float(
            ((emp_revenues[:, 0] >= pi_NE[0]) & (emp_revenues[:, 1] >= pi_NE[1])).mean()
        ),
    }


def _expand_box(
    points_list: list[np.ndarray],
    benchmarks: list[tuple[float, float]],
    *,
    pad: float = 0.08,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute axis (xlim, ylim) covering the union of all empirical clouds
    plus benchmark points, with relative padding ``pad`` on each side."""
    bench = np.array(benchmarks)
    xs = np.concatenate([pts[:, 0] for pts in points_list] + [bench[:, 0]])
    ys = np.concatenate([pts[:, 1] for pts in points_list] + [bench[:, 1]])
    x_lo, x_hi = float(xs.min()), float(xs.max())
    y_lo, y_hi = float(ys.min()), float(ys.max())
    x_pad = pad * (x_hi - x_lo + 1e-9)
    y_pad = pad * (y_hi - y_lo + 1e-9)
    return (x_lo - x_pad, x_hi + x_pad), (y_lo - y_pad, y_hi + y_pad)


def _rerender_from_cache(
    result: dict,
    *,
    tag: str,
    xlim_price: tuple[float, float],
    ylim_price: tuple[float, float],
    xlim_revenue: tuple[float, float],
    ylim_revenue: tuple[float, float],
) -> None:
    """Re-plot the price/revenue scatters from cached arrays (no re-simulation)."""
    p_NE, p_C = result["p_NE"], result["p_C"]
    pi_NE, pi_C = result["pi_NE"], result["pi_C"]
    fig_price = _plot_region(
        title_suffix="segment",
        theory=result["theory_prices"],
        empirical=result["emp_prices"],
        benchmarks={"NE": (float(p_NE[0]), float(p_NE[1])),
                    "C":  (float(p_C[0]),  float(p_C[1]))},
        xlabel=r"$\bar p_1$", ylabel=r"$\bar p_2$",
        xlim=xlim_price, ylim=ylim_price,
    )
    export_figure(fig_price, f"fig_pseudoequilibria_continuum_{tag}_region",
                  strip_title=True, tight_bbox=False, dpi=300)
    plt.close(fig_price)
    # Report revenue as the (unit-free) surplus-capture ratio S_i, consistent
    # with the paper's performance metric: S_i = (Pi_i - Pi_i^NE)/(Pi_i^C - Pi_i^NE),
    # so Nash -> 0 and collusive -> 1. This is an affine rescaling of the raw
    # per-period revenue, so the cloud geometry (and the above/below-Nash
    # fractions) is unchanged; only the axes are normalized.
    pi_NE = np.asarray(pi_NE, dtype=float)
    pi_C = np.asarray(pi_C, dtype=float)
    denom = pi_C - pi_NE

    def _to_surplus(arr: np.ndarray) -> np.ndarray:
        return (np.asarray(arr, dtype=float) - pi_NE) / denom

    s_xlim = ((xlim_revenue[0] - pi_NE[0]) / denom[0],
              (xlim_revenue[1] - pi_NE[0]) / denom[0])
    s_ylim = ((ylim_revenue[0] - pi_NE[1]) / denom[1],
              (ylim_revenue[1] - pi_NE[1]) / denom[1])
    fig_rev = _plot_region(
        title_suffix="segment",
        theory=_to_surplus(result["theory_revenues"]),
        empirical=_to_surplus(result["emp_revenues"]),
        benchmarks={"NE": (0.0, 0.0), "C": (1.0, 1.0)},
        xlabel=r"$S_1$", ylabel=r"$S_2$",
        xlim=s_xlim, ylim=s_ylim,
    )
    export_figure(fig_rev, f"fig_pseudoequilibria_continuum_{tag}_revenue",
                  strip_title=True, tight_bbox=False, dpi=300)
    plt.close(fig_rev)


def run_comparison(
    *,
    horizon: int = 80_000,
    n_seeds: int = 30,
    base_seed: int = 833,
    c: float = 0.3,
    eta_sym: float = 0.85,
    eta_dominant: float = 0.2,
    eta_dominated: float = 0.9,
    quick: bool = False,
) -> None:
    """Produce the paper's pseudo-equilibria comparison figure: symmetric vs asymmetric exploration.

    Both panels use the *same symmetric demand*; they differ only in the
    per-seller exploration schedule (``nu_n^2 = c(n+1)^{-eta}``, so a smaller
    ``eta`` decays more slowly and accumulates more price variance):

    * ``symexplore``: both sellers use exponent ``eta_sym``,
    * ``asymexplore``: seller 0 uses the slower-decaying ``eta_dominant`` and
      seller 1 the faster-decaying ``eta_dominated``, so seller 0 accumulates
      more price variance and becomes variance-dominant.

    Renders four PDFs (``fig_pseudoequilibria_continuum_{symexplore,asymexplore}_{region,revenue}``)
    on *shared* axes so the two regimes are directly comparable.
    """
    sym = main(
        horizon=horizon, n_seeds=n_seeds, base_seed=base_seed,
        c=c, eta=eta_sym, eta2=eta_sym, name_tag="_symexplore", quick=quick,
    )
    asym = main(
        horizon=horizon, n_seeds=n_seeds, base_seed=base_seed,
        c=c, eta=eta_dominant, eta2=eta_dominated, name_tag="_asymexplore", quick=quick,
    )
    price_xlim, price_ylim = _expand_box(
        [sym["emp_prices"], asym["emp_prices"]],
        [tuple(sym["p_NE"]), tuple(sym["p_C"])],
    )
    rev_xlim, rev_ylim = _expand_box(
        [sym["emp_revenues"], asym["emp_revenues"]],
        [tuple(sym["pi_NE"]), tuple(sym["pi_C"])],
    )
    for tag, res in (("symexplore", sym), ("asymexplore", asym)):
        _rerender_from_cache(
            res, tag=tag,
            xlim_price=price_xlim, ylim_price=price_ylim,
            xlim_revenue=rev_xlim, ylim_revenue=rev_ylim,
        )
    print("=== comparison summary (fraction of seeds) ===")
    for tag, res in (("symmetric ", sym), ("asymmetric", asym)):
        print(
            f"  {tag} exploration: both-above-NE={res['frac_both_above_NE']:.3f}  "
            f"any-below-NE={res['frac_below_NE_any']:.3f}  "
            f"both-below-NE={res['frac_below_NE_both']:.3f}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", action="store_true",
                        help="run the symmetric-vs-asymmetric exploration comparison")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--horizon", type=int, default=80_000)
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--eta-dominant", type=float, default=0.2)
    parser.add_argument("--eta-dominated", type=float, default=0.9)
    args = parser.parse_args()

    if args.comparison:
        run_comparison(
            horizon=args.horizon, n_seeds=args.n_seeds,
            eta_dominant=args.eta_dominant, eta_dominated=args.eta_dominated,
            quick=args.quick,
        )
    else:
        main(quick=args.quick, horizon=args.horizon, n_seeds=args.n_seeds)
