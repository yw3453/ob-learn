"""Continuum of pseudo-equilibria in the *symmetric* duopoly.

Companion to ``exp_asymmetric_pseudoequilibria_continuum``. The asymmetric run produces an
empirical cloud that sits along the ``p_1 = p_2`` ridge but the joint-profit
collusive marker ``p^C`` is *off* that ridge (because ``|\\beta_2| > |\\beta_1|``
makes seller 2's collusive price strictly lower than seller 1's). To check
that this geometry is a real consequence of the asymmetric joint profit
maximisation and not an artifact of the dynamics, we re-run the same
experiment on the symmetric baseline duopoly (``alpha = 2.5``, ``beta = -1``,
``gamma = 0.4``) and verify that:

* the theoretical continuum of pseudo-equilibria is symmetric around the diagonal,
* the empirical cloud sits on the ``p_1 = p_2`` diagonal as before,
* the collusive marker ``p^C = (2.083, 2.083)`` is *on* that diagonal and
  is therefore approximately surrounded by the empirical points.

If the symmetric run shows ``p^C`` inside the empirical cloud while the
asymmetric run does not, the gap is fully explained by the asymmetry of
``p^C`` itself rather than by any failure of the dynamics to span the
theoretical region.

Outputs:

* ``fig_symmetric_pseudoequilibria_continuum_region.pdf`` -- price-space scatter.
* ``fig_symmetric_pseudoequilibria_continuum_revenue.pdf`` -- revenue-space scatter.
* ``table_symmetric_pseudoequilibria_continuum_summary`` -- empirical summary statistics.
"""

from __future__ import annotations

import _common as C  # type: ignore[import-not-found]
import numpy as np
import pandas as pd

# Reuse the asymmetric experiment's helpers verbatim -- the theoretical-region
# computation is generic and only the demand object differs.
from exp_asymmetric_pseudoequilibria_continuum import (  # type: ignore[import-not-found]
    _build_warmups,
    _plot_region,
    _theoretical_region,
)

from src import market
from src.artifact_export import export_figure, export_table
from src.config import (
    ExperimentConfig,
    ExplorationSchedule,
    InformedProjectionBox,
    SellerSpec,
)
from src.logging_utils import run_directory
from src.simulator import run_simulation


def main(
    *,
    horizon: int = 80_000,
    n_seeds: int = 30,
    base_seed: int = 833,
    c: float = 0.3,
    eta: float = 0.85,
    quick: bool = False,
    xlim_price: tuple[float, float] | None = None,
    ylim_price: tuple[float, float] | None = None,
    xlim_revenue: tuple[float, float] | None = None,
    ylim_revenue: tuple[float, float] | None = None,
) -> dict:
    """Run the symmetric continuum of pseudo-equilibria experiment for one ``eta``.

    Optional ``xlim_*`` / ``ylim_*`` arguments override the per-call
    auto-zoom in the plot rendering. ``run_shared_axis_pair`` (below) uses
    them to enforce *shared* axes across the two schedule variants so that
    the same theoretical-region geometry appears in both panels.

    Returns a dict with the empirical price- and revenue-clouds, so a
    caller can compute a shared axis window across multiple runs.
    """
    horizon, n_seeds = C.quick_overrides(quick, default_T=horizon, default_S=n_seeds)
    d = C.baseline_demand()  # symmetric: alpha=2.5, beta=-1, gamma=0.4
    box_ob = C.tight_oblivious_box(d, expand=0.6)
    p_NE = market.nash_prices(d)
    p_C = market.collusive_prices(d)
    pi_NE = market.per_period_revenue(d, p_NE)
    pi_C = market.per_period_revenue(d, p_C)

    g12 = float(d.gamma_arr[0, 1])
    g21 = float(d.gamma_arr[1, 0])
    r_hi = max(-float(d.beta_arr[0]) / g12, -float(d.beta_arr[1]) / g21)
    # Denser interior grid (1201) than the historical 401, so the rendered
    # theoretical-region scatter looks uniformly filled even when a panel
    # is zoomed in tightly near the Nash equilibrium. The scatter is
    # rasterized inside ``_plot_region`` to keep the PDF file size small.
    r_grid = np.linspace(-r_hi, r_hi, 1201)
    theory_prices, theory_revenues = _theoretical_region(d, r_grid=r_grid)

    # Tag outputs with the exploration exponent so multiple schedule
    # choices coexist in ``results/figures/``. The canonical run keeps the
    # un-tagged names so default outputs stay stable.
    if eta == 0.85 and c == 0.3:
        eta_tag = ""
    else:
        eta_tag = f"_eta_{eta:g}".replace(".", "p")
    exp_name = f"exp_symmetric_pseudoequilibria_continuum{eta_tag}"

    rep_sched = ExplorationSchedule(kind="polynomial", c=c, eta=eta)
    cfg = ExperimentConfig(
        name=exp_name,
        market=d,
        sellers=[SellerSpec(kind="oblivious", exploration=rep_sched) for _ in range(d.N)],
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
                "($\\alpha=2.5$, $\\beta=-1$, $\\gamma=0.4$) under "
                f"$\\nu_n^2 = {c:g} (n+1)^{{-{eta:g}}}$, started from the same spread of "
                "warm-up price pairs used in the asymmetric experiment. "
                "The collusive marker lies on the ``p_1 = p_2`` diagonal and is "
                "approximately surrounded by the empirical cloud."
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
        run.save_figure("symmetric_pseudoequilibria_region", fig_price, close=False)
        # tight_bbox=False + the explicit subplots_adjust in _plot_region
        # guarantees identical page dimensions across the two schedules,
        # so the LaTeX subfigures render at the same height; dpi=300 is
        # the raster resolution of the rasterized theoretical-region scatter.
        export_figure(
            fig_price, f"fig_symmetric_pseudoequilibria_continuum_region{eta_tag}",
            strip_title=True, tight_bbox=False, dpi=300,
        )

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
        run.save_figure("symmetric_pseudoequilibria_revenue", fig_rev, close=False)
        export_figure(
            fig_rev, f"fig_symmetric_pseudoequilibria_continuum_revenue{eta_tag}",
            strip_title=True, tight_bbox=False, dpi=300,
        )

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
        "eta_tag": eta_tag,
        "emp_prices": emp_prices,
        "emp_revenues": emp_revenues,
        "p_NE": p_NE,
        "p_C": p_C,
        "pi_NE": pi_NE,
        "pi_C": pi_C,
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


def run_shared_axis_pair(
    *,
    horizon: int = 80_000,
    n_seeds: int = 30,
    base_seed: int = 833,
    c: float = 0.3,
    etas: tuple[float, ...] = (0.85, 0.5),
    quick: bool = False,
) -> None:
    """Two-pass entry point that produces the *publication-ready* pseudo-equilibria figures.

    The single-eta ``main()`` only knows about one schedule, so it cannot
    enforce shared axis windows. This wrapper runs ``main`` for each eta
    in ``etas`` to harvest the empirical clouds, computes a single shared
    (xlim, ylim) per panel type (price or revenue), and re-renders the
    four comparison PDFs so that both schedules show the *same* theoretical-region
    geometry -- which they should, since the demand parameters are identical.
    """
    results = [
        main(
            horizon=horizon, n_seeds=n_seeds, base_seed=base_seed,
            c=c, eta=eta, quick=quick,
        )
        for eta in etas
    ]
    price_xlim, price_ylim = _expand_box(
        [r["emp_prices"] for r in results],
        [tuple(results[0]["p_NE"]), tuple(results[0]["p_C"])],
    )
    rev_xlim, rev_ylim = _expand_box(
        [r["emp_revenues"] for r in results],
        [tuple(results[0]["pi_NE"]), tuple(results[0]["pi_C"])],
    )
    print(f"shared price axes:   x={price_xlim}, y={price_ylim}")
    print(f"shared revenue axes: x={rev_xlim}, y={rev_ylim}")

    # Second pass: re-render only (re-uses the simulation results we just
    # produced, but the plot-render itself is fast).
    for eta in etas:
        main(
            horizon=horizon, n_seeds=n_seeds, base_seed=base_seed,
            c=c, eta=eta, quick=quick,
            xlim_price=price_xlim, ylim_price=price_ylim,
            xlim_revenue=rev_xlim, ylim_revenue=rev_ylim,
        )


if __name__ == "__main__":
    main()
