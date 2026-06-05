"""Cumulative revenue in an all-oblivious duopoly.

In the symmetric all-oblivious duopoly, plot ``R_{T,i} = sum_t p_{t,i} d_{t,i}``
against ``T * Pi_{NE}`` and ``T * Pi_C``, both under the Fast regime (where
``mse_price`` decays as ``1/n``) and under a sublinear/decaying-exploration
regime where short-run excursions are visible. Feeds the ob-ob row of the
meta-revenue summary via ``experiments/build_meta_revenue_summary.py``.
"""

from __future__ import annotations

import _common as C  # type: ignore[import-not-found]
import numpy as np
import pandas as pd

from src import benchmarks
from src.artifact_export import export_figure, export_table
from src.config import ExplorationSchedule
from src.logging_utils import run_directory
from src.plotting import plot_cumulative_revenue, plot_sample_paths
from src.simulator import run_simulation


def main(
    *,
    horizon: int = 60_000,
    n_seeds: int = 200,
    base_seed: int = 23,
    quick: bool = False,
) -> None:
    horizon, n_seeds = C.quick_overrides(quick, default_T=horizon, default_S=n_seeds)
    d = C.revenue_duopoly()  # larger gamma => more visible revenue gap.
    box_ob = C.tight_oblivious_box(d, expand=0.5)

    # Three persistent-exploration schedules contrast revenue regimes.
    # All are *constant* and *persistent*; in the convergent (fast) regime
    # prices settle near ``p^{NE}`` with revenue
    # ``\approx \Pi^{NE} - |\beta_i| \nu^2``.
    #
    # ``low_const_04`` (``\nu^2 = 0.04``) is the smallest convergent cell
    # we report -- it sits just above the revenue duopoly's empirical
    # fast-regime threshold (a touch above ``\nu^2 = 0.03`` per spot
    # checks), so prices converge to ``p^{NE}`` and revenue lands at
    # the textbook ``\Pi^{NE} - |\beta| \nu^2``. Horizon is bumped to
    # ``T = 120{,}000`` for that cell so the running mean settles well
    # within the noise band.
    schedules = {
        "high_const":   ExplorationSchedule(kind="constant", nu=float(np.sqrt(0.20))),
        "low_const":    ExplorationSchedule(kind="constant", nu=float(np.sqrt(0.05))),
        "low_const_04": ExplorationSchedule(kind="constant", nu=float(np.sqrt(0.04))),
    }
    horizon_per_label = {
        "high_const":   horizon,
        "low_const":    horizon,
        "low_const_04": max(horizon, 120_000),
    }

    rep_sched = next(iter(schedules.values()))
    cfg = C.base_config(
        name="exp_ob_ob_revenue",
        market=d,
        sellers=C.make_oblivious_sellers(d.N, rep_sched),
        horizon=horizon,
        n_seeds=n_seeds,
        base_seed=base_seed,
        log_every=max(1, horizon // 1000),
        oblivious_box=box_ob,
    )

    pi_ref = benchmarks.benchmark_per_period_revenues(d)

    with run_directory("exp_ob_ob_revenue", cfg) as run:
        run.logger.info("Pi_NE = %s, Pi_C = %s", pi_ref["NE"].tolist(), pi_ref["collusive"].tolist())
        run.log_event("benchmark_revenues", **{k: v.tolist() for k, v in pi_ref.items()})

        rows = []
        for label, sched in schedules.items():
            T_label = horizon_per_label.get(label, horizon)
            sub_cfg = C.base_config(
                name=f"obob_{label}",
                market=d,
                sellers=C.make_oblivious_sellers(d.N, sched),
                horizon=T_label,
                n_seeds=n_seeds,
                base_seed=base_seed,
                log_every=max(1, T_label // 1000),
                oblivious_box=box_ob,
            )
            run.logger.info("running ob-ob revenue under %s", label)
            res = run_simulation(sub_cfg, logger=run.logger)
            cum = benchmarks.cumulative_revenue(res)
            avg = benchmarks.average_revenue(res)
            stats = benchmarks.revenue_summary_statistics(res)
            run.log_event(
                "obob_revenue",
                schedule=label,
                avg_revenue_mean=stats["mean"].tolist(),
                avg_revenue_p05=stats["p05"].tolist(),
                avg_revenue_p95=stats["p95"].tolist(),
            )
            run.save_trajectory(
                f"obob_{label}",
                **res.trajectories_dict(),
                cumulative_revenue=cum,
                average_revenue=avg,
            )
            rows.append(
                dict(
                    schedule=label,
                    avg_R_T_seller0_mean=float(stats["mean"][0]),
                    avg_R_T_seller1_mean=float(stats["mean"][1]),
                    avg_R_T_seller0_p05=float(stats["p05"][0]),
                    avg_R_T_seller0_p95=float(stats["p95"][0]),
                    avg_R_T_seller1_p05=float(stats["p05"][1]),
                    avg_R_T_seller1_p95=float(stats["p95"][1]),
                    Pi_NE_seller0=float(pi_ref["NE"][0]),
                    Pi_C_seller0=float(pi_ref["collusive"][0]),
                )
            )
            cum_fig = plot_cumulative_revenue(res, title=f"All-oblivious, {label}")
            run.save_figure(f"cumulative_revenue_{label}", cum_fig, close=False)
            sp_fig = plot_sample_paths(res, n_paths=5, title=f"All-oblivious, {label}")
            run.save_figure(f"sample_paths_{label}", sp_fig, close=False)
            if label == "high_const":
                # Export the high-exploration case: shows revenue
                # systematically below NE due to the explicit dithering cost.
                export_figure(cum_fig, "fig_ob_ob_cumulative_revenue_high_const", strip_title=True)
                export_figure(sp_fig, "fig_ob_ob_sample_paths_high_const", strip_title=True)
            else:
                import matplotlib.pyplot as _plt

                _plt.close(cum_fig)
                _plt.close(sp_fig)

        df = pd.DataFrame(rows)
        run.save_summary("obob_revenue_summary", df)
        export_table(df, "table_ob_ob_revenue", caption=(
            "All-oblivious revenue under three exploration schedules in the "
            "revenue duopoly ($\\alpha=2.5$, $\\beta=-1$, $\\gamma=0.6$, "
            "$[l,u]=[0.5, 3.5]$). Reference benchmarks: $\\Pi^{NE}_0 = 3.189$ "
            "and $\\Pi^{C}_0 = 3.906$. Horizons: $T=60{,}000$ for "
            "`high_const` and `low_const`; $T=120{,}000$ for `low_const_04`. "
            "$S=200$ in all cells."
        ))
        run.logger.info("exp_ob_ob_revenue finished")


if __name__ == "__main__":
    main()
