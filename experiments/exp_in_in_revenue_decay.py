"""In-in cumulative revenue under decaying exploration.

In the revenue duopoly
(``\\gamma = 0.6``, used by ``exp_ob_ob_revenue`` / ``exp_ob_in_revenue``), if *both* sellers are
informed (``mean_price`` forecast) and run polynomially decaying exploration
``\\nu_n^2 = c (n+1)^{-\\eta}``, the per-period revenue at the horizon should
converge to ``\\Pi^{NE}`` from below as the exploration tax vanishes.

We sweep three values ``\\eta \\in \\{0.3, 0.5, 0.7\\}`` (with ``c=0.30``
common to all). The headline cell ``\\eta = 0.5`` puts ``\\nu_n^2`` on the
order of ``n^{-0.5}``; the bracketing cells stress test slower / faster
decay.

Outputs (saved under ``runs/exp_in_in_revenue_decay`` and exported to
``results/figures/``):

  * ``table_in_in_decay_revenue.{md,csv}`` -- mean and 5/95 % range of
    average-revenue-per-period at ``T`` for each ``\\eta``.
  * ``fig_in_in_decay_cumulative_revenue.pdf`` -- cumulative revenue curves
    against the ``T \\Pi^{NE}`` reference for the headline ``\\eta = 0.5``
    schedule.
"""

from __future__ import annotations

import _common as C  # type: ignore[import-not-found]
import pandas as pd

from src import benchmarks
from src.artifact_export import export_figure, export_table
from src.config import ExplorationSchedule, SellerSpec
from src.logging_utils import run_directory
from src.plotting import plot_cumulative_revenue
from src.simulator import run_simulation


def make_all_informed(
    schedule: ExplorationSchedule,
    *,
    forecast_rule: str = "mean_price",
    n: int = 2,
) -> list[SellerSpec]:
    return [
        SellerSpec(kind="informed", forecast_rule=forecast_rule, exploration=schedule)  # type: ignore[arg-type]
        for _ in range(n)
    ]


def main(
    *,
    horizon: int = 60_000,
    n_seeds: int = 200,
    base_seed: int = 47,
    quick: bool = False,
) -> None:
    horizon, n_seeds = C.quick_overrides(quick, default_T=horizon, default_S=n_seeds)
    d = C.revenue_duopoly()  # same market as exp_ob_ob_revenue / exp_ob_in_revenue.

    pi_ref = benchmarks.benchmark_per_period_revenues(d)

    # Three decaying schedules; the middle one is the headline n^{-0.5} cell.
    eta_grid = (0.3, 0.5, 0.7)
    c_common = 0.30

    rep_sched = ExplorationSchedule(kind="polynomial", c=c_common, eta=eta_grid[1])
    cfg = C.base_config(
        name="exp_in_in_revenue_decay",
        market=d,
        sellers=make_all_informed(rep_sched),
        horizon=horizon,
        n_seeds=n_seeds,
        base_seed=base_seed,
        log_every=max(1, horizon // 1000),
    )

    with run_directory("exp_in_in_revenue_decay", cfg) as run:
        run.logger.info(
            "Pi_NE=%s, Pi_C=%s",
            pi_ref["NE"].tolist(),
            pi_ref["collusive"].tolist(),
        )
        rows: list[dict] = []
        cum_fig_export = None
        for eta in eta_grid:
            label = f"inin_mean_price_eta_{eta:.2f}".replace(".", "p")
            sched = ExplorationSchedule(kind="polynomial", c=c_common, eta=float(eta))
            sub_cfg = C.base_config(
                name=label,
                market=d,
                sellers=make_all_informed(sched),
                horizon=horizon,
                n_seeds=n_seeds,
                base_seed=base_seed,
                log_every=cfg.log_every,
            )
            run.logger.info("running in-in revenue eta=%.2f c=%.2f", eta, c_common)
            res = run_simulation(sub_cfg, logger=run.logger)
            cum = benchmarks.cumulative_revenue(res)
            avg = benchmarks.average_revenue(res)
            stats = benchmarks.revenue_summary_statistics(res)
            run.log_event(
                "inin_revenue",
                eta=float(eta),
                c=c_common,
                avg_revenue_mean=stats["mean"].tolist(),
                avg_revenue_p05=stats["p05"].tolist(),
                avg_revenue_p95=stats["p95"].tolist(),
            )
            run.save_trajectory(
                label,
                **res.trajectories_dict(),
                cumulative_revenue=cum,
                average_revenue=avg,
            )
            rows.append(
                dict(
                    schedule=f"mean_price (eta={eta:.2f}, c={c_common})",
                    eta=float(eta),
                    c=c_common,
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
            cum_fig = plot_cumulative_revenue(
                res,
                title=f"All-informed, mean_price, eta={eta:.2f}",
            )
            run.save_figure(f"cumulative_revenue_{label}", cum_fig, close=False)
            if abs(eta - 0.5) < 1e-9:
                cum_fig_export = cum_fig
            else:
                import matplotlib.pyplot as _plt

                _plt.close(cum_fig)

        df = pd.DataFrame(rows)
        run.save_summary("inin_revenue_decaying_summary", df)
        export_table(df, "table_in_in_decay_revenue", caption=(
            "All-informed (mean\\_price) revenue under three decaying "
            "exploration schedules ``\\nu_n^2 = c (n+1)^{-\\eta}`` with "
            "$c = 0.30$ in the revenue duopoly ($\\alpha=2.5$, "
            "$\\beta=-1$, $\\gamma=0.6$, $[l,u]=[0.5, 3.5]$). Reference "
            "benchmarks: $\\Pi^{NE}_0 = 3.189$ and $\\Pi^{C}_0 = 3.906$. "
            "$T = 60{,}000$, $S = 200$."
        ))
        if cum_fig_export is not None:
            export_figure(
                cum_fig_export,
                "fig_in_in_decay_cumulative_revenue",
                strip_title=True,
            )
            import matplotlib.pyplot as _plt

            _plt.close(cum_fig_export)
        run.logger.info("exp_in_in_revenue_decay finished")


if __name__ == "__main__":
    main()
