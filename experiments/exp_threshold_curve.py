"""Empirical-vs-formal threshold curve over a dense ``nu^2`` sweep.

The convergence theory predicts MSE convergence of order ``n^{-(1 - rho)}``
with ``rho = gamma_bar L_phi^{ob} C_x / C_M``. For the baseline symmetric duopoly
the formal ``C_M`` upper bound is ~1, so the predicted "fast" regime is
unreachable. Here we trace the *empirical*
regime label (``slope_price < -0.5`` say) over a much denser ``nu^2`` grid.

The output is two figures: (i) tail MSE-of-price slope vs ``nu^2`` overlaid
with the formal threshold; (ii) MSE-of-theta tail level vs ``nu^2``. We also
write a CSV summary table giving slopes and final MSEs at each grid point.
"""

from __future__ import annotations

import _common as C  # type: ignore[import-not-found]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import analysis, market
from src.artifact_export import export_figure, export_table
from src.config import ExplorationSchedule
from src.logging_utils import run_directory
from src.plotting import SQUARE_FIGSIZE, report_style, smart_legend, square_box
from src.simulator import run_simulation


def main(
    *,
    horizon: int = 60_000,
    n_seeds: int = 80,
    base_seed: int = 101,
    quick: bool = False,
) -> None:
    horizon, n_seeds = C.quick_overrides(quick, default_T=horizon, default_S=n_seeds)
    d = C.baseline_demand()
    box_ob = C.tight_oblivious_box(d, expand=0.5)
    info = market.predicted_regime(d, box_ob, nu_squared=1.0)
    threshold = float(info["threshold"])  # gamma_bar * L_phi^{ob} * C_x

    # Dense log-spaced grid: 0.001 ... 0.3, 16 points.
    nu_squared_grid = np.geomspace(1e-3, 0.3, 16)

    rep_sched = ExplorationSchedule(kind="constant", nu=float(np.sqrt(nu_squared_grid[0])))
    cfg = C.base_config(
        name="exp_threshold_curve",
        market=d,
        sellers=C.make_oblivious_sellers(d.N, rep_sched),
        horizon=horizon,
        n_seeds=n_seeds,
        base_seed=base_seed,
        log_every=max(1, horizon // 1000),
        oblivious_box=box_ob,
    )

    with run_directory("exp_threshold_curve", cfg) as run:
        run.logger.info("formal threshold = gamma_bar * L_phi_ob * C_x = %.4f", threshold)
        run.logger.info("nu^2 grid: %s", nu_squared_grid.tolist())
        rows = []
        for nu2 in nu_squared_grid:
            nu = float(np.sqrt(nu2))
            sched = ExplorationSchedule(kind="constant", nu=nu)
            sub_cfg = C.base_config(
                name=f"const_nu2_{nu2:.4f}",
                market=d,
                sellers=C.make_oblivious_sellers(d.N, sched),
                horizon=horizon,
                n_seeds=n_seeds,
                base_seed=base_seed,
                log_every=cfg.log_every,
                oblivious_box=box_ob,
            )
            run.logger.info("running constant nu^2=%.5f (nu=%.4f)", nu2, nu)
            res = run_simulation(sub_cfg, logger=run.logger)
            mse_theta = analysis.mse_theta_oblivious(res)
            mse_p = analysis.mse_price(res)
            n_axis = res.log_steps + 1.0
            tail_theta = analysis.fit_loglog_slope(n_axis, mse_theta.mean(axis=1))
            tail_price = analysis.fit_loglog_slope(n_axis, mse_p.mean(axis=1))
            pred = analysis.predicted_rates(d, box_ob, nu_squared=nu2)
            cm_upper = float(market.cm_upper_bound(d, float(nu2)))
            row = dict(
                nu_squared=float(nu2),
                regime_label=pred["regime"],
                slope_predicted=float(pred["slope"]),
                slope_theta_fit=float(tail_theta["slope"]),
                slope_price_fit=float(tail_price["slope"]),
                mse_theta_final=float(mse_theta[-1].mean()),
                mse_price_final=float(mse_p[-1].mean()),
                rho=float(pred["rho"]),
                C_M_upper_bound=cm_upper,
                # Conservative empirical lower bound on the allowable
                # gamma_bar * L_phi^{ob} * C_x consistent with the observed
                # fast-rate behaviour (slope of price-MSE <= -0.85, say).
                allowed_threshold_lb=(
                    cm_upper if float(tail_price["slope"]) < -0.5 else float("nan")
                ),
            )
            rows.append(row)
            run.log_event("threshold_run", **row)
            run.save_trajectory(
                f"const_nu2_{nu2:.4f}",
                **res.trajectories_dict(),
                mse_theta=mse_theta,
                mse_price=mse_p,
            )

        df = pd.DataFrame(rows)
        run.save_summary("threshold_curve_summary", df)
        export_table(df, "table_threshold_curve_threshold_curve", caption=(
            "Tail MSE slopes for prices and $\\theta^{ob}$ along a 16-point "
            "log-spaced $\\nu^2$ grid in the symmetric baseline duopoly. The "
            "closed-form upper bound on the persistent-excitation constant "
            "$C_M(\\nu^2)$ doubles as a lower bound on the *empirical* "
            "allowable threshold $\\bar\\gamma L_\\phi^{ob} C_x$ whenever the "
            "price MSE slope is below $-0.5$ (column "
            "``allowed\\_threshold\\_lb'')."
        ))

        # Build the empirical-vs-formal plot: slope of price MSE vs nu^2.
        with report_style():
            fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
            x = df["nu_squared"].to_numpy()
            ax.plot(x, df["slope_price_fit"], "o-", lw=1.6, label="empirical (price)")
            ax.plot(x, df["slope_theta_fit"], "s--", lw=1.4, label=r"empirical ($\hat\theta^{ob}$)")
            ax.axhline(-1.0, color="black", lw=1.2, linestyle=":", label=r"slope $-1$")
            ax.axhline(0.0, color="tab:gray", lw=0.8)
            ax.set_xscale("log")
            ax.set_xlabel(r"$\nu^2$")
            ax.set_ylabel("tail log-log slope")
            smart_legend(ax, fontsize=11)
            square_box(ax)
            fig.tight_layout()
        run.save_figure("threshold_curve_slopes", fig, close=False)
        export_figure(fig, "fig_threshold_curve_threshold_curve_slopes", strip_title=True)

        with report_style():
            fig2, ax2 = plt.subplots(figsize=SQUARE_FIGSIZE)
            ax2.plot(x, df["mse_price_final"], "o-", lw=1.6, label="MSE(price)")
            ax2.plot(x, df["mse_theta_final"], "s--", lw=1.4, label=r"MSE($\hat\theta^{ob}$)")
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.set_xlabel(r"$\nu^2$")
            ax2.set_ylabel("MSE at $T$")
            smart_legend(ax2, fontsize=11)
            square_box(ax2)
            fig2.tight_layout()
        run.save_figure("threshold_curve_final_mse", fig2, close=False)
        export_figure(fig2, "fig_threshold_curve_threshold_curve_final_mse", strip_title=True)
        run.logger.info("exp_threshold_curve finished")


if __name__ == "__main__":
    main()
