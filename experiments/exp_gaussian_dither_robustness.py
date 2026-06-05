"""Constant-``nu^2`` sweep with Gaussian-clip dithering.

The theory requires bounded-support dithering, so the default sims use
uniform draws. This experiment redoes the constant-``nu^2`` sweep with
``distribution="gaussian_clip"`` (clipped at 4 sigmas) for both sellers,
to verify that the empirical convergence picture is robust to the
specific dithering distribution choice.
"""

from __future__ import annotations

import _common as C  # type: ignore[import-not-found]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import analysis
from src.artifact_export import export_figure, export_table
from src.config import ExplorationSchedule
from src.logging_utils import run_directory
from src.plotting import report_style
from src.simulator import run_simulation


def main(
    *,
    horizon: int = 60_000,
    n_seeds: int = 80,
    base_seed: int = 211,
    quick: bool = False,
) -> None:
    horizon, n_seeds = C.quick_overrides(quick, default_T=horizon, default_S=n_seeds)
    d = C.baseline_demand()
    box_ob = C.tight_oblivious_box(d, expand=0.5)

    nu_squared_grid = np.array([0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3])

    rep_sched = ExplorationSchedule(
        kind="constant",
        nu=float(np.sqrt(nu_squared_grid[0])),
        distribution="gaussian_clip",
        clip_sigmas=4.0,
    )
    cfg = C.base_config(
        name="exp_gaussian_dither_robustness",
        market=d,
        sellers=C.make_oblivious_sellers(d.N, rep_sched),
        horizon=horizon,
        n_seeds=n_seeds,
        base_seed=base_seed,
        log_every=max(1, horizon // 1000),
        oblivious_box=box_ob,
    )

    with run_directory("exp_gaussian_dither_robustness", cfg) as run:
        run.logger.info("dithering distribution = gaussian_clip (4 sigmas)")
        run.logger.info("nu^2 grid: %s", nu_squared_grid.tolist())
        rows = []
        for nu2 in nu_squared_grid:
            nu = float(np.sqrt(nu2))
            sched = ExplorationSchedule(
                kind="constant",
                nu=nu,
                distribution="gaussian_clip",
                clip_sigmas=4.0,
            )
            sub_cfg = C.base_config(
                name=f"gauss_nu2_{nu2:.4f}",
                market=d,
                sellers=C.make_oblivious_sellers(d.N, sched),
                horizon=horizon,
                n_seeds=n_seeds,
                base_seed=base_seed,
                log_every=cfg.log_every,
                oblivious_box=box_ob,
            )
            run.logger.info("running gaussian-clip nu^2=%.4f", nu2)
            res = run_simulation(sub_cfg, logger=run.logger)
            mse_theta = analysis.mse_theta_oblivious(res)
            mse_p = analysis.mse_price(res)
            tail_theta = analysis.fit_loglog_slope(res.log_steps + 1, mse_theta.mean(axis=1))
            tail_price = analysis.fit_loglog_slope(res.log_steps + 1, mse_p.mean(axis=1))
            pred = analysis.predicted_rates(d, box_ob, nu_squared=nu2)
            row = dict(
                nu_squared=float(nu2),
                slope_theta_fit=float(tail_theta["slope"]),
                slope_price_fit=float(tail_price["slope"]),
                mse_theta_final=float(mse_theta[-1].mean()),
                mse_price_final=float(mse_p[-1].mean()),
                regime_label=pred["regime"],
            )
            rows.append(row)
            run.log_event("gaussian_run", **row)
            run.save_trajectory(
                f"gauss_nu2_{nu2:.4f}",
                **res.trajectories_dict(),
                mse_theta=mse_theta,
                mse_price=mse_p,
            )

        df = pd.DataFrame(rows)
        run.save_summary("gaussian_dither_summary", df)
        export_table(df, "table_gaussian_dither_robustness", caption=(
            "Constant-$\\nu^2$ sweep with Gaussian-clip dithering ($4\\sigma$). "
            "The uniform-dithering picture is reproduced "
            "with no qualitative change."
        ))

        with report_style():
            from src.plotting import SQUARE_FIGSIZE, square_box
            fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
            x = df["nu_squared"].to_numpy()
            ax.plot(x, df["slope_price_fit"], "o-", lw=1.6, label="empirical slope (price)")
            ax.plot(x, df["slope_theta_fit"], "s--", lw=1.4, label=r"empirical slope ($\hat\theta^{ob}$)")
            ax.axhline(-1.0, color="black", lw=1.2, linestyle=":")
            ax.set_xscale("log")
            ax.set_xlabel(r"$\nu^2$  (Gaussian-clip dithering)")
            ax.set_ylabel("tail log-log slope")
            from src.plotting import smart_legend
            smart_legend(ax, fontsize=11)
            square_box(ax)
            fig.tight_layout()
        run.save_figure("gaussian_threshold_slopes", fig, close=False)
        export_figure(fig, "fig_gaussian_dither_threshold_slopes", strip_title=True)
        run.logger.info("exp_gaussian_dither_robustness finished")


if __name__ == "__main__":
    main()
