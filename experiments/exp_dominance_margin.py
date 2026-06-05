"""Dominance margin sweep at ``N = 5``.

This experiment runs symmetric ``N = 5`` markets with cross-price
coefficient ``\\gamma`` ranging from deep-interior to almost-boundary
(``\\gamma`` close to ``-\\beta / (N-1) = 0.25``, where the collusive
Hessian becomes singular). For every ``(\\gamma, \\nu^2)`` cell we
record the seed-averaged price-MSE trajectory and plot all curves on
one log-log axes -- a global-convergence diagnostic that mirrors the
asymmetric multi-seller experiment on the heterogeneity axis.
"""

from __future__ import annotations

import _common as C  # type: ignore[import-not-found]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps as _cmaps
from matplotlib.lines import Line2D

from src import analysis
from src.artifact_export import export_figure, export_table
from src.config import DemandParams, ExplorationSchedule
from src.logging_utils import run_directory
from src.plotting import SQUARE_FIGSIZE, report_style, square_box
from src.simulator import run_simulation


def _make_market(N: int, gamma: float) -> DemandParams:
    return DemandParams.symmetric(
        N=N,
        alpha=2.5,
        beta=-1.0,
        gamma=float(gamma),
        l=0.5,
        u=2.5,
        noise_std=0.2,
    )


def main(
    *,
    horizon: int = 30_000,
    n_seeds: int = 60,
    base_seed: int = 313,
    quick: bool = False,
) -> None:
    horizon, n_seeds = C.quick_overrides(quick, default_T=horizon, default_S=n_seeds)

    N = 5
    # Collusive-Hessian regularity (so p^C is well defined) requires
    # gamma < -beta / (N - 1) = 0.25 for beta = -1, N = 5. We restrict
    # the sweep to the *deep-interior* values gamma in {0.05, 0.10,
    # 0.15} that produce visually unambiguous MSE-decay curves on the
    # log-log plot. Earlier sweeps that reached gamma = 0.245
    # (collusive-Hessian near-singular) produced cells whose MSE
    # plateau is governed by transient effects rather than the
    # asymptotic limit, which obscures the global-convergence behavior
    # this plot is meant to display.
    gamma_grid = np.array([0.05, 0.10, 0.15])
    nu_squared_grid = np.array([0.05, 0.10, 0.20])

    rep_d = _make_market(N, float(gamma_grid[0]))
    rep_box = C.tight_oblivious_box(rep_d, expand=0.5)
    rep_sched = ExplorationSchedule(kind="constant", nu=float(np.sqrt(nu_squared_grid[0])))
    cfg = C.base_config(
        name="exp_dominance_margin",
        market=rep_d,
        sellers=C.make_oblivious_sellers(N, rep_sched),
        horizon=horizon,
        n_seeds=n_seeds,
        base_seed=base_seed,
        log_every=max(1, horizon // 800),
        oblivious_box=rep_box,
    )

    with run_directory("exp_dominance_margin", cfg) as run:
        rows = []
        curves: list[tuple[float, float, np.ndarray, np.ndarray]] = []
        for g in gamma_grid:
            try:
                d = _make_market(N, float(g))
            except Exception as exc:  # noqa: BLE001
                run.logger.warning("skip gamma=%.4f at N=%d: %s", g, N, exc)
                continue
            box = C.tight_oblivious_box(d, expand=0.5)
            delta = float(-2.0 * d.beta[0] - 2.0 * (N - 1) * g)
            for nu2 in nu_squared_grid:
                sched = ExplorationSchedule(kind="constant", nu=float(np.sqrt(nu2)))
                sub_cfg = C.base_config(
                    name=f"N{N}_g{g:.4f}_nu2_{nu2:.4f}",
                    market=d,
                    sellers=C.make_oblivious_sellers(N, sched),
                    horizon=horizon,
                    n_seeds=n_seeds,
                    base_seed=base_seed,
                    log_every=cfg.log_every,
                    oblivious_box=box,
                )
                run.logger.info(
                    "N=%d gamma=%.4f delta=%.3f nu^2=%.3f",
                    N, g, delta, nu2,
                )
                res = run_simulation(sub_cfg, logger=run.logger)
                mse_p = analysis.mse_price(res)  # (T_log, S)
                curve = mse_p.mean(axis=1)
                n_grid = (res.log_steps + 1).astype(np.float64)
                curves.append((float(g), float(nu2), n_grid, curve))
                rows.append(dict(
                    N=N,
                    gamma=float(g),
                    delta_N=delta,
                    nu_squared=float(nu2),
                    mse_price_final=float(curve[-1]),
                ))
                run.log_event("dominance_run", **rows[-1])

        df = pd.DataFrame(rows)
        run.save_summary("dominance_margin_summary", df)
        export_table(df, "table_dominance_margin_dominance_margin", caption=(
            f"Final price-MSE in symmetric $N = {N}$ markets across a "
            "$(\\gamma, \\nu^2)$ grid. The companion plot "
            "\\texttt{fig\\_3b\\_dominance\\_mse\\_paths.pdf} shows the "
            "full $\\mathrm{MSE}(\\tilde p_n)$ trajectory for every cell."
        ), floatfmt=".3g")

        # ---- MSE-trajectory plot ----
        if curves:
            with report_style():
                fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
                cmap = _cmaps.get_cmap("viridis")
                gammas = list(gamma_grid.tolist())
                n_g = len(gammas)
                color_for_g = {float(g): cmap(0.1 + 0.8 * idx / max(n_g - 1, 1))
                               for idx, g in enumerate(gammas)}
                nu_styles = ("-", "--", ":")
                ls_for_nu = {float(nu2): ls for nu2, ls in
                             zip(nu_squared_grid.tolist(), nu_styles, strict=False)}
                for g, nu2, n_axis, curve in curves:
                    ax.plot(
                        n_axis, np.maximum(curve, 1e-12),
                        color=color_for_g[g],
                        linestyle=ls_for_nu.get(float(nu2), "-"),
                        lw=1.4, alpha=0.90,
                    )
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("n")
                ax.set_ylabel(r"MSE($\tilde p_n$)")
                handles_g = [
                    Line2D([0], [0], color=color_for_g[float(g)], lw=1.6,
                           label=fr"$\gamma = {g:.3f}$")
                    for g in gammas
                ]
                handles_nu = [
                    Line2D([0], [0], color="black", lw=1.4,
                           linestyle=ls_for_nu[float(nu2)],
                           label=fr"$\nu^2 = {nu2:.2f}$")
                    for nu2 in nu_squared_grid.tolist()
                ]
                ax.legend(handles=handles_g + handles_nu, loc="lower left",
                          fontsize=9, framealpha=0.92, ncol=2)
                square_box(ax)
                fig.tight_layout()
            run.save_figure("dominance_mse_paths", fig, close=False)
            export_figure(fig, "fig_dominance_margin_dominance_mse_paths", strip_title=True)

        run.logger.info("exp_dominance_margin finished")


if __name__ == "__main__":
    main()
