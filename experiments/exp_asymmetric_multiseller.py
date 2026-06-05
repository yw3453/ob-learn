"""Convergence in *asymmetric* markets with many sellers.

This experiment stress-tests convergence in markets that depart from the
baseline duopoly along two axes simultaneously: number of sellers (``N \\in
\\{3, 5, 10\\}``) and heterogeneity (each seller has a different
``\\alpha_i``, ``\\beta_i``, and ``\\gamma_{i,j}``). For every cell in the
``(N, \\nu^2)`` grid we record the per-step seed-averaged price-MSE and
plot every trajectory on a single log-log axes. This is the
global-convergence diagnostic: every curve has to bend to zero with ``n``
for convergence to hold across the grid.
"""

from __future__ import annotations

import _common as C  # type: ignore[import-not-found]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps as _cmaps
from matplotlib.lines import Line2D

from src import analysis, market
from src.artifact_export import export_figure, export_table
from src.config import ExplorationSchedule
from src.logging_utils import run_directory
from src.plotting import SQUARE_FIGSIZE, report_style, square_box
from src.simulator import run_simulation


def main(
    *,
    horizon: int = 50_000,
    n_seeds: int = 60,
    base_seed: int = 13,
    quick: bool = False,
) -> None:
    horizon, n_seeds = C.quick_overrides(quick, default_T=horizon, default_S=n_seeds)
    Ns = [3, 5, 10]
    nu2_grid = np.array([0.05, 0.10, 0.20])

    rep_market = C.asymmetric_market(Ns[0], base_seed=base_seed)
    rep_box = C.tight_oblivious_box(rep_market, expand=0.5)
    rep_sched = ExplorationSchedule(kind="constant", nu=0.3)
    cfg = C.base_config(
        name="exp_asymmetric_multiseller",
        market=rep_market,
        sellers=C.make_oblivious_sellers(rep_market.N, rep_sched),
        horizon=horizon,
        n_seeds=n_seeds,
        base_seed=base_seed,
        log_every=max(1, horizon // 1000),
        oblivious_box=rep_box,
    )

    with run_directory("exp_asymmetric_multiseller", cfg) as run:
        run.logger.info("sweeping asymmetric N=%s, nu^2=%s", Ns, nu2_grid.tolist())
        rows: list[dict] = []
        # Each entry: (N, nu^2, n_grid, mse_p mean over seeds)
        curves: list[tuple[int, float, np.ndarray, np.ndarray]] = []

        for N in Ns:
            try:
                d = C.asymmetric_market(N, base_seed=base_seed)
            except ValueError as exc:
                run.logger.warning("skipping N=%d: %s", N, exc)
                continue
            box = C.tight_oblivious_box(d, expand=0.5)
            p_NE = market.nash_prices(d)
            p_C = market.collusive_prices(d)
            run.logger.info(
                "N=%d: alpha=%s beta=%s p_NE=%s p_C=%s",
                N, np.round(d.alpha_arr, 3).tolist(),
                np.round(d.beta_arr, 3).tolist(),
                np.round(p_NE, 3).tolist(),
                np.round(p_C, 3).tolist(),
            )
            for nu2 in nu2_grid:
                sched = ExplorationSchedule(kind="constant", nu=float(np.sqrt(nu2)))
                sub_cfg = C.base_config(
                    name=f"asym_N{N}_nu2_{nu2:.3f}",
                    market=d,
                    sellers=C.make_oblivious_sellers(N, sched),
                    horizon=horizon,
                    n_seeds=n_seeds,
                    base_seed=base_seed,
                    log_every=cfg.log_every,
                    oblivious_box=box,
                )
                run.logger.info("N=%d asymmetric nu^2=%.4f", N, nu2)
                res = run_simulation(sub_cfg, logger=run.logger)
                mse_p = analysis.mse_price(res)            # (T_log, S)
                mse_p_curve = mse_p.mean(axis=1)           # (T_log,)
                n_grid = res.log_steps + 1
                curves.append((N, float(nu2), n_grid.astype(np.float64), mse_p_curve))
                rows.append(dict(
                    N=N,
                    nu_squared=float(nu2),
                    mse_price_final=float(mse_p_curve[-1]),
                ))
                run.log_event("asym_N_run", **rows[-1])

        df = pd.DataFrame(rows)
        run.save_summary("asymmetric_more_sellers_summary", df)
        export_table(
            df, "table_asymmetric_multiseller_asymmetric_more_sellers",
            caption=(
                "Final price-MSE in *asymmetric* $N$-seller all-oblivious "
                "markets (cross-price coefficients scaled $\\propto 1/(N-1)$ "
                "to preserve diagonal dominance). The companion plot "
                "\\texttt{fig\\_3h\\_asymmetric\\_mse\\_paths.pdf} shows the "
                "full $\\mathrm{MSE}(\\tilde p_n)$ trajectory for every cell."
            ),
            floatfmt=".3g",
        )

        # ---- MSE-trajectory plot: each cell as one curve, colour by N. ----
        if curves:
            with report_style():
                fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
                cmap = _cmaps.get_cmap("viridis")
                n_lines = max(len(Ns), 1)
                color_for_N = {N_: cmap(0.1 + 0.8 * idx / max(n_lines - 1, 1))
                               for idx, N_ in enumerate(Ns)}
                ls_for_nu = {float(nu2): ls for nu2, ls in
                             zip(nu2_grid.tolist(), ("-", "--", ":"), strict=False)}
                for N_, nu2, n_axis, curve in curves:
                    ax.plot(
                        n_axis, np.maximum(curve, 1e-12),
                        color=color_for_N[N_],
                        linestyle=ls_for_nu.get(float(nu2), "-"),
                        lw=1.4, alpha=0.90,
                    )
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("n")
                ax.set_ylabel(r"MSE($\tilde p_n$)")
                handles_N = [
                    Line2D([0], [0], color=color_for_N[N_], lw=1.6, label=f"N = {N_}")
                    for N_ in Ns if N_ in color_for_N
                ]
                handles_nu = [
                    Line2D([0], [0], color="black", lw=1.4,
                           linestyle=ls_for_nu[float(nu2)],
                           label=fr"$\nu^2 = {nu2:.2f}$")
                    for nu2 in nu2_grid.tolist()
                ]
                ax.legend(handles=handles_N + handles_nu, loc="lower left",
                          fontsize=10, framealpha=0.92, ncol=2)
                square_box(ax)
                fig.tight_layout()
            run.save_figure("asymmetric_mse_paths", fig, close=False)
            export_figure(fig, "fig_asymmetric_multiseller_asymmetric_mse_paths", strip_title=True)

        run.logger.info("exp_asymmetric_multiseller finished")


if __name__ == "__main__":
    main()
