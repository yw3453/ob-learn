"""Small-gain stress test in a mixed market.

Symmetric ``N = 5`` mixed market with ``|I^{ob}| = 2`` and ``|I^{in}| = 3``.
Fix ``alpha = 2.5``, ``beta = -1``; vary the cross-price coefficient
``gamma in {0.05, 0.10, 0.15, 0.20, 0.25}``; vary oblivious dithering
``nu^2 in {0.05, 0.10, 0.20}``. Informed sellers use the running-mean
forecast with the polynomial schedule ``nu_n^2 = c (n+1)^{-eta}``, fixed
``eta = 0.25`` (so ``eta_max = 0.25 < 1/2``). Horizon ``T = 5e4``,
``S = 100`` seeds per cell.

For every cell we compute the small-gain margin
``C_M - C_x [2 bar_gamma^{ob} L_phi^{ob} + bar_Delta + bar_Theta +
L_phi^{ob} bar_Psi / (bar_c_diag - bar_D)]`` (using
:func:`src.market.master_theorem_smallgain`) and tag each cell as
"small-gain holds" or "small-gain violated". The plot is a single
log-log plot of the seed-averaged price MSE for every cell, colour-coded
by whether the small-gain condition holds.

The acceptance criterion is qualitative: the price MSE bends cleanly
toward zero on log-log axes for every cell, including the ones where the
small-gain condition is violated.
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
from src.config import DemandParams, ExplorationSchedule, InformedProjectionBox
from src.logging_utils import run_directory
from src.plotting import SQUARE_FIGSIZE, report_style, square_box
from src.simulator import run_simulation

_N = 5
_N_OB = 2
_N_IN = 3
_OB_IDX = list(range(_N_OB))
_IN_IDX = list(range(_N_OB, _N_OB + _N_IN))
_ETA_INFORMED = 0.25
_C_INFORMED = 0.10  # leading constant in nu_n^2 = c (n+1)^{-eta}

_GAMMA_GRID = (0.04, 0.08, 0.12, 0.16, 0.20)
# All strictly inside the diagonal-dominance boundary gamma < 1/(N-1) = 0.25.
_NU2_GRID = (0.05, 0.10, 0.20)


def _make_market(gamma: float) -> DemandParams:
    return C.symmetric_market(_N, gamma=gamma, noise_std=0.2)


def main(
    *,
    horizon: int = 50_000,
    n_seeds: int = 100,
    base_seed: int = 41,
    quick: bool = False,
) -> None:
    horizon, n_seeds = C.quick_overrides(quick, default_T=horizon, default_S=n_seeds)

    rep_d = _make_market(_GAMMA_GRID[0])
    rep_box_ob = C.tight_oblivious_box(rep_d, expand=0.5)
    rep_box_in = InformedProjectionBox.from_demand(rep_d)
    rep_sched_ob = ExplorationSchedule(kind="constant", nu=float(np.sqrt(_NU2_GRID[0])))
    rep_sched_in = ExplorationSchedule(kind="polynomial", c=_C_INFORMED, eta=_ETA_INFORMED)

    cfg = C.base_config(
        name="exp_mixed_small_gain",
        market=rep_d,
        sellers=C.make_mixed_sellers(
            n_ob=_N_OB, n_in=_N_IN,
            oblivious_schedule=rep_sched_ob,
            informed_schedule=rep_sched_in,
            forecast_rule="mean_price",
        ),
        horizon=horizon,
        n_seeds=n_seeds,
        base_seed=base_seed,
        log_every=max(1, horizon // 1000),
        oblivious_box=rep_box_ob,
        informed_box=rep_box_in,
    )

    with run_directory("exp_mixed_small_gain", cfg) as run:
        run.logger.info(
            "N=%d, n_ob=%d, n_in=%d, gamma_grid=%s, nu2_grid=%s, eta_informed=%.2f",
            _N, _N_OB, _N_IN, list(_GAMMA_GRID), list(_NU2_GRID), _ETA_INFORMED,
        )

        rows: list[dict] = []
        # Each entry: (gamma, nu^2, n_grid, mse_price seed-mean, condition_holds)
        curves: list[tuple[float, float, np.ndarray, np.ndarray, bool]] = []

        for gamma in _GAMMA_GRID:
            d = _make_market(gamma)
            box_ob = C.tight_oblivious_box(d, expand=0.5)
            box_in = InformedProjectionBox.from_demand(d)
            p_NE = market.nash_prices(d)
            run.logger.info(
                "gamma=%.3f, p_NE=%s", gamma, np.round(p_NE, 3).tolist()
            )
            # Use half the minimum true |beta_i| as the |beta|_min in the
            # L_phi^{in,theta} bound; the simulator's projection box is
            # generous, so reading |beta|_min off it would inflate the
            # bound to an unrealistic value.
            beta_abs_min = 0.5 * float(np.min(np.abs(d.beta_arr)))
            for nu2 in _NU2_GRID:
                sched_ob = ExplorationSchedule(kind="constant", nu=float(np.sqrt(nu2)))
                sched_in = ExplorationSchedule(
                    kind="polynomial", c=_C_INFORMED, eta=_ETA_INFORMED
                )
                smallgain = market.master_theorem_smallgain(
                    d, _OB_IDX, _IN_IDX, box_ob, box_in,
                    nu_squared=nu2, beta_abs_min=beta_abs_min,
                )
                sub_cfg = C.base_config(
                    name=f"M1_gamma{gamma:.2f}_nu2{nu2:.2f}",
                    market=d,
                    sellers=C.make_mixed_sellers(
                        n_ob=_N_OB, n_in=_N_IN,
                        oblivious_schedule=sched_ob,
                        informed_schedule=sched_in,
                        forecast_rule="mean_price",
                    ),
                    horizon=horizon,
                    n_seeds=n_seeds,
                    base_seed=base_seed,
                    log_every=cfg.log_every,
                    oblivious_box=box_ob,
                    informed_box=box_in,
                )
                run.logger.info(
                    "running gamma=%.3f nu^2=%.3f: margin=%+.4f, condition_holds=%s",
                    gamma, nu2, smallgain["margin"], smallgain["condition_holds"],
                )
                res = run_simulation(sub_cfg, logger=run.logger)
                mse_p = analysis.mse_price(res)  # (T_log, S)
                mse_curve = mse_p.mean(axis=1)
                n_grid = res.log_steps + 1
                curves.append((
                    float(gamma), float(nu2),
                    n_grid.astype(np.float64), mse_curve,
                ))
                # smallgain already contains "nu_squared"; merge cell-level
                # metadata via union to avoid duplicate keys.
                row = {
                    "gamma": float(gamma),
                    "mse_price_final": float(mse_curve[-1]),
                    **smallgain,
                }
                rows.append(row)
                run.log_event("M1_cell", **row)

        df = pd.DataFrame(rows)
        run.save_summary("M1_smallgain_cells", df)
        export_table(
            df, "table_mixed_small_gain_smallgain",
            caption=(
                "Small-gain stress test in a "
                "symmetric $N=5$ mixed market ($|\\mathcal I^{ob}|=2$, "
                "$|\\mathcal I^{in}|=3$). For each $(\\gamma,\\,\\nu^2)$ cell "
                "we report the small-gain margin "
                "$C_M - C_x[2\\bar\\gamma^{ob}L_\\phi^{ob} + \\bar\\Delta + "
                "\\bar\\Theta + L_\\phi^{ob}\\bar\\Psi/(\\bar c_{\\mathrm{diag}}"
                " - \\bar D)]$ and the seed-averaged final price MSE."
            ),
            floatfmt=".3g",
        )

        # ---- MSE-trajectory plot: color by gamma, linestyle by nu^2 ----
        if curves:
            with report_style():
                fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
                cmap = _cmaps.get_cmap("viridis")
                color_for_gamma = {
                    float(g): cmap(0.10 + 0.80 * idx / max(len(_GAMMA_GRID) - 1, 1))
                    for idx, g in enumerate(_GAMMA_GRID)
                }
                ls_for_nu = {
                    float(nu2): ls
                    for nu2, ls in zip(_NU2_GRID, ("-", "--", ":"), strict=False)
                }
                for gamma_v, nu2_v, n_axis, curve in curves:
                    ax.plot(
                        n_axis, np.maximum(curve, 1e-12),
                        color=color_for_gamma[float(gamma_v)],
                        linestyle=ls_for_nu.get(float(nu2_v), "-"),
                        lw=1.3, alpha=0.90,
                    )
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel(r"$n$")
                ax.set_ylabel(r"MSE($\tilde p_n$)")
                handles_g = [
                    Line2D([0], [0],
                           color=color_for_gamma[float(g)], lw=1.6,
                           label=fr"$\gamma = {g:.2f}$")
                    for g in _GAMMA_GRID
                ]
                handles_nu = [
                    Line2D([0], [0], color="black", lw=1.4,
                           linestyle=ls_for_nu[float(nu2)],
                           label=fr"$\nu^2 = {nu2:.2f}$")
                    for nu2 in _NU2_GRID
                ]
                ax.legend(
                    handles=handles_g + handles_nu, loc="lower left",
                    fontsize=10, framealpha=0.92, ncol=2,
                )
                square_box(ax)
                fig.tight_layout()
            run.save_figure("M1_mse_paths", fig, close=False)
            export_figure(fig, "fig_mixed_small_gain_smallgain_mse_paths", strip_title=True)

        run.logger.info("exp_mixed_small_gain finished")


if __name__ == "__main__":
    main()
