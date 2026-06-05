"""Multi-seller asymmetric mixed markets.

Asymmetric mixed markets at ``N \\in \\{3, 5, 10\\}`` with one
oblivious seller and the rest informed. Demand parameters are drawn
once per ``N`` from the asymmetric sampler used in
:func:`src._common.asymmetric_market`:

* ``alpha_i ~ N(2.5, 0.4^2)`` clipped to ``[1.5, 3.5]``;
* ``beta_i ~ N(-1, 0.2^2)`` clipped to ``[-1.5, -0.7]``;
* ``gamma_{i,j} = (0.4 / (N - 1)) * U[0.7, 1.3]``.

Informed sellers use the running-mean forecast with ``eta(j) = 0.25``
(so ``eta_max < 1/2``). The oblivious dithering variance is swept over
``nu^2 \\in \\{0.05, 0.10, 0.20\\}``. Horizon ``T = 5e4``, ``S = 100``
seeds per cell. The plot is a single log-log plot of seed-averaged price
MSE for every ``(N, nu^2)`` cell.

The acceptance criterion is qualitative: every cell's trajectory bends
to zero on log-log axes, confirming that the convergence prediction is
robust to asymmetry, to ``N``, and to ``nu^2``.
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
from src.config import ExplorationSchedule, InformedProjectionBox
from src.logging_utils import run_directory
from src.plotting import SQUARE_FIGSIZE, report_style, square_box
from src.simulator import run_simulation

_NS = (3, 5, 10)
_NU2_GRID = (0.05, 0.10, 0.20)
_ETA_INFORMED = 0.25
_C_INFORMED = 0.10  # nu_n^2 = c (n+1)^{-eta} leading constant


def main(
    *,
    horizon: int = 50_000,
    n_seeds: int = 100,
    base_seed: int = 53,
    quick: bool = False,
) -> None:
    horizon, n_seeds = C.quick_overrides(quick, default_T=horizon, default_S=n_seeds)

    rep_d = C.asymmetric_market(_NS[0], base_seed=base_seed)
    rep_box_ob = C.tight_oblivious_box(rep_d, expand=0.5)
    rep_box_in = InformedProjectionBox.from_demand(rep_d)
    rep_sched_ob = ExplorationSchedule(kind="constant", nu=float(np.sqrt(_NU2_GRID[0])))
    rep_sched_in = ExplorationSchedule(kind="polynomial", c=_C_INFORMED, eta=_ETA_INFORMED)

    cfg = C.base_config(
        name="exp_mixed_multiseller",
        market=rep_d,
        sellers=C.make_mixed_sellers(
            n_ob=1, n_in=_NS[0] - 1,
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

    with run_directory("exp_mixed_multiseller", cfg) as run:
        run.logger.info(
            "asymmetric mixed sweep: N in %s, nu^2 in %s, eta_informed=%.2f",
            list(_NS), list(_NU2_GRID), _ETA_INFORMED,
        )

        rows: list[dict] = []
        # Each entry: (N, nu^2, n_grid, mse_p seed-mean)
        curves: list[tuple[int, float, np.ndarray, np.ndarray]] = []

        for N in _NS:
            try:
                d = C.asymmetric_market(N, base_seed=base_seed)
            except ValueError as exc:
                run.logger.warning("skipping N=%d: %s", N, exc)
                continue
            n_ob, n_in = 1, N - 1
            ob_idx = list(range(n_ob))
            in_idx = list(range(n_ob, N))
            box_ob = C.tight_oblivious_box(d, expand=0.5)
            box_in = InformedProjectionBox.from_demand(d)
            p_NE = market.nash_prices(d)
            p_C = market.collusive_prices(d)
            run.logger.info(
                "N=%d: alpha=%s beta=%s p_NE=%s p_C=%s",
                N,
                np.round(d.alpha_arr, 3).tolist(),
                np.round(d.beta_arr, 3).tolist(),
                np.round(p_NE, 3).tolist(),
                np.round(p_C, 3).tolist(),
            )
            beta_abs_min = 0.5 * float(np.min(np.abs(d.beta_arr)))
            for nu2 in _NU2_GRID:
                sched_ob = ExplorationSchedule(kind="constant", nu=float(np.sqrt(nu2)))
                sched_in = ExplorationSchedule(
                    kind="polynomial", c=_C_INFORMED, eta=_ETA_INFORMED
                )
                smallgain = market.master_theorem_smallgain(
                    d, ob_idx, in_idx, box_ob, box_in,
                    nu_squared=nu2, beta_abs_min=beta_abs_min,
                )
                sub_cfg = C.base_config(
                    name=f"M2_N{N}_nu2_{nu2:.3f}",
                    market=d,
                    sellers=C.make_mixed_sellers(
                        n_ob=n_ob, n_in=n_in,
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
                    "running N=%d nu^2=%.3f: margin=%+.4f, condition_holds=%s",
                    N, nu2, smallgain["margin"], smallgain["condition_holds"],
                )
                res = run_simulation(sub_cfg, logger=run.logger)
                mse_p = analysis.mse_price(res)
                mse_curve = mse_p.mean(axis=1)
                n_grid = res.log_steps + 1
                curves.append((
                    N, float(nu2), n_grid.astype(np.float64), mse_curve,
                ))
                rows.append(dict(
                    N=N,
                    nu_squared=float(nu2),
                    mse_price_final=float(mse_curve[-1]),
                    margin=float(smallgain["margin"]),
                    condition_holds=bool(smallgain["condition_holds"]),
                ))
                run.log_event("M2_cell", **rows[-1])

        df = pd.DataFrame(rows)
        run.save_summary("M2_multiseller_cells", df)
        export_table(
            df, "table_mixed_multiseller_multiseller",
            caption=(
                "Asymmetric mixed markets at $N \\in \\{3, 5, 10\\}$ "
                "with one oblivious seller and $N-1$ informed sellers. "
                "Each cell reports the final seed-averaged price MSE and the "
                "small-gain margin."
            ),
            floatfmt=".3g",
        )

        # ---- MSE-trajectory plot: each cell as one curve. ----
        if curves:
            with report_style():
                fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
                cmap = _cmaps.get_cmap("viridis")
                n_lines = max(len(_NS), 1)
                color_for_N = {
                    N_: cmap(0.10 + 0.80 * idx / max(n_lines - 1, 1))
                    for idx, N_ in enumerate(_NS)
                }
                ls_for_nu = {
                    float(nu2): ls
                    for nu2, ls in zip(_NU2_GRID, ("-", "--", ":"), strict=False)
                }
                for N_, nu2, n_axis, curve in curves:
                    ax.plot(
                        n_axis, np.maximum(curve, 1e-12),
                        color=color_for_N.get(N_, "tab:blue"),
                        linestyle=ls_for_nu.get(float(nu2), "-"),
                        lw=1.4, alpha=0.90,
                    )
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel(r"$n$")
                ax.set_ylabel(r"MSE($\tilde p_n$)")
                handles_N = [
                    Line2D([0], [0], color=color_for_N[N_], lw=1.6, label=f"N = {N_}")
                    for N_ in _NS if N_ in color_for_N
                ]
                handles_nu = [
                    Line2D([0], [0], color="black", lw=1.4,
                           linestyle=ls_for_nu[float(nu2)],
                           label=fr"$\nu^2 = {nu2:.2f}$")
                    for nu2 in _NU2_GRID
                ]
                ax.legend(
                    handles=handles_N + handles_nu, loc="lower left",
                    fontsize=10, framealpha=0.92, ncol=2,
                )
                square_box(ax)
                fig.tight_layout()
            run.save_figure("M2_mse_paths", fig, close=False)
            export_figure(fig, "fig_mixed_multiseller_multiseller_mse_paths", strip_title=True)

        run.logger.info("exp_mixed_multiseller finished")


if __name__ == "__main__":
    main()
