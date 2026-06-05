"""Cumulative revenue in a mixed (ob-in) duopoly.

For each forecast/exploration cell, computes cumulative revenues per seller;
for the symmetric perfect-prediction case verifies
``Pi_NE < Pi_1^* < Pi_2^* < Pi_C``.

All cells share the persistent oblivious exploration regime
(constant ``nu^2 = 0.10``) under the ``mean_price`` and
``perfect_prediction`` forecasts. The informed seller decays as
``nu_n^2 = 0.10 (n+1)^{-eta}``; we evaluate two decay rates
``eta in {0.5, 0.25}``, both strictly inside the admissible range
``eta in [0, 1]``. The ``lag1_autocorr`` cell (eta = 0.5) is kept as an
ablation that documents the Jensen cost of failing to de-noise.
"""

from __future__ import annotations

import _common as C  # type: ignore[import-not-found]
import numpy as np
import pandas as pd

from src import benchmarks, market
from src.artifact_export import export_figure, export_table
from src.config import ExperimentConfig, ExplorationSchedule, InformedProjectionBox
from src.logging_utils import run_directory
from src.plotting import plot_cumulative_revenue
from src.simulator import run_simulation

_THEOREM_OB = ExplorationSchedule(kind="constant", nu=float(np.sqrt(0.10)))
_THEOREM_IN_ETA05 = ExplorationSchedule(kind="polynomial", c=0.10, eta=0.5)
_THEOREM_IN_ETA025 = ExplorationSchedule(kind="polynomial", c=0.10, eta=0.25)

# (cell_name, forecast_rule, informed_schedule). The cell name is the
# stable identifier used in the saved trajectories / summary CSV.
_CELLS: tuple[tuple[str, str, ExplorationSchedule], ...] = (
    ("mean_price_eta05", "mean_price", _THEOREM_IN_ETA05),
    ("mean_price_eta025", "mean_price", _THEOREM_IN_ETA025),
    ("perfect_prediction", "perfect_prediction", _THEOREM_IN_ETA05),
    ("lag1_autocorr", "lag1_autocorr", _THEOREM_IN_ETA05),
)


def main(
    *,
    horizon: int = 60_000,
    n_seeds: int = 200,
    base_seed: int = 29,
    quick: bool = False,
) -> None:
    horizon, n_seeds = C.quick_overrides(quick, default_T=horizon, default_S=n_seeds)
    d = C.revenue_duopoly()  # bigger NE-collusive gap => more visible revenue diffs.
    box_ob = C.tight_oblivious_box(d, expand=0.5)
    p_NE = market.nash_prices(d)
    p_C = market.collusive_prices(d)
    p_S = np.array(market.stackelberg_duopoly(d))
    pi_NE = market.per_period_revenue(d, p_NE)
    pi_C = market.per_period_revenue(d, p_C)
    pi_S = market.per_period_revenue(d, p_S)

    cfg = C.base_config(
        name="exp_ob_in_revenue",
        market=d,
        sellers=C.make_mixed_duopoly(
            oblivious_schedule=_THEOREM_OB,
            informed_schedule=_THEOREM_IN_ETA05,
            forecast_rule="mean_price",
        ),
        horizon=horizon,
        n_seeds=n_seeds,
        base_seed=base_seed,
        log_every=max(1, horizon // 1000),
        oblivious_box=box_ob,
    )

    with run_directory("exp_ob_in_revenue", cfg) as run:
        run.logger.info(
            "p_NE=%s p_C=%s Stackelberg=%s | Pi_NE=%s Pi_C=%s Pi_S=%s",
            p_NE.tolist(), p_C.tolist(), p_S.tolist(),
            pi_NE.tolist(), pi_C.tolist(), pi_S.tolist(),
        )
        run.log_event(
            "stackelberg_check",
            condition_holds=bool((pi_NE < pi_S).all() and (pi_S[0] < pi_S[1]) and (pi_S < pi_C).all()),
            Pi_NE=pi_NE.tolist(),
            Pi_C=pi_C.tolist(),
            Pi_stackelberg=pi_S.tolist(),
        )

        rows = []
        for cell_name, rule, in_schedule in _CELLS:
            sub_cfg = ExperimentConfig(
                name=f"obin_{cell_name}",
                market=d,
                sellers=C.make_mixed_duopoly(
                    oblivious_schedule=_THEOREM_OB,
                    informed_schedule=in_schedule,
                    forecast_rule=rule,
                ),
                oblivious_projection=box_ob,
                informed_projection=InformedProjectionBox.from_demand(d),
                horizon=horizon,
                n_seeds=n_seeds,
                base_seed=base_seed,
                log_every=cfg.log_every,
            )
            run.logger.info(
                "running ob-in revenue cell %s (rule=%s, ob=%s, in=%s)",
                cell_name, rule, _THEOREM_OB, in_schedule,
            )
            res = run_simulation(sub_cfg, logger=run.logger)
            cum = benchmarks.cumulative_revenue(res)
            avg = benchmarks.average_revenue(res)
            stats = benchmarks.revenue_summary_statistics(res)
            run.log_event(
                "obin_revenue",
                cell=cell_name,
                rule=rule,
                informed_eta=in_schedule.eta,
                avg_revenue_mean=stats["mean"].tolist(),
                avg_revenue_p05=stats["p05"].tolist(),
                avg_revenue_p95=stats["p95"].tolist(),
                final_p1_mean=float(res.tilde_p[-1, 0].mean()),
                final_p2_mean=float(res.tilde_p[-1, 1].mean()),
            )
            run.save_trajectory(
                f"obin_{cell_name}",
                **res.trajectories_dict(),
                cumulative_revenue=cum,
                average_revenue=avg,
            )
            rows.append(
                dict(
                    cell=cell_name,
                    rule=rule,
                    informed_eta=float(in_schedule.eta or 0.0),
                    avg_R_T_seller0_mean=float(stats["mean"][0]),
                    avg_R_T_seller1_mean=float(stats["mean"][1]),
                    avg_R_T_seller0_p05=float(stats["p05"][0]),
                    avg_R_T_seller0_p95=float(stats["p95"][0]),
                    avg_R_T_seller1_p05=float(stats["p05"][1]),
                    avg_R_T_seller1_p95=float(stats["p95"][1]),
                    Pi_NE_seller0=float(pi_NE[0]),
                    Pi_NE_seller1=float(pi_NE[1]),
                    Pi_C_seller0=float(pi_C[0]),
                    Pi_S_seller0=float(pi_S[0]),
                    Pi_S_seller1=float(pi_S[1]),
                )
            )
            cum_fig = plot_cumulative_revenue(res, title=f"Mixed ob-in, cell={cell_name}")
            run.save_figure(f"cumulative_revenue_{cell_name}", cum_fig, close=False)
            # Export the perfect_prediction case (Stackelberg) and the canonical
            # eta = 0.5 mean_price baseline.
            if cell_name in ("perfect_prediction", "mean_price_eta05"):
                export_figure(cum_fig, f"fig_ob_in_cumrev_{cell_name}", strip_title=True)
            else:
                import matplotlib.pyplot as _plt

                _plt.close(cum_fig)

        df = pd.DataFrame(rows)
        run.save_summary("obin_revenue_summary", df)
        export_table(df, "table_ob_in_obin_revenue", caption=(
            "Mixed-market revenue per forecast rule. Reference benchmarks: "
            "$\\Pi^{NE}$, $\\Pi^{C}$, $\\Pi^{*}$ (Stackelberg)."
        ))
        run.logger.info("exp_ob_in_revenue finished")


if __name__ == "__main__":
    main()
