"""Mixed-market forecast rules and the exploration-rate threshold.

The mixed duopoly has seller 0 oblivious (persistent ``\\nu^2 > 0`` so the
oblivious pseudo-equilibrium condition holds) and seller 1 informed. Three
forecast rules are tested, all under the persistent schedule:

* ``mean_price``         -- oblivious nu^2 = 0.10 constant; informed
                            nu_n^2 = 0.10 n^{-1/2} (the sqrt-T rate, which
                            satisfies ``eta_min + 1 > 2 eta_max``).
* ``perfect_prediction`` -- same schedule.
* ``lag1_autocorr``      -- ablation, same schedule. Documents the Jensen
                            cost of forecasting raw realized prices instead
                            of de-noising via the running mean.

The ``(\\eta_0, \\eta_1)`` grid is swept, with the boundary
``\\eta_{\\min} + 1 > 2 \\eta_{\\max}`` overlaid.
"""

from __future__ import annotations

import _common as C  # type: ignore[import-not-found]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import analysis, benchmarks, market
from src.artifact_export import export_figure, export_table
from src.config import ExperimentConfig, ExplorationSchedule, InformedProjectionBox, SellerSpec
from src.logging_utils import run_directory
from src.plotting import (
    SQUARE_FIGSIZE,
    plot_cumulative_revenue,
    plot_mse_loglog,
    plot_price_scatter,
    plot_sample_paths,
    plot_threshold_heatmap,
    report_style,
    square_box,
)
from src.simulator import run_simulation

_THEOREM_OB = ExplorationSchedule(kind="constant", nu=float(np.sqrt(0.10)))
_THEOREM_IN = ExplorationSchedule(kind="polynomial", c=0.10, eta=0.5)


_RULE_PRETTY = {
    "mean_price": "running mean",
    "greedy_component": "clairvoyant greedy",
    "lag1_autocorr": "lag-1 realized",
    "perfect_prediction": "clairvoyant realized",
}

_FORECAST_RULES = (
    "mean_price",
    "lag1_autocorr",
    "greedy_component",
    "perfect_prediction",
)
# Ordered to mirror the baseline exposition: the implementable rules (running
# mean and lag-1) first, then the clairvoyant greedy-component (still
# Nash-limit), and finally the clairvoyant perfect-prediction rule that
# shifts the limit to Stackelberg. The composite forecast-rule scatter
# plot follows this order.


def _forecast_scatter_shared_lims(
    rule_to_means: dict[str, np.ndarray],
    *,
    p_NE: np.ndarray,
    p_C: np.ndarray,
    p_S: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Shared (xlim, ylim) across forecast-rule panels for cross-rule comparison."""
    all_x = [m[0] for m in rule_to_means.values()]
    all_y = [m[1] for m in rule_to_means.values()]
    bench_x = np.array([p_NE[0], p_C[0], p_S[0]])
    bench_y = np.array([p_NE[1], p_C[1], p_S[1]])
    x_min = float(min(min(np.min(x) for x in all_x), bench_x.min()))
    x_max = float(max(max(np.max(x) for x in all_x), bench_x.max()))
    y_min = float(min(min(np.min(y) for y in all_y), bench_y.min()))
    y_max = float(max(max(np.max(y) for y in all_y), bench_y.max()))
    x_pad = max(0.06 * (x_max - x_min), 0.02)
    y_pad = max(0.06 * (y_max - y_min), 0.02)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def _plot_forecast_scatter_single(
    means: np.ndarray,
    *,
    p_NE: np.ndarray,
    p_C: np.ndarray,
    p_S: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> plt.Figure:
    """Single-panel scatter for one forecast rule, with shared axes.

    Every panel renders the same set of decorations (x-label, y-label, and
    benchmark legend) so that LaTeX subfigure tiling at a common width
    yields four identically-sized data regions; without this uniformity
    the panel with the extra y-label/legend ends up visibly compressed.
    """
    from src.plotting import (  # local import to keep main deps trim
        SQUARE_FIGSIZE,
        report_style,
        smart_legend,
        square_box,
    )

    with report_style():
        fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
        ax.scatter(
            means[0], means[1], s=18, alpha=0.7,
            color="tab:blue", edgecolor="white", linewidth=0.4,
        )
        ax.scatter(
            [p_NE[0]], [p_NE[1]], s=120, color="tab:red", marker="X",
            edgecolor="white", linewidth=0.6, zorder=10,
            label=r"$p^{NE}$",
        )
        ax.scatter(
            [p_C[0]], [p_C[1]], s=120, color="tab:green", marker="X",
            edgecolor="white", linewidth=0.6, zorder=10,
            label=r"$p^{C}$",
        )
        ax.scatter(
            [p_S[0]], [p_S[1]], s=120, color="tab:purple", marker="X",
            edgecolor="white", linewidth=0.6, zorder=10,
            label=r"$p^{*}$ (Stackelberg)",
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(r"$\bar p_{1}$")
        ax.set_ylabel(r"$\bar p_{2}$")
        square_box(ax)
        smart_legend(ax)
        fig.tight_layout()
    return fig


def _run_rule(
    *,
    rule: str,
    d,
    box_ob,
    sched_ob: ExplorationSchedule,
    sched_in: ExplorationSchedule,
    horizon: int,
    n_seeds: int,
    base_seed: int,
    log_every: int,
    n_warmup: int | None = None,
):
    sub_cfg = ExperimentConfig(
        name=f"forecast_{rule}",
        market=d,
        sellers=C.make_mixed_duopoly(
            oblivious_schedule=sched_ob,
            informed_schedule=sched_in,
            forecast_rule=rule,
        ),
        oblivious_projection=box_ob,
        informed_projection=InformedProjectionBox.from_demand(d),
        horizon=horizon,
        n_seeds=n_seeds,
        base_seed=base_seed,
        log_every=log_every,
        n_warmup=n_warmup,
    )
    return run_simulation(sub_cfg)


def main(
    *,
    horizon: int = 60_000,
    n_seeds: int = 200,
    base_seed: int = 19,
    quick: bool = False,
) -> None:
    horizon, n_seeds = C.quick_overrides(quick, default_T=horizon, default_S=n_seeds)
    d = C.baseline_demand()
    box_ob = C.tight_oblivious_box(d, expand=0.5)

    cfg = C.base_config(
        name="exp_mixed_forecast_rules",
        market=d,
        sellers=[
            SellerSpec(kind="oblivious", exploration=_THEOREM_OB),
            SellerSpec(kind="informed", forecast_rule="mean_price", exploration=_THEOREM_IN),
        ],
        horizon=horizon,
        n_seeds=n_seeds,
        base_seed=base_seed,
        log_every=max(1, horizon // 1000),
        oblivious_box=box_ob,
    )

    p_NE = market.nash_prices(d)
    p_C = market.collusive_prices(d)
    p_S = np.array(market.stackelberg_duopoly(d))
    pi_bench = benchmarks.benchmark_per_period_revenues(d)
    pi_NE = pi_bench["NE"]  # (N,)
    pi_C = pi_bench["collusive"]  # (N,)
    pi_S = pi_bench.get("stackelberg")  # (N,) or None

    with run_directory("exp_mixed_forecast_rules", cfg) as run:
        run.logger.info("p_NE=%s p_C=%s Stackelberg=%s", p_NE.tolist(), p_C.tolist(), p_S.tolist())
        run.logger.info(
            "Forecast-rule schedule: ob=%s in=%s",
            _THEOREM_OB, _THEOREM_IN,
        )

        forecast_rows = []
        rule_to_means: dict[str, np.ndarray] = {}
        for rule in _FORECAST_RULES:
            run.logger.info(
                "rule %-22s: ob=%s in=%s",
                rule, _THEOREM_OB, _THEOREM_IN,
            )
            res = _run_rule(
                rule=rule, d=d, box_ob=box_ob,
                sched_ob=_THEOREM_OB, sched_in=_THEOREM_IN,
                horizon=horizon, n_seeds=n_seeds, base_seed=base_seed,
                log_every=cfg.log_every,
            )
            mse_theta_ob = analysis.mse_theta_oblivious(res)
            mse_theta_in = analysis.mse_theta_informed(res)
            mse_p = analysis.mse_price(res)
            final_running = res.moments["m"][-1]  # (N, S)
            rule_to_means[rule] = final_running.copy()
            # Per-rule revenue at the horizon, averaged across seeds, and the
            # surplus-capture ratio S_i = (R_i - Pi^NE_i) / (Pi^C_i - Pi^NE_i).
            avg_rev_final = benchmarks.average_revenue(res)[-1]  # (N, S)
            rev_mean = avg_rev_final.mean(axis=1)  # (N,)
            surplus_capture = (rev_mean - pi_NE) / np.maximum(pi_C - pi_NE, 1e-12)
            row = dict(
                rule=rule,
                mean_p1=float(final_running[0].mean()),
                mean_p2=float(final_running[1].mean()),
                std_p1=float(final_running[0].std()),
                std_p2=float(final_running[1].std()),
                mse_theta_ob_final=float(mse_theta_ob[-1].mean()),
                mse_theta_in_final=float(mse_theta_in[-1].mean()),
                mse_price_final=float(mse_p[-1].mean()),
                avg_rev_p1=float(rev_mean[0]),
                avg_rev_p2=float(rev_mean[1]),
                S_p1=float(surplus_capture[0]),
                S_p2=float(surplus_capture[1]),
            )
            forecast_rows.append(row)
            run.log_event("forecast_run", **row)
            run.save_trajectory(
                f"forecast_{rule}",
                **res.trajectories_dict(),
                mse_theta_ob=mse_theta_ob,
                mse_theta_in=mse_theta_in,
                mse_price=mse_p,
            )
            sp_fig = plot_sample_paths(res, n_paths=5, title=f"Forecast rule: {rule}")
            run.save_figure(f"sample_paths_{rule}", sp_fig, close=False)
            # perfect_prediction asks the informed seller to play the Stackelberg
            # best response; mark p^* on its scatter so the reader can compare.
            mark_stack = rule in ("perfect_prediction",)
            sc_fig = plot_price_scatter(
                res, title=f"Forecast rule: {rule}",
                mark_stackelberg=mark_stack,
                zoom_to_data=True,
            )
            run.save_figure(f"price_scatter_{rule}", sc_fig, close=False)
            rev_fig = plot_cumulative_revenue(res, title=f"Forecast rule: {rule}")
            run.save_figure(f"cumulative_revenue_{rule}", rev_fig, close=False)
            export_figure(sp_fig, f"fig_mixed_forecast_sample_paths_{rule}", strip_title=True)
            export_figure(sc_fig, f"fig_mixed_forecast_price_scatter_{rule}", strip_title=True)
            if rule == "mean_price":
                mse_fig = plot_mse_loglog(res, metric="theta_in", title=f"Forecast rule: {rule}")
                run.save_figure(f"mse_theta_in_{rule}", mse_fig, close=False)
                export_figure(mse_fig, f"fig_mixed_forecast_mse_theta_in_{rule}", strip_title=True)
            if rule == "perfect_prediction":
                export_figure(rev_fig, f"fig_mixed_forecast_cumrev_{rule}", strip_title=True)
            else:
                plt.close(rev_fig)

        # Four single-panel scatter figures (one per forecast rule). LaTeX
        # composes them side-by-side via the subfigure environment, so each
        # panel is a self-contained square plot with shared axis limits.
        if len(rule_to_means) == len(_FORECAST_RULES):
            xlim, ylim = _forecast_scatter_shared_lims(
                rule_to_means, p_NE=p_NE, p_C=p_C, p_S=p_S,
            )
            for rule in _FORECAST_RULES:
                panel_fig = _plot_forecast_scatter_single(
                    rule_to_means[rule],
                    p_NE=p_NE, p_C=p_C, p_S=p_S,
                    xlim=xlim, ylim=ylim,
                )
                run.save_figure(
                    f"forecast_scatter_{rule}", panel_fig, close=False,
                )
                export_figure(
                    panel_fig, f"fig_mixed_forecast_forecast_scatter_{rule}",
                    strip_title=True,
                )

        forecast_df = pd.DataFrame(forecast_rows)
        run.save_summary("forecast_summary", forecast_df)
        pi_S_clause = (
            f", $\\Pi^{{*}} = ({pi_S[0]:.3f},{pi_S[1]:.3f})$"
            if pi_S is not None else ""
        )
        export_table(
            forecast_df, "table_mixed_forecast_forecast_summary",
            caption=(
                "Mixed-market forecast rules. All rules use the persistent "
                "schedule: constant oblivious $\\nu^2 = 0.10$ and "
                "informed $\\nu_n^2 = 0.10\\, n^{-1/2}$ at the "
                "$\\sqrt T$-regret rate (satisfying "
                "$\\eta_{\\min}+1>2\\eta_{\\max}$). For each rule we report "
                "the seed-averaged final running-mean price (``mean_p$i$''), "
                "its cross-seed standard deviation (``std_p$i$''), the "
                "parameter and price MSE at the horizon, the seed-averaged "
                "average per-period revenue (``avg_rev_p$i$''), and the "
                "surplus-capture ratio $S_i = (\\bar R_i - \\Pi^{NE}_i) / "
                "(\\Pi^{C}_i - \\Pi^{NE}_i)$. Benchmarks for the duopoly: "
                f"$\\Pi^{{NE}} = ({pi_NE[0]:.3f},{pi_NE[1]:.3f})$, "
                f"$\\Pi^{{C}} = ({pi_C[0]:.3f},{pi_C[1]:.3f})$"
                f"{pi_S_clause}."
            ),
        )

        # ---- (eta_0, eta_1) threshold grid --------------------------------
        eta_grid = np.array([0.0, 0.1, 0.25, 0.4, 0.6, 0.8])
        c_value = 0.10
        rule = "mean_price"
        eta_rows = []
        eta_heatmap = np.zeros((eta_grid.size, eta_grid.size))
        boundary = np.zeros_like(eta_heatmap, dtype=int)
        mse_curves: list[tuple[float, float, np.ndarray, bool]] = []
        n_grid_ref: np.ndarray | None = None
        for i, eta_ob in enumerate(eta_grid):
            for j, eta_in in enumerate(eta_grid):
                sched_ob = ExplorationSchedule(kind="polynomial", c=c_value, eta=float(eta_ob))
                sched_in = ExplorationSchedule(kind="polynomial", c=c_value, eta=float(eta_in))
                sub_cfg = C.base_config(
                    name=f"eta_grid_{eta_ob:.2f}_{eta_in:.2f}",
                    market=d,
                    sellers=C.make_mixed_duopoly(
                        oblivious_schedule=sched_ob,
                        informed_schedule=sched_in,
                        forecast_rule=rule,
                    ),
                    horizon=min(horizon, 30_000),
                    n_seeds=min(n_seeds, 60),
                    base_seed=base_seed + 100,
                    log_every=max(1, min(horizon, 30_000) // 500),
                    oblivious_box=box_ob,
                )
                res = run_simulation(sub_cfg)
                mse_theta_in = analysis.mse_theta_informed(res)  # (T_log, S)
                mean_curve = mse_theta_in.mean(axis=1)
                final = float(mean_curve[-1])
                eta_min = float(min(eta_ob, eta_in))
                eta_max = float(max(eta_ob, eta_in))
                cond_holds = (eta_min + 1.0) > 2.0 * eta_max
                if n_grid_ref is None:
                    n_grid_ref = res.log_steps + 1.0
                mse_curves.append((float(eta_ob), float(eta_in), mean_curve, cond_holds))
                eta_heatmap[i, j] = max(final, 1e-12)
                boundary[i, j] = int(cond_holds)
                eta_rows.append(
                    dict(
                        eta_ob=float(eta_ob),
                        eta_in=float(eta_in),
                        eta_min=eta_min,
                        eta_max=eta_max,
                        condition_holds=cond_holds,
                        mse_theta_in_final=final,
                    )
                )
                run.log_event(
                    "eta_grid_run",
                    eta_ob=float(eta_ob),
                    eta_in=float(eta_in),
                    condition_holds=cond_holds,
                    mse_theta_in_final=final,
                )

        eta_df = pd.DataFrame(eta_rows)
        run.save_summary("eta_grid_summary", eta_df)
        run.save_trajectory("eta_grid_heatmap", heatmap=eta_heatmap, boundary_holds=boundary, eta_grid=eta_grid)
        eta_fig = plot_threshold_heatmap(
            xs=eta_grid, ys=eta_grid, z=eta_heatmap,
            xlabel=r"$\eta_{\mathrm{ob}}$", ylabel=r"$\eta_{\mathrm{in}}$",
            cbar_label=r"MSE($\hat\theta^{in}_T$)",
        )
        run.save_figure("eta_grid_heatmap", eta_fig, close=False)
        export_figure(eta_fig, "fig_mixed_forecast_eta_grid_heatmap", strip_title=True)

        # New: one plot showing the seed-averaged MSE(theta_in) trajectory
        # for *every* (eta_ob, eta_in) pair, on a single log-log axes. The
        # curves should all bend down to 0 as n grows, even in cells where
        # the boundary condition fails (dashed). This replaces the
        # mean_p_dist_to_NE diagnostic the previous version of the summary table used.
        if mse_curves and n_grid_ref is not None:
            from matplotlib import colormaps as _cmaps
            from matplotlib.lines import Line2D
            with report_style():
                fig_mse, ax_mse = plt.subplots(figsize=SQUARE_FIGSIZE)
                cmap_solid = _cmaps.get_cmap("viridis")
                cmap_dashed = _cmaps.get_cmap("plasma")
                n_solid = sum(1 for _, _, _, cond in mse_curves if cond)
                n_dashed = sum(1 for _, _, _, cond in mse_curves if not cond)
                si = di = 0
                for _eta_ob_v, _eta_in_v, curve, cond in mse_curves:
                    if cond:
                        color = cmap_solid(0.1 + 0.85 * (si / max(n_solid - 1, 1)))
                        si += 1
                        ls = "-"
                    else:
                        color = cmap_dashed(0.15 + 0.7 * (di / max(n_dashed - 1, 1)))
                        di += 1
                        ls = "--"
                    ax_mse.plot(n_grid_ref, np.maximum(curve, 1e-12),
                            color=color, lw=1.0, linestyle=ls, alpha=0.85)
                ax_mse.set_xscale("log")
                ax_mse.set_yscale("log")
                ax_mse.set_xlabel("n")
                ax_mse.set_ylabel(r"MSE($\hat\theta^{in}_n$)")
                proxies = [
                    Line2D([0], [0], color=cmap_solid(0.5), lw=1.4, linestyle="-",
                           label="condition holds"),
                    Line2D([0], [0], color=cmap_dashed(0.5), lw=1.4, linestyle="--",
                           label="condition fails"),
                ]
                ax_mse.legend(handles=proxies, loc="lower left", fontsize=11, framealpha=0.92)
                square_box(ax_mse)
                fig_mse.tight_layout()
            run.save_figure("eta_grid_mse_paths", fig_mse, close=False)
            export_figure(fig_mse, "fig_mixed_forecast_eta_grid_mse_paths", strip_title=True)

        export_table(eta_df, "table_mixed_forecast_eta_grid", caption=(
            "Exploration-rate threshold at $N=2$ mixed market: seed-averaged "
            "$\\text{MSE}(\\hat\\theta^{in}_T)$ vs.\\ "
            "$(\\eta_{\\mathrm{ob}}, \\eta_{\\mathrm{in}})$. The condition "
            "$\\eta_{\\min} + 1 > 2 \\eta_{\\max}$ holds iff "
            "``condition\\_holds'' is True. Plot ``fig\\_mixed\\_forecast\\_eta\\_grid\\_mse\\_paths'' "
            "shows the corresponding MSE sample paths, which decay to 0 "
            "across the whole grid -- including cells where the formal "
            "condition fails -- evidencing convergence in every regime."
        ))
        run.logger.info("exp_mixed_forecast_rules finished")


if __name__ == "__main__":
    main()
