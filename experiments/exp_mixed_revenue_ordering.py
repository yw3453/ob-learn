"""Revenue ordering in asymmetric multi-seller markets (N >= 3).

Same asymmetric demand-primitive sampling as :mod:`exp_mixed_multiseller`,
restricted to ``N \\in \\{3, 5\\}``. For each ``N``, the composition
``|I^{ob}| \\in \\{0, 1, ..., N\\}`` is swept across the full strategy-game
grid (so the same demand parameters are reused, with only the
informed/oblivious tagging changing). Pure-type endpoints (``|I^{ob}| = 0``
and ``|I^{ob}| = N``) generalize the on-diagonal ``in``-``in`` and
``ob``-``ob`` cells; intermediate compositions generalize the off-diagonal
``ob``-``in`` cell. All cells use a single oblivious dithering
``nu^2 = 0.10`` and informed schedule ``nu_n^2 = 0.10 (n+1)^{-0.25}``.

For every cell we compute per-seller average per-period revenues at
horizons ``T \\in \\{10^4, 10^5\\}`` and the per-seller surplus-capture
ratios ``S_i = (R_{T,i}/T - Pi_i^{NE}) / (Pi_i^{C} - Pi_i^{NE})``,
then group-average over ``i \\in I^{ob}`` and ``i \\in I^{in}``.

The acceptance criterion is that for every ``(N, |I^{ob}|)`` pair, the
group-mean informed surplus-capture ``S^{in}`` is strictly above the
group-mean oblivious surplus-capture ``S^{ob}`` at ``T = 10^5``, and
the gap widens with ``T`` as the oblivious sellers pay a persistent
exploration tax while the informed sellers' tax decays.
"""

from __future__ import annotations

import _common as C  # type: ignore[import-not-found]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import market
from src.artifact_export import export_figure, export_table
from src.config import ExplorationSchedule, InformedProjectionBox
from src.logging_utils import run_directory
from src.plotting import SQUARE_FIGSIZE, report_style, square_box
from src.simulator import SimulationResult, run_simulation

_NS = (3, 5)
_NU2_OB = 0.10
_ETA_INFORMED = 0.25
_C_INFORMED = 0.10
_HORIZONS_FOR_TABLE_DEFAULT = (10_000, 100_000)


def _average_revenue_up_to(result: SimulationResult, t_target: int) -> np.ndarray:
    """Per-seller average per-period revenue averaged over log snapshots up to ``t_target``.

    Returns shape ``(N, S)``. Uses the standard convention of the existing
    revenue scripts: average over all log snapshots ``t_k <= t_target``,
    which approximates ``(1 / T) \\sum_{t<=T} p_{t,i} d_{t,i}`` when
    ``log_every`` is small relative to ``T``.
    """
    inst = result.prices * result.demands  # (T_log, N, S)
    idx = int(np.searchsorted(result.log_steps, t_target, side="right")) - 1
    if idx < 0:
        idx = 0
    return inst[: idx + 1].mean(axis=0)


def main(
    *,
    horizon: int = 100_000,
    n_seeds: int = 80,
    base_seed: int = 67,
    quick: bool = False,
    c_informed: float = _C_INFORMED,
    eta_informed: float = _ETA_INFORMED,
    horizons_for_table: tuple[int, ...] = _HORIZONS_FOR_TABLE_DEFAULT,
    run_name_suffix: str = "",
    figure_basename: str = "fig_mixed_revenue_ordering_surplus_bars",
    table_basename: str = "table_mixed_revenue_ordering_revenue_ordering",
) -> None:
    horizon, n_seeds = C.quick_overrides(quick, default_T=horizon, default_S=n_seeds)

    rep_d = C.asymmetric_market(_NS[0], base_seed=base_seed)
    rep_box_ob = C.tight_oblivious_box(rep_d, expand=0.5)
    rep_box_in = InformedProjectionBox.from_demand(rep_d)
    rep_sched_ob = ExplorationSchedule(kind="constant", nu=float(np.sqrt(_NU2_OB)))
    rep_sched_in = ExplorationSchedule(kind="polynomial", c=c_informed, eta=eta_informed)

    run_name = "exp_mixed_revenue_ordering" + (
        f"_{run_name_suffix}" if run_name_suffix else ""
    )
    cfg = C.base_config(
        name=run_name,
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

    with run_directory(run_name, cfg) as run:
        run.logger.info(
            "asymmetric revenue sweep: N in %s, |I^ob| swept across {0,...,N}, "
            "nu^2_ob=%.3f, c_informed=%.4f, eta_informed=%.2f, T=%d, S=%d, "
            "horizons_to_report=%s",
            list(_NS), _NU2_OB, c_informed, eta_informed, horizon, n_seeds,
            list(horizons_for_table),
        )

        rows: list[dict] = []
        for N in _NS:
            try:
                d = C.asymmetric_market(N, base_seed=base_seed)
            except ValueError as exc:
                run.logger.warning("skipping N=%d: %s", N, exc)
                continue
            box_ob = C.tight_oblivious_box(d, expand=0.5)
            box_in = InformedProjectionBox.from_demand(d)
            p_NE = market.nash_prices(d)
            p_C = market.collusive_prices(d)
            pi_NE = market.per_period_revenue(d, p_NE)
            pi_C = market.per_period_revenue(d, p_C)
            surplus = pi_C - pi_NE  # (N,) per-seller gap
            if (surplus <= 0).any():
                run.logger.warning(
                    "N=%d: some sellers have non-positive surplus gap "
                    "(Pi_C - Pi_NE). Surplus-capture ratios are ill-defined "
                    "for those sellers; reporting raw revenues instead.", N,
                )
            run.logger.info(
                "N=%d: alpha=%s beta=%s p_NE=%s p_C=%s "
                "Pi_NE=%s Pi_C=%s",
                N,
                np.round(d.alpha_arr, 3).tolist(),
                np.round(d.beta_arr, 3).tolist(),
                np.round(p_NE, 3).tolist(),
                np.round(p_C, 3).tolist(),
                np.round(pi_NE, 3).tolist(),
                np.round(pi_C, 3).tolist(),
            )
            beta_abs_min = 0.5 * float(np.min(np.abs(d.beta_arr)))
            for n_ob in range(0, N + 1):
                n_in = N - n_ob
                sched_ob = ExplorationSchedule(
                    kind="constant", nu=float(np.sqrt(_NU2_OB))
                )
                sched_in = ExplorationSchedule(
                    kind="polynomial", c=c_informed, eta=eta_informed
                )
                ob_idx = list(range(n_ob))
                in_idx = list(range(n_ob, N))
                smallgain = market.master_theorem_smallgain(
                    d, ob_idx, in_idx, box_ob, box_in,
                    nu_squared=_NU2_OB, beta_abs_min=beta_abs_min,
                )
                sub_cfg = C.base_config(
                    name=f"M3_N{N}_ob{n_ob}",
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
                    "running N=%d, |I^ob|=%d, |I^in|=%d: margin=%+.4f",
                    N, n_ob, n_in, smallgain["margin"],
                )
                res = run_simulation(sub_cfg, logger=run.logger)

                # Reuse the same simulation across both reported horizons by
                # averaging over the appropriate snapshot prefix. In quick
                # mode the configured ``horizon`` may be below the default
                # 1e5; fall back to ``horizon`` so the summary table still has rows.
                horizons_to_report = [
                    t for t in horizons_for_table if t <= horizon
                ] or [horizon - 1]
                for t_target in horizons_to_report:
                    avg_rev = _average_revenue_up_to(res, t_target)  # (N, S)
                    seller_mean_rev = avg_rev.mean(axis=1)  # (N,)
                    surplus_per_seller = np.where(
                        surplus > 1e-9,
                        (avg_rev.mean(axis=1) - pi_NE) / np.maximum(surplus, 1e-12),
                        np.nan,
                    )  # (N,)
                    s_ob_group = (
                        float(np.nanmean(surplus_per_seller[ob_idx])) if ob_idx else float("nan")
                    )
                    s_in_group = (
                        float(np.nanmean(surplus_per_seller[in_idx])) if in_idx else float("nan")
                    )
                    rev_seed = (res.prices * res.demands)  # (T_log, N, S)
                    # Per-seed time-averaged revenue up to t_target, then group-aggregate.
                    idx = int(np.searchsorted(res.log_steps, t_target, side="right")) - 1
                    if idx < 0:
                        idx = 0
                    avg_per_seed = rev_seed[: idx + 1].mean(axis=0)  # (N, S)
                    s_per_seed = np.where(
                        surplus[:, None] > 1e-9,
                        (avg_per_seed - pi_NE[:, None])
                        / np.maximum(surplus[:, None], 1e-12),
                        np.nan,
                    )  # (N, S)
                    s_ob_per_seed = (
                        np.nanmean(s_per_seed[ob_idx, :], axis=0)
                        if ob_idx else np.full(n_seeds, np.nan)
                    )
                    s_in_per_seed = (
                        np.nanmean(s_per_seed[in_idx, :], axis=0)
                        if in_idx else np.full(n_seeds, np.nan)
                    )
                    row = dict(
                        N=N,
                        n_ob=n_ob,
                        n_in=n_in,
                        t_target=int(t_target),
                        S_ob_group_mean=s_ob_group,
                        S_in_group_mean=s_in_group,
                        S_ob_seed_p05=float(np.nanpercentile(s_ob_per_seed, 5)),
                        S_ob_seed_p95=float(np.nanpercentile(s_ob_per_seed, 95)),
                        S_in_seed_p05=float(np.nanpercentile(s_in_per_seed, 5)),
                        S_in_seed_p95=float(np.nanpercentile(s_in_per_seed, 95)),
                        gap_S_in_minus_ob=s_in_group - s_ob_group,
                        margin=float(smallgain["margin"]),
                        condition_holds=bool(smallgain["condition_holds"]),
                        avg_rev_per_seller=seller_mean_rev.tolist(),
                    )
                    rows.append(row)
                    run.log_event(
                        "M3_cell",
                        **{k: v for k, v in row.items() if k != "avg_rev_per_seller"},
                    )

        df = pd.DataFrame(rows)
        run.save_summary("M3_revenue_ordering", df)

        # ---- Bar plot of group-mean surplus capture at the longest horizon. --
        if not df.empty:
            t_target_for_plot = int(df["t_target"].max())
            df_plot = df[df["t_target"] == t_target_for_plot].copy()
            with report_style():
                Ns_seen = sorted(df_plot["N"].unique())
                fig, axes = plt.subplots(
                    1, len(Ns_seen),
                    figsize=(SQUARE_FIGSIZE[0] * len(Ns_seen), SQUARE_FIGSIZE[1]),
                    sharey=True,
                    squeeze=False,
                )
                axes = axes[0]
                for ax_i, N_val in zip(axes, Ns_seen, strict=False):
                    sub = df_plot[df_plot["N"] == N_val].sort_values("n_ob")
                    xs = np.arange(len(sub))
                    width = 0.36
                    s_ob = sub["S_ob_group_mean"].to_numpy()
                    s_in = sub["S_in_group_mean"].to_numpy()
                    err_ob = np.array([
                        np.clip(s_ob - sub["S_ob_seed_p05"].to_numpy(), 0.0, None),
                        np.clip(sub["S_ob_seed_p95"].to_numpy() - s_ob, 0.0, None),
                    ])
                    err_in = np.array([
                        np.clip(s_in - sub["S_in_seed_p05"].to_numpy(), 0.0, None),
                        np.clip(sub["S_in_seed_p95"].to_numpy() - s_in, 0.0, None),
                    ])
                    ax_i.bar(
                        xs - width / 2, s_ob, width=width,
                        color="tab:red", edgecolor="black", linewidth=0.6,
                        yerr=err_ob, capsize=3,
                        label=r"$\bar S^{ob}$ (oblivious)",
                    )
                    ax_i.bar(
                        xs + width / 2, s_in, width=width,
                        color="tab:blue", edgecolor="black", linewidth=0.6,
                        yerr=err_in, capsize=3,
                        label=r"$\bar S^{in}$ (informed)",
                    )
                    ax_i.axhline(0.0, color="black", lw=1.0, linestyle="-")
                    ax_i.set_xticks(xs)
                    ax_i.set_xticklabels(
                        [f"{int(n_ob)}/{int(N_val - n_ob)}"
                         for n_ob in sub["n_ob"]]
                    )
                    ax_i.set_xlabel(
                        r"composition $|\mathcal{I}^{ob}|/|\mathcal{I}^{in}|$"
                    )
                    ax_i.set_title(fr"$N = {N_val}$")
                    square_box(ax_i)
                axes[0].set_ylabel("group-mean surplus capture ratio")
                axes[-1].legend(
                    loc="lower right", fontsize=10, framealpha=0.92,
                )
                fig.tight_layout()
            run.save_figure("M3_surplus_bars", fig, close=False)
            export_figure(fig, figure_basename, strip_title=True)
            run.logger.info(
                "M3 surplus-bar plot exported at t_target=%d", t_target_for_plot,
            )

        # Drop the bulky list-valued column from the exported LaTeX summary table,
        # tolerating the case where df is empty / the column is absent.
        df_export = df.drop(columns=["avg_rev_per_seller"], errors="ignore")
        export_table(
            df_export, table_basename,
            caption=(
                "Surplus-capture ratios in asymmetric multi-seller markets at "
                "$N \\in \\{3, 5\\}$, sweeping the full composition "
                "$|\\mathcal I^{ob}| \\in \\{0, 1, \\ldots, N\\}$. "
                "$S^{ob}$, $S^{in}$ are group means of "
                "$S_i = (R_{T,i}/T - \\Pi_i^{NE}) / (\\Pi_i^{C} - \\Pi_i^{NE})$ "
                "across the oblivious and informed sellers respectively (empty "
                "for endpoint cells where the corresponding group is empty). "
                "Brackets are cross-seed $5\\%$--$95\\%$ ranges over $S$ seeds."
            ),
            floatfmt=".3g",
        )

        run.logger.info("exp_mixed_revenue_ordering finished")


if __name__ == "__main__":
    main()
