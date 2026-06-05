"""All-informed market at the sqrt-T exploration rate.

In an all-informed market with the running-mean forecast and exploration
``nu_{n,i}^2 = c_i n^{-1/2}``, the realized price-MSE
``E ||p_n - p^{NE}||_2^2`` decays at the ``sqrt(T)`` rate, i.e. with a
log--log slope ``\\approx -1/2``.

Two panels:

* **EA-symm** (symmetric all-informed market):
    - ``N \\in \\{2, 5\\}``, common ``alpha = 2.5``, ``beta = -1``.
    - ``gamma`` chosen inside the diagonal-dominance regime
      ``(N - 1) gamma < |beta|``:
        * ``N = 2``: ``gamma = 0.4`` (baseline duopoly default).
        * ``N = 5``: ``gamma = 0.10`` (so ``(N - 1) gamma = 0.4 < 1``).
    - Informed leading constant sweep ``c \\in \\{0.05, 0.10, 0.20\\}``;
      ``eta = 0.5`` on every seller.
* **EA-asym** (asymmetric all-informed market):
    - ``N \\in \\{3, 5\\}`` with heterogeneous demand primitives drawn
      once per ``N`` from the same distribution as the multi-seller mixed
      experiment.
    - Same ``c`` sweep and ``eta = 0.5`` schedule.

For each ``(panel, N, c)`` cell we record (i) the realized-price MSE
trajectory (seed mean), (ii) the regularity quantity
``lambda_max(B + B^T)`` with ``B = I - (1/2) diag(1/beta_i) Gamma``
(convergence requires this to be ``< 1``), (iii) the empirical log-log
tail slope.

Outputs (under ``results/figures/``):

* ``fig_all_informed_allinformed_mse_paths.pdf``: combined two-panel plot.
* ``fig_all_informed_symm_mse_paths.pdf``: symmetric panel only.
* ``fig_all_informed_asym_mse_paths.pdf``: asymmetric panel only.
* ``table_all_informed_allinformed.{csv,md}``: per-cell summary summary table.
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
from src.config import (
    DemandParams,
    ExplorationSchedule,
    InformedProjectionBox,
    SellerSpec,
)
from src.logging_utils import run_directory
from src.plotting import SQUARE_FIGSIZE, report_style, square_box
from src.simulator import run_simulation

# ---- Sweep parameters ------------------------------------------------------
_SYMM_NS = (2, 5)
_ASYM_NS = (3, 5)
_C_GRID = (0.05, 0.10, 0.20)
_ETA = 0.5  # sqrt-T exploration rate.

# Symmetric markets: gamma must stay strictly inside (N - 1) gamma < |beta| = 1.
_SYMM_GAMMA = {2: 0.4, 5: 0.10}

# EA-stress: an asymmetric N=3 market designed so that every cell in the
# sweep *violates* the regularity quantity
# lambda_max(B + B^T) > 1 while the underlying ODE (rho(B) < 1) and
# Nash existence (own-price and collusive dominance) all stay healthy.
# In symmetric markets the two coincide -- the validator's own-price
# dominance is exactly the regularity quantity -- so we have to
# introduce a "fragile" seller with small |beta_3| = 0.05 to decouple
# them and push lambda_max above 1.
#
# Topology: sellers 1, 2 are symmetric peers (beta = -1); seller 3 has
# small |beta_3| = 0.05. Cross-effects on seller 3 are small
# (gamma_13 = gamma_23 = 0.025, gamma_31 = gamma_32 = 0.024), close to
# the own-price + collusive boundaries so M_{1,3} approaches its
# structural maximum. The swept dimension is gamma_12 = gamma_21 in
# {0.88, 0.91, 0.94, 0.96}, which yields lambda_max(B + B^T) in
# {1.01, 1.03, 1.06, 1.08} (every cell violates the regularity) while
# rho(B) stays in {0.45, 0.47, 0.48, 0.49}, comfortably stable.
_STRESS_N = 3
_STRESS_BETA = (-1.0, -1.0, -0.05)
_STRESS_ALPHA = (1.5, 1.5, 0.15)  # keeps p_NE inside [0.5, 2.5] across sweep.
_STRESS_GAMMA13 = 0.025  # gamma_{1,3} = gamma_{2,3}
_STRESS_GAMMA31 = 0.024  # gamma_{3,1} = gamma_{3,2}
_STRESS_GAMMAS = (0.88, 0.91, 0.94, 0.96)
_STRESS_C = 0.10


# ---------------------------------------------------------------------------
# Market and seller helpers
# ---------------------------------------------------------------------------


def _make_all_informed(
    N: int, schedule: ExplorationSchedule, *, forecast_rule: str = "mean_price"
) -> list[SellerSpec]:
    return [
        SellerSpec(
            kind="informed",
            forecast_rule=forecast_rule,  # type: ignore[arg-type]
            exploration=schedule,
        )
        for _ in range(N)
    ]


def _make_symm_market(N: int) -> DemandParams:
    return C.symmetric_market(N, gamma=_SYMM_GAMMA[N], noise_std=0.2)


def _make_asym_market(N: int, *, base_seed: int) -> DemandParams:
    # Same primitive sampler as the multi-seller mixed experiment.
    return C.asymmetric_market(N, base_seed=base_seed)


def _make_stress_market(gamma_12: float) -> DemandParams:
    """``N = 3`` asymmetric market where ``gamma_12 = gamma_21`` is the
    swept coordinate; all other primitives are fixed (see module-level
    constants). Returns a valid ``DemandParams`` (own-price and
    collusive dominance hold for every cell in ``_STRESS_GAMMAS``).
    """
    G = [
        [0.0, gamma_12, _STRESS_GAMMA13],
        [gamma_12, 0.0, _STRESS_GAMMA13],
        [_STRESS_GAMMA31, _STRESS_GAMMA31, 0.0],
    ]
    return DemandParams(
        N=_STRESS_N,
        alpha=list(_STRESS_ALPHA),
        beta=list(_STRESS_BETA),
        gamma=G,
        l=0.5,
        u=2.5,
        noise_kind="uniform",
        noise_std=0.2,
    )


# ---------------------------------------------------------------------------
# Regularity quantity lambda_max(B + B^T).
#
# Gamma is the Nash matrix
# with diagonal 2*beta_i and off-diagonal gamma_{i,j}; B = I - (1/2)
# diag(1/beta_i) Gamma. Note diag(B) = 0 since diag(diag(1/beta) Gamma)
# = 2. The off-diagonal entry is B_{ij} = -gamma_{i,j} / (2 beta_i)
# = gamma_{i,j} / (2 |beta_i|) (since beta_i < 0).
# ---------------------------------------------------------------------------


def _corollary_regularity(d: DemandParams) -> dict[str, float | bool]:
    Gamma = market.gamma_matrix(d)
    diag_inv_beta = np.diag(1.0 / d.beta_arr)
    B = np.eye(d.N) - 0.5 * diag_inv_beta @ Gamma
    sym = B + B.T
    lam = float(np.linalg.eigvalsh(sym).max())
    return {
        "lambda_max_B_plus_Bt": lam,
        "regularity_holds": bool(lam < 1.0),
    }


# ---------------------------------------------------------------------------
# Realized-price MSE (implication statement is on p_n, not tilde p_n).
# ---------------------------------------------------------------------------


def _mse_realized_price(result) -> np.ndarray:
    """``sum_i (p_{n,i} - p_i^{NE})^2`` per (log step, seed). Shape ``(T_log, S)``."""
    p_NE = market.nash_prices(result.config.market)
    diff = result.prices - p_NE[None, :, None]  # (T_log, N, S)
    return np.sum(diff**2, axis=1)


# ---------------------------------------------------------------------------
# One cell of the sweep.
# ---------------------------------------------------------------------------


def _run_cell(
    panel: str,
    N: int,
    c_lead: float,
    d: DemandParams,
    horizon: int,
    n_seeds: int,
    base_seed: int,
    log_every: int,
    logger,
) -> tuple[np.ndarray, np.ndarray, dict]:
    sched = ExplorationSchedule(kind="polynomial", c=float(c_lead), eta=_ETA)
    box_ob = C.tight_oblivious_box(d, expand=0.5)
    box_in = InformedProjectionBox.from_demand(d)
    sub_cfg = C.base_config(
        name=f"EA_{panel}_N{N}_c{c_lead:.2f}",
        market=d,
        sellers=_make_all_informed(N, sched),
        horizon=horizon,
        n_seeds=n_seeds,
        base_seed=base_seed,
        log_every=log_every,
        oblivious_box=box_ob,
        informed_box=box_in,
    )
    reg = _corollary_regularity(d)
    logger.info(
        "running EA-%s N=%d c=%.3f eta=%.2f: lambda_max(B+B^T)=%.3f (holds=%s)",
        panel, N, c_lead, _ETA, reg["lambda_max_B_plus_Bt"], reg["regularity_holds"],
    )
    res = run_simulation(sub_cfg, logger=logger)
    mse_realized = _mse_realized_price(res)  # (T_log, S)
    mse_curve = mse_realized.mean(axis=1)
    n_grid = (res.log_steps + 1).astype(np.float64)
    slope_info = analysis.fit_loglog_slope(n_grid, mse_curve, tail_fraction=0.5)
    meta = dict(
        panel=panel,
        N=int(N),
        c=float(c_lead),
        eta=float(_ETA),
        lambda_max_B_plus_Bt=float(reg["lambda_max_B_plus_Bt"]),
        regularity_holds=bool(reg["regularity_holds"]),
        mse_price_final=float(mse_curve[-1]),
        slope_tail=float(slope_info["slope"]),
        slope_n_used=int(slope_info["n_used"]),
    )
    return n_grid, mse_curve, meta


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _color_for_N(N_grid: tuple[int, ...]):
    cmap = _cmaps.get_cmap("viridis")
    return {
        int(N_): cmap(0.10 + 0.80 * idx / max(len(N_grid) - 1, 1))
        for idx, N_ in enumerate(N_grid)
    }


def _ls_for_c():
    return {float(c): ls for c, ls in zip(_C_GRID, ("-", "--", ":"), strict=False)}


def _plot_panel(
    ax, curves: list[tuple[int, float, np.ndarray, np.ndarray]], N_grid: tuple[int, ...]
) -> None:
    color_for_N = _color_for_N(N_grid)
    ls_for_c = _ls_for_c()
    for N_, c_, n_axis, curve in curves:
        ax.plot(
            n_axis, np.maximum(curve, 1e-12),
            color=color_for_N.get(int(N_), "tab:blue"),
            linestyle=ls_for_c.get(float(c_), "-"),
            lw=1.4, alpha=0.90,
        )
    # Reference 1/sqrt(n) slope line (anchored at the upper-left of the data).
    if curves:
        n_min = min(n_axis.min() for _, _, n_axis, _ in curves)
        n_max = max(n_axis.max() for _, _, n_axis, _ in curves)
        # Anchor: pick the maximum MSE at the earliest log step across cells.
        anchors_at_start = [
            np.maximum(curve, 1e-12)[np.argmin(np.abs(n_axis - n_min))]
            for _, _, n_axis, curve in curves
        ]
        anchor = float(np.max(anchors_at_start))
        n_ref = np.geomspace(max(n_min, 1.0), n_max, num=64)
        y_ref = anchor * (n_ref / n_ref[0]) ** (-0.5)
        ax.plot(
            n_ref, y_ref, color="0.30", lw=1.0, linestyle=(0, (4, 3)),
            label=r"reference $n^{-1/2}$",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"MSE($p_n$)")
    handles_N = [
        Line2D([0], [0], color=color_for_N[int(N_)], lw=1.6, label=f"N = {N_}")
        for N_ in N_grid
    ]
    handles_c = [
        Line2D([0], [0], color="black", lw=1.4,
               linestyle=ls_for_c[float(c_)],
               label=fr"$c = {c_:.2f}$")
        for c_ in _C_GRID
    ]
    ref_handle = [
        Line2D([0], [0], color="0.30", lw=1.0, linestyle=(0, (4, 3)),
               label=r"$n^{-1/2}$"),
    ]
    ax.legend(
        handles=handles_N + handles_c + ref_handle, loc="lower left",
        fontsize=9, framealpha=0.92, ncol=2,
    )
    square_box(ax)


def _save_single_panel_figure(
    curves: list[tuple[int, float, np.ndarray, np.ndarray]],
    N_grid: tuple[int, ...],
    *,
    name: str,
    run,
) -> None:
    if not curves:
        return
    with report_style():
        fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
        _plot_panel(ax, curves, N_grid)
        fig.tight_layout()
    run.save_figure(f"{name}", fig, close=False)
    export_figure(fig, f"fig_{name}_mse_paths", strip_title=True)


def _plot_stress_panel(
    ax, curves: list[tuple[float, np.ndarray, np.ndarray]]
) -> None:
    """Stress panel: color by lambda_max(B+B^T), single linestyle."""
    cmap = _cmaps.get_cmap("plasma")
    n_curves = max(len(curves), 1)
    handles = []
    for idx, (lam, n_axis, curve) in enumerate(curves):
        color = cmap(0.15 + 0.70 * idx / max(n_curves - 1, 1))
        ls = "-" if lam < 1.0 else "--"
        ax.plot(
            n_axis, np.maximum(curve, 1e-12),
            color=color, linestyle=ls, lw=1.4, alpha=0.92,
        )
        handles.append(
            Line2D([0], [0], color=color, lw=1.6, linestyle=ls,
                   label=fr"$\lambda_{{\max}}(B + B^\top) = {lam:.2f}$")
        )
    if curves:
        n_min = min(n_axis.min() for _, n_axis, _ in curves)
        n_max = max(n_axis.max() for _, n_axis, _ in curves)
        anchors_at_start = [
            np.maximum(curve, 1e-12)[np.argmin(np.abs(n_axis - n_min))]
            for _, n_axis, curve in curves
        ]
        anchor = float(np.max(anchors_at_start))
        n_ref = np.geomspace(max(n_min, 1.0), n_max, num=64)
        y_ref = anchor * (n_ref / n_ref[0]) ** (-0.5)
        ax.plot(n_ref, y_ref, color="0.30", lw=1.0, linestyle=(0, (4, 3)))
        handles.append(
            Line2D([0], [0], color="0.30", lw=1.0, linestyle=(0, (4, 3)),
                   label=r"reference $n^{-1/2}$")
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"MSE($p_n$)")
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.92)
    square_box(ax)


def _save_stress_figure(
    curves_stress: list[tuple[float, np.ndarray, np.ndarray]], *, run
) -> None:
    if not curves_stress:
        return
    with report_style():
        fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
        _plot_stress_panel(ax, curves_stress)
        fig.tight_layout()
    run.save_figure("EA_stress", fig, close=False)
    export_figure(fig, "fig_all_informed_stress_mse_paths", strip_title=True)


def _save_combined_figure(
    curves_symm: list[tuple[int, float, np.ndarray, np.ndarray]],
    curves_asym: list[tuple[int, float, np.ndarray, np.ndarray]],
    curves_stress: list[tuple[float, np.ndarray, np.ndarray]],
    *,
    run,
) -> None:
    if not curves_symm and not curves_asym and not curves_stress:
        return
    with report_style():
        fig, axes = plt.subplots(
            1, 3, figsize=(3.0 * SQUARE_FIGSIZE[0], SQUARE_FIGSIZE[1]),
        )
        _plot_panel(axes[0], curves_symm, _SYMM_NS)
        _plot_panel(axes[1], curves_asym, _ASYM_NS)
        _plot_stress_panel(axes[2], curves_stress)
        axes[0].set_title("EA-symm: symmetric all-informed")
        axes[1].set_title("EA-asym: asymmetric all-informed")
        axes[2].set_title(
            fr"EA-stress: asymmetric $N = {_STRESS_N}$ regularity sweep"
        )
        fig.tight_layout()
    run.save_figure("EA_allinformed_combined", fig, close=False)
    export_figure(fig, "fig_all_informed_allinformed_mse_paths", strip_title=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    *,
    horizon: int = 50_000,
    n_seeds_symm: int = 100,
    n_seeds_asym: int = 100,
    base_seed: int = 61,
    quick: bool = False,
) -> None:
    horizon_symm, n_seeds_symm = C.quick_overrides(
        quick, default_T=horizon, default_S=n_seeds_symm,
    )
    horizon_asym, n_seeds_asym = C.quick_overrides(
        quick, default_T=horizon, default_S=n_seeds_asym,
    )

    # Representative cfg for run_directory bookkeeping.
    rep_N = _SYMM_NS[0]
    rep_d = _make_symm_market(rep_N)
    rep_sched = ExplorationSchedule(kind="polynomial", c=_C_GRID[0], eta=_ETA)
    cfg = C.base_config(
        name="exp_all_informed_stress",
        market=rep_d,
        sellers=_make_all_informed(rep_N, rep_sched),
        horizon=horizon_symm,
        n_seeds=n_seeds_symm,
        base_seed=base_seed,
        log_every=max(1, horizon_symm // 1000),
        oblivious_box=C.tight_oblivious_box(rep_d, expand=0.5),
        informed_box=InformedProjectionBox.from_demand(rep_d),
    )

    with run_directory("exp_all_informed_stress", cfg) as run:
        run.logger.info(
            "EA sweep: SYMM N=%s, ASYM N=%s, c_grid=%s, eta=%.2f, T_symm=%d, T_asym=%d, "
            "S_symm=%d, S_asym=%d",
            list(_SYMM_NS), list(_ASYM_NS), list(_C_GRID), _ETA,
            horizon_symm, horizon_asym, n_seeds_symm, n_seeds_asym,
        )
        run.logger.info(
            "EA-stress: N=%d asymmetric (beta=%s, alpha=%s), gamma_12 in %s, c=%.2f, eta=%.2f",
            _STRESS_N, list(_STRESS_BETA), list(_STRESS_ALPHA),
            list(_STRESS_GAMMAS), _STRESS_C, _ETA,
        )

        rows: list[dict] = []
        curves_symm: list[tuple[int, float, np.ndarray, np.ndarray]] = []
        curves_asym: list[tuple[int, float, np.ndarray, np.ndarray]] = []
        curves_stress: list[tuple[float, np.ndarray, np.ndarray]] = []

        # ---- EA-symm ------------------------------------------------------
        for N in _SYMM_NS:
            d = _make_symm_market(N)
            p_NE = market.nash_prices(d)
            run.logger.info(
                "EA-symm N=%d, gamma=%.3f, p_NE=%s",
                N, _SYMM_GAMMA[N], np.round(p_NE, 3).tolist(),
            )
            for c_lead in _C_GRID:
                n_grid, mse_curve, meta = _run_cell(
                    panel="symm",
                    N=N,
                    c_lead=c_lead,
                    d=d,
                    horizon=horizon_symm,
                    n_seeds=n_seeds_symm,
                    base_seed=base_seed,
                    log_every=cfg.log_every,
                    logger=run.logger,
                )
                curves_symm.append((int(N), float(c_lead), n_grid, mse_curve))
                rows.append(meta)
                run.log_event("EA_symm_cell", **meta)

        # ---- EA-asym ------------------------------------------------------
        for N in _ASYM_NS:
            try:
                d = _make_asym_market(N, base_seed=base_seed)
            except ValueError as exc:
                run.logger.warning("skipping asymmetric N=%d: %s", N, exc)
                continue
            p_NE = market.nash_prices(d)
            run.logger.info(
                "EA-asym N=%d: alpha=%s beta=%s p_NE=%s",
                N,
                np.round(d.alpha_arr, 3).tolist(),
                np.round(d.beta_arr, 3).tolist(),
                np.round(p_NE, 3).tolist(),
            )
            for c_lead in _C_GRID:
                n_grid, mse_curve, meta = _run_cell(
                    panel="asym",
                    N=N,
                    c_lead=c_lead,
                    d=d,
                    horizon=horizon_asym,
                    n_seeds=n_seeds_asym,
                    base_seed=base_seed,
                    log_every=cfg.log_every,
                    logger=run.logger,
                )
                curves_asym.append((int(N), float(c_lead), n_grid, mse_curve))
                rows.append(meta)
                run.log_event("EA_asym_cell", **meta)

        # ---- EA-stress ----------------------------------------------------
        for gamma_12 in _STRESS_GAMMAS:
            d = _make_stress_market(gamma_12)
            p_NE = market.nash_prices(d)
            run.logger.info(
                "EA-stress gamma_12=%.3f: p_NE=%s",
                gamma_12, np.round(p_NE, 3).tolist(),
            )
            n_grid, mse_curve, meta = _run_cell(
                panel="stress",
                N=_STRESS_N,
                c_lead=_STRESS_C,
                d=d,
                horizon=horizon_symm,
                n_seeds=n_seeds_symm,
                base_seed=base_seed,
                log_every=cfg.log_every,
                logger=run.logger,
            )
            meta = {**meta, "gamma_12": float(gamma_12)}
            curves_stress.append((meta["lambda_max_B_plus_Bt"], n_grid, mse_curve))
            rows.append(meta)
            run.log_event("EA_stress_cell", **meta)

        # ---- Persist tables + figures ------------------------------------
        df = pd.DataFrame(rows)
        run.save_summary("EA_allinformed_cells", df)
        export_table(
            df, "table_all_informed_allinformed",
            caption=(
                "Convergence in all-informed "
                "markets at exploration rate $\\eta_i = 1/2$. Panels "
                "``symm'' and ``asym'' satisfy the regularity condition "
                "$\\lambda_{\\max}(B + B^\\top) < 1$; panel ``stress'' "
                "sweeps the cross-price coefficient $\\gamma_{12} = "
                "\\gamma_{21}$ in an asymmetric $N = 3$ market with one "
                "``fragile'' seller ($\\beta_3 = -0.05$), chosen so that "
                "every cell violates the regularity condition "
                "($\\lambda_{\\max}(B + B^\\top) > 1$) while the "
                "underlying ODE remains stable ($\\rho(B) < 1$). For "
                "each cell we report $\\lambda_{\\max}(B + B^\\top)$, the "
                "seed-averaged final realized-price MSE, and the "
                "empirical log--log tail slope (predicted: $-1/2$ when "
                "the regularity condition holds; recorded but not predicted "
                "for the stress cells)."
            ),
            floatfmt=".3g",
        )

        _save_single_panel_figure(
            curves_symm, _SYMM_NS, name="EA_symm", run=run,
        )
        _save_single_panel_figure(
            curves_asym, _ASYM_NS, name="EA_asym", run=run,
        )
        _save_stress_figure(curves_stress, run=run)
        _save_combined_figure(
            curves_symm, curves_asym, curves_stress, run=run,
        )

        run.logger.info("exp_all_informed_stress finished")


if __name__ == "__main__":
    main()
