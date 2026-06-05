"""Mixed-market stress test: denominator condition holds, small-gain fails.

Companion to ``exp_mixed_small_gain.py``: that experiment violates both the
denominator condition ``bar_c_diag - bar_D > 0`` and the small-gain
inequality jointly (the box-Lipschitz ``L_phi^{ob}`` alone exceeds 2 under
the default primitives, driving ``bar_c_diag`` strongly negative). The
present experiment chooses demand primitives and projection boxes that keep
both Lipschitz envelopes below 2, so ``bar_c_diag - bar_D > 0`` (the
denominator condition holds), while the oblivious-side cross-coupling and
the running-mean spillback inflate the small-gain bracket above ``C_M``, so
the small-gain inequality still fails.

Symmetric ``N = 5`` mixed market with ``|I^{ob}| = 4`` and ``|I^{in}| = 1``.
We use ``alpha = 3.0, beta = -2.0, u = 2.0, l = 0.001`` so that ``p^NE
approx 0.79`` sits well inside the demand box ``[l, u] = [0.001, 2.0]``
(uniform-dither clip rate is 0 at every ``nu^2 <= 0.20``), a tight
oblivious projection box (``expand = 0.3``), and a tight *informed*
projection box that clamps ``|beta_j|`` to ``0.95 |beta|`` from below so
``L_phi^{in, theta}`` stays moderate and ``bar_c_diag - bar_D`` lifts
clear of zero. We sweep ``gamma in {0.05, 0.08, 0.10, 0.12}`` (strictly
inside own-price-dominance threshold ``|beta|/(N-1) = 0.5``) and
oblivious dithering ``nu^2 in {0.05, 0.10, 0.20}``. Informed seller uses
the running-mean forecast with ``eta(j) = 0.25``. Horizon ``T = 5e4``,
``S = 100`` seeds.

For every cell we verify ``K_2 > 0`` (denominator condition holds) and
``margin < 0`` (small-gain inequality fails), then we plot the seed-averaged
price MSE on log-log axes; convergence in every cell is the qualitative
acceptance criterion. Under these primitives every cell has
``K_2 in [0.73, 0.82]`` (denominator condition holds with comfortable
margin) and the small-gain margin sits in ``[-2.27, -0.76]`` (small-gain
inequality fails unambiguously), and a clip-free realized-price dynamics
yields clean log-log decay without a Jensen-bias plateau.
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
)
from src.logging_utils import run_directory
from src.plotting import SQUARE_FIGSIZE, report_style, square_box
from src.simulator import run_simulation

_N = 5
_N_OB = 4
_N_IN = 1
_OB_IDX = list(range(_N_OB))
_IN_IDX = list(range(_N_OB, _N_OB + _N_IN))
_ETA_INFORMED = 0.25
_C_INFORMED = 0.10  # nu_n^2 = c (n+1)^{-eta}

_ALPHA = 3.0
_BETA = -2.0
_L = 0.001
_U = 2.0
_NOISE_STD = 0.2

_GAMMA_GRID = (0.05, 0.08, 0.10, 0.12)
# All strictly inside diagonal-dominance gamma < |beta|/(N-1) = 0.5.
_NU2_GRID = (0.05, 0.10, 0.20)

_OB_EXPAND = 0.3
# Tight informed projection box: clamp |beta_j| within 5% of truth so
# L_phi^{in,theta} stays moderate at u = 2 (otherwise bar_c_diag goes
# negative and the denominator condition breaks).
_BETA_ABS_MIN_FRAC = 0.95


def _make_market(gamma: float) -> DemandParams:
    return DemandParams.symmetric(
        N=_N, alpha=_ALPHA, beta=_BETA, gamma=gamma,
        l=_L, u=_U, noise_std=_NOISE_STD,
    )


def main(
    *,
    horizon: int = 50_000,
    n_seeds: int = 100,
    base_seed: int = 67,
    quick: bool = False,
) -> None:
    horizon, n_seeds = C.quick_overrides(quick, default_T=horizon, default_S=n_seeds)

    rep_d = _make_market(_GAMMA_GRID[0])
    rep_box_ob = C.tight_oblivious_box(rep_d, expand=_OB_EXPAND)
    rep_box_in = C.tight_informed_box(rep_d, beta_abs_min_frac=_BETA_ABS_MIN_FRAC)
    rep_sched_ob = ExplorationSchedule(kind="constant", nu=float(np.sqrt(_NU2_GRID[0])))
    rep_sched_in = ExplorationSchedule(kind="polynomial", c=_C_INFORMED, eta=_ETA_INFORMED)

    cfg = C.base_config(
        name="exp_mixed_small_gain_iv_holds",
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

    with run_directory("exp_mixed_small_gain_iv_holds", cfg) as run:
        run.logger.info(
            "N=%d, n_ob=%d, n_in=%d, alpha=%.2f, beta=%.2f, u=%.2f, l=%.3f, "
            "gamma_grid=%s, nu2_grid=%s, eta_informed=%.2f, ob_expand=%.2f, "
            "beta_abs_min_frac=%.2f",
            _N, _N_OB, _N_IN, _ALPHA, _BETA, _U, _L,
            list(_GAMMA_GRID), list(_NU2_GRID), _ETA_INFORMED, _OB_EXPAND,
            _BETA_ABS_MIN_FRAC,
        )

        rows: list[dict] = []
        curves: list[tuple[float, float, np.ndarray, np.ndarray]] = []

        for gamma in _GAMMA_GRID:
            d = _make_market(gamma)
            box_ob = C.tight_oblivious_box(d, expand=_OB_EXPAND)
            box_in = C.tight_informed_box(d, beta_abs_min_frac=_BETA_ABS_MIN_FRAC)
            p_NE = market.nash_prices(d)
            run.logger.info(
                "gamma=%.3f, p_NE=%s", gamma, np.round(p_NE, 3).tolist()
            )
            beta_abs_min = _BETA_ABS_MIN_FRAC * float(np.min(np.abs(d.beta_arr)))
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
                    name=f"M1b_gamma{gamma:.2f}_nu2{nu2:.2f}",
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
                    "running gamma=%.3f nu^2=%.3f: K_2=%+.4f, margin=%+.4f, "
                    "iv_holds=%s, ii_fails=%s",
                    gamma, nu2, smallgain["K_2"], smallgain["margin"],
                    smallgain["K_2"] > 0, smallgain["margin"] < 0,
                )
                res = run_simulation(sub_cfg, logger=run.logger)
                mse_p = analysis.mse_price(res)
                mse_curve = mse_p.mean(axis=1)
                n_grid = res.log_steps + 1
                curves.append((
                    float(gamma), float(nu2),
                    n_grid.astype(np.float64), mse_curve,
                ))
                row = {
                    "gamma": float(gamma),
                    "mse_price_final": float(mse_curve[-1]),
                    **smallgain,
                }
                rows.append(row)
                run.log_event("M1b_cell", **row)

        df = pd.DataFrame(rows)
        run.save_summary("M1b_iv_holds_cells", df)
        export_table(
            df, "table_mixed_small_gain_iv_holds_iv_holds",
            caption=(
                "Mixed-market stress test where the denominator condition "
                "holds but the small-gain inequality fails. Symmetric $N=5$ "
                "market with $|\\mathcal I^{ob}|=4, |\\mathcal I^{in}|=1$, "
                f"$\\alpha={_ALPHA}, \\beta={_BETA}, u={_U}$, with a tight "
                "informed projection box clamping $|\\beta_j| \\geq "
                f"{_BETA_ABS_MIN_FRAC}|\\beta|$. For each $(\\gamma, \\nu^2)$ "
                "cell we report $K_2 = \\bar c_{\\mathrm{diag}} - \\bar D > 0$ "
                "(denominator condition holds), the small-gain margin "
                "$C_M - C_x[2\\bar\\gamma^{ob}L_\\phi^{ob} + \\bar\\Delta + "
                "\\bar\\Theta + L_\\phi^{ob}\\bar\\Psi/(\\bar c_{\\mathrm{diag}} - "
                "\\bar D)] < 0$ (small-gain inequality fails), and the "
                "seed-averaged final price MSE."
            ),
            floatfmt=".3g",
        )

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
            run.save_figure("M1b_mse_paths", fig, close=False)
            export_figure(fig, "fig_mixed_small_gain_iv_holds_iv_holds_mse_paths", strip_title=True)

        run.logger.info("exp_mixed_small_gain_iv_holds finished")


if __name__ == "__main__":
    main()
