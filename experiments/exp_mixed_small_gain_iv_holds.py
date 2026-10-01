"""Mixed-market stress test: condition (ii) holds, (iv) fails (log-log regret).

Symmetric ``N = 5`` mixed market (``|I^{ob}| = 4``, ``|I^{in}| = 1``) with
primitives (``alpha=3, beta=-2, u=2``) and a tight informed projection box chosen
so that the informed-side condition (ii) holds while the oblivious-side (iv) fails.
Sweep ``gamma`` and ``nu^2``; plot aggregate cumulative *learning* regret on
log-log axes with a fitted sublinear power reference. Cached + parallel,
``T = 1e4``.
"""

from __future__ import annotations

import _common as C  # type: ignore[import-not-found]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps as _cmaps
from matplotlib.lines import Line2D

from ob_learn import analysis, benchmarks, figdata, market, parallel
from ob_learn.artifact_export import export_figure, export_table
from ob_learn.config import DemandParams, ExplorationSchedule
from ob_learn.logging_utils import run_directory
from ob_learn.plotting import (
    SQUARE_FIGSIZE,
    loglog_regret_axes,
    perperiod_reference,
    regret_reference,
    report_style,
    style_regret_axes,
    truncate_cells,
)
from ob_learn.simulator import run_simulation

_EXP_ID = "exp_mixed_small_gain_iv_holds"
_N = 5
_N_OB = 4
_N_IN = 1
_OB_IDX = list(range(_N_OB))
_IN_IDX = list(range(_N_OB, _N_OB + _N_IN))
_ETA_INFORMED = 0.25
_C_INFORMED = 0.10
_ALPHA = 3.0
_BETA = -2.0
_L = 0.001
_U = 2.0
_NOISE_STD = 0.2
_GAMMA_GRID = (0.05, 0.08, 0.10, 0.12)
_NU2_GRID = (0.05, 0.10, 0.20)
_OB_EXPAND = 0.3
_BETA_ABS_MIN_FRAC = 0.95


def _make_market(gamma: float) -> DemandParams:
    return DemandParams.symmetric(
        N=_N, alpha=_ALPHA, beta=_BETA, gamma=gamma, l=_L, u=_U, noise_std=_NOISE_STD,
    )


def _worker(spec: dict) -> dict:
    gamma = float(spec["gamma"])
    nu2 = float(spec["nu2"])
    d = _make_market(gamma)
    box_ob = C.tight_oblivious_box(d, expand=_OB_EXPAND)
    box_in = C.tight_informed_box(d, beta_abs_min_frac=_BETA_ABS_MIN_FRAC)
    beta_abs_min = _BETA_ABS_MIN_FRAC * float(np.min(np.abs(d.beta_arr)))
    simple = market.simplified_smallgain(d, _OB_IDX, _IN_IDX, box_ob, box_in,
                                          nu_squared=nu2, beta_abs_min=beta_abs_min)
    sg = market.master_theorem_smallgain(d, _OB_IDX, _IN_IDX, box_ob, box_in,
                                         nu_squared=nu2, beta_abs_min=beta_abs_min)
    cfg = C.base_config(
        name=f"M1b_gamma{gamma:.2f}_nu2{nu2:.2f}", market=d,
        sellers=C.make_mixed_sellers(
            n_ob=_N_OB, n_in=_N_IN,
            oblivious_schedule=ExplorationSchedule(kind="constant", nu=float(np.sqrt(nu2))),
            informed_schedule=ExplorationSchedule(kind="polynomial", c=_C_INFORMED, eta=_ETA_INFORMED),
            forecast_rule="mean_price"),
        horizon=int(spec["horizon"]), n_seeds=int(spec["n_seeds"]),
        base_seed=int(spec["base_seed"]), log_every=int(spec["log_every"]),
        oblivious_box=box_ob, informed_box=box_in,
    )
    res = run_simulation(cfg, progress=False, compute_moments=False)
    n_grid = (res.log_steps + 1).astype(np.float64)
    reg = benchmarks.cumulative_learning_regret(res, d).sum(axis=1)
    pp = benchmarks.per_period_learning_regret(res, d).sum(axis=1)
    reg_mean = reg.mean(axis=1)
    pp_mean = pp.mean(axis=1)
    slope = float(analysis.fit_loglog_slope(n_grid, np.maximum(reg_mean, 1e-12))["slope"])
    pp_slope = float(analysis.fit_loglog_slope(n_grid, np.maximum(pp_mean, 1e-12))["slope"])
    return {
        "params": {"gamma": gamma, "nu_squared": nu2,
                   "learning_regret_final": float(reg_mean[-1]), "regret_slope_tail": slope,
                   "pp_slope_tail": pp_slope,
                   "L": float(simple["L"]), "bar_mu": float(simple["bar_mu"]),
                   "cond_ii_value": float(simple["cond_ii_value"]),
                   "cond_ii_holds": bool(simple["cond_ii_holds"]),
                   "cond_iv_holds": bool(simple["cond_iv_holds"]),
                   "K_2": float(sg["K_2"]), "margin": float(sg["margin"])},
        "n": n_grid, "regret": reg_mean,
        "regret_p25": np.percentile(reg, 25, axis=1),
        "regret_p75": np.percentile(reg, 75, axis=1),
        "pp": pp_mean,
        "pp_p25": np.percentile(pp, 25, axis=1),
        "pp_p75": np.percentile(pp, 75, axis=1),
    }


def _specs(*, horizon, n_seeds, base_seed, log_every) -> list[dict]:
    return [{"gamma": g, "nu2": nu2, "horizon": horizon, "n_seeds": n_seeds,
             "base_seed": base_seed, "log_every": log_every}
            for g in _GAMMA_GRID for nu2 in _NU2_GRID]


def _plot(cells: list[dict], *, cumulative: bool = False) -> plt.Figure:
    with report_style():
        fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
        cmap = _cmaps.get_cmap("viridis")
        color_for_gamma = {float(g): cmap(0.10 + 0.80 * idx / max(len(_GAMMA_GRID) - 1, 1))
                           for idx, g in enumerate(_GAMMA_GRID)}
        ls_for_nu = {float(nu2): ls for nu2, ls in zip(_NU2_GRID, ("-", "--", ":"), strict=False)}
        key = "regret" if cumulative else "pp"
        slope_key = "regret_slope_tail" if cumulative else "pp_slope_tail"
        for cell in cells:
            g = float(cell["params"]["gamma"])
            nu2 = float(cell["params"]["nu_squared"])
            ax.plot(cell["n"], np.maximum(cell[key], 1e-12),
                    color=color_for_gamma[g], linestyle=ls_for_nu.get(nu2, "-"),
                    lw=1.3, alpha=0.90)
        ref_slope = float(np.nanmedian([float(c["params"][slope_key]) for c in cells]))
        if cumulative:
            regret_reference(ax, [(c["n"], c["regret"]) for c in cells], kind="power", slope=ref_slope)
            style_regret_axes(ax)
            loc = "upper left"
        else:
            perperiod_reference(ax, [(c["n"], c["pp"]) for c in cells], slope=ref_slope)
            loglog_regret_axes(ax, ylabel=r"per-period learning regret")
            loc = "lower left"
        handles_g = [Line2D([0], [0], color=color_for_gamma[float(g)], lw=1.6,
                            label=fr"$\gamma = {g:.2f}$") for g in _GAMMA_GRID]
        handles_nu = [Line2D([0], [0], color="black", lw=1.4, linestyle=ls_for_nu[float(nu2)],
                             label=fr"$\nu^2 = {nu2:.2f}$") for nu2 in _NU2_GRID]
        ref_handle = [Line2D([0], [0], color="0.30", lw=1.1, linestyle=(0, (4, 3)),
                             label=fr"$\propto n^{{{ref_slope:.2f}}}$")]
        ax.legend(handles=handles_g + handles_nu + ref_handle, loc=loc,
                  fontsize=10, framealpha=0.92, ncol=2)
        fig.tight_layout()
    return fig


def main(
    *,
    horizon: int = 10_000,
    n_seeds: int = 100,
    base_seed: int = 67,
    quick: bool = False,
    replot: bool = False,
    overwrite: bool = False,
    cumulative: bool = False,
    plot_max_n: float | None = None,
    max_workers: int = parallel.DEFAULT_WORKERS,
) -> None:
    horizon, n_seeds = C.quick_overrides(quick, default_T=horizon, default_S=n_seeds)
    log_every = max(1, horizon // 1000)

    rep_d = _make_market(_GAMMA_GRID[0])
    cfg = C.base_config(
        name=_EXP_ID, market=rep_d,
        sellers=C.make_mixed_sellers(
            n_ob=_N_OB, n_in=_N_IN,
            oblivious_schedule=ExplorationSchedule(kind="constant", nu=float(np.sqrt(_NU2_GRID[0]))),
            informed_schedule=ExplorationSchedule(kind="polynomial", c=_C_INFORMED, eta=_ETA_INFORMED),
            forecast_rule="mean_price"),
        horizon=horizon, n_seeds=n_seeds, base_seed=base_seed, log_every=log_every,
        oblivious_box=C.tight_oblivious_box(rep_d, expand=_OB_EXPAND),
        informed_box=C.tight_informed_box(rep_d, beta_abs_min_frac=_BETA_ABS_MIN_FRAC),
    )

    with run_directory(_EXP_ID, cfg) as run:
        if replot and figdata.figdata_exists(_EXP_ID) and not overwrite:
            run.logger.info("replot: loading cached figdata for %s", _EXP_ID)
            cells = figdata.load_figdata(_EXP_ID)
        else:
            run.logger.info("N=%d ob=%d in=%d gamma=%s nu2=%s T=%d S=%d (workers=%d)",
                            _N, _N_OB, _N_IN, list(_GAMMA_GRID), list(_NU2_GRID),
                            horizon, n_seeds, max_workers)
            specs = _specs(horizon=horizon, n_seeds=n_seeds, base_seed=base_seed, log_every=log_every)
            cells = parallel.map_cells(_worker, specs, max_workers=max_workers, logger=run.logger)
            figdata.save_figdata(_EXP_ID, cells)

        df = pd.DataFrame([dict(c["params"]) for c in cells])
        run.save_summary("M1b_iv_holds_cells", df)
        export_table(
            df, "table_mixed_small_gain_iv_holds_iv_holds",
            caption=(
                "Mixed-market stress test where the informed-side condition (ii) "
                "holds but the oblivious-side condition (iv) fails. Symmetric "
                "$N=5$ market with $|\\mathcal I^{ob}|=4, |\\mathcal I^{in}|=1$, "
                f"$\\alpha={_ALPHA}, \\beta={_BETA}, u={_U}$. Each cell reports "
                "$L + 2\\bar\\mu < 2$ (condition (ii) holds), condition (iv) "
                "(fails), the primitive $K_2$/margin, and the final cumulative "
                "learning regret (sublinear throughout)."
            ),
            floatfmt=".3g",
        )

        suffix = "_cumulative" if cumulative else ""
        fig = _plot(truncate_cells(cells, plot_max_n), cumulative=cumulative)
        run.save_figure(f"M1b_regret_paths{suffix}", fig, close=False)
        export_figure(fig, f"fig_M1b_iv_holds_regret_paths{suffix}", strip_title=True)
        run.logger.info("%s finished", _EXP_ID)


if __name__ == "__main__":
    main()
