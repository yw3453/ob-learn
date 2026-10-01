"""Dominance-margin sweep at ``N = 5`` (all-oblivious).

Symmetric ``N = 5`` markets with cross-price coefficient ``gamma`` from
deep-interior to near the collusive-Hessian boundary, crossed with dithering
``nu^2``. For each ``(gamma, nu^2)`` cell we plot the aggregate cumulative
*learning* regret on **semilog-x** axes (log-scaled ``n``, linear regret), where
logarithmic regret appears as a straight line. Cached + parallel like the other
regret sweeps.
"""

from __future__ import annotations

import _common as C  # type: ignore[import-not-found]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps as _cmaps
from matplotlib.lines import Line2D

from ob_learn import analysis, benchmarks, figdata, parallel
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

_EXP_ID = "exp_dominance_margin"
_N = 5
_GAMMA_GRID = (0.05, 0.10, 0.15)
_NU2_GRID = (0.05, 0.10, 0.20)


def _make_market(gamma: float) -> DemandParams:
    return DemandParams.symmetric(
        N=_N, alpha=2.5, beta=-1.0, gamma=float(gamma), l=0.5, u=2.5, noise_std=0.2,
    )


def _worker(spec: dict) -> dict:
    gamma = float(spec["gamma"])
    nu2 = float(spec["nu2"])
    d = _make_market(gamma)
    box = C.tight_oblivious_box(d, expand=0.5)
    sched = ExplorationSchedule(kind="constant", nu=float(np.sqrt(nu2)))
    cfg = C.base_config(
        name=f"N{_N}_g{gamma:.4f}_nu2_{nu2:.4f}",
        market=d, sellers=C.make_oblivious_sellers(_N, sched),
        horizon=int(spec["horizon"]), n_seeds=int(spec["n_seeds"]),
        base_seed=int(spec["base_seed"]), log_every=int(spec["log_every"]),
        oblivious_box=box,
    )
    res = run_simulation(cfg, progress=False, compute_moments=False)
    n_grid = (res.log_steps + 1).astype(np.float64)
    reg = benchmarks.cumulative_learning_regret(res, d).sum(axis=1)   # (T_log, S)
    pp = benchmarks.per_period_learning_regret(res, d).sum(axis=1)    # (T_log, S)
    reg_mean = reg.mean(axis=1)
    pp_mean = pp.mean(axis=1)
    delta = float(-2.0 * d.beta[0] - 2.0 * (_N - 1) * gamma)
    return {
        "params": {"N": _N, "gamma": gamma, "nu_squared": nu2, "delta_N": delta,
                   "learning_regret_final": float(reg_mean[-1]),
                   "pp_slope_tail": float(analysis.fit_loglog_slope(n_grid, pp_mean)["slope"])},
        "n": n_grid,
        "regret": reg_mean,
        "regret_p25": np.percentile(reg, 25, axis=1),
        "regret_p75": np.percentile(reg, 75, axis=1),
        "pp": pp_mean,
        "pp_p25": np.percentile(pp, 25, axis=1),
        "pp_p75": np.percentile(pp, 75, axis=1),
    }


def _specs(*, horizon, n_seeds, base_seed, log_every) -> list[dict]:
    return [
        {"gamma": g, "nu2": nu2, "horizon": horizon, "n_seeds": n_seeds,
         "base_seed": base_seed, "log_every": log_every}
        for g in _GAMMA_GRID for nu2 in _NU2_GRID
    ]


def _plot(cells: list[dict], *, cumulative: bool = False) -> plt.Figure:
    with report_style():
        fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
        cmap = _cmaps.get_cmap("viridis")
        color_for_g = {float(g): cmap(0.1 + 0.8 * idx / max(len(_GAMMA_GRID) - 1, 1))
                       for idx, g in enumerate(_GAMMA_GRID)}
        ls_for_nu = {float(nu2): ls for nu2, ls in
                     zip(_NU2_GRID, ("-", "--", ":"), strict=False)}
        key = "regret" if cumulative else "pp"
        for cell in cells:
            g = float(cell["params"]["gamma"])
            nu2 = float(cell["params"]["nu_squared"])
            ax.plot(cell["n"], cell[key], color=color_for_g[g],
                    linestyle=ls_for_nu.get(nu2, "-"), lw=1.4, alpha=0.90)
        if cumulative:
            regret_reference(ax, [(c["n"], c["regret"]) for c in cells], kind="log")
            style_regret_axes(ax)
            ref_lab = r"$\propto \log n$"
        else:
            perperiod_reference(ax, [(c["n"], c["pp"]) for c in cells], slope=-1.0)
            loglog_regret_axes(ax, ylabel=r"per-period learning regret")
            ref_lab = r"$\propto n^{-1}$"
        handles_g = [Line2D([0], [0], color=color_for_g[float(g)], lw=1.6,
                            label=fr"$\gamma = {g:.3f}$") for g in _GAMMA_GRID]
        handles_nu = [Line2D([0], [0], color="black", lw=1.4, linestyle=ls_for_nu[float(nu2)],
                             label=fr"$\nu^2 = {nu2:.2f}$") for nu2 in _NU2_GRID]
        ref_handle = [Line2D([0], [0], color="0.30", lw=1.1, linestyle=(0, (4, 3)), label=ref_lab)]
        loc = "upper left" if cumulative else "lower left"
        ax.legend(handles=handles_g + handles_nu + ref_handle, loc=loc,
                  fontsize=9, framealpha=0.92, ncol=2)
        fig.tight_layout()
    return fig


def main(
    *,
    horizon: int = 10_000,
    n_seeds: int = 60,
    base_seed: int = 313,
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
        sellers=C.make_oblivious_sellers(_N, ExplorationSchedule(kind="constant", nu=0.3)),
        horizon=horizon, n_seeds=n_seeds, base_seed=base_seed,
        log_every=log_every, oblivious_box=C.tight_oblivious_box(rep_d, expand=0.5),
    )

    with run_directory(_EXP_ID, cfg) as run:
        if replot and figdata.figdata_exists(_EXP_ID) and not overwrite:
            run.logger.info("replot: loading cached figdata for %s", _EXP_ID)
            cells = figdata.load_figdata(_EXP_ID)
        else:
            run.logger.info("sweeping gamma=%s nu^2=%s, T=%d, S=%d (workers=%d)",
                            list(_GAMMA_GRID), list(_NU2_GRID), horizon, n_seeds, max_workers)
            specs = _specs(horizon=horizon, n_seeds=n_seeds, base_seed=base_seed,
                           log_every=log_every)
            cells = parallel.map_cells(_worker, specs, max_workers=max_workers,
                                       logger=run.logger)
            figdata.save_figdata(_EXP_ID, cells)

        df = pd.DataFrame([dict(c["params"]) for c in cells])
        run.save_summary("dominance_margin_summary", df)
        export_table(df, "table_dominance_margin_dominance_margin", caption=(
            f"Final cumulative learning regret in symmetric $N = {_N}$ markets "
            "across a $(\\gamma, \\nu^2)$ grid. The companion plot "
            "\\texttt{fig\\_3b\\_dominance\\_regret\\_paths.pdf} shows the "
            "cumulative learning-regret trajectory for every cell."
        ), floatfmt=".3g")

        suffix = "_cumulative" if cumulative else ""
        fig = _plot(truncate_cells(cells, plot_max_n), cumulative=cumulative)
        run.save_figure(f"dominance_regret_paths{suffix}", fig, close=False)
        export_figure(fig, f"fig_3b_dominance_regret_paths{suffix}", strip_title=True)
        run.logger.info("%s finished", _EXP_ID)


if __name__ == "__main__":
    main()
