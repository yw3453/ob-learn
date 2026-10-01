"""Multi-seller asymmetric mixed markets (log-log learning regret).

Asymmetric mixed markets at ``N in {3, 5, 10}`` with one oblivious seller and
``N-1`` informed sellers (running-mean forecast, ``eta = 0.25``); sweep the
oblivious dithering ``nu^2``. Aggregate cumulative *learning* regret on log-log
axes with a fitted sublinear power reference. Cached + parallel, ``T = 1e4``.
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
from ob_learn.config import ExplorationSchedule, InformedProjectionBox
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

_EXP_ID = "exp_mixed_multiseller"
_NS = (3, 5, 10)
_NU2_GRID = (0.05, 0.10, 0.20)
_ETA_INFORMED = 0.25
_C_INFORMED = 0.10


def _worker(spec: dict) -> dict:
    N = int(spec["N"])
    nu2 = float(spec["nu2"])
    d = C.asymmetric_market(N, base_seed=int(spec["base_seed"]))
    n_ob, n_in = 1, N - 1
    ob_idx = list(range(n_ob))
    in_idx = list(range(n_ob, N))
    box_ob = C.tight_oblivious_box(d, expand=0.5)
    box_in = InformedProjectionBox.from_demand(d)
    beta_abs_min = 0.5 * float(np.min(np.abs(d.beta_arr)))
    simple = market.simplified_smallgain(d, ob_idx, in_idx, box_ob, box_in,
                                          nu_squared=nu2, beta_abs_min=beta_abs_min)
    sg = market.master_theorem_smallgain(d, ob_idx, in_idx, box_ob, box_in,
                                         nu_squared=nu2, beta_abs_min=beta_abs_min)
    cfg = C.base_config(
        name=f"M2_N{N}_nu2_{nu2:.3f}", market=d,
        sellers=C.make_mixed_sellers(
            n_ob=n_ob, n_in=n_in,
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
        "params": {"N": N, "nu_squared": nu2,
                   "learning_regret_final": float(reg_mean[-1]), "regret_slope_tail": slope,
                   "pp_slope_tail": pp_slope,
                   "L": float(simple["L"]), "bar_mu": float(simple["bar_mu"]),
                   "cond_ii_value": float(simple["cond_ii_value"]),
                   "cond_ii_holds": bool(simple["cond_ii_holds"]),
                   "cond_iv_holds": bool(simple["cond_iv_holds"]),
                   "margin": float(sg["margin"])},
        "n": n_grid, "regret": reg_mean,
        "regret_p25": np.percentile(reg, 25, axis=1),
        "regret_p75": np.percentile(reg, 75, axis=1),
        "pp": pp_mean,
        "pp_p25": np.percentile(pp, 25, axis=1),
        "pp_p75": np.percentile(pp, 75, axis=1),
    }


def _specs(*, horizon, n_seeds, base_seed, log_every, Ns) -> list[dict]:
    return [{"N": N, "nu2": nu2, "horizon": horizon, "n_seeds": n_seeds,
             "base_seed": base_seed, "log_every": log_every}
            for N in Ns for nu2 in _NU2_GRID]


def _plot(cells: list[dict], Ns, *, cumulative: bool = False) -> plt.Figure:
    with report_style():
        fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
        cmap = _cmaps.get_cmap("viridis")
        color_for_N = {N_: cmap(0.10 + 0.80 * idx / max(len(Ns) - 1, 1))
                       for idx, N_ in enumerate(Ns)}
        ls_for_nu = {float(nu2): ls for nu2, ls in zip(_NU2_GRID, ("-", "--", ":"), strict=False)}
        key = "regret" if cumulative else "pp"
        slope_key = "regret_slope_tail" if cumulative else "pp_slope_tail"
        for cell in cells:
            N_ = int(cell["params"]["N"])
            nu2 = float(cell["params"]["nu_squared"])
            ax.plot(cell["n"], np.maximum(cell[key], 1e-12),
                    color=color_for_N.get(N_, "tab:blue"), linestyle=ls_for_nu.get(nu2, "-"),
                    lw=1.4, alpha=0.90)
        ref_slope = float(np.nanmedian([float(c["params"][slope_key]) for c in cells]))
        if cumulative:
            regret_reference(ax, [(c["n"], c["regret"]) for c in cells], kind="power", slope=ref_slope)
            style_regret_axes(ax)
            loc = "upper left"
        else:
            perperiod_reference(ax, [(c["n"], c["pp"]) for c in cells], slope=ref_slope)
            loglog_regret_axes(ax, ylabel=r"per-period learning regret")
            loc = "lower left"
        handles_N = [Line2D([0], [0], color=color_for_N[N_], lw=1.6, label=f"N = {N_}") for N_ in Ns]
        handles_nu = [Line2D([0], [0], color="black", lw=1.4, linestyle=ls_for_nu[float(nu2)],
                             label=fr"$\nu^2 = {nu2:.2f}$") for nu2 in _NU2_GRID]
        ref_handle = [Line2D([0], [0], color="0.30", lw=1.1, linestyle=(0, (4, 3)),
                             label=fr"$\propto n^{{{ref_slope:.2f}}}$")]
        ax.legend(handles=handles_N + handles_nu + ref_handle, loc=loc,
                  fontsize=10, framealpha=0.92, ncol=2)
        fig.tight_layout()
    return fig


def main(
    *,
    horizon: int = 10_000,
    n_seeds: int = 100,
    base_seed: int = 53,
    quick: bool = False,
    replot: bool = False,
    overwrite: bool = False,
    cumulative: bool = False,
    plot_max_n: float | None = None,
    max_workers: int = parallel.DEFAULT_WORKERS,
) -> None:
    horizon, n_seeds = C.quick_overrides(quick, default_T=horizon, default_S=n_seeds)
    log_every = max(1, horizon // 1000)
    Ns = _NS if not quick else (3, 5)

    rep_d = C.asymmetric_market(Ns[0], base_seed=base_seed)
    cfg = C.base_config(
        name=_EXP_ID, market=rep_d,
        sellers=C.make_mixed_sellers(
            n_ob=1, n_in=Ns[0] - 1,
            oblivious_schedule=ExplorationSchedule(kind="constant", nu=float(np.sqrt(_NU2_GRID[0]))),
            informed_schedule=ExplorationSchedule(kind="polynomial", c=_C_INFORMED, eta=_ETA_INFORMED),
            forecast_rule="mean_price"),
        horizon=horizon, n_seeds=n_seeds, base_seed=base_seed, log_every=log_every,
        oblivious_box=C.tight_oblivious_box(rep_d, expand=0.5),
        informed_box=InformedProjectionBox.from_demand(rep_d),
    )

    with run_directory(_EXP_ID, cfg) as run:
        if replot and figdata.figdata_exists(_EXP_ID) and not overwrite:
            run.logger.info("replot: loading cached figdata for %s", _EXP_ID)
            cells = figdata.load_figdata(_EXP_ID)
        else:
            run.logger.info("asymmetric mixed N=%s nu2=%s T=%d S=%d (workers=%d)",
                            list(Ns), list(_NU2_GRID), horizon, n_seeds, max_workers)
            specs = _specs(horizon=horizon, n_seeds=n_seeds, base_seed=base_seed,
                           log_every=log_every, Ns=Ns)
            cells = parallel.map_cells(_worker, specs, max_workers=max_workers, logger=run.logger)
            figdata.save_figdata(_EXP_ID, cells)

        df = pd.DataFrame([dict(c["params"]) for c in cells])
        run.save_summary("M2_multiseller_cells", df)
        export_table(
            df, "table_mixed_multiseller_multiseller",
            caption=(
                "Asymmetric mixed markets at $N \\in \\{3, 5, 10\\}$ with one "
                "oblivious seller and $N-1$ informed sellers. Each cell reports "
                "the final cumulative learning regret and the informed-side "
                "$L + 2\\bar\\mu$ (condition (ii); it exceeds $2$ in every cell "
                "so (ii) and (iv) fail), yet the learning regret is sublinear."
            ),
            floatfmt=".3g",
        )

        suffix = "_cumulative" if cumulative else ""
        fig = _plot(truncate_cells(cells, plot_max_n), Ns, cumulative=cumulative)
        run.save_figure(f"M2_regret_paths{suffix}", fig, close=False)
        export_figure(fig, f"fig_M2_multiseller_regret_paths{suffix}", strip_title=True)
        run.logger.info("%s finished", _EXP_ID)


if __name__ == "__main__":
    main()
