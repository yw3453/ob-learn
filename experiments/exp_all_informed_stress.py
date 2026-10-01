"""All-informed market at the sqrt-T exploration rate.

With the running-mean forecast and exploration ``nu_{n,i}^2 = c_i n^{-1/2}``, the
aggregate cumulative *learning* regret grows like ``sqrt(n)`` (slope ``1/2`` on
log-log). Two panels: ``EA-symm`` (symmetric, ``N in {2,5}``) and ``EA-asym``
(asymmetric, ``N in {3,5}``), each sweeping the informed leading constant
``c in {0.05, 0.10, 0.20}`` at ``eta = 1/2``.

Log-log axes make the ``sqrt(n)`` rate a straight line at horizon
``T = 1e4``. Cached + parallel.
"""

from __future__ import annotations

import _common as C  # type: ignore[import-not-found]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps as _cmaps
from matplotlib.lines import Line2D

from ob_learn import benchmarks, figdata, market, parallel
from ob_learn.artifact_export import export_figure, export_table
from ob_learn.config import (
    DemandParams,
    ExplorationSchedule,
    InformedProjectionBox,
    SellerSpec,
)
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

_EXP_ID = "exp_all_informed_stress"
_SYMM_NS = (2, 5)
_ASYM_NS = (3, 5)
_C_GRID = (0.05, 0.10, 0.20)
_ETA = 0.5
_SYMM_GAMMA = {2: 0.4, 5: 0.10}


def _make_all_informed(N: int, schedule: ExplorationSchedule) -> list[SellerSpec]:
    return [SellerSpec(kind="informed", forecast_rule="mean_price", exploration=schedule)
            for _ in range(N)]


def _make_symm_market(N: int) -> DemandParams:
    return C.symmetric_market(N, gamma=_SYMM_GAMMA[N], noise_std=0.2)


def _make_asym_market(N: int, *, base_seed: int) -> DemandParams:
    return C.asymmetric_market(N, base_seed=base_seed)


def _corollary_regularity(d: DemandParams) -> float:
    Gamma = market.gamma_matrix(d)
    B = np.eye(d.N) - 0.5 * np.diag(1.0 / d.beta_arr) @ Gamma
    return float(np.linalg.eigvalsh(B + B.T).max())


def _worker(spec: dict) -> dict:
    panel = str(spec["panel"])
    N = int(spec["N"])
    c_lead = float(spec["c"])
    base_seed = int(spec["base_seed"])
    d = _make_symm_market(N) if panel == "symm" else _make_asym_market(N, base_seed=base_seed)
    sched = ExplorationSchedule(kind="polynomial", c=c_lead, eta=_ETA)
    cfg = C.base_config(
        name=f"EA_{panel}_N{N}_c{c_lead:.2f}", market=d,
        sellers=_make_all_informed(N, sched),
        horizon=int(spec["horizon"]), n_seeds=int(spec["n_seeds"]),
        base_seed=base_seed, log_every=int(spec["log_every"]),
        oblivious_box=C.tight_oblivious_box(d, expand=0.5),
        informed_box=InformedProjectionBox.from_demand(d),
    )
    res = run_simulation(cfg, progress=False, compute_moments=False)
    n_grid = (res.log_steps + 1).astype(np.float64)
    reg = benchmarks.cumulative_learning_regret(res, d).sum(axis=1)
    pp = benchmarks.per_period_learning_regret(res, d).sum(axis=1)
    reg_mean = reg.mean(axis=1)
    pp_mean = pp.mean(axis=1)
    from ob_learn import analysis
    slope = float(analysis.fit_loglog_slope(n_grid, reg_mean, tail_fraction=0.5)["slope"])
    pp_slope = float(analysis.fit_loglog_slope(n_grid, pp_mean, tail_fraction=0.5)["slope"])
    lam = _corollary_regularity(d)
    return {
        "params": {"panel": panel, "N": N, "c": c_lead, "eta": _ETA,
                   "lambda_max_B_plus_Bt": lam, "regularity_holds": bool(lam < 1.5),
                   "learning_regret_final": float(reg_mean[-1]), "regret_slope_tail": slope,
                   "pp_slope_tail": pp_slope},
        "n": n_grid,
        "regret": reg_mean,
        "regret_p25": np.percentile(reg, 25, axis=1),
        "regret_p75": np.percentile(reg, 75, axis=1),
        "pp": pp_mean,
        "pp_p25": np.percentile(pp, 25, axis=1),
        "pp_p75": np.percentile(pp, 75, axis=1),
    }


def _specs(*, horizon, n_seeds, base_seed, log_every, symm_ns, asym_ns) -> list[dict]:
    specs = []
    for N in symm_ns:
        for c in _C_GRID:
            specs.append({"panel": "symm", "N": N, "c": c, "horizon": horizon,
                          "n_seeds": n_seeds, "base_seed": base_seed, "log_every": log_every})
    for N in asym_ns:
        for c in _C_GRID:
            specs.append({"panel": "asym", "N": N, "c": c, "horizon": horizon,
                          "n_seeds": n_seeds, "base_seed": base_seed, "log_every": log_every})
    return specs


def _color_for_N(N_grid: tuple[int, ...]):
    cmap = _cmaps.get_cmap("viridis")
    return {int(N_): cmap(0.10 + 0.80 * idx / max(len(N_grid) - 1, 1))
            for idx, N_ in enumerate(N_grid)}


def _plot_panel(ax, cells: list[dict], N_grid: tuple[int, ...], *, cumulative: bool = False) -> None:
    color_for_N = _color_for_N(N_grid)
    ls_for_c = {float(c): ls for c, ls in zip(_C_GRID, ("-", "--", ":"), strict=False)}
    key = "regret" if cumulative else "pp"
    for cell in cells:
        N_ = int(cell["params"]["N"])
        c_ = float(cell["params"]["c"])
        ax.plot(cell["n"], np.maximum(cell[key], 1e-12),
                color=color_for_N.get(N_, "tab:blue"), linestyle=ls_for_c.get(c_, "-"),
                lw=1.4, alpha=0.90)
    if cumulative:
        regret_reference(ax, [(c["n"], c["regret"]) for c in cells], kind="sqrt")
        style_regret_axes(ax)
        ref_lab = r"$\propto \sqrt{n}$"
        loc = "upper left"
    else:
        perperiod_reference(ax, [(c["n"], c["pp"]) for c in cells], slope=-0.5)
        loglog_regret_axes(ax, ylabel=r"per-period learning regret")
        ref_lab = r"$\propto n^{-1/2}$"
        loc = "lower left"
    handles_N = [Line2D([0], [0], color=color_for_N[int(N_)], lw=1.6, label=f"N = {N_}")
                 for N_ in N_grid]
    handles_c = [Line2D([0], [0], color="black", lw=1.4, linestyle=ls_for_c[float(c_)],
                        label=fr"$c = {c_:.2f}$") for c_ in _C_GRID]
    ref_handle = [Line2D([0], [0], color="0.30", lw=1.1, linestyle=(0, (4, 3)), label=ref_lab)]
    ax.legend(handles=handles_N + handles_c + ref_handle, loc=loc,
              fontsize=9, framealpha=0.92, ncol=2)


def _panel_cells(cells, panel):
    return [c for c in cells if c["params"]["panel"] == panel]


def _save_single(cells, N_grid, *, panel, name, run, cumulative=False) -> None:
    sub = _panel_cells(cells, panel)
    if not sub:
        return
    suffix = "_cumulative" if cumulative else ""
    with report_style():
        fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
        _plot_panel(ax, sub, N_grid, cumulative=cumulative)
        fig.tight_layout()
    run.save_figure(f"{name}{suffix}", fig, close=False)
    export_figure(fig, f"fig_{name}_regret_paths{suffix}", strip_title=True)


def _save_combined(cells, run, cumulative=False) -> None:
    suffix = "_cumulative" if cumulative else ""
    with report_style():
        fig, axes = plt.subplots(1, 2, figsize=(2.0 * SQUARE_FIGSIZE[0], SQUARE_FIGSIZE[1]))
        _plot_panel(axes[0], _panel_cells(cells, "symm"), _SYMM_NS, cumulative=cumulative)
        _plot_panel(axes[1], _panel_cells(cells, "asym"), _ASYM_NS, cumulative=cumulative)
        axes[0].set_title("EA-symm: symmetric all-informed")
        axes[1].set_title("EA-asym: asymmetric all-informed")
        fig.tight_layout()
    run.save_figure(f"EA_allinformed_combined{suffix}", fig)


def main(
    *,
    horizon: int = 10_000,
    n_seeds: int = 100,
    base_seed: int = 61,
    quick: bool = False,
    replot: bool = False,
    overwrite: bool = False,
    cumulative: bool = False,
    plot_max_n: float | None = None,
    max_workers: int = parallel.DEFAULT_WORKERS,
) -> None:
    horizon, n_seeds = C.quick_overrides(quick, default_T=horizon, default_S=n_seeds)
    log_every = max(1, horizon // 1000)
    symm_ns = _SYMM_NS if not quick else (2,)
    asym_ns = _ASYM_NS if not quick else (3,)

    rep_d = _make_symm_market(_SYMM_NS[0])
    cfg = C.base_config(
        name=_EXP_ID, market=rep_d,
        sellers=_make_all_informed(rep_d.N, ExplorationSchedule(kind="polynomial", c=_C_GRID[0], eta=_ETA)),
        horizon=horizon, n_seeds=n_seeds, base_seed=base_seed, log_every=log_every,
        oblivious_box=C.tight_oblivious_box(rep_d, expand=0.5),
        informed_box=InformedProjectionBox.from_demand(rep_d),
    )

    with run_directory(_EXP_ID, cfg) as run:
        if replot and figdata.figdata_exists(_EXP_ID) and not overwrite:
            run.logger.info("replot: loading cached figdata for %s", _EXP_ID)
            cells = figdata.load_figdata(_EXP_ID)
        else:
            run.logger.info("EA sweep SYMM=%s ASYM=%s c=%s eta=%.2f T=%d S=%d (workers=%d)",
                            list(symm_ns), list(asym_ns), list(_C_GRID), _ETA,
                            horizon, n_seeds, max_workers)
            specs = _specs(horizon=horizon, n_seeds=n_seeds, base_seed=base_seed,
                           log_every=log_every, symm_ns=symm_ns, asym_ns=asym_ns)
            cells = parallel.map_cells(_worker, specs, max_workers=max_workers, logger=run.logger)
            figdata.save_figdata(_EXP_ID, cells)

        df = pd.DataFrame([dict(c["params"]) for c in cells])
        run.save_summary("EA_allinformed_cells", df)
        export_table(
            df, "table_all_informed_allinformed",
            caption=(
                "Convergence in all-informed markets at exploration rate "
                "$\\eta_i = 1/2$. Both panels satisfy the regularity condition "
                "$\\lambda_{\\max}(B + B^\\top) < 3/2$. For each cell we report "
                "$\\lambda_{\\max}(B + B^\\top)$, the final cumulative learning "
                "regret, and the empirical log--log slope (predicted: $1/2$)."
            ),
            floatfmt=".3g",
        )

        cells = truncate_cells(cells, plot_max_n)
        _save_single(cells, _SYMM_NS, panel="symm", name="EA_symm", run=run, cumulative=cumulative)
        _save_single(cells, _ASYM_NS, panel="asym", name="EA_asym", run=run, cumulative=cumulative)
        _save_combined(cells, run, cumulative=cumulative)
        run.logger.info("%s finished", _EXP_ID)


if __name__ == "__main__":
    main()
