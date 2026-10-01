"""Robustness of the oblivious-vs-informed dichotomy to the learning rule.

Re-runs the paper's core phenomena under alternative estimators/exploration
schemes (ridge + shrinking penalty; epsilon-greedy) and shows the story is
unchanged. Two artefacts:

1. **Continuum scatter** (scenario 1): limiting average prices of an all-oblivious
   duopoly under decaying exploration, symmetric vs. asymmetric, per rule.
2. **Learning-regret panels** (scenarios 2 / mixed / 3): aggregate cumulative
   learning regret ``sum_i Delta_i^le(n)`` under persistent-oblivious,
   mixed, and decaying all-informed exploration. Each panel uses its regime's
   rate-revealing axis: **semilog-x** (``propto log n``) for all-oblivious,
   **log-log** (fitted power) for mixed, **log-log** (``propto sqrt(n)``) for
   all-informed.

Everything is cached to ``results/figdata/`` and cells run in parallel; the
regret panels use ``mse_horizon = 1e4``,
while the continuum keeps its own (shorter) horizon.
"""

from __future__ import annotations

from collections import defaultdict

import _common as C  # type: ignore[import-not-found]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ob_learn import analysis, benchmarks, figdata, market, parallel
from ob_learn.artifact_export import export_figure
from ob_learn.config import ExperimentConfig, ExplorationSchedule, SellerSpec
from ob_learn.logging_utils import run_directory
from ob_learn.plotting import (
    SQUARE_FIGSIZE,
    loglog_regret_axes,
    perperiod_reference,
    regret_reference,
    report_style,
    square_box,
    style_regret_axes,
    truncate_cells,
)
from ob_learn.simulator import run_simulation

_EXP_ID = "exp_learning_rule_robustness"

NU2_PERSISTENT = 0.20
DITHER_DECAY_INFORMED = dict(c=0.10, eta=0.5)
RIDGE_DECAY = 0.5
FAMILY_CMAP = {"ridge": "Oranges", "eps": "Greens"}
FAMILY_BASE = {"ridge": "tab:orange", "eps": "tab:green"}

CONT_DITHER = dict(c_sym=0.30, eta_sym=0.85, c_dom=0.50, eta_dom=0.25, c_weak=0.25, eta_weak=0.80)
CONT_RIDGE_LAMBDA = 0.20
CONT_EPS = dict(c_sym=0.50, eta_sym=0.85, c_dom=0.80, eta_dom=0.15, c_weak=0.40, eta_weak=0.80)


def _sched_poly(c, eta):
    return ExplorationSchedule(kind="polynomial", c=c, eta=eta)


def _sched_const(nu2):
    return ExplorationSchedule(kind="constant", nu=float(np.sqrt(nu2)))


def _dummy():
    return ExplorationSchedule(kind="constant", nu=0.0)


def _shade(family, k, m):
    cmap = plt.get_cmap(FAMILY_CMAP[family])
    return cmap(0.68) if m <= 1 else cmap(0.42 + 0.46 * (k / (m - 1)))


# ---------------------------------------------------------------------------
# Continuum scatter (scenario 1)
# ---------------------------------------------------------------------------

def _build_warmups(d, *, n_configs, seed, n_rows=4):
    rng = np.random.default_rng(seed)
    centers = [(d.l + 0.2, d.l + 0.2), (d.u - 0.2, d.u - 0.2), (d.l + 0.2, d.u - 0.2),
               (d.u - 0.2, d.l + 0.2), (1.5, 1.5), (2.0, 2.0)]
    per = max(1, n_configs // len(centers))
    out = []
    for cx, cy in centers:
        for _ in range(per):
            block = []
            for _r in range(n_rows):
                jx, jy = rng.uniform(-0.15, 0.15, size=2)
                block.append([float(np.clip(cx + jx, d.l + 0.05, d.u - 0.05)),
                              float(np.clip(cy + jy, d.l + 0.05, d.u - 0.05))])
            out.append(block)
    return out


def _cont_sellers(family, N, regime):
    sellers = []
    if family == "ridge":
        for i in range(N):
            if regime == "symmetric":
                c, eta = CONT_DITHER["c_sym"], CONT_DITHER["eta_sym"]
            elif i == 0:
                c, eta = CONT_DITHER["c_dom"], CONT_DITHER["eta_dom"]
            else:
                c, eta = CONT_DITHER["c_weak"], CONT_DITHER["eta_weak"]
            sellers.append(SellerSpec(kind="oblivious", exploration=_sched_poly(c, eta)))
        return sellers, dict(estimator="ridge", ridge_lambda=CONT_RIDGE_LAMBDA,
                             ridge_decay=RIDGE_DECAY, exploration_method="dither")
    for i in range(N):
        if regime == "symmetric":
            c, eta = CONT_EPS["c_sym"], CONT_EPS["eta_sym"]
        elif i == 0:
            c, eta = CONT_EPS["c_dom"], CONT_EPS["eta_dom"]
        else:
            c, eta = CONT_EPS["c_weak"], CONT_EPS["eta_weak"]
        sellers.append(SellerSpec(kind="oblivious", exploration=_dummy(), epsilon_c=c, epsilon_eta=eta))
    return sellers, dict(exploration_method="epsilon_greedy")


def _continuum_worker(spec: dict) -> dict:
    family = spec["family"]
    regime = spec["regime"]
    d = C.baseline_demand()
    box_ob = C.tight_oblivious_box(d, expand=0.6)
    box_in = C.tight_informed_box(d)
    reg_off = 0 if regime == "symmetric" else 500_000
    fam_off = 104729 * (hash(family) % 97)
    warmups = _build_warmups(d, n_configs=int(spec["n_warmups"]),
                             seed=int(spec["base_seed"]) + reg_off + fam_off)
    emp = []
    for k, warm in enumerate(warmups):
        sellers, kwargs = _cont_sellers(family, d.N, regime)
        cfg = ExperimentConfig(
            name=f"rob_cont_{family}_{regime}_w{k:03d}", market=d, sellers=sellers,
            oblivious_projection=box_ob, informed_projection=box_in,
            horizon=int(spec["horizon"]), n_seeds=int(spec["n_seeds"]),
            base_seed=int(spec["base_seed"]) + reg_off + 1000 * k,
            log_every=int(spec["log_every"]), initial_prices=warm, n_warmup=len(warm), **kwargs)
        res = run_simulation(cfg, progress=False)
        final_m = res.moments["m"][-1]
        for s in range(int(spec["n_seeds"])):
            emp.append(final_m[:, s].copy())
    pts = np.asarray(emp)
    return {"params": {"kind": "cont", "family": family, "regime": regime,
                       "n_points": int(pts.shape[0]),
                       "price_std": float(np.mean(pts.std(axis=0)))},
            "points": pts}


# ---------------------------------------------------------------------------
# Learning-regret panels (scenarios 2 / mixed / 3)
# ---------------------------------------------------------------------------

def _regret_worker(spec: dict) -> dict:
    d = C.baseline_demand()
    box_ob = C.tight_oblivious_box(d, expand=0.6)
    box_in = C.tight_informed_box(d)
    cfg = ExperimentConfig(
        name=spec["name"], market=d, sellers=spec["sellers"],
        oblivious_projection=box_ob, informed_projection=box_in,
        horizon=int(spec["horizon"]), n_seeds=int(spec["n_seeds"]),
        base_seed=int(spec["base_seed"]), log_every=int(spec["log_every"]), **spec["kwargs"])
    res = run_simulation(cfg, progress=False, compute_moments=False)
    n_grid = (res.log_steps + 1).astype(np.float64)
    reg = benchmarks.cumulative_learning_regret(res, d).sum(axis=1).mean(axis=1)
    pp = benchmarks.per_period_learning_regret(res, d).sum(axis=1).mean(axis=1)
    slope = float(analysis.fit_loglog_slope(n_grid, np.maximum(reg, 1e-12))["slope"])
    pp_slope = float(analysis.fit_loglog_slope(n_grid, np.maximum(pp, 1e-12))["slope"])
    return {"params": {"kind": "reg", "panel": spec["panel"], "label": spec["label"],
                       "family": spec["family"], "regret_slope_tail": slope,
                       "pp_slope_tail": pp_slope},
            "n": n_grid, "regret": reg, "pp": pp}


def _ob_persistent_specs(N):
    specs = []
    for eps in (0.05, 0.10, 0.20):
        specs.append(dict(name=f"rob_ob_eps{eps:g}", panel="ob",
                          label=rf"$\varepsilon$-greedy ($\varepsilon={eps:g}$)", family="eps",
                          sellers=[SellerSpec(kind="oblivious", exploration=_dummy(),
                                              epsilon_c=eps, epsilon_eta=0.0) for _ in range(N)],
                          kwargs=dict(exploration_method="epsilon_greedy")))
    for lam in (0.25, 0.5, 1.0):
        specs.append(dict(name=f"rob_ob_ridge{lam:g}", panel="ob",
                          label=rf"Ridge ($\lambda_0={lam:g}$)", family="ridge",
                          sellers=[SellerSpec(kind="oblivious", exploration=_sched_const(NU2_PERSISTENT))
                                   for _ in range(N)],
                          kwargs=dict(estimator="ridge", ridge_lambda=lam, ridge_decay=RIDGE_DECAY,
                                      exploration_method="dither")))
    return specs


def _mixed_specs():
    specs = []
    for lam in (0.25, 0.5, 1.0):
        specs.append(dict(name=f"rob_mixed_ridge{lam:g}", panel="mixed",
                          label=rf"Ridge ($\lambda_0={lam:g}$)", family="ridge",
                          sellers=[SellerSpec(kind="oblivious", exploration=_sched_const(NU2_PERSISTENT)),
                                   SellerSpec(kind="informed", forecast_rule="mean_price",
                                              exploration=_sched_poly(**DITHER_DECAY_INFORMED))],
                          kwargs=dict(estimator="ridge", ridge_lambda=lam, ridge_decay=RIDGE_DECAY,
                                      exploration_method="dither")))
    for eps in (0.05, 0.10, 0.20):
        specs.append(dict(name=f"rob_mixed_eps{eps:g}", panel="mixed",
                          label=rf"$\varepsilon$-greedy ($\varepsilon_{{\mathrm{{ob}}}}={eps:g}$)", family="eps",
                          sellers=[SellerSpec(kind="oblivious", exploration=_dummy(), epsilon_c=eps, epsilon_eta=0.0),
                                   SellerSpec(kind="informed", forecast_rule="mean_price",
                                              exploration=_dummy(), epsilon_c=0.40, epsilon_eta=0.6)],
                          kwargs=dict(exploration_method="epsilon_greedy")))
    return specs


def _informed_specs(N):
    specs = []
    for lam in (0.25, 0.5, 1.0):
        specs.append(dict(name=f"rob_in_ridge{lam:g}", panel="informed",
                          label=rf"Ridge ($\lambda_0={lam:g}$)", family="ridge",
                          sellers=[SellerSpec(kind="informed", forecast_rule="mean_price",
                                              exploration=_sched_poly(**DITHER_DECAY_INFORMED)) for _ in range(N)],
                          kwargs=dict(estimator="ridge", ridge_lambda=lam, ridge_decay=RIDGE_DECAY,
                                      exploration_method="dither")))
    for c, eta in ((0.40, 0.5), (0.30, 0.6), (0.20, 0.7)):
        specs.append(dict(name=f"rob_in_eps_c{c:g}_e{eta:g}", panel="informed",
                          label=rf"$\varepsilon$-greedy ($c={c:g},\eta={eta:g}$)", family="eps",
                          sellers=[SellerSpec(kind="informed", forecast_rule="mean_price",
                                              exploration=_dummy(), epsilon_c=c, epsilon_eta=eta) for _ in range(N)],
                          kwargs=dict(exploration_method="epsilon_greedy")))
    return specs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_continuum_panel(pts, color, p_NE, p_C, *, xlim, ylim):
    with report_style():
        fig, ax = plt.subplots(figsize=(SQUARE_FIGSIZE[0] * 0.72, SQUARE_FIGSIZE[1] * 0.72))
        ax.scatter(pts[:, 0], pts[:, 1], s=11, color=color, alpha=0.55,
                   edgecolor="white", linewidth=0.3, rasterized=True)
        ax.plot([p_NE[0], p_C[0]], [p_NE[1], p_C[1]], color="tab:gray", linestyle="--", lw=1.0, zorder=1)
        ax.scatter([p_NE[0]], [p_NE[1]], s=95, color="tab:red", marker="X",
                   edgecolor="white", linewidth=0.6, zorder=10, label="Nash")
        ax.scatter([p_C[0]], [p_C[1]], s=95, color="tab:green", marker="X",
                   edgecolor="white", linewidth=0.6, zorder=10, label="Collusive")
        ax.set_xlabel(r"$\bar p_1$")
        ax.set_ylabel(r"$\bar p_2$")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        square_box(ax)
        fig.tight_layout()
    return fig


def _panel_colors(reg_cells, panel):
    cells = [c for c in reg_cells if c["params"]["panel"] == panel]
    totals = defaultdict(int)
    for c in cells:
        totals[c["params"]["family"]] += 1
    idx = defaultdict(int)
    colored = []
    for c in cells:
        fam = c["params"]["family"]
        colored.append((c, _shade(fam, idx[fam], totals[fam])))
        idx[fam] += 1
    return colored


# Per-panel config: (per-period reference slope | None=fitted, cumulative ref kind).
_PANEL_CFG = {
    "ob": (-1.0, "log"),
    "mixed": (None, "power"),
    "informed": (-0.5, "sqrt"),
}


def _plot_regret_panel(ax, reg_cells, panel, *, title, cumulative):
    colored = _panel_colors(reg_cells, panel)
    pp_slope, cum_kind = _PANEL_CFG[panel]
    key = "regret" if cumulative else "pp"
    for cell, color in colored:
        ax.plot(cell["n"], np.maximum(cell[key], 1e-12), color=color, lw=1.5,
                alpha=0.92, label=cell["params"]["label"])
    pairs = [(c["n"], c[key]) for c, _ in colored]
    if cumulative:
        if cum_kind == "power":
            slopes = [float(c["params"]["regret_slope_tail"]) for c, _ in colored]
            regret_reference(ax, pairs, kind="power", slope=float(np.nanmedian(slopes)))
        else:
            regret_reference(ax, pairs, kind=cum_kind)
        # Cumulative view: original (linear) scale on both axes.
        style_regret_axes(ax, ylabel="")
    else:
        s = pp_slope if pp_slope is not None else float(
            np.nanmedian([float(c["params"]["pp_slope_tail"]) for c, _ in colored]))
        perperiod_reference(ax, pairs, slope=s)
        loglog_regret_axes(ax, ylabel="")
    ax.set_title(title)
    ax.legend(fontsize=8, framealpha=0.9, loc="upper right" if cumulative else "lower left")


def _plot_regret_figure(reg_cells, *, cumulative=False):
    ylabel = r"cumulative learning regret" if cumulative else r"per-period learning regret"
    with report_style():
        fig, axes = plt.subplots(1, 3, figsize=(3.0 * SQUARE_FIGSIZE[0] * 0.92, SQUARE_FIGSIZE[1]))
        _plot_regret_panel(axes[0], reg_cells, "ob", title="All-oblivious, persistent exploration",
                           cumulative=cumulative)
        _plot_regret_panel(axes[1], reg_cells, "mixed", title="Mixed market", cumulative=cumulative)
        _plot_regret_panel(axes[2], reg_cells, "informed", title="All-informed, decaying exploration",
                           cumulative=cumulative)
        for ax in axes:
            ax.set_ylabel(ylabel)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    *,
    horizon: int = 40_000,
    mse_horizon: int = 10_000,
    n_seeds: int = 16,
    base_seed: int = 4242,
    n_warmups: int = 36,
    mse_seeds: int = 90,
    quick: bool = False,
    replot: bool = False,
    overwrite: bool = False,
    cumulative: bool = False,
    plot_max_n: float | None = None,
    max_workers: int = parallel.DEFAULT_WORKERS,
) -> None:
    if quick:
        horizon = min(horizon, 5_000)
        mse_horizon = min(mse_horizon, 5_000)
        n_seeds = min(n_seeds, 6)
        n_warmups = min(n_warmups, 8)
        mse_seeds = min(mse_seeds, 12)

    d = C.baseline_demand()
    p_NE = market.nash_prices(d)
    p_C = market.collusive_prices(d)
    cont_log_every = max(1, horizon // 400)
    mse_log_every = max(1, mse_horizon // 1000)

    rep_cfg = ExperimentConfig(
        name=_EXP_ID, market=d,
        sellers=[SellerSpec(kind="oblivious", exploration=_sched_poly(0.3, 0.85)) for _ in range(d.N)],
        oblivious_projection=C.tight_oblivious_box(d, expand=0.6),
        informed_projection=C.tight_informed_box(d),
        horizon=horizon, n_seeds=n_seeds, base_seed=base_seed, log_every=cont_log_every,
        estimator="ridge", ridge_lambda=0.5, exploration_method="dither")

    with run_directory(_EXP_ID, rep_cfg) as run:
        if replot and figdata.figdata_exists(_EXP_ID) and not overwrite:
            run.logger.info("replot: loading cached figdata for %s", _EXP_ID)
            cells = figdata.load_figdata(_EXP_ID)
        else:
            run.logger.info("robustness: continuum T=%d S=%d n_warm=%d; regret T=%d S=%d (workers=%d)",
                            horizon, n_seeds, n_warmups, mse_horizon, mse_seeds, max_workers)
            cont_specs = [{"family": f, "regime": r, "horizon": horizon, "n_seeds": n_seeds,
                           "base_seed": base_seed, "n_warmups": n_warmups, "log_every": cont_log_every}
                          for f in ("ridge", "eps") for r in ("symmetric", "asymmetric")]
            cont_cells = parallel.map_cells(_continuum_worker, cont_specs,
                                            max_workers=max_workers, logger=run.logger)
            reg_specs = _ob_persistent_specs(d.N) + _mixed_specs() + _informed_specs(d.N)
            for s in reg_specs:
                s.update(horizon=mse_horizon, n_seeds=mse_seeds, base_seed=base_seed,
                         log_every=mse_log_every)
            reg_cells = parallel.map_cells(_regret_worker, reg_specs,
                                           max_workers=max_workers, logger=run.logger)
            cells = cont_cells + reg_cells
            figdata.save_figdata(_EXP_ID, cells)

        cont_cells = [c for c in cells if c["params"]["kind"] == "cont"]
        reg_cells = [c for c in cells if c["params"]["kind"] == "reg"]

        # Continuum scatter panels (shared axis window).
        all_pts = np.concatenate([c["points"] for c in cont_cells], axis=0)
        pad = 0.06
        xs = np.concatenate([all_pts[:, 0], [p_NE[0], p_C[0]]])
        ys = np.concatenate([all_pts[:, 1], [p_NE[1], p_C[1]]])
        xlim = (xs.min() - pad, xs.max() + pad)
        ylim = (ys.min() - pad, ys.max() + pad)
        for c in cont_cells:
            fam = c["params"]["family"]
            regime = c["params"]["regime"]
            figp = _plot_continuum_panel(c["points"], FAMILY_BASE[fam], p_NE, p_C, xlim=xlim, ylim=ylim)
            run.save_figure(f"robustness_{fam}_{regime}", figp, close=False)
            export_figure(figp, f"fig_robustness_{fam}_{regime}", strip_title=False,
                          tight_bbox=False, dpi=300)
        run.save_summary("continuum_spread", pd.DataFrame([dict(c["params"]) for c in cont_cells]))

        suffix = "_cumulative" if cumulative else ""
        fig_m = _plot_regret_figure(truncate_cells(reg_cells, plot_max_n), cumulative=cumulative)
        run.save_figure(f"robustness_regret{suffix}", fig_m, close=False)
        export_figure(fig_m, f"fig_robustness_regret{suffix}", strip_title=False)
        run.logger.info("%s finished", _EXP_ID)


if __name__ == "__main__":
    main()
