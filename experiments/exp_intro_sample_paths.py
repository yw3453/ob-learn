"""Intro sample paths: three discrete-time price trajectories.

Produces the paper's introductory sample-path figure (``fig1_near_NE``,
``fig1_intermediate``, ``fig1_near_C``): three discrete-time sample paths over
``T = 10\\,000`` periods with ``\\Theta(\\sqrt n)`` cumulative exploration and a
*very small* leading constant ``c = 0.05`` (so ``\\nu_n^2 = 0.025 / \\sqrt n``,
well below the empirical fast-regime threshold for the baseline duopoly). With very small
``\\nu`` the seed-to-seed history dependence is maximal: the OLS regression's
column space is near-singular, the empirical regression ratios ``r_i`` are
determined by warm-up and early-period noise rather than by dithering, and the
long-run average prices freeze onto a wide range of pseudo-equilibria. Each
panel overlays the raw per-period prices for one fixed seed (``FIG1_SEEDS``).

The named figures are written to ``results/figures/`` in addition to the run
directory under ``results/``.
"""

from __future__ import annotations

import _common as C  # type: ignore[import-not-found]
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from ob_learn import market
from ob_learn.artifact_export import export_figure
from ob_learn.config import ExperimentConfig, ExplorationSchedule
from ob_learn.logging_utils import run_directory
from ob_learn.plotting import SQUARE_FIGSIZE, report_style, smart_legend, square_box
from ob_learn.simulator import run_simulation

# Per-panel generator seeds: panel ``label`` simulates with
# ``np.random.default_rng(FIG1_SEEDS[label])``.
FIG1_SEEDS: dict[str, int] = {
    "near_NE": 676765432,
    "intermediate": 1768135341,
    "near_C": 2639215061,
}


def _fig1_three_sample_paths(run, cfg: ExperimentConfig, *, burn_in: int = 50) -> None:
    d = cfg.market
    horizon = cfg.horizon
    res = run_simulation(
        cfg, logger=run.logger, child_seeds=np.array(list(FIG1_SEEDS.values()), dtype=np.uint32)
    )

    p_NE = float(np.mean(market.nash_prices(d)))
    p_C = float(np.mean(market.collusive_prices(d)))

    n_axis = res.log_steps + 1.0
    keep = n_axis >= burn_in

    # Default wider y-limits: pad below NE so the dip below NE in the
    # near-NE panel is visible.
    span = p_C - p_NE
    default_ymin = p_NE - 0.30 * span
    default_ymax = p_C + 0.10 * span
    # The near-C panel's raw prices reach well into the [1.4, 2.3] band
    # under sqrt(n) dithering, so we widen its y-range explicitly.
    ylims_per_label = {
        "near_NE": (default_ymin, default_ymax),
        "intermediate": (default_ymin, default_ymax),
        "near_C": (1.4, 2.3),
    }

    for idx, (label, seed) in enumerate(FIG1_SEEDS.items()):
        ymin, ymax = ylims_per_label.get(label, (default_ymin, default_ymax))
        with report_style():
            fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
            ax.plot(n_axis[keep], res.prices[keep, 0, idx], color="tab:blue",
                    lw=0.7, alpha=0.85, label=r"seller 1: $p_{n,1}$")
            ax.plot(n_axis[keep], res.prices[keep, 1, idx], color="tab:orange",
                    lw=0.7, alpha=0.85, label=r"seller 2: $p_{n,2}$")
            ax.axhline(p_NE, color="tab:red", linestyle=":", lw=1.3, label=r"$p^{NE}$")
            ax.axhline(p_C, color="tab:green", linestyle=":", lw=1.3, label=r"$p^{C}$")
            ax.set_xlabel("n")
            ax.set_ylabel("price")
            ax.set_xlim(burn_in, horizon)
            ax.set_ylim(ymin, ymax)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
            smart_legend(ax, fontsize=11)
            square_box(ax)
            fig.tight_layout()
        run.save_figure(f"fig1_{label}_seed_{seed}", fig, close=False)
        export_figure(fig, f"fig1_{label}", strip_title=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(*, horizon: int = 10_000, quick: bool = False) -> None:
    horizon, _ = C.quick_overrides(quick, default_T=horizon, default_S=len(FIG1_SEEDS))
    d = C.baseline_demand()
    # Small leading constant: nu_n^2 = 0.025 / sqrt(n), below the baseline
    # duopoly's empirical fast-regime threshold (nu^2 ~= 0.014) over the whole
    # horizon, so the long-run prices depend on early-period noise.
    sched = ExplorationSchedule(kind="sqrt_n", c=0.05)
    cfg = C.base_config(
        name="exp_intro_sample_paths",
        market=d,
        sellers=C.make_oblivious_sellers(d.N, sched),
        horizon=horizon,
        n_seeds=len(FIG1_SEEDS),
        log_every=1,
        oblivious_box=C.tight_oblivious_box(d, expand=0.7),
    )
    with run_directory("exp_intro_sample_paths", cfg) as run:
        run.logger.info(
            "p_NE=%s, p_C=%s",
            market.nash_prices(d).tolist(),
            market.collusive_prices(d).tolist(),
        )
        run.logger.info(
            "Intro sample paths: three discrete sample paths under sqrt_n exploration",
        )
        _fig1_three_sample_paths(run, cfg)
        run.logger.info("exp_intro_sample_paths finished (results/figures/ updated)")


if __name__ == "__main__":
    main()
