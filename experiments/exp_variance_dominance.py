"""Variance dominance and the endogenous Stackelberg regret dichotomy (ob-ob).

Illustrates the paper's variance-dominance theorem. Under *top-two variance
dominance* -- one seller's cumulative executed-price variance ``J_{n,k}``
summably dominates every rival's -- the variance-dominant seller becomes an
endogenous Stackelberg **follower**, while each dominated seller freezes as a
committed **leader**:

* **Convergence.** The dominant seller's greedy price converges (pointwise) to
  its *own* oracle best response to the frozen rival profile,
  ``tilde p_{n,dom} -> BR_dom(q_bar_{-dom})``; each dominated seller's greedy
  price freezes at a random ``q_bar_l in [l, u]`` that is generically *not* its
  own best response.
* **Regret dichotomy.** The follower attains **sublinear** dynamic regret
  ``Delta(T) = o(T)`` (its per-period regret decays to zero), whereas each
  frozen leader incurs **linear** regret ``Delta(T) = Theta(T)`` (its per-period
  regret plateaus at the positive constant ``|beta_l|(BR_l(q_bar) - q_bar_l)^2``).

The dominance premise requires the cumulative executed-price variances
``J_{n,k}`` to grow at *different orders*. We use decaying schedules
``nu_{m,k}^2 = c (m+1)^{-eta_k}`` with mismatched ``eta``: the dominant seller
gets the *smallest* exponent (``eta_dom``, largest ``J``), and each dominated
seller a larger exponent (fast-decaying ``J``), so
``J_{dominated}/J_{dom} = Theta(n^{-(eta_dominated - eta_dom)}) -> 0``.

Both sellers explore *sublinearly* (``eta_k > 0`` for all ``k``), so this is the
sublinear-exploration regime (not the persistent/linear ceiling). The dominant
seller's exponent ``eta_dom`` is kept strictly positive but small so that its
own per-period exploration floor still decays and its regret is genuinely
sublinear, while remaining the smallest exponent so it is variance-dominant.

Setup
-----
Symmetric revenue-duopoly market (``gamma = 0.6``, ``[l, u] = [0.5, 3.5]``;
``p^NE = 1.786``, ``Pi^NE = 3.19``, ``Pi^C = 3.91``) so the Nash-collusive gap
is wide. Oblivious projection box ``(a, b) in [1.2, 8.0] x [-2.0, -0.5]``, wide
enough that the dominant seller's limiting estimate ``(alpha + gamma q_inf,
beta)`` stays interior (the interior-projection hypothesis of the proposition).
Anti-correlated warm-up prices induce a persistent cross-correlation so the
dominated seller freezes off its own best response (non-vanishing ``r``), the
generic linear-regret case.

An ``N = 3`` configuration (one dominant follower, two dominated leaders)
exhibits the ``N``-general "one follower, many frozen leaders" structure.

Outputs:

* ``fig_variance_dominance_regret_N2`` and ``fig_variance_dominance_regret_N3``:
  cumulative dynamic regret ``Delta_i(n)`` vs ``n`` on the *original* (linear)
  scale, exported as two separate single-panel square figures (the ``N = 2``
  dominance sweep and the ``N = 3`` configuration) that are placed side by side
  in LaTeX via subfigures. In both, each frozen leader (red, dashed) grows
  linearly while the variance-dominant follower (blue, solid) grows sublinearly
  and stays near-flat. A per-run summary CSV of the ``J``-ratio, cross-regression
  ratios, fitted regret growth exponents, running-mean prices, and
  surplus-capture ratios is saved alongside the run (not exported as a table).
"""

from __future__ import annotations

import re

import _common as C  # type: ignore[import-not-found]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ob_learn import analysis, benchmarks, market
from ob_learn.artifact_export import export_figure
from ob_learn.config import DemandParams, ExplorationSchedule, ProjectionBox, SellerSpec
from ob_learn.logging_utils import run_directory
from ob_learn.plotting import SQUARE_FIGSIZE, report_style, square_box
from ob_learn.simulator import run_simulation

# The N=2 and N=3 cumulative-regret panels are exported as two separate
# single-panel square figures (report_style, SQUARE_FIGSIZE) and placed
# side by side in LaTeX via subfigures; see fig:variance-dominance-regret.

# Shared leading constant of the exploration schedule nu_{n,k}^2 = c (n+1)^{-eta}.
_C_SHARED = 0.05

# Dominant (follower) exploration exponent: smallest exponent so its cumulative
# variance J dominates, yet strictly positive so its own exploration floor
# decays and its regret is genuinely sublinear (slope ~ 1 - eta_dom < 1).
_ETA_DOM = 0.3

# Dominated exponents for the N=2 sweep. Each lies in (eta_dom, 1): the lower
# end (> eta_dom) makes the dominance ratio J_{dominated}/J_{dom} =
# Theta(n^{-(eta - eta_dom)}) collapse, while the upper end (< 1) keeps the
# dominated seller's own cumulative variance J ~ n^{1-eta} above the
# log n log log n exploration floor required by the paper's convergence theorem,
# so it still satisfies that theorem's hypotheses (eta >= 1 would violate them).
_ETA_DOMINATED_GRID = (0.5, 0.7, 0.9)

# Oblivious projection box (matches the appendix): wide enough that the dominant
# seller's limiting estimate stays interior.
_BOX_KW = dict(a_low=1.2, a_high=8.0, b_low=-2.0, b_high=-0.5)


def _poly(c: float, eta: float) -> ExplorationSchedule:
    """Polynomial schedule ``nu_n^2 = c (n+1)^{-eta}``."""
    return ExplorationSchedule(kind="polynomial", c=float(c), eta=float(eta))


def _box(N: int) -> ProjectionBox:
    return ProjectionBox(
        a_low=[_BOX_KW["a_low"]] * N, a_high=[_BOX_KW["a_high"]] * N,
        b_low=[_BOX_KW["b_low"]] * N, b_high=[_BOX_KW["b_high"]] * N,
    )


def _regret_exponent(n_axis: np.ndarray, cum_regret_seedmean: np.ndarray) -> float:
    """Fitted log--log growth exponent of cumulative regret over the tail."""
    return float(analysis.fit_loglog_slope(
        n_axis, cum_regret_seedmean, tail_fraction=0.5,
    )["slope"])


def _run_config(
    *,
    d: DemandParams,
    schedules: list[ExplorationSchedule],
    initial_prices: list[list[float]],
    horizon: int,
    n_seeds: int,
    base_seed: int,
    log_every: int,
    name: str,
    logger,
) -> dict:
    """Run one dominance configuration; return regret/price/surplus diagnostics."""
    box_ob = _box(d.N)
    sellers = [SellerSpec(kind="oblivious", exploration=s) for s in schedules]
    cfg = C.base_config(
        name=name, market=d, sellers=sellers,
        horizon=horizon, n_seeds=n_seeds, base_seed=base_seed,
        log_every=log_every, oblivious_box=box_ob,
    ).model_copy(update={
        "initial_prices": initial_prices,
        "n_warmup": len(initial_prices),
    })
    res = run_simulation(cfg, logger=logger)

    n_axis = res.log_steps.astype(np.float64) + 1.0
    J = res.moments["J"]      # (T_log, N, S)
    rmat = res.moments["r"]   # (T_log, N, N, S)
    m_traj = res.moments["m"]  # (T_log, N, S)

    cum_regret = benchmarks.cumulative_regret(res, d).mean(axis=2)   # (T_log, N)
    per_regret = benchmarks.per_period_regret(res, d).mean(axis=2)   # (T_log, N)
    avg_rev = benchmarks.average_revenue(res)[-1]                    # (N, S)

    p_NE = market.nash_prices(d)
    p_C = market.collusive_prices(d)
    pi_NE = market.per_period_revenue(d, p_NE)
    pi_C = market.per_period_revenue(d, p_C)
    S_i = (avg_rev.mean(axis=1) - pi_NE) / (pi_C - pi_NE)            # (N,)

    return dict(
        res=res, d=d, n_axis=n_axis, J=J, rmat=rmat, m_traj=m_traj,
        cum_regret=cum_regret, per_regret=per_regret,
        avg_rev=avg_rev, S_i=S_i, p_NE=p_NE, pi_NE=pi_NE, pi_C=pi_C,
    )


def main(
    *,
    horizon: int = 200_000,
    n_seeds: int = 120,
    base_seed: int = 5_731,
    quick: bool = False,
) -> None:
    horizon, n_seeds = C.quick_overrides(quick, default_T=horizon, default_S=n_seeds)
    log_every = max(1, horizon // 800)

    # ---- N = 2 market (dominant = seller 2, dominated = seller 1) ----------
    d2 = C.revenue_duopoly()
    p_NE2 = market.nash_prices(d2)
    p_C2 = market.collusive_prices(d2)
    # Anti-correlated warm-up: seller 1 and 2 start on opposite corners so the
    # empirical cross-covariance is strongly signed -> the dominated seller's
    # cross-regression coefficient stays non-zero and it freezes off its BR.
    init2 = [[0.6, 2.6], [2.6, 0.6]]

    eta_grid = _ETA_DOMINATED_GRID[:2] if quick else _ETA_DOMINATED_GRID

    rep_sellers = [SellerSpec(kind="oblivious", exploration=_poly(_C_SHARED, _ETA_DOM))
                   for _ in range(d2.N)]
    with run_directory("exp_variance_dominance", C.base_config(
        name="exp_variance_dominance", market=d2, sellers=rep_sellers,
        horizon=horizon, n_seeds=n_seeds, base_seed=base_seed,
        log_every=log_every, oblivious_box=_box(d2.N),
    )) as run:
        run.logger.info(
            "N=2 revenue_duopoly: p_NE=%s p_C=%s Pi_NE=%s Pi_C=%s; "
            "eta_dom=%.3f, eta_dominated_grid=%s, c=%.3f, T=%d, S=%d",
            p_NE2.tolist(), p_C2.tolist(),
            market.per_period_revenue(d2, p_NE2).tolist(),
            market.per_period_revenue(d2, p_C2).tolist(),
            _ETA_DOM, list(eta_grid), _C_SHARED, horizon, n_seeds,
        )

        rows: list[dict] = []
        # (label, follower cum-regret curve, leader cum-regret curve, n_axis)
        regret_curves: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []

        for eta1 in eta_grid:
            # seller 0 = dominated (large eta), seller 1 = dominant (small eta).
            scheds = [_poly(_C_SHARED, eta1), _poly(_C_SHARED, _ETA_DOM)]
            out = _run_config(
                d=d2, schedules=scheds, initial_prices=init2,
                horizon=horizon, n_seeds=n_seeds,
                base_seed=base_seed + int(100 * eta1),
                log_every=log_every,
                name=f"varDom_eta1_{eta1:.2f}_etaDom_{_ETA_DOM:.2f}",
                logger=run.logger,
            )
            n_axis = out["n_axis"]
            dom, sub = 1, 0  # dominant (follower) index, dominated (leader) index
            reg_follower = out["cum_regret"][:, dom]
            reg_leader = out["cum_regret"][:, sub]
            slope_follower = _regret_exponent(n_axis, reg_follower)
            slope_leader = _regret_exponent(n_axis, reg_leader)

            J_ratio_T = float((out["J"][-1, sub, :] / np.maximum(out["J"][-1, dom, :], 1e-12)).mean())
            r_dom_from_sub = float(out["rmat"][-1, dom, sub, :].mean())  # -> 0
            r_sub_from_dom = float(out["rmat"][-1, sub, dom, :].mean())  # -> large
            pbar_dom = float(out["m_traj"][-1, dom].mean())
            pbar_sub = float(out["m_traj"][-1, sub].mean())

            label = fr"dominated ($\eta={eta1:.1f}$)"
            regret_curves.append((label, reg_follower, reg_leader, n_axis))

            row = dict(
                eta_dominated=float(eta1),
                eta_dominant=float(_ETA_DOM),
                J_ratio_T=J_ratio_T,
                r_dom_from_dominated=r_dom_from_sub,
                r_dominated_from_dom=r_sub_from_dom,
                pbar_follower=pbar_dom,
                pbar_leader=pbar_sub,
                p_NE=float(out["p_NE"][0]),
                regret_slope_follower=slope_follower,
                regret_slope_leader=slope_leader,
                S_follower=float(out["S_i"][dom]),
                S_leader=float(out["S_i"][sub]),
            )
            rows.append(row)
            run.log_event("var_dom_cell", **row)
            run.logger.info(
                "eta_lead=%.2f: J_ratio=%.3g, r(dom<-lead)=%.3f, r(lead<-dom)=%.3f, "
                "regret slope follower=%.3f leader=%.3f, S_follower=%.3f S_leader=%.3f",
                eta1, J_ratio_T, r_dom_from_sub, r_sub_from_dom,
                slope_follower, slope_leader, out["S_i"][dom], out["S_i"][sub],
            )

        df = pd.DataFrame(rows)
        # Kept as an internal run record; the paper reports the key numbers in
        # prose and the figure caption rather than in a table.
        run.save_summary("variance_dominance_summary", df)
        _save_regret_npz(run, regret_curves, tag="N2")

        # ---- N = 3 configuration: one follower, two frozen leaders --------
        d3 = DemandParams.symmetric(
            N=3, alpha=2.5, beta=-1.0, gamma=0.3, l=0.5, u=3.5, noise_std=0.2,
        )
        eta_lead3 = 0.7
        # seller 2 = dominant (follower); sellers 0, 1 = dominated leaders.
        scheds3 = [_poly(_C_SHARED, eta_lead3), _poly(_C_SHARED, eta_lead3),
                   _poly(_C_SHARED, _ETA_DOM)]
        # Anti-correlated 3-cycle warm-up (distinct own prices per seller).
        init3 = [[0.6, 2.6, 1.6], [2.6, 1.6, 0.6], [1.6, 0.6, 2.6]]
        run.logger.info(
            "N=3 symmetric (gamma=0.3): follower eta=%.2f, leaders eta=%.2f",
            _ETA_DOM, eta_lead3,
        )
        out3 = _run_config(
            d=d3, schedules=scheds3, initial_prices=init3,
            horizon=horizon, n_seeds=n_seeds, base_seed=base_seed + 999,
            log_every=log_every, name="varDom_N3", logger=run.logger,
        )
        n3 = out3["n_axis"]
        follower3 = out3["cum_regret"][:, 2]
        leaders3 = [out3["cum_regret"][:, 0], out3["cum_regret"][:, 1]]
        slope_f3 = _regret_exponent(n3, follower3)
        slopes_l3 = [_regret_exponent(n3, lc) for lc in leaders3]
        p_NE3 = out3["p_NE"]
        n3_rows = [dict(
            seller=("follower" if k == 2 else f"leader_{k+1}"),
            eta=(_ETA_DOM if k == 2 else eta_lead3),
            regret_slope=_regret_exponent(n3, out3["cum_regret"][:, k]),
            pbar=float(out3["m_traj"][-1, k].mean()),
            p_NE=float(p_NE3[k]),
            S_i=float(out3["S_i"][k]),
        ) for k in range(3)]
        df3 = pd.DataFrame(n3_rows)
        run.save_summary("variance_dominance_N3", df3)
        run.logger.info(
            "N=3: follower regret slope=%.3f, leader slopes=%s, S=%s",
            slope_f3, [round(s, 3) for s in slopes_l3],
            np.round(out3["S_i"], 3).tolist(),
        )
        np.savez(
            run.directory / "regret_curves_N3.npz",
            n_axis=n3, follower=follower3,
            leader_0=leaders3[0], leader_1=leaders3[1],
        )

        # ---- separate N=2 and N=3 cumulative-regret panels ----------------
        # Exported as two single-panel square figures placed side by side in
        # LaTeX via subfigures (fig:variance-dominance-regret).
        fig2 = fig_regret_N2(regret_curves)
        run.save_figure("regret_cumulative_N2", fig2, close=False)
        export_figure(fig2, "fig_variance_dominance_regret_N2",
                      strip_title=False)
        fig3 = fig_regret_N3(n3, follower3, leaders3)
        run.save_figure("regret_cumulative_N3", fig3, close=False)
        export_figure(fig3, "fig_variance_dominance_regret_N3",
                      strip_title=False)

        run.logger.info("exp_variance_dominance finished")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _save_regret_npz(run, curves, *, tag: str) -> None:
    """Persist the raw regret curves so figures can be regenerated offline."""
    data: dict[str, np.ndarray] = {}
    for idx, (lab, follower, leader, n_axis) in enumerate(curves):
        data[f"n_axis_{idx}"] = n_axis
        data[f"follower_{idx}"] = follower
        data[f"leader_{idx}"] = leader
        data[f"label_{idx}"] = np.array(lab)
    np.savez(run.directory / f"regret_curves_{tag}.npz", **data)


def _blue(idx: int, n_cells: int):
    return plt.get_cmap("Blues")(0.45 + 0.50 * idx / max(n_cells - 1, 1))


def _red(idx: int, n_cells: int):
    return plt.get_cmap("Reds")(0.45 + 0.50 * idx / max(n_cells - 1, 1))


def _style_regret_axis(ax) -> None:
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"cumulative regret $\Delta_i(n)$")
    ax.margins(x=0)
    ax.set_ylim(bottom=0.0)
    ax.ticklabel_format(axis="both", style="sci", scilimits=(-3, 4))
    square_box(ax)


# Distinct per-pair colors (ColorBrewer Dark2): each color identifies one
# market, i.e. a (dominant, dominated) pair. Within a pair, role is encoded by
# linestyle (solid = dominant, dashed = dominated), so the correspondence
# between the two curves of a pair is readable from the shared color.
_PAIR_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]


def _eta_text(label: str) -> str:
    """Extract the numeric exponent from a curve label like
    ``dominated ($\\eta=0.5$)`` and return it as ``$\\eta=0.5$``."""
    m = re.search(r"([-+]?\d*\.?\d+)", label)
    return rf"$\eta={float(m.group(1)):.1f}$" if m else label


def fig_regret_N2(curves):
    """Single square panel: cumulative dynamic regret ``Delta_i(n)`` for the
    ``N=2`` dominance sweep. Each market (a dominant, dominated pair) is drawn in
    one color; the dominant seller (solid, ``eta=0.3``) grows sublinearly and the
    dominated seller (dashed, exponent ``eta``) grows linearly. A color legend
    identifies the pair by the dominated exponent, and a linestyle legend the
    role. Returns the figure (caller saves)."""
    with report_style():
        fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
        color_handles = []
        for idx, (lab, follower, leader, n_axis) in enumerate(curves):
            color = _PAIR_COLORS[idx % len(_PAIR_COLORS)]
            ax.plot(n_axis, follower, color=color, lw=2.0, linestyle="-")
            ax.plot(n_axis, leader, color=color, lw=2.0, linestyle="--")
            color_handles.append(
                Line2D([0], [0], color=color, lw=2.4, linestyle="-",
                       label=_eta_text(lab)),
            )
        role_handles = [
            Line2D([0], [0], color="0.35", lw=2.4, linestyle="-",
                   label=r"dominant ($\eta=0.3$)"),
            Line2D([0], [0], color="0.35", lw=2.0, linestyle="--",
                   label="dominated"),
        ]
        color_legend = ax.legend(
            handles=color_handles, loc="upper left", framealpha=0.9,
            fontsize=11, title=r"dominated $\eta$", title_fontsize=11,
        )
        ax.add_artist(color_legend)
        ax.legend(handles=role_handles, loc="lower right", framealpha=0.9,
                  fontsize=11)
        _style_regret_axis(ax)
        fig.tight_layout()
    return fig


def fig_regret_N3(n3_axis, follower3, leaders3):
    """Single square panel: cumulative dynamic regret ``Delta_i(n)`` for the
    ``N=3`` configuration (one dominant follower, two dominated leaders). The
    follower (blue, solid) grows sublinearly; the two leaders (red, dashed) grow
    linearly. Returns the figure (caller saves)."""
    with report_style():
        fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
        n3 = np.asarray(n3_axis, dtype=np.float64)
        ax.plot(n3, follower3, color=_blue(1, 2), lw=2.4, linestyle="-")
        for j, lc in enumerate(leaders3):
            ax.plot(n3, lc, color=_red(j, 2), lw=2.0, linestyle="--")
        ax.legend(handles=[
            Line2D([0], [0], color=_blue(1, 2), lw=2.4, linestyle="-",
                   label=r"dominant ($\eta=0.3$)"),
            Line2D([0], [0], color=_red(0, 2), lw=2.0, linestyle="--",
                   label=r"dominated ($\eta=0.7$)"),
        ], loc="upper left", framealpha=0.9)
        _style_regret_axis(ax)
        fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
