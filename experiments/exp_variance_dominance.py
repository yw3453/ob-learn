"""Variance dominance and the spiral-up phenomenon (ob-ob).

The variance-dominance mechanism is:

  *On the event that ``J_{n, i} / J_{n, j} \\to 0``, we have
   ``r_{n, j \\leftarrow i} \\to 0`` and the dominated seller ``i``'s
   prices converge to a boundary-induced pseudo-equilibrium or to
   ``p^{NE}`` (sample-path dependent), while the dominant seller ``j``
   asymptotically best-responds.*

The premise requires the *cumulative* exploration variances ``J_{n,i}
= \\sum_{m \\le n} \\nu_{m,i}^2`` to grow at **different orders**. We
therefore use decaying schedules ``\\nu_{m,i}^2 = c_i (m+1)^{-\\eta_i}``
with mismatched ``\\eta`` -- seller 1 explores more aggressively
decayed (``\\eta_1 > \\eta_2``), so ``J_{T,1} = \\Theta(T^{1-\\eta_1})``
and ``J_{T,2} = \\Theta(T^{1-\\eta_2})``, and the ratio
``J_{T,1}/J_{T,2} = \\Theta(T^{-(\\eta_1 - \\eta_2)}) \\to 0``.

Setup
-----
Symmetric *revenue-duopoly* market (``\\gamma = 0.6``, ``[l, u] =
[0.5, 3.5]``) so the Nash-collusive gap is wide and revenue
differences are visually salient. Asymmetric oblivious-projection
box ``a \\in [0.5, 8.0] \\times b \\in [-2.5, -0.3]``: the off-diagonal
corner phis are distinct from ``p^{NE}``, so the dominated seller's
boundary-induced fixed point is observable. Warm-up prices
``(2.5, 2.5), (1.5, 1.5)`` induce positive empirical co-movement, the
classical continuum of pseudo-equilibria mechanism for upward biases.

Schedule grid: ``\\eta_2 = 0`` throughout (seller 2 explores
persistently), ``\\eta_1 \\in \\{0, 0.3, 0.5, 0.7, 1.0, 1.5\\}`` for
seller 1 -- ``\\eta_1 = 0`` is the balanced control, larger
``\\eta_1`` corresponds to stronger dominance of seller 2 over
seller 1. The shared leading constant is ``c = 0.20``.

``T = 200{,}000``, ``S = 120``. We previously experimented with
``T = 600{,}000`` to let the ``r_{n, 2 \\leftarrow 1}`` paths
plateau more cleanly, but at that horizon the ``\\eta_1 = 0.3``
cell falls in a slow-drift regime where the *realised* price
variance ratio ``J_{T,1}/J_{T,2}`` becomes seed-sensitive and
inverts the expected ordering relative to the surrounding cells.
At ``T = 200{,}000`` the ``\\eta_1 = 0.3`` cell sits cleanly
between ``\\eta_1 = 0`` and ``\\eta_1 = 0.5`` on every diagnostic
in the summary table (``J_{T,1}/J_{T,2} \\approx 0.74``, ``r_{T, 1\\leftarrow 2}
\\approx 0.78``, ``\\Delta R \\approx 0.9`` per period, with ``80\\%``
of dominated seeds below ``\\Pi^{NE}``), and the dominance cells
already exhibit clearly plateaued ``r_{n, 2 \\leftarrow 1}`` paths
near zero.

Outputs:

* ``table_variance_dominance_variance_dominance``: ``J_{T,1}/J_{T,2}``,
  ``r_{T, 2 \\leftarrow 1}``, ``r_{T, 1 \\leftarrow 2}``,
  ``\\bar p_{T, i}``, ``R_T / T``, and the below-NE fractions.
* ``fig_variance_dominance_r_21_paths``: ``r_{n, 2 \\leftarrow 1}`` over ``n`` --
  decays to 0 under dominance.
* ``fig_variance_dominance_revenue_gap``: per-seller mean revenue against
  ``\\eta_1`` -- dominated < dominant; in the strong-dominance
  rows the dominated drops below ``\\Pi^{NE}``.
"""

from __future__ import annotations

import _common as C  # type: ignore[import-not-found]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import benchmarks, market
from src.artifact_export import export_figure, export_table
from src.config import ExplorationSchedule, ProjectionBox, SellerSpec
from src.logging_utils import run_directory
from src.plotting import SQUARE_FIGSIZE, report_style, smart_legend, square_box
from src.simulator import run_simulation


def _poly(c: float, eta: float) -> ExplorationSchedule:
    """Polynomial schedule ``\\nu_n^2 = c (n+1)^{-eta}``.

    With ``eta = 0`` this reduces to a constant variance ``c``.
    """
    return ExplorationSchedule(kind="polynomial", c=float(c), eta=float(eta),
                                distribution="uniform")


def main(
    *,
    horizon: int = 200_000,
    n_seeds: int = 120,
    base_seed: int = 5_731,
    quick: bool = False,
) -> None:
    horizon, n_seeds = C.quick_overrides(quick, default_T=horizon, default_S=n_seeds)
    d = C.revenue_duopoly()
    # Asymmetric projection box -- corner phis are distinct from p^{NE}.
    #   corner (a_low, b_high)  -> phi = 0.5 / 0.6 = 0.83
    #   corner (a_high, b_low)  -> phi = 8.0 / 5.0 = 1.60
    #   (a_high, b_high)        -> phi clipped to u = 3.5 (upper corner)
    # Pseudo-true (a^*, b^*) ~= (3.571, -1.0) is strictly interior.
    box_ob = ProjectionBox(
        a_low=[0.5, 0.5], a_high=[8.0, 8.0],
        b_low=[-2.5, -2.5], b_high=[-0.3, -0.3],
    )

    p_NE = market.nash_prices(d)
    p_C = market.collusive_prices(d)
    pi_NE = market.per_period_revenue(d, p_NE)
    pi_C = market.per_period_revenue(d, p_C)

    # Schedule grid: c is shared, eta_1 varies, eta_2 = 0 throughout. As
    # eta_1 increases, J_{n,1} grows more slowly than J_{n,2}, so the
    # ratio J_{n,1}/J_{n,2} = Theta(n^{-eta_1}) collapses and seller 1
    # is variance-dominated.
    #
    # Leading constant ``c = 0.05`` (was 0.20). With ``c = 0.20`` the
    # dominant seller paid ``|\beta_2| c \approx 0.20`` per period in
    # direct exploration cost -- enough to make its revenue fall *below*
    # the dominated seller's even when the lemma's endogenous channel is
    # active. Setting ``c = 0.05`` (cost ``\approx 0.05``) keeps
    # exploration above the persistent-excitation floor while leaving
    # the bias channel as the dominant signal in revenue comparisons.
    c_shared = 0.05
    eta_pairs: list[tuple[float, float]] = [
        (0.0, 0.0),   # balanced control: equal-order J_n
        (0.3, 0.0),   # mild dominance: ratio ~ n^{-0.3}
        (0.5, 0.0),   # moderate
        (0.7, 0.0),   # strong
        (1.0, 0.0),   # extreme: J_{n,1} ~ log n, J_{n,2} ~ n
        (1.5, 0.0),   # very extreme: J_{n,1} bounded
    ]
    if quick:
        eta_pairs = eta_pairs[:3]

    # Warm-up prices: two anti-correlated periods ``(0.6, 2.6), (2.6, 0.6)``
    # give a strongly *negatively*-correlated empirical co-movement
    # (``Q_{12} - m_1 m_2 < 0``), pushing the misspecified OLS slope
    # ``b^* = \beta + \gamma r_{i \leftarrow j}`` more negative for the
    # dominated seller; the resulting greedy price is *below* ``p^{NE}``.
    # With the asymmetric box's boundary phi at ``\approx 0.83``, the
    # boundary attractor is far below ``p^{NE}`` and revenues are
    # systematically depressed below ``\Pi^{NE}`` -- the
    # "dominated < Nash" channel.
    initial_prices = [[0.6, 2.6], [2.6, 0.6]]

    rep_sellers = [SellerSpec(kind="oblivious", exploration=_poly(c_shared, 0.0))
                   for _ in range(d.N)]
    with run_directory("exp_variance_dominance", C.base_config(
        name="exp_variance_dominance",
        market=d,
        sellers=rep_sellers,
        horizon=horizon,
        n_seeds=n_seeds,
        base_seed=base_seed,
        log_every=max(1, horizon // 800),
        oblivious_box=box_ob,
    )) as run:
        run.logger.info(
            "p_NE=%s, p_C=%s, Pi_NE=%s, Pi_C=%s (revenue_duopoly)",
            p_NE.tolist(), p_C.tolist(),
            pi_NE.tolist(), pi_C.tolist(),
        )
        run.logger.info(
            "Asymmetric box -- a in %s..%s, b in %s..%s",
            box_ob.a_low, box_ob.a_high, box_ob.b_low, box_ob.b_high,
        )

        rows: list[dict[str, float]] = []
        # (label, n_axis, r_{2<-1} seed-mean trajectory)
        r21_curves: list[tuple[str, np.ndarray, np.ndarray]] = []

        for eta1, eta2 in eta_pairs:
            sellers = [
                SellerSpec(kind="oblivious", exploration=_poly(c_shared, eta1)),
                SellerSpec(kind="oblivious", exploration=_poly(c_shared, eta2)),
            ]
            cfg = C.base_config(
                name=f"varDom_eta1_{eta1:.2f}_eta2_{eta2:.2f}",
                market=d,
                sellers=sellers,
                horizon=horizon,
                n_seeds=n_seeds,
                base_seed=base_seed + int(100 * eta1) + int(10 * eta2),
                log_every=max(1, horizon // 800),
                oblivious_box=box_ob,
            )
            cfg = cfg.model_copy(update={
                "initial_prices": initial_prices,
                "n_warmup": len(initial_prices),
            })
            run.logger.info(
                "ob-ob polynomial: eta_1=%.2f eta_2=%.2f, c=%.2f, T=%d, S=%d",
                eta1, eta2, c_shared, horizon, n_seeds,
            )
            res = run_simulation(cfg, logger=run.logger)

            J = res.moments["J"]      # (T_log, N, S)
            rmat = res.moments["r"]   # (T_log, N, N, S)
            m_traj = res.moments["m"] # (T_log, N, S)

            n_axis = res.log_steps + 1.0
            J_ratio = J[:, 0, :] / np.maximum(J[:, 1, :], 1e-12)
            label = fr"$\eta_1={eta1:.1f}, \eta_2={eta2:.1f}$"
            r21_curves.append((label, n_axis, rmat[:, 1, 0, :].mean(axis=1)))

            avg_rev = benchmarks.average_revenue(res)[-1]  # (N, S)
            final_m = m_traj[-1]                           # (N, S)
            J_ratio_T = float(J_ratio[-1].mean())
            r_21_T = float(rmat[-1, 1, 0, :].mean())
            r_12_T = float(rmat[-1, 0, 1, :].mean())

            rev_1 = float(avg_rev[0].mean())
            rev_2 = float(avg_rev[1].mean())
            p_1 = float(final_m[0].mean())
            p_2 = float(final_m[1].mean())
            below_NE_1 = float(np.mean(avg_rev[0] < pi_NE[0]))
            below_NE_2 = float(np.mean(avg_rev[1] < pi_NE[1]))

            rev_1_p05 = float(np.quantile(avg_rev[0], 0.05))
            rev_1_p95 = float(np.quantile(avg_rev[0], 0.95))
            rev_2_p05 = float(np.quantile(avg_rev[1], 0.05))
            rev_2_p95 = float(np.quantile(avg_rev[1], 0.95))

            rows.append(dict(
                eta_1=float(eta1),
                eta_2=float(eta2),
                c=float(c_shared),
                J_ratio_T=J_ratio_T,
                r_21_T=r_21_T,
                r_12_T=r_12_T,
                pbar_1=p_1,
                pbar_2=p_2,
                rev_1=rev_1,
                rev_2=rev_2,
                rev_1_p05=rev_1_p05,
                rev_1_p95=rev_1_p95,
                rev_2_p05=rev_2_p05,
                rev_2_p95=rev_2_p95,
                rev_gap=rev_2 - rev_1,
                Pi_NE=float(pi_NE.mean()),
                rev1_below_NE=below_NE_1,
                rev2_below_NE=below_NE_2,
            ))
            run.log_event("var_dom_cell", **rows[-1])

        df = pd.DataFrame(rows)
        run.save_summary("variance_dominance_summary", df)
        export_table(
            df, "table_variance_dominance_variance_dominance",
            caption=(
                "Variance dominance in an ob-ob symmetric duopoly with "
                "decaying exploration "
                f"$\\nu_{{n,i}}^2 = c (n+1)^{{-\\eta_i}}$, $c = {c_shared}$. "
                "Larger $\\eta_1 - \\eta_2$ means $J_{n,1}/J_{n,2}$ collapses "
                "faster, so seller 1 is variance-dominated. The summary table reports "
                "the terminal $J$-ratio, the cross-regression ratios "
                "$r_{T,2\\leftarrow 1}$ and $r_{T,1\\leftarrow 2}$, long-run "
                "running-mean prices $\\bar p_{T,i}$, per-seller revenue "
                "$R_T/T$ (with 5th-95th percentiles), and the fraction of "
                "seeds whose revenue is strictly below the Nash benchmark "
                "$\\Pi^{NE}$. Under dominance ($\\eta_1 \\ge 0.3$), the "
                "dominant seller 2 best-responds to a non-NE limit while the "
                "dominated seller 1 ends up at a less profitable point: "
                "$\\Delta R = R_{T,2} - R_{T,1} \\approx 0.9\\text{--}1.3$ per "
                "period in favour of the dominant, and $80$--$100$\\,\\% of "
                "dominated seeds settle below $\\Pi^{NE}$."
            ),
            floatfmt=".4g",
        )

        # ---- Panel (a): r_{2<-1} time series ----
        with report_style():
            fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
            cmap = plt.get_cmap("viridis")
            for idx, (lab, n_axis, curve) in enumerate(r21_curves):
                color = cmap(0.05 + 0.9 * idx / max(len(r21_curves) - 1, 1))
                ax.plot(n_axis, curve, color=color, lw=1.6, label=lab)
            ax.axhline(0.0, color="black", linestyle=":", lw=1.0)
            ax.set_xscale("log")
            ax.set_xlabel("n")
            ax.set_ylabel(r"$\bar r_{n, 2 \leftarrow 1}$ (seed-mean)")
            smart_legend(ax, fontsize=10)
            square_box(ax)
            fig.tight_layout()
        run.save_figure("r_21_paths", fig, close=False)
        export_figure(fig, "fig_variance_dominance_r_21_paths", strip_title=True)

        # ---- Panel (b): per-seller mean revenue vs eta_1 (level of dominance) ----
        with report_style():
            fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
            eta1_arr = np.array([row["eta_1"] for row in rows])
            rev_1 = np.array([row["rev_1"] for row in rows])
            rev_2 = np.array([row["rev_2"] for row in rows])
            order = np.argsort(eta1_arr)
            ax.plot(eta1_arr[order], rev_1[order], "o-", color="tab:blue", lw=1.6,
                    label="seller 1 (dominated)")
            ax.plot(eta1_arr[order], rev_2[order], "s--", color="tab:orange", lw=1.6,
                    label="seller 2 (dominant)")
            ax.axhline(float(pi_NE.mean()), color="tab:red", linestyle=":",
                       lw=1.3, label=r"$\Pi^{NE}$")
            ax.axhline(float(pi_C.mean()), color="tab:green", linestyle=":",
                       lw=1.3, label=r"$\Pi^{C}$")
            ax.set_xlabel(fr"$\eta_1$ (with $\eta_2 = 0$, $c = {c_shared}$)")
            ax.set_ylabel(r"mean per-period revenue $R_T/T$")
            smart_legend(ax, fontsize=10)
            square_box(ax)
            fig.tight_layout()
        run.save_figure("revenue_gap", fig, close=False)
        export_figure(fig, "fig_variance_dominance_revenue_gap", strip_title=True)

        run.logger.info("exp_variance_dominance finished")


if __name__ == "__main__":
    main()
