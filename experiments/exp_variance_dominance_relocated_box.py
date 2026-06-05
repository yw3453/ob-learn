"""Variance dominance with a relocated oblivious box.

A re-run of :mod:`exp_variance_dominance` with a *relocated* asymmetric
projection box that satisfies

    ``ell_- = a_low  / (-2 * b_high)  in (q_-, p^{NE})``   (reverse regime)
    ``ell_+ = a_high / (-2 * b_low)   >    p^{NE}``        (strict-dominance regime)

so that the two boundary lock candidates straddle the
``Delta(q) = R_2 - R_1`` sign-change interval ``(q_-, p^{NE})``: the
``ell_-`` candidate would imply ``R_1 > R_2`` if reached, while the
``ell_+`` candidate would imply ``R_2 > R_1``.

The probe asks whether, under variance dominance, the dominated
empirically locks at the ``ell_+`` corner (so the baseline
"dominated earns less" conclusion still holds) or at the ``ell_-``
corner (which would flip the conclusion).

Symmetric duopoly primitives are identical to ``exp_variance_dominance``:
``alpha=2.5, beta=-1, gamma=0.6, [l, u] = [0.5, 3.5]``, so
``p^{NE} = 2.5/1.4 ~= 1.786`` and ``q_- = 2.5/2.6 ~= 0.962``.

Projection box: ``(a_low, a_high) = (1.2, 8.0)``,
``(b_low, b_high) = (-2.0, -0.5)``, giving

* ``ell_- = 1.2 / (2 * 0.5) = 1.20``  (in (0.962, 1.786))
* ``ell_+ = 8.0 / (2 * 2.0) = 2.00``  (>  1.786)
* pseudo-true ``(a^*, b^*) ~= (3.57, -1.0)`` is strictly interior.

Schedule difference vs ``exp_variance_dominance``: the dominant
seller's decay exponent is set to ``eta_2 = 0.01`` (technically sublinear
so that ``J_{n, 2} = o(n)`` and the sublinear-``J`` hypothesis is
satisfied) instead of the original ``eta_2 = 0`` (linear).
All other knobs (warm-up, horizon, seeds, exploration scale) match
``exp_variance_dominance``.

Outputs:

* ``table_variance_dominance_relocated_box``: same column layout as the
  baseline variance-dominance summary, restricted to the new box.
* ``fig_variance_dominance_relocated_box_r_21_paths``: ``r_{n, 2 <- 1}`` paths.
* ``fig_variance_dominance_relocated_box_revenue_gap``: per-seller mean revenue against ``eta_1``.
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
    """Polynomial schedule ``\\nu_n^2 = c (n+1)^{-eta}`` (uniform dithering)."""
    return ExplorationSchedule(kind="polynomial", c=float(c), eta=float(eta),
                                distribution="uniform")


def main(
    *,
    horizon: int = 200_000,
    n_seeds: int = 120,
    base_seed: int = 7_311,
    quick: bool = False,
) -> None:
    horizon, n_seeds = C.quick_overrides(quick, default_T=horizon, default_S=n_seeds)
    d = C.revenue_duopoly()
    # Relocated asymmetric projection box:
    #   ell_- = a_low / (-2 b_high) = 1.2/1.0 = 1.20   in (q_-, p^{NE}) = (0.962, 1.786)
    #   ell_+ = a_high/(-2 b_low)   = 8.0/4.0 = 2.00   >  p^{NE} = 1.786
    # Pseudo-true (a^*, b^*) ~= (3.571, -1.0) is strictly interior.
    box_ob = ProjectionBox(
        a_low=[1.2, 1.2], a_high=[8.0, 8.0],
        b_low=[-2.0, -2.0], b_high=[-0.5, -0.5],
    )

    p_NE = market.nash_prices(d)
    p_C = market.collusive_prices(d)
    pi_NE = market.per_period_revenue(d, p_NE)
    pi_C = market.per_period_revenue(d, p_C)

    c_shared = 0.05
    # Dominant seller now uses eta_2 = 0.01 (technically sublinear so that
    # J_{n,2} = o(n), but ``almost linear'') to satisfy the sublinear-J
    # hypothesis while staying empirically close to the rational
    # linear-exploration limit in the spiral-up regime.
    eta_pairs: list[tuple[float, float]] = [
        (0.0, 0.01),
        (0.3, 0.01),
        (0.5, 0.01),
        (0.7, 0.01),
        (1.0, 0.01),
        (1.5, 0.01),
    ]
    if quick:
        eta_pairs = eta_pairs[:3]

    initial_prices = [[0.6, 2.6], [2.6, 0.6]]

    rep_sellers = [SellerSpec(kind="oblivious", exploration=_poly(c_shared, 0.0))
                   for _ in range(d.N)]
    with run_directory("exp_variance_dominance_relocated_box", C.base_config(
        name="exp_variance_dominance_relocated_box",
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
            "Relocated box -- a in %s..%s, b in %s..%s",
            box_ob.a_low, box_ob.a_high, box_ob.b_low, box_ob.b_high,
        )
        # Print diagnostic lock candidates.
        alpha = float(d.alpha[0])
        b = float(abs(d.beta[0]))
        gamma = float(d.gamma[0][1])
        ell_minus = float(box_ob.a_low[0] / (-2.0 * box_ob.b_high[0]))
        ell_plus = float(box_ob.a_high[0] / (-2.0 * box_ob.b_low[0]))
        q_minus = alpha / (2.0 * b + gamma)
        run.logger.info(
            "lock candidates: ell_- = %.4f (q_- = %.4f), ell_+ = %.4f (p_NE = %.4f)",
            ell_minus, q_minus, ell_plus, float(p_NE[0]),
        )

        rows: list[dict[str, float]] = []
        r21_curves: list[tuple[str, np.ndarray, np.ndarray]] = []

        for eta1, eta2 in eta_pairs:
            sellers = [
                SellerSpec(kind="oblivious", exploration=_poly(c_shared, eta1)),
                SellerSpec(kind="oblivious", exploration=_poly(c_shared, eta2)),
            ]
            cfg = C.base_config(
                name=f"varDom2_eta1_{eta1:.2f}_eta2_{eta2:.2f}",
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

            J = res.moments["J"]
            rmat = res.moments["r"]
            m_traj = res.moments["m"]

            n_axis = res.log_steps + 1.0
            J_ratio = J[:, 0, :] / np.maximum(J[:, 1, :], 1e-12)
            label = fr"$\eta_1={eta1:.1f}, \eta_2={eta2:.1f}$"
            r21_curves.append((label, n_axis, rmat[:, 1, 0, :].mean(axis=1)))

            avg_rev = benchmarks.average_revenue(res)[-1]
            final_m = m_traj[-1]
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

            # Cross-seed quantiles of seller-1's running-mean price
            # to diagnose which lock the dominated converges to (ell_-,
            # p^{NE}, or ell_+).
            p1_p05 = float(np.quantile(final_m[0], 0.05))
            p1_p25 = float(np.quantile(final_m[0], 0.25))
            p1_p50 = float(np.quantile(final_m[0], 0.50))
            p1_p75 = float(np.quantile(final_m[0], 0.75))
            p1_p95 = float(np.quantile(final_m[0], 0.95))

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
                p1_p05=p1_p05,
                p1_p25=p1_p25,
                p1_p50=p1_p50,
                p1_p75=p1_p75,
                p1_p95=p1_p95,
            ))
            run.log_event("var_dom2_cell", **rows[-1])

        df = pd.DataFrame(rows)
        run.save_summary("variance_dominance_relocated_box_summary", df)
        export_table(
            df, "table_variance_dominance_relocated_box",
            caption=(
                "Variance dominance under the \\emph{relocated} oblivious "
                "projection box "
                "$(a, b) \\in [1.2, 8.0]\\times[-2.0, -0.5]$, which places "
                "$\\ell_- = 1.20$ in the reverse regime $(q_-, p^{NE}) = "
                "(0.962, 1.786)$ and $\\ell_+ = 2.00$ above $p^{NE} = 1.786$. "
                "All other primitives match the baseline variance-dominance setup. "
                "The ``p1\\_p$\\cdot$'' "
                "columns report cross-seed quantiles of the dominated "
                "seller's running-mean price $\\bar p_{T,1}$ to diagnose "
                "which boundary lock the dominated converges to."
            ),
            floatfmt=".4g",
        )

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
        export_figure(fig, "fig_variance_dominance_relocated_box_r_21_paths", strip_title=True)

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
            ax.set_xlabel(fr"$\eta_1$ (with $\eta_2 = 0.01$, $c = {c_shared}$)")
            ax.set_ylabel(r"mean per-period revenue $R_T/T$")
            smart_legend(ax, fontsize=10)
            square_box(ax)
            fig.tight_layout()
        run.save_figure("revenue_gap", fig, close=False)
        export_figure(fig, "fig_variance_dominance_relocated_box_revenue_gap", strip_title=True)

        run.logger.info("exp_variance_dominance_relocated_box finished")


if __name__ == "__main__":
    main()
