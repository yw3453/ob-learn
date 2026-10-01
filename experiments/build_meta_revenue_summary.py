"""Compile cross-experiment revenue summary tables.

Reads the latest summary CSVs from the three composition families
(``exp_ob_ob_revenue``, ``exp_ob_in_revenue``, ``exp_in_in_revenue_decay``)
and writes:

    results/tables/table_rho_nash_relative.{md,csv}
    results/tables/table_surplus_capture.{md,csv}
    results/tables/table_rho_S_numeric.csv

Usage (after the three upstream experiments have produced summary CSVs)::

    uv run python experiments/build_meta_revenue_summary.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# This file lives at ``<repo>/experiments/...``; ``parents[1]`` is the repo root.
CODE_DIR = Path(__file__).resolve().parents[1]
RESULTS = CODE_DIR / "results"
TABLES = RESULTS / "tables"

# Reference benchmarks (revenue duopoly, symmetric γ=0.6, [l,u]=[0.5,3.5]).
PI_NE = 3.188775510204082
PI_C = 3.90625
SURPLUS = PI_C - PI_NE  # 0.717474489...


def _latest(prefix: str) -> Path:
    candidates = sorted(RESULTS.glob(f"*_{prefix}_*"))
    if not candidates:
        raise FileNotFoundError(f"no runs found for {prefix}")
    return candidates[-1]


def _read_summary(prefix: str, fname: str) -> pd.DataFrame:
    run_dir = _latest(prefix)
    return pd.read_csv(run_dir / "summary" / fname)


def _rho(rev: float) -> float:
    return rev / PI_NE


def _S(rev: float) -> float:
    return (rev - PI_NE) / SURPLUS


def _fmt_rho(mean: float, lo: float, hi: float) -> str:
    return f"{_rho(mean):.3f} ({_rho(lo):.3f}, {_rho(hi):.3f})"


def _fmt_S(mean: float, lo: float, hi: float) -> str:
    return f"{_S(mean):+.3f} ({_S(lo):+.3f}, {_S(hi):+.3f})"


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)

    rows_rho: list[dict] = []
    rows_S: list[dict] = []

    # --- ob-ob ---------------------------------------------------------------
    df_2a = _read_summary("exp_ob_ob_revenue", "obob_revenue_summary.csv")
    schedule_pretty_2a = {
        "high_const": r"ν²=0.20 (high_const)",
        "low_const": r"ν²=0.05 (low_const)",
    }
    for _, r in df_2a.iterrows():
        cell = schedule_pretty_2a.get(r["schedule"], r["schedule"])
        rows_rho.append(dict(
            composition="ob-ob",
            cell=cell,
            rho_seller0=_fmt_rho(r["avg_R_T_seller0_mean"], r["avg_R_T_seller0_p05"], r["avg_R_T_seller0_p95"]),
            rho_seller1=_fmt_rho(r["avg_R_T_seller1_mean"], r["avg_R_T_seller1_p05"], r["avg_R_T_seller1_p95"]),
        ))
        rows_S.append(dict(
            composition="ob-ob",
            cell=cell,
            S_seller0=_fmt_S(r["avg_R_T_seller0_mean"], r["avg_R_T_seller0_p05"], r["avg_R_T_seller0_p95"]),
            S_seller1=_fmt_S(r["avg_R_T_seller1_mean"], r["avg_R_T_seller1_p05"], r["avg_R_T_seller1_p95"]),
        ))

    # --- in-in ---------------------------------------------------------------
    df_2c2 = _read_summary("exp_in_in_revenue_decay", "inin_revenue_decaying_summary.csv")
    for _, r in df_2c2.iterrows():
        cell = f"mean_price (η={r['eta']:.1f}, c={r['c']:.2f})"
        rows_rho.append(dict(
            composition="in-in",
            cell=cell,
            rho_seller0=_fmt_rho(r["avg_R_T_seller0_mean"], r["avg_R_T_seller0_p05"], r["avg_R_T_seller0_p95"]),
            rho_seller1=_fmt_rho(r["avg_R_T_seller1_mean"], r["avg_R_T_seller1_p05"], r["avg_R_T_seller1_p95"]),
        ))
        rows_S.append(dict(
            composition="in-in",
            cell=cell,
            S_seller0=_fmt_S(r["avg_R_T_seller0_mean"], r["avg_R_T_seller0_p05"], r["avg_R_T_seller0_p95"]),
            S_seller1=_fmt_S(r["avg_R_T_seller1_mean"], r["avg_R_T_seller1_p05"], r["avg_R_T_seller1_p95"]),
        ))

    # --- ob-in ---------------------------------------------------------------
    df_2b = _read_summary("exp_ob_in_revenue", "obin_revenue_summary.csv")
    def rule_label(r) -> str:
        return f"{r['rule']} (η={r['informed_eta']:g})"

    for _, r in df_2b.iterrows():
        cell = rule_label(r)
        rows_rho.append(dict(
            composition="ob-in",
            cell=cell,
            rho_seller0=_fmt_rho(r["avg_R_T_seller0_mean"], r["avg_R_T_seller0_p05"], r["avg_R_T_seller0_p95"]),
            rho_seller1=_fmt_rho(r["avg_R_T_seller1_mean"], r["avg_R_T_seller1_p05"], r["avg_R_T_seller1_p95"]),
        ))
        rows_S.append(dict(
            composition="ob-in",
            cell=cell,
            S_seller0=_fmt_S(r["avg_R_T_seller0_mean"], r["avg_R_T_seller0_p05"], r["avg_R_T_seller0_p95"]),
            S_seller1=_fmt_S(r["avg_R_T_seller1_mean"], r["avg_R_T_seller1_p05"], r["avg_R_T_seller1_p95"]),
        ))

    df_rho = pd.DataFrame(rows_rho)
    df_S = pd.DataFrame(rows_S)

    def _numeric_means() -> pd.DataFrame:
        out_rows: list[dict] = []
        for src_df, kind in ((df_2a, "ob-ob"), (df_2c2, "in-in"), (df_2b, "ob-in")):
            for _, r in src_df.iterrows():
                if kind == "ob-ob":
                    rev0 = r["avg_R_T_seller0_mean"]
                    rev1 = r["avg_R_T_seller1_mean"]
                    cell = schedule_pretty_2a.get(r["schedule"], r["schedule"])
                elif kind == "in-in":
                    rev0 = r["avg_R_T_seller0_mean"]
                    rev1 = r["avg_R_T_seller1_mean"]
                    cell = f"mean_price (η={r['eta']:.1f}, c={r['c']:.2f})"
                else:  # ob-in
                    rev0 = r["avg_R_T_seller0_mean"]
                    rev1 = r["avg_R_T_seller1_mean"]
                    cell = rule_label(r)
                out_rows.append(dict(
                    composition=kind,
                    cell=cell,
                    rho_seller0_mean=_rho(rev0),
                    rho_seller1_mean=_rho(rev1),
                    S_seller0_mean=_S(rev0),
                    S_seller1_mean=_S(rev1),
                ))
        return pd.DataFrame(out_rows)

    df_numeric = _numeric_means()

    rho_caption = (
        "**Nash-relative revenue** $\\rho_i = R_{T,i}/(T\\,\\Pi_i^{NE})$ across "
        "the composition grid. All cells use the same revenue duopoly "
        "($\\alpha=2.5$, $\\beta=-1$, $\\gamma=0.6$, $[l,u]=[0.5, 3.5]$), so "
        "$\\Pi_i^{NE}=3.189$. Values are seed means with 5%–95% cross-seed "
        "ranges in parentheses."
    )
    S_caption = (
        "**Surplus-capture ratio** $S_i = (R_{T,i}/T - \\Pi_i^{NE})/(\\Pi_i^{C} - \\Pi_i^{NE})$ "
        "across the meta-game cells. The symmetric revenue duopoly gives "
        "$\\Pi^{NE}=3.189$ and $\\Pi^{C}=3.906$, so the denominator is "
        "$\\Pi^{C}-\\Pi^{NE}=0.717$. $S_i=0$ corresponds to Nash, $S_i=1$ to "
        "the symmetric collusive benchmark. Mean across seeds with 5%–95% "
        "cross-seed range in parentheses."
    )

    out_rho_md = TABLES / "table_rho_nash_relative.md"
    out_rho_csv = TABLES / "table_rho_nash_relative.csv"
    out_S_md = TABLES / "table_surplus_capture.md"
    out_S_csv = TABLES / "table_surplus_capture.csv"
    out_numeric_csv = TABLES / "table_rho_S_numeric.csv"

    with open(out_rho_md, "w") as f:
        f.write(rho_caption + "\n\n")
        f.write(df_rho.to_markdown(index=False))
        f.write("\n")
    df_rho.to_csv(out_rho_csv, index=False)

    with open(out_S_md, "w") as f:
        f.write(S_caption + "\n\n")
        f.write(df_S.to_markdown(index=False))
        f.write("\n")
    df_S.to_csv(out_S_csv, index=False)

    df_numeric.to_csv(out_numeric_csv, index=False)

    print(f"wrote {out_rho_md}")
    print(f"wrote {out_rho_csv}")
    print(f"wrote {out_S_md}")
    print(f"wrote {out_S_csv}")
    print(f"wrote {out_numeric_csv}")


if __name__ == "__main__":
    main()
