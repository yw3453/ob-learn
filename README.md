# Oblivious Learning and Collusive Pricing

This repository contains the code for **"Oblivious Learning and
Collusive Pricing"** ([arXiv](https://arxiv.org/abs/2606.05363)).

The codebase combines a reusable simulation library in [`ob_learn/`](ob_learn/) with a
suite of experiment scripts in [`experiments/`](experiments/). Runs produce
timestamped artifacts under [`results/`](results/), including logs, compressed
trajectories, summary tables, and generated figures. The figures and tables
shipped with the paper live in [`results/figures/`](results/figures/) and
[`results/tables/`](results/tables/).

## Quick start

```bash
# Install dependencies (uv resolves them from uv.lock).
uv sync

# Optional: verify the install with the unit tests (~5s).
uv run pytest

# List discoverable experiments.
uv run ob-learn list

# Run a single experiment (append --quick for a fast smoke run).
uv run ob-learn run learning-rule-robustness --quick
uv run ob-learn run variance-dominance
```

Each invocation writes a self-contained directory to `results/` with the config
snapshot, per-seed trajectories, summary CSVs, and figures, and also exports the
named figures/tables to `results/figures/` and `results/tables/`.

## Reproducing the paper's figures and tables

Every shipped figure is produced by a single experiment; running it writes the
figure to `results/figures/` under the exact filename used in the manuscript.

**Cached runs and fast re-plotting.** The regret sweeps (global-convergence,
all-informed, mixed, robustness) cache their seed-reduced, figure-ready curves to
`results/figdata/<experiment>.npz` (a few MB total, committed to the repo). This
lets you redraw a figure without re-simulating:

```bash
uv run ob-learn run mixed-small-gain --replot     # redraw from cached figdata
uv run ob-learn run mixed-small-gain --overwrite  # force a fresh simulation
uv run ob-learn run mixed-small-gain --workers 4  # cell-level parallelism (default: min(4, #cores))
```

Cells of each parameter sweep are independent and run in parallel across worker
processes; `--replot` skips simulation entirely and rebuilds the figure from the
cached curves.

| Manuscript figure (filename) | Command |
| --- | --- |
| Intro sample paths (`fig1_near_NE`, `fig1_intermediate`, `fig1_near_C`) | `uv run ob-learn run intro-sample-paths` |
| Pseudo-equilibrium region + surplus capture (`fig_pseudoequilibria_continuum_{sym,asym}explore_{region,revenue}`) | `uv run ob-learn run pseudoequilibria-continuum` |
| Variance-dominance regret (`fig_variance_dominance_regret_N2/N3`) | `uv run ob-learn run variance-dominance` |
| Global-convergence stress tests (`fig_3h_asymmetric_regret_paths`, `fig_3b_dominance_regret_paths`) | `uv run ob-learn run asymmetric-multiseller` and `uv run ob-learn run dominance-margin` |
| All-informed learning regret (`fig_EA_symm_regret_paths`, `fig_EA_asym_regret_paths`) | `uv run ob-learn run all-informed-stress` |
| Mixed-market learning regret (`fig_M1_smallgain_regret_paths`, `fig_M2_multiseller_regret_paths`, `fig_M1b_iv_holds_regret_paths`) | `uv run ob-learn run mixed-small-gain`, `mixed-multiseller`, `mixed-small-gain-iv-holds` |
| Learning-rule robustness appendix (`fig_robustness_{ridge,eps}_{symmetric,asymmetric}`, `fig_robustness_regret`) | `uv run ob-learn run learning-rule-robustness` |
| Numerical revenue comparison (`table_ob_ob_revenue`, `table_ob_in_obin_revenue`, `table_in_in_decay_revenue`, `table_surplus_capture`) | `uv run ob-learn run ob-ob-revenue`, `ob-in-revenue`, `in-in-revenue-decay`, then `uv run python experiments/build_meta_revenue_summary.py` |

## Layout

```
ob-learn/
├── README.md
├── pyproject.toml           # dependencies + entry point
├── uv.lock                  # pinned resolution
├── LICENSE
├── ob_learn/                # library
├── tests/                   # pytest suite (run with `uv run pytest`)
├── experiments/             # one script per experiment family
└── results/
    ├── figures/             # exported figure files (the shipped paper figures)
    ├── tables/              # exported summary tables
    ├── figdata/             # cached figure-ready curves for `--replot`
    └── <YYYYMMDD-HHMMSS>_<exp>_<hash>/   # per-run artifacts (generated on each run)
```

## Per-run output structure

```
results/<YYYYMMDD-HHMMSS>_<experiment>_<short_hash>/
├── config.yaml + config.json    # full ExperimentConfig snapshot
├── env.txt + git_info.txt       # python/numpy versions, commit + working-tree status
├── run.log                      # human-readable, timestamped
├── events.jsonl                 # phase transitions, regime predictions, fit slopes
├── trajectories/*.npz           # per-seed prices, demands, estimates (compressed)
├── summary/*.csv                # tidy MSE / revenue / slope tables
├── figures/                     # *.pdf and *.png for browsing
└── done.flag                    # written last; aborted runs are detectable
```

## CLI options

```
uv run ob-learn run <key>   [--horizon T] [--seeds S] [--base-seed N] [--quick]
                                 [--replot] [--overwrite] [--workers W]
uv run ob-learn run-all     [--horizon T] [--seeds S] [--base-seed N] [--quick]
                                 [--only KEY ...] [--skip KEY ...] [--continue-on-error]
```

* `--quick` shrinks $T$ and $S$ to a smoke-test size.
* Set `TQDM_DISABLE=1` to silence progress bars in non-interactive shells.


## Citation

```bibtex
@article{wu2026oblivious,
  title   = {Oblivious Learning and Collusive Pricing},
  author  = {Wu, Yuhang and Zeevi, Assaf},
  journal = {arXiv preprint arXiv:2606.05363},
  year    = {2026}
}
```

## License

This project is licensed under the MIT License; see [`LICENSE`](LICENSE).
