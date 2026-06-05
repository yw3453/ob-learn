# Oblivious Learning and Algorithmic Collusion

This repository contains code for the paper **"Should Demand Models Incorporate Competitor Prices? Oblivious Learning and Algorithmic Collusion"** ([arXiv](https://arxiv.org/abs/2606.05363)).

The codebase combines a reusable simulation library in [`src/`](src/) with a
suite of experiment scripts in [`experiments/`](experiments/). Runs produce
timestamped artifacts under [`results/`](results/), including logs, compressed
trajectories, summary tables, and generated figures.

Licensed under the [MIT License](LICENSE). Python 3.13+, all dependencies
pinned via [`uv`](https://github.com/astral-sh/uv).

## Quick start

```bash
# Install dependencies (uv resolves them from uv.lock).
uv sync

# Optional: verify the install with the unit tests (~5s).
uv run pytest

# List discoverable experiments.
uv run ob-learn list

# Run a single experiment.
uv run ob-learn run mixed-forecast-rules --quick
uv run ob-learn run mixed-forecast-rules

# Run the full benchmark suite.
uv run ob-learn run-all
```

Each invocation writes a self-contained directory to `results/` with the
config snapshot, per-seed trajectories, summary CSVs, and PDFs. After running
a meta-game block (`ob-ob-revenue`, `ob-in-revenue`, `in-in-revenue-decay`,
`variance-dominance`), compile the cross-experiment
revenue summary with:

```bash
uv run python experiments/build_meta_revenue_summary.py
```

## Layout

```
.
├── README.md
├── LICENSE
├── pyproject.toml           # dependencies + entry point
├── uv.lock                  # pinned resolution
├── src/                     # library
│   └── tests/               # pytest suite (run with `uv run pytest`)
├── experiments/             # one script per experiment family
└── results/
    ├── figures/             # exported figure files
    ├── tables/              # exported summary tables
    └── <YYYYMMDD-HHMMSS>_<exp>_<hash>/   # per-run artifacts
```

## Library (`src/`)

| Module             | Role |
| ------------------ | ---- |
| `config.py`        | Pydantic-validated `DemandParams`, `ProjectionBox`, `ExplorationSchedule`, `SellerSpec`, `ExperimentConfig`. Enforces dominance and projection-box conditions. |
| `market.py`        | Closed-form Nash, collusive, and Stackelberg prices; pseudo-true oblivious estimates; threshold constants (`gamma_bar`, `L_phi_oblivious`, `C_x_oblivious`). |
| `exploration.py`   | `nu_at(schedule, n)` and `sample(...)`; constant, polynomial $\nu_n^2 = c n^{-\eta}$, $\sqrt n$, and Gaussian-clip variants. |
| `estimators.py`    | Vectorised iterated OLS for oblivious (2-D) and informed ($N+1$-D) regressions; projection onto the box; greedy revenue-maximising price. |
| `sellers.py`       | Plug-in forecast rules: `mean_price`, `perfect_prediction`, `greedy_component`, `lag1`, `oracle_nash`. |
| `simulator.py`     | Vectorised-over-seeds main loop. Warm-up that guarantees full-rank designs; deferred-forecast ordering for Stackelberg-style informed sellers; structured per-step logging. |
| `analysis.py`      | Log-log slope fits, predicted-rate utilities, regression-ratio and surplus-capture estimators. |
| `benchmarks.py`    | Cumulative revenue, Nash/collusive/Stackelberg references, cross-seed bands. |
| `plotting.py`      | `plot_mse_loglog`, `plot_sample_paths`, `plot_threshold_heatmap`, `plot_excursion_overlay`, etc. Each saves PDF + PNG and closes figures to bound memory across long sweeps. |
| `ode.py`           | Discrete-time and continuous-time $(m, Q)$ dynamics used by the excursion experiment. |
| `artifact_export.py` | Shared utilities that export figures/tables to `results/figures/` and `results/tables/`. |
| `logging_utils.py` | `run_directory(...)` context manager: creates timestamped output dir, dumps configs/env, opens a `RichHandler` logger, writes `done.flag` last. |
| `cli.py`           | The `ob-learn` entry point (`list`, `run`, `run-all`). |

## Experiments

Run `uv run ob-learn list` to see the available configurations.

| Configuration | Focus |
| ------------ | ----- |
| `mixed-forecast-rules` | Mixed-market forecast-rule comparison. |
| `ob-ob-revenue` | Oblivious-oblivious revenue regimes. |
| `ob-in-revenue` | Oblivious-informed revenue regimes. |
| `in-in-revenue-decay` | Informed-informed decaying exploration. |
| `variance-dominance` | Dominance under asymmetric exploration rates. |
| `variance-dominance-relocated-box` | Dominance robustness under a relocated projection box. |
| `excursion-dynamics` | ODE and discrete excursion dynamics. |
| `threshold-curve` | Regime transition along exploration variance grid. |
| `dominance-margin` | Dominance-margin stress sweep. |
| `asymmetric-pseudoequilibria-continuum` | Asymmetric continuum of pseudo-equilibria analysis. |
| `symmetric-pseudoequilibria-continuum` | Symmetric continuum of pseudo-equilibria analysis. |
| `gaussian-dither-robustness` | Robustness to Gaussian-clip dithering. |
| `asymmetric-multiseller` | Asymmetric multi-seller stress tests. |
| `all-informed-stress` | All-informed market stress tests. |
| `mixed-small-gain` | Mixed-market small-gain stress test. |
| `mixed-small-gain-iv-holds` | Mixed-market small-gain variant where the denominator condition holds. |
| `mixed-multiseller` | Mixed-market multi-seller stress tests. |
| `mixed-revenue-ordering` | Revenue ordering in mixed strategy populations. |

## Per-run output structure

Every experiment writes a directory under `results/`:

```
results/<YYYYMMDD-HHMMSS>_<experiment>_<short_hash>/
├── config.yaml + config.json    # full ExperimentConfig snapshot
├── env.txt + git_info.txt       # python/numpy versions, commit + diff
├── run.log                      # human-readable, timestamped
├── events.jsonl                 # phase transitions, regime predictions, fit slopes
├── trajectories/*.npz           # per-seed prices, demands, estimates (compressed)
├── summary/*.csv                # tidy MSE / revenue / slope tables
├── figures/                     # *.pdf and *.png for browsing
└── done.flag                    # written last; aborted runs are detectable
```

Named figures and summary tables are additionally exported to
`results/figures/` / `results/tables/` via
`artifact_export.export_figure(...)` and `artifact_export.export_table(...)`.

## CLI options

```
uv run ob-learn run <key>   [--horizon T] [--seeds S] [--base-seed N] [--quick]
uv run ob-learn run-all     [--horizon T] [--seeds S] [--base-seed N] [--quick]
                                 [--only KEY ...] [--skip KEY ...] [--continue-on-error]
```

* `--quick` shrinks $T$ and $S$ to a smoke-test size (per-script tuned).
* Set `OB_LEARN_LOG_LEVEL=DEBUG` for verbose console logs without
  affecting the file logs.
* Set `TQDM_DISABLE=1` to silence progress bars even in interactive shells.

## Dependencies

Core: `numpy`, `scipy`, `pandas`, `matplotlib`, `pydantic`, `pyyaml`, `tqdm`,
`rich`. Dev: `pytest`, `pytest-cov`, `ruff`, `mypy`. All are pinned in
`pyproject.toml` / `uv.lock` and installed via `uv sync`. Python 3.13 is
required.

## Tests

```bash
uv run pytest                # ~5s
uv run ruff check .          # lint
uv run mypy src/             # static types
```

The test suite covers Nash / collusive / Stackelberg closed forms, OLS
recovery on synthetic data, simulator invariants (price always in $[l, u]$,
full-rank designs after warm-up), and revenue identities at the Nash and
collusive benchmarks.

## Citation

```bibtex
@misc{wu2026demandmodelsincorporatecompetitor,
      title={Should Demand Models Incorporate Competitor Prices? Oblivious Learning and Algorithmic Collusion}, 
      author={Yuhang Wu and Assaf Zeevi},
      year={2026},
      eprint={2606.05363},
      archivePrefix={arXiv},
      primaryClass={cs.GT},
      url={https://arxiv.org/abs/2606.05363}, 
}
```

## License

MIT — see [LICENSE](LICENSE).
