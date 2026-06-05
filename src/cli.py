"""Command-line entry point.

Each ``experiments/exp_*.py`` script defines a ``main(...)`` function. The CLI
maintains a curated registry with descriptive experiment IDs.

Usage::

    uv run ob-learn list
    uv run ob-learn run mixed-forecast-rules --horizon 50000 --seeds 100
    uv run ob-learn run ob-ob-revenue --horizon 20000 --seeds 50 --quick
    uv run ob-learn run-all                # run the full suite
    uv run ob-learn run-all --quick        # smoke-test the full suite
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import traceback
from pathlib import Path

# ``<repo>/src/cli.py`` -> ``parents[1]`` is the repo root.
CODE_DIR = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = CODE_DIR / "experiments"


EXPERIMENT_REGISTRY: list[dict[str, str]] = [
    {"id": "mixed-forecast-rules", "file": "exp_mixed_forecast_rules.py"},
    {"id": "ob-ob-revenue", "file": "exp_ob_ob_revenue.py"},
    {"id": "ob-in-revenue", "file": "exp_ob_in_revenue.py"},
    {"id": "in-in-revenue-decay", "file": "exp_in_in_revenue_decay.py"},
    {"id": "variance-dominance", "file": "exp_variance_dominance.py"},
    {"id": "variance-dominance-relocated-box", "file": "exp_variance_dominance_relocated_box.py"},
    {"id": "excursion-dynamics", "file": "exp_excursion_dynamics.py"},
    {"id": "threshold-curve", "file": "exp_threshold_curve.py"},
    {"id": "dominance-margin", "file": "exp_dominance_margin.py"},
    {"id": "asymmetric-pseudoequilibria-continuum", "file": "exp_asymmetric_pseudoequilibria_continuum.py"},
    {"id": "symmetric-pseudoequilibria-continuum", "file": "exp_symmetric_pseudoequilibria_continuum.py"},
    {"id": "gaussian-dither-robustness", "file": "exp_gaussian_dither_robustness.py"},
    {"id": "asymmetric-multiseller", "file": "exp_asymmetric_multiseller.py"},
    {"id": "all-informed-stress", "file": "exp_all_informed_stress.py"},
    {"id": "mixed-small-gain", "file": "exp_mixed_small_gain.py"},
    {"id": "mixed-small-gain-iv-holds", "file": "exp_mixed_small_gain_iv_holds.py"},
    {"id": "mixed-multiseller", "file": "exp_mixed_multiseller.py"},
    {"id": "mixed-revenue-ordering", "file": "exp_mixed_revenue_ordering.py"},
]

RUN_ALL_ORDER: list[str] = [str(item["id"]) for item in EXPERIMENT_REGISTRY]


def _discover_experiments() -> dict[str, Path]:
    canonical: dict[str, Path] = {}
    for item in EXPERIMENT_REGISTRY:
        exp_id = item["id"]
        path = EXPERIMENTS_DIR / item["file"]
        if not path.exists():
            continue
        canonical[exp_id] = path
    return canonical


def _load_module(path: Path):
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ob-learn", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list available experiments")

    run = sub.add_parser("run", help="run an experiment")
    run.add_argument("name", help="experiment ID")
    run.add_argument("--horizon", type=int, default=None)
    run.add_argument("--seeds", type=int, default=None)
    run.add_argument("--base-seed", type=int, default=None)
    run.add_argument("--quick", action="store_true", help="quick smoke run with reduced T and S")
    run_all = sub.add_parser(
        "run-all",
        help="run every experiment in registry order",
    )
    run_all.add_argument("--horizon", type=int, default=None)
    run_all.add_argument("--seeds", type=int, default=None)
    run_all.add_argument("--base-seed", type=int, default=None)
    run_all.add_argument("--quick", action="store_true", help="smoke-test mode for every experiment")
    run_all.add_argument(
        "--continue-on-error",
        action="store_true",
        help="if one experiment fails, log the traceback and move on",
    )
    run_all.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="restrict to a subset of experiment IDs",
    )
    run_all.add_argument(
        "--skip",
        nargs="+",
        default=None,
        help="skip these experiment names",
    )

    args = parser.parse_args(argv)
    experiments = _discover_experiments()

    if args.cmd == "list":
        if not experiments:
            print("(no experiments found in experiments/)")
            return 0
        for name in sorted(experiments):
            print(f"  {name}\t{experiments[name].name}")
        return 0

    if args.cmd == "run":
        if args.name not in experiments:
            known = sorted(experiments)
            print(f"unknown experiment {args.name!r}; known: {known}", file=sys.stderr)
            return 2
        mod = _load_module(experiments[args.name])
        kwargs: dict[str, object] = {}
        if args.horizon is not None:
            kwargs["horizon"] = args.horizon
        if args.seeds is not None:
            kwargs["n_seeds"] = args.seeds
        if args.base_seed is not None:
            kwargs["base_seed"] = args.base_seed
        if args.quick:
            kwargs["quick"] = True
        mod.main(**kwargs)
        return 0

    if args.cmd == "run-all":
        names = [n for n in RUN_ALL_ORDER if n in experiments]
        if args.only:
            requested = set(args.only)
            unknown = requested - set(experiments)
            if unknown:
                print(f"unknown experiment(s): {sorted(unknown)}", file=sys.stderr)
                return 2
            names = [n for n in names if n in requested]
        if args.skip:
            names = [n for n in names if n not in set(args.skip)]
        if not names:
            print("no experiments to run", file=sys.stderr)
            return 2

        run_all_kwargs: dict[str, object] = {}
        if args.horizon is not None:
            run_all_kwargs["horizon"] = args.horizon
        if args.seeds is not None:
            run_all_kwargs["n_seeds"] = args.seeds
        if args.base_seed is not None:
            run_all_kwargs["base_seed"] = args.base_seed
        if args.quick:
            run_all_kwargs["quick"] = True

        total = len(names)
        t0 = time.monotonic()
        failures: list[tuple[str, str]] = []
        for i, name in enumerate(names, 1):
            header = f"[{i}/{total}] ob-learn run {name}"
            bar = "=" * max(20, 80 - len(header) - 1)
            print(f"\n{header} {bar}", flush=True)
            try:
                mod = _load_module(experiments[name])
                mod.main(**run_all_kwargs)
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                msg = f"experiment {name!r} failed: {exc.__class__.__name__}: {exc}"
                print(msg, file=sys.stderr)
                if args.continue_on_error:
                    print(tb, file=sys.stderr)
                    failures.append((name, msg))
                    continue
                raise
        elapsed = time.monotonic() - t0
        print(f"\nrun-all done in {elapsed:.1f}s ({total} experiment(s))", flush=True)
        if failures:
            print(f"FAILED: {len(failures)} experiment(s):", file=sys.stderr)
            for name, msg in failures:
                print(f"  {name}: {msg}", file=sys.stderr)
            return 1
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
