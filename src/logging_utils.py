"""Run-directory creation and structured logging.

A "run" is one execution of an experiment script; it produces a single
self-contained directory under ``results/`` containing the configuration,
console + structured logs, raw trajectories, summary CSVs, and figures. The
:class:`Run` context manager handles directory layout, atomic ``done.flag``
emission, and a few convenience methods that experiment scripts use.

Layout::

    results/<YYYYMMDD-HHMMSS>_<experiment>_<short_hash>/
      config.yaml
      config.json
      env.txt
      git_info.txt
      run.log
      events.jsonl
      trajectories/*.npz
      summary/*.csv
      figures/*.{pdf,png}
      done.flag

Anything written through the :class:`Run` API is mirrored to ``run.log``
(human-readable) and ``events.jsonl`` (machine-readable, one JSON per line).
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]
from rich.logging import RichHandler

from .config import ExperimentConfig

# ``code/src/logging_utils.py`` -> ``parents[1] == code/``.
CODE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = CODE_DIR / "results"


def _short_hash(payload: str, n: int = 8) -> str:
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:n]


def _git_info(cwd: Path) -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if commit.returncode != 0:
            return "(not a git repository)"
        head = commit.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        return f"HEAD: {head}\n\nstatus:\n{status.stdout.strip() or '(clean)'}\n"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "(git unavailable)"


def _env_info() -> str:
    pieces = [
        f"python: {sys.version.replace(chr(10), ' ')}",
        f"platform: {platform.platform()}",
        f"machine: {platform.machine()}",
    ]
    for mod_name in ("numpy", "scipy", "pandas", "matplotlib", "pydantic"):
        try:
            mod = __import__(mod_name)
            pieces.append(f"{mod_name}: {getattr(mod, '__version__', '(unknown)')}")
        except ImportError:
            pieces.append(f"{mod_name}: (not installed)")
    return "\n".join(pieces) + "\n"


@dataclass
class Run:
    """Handle to a single run directory.

    Created by :func:`run_directory`; experiment scripts use the convenience
    methods (:meth:`save_trajectory`, :meth:`save_summary`, :meth:`save_figure`,
    :meth:`log_event`) rather than touching the file system directly.
    """

    name: str
    directory: Path
    logger: logging.Logger
    config: ExperimentConfig
    _events_path: Path = field(repr=False)
    _start_time: float = field(default_factory=time.monotonic, repr=False)

    @property
    def trajectories_dir(self) -> Path:
        return self.directory / "trajectories"

    @property
    def summary_dir(self) -> Path:
        return self.directory / "summary"

    @property
    def figures_dir(self) -> Path:
        return self.directory / "figures"

    def log_event(self, event: str, **fields: Any) -> None:
        """Append a JSON event to ``events.jsonl`` and mirror to ``run.log``."""
        record: dict[str, Any] = {
            "ts": _dt.datetime.now(_dt.UTC).isoformat(timespec="milliseconds"),
            "event": event,
        }
        record.update(_jsonify(fields))
        with self._events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        extras = {k: v for k, v in record.items() if k not in {"ts", "event"}}
        self.logger.info("event %s %s", event, json.dumps(extras))

    def save_trajectory(self, name: str, **arrays: np.ndarray) -> Path:
        """Save one or more arrays under ``trajectories/<name>.npz`` (compressed)."""
        path = self.trajectories_dir / f"{name}.npz"
        payload: dict[str, Any] = dict(arrays)
        np.savez_compressed(path, **payload)
        self.logger.info("saved trajectory %s (%s)", path, _human_size(path))
        return path

    def save_summary(self, name: str, df: pd.DataFrame) -> Path:
        path = self.summary_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        self.logger.info("saved summary %s (%d rows)", path, len(df))
        return path

    def save_figure(
        self,
        name: str,
        fig: Any,
        formats: tuple[str, ...] = ("pdf", "png"),
        close: bool = True,
    ) -> list[Path]:
        paths: list[Path] = []
        for ext in formats:
            path = self.figures_dir / f"{name}.{ext}"
            fig.savefig(path, bbox_inches="tight")
            paths.append(path)
        self.logger.info("saved plot %s (%s)", name, ", ".join(p.suffix.lstrip(".") for p in paths))
        if close:
            try:
                import matplotlib.pyplot as _plt

                _plt.close(fig)
            except Exception:  # noqa: BLE001
                pass
        return paths

    @property
    def elapsed_sec(self) -> float:
        return time.monotonic() - self._start_time


@contextmanager
def run_directory(
    name: str,
    config: ExperimentConfig,
    *,
    base_dir: Path | None = None,
):
    """Context manager that creates a results directory and a configured logger.

    The directory is named ``<YYYYMMDD-HHMMSS>_<name>_<hash>`` where the hash is
    derived from the JSON-serialized config (so the same config produces a
    directory whose tail can be diffed across runs). The ``done.flag`` file is
    written only when the context exits normally.
    """
    base = base_dir or DEFAULT_RESULTS_DIR
    base.mkdir(parents=True, exist_ok=True)
    cfg_payload = config.model_dump_json()
    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    short = _short_hash(cfg_payload)
    directory = base / f"{timestamp}_{name}_{short}"
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "trajectories").mkdir()
    (directory / "summary").mkdir()
    (directory / "figures").mkdir()

    cfg_dict = config.model_dump(mode="json")
    (directory / "config.json").write_text(json.dumps(cfg_dict, indent=2))
    (directory / "config.yaml").write_text(yaml.safe_dump(cfg_dict, sort_keys=False))
    (directory / "env.txt").write_text(_env_info())
    (directory / "git_info.txt").write_text(_git_info(CODE_DIR))

    logger, file_handler = _build_logger(name, directory / "run.log")
    events_path = directory / "events.jsonl"
    events_path.touch()

    run = Run(
        name=name,
        directory=directory,
        logger=logger,
        config=config,
        _events_path=events_path,
    )
    logger.info("starting run %s in %s", name, directory)

    success = False
    try:
        yield run
        success = True
    except Exception:  # noqa: BLE001
        logger.exception("run %s failed", name)
        raise
    finally:
        elapsed = run.elapsed_sec
        logger.info("run %s finished in %.2f s (success=%s)", name, elapsed, success)
        if success:
            (directory / "done.flag").write_text(f"elapsed_sec={elapsed:.6f}\n")
        logger.removeHandler(file_handler)
        with contextlib.suppress(Exception):
            file_handler.close()


def _build_logger(name: str, log_path: Path) -> tuple[logging.Logger, logging.Handler]:
    logger = logging.getLogger(f"src.{name}.{log_path.parent.name}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    # Clear any pre-existing handlers (re-runs in interactive sessions).
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logger.addHandler(fh)

    ch = RichHandler(
        level=os.environ.get("OB_LEARN_LOG_LEVEL", "INFO"),
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
    )
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    return logger, fh


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _human_size(path: Path) -> str:
    try:
        nbytes = float(path.stat().st_size)
    except FileNotFoundError:
        return "?"
    for unit in ("B", "KiB", "MiB", "GiB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TiB"
