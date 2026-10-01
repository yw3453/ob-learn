"""Lightweight cache of *figure-ready* data for the regret experiments.

Each experiment reduces its simulation runs to a handful of per-cell curves
(seed-mean cumulative learning regret, percentile bands, and a couple of
companion curves) and stores them under ``results/figdata/<exp_id>.npz`` plus a
JSON sidecar with the per-cell parameters. This lets the plotting layer redraw a
figure (axis style, legend, references, horizon view) without re-running the
(expensive) simulations. The cached arrays are seed-reduced, so the whole suite
is only a few megabytes.

A "cell" is a plain ``dict`` with a JSON-serialisable ``"params"`` entry plus
any number of 1-D (or small 2-D) numpy arrays, e.g.::

    {"params": {"gamma": 0.1, "nu2": 0.05},
     "n": n_grid, "regret": reg_mean, "regret_p25": ..., "regret_p75": ...,
     "regret_realized": ..., "mse": ...}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
FIGDATA_DIR = CODE_DIR / "results" / "figdata"


def figdata_paths(exp_id: str, *, base: Path | None = None) -> tuple[Path, Path]:
    d = base or FIGDATA_DIR
    return d / f"{exp_id}.npz", d / f"{exp_id}.json"


def figdata_exists(exp_id: str, *, base: Path | None = None) -> bool:
    npz, meta = figdata_paths(exp_id, base=base)
    return npz.exists() and meta.exists()


def save_figdata(exp_id: str, cells: list[dict], *, base: Path | None = None) -> Path:
    """Persist a list of cell dicts to ``results/figdata/<exp_id>.{npz,json}``.

    Every non-``"params"`` value in each cell is stored as an array; the
    ``"params"`` dicts (JSON-serialisable scalars/strings) go to the sidecar.
    Returns the path to the ``.npz`` file.
    """
    npz_path, meta_path = figdata_paths(exp_id, base=base)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    meta: list[dict] = []
    for k, cell in enumerate(cells):
        meta.append(dict(cell.get("params", {})))
        for key, val in cell.items():
            if key == "params":
                continue
            arrays[f"c{k}__{key}"] = np.asarray(val, dtype=np.float64)
    np.savez_compressed(npz_path, **arrays)  # type: ignore[arg-type]
    meta_path.write_text(json.dumps({"exp_id": exp_id, "n_cells": len(cells),
                                     "params": meta}, indent=2))
    return npz_path


def load_figdata(exp_id: str, *, base: Path | None = None) -> list[dict]:
    """Load the cell dicts previously saved by :func:`save_figdata`."""
    npz_path, meta_path = figdata_paths(exp_id, base=base)
    meta = json.loads(meta_path.read_text())
    with np.load(npz_path) as npz:
        files = list(npz.files)
        cells: list[dict] = []
        for k in range(int(meta["n_cells"])):
            cell: dict = {"params": meta["params"][k]}
            prefix = f"c{k}__"
            for key in files:
                if key.startswith(prefix):
                    cell[key[len(prefix):]] = npz[key]
            cells.append(cell)
    return cells
