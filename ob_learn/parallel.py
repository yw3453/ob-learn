"""Cell-level parallelism for the experiment sweeps.

Each parameter cell of a sweep is an independent, deterministic simulation, so
the sweep is parallelizable. :func:`map_cells` runs a picklable worker
over a list of plain-dict specs using a process pool (default 4 workers, the
core count of the reference machine). On Linux the pool uses the ``fork`` start
method, so worker processes inherit the parent's imported modules and
``sys.path``; the worker function is resolved by re-import in the child.

Set ``max_workers=1`` to run serially (useful for debugging / quick smoke runs).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

DEFAULT_WORKERS = min(4, os.cpu_count() or 1)


def map_cells(
    worker: Callable[[dict], Any],
    specs: Sequence[dict],
    *,
    max_workers: int = DEFAULT_WORKERS,
    logger: Any | None = None,
) -> list[Any]:
    """Apply ``worker`` to each spec, preserving input order.

    ``worker`` must be a top-level (picklable) function taking a single
    plain-dict spec and returning a small, picklable result. Results are
    returned in the same order as ``specs``.
    """
    specs = list(specs)
    n = len(specs)
    if max_workers <= 1 or n <= 1:
        out = []
        for i, s in enumerate(specs):
            out.append(worker(s))
            if logger is not None:
                logger.info("cell %d/%d done (serial)", i + 1, n)
        return out

    results: list[Any] = [None] * n
    with ProcessPoolExecutor(max_workers=min(max_workers, n)) as ex:
        fut_to_idx = {ex.submit(worker, spec): i for i, spec in enumerate(specs)}
        done = 0
        for fut in as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            results[idx] = fut.result()
            done += 1
            if logger is not None:
                logger.info("cell %d/%d done", done, n)
    return results
