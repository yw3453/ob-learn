"""Plot factories used by experiments.

All plots are produced with ``matplotlib`` and return a ``Figure`` object so
callers can :meth:`Run.save_figure` them. The defaults aim for publication-ready
figures: axis labels, a small legend, and consistent sizing. Figures are
deliberately not styled with seaborn or other heavy deps.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from . import market
from .simulator import SimulationResult

# ---------------------------------------------------------------------------
# Report style: no titles, larger font, vector PDF
# ---------------------------------------------------------------------------

SQUARE_FIGSIZE = (5.6, 5.6)
"""Figure size used by default plots.

The *core data area* (axes box) is always forced to a 1:1 aspect ratio via
:func:`square_box`; the plot itself is slightly larger to leave room for
axis labels and the colorbar. The figsize is therefore not strictly square
in width × height -- only the bounding box of the axes is.
"""

REPORT_RC = {
    "font.size": 14.0,
    "axes.labelsize": 16.0,
    "axes.titlesize": 16.0,
    "xtick.labelsize": 13.0,
    "ytick.labelsize": 13.0,
    "legend.fontsize": 13.0,
    "lines.linewidth": 1.6,
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.bbox": "tight",
    "figure.figsize": SQUARE_FIGSIZE,
    "pdf.fonttype": 42,  # TrueType, publication-friendly
    "ps.fonttype": 42,
}


def square_box(ax: plt.Axes, aspect: float = 1.0) -> plt.Axes:
    """Force the axes' bounding box to a fixed aspect ratio (default square).

    Uses :meth:`Axes.set_box_aspect`, which keeps the axes' *physical* box
    fixed regardless of the data ranges, so log-scaled or asymmetric data
    still render in a square frame.
    """
    ax.set_box_aspect(aspect)
    return ax


@contextmanager
def report_style(extra: dict[str, Any] | None = None):
    """Context manager that applies ``REPORT_RC`` (and optional overrides) inside.

    Use around plot-building code that should produce export-ready output::

        with report_style():
            fig, ax = plt.subplots()
            ...  # build the plot; exported figures use caption text, not titles
    """
    rc = dict(REPORT_RC)
    if extra:
        rc.update(extra)
    with mpl.rc_context(rc=rc):  # type: ignore[arg-type]
        yield


def smart_legend(
    ax: plt.Axes,
    *,
    preferred: tuple[str, ...] = ("upper right", "upper left", "lower right", "lower left", "center right"),
    fontsize: int = 12,
    framealpha: float = 0.92,
    **kwargs: Any,
) -> Any:
    """Place a legend at the location with the least overlap with plotted data.

    We score each candidate location by the number of data points (across all
    Line2D / Path artists) that lie inside a 0.28-by-0.28 axis-fraction box at
    that corner, and pick the location with the lowest score. Ties are broken
    by the order in ``preferred``.

    Use this instead of ``ax.legend(loc="best", ...)`` whenever ``"best"``
    visibly overlaps the curves (matplotlib's heuristic is line-based and
    blind to scatter points).
    """
    fig = ax.figure
    fig.canvas.draw_idle()

    LOC_BOX = {
        "upper right":  (0.72, 0.72, 1.00, 1.00),
        "upper left":   (0.00, 0.72, 0.28, 1.00),
        "lower right":  (0.72, 0.00, 1.00, 0.28),
        "lower left":   (0.00, 0.00, 0.28, 0.28),
        "center right": (0.72, 0.36, 1.00, 0.64),
        "center left":  (0.00, 0.36, 0.28, 0.64),
        "upper center": (0.36, 0.72, 0.64, 1.00),
        "lower center": (0.36, 0.00, 0.64, 0.28),
    }
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    log_x = ax.get_xscale() == "log"
    log_y = ax.get_yscale() == "log"

    def _to_frac(xy: np.ndarray) -> np.ndarray:
        """Convert (n, 2) of data coords to axis fraction in [0, 1]^2."""
        x = xy[:, 0]
        y = xy[:, 1]
        if log_x:
            x = np.log10(np.clip(x, 1e-300, None))
            x0 = np.log10(max(xlim[0], 1e-300))
            x1 = np.log10(max(xlim[1], 1e-300))
        else:
            x0, x1 = xlim
        if log_y:
            y = np.log10(np.clip(y, 1e-300, None))
            y0 = np.log10(max(ylim[0], 1e-300))
            y1 = np.log10(max(ylim[1], 1e-300))
        else:
            y0, y1 = ylim
        if x1 == x0 or y1 == y0:
            return np.zeros_like(xy)
        fx = (x - x0) / (x1 - x0)
        fy = (y - y0) / (y1 - y0)
        return np.column_stack([fx, fy])

    pts_list: list[np.ndarray] = []
    for line in ax.lines:
        xy = np.column_stack([line.get_xdata(), line.get_ydata()])
        if xy.size:
            pts_list.append(np.asarray(xy, dtype=np.float64))
    for coll in ax.collections:
        offs = coll.get_offsets()
        try:
            arr = np.asarray(offs)
            if arr.ndim == 2 and arr.shape[1] == 2 and arr.size:
                pts_list.append(arr)
        except Exception:
            pass

    if not pts_list:
        return ax.legend(loc=preferred[0], fontsize=fontsize, framealpha=framealpha, **kwargs)  # type: ignore[call-overload]

    pts = np.vstack(pts_list)
    frac = _to_frac(pts)
    # Keep only points inside the visible axes.
    in_axes = (frac[:, 0] >= 0.0) & (frac[:, 0] <= 1.0) & (frac[:, 1] >= 0.0) & (frac[:, 1] <= 1.0)
    frac = frac[in_axes]

    if frac.size == 0:
        return ax.legend(loc=preferred[0], fontsize=fontsize, framealpha=framealpha, **kwargs)  # type: ignore[call-overload]

    scores: list[tuple[float, int, str]] = []
    for rank, loc in enumerate(preferred):
        if loc not in LOC_BOX:
            continue
        x0, y0, x1, y1 = LOC_BOX[loc]
        inside = (
            (frac[:, 0] >= x0) & (frac[:, 0] <= x1) & (frac[:, 1] >= y0) & (frac[:, 1] <= y1)
        )
        scores.append((float(inside.sum()) / max(frac.shape[0], 1), rank, loc))
    scores.sort()  # ascending overlap, ties by rank
    best_loc = scores[0][2]
    return ax.legend(loc=best_loc, fontsize=fontsize, framealpha=framealpha, **kwargs)  # type: ignore[call-overload]

def _anchor_above(
    curves: Sequence[tuple[np.ndarray, np.ndarray]],
    shape_fn,
    *,
    margin: float = 1.25,
) -> float:
    """Scale factor ``A`` so that ``A*shape_fn(n)`` sits slightly (``margin``x)
    above every curve at its tightest point (and above elsewhere when the data
    tracks the reference shape)."""
    best = 0.0
    for n, y in curves:
        n = np.asarray(n, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        sh = np.asarray(shape_fn(n), dtype=np.float64)
        m = (sh > 1e-12) & np.isfinite(y) & (y > 0)
        if m.any():
            best = max(best, float(np.nanmax(y[m] / sh[m])))
    return margin * max(best, 1e-300)


def regret_reference(
    ax: plt.Axes,
    curves: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    kind: str,
    slope: float | None = None,
) -> str:
    """Overlay a theory reference curve on a cumulative-learning-regret plot.

    ``curves`` is a list of ``(n_axis, curve)`` pairs (any extra leading fields
    should be stripped by the caller). The reference is anchored to the topmost
    curve at the right-hand end of the horizon and drawn as a dashed grey curve;
    it is designed for a *linear*-axis plot (matching the variance-dominance
    regret figures).

    ``kind``:

    * ``"log"``   -- ``propto log n`` (all-oblivious learning regret).
    * ``"sqrt"``  -- ``propto sqrt(n)`` (all-informed learning regret).
    * ``"power"`` -- ``propto n**slope`` for a fitted sublinear ``slope`` (mixed).

    Returns the legend label for the reference so the caller can add a handle.
    """
    if not curves:
        return ""
    n_hi = max(float(np.asarray(n).max()) for n, _ in curves)
    n_lo = min(float(np.asarray(n).min()) for n, _ in curves)
    if kind == "log":
        def shape_fn(x):
            return np.log(np.maximum(np.asarray(x, float), 1.0)) / np.log(max(n_hi, 1.0 + 1e-9))
        label = r"reference $\propto \log n$"
    elif kind == "sqrt":
        def shape_fn(x):
            return np.sqrt(np.asarray(x, float) / n_hi)
        label = r"reference $\propto \sqrt{n}$"
    elif kind == "power":
        s = float(slope) if slope is not None else 0.5

        def shape_fn(x):
            return (np.asarray(x, float) / n_hi) ** s
        label = rf"reference $\propto n^{{{s:.2f}}}$"
    else:
        raise ValueError(f"unknown reference kind {kind!r}")
    n_ref = np.geomspace(max(n_lo, 1.0), n_hi, num=256)
    A = _anchor_above(curves, shape_fn)
    ax.plot(n_ref, A * shape_fn(n_ref), color="0.30", lw=1.1, linestyle=(0, (4, 3)),
            label=label, zorder=1)
    return label


def truncate_cells(cells: Sequence[dict], max_n: float | None) -> list[dict]:
    """Return cells with all ``n``-aligned arrays cropped to ``n <= max_n``.

    A display-only helper: the cached curves run to the full simulation horizon,
    but the figures can show a shorter window (e.g. before slow finite-horizon
    transients appear). Cells without an ``"n"`` key (e.g. scatter points) and
    arrays whose length differs from ``n`` are passed through untouched.
    """
    if max_n is None:
        return list(cells)
    out: list[dict] = []
    for cell in cells:
        if "n" not in cell:
            out.append(cell)
            continue
        n = np.asarray(cell["n"])
        mask = n <= float(max_n)
        new = {"params": cell.get("params", {})}
        for key, val in cell.items():
            if key == "params":
                continue
            arr = np.asarray(val)
            new[key] = arr[mask] if arr.shape == n.shape else arr
        out.append(new)
    return out


def perperiod_reference(
    ax: plt.Axes,
    curves: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    slope: float,
) -> str:
    """Overlay a decaying power-law reference ``propto n^slope`` (``slope < 0``)
    for a *per-period* learning-regret log-log plot.

    Anchored to sit slightly above all curves (same slope), which reads more
    clearly as a rate guide than a line through the cloud. Returns the label.
    """
    if not curves:
        return ""
    n_hi = max(float(np.asarray(n).max()) for n, _ in curves)
    n_lo = min(float(np.asarray(n).min()) for n, _ in curves)

    def shape_fn(x):
        return np.asarray(x, dtype=np.float64) ** slope

    n_ref = np.geomspace(max(n_lo, 1.0), n_hi, num=256)
    A = _anchor_above(curves, shape_fn)
    label = rf"reference $\propto n^{{{slope:.2g}}}$"
    ax.plot(n_ref, A * shape_fn(n_ref), color="0.30", lw=1.1, linestyle=(0, (4, 3)),
            label=label, zorder=1)
    return label


def style_regret_axes(ax: plt.Axes, *, xlabel: str = r"$n$",
                      ylabel: str = r"cumulative learning regret") -> None:
    """Apply the linear cumulative-regret axis style (cf. variance dominance)."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.margins(x=0)
    ax.set_ylim(bottom=0.0)
    ax.ticklabel_format(axis="both", style="sci", scilimits=(-3, 4))
    square_box(ax)


def loglog_regret_axes(ax: plt.Axes, *, xlabel: str = r"$n$",
                       ylabel: str = r"cumulative learning regret") -> None:
    """Log-log axis: a power law ``propto n^a`` is a straight line of slope ``a``.

    Used for per-period learning regret (decaying, slope ``< 0``) and for
    cumulative polynomial regret (mixed / all-informed)."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    square_box(ax)


def plot_sample_paths(
    result: SimulationResult,
    *,
    n_paths: int = 5,
    sellers: Sequence[int] | None = None,
    title: str | None = None,
    with_benchmarks: bool = True,
) -> plt.Figure:
    """Plot a handful of sample-path price trajectories with NE / collusive lines."""
    if sellers is None:
        sellers = list(range(result.config.market.N))
    n_paths = min(n_paths, result.config.n_seeds)
    n = result.log_steps
    fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
    cmap = plt.get_cmap("tab10")
    for k, seller in enumerate(sellers):
        for s in range(n_paths):
            ax.plot(
                n,
                result.prices[:, seller, s],
                color=cmap(k % 10),
                lw=0.8,
                alpha=0.7,
                label=f"seller {seller}" if s == 0 else None,
            )
    if with_benchmarks:
        d = result.config.market
        p_NE = market.nash_prices(d)
        p_C = market.collusive_prices(d)
        ax.axhline(float(np.mean(p_NE)), color="tab:red", linestyle="--", lw=1.0, label=r"$p^{NE}$")
        ax.axhline(float(np.mean(p_C)), color="tab:green", linestyle="--", lw=1.0, label=r"$p^{C}$")
    ax.set_xlabel("n")
    ax.set_ylabel("price")
    if title:
        ax.set_title(title)
    smart_legend(ax)
    square_box(ax)
    fig.tight_layout()
    return fig


def plot_cumulative_revenue(
    result: SimulationResult,
    *,
    title: str | None = None,
    show_quantiles: bool = True,
) -> plt.Figure:
    """Plot cumulative revenue per seller with ``T * Pi_NE`` and ``T * Pi_C`` overlays."""
    from . import benchmarks

    d = result.config.market
    cum = benchmarks.cumulative_revenue(result)
    n = result.log_steps + 1.0
    pi = benchmarks.benchmark_per_period_revenues(d)
    fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
    cmap = plt.get_cmap("tab10")
    for i in range(d.N):
        mean = cum[:, i, :].mean(axis=1)
        ax.plot(n - 1, mean, color=cmap(i % 10), lw=1.5, label=f"seller {i}")
        if show_quantiles:
            p25 = np.percentile(cum[:, i, :], 25, axis=1)
            p75 = np.percentile(cum[:, i, :], 75, axis=1)
            ax.fill_between(n - 1, p25, p75, alpha=0.15, color=cmap(i % 10))
    # Per-seller benchmark lines (averaged across sellers since they may differ).
    for label, vec in pi.items():
        avg = float(np.mean(vec))
        ax.plot(n - 1, n * avg, linestyle="--", lw=1.0, label=fr"$T \cdot \Pi_{{{label}}}$")
    ax.set_xlabel("n")
    ax.set_ylabel("cumulative revenue")
    if title:
        ax.set_title(title)
    smart_legend(ax, fontsize=10)
    square_box(ax)
    fig.tight_layout()
    return fig


