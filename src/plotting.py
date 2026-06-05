"""Plot factories used by experiments.

All plots are produced with ``matplotlib`` and return a ``Figure`` object so
callers can :meth:`Run.save_figure` them. The defaults aim for publication-ready
figures: axis labels, a small legend, and consistent sizing. Figures are
deliberately not styled with seaborn or other heavy deps.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from . import analysis, market
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
            fig = plot_excursion_overlay(...)
            fig.gca().set_title("")  # exported figures use caption text
    """
    rc = dict(REPORT_RC)
    if extra:
        rc.update(extra)
    with mpl.rc_context(rc=rc):
        yield


def strip_titles(fig: plt.Figure) -> plt.Figure:
    """Remove every axis title (captions carry context instead)."""
    for ax in fig.axes:
        ax.set_title("")
    return fig


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
        return ax.legend(loc=preferred[0], fontsize=fontsize, framealpha=framealpha, **kwargs)

    pts = np.vstack(pts_list)
    frac = _to_frac(pts)
    # Keep only points inside the visible axes.
    in_axes = (frac[:, 0] >= 0.0) & (frac[:, 0] <= 1.0) & (frac[:, 1] >= 0.0) & (frac[:, 1] <= 1.0)
    frac = frac[in_axes]

    if frac.size == 0:
        return ax.legend(loc=preferred[0], fontsize=fontsize, framealpha=framealpha, **kwargs)

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
    return ax.legend(loc=best_loc, fontsize=fontsize, framealpha=framealpha, **kwargs)

def plot_mse_loglog(
    result: SimulationResult,
    *,
    metric: str = "price",
    title: str | None = None,
    overlay_predicted_rate: bool = True,
) -> plt.Figure:
    """Log-log plot of MSE vs ``n`` with predicted-rate overlay.

    ``metric`` is one of ``"price"``, ``"theta_ob"``, ``"theta_in"``.
    """
    n = np.asarray(result.log_steps, dtype=np.float64) + 1.0
    if metric == "price":
        curve = analysis.mse_price(result)
        ylabel = r"MSE($\tilde p_n$)"
    elif metric == "theta_ob":
        curve = analysis.mse_theta_oblivious(result)
        ylabel = r"MSE($\hat\theta^{ob}_n$)"
    elif metric == "theta_in":
        curve = analysis.mse_theta_informed(result)
        ylabel = r"MSE($\hat\theta^{in}_n$)"
    else:
        raise ValueError(metric)

    fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
    mean = curve.mean(axis=1)
    p25 = np.percentile(curve, 25, axis=1)
    p75 = np.percentile(curve, 75, axis=1)
    ax.fill_between(n, p25, p75, alpha=0.2, color="tab:blue", label="25--75%ile")
    ax.plot(n, mean, color="tab:blue", lw=1.5, label="mean")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if overlay_predicted_rate:
        # Use the first oblivious schedule's nu^2 at the tail to predict rate.
        if result.config.market.N >= 2:
            seller_specs = result.config.sellers
            ob_idx = result.ob_seller_idx
            if ob_idx.size > 0:
                from .exploration import nu_at  # local import avoids cycle

                first_ob = int(ob_idx[0])
                nu_tail = nu_at(seller_specs[first_ob].exploration, int(n[-1]))
                info = analysis.predicted_rates(
                    result.config.market,
                    result.config.oblivious_projection,
                    nu_squared=nu_tail**2,
                )
                slope = float(info["slope"])
                regime = info["regime"]
                if np.isfinite(slope):
                    j = len(n) - 1
                    y_anchor = max(float(mean[j]), 1e-12)
                    n_anchor = float(n[j])
                    ref = y_anchor * (n / n_anchor) ** slope
                    ax.plot(n, ref, color="black", lw=1.0, linestyle=":", label=f"slope={slope:.2f}")
                    del regime  # not surfaced in the legend

    smart_legend(ax)
    square_box(ax)
    fig.tight_layout()
    return fig


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


def plot_threshold_heatmap(
    xs: np.ndarray,
    ys: np.ndarray,
    z: np.ndarray,
    *,
    xlabel: str = "x",
    ylabel: str = "y",
    title: str | None = None,
    cbar_label: str | None = None,
    log_color: bool = True,
) -> plt.Figure:
    """Heatmap of ``z[i, j]`` over a 2D parameter sweep."""
    from matplotlib.colors import LogNorm

    fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
    norm = LogNorm(vmin=max(np.nanmin(z), 1e-12), vmax=np.nanmax(z)) if log_color else None
    im = ax.pcolormesh(xs, ys, z, shading="auto", cmap="viridis", norm=norm)
    cbar = fig.colorbar(im, ax=ax)
    if cbar_label:
        cbar.set_label(cbar_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    square_box(ax)
    fig.tight_layout()
    return fig


def plot_price_scatter(
    result: SimulationResult,
    *,
    sellers: tuple[int, int] = (0, 1),
    use_running_mean: bool = True,
    title: str | None = None,
    mark_stackelberg: bool = False,
    zoom_to_data: bool = False,
) -> plt.Figure:
    """Scatter of long-run ``(p_1, p_2)`` per seed.

    When ``mark_stackelberg`` is true, additionally mark the Stackelberg
    duopoly outcome (oblivious seller as leader). ``zoom_to_data`` chooses
    axis limits from the empirical scatter plus benchmarks so that the
    points (rather than the full price box) fill the plot.
    """
    if use_running_mean:
        # Use the running mean at the final log step.
        m = result.moments["m"]  # (T_log, N, S)
        end = m[-1]
    else:
        end = result.prices[-1]
    fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
    i, j = sellers
    ax.scatter(end[i], end[j], s=18, alpha=0.7, color="tab:blue", edgecolor="white", linewidth=0.4)
    d = result.config.market
    p_NE = market.nash_prices(d)
    p_C = market.collusive_prices(d)
    extras: list[tuple[float, float]] = [(float(p_NE[i]), float(p_NE[j])), (float(p_C[i]), float(p_C[j]))]
    ax.scatter([p_NE[i]], [p_NE[j]], s=120, color="tab:red", marker="X",
               edgecolor="white", linewidth=0.6, zorder=10, label=r"$p^{NE}$")
    ax.scatter([p_C[i]], [p_C[j]], s=120, color="tab:green", marker="X",
               edgecolor="white", linewidth=0.6, zorder=10, label=r"$p^{C}$")
    if mark_stackelberg and d.N == 2:
        p_S = np.array(market.stackelberg_duopoly(d))
        ax.scatter([p_S[i]], [p_S[j]], s=120, color="tab:purple", marker="X",
                   edgecolor="white", linewidth=0.6, zorder=10,
                   label=r"$p^{*}$ (Stackelberg)")
        extras.append((float(p_S[i]), float(p_S[j])))
    ax.set_xlabel(f"$\\bar p_{{{i+1}}}$")
    ax.set_ylabel(f"$\\bar p_{{{j+1}}}$")
    if zoom_to_data:
        xs = np.concatenate([end[i], [pt[0] for pt in extras]])
        ys = np.concatenate([end[j], [pt[1] for pt in extras]])
        x_lo, x_hi = float(xs.min()), float(xs.max())
        y_lo, y_hi = float(ys.min()), float(ys.max())
        x_pad = max(0.08 * (x_hi - x_lo), 0.02)
        y_pad = max(0.08 * (y_hi - y_lo), 0.02)
        ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
        ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
    if title:
        ax.set_title(title)
    smart_legend(ax)
    square_box(ax)
    fig.tight_layout()
    return fig


def plot_revenue_bars(
    summary: dict[str, dict[str, np.ndarray]],
    *,
    benchmarks: dict[str, np.ndarray] | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Bar chart of mean ``R_T / T`` per (composition, seller).

    ``summary[name]["mean"]`` should be ``(N,)`` per composition; ``benchmarks``
    is a dict like ``{"NE": (N,), "collusive": (N,)}`` for reference lines.
    """
    names: list[str] = list(summary.keys())
    N = len(next(iter(summary.values()))["mean"])
    width = 0.8 / N
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
    cmap = plt.get_cmap("tab10")
    for i in range(N):
        means = np.array([summary[name]["mean"][i] for name in names])
        p05 = np.array([summary[name].get("p05", means)[i] for name in names])
        p95 = np.array([summary[name].get("p95", means)[i] for name in names])
        # ``yerr`` requires non-negative entries; clamp on either side in case
        # some compositions only reported per-seller mean (no percentiles).
        low_err = np.clip(means - p05, 0.0, None)
        high_err = np.clip(p95 - means, 0.0, None)
        ax.bar(
            x + i * width - 0.4 + width / 2,
            means,
            width=width,
            color=cmap(i % 10),
            label=f"seller {i}",
            yerr=np.array([low_err, high_err]),
            capsize=2,
        )
    ax.set_xticks(x, names)
    if benchmarks is not None:
        for label, vec in benchmarks.items():
            ax.axhline(float(np.mean(vec)), linestyle="--", lw=1.0, label=fr"$\Pi^{{{label}}}$")
    ax.set_ylabel(r"$R_T / T$")
    if title:
        ax.set_title(title)
    smart_legend(ax, fontsize=10)
    square_box(ax)
    fig.tight_layout()
    return fig


def plot_excursion_overlay(
    n_log: np.ndarray,
    discrete_mean_price: np.ndarray,
    ode_solution: dict[str, Any] | None,
    *,
    p_NE: float,
    p_C: float,
    title: str | None = None,
) -> plt.Figure:
    """Plot mean price (averaged across seeds) with an ODE trajectory overlay."""
    fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
    ax.plot(n_log, discrete_mean_price, color="tab:blue", lw=1.0, label="discrete (mean)")
    if ode_solution is not None:
        ax.plot(ode_solution["t"], ode_solution["m_avg"], color="tab:orange", lw=1.5, label="ODE")
    ax.axhline(p_NE, color="tab:red", linestyle="--", lw=1.0, label=r"$p^{NE}$")
    ax.axhline(p_C, color="tab:green", linestyle="--", lw=1.0, label=r"$p^{C}$")
    ax.set_xlabel("n")
    ax.set_ylabel(r"$\bar p_n$")
    if title:
        ax.set_title(title)
    smart_legend(ax)
    square_box(ax)
    fig.tight_layout()
    return fig


def grid_figure(panels: Iterable[plt.Figure], *, cols: int = 2) -> plt.Figure:
    """Compose multiple existing figures into a grid (best-effort)."""
    panels = list(panels)
    n = len(panels)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * SQUARE_FIGSIZE[0], rows * SQUARE_FIGSIZE[1]))
    axes = np.atleast_2d(axes)
    for k, p in enumerate(panels):
        r, c = k // cols, k % cols
        # Re-render onto the new axes by copying the children of the first axis.
        src = p.axes[0]
        dst = axes[r, c]
        for line in src.lines:
            dst.plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color(), lw=line.get_linewidth())
        dst.set_xscale(src.get_xscale())
        dst.set_yscale(src.get_yscale())
        dst.set_xlabel(src.get_xlabel())
        dst.set_ylabel(src.get_ylabel())
        if src.get_title():
            dst.set_title(src.get_title())
        if src.get_legend() is not None:
            dst.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig
