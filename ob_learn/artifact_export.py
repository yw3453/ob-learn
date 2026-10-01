"""Helpers for exporting reusable figures and tables.

Experiment scripts write run-specific artifacts under timestamped directories in
``results/``. This module additionally supports exporting named figures to
``results/figures/`` and named tables to ``results/tables/`` for convenient
cross-run summaries.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FIGURE_DIR = CODE_DIR / "results" / "figures"
DEFAULT_TABLE_DIR = CODE_DIR / "results" / "tables"


def artifact_figures_dir(*, base: Path | None = None) -> Path:
    base = base or DEFAULT_FIGURE_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base


def artifact_tables_dir(*, base: Path | None = None) -> Path:
    base = base or DEFAULT_TABLE_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base


def export_figure(
    fig: plt.Figure,
    name: str,
    *,
    base: Path | None = None,
    formats: tuple[str, ...] = ("pdf",),
    close: bool = True,
    strip_title: bool = True,
    tight_bbox: bool = True,
    dpi: int | None = None,
) -> list[Path]:
    """Save ``fig`` under ``results/figures/<name>.<ext>``."""
    base = artifact_figures_dir(base=base)
    if strip_title:
        for ax in fig.axes:
            ax.set_title("")
    paths: list[Path] = []
    for ext in formats:
        out = base / f"{name}.{ext}"
        if tight_bbox and dpi is not None:
            fig.savefig(out, bbox_inches="tight", dpi=dpi)
        elif tight_bbox:
            fig.savefig(out, bbox_inches="tight")
        elif dpi is not None:
            fig.savefig(out, dpi=dpi)
        else:
            fig.savefig(out)
        paths.append(out)
    if close:
        with contextlib.suppress(Exception):
            plt.close(fig)
    return paths


def export_table(
    df: pd.DataFrame,
    name: str,
    *,
    base: Path | None = None,
    caption: str | None = None,
    floatfmt: str = ".4g",
) -> dict[str, Path]:
    """Save ``df`` under ``results/tables/<name>.{csv,md}``."""
    base = artifact_tables_dir(base=base)
    csv_path = base / f"{name}.csv"
    md_path = base / f"{name}.md"
    df.to_csv(csv_path, index=False)
    md = io.StringIO()
    if caption:
        md.write(f"**{caption}**\n\n")
    md.write(df.to_markdown(index=False, floatfmt=floatfmt))
    md.write("\n")
    md_path.write_text(md.getvalue(), encoding="utf-8")
    return {"csv": csv_path, "md": md_path}


