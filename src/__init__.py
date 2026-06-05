"""Library for dynamic pricing learning simulations.

Experiment scripts under ``experiments/`` build :class:`src.config.ExperimentConfig`
instances and call :func:`src.simulator.run_simulation`. See the top-level
``README.md`` for the layout and entry points.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ob-learn")
except PackageNotFoundError:  # editable installs without metadata
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
