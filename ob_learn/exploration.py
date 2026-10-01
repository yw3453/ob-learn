"""Exploration-noise schedules ``Var(z_{n,i}) = nu_n^2``.

The noise ``z_{n,i}`` is mean-zero, independent across ``(n, i)``, and
distributed as :math:`\\mathrm{Uniform}[-\\sqrt{3} \\nu, +\\sqrt{3} \\nu]`, which
is bounded and has variance ``nu^2`` exactly. Demand noise is uniform in the
same way.

Schedules:

* :class:`ConstantSchedule` -- ``nu_n^2 = nu^2``.
* :class:`PolynomialSchedule` -- ``nu_n^2 = c * (n+1)^{-eta}``.
* :class:`SqrtNSchedule` -- ``nu_n^2 = c / (2 * sqrt(n+1))`` so that
  ``sum_{m <= n} nu_m^2 = Theta(sqrt(n))``.
"""

from __future__ import annotations

import numpy as np

from .config import ExplorationSchedule

_SQRT3 = float(np.sqrt(3.0))


def nu_at(schedule: ExplorationSchedule, n: int) -> float:
    """Return ``nu_n`` (standard deviation) for one step ``n`` (1-indexed)."""
    if schedule.kind == "constant":
        return float(schedule.nu or 0.0)
    if schedule.kind == "polynomial":
        c = schedule.c or 0.0
        eta = schedule.eta or 0.0
        return float(np.sqrt(c)) * (max(n, 1)) ** (-eta / 2.0)
    if schedule.kind == "sqrt_n":
        c = schedule.c or 0.0
        return float(np.sqrt(c / (2.0 * np.sqrt(max(n, 1)))))
    raise ValueError(f"unknown exploration kind: {schedule.kind}")


def sample_demand_noise(
    noise_std: float,
    rng: np.random.Generator,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Draw demand-noise innovations ``epsilon_{n,i}`` (mean zero, bounded)."""
    if noise_std == 0.0:
        return np.zeros(shape, dtype=np.float64)
    a = _SQRT3 * noise_std
    return rng.uniform(low=-a, high=a, size=shape)
