"""Exploration-noise schedules ``Var(z_{n,i}) = nu_n^2``.

The noise ``z_{n,i}`` is mean-zero, independent across ``(n, i)``,
and has **bounded support**. The default distribution is
:math:`\\mathrm{Uniform}[-\\sqrt{3} \\nu, +\\sqrt{3} \\nu]`, which is bounded
and has variance ``nu^2`` exactly. A Gaussian-with-clip option is provided as
an alternative; its variance is approximated to be ``nu^2`` (with a small bias
when the clip is tight).

Schedules:

* :class:`ConstantSchedule` -- ``nu_n^2 = nu^2``.
* :class:`PolynomialSchedule` -- ``nu_n^2 = c * (n+1)^{-eta}``.
* :class:`SqrtNSchedule` -- ``nu_n^2 = c / (2 * sqrt(n+1))`` so that
  ``sum_{m <= n} nu_m^2 = Theta(sqrt(n))``.

Each schedule exposes ``nu(n) -> float`` and
``sample(n, rng, shape) -> ndarray``.
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


def sample(
    schedule: ExplorationSchedule,
    n: int,
    rng: np.random.Generator,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Draw bounded mean-zero perturbations with std ``nu_n`` and given shape."""
    nu = nu_at(schedule, n)
    if nu == 0.0:
        return np.zeros(shape, dtype=np.float64)
    if schedule.distribution == "uniform":
        a = _SQRT3 * nu
        return rng.uniform(low=-a, high=a, size=shape)
    if schedule.distribution == "gaussian_clip":
        clip = schedule.clip_sigmas * nu
        z = rng.normal(loc=0.0, scale=nu, size=shape)
        return np.clip(z, -clip, clip)
    raise ValueError(f"unknown exploration distribution: {schedule.distribution}")


def sample_demand_noise(
    noise_kind: str,
    noise_std: float,
    noise_clip_sigmas: float,
    rng: np.random.Generator,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Draw demand-noise innovations ``epsilon_{n,i}`` (mean zero, bounded)."""
    if noise_std == 0.0:
        return np.zeros(shape, dtype=np.float64)
    if noise_kind == "uniform":
        a = _SQRT3 * noise_std
        return rng.uniform(low=-a, high=a, size=shape)
    if noise_kind == "gaussian_clip":
        clip = noise_clip_sigmas * noise_std
        z = rng.normal(loc=0.0, scale=noise_std, size=shape)
        return np.clip(z, -clip, clip)
    raise ValueError(f"unknown noise kind: {noise_kind}")
