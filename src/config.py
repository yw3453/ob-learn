"""Pydantic-validated configuration objects.

All experiment scripts construct one or more :class:`ExperimentConfig` instances
and pass them to :func:`src.simulator.run_simulation`. The configs
serialize to YAML/JSON so they can be persisted to a run directory and reloaded.

The objects map onto the model components as follows:

* :class:`DemandParams` is the linear demand model.
* :class:`ProjectionBox` and :class:`InformedProjectionBox` are the projection
  sets ``Theta_i^{ob}`` and ``Theta_i^{in}``.
* :class:`ExplorationSchedule` parameterizes ``Var(z_{n,i})`` for each seller.
* :class:`SellerSpec` ties together each seller's modeling choice.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


def _as_float_array(seq: list[float] | np.ndarray) -> np.ndarray:
    return np.asarray(seq, dtype=np.float64)


class DemandParams(BaseModel):
    """Linear demand model parameters.

    ``d_{n,i} = alpha[i] + beta[i] * p_{n,i} + sum_{j!=i} gamma[i][j] * p_{n,j} + eps_{n,i}``

    ``gamma`` is stored as an ``N x N`` matrix; the diagonal is ignored.
    ``gamma[i][j] != gamma[j][i]`` is allowed.
    """

    model_config = ConfigDict(frozen=True)

    N: int = Field(ge=2)
    alpha: list[float]
    beta: list[float]
    gamma: list[list[float]]
    l: float = Field(gt=0.0)
    u: float
    noise_kind: Literal["uniform", "gaussian_clip"] = "uniform"
    noise_std: float = Field(ge=0.0)
    noise_clip_sigmas: float = Field(default=4.0, gt=0.0)
    """For ``gaussian_clip`` noise: clip at ``noise_clip_sigmas * noise_std``."""

    @model_validator(mode="after")
    def _validate(self) -> DemandParams:
        N = self.N
        if len(self.alpha) != N or len(self.beta) != N:
            raise ValueError("alpha and beta must have length N")
        if len(self.gamma) != N or any(len(row) != N for row in self.gamma):
            raise ValueError("gamma must be N x N")
        if self.u <= self.l:
            raise ValueError(f"Need u > l, got l={self.l}, u={self.u}")
        for i in range(N):
            if self.alpha[i] <= 0:
                raise ValueError(f"alpha[{i}]={self.alpha[i]} must be > 0")
            if self.beta[i] >= 0:
                raise ValueError(f"beta[{i}]={self.beta[i]} must be < 0")
            for j in range(N):
                if i == j:
                    continue
                if self.gamma[i][j] <= 0:
                    raise ValueError(
                        f"gamma[{i}][{j}]={self.gamma[i][j]} must be > 0 for cross terms"
                    )
            gamma_i = sum(self.gamma[i][j] for j in range(N) if j != i)
            if -self.beta[i] <= gamma_i:
                raise ValueError(
                    f"Own-price dominance fails for seller {i}: "
                    f"-beta_i={-self.beta[i]} <= gamma_i={gamma_i}"
                )
            gamma_i_col = sum(self.gamma[j][i] for j in range(N) if j != i)
            if -2.0 * self.beta[i] <= gamma_i + gamma_i_col:
                raise ValueError(
                    f"Collusive dominance fails for seller {i}: "
                    f"-2*beta_i={-2 * self.beta[i]} <= gamma_i + gamma_i^col"
                    f" = {gamma_i + gamma_i_col}"
                )
        return self

    @property
    def alpha_arr(self) -> np.ndarray:
        return _as_float_array(self.alpha)

    @property
    def beta_arr(self) -> np.ndarray:
        return _as_float_array(self.beta)

    @property
    def gamma_arr(self) -> np.ndarray:
        g = np.asarray(self.gamma, dtype=np.float64)
        np.fill_diagonal(g, 0.0)
        return g

    @property
    def gamma_row_sums(self) -> np.ndarray:
        """``gamma_i = sum_{j != i} gamma[i][j]``."""
        return self.gamma_arr.sum(axis=1)

    @property
    def gamma_col_sums(self) -> np.ndarray:
        """``gamma_i^{col} = sum_{j != i} gamma[j][i]``."""
        return self.gamma_arr.sum(axis=0)

    @classmethod
    def symmetric(
        cls,
        N: int,
        alpha: float,
        beta: float,
        gamma: float,
        l: float,
        u: float,
        noise_std: float = 0.2,
        noise_kind: Literal["uniform", "gaussian_clip"] = "uniform",
    ) -> DemandParams:
        gamma_mat = [[0.0 if i == j else float(gamma) for j in range(N)] for i in range(N)]
        return cls(
            N=N,
            alpha=[float(alpha)] * N,
            beta=[float(beta)] * N,
            gamma=gamma_mat,
            l=float(l),
            u=float(u),
            noise_kind=noise_kind,
            noise_std=float(noise_std),
        )


class ProjectionBox(BaseModel):
    """Per-seller box ``Theta_i^{ob} = [a_low, a_high] x [b_low, b_high]``.

    Stored as length-N lists (one box per seller). The box must satisfy
    ``a_low > 0`` and ``b_high < 0``.
    """

    model_config = ConfigDict(frozen=True)

    a_low: list[float]
    a_high: list[float]
    b_low: list[float]
    b_high: list[float]

    @model_validator(mode="after")
    def _validate(self) -> ProjectionBox:
        N = len(self.a_low)
        if not (len(self.a_high) == len(self.b_low) == len(self.b_high) == N):
            raise ValueError("a_low, a_high, b_low, b_high must all have the same length")
        for i in range(N):
            if self.a_low[i] <= 0:
                raise ValueError(f"a_low[{i}]={self.a_low[i]} must be > 0")
            if self.a_high[i] <= self.a_low[i]:
                raise ValueError(f"need a_high > a_low at seller {i}")
            if self.b_high[i] >= 0:
                raise ValueError(f"b_high[{i}]={self.b_high[i]} must be < 0")
            if self.b_low[i] >= self.b_high[i]:
                raise ValueError(f"need b_low < b_high at seller {i}")
        return self

    @classmethod
    def from_demand(
        cls,
        d: DemandParams,
        margin: float = 2.0,
    ) -> ProjectionBox:
        """Construct a generous box that contains each seller's pseudo-true target.

        We size the box conservatively from the demand parameters using ``margin``
        as a multiplicative factor on the natural scale; the user can override.
        """
        N = d.N
        max_alpha = max(d.alpha)
        max_gamma_row = max(sum(d.gamma[i][j] for j in range(N) if j != i) for i in range(N))
        a_high_val = margin * (max_alpha + max_gamma_row * d.u)
        a_low_val = max(1e-6, min(d.alpha) / margin)
        b_low_val = margin * min(d.beta)  # most negative
        b_high_val = max(b for b in d.beta) / margin
        return cls(
            a_low=[a_low_val] * N,
            a_high=[a_high_val] * N,
            b_low=[b_low_val] * N,
            b_high=[b_high_val] * N,
        )


class InformedProjectionBox(BaseModel):
    """Per-seller box ``Theta_i^{in}`` for the informed regression.

    Each seller has an ``(N+1)``-dimensional parameter vector
    ``(alpha_i, theta_{i,1}, ..., theta_{i,N})`` where ``theta_{i,i} = beta_i``
    (own-price coefficient, negative) and ``theta_{i,k} = gamma_{i,k}`` for
    ``k != i`` (positive). The box is stored as ``low[i]``, ``high[i]``,
    each a length-``N+1`` list.
    """

    model_config = ConfigDict(frozen=True)

    low: list[list[float]]
    high: list[list[float]]

    @model_validator(mode="after")
    def _validate(self) -> InformedProjectionBox:
        if len(self.low) != len(self.high):
            raise ValueError("low and high must have the same number of sellers")
        N = len(self.low)
        for i in range(N):
            if len(self.low[i]) != len(self.high[i]):
                raise ValueError(f"seller {i}: low and high must have the same length")
            if any(self.high[i][k] <= self.low[i][k] for k in range(len(self.low[i]))):
                raise ValueError(f"seller {i}: need high > low component-wise")
        return self

    @classmethod
    def from_demand(cls, d: DemandParams, margin: float = 5.0) -> InformedProjectionBox:
        N = d.N
        max_alpha = max(d.alpha)
        max_gamma = max(d.gamma[i][j] for i in range(N) for j in range(N) if i != j)
        max_abs_beta = max(-b for b in d.beta)
        low = [[0.0] * (N + 1) for _ in range(N)]
        high = [[0.0] * (N + 1) for _ in range(N)]
        for i in range(N):
            low[i][0] = 1e-6
            high[i][0] = margin * max_alpha
            for k in range(1, N + 1):
                if (k - 1) == i:
                    low[i][k] = -margin * max_abs_beta
                    high[i][k] = -1e-6
                else:
                    low[i][k] = 1e-6
                    high[i][k] = margin * max_gamma
        return cls(low=low, high=high)


class ExplorationSchedule(BaseModel):
    """Defines ``Var(z_{n,i}) = nu_n^2`` for one seller.

    Modes:

    * ``constant``: ``nu_n = nu`` for all ``n``.
    * ``polynomial``: ``nu_n^2 = c * (n + 1)^{-eta}`` (one-indexed). With
      ``eta = 0`` this reduces to ``constant`` with ``nu = sqrt(c)``.
    * ``sqrt_n``: cumulative variance ``sum_{m<=n} nu_m^2 = Theta(sqrt(n))``,
      i.e. ``nu_n^2 = c / (2 * sqrt(n))``.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["constant", "polynomial", "sqrt_n"]
    nu: float | None = None
    c: float | None = None
    eta: float | None = None
    distribution: Literal["uniform", "gaussian_clip"] = "uniform"
    clip_sigmas: float = Field(default=4.0, gt=0.0)
    """For ``gaussian_clip``: clip at ``clip_sigmas * nu_n``."""

    @model_validator(mode="after")
    def _validate(self) -> ExplorationSchedule:
        if self.kind == "constant":
            if self.nu is None or self.nu < 0:
                raise ValueError("constant exploration requires nu >= 0")
        elif self.kind == "polynomial":
            if self.c is None or self.c <= 0:
                raise ValueError("polynomial exploration requires c > 0")
            if self.eta is None or self.eta < 0:
                raise ValueError("polynomial exploration requires eta >= 0")
        elif self.kind == "sqrt_n":
            if self.c is None or self.c <= 0:
                raise ValueError("sqrt_n exploration requires c > 0")
        return self


_FORECAST_RULES = ("mean_price", "perfect_prediction", "greedy_component", "lag1_autocorr", "oracle_nash")


class SellerSpec(BaseModel):
    """One seller's modeling and exploration choice."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["oblivious", "informed"]
    forecast_rule: (
        Literal["mean_price", "perfect_prediction", "greedy_component", "lag1_autocorr", "oracle_nash"]
        | None
    ) = None
    exploration: ExplorationSchedule

    @model_validator(mode="after")
    def _validate(self) -> SellerSpec:
        if self.kind == "informed" and self.forecast_rule is None:
            raise ValueError("informed sellers must specify a forecast_rule")
        if self.kind == "oblivious" and self.forecast_rule is not None:
            raise ValueError("oblivious sellers must not specify a forecast_rule")
        return self


class ExperimentConfig(BaseModel):
    """The full specification of one experiment run.

    A single run is a vectorized simulation across ``n_seeds`` independent
    seeds for ``horizon`` steps.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    market: DemandParams
    sellers: list[SellerSpec]
    oblivious_projection: ProjectionBox
    informed_projection: InformedProjectionBox
    horizon: int = Field(ge=10)
    n_seeds: int = Field(ge=1, default=1)
    base_seed: int = 0
    log_every: int = Field(ge=1, default=1)
    """Store full-detail trajectories every ``log_every`` steps. Aggregate
    summaries are still computed at every step."""
    initial_prices: list[list[float]] | None = None
    """Optional ``(N_warmup, N)`` list of initial prices to use as a warm-up.
    If ``None``, warm-up uses uniform draws across ``[l, u]``."""
    n_warmup: int | None = None
    """Number of warm-up steps with pure exploration to seed the OLS designs.
    Defaults to ``max(N + 2, 4)`` if not given."""

    @model_validator(mode="after")
    def _validate(self) -> ExperimentConfig:
        N = self.market.N
        if len(self.sellers) != N:
            raise ValueError(f"need {N} seller specs, got {len(self.sellers)}")
        if len(self.oblivious_projection.a_low) != N:
            raise ValueError("oblivious_projection must have one box per seller")
        if len(self.informed_projection.low) != N:
            raise ValueError("informed_projection must have one box per seller")
        for i, spec in enumerate(self.sellers):
            if spec.kind == "informed":
                expected = N + 1
                if len(self.informed_projection.low[i]) != expected:
                    raise ValueError(
                        f"informed seller {i}: projection box dim must equal N+1={expected}"
                    )
        return self
