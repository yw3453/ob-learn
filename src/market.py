"""Market-level closed-form quantities derived from :class:`DemandParams`.

* :func:`nash_prices`             solves ``Gamma p^{NE} = -alpha``.
* :func:`collusive_prices`        solves ``H p^{C} = -alpha``.
* :func:`stackelberg_duopoly`     Stackelberg prices for the duopoly.
* :func:`pseudo_true_oblivious`   pseudo-true oblivious estimates.
* :func:`gamma_bar`               threshold constant ``gamma_bar``.
* :func:`L_phi_oblivious`         Lipschitz constant of the oblivious greedy map
  ``phi^{ob}(theta) = -a/(2b)`` over a 2-D box.
* :func:`C_x`                     uniform bound on ``||x_{n,i}^{ob}||_2``.
* :func:`per_period_revenue`      per-period revenue.
"""

from __future__ import annotations

import numpy as np

from .config import DemandParams, InformedProjectionBox, ProjectionBox


def gamma_matrix(d: DemandParams) -> np.ndarray:
    """Best-response operator ``Gamma`` with ``Gamma_ii = 2 beta_i``, ``Gamma_ij = gamma[i][j]``."""
    Gamma = d.gamma_arr.copy()
    for i in range(d.N):
        Gamma[i, i] = 2.0 * d.beta[i]
    return Gamma


def collusive_hessian(d: DemandParams) -> np.ndarray:
    """Hessian of the joint revenue, ``H_ii = 2 beta_i``, ``H_ij = gamma[i][j] + gamma[j][i]``."""
    G = d.gamma_arr
    H = G + G.T
    for i in range(d.N):
        H[i, i] = 2.0 * d.beta[i]
    return H


def nash_prices(d: DemandParams) -> np.ndarray:
    """Solve ``Gamma p^{NE} = -alpha``."""
    Gamma = gamma_matrix(d)
    return np.linalg.solve(Gamma, -d.alpha_arr)


def collusive_prices(d: DemandParams) -> np.ndarray:
    """Solve ``H p^{C} = -alpha``."""
    H = collusive_hessian(d)
    return np.linalg.solve(H, -d.alpha_arr)


def stackelberg_duopoly(d: DemandParams) -> tuple[float, float]:
    """Stackelberg prices for the duopoly (seller 1 leader, seller 2 follower).

    Closed form::

        p_1^* = (-2 beta_2 alpha_1 + gamma_{1,2} alpha_2)
                / (4 beta_1 beta_2 - 2 gamma_{1,2} gamma_{2,1}),
        p_2^* = (alpha_2 + gamma_{2,1} p_1^*) / (-2 beta_2).
    """
    if d.N != 2:
        raise ValueError("stackelberg_duopoly is only defined for N=2")
    a1, a2 = d.alpha
    b1, b2 = d.beta
    g12 = d.gamma[0][1]
    g21 = d.gamma[1][0]
    denom = 4.0 * b1 * b2 - 2.0 * g12 * g21
    p1 = (-2.0 * b2 * a1 + g12 * a2) / denom
    p2 = (a2 + g21 * p1) / (-2.0 * b2)
    return float(p1), float(p2)


def pseudo_true_oblivious(d: DemandParams, p_NE: np.ndarray | None = None) -> np.ndarray:
    """Per-seller pseudo-true oblivious target ``theta_i^{*, ob} = (a_i^*, b_i^*)``.

    ``a_i^* = alpha_i + sum_{j != i} gamma[i][j] p_j^{NE}``, ``b_i^* = beta_i``.
    """
    if p_NE is None:
        p_NE = nash_prices(d)
    G = d.gamma_arr
    a_star = d.alpha_arr + G @ p_NE
    b_star = d.beta_arr
    return np.stack([a_star, b_star], axis=1)


def true_informed_theta(d: DemandParams) -> np.ndarray:
    """``(N, N+1)`` array of true informed parameters per seller.

    Layout: ``theta[i, 0] = alpha_i``, ``theta[i, k] = beta_i`` if ``k == i + 1``,
    ``theta[i, k] = gamma[i][k - 1]`` otherwise (for ``k = 1, ..., N``). This
    matches the regressor ``x_{n,i}^{in} = (1, p_{n,1}, ..., p_{n,N})``.
    """
    N = d.N
    theta = np.zeros((N, N + 1), dtype=np.float64)
    theta[:, 0] = d.alpha_arr
    for i in range(N):
        for k in range(1, N + 1):
            j = k - 1
            theta[i, k] = d.beta[i] if j == i else d.gamma[i][j]
    return theta


def gamma_bar(d: DemandParams) -> float:
    """``gamma_bar = (max_i gamma_i + max_i gamma_i^{col}) / 2``."""
    return 0.5 * (float(d.gamma_row_sums.max()) + float(d.gamma_col_sums.max()))


def L_phi_oblivious(box: ProjectionBox) -> float:
    """Lipschitz constant of ``phi^{ob}(a, b) = -a / (2 b)`` over the projection box.

    The 2-norm of ``grad phi`` equals
    ``(1 / (2 |b|)) sqrt(1 + a^2 / b^2)``,
    which is monotone in ``a`` and in ``1 / |b|``. The supremum over
    ``[a_low, a_high] x [b_low, b_high]`` (with ``b_high < 0``) is therefore at
    ``a = a_high`` and ``b = b_high`` (smallest ``|b|``).
    """
    a_high = np.asarray(box.a_high)
    b_high = np.asarray(box.b_high)
    grad = np.sqrt(1.0 + (a_high / b_high) ** 2) / (2.0 * np.abs(b_high))
    return float(np.max(grad))


def C_x_oblivious(d: DemandParams) -> float:
    """Uniform bound on ``||x_{n,i}^{ob}||_2 = ||(1, p_{n,i})||_2``."""
    return float(np.sqrt(1.0 + d.u**2))


def C_x_informed(d: DemandParams) -> float:
    """Uniform bound on ``||x_{n,i}^{in}||_2 = ||(1, p_{n,1}, ..., p_{n,N})||_2``."""
    return float(np.sqrt(1.0 + d.N * d.u**2))


def per_period_revenue(d: DemandParams, p: np.ndarray) -> np.ndarray:
    """Expected per-period revenue under prices ``p``: ``Pi_i(p) = p_i (alpha_i + beta_i p_i + sum gamma[i,j] p_j)``."""
    p = np.asarray(p, dtype=np.float64)
    expected_d = d.alpha_arr + d.beta_arr * p + d.gamma_arr @ p
    return p * expected_d


def cm_upper_bound(d: DemandParams, nu_squared: float, p_mean: float | None = None) -> float:
    """Estimate the smallest eigenvalue of ``E[(1, p)(1, p)^T | F_{n-1}]``.

    For the oblivious regressor ``x = (1, p)``, the conditional second-moment
    matrix has the structure ``[[1, m], [m, m^2 + V]]`` where ``m = E[p]`` and
    ``V = Var(p)``. Its smallest eigenvalue equals
    ``(T - sqrt(T^2 - 4 V)) / 2`` with ``T = 1 + m^2 + V``. As ``V -> 0`` this
    behaves like ``V / (1 + m^2)``; as ``V -> infty`` it tends to 1.

    With ``Var(z) = nu^2`` and a deterministic component centered near
    ``p^{NE}``, ``V >= nu^2``, so this returns the lambda_min for ``V = nu^2``
    -- the smallest C_M one can plausibly attain at that ``nu^2`` is below
    this value (assuming the box hasn't saturated).
    """
    if p_mean is None:
        p_NE = nash_prices(d)
        p_mean = float(np.mean(p_NE))
    V = max(nu_squared, 1e-12)
    T = 1.0 + p_mean**2 + V
    return float((T - np.sqrt(max(T**2 - 4.0 * V, 0.0))) / 2.0)


def master_theorem_smallgain(
    d: DemandParams,
    ob_idx: list[int] | tuple[int, ...] | np.ndarray,
    in_idx: list[int] | tuple[int, ...] | np.ndarray,
    box_ob: ProjectionBox,
    box_in: InformedProjectionBox,
    nu_squared: float,
    *,
    beta_abs_min: float | None = None,
) -> dict[str, float | bool]:
    """Compute the constants in the oblivious-side small-gain condition.

    Returns a dict with all constants plus the LHS, RHS, and margin of the
    small-gain inequality. The LHS is ``C_M`` (estimated from
    :func:`cm_upper_bound`) and the RHS is ``C_x [2 bar_gamma_ob L_phi_ob +
    bar_Delta + bar_Theta + L_phi_ob bar_Psi / (bar_c_diag - bar_D)]``. The
    small-gain condition holds iff ``margin > 0``.

    Also returns the derived constants ``K_1``, ``K_2``, used to optimise the
    rate function ``c*(lambda)``.

    Parameters
    ----------
    ob_idx, in_idx : sequences of seller indices in ``[0, N)``.
        Must partition ``[N]``. Either may be empty.
    box_ob : :class:`ProjectionBox`
        The full ``N``-seller oblivious projection box; only entries for
        ``ob_idx`` are used to compute ``L_phi^{ob}``.
    box_in : :class:`InformedProjectionBox`
        The full ``N``-seller informed projection box; only entries for
        ``in_idx`` are used to compute ``L_phi^{in, theta}``.
    nu_squared : float
        The (common) oblivious-side exploration variance ``nu^2``, used
        to estimate the persistent-excitation constant ``C_M`` via
        :func:`cm_upper_bound`.
    beta_abs_min : float, optional
        Override the ``|beta|_{\\min}`` value used in the upper bound on
        ``L_phi^{in, theta}``. By default we read it from ``box_in`` (the
        smallest absolute value attained by the beta-coordinate of any
        informed seller). The default reads a tight upper bound only if
        the box is tight around the truth; for a generous projection box
        such as :meth:`InformedProjectionBox.from_demand` it makes
        ``L_phi^{in, theta}`` enormous, so callers that simulate with a
        generous box can pass a tighter ``beta_abs_min`` (e.g., half of
        the minimum true ``|beta_i|``) for a more representative bound.
    """
    from .config import InformedProjectionBox  # local: avoid top-level cycle

    if not isinstance(box_in, InformedProjectionBox):
        raise TypeError(
            f"box_in must be an InformedProjectionBox, got {type(box_in)!r}"
        )
    ob_idx = np.asarray(ob_idx, dtype=np.int64)
    in_idx = np.asarray(in_idx, dtype=np.int64)
    G = d.gamma_arr  # (N, N) with diagonal 0
    beta = d.beta_arr  # (N,)
    N = d.N

    # ----- Persistent-excitation and envelope constants. -------------------
    C_M = cm_upper_bound(d, nu_squared)
    C_x = C_x_oblivious(d)

    # ----- Oblivious-side Lipschitz. ---------------------------------------
    if ob_idx.size > 0:
        # Restrict the oblivious box to ob_idx so L_phi^{ob} is taken over
        # exactly the relevant rows.
        sub_box_ob = ProjectionBox(
            a_low=[box_ob.a_low[i] for i in ob_idx.tolist()],
            a_high=[box_ob.a_high[i] for i in ob_idx.tolist()],
            b_low=[box_ob.b_low[i] for i in ob_idx.tolist()],
            b_high=[box_ob.b_high[i] for i in ob_idx.tolist()],
        )
        L_phi_ob = L_phi_oblivious(sub_box_ob)
    else:
        L_phi_ob = 0.0

    # ----- Informed-side Lipschitz in theta. -------------------------------
    # Upper bound on L_phi^{in,theta} <= (2 |beta|_min)^{-1} sqrt(1 + u^2 /
    # |beta|_min^2 + (N - 1) u^2), with |beta|_min the smallest absolute
    # value attained by the informed projection box's beta-coordinate.
    if in_idx.size > 0:
        if beta_abs_min is None:
            beta_abs_min_per_seller = []
            for j in in_idx.tolist():
                # In InformedProjectionBox.low[j], coordinate (j+1) is beta_j;
                # both endpoints are negative, so |beta|-min is |high[j+1]|.
                b_high_j = float(box_in.high[j][j + 1])
                b_low_j = float(box_in.low[j][j + 1])
                beta_abs_min_per_seller.append(min(abs(b_high_j), abs(b_low_j)))
            beta_abs_min_eff = (
                min(beta_abs_min_per_seller) if beta_abs_min_per_seller else 0.0
            )
        else:
            beta_abs_min_eff = float(beta_abs_min)
        if beta_abs_min_eff <= 0:
            raise ValueError(
                "informed projection box has a beta-coordinate that crosses 0; "
                "master-condition L_phi^{in,theta} is undefined."
            )
        L_phi_in_theta = (
            1.0 / (2.0 * beta_abs_min_eff)
            * float(np.sqrt(1.0 + (d.u**2) / (beta_abs_min_eff**2) + (N - 1) * d.u**2))
        )
    else:
        L_phi_in_theta = 0.0
        beta_abs_min_eff = float("nan")

    # ----- bar_gamma^{ob} ---------------------------------------------------
    if ob_idx.size > 0:
        ob_set = set(ob_idx.tolist())
        # Row sums and column sums restricted to ob_idx.
        gamma_ob_row = np.zeros(ob_idx.size)
        gamma_ob_col = np.zeros(ob_idx.size)
        for k, i in enumerate(ob_idx.tolist()):
            for j in ob_idx.tolist():
                if j == i:
                    continue
                gamma_ob_row[k] += abs(G[i, j])
                gamma_ob_col[k] += abs(G[j, i])
        bar_gamma_ob = 0.5 * (float(gamma_ob_row.max()) + float(gamma_ob_col.max()))
        # gamma_j = sum_{k != j} |gamma_{j, k}|.
        gamma_full_row = np.abs(G).sum(axis=1)  # (N,)
        del ob_set  # set was only used above for clarity
    else:
        bar_gamma_ob = 0.0
        gamma_full_row = np.abs(G).sum(axis=1)

    # ----- bar_Delta = max_i Sum_{j in I^in} |gamma_{i,j}| gamma_j / (2 |beta_j|) -
    if ob_idx.size > 0 and in_idx.size > 0:
        delta_per_ob = np.zeros(ob_idx.size)
        for k, i in enumerate(ob_idx.tolist()):
            for j in in_idx.tolist():
                delta_per_ob[k] += abs(G[i, j]) * gamma_full_row[j] / (2.0 * abs(beta[j]))
        bar_Delta = float(delta_per_ob.max())
    else:
        bar_Delta = 0.0

    # ----- bar_Psi_k = sum_{i in I^ob} sum_{j in I^in \ {k}}
    #                  |gamma_{i,j}| |gamma_{j,k}| / (2 |beta_j|)
    if ob_idx.size > 0 and in_idx.size > 0:
        psi_per_k = np.zeros(N)
        for k in range(N):
            total = 0.0
            for i in ob_idx.tolist():
                for j in in_idx.tolist():
                    if j == k:
                        continue
                    total += abs(G[i, j]) * abs(G[j, k]) / (2.0 * abs(beta[j]))
            psi_per_k[k] = total
        bar_Psi = float(psi_per_k.max())
    else:
        bar_Psi = 0.0

    # ----- bar_D_k = sum_{j in I^in \ {k}} |gamma_{j, k}| / (2 |beta_j|) ----
    if in_idx.size > 0:
        D_per_k = np.zeros(N)
        for k in range(N):
            total = 0.0
            for j in in_idx.tolist():
                if j == k:
                    continue
                total += abs(G[j, k]) / (2.0 * abs(beta[j]))
            D_per_k[k] = total
        bar_D = float(D_per_k.max())
    else:
        bar_D = 0.0

    # ----- bar_c_diag = min(2 - L_phi^{ob}, 2 - max_j gamma_j/(2|beta_j|) - L_phi^{in,theta}) -
    if in_idx.size > 0:
        worst_informed_self = max(
            gamma_full_row[j] / (2.0 * abs(beta[j])) for j in in_idx.tolist()
        )
        bar_c_diag = min(2.0 - L_phi_ob, 2.0 - worst_informed_self - L_phi_in_theta)
    else:
        # All-oblivious: only the oblivious self-decay matters.
        bar_c_diag = 2.0 - L_phi_ob

    # ----- bar_Theta = L_phi^{in,theta} max_{i in I^ob} sum_{j in I^in} |gamma_{i,j}| -
    if ob_idx.size > 0 and in_idx.size > 0:
        row_sums = np.zeros(ob_idx.size)
        for k, i in enumerate(ob_idx.tolist()):
            for j in in_idx.tolist():
                row_sums[k] += abs(G[i, j])
        bar_Theta = L_phi_in_theta * float(row_sums.max())
    else:
        bar_Theta = 0.0

    # ----- Assemble the small-gain margin ----------------------------------
    denom = bar_c_diag - bar_D
    if in_idx.size == 0:
        # All-oblivious: the running-mean track disappears and the small-gain
        # condition collapses to C_M > 2 bar_gamma^{ob} L_phi^{ob} C_x.
        rhs = C_x * (2.0 * bar_gamma_ob * L_phi_ob)
        denom = float("nan")
    else:
        if denom <= 0:
            rhs = float("inf")
        else:
            rhs = C_x * (
                2.0 * bar_gamma_ob * L_phi_ob
                + bar_Delta
                + bar_Theta
                + L_phi_ob * bar_Psi / denom
            )
    lhs = C_M
    margin = lhs - rhs
    condition_holds = bool(np.isfinite(margin) and margin > 0)

    # ----- K_1, K_2 (eq:K1K2-ext) -------------------------------------------
    K1 = C_M - C_x * (2.0 * bar_gamma_ob * L_phi_ob + bar_Delta + bar_Theta)
    K2 = denom if in_idx.size > 0 else float("nan")

    return dict(
        N=N,
        n_ob=int(ob_idx.size),
        n_in=int(in_idx.size),
        nu_squared=float(nu_squared),
        C_M=float(C_M),
        C_x=float(C_x),
        L_phi_ob=float(L_phi_ob),
        L_phi_in_theta=float(L_phi_in_theta),
        beta_abs_min=float(beta_abs_min_eff),
        bar_gamma_ob=float(bar_gamma_ob),
        bar_Delta=float(bar_Delta),
        bar_Psi=float(bar_Psi),
        bar_D=float(bar_D),
        bar_c_diag=float(bar_c_diag),
        bar_Theta=float(bar_Theta),
        K_1=float(K1),
        K_2=float(K2),
        rhs=float(rhs),
        lhs=float(lhs),
        margin=float(margin),
        condition_holds=condition_holds,
    )


def predicted_regime(
    d: DemandParams,
    box: ProjectionBox,
    nu_squared: float,
) -> dict[str, float | str]:
    """Estimate which convergence regime applies given an exploration variance ``nu^2``.

    Uses the upper bound on the persistent-excitation constant ``C_M`` from
    :func:`cm_upper_bound` -- the smallest eigenvalue of
    ``E[x_{n,i}^{ob} (x_{n,i}^{ob})^T]`` cannot exceed this value when the
    deterministic part of the price is concentrated near ``p^{NE}``. Note that
    when prices saturate the projection box ``[l, u]``, the *actual* ``C_M``
    can be even smaller than this upper bound (because realized ``Var(p)`` is
    capped by the box), so this regime label is best interpreted as an
    *optimistic* prediction.
    """
    gb = gamma_bar(d)
    lp = L_phi_oblivious(box)
    cx = C_x_oblivious(d)
    threshold = gb * lp * cx
    cm = cm_upper_bound(d, nu_squared)
    if cm > 2.0 * threshold:
        regime = "fast"
    elif np.isclose(cm, 2.0 * threshold):
        regime = "critical"
    elif cm > threshold:
        regime = "slow"
    else:
        regime = "below"
    rho = threshold / cm if cm > 0 else float("nan")
    return {
        "gamma_bar": gb,
        "L_phi_ob": lp,
        "C_x": cx,
        "threshold": threshold,  # gb * lp * cx, threshold for "slow"
        "threshold_fast": 2.0 * threshold,
        "C_M_estimate": cm,
        "rho": rho,
        "regime": regime,
        "nu_squared": float(nu_squared),
    }
