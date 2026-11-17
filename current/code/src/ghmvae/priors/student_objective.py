"""Weighted Student-t likelihood and exact derivatives in Cholesky coordinates.

This standalone numerical module is also copied into audited vendor snapshots.
The parameter order is [mu, log(diag(L)), strict_lower(L), log(nu)], Sigma=L L'.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import digamma, gammaln


def student_negative_loglikelihood(u, x, weights):
    """Return minus sum_i w_i log t(x_i) and its analytic gradient.

    With v_i=Sigma^{-1}(x_i-mu), delta_i=(x_i-mu)'v_i and
    tau_i=(nu+d)/(nu+delta_i), the log-likelihood derivatives are
    g_mu=sum_i w_i tau_i v_i and
    g_L=[sum_i w_i tau_i v_i v_i' - sum_i w_i Sigma^{-1}]L.
    Log-diagonal and log-nu coordinates use the chain rule.
    """
    u, x, weights = np.asarray(u, float), np.asarray(x, float), np.asarray(weights, float)
    if x.ndim != 2:
        raise ValueError("Student-t observations must be a matrix")
    n, d = x.shape
    lower = np.tril_indices(d, k=-1)
    expected = 2 * d + len(lower[0]) + 1
    if u.shape != (expected,) or weights.shape != (n,) or np.any(weights < 0):
        raise ValueError("Invalid Student-t parameter vector or weights")
    if not (np.isfinite(x).all() and np.isfinite(u).all() and np.isfinite(weights).all()):
        raise ValueError("Nonfinite Student-t inputs")
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        chol = np.zeros((d, d))
        chol[np.diag_indices(d)] = np.exp(u[d:2*d])
        chol[lower] = u[2*d:-1]
        nu = np.exp(u[-1])
        if nu <= 0 or np.any(np.diag(chol) <= 0):
            raise ValueError("Nonpositive Student-t scale or degrees of freedom")
        residual = x - u[:d]
        white = solve_triangular(chol, residual.T, lower=True, check_finite=False)
        delta = np.sum(white * white, axis=0)
        log_tail = np.log1p(delta / nu)
        constant = (gammaln((nu+d)/2) - gammaln(nu/2)
                    - d/2 * (np.log(nu) + np.log(np.pi)) - np.sum(u[d:2*d]))
        log_density = constant - (nu+d)/2 * log_tail
        value = -np.sum(weights * log_density)

        precision_residual = solve_triangular(chol.T, white, lower=False, check_finite=False).T
        tau = (nu+d) / (nu+delta)
        weighted_tau = weights * tau
        g_mu = np.sum(weighted_tau[:, None] * precision_residual, axis=0)
        inverse_chol = solve_triangular(chol, np.eye(d), lower=True, check_finite=False)
        precision = inverse_chol.T @ inverse_chol
        g_chol = (precision_residual.T @ (weighted_tau[:, None] * precision_residual)
                  - weights.sum() * precision) @ chol
        g_log_diag = np.diag(g_chol) * np.diag(chol)
        g_nu = 0.5 * (digamma((nu+d)/2) - digamma(nu/2) - d/nu - log_tail
                      + (nu+d) / nu * (delta / (nu+delta)))
        gradient = -np.concatenate([g_mu, g_log_diag, g_chol[lower],
                                    [nu * np.sum(weights * g_nu)]])
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise ValueError("Nonfinite Student-t objective or gradient")
    return float(value), gradient
