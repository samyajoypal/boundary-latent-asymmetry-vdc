"""Student-t density and fitting derivatives without large-df cancellation.

Standalone NumPy/SciPy module copied into pinned FMVMM snapshots. The Gamma
and digamma recurrences are DLMF 5.5.1-2; the half-step expansion follows 5.11.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import eigh, solve_triangular
from scipy.special import digamma, poch


def student_log_normalizer(df, dimension):
    df = float(df)
    if math.isnan(df) or df <= 0 or dimension < 1:
        raise ValueError("Student-t requires positive df and dimension")
    gaussian = -0.5 * dimension * math.log(2.0 * math.pi)
    if math.isinf(df):
        return gaussian
    m, odd = divmod(dimension, 2)
    correction = math.fsum(math.log1p((2 * j + odd) / df) for j in range(m))
    if odd:
        x = df / 2.0
        if x < 64:
            correction += math.log(float(poch(x, 0.5))) - 0.5 * math.log(x)
        else:
            y = 1.0 / x
            correction += y * (-1/8 + y*y * (1/192 + y*y * (-1/640
                           + y*y * (17/14336 + y*y * (-31/18432)))))
    return gaussian + correction


def _normalizer_lognu_score(df, dimension):
    m, odd = divmod(dimension, 2)
    result = -math.fsum((2 * j + odd) / (df + 2 * j + odd) for j in range(m))
    if odd:
        x = df / 2.0
        if x < 64:
            result += x * (digamma(x + 0.5) - digamma(x)) - 0.5
        else:
            # Differentiate the half-step log-Gamma ratio expansion. Keeping
            # inverse powers avoids subtracting two nearly equal digammas.
            y = 1.0 / x
            result += y * (1/8 + y*y * (-1/64 + y*y * (1/128
                       + y*y * (-17/2048 + y*y * (31/2048)))))
    return result


def student_lognu_score(delta, df, dimension):
    """Derivative of log t with respect to log(df), with scatter fixed."""
    delta = np.asarray(delta, dtype=float)
    ratio = delta / df
    small = ratio < 1e-3
    remainder = np.empty_like(ratio)
    r = ratio[small]
    remainder[small] = delta[small] * r * (1/2 + r * (-2/3 + r * (3/4
                            + r * (-4/5 + r * (5/6 + r * (-6/7 + r * 7/8))))))
    r = ratio[~small]
    remainder[~small] = df * (np.log1p(r) - r / (1 + r))
    return (_normalizer_lognu_score(df, dimension) - 0.5 * remainder
            + 0.5 * dimension * (ratio / (1 + ratio)))


def student_logpdf(x, loc, shape, df, allow_singular=False):
    """Location/scatter log density with SciPy-compatible singular semantics."""
    loc = np.atleast_1d(np.asarray(loc, float))
    dimension = loc.size
    x, shape = np.asarray(x, float), np.asarray(shape, float)
    df = float(np.asarray(df).reshape(-1)[0])
    if shape.ndim == 0:
        shape = np.eye(dimension) * shape
    elif shape.ndim == 1:
        shape = np.diag(shape)
    if shape.shape != (dimension, dimension) or not np.isfinite(shape).all():
        raise ValueError("Invalid Student-t scatter")
    if dimension == 1 and x.ndim == 1:
        x = x[:, None]
    if math.isinf(df) and df > 0:
        from scipy.stats import multivariate_normal
        return multivariate_normal.logpdf(x, loc, shape, allow_singular=allow_singular)
    constant = student_log_normalizer(df, dimension)
    values, vectors = eigh(shape, lower=True, check_finite=False)
    tolerance = 1e6 * np.finfo(float).eps * np.max(np.abs(values))
    positive = values > tolerance
    if values.min() < -tolerance or (not allow_singular and not positive.all()):
        raise np.linalg.LinAlgError("Student-t scatter is not positive definite")
    white = (x - loc) @ (vectors[:, positive] / np.sqrt(values[positive]))
    delta = np.square(white).sum(axis=-1)
    logdet = np.log(values[positive]).sum()
    return np.squeeze(constant - 0.5 * logdet - (0.5 * df + 0.5 * dimension) * np.log1p(delta / df))


def student_negative_loglikelihood(u, x, weights):
    """Weighted negative log density and gradient in Cholesky/log-df coordinates."""
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
        white = solve_triangular(chol, (x - u[:d]).T, lower=True, check_finite=False)
        delta = np.sum(white * white, axis=0)
        constant = student_log_normalizer(nu, d) - np.sum(u[d:2*d])
        log_density = constant - (0.5 * nu + 0.5 * d) * np.log1p(delta / nu)
        value = -np.sum(weights * log_density)
        precision_residual = solve_triangular(chol.T, white, lower=False, check_finite=False).T
        weighted_tau = weights * ((nu + d) / (nu + delta))
        g_mu = np.sum(weighted_tau[:, None] * precision_residual, axis=0)
        inverse_chol = solve_triangular(chol, np.eye(d), lower=True, check_finite=False)
        precision = inverse_chol.T @ inverse_chol
        g_chol = (precision_residual.T @ (weighted_tau[:, None] * precision_residual)
                  - weights.sum() * precision) @ chol
        g_log_diag = np.diag(g_chol) * np.diag(chol)
        gradient = -np.concatenate([g_mu, g_log_diag, g_chol[lower],
                                    [np.sum(weights * student_lognu_score(delta, nu, d))]])
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise ValueError("Nonfinite Student-t objective or gradient")
    return float(value), gradient
