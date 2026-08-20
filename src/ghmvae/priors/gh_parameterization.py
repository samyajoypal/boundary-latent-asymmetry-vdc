from __future__ import annotations

import math


def _bessel_ratio(lmbda: float, alpha_bar: float) -> float:
    from scipy.special import kve

    denominator = float(kve(float(lmbda), float(alpha_bar)))
    numerator = float(kve(float(lmbda) + 1.0, float(alpha_bar)))
    ratio = numerator / denominator
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise FloatingPointError("Could not evaluate the positive Bessel-K ratio.")
    return ratio


def unit_mean_gig_parameters(lmbda: float, alpha_bar: float) -> tuple[float, float]:
    """Return ``(chi, psi)`` for the canonical ``E(W)=1`` GIG law.

    ``alpha_bar=0`` denotes the variance-gamma or Student-t boundary. The
    Student-t boundary requires more than two degrees of freedom because its
    inverse-gamma mixing variable must have a finite mean.
    """

    lmbda = float(lmbda)
    alpha_bar = float(alpha_bar)
    if alpha_bar < 0.0:
        raise ValueError("alpha_bar must be non-negative.")
    if alpha_bar == 0.0:
        if lmbda > 0.0:
            return 0.0, 2.0 * lmbda
        if lmbda < -1.0:
            return -2.0 * (lmbda + 1.0), 0.0
        raise ValueError(
            "A unit-mean GIG boundary requires lambda > 0 (VG) or lambda < -1 (Student-t)."
        )

    ratio = _bessel_ratio(lmbda, alpha_bar)
    return alpha_bar / ratio, alpha_bar * ratio


def gig_mean(lmbda: float, chi: float, psi: float) -> float:
    """Evaluate the GIG mean, including the VG and Student-t boundaries."""

    lmbda = float(lmbda)
    chi = float(chi)
    psi = float(psi)
    if chi < 0.0 or psi < 0.0:
        raise ValueError("chi and psi must be non-negative.")
    if chi > 0.0 and psi > 0.0:
        alpha_bar = math.sqrt(chi * psi)
        return math.sqrt(chi / psi) * _bessel_ratio(lmbda, alpha_bar)
    if chi == 0.0 and psi > 0.0 and lmbda > 0.0:
        return 2.0 * lmbda / psi
    if psi == 0.0 and chi > 0.0 and lmbda < -1.0:
        return chi / (-2.0 * (lmbda + 1.0))
    raise ValueError("The supplied GIG parameters do not have a finite mean in a supported regime.")
