"""Stable Bessel calculations for generalized inverse Gaussian expectations."""
from __future__ import annotations

import numpy as np
from scipy.special import kve


def log_bessel_k(order, argument):
    """Return log K_order(argument) without exponential tail underflow."""
    return np.log(kve(order, argument)) - np.asarray(argument)


def log_bessel_order_derivative(order, argument):
    """Central difference of log K with respect to its real-valued order.

    The exponential scaling is independent of order, so differentiate log kve
    directly. This avoids both a 0/0 ratio and subtraction of a large argument.
    """
    order = np.asarray(order, dtype=float)
    step = np.cbrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(order))
    return (np.log(kve(order + step, argument))
            - np.log(kve(order - step, argument))) / (2 * step)
