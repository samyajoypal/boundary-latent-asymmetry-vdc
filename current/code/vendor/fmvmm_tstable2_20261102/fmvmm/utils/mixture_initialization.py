"""An additional label-free start for tail-dominated mixture fits."""
from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture


def central_gmm_partition(data, n_components):
    """Initialize on a fixed central 80% subset, then assign EVERY observation.

    This only proposes starting groups. The component fits, mixture likelihood,
    and comparison with the ordinary GMM start use the complete training bank.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError("Initialization requires a finite two-dimensional data matrix")
    if not 1 <= n_components <= len(data):
        raise ValueError("Invalid number of mixture components")
    median = np.median(data, axis=0)
    mad = np.maximum(np.median(np.abs(data - median), axis=0), 1e-6)
    radius = np.linalg.norm((data - median) / mad, axis=1)
    central = radius <= np.quantile(radius, .8)
    if central.sum() < n_components:
        raise ValueError("Too few observations for the central-data initialization")
    model = GaussianMixture(n_components=n_components, covariance_type="full")
    model.fit(data[central])
    return model.predict(data)


def select_likelihood_start(results):
    """Choose a successful finite-likelihood start; ties retain the first start."""
    valid = [r for r in results if r.get("worked") and np.isfinite(r["log_likelihood"])]
    if not valid:
        raise ValueError("No start produced a finite fitted training likelihood")
    return max(valid, key=lambda r: r["log_likelihood"])
