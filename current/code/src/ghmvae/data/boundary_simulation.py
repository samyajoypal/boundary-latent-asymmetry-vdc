"""Information-preserving observations and independent known-density scoring."""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.special import gammaln, kve, logsumexp
from scipy.stats import multivariate_normal


def observation_map(latent, input_dim=16, strength=.35, noise=.05,
                    map_seed=44501, noise_seed=0):
    z = np.asarray(latent, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != 2 or input_dim <= 2:
        raise ValueError("Expected two latent dimensions and more than two observed dimensions")
    if strength < 0 or noise < 0 or not np.isfinite(z).all():
        raise ValueError("Nonnegative strengths and finite latent values are required")
    rng = np.random.default_rng(map_seed)
    q, _ = np.linalg.qr(rng.normal(size=(input_dim, input_dim)))
    a, b = q[:, :2].T, q[:, 2:].T
    w = rng.normal(size=(2, input_dim - 2)) / math.sqrt(2)
    scale = math.sqrt(input_dim / 2)
    # AA'=I, BA'=0: the noiseless map has the exact left inverse x A'/scale.
    raw = scale * (z @ a + strength * np.tanh(z @ w) @ b)
    raw += scale * noise * np.random.default_rng(noise_seed).normal(size=raw.shape)
    center = raw.mean(0)
    rms = float(np.sqrt(np.mean((raw - center)**2)))
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError("Degenerate observation scale")
    x = (raw - center) / rms
    recovered = ((x * rms + center) / scale) @ a.T
    metadata = {"decoder": "orthogonal_residual", "map_seed": map_seed,
                "strength": strength, "noise_before_global_standardization": noise,
                "scale": scale, "center": center.tolist(), "rms": rms,
                "linear_basis": a.tolist(), "residual_basis": b.tolist(),
                "residual_weights": w.tolist(),
                "noiseless_inverse": "z=((x*rms+center)/scale) @ linear_basis.T",
                "noiseless_lower_lipschitz_bound": scale / rms,
                "latent_recovery_mse": float(np.mean((recovered - z)**2))}
    return torch.tensor(x, dtype=torch.float32), metadata


def component_log_density(z, component):
    """Double-precision GH integral, independent of the production Torch scorer."""
    z = np.asarray(z, dtype=np.float64)
    mu, sigma = np.asarray(component["mu"]), np.asarray(component["sigma"])
    if component["family"] == "gaussian":
        return multivariate_normal.logpdf(z, mean=mu, cov=sigma)
    gamma = np.asarray(component["beta"])
    lmbda, chi, psi = component["lambda"], component["chi"], component["psi"]
    inverse, d = np.linalg.inv(sigma), z.shape[1]
    delta = z - mu
    a = chi + np.einsum("ni,ij,nj->n", delta, inverse, delta)
    b = float(psi + gamma @ inverse @ gamma)
    v = lmbda - d / 2
    if chi > 0 and psi > 0:
        argument = math.sqrt(chi * psi)
        log_c = lmbda / 2 * math.log(psi / chi) - math.log(2) - math.log(kve(lmbda, argument)) + argument
    elif psi == 0 and chi > 0 and lmbda < 0:
        log_c = -lmbda * math.log(chi / 2) - gammaln(-lmbda)
    else:
        raise ValueError("Unsupported calibration mixing law")
    if b == 0:
        log_integral = gammaln(-v) + v * np.log(a / 2)
    else:
        argument = np.sqrt(a * b)
        log_integral = math.log(2) + v / 2 * np.log(a / b) + np.log(kve(v, argument)) - argument
    result = (log_c - d / 2 * math.log(2 * math.pi)
              - .5 * np.linalg.slogdet(sigma)[1] + delta @ inverse @ gamma + log_integral)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite known-density evaluation")
    return result


def true_posterior(z, components):
    joints = np.column_stack([component_log_density(z, c) + math.log(c["weight"]) for c in components])
    score = logsumexp(joints, axis=1)
    return np.exp(joints - score[:, None]), score
