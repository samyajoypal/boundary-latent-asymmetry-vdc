from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
FMVMM = Path(os.environ["FMVMM_PATH"]).expanduser() if os.environ.get("FMVMM_PATH") else None
for path in (SRC, FMVMM):
    if path is not None and path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fmvmm.distributions import multivariate_genhyperbolic as mghp
from fmvmm.distributions import multivariate_genskewt as mgst
from fmvmm.distributions import multivariate_hyperbolic as mvhb
from fmvmm.distributions import multivariate_norm as mvn
from fmvmm.distributions import multivariate_norminvgauss as mnig
from fmvmm.distributions import multivariate_skew_laplace as mvsl
from fmvmm.distributions import multivariate_skewnorm as mvsn
from fmvmm.distributions import multivariate_t as mvt
from fmvmm.distributions import multivariate_vargamma as mvvg
from ghmvae.priors.torch_fmvmm import component_from_fmvmm


def _check(name: str, dist_module, alpha, x: np.ndarray) -> dict[str, float | str]:
    torch_component = component_from_fmvmm(dist_module, alpha)
    z = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    torch_lp = torch_component.log_prob(z)
    torch_lp.sum().backward()
    ref_lp = np.asarray(dist_module.logpdf(x, *alpha), dtype=float)
    diff = np.abs(torch_lp.detach().numpy() - ref_lp)
    grad_ok = bool(torch.isfinite(z.grad).all())
    return {
        "name": name,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "grad_ok": grad_ok,
    }


def main() -> None:
    x = np.array(
        [
            [-1.0, 0.2],
            [0.1, -0.4],
            [1.5, 0.8],
            [2.2, -1.0],
        ],
        dtype=float,
    )
    mu = np.array([0.2, -0.1])
    sigma = np.array([[1.4, 0.25], [0.25, 0.9]])
    gamma = np.array([0.25, -0.15])
    shape = np.array([1.0, -0.8])

    cases = [
        ("mvn", mvn, (mu, sigma)),
        ("mvt", mvt, (mu, sigma, np.array([5.0]))),
        ("mvsn", mvsn, (mu, sigma, shape)),
        ("mvsl", mvsl, (mu, sigma, gamma)),
        ("mghp", mghp, (0.7, 1.3, 2.2, mu, sigma, gamma)),
        ("mgst", mgst, (-2.4, 4.0, mu, sigma, gamma)),
        ("mvhb", mvhb, (1.2, 2.4, mu, sigma, gamma)),
        ("mnig", mnig, (1.1, 1.1, mu, sigma, gamma)),
        ("mvvg", mvvg, (1.4, 2.3, mu, sigma, gamma)),
    ]
    results = [_check(*case, x=x) for case in cases]
    print(json.dumps(results, indent=2))
    failures = [r for r in results if r["max_abs_diff"] > 5e-4 or not r["grad_ok"]]
    if failures:
        raise SystemExit(f"Torch/FMVM density check failed: {failures}")


if __name__ == "__main__":
    main()
