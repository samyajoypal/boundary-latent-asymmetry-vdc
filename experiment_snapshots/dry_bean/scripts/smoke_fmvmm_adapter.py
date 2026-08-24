from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ghmvae.data import make_nonlinear_mixture
from ghmvae.priors.fmvmm_adapter import FMVMMPriorFitter


def main() -> None:
    _, labels, z = make_nonlinear_mixture(n_samples=90, n_components=3, seed=19)
    fitter = FMVMMPriorFitter(
        n_components=3,
        fmvmm_path=os.environ.get("FMVMM_PATH"),
        list_of_dist=("mvn",),
        candidate_combinations=(("mvn", "mvn", "mvn"),),
        max_iter=5,
        verbose=False,
        debug=False,
    )
    result = fitter.fit(z)
    prior = result.prior
    responsibilities = prior.responsibilities(z)
    pred = responsibilities.argmax(dim=1)
    summary = {
        "best_mixture": result.diagnostics["best_mixture"],
        "differentiable": result.diagnostics["differentiable"],
        "responsibility_shape": list(responsibilities.shape),
        "assigned_clusters": sorted(int(v) for v in set(pred.tolist())),
        "label_count": int(labels.numel()),
        "best_bic": result.diagnostics["best_bic"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if responsibilities.shape != (z.shape[0], 3):
        raise SystemExit("Bad responsibility shape from FMVMM adapter.")
    if not result.diagnostics["differentiable"]:
        raise SystemExit("FMVMM adapter did not return a differentiable torch prior.")
    if len(summary["assigned_clusters"]) < 2:
        raise SystemExit("FMVMM adapter collapsed to one assigned cluster.")


if __name__ == "__main__":
    main()
