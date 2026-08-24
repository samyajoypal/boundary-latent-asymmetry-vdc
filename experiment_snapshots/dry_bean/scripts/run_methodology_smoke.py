from __future__ import annotations

import copy
import json
import math
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ghmvae.data import make_skewed_heavy_mixture
from ghmvae.evaluation import best_map_accuracy, cluster_purity
from ghmvae.models import MLPVAE
from ghmvae.priors import DiagonalGaussianMixtureFitter
from ghmvae.priors.fmvmm_adapter import FMVMMPriorFitter
from ghmvae.training import AlternatingTrainConfig, TrainConfig, alternating_train, encode_dataset


def _evaluate(model: MLPVAE, prior, x: torch.Tensor, labels: torch.Tensor) -> dict[str, object]:
    encoded = encode_dataset(model, x, batch_size=128)
    resp = prior.responsibilities(encoded["mu"])
    pred = resp.argmax(dim=1)
    return {
        "responsibility_shape": list(resp.shape),
        "assigned_clusters": sorted(int(v) for v in torch.unique(pred).tolist()),
        "purity": cluster_purity(labels, pred),
        "best_map_accuracy": best_map_accuracy(labels, pred),
    }


def _run(label: str, model: MLPVAE, fitter, x: torch.Tensor, labels: torch.Tensor) -> dict[str, object]:
    config = AlternatingTrainConfig(
        pretrain=TrainConfig(epochs=4, batch_size=96, lr=1e-3, beta=1.0, seed=101),
        finetune=TrainConfig(epochs=2, batch_size=96, lr=8e-4, beta=1.0, seed=103),
        cycles=1,
    )
    prior, history = alternating_train(model, x, fitter, config)
    train_tail = history["cycles"][-1]["train"][-1]
    evaluation = _evaluate(model, prior, x, labels)
    diagnostics = history["cycles"][-1]["fit_diagnostics"]
    out = {
        "label": label,
        "prior_name": prior.name,
        "final_loss": train_tail["loss"],
        "final_elbo": train_tail["elbo"],
        "fit_diagnostics": diagnostics,
        **evaluation,
    }
    if not math.isfinite(float(out["final_loss"])):
        raise SystemExit(f"{label} produced a non-finite loss.")
    if len(out["assigned_clusters"]) < 2:
        raise SystemExit(f"{label} collapsed to one assigned cluster.")
    return out


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(31)
    x, labels, _ = make_skewed_heavy_mixture(
        n_samples=210,
        input_dim=8,
        n_components=3,
        skew_strength=0.9,
        contamination=0.04,
        seed=31,
    )
    base_model = MLPVAE(input_dim=x.shape[1], latent_dim=2, hidden_dims=(40, 40))
    baseline_model = copy.deepcopy(base_model)
    proposed_model = copy.deepcopy(base_model)

    baseline = _run(
        "diag_gmm_baseline",
        baseline_model,
        DiagonalGaussianMixtureFitter(n_components=3, n_iter=20, seed=37),
        x,
        labels,
    )

    gh_candidates = (
        ("mgst", "mgst", "mgst"),
        ("mnig", "mnig", "mnig"),
        ("mvvg", "mvvg", "mvvg"),
        ("mvhb", "mvhb", "mvhb"),
        ("mghp", "mghp", "mghp"),
    )
    proposed = _run(
        "fmvmm_gh_family",
        proposed_model,
        FMVMMPriorFitter(
            n_components=3,
            fmvmm_path=os.environ.get("FMVMM_PATH"),
            list_of_dist=("mgst", "mnig", "mvvg", "mvhb", "mghp"),
            candidate_combinations=gh_candidates,
            max_iter=4,
            tol=1e-3,
            gh_mstep_kwargs={"nit": 30, "reltol": 1e-4, "abstol": 1e-5, "silent": True},
            verbose=False,
            debug=False,
            differentiable=True,
            require_differentiable=True,
        ),
        x,
        labels,
    )

    summary = {
        "dataset": "skewed_heavy_nonlinear_synthetic",
        "baseline": baseline,
        "proposed": proposed,
        "accuracy_delta": proposed["best_map_accuracy"] - baseline["best_map_accuracy"],
        "purity_delta": proposed["purity"] - baseline["purity"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not proposed["fit_diagnostics"].get("differentiable", False):
        raise SystemExit("Proposed FMVMM prior was not differentiable.")


if __name__ == "__main__":
    main()
