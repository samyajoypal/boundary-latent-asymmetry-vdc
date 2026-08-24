from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from ghmvae.data import make_nonlinear_mixture
from ghmvae.evaluation import best_map_accuracy, cluster_purity
from ghmvae.models import MLPVAE
from ghmvae.priors import DiagonalGaussianMixtureFitter
from ghmvae.training import AlternatingTrainConfig, TrainConfig, alternating_train, encode_dataset


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(5)
    x, labels, _ = make_nonlinear_mixture(n_samples=240, input_dim=8, n_components=3, seed=7)
    model = MLPVAE(input_dim=x.shape[1], latent_dim=2, hidden_dims=(32, 32))
    fitter = DiagonalGaussianMixtureFitter(n_components=3, n_iter=15, seed=11)
    config = AlternatingTrainConfig(
        pretrain=TrainConfig(epochs=3, batch_size=64, lr=1e-3, beta=1.0, seed=13),
        finetune=TrainConfig(epochs=3, batch_size=64, lr=8e-4, beta=1.0, seed=17),
        cycles=2,
    )
    prior, history = alternating_train(model, x, fitter, config)
    encoded = encode_dataset(model, x, batch_size=128)
    resp = prior.responsibilities(encoded["mu"])
    pred = resp.argmax(dim=1)

    final_loss = history["cycles"][-1]["train"][-1]["loss"]
    summary = {
        "final_prior": prior.name,
        "responsibility_shape": list(resp.shape),
        "assigned_clusters": sorted(int(v) for v in torch.unique(pred).tolist()),
        "purity": cluster_purity(labels, pred),
        "best_map_accuracy": best_map_accuracy(labels, pred),
        "final_loss": final_loss,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if not math.isfinite(final_loss):
        raise SystemExit("Smoke test failed: final loss is not finite.")
    if resp.shape != (x.shape[0], 3):
        raise SystemExit(f"Smoke test failed: bad responsibility shape {tuple(resp.shape)}.")
    if len(summary["assigned_clusters"]) < 2:
        raise SystemExit("Smoke test failed: fitted prior collapsed to one assigned cluster.")


if __name__ == "__main__":
    main()
