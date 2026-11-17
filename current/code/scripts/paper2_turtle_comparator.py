"""Run the unmodified official TURTLE optimizer on our matched feature inputs."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from scripts.paper2_fit_diagnostics import sha256, write_json
from scripts.run_realdata_benchmark import _metrics


def load_official(vendor):
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    utils = load("turtle_official_utils", vendor / "utils.py")
    with patch.dict(sys.modules, {"utils": utils}):
        return load("turtle_official_run", vendor / "run_turtle.py")


def fit_turtle(x, indices, classes, seed, destination, *, device="cuda", iterations=6000, inner_steps=10):
    """No class labels enter fitting; final weights, not best labelled epoch."""
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("TURTLE confirmation requires CUDA")
    vendor = Path("vendor/turtle_reference")
    official = load_official(vendor)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    official.datasets_to_c["matched_features"] = int(classes)
    official.get_cluster_acc = lambda *_: (float("nan"), float("nan"))
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="paper2-turtle-") as temporary:
        root = Path(temporary)
        feature = root / "representations" / "matched_input"
        feature.mkdir(parents=True)
        (root / "labels").mkdir()
        np.save(feature / "matched_features_train.npy", x[indices])
        np.save(feature / "matched_features_val.npy", x)
        np.save(root / "labels" / "matched_features_val.npy", np.zeros(len(x), dtype=np.int64))
        args = SimpleNamespace(dataset="matched_features", phis=["matched_input"], gamma=10., T=iterations,
                               inner_lr=.001, outer_lr=.001, batch_size=10000, warm_start=False,
                               M=inner_steps, cross_val=False, device=device, root_dir=str(root), seed=int(seed))
        official._parse_args = lambda _: args
        official.run()
        checkpoints = list((root / "task_checkpoints").rglob("*.pt"))
        if len(checkpoints) != 1:
            raise ValueError("Expected the single fixed-final-iteration checkpoint")
        state = torch.load(checkpoints[0], map_location="cpu", weights_only=False)["phi1"]
    model = torch.nn.Linear(x.shape[1], classes).to(device)
    model.load_state_dict(state)
    with torch.no_grad():
        probabilities = torch.cat([model(torch.from_numpy(x[i:i + 1024]).to(device)).softmax(1).cpu()
                                   for i in range(0, len(x), 1024)]).numpy()
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite TURTLE predictions")
    torch.save(state, destination / "checkpoint.pt")
    return probabilities, {"runtime_seconds": time.perf_counter() - start,
                           "upstream_commit": "4674efab38e8f3cd0a4cd1f25d3ffc88be3dcabc",
                           "upstream_source_sha256": {name: sha256(vendor / name) for name in ("run_turtle.py", "utils.py")},
                           "configuration": vars(args), "evaluation_labels_used_during_fit": False,
                           "scope": "Official single-space TURTLE optimization with fixed defaults, matched input features and optimization rows. Not a reproduction of published CLIP/DINO feature results. Upstream source is retained for local research; redistribution requires a license review."}
