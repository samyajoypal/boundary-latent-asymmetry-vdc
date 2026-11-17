"""Opt-in immutable pretraining checkpoints for sequential matched comparisons."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .mixture import pretrain_deterministic_latent


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    header = json.dumps([str(value.dtype), list(value.shape)]).encode()
    return hashlib.sha256(header + value.numpy().tobytes()).hexdigest()


def state_hash(state):
    return hashlib.sha256(json.dumps({key: tensor_hash(value) for key, value in state.items()},
                                    sort_keys=True).encode()).hexdigest()


def shared_pretrain(model, x, config, path, trainer=pretrain_deterministic_latent):
    """Reuse only when data, initial weights, architecture and settings match.

    Callers must serialize methods within a dataset/seed task. No labels enter
    the cache key. Checkpoints are local experiment artifacts, not untrusted input.
    """
    path = Path(path)
    metadata = {"schema_version": 1, "data_sha256": tensor_hash(x),
                "initial_state_sha256": state_hash(model.state_dict()),
                "architecture": repr(model), "config": asdict(config),
                "torch_version": str(torch.__version__), "torch_threads": torch.get_num_threads(),
                "training_source_sha256": hashlib.sha256(
                    (Path(__file__).parent / "mixture.py").read_bytes()).hexdigest()}
    fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    loaded = path.exists()
    if loaded:
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["fingerprint"] != fingerprint or saved["metadata"] != metadata:
            raise ValueError("Shared pretraining checkpoint does not match data/model/configuration")
        if state_hash(saved["model_state_dict"]) != saved["state_sha256"]:
            raise ValueError("Shared pretraining checkpoint state hash mismatch")
        model.to(config.device)
        model.load_state_dict(saved["model_state_dict"], strict=True)
        torch.set_rng_state(saved["torch_rng_state"])
        if saved["cuda_rng_states"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng_states"])
        model.train(saved["training"])
    else:
        start = time.perf_counter()
        history = trainer(model, x, config)
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        saved = {"metadata": metadata, "fingerprint": fingerprint, "model_state_dict": state,
                 "state_sha256": state_hash(state), "history": history,
                 "training_seconds": time.perf_counter() - start,
                 "training": model.training, "torch_rng_state": torch.get_rng_state(),
                 "cuda_rng_states": torch.cuda.get_rng_state_all() if str(config.device).startswith("cuda") else None}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            torch.save(saved, handle)
    receipt = {"fingerprint": fingerprint, "state_sha256": saved["state_sha256"],
               "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "loaded": loaded, "shared_training_seconds": saved["training_seconds"]}
    return saved["history"], receipt
