"""Pinned public text inputs and explicitly supervised pretrained image features."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

from ghmvae.data.representation_benchmarks import cache_paths, file_sha256, load_benchmark


REUTERS_URL = "https://raw.githubusercontent.com/slim1017/VaDE/master/dataset/reuters10k/reuters10k.mat"
REUTERS_SHA256 = "aa0774a824889efe0597f0d42c6a795c99f312f89819afde957746fb7f0588b1"
EXPECTED = {"reuters10k": (10000, 2000, 4), "stl10_resnet50": (13000, 2048, 10),
            "cifar10_resnet50": (60000, 2048, 10), "cifar10020_resnet50": (60000, 2048, 20)}


def parse_reuters(path):
    from scipy.io import loadmat

    if file_sha256(path) != REUTERS_SHA256:
        raise ValueError("Reuters-10K differs from the pinned original VaDE release")
    values = loadmat(path)
    x, labels = np.asarray(values["X"], dtype=np.float32), np.asarray(values["Y"]).reshape(-1)
    if x.shape != (10000, 2000) or labels.shape != (10000,) or not np.isfinite(x).all():
        raise ValueError("Unexpected Reuters-10K data")
    if not np.array_equal(np.unique(labels), np.arange(4)):
        raise ValueError("Unexpected Reuters-10K labels")
    return {"x": x, "y": labels.astype(np.int64), "source_indices": np.arange(len(x))}


def scale_inputs(x, name):
    """A scalar rescaling preserves Euclidean distances up to a common factor."""
    values = np.asarray(x, dtype=np.float32)
    if name.endswith("_resnet50"):
        values = values - values.mean(axis=0, dtype=np.float64).astype(np.float32)
    rms = np.sqrt(np.mean(np.square(values), dtype=np.float64))
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError("Degenerate benchmark input")
    return (values / rms).astype(np.float32)


def save_cache(data_home, name, arrays, metadata):
    target, metadata_path = cache_paths(data_home, name)
    if target.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to replace a benchmark cache: {name}")
    n, d, k = EXPECTED[name]
    x, y = arrays["x"], arrays["y"]
    if x.shape != (n, d) or y.shape != (n,) or not np.isfinite(x).all():
        raise ValueError(f"Incorrect feature matrix for {name}")
    if not np.array_equal(np.unique(y), np.arange(k)):
        raise ValueError(f"Incorrect class coverage for {name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, target)
    metadata.update(schema_version=1, dataset=name, n_samples=n, input_dim=d, n_classes=k,
                    dataset_cache_sha256=file_sha256(target))
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def prepare_reuters(data_home):
    target, _ = cache_paths(data_home, "reuters10k")
    if target.exists():
        return load_benchmark(data_home, "reuters10k")[2]
    path = target.parent / "reuters10k.mat"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(".download")
        with urllib.request.urlopen(REUTERS_URL, timeout=180) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        os.replace(temporary, path)
    arrays = parse_reuters(path)
    return save_cache(data_home, "reuters10k", arrays, {
        "dataset_source": "original_vade_reuters10k_tfidf2000",
        "dataset_url": "https://github.com/slim1017/VaDE/tree/master/dataset/reuters10k",
        "source_archive_sha256": REUTERS_SHA256,
        "preprocessing": "Original 10000 by 2000 published TF-IDF matrix, unchanged row order; a single global RMS scale in the runner; no coordinate whitening or additional topic filtering.",
        "evaluation_scope": "Transductive clustering of the published four-topic subset. Original benchmark construction used topic labels; the new fitting and family selection do not.",
        "external_pretraining": "None; supplied text features only.",
        "redistribution": "Download from the original source; do not assume a redistribution license for source documents.",
    })


def prepare_image_features(data_home, name, image_home, device="cuda", batch_size=64):
    import torch
    import torchvision
    from torch.utils.data import ConcatDataset, DataLoader
    from torchvision.datasets import CIFAR10, CIFAR100, STL10
    from torchvision.models import ResNet50_Weights, resnet50

    if name not in EXPECTED or not name.endswith("_resnet50"):
        raise ValueError(f"Not an image feature benchmark: {name}")
    target, _ = cache_paths(data_home, name)
    if target.exists():
        return load_benchmark(data_home, name)[2]
    weights = ResNet50_Weights.IMAGENET1K_V1
    transform = weights.transforms()
    if name == "stl10_resnet50":
        parts = [STL10(str(image_home), split=s, transform=transform, download=True) for s in ("train", "test")]
        url = "https://cs.stanford.edu/~acoates/stl10/"
    else:
        dataset_class = CIFAR10 if name == "cifar10_resnet50" else CIFAR100
        parts = [dataset_class(str(image_home), train=s, transform=transform, download=True) for s in (True, False)]
        url = "https://www.cs.toronto.edu/~kriz/cifar.html"
    data = ConcatDataset(parts)
    # The checkpoint and preprocessing are fixed, never chosen using target labels.
    torch.manual_seed(16000)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    model = resnet50(weights=weights).to(device).eval()
    model.fc = torch.nn.Identity()
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
    features, labels = [], []
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=device.startswith("cuda"))
    with torch.inference_mode():
        for index, (images, batch_labels) in enumerate(loader):
            features.append(model(images.to(device)).cpu().numpy())
            labels.append(batch_labels.numpy())
            if index % 100 == 0:
                print(f"{name}: {min((index + 1) * batch_size, len(data))}/{len(data)}", flush=True)
    y = np.concatenate(labels).astype(np.int64)
    if name == "cifar10020_resnet50":
        from scripts.run_realdata_benchmark import CIFAR100_FINE_TO_COARSE
        y = np.asarray(CIFAR100_FINE_TO_COARSE, dtype=np.int64)[y]
    arrays = {"x": np.concatenate(features).astype(np.float32), "y": y,
              "source_indices": np.arange(len(data)),
              "original_partition": np.concatenate([np.full(len(p), i, dtype=np.int8) for i, p in enumerate(parts)])}
    return save_cache(data_home, name, arrays, {
        "dataset_source": name + "_all_train_test_fixed_features", "dataset_url": url,
        "preprocessing": "All original train then test images; fixed ImageNet resize/crop/normalization; global-average-pooled 2048 features; feature centering and a single RMS scale in the runner. No target-label fitting, augmentation selection, class filtering, or subsampling.",
        "evaluation_scope": "Transductive clustering of all train/test images; no STL-10 unlabeled extra split. Feature-space VAE, not raw-pixel generation or unseen-image evaluation.",
        "external_pretraining": "Supervised ImageNet-1K ResNet-50; source classes overlap semantically with target classes. Shared frozen extractor for every comparator; not a fully unsupervised image representation.",
        "feature_extractor": "torchvision.models.resnet50:IMAGENET1K_V1",
        "feature_checkpoint_url": weights.url, "feature_checkpoint_sha256": file_sha256(checkpoint),
        "feature_transform": repr(transform), "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "redistribution": "Download original images from their providers; do not redistribute raw image archives in the code release.",
    })
