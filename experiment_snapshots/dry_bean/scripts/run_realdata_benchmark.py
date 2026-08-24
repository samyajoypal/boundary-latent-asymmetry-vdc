from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import sys
import time
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.io import arff
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ghmvae.models import AffineGaugeVAE, ConvContrastiveClusterer, ConvVAE, MLPVAE
from ghmvae.data import unlabeled_optimization_selection_split
from ghmvae.experiments import json_safe, write_json
from ghmvae.priors import (
    ComponentwiseEnrichmentConfig,
    DiagonalGaussianMixtureFitter,
    LearnableDiagonalGaussianMixturePrior,
    PriorCandidate,
    StandardNormalPrior,
    fit_and_select_prior,
    fit_componentwise_gh_enrichment,
    prior_assignment_diagnostics,
    prior_regularity_diagnostics,
)
from ghmvae.priors.fmvmm_adapter import FMVMMPriorFitter
from ghmvae.training import (
    AlternatingTrainConfig,
    DeterministicPretrainConfig,
    MixtureTrainConfig,
    TrainConfig,
    accept_prior_candidate,
    accepted_prior_refit,
    alternating_train,
    diagonal_gaussian_forward_kl,
    encode_dataset,
    encode_mixture_responsibilities,
    posterior_sample_bank,
    pretrain_deterministic_latent,
    train_mixture_vae,
    train_vae,
)


SKLEARN_DATASETS = {
    "digits": datasets.load_digits,
    "wine": datasets.load_wine,
    "breast_cancer": datasets.load_breast_cancer,
    "iris": datasets.load_iris,
}

OPENML_DATASETS = {
    "mnist": {"name": "mnist_784", "version": 1},
    "fashion_mnist": {"name": "Fashion-MNIST", "version": 1},
    "usps": {"name": "usps", "version": 2},
}

FETCH_DATASETS = {
    "olivetti_faces",
    "olivetti_faces_10",
    "20newsgroups_10",
    "20newsgroups_20",
}

TORCHVISION_IMAGE_DATASETS = {
    "cifar10",
    "stl10",
    "cifar100_20",
}

UCI_TABULAR_DATASETS = {
    "dry_bean",
}

DRY_BEAN_URL = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
DRY_BEAN_ARCHIVE_SHA256 = "0a64eff5be87f48c3dbbfc0a12a56c5d5b5167ef8e61cd45d69b3e7c7130c06f"
DRY_BEAN_ARCHIVE_MEMBER = "DryBeanDataset/Dry_Bean_Dataset.arff"

SEMANTIC_OUTLIER_DEFAULTS = {
    "mnist": "fashion_mnist",
    "fashion_mnist": "mnist",
    "usps": "mnist",
    "cifar10": "cifar100_20",
    "cifar100_20": "cifar10",
    "stl10": "cifar10",
}

NEWSGROUPS_10_CATEGORIES = (
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "rec.autos",
    "rec.motorcycles",
    "sci.crypt",
    "sci.electronics",
    "sci.space",
)

DATASET_NAMES = tuple(
    sorted(
        set(SKLEARN_DATASETS)
        | set(OPENML_DATASETS)
        | FETCH_DATASETS
        | TORCHVISION_IMAGE_DATASETS
        | UCI_TABULAR_DATASETS
    )
)

CIFAR100_FINE_TO_COARSE = np.asarray(
    [
        4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
        3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
        6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
        0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
        5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
        16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
        10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
        2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
        16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
        18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
    ],
    dtype=np.int64,
)


DEFAULT_METHODS = (
    "raw_kmeans",
    "raw_gmm",
    "ae_kmeans",
    "dec",
    "idec",
    "conv_cc",
    "vae_kmeans",
    "vae_gmm",
    "vae_diag_gmm",
    "vae_fmvmm_mvt",
    "vae_fmvmm_gh_family",
    "diag_gmm_prior",
    "fmvmm_mvn",
    "fmvmm_mvt",
    "fmvmm_mvsn",
    "fmvmm_mgst",
    "fmvmm_gh_family",
    "vade_diag",
    "ghvade_mvsn",
    "ghvade_mvt",
    "ghvade_gh_family",
    "ghvade_tail_select",
    "ghvade_conservative",
    "ghvade_adaptive",
    "ghvade_guarded_adaptive",
)


FMVMM_METHOD_FAMILIES = {
    "fmvmm_mvn": ("mvn",),
    "fmvmm_mvt": ("mvt",),
    "fmvmm_mvsn": ("mvsn",),
    "fmvmm_mgst": ("mgst",),
    "fmvmm_mnig": ("mnig",),
    "fmvmm_mvvg": ("mvvg",),
    "fmvmm_mvhb": ("mvhb",),
    "fmvmm_mghp": ("mghp",),
    "fmvmm_gh_family": ("mvn", "mvt", "mvsn", "mgst", "mnig", "mvvg", "mvhb", "mghp"),
}

ADAPTIVE_FAMILY_ORDER = {
    "mvn": ("gaussian", 0),
    "mvt": ("student_t", 1),
    "mvsn": ("skew_normal", 2),
    "mvvg": ("variance_gamma", 3),
    "mvhb": ("hyperbolic", 4),
    "mnig": ("nig", 5),
    "mgst": ("skew_t", 6),
    "mghp": ("full_gh", 7),
}

ADAPTIVE_FAMILY_CODE = {
    name: code for code, (name, _) in ADAPTIVE_FAMILY_ORDER.items()
}


def _base_fmvmm_method(method: str) -> str:
    method = method.removesuffix("_mi")
    method = method.removeprefix("conv_")
    if method.startswith("ghvade_"):
        return "fmvmm_" + method.removeprefix("ghvade_")
    if method.startswith("vae_"):
        return method.removeprefix("vae_")
    if method.startswith("latent_"):
        return method.removeprefix("latent_")
    return method


def _load_torchvision_image_dataset(
    name: str,
    data_home: str | None,
    download_datasets: bool,
) -> tuple[np.ndarray, np.ndarray, str, tuple[int, int, int]]:
    try:
        from torchvision import datasets as tv_datasets
    except ImportError as exc:
        raise ImportError(
            f"Dataset {name!r} requires torchvision. Install the optional vision dependencies."
        ) from exc
    root = data_home or str(ROOT / "data" / "torchvision")
    if name == "cifar10":
        train = tv_datasets.CIFAR10(root=root, train=True, download=download_datasets)
        test = tv_datasets.CIFAR10(root=root, train=False, download=download_datasets)
        x = np.concatenate([np.asarray(train.data), np.asarray(test.data)], axis=0)
        y = np.asarray(list(train.targets) + list(test.targets), dtype=np.int64)
        source = "torchvision_cifar10_train_test"
        image_shape = (3, 32, 32)
        x = np.transpose(x.astype(np.float32) / 255.0, (0, 3, 1, 2))
    elif name == "cifar100_20":
        train = tv_datasets.CIFAR100(root=root, train=True, download=download_datasets)
        test = tv_datasets.CIFAR100(root=root, train=False, download=download_datasets)
        x = np.concatenate([np.asarray(train.data), np.asarray(test.data)], axis=0)
        fine = np.asarray(list(train.targets) + list(test.targets), dtype=np.int64)
        y = CIFAR100_FINE_TO_COARSE[fine]
        source = "torchvision_cifar100_train_test_coarse20"
        image_shape = (3, 32, 32)
        x = np.transpose(x.astype(np.float32) / 255.0, (0, 3, 1, 2))
    elif name == "stl10":
        data = tv_datasets.STL10(root=root, split="train+test", download=download_datasets)
        x = np.asarray(data.data, dtype=np.float32) / 255.0
        y = np.asarray(data.labels, dtype=np.int64)
        source = "torchvision_stl10_train_test"
        image_shape = (3, 96, 96)
    else:
        raise ValueError(f"Unknown torchvision image dataset {name!r}.")
    return x.reshape(x.shape[0], -1), y, source, image_shape


def _resize_flat_images(
    x: np.ndarray,
    source_shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    if source_shape == target_shape:
        return x.astype(np.float32, copy=False)
    images = torch.from_numpy(x.reshape(x.shape[0], *source_shape)).float()
    if source_shape[0] != target_shape[0]:
        if source_shape[0] == 1 and target_shape[0] == 3:
            images = images.repeat(1, 3, 1, 1)
        elif source_shape[0] == 3 and target_shape[0] == 1:
            images = images.mean(dim=1, keepdim=True)
        else:
            raise ValueError(f"Cannot convert image channels from {source_shape} to {target_shape}.")
    if tuple(images.shape[2:]) != tuple(target_shape[1:]):
        images = F.interpolate(images, size=target_shape[1:], mode="bilinear", align_corners=False)
    return images.reshape(images.shape[0], -1).numpy().astype(np.float32)


def _sample_rows(x: np.ndarray, n_rows: int, rng: np.random.Generator) -> np.ndarray:
    replace = x.shape[0] < int(n_rows)
    indices = rng.choice(x.shape[0], size=int(n_rows), replace=replace)
    return x[indices].astype(np.float32, copy=True)


def _make_contamination_outliers(
    x: np.ndarray,
    n_outliers: int,
    rng: np.random.Generator,
    *,
    mode: str,
    corruption_severity: float,
) -> np.ndarray:
    if n_outliers <= 0:
        return np.empty((0, x.shape[1]), dtype=np.float32)
    severity = float(np.clip(corruption_severity, 0.0, 1.0))
    if mode == "uniform":
        low = np.min(x, axis=0)
        high = np.max(x, axis=0)
        return rng.uniform(low=low, high=high, size=(n_outliers, x.shape[1])).astype(np.float32)
    if mode == "gaussian_noise":
        base = _sample_rows(x, n_outliers, rng)
        scale = np.maximum(np.std(x, axis=0, dtype=np.float64), 1e-3).astype(np.float32)
        return (base + rng.normal(0.0, severity * scale, size=base.shape)).astype(np.float32)
    if mode == "salt_pepper":
        base = _sample_rows(x, n_outliers, rng)
        low = np.min(x, axis=0)
        high = np.max(x, axis=0)
        mask = rng.random(base.shape) < severity
        salt = rng.random(base.shape) < 0.5
        outliers = base.copy()
        outliers[mask & salt] = np.broadcast_to(high, base.shape)[mask & salt]
        outliers[mask & ~salt] = np.broadcast_to(low, base.shape)[mask & ~salt]
        return outliers.astype(np.float32)
    raise ValueError(f"Unsupported contamination mode {mode!r}.")


def _dry_bean_arff_path(
    data_home: str | None,
    download_datasets: bool,
) -> Path:
    root = Path(data_home) if data_home else ROOT / "data" / "uci"
    dataset_dir = root / "dry_bean"
    target = dataset_dir / "Dry_Bean_Dataset.arff"
    if target.exists():
        return target
    if not download_datasets:
        raise FileNotFoundError(
            f"Dry Bean data are not cached at {target}. Re-run with "
            "--download-datasets or use scripts/precache_datasets.py."
        )

    dataset_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dataset_dir / "dry_bean_dataset.zip"
    if not archive_path.exists():
        partial_path = archive_path.with_suffix(".zip.part")
        urllib.request.urlretrieve(DRY_BEAN_URL, partial_path)
        os.replace(partial_path, archive_path)
    archive_digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if archive_digest != DRY_BEAN_ARCHIVE_SHA256:
        raise ValueError(
            "Dry Bean archive checksum mismatch: "
            f"expected {DRY_BEAN_ARCHIVE_SHA256}, observed {archive_digest}."
        )
    with zipfile.ZipFile(archive_path) as archive:
        target.write_bytes(archive.read(DRY_BEAN_ARCHIVE_MEMBER))
    return target


def _load_dry_bean_dataset(
    data_home: str | None,
    download_datasets: bool,
) -> tuple[np.ndarray, np.ndarray]:
    path = _dry_bean_arff_path(data_home, download_datasets)
    table, _ = arff.loadarff(path)
    names = list(table.dtype.names or ())
    class_name = next((name for name in names if name.casefold() == "class"), None)
    if class_name is None:
        raise ValueError(f"Dry Bean ARFF at {path} has no Class field.")
    feature_names = [name for name in names if name != class_name]
    x = np.column_stack(
        [np.asarray(table[name], dtype=np.float64) for name in feature_names]
    ).astype(np.float32)
    raw_labels = np.asarray(table[class_name])
    labels = np.asarray(
        [
            value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
            for value in raw_labels
        ]
    )
    _, y = np.unique(labels, return_inverse=True)
    if x.ndim != 2 or x.shape[1] != 16:
        raise ValueError(f"Expected 16 Dry Bean features, observed shape {x.shape}.")
    if not np.isfinite(x).all():
        raise ValueError("Dry Bean features contain non-finite values.")
    return x, y.astype(np.int64)


def _load_base_dataset(
    name: str,
    seed: int,
    data_home: str | None,
    download_datasets: bool,
) -> tuple[np.ndarray, np.ndarray, str, tuple[int, int, int] | None]:
    image_shape: tuple[int, int, int] | None = None
    if name in SKLEARN_DATASETS:
        bunch = SKLEARN_DATASETS[name]()
        source = "sklearn"
        x = np.asarray(bunch.data, dtype=np.float32)
        y = np.asarray(bunch.target, dtype=np.int64)
        if name == "digits":
            image_shape = (1, 8, 8)
    elif name in OPENML_DATASETS:
        spec = OPENML_DATASETS[name]
        bunch = datasets.fetch_openml(
            name=str(spec["name"]),
            version=int(spec["version"]),
            as_frame=False,
            parser="auto",
            data_home=data_home,
        )
        source = "openml"
        x = np.asarray(bunch.data, dtype=np.float32)
        _, y = np.unique(np.asarray(bunch.target).astype(str), return_inverse=True)
        y = y.astype(np.int64)
        image_shape = (1, 28, 28) if name in {"mnist", "fashion_mnist"} else (1, 16, 16)
    elif name in TORCHVISION_IMAGE_DATASETS:
        x, y, source, image_shape = _load_torchvision_image_dataset(
            name, data_home, download_datasets
        )
    elif name == "dry_bean":
        x, y = _load_dry_bean_dataset(data_home, download_datasets)
        source = "uci_ml_repository_602_dry_bean"
    elif name in {"olivetti_faces", "olivetti_faces_10"}:
        bunch = datasets.fetch_olivetti_faces(data_home=data_home, shuffle=False)
        source = "sklearn_olivetti_faces"
        x = np.asarray(bunch.data, dtype=np.float32)
        y = np.asarray(bunch.target, dtype=np.int64)
        image_shape = (1, 64, 64)
        if name == "olivetti_faces_10":
            keep = y < 10
            x = x[keep]
            y = y[keep]
    elif name in {"20newsgroups_10", "20newsgroups_20"}:
        categories = NEWSGROUPS_10_CATEGORIES if name == "20newsgroups_10" else None
        bunch = datasets.fetch_20newsgroups(
            data_home=data_home,
            subset="all",
            categories=categories,
            remove=("headers", "footers", "quotes"),
        )
        source = "sklearn_20newsgroups_tfidf_svd"
        vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            stop_words="english",
            norm="l2",
        )
        sparse = vectorizer.fit_transform(bunch.data)
        max_components = max(2, min(256, sparse.shape[0] - 1, sparse.shape[1] - 1))
        x = TruncatedSVD(n_components=max_components, random_state=int(seed) + 41).fit_transform(sparse)
        x = x.astype(np.float32)
        y = np.asarray(bunch.target, dtype=np.int64)
    else:
        raise ValueError(f"Unknown dataset {name!r}. Available: {list(DATASET_NAMES)}")
    return x, y, source, image_shape


def _load_dataset(
    name: str,
    max_samples: int | None,
    seed: int,
    contamination: float = 0.0,
    data_home: str | None = None,
    download_datasets: bool = False,
    contamination_mode: str = "uniform",
    semantic_outlier_dataset: str | None = None,
    corruption_severity: float = 0.5,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, dict[str, object]]:
    x, y, source, image_shape = _load_base_dataset(
        name, seed, data_home, download_datasets
    )
    if max_samples is not None and x.shape[0] > int(max_samples):
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=int(max_samples), random_state=int(seed))
        idx, _ = next(splitter.split(x, y))
        x = x[idx]
        y = y[idx]
    eval_mask = np.ones(y.shape[0], dtype=bool)
    n_clean = int(x.shape[0])
    n_outliers = int(round(max(float(contamination), 0.0) * n_clean))
    if n_outliers > 0:
        rng = np.random.default_rng(int(seed) + 5003)
        mode = str(contamination_mode)
        if mode == "semantic":
            if image_shape is None:
                raise ValueError("Semantic outlier contamination requires image-shaped datasets.")
            outlier_name = semantic_outlier_dataset or SEMANTIC_OUTLIER_DEFAULTS.get(name)
            if outlier_name not in DATASET_NAMES:
                raise ValueError(f"No semantic outlier source configured for dataset {name!r}.")
            source_x, _, _, source_shape = _load_base_dataset(
                str(outlier_name), seed + 9001, data_home, download_datasets
            )
            if source_shape is None:
                raise ValueError(f"Semantic outlier source {outlier_name!r} is not image-shaped.")
            source_x = _resize_flat_images(source_x, source_shape, image_shape)
            outliers = _sample_rows(source_x, n_outliers, rng)
        elif name == "digits" and mode == "uniform":
            low = float(np.min(x))
            high = float(np.max(x))
            outliers = np.empty((n_outliers, x.shape[1]), dtype=np.float32)
            half = n_outliers // 2
            outliers[:half] = low
            outliers[half:] = high
            if n_outliers > 2:
                noisy = rng.uniform(low=low, high=high, size=outliers.shape).astype(np.float32)
                keep_solid = max(2, n_outliers // 3)
                outliers[keep_solid:] = noisy[keep_solid:]
        else:
            outliers = _make_contamination_outliers(
                x,
                n_outliers,
                rng,
                mode=mode,
                corruption_severity=corruption_severity,
            )
        x = np.vstack([x, outliers])
        y = np.concatenate([y, -np.ones(n_outliers, dtype=np.int64)])
        eval_mask = np.concatenate([eval_mask, np.zeros(n_outliers, dtype=bool)])
    x = StandardScaler().fit_transform(x).astype(np.float32)
    meta = {
        "dataset": name,
        "dataset_source": source,
        "n_samples": int(x.shape[0]),
        "n_eval_samples": int(eval_mask.sum()),
        "n_outliers": int(n_outliers),
        "contamination": float(contamination),
        "contamination_mode": str(contamination_mode),
        "semantic_outlier_dataset": (
            semantic_outlier_dataset
            or (SEMANTIC_OUTLIER_DEFAULTS.get(name, "") if contamination_mode == "semantic" else "")
        ),
        "corruption_severity": float(corruption_severity),
        "input_dim": int(x.shape[1]),
        "n_classes": int(np.unique(y[eval_mask]).size),
    }
    if name == "dry_bean":
        meta.update(
            {
                "dataset_doi": "10.24432/C50S4B",
                "dataset_url": "https://archive.ics.uci.edu/dataset/602/dry+bean+dataset",
                "dataset_license": "CC BY 4.0",
            }
        )
    if image_shape is not None:
        meta["image_shape"] = list(image_shape)
    return torch.from_numpy(x), y, eval_mask, meta


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    true_index = {label: i for i, label in enumerate(true_labels)}
    pred_index = {label: i for i, label in enumerate(pred_labels)}
    cm = np.zeros((len(true_labels), len(pred_labels)), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        cm[true_index[yt], pred_index[yp]] += 1
    if cm.size:
        row_ind, col_ind = linear_sum_assignment(cm.max() - cm)
        acc = float(cm[row_ind, col_ind].sum() / max(len(y_true), 1))
        purity = float(cm.max(axis=0).sum() / max(len(y_true), 1))
    else:
        acc = 0.0
        purity = 0.0
    return {
        "acc": acc,
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "purity": purity,
        "n_assigned": int(len(pred_labels)),
    }


EXTERNAL_METRIC_NAMES = frozenset({"acc", "ari", "nmi", "purity"})


def _is_external_metric_key(key: object) -> bool:
    name = str(key).casefold()
    return name in EXTERNAL_METRIC_NAMES or any(
        name.endswith(f"_{metric}") for metric in EXTERNAL_METRIC_NAMES
    )


def _strip_external_metrics(value: object) -> object:
    """Recursively remove class-label metrics while retaining validity fields."""

    if isinstance(value, dict):
        return {
            key: _strip_external_metrics(item)
            for key, item in value.items()
            if not _is_external_metric_key(key)
        }
    if isinstance(value, list):
        return [_strip_external_metrics(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_external_metrics(item) for item in value)
    return value


def _blind_evaluation_labels(
    labels: np.ndarray,
    eval_mask: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Destroy observation-label correspondence while preserving class counts."""

    labels = np.asarray(labels)
    eval_mask = np.asarray(eval_mask, dtype=bool)
    if labels.ndim != 1 or eval_mask.shape != labels.shape:
        raise ValueError("labels and eval_mask must be aligned one-dimensional arrays.")
    indices = np.flatnonzero(eval_mask)
    blinded = labels.copy()
    if indices.size <= 1:
        return blinded
    rng = np.random.default_rng(int(seed))
    permuted = labels[indices][rng.permutation(indices.size)]
    if np.array_equal(permuted, labels[indices]):
        permuted = np.roll(permuted, 1)
    blinded[indices] = permuted
    return blinded


def _soft_graph_assortativity(
    features: np.ndarray,
    responsibilities: np.ndarray,
    n_neighbors: int,
) -> dict[str, float | int]:
    """Measure label-free assignment agreement on a fixed feature graph."""

    features = np.asarray(features, dtype=np.float64)
    responsibilities = np.asarray(responsibilities, dtype=np.float64)
    if features.ndim != 2 or responsibilities.ndim != 2:
        raise ValueError("features and responsibilities must be matrices.")
    if features.shape[0] != responsibilities.shape[0]:
        raise ValueError("features and responsibilities need equal row counts.")
    n_observations = int(features.shape[0])
    if n_observations < 2:
        return {
            "graph_n_observations": n_observations,
            "graph_n_neighbors": 0,
            "graph_soft_edge_agreement": math.nan,
            "graph_soft_null_agreement": math.nan,
            "graph_soft_assortativity": math.nan,
            "graph_hard_edge_agreement": math.nan,
            "graph_hard_null_agreement": math.nan,
            "graph_hard_assortativity": math.nan,
        }
    if not np.isfinite(features).all() or not np.isfinite(responsibilities).all():
        raise ValueError("Graph diagnostics require finite inputs.")
    scale = features.std(axis=0, keepdims=True)
    standardized = (features - features.mean(axis=0, keepdims=True)) / np.maximum(
        scale, 1e-8
    )
    neighbor_count = min(max(int(n_neighbors), 1), n_observations - 1)
    indices = NearestNeighbors(n_neighbors=neighbor_count + 1).fit(
        standardized
    ).kneighbors(standardized, return_distance=False)[:, 1:]
    source = np.repeat(np.arange(n_observations), neighbor_count)
    target = indices.reshape(-1)

    soft_edge = float(
        np.einsum("ij,ij->i", responsibilities[source], responsibilities[target]).mean()
    )
    soft_mass = responsibilities.mean(axis=0)
    soft_null = float(np.square(soft_mass).sum())
    soft_score = float((soft_edge - soft_null) / max(1.0 - soft_null, 1e-12))

    hard = responsibilities.argmax(axis=1)
    hard_edge = float((hard[source] == hard[target]).mean())
    hard_mass = np.bincount(hard, minlength=responsibilities.shape[1]) / n_observations
    hard_null = float(np.square(hard_mass).sum())
    hard_score = float((hard_edge - hard_null) / max(1.0 - hard_null, 1e-12))
    return {
        "graph_n_observations": n_observations,
        "graph_n_neighbors": neighbor_count,
        "graph_soft_edge_agreement": soft_edge,
        "graph_soft_null_agreement": soft_null,
        "graph_soft_assortativity": soft_score,
        "graph_hard_edge_agreement": hard_edge,
        "graph_hard_null_agreement": hard_null,
        "graph_hard_assortativity": hard_score,
    }


def _graph_tournament_decision(
    candidate_score: float,
    reference_score: float,
    minimum_gain: float,
) -> tuple[bool, float]:
    """Apply the prespecified strict graph-improvement rule."""

    minimum_gain = float(minimum_gain)
    if minimum_gain < 0.0:
        raise ValueError("minimum graph-assortativity gain must be nonnegative.")
    gain = float(candidate_score) - float(reference_score)
    accepted = math.isfinite(gain) and gain > minimum_gain
    return bool(accepted), gain


def _fit_kmeans(
    name: str,
    features: np.ndarray,
    y: np.ndarray,
    eval_mask: np.ndarray,
    n_clusters: int,
    seed: int,
) -> dict[str, object]:
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=int(seed))
    pred = model.fit_predict(features)
    return {"method": name, "status": "ok", **_metrics(y[eval_mask], pred[eval_mask])}


def _raw_gmm_covariance_type(features: np.ndarray) -> str:
    """Avoid intractable full covariances for very high-dimensional raw baselines."""
    if features.shape[1] > max(512, features.shape[0] // 2):
        return "diag"
    return "full"


def _fit_gmm(
    name: str,
    features: np.ndarray,
    y: np.ndarray,
    eval_mask: np.ndarray,
    n_clusters: int,
    seed: int,
) -> dict[str, object]:
    covariance_type = _raw_gmm_covariance_type(features)
    model = GaussianMixture(
        n_components=n_clusters,
        covariance_type=covariance_type,
        reg_covar=1e-5,
        n_init=3,
        max_iter=300,
        random_state=int(seed),
    )
    pred = model.fit_predict(features)
    return {
        "method": name,
        "status": "ok",
        "gmm_covariance_type": covariance_type,
        "bic": float(model.bic(features)),
        **_metrics(y[eval_mask], pred[eval_mask]),
    }


def _build_model(input_dim: int, latent_dim: int, hidden_dims: Iterable[int], seed: int) -> MLPVAE:
    torch.manual_seed(int(seed))
    return MLPVAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=tuple(int(v) for v in hidden_dims))


class _MLPAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        encoder_layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden_dims:
            encoder_layers.extend([nn.Linear(prev, width), nn.ReLU()])
            prev = width
        encoder_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        prev = latent_dim
        for width in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(prev, width), nn.ReLU()])
            prev = width
        decoder_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decoder(z), z


def _train_autoencoder(
    x: torch.Tensor,
    input_dim: int,
    latent_dim: int,
    hidden_dims: tuple[int, ...],
    args: argparse.Namespace,
    seed: int,
    epochs: int | None = None,
    lr: float | None = None,
) -> tuple[_MLPAutoencoder, np.ndarray, list[dict[str, float]]]:
    torch.manual_seed(int(seed))
    device = torch.device(args.device)
    model = _MLPAutoencoder(input_dim, latent_dim, hidden_dims).to(device)
    x_device = x.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.ae_lr if lr is None else lr))
    history: list[dict[str, float]] = []
    batch_size = min(int(args.batch_size), int(x_device.shape[0]))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 701)

    n_epochs = int(args.ae_epochs if epochs is None else epochs)
    for epoch in range(n_epochs):
        perm = torch.randperm(int(x_device.shape[0]), generator=generator)
        total_loss = 0.0
        n_seen = 0
        for start in range(0, int(x_device.shape[0]), batch_size):
            batch = x_device[perm[start : start + batch_size]]
            recon, _ = model(batch)
            loss = F.mse_loss(recon, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(batch.shape[0])
            n_seen += int(batch.shape[0])
        history.append({"epoch": float(epoch + 1), "recon_loss": total_loss / max(n_seen, 1)})

    with torch.no_grad():
        z = model.encode(x_device).detach().cpu().numpy()
    return model.cpu(), z, history


def _linear_layers(module: nn.Sequential) -> list[nn.Linear]:
    return [layer for layer in module if isinstance(layer, nn.Linear)]


def _copy_linear_parameters(source: nn.Linear, target: nn.Linear) -> None:
    if source.weight.shape != target.weight.shape or source.bias.shape != target.bias.shape:
        raise ValueError("Cannot copy warm-start parameters between incompatible layers.")
    target.weight.data.copy_(source.weight.data)
    target.bias.data.copy_(source.bias.data)


def _warmstart_vae_from_autoencoder(
    model: MLPVAE,
    ae: _MLPAutoencoder,
    logvar: float,
) -> None:
    ae_encoder = _linear_layers(ae.encoder)
    ae_decoder = _linear_layers(ae.decoder)
    vae_encoder = _linear_layers(model.encoder)
    vae_decoder = _linear_layers(model.decoder)

    if len(ae_encoder) != len(vae_encoder) + 1:
        raise ValueError("Autoencoder/VAE encoder structures are incompatible.")
    if len(ae_decoder) != len(vae_decoder) + 1:
        raise ValueError("Autoencoder/VAE decoder structures are incompatible.")

    for source, target in zip(ae_encoder[:-1], vae_encoder):
        _copy_linear_parameters(source, target)
    _copy_linear_parameters(ae_encoder[-1], model.fc_mu)
    model.fc_logvar.weight.data.zero_()
    model.fc_logvar.bias.data.fill_(float(logvar))

    for source, target in zip(ae_decoder[:-1], vae_decoder):
        _copy_linear_parameters(source, target)
    _copy_linear_parameters(ae_decoder[-1], model.fc_out)


def _maybe_warmstart_vae(
    model: MLPVAE,
    x: torch.Tensor,
    input_dim: int,
    latent_dim: int,
    hidden_dims: tuple[int, ...],
    args: argparse.Namespace,
    seed: int,
) -> dict[str, float]:
    warm_epochs = int(getattr(args, "vae_warmstart_ae_epochs", 0))
    if warm_epochs <= 0:
        return {}

    ae, _, history = _train_autoencoder(
        x,
        input_dim,
        latent_dim,
        hidden_dims,
        args,
        seed + 1701,
        epochs=warm_epochs,
        lr=float(getattr(args, "vae_warmstart_lr", args.ae_lr)),
    )
    _warmstart_vae_from_autoencoder(
        model,
        ae,
        logvar=float(getattr(args, "vae_warmstart_logvar", -4.0)),
    )
    return {
        "warmstart_ae_epochs": float(warm_epochs),
        "warmstart_recon_loss": float(history[-1]["recon_loss"]) if history else math.nan,
    }


def _student_t_assign(z: torch.Tensor, centers: torch.Tensor, alpha: float) -> torch.Tensor:
    dist2 = torch.cdist(z, centers).pow(2)
    q = (1.0 + dist2 / float(alpha)).pow(-(float(alpha) + 1.0) / 2.0)
    return q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)


def _target_distribution(q: torch.Tensor) -> torch.Tensor:
    weight = q.pow(2) / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
    return weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-12)


def _fit_ae_kmeans(
    method: str,
    x: torch.Tensor,
    y: np.ndarray,
    eval_mask: np.ndarray,
    input_dim: int,
    latent_dim: int,
    hidden_dims: tuple[int, ...],
    n_clusters: int,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, object]:
    _, z, history = _train_autoencoder(x, input_dim, latent_dim, hidden_dims, args, seed)
    result = _fit_kmeans(method, z, y, eval_mask, n_clusters, seed)
    result["ae_final_recon_loss"] = float(history[-1]["recon_loss"]) if history else math.nan
    return result


def _fit_dec_like(
    method: str,
    x: torch.Tensor,
    y: np.ndarray,
    eval_mask: np.ndarray,
    input_dim: int,
    latent_dim: int,
    hidden_dims: tuple[int, ...],
    n_clusters: int,
    args: argparse.Namespace,
    seed: int,
    use_reconstruction: bool,
) -> dict[str, object]:
    model, z_init, history = _train_autoencoder(x, input_dim, latent_dim, hidden_dims, args, seed)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=int(seed)).fit(z_init)
    device = torch.device(args.device)
    model = model.to(device)
    x_device = x.to(device)
    centers = nn.Parameter(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(list(model.encoder.parameters()) + [centers], lr=float(args.dec_lr))
    if use_reconstruction:
        optimizer = torch.optim.Adam(list(model.parameters()) + [centers], lr=float(args.dec_lr))

    final_kl = math.nan
    final_recon = math.nan
    for _ in range(int(args.dec_epochs)):
        recon, z = model(x_device)
        q = _student_t_assign(z, centers, args.dec_alpha)
        with torch.no_grad():
            p = _target_distribution(q).detach()
        kl_loss = F.kl_div((q + 1e-12).log(), p, reduction="batchmean")
        recon_loss = F.mse_loss(recon, x_device)
        loss = kl_loss
        if use_reconstruction:
            loss = loss + float(args.idec_recon_weight) * recon_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_kl = float(kl_loss.item())
        final_recon = float(recon_loss.item())

    with torch.no_grad():
        _, z = model(x_device)
        q = _student_t_assign(z, centers, args.dec_alpha).detach().cpu().numpy()
    pred = q.argmax(axis=1)
    return {
        "method": method,
        "status": "ok",
        "ae_final_recon_loss": float(history[-1]["recon_loss"]) if history else math.nan,
        "dec_final_kl": final_kl,
        "dec_final_recon_loss": final_recon,
        **_metrics(y[eval_mask], pred[eval_mask]),
    }


def _contrastive_augment(
    x: torch.Tensor,
    image_shape: tuple[int, int, int],
    generator: torch.Generator,
    noise_scale: float,
    mask_probability: float,
) -> torch.Tensor:
    images = x.reshape(x.shape[0], *image_shape).clone()
    max_shift = 1 if min(image_shape[1:]) < 16 else 2
    shifts = torch.randint(
        -max_shift,
        max_shift + 1,
        (images.shape[0], 2),
        generator=generator,
    )
    shifted = torch.zeros_like(images)
    for index, (row_shift, col_shift) in enumerate(shifts.tolist()):
        rolled = torch.roll(images[index], shifts=(row_shift, col_shift), dims=(1, 2))
        if row_shift > 0:
            rolled[:, :row_shift, :] = 0.0
        elif row_shift < 0:
            rolled[:, row_shift:, :] = 0.0
        if col_shift > 0:
            rolled[:, :, :col_shift] = 0.0
        elif col_shift < 0:
            rolled[:, :, col_shift:] = 0.0
        shifted[index] = rolled
    if mask_probability > 0.0:
        keep = torch.rand(shifted.shape, generator=generator) >= float(mask_probability)
        shifted = shifted * keep.to(shifted.dtype)
    if noise_scale > 0.0:
        shifted = shifted + float(noise_scale) * torch.randn(
            shifted.shape, generator=generator, dtype=shifted.dtype
        )
    return shifted.reshape(x.shape[0], -1)


def _paired_nt_xent(first: torch.Tensor, second: torch.Tensor, temperature: float) -> torch.Tensor:
    if first.shape != second.shape:
        raise ValueError("Contrastive views must have matching shapes.")
    n_items = int(first.shape[0])
    if n_items < 2:
        raise ValueError("Contrastive batches require at least two items.")
    representations = F.normalize(torch.cat([first, second], dim=0), dim=1)
    logits = representations @ representations.T / float(temperature)
    logits.fill_diagonal_(torch.finfo(logits.dtype).min)
    targets = torch.arange(2 * n_items, device=logits.device)
    targets = (targets + n_items) % (2 * n_items)
    return F.cross_entropy(logits, targets)


def _fit_contrastive_clustering(
    method: str,
    x: torch.Tensor,
    y: np.ndarray,
    eval_mask: np.ndarray,
    latent_dim: int,
    n_clusters: int,
    args: argparse.Namespace,
    seed: int,
    image_shape: tuple[int, int, int] | None,
    artifact_dir: Path | None = None,
) -> dict[str, object]:
    if image_shape is None:
        raise ValueError("Contrastive clustering requires image-shaped input.")
    torch.manual_seed(int(seed))
    device = torch.device(args.device)
    model = ConvContrastiveClusterer(
        image_shape=image_shape,
        latent_dim=latent_dim,
        n_clusters=n_clusters,
        channels=tuple(args.conv_channels),
        projection_dim=args.cc_projection_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.cc_lr))
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 8101)
    batch_size = min(int(args.batch_size), int(x.shape[0]))
    history: list[dict[str, float]] = []

    for epoch in range(int(args.cc_epochs)):
        permutation = torch.randperm(int(x.shape[0]), generator=generator)
        totals = {"loss": 0.0, "instance_loss": 0.0, "cluster_loss": 0.0, "entropy_loss": 0.0}
        n_seen = 0
        model.train()
        for start in range(0, int(x.shape[0]), batch_size):
            batch = x[permutation[start : start + batch_size]]
            if batch.shape[0] < 2:
                continue
            first = _contrastive_augment(
                batch,
                image_shape,
                generator,
                args.cc_noise_scale,
                args.cc_mask_probability,
            ).to(device)
            second = _contrastive_augment(
                batch,
                image_shape,
                generator,
                args.cc_noise_scale,
                args.cc_mask_probability,
            ).to(device)
            _, instance_first, cluster_first = model(first)
            _, instance_second, cluster_second = model(second)
            instance_loss = _paired_nt_xent(
                instance_first, instance_second, args.cc_temperature
            )
            cluster_loss = _paired_nt_xent(
                cluster_first.T, cluster_second.T, args.cc_cluster_temperature
            )
            mean_first = cluster_first.mean(dim=0).clamp_min(1e-12)
            mean_second = cluster_second.mean(dim=0).clamp_min(1e-12)
            entropy_loss = (
                math.log(n_clusters)
                + 0.5
                * (
                    torch.sum(mean_first * mean_first.log())
                    + torch.sum(mean_second * mean_second.log())
                )
            )
            loss = (
                instance_loss
                + float(args.cc_cluster_weight) * cluster_loss
                + float(args.cc_entropy_weight) * entropy_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = int(batch.shape[0])
            totals["loss"] += float(loss.item()) * count
            totals["instance_loss"] += float(instance_loss.item()) * count
            totals["cluster_loss"] += float(cluster_loss.item()) * count
            totals["entropy_loss"] += float(entropy_loss.item()) * count
            n_seen += count
        history.append(
            {"epoch": float(epoch + 1), **{key: value / max(n_seen, 1) for key, value in totals.items()}}
        )

    model.eval()
    with torch.no_grad():
        features, _, probabilities = model(x.to(device))
    pred = probabilities.argmax(dim=1).cpu().numpy()
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifact_dir / "embeddings.npz",
            mu=features.cpu().numpy(),
            responsibilities=probabilities.cpu().numpy(),
            labels=y,
            eval_mask=eval_mask,
        )
        torch.save(
            {
                "schema_version": 1,
                "method": method,
                "seed": int(seed),
                "model_state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "model_config": {
                    "image_shape": list(image_shape),
                    "latent_dim": int(latent_dim),
                    "n_clusters": int(n_clusters),
                    "conv_channels": list(args.conv_channels),
                },
            },
            artifact_dir / "checkpoint.pt",
        )
        write_json(artifact_dir / "training_history.json", history)
    tail = history[-1] if history else {}
    return {
        "method": method,
        "status": "ok",
        "final_loss": float(tail.get("loss", math.nan)),
        "final_instance_loss": float(tail.get("instance_loss", math.nan)),
        "final_cluster_loss": float(tail.get("cluster_loss", math.nan)),
        "active_components": int((probabilities.mean(dim=0) > 0.01).sum().item()),
        **_metrics(y[eval_mask], pred[eval_mask]),
    }


def _standard_vae_embedding(
    x: torch.Tensor,
    input_dim: int,
    latent_dim: int,
    hidden_dims: tuple[int, ...],
    args: argparse.Namespace,
    seed: int,
) -> tuple[MLPVAE, np.ndarray, list[dict[str, float]]]:
    model = _build_model(input_dim, latent_dim, hidden_dims, seed)
    warmstart_metrics = _maybe_warmstart_vae(model, x, input_dim, latent_dim, hidden_dims, args, seed)
    history = train_vae(
        model,
        x,
        StandardNormalPrior(latent_dim),
        TrainConfig(
            epochs=args.pretrain_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            beta=args.beta,
            recon_variance=args.recon_variance,
            device=args.device,
            seed=seed + 11,
        ),
    )
    if history and warmstart_metrics:
        history[-1].update(warmstart_metrics)
    encoded = encode_dataset(model, x, batch_size=args.batch_size, device=args.device)
    return model, encoded["mu"].numpy(), history


def _alternating_config(args: argparse.Namespace, seed: int) -> AlternatingTrainConfig:
    return AlternatingTrainConfig(
        pretrain=TrainConfig(
            epochs=args.pretrain_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            beta=args.beta,
            recon_variance=args.recon_variance,
            device=args.device,
            seed=seed + 101,
        ),
        finetune=TrainConfig(
            epochs=args.finetune_epochs,
            batch_size=args.batch_size,
            lr=args.finetune_lr,
            beta=args.beta,
            recon_variance=args.recon_variance,
            device=args.device,
            seed=seed + 103,
        ),
        cycles=args.cycles,
    )


def _fmvmm_fitter(
    method: str,
    n_clusters: int,
    args: argparse.Namespace,
    seed: int,
    *,
    stabilize_fixed_skew_t: bool = False,
    stabilize_fixed_skew_normal: bool = False,
) -> FMVMMPriorFitter:
    method = _base_fmvmm_method(method)
    try:
        families = FMVMM_METHOD_FAMILIES[method]
    except KeyError:
        raise ValueError(f"Unknown FMVMM method {method!r}")

    candidate_combinations = [tuple([family] * n_clusters) for family in families]
    return FMVMMPriorFitter(
        n_components=n_clusters,
        fmvmm_path=args.fmvmm_path,
        list_of_dist=families,
        candidate_combinations=candidate_combinations,
        max_iter=args.fmvmm_iter,
        tol=args.fmvmm_tol,
        em_type=(
            "hard"
            if stabilize_fixed_skew_normal and families == ("mvsn",)
            else "soft"
        ),
        initialization=args.fmvmm_initialization,
        n_jobs=args.fmvmm_jobs,
        gh_mstep_kwargs={
            "nit": args.gh_mstep_nit,
            "reltol": 1e-4,
            "abstol": 1e-5,
            "silent": True,
        },
        init_fit_kwargs=(
            {"mgst": {"nit": 1, "silent": True}}
            if stabilize_fixed_skew_t and families == ("mgst",)
            else (
                {"mvsn": {"maxiter": 20}}
                if stabilize_fixed_skew_normal and families == ("mvsn",)
                else {}
            )
        ),
        verbose=False,
        debug=args.fmvmm_debug,
        differentiable=True,
        require_differentiable=True,
        standardize=args.fmvmm_standardize,
        random_seed=seed + 7001,
        fit_timeout_seconds=args.fmvmm_fit_timeout_seconds,
    )


def _adaptive_prior_candidates(
    n_clusters: int,
    args: argparse.Namespace,
    seed: int,
) -> list[PriorCandidate]:
    candidates: list[PriorCandidate] = []
    for family in args.adaptive_families:
        name, rank = ADAPTIVE_FAMILY_ORDER[family]
        candidates.append(
            PriorCandidate(
                name=name,
                fitter=_fmvmm_fitter(f"fmvmm_{family}", n_clusters, args, seed),
                complexity_rank=rank,
            )
        )
    return candidates


def _conservative_prior_candidates(
    n_clusters: int,
    args: argparse.Namespace,
    seed: int,
) -> list[PriorCandidate]:
    """Nested prior bank for the deployable conservative selector.

    The candidate order encodes the modelling claim: stay Gaussian unless a
    heavier-tailed or full GH-family prior earns enough held-out latent score.
    """

    return [
        PriorCandidate(
            name="gaussian",
            fitter=DiagonalGaussianMixtureFitter(
                n_components=n_clusters,
                n_iter=args.diag_gmm_iter,
                seed=seed + 6111,
            ),
            complexity_rank=0,
        ),
        PriorCandidate(
            name="student_t",
            fitter=_fmvmm_fitter("fmvmm_mvt", n_clusters, args, seed + 1),
            complexity_rank=1,
        ),
        PriorCandidate(
            name="gh_family",
            fitter=_fmvmm_fitter("fmvmm_gh_family", n_clusters, args, seed + 2),
            complexity_rank=2,
        ),
    ]


def _tail_prior_candidates(
    n_clusters: int,
    args: argparse.Namespace,
    seed: int,
) -> list[PriorCandidate]:
    """Stable non-Gaussian selector: Student-t versus the GH catalogue."""

    return [
        PriorCandidate(
            name="student_t",
            fitter=_fmvmm_fitter("fmvmm_mvt", n_clusters, args, seed + 1),
            complexity_rank=0,
        ),
        PriorCandidate(
            name="gh_family",
            fitter=_fmvmm_fitter("fmvmm_gh_family", n_clusters, args, seed + 2),
            complexity_rank=1,
        ),
    ]


def _fit_diag_prior_on_features(
    method: str,
    features: np.ndarray,
    y: np.ndarray,
    eval_mask: np.ndarray,
    n_clusters: int,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, object]:
    fitter = DiagonalGaussianMixtureFitter(
        n_components=n_clusters,
        n_iter=args.diag_gmm_iter,
        seed=seed + 131,
    )
    fit_result = fitter.fit(torch.from_numpy(features).float())
    resp = fit_result.prior.responsibilities(torch.from_numpy(features).float()).detach().cpu().numpy()
    pred = resp.argmax(axis=1)
    return {
        "method": method,
        "status": "ok",
        "prior_name": fit_result.prior.name,
        "fit_diagnostics": fit_result.diagnostics,
        **_metrics(y[eval_mask], pred[eval_mask]),
    }


def _fit_fmvmm_on_features(
    method: str,
    features: np.ndarray,
    y: np.ndarray,
    eval_mask: np.ndarray,
    n_clusters: int,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, object]:
    fitter = _fmvmm_fitter(method, n_clusters, args, seed)
    fit_result = fitter.fit(torch.from_numpy(features).float())
    resp = fit_result.prior.responsibilities(torch.from_numpy(features).float()).detach().cpu().numpy()
    pred = resp.argmax(axis=1)
    out = {
        "method": method,
        "status": "ok",
        "prior_name": fit_result.prior.name,
        "fit_diagnostics": fit_result.diagnostics,
        **_metrics(y[eval_mask], pred[eval_mask]),
    }
    if "best_bic" in fit_result.diagnostics:
        out["bic"] = float(fit_result.diagnostics["best_bic"])
    return out


def _fit_prior_vae(
    method: str,
    x: torch.Tensor,
    y: np.ndarray,
    eval_mask: np.ndarray,
    input_dim: int,
    latent_dim: int,
    hidden_dims: tuple[int, ...],
    n_clusters: int,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, object]:
    if method == "diag_gmm_prior":
        fitter = DiagonalGaussianMixtureFitter(
            n_components=n_clusters,
            n_iter=args.diag_gmm_iter,
            seed=seed + 31,
        )
    else:
        fitter = _fmvmm_fitter(method, n_clusters, args, seed)

    model = _build_model(input_dim, latent_dim, hidden_dims, seed)
    warmstart_metrics = _maybe_warmstart_vae(model, x, input_dim, latent_dim, hidden_dims, args, seed)
    prior, history = alternating_train(model, x, fitter, _alternating_config(args, seed))
    encoded = encode_dataset(model, x, batch_size=args.batch_size, device=args.device)
    resp = prior.responsibilities(encoded["mu"]).detach().cpu().numpy()
    pred = resp.argmax(axis=1)
    fit_diagnostics = history["cycles"][-1]["fit_diagnostics"] if history["cycles"] else {}
    train_tail = history["cycles"][-1]["train"][-1] if history["cycles"] else {}
    out = {
        "method": method,
        "status": "ok",
        "prior_name": prior.name,
        "final_loss": float(train_tail.get("loss", math.nan)),
        "final_elbo": float(train_tail.get("elbo", math.nan)),
        "fit_diagnostics": fit_diagnostics,
        **_metrics(y[eval_mask], pred[eval_mask]),
    }
    out.update(warmstart_metrics)
    if "best_bic" in fit_diagnostics:
        out["bic"] = float(fit_diagnostics["best_bic"])
    return out


def _guard_min_count(n_clusters: int, fraction: float, minimum: int) -> int:
    return max(int(minimum), int(math.ceil(float(fraction) * int(n_clusters))))


def _guard_fallback_method(use_conv: bool, use_mi: bool, args: argparse.Namespace) -> str:
    family = str(args.adaptive_guard_fallback_family)
    suffix = "_mi" if use_mi and bool(args.adaptive_guard_preserve_mi) else ""
    prefix = "conv_" if use_conv else ""
    return f"{prefix}ghvade_{family}{suffix}"


@torch.no_grad()
def _prior_active_components(prior: object, z_bank: torch.Tensor, device: str) -> int:
    responsibilities = getattr(prior, "responsibilities", None)
    if not callable(responsibilities):
        return -1
    flat = z_bank.reshape(-1, z_bank.shape[-1]).to(torch.device(device))
    prior.to(device)  # type: ignore[attr-defined]
    resp = responsibilities(flat).detach().cpu()
    return int((resp.mean(dim=0) > 0.01).sum())


def _guard_trigger(
    result: dict[str, object],
    n_clusters: int,
    args: argparse.Namespace,
) -> tuple[bool, str, dict[str, object]]:
    min_active = _guard_min_count(
        n_clusters,
        args.adaptive_guard_min_active_fraction,
        args.adaptive_guard_min_active,
    )
    min_assigned = _guard_min_count(
        n_clusters,
        args.adaptive_guard_min_assigned_fraction,
        args.adaptive_guard_min_assigned,
    )
    active_components = int(float(result.get("active_components", 0) or 0))
    n_assigned = int(float(result.get("n_assigned", 0) or 0))
    reasons: list[str] = []
    if active_components < min_active:
        reasons.append(f"active_components={active_components}<min_active={min_active}")
    if n_assigned < min_assigned:
        reasons.append(f"n_assigned={n_assigned}<min_assigned={min_assigned}")
    diagnostics = {
        "guard_min_active_components": int(min_active),
        "guard_min_assigned_clusters": int(min_assigned),
        "guard_observed_active_components": int(active_components),
        "guard_observed_assigned_clusters": int(n_assigned),
    }
    return bool(reasons), ";".join(reasons), diagnostics


def _guarded_fallback_result(
    *,
    initial: dict[str, object],
    fallback: dict[str, object],
    method: str,
    fallback_method: str,
    reason: str,
    guard_diagnostics: dict[str, object],
) -> dict[str, object]:
    out = dict(fallback)
    out["method"] = method
    out["guard_triggered"] = True
    out["guard_reason"] = reason
    out["guard_fallback_method"] = fallback_method
    out["guard_initial_selected_family"] = initial.get("selected_family")
    out["guard_initial_prior_name"] = initial.get("prior_name")
    for key in ("acc", "ari", "nmi", "purity", "n_assigned", "active_components"):
        if key in initial:
            out[f"guard_initial_{key}"] = initial[key]
    out.update(guard_diagnostics)
    fit_diagnostics = dict(out.get("fit_diagnostics") or {})
    fit_diagnostics["guarded_adaptive"] = {
        "triggered": True,
        "reason": reason,
        "fallback_method": fallback_method,
        "initial_selected_family": initial.get("selected_family"),
        **guard_diagnostics,
    }
    out["fit_diagnostics"] = fit_diagnostics
    return out


def _componentwise_fallback_result(
    *,
    initial: dict[str, object],
    fallback: dict[str, object],
    method: str,
    fallback_method: str,
    rejection_reasons: list[str],
) -> dict[str, object]:
    """Return the matched Student-t run after an unsafe GH continuation."""

    out = dict(fallback)
    out["method"] = method
    out["componentwise_fallback_triggered"] = True
    out["componentwise_fallback_method"] = fallback_method
    out["componentwise_fallback_reasons"] = list(rejection_reasons)
    out["componentwise_attempt_selected_family"] = initial.get("selected_family")
    out["componentwise_attempt_prior_name"] = initial.get("prior_name")
    out["componentwise_attempt_fit_diagnostics"] = initial.get("fit_diagnostics")
    out["componentwise_attempt_prior_update_history"] = initial.get(
        "prior_update_history"
    )
    for key in (
        "acc",
        "ari",
        "nmi",
        "purity",
        "n_assigned",
        "active_components",
        "graph_soft_assortativity",
        "graph_hard_assortativity",
    ):
        if key in initial:
            out[f"componentwise_attempt_{key}"] = initial[key]
    fit_diagnostics = dict(out.get("fit_diagnostics") or {})
    fit_diagnostics["componentwise_transactional_fallback"] = {
        "triggered": True,
        "reasons": list(rejection_reasons),
        "fallback_method": fallback_method,
        "attempt_selected_family": initial.get("selected_family"),
    }
    out["fit_diagnostics"] = fit_diagnostics
    return out


def _prior_parameter_summary(prior: nn.Module) -> dict[str, object]:
    base_prior = getattr(prior, "base_prior", prior)
    components = getattr(base_prior, "components", None)
    log_weights = getattr(base_prior, "log_weights", None)
    summary: dict[str, object] = {
        "prior_class": type(prior).__name__,
        "base_prior_class": type(base_prior).__name__,
    }
    if isinstance(log_weights, torch.Tensor):
        summary["weights"] = log_weights.detach().float().softmax(dim=0).cpu().tolist()
    if components is None:
        return summary
    component_records: list[dict[str, object]] = []
    for index, component in enumerate(components):
        record: dict[str, object] = {
            "component": int(index),
            "family": str(getattr(component, "family", type(component).__name__)),
        }
        for attribute in ("df", "lmbda", "chi", "psi"):
            value = getattr(component, attribute, None)
            if value is not None:
                record[attribute] = float(value)
        for attribute in ("a", "gamma"):
            value = getattr(component, attribute, None)
            if isinstance(value, torch.Tensor):
                record[f"{attribute}_norm"] = float(
                    torch.linalg.vector_norm(value.detach().float()).cpu()
                )
        skewness_norm = getattr(component, "skewness_norm", None)
        if isinstance(skewness_norm, torch.Tensor):
            record["skewness_norm"] = float(skewness_norm.detach().float().cpu())
        component_records.append(record)
    summary["components"] = component_records
    return summary


def _safe_assignment_floors(
    anchor: dict[str, object],
    minimum_active_components: int,
    tolerance: float,
) -> dict[str, float | int]:
    return {
        "minimum_active_components": int(minimum_active_components),
        "minimum_normalized_marginal_entropy": float(
            anchor["normalized_marginal_entropy"]
        )
        - float(tolerance),
        "minimum_normalized_assignment_mi": float(
            anchor["normalized_assignment_mi"]
        )
        - float(tolerance),
        "minimum_draw_assignment_stability": float(
            anchor["draw_assignment_stability"]
        )
        - float(tolerance),
    }


def _safe_assignment_violations(
    assignment: dict[str, float | int],
    floors: dict[str, float | int],
) -> list[str]:
    violations: list[str] = []
    if int(assignment["active_components"]) < int(
        floors["minimum_active_components"]
    ):
        violations.append("component_collapse")
    comparisons = (
        ("normalized_marginal_entropy", "minimum_normalized_marginal_entropy"),
        ("normalized_assignment_mi", "minimum_normalized_assignment_mi"),
        ("draw_assignment_stability", "minimum_draw_assignment_stability"),
    )
    for metric, floor_name in comparisons:
        if float(assignment[metric]) < float(floors[floor_name]):
            violations.append(f"degraded_{metric}")
    return violations


def _component_enrichment_config(
    args: argparse.Namespace,
) -> ComponentwiseEnrichmentConfig:
    return ComponentwiseEnrichmentConfig(
        families=tuple(args.component_enrichment_families),
        familywise_alpha=args.safe_familywise_alpha,
        confidence_z=args.safe_confidence_z,
        boundary_confidence_z=args.component_boundary_confidence_z,
        score_margin=args.shape_score_margin,
        component_skew_threshold=args.shape_skew_threshold,
        component_skew_signal_threshold=args.component_skew_signal_threshold,
        minimum_effective_component_size=args.component_minimum_effective_size,
        minimum_anchor_normalized_entropy=(
            args.component_minimum_anchor_normalized_entropy
        ),
        minimum_active_fraction=args.safe_min_active_fraction,
        assignment_tolerance=args.safe_assignment_tolerance,
        component_mass_threshold=args.safe_component_mass_threshold,
        core_confidence_threshold=args.safe_core_confidence_threshold,
        minimum_core_fraction=args.safe_minimum_core_fraction,
        minimum_core_retention=args.safe_minimum_core_retention,
        boundary_log_odds_threshold=args.component_boundary_log_odds_threshold,
        minimum_boundary_effective_size=(
            args.component_minimum_boundary_effective_size
        ),
        minimum_boundary_score_gain=args.component_minimum_boundary_score_gain,
        minimum_boundary_leverage=args.component_minimum_boundary_leverage,
        minimum_decision_change_fraction=(
            args.component_minimum_decision_change_fraction
        ),
        graph_refinement=args.component_fit_graph_refinement,
        graph_neighbors=args.component_graph_neighbors,
        graph_density_one_se_multiplier=(
            args.component_graph_density_one_se_multiplier
        ),
        minimum_optimization_graph_gain=(
            args.component_minimum_optimization_graph_gain
        ),
        minimum_weight=args.safe_minimum_weight,
        minimum_covariance_eigenvalue=args.safe_minimum_covariance_eigenvalue,
        maximum_covariance_condition=args.safe_maximum_covariance_condition,
        minimum_interior_shape=args.safe_minimum_interior_shape,
        max_fit_iterations=args.gh_mstep_nit,
        fit_timeout_seconds=args.fmvmm_fit_timeout_seconds,
    )


def _fit_mixture_aware_vae(
    method: str,
    x: torch.Tensor,
    y: np.ndarray,
    eval_mask: np.ndarray,
    input_dim: int,
    latent_dim: int,
    hidden_dims: tuple[int, ...],
    n_clusters: int,
    args: argparse.Namespace,
    seed: int,
    artifact_dir: Path | None = None,
    image_shape: tuple[int, int, int] | None = None,
) -> dict[str, object]:
    method_start = time.perf_counter()
    use_mi = method.endswith("_mi")
    base_method = method.removesuffix("_mi")
    use_conv = base_method.startswith("conv_")
    base_method = base_method.removeprefix("conv_")
    guarded_adaptive = base_method == "ghvade_guarded_adaptive"
    safe_adaptive = base_method == "ghvade_safe_adaptive"
    trust_component_adaptive = (
        base_method == "ghvade_trust_component_adaptive"
    )
    trust_student_t = base_method == "ghvade_trust_mvt"
    component_adaptive = base_method in {
        "ghvade_component_adaptive",
        "ghvade_trust_component_adaptive",
    }
    posterior_anchor_weight = float(args.mixture_posterior_anchor_weight)
    if (trust_component_adaptive or trust_student_t) and posterior_anchor_weight == 0.0:
        posterior_anchor_weight = 10.0
    posterior_anchor_max_validation_kl = (
        float(args.mixture_posterior_anchor_max_validation_kl)
        if (
            trust_component_adaptive or trust_student_t
        )
        and float(args.mixture_posterior_anchor_max_validation_kl) > 0.0
        else None
    )
    posthoc_component_adaptive = (
        base_method == "ghvade_posthoc_component_adaptive"
    )
    posthoc_student_t = base_method == "ghvade_posthoc_mvt"
    posthoc_protocol = posthoc_component_adaptive or posthoc_student_t
    staged_component_adaptive = (
        base_method == "ghvade_staged_component_adaptive"
    )
    staged_student_t = base_method == "ghvade_staged_mvt"
    staged_protocol = staged_component_adaptive or staged_student_t
    three_way_protocol = (
        posthoc_protocol
        or staged_protocol
        or trust_component_adaptive
        or trust_student_t
    )
    safe_protocol = safe_adaptive or component_adaptive
    fit_base_method = (
        "ghvade_mvt"
        if posthoc_protocol or staged_protocol or trust_student_t
        else (
            "ghvade_component_adaptive"
            if trust_component_adaptive
            else ("ghvade_adaptive" if guarded_adaptive else base_method)
        )
    )
    if use_conv:
        if image_shape is None:
            image_side = int(round(math.sqrt(input_dim)))
            if image_side * image_side != input_dim:
                raise ValueError("Convolutional mixture methods require image_shape metadata.")
            image_shape = (1, image_side, image_side)
        if math.prod(image_shape) != input_dim:
            raise ValueError(f"image_shape {image_shape} is incompatible with input_dim={input_dim}.")
        torch.manual_seed(int(seed))
        model = ConvVAE(
            image_shape=image_shape,
            latent_dim=latent_dim,
            channels=tuple(args.conv_channels),
        )
    else:
        model = _build_model(input_dim, latent_dim, hidden_dims, seed)

    development_x, selection_x, development_indices, selection_indices = unlabeled_optimization_selection_split(
        x,
        selection_fraction=args.mixture_selection_fraction,
        seed=seed + 2099,
    )
    neural_validation_indices: torch.Tensor | None = None
    if three_way_protocol:
        (
            training_x,
            neural_validation_x,
            neural_training_relative_indices,
            neural_validation_relative_indices,
        ) = unlabeled_optimization_selection_split(
            development_x,
            selection_fraction=args.mixture_neural_validation_fraction,
            seed=seed + 2100,
        )
        training_indices = development_indices[neural_training_relative_indices]
        neural_validation_indices = development_indices[
            neural_validation_relative_indices
        ]
        optimization_indices = development_indices
    else:
        training_x = development_x
        neural_validation_x = selection_x
        training_indices = development_indices
        optimization_indices = training_indices
    split_diagnostics = {
        "selection_fraction": float(args.mixture_selection_fraction),
        "n_optimization": int(development_x.shape[0]),
        "n_selection": int(selection_x.shape[0]),
        "optimization_indices_sha256": hashlib.sha256(optimization_indices.numpy().tobytes()).hexdigest(),
        "selection_indices_sha256": hashlib.sha256(selection_indices.numpy().tobytes()).hexdigest(),
        "split_seed": int(seed + 2099),
    }
    if three_way_protocol:
        split_diagnostics.update(
            {
                "independent_family_selection_split": True,
                "neural_validation_fraction_of_optimization": float(
                    args.mixture_neural_validation_fraction
                ),
                "n_neural_training": int(training_x.shape[0]),
                "n_neural_validation": int(neural_validation_x.shape[0]),
                "neural_training_indices_sha256": hashlib.sha256(
                    training_indices.numpy().tobytes()
                ).hexdigest(),
                "neural_validation_indices_sha256": hashlib.sha256(
                    neural_validation_indices.numpy().tobytes()
                ).hexdigest(),
                "neural_split_seed": int(seed + 2100),
            }
        )
    pretrain_start = time.perf_counter()
    pretrain_history = pretrain_deterministic_latent(
        model,
        training_x,
        DeterministicPretrainConfig(
            epochs=args.mixture_pretrain_epochs,
            batch_size=args.batch_size,
            lr=args.mixture_pretrain_lr,
            device=args.device,
            seed=seed + 2101,
            initial_logvar=args.mixture_initial_logvar,
        ),
    )
    if args.mixture_affine_gauge:
        model = AffineGaugeVAE(model)
    encoded = encode_dataset(model, training_x, batch_size=args.batch_size, device=args.device)
    training_latent_mean_anchor = encoded["mu"].detach().clone()
    training_posterior_logvar_anchor = encoded["logvar"].detach().clone()
    neural_validation_anchor_encoding = encode_dataset(
        model,
        neural_validation_x,
        batch_size=args.batch_size,
        device=args.device,
    )
    neural_validation_latent_mean_anchor = neural_validation_anchor_encoding[
        "mu"
    ].detach().clone()
    neural_validation_posterior_logvar_anchor = neural_validation_anchor_encoding[
        "logvar"
    ].detach().clone()
    selection_graph_features = encode_dataset(
        model,
        selection_x,
        batch_size=args.batch_size,
        device=args.device,
    )["mu"].numpy()
    pretrain_all_encoding = (
        encode_dataset(
            model,
            x,
            batch_size=args.batch_size,
            device=args.device,
        )
        if artifact_dir is not None
        else None
    )
    pretrain_seconds = time.perf_counter() - pretrain_start
    mixture_stage_start = time.perf_counter()
    neural_training_seconds = 0.0

    fit_diagnostics: dict[str, object]
    prior_update_history: list[dict[str, object]] = []
    training_state: dict[str, object] = {}
    selected_family = "gaussian"
    locked_fitter: FMVMMPriorFitter | None = None
    safe_assignment_thresholds: dict[str, float | int] | None = None
    safe_training_assignment_anchor: torch.Tensor | None = None
    staged_anchor_prior = None
    componentwise_fallback_required = False
    componentwise_fallback_reasons: list[str] = []
    if staged_protocol and int(args.mixture_cycles) < 2:
        raise ValueError("Staged component selection requires at least two mixture cycles.")
    if fit_base_method == "vade_diag":
        fitted = DiagonalGaussianMixtureFitter(
            n_components=n_clusters,
            n_iter=args.diag_gmm_iter,
            seed=seed + 2111,
        ).fit(encoded["mu"])
        prior = LearnableDiagonalGaussianMixturePrior.from_fitted(fitted.prior)
        fit_diagnostics = fitted.diagnostics
        neural_start = time.perf_counter()
        history = train_mixture_vae(
            model,
            training_x,
            prior,
            MixtureTrainConfig(
                epochs=args.mixture_epochs,
                batch_size=args.batch_size,
                lr=args.mixture_lr,
                prior_lr=args.mixture_prior_lr,
                beta=args.beta,
                recon_variance=args.recon_variance,
                n_mc_samples=args.mixture_mc_samples,
                responsibility_temperature=args.mixture_temperature,
                cluster_mi_weight=args.mixture_mi_weight if use_mi else 0.0,
                optimize_prior=True,
                grad_clip_norm=args.mixture_grad_clip,
                early_stopping_patience=args.mixture_patience,
                device=args.device,
                seed=seed + 2121,
                objective="factorized" if use_mi else "collapsed",
                latent_mean_anchor_weight=args.mixture_latent_anchor_weight,
                posterior_anchor_weight=posterior_anchor_weight,
                posterior_anchor_max_validation_kl=(
                    posterior_anchor_max_validation_kl
                ),
            ),
            validation_x=neural_validation_x,
            validation_mc_samples=args.mixture_validation_mc_samples,
            validation_seed=seed + 2127,
            state_sink=training_state,
            latent_mean_anchor=training_latent_mean_anchor,
            posterior_anchor_logvar=training_posterior_logvar_anchor,
            validation_latent_mean_anchor=neural_validation_latent_mean_anchor,
            validation_posterior_anchor_logvar=(
                neural_validation_posterior_logvar_anchor
            ),
        )
        neural_training_seconds += time.perf_counter() - neural_start
    elif fit_base_method.startswith("ghvade_"):
        history = []
        z_bank = posterior_sample_bank(
            model,
            training_x,
            n_mc_samples=args.mixture_refit_mc_samples,
            batch_size=args.batch_size,
            device=args.device,
            seed=seed + 2129,
        )
        if safe_protocol:
            anchor_fitter = _fmvmm_fitter("fmvmm_mvt", n_clusters, args, seed)
            locked_fitter = anchor_fitter
            prior, fit_diagnostics = accepted_prior_refit(
                None,
                anchor_fitter,
                z_bank,
                tolerance=args.prior_acceptance_tolerance,
                device=args.device,
            )
            anchor_regularity = prior_regularity_diagnostics(
                prior,
                minimum_weight=args.safe_minimum_weight,
                minimum_covariance_eigenvalue=(
                    args.safe_minimum_covariance_eigenvalue
                ),
                maximum_covariance_condition=args.safe_maximum_covariance_condition,
                minimum_interior_shape=args.safe_minimum_interior_shape,
            )
            fit_diagnostics["regularity"] = anchor_regularity
            if not anchor_regularity["regular"]:
                raise RuntimeError(
                    "The Student-t anchor lies outside the declared regular parameter "
                    f"region: {anchor_regularity['violations']}"
                )
            selected_family = "student_t"
            fit_diagnostics["safe_anchor_initialization"] = True
            fit_diagnostics["safe_family_locked"] = False
            if component_adaptive:
                anchor_fit_diagnostics = fit_diagnostics
                component_anchor_prior = prior
                selection_bank = posterior_sample_bank(
                    model,
                    selection_x,
                    n_mc_samples=args.mixture_refit_mc_samples,
                    batch_size=args.batch_size,
                    device=args.device,
                    seed=seed + 2130,
                )
                enrichment = fit_componentwise_gh_enrichment(
                    prior,
                    z_bank,
                    selection_bank,
                    config=_component_enrichment_config(args),
                    fmvmm_path=args.fmvmm_path,
                    device=args.device,
                )
                prior, fit_diagnostics = accept_prior_candidate(
                    prior,
                    enrichment.prior,
                    selection_bank,
                    candidate_diagnostics=enrichment.diagnostics,
                    tolerance=args.prior_acceptance_tolerance,
                    device=args.device,
                )
                if fit_diagnostics["prior_update_accepted"] and enrichment.diagnostics[
                    "promoted"
                ]:
                    selected_family = f"student_t+{enrichment.selected_family}"
                    safe_training_assignment_anchor = (
                        encode_mixture_responsibilities(
                            model,
                            training_x,
                            component_anchor_prior,
                            batch_size=args.batch_size,
                            device=args.device,
                            n_mc_samples=args.mixture_eval_mc_samples,
                            seed=seed + 2131,
                        )
                    )
                safe_assignment_thresholds = _safe_assignment_floors(
                    enrichment.diagnostics["anchor_assignment"],
                    int(enrichment.diagnostics["minimum_active_components"]),
                    args.safe_assignment_tolerance,
                )
                fit_diagnostics["safe_anchor_fit"] = anchor_fit_diagnostics
                fit_diagnostics["safe_anchor_initialization"] = True
                fit_diagnostics["safe_family_locked"] = True
                fit_diagnostics["safe_locked_family"] = selected_family
                fit_diagnostics["safe_enriched_component"] = (
                    enrichment.selected_component
                )
        elif fit_base_method == "ghvade_adaptive":
            selection_bank = posterior_sample_bank(
                model,
                selection_x,
                n_mc_samples=args.mixture_refit_mc_samples,
                batch_size=args.batch_size,
                device=args.device,
                seed=seed + 2130,
            )
            selection = fit_and_select_prior(
                _adaptive_prior_candidates(n_clusters, args, seed),
                z_bank,
                selection_bank,
                device=args.device,
                selection_rule=args.prior_selection_rule,
                shape_skew_threshold=args.shape_skew_threshold,
                shape_score_margin=args.shape_score_margin,
            )
            prior, fit_diagnostics = accept_prior_candidate(
                None,
                selection.prior,
                z_bank,
                candidate_diagnostics=selection.diagnostics,
                tolerance=args.prior_acceptance_tolerance,
                device=args.device,
            )
            selected_family = selection.candidate_name
            if guarded_adaptive:
                initial_active = _prior_active_components(prior, selection_bank, args.device)
                fit_diagnostics["guard_initial_prior_active_components"] = int(initial_active)
                min_initial_active = _guard_min_count(
                    n_clusters,
                    args.adaptive_guard_min_active_fraction,
                    args.adaptive_guard_min_active,
                )
                if 0 <= initial_active < min_initial_active:
                    fallback_method = _guard_fallback_method(use_conv, use_mi, args)
                    fallback_base = fallback_method.removesuffix("_mi").removeprefix("conv_")
                    fallback_fitter = _fmvmm_fitter(fallback_base, n_clusters, args, seed)
                    prior, fit_diagnostics = accepted_prior_refit(
                        None,
                        fallback_fitter,
                        z_bank,
                        tolerance=args.prior_acceptance_tolerance,
                        device=args.device,
                    )
                    fit_diagnostics["guard_triggered_pretrain"] = True
                    fit_diagnostics["guard_reason_pretrain"] = (
                        f"initial_active_components={initial_active}<min_active={min_initial_active}"
                    )
                    fit_diagnostics["guard_fallback_method"] = fallback_method
                    selected_family = args.adaptive_guard_fallback_family
        elif fit_base_method in {"ghvade_conservative", "ghvade_tail_select"}:
            selection_bank = posterior_sample_bank(
                model,
                selection_x,
                n_mc_samples=args.mixture_refit_mc_samples,
                batch_size=args.batch_size,
                device=args.device,
                seed=seed + 2130,
            )
            candidates = (
                _conservative_prior_candidates(n_clusters, args, seed)
                if fit_base_method == "ghvade_conservative"
                else _tail_prior_candidates(n_clusters, args, seed)
            )
            selection = fit_and_select_prior(
                candidates,
                z_bank,
                selection_bank,
                device=args.device,
                selection_rule=args.prior_selection_rule,
                shape_skew_threshold=args.shape_skew_threshold,
                shape_score_margin=args.shape_score_margin,
            )
            prior, fit_diagnostics = accept_prior_candidate(
                None,
                selection.prior,
                z_bank,
                candidate_diagnostics=selection.diagnostics,
                tolerance=args.prior_acceptance_tolerance,
                device=args.device,
            )
            selected_family = selection.candidate_name
        else:
            fitter = _fmvmm_fitter(
                fit_base_method,
                n_clusters,
                args,
                seed,
                stabilize_fixed_skew_t=fit_base_method == "ghvade_mgst",
                stabilize_fixed_skew_normal=fit_base_method == "ghvade_mvsn",
            )
            prior, fit_diagnostics = accepted_prior_refit(
                None,
                fitter,
                z_bank,
                tolerance=args.prior_acceptance_tolerance,
                device=args.device,
            )
            selected_family = _base_fmvmm_method(fit_base_method).removeprefix(
                "fmvmm_"
            )
            if three_way_protocol and selected_family == "mvt":
                selected_family = "student_t"
        fit_diagnostics["active_prior_family"] = selected_family
        fit_diagnostics["cycle"] = -1
        prior_update_history.append(fit_diagnostics)
        for cycle in range(int(args.mixture_cycles)):
            safe_guard_cycle = (
                (safe_adaptive and cycle > 0)
                or (component_adaptive and selected_family != "student_t")
                or (
                    staged_component_adaptive
                    and selected_family != "student_t"
                )
            )
            safe_model_state_before = None
            safe_cycle_selection_bank = None
            safe_neural_block_diagnostics: dict[str, object] = {}
            if safe_guard_cycle:
                safe_model_state_before = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            neural_start = time.perf_counter()
            phase = train_mixture_vae(
                model,
                training_x,
                prior,
                MixtureTrainConfig(
                    epochs=args.mixture_epochs,
                    batch_size=args.batch_size,
                    lr=args.mixture_lr,
                    beta=args.beta,
                    recon_variance=args.recon_variance,
                    n_mc_samples=args.mixture_mc_samples,
                    responsibility_temperature=args.mixture_temperature,
                    cluster_mi_weight=args.mixture_mi_weight if use_mi else 0.0,
                    optimize_prior=False,
                    grad_clip_norm=args.mixture_grad_clip,
                    early_stopping_patience=args.mixture_patience,
                    device=args.device,
                    seed=seed + 2131 + cycle,
                    objective="factorized" if use_mi else "collapsed",
                    assignment_anchor_weight=(
                        args.component_anchor_weight
                        if component_adaptive
                        and safe_training_assignment_anchor is not None
                        else 0.0
                    ),
                    assignment_anchor_confidence_threshold=(
                        args.component_anchor_confidence_threshold
                    ),
                    latent_mean_anchor_weight=args.mixture_latent_anchor_weight,
                    posterior_anchor_weight=posterior_anchor_weight,
                    posterior_anchor_max_validation_kl=(
                        posterior_anchor_max_validation_kl
                    ),
                ),
                validation_x=neural_validation_x,
                validation_mc_samples=args.mixture_validation_mc_samples,
                validation_seed=seed + 2137 + cycle,
                state_sink=training_state,
                assignment_anchor=safe_training_assignment_anchor,
                latent_mean_anchor=training_latent_mean_anchor,
                posterior_anchor_logvar=training_posterior_logvar_anchor,
                validation_latent_mean_anchor=neural_validation_latent_mean_anchor,
                validation_posterior_anchor_logvar=(
                    neural_validation_posterior_logvar_anchor
                ),
            )
            neural_training_seconds += time.perf_counter() - neural_start
            history.extend(phase)
            if safe_guard_cycle:
                if safe_assignment_thresholds is None or safe_model_state_before is None:
                    raise RuntimeError(
                        "Safe adaptive assignment thresholds were not initialized."
                    )
                safe_cycle_selection_bank = posterior_sample_bank(
                    model,
                    selection_x,
                    n_mc_samples=args.mixture_refit_mc_samples,
                    batch_size=args.batch_size,
                    device=args.device,
                    seed=seed + 2142 + cycle,
                )
                neural_assignment = prior_assignment_diagnostics(
                    prior,
                    safe_cycle_selection_bank,
                    device=args.device,
                    component_mass_threshold=args.safe_component_mass_threshold,
                )
                neural_violations = _safe_assignment_violations(
                    neural_assignment,
                    safe_assignment_thresholds,
                )
                neural_block_accepted = not neural_violations
                if not neural_block_accepted:
                    model.load_state_dict(safe_model_state_before)
                    safe_cycle_selection_bank = posterior_sample_bank(
                        model,
                        selection_x,
                        n_mc_samples=args.mixture_refit_mc_samples,
                        batch_size=args.batch_size,
                        device=args.device,
                        seed=seed + 2142 + cycle,
                    )
                    if staged_component_adaptive and staged_anchor_prior is not None:
                        prior = staged_anchor_prior
                        selected_family = "student_t"
                        fallback_start = time.perf_counter()
                        fallback_phase = train_mixture_vae(
                            model,
                            training_x,
                            prior,
                            MixtureTrainConfig(
                                epochs=args.mixture_epochs,
                                batch_size=args.batch_size,
                                lr=args.mixture_lr,
                                beta=args.beta,
                                recon_variance=args.recon_variance,
                                n_mc_samples=args.mixture_mc_samples,
                                responsibility_temperature=args.mixture_temperature,
                                cluster_mi_weight=(
                                    args.mixture_mi_weight if use_mi else 0.0
                                ),
                                optimize_prior=False,
                                grad_clip_norm=args.mixture_grad_clip,
                                early_stopping_patience=args.mixture_patience,
                                device=args.device,
                                seed=seed + 2131 + cycle,
                                objective="factorized" if use_mi else "collapsed",
                                latent_mean_anchor_weight=(
                                    args.mixture_latent_anchor_weight
                                ),
                                posterior_anchor_weight=(
                                    posterior_anchor_weight
                                ),
                                posterior_anchor_max_validation_kl=(
                                    posterior_anchor_max_validation_kl
                                ),
                            ),
                            validation_x=neural_validation_x,
                            validation_mc_samples=args.mixture_validation_mc_samples,
                            validation_seed=seed + 2137 + cycle,
                            state_sink=training_state,
                            latent_mean_anchor=training_latent_mean_anchor,
                            posterior_anchor_logvar=(
                                training_posterior_logvar_anchor
                            ),
                            validation_latent_mean_anchor=(
                                neural_validation_latent_mean_anchor
                            ),
                            validation_posterior_anchor_logvar=(
                                neural_validation_posterior_logvar_anchor
                            ),
                        )
                        neural_training_seconds += (
                            time.perf_counter() - fallback_start
                        )
                        history.extend(fallback_phase)
                safe_neural_block_diagnostics = {
                    "safe_neural_block_accepted": neural_block_accepted,
                    "safe_neural_assignment": neural_assignment,
                    "safe_neural_rejection_reasons": neural_violations,
                    "safe_assignment_thresholds": safe_assignment_thresholds,
                }
                if component_adaptive and not neural_block_accepted:
                    componentwise_fallback_required = True
                    componentwise_fallback_reasons.extend(
                        f"cycle_{cycle}:{reason}" for reason in neural_violations
                    )
                if staged_component_adaptive and not neural_block_accepted:
                    safe_neural_block_diagnostics.update(
                        {
                            "staged_promotion_revoked": True,
                            "staged_fallback_family": "student_t",
                            "staged_student_t_block_rerun": True,
                        }
                    )
            z_bank = posterior_sample_bank(
                model,
                training_x,
                n_mc_samples=args.mixture_refit_mc_samples,
                batch_size=args.batch_size,
                device=args.device,
                seed=seed + 2141 + cycle,
            )
            if safe_adaptive and cycle == 0:
                selection_bank = posterior_sample_bank(
                    model,
                    selection_x,
                    n_mc_samples=args.mixture_refit_mc_samples,
                    batch_size=args.batch_size,
                    device=args.device,
                    seed=seed + 2142 + cycle,
                )
                selection = fit_and_select_prior(
                    _adaptive_prior_candidates(n_clusters, args, seed + cycle + 1),
                    z_bank,
                    selection_bank,
                    device=args.device,
                    selection_rule="student_t_safe",
                    shape_skew_threshold=args.shape_skew_threshold,
                    shape_score_margin=args.shape_score_margin,
                    safe_confidence_z=args.safe_confidence_z,
                    safe_familywise_alpha=args.safe_familywise_alpha,
                    safe_min_active_fraction=args.safe_min_active_fraction,
                    safe_assignment_tolerance=args.safe_assignment_tolerance,
                    safe_component_mass_threshold=args.safe_component_mass_threshold,
                    safe_minimum_weight=args.safe_minimum_weight,
                    safe_minimum_covariance_eigenvalue=(
                        args.safe_minimum_covariance_eigenvalue
                    ),
                    safe_maximum_covariance_condition=(
                        args.safe_maximum_covariance_condition
                    ),
                    safe_minimum_interior_shape=args.safe_minimum_interior_shape,
                    safe_core_confidence_threshold=(
                        args.safe_core_confidence_threshold
                    ),
                    safe_minimum_core_fraction=args.safe_minimum_core_fraction,
                    safe_minimum_core_retention=args.safe_minimum_core_retention,
                )
                candidate_family = selection.candidate_name
                prior, fit_diagnostics = accept_prior_candidate(
                    prior,
                    selection.prior,
                    selection_bank,
                    candidate_diagnostics=selection.diagnostics,
                    tolerance=args.prior_acceptance_tolerance,
                    device=args.device,
                )
                if fit_diagnostics["prior_update_accepted"]:
                    selected_family = candidate_family
                safe_assignment_thresholds = _safe_assignment_floors(
                    selection.diagnostics["safe_anchor_assignment"],
                    int(selection.diagnostics["safe_minimum_active_components"]),
                    args.safe_assignment_tolerance,
                )
                family_code = ADAPTIVE_FAMILY_CODE[selected_family]
                locked_fitter = _fmvmm_fitter(
                    f"fmvmm_{family_code}", n_clusters, args, seed + cycle + 1
                )
                fit_diagnostics["safe_anchor_initialization"] = False
                fit_diagnostics["safe_family_locked"] = True
                fit_diagnostics["safe_locked_family"] = selected_family
            elif component_adaptive and selected_family == "student_t":
                if locked_fitter is None:
                    raise RuntimeError("Componentwise enrichment lost its Student-t fitter.")
                prior, fit_diagnostics = accepted_prior_refit(
                    prior,
                    locked_fitter,
                    z_bank,
                    tolerance=args.prior_acceptance_tolerance,
                    device=args.device,
                )
            elif safe_adaptive:
                if locked_fitter is None:
                    raise RuntimeError("Safe adaptive family was not locked after anchor warm-up.")
                if safe_assignment_thresholds is None:
                    raise RuntimeError(
                        "Safe adaptive assignment thresholds were not initialized."
                    )
                if safe_cycle_selection_bank is None:
                    safe_cycle_selection_bank = posterior_sample_bank(
                        model,
                        selection_x,
                        n_mc_samples=args.mixture_refit_mc_samples,
                        batch_size=args.batch_size,
                        device=args.device,
                        seed=seed + 2142 + cycle,
                    )
                flat_bank = z_bank.reshape(-1, z_bank.shape[-1])
                fit_result = locked_fitter.fit(flat_bank)
                regularity = prior_regularity_diagnostics(
                    fit_result.prior,
                    minimum_weight=args.safe_minimum_weight,
                    minimum_covariance_eigenvalue=(
                        args.safe_minimum_covariance_eigenvalue
                    ),
                    maximum_covariance_condition=args.safe_maximum_covariance_condition,
                    minimum_interior_shape=args.safe_minimum_interior_shape,
                )
                candidate_assignment = prior_assignment_diagnostics(
                    fit_result.prior,
                    safe_cycle_selection_bank,
                    device=args.device,
                    component_mass_threshold=args.safe_component_mass_threshold,
                )
                assignment_violations = _safe_assignment_violations(
                    candidate_assignment,
                    safe_assignment_thresholds,
                )
                if regularity["regular"] and not assignment_violations:
                    prior, fit_diagnostics = accept_prior_candidate(
                        prior,
                        fit_result.prior,
                        safe_cycle_selection_bank,
                        candidate_diagnostics={
                            **fit_result.diagnostics,
                            "regularity": regularity,
                            "safe_candidate_assignment": candidate_assignment,
                            "safe_assignment_thresholds": safe_assignment_thresholds,
                            "safe_family_locked": True,
                            "safe_locked_family": selected_family,
                        },
                        tolerance=args.prior_acceptance_tolerance,
                        device=args.device,
                    )
                else:
                    fit_diagnostics = {
                        **fit_result.diagnostics,
                        "regularity": regularity,
                        "safe_candidate_assignment": candidate_assignment,
                        "safe_assignment_thresholds": safe_assignment_thresholds,
                        "safe_assignment_rejection_reasons": assignment_violations,
                        "prior_update_accepted": False,
                        "prior_rejection_reason": (
                            "irregular_locked_family_refit"
                            if not regularity["regular"]
                            else "unsafe_locked_family_refit"
                        ),
                        "safe_family_locked": True,
                        "safe_locked_family": selected_family,
                    }
                fit_diagnostics.update(safe_neural_block_diagnostics)
            elif component_adaptive:
                fit_diagnostics = {
                    "prior_update_accepted": False,
                    "prior_rejection_reason": "componentwise_family_locked_no_refit",
                    "safe_family_locked": True,
                    "safe_locked_family": selected_family,
                    **safe_neural_block_diagnostics,
                }
            elif staged_component_adaptive and selected_family != "student_t":
                fit_diagnostics = {
                    "prior_update_accepted": False,
                    "prior_rejection_reason": "staged_component_family_locked_no_refit",
                    "safe_family_locked": True,
                    "safe_locked_family": selected_family,
                    **safe_neural_block_diagnostics,
                }
            elif fit_base_method == "ghvade_adaptive":
                selection_bank = posterior_sample_bank(
                    model,
                    selection_x,
                    n_mc_samples=args.mixture_refit_mc_samples,
                    batch_size=args.batch_size,
                    device=args.device,
                    seed=seed + 2142 + cycle,
                )
                selection = fit_and_select_prior(
                    _adaptive_prior_candidates(n_clusters, args, seed + cycle + 1),
                    z_bank,
                    selection_bank,
                    device=args.device,
                    selection_rule=args.prior_selection_rule,
                    shape_skew_threshold=args.shape_skew_threshold,
                    shape_score_margin=args.shape_score_margin,
                )
                candidate_family = selection.candidate_name
                prior, fit_diagnostics = accept_prior_candidate(
                    prior,
                    selection.prior,
                    z_bank,
                    candidate_diagnostics=selection.diagnostics,
                    tolerance=args.prior_acceptance_tolerance,
                    device=args.device,
                )
                if fit_diagnostics["prior_update_accepted"]:
                    selected_family = candidate_family
            elif fit_base_method in {"ghvade_conservative", "ghvade_tail_select"}:
                selection_bank = posterior_sample_bank(
                    model,
                    selection_x,
                    n_mc_samples=args.mixture_refit_mc_samples,
                    batch_size=args.batch_size,
                    device=args.device,
                    seed=seed + 2142 + cycle,
                )
                candidates = (
                    _conservative_prior_candidates(n_clusters, args, seed + cycle + 1)
                    if fit_base_method == "ghvade_conservative"
                    else _tail_prior_candidates(n_clusters, args, seed + cycle + 1)
                )
                selection = fit_and_select_prior(
                    candidates,
                    z_bank,
                    selection_bank,
                    device=args.device,
                    selection_rule=args.prior_selection_rule,
                    shape_skew_threshold=args.shape_skew_threshold,
                    shape_score_margin=args.shape_score_margin,
                )
                candidate_family = selection.candidate_name
                prior, fit_diagnostics = accept_prior_candidate(
                    prior,
                    selection.prior,
                    z_bank,
                    candidate_diagnostics=selection.diagnostics,
                    tolerance=args.prior_acceptance_tolerance,
                    device=args.device,
                )
                if fit_diagnostics["prior_update_accepted"]:
                    selected_family = candidate_family
            else:
                prior, fit_diagnostics = accepted_prior_refit(
                    prior,
                    fitter,
                    z_bank,
                    tolerance=args.prior_acceptance_tolerance,
                    device=args.device,
                )
            if safe_neural_block_diagnostics:
                fit_diagnostics.update(safe_neural_block_diagnostics)
            fit_diagnostics["active_prior_family"] = selected_family
            fit_diagnostics["cycle"] = cycle
            prior_update_history.append(fit_diagnostics)
            if staged_protocol and cycle == 0:
                staged_optimization_bank = posterior_sample_bank(
                    model,
                    development_x,
                    n_mc_samples=args.mixture_refit_mc_samples,
                    batch_size=args.batch_size,
                    device=args.device,
                    seed=seed + 2161,
                )
                staged_selection_bank = posterior_sample_bank(
                    model,
                    selection_x,
                    n_mc_samples=args.mixture_refit_mc_samples,
                    batch_size=args.batch_size,
                    device=args.device,
                    seed=seed + 2162,
                )
                staged_fitter = _fmvmm_fitter(
                    "fmvmm_mvt", n_clusters, args, seed + 61
                )
                staged_anchor_prior, staged_anchor_diagnostics = accepted_prior_refit(
                    None,
                    staged_fitter,
                    staged_optimization_bank,
                    tolerance=args.prior_acceptance_tolerance,
                    device=args.device,
                )
                staged_regularity = prior_regularity_diagnostics(
                    staged_anchor_prior,
                    minimum_weight=args.safe_minimum_weight,
                    minimum_covariance_eigenvalue=(
                        args.safe_minimum_covariance_eigenvalue
                    ),
                    maximum_covariance_condition=(
                        args.safe_maximum_covariance_condition
                    ),
                    minimum_interior_shape=args.safe_minimum_interior_shape,
                )
                if not staged_regularity["regular"]:
                    raise RuntimeError(
                        "The staged Student-t anchor lies outside the declared "
                        f"regular parameter region: {staged_regularity['violations']}"
                    )
                staged_anchor_diagnostics["regularity"] = staged_regularity
                staged_anchor_diagnostics["staged_anchor_refit"] = True
                prior = staged_anchor_prior
                selected_family = "student_t"
                if staged_component_adaptive:
                    enrichment = fit_componentwise_gh_enrichment(
                        staged_anchor_prior,
                        staged_optimization_bank,
                        staged_selection_bank,
                        config=_component_enrichment_config(args),
                        fmvmm_path=args.fmvmm_path,
                        device=args.device,
                    )
                    prior, staged_diagnostics = accept_prior_candidate(
                        staged_anchor_prior,
                        enrichment.prior,
                        staged_selection_bank,
                        candidate_diagnostics=enrichment.diagnostics,
                        tolerance=args.prior_acceptance_tolerance,
                        device=args.device,
                    )
                    if (
                        staged_diagnostics["prior_update_accepted"]
                        and enrichment.diagnostics["promoted"]
                    ):
                        selected_family = (
                            f"student_t+{enrichment.selected_family}"
                        )
                    safe_assignment_thresholds = _safe_assignment_floors(
                        enrichment.diagnostics["anchor_assignment"],
                        int(enrichment.diagnostics["minimum_active_components"]),
                        args.safe_assignment_tolerance,
                    )
                    staged_diagnostics["staged_anchor_fit"] = (
                        staged_anchor_diagnostics
                    )
                else:
                    staged_diagnostics = staged_anchor_diagnostics
                staged_diagnostics.update(
                    {
                        "active_prior_family": selected_family,
                        "cycle": cycle,
                        "staged_family_selection": True,
                        "staged_neural_backbone_family": "student_t",
                        "staged_selection_after_cycle": cycle,
                    }
                )
                prior_update_history.append(staged_diagnostics)
        if posthoc_protocol:
            posthoc_optimization_bank = posterior_sample_bank(
                model,
                development_x,
                n_mc_samples=args.mixture_refit_mc_samples,
                batch_size=args.batch_size,
                device=args.device,
                seed=seed + 2181,
            )
            posthoc_selection_bank = posterior_sample_bank(
                model,
                selection_x,
                n_mc_samples=args.mixture_refit_mc_samples,
                batch_size=args.batch_size,
                device=args.device,
                seed=seed + 2182,
            )
            posthoc_anchor_fitter = _fmvmm_fitter(
                "fmvmm_mvt", n_clusters, args, seed + 81
            )
            posthoc_anchor, posthoc_anchor_diagnostics = accepted_prior_refit(
                None,
                posthoc_anchor_fitter,
                posthoc_optimization_bank,
                tolerance=args.prior_acceptance_tolerance,
                device=args.device,
            )
            posthoc_anchor_regularity = prior_regularity_diagnostics(
                posthoc_anchor,
                minimum_weight=args.safe_minimum_weight,
                minimum_covariance_eigenvalue=(
                    args.safe_minimum_covariance_eigenvalue
                ),
                maximum_covariance_condition=args.safe_maximum_covariance_condition,
                minimum_interior_shape=args.safe_minimum_interior_shape,
            )
            if not posthoc_anchor_regularity["regular"]:
                raise RuntimeError(
                    "The final Student-t anchor lies outside the declared regular "
                    f"parameter region: {posthoc_anchor_regularity['violations']}"
                )
            posthoc_anchor_diagnostics["regularity"] = posthoc_anchor_regularity
            posthoc_anchor_diagnostics["posthoc_anchor_refit"] = True
            prior = posthoc_anchor
            selected_family = "student_t"
            if posthoc_component_adaptive:
                enrichment = fit_componentwise_gh_enrichment(
                    posthoc_anchor,
                    posthoc_optimization_bank,
                    posthoc_selection_bank,
                    config=_component_enrichment_config(args),
                    fmvmm_path=args.fmvmm_path,
                    device=args.device,
                )
                prior, fit_diagnostics = accept_prior_candidate(
                    posthoc_anchor,
                    enrichment.prior,
                    posthoc_selection_bank,
                    candidate_diagnostics=enrichment.diagnostics,
                    tolerance=args.prior_acceptance_tolerance,
                    device=args.device,
                )
                if (
                    fit_diagnostics["prior_update_accepted"]
                    and enrichment.diagnostics["promoted"]
                ):
                    selected_family = f"student_t+{enrichment.selected_family}"
                fit_diagnostics["posthoc_anchor_fit"] = posthoc_anchor_diagnostics
            else:
                fit_diagnostics = posthoc_anchor_diagnostics
            fit_diagnostics.update(
                {
                    "active_prior_family": selected_family,
                    "cycle": int(args.mixture_cycles),
                    "posthoc_family_selection": True,
                    "posthoc_neural_backbone_family": "student_t",
                    "posthoc_representation_frozen_during_selection": True,
                }
            )
            prior_update_history.append(fit_diagnostics)
    else:
        raise ValueError(f"Unknown mixture-aware method {method!r}.")

    mixture_pipeline_seconds = time.perf_counter() - mixture_stage_start
    final_training_encoding = encode_dataset(
        model,
        training_x,
        batch_size=args.batch_size,
        device=args.device,
    )
    final_training_logvar = final_training_encoding["logvar"]
    final_training_mu = final_training_encoding["mu"]
    training_latent_delta = final_training_mu - training_latent_mean_anchor
    training_latent_l2 = torch.linalg.vector_norm(training_latent_delta, dim=1)
    training_latent_coordinate_mse = float(training_latent_delta.square().mean())
    final_validation_encoding = encode_dataset(
        model,
        neural_validation_x,
        batch_size=args.batch_size,
        device=args.device,
    )
    final_validation_mu = final_validation_encoding["mu"]
    final_validation_logvar = final_validation_encoding["logvar"]
    validation_latent_delta = (
        final_validation_mu - neural_validation_latent_mean_anchor
    )
    validation_latent_l2 = torch.linalg.vector_norm(validation_latent_delta, dim=1)
    training_posterior_anchor_kl = diagonal_gaussian_forward_kl(
        training_latent_mean_anchor,
        training_posterior_logvar_anchor,
        final_training_mu,
        final_training_logvar,
    )
    validation_posterior_anchor_kl = diagonal_gaussian_forward_kl(
        neural_validation_latent_mean_anchor,
        neural_validation_posterior_logvar_anchor,
        final_validation_mu,
        final_validation_logvar,
    )
    evaluation_start = time.perf_counter()
    responsibilities = encode_mixture_responsibilities(
        model,
        x,
        prior,
        batch_size=args.batch_size,
        device=args.device,
        n_mc_samples=args.mixture_eval_mc_samples,
        seed=seed + 2151,
    ).numpy()
    evaluation_seconds = time.perf_counter() - evaluation_start
    pred = responsibilities.argmax(axis=1)
    selection_responsibilities = responsibilities[selection_indices.numpy()]
    graph_diagnostics = _soft_graph_assortativity(
        selection_graph_features,
        selection_responsibilities,
        args.component_graph_neighbors,
    )
    tail = history[-1] if history else {}
    prior_parameter_summary = _prior_parameter_summary(prior)
    final_prior_regularity = prior_regularity_diagnostics(
        prior,
        minimum_weight=args.safe_minimum_weight,
        minimum_covariance_eigenvalue=args.safe_minimum_covariance_eigenvalue,
        maximum_covariance_condition=args.safe_maximum_covariance_condition,
        minimum_interior_shape=args.safe_minimum_interior_shape,
    )
    if artifact_dir is not None:
        if pretrain_all_encoding is None:  # pragma: no cover - guarded by artifact_dir
            raise RuntimeError("Pretraining embeddings were not captured for artifact output.")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        encoded_all = encode_dataset(model, x, batch_size=args.batch_size, device=args.device)
        np.savez_compressed(
            artifact_dir / "embeddings.npz",
            mu=encoded_all["mu"].numpy(),
            logvar=encoded_all["logvar"].numpy(),
            pretrain_mu=pretrain_all_encoding["mu"].numpy(),
            pretrain_logvar=pretrain_all_encoding["logvar"].numpy(),
            responsibilities=responsibilities,
            labels=y,
            eval_mask=eval_mask,
            optimization_indices=training_indices.numpy(),
            selection_indices=selection_indices.numpy(),
        )
        torch.save(
            {
                "schema_version": 1,
                "method": method,
                "seed": int(seed),
                "selected_family": selected_family,
                "model_class": type(model).__name__,
                "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "prior_class": type(prior).__name__,
                "prior_name": prior.name,
                "prior_parameter_summary": prior_parameter_summary,
                "prior_state_dict": {key: value.detach().cpu() for key, value in prior.state_dict().items()},
                "optimizer_state_dict": training_state.get("optimizer_state_dict"),
                "optimizer_metadata": {
                    key: value for key, value in training_state.items() if key != "optimizer_state_dict"
                },
                "model_config": {
                    "input_dim": int(input_dim),
                    "latent_dim": int(latent_dim),
                    "hidden_dims": list(hidden_dims),
                    "conv_channels": list(args.conv_channels) if use_conv else None,
                    "image_shape": list(image_shape) if use_conv and image_shape is not None else None,
                    "latent_mean_anchor_weight": float(
                        args.mixture_latent_anchor_weight
                    ),
                    "posterior_anchor_weight": float(
                        posterior_anchor_weight
                    ),
                    "posterior_anchor_max_validation_kl": (
                        posterior_anchor_max_validation_kl
                    ),
                },
                "optimization_indices": training_indices,
                "selection_indices": selection_indices,
            },
            artifact_dir / "checkpoint.pt",
        )
        write_json(
            artifact_dir / "training_history.json",
            {
                "pretrain": pretrain_history,
                "neural": history,
                "prior_updates": prior_update_history,
                "latent_mean_trust_region": {
                    "weight": float(args.mixture_latent_anchor_weight),
                    "training_coordinate_mse": training_latent_coordinate_mse,
                    "training_mean_l2": float(training_latent_l2.mean()),
                    "training_max_l2": float(training_latent_l2.max()),
                    "validation_coordinate_mse": float(
                        validation_latent_delta.square().mean()
                    ),
                    "validation_mean_l2": float(validation_latent_l2.mean()),
                    "validation_max_l2": float(validation_latent_l2.max()),
                    "posterior_weight": float(
                        posterior_anchor_weight
                    ),
                    "posterior_max_validation_kl": (
                        posterior_anchor_max_validation_kl
                    ),
                    "training_posterior_kl": float(
                        training_posterior_anchor_kl.mean()
                    ),
                    "validation_posterior_kl": float(
                        validation_posterior_anchor_kl.mean()
                    ),
                    "training_logvar_mean_absolute_drift": float(
                        (final_training_logvar - training_posterior_logvar_anchor)
                        .abs()
                        .mean()
                    ),
                    "validation_logvar_mean_absolute_drift": float(
                        (
                            final_validation_logvar
                            - neural_validation_posterior_logvar_anchor
                        )
                        .abs()
                        .mean()
                    ),
                },
            },
        )
        write_json(
            artifact_dir / "prior_metadata.json",
            {
                "prior_class": type(prior).__name__,
                "prior_name": prior.name,
                "selected_family": selected_family,
                "parameter_summary": prior_parameter_summary,
                "state_dict_shapes": {
                    key: list(value.shape) for key, value in prior.state_dict().items()
                },
                "final_fit_diagnostics": fit_diagnostics,
            },
        )
        write_json(artifact_dir / "split.json", split_diagnostics)
    out = {
        "method": method,
        "status": "ok",
        "prior_name": prior.name,
        "mixture_pretrain_recon_loss": float(pretrain_history[-1]["recon_loss"]) if pretrain_history else math.nan,
        "final_loss": float(tail.get("loss", math.nan)),
        "final_elbo": float(tail.get("elbo", math.nan)),
        "final_cluster_mi": float(tail.get("cluster_mutual_information", math.nan)),
        "latent_mean_anchor_weight": float(args.mixture_latent_anchor_weight),
        "posterior_anchor_weight": posterior_anchor_weight,
        "posterior_anchor_max_validation_kl": (
            posterior_anchor_max_validation_kl
        ),
        "latent_mean_anchor_training_coordinate_mse": training_latent_coordinate_mse,
        "latent_mean_anchor_training_rms_l2": float(
            math.sqrt(max(training_latent_coordinate_mse * latent_dim, 0.0))
        ),
        "latent_mean_anchor_training_mean_l2": float(training_latent_l2.mean()),
        "latent_mean_anchor_training_max_l2": float(training_latent_l2.max()),
        "latent_mean_anchor_validation_coordinate_mse": float(
            validation_latent_delta.square().mean()
        ),
        "latent_mean_anchor_validation_mean_l2": float(validation_latent_l2.mean()),
        "latent_mean_anchor_validation_max_l2": float(validation_latent_l2.max()),
        "posterior_anchor_training_kl": float(training_posterior_anchor_kl.mean()),
        "posterior_anchor_validation_kl": float(
            validation_posterior_anchor_kl.mean()
        ),
        "posterior_anchor_training_logvar_mean_absolute_drift": float(
            (final_training_logvar - training_posterior_logvar_anchor).abs().mean()
        ),
        "posterior_anchor_validation_logvar_mean_absolute_drift": float(
            (
                final_validation_logvar
                - neural_validation_posterior_logvar_anchor
            )
            .abs()
            .mean()
        ),
        "active_components": int((responsibilities.mean(axis=0) > 0.01).sum()),
        "fit_diagnostics": fit_diagnostics,
        "final_prior_regularity": final_prior_regularity,
        "prior_update_history": prior_update_history,
        "selected_family": selected_family,
        "split_diagnostics": split_diagnostics,
        "pretrain_seconds": float(pretrain_seconds),
        "neural_training_seconds": float(neural_training_seconds),
        "prior_update_pipeline_seconds": float(
            max(mixture_pipeline_seconds - neural_training_seconds, 0.0)
        ),
        "evaluation_seconds": float(evaluation_seconds),
        "method_internal_seconds": float(time.perf_counter() - method_start),
        "n_prior_updates": int(len(prior_update_history)),
        "n_rejected_prior_updates": int(
            sum(update.get("prior_update_accepted") is False for update in prior_update_history)
        ),
        "n_failed_candidate_fits": int(
            sum(
                candidate.get("status") == "failed"
                for update in prior_update_history
                for candidate in update.get("candidates", [])
            )
        ),
        "candidate_fit_and_score_seconds": float(
            sum(
                float(candidate.get("fit_and_score_seconds", 0.0))
                for update in prior_update_history
                for candidate in update.get("candidates", [])
            )
        ),
        **graph_diagnostics,
        **_metrics(y[eval_mask], pred[eval_mask]),
    }
    component_parameters = prior_parameter_summary.get("components", [])
    if isinstance(component_parameters, list):
        for field in ("a_norm", "gamma_norm", "skewness_norm", "df", "lmbda", "chi", "psi"):
            values = [
                float(component[field])
                for component in component_parameters
                if isinstance(component, dict) and field in component and math.isfinite(float(component[field]))
            ]
            if values:
                out[f"component_{field}_mean"] = float(np.mean(values))
    if "best_bic" in fit_diagnostics:
        out["bic"] = float(fit_diagnostics["best_bic"])
    if component_adaptive and componentwise_fallback_required:
        fallback_prefix = "conv_" if use_conv else ""
        fallback_suffix = "_mi" if use_mi else ""
        fallback_base = "ghvade_trust_mvt" if trust_component_adaptive else "ghvade_mvt"
        fallback_method = f"{fallback_prefix}{fallback_base}{fallback_suffix}"
        fallback_artifact_dir = (
            artifact_dir / "componentwise_fallback"
            if artifact_dir is not None
            else None
        )
        fallback = _fit_mixture_aware_vae(
            fallback_method,
            x,
            y,
            eval_mask,
            input_dim,
            latent_dim,
            hidden_dims,
            n_clusters,
            args,
            seed,
            artifact_dir=fallback_artifact_dir,
            image_shape=image_shape,
        )
        return _componentwise_fallback_result(
            initial=out,
            fallback=fallback,
            method=method,
            fallback_method=fallback_method,
            rejection_reasons=componentwise_fallback_reasons,
        )
    if trust_component_adaptive and selected_family != "student_t":
        fallback_prefix = "conv_" if use_conv else ""
        fallback_suffix = "_mi" if use_mi else ""
        fallback_method = f"{fallback_prefix}ghvade_trust_mvt{fallback_suffix}"
        fallback_artifact_dir = (
            artifact_dir / "student_t_reference"
            if artifact_dir is not None
            else None
        )
        reference_start = time.perf_counter()
        fallback = _fit_mixture_aware_vae(
            fallback_method,
            x,
            y,
            eval_mask,
            input_dim,
            latent_dim,
            hidden_dims,
            n_clusters,
            args,
            seed,
            artifact_dir=fallback_artifact_dir,
            image_shape=image_shape,
        )
        reference_seconds = float(time.perf_counter() - reference_start)
        candidate_graph_score = float(out["graph_hard_assortativity"])
        reference_graph_score = float(fallback["graph_hard_assortativity"])
        minimum_graph_gain = float(
            args.component_minimum_graph_assortativity_gain
        )
        graph_accepted, graph_gain = _graph_tournament_decision(
            candidate_graph_score,
            reference_graph_score,
            minimum_graph_gain,
        )
        tournament = {
            "componentwise_tournament_accepted": bool(graph_accepted),
            "componentwise_tournament_score": "graph_hard_assortativity",
            "componentwise_tournament_candidate_score": candidate_graph_score,
            "componentwise_tournament_student_t_score": reference_graph_score,
            "componentwise_tournament_score_gain": graph_gain,
            "componentwise_tournament_minimum_gain": minimum_graph_gain,
            "componentwise_tournament_reference_seconds": reference_seconds,
            "componentwise_tournament_student_t_acc": float(fallback["acc"]),
            "componentwise_tournament_student_t_ari": float(fallback["ari"]),
            "componentwise_tournament_student_t_nmi": float(fallback["nmi"]),
        }
        if not graph_accepted:
            graph_rejection_reason = (
                "nonfinite_graph_assortativity_gain"
                if not math.isfinite(graph_gain)
                else "insufficient_graph_assortativity_gain"
            )
            rejected = _componentwise_fallback_result(
                initial=out,
                fallback=fallback,
                method=method,
                fallback_method=fallback_method,
                rejection_reasons=[graph_rejection_reason],
            )
            rejected.update(tournament)
            return rejected
        out.update(tournament)
    if guarded_adaptive:
        trigger, reason, guard_diagnostics = _guard_trigger(out, n_clusters, args)
        out["guard_triggered"] = False
        out.update(guard_diagnostics)
        if trigger:
            fallback_method = _guard_fallback_method(use_conv, use_mi, args)
            fallback_artifact_dir = artifact_dir / "guard_fallback" if artifact_dir is not None else None
            fallback = _fit_mixture_aware_vae(
                fallback_method,
                x,
                y,
                eval_mask,
                input_dim,
                latent_dim,
                hidden_dims,
                n_clusters,
                args,
                seed,
                artifact_dir=fallback_artifact_dir,
                image_shape=image_shape,
            )
            return _guarded_fallback_result(
                initial=out,
                fallback=fallback,
                method=method,
                fallback_method=fallback_method,
                reason=reason,
                guard_diagnostics=guard_diagnostics,
            )
    return out


def _run_one(
    dataset_name: str,
    seed: int,
    args: argparse.Namespace,
    artifact_root: Path | None = None,
) -> list[dict[str, object]]:
    x, y, eval_mask, meta = _load_dataset(
        dataset_name,
        args.max_samples,
        seed,
        args.contamination,
        data_home=args.data_home,
        download_datasets=args.download_datasets,
        contamination_mode=args.contamination_mode,
        semantic_outlier_dataset=args.semantic_outlier_dataset,
        corruption_severity=args.corruption_severity,
    )
    blind_external_metrics = bool(args.blind_external_metrics)
    if blind_external_metrics:
        y = _blind_evaluation_labels(y, eval_mask, seed=seed + 9059)
    n_clusters = int(meta["n_classes"])
    input_dim = int(meta["input_dim"])
    image_shape = tuple(int(value) for value in meta["image_shape"]) if "image_shape" in meta else None
    latent_dim = args.latent_dim if args.latent_dim > 0 else max(2, min(8, n_clusters))
    hidden_dims = tuple(args.hidden_dims)

    rows: list[dict[str, object]] = []
    dataset_hash = hashlib.sha256()
    dataset_hash.update(x.numpy().tobytes())
    if not blind_external_metrics:
        dataset_hash.update(y.tobytes())
    dataset_hash.update(eval_mask.tobytes())
    common = {
        "dataset": dataset_name,
        "dataset_sha256": dataset_hash.hexdigest(),
        "dataset_hash_scope": (
            "features_and_eval_mask"
            if blind_external_metrics
            else "features_labels_and_eval_mask"
        ),
        "external_metrics_blinded": blind_external_metrics,
        "seed": int(seed),
        **meta,
        "latent_dim": int(latent_dim),
    }
    x_np = x.numpy()

    for method in args.methods:
        method_start = time.perf_counter()
        cpu_start = time.process_time()
        if str(args.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            if method == "raw_kmeans":
                result = _fit_kmeans(method, x_np, y, eval_mask, n_clusters, seed)
            elif method == "raw_gmm":
                result = _fit_gmm(method, x_np, y, eval_mask, n_clusters, seed)
            elif method == "ae_kmeans":
                result = _fit_ae_kmeans(
                    method,
                    x,
                    y,
                    eval_mask,
                    input_dim,
                    latent_dim,
                    hidden_dims,
                    n_clusters,
                    args,
                    seed,
                )
            elif method in {"dec", "idec"}:
                result = _fit_dec_like(
                    method,
                    x,
                    y,
                    eval_mask,
                    input_dim,
                    latent_dim,
                    hidden_dims,
                    n_clusters,
                    args,
                    seed,
                    use_reconstruction=(method == "idec"),
                )
            elif method == "conv_cc":
                result = _fit_contrastive_clustering(
                    method,
                    x,
                    y,
                    eval_mask,
                    latent_dim,
                    n_clusters,
                    args,
                    seed,
                    image_shape,
                    artifact_dir=(artifact_root / dataset_name / f"seed_{seed}" / method) if artifact_root else None,
                )
            elif method in {"vae_kmeans", "vae_gmm", "vae_diag_gmm"} or method.startswith("vae_fmvmm_"):
                _, z_mu, history = _standard_vae_embedding(
                    x, input_dim, latent_dim, hidden_dims, args, seed
                )
                if method == "vae_kmeans":
                    result = _fit_kmeans(method, z_mu, y, eval_mask, n_clusters, seed)
                elif method == "vae_gmm":
                    result = _fit_gmm(method, z_mu, y, eval_mask, n_clusters, seed)
                elif method == "vae_diag_gmm":
                    result = _fit_diag_prior_on_features(method, z_mu, y, eval_mask, n_clusters, args, seed)
                else:
                    result = _fit_fmvmm_on_features(method, z_mu, y, eval_mask, n_clusters, args, seed)
                result["pretrain_final_loss"] = float(history[-1].get("loss", math.nan))
            elif method == "diag_gmm_prior" or method.startswith("fmvmm_"):
                result = _fit_prior_vae(
                    method,
                    x,
                    y,
                    eval_mask,
                    input_dim,
                    latent_dim,
                    hidden_dims,
                    n_clusters,
                    args,
                    seed,
                )
            elif (
                method.startswith("vade_diag")
                or method.startswith("ghvade_")
                or method.startswith("conv_vade_diag")
                or method.startswith("conv_ghvade_")
            ):
                result = _fit_mixture_aware_vae(
                    method,
                    x,
                    y,
                    eval_mask,
                    input_dim,
                    latent_dim,
                    hidden_dims,
                    n_clusters,
                    args,
                    seed,
                    artifact_dir=(artifact_root / dataset_name / f"seed_{seed}" / method) if artifact_root else None,
                    image_shape=image_shape,
                )
            else:
                raise ValueError(f"Unknown method {method!r}")
        except Exception as exc:
            result = {"method": method, "status": "failed", "error": repr(exc)}
        result["runtime_seconds"] = float(time.perf_counter() - method_start)
        result["cpu_process_seconds"] = float(time.process_time() - cpu_start)
        peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        result["peak_rss_mb"] = (
            peak_rss / (1024.0 * 1024.0) if sys.platform == "darwin" else peak_rss / 1024.0
        )
        result["device"] = str(args.device)
        result["requested_cpu_cores"] = int(
            os.environ.get("SLURM_CPUS_PER_TASK", args.torch_threads)
        )
        result["gpu_peak_memory_mb"] = (
            float(torch.cuda.max_memory_allocated() / (1024.0**2))
            if str(args.device).startswith("cuda") and torch.cuda.is_available()
            else 0.0
        )
        if blind_external_metrics:
            stripped = _strip_external_metrics(result)
            if not isinstance(stripped, dict):  # pragma: no cover - structural guard
                raise TypeError("Blinded result must remain a dictionary.")
            result = stripped
        rows.append({**common, **result})
        print(json.dumps(json_safe(rows[-1]), sort_keys=True, allow_nan=False))
    return rows


def _write_outputs(rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    write_json(json_path, rows)

    fieldnames = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if not isinstance(value, (dict, list, tuple))
        }
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    summary: dict[str, dict[str, object]] = {}
    for dataset in sorted({str(row["dataset"]) for row in ok_rows}):
        dataset_rows = [row for row in ok_rows if row["dataset"] == dataset]
        if dataset_rows:
            if all(bool(row.get("external_metrics_blinded")) for row in dataset_rows):
                summary[dataset] = {
                    "external_metrics_blinded": True,
                    "n_ok_rows": len(dataset_rows),
                    "methods": sorted({str(row["method"]) for row in dataset_rows}),
                    "selected_families": sorted(
                        {
                            str(row["selected_family"])
                            for row in dataset_rows
                            if row.get("selected_family") is not None
                        }
                    ),
                }
            else:
                scored_rows = [row for row in dataset_rows if "acc" in row]
                if not scored_rows:
                    continue
                best = max(scored_rows, key=lambda row: float(row.get("acc", -1.0)))
                summary[dataset] = {
                    "best_method": str(best["method"]),
                    "best_acc": float(best.get("acc", math.nan)),
                    "best_nmi": float(best.get("nmi", math.nan)),
                    "best_ari": float(best.get("ari", math.nan)),
                }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(json_safe({"output_dir": str(output_dir), "summary": summary}), indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quick real-data benchmarks for GH-mixture VAE clustering.")
    parser.add_argument("--datasets", nargs="+", default=["digits", "wine", "breast_cancer"], choices=DATASET_NAMES)
    parser.add_argument("--data-home", type=str, default=str(ROOT / "data" / "openml"))
    parser.add_argument("--download-datasets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--blind-external-metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Scramble evaluation labels and remove ACC, ARI, NMI, and purity "
            "from all result artifacts."
        ),
    )
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--max-samples", type=int, default=700)
    parser.add_argument("--contamination", type=float, default=0.0)
    parser.add_argument(
        "--contamination-mode",
        choices=("uniform", "gaussian_noise", "salt_pepper", "semantic"),
        default="uniform",
    )
    parser.add_argument("--semantic-outlier-dataset", type=str, default=None)
    parser.add_argument("--corruption-severity", type=float, default=0.5)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[96, 64])
    parser.add_argument("--conv-channels", nargs=2, type=int, default=[32, 64])
    parser.add_argument("--pretrain-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--ae-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=8e-4)
    parser.add_argument("--ae-lr", type=float, default=1e-3)
    parser.add_argument("--vae-warmstart-ae-epochs", type=int, default=0)
    parser.add_argument("--vae-warmstart-lr", type=float, default=1e-3)
    parser.add_argument("--vae-warmstart-logvar", type=float, default=-4.0)
    parser.add_argument("--dec-epochs", type=int, default=20)
    parser.add_argument("--dec-lr", type=float, default=1e-3)
    parser.add_argument("--dec-alpha", type=float, default=1.0)
    parser.add_argument("--idec-recon-weight", type=float, default=1.0)
    parser.add_argument("--cc-epochs", type=int, default=80)
    parser.add_argument("--cc-lr", type=float, default=1e-3)
    parser.add_argument("--cc-projection-dim", type=int, default=64)
    parser.add_argument("--cc-temperature", type=float, default=0.5)
    parser.add_argument("--cc-cluster-temperature", type=float, default=1.0)
    parser.add_argument("--cc-cluster-weight", type=float, default=1.0)
    parser.add_argument("--cc-entropy-weight", type=float, default=1.0)
    parser.add_argument("--cc-noise-scale", type=float, default=0.05)
    parser.add_argument("--cc-mask-probability", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--recon-variance", type=float, default=1.0)
    parser.add_argument("--mixture-pretrain-epochs", type=int, default=40)
    parser.add_argument("--mixture-pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--mixture-initial-logvar", type=float, default=-4.0)
    parser.add_argument(
        "--mixture-affine-gauge",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--mixture-epochs", type=int, default=30)
    parser.add_argument("--mixture-cycles", type=int, default=2)
    parser.add_argument("--mixture-lr", type=float, default=8e-4)
    parser.add_argument("--mixture-prior-lr", type=float, default=3e-4)
    parser.add_argument(
        "--mixture-latent-anchor-weight",
        type=float,
        default=0.0,
        help=(
            "Penalty on posterior-mean drift from the deterministic pretraining "
            "representation; applied identically to all mixture-aware methods."
        ),
    )
    parser.add_argument(
        "--mixture-posterior-anchor-weight",
        type=float,
        default=0.0,
        help=(
            "Penalty on KL(q_pretrain(z|x) || q_current(z|x)); applied "
            "identically to all mixture-aware methods."
        ),
    )
    parser.add_argument(
        "--mixture-posterior-anchor-max-validation-kl",
        type=float,
        default=0.025,
        help=(
            "Maximum mean held-out KL from the pretrained posterior for trust "
            "methods; nonpositive disables the hard checkpoint constraint."
        ),
    )
    parser.add_argument("--mixture-mc-samples", type=int, default=3)
    parser.add_argument("--mixture-refit-mc-samples", type=int, default=3)
    parser.add_argument("--mixture-eval-mc-samples", type=int, default=32)
    parser.add_argument("--mixture-validation-mc-samples", type=int, default=3)
    parser.add_argument("--mixture-selection-fraction", type=float, default=0.1)
    parser.add_argument(
        "--mixture-neural-validation-fraction", type=float, default=0.1
    )
    parser.add_argument("--prior-acceptance-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--prior-selection-rule",
        choices=("one_standard_error", "best_score", "shape_gated", "student_t_safe"),
        default="one_standard_error",
    )
    parser.add_argument("--shape-skew-threshold", type=float, default=0.25)
    parser.add_argument("--shape-score-margin", type=float, default=0.01)
    parser.add_argument(
        "--safe-confidence-z",
        type=float,
        default=0.0,
        help="One-sided z cutoff; nonpositive derives a Bonferroni cutoff.",
    )
    parser.add_argument("--safe-familywise-alpha", type=float, default=0.05)
    parser.add_argument("--safe-min-active-fraction", type=float, default=0.8)
    parser.add_argument("--safe-assignment-tolerance", type=float, default=0.05)
    parser.add_argument("--safe-component-mass-threshold", type=float, default=0.01)
    parser.add_argument("--safe-minimum-weight", type=float, default=1e-4)
    parser.add_argument(
        "--safe-minimum-covariance-eigenvalue", type=float, default=1e-5
    )
    parser.add_argument(
        "--safe-maximum-covariance-condition", type=float, default=1e6
    )
    parser.add_argument("--safe-minimum-interior-shape", type=float, default=1e-4)
    parser.add_argument("--safe-core-confidence-threshold", type=float, default=0.8)
    parser.add_argument("--safe-minimum-core-fraction", type=float, default=0.05)
    parser.add_argument("--safe-minimum-core-retention", type=float, default=0.95)
    parser.add_argument(
        "--component-enrichment-families",
        nargs="+",
        choices=("mgst", "mvvg", "mvhb", "mnig", "mghp"),
        default=["mgst", "mghp"],
    )
    parser.add_argument("--component-minimum-effective-size", type=float, default=30.0)
    parser.add_argument(
        "--component-skew-signal-threshold", type=float, default=2.0
    )
    parser.add_argument(
        "--component-minimum-anchor-normalized-entropy", type=float, default=0.8
    )
    parser.add_argument(
        "--component-boundary-log-odds-threshold", type=float, default=2.0
    )
    parser.add_argument(
        "--component-boundary-confidence-z",
        type=float,
        default=1.6448536269514722,
    )
    parser.add_argument(
        "--component-minimum-boundary-effective-size", type=float, default=10.0
    )
    parser.add_argument(
        "--component-minimum-boundary-score-gain", type=float, default=0.0
    )
    parser.add_argument(
        "--component-minimum-boundary-leverage", type=float, default=0.01
    )
    parser.add_argument(
        "--component-minimum-decision-change-fraction", type=float, default=0.005
    )
    parser.add_argument(
        "--component-minimum-graph-assortativity-gain", type=float, default=0.0
    )
    parser.add_argument("--component-graph-neighbors", type=int, default=10)
    parser.add_argument(
        "--component-fit-graph-refinement",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--component-graph-density-one-se-multiplier",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--component-minimum-optimization-graph-gain",
        type=float,
        default=0.0,
    )
    parser.add_argument("--component-anchor-weight", type=float, default=0.0)
    parser.add_argument(
        "--component-anchor-confidence-threshold", type=float, default=0.0
    )
    parser.add_argument("--adaptive-guard-min-active-fraction", type=float, default=0.5)
    parser.add_argument("--adaptive-guard-min-assigned-fraction", type=float, default=0.5)
    parser.add_argument("--adaptive-guard-min-active", type=int, default=2)
    parser.add_argument("--adaptive-guard-min-assigned", type=int, default=2)
    parser.add_argument("--adaptive-guard-fallback-family", choices=("mvn", "mvt"), default="mvt")
    parser.add_argument("--adaptive-guard-preserve-mi", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--adaptive-families",
        nargs="+",
        choices=tuple(ADAPTIVE_FAMILY_ORDER),
        default=list(ADAPTIVE_FAMILY_ORDER),
    )
    parser.add_argument("--mixture-temperature", type=float, default=1.0)
    parser.add_argument("--mixture-mi-weight", type=float, default=0.1)
    parser.add_argument("--mixture-grad-clip", type=float, default=10.0)
    parser.add_argument("--mixture-patience", type=int, default=10)
    parser.add_argument("--diag-gmm-iter", type=int, default=40)
    parser.add_argument("--fmvmm-iter", type=int, default=4)
    parser.add_argument("--fmvmm-tol", type=float, default=1e-3)
    parser.add_argument("--fmvmm-jobs", type=int, default=1)
    parser.add_argument("--gh-mstep-nit", type=int, default=25)
    parser.add_argument("--fmvmm-path", type=str, default=os.environ.get("FMVMM_PATH"))
    parser.add_argument(
        "--fmvmm-standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fmvmm-debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fmvmm-initialization", choices=["kmeans", "gmm"], default="gmm")
    parser.add_argument("--fmvmm-fit-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "realdata_benchmark")
    parser.add_argument("--run-id", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(int(args.torch_threads))
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_id
    rows: list[dict[str, object]] = []
    for dataset_name in args.datasets:
        for seed in args.seeds:
            rows.extend(_run_one(dataset_name, seed, args, artifact_root=output_dir / "artifacts"))
    _write_outputs(rows, output_dir)


if __name__ == "__main__":
    main()
