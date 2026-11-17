"""Fixed, label-independent inputs for the Paper 2 representation study."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer


PUBLICATION_BENCHMARKS = ("reuters10k", "stl10_resnet50", "cifar10_resnet50", "cifar10020_resnet50")
BENCHMARKS = ("har561", "20newsgroups_tfidf", "isolet617", "coil20_64", *PUBLICATION_BENCHMARKS)
COIL20_URL = "https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip"
COIL20_ARCHIVE_SHA256 = "517c5594820eb40066ba0ff6842e7f09392bf7fde849bf9cb9c28445b0f29e88"
ISOLET_URL = "https://archive.ics.uci.edu/static/public/54/isolet.zip"
ISOLET_ARCHIVE_SHA256 = "fe2e0d45f1057d112e051d309070992fc178e7f6922cc7b4e8b2bd310303a053"
HAR_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip"
)
HAR_ARCHIVE_SHA256 = "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_paths(data_home: str | Path, name: str) -> tuple[Path, Path]:
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown representation benchmark: {name}")
    folder = Path(data_home) / name
    return folder / "dataset.npz", folder / "metadata.json"


def benchmark_metadata(data_home: str | Path, name: str) -> dict:
    _, metadata_path = cache_paths(data_home, name)
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def parse_har_archive(path: Path) -> dict[str, np.ndarray]:
    """Read named members without extracting arbitrary archive paths."""
    with zipfile.ZipFile(path) as outer:
        nested = [n for n in outer.namelist() if n.endswith("UCI HAR Dataset.zip")]
        if nested:
            with zipfile.ZipFile(io.BytesIO(outer.read(nested[0]))) as inner:
                return _parse_har_members(inner)
        return _parse_har_members(outer)


def _parse_har_members(archive: zipfile.ZipFile) -> dict[str, np.ndarray]:
    def read(suffix: str, dtype: type) -> np.ndarray:
        matches = [n for n in archive.namelist() if n.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"Expected one HAR member ending in {suffix!r}")
        return np.loadtxt(io.BytesIO(archive.read(matches[0])), dtype=dtype)

    features, labels, subjects, partitions = [], [], [], []
    for index, split in enumerate(("train", "test")):
        x = np.atleast_2d(read(f"/{split}/X_{split}.txt", np.float32))
        y = np.atleast_1d(read(f"/{split}/y_{split}.txt", np.int64))
        subject = np.atleast_1d(read(f"/{split}/subject_{split}.txt", np.int64))
        if len(x) != len(y) or len(x) != len(subject):
            raise ValueError("HAR feature, label, and subject lengths differ")
        features.append(x)
        labels.append(y - 1)
        subjects.append(subject)
        partitions.append(np.full(len(y), index, dtype=np.int8))
    return {
        "x": np.concatenate(features),
        "y": np.concatenate(labels),
        "subject": np.concatenate(subjects),
        "original_partition": np.concatenate(partitions),
    }


def vectorize_newsgroups(documents: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    vectorizer = TfidfVectorizer(
        max_features=2000, min_df=2, stop_words="english", norm="l2",
        sublinear_tf=True, dtype=np.float32,
    )
    sparse = vectorizer.fit_transform(documents)
    # Empty documents cannot support a text-representation comparison.
    # Exclusion depends only on the feature matrix, never on class labels.
    keep = np.asarray(sparse.getnnz(axis=1) > 0)
    return sparse[keep].toarray(), keep, vectorizer.get_feature_names_out().tolist()


def parse_isolet_archive(path: Path) -> dict[str, np.ndarray]:
    """Read the two original Unix-compressed partitions without extracting paths."""
    features, labels, partitions = [], [], []
    with zipfile.ZipFile(path) as archive:
        for index, name in enumerate(("isolet1+2+3+4.data.Z", "isolet5.data.Z")):
            raw = subprocess.run(["gzip", "-dc"], input=archive.read(name),
                                 capture_output=True, check=True).stdout
            values = np.atleast_2d(np.loadtxt(io.BytesIO(raw), delimiter=",", dtype=np.float32))
            if values.shape[1] != 618 or not np.isfinite(values).all():
                raise ValueError("ISOLET requires 617 finite features and one label")
            y = values[:, -1]
            if not np.equal(y, np.floor(y)).all() or np.any((y < 1) | (y > 26)):
                raise ValueError("Invalid ISOLET class labels")
            features.append(values[:, :-1])
            labels.append(y.astype(np.int64) - 1)
            partitions.append(np.full(len(y), index, dtype=np.int8))
    return {"x": np.concatenate(features), "y": np.concatenate(labels),
            "original_partition": np.concatenate(partitions)}


def parse_coil20_archive(path: Path) -> dict[str, np.ndarray]:
    """Retain every object and view in a fixed order; never extract ZIP paths."""
    from PIL import Image

    names = [f"coil-20-proc/obj{obj}__{view}.png"
             for obj in range(1, 21) for view in range(72)]
    images = []
    with zipfile.ZipFile(path) as archive:
        actual = [name for name in archive.namelist() if name.lower().endswith(".png")]
        if len(actual) != len(names) or set(actual) != set(names):
            raise ValueError("COIL-20 requires exactly 20 objects with 72 views each")
        for name in names:
            with Image.open(io.BytesIO(archive.read(name))) as image:
                if image.size != (128, 128) or image.mode != "L":
                    raise ValueError(f"Unexpected COIL-20 image format: {name}")
                resized = image.resize((64, 64), Image.Resampling.BOX)
                images.append(np.asarray(resized, dtype=np.float32).reshape(-1) / 255.0)
    return {"x": np.stack(images), "y": np.repeat(np.arange(20), 72),
            "view_degrees": np.tile(np.arange(72) * 5, 20)}


def prepare_benchmark(data_home: str | Path, name: str) -> dict:
    if name in PUBLICATION_BENCHMARKS:
        raise ValueError("Use scripts.prepare_paper2_four_benchmarks for these pinned inputs")
    target, metadata_path = cache_paths(data_home, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or metadata_path.exists():
        _, _, metadata = load_benchmark(data_home, name)
        return metadata
    if name == "har561":
        archive_path = target.parent / "source.zip"
        if not archive_path.exists():
            temporary = archive_path.with_suffix(".download")
            with urllib.request.urlopen(HAR_URL, timeout=120) as source, temporary.open("wb") as stream:
                shutil.copyfileobj(source, stream)
            os.replace(temporary, archive_path)
        if file_sha256(archive_path) != HAR_ARCHIVE_SHA256:
            raise ValueError("HAR source archive differs from the verified 2026-10-20 download")
        arrays = parse_har_archive(archive_path)
        if arrays["x"].shape != (10299, 561):
            raise ValueError(f"Unexpected HAR shape: {arrays['x'].shape}")
        metadata = {
            "dataset_source": "uci_har_561_train_test",
            "dataset_url": "https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones",
            "dataset_doi": "10.24432/C54S4K",
            "dataset_license": "CC BY 4.0",
            "source_archive_sha256": file_sha256(archive_path),
            "n_classes": 6,
            "n_subjects": int(np.unique(arrays["subject"]).size),
            "preprocessing": "All 561 supplied features; featurewise standardization in runner.",
            "evaluation_scope": "Transductive clustering; overlapping windows; not unseen-subject prediction.",
        }
    elif name == "isolet617":
        archive_path = target.parent / "source.zip"
        if not archive_path.exists():
            temporary = archive_path.with_suffix(".download")
            with urllib.request.urlopen(ISOLET_URL, timeout=120) as source, temporary.open("wb") as stream:
                shutil.copyfileobj(source, stream)
            os.replace(temporary, archive_path)
        if file_sha256(archive_path) != ISOLET_ARCHIVE_SHA256:
            raise ValueError("ISOLET source differs from the verified 2026-10-23 download")
        arrays = parse_isolet_archive(archive_path)
        if arrays["x"].shape != (7797, 617):
            raise ValueError(f"Unexpected ISOLET shape: {arrays['x'].shape}")
        metadata = {
            "dataset_source": "uci_isolet_all26_train_test",
            "dataset_url": "https://archive.ics.uci.edu/dataset/54/isolet",
            "dataset_doi": "10.24432/C51G69", "dataset_license": "CC BY 4.0",
            "source_archive_sha256": file_sha256(archive_path), "n_classes": 26,
            "n_subjects": 150,
            "preprocessing": "All 617 supplied acoustic features; featurewise standardization in runner; no class filtering.",
            "evaluation_scope": "Transductive clustering of all letters; repeated speakers; not unseen-speaker prediction.",
        }
    elif name == "coil20_64":
        archive_path = target.parent / "source.zip"
        if not archive_path.exists():
            temporary = archive_path.with_suffix(".download")
            with urllib.request.urlopen(COIL20_URL, timeout=120) as source, temporary.open("wb") as stream:
                shutil.copyfileobj(source, stream)
            os.replace(temporary, archive_path)
        if file_sha256(archive_path) != COIL20_ARCHIVE_SHA256:
            raise ValueError("COIL-20 source differs from the verified 2026-10-23 download")
        arrays = parse_coil20_archive(archive_path)
        metadata = {
            "dataset_source": "columbia_coil20_processed_all_objects_64",
            "dataset_url": "https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php",
            "source_archive_sha256": file_sha256(archive_path), "n_classes": 20,
            "preprocessing": "All 1440 grayscale images; fixed 2x2 BOX downsampling to 64x64; pixels divided by 255; featurewise standardization in runner; no PCA or pretrained features.",
            "evaluation_scope": "Transductive object-identity clustering; correlated views of 20 objects; not unseen-object generalization.",
            "redistribution": "Obtain images from Columbia; do not redistribute the source archive in the code release.",
        }
    else:
        bunch = fetch_20newsgroups(
            data_home=str(Path(data_home) / "sklearn"),
            subset="all", shuffle=False, categories=None,
            remove=("headers", "footers", "quotes"),
        )
        x, keep, vocabulary = vectorize_newsgroups(bunch.data)
        arrays = {
            "x": x,
            "y": np.asarray(bunch.target, dtype=np.int64)[keep],
            "source_indices": np.flatnonzero(keep),
        }
        corpus = hashlib.sha256()
        for document in bunch.data:
            encoded = document.encode("utf-8")
            corpus.update(len(encoded).to_bytes(8, "big"))
            corpus.update(encoded)
        metadata = {
            "dataset_source": "20newsgroups_all20_tfidf2000_no_svd",
            "dataset_url": "https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset",
            "n_classes": 20,
            "n_source_documents": len(bunch.data),
            "n_empty_documents_removed": int((~keep).sum()),
            "stripped_corpus_sha256": corpus.hexdigest(),
            "vocabulary": vocabulary,
            "preprocessing": "All 20 topics; metadata removed; sublinear TF-IDF; 2000 terms; min_df=2; L2 rows; global RMS scaling in runner; no SVD.",
            "evaluation_scope": "Transductive clustering; vocabulary fitted without labels on all documents.",
        }
    if not np.isfinite(arrays["x"]).all():
        raise ValueError("Non-finite benchmark features")
    if np.unique(arrays["y"]).size != metadata["n_classes"]:
        raise ValueError("Unexpected benchmark class count")
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, target)
    metadata.update({
        "schema_version": 1,
        "dataset": name,
        "n_samples": len(arrays["x"]),
        "input_dim": arrays["x"].shape[1],
        "dataset_cache_sha256": file_sha256(target),
    })
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def load_benchmark(data_home: str | Path, name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    target, metadata_path = cache_paths(data_home, name)
    if not target.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing {name} cache. Run scripts/prepare_paper2_representation_data.py first."
        )
    metadata = benchmark_metadata(data_home, name)
    if file_sha256(target) != metadata["dataset_cache_sha256"]:
        raise ValueError(f"Cache checksum mismatch for {name}")
    with np.load(target, allow_pickle=False) as saved:
        x = saved["x"].astype(np.float32)
        y = saved["y"].astype(np.int64)
    if x.shape != (metadata["n_samples"], metadata["input_dim"]) or len(y) != len(x):
        raise ValueError(f"Cache shape mismatch for {name}")
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite features for {name}")
    return x, y, metadata
