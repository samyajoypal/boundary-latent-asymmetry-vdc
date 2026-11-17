"""Frozen-checkpoint diagnostics; never use labels for fitting or selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from ghmvae.models.vae import MLPVAE
from ghmvae.priors.torch_fmvmm import (
    GeneralizedHyperbolicComponent,
    MultivariateNormalComponent,
    MultivariateTComponent,
    SkewNormalComponent,
    TorchMixturePrior,
)
from ghmvae.priors.transformed import AffineTransformedPrior


SOURCE = Path("results/paper2_representation_density_repair")
OUTPUT = Path("results/paper2_fixed_diagnostic_20261022")
DATASETS = ("har561", "20newsgroups_tfidf")
SEEDS = (12000, 12001, 12002)
METHODS = ("ghvade_mvt", "ghvade_mvn", "ghvade_adaptive")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, torch.Tensor):
            return item.detach().cpu().tolist()
        raise TypeError(type(item).__name__)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=convert) + "\n")


def artifact(dataset, method, seed):
    return (SOURCE / f"pilot__{dataset}__{method}__seed{seed}" /
            "artifacts" / dataset / f"seed_{seed}" / method)


def aggregate_posterior_moments(mu, logvar):
    """Exact marginal moments of the finite aggregate diagonal-Gaussian q."""
    mu = np.asarray(mu, dtype=np.float64)
    variance = np.exp(np.asarray(logvar, dtype=np.float64))
    centered = mu - mu.mean(axis=0)
    between = np.mean(centered**2, axis=0)
    within = variance.mean(axis=0)
    third = np.mean(centered**3, axis=0)
    third_noise = 3 * np.mean(centered * variance, axis=0)
    mean_skew = third / np.maximum(between, 1e-20)**1.5
    q_skew = (third + third_noise) / np.maximum(between + within, 1e-20)**1.5
    return {
        "mean_coordinate_skewness": mean_skew,
        "aggregate_q_coordinate_skewness": q_skew,
        "mean_abs_mean_skewness": float(np.abs(mean_skew).mean()),
        "mean_abs_aggregate_q_skewness": float(np.abs(q_skew).mean()),
        "mean_posterior_variance_fraction": float(np.mean(within / (between + within))),
        "between_variance": between,
        "within_variance": within,
        "third_central_moment": third + third_noise,
    }


def restore_prior(checkpoint):
    """Rebuild supported saved priors, then strictly restore every tensor."""
    state = checkpoint["prior_state_dict"]
    affine = checkpoint["prior_class"] == "AffineTransformedPrior"
    if not affine and checkpoint["prior_class"] != "TorchMixturePrior":
        raise ValueError("Unsupported checkpoint prior class")
    prefix = "base_prior." if affine else ""
    components = []
    for index, record in enumerate(checkpoint["prior_parameter_summary"]["components"]):
        key = f"{prefix}components.{index}."
        mu, sigma = state[key + "mu"], state[key + "sigma"]
        family = record["family"]
        if family == "mvn":
            component = MultivariateNormalComponent(mu, sigma)
        elif family == "mvt":
            component = MultivariateTComponent(mu, sigma, record["df"])
        elif family == "mvsn":
            component = SkewNormalComponent(mu, sigma, torch.zeros_like(mu))
        elif family in {"mghp", "mgst", "mnig", "mvhb", "mvvg"}:
            component = GeneralizedHyperbolicComponent(
                record["lmbda"], record["chi"], record["psi"], mu, sigma,
                state[key + "gamma"], family=family,
            )
        else:
            raise ValueError(f"Unsupported saved family: {family}")
        components.append(component)
    prior = TorchMixturePrior(state[prefix + "log_weights"], components)
    if affine:
        prior = AffineTransformedPrior(prior, state["center"], state["scale"])
    prior.load_state_dict(state, strict=True)
    return prior.eval()


@torch.no_grad()
def decoder_metrics(model, prior, x, mu, logvar, seed, draws=32, batch_size=128):
    """Per-observation Gaussian observation-model diagnostics at variance one.

    Draws are shared across methods. MSE of E[decoder(Z)] and expected MSE
    are different estimands; both are retained, with the MC variance identity.
    """
    if draws < 2 or draws % 2:
        raise ValueError("Use a positive even draw count of at least two")
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    records = {}
    for start in range(0, len(x), batch_size):
        xb = x[start:start + batch_size].float()
        m = mu[start:start + batch_size].float()
        lv = logvar[start:start + batch_size].float()
        eps = torch.randn((draws, len(xb), m.shape[1]), generator=generator)
        z = m.unsqueeze(0) + torch.exp(0.5 * lv).unsqueeze(0) * eps
        decoded = model.decode(z.reshape(-1, m.shape[1])).reshape(draws, len(xb), -1)
        point = model.decode(m)
        mean = decoded.mean(dim=0)
        squared = (decoded - xb.unsqueeze(0)).square().mean(dim=2)
        logp = prior.log_prob(z.reshape(-1, m.shape[1])).reshape(draws, len(xb))
        entropy = 0.5 * (1 + math.log(2 * math.pi) + lv).sum(dim=1)
        recon_lp = -0.5 * xb.shape[1] * (math.log(2 * math.pi) + squared)
        elbo_draw = recon_lp + logp + entropy.unsqueeze(0)
        cosine_den = torch.linalg.vector_norm(mean, dim=1) * torch.linalg.vector_norm(xb, dim=1)
        metrics = {
            "mean_latent_decoder_mse": (point - xb).square().mean(dim=1),
            "posterior_mean_decoder_mse": (mean - xb).square().mean(dim=1),
            "posterior_mean_decoder_mae": (mean - xb).abs().mean(dim=1),
            "expected_decoder_mse": squared.mean(dim=0),
            "decoder_variance": (decoded - mean.unsqueeze(0)).square().mean(dim=(0, 2)),
            "cosine_similarity": (mean * xb).sum(dim=1) / cosine_den.clamp_min(1e-20),
            "reconstruction_log_prob": recon_lp.mean(dim=0),
            "latent_log_prior": logp.mean(dim=0),
            "posterior_entropy": entropy,
            "estimated_kl": -logp.mean(dim=0) - entropy,
            "elbo": elbo_draw.mean(dim=0),
            "elbo_per_dimension": elbo_draw.mean(dim=0) / xb.shape[1],
            "elbo_mc_half1": elbo_draw[:draws // 2].mean(dim=0),
            "elbo_mc_half2": elbo_draw[draws // 2:].mean(dim=0),
        }
        for name, value in metrics.items():
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite metric: {name}")
            records.setdefault(name, []).append(value.double().numpy())
    return {name: np.concatenate(parts) for name, parts in records.items()}


def run_reconstruction(dataset, seed, output=OUTPUT):
    from scripts.run_realdata_benchmark import _load_dataset
    torch.set_num_threads(4)
    x, labels, mask, data_meta = _load_dataset(
        dataset, 20000, seed, data_home="data/paper2_representation", download_datasets=False,
    )
    folder = Path(output) / f"decoder__{dataset}__seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": dataset, "seed": seed, "mc_draws": 32,
        "data_cache_sha256": sha256(Path("data/paper2_representation") / dataset / "dataset.npz"),
        "observation_model": "Gaussian with fixed variance 1; beta=1",
        "scope": "transductive diagnostic; selection split was used for early stopping",
        "data_metadata": data_meta, "methods": {},
    }
    reference_split = None
    reference_pretrain = None
    per_method = {}
    for method in METHODS:
        source = artifact(dataset, method, seed)
        saved = np.load(source / "embeddings.npz")
        if not np.array_equal(labels, saved["labels"]) or not np.array_equal(mask, saved["eval_mask"]):
            raise ValueError("Dataset order or labels do not match saved checkpoint")
        splits = {key: saved[key] for key in ("optimization_indices", "selection_indices")}
        if reference_split is None:
            reference_split = splits
            reference_pretrain = saved["pretrain_mu"].copy()
        elif any(not np.array_equal(splits[key], reference_split[key]) for key in splits):
            raise ValueError("Paired split mismatch")
        checkpoint = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=False)
        if checkpoint["model_class"] != "MLPVAE":
            raise ValueError("Expected saved MLPVAE")
        cfg = checkpoint["model_config"]
        model = MLPVAE(cfg["input_dim"], cfg["latent_dim"], cfg["hidden_dims"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        prior = restore_prior(checkpoint)
        mu, lv = torch.from_numpy(saved["mu"]), torch.from_numpy(saved["logvar"])
        metrics = decoder_metrics(model, prior, x, mu, lv, seed + 99103)
        metrics["constant_mean_mse"] = (x - x[splits["optimization_indices"]].mean(0)).square().mean(1).numpy()
        np.savez_compressed(folder / f"{method}.npz", **metrics, **splits)
        entry = {
            "checkpoint_sha256": sha256(source / "checkpoint.pt"),
            "embeddings_sha256": sha256(source / "embeddings.npz"),
            "selected_family": checkpoint["selected_family"],
            "pretrain_max_abs_difference_from_student": float(np.max(np.abs(saved["pretrain_mu"] - reference_pretrain))),
            "moments": {}, "metrics": {},
        }
        for split, indices in splits.items():
            entry["metrics"][split] = {key: float(values[indices].mean()) for key, values in metrics.items()}
            entry["moments"][split] = {
                stage: aggregate_posterior_moments(saved[mp][indices], saved[vp][indices])
                for stage, mp, vp in (("pretrain", "pretrain_mu", "pretrain_logvar"), ("final", "mu", "logvar"))
            }
        report["methods"][method] = entry
        per_method[method] = metrics
        print(dataset, seed, method, entry["metrics"]["selection_indices"], flush=True)
    report["adaptive_minus_student"] = {
        split: {key: float((per_method["ghvade_adaptive"][key][indices] -
                           per_method["ghvade_mvt"][key][indices]).mean())
                for key in per_method["ghvade_mvt"]}
        for split, indices in reference_split.items()
    }
    write_json(folder / "results.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize_reconstruction()
    elif args.dataset is not None and args.seed is not None:
        run_reconstruction(args.dataset, args.seed)
    else:
        parser.error("Specify --summarize or both --dataset and --seed")


def summarize_reconstruction(output=OUTPUT):
    report = {"scope": "Three development seeds; transductive evaluation, not untouched test data",
              "datasets": {}}
    for dataset in DATASETS:
        rows = [json.loads((Path(output) / f"decoder__{dataset}__seed{seed}" / "results.json").read_text()) for seed in SEEDS]
        result = {"splits": {}}
        for split in ("optimization_indices", "selection_indices"):
            methods = {}
            for method in METHODS:
                methods[method] = {}
                for metric in rows[0]["methods"][method]["metrics"][split]:
                    values = np.array([row["methods"][method]["metrics"][split][metric] for row in rows])
                    methods[method][metric] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1)), "per_seed": values}
            contrasts = {}
            for metric in rows[0]["adaptive_minus_student"][split]:
                values = np.array([row["adaptive_minus_student"][split][metric] for row in rows])
                contrasts[metric] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1)), "per_seed": values,
                                     "positive_seeds": int((values > 0).sum()), "negative_seeds": int((values < 0).sum())}
            result["splits"][split] = {"absolute": methods, "adaptive_minus_student": contrasts}
        result["adaptive_selected_families"] = [row["methods"]["ghvade_adaptive"]["selected_family"] for row in rows]
        result["pretrain_mismatch_seeds"] = [row["seed"] for row in rows if row["methods"]["ghvade_adaptive"]["pretrain_max_abs_difference_from_student"] > 0]
        report["datasets"][dataset] = result
    write_json(Path(output) / "reconstruction_summary.json", report)
    for dataset, result in report["datasets"].items():
        print(dataset, "adaptive selected", result["adaptive_selected_families"], "mismatched pretrain seeds", result["pretrain_mismatch_seeds"])
        selected = result["splits"]["selection_indices"]
        for metric in ("posterior_mean_decoder_mse", "expected_decoder_mse", "posterior_mean_decoder_mae", "cosine_similarity", "elbo", "estimated_kl"):
            print(metric, {method: selected["absolute"][method][metric]["mean"] for method in METHODS},
                  "delta", selected["adaptive_minus_student"][metric])
    return report


if __name__ == "__main__":
    main()
