from __future__ import annotations

import contextlib
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch import Tensor

from ghmvae.priors.base import Prior, PriorFitResult, PriorFitter


@contextlib.contextmanager
def _fit_timeout(seconds: float | None) -> Iterator[None]:
    if seconds is None or seconds <= 0.0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _raise_timeout(signum: int, frame: Any) -> None:
        raise TimeoutError(f"FMVMM candidate fit exceeded {seconds:.1f} seconds.")

    old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def _logsumexp_numpy(a: Any, axis: int = 1) -> Any:
    import numpy as np

    max_a = np.max(a, axis=axis, keepdims=True)
    return (max_a + np.log(np.sum(np.exp(a - max_a), axis=axis, keepdims=True))).squeeze(axis)


class FMVMMNumpyPrior(Prior):
    """Adapter around an FMVMM fitted mixture.

    This object is useful for model selection, latent scoring, and
    responsibilities. It is not differentiable with respect to torch tensors;
    a torch-native GH log-density can be added later behind the same Prior API.
    """

    name = "fmvmm_numpy"

    def __init__(self, weights: Any, alphas: Sequence[Any], dist_comb: Sequence[Any], model: Any | None = None) -> None:
        super().__init__()
        import numpy as np

        self.weights = np.asarray(weights, dtype=float)
        self.weights = self.weights / self.weights.sum()
        self.alphas = list(alphas)
        self.dist_comb = tuple(dist_comb)
        self.model = model

    def log_prob(self, z: Tensor) -> Tensor:
        raise NotImplementedError(
            "FMVMMNumpyPrior cannot be used inside a differentiable torch ELBO yet. "
            "Use log_prob_numpy for scoring, or implement a torch-native GH-family prior."
        )

    def log_prob_numpy(self, z: Any) -> Any:
        import numpy as np

        z = np.asarray(z, dtype=float)
        comp = []
        for weight, alpha, dist in zip(self.weights, self.alphas, self.dist_comb):
            log_pdf = np.asarray(dist.logpdf(z, *alpha), dtype=float)
            comp.append(np.log(weight) + np.where(np.isfinite(log_pdf), log_pdf, -1e300))
        return _logsumexp_numpy(np.column_stack(comp), axis=1)

    def responsibilities_numpy(self, z: Any) -> Any:
        import numpy as np

        z = np.asarray(z, dtype=float)
        comp = []
        for weight, alpha, dist in zip(self.weights, self.alphas, self.dist_comb):
            log_pdf = np.asarray(dist.logpdf(z, *alpha), dtype=float)
            comp.append(np.log(weight) + np.where(np.isfinite(log_pdf), log_pdf, -1e300))
        log_joint = np.column_stack(comp)
        log_norm = _logsumexp_numpy(log_joint, axis=1)[:, None]
        return np.exp(log_joint - log_norm)

    def predict_numpy(self, z: Any) -> Any:
        return self.responsibilities_numpy(z).argmax(axis=1)


@dataclass
class FMVMMPriorFitter(PriorFitter):
    n_components: int
    fmvmm_path: str | None = None
    list_of_dist: Sequence[str] = ("mghp",)
    candidate_combinations: Sequence[Sequence[str]] | None = None
    max_iter: int = 25
    tol: float = 1e-4
    em_type: str = "soft"
    initialization: str = "kmeans"
    n_jobs: int = 1
    gh_mstep_kwargs: Mapping[str, Any] | None = None
    init_fit_kwargs: Mapping[str, Any] | None = None
    verbose: bool = False
    debug: bool = False
    differentiable: bool = True
    require_differentiable: bool = False
    standardize: bool = False
    retry_failed_fit: bool = True
    fallback_winsor_quantile: float = 0.005
    random_seed: int | None = None
    fit_timeout_seconds: float | None = None

    def fit(self, z: Tensor) -> PriorFitResult:
        import numpy as np

        fit_start = time.perf_counter()

        if self.fmvmm_path is not None:
            path = str(Path(self.fmvmm_path).expanduser())
            if path not in sys.path:
                sys.path.insert(0, path)

        from fmvmm.mixtures.FMVMM import fmvmm

        if self.initialization == "gmm_multistart" and getattr(fmvmm, "gmm_multistart_version", None) != 1:
            raise RuntimeError("gmm_multistart requires the audited multistart vendor patch")

        z_fit = z.detach().float().cpu()
        center = z_fit.mean(dim=0)
        scale = z_fit.std(dim=0, unbiased=False).clamp_min(1e-6)
        if self.standardize:
            z_fit = (z_fit - center) / scale
        z_np = z_fit.numpy().astype(float)
        if self.random_seed is not None:
            np.random.seed(int(self.random_seed))
        def make_model(initialization: str) -> Any:
            return fmvmm(
                n_clusters=int(self.n_components),
                list_of_dist=list(self.list_of_dist),
                candidate_combinations=self.candidate_combinations,
                max_iter=int(self.max_iter),
                tol=float(self.tol),
                em_type=self.em_type,
                initialization=initialization,
                n_jobs=int(self.n_jobs),
                gh_mstep_kwargs=dict(self.gh_mstep_kwargs or {}),
                init_fit_kwargs=dict(self.init_fit_kwargs or {}),
                verbose=bool(self.verbose),
                debug=bool(self.debug),
            )

        attempts: list[tuple[str, str, Any]] = [("primary", self.initialization, z_np)]
        if self.retry_failed_fit and self.initialization != "gmm_multistart":
            alternate = "kmeans" if self.initialization == "gmm" else "gmm"
            attempts.append(("alternate_initialization", alternate, z_np))
            quantile = min(max(float(self.fallback_winsor_quantile), 0.0), 0.1)
            if quantile > 0.0:
                lower = np.quantile(z_np, quantile, axis=0)
                upper = np.quantile(z_np, 1.0 - quantile, axis=0)
                clipped = np.clip(z_np, lower, upper)
                attempts.append(("winsorized_primary", self.initialization, clipped))
                attempts.append(("winsorized_alternate", alternate, clipped))

        model = None
        fit_attempt = "failed"
        fit_initialization = self.initialization
        failed_attempts: list[dict[str, str]] = []
        for attempt_name, initialization, attempt_data in attempts:
            candidate_model = make_model(initialization)
            try:
                with _fit_timeout(self.fit_timeout_seconds):
                    candidate_model.fit(attempt_data)
            except Exception as exc:
                failed_attempts.append(
                    {
                        "attempt": attempt_name,
                        "initialization": initialization,
                        "error": repr(exc),
                    }
                )
                continue
            if candidate_model.worked_dist:
                model = candidate_model
                fit_attempt = attempt_name
                fit_initialization = initialization
                break
            failed_attempts.append(
                {
                    "attempt": attempt_name,
                    "initialization": initialization,
                    "error": "no_worked_distribution",
                }
            )
        if model is None:
            raise RuntimeError(f"FMVMM did not fit any candidate mixture: {failed_attempts}")

        best_idx = int(np.argmin(model.list_bic))
        numpy_prior = FMVMMNumpyPrior(
            weights=model.list_pi[best_idx],
            alphas=model.list_alpha[best_idx],
            dist_comb=model.worked_dist[best_idx],
            model=model,
        )
        prior: Prior = numpy_prior
        differentiable = False
        if self.differentiable:
            try:
                from ghmvae.priors.torch_fmvmm import torch_prior_from_fmvmm

                prior = torch_prior_from_fmvmm(
                    weights=model.list_pi[best_idx],
                    alphas=model.list_alpha[best_idx],
                    dist_comb=model.worked_dist[best_idx],
                )
                if self.standardize:
                    from ghmvae.priors.transformed import AffineTransformedPrior

                    prior = AffineTransformedPrior(prior, center=center, scale=scale)
                differentiable = True
            except NotImplementedError:
                if self.require_differentiable:
                    raise
                prior = numpy_prior
        diagnostics = {
            "best_idx": best_idx,
            "best_bic": float(model.list_bic[best_idx]),
            "best_mixture": [dist.__name__ for dist in model.worked_dist[best_idx]],
            "n_worked": len(model.worked_dist),
            "n_failed": len(model.not_worked_dist),
            "differentiable": differentiable,
            "standardized": bool(self.standardize),
            "initialization": fit_initialization,
            "fit_attempt": fit_attempt,
            "failed_attempts": failed_attempts,
            "fit_seconds": float(time.perf_counter() - fit_start),
            "fallback_winsor_quantile": (
                float(self.fallback_winsor_quantile) if fit_attempt.startswith("winsorized") else 0.0
            ),
        }
        starts = getattr(model, "list_start_diagnostics", None)
        if starts is not None:
            diagnostics["initialization_diagnostics"] = starts[best_idx]
        return PriorFitResult(prior=prior, diagnostics=diagnostics)
