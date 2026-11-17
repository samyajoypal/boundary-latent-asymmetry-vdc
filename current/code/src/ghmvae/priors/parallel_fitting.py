"""Process-isolated CPU candidate fits; density scoring stays in the parent."""
from __future__ import annotations

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from ghmvae.priors.fmvmm_adapter import FMVMMPriorFitter


def _attempt(fitter, points, worker=False):
    if worker:
        torch.set_num_threads(1)
    start = time.perf_counter()
    with threadpool_limits(limits=1) if worker else nullcontext():
        try:
            result, error = fitter.fit(points), None
        except Exception as exc:
            result, error = None, repr(exc)
    return result, error, time.perf_counter() - start, np.random.get_state()


def candidate_attempts(fitters, points, workers=1):
    """Yield in catalogue order, never completion order, with seeded isolation.

    Worker failures propagate instead of silently removing a candidate. Ordinary
    fitter exceptions retain the existing selector failure-recording behavior.
    """
    if not isinstance(workers, int) or workers < 1 or workers > 8:
        raise ValueError("candidate_workers must be an integer between one and eight")
    if workers == 1:
        for fitter in fitters:
            yield _attempt(fitter, points)
        return
    if not all(isinstance(f, FMVMMPriorFitter) and f.random_seed is not None
               and f.n_jobs == 1 and f.require_differentiable for f in fitters):
        raise ValueError("Parallel fits require explicitly seeded, differentiable FMVMM fitters with n_jobs=1")
    points = points.detach().float().cpu().contiguous().share_memory_()
    with ProcessPoolExecutor(max_workers=min(workers, len(fitters)),
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(_attempt, fitter, points, True) for fitter in fitters]
        for future in futures:
            outcome = future.result()
            # Match the parent's NumPy state after the corresponding serial fit.
            np.random.set_state(outcome[3])
            yield outcome
