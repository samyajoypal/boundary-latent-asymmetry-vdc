"""Inspect or execute one recorded configuration without a cluster-specific launcher."""
import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"


def verify_source():
    for relative, item in json.loads((ROOT / "source_manifest.json").read_text()).items():
        if hashlib.sha256((CODE / relative).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Scientific source changed: {relative}")


@lru_cache(maxsize=1)
def recorded_runs():
    return json.loads((ROOT / "protocols/runs.json").read_text())


def command(args):
    matches = [c for c in recorded_runs() if all(c[k] == getattr(args, k) for k in ("study", "design", "seed", "regime", "method"))]
    if len(matches) != 1:
        raise ValueError("Choose exactly one recorded study/design/seed/regime/method configuration")
    c = matches[0]
    options = dict(c["arguments"])
    output = args.output.resolve()
    options["fmvmm_path"] = str(CODE / "vendor/fmvmm_tstable2_20261102")
    options["mixture_pretrain_checkpoint"] = str(output / "pretraining" / f"{args.study}__{args.design}__seed{args.seed}.pt")
    if args.study == "image":
        options["data_home"] = str(args.data_home.resolve())
    runner = "run_simulation_benchmark.py" if args.study == "simulation" else "run_realdata_benchmark.py"
    run_id = f"{args.study}__{args.design}__{args.regime}__{args.method}__seed{args.seed}"
    result = [sys.executable, str(CODE / "scripts" / runner), "--output-dir", str(output), "--run-id", run_id]
    for key, value in sorted(options.items()):
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            result.append(flag if value else "--no-" + key.replace("_", "-"))
        elif isinstance(value, list):
            result.extend([flag, *map(str, value)])
        else:
            result.extend([flag, str(value)])
    return result, output / run_id


def turtle(args):
    if args.study != "image" or args.regime != "joint_training" or args.seed not in range(16200, 16220):
        raise ValueError("TURTLE is a joint-training image comparator on the recorded twenty seeds")
    study = json.loads((ROOT / "protocols/studies.json").read_text())
    datasets = study["data_hashes"]
    if args.design not in datasets:
        raise ValueError("Unknown image dataset")
    if not args.execute:
        print("Will run the pinned single-space TURTLE comparator for", args.design, args.seed)
        return
    for relative, expected in study["turtle_source"].items():
        path = CODE / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Obtain the pinned TURTLE source described in current/README.md")
    os.chdir(CODE)
    sys.path[:0] = [str(CODE), str(CODE / "src")]
    import numpy as np
    import torch
    from ghmvae.data import unlabeled_optimization_selection_split
    from scripts.run_realdata_benchmark import _load_dataset, _metrics
    from scripts.paper2_turtle_comparator import fit_turtle
    from scripts.paper2_fit_diagnostics import write_json
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    x, y, mask, meta = _load_dataset(args.design, 60000, args.seed, data_home=str(args.data_home))
    _, _, indices, selection = unlabeled_optimization_selection_split(x, selection_fraction=.2, seed=args.seed + 2099)
    destination = args.output / f"image__{args.design}__joint_training__turtle_1space__seed{args.seed}"
    probabilities, info = fit_turtle(x.numpy(), indices.numpy(), meta["n_classes"], args.seed, destination)
    np.savez_compressed(destination / "embeddings.npz", responsibilities=probabilities, labels=y,
                        eval_mask=mask, optimization_indices=indices.numpy(), selection_indices=selection.numpy())
    write_json(destination / "results.json", dict(status="ok", dataset=args.design, seed=args.seed,
        method="turtle_1space", **_metrics(y[mask], probabilities[mask].argmax(1)), **info))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", choices=("simulation", "image"), required=True)
    p.add_argument("--design", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--regime", choices=("fixed_representation", "joint_training"), default="joint_training")
    p.add_argument("--method", default="ghvade_adaptive")
    p.add_argument("--data-home", type=Path, default=ROOT / "data/features")
    p.add_argument("--output", type=Path, default=ROOT / "outputs")
    p.add_argument("--execute", action="store_true", help="Actually run; otherwise print the command only")
    args = p.parse_args()
    args.data_home, args.output = args.data_home.resolve(), args.output.resolve()
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      NUMEXPR_NUM_THREADS="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    verify_source()
    if args.method == "turtle_1space":
        return turtle(args)
    cmd, destination = command(args)
    print(shlex.join(cmd), flush=True)
    if not args.execute:
        return
    if destination.exists():
        raise FileExistsError(f"Retain existing outputs; choose a new --output root: {destination}")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
               PYTHONPATH=os.pathsep.join([str(CODE / "src"), str(CODE)]))
    subprocess.run(cmd, cwd=CODE, env=env, check=True)
    rows = json.loads((destination / "results.json").read_text())
    if len(rows) != 1 or rows[0]["status"] != "ok":
        raise RuntimeError("The runner completed but the scientific result did not succeed")


if __name__ == "__main__":
    main()
