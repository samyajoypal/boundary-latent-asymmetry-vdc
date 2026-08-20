from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_realdata_benchmark import DATASET_NAMES, _load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and verify cached benchmark datasets.")
    parser.add_argument("--datasets", nargs="+", required=True, choices=DATASET_NAMES)
    parser.add_argument("--data-home", type=str, default="data/torchvision")
    parser.add_argument("--max-samples", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.data_home).mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        x, y, eval_mask, meta = _load_dataset(
            dataset,
            max_samples=args.max_samples,
            seed=0,
            contamination=0.0,
            data_home=args.data_home,
            download_datasets=True,
        )
        print(
            json.dumps(
                {
                    "dataset": dataset,
                    "x_shape": list(x.shape),
                    "y_shape": list(y.shape),
                    "n_eval": int(eval_mask.sum()),
                    "meta": meta,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
