from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_simulation_results import _paired_delta, _write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute paired seed-level ACC comparisons.")
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--treatments", nargs="+", required=True)
    parser.add_argument("--controls", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.csv_file.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    datasets = sorted({row.get("dataset", "") for row in ok_rows})
    comparisons = [
        _paired_delta(ok_rows, dataset, treatment, control)
        for dataset in datasets
        for treatment in args.treatments
        for control in args.controls
    ]
    _write_csv(args.output, comparisons)
    print(json.dumps(comparisons, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
