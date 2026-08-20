from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        dataset = row.get("dataset", "")
        method = row.get("method", "")
        groups[(dataset, method)].append(row)

    summary: list[dict[str, object]] = []
    for (dataset, method), group_rows in sorted(groups.items()):
        item: dict[str, object] = {
            "dataset": dataset,
            "method": method,
            "n_runs": len(group_rows),
        }
        for metric in ("acc", "nmi", "ari", "purity"):
            values = [v for row in group_rows if (v := _float(row, metric)) is not None]
            if values:
                item[f"{metric}_mean"] = mean(values)
                item[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        summary.append(item)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize benchmark CSV files by dataset and method.")
    parser.add_argument("csv_files", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for csv_file in args.csv_files:
        rows.extend(_read_csv(csv_file))
    summary = summarize(rows)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
