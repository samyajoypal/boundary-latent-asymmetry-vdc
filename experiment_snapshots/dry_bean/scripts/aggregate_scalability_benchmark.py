from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate prior scalability tasks.")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results" / "scalability_v9")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "scalability_v9_summary.csv")
    parser.add_argument("--expected", type=int, default=63)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.input_dir.glob("task_*.json"))]
    if len(rows) != args.expected and not args.allow_incomplete:
        raise SystemExit(f"Expected {args.expected} task files, found {len(rows)}.")
    groups: dict[tuple[str, str, int, int, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["cell_id"]),
            str(row["method"]),
            int(row["n_samples"]),
            int(row["latent_dim"]),
            int(row["n_clusters"]),
        )
        groups.setdefault(key, []).append(row)
    summary: list[dict[str, object]] = []
    for key, group in sorted(groups.items()):
        ok = [row for row in group if row["status"] == "ok"]
        elapsed = [float(row["elapsed_seconds"]) for row in ok]
        memory = [float(row["peak_rss_mb"]) for row in ok]
        summary.append(
            {
                "cell_id": key[0],
                "method": key[1],
                "n_samples": key[2],
                "latent_dim": key[3],
                "n_clusters": key[4],
                "n_ok": len(ok),
                "n_failed": len(group) - len(ok),
                "elapsed_seconds_mean": mean(elapsed) if elapsed else "",
                "elapsed_seconds_std": stdev(elapsed) if len(elapsed) > 1 else 0.0 if elapsed else "",
                "peak_rss_mb_mean": mean(memory) if memory else "",
                "failed_candidates_total": sum(int(row["n_failed_candidates"]) for row in ok),
                "selected_families": json.dumps(Counter(str(row["selected_family"]) for row in ok), sort_keys=True),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]) if summary else [])
        if summary:
            writer.writeheader()
            writer.writerows(summary)
    print(json.dumps({"n_tasks": len(rows), "n_groups": len(summary), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()

