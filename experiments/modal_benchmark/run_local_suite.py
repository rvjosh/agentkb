"""Run the complete local suite with cold searches in fresh subprocesses."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from experiments.modal_benchmark.common import OFFLINE_ENV


def invoke(arguments: list[str]) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "experiments.modal_benchmark.local_runner", *arguments],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, **OFFLINE_ENV},
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    return json.loads(completed.stdout.strip().splitlines()[-1]), elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    common_search = [
        "--index",
        str(args.index),
        "--queries",
        str(args.queries),
    ]
    warm, warm_client_ms = invoke(["warm", *common_search, "--repetitions", "1"])
    warm["subprocess_end_to_end_ms"] = warm_client_ms

    cold = []
    for query_index in (0, 4, 10):
        result, client_ms = invoke(
            ["cold", *common_search, "--query-index", str(query_index)]
        )
        result["subprocess_end_to_end_ms"] = client_ms
        cold.append(result)

    batch_path = args.snapshot_root / "batch-1411.jsonl"
    scratch = str(args.snapshot_root)
    smoke, smoke_client_ms = invoke(
        [
            "batch",
            "--batch",
            str(batch_path),
            "--scratch-root",
            scratch,
            "--limit",
            "10",
        ]
    )
    smoke["subprocess_end_to_end_ms"] = smoke_client_ms
    full, full_client_ms = invoke(
        [
            "batch",
            "--batch",
            str(batch_path),
            "--scratch-root",
            scratch,
            "--limit",
            "1411",
        ]
    )
    full["subprocess_end_to_end_ms"] = full_client_ms
    payload = {
        "environment": "local",
        "warm": warm,
        "cold": cold,
        "batch_smoke": smoke,
        "batch_full": full,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "warm_queries": len(warm["queries"]),
                "cold_queries": len(cold),
                "batch_count": full["count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
