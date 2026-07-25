"""Run the complete authenticated Modal suite against the deployed app."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import modal

from experiments.modal_benchmark.modal_app import APP_NAME


def timed_remote(call: Any, *args: Any) -> dict[str, Any]:
    started = time.perf_counter()
    result = call.remote(*args)
    result["client_end_to_end_ms"] = (time.perf_counter() - started) * 1000
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    query_set = json.loads(args.queries.read_text())

    cls = modal.Cls.from_name(APP_NAME, "WarmSearch")
    warm_instance = cls()
    warm = timed_remote(warm_instance.run, query_set, 1)

    cold_function = modal.Function.from_name(APP_NAME, "cold_search")
    cold = [
        timed_remote(cold_function, query_set[index]) for index in (0, 4, 10)
    ]

    batch_function = modal.Function.from_name(APP_NAME, "benchmark_batch")
    smoke = timed_remote(batch_function, 10)
    full = timed_remote(batch_function, 1411)
    payload = {
        "environment": "modal",
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
