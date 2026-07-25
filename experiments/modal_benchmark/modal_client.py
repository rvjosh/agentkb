"""Authenticated SDK caller for the deployed benchmark app."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import modal

from experiments.modal_benchmark.modal_app import APP_NAME


def queries(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def call_timed(function: Any, *args: Any) -> dict[str, Any]:
    started = time.perf_counter()
    result = function.remote(*args)
    result["client_end_to_end_ms"] = (time.perf_counter() - started) * 1000
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["build", "batch", "warm", "cold"])
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--query-index", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.action == "build":
        function = modal.Function.from_name(APP_NAME, "build_generation")
        result = call_timed(function)
    elif args.action == "batch":
        function = modal.Function.from_name(APP_NAME, "benchmark_batch")
        result = call_timed(function, args.limit)
    elif args.action == "cold":
        function = modal.Function.from_name(APP_NAME, "cold_search")
        result = call_timed(function, queries(args.queries)[args.query_index])
    else:
        cls = modal.Cls.from_name(APP_NAME, "WarmSearch")
        instance = cls()
        started = time.perf_counter()
        result = instance.run.remote(queries(args.queries), args.repetitions)
        result["client_end_to_end_ms"] = (time.perf_counter() - started) * 1000
    payload = json.dumps(result, sort_keys=True)
    if args.output:
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
