"""Run the bounded authenticated Modal GPU follow-up suite."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import modal

from experiments.modal_benchmark.gpu_followup_modal_app import APP_NAME


def timed_remote(call: Any, *args: Any) -> dict[str, Any]:
    started = time.perf_counter()
    result = call.remote(*args)
    result["client_end_to_end_ms"] = (time.perf_counter() - started) * 1000
    return result


def call_class(class_name: str, argument: Any) -> dict[str, Any]:
    remote_class = modal.Cls.from_name(APP_NAME, class_name)
    return timed_remote(remote_class().run, argument)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=["build", "search", "batch", "snapshot"],
        required=True,
    )
    parser.add_argument("--snapshot-attempts", type=int, default=8)
    args = parser.parse_args()
    query_set = json.loads(args.queries.read_text()) if args.queries else None

    if args.phase == "build":
        function = modal.Function.from_name(APP_NAME, "build_generation")
        payload = timed_remote(function)
    elif args.phase == "search":
        if query_set is None:
            parser.error("--queries is required for the search phase")
        payload = {
            "t4": {
                "cold": [
                    call_class("T4ColdSearch", query_set[index])
                    for index in (0, 4, 10)
                ],
                "warm": call_class("T4WarmSearch", query_set),
            },
            "l4": {
                "cold": [
                    call_class("L4ColdSearch", query_set[index])
                    for index in (0, 4, 10)
                ],
                "warm": call_class("L4WarmSearch", query_set),
            },
        }
    elif args.phase == "batch":
        payload = {}
        for gpu in ("t4", "l4"):
            function = modal.Function.from_name(
                APP_NAME, f"benchmark_batch_{gpu}"
            )
            payload[gpu] = timed_remote(function, 1411)
    else:
        if query_set is None:
            parser.error("--queries is required for the snapshot phase")
        attempts = max(1, min(args.snapshot_attempts, 8))
        payload = {
            "attempts": [
                call_class(
                    "T4SnapshotSearch",
                    query_set[(attempt * 4) % len(query_set)],
                )
                for attempt in range(attempts)
            ]
        }

    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "phase": args.phase,
                "completed": (
                    len(payload["attempts"])
                    if args.phase == "snapshot"
                    else len(payload)
                    if args.phase in {"search", "batch"}
                    else True
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
