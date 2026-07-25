"""Create the durable sanitized aggregate for the GPU follow-up benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.modal_benchmark.common import (
    L4_GPU_RATE_PER_SECOND,
    PRIOR_T4_COLD_P50_MS,
    T4_GPU_RATE_PER_SECOND,
    classify_snapshot_attempts,
    evaluate_l4_gates,
    gpu_cost_usd,
    summarize_ms,
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _summary(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return summarize_ms([float(record[key]) for record in records])


def _nested_summary(
    records: list[dict[str, Any]], section: str, key: str
) -> dict[str, Any]:
    return summarize_ms([float(record[section][key]) for record in records])


def _query_summary(queries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        key: _summary(queries, key)
        for key in (
            "end_to_end_ms",
            "model_query_encode_ms",
            "semantic_search_ms",
            "fts_ms",
            "fusion_ms",
            "result_hydration_ms",
        )
    }


def _baseline_variant(payload: dict[str, Any], rate: float) -> dict[str, Any]:
    cold = payload["cold"]
    warm = payload["warm"]
    cold_container = _summary(cold, "container_end_to_end_ms")
    return {
        "cold": {
            "classification": "fresh_container",
            "client_end_to_end": _summary(cold, "client_end_to_end_ms"),
            "container_end_to_end": cold_container,
            "initialization": {
                key: _nested_summary(cold, "initialization", key)
                for key in (
                    "model_load_ms",
                    "volume_copy_ms",
                    "metadata_load_ms",
                    "index_load_ms",
                    "initialization_ms",
                )
            },
            "query_end_to_end": summarize_ms(
                [
                    float(record["queries"][0]["end_to_end_ms"])
                    for record in cold
                ]
            ),
            "published_gpu_cost_at_container_p50_usd": gpu_cost_usd(
                float(cold_container["p50_ms"]), rate
            ),
        },
        "warm_12_query_pass": {
            "client_end_to_end_ms": warm["client_end_to_end_ms"],
            "container_end_to_end_ms": warm["container_end_to_end_ms"],
            "query_count": len(warm["queries"]),
            "query_timing": _query_summary(warm["queries"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    generation = _load(args.root / "generation-build-raw.json")
    search = _load(args.root / "search-raw.json")
    batch = _load(args.root / "batch-raw.json")
    snapshot_attempts = _load(args.root / "snapshot-all-raw.json")["attempts"]
    logs = (args.root / "modal-logs-final-raw.txt").read_text()

    creation_count = logs.count(
        "Snapshot created. Restoring Function from memory snapshot."
    )
    post_restore_count = logs.count("AGENTKB_SNAPSHOT_POST_RESTORE_READY")
    classifications = classify_snapshot_attempts(
        len(snapshot_attempts),
        list(range(creation_count, len(snapshot_attempts))),
        restoration_proof=creation_count > 0
        and post_restore_count == len(snapshot_attempts),
        creation_attempt_indexes=list(range(creation_count)),
    )
    restored = [
        attempt
        for attempt, classification in zip(
            snapshot_attempts, classifications, strict=True
        )
        if classification == "restored"
    ]
    restored_client = _summary(restored, "client_end_to_end_ms")
    restored_container = _summary(restored, "post_restore_container_ms")

    t4 = _baseline_variant(search["t4"], T4_GPU_RATE_PER_SECOND)
    l4 = _baseline_variant(search["l4"], L4_GPU_RATE_PER_SECOND)
    l4_gates = evaluate_l4_gates(
        float(l4["cold"]["container_end_to_end"]["p50_ms"]),
        float(batch["l4"]["total_ms"]),
    )

    batch_summary = {}
    for gpu, rate in (
        ("t4", T4_GPU_RATE_PER_SECOND),
        ("l4", L4_GPU_RATE_PER_SECOND),
    ):
        record = batch[gpu]
        batch_summary[gpu] = {
            "count": record["count"],
            "client_end_to_end_ms": record["client_end_to_end_ms"],
            "internal_total_ms": record["total_ms"],
            "stage_split_ms": {
                key: record[key]
                for key in (
                    "model_load_ms",
                    "encode_ms",
                    "sqlite_ms",
                    "plaid_ms",
                )
            },
            "documents_per_second": record["documents_per_second"],
            "published_gpu_cost_at_internal_total_usd": gpu_cost_usd(
                float(record["total_ms"]), rate
            ),
        }

    t4_client_p50 = float(t4["cold"]["client_end_to_end"]["p50_ms"])
    snapshot_client_p50 = float(restored_client["p50_ms"])
    payload = {
        "schema": 1,
        "experiment": "agentkb-modal-gpu-followup-20260725",
        "resources": {
            "app": "agentkb-gpu-benchmark-20260725",
            "volume": "agentkb-gpu-benchmark-20260725-data",
            "max_containers_per_worker": 1,
            "public_endpoint": False,
        },
        "modal": {
            "authenticated_profile": True,
            "client_version": "1.5.3",
            "gpu_snapshot_feature_status": "alpha",
        },
        "corpus": {
            "generation_count": generation["count"],
            "staged_build_count": batch["t4"]["count"],
            "one_shared_immutable_generation": True,
        },
        "generation_build_t4": {
            "client_end_to_end_ms": generation["client_end_to_end_ms"],
            "internal_total_ms": generation["total_ms"],
            "stage_split_ms": {
                key: generation[key]
                for key in (
                    "model_load_ms",
                    "encode_ms",
                    "sqlite_ms",
                    "plaid_ms",
                )
            },
        },
        "published_rates_usd_per_gpu_second": {
            "t4": T4_GPU_RATE_PER_SECOND,
            "l4": L4_GPU_RATE_PER_SECOND,
            "source": "https://modal.com/pricing",
            "checked_on": "2026-07-25",
        },
        "search": {"t4_baseline": t4, "l4_baseline": l4},
        "snapshot_t4": {
            "configuration": {
                "enable_memory_snapshot": True,
                "enable_gpu_snapshot": True,
                "snap_boundary": "model initialization and representative query forward pass only",
                "index_opened_after_restore": True,
            },
            "proof": {
                "source": "exact app logs",
                "documented_restore_log_count": creation_count,
                "post_restore_marker_count": post_restore_count,
                "verified": creation_count > 0
                and post_restore_count == len(snapshot_attempts),
            },
            "attempt_count": len(snapshot_attempts),
            "classifications": [
                {
                    "attempt": index + 1,
                    "classification": classification,
                    "client_end_to_end_ms": attempt["client_end_to_end_ms"],
                    "post_restore_container_ms": attempt[
                        "post_restore_container_ms"
                    ],
                }
                for index, (attempt, classification) in enumerate(
                    zip(snapshot_attempts, classifications, strict=True)
                )
            ],
            "restored_cold": {
                "client_end_to_end": restored_client,
                "post_restore_container": restored_container,
                "post_restore_initialization": {
                    key: _nested_summary(
                        restored, "post_restore_initialization", key
                    )
                    for key in (
                        "volume_copy_ms",
                        "metadata_load_ms",
                        "index_load_ms",
                        "post_model_initialization_ms",
                    )
                },
                "query_end_to_end": summarize_ms(
                    [
                        float(attempt["queries"][0]["end_to_end_ms"])
                        for attempt in restored
                    ]
                ),
                "published_gpu_cost_at_container_p50_usd": gpu_cost_usd(
                    float(restored_container["p50_ms"]),
                    T4_GPU_RATE_PER_SECOND,
                ),
            },
            "snapshot_creation_initialization": {
                key: _nested_summary(
                    snapshot_attempts, "snapshot_initialization", key
                )
                for key in (
                    "model_load_ms",
                    "representative_forward_ms",
                    "snapshotted_initialization_ms",
                )
            },
            "client_p50_improvement_vs_same_deployment_t4_percent": (
                (t4_client_p50 - snapshot_client_p50) / t4_client_p50 * 100
            ),
            "client_p50_improvement_vs_prior_18_32s_t4_percent": (
                (PRIOR_T4_COLD_P50_MS - snapshot_client_p50)
                / PRIOR_T4_COLD_P50_MS
                * 100
            ),
        },
        "batch_1411": batch_summary,
        "gates": l4_gates,
        "billing": {
            "actual_cost_usd": None,
            "status": "pending authoritative billing report",
        },
        "decision": {
            "interactive_session_start_warming": "baseline T4",
            "refresh_and_indexing_jobs": "L4 candidate",
            "gpu_snapshots": (
                "worked, but save only a trivial amount per week for SessionStart "
                "warming and remain alpha; not worth first-deployment complexity"
            ),
            "reason": (
                "The one staged 1,411-document build was 28.04s on L4 versus "
                "68.37s on T4 and cheaper on measured container time; this is "
                "one staged-build sample."
            ),
            "session_start_hook": "not wired",
        },
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
