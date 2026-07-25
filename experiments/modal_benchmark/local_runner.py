"""Local CPU side of the benchmark; writes only to caller-provided /tmp paths."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from agentkb.encoder import get_encoder
from agentkb.store import IndexStore

from experiments.modal_benchmark.common import (
    apply_offline_environment,
    indexable_docs,
    load_jsonl,
    run_timed_query,
    validate_generation,
    validate_tmp_root,
)


def load_queries(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def initialize(index_dir: Path) -> tuple[IndexStore, Any, dict[str, float]]:
    apply_offline_environment()
    started = time.perf_counter()
    encoder = get_encoder()
    model_started = time.perf_counter()
    _ = encoder.dim
    model_load_ms = (time.perf_counter() - model_started) * 1000
    store_started = time.perf_counter()
    store = IndexStore(index_dir, read_only=True)
    store.validate_for_search()
    metadata_load_ms = (time.perf_counter() - store_started) * 1000
    plaid_started = time.perf_counter()
    store._load_plaid_index()
    plaid_load_ms = (time.perf_counter() - plaid_started) * 1000
    return store, encoder, {
        "model_load_ms": model_load_ms,
        "metadata_load_ms": metadata_load_ms,
        "index_load_ms": plaid_load_ms,
        "initialization_ms": (time.perf_counter() - started) * 1000,
    }


def command_cold(args: argparse.Namespace) -> dict[str, Any]:
    queries = load_queries(args.queries)
    store, encoder, initialization = initialize(args.index)
    try:
        result = run_timed_query(store, encoder, queries[args.query_index])
    finally:
        store.close()
    return {"environment": "local", "initialization": initialization, "query": result}


def command_warm(args: argparse.Namespace) -> dict[str, Any]:
    queries = load_queries(args.queries)
    store, encoder, initialization = initialize(args.index)
    results = []
    try:
        for _ in range(args.repetitions):
            for query in queries:
                results.append(run_timed_query(store, encoder, query))
    finally:
        store.close()
    return {
        "environment": "local",
        "initialization": initialization,
        "queries": results,
    }


def command_batch(args: argparse.Namespace) -> dict[str, Any]:
    apply_offline_environment()
    records = load_jsonl(args.batch)[: args.limit]
    docs = indexable_docs(records)
    texts = [doc["content"] for doc in docs]
    encoder = get_encoder()
    model_started = time.perf_counter()
    _ = encoder.dim
    model_load_ms = (time.perf_counter() - model_started) * 1000
    total_started = time.perf_counter()
    encode_started = time.perf_counter()
    embeddings = encoder.encode_documents(texts)
    encode_ms = (time.perf_counter() - encode_started) * 1000

    scratch_parent = validate_tmp_root(args.scratch_root)
    with tempfile.TemporaryDirectory(
        prefix="agentkb-modal-benchmark-index-", dir=scratch_parent
    ) as scratch:
        index_dir = Path(scratch) / "index"
        store = IndexStore(index_dir)
        sqlite_started = time.perf_counter()
        store.create()
        ids = store.add_documents(docs)
        sqlite_ms = (time.perf_counter() - sqlite_started) * 1000
        plaid_started = time.perf_counter()
        store.append_plaid_index(ids, embeddings)
        plaid_ms = (time.perf_counter() - plaid_started) * 1000
        validation = validate_generation(index_dir, len(docs))
        store.close()
    total_ms = (time.perf_counter() - total_started) * 1000
    return {
        "environment": "local",
        "count": len(docs),
        "model_load_ms": model_load_ms,
        "encode_ms": encode_ms,
        "sqlite_ms": sqlite_ms,
        "plaid_ms": plaid_ms,
        "total_ms": total_ms,
        "documents_per_second": len(docs) / (total_ms / 1000),
        "validation": validation,
    }


def command_validate(args: argparse.Namespace) -> dict[str, Any]:
    return validate_generation(args.index, args.expected_count)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("cold", "warm"):
        command = subparsers.add_parser(name)
        command.add_argument("--index", type=Path, required=True)
        command.add_argument("--queries", type=Path, required=True)
        if name == "cold":
            command.add_argument("--query-index", type=int, required=True)
        else:
            command.add_argument("--repetitions", type=int, default=1)
    batch = subparsers.add_parser("batch")
    batch.add_argument("--batch", type=Path, required=True)
    batch.add_argument("--scratch-root", type=Path, required=True)
    batch.add_argument("--limit", type=int, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--index", type=Path, required=True)
    validate.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()
    result = {
        "cold": command_cold,
        "warm": command_warm,
        "batch": command_batch,
        "validate": command_validate,
    }[args.command](args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
