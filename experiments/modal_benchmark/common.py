"""Deterministic corpus, generation validation, timing, and comparison helpers."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np

from agentkb.encoder import DEFAULT_MODEL
from agentkb.search import rrf_fuse, search
from agentkb.store import IndexStore
from agentkb.utils import chunk_markdown
from agentkb.wiki.parser import WIKI_SPEC


SNAPSHOT_SCHEMA = 1
BATCH_SIZE = 1_411
TOP_K = 10
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_tmp_root(path: Path) -> Path:
    """Allow benchmark artifacts only in an exact, purpose-named /tmp child."""
    expanded = path.expanduser().absolute()
    resolved = expanded.resolve()
    if expanded.parent != Path("/tmp") or not resolved.name.startswith(
        "agentkb-modal-benchmark-"
    ):
        raise ValueError(f"unsafe benchmark temporary root: {resolved}")
    return resolved


def _canonical_identity(doc: dict[str, Any]) -> str:
    identity = {
        "collection": doc["collection"],
        "file": doc["file"],
        "line": doc["line"],
        "title": doc["title"],
        "section": doc["section"],
    }
    return sha256_bytes(canonical_json(identity).encode())


def snapshot_documents(wiki_root: Path) -> list[dict[str, Any]]:
    """Parse the configured wiki through AgentKB's real parser and formatter."""
    documents: list[dict[str, Any]] = []
    for rel, entry in WIKI_SPEC.list_files(wiki_root).items():
        for chunk in chunk_markdown(entry.path, relative_to=wiki_root):
            structured = WIKI_SPEC.make_structured_text(chunk, entry)
            doc = {
                "collection": entry.collection,
                "file": chunk["file"],
                "line": chunk["line"],
                "name": chunk["title"],
                "unit_type": "chunk",
                "content": structured,
                "raw_content": chunk["content"],
                "title": chunk["title"],
                "section": chunk["section"],
                "tags": chunk.get("tags", []),
            }
            doc["canonical_id"] = _canonical_identity(doc)
            documents.append(doc)

    documents.sort(
        key=lambda d: (
            d["collection"],
            d["file"],
            d["line"],
            d["title"],
            d["section"],
        )
    )
    identities = [d["canonical_id"] for d in documents]
    if len(identities) != len(set(identities)):
        duplicates = len(identities) - len(set(identities))
        raise ValueError(f"snapshot has {duplicates} duplicate canonical identities")
    return documents


def write_snapshot(wiki_root: Path, output_root: Path) -> dict[str, Any]:
    output_root = validate_tmp_root(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    docs = snapshot_documents(wiki_root)
    if len(docs) < BATCH_SIZE:
        raise ValueError(f"need at least {BATCH_SIZE} chunks, found {len(docs)}")

    serialized = [canonical_json(doc) for doc in docs]
    corpus_bytes = ("\n".join(serialized) + "\n").encode()
    corpus_hash = sha256_bytes(corpus_bytes)
    (output_root / "corpus.jsonl").write_bytes(corpus_bytes)

    selected = sorted(
        docs,
        key=lambda d: sha256_bytes(
            f"{corpus_hash}:{d['canonical_id']}".encode()
        ),
    )[:BATCH_SIZE]
    batch_bytes = ("\n".join(canonical_json(doc) for doc in selected) + "\n").encode()
    batch_hash = sha256_bytes(batch_bytes)
    (output_root / "batch-1411.jsonl").write_bytes(batch_bytes)

    manifest = {
        "schema": SNAPSHOT_SCHEMA,
        "model": DEFAULT_MODEL,
        "corpus_count": len(docs),
        "corpus_hash": corpus_hash,
        "batch_count": len(selected),
        "batch_hash": batch_hash,
        "source_file_count": len(WIKI_SPEC.list_files(wiki_root)),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def indexable_docs(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop benchmark-only identity metadata before using the real IndexStore."""
    return [{k: v for k, v in record.items() if k != "canonical_id"} for record in records]


def file_tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def validate_generation(index_dir: Path, expected_count: int) -> dict[str, Any]:
    """Assert SQLite, FTS, and both PLAID mappings form one clean generation."""
    store = IndexStore(index_dir, read_only=True)
    store.validate_for_search()
    conn = store._connect()
    doc_ids = {
        int(row[0]) for row in conn.execute("SELECT id FROM documents").fetchall()
    }
    sqlite_count = len(doc_ids)
    fts_count = int(
        conn.execute("SELECT COUNT(*) FROM documents_fts").fetchone()[0]
    )
    forward, reverse = store._load_read_only_plaid_mappings()
    mapping_count = len(forward)
    reverse_count = len(reverse)
    mapped_doc_ids = {int(key) for key in forward}
    reverse_doc_ids = {int(value) for value in reverse.values()}
    store.close()

    counts = {sqlite_count, fts_count, mapping_count, reverse_count, expected_count}
    if len(counts) != 1:
        raise ValueError(
            "generation count mismatch: "
            f"expected={expected_count}, sqlite={sqlite_count}, fts={fts_count}, "
            f"forward={mapping_count}, reverse={reverse_count}"
        )
    if mapped_doc_ids != doc_ids or reverse_doc_ids != doc_ids:
        raise ValueError("generation has orphan or missing PLAID mappings")
    return {
        "canonical_count": expected_count,
        "sqlite_count": sqlite_count,
        "fts_count": fts_count,
        "plaid_mapping_count": mapping_count,
        "plaid_reverse_mapping_count": reverse_count,
        "tree_hash": file_tree_hash(index_dir),
    }


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("cannot calculate percentile of an empty sample")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_ms(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "min_ms": min(values),
        "p50_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values),
    }


def query_result_ids(results: list[Any]) -> list[str]:
    """Use stable path/line identities in temporary raw results."""
    return [
        sha256_bytes(
            canonical_json(
                {
                    "collection": result.collection,
                    "file": result.relative_path,
                    "line": result.line,
                    "title": result.title,
                    "section": result.section,
                }
            ).encode()
        )
        for result in results
    ]


def run_timed_query(
    store: IndexStore,
    encoder: Any,
    query: dict[str, Any],
    *,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    trace = SimpleNamespace(
        original_query=query["text"],
        semantic_query=query["text"],
        scope=query["scope"],
        top_k=top_k,
        semantic_ranking=[],
        keyword_ranking=[],
        rrf_ranking=[],
        final_results=[],
    )
    started = time.perf_counter()
    encode_started = time.perf_counter()
    embedding = encoder.encode_query(query["text"])
    encode_ms = (time.perf_counter() - encode_started) * 1000

    semantic_ms = 0.0
    fts_ms = 0.0
    original_semantic = store.semantic_search
    original_keyword = store.keyword_search

    def timed_semantic(*args: Any, **kwargs: Any) -> Any:
        nonlocal semantic_ms
        stage = time.perf_counter()
        result = original_semantic(*args, **kwargs)
        semantic_ms += (time.perf_counter() - stage) * 1000
        return result

    def timed_keyword(*args: Any, **kwargs: Any) -> Any:
        nonlocal fts_ms
        stage = time.perf_counter()
        result = original_keyword(*args, **kwargs)
        fts_ms += (time.perf_counter() - stage) * 1000
        return result

    store.semantic_search = timed_semantic  # type: ignore[method-assign]
    store.keyword_search = timed_keyword  # type: ignore[method-assign]
    pipeline_started = time.perf_counter()
    try:
        results = search(
            store,
            embedding,
            query["text"],
            scope=query["scope"],
            top_k=top_k,
            semantic_only=False,
            trace=trace,
        )
    finally:
        store.semantic_search = original_semantic  # type: ignore[method-assign]
        store.keyword_search = original_keyword  # type: ignore[method-assign]
    pipeline_ms = (time.perf_counter() - pipeline_started) * 1000

    fusion_started = time.perf_counter()
    rrf_fuse(trace.semantic_ranking, trace.keyword_ranking, alpha=0.75)
    fusion_ms = (time.perf_counter() - fusion_started) * 1000
    return {
        "label": query["label"],
        "scope": query["scope"],
        "end_to_end_ms": (time.perf_counter() - started) * 1000,
        "model_query_encode_ms": encode_ms,
        "semantic_search_ms": semantic_ms,
        "fts_ms": fts_ms,
        "fusion_ms": fusion_ms,
        "result_hydration_ms": max(
            0.0, pipeline_ms - semantic_ms - fts_ms - fusion_ms
        ),
        "result_ids": query_result_ids(results),
        "scores": [float(result.score) for result in results],
        "result_count": len(results),
    }


def compare_query_results(
    local: list[dict[str, Any]],
    remote: list[dict[str, Any]],
    *,
    score_atol: float = 1e-3,
) -> dict[str, Any]:
    local_by_label = {item["label"]: item for item in local}
    remote_by_label = {item["label"]: item for item in remote}
    labels_equal = set(local_by_label) == set(remote_by_label)
    comparisons = []
    for label in sorted(set(local_by_label) & set(remote_by_label)):
        left = local_by_label[label]
        right = remote_by_label[label]
        ids_equal = left["result_ids"] == right["result_ids"]
        scores_equal = len(left["scores"]) == len(right["scores"]) and bool(
            np.allclose(left["scores"], right["scores"], atol=score_atol, rtol=0)
        )
        comparisons.append(
            {
                "label": label,
                "top_k_ids_equal": ids_equal,
                "scores_within_tolerance": scores_equal,
                "max_score_delta": max(
                    (
                        abs(a - b)
                        for a, b in zip(left["scores"], right["scores"], strict=False)
                    ),
                    default=0.0,
                ),
            }
        )
    return {
        "labels_equal": labels_equal,
        "queries_compared": len(comparisons),
        "all_top_k_ids_equal": labels_equal
        and all(item["top_k_ids_equal"] for item in comparisons),
        "all_scores_within_tolerance": labels_equal
        and all(item["scores_within_tolerance"] for item in comparisons),
        "score_atol": score_atol,
        "per_query": comparisons,
    }


def apply_offline_environment() -> None:
    os.environ.update(OFFLINE_ENV)
