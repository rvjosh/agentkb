"""Python-only AgentKB inference and index operations used by the Modal adapter."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from agentkb.encoder import DEFAULT_MODEL, ColBERTEncoder
from agentkb.modal_backend.generations import (
    generation_path,
    install_generation,
    read_generation_manifest,
    staged_paths,
    validate_generation_id,
)
from agentkb.search import search
from agentkb.store import IndexStore


MANIFEST_SCHEMA = 1
HASH_RE_LENGTH = 64


def download_model() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(DEFAULT_MODEL)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _load_staged(
    volume_root: Path, generation_id: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    corpus_path, manifest_path = staged_paths(volume_root, generation_id)
    try:
        corpus_bytes = corpus_path.read_bytes()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"staged generation is incomplete: {generation_id}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError("staged manifest must be a JSON object")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("staged manifest has an unsupported schema")
    if manifest.get("generation_id") != generation_id:
        raise ValueError("staged manifest generation_id does not match its path")
    if manifest.get("model") != DEFAULT_MODEL:
        raise ValueError(f"staged manifest model must be {DEFAULT_MODEL}")
    expected_count = manifest.get("corpus_count")
    if (
        not isinstance(expected_count, int)
        or isinstance(expected_count, bool)
        or expected_count < 1
    ):
        raise ValueError("staged manifest corpus_count must be a positive integer")
    expected_hash = manifest.get("corpus_hash")
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != HASH_RE_LENGTH
        or any(character not in "0123456789abcdef" for character in expected_hash)
    ):
        raise ValueError("staged manifest corpus_hash must be lowercase SHA-256")
    actual_hash = _sha256_bytes(corpus_bytes)
    if actual_hash != expected_hash:
        raise ValueError(
            f"staged corpus hash mismatch: expected {expected_hash}, got {actual_hash}"
        )

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(corpus_bytes.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid corpus JSONL at line {line_number}") from exc
        if not isinstance(record, dict) or not isinstance(record.get("content"), str):
            raise ValueError(
                f"corpus line {line_number} must be an object with string content"
            )
        records.append(record)
    if len(records) != expected_count:
        raise ValueError(
            f"staged corpus count mismatch: expected {expected_count}, got {len(records)}"
        )
    return records, manifest


def validate_index(
    index_dir: Path,
    *,
    expected_count: int,
    expected_model: str,
    expected_corpus_hash: str,
) -> dict[str, Any]:
    store = IndexStore(index_dir, read_only=True)
    try:
        store.validate_for_search()
        conn = store._connect()
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"SQLite integrity check failed: {integrity}")
        doc_ids = {
            int(row[0]) for row in conn.execute("SELECT id FROM documents").fetchall()
        }
        sqlite_count = len(doc_ids)
        fts_count = int(
            conn.execute("SELECT COUNT(*) FROM documents_fts").fetchone()[0]
        )
        forward, reverse = store._load_read_only_plaid_mappings()
        mapped_doc_ids = {int(key) for key in forward}
        reverse_doc_ids = {int(value) for value in reverse.values()}
    except sqlite3.Error as exc:
        raise ValueError(f"SQLite validation failed: {exc}") from exc
    finally:
        store.close()

    counts = {
        expected_count,
        sqlite_count,
        fts_count,
        len(forward),
        len(reverse),
    }
    if len(counts) != 1:
        raise ValueError(
            "generation count mismatch: "
            f"expected={expected_count}, sqlite={sqlite_count}, fts={fts_count}, "
            f"forward={len(forward)}, reverse={len(reverse)}"
        )
    if mapped_doc_ids != doc_ids or reverse_doc_ids != doc_ids:
        raise ValueError("generation has orphan or missing PLAID mappings")

    state_path = index_dir / "state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read index state: {exc}") from exc
    if state.get("__model__") != expected_model:
        raise ValueError("index model does not match the manifest")
    if state.get("__corpus_hash__") != expected_corpus_hash:
        raise ValueError("index corpus hash does not match the manifest")
    return {
        "sqlite_count": sqlite_count,
        "fts_count": fts_count,
        "plaid_mapping_count": len(forward),
        "plaid_reverse_mapping_count": len(reverse),
        "index_tree_hash": _tree_hash(index_dir),
    }


def build_and_publish(
    volume_root: Path,
    generation_id: str,
    *,
    commit: Callable[[], None],
) -> dict[str, Any]:
    generation_id = validate_generation_id(generation_id)
    records, staged_manifest = _load_staged(volume_root, generation_id)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="agentkb-modal-build-") as scratch_name:
        local_generation = Path(scratch_name) / generation_id
        index_dir = local_generation / "index"
        index_dir.mkdir(parents=True)
        encoder = ColBERTEncoder(staged_manifest["model"], device="cuda")
        embeddings = encoder.encode_documents([record["content"] for record in records])
        docs = [
            {key: value for key, value in record.items() if key != "canonical_id"}
            for record in records
        ]
        store = IndexStore(index_dir)
        try:
            store.create()
            ids = store.add_documents(docs)
            store.append_plaid_index(ids, embeddings)
            store.save_state(
                {
                    "__model__": staged_manifest["model"],
                    "__corpus_hash__": staged_manifest["corpus_hash"],
                }
            )
        finally:
            store.close()

        validation = validate_index(
            index_dir,
            expected_count=staged_manifest["corpus_count"],
            expected_model=staged_manifest["model"],
            expected_corpus_hash=staged_manifest["corpus_hash"],
        )
        final_manifest = {
            **staged_manifest,
            "validation": validation,
        }
        (local_generation / "manifest.json").write_text(
            json.dumps(final_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        def validate_copy(path: Path) -> None:
            copied = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            if copied != final_manifest:
                raise ValueError("copied generation manifest changed")
            copied_validation = validate_index(
                path / "index",
                expected_count=staged_manifest["corpus_count"],
                expected_model=staged_manifest["model"],
                expected_corpus_hash=staged_manifest["corpus_hash"],
            )
            if copied_validation != validation:
                raise ValueError("copied generation index changed")

        pointer = install_generation(
            volume_root,
            generation_id,
            local_generation,
            validate_copy=validate_copy,
            commit=commit,
        )
    return {
        "schema": 1,
        "generation_id": generation_id,
        "previous_generation_id": pointer["previous_generation_id"],
        "model": staged_manifest["model"],
        "corpus_count": staged_manifest["corpus_count"],
        "corpus_hash": staged_manifest["corpus_hash"],
        "validation": validation,
        "duration_ms": (time.perf_counter() - started) * 1000,
    }


class SearchRuntime:
    def __init__(self, volume_root: Path, generation_id: str):
        self.generation_id = validate_generation_id(generation_id)
        self.manifest = read_generation_manifest(volume_root, self.generation_id)
        self.scratch = tempfile.TemporaryDirectory(prefix="agentkb-modal-search-")
        self.local_generation = Path(self.scratch.name) / self.generation_id
        shutil.copytree(
            generation_path(volume_root, self.generation_id),
            self.local_generation,
        )
        local_manifest = json.loads(
            (self.local_generation / "manifest.json").read_text(encoding="utf-8")
        )
        if local_manifest != self.manifest:
            raise ValueError("local generation manifest changed during copy")
        validate_index(
            self.local_generation / "index",
            expected_count=self.manifest["corpus_count"],
            expected_model=self.manifest["model"],
            expected_corpus_hash=self.manifest["corpus_hash"],
        )
        self.encoder = ColBERTEncoder(self.manifest["model"], device="cuda")
        _ = self.encoder.dim
        self.store = IndexStore(self.local_generation / "index", read_only=True)
        self.store.validate_for_search()
        self.store._load_plaid_index()

    def warm(self) -> dict[str, Any]:
        return {
            "schema": 1,
            "generation_id": self.generation_id,
            "model": self.manifest["model"],
            "corpus_count": self.manifest["corpus_count"],
            "ready": True,
        }

    def search(self, query: str, k: int) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 100:
            raise ValueError("k must be an integer between 1 and 100")
        embedding = self.encoder.encode_query(query)
        results = search(
            self.store,
            embedding,
            query,
            scope="all",
            top_k=k,
        )
        return {
            "schema": 1,
            "generation_id": self.generation_id,
            "query": query,
            "k": k,
            "results": [
                result.to_json(include_content=True) for result in results
            ],
        }

    def close(self) -> None:
        self.store.close()
        self.scratch.cleanup()
