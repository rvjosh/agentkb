"""Python-only AgentKB inference and index operations used by the Modal adapter."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Callable, Iterable, Iterator, TextIO

import numpy as np

from agentkb.encoder import DEFAULT_MODEL, ColBERTEncoder
from agentkb.fast_plaid_immutable import inspect_immutable_premerged_artifacts
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
DOCUMENT_BATCH_SIZE = 256
PLAID_KMEANS_SAMPLE_SIZE = 16_384
PLAID_CREATE_COUNT = 1
PLAID_PERMUTATION_ALGORITHM = "sha256-key-sort-v1"
CORPUS_COLLECTIONS = {"wiki", "wiki:source", "chats"}
SOURCE_MODES = {"upstream", "projection", "human-dependent", "disabled-costly"}
SOURCE_STATES = {"fresh", "fallback", "stale", "failed"}


def _validate_source_metadata(
    manifest: dict[str, Any], expected_count: int | None = None
) -> None:
    collection_counts = manifest.get("collection_counts")
    sources = manifest.get("sources")
    if collection_counts is None and sources is None:
        return
    if not isinstance(collection_counts, dict) or set(collection_counts) != CORPUS_COLLECTIONS:
        raise ValueError("manifest collection_counts is invalid")
    for collection, counts in collection_counts.items():
        if not isinstance(counts, dict) or set(counts) != {"documents", "files"}:
            raise ValueError(f"manifest collection_counts.{collection} is invalid")
        for field in ("documents", "files"):
            value = counts[field]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"manifest collection_counts.{collection}.{field} is invalid"
                )
    if expected_count is not None and sum(
        counts["documents"] for counts in collection_counts.values()
    ) != expected_count:
        raise ValueError("manifest collection document counts do not match corpus_count")
    source_file_counts = manifest.get("source_file_counts")
    if not isinstance(source_file_counts, dict) or set(source_file_counts) != CORPUS_COLLECTIONS:
        raise ValueError("manifest source_file_counts is invalid")
    for collection in CORPUS_COLLECTIONS:
        if source_file_counts[collection] != collection_counts[collection]["files"]:
            raise ValueError(
                f"manifest file counts do not match for collection {collection}"
            )
    if (
        not isinstance(sources, dict)
        or sources.get("schema") != 1
        or not isinstance(sources.get("items"), list)
    ):
        raise ValueError("manifest sources is invalid")
    source_ids: set[str] = set()
    for index, source in enumerate(sources["items"]):
        if not isinstance(source, dict):
            raise ValueError(f"manifest sources.items[{index}] is invalid")
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not source_id or source_id in source_ids:
            raise ValueError(f"manifest sources.items[{index}].source_id is invalid")
        source_ids.add(source_id)
        if source.get("mode") not in SOURCE_MODES:
            raise ValueError(f"manifest source {source_id} mode is invalid")
        if source.get("state") not in SOURCE_STATES:
            raise ValueError(f"manifest source {source_id} state is invalid")
        for field in (
            "operation",
            "started_at",
            "finished_at",
            "root",
        ):
            if not isinstance(source.get(field), str):
                raise ValueError(f"manifest source {source_id} {field} is invalid")
        for field in (
            "duration_ms",
            "source_file_count",
            "exported_document_count",
        ):
            value = source.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"manifest source {source_id} {field} is invalid")
        for field in (
            "newest_source_timestamp",
            "warning",
            "error",
        ):
            if source.get(field) is not None and not isinstance(source[field], str):
                raise ValueError(f"manifest source {source_id} {field} is invalid")
        for field in ("freshness_threshold_minutes", "age_minutes"):
            value = source.get(field)
            if value is not None and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(f"manifest source {source_id} {field} is invalid")


def download_model() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(DEFAULT_MODEL)


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
) -> tuple[Path, dict[str, Any]]:
    corpus_path, manifest_path = staged_paths(volume_root, generation_id)
    try:
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
    digest = hashlib.sha256()
    actual_count = 0
    try:
        with corpus_path.open("rb") as corpus:
            for line_number, line in enumerate(corpus, start=1):
                digest.update(line)
                if not line.endswith(b"\n") or line.endswith(b"\r\n"):
                    raise ValueError(
                        f"corpus line {line_number} must be LF-terminated"
                    )
                if line == b"\n":
                    raise ValueError(
                        f"corpus line {line_number} must not be empty"
                    )
                actual_count += 1
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"staged generation is incomplete: {generation_id}"
        ) from exc
    if actual_count == 0:
        raise ValueError("staged corpus must not be empty")
    actual_hash = digest.hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(
            f"staged corpus hash mismatch: expected {expected_hash}, got {actual_hash}"
        )

    if actual_count != expected_count:
        raise ValueError(
            f"staged corpus count mismatch: expected {expected_count}, "
            f"got {actual_count}"
        )
    _validate_source_metadata(manifest, actual_count)
    return corpus_path, manifest


def _iter_staged_records(corpus_path: Path) -> Iterator[dict[str, Any]]:
    canonical_ids: set[str] = set()
    with corpus_path.open("rb") as corpus:
        for line_number, line in enumerate(corpus, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid corpus JSONL at line {line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"corpus line {line_number} must be an object")
            collection = record.get("collection")
            relative_file = record.get("file")
            canonical_id = record.get("canonical_id")
            if (
                collection not in CORPUS_COLLECTIONS
                or not isinstance(relative_file, str)
                or not relative_file
                or PurePosixPath(relative_file).is_absolute()
                or ".." in PurePosixPath(relative_file).parts
                or not isinstance(record.get("content"), str)
                or not isinstance(canonical_id, str)
                or len(canonical_id) != HASH_RE_LENGTH
                or any(
                    character not in "0123456789abcdef"
                    for character in canonical_id
                )
            ):
                raise ValueError(f"corpus line {line_number} is invalid")
            if canonical_id in canonical_ids:
                raise ValueError(
                    f"corpus line {line_number} has a duplicate canonical_id"
                )
            canonical_ids.add(canonical_id)
            yield record


@dataclass(frozen=True)
class _EmbeddingDescriptor:
    document_id: int
    offset: int
    length: int
    dimension: int


class _MappedEmbeddingCorpus:
    """Aligned document views backed by one copy-on-write FP16 memmap."""

    def __init__(
        self,
        *,
        storage: np.memmap,
        descriptors: list[_EmbeddingDescriptor],
        embedding_bytes: int,
    ) -> None:
        self._storage = storage
        self.document_ids = [item.document_id for item in descriptors]
        element_size = np.dtype(np.float16).itemsize
        self.embeddings = [
            storage[
                item.offset // element_size : item.offset // element_size
                + item.length * item.dimension
            ].reshape(item.length, item.dimension)
            for item in descriptors
        ]
        self.dimension = descriptors[0].dimension
        self.embedding_bytes = embedding_bytes

    def close(self) -> None:
        self.embeddings.clear()
        mmap = getattr(self._storage, "_mmap", None)
        if mmap is not None:
            mmap.close()


class _EmbeddingStage:
    """Append-only FP16 payload and descriptor ledger for a bounded build."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.embeddings_path = directory / "document-embeddings.fp16"
        self.descriptors_path = directory / "document-embeddings.jsonl"
        self._embeddings_file: BinaryIO | None = None
        self._descriptors_file: TextIO | None = None
        self._dimension: int | None = None
        self._document_ids: set[int] = set()

    def __enter__(self) -> "_EmbeddingStage":
        try:
            self.directory.mkdir(parents=True, exist_ok=False)
            self._embeddings_file = self.embeddings_path.open("xb")
            self._descriptors_file = self.descriptors_path.open(
                "x", encoding="utf-8", newline="\n"
            )
        except Exception:
            self.cleanup()
            raise
        return self

    def append(self, document_ids: list[int], embeddings: list[Any]) -> None:
        if self._embeddings_file is None or self._descriptors_file is None:
            raise RuntimeError("embedding stage is not open")
        if not document_ids or len(document_ids) != len(embeddings):
            raise ValueError("staged document IDs and embeddings must be aligned")

        for document_id, embedding in zip(document_ids, embeddings):
            if (
                not isinstance(document_id, int)
                or isinstance(document_id, bool)
                or document_id < 1
                or document_id in self._document_ids
            ):
                raise ValueError("staged document IDs must be unique positive integers")
            if hasattr(embedding, "detach"):
                embedding = embedding.detach().cpu().numpy()
            array = np.ascontiguousarray(embedding, dtype=np.float16)
            if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
                raise ValueError("document embedding must have shape (tokens, dimension)")
            dimension = int(array.shape[1])
            if self._dimension is None:
                self._dimension = dimension
            elif dimension != self._dimension:
                raise ValueError("document embedding dimensions do not match")

            offset = self._embeddings_file.tell()
            array.tofile(self._embeddings_file)
            descriptor = {
                "document_id": document_id,
                "offset": offset,
                "length": int(array.shape[0]),
                "dimension": dimension,
            }
            self._descriptors_file.write(
                json.dumps(descriptor, separators=(",", ":"), sort_keys=True) + "\n"
            )
            self._document_ids.add(document_id)

    def open_corpus(self, *, expected_count: int) -> _MappedEmbeddingCorpus:
        self._finish_writes()
        descriptors: list[_EmbeddingDescriptor] = []
        expected_offset = 0
        document_ids: set[int] = set()
        with self.descriptors_path.open(encoding="utf-8") as metadata:
            for line_number, line in enumerate(metadata, start=1):
                try:
                    value = json.loads(line)
                    descriptor = _EmbeddingDescriptor(
                        document_id=value["document_id"],
                        offset=value["offset"],
                        length=value["length"],
                        dimension=value["dimension"],
                    )
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(
                        f"invalid embedding descriptor at line {line_number}"
                    ) from exc
                if (
                    not isinstance(descriptor.document_id, int)
                    or isinstance(descriptor.document_id, bool)
                    or descriptor.document_id < 1
                    or descriptor.document_id in document_ids
                    or not isinstance(descriptor.offset, int)
                    or isinstance(descriptor.offset, bool)
                    or descriptor.offset != expected_offset
                    or not isinstance(descriptor.length, int)
                    or isinstance(descriptor.length, bool)
                    or descriptor.length < 1
                    or not isinstance(descriptor.dimension, int)
                    or isinstance(descriptor.dimension, bool)
                    or descriptor.dimension < 1
                    or (
                        descriptors
                        and descriptor.dimension != descriptors[0].dimension
                    )
                ):
                    raise ValueError(
                        f"invalid embedding descriptor at line {line_number}"
                    )
                expected_offset += (
                    descriptor.length
                    * descriptor.dimension
                    * np.dtype(np.float16).itemsize
                )
                document_ids.add(descriptor.document_id)
                descriptors.append(descriptor)

        embedding_bytes = self.embeddings_path.stat().st_size
        if len(descriptors) != expected_count:
            raise ValueError(
                f"staged embedding count mismatch: expected {expected_count}, "
                f"got {len(descriptors)}"
            )
        if embedding_bytes != expected_offset:
            raise ValueError("staged FP16 payload size does not match its descriptors")
        storage = np.memmap(
            self.embeddings_path,
            dtype=np.float16,
            mode="c",
            shape=(embedding_bytes // np.dtype(np.float16).itemsize,),
        )
        return _MappedEmbeddingCorpus(
            storage=storage,
            descriptors=descriptors,
            embedding_bytes=embedding_bytes,
        )

    def _finish_writes(self) -> None:
        for handle in (self._embeddings_file, self._descriptors_file):
            if handle is not None:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
        self._embeddings_file = None
        self._descriptors_file = None

    def cleanup(self) -> None:
        for handle in (self._embeddings_file, self._descriptors_file):
            if handle is not None and not handle.closed:
                handle.close()
        self._embeddings_file = None
        self._descriptors_file = None
        for path in (self.embeddings_path, self.descriptors_path):
            path.unlink(missing_ok=True)
        try:
            self.directory.rmdir()
        except FileNotFoundError:
            pass

    def __exit__(self, *_: object) -> None:
        self.cleanup()


def _corpus_permutation(corpus_hash: str, count: int) -> list[int]:
    """Return a stable corpus-wide order keyed by the staged corpus SHA-256."""
    if (
        len(corpus_hash) != HASH_RE_LENGTH
        or any(character not in "0123456789abcdef" for character in corpus_hash)
    ):
        raise ValueError("corpus hash must be lowercase SHA-256")
    if count < 1:
        raise ValueError("permutation count must be positive")
    seed = bytes.fromhex(corpus_hash)
    return sorted(
        range(count),
        key=lambda index: hashlib.sha256(
            seed + index.to_bytes(8, byteorder="big")
        ).digest(),
    )


def _create_global_plaid(
    *,
    embedding_stage: _EmbeddingStage,
    store: Any,
    corpus_hash: str,
    expected_count: int,
) -> dict[str, int | str]:
    """Create PLAID once from the complete staged corpus in permuted order."""
    mapped = embedding_stage.open_corpus(expected_count=expected_count)
    try:
        permutation = _corpus_permutation(corpus_hash, expected_count)
        store.create_plaid_index(
            [mapped.document_ids[index] for index in permutation],
            [mapped.embeddings[index] for index in permutation],
            n_samples_kmeans=PLAID_KMEANS_SAMPLE_SIZE,
        )
        return {
            "embedding_dimension": mapped.dimension,
            "staged_embedding_bytes": mapped.embedding_bytes,
            "plaid_create_count": PLAID_CREATE_COUNT,
            "plaid_kmeans_sample_size": PLAID_KMEANS_SAMPLE_SIZE,
            "plaid_permutation_algorithm": PLAID_PERMUTATION_ALGORITHM,
        }
    finally:
        mapped.close()


def _build_index_batches(
    records: Iterable[dict[str, Any]],
    *,
    encoder: Any,
    store: Any,
    embedding_stage: _EmbeddingStage,
    expected_count: int,
) -> tuple[int, int]:
    document_count = 0
    batch_count = 0
    iterator = iter(records)
    while batch := list(itertools.islice(iterator, DOCUMENT_BATCH_SIZE)):
        texts = [record["content"] for record in batch]
        embeddings = encoder.encode_documents(texts)
        if len(embeddings) != len(batch):
            raise ValueError("encoder returned the wrong number of embeddings")
        docs = [
            {key: value for key, value in record.items() if key != "canonical_id"}
            for record in batch
        ]
        ids = store.add_documents(docs)
        if len(ids) != len(batch):
            raise ValueError("IndexStore returned the wrong number of document IDs")
        embedding_stage.append(ids, embeddings)
        document_count += len(batch)
        batch_count += 1
        del texts, embeddings, docs, ids, batch
    if document_count != expected_count:
        raise ValueError(
            f"built corpus count mismatch: expected {expected_count}, "
            f"got {document_count}"
        )
    return document_count, batch_count


def validate_index(
    index_dir: Path,
    *,
    expected_count: int,
    expected_model: str,
    expected_corpus_hash: str,
    expected_embedding_dimension: int,
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
    immutable_premerged = inspect_immutable_premerged_artifacts(
        index_dir / "plaid" / "fast_plaid_index",
        expected_document_count=expected_count,
        expected_embedding_dimension=expected_embedding_dimension,
    )
    return {
        "sqlite_count": sqlite_count,
        "fts_count": fts_count,
        "plaid_mapping_count": len(forward),
        "plaid_reverse_mapping_count": len(reverse),
        "index_tree_hash": _tree_hash(index_dir),
        "immutable_premerged": immutable_premerged,
    }


def validate_search_build_certificate(
    generation_dir: Path,
    manifest: dict[str, Any],
    generation_id: str,
) -> None:
    """Validate the immutable build certificate and required search artifacts."""
    generation_id = validate_generation_id(generation_id)
    if generation_dir.name != generation_id:
        raise ValueError("generation directory does not match generation_id")
    if generation_dir.is_symlink():
        raise ValueError("generation directory must not be a symlink")

    def positive_integer(value: object, field: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{field} must be a positive integer")
        return value

    def sha256(value: object, field: str) -> str:
        if (
            not isinstance(value, str)
            or len(value) != HASH_RE_LENGTH
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{field} must be lowercase SHA-256")
        return value

    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or isinstance(manifest.get("schema"), bool)
    ):
        raise ValueError("generation manifest has an unsupported schema")
    if manifest.get("generation_id") != generation_id:
        raise ValueError("generation manifest generation_id does not match its path")
    if manifest.get("model") != DEFAULT_MODEL:
        raise ValueError(f"generation manifest model must be {DEFAULT_MODEL}")
    corpus_count = positive_integer(
        manifest.get("corpus_count"), "generation manifest corpus_count"
    )
    corpus_hash = sha256(
        manifest.get("corpus_hash"), "generation manifest corpus_hash"
    )
    _validate_source_metadata(manifest, corpus_count)

    build = manifest.get("build")
    if not isinstance(build, dict):
        raise ValueError("generation manifest build certificate is required")
    document_batch_size = positive_integer(
        build.get("document_batch_size"), "build.document_batch_size"
    )
    if document_batch_size != DOCUMENT_BATCH_SIZE:
        raise ValueError(f"build.document_batch_size must equal {DOCUMENT_BATCH_SIZE}")
    document_batch_count = positive_integer(
        build.get("document_batch_count"), "build.document_batch_count"
    )
    if document_batch_count != (
        corpus_count + document_batch_size - 1
    ) // document_batch_size:
        raise ValueError("build.document_batch_count does not cover corpus_count")
    embedding_dimension = positive_integer(
        build.get("embedding_dimension"), "build.embedding_dimension"
    )
    staged_embedding_bytes = positive_integer(
        build.get("staged_embedding_bytes"), "build.staged_embedding_bytes"
    )
    if staged_embedding_bytes % (embedding_dimension * np.dtype(np.float16).itemsize):
        raise ValueError(
            "build.staged_embedding_bytes is not aligned to embedding_dimension"
        )
    plaid_create_count = positive_integer(
        build.get("plaid_create_count"), "build.plaid_create_count"
    )
    if plaid_create_count != PLAID_CREATE_COUNT:
        raise ValueError(f"build.plaid_create_count must equal {PLAID_CREATE_COUNT}")
    plaid_kmeans_sample_size = positive_integer(
        build.get("plaid_kmeans_sample_size"), "build.plaid_kmeans_sample_size"
    )
    if plaid_kmeans_sample_size != PLAID_KMEANS_SAMPLE_SIZE:
        raise ValueError(
            "build.plaid_kmeans_sample_size must equal "
            f"{PLAID_KMEANS_SAMPLE_SIZE}"
        )
    if build.get("plaid_permutation_algorithm") != PLAID_PERMUTATION_ALGORITHM:
        raise ValueError(
            "build.plaid_permutation_algorithm must equal "
            f"{PLAID_PERMUTATION_ALGORITHM}"
        )

    validation = manifest.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("generation manifest validation certificate is required")
    for field in (
        "sqlite_count",
        "fts_count",
        "plaid_mapping_count",
        "plaid_reverse_mapping_count",
    ):
        if positive_integer(validation.get(field), f"validation.{field}") != corpus_count:
            raise ValueError(f"validation.{field} does not match corpus_count")
    sha256(validation.get("index_tree_hash"), "validation.index_tree_hash")
    immutable_premerged = validation.get("immutable_premerged")
    if not isinstance(immutable_premerged, dict):
        raise ValueError(
            "validation.immutable_premerged runtime certificate is required"
        )

    root = generation_dir.resolve()

    def required(path: Path, kind: str) -> None:
        try:
            path.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError(f"generation artifact escapes its directory: {path}") from exc
        if path.is_symlink():
            raise ValueError(f"generation artifact must not be a symlink: {path}")
        predicate = path.is_file if kind == "file" else path.is_dir
        if not predicate():
            raise FileNotFoundError(f"missing generation artifact: {path}")

    index_dir = generation_dir / "index"
    plaid_dir = index_dir / "plaid"
    fast_plaid_dir = plaid_dir / "fast_plaid_index"
    required(generation_dir / "manifest.json", "file")
    required(index_dir, "directory")
    required(index_dir / "metadata.db", "file")
    required(index_dir / "state.json", "file")
    required(plaid_dir, "directory")
    required(fast_plaid_dir, "directory")
    if not any(
        path.is_file() and not path.is_symlink()
        for path in fast_plaid_dir.iterdir()
    ):
        raise FileNotFoundError(f"missing FastPLAID artifacts: {fast_plaid_dir}")

    mapping_pairs = (
        (
            plaid_dir / "documents_ids_to_plaid_ids.sqlite",
            plaid_dir / "plaid_ids_to_documents_ids.sqlite",
        ),
        (
            plaid_dir / "documents_ids_to_plaid_ids.pkl",
            plaid_dir / "plaid_ids_to_documents_ids.pkl",
        ),
    )
    if not any(
        all(path.is_file() and not path.is_symlink() for path in pair)
        for pair in mapping_pairs
    ):
        raise FileNotFoundError(f"missing PLAID document ID mappings: {plaid_dir}")

    inspected_premerged = inspect_immutable_premerged_artifacts(
        fast_plaid_dir,
        expected_document_count=corpus_count,
        expected_embedding_dimension=embedding_dimension,
    )
    if inspected_premerged != immutable_premerged:
        raise ValueError(
            "immutable premerged artifacts do not match their runtime certificate"
        )

    try:
        state = json.loads((index_dir / "state.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read index state: {exc}") from exc
    if not isinstance(state, dict):
        raise ValueError("index state must be a JSON object")
    if state.get("__model__") != manifest["model"]:
        raise ValueError("index model does not match the manifest")
    if state.get("__corpus_hash__") != corpus_hash:
        raise ValueError("index corpus hash does not match the manifest")


def build_and_publish(
    volume_root: Path,
    generation_id: str,
    *,
    commit: Callable[[], None],
) -> dict[str, Any]:
    generation_id = validate_generation_id(generation_id)
    corpus_path, staged_manifest = _load_staged(volume_root, generation_id)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="agentkb-modal-build-") as scratch_name:
        local_generation = Path(scratch_name) / generation_id
        index_dir = local_generation / "index"
        index_dir.mkdir(parents=True)
        encoder = ColBERTEncoder(staged_manifest["model"], device="cuda")
        store = IndexStore(index_dir)
        try:
            store.create()
            with _EmbeddingStage(local_generation / "embedding-stage") as stage:
                _, batch_count = _build_index_batches(
                    _iter_staged_records(corpus_path),
                    encoder=encoder,
                    store=store,
                    embedding_stage=stage,
                    expected_count=staged_manifest["corpus_count"],
                )
                plaid_metrics = _create_global_plaid(
                    embedding_stage=stage,
                    store=store,
                    corpus_hash=staged_manifest["corpus_hash"],
                    expected_count=staged_manifest["corpus_count"],
                )
                build_metrics = {
                    "document_batch_size": DOCUMENT_BATCH_SIZE,
                    "document_batch_count": batch_count,
                    **plaid_metrics,
                }
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
            expected_embedding_dimension=build_metrics["embedding_dimension"],
        )
        final_manifest = {
            **staged_manifest,
            "build": build_metrics,
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
                expected_embedding_dimension=build_metrics["embedding_dimension"],
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
        **build_metrics,
        "validation": validation,
        "duration_ms": (time.perf_counter() - started) * 1000,
    }


class SearchRuntime:
    def __init__(self, volume_root: Path, generation_id: str):
        started = time.perf_counter()
        self.generation_id = validate_generation_id(generation_id)
        artifact_mount_started = time.perf_counter()
        self.generation = generation_path(volume_root, self.generation_id)
        self.manifest = read_generation_manifest(volume_root, self.generation_id)
        artifact_mount_ms = (time.perf_counter() - artifact_mount_started) * 1000
        certificate_started = time.perf_counter()
        validate_search_build_certificate(
            self.generation,
            self.manifest,
            self.generation_id,
        )
        certificate_ms = (time.perf_counter() - certificate_started) * 1000
        model_started = time.perf_counter()
        self.encoder = ColBERTEncoder(self.manifest["model"], device="cuda")
        _ = self.encoder.dim
        model_ms = (time.perf_counter() - model_started) * 1000
        index_started = time.perf_counter()
        self.store = IndexStore(
            self.generation / "index",
            read_only=True,
            immutable_premerged=self.manifest["validation"]["immutable_premerged"],
        )
        try:
            self.store.validate_for_search()
            self.store._load_plaid_index()
        except BaseException:
            self.store.close()
            raise
        index_load_ms = (time.perf_counter() - index_started) * 1000
        self.startup_timing_ms = {
            "artifact_mount": artifact_mount_ms,
            "certificate": certificate_ms,
            "model": model_ms,
            "index_load": index_load_ms,
            "total": (time.perf_counter() - started) * 1000,
        }

    def warm(self) -> dict[str, Any]:
        return {
            "schema": 1,
            "generation_id": self.generation_id,
            "model": self.manifest["model"],
            "corpus_count": self.manifest["corpus_count"],
            "startup_timing_ms": self.startup_timing_ms,
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
            "results": [_wire_search_result(result) for result in results],
        }

    def close(self) -> None:
        self.store.close()


def _wire_search_result(result: Any) -> dict[str, Any]:
    """Serialize only the stored relative path, never a container-local path."""
    value = result.to_json(include_content=True)
    relative_path = result.relative_path
    if not relative_path or Path(relative_path).is_absolute():
        raise ValueError("search result is missing its stored relative path")
    value["file"] = relative_path
    value["path"] = relative_path
    value["filename"] = Path(relative_path).name
    value["relative_path"] = relative_path
    return value
