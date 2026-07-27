"""Model-free production corpus export for the private Modal backend."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import re
import sqlite3
import stat
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from agentkb.chats.parser import CHAT_SPEC
from agentkb.chats.renderer import (
    export_all_sessions,
    export_readable,
    migrate_sessions_layout,
)
from agentkb.encoder import DEFAULT_MODEL
from agentkb.indexing import IndexSpec, list_markdown_files
from agentkb.modal_backend.generations import validate_generation_id
from agentkb.utils import chunk_markdown
from agentkb.wiki.parser import WIKI_SPEC


SCHEMA = 1
SOURCES_SCHEMA = 1
CORPUS_FILENAME = "corpus.jsonl"
MANIFEST_FILENAME = "manifest.json"
COLLECTIONS = ("chats", "wiki", "wiki:source")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
ARCHIVE_POINTER_SCHEMA = 1


def _sanitize_json_strings(value: Any) -> Any:
    """Escape lone surrogates without changing valid Unicode source text."""
    if isinstance(value, str):
        return value.encode("utf-8", errors="backslashreplace").decode("utf-8")
    if isinstance(value, list):
        return [_sanitize_json_strings(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_json_strings(item) for item in value)
    if isinstance(value, dict):
        return {
            _sanitize_json_strings(key): _sanitize_json_strings(item)
            for key, item in value.items()
        }
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _sanitize_json_strings(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_id(record: dict[str, Any]) -> str:
    identity = [
        record["collection"],
        record["file"],
        record["line"],
        record["title"],
        record["section"],
    ]
    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


def validate_output_directory(output_dir: Path) -> Path:
    """Require an explicitly created, empty, narrow output directory."""
    if not output_dir.is_absolute():
        raise ValueError("output directory must be an explicit absolute path")
    if output_dir.is_symlink():
        raise ValueError("output directory must not be a symlink")
    resolved = output_dir.resolve()
    repo_root = Path(__file__).resolve().parents[3]
    forbidden = {Path("/"), Path.home().resolve(), repo_root}
    if resolved in forbidden:
        raise ValueError("output directory is too broad")
    if not resolved.is_dir():
        raise ValueError("output directory must already exist")
    if any(resolved.iterdir()):
        raise ValueError("output directory must be empty")
    return resolved


def prepare_chats(chats_root: Path) -> Path:
    """Run the normal local chat copy/render preparation without indexing."""
    sessions_dir = chats_root / "sessions"
    readable_dir = chats_root / "readable"
    migrate_sessions_layout(sessions_dir)
    export_all_sessions(sessions_dir)
    if sessions_dir.exists():
        export_readable(sessions_dir, readable_dir)
    return readable_dir


def _source_entries(
    roots_and_specs: tuple[tuple[Path, IndexSpec], ...],
) -> Iterator[tuple[str, str, Path, IndexSpec, Any]]:
    entries = (
        (entry.collection, relative_file, root, spec, entry)
        for root, spec in roots_and_specs
        for relative_file, entry in spec.list_files(root).items()
    )
    yield from sorted(entries, key=lambda item: (item[0], item[1]))


def _records_for_entry(
    root: Path,
    spec: IndexSpec,
    relative_file: str,
    entry: Any,
    *,
    stored_prefix: str = "",
) -> Iterator[dict[str, Any]]:
    for chunk in chunk_markdown(entry.path, relative_to=root):
        stored_file = f"{stored_prefix}{chunk['file']}"
        record: dict[str, Any] = {
            "collection": entry.collection,
            "content": spec.make_structured_text(chunk, entry),
            "file": stored_file,
            "line": chunk["line"],
            "name": chunk["title"],
            "raw_content": chunk["content"],
            "section": chunk["section"],
            "tags": chunk.get("tags", []),
            "title": chunk["title"],
            "unit_type": "chunk",
        }
        if chunk["file"] != relative_file:
            raise ValueError("parser returned a non-canonical relative file path")
        record["canonical_id"] = _canonical_id(record)
        yield record


def _markdown_records(
    *,
    root: Path,
    collection: str,
    stored_prefix: str,
) -> Iterator[tuple[str, dict[str, Any]]]:
    spec = CHAT_SPEC if collection == "chats" else WIKI_SPEC
    entries = list_markdown_files(root, collection=collection)
    for relative_file, entry in sorted(entries.items()):
        for record in _records_for_entry(
            root,
            spec,
            relative_file,
            entry,
            stored_prefix=stored_prefix,
        ):
            yield relative_file, record


def _jsonl_records(
    *,
    root: Path,
    collection: str,
    stored_prefix: str,
    include: set[str] | None = None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    for path in sorted(root.rglob("*.jsonl")):
        relative_file = path.relative_to(root).as_posix()
        if include is not None and relative_file not in include:
            continue
        stored_file = f"{stored_prefix}{relative_file}"
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL in {path} at line {line_number}"
                    ) from exc
                raw_content = json.dumps(
                    value, ensure_ascii=False, indent=2, sort_keys=True
                )
                record = {
                    "collection": collection,
                    "content": (
                        f"[{collection}] {relative_file} > item-{line_number}\n\n"
                        f"{raw_content}"
                    ),
                    "file": stored_file,
                    "line": line_number,
                    "name": relative_file,
                    "raw_content": raw_content,
                    "section": f"item-{line_number}",
                    "tags": [],
                    "title": relative_file,
                    "unit_type": "item",
                }
                record["canonical_id"] = _canonical_id(record)
                yield relative_file, record


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    texts = []
    for block in content:
        if isinstance(block, str):
            texts.append(block)
        elif isinstance(block, dict) and block.get("type") in {
            "text",
            "input_text",
            "output_text",
            "summary_text",
        }:
            text = block.get("text", "")
            if isinstance(text, str):
                texts.append(text)
    return "\n".join(texts).strip()


def _iter_history_messages(
    source: str, compressed_blob: Path, expected_sha256: str
) -> Iterator[tuple[str, str, str]]:
    process = subprocess.Popen(
        ["zstd", "-dc", str(compressed_blob)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    digest = hashlib.sha256()
    codex_messages: list[tuple[str, str, str, bool]] = []
    saw_codex_event_user = False
    try:
        for raw_line in process.stdout:
            digest.update(raw_line)
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            timestamp = str(obj.get("timestamp", ""))
            if source == "claude":
                role = obj.get("type")
                if role not in {"user", "assistant"} or obj.get("isMeta"):
                    continue
                message = obj.get("message", {})
                if not isinstance(message, dict):
                    continue
                text = _extract_text(message.get("content"))
                if text:
                    yield str(role), text, timestamp
                continue

            payload = obj.get("payload", {})
            if not isinstance(payload, dict):
                continue
            if obj.get("type") == "event_msg" and payload.get("type") == "user_message":
                text = payload.get("message", "")
                if isinstance(text, str) and text.strip():
                    saw_codex_event_user = True
                    codex_messages.append(("user", text.strip(), timestamp, False))
                continue
            if obj.get("type") != "response_item" or payload.get("type") != "message":
                continue
            role = payload.get("role")
            if role not in {"user", "assistant"}:
                continue
            text = _extract_text(payload.get("content"))
            if text:
                codex_messages.append(
                    (str(role), text, timestamp, role == "user")
                )
    finally:
        process.stdout.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"cannot decompress {compressed_blob}: {stderr}")
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"history blob hash mismatch: {compressed_blob}")
    for role, text, timestamp, fallback_user in codex_messages:
        if fallback_user and saw_codex_event_user:
            continue
        yield role, text, timestamp


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _private_file(path: Path, expected_size: int | None = None) -> os.stat_result:
    metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise ValueError(f"central history private file is unsafe: {path.name}")
    if expected_size is not None and metadata.st_size != expected_size:
        raise ValueError(f"central history file size mismatch: {path.name}")
    return metadata


def _exact_keys(value: dict[str, Any], expected: set[str]) -> bool:
    return set(value) == expected


def _generation_file(root: Path, filename: Any) -> Path:
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("central history generation filename must be a basename")
    path = root / filename
    if path.parent != root:
        raise ValueError("central history generation filename escaped its root")
    return path


def _load_history_pointer(backup_root: Path) -> tuple[dict[str, Any], Path, Path]:
    requested_metadata = backup_root.lstat()
    if stat.S_ISLNK(requested_metadata.st_mode):
        raise ValueError("central history mirror root must not be a symlink")
    root = backup_root.resolve()
    root_metadata = root.lstat()
    if (
        stat.S_ISLNK(root_metadata.st_mode)
        or not stat.S_ISDIR(root_metadata.st_mode)
        or stat.S_IMODE(root_metadata.st_mode) != 0o700
    ):
        raise ValueError("central history mirror root must be a private 0700 directory")
    current = root / "current.json"
    metadata = _private_file(current)
    if metadata.st_size <= 0 or metadata.st_size > 64 * 1024:
        raise ValueError("central history current.json size is invalid")
    pointer = json.loads(current.read_text(encoding="utf-8"))
    if not isinstance(pointer, dict) or not _exact_keys(
        pointer,
        {
            "schemaVersion",
            "archiveSchema",
            "catalogSchema",
            "database",
            "catalog",
            "sqliteRuntimeVersion",
            "referencedBlobCount",
            "verifiedBlobCount",
            "verifiedBytes",
            "knownParserProvenanceCount",
            "legacyParserProvenanceCount",
            "integrityCheck",
            "foreignKeyCheck",
        },
    ):
        raise ValueError("central history current.json has unknown or missing fields")
    database = pointer.get("database")
    catalog = pointer.get("catalog")
    counts = (
        "referencedBlobCount",
        "verifiedBlobCount",
        "verifiedBytes",
        "knownParserProvenanceCount",
        "legacyParserProvenanceCount",
    )
    if (
        pointer.get("schemaVersion") != ARCHIVE_POINTER_SCHEMA
        or pointer.get("archiveSchema") != 4
        or pointer.get("catalogSchema") != 1
        or not isinstance(pointer.get("sqliteRuntimeVersion"), str)
        or any(
            not isinstance(pointer.get(name), int) or pointer[name] < 0
            for name in counts
        )
        or pointer["verifiedBlobCount"] != pointer["referencedBlobCount"]
        or pointer.get("integrityCheck") != "ok"
        or pointer.get("foreignKeyCheck") != "ok"
        or not isinstance(database, dict)
        or not isinstance(catalog, dict)
    ):
        raise ValueError("central history current.json schema is invalid")
    if not _exact_keys(
        database,
        {
            "filename",
            "sha256",
            "compressedSha256",
            "bytes",
            "compressedBytes",
            "logicalFingerprint",
        },
    ) or not _exact_keys(
        catalog,
        {"filename", "sha256", "bytes", "recordCount", "fingerprint"},
    ):
        raise ValueError("central history generation entries are invalid")
    database_sha = database.get("sha256")
    catalog_sha = catalog.get("sha256")
    if (
        not isinstance(database_sha, str)
        or SHA256_PATTERN.fullmatch(database_sha) is None
        or database.get("filename") != f"history-index-{database_sha}.sqlite3.zst"
        or not isinstance(database.get("compressedSha256"), str)
        or SHA256_PATTERN.fullmatch(database["compressedSha256"]) is None
        or not isinstance(database.get("bytes"), int)
        or database["bytes"] <= 0
        or not isinstance(database.get("compressedBytes"), int)
        or database["compressedBytes"] <= 0
        or not isinstance(database.get("logicalFingerprint"), str)
        or SHA256_PATTERN.fullmatch(database["logicalFingerprint"]) is None
        or not isinstance(catalog_sha, str)
        or SHA256_PATTERN.fullmatch(catalog_sha) is None
        or catalog.get("filename") != f"provenance-catalog-{catalog_sha}.jsonl"
        or not isinstance(catalog.get("bytes"), int)
        or catalog["bytes"] < 0
        or not isinstance(catalog.get("recordCount"), int)
        or catalog["recordCount"] < 0
        or not isinstance(catalog.get("fingerprint"), str)
        or SHA256_PATTERN.fullmatch(catalog["fingerprint"]) is None
    ):
        raise ValueError("central history generation identity is invalid")
    compressed = _generation_file(root, database["filename"])
    catalog_path = _generation_file(root, catalog["filename"])
    _private_file(compressed, database["compressedBytes"])
    _private_file(catalog_path, catalog["bytes"])
    if _hash_file(compressed) != database["compressedSha256"]:
        raise ValueError("central history compressed database hash mismatch")
    if _hash_file(catalog_path) != catalog_sha:
        raise ValueError("central history catalog hash mismatch")
    return pointer, compressed, catalog_path


def _decompress_history_index(backup_root: Path) -> Path:
    pointer, compressed, _ = _load_history_pointer(backup_root)
    handle = tempfile.NamedTemporaryFile(
        prefix="agentkb-history-", suffix=".sqlite3", delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            result = subprocess.run(
                ["zstd", "-dc", str(compressed)],
                stdout=handle,
                stderr=subprocess.PIPE,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(
                "cannot decompress central history snapshot: "
                + result.stderr.decode("utf-8", errors="replace").strip()
            )
        os.chmod(temporary, 0o600)
        _private_file(temporary, pointer["database"]["bytes"])
        if _hash_file(temporary) != pointer["database"]["sha256"]:
            raise ValueError("central history uncompressed database hash mismatch")
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _history_records(
    backup_root: Path,
) -> Iterator[tuple[str, dict[str, Any]]]:
    database_path = _decompress_history_index(backup_root)
    try:
        connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        has_schema_meta = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'schema_meta'
            """
        ).fetchone()
        if has_schema_meta is None:
            schema_version = 1
        else:
            schema = connection.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()
            if schema is None:
                raise ValueError(
                    "central history snapshot is missing its schema version"
                )
            try:
                schema_version = int(schema["value"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "central history snapshot has a malformed schema version"
                ) from exc
        if schema_version not in {1, 2, 3, 4}:
            raise ValueError(
                f"unsupported central history schema version: {schema_version}"
            )
        sessions_relation = (
            "publication_eligible_sessions"
            if schema_version in {2, 3, 4}
            else "transcripts"
        )
        rows = connection.execute(
            f"""
            WITH present_versions AS (
                SELECT
                    t.source,
                    t.native_session_id,
                    v.sha256,
                    v.blob_path,
                    v.title,
                    v.cwd,
                    v.start_time,
                    v.end_time,
                    v.created_at,
                    ROW_NUMBER() OVER (
                        PARTITION BY t.source, t.native_session_id
                        ORDER BY v.created_at DESC, v.sha256 DESC
                    ) AS version_rank
                FROM {sessions_relation} t
                JOIN versions v ON v.transcript_id = t.id
                WHERE v.parser_status = 'ok'
                  AND EXISTS (
                      SELECT 1
                      FROM observations o
                      WHERE o.version_sha256 = v.sha256
                        AND o.present_at_last_scan = 1
                  )
            )
            SELECT *
            FROM present_versions
            WHERE version_rank = 1
            ORDER BY source, native_session_id
            """
        ).fetchall()
        connection.close()
        for row in rows:
            source = str(row["source"])
            session_id = str(row["native_session_id"])
            relative_blob = Path(str(row["blob_path"]))
            if relative_blob.is_absolute() or ".." in relative_blob.parts:
                raise ValueError("central history blob path escaped its root")
            blob = (backup_root / f"{relative_blob}.zst").resolve()
            if not blob.is_relative_to(backup_root.resolve()) or not blob.is_file():
                raise FileNotFoundError(f"central history blob is missing: {blob}")
            stored_file = f"agent-history-central/{source}/{session_id}.md"
            session_key = f"{source}/{session_id}"
            for ordinal, (role, text, timestamp) in enumerate(
                _iter_history_messages(source, blob, str(row["sha256"])),
                start=1,
            ):
                section = f"message-{ordinal:06d}"
                metadata = [
                    f"Source: {source}",
                    f"Session: {session_id}",
                    f"Role: {role}",
                ]
                if timestamp:
                    metadata.append(f"Timestamp: {timestamp}")
                if row["cwd"]:
                    metadata.append(f"Working directory: {row['cwd']}")
                record = {
                    "collection": "chats",
                    "content": (
                        f"[chats] {session_id} > {section}\n"
                        + "\n".join(metadata)
                        + f"\n\n{text}"
                    ),
                    "file": stored_file,
                    "line": ordinal,
                    "name": session_id,
                    "raw_content": text,
                    "section": section,
                    "tags": ["agent-history-central", source, role],
                    "title": session_id,
                    "unit_type": "message",
                }
                record["canonical_id"] = _canonical_id(record)
                yield session_key, record
    finally:
        database_path.unlink(missing_ok=True)


@contextmanager
def _archive_generation_lock(backup_root: Path, already_held: bool):
    if already_held:
        yield
        return
    lock_path = (
        Path.home()
        / "Library"
        / "Application Support"
        / "agent-history-archive"
        / "generation.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(lock_path.parent, 0o700)
    acquired = False
    for attempt in range(120):
        result = subprocess.run(
            ["/usr/bin/shlock", "-f", str(lock_path), "-p", str(os.getpid())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode == 0:
            acquired = True
            break
        if attempt < 119:
            time.sleep(1)
    if not acquired:
        raise TimeoutError("central history generation remained locked")
    try:
        yield
    finally:
        try:
            if lock_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
                lock_path.unlink(missing_ok=True)
        except FileNotFoundError:
            pass


def _validate_source_plan(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != SOURCES_SCHEMA:
        raise ValueError("source plan must be a schema-1 object")
    sources = value.get("sources")
    if not isinstance(sources, list):
        raise ValueError("source plan sources must be a list")
    ids: set[str] = set()
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("source_id"), str):
            raise ValueError("source plan contains an invalid source")
        source_id = source["source_id"]
        if source_id in ids:
            raise ValueError(f"duplicate source plan source_id: {source_id}")
        ids.add(source_id)
    return value


def export_corpus(
    *,
    generation_id: str,
    wiki_root: Path,
    chats_root: Path,
    output_dir: Path,
    exported_at: datetime | None = None,
    source_plan: dict[str, Any] | None = None,
    archive_lock_held: bool = False,
) -> dict[str, Any]:
    """Export the default wiki, wiki:source, and chats corpus."""
    generation_id = validate_generation_id(generation_id)
    output_dir = validate_output_directory(output_dir)
    validated_plan = (
        _validate_source_plan(source_plan)
        if source_plan is not None
        else {"schema": SOURCES_SCHEMA, "sources": []}
    )
    readable_root = (
        prepare_chats(chats_root.expanduser().resolve())
        if validated_plan.get("include_local_chats", True)
        else chats_root.expanduser().resolve() / "readable"
    )

    corpus_path = output_dir / CORPUS_FILENAME
    digest = hashlib.sha256()
    corpus_count = 0
    canonical_ids: set[str] = set()
    source_file_counts = {collection: 0 for collection in COLLECTIONS}
    collection_files: dict[str, set[str]] = {
        collection: set() for collection in COLLECTIONS
    }
    collection_document_counts = {collection: 0 for collection in COLLECTIONS}
    source_receipts = {
        source["source_id"]: {**source, "exported_document_count": 0}
        for source in validated_plan["sources"]
    }
    source_files: dict[str, set[str]] = {
        source_id: set() for source_id in source_receipts
    }
    roots_and_specs = (
        (wiki_root.expanduser().resolve(), WIKI_SPEC),
        (readable_root, CHAT_SPEC),
    )
    with corpus_path.open("wb") as corpus:
        for collection, relative_file, root, spec, entry in _source_entries(
            roots_and_specs
        ):
            file_has_records = False
            for record in _records_for_entry(root, spec, relative_file, entry):
                canonical_id = record["canonical_id"]
                if canonical_id in canonical_ids:
                    raise ValueError(f"duplicate canonical_id: {canonical_id}")
                canonical_ids.add(canonical_id)
                line = (_canonical_json(record) + "\n").encode("utf-8")
                corpus.write(line)
                digest.update(line)
                corpus_count += 1
                collection_document_counts[collection] += 1
                file_has_records = True
                base_source_id = (
                    "wiki-pages"
                    if collection == "wiki"
                    else "wiki-raw"
                    if collection == "wiki:source"
                    else "local-readable-chats"
                )
                if base_source_id in source_receipts:
                    source_receipts[base_source_id]["exported_document_count"] += 1
                    source_files[base_source_id].add(relative_file)
                normalized = relative_file.replace("\\", "/")
                represented = {
                    "github-stars": (
                        "github-star" in normalized
                        or normalized == "wiki/note-github-starred-repositories.md"
                    ),
                    "reddit-saved": "reddit-saved" in normalized,
                }
                for source_id, matches in represented.items():
                    if matches and source_id in source_receipts:
                        source_receipts[source_id]["exported_document_count"] += 1
                        source_files[source_id].add(relative_file)
            if file_has_records:
                source_file_counts[collection] += 1
                collection_files[collection].add(relative_file)

        for source in validated_plan["sources"]:
            source_id = source["source_id"]
            for export_root in source.get("export_roots", []):
                root = Path(export_root["path"]).expanduser().resolve()
                collection = export_root["collection"]
                prefix = export_root["prefix"]
                kind = export_root.get("kind", "markdown")
                iterator = (
                    _markdown_records(
                        root=root,
                        collection=collection,
                        stored_prefix=prefix,
                    )
                    if kind == "markdown"
                    else _jsonl_records(
                        root=root,
                        collection=collection,
                        stored_prefix=prefix,
                        include=(
                            set(export_root["include"])
                            if export_root.get("include")
                            else None
                        ),
                    )
                )
                for relative_file, record in iterator:
                    canonical_id = record["canonical_id"]
                    if canonical_id in canonical_ids:
                        raise ValueError(f"duplicate canonical_id: {canonical_id}")
                    canonical_ids.add(canonical_id)
                    line = (_canonical_json(record) + "\n").encode("utf-8")
                    corpus.write(line)
                    digest.update(line)
                    corpus_count += 1
                    collection_document_counts[collection] += 1
                    source_receipts[source_id]["exported_document_count"] += 1
                    source_files[source_id].add(
                        f"{export_root['path']}:{relative_file}"
                    )
                    collection_key = (
                        f"{source_id}:{export_root['path']}:{relative_file}"
                    )
                    if collection_key not in collection_files[collection]:
                        collection_files[collection].add(collection_key)
                        source_file_counts[collection] += 1

        history_source = source_receipts.get("agent-history-central")
        if history_source is not None:
            backup_root = Path(history_source["root"]).expanduser().resolve()
            with _archive_generation_lock(backup_root, archive_lock_held):
                for session_key, record in _history_records(backup_root):
                    canonical_id = record["canonical_id"]
                    if canonical_id in canonical_ids:
                        raise ValueError(f"duplicate canonical_id: {canonical_id}")
                    canonical_ids.add(canonical_id)
                    line = (_canonical_json(record) + "\n").encode("utf-8")
                    corpus.write(line)
                    digest.update(line)
                    corpus_count += 1
                    collection_document_counts["chats"] += 1
                    history_source["exported_document_count"] += 1
                    source_files["agent-history-central"].add(session_key)
                    collection_key = f"agent-history-central:{session_key}"
                    if collection_key not in collection_files["chats"]:
                        collection_files["chats"].add(collection_key)
                        source_file_counts["chats"] += 1

    for source_id, receipt in source_receipts.items():
        receipt["source_file_count"] = len(source_files[source_id]) or receipt.get(
            "source_file_count", 0
        )
        receipt.pop("export_roots", None)
    if corpus_count == 0:
        raise ValueError("production corpus is empty")

    corpus_hash = digest.hexdigest()
    timestamp = exported_at or datetime.now(timezone.utc)
    manifest = {
        "schema": SCHEMA,
        "generation_id": generation_id,
        "model": DEFAULT_MODEL,
        "corpus_count": corpus_count,
        "corpus_hash": corpus_hash,
        "source_file_counts": source_file_counts,
        "collection_counts": {
            collection: {
                "documents": collection_document_counts[collection],
                "files": source_file_counts[collection],
            }
            for collection in COLLECTIONS
        },
        "sources": {
            "schema": SOURCES_SCHEMA,
            "items": [
                source_receipts[source_id] for source_id in sorted(source_receipts)
            ],
        },
        "exported_at": timestamp.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest_path.write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
    return {
        "schema": SCHEMA,
        "generation_id": generation_id,
        "corpus_path": str(corpus_path),
        "manifest_path": str(manifest_path),
        "corpus_count": corpus_count,
        "corpus_hash": corpus_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--wiki-root", required=True, type=Path)
    parser.add_argument("--chats-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-plan-json")
    parser.add_argument("--archive-lock-held", action="store_true")
    args = parser.parse_args()
    source_plan = (
        json.loads(args.source_plan_json) if args.source_plan_json else None
    )
    result = export_corpus(
        generation_id=args.generation_id,
        wiki_root=args.wiki_root,
        chats_root=args.chats_root,
        output_dir=args.output_dir,
        source_plan=source_plan,
        archive_lock_held=args.archive_lock_held,
    )
    print(_canonical_json(result))


if __name__ == "__main__":
    main()
