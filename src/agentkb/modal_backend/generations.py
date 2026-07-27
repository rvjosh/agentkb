"""Filesystem contract for immutable Modal index generations."""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

GENERATION_ID_RE = re.compile(r"^g-\d{8}T\d{6}Z-[0-9a-f]{12}$")
POINTER_SCHEMA = 1
INVENTORY_SCHEMA = 1
MAX_GENERATION_ENTRIES = 1_000
MAX_RECEIPTS = 10_000
MAX_METADATA_DB_BYTES = 2 * 1024 * 1024 * 1024
MAX_STAGED_CORPUS_BYTES = 512 * 1024 * 1024
MAX_STAGED_RECORDS = 1_000_000
MAX_STAGED_LINE_BYTES = 4 * 1024 * 1024
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")
SOURCES = {"claude", "codex"}


def validate_generation_id(value: object) -> str:
    if not isinstance(value, str) or not GENERATION_ID_RE.fullmatch(value):
        raise ValueError(
            "generation_id must match g-YYYYMMDDTHHMMSSZ-<12 lowercase hex>"
        )
    return value


def staged_paths(volume_root: Path, generation_id: str) -> tuple[Path, Path]:
    generation_id = validate_generation_id(generation_id)
    root = volume_root / "staged" / generation_id
    return root / "corpus.jsonl", root / "manifest.json"


def generation_path(volume_root: Path, generation_id: str) -> Path:
    return volume_root / "generations" / validate_generation_id(generation_id)


def pointer_path(volume_root: Path) -> Path:
    return volume_root / "current.json"


def _safe_directory(root: Path, name: str, *, create: bool = False) -> Path:
    path = root / name
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if path.exists() and not path.is_dir():
        raise ValueError(f"{name} must be a directory")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{name} escapes the configured volume root") from exc
    return path


def _bounded_children(path: Path, *, limit: int) -> list[Path]:
    if not path.exists():
        return []
    children: list[Path] = []
    for child in path.iterdir():
        children.append(child)
        if len(children) > limit:
            raise ValueError(f"{path.name} contains more than {limit} entries")
    return sorted(children, key=lambda child: child.name)


def _validate_inventory_directory(
    volume_root: Path, child: Path, *, staged: bool
) -> dict[str, Any]:
    kind = "staged generation" if staged else "generation"
    if child.is_symlink() or not child.is_dir():
        raise ValueError(f"{kind} entry must be a real directory: {child.name}")
    generation_id = validate_generation_id(child.name)
    expected = volume_root / ("staged" if staged else "generations") / generation_id
    if child.resolve() != expected.resolve():
        raise ValueError(f"{kind} path escapes its configured root: {generation_id}")
    manifest_path = child / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"{kind} is missing a real manifest: {generation_id}")
    manifest = _read_json_object(manifest_path)
    if manifest.get("generation_id") != generation_id:
        raise ValueError(f"{kind} manifest does not match: {generation_id}")
    if staged:
        corpus = child / "corpus.jsonl"
        if corpus.is_symlink() or not corpus.is_file():
            raise ValueError(
                f"staged generation is missing a real corpus: {generation_id}"
            )
    else:
        database = child / "index" / "metadata.db"
        if (
            (child / "index").is_symlink()
            or database.is_symlink()
            or not database.is_file()
        ):
            raise ValueError(
                f"generation is missing a real metadata database: {generation_id}"
            )
    return {
        "generation_id": generation_id,
        "type": "staged" if staged else "generation",
    }


def inventory_generations(
    volume_root: Path, *, max_entries: int = MAX_GENERATION_ENTRIES
) -> dict[str, Any]:
    """Return a bounded, metadata-only inventory or reject ambiguous state."""
    # Modal may expose the configured mount root itself through a symlink.
    # Child directories and every target still reject symlinks and are resolved
    # beneath this canonical root.
    if not volume_root.is_dir():
        raise ValueError("configured volume root must resolve to a directory")
    if (
        not isinstance(max_entries, int)
        or isinstance(max_entries, bool)
        or max_entries < 1
        or max_entries > MAX_GENERATION_ENTRIES
    ):
        raise ValueError(f"max_entries must be between 1 and {MAX_GENERATION_ENTRIES}")
    pointer = read_pointer(volume_root)
    current = pointer["current_generation_id"] if pointer else None
    previous = pointer["previous_generation_id"] if pointer else None
    if current is not None and current == previous:
        raise ValueError("current and previous generation IDs are ambiguous")

    generations_root = _safe_directory(volume_root, "generations")
    staged_root = _safe_directory(volume_root, "staged")
    built_children = _bounded_children(generations_root, limit=max_entries)
    staged_children = _bounded_children(staged_root, limit=max_entries)
    if len(built_children) + len(staged_children) > max_entries:
        raise ValueError(f"generation inventory exceeds {max_entries} entries")
    built = [
        _validate_inventory_directory(volume_root, child, staged=False)
        for child in built_children
    ]
    staged = [
        _validate_inventory_directory(volume_root, child, staged=True)
        for child in staged_children
    ]
    built_ids = {item["generation_id"] for item in built}
    staged_ids = {item["generation_id"] for item in staged}
    overlap = built_ids & staged_ids
    if overlap:
        raise ValueError(
            f"generation ID exists in built and staged state: {sorted(overlap)[0]}"
        )
    if current is not None and current not in built_ids:
        raise ValueError("current pointer does not identify an inventoried generation")
    if previous is not None and previous not in built_ids:
        raise ValueError("previous pointer does not identify an inventoried generation")

    for item in built:
        generation_id = item["generation_id"]
        item["classification"] = (
            "current"
            if generation_id == current
            else "previous"
            if generation_id == previous
            else "orphan"
        )
    for item in staged:
        item["classification"] = "staged"
    items = sorted(
        [*built, *staged],
        key=lambda item: (item["generation_id"], item["type"]),
    )
    counts = {
        classification: sum(item["classification"] == classification for item in items)
        for classification in ("current", "previous", "orphan", "staged")
    }
    return {
        "schema": INVENTORY_SCHEMA,
        "current_generation_id": current,
        "previous_generation_id": previous,
        "items": items,
        "counts": counts,
    }


def validate_session_key(source: object, session_id: object) -> tuple[str, str, str]:
    if source not in SOURCES:
        raise ValueError("source must be exactly claude or codex")
    if not isinstance(session_id, str) or not SESSION_ID_RE.fullmatch(session_id):
        raise ValueError("session_id must be a safe canonical identifier")
    stored_file = f"agent-history-central/{source}/{session_id}.md"
    return str(source), session_id, stored_file


def _scan_built_generation(database: Path, stored_file: str) -> int:
    if database.stat().st_size > MAX_METADATA_DB_BYTES:
        raise ValueError("metadata.db exceeds the bounded scan size")
    uri = f"{database.resolve().as_uri()}?mode=ro&immutable=1"
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            relation = connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'documents'
                """
            ).fetchone()
            if relation is None:
                raise ValueError("metadata.db is missing documents")
            row = connection.execute(
                "SELECT COUNT(*) FROM documents WHERE file = ?", (stored_file,)
            ).fetchone()
    except sqlite3.Error as exc:
        raise ValueError(f"cannot verify metadata.db: {exc}") from exc
    assert row is not None
    return int(row[0])


def _scan_staged_corpus(corpus: Path, stored_file: str) -> tuple[int, int]:
    if corpus.stat().st_size > MAX_STAGED_CORPUS_BYTES:
        raise ValueError("staged corpus exceeds the bounded scan size")
    matches = 0
    records = 0
    with corpus.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if len(line) > MAX_STAGED_LINE_BYTES:
                raise ValueError(f"staged corpus line {line_number} is oversized")
            if not line.endswith(b"\n") or line == b"\n":
                raise ValueError(f"staged corpus line {line_number} is malformed")
            records += 1
            if records > MAX_STAGED_RECORDS:
                raise ValueError("staged corpus exceeds the bounded record count")
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"staged corpus line {line_number} is malformed"
                ) from exc
            if not isinstance(record, dict) or not isinstance(record.get("file"), str):
                raise ValueError(f"staged corpus line {line_number} has no exact file")
            if record["file"] == stored_file:
                matches += 1
    return matches, records


def find_session_presence(
    volume_root: Path, source: object, session_id: object
) -> dict[str, Any]:
    """Verify exact stored-file presence without returning transcript content."""
    source, session_id, stored_file = validate_session_key(source, session_id)
    inventory = inventory_generations(volume_root)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    total_matches = 0
    for item in inventory["items"]:
        generation_id = item["generation_id"]
        try:
            if item["type"] == "generation":
                matches = _scan_built_generation(
                    generation_path(volume_root, generation_id)
                    / "index"
                    / "metadata.db",
                    stored_file,
                )
                scanned_records = None
            else:
                corpus, _ = staged_paths(volume_root, generation_id)
                matches, scanned_records = _scan_staged_corpus(corpus, stored_file)
            result = {
                **item,
                "exact_match_count": matches,
                "verified": True,
            }
            if scanned_records is not None:
                result["scanned_record_count"] = scanned_records
            results.append(result)
            total_matches += matches
        except (OSError, ValueError) as exc:
            failure = {
                "generation_id": generation_id,
                "type": item["type"],
                "classification": item["classification"],
                "error": str(exc),
            }
            failures.append(failure)
            results.append(
                {
                    **item,
                    "exact_match_count": 0,
                    "verified": False,
                }
            )
    return {
        "schema": 1,
        "source": source,
        "session_id": session_id,
        "canonical_file": stored_file,
        "results": results,
        "total_exact_match_count": total_matches,
        "verification_failures": failures,
        "verified": not failures,
    }


def _receipt_root(volume_root: Path, *, create: bool = False) -> Path:
    return _safe_directory(volume_root, "deletion-receipts", create=create)


_INTENT_FIELDS = {
    "schema",
    "operation_id",
    "state",
    "target_id",
    "target_type",
    "expected_current_generation_id",
    "classification",
    "actor",
    "reason",
    "exact_session_key",
    "started_at",
    "counts",
}
_COMPLETE_FIELDS = _INTENT_FIELDS | {"finished_at", "verification"}


def _validate_operation_id(value: object, operation_dir: Path) -> str:
    try:
        parsed_operation = uuid.UUID(str(value))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("deletion receipt has an invalid operation ID") from exc
    operation_id = str(parsed_operation)
    if operation_id != value or operation_dir.name != operation_id:
        raise ValueError("deletion receipt operation ID does not match its path")
    return operation_id


def _validate_timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"deletion receipt has an invalid {field}")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"deletion receipt has an invalid {field}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"deletion receipt has an invalid {field}")
    return parsed


def _validate_receipt_identity(receipt: dict[str, Any], operation_dir: Path) -> None:
    if receipt.get("schema") != 1 or isinstance(receipt.get("schema"), bool):
        raise ValueError("deletion receipt has an unsupported schema")
    _validate_operation_id(receipt.get("operation_id"), operation_dir)
    target_id = validate_generation_id(receipt.get("target_id"))
    expected_current = validate_generation_id(
        receipt.get("expected_current_generation_id")
    )
    if target_id == expected_current:
        raise ValueError("deletion receipt target must not be current")
    if receipt.get("target_type") not in {"generation", "staged"}:
        raise ValueError("deletion receipt has an invalid target type")
    if receipt.get("classification") not in {"previous", "orphan", "staged"}:
        raise ValueError("deletion receipt has an invalid classification")
    if (receipt["target_type"] == "staged") != (receipt["classification"] == "staged"):
        raise ValueError("deletion receipt classification does not match target type")
    for field, limit in (("actor", 200), ("reason", 1_000)):
        value = receipt.get(field)
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or len(value) > limit
        ):
            raise ValueError(f"deletion receipt has an invalid {field}")
    exact_session_key = receipt.get("exact_session_key")
    if exact_session_key is not None:
        if not isinstance(exact_session_key, str):
            raise ValueError("deletion receipt has an invalid exact session key")
        parts = exact_session_key.split("/", 1)
        if len(parts) != 2:
            raise ValueError("deletion receipt has an invalid exact session key")
        validate_session_key(parts[0], parts[1])
    _validate_timestamp(receipt.get("started_at"), "started_at")


def _validate_intent(intent: dict[str, Any], operation_dir: Path) -> dict[str, Any]:
    if set(intent) != _INTENT_FIELDS or intent.get("state") != "intent":
        raise ValueError("deletion intent has invalid fields or state")
    _validate_receipt_identity(intent, operation_dir)
    counts = intent.get("counts")
    if (
        not isinstance(counts, dict)
        or set(counts) != {"exact_match_count"}
        or not isinstance(counts.get("exact_match_count"), int)
        or isinstance(counts.get("exact_match_count"), bool)
        or counts["exact_match_count"] < 0
    ):
        raise ValueError("deletion intent has an invalid exact-match count")
    return intent


def _validate_completion(
    complete: dict[str, Any],
    intent: dict[str, Any],
    operation_dir: Path,
) -> dict[str, Any]:
    if set(complete) != _COMPLETE_FIELDS or complete.get("state") != "complete":
        raise ValueError("deletion completion has invalid fields or state")
    _validate_receipt_identity(complete, operation_dir)
    for field in _INTENT_FIELDS - {"state", "counts"}:
        if complete[field] != intent[field]:
            raise ValueError("deletion completion does not derive from its intent")
    counts = complete.get("counts")
    verification = complete.get("verification")
    if (
        not isinstance(counts, dict)
        or set(counts) != {"directories_deleted", "exact_match_count"}
        or counts.get("directories_deleted") != 1
        or isinstance(counts.get("directories_deleted"), bool)
        or counts.get("exact_match_count") != intent["counts"]["exact_match_count"]
        or not isinstance(verification, dict)
        or set(verification)
        != {
            "target_absent",
            "pointer_consistent",
            "current_generation_id",
            "previous_generation_id",
            "exact_match_count",
        }
        or verification.get("target_absent") is not True
        or verification.get("pointer_consistent") is not True
        or verification.get("current_generation_id")
        != intent["expected_current_generation_id"]
        or verification.get("exact_match_count")
        != intent["counts"]["exact_match_count"]
    ):
        raise ValueError("deletion completion has invalid counts or verification")
    previous = verification["previous_generation_id"]
    if previous is not None:
        previous = validate_generation_id(previous)
    if previous == intent["target_id"]:
        raise ValueError("deletion completion still references its target")
    finished_at = _validate_timestamp(complete.get("finished_at"), "finished_at")
    if finished_at < _validate_timestamp(intent["started_at"], "started_at"):
        raise ValueError("deletion completion predates its intent")
    return complete


def _load_deletion_operations(
    volume_root: Path,
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    root = _receipt_root(volume_root)
    operations: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    for operation_dir in _bounded_children(root, limit=MAX_RECEIPTS):
        if operation_dir.is_symlink() or not operation_dir.is_dir():
            raise ValueError(
                "deletion receipt root contains an invalid operation entry"
            )
        _validate_operation_id(operation_dir.name, operation_dir)
        entries = _bounded_children(operation_dir, limit=2)
        if any(entry.is_symlink() or not entry.is_file() for entry in entries):
            raise ValueError(
                "deletion receipt operation contains a symlink or non-file"
            )
        if {entry.name for entry in entries} not in (
            {"intent.json"},
            {"intent.json", "complete.json"},
        ):
            raise ValueError("deletion receipt operation has malformed entries")
        intent = _validate_intent(
            _read_json_object(operation_dir / "intent.json"), operation_dir
        )
        complete_path = operation_dir / "complete.json"
        complete = (
            _validate_completion(
                _read_json_object(complete_path), intent, operation_dir
            )
            if complete_path.exists()
            else None
        )
        operations.append((intent, complete))
    return operations


def _write_operation_file(path: Path, value: dict[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _operation_identity(receipt: dict[str, Any]) -> tuple[object, ...]:
    return (
        receipt["target_id"],
        receipt["target_type"],
        receipt["expected_current_generation_id"],
        receipt["actor"],
        receipt["reason"],
        receipt["exact_session_key"],
    )


def _requested_identity(
    target_id: str,
    target_type: str,
    expected_current: str,
    actor: str,
    reason: str,
    exact_session_key: str | None,
) -> tuple[object, ...]:
    return (
        target_id,
        target_type,
        expected_current,
        actor,
        reason,
        exact_session_key,
    )


def _require_unique_active_intent(
    operations: list[tuple[dict[str, Any], dict[str, Any] | None]],
    intent: dict[str, Any],
) -> None:
    exact = [
        operation
        for operation in operations
        if _operation_identity(operation[0]) == _operation_identity(intent)
    ]
    if exact != [(intent, None)]:
        raise ValueError("deletion intent is not the unique active exact operation")
    if any(
        other_intent["operation_id"] != intent["operation_id"]
        and other_complete is None
        and other_intent["target_id"] == intent["target_id"]
        and other_intent["target_type"] == intent["target_type"]
        and other_intent["expected_current_generation_id"]
        == intent["expected_current_generation_id"]
        for other_intent, other_complete in operations
    ):
        raise ValueError("conflicting incomplete deletion operation exists")


def _result_for_receipt(
    receipt: dict[str, Any],
    *,
    deleted: bool,
    idempotent: bool,
    dry_run: bool = False,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "dry_run": dry_run,
        "deleted": deleted,
        "idempotent": idempotent,
        "target_id": receipt["target_id"],
        "target_type": receipt["target_type"],
        "classification": receipt["classification"],
        "current_generation_id": receipt["expected_current_generation_id"],
        "operation_id": receipt["operation_id"],
        "receipt": receipt,
    }


def _timestamp(now: Callable[[], datetime] | None) -> str:
    return (
        (now or (lambda: datetime.now(timezone.utc)))()
        .astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _complete_deletion_operation(
    volume_root: Path,
    intent: dict[str, Any],
    *,
    target: Path,
    commit: Callable[[], None],
    reload: Callable[[], None],
    now: Callable[[], datetime] | None,
) -> dict[str, Any]:
    expected_current = intent["expected_current_generation_id"]
    target_id = intent["target_id"]
    reload()
    pointer = read_pointer(volume_root, required=True)
    assert pointer is not None
    _require_unique_active_intent(_load_deletion_operations(volume_root), intent)
    if target.exists() or pointer["current_generation_id"] != expected_current:
        raise RuntimeError("cannot complete deletion without verified target absence")
    if target_id in (
        pointer["current_generation_id"],
        pointer["previous_generation_id"],
    ):
        raise RuntimeError("cannot complete deletion while pointer references target")
    complete = {
        **intent,
        "state": "complete",
        "finished_at": _timestamp(now),
        "counts": {
            "directories_deleted": 1,
            "exact_match_count": intent["counts"]["exact_match_count"],
        },
        "verification": {
            "target_absent": True,
            "pointer_consistent": True,
            "current_generation_id": pointer["current_generation_id"],
            "previous_generation_id": pointer["previous_generation_id"],
            "exact_match_count": intent["counts"]["exact_match_count"],
        },
    }
    operation_dir = _receipt_root(volume_root) / intent["operation_id"]
    complete_path = operation_dir / "complete.json"
    _validate_completion(complete, intent, operation_dir)
    _write_operation_file(complete_path, complete)
    try:
        commit()
    except BaseException:
        complete_path.unlink(missing_ok=True)
        raise
    reload()
    durable = next(
        (
            saved_complete
            for saved_intent, saved_complete in _load_deletion_operations(volume_root)
            if saved_intent["operation_id"] == intent["operation_id"]
        ),
        None,
    )
    if durable != complete:
        raise RuntimeError("completed deletion receipt did not persist")
    final_pointer = read_pointer(volume_root, required=True)
    assert final_pointer is not None
    if (
        final_pointer["current_generation_id"] != expected_current
        or target.exists()
        or target_id
        in (
            final_pointer["current_generation_id"],
            final_pointer["previous_generation_id"],
        )
    ):
        raise RuntimeError("completed deletion verification did not persist")
    return complete


def delete_generation(
    volume_root: Path,
    target_id: str,
    *,
    target_type: str,
    expected_current_generation_id: str,
    force: bool,
    actor: str,
    reason: str,
    commit: Callable[[], None],
    reload: Callable[[], None] = lambda: None,
    exact_session_key: str | None = None,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Dry-run or delete one exact non-current built/staged generation."""
    target_id = validate_generation_id(target_id)
    expected_current = validate_generation_id(expected_current_generation_id)
    if target_type not in {"generation", "staged"}:
        raise ValueError("target_type must be generation or staged")
    if not isinstance(actor, str) or not actor.strip() or len(actor) > 200:
        raise ValueError("actor must be a non-empty bounded string")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 1_000:
        raise ValueError("reason must be a non-empty bounded string")
    actor = actor.strip()
    reason = reason.strip()
    session_key_parts: tuple[str, str] | None = None
    if exact_session_key is not None:
        parts = exact_session_key.split("/", 1)
        if len(parts) != 2:
            raise ValueError("exact_session_key must be source/session-id")
        validate_session_key(parts[0], parts[1])
        session_key_parts = (parts[0], parts[1])

    reload()
    pointer = read_pointer(volume_root, required=True)
    assert pointer is not None
    if pointer["current_generation_id"] != expected_current:
        raise ValueError("current generation drifted from expected current")
    if target_id == pointer["current_generation_id"]:
        raise ValueError("refusing to delete the current generation")
    operations = _load_deletion_operations(volume_root)
    requested_identity = _requested_identity(
        target_id,
        target_type,
        expected_current,
        actor,
        reason,
        exact_session_key,
    )
    exact_operations = [
        operation
        for operation in operations
        if _operation_identity(operation[0]) == requested_identity
    ]
    incomplete = [operation for operation in exact_operations if operation[1] is None]
    completed = [
        operation for operation in exact_operations if operation[1] is not None
    ]
    if len(incomplete) > 1:
        raise ValueError("duplicate matching incomplete deletion operations")
    if len(completed) > 1 or (incomplete and completed):
        raise ValueError("duplicate matching deletion operations")

    target = (
        generation_path(volume_root, target_id)
        if target_type == "generation"
        else staged_paths(volume_root, target_id)[0].parent
    )
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise ValueError("exact deletion target must be a real directory")
    pointer_references_target = target_id in (
        pointer["current_generation_id"],
        pointer["previous_generation_id"],
    )
    if completed:
        if target.exists() or pointer_references_target:
            raise ValueError("completed deletion target was recreated or re-referenced")
        complete = completed[0][1]
        assert complete is not None
        return _result_for_receipt(
            complete, deleted=False, idempotent=True, dry_run=not force
        )
    if not target.exists():
        if not incomplete:
            raise FileNotFoundError(
                "target is absent and no exact deletion receipt intent or completion exists"
            )
        if pointer_references_target:
            raise ValueError("absent deletion target remains pointer-referenced")
        intent = incomplete[0][0]
        if not force:
            return {
                "schema": 1,
                "dry_run": True,
                "deleted": False,
                "idempotent": False,
                "target_id": target_id,
                "target_type": target_type,
                "classification": intent["classification"],
                "current_generation_id": expected_current,
                "operation_id": intent["operation_id"],
                "receipt": None,
            }
        complete = _complete_deletion_operation(
            volume_root,
            intent,
            target=target,
            commit=commit,
            reload=reload,
            now=now,
        )
        return _result_for_receipt(complete, deleted=False, idempotent=True)

    inventory = inventory_generations(volume_root)
    matching = [
        item
        for item in inventory["items"]
        if item["generation_id"] == target_id and item["type"] == target_type
    ]
    if not matching:
        raise ValueError("exact target path does not match the requested target type")
    classification = matching[0]["classification"]
    if incomplete:
        original_classification = incomplete[0][0]["classification"]
        if not (
            classification == original_classification
            or (
                original_classification == "previous"
                and classification == "orphan"
                and pointer["previous_generation_id"] is None
            )
        ):
            raise ValueError("target classification conflicts with deletion intent")
        exact_match_count = incomplete[0][0]["counts"]["exact_match_count"]
    elif session_key_parts is not None:
        presence = find_session_presence(
            volume_root, session_key_parts[0], session_key_parts[1]
        )
        target_presence = next(
            (
                item
                for item in presence["results"]
                if item["generation_id"] == target_id and item["type"] == target_type
            ),
            None,
        )
        if (
            target_presence is None
            or not target_presence["verified"]
            or presence["verification_failures"]
        ):
            raise ValueError("exact-session verification failed closed")
        exact_match_count = target_presence["exact_match_count"]
        if exact_match_count < 1:
            raise ValueError("exact session is absent from the deletion target")
    else:
        exact_match_count = 0

    if classification == "current":
        raise ValueError("refusing to delete the current generation")
    if target_type == "generation" and classification not in {"previous", "orphan"}:
        raise ValueError("built generation has an invalid deletion classification")
    if target_type == "staged" and classification != "staged":
        raise ValueError("staged generation has an invalid deletion classification")
    if not force:
        return {
            "schema": 1,
            "dry_run": True,
            "deleted": False,
            "idempotent": False,
            "target_id": target_id,
            "target_type": target_type,
            "classification": classification,
            "current_generation_id": expected_current,
            "operation_id": (incomplete[0][0]["operation_id"] if incomplete else None),
            "receipt": None,
        }

    conflicting_incomplete = [
        intent
        for intent, complete in operations
        if complete is None
        and intent["target_id"] == target_id
        and intent["target_type"] == target_type
        and intent["expected_current_generation_id"] == expected_current
        and _operation_identity(intent) != requested_identity
    ]
    if conflicting_incomplete:
        raise ValueError("conflicting incomplete deletion operation exists")
    if not incomplete and any(
        intent["target_id"] == target_id and intent["target_type"] == target_type
        for intent, _complete in operations
    ):
        raise ValueError("target has another deletion operation and cannot be reused")

    if incomplete:
        intent = incomplete[0][0]
    else:
        # Re-read immediately before creating the durable intent, the first mutation.
        reload()
        immediate_pointer = read_pointer(volume_root, required=True)
        assert immediate_pointer is not None
        if immediate_pointer["current_generation_id"] != expected_current:
            raise ValueError("current generation drifted immediately before intent")
        if target_id == immediate_pointer["current_generation_id"]:
            raise ValueError("refusing to delete the current generation")
        immediate_inventory = inventory_generations(volume_root)
        immediate_match = next(
            (
                item
                for item in immediate_inventory["items"]
                if item["generation_id"] == target_id and item["type"] == target_type
            ),
            None,
        )
        if (
            immediate_match is None
            or immediate_match["classification"] != classification
        ):
            raise ValueError("target classification drifted immediately before intent")
        receipt_root = _receipt_root(volume_root, create=True)
        if len(_bounded_children(receipt_root, limit=MAX_RECEIPTS)) >= MAX_RECEIPTS:
            raise ValueError("deletion receipt operation limit reached")
        operation_id = str(uuid.uuid4())
        operation_dir = receipt_root / operation_id
        operation_dir.mkdir()
        intent = {
            "schema": 1,
            "operation_id": operation_id,
            "state": "intent",
            "target_id": target_id,
            "target_type": target_type,
            "expected_current_generation_id": expected_current,
            "classification": classification,
            "actor": actor,
            "reason": reason,
            "exact_session_key": exact_session_key,
            "started_at": _timestamp(now),
            "counts": {"exact_match_count": exact_match_count},
        }
        _validate_intent(intent, operation_dir)
        _write_operation_file(operation_dir / "intent.json", intent)
        try:
            commit()
        except BaseException:
            shutil.rmtree(operation_dir)
            raise
        reload()
        durable_operations = _load_deletion_operations(volume_root)
        _require_unique_active_intent(durable_operations, intent)

    # Re-read immediately before the first destructive mutation.
    reload()
    pointer = read_pointer(volume_root, required=True)
    assert pointer is not None
    if pointer["current_generation_id"] != expected_current:
        raise ValueError("current generation drifted immediately before deletion")
    if target_id == pointer["current_generation_id"]:
        raise ValueError("refusing to delete the current generation")
    _require_unique_active_intent(_load_deletion_operations(volume_root), intent)
    immediate_inventory = inventory_generations(volume_root)
    immediate_match = next(
        (
            item
            for item in immediate_inventory["items"]
            if item["generation_id"] == target_id and item["type"] == target_type
        ),
        None,
    )
    if immediate_match is None:
        raise RuntimeError("exact target disappeared immediately before deletion")
    immediate_classification = immediate_match["classification"]
    original_classification = intent["classification"]
    if not (
        immediate_classification == original_classification
        or (
            original_classification == "previous"
            and immediate_classification == "orphan"
            and pointer["previous_generation_id"] is None
        )
    ):
        raise ValueError("target classification drifted immediately before deletion")
    if original_classification == "previous":
        if immediate_classification == "previous":
            if pointer["previous_generation_id"] != target_id:
                raise ValueError("previous generation classification drifted")
            _atomic_write_json(
                pointer_path(volume_root), {**pointer, "previous_generation_id": None}
            )
            commit()
            reload()
            pointer = read_pointer(volume_root, required=True)
            assert pointer is not None
        if (
            pointer["current_generation_id"] != expected_current
            or pointer["previous_generation_id"] is not None
        ):
            raise RuntimeError("previous pointer clearing did not persist")
    elif pointer["previous_generation_id"] == target_id:
        raise ValueError("target became previous generation before deletion")
    if not target.is_dir() or target.is_symlink():
        raise RuntimeError("exact target disappeared or changed before deletion")
    shutil.rmtree(target)
    commit()
    reload()
    final_pointer = read_pointer(volume_root, required=True)
    assert final_pointer is not None
    if final_pointer["current_generation_id"] != expected_current:
        raise RuntimeError("current pointer changed during deletion")
    if target.exists():
        raise RuntimeError("target directory still exists after deletion")
    if target_id in (
        final_pointer["current_generation_id"],
        final_pointer["previous_generation_id"],
    ):
        raise RuntimeError("deleted target remains pointer-referenced")
    complete = _complete_deletion_operation(
        volume_root,
        intent,
        target=target,
        commit=commit,
        reload=reload,
        now=now,
    )
    return _result_for_receipt(complete, deleted=True, idempotent=False)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def read_pointer(volume_root: Path, *, required: bool = False) -> dict[str, Any] | None:
    path = pointer_path(volume_root)
    try:
        value = _read_json_object(path)
    except FileNotFoundError:
        if required:
            raise RuntimeError("AgentKB has no current generation; build one first")
        return None

    if value.get("schema") != POINTER_SCHEMA:
        raise ValueError("current.json has an unsupported schema")
    current = validate_generation_id(value.get("current_generation_id"))
    previous_raw = value.get("previous_generation_id")
    previous = None if previous_raw is None else validate_generation_id(previous_raw)
    published_at = value.get("published_at")
    if not isinstance(published_at, str) or not published_at:
        raise ValueError("current.json published_at must be a non-empty string")
    return {
        "schema": POINTER_SCHEMA,
        "current_generation_id": current,
        "previous_generation_id": previous,
        "published_at": published_at,
    }


def resolve_current(volume_root: Path) -> str:
    pointer = read_pointer(volume_root, required=True)
    assert pointer is not None
    return pointer["current_generation_id"]


def read_generation_manifest(volume_root: Path, generation_id: str) -> dict[str, Any]:
    path = generation_path(volume_root, generation_id) / "manifest.json"
    manifest = _read_json_object(path)
    if manifest.get("generation_id") != generation_id:
        raise ValueError(f"manifest generation_id does not match {generation_id}")
    return manifest


def read_status(volume_root: Path) -> dict[str, Any]:
    pointer = read_pointer(volume_root)
    if pointer is None:
        return {
            "schema": 1,
            "current_generation_id": None,
            "previous_generation_id": None,
            "published_at": None,
            "current_manifest": None,
            "previous_manifest": None,
        }
    current = pointer["current_generation_id"]
    previous = pointer["previous_generation_id"]
    return {
        **pointer,
        "current_manifest": read_generation_manifest(volume_root, current),
        "previous_manifest": (
            read_generation_manifest(volume_root, previous) if previous else None
        ),
    }


def _validate_pointer_references(volume_root: Path, pointer: dict[str, Any]) -> None:
    for field in ("current_generation_id", "previous_generation_id"):
        generation_id = pointer[field]
        if generation_id is None:
            continue
        if not generation_path(volume_root, generation_id).is_dir():
            raise RuntimeError(f"{field} generation directory does not exist")
        read_generation_manifest(volume_root, generation_id)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def publish_pointer(
    volume_root: Path,
    generation_id: str,
    *,
    now: Callable[[], datetime] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    generation_id = validate_generation_id(generation_id)
    if not generation_path(volume_root, generation_id).is_dir():
        raise FileNotFoundError(f"generation is not installed: {generation_id}")
    old_pointer = read_pointer(volume_root)
    pointer = {
        "schema": POINTER_SCHEMA,
        "current_generation_id": generation_id,
        "previous_generation_id": (
            old_pointer["current_generation_id"] if old_pointer else None
        ),
        "published_at": (now or (lambda: datetime.now(timezone.utc)))()
        .astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }
    _atomic_write_json(pointer_path(volume_root), pointer)
    return pointer, old_pointer


def restore_pointer(volume_root: Path, old_pointer: dict[str, Any] | None) -> None:
    path = pointer_path(volume_root)
    if old_pointer is None:
        path.unlink(missing_ok=True)
    else:
        _atomic_write_json(path, old_pointer)


def prune_previous_generation(
    volume_root: Path,
    generation_id: str,
    *,
    dry_run: bool,
    commit: Callable[[], None],
    reload: Callable[[], None] = lambda: None,
) -> dict[str, Any]:
    """Delete exactly the currently recorded previous generation, fail closed."""
    generation_id = validate_generation_id(generation_id)
    reload()
    pointer = read_pointer(volume_root, required=True)
    assert pointer is not None
    current = pointer["current_generation_id"]
    previous = pointer["previous_generation_id"]

    # This check deliberately precedes the previous-pointer check. A malformed
    # pointer that names the current generation twice must still refuse current.
    if generation_id == current:
        raise ValueError("refusing to prune the current generation")
    if generation_id != previous:
        raise ValueError("target generation must exactly equal previous_generation_id")
    target = generation_path(volume_root, generation_id)
    if not target.is_dir():
        raise FileNotFoundError(f"generation directory does not exist: {generation_id}")
    read_generation_manifest(volume_root, generation_id)
    _validate_pointer_references(volume_root, pointer)

    result = {
        "schema": 1,
        "dry_run": dry_run,
        "deleted": False,
        "target_generation_id": generation_id,
        "current_generation_id": current,
        "previous_generation_id": previous,
        "final_previous_generation_id": previous,
    }
    if dry_run:
        return result

    cleared_pointer = {**pointer, "previous_generation_id": None}
    _atomic_write_json(pointer_path(volume_root), cleared_pointer)
    commit()

    reload()
    before_delete = read_pointer(volume_root, required=True)
    assert before_delete is not None
    _validate_pointer_references(volume_root, before_delete)
    if not target.is_dir():
        raise RuntimeError("target generation disappeared before deletion")
    read_generation_manifest(volume_root, generation_id)
    if generation_id in (
        before_delete["current_generation_id"],
        before_delete["previous_generation_id"],
    ):
        raise RuntimeError(
            "target generation became referenced after clearing; refusing deletion"
        )

    shutil.rmtree(target)
    # If this commit fails, the already-committed cleared pointer remains the
    # safe state. Do not restore it or otherwise compensate.
    commit()

    reload()
    final_pointer = read_pointer(volume_root, required=True)
    assert final_pointer is not None
    if target.exists():
        raise RuntimeError("target generation still exists after deletion commit")
    if generation_id in (
        final_pointer["current_generation_id"],
        final_pointer["previous_generation_id"],
    ):
        raise RuntimeError("deleted generation is still referenced")
    _validate_pointer_references(volume_root, final_pointer)
    return {
        **result,
        "deleted": True,
        "final_previous_generation_id": final_pointer["previous_generation_id"],
    }


def install_generation(
    volume_root: Path,
    generation_id: str,
    local_generation: Path,
    *,
    validate_copy: Callable[[Path], None],
    commit: Callable[[], None],
) -> dict[str, Any]:
    """Install, publish, and commit one generation without overwriting a directory."""
    generation_id = validate_generation_id(generation_id)
    destination = generation_path(volume_root, generation_id)
    generations = destination.parent
    generations.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"generation already exists: {generation_id}")

    temporary = generations / f".{generation_id}.tmp-{uuid.uuid4().hex}"
    pointer_written = False
    destination_installed = False
    old_pointer: dict[str, Any] | None = None
    try:
        shutil.copytree(local_generation, temporary)
        validate_copy(temporary)
        os.rename(temporary, destination)
        destination_installed = True
        pointer, old_pointer = publish_pointer(volume_root, generation_id)
        pointer_written = True
        try:
            commit()
        except BaseException:
            restore_pointer(volume_root, old_pointer)
            pointer_written = False
            shutil.rmtree(destination)
            destination_installed = False
            try:
                commit()
            except BaseException:
                pass
            raise
        return pointer
    except BaseException:
        if pointer_written:
            restore_pointer(volume_root, old_pointer)
        if destination_installed and destination.exists():
            shutil.rmtree(destination)
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
