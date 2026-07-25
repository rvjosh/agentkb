"""Filesystem contract for immutable Modal index generations."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


GENERATION_ID_RE = re.compile(r"^g-\d{8}T\d{6}Z-[0-9a-f]{12}$")
POINTER_SCHEMA = 1


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
    previous = (
        None if previous_raw is None else validate_generation_id(previous_raw)
    )
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


def read_generation_manifest(
    volume_root: Path, generation_id: str
) -> dict[str, Any]:
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


def _validate_pointer_references(
    volume_root: Path, pointer: dict[str, Any]
) -> None:
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
        raise ValueError(
            "target generation must exactly equal previous_generation_id"
        )
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
    if generation_id in (
        before_delete["current_generation_id"],
        before_delete["previous_generation_id"],
    ):
        raise RuntimeError(
            "target generation became referenced after clearing; refusing deletion"
        )
    _validate_pointer_references(volume_root, before_delete)
    if not target.is_dir():
        raise RuntimeError("target generation disappeared before deletion")
    read_generation_manifest(volume_root, generation_id)

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
