from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agentkb.modal_backend.generations import (
    generation_path,
    install_generation,
    publish_pointer,
    read_pointer,
    staged_paths,
    validate_generation_id,
)


FIRST = "g-20260725T120000Z-001122aabbcc"
SECOND = "g-20260725T130000Z-ddeeff001122"


def _local_generation(root: Path, generation_id: str) -> Path:
    local = root / f"local-{generation_id}"
    (local / "index").mkdir(parents=True)
    (local / "index" / "complete").write_text("yes")
    (local / "manifest.json").write_text(
        json.dumps({"schema": 1, "generation_id": generation_id})
    )
    return local


def _install_directory(root: Path, generation_id: str) -> None:
    directory = generation_path(root, generation_id)
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(
        json.dumps({"schema": 1, "generation_id": generation_id})
    )


def test_generation_id_is_validated_before_paths(tmp_path):
    assert staged_paths(tmp_path, FIRST)[0] == (
        tmp_path / "staged" / FIRST / "corpus.jsonl"
    )
    for invalid in ("../escape", "/absolute", FIRST.upper(), "g-short"):
        with pytest.raises(ValueError):
            validate_generation_id(invalid)
        with pytest.raises(ValueError):
            staged_paths(tmp_path, invalid)
        with pytest.raises(ValueError):
            generation_path(tmp_path, invalid)


def test_publish_pointer_is_atomic_and_tracks_previous(tmp_path, monkeypatch):
    _install_directory(tmp_path, FIRST)
    _install_directory(tmp_path, SECOND)
    fixed = lambda: datetime(2026, 7, 25, 12, tzinfo=timezone.utc)
    first, _ = publish_pointer(tmp_path, FIRST, now=fixed)

    replacements = []
    real_replace = __import__("os").replace

    def observe_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(
        "agentkb.modal_backend.generations.os.replace", observe_replace
    )
    second, old = publish_pointer(tmp_path, SECOND, now=fixed)

    assert old == first
    assert second["current_generation_id"] == SECOND
    assert second["previous_generation_id"] == FIRST
    assert read_pointer(tmp_path) == second
    assert replacements[-1][1] == tmp_path / "current.json"
    assert replacements[-1][0].parent == tmp_path


def test_validation_failure_leaves_no_generation_or_pointer(tmp_path):
    local = _local_generation(tmp_path, FIRST)

    def fail_validation(_):
        raise ValueError("copied index is incomplete")

    with pytest.raises(ValueError, match="incomplete"):
        install_generation(
            tmp_path,
            FIRST,
            local,
            validate_copy=fail_validation,
            commit=lambda: None,
        )

    assert not generation_path(tmp_path, FIRST).exists()
    assert read_pointer(tmp_path) is None
    assert list((tmp_path / "generations").iterdir()) == []


def test_commit_failure_restores_existing_pointer(tmp_path):
    _install_directory(tmp_path, FIRST)
    publish_pointer(tmp_path, FIRST)
    original = read_pointer(tmp_path)
    local = _local_generation(tmp_path, SECOND)
    calls = 0

    def fail_then_compensate():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("commit failed")

    with pytest.raises(RuntimeError, match="commit failed"):
        install_generation(
            tmp_path,
            SECOND,
            local,
            validate_copy=lambda path: (
                path / "index" / "complete"
            ).read_text(),
            commit=fail_then_compensate,
        )

    assert read_pointer(tmp_path) == original
    assert calls == 2
    assert not generation_path(tmp_path, SECOND).exists()


def test_existing_generation_is_never_overwritten(tmp_path):
    _install_directory(tmp_path, FIRST)
    marker = generation_path(tmp_path, FIRST) / "marker"
    marker.write_text("original")

    with pytest.raises(FileExistsError):
        install_generation(
            tmp_path,
            FIRST,
            _local_generation(tmp_path, FIRST),
            validate_copy=lambda _: None,
            commit=lambda: None,
        )
    assert marker.read_text() == "original"
