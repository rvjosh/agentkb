from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from pathlib import Path

import pytest

from agentkb.modal_backend import generations

CURRENT = "g-20260727T010000Z-001122aabbcc"
PREVIOUS = "g-20260727T000000Z-112233aabbcc"
ORPHAN = "g-20260726T230000Z-223344aabbcc"
STAGED = "g-20260727T020000Z-334455aabbcc"
UNRELATED = "agent-history-central/codex/unrelated.md"
TARGET = "agent-history-central/codex/target-session.md"


class InjectedFailure(RuntimeError):
    pass


def _built(root: Path, generation_id: str, files: list[str]) -> Path:
    directory = generations.generation_path(root, generation_id)
    (directory / "index").mkdir(parents=True)
    (directory / "manifest.json").write_text(
        json.dumps({"schema": 1, "generation_id": generation_id})
    )
    with sqlite3.connect(directory / "index" / "metadata.db") as connection:
        connection.execute(
            "CREATE TABLE documents (id INTEGER PRIMARY KEY, file TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO documents(file) VALUES (?)", [(item,) for item in files]
        )
    return directory


def _staged(root: Path, generation_id: str, files: list[str]) -> Path:
    corpus, manifest = generations.staged_paths(root, generation_id)
    corpus.parent.mkdir(parents=True)
    corpus.write_text("".join(json.dumps({"file": item}) + "\n" for item in files))
    manifest.write_text(json.dumps({"schema": 1, "generation_id": generation_id}))
    return corpus.parent


def _volume(root: Path) -> None:
    _built(root, PREVIOUS, [TARGET, UNRELATED])
    _built(root, CURRENT, [UNRELATED])
    _built(root, ORPHAN, [TARGET, TARGET, UNRELATED])
    _staged(root, STAGED, [TARGET, UNRELATED])
    generations.publish_pointer(root, PREVIOUS)
    generations.publish_pointer(root, CURRENT)


def test_inventory_classifies_every_generation_and_staging(tmp_path):
    _volume(tmp_path)
    result = generations.inventory_generations(tmp_path)
    assert result["counts"] == {
        "current": 1,
        "previous": 1,
        "orphan": 1,
        "staged": 1,
    }
    assert {
        (item["generation_id"], item["classification"]) for item in result["items"]
    } == {
        (CURRENT, "current"),
        (PREVIOUS, "previous"),
        (ORPHAN, "orphan"),
        (STAGED, "staged"),
    }
    assert "manifest" not in json.dumps(result)


def test_inventory_accepts_a_symlinked_mount_root_but_not_symlinked_children(
    tmp_path,
):
    real_root = tmp_path / "mounted-volume"
    real_root.mkdir()
    _volume(real_root)
    mount_root = tmp_path / "agentkb-data"
    mount_root.symlink_to(real_root, target_is_directory=True)
    assert generations.inventory_generations(mount_root)["counts"]["current"] == 1


@pytest.mark.parametrize("root_name", ["generations", "staged"])
def test_inventory_rejects_malformed_and_symlink_entries(tmp_path, root_name):
    _volume(tmp_path)
    bad = tmp_path / root_name / "not-a-generation"
    bad.mkdir()
    with pytest.raises(ValueError, match="generation_id"):
        generations.inventory_generations(tmp_path)
    bad.rmdir()
    (tmp_path / root_name / "not-a-generation").symlink_to(
        tmp_path / ("staged" if root_name == "generations" else "generations"),
        target_is_directory=True,
    )
    with pytest.raises(ValueError, match="real directory"):
        generations.inventory_generations(tmp_path)


def test_inventory_rejects_duplicate_and_unbounded_state(tmp_path):
    _volume(tmp_path)
    _staged(tmp_path, ORPHAN, [TARGET])
    with pytest.raises(ValueError, match="built and staged"):
        generations.inventory_generations(tmp_path)
    generations.staged_paths(tmp_path, ORPHAN)[0].parent.rename(tmp_path / "held")
    with pytest.raises(ValueError, match="exceeds"):
        generations.inventory_generations(tmp_path, max_entries=3)


def test_find_session_uses_exact_file_identity_across_all_corpora(tmp_path):
    _volume(tmp_path)
    result = generations.find_session_presence(tmp_path, "codex", "target-session")
    assert result["verified"] is True
    assert result["canonical_file"] == TARGET
    assert result["total_exact_match_count"] == 4
    assert {
        item["generation_id"]: item["exact_match_count"] for item in result["results"]
    } == {
        CURRENT: 0,
        PREVIOUS: 1,
        ORPHAN: 2,
        STAGED: 1,
    }
    assert TARGET not in json.dumps(result).replace(result["canonical_file"], "")


def test_find_session_fails_closed_for_malformed_and_oversized_staging(
    tmp_path, monkeypatch
):
    _volume(tmp_path)
    corpus, _ = generations.staged_paths(tmp_path, STAGED)
    corpus.write_text("{bad json}\n")
    malformed = generations.find_session_presence(tmp_path, "codex", "target-session")
    assert malformed["verified"] is False
    assert malformed["verification_failures"][0]["generation_id"] == STAGED
    assert malformed["total_exact_match_count"] == 3

    corpus.write_text(json.dumps({"file": TARGET}) + "\n")
    monkeypatch.setattr(generations, "MAX_STAGED_CORPUS_BYTES", 1)
    oversized = generations.find_session_presence(tmp_path, "codex", "target-session")
    assert oversized["verified"] is False
    assert "bounded scan size" in oversized["verification_failures"][0]["error"]


def test_exact_session_key_refuses_a_target_without_an_exact_match(tmp_path):
    _volume(tmp_path)
    clean_orphan = "g-20260726T220000Z-556677889900"
    _built(tmp_path, clean_orphan, [UNRELATED])
    with pytest.raises(ValueError, match="absent from the deletion target"):
        generations.delete_generation(
            tmp_path,
            clean_orphan,
            target_type="generation",
            expected_current_generation_id=CURRENT,
            force=True,
            actor="test",
            reason="wrong target",
            exact_session_key="codex/target-session",
            commit=lambda: pytest.fail("unrelated generation committed"),
        )


def test_delete_dry_run_is_immutable_and_refuses_current_or_drift(tmp_path):
    _volume(tmp_path)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    result = generations.delete_generation(
        tmp_path,
        ORPHAN,
        target_type="generation",
        expected_current_generation_id=CURRENT,
        force=False,
        actor="test",
        reason="dry run",
        commit=lambda: pytest.fail("dry run committed"),
    )
    assert result["dry_run"] is True
    assert result["deleted"] is False
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before
    assert not (tmp_path / "deletion-receipts").exists()

    for target, expected, message in (
        (CURRENT, CURRENT, "current"),
        (ORPHAN, PREVIOUS, "drifted"),
    ):
        with pytest.raises(ValueError, match=message):
            generations.delete_generation(
                tmp_path,
                target,
                target_type="generation",
                expected_current_generation_id=expected,
                force=True,
                actor="test",
                reason="refusal",
                commit=lambda: None,
            )


@pytest.mark.parametrize(
    ("target", "target_type", "classification"),
    [
        (PREVIOUS, "generation", "previous"),
        (ORPHAN, "generation", "orphan"),
        (STAGED, "staged", "staged"),
    ],
)
def test_delete_exact_target_receipt_idempotence_and_unrelated_survival(
    tmp_path, target, target_type, classification
):
    _volume(tmp_path)
    commits: list[tuple[str | None, bool]] = []

    def commit():
        commits.append(
            (
                generations.read_pointer(tmp_path)["previous_generation_id"],
                (generations.generation_path(tmp_path, CURRENT)).exists(),
            )
        )

    result = generations.delete_generation(
        tmp_path,
        target,
        target_type=target_type,
        expected_current_generation_id=CURRENT,
        force=True,
        actor="test-actor",
        reason="privacy erasure",
        exact_session_key="codex/target-session",
        commit=commit,
    )
    assert result["deleted"] is True
    assert result["classification"] == classification
    assert result["receipt"]["state"] == "complete"
    assert result["receipt"]["exact_session_key"] == "codex/target-session"
    assert result["receipt"]["counts"]["exact_match_count"] >= 1
    assert "content" not in json.dumps(result["receipt"])
    target_path = (
        generations.generation_path(tmp_path, target)
        if target_type == "generation"
        else generations.staged_paths(tmp_path, target)[0].parent
    )
    assert not target_path.exists()
    assert generations.generation_path(tmp_path, CURRENT).is_dir()
    assert generations.read_pointer(tmp_path)["current_generation_id"] == CURRENT
    if target == PREVIOUS:
        assert commits[0][0] == PREVIOUS
        assert commits[1][0] is None
        assert generations.read_pointer(tmp_path)["previous_generation_id"] is None
    else:
        assert generations.generation_path(tmp_path, PREVIOUS).is_dir()

    repeated = generations.delete_generation(
        tmp_path,
        target,
        target_type=target_type,
        expected_current_generation_id=CURRENT,
        force=True,
        actor="test-actor",
        reason="privacy erasure",
        exact_session_key="codex/target-session",
        commit=lambda: pytest.fail("idempotent replay committed"),
    )
    assert repeated["idempotent"] is True
    assert repeated["operation_id"] == result["operation_id"]


def test_absent_target_without_receipt_is_not_idempotent(tmp_path):
    _volume(tmp_path)
    missing = "g-20260726T220000Z-445566aabbcc"
    with pytest.raises(FileNotFoundError, match="receipt"):
        generations.delete_generation(
            tmp_path,
            missing,
            target_type="generation",
            expected_current_generation_id=CURRENT,
            force=True,
            actor="test",
            reason="missing",
            commit=lambda: None,
        )


def test_retry_after_durable_intent_before_deletion(tmp_path):
    _volume(tmp_path)
    reload_calls = 0

    def reload():
        nonlocal reload_calls
        reload_calls += 1
        if reload_calls == 4:
            raise InjectedFailure("after durable intent")

    with pytest.raises(InjectedFailure, match="durable intent"):
        generations.delete_generation(
            tmp_path,
            ORPHAN,
            target_type="generation",
            expected_current_generation_id=CURRENT,
            force=True,
            actor="test",
            reason="resume intent",
            exact_session_key="codex/target-session",
            commit=lambda: None,
            reload=reload,
        )

    operations = list((tmp_path / "deletion-receipts").iterdir())
    assert len(operations) == 1
    intent_before = (operations[0] / "intent.json").read_bytes()
    assert not (operations[0] / "complete.json").exists()
    assert generations.generation_path(tmp_path, ORPHAN).is_dir()

    result = generations.delete_generation(
        tmp_path,
        ORPHAN,
        target_type="generation",
        expected_current_generation_id=CURRENT,
        force=True,
        actor="test",
        reason="resume intent",
        exact_session_key="codex/target-session",
        commit=lambda: None,
    )
    assert result["deleted"] is True
    assert result["operation_id"] == operations[0].name
    assert (operations[0] / "intent.json").read_bytes() == intent_before
    assert (operations[0] / "complete.json").is_file()


def test_retry_previous_after_pointer_was_cleared(tmp_path):
    _volume(tmp_path)
    commit_calls = 0

    def commit():
        nonlocal commit_calls
        commit_calls += 1
        if commit_calls == 2:
            raise InjectedFailure("pointer commit")

    with pytest.raises(InjectedFailure, match="pointer commit"):
        generations.delete_generation(
            tmp_path,
            PREVIOUS,
            target_type="generation",
            expected_current_generation_id=CURRENT,
            force=True,
            actor="test",
            reason="resume previous",
            commit=commit,
        )
    assert generations.read_pointer(tmp_path)["previous_generation_id"] is None
    assert generations.generation_path(tmp_path, PREVIOUS).is_dir()

    result = generations.delete_generation(
        tmp_path,
        PREVIOUS,
        target_type="generation",
        expected_current_generation_id=CURRENT,
        force=True,
        actor="test",
        reason="resume previous",
        commit=lambda: None,
    )
    assert result["classification"] == "previous"
    assert result["deleted"] is True
    assert not generations.generation_path(tmp_path, PREVIOUS).exists()


def test_retry_target_absent_with_incomplete_intent_appends_completion(tmp_path):
    _volume(tmp_path)
    commit_calls = 0

    def commit():
        nonlocal commit_calls
        commit_calls += 1
        if commit_calls == 2:
            raise InjectedFailure("deletion commit")

    with pytest.raises(InjectedFailure, match="deletion commit"):
        generations.delete_generation(
            tmp_path,
            ORPHAN,
            target_type="generation",
            expected_current_generation_id=CURRENT,
            force=True,
            actor="test",
            reason="resume absent",
            commit=commit,
        )
    assert not generations.generation_path(tmp_path, ORPHAN).exists()
    operation = next((tmp_path / "deletion-receipts").iterdir())
    assert not (operation / "complete.json").exists()

    dry_run = generations.delete_generation(
        tmp_path,
        ORPHAN,
        target_type="generation",
        expected_current_generation_id=CURRENT,
        force=False,
        actor="test",
        reason="resume absent",
        commit=lambda: pytest.fail("dry-run retry committed"),
    )
    assert dry_run["dry_run"] is True
    assert dry_run["operation_id"] == operation.name
    assert not (operation / "complete.json").exists()

    result = generations.delete_generation(
        tmp_path,
        ORPHAN,
        target_type="generation",
        expected_current_generation_id=CURRENT,
        force=True,
        actor="test",
        reason="resume absent",
        commit=lambda: None,
    )
    assert result["idempotent"] is True
    assert result["deleted"] is False
    assert result["operation_id"] == operation.name
    assert (operation / "complete.json").is_file()


def test_completion_commit_failure_is_resumable(tmp_path):
    _volume(tmp_path)
    commit_calls = 0

    def commit():
        nonlocal commit_calls
        commit_calls += 1
        if commit_calls == 3:
            raise InjectedFailure("completion commit")

    with pytest.raises(InjectedFailure, match="completion commit"):
        generations.delete_generation(
            tmp_path,
            ORPHAN,
            target_type="generation",
            expected_current_generation_id=CURRENT,
            force=True,
            actor="test",
            reason="retry completion",
            commit=commit,
        )
    operation = next((tmp_path / "deletion-receipts").iterdir())
    assert (operation / "intent.json").is_file()
    assert not (operation / "complete.json").exists()
    assert not generations.generation_path(tmp_path, ORPHAN).exists()

    result = generations.delete_generation(
        tmp_path,
        ORPHAN,
        target_type="generation",
        expected_current_generation_id=CURRENT,
        force=True,
        actor="test",
        reason="retry completion",
        commit=lambda: None,
    )
    assert result["idempotent"] is True
    assert (operation / "complete.json").is_file()


def test_conflicting_identity_does_not_adopt_incomplete_intent(tmp_path):
    _volume(tmp_path)
    reload_calls = 0

    def reload():
        nonlocal reload_calls
        reload_calls += 1
        if reload_calls == 4:
            raise InjectedFailure("pause")

    with pytest.raises(InjectedFailure):
        generations.delete_generation(
            tmp_path,
            ORPHAN,
            target_type="generation",
            expected_current_generation_id=CURRENT,
            force=True,
            actor="original",
            reason="exact reason",
            exact_session_key="codex/target-session",
            commit=lambda: None,
            reload=reload,
        )

    with pytest.raises(ValueError, match="conflicting incomplete"):
        generations.delete_generation(
            tmp_path,
            ORPHAN,
            target_type="generation",
            expected_current_generation_id=CURRENT,
            force=True,
            actor="different",
            reason="exact reason",
            exact_session_key="codex/target-session",
            commit=lambda: pytest.fail("conflicting retry committed"),
        )
    assert generations.generation_path(tmp_path, ORPHAN).is_dir()


def test_duplicate_matching_incomplete_operations_fail_closed(tmp_path):
    _volume(tmp_path)
    reload_calls = 0

    def reload():
        nonlocal reload_calls
        reload_calls += 1
        if reload_calls == 4:
            raise InjectedFailure("pause")

    with pytest.raises(InjectedFailure):
        generations.delete_generation(
            tmp_path,
            ORPHAN,
            target_type="generation",
            expected_current_generation_id=CURRENT,
            force=True,
            actor="test",
            reason="duplicate",
            commit=lambda: None,
            reload=reload,
        )
    receipt_root = tmp_path / "deletion-receipts"
    original = next(receipt_root.iterdir())
    duplicate_id = str(uuid.uuid4())
    duplicate = receipt_root / duplicate_id
    shutil.copytree(original, duplicate)
    duplicate_intent = json.loads((duplicate / "intent.json").read_text())
    duplicate_intent["operation_id"] = duplicate_id
    (duplicate / "intent.json").write_text(json.dumps(duplicate_intent))

    with pytest.raises(ValueError, match="duplicate matching incomplete"):
        generations.delete_generation(
            tmp_path,
            ORPHAN,
            target_type="generation",
            expected_current_generation_id=CURRENT,
            force=True,
            actor="test",
            reason="duplicate",
            commit=lambda: pytest.fail("duplicate retry committed"),
        )
    assert generations.generation_path(tmp_path, ORPHAN).is_dir()


@pytest.mark.parametrize("entry_kind", ["malformed", "symlink", "overflow"])
def test_invalid_operation_entries_fail_closed(tmp_path, monkeypatch, entry_kind):
    _volume(tmp_path)
    receipt_root = tmp_path / "deletion-receipts"
    receipt_root.mkdir()
    if entry_kind == "malformed":
        operation = receipt_root / str(uuid.uuid4())
        operation.mkdir()
        (operation / "unexpected.json").write_text("{}")
    elif entry_kind == "symlink":
        (receipt_root / str(uuid.uuid4())).symlink_to(tmp_path / "generations")
    else:
        monkeypatch.setattr(generations, "MAX_RECEIPTS", 1)
        (receipt_root / str(uuid.uuid4())).mkdir()
        (receipt_root / str(uuid.uuid4())).mkdir()

    with pytest.raises(ValueError):
        generations.delete_generation(
            tmp_path,
            ORPHAN,
            target_type="generation",
            expected_current_generation_id=CURRENT,
            force=False,
            actor="test",
            reason="invalid receipts",
            commit=lambda: pytest.fail("invalid state committed"),
        )
    assert generations.generation_path(tmp_path, ORPHAN).is_dir()
