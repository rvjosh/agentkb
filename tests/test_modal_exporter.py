from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agentkb.encoder import DEFAULT_MODEL
from agentkb.modal_backend import exporter


GENERATION_ID = "g-20260725T123456Z-001122aabbcc"


def test_canonical_json_escapes_lone_surrogates_but_preserves_valid_unicode():
    encoded = exporter._canonical_json(
        {"malformed": "\udcff", "valid": "naïve café"}
    ).encode("utf-8")
    assert json.loads(encoded) == {
        "malformed": "\\udcff",
        "valid": "naïve café",
    }


def _write_corpus(wiki_root: Path, chats_root: Path) -> None:
    (wiki_root / "wiki").mkdir(parents=True)
    (wiki_root / "sources").mkdir()
    (wiki_root / "wiki" / "page.md").write_text(
        "---\ntitle: Page\ntags: [one]\n---\n# Topic\nWiki body.\n"
    )
    (wiki_root / "sources" / "ref.rst").write_text(
        "Reference\n=========\n\nSource body.\n"
    )
    readable = chats_root / "readable" / "2026-07"
    readable.mkdir(parents=True)
    (readable / "chat.md").write_text(
        "---\ntitle: Chat\ntags: [codex]\n---\n# User\nChat body.\n"
    )


def test_export_is_canonical_model_free_and_has_exact_default_collections(
    tmp_path, monkeypatch
):
    wiki_root = tmp_path / "wiki-root"
    chats_root = tmp_path / "chats-root"
    output = tmp_path / "agentkb-modal-refresh-output"
    output.mkdir()
    _write_corpus(wiki_root, chats_root)
    monkeypatch.setattr(exporter, "prepare_chats", lambda root: root / "readable")

    result = exporter.export_corpus(
        generation_id=GENERATION_ID,
        wiki_root=wiki_root,
        chats_root=chats_root,
        output_dir=output,
        exported_at=datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc),
    )

    corpus_bytes = (output / "corpus.jsonl").read_bytes()
    assert corpus_bytes.endswith(b"\n")
    assert b"\n\n" not in corpus_bytes
    records = [json.loads(line) for line in corpus_bytes.splitlines()]
    assert [record["collection"] for record in records] == [
        "chats",
        "wiki",
        "wiki:source",
    ]
    assert all(not Path(record["file"]).is_absolute() for record in records)
    assert len({record["canonical_id"] for record in records}) == 3

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest == {
        "schema": 1,
        "generation_id": GENERATION_ID,
        "model": DEFAULT_MODEL,
        "corpus_count": 3,
        "corpus_hash": hashlib.sha256(corpus_bytes).hexdigest(),
        "source_file_counts": {"chats": 1, "wiki": 1, "wiki:source": 1},
        "collection_counts": {
            "chats": {"documents": 1, "files": 1},
            "wiki": {"documents": 1, "files": 1},
            "wiki:source": {"documents": 1, "files": 1},
        },
        "sources": {"schema": 1, "items": []},
        "exported_at": "2026-07-25T12:00:00Z",
    }
    assert result["corpus_hash"] == manifest["corpus_hash"]


def test_export_is_byte_deterministic_except_manifest_timestamp(tmp_path, monkeypatch):
    wiki_root = tmp_path / "wiki-root"
    chats_root = tmp_path / "chats-root"
    _write_corpus(wiki_root, chats_root)
    monkeypatch.setattr(exporter, "prepare_chats", lambda root: root / "readable")
    outputs = [tmp_path / "agentkb-modal-refresh-a", tmp_path / "agentkb-modal-refresh-b"]
    for output in outputs:
        output.mkdir()
        exporter.export_corpus(
            generation_id=GENERATION_ID,
            wiki_root=wiki_root,
            chats_root=chats_root,
            output_dir=output,
        )
    assert (outputs[0] / "corpus.jsonl").read_bytes() == (
        outputs[1] / "corpus.jsonl"
    ).read_bytes()


def test_export_streams_more_than_two_remote_batches_in_source_order(
    tmp_path, monkeypatch
):
    wiki_root = tmp_path / "wiki-root"
    chats_root = tmp_path / "chats-root"
    output = tmp_path / "agentkb-modal-refresh-many"
    output.mkdir()
    pages = wiki_root / "wiki"
    pages.mkdir(parents=True)
    for index in reversed(range(513)):
        (pages / f"page-{index:04d}.md").write_text(
            f"# Page {index}\n\nBody {index}.\n"
        )
    monkeypatch.setattr(exporter, "prepare_chats", lambda root: root / "readable")

    result = exporter.export_corpus(
        generation_id=GENERATION_ID,
        wiki_root=wiki_root,
        chats_root=chats_root,
        output_dir=output,
    )

    records = [
        json.loads(line)
        for line in (output / "corpus.jsonl").read_text().splitlines()
    ]
    assert result["corpus_count"] == 513
    assert [record["file"] for record in records] == sorted(
        record["file"] for record in records
    )


def test_export_rejects_duplicate_canonical_ids_while_streaming(
    tmp_path, monkeypatch
):
    wiki_root = tmp_path / "wiki-root"
    chats_root = tmp_path / "chats-root"
    output = tmp_path / "agentkb-modal-refresh-duplicates"
    output.mkdir()
    _write_corpus(wiki_root, chats_root)
    monkeypatch.setattr(exporter, "prepare_chats", lambda root: root / "readable")
    monkeypatch.setattr(exporter, "_canonical_id", lambda record: "a" * 64)

    with pytest.raises(ValueError, match="duplicate canonical_id"):
        exporter.export_corpus(
            generation_id=GENERATION_ID,
            wiki_root=wiki_root,
            chats_root=chats_root,
            output_dir=output,
        )


def test_output_directory_must_be_absolute_existing_and_empty(tmp_path):
    with pytest.raises(ValueError, match="absolute"):
        exporter.validate_output_directory(Path("relative"))
    with pytest.raises(ValueError, match="already exist"):
        exporter.validate_output_directory(tmp_path / "missing")
    occupied = tmp_path / "agentkb-modal-refresh-occupied"
    occupied.mkdir()
    (occupied / "keep").write_text("do not delete")
    with pytest.raises(ValueError, match="empty"):
        exporter.validate_output_directory(occupied)
    assert (occupied / "keep").read_text() == "do not delete"


def test_prepare_chats_runs_all_sources_copy_and_render(tmp_path, monkeypatch):
    calls: list[tuple[str, Path]] = []
    sessions = tmp_path / "chats" / "sessions"
    sessions.mkdir(parents=True)
    monkeypatch.setattr(
        exporter,
        "migrate_sessions_layout",
        lambda path: calls.append(("migrate", path)),
    )
    monkeypatch.setattr(
        exporter,
        "export_all_sessions",
        lambda path: calls.append(("copy", path)),
    )
    monkeypatch.setattr(
        exporter,
        "export_readable",
        lambda source, destination: calls.append(("render", destination)),
    )
    assert exporter.prepare_chats(tmp_path / "chats") == tmp_path / "chats" / "readable"
    assert calls == [
        ("migrate", sessions),
        ("copy", sessions),
        ("render", tmp_path / "chats" / "readable"),
    ]


def test_external_prefix_keeps_canonical_id_stable_across_content_change(tmp_path):
    root = tmp_path / "external"
    root.mkdir()
    path = root / "item.jsonl"
    path.write_text('{"body":"first"}\n')
    first = list(
        exporter._jsonl_records(
            root=root,
            collection="wiki:source",
            stored_prefix="readwise-tweets/",
        )
    )[0][1]
    path.write_text('{"body":"changed"}\n')
    second = list(
        exporter._jsonl_records(
            root=root,
            collection="wiki:source",
            stored_prefix="readwise-tweets/",
        )
    )[0][1]
    assert first["file"] == "readwise-tweets/item.jsonl"
    assert first["canonical_id"] == second["canonical_id"]
    assert first["raw_content"] != second["raw_content"]


def test_central_history_selects_latest_present_version_and_deduplicates_observations(
    tmp_path,
):
    backup = tmp_path / "backup"
    raw_root = backup / "raw" / "codex"
    raw_root.mkdir(parents=True)
    database = tmp_path / "index.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE transcripts (
          id INTEGER PRIMARY KEY, source TEXT, native_session_id TEXT, created_at TEXT
        );
        CREATE TABLE versions (
          sha256 TEXT PRIMARY KEY, transcript_id INTEGER, blob_path TEXT,
          parser_status TEXT, title TEXT, cwd TEXT, start_time TEXT, end_time TEXT,
          created_at TEXT
        );
        CREATE TABLE observations (
          id INTEGER PRIMARY KEY, version_sha256 TEXT, present_at_last_scan INTEGER
        );
        """
    )
    session_id = "11111111-2222-3333-4444-555555555555"

    def add_version(name: str, text: str, created_at: str) -> str:
        payload = (
            json.dumps(
                {
                    "timestamp": "2026-07-26T12:00:00Z",
                    "type": "event_msg",
                    "payload": {"type": "user_message", "message": text},
                },
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        sha = hashlib.sha256(payload).hexdigest()
        relative = f"raw/codex/{name}.jsonl"
        compressed = backup / f"{relative}.zst"
        compressed.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["zstd", "-q", "-o", str(compressed)],
            input=payload,
            check=True,
        )
        connection.execute(
            """
            INSERT INTO versions(
              sha256, transcript_id, blob_path, parser_status, title, cwd,
              start_time, end_time, created_at
            ) VALUES (?, 1, ?, 'ok', ?, '/repo', NULL, NULL, ?)
            """,
            (sha, relative, text, created_at),
        )
        return sha

    connection.execute(
        "INSERT INTO transcripts VALUES (1, 'codex', ?, '2026-07-26')",
        (session_id,),
    )
    old_sha = add_version("old", "old text", "2026-07-26T10:00:00Z")
    new_sha = add_version("new", "new text", "2026-07-26T11:00:00Z")
    connection.executemany(
        "INSERT INTO observations(version_sha256, present_at_last_scan) VALUES (?, 1)",
        [(old_sha,), (new_sha,), (new_sha,)],
    )
    connection.commit()
    connection.close()
    subprocess.run(
        ["zstd", "-q", "-f", "-o", str(backup / "index.sqlite3.zst"), str(database)],
        check=True,
    )

    records = list(exporter._history_records(backup))
    assert len(records) == 1
    assert records[0][0] == f"codex/{session_id}"
    assert records[0][1]["raw_content"] == "new text"
    assert records[0][1]["file"] == (
        f"agent-history-central/codex/{session_id}.md"
    )
