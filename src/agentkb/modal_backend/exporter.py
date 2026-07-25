"""Model-free production corpus export for the private Modal backend."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from agentkb.indexing import IndexSpec
from agentkb.modal_backend.generations import validate_generation_id
from agentkb.utils import chunk_markdown
from agentkb.wiki.parser import WIKI_SPEC


SCHEMA = 1
CORPUS_FILENAME = "corpus.jsonl"
MANIFEST_FILENAME = "manifest.json"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
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
    root: Path, spec: IndexSpec, relative_file: str, entry: Any
) -> Iterator[dict[str, Any]]:
    for chunk in chunk_markdown(entry.path, relative_to=root):
        record: dict[str, Any] = {
            "collection": entry.collection,
            "content": spec.make_structured_text(chunk, entry),
            "file": chunk["file"],
            "line": chunk["line"],
            "name": chunk["title"],
            "raw_content": chunk["content"],
            "section": chunk["section"],
            "tags": chunk.get("tags", []),
            "title": chunk["title"],
            "unit_type": "chunk",
        }
        if record["file"] != relative_file:
            raise ValueError("parser returned a non-canonical relative file path")
        record["canonical_id"] = _canonical_id(record)
        yield record


def export_corpus(
    *,
    generation_id: str,
    wiki_root: Path,
    chats_root: Path,
    output_dir: Path,
    exported_at: datetime | None = None,
) -> dict[str, Any]:
    """Export the default wiki, wiki:source, and chats corpus."""
    generation_id = validate_generation_id(generation_id)
    output_dir = validate_output_directory(output_dir)
    readable_root = prepare_chats(chats_root.expanduser().resolve())

    corpus_path = output_dir / CORPUS_FILENAME
    digest = hashlib.sha256()
    corpus_count = 0
    canonical_ids: set[str] = set()
    source_file_counts = {
        "chats": 0,
        "wiki": 0,
        "wiki:source": 0,
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
                file_has_records = True
            if file_has_records:
                source_file_counts[collection] += 1
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
    args = parser.parse_args()
    result = export_corpus(
        generation_id=args.generation_id,
        wiki_root=args.wiki_root,
        chats_root=args.chats_root,
        output_dir=args.output_dir,
    )
    print(_canonical_json(result))


if __name__ == "__main__":
    main()
