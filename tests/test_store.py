"""Tests for agentkb.store — SQLite metadata, FTS5 keyword search, document CRUD.

IndexStore is the on-disk storage layer that sits behind search. It has two parts:
1. SQLite with FTS5 — stores document metadata (file, line, title, section, tags,
   content) and provides keyword search via trigram tokenization.
2. PLAID vector index — stores ColBERT embeddings for semantic search (tested in
   test_store_plaid.py separately since it requires the encoder).

Every chunk produced by the wiki or chat parsers ends up as a row in this store.
When you run `agentkb search "query"`, the search pipeline queries both FTS5 and
PLAID, then fuses the results with RRF.
"""

import json
import pickle
import sqlite3

import pytest

from agentkb.store import IndexStore, Document, sanitize_fts5_query


# --- sanitize_fts5_query ---
# User search queries go through FTS5 for keyword matching. But FTS5 has its
# own syntax (AND, OR, NOT, NEAR are operators; unquoted special chars can break
# queries). sanitize_fts5_query wraps each token in quotes so things like "C++"
# or "node.js" are treated as literal text, not syntax errors.


def test_sanitize_basic():
    """Wraps each word in quotes for safe FTS5 matching."""
    assert sanitize_fts5_query("hello world") == '"hello" "world"'


def test_sanitize_strips_operators():
    """Removes FTS5 boolean operators (AND, OR, NOT, NEAR)."""
    assert sanitize_fts5_query("foo AND bar") == '"foo" "bar"'
    assert sanitize_fts5_query("NOT bad") == '"bad"'


def test_sanitize_special_chars():
    """Preserves special chars inside tokens (C++, node.js)."""
    result = sanitize_fts5_query("C++ node.js")
    assert '"C++"' in result
    assert '"node.js"' in result


def test_sanitize_empty():
    """Returns empty string for garbage input."""
    assert sanitize_fts5_query("!@#$") == ""


def test_read_only_plaid_mappings_are_loaded_once_per_store(tmp_path):
    plaid_dir = tmp_path / "index" / "plaid"
    plaid_dir.mkdir(parents=True)
    paths = (
        plaid_dir / "documents_ids_to_plaid_ids.pkl",
        plaid_dir / "plaid_ids_to_documents_ids.pkl",
    )
    paths[0].write_bytes(pickle.dumps({"1": 0}))
    paths[1].write_bytes(pickle.dumps({0: "1"}))
    store = IndexStore(tmp_path / "index", read_only=True)

    first = store._load_read_only_plaid_mappings()
    paths[0].write_bytes(b"invalid after first read")
    second = store._load_read_only_plaid_mappings()

    assert second is first
    store.close()
    with pytest.raises(RuntimeError, match="cannot read PLAID"):
        store._load_read_only_plaid_mappings()


def test_read_only_metadata_uses_immutable_sqlite_uri(tmp_path, monkeypatch):
    index_dir = tmp_path / "index"
    writable = IndexStore(index_dir)
    writable.create()
    writable.close()
    calls: list[tuple[str, bool]] = []
    real_connect = sqlite3.connect

    def recording_connect(database, *args, **kwargs):
        calls.append((database, kwargs.get("uri", False)))
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", recording_connect)
    store = IndexStore(index_dir, read_only=True)
    try:
        store._connect()
    finally:
        store.close()

    assert len(calls) == 1
    assert calls[0][1] is True
    assert calls[0][0].endswith("?mode=ro&immutable=1")


def test_immutable_premerged_store_never_copies_and_closes_backend(
    tmp_path, monkeypatch
):
    fast_plaid_dir = tmp_path / "index" / "plaid" / "fast_plaid_index"
    fast_plaid_dir.mkdir(parents=True)
    closed = []

    class Backend:
        def close(self):
            closed.append(True)

    monkeypatch.setattr(
        "agentkb.fast_plaid_immutable.load_immutable_premerged_fast_plaid",
        lambda path, certificate: (
            Backend()
            if path == fast_plaid_dir and certificate == {"schema": 1}
            else pytest.fail("wrong immutable adapter arguments")
        ),
    )
    monkeypatch.setattr(
        "agentkb.store.shutil.copytree",
        lambda *_args, **_kwargs: pytest.fail("immutable mode copied FastPLAID"),
    )
    store = IndexStore(
        tmp_path / "index",
        read_only=True,
        immutable_premerged={"schema": 1},
    )
    monkeypatch.setattr(
        store,
        "_load_read_only_plaid_mappings",
        lambda: ({"1": 0}, {0: "1"}),
    )

    store._load_read_only_plaid_index()
    store.close()

    assert closed == [True]


# --- IndexStore CRUD ---
# These test the basic document lifecycle: creating the schema, inserting
# documents, retrieving them by ID or collection, and deleting them by file.
# The "collection" field is how wiki vs chat documents are separated in
# the same store (values: "wiki", "wiki:source", "chats").


def test_create_and_add_documents(tmp_path):
    """create() sets up the schema; add_documents() inserts and returns IDs."""
    store = IndexStore(tmp_path / "idx")
    store.create()

    ids = store.add_documents([
        {"collection": "wiki", "file": "tools/git.md", "line": 1,
         "content": "Git basics", "raw_content": "# Git\nBasics"},
        {"collection": "wiki", "file": "tools/python.md", "line": 10,
         "content": "Python tips", "raw_content": "# Python\nTips"},
    ])
    assert len(ids) == 2
    assert store.document_count() == 2
    store.close()


def test_get_document_by_id(tmp_path):
    """Retrieves a document by its auto-generated ID.

    After RRF fusion produces a ranked list of doc IDs, the search pipeline
    calls get_document_by_id to hydrate each result with its full content.
    """
    store = IndexStore(tmp_path / "idx")
    store.create()
    ids = store.add_documents([
        {"collection": "wiki", "file": "a.md", "content": "aaa",
         "title": "Page A", "section": "Intro", "tags": ["test"]},
    ])

    doc = store.get_document_by_id(ids[0])
    assert isinstance(doc, Document)
    assert doc.file == "a.md"
    assert doc.title == "Page A"
    assert doc.section == "Intro"
    assert json.loads(doc.tags) == ["test"]
    store.close()


def test_get_document_by_id_missing(tmp_path):
    """Returns None for a nonexistent ID."""
    store = IndexStore(tmp_path / "idx")
    store.create()
    assert store.get_document_by_id(999) is None
    store.close()


def test_get_document_ids_by_collection(tmp_path):
    """Filters document IDs by collection name.

    When searching with --scope wiki, only wiki and wiki:source documents
    are considered. This collection filter is how that scoping works.
    """
    store = IndexStore(tmp_path / "idx")
    store.create()
    store.add_documents([
        {"collection": "wiki", "file": "a.md", "content": "wiki content"},
        {"collection": "chats", "file": "b.md", "content": "chat content"},
        {"collection": "wiki", "file": "c.md", "content": "more wiki"},
    ])

    assert len(store.get_document_ids(collection="wiki")) == 2
    assert len(store.get_document_ids(collection="chats")) == 1
    assert len(store.get_document_ids()) == 3
    store.close()


# document_count vs file_count matters for the `agentkb status` display.
# A single wiki page might produce 5 chunks (one per heading section), so
# document_count=5 but file_count=1. The status command shows both.
def test_document_count_and_file_count(tmp_path):
    """document_count counts chunks; file_count counts distinct files."""
    store = IndexStore(tmp_path / "idx")
    store.create()
    # Two chunks from the same file, one from another
    store.add_documents([
        {"collection": "wiki", "file": "a.md", "content": "chunk 1"},
        {"collection": "wiki", "file": "a.md", "content": "chunk 2"},
        {"collection": "wiki", "file": "b.md", "content": "chunk 3"},
    ])
    assert store.document_count() == 3
    assert store.file_count() == 2
    assert store.document_count(collection="wiki") == 3
    store.close()


# delete_documents_by_file is used during incremental indexing. When a wiki
# page changes, all its old chunks are deleted before the new chunks are
# inserted. This avoids stale chunks from a previous version lingering in
# search results.
def test_delete_documents_by_file(tmp_path):
    """Removes all chunks belonging to given file paths."""
    store = IndexStore(tmp_path / "idx")
    store.create()
    store.add_documents([
        {"collection": "wiki", "file": "keep.md", "content": "keep this"},
        {"collection": "wiki", "file": "remove.md", "content": "remove this"},
    ])
    assert store.document_count() == 2

    store.delete_documents_by_file({"remove.md"})
    assert store.document_count() == 1
    [remaining_id] = store.get_document_ids()
    assert store.get_document_by_id(remaining_id).file == "keep.md"
    store.close()


# --- FTS5 keyword search ---
# FTS5 with trigram tokenization handles the keyword side of search. It catches
# exact matches that semantic search might miss (e.g., searching for a specific
# function name like "encode_query" or an error message). The results from FTS5
# are fused with PLAID semantic results via RRF in the search pipeline.


def test_keyword_search(tmp_path):
    """FTS5 trigram search finds matching documents."""
    store = IndexStore(tmp_path / "idx")
    store.create()
    store.add_documents([
        {"collection": "wiki", "file": "a.md", "content": "Python asyncio event loop"},
        {"collection": "wiki", "file": "b.md", "content": "JavaScript promises and callbacks"},
        {"collection": "wiki", "file": "c.md", "content": "Python decorators and metaclasses"},
    ])

    results = store.keyword_search("Python")
    doc_ids = [doc_id for doc_id, _ in results]
    # Both Python docs should match
    assert len(doc_ids) >= 2
    store.close()


def test_keyword_search_by_collection(tmp_path):
    """Collection filter restricts keyword search to a specific collection."""
    store = IndexStore(tmp_path / "idx")
    store.create()
    store.add_documents([
        {"collection": "wiki", "file": "a.md", "content": "Python guide"},
        {"collection": "chats", "file": "b.md", "content": "Python discussion"},
    ])

    wiki_results = store.keyword_search("Python", collection="wiki")
    assert len(wiki_results) == 1

    chat_results = store.keyword_search("Python", collection="chats")
    assert len(chat_results) == 1
    store.close()


# --- Incremental indexing state ---
# The state file (state.json) stores a mapping of {filename: content_hash}.
# On the next index run, the indexer compares current hashes to saved hashes
# to figure out which files are new, changed, or deleted. This is what makes
# `agentkb index` fast on subsequent runs — only changed files get re-encoded.


def test_state_roundtrip(tmp_path):
    """save_state/load_state persist file hashes for incremental indexing."""
    store = IndexStore(tmp_path / "idx")
    store.create()

    state = {"__model__": "test-model", "a.md": "abc123", "b.md": "def456"}
    store.save_state(state)

    loaded = store.load_state()
    assert loaded == state
    store.close()


def test_state_empty_when_missing(tmp_path):
    """load_state returns {} when no state file exists."""
    store = IndexStore(tmp_path / "idx")
    store.create()
    assert store.load_state() == {}
    store.close()
