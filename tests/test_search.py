"""Tests for agentkb.search — RRF fusion, regex helpers, result formatting.

search.py is the search pipeline orchestrator. When you run `agentkb search "query"`,
it runs two parallel searches (PLAID semantic + FTS5 keyword), fuses them with
Reciprocal Rank Fusion (RRF), applies post-filters (regex, glob), and formats
the results. This file tests each of those components independently.
"""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
from io import StringIO
import pickle
from pathlib import Path
import sqlite3
import tempfile

from click.testing import CliRunner
import numpy as np
import pytest

from agentkb.output import echo_status

from agentkb import cli
from agentkb.search import (
    rrf_fuse,
    strip_regex_for_semantic,
    merge_query_with_pattern,
    _compile_pattern,
    _matches_globs,
    SearchResult,
    merge_multi_collection,
    search,
    search_transcript_sessions,
)
from agentkb.store import Document, IndexStore


# --- rrf_fuse ---
# RRF (Reciprocal Rank Fusion) is how semantic and keyword search results are
# combined into a single ranking. Semantic search is good at understanding meaning
# ("how do I handle errors" matches "exception handling"), while keyword search
# catches exact terms ("encode_query" finds the literal function name). RRF
# merges both by assigning scores based on rank position, not raw score — this
# works because raw scores from PLAID and FTS5 are on completely different scales.
# The alpha parameter controls the balance (default 0.75 = semantic 3x keyword).


def test_rrf_fuse_basic():
    """RRF combines two rankings into a single fused ranking."""
    semantic = [(1, 10.0), (2, 8.0), (3, 5.0)]
    keyword = [(2, 20.0), (3, 15.0), (4, 10.0)]

    fused = rrf_fuse(semantic, keyword)
    fused_ids = [doc_id for doc_id, _ in fused]

    # Doc 2 appears in both rankings, so it should rank highest
    assert fused_ids[0] == 2
    # All docs from both rankings should appear
    assert set(fused_ids) == {1, 2, 3, 4}


def test_rrf_fuse_pure_semantic():
    """alpha=1.0 means only semantic ranking matters."""
    semantic = [(1, 10.0), (2, 5.0)]
    keyword = [(2, 20.0), (3, 15.0)]

    fused = rrf_fuse(semantic, keyword, alpha=1.0)
    fused_ids = [doc_id for doc_id, _ in fused]
    # Keyword results get zero weight, so doc 3 still appears but with 0 score
    assert fused_ids[0] == 1  # top semantic result wins


def test_rrf_fuse_pure_keyword():
    """alpha=0.0 means only keyword ranking matters."""
    semantic = [(1, 10.0), (2, 5.0)]
    keyword = [(3, 20.0), (4, 15.0)]

    fused = rrf_fuse(semantic, keyword, alpha=0.0)
    fused_ids = [doc_id for doc_id, _ in fused]
    assert fused_ids[0] == 3  # top keyword result wins


# --- strip_regex_for_semantic / merge_query_with_pattern ---
# When the user passes -e (regex filter) alongside a search query, the regex
# pattern contains useful semantic information buried under metacharacters.
# For example, `agentkb search "error handling" -e "async\s+fn"` — the regex
# tells us the user cares about "async fn". strip_regex_for_semantic extracts
# those meaningful words, and merge_query_with_pattern appends them to the
# semantic query to improve ColBERT retrieval without duplicating tokens.


def test_strip_regex_basic():
    r"""Removes regex metacharacters, keeps meaningful words."""
    assert strip_regex_for_semantic(r"async\s+fn") == "async fn"


def test_strip_regex_alternation():
    """Converts | to space (OR alternatives become separate terms)."""
    assert strip_regex_for_semantic("foo|bar") == "foo bar"


def test_strip_regex_complex():
    """Handles character classes, quantifiers, groups."""
    result = strip_regex_for_semantic(r"Result<[^>]*>")
    assert "Result" in result


def test_merge_query_with_pattern():
    """Appends unique tokens from regex pattern to the query."""
    result = merge_query_with_pattern("error handling", r"async\s+fn")
    assert "error handling" in result
    assert "async" in result
    assert "fn" in result


def test_merge_query_with_pattern_deduplicates():
    """Tokens already in the query are not repeated."""
    result = merge_query_with_pattern("async error", r"async\s+fn")
    # "async" is in both, should only appear once
    assert result.count("async") == 1


def test_merge_empty_pattern():
    """Empty pattern returns the original query unchanged."""
    assert merge_query_with_pattern("hello", "") == "hello"


# --- _compile_pattern ---
# The -e, -F, and -w flags give users grep-like content filtering on top of
# search results. After RRF fusion selects the top documents, each result's
# content is checked against this compiled pattern. Results that don't match
# are discarded. This is a post-filter — it doesn't affect which documents
# the search finds, just which ones survive to be shown.


def test_compile_pattern_regex():
    """Compiles a regex pattern for content filtering."""
    pat = _compile_pattern(r"def\s+\w+")
    assert pat is not None
    assert pat.search("def my_func():")
    assert not pat.search("class MyClass:")


def test_compile_pattern_fixed():
    """fixed=True escapes regex metacharacters for literal matching."""
    pat = _compile_pattern("foo.bar", fixed=True)
    assert pat.search("foo.bar")
    assert not pat.search("fooXbar")  # . is literal, not wildcard


def test_compile_pattern_word():
    r"""word=True adds \b word boundaries."""
    pat = _compile_pattern("test", word=True)
    assert pat.search("run test now")
    assert not pat.search("testing")  # "test" is inside "testing"


def test_compile_pattern_none():
    """Returns None when no pattern is given."""
    assert _compile_pattern(None) is None
    assert _compile_pattern("") is None


# --- _matches_globs ---
# The --include and --exclude flags let users filter results by file path pattern.
# For example, `--include "*.py"` only shows Python files, `--exclude-dir tests`
# hides test directories. This is applied as a post-filter after RRF fusion.


def test_matches_globs():
    """Checks if a filepath matches any glob pattern."""
    assert _matches_globs("tools/git.md", ["*.md"])
    assert _matches_globs("tools/git.md", ["tools/*"])
    assert not _matches_globs("tools/git.md", ["*.py"])


def test_matches_globs_empty():
    """Empty pattern list matches nothing."""
    assert not _matches_globs("anything.md", [])


# --- SearchResult ---
# SearchResult is the output dataclass. format_terminal produces the human-readable
# output you see in the terminal (with [collection] tag, file:line, score, and a
# content snippet). to_json produces the machine-readable output used by agents
# (via --json flag). The terminal format includes context_lines of the content
# to give a preview without overwhelming.


def test_search_result_format_terminal():
    """format_terminal renders a human-readable snippet."""
    r = SearchResult(
        collection="wiki",
        file="tools/git.md",
        line=5,
        score=0.85,
        title="Git",
        section="Rebasing",
        raw_content="Line 1\nLine 2\nLine 3",
    )
    output = r.format_terminal(context_lines=2)
    assert "[wiki]" in output
    assert "tools/git.md:5" in output
    assert "0.85" in output
    assert "Git > Rebasing" in output
    assert "Line 1" in output
    assert "1 more lines" in output


def test_search_result_to_json():
    """to_json produces compact metadata by default."""
    r = SearchResult(
        collection="chats",
        file="/tmp/agentkb/chats/readable/2024-01/session.md",
        line=1,
        score=0.7123456,
        relative_path="2024-01/session.md",
        name="my session",
        raw_content="the content",
    )
    j = r.to_json()
    assert j["collection"] == "chats"
    assert j["file"] == "/tmp/agentkb/chats/readable/2024-01/session.md"
    assert j["path"] == "/tmp/agentkb/chats/readable/2024-01/session.md"
    assert j["filename"] == "session.md"
    assert j["relative_path"] == "2024-01/session.md"
    assert j["score"] == 0.7123  # rounded to 4 decimals
    assert "content" not in j
    assert j["name"] == "my session"


def test_search_result_to_json_can_include_content():
    """Full content remains available when explicitly requested."""
    r = SearchResult(
        collection="wiki",
        file="/tmp/agentkb/wiki/wiki/tools/git.md",
        line=5,
        score=0.9,
        raw_content="the content",
    )
    assert r.to_json(include_content=True)["content"] == "the content"


class _FakeSearchStore:
    def __init__(self, root):
        self.root = root

    def get_document_ids(self, collection=None):
        return [1] if collection == "wiki" else []

    def semantic_search(self, query_embedding, top_k=50, subset_ids=None):
        return [(1, 0.9)]

    def keyword_search(self, query, collection=None, limit=50):
        return []

    def get_document_by_id(self, doc_id):
        if doc_id != 1:
            return None
        return Document(
            id=1,
            collection="wiki",
            file="wiki/tools/git.md",
            line=5,
            name="Git",
            unit_type="chunk",
            content="structured",
            raw_content="raw",
            title="Git",
            section="Rebasing",
            tags='["tools"]',
        )

    def resolve_file_path(self, file):
        return str((self.root / file).resolve())


def test_search_resolves_results_to_absolute_paths(tmp_path):
    """Hydrated search results expose absolute paths and keep the relative path."""
    results = search(
        store=_FakeSearchStore(tmp_path),
        query_embedding="fake-embedding",
        query_text="git rebase",
        scope="wiki:notes",
        top_k=1,
        semantic_only=True,
    )

    assert len(results) == 1
    assert results[0].file == str((tmp_path / "wiki/tools/git.md").resolve())
    assert results[0].relative_path == "wiki/tools/git.md"


class _AdaptiveTranscriptStore:
    def __init__(self, rankings, documents):
        self.rankings = rankings
        self.documents = documents
        self.calls = []

    def semantic_search(self, _embedding, top_k=50, subset_ids=None):
        self.calls.append((top_k, list(subset_ids)))
        return self.rankings[:top_k]

    def get_document_by_id(self, doc_id):
        return self.documents.get(doc_id)

    def resolve_file_path(self, file):
        return f"/resolved/{file}"


def _transcript_document(doc_id, session_id, line):
    return Document(
        id=doc_id,
        collection="chats",
        file=f"agent-history-central/codex/{session_id}.md",
        line=line,
        name=session_id,
        unit_type="chunk",
        content=f"chunk-{doc_id}",
        raw_content=f"raw-{doc_id}",
    )


def test_transcript_session_search_adapts_and_preserves_first_representatives():
    rankings = [
        (1, 0.99),
        (2, 0.98),
        (3, 0.97),
        (4, 0.96),
        (5, 0.95),
        (6, 0.90),
        (7, 0.80),
    ]
    identities = {
        **{doc_id: ("codex", "session-a") for doc_id in range(1, 6)},
        6: ("claude", "session-b"),
        7: ("codex", "session-c"),
    }
    documents = {
        doc_id: _transcript_document(doc_id, identity[1], doc_id)
        for doc_id, identity in identities.items()
    }
    store = _AdaptiveTranscriptStore(rankings, documents)

    results = search_transcript_sessions(
        store,
        "embedding",
        top_k=3,
        eligible_doc_ids=tuple(identities),
        session_identities=identities,
    )

    assert [result.name for result in results] == [
        "session-a",
        "session-b",
        "session-c",
    ]
    assert [result.line for result in results] == [1, 6, 7]
    assert [call[0] for call in store.calls] == [3, 6, 7]
    assert all(call[1] == list(identities) for call in store.calls)


def test_transcript_session_search_stops_safely_at_subset_exhaustion():
    identities = {
        1: ("codex", "session-a"),
        2: ("codex", "session-a"),
    }
    store = _AdaptiveTranscriptStore(
        [(1, 0.9), (2, 0.8)],
        {
            1: _transcript_document(1, "session-a", 1),
            2: _transcript_document(2, "session-a", 2),
        },
    )

    results = search_transcript_sessions(
        store,
        "embedding",
        top_k=3,
        eligible_doc_ids=(1, 2),
        session_identities=identities,
    )

    assert [result.name for result in results] == ["session-a"]
    assert store.calls == [(2, [1, 2])]


# --- merge_multi_collection ---
# When searching with --scope all, the search pipeline runs against both the
# wiki and chats indexes separately, producing two ranked lists. These can't
# be compared by raw score (different indexes, different document characteristics),
# so merge_multi_collection uses RRF to combine them by rank position. It also
# deduplicates results that appear in both (same file+line).


def test_merge_multi_collection():
    """Merges results from multiple stores using RRF, deduplicating by file+line."""
    wiki_results = [
        SearchResult(collection="wiki", file="a.md", line=1, score=0.9, content="A"),
        SearchResult(collection="wiki", file="b.md", line=1, score=0.7, content="B"),
    ]
    chat_results = [
        SearchResult(collection="chats", file="c.md", line=1, score=0.8, content="C"),
        SearchResult(collection="chats", file="a.md", line=1, score=0.6, content="A from chats"),
    ]

    merged = merge_multi_collection([wiki_results, chat_results], top_k=10)
    keys = [(r.file, r.line) for r in merged]
    # a.md:1 appears in both lists, should be ranked high and deduplicated
    assert keys.count(("a.md", 1)) == 1
    assert len(merged) == 3  # a.md, b.md, c.md (deduplicated)


class _FakeEncoder:
    def encode_query(self, _query):
        return "fake-embedding"


class _FakeTrace:
    def __init__(self, **_kwargs):
        pass

    def save(self):
        pass


def test_cli_search_json_no_indexes_stays_valid_json(monkeypatch):
    """--json must keep stdout machine-readable even when no indexes exist."""
    runner = CliRunner()

    monkeypatch.setattr("agentkb.wiki.ensure_search_store", lambda *, json_output=False: None)
    monkeypatch.setattr("agentkb.chats.ensure_search_store", lambda *, json_output=False: None)

    result = runner.invoke(cli.main, ["search", "query", "--json"])

    assert result.exit_code == 0
    assert '"results": []' in result.output
    assert '"message": "[agentkb] No indexes found.' in result.output


def test_cli_search_json_sends_status_to_stderr_not_stdout(monkeypatch):
    """Status chatter should go to stderr so stdout remains pure JSON."""
    from agentkb.output import echo_status as real_echo_status

    def fake_ensure_wiki_store(*, json_output=False):
        real_echo_status("[agentkb] Updating Wiki index...", json_output=json_output)
        return object()

    monkeypatch.setattr("agentkb.wiki.ensure_search_store", fake_ensure_wiki_store)
    monkeypatch.setattr("agentkb.chats.ensure_search_store", lambda *, json_output=False: None)

    monkeypatch.setattr("agentkb.cli.get_encoder", lambda: _FakeEncoder())
    monkeypatch.setattr("agentkb.cli.DEFAULT_MODEL", "fake-model")
    monkeypatch.setattr("agentkb.cli.merge_query_with_pattern", lambda query, pattern: query)
    monkeypatch.setattr("agentkb.cli.run_search", lambda **kwargs: [
        SearchResult(collection="wiki", file="tools/test.md", line=1, score=0.9, content="A")
    ])
    monkeypatch.setattr("agentkb.cli.merge_multi_collection", lambda results, top_k: results[0])
    monkeypatch.setattr("agentkb.cli.SearchTrace", _FakeTrace)

    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        cli.search.callback(
            query="query",
            scope="wiki",
            pattern=None,
            fixed=False,
            word=False,
            files_only=False,
            full_content=False,
            top_k=15,
            context_lines=6,
            json_output=True,
            include=(),
            exclude=(),
            exclude_dir=(),
            semantic_only=False,
            no_refresh=False,
        )

    assert '"results": [' in stdout.getvalue()
    assert '"filename": "test.md"' in stdout.getvalue()
    assert '"content"' not in stdout.getvalue()
    assert "[agentkb] Updating Wiki index..." not in stdout.getvalue()
    assert "[agentkb] Updating Wiki index..." in stderr.getvalue()


def test_cli_search_json_full_content_is_opt_in(monkeypatch):
    """--json omits section content unless -c is supplied."""
    monkeypatch.setattr("agentkb.wiki.ensure_search_store", lambda *, json_output=False: object())
    monkeypatch.setattr("agentkb.chats.ensure_search_store", lambda *, json_output=False: None)
    monkeypatch.setattr("agentkb.cli.get_encoder", lambda: _FakeEncoder())
    monkeypatch.setattr("agentkb.cli.DEFAULT_MODEL", "fake-model")
    monkeypatch.setattr("agentkb.cli.merge_query_with_pattern", lambda query, pattern: query)
    monkeypatch.setattr("agentkb.cli.run_search", lambda **kwargs: [
        SearchResult(
            collection="wiki",
            file="/tmp/agentkb/wiki/wiki/tools/test.md",
            line=1,
            score=0.9,
            content="structured",
            raw_content="raw section",
        )
    ])
    monkeypatch.setattr("agentkb.cli.SearchTrace", _FakeTrace)

    stdout = StringIO()
    with redirect_stdout(stdout):
        cli.search.callback(
            query="query",
            scope="wiki",
            pattern=None,
            fixed=False,
            word=False,
            files_only=False,
            full_content=True,
            top_k=15,
            context_lines=6,
            json_output=True,
            include=(),
            exclude=(),
            exclude_dir=(),
            semantic_only=False,
            no_refresh=False,
        )

    assert '"content": "raw section"' in stdout.getvalue()


def test_cli_search_json_chat_reindex_stays_valid_json(monkeypatch, tmp_path):
    """A chat reindex during search must keep stdout as valid JSON."""
    chats_root = tmp_path / "chats"
    sessions_dir = chats_root / "sessions"
    readable_dir = chats_root / "readable"
    sessions_dir.mkdir(parents=True)
    readable_dir.mkdir(parents=True)

    monkeypatch.setattr(cli.paths, "chats_dir", lambda: chats_root)
    monkeypatch.setattr(cli.paths, "chats_sessions_dir", lambda: sessions_dir)
    monkeypatch.setattr(cli.paths, "chats_readable_dir", lambda: readable_dir)
    monkeypatch.setattr("agentkb.store.IndexStore", lambda _path: object())
    monkeypatch.setattr("agentkb.chats.renderer.migrate_sessions_layout", lambda _sessions_dir: False)
    monkeypatch.setattr("agentkb.chats.renderer.export_all_sessions", lambda _sessions_dir: {"copied": 0, "skipped": 0, "total": 0})
    monkeypatch.setattr("agentkb.chats.renderer.export_readable", lambda _sessions_dir, _readable_dir: {"generated": 0})

    seen = {}

    def fake_build_chat_index(projects_dir, index_dir, model_name=None, incremental=True,
                              rebuild=False, tracked_only=False, json_output=False):
        seen["json_output"] = json_output
        echo_status("[agentkb] Chat index: fake rebuild", json_output=json_output)
        index_dir.mkdir(parents=True, exist_ok=True)
        return {"sessions_parsed": 0, "chunks_indexed": 0}

    # agentkb.chats re-exports these at module load, so patch the re-exports.
    monkeypatch.setattr("agentkb.chats.build_chat_index", fake_build_chat_index)
    monkeypatch.setattr("agentkb.chats.chat_index_is_stale", lambda _readable_dir, _index_dir: False)
    monkeypatch.setattr("agentkb.cli.get_encoder", lambda: _FakeEncoder())
    monkeypatch.setattr("agentkb.cli.DEFAULT_MODEL", "fake-model")
    monkeypatch.setattr("agentkb.cli.merge_query_with_pattern", lambda query, pattern: query)
    monkeypatch.setattr("agentkb.cli.run_search", lambda **kwargs: [])
    monkeypatch.setattr("agentkb.cli.SearchTrace", _FakeTrace)

    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        cli.search.callback(
            query="query",
            scope="chats",
            pattern=None,
            fixed=False,
            word=False,
            files_only=False,
            full_content=False,
            top_k=15,
            context_lines=6,
            json_output=True,
            include=(),
            exclude=(),
            exclude_dir=(),
            semantic_only=False,
            no_refresh=False,
        )

    assert seen["json_output"] is True
    assert '"results": []' in stdout.getvalue()
    assert "[agentkb] Chat index: fake rebuild" not in stdout.getvalue()
    assert "[agentkb] Chat index: fake rebuild" in stderr.getvalue()


def _build_real_index(
    index_dir: Path,
    content_root: Path,
    collection: str,
    *,
    legacy_mappings: bool = False,
) -> np.ndarray:
    """Build a small usable FastPLAID index without loading an encoder model."""
    content_root.mkdir(parents=True, exist_ok=True)
    store = IndexStore(index_dir, content_root=content_root)
    store.create()
    docs = [
        {
            "collection": collection,
            "file": f"existing-{number}.md",
            "line": 1,
            "name": f"Existing {number}",
            "unit_type": "chunk",
            "content": f"existing indexed content {number}",
            "raw_content": f"existing indexed content {number}",
        }
        for number in range(32)
    ]
    ids = store.add_documents(docs)
    rng = np.random.default_rng(7)
    embeddings = [
        rng.normal(size=(16, 128)).astype(np.float32)
        for _ in docs
    ]
    embeddings = [
        embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
        for embedding in embeddings
    ]
    store.append_plaid_index(ids, embeddings)
    store.save_state({"existing.md": "unchanged"})
    store.close()

    # Exercise the same WAL-mode metadata shape that previously created
    # -wal/-shm files when opened with mode=ro alone.
    conn = sqlite3.connect(index_dir / "metadata.db")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.close()

    if legacy_mappings:
        plaid_dir = index_dir / "plaid"
        for stem in ("documents_ids_to_plaid_ids", "plaid_ids_to_documents_ids"):
            sqlite_path = plaid_dir / f"{stem}.sqlite"
            uri = f"{sqlite_path.resolve().as_uri()}?mode=ro&immutable=1"
            with sqlite3.connect(uri, uri=True) as conn:
                rows = conn.execute('SELECT key, value FROM "unnamed"').fetchall()
            mapping = {key: pickle.loads(value) for key, value in rows}
            if stem == "plaid_ids_to_documents_ids":
                mapping = {int(key): value for key, value in mapping.items()}
            with (plaid_dir / f"{stem}.pkl").open("wb") as handle:
                pickle.dump(mapping, handle)
            sqlite_path.unlink()

    return embeddings[0]


def _artifact_snapshot(path: Path) -> dict[str, tuple[bool, int, str, int]]:
    """Capture existence, content hash, size, and mtime for a complete artifact tree."""
    candidates = [path]
    if path.is_dir():
        candidates.extend(sorted(path.rglob("*")))

    snapshot = {}
    for candidate in candidates:
        relative = "." if candidate == path else str(candidate.relative_to(path))
        exists = candidate.exists()
        if candidate.is_file():
            content = candidate.read_bytes()
        elif candidate.is_dir():
            content = "\n".join(sorted(child.name for child in candidate.iterdir())).encode()
        else:
            content = b""
        mtime = candidate.stat().st_mtime_ns if exists else 0
        snapshot[relative] = (
            exists,
            len(content),
            hashlib.sha256(content).hexdigest(),
            mtime,
        )
    return snapshot


class _FixtureEncoder:
    def __init__(self, query_embedding):
        self.query_embedding = query_embedding

    def encode_query(self, _query):
        return self.query_embedding


@pytest.mark.parametrize(
    ("legacy_mappings", "traceability_present"),
    [(False, False), (True, True)],
)
def test_cli_search_no_refresh_real_semantic_path_preserves_source_artifacts(
    monkeypatch, tmp_path, legacy_mappings, traceability_present
):
    """The production semantic path leaves current/legacy indexes and trace state untouched."""
    wiki_root = tmp_path / "wiki"
    query_embedding = _build_real_index(
        wiki_root / ".index",
        wiki_root,
        "wiki",
        legacy_mappings=legacy_mappings,
    )
    trace_db = tmp_path / "traceability.db"
    if traceability_present:
        with sqlite3.connect(trace_db) as conn:
            conn.execute("CREATE TABLE existing_trace (value TEXT)")
            conn.execute("INSERT INTO existing_trace VALUES ('untouched')")

    monkeypatch.setattr(cli.paths, "agentkb_home", lambda: tmp_path)
    monkeypatch.setattr(cli.paths, "wiki_dir", lambda: wiki_root)
    monkeypatch.setattr(
        "agentkb.wiki.ensure_search_store",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("wiki refresh called")),
    )
    monkeypatch.setattr(
        "agentkb.cli.get_encoder",
        lambda: _FixtureEncoder(query_embedding),
    )
    monkeypatch.setattr(
        "agentkb.cli.SearchTrace",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("trace created")),
    )

    watched = [
        wiki_root / ".index",
        trace_db,
        Path(f"{trace_db}-wal"),
        Path(f"{trace_db}-shm"),
        Path(f"{trace_db}-journal"),
    ]
    before = {str(path): _artifact_snapshot(path) for path in watched}
    temp_before = set(Path(tempfile.gettempdir()).glob("agentkb-plaid-readonly-*"))

    result = CliRunner().invoke(
        cli.main,
        ["search", "--no-refresh", "--semantic-only", "-s", "wiki", "existing"],
    )

    assert result.exit_code == 0, result.output
    assert "[wiki]" in result.output
    assert {str(path): _artifact_snapshot(path) for path in watched} == before
    assert set(Path(tempfile.gettempdir()).glob("agentkb-plaid-readonly-*")) == temp_before
    if legacy_mappings:
        assert not (wiki_root / ".index/plaid/documents_ids_to_plaid_ids.sqlite").exists()
        assert not (wiki_root / ".index/plaid/plaid_ids_to_documents_ids.sqlite").exists()


def test_cli_search_no_refresh_fails_when_requested_index_is_missing(monkeypatch, tmp_path):
    """Every requested scope must already have a usable index."""
    wiki_root = tmp_path / "wiki"
    monkeypatch.setattr(cli.paths, "wiki_dir", lambda: wiki_root)

    result = CliRunner().invoke(
        cli.main, ["search", "--no-refresh", "-s", "wiki", "query"]
    )

    assert result.exit_code == 1
    assert "--no-refresh requires a usable wiki index" in result.output
    assert "missing metadata database" in result.output
    assert "Run `agentkb index`" in result.output


def test_cli_search_no_refresh_missing_plaid_creates_no_artifact(monkeypatch, tmp_path):
    """Missing PLAID fails without creating the directory or mapping files."""
    wiki_root = tmp_path / "wiki"
    index_dir = wiki_root / ".index"
    store = IndexStore(index_dir)
    store.create()
    store.close()
    monkeypatch.setattr(cli.paths, "wiki_dir", lambda: wiki_root)
    before = _artifact_snapshot(index_dir)

    result = CliRunner().invoke(
        cli.main, ["search", "--no-refresh", "-s", "wiki", "query"]
    )

    assert result.exit_code == 1
    assert "--no-refresh requires a usable wiki index" in result.output
    assert "missing or empty PLAID index" in result.output
    assert _artifact_snapshot(index_dir) == before
    assert not (index_dir / "plaid").exists()


def test_cli_search_no_refresh_corrupt_plaid_preserves_every_artifact(
    monkeypatch, tmp_path
):
    """Corrupt FastPLAID fails through the real loader without repairing source files."""
    wiki_root = tmp_path / "wiki"
    index_dir = wiki_root / ".index"
    query_embedding = _build_real_index(index_dir, wiki_root, "wiki")
    (index_dir / "plaid/fast_plaid_index/metadata.json").write_text("{corrupt")
    monkeypatch.setattr(cli.paths, "wiki_dir", lambda: wiki_root)
    monkeypatch.setattr(
        "agentkb.cli.get_encoder",
        lambda: _FixtureEncoder(query_embedding),
    )
    before = _artifact_snapshot(index_dir)

    result = CliRunner().invoke(
        cli.main,
        ["search", "--no-refresh", "--semantic-only", "-s", "wiki", "query"],
    )

    assert result.exit_code == 1
    assert "--no-refresh could not use the existing wiki index" in result.output
    assert _artifact_snapshot(index_dir) == before
