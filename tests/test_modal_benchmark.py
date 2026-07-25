"""Focused tests for the contained Modal benchmark helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.modal_benchmark.common import (
    BATCH_SIZE,
    compare_query_results,
    percentile,
    snapshot_documents,
    validate_tmp_root,
    write_snapshot,
)


def _wiki(root: Path) -> Path:
    (root / "wiki").mkdir(parents=True)
    (root / "sources").mkdir()
    (root / "wiki" / "b.md").write_text("# B\n\nSecond page")
    (root / "wiki" / "a.md").write_text(
        "---\ntitle: A page\ntags: [test]\n---\n# First\n\nBody\n## Nested\n\nMore"
    )
    (root / "sources" / "source.rst").write_text(
        "Source\n======\n\nRaw source body"
    )
    return root


def test_snapshot_documents_is_deterministic_and_unique(tmp_path):
    root = _wiki(tmp_path / "corpus")
    first = snapshot_documents(root)
    second = snapshot_documents(root)
    assert first == second
    assert len(first) == len({doc["canonical_id"] for doc in first})
    keys = [(doc["collection"], doc["file"], doc["line"]) for doc in first]
    assert keys == sorted(keys)
    assert first[0]["content"].startswith("[wiki]")


def test_write_snapshot_selects_exact_deterministic_batch(tmp_path, monkeypatch):
    docs = []
    for index in range(BATCH_SIZE + 3):
        docs.append(
            {
                "collection": "wiki",
                "file": f"wiki/{index:04}.md",
                "line": 1,
                "name": str(index),
                "unit_type": "chunk",
                "content": f"[wiki] {index}",
                "raw_content": str(index),
                "title": str(index),
                "section": "(full page)",
                "tags": [],
                "canonical_id": f"{index:064x}",
            }
        )
    monkeypatch.setattr(
        "experiments.modal_benchmark.common.snapshot_documents", lambda _: docs
    )
    monkeypatch.setattr(
        "experiments.modal_benchmark.common.WIKI_SPEC.list_files",
        lambda _: {"one": object()},
    )
    output = Path("/tmp") / f"agentkb-modal-benchmark-test-{tmp_path.name}"
    manifest = write_snapshot(tmp_path, output)
    try:
        assert manifest["batch_count"] == BATCH_SIZE
        assert sum(1 for _ in (output / "batch-1411.jsonl").open()) == BATCH_SIZE
        assert json.loads((output / "manifest.json").read_text()) == manifest
    finally:
        for child in output.iterdir():
            child.unlink()
        output.rmdir()


def test_validate_tmp_root_rejects_broad_or_unrelated_paths(tmp_path):
    with pytest.raises(ValueError):
        validate_tmp_root(Path("/tmp"))
    with pytest.raises(ValueError):
        validate_tmp_root(tmp_path / "agentkb-modal-benchmark-x")
    with pytest.raises(ValueError):
        validate_tmp_root(Path("/tmp/unrelated"))


def test_percentile_uses_linear_interpolation():
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert percentile([1.0, 2.0, 3.0], 0.95) == pytest.approx(2.9)


def test_compare_query_results_checks_order_and_score_tolerance():
    local = [
        {"label": "a", "result_ids": ["x", "y"], "scores": [2.0, 1.0]},
        {"label": "b", "result_ids": ["z"], "scores": [3.0]},
    ]
    remote = [
        {"label": "a", "result_ids": ["x", "y"], "scores": [2.0001, 1.0]},
        {"label": "b", "result_ids": ["z"], "scores": [3.0001]},
    ]
    result = compare_query_results(local, remote, score_atol=0.001)
    assert result["all_top_k_ids_equal"]
    assert result["all_scores_within_tolerance"]


def test_compare_query_results_detects_ranking_change():
    local = [{"label": "a", "result_ids": ["x", "y"], "scores": [2.0, 1.0]}]
    remote = [{"label": "a", "result_ids": ["y", "x"], "scores": [2.0, 1.0]}]
    result = compare_query_results(local, remote)
    assert not result["all_top_k_ids_equal"]
