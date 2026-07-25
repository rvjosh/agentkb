"""Focused tests for the contained Modal benchmark helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.modal_benchmark.common import (
    BATCH_SIZE,
    classify_snapshot_attempts,
    compare_query_results,
    evaluate_l4_gates,
    gpu_cost_usd,
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


def test_followup_cost_and_l4_gates_are_strict():
    assert gpu_cost_usd(10_000, 0.000164) == pytest.approx(0.00164)
    passing = evaluate_l4_gates(13_529.9, 41_369.9)
    assert passing["cold"]["passed"]
    assert passing["build_1411"]["passed"]
    failing = evaluate_l4_gates(13_530.0, 41_370.0)
    assert not failing["cold"]["passed"]
    assert not failing["build_1411"]["passed"]


def test_snapshot_classification_requires_proof_and_is_bounded():
    assert classify_snapshot_attempts(
        4,
        [2, 3],
        restoration_proof=True,
        creation_attempt_indexes=[0, 1],
    ) == ["snapshot_creation", "snapshot_creation", "restored", "restored"]
    assert classify_snapshot_attempts(
        3, [0, 1, 2], restoration_proof=False
    ) == ["unverified", "unverified", "unverified"]
    with pytest.raises(ValueError):
        classify_snapshot_attempts(9, [], restoration_proof=True)
    with pytest.raises(ValueError):
        classify_snapshot_attempts(3, [3], restoration_proof=True)
    with pytest.raises(ValueError):
        classify_snapshot_attempts(
            3, [1], restoration_proof=True, creation_attempt_indexes=[1]
        )


def test_original_modal_app_preserves_reproduction_contract():
    root = Path(__file__).parents[1] / "experiments" / "modal_benchmark"
    source = (root / "modal_app.py").read_text()
    assert 'APP_NAME = "agentkb-benchmark-20260725"' in source
    assert 'VOLUME_NAME = "agentkb-benchmark-20260725-data"' in source
    assert "GPU_OPTIONS = {" in source
    assert "class WarmSearch:" in source
    assert "def cold_search(" in source
    assert "def benchmark_batch(" in source
    assert "gpu_options(" not in source

    for caller in ("modal_client.py", "run_modal_suite.py"):
        caller_source = (root / caller).read_text()
        assert (
            "from experiments.modal_benchmark.modal_app import APP_NAME"
            in caller_source
        )
    assert "--summary-only" in (root / "README.md").read_text()


def test_followup_modal_app_uses_exact_bounded_private_configuration():
    root = (
        Path(__file__).parents[1]
        / "experiments"
        / "modal_benchmark"
    )
    source = (root / "gpu_followup_modal_app.py").read_text()
    assert 'APP_NAME = "agentkb-gpu-benchmark-20260725"' in source
    assert 'VOLUME_NAME = "agentkb-gpu-benchmark-20260725-data"' in source
    assert '"max_containers": 1' in source
    assert 'gpu_options("T4")' in source
    assert 'gpu_options("L4")' in source
    assert "enable_memory_snapshot=True" in source
    assert 'experimental_options={"enable_gpu_snapshot": True}' in source
    assert "@modal.enter(snap=True)" in source
    assert "@modal.web_endpoint" not in source
    assert "@modal.asgi_app" not in source
    assert "@modal.wsgi_app" not in source

    runner_source = (root / "followup_runner.py").read_text()
    assert (
        "from experiments.modal_benchmark.gpu_followup_modal_app import APP_NAME"
        in runner_source
    )
    assert 'choices=["build", "search", "batch", "snapshot"]' in runner_source

    readme = (root / "README.md").read_text()
    assert "-m experiments.modal_benchmark.gpu_followup_modal_app" in readme
    assert "--phase build" in readme
