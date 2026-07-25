from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agentkb.encoder import DEFAULT_MODEL
from agentkb.fast_plaid_immutable import inspect_immutable_premerged_artifacts
from agentkb.modal_backend import runtime
from agentkb.modal_backend.generations import staged_paths
from agentkb.store import IndexStore


GENERATION_ID = "g-20260725T123456Z-001122aabbcc"


def _record(index: int) -> dict:
    return {
        "canonical_id": hashlib.sha256(str(index).encode()).hexdigest(),
        "collection": ["wiki", "wiki:source", "chats"][index % 3],
        "content": f"content {index}",
        "file": f"collection/file-{index:04d}.md",
        "line": index + 1,
    }


def _stage(volume_root: Path, count: int) -> Path:
    corpus_path, manifest_path = staged_paths(volume_root, GENERATION_ID)
    corpus_path.parent.mkdir(parents=True)
    digest = hashlib.sha256()
    with corpus_path.open("wb") as corpus:
        for index in range(count):
            line = (
                json.dumps(
                    _record(index),
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            ).encode()
            corpus.write(line)
            digest.update(line)
    manifest_path.write_text(
        json.dumps(
            {
                "schema": 1,
                "generation_id": GENERATION_ID,
                "model": DEFAULT_MODEL,
                "corpus_count": count,
                "corpus_hash": digest.hexdigest(),
            }
        )
    )
    return corpus_path


class FakeEncoder:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.next_index = 0

    def encode_documents(self, texts: list[str]) -> list[np.ndarray]:
        self.batch_sizes.append(len(texts))
        embeddings = []
        for text in texts:
            index = int(text.removeprefix("content "))
            assert index == self.next_index
            self.next_index += 1
            embeddings.append(
                np.full((index % 3 + 1, 4), index, dtype=np.float32)
            )
        return embeddings


class FakeStore:
    def __init__(self) -> None:
        self.next_id = 1
        self.documents: list[dict] = []
        self.create_calls: list[dict] = []

    def add_documents(self, docs: list[dict]) -> list[int]:
        assert all("canonical_id" not in doc for doc in docs)
        ids = list(range(self.next_id, self.next_id + len(docs)))
        self.next_id += len(docs)
        self.documents.extend(docs)
        return ids

    def create_plaid_index(
        self,
        ids: list[int],
        embeddings: list[np.ndarray],
        *,
        n_samples_kmeans: int,
    ) -> None:
        assert len(ids) == len(embeddings)
        self.create_calls.append(
            {
                "ids": ids.copy(),
                "values": [float(embedding[0, 0]) for embedding in embeddings],
                "memmap_views": all(
                    isinstance(embedding, np.memmap) for embedding in embeddings
                ),
                "n_samples_kmeans": n_samples_kmeans,
            }
        )


def test_staged_validation_and_build_are_streaming_and_batch_bounded(tmp_path):
    count = runtime.DOCUMENT_BATCH_SIZE * 2 + 1
    corpus_path = _stage(tmp_path, count)

    loaded_path, manifest = runtime._load_staged(tmp_path, GENERATION_ID)
    assert loaded_path == corpus_path
    assert manifest["corpus_count"] == count

    encoder = FakeEncoder()
    store = FakeStore()
    stage_path = tmp_path / "embedding-stage"
    with runtime._EmbeddingStage(stage_path) as stage:
        built_count, batch_count = runtime._build_index_batches(
            runtime._iter_staged_records(corpus_path),
            encoder=encoder,
            store=store,
            embedding_stage=stage,
            expected_count=count,
        )
        metrics = runtime._create_global_plaid(
            embedding_stage=stage,
            store=store,
            corpus_hash=manifest["corpus_hash"],
            expected_count=count,
        )
        descriptors = [
            json.loads(line)
            for line in stage.descriptors_path.read_text().splitlines()
        ]

    assert built_count == count
    assert batch_count == math.ceil(count / runtime.DOCUMENT_BATCH_SIZE)
    assert encoder.batch_sizes == [
        runtime.DOCUMENT_BATCH_SIZE,
        runtime.DOCUMENT_BATCH_SIZE,
        1,
    ]
    assert max(encoder.batch_sizes) <= runtime.DOCUMENT_BATCH_SIZE
    assert [doc["content"] for doc in store.documents] == [
        f"content {index}" for index in range(count)
    ]
    assert len(descriptors) == count
    assert descriptors[0] == {
        "dimension": 4,
        "document_id": 1,
        "length": 1,
        "offset": 0,
    }
    assert descriptors[1]["offset"] == 8
    assert descriptors[-1]["offset"] < metrics["staged_embedding_bytes"]
    assert len(store.create_calls) == 1
    create = store.create_calls[0]
    assert create["memmap_views"]
    assert create["n_samples_kmeans"] == runtime.PLAID_KMEANS_SAMPLE_SIZE
    assert create["ids"] != list(range(1, count + 1))
    assert create["values"] == [float(doc_id - 1) for doc_id in create["ids"]]
    assert metrics["plaid_create_count"] == 1
    assert metrics["plaid_kmeans_sample_size"] == 16_384
    assert metrics["plaid_permutation_algorithm"] == "sha256-key-sort-v1"
    assert not stage_path.exists()


def test_corpus_permutation_is_hash_deterministic_and_complete():
    first = runtime._corpus_permutation("a" * 64, 1_000)
    assert first == runtime._corpus_permutation("a" * 64, 1_000)
    assert first != runtime._corpus_permutation("b" * 64, 1_000)
    assert sorted(first) == list(range(1_000))


def test_embedding_stage_cleans_payload_and_metadata_after_failure(tmp_path):
    stage_path = tmp_path / "failed-stage"
    with pytest.raises(RuntimeError, match="PLAID failed"):
        with runtime._EmbeddingStage(stage_path) as stage:
            stage.append([7], [np.ones((2, 3), dtype=np.float32)])

            class FailingStore:
                def create_plaid_index(self, *_: object, **__: object) -> None:
                    raise RuntimeError("PLAID failed")

            runtime._create_global_plaid(
                embedding_stage=stage,
                store=FailingStore(),
                corpus_hash="c" * 64,
                expected_count=1,
            )
    assert not stage_path.exists()


def test_store_uses_one_initial_pylate_add_with_explicit_kmeans_sample(
    tmp_path, monkeypatch
):
    calls: list[tuple[str, object]] = []

    class FakePlaid:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def add_documents(self, **kwargs):
            calls.append(("add", kwargs))

    monkeypatch.setitem(
        sys.modules,
        "pylate",
        SimpleNamespace(indexes=SimpleNamespace(PLAID=FakePlaid)),
    )
    store = IndexStore(tmp_path / "index")
    embeddings = [np.ones((2, 4), dtype=np.float16) for _ in range(3)]
    store.create_plaid_index(
        [11, 22, 33],
        embeddings,
        n_samples_kmeans=runtime.PLAID_KMEANS_SAMPLE_SIZE,
    )

    assert calls == [
        (
            "init",
            {
                "index_folder": str(tmp_path / "index"),
                "index_name": "plaid",
                "override": True,
                "n_samples_kmeans": 16_384,
            },
        ),
        (
            "add",
            {
                "documents_ids": ["11", "22", "33"],
                "documents_embeddings": embeddings,
            },
        ),
    ]


def test_remote_staged_validation_never_reads_the_whole_corpus():
    source = Path(runtime.__file__).read_text()
    load_staged = source[
        source.index("def _load_staged(") : source.index(
            "\ndef _iter_staged_records("
        )
    ]
    assert ".read_bytes()" not in load_staged
    assert 'corpus_path.open("rb")' in load_staged


def _write_search_generation(volume_root: Path) -> tuple[Path, dict]:
    generation = volume_root / "generations" / GENERATION_ID
    plaid = generation / "index" / "plaid"
    fast_plaid = plaid / "fast_plaid_index"
    fast_plaid.mkdir(parents=True)
    (generation / "index" / "metadata.db").write_bytes(b"sqlite")
    (fast_plaid / "metadata.json").write_text(
        json.dumps({"num_chunks": 1, "nbits": 4})
    )
    (fast_plaid / "doclens.0.json").write_text("[1]")
    np.save(fast_plaid / "centroids.npy", np.ones((2, 128), dtype=np.float16))
    np.save(fast_plaid / "avg_residual.npy", np.ones(1, dtype=np.float32))
    np.save(fast_plaid / "bucket_cutoffs.npy", np.ones(2, dtype=np.float32))
    np.save(fast_plaid / "bucket_weights.npy", np.ones(2, dtype=np.float32))
    np.save(fast_plaid / "ivf.npy", np.ones(1, dtype=np.int64))
    np.save(fast_plaid / "ivf_lengths.npy", np.ones(1, dtype=np.int32))
    np.save(fast_plaid / "merged_codes.npy", np.ones(1, dtype=np.int64))
    np.save(fast_plaid / "merged_residuals.npy", np.ones((1, 16), dtype=np.uint8))
    (plaid / "documents_ids_to_plaid_ids.sqlite").write_bytes(b"mapping")
    (plaid / "plaid_ids_to_documents_ids.sqlite").write_bytes(b"mapping")
    manifest = {
        "schema": 1,
        "generation_id": GENERATION_ID,
        "model": DEFAULT_MODEL,
        "corpus_count": 1,
        "corpus_hash": "a" * 64,
        "build": {
            "document_batch_size": runtime.DOCUMENT_BATCH_SIZE,
            "document_batch_count": 1,
            "embedding_dimension": 128,
            "staged_embedding_bytes": 256,
            "plaid_create_count": 1,
            "plaid_kmeans_sample_size": runtime.PLAID_KMEANS_SAMPLE_SIZE,
            "plaid_permutation_algorithm": runtime.PLAID_PERMUTATION_ALGORITHM,
        },
        "validation": {
            "sqlite_count": 1,
            "fts_count": 1,
            "plaid_mapping_count": 1,
            "plaid_reverse_mapping_count": 1,
            "index_tree_hash": "b" * 64,
            "immutable_premerged": inspect_immutable_premerged_artifacts(
                fast_plaid,
                expected_document_count=1,
                expected_embedding_dimension=128,
            ),
        },
    }
    (generation / "index" / "state.json").write_text(
        json.dumps(
            {
                "__model__": DEFAULT_MODEL,
                "__corpus_hash__": manifest["corpus_hash"],
            }
        )
    )
    (generation / "manifest.json").write_text(json.dumps(manifest))
    return generation, manifest


def test_search_build_certificate_accepts_published_generation(tmp_path):
    generation, manifest = _write_search_generation(tmp_path)

    runtime.validate_search_build_certificate(
        generation,
        manifest,
        GENERATION_ID,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("generation_id", "g-20260725T123456Z-aabbccddeeff", "generation_id"),
        ("model", "wrong/model", "manifest model"),
        ("corpus_count", 0, "corpus_count"),
        ("build", None, "build certificate"),
        ("validation", None, "validation certificate"),
    ],
)
def test_search_build_certificate_rejects_malformed_or_missing_certificate(
    tmp_path, field, value, message
):
    generation, manifest = _write_search_generation(tmp_path)
    manifest[field] = value

    with pytest.raises(ValueError, match=message):
        runtime.validate_search_build_certificate(
            generation,
            manifest,
            GENERATION_ID,
        )


def test_search_build_certificate_rejects_invalid_build_and_validation(tmp_path):
    generation, manifest = _write_search_generation(tmp_path)
    manifest["build"]["document_batch_count"] = 2
    with pytest.raises(ValueError, match="does not cover corpus_count"):
        runtime.validate_search_build_certificate(
            generation,
            manifest,
            GENERATION_ID,
        )

    manifest["build"]["document_batch_count"] = 1
    manifest["validation"]["sqlite_count"] = 2
    with pytest.raises(ValueError, match="does not match corpus_count"):
        runtime.validate_search_build_certificate(
            generation,
            manifest,
            GENERATION_ID,
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        "index/metadata.db",
        "index/state.json",
        "index/plaid/fast_plaid_index/metadata.json",
        "index/plaid/fast_plaid_index/merged_codes.npy",
        "index/plaid/fast_plaid_index/merged_residuals.npy",
        "index/plaid/documents_ids_to_plaid_ids.sqlite",
    ],
)
def test_search_build_certificate_rejects_missing_artifacts(
    tmp_path, relative_path
):
    generation, manifest = _write_search_generation(tmp_path)
    (generation / relative_path).unlink()

    with pytest.raises(FileNotFoundError):
        runtime.validate_search_build_certificate(
            generation,
            manifest,
            GENERATION_ID,
        )


def test_search_runtime_skips_full_validation_and_reports_startup_timings(
    tmp_path, monkeypatch
):
    _write_search_generation(tmp_path)

    class SearchEncoder:
        def __init__(self, model, *, device):
            assert model == DEFAULT_MODEL
            assert device == "cuda"
            self.dim = 128

    class SearchStore:
        def __init__(self, index_dir, *, read_only, immutable_premerged):
            assert index_dir == (
                tmp_path / "generations" / GENERATION_ID / "index"
            )
            assert read_only
            assert immutable_premerged["schema"] == 1
            self.validated = False
            self.loaded = False

        def validate_for_search(self):
            self.validated = True

        def _load_plaid_index(self):
            assert self.validated
            self.loaded = True

        def close(self):
            pass

    monkeypatch.setattr(runtime, "ColBERTEncoder", SearchEncoder)
    monkeypatch.setattr(runtime, "IndexStore", SearchStore)
    monkeypatch.setattr(
        runtime.shutil,
        "copytree",
        lambda *_args, **_kwargs: pytest.fail("production copied the generation"),
    )
    monkeypatch.setattr(
        runtime,
        "validate_index",
        lambda *_args, **_kwargs: pytest.fail("startup reran full validation"),
    )

    search_runtime = runtime.SearchRuntime(tmp_path, GENERATION_ID)
    try:
        result = search_runtime.warm()
        assert search_runtime.store.loaded
        assert set(result["startup_timing_ms"]) == {
            "artifact_mount",
            "certificate",
            "model",
            "index_load",
            "total",
        }
        assert all(value >= 0 for value in result["startup_timing_ms"].values())
    finally:
        search_runtime.close()
