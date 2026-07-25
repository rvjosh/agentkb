"""Private, bounded Modal application for the AgentKB GPU follow-up benchmark."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import modal


APP_NAME = "agentkb-gpu-benchmark-20260725"
VOLUME_NAME = "agentkb-gpu-benchmark-20260725-data"
VOLUME_ROOT = Path("/agentkb-gpu-benchmark-data")
CORPUS_DIR = VOLUME_ROOT / "corpus"
GENERATION_DIR = VOLUME_ROOT / "generation"
REPRESENTATIVE_WARMUP_QUERY = "agent knowledge retrieval"


def download_model() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download("lightonai/GTE-ModernColBERT-v1")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "click==8.3.2",
        "fast-plaid==1.3.0.290",
        "numpy==2.4.4",
        "pylate==1.4.0",
        "pyyaml==6.0.3",
        "scikit-learn==1.8.0",
        "sentence-transformers==5.1.1",
        "sqlitedict==2.1.0",
        "torch==2.9.0",
        "transformers==4.56.2",
    )
    .env(
        {
            "HF_HOME": "/models",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "PYTHONPATH": "/root",
        }
    )
    .run_function(
        download_model,
        env={"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"},
    )
    .add_local_dir("src/agentkb", remote_path="/root/agentkb", copy=True)
    .add_local_dir(
        "experiments/modal_benchmark",
        remote_path="/root/experiments/modal_benchmark",
        copy=True,
    )
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
app = modal.App(APP_NAME)


def gpu_options(gpu: str) -> dict[str, Any]:
    """Return the bounded SDK-only configuration shared by every remote worker."""
    return {
        "image": image,
        "gpu": gpu,
        "volumes": {VOLUME_ROOT: volume},
        "min_containers": 0,
        "max_containers": 1,
        "buffer_containers": 0,
        "scaledown_window": 15,
        "startup_timeout": 300,
    }


class GpuEncoder:
    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        if self._model is None:
            from pylate import models

            self._model = models.ColBERT(
                model_name_or_path="lightonai/GTE-ModernColBERT-v1",
                device="cuda",
            )

    @property
    def dim(self) -> int:
        self.load()
        return self._model.get_sentence_embedding_dimension()

    def encode_documents(
        self, texts: list[str], batch_size: int = 32
    ) -> list[Any]:
        import numpy as np

        self.load()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            is_query=False,
            show_progress_bar=False,
        )
        return [np.array(embedding, dtype=np.float32) for embedding in embeddings]

    def encode_query(self, query: str) -> Any:
        import numpy as np

        self.load()
        embeddings = self._model.encode(
            [query],
            batch_size=1,
            is_query=True,
            show_progress_bar=False,
        )
        return np.array(embeddings[0], dtype=np.float32)


def build_scratch_index(
    records: list[dict[str, Any]], index_dir: Path
) -> dict[str, Any]:
    from agentkb.store import IndexStore
    from experiments.modal_benchmark.common import (
        indexable_docs,
        validate_generation,
    )

    docs = indexable_docs(records)
    texts = [doc["content"] for doc in docs]
    encoder = GpuEncoder()
    model_started = time.perf_counter()
    _ = encoder.dim
    model_load_ms = (time.perf_counter() - model_started) * 1000
    total_started = time.perf_counter()
    encode_started = time.perf_counter()
    embeddings = encoder.encode_documents(texts)
    encode_ms = (time.perf_counter() - encode_started) * 1000
    store = IndexStore(index_dir)
    sqlite_started = time.perf_counter()
    store.create()
    ids = store.add_documents(docs)
    sqlite_ms = (time.perf_counter() - sqlite_started) * 1000
    plaid_started = time.perf_counter()
    store.append_plaid_index(ids, embeddings)
    plaid_ms = (time.perf_counter() - plaid_started) * 1000
    store.close()
    validation = validate_generation(index_dir, len(docs))
    total_ms = (time.perf_counter() - total_started) * 1000
    return {
        "count": len(docs),
        "model_load_ms": model_load_ms,
        "encode_ms": encode_ms,
        "sqlite_ms": sqlite_ms,
        "plaid_ms": plaid_ms,
        "total_ms": total_ms,
        "documents_per_second": len(docs) / (total_ms / 1000),
        "validation": validation,
    }


@app.function(**gpu_options("T4"), timeout=1800)
def build_generation() -> dict[str, Any]:
    """Build the one immutable search generation used by every search variant."""
    from experiments.modal_benchmark.common import load_jsonl

    if GENERATION_DIR.exists():
        raise RuntimeError("immutable benchmark generation already exists")
    records = load_jsonl(CORPUS_DIR / "corpus.jsonl")
    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text())
    with tempfile.TemporaryDirectory(
        prefix="agentkb-gpu-benchmark-generation-"
    ) as scratch:
        index_dir = Path(scratch) / "index"
        result = build_scratch_index(records, index_dir)
        (index_dir / "state.json").write_text(
            json.dumps(
                {
                    "__model__": manifest["model"],
                    "__benchmark_corpus_hash__": manifest["corpus_hash"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        result["validation"]["tree_hash"] = (
            __import__(
                "experiments.modal_benchmark.common",
                fromlist=["file_tree_hash"],
            ).file_tree_hash(index_dir)
        )
        shutil.copytree(index_dir, GENERATION_DIR)
        volume.commit()
    return result


def benchmark_batch(limit: int) -> dict[str, Any]:
    from experiments.modal_benchmark.common import load_jsonl

    records = load_jsonl(CORPUS_DIR / "batch-1411.jsonl")[:limit]
    with tempfile.TemporaryDirectory(
        prefix="agentkb-gpu-benchmark-batch-"
    ) as scratch:
        return build_scratch_index(records, Path(scratch) / "index")


@app.function(**gpu_options("T4"), timeout=900, single_use_containers=True)
def benchmark_batch_t4(limit: int) -> dict[str, Any]:
    return benchmark_batch(limit)


@app.function(**gpu_options("L4"), timeout=900, single_use_containers=True)
def benchmark_batch_l4(limit: int) -> dict[str, Any]:
    return benchmark_batch(limit)


def initialize_index(encoder: GpuEncoder) -> tuple[Any, dict[str, float], Any]:
    """Copy and open the immutable generation after container/snapshot restore."""
    from agentkb.store import IndexStore

    scratch = tempfile.TemporaryDirectory(
        prefix="agentkb-gpu-benchmark-search-"
    )
    copy_started = time.perf_counter()
    local_index = Path(scratch.name) / "index"
    shutil.copytree(GENERATION_DIR, local_index)
    copy_ms = (time.perf_counter() - copy_started) * 1000
    metadata_started = time.perf_counter()
    store = IndexStore(local_index, read_only=True)
    store.validate_for_search()
    metadata_ms = (time.perf_counter() - metadata_started) * 1000
    plaid_started = time.perf_counter()
    store._load_plaid_index()
    plaid_ms = (time.perf_counter() - plaid_started) * 1000
    return store, {
        "volume_copy_ms": copy_ms,
        "metadata_load_ms": metadata_ms,
        "index_load_ms": plaid_ms,
        "post_model_initialization_ms": copy_ms + metadata_ms + plaid_ms,
    }, scratch


def initialize_baseline_search() -> tuple[Any, Any, dict[str, float], Any]:
    encoder = GpuEncoder()
    model_started = time.perf_counter()
    _ = encoder.dim
    model_ms = (time.perf_counter() - model_started) * 1000
    store, index_initialization, scratch = initialize_index(encoder)
    initialization = {
        "model_load_ms": model_ms,
        **index_initialization,
        "initialization_ms": (
            model_ms + index_initialization["post_model_initialization_ms"]
        ),
    }
    return store, encoder, initialization, scratch


def run_queries(
    store: Any, encoder: Any, queries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    from experiments.modal_benchmark.common import run_timed_query

    timings = []
    for query in queries:
        result = run_timed_query(store, encoder, query)
        timings.append(
            {
                key: value
                for key, value in result.items()
                if key not in {"result_ids", "scores"}
            }
        )
    return timings


class _BaselineSearch:
    def _enter(self) -> None:
        self.container_started = time.perf_counter()
        self.store, self.encoder, self.initialization, self.scratch = (
            initialize_baseline_search()
        )

    def _run(self, queries: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "initialization": self.initialization,
            "queries": run_queries(self.store, self.encoder, queries),
            "container_end_to_end_ms": (
                time.perf_counter() - self.container_started
            )
            * 1000,
        }

    @modal.exit()
    def _exit(self) -> None:
        self.store.close()
        self.scratch.cleanup()


@app.cls(**gpu_options("T4"), timeout=900, single_use_containers=True)
class T4ColdSearch(_BaselineSearch):
    @modal.enter()
    def enter(self) -> None:
        self._enter()

    @modal.method()
    def run(self, query: dict[str, Any]) -> dict[str, Any]:
        return self._run([query])


@app.cls(**gpu_options("T4"), timeout=900)
class T4WarmSearch(_BaselineSearch):
    @modal.enter()
    def enter(self) -> None:
        self._enter()

    @modal.method()
    def run(self, queries: list[dict[str, Any]]) -> dict[str, Any]:
        return self._run(queries)


@app.cls(**gpu_options("L4"), timeout=900, single_use_containers=True)
class L4ColdSearch(_BaselineSearch):
    @modal.enter()
    def enter(self) -> None:
        self._enter()

    @modal.method()
    def run(self, query: dict[str, Any]) -> dict[str, Any]:
        return self._run([query])


@app.cls(**gpu_options("L4"), timeout=900)
class L4WarmSearch(_BaselineSearch):
    @modal.enter()
    def enter(self) -> None:
        self._enter()

    @modal.method()
    def run(self, queries: list[dict[str, Any]]) -> dict[str, Any]:
        return self._run(queries)


@app.cls(
    **gpu_options("T4"),
    timeout=900,
    single_use_containers=True,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class T4SnapshotSearch:
    @modal.enter(snap=True)
    def initialize_model_for_snapshot(self) -> None:
        model_started = time.perf_counter()
        self.encoder = GpuEncoder()
        _ = self.encoder.dim
        model_load_ms = (time.perf_counter() - model_started) * 1000
        warmup_started = time.perf_counter()
        _ = self.encoder.encode_query(REPRESENTATIVE_WARMUP_QUERY)
        warmup_ms = (time.perf_counter() - warmup_started) * 1000
        self.snapshot_initialization = {
            "model_load_ms": model_load_ms,
            "representative_forward_ms": warmup_ms,
            "snapshotted_initialization_ms": model_load_ms + warmup_ms,
        }
        print("AGENTKB_SNAPSHOT_MODEL_READY")

    @modal.enter()
    def initialize_index_after_restore(self) -> None:
        self.restore_started = time.perf_counter()
        self.store, self.post_restore_initialization, self.scratch = (
            initialize_index(self.encoder)
        )
        print("AGENTKB_SNAPSHOT_POST_RESTORE_READY")

    @modal.method()
    def run(self, query: dict[str, Any]) -> dict[str, Any]:
        return {
            "snapshot_initialization": self.snapshot_initialization,
            "post_restore_initialization": self.post_restore_initialization,
            "queries": run_queries(self.store, self.encoder, [query]),
            "post_restore_container_ms": (
                time.perf_counter() - self.restore_started
            )
            * 1000,
        }

    @modal.exit()
    def exit(self) -> None:
        self.store.close()
        self.scratch.cleanup()
