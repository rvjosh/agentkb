"""Private, bounded Modal application for the AgentKB benchmark."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import modal


APP_NAME = "agentkb-benchmark-20260725"
VOLUME_NAME = "agentkb-benchmark-20260725-data"
VOLUME_ROOT = Path("/agentkb-benchmark-data")
CORPUS_DIR = VOLUME_ROOT / "corpus"
GENERATION_DIR = VOLUME_ROOT / "generation"


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

GPU_OPTIONS = {
    "image": image,
    "gpu": "T4",
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


@app.function(**GPU_OPTIONS, timeout=1800)
def build_generation() -> dict[str, Any]:
    from experiments.modal_benchmark.common import load_jsonl

    records = load_jsonl(CORPUS_DIR / "corpus.jsonl")
    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text())
    with tempfile.TemporaryDirectory(
        prefix="agentkb-modal-benchmark-generation-"
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
        if GENERATION_DIR.exists():
            shutil.rmtree(GENERATION_DIR)
        shutil.copytree(index_dir, GENERATION_DIR)
        volume.commit()
    return result


@app.function(**GPU_OPTIONS, timeout=900)
def benchmark_batch(limit: int) -> dict[str, Any]:
    from experiments.modal_benchmark.common import load_jsonl

    records = load_jsonl(CORPUS_DIR / "batch-1411.jsonl")[:limit]
    with tempfile.TemporaryDirectory(
        prefix="agentkb-modal-benchmark-batch-"
    ) as scratch:
        return build_scratch_index(records, Path(scratch) / "index")


def initialize_search() -> tuple[Any, Any, dict[str, float], Any]:
    from agentkb.store import IndexStore

    scratch = tempfile.TemporaryDirectory(
        prefix="agentkb-modal-benchmark-search-"
    )
    copy_started = time.perf_counter()
    local_index = Path(scratch.name) / "index"
    shutil.copytree(GENERATION_DIR, local_index)
    copy_ms = (time.perf_counter() - copy_started) * 1000
    encoder = GpuEncoder()
    model_started = time.perf_counter()
    _ = encoder.dim
    model_ms = (time.perf_counter() - model_started) * 1000
    metadata_started = time.perf_counter()
    store = IndexStore(local_index, read_only=True)
    store.validate_for_search()
    metadata_ms = (time.perf_counter() - metadata_started) * 1000
    plaid_started = time.perf_counter()
    store._load_plaid_index()
    plaid_ms = (time.perf_counter() - plaid_started) * 1000
    return store, encoder, {
        "volume_copy_ms": copy_ms,
        "model_load_ms": model_ms,
        "metadata_load_ms": metadata_ms,
        "index_load_ms": plaid_ms,
        "initialization_ms": copy_ms + model_ms + metadata_ms + plaid_ms,
    }, scratch


@app.cls(**GPU_OPTIONS, timeout=900)
class WarmSearch:
    @modal.enter()
    def enter(self) -> None:
        self.store, self.encoder, self.initialization, self.scratch = (
            initialize_search()
        )

    @modal.method()
    def run(self, queries: list[dict[str, Any]], repetitions: int) -> dict[str, Any]:
        from experiments.modal_benchmark.common import run_timed_query

        results = []
        for _ in range(repetitions):
            for query in queries:
                results.append(run_timed_query(self.store, self.encoder, query))
        return {"initialization": self.initialization, "queries": results}


@app.function(
    **GPU_OPTIONS,
    timeout=900,
    single_use_containers=True,
)
def cold_search(query: dict[str, Any]) -> dict[str, Any]:
    from experiments.modal_benchmark.common import run_timed_query

    started = time.perf_counter()
    store, encoder, initialization, scratch = initialize_search()
    try:
        result = run_timed_query(store, encoder, query)
    finally:
        store.close()
        scratch.cleanup()
    return {
        "initialization": initialization,
        "query": result,
        "container_end_to_end_ms": (time.perf_counter() - started) * 1000,
    }
