"""Thin private Modal deployment adapter for AgentKB."""

from pathlib import Path
from typing import Any

import modal

from agentkb.modal_backend.generations import (
    read_status,
    resolve_current,
    validate_generation_id,
)


APP_NAME = "agentkb"
VOLUME_NAME = "agentkb-data"
VOLUME_ROOT = Path("/agentkb-data")


def _download_model() -> None:
    from agentkb.modal_backend.runtime import download_model

    download_model()


gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "click==8.3.2",
        "fast-plaid==1.3.0.290",
        "huggingface-hub==0.35.3",
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
    .add_local_dir("src/agentkb", remote_path="/root/agentkb", copy=True)
    .run_function(
        _download_model,
        env={"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"},
    )
)

router_image = modal.Image.debian_slim(
    python_version="3.12"
).add_local_dir("src/agentkb", remote_path="/root/agentkb", copy=True)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
app = modal.App(APP_NAME)
volume_mount = {VOLUME_ROOT: volume}


@app.function(
    image=gpu_image,
    gpu="L4",
    volumes=volume_mount,
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    single_use_containers=True,
    timeout=3600,
    startup_timeout=300,
)
def build_generation(generation_id: str) -> dict[str, Any]:
    from agentkb.modal_backend.runtime import build_and_publish

    validate_generation_id(generation_id)
    volume.reload()
    return build_and_publish(VOLUME_ROOT, generation_id, commit=volume.commit)


@app.cls(
    image=gpu_image,
    gpu="T4",
    volumes=volume_mount,
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=600,
    enable_memory_snapshot=False,
    timeout=900,
    startup_timeout=300,
)
class GenerationSearch:
    generation_id: str = modal.parameter()

    @modal.enter()
    def enter(self) -> None:
        from agentkb.modal_backend.runtime import SearchRuntime

        validate_generation_id(self.generation_id)
        volume.reload()
        self.runtime = SearchRuntime(VOLUME_ROOT, self.generation_id)

    @modal.method()
    def warm(self) -> dict[str, Any]:
        return self.runtime.warm()

    @modal.method()
    def search(self, query: str, k: int) -> dict[str, Any]:
        return self.runtime.search(query, k)

    @modal.exit()
    def exit(self) -> None:
        self.runtime.close()


@app.function(image=router_image, volumes=volume_mount, timeout=60)
def status() -> dict[str, Any]:
    volume.reload()
    return read_status(VOLUME_ROOT)


@app.function(image=router_image, volumes=volume_mount, timeout=900)
def warm_current() -> dict[str, Any]:
    volume.reload()
    generation_id = resolve_current(VOLUME_ROOT)
    return GenerationSearch(generation_id=generation_id).warm.remote()


@app.function(image=router_image, volumes=volume_mount, timeout=900)
def search_current(query: str, k: int) -> dict[str, Any]:
    volume.reload()
    generation_id = resolve_current(VOLUME_ROOT)
    return GenerationSearch(generation_id=generation_id).search.remote(query, k)
