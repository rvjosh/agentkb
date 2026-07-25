from __future__ import annotations

import ast
import sys
from types import SimpleNamespace
from pathlib import Path


ADAPTER = (
    Path(__file__).parents[1] / "src" / "agentkb" / "modal_backend" / "app.py"
)


def _decorator_keywords(node: ast.FunctionDef | ast.ClassDef, name: str) -> dict:
    for decorator in node.decorator_list:
        if (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == name
        ):
            return {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in decorator.keywords
                if keyword.arg
                and isinstance(
                    keyword.value,
                    (ast.Constant, ast.Dict, ast.List, ast.Tuple),
                )
            }
    raise AssertionError(f"missing @{name} decorator on {node.name}")


def test_adapter_has_private_named_resources_and_no_web_endpoints():
    source = ADAPTER.read_text()
    assert 'APP_NAME = "agentkb"' in source
    assert 'VOLUME_NAME = "agentkb-data"' in source
    assert "create_if_missing=False" in source
    assert "@modal.web_endpoint" not in source
    assert "@modal.asgi_app" not in source
    assert "@modal.wsgi_app" not in source


def test_gpu_image_adds_agentkb_before_model_download():
    source = ADAPTER.read_text()
    add_source = source.index(
        '.add_local_dir("src/agentkb", remote_path="/root/agentkb", copy=True)'
    )
    download_model = source.index(".run_function(")
    assert add_source < download_model


def test_builder_and_search_resource_contract():
    tree = ast.parse(ADAPTER.read_text())
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    classes = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }

    builder = _decorator_keywords(functions["build_generation"], "function")
    assert builder == {
        "gpu": "L4",
        "min_containers": 0,
        "max_containers": 1,
        "buffer_containers": 0,
        "single_use_containers": True,
        "timeout": 3600,
        "startup_timeout": 300,
    }

    search = _decorator_keywords(classes["GenerationSearch"], "cls")
    assert search == {
        "gpu": "T4",
        "min_containers": 0,
        "max_containers": 1,
        "buffer_containers": 0,
        "scaledown_window": 600,
        "enable_memory_snapshot": False,
        "timeout": 900,
        "startup_timeout": 300,
    }
    class_source = ast.get_source_segment(ADAPTER.read_text(), classes["GenerationSearch"])
    assert "generation_id: str = modal.parameter()" in class_source
    assert "resolve_current" not in class_source


def test_adapter_exposes_only_sdk_functions_and_generation_methods():
    source = ADAPTER.read_text()
    for function_name in (
        "build_generation",
        "status",
        "warm_current",
        "search_current",
    ):
        assert f"def {function_name}(" in source
    assert "@modal.method()" in source
    assert "def warm(" in source
    assert "def search(" in source


def test_encoder_device_is_forwarded_without_loading_a_real_model(monkeypatch):
    calls = []
    fake_models = SimpleNamespace(
        ColBERT=lambda **kwargs: calls.append(kwargs) or object()
    )
    monkeypatch.setitem(sys.modules, "pylate", SimpleNamespace(models=fake_models))

    from agentkb.encoder import ColBERTEncoder

    encoder = ColBERTEncoder("test-model", device="cuda")
    encoder._load()
    assert calls == [
        {"model_name_or_path": "test-model", "device": "cuda"}
    ]
