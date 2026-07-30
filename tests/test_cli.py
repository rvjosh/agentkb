"""Tests for the top-level AgentKB CLI."""

import json
from importlib.metadata import version
from pathlib import Path

from click.testing import CliRunner

from agentkb.cli import main
from agentkb.encoder import DEFAULT_MODEL, ModelCacheMissingError


def test_version_reports_installed_distribution_version():
    result = CliRunner().invoke(main, ["--version"], prog_name="agentkb")

    assert result.exit_code == 0
    assert result.output == f"agentkb, version {version('agentkb')}\n"


def test_model_cache_json_is_stable_and_content_free(monkeypatch):
    monkeypatch.setattr(
        "agentkb.cli.require_cached_model",
        lambda model: Path("/private/cache/snapshots/secret-revision"),
    )

    result = CliRunner().invoke(main, ["model-cache", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "cached": True,
        "model": DEFAULT_MODEL,
        "recovery": None,
    }
    assert "private/cache" not in result.output


def test_model_cache_missing_json_exits_nonzero_with_recovery(monkeypatch):
    def missing(model):
        raise ModelCacheMissingError(model)

    monkeypatch.setattr("agentkb.cli.require_cached_model", missing)

    result = CliRunner().invoke(
        main, ["model-cache", "--model", "owner/model", "--json"],
    )

    assert result.exit_code == 1
    assert json.loads(result.output) == {
        "cached": False,
        "model": "owner/model",
        "recovery": (
            "Hugging Face model owner/model is not cached. "
            "While online, run `uv run hf download owner/model`, then retry; "
            "or use the remote AgentKB path."
        ),
    }


def test_cached_only_index_preflights_before_sources_or_index_mutation(monkeypatch):
    calls = []

    def missing(model):
        calls.append(("preflight", model))
        raise ModelCacheMissingError(model)

    def forbidden(*args, **kwargs):
        calls.append(("forbidden", args, kwargs))
        raise AssertionError("index pipeline ran before model-cache preflight")

    monkeypatch.setattr("agentkb.cli.require_cached_model", missing)
    monkeypatch.setattr("agentkb.cli._sync_pull_for_index", forbidden)
    monkeypatch.setattr("agentkb.cli._sync_push_for_index", forbidden)
    monkeypatch.setattr("agentkb.cli.wiki_store.reindex", forbidden)
    monkeypatch.setattr("agentkb.cli.chats_store.reindex", forbidden)
    monkeypatch.setattr("agentkb.cli.communications_store.reindex", forbidden)

    result = CliRunner().invoke(main, ["index", "--cached-only", "--rebuild"])

    assert result.exit_code == 1
    assert calls == [("preflight", DEFAULT_MODEL)]
    assert DEFAULT_MODEL in result.output
    assert "uv run hf download" in result.output
    assert "remote AgentKB path" in result.output


def test_cached_only_index_disables_network_and_forwards_model(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "agentkb.cli.require_cached_model",
        lambda model: calls.append(("preflight", model)) or Path("/cached"),
    )
    monkeypatch.setattr("agentkb.cli._sync_pull_for_index", lambda: calls.append(("pull",)))
    monkeypatch.setattr("agentkb.cli._sync_push_for_index", lambda: calls.append(("push",)))
    monkeypatch.setattr(
        "agentkb.cli.wiki_store.reindex",
        lambda **kwargs: calls.append(("wiki", kwargs)) or {},
    )
    monkeypatch.setattr(
        "agentkb.cli.chats_store.reindex",
        lambda **kwargs: calls.append(("chats", kwargs)) or {},
    )
    monkeypatch.setattr(
        "agentkb.cli.communications_store.reindex",
        lambda **kwargs: calls.append(("communications", kwargs)) or {},
    )

    result = CliRunner().invoke(
        main,
        ["index", "--cached-only", "--model", "owner/model", "--rebuild"],
    )

    assert result.exit_code == 0
    assert calls == [
        ("preflight", "owner/model"),
        ("wiki", {"model": "owner/model", "rebuild": True}),
        ("chats", {"model": "owner/model", "rebuild": True}),
        (
            "communications",
            {"model": "owner/model", "fetch": False, "rebuild": True},
        ),
    ]
