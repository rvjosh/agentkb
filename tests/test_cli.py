"""Tests for the top-level AgentKB CLI."""

from importlib.metadata import version

from click.testing import CliRunner

from agentkb.cli import main


def test_version_reports_installed_distribution_version():
    result = CliRunner().invoke(main, ["--version"], prog_name="agentkb")

    assert result.exit_code == 0
    assert result.output == f"agentkb, version {version('agentkb')}\n"
