"""Tests for the benchmark CLI tool."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

try:
    sys.path.append(Path(__file__).parent.parent.as_posix())
    from tools import benchmark, profiler
    from typer.testing import CliRunner
except ImportError:
    pytest.skip("benchmark tool not available", allow_module_level=True)

from ome_writers._stream import AVAILABLE_BACKENDS

runner = CliRunner()

BACKENDS = [f"-b={backend}" for backend in AVAILABLE_BACKENDS]
COMMON = ["--iterations", "1", "--warmups", "0"]


def test_benchmark_cli_with_dims() -> None:
    """Smoke test to ensure benchmark runs with --dims."""
    result = runner.invoke(
        benchmark.app, ["-d", "t:3,y:256:64,x:256:64", *COMMON, *BACKENDS]
    )

    assert result.exit_code == 0
    assert "Benchmark Results" in result.stdout


def test_benchmark_cli_with_settings(tmp_path: Path) -> None:
    """Smoke test to ensure benchmark runs with --settings-file."""
    # Create a settings file
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(
        json.dumps(
            {
                "dimensions": [
                    {"name": "c", "count": 2},
                    {"name": "y", "count": 64, "chunk_size": 64},
                    {"name": "x", "count": 64, "chunk_size": 64},
                ],
                "dtype": "uint8",
            }
        )
    )

    result = runner.invoke(
        benchmark.app, ["-f", str(settings_file), *COMMON, *BACKENDS]
    )

    assert result.exit_code == 0
    assert "Benchmark Results" in result.stdout
    # Verify dtype from settings file was used
    assert "uint8" in result.stdout


@pytest.mark.parametrize("backend", list(AVAILABLE_BACKENDS))
def test_profile_cli_with_dims(backend: str) -> None:
    """Smoke test to ensure benchmark runs with --dims."""
    result = runner.invoke(profiler.app, ["-d", "t:3,y:256:64,x:256:64", "-b", backend])

    assert result.exit_code == 0
    assert "Top 20 functions by time" in result.stdout
