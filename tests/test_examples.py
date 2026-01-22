import contextlib
import os
import sys as sys_module
from collections.abc import Iterator
from pathlib import Path

import pytest

SKIP_EXAMPLES = {"pymmcore_plus_example.py"}

EXAMPLE_DIR = Path(__file__).parent.parent / "examples"
EXAMPLES = [p for p in sorted(EXAMPLE_DIR.glob("*.py")) if p.name not in SKIP_EXAMPLES]


@contextlib.contextmanager
def _chdir_argv(path: Path, argv: list[str]) -> Iterator[None]:
    """Context manager to change the current working directory."""
    original_cwd = Path.cwd()
    original_argv, sys_module.argv = sys_module.argv, argv
    try:
        os.chdir(path)
        yield
    finally:
        # Restore original cwd and sys.argv
        os.chdir(original_cwd)
        sys_module.argv = original_argv


@pytest.mark.parametrize("example_path", EXAMPLES, ids=lambda p: p.stem)
def test_example_runs(example_path: Path, tmp_path: Path, any_backend: str) -> None:
    """Test that each example script runs without error."""
    code = example_path.read_text()
    with _chdir_argv(tmp_path, [str(example_path), any_backend]):
        try:
            exec(code, {"sys": sys_module, "__name__": "__main__"})
        except NotImplementedError as e:
            if "does not support settings" in str(e):
                pytest.xfail(f"Example {example_path.name} uses unsupported settings.")
            raise

    # Validate that example created output files
    output_files = list(tmp_path.glob("*.ome.*")) + list(tmp_path.glob("*.ome.tiff"))
    assert output_files, f"Example {example_path.name} did not create output files"
