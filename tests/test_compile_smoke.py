"""Smoke test that all source files compile."""

import py_compile
from pathlib import Path


def test_all_source_files_compile():
    root = Path(__file__).resolve().parents[1]

    for source in (root / "src").rglob("*.py"):
        py_compile.compile(str(source), doraise=True)
