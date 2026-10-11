"""Shared pytest configuration.

* Puts ``tests/reference`` on ``sys.path`` so the frozen oracle package
  ``mna_reference`` (a renamed copy of the pure-Python core, see
  ``tests/reference/README.md``) is importable, and ``tests/`` itself so the
  corpus (``corpus.py``) and the differential helpers import the same way
  under every pytest import mode.
* Exposes ``reference_pkg`` / ``current_pkg`` fixtures.
* Adds ``--run-perf``; tests marked ``perf`` are skipped unless it is given.
* Registers the ``perf``, ``slow`` and ``differential`` markers (they may
  also be registered in ``pyproject.toml``; duplicates are harmless).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = TESTS_DIR / "reference"

for _path in (TESTS_DIR, REFERENCE_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

_MARKERS = (
    "perf: performance/benchmark test; skipped unless --run-perf is given",
    "slow: long-running test (large circuits); runs by default",
    "differential: compares mna_simulation against the frozen mna_reference oracle",
)


def pytest_addoption(parser: pytest.Parser) -> None:
    try:
        parser.addoption(
            "--run-perf",
            action="store_true",
            default=False,
            help="run tests marked 'perf' (skipped by default)",
        )
    except ValueError:
        # Another conftest/plugin already registered the option.
        pass


def pytest_configure(config: pytest.Config) -> None:
    for marker in _MARKERS:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-perf", default=False):
        return
    skip_perf = pytest.mark.skip(reason="perf test: pass --run-perf to run")
    for item in items:
        # The marker, not ``"perf" in item.keywords``: keywords also hold parent node
        # names, so every test under tests/perf/ would match the directory name.
        if item.get_closest_marker("perf") is not None:
            item.add_marker(skip_perf)


@pytest.fixture(scope="session")
def reference_pkg():
    """The frozen oracle package ``mna_reference``."""

    import mna_reference

    location = Path(mna_reference.__file__).resolve().parent
    assert location == REFERENCE_DIR / "mna_reference", f"mna_reference imported from unexpected location {location}"
    return mna_reference


@pytest.fixture(scope="session")
def current_pkg():
    """The live package under test, ``mna_simulation``."""

    import mna_simulation

    return mna_simulation
