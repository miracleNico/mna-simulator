"""Access to the mandatory C++ extension module ``mna_simulation._core``.

The numerical core is being moved into a compiled extension (``cpp/`` in the
repository, built by scikit-build-core + nanobind). It is not optional: a
working installation always contains it. This module only adds a clear error
message for the case where the extension is missing or fails to load, e.g. when
the package directory is used straight from a source checkout without building.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

_BUILD_HELP = (
    "mna_simulation._core (the C++ numerical core) could not be imported. "
    "The extension is mandatory; build and install it from the repository root with\n"
    "    python -m pip install -e \".[dev]\"\n"
    "which compiles cpp/ with CMake >= 3.26 and a C++17 compiler (GCC or Clang on Linux, "
    "Visual Studio 2022 Build Tools on Windows). Run `python -m pip install -e .` again "
    "after changing C++ sources. Original error: {error}"
)


def require_core() -> ModuleType:
    """Return the ``mna_simulation._core`` module or raise an explanatory ImportError."""

    try:
        from .. import _core
    except ImportError as exc:
        raise ImportError(_BUILD_HELP.format(error=exc), name="mna_simulation._core") from exc
    return _core


def core_version() -> str:
    """Version string of the native core, e.g. ``"0.70.0-dev"``."""

    return str(require_core().version())


def build_info() -> dict[str, Any]:
    """How the native core was built: compiler, library versions, build flags."""

    return dict(require_core().build_info())
