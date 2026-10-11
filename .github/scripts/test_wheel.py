"""Run the test suite against an *installed* mna_simulation wheel.

Used by the Wheels workflow (cibuildwheel's test step and the Python 3.13 job).
The checkout contains the ``mna_simulation`` source tree without a compiled
``_core``; running pytest inside it would shadow the installed wheel (the test
conftest puts repository paths on ``sys.path``). So this script copies ``tests/``,
``benchmarks/`` and ``pyproject.toml`` (pytest configuration) to a temporary
directory outside the checkout, installs the test dependencies into the current
interpreter (the test venv of cibuildwheel on Windows ships without ``pip.exe``,
so ``ensurepip`` runs first) and runs pytest there.

Usage: python test_wheel.py <checkout> [extra pytest args]
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TEST_REQUIREMENTS = ("pytest>=8,<9", "httpx>=0.27,<1", "threadpoolctl>=3")


def run(*args: str, **kwargs: object) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True, **kwargs)  # type: ignore[arg-type]


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    checkout = Path(argv[0]).resolve()
    extra = argv[1:]
    try:
        import pip  # noqa: F401
    except ImportError:
        run(sys.executable, "-m", "ensurepip", "--upgrade")
    run(sys.executable, "-m", "pip", "install", "--quiet", *TEST_REQUIREMENTS)

    with tempfile.TemporaryDirectory(prefix="mna-wheel-test-") as tmp:
        root = Path(tmp) / "run"
        root.mkdir()
        for name in ("tests", "benchmarks"):
            shutil.copytree(checkout / name, root / name, ignore=shutil.ignore_patterns("__pycache__", ".venv"))
        shutil.copy2(checkout / "pyproject.toml", root / "pyproject.toml")
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        # Prove the installed package is the one under test (not the checkout).
        check = (
            "import mna_simulation, mna_simulation._core as c, pathlib, sys;"
            f"p = pathlib.Path(mna_simulation.__file__).resolve(); bad = pathlib.Path({str(checkout)!r});"
            "print('mna_simulation from', p, 'core', c.version());"
            "sys.exit(1 if bad in p.parents else 0)"
        )
        run(sys.executable, "-c", check, cwd=root, env=env)
        try:
            run(sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-x", "tests", *extra, cwd=root, env=env)
        except subprocess.CalledProcessError as exc:
            return exc.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
