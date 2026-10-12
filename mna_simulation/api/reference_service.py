"""Pure-Python core mode: serve requests with the frozen pre-migration Python core.

The default service (:class:`~mna_simulation.api.service.SimulationService`) runs every
analysis in the native C++ core. :class:`ReferenceSimulationService` runs the parser,
builder and solvers of ``mna_reference`` instead: the pure-Python ``mna_simulation``
package as it was before the C++ migration, frozen under ``tests/reference/`` as the test
oracle (``tests/reference/README.md``). It keeps the HTTP/CLI contract of the default
service:

* request options are validated by the current rules first (``PythonBackend.build_options``),
  so a request is rejected with the same ``ValueError`` (HTTP 400) in both modes; a
  ``ValueError`` from the frozen core's own option parsing is tagged the same way;
* the frozen core's exceptions are re-raised as the matching ``mna_simulation.errors``
  classes (same ``code`` and ``message``);
* every run leases one thread from the thread budget (the pure-Python core is serial)
  and keeps BLAS on one thread, like the reference side of ``benchmarks/compare_backends.py``;
* result metadata carries ``core_backend: "python"`` and the frozen commit as ``core_version``.

The oracle is not part of the wheel, so this mode needs a source checkout (an editable
install): ``tests/reference`` is appended to ``sys.path`` when ``mna_reference`` is not
importable already. The oracle itself is never modified here.
"""

from __future__ import annotations

import importlib
import sys
import time
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from .contracts import AnalysisRequest, CircuitIR, SimulationResponse
from .service import INVALID_OPTION_NOTE, is_invalid_option_error
from .. import errors
from ..backends.python_backend import PythonBackend
from ..native.threads import ThreadBudget, default_budget, pin_blas_threads

REFERENCE_DIR = Path(__file__).resolve().parents[2] / "tests" / "reference"
REFERENCE_PACKAGE = "mna_reference"


class ReferenceCoreUnavailable(RuntimeError):
    """The frozen Python core (``tests/reference/mna_reference``) cannot be imported."""


def load_reference_package() -> ModuleType:
    """Import the frozen ``mna_reference`` package (from ``tests/reference`` if needed)."""

    try:
        return importlib.import_module(REFERENCE_PACKAGE)
    except ImportError:
        pass
    if not (REFERENCE_DIR / REFERENCE_PACKAGE / "__init__.py").is_file():
        raise ReferenceCoreUnavailable(
            "The Python core needs the frozen pre-migration package tests/reference/mna_reference, "
            "which only exists in a source checkout (it is not shipped in wheels). "
            "Run from the repository with an editable install, or use the default C++ core."
        )
    if str(REFERENCE_DIR) not in sys.path:
        sys.path.append(str(REFERENCE_DIR))
    return importlib.import_module(REFERENCE_PACKAGE)


@lru_cache(maxsize=1)
def reference_core_version() -> str:
    """``python-reference@<commit>``: the commit the frozen core was copied from."""

    commit = "unknown"
    # Next to the package that is actually imported: under tests/reference of a checkout, or
    # wherever a test run put it on sys.path when the package itself is an installed wheel.
    try:
        frozen_at = Path(load_reference_package().__file__ or "").resolve().parent / "FROZEN_AT.txt"
    except ReferenceCoreUnavailable:
        frozen_at = REFERENCE_DIR / REFERENCE_PACKAGE / "FROZEN_AT.txt"
    try:
        for line in frozen_at.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "commit" and value.strip():
                commit = value.strip()[:7]
                break
    except OSError:
        pass
    return f"python-reference@{commit}"


def _to_live_error(exc: Exception) -> errors.CircuitSimulatorError:
    """The ``mna_simulation.errors`` counterpart of a frozen-core simulator error."""

    cls = getattr(errors, type(exc).__name__, None)
    if not (isinstance(cls, type) and issubclass(cls, errors.CircuitSimulatorError)):
        cls = errors.CircuitSimulatorError
    return cls(getattr(exc, "message", str(exc)), code=getattr(exc, "code", None))


class ReferenceSimulationService:
    """``SimulationService`` contract on top of the frozen pure-Python core."""

    core_backend = "python"

    def __init__(self, budget: ThreadBudget | None = None) -> None:
        reference = load_reference_package()
        self._contracts = importlib.import_module(f"{REFERENCE_PACKAGE}.api.contracts")
        self._errors = importlib.import_module(f"{REFERENCE_PACKAGE}.errors")
        reference_backend = importlib.import_module(f"{REFERENCE_PACKAGE}.backends.python_backend").PythonBackend
        reference_service = importlib.import_module(f"{REFERENCE_PACKAGE}.api.service").SimulationService

        class _TaggedOptionsBackend(reference_backend):  # type: ignore[misc, valid-type]
            """Tags option-parsing ``ValueError``s of the frozen core like the live service does."""

            def build_options(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202
                try:
                    return super().build_options(*args, **kwargs)
                except ValueError as exc:
                    if not is_invalid_option_error(exc):
                        exc.add_note(INVALID_OPTION_NOTE)
                    raise

        self.reference = reference
        self.backend = _TaggedOptionsBackend()
        self._service = reference_service(backend=self.backend)
        self._validator = PythonBackend()
        self._budget = budget

    @property
    def budget(self) -> ThreadBudget:
        """Thread budget leased by every run (the process-wide default unless one was given)."""

        return self._budget if self._budget is not None else default_budget()

    def run_request(self, request: AnalysisRequest) -> SimulationResponse:
        """Run a request on the frozen Python core and return the live response model."""

        # Same option rules (and HTTP 400s) as the C++ mode; the circuit's own directive
        # parameters are left to the frozen parser.
        try:
            validated = self._validator.build_options(CircuitIR(components=[]), overrides=request.options, mode=request.mode)
        except ValueError as exc:
            if not is_invalid_option_error(exc):
                exc.add_note(INVALID_OPTION_NOTE)
            raise
        threads_requested: int | str = validated.num_threads if validated.num_threads is not None else "auto"

        reference_request = self._contracts.AnalysisRequest.model_validate(request.model_dump(mode="json"))
        budget = self.budget
        with budget.lease(1) as threads_used, pin_blas_threads():
            started = time.perf_counter()
            try:
                reference_response = self._service.run_request(reference_request)
            except self._errors.CircuitSimulatorError as exc:
                raise _to_live_error(exc) from exc
            elapsed_ms = (time.perf_counter() - started) * 1000.0

        metadata = dict(reference_response.metadata)
        metadata.update(
            {
                "threads_requested": threads_requested,
                "threads_used": threads_used,
                "thread_budget_total": budget.total,
                "core_backend": self.core_backend,
                "core_version": reference_core_version(),
                "elapsed_ms": round(elapsed_ms, 3),
            }
        )
        fields = {name: getattr(reference_response, name) for name in SimulationResponse.model_fields}
        fields["mode"] = str(getattr(reference_response.mode, "value", reference_response.mode))
        fields["metadata"] = metadata
        return SimulationResponse.model_validate(fields)

    def healthcheck(self) -> dict[str, object]:
        """Return backend information for the UI (``build_info`` is ``None``: no native core runs)."""

        return {
            "status": "ok",
            "capabilities": asdict(self.backend.capabilities),
            "models": [model.name for model in self.backend.registry.list_models()],
            "core_backend": self.core_backend,
            "core_version": reference_core_version(),
            "build_info": None,
            "thread_budget": self.budget.snapshot(),
        }
