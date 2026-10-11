"""Reserved adapter contract for a future C++ backend."""

from __future__ import annotations

from ..api.contracts import BackendCapabilities


class CppBackend:
    """Placeholder adapter for a future pybind11/C++ backend."""

    def __init__(self) -> None:
        self.capabilities = BackendCapabilities(
            supports_behavioral_sources=False,
            supports_harmonic_balance=False,
            supports_sparse_linear_solver=True,
            supports_cpp_acceleration=True,
        )

    def run(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise NotImplementedError("C++ backend is not implemented yet.")
