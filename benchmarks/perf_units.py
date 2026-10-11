"""Comparison units for ``benchmarks/compare_backends.py``: frozen Python reference vs C++ core.

Each :class:`Unit` pairs one callable of the frozen oracle ``mna_reference``
(the pure-Python core as of the C++ migration freeze, ``tests/reference/``)
with the native ``mna_simulation._core`` callable that replaces it, on the
same inputs (``CPP_MIGRATION_PLAN.md`` section 5):

* ``setup(smoke=False)`` builds the shared inputs once (untimed). Workloads
  come from ``benchmarks/workload_netlists.py`` with the sizes of
  ``PERFORMANCE_PLAN.md`` section 4.1. ``smoke=True`` builds a tiny variant
  that the harness tests use to exercise every callable quickly; its timings
  mean nothing.
* ``reference(inputs)`` runs the frozen Python callable (timed; the harness
  pins BLAS to one thread).
* ``core_setup(inputs)`` prepares native inputs once (untimed, e.g. CSC
  arrays). Defaults to the identity. May raise :class:`UnitUnavailable`.
* ``core(core_inputs, num_threads)`` runs the native callable (timed at every
  requested thread count). Until its phase lands it raises
  ``UnitUnavailable("lands in P1: ...")``.
* ``parity(ref_out, core_out)`` returns the maximum relative error (a
  subspace distance for MOR bases), or ``None`` when nothing is comparable.
  ``tolerance`` is the bound for that number (``Unit.parity_ok``).

Adding a native implementation: replace the body of the matching
``_core_<unit id>`` function in the "native side" section (and give the unit a
``core_setup`` if the native call needs prepared inputs, for example a built
``Circuit``). The CLI, the JSON/Markdown reports, the perf gate
(``tests/perf/test_perf_gate.py``) and the harness tests pick it up with no
other change. Never modify the ``reference`` side: it measures the oracle.

Parity metric (``max_rel_error``): ``max|ref - core| / max|ref|`` over all
entries (infinity norms; ``0`` for empty arrays; ``inf`` on shape mismatch or
a NaN in only one of the two arrays).
"""

from __future__ import annotations

import functools
import re
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
REFERENCE_DIR = REPO_ROOT / "tests" / "reference"

# ``workload_netlists`` lives next to this file; the oracle ``mna_reference`` is
# importable only from ``tests/reference`` (it never ships in a wheel).
for _path in (BENCH_DIR, REFERENCE_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

CATEGORIES: tuple[str, ...] = (
    "device_eval",
    "assembly",
    "linear_solve",
    "krylov",
    "newton",
    "transient",
    "ac",
    "hb",
    "mor",
)

# Parity tolerances (CPP_MIGRATION_PLAN.md sections 4 and 6).
TOL_BUILD = 1e-15  # P1: dense G, C, b_dc equal to the reference to 1e-15 relative
TOL_KERNEL = 1e-13  # P1: f and J equal to the reference to 1e-13
TOL_LINEAR = 1e-9  # linear analyses and direct solves
TOL_KRYLOV = 1e-6  # iterative solutions converged to a 1e-9 relative residual
TOL_NONLINEAR = 1e-6  # Newton-based analyses
TOL_HB = 1e-6
TOL_MOR = 5e-2


class UnitUnavailable(RuntimeError):
    """The native side of a unit does not exist yet.

    The message names the phase that provides it, e.g. ``"lands in P1: ..."``.
    """

    @property
    def phase(self) -> str | None:
        match = re.search(r"\bP\d+\b", str(self))
        return match.group(0) if match else None


def _identity(inputs: Any) -> Any:
    return inputs


@dataclass(frozen=True)
class Unit:
    """One comparison unit: the same work done by the oracle and by the native core."""

    id: str
    category: str
    size: dict[str, int]
    setup: Callable[..., Any]
    reference: Callable[[Any], Any]
    core: Callable[[Any, int], Any]
    parity: Callable[[Any, Any], float | None]
    phase: str
    tolerance: float
    description: str
    core_setup: Callable[[Any], Any] = field(default=_identity)

    def parity_ok(self, error: float | None) -> bool | None:
        if error is None:
            return None
        return bool(np.isfinite(error) and error <= self.tolerance)

    def size_text(self) -> str:
        return " ".join(f"{key}={value}" for key, value in self.size.items())


# ---------------------------------------------------------------------------------------------
# Parity helpers
# ---------------------------------------------------------------------------------------------


def max_rel_error(ref: Any, core: Any) -> float:
    """``max|ref - core| / max|ref|`` (see the module docstring)."""

    a = np.asarray(_dense(ref))
    b = np.asarray(_dense(core))
    if a.shape != b.shape:
        if a.size != b.size:
            return float("inf")
        b = b.reshape(a.shape)
    if a.size == 0:
        return 0.0
    a_nan = np.isnan(a)
    if np.any(a_nan != np.isnan(b)):
        return float("inf")
    a = a[~a_nan]
    b = b[~a_nan]
    diff = np.where(a == b, 0.0, np.abs(a - b))  # equal infinities count as exact
    if not np.all(np.isfinite(diff)):
        return float("inf")
    scale = float(np.max(np.abs(a[np.isfinite(a)]), initial=0.0))
    return float(np.max(diff, initial=0.0) / max(scale, np.finfo(float).tiny))


def subspace_distance(ref_basis: Any, core_basis: Any) -> float:
    """Sine of the largest principal angle between the column spaces (0 = same space)."""

    a = np.asarray(ref_basis, dtype=float)
    b = np.asarray(core_basis, dtype=float)
    if a.ndim != 2 or a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    # Largest singular value of (I - Qa Qa^T) Qb = sin of the largest principal angle.
    # (sqrt(1 - cos^2) would turn a one-ulp error in cos into sqrt(eps) ~ 1.5e-8.)
    residual = qb - qa @ (qa.T @ qb)
    return float(min(np.linalg.norm(residual, 2), 1.0))


def _dense(value: Any) -> Any:
    return value.toarray() if hasattr(value, "toarray") else value


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _first(value: Any) -> Any:
    """``x`` from ``(x, info)`` tuples, ``value`` otherwise."""

    if isinstance(value, tuple) and value:
        return value[0]
    return value


def parity_array(ref_out: Any, core_out: Any) -> float:
    return max_rel_error(_first(ref_out), _first(core_out))


def parity_f_and_j(ref_out: Any, core_out: Any) -> float:
    """Both outputs are ``(f, J)``; the larger of the two relative errors."""

    f_ref, j_ref = ref_out
    f_core, j_core = core_out
    return max(max_rel_error(np.reshape(f_ref, -1), np.reshape(f_core, -1)), max_rel_error(j_ref, j_core))


def parity_matrices(ref_out: Any, core_out: Any) -> float:
    """Circuit build: dense ``G``, ``C``, ``b_dc``, ``b_ac`` (attributes or mapping keys)."""

    worst = 0.0
    for name in ("G", "C", "b_dc", "b_ac"):
        ref_value = _field(ref_out, name)
        core_value = _field(core_out, name)
        if ref_value is None or core_value is None:
            if ref_value is None and core_value is None:
                continue
            return float("inf")
        worst = max(worst, max_rel_error(ref_value, core_value))
    return worst


def analysis_values(result: Any) -> Any:
    """The numeric payload of a ``SimulationResult`` (or of an array passed straight through)."""

    if isinstance(result, np.ndarray):
        return result
    harmonic_state = _field(result, "harmonic_state")
    if harmonic_state is not None:
        return np.asarray(harmonic_state)
    waveform = _field(result, "waveform")
    if waveform is not None:
        return np.asarray(_field(waveform, "values"), dtype=float)
    spectrum = _field(result, "spectrum")
    if spectrum is not None:
        return np.asarray(_field(spectrum, "magnitudes"), dtype=float)
    dc_solution = _field(result, "dc_solution")
    if dc_solution is not None:
        return np.asarray(dc_solution)
    raise ValueError(f"no numeric payload in {type(result).__name__}")


def parity_analysis(ref_out: Any, core_out: Any) -> float:
    return max_rel_error(analysis_values(ref_out), analysis_values(core_out))


def parity_basis(ref_out: Any, core_out: Any) -> float:
    return subspace_distance(ref_out, core_out)


def parity_tpwl(ref_out: Any, core_out: Any) -> float:
    """TPWL training: training trajectory (relative error) and POD basis (subspace distance)."""

    snapshots = max_rel_error(_field(ref_out, "snapshots"), _field(core_out, "snapshots"))
    basis = subspace_distance(_field(ref_out, "basis"), _field(core_out, "basis"))
    return max(snapshots, basis)


# ---------------------------------------------------------------------------------------------
# Workloads (shared, cached per process)
# ---------------------------------------------------------------------------------------------


def _reference_modules() -> SimpleNamespace:
    """Import the frozen oracle lazily so ``--list`` stays fast."""

    from mna_reference import core_advanced, core_basic, mna_builder, netlist, solvers
    from mna_reference.api import contracts
    from mna_reference.mor import linear as mor_linear
    from mna_reference.mor import metadata as mor_metadata
    from mna_reference.mor import selectors as mor_selectors
    from mna_reference.mor import tpwl as mor_tpwl

    return SimpleNamespace(
        contracts=contracts,
        core_advanced=core_advanced,
        core_basic=core_basic,
        mna_builder=mna_builder,
        netlist=netlist,
        solvers=solvers,
        mor_linear=mor_linear,
        mor_metadata=mor_metadata,
        mor_selectors=mor_selectors,
        mor_tpwl=mor_tpwl,
    )


def _hb_mixer_variant(second_tone: str, harmonics: int) -> str:
    """The close-tone diode mixer with another second tone.

    ``run_advanced_analysis`` raises the harmonic count to the number the tones
    need (``1.005 GHz`` on a 5 MHz grid needs 201), so a real ``H = 64`` run
    needs coarser tone spacing: ``1.02 GHz`` puts the tones on a 20 MHz grid
    (harmonics 50 and 51). The smoke variant uses 1.25 GHz (250 MHz grid, H = 8).
    """

    import workload_netlists as wl

    text = wl.HB_CLOSE_TONE_MIXER.replace("SIN 40u 1.005G", f"SIN 40u {second_tone}")
    return text.replace(".hb 205", f".hb {harmonics}")


def _netlist(name: str, smoke: bool) -> str:
    import workload_netlists as wl

    if name == "sram":
        if smoke:
            return wl.sram_netlist(rows=2, cols=2, step="1n", selected_row=1, selected_col=1)
        return wl.sram_netlist(rows=10, cols=10, step="0.25n")
    if name == "rlc":
        return wl.rlc_mesh_netlist(4, 4) if smoke else wl.rlc_mesh_netlist(23, 23)
    if name == "bjt":
        return wl.BJT_THREE_STAGE
    if name == "mixer_h205":
        return _hb_mixer_variant("1.25G", 8) if smoke else wl.HB_CLOSE_TONE_MIXER
    if name == "mixer_h64":
        return _hb_mixer_variant("1.25G", 8) if smoke else _hb_mixer_variant("1.02G", 64)
    raise KeyError(name)


@functools.lru_cache(maxsize=None)
def _workload(name: str, smoke: bool) -> SimpleNamespace:
    """Parsed circuit and MNA problem built by the oracle; shared by every unit using it."""

    ref = _reference_modules()
    text = _netlist(name, smoke)
    circuit = ref.netlist.parse_netlist_text(text)
    problem = ref.mna_builder.build_mna_problem(circuit)
    solver_f_str = problem.solver_f_str if problem.solver_f_str is not None else problem.f_str
    return SimpleNamespace(
        name=name,
        netlist=text,
        circuit=circuit,
        problem=problem,
        n=int(problem.G.shape[0]),
        solver_f_str=solver_f_str,
        labels=list(problem.metadata["labels"]),
    )


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _options(mode: str, **kwargs: Any) -> Any:
    contracts = _reference_modules().contracts
    return contracts.AnalysisOptions(mode=contracts.AnalysisMode(mode), **kwargs)


# Time steps of the operators used by the solve units: the transient steps of the
# RLC (40 x 1 ns) and SRAM (6 ns / 0.25 ns) workloads.
H_RLC = 1e-9
H_SRAM = 0.25e-9
F_RLC = 1e8  # inside the RLC AC sweep (1 MHz .. 1 GHz)
F_SRAM = 1e9


# ---------------------------------------------------------------------------------------------
# Setups (untimed). Every setup takes ``smoke`` and returns a dict.
# ---------------------------------------------------------------------------------------------


def _setup_device_eval_mos(smoke: bool = False) -> dict[str, Any]:
    w = _workload("sram", smoke)
    x = _rng(1).uniform(0.0, 1.2, size=w.n)
    return {"workload": w, "devices": w.problem.level1_mos_devices, "n": w.n, "x": x}


def _setup_compiled_nl(name: str, smoke: bool, seed: int, high: float) -> dict[str, Any]:
    ref = _reference_modules()
    w = _workload(name, smoke)
    f_func, j_func = ref.solvers.compile_nl_functions(
        w.solver_f_str,
        w.n,
        level1_mos_devices=w.problem.level1_mos_devices,
        level1_bjt_devices=w.problem.level1_bjt_devices,
    )
    x = _rng(seed).uniform(0.0, high, size=w.n)
    return {"workload": w, "n": w.n, "x": x, "f_func": f_func, "j_func": j_func}


def _setup_device_eval_bjt(smoke: bool = False) -> dict[str, Any]:
    return _setup_compiled_nl("bjt", smoke, seed=2, high=1.0)


def _setup_device_eval_diode(smoke: bool = False) -> dict[str, Any]:
    return _setup_compiled_nl("mixer_h205", smoke, seed=3, high=0.7)


def _setup_circuit_build(name: str, smoke: bool) -> dict[str, Any]:
    w = _workload(name, smoke)
    return {"workload": w, "netlist": w.netlist, "circuit": w.circuit}


def _setup_circuit_build_sram(smoke: bool = False) -> dict[str, Any]:
    return _setup_circuit_build("sram", smoke)


def _setup_circuit_build_rlc(smoke: bool = False) -> dict[str, Any]:
    return _setup_circuit_build("rlc", smoke)


def _setup_assembly_sram(smoke: bool = False) -> dict[str, Any]:
    inputs = _setup_compiled_nl("sram", smoke, seed=4, high=1.2)
    inputs["G"] = inputs["workload"].problem.G
    return inputs


def _setup_direct_real(name: str, h: float, smoke: bool, seed: int) -> dict[str, Any]:
    w = _workload(name, smoke)
    A = np.ascontiguousarray(w.problem.G + w.problem.C / h)
    b = _rng(seed).standard_normal(w.n)
    return {"workload": w, "A": A, "b": b, "h": h}


def _setup_direct_solve_rlc(smoke: bool = False) -> dict[str, Any]:
    return _setup_direct_real("rlc", H_RLC, smoke, seed=10)


def _setup_direct_solve_sram(smoke: bool = False) -> dict[str, Any]:
    return _setup_direct_real("sram", H_SRAM, smoke, seed=11)


def _setup_direct_complex(name: str, frequency: float, smoke: bool, seed: int) -> dict[str, Any]:
    w = _workload(name, smoke)
    omega = 2.0 * np.pi * frequency
    A = np.ascontiguousarray(w.problem.G + 1j * omega * w.problem.C)
    rng = _rng(seed)
    b = rng.standard_normal(w.n) + 1j * rng.standard_normal(w.n)
    return {"workload": w, "A": A, "b": b, "omega": omega}


def _setup_direct_solve_complex_rlc(smoke: bool = False) -> dict[str, Any]:
    return _setup_direct_complex("rlc", F_RLC, smoke, seed=12)


def _setup_direct_solve_complex_sram(smoke: bool = False) -> dict[str, Any]:
    return _setup_direct_complex("sram", F_SRAM, smoke, seed=13)


def _setup_krylov(method: str) -> Callable[..., dict[str, Any]]:
    def setup(smoke: bool = False) -> dict[str, Any]:
        w = _workload("rlc", smoke)
        A = (w.problem.G_sparse + w.problem.C_sparse / H_RLC).tocsr()
        b = _rng(20).standard_normal(w.n)
        return {"workload": w, "A": A, "b": b, "method": method, "tol": 1e-9, "max_iter": 2000, "restart": 80}

    setup.__name__ = f"_setup_krylov_{method}"
    return setup


def _setup_basic(name: str, mode: str, full: dict[str, Any], smoke_overrides: dict[str, Any]) -> Callable[..., dict[str, Any]]:
    """Setup for ``run_basic_analysis`` / ``run_advanced_analysis`` on a workload."""

    def setup(smoke: bool = False) -> dict[str, Any]:
        w = _workload(name, smoke)
        params = {**full, **(smoke_overrides if smoke else {})}
        return {"workload": w, "problem": w.problem, "netlist": w.netlist, "options": _options(mode, **params)}

    setup.__name__ = f"_setup_{mode}_{name}"
    return setup


def _setup_mor_basis_rlc(smoke: bool = False) -> dict[str, Any]:
    """Arguments of ``build_linear_krylov_basis`` as ``solve_transient_mor`` builds them."""

    ref = _reference_modules()
    w = _workload("rlc", smoke)
    corner = "V(n_3_3)" if smoke else "V(n_22_22)"
    outputs = ["V(n_0_0)", corner]
    selector, output_labels = ref.mor_selectors.build_output_selector(w.labels, outputs)
    inputs = ref.mor_linear._source_directions(w.problem)
    order, _mode = ref.mor_metadata.resolve_mor_order(
        w.n, ref.mor_linear.count_input_directions(w.problem), len(output_labels), "auto"
    )
    shifts = [0.0, 1.0 / H_RLC]
    return {
        "workload": w,
        "G": w.problem.G,
        "C": w.problem.C,
        "inputs": inputs,
        "selector": selector,
        "shifts": shifts,
        "order": order,
    }


def _setup_tpwl_sram(smoke: bool = False) -> dict[str, Any]:
    """Arguments of ``_train_tpwl`` as ``solve_transient_tpwl`` builds them (one output)."""

    ref = _reference_modules()
    w = _workload("sram", smoke)
    order, _mode = ref.mor_metadata.resolve_mor_order(w.n, ref.mor_linear.count_input_directions(w.problem), 1, "auto")
    return {
        "workload": w,
        "problem": w.problem,
        "netlist": w.netlist,
        "t_stop": 6e-9,
        "t_step": 1e-9 if smoke else H_SRAM,
        "order": order,
        "max_iter": 200,
        "v_tol": 1e-9,
        "f_tol": 1e-9,
        "output_count": 1,
    }


# ---------------------------------------------------------------------------------------------
# Reference side (frozen ``mna_reference`` callables; timed)
# ---------------------------------------------------------------------------------------------


def _ref_device_eval_mos(inputs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    solvers = _reference_modules().solvers
    return solvers._evaluate_level1_mos_devices(inputs["devices"], inputs["n"], inputs["x"])


def _ref_compiled_f_and_j(inputs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    x = inputs["x"]
    return inputs["f_func"](*x), inputs["j_func"](*x)


def _ref_circuit_build(inputs: dict[str, Any]) -> Any:
    return _reference_modules().mna_builder.build_mna_problem(inputs["circuit"])


def _ref_assembly(inputs: dict[str, Any]) -> np.ndarray:
    return inputs["G"] + inputs["j_func"](*inputs["x"])


def _ref_direct_solve(inputs: dict[str, Any]) -> np.ndarray:
    return np.linalg.solve(inputs["A"], inputs["b"])


def _ref_krylov(inputs: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    solvers = _reference_modules().solvers
    return solvers.krylov_solve_linear(
        inputs["A"],
        inputs["b"],
        tol=inputs["tol"],
        max_iter=inputs["max_iter"],
        restart=inputs["restart"],
        method=inputs["method"],
    )


def _ref_basic(inputs: dict[str, Any]) -> Any:
    return _reference_modules().core_basic.run_basic_analysis(inputs["problem"], inputs["options"])


def _ref_advanced(inputs: dict[str, Any]) -> Any:
    return _reference_modules().core_advanced.run_advanced_analysis(inputs["problem"], inputs["options"])


def _ref_mor_basis(inputs: dict[str, Any]) -> np.ndarray:
    mor_linear = _reference_modules().mor_linear
    return mor_linear.build_linear_krylov_basis(
        inputs["G"], inputs["C"], inputs["inputs"], inputs["selector"], inputs["shifts"], inputs["order"]
    )


def _ref_tpwl(inputs: dict[str, Any]) -> dict[str, Any]:
    mor_tpwl = _reference_modules().mor_tpwl
    return mor_tpwl._train_tpwl(
        inputs["problem"],
        t_stop=inputs["t_stop"],
        t_step=inputs["t_step"],
        order=inputs["order"],
        max_iter=inputs["max_iter"],
        v_tol=inputs["v_tol"],
        f_tol=inputs["f_tol"],
        output_count=inputs["output_count"],
    )


# ---------------------------------------------------------------------------------------------
# Native side (``mna_simulation._core``; timed). One function per unit: replace the body
# when the native callable lands. ``num_threads`` is the requested thread count.
# ---------------------------------------------------------------------------------------------


def _native() -> Any:
    from mna_simulation.native import require_core

    return require_core()


# P0: KLU direct solve (``_core.klu_solve``: analyze + factor + solve, serial).


def _core_setup_csc(inputs: dict[str, Any]) -> dict[str, Any]:
    """CSC int32 arrays of the reference operator for ``_core.klu_solve``."""

    import scipy.sparse as sps

    _native()  # fail here, untimed, when the extension is missing
    csc = sps.csc_matrix(inputs["A"])
    csc.sum_duplicates()
    csc.sort_indices()
    return {
        "Ap": np.ascontiguousarray(csc.indptr, dtype=np.int32),
        "Ai": np.ascontiguousarray(csc.indices, dtype=np.int32),
        "Ax": np.ascontiguousarray(csc.data, dtype=np.float64),
        "b": np.ascontiguousarray(inputs["b"], dtype=np.float64),
    }


def _core_direct_solve_rlc_1035(core_inputs: dict[str, Any], num_threads: int) -> np.ndarray:
    return _native().klu_solve(core_inputs["Ap"], core_inputs["Ai"], core_inputs["Ax"], core_inputs["b"])


def _core_direct_solve_sram_248(core_inputs: dict[str, Any], num_threads: int) -> np.ndarray:
    return _native().klu_solve(core_inputs["Ap"], core_inputs["Ai"], core_inputs["Ax"], core_inputs["b"])


# P1: expression VM, circuit builder, device kernels.


class KernelJacobian:
    """f and device-Jacobian values written by ``_core.evaluate_devices`` into caller buffers.

    The timed unit never materializes a dense matrix; ``dense()`` (used only by the untimed
    parity check) scatters the CSC values into an (n, n) array.
    """

    def __init__(self, f: np.ndarray, values: np.ndarray, col_ptr: np.ndarray, row_idx: np.ndarray, n: int) -> None:
        self.f, self.values, self.col_ptr, self.row_idx, self.n = f, values, col_ptr, row_idx, n

    def dense(self) -> np.ndarray:
        import scipy.sparse as sps

        return sps.csc_matrix((self.values, self.row_idx, self.col_ptr), shape=(self.n, self.n)).toarray()

    def as_f_and_j(self) -> tuple[np.ndarray, np.ndarray]:
        return self.f.reshape(self.n, 1), self.dense()


def parity_kernel_f_and_j(ref_out: Any, core_out: Any) -> float:
    return parity_f_and_j(ref_out, core_out.as_f_and_j() if isinstance(core_out, KernelJacobian) else core_out)


def parity_kernel_matrix(ref_out: Any, core_out: Any) -> float:
    return parity_array(ref_out, core_out.dense() if isinstance(core_out, KernelJacobian) else core_out)


def _core_setup_kernel(inputs: dict[str, Any]) -> dict[str, Any]:
    """Compile the circuit natively (untimed) and preallocate the kernel's output buffers."""

    from mna_simulation.mna_builder import build_mna_problem

    _native()
    w = inputs["workload"]
    problem = build_mna_problem(w.circuit)
    circuit = problem.native
    col_ptr, row_idx = circuit.pattern()
    n = problem.G.shape[0]
    out = {
        "core": _native(),  # resolved once; a per-call import lookup would dominate a 10 us kernel
        "circuit": circuit,
        "x": np.ascontiguousarray(inputs["x"], dtype=float),
        "f_out": np.zeros(n),
        "j_out": np.zeros(circuit.nnz),
        "col_ptr": np.asarray(col_ptr),
        "row_idx": np.asarray(row_idx),
        "n": n,
    }
    if "G" in inputs:
        # G values in pattern order; the assembly unit adds the device Jacobian to them.
        out["g_vals"] = np.ascontiguousarray(circuit.values("G"), dtype=float)
        out["a_out"] = np.zeros(circuit.nnz)
    return out


def _core_kernel_f_and_j(core_inputs: dict[str, Any], num_threads: int) -> KernelJacobian:
    c = core_inputs
    c["core"].evaluate_devices(c["circuit"], c["x"], 0.0, num_threads, c["f_out"], c["j_out"])
    return KernelJacobian(c["f_out"], c["j_out"], c["col_ptr"], c["row_idx"], c["n"])


def _core_device_eval_sram_mos_622(core_inputs: dict[str, Any], num_threads: int) -> Any:
    return _core_kernel_f_and_j(core_inputs, num_threads)


def _core_device_eval_bjt_3stage(core_inputs: dict[str, Any], num_threads: int) -> Any:
    return _core_kernel_f_and_j(core_inputs, num_threads)


def _core_device_eval_diode_mixer(core_inputs: dict[str, Any], num_threads: int) -> Any:
    return _core_kernel_f_and_j(core_inputs, num_threads)


def _core_circuit_build(core_inputs: dict[str, Any], num_threads: int) -> Any:
    from mna_simulation.mna_builder import build_mna_problem

    return build_mna_problem(core_inputs["circuit"])


def _core_circuit_build_sram_248(core_inputs: dict[str, Any], num_threads: int) -> Any:
    return _core_circuit_build(core_inputs, num_threads)


def _core_circuit_build_rlc_1035(core_inputs: dict[str, Any], num_threads: int) -> Any:
    return _core_circuit_build(core_inputs, num_threads)


def _core_assembly_jacobian_sram_248(core_inputs: dict[str, Any], num_threads: int) -> KernelJacobian:
    """Newton matrix values G + J(x) in the CSC pattern: kernel evaluation plus one vector add."""

    c = core_inputs
    c["core"].evaluate_devices(c["circuit"], c["x"], 0.0, num_threads, c["f_out"], c["j_out"])
    np.add(c["g_vals"], c["j_out"], out=c["a_out"])
    return KernelJacobian(c["f_out"], c["a_out"], c["col_ptr"], c["row_idx"], c["n"])


# P2: complex KLU, Krylov solvers, Newton .op.


def _core_setup_csc_any(inputs: dict[str, Any]) -> dict[str, Any]:
    """CSC int32 arrays (real or complex) of the reference operator for ``_core.direct_solve``."""

    from mna_simulation.native import linalg

    _native()
    op = linalg.to_csc_arrays(inputs["A"])
    return {"op": op, "b": np.ascontiguousarray(inputs["b"])}


def _core_direct_solve_csc(core_inputs: dict[str, Any], num_threads: int) -> np.ndarray:
    op = core_inputs["op"]
    return _native().direct_solve(op.col_ptr, op.row_idx, op.values, core_inputs["b"])


def _core_direct_solve_complex_rlc_1035(core_inputs: dict[str, Any], num_threads: int) -> np.ndarray:
    return _core_direct_solve_csc(core_inputs, num_threads)


def _core_direct_solve_complex_sram_248(core_inputs: dict[str, Any], num_threads: int) -> np.ndarray:
    return _core_direct_solve_csc(core_inputs, num_threads)


def _core_krylov(core_inputs: dict[str, Any], num_threads: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Native ``krylov_solve_linear`` with the reference's call shape (same conversions timed)."""

    from mna_simulation.native import linalg

    return linalg.krylov_solve_linear(
        core_inputs["A"],
        core_inputs["b"],
        tol=core_inputs["tol"],
        max_iter=core_inputs["max_iter"],
        restart=core_inputs["restart"],
        method=core_inputs["method"],
        num_threads=num_threads,
    )


def _core_krylov_gmres_rlc_1035(core_inputs: dict[str, Any], num_threads: int) -> Any:
    return _core_krylov(core_inputs, num_threads)


def _core_krylov_minres_rlc_1035(core_inputs: dict[str, Any], num_threads: int) -> Any:
    return _core_krylov(core_inputs, num_threads)


def _core_krylov_cg_rlc_1035(core_inputs: dict[str, Any], num_threads: int) -> Any:
    return _core_krylov(core_inputs, num_threads)


def _core_setup_basic(inputs: dict[str, Any]) -> dict[str, Any]:
    """The live package's problem and options for ``run_basic_analysis`` (untimed).

    The problem is built by ``mna_simulation`` from the same netlist (its compiled native
    circuit is part of the problem, like the reference's dense matrices are part of its
    problem); the options copy the reference options field by field.
    """

    import dataclasses

    from mna_simulation.api import contracts
    from mna_simulation.mna_builder import build_mna_problem
    from mna_simulation.netlist import parse_netlist_text

    _native()
    ref_options = inputs["options"]
    fields = {field.name for field in dataclasses.fields(contracts.AnalysisOptions)}
    values = {name: getattr(ref_options, name) for name in fields if hasattr(ref_options, name) and name != "mode"}
    options = contracts.AnalysisOptions(mode=contracts.AnalysisMode(ref_options.mode.value), **values)
    return {"problem": build_mna_problem(parse_netlist_text(inputs["netlist"])), "options": options}


def _core_basic(core_inputs: dict[str, Any], num_threads: int) -> Any:
    """``mna_simulation.core_basic.run_basic_analysis`` (native .op / .tran) at ``num_threads``."""

    import dataclasses

    from mna_simulation.core_basic import run_basic_analysis

    options = dataclasses.replace(core_inputs["options"], num_threads=num_threads)
    return run_basic_analysis(core_inputs["problem"], options)


def _core_op_sram_248(core_inputs: Any, num_threads: int) -> Any:
    return _core_basic(core_inputs, num_threads)


# P3: transient and AC drivers.


def _core_tran_sram_248(core_inputs: Any, num_threads: int) -> Any:
    return _core_basic(core_inputs, num_threads)


def _core_tran_bjt_3stage_2001(core_inputs: Any, num_threads: int) -> Any:
    return _core_basic(core_inputs, num_threads)


def _core_tran_rlc_1035_dense(core_inputs: Any, num_threads: int) -> Any:
    return _core_basic(core_inputs, num_threads)


def _core_tran_rlc_1035_krylov(core_inputs: Any, num_threads: int) -> Any:
    return _core_basic(core_inputs, num_threads)


def _core_setup_ac_live(inputs: dict[str, Any]) -> dict[str, Any]:
    """Live ``MnaProblem`` and ``AnalysisOptions`` for the AC units (untimed, like the reference's prebuilt problem)."""

    import dataclasses

    from mna_simulation.api import contracts
    from mna_simulation.mna_builder import build_mna_problem
    from mna_simulation.netlist import parse_netlist_text

    _native()
    ref_options = inputs["options"]
    live_fields = {f.name for f in dataclasses.fields(contracts.AnalysisOptions)}
    values = {
        f.name: getattr(ref_options, f.name)
        for f in dataclasses.fields(ref_options)
        if f.name in live_fields and f.name != "mode"
    }
    options = contracts.AnalysisOptions(mode=contracts.AnalysisMode(ref_options.mode.value), **values)
    return {"problem": build_mna_problem(parse_netlist_text(inputs["netlist"])), "options": options}


def _core_ac_live(core_inputs: dict[str, Any], num_threads: int) -> Any:
    """The live ``run_basic_analysis`` with ``num_threads`` passed through ``AnalysisOptions``."""

    import dataclasses

    from mna_simulation.core_basic import run_basic_analysis

    options = dataclasses.replace(core_inputs["options"], num_threads=int(num_threads))
    return run_basic_analysis(core_inputs["problem"], options)


def _core_ac_rlc_1035(core_inputs: Any, num_threads: int) -> Any:
    return _core_ac_live(core_inputs, num_threads)


def _core_ac_bjt_3stage(core_inputs: Any, num_threads: int) -> Any:
    return _core_ac_live(core_inputs, num_threads)


# P4: harmonic balance.


def _core_hb_mixer_h205(core_inputs: Any, num_threads: int) -> Any:
    raise UnitUnavailable("lands in P4: native harmonic balance")


def _core_hb_mixer_h64(core_inputs: Any, num_threads: int) -> Any:
    raise UnitUnavailable("lands in P4: native harmonic balance")


# P5: model-order reduction.


def _core_mor_basis_rlc_1035(core_inputs: Any, num_threads: int) -> Any:
    raise UnitUnavailable("lands in P5: native rational Krylov basis (mor_linear_basis)")


def _core_tpwl_train_sram_248(core_inputs: Any, num_threads: int) -> Any:
    raise UnitUnavailable("lands in P5: native TPWL training (tpwl_train)")


# ---------------------------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------------------------

UNITS: dict[str, Unit] = {}


def _register(**kwargs: Any) -> Unit:
    unit = Unit(**kwargs)
    if unit.category not in CATEGORIES:
        raise ValueError(f"unknown category {unit.category!r} for unit {unit.id!r}")
    if unit.id in UNITS:
        raise ValueError(f"duplicate unit id {unit.id!r}")
    UNITS[unit.id] = unit
    return unit


_SRAM = {"n": 248, "devices": 622}
_RLC = {"n": 1035}

# device_eval ---------------------------------------------------------------------------------
_register(
    id="device_eval_sram_mos_622",
    category="device_eval",
    size=dict(_SRAM),
    setup=_setup_device_eval_mos,
    reference=_ref_device_eval_mos,
    core=_core_device_eval_sram_mos_622,
    core_setup=_core_setup_kernel,
    parity=parity_kernel_f_and_j,
    phase="P1",
    tolerance=TOL_KERNEL,
    description="f and dense J of the 622 Level-1 MOS devices of the 10x10 SRAM (solvers._evaluate_level1_mos_devices)",
)
_register(
    id="device_eval_bjt_3stage",
    category="device_eval",
    size={"n": 14, "devices": 4},
    setup=_setup_device_eval_bjt,
    reference=_ref_compiled_f_and_j,
    core=_core_device_eval_bjt_3stage,
    core_setup=_core_setup_kernel,
    parity=parity_kernel_f_and_j,
    phase="P1",
    tolerance=TOL_KERNEL,
    description="f and J of the three-stage BJT amplifier (compile_nl_functions)",
)
_register(
    id="device_eval_diode_mixer",
    category="device_eval",
    size={"n": 3, "devices": 1},
    setup=_setup_device_eval_diode,
    reference=_ref_compiled_f_and_j,
    core=_core_device_eval_diode_mixer,
    core_setup=_core_setup_kernel,
    parity=parity_kernel_f_and_j,
    phase="P1",
    tolerance=TOL_KERNEL,
    description="f and J of the close-tone diode mixer's symbolic diode net (compile_nl_functions)",
)

# assembly ------------------------------------------------------------------------------------
_register(
    id="circuit_build_sram_248",
    category="assembly",
    size=dict(_SRAM),
    setup=_setup_circuit_build_sram,
    reference=_ref_circuit_build,
    core=_core_circuit_build_sram_248,
    parity=parity_matrices,
    phase="P1",
    tolerance=TOL_BUILD,
    description="MNA build of the parsed 10x10 SRAM (mna_builder.build_mna_problem)",
)
_register(
    id="circuit_build_rlc_1035",
    category="assembly",
    size=dict(_RLC),
    setup=_setup_circuit_build_rlc,
    reference=_ref_circuit_build,
    core=_core_circuit_build_rlc_1035,
    parity=parity_matrices,
    phase="P1",
    tolerance=TOL_BUILD,
    description="MNA build of the parsed 23x23 RLC mesh (mna_builder.build_mna_problem)",
)
_register(
    id="assembly_jacobian_sram_248",
    category="assembly",
    size=dict(_SRAM),
    setup=_setup_assembly_sram,
    reference=_ref_assembly,
    core=_core_assembly_jacobian_sram_248,
    core_setup=_core_setup_kernel,
    parity=parity_kernel_matrix,
    phase="P1",
    tolerance=TOL_KERNEL,
    description="dense Newton Jacobian G + Jf(x) of the 10x10 SRAM (compile_nl_functions)",
)

# linear_solve --------------------------------------------------------------------------------
_register(
    id="direct_solve_rlc_1035",
    category="linear_solve",
    size={"n": 1035, "nnz": 4071},
    setup=_setup_direct_solve_rlc,
    reference=_ref_direct_solve,
    core=_core_direct_solve_rlc_1035,
    core_setup=_core_setup_csc,
    parity=parity_array,
    phase="P0",
    tolerance=TOL_LINEAR,
    description="factor + solve of G + C/h (h = 1 ns) on the RLC mesh: np.linalg.solve vs _core.klu_solve",
)
_register(
    id="direct_solve_sram_248",
    category="linear_solve",
    size={"n": 248, "nnz": 2150},
    setup=_setup_direct_solve_sram,
    reference=_ref_direct_solve,
    core=_core_direct_solve_sram_248,
    core_setup=_core_setup_csc,
    parity=parity_array,
    phase="P0",
    tolerance=TOL_LINEAR,
    description="factor + solve of G + C/h (h = 0.25 ns) on the SRAM: np.linalg.solve vs _core.klu_solve",
)
_register(
    id="direct_solve_complex_rlc_1035",
    category="linear_solve",
    size={"n": 1035, "nnz": 4071},
    setup=_setup_direct_solve_complex_rlc,
    reference=_ref_direct_solve,
    core=_core_direct_solve_complex_rlc_1035,
    core_setup=_core_setup_csc_any,
    parity=parity_array,
    phase="P2",
    tolerance=TOL_LINEAR,
    description="complex factor + solve of G + j*omega*C (f = 100 MHz) on the RLC mesh (np.linalg.solve)",
)
_register(
    id="direct_solve_complex_sram_248",
    category="linear_solve",
    size={"n": 248, "nnz": 2150},
    setup=_setup_direct_solve_complex_sram,
    reference=_ref_direct_solve,
    core=_core_direct_solve_complex_sram_248,
    core_setup=_core_setup_csc_any,
    parity=parity_array,
    phase="P2",
    tolerance=TOL_LINEAR,
    description="complex factor + solve of G + j*omega*C (f = 1 GHz) on the SRAM (np.linalg.solve)",
)

# krylov --------------------------------------------------------------------------------------
for _method, _core_fn in (
    ("gmres", _core_krylov_gmres_rlc_1035),
    ("minres", _core_krylov_minres_rlc_1035),
    ("cg", _core_krylov_cg_rlc_1035),
):
    _register(
        id=f"krylov_{_method}_rlc_1035",
        category="krylov",
        size={"n": 1035, "nnz": 4071},
        setup=_setup_krylov(_method),
        reference=_ref_krylov,
        core=_core_fn,
        parity=parity_array,
        phase="P2",
        tolerance=TOL_KRYLOV,
        description=f"krylov_solve_linear(method={_method!r}) on the sparse RLC operator G + C/h (h = 1 ns)",
    )

# newton --------------------------------------------------------------------------------------
_register(
    id="op_sram_248",
    category="newton",
    size=dict(_SRAM),
    setup=_setup_basic("sram", "op", {"max_iter": 200}, {}),
    reference=_ref_basic,
    core=_core_op_sram_248,
    core_setup=_core_setup_basic,
    parity=parity_analysis,
    phase="P2",
    tolerance=TOL_NONLINEAR,
    description="SRAM .op through core_basic.run_basic_analysis",
)

# transient -----------------------------------------------------------------------------------
_register(
    id="tran_sram_248",
    category="transient",
    size={**_SRAM, "steps": 24},
    setup=_setup_basic(
        "sram", "tran", {"tran_stop": 6e-9, "tran_step": 0.25e-9, "max_iter": 200}, {"tran_step": 1e-9}
    ),
    reference=_ref_basic,
    core=_core_tran_sram_248,
    core_setup=_core_setup_basic,
    parity=parity_analysis,
    phase="P3",
    tolerance=TOL_NONLINEAR,
    description="SRAM .tran 6n/0.25n (UI demo) through run_basic_analysis",
)
_register(
    id="tran_bjt_3stage_2001",
    category="transient",
    size={"n": 14, "devices": 4, "steps": 2000},
    setup=_setup_basic("bjt", "tran", {"tran_stop": 10e-3, "tran_step": 5e-6}, {"tran_stop": 50e-6}),
    reference=_ref_basic,
    core=_core_tran_bjt_3stage_2001,
    core_setup=_core_setup_basic,
    parity=parity_analysis,
    phase="P3",
    tolerance=TOL_NONLINEAR,
    description="three-stage BJT amplifier .tran 10m/5u (2001 points) through run_basic_analysis",
)
_register(
    id="tran_rlc_1035_dense",
    category="transient",
    size={**_RLC, "steps": 40},
    setup=_setup_basic("rlc", "tran", {"tran_stop": 40e-9, "tran_step": 1e-9}, {"tran_stop": 5e-9}),
    reference=_ref_basic,
    core=_core_tran_rlc_1035_dense,
    core_setup=_core_setup_basic,
    parity=parity_analysis,
    phase="P3",
    tolerance=TOL_LINEAR,
    description="RLC mesh .tran 40 x 1 ns, direct solves, through run_basic_analysis",
)
_register(
    id="tran_rlc_1035_krylov",
    category="transient",
    size={**_RLC, "steps": 40},
    setup=_setup_basic(
        "rlc", "tran", {"tran_stop": 40e-9, "tran_step": 1e-9, "use_krylov": True}, {"tran_stop": 5e-9}
    ),
    reference=_ref_basic,
    core=_core_tran_rlc_1035_krylov,
    core_setup=_core_setup_basic,
    parity=parity_analysis,
    phase="P3",
    tolerance=TOL_KRYLOV,
    description="RLC mesh .tran 40 x 1 ns, sparse Krylov, through run_basic_analysis",
)

# ac ------------------------------------------------------------------------------------------
_register(
    id="ac_rlc_1035",
    category="ac",
    size={**_RLC, "points": 100},
    setup=_setup_basic("rlc", "ac", {"ac_start": 1e6, "ac_stop": 1e9, "ac_points": 100}, {"ac_points": 5}),
    reference=_ref_basic,
    core=_core_ac_rlc_1035,
    core_setup=_core_setup_ac_live,
    parity=parity_analysis,
    phase="P3",
    tolerance=TOL_LINEAR,
    description="RLC mesh .ac 1 MHz..1 GHz, 100 points, through run_basic_analysis",
)
_register(
    id="ac_bjt_3stage",
    category="ac",
    size={"n": 14, "devices": 4, "points": 400},
    setup=_setup_basic("bjt", "ac", {"ac_start": 10.0, "ac_stop": 1e6, "ac_points": 400}, {"ac_points": 5}),
    reference=_ref_basic,
    core=_core_ac_bjt_3stage,
    core_setup=_core_setup_ac_live,
    parity=parity_analysis,
    phase="P3",
    tolerance=TOL_NONLINEAR,
    description="three-stage BJT amplifier .ac 10 Hz..1 MHz, 400 points (Level-1 op + sweep)",
)

# hb ------------------------------------------------------------------------------------------
_register(
    id="hb_mixer_h205",
    category="hb",
    size={"n": 3, "harmonics": 205, "dim": 1233},
    setup=_setup_basic("mixer_h205", "hb", {"hb_harmonics": 205, "max_iter": 100, "f_tol": 1e-6}, {"hb_harmonics": 8}),
    reference=_ref_advanced,
    core=_core_hb_mixer_h205,
    parity=parity_analysis,
    phase="P4",
    tolerance=TOL_HB,
    description="close-tone diode mixer (1 GHz + 1.005 GHz) .hb 205 through core_advanced.run_advanced_analysis",
)
_register(
    id="hb_mixer_h64",
    category="hb",
    size={"n": 3, "harmonics": 64, "dim": 387},
    setup=_setup_basic("mixer_h64", "hb", {"hb_harmonics": 64, "max_iter": 100, "f_tol": 1e-6}, {"hb_harmonics": 8}),
    reference=_ref_advanced,
    core=_core_hb_mixer_h64,
    parity=parity_analysis,
    phase="P4",
    tolerance=TOL_HB,
    description="diode mixer with tones on a 20 MHz grid (1 GHz + 1.02 GHz) .hb 64 through run_advanced_analysis",
)

# mor -----------------------------------------------------------------------------------------
_register(
    id="mor_basis_rlc_1035",
    category="mor",
    size={"n": 1035, "order": 12},
    setup=_setup_mor_basis_rlc,
    reference=_ref_mor_basis,
    core=_core_mor_basis_rlc_1035,
    parity=parity_basis,
    phase="P5",
    tolerance=TOL_MOR,
    description=(
        "rational Krylov basis (shifts 0, 1/h; outputs V(n_0_0), V(n_22_22); auto order) "
        "via mor.linear.build_linear_krylov_basis"
    ),
)
_register(
    id="tpwl_train_sram_248",
    category="mor",
    size={**_SRAM, "steps": 24, "order": 24},
    setup=_setup_tpwl_sram,
    reference=_ref_tpwl,
    core=_core_tpwl_train_sram_248,
    parity=parity_tpwl,
    phase="P5",
    tolerance=TOL_MOR,
    description="TPWL training on the SRAM .tran 6n/0.25n (full transient, POD, local models) via mor.tpwl._train_tpwl",
)


# ---------------------------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------------------------


def all_units() -> list[Unit]:
    return list(UNITS.values())


def get_unit(unit_id: str) -> Unit:
    try:
        return UNITS[unit_id]
    except KeyError:
        raise KeyError(f"unknown unit {unit_id!r}; known units: {', '.join(UNITS)}") from None


def select_units(ids: Iterable[str] | None = None, categories: Iterable[str] | None = None) -> list[Unit]:
    """Units by id (in the given order) and/or category (in registry order)."""

    selected: list[Unit] = []
    if ids:
        selected = [get_unit(unit_id) for unit_id in ids]
    else:
        selected = all_units()
    if categories:
        wanted = set(categories)
        unknown = wanted.difference(CATEGORIES)
        if unknown:
            raise KeyError(f"unknown categories {sorted(unknown)}; known: {', '.join(CATEGORIES)}")
        selected = [unit for unit in selected if unit.category in wanted]
    return selected
