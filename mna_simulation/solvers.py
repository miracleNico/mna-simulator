"""Numerical solvers shared by basic and advanced cores."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .api.contracts import MnaProblem, SpectrumResult, WaveformResult
from .errors import error_handler

# Linear algebra: every direct and Krylov solve runs in the native core
# (mna_simulation/native/linalg.py keeps the reference signatures and the
# metadata/acceptance rules; the pure-Python implementations live only in the
# frozen test oracle tests/reference/mna_reference).
from .native.linalg import (  # noqa: E402,F401 - re-exported facades
    _krylov_method_name,
    _krylov_route,
    classify_krylov_matrix,
    describe_krylov_choice,
    krylov_solve_linear,
    resolve_krylov_rank,
    solve_linear,
)

from .native.expressions import _prepare_sympy_expression, _sympy_locals  # noqa: E402,F401


def _native_core():
    core = _NATIVE_CORE.get("core")
    if core is None:
        from .native import require_core

        core = require_core()
        _NATIVE_CORE["core"] = core
    return core


_NATIVE_CORE: dict[str, object] = {}


def _nmos_level1_forward(
    vgs: float,
    vds: float,
    beta: float,
    vth: float,
    lambda_: float,
) -> tuple[float, float, float]:
    """Return IDS, dIDS/dVGS, dIDS/dVDS for an NMOS with VDS >= 0."""

    if vgs <= vth:
        return 0.0, 0.0, 0.0
    vov = vgs - vth
    if vds < vov:
        channel = vov * vds - 0.5 * vds * vds
        modulation = 1.0 + lambda_ * vds
        current = beta * channel * modulation
        d_vgs = beta * vds * modulation
        d_vds = beta * ((vov - vds) * modulation + channel * lambda_)
        return current, d_vgs, d_vds
    current = 0.5 * beta * vov * vov * (1.0 + lambda_ * vds)
    d_vgs = beta * vov * (1.0 + lambda_ * vds)
    d_vds = 0.5 * beta * vov * vov * lambda_
    return current, d_vgs, d_vds


def _nmos_level1_current_and_derivatives(
    vd: float,
    vg: float,
    vs: float,
    beta: float,
    vth: float,
    lambda_: float,
) -> tuple[float, float, float, float]:
    """Return drain-to-source current and derivatives for a symmetric NMOS (native kernel)."""

    return _native_core().eval_mos_level1(1, float(beta), float(vth), float(lambda_), float(vd), float(vg), float(vs))


def _level1_mos_current_and_derivatives(
    device: dict[str, object],
    vd: float,
    vg: float,
    vs: float,
) -> tuple[float, float, float, float]:
    """(ids, dIds/dVd, dIds/dVg, dIds/dVs) of a Level-1 MOS dict (native kernel, PMOS mirrored)."""

    sign = -1 if str(device["type"]).upper() == "PMOS" else 1
    return _native_core().eval_mos_level1(
        sign,
        float(device["beta"]),
        float(device["vth"]),
        float(device["lambda"]),
        float(vd),
        float(vg),
        float(vs),
    )


def _evaluate_device_dicts(
    size: int,
    x_values: np.ndarray,
    mos_devices: list[dict[str, object]] | None = None,
    bjt_devices: list[dict[str, object]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    from .native.circuit import bjt_table, build_circuit, mos_table

    circuit = build_circuit(
        size,
        mos=mos_table(mos_devices) if mos_devices else None,
        bjt=bjt_table(bjt_devices) if bjt_devices else None,
    )
    x = np.ascontiguousarray(np.asarray(x_values, dtype=float).reshape(-1)[:size])
    f, jacobian = _native_core().device_f_jacobian_dense(circuit, x, 0.0)
    return f, jacobian


def _evaluate_level1_mos_devices(
    devices: list[dict[str, object]],
    size: int,
    x_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(f (size, 1), dense J (size, size)) of a Level-1 MOS dict list (native kernels)."""

    return _evaluate_device_dicts(size, x_values, mos_devices=devices)


def _bjt_level1_npn_currents_and_derivatives(
    vc: float,
    vb: float,
    ve: float,
    is_: float,
    bf: float,
    br: float,
    vaf: float,
    var: float,
) -> tuple[tuple[float, float, float], tuple[tuple[float, float, float], ...]]:
    """Return NPN terminal currents and dI/d(Vc,Vb,Ve) (native kernel).

    Currents are positive when leaving the external terminal into the compact
    device. This Ebers-Moll-lite model is intentionally small: transport
    currents, finite forward/reverse beta, and simple Early modulation.
    """

    return _native_core().eval_bjt_level1(
        1, float(is_), float(bf), float(br), float(vaf), float(var), float(vc), float(vb), float(ve)
    )


def _level1_bjt_currents_and_derivatives(
    device: dict[str, object],
    vc: float,
    vb: float,
    ve: float,
) -> tuple[tuple[float, float, float], tuple[tuple[float, float, float], ...]]:
    """Terminal currents and derivatives of a Level-1 BJT dict (native kernel, PNP mirrored)."""

    sign = -1 if str(device["type"]).upper() == "QPNP" else 1
    return _native_core().eval_bjt_level1(
        sign,
        float(device["is"]),
        float(device["bf"]),
        float(device["br"]),
        float(device["vaf"]),
        float(device["var"]),
        float(vc),
        float(vb),
        float(ve),
    )


def _evaluate_level1_bjt_devices(
    devices: list[dict[str, object]],
    size: int,
    x_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(f (size, 1), dense J (size, size)) of a Level-1 BJT dict list (native kernels)."""

    return _evaluate_device_dicts(size, x_values, bjt_devices=devices)


def compile_nl_functions(
    f_str_vector: np.ndarray,
    size: int,
    level1_mos_devices: list[dict[str, object]] | None = None,
    level1_bjt_devices: list[dict[str, object]] | None = None,
):
    """Compile nonlinear current expressions and their Jacobian.

    Returns ``f_func(*x) -> (size, 1)`` and ``j_func(*x) -> (size, size)``
    evaluated by the native core: every non-"0" row of ``f_str_vector`` becomes
    a behavioral program (forward-mode derivatives) injected at that row, and
    the Level-1 MOS/BJT dictionaries become native devices.
    """

    from .native.circuit import compile_device_functions

    return compile_device_functions(f_str_vector, size, level1_mos_devices, level1_bjt_devices)


def compile_time_source_func(b_time_str_vector: np.ndarray):
    """Compile time-domain source expressions into a callable ``evaluate(t) -> (n, 1)``."""

    from .native.expressions import compile_expression

    size = b_time_str_vector.shape[0]
    rows: list[int] = []
    handles: list[object] = []
    for row, expr_str in enumerate(b_time_str_vector.flatten()):
        text = str(expr_str)
        if text == "0":
            continue
        rows.append(row)
        handles.append(compile_expression(text, time_dependent=True, n_unknowns=0).handle)
    evaluate_program = _native_core().evaluate_program
    no_unknowns = np.zeros(0)
    entries = list(zip(rows, handles))

    def evaluate(t_value: float) -> np.ndarray:
        t = float(t_value)
        values = np.zeros((size, 1), dtype=float)
        for row, handle in entries:
            values[row, 0] = evaluate_program(handle, no_unknowns, t)
        return values

    return evaluate


def is_nonlinear(f_str_vector: np.ndarray) -> bool:
    """Return True when the symbolic nonlinear vector is non-zero."""

    return any(expr != "0" for expr in f_str_vector.flatten())


def _transient_time_points(t_stop: float, t_step: float) -> np.ndarray:
    """Return monotonic transient sample points ending exactly at ``t_stop``."""

    if t_stop < 0:
        error_handler("TRAN_ERROR: t_stop must be non-negative.")
    if t_step <= 0:
        error_handler("TRAN_ERROR: t_step must be positive.")
    if t_stop == 0:
        return np.array([0.0], dtype=float)

    tolerance = max(abs(t_stop), abs(t_step), np.finfo(float).eps) * 1e-9
    points = [0.0]
    current = 0.0
    while current < t_stop - tolerance:
        next_time = current + t_step
        current = t_stop if next_time >= t_stop - tolerance else next_time
        if current <= points[-1]:
            error_handler("TRAN_ERROR: transient time grid failed to advance.")
        points.append(float(current))
    if points[-1] != t_stop:
        points[-1] = float(t_stop)
    return np.asarray(points, dtype=float)


def _finite_vector_norm(vector: np.ndarray) -> float:
    if not np.all(np.isfinite(vector)):
        return np.inf
    return float(np.linalg.norm(vector))


def _damped_newton_step(
    x: np.ndarray,
    delta_x: np.ndarray,
    current_norm: float,
    residual_at: Callable[[np.ndarray], np.ndarray],
    max_trials: int = 28,
) -> tuple[np.ndarray, float, float]:
    """Pick a finite damped Newton step without committing the full step first."""

    best_x = x
    best_norm = current_norm
    best_scale = 0.0
    scale = 1.0
    for _ in range(max_trials):
        trial = x + delta_x * scale
        try:
            trial_norm = _finite_vector_norm(residual_at(trial))
        except (FloatingPointError, OverflowError, ValueError):
            trial_norm = np.inf
        if np.isfinite(trial_norm) and trial_norm < best_norm:
            best_x = trial
            best_norm = trial_norm
            best_scale = scale
        if np.isfinite(trial_norm) and trial_norm <= current_norm:
            return trial, trial_norm, scale
        scale *= 0.5

    if best_scale > 0.0:
        return best_x, best_norm, best_scale
    return x, current_norm, 0.0


def solve_dc_nr(
    G: np.ndarray,
    f_str_vector: np.ndarray,
    b: np.ndarray,
    max_iter: int = 10000,
    v_tol: float = 1e-9,
    f_tol: float = 1e-9,
    init_cond: np.ndarray | str = "DEFAULT",
    use_krylov: bool = False,
    krylov_tol: float = 1e-9,
    krylov_max_iter: int = 2000,
    krylov_restart: int = 80,
    krylov_rank: int | str | None = "auto",
    krylov_method: str = "auto",
    krylov_stats: list[dict[str, object]] | None = None,
    level1_mos_devices: list[dict[str, object]] | None = None,
    level1_bjt_devices: list[dict[str, object]] | None = None,
) -> np.ndarray:
    """Solve the DC operating point using linear solve or Newton-Raphson."""

    size = G.shape[0]
    if not is_nonlinear(f_str_vector) and not level1_mos_devices and not level1_bjt_devices:
        try:
            return solve_linear(
                G,
                b,
                use_krylov=use_krylov,
                krylov_tol=krylov_tol,
                krylov_max_iter=krylov_max_iter,
                krylov_restart=krylov_restart,
                krylov_rank=krylov_rank,
                krylov_method=krylov_method,
                krylov_stats=krylov_stats,
            )
        except np.linalg.LinAlgError as exc:
            error_handler(f"NR_CONVERGENCE: Linear system is singular (G matrix): {exc}")
            raise

    f_func, Jf_func = compile_nl_functions(
        f_str_vector,
        size,
        level1_mos_devices=level1_mos_devices,
        level1_bjt_devices=level1_bjt_devices,
    )
    if isinstance(init_cond, str):
        if init_cond == "DEFAULT":
            x = np.zeros((size, 1))
        elif init_cond == "RANDOM":
            x = np.random.uniform(-1.0, 1.0, size=(size, 1))
        else:
            error_handler(f"NETLIST_FATAL: Unsupported init condition '{init_cond}'.")
    else:
        x = init_cond.copy()

    for iteration in range(max_iter):
        x_flat = x.flatten()
        f_k = np.asarray(f_func(*x_flat), dtype=float).reshape(size, 1)
        F_k = (G @ x) + f_k - b

        norm_F_now = _finite_vector_norm(F_k)
        if norm_F_now < f_tol:
            return x

        Jf_k = np.asarray(Jf_func(*x_flat), dtype=float)
        J_k = G + Jf_k

        try:
            delta_x = solve_linear(
                J_k,
                -F_k,
                use_krylov=use_krylov,
                krylov_tol=krylov_tol,
                krylov_max_iter=krylov_max_iter,
                krylov_restart=krylov_restart,
                krylov_rank=krylov_rank,
                krylov_method=krylov_method,
                krylov_stats=krylov_stats,
            )
        except np.linalg.LinAlgError as exc:
            error_handler(f"JACOBIAN_SINGULAR: Jacobian is singular at iteration {iteration}.")
            raise exc

        def residual_at(trial: np.ndarray) -> np.ndarray:
            trial_flat = trial.flatten()
            f_trial = np.asarray(f_func(*trial_flat), dtype=float).reshape(size, 1)
            return (G @ trial) + f_trial - b

        x_next, next_norm, step_scale = _damped_newton_step(x, delta_x, norm_F_now, residual_at)
        accepted_delta = x_next - x
        x = x_next
        if next_norm < f_tol:
            return x
        if step_scale == 0.0:
            error_handler(f"NR_CONVERGENCE: Newton step could not reduce residual at iteration {iteration}.")
        if np.linalg.norm(accepted_delta) < v_tol:
            return x

    error_handler(f"NR_CONVERGENCE: Solver did not converge after {max_iter} iterations.")


def solve_transient(
    problem: MnaProblem,
    t_stop: float,
    t_step: float,
    max_iter: int = 10000,
    v_tol: float = 1e-9,
    f_tol: float = 1e-9,
    init_cond: np.ndarray | None = None,
    use_krylov: bool = False,
    krylov_tol: float = 1e-9,
    krylov_max_iter: int = 2000,
    krylov_restart: int = 80,
    krylov_rank: int | str | None = "auto",
    krylov_method: str = "auto",
    krylov_stats: list[dict[str, object]] | None = None,
) -> WaveformResult:
    """Solve transient response with backward Euler."""

    size = problem.G.shape[0]
    solver_f_str = problem.solver_f_str if problem.solver_f_str is not None else problem.f_str
    level1_devices = problem.level1_mos_devices
    level1_bjt_devices = problem.level1_bjt_devices
    has_nonlinear_terms = is_nonlinear(solver_f_str) or bool(level1_devices) or bool(level1_bjt_devices)
    if has_nonlinear_terms:
        f_func, Jf_func = compile_nl_functions(
            solver_f_str,
            size,
            level1_mos_devices=level1_devices,
            level1_bjt_devices=level1_bjt_devices,
        )
    else:
        f_func = lambda *args: np.zeros((size, 1))  # noqa: E731
        Jf_func = lambda *args: np.zeros((size, size))  # noqa: E731

    b_time_func = compile_time_source_func(problem.b_time_str)

    if init_cond is None:
        b_t0 = problem.b_dc + b_time_func(0.0)
        x_prev = solve_dc_nr(
            problem.G,
            solver_f_str,
            b_t0,
            max_iter=max_iter,
            v_tol=v_tol,
            f_tol=f_tol,
            use_krylov=use_krylov,
            krylov_tol=krylov_tol,
            krylov_max_iter=krylov_max_iter,
            krylov_restart=krylov_restart,
            krylov_rank=krylov_rank,
            krylov_method=krylov_method,
            krylov_stats=krylov_stats,
            level1_mos_devices=level1_devices,
            level1_bjt_devices=level1_bjt_devices,
        )
    else:
        x_prev = init_cond.copy()

    t_points = _transient_time_points(t_stop, t_step)
    values = np.zeros((size, len(t_points)))
    values[:, 0] = x_prev.flatten()
    sparse_linear_krylov = (
        use_krylov
        and not has_nonlinear_terms
        and problem.G_sparse is not None
        and problem.C_sparse is not None
    )
    cached_step_size: float | None = None
    cached_C_h_sparse = None
    cached_G_be_sparse = None

    for step in range(1, len(t_points)):
        step_size = float(t_points[step] - t_points[step - 1])
        x_k = x_prev.copy()
        b_now = problem.b_dc + b_time_func(float(t_points[step]))

        if sparse_linear_krylov:
            if cached_step_size != step_size:
                cached_step_size = step_size
                cached_C_h_sparse = problem.C_sparse / step_size
                cached_G_be_sparse = problem.G_sparse + cached_C_h_sparse
            b_be = (cached_C_h_sparse @ x_prev) + b_now
            x_next = solve_linear(
                cached_G_be_sparse,
                b_be,
                use_krylov=use_krylov,
                krylov_tol=krylov_tol,
                krylov_max_iter=krylov_max_iter,
                krylov_restart=krylov_restart,
                krylov_rank=krylov_rank,
                krylov_method=krylov_method,
                krylov_stats=krylov_stats,
                x0=x_prev,
            )
            values[:, step] = x_next.flatten()
            x_prev = x_next
            continue

        C_h = problem.C / step_size
        G_be = problem.G + C_h
        b_be = (C_h @ x_prev) + b_now

        if not has_nonlinear_terms:
            x_next = solve_linear(
                G_be,
                b_be,
                use_krylov=use_krylov,
                krylov_tol=krylov_tol,
                krylov_max_iter=krylov_max_iter,
                krylov_restart=krylov_restart,
                krylov_rank=krylov_rank,
                krylov_method=krylov_method,
                krylov_stats=krylov_stats,
                x0=x_prev,
            )
            values[:, step] = x_next.flatten()
            x_prev = x_next
            continue

        converged = False

        for iteration in range(max_iter):
            x_flat = x_k.flatten()
            f_k = np.asarray(f_func(*x_flat), dtype=float).reshape(size, 1)
            F_k = (G_be @ x_k) + f_k - b_be

            norm_F = _finite_vector_norm(F_k)
            if norm_F < f_tol:
                converged = True
                break

            Jf_k = np.asarray(Jf_func(*x_flat), dtype=float)
            J_k = G_be + Jf_k

            try:
                delta_x = solve_linear(
                    J_k,
                    -F_k,
                    use_krylov=use_krylov,
                    krylov_tol=krylov_tol,
                    krylov_max_iter=krylov_max_iter,
                    krylov_restart=krylov_restart,
                    krylov_rank=krylov_rank,
                    krylov_method=krylov_method,
                    krylov_stats=krylov_stats,
                )
            except np.linalg.LinAlgError as exc:
                error_handler(f"JACOBIAN_SINGULAR: Jacobian singular at t={t_points[step]:.2e}, iter={iteration}")
                raise exc

            def residual_at(trial: np.ndarray) -> np.ndarray:
                trial_flat = trial.flatten()
                f_trial = np.asarray(f_func(*trial_flat), dtype=float).reshape(size, 1)
                return (G_be @ trial) + f_trial - b_be

            x_next, next_norm, step_scale = _damped_newton_step(x_k, delta_x, norm_F, residual_at)
            accepted_delta = x_next - x_k
            x_k = x_next
            if next_norm < f_tol:
                converged = True
                break
            if step_scale == 0.0:
                error_handler(f"NR_CONVERGENCE: Newton step could not reduce residual at t={t_points[step]:.2e}, iter={iteration}")
            if np.linalg.norm(accepted_delta) < v_tol:
                converged = True
                break

        if not converged:
            error_handler(f"NR_CONVERGENCE: NR failed to converge at t={t_points[step]:.2e} s")

        values[:, step] = x_k.flatten()
        x_prev = x_k

    return WaveformResult(time=t_points, values=values, labels=problem.metadata["labels"])


def _ac_thread_count(num_threads: object) -> int:
    """Threads for the native AC sweep: a positive integer as given, anything else 1.

    The service replaces ``None``/``"auto"`` by the leased count before the
    backend runs; direct callers that leave it unset get a serial sweep.
    """

    if isinstance(num_threads, (int, np.integer)) and not isinstance(num_threads, bool) and num_threads > 0:
        return int(num_threads)
    return 1


def _ac_krylov_entry(
    entry: dict[str, object],
    n: int,
    krylov_method: str,
    resolved_rank: int,
    rank_mode: str,
    krylov_max_iter: int,
) -> dict[str, object]:
    """The ``krylov_stats`` entry ``solve_linear`` records for one AC frequency.

    Same keys and values as ``native.linalg.krylov_solve_linear`` on the dense
    complex system the pre-native AC loop passed (operator storage ``dense``
    on the small-system kernels, ``csr`` on the SciPy-semantics ones), plus
    ``used_direct_fallback``.
    """

    nnz = int(entry["matrix_nnz"])
    density = float(nnz / (n * n)) if n > 0 else 0.0
    route = str(entry["route"])
    info: dict[str, object] = {"method": entry["method"], "engine": "mna_core"}
    if entry["scipy_semantics"]:
        info.update(
            {
                "operator_storage": "csr",
                "matrix_nnz": nnz,
                "matrix_density": density,
                "converged": bool(entry["converged"]),
                "iterations": int(entry["iterations"]),
                "relative_residual": float(entry["relative_residual"]),
                "iteration_budget": int(entry["iteration_budget"]),
                "effective_restart": entry["effective_restart"],
                "exit_code": int(entry["exit_code"]),
            }
        )
    else:
        info.update(
            {
                "converged": bool(entry["converged"]),
                "iterations": int(entry["iterations"]),
                "relative_residual": float(entry["relative_residual"]),
            }
        )
    info["requested_method"] = krylov_method
    info["matrix_kind"] = entry["matrix_kind"]
    info["route"] = route
    info["matrix_dimension"] = n
    info["matrix_nnz"] = info.get("matrix_nnz", nnz)
    info["matrix_density"] = info.get("matrix_density", density)
    info["operator_storage"] = info.get("operator_storage", "dense")
    info["resolved_rank"] = resolved_rank
    info["rank_mode"] = rank_mode
    info["iteration_budget"] = info.get("iteration_budget", int(krylov_max_iter) if route == "general" else resolved_rank)
    if route == "general":
        info["effective_restart"] = info.get("effective_restart", min(resolved_rank, n))
    info["threads_used"] = int(entry["threads_used"])
    return {**info, "used_direct_fallback": bool(entry["used_direct_fallback"])}


def solve_ac(
    problem: MnaProblem,
    f_start: float,
    f_stop: float,
    points: int,
    use_krylov: bool = False,
    krylov_tol: float = 1e-9,
    krylov_max_iter: int = 2000,
    krylov_restart: int = 80,
    krylov_rank: int | str | None = "auto",
    krylov_method: str = "auto",
    krylov_stats: list[dict[str, object]] | None = None,
    *,
    max_iter: int = 10000,
    v_tol: float = 1e-9,
    f_tol: float = 1e-9,
    num_threads: int | str | None = 1,
) -> SpectrumResult:
    """Solve the small-signal AC response over a linear frequency sweep (native core).

    Nonlinear circuits are linearized at the ``b_dc`` operating point of
    :func:`solve_dc_nr`, which receives ``max_iter``/``v_tol``/``f_tol``
    (CPP_MIGRATION_PLAN.md 2.4.1). The sweep runs in ``_core.ac_sweep``:
    ``G + gmin * gmin_stamps + J(x_op) + j*omega*C`` in the compiled circuit's
    pattern, complex KLU per frequency (or the Krylov kernels with
    ``solve_linear``'s acceptance and fallback rule, recording the same
    ``krylov_stats`` entries), parallel over frequencies on ``num_threads``
    threads with bitwise identical results for any thread count.
    ``points <= 0`` and a singular system raise ``AC_ERROR``; ``f_start = 0``
    is a plain DC point.
    """

    if points is not None and points <= 0:
        error_handler(f"AC_ERROR: AC sweep needs at least one frequency point (points = {points}).")

    solver_f_str = problem.solver_f_str if problem.solver_f_str is not None else problem.f_str
    level1_devices = problem.level1_mos_devices
    level1_bjt_devices = problem.level1_bjt_devices
    spectrum_metadata: dict[str, object] = {}
    x_op: np.ndarray | None = None
    if is_nonlinear(solver_f_str) or level1_devices or level1_bjt_devices:
        dc_solution = solve_dc_nr(
            problem.G,
            solver_f_str,
            problem.b_dc,
            max_iter=max_iter,
            v_tol=v_tol,
            f_tol=f_tol,
            use_krylov=use_krylov,
            krylov_tol=krylov_tol,
            krylov_max_iter=krylov_max_iter,
            krylov_restart=krylov_restart,
            krylov_rank=krylov_rank,
            krylov_method=krylov_method,
            krylov_stats=krylov_stats,
            level1_mos_devices=level1_devices,
            level1_bjt_devices=level1_bjt_devices,
        )
        x_op = np.ascontiguousarray(np.asarray(dc_solution, dtype=float).reshape(-1))
        spectrum_metadata["op_used"] = True
        spectrum_metadata["operating_point"] = {
            "labels": problem.metadata["labels"],
            "values": np.asarray(dc_solution).flatten().tolist(),
        }
        spectrum_metadata["linearized_device_count"] = len(level1_devices) + len(level1_bjt_devices)
    else:
        spectrum_metadata["op_used"] = False
        spectrum_metadata["linearized_device_count"] = 0

    circuit = problem.native
    if circuit is None:
        # Only a behavioral expression the native VM cannot compile leaves no circuit, and the
        # operating point above has already raised the reference error for it.
        error_handler("NETLIST_FATAL: The circuit could not be compiled for the native core.")
    n = int(circuit.n)

    frequencies = np.linspace(f_start, f_stop, points)
    options: dict[str, object] = {"use_krylov": bool(use_krylov)}
    resolved_rank, rank_mode = 0, "auto"
    if use_krylov:
        _krylov_route("general", krylov_method)  # unknown method: the facade's ValueError, before any solve
        resolved_rank, rank_mode = resolve_krylov_rank(n, krylov_restart if krylov_rank is None else krylov_rank, krylov_restart)
        options.update(
            {
                "krylov_method": str(krylov_method),
                "krylov_tol": float(krylov_tol),
                "krylov_max_iter": int(krylov_max_iter),
                "krylov_rank": int(resolved_rank),
            }
        )

    core = _native_core()
    try:
        magnitudes, info = core.ac_sweep(
            circuit,
            x_op,
            np.ascontiguousarray(np.asarray(problem.b_ac, dtype=complex).reshape(-1)),
            np.ascontiguousarray(frequencies, dtype=float),
            float(problem.gmin),
            options,
            _ac_thread_count(num_threads),
        )
    except core.CoreError as exc:
        if getattr(exc, "code", None) == "AC_ERROR":
            error_handler(f"AC_ERROR: {exc}")
        raise

    if use_krylov and krylov_stats is not None:
        krylov_stats.extend(
            _ac_krylov_entry(entry, n, krylov_method, resolved_rank, rank_mode, krylov_max_iter) for entry in info["krylov"]
        )

    return SpectrumResult(
        frequencies=frequencies,
        magnitudes=magnitudes,
        labels=problem.metadata["labels"],
        metadata=spectrum_metadata,
    )


def create_fourier_matrices(n_harmonics: int, time_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return DFT and inverse DFT matrices for real-valued harmonics."""

    S = 2 * n_harmonics + 1
    K = len(time_points)
    Gamma = np.zeros((S, K))
    Gamma_inv = np.zeros((K, S))

    for k in range(K):
        Gamma_inv[k, 0] = 1.0
        for h in range(1, n_harmonics + 1):
            angle = 2 * np.pi * h * k / K
            Gamma_inv[k, 2 * h - 1] = np.cos(angle)
            Gamma_inv[k, 2 * h] = np.sin(angle)

    Gamma[0, :] = 1.0 / K
    for h in range(1, n_harmonics + 1):
        for k in range(K):
            angle = 2 * np.pi * h * k / K
            Gamma[2 * h - 1, k] = (2.0 / K) * np.cos(angle)
            Gamma[2 * h, k] = (2.0 / K) * np.sin(angle)

    return Gamma, Gamma_inv


def reconstruct_time_domain(x_bar: np.ndarray, num_vars: int, H: int, omega_0: float, t_array: np.ndarray) -> np.ndarray:
    """Reconstruct time-domain waveform from harmonic coefficients."""

    freq_dim = 2 * H + 1
    results = np.zeros((num_vars, len(t_array)))
    reshaped = x_bar.reshape((num_vars, freq_dim))

    for i in range(num_vars):
        values = np.full_like(t_array, reshaped[i, 0], dtype=float)
        for h in range(1, H + 1):
            a_h = reshaped[i, 2 * h - 1]
            b_h = reshaped[i, 2 * h]
            values += a_h * np.cos(h * omega_0 * t_array) + b_h * np.sin(h * omega_0 * t_array)
        results[i, :] = values

    return results


def solve_harmonic_balance(
    problem: MnaProblem,
    omega_0: float,
    H: int,
    max_iter: int = 100,
    tol: float = 1e-6,
    use_krylov: bool = False,
    krylov_tol: float = 1e-9,
    krylov_max_iter: int = 2000,
    krylov_restart: int = 80,
    krylov_rank: int | str | None = "auto",
    krylov_method: str = "auto",
    krylov_stats: list[dict[str, object]] | None = None,
) -> np.ndarray:
    """Solve the harmonic balance system."""

    num_vars = problem.G.shape[0]
    freq_dim = 2 * H + 1
    total_dim = num_vars * freq_dim

    G_bar = np.kron(problem.G, np.eye(freq_dim))
    C_bar = np.kron(problem.C, np.eye(freq_dim))

    Omega_block = np.zeros((freq_dim, freq_dim))
    for h in range(1, H + 1):
        w_h = h * omega_0
        idx_cos = 2 * h - 1
        idx_sin = 2 * h
        Omega_block[idx_cos, idx_sin] = w_h
        Omega_block[idx_sin, idx_cos] = -w_h

    Omega_bar = np.kron(np.eye(num_vars), Omega_block)
    Y_bar = G_bar + (C_bar @ Omega_bar)

    K_samples = 2 * H + 1
    period = 2 * np.pi / omega_0
    t_points = np.linspace(0, period, K_samples, endpoint=False)
    Gamma, Gamma_inv = create_fourier_matrices(H, t_points)

    b_bar = np.zeros((total_dim, 1))
    for index in range(num_vars):
        base = index * freq_dim
        b_bar[base] = problem.b_dc[index]

    b_time_func = compile_time_source_func(problem.b_time_str)
    b_time_matrix = np.zeros((num_vars, K_samples))
    for k, t_value in enumerate(t_points):
        b_time_matrix[:, k] = b_time_func(float(t_value)).flatten()
    for index in range(num_vars):
        base = index * freq_dim
        b_bar[base : base + freq_dim, 0] += Gamma @ b_time_matrix[index, :]

    solver_f_str = problem.solver_f_str if problem.solver_f_str is not None else problem.f_str
    level1_devices = problem.level1_mos_devices
    level1_bjt_devices = problem.level1_bjt_devices
    if not is_nonlinear(solver_f_str) and not level1_devices and not level1_bjt_devices:
        try:
            return solve_linear(
                Y_bar,
                b_bar,
                use_krylov=use_krylov,
                krylov_tol=krylov_tol,
                krylov_max_iter=krylov_max_iter,
                krylov_restart=krylov_restart,
                krylov_rank=krylov_rank,
                krylov_method=krylov_method,
                krylov_stats=krylov_stats,
            )
        except np.linalg.LinAlgError as exc:
            error_handler("HB_ERROR: Linear harmonic solve failed.")
            raise exc

    f_func, Jf_func = compile_nl_functions(
        solver_f_str,
        num_vars,
        level1_mos_devices=level1_devices,
        level1_bjt_devices=level1_bjt_devices,
    )
    x_bar = np.zeros((total_dim, 1))

    for _ in range(max_iter):
        x_reshaped = x_bar.reshape((num_vars, freq_dim))
        x_time = x_reshaped @ Gamma_inv.T

        i_nl_time = np.zeros((num_vars, K_samples))
        G_nl_time = np.zeros((num_vars, num_vars, K_samples))

        for k in range(K_samples):
            v_inst = x_time[:, k]
            i_nl_time[:, k] = np.asarray(f_func(*v_inst), dtype=float).flatten()
            G_nl_time[:, :, k] = np.asarray(Jf_func(*v_inst), dtype=float)

        f_bar = (i_nl_time @ Gamma.T).reshape((total_dim, 1))
        residual = (Y_bar @ x_bar) + f_bar - b_bar

        if np.linalg.norm(residual) < tol:
            return x_bar

        J_nonlinear = np.zeros((total_dim, total_dim))
        for r in range(num_vars):
            for c in range(num_vars):
                g_waveform = G_nl_time[r, c, :]
                weighted = g_waveform[:, None] * Gamma_inv
                block = Gamma @ weighted
                r_idx = r * freq_dim
                c_idx = c * freq_dim
                J_nonlinear[r_idx : r_idx + freq_dim, c_idx : c_idx + freq_dim] = block

        try:
            delta_x = solve_linear(
                Y_bar + J_nonlinear,
                -residual,
                use_krylov=use_krylov,
                krylov_tol=krylov_tol,
                krylov_max_iter=krylov_max_iter,
                krylov_restart=krylov_restart,
                krylov_rank=krylov_rank,
                krylov_method=krylov_method,
                krylov_stats=krylov_stats,
            )
        except np.linalg.LinAlgError as exc:
            error_handler("HB_ERROR: Singular HB Jacobian.")
            raise exc

        x_bar = x_bar + delta_x

    error_handler("HB_ERROR: Harmonic balance solver did not converge within limit.")
