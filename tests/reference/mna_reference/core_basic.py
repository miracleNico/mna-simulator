"""Basic analysis core for .op, .tran, and .ac."""

from __future__ import annotations

import numpy as np

from .api.contracts import ComponentRecord
from .api.contracts import AnalysisMode, AnalysisOptions, MnaProblem, SimulationResult
from .errors import error_handler
from .mna_builder import build_mna_problem
from .solvers import (
    _level1_bjt_currents_and_derivatives,
    _level1_mos_current_and_derivatives,
    compile_time_source_func,
    describe_krylov_choice,
    is_nonlinear,
    solve_ac,
    solve_dc_nr,
    solve_transient,
)
from .utils import parse_value


def _linear_solver_kwargs(options: AnalysisOptions, stats: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "use_krylov": options.use_krylov,
        "krylov_tol": options.krylov_tol,
        "krylov_max_iter": options.krylov_max_iter,
        "krylov_restart": options.krylov_restart,
        "krylov_rank": options.krylov_rank,
        "krylov_method": options.krylov_method,
        "krylov_stats": stats,
    }


def _result_metadata(
    problem: MnaProblem,
    options: AnalysisOptions,
    stats: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    metadata = dict(problem.metadata)
    metadata["nonlinear_device_count"] = len(problem.level1_mos_devices) + len(problem.level1_bjt_devices)
    metadata["nonlinear_mos_device_count"] = len(problem.level1_mos_devices)
    metadata["nonlinear_bjt_device_count"] = len(problem.level1_bjt_devices)
    if options.use_krylov:
        choice = describe_krylov_choice(
            problem.G,
            rank=options.krylov_rank,
            fallback_restart=options.krylov_restart,
            max_iter=options.krylov_max_iter,
            method=options.krylov_method,
        )
        last = stats[-1] if stats else choice
        methods = sorted({str(entry.get("method", "")) for entry in (stats or [choice]) if entry.get("method")})
        metadata.update(
            {
                "linear_solver": "krylov",
                "krylov_matrix_kind": last.get("matrix_kind", choice["matrix_kind"]),
                "krylov_policy": "manual override allowed; auto: positive-definite=CG, symmetric=MINRES/CR, general=Arnoldi/GMRES",
                "krylov_requested_method": last.get("requested_method", options.krylov_method),
                "krylov_method": last.get("method"),
                "krylov_engine": last.get("engine", "python"),
                "krylov_methods": methods,
                "krylov_matrix_dimension": last.get("matrix_dimension", choice["matrix_dimension"]),
                "matrix_nnz": last.get("matrix_nnz"),
                "matrix_density": last.get("matrix_density"),
                "krylov_operator_storage": last.get("operator_storage"),
                "krylov_rank_mode": last.get("rank_mode", choice["rank_mode"]),
                "krylov_resolved_rank": last.get("resolved_rank", choice["resolved_rank"]),
                "krylov_iteration_budget": last.get("iteration_budget", choice["iteration_budget"]),
                "krylov_iterations": sum(int(entry.get("iterations", 0) or 0) for entry in (stats or [])),
                "krylov_solve_count": len(stats or []),
                "krylov_converged": all(bool(entry.get("converged")) for entry in (stats or [])) if stats else None,
                "krylov_used_direct_fallback": any(bool(entry.get("used_direct_fallback")) for entry in (stats or [])),
            }
        )
    return metadata


def _terminal_voltage(problem: MnaProblem, solution: np.ndarray, node: str | None) -> float:
    if node is None or node == "0":
        return 0.0
    index = problem.index_map.node_to_index.get(node)
    if index is None:
        return 0.0
    return float(np.asarray(solution).flatten()[index])


def _bjt_operating_point(component: ComponentRecord, problem: MnaProblem, solution: np.ndarray) -> dict[str, object]:
    collector = component.node1
    base = component.ctrl_node1
    emitter = component.node2
    vc = _terminal_voltage(problem, solution, collector)
    vb = _terminal_voltage(problem, solution, base)
    ve = _terminal_voltage(problem, solution, emitter)
    gm = parse_value(component.value or "40m")
    r_pi = parse_value(component.value2 or "2.5k")
    r_o = parse_value(component.value3 or "100k")
    vbe = vb - ve
    vce = vc - ve
    ib = vbe / r_pi if r_pi > 0 else 0.0
    ic_ro = vce / r_o if r_o > 0 else 0.0
    ic = gm * vbe + ic_ro
    return {
        "name": component.name,
        "type": component.type,
        "model": "small_signal_hybrid_pi",
        "vc": vc,
        "vb": vb,
        "ve": ve,
        "vbe": vbe,
        "vce": vce,
        "ic": ic,
        "ib": ib,
        "ie": -(ic + ib),
        "gm": gm,
        "rpi": r_pi,
        "ro": r_o,
        "note": "linearized small-signal device, not physical DC bias extraction",
    }


def _bjt_level1_region(component_type: str, vc: float, vb: float, ve: float) -> str:
    if component_type.upper() == "QPNP":
        veb = ve - vb
        vcb = vc - vb
        if veb <= 0.0 and vcb <= 0.0:
            return "cutoff"
        if veb > 0.0 and vcb <= 0.0:
            return "forward_active"
        if veb <= 0.0 and vcb > 0.0:
            return "reverse_active"
        return "saturation"
    vbe = vb - ve
    vbc = vb - vc
    if vbe <= 0.0 and vbc <= 0.0:
        return "cutoff"
    if vbe > 0.0 and vbc <= 0.0:
        return "forward_active"
    if vbe <= 0.0 and vbc > 0.0:
        return "reverse_active"
    return "saturation"


def _bjt_level1_operating_point(component: ComponentRecord, problem: MnaProblem, solution: np.ndarray) -> dict[str, object]:
    collector = component.node1
    base = component.ctrl_node1
    emitter = component.node2
    vc = _terminal_voltage(problem, solution, collector)
    vb = _terminal_voltage(problem, solution, base)
    ve = _terminal_voltage(problem, solution, emitter)
    is_ = parse_value(component.value or "1e-15")
    bf = parse_value(component.value2 or "150")
    br = parse_value(component.value3 or "3")
    vaf = parse_value(component.metadata.get("vaf", "100"))
    var = parse_value(component.metadata.get("var", "25"))
    currents, derivatives = _level1_bjt_currents_and_derivatives(
        {
            "type": component.type,
            "is": is_,
            "bf": bf,
            "br": br,
            "vaf": vaf,
            "var": var,
        },
        vc,
        vb,
        ve,
    )
    ic, ib, ie = currents
    row_c = derivatives[0]
    row_b = derivatives[1]
    gm = row_c[1]
    g_o = row_c[0]
    r_pi = (1.0 / row_b[1]) if abs(row_b[1]) > 1e-30 else np.inf
    return {
        "name": component.name,
        "type": component.type,
        "model": "level1",
        "region": _bjt_level1_region(component.type, vc, vb, ve),
        "vc": vc,
        "vb": vb,
        "ve": ve,
        "vbe": vb - ve,
        "vbc": vb - vc,
        "vce": vc - ve,
        "ic": ic,
        "ib": ib,
        "ie": ie,
        "gm": gm,
        "rpi": r_pi,
        "ro": (1.0 / g_o) if abs(g_o) > 1e-30 else np.inf,
        "is": is_,
        "bf": bf,
        "br": br,
        "vaf": vaf,
        "var": var,
    }


def _mos_small_signal_operating_point(component: ComponentRecord, problem: MnaProblem, solution: np.ndarray) -> dict[str, object]:
    drain = component.node1
    gate = component.ctrl_node1
    source = component.node2
    vd = _terminal_voltage(problem, solution, drain)
    vg = _terminal_voltage(problem, solution, gate)
    vs = _terminal_voltage(problem, solution, source)
    gm = parse_value(component.value or "5m")
    r_o = parse_value(component.value2 or "50k")
    gmb = parse_value(component.metadata.get("gmb", "0"))
    vgs = vg - vs
    vds = vd - vs
    ids = gm * vgs + (vds / r_o if r_o > 0 else 0.0) + gmb * (0.0 - vs)
    return {
        "name": component.name,
        "type": component.type,
        "model": "small_signal",
        "vd": vd,
        "vg": vg,
        "vs": vs,
        "vgs": vgs,
        "vds": vds,
        "ids": ids,
        "gm": gm,
        "ro": r_o,
        "gmb": gmb,
        "note": "linearized small-signal device, not physical DC bias extraction",
    }


def _mos_level1_region(component_type: str, vd: float, vg: float, vs: float, vth: float) -> str:
    if component_type.upper() == "PMOS":
        vgs_eff = vs - vg
        vds_eff = vs - vd
    else:
        vgs_eff = vg - vs
        vds_eff = vd - vs
    if vds_eff < 0:
        return "reverse"
    if vgs_eff <= vth:
        return "off"
    return "triode" if vds_eff < (vgs_eff - vth) else "saturation"


def _mos_level1_operating_point(component: ComponentRecord, problem: MnaProblem, solution: np.ndarray) -> dict[str, object]:
    drain = component.node1
    gate = component.ctrl_node1
    source = component.node2
    vd = _terminal_voltage(problem, solution, drain)
    vg = _terminal_voltage(problem, solution, gate)
    vs = _terminal_voltage(problem, solution, source)
    beta = parse_value(component.value or "1m")
    vth = abs(parse_value(component.value2 or "0.4"))
    lambda_ = parse_value(component.value3 or "0")
    ids, gds, gm, d_source = _level1_mos_current_and_derivatives(
        {
            "type": component.type,
            "beta": beta,
            "vth": vth,
            "lambda": lambda_,
        },
        vd,
        vg,
        vs,
    )
    return {
        "name": component.name,
        "type": component.type,
        "model": "level1",
        "region": _mos_level1_region(component.type, vd, vg, vs, vth),
        "vd": vd,
        "vg": vg,
        "vs": vs,
        "vgs": vg - vs,
        "vds": vd - vs,
        "ids": ids,
        "gm": gm,
        "gds": gds,
        "dIds_dVs": d_source,
        "beta": beta,
        "vth": vth,
        "lambda": lambda_,
    }


def _device_operating_points(problem: MnaProblem, solution: np.ndarray) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for component in problem.components:
        if component.type in {"QNPN", "QPNP"}:
            if component.metadata.get("model") == "level1":
                points.append(_bjt_level1_operating_point(component, problem, solution))
            else:
                points.append(_bjt_operating_point(component, problem, solution))
        elif component.type in {"NMOS", "PMOS"}:
            if component.metadata.get("model") == "level1":
                points.append(_mos_level1_operating_point(component, problem, solution))
            else:
                points.append(_mos_small_signal_operating_point(component, problem, solution))
    return points


def _has_nonlinear_terms(problem: MnaProblem) -> bool:
    solver_f_str = problem.solver_f_str if problem.solver_f_str is not None else problem.f_str
    return is_nonlinear(solver_f_str) or bool(problem.level1_mos_devices) or bool(problem.level1_bjt_devices)


def run_basic_analysis(problem: MnaProblem, options: AnalysisOptions) -> SimulationResult:
    """Run one of the basic analyses on a prebuilt MNA problem."""

    labels = problem.metadata.get("labels", [])
    krylov_stats: list[dict[str, object]] | None = [] if options.use_krylov else None
    linear_kwargs = _linear_solver_kwargs(options, krylov_stats)
    solver_f_str = problem.solver_f_str if problem.solver_f_str is not None else problem.f_str
    has_level1_devices = bool(problem.level1_mos_devices or problem.level1_bjt_devices)

    if options.mode == AnalysisMode.SHOW_MATRIX:
        metadata = _result_metadata(problem, options, krylov_stats)
        if options.use_mor:
            from .mor.metadata import disabled_metadata

            metadata.update(disabled_metadata(".show_matrix requires the full MNA system.", options.mor_method))
        return SimulationResult(
            mode=options.mode,
            status="ok",
            matrices={
                "G": problem.G,
                "C": problem.C,
                "f_str": problem.f_str,
                "b_dc": problem.b_dc,
                "b_ac": problem.b_ac,
                "b_time_str": problem.b_time_str,
            },
            labels=labels,
            metadata=metadata,
        )

    if options.mode == AnalysisMode.OP:
        init_cond = options.init_condition if options.init_condition is not None else "DEFAULT"
        solution = solve_dc_nr(
            problem.G,
            solver_f_str,
            problem.b_dc,
            max_iter=options.max_iter,
            v_tol=options.v_tol,
            f_tol=options.f_tol,
            init_cond=init_cond,
            level1_mos_devices=problem.level1_mos_devices,
            level1_bjt_devices=problem.level1_bjt_devices,
            **linear_kwargs,
        )
        metadata = _result_metadata(problem, options, krylov_stats)
        if options.use_mor:
            from .mor.metadata import disabled_metadata

            metadata.update(disabled_metadata(".op solves and reports the full state in MOR v1.", options.mor_method))
        device_points = _device_operating_points(problem, solution)
        if device_points:
            metadata["device_operating_points"] = device_points
        return SimulationResult(
            mode=options.mode,
            status="ok",
            dc_solution=solution,
            matrices={
                "G": problem.G,
                "C": problem.C,
                "f_str": problem.f_str,
                "b_dc": problem.b_dc,
                "b_ac": problem.b_ac,
                "b_time_str": problem.b_time_str,
            },
            labels=labels,
            metadata=metadata,
        )

    if options.mode == AnalysisMode.TRAN:
        if options.use_mor:
            if _has_nonlinear_terms(problem):
                if options.mor_method == "linear_krylov":
                    error_handler("MOR_CONFIG: linear_krylov MOR cannot be used for nonlinear transient; choose auto or tpwl.")
                from .mor.tpwl import solve_transient_tpwl

                waveform, mor_metadata = solve_transient_tpwl(
                    problem,
                    t_stop=options.tran_stop or 0.1,
                    t_step=options.tran_step or 1e-5,
                    mor_output_nodes=options.mor_output_nodes,
                    mor_order=options.mor_order,
                    requested_method=options.mor_method,
                    validate=options.mor_validate,
                    probe_nodes=options.probe_nodes,
                    max_iter=options.max_iter,
                    v_tol=options.v_tol,
                    f_tol=options.f_tol,
                )
                metadata = _result_metadata(problem, options, krylov_stats)
                metadata.update(mor_metadata)
                return SimulationResult(
                    mode=options.mode,
                    status="ok",
                    waveform=waveform,
                    labels=waveform.labels,
                    metadata=metadata,
                )
            from .mor.linear import solve_transient_mor

            init_cond = options.init_condition if not isinstance(options.init_condition, str) else None
            waveform, mor_metadata = solve_transient_mor(
                problem,
                t_stop=options.tran_stop or 0.1,
                t_step=options.tran_step or 1e-5,
                mor_output_nodes=options.mor_output_nodes,
                mor_order=options.mor_order,
                requested_method=options.mor_method,
                validate=options.mor_validate,
                probe_nodes=options.probe_nodes,
                max_iter=options.max_iter,
                v_tol=options.v_tol,
                f_tol=options.f_tol,
                init_cond=init_cond,
                linear_kwargs=linear_kwargs,
            )
            if options.mor_method == "tpwl":
                mor_metadata["mor_fallback_reason"] = "TPWL/POD is nonlinear transient MOR; linear circuits route to Linear Krylov MOR."
            metadata = _result_metadata(problem, options, krylov_stats)
            metadata.update(mor_metadata)
            metadata["op_used"] = False
            return SimulationResult(
                mode=options.mode,
                status="ok",
                waveform=waveform,
                labels=waveform.labels,
                metadata=metadata,
            )

        init_cond = options.init_condition if not isinstance(options.init_condition, str) else None
        op_used = False
        op_solution: np.ndarray | None = None
        if init_cond is None and has_level1_devices:
            b_time_func = compile_time_source_func(problem.b_time_str)
            op_solution = solve_dc_nr(
                problem.G,
                solver_f_str,
                problem.b_dc + b_time_func(0.0),
                max_iter=options.max_iter,
                v_tol=options.v_tol,
                f_tol=options.f_tol,
                init_cond="DEFAULT",
                level1_mos_devices=problem.level1_mos_devices,
                level1_bjt_devices=problem.level1_bjt_devices,
                **linear_kwargs,
            )
            init_cond = op_solution
            op_used = True
        waveform = solve_transient(
            problem,
            t_stop=options.tran_stop or 0.1,
            t_step=options.tran_step or 1e-5,
            max_iter=options.max_iter,
            v_tol=options.v_tol,
            f_tol=options.f_tol,
            init_cond=init_cond,
            **linear_kwargs,
        )
        metadata = _result_metadata(problem, options, krylov_stats)
        metadata["op_used"] = op_used
        if op_solution is not None:
            metadata["operating_point"] = {
                "labels": labels,
                "values": np.asarray(op_solution).flatten().tolist(),
            }
            metadata["device_operating_points"] = _device_operating_points(problem, op_solution)
        return SimulationResult(
            mode=options.mode,
            status="ok",
            waveform=waveform,
            labels=waveform.labels,
            metadata=metadata,
        )

    if options.mode == AnalysisMode.AC:
        if options.use_mor:
            if options.mor_method == "tpwl":
                error_handler("MOR_CONFIG: tpwl MOR is transient-only; use auto or linear_krylov for .ac.")
            from .mor.linear import solve_ac_mor

            spectrum = solve_ac_mor(
                problem,
                f_start=options.ac_start or 1.0,
                f_stop=options.ac_stop or 1e6,
                points=options.ac_points or 100,
                mor_output_nodes=options.mor_output_nodes,
                mor_order=options.mor_order,
                requested_method=options.mor_method,
                validate=options.mor_validate,
                probe_nodes=options.probe_nodes,
                max_iter=options.max_iter,
                v_tol=options.v_tol,
                f_tol=options.f_tol,
                linear_kwargs=linear_kwargs,
            )
            metadata = _result_metadata(problem, options, krylov_stats)
            metadata.update(spectrum.metadata)
            operating_point = spectrum.metadata.get("operating_point")
            if isinstance(operating_point, dict) and operating_point.get("values") is not None:
                op_solution = np.asarray(operating_point["values"], dtype=float).reshape(-1, 1)
                device_points = _device_operating_points(problem, op_solution)
                if device_points:
                    metadata["device_operating_points"] = device_points
            return SimulationResult(
                mode=options.mode,
                status="ok",
                spectrum=spectrum,
                labels=spectrum.labels,
                metadata=metadata,
            )

        spectrum = solve_ac(
            problem,
            f_start=options.ac_start or 1.0,
            f_stop=options.ac_stop or 1e6,
            points=options.ac_points or 100,
            **linear_kwargs,
        )
        metadata = _result_metadata(problem, options, krylov_stats)
        metadata.update(spectrum.metadata)
        operating_point = spectrum.metadata.get("operating_point")
        if isinstance(operating_point, dict) and operating_point.get("values") is not None:
            op_solution = np.asarray(operating_point["values"], dtype=float).reshape(-1, 1)
            device_points = _device_operating_points(problem, op_solution)
            if device_points:
                metadata["device_operating_points"] = device_points
        return SimulationResult(
            mode=options.mode,
            status="ok",
            spectrum=spectrum,
            labels=spectrum.labels,
            metadata=metadata,
        )

    raise ValueError(f"Unsupported basic analysis mode: {options.mode}")


def build_and_run_basic(circuit, options: AnalysisOptions, gmin: float = 0.0) -> SimulationResult:
    """Construct the MNA problem and execute a basic analysis."""

    problem = build_mna_problem(circuit, gmin=gmin)
    return run_basic_analysis(problem, options)
