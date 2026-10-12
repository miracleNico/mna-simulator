"""FastAPI application for the simulator service."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from mna_simulation.api.contracts import (
    AnalysisMode,
    AnalysisRequest,
    SchematicAnalysis,
    SchematicDocument,
    SimulationResponse,
)
from mna_simulation.api.service import create_service, is_invalid_option_error
from mna_simulation.errors import CircuitSimulatorError
from mna_simulation.native.threads import budget_total
from mna_simulation.schematic_pipeline import schematic_to_circuit_ir

_EXECUTOR_LOCK = threading.Lock()


def _new_executor() -> ThreadPoolExecutor:
    # One worker per budget thread: requests beyond that queue here, and each running
    # request additionally holds a lease from the service's thread budget.
    return ThreadPoolExecutor(max_workers=budget_total(), thread_name_prefix="mna-sim")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    health = service.healthcheck()
    logging.getLogger("uvicorn.error").info("MNA core: %s (%s)", health["core_backend"], health["core_version"])
    app.state.executor = _new_executor()
    try:
        yield
    finally:
        executor = app.state.executor
        app.state.executor = None
        executor.shutdown(wait=True, cancel_futures=True)


app = FastAPI(title="MNA Simulation Service", version="0.70.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# MNA_CORE selects the numerical core: "cpp" (default, the native core) or "python" (the frozen
# pre-migration pure-Python core, source checkouts only; mna_simulation/api/reference_service.py).
service = create_service()


def _executor() -> ThreadPoolExecutor:
    """The simulation executor (created lazily when the lifespan did not run)."""

    executor = getattr(app.state, "executor", None)
    if executor is None:
        with _EXECUTOR_LOCK:
            executor = getattr(app.state, "executor", None)
            if executor is None:
                executor = app.state.executor = _new_executor()
    return executor


async def _run_simulation(request: AnalysisRequest) -> SimulationResponse:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor(), service.run_request, request)


@app.get("/health")
def health() -> dict[str, object]:
    """Return service health, backend capabilities, core version, and thread budget."""

    return service.healthcheck()


@app.post("/api/simulate", response_model=SimulationResponse)
async def simulate(request: AnalysisRequest) -> SimulationResponse:
    """Run a simulation request on the bounded simulation executor."""

    try:
        return await _run_simulation(request)
    except CircuitSimulatorError as exc:
        raise HTTPException(status_code=400, detail={"code": exc.code, "message": exc.message}) from exc
    except ValueError as exc:
        if not is_invalid_option_error(exc):
            raise
        raise HTTPException(status_code=400, detail={"code": "INVALID_OPTIONS", "message": str(exc)}) from exc


def _resolve_pin_refs_to_labels(
    schematic: SchematicDocument,
    pin_refs: list[dict],
    all_labels: list[str],
) -> tuple[list[int], list[str]]:
    """Translate ``pin_refs`` (component_id + pin) into label indices.

    Uses the schematic pipeline's net_assignments to map each pin to a net name,
    then finds the matching ``V(<net>)`` / ``I(<name>)`` labels. If a ref cannot
    be resolved, it is silently skipped. If nothing resolves, returns all labels
    so the client always has something to display.
    """

    if not pin_refs:
        return list(range(len(all_labels))), list(all_labels)

    try:
        converted = schematic_to_circuit_ir(schematic=schematic)
    except CircuitSimulatorError:
        return list(range(len(all_labels))), list(all_labels)

    net_map = converted.net_assignments  # pin_key -> net name (e.g. "n1", "0")
    wanted_nets: list[str] = []
    for ref in pin_refs:
        cid = ref.get("component_id")
        pin = ref.get("pin")
        if not cid or not pin:
            continue
        key = f"cp:{cid}:{pin}"
        net = net_map.get(key)
        if net and net != "0":
            wanted_nets.append(net)

    idxs: list[int] = []
    kept: list[str] = []
    for i, lbl in enumerate(all_labels):
        for n in wanted_nets:
            if lbl == f"V({n})" or lbl.endswith(f"({n})"):
                idxs.append(i)
                kept.append(lbl)
                break
    if not idxs:
        return list(range(len(all_labels))), list(all_labels)
    return idxs, kept


@app.websocket("/ws/dyn")
async def dyn_stream(ws: WebSocket) -> None:
    """Realtime ``.dyn`` streaming endpoint.

    Protocol (JSON messages):
        client -> {
            "schematic": SchematicDocument,
            "speed": <s_per_s>,                    # sim seconds per wall-clock second
            "t_stop": "10m",
            "t_step": "0.1m",
            "pin_refs": [{"component_id": "...", "pin": "p"}, ...],  # optional
            "nodes": ["n1", ...],                  # optional, legacy label filter
            "continuous": false,                   # optional, loop playback forever
            "use_krylov": false,                   # optional, auto Krylov linear solve
            "num_threads": "auto",                 # optional, "auto" or a positive integer
            "max_frame_rate": 60                   # optional display stream cap
        }
        server -> {"type": "meta", "labels": [...], "t_stop": <float>, "speed": <float>, "sim_span": <float>, "continuous": <bool>,
                   "threads_used": <int>, "elapsed_ms": <float>}
        server -> {"type": "frame", "t": <float>, "values": [...] }   (many)
        server -> {"type": "loop"}     # emitted between playback loops when continuous
        server -> {"type": "done"}
    """

    await ws.accept()
    try:
        raw = await ws.receive_text()
        payload = json.loads(raw)
        schematic_dict = payload.get("schematic") or {}
        if not schematic_dict:
            await ws.send_text(json.dumps({"type": "error", "message": "missing schematic"}))
            await ws.close()
            return
        schematic = SchematicDocument(**schematic_dict)
        speed = float(payload.get("speed", 1e-3))
        t_stop = payload.get("t_stop", "10m")
        t_step = payload.get("t_step", "0.1m")
        pin_refs = payload.get("pin_refs") or []
        nodes_filter = payload.get("nodes") or []
        continuous = bool(payload.get("continuous", False))
        use_krylov = bool(payload.get("use_krylov", False))
        krylov_rank = payload.get("krylov_rank", "auto")
        krylov_method = payload.get("krylov_method", "auto")
        num_threads = payload.get("num_threads")
        max_frame_rate = max(1.0, float(payload.get("max_frame_rate", 60.0)))

        # Force the schematic to run as .tran with the requested params. Writing
        # them into schematic.analysis (not options) means they hit the netlist
        # emitter as text, then the netlist parser converts to floats — avoiding
        # the "arange on StrDType" trap where string overrides clobber the parsed
        # directive's floats inside build_options.
        schematic.analysis = SchematicAnalysis(
            mode=AnalysisMode.TRAN,
            params={"t_stop": t_stop, "t_step": t_step},
        )
        options: dict[str, object] = {
            "use_krylov": use_krylov,
            "krylov_rank": krylov_rank,
            "krylov_method": krylov_method,
            "output_max_points": 6000,
        }
        if num_threads is not None:
            options["num_threads"] = num_threads
        request = AnalysisRequest(
            netlist_text="",
            mode=AnalysisMode.TRAN,
            options=options,
            schematic=schematic,
        )
        try:
            response = await _run_simulation(request)
        except CircuitSimulatorError as exc:
            await ws.send_text(json.dumps({"type": "error", "message": exc.message}))
            await ws.close()
            return
        except ValueError as exc:
            if not is_invalid_option_error(exc):
                raise
            await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
            await ws.close()
            return

        waveform = response.waveform
        if waveform is None:
            await ws.send_text(json.dumps({"type": "error", "message": "no waveform produced"}))
            await ws.close()
            return

        times = waveform["time"]
        values = waveform["values"]
        labels = waveform["labels"]

        # Detect shape: solver stores values[:, step] so nested list is
        # (num_labels, num_time_points). core_advanced may return the transpose,
        # so auto-detect from the dimensions.
        num_labels = len(labels)
        num_time = len(times)
        inner_len = len(values[0]) if values and isinstance(values[0], list) else 0
        is_label_major = len(values) == num_labels and inner_len == num_time
        is_time_major = len(values) == num_time and inner_len == num_labels
        if not (is_label_major or is_time_major):
            # Heuristic: if outer dim matches num_time, treat as time-major.
            is_time_major = len(values) == num_time

        if pin_refs:
            idxs, kept_labels = _resolve_pin_refs_to_labels(schematic, pin_refs, labels)
        elif nodes_filter:
            idxs = [i for i, lbl in enumerate(labels) if lbl in nodes_filter]
            kept_labels = [labels[i] for i in idxs]
            if not idxs:
                idxs = list(range(len(labels)))
                kept_labels = list(labels)
        else:
            idxs = list(range(len(labels)))
            kept_labels = list(labels)

        total_sim_span = (float(times[-1]) - float(times[0])) if times else 0.0

        await ws.send_text(
            json.dumps(
                {
                    "type": "meta",
                    "labels": kept_labels,
                    "t_stop": float(times[-1]) if times else 0.0,
                    "sim_span": total_sim_span,
                    "speed": speed,
                    "continuous": continuous,
                    "threads_used": response.metadata.get("threads_used"),
                    "elapsed_ms": response.metadata.get("elapsed_ms"),
                }
            )
        )

        effective_speed = speed if speed > 0 else 1e-3
        min_frame_interval = 1.0 / max_frame_rate

        async def stream_once(loop_wall0: float, sim_shift: float) -> bool:
            """Send one full playback. Returns False if client disconnected."""
            t0_sim_local = float(times[0]) if times else 0.0
            last_sent_wall_offset = -float("inf")
            for step_idx, t_sim in enumerate(times):
                wall_offset = (float(t_sim) - t0_sim_local) / effective_speed
                is_last_sample = step_idx == len(times) - 1
                if wall_offset - last_sent_wall_offset < min_frame_interval and not is_last_sample:
                    continue
                target_wall = loop_wall0 + wall_offset
                now = time.monotonic()
                if target_wall > now:
                    try:
                        await asyncio.sleep(target_wall - now)
                    except asyncio.CancelledError:
                        return False
                if is_time_major:
                    row = values[step_idx] if step_idx < len(values) else []
                    filtered = [float(row[i]) if i < len(row) else 0.0 for i in idxs]
                else:  # label-major (solver default: values[:, step])
                    filtered = [
                        float(values[lbl_idx][step_idx])
                        if lbl_idx < len(values) and step_idx < len(values[lbl_idx])
                        else 0.0
                        for lbl_idx in idxs
                    ]
                try:
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "frame",
                                "t": float(t_sim) + sim_shift,
                                "values": filtered,
                            }
                        )
                    )
                except WebSocketDisconnect:
                    return False
                except RuntimeError:
                    return False
                last_sent_wall_offset = wall_offset
            return True

        t0_wall = time.monotonic()
        if continuous:
            sim_shift = 0.0
            loop_idx = 0
            while True:
                loop_start_wall = t0_wall + (total_sim_span * loop_idx) / effective_speed
                ok = await stream_once(loop_start_wall, sim_shift)
                if not ok:
                    return
                try:
                    await ws.send_text(json.dumps({"type": "loop"}))
                except (WebSocketDisconnect, RuntimeError):
                    return
                sim_shift += total_sim_span
                loop_idx += 1
        else:
            ok = await stream_once(t0_wall, 0.0)
            if not ok:
                return
            try:
                await ws.send_text(json.dumps({"type": "done"}))
            except (WebSocketDisconnect, RuntimeError):
                return
    except WebSocketDisconnect:
        return
    except Exception as exc:  # noqa: BLE001
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:  # noqa: BLE001
            pass
    finally:
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass
