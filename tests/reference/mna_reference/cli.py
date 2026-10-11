"""CLI entry point for file-based and inline netlist runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api.contracts import AnalysisMode, AnalysisRequest
from .api.service import SimulationService
from .backends.python_backend import PythonBackend
from .errors import CircuitSimulatorError


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(prog="mna-sim", description="Run the MNA circuit simulator.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a simulation from a netlist file.")
    run_parser.add_argument("path", type=Path, help="Path to a .cir or .sp netlist.")
    run_parser.add_argument(
        "--mode",
        choices=[mode.value for mode in AnalysisMode],
        default=None,
        help="Override the analysis mode.",
    )
    run_parser.add_argument("--json", action="store_true", help="Emit JSON output.")

    text_parser = subparsers.add_parser("run-text", help="Run a simulation from inline netlist text.")
    text_parser.add_argument("netlist_text", help="Raw netlist text.")
    text_parser.add_argument(
        "--mode",
        choices=[mode.value for mode in AnalysisMode],
        default=None,
        help="Override the analysis mode.",
    )
    text_parser.add_argument("--json", action="store_true", help="Emit JSON output.")

    return parser


def _print_response(response, as_json: bool) -> None:
    if as_json:
        print(json.dumps(response.model_dump(), indent=2))
        return
    print(f"Mode: {response.mode}")
    print(f"Status: {response.status}")
    if response.diagnostics:
        print("Diagnostics:")
        for line in response.diagnostics:
            print(f"  - {line}")
    if response.dc_solution is not None:
        print("DC Solution:")
        print(response.dc_solution)
    if response.waveform is not None:
        print("Waveform labels:", response.waveform["labels"])
        print("Time samples:", len(response.waveform["time"]))
    if response.spectrum is not None:
        print("Spectrum labels:", response.spectrum["labels"])
        print("Frequency points:", len(response.spectrum["frequencies"]))
    if response.matrices is not None:
        print("Matrices available:", ", ".join(response.matrices.keys()))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)
    backend = PythonBackend()
    service = SimulationService(backend)

    try:
        if args.command == "run":
            text = args.path.read_text(encoding="utf-8")
        else:
            text = args.netlist_text

        mode = AnalysisMode(args.mode) if args.mode else None
        response = service.run_request(AnalysisRequest(netlist_text=text, mode=mode))
        _print_response(response, as_json=args.json)
        return 0
    except CircuitSimulatorError as exc:
        print(f"{exc.code}: {exc.message}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
