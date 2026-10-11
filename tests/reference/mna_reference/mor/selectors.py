"""Output selector utilities for model-order reduction."""

from __future__ import annotations

import numpy as np

from ..errors import error_handler


def _label_aliases(label: str) -> tuple[str, ...]:
    cleaned = str(label).strip()
    if cleaned.startswith("V(") or cleaned.startswith("I("):
        return (cleaned,)
    return (cleaned, f"V({cleaned})", f"I({cleaned})")


def normalize_output_label(label: str, available_labels: list[str]) -> str:
    """Resolve a requested output label against full MNA labels."""

    available = set(available_labels)
    for candidate in _label_aliases(label):
        if candidate in available:
            return candidate
    error_handler(
        "MOR_CONFIG: MOR output label "
        f"'{label}' is not present in simulation labels ({', '.join(available_labels)})."
    )
    raise RuntimeError("unreachable")


def resolve_output_labels(available_labels: list[str], requested: list[str]) -> list[str]:
    """Normalize and de-duplicate requested MOR labels."""

    if not requested:
        error_handler("MOR_CONFIG: MOR requires at least one output node, e.g. V(out) or I(L1).")
    outputs: list[str] = []
    seen: set[str] = set()
    for label in requested:
        normalized = normalize_output_label(label, available_labels)
        if normalized in seen:
            continue
        seen.add(normalized)
        outputs.append(normalized)
    return outputs


def build_output_selector(available_labels: list[str], outputs: list[str]) -> tuple[np.ndarray, list[str]]:
    """Build ``S`` so selected outputs are ``y = S x``."""

    normalized = resolve_output_labels(available_labels, outputs)
    label_to_index = {label: index for index, label in enumerate(available_labels)}
    selector = np.zeros((len(normalized), len(available_labels)), dtype=float)
    for row, label in enumerate(normalized):
        selector[row, label_to_index[label]] = 1.0
    return selector, normalized


def validate_probe_subset(available_labels: list[str], probe_nodes: list[str], mor_outputs: list[str]) -> None:
    """Ensure display/probe labels are contained in the reduced output set."""

    if not probe_nodes:
        return
    output_set = set(resolve_output_labels(available_labels, mor_outputs))
    for probe in probe_nodes:
        normalized = normalize_output_label(probe, available_labels)
        if normalized not in output_set:
            error_handler(
                "MOR_CONFIG: Display nodes must be a subset of MOR output nodes when MOR is enabled "
                f"('{normalized}' is not in {sorted(output_set)})."
            )
