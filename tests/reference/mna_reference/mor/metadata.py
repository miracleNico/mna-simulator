"""Metadata and option helpers for model-order reduction."""

from __future__ import annotations

import numpy as np

from ..errors import error_handler


def resolve_mor_order(
    matrix_dimension: int,
    num_inputs: int,
    num_outputs: int,
    requested_order: int | str | None = "auto",
) -> tuple[int, str]:
    """Resolve MOR order from UI/API options."""

    n = max(1, int(matrix_dimension))
    if requested_order is None:
        requested_order = "auto"
    if isinstance(requested_order, str):
        cleaned = requested_order.strip().lower()
        if cleaned == "auto" or cleaned == "":
            count = max(1, int(num_inputs) + int(num_outputs))
            return min(n, max(10, min(120, 4 * count))), "auto"
        requested_order = cleaned
    order = int(requested_order)
    if order < 1:
        error_handler("MOR_CONFIG: mor_order must be 'auto' or a positive integer.")
    return min(n, order), "manual"


def compression_ratio(original_dimension: int, reduced_dimension: int) -> float:
    if reduced_dimension <= 0:
        return 0.0
    return float(original_dimension / reduced_dimension)


def disabled_metadata(reason: str, requested_method: str = "auto") -> dict[str, object]:
    return {
        "mor_requested": True,
        "mor_used": False,
        "mor_requested_method": requested_method,
        "mor_fallback_reason": reason,
    }


def used_metadata(
    *,
    method: str,
    requested_method: str,
    original_dimension: int,
    reduced_dimension: int,
    resolved_order: int | None = None,
    order_mode: str,
    output_labels: list[str],
    validate: bool,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    order_budget = int(resolved_order if resolved_order is not None else reduced_dimension)
    metadata: dict[str, object] = {
        "mor_requested": True,
        "mor_used": True,
        "mor_method": method,
        "mor_requested_method": requested_method,
        "mor_order": order_budget,
        "mor_resolved_order": order_budget,
        "mor_order_mode": order_mode,
        "mor_basis_size": int(reduced_dimension),
        "mor_original_dimension": int(original_dimension),
        "mor_reduced_dimension": int(reduced_dimension),
        "mor_compression_ratio": compression_ratio(original_dimension, reduced_dimension),
        "mor_output_nodes": output_labels,
        "mor_validate": bool(validate),
    }
    if extra:
        metadata.update(extra)
    return metadata


def matrix_nnz(matrix: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(matrix)))
