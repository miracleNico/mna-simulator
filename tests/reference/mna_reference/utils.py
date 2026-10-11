"""Shared utility helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np

from .errors import error_handler

VALUE_PATTERN = re.compile(
    r"([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)_?([a-zA-Z%]*)",
    re.IGNORECASE,
)

SUFFIX_FACTORS = {
    "G": 1e9,
    "MEG": 1e6,
    "K": 1e3,
    "M": 1e-3,
    "U": 1e-6,
    "N": 1e-9,
    "P": 1e-12,
    "F": 1e-15,
    "%": 1e-2,
    "": 1.0,
}

DEFAULT_GMIN_STEPS = [0.0, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]


def parse_value(value: str | float | int) -> float:
    """Convert metric-suffixed strings to floating point values."""

    if isinstance(value, (int, float)):
        return float(value)

    match = VALUE_PATTERN.fullmatch(value.strip())
    if not match:
        error_handler(f"PARSE_ERROR: Could not parse value: {value}")

    assert match is not None
    number_part = match.group(1)
    suffix = match.group(5).upper()

    if suffix not in SUFFIX_FACTORS:
        error_handler(f"PARSE_ERROR: Unsupported suffix '{suffix}' in value '{value}'.")

    return float(number_part) * SUFFIX_FACTORS[suffix]


def load_text_file(path: str | Path) -> str:
    """Read a UTF-8 text file."""

    return Path(path).read_text(encoding="utf-8")


def ensure_iterable_array(values: Iterable[float] | np.ndarray) -> np.ndarray:
    """Convert an iterable to a float ndarray."""

    if isinstance(values, np.ndarray):
        return values.astype(float, copy=False)
    return np.asarray(list(values), dtype=float)


def format_voltage_labels(node_names: list[str], branch_names: list[str]) -> list[str]:
    """Generate human-friendly labels for result arrays."""

    node_labels = [f"V({name})" for name in node_names]
    branch_labels = [f"I({name})" for name in branch_names]
    return node_labels + branch_labels


def ndarray_to_list(array: np.ndarray | None) -> object:
    """Convert ndarrays to JSON-safe lists."""

    if array is None:
        return None
    if np.iscomplexobj(array):
        if np.allclose(np.imag(array), 0.0):
            return np.real(array).tolist()
        return np.vectorize(lambda value: {"real": float(np.real(value)), "imag": float(np.imag(value))})(array).tolist()
    return array.tolist()
