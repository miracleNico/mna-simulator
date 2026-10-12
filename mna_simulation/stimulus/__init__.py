"""Python-defined digital stimulus for the analog core (MIXED_SIGNAL.md phase D1).

:class:`DigitalSignal` turns bits, events or raw points into the corner points of a
piecewise-linear source (``PWL`` netlist line or ``stimuli`` request entry);
:mod:`mna_simulation.stimulus.usb` builds USB low-/full-speed packets on D+/D-.
Pure Python and NumPy: nothing here imports the simulator core.
"""

from __future__ import annotations

from . import usb
from .signal import DigitalSignal

__all__ = ["DigitalSignal", "usb"]
