"""Independent NumPy reference of the model-card device equations (phase M2).

Written from the SPICE3f5 equations (``dioload.c``/``diotemp.c``, ``bjtload.c``,
``mos1load.c``) and the choices recorded in ``HIERARCHY_AND_MODELS_PLAN.md`` 3.2 / 3.4, not
from the C++ kernels: every function evaluates arrays of bias points and returns the
terminal currents with their Jacobian, obtained by the chain rule through the junction /
channel variables rather than in the kernels' operation order. The tests in this package
check the Jacobians by central finite differences and compare the native kernels
(``mna_simulation._core``) against these functions.
"""

from .common import EXP_CLIP, VT, limexp
from .gummel_poon import GummelPoonCard, gummel_poon
from .mos_card import mos_card, threshold
from .spice_diode import breakdown_voltage, spice_diode

__all__ = [
    "EXP_CLIP",
    "GummelPoonCard",
    "VT",
    "breakdown_voltage",
    "gummel_poon",
    "limexp",
    "mos_card",
    "spice_diode",
    "threshold",
]
