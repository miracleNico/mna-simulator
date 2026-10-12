"""SPICE-dialect support of the netlist parser.

* ``values``: SPICE value literals (``T``, ``MIL``, ``MEG``, unit words) and token helpers.
* ``expressions``: the restricted evaluator of ``.param`` values and ``{...}`` expressions.
* ``model_cards``: ``.model`` card parsing and the supported parameter lists.
* ``sources``: SPICE source forms (``DC``/``AC``, ``SIN``, ``PULSE``, ``PWL``, ``EXP``) and
  their waveform expressions.
* ``context``: per-parse state (diagnostics, model lookup, locations).
* ``frontend``: lexing, ``.include``/``.lib``, ``.subckt``/``X`` flattening, ``.param``
  scopes (imported lazily by ``netlist.parse_netlist_text``).
"""
