"""Netlist generators for the representative performance workloads.

The SRAM and RLC generators mirror the ones used by the regression tests
(``tests/test_physical_sram.py`` and ``tests/test_krylov_solvers.py``) so the
benchmark numbers describe the same circuits the UI demos and tests exercise.
"""

from __future__ import annotations


def sram_netlist(rows: int = 10, cols: int = 10, step: str = "0.25n", selected_row: int = 3, selected_col: int = 4) -> str:
    """Hierarchy-flattened 6T Level-1 MOS SRAM array with write/hold/read drivers."""

    v_high = "1.2"
    lines = [
        "GND gnd",
        f"VDD vdd gnd DC {v_high}",
        f"VPCH pch gnd FUNC 0 Piecewise((0, t < 1e-9), ({v_high}, t < 3.5e-9), (0, t < 4.5e-9), ({v_high}, True))",
        f"VWR wr gnd FUNC 0 Piecewise((0, t < 1e-9), ({v_high}, t < 2.5e-9), (0, True))",
        f"VWRB wr_b gnd FUNC 0 Piecewise(({v_high}, t < 1e-9), (0, t < 2.5e-9), ({v_high}, True))",
    ]
    for row in range(rows):
        if row == selected_row:
            lines.append(
                f"VWL{row} wl_{row} gnd FUNC 0 Piecewise((0, t < 1e-9), ({v_high}, t < 2.5e-9), "
                f"(0, t < 4.5e-9), ({v_high}, t < 6e-9), (0, True))"
            )
        else:
            lines.append(f"VWL{row} wl_{row} gnd DC 0")
    for col in range(cols):
        lines.extend(
            [
                f"MPCHBL{col} bl_{col} pch vdd PMOS LEVEL1 3m 0.4 0.02 2f 1f",
                f"MPCHBLB{col} blb_{col} pch vdd PMOS LEVEL1 3m 0.4 0.02 2f 1f",
                f"CBL{col} bl_{col} gnd 20f",
                f"CBLB{col} blb_{col} gnd 20f",
            ]
        )
    lines.extend(
        [
            f"MWRBL bl_{selected_col} wr_b vdd PMOS LEVEL1 8m 0.4 0.02 2f 1f",
            f"MWRBLB blb_{selected_col} wr gnd NMOS LEVEL1 8m 0.4 0.02 2f 1f",
        ]
    )
    for row in range(rows):
        for col in range(cols):
            q = f"q_{row}_{col}"
            qb = f"qb_{row}_{col}"
            q_starts_high = False if (row == selected_row and col == selected_col) else (row + col) % 2 == 0
            lines.extend(
                [
                    f"MPQ{row}_{col} {q} {qb} vdd PMOS LEVEL1 0.7m 0.4 0.02 2f 1f",
                    f"MPQB{row}_{col} {qb} {q} vdd PMOS LEVEL1 0.7m 0.4 0.02 2f 1f",
                    f"MNQ{row}_{col} {q} {qb} gnd NMOS LEVEL1 1.8m 0.4 0.02 2f 1f",
                    f"MNQB{row}_{col} {qb} {q} gnd NMOS LEVEL1 1.8m 0.4 0.02 2f 1f",
                    f"MAXQ{row}_{col} {q} wl_{row} bl_{col} NMOS LEVEL1 2.5m 0.4 0.02 2f 1f",
                    f"MAXQB{row}_{col} {qb} wl_{row} blb_{col} NMOS LEVEL1 2.5m 0.4 0.02 2f 1f",
                    f"CQ{row}_{col} {q} gnd 4f",
                    f"CQB{row}_{col} {qb} gnd 4f",
                    f"RINITQ{row}_{col} {q} {'vdd' if q_starts_high else 'gnd'} 200Meg",
                    f"RINITQB{row}_{col} {qb} {'gnd' if q_starts_high else 'vdd'} 200Meg",
                ]
            )
    lines.extend([f".tran 6n {step}", ".end"])
    return "\n".join(lines)


def rlc_mesh_netlist(rows: int = 23, cols: int = 23) -> str:
    """Distributed RLC mesh; 23x23 gives 1035 MNA unknowns."""

    lines = ["IIN n_0_0 0 STEP 1m 20n"]
    for row in range(rows):
        for col in range(cols):
            node = f"n_{row}_{col}"
            lines.append(f"C_{row}_{col} {node} 0 200p")
            lines.append(f"RG_{row}_{col} {node} 0 50Meg")
            if col + 1 < cols:
                lines.append(f"R_H_{row}_{col} {node} n_{row}_{col + 1} 25")
            if row + 1 < rows:
                lines.append(f"L_V_{row}_{col} {node} n_{row + 1}_{col} 20n")
    lines.append(f"RLOAD n_{rows - 1}_{cols - 1} 0 50")
    lines.append(".tran 200n 10n")
    lines.append(".end")
    return "\n".join(lines)


BJT_THREE_STAGE = """
VCC vcc 0 DC 15
Vin vin 0 SIN 0.001 1k
Cin vin inp 1u
Cref inn 0 1u
RbpT vcc inp 150k
RbpB inp 0 20k
RbnT vcc inn 150k
RbnB inn 0 20k
Qd1 c1 inp etail QNPN LEVEL1 1e-15 150 3 100 25 4p 2p 50 0.5 5
Qd2 dout inn etail QNPN LEVEL1 1e-15 150 3 100 25 4p 2p 50 0.5 5
Rc1 vcc c1 5.6k
Rc2 vcc dout 5.6k
Rtail etail 0 2.2k
Cd dout midin 1u
RmT vcc midin 150k
RmB midin 0 20k
Qm mout midin me QNPN LEVEL1 1e-15 150 3 100 25 4p 2p 50 0.5 5
Rcm vcc mout 12k
Rem me 0 1.5k
Cmo mout fbase 1u
RfT vcc fbase 68k
RfB fbase 0 75k
Qo vcc fbase out QNPN LEVEL1 1e-15 150 3 100 25 4p 2p 50 0.5 5
Reo out 0 1.2k
Rload out 0 10k
.tran 10m 5u
.end
"""

HB_CLOSE_TONE_MIXER = """
IBIAS mix 0 DC 50u
I1 mix 0 SIN 40u 1G
I2 mix 0 SIN 40u 1.005G
RLOAD mix 0 10k
D1 mix 0 1e-12
GENV env 0 mix 0 1m
LENV env 0 1u
CENV env 0 1.013n
RQ env 0 4.3k
.hb 205
.end
"""
