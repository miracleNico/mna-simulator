### Python reference vs C++ core (2026-10-11T22:28:13+00:00)

- Host: `vm`, Linux-6.18.44-fc-v114-x86_64-with-glibc2.39, 4 CPUs
- Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1
- Native core: 0.70.0-dev (gcc 13.3.0, Release)
- Commit: `057478f902f7957d5a3eb9aa39db8646337b190f`
- Load average: 0.06, 0.54, 1.44
- Method: 1 warm-up + 7 reps, median (MAD recorded), GC off while timing, BLAS pinned to 1 thread, core threads 1, 4

| unit | size | reference ms | core ms (1t) | core ms (4t) | speedup 1t | best speedup | parity |
|---|---|---:|---:|---:|---:|---:|---|
| `device_eval_sram_mos_622` | n=248 devices=622 | 2.54 | 0.0756 | 0.7145 | 33.6x | 33.6x | 0.0e+00 ok |
| `device_eval_bjt_3stage` | n=14 devices=4 | 0.4646 | 0.0547 | 0.7528 | 8.50x | 8.50x | 0.0e+00 ok |
| `device_eval_diode_mixer` | n=3 devices=1 | 0.1113 | 0.0551 | 0.5331 | 2.02x | 2.02x | 0.0e+00 ok |
| `circuit_build_sram_248` | n=248 devices=622 | 160.1 | 9.71 | 9.58 | 16.5x | 16.7x | 0.0e+00 ok |
| `circuit_build_rlc_1035` | n=1035 | 132.9 | 9.37 | 8.39 | 14.2x | 15.8x | 0.0e+00 ok |
| `netlist_parse_sram_248` | n=248 devices=622 components=1057 | 11.07 | 3.50 | 3.29 | 3.16x | 3.37x | 0.0e+00 ok |
| `assembly_jacobian_sram_248` | n=248 devices=622 | 3.05 | 0.1234 | 0.8601 | 24.8x | 24.8x | 0.0e+00 ok |
| `direct_solve_rlc_1035` | n=1035 nnz=4071 | 35.55 | 0.9458 | 0.8919 | 37.6x | 39.9x | 2.2e-15 ok |
| `direct_solve_sram_248` | n=248 nnz=2150 | 1.14 | 0.2258 | 0.2537 | 5.04x | 5.04x | 2.4e-16 ok |
| `direct_solve_complex_rlc_1035` | n=1035 nnz=4071 | 78.92 | 1.12 | 1.07 | 70.2x | 74.0x | 9.6e-15 ok |
| `direct_solve_complex_sram_248` | n=248 nnz=2150 | 2.80 | 0.2834 | 0.2699 | 9.87x | 10.4x | 3.4e-16 ok |
| `krylov_gmres_rlc_1035` | n=1035 nnz=4071 | 14.87 | 3.13 | 3.00 | 4.75x | 4.96x | 6.2e-15 ok |
| `krylov_minres_rlc_1035` | n=1035 nnz=4071 | 3.35 | 0.8367 | 0.8519 | 4.00x | 4.00x | 3.9e-15 ok |
| `krylov_cg_rlc_1035` | n=1035 nnz=4071 | 2.55 | 0.8460 | 0.8694 | 3.02x | 3.02x | 4.8e-16 ok |
| `op_sram_248` | n=248 devices=622 | 52.17 | 3.59 | 3.63 | 14.5x | 14.5x | 2.6e-22 ok |
| `tran_sram_248` | n=248 devices=622 steps=24 | 841.6 | 6.41 | 6.99 | 131.3x | 131.3x | 1.9e-15 ok |
| `tran_bjt_3stage_2001` | n=14 devices=4 steps=2000 | 1473.6 | 4.40 | 4.60 | 334.8x | 334.8x | 3.6e-12 ok |
| `tran_rlc_1035_dense` | n=1035 steps=40 | 2297.3 | 8.47 | 9.53 | 271.3x | 271.3x | 8.4e-15 ok |
| `tran_rlc_1035_krylov` | n=1035 steps=40 | 794.3 | 14.27 | 14.04 | 55.7x | 56.6x | 7.1e-16 ok |
| `ac_rlc_1035` | n=1035 points=100 | 9587.8 | 60.00 | 20.00 | 159.8x | 479.4x | 0.0e+00 ok |
| `ac_bjt_3stage` | n=14 devices=4 points=400 | 24.71 | 2.23 | 2.46 | 11.1x | 11.1x | 7.3e-12 ok |
| `hb_mixer_h205` | n=3 harmonics=205 dim=1233 | 4427.5 | 650.9 | 540.1 | 6.80x | 8.20x | 1.1e-14 ok |
| `hb_mixer_h64` | n=3 harmonics=64 dim=387 | 317.5 | 29.61 | 31.55 | 10.7x | 10.7x | 2.4e-16 ok |
| `mor_basis_rlc_1035` | n=1035 order=12 | 237.8 | 8.08 | 7.91 | 29.4x | 30.1x | 9.2e-13 ok |
| `tpwl_train_sram_248` | n=248 devices=622 steps=24 order=24 | 802.7 | 5.48 | 5.87 | 146.4x | 146.4x | 2.6e-05 ok |
| `mor_ac_rlc_1035` | n=1035 points=100 order=12 | 333.3 | 7.06 | 7.24 | 47.2x | 47.2x | 2.3e-14 ok |
| `mor_tran_rlc_1035` | n=1035 steps=40 order=12 | 748.2 | 10.48 | 10.07 | 71.4x | 74.3x | 1.5e-12 ok |
