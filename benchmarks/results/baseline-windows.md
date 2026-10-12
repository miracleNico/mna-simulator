### Python reference vs C++ core (2026-10-12T18:08:56+00:00)

- Host: `MIRACLE_LAPTOP`, Windows-11-10.0.26200-SP0, 16 CPUs
- Python 3.14.2, NumPy 2.5.4, SciPy 1.18.1
- Native core: 0.70.0-dev (msvc 19.38.33135, Release)
- Commit: `4b0d65adf4b0654fadfa0c7a9ae0775c8c9e7ce3`
- Load average: n/a (Windows has none). CPU utilisation sampled every 5 s during the run, the benchmark itself included: mean 12.6%, max 17.6% (one busy thread of 16 is 6.25%); 11% idle before the run. AMD Ryzen 9 6900HS (8 cores, 16 threads), on AC power, Turbo power mode
- Method: 1 warm-up + 7 reps, median (MAD recorded), GC off while timing, BLAS pinned to 1 thread, core threads 1, 4

| unit | size | reference ms | core ms (1t) | core ms (4t) | speedup 1t | best speedup | parity |
|---|---|---:|---:|---:|---:|---:|---|
| `device_eval_sram_mos_622` | n=248 devices=622 | 1.64 | 0.0434 | 1.31 | 37.7x | 37.7x | 0.0e+00 ok |
| `device_eval_bjt_3stage` | n=14 devices=4 | 0.2333 | 0.0265 | 1.17 | 8.80x | 8.80x | 0.0e+00 ok |
| `device_eval_diode_mixer` | n=3 devices=1 | 0.0529 | 0.0227 | 0.7757 | 2.33x | 2.33x | 0.0e+00 ok |
| `circuit_build_sram_248` | n=248 devices=622 | 116.1 | 5.14 | 5.35 | 22.6x | 22.6x | 0.0e+00 ok |
| `circuit_build_rlc_1035` | n=1035 | 90.61 | 4.64 | 4.58 | 19.5x | 19.8x | 0.0e+00 ok |
| `netlist_parse_sram_248` | n=248 devices=622 components=1057 | 7.55 | 2.08 | 2.11 | 3.62x | 3.62x | 0.0e+00 ok |
| `assembly_jacobian_sram_248` | n=248 devices=622 | 1.70 | 0.0554 | 1.22 | 30.6x | 30.6x | 0.0e+00 ok |
| `direct_solve_rlc_1035` | n=1035 nnz=4071 | 17.14 | 0.5040 | 0.5090 | 34.0x | 34.0x | 2.0e-15 ok |
| `direct_solve_sram_248` | n=248 nnz=2150 | 0.4603 | 0.1311 | 0.1311 | 3.51x | 3.51x | 2.4e-16 ok |
| `direct_solve_complex_rlc_1035` | n=1035 nnz=4071 | 55.69 | 0.6518 | 0.6436 | 85.4x | 86.5x | 1.0e-14 ok |
| `direct_solve_complex_sram_248` | n=248 nnz=2150 | 1.17 | 0.1609 | 0.1575 | 7.26x | 7.42x | 2.5e-16 ok |
| `krylov_gmres_rlc_1035` | n=1035 nnz=4071 | 5.13 | 1.93 | 1.80 | 2.66x | 2.84x | 3.8e-15 ok |
| `krylov_minres_rlc_1035` | n=1035 nnz=4071 | 1.58 | 0.4367 | 0.4230 | 3.62x | 3.74x | 4.2e-15 ok |
| `krylov_cg_rlc_1035` | n=1035 nnz=4071 | 1.18 | 0.4489 | 0.4406 | 2.62x | 2.67x | 5.7e-16 ok |
| `op_sram_248` | n=248 devices=622 | 29.61 | 2.34 | 2.37 | 12.6x | 12.6x | 2.9e-22 ok |
| `tran_sram_248` | n=248 devices=622 steps=24 | 455.5 | 4.42 | 4.21 | 103.1x | 108.1x | 1.9e-15 ok |
| `tran_bjt_3stage_2001` | n=14 devices=4 steps=2000 | 662.9 | 3.14 | 3.09 | 211.0x | 214.8x | 3.7e-12 ok |
| `tran_rlc_1035_dense` | n=1035 steps=40 | 1117.5 | 4.23 | 4.73 | 264.5x | 264.5x | 7.3e-15 ok |
| `tran_rlc_1035_krylov` | n=1035 steps=40 | 345.7 | 8.04 | 8.14 | 43.0x | 43.0x | 7.1e-16 ok |
| `ac_rlc_1035` | n=1035 points=100 | 6341.8 | 34.64 | 10.30 | 183.1x | 615.6x | 0.0e+00 ok |
| `ac_bjt_3stage` | n=14 devices=4 points=400 | 10.23 | 1.41 | 2.21 | 7.25x | 7.25x | 7.3e-12 ok |
| `hb_mixer_h205` | n=3 harmonics=205 dim=1233 | 2531.3 | 388.2 | 301.1 | 6.52x | 8.41x | 5.2e-15 ok |
| `hb_mixer_h64` | n=3 harmonics=64 dim=387 | 164.2 | 18.38 | 18.19 | 8.94x | 9.03x | 2.8e-16 ok |
| `mor_basis_rlc_1035` | n=1035 order=12 | 108.0 | 3.66 | 3.69 | 29.5x | 29.5x | 9.2e-13 ok |
| `tpwl_train_sram_248` | n=248 devices=622 steps=24 order=24 | 449.1 | 3.63 | 3.63 | 123.9x | 123.9x | 3.7e-05 ok |
| `mor_ac_rlc_1035` | n=1035 points=100 order=12 | 173.9 | 3.85 | 3.80 | 45.2x | 45.8x | 2.6e-14 ok |
| `mor_tran_rlc_1035` | n=1035 steps=40 order=12 | 380.1 | 5.54 | 5.56 | 68.7x | 68.7x | 1.1e-12 ok |
