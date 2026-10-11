### Python reference vs C++ core (2026-10-11T21:52:11+00:00)

- Host: `vm`, Linux-6.18.44-fc-v114-x86_64-with-glibc2.39, 4 CPUs
- Python 3.13.16, NumPy 2.5.4, SciPy 1.18.1
- Native core: 0.70.0-dev (gcc 13.3.0, Release)
- Commit: `04c46a36f4da4f4cfc8500e7a664c282ea1443bd`
- Load average: 4.26, 5.14, 5.10
- Method: 1 warm-up + 7 reps, median (MAD recorded), GC off while timing, BLAS pinned to 1 thread, core threads 1, 4
- **Warning:** 1-minute load average is 4.26 (> 0.5); timings are not representative. Rerun on an idle machine before comparing or committing a baseline.

| unit | size | reference ms | core ms (1t) | core ms (4t) | speedup 1t | best speedup | parity |
|---|---|---:|---:|---:|---:|---:|---|
| `device_eval_sram_mos_622` | n=248 devices=622 | 2.57 | 0.0883 | 0.9454 | 29.1x | 29.1x | 0.0e+00 ok |
| `device_eval_bjt_3stage` | n=14 devices=4 | 0.4073 | 0.0574 | 0.6204 | 7.10x | 7.10x | 0.0e+00 ok |
| `device_eval_diode_mixer` | n=3 devices=1 | 0.1149 | 0.0527 | 0.5297 | 2.18x | 2.18x | 0.0e+00 ok |
| `circuit_build_sram_248` | n=248 devices=622 | 172.8 | 8.74 | 8.71 | 19.8x | 19.8x | 0.0e+00 ok |
| `circuit_build_rlc_1035` | n=1035 | 147.1 | 7.43 | 7.23 | 19.8x | 20.3x | 0.0e+00 ok |
| `netlist_parse_sram_248` | n=248 devices=622 components=1057 | 11.45 | 3.28 | 3.70 | 3.49x | 3.49x | 0.0e+00 ok |
| `assembly_jacobian_sram_248` | n=248 devices=622 | 3.09 | 0.1089 | 0.7246 | 28.4x | 28.4x | 0.0e+00 ok |
| `direct_solve_rlc_1035` | n=1035 nnz=4071 | 34.06 | 0.8843 | 0.8760 | 38.5x | 38.9x | 2.2e-15 ok |
| `direct_solve_sram_248` | n=248 nnz=2150 | 1.13 | 0.2560 | 0.2348 | 4.40x | 4.80x | 2.4e-16 ok |
| `direct_solve_complex_rlc_1035` | n=1035 nnz=4071 | 84.72 | 1.06 | 1.14 | 79.6x | 79.6x | 9.6e-15 ok |
| `direct_solve_complex_sram_248` | n=248 nnz=2150 | 2.40 | 0.2875 | 0.3036 | 8.34x | 8.34x | 3.4e-16 ok |
| `krylov_gmres_rlc_1035` | n=1035 nnz=4071 | 10.56 | 3.12 | 3.16 | 3.39x | 3.39x | 6.2e-15 ok |
| `krylov_minres_rlc_1035` | n=1035 nnz=4071 | 3.03 | 0.8311 | 0.8299 | 3.65x | 3.65x | 3.9e-15 ok |
| `krylov_cg_rlc_1035` | n=1035 nnz=4071 | 2.86 | 0.7619 | 0.8103 | 3.75x | 3.75x | 4.8e-16 ok |
| `op_sram_248` | n=248 devices=622 | 52.02 | 3.62 | 3.66 | 14.4x | 14.4x | 2.6e-22 ok |
| `tran_sram_248` | n=248 devices=622 steps=24 | 840.8 | 6.82 | 7.00 | 123.3x | 123.3x | 1.9e-15 ok |
| `tran_bjt_3stage_2001` | n=14 devices=4 steps=2000 | 1494.3 | 4.22 | 4.41 | 354.2x | 354.2x | 3.6e-12 ok |
| `tran_rlc_1035_dense` | n=1035 steps=40 | 2224.0 | 8.17 | 8.19 | 272.1x | 272.1x | 8.4e-15 ok |
| `tran_rlc_1035_krylov` | n=1035 steps=40 | 655.9 | 13.40 | 13.91 | 48.9x | 48.9x | 7.1e-16 ok |
| `ac_rlc_1035` | n=1035 points=100 | 9592.2 | 55.01 | 18.32 | 174.4x | 523.7x | 0.0e+00 ok |
| `ac_bjt_3stage` | n=14 devices=4 points=400 | 24.25 | 2.04 | 2.05 | 11.9x | 11.9x | 7.3e-12 ok |
| `hb_mixer_h205` | n=3 harmonics=205 dim=1233 | 4426.9 | 706.2 | 693.3 | 6.27x | 6.39x | 1.1e-14 ok |
| `hb_mixer_h64` | n=3 harmonics=64 dim=387 | 311.9 | 32.55 | 30.28 | 9.58x | 10.3x | 2.4e-16 ok |
| `mor_basis_rlc_1035` | n=1035 order=12 | 232.3 | 7.77 | 7.85 | 29.9x | 29.9x | 9.2e-13 ok |
| `tpwl_train_sram_248` | n=248 devices=622 steps=24 order=24 | 894.1 | 6.83 | 6.63 | 130.9x | 134.9x | 2.6e-05 ok |
| `mor_ac_rlc_1035` | n=1035 points=100 order=12 | 335.9 | 7.15 | 6.83 | 47.0x | 49.2x | 2.3e-14 ok |
| `mor_tran_rlc_1035` | n=1035 steps=40 order=12 | 777.6 | 9.74 | 9.53 | 79.8x | 81.6x | 1.5e-12 ok |
