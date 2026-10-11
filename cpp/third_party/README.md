# Vendored third-party code

Everything the native core (`mna_simulation._core`) needs to compile is checked
in here, so a build needs only a C/C++17 compiler, CMake >= 3.26 and, for the
Python module, `scikit-build-core` + `nanobind` from PyPI. Nothing is fetched at
configure time. The vendored trees are unmodified upstream files; only whole
files and directories were left out.

| Component | Version | Upstream | License (SPDX) | Used by |
|---|---|---|---|---|
| Eigen | 3.4.0 (tarball `eigen-3.4.0.tar.gz`, sha256 `8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72`) | https://gitlab.com/libeigen/eigen | `MPL-2.0` (a few headers: BSD-3-Clause, Apache-2.0, LGPL-2.1-or-later; see below) | `mna_core` (header-only) |
| SuiteSparse_config | SuiteSparse v7.8.3 (commit `d3c4926d2c47fd6ae558e898bfc072ade210a2a1`) | https://github.com/DrTimothyAldenDavis/SuiteSparse | `BSD-3-Clause` | `klu_vendored` |
| AMD | 3.3.3 (SuiteSparse v7.8.3) | same | `BSD-3-Clause` | `klu_vendored` |
| COLAMD | 3.3.4 (SuiteSparse v7.8.3) | same | `BSD-3-Clause` | `klu_vendored` |
| BTF | 2.3.2 (SuiteSparse v7.8.3) | same | `LGPL-2.1-or-later` | `klu_vendored` |
| KLU | 2.3.5 (SuiteSparse v7.8.3) | same | `LGPL-2.1-or-later` | `klu_vendored` |
| doctest | 2.5.0 (tag `v2.5.0`, commit `d44d4f6e66232d716af82f00a063759e9d0e50d6`) | https://github.com/doctest/doctest | `MIT` | `mna_core_tests` only (never in the wheel) |

Not vendored, but compiled into the extension at build time from the
`nanobind` package on PyPI: nanobind (`BSD-3-Clause`) and its bundled
tsl::robin_map (`MIT`). Their notices are in `LICENSES/` too.

`LICENSES/` holds every license text above. The wheel ships the whole folder in
`*.dist-info/licenses/` (`project.license-files` in `pyproject.toml`), which
covers the "reproduce the notice in binary distributions" clauses.

## LGPL note (KLU, BTF)

KLU and BTF are LGPL-2.1-or-later and are **statically linked** into
`mna_simulation._core`. This project is open source and the complete build
(including these unmodified KLU/BTF sources and the CMake scripts) is in this
repository, so anyone can rebuild the extension against a modified KLU/BTF,
which is what LGPL-2.1 section 6 asks of a statically linked work. Keep it that
way: do not distribute closed-source builds of `_core` without revisiting this
(the alternatives are dynamic linking of KLU or a commercial KLU license).

## Eigen license guard

`mna_core` defines `EIGEN_MPL2_ONLY`, so including one of Eigen's few
LGPL-licensed headers (e.g. `IncompleteLUT.h`) is a compile error.
`EIGEN_DONT_PARALLELIZE` is defined as well: threading is owned by
`mna/threads.hpp`, Eigen never starts threads of its own. The BSD-licensed
files are Intel MKL/LAPACKE bindings that we do not enable, and the
Apache-2.0 file is `Core/arch/Default/BFloat16.h`; their texts are still
included because the files are present.

## What is here and what was dropped

```
eigen/
  Eigen/                     the complete Eigen/ header tree (6.8 MB)
  COPYING.MPL2 COPYING.README COPYING.BSD COPYING.LGPL COPYING.APACHE
  (dropped: everything else in the release tarball -- unsupported/, test/,
   doc/, bench/, demos/, blas/, lapack/, cmake/, scripts/, failtest/, debug/,
   ci/, build files -- and COPYING.GPL / COPYING.MINPACK, which no file under
   Eigen/ refers to; MINPACK code lives only in unsupported/)
suitesparse/
  SuiteSparse_config/  SuiteSparse_config.c  SuiteSparse_config.h  README.txt
  AMD/ COLAMD/ BTF/ KLU/
    Source/*.c   Include/*.h   README.txt   Doc/License.txt   Doc/lesser.txt (BTF, KLU)
  (dropped: every other SuiteSparse package; per package: Fortran Source/*.f
   (AMD's amd.f, amdbar.f), MATLAB/, Demo/, Tcov/, User/, Matrix/, Config/,
   build/, Doc/ except the license files, Makefile, CMakeLists.txt; and
   SuiteSparse_config's Config/, cmake_modules/, build/, Makefile,
   CMakeLists.txt)
doctest/
  doctest.h  LICENSE.txt
  (dropped: everything else, including doctest/parts and extensions)
LICENSES/
  one file per license text, prefixed with the component name
```

`SuiteSparse_config.h` is the pre-generated header committed upstream (it is
normally produced by SuiteSparse's CMake from `Config/SuiteSparse_config.h.in`).
We do not run SuiteSparse's CMake; `cpp/CMakeLists.txt` compiles the 95 C
files of the five packages directly into the static library `klu_vendored`
with warnings off and `NTIMER` defined. `NTIMER` matters: the shipped header
enables a `clock_gettime()` timer that MSVC does not have; none of AMD, COLAMD,
BTF or KLU uses the timer.

## Update procedure

Run from the repository root; `$SRC` is a scratch directory outside the repo.

### SuiteSparse (KLU, BTF, AMD, COLAMD, SuiteSparse_config)

```sh
TAG=v7.8.3                         # new release tag
git clone --depth 1 --branch "$TAG" https://github.com/DrTimothyAldenDavis/SuiteSparse "$SRC/ss"
T=cpp/third_party/suitesparse
rm -rf "$T" && mkdir -p "$T/SuiteSparse_config"
cp "$SRC"/ss/SuiteSparse_config/{SuiteSparse_config.c,SuiteSparse_config.h,README.txt} "$T/SuiteSparse_config/"
for p in AMD COLAMD BTF KLU; do
  mkdir -p "$T/$p/Source" "$T/$p/Include" "$T/$p/Doc"
  cp "$SRC"/ss/$p/Source/*.c  "$T/$p/Source/"
  cp "$SRC"/ss/$p/Include/*.h "$T/$p/Include/"
  cp "$SRC"/ss/$p/README.txt  "$T/$p/"
  cp "$SRC"/ss/$p/Doc/License.txt "$T/$p/Doc/"
  cp "$SRC"/ss/$p/Doc/License.txt "cpp/third_party/LICENSES/SuiteSparse-$p-License.txt"
  if [ -f "$SRC/ss/$p/Doc/lesser.txt" ]; then
    cp "$SRC/ss/$p/Doc/lesser.txt" "$T/$p/Doc/"
    cp "$SRC/ss/$p/Doc/lesser.txt" "cpp/third_party/LICENSES/SuiteSparse-$p-lesser.txt"
  fi
done
(cd "$SRC/ss" && git rev-parse HEAD)   # record the commit in the table above
```

Then check: the copyright line in `LICENSES/SuiteSparse_config-License.txt`
against `SuiteSparse_config/README.txt`; that `SuiteSparse_config.h` still
needs no configure step (`grep cmakedefine` must find nothing); the package
versions (`AMD_MAIN_VERSION`, ... `KLU_SUBSUB_VERSION` in the headers) for the
table above and for the expected values in `cpp/tests/test_build_info.cpp` and
`tests/test_native_smoke.py`; and that the file-count guard in
`cpp/CMakeLists.txt` (`_klu_source_count`) still fits.

### Eigen

```sh
VER=3.4.0
curl -L -o "$SRC/eigen.tar.gz" "https://gitlab.com/libeigen/eigen/-/archive/$VER/eigen-$VER.tar.gz"
sha256sum "$SRC/eigen.tar.gz"      # record in the table above
tar -xzf "$SRC/eigen.tar.gz" -C "$SRC"
rm -rf cpp/third_party/eigen && mkdir -p cpp/third_party/eigen
cp -r "$SRC/eigen-$VER/Eigen" cpp/third_party/eigen/
for f in COPYING.MPL2 COPYING.README COPYING.BSD COPYING.LGPL COPYING.APACHE; do
  cp "$SRC/eigen-$VER/$f" cpp/third_party/eigen/
  cp "$SRC/eigen-$VER/$f" "cpp/third_party/LICENSES/Eigen-$f"
done
```

Re-check which headers under `Eigen/` carry non-MPL2 licenses
(`grep -rl -i "Lesser General\|Apache License\|Redistribution and use" cpp/third_party/eigen/Eigen`)
and update the notes above.

### doctest

```sh
TAG=v2.5.0
git clone --depth 1 --branch "$TAG" https://github.com/doctest/doctest "$SRC/doctest"
cp "$SRC/doctest/doctest/doctest.h" "$SRC/doctest/LICENSE.txt" cpp/third_party/doctest/
cp "$SRC/doctest/LICENSE.txt" cpp/third_party/LICENSES/doctest-LICENSE.txt
(cd "$SRC/doctest" && git rev-parse HEAD)
```

### nanobind / robin-map notices

When the minimum or the pinned nanobind version changes, refresh
`LICENSES/nanobind-LICENSE.txt` from the installed wheel
(`nanobind-*.dist-info/licenses/LICENSE`) and `LICENSES/robin-map-LICENSE.txt`
from the header comment of `nanobind/ext/robin_map/include/tsl/robin_map.h`.

### After any update

```sh
cmake -S cpp -B build-cpp -DMNA_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpp -j && ctest --test-dir build-cpp --output-on-failure
python -m pip install -e ".[dev]" && python -m pytest -q
```

and update the version table in this file in the same commit.
