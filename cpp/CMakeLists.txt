cmake_minimum_required(VERSION 3.26)

project(mna_core
  VERSION 0.70.0
  DESCRIPTION "Native numerical core of mna-simulation"
  LANGUAGES C CXX)

# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------
option(MNA_BUILD_TESTS "Build the doctest unit tests (mna_core_tests)" ON)
option(MNA_ENABLE_SANITIZERS "Build with AddressSanitizer + UndefinedBehaviorSanitizer (GCC/Clang)" OFF)
option(MNA_ENABLE_COVERAGE "Instrument our own code with --coverage (GCC/Clang)" OFF)
option(MNA_NATIVE_ARCH "Tune for the build machine (-march=native, /arch:AVX2). Never for wheels." OFF)
set(MNA_BUILD_PYTHON_MODULE "AUTO" CACHE STRING
  "Build the nanobind extension _core: ON, OFF, or AUTO (build when Python >= 3.12 and nanobind are found)")
set_property(CACHE MNA_BUILD_PYTHON_MODULE PROPERTY STRINGS AUTO ON OFF)

if(SKBUILD)
  # scikit-build-core drives this project to produce the wheel: the module is the point.
  set(MNA_BUILD_PYTHON_MODULE ON)
endif()

get_property(_mna_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _mna_multi_config AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ------------------------------------------------------------------------------
# Global language settings
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Everything ends up statically linked into one Python extension: keep the
# vendored and internal symbols out of its dynamic symbol table.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# Release builds use CMake's defaults: -O3 -DNDEBUG (GCC/Clang), /O2 /Ob2 /DNDEBUG (MSVC).

set(MNA_THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

find_package(Threads REQUIRED)

if(MNA_ENABLE_SANITIZERS)
  if(MSVC)
    message(FATAL_ERROR "MNA_ENABLE_SANITIZERS is only supported with GCC/Clang")
  endif()
  # Applied to every target (vendored C included) so ASan sees all allocations.
  add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

if(MNA_ENABLE_COVERAGE AND MSVC)
  message(FATAL_ERROR "MNA_ENABLE_COVERAGE is only supported with GCC/Clang")
endif()

# Warning / optimisation / instrumentation flags for our own targets only.
function(mna_project_options target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /EHsc /permissive- /utf-8 /bigobj /Zc:__cplusplus)
    target_compile_definitions(${target} PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS)
    if(MNA_NATIVE_ARCH)
      target_compile_options(${target} PRIVATE /arch:AVX2)
    endif()
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    if(MNA_NATIVE_ARCH)
      target_compile_options(${target} PRIVATE -march=native)
    endif()
    if(MNA_ENABLE_COVERAGE)
      target_compile_options(${target} PRIVATE --coverage)
      target_link_options(${target} PRIVATE --coverage)
    endif()
  endif()
endfunction()

# ------------------------------------------------------------------------------
# klu_vendored: SuiteSparse_config + AMD + COLAMD + BTF + KLU, compiled as C
# ------------------------------------------------------------------------------
set(_ss "${MNA_THIRD_PARTY_DIR}/suitesparse")
file(GLOB _klu_sources CONFIGURE_DEPENDS
  "${_ss}/AMD/Source/*.c"
  "${_ss}/COLAMD/Source/*.c"
  "${_ss}/BTF/Source/*.c"
  "${_ss}/KLU/Source/*.c")
list(LENGTH _klu_sources _klu_source_count)
if(_klu_source_count LESS 90)
  message(FATAL_ERROR "Vendored SuiteSparse sources missing under ${_ss} (found ${_klu_source_count} files)")
endif()

add_library(klu_vendored STATIC "${_ss}/SuiteSparse_config/SuiteSparse_config.c" ${_klu_sources})
target_include_directories(klu_vendored SYSTEM PUBLIC
  "${_ss}/SuiteSparse_config"
  "${_ss}/AMD/Include"
  "${_ss}/COLAMD/Include"
  "${_ss}/BTF/Include"
  "${_ss}/KLU/Include")
# NTIMER: the shipped SuiteSparse_config.h enables a clock_gettime() timer that
# MSVC lacks; AMD/COLAMD/BTF/KLU never call it, so compile it out everywhere.
target_compile_definitions(klu_vendored PUBLIC NTIMER)
if(MSVC)
  target_compile_options(klu_vendored PRIVATE /W0)
  target_compile_definitions(klu_vendored PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
  target_compile_options(klu_vendored PRIVATE -w)
endif()
if(UNIX AND NOT APPLE)
  target_link_libraries(klu_vendored PUBLIC m)
endif()

# ------------------------------------------------------------------------------
# mna_core: our C++ library
# ------------------------------------------------------------------------------
# Every .cpp under src/ belongs to the library: new units add files, not CMake edits.
file(GLOB_RECURSE _mna_core_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(mna_core STATIC ${_mna_core_sources})
target_include_directories(mna_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(mna_core SYSTEM PUBLIC "${MNA_THIRD_PARTY_DIR}/eigen")
target_compile_features(mna_core PUBLIC cxx_std_17)
target_link_libraries(mna_core PUBLIC klu_vendored Threads::Threads)
target_compile_definitions(mna_core
  PUBLIC
    # Refuse to compile any LGPL-only Eigen header; Eigen never spawns threads of its own.
    EIGEN_MPL2_ONLY
    EIGEN_DONT_PARALLELIZE
  PRIVATE
    MNA_BUILD_TYPE="$<CONFIG>"
    MNA_SANITIZERS=$<BOOL:${MNA_ENABLE_SANITIZERS}>
    MNA_COVERAGE=$<BOOL:${MNA_ENABLE_COVERAGE}>
    MNA_NATIVE_ARCH_BUILD=$<BOOL:${MNA_NATIVE_ARCH}>)
mna_project_options(mna_core)

# ------------------------------------------------------------------------------
# _core: nanobind extension module
# ------------------------------------------------------------------------------
if(MNA_BUILD_PYTHON_MODULE STREQUAL "AUTO")
  set(_mna_py_quiet QUIET)
  set(_mna_py_required "")
else()
  set(_mna_py_quiet "")
  set(_mna_py_required REQUIRED)
endif()

set(_mna_module_built OFF)
if(NOT MNA_BUILD_PYTHON_MODULE STREQUAL "OFF")
  # Development.SABIModule enables the stable ABI (abi3) build; nanobind falls
  # back to the version-specific ABI when it is unavailable.
  find_package(Python 3.12 ${_mna_py_quiet} ${_mna_py_required}
    COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule)
  if(Python_FOUND)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      OUTPUT_VARIABLE nanobind_ROOT
      RESULT_VARIABLE _mna_nb_rc
      ERROR_QUIET)
    if(_mna_nb_rc EQUAL 0)
      find_package(nanobind CONFIG ${_mna_py_quiet} ${_mna_py_required})
    elseif(_mna_py_required)
      message(FATAL_ERROR "nanobind is not importable by ${Python_EXECUTABLE}; pip install nanobind")
    endif()
  endif()

  if(Python_FOUND AND nanobind_FOUND)
    # module.cpp defines the module; every other bindings/*.cpp registers itself
    # through bindings/registry.hpp, so new units never edit module.cpp.
    file(GLOB _mna_binding_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/bindings/*.cpp")
    nanobind_add_module(_core STABLE_ABI NB_STATIC NB_SUPPRESS_WARNINGS ${_mna_binding_sources})
    target_link_libraries(_core PRIVATE mna_core)
    mna_project_options(_core)
    install(TARGETS _core LIBRARY DESTINATION mna_simulation)
    set(_mna_module_built ON)
    if(MNA_ENABLE_SANITIZERS)
      message(WARNING "_core is built with sanitizers: importing it needs the sanitizer runtime preloaded")
    endif()
  else()
    message(STATUS "mna_core: Python >= 3.12 with nanobind not found; skipping the _core module")
  endif()
endif()

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------
if(MNA_BUILD_TESTS)
  enable_testing()
  file(GLOB _mna_test_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
  add_executable(mna_core_tests ${_mna_test_sources})
  target_include_directories(mna_core_tests SYSTEM PRIVATE "${MNA_THIRD_PARTY_DIR}")
  target_link_libraries(mna_core_tests PRIVATE mna_core)
  mna_project_options(mna_core_tests)
  add_test(NAME mna_core_tests COMMAND mna_core_tests)
endif()

message(STATUS "mna_core ${PROJECT_VERSION}: build type '${CMAKE_BUILD_TYPE}', tests ${MNA_BUILD_TESTS}, "
               "python module ${_mna_module_built}, sanitizers ${MNA_ENABLE_SANITIZERS}, "
               "coverage ${MNA_ENABLE_COVERAGE}, native arch ${MNA_NATIVE_ARCH}")
