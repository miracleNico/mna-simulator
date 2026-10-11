#include <Eigen/Core>
#include <klu.h>

#include <string>
#include <thread>

#include "mna/version.hpp"

#ifndef MNA_BUILD_TYPE
#define MNA_BUILD_TYPE ""
#endif
#ifndef MNA_SANITIZERS
#define MNA_SANITIZERS 0
#endif
#ifndef MNA_COVERAGE
#define MNA_COVERAGE 0
#endif
#ifndef MNA_NATIVE_ARCH_BUILD
#define MNA_NATIVE_ARCH_BUILD 0
#endif

namespace mna {
namespace {

std::string dotted(long a, long b, long c) {
    return std::to_string(a) + "." + std::to_string(b) + "." + std::to_string(c);
}

void compiler_identity(std::string& name, std::string& ver) {
#if defined(__clang__)
#if defined(_MSC_VER)
    name = "clang-cl";
#elif defined(__apple_build_version__)
    name = "apple-clang";
#else
    name = "clang";
#endif
    ver = dotted(__clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__GNUC__)
    name = "gcc";
    ver = dotted(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
    name = "msvc";
    // _MSC_FULL_VER is MMmmbbbbb, e.g. 194435207 -> 19.44.35207
    ver = dotted(_MSC_FULL_VER / 10000000L, (_MSC_FULL_VER / 100000L) % 100L, _MSC_FULL_VER % 100000L);
#else
    name = "unknown";
    ver = "0.0.0";
#endif
}

}  // namespace

const char* version() noexcept { return kVersionString; }

BuildInfo build_info() {
    BuildInfo info;
    info.version = kVersionString;
    compiler_identity(info.compiler, info.compiler_version);
#if defined(_MSVC_LANG)
    // MSVC reports __cplusplus as 199711L unless /Zc:__cplusplus is given.
    info.cxx_standard = static_cast<long>(_MSVC_LANG);
#else
    info.cxx_standard = static_cast<long>(__cplusplus);
#endif
    info.build_type = MNA_BUILD_TYPE;
    if (info.build_type.empty()) {
        info.build_type = "unspecified";
    }
    info.threads_impl = "std::thread";
    info.hardware_concurrency = std::thread::hardware_concurrency();
    info.eigen_version = dotted(EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION);
    info.suitesparse_version = dotted(SUITESPARSE_MAIN_VERSION, SUITESPARSE_SUB_VERSION, SUITESPARSE_SUBSUB_VERSION);
    info.klu_version = dotted(KLU_MAIN_VERSION, KLU_SUB_VERSION, KLU_SUBSUB_VERSION);
#if defined(_WIN32)
    info.platform = "windows";
#elif defined(__APPLE__)
    info.platform = "darwin";
#elif defined(__linux__)
    info.platform = "linux";
#else
    info.platform = "unknown";
#endif
    info.sanitizers = MNA_SANITIZERS != 0;
    info.coverage = MNA_COVERAGE != 0;
    info.native_arch = MNA_NATIVE_ARCH_BUILD != 0;
#if defined(NDEBUG)
    info.debug_asserts = false;
#else
    info.debug_asserts = true;
#endif
    return info;
}

}  // namespace mna
