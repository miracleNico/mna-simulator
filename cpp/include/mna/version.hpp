// Version string and build description of the native core.
#pragma once

#include <string>

namespace mna {

inline constexpr int kVersionMajor = 0;
inline constexpr int kVersionMinor = 70;
inline constexpr int kVersionPatch = 0;
inline constexpr const char* kVersionString = "0.70.0-dev";

/// Version string of the native core, e.g. "0.70.0-dev".
const char* version() noexcept;

/// How this binary was built. Reported to Python as a dict by
/// `mna_simulation._core.build_info()`.
struct BuildInfo {
    std::string version;
    std::string compiler;          // "gcc", "clang", "msvc", ...
    std::string compiler_version;  // "13.3.0", "19.44.35207", ...
    long cxx_standard = 0;         // __cplusplus (or _MSVC_LANG on MSVC), e.g. 201703
    std::string build_type;        // CMake config, e.g. "Release"
    std::string threads_impl;      // "std::thread"
    unsigned hardware_concurrency = 0;
    std::string eigen_version;        // "3.4.0"
    std::string suitesparse_version;  // "7.8.3"
    std::string klu_version;          // "2.3.5"
    std::string platform;             // "linux", "windows", "darwin", "unknown"
    bool sanitizers = false;
    bool coverage = false;
    bool native_arch = false;
    bool debug_asserts = false;  // NDEBUG not defined
};

BuildInfo build_info();

}  // namespace mna
