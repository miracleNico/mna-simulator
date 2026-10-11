// Error type shared by the native core. Codes mirror the string codes raised by
// the Python core (mna_simulation/errors.py) so the two can be translated 1:1.
#pragma once

#include <stdexcept>
#include <string>

namespace mna {

enum class ErrorCode : int {
    Unknown = 0,
    // Raised as NetlistError on the Python side.
    NetlistFatal,
    ParseError,
    ComponentError,
    ModelError,
    MorConfig,
    // Raised as SolverConvergenceError on the Python side.
    NrConvergence,
    JacobianSingular,
    SolverRetry,
    LuConvergence,
    HbError,
    AcError,
    // Raised as NetlistError on the Python side (an input error, like MorConfig).
    TranError,
};

/// The Python-side code string, e.g. "JACOBIAN_SINGULAR".
const char* error_code_name(ErrorCode code) noexcept;

class Error : public std::runtime_error {
public:
    Error(ErrorCode code, const std::string& message);

    ErrorCode code() const noexcept { return code_; }
    const char* code_name() const noexcept { return error_code_name(code_); }
    /// The message without the code prefix (what() returns the same text).
    const char* message() const noexcept { return what(); }

private:
    ErrorCode code_;
};

}  // namespace mna
