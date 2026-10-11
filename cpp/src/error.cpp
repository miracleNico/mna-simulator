#include "mna/error.hpp"

namespace mna {

const char* error_code_name(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::NetlistFatal: return "NETLIST_FATAL";
        case ErrorCode::ParseError: return "PARSE_ERROR";
        case ErrorCode::ComponentError: return "COMPONENT_ERROR";
        case ErrorCode::ModelError: return "MODEL_ERROR";
        case ErrorCode::MorConfig: return "MOR_CONFIG";
        case ErrorCode::NrConvergence: return "NR_CONVERGENCE";
        case ErrorCode::JacobianSingular: return "JACOBIAN_SINGULAR";
        case ErrorCode::SolverRetry: return "SOLVER_RETRY";
        case ErrorCode::LuConvergence: return "LU_CONVERGENCE";
        case ErrorCode::HbError: return "HB_ERROR";
        case ErrorCode::AcError: return "AC_ERROR";
        case ErrorCode::Unknown: break;
    }
    return "UNKNOWN";
}

Error::Error(ErrorCode code, const std::string& message) : std::runtime_error(message), code_(code) {}

}  // namespace mna
