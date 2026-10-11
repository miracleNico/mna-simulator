#include "mna/mor/krylov_basis.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mna/error.hpp"
#include "mna/linear/klu_solver.hpp"
#include "mna/mor/safe_solve.hpp"
#include "mna/threads.hpp"

namespace mna::mor {
namespace {

[[noreturn]] void bad_input(const std::string& message) {
    throw Error(ErrorCode::NetlistFatal, "mor_krylov_basis: " + message);
}

/// One shifted system A = G + s C, factored once and reused for every block.
class ShiftSystem {
public:
    ShiftSystem(const SparseOperands& operands, const Shift& shift) : ops_(&operands), shift_(shift) {}

    bool factored() const noexcept { return factored_; }
    bool singular() const noexcept { return singular_; }

    /// KLU analysis + factorization; a singular matrix switches this system to
    /// the least-squares fallback. Idempotent.
    void factor() {
        if (factored_) {
            return;
        }
        const linear::CscPattern& pattern = *ops_->pattern;
        const auto nnz = static_cast<std::size_t>(pattern.nnz());
        try {
            if (!shift_.is_complex) {
                // np.asarray(G, dtype=float) + shift * np.asarray(C)
                const double s = shift_.value.real();
                real_values_.resize(nnz);
                for (std::size_t k = 0; k < nnz; ++k) {
                    real_values_[k] = ops_->g[k] + s * ops_->c[k];
                }
                real_.analyze(pattern);
                real_.factor(real_values_.data(), real_values_.size());
            } else {
                // np.asarray(G, dtype=complex) + shift * np.asarray(C): (a + bj) * (c + 0j) added to (g + 0j).
                const double a = shift_.value.real();
                const double b = shift_.value.imag();
                complex_values_.resize(nnz);
                for (std::size_t k = 0; k < nnz; ++k) {
                    const double c = ops_->c[k];
                    complex_values_[k] = Complex(ops_->g[k] + (a * c - b * 0.0), 0.0 + (a * 0.0 + b * c));
                }
                complex_.analyze(pattern);
                complex_.factor(complex_values_.data(), complex_values_.size());
            }
        } catch (const Error& e) {
            if (e.code() != ErrorCode::JacobianSingular) {
                throw;
            }
            singular_ = true;
        }
        if (!singular_) {
            // The values are only needed again by the dense least-squares fallback.
            std::vector<double>().swap(real_values_);
            std::vector<Complex>().swap(complex_values_);
        }
        factored_ = true;
    }

    /// Solves op(A) X = rhs (op = transpose for adjoint solves) for a real
    /// column-major n x cols block and returns the real candidate block: X
    /// itself for a real shift, [Re X, Im X] (Im X only when its norm is > 0)
    /// for a complex one. `out_cols` receives the candidate's column count.
    std::vector<double> solve(bool transpose, std::vector<double>&& rhs, std::int32_t cols, std::int32_t& out_cols) {
        factor();
        const auto n = static_cast<std::size_t>(ops_->pattern->n);
        out_cols = 0;
        if (cols <= 0) {
            return {};
        }
        if (!shift_.is_complex) {
            if (singular_) {
                dense_real(transpose).solve(rhs.data(), cols);
            } else if (transpose) {
                real_.tsolve(rhs.data(), rhs.size(), cols, false);
            } else {
                real_.solve(rhs.data(), rhs.size(), cols);
            }
            out_cols = cols;
            return std::move(rhs);
        }
        std::vector<Complex> x(rhs.begin(), rhs.end());
        if (singular_) {
            dense_complex(transpose).solve(x.data(), cols);
        } else if (transpose) {
            complex_.tsolve(x.data(), x.size(), cols, false);
        } else {
            complex_.solve(x.data(), x.size(), cols);
        }
        // _orthogonalize_block: real parts, then the imaginary parts when norm(imag) > 0.
        double imag_squares = 0.0;
        for (const Complex& value : x) {
            imag_squares += value.imag() * value.imag();
        }
        const bool with_imag = imag_squares > 0.0;
        const auto ccols = static_cast<std::size_t>(cols);
        std::vector<double> out(n * ccols * (with_imag ? 2 : 1));
        for (std::size_t k = 0; k < n * ccols; ++k) {
            out[k] = x[k].real();
        }
        if (with_imag) {
            for (std::size_t k = 0; k < n * ccols; ++k) {
                out[n * ccols + k] = x[k].imag();
            }
        }
        out_cols = with_imag ? 2 * cols : cols;
        return out;
    }

private:
    template <class Scalar>
    std::vector<Scalar> dense_matrix(const std::vector<Scalar>& values, bool transpose) const {
        const linear::CscPattern& pattern = *ops_->pattern;
        const auto n = static_cast<std::size_t>(pattern.n);
        std::vector<Scalar> dense(n * n, Scalar(0.0));
        for (std::size_t col = 0; col < n; ++col) {
            for (auto p = static_cast<std::size_t>(pattern.col_ptr[col]);
                 p < static_cast<std::size_t>(pattern.col_ptr[col + 1]); ++p) {
                const auto row = static_cast<std::size_t>(pattern.row_idx[p]);
                // column-major: (row, col) at row + col * n; the transpose swaps them.
                dense[transpose ? col + row * n : row + col * n] = values[p];
            }
        }
        return dense;
    }

    DenseSafeSolver<double>& dense_real(bool transpose) {
        const int which = transpose ? 1 : 0;
        if (!dense_ready_[which]) {
            const std::vector<double> dense = dense_matrix(real_values_, transpose);
            dense_real_[which].factor(ops_->pattern->n, dense.data(), /*known_singular=*/true);
            dense_ready_[which] = true;
        }
        return dense_real_[which];
    }

    DenseSafeSolver<Complex>& dense_complex(bool transpose) {
        const int which = transpose ? 1 : 0;
        if (!dense_ready_[which]) {
            const std::vector<Complex> dense = dense_matrix(complex_values_, transpose);
            dense_complex_[which].factor(ops_->pattern->n, dense.data(), /*known_singular=*/true);
            dense_ready_[which] = true;
        }
        return dense_complex_[which];
    }

    const SparseOperands* ops_;
    Shift shift_;
    bool factored_ = false;
    bool singular_ = false;
    linear::KluSolver real_;
    linear::KluSolverZ complex_;
    std::vector<double> real_values_;
    std::vector<Complex> complex_values_;
    DenseSafeSolver<double> dense_real_[2];
    DenseSafeSolver<Complex> dense_complex_[2];
    bool dense_ready_[2] = {false, false};
};

/// y = -(C v) or y = -(C^T v) for every column of the n x cols block (column-major).
std::vector<double> negated_c_times(const SparseOperands& ops, const double* block, std::int32_t cols, bool transpose) {
    const linear::CscPattern& pattern = *ops.pattern;
    const auto n = static_cast<std::size_t>(pattern.n);
    std::vector<double> out(n * static_cast<std::size_t>(cols), 0.0);
    for (std::int32_t q = 0; q < cols; ++q) {
        const double* v = block + static_cast<std::size_t>(q) * n;
        double* y = out.data() + static_cast<std::size_t>(q) * n;
        for (std::size_t col = 0; col < n; ++col) {
            const auto begin = static_cast<std::size_t>(pattern.col_ptr[col]);
            const auto end = static_cast<std::size_t>(pattern.col_ptr[col + 1]);
            if (transpose) {
                double sum = 0.0;
                for (std::size_t p = begin; p < end; ++p) {
                    sum += ops.c[p] * v[static_cast<std::size_t>(pattern.row_idx[p])];
                }
                y[col] = sum;
            } else {
                const double vj = v[col];
                for (std::size_t p = begin; p < end; ++p) {
                    y[static_cast<std::size_t>(pattern.row_idx[p])] += ops.c[p] * vj;
                }
            }
        }
        for (std::size_t i = 0; i < n; ++i) {
            y[i] = -y[i];
        }
    }
    return out;
}

void validate(const SparseOperands& operands, const double* inputs, std::int32_t input_cols, const double* outputs,
              std::int32_t output_cols) {
    if (operands.pattern == nullptr) {
        bad_input("missing CSC pattern");
    }
    const linear::CscPattern& pattern = *operands.pattern;
    if (pattern.n <= 0) {
        bad_input("the system is empty (n = 0)");
    }
    if (pattern.col_ptr.size() != static_cast<std::size_t>(pattern.n) + 1) {
        bad_input("col_ptr must have n + 1 entries");
    }
    if (pattern.nnz() > 0 && (operands.g == nullptr || operands.c == nullptr)) {
        bad_input("missing G or C values");
    }
    if (input_cols < 0 || output_cols < 0) {
        bad_input("negative block width");
    }
    if (input_cols > 0 && inputs == nullptr) {
        bad_input("missing input block");
    }
    if (output_cols > 0 && outputs == nullptr) {
        bad_input("missing output block");
    }
}

}  // namespace

std::int32_t orthogonalize_block(std::vector<double>& basis, std::int32_t n, std::int32_t& basis_cols,
                                 const double* candidate, std::int32_t cols, std::int32_t max_columns,
                                 double tolerance) {
    if (basis_cols >= max_columns || cols <= 0) {
        return 0;
    }
    const auto un = static_cast<std::size_t>(n);
    basis.reserve(un * static_cast<std::size_t>(max_columns));
    Eigen::VectorXd v(n);
    Eigen::VectorXd coefficients;
    std::int32_t accepted = 0;
    for (std::int32_t col = 0; col < cols; ++col) {
        for (std::size_t i = 0; i < un; ++i) {
            v[static_cast<Eigen::Index>(i)] = candidate[static_cast<std::size_t>(col) * un + i];
        }
        if (basis_cols > 0) {
            // vector - basis @ (basis.T @ vector)
            const Eigen::Map<const Eigen::MatrixXd> B(basis.data(), n, basis_cols);
            coefficients.noalias() = B.transpose() * v;
            const Eigen::VectorXd projection = B * coefficients;
            v -= projection;
        }
        const double norm_value = std::sqrt(v.squaredNorm());
        if (norm_value <= tolerance) {  // a NaN norm is accepted, as in the Python loop
            continue;
        }
        const std::size_t offset = basis.size();
        basis.resize(offset + un);
        for (std::size_t i = 0; i < un; ++i) {
            basis[offset + i] = v[static_cast<Eigen::Index>(i)] / norm_value;
        }
        ++basis_cols;
        ++accepted;
        if (basis_cols >= max_columns) {
            break;
        }
    }
    return accepted;
}

KrylovBasisResult krylov_basis(const SparseOperands& operands, const double* inputs, std::int32_t input_cols,
                               const double* outputs, std::int32_t output_cols, const std::vector<Shift>& shifts,
                               std::int64_t order, int num_threads) {
    validate(operands, inputs, input_cols, outputs, output_cols);
    const std::int32_t n = operands.pattern->n;
    const auto un = static_cast<std::size_t>(n);
    const std::int32_t target =
        static_cast<std::int32_t>((std::max)(std::int64_t{1}, (std::min)(order, static_cast<std::int64_t>(n))));

    KrylovBasisResult result;
    result.n = n;
    std::vector<double>& basis = result.basis;
    basis.reserve(un * static_cast<std::size_t>(target));
    std::int32_t k = 0;
    if (input_cols > 0) {
        orthogonalize_block(basis, n, k, inputs, input_cols, target);
    }
    orthogonalize_block(basis, n, k, outputs, output_cols, target);

    std::vector<std::unique_ptr<ShiftSystem>> systems;
    systems.reserve(shifts.size());
    for (const Shift& shift : shifts) {
        systems.push_back(std::make_unique<ShiftSystem>(operands, shift));
    }
    const int threads = n >= kMorParallelMinUnknowns ? effective_threads(num_threads) : 1;
    if (threads > 1 && systems.size() > 1) {
        // Factor every shift up front, one system per thread; the sweep below is sequential.
        result.threads_used = parallel_for(static_cast<std::int64_t>(systems.size()), threads,
                                           [&systems](std::int64_t begin, std::int64_t end) {
                                               for (std::int64_t i = begin; i < end; ++i) {
                                                   systems[static_cast<std::size_t>(i)]->factor();
                                               }
                                           });
    }

    struct State {
        bool adjoint;
        std::size_t shift;
        std::int32_t first;
        std::int32_t count;
    };
    std::vector<State> states;
    auto block_copy = [un](const double* block, std::int32_t cols) {
        return std::vector<double>(block, block + un * static_cast<std::size_t>(cols));
    };
    auto finish = [&]() {
        for (const auto& system : systems) {
            if (system->factored()) {
                ++result.factorizations;
                result.lstsq_shifts += system->singular() ? 1 : 0;
            }
        }
        if (k == 0) {
            basis.assign(un, 0.0);
            basis[0] = 1.0;
            k = 1;
        }
        result.columns = k;
        return std::move(result);
    };

    for (std::size_t s = 0; s < systems.size(); ++s) {
        ShiftSystem& system = *systems[s];
        std::int32_t cand_cols = 0;
        if (input_cols > 0) {
            const std::vector<double> cand = system.solve(false, block_copy(inputs, input_cols), input_cols, cand_cols);
            ++result.solves;
            const std::int32_t accepted = orthogonalize_block(basis, n, k, cand.data(), cand_cols, target);
            if (accepted > 0) {
                states.push_back({false, s, k - accepted, accepted});
            }
        }
        const std::vector<double> cand_adj =
            system.solve(true, block_copy(outputs, output_cols), output_cols, cand_cols);
        ++result.solves;
        const std::int32_t accepted_adj = orthogonalize_block(basis, n, k, cand_adj.data(), cand_cols, target);
        if (accepted_adj > 0) {
            states.push_back({true, s, k - accepted_adj, accepted_adj});
        }
        if (k >= target) {
            return finish();
        }
    }

    std::int32_t previous_columns = -1;
    while (k < target && k != previous_columns) {
        previous_columns = k;
        for (State& state : states) {
            const double* block = basis.data() + static_cast<std::size_t>(state.first) * un;
            std::vector<double> rhs = negated_c_times(operands, block, state.count, state.adjoint);
            std::int32_t cand_cols = 0;
            const std::vector<double> cand =
                systems[state.shift]->solve(state.adjoint, std::move(rhs), state.count, cand_cols);
            ++result.solves;
            const std::int32_t accepted = orthogonalize_block(basis, n, k, cand.data(), cand_cols, target);
            if (accepted > 0) {
                state.first = k - accepted;
                state.count = accepted;
            }
            if (k >= target) {
                break;
            }
        }
    }
    return finish();
}

}  // namespace mna::mor
