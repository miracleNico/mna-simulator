// Python bindings of the linear-algebra unit: matrix classification, direct
// solves (KLU / dense LU, real and complex), the Krylov solvers and a reusable
// LinearSolver object. Every heavy call releases the GIL.
//
// Arrays: CSC structure as int32 (col_ptr of size n + 1, row_idx), values and
// right-hand sides as float64 or complex128 with any strides. Right-hand sides
// of direct solves may be 1-D (n,) or 2-D (n, k) in any memory order; results
// come back with the same shape (2-D results are Fortran-ordered).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "mna/error.hpp"
#include "mna/linear/classify.hpp"
#include "mna/linear/dense_solver.hpp"
#include "mna/linear/klu_solver.hpp"
#include "mna/linear/krylov.hpp"
#include "mna/linear/linear_solver.hpp"
#include "mna/linear/sparse.hpp"
#include "mna/threads.hpp"
#include "registry.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using mna::linear::Complex;
using mna::linear::CscPattern;

using IndexArray = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using AnyArray = nb::ndarray<nb::ro, nb::device::cpu>;

[[noreturn]] void bad_input(const std::string& message) { throw mna::Error(mna::ErrorCode::NetlistFatal, message); }

bool is_complex_array(const AnyArray& a, const char* name) {
    if (a.dtype() == nb::dtype<double>()) {
        return false;
    }
    if (a.dtype() == nb::dtype<Complex>()) {
        return true;
    }
    throw nb::type_error((std::string(name) + " must be a float64 or complex128 array").c_str());
}

std::int32_t dimension_of(const IndexArray& col_ptr, const char* where) {
    if (col_ptr.shape(0) == 0) {
        bad_input(std::string(where) + ": col_ptr must have n + 1 entries");
    }
    const std::size_t n = col_ptr.shape(0) - 1;
    if (n > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max() - 1)) {
        bad_input(std::string(where) + ": matrix too large for 32-bit indices");
    }
    return static_cast<std::int32_t>(n);
}

CscPattern pattern_of(const IndexArray& col_ptr, const IndexArray& row_idx, const char* where) {
    const std::int32_t n = dimension_of(col_ptr, where);
    return CscPattern::from_arrays(n, col_ptr.data(), col_ptr.shape(0), row_idx.data(), row_idx.shape(0));
}

// Element (i, j) of a 1-D (j == 0) or 2-D array of T (double or Complex) with arbitrary strides.
template <class T>
T element(const AnyArray& a, std::size_t i, std::size_t j) {
    const std::int64_t offset = static_cast<std::int64_t>(i) * a.stride(0) +
                                (a.ndim() > 1 ? static_cast<std::int64_t>(j) * a.stride(1) : 0);
    return static_cast<const T*>(a.data())[offset];
}

// Copies a real or complex array into a T vector: 2-D arrays column-major
// (rows x cols), anything 1-D as is. Complex input into a real vector is refused.
template <class T>
std::vector<T> read_columns(const AnyArray& a, const char* name, std::size_t& rows, std::size_t& cols) {
    if (a.ndim() == 1) {
        rows = a.shape(0);
        cols = 1;
    } else if (a.ndim() == 2) {
        rows = a.shape(0);
        cols = a.shape(1);
    } else {
        bad_input(std::string(name) + " must be 1-D or 2-D");
    }
    const bool complex_input = is_complex_array(a, name);
    if constexpr (std::is_same_v<T, double>) {
        if (complex_input) {
            throw nb::type_error((std::string(name) + ": complex values for a real operation").c_str());
        }
    }
    std::vector<T> out(rows * cols);
    for (std::size_t j = 0; j < cols; ++j) {
        for (std::size_t i = 0; i < rows; ++i) {
            if constexpr (std::is_same_v<T, double>) {
                out[i + j * rows] = element<double>(a, i, j);
            } else {
                out[i + j * rows] = complex_input ? element<Complex>(a, i, j) : T(element<double>(a, i, j));
            }
        }
    }
    return out;
}

// Flattens any 1-D or 2-D array in C order (NumPy's reshape(-1)).
template <class T>
std::vector<T> read_flat(const AnyArray& a, const char* name) {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<T> colmajor = read_columns<T>(a, name, rows, cols);
    if (cols == 1) {
        return colmajor;
    }
    std::vector<T> flat(rows * cols);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            flat[i * cols + j] = colmajor[i + j * rows];
        }
    }
    return flat;
}

// Hands a column-major vector to NumPy without copying: shape (rows,) when
// one_d, else (rows, cols) Fortran-ordered. Needs the GIL.
template <class T>
nb::object to_numpy(std::vector<T>&& values, std::size_t rows, std::size_t cols, bool one_d) {
    auto* holder = new std::vector<T>(std::move(values));
    if (holder->empty()) {
        holder->reserve(1);  // keep data() non-null for zero-length arrays
    }
    nb::capsule owner(holder, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    using Out = nb::ndarray<nb::numpy, T>;
    if (one_d) {
        const std::size_t shape[1] = {rows};
        return nb::cast(Out(holder->data(), 1, shape, owner));
    }
    const std::size_t shape[2] = {rows, cols};
    const std::int64_t strides[2] = {1, static_cast<std::int64_t>(rows)};
    return nb::cast(Out(holder->data(), 2, shape, owner, strides));
}

// ---------------------------------------------------------------------------
// classify_matrix
// ---------------------------------------------------------------------------

std::string classify_matrix(IndexArray col_ptr, IndexArray row_idx, AnyArray vals, double symmetry_tol) {
    CscPattern pattern = pattern_of(col_ptr, row_idx, "classify_matrix");
    const bool complex_values = is_complex_array(vals, "vals");
    mna::linear::MatrixKind kind = mna::linear::MatrixKind::General;
    if (complex_values) {
        const std::vector<Complex> values = read_flat<Complex>(vals, "vals");
        nb::gil_scoped_release release;
        const auto A = mna::linear::CsrMatrix<Complex>::from_csc(pattern, values.data(), values.size());
        kind = mna::linear::classify_matrix(A, pattern, values.data(), symmetry_tol);
    } else {
        const std::vector<double> values = read_flat<double>(vals, "vals");
        nb::gil_scoped_release release;
        const auto A = mna::linear::CsrMatrix<double>::from_csc(pattern, values.data(), values.size());
        kind = mna::linear::classify_matrix(A, pattern, values.data(), symmetry_tol);
    }
    return mna::linear::matrix_kind_name(kind);
}

// ---------------------------------------------------------------------------
// direct_solve
// ---------------------------------------------------------------------------

template <class T>
void factor_and_solve(const CscPattern& pattern, const std::vector<T>& values, std::vector<T>& rhs, std::size_t nrhs) {
    auto solver = mna::linear::make_linear_solver<T>(pattern);
    solver->factor(values.data(), values.size());
    solver->solve(rhs.data(), rhs.size(), static_cast<std::int32_t>(nrhs));
}

nb::object direct_solve(IndexArray col_ptr, IndexArray row_idx, AnyArray vals, AnyArray b) {
    CscPattern pattern = pattern_of(col_ptr, row_idx, "direct_solve");
    const auto n = static_cast<std::size_t>(pattern.n);
    const bool complex_values = is_complex_array(vals, "vals");
    const bool complex_rhs = is_complex_array(b, "b");
    const bool one_d = b.ndim() == 1;
    std::size_t rows = 0;
    std::size_t cols = 0;

    if (complex_values) {
        const std::vector<Complex> values = read_flat<Complex>(vals, "vals");
        std::vector<Complex> rhs = read_columns<Complex>(b, "b", rows, cols);
        if (rows != n) {
            bad_input("direct_solve: b has " + std::to_string(rows) + " rows, expected n = " + std::to_string(n));
        }
        {
            nb::gil_scoped_release release;
            factor_and_solve(pattern, values, rhs, cols);
        }
        return to_numpy(std::move(rhs), rows, cols, one_d);
    }

    const std::vector<double> values = read_flat<double>(vals, "vals");
    if (!complex_rhs) {
        std::vector<double> rhs = read_columns<double>(b, "b", rows, cols);
        if (rows != n) {
            bad_input("direct_solve: b has " + std::to_string(rows) + " rows, expected n = " + std::to_string(n));
        }
        {
            nb::gil_scoped_release release;
            factor_and_solve(pattern, values, rhs, cols);
        }
        return to_numpy(std::move(rhs), rows, cols, one_d);
    }

    // Real matrix, complex right-hand side: one real factorization, real and
    // imaginary parts solved as 2k right-hand sides (what a complex LU of a
    // real matrix computes, without the complex arithmetic).
    const std::vector<Complex> crhs = read_columns<Complex>(b, "b", rows, cols);
    if (rows != n) {
        bad_input("direct_solve: b has " + std::to_string(rows) + " rows, expected n = " + std::to_string(n));
    }
    std::vector<double> split(2 * rows * cols);
    for (std::size_t j = 0; j < cols; ++j) {
        for (std::size_t i = 0; i < rows; ++i) {
            split[i + j * rows] = crhs[i + j * rows].real();
            split[i + (cols + j) * rows] = crhs[i + j * rows].imag();
        }
    }
    {
        nb::gil_scoped_release release;
        factor_and_solve(pattern, values, split, 2 * cols);
    }
    std::vector<Complex> x(rows * cols);
    for (std::size_t j = 0; j < cols; ++j) {
        for (std::size_t i = 0; i < rows; ++i) {
            x[i + j * rows] = Complex(split[i + j * rows], split[i + (cols + j) * rows]);
        }
    }
    return to_numpy(std::move(x), rows, cols, one_d);
}

// ---------------------------------------------------------------------------
// krylov_solve
// ---------------------------------------------------------------------------

template <class T>
nb::tuple krylov_solve_typed(const CscPattern& pattern, const AnyArray& vals, const AnyArray& b,
                             const std::optional<AnyArray>& x0, const std::string& method, double tol,
                             std::int64_t max_iter, std::int64_t restart, int num_threads, double symmetry_tol) {
    using namespace mna::linear;
    const auto n = static_cast<std::size_t>(pattern.n);
    const std::vector<T> values = read_flat<T>(vals, "vals");
    const std::vector<T> rhs = read_flat<T>(b, "b");
    if (rhs.size() != n) {
        bad_input("krylov_solve: b has " + std::to_string(rhs.size()) + " entries, expected n = " + std::to_string(n));
    }
    std::vector<T> x(n, T{});
    if (x0.has_value()) {
        x = read_flat<T>(*x0, "x0");
        if (x.size() != n) {
            bad_input("krylov_solve: x0 has " + std::to_string(x.size()) + " entries, expected n = " +
                      std::to_string(n));
        }
    }
    const bool classify = method_is_auto(method);
    MatrixKind kind = MatrixKind::General;
    KrylovInfo info;
    std::int64_t nnz = 0;
    {
        nb::gil_scoped_release release;
        const CsrMatrix<T> A = CsrMatrix<T>::from_csc(pattern, values.data(), values.size());
        nnz = A.nnz();
        if (classify) {
            kind = classify_matrix(A, pattern, values.data(), symmetry_tol);
        }
        const KrylovRoute route = krylov_route(kind, method);
        KrylovSettings settings;
        settings.tol = tol;
        settings.max_iter = max_iter;
        settings.rank = restart;
        settings.num_threads = mna::effective_threads(num_threads);
        info = krylov_solve(A, rhs.data(), x.data(), route, settings);
    }

    nb::dict d;
    d["method"] = info.method;
    d["engine"] = "mna_core";
    d["kernel"] = krylov_algorithm_name(info.algorithm);
    d["route"] = krylov_route_name(info.route);
    if (classify) {
        d["matrix_kind"] = matrix_kind_name(kind);
    }
    d["converged"] = info.converged;
    d["iterations"] = info.iterations;
    d["relative_residual"] = info.relative_residual;
    d["iteration_budget"] = info.iteration_budget;
    if (info.effective_restart >= 0) {
        d["effective_restart"] = info.effective_restart;
    } else if (info.scipy_semantics) {
        d["effective_restart"] = nb::none();
    }
    if (info.scipy_semantics) {
        d["exit_code"] = info.exit_code;
    }
    d["scipy_semantics"] = info.scipy_semantics;
    d["matrix_dimension"] = static_cast<std::int64_t>(n);
    d["matrix_nnz"] = nnz;
    d["threads_used"] = info.threads_used;
    return nb::make_tuple(to_numpy(std::move(x), n, 1, true), d);
}

nb::tuple krylov_solve(IndexArray col_ptr, IndexArray row_idx, AnyArray vals, AnyArray b, const std::string& method,
                       double tol, std::int64_t max_iter, std::int64_t restart, std::optional<AnyArray> x0,
                       int num_threads, double symmetry_tol) {
    CscPattern pattern = pattern_of(col_ptr, row_idx, "krylov_solve");
    const bool complex_system = is_complex_array(vals, "vals") || is_complex_array(b, "b") ||
                                (x0.has_value() && is_complex_array(*x0, "x0"));
    if (complex_system) {
        return krylov_solve_typed<Complex>(pattern, vals, b, x0, method, tol, max_iter, restart, num_threads,
                                           symmetry_tol);
    }
    return krylov_solve_typed<double>(pattern, vals, b, x0, method, tol, max_iter, restart, num_threads,
                                      symmetry_tol);
}

// ---------------------------------------------------------------------------
// LinearSolver
// ---------------------------------------------------------------------------

class PyLinearSolver {
public:
    PyLinearSolver(IndexArray col_ptr, IndexArray row_idx, bool complex_values, bool dense)
        : complex_(complex_values) {
        CscPattern pattern = pattern_of(col_ptr, row_idx, "LinearSolver");
        const auto kind = dense ? mna::linear::SolverKind::Dense : mna::linear::SolverKind::Auto;
        nb::gil_scoped_release release;
        if (complex_) {
            complex_solver_ = mna::linear::make_linear_solver<Complex>(pattern, kind);
        } else {
            real_solver_ = mna::linear::make_linear_solver<double>(pattern, kind);
        }
        nnz_ = pattern.nnz();
    }

    void factor(const AnyArray& values) { factor_impl(values, false); }
    std::string refactor(const AnyArray& values) { return factor_impl(values, true); }

    nb::object solve(const AnyArray& b) { return solve_impl(b, false, false); }
    nb::object tsolve(const AnyArray& b, bool conjugate) { return solve_impl(b, true, conjugate); }

    double rcond() {
        nb::gil_scoped_release release;
        const std::lock_guard<std::mutex> lock(mutex_);
        return complex_ ? complex_solver_->rcond() : real_solver_->rcond();
    }

    std::int64_t nnz_lu() const { return complex_ ? complex_solver_->nnz_lu() : real_solver_->nnz_lu(); }
    std::int64_t nnz() const { return nnz_; }
    std::int32_t n() const { return complex_ ? complex_solver_->size() : real_solver_->size(); }
    bool is_complex() const { return complex_; }
    bool factored() const { return complex_ ? complex_solver_->factored() : real_solver_->factored(); }
    std::string kind() const { return complex_ ? complex_solver_->kind() : real_solver_->kind(); }
    nb::object path_of_last_refactor() const {
        if (!last_path_.has_value()) {
            return nb::none();
        }
        return nb::str(mna::linear::factor_path_name(*last_path_));
    }

private:
    std::string factor_impl(const AnyArray& values, bool refactor) {
        const char* name = "values";
        std::optional<mna::linear::FactorPath> path;
        if (complex_) {
            const std::vector<Complex> v = read_flat<Complex>(values, name);
            nb::gil_scoped_release release;
            const std::lock_guard<std::mutex> lock(mutex_);
            if (refactor) {
                path = complex_solver_->refactor(v.data(), v.size());
            } else {
                complex_solver_->factor(v.data(), v.size());
            }
        } else {
            const std::vector<double> v = read_flat<double>(values, name);
            nb::gil_scoped_release release;
            const std::lock_guard<std::mutex> lock(mutex_);
            if (refactor) {
                path = real_solver_->refactor(v.data(), v.size());
            } else {
                real_solver_->factor(v.data(), v.size());
            }
        }
        if (!path.has_value()) {
            return "full_factor";
        }
        last_path_ = path;
        return mna::linear::factor_path_name(*path);
    }

    template <class T, class Solver>
    void run(Solver& solver, std::vector<T>& rhs, std::size_t cols, bool transpose, bool conjugate) {
        nb::gil_scoped_release release;
        const std::lock_guard<std::mutex> lock(mutex_);
        if (transpose) {
            solver.tsolve(rhs.data(), rhs.size(), static_cast<std::int32_t>(cols), conjugate);
        } else {
            solver.solve(rhs.data(), rhs.size(), static_cast<std::int32_t>(cols));
        }
    }

    nb::object solve_impl(const AnyArray& b, bool transpose, bool conjugate) {
        const bool one_d = b.ndim() == 1;
        std::size_t rows = 0;
        std::size_t cols = 0;
        if (complex_) {
            std::vector<Complex> rhs = read_columns<Complex>(b, "b", rows, cols);
            run<Complex>(*complex_solver_, rhs, cols, transpose, conjugate);
            return to_numpy(std::move(rhs), rows, cols, one_d);
        }
        if (!is_complex_array(b, "b")) {
            std::vector<double> rhs = read_columns<double>(b, "b", rows, cols);
            run<double>(*real_solver_, rhs, cols, transpose, conjugate);
            return to_numpy(std::move(rhs), rows, cols, one_d);
        }
        // Real factorization, complex right-hand side: solve both parts.
        const std::vector<Complex> crhs = read_columns<Complex>(b, "b", rows, cols);
        std::vector<double> split(2 * rows * cols);
        for (std::size_t j = 0; j < cols; ++j) {
            for (std::size_t i = 0; i < rows; ++i) {
                split[i + j * rows] = crhs[i + j * rows].real();
                split[i + (cols + j) * rows] = crhs[i + j * rows].imag();
            }
        }
        run<double>(*real_solver_, split, 2 * cols, transpose, false);
        std::vector<Complex> x(rows * cols);
        for (std::size_t j = 0; j < cols; ++j) {
            for (std::size_t i = 0; i < rows; ++i) {
                // A^H = A^T for a real A, so a conjugate transpose solve needs no extra care.
                x[i + j * rows] = Complex(split[i + j * rows], split[i + (cols + j) * rows]);
            }
        }
        return to_numpy(std::move(x), rows, cols, one_d);
    }

    bool complex_ = false;
    std::int64_t nnz_ = 0;
    std::unique_ptr<mna::linear::LinearSolver<double>> real_solver_;
    std::unique_ptr<mna::linear::LinearSolver<Complex>> complex_solver_;
    std::optional<mna::linear::FactorPath> last_path_;
    std::mutex mutex_;
};

void bind_linear(nb::module_& m) {
    m.attr("DENSE_MAX_N") = mna::linear::kDenseMaxN;
    m.attr("PARALLEL_SPMV_MIN_N") = mna::linear::kParallelSpmvMinN;
    m.attr("SCIPY_ENGINE_MIN_N") = mna::linear::kScipyEngineMinN;
    m.attr("CHOLESKY_MAX_N") = mna::linear::kCholeskyMaxN;
    m.attr("REFACTOR_RCOND_FLOOR") = mna::linear::kRefactorRcondFloor;

    m.def("classify_matrix", &classify_matrix, "col_ptr"_a, "row_idx"_a, "vals"_a,
          "symmetry_tol"_a = mna::linear::kDefaultSymmetryTol,
          "Classify a square CSC matrix for Krylov method selection: 'general', 'symmetric' or\n"
          "'positive_definite' (sparse rule of solvers.classify_krylov_matrix).");
    m.def("direct_solve", &direct_solve, "col_ptr"_a, "row_idx"_a, "vals"_a, "b"_a,
          "Solve A X = B for a square CSC matrix (real or complex values) with the direct solver the\n"
          "dense/KLU policy selects. B is (n,) or (n, k) in any memory order; the result has B's shape and\n"
          "is complex when A or B is. Raises CoreError JACOBIAN_SINGULAR for a singular matrix.");
    m.def("krylov_solve", &krylov_solve, "col_ptr"_a, "row_idx"_a, "vals"_a, "b"_a, "method_route"_a = "auto",
          "tol"_a = 1e-9, "max_iter"_a = 2000, "restart"_a = 80, "x0"_a = nb::none(), "num_threads"_a = 1,
          "symmetry_tol"_a = mna::linear::kDefaultSymmetryTol,
          "Solve A x = b with the Krylov kernel the reference would run for the route of `method_route`\n"
          "(a route name or any solvers._krylov_route alias; 'auto' classifies the matrix first).\n"
          "`restart` is the resolved Krylov rank. Returns (x, info).");

    nb::class_<PyLinearSolver>(m, "LinearSolver",
                               "Direct solver bound to one CSC sparsity pattern: factor / refactor (pivot reuse with\n"
                               "fallback) / solve / tsolve. KLU unless dense=True (Eigen PartialPivLU).")
        .def(nb::init<IndexArray, IndexArray, bool, bool>(), "col_ptr"_a, "row_idx"_a, "complex"_a = false,
             "dense"_a = false)
        .def("factor", &PyLinearSolver::factor, "values"_a, "Numeric factorization with partial pivoting.")
        .def("refactor", &PyLinearSolver::refactor, "values"_a,
             "Refactor reusing the pivot order; returns 'refactor' or 'full_factor' (fallback).")
        .def("solve", &PyLinearSolver::solve, "b"_a, "Solve A X = B; B is (n,) or (n, k).")
        .def("tsolve", &PyLinearSolver::tsolve, "b"_a, "conjugate"_a = false,
             "Solve A^T X = B (A^H X = B with conjugate=True).")
        .def("rcond", &PyLinearSolver::rcond, "Pivot-ratio condition estimate min|u_ii| / max|u_ii|.")
        .def_prop_ro("nnz_lu", &PyLinearSolver::nnz_lu)
        .def_prop_ro("nnz", &PyLinearSolver::nnz)
        .def_prop_ro("n", &PyLinearSolver::n)
        .def_prop_ro("is_complex", &PyLinearSolver::is_complex)
        .def_prop_ro("factored", &PyLinearSolver::factored)
        .def_prop_ro("kind", &PyLinearSolver::kind)
        .def_prop_ro("path_of_last_refactor", &PyLinearSolver::path_of_last_refactor);
}

const mna::bindings::Registrar reg_linear{"linear", &bind_linear};

}  // namespace
