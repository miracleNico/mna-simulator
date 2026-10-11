// Bindings of the harmonic-balance unit (mna/analyses/hb.hpp): hb_solve,
// hb_reconstruct, the Fourier matrices and the sample grid. Every call
// releases the GIL for its native work.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mna/analyses/hb.hpp"
#include "mna/circuit/circuit.hpp"
#include "mna/error.hpp"
#include "mna/expr/program.hpp"
#include "mna/linear/classify.hpp"
#include "mna/linear/krylov.hpp"
#include "mna/threads.hpp"
#include "registry.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

namespace hb = mna::analyses;
using mna::circuit::Circuit;

template <class T>
using In1D = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
nb::ndarray<nb::numpy, T> to_numpy(std::vector<T>&& values, std::initializer_list<std::size_t> shape) {
    auto* holder = new std::vector<T>(std::move(values));
    if (holder->empty()) {
        holder->reserve(1);  // keep data() non-null for zero-length arrays
    }
    nb::capsule owner(holder, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    return nb::ndarray<nb::numpy, T>(holder->data(), shape, owner);
}

[[noreturn]] void bad_input(const std::string& message) { throw mna::Error(mna::ErrorCode::NetlistFatal, message); }

nb::tuple fourier_matrices(std::int64_t harmonics, std::int64_t num_points) {
    hb::FourierMatrices m;
    {
        nb::gil_scoped_release release;
        m = hb::hb_fourier_matrices(harmonics, num_points);
    }
    const auto S = static_cast<std::size_t>(m.rows);
    const auto K = static_cast<std::size_t>(m.points);
    return nb::make_tuple(to_numpy(std::move(m.gamma), {S, K}), to_numpy(std::move(m.gamma_inv), {K, S}));
}

nb::ndarray<nb::numpy, double> time_points(double omega0, std::int64_t num_points) {
    std::vector<double> t = hb::hb_time_points(omega0, num_points);
    const std::size_t size = t.size();
    return to_numpy(std::move(t), {size});
}

nb::ndarray<nb::numpy, double> reconstruct(In1D<double> x_bar, std::int64_t n, std::int64_t harmonics, double omega0,
                                           In1D<double> t, int num_threads) {
    if (n < 0 || harmonics < 0) {
        throw std::invalid_argument("hb_reconstruct: num_vars and H must be non-negative");
    }
    const std::int64_t K = 2 * harmonics + 1;
    if (static_cast<std::int64_t>(x_bar.shape(0)) != n * K) {
        throw std::invalid_argument("hb_reconstruct: cannot reshape array of size " + std::to_string(x_bar.shape(0)) +
                                    " into shape (" + std::to_string(n) + "," + std::to_string(K) + ")");
    }
    const auto rows = static_cast<std::size_t>(n);
    const std::size_t cols = t.shape(0);
    std::vector<double> out(rows * cols);
    {
        nb::gil_scoped_release release;
        hb::hb_reconstruct(x_bar.data(), n, harmonics, omega0, t.data(), static_cast<std::int64_t>(cols),
                           mna::effective_threads(num_threads), out.data());
    }
    return to_numpy(std::move(out), {rows, cols});
}

nb::ndarray<nb::numpy, double> jacobian_block(std::int64_t harmonics, In1D<double> g) {
    if (harmonics < 0) {
        throw std::invalid_argument("hb_jacobian_block: harmonics must be non-negative");
    }
    const std::int64_t K = 2 * harmonics + 1;
    if (static_cast<std::int64_t>(g.shape(0)) != K) {
        throw std::invalid_argument("hb_jacobian_block: g must have 2 H + 1 samples");
    }
    const auto k = static_cast<std::size_t>(K);
    std::vector<double> block(k * k);
    {
        nb::gil_scoped_release release;
        const hb::FourierMatrices m = hb::hb_fourier_matrices(harmonics, K);
        std::vector<double> scratch(k * static_cast<std::size_t>(hb::kHbBlockChunk));
        std::vector<double> gram(k * k);
        for (std::int64_t c0 = 0; c0 < K; c0 += hb::kHbBlockChunk) {
            hb::hb_gram_columns(m, g.data(), c0, (std::min)(K, c0 + hb::kHbBlockChunk), scratch.data(), gram.data());
        }
        const double dc = 1.0 / static_cast<double>(K);
        const double ac = 2.0 / static_cast<double>(K);
        for (std::size_t a = 0; a < k; ++a) {
            for (std::size_t b = 0; b < k; ++b) {
                const double upper = a <= b ? gram[a + b * k] : gram[b + a * k];
                block[a * k + b] = (a == 0 ? dc : ac) * upper;
            }
        }
    }
    return to_numpy(std::move(block), {k, k});
}

nb::tuple btf_solve(In1D<std::int32_t> col_ptr, In1D<std::int32_t> row_idx, In1D<double> values, In1D<double> b,
                    int num_threads) {
    if (col_ptr.shape(0) == 0) {
        bad_input("btf_solve: col_ptr must have n + 1 entries");
    }
    const auto n = static_cast<std::int32_t>(col_ptr.shape(0) - 1);
    if (b.shape(0) != static_cast<std::size_t>(n)) {
        bad_input("btf_solve: b must have n entries");
    }
    if (values.shape(0) != row_idx.shape(0)) {
        bad_input("btf_solve: values and row_idx differ in length");
    }
    std::vector<double> x(b.data(), b.data() + b.shape(0));
    hb::BtfSolver solver;
    {
        nb::gil_scoped_release release;
        solver.analyze(n, col_ptr.data(), row_idx.data());
        solver.factor(values.data(), mna::effective_threads(num_threads));
        solver.solve(x.data());
    }
    nb::dict info;
    info["blocks"] = solver.blocks();
    info["dense_blocks"] = solver.dense_blocks();
    info["klu_blocks"] = solver.klu_blocks();
    info["singleton_blocks"] = solver.singleton_blocks();
    info["largest_block"] = solver.largest_block();
    info["structural_rank"] = solver.structural_rank();
    info["nnz_factors"] = solver.nnz_factors();
    info["threads_used"] = solver.threads_used();
    const std::size_t size = x.size();
    return nb::make_tuple(to_numpy(std::move(x), {size}), info);
}

nb::dict krylov_record(const hb::HbKrylovRecord& r) {
    nb::dict d;
    d["method"] = r.info.method;
    d["kernel"] = mna::linear::krylov_algorithm_name(r.info.algorithm);
    d["route"] = mna::linear::krylov_route_name(r.info.route);
    d["matrix_kind"] = mna::linear::matrix_kind_name(r.matrix_kind);
    d["converged"] = r.info.converged;
    d["iterations"] = r.info.iterations;
    d["relative_residual"] = r.info.relative_residual;
    d["iteration_budget"] = r.info.iteration_budget;
    if (r.info.effective_restart >= 0) {
        d["effective_restart"] = r.info.effective_restart;
    } else {
        d["effective_restart"] = nb::none();
    }
    d["exit_code"] = r.info.exit_code;
    d["scipy_semantics"] = r.info.scipy_semantics;
    d["matrix_nnz"] = r.matrix_nnz;
    d["used_direct_fallback"] = r.used_direct_fallback;
    d["threads_used"] = r.info.threads_used;
    return d;
}

nb::dict stats_dict(const hb::HbStats& s) {
    nb::dict d;
    d["linear"] = s.linear;
    d["harmonics"] = s.harmonics;
    d["samples"] = s.samples;
    d["dimension"] = s.dimension;
    d["newton_iterations"] = s.newton_iterations;
    d["linear_solves"] = s.linear_solves;
    d["factorizations"] = s.factorizations;
    d["residual_norm"] = s.residual_norm;
    d["y_nnz"] = s.y_nnz;
    d["device_slots"] = s.device_slots;
    d["matrix_nnz"] = s.matrix_nnz;
    d["stored_nnz"] = s.stored_nnz;
    d["btf_blocks"] = s.btf_blocks;
    d["dense_blocks"] = s.dense_blocks;
    d["klu_blocks"] = s.klu_blocks;
    d["singleton_blocks"] = s.singleton_blocks;
    d["largest_block"] = s.largest_block;
    d["factor_nnz"] = s.factor_nnz;
    d["memory_bytes"] = s.memory_bytes;
    d["threads_used"] = s.threads_used;
    nb::dict timings;
    timings["setup"] = s.setup_seconds;
    timings["evaluate"] = s.evaluate_seconds;
    timings["jacobian"] = s.jacobian_seconds;
    timings["factor"] = s.factor_seconds;
    timings["solve"] = s.solve_seconds;
    d["timings_s"] = timings;
    nb::list krylov;
    for (const hb::HbKrylovRecord& r : s.krylov) {
        krylov.append(krylov_record(r));
    }
    d["krylov"] = krylov;
    return d;
}

nb::tuple solve(const Circuit& circuit, double omega0, std::int64_t harmonics, In1D<double> b_dc,
                In1D<std::int64_t> source_rows, nb::handle source_programs, double gmin, bool nonlinear,
                std::int64_t max_iter, double tol, bool use_krylov, const std::string& krylov_method,
                double krylov_tol, std::int64_t krylov_max_iter, std::int64_t krylov_rank, int num_threads,
                std::int64_t max_nnz) {
    const auto n = static_cast<std::size_t>(circuit.n());
    if (b_dc.shape(0) != n) {
        bad_input("hb_solve: b_dc has " + std::to_string(b_dc.shape(0)) + " entries, expected n = " +
                  std::to_string(n));
    }
    std::vector<hb::HbTimeSource> sources;
    for (nb::handle item : source_programs) {
        std::shared_ptr<mna::expr::Program> program;
        if (!nb::try_cast<std::shared_ptr<mna::expr::Program>>(item, program) || !program) {
            bad_input("hb_solve: source_programs must contain _core.Program objects");
        }
        hb::HbTimeSource source;
        source.program = std::move(program);
        sources.push_back(std::move(source));
    }
    if (sources.size() != source_rows.shape(0)) {
        bad_input("hb_solve: " + std::to_string(source_rows.shape(0)) + " source rows for " +
                  std::to_string(sources.size()) + " programs");
    }
    for (std::size_t q = 0; q < sources.size(); ++q) {
        const std::int64_t row = source_rows.data()[q];
        if (row < 0 || row >= static_cast<std::int64_t>(n)) {
            bad_input("hb_solve: time source row " + std::to_string(row) + " outside [0, " + std::to_string(n) + ")");
        }
        sources[q].row = static_cast<std::int32_t>(row);
    }

    hb::HbOptions options;
    options.omega0 = omega0;
    options.harmonics = harmonics;
    options.max_iter = max_iter;
    options.tol = tol;
    options.gmin = gmin;
    options.nonlinear = nonlinear;
    options.use_krylov = use_krylov;
    options.krylov_method = krylov_method;
    options.krylov_tol = krylov_tol;
    options.krylov_max_iter = krylov_max_iter;
    options.krylov_rank = krylov_rank;
    options.num_threads = mna::effective_threads(num_threads);
    options.max_nnz = max_nnz;

    hb::HbResult result;
    {
        nb::gil_scoped_release release;
        result = hb::hb_solve(circuit, b_dc.data(), sources, options);
    }
    const std::size_t size = result.x_bar.size();
    return nb::make_tuple(to_numpy(std::move(result.x_bar), {size}), stats_dict(result.stats));
}

void bind_hb(nb::module_& m) {
    m.attr("HB_MAX_DIMENSION") = hb::kHbMaxDimension;
    m.attr("HB_DEFAULT_MAX_NNZ") = hb::kHbDefaultMaxNnz;
    m.attr("BTF_DENSE_MAX_SMALL") = hb::kBtfDenseMaxSmall;
    m.attr("BTF_DENSE_FRACTION") = hb::kBtfDenseFraction;

    m.def("hb_fourier_matrices", &fourier_matrices, "n_harmonics"_a, "num_points"_a,
          "(Gamma (S, K), Gamma_inv (K, S)) of solvers.create_fourier_matrices for K = num_points time points.");
    m.def("hb_time_points", &time_points, "omega0"_a, "num_points"_a,
          "numpy.linspace(0, 2 pi / omega0, num_points, endpoint=False), bit for bit.");
    m.def("hb_reconstruct", &reconstruct, "x_bar"_a, "num_vars"_a, "harmonics"_a, "omega0"_a, "t"_a,
          "num_threads"_a = 1,
          "Time-domain waveforms (num_vars, len(t)) from harmonic coefficients (solvers.reconstruct_time_domain);\n"
          "parallel over time points, bitwise identical for any thread count.");
    m.def("hb_jacobian_block", &jacobian_block, "harmonics"_a, "g"_a,
          "Gamma diag(g) Gamma_inv (K x K) as hb_solve assembles a device-Jacobian block (for tests).");
    m.def("btf_solve", &btf_solve, "col_ptr"_a, "row_idx"_a, "values"_a, "b"_a, "num_threads"_a = 1,
          "Solve A x = b with the block-triangular (BTF + KLU / dense LU) solver of hb_solve (for tests).\n"
          "Returns (x, info). Raises CoreError(JACOBIAN_SINGULAR) for a singular matrix.");
    m.def("hb_solve", &solve, "circuit"_a, "omega0"_a, "harmonics"_a, "b_dc"_a, "source_rows"_a,
          "source_programs"_a, "gmin"_a = 0.0, "nonlinear"_a = true, "max_iter"_a = 100, "tol"_a = 1e-6,
          "use_krylov"_a = false, "krylov_method"_a = "auto", "krylov_tol"_a = 1e-9, "krylov_max_iter"_a = 2000,
          "krylov_rank"_a = 80, "num_threads"_a = 1, "max_nnz"_a = hb::kHbDefaultMaxNnz,
          "Harmonic balance on a compiled circuit (solvers.solve_harmonic_balance). Time sources are\n"
          "(row, time-dependent Program) pairs; krylov_rank is the resolved rank. Returns (x_bar (n K,), stats).\n"
          "Raises CoreError(HB_ERROR) with the reference texts on failure and for the memory policy.");
}

const mna::bindings::Registrar reg_hb{"hb", &bind_hb};

}  // namespace
