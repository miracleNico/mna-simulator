// Bindings of the model-order-reduction kernels (mna/mor/*.hpp), the native
// side of mna_simulation/native/mor.py.
//
//   mor_krylov_basis      rational Krylov basis on CSC operands (krylov_basis.hpp)
//   mor_project           V^T A V for a CSC matrix (projection.hpp)
//   mor_reduced_ac        reduced frequency loop of solve_ac_mor (reduced_ac.hpp)
//   mor_reduced_transient reduced step loop of solve_transient_mor (reduced_transient.hpp)
//   mor_pod_basis         POD basis of the TPWL snapshots (pod.hpp)
//   mor_training_indexes  TPWL training indexes (tpwl.hpp)
//   mor_tpwl_models       TPWL local models on a compiled circuit (tpwl.hpp)
//   mor_tpwl_simulate     TPWL reduced step loop (tpwl.hpp)
//   mor_dense_operands    union CSC pattern of two dense matrices (projection.hpp)
//   mor_safe_solve        dense _safe_solve (safe_solve.hpp)
//
// Arrays: float64 / complex128 / int32 / int64 NumPy arrays with any strides
// (copied once); 2-D results are Fortran-ordered views of the native
// column-major storage. Every kernel runs with the GIL released. Malformed
// shapes raise CoreError(NETLIST_FATAL); the least-squares fallback on a
// non-finite matrix raises CoreError(JACOBIAN_SINGULAR) (numpy's LinAlgError).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mna/analyses/transient.hpp"
#include "mna/circuit/circuit.hpp"
#include "mna/error.hpp"
#include "mna/expr/program.hpp"
#include "mna/linear/linear_solver.hpp"
#include "mna/mor/krylov_basis.hpp"
#include "mna/mor/pod.hpp"
#include "mna/mor/projection.hpp"
#include "mna/mor/reduced_ac.hpp"
#include "mna/mor/reduced_transient.hpp"
#include "mna/mor/safe_solve.hpp"
#include "mna/mor/tpwl.hpp"
#include "mna/threads.hpp"
#include "registry.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using mna::circuit::Circuit;
using Complex = std::complex<double>;

template <class T>
using Arr1 = nb::ndarray<const T, nb::ndim<1>, nb::device::cpu>;
template <class T>
using Arr2 = nb::ndarray<const T, nb::ndim<2>, nb::device::cpu>;
using Arr3 = nb::ndarray<const double, nb::ndim<3>, nb::device::cpu>;

[[noreturn]] void bad_input(const std::string& message) { throw mna::Error(mna::ErrorCode::NetlistFatal, message); }

void expect(bool ok, const char* fn, const std::string& message) {
    if (!ok) {
        bad_input(std::string(fn) + ": " + message);
    }
}

std::string shape_text(std::size_t a, std::size_t b) {
    return "(" + std::to_string(a) + ", " + std::to_string(b) + ")";
}

std::int32_t checked_dim(std::size_t value, const char* fn, const char* what) {
    expect(value <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max() - 1), fn,
           std::string(what) + " too large for 32-bit indices");
    return static_cast<std::int32_t>(value);
}

template <class T>
std::vector<T> copy1(const Arr1<T>& a) {
    std::vector<T> out(a.shape(0));
    const T* data = a.data();
    const std::int64_t stride = a.stride(0);
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = data[static_cast<std::int64_t>(i) * stride];
    }
    return out;
}

/// Column-major copy of a 2-D array with any strides.
template <class T>
std::vector<T> copy2(const Arr2<T>& a) {
    const std::size_t rows = a.shape(0);
    const std::size_t cols = a.shape(1);
    std::vector<T> out(rows * cols);
    const T* data = a.data();
    const std::int64_t s0 = a.stride(0);
    const std::int64_t s1 = a.stride(1);
    for (std::size_t j = 0; j < cols; ++j) {
        for (std::size_t i = 0; i < rows; ++i) {
            out[i + j * rows] = data[static_cast<std::int64_t>(i) * s0 + static_cast<std::int64_t>(j) * s1];
        }
    }
    return out;
}

/// Hands a vector to NumPy without copying: 1-D, or 2-D/3-D with explicit element strides.
template <class T>
nb::object to_numpy(std::vector<T>&& values, std::initializer_list<std::size_t> shape,
                    std::initializer_list<std::int64_t> strides) {
    auto* holder = new std::vector<T>(std::move(values));
    if (holder->empty()) {
        holder->reserve(1);  // keep data() non-null for zero-length arrays
    }
    nb::capsule owner(holder, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    const std::vector<std::size_t> shp(shape);
    const std::vector<std::int64_t> str(strides);
    using Out = nb::ndarray<nb::numpy, T>;
    return nb::cast(Out(holder->data(), shp.size(), shp.data(), owner, str.data()));
}

/// Column-major rows x cols as a Fortran-ordered (rows, cols) array.
template <class T>
nb::object fortran(std::vector<T>&& values, std::size_t rows, std::size_t cols) {
    return to_numpy(std::move(values), {rows, cols}, {1, static_cast<std::int64_t>(rows)});
}

/// Row-major rows x cols as a C-ordered (rows, cols) array.
template <class T>
nb::object c_order(std::vector<T>&& values, std::size_t rows, std::size_t cols) {
    return to_numpy(std::move(values), {rows, cols}, {static_cast<std::int64_t>(cols), 1});
}

template <class T>
nb::object flat(std::vector<T>&& values) {
    const std::size_t size = values.size();
    return to_numpy(std::move(values), {size}, {1});
}

mna::linear::CscPattern pattern_of(const Arr1<std::int32_t>& col_ptr, const Arr1<std::int32_t>& row_idx,
                                   const char* fn) {
    expect(col_ptr.shape(0) >= 1, fn, "col_ptr must have n + 1 entries");
    const std::int32_t n = checked_dim(col_ptr.shape(0) - 1, fn, "n");
    const std::vector<std::int32_t> ap = copy1(col_ptr);
    const std::vector<std::int32_t> ai = copy1(row_idx);
    return mna::linear::CscPattern::from_arrays(n, ap.data(), ap.size(), ai.data(), ai.size());
}

std::vector<mna::analyses::TimeSource> time_sources(std::int32_t n, const Arr1<std::int64_t>& rows,
                                                    const nb::handle& programs, const char* fn) {
    std::vector<mna::analyses::TimeSource> sources;
    const std::vector<std::int64_t> row_values = copy1(rows);
    std::size_t index = 0;
    for (nb::handle item : programs) {
        expect(index < row_values.size(), fn, "more source programs than source rows");
        std::shared_ptr<mna::expr::Program> program;
        expect(nb::try_cast<std::shared_ptr<mna::expr::Program>>(item, program) && program, fn,
               "source_programs must contain _core.Program objects");
        const std::int64_t row = row_values[index];
        expect(row >= 0 && row < n, fn,
               "source row " + std::to_string(row) + " outside [0, " + std::to_string(n) + ")");
        sources.push_back({static_cast<std::int32_t>(row), std::move(program)});
        ++index;
    }
    expect(index == row_values.size(), fn,
           std::to_string(row_values.size()) + " source rows but " + std::to_string(index) + " programs");
    return sources;
}

/// The reduced operands shared by the reduced loops: Gr, Cr (k x k), Sr (p x k).
struct Reduced {
    std::int32_t k = 0;
    std::int32_t p = 0;
    std::vector<double> gr, cr, sr;
};

Reduced reduced_operands(const Arr2<double>& gr, const Arr2<double>& cr, const Arr2<double>& sr, const char* fn) {
    Reduced r;
    r.k = checked_dim(gr.shape(0), fn, "k");
    expect(gr.shape(1) == gr.shape(0), fn, "Gr must be square, got " + shape_text(gr.shape(0), gr.shape(1)));
    expect(
        cr.shape(0) == gr.shape(0) && cr.shape(1) == gr.shape(0), fn,
        "Cr has shape " + shape_text(cr.shape(0), cr.shape(1)) + ", expected " + shape_text(gr.shape(0), gr.shape(0)));
    expect(sr.shape(1) == gr.shape(0), fn,
           "Sr has " + std::to_string(sr.shape(1)) + " columns, expected k = " + std::to_string(r.k));
    r.p = checked_dim(sr.shape(0), fn, "output count");
    r.gr = copy2(gr);
    r.cr = copy2(cr);
    r.sr = copy2(sr);
    return r;
}

// ---------------------------------------------------------------------------

nb::tuple mor_krylov_basis(Arr1<std::int32_t> col_ptr, Arr1<std::int32_t> row_idx, Arr1<double> g_values,
                           Arr1<double> c_values, Arr2<double> inputs, Arr2<double> selector, Arr1<Complex> shifts,
                           std::vector<bool> complex_shifts, std::int64_t order, int num_threads) {
    const char* fn = "mor_krylov_basis";
    const mna::linear::CscPattern pattern = pattern_of(col_ptr, row_idx, fn);
    const auto n = static_cast<std::size_t>(pattern.n);
    const auto nnz = static_cast<std::size_t>(pattern.nnz());
    expect(pattern.n > 0, fn, "the system is empty (n = 0)");
    expect(g_values.shape(0) == nnz, fn,
           "g_values has " + std::to_string(g_values.shape(0)) + " entries, expected nnz = " + std::to_string(nnz));
    expect(c_values.shape(0) == nnz, fn,
           "c_values has " + std::to_string(c_values.shape(0)) + " entries, expected nnz = " + std::to_string(nnz));
    expect(inputs.shape(0) == n, fn,
           "inputs has " + std::to_string(inputs.shape(0)) + " rows, expected n = " + std::to_string(n));
    expect(selector.shape(1) == n, fn,
           "selector has " + std::to_string(selector.shape(1)) + " columns, expected n = " + std::to_string(n));
    expect(shifts.shape(0) == complex_shifts.size(), fn, "shifts and complex_shifts differ in length");
    const std::vector<double> g = copy1(g_values);
    const std::vector<double> c = copy1(c_values);
    const std::vector<double> in = copy2(inputs);
    const auto input_cols = checked_dim(inputs.shape(1), fn, "input count");
    const auto output_cols = checked_dim(selector.shape(0), fn, "output count");
    // selector is p x n; its transpose (the output seeds) as column-major n x p.
    std::vector<double> outputs(n * static_cast<std::size_t>(output_cols));
    for (std::size_t r = 0; r < static_cast<std::size_t>(output_cols); ++r) {
        for (std::size_t i = 0; i < n; ++i) {
            outputs[i + r * n] = selector.data()[static_cast<std::int64_t>(r) * selector.stride(0) +
                                                 static_cast<std::int64_t>(i) * selector.stride(1)];
        }
    }
    std::vector<mna::mor::Shift> shift_list;
    const std::vector<Complex> shift_values = copy1(shifts);
    for (std::size_t s = 0; s < shift_values.size(); ++s) {
        shift_list.push_back({shift_values[s], static_cast<bool>(complex_shifts[s])});
    }
    mna::mor::SparseOperands ops;
    ops.pattern = &pattern;
    ops.g = g.data();
    ops.c = c.data();

    mna::mor::KrylovBasisResult result;
    {
        nb::gil_scoped_release release;
        result = mna::mor::krylov_basis(ops, in.data(), input_cols, outputs.data(), output_cols, shift_list, order,
                                        mna::effective_threads(num_threads));
    }
    nb::dict info;
    info["columns"] = result.columns;
    info["factorizations"] = result.factorizations;
    info["solves"] = result.solves;
    info["lstsq_shifts"] = result.lstsq_shifts;
    info["threads_used"] = result.threads_used;
    const auto cols = static_cast<std::size_t>(result.columns);
    return nb::make_tuple(fortran(std::move(result.basis), n, cols), info);
}

nb::object mor_project(Arr1<std::int32_t> col_ptr, Arr1<std::int32_t> row_idx, Arr1<double> values, Arr2<double> V) {
    const char* fn = "mor_project";
    const mna::linear::CscPattern pattern = pattern_of(col_ptr, row_idx, fn);
    const auto n = static_cast<std::size_t>(pattern.n);
    expect(
        values.shape(0) == static_cast<std::size_t>(pattern.nnz()), fn,
        "values has " + std::to_string(values.shape(0)) + " entries, expected nnz = " + std::to_string(pattern.nnz()));
    expect(V.shape(0) == n, fn, "V has " + std::to_string(V.shape(0)) + " rows, expected n = " + std::to_string(n));
    const std::int32_t k = checked_dim(V.shape(1), fn, "k");
    const std::vector<double> vals = copy1(values);
    const std::vector<double> basis = copy2(V);
    std::vector<double> out;
    {
        nb::gil_scoped_release release;
        out = mna::mor::project_matrix(pattern, vals.data(), basis.data(), k);
    }
    const auto uk = static_cast<std::size_t>(k);
    return fortran(std::move(out), uk, uk);
}

nb::tuple mor_reduced_ac(Arr2<double> gr, Arr2<double> cr, Arr1<Complex> br, Arr2<double> sr, Arr1<double> frequencies,
                         int num_threads) {
    const char* fn = "mor_reduced_ac";
    const Reduced r = reduced_operands(gr, cr, sr, fn);
    expect(br.shape(0) == static_cast<std::size_t>(r.k), fn,
           "Br has " + std::to_string(br.shape(0)) + " entries, expected k = " + std::to_string(r.k));
    const std::vector<Complex> b = copy1(br);
    const std::vector<double> f = copy1(frequencies);
    mna::mor::ReducedAcResult result;
    {
        nb::gil_scoped_release release;
        result = mna::mor::reduced_ac(r.k, r.gr.data(), r.cr.data(), b.data(), r.p, r.sr.data(), f.data(),
                                      static_cast<std::int64_t>(f.size()), mna::effective_threads(num_threads));
    }
    nb::dict info;
    info["threads_used"] = result.threads_used;
    info["lstsq_points"] = result.lstsq_points;
    return nb::make_tuple(c_order(std::move(result.magnitudes), static_cast<std::size_t>(r.p), f.size()), info);
}

nb::tuple mor_reduced_transient(Arr2<double> gr, Arr2<double> cr, Arr2<double> V, Arr2<double> sr, Arr1<double> x0,
                                Arr1<double> t_grid, Arr1<double> b_dc, Arr1<std::int64_t> source_rows,
                                nb::handle source_programs) {
    const char* fn = "mor_reduced_transient";
    const Reduced r = reduced_operands(gr, cr, sr, fn);
    expect(V.shape(1) == static_cast<std::size_t>(r.k), fn,
           "V has " + std::to_string(V.shape(1)) + " columns, expected k = " + std::to_string(r.k));
    const std::int32_t n = checked_dim(V.shape(0), fn, "n");
    expect(x0.shape(0) == V.shape(0), fn,
           "x0 has " + std::to_string(x0.shape(0)) + " entries, expected n = " + std::to_string(n));
    expect(b_dc.shape(0) == V.shape(0), fn,
           "b_dc has " + std::to_string(b_dc.shape(0)) + " entries, expected n = " + std::to_string(n));
    const auto sources = time_sources(n, source_rows, source_programs, fn);
    const std::vector<double> basis = copy2(V);
    const std::vector<double> x = copy1(x0);
    const std::vector<double> grid = copy1(t_grid);
    const std::vector<double> b = copy1(b_dc);
    mna::mor::ReducedTransientResult result;
    {
        nb::gil_scoped_release release;
        result = mna::mor::reduced_transient(n, r.k, r.gr.data(), r.cr.data(), basis.data(), r.p, r.sr.data(), x.data(),
                                             grid, b.data(), sources);
    }
    nb::dict info;
    info["failed"] = result.failed;
    info["failed_step"] = result.failed_step;
    info["factorizations"] = result.factorizations;
    info["lstsq_steps"] = result.lstsq_steps;
    return nb::make_tuple(c_order(std::move(result.values), static_cast<std::size_t>(r.p), grid.size()), info);
}

nb::tuple mor_pod_basis(Arr2<double> snapshots, std::int64_t order) {
    const char* fn = "mor_pod_basis";
    const std::int32_t n = checked_dim(snapshots.shape(0), fn, "n");
    const auto T = static_cast<std::int64_t>(snapshots.shape(1));
    const std::vector<double> s = copy2(snapshots);
    mna::mor::PodResult result;
    {
        nb::gil_scoped_release release;
        result = mna::mor::pod_basis(n, T, s.data(), order);
    }
    nb::dict info;
    info["rank"] = result.rank;
    info["singular_values"] = flat(std::move(result.singular_values));
    const auto cols = static_cast<std::size_t>(result.columns);
    return nb::make_tuple(fortran(std::move(result.basis), static_cast<std::size_t>(n), cols), info);
}

nb::object mor_training_indexes(Arr2<double> snapshots, std::int64_t basis_columns, std::int64_t output_count) {
    const char* fn = "mor_training_indexes";
    const std::int32_t n = checked_dim(snapshots.shape(0), fn, "n");
    const auto T = static_cast<std::int64_t>(snapshots.shape(1));
    const std::vector<double> s = copy2(snapshots);
    std::vector<std::int64_t> indexes;
    {
        nb::gil_scoped_release release;
        indexes = mna::mor::training_indexes(n, T, s.data(), basis_columns, output_count);
    }
    return flat(std::move(indexes));
}

nb::dict mor_tpwl_models(const Circuit& circuit, Arr2<double> snapshots, Arr1<std::int64_t> indexes, Arr2<double> V,
                         int num_threads) {
    const char* fn = "mor_tpwl_models";
    const auto n = static_cast<std::size_t>(circuit.n());
    expect(snapshots.shape(0) == n, fn,
           "snapshots has " + std::to_string(snapshots.shape(0)) + " rows, expected n = " + std::to_string(n));
    expect(V.shape(0) == n, fn, "V has " + std::to_string(V.shape(0)) + " rows, expected n = " + std::to_string(n));
    const std::int32_t k = checked_dim(V.shape(1), fn, "k");
    const auto T = static_cast<std::int64_t>(snapshots.shape(1));
    const std::vector<double> s = copy2(snapshots);
    const std::vector<std::int64_t> idx = copy1(indexes);
    const std::vector<double> basis = copy2(V);
    mna::mor::TpwlModels models;
    {
        nb::gil_scoped_release release;
        models = mna::mor::tpwl_local_models(circuit, T, s.data(), idx, k, basis.data(),
                                             mna::effective_threads(num_threads));
    }
    const auto count = static_cast<std::size_t>(models.count);
    const auto uk = static_cast<std::size_t>(k);
    const auto unnz = static_cast<std::size_t>(models.nnz);
    const auto ik = static_cast<std::int64_t>(uk);
    nb::dict out;
    out["z"] = c_order(std::move(models.z), count, uk);
    out["f"] = c_order(std::move(models.f), count, n);
    out["jvals"] = c_order(std::move(models.jvals), count, unnz);
    // Model m's Jr is column-major k x k: element [m, r, c] at m*k*k + r + c*k.
    out["jr"] = to_numpy(std::move(models.jr), {count, uk, uk}, {ik * ik, 1, ik});
    out["q"] = c_order(std::move(models.q), count, uk);
    out["threads_used"] = models.threads_used;
    return out;
}

nb::tuple mor_tpwl_simulate(Arr2<double> gr, Arr2<double> cr, Arr2<double> V, Arr2<double> sr, Arr2<double> model_z,
                            Arr3 model_jr, Arr2<double> model_q, Arr1<double> x0, Arr1<double> t_grid,
                            Arr1<double> b_dc, Arr1<std::int64_t> source_rows, nb::handle source_programs) {
    const char* fn = "mor_tpwl_simulate";
    const Reduced r = reduced_operands(gr, cr, sr, fn);
    const auto uk = static_cast<std::size_t>(r.k);
    expect(V.shape(1) == uk, fn,
           "V has " + std::to_string(V.shape(1)) + " columns, expected k = " + std::to_string(r.k));
    const std::int32_t n = checked_dim(V.shape(0), fn, "n");
    const std::size_t count = model_z.shape(0);
    expect(count >= 1, fn, "no local models");
    expect(model_z.shape(1) == uk, fn, "model_z must have shape (models, k)");
    expect(model_q.shape(0) == count && model_q.shape(1) == uk, fn, "model_q must have shape (models, k)");
    expect(model_jr.shape(0) == count && model_jr.shape(1) == uk && model_jr.shape(2) == uk, fn,
           "model_jr must have shape (models, k, k)");
    expect(x0.shape(0) == V.shape(0), fn,
           "x0 has " + std::to_string(x0.shape(0)) + " entries, expected n = " + std::to_string(n));
    expect(b_dc.shape(0) == V.shape(0), fn,
           "b_dc has " + std::to_string(b_dc.shape(0)) + " entries, expected n = " + std::to_string(n));
    const auto sources = time_sources(n, source_rows, source_programs, fn);
    const std::vector<double> basis = copy2(V);
    // Model-major copies: z and q row by row, Jr column-major per model.
    std::vector<double> z(count * uk), q(count * uk), jr(count * uk * uk);
    for (std::size_t m = 0; m < count; ++m) {
        for (std::size_t c = 0; c < uk; ++c) {
            const auto im = static_cast<std::int64_t>(m);
            const auto ic = static_cast<std::int64_t>(c);
            z[m * uk + c] = model_z.data()[im * model_z.stride(0) + ic * model_z.stride(1)];
            q[m * uk + c] = model_q.data()[im * model_q.stride(0) + ic * model_q.stride(1)];
            for (std::size_t row = 0; row < uk; ++row) {
                const auto ir = static_cast<std::int64_t>(row);
                jr[m * uk * uk + row + c * uk] =
                    model_jr.data()[im * model_jr.stride(0) + ir * model_jr.stride(1) + ic * model_jr.stride(2)];
            }
        }
    }
    const std::vector<double> x = copy1(x0);
    const std::vector<double> grid = copy1(t_grid);
    const std::vector<double> b = copy1(b_dc);
    mna::mor::TpwlSimulation result;
    {
        nb::gil_scoped_release release;
        result = mna::mor::tpwl_simulate(n, r.k, r.gr.data(), r.cr.data(), basis.data(), r.p, r.sr.data(),
                                         static_cast<std::int64_t>(count), z.data(), jr.data(), q.data(), x.data(),
                                         grid, b.data(), sources);
    }
    nb::dict info;
    info["factorizations"] = result.factorizations;
    info["lstsq_steps"] = result.lstsq_steps;
    info["model"] = flat(std::move(result.model));
    return nb::make_tuple(c_order(std::move(result.values), static_cast<std::size_t>(r.p), grid.size()), info);
}

nb::tuple mor_dense_operands(Arr2<double> G, Arr2<double> C) {
    const char* fn = "mor_dense_operands";
    expect(G.shape(0) == G.shape(1), fn, "G must be square, got " + shape_text(G.shape(0), G.shape(1)));
    expect(C.shape(0) == G.shape(0) && C.shape(1) == G.shape(0), fn,
           "C has shape " + shape_text(C.shape(0), C.shape(1)) + ", expected " + shape_text(G.shape(0), G.shape(0)));
    const std::int32_t n = checked_dim(G.shape(0), fn, "n");
    mna::mor::DenseUnionOperands ops;
    {
        nb::gil_scoped_release release;
        ops = mna::mor::dense_union_operands(n, G.data(), G.stride(0), G.stride(1), C.data(), C.stride(0), C.stride(1));
    }
    return nb::make_tuple(flat(std::move(ops.pattern.col_ptr)), flat(std::move(ops.pattern.row_idx)),
                          flat(std::move(ops.g)), flat(std::move(ops.c)));
}

template <class T>
nb::tuple mor_safe_solve(Arr2<T> a, Arr2<T> b) {
    const char* fn = "mor_safe_solve";
    expect(a.shape(0) == a.shape(1), fn, "the matrix must be square, got " + shape_text(a.shape(0), a.shape(1)));
    expect(b.shape(0) == a.shape(0), fn,
           "the right-hand side has " + std::to_string(b.shape(0)) + " rows, expected " + std::to_string(a.shape(0)));
    const std::int32_t n = checked_dim(a.shape(0), fn, "n");
    const std::int32_t nrhs = checked_dim(b.shape(1), fn, "right-hand-side count");
    const std::vector<T> matrix = copy2(a);
    const std::vector<T> rhs = copy2(b);
    std::vector<T> x;
    mna::mor::SolvePath path = mna::mor::SolvePath::LU;
    {
        nb::gil_scoped_release release;
        x = mna::mor::safe_solve<T>(n, matrix.data(), rhs.data(), nrhs, &path);
    }
    return nb::make_tuple(fortran(std::move(x), static_cast<std::size_t>(n), static_cast<std::size_t>(nrhs)),
                          mna::mor::solve_path_name(path));
}

void bind_mor(nb::module_& m) {
    m.attr("MOR_PARALLEL_MIN_UNKNOWNS") = mna::mor::kMorParallelMinUnknowns;

    m.def("mor_krylov_basis", &mor_krylov_basis, "col_ptr"_a, "row_idx"_a, "g_values"_a, "c_values"_a, "inputs"_a,
          "selector"_a, "shifts"_a, "complex_shifts"_a, "order"_a, "num_threads"_a = 1,
          "Output-aware rational Krylov basis of mor.linear.build_linear_krylov_basis on CSC operands.\n"
          "G and C share the int32 pattern (col_ptr, row_idx); inputs: float64 (n, m) (m may be 0);\n"
          "selector: float64 (p, n); shifts: complex128 (s,) with complex_shifts[i] telling whether shift i is\n"
          "complex (numpy.iscomplexobj). Each shifted system is factored once with KLU (real or complex) and\n"
          "reused for every block; a singular one falls back to dense least squares. Returns (V (n, k)\n"
          "Fortran-ordered, info) with info keys columns, factorizations, solves, lstsq_shifts, threads_used.");
    m.def("mor_project", &mor_project, "col_ptr"_a, "row_idx"_a, "values"_a, "V"_a,
          "V^T A V (k, k) for a CSC matrix A (values in pattern order) and a dense basis V (n, k).");
    m.def("mor_reduced_ac", &mor_reduced_ac, "Gr"_a, "Cr"_a, "Br"_a, "Sr"_a, "frequencies"_a, "num_threads"_a = 1,
          "Reduced AC loop of solve_ac_mor: |Sr _safe_solve(Gr + j*2*pi*f*Cr, Br)| for every frequency.\n"
          "Returns (magnitudes (p, points), info) with info keys threads_used, lstsq_points. Parallel over\n"
          "frequencies when points >= 2 * threads; bitwise identical for any thread count.");
    m.def("mor_reduced_transient", &mor_reduced_transient, "Gr"_a, "Cr"_a, "V"_a, "Sr"_a, "x0"_a, "t_grid"_a, "b_dc"_a,
          "source_rows"_a, "source_programs"_a,
          "Reduced Backward-Euler loop of solve_transient_mor. Returns (values (p, T), info) with info keys\n"
          "failed (non-finite or ||z|| > 1e12: that step and the rest are NaN), failed_step, factorizations,\n"
          "lstsq_steps.");
    m.def("mor_pod_basis", &mor_pod_basis, "snapshots"_a, "order"_a,
          "POD basis of mor.tpwl._pod_basis (thin SVD, rank rule s_i > max(s_0, 1) * 1e-12). Returns\n"
          "(V (n, r) Fortran-ordered, info) with info keys rank and singular_values.");
    m.def("mor_training_indexes", &mor_training_indexes, "snapshots"_a, "basis_columns"_a, "output_count"_a,
          "TPWL training indexes of mor.tpwl._training_indexes (sorted, unique, int64).");
    m.def("mor_tpwl_models", &mor_tpwl_models, "circuit"_a, "snapshots"_a, "indexes"_a, "V"_a, "num_threads"_a = 1,
          "TPWL local models at the snapshot columns `indexes`: dict with z (m, k), f (m, n), jvals (m, nnz)\n"
          "(device Jacobian in the circuit pattern), jr (m, k, k) = V^T J V, q (m, k) = V^T (J x - f) and\n"
          "threads_used (parallel over models only from MOR_PARALLEL_MIN_UNKNOWNS unknowns).");
    m.def("mor_tpwl_simulate", &mor_tpwl_simulate, "Gr"_a, "Cr"_a, "V"_a, "Sr"_a, "model_z"_a, "model_jr"_a,
          "model_q"_a, "x0"_a, "t_grid"_a, "b_dc"_a, "source_rows"_a, "source_programs"_a,
          "TPWL reduced step loop of solve_transient_tpwl (nearest local model per step). Returns\n"
          "(values (p, T), info) with info keys factorizations, lstsq_steps and model (model index per step).");
    m.def("mor_dense_operands", &mor_dense_operands, "G"_a, "C"_a,
          "Union CSC pattern of two dense float64 (n, n) matrices: (col_ptr int32, row_idx int32, g, c), an\n"
          "entry stored where G or C is nonzero (NaN included).");
    m.def("mor_safe_solve", &mor_safe_solve<double>, "a"_a, "b"_a,
          "Dense _safe_solve(a, b) for float64 a (n, n) and b (n, r): (x (n, r), path) with path 'lu',\n"
          "'least_squares' or 'regularized_least_squares'.");
    m.def("mor_safe_solve", &mor_safe_solve<Complex>, "a"_a, "b"_a,
          "Dense _safe_solve(a, b) for complex128 a (n, n) and b (n, r).");
}

const mna::bindings::Registrar reg_mor{"mor", &bind_mor};

}  // namespace
