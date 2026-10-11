#include "mna/mor/reduced_ac.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "mna/error.hpp"
#include "mna/mor/safe_solve.hpp"
#include "mna/threads.hpp"

namespace mna::mor {
namespace {

// numpy.pi; 2 * pi * f is evaluated left to right as in `2.0 * np.pi * float(frequency)`.
constexpr double kPi = 3.14159265358979323846;

}  // namespace

ReducedAcResult reduced_ac(std::int32_t k, const double* gr, const double* cr, const std::complex<double>* br,
                           std::int32_t outputs, const double* sr, const double* frequencies, std::int64_t points,
                           int num_threads) {
    if (k < 0 || outputs < 0 || points < 0) {
        throw Error(ErrorCode::NetlistFatal, "mor_reduced_ac: negative dimension");
    }
    if ((k > 0 && (gr == nullptr || cr == nullptr || br == nullptr)) || (points > 0 && frequencies == nullptr) ||
        (k > 0 && outputs > 0 && sr == nullptr)) {
        throw Error(ErrorCode::NetlistFatal, "mor_reduced_ac: null input array");
    }
    ReducedAcResult result;
    result.outputs = outputs;
    result.points = points;
    const auto uk = static_cast<std::size_t>(k);
    const auto up = static_cast<std::size_t>(outputs);
    const auto npoints = static_cast<std::size_t>(points);
    std::vector<double> columns(up * npoints, 0.0);  // frequency-major: column i at i * outputs
    std::vector<unsigned char> lstsq(npoints, 0);

    const double work =
        static_cast<double>(points) * static_cast<double>(k) * static_cast<double>(k) * static_cast<double>(k);
    const int threads = work >= kReducedAcParallelMinWork ? effective_threads(num_threads) : 1;
    const std::int64_t frequency_threads = (std::min)(static_cast<std::int64_t>(threads), points / 2);

    std::atomic<std::int64_t> next{0};
    std::atomic<bool> aborted{false};
    auto sweep = [&](std::int64_t, std::int64_t) {
        std::vector<Complex> a(uk * uk);
        std::vector<Complex> z(uk);
        DenseSafeSolver<Complex> solver;
        try {
            for (;;) {
                const std::int64_t i = next.fetch_add(1);
                if (i >= points || aborted.load()) {
                    return;
                }
                const double omega = 2.0 * kPi * frequencies[i];
                for (std::size_t e = 0; e < uk * uk; ++e) {
                    a[e] = Complex(gr[e], omega * cr[e]);
                }
                solver.factor(k, a.data());
                std::copy(br, br + uk, z.begin());
                solver.solve(z.data(), 1);
                const auto u = static_cast<std::size_t>(i);
                lstsq[u] = solver.path() == SolvePath::LU ? 0 : 1;
                double* column = columns.data() + u * up;
                for (std::size_t r = 0; r < up; ++r) {
                    // (Sr @ z)[r]: Sr real, z complex.
                    Complex y(0.0, 0.0);
                    for (std::size_t c = 0; c < uk; ++c) {
                        y += sr[r + c * up] * z[c];
                    }
                    column[r] = std::abs(y);
                }
            }
        } catch (...) {
            aborted.store(true);
            throw;
        }
    };
    if (frequency_threads >= 2) {
        result.threads_used = parallel_for(frequency_threads, static_cast<int>(frequency_threads), sweep);
    } else {
        sweep(0, 1);
    }

    result.magnitudes.assign(up * npoints, 0.0);
    for (std::size_t i = 0; i < npoints; ++i) {
        for (std::size_t r = 0; r < up; ++r) {
            result.magnitudes[r * npoints + i] = columns[i * up + r];
        }
        result.lstsq_points += lstsq[i];
    }
    return result;
}

}  // namespace mna::mor
