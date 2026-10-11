// Small std::thread based parallel loop for coarse-grained native work.
//
// Design rules (see PERFORMANCE_PLAN.md, section 5):
//   * No global state: the thread count is an argument of every call, so two
//     Python threads calling into the core with different counts never
//     interfere. Nothing here reads or writes process-wide settings.
//   * Determinism: [0, count) is split into chunks whose boundaries depend only
//     on `count`, never on the thread count. A kernel that writes per-index or
//     per-chunk results and reduces them in index order produces bit-identical
//     output for every thread count.
//   * num_threads <= 1 runs inline on the calling thread (no thread is created).
//   * The first exception thrown by `fn` on any thread is rethrown on the
//     calling thread after all workers have been joined; chunks not yet started
//     are skipped once an exception has been seen.
//
// `fn` is invoked concurrently from several threads and must be safe for that.
// Callers that take a user-supplied thread count should pass it through
// effective_threads() first; parallel_for itself only caps at the chunk count,
// so tests can exercise more threads than the machine has cores.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace mna {

/// Upper bound on the number of chunks one parallel loop is split into.
inline constexpr std::int64_t kMaxParallelChunks = 256;

struct ChunkRange {
    std::int64_t begin;
    std::int64_t end;
};

/// Number of chunks [0, count) is split into: min(count, kMaxParallelChunks).
inline std::int64_t chunk_count(std::int64_t count) noexcept {
    if (count <= 0) {
        return 0;
    }
    return (std::min)(count, kMaxParallelChunks);
}

/// Half-open index range of chunk `chunk` (0 <= chunk < chunk_count(count)).
/// Chunk sizes differ by at most one; the larger chunks come first.
inline ChunkRange chunk_range(std::int64_t count, std::int64_t chunk) noexcept {
    const std::int64_t n = chunk_count(count);
    const std::int64_t base = count / n;
    const std::int64_t rem = count % n;
    const std::int64_t begin = chunk * base + (std::min)(chunk, rem);
    return {begin, begin + base + (chunk < rem ? 1 : 0)};
}

/// std::thread::hardware_concurrency(), with "unknown" (0) reported as 1.
inline unsigned hardware_threads() noexcept {
    static const unsigned value = [] {
        const unsigned hc = std::thread::hardware_concurrency();
        return hc == 0 ? 1u : hc;
    }();
    return value;
}

/// Thread count to use for a request: 0 or negative -> 1, capped at
/// hardware_threads().
inline int effective_threads(int requested) noexcept {
    if (requested <= 1) {
        return 1;
    }
    const unsigned hw = hardware_threads();
    return static_cast<unsigned>(requested) > hw ? static_cast<int>(hw) : requested;
}

/// Calls fn(chunk_index, begin, end) once for every chunk of [0, count).
/// Chunks are statically assigned to threads in contiguous blocks; the caller
/// runs block 0. Returns the number of threads that took part (1 if inline).
template <class Fn>
int parallel_for_chunks(std::int64_t count, int num_threads, Fn&& fn) {
    const std::int64_t nchunks = chunk_count(count);
    const std::int64_t k = (std::min)(static_cast<std::int64_t>(num_threads < 1 ? 1 : num_threads),
                                      (std::max)(nchunks, std::int64_t{1}));
    if (k <= 1) {
        for (std::int64_t c = 0; c < nchunks; ++c) {
            const ChunkRange r = chunk_range(count, c);
            fn(c, r.begin, r.end);
        }
        return 1;
    }

    std::atomic<bool> failed{false};
    std::exception_ptr first_error;
    std::mutex error_mutex;

    auto run_block = [&](std::int64_t block) {
        const std::int64_t c_begin = block * nchunks / k;
        const std::int64_t c_end = (block + 1) * nchunks / k;
        try {
            for (std::int64_t c = c_begin; c < c_end; ++c) {
                if (failed.load(std::memory_order_relaxed)) {
                    return;
                }
                const ChunkRange r = chunk_range(count, c);
                fn(c, r.begin, r.end);
            }
        } catch (...) {
            const std::lock_guard<std::mutex> lock(error_mutex);
            if (!first_error) {
                first_error = std::current_exception();
            }
            failed.store(true, std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(k - 1));
    std::int64_t next_block = 1;
    try {
        for (; next_block < k; ++next_block) {
            workers.emplace_back(run_block, next_block);
        }
    } catch (...) {
        // The OS refused another thread: the caller runs the remaining blocks.
    }

    run_block(0);
    for (std::int64_t block = next_block; block < k; ++block) {
        run_block(block);
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
    if (first_error) {
        std::rethrow_exception(first_error);
    }
    return static_cast<int>(workers.size()) + 1;
}

/// Calls fn(begin, end) once for every chunk of [0, count); see
/// parallel_for_chunks. Returns the number of threads that took part.
template <class Fn>
int parallel_for(std::int64_t count, int num_threads, Fn&& fn) {
    return parallel_for_chunks(count, num_threads,
                               [&fn](std::int64_t, std::int64_t begin, std::int64_t end) { fn(begin, end); });
}

}  // namespace mna
