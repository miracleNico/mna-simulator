#include <doctest/doctest.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#include "mna/threads.hpp"

TEST_CASE("chunk boundaries depend only on the count and tile the range") {
    CHECK(mna::chunk_count(0) == 0);
    CHECK(mna::chunk_count(-5) == 0);
    CHECK(mna::chunk_count(1) == 1);
    CHECK(mna::chunk_count(100) == 100);
    CHECK(mna::chunk_count(1000000) == mna::kMaxParallelChunks);
    for (std::int64_t count : {1, 2, 7, 255, 256, 257, 1000, 123457}) {
        const std::int64_t n = mna::chunk_count(count);
        std::int64_t expected_begin = 0;
        for (std::int64_t c = 0; c < n; ++c) {
            const mna::ChunkRange r = mna::chunk_range(count, c);
            CHECK(r.begin == expected_begin);
            CHECK(r.end > r.begin);
            expected_begin = r.end;
        }
        CHECK(expected_begin == count);
    }
}

TEST_CASE("effective_threads clamps requests") {
    CHECK(mna::effective_threads(0) == 1);
    CHECK(mna::effective_threads(-3) == 1);
    CHECK(mna::effective_threads(1) == 1);
    const int hw = static_cast<int>(mna::hardware_threads());
    CHECK(hw >= 1);
    CHECK(mna::effective_threads(hw) == hw);
    CHECK(mna::effective_threads(hw + 7) == hw);
    CHECK(mna::effective_threads(1 << 30) == hw);
}

TEST_CASE("parallel_for visits every index exactly once") {
    for (std::int64_t count : {0, 1, 3, 8, 64, 1000, 4099}) {
        for (int threads = 1; threads <= 8; ++threads) {
            CAPTURE(count);
            CAPTURE(threads);
            std::vector<std::atomic<int>> hits(static_cast<std::size_t>(count));
            for (auto& h : hits) {
                h.store(0);
            }
            const int used = mna::parallel_for(count, threads, [&](std::int64_t begin, std::int64_t end) {
                REQUIRE(begin >= 0);
                REQUIRE(end <= count);
                for (std::int64_t i = begin; i < end; ++i) {
                    hits[static_cast<std::size_t>(i)].fetch_add(1, std::memory_order_relaxed);
                }
            });
            CHECK(used >= 1);
            CHECK(used <= threads);
            if (count > 0) {
                CHECK(used <= mna::chunk_count(count));
            }
            for (std::int64_t i = 0; i < count; ++i) {
                CHECK(hits[static_cast<std::size_t>(i)].load() == 1);
            }
        }
    }
}

TEST_CASE("more threads than items") {
    for (std::int64_t count : {1, 2, 3}) {
        std::vector<int> hits(static_cast<std::size_t>(count), 0);
        const int used = mna::parallel_for(count, 8, [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t i = begin; i < end; ++i) {
                ++hits[static_cast<std::size_t>(i)];  // one chunk per item: no two threads share an index
            }
        });
        CHECK(used == count);
        for (int h : hits) {
            CHECK(h == 1);
        }
    }
}

TEST_CASE("num_threads <= 1 runs inline on the calling thread") {
    const std::thread::id caller = std::this_thread::get_id();
    for (int threads : {-1, 0, 1}) {
        bool all_inline = true;
        const int used = mna::parallel_for(1000, threads, [&](std::int64_t, std::int64_t) {
            all_inline = all_inline && std::this_thread::get_id() == caller;
        });
        CHECK(used == 1);
        CHECK(all_inline);
    }
}

TEST_CASE("chunked reductions are bitwise identical across thread counts") {
    const std::int64_t count = 100003;
    auto term = [](std::int64_t i) { return std::sin(0.001 * static_cast<double>(i)) / (1.0 + static_cast<double>(i)); };

    auto reduce = [&](int threads) {
        std::vector<double> partial(static_cast<std::size_t>(mna::chunk_count(count)), 0.0);
        mna::parallel_for_chunks(count, threads, [&](std::int64_t chunk, std::int64_t begin, std::int64_t end) {
            double s = 0.0;
            for (std::int64_t i = begin; i < end; ++i) {
                s += term(i);
            }
            partial[static_cast<std::size_t>(chunk)] = s;
        });
        double total = 0.0;
        for (double p : partial) {
            total += p;
        }
        return total;
    };

    const double reference = reduce(1);
    for (int threads = 2; threads <= 8; ++threads) {
        const double value = reduce(threads);
        CHECK(std::memcmp(&value, &reference, sizeof(double)) == 0);
    }
}

TEST_CASE("the first exception propagates after all threads join") {
    for (int threads : {1, 2, 4, 8}) {
        CAPTURE(threads);
        std::atomic<int> calls{0};
        CHECK_THROWS_WITH_AS(mna::parallel_for(1000, threads,
                                               [&](std::int64_t begin, std::int64_t) {
                                                   calls.fetch_add(1);
                                                   if (begin == 0) {
                                                       throw std::runtime_error("boom");
                                                   }
                                               }),
                             "boom", std::runtime_error);
        CHECK(calls.load() >= 1);
    }
    // Still usable afterwards.
    std::atomic<std::int64_t> sum{0};
    mna::parallel_for(100, 4, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t i = begin; i < end; ++i) {
            sum.fetch_add(i);
        }
    });
    CHECK(sum.load() == 4950);
}
