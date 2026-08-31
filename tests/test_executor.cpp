#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <future>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>
#include "vrp/Executor.hpp"

namespace {

std::vector<int> visitCounts(std::size_t n, std::size_t threads) {
    std::vector<int> counts(n, 0);
    auto exec = vrp::makeExecutor(threads);
    exec->parallelFor(n, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            counts[i]++;  // disjoint ranges, so no synchronisation is needed
        }
    });
    return counts;
}

}  // namespace

TEST_CASE("chunkRange partitions exactly and disjointly", "[executor]") {
    for (std::size_t n : {0u, 1u, 2u, 7u, 100u, 1001u}) {
        for (std::size_t parts : {1u, 2u, 3u, 8u, 16u}) {
            std::size_t covered = 0;
            std::size_t previousEnd = 0;
            for (std::size_t i = 0; i < parts; ++i) {
                const auto [begin, end] = vrp::chunkRange(n, parts, i);
                REQUIRE(begin <= end);
                REQUIRE(begin == previousEnd);
                REQUIRE(end <= n);
                covered += end - begin;
                previousEnd = end;
            }
            REQUIRE(covered == n);
            REQUIRE(previousEnd == n);
        }
    }
}

// A wedged pool would otherwise hang the suite and report nothing; here the
// wait is bounded, so it fails as a named assertion. Declared first because
// Catch2 runs cases in declaration order.
TEST_CASE("parallelFor finishes within a bounded time", "[executor]") {
    constexpr std::size_t kN = 256;
    // kRounds is the detector, not the workload: against a notify-before-publish
    // mutant, 3000 rounds caught it empirically every time, 300 about half the
    // time, 10 almost never. Do not trim it -- it is the project's only
    // lost-wakeup detector, on a platform with no TSan behind it.
    constexpr int kRounds = 3000;

    // Heap state owned by the task: on a wedge the runner is detached, so it
    // must not point into a destroyed frame, and the executor must outlive it
    // too or teardown hangs joining its workers.
    struct State {
        std::unique_ptr<vrp::Executor> exec;
        std::vector<int> counts;
    };
    auto state = std::make_shared<State>(State{vrp::makeExecutor(4), std::vector<int>(kN, 0)});

    for (int round = 0; round < kRounds; ++round) {
        // packaged_task, not std::async: an async future blocks in its own
        // destructor, reintroducing the hang this test converts to an assertion.
        std::packaged_task<void()> task([state] {
            state->exec->parallelFor(kN, [&state](std::size_t begin, std::size_t end) {
                for (std::size_t i = begin; i < end; ++i) {
                    state->counts[i]++;
                }
            });
        });
        std::future<void> done = task.get_future();
        std::thread runner(std::move(task));

        if (done.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
            runner.detach();
            FAIL("parallelFor did not return within 5s on round "
                 << round << " -- the pool is deadlocked");
        }
        runner.join();
        done.get();
    }

    for (std::size_t i = 0; i < kN; ++i) {
        INFO("i=" << i);
        REQUIRE(state->counts[i] == kRounds);
    }
}

TEST_CASE("every index is visited exactly once", "[executor]") {
    for (std::size_t threads : {1u, 2u, 4u, 8u}) {
        for (std::size_t n : {0u, 1u, 3u, 64u, 1000u}) {
            const std::vector<int> counts = visitCounts(n, threads);
            REQUIRE(counts.size() == n);
            for (std::size_t i = 0; i < n; ++i) {
                INFO("threads=" << threads << " n=" << n << " i=" << i);
                REQUIRE(counts[i] == 1);
            }
        }
    }
}

TEST_CASE("n smaller than the thread count is handled", "[executor]") {
    const std::vector<int> counts = visitCounts(3, 8);
    REQUIRE(counts == std::vector<int>{1, 1, 1});
}

TEST_CASE("threads == 1 yields a serial executor", "[executor]") {
    auto exec = vrp::makeExecutor(1);
    REQUIRE(exec->threadCount() == 1);
    std::size_t calls = 0;
    exec->parallelFor(10, [&](std::size_t begin, std::size_t end) {
        calls++;
        REQUIRE(begin == 0);
        REQUIRE(end == 10);
    });
    REQUIRE(calls == 1);
}

TEST_CASE("an empty range invokes the body zero times", "[executor]") {
    for (std::size_t threads : {1u, 4u}) {
        auto exec = vrp::makeExecutor(threads);
        std::atomic<int> calls{0};
        exec->parallelFor(0, [&](std::size_t, std::size_t) { calls++; });
        REQUIRE(calls.load() == 0);
    }
}

TEST_CASE("the pool is reusable across many calls", "[executor]") {
    auto exec = vrp::makeExecutor(4);
    std::vector<int> counts(50, 0);
    for (int round = 0; round < 100; ++round) {
        exec->parallelFor(50, [&](std::size_t begin, std::size_t end) {
            for (std::size_t i = begin; i < end; ++i) {
                counts[i]++;
            }
        });
    }
    for (int c : counts) {
        REQUIRE(c == 100);
    }
}

TEST_CASE("work is accumulated correctly under contention", "[executor]") {
    auto exec = vrp::makeExecutor(8);
    constexpr std::size_t kN = 10000;
    std::vector<std::size_t> values(kN, 0);
    // += rather than =: assignment is idempotent and would be blind to a chunk
    // visited twice.
    exec->parallelFor(kN, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            values[i] += i * 2;
        }
    });
    const std::size_t sum = std::accumulate(values.begin(), values.end(), std::size_t{0});
    REQUIRE(sum == (kN - 1) * kN);  // 2 * sum(0..kN-1)
}

// A pool that cached the first body pointer passes every test above: they only
// ever hand one executor a single body.
TEST_CASE("each call uses the body it was given, not a cached one", "[executor]") {
    auto exec = vrp::makeExecutor(4);
    constexpr std::size_t kN = 64;
    std::vector<int> ones(kN, 0);
    std::vector<int> twos(kN, 0);

    const vrp::ParallelBody writeOnes = [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            ones[i] = 1;
        }
    };
    const vrp::ParallelBody writeTwos = [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            twos[i] = 2;
        }
    };

    exec->parallelFor(kN, writeOnes);
    exec->parallelFor(kN, writeTwos);

    for (std::size_t i = 0; i < kN; ++i) {
        INFO("i=" << i);
        REQUIRE(ones[i] == 1);
        REQUIRE(twos[i] == 2);
    }
}

// The sleep is load-bearing: chunk 0 throws while chunks 1..3 are still in
// flight, the only ordering in which failing to drain corrupts the pool.
TEST_CASE("an exception from the caller chunk leaves the pool reusable", "[executor]") {
    struct BodyError {};

    auto exec = vrp::makeExecutor(4);
    constexpr std::size_t kN = 128;
    std::vector<int> counts(kN, 0);

    const vrp::ParallelBody thrower = [&](std::size_t begin, std::size_t end) {
        if (begin == 0) {
            throw BodyError{};
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        for (std::size_t i = begin; i < end; ++i) {
            counts[i]++;
        }
    };
    REQUIRE_THROWS_AS(exec->parallelFor(kN, thrower), BodyError);

    const std::size_t firstWorkerIndex = vrp::chunkRange(kN, 4, 1).first;
    for (std::size_t i = firstWorkerIndex; i < kN; ++i) {
        INFO("i=" << i);
        REQUIRE(counts[i] == 1);
    }

    std::vector<int> second(kN, 0);
    exec->parallelFor(kN, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            second[i]++;
        }
    });
    for (std::size_t i = 0; i < kN; ++i) {
        INFO("i=" << i);
        REQUIRE(second[i] == 1);
    }
}
