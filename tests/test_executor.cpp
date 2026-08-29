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

// Records how many times each index was visited, across any thread count.
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
                REQUIRE(begin == previousEnd);  // contiguous, no gaps
                REQUIRE(end <= n);
                covered += end - begin;
                previousEnd = end;
            }
            REQUIRE(covered == n);      // full coverage
            REQUIRE(previousEnd == n);  // ends exactly at n
        }
    }
}

// A lost wakeup or a missed decrement makes parallelFor block forever, and a
// suite that merely hangs reports nothing at all. Here the wait is bounded, so
// the failure arrives as a named assertion instead.
//
// Declared here on purpose: Catch2 runs test cases in declaration order, so this
// is the first pool-using case to run and therefore the first to notice a wedged
// pool, before any unbounded test can hang ahead of it.
TEST_CASE("parallelFor finishes within a bounded time", "[executor]") {
    constexpr std::size_t kN = 256;
    // A lost wakeup is itself a race, and a mild one: the notify usually wakes a
    // worker that re-checks its predicate only after the publisher has released
    // the mutex, by which point the new epoch is visible and nothing is lost.
    // So the round count is the detector, and it must stay high.
    //
    // Measured against a notify-before-publish mutant:
    //     10 rounds     5/30   (17%)
    //    300 rounds    16/30   (53%)
    //   3000 rounds    85/85   (100%)
    // Rounds keep paying all the way to 3000 -- the observed round-at-detection
    // peaked around 1288, comfortably inside this cap. Outcomes within one
    // process are strongly correlated (10x the rounds does not give 10x the
    // detection), but they are NOT fixed per process, so raising this number
    // genuinely buys detection and lowering it genuinely loses it.
    //
    // Treat 100% as "empirically 100% across 85 invocations on one machine",
    // not as proven deterministic. The decisive race is between two adjacent
    // statements inside parallelFor that no external thread can interleave with,
    // so no black-box test can force it: an experiment that made the workers
    // definitely parked first made detection worse (10%), not better. A
    // deterministic detector would need a test-only seam inside the production
    // synchronisation, which is deliberately not on offer.
    //
    // Costs ~0.3-0.5 s against the 60 s CTest timeout. Do not trim it to make
    // the suite faster; that trades away the only lost-wakeup detector the
    // project has, on a platform with no TSan behind it.
    constexpr int kRounds = 3000;

    // State on the heap and owned by the runner task itself. If the pool ever
    // does wedge, the runner is detached rather than joined, so it must not be
    // left pointing into a destroyed test frame -- and the executor must stay
    // alive too, since ~ThreadPoolExecutor joins its workers and would hang
    // again during teardown, swallowing the report we just produced.
    struct State {
        std::unique_ptr<vrp::Executor> exec;
        std::vector<int> counts;
    };
    auto state = std::make_shared<State>(State{vrp::makeExecutor(4), std::vector<int>(kN, 0)});

    for (int round = 0; round < kRounds; ++round) {
        // packaged_task, not std::async: an async future blocks in its own
        // destructor, which would reintroduce the very hang this test exists to
        // convert into an assertion.
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
    // += rather than =, so the sum is sensitive to a chunk being visited twice.
    // A plain assignment is idempotent, which made this case blind to every
    // double-visit mutant and left it only able to detect unvisited indices.
    exec->parallelFor(kN, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            values[i] += i * 2;
        }
    });
    const std::size_t sum = std::accumulate(values.begin(), values.end(), std::size_t{0});
    REQUIRE(sum == (kN - 1) * kN);  // 2 * sum(0..kN-1)
}

// A pool that cached the first body pointer and reused it would still pass every
// test above, because they only ever hand one executor a single body. Two named,
// simultaneously live bodies writing distinct values pin that down.
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

// Fix for the caller-throws path. The workers are deliberately slow and chunk 0
// throws immediately, so the exception propagates while chunks 1..3 are still in
// flight -- the only ordering in which failing to drain actually corrupts the
// pool. Without the drain, the next parallelFor resets outstanding_ underneath
// those stale workers and their decrements underflow it.
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

    // parallelFor drained before rethrowing, so every worker chunk has finished.
    const std::size_t firstWorkerIndex = vrp::chunkRange(kN, 4, 1).first;
    for (std::size_t i = firstWorkerIndex; i < kN; ++i) {
        INFO("i=" << i);
        REQUIRE(counts[i] == 1);
    }

    // And the executor is still usable rather than wedged or underflowed.
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
