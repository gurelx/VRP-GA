#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

#include "vrp/Executor.hpp"
#include "vrp/Operators.hpp"
#include "vrp/Population.hpp"
#include "vrp/Problem.hpp"
#include "vrp/Rng.hpp"

namespace {

std::vector<int> identityRoute(std::size_t n) {
    std::vector<int> r(n);
    std::iota(r.begin(), r.end(), 1);
    return r;
}

bool isPermutation(std::span<const int> route, std::size_t n) {
    if (route.size() != n) {
        return false;
    }
    std::vector<int> sorted(route.begin(), route.end());
    std::sort(sorted.begin(), sorted.end());
    for (std::size_t i = 0; i < n; ++i) {
        if (sorted[i] != static_cast<int>(i + 1)) {
            return false;
        }
    }
    return true;
}

std::vector<int> shuffledRoute(std::size_t n, vrp::Rng& rng) {
    std::vector<int> r = identityRoute(n);
    for (std::size_t j = n; j > 1; --j) {
        std::swap(r[j - 1], r[rng.below(static_cast<std::uint32_t>(j))]);
    }
    return r;
}

// A deliberately naive, independent restatement of order crossover: copy
// p1[lo..hi] verbatim, then walk p2 and drop the genes already present by a
// linear std::find over the segment. It shares no logic with the seen-bitmask
// implementation, so agreement between the two is evidence rather than
// tautology.
std::vector<int> referenceOrderCrossover(const std::vector<int>& p1,
                                         const std::vector<int>& p2, std::size_t lo,
                                         std::size_t hi) {
    const std::size_t n = p1.size();
    std::vector<int> child(n, 0);
    for (std::size_t i = lo; i <= hi; ++i) {
        child[i] = p1[i];
    }

    std::vector<std::size_t> freeSlots;
    for (std::size_t i = 0; i < n; ++i) {
        if (i < lo || i > hi) {
            freeSlots.push_back(i);
        }
    }

    const auto segBegin = p1.begin() + static_cast<std::ptrdiff_t>(lo);
    const auto segEnd = p1.begin() + static_cast<std::ptrdiff_t>(hi) + 1;
    std::size_t k = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const int gene = p2[i];
        if (std::find(segBegin, segEnd, gene) != segEnd) {
            continue;
        }
        // The oracle relies on the same permutation precondition the operator
        // does. Guard it so a future fixture with mismatched parents fails an
        // assertion here rather than running the oracle off the end.
        REQUIRE(k < freeSlots.size());
        child[freeSlots[k]] = gene;
        ++k;
    }
    return child;
}

// orderCrossover draws its segment as two below(n) values and orders them, so a
// mirror Rng replays exactly which segment a given seed selects.
std::pair<std::size_t, std::size_t> replaySegment(std::uint64_t seed, std::size_t n) {
    vrp::Rng mirror(seed);
    auto lo = static_cast<std::size_t>(mirror.below(static_cast<std::uint32_t>(n)));
    auto hi = static_cast<std::size_t>(mirror.below(static_cast<std::uint32_t>(n)));
    if (lo > hi) {
        std::swap(lo, hi);
    }
    return {lo, hi};
}

}  // namespace

TEST_CASE("crossover always produces a valid permutation", "[operators]") {
    constexpr std::size_t kN = 19;
    std::vector<int> p1 = identityRoute(kN);
    std::vector<int> p2 = identityRoute(kN);
    std::reverse(p2.begin(), p2.end());

    std::vector<int> child(kN);
    std::vector<char> seen(kN + 1, 0);

    for (std::uint64_t seed = 0; seed < 2000; ++seed) {
        vrp::Rng rng(seed);
        vrp::ops::orderCrossover(p1, p2, child, seen, rng);
        INFO("seed " << seed);
        REQUIRE(isPermutation(child, kN));
    }
}

TEST_CASE("crossover handles shuffled parents", "[operators]") {
    constexpr std::size_t kN = 19;
    std::vector<int> child(kN);
    std::vector<char> seen(kN + 1, 0);

    for (std::uint64_t seed = 0; seed < 500; ++seed) {
        vrp::Rng shuffler(seed + 100000);
        std::vector<int> p1 = shuffledRoute(kN, shuffler);
        std::vector<int> p2 = shuffledRoute(kN, shuffler);
        vrp::Rng rng(seed);
        vrp::ops::orderCrossover(p1, p2, child, seen, rng);
        INFO("seed " << seed);
        REQUIRE(isPermutation(child, kN));
    }
}

TEST_CASE("crossover of identical parents reproduces that parent", "[operators]") {
    constexpr std::size_t kN = 19;
    const std::vector<int> p1 = identityRoute(kN);
    const std::vector<int> p2 = identityRoute(kN);
    std::vector<int> child(kN);
    std::vector<char> seen(kN + 1, 0);

    for (std::uint64_t seed = 0; seed < 200; ++seed) {
        vrp::Rng rng(seed);
        vrp::ops::orderCrossover(p1, p2, child, seen, rng);
        INFO("seed " << seed);
        REQUIRE(child == p1);
    }
}

// "Output is a permutation" accepts an operator that ignores a parent outright:
// sourcing the segment from p2 instead of p1 collapses the whole result to p2,
// and copying p1 wholesale ignores p2. Both are still permutations. This pins
// the structure that separates them.
TEST_CASE("crossover recombines instead of echoing one parent", "[operators]") {
    constexpr std::size_t kN = 19;
    const std::vector<int> p1 = identityRoute(kN);
    std::vector<int> p2 = identityRoute(kN);
    std::reverse(p2.begin(), p2.end());

    std::vector<int> child(kN);
    std::vector<char> seen(kN + 1, 0);

    std::size_t partialSegments = 0;
    bool sawLowAtZero = false;
    bool sawHighAtEnd = false;
    bool sawSingleton = false;
    for (std::uint64_t seed = 0; seed < 1000; ++seed) {
        const auto [lo, hi] = replaySegment(seed, kN);
        sawLowAtZero = sawLowAtZero || lo == 0;
        sawHighAtEnd = sawHighAtEnd || hi == kN - 1;
        sawSingleton = sawSingleton || lo == hi;
        vrp::Rng rng(seed);
        vrp::ops::orderCrossover(p1, p2, child, seen, rng);
        INFO("seed " << seed << " segment " << lo << ".." << hi);

        // The copied segment must be p1's genes, at p1's positions.
        for (std::size_t i = lo; i <= hi; ++i) {
            REQUIRE(child[i] == p1[i]);
        }

        const std::size_t segmentLength = hi - lo + 1;
        if (segmentLength >= 2) {
            // p1 and p2 agree only at the single middle index here, so a
            // segment of two or more positions cannot leave the child equal to
            // p2 -- which is exactly what a p2-sourced segment would produce.
            REQUIRE(child != p2);
        }
        if (kN - segmentLength >= 2) {
            // Two or more genes come from p2, in p2's (reversed) order, so the
            // child cannot still be p1.
            REQUIRE(child != p1);
            ++partialSegments;
        }
    }
    REQUIRE(partialSegments > 0);
    // Boundary coverage is a property of the seed range, not of the operator.
    // Assert it rather than trusting it, so a later change to kN or to the seed
    // count cannot quietly stop exercising the edges the write cursor turns on.
    REQUIRE(sawLowAtZero);
    REQUIRE(sawHighAtEnd);
    REQUIRE(sawSingleton);
}

TEST_CASE("crossover matches an independent reference implementation", "[operators]") {
    constexpr std::size_t kN = 19;
    std::vector<int> child(kN);
    std::vector<char> seen(kN + 1, 0);

    bool sawLowAtZero = false;
    bool sawHighAtEnd = false;
    bool sawSingleton = false;
    for (std::uint64_t seed = 0; seed < 1000; ++seed) {
        vrp::Rng shuffler(seed + 900000);
        const std::vector<int> p1 = shuffledRoute(kN, shuffler);
        const std::vector<int> p2 = shuffledRoute(kN, shuffler);

        const auto [lo, hi] = replaySegment(seed, kN);
        sawLowAtZero = sawLowAtZero || lo == 0;
        sawHighAtEnd = sawHighAtEnd || hi == kN - 1;
        sawSingleton = sawSingleton || lo == hi;
        const std::vector<int> expected = referenceOrderCrossover(p1, p2, lo, hi);

        vrp::Rng rng(seed);
        vrp::ops::orderCrossover(p1, p2, child, seen, rng);
        INFO("seed " << seed << " segment " << lo << ".." << hi);
        // Distinct parents are what makes the comparison meaningful; identical
        // ones would satisfy the oracle for an operator that ignored p2.
        REQUIRE(p1 != p2);
        REQUIRE(child == expected);
    }
    REQUIRE(sawLowAtZero);
    REQUIRE(sawHighAtEnd);
    REQUIRE(sawSingleton);
}

// Task 7 hoists one scratch buffer out of the offspring loop and hands it to
// every crossover in a chunk, each with different parents. A stale mask would
// silently drop genes, so the reused buffer must give bit-identical results to
// a fresh one.
TEST_CASE("crossover clears its scratch buffer on entry", "[operators]") {
    constexpr std::size_t kN = 19;
    std::vector<char> reused(kN + 1, 0);
    std::vector<int> viaReused(kN);
    std::vector<int> viaFresh(kN);

    for (std::uint64_t seed = 0; seed < 500; ++seed) {
        vrp::Rng shuffler(seed + 700000);
        const std::vector<int> p1 = shuffledRoute(kN, shuffler);
        const std::vector<int> p2 = shuffledRoute(kN, shuffler);

        vrp::Rng a(seed);
        vrp::ops::orderCrossover(p1, p2, viaReused, reused, a);

        std::vector<char> fresh(kN + 1, 0);
        vrp::Rng b(seed);
        vrp::ops::orderCrossover(p1, p2, viaFresh, fresh, b);

        INFO("seed " << seed);
        REQUIRE(isPermutation(viaReused, kN));
        REQUIRE(viaReused == viaFresh);
    }
}

// SUBSUMED, kept as documentation of the calling pattern. Every property this
// checks is a strict subset of "crossover clears its scratch buffer on entry",
// which runs 500 different parent pairs through a reused buffer and requires
// bit-identical results against a fresh one. This case cannot fail unless that
// one does.
TEST_CASE("crossover reuses the scratch buffer safely", "[operators]") {
    constexpr std::size_t kN = 19;
    std::vector<int> p1 = identityRoute(kN);
    std::vector<int> p2 = identityRoute(kN);
    std::reverse(p2.begin(), p2.end());
    std::vector<char> seen(kN + 1, 0);

    std::vector<int> first(kN);
    std::vector<int> second(kN);
    vrp::Rng a(555);
    vrp::ops::orderCrossover(p1, p2, first, seen, a);
    vrp::ops::orderCrossover(p1, p2, second, seen, a);
    REQUIRE(isPermutation(first, kN));
    REQUIRE(isPermutation(second, kN));
}

TEST_CASE("mutation rate 0 never alters a route", "[operators]") {
    std::vector<int> route = identityRoute(19);
    const std::vector<int> original = route;
    vrp::Rng rng(1);
    for (int i = 0; i < 5000; ++i) {
        vrp::ops::swapMutate(route, 0.0, rng);
    }
    REQUIRE(route == original);
}

// The rate test above can only fail on a draw of exactly 0.0, which a random
// seed will never produce. xoshiro256** emits rotl(s[1] * 5, 7) * 9, so a state
// whose second word is zero emits 0, and unit() is then exactly 0.0. That is
// the one input separating `>= rate` from `> rate` at rate 0.
TEST_CASE("mutation rate 0 declines even on a zero draw", "[operators]") {
    vrp::Rng rng = vrp::Rng::fromState({1, 0, 0, 0});
    vrp::Rng probe = vrp::Rng::fromState({1, 0, 0, 0});
    REQUIRE(probe.unit() == 0.0);

    std::vector<int> route = identityRoute(19);
    const std::vector<int> original = route;
    vrp::ops::swapMutate(route, 0.0, rng);
    REQUIRE(route == original);
}

TEST_CASE("mutation rate 1 always swaps two distinct positions", "[operators]") {
    for (std::uint64_t seed = 0; seed < 2000; ++seed) {
        std::vector<int> route = identityRoute(19);
        vrp::Rng rng(seed);
        vrp::ops::swapMutate(route, 1.0, rng);
        INFO("seed " << seed);
        REQUIRE(isPermutation(route, 19));
        // Exactly two positions must differ; a self-swap would leave zero.
        std::size_t differences = 0;
        for (std::size_t i = 0; i < route.size(); ++i) {
            if (route[i] != static_cast<int>(i + 1)) {
                differences++;
            }
        }
        REQUIRE(differences == 2);
    }
}

// The exclusion rule must not bias the second index: over many seeds every
// ordered pair of distinct positions should be reachable.
TEST_CASE("mutation reaches every distinct pair of positions", "[operators]") {
    constexpr std::size_t kN = 6;
    std::vector<std::vector<int>> hits(kN, std::vector<int>(kN, 0));
    for (std::uint64_t seed = 0; seed < 20000; ++seed) {
        std::vector<int> route = identityRoute(kN);
        vrp::Rng rng(seed);
        vrp::ops::swapMutate(route, 1.0, rng);
        std::size_t a = kN;
        std::size_t b = kN;
        for (std::size_t i = 0; i < kN; ++i) {
            if (route[i] != static_cast<int>(i + 1)) {
                (a == kN ? a : b) = i;
            }
        }
        REQUIRE(a < kN);
        REQUIRE(b < kN);
        hits[a][b]++;
    }
    for (std::size_t i = 0; i < kN; ++i) {
        for (std::size_t j = 0; j < kN; ++j) {
            INFO("pair " << i << "," << j);
            if (i == j) {
                REQUIRE(hits[i][j] == 0);
            } else if (i < j) {
                REQUIRE(hits[i][j] > 0);
            }
        }
    }
}

// SUBSUMED. Swapping two in-range positions preserves a permutation for ANY
// choice of indices, so this cannot fail against any implementation that
// swaps -- including one that self-swaps or ignores the rate entirely. The
// properties it looks like it covers are actually covered by "mutation rate 1
// always swaps two distinct positions" (exactly two positions change) and
// "mutation reaches every distinct pair of positions" (index selection).
TEST_CASE("mutation preserves permutations under repeated application",
          "[operators]") {
    std::vector<int> route = identityRoute(19);
    vrp::Rng rng(42);
    for (int i = 0; i < 10000; ++i) {
        vrp::ops::swapMutate(route, 0.5, rng);
        REQUIRE(isPermutation(route, 19));
    }
}

TEST_CASE("mutation at rate one half sometimes fires and sometimes declines",
          "[operators]") {
    std::size_t fired = 0;
    for (std::uint64_t seed = 0; seed < 1000; ++seed) {
        std::vector<int> route = identityRoute(19);
        const std::vector<int> original = route;
        vrp::Rng rng(seed);
        vrp::ops::swapMutate(route, 0.5, rng);
        if (route != original) {
            ++fired;
        }
    }
    REQUIRE(fired > 300);
    REQUIRE(fired < 700);
}

TEST_CASE("mutation leaves a one-element route alone", "[operators]") {
    std::vector<int> route{1};
    vrp::Rng rng(1);
    vrp::ops::swapMutate(route, 1.0, rng);
    REQUIRE(route == std::vector<int>{1});
}

TEST_CASE("mutation leaves an empty route alone", "[operators]") {
    std::vector<int> route;
    vrp::Rng rng(1);
    vrp::ops::swapMutate(route, 1.0, rng);
    REQUIRE(route.empty());
}

// SUBSUMED. Rng::below(popSize) already returns a value below popSize, so this
// holds for any implementation that returns a drawn index at all -- including
// one that picks the worst, or draws the wrong number of times. The real
// coverage is "tournament selection returns the fittest drawn candidate" and
// "tournament selection draws exactly tournamentSize candidates".
TEST_CASE("tournament selection returns an in-range index", "[operators]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    const vrp::Population pop(100, problem, 11, *exec);
    vrp::Rng rng(3);
    for (int i = 0; i < 5000; ++i) {
        REQUIRE(vrp::ops::tournamentSelect(pop, 5, rng) < pop.size());
    }
}

TEST_CASE("tournament selection favours fitter individuals", "[operators]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    const vrp::Population pop(1000, problem, 13, *exec);

    // Mean fitness of tournament winners must beat the population mean.
    double populationMean = 0.0;
    for (std::size_t i = 0; i < pop.size(); ++i) {
        populationMean += pop.fitness(i);
    }
    populationMean /= static_cast<double>(pop.size());

    vrp::Rng rng(17);
    constexpr int kDraws = 5000;
    double winnerMean = 0.0;
    for (int i = 0; i < kDraws; ++i) {
        winnerMean += pop.fitness(vrp::ops::tournamentSelect(pop, 5, rng));
    }
    winnerMean /= static_cast<double>(kDraws);

    REQUIRE(winnerMean < populationMean);
}

// A winner-beats-the-mean check survives an off-by-one in the draw count: four
// draws still beat the mean. Pin the count by consuming the same number of
// below() values from a mirror generator and requiring the two generators to
// be left in the same state.
TEST_CASE("tournament selection draws exactly tournamentSize candidates",
          "[operators]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    const vrp::Population pop(64, problem, 23, *exec);
    const auto popSize = static_cast<std::uint32_t>(pop.size());

    for (const std::size_t k : {std::size_t{1}, std::size_t{2}, std::size_t{5},
                                std::size_t{16}}) {
        for (std::uint64_t seed = 0; seed < 200; ++seed) {
            vrp::Rng actual(seed);
            vrp::Rng mirror(seed);
            (void)vrp::ops::tournamentSelect(pop, k, actual);
            for (std::size_t d = 0; d < k; ++d) {
                (void)mirror.below(popSize);
            }
            INFO("tournamentSize " << k << " seed " << seed);
            for (int probe = 0; probe < 4; ++probe) {
                REQUIRE(actual.next() == mirror.next());
            }
        }
    }
}

TEST_CASE("tournament selection returns the fittest drawn candidate", "[operators]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    const vrp::Population pop(64, problem, 29, *exec);
    const auto popSize = static_cast<std::uint32_t>(pop.size());
    constexpr std::size_t kSize = 5;

    for (std::uint64_t seed = 0; seed < 2000; ++seed) {
        vrp::Rng actual(seed);
        vrp::Rng mirror(seed);
        const std::size_t got = vrp::ops::tournamentSelect(pop, kSize, actual);

        std::size_t best = mirror.below(popSize);
        for (std::size_t d = 1; d < kSize; ++d) {
            const std::size_t candidate = mirror.below(popSize);
            if (pop.fitness(candidate) < pop.fitness(best)) {
                best = candidate;
            }
        }
        INFO("seed " << seed);
        REQUIRE(got == best);
    }
}

TEST_CASE("tournament selection of size one returns the drawn index", "[operators]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    const vrp::Population pop(64, problem, 31, *exec);
    const auto popSize = static_cast<std::uint32_t>(pop.size());

    for (std::uint64_t seed = 0; seed < 1000; ++seed) {
        vrp::Rng actual(seed);
        vrp::Rng mirror(seed);
        const std::size_t got = vrp::ops::tournamentSelect(pop, 1, actual);
        INFO("seed " << seed);
        REQUIRE(got == static_cast<std::size_t>(mirror.below(popSize)));
    }
}

TEST_CASE("tournament selection treats size zero as a single draw", "[operators]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    const vrp::Population pop(64, problem, 37, *exec);
    const auto popSize = static_cast<std::uint32_t>(pop.size());

    for (std::uint64_t seed = 0; seed < 1000; ++seed) {
        vrp::Rng actual(seed);
        vrp::Rng mirror(seed);
        const std::size_t got = vrp::ops::tournamentSelect(pop, 0, actual);
        INFO("seed " << seed);
        REQUIRE(got == static_cast<std::size_t>(mirror.below(popSize)));
    }
}

// Stronger tournaments must select harder. This is independent of the mirror
// replays above: it reads only the operator's observable selection pressure.
TEST_CASE("larger tournaments select more aggressively", "[operators]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    const vrp::Population pop(500, problem, 41, *exec);

    const auto meanWinner = [&pop](std::size_t size) {
        vrp::Rng rng(9);
        constexpr int kDraws = 4000;
        double total = 0.0;
        for (int i = 0; i < kDraws; ++i) {
            total += pop.fitness(vrp::ops::tournamentSelect(pop, size, rng));
        }
        return total / static_cast<double>(kDraws);
    };

    REQUIRE(meanWinner(8) < meanWinner(2));
    REQUIRE(meanWinner(2) < meanWinner(1));
}

// Task 7 seeds one generator per work item, so an operator that quietly changes
// how many draws it takes shifts every downstream stream position without
// changing any single offspring in a way these tests would otherwise notice.
// replaySegment mirrors only the FIRST two draws, so a spurious third draw
// appended here would pass every other crossover case in this file, the
// reference oracle included. Pin the count directly: run a mirror generator
// forward by the expected draws and require both to be at the same position.
TEST_CASE("crossover consumes exactly two segment draws", "[operators]") {
    constexpr std::size_t kN = 19;
    const auto n32 = static_cast<std::uint32_t>(kN);
    std::vector<int> child(kN);
    std::vector<char> seen(kN + 1, 0);

    for (std::uint64_t seed = 0; seed < 500; ++seed) {
        vrp::Rng shuffler(seed + 300000);
        const std::vector<int> p1 = shuffledRoute(kN, shuffler);
        const std::vector<int> p2 = shuffledRoute(kN, shuffler);

        vrp::Rng actual(seed);
        vrp::Rng mirror(seed);
        vrp::ops::orderCrossover(p1, p2, child, seen, actual);
        (void)mirror.below(n32);
        (void)mirror.below(n32);

        INFO("seed " << seed);
        for (int probe = 0; probe < 4; ++probe) {
            REQUIRE(actual.next() == mirror.next());
        }
    }
}

// Same pin for mutation, across both of its paths: one unit() draw when the
// rate declines, that plus two below() draws when it fires. The branch taken is
// cross-checked against the mirror's own draw, so this also pins that the
// decision is `unit() < rate` rather than something merely correlated with it.
TEST_CASE("mutation consumes one draw when it declines and three when it fires",
          "[operators]") {
    constexpr std::size_t kN = 19;
    const auto n32 = static_cast<std::uint32_t>(kN);
    std::size_t declines = 0;
    std::size_t fires = 0;

    for (const double rate : {0.0, 0.5, 1.0}) {
        for (std::uint64_t seed = 0; seed < 400; ++seed) {
            std::vector<int> route = identityRoute(kN);
            const std::vector<int> before = route;
            vrp::Rng actual(seed);
            vrp::Rng mirror(seed);
            vrp::ops::swapMutate(route, rate, actual);

            const double draw = mirror.unit();
            INFO("rate " << rate << " seed " << seed);
            if (draw < rate) {
                (void)mirror.below(n32);
                (void)mirror.below(n32 - 1);
                REQUIRE(route != before);
                ++fires;
            } else {
                REQUIRE(route == before);
                ++declines;
            }
            for (int probe = 0; probe < 4; ++probe) {
                REQUIRE(actual.next() == mirror.next());
            }
        }
    }
    REQUIRE(declines > 0);
    REQUIRE(fires > 0);
}

// The short-route early return must consume nothing at all. This is the one
// asymmetry in the operator's stream footprint, and the only thing keeping it
// invisible is that every route in a population is currently the same length.
TEST_CASE("mutation consumes no draws on a route shorter than two", "[operators]") {
    for (std::uint64_t seed = 0; seed < 200; ++seed) {
        std::vector<int> single{1};
        std::vector<int> empty;
        vrp::Rng actual(seed);
        vrp::Rng mirror(seed);
        vrp::ops::swapMutate(single, 1.0, actual);
        vrp::ops::swapMutate(empty, 1.0, actual);

        INFO("seed " << seed);
        for (int probe = 0; probe < 4; ++probe) {
            REQUIRE(actual.next() == mirror.next());
        }
    }
}

// n == 1 forces below(1) == 0 twice, so low == high == 0: the segment is the
// whole route and the fill loop finds nothing to place.
TEST_CASE("crossover of a single-gene route reproduces that gene", "[operators]") {
    const std::vector<int> p1{1};
    const std::vector<int> p2{1};
    std::vector<int> child(1, 0);
    std::vector<char> seen(2, 0);

    for (std::uint64_t seed = 0; seed < 200; ++seed) {
        vrp::Rng rng(seed);
        vrp::ops::orderCrossover(p1, p2, child, seen, rng);
        INFO("seed " << seed);
        REQUIRE(child == p1);
    }
}

TEST_CASE("tournament selection on a single-individual population", "[operators]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    const vrp::Population pop(1, problem, 43, *exec);
    REQUIRE(pop.size() == 1);

    vrp::Rng rng(5);
    for (const std::size_t k : {std::size_t{0}, std::size_t{1}, std::size_t{7}}) {
        for (int i = 0; i < 200; ++i) {
            INFO("tournamentSize " << k);
            REQUIRE(vrp::ops::tournamentSelect(pop, k, rng) == 0);
        }
    }
}
