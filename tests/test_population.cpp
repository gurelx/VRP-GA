#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>
#include "vrp/Executor.hpp"
#include "vrp/Population.hpp"
#include "vrp/Problem.hpp"
#include "vrp/Rng.hpp"

using Catch::Matchers::WithinAbs;

namespace {

bool isPermutationOfCustomers(std::span<const int> route, std::size_t customerCount) {
    if (route.size() != customerCount) {
        return false;
    }
    std::vector<int> sorted(route.begin(), route.end());
    std::sort(sorted.begin(), sorted.end());
    for (std::size_t i = 0; i < customerCount; ++i) {
        if (sorted[i] != static_cast<int>(i + 1)) {
            return false;
        }
    }
    return true;
}

}  // namespace

TEST_CASE("every generated route is a valid permutation", "[population]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    const vrp::Population pop(500, problem, 42, *exec);

    REQUIRE(pop.size() == 500);
    REQUIRE(pop.routeLength() == problem.customerCount());
    for (std::size_t i = 0; i < pop.size(); ++i) {
        INFO("individual " << i);
        REQUIRE(isPermutationOfCustomers(pop.route(i), problem.customerCount()));
    }
}

TEST_CASE("no route contains the depot", "[population]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    const vrp::Population pop(200, problem, 7, *exec);
    for (std::size_t i = 0; i < pop.size(); ++i) {
        for (int gene : pop.route(i)) {
            REQUIRE(gene >= 1);
            REQUIRE(gene <= static_cast<int>(problem.customerCount()));
        }
    }
}

TEST_CASE("stored fitness matches independent recomputation", "[population]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    const vrp::Population pop(200, problem, 99, *exec);
    for (std::size_t i = 0; i < pop.size(); ++i) {
        REQUIRE_THAT(pop.fitness(i),
                     WithinAbs(problem.routeDistance(pop.route(i)), 1e-12));
    }
}

TEST_CASE("population construction is identical across thread counts", "[population]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    // Not a multiple of 3 or 8, so chunkRange's ragged-remainder branch carries
    // the comparison rather than the even-split path.
    constexpr std::size_t kSize = 1003;

    auto serial = vrp::makeExecutor(1);
    const vrp::Population a(kSize, problem, 12345, *serial);

    // Self-validating guard: a constructor that emitted one fixed permutation
    // for every individual would satisfy every cross-thread comparison below
    // without saying anything at all.
    REQUIRE_FALSE(std::equal(a.route(0).begin(), a.route(0).end(),
                             a.route(1).begin(), a.route(1).end()));

    for (const std::size_t threads : {std::size_t{3}, std::size_t{8}}) {
        auto threaded = vrp::makeExecutor(threads);
        // Constructed repeatedly: a divergence that only shows on some
        // interleavings -- which is what a race would actually produce -- would
        // otherwise flake in the passing direction.
        for (int attempt = 0; attempt < 3; ++attempt) {
            const vrp::Population b(kSize, problem, 12345, *threaded);
            REQUIRE(b.size() == a.size());
            REQUIRE(b.routeLength() == a.routeLength());
            for (std::size_t i = 0; i < a.size(); ++i) {
                INFO("threads " << threads << " attempt " << attempt << " individual "
                                << i);
                REQUIRE(std::equal(a.route(i).begin(), a.route(i).end(),
                                   b.route(i).begin(), b.route(i).end()));
                REQUIRE(a.fitness(i) == b.fitness(i));  // exact: same ops, same order
            }
        }
    }
}

// Agreement between executors is not conformance to the specified derivation:
// any wrong-but-consistent seeding scheme satisfies the test above unchanged.
// This one replays mixSeed(seed, kInitDomain, i) and the backward Fisher-Yates
// by hand, pinning the domain tag, the item index and the shuffle direction.
// A kInitDomain regression in particular would be invisible everywhere else,
// while silently correlating the initial population with generation 0.
TEST_CASE("the initial shuffle reproduces the specified seed stream", "[population]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    constexpr std::uint64_t kSeed = 12345;
    constexpr std::size_t kIndividual = 7;
    const vrp::Population pop(20, problem, kSeed, *exec);

    const std::size_t length = problem.customerCount();
    std::vector<int> expected(length);
    for (std::size_t j = 0; j < length; ++j) {
        expected[j] = static_cast<int>(j + 1);
    }
    vrp::Rng rng(vrp::mixSeed(kSeed, vrp::kInitDomain, kIndividual));
    for (std::size_t j = length; j > 1; --j) {
        const std::uint32_t k = rng.below(static_cast<std::uint32_t>(j));
        std::swap(expected[j - 1], expected[k]);
    }

    REQUIRE(std::equal(pop.route(kIndividual).begin(), pop.route(kIndividual).end(),
                       expected.begin(), expected.end()));
}

TEST_CASE("setRoute updates the route and its fitness together", "[population]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    vrp::Population pop(10, problem, 1, *exec);

    std::vector<int> replacement(problem.customerCount());
    for (std::size_t i = 0; i < replacement.size(); ++i) {
        replacement[i] = static_cast<int>(i + 1);
    }
    pop.setRoute(3, replacement, problem);

    REQUIRE(std::equal(pop.route(3).begin(), pop.route(3).end(), replacement.begin()));
    REQUIRE_THAT(pop.fitness(3), WithinAbs(problem.routeDistance(replacement), 1e-12));
}

TEST_CASE("bestIndex and worstIndex break ties toward the lowest index",
          "[population]") {
    // Two locations means every route is the single-customer tour, so all
    // fitness values tie and index 0 must win both queries.
    const vrp::Problem problem(std::vector<vrp::Point>{{0, 0}, {3, 4}});
    auto exec = vrp::makeExecutor(1);
    const vrp::Population pop(5, problem, 1, *exec);
    REQUIRE(pop.bestIndex() == 0);
    REQUIRE(pop.worstIndex() == 0);
}

TEST_CASE("bestIndex and worstIndex find the extremes", "[population]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    const vrp::Population pop(300, problem, 5, *exec);

    const std::size_t best = pop.bestIndex();
    const std::size_t worst = pop.worstIndex();
    for (std::size_t i = 0; i < pop.size(); ++i) {
        REQUIRE(pop.fitness(best) <= pop.fitness(i));
        REQUIRE(pop.fitness(worst) >= pop.fitness(i));
    }
}

// SUBSUMED, kept for the postcondition it documents: this case CANNOT detect an
// evaluateAll that does nothing. Reversing a route walks the same cycle
// backwards, and on a symmetric distance matrix that is the same multiset of
// edges, so the tour length is unchanged to within summation rounding (~1e-14,
// inside the tolerance below). "evaluateAll replaces fitness a direct route
// edit made stale" is the case that actually covers the no-op.
TEST_CASE("evaluateAll refreshes every fitness value", "[population]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(4);
    vrp::Population pop(400, problem, 3, *exec);

    // Reverse each route in place, bypassing setRoute so fitness goes stale.
    for (std::size_t i = 0; i < pop.size(); ++i) {
        std::span<int> r = pop.route(i);
        std::reverse(r.begin(), r.end());
    }
    pop.evaluateAll(problem, *exec);

    for (std::size_t i = 0; i < pop.size(); ++i) {
        REQUIRE_THAT(pop.fitness(i),
                     WithinAbs(problem.routeDistance(pop.route(i)), 1e-12));
    }
}

TEST_CASE("the shape constructor allocates without randomising", "[population]") {
    const vrp::Population pop(7, 19);
    REQUIRE(pop.size() == 7);
    REQUIRE(pop.routeLength() == 19);
}

TEST_CASE("swap exchanges contents", "[population]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    vrp::Population a(10, problem, 1, *exec);
    vrp::Population b(10, problem, 2, *exec);

    const std::vector<int> firstOfA(a.route(0).begin(), a.route(0).end());
    const std::vector<int> firstOfB(b.route(0).begin(), b.route(0).end());
    REQUIRE(firstOfA != firstOfB);

    a.swap(b);
    REQUIRE(std::equal(a.route(0).begin(), a.route(0).end(), firstOfB.begin()));
    REQUIRE(std::equal(b.route(0).begin(), b.route(0).end(), firstOfA.begin()));
}

// ---------------------------------------------------------------------------
// The cases below close gaps the ones above leave open. Each names the wrong
// implementation it exists to catch.
// ---------------------------------------------------------------------------

TEST_CASE("a default-constructed population is empty", "[population]") {
    const vrp::Population pop;
    REQUIRE(pop.size() == 0);
    REQUIRE(pop.routeLength() == 0);
}

// Catches a Fisher-Yates loop that stops one iteration early: the routes stay
// valid permutations, so every other test still passes, but one position would
// hold the same customer in every individual.
TEST_CASE("the shuffle reaches every position", "[population]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    const vrp::Population pop(500, problem, 2024, *exec);
    const std::size_t length = pop.routeLength();

    for (std::size_t j = 0; j < length; ++j) {
        std::vector<bool> seen(length + 1, false);
        for (std::size_t i = 0; i < pop.size(); ++i) {
            seen[static_cast<std::size_t>(pop.route(i)[j])] = true;
        }
        const std::size_t distinct =
            static_cast<std::size_t>(std::count(seen.begin(), seen.end(), true));
        INFO("position " << j << " saw " << distinct << " distinct customers");
        REQUIRE(distinct == length);
    }
}

// The reversal used above is distance-preserving on a symmetric matrix, so it
// cannot tell evaluateAll apart from a no-op. A rotation moves which customers
// neighbour the depot and therefore does change the tour length.
TEST_CASE("evaluateAll replaces fitness a direct route edit made stale",
          "[population]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(4);
    vrp::Population pop(400, problem, 3, *exec);

    std::vector<double> before(pop.size());
    for (std::size_t i = 0; i < pop.size(); ++i) {
        before[i] = pop.fitness(i);
    }
    for (std::size_t i = 0; i < pop.size(); ++i) {
        std::span<int> r = pop.route(i);
        std::rotate(r.begin(), r.begin() + 1, r.end());
    }
    pop.evaluateAll(problem, *exec);

    std::size_t changed = 0;
    for (std::size_t i = 0; i < pop.size(); ++i) {
        INFO("individual " << i);
        REQUIRE_THAT(pop.fitness(i),
                     WithinAbs(problem.routeDistance(pop.route(i)), 1e-12));
        if (pop.fitness(i) != before[i]) {
            ++changed;
        }
    }
    REQUIRE(changed > pop.size() / 2);
}

TEST_CASE("the shape constructor zero-fills routes and fitness", "[population]") {
    const vrp::Population pop(7, 19);
    // Asserted here rather than borrowed from the case above: without it a
    // non-allocating constructor would be detected only by the loop below
    // reading out of bounds, which is not a detection at all.
    REQUIRE(pop.size() == 7);
    for (std::size_t i = 0; i < pop.size(); ++i) {
        INFO("individual " << i);
        REQUIRE(pop.route(i).size() == 19);
        for (int gene : pop.route(i)) {
            REQUIRE(gene == 0);
        }
        REQUIRE(pop.fitness(i) == 0.0);
    }
}

// Catches a stride error in setRoute. Individual 0 and the LAST individual are
// the two that matter: a stride bug at the end walks off the buffer rather than
// merely into a neighbour.
TEST_CASE("setRoute leaves the other individuals untouched", "[population]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    constexpr std::size_t kSize = 10;
    vrp::Population pop(kSize, problem, 4, *exec);

    std::vector<std::vector<int>> before;
    std::vector<double> beforeFitness;
    for (std::size_t i = 0; i < kSize; ++i) {
        before.emplace_back(pop.route(i).begin(), pop.route(i).end());
        beforeFitness.push_back(pop.fitness(i));
    }

    std::vector<int> replacement(problem.customerCount());
    for (std::size_t i = 0; i < replacement.size(); ++i) {
        replacement[i] = static_cast<int>(problem.customerCount() - i);
    }
    pop.setRoute(0, replacement, problem);
    pop.setRoute(kSize - 1, replacement, problem);

    for (const std::size_t i : {std::size_t{0}, kSize - 1}) {
        INFO("rewritten individual " << i);
        REQUIRE(std::equal(pop.route(i).begin(), pop.route(i).end(),
                           replacement.begin(), replacement.end()));
        REQUIRE_THAT(pop.fitness(i),
                     WithinAbs(problem.routeDistance(replacement), 1e-12));
    }
    for (std::size_t i = 1; i + 1 < kSize; ++i) {
        INFO("untouched individual " << i);
        REQUIRE(std::equal(pop.route(i).begin(), pop.route(i).end(),
                           before[i].begin(), before[i].end()));
        REQUIRE(pop.fitness(i) == beforeFitness[i]);
    }
}

// setRoute must be callable from inside a parallelFor for distinct i, which is
// exactly how the offspring buffer is filled later on.
TEST_CASE("setRoute is safe to call concurrently for distinct individuals",
          "[population]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto threaded = vrp::makeExecutor(8);
    vrp::Population destination(1000, problem, 11, *threaded);
    const vrp::Population source(1000, problem, 22, *threaded);

    threaded->parallelFor(destination.size(), [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            destination.setRoute(i, source.route(i), problem);
        }
    });

    for (std::size_t i = 0; i < destination.size(); ++i) {
        INFO("individual " << i);
        REQUIRE(std::equal(destination.route(i).begin(), destination.route(i).end(),
                           source.route(i).begin(), source.route(i).end()));
        REQUIRE(destination.fitness(i) == source.fitness(i));
    }
}

TEST_CASE("swap exchanges routes, fitness and shape", "[population]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    vrp::Population a(10, problem, 1, *exec);
    vrp::Population b(4, 3);

    const std::vector<int> firstRouteOfA(a.route(0).begin(), a.route(0).end());
    const double firstFitnessOfA = a.fitness(0);
    REQUIRE(firstFitnessOfA != 0.0);

    a.swap(b);

    REQUIRE(a.size() == 4);
    REQUIRE(a.routeLength() == 3);
    REQUIRE(a.fitness(0) == 0.0);
    for (int gene : a.route(0)) {
        REQUIRE(gene == 0);
    }

    REQUIRE(b.size() == 10);
    REQUIRE(b.routeLength() == problem.customerCount());
    REQUIRE(b.fitness(0) == firstFitnessOfA);
    REQUIRE(std::equal(b.route(0).begin(), b.route(0).end(), firstRouteOfA.begin(),
                       firstRouteOfA.end()));
}

// The hidden friend is what makes the two-step form -- which the standard
// algorithms use internally -- select the buffer swap over three moves.
TEST_CASE("an unqualified swap finds the member through ADL", "[population]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    vrp::Population a(10, problem, 1, *exec);
    vrp::Population b(4, 3);

    const std::vector<int> firstRouteOfA(a.route(0).begin(), a.route(0).end());

    using std::swap;
    swap(a, b);

    REQUIRE(a.size() == 4);
    REQUIRE(a.routeLength() == 3);
    REQUIRE(b.size() == 10);
    REQUIRE(std::equal(b.route(0).begin(), b.route(0).end(), firstRouteOfA.begin(),
                       firstRouteOfA.end()));
}
