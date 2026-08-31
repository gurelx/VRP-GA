#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>
#include "vrp/Config.hpp"
#include "vrp/Executor.hpp"
#include "vrp/Population.hpp"
#include "vrp/Problem.hpp"
#include "vrp/Solver.hpp"
#include "vrp/Strategy.hpp"
#include "SolverTrace.hpp"

using vrp_test::kindName;
using vrp_test::referenceTrace;
using vrp_test::Trace;

// Every comparison in this file is DIFFERENTIAL: it runs the same library code
// twice, so a wrong-but-consistent implementation -- a transposed mixSeed, a
// crossover handed one parent twice -- passes everything here. Conformance to
// the specified seed derivation is pinned by test_population.cpp and
// test_strategies.cpp. Exact `==` on doubles is intended: per-item seeding
// makes every value bit-identical across thread counts, and a tolerance would
// hide precisely the divergence these tests exist to catch.

namespace {

// Steady-state replaces one individual per generation, so at 25 generations its
// incumbent best never moves and its progress trace is a constant, which
// discriminates nothing. Give it a count past that threshold instead.
vrp::GaParams determinismParams(std::uint64_t seed, vrp::StrategyKind kind) {
    vrp::GaParams params;
    params.populationSize = 512;
    params.generations = kind == vrp::StrategyKind::SteadyState ? 200 : 25;
    params.seed = seed;
    return params;
}

Trace solverTrace(const vrp::Problem& problem, const vrp::GaParams& params,
                  vrp::StrategyKind kind, std::size_t threads) {
    vrp::Solver solver(problem, params, vrp::makeStrategy(kind),
                       vrp::makeExecutor(threads));
    Trace trace;
    const vrp::RunResult result = solver.run([&](std::size_t generation, double best) {
        trace.generations.push_back(generation);
        trace.best.push_back(best);
    });
    trace.bestRoute = result.bestRoute;
    trace.bestDistance = result.bestDistance;
    trace.generationsRun = result.generationsRun;
    return trace;
}

vrp::RunResult runWith(const vrp::Problem& problem, vrp::StrategyKind kind,
                       std::size_t threads, std::uint64_t seed) {
    vrp::Solver solver(problem, determinismParams(seed, kind), vrp::makeStrategy(kind),
                       vrp::makeExecutor(threads));
    return solver.run();
}

// A whole population, flattened: Solver returns only the winner, so seeing
// every individual means running the loop by hand.
struct Snapshot {
    std::size_t routeLength = 0;
    std::vector<int> routes;
    std::vector<double> fitness;
    std::size_t bestIndex = 0;
    std::vector<int> bestRoute;
    double bestDistance = 0.0;
};

// Lowest-index tie-breaking, matching Population::bestIndex. Recomputed rather
// than carried, so the discrimination check below can ask what a best-only
// comparison would conclude about a MODIFIED snapshot.
std::size_t bestIndexOf(const Snapshot& snapshot) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < snapshot.fitness.size(); ++i) {
        if (snapshot.fitness[i] < snapshot.fitness[best]) {
            best = i;
        }
    }
    return best;
}

Snapshot evolveSnapshot(const vrp::Problem& problem, const vrp::GaParams& params,
                        vrp::StrategyKind kind, std::size_t threads) {
    auto executor = vrp::makeExecutor(threads);
    auto strategy = vrp::makeStrategy(kind);
    vrp::Population population(params.populationSize, problem, params.seed, *executor);
    for (std::size_t g = 0; g < params.generations; ++g) {
        strategy->step(g, population, problem, params, *executor);
    }

    Snapshot snapshot;
    snapshot.routeLength = population.routeLength();
    snapshot.fitness.reserve(population.size());
    snapshot.routes.reserve(population.size() * population.routeLength());
    for (std::size_t i = 0; i < population.size(); ++i) {
        snapshot.fitness.push_back(population.fitness(i));
        for (const int gene : population.route(i)) {
            snapshot.routes.push_back(gene);
        }
    }
    snapshot.bestIndex = bestIndexOf(snapshot);
    const std::span<const int> best = population.route(snapshot.bestIndex);
    snapshot.bestRoute.assign(best.begin(), best.end());
    snapshot.bestDistance = snapshot.fitness[snapshot.bestIndex];
    return snapshot;
}

}  // namespace

// The guard for everything else here: a Solver that ignored params.seed would
// satisfy every repeatability and cross-thread comparison in this file.
TEST_CASE("different seeds produce different results", "[determinism]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    const vrp::RunResult a = runWith(problem, vrp::StrategyKind::Generational, 1, 1);
    const vrp::RunResult b = runWith(problem, vrp::StrategyKind::Generational, 1, 2);
    REQUIRE(a.bestRoute != b.bestRoute);
}

TEST_CASE("the same seed reproduces the same result", "[determinism]") {
    // Two INDEPENDENT Solver instances; the same-instance case lives in
    // test_solver.cpp. The threaded pass is what a serial repeat cannot see.
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    for (const vrp::StrategyKind kind :
         {vrp::StrategyKind::SteadyState, vrp::StrategyKind::Generational}) {
        for (const std::size_t threads : {std::size_t{1}, std::size_t{8}}) {
            INFO("strategy = " << kindName(kind) << ", threads = " << threads);
            const vrp::RunResult first = runWith(problem, kind, threads, 777);
            const vrp::RunResult second = runWith(problem, kind, threads, 777);
            REQUIRE(first.bestRoute == second.bestRoute);
            REQUIRE(first.bestDistance == second.bestDistance);
        }
    }
}

TEST_CASE("a threaded solver run reproduces a hand-run serial loop", "[determinism]") {
    // The reference is a separate serial loop, so a defect inside Solver::run's
    // own loop cannot cancel out of the comparison. Steady-state ignores its
    // executor entirely, so it is no evidence about threading here -- only a
    // guard for the day it starts scheduling work.
    const vrp::Problem problem = vrp::Problem::defaultInstance();

    for (const vrp::StrategyKind kind :
         {vrp::StrategyKind::SteadyState, vrp::StrategyKind::Generational}) {
        INFO("strategy = " << kindName(kind));
        const vrp::GaParams params = determinismParams(31337, kind);
        const Trace expected = referenceTrace(problem, params, kind);

        // The fixture has to discriminate: with a flat `best`, a trace shifted
        // by a generation would compare equal and pin nothing.
        REQUIRE(expected.generationsRun == params.generations);
        REQUIRE(expected.best.size() == params.generations);
        REQUIRE(expected.best.front() != expected.best.back());

        for (const std::size_t threads :
             {std::size_t{1}, std::size_t{2}, std::size_t{4}, std::size_t{8}}) {
            INFO("threads = " << threads);
            const Trace actual = solverTrace(problem, params, kind, threads);
            REQUIRE(actual.generations == expected.generations);
            REQUIRE(actual.best == expected.best);
            REQUIRE(actual.bestRoute == expected.bestRoute);
            REQUIRE(actual.bestDistance == expected.bestDistance);
            REQUIRE(actual.generationsRun == expected.generationsRun);
        }
    }
}

TEST_CASE("entire populations are bit-identical across thread counts", "[determinism]") {
    // Compare every individual, not just the winner: a single divergent slot is
    // a divergent run.
    const vrp::Problem problem = vrp::Problem::defaultInstance();

    for (const vrp::StrategyKind kind :
         {vrp::StrategyKind::SteadyState, vrp::StrategyKind::Generational}) {
        INFO("strategy = " << kindName(kind));
        const vrp::GaParams params = determinismParams(8675309, kind);
        const Snapshot serial = evolveSnapshot(problem, params, kind, 1);
        REQUIRE(serial.fitness.size() == params.populationSize);
        REQUIRE(serial.routes.size() == params.populationSize * serial.routeLength);

        for (const std::size_t threads :
             {std::size_t{2}, std::size_t{4}, std::size_t{8}}) {
            INFO("threads = " << threads);
            const Snapshot threaded = evolveSnapshot(problem, params, kind, threads);
            REQUIRE(threaded.routes == serial.routes);
            REQUIRE(threaded.fitness == serial.fitness);
        }

        // Discrimination: perturb one NON-best individual and show that the
        // whole-population comparison rejects it while a best-only comparison
        // would not.
        Snapshot perturbed = serial;
        const std::size_t victim = static_cast<std::size_t>(
            std::max_element(serial.fitness.begin(), serial.fitness.end()) -
            serial.fitness.begin());
        REQUIRE(victim != serial.bestIndex);

        int* const genes = perturbed.routes.data() + victim * perturbed.routeLength;
        std::swap(genes[0], genes[1]);
        perturbed.fitness[victim] = problem.routeDistance(
            std::span<const int>(genes, perturbed.routeLength));

        // BOTH halves must see it: the cross-thread comparison above leans on
        // both vectors, so the fitness check is not redundant with the routes one.
        REQUIRE(perturbed.routes != serial.routes);
        REQUIRE(perturbed.fitness != serial.fitness);

        // The best-only comparison does not: the winner has not moved.
        const std::size_t perturbedBest = bestIndexOf(perturbed);
        REQUIRE(perturbedBest == serial.bestIndex);
        REQUIRE(perturbed.fitness[perturbedBest] == serial.bestDistance);
        REQUIRE(std::equal(serial.bestRoute.begin(), serial.bestRoute.end(),
                           perturbed.routes.begin() +
                               static_cast<std::ptrdiff_t>(perturbedBest *
                                                           perturbed.routeLength)));
    }
}

TEST_CASE("repeated threaded runs are stable", "[determinism]") {
    // A race flakes in the PASSING direction, so ten clean attempts prove
    // nothing; the value is cheap repetition of the schedule, on a toolchain
    // with no TSan. The reference is taken at ONE thread deliberately: at
    // eight, chunk-relative seeding makes every run identically wrong and
    // walks straight through this.
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    const vrp::GaParams params =
        determinismParams(555, vrp::StrategyKind::Generational);

    const Snapshot reference =
        evolveSnapshot(problem, params, vrp::StrategyKind::Generational, 1);
    for (int attempt = 0; attempt < 10; ++attempt) {
        const Snapshot again =
            evolveSnapshot(problem, params, vrp::StrategyKind::Generational, 8);
        INFO("attempt " << attempt);
        REQUIRE(again.routes == reference.routes);
        REQUIRE(again.fitness == reference.fitness);
    }
}
