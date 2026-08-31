#pragma once

// referenceTrace() is a DIFFERENTIAL oracle: it calls the same Population
// constructor and the same EvolutionStrategy::step as the code under test, so
// it can only catch defects in Solver's own generation loop. A
// wrong-but-consistent implementation passes it; conformance to the specified
// seed derivation is pinned by tests/test_population.cpp and
// tests/test_strategies.cpp. Shared by test_solver.cpp and
// test_determinism.cpp, so an edit here retunes both suites.

#include <cstddef>
#include <span>
#include <vector>

#include "vrp/Config.hpp"
#include "vrp/Executor.hpp"
#include "vrp/Population.hpp"
#include "vrp/Problem.hpp"
#include "vrp/Strategy.hpp"

namespace vrp_test {

// The per-generation `best` sequence is the only window onto the intermediate
// populations: a divergence that reconverges is invisible in bestRoute alone.
struct Trace {
    std::vector<std::size_t> generations;  // as reported to the callback
    std::vector<double> best;              // best distance as reported, per generation
    std::vector<int> bestRoute;
    double bestDistance = 0.0;
    std::size_t generationsRun = 0;
};

inline const char* kindName(vrp::StrategyKind kind) {
    return kind == vrp::StrategyKind::SteadyState ? "steady-state" : "generational";
}

// A hand-run generation loop, kept separate from Solver's own, recording the
// incumbent best AFTER each step. Always runs at one thread.
inline Trace referenceTrace(const vrp::Problem& problem, const vrp::GaParams& params,
                            vrp::StrategyKind kind) {
    auto executor = vrp::makeExecutor(1);
    auto strategy = vrp::makeStrategy(kind);
    vrp::Population population(params.populationSize, problem, params.seed, *executor);

    Trace trace;
    for (std::size_t generation = 0; generation < params.generations; ++generation) {
        strategy->step(generation, population, problem, params, *executor);
        trace.generations.push_back(generation);
        trace.best.push_back(population.fitness(population.bestIndex()));
    }
    trace.generationsRun = params.generations;

    const std::size_t best = population.bestIndex();
    const std::span<const int> route = population.route(best);
    trace.bestRoute.assign(route.begin(), route.end());
    trace.bestDistance = population.fitness(best);
    return trace;
}

}  // namespace vrp_test
