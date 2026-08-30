#pragma once

// Shared test support for tests/test_solver.cpp and tests/test_determinism.cpp.
//
// These three helpers were byte-identical copies in both files. The copies were
// kept deliberately at first -- so that one edit could not silently retune both
// files' oracles at once -- but the price was that they could drift apart
// unnoticed, and a drifted oracle is worse than a shared one: the two files
// would then be comparing against subtly different definitions of "the same
// run" while still reading as though they agreed. They are shared here instead,
// and the hazard the old comments described no longer exists.
//
// WHAT THIS FILE IS, AND IS NOT. referenceTrace() is a DIFFERENTIAL oracle. It
// calls the same Population constructor and the same EvolutionStrategy::step as
// the code under test, so it can only detect defects that live in Solver's own
// generation loop -- an off-by-one, a report issued before the step instead of
// after, the wrong generation index handed to the strategy. It is NOT an
// independent computation of the specified answer, and no test may cite it as
// evidence that the specified seed derivation is the one implemented. That
// claim belongs to the per-slot oracles in tests/test_population.cpp and
// tests/test_strategies.cpp, which replay mixSeed and the operator sequence by
// hand. A wrong-but-consistent implementation passes everything built on this
// header.
//
// Because both suites now share these definitions, a change here retunes both
// at once. That is the intended trade -- but check both call sites before
// editing: test_solver.cpp uses it at one thread as Solver's loop oracle, and
// test_determinism.cpp uses it as the serial reference that threaded runs must
// reproduce.

#include <cstddef>
#include <span>
#include <vector>

#include "vrp/Config.hpp"
#include "vrp/Executor.hpp"
#include "vrp/Population.hpp"
#include "vrp/Problem.hpp"
#include "vrp/Strategy.hpp"

namespace vrp_test {

// Everything one run can be observed to produce, so a comparison covers the
// whole contract rather than the one field a mutant happened to leave alone.
// The per-generation `best` sequence is the only window Solver offers onto the
// intermediate populations: a run that diverged at generation 3 and reconverged
// by the last generation agrees on bestRoute and dies on `best`.
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

// A hand-run generation loop, kept separate from Solver's own: build the
// population from the seed, apply the strategy exactly `generations` times, and
// record the incumbent best AFTER each step. A loop that steps once too often
// or too few times, one that reports before stepping instead of after, or one
// that hands the strategy the wrong generation index all diverge from this.
//
// Always runs at one thread. Callers that want a thread-count comparison drive
// Solver at N threads and compare against this.
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
