#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "vrp/Config.hpp"
#include "vrp/Executor.hpp"
#include "vrp/Problem.hpp"
#include "vrp/Strategy.hpp"

namespace vrp {

struct RunResult {
    // A copy of the winning route, never a view into the population: the
    // population dies with run(), and the generational strategy swaps its
    // buffer out from under any span every generation.
    std::vector<int> bestRoute;
    double bestDistance = 0.0;
    // Generations actually executed, counted by the loop rather than restated
    // from params.generations, so the two cannot drift apart silently.
    std::size_t generationsRun = 0;
    // Covers the whole of run(): initial population construction, every
    // generation, and the progress callback's own cost, since the callback is
    // invoked inside the timed region. A caller whose callback does real work
    // -- printing to a synchronous console, say -- is measuring that too.
    double elapsedSeconds = 0.0;
};

// Called after each generation has been applied, so `bestDistance` is the best
// of the population the generation just produced, never the one before it.
// `generation` is the 0-based index of that generation: a run of N generations
// reports 0 .. N-1.
using ProgressCallback = std::function<void(std::size_t generation, double bestDistance)>;

// Owns the generation loop. Performs no I/O -- reporting goes through the
// callback -- so it stays testable and the printing lives in the CLI's
// Reporter. `problem` must outlive the Solver: it is held by reference
// deliberately, so that one distance matrix is shared rather than copied.
class Solver {
public:
    // Preconditions: `strategy` and `executor` are non-null, and
    // params.populationSize >= 1. All three are asserted in debug builds; the
    // CLI rejects --population 0 before it ever reaches here.
    //
    // populationSize == 0 is not merely unspecified. run() would call
    // Population::route(0) on an empty population, which returns a span of
    // routeLength() ints over a zero-size allocation, and then copy it into
    // RunResult::bestRoute -- a concrete out-of-bounds read of
    // customerCount() * sizeof(int) bytes past a heap block, in a build where
    // the assert is compiled out. Hence the fault rather than a tolerated
    // empty result.
    Solver(const Problem& problem, GaParams params,
           std::unique_ptr<EvolutionStrategy> strategy, std::unique_ptr<Executor> executor);

    // Runs params.generations generations from a freshly seeded population.
    // Repeatable on one instance: the population is rebuilt from params.seed
    // on every call, and EvolutionStrategy::step is required to depend only on
    // its arguments (see Strategy.hpp), so a second run() reproduces the first.
    // Solver cannot enforce the second half of that; it relies on it.
    RunResult run(const ProgressCallback& progress = {});

    const EvolutionStrategy& strategy() const noexcept { return *strategy_; }
    const Executor& executor() const noexcept { return *executor_; }

private:
    const Problem& problem_;
    GaParams params_;
    std::unique_ptr<EvolutionStrategy> strategy_;
    std::unique_ptr<Executor> executor_;
};

}  // namespace vrp
