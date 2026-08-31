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
    // A copy, never a view: the population dies with run(), and the
    // generational strategy swaps its buffer every generation.
    std::vector<int> bestRoute;
    double bestDistance = 0.0;
    // Counted by the loop, not restated from params.generations.
    std::size_t generationsRun = 0;
    // Covers all of run(): the initial population, every generation, and the
    // progress callback's own cost.
    double elapsedSeconds = 0.0;
};

// Called after the generation has been applied, so `bestDistance` is that
// generation's. `generation` is 0-based: a run of N reports 0 .. N-1.
using ProgressCallback = std::function<void(std::size_t generation, double bestDistance)>;

// Owns the generation loop and performs no I/O; reporting goes through the
// callback. `problem` is held by reference and must outlive the Solver.
class Solver {
public:
    // Preconditions: `strategy` and `executor` are non-null, and
    // params.populationSize >= 1 -- an empty population makes run() read out of
    // bounds in a build where the assert is compiled out.
    Solver(const Problem& problem, GaParams params,
           std::unique_ptr<EvolutionStrategy> strategy, std::unique_ptr<Executor> executor);

    // Runs params.generations generations from a freshly seeded population.
    // Repeatable on one instance, given EvolutionStrategy::step's contract.
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
