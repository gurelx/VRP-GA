#include "vrp/Solver.hpp"

#include <cassert>
#include <chrono>
#include <span>
#include <utility>

#include "vrp/Population.hpp"

namespace vrp {

Solver::Solver(const Problem& problem, GaParams params,
               std::unique_ptr<EvolutionStrategy> strategy,
               std::unique_ptr<Executor> executor)
    : problem_(problem),
      params_(params),
      strategy_(std::move(strategy)),
      executor_(std::move(executor)) {
    assert(strategy_ != nullptr && "Solver requires a strategy");
    assert(executor_ != nullptr && "Solver requires an executor");
    assert(params_.populationSize > 0 && "Solver requires a non-empty population");
}

RunResult Solver::run(const ProgressCallback& progress) {
    const auto started = std::chrono::steady_clock::now();

    Population population(params_.populationSize, problem_, params_.seed, *executor_);

    RunResult result;
    for (std::size_t generation = 0; generation < params_.generations; ++generation) {
        strategy_->step(generation, population, problem_, params_, *executor_);
        ++result.generationsRun;
        if (progress) {
            progress(generation, population.fitness(population.bestIndex()));
        }
    }

    const std::size_t best = population.bestIndex();
    const std::span<const int> bestRoute = population.route(best);
    result.bestRoute.assign(bestRoute.begin(), bestRoute.end());
    result.bestDistance = population.fitness(best);
    result.elapsedSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return result;
}

}  // namespace vrp
