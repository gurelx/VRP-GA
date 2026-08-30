#pragma once

#include <cstddef>
#include <memory>

#include "vrp/Config.hpp"
#include "vrp/Executor.hpp"
#include "vrp/Population.hpp"
#include "vrp/Problem.hpp"

namespace vrp {

// Advances a population by one generation. Implementations contain no threads;
// they schedule through the Executor they are handed.
class EvolutionStrategy {
public:
    virtual ~EvolutionStrategy() = default;
    virtual const char* name() const noexcept = 0;

    // Binding on every implementation: step() must depend only on its
    // arguments. Scratch kept between calls -- the two strategies here keep a
    // buffer population, an ordering, a child and a seen-set -- must be fully
    // written before it is read, so that a run of generations 0..N-1 over a
    // freshly seeded population yields the same result no matter how many
    // earlier runs the same object has performed.
    //
    // This is what makes Solver::run() repeatable on one instance, and it is
    // stated here rather than there because Solver cannot enforce it: a
    // strategy that carried a live RNG, a cumulative generation counter or an
    // adaptive mutation rate across calls would break that guarantee without
    // Solver being able to tell.
    virtual void step(std::size_t generation, Population& population,
                      const Problem& problem, const GaParams& params,
                      Executor& executor) = 0;
};

std::unique_ptr<EvolutionStrategy> makeStrategy(StrategyKind kind);

}  // namespace vrp
