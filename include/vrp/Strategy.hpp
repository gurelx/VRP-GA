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
    // arguments. Scratch kept between calls must be fully written before it is
    // read, or Solver::run() stops being repeatable on one instance.
    virtual void step(std::size_t generation, Population& population,
                      const Problem& problem, const GaParams& params,
                      Executor& executor) = 0;
};

std::unique_ptr<EvolutionStrategy> makeStrategy(StrategyKind kind);

}  // namespace vrp
