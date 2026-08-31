#pragma once

#include <cstddef>
#include <cstdint>

namespace vrp {

enum class StrategyKind {
    SteadyState,
    Generational,
};

struct GaParams {
    std::size_t populationSize = 100000;
    std::size_t generations = 100;
    double mutationRate = 0.2;
    std::size_t tournamentSize = 5;
    std::size_t eliteCount = 1;  // generational only
    std::uint64_t seed = 42;
};

}  // namespace vrp
