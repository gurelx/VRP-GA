#include "vrp/Population.hpp"

#include <algorithm>
#include <cassert>
#include <utility>

#include "vrp/Rng.hpp"

namespace vrp {

Population::Population(std::size_t size, std::size_t routeLength)
    : routeLength_(routeLength), routes_(size * routeLength), fitness_(size, 0.0) {}

Population::Population(std::size_t size, const Problem& problem, std::uint64_t seed,
                       Executor& executor)
    : routeLength_(problem.customerCount()),
      routes_(size * problem.customerCount()),
      fitness_(size, 0.0) {
    const std::size_t length = routeLength_;
    executor.parallelFor(size, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            int* r = routes_.data() + i * length;
            for (std::size_t j = 0; j < length; ++j) {
                r[j] = static_cast<int>(j + 1);
            }
            // Seeded per item, never per thread, so the shuffle is independent
            // of how the range was chunked.
            Rng rng(mixSeed(seed, kInitDomain, i));
            for (std::size_t j = length; j > 1; --j) {
                const std::uint32_t k = rng.below(static_cast<std::uint32_t>(j));
                std::swap(r[j - 1], r[k]);
            }
            fitness_[i] = problem.routeDistance(std::span<const int>(r, length));
        }
    });
}

void Population::setRoute(std::size_t i, std::span<const int> value,
                          const Problem& problem) {
    assert(i < fitness_.size() && "setRoute index out of range");
    // The copy below is sized by the span, not by routeLength_, so a mismatch
    // is a buffer overrun into the next individual or a stale-gene tail rather
    // than a truncation. Fault here instead.
    assert(value.size() == routeLength_ &&
           "setRoute requires a route of exactly routeLength()");
    int* destination = routes_.data() + i * routeLength_;
    std::copy(value.begin(), value.end(), destination);
    fitness_[i] = problem.routeDistance(std::span<const int>(destination, routeLength_));
}

void Population::evaluateAll(const Problem& problem, Executor& executor) {
    const std::size_t length = routeLength_;
    executor.parallelFor(fitness_.size(), [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            fitness_[i] = problem.routeDistance(
                std::span<const int>(routes_.data() + i * length, length));
        }
    });
}

std::size_t Population::bestIndex() const noexcept {
    assert(!fitness_.empty() && "bestIndex requires a non-empty population");
    std::size_t best = 0;
    for (std::size_t i = 1; i < fitness_.size(); ++i) {
        if (fitness_[i] < fitness_[best]) {  // strict: ties keep the lower index
            best = i;
        }
    }
    return best;
}

std::size_t Population::worstIndex() const noexcept {
    assert(!fitness_.empty() && "worstIndex requires a non-empty population");
    std::size_t worst = 0;
    for (std::size_t i = 1; i < fitness_.size(); ++i) {
        if (fitness_[i] > fitness_[worst]) {  // strict: ties keep the lower index
            worst = i;
        }
    }
    return worst;
}

void Population::swap(Population& other) noexcept {
    std::swap(routeLength_, other.routeLength_);
    routes_.swap(other.routes_);
    fitness_.swap(other.fitness_);
}

}  // namespace vrp
