#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "vrp/Executor.hpp"
#include "vrp/Problem.hpp"

namespace vrp {

// One contiguous route buffer plus a parallel fitness array, so fitness cannot
// go stale against its route. route() hands out views into that buffer and
// swap() exchanges buffers, so a span taken before a swap goes on naming the
// OTHER population's data -- re-obtain spans after one.
class Population {
public:
    Population() = default;

    // Allocates without randomising.
    Population(std::size_t size, std::size_t routeLength);

    // Randomised initial population. Each individual is seeded from
    // mixSeed(seed, kInitDomain, i), so chunking cannot change the result.
    Population(std::size_t size, const Problem& problem, std::uint64_t seed,
               Executor& executor);

    std::size_t size() const noexcept { return fitness_.size(); }
    std::size_t routeLength() const noexcept { return routeLength_; }

    // Precondition: i < size().
    std::span<const int> route(std::size_t i) const noexcept {
        assert(i < size() && "route index out of range");
        return {routes_.data() + i * routeLength_, routeLength_};
    }

    // Precondition: i < size(). Writing through this span leaves fitness(i)
    // stale until evaluateAll() runs; setRoute() cannot desync.
    std::span<int> route(std::size_t i) noexcept {
        assert(i < size() && "route index out of range");
        return {routes_.data() + i * routeLength_, routeLength_};
    }

    // Tour distance depot -> route(i) -> depot: a cost to minimise, not a score
    // to maximise. Precondition: i < size().
    double fitness(std::size_t i) const noexcept {
        assert(i < size() && "fitness index out of range");
        return fitness_[i];
    }

    // Writes the route and its fitness together. Preconditions: i < size() and
    // value.size() == routeLength() -- the copy is sized by the span, so any
    // other length overruns the next individual or leaves a stale tail. Given
    // those, the call is safe to make concurrently for distinct i.
    void setRoute(std::size_t i, std::span<const int> value, const Problem& problem);

    void evaluateAll(const Problem& problem, Executor& executor);

    // Ties resolve to the lowest index, which determinism depends on.
    // Precondition: size() > 0.
    std::size_t bestIndex() const noexcept;
    std::size_t worstIndex() const noexcept;

    void swap(Population& other) noexcept;

    // Hidden friend, so the two-step `using std::swap; swap(a, b);` selects
    // this buffer swap rather than the generic three-move fallback.
    friend void swap(Population& a, Population& b) noexcept { a.swap(b); }

private:
    std::size_t routeLength_ = 0;
    std::vector<int> routes_;
    std::vector<double> fitness_;
};

}  // namespace vrp
