#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "vrp/Executor.hpp"
#include "vrp/Problem.hpp"

namespace vrp {

// One contiguous buffer of routes plus a parallel fitness array. Storing them
// together is what keeps fitness from going stale against its route.
//
// Span lifetime: route() hands out views into this object's buffer, and swap()
// exchanges buffers rather than moving elements. A span obtained before a swap
// therefore keeps naming its original data, which now belongs to the OTHER
// population. Task 7 double-buffers via swap, so re-obtain spans after one
// rather than holding them across it.
class Population {
public:
    Population() = default;

    // Allocates without randomising. Used for offspring double-buffering.
    Population(std::size_t size, std::size_t routeLength);

    // Randomised initial population. Each individual is seeded from
    // mixSeed(seed, kInitDomain, i), so the result does not depend on how the
    // executor distributed the work.
    Population(std::size_t size, const Problem& problem, std::uint64_t seed,
               Executor& executor);

    std::size_t size() const noexcept { return fitness_.size(); }
    std::size_t routeLength() const noexcept { return routeLength_; }

    // Precondition: i < size(). An out-of-range index would silently alias a
    // neighbouring individual inside the flat buffer and return a plausible
    // wrong route, so fault instead -- the same reasoning Problem::distance
    // gives for its own bounds asserts.
    std::span<const int> route(std::size_t i) const noexcept {
        assert(i < size() && "route index out of range");
        return {routes_.data() + i * routeLength_, routeLength_};
    }

    // Precondition: i < size(). Writing through this span leaves fitness(i)
    // stale until evaluateAll() runs; setRoute() is the variant that cannot
    // desync, and is what callers should reach for by default.
    std::span<int> route(std::size_t i) noexcept {
        assert(i < size() && "route index out of range");
        return {routes_.data() + i * routeLength_, routeLength_};
    }

    // Tour distance depot -> route(i) -> depot. Lower is better: this is a cost
    // to minimise, not a score to maximise. Precondition: i < size().
    double fitness(std::size_t i) const noexcept {
        assert(i < size() && "fitness index out of range");
        return fitness_[i];
    }

    // Writes the route and its fitness together.
    //
    // Preconditions: i < size() and value.size() == routeLength(). The length
    // is a requirement, not a hint -- the copy is sized by the span, so an
    // oversized value overruns into the next individual (or off the end of the
    // buffer on the last one), and an undersized one leaves stale genes in the
    // tail while fitness is computed over the full routeLength(), which is the
    // very desync this class exists to prevent. Both are asserted.
    //
    // Given those preconditions the call is safe to make concurrently for
    // distinct i: the route and fitness writes are then disjoint.
    void setRoute(std::size_t i, std::span<const int> value, const Problem& problem);

    void evaluateAll(const Problem& problem, Executor& executor);

    // Ties resolve to the lowest index. Precondition: size() > 0 -- there is no
    // index to name otherwise, and returning 0 on an empty population would
    // turn the natural fitness(bestIndex()) into a silent out-of-range read.
    std::size_t bestIndex() const noexcept;
    std::size_t worstIndex() const noexcept;

    void swap(Population& other) noexcept;

    // Hidden friend, so the two-step `using std::swap; swap(a, b);` -- what the
    // standard algorithms perform internally -- selects this buffer swap rather
    // than the generic three-move fallback.
    friend void swap(Population& a, Population& b) noexcept { a.swap(b); }

private:
    std::size_t routeLength_ = 0;
    std::vector<int> routes_;      // size * routeLength_
    std::vector<double> fitness_;  // size
};

}  // namespace vrp
