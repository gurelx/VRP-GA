#include "vrp/Strategy.hpp"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

#include "vrp/Operators.hpp"
#include "vrp/Rng.hpp"

namespace vrp {
namespace {

// Preserves the semantics of the original sequential program: two tournament
// parents, one child, replacing the worst individual. Inherently serial; the
// executor is unused here, so --threads does not accelerate this strategy.
class SteadyStateStrategy final : public EvolutionStrategy {
public:
    const char* name() const noexcept override { return "steady-state"; }

    void step(std::size_t generation, Population& population, const Problem& problem,
              const GaParams& params, Executor& /*executor*/) override {
        // Const view, so route() yields span<const int> and no path here can
        // write a gene without going through setRoute and its fitness update.
        const Population& current = population;
        // Slot 0: one child per generation, so there is only one stream to
        // name. Still mixSeed(seed, generation, slot) -- the same rule the
        // generational strategy follows, not a special case.
        Rng rng(mixSeed(params.seed, generation, std::size_t{0}));

        const std::size_t first = ops::tournamentSelect(current, params.tournamentSize, rng);
        const std::size_t second = ops::tournamentSelect(current, params.tournamentSize, rng);

        child_.resize(population.routeLength());
        // customerCount + 1 because genes are 1-based; sizing it customerCount
        // puts the highest gene one past the end.
        seen_.assign(problem.customerCount() + 1, 0);
        ops::orderCrossover(current.route(first), current.route(second), child_, seen_, rng);
        ops::swapMutate(child_, params.mutationRate, rng);

        // The worst individual, not the best: this is the replacement that
        // makes the incumbent best unlosable.
        population.setRoute(population.worstIndex(), child_, problem);
    }

private:
    std::vector<int> child_;
    std::vector<char> seen_;
};

// Builds a full offspring population each generation, carrying eliteCount best
// individuals forward unchanged.
class GenerationalStrategy final : public EvolutionStrategy {
public:
    const char* name() const noexcept override { return "generational"; }

    void step(std::size_t generation, Population& population, const Problem& problem,
              const GaParams& params, Executor& executor) override {
        const Population& current = population;
        const std::size_t count = population.size();
        const std::size_t length = population.routeLength();

        if (next_.size() != count || next_.routeLength() != length) {
            next_ = Population(count, length);
        }

        const std::size_t elites = std::min(params.eliteCount, count);

        // Total order (fitness, index) makes the elite set deterministic even
        // though partial_sort is not stable. Fitness alone would leave the
        // choice among tied individuals to the algorithm's internal moves.
        order_.resize(count);
        std::iota(order_.begin(), order_.end(), std::size_t{0});
        std::partial_sort(
            order_.begin(), order_.begin() + static_cast<std::ptrdiff_t>(elites),
            order_.end(), [&current](std::size_t a, std::size_t b) {
                const double fa = current.fitness(a);
                const double fb = current.fitness(b);
                return fa != fb ? fa < fb : a < b;
            });
        for (std::size_t i = 0; i < elites; ++i) {
            next_.setRoute(i, current.route(order_[i]), problem);
        }

        // Offspring occupy [elites, count); the loop counts from 0 so the
        // executor sees a range starting at 0, and shifts by `elites` to get
        // the slot. Writing at k rather than k + elites would overwrite the
        // elites just copied and leave the tail of next_ stale.
        const std::size_t offspring = count - elites;
        executor.parallelFor(offspring, [&](std::size_t begin, std::size_t end) {
            // Scratch is per chunk, not per child -- this is why the body takes
            // a range instead of an index. orderCrossover clears `seen` itself,
            // so reuse across children carries nothing forward. (These two
            // allocations are the only throwing operations in the body; a
            // bad_alloc on a worker would terminate, per Executor's contract.)
            std::vector<char> seen(problem.customerCount() + 1, 0);
            std::vector<int> child(length);
            for (std::size_t k = begin; k < end; ++k) {
                const std::size_t slot = k + elites;
                // Seeded by the absolute slot, never by a chunk-relative index
                // or a thread identity, so chunking cannot change the outcome:
                // this is the whole reason a threaded run reproduces a serial
                // one bit for bit.
                Rng rng(mixSeed(params.seed, generation, slot));
                const std::size_t a =
                    ops::tournamentSelect(current, params.tournamentSize, rng);
                const std::size_t b =
                    ops::tournamentSelect(current, params.tournamentSize, rng);
                ops::orderCrossover(current.route(a), current.route(b), child, seen, rng);
                ops::swapMutate(child, params.mutationRate, rng);
                // Distinct slots across the whole range, so the concurrent
                // writes are disjoint.
                next_.setRoute(slot, child, problem);
            }
        });

        // Without this the generation is computed and thrown away. Spans taken
        // before the swap now name the other buffer, so nothing may hold one
        // across it.
        population.swap(next_);
    }

private:
    Population next_;
    std::vector<std::size_t> order_;
};

}  // namespace

std::unique_ptr<EvolutionStrategy> makeStrategy(StrategyKind kind) {
    switch (kind) {
        case StrategyKind::Generational:
            return std::make_unique<GenerationalStrategy>();
        case StrategyKind::SteadyState:
            break;
    }
    return std::make_unique<SteadyStateStrategy>();
}

}  // namespace vrp
