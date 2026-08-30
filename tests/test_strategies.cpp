#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include "vrp/Config.hpp"
#include "vrp/Executor.hpp"
#include "vrp/Operators.hpp"
#include "vrp/Population.hpp"
#include "vrp/Problem.hpp"
#include "vrp/Rng.hpp"
#include "vrp/Strategy.hpp"

namespace {

bool isPermutation(std::span<const int> route, std::size_t n) {
    std::vector<int> sorted(route.begin(), route.end());
    std::sort(sorted.begin(), sorted.end());
    for (std::size_t i = 0; i < n; ++i) {
        if (sorted[i] != static_cast<int>(i + 1)) {
            return false;
        }
    }
    return true;
}

vrp::GaParams smallParams() {
    vrp::GaParams params;
    params.populationSize = 200;
    params.generations = 30;
    params.seed = 4242;
    return params;
}

struct Offspring {
    std::vector<int> route;
    std::size_t firstParent = 0;
    std::size_t secondParent = 0;
};

// A literal restatement of the specified derivation for one slot: seed from
// mixSeed(seed, generation, slot), two tournaments, order crossover, swap
// mutation, in that order. It shares no code with Strategy.cpp, takes `slot`
// from the caller rather than from anything a strategy exposes, and allocates
// its scratch fresh so no carryover between slots can mask a defect.
Offspring expectedOffspring(const vrp::Population& previous, const vrp::Problem& problem,
                            const vrp::GaParams& params, std::size_t generation,
                            std::size_t slot) {
    std::vector<char> seen(problem.customerCount() + 1, 0);
    Offspring result;
    result.route.resize(previous.routeLength());

    vrp::Rng rng(vrp::mixSeed(params.seed, generation, slot));
    result.firstParent = vrp::ops::tournamentSelect(previous, params.tournamentSize, rng);
    result.secondParent = vrp::ops::tournamentSelect(previous, params.tournamentSize, rng);
    vrp::ops::orderCrossover(previous.route(result.firstParent),
                             previous.route(result.secondParent), result.route, seen, rng);
    vrp::ops::swapMutate(result.route, params.mutationRate, rng);
    return result;
}

// Forwards to a real executor and counts the calls. parallelFor amortises a
// condition-variable round trip per call, so "how many times per generation"
// is part of the strategies' contract, not an implementation detail: a version
// that scheduled once per child would be functionally correct and unusable.
class CountingExecutor final : public vrp::Executor {
public:
    explicit CountingExecutor(std::size_t threads) : inner_(vrp::makeExecutor(threads)) {}

    std::size_t threadCount() const noexcept override { return inner_->threadCount(); }

    void parallelFor(std::size_t n, const vrp::ParallelBody& body) override {
        ++calls;
        inner_->parallelFor(n, body);
    }

    std::size_t calls = 0;

private:
    std::unique_ptr<vrp::Executor> inner_;
};

}  // namespace

TEST_CASE("GaParams carries the documented defaults", "[strategy]") {
    // Nothing in this task reads these -- both strategies take the size from
    // the Population and the generation index from their caller -- so a typo
    // would survive Task 7 entirely and surface as a silent behaviour change
    // once Task 8 wires the CLI to them.
    constexpr vrp::GaParams params{};
    STATIC_REQUIRE(params.populationSize == 100000);
    STATIC_REQUIRE(params.generations == 100);
    STATIC_REQUIRE(params.mutationRate == 0.2);
    STATIC_REQUIRE(params.tournamentSize == 5);
    STATIC_REQUIRE(params.eliteCount == 1);
    STATIC_REQUIRE(params.seed == 42);
}

// VACUOUS BY CONSTRUCTION: pins the makeStrategy switch dispatch and the two
// name strings, and nothing else. Kept because the names are part of the
// public interface Task 8 prints.
TEST_CASE("both strategies are named", "[strategy]") {
    REQUIRE(std::string(vrp::makeStrategy(vrp::StrategyKind::SteadyState)->name()) ==
            "steady-state");
    REQUIRE(std::string(vrp::makeStrategy(vrp::StrategyKind::Generational)->name()) ==
            "generational");
}

TEST_CASE("populations stay valid across generations", "[strategy]") {
    for (auto kind : {vrp::StrategyKind::SteadyState, vrp::StrategyKind::Generational}) {
        const vrp::Problem problem = vrp::Problem::defaultInstance();
        auto exec = vrp::makeExecutor(1);
        const vrp::GaParams params = smallParams();
        vrp::Population pop(params.populationSize, problem, params.seed, *exec);
        auto strategy = vrp::makeStrategy(kind);

        for (std::size_t g = 0; g < params.generations; ++g) {
            strategy->step(g, pop, problem, params, *exec);
        }
        for (std::size_t i = 0; i < pop.size(); ++i) {
            INFO(strategy->name() << " individual " << i);
            REQUIRE(isPermutation(pop.route(i), problem.customerCount()));
        }
    }
}

// WEAK: re-asserts Population::setRoute's own invariant, which Task 5 already
// owns and tests. It can only fail if a strategy writes genes through the
// non-const route() accessor instead of setRoute -- worth keeping as a guard
// against exactly that, but it is not evidence about the strategies.
TEST_CASE("fitness stays consistent with routes after every step", "[strategy]") {
    for (auto kind : {vrp::StrategyKind::SteadyState, vrp::StrategyKind::Generational}) {
        const vrp::Problem problem = vrp::Problem::defaultInstance();
        auto exec = vrp::makeExecutor(1);
        const vrp::GaParams params = smallParams();
        vrp::Population pop(params.populationSize, problem, params.seed, *exec);
        auto strategy = vrp::makeStrategy(kind);

        for (std::size_t g = 0; g < 10; ++g) {
            strategy->step(g, pop, problem, params, *exec);
            for (std::size_t i = 0; i < pop.size(); ++i) {
                INFO(strategy->name() << " gen " << g << " individual " << i);
                REQUIRE(pop.fitness(i) == problem.routeDistance(pop.route(i)));
            }
        }
    }
}

TEST_CASE("best fitness never regresses", "[strategy]") {
    // Steady-state only ever replaces the worst individual; generational keeps
    // eliteCount >= 1. Either way the incumbent best cannot be lost.
    for (auto kind : {vrp::StrategyKind::SteadyState, vrp::StrategyKind::Generational}) {
        const vrp::Problem problem = vrp::Problem::defaultInstance();
        auto exec = vrp::makeExecutor(1);
        const vrp::GaParams params = smallParams();
        vrp::Population pop(params.populationSize, problem, params.seed, *exec);
        auto strategy = vrp::makeStrategy(kind);

        double previousBest = pop.fitness(pop.bestIndex());
        for (std::size_t g = 0; g < params.generations; ++g) {
            strategy->step(g, pop, problem, params, *exec);
            const double best = pop.fitness(pop.bestIndex());
            INFO(strategy->name() << " generation " << g);
            REQUIRE(best <= previousBest);
            previousBest = best;
        }
    }
}

TEST_CASE("evolution improves on the initial population", "[strategy]") {
    for (auto kind : {vrp::StrategyKind::SteadyState, vrp::StrategyKind::Generational}) {
        const vrp::Problem problem = vrp::Problem::defaultInstance();
        auto exec = vrp::makeExecutor(1);
        vrp::GaParams params = smallParams();
        params.generations = 200;
        vrp::Population pop(params.populationSize, problem, params.seed, *exec);
        auto strategy = vrp::makeStrategy(kind);

        const double initialBest = pop.fitness(pop.bestIndex());
        for (std::size_t g = 0; g < params.generations; ++g) {
            strategy->step(g, pop, problem, params, *exec);
        }
        INFO(strategy->name());
        REQUIRE(pop.fitness(pop.bestIndex()) < initialBest);
    }
}

// ONE-SIDED: `changed <= 1` is satisfied by an implementation that changes
// nothing at all. "steady state offspring follows the specified seeding and
// replacement" below is the two-sided form -- it names the slot that must
// change, says what must land in it, and requires every other slot to be
// untouched.
TEST_CASE("steady state replaces exactly one individual per step", "[strategy]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    const vrp::GaParams params = smallParams();
    vrp::Population pop(params.populationSize, problem, params.seed, *exec);
    auto strategy = vrp::makeStrategy(vrp::StrategyKind::SteadyState);

    std::vector<double> before(pop.size());
    for (std::size_t i = 0; i < pop.size(); ++i) {
        before[i] = pop.fitness(i);
    }
    strategy->step(0, pop, problem, params, *exec);

    std::size_t changed = 0;
    for (std::size_t i = 0; i < pop.size(); ++i) {
        if (pop.fitness(i) != before[i]) {
            changed++;
        }
    }
    REQUIRE(changed <= 1);  // the replacement may coincidentally tie
}

TEST_CASE("generational elitism carries the best individual forward",
          "[strategy]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    vrp::GaParams params = smallParams();
    params.eliteCount = 3;
    vrp::Population pop(params.populationSize, problem, params.seed, *exec);
    auto strategy = vrp::makeStrategy(vrp::StrategyKind::Generational);

    const std::vector<int> bestBefore(pop.route(pop.bestIndex()).begin(),
                                      pop.route(pop.bestIndex()).end());
    const double bestFitnessBefore = pop.fitness(pop.bestIndex());

    strategy->step(0, pop, problem, params, *exec);

    // The incumbent best must still be present, at slot 0 by construction.
    REQUIRE(pop.fitness(0) == bestFitnessBefore);
    REQUIRE(std::equal(pop.route(0).begin(), pop.route(0).end(), bestBefore.begin()));
}

// ---------------------------------------------------------------------------
// Everything above is satisfied by any per-slot seeding: valid permutations, a
// preserved elite and a population that improves are all insensitive to which
// stream each offspring draws from. The cases below pin the seeding itself,
// which is what makes a threaded run reproduce a serial one.
// ---------------------------------------------------------------------------

TEST_CASE("generational results are identical at every thread count", "[strategy]") {
    // A chunk-relative or thread-derived seed passes every test above and fails
    // here. The whole final population must agree bit for bit, not just the
    // best individual: a single divergent slot is a divergent run.
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    vrp::GaParams params = smallParams();
    params.populationSize = 97;  // prime, so no thread count divides it evenly
    params.generations = 15;
    params.eliteCount = 2;

    auto run = [&](std::size_t threads) {
        auto exec = vrp::makeExecutor(threads);
        vrp::Population pop(params.populationSize, problem, params.seed, *exec);
        auto strategy = vrp::makeStrategy(vrp::StrategyKind::Generational);
        for (std::size_t g = 0; g < params.generations; ++g) {
            strategy->step(g, pop, problem, params, *exec);
        }
        return pop;
    };

    const vrp::Population reference = run(1);
    for (std::size_t threads : {std::size_t{2}, std::size_t{4}, std::size_t{8}}) {
        const vrp::Population other = run(threads);
        REQUIRE(other.size() == reference.size());
        REQUIRE(other.routeLength() == reference.routeLength());
        for (std::size_t i = 0; i < reference.size(); ++i) {
            INFO("threads " << threads << " individual " << i);
            REQUIRE(std::equal(reference.route(i).begin(), reference.route(i).end(),
                               other.route(i).begin()));
            REQUIRE(other.fitness(i) == reference.fitness(i));
        }
    }
}

// VACUOUS FOR THIS TASK: SteadyStateStrategy::step ignores its Executor
// entirely, so the only thing that varies with the thread count here is the
// Population constructor, whose determinism Task 5 owns and tests. Kept as a
// regression guard for the day the strategy starts scheduling work -- but it
// is not evidence that this task's code is chunk-independent, and it must not
// be counted as such.
TEST_CASE("steady state results are identical at every thread count", "[strategy]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    const vrp::GaParams params = smallParams();

    auto run = [&](std::size_t threads) {
        auto exec = vrp::makeExecutor(threads);
        vrp::Population pop(params.populationSize, problem, params.seed, *exec);
        auto strategy = vrp::makeStrategy(vrp::StrategyKind::SteadyState);
        for (std::size_t g = 0; g < params.generations; ++g) {
            strategy->step(g, pop, problem, params, *exec);
        }
        return pop;
    };

    const vrp::Population reference = run(1);
    const vrp::Population other = run(4);
    REQUIRE(other.size() == reference.size());
    for (std::size_t i = 0; i < reference.size(); ++i) {
        INFO("individual " << i);
        REQUIRE(std::equal(reference.route(i).begin(), reference.route(i).end(),
                           other.route(i).begin()));
        REQUIRE(other.fitness(i) == reference.fitness(i));
    }
}

TEST_CASE("generational offspring follow the specified per-slot seeding",
          "[strategy]") {
    // Cross-thread agreement only proves the runs agree with each other. It
    // cannot tell mixSeed(seed, generation, slot) from the transposed
    // mixSeed(seed, slot, generation), which is just as chunk-independent and
    // just as wrong. Replay the specified stream against a snapshot of the
    // parent population instead.
    //
    // Run at 4 threads as well as 1, so "the threaded result matches the
    // specification" is one assertion rather than a chain through the
    // cross-thread-count case.
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    vrp::GaParams params = smallParams();
    params.populationSize = 64;
    params.eliteCount = 2;

    // A generation index that is neither 0 nor a valid slot for most slots, so
    // the transposed argument order cannot silently coincide.
    const std::size_t generation = 7;

    for (std::size_t threads : {std::size_t{1}, std::size_t{4}}) {
        auto exec = vrp::makeExecutor(threads);
        vrp::Population pop(params.populationSize, problem, params.seed, *exec);
        auto strategy = vrp::makeStrategy(vrp::StrategyKind::Generational);

        const vrp::Population previous = pop;
        strategy->step(generation, pop, problem, params, *exec);

        for (std::size_t slot = params.eliteCount; slot < previous.size(); ++slot) {
            const Offspring expected =
                expectedOffspring(previous, problem, params, generation, slot);
            INFO("threads " << threads << " slot " << slot);
            REQUIRE(std::equal(expected.route.begin(), expected.route.end(),
                               pop.route(slot).begin()));
            REQUIRE(pop.fitness(slot) == problem.routeDistance(expected.route));
        }
    }
}

TEST_CASE("steady state offspring follows the specified seeding and replacement",
          "[strategy]") {
    // The default strategy, and until now the only one with no oracle: every
    // other test that touches it asserts an invariant that a wrongly seeded,
    // wrongly parented or wrongly parameterised implementation also satisfies.
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    vrp::GaParams params = smallParams();
    params.populationSize = 64;
    vrp::Population pop(params.populationSize, problem, params.seed, *exec);
    auto strategy = vrp::makeStrategy(vrp::StrategyKind::SteadyState);

    // Non-zero, so mixSeed(seed, 0, generation) cannot coincide with the
    // specified mixSeed(seed, generation, 0).
    const std::size_t generation = 5;
    const vrp::Population previous = pop;
    const std::size_t target = previous.worstIndex();

    strategy->step(generation, pop, problem, params, *exec);

    const Offspring expected = expectedOffspring(previous, problem, params, generation, 0);

    // The fixture has to discriminate: if the two tournaments happened to
    // return the same index, an implementation that crossed one parent with
    // itself would be invisible below.
    REQUIRE(expected.firstParent != expected.secondParent);

    REQUIRE(std::equal(expected.route.begin(), expected.route.end(),
                       pop.route(target).begin()));
    REQUIRE(pop.fitness(target) == problem.routeDistance(expected.route));

    // Two-sided: the child lands on the worst individual and nowhere else.
    for (std::size_t i = 0; i < previous.size(); ++i) {
        if (i == target) {
            continue;
        }
        INFO("slot " << i);
        REQUIRE(std::equal(previous.route(i).begin(), previous.route(i).end(),
                           pop.route(i).begin()));
        REQUIRE(pop.fitness(i) == previous.fitness(i));
    }
}

TEST_CASE("strategies schedule at most one parallelFor per generation", "[strategy]") {
    // Granularity is a correctness-adjacent property that no output check can
    // see: scheduling per child computes the same population and pays a
    // condition-variable round trip per individual.
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    vrp::GaParams params = smallParams();
    params.populationSize = 40;
    const std::size_t generations = 5;

    SECTION("generational schedules exactly one per generation") {
        CountingExecutor exec(2);
        vrp::Population pop(params.populationSize, problem, params.seed, exec);
        const std::size_t afterConstruction = exec.calls;
        auto strategy = vrp::makeStrategy(vrp::StrategyKind::Generational);
        for (std::size_t g = 0; g < generations; ++g) {
            strategy->step(g, pop, problem, params, exec);
        }
        REQUIRE(exec.calls - afterConstruction == generations);
    }

    SECTION("steady state schedules none: its generation loop is serial") {
        CountingExecutor exec(2);
        vrp::Population pop(params.populationSize, problem, params.seed, exec);
        const std::size_t afterConstruction = exec.calls;
        auto strategy = vrp::makeStrategy(vrp::StrategyKind::SteadyState);
        for (std::size_t g = 0; g < generations; ++g) {
            strategy->step(g, pop, problem, params, exec);
        }
        REQUIRE(exec.calls == afterConstruction);
    }
}

TEST_CASE("generational with no elites derives every slot from the oracle", "[strategy]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    vrp::GaParams params = smallParams();
    params.populationSize = 32;
    params.eliteCount = 0;
    vrp::Population pop(params.populationSize, problem, params.seed, *exec);
    auto strategy = vrp::makeStrategy(vrp::StrategyKind::Generational);

    const std::size_t generation = 3;
    const vrp::Population previous = pop;
    strategy->step(generation, pop, problem, params, *exec);

    // Offspring start at slot 0 and the shift by `elites` is genuinely zero:
    // nothing is carried forward and no slot is skipped.
    for (std::size_t slot = 0; slot < previous.size(); ++slot) {
        const Offspring expected =
            expectedOffspring(previous, problem, params, generation, slot);
        INFO("slot " << slot);
        REQUIRE(std::equal(expected.route.begin(), expected.route.end(),
                           pop.route(slot).begin()));
    }
}

TEST_CASE("generational clamps an elite count at or above the population size",
          "[strategy]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    vrp::GaParams params = smallParams();
    params.populationSize = 16;
    params.eliteCount = params.populationSize + 5;  // deliberately past the end
    vrp::Population pop(params.populationSize, problem, params.seed, *exec);
    auto strategy = vrp::makeStrategy(vrp::StrategyKind::Generational);

    std::vector<double> before(pop.size());
    for (std::size_t i = 0; i < pop.size(); ++i) {
        before[i] = pop.fitness(i);
    }
    std::sort(before.begin(), before.end());

    // Without the std::min clamp, count - elites underflows and the offspring
    // range becomes astronomically large.
    strategy->step(0, pop, problem, params, *exec);

    REQUIRE(pop.size() == params.populationSize);
    std::vector<double> after(pop.size());
    for (std::size_t i = 0; i < pop.size(); ++i) {
        after[i] = pop.fitness(i);
    }
    // Everyone carried forward, nobody bred, and the elite order is the total
    // order on (fitness, index), so the result is the population sorted.
    REQUIRE(std::is_sorted(after.begin(), after.end()));
    REQUIRE(after == before);
}

TEST_CASE("generational elites break fitness ties by index", "[strategy]") {
    // Customers 1..3 share a location, so any ordering of them accumulates the
    // same distances in the same order and ties bit-exactly: three distinct
    // routes with one fitness. Customer 4 is far away, and visiting it early
    // pays the detour twice. With eliteCount = 2 the tie straddles the elite
    // boundary, which is the only place a comparator on fitness alone can
    // diverge from the total order on (fitness, index).
    const vrp::Problem problem(
        std::vector<vrp::Point>{{0, 0}, {10, 0}, {10, 0}, {10, 0}, {0, 100}});

    const std::vector<std::vector<int>> routes = {
        {1, 4, 2, 3},  // strictly worse
        {1, 2, 3, 4},
        {2, 1, 3, 4},
        {3, 2, 1, 4},
    };
    vrp::Population pop(routes.size(), problem.customerCount());
    for (std::size_t i = 0; i < routes.size(); ++i) {
        pop.setRoute(i, routes[i], problem);
    }
    REQUIRE(pop.fitness(1) == pop.fitness(2));
    REQUIRE(pop.fitness(2) == pop.fitness(3));
    REQUIRE(pop.fitness(0) > pop.fitness(1));

    vrp::GaParams params;
    params.populationSize = routes.size();
    params.eliteCount = 2;
    params.seed = 99;
    auto exec = vrp::makeExecutor(1);
    auto strategy = vrp::makeStrategy(vrp::StrategyKind::Generational);
    strategy->step(0, pop, problem, params, *exec);

    // Individuals 1, 2 and 3 all tie for best; the total order picks 1 then 2,
    // in that order, and puts them in the leading slots.
    REQUIRE(std::equal(routes[1].begin(), routes[1].end(), pop.route(0).begin()));
    REQUIRE(std::equal(routes[2].begin(), routes[2].end(), pop.route(1).begin()));
}
