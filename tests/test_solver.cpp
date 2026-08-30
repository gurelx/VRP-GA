#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <thread>
#include <utility>
#include <vector>
#include "vrp/Config.hpp"
#include "vrp/Executor.hpp"
#include "vrp/Population.hpp"
#include "vrp/Problem.hpp"
#include "vrp/Solver.hpp"
#include "vrp/Strategy.hpp"

namespace {

vrp::GaParams smallParams() {
    vrp::GaParams params;
    params.populationSize = 300;
    params.generations = 40;
    params.seed = 2024;
    return params;
}

const char* kindName(vrp::StrategyKind kind) {
    return kind == vrp::StrategyKind::SteadyState ? "steady-state" : "generational";
}

// A six-point instance that is emphatically not Problem::defaultInstance():
// five customers rather than nineteen, and different coordinates. Every other
// test in this suite -- and, at the time of writing, every other Problem
// anywhere in the test tree -- uses the default instance, which would let a
// run() that ignored the Problem it was handed and built its own default go
// entirely unnoticed.
vrp::Problem customProblem() {
    return vrp::Problem(std::vector<vrp::Point>{
        {0, 0}, {10, 0}, {10, 10}, {0, 10}, {20, 5}, {30, 30}});
}

// The exhaustive best over all 5! tours. Five customers is small enough to
// state the answer outright instead of asserting a property of it, and the
// answer is derived from the instance's own geometry rather than from anything
// the solver produced.
double bruteForceOptimum(const vrp::Problem& problem) {
    std::vector<int> route(problem.customerCount());
    std::iota(route.begin(), route.end(), 1);
    double best = std::numeric_limits<double>::infinity();
    do {
        best = std::min(best, problem.routeDistance(route));
    } while (std::next_permutation(route.begin(), route.end()));
    return best;
}

// Forwards every call to a real executor and records how it was driven.
// Executor is an abstract base, so this needs nothing from the library beyond
// the interface itself.
class CountingExecutor final : public vrp::Executor {
public:
    explicit CountingExecutor(std::unique_ptr<vrp::Executor> inner)
        : inner_(std::move(inner)) {}

    std::size_t threadCount() const noexcept override { return inner_->threadCount(); }

    void parallelFor(std::size_t n, const vrp::ParallelBody& body) override {
        itemCounts_.push_back(n);
        inner_->parallelFor(n, body);
    }

    const std::vector<std::size_t>& itemCounts() const noexcept { return itemCounts_; }

private:
    std::unique_ptr<vrp::Executor> inner_;
    std::vector<std::size_t> itemCounts_;
};

// Everything one run can be observed to produce, so a comparison covers the
// whole contract rather than the one field a mutant happened to leave alone.
struct Trace {
    std::vector<std::size_t> generations;  // as reported to the callback
    std::vector<double> best;              // best distance as reported, per generation
    std::vector<int> bestRoute;
    double bestDistance = 0.0;
    std::size_t generationsRun = 0;
};

// A hand-run generation loop, sharing no code with Solver: build the population
// from the seed, apply the strategy exactly `generations` times, and record the
// incumbent best AFTER each step. A loop that steps once too often or too few
// times, one that reports before stepping instead of after, or one that hands
// the strategy the wrong generation index all diverge from this.
Trace referenceTrace(const vrp::Problem& problem, const vrp::GaParams& params,
                     vrp::StrategyKind kind) {
    auto executor = vrp::makeExecutor(1);
    auto strategy = vrp::makeStrategy(kind);
    vrp::Population population(params.populationSize, problem, params.seed, *executor);

    Trace trace;
    for (std::size_t generation = 0; generation < params.generations; ++generation) {
        strategy->step(generation, population, problem, params, *executor);
        trace.generations.push_back(generation);
        trace.best.push_back(population.fitness(population.bestIndex()));
    }
    trace.generationsRun = params.generations;

    const std::size_t best = population.bestIndex();
    const std::span<const int> route = population.route(best);
    trace.bestRoute.assign(route.begin(), route.end());
    trace.bestDistance = population.fitness(best);
    return trace;
}

Trace solverTrace(vrp::Solver& solver) {
    Trace trace;
    const vrp::RunResult result =
        solver.run([&](std::size_t generation, double best) {
            trace.generations.push_back(generation);
            trace.best.push_back(best);
        });
    trace.bestRoute = result.bestRoute;
    trace.bestDistance = result.bestDistance;
    trace.generationsRun = result.generationsRun;
    return trace;
}

}  // namespace

// SUBSUMED. Every index into the population holds a valid permutation, so this
// catches no index error and no loop-count error; it only rules out a garbage
// or empty route. What actually covers the returned route is "solver reproduces
// a hand-run generation loop" (exact equality against an independent oracle)
// and "zero generations returns the initial best". Kept because the brief
// specifies it and it is the one case that fails loudly if run() returns a
// default-constructed RunResult.
TEST_CASE("solver returns a valid best route", "[solver]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    const vrp::GaParams params = smallParams();
    vrp::Solver solver(problem, params,
                       vrp::makeStrategy(vrp::StrategyKind::Generational),
                       vrp::makeExecutor(1));
    const vrp::RunResult result = solver.run();

    REQUIRE(result.bestRoute.size() == problem.customerCount());
    std::vector<int> sorted = result.bestRoute;
    std::sort(sorted.begin(), sorted.end());
    for (std::size_t i = 0; i < sorted.size(); ++i) {
        REQUIRE(sorted[i] == static_cast<int>(i + 1));
    }
}

TEST_CASE("reported distance matches the reported route", "[solver]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    vrp::Solver solver(problem, smallParams(),
                       vrp::makeStrategy(vrp::StrategyKind::Generational),
                       vrp::makeExecutor(1));
    const vrp::RunResult result = solver.run();
    REQUIRE(result.bestDistance == problem.routeDistance(result.bestRoute));
}

TEST_CASE("the progress callback fires once per generation", "[solver]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    const vrp::GaParams params = smallParams();
    vrp::Solver solver(problem, params,
                       vrp::makeStrategy(vrp::StrategyKind::SteadyState),
                       vrp::makeExecutor(1));

    std::vector<std::size_t> generations;
    std::vector<double> reported;
    const vrp::RunResult result =
        solver.run([&](std::size_t generation, double best) {
            generations.push_back(generation);
            reported.push_back(best);
        });

    REQUIRE(generations.size() == params.generations);
    REQUIRE(generations.front() == 0);
    REQUIRE(generations.back() == params.generations - 1);
    REQUIRE(result.generationsRun == params.generations);

    // Ties generationsRun to the number of iterations that were actually
    // observed to happen, rather than letting it restate the request.
    REQUIRE(generations.size() == result.generationsRun);

    // 0, 1, ... N-1 with nothing skipped or repeated.
    for (std::size_t i = 0; i < generations.size(); ++i) {
        REQUIRE(generations[i] == i);
    }

    // Ties the last report to the returned answer. Weak on its own, and
    // deliberately not credited with more than it does: under steady-state the
    // incumbent best is flat across the late generations, so a report shifted
    // by one generation still has a matching back(). The mutation run bears
    // this out -- a callback fired before the step instead of after survives
    // this line and dies only on the Trace comparison below.
    REQUIRE(reported.back() == result.bestDistance);
}

// SUBSUMED, and weaker than it looks. Monotonicity is GenerationalStrategy's
// elitism invariant, established in Task 7 -- this restates it through Solver
// rather than testing Solver. It holds for a sequence shifted by one, for a
// report issued before the step, and even for fitness(0) in place of
// fitness(bestIndex()), since slot 0 is where the elite is written. It killed
// none of the ten mutants. The Trace comparison is what covers the reported
// sequence.
TEST_CASE("reported progress never worsens", "[solver]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    vrp::Solver solver(problem, smallParams(),
                       vrp::makeStrategy(vrp::StrategyKind::Generational),
                       vrp::makeExecutor(1));

    double previous = std::numeric_limits<double>::infinity();
    solver.run([&](std::size_t, double best) {
        REQUIRE(best <= previous);
        previous = best;
    });
}

TEST_CASE("solver reproduces a hand-run generation loop", "[solver]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    const vrp::GaParams params = smallParams();

    for (const vrp::StrategyKind kind :
         {vrp::StrategyKind::SteadyState, vrp::StrategyKind::Generational}) {
        INFO("strategy = " << kindName(kind));
        const Trace expected = referenceTrace(problem, params, kind);

        // This comparison can only detect a report shifted by one generation if
        // the incumbent best actually moves during the run, so check that it
        // does rather than assuming it: if every reported value were equal, a
        // shifted sequence would compare equal too.
        REQUIRE(expected.best.front() != expected.best.back());

        vrp::Solver solver(problem, params, vrp::makeStrategy(kind), vrp::makeExecutor(1));
        const Trace actual = solverTrace(solver);

        REQUIRE(actual.generations == expected.generations);
        REQUIRE(actual.best == expected.best);
        REQUIRE(actual.bestRoute == expected.bestRoute);
        REQUIRE(actual.bestDistance == expected.bestDistance);
        REQUIRE(actual.generationsRun == expected.generationsRun);
    }
}

TEST_CASE("solver evaluates against the problem it was handed", "[solver]") {
    // Wiring test for the Problem reference. A run() that ignored problem_ and
    // constructed a default instance of its own passes every other case in
    // this file, because every other case hands it the default instance.
    const vrp::Problem problem = customProblem();
    REQUIRE(problem.customerCount() == 5);
    REQUIRE(problem.customerCount() != vrp::Problem::defaultInstance().customerCount());

    vrp::GaParams params;
    params.populationSize = 200;
    params.generations = 20;
    params.seed = 90210;

    vrp::Solver solver(problem, params,
                       vrp::makeStrategy(vrp::StrategyKind::Generational),
                       vrp::makeExecutor(1));
    const vrp::RunResult result = solver.run();

    // The route's shape follows this instance's customer count, not the
    // default's nineteen.
    REQUIRE(result.bestRoute.size() == 5);
    std::vector<int> sorted = result.bestRoute;
    std::sort(sorted.begin(), sorted.end());
    REQUIRE(sorted == std::vector<int>{1, 2, 3, 4, 5});

    // And the distance comes from this instance's geometry, not the default's:
    // the same five genes score differently under the two.
    REQUIRE(result.bestDistance == problem.routeDistance(result.bestRoute));
    REQUIRE(result.bestDistance == bruteForceOptimum(problem));
    REQUIRE(result.bestDistance !=
            vrp::Problem::defaultInstance().routeDistance(result.bestRoute));
}

TEST_CASE("solver schedules through the executor it was handed", "[solver]") {
    // Wiring test for the Executor. A run() that ignored executor_ and used a
    // local SerialExecutor would return identical numbers -- results are
    // thread-count independent by design, which is the whole point of Task 9 --
    // so the only observable difference is whether this object was driven.
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    vrp::GaParams params;
    params.populationSize = 64;
    params.generations = 7;
    params.seed = 5150;

    SECTION("generational drives it once for the population and once per generation") {
        auto spy = std::make_unique<CountingExecutor>(vrp::makeExecutor(1));
        const CountingExecutor* observer = spy.get();
        vrp::Solver solver(problem, params,
                           vrp::makeStrategy(vrp::StrategyKind::Generational),
                           std::move(spy));
        const vrp::RunResult result = solver.run();

        REQUIRE_FALSE(observer->itemCounts().empty());
        REQUIRE(observer->itemCounts().size() == 1 + params.generations);
        REQUIRE(observer->itemCounts().front() == params.populationSize);
        for (std::size_t i = 1; i < observer->itemCounts().size(); ++i) {
            INFO("call " << i);
            REQUIRE(observer->itemCounts()[i] > 0);
        }
        REQUIRE(result.bestRoute.size() == problem.customerCount());
    }

    SECTION("steady-state drives it only for the population") {
        // Not an accident worth preserving by luck: SteadyStateStrategy is
        // inherently serial and documents that it never touches the executor,
        // so the single call is the population constructor's.
        auto spy = std::make_unique<CountingExecutor>(vrp::makeExecutor(1));
        const CountingExecutor* observer = spy.get();
        vrp::Solver solver(problem, params,
                           vrp::makeStrategy(vrp::StrategyKind::SteadyState),
                           std::move(spy));
        const vrp::RunResult result = solver.run();

        REQUIRE(observer->itemCounts().size() == 1);
        REQUIRE(observer->itemCounts().front() == params.populationSize);
        REQUIRE(result.bestRoute.size() == problem.customerCount());
    }
}

TEST_CASE("running the same solver twice reproduces the run", "[solver]") {
    // Both strategies keep scratch between step() calls -- next_, order_,
    // child_, seen_ -- so a second run() begins against a dirty strategy
    // object. It has to match both the first run and a run from a fresh
    // strategy; matching only itself would still hide carried-over state.
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    const vrp::GaParams params = smallParams();

    for (const vrp::StrategyKind kind :
         {vrp::StrategyKind::SteadyState, vrp::StrategyKind::Generational}) {
        INFO("strategy = " << kindName(kind));
        const Trace fresh = referenceTrace(problem, params, kind);

        vrp::Solver solver(problem, params, vrp::makeStrategy(kind), vrp::makeExecutor(1));
        const Trace first = solverTrace(solver);
        const Trace second = solverTrace(solver);

        REQUIRE(first.best == fresh.best);
        REQUIRE(second.best == first.best);
        REQUIRE(second.bestRoute == first.bestRoute);
        REQUIRE(second.bestDistance == first.bestDistance);
        REQUIRE(second.generationsRun == first.generationsRun);
    }
}

TEST_CASE("zero generations returns the initial best", "[solver]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    vrp::GaParams params = smallParams();
    params.generations = 0;
    vrp::Solver solver(problem, params,
                       vrp::makeStrategy(vrp::StrategyKind::Generational),
                       vrp::makeExecutor(1));

    std::vector<std::size_t> generations;
    const vrp::RunResult result =
        solver.run([&](std::size_t generation, double) { generations.push_back(generation); });

    REQUIRE(result.generationsRun == 0);
    REQUIRE(generations.empty());
    REQUIRE(result.bestRoute.size() == problem.customerCount());

    // Not a default-constructed result: it is the best of the population the
    // seed produced, which is exactly what evolving nothing should yield.
    const Trace expected = referenceTrace(problem, params, vrp::StrategyKind::Generational);
    REQUIRE(result.bestRoute == expected.bestRoute);
    REQUIRE(result.bestDistance == expected.bestDistance);
    REQUIRE(result.bestDistance == problem.routeDistance(result.bestRoute));
    REQUIRE(result.bestDistance > 0.0);
}

// SUBSUMED, and cannot fail: elapsedSeconds is a duration between two reads of
// a monotonic clock, so it is >= 0 for every implementation including one that
// times nothing at all. Both timing mutants pass it. The two tests below are
// what actually pin the measured interval. Retained only because the brief
// specifies it.
TEST_CASE("elapsed time is recorded", "[solver]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    vrp::Solver solver(problem, smallParams(),
                       vrp::makeStrategy(vrp::StrategyKind::Generational),
                       vrp::makeExecutor(1));
    const vrp::RunResult result = solver.run();
    REQUIRE(result.elapsedSeconds >= 0.0);
}

TEST_CASE("elapsed time covers the initial population", "[solver]") {
    // Self-calibrating rather than threshold-based: the solver's interval is
    // nested inside the one measured here, so it can never be longer, and with
    // no generations to run, the population construction is the whole of what
    // it should be timing. A clock started after that construction reports
    // near zero against a wall time of tens of milliseconds.
    //
    // The population is large because this test is thinnest in a release
    // build, where the construction is the only thing standing between the two
    // clock reads: at 50000 individuals it runs in ~2.4 ms optimised, so a
    // preemption in the microseconds outside the solver's own interval could
    // move the ratio materially. 400000 costs ~20 ms optimised and ~250 ms
    // unoptimised, an order of magnitude more headroom. Widen the margin here
    // rather than lowering the ratio, which would blunt the test everywhere.
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    vrp::GaParams params;
    params.populationSize = 400000;
    params.generations = 0;
    params.seed = 11;
    vrp::Solver solver(problem, params,
                       vrp::makeStrategy(vrp::StrategyKind::Generational),
                       vrp::makeExecutor(1));

    const auto before = std::chrono::steady_clock::now();
    const vrp::RunResult result = solver.run();
    const double wall =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - before).count();

    REQUIRE(wall > 0.0);
    REQUIRE(result.elapsedSeconds <= wall);
    REQUIRE(result.elapsedSeconds >= 0.5 * wall);
}

TEST_CASE("elapsed time covers the generation loop", "[solver]") {
    // The mirror of the test above: a tiny population and a callback that
    // sleeps, so the loop is nearly all of the wall time. A clock stopped
    // before the loop, or started after it, reports a small fraction of it.
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    vrp::GaParams params;
    params.populationSize = 100;
    params.generations = 5;
    params.seed = 11;
    vrp::Solver solver(problem, params,
                       vrp::makeStrategy(vrp::StrategyKind::Generational),
                       vrp::makeExecutor(1));

    const auto before = std::chrono::steady_clock::now();
    const vrp::RunResult result = solver.run([](std::size_t, double) {
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    });
    const double wall =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - before).count();

    REQUIRE(wall >= 0.05);  // the five sleeps happened
    REQUIRE(result.elapsedSeconds <= wall);
    REQUIRE(result.elapsedSeconds >= 0.5 * wall);
}
