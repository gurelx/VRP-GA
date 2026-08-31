#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>
#include "vrp/Config.hpp"
#include "vrp/Executor.hpp"
#include "vrp/Population.hpp"
#include "vrp/Problem.hpp"
#include "vrp/Solver.hpp"
#include "vrp/Strategy.hpp"
#include "SolverTrace.hpp"

using vrp_test::kindName;
using vrp_test::referenceTrace;
using vrp_test::Trace;

namespace {

vrp::GaParams smallParams() {
    vrp::GaParams params;
    params.populationSize = 300;
    params.generations = 40;
    params.seed = 2024;
    return params;
}

// Deliberately not Problem::defaultInstance(): every other case here uses the
// default, which would let a run() that built its own default go unnoticed.
vrp::Problem customProblem() {
    return vrp::Problem(std::vector<vrp::Point>{
        {0, 0}, {10, 0}, {10, 10}, {0, 10}, {20, 5}, {30, 30}});
}

// The exhaustive best over all 5! tours, derived from the instance's geometry
// rather than from anything the solver produced.
double bruteForceOptimum(const vrp::Problem& problem) {
    std::vector<int> route(problem.customerCount());
    std::iota(route.begin(), route.end(), 1);
    double best = std::numeric_limits<double>::infinity();
    do {
        best = std::min(best, problem.routeDistance(route));
    } while (std::next_permutation(route.begin(), route.end()));
    return best;
}

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

// solverTrace stays here rather than in SolverTrace.hpp: this file drives a
// caller-owned Solver, so two runs can share one instance and its scratch.
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

// SUBSUMED by "solver reproduces a hand-run generation loop"; kept as the one
// case that fails loudly if run() returns a default-constructed RunResult.
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

    // Ties generationsRun to the iterations actually observed, rather than
    // letting it restate the request.
    REQUIRE(generations.size() == result.generationsRun);

    for (std::size_t i = 0; i < generations.size(); ++i) {
        REQUIRE(generations[i] == i);
    }

    // Weak on its own: under steady-state the incumbent best is flat across the
    // late generations, so a report shifted by one still matches here and dies
    // only on the Trace comparison below.
    REQUIRE(reported.back() == result.bestDistance);
}

// SUBSUMED by "solver reproduces a hand-run generation loop": monotonicity is
// GenerationalStrategy's elitism invariant, owned by test_strategies.cpp, and
// holds for a sequence shifted by one or reported before the step.
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

        // A shifted report is only detectable if the incumbent best actually
        // moves during the run, so check that it does rather than assuming it.
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
    // Wiring test for the Problem reference: a run() that ignored problem_ and
    // built its own default passes every other case in this file.
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

    REQUIRE(result.bestRoute.size() == 5);
    std::vector<int> sorted = result.bestRoute;
    std::sort(sorted.begin(), sorted.end());
    REQUIRE(sorted == std::vector<int>{1, 2, 3, 4, 5});

    REQUIRE(result.bestDistance == problem.routeDistance(result.bestRoute));
    REQUIRE(result.bestDistance == bruteForceOptimum(problem));
    REQUIRE(result.bestDistance !=
            vrp::Problem::defaultInstance().routeDistance(result.bestRoute));
}

TEST_CASE("solver schedules through the executor it was handed", "[solver]") {
    // Wiring test for the Executor: results are thread-count independent by
    // design, so whether this object was driven is the only observable.
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
        // The single call is the population constructor's; steady-state never
        // schedules.
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
    // Both strategies keep scratch between step() calls, so a second run()
    // begins against a dirty strategy. It has to match both the first run and a
    // run from a fresh strategy; matching only itself would hide carried state.
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
    // seed produced.
    const Trace expected = referenceTrace(problem, params, vrp::StrategyKind::Generational);
    REQUIRE(result.bestRoute == expected.bestRoute);
    REQUIRE(result.bestDistance == expected.bestDistance);
    REQUIRE(result.bestDistance == problem.routeDistance(result.bestRoute));
    REQUIRE(result.bestDistance > 0.0);
}

// SUBSUMED by the two cases below, and cannot fail: elapsedSeconds >= 0 holds
// even for an implementation that times nothing.
TEST_CASE("elapsed time is recorded", "[solver]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    vrp::Solver solver(problem, smallParams(),
                       vrp::makeStrategy(vrp::StrategyKind::Generational),
                       vrp::makeExecutor(1));
    const vrp::RunResult result = solver.run();
    REQUIRE(result.elapsedSeconds >= 0.0);
}

TEST_CASE("elapsed time covers the initial population", "[solver]") {
    // Self-calibrating: the solver's interval is nested inside the wall time
    // measured here, and with no generations the population construction is all
    // of it. The population is deliberately huge -- shrinking it leaves too
    // little between the two clock reads for the ratio to hold in Release.
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
    // The mirror of the case above: a tiny population and a sleeping callback,
    // so the loop is nearly all of the wall time.
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
