#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>
#include "vrp/Config.hpp"
#include "vrp/Executor.hpp"
#include "vrp/Population.hpp"
#include "vrp/Problem.hpp"
#include "vrp/Solver.hpp"
#include "vrp/Strategy.hpp"

// ---------------------------------------------------------------------------
// The determinism suite, at the Solver layer.
//
// The layers below already own two distinct claims, and this file must not
// restate either of them:
//
//   * CONFORMANCE. Task 5 replays mixSeed(seed, kInitDomain, i) and the shuffle
//     by hand; Task 7 replays mixSeed(seed, generation, slot) and the whole
//     operator sequence by hand, per slot, for both strategies. Those oracles
//     are what rule out a wrong-but-consistent implementation -- a transposed
//     mixSeed, a chunk-relative index, a crossover handed one parent twice --
//     which cross-thread agreement is structurally blind to.
//
//   * AGREEMENT AT THE STRATEGY LAYER. Task 7 already compares whole
//     populations across thread counts for a hand-run generation loop.
//
// What is genuinely missing, and what this file adds, is the composition:
// Solver hands ONE executor to two chunk-independent consumers -- the
// Population constructor and the strategy -- and drives them in a loop for
// N generations. Nothing so far asserts that the composed, threaded
// Solver::run() reproduces a serial one at all; Task 8's loop oracle runs at
// threads = 1 only. The central test below asserts it over the entire progress
// trace rather than the final answer alone, so a divergence at generation 3
// that later washes out of the best individual is still a failure.
//
// BE PRECISE ABOUT WHAT THAT IS WORTH. Every comparison in this file is
// DIFFERENTIAL: it runs the same library code twice and compares the two
// results. None of it is an independent computation of the specified answer,
// and none of it can be -- the only way to observe a whole population is to
// drive Population and EvolutionStrategy::step, which is the code under test.
// Under the chunk-INdependent defects (a transposed mixSeed, a crossover
// handed the same parent twice) both sides are equally wrong and every
// comparison here passes. That is a statement about THOSE defects, not a
// universal: `different seeds produce different results` below is itself a
// detector for one chunk-independent defect -- ignoring params.seed
// altogether -- and it fires on it. Only the per-slot oracles in
// test_population.cpp and test_strategies.cpp rule out the rest, and the link
// from them to this file is an INDUCTIVE argument across files: Task 7 pins a
// single generation at population 64, eliteCount 2, generation 7, while this
// chains 25 generations at 512 with eliteCount 1 -- not an entailment.
//
// So: do not cite anything here as evidence that the specified seed derivation
// is the one implemented -- that is the oracles' claim, not this file's. Do
// cite it for the two things it does earn: that the thread count changes no
// value, and -- from the hand-run loop, at threads = 1, which is not a
// thread-count claim at all -- that Solver's own loop structure is right: no
// off-by-one, no report issued before the step instead of after, no wrong
// generation index handed to the strategy.
//
// Exact `==` on doubles is intended everywhere in this file. Per-item seeding
// makes every value bit-identical across thread counts; a tolerance would hide
// precisely the divergence these tests exist to catch. If one of these
// comparisons ever fails, the fix is to pin the order of operations, never to
// loosen the comparison.
// ---------------------------------------------------------------------------

namespace {

// The two strategies do wildly different amounts of work per generation --
// generational breeds a full population, steady-state replaces exactly one
// individual -- so a shared generation count cannot give both a run in which
// evolution actually happened. At 512 individuals the specified 25 generations
// leave steady-state's incumbent best untouched (measured: it first improves at
// generation 122), which makes its progress trace a constant and its final
// population 95% the initial one. A constant trace discriminates nothing: a
// comparison against it would pass for a run shifted by a generation, or one
// reading the wrong population. Give steady-state a count past that threshold
// instead; at one child per generation it costs almost nothing.
vrp::GaParams determinismParams(std::uint64_t seed, vrp::StrategyKind kind) {
    vrp::GaParams params;
    params.populationSize = 512;
    params.generations = kind == vrp::StrategyKind::SteadyState ? 200 : 25;
    params.seed = seed;
    return params;
}

const char* kindName(vrp::StrategyKind kind) {
    return kind == vrp::StrategyKind::SteadyState ? "steady-state" : "generational";
}

// Everything one Solver run can be observed to produce. The per-generation
// `best` sequence is the only window Solver offers onto the intermediate
// populations, and it is what makes this comparison stronger than RunResult
// alone: a run that diverged at generation 3 and reconverged by generation 24
// agrees on bestRoute and dies here.
struct Trace {
    std::vector<std::size_t> generations;
    std::vector<double> best;
    std::vector<int> bestRoute;
    double bestDistance = 0.0;
    std::size_t generationsRun = 0;
};

// A serial generation loop kept separate from Solver's own, so that a defect
// inside Solver::run's loop -- an off-by-one, a report issued before the step
// instead of after, the wrong generation index handed to the strategy -- cannot
// cancel out of the comparison by appearing identically on both sides.
//
// It is NOT independent of the library, and the comment it replaced wrongly
// said it was. It calls the same Population constructor and the same
// EvolutionStrategy::step as the code under test, so it is a differential
// oracle over Solver's LOOP, not a restatement of the specified computation.
//
// DUPLICATION, DELIBERATE AND LOAD-BEARING: `Trace`, `referenceTrace` and
// `kindName` are byte-identical to the helpers in tests/test_solver.cpp, which
// uses them for the same purpose at threads = 1. They are copied rather than
// shared so that one edit cannot silently retune both files' oracles at once;
// the price is that the two copies can drift apart unnoticed. If you change
// one, change the other, or extract both into a shared test-support header.
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

Trace solverTrace(const vrp::Problem& problem, const vrp::GaParams& params,
                  vrp::StrategyKind kind, std::size_t threads) {
    // `problem` is a caller-owned reference precisely because Solver stores one:
    // a Problem constructed inside this function would still outlive the solver,
    // but making the caller own it keeps that requirement visible.
    vrp::Solver solver(problem, params, vrp::makeStrategy(kind),
                       vrp::makeExecutor(threads));
    Trace trace;
    const vrp::RunResult result = solver.run([&](std::size_t generation, double best) {
        trace.generations.push_back(generation);
        trace.best.push_back(best);
    });
    trace.bestRoute = result.bestRoute;
    trace.bestDistance = result.bestDistance;
    trace.generationsRun = result.generationsRun;
    return trace;
}

vrp::RunResult runWith(const vrp::Problem& problem, vrp::StrategyKind kind,
                       std::size_t threads, std::uint64_t seed) {
    vrp::Solver solver(problem, determinismParams(seed, kind), vrp::makeStrategy(kind),
                       vrp::makeExecutor(threads));
    return solver.run();
}

// A whole population, flattened, so two runs can be compared element by
// element. Solver deliberately returns only the winner, so seeing every
// individual means running the loop by hand -- there is no other window.
struct Snapshot {
    std::size_t routeLength = 0;
    std::vector<int> routes;
    std::vector<double> fitness;
    std::size_t bestIndex = 0;
    std::vector<int> bestRoute;
    double bestDistance = 0.0;
};

// Lowest-index tie-breaking, matching Population::bestIndex. Recomputed rather
// than carried, so the discrimination check below can ask what a best-only
// comparison would have concluded about a MODIFIED snapshot.
std::size_t bestIndexOf(const Snapshot& snapshot) {
    std::size_t best = 0;
    for (std::size_t i = 1; i < snapshot.fitness.size(); ++i) {
        if (snapshot.fitness[i] < snapshot.fitness[best]) {
            best = i;
        }
    }
    return best;
}

Snapshot evolveSnapshot(const vrp::Problem& problem, const vrp::GaParams& params,
                        vrp::StrategyKind kind, std::size_t threads) {
    auto executor = vrp::makeExecutor(threads);
    auto strategy = vrp::makeStrategy(kind);
    vrp::Population population(params.populationSize, problem, params.seed, *executor);
    for (std::size_t g = 0; g < params.generations; ++g) {
        strategy->step(g, population, problem, params, *executor);
    }

    Snapshot snapshot;
    snapshot.routeLength = population.routeLength();
    snapshot.fitness.reserve(population.size());
    snapshot.routes.reserve(population.size() * population.routeLength());
    for (std::size_t i = 0; i < population.size(); ++i) {
        snapshot.fitness.push_back(population.fitness(i));
        for (const int gene : population.route(i)) {
            snapshot.routes.push_back(gene);
        }
    }
    snapshot.bestIndex = bestIndexOf(snapshot);
    const std::span<const int> best = population.route(snapshot.bestIndex);
    snapshot.bestRoute.assign(best.begin(), best.end());
    snapshot.bestDistance = snapshot.fitness[snapshot.bestIndex];
    return snapshot;
}

}  // namespace

// ---------------------------------------------------------------------------
// The guard that keeps every other case in this file from passing vacuously.
//
// A Solver that never read params.seed at all -- one that hard-coded a
// constant, or derived everything from a fixed default -- satisfies every
// repeatability and every cross-thread comparison here perfectly. This is the
// one test that fails on it, so it is a precondition for reading anything below
// as evidence, not a throwaway.
// ---------------------------------------------------------------------------
TEST_CASE("different seeds produce different results", "[determinism]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    const vrp::RunResult a = runWith(problem, vrp::StrategyKind::Generational, 1, 1);
    const vrp::RunResult b = runWith(problem, vrp::StrategyKind::Generational, 1, 2);
    REQUIRE(a.bestRoute != b.bestRoute);
}

TEST_CASE("the same seed reproduces the same result", "[determinism]") {
    // Two INDEPENDENT Solver instances, not two run() calls on one -- Task 8
    // owns the same-instance case, including the carried strategy scratch.
    // What is left here is cross-instance state, and the threaded pass is the
    // part that is not subsumed: a thread pool is the natural home for
    // thread-local scratch or a worker-identity-derived stream, neither of
    // which a serial repeat can see.
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    for (const vrp::StrategyKind kind :
         {vrp::StrategyKind::SteadyState, vrp::StrategyKind::Generational}) {
        for (const std::size_t threads : {std::size_t{1}, std::size_t{8}}) {
            INFO("strategy = " << kindName(kind) << ", threads = " << threads);
            const vrp::RunResult first = runWith(problem, kind, threads, 777);
            const vrp::RunResult second = runWith(problem, kind, threads, 777);
            REQUIRE(first.bestRoute == second.bestRoute);
            REQUIRE(first.bestDistance == second.bestDistance);
        }
    }
}

// ---------------------------------------------------------------------------
// The load-bearing test.
// ---------------------------------------------------------------------------
TEST_CASE("a threaded solver run reproduces a hand-run serial loop", "[determinism]") {
    // Per-item seeding means the thread count must not influence any value.
    // The reference is a separate serial loop rather than a serial Solver run,
    // so a defect inside Solver::run's own loop cannot cancel out of the
    // comparison. That is the whole of the extra claim: this remains a
    // differential check, and it fires on chunk-relative seeding only because
    // that particular wrongness happens to depend on the chunking. A
    // chunk-independent defect appears identically on both sides and passes
    // here -- see the header comment.
    //
    // The threads = 1 arm is an anchor for the threaded arms, not new
    // coverage: test_solver.cpp already asserts exactly it, with the same five
    // REQUIREs against the same oracle.
    //
    // The comparison covers the whole progress trace, not just the returned
    // best: a divergence at generation 3 that reconverges by generation 24
    // leaves bestRoute and bestDistance identical and is invisible to a
    // result-only check.
    //
    // Steady-state is included but is weak here by construction, and must not
    // be counted as evidence about threading: SteadyStateStrategy ignores its
    // executor entirely, so the only thing the thread count reaches is the
    // Population constructor, whose determinism Task 5 owns. It stays as a
    // regression guard for the day that strategy starts scheduling work.
    const vrp::Problem problem = vrp::Problem::defaultInstance();

    for (const vrp::StrategyKind kind :
         {vrp::StrategyKind::SteadyState, vrp::StrategyKind::Generational}) {
        INFO("strategy = " << kindName(kind));
        const vrp::GaParams params = determinismParams(31337, kind);
        const Trace expected = referenceTrace(problem, params, kind);

        // The fixture has to discriminate. If the best were flat across the
        // whole run, a trace shifted by a generation -- or one recorded from
        // the wrong population -- would compare equal, and the `best` vector
        // would be contributing nothing.
        REQUIRE(expected.generationsRun == params.generations);
        REQUIRE(expected.best.size() == params.generations);
        REQUIRE(expected.best.front() != expected.best.back());

        for (const std::size_t threads :
             {std::size_t{1}, std::size_t{2}, std::size_t{4}, std::size_t{8}}) {
            INFO("threads = " << threads);
            const Trace actual = solverTrace(problem, params, kind, threads);
            REQUIRE(actual.generations == expected.generations);
            REQUIRE(actual.best == expected.best);
            REQUIRE(actual.bestRoute == expected.bestRoute);
            REQUIRE(actual.bestDistance == expected.bestDistance);
            REQUIRE(actual.generationsRun == expected.generationsRun);
        }
    }
}

TEST_CASE("entire populations are bit-identical across thread counts", "[determinism]") {
    // Comparing only the winner could hide a divergence anywhere else in the
    // population, so compare all 512 individuals. Largely a scale-up of Task
    // 7's cross-thread case for generational -- 512 individuals rather than 97
    // -- and vacuous for steady-state for the reason given above. It earns its
    // place on the discrimination check below, which is what justifies every
    // whole-population comparison in the project.
    const vrp::Problem problem = vrp::Problem::defaultInstance();

    for (const vrp::StrategyKind kind :
         {vrp::StrategyKind::SteadyState, vrp::StrategyKind::Generational}) {
        INFO("strategy = " << kindName(kind));
        const vrp::GaParams params = determinismParams(8675309, kind);
        const Snapshot serial = evolveSnapshot(problem, params, kind, 1);
        REQUIRE(serial.fitness.size() == params.populationSize);
        REQUIRE(serial.routes.size() == params.populationSize * serial.routeLength);

        for (const std::size_t threads :
             {std::size_t{2}, std::size_t{4}, std::size_t{8}}) {
            INFO("threads = " << threads);
            const Snapshot threaded = evolveSnapshot(problem, params, kind, threads);
            REQUIRE(threaded.routes == serial.routes);
            REQUIRE(threaded.fitness == serial.fitness);
        }

        // Discrimination: demonstrate, rather than assert, that comparing the
        // whole population is strictly stronger than comparing the winner.
        // Perturb one NON-best individual and show that the comparison above
        // rejects the result while bestRoute and bestDistance -- recomputed
        // from the perturbed snapshot, not carried over from the original --
        // still agree exactly. A best-only comparison would pass this.
        Snapshot perturbed = serial;
        const std::size_t victim = static_cast<std::size_t>(
            std::max_element(serial.fitness.begin(), serial.fitness.end()) -
            serial.fitness.begin());
        REQUIRE(victim != serial.bestIndex);

        int* const genes = perturbed.routes.data() + victim * perturbed.routeLength;
        std::swap(genes[0], genes[1]);
        perturbed.fitness[victim] = problem.routeDistance(
            std::span<const int>(genes, perturbed.routeLength));

        // The whole-population comparison sees it -- on BOTH halves. The
        // fitness assertion is not redundant with the routes one: without it,
        // a routeDistance that happened to be invariant under this particular
        // gene swap would leave `fitness` unshown to discriminate anything,
        // and the cross-thread comparison above leans on both vectors.
        REQUIRE(perturbed.routes != serial.routes);
        REQUIRE(perturbed.fitness != serial.fitness);

        // The best-only comparison does not: the winner has not moved, and it
        // is still the same route with the same distance.
        const std::size_t perturbedBest = bestIndexOf(perturbed);
        REQUIRE(perturbedBest == serial.bestIndex);
        REQUIRE(perturbed.fitness[perturbedBest] == serial.bestDistance);
        REQUIRE(std::equal(serial.bestRoute.begin(), serial.bestRoute.end(),
                           perturbed.routes.begin() +
                               static_cast<std::ptrdiff_t>(perturbedBest *
                                                           perturbed.routeLength)));
    }
}

TEST_CASE("repeated threaded runs are stable", "[determinism]") {
    // Honest about what this can and cannot do. It is the suite's only defence
    // against a timing-dependent fault, because MinGW-w64 UCRT ships no TSan
    // and there is no dynamic race checking available on this toolchain at all.
    // But a race flakes in the PASSING direction: ten clean attempts are ten
    // interleavings that happened not to lose, not a proof that none can. Its
    // real value is cheap repetition of the schedule -- ten independent
    // thread-pool lifetimes, 260 parallelFor epochs -- against a lost wakeup,
    // a missed decrement or a torn write that a single run would step over.
    // (Ten and 260, not eleven and 286: the reference below runs at one
    // thread, and makeExecutor(1) returns a SerialExecutor, which has no
    // workers and runs the body inline -- no pool lifetime and no epoch. Each
    // of the ten THREADED attempts issues 26 parallelFor calls: one for the
    // Population constructor plus one per generation.)
    //
    // The reference is taken at ONE thread, not eight, and that is free: the
    // ten threaded attempts cost the same and sample the same interleavings,
    // but each now also has to match a serial run. With an 8-thread reference
    // this case was blind to every deterministic defect -- chunk-relative
    // seeding makes all eleven runs identically wrong and walks straight
    // through it. The only price is a less specific failure message, and the
    // two cross-thread cases above disambiguate that.
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    const vrp::GaParams params =
        determinismParams(555, vrp::StrategyKind::Generational);

    const Snapshot reference =
        evolveSnapshot(problem, params, vrp::StrategyKind::Generational, 1);
    for (int attempt = 0; attempt < 10; ++attempt) {
        const Snapshot again =
            evolveSnapshot(problem, params, vrp::StrategyKind::Generational, 8);
        INFO("attempt " << attempt);
        REQUIRE(again.routes == reference.routes);
        REQUIRE(again.fitness == reference.fitness);
    }
}
