# VRP-GA: CMake Restructure and Core Library Extraction

- **Date:** 2026-08-28
- **Status:** Approved for planning
- **Scope:** Architectural — new build system, new layout, extracted interfaces

## Context

The repository holds two flat translation units at its root. `main.cpp` is a
sequential genetic algorithm; `main_parallel.cpp` is a hybrid MPI + OpenMP +
Intel TBB variant. There is no build system, and two compiled binaries (`main`,
`main_parallel`) are committed.

The two files duplicate every genetic operator in divergent forms, and the
duplication has drifted: the two `populate()` functions disagree on whether
`numLocations` counts customers or locations, and the two crossovers implement
different operators. The existing README documents a set of correctness defects
in both files and instructs the reader to hand-edit a constant before running.

The root problem is not the missing folders. It is that two orthogonal concerns
— *how the population evolves* and *how work is scheduled* — were fused into two
separate copies of a whole program. Separating them removes the duplication and
is what makes serial-vs-threaded timings comparable.

## Goals

1. A proper CMake build with library/executable/test separation.
2. One dependency-free core library shared by every execution mode.
3. Both evolution strategies preserved and selectable at run time.
4. The documented correctness defects fixed, with tests that keep them fixed.
5. Serial and threaded runs comparable — same binary, same seed, one flag.

## Non-goals

- Dataset file loading or a dataset format. The 20-location instance stays in
  code as `Problem::defaultInstance()`.
- A CLI argument framework. Hand-rolled parsing, roughly 60 lines.
- Multi-vehicle, capacity, or time-window modelling. The problem stays
  single-depot, single-vehicle Euclidean TSP.
- Rewriting git history to purge the committed binaries. They are deleted from
  the working tree; they remain reachable in history.

## Decisions

| Question | Decision |
|---|---|
| Refactor depth | Extract a shared core library and fix the known defects |
| Parallel stack | Drop MPI and TBB; C++ standard library threading only |
| Tests | Catch2 v3 via `FetchContent`, registered with CTest |
| Evolution strategy | Keep both strategies, selectable at run time |
| Mode selection | `--threads` alone decides; `--threads 1` is the serial path |

Dropping MPI and TBB is a deliberate trade. It discards the distributed-memory
and task-scheduler dimensions that the README currently advertises, in exchange
for a project that builds and is testable on the available toolchain (CMake
4.3.2, Ninja, GCC 16.1 MinGW-w64 UCRT — no MPI, no TBB, no vcpkg or Conan).
Correctness that can be demonstrated was judged worth more than parallel
mechanisms that could not be compiled, run, or verified here.

## Repository layout

```text
VRP-GA/
├── CMakeLists.txt
├── CMakePresets.json
├── cmake/
│   └── ProjectWarnings.cmake
├── include/vrp/
│   ├── Config.hpp        Executor.hpp     Operators.hpp
│   ├── Population.hpp    Problem.hpp      Rng.hpp
│   ├── Solver.hpp        Strategy.hpp
├── src/
│   ├── CMakeLists.txt
│   ├── vrp/              # vrp_core implementation
│   └── app/              # Cli, Reporter, main
├── tests/
│   ├── CMakeLists.txt
│   └── test_*.cpp
├── docs/superpowers/specs/
├── README.md
└── .gitignore
```

Public headers live in `include/vrp/` rather than beside their sources, so the
public surface of the library is exactly the file list of one directory.

## Build system

### Targets

| Target | Type | Contents | Links |
|---|---|---|---|
| `vrp_warnings` | INTERFACE | warning flags only | — |
| `vrp_core` | STATIC | model, operators, strategies, executors, solver | `Threads::Threads`; `vrp_warnings` (PRIVATE) |
| `vrp_app` | STATIC | `Cli`, `Reporter` | `vrp_core` |
| `vrp_ga` | EXECUTABLE | `main.cpp` only | `vrp_app` |
| `vrp_tests` | EXECUTABLE | `tests/test_*.cpp` | `vrp_core`, `vrp_app`, `Catch2::Catch2WithMain` |

`vrp_app` exists as its own target so argument parsing is testable without
invoking a subprocess; `vrp_ga` is then a trivial `main`.

### Configuration

- `cmake_minimum_required(VERSION 3.24)` — needed for the `FIND_PACKAGE_ARGS`
  argument to `FetchContent_Declare`.
- `CMAKE_CXX_STANDARD 20`, `CXX_STANDARD_REQUIRED ON`, `CXX_EXTENSIONS OFF`.
  C++20 supplies `std::jthread`, `<barrier>`, and `std::span`.
- Options: `VRP_BUILD_TESTS` (ON when top-level), `VRP_WARNINGS_AS_ERRORS`
  (OFF).
- Catch2 is declared with `FIND_PACKAGE_ARGS` so an installed copy is preferred
  and the download happens only when none is present.
- `CMakePresets.json`: `debug` and `release`, both Ninja.

## Core library design

### Problem

Owns the location list and a precomputed distance matrix, so route evaluation
performs array lookups and additions with no `sqrt` or `pow` in the hot loop.

```cpp
struct Point { int x, y; };

class Problem {
public:
    static Problem defaultInstance();               // the existing 20 coordinates
    explicit Problem(std::vector<Point> locations); // locations[0] is the depot

    std::size_t locationCount() const noexcept;
    std::size_t customerCount() const noexcept;     // locationCount() - 1
    double distance(std::size_t a, std::size_t b) const noexcept;
    double routeDistance(std::span<const int> route) const noexcept;

private:
    std::vector<Point>  locations_;
    std::vector<double> matrix_;                    // n * n, row-major
};
```

`customerCount()` is derived, never configured. This structurally removes the
out-of-bounds defect: no caller can supply a count that disagrees with the
coordinate array.

### Population

Flat storage. A `vector<vector<int>>` of 100 000 routes is 100 000 separate heap
allocations with poor locality; one contiguous buffer replaces it. Routes and
their fitness live in the same object and are updated together, which is what
prevents the stale-fitness defect from recurring.

```cpp
class Population {
public:
    Population(std::size_t size, const Problem&, std::uint64_t seed, Executor&);

    std::size_t size() const noexcept;
    std::size_t routeLength() const noexcept;
    std::span<const int> route(std::size_t i) const noexcept;
    std::span<int>       route(std::size_t i) noexcept;
    double fitness(std::size_t i) const noexcept;

    void setRoute(std::size_t i, std::span<const int>, const Problem&); // re-evaluates
    void evaluateAll(const Problem&, Executor&);
    std::size_t bestIndex() const noexcept;   // ties resolved to lowest index
    std::size_t worstIndex() const noexcept;  // ties resolved to lowest index

private:
    std::size_t         routeLength_;
    std::vector<int>    routes_;   // size * routeLength_
    std::vector<double> fitness_;  // size
};
```

Tie-breaking on the lowest index is required for determinism, not cosmetic.

### Rng

Per-work-item seeding, not per-thread seeding. Each unit of work derives its
seed from `(master_seed, generation, index)`, so offspring *i* of generation *g*
receives the same seed regardless of how work was distributed. A run at
`--threads 8` therefore produces output bit-identical to `--threads 1`.

That property is the keystone of the design: it converts the question of whether
the threaded path is correct from a judgment call into an equality assertion.

Per-item seeding rules out `std::mt19937`, whose 2.5 KB state makes roughly
10^7 constructions unaffordable. `Rng` is therefore a xoshiro256\*\* generator
(four 64-bit words) seeded through splitmix64 — about 30 lines, testable
against published reference vectors.

```cpp
class Rng {
public:
    explicit Rng(std::uint64_t seed) noexcept;
    std::uint64_t next() noexcept;
    std::uint32_t below(std::uint32_t bound) noexcept;  // Lemire, unbiased
    double unit() noexcept;                             // [0, 1)
private:
    std::uint64_t s_[4];
};

std::uint64_t mixSeed(std::uint64_t base, std::uint64_t a, std::uint64_t b) noexcept;
```

`below()` uses the method of Lemire rather than `%`, removing the modulo bias
present throughout the current `rand() % n` calls.

Seed domains are separated so that distinct kinds of work never collide on a
seed. Initial population construction uses `mixSeed(seed, kInitDomain, i)`;
offspring generation uses `mixSeed(seed, generation, i)`. `kInitDomain` is a
constant chosen outside the range of any valid generation index, so building
individual *i* and producing offspring *i* of generation 0 draw different
streams.

### Executor

```cpp
class Executor {
public:
    virtual ~Executor() = default;
    virtual std::size_t threadCount() const noexcept = 0;
    virtual void parallelFor(std::size_t n,
        const std::function<void(std::size_t begin, std::size_t end)>& body) = 0;
};

std::unique_ptr<Executor> makeExecutor(std::size_t threads);  // 1 -> SerialExecutor
```

`makeExecutor` requires `threads >= 1`. The sentinel `--threads 0` is resolved
to `std::thread::hardware_concurrency()` inside `Cli`, so the core library never
sees a zero and carries no "auto" concept.

The body receives a *range*, not a single index, so per-chunk scratch buffers
(notably the crossover `seen` mask) are allocated once per chunk instead of once
per item. `ThreadPoolExecutor` holds persistent `std::jthread` workers rather
than spawning per call. Ranges are contiguous static chunks, so coverage is
deterministic and disjoint.

The executor knows nothing about routes; the strategies contain no threads.

### EvolutionStrategy

```cpp
enum class StrategyKind { SteadyState, Generational };

struct GaParams {
    std::size_t   populationSize = 100000;
    std::size_t   generations    = 100;
    double        mutationRate   = 0.2;
    std::size_t   tournamentSize = 5;
    std::size_t   eliteCount     = 1;
    std::uint64_t seed           = 42;
};

class EvolutionStrategy {
public:
    virtual ~EvolutionStrategy() = default;
    virtual const char* name() const noexcept = 0;
    virtual void step(std::size_t generation, Population&, const Problem&,
                      const GaParams&, Executor&) = 0;
};

std::unique_ptr<EvolutionStrategy> makeStrategy(StrategyKind);
```

- **SteadyStateStrategy** — preserves the semantics of `main.cpp`: two
  tournament parents, one child, replacing the worst individual. Inherently
  serial. Under `--threads > 1` only initial construction and evaluation
  parallelize; the generation loop does not. This is documented rather than
  hidden.
- **GenerationalStrategy** — builds a full offspring population per generation
  via `parallelFor`, carrying `eliteCount` best individuals forward unchanged.
  Uses a double-buffered `Population` and swaps.

### Operators

```cpp
namespace vrp::ops {
std::size_t tournamentSelect(const Population&, std::size_t tournamentSize, Rng&);
void orderCrossover(std::span<const int> p1, std::span<const int> p2,
                    std::span<int> child, std::vector<char>& seenScratch, Rng&);
void swapMutate(std::span<int> route, double rate, Rng&);
}
```

One order crossover replaces the two divergent implementations, adopting the
segment-based operator from `main_parallel.cpp`. The `seenScratch` mask replaces
the per-gene `std::find` scan, taking crossover from O(n^2) to O(n).

### Solver

```cpp
struct RunResult {
    std::vector<int> bestRoute;
    double           bestDistance;
    std::size_t      generationsRun;
    double           elapsedSeconds;
};

using ProgressCallback =
    std::function<void(std::size_t generation, double bestDistance)>;

class Solver {
public:
    Solver(const Problem&, GaParams, std::unique_ptr<EvolutionStrategy>,
           std::unique_ptr<Executor>);
    RunResult run(const ProgressCallback& = {});
};
```

`Solver` owns the generation loop and reports through a callback, so the core
library performs no I/O and stays fully testable.

## Defect inventory

Each defect below is fixed during extraction and pinned by a named test.

### Undefined behaviour — main_parallel.cpp

| # | Defect | Resolution |
|---|---|---|
| 1 | `populate()` shuffles one *shared* `route` and calls `push_back` on a shared `routes` inside `#pragma omp parallel for` | Pre-sized flat storage, per-item `Rng`, index writes |
| 2 | `calcFitness()` calls `push_back` inside an OpenMP loop; racy growth, and arrival order breaks the correspondence between `fitness[i]` and `routes[i]` | Pre-sized vector, index writes |
| 3 | `selectParent()` parallelises a 5-iteration tournament, racing on `best_fitness` and `best_route` | Serial; parallelism belongs at offspring granularity |
| 4 | `rand()` invoked from inside OpenMP regions | Per-item `Rng` |
| 5 | `#define __cdecl` and `#define __stdcall` to nothing at file scope | Removed with MPI |

### Incorrect results

| # | Defect | Resolution |
|---|---|---|
| 6 | `calcRouteDistance()` parallelises a loop carrying `prev` and accumulates without reduction | Serial matrix lookups |
| 7 | `main.cpp` reads `locations[20]`; valid indices are 0–19 | `customerCount()` derived from the coordinate array |
| 8 | Best route selected by indexing the fitness of the *previous* generation into the *new* population | Fitness stored inside `Population` |
| 9 | `MPI_Reduce` sends 20 ints for a 19-customer route; element-wise `MPI_MIN` over permutations is not a permutation; the buffer is never read | Removed with MPI |
| 10 | `gen % 10 == 0` never reports the final generation | Report on interval and always on the last generation |
| 11 | Generational replacement has no elitism, so the best solution can be lost | `eliteCount`, default 1 |

### Quality

| # | Defect | Resolution |
|---|---|---|
| 12 | Per-gene `std::find` makes crossover O(n^2) | `seen` bitmask |
| 13 | `pow(x, 2)` and `sqrt` in the hot path | Precomputed distance matrix |
| 14 | `rand() % n` modulo bias | Lemire bounded generation |
| 15 | `swapMutate` may draw `idx1 == idx2` and silently no-op | Second index drawn from the remaining range |
| 16 | `srand(time(0))` makes runs unreproducible | Explicit `--seed`, default 42 |

## Testing

Catch2 v3, registered through `catch_discover_tests`.

| File | Covers |
|---|---|
| `test_determinism.cpp` | Same seed at 1, 4, and 8 threads yields bit-identical final populations, for both strategies. Same seed twice is identical. Pins defects 1–4, 6, 8. |
| `test_executor.cpp` | `parallelFor` covers `[0, n)` exactly once, with no gaps or overlaps; correct at `n = 0`, `n = 1`, and `n < threadCount` |
| `test_operators.cpp` | Property tests across many seeds: crossover output is always a valid permutation and preserves the segment from parent 1; `mutationRate = 0` never alters a route; tournament indices in range. Pins 12, 15. |
| `test_population.cpp` | Every generated route is a valid permutation of `1..customerCount`; fitness matches independent recomputation; `bestIndex` and `worstIndex` tie-breaking. Pins 7, 8. |
| `test_problem.cpp` | Matrix symmetric, zero diagonal, agrees with a `std::hypot` reference; `defaultInstance()` has 20 locations and 19 customers |
| `test_strategies.cpp` | Best fitness never regresses under elitism; population remains valid across generations. Pins 11. |
| `test_rng.cpp` | xoshiro256\*\* and splitmix64 reference vectors; `below()` range and absence of modulo bias; `unit()` within `[0, 1)`. Pins 14. |
| `test_cli.cpp` | Defaults, rejection of invalid values, `--threads 0` resolving to `hardware_concurrency` |

MinGW-w64 UCRT ships no sanitizer runtimes, so ThreadSanitizer, AddressSanitizer
and UBSan are all unavailable and the project defines no sanitizer presets. No
dynamic analysis is available on this toolchain, which leaves the test suite as
the only evidence for the thread pool: it compensates with repetition rather
than detection, and repetition is not proof.

## Command-line interface

```text
vrp_ga [options]

  --strategy   steady-state | generational   (default: steady-state)
  --population N                             (default: 100000)
  --generations N                            (default: 100)
  --mutation   R                             (default: 0.2)
  --tournament N                             (default: 5)
  --elite      N                             (default: 1, generational only)
  --threads    N                             (default: 1; 0 = hardware_concurrency)
  --seed       N                             (default: 42; 0 = draw from random_device)
  --quiet
  --help
```

Defaults reproduce the original sequential program, so a bare `vrp_ga` serves as
a regression baseline. A fixed default seed makes two consecutive runs
comparable; `--seed 0` restores randomised behaviour.

Invalid input is rejected with a message and a non-zero exit status: unknown
flags, non-numeric values, `--population 0`, `--tournament 0`, mutation outside
`[0, 1]`, and `--elite` greater than or equal to `--population`.

## README

The README requires rewriting rather than patching: MPI, OpenMP, and TBB form
its headline, its requirements section, both build recipes, and much of its
configuration table.

- Rewrite: Highlights, Requirements, Build and run, Configuration, Repository
  structure, the implementation comparison table.
- Remove: the manual `numLocations = 19` correction, the MPI/TBB build recipe,
  and the resolved entries under Current limitations.
- Retain: the honest framing that this is a stochastic heuristic, that
  "Optimal" means "best found", and that the model is single-depot,
  single-vehicle Euclidean TSP without capacities or time windows.
- Add: `--threads` benchmarking guidance, and the note that steady-state does
  not parallelise its generation loop.

## Migration

1. Scaffold CMake, presets, warnings module, `.gitignore` (`build/`).
2. `Rng`, `Problem`, `Population` with tests.
3. `Executor` (serial and thread pool) with tests.
4. `Operators` with property tests.
5. Both strategies plus `Solver` with tests.
6. `Cli`, `Reporter`, `main`; delete `main.cpp` and `main_parallel.cpp`.
7. `git rm` the committed `main` and `main_parallel` binaries.
8. Determinism suite across thread counts.
9. README rewrite.

## Risks

- **Behaviour is not preserved bit-for-bit.** The RNG changes, so results differ
  numerically from the current program even at identical parameters. Route
  *quality* is the invariant under test, not exact output.
- **Steady-state gains little from threading.** Documented in both the CLI help
  and the README so `--threads` is not read as a universal speedup.
- **Reduced scope of parallelism.** Dropping MPI removes distributed-memory
  execution. If it is ever restored, the `Executor` seam is where it attaches,
  but per-item deterministic seeding would need revisiting across ranks.
