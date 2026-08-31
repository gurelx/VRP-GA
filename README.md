# Genetic Algorithm Route Optimization in C++

A C++20 genetic algorithm that searches for short delivery routes, built as a
CMake project with a `std::thread` execution backend and a Catch2 test suite.

The algorithm evolves permutations of customer visits to find shorter routes
that start and end at a depot. Results are reproducible: the same `--seed`
produces the same route regardless of how many threads the run used.

> **Scope:** the model uses one vehicle, one depot, and Euclidean distances. It
> is a Traveling Salesman Problem (TSP) formulation within the broader
> vehicle-routing domain. 

## Highlights

- **Permutation encoding** — each candidate route is an ordering of customer visits.
- **Tournament selection** — parents win a comparison among `--tournament` randomly drawn candidates.
- **Order crossover (OX)** — offspring combine parent routes without duplicating customers.
- **Swap mutation** — `--mutation` controls the probability of swapping two positions.
- **Two selectable strategies** — steady-state and generational, chosen at runtime.
- **Thread-count-independent results** — per-item RNG seeding means `--threads` changes the runtime, never the answer.
- **Precomputed distance matrix** — route evaluation is lookups and additions, with no `sqrt` in the hot loop.

## Problem model

Location `0` is the depot; the rest are customers with integer 2-D coordinates.
A candidate solution visits every customer exactly once and returns to the
depot:

```text
Depot → Customer 4 → Customer 1 → … → Customer 7 → Depot
```

The objective is to minimise the total Euclidean distance of the tour. Lower
fitness is better. The built-in dataset holds **20 coordinates: one depot and 19
customers**, and the customer count is derived from the coordinate list rather
than configured separately, so the two cannot disagree.

This is a **stochastic heuristic**. It searches for good routes; it does not
prove global optimality.

## How the algorithm works

1. **Initialise** a population of shuffled customer permutations.
2. **Evaluate** each route's total distance.
3. **Select** parents by tournament selection.
4. **Recombine** the parents with order crossover to produce a child.
5. **Mutate** the child with a probabilistic position swap.
6. **Replace** part of the population and repeat.

Step 6 is what separates the two strategies:

| | `steady-state` (default) | `generational` |
| --- | --- | --- |
| Offspring per generation | One | One per non-elite individual |
| Replacement | Overwrites the worst individual | Replaces the whole population |
| Elitism | Implicit — the best is never overwritten | Explicit — `--elite` best carried forward |
| Uses `--threads` | **No.** Its generation loop is serial. | Yes — offspring are bred in parallel |
| Cost per generation | Negligible | Proportional to population size |

## Getting started

### Requirements

- **CMake 3.24 or newer**
- A **C++20** compiler
- **Ninja** (the supplied presets specify it)
- A network connection on first configure — **Catch2 v3.7.1 is fetched
  automatically** by `FetchContent`, or reused if already installed

No MPI, no OpenMP, and no TBB. The only concurrency dependency is the standard
library's threading support, found via CMake's `Threads` package.

### Build and test

```powershell
cmake --preset release
cmake --build --preset release
ctest --preset release
```

Two presets exist, `debug` and `release`. Both build the test suite and both
treat compiler warnings as errors:

| Preset | Build type | Binary directory | Warnings as errors |
| --- | --- | --- | --- |
| `debug` | `Debug` — asserts active | `build/debug/` | Yes |
| `release` | `Release` — `-O3 -DNDEBUG` | `build/release/` | Yes |

Warnings-as-errors is enabled in **both** deliberately. The bounds checks
throughout the library are `assert`s, which `Release` compiles out, so the two
configurations do not compile the same code: `Debug` sees the assertion bodies,
`Release` sees the assertion-free ones the optimiser then works on. A diagnostic
that only one configuration emits is therefore not real coverage.

### Run

```powershell
.\build\release\src\vrp_ga.exe --help
```

```text
vrp_ga [options]

  --strategy    steady-state | generational  (default: steady-state)
  --population  N    population size         (default: 100000)
  --generations N    iterations              (default: 100)
  --mutation    R    swap probability, 0..1  (default: 0.2)
  --tournament  N    candidates per parent   (default: 5)
  --elite       N    survivors, generational (default: 1)
  --threads     N    0 selects all cores     (default: 1)
  --seed        N    0 draws a random seed   (default: 42)
  --quiet            suppress per-generation output
  --help, -h         show this message

Note: steady-state replaces one individual per generation, so its
generation loop does not parallelise. Use --strategy generational to
see a speedup from --threads.
```

## Command-line options

| Flag | Value | Default | Meaning |
| --- | --- | --- | --- |
| `--strategy` | `steady-state` \| `generational` | `steady-state` | Replacement scheme; see the table above. |
| `--population` | `N ≥ 1` | `100000` | Number of individuals. Rejected at 0. |
| `--generations` | `N ≥ 0` | `100` | Iterations to run. 0 reports the initial population's best. |
| `--mutation` | `R` in `[0, 1]` | `0.2` | Probability of swapping two positions in a child. |
| `--tournament` | `N ≥ 1` | `5` | Candidates drawn per parent selection. Rejected at 0. |
| `--elite` | `N` | `1` | Best individuals carried forward unchanged. Only *used* by `generational`, but the `--elite < --population` guard is enforced for **both** strategies — so `--strategy steady-state --population 1` is rejected on the default `--elite 1`, a flag you never typed. The error names both values for that reason. |
| `--threads` | `N` | `1` | Worker threads. `0` selects all cores. Affects `generational` only. |
| `--seed` | `N` | `42` | RNG seed. `0` draws a random one and prints it. |
| `--quiet` | — | off | Suppress per-generation progress. Use this when timing. |
| `--help`, `-h` | — | — | Print the message above and exit 0. |

## Reading the output

A bare run uses every default:

```powershell
.\build\release\src\vrp_ga.exe
```

```text
Strategy:       steady-state
Threads:        1
Seed:           42
Generation 10: best distance = 82.2603
Generation 20: best distance = 82.2603
Generation 30: best distance = 82.2603
...
Generation 100: best distance = 82.2603

Best route found: depot -> 1 -> 4 -> 6 -> 2 -> 3 -> 10 -> 19 -> 17 -> 5 -> 18 -> 15 -> 16 -> 14 -> 7 -> 13 -> 12 -> 9 -> 11 -> 8 -> depot
Best distance:  82.2603
Generations:    100
Elapsed:        0.0167221 s
```

Route numbers are indices into the coordinate list, and distance is in the
coordinate system's own units. `Elapsed` covers the whole of the run: building
the initial population, every generation, and the progress callback itself.

**The label says "best route found", and that is exactly what it means.** It is
the best route this run happened to reach, not a proven optimum. Different
seeds give different answers.

**The defaults barely search.** As the output above shows, the best distance
never improves. Steady-state produces one child per generation, so 100
generations replace 100 of 100000 individuals — 0.1% of the population. The
defaults are inherited from the original program and preserved for continuity;
they are not a good configuration. For an actual search, use `generational`, or
give steady-state a generation count on the order of its population size.

## Testing

```powershell
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

134 tests across 11 groups. Both presets run the same suite; `debug` exercises
the `assert` preconditions, `release` exercises the optimised code paths.

`ctest -R` filters on **test names**, which are the Catch2 `TEST_CASE`
descriptions — not on tags:

```powershell
ctest --preset debug -R "seed"
```

To select by tag, run the test binary directly:

```powershell
.\build\debug\tests\vrp_tests.exe "[determinism]"
.\build\debug\tests\vrp_tests.exe --list-tags
```

| Tag | Cases | Covers |
| --- | --- | --- |
| `[rng]` | 8 | xoshiro256\*\*, `splitmix64`, `mixSeed` domain separation |
| `[problem]` | 9 | Distance matrix, route evaluation |
| `[executor]` | 10 | Chunking, the thread pool, serial fallback |
| `[population]` | 19 | Seeded initialisation, per-slot seeding oracle |
| `[operators]` | 27 | Selection, crossover, mutation, RNG draw counts |
| `[strategy]` | 16 | Both strategies, per-slot oracles, executor granularity |
| `[solver]` | 12 | Generation loop, progress reporting, timing |
| `[determinism]` | 5 | Cross-thread agreement at the `Solver` layer |
| `[cli]` | 19 | Argument parsing, validation, sentinels |
| `[reporter]` | 8 | Output formatting and intervals |
| `[smoke]` | 1 | Project identity wired through from CMake |

Tests time out at 60 seconds each, so a wedged thread pool fails by name
instead of hanging until CTest's 1500-second default.

## Out of Scope

The following are outside the current model, not defects:

- **Multiple vehicles and capacities** — one vehicle, unlimited capacity.
- **Time windows** — no temporal constraints on visits.
- **Dataset loading** — the 20 coordinates are compiled in; there is no file
  format or `--input` flag. Changing the instance means editing
  `Problem::defaultInstance()` in `src/vrp/Problem.cpp`.
- **A parallel steady-state strategy** — steady-state is inherently serial, and
  its generation loop ignores `--threads` entirely.
- **Local search** — no 2-opt or similar improvement operator runs on offspring.
