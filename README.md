# Genetic Algorithm Route Optimization in C++

A C++20 genetic algorithm that searches for short delivery routes, built as a
CMake project with a `std::thread` execution backend and a Catch2 test suite.

The algorithm evolves permutations of customer visits to find shorter routes
that start and end at a depot. Results are reproducible: the same `--seed`
produces the same route regardless of how many threads the run used.

> **Scope:** the model uses one vehicle, one depot, and Euclidean distances. It
> is a Traveling Salesman Problem (TSP) formulation within the broader
> vehicle-routing domain. Vehicle capacities, multiple vehicles, and time
> windows are **not** implemented.

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

On Linux and macOS the executable is at `build/release/src/vrp_ga`; the CMake
commands above are identical on every platform.

**On Windows, run the binaries from a shell whose `PATH` has the compiler's
`bin` directory first.** Both `vrp_ga.exe` and `vrp_tests.exe` link
`libstdc++-6.dll`, `libgcc_s_seh-1.dll` and `libwinpthread-1.dll` dynamically.
`ctest` is insulated — `tests/CMakeLists.txt` passes the compiler directory to
`catch_discover_tests` via `DL_PATHS` — but every **direct** invocation in this
README depends on the ambient `PATH`, and a shell that ships its own copies of
those DLLs (Git Bash and MSYS shells do) will load the wrong ones. PowerShell
with the toolchain on `PATH` is the configuration everything here was verified
in.

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

Invalid input is rejected before any work starts, with a message naming the
flag, and exits with status **2**:

```powershell
.\build\release\src\vrp_ga.exe --bogus
```

The message, a blank line, and then the full usage text all go to **stderr**:

```text
error: unknown option: --bogus

vrp_ga [options]
...
```

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

## Benchmarking

Vary `--threads` with everything else fixed, and use `--quiet` — console I/O
happens inside the timed region and will otherwise dominate short runs.

Because `--threads` only reaches the `generational` strategy, benchmark that
one. Steady-state's generation loop never touches the executor, so its runtime
is flat in the thread count (measured: 0.00233 s at 1 thread, 0.00220 s at 8 —
the difference is noise, and the only threaded work in it is the one-off
construction of the initial population).

```powershell
.\build\release\src\vrp_ga.exe --strategy generational --population 20000 --generations 100 --threads 1 --quiet
.\build\release\src\vrp_ga.exe --strategy generational --population 20000 --generations 100 --threads 8 --quiet
```

```text
Strategy:       generational
Threads:        1
Seed:           42

Best route found: depot -> 1 -> 17 -> 5 -> 16 -> 15 -> 18 -> 2 -> 6 -> 19 -> 14 -> 7 -> 13 -> 12 -> 11 -> 10 -> 8 -> 9 -> 3 -> 4 -> depot
Best distance:  52.8269
Generations:    100
Elapsed:        0.269228 s

Strategy:       generational
Threads:        8
Seed:           42

Best route found: depot -> 1 -> 17 -> 5 -> 16 -> 15 -> 18 -> 2 -> 6 -> 19 -> 14 -> 7 -> 13 -> 12 -> 11 -> 10 -> 8 -> 9 -> 3 -> 4 -> depot
Best distance:  52.8269
Generations:    100
Elapsed:        0.0625685 s
```

**Identical route, identical distance, 4.3× less time.** That is the intended
and tested behaviour, not a coincidence of this seed.

A sweep over the same configuration. These are **separate runs** from the pair
above, so the 8-thread ratio here is 3.97× rather than that pair's 4.30×; the
difference is run-to-run variance, not a change in configuration:

| `--threads` | Best distance | Elapsed | Speedup |
| --- | --- | --- | --- |
| 1 | 52.8269 | 0.2606 s | 1.00× |
| 2 | 52.8269 | 0.1634 s | 1.60× |
| 4 | 52.8269 | 0.1013 s | 2.57× |
| 8 | 52.8269 | 0.0657 s | 3.97× |
| 16 | 52.8269 | 0.0438 s | 5.95× |

Measured on an AMD Ryzen 9 9900X (12 cores, 24 threads) with GCC 16.1
(MinGW-w64 UCRT), one run per row; times vary a few percent between runs, and
the distances are exact across all of them.

The 16-thread row exceeds the 12 physical cores, so part of its 5.95× comes from
simultaneous multithreading rather than additional cores — expect the returns
past 12 threads to be smaller than the row alone suggests.

## Determinism, and what the tests actually establish

Every work item is seeded from `mixSeed(seed, generation, item)` rather than
from a shared generator, so the value each individual receives depends on *its
own index*, not on which thread or chunk processed it. The chunk boundaries
therefore drop out of the result.

Being precise about the evidence, because these are three different claims:

- **Threading does not change the result.** This is directly tested. The
  determinism suite compares whole populations — all 512 individuals, routes
  and fitness — across 1, 2, 4 and 8 threads, plus the entire per-generation
  progress trace, so a divergence at an intermediate generation that later
  washes out of the winner is still caught. Comparisons are exact `==` on
  doubles by design; a tolerance would hide what they exist to find.

- **Conformance to the specified seeding is a separate claim, established
  elsewhere.** The determinism suite is *differential*: it runs the same
  library code twice and compares. An implementation that is wrong but
  consistent across thread counts passes all of it. What rules that out are the
  per-slot oracles in `tests/test_population.cpp` and `tests/test_strategies.cpp`
  — but they are independent to different degrees, and the difference matters:

  - `test_population.cpp` re-implements the initial shuffle outright. It
    hand-rolls the backward Fisher-Yates loop and calls only `mixSeed` and
    `Rng`, never `Population`'s own shuffle, so it genuinely pins the domain
    tag, the item index and the shuffle direction against an independent
    computation.
  - `test_strategies.cpp`'s `expectedOffspring` is independent of
    `Strategy.cpp` but calls the library's own `tournamentSelect`,
    `orderCrossover` and `swapMutate`. It therefore pins the *composition* —
    the per-slot seed, the operator order, which parents feed the crossover —
    and not the operators' internals, which the `[operators]` tests pin
    separately against mirror generators.

  Do not read a passing determinism suite as evidence that the seeding is the
  specified one.

- **The thread pool's freedom from data races rests on the test suite alone.**
  MinGW-w64 UCRT ships no sanitizer runtimes, so ThreadSanitizer,
  AddressSanitizer and UBSan are all unavailable on this toolchain — there is
  no dynamic analysis available here at all, and the project therefore defines
  no sanitizer presets. The suite compensates with repetition rather than
  detection (ten independent pool lifetimes, 260 `parallelFor` epochs, each
  required to match a serial reference), but repetition is not proof: a race
  flakes in the *passing* direction, and ten clean runs are ten interleavings
  that happened not to lose.

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

### Windows path length

On Windows, build this project from a **short source path**. Catch2's build tree
nests deeply (`build/release/_deps/catch2-build/src/CMakeFiles/Catch2.dir/…`),
and a long source path pushes generated dependency files past the 260-character
`MAX_PATH` limit. The failure is unmistakable once you know it:

```text
fatal error: opening dependency file _deps\catch2-build\src\CMakeFiles\...\*.cpp.obj.d:
No such file or directory
```

**Move the checkout closer to the drive root.** That is the remedy, and it is
the one that was tested: the same clone that failed from a 144-character path
built and passed all 134 tests from `C:\vrpchk`.

Two things that sound like fixes and are not — one ruled out by construction,
one tested here:

- **Enabling long paths in Windows and Git does not help** (ruled out by
  construction, not executed). The registry's `LongPathsEnabled` lifts the limit
  only for processes whose manifest marks them `longPathAware`, and the
  MinGW-w64 GCC binaries are not manifested that way, so the setting never
  reaches the compiler writing the file. The `\\?\` prefix is a *different*
  route past `MAX_PATH` — a Win32 API convention requiring an absolute path —
  and the path in the error is relative
  (`_deps\catch2-build\src\CMakeFiles\…`), so it is not in play either.
- **`CMAKE_OBJECT_PATH_MAX` only warns** (tested here). Set to 240 — both in
  `tests/CMakeLists.txt` before `FetchContent_MakeAvailable(Catch2)` and as a
  `-D` cache entry — CMake reports "The object file directory … is too long"
  for the Catch2 target and then builds with the long path anyway. The build
  fails identically, with the same five errors.

## Repository structure

```text
VRP-GA/
├── CMakeLists.txt              # Top-level project, options, version
├── CMakePresets.json           # debug and release presets
├── cmake/
│   └── ProjectWarnings.cmake   # The vrp_warnings interface target
├── include/vrp/                # Public headers for the vrp_core library
│   ├── Config.hpp              # GaParams, StrategyKind
│   ├── Executor.hpp            # Executor interface, thread pool, chunking
│   ├── Operators.hpp           # Selection, crossover, mutation
│   ├── Population.hpp          # Routes plus their fitness, kept in step
│   ├── Problem.hpp             # Locations and the distance matrix
│   ├── Rng.hpp                 # xoshiro256**, mixSeed
│   ├── Solver.hpp              # Generation loop, RunResult
│   ├── Strategy.hpp            # EvolutionStrategy interface
│   └── Version.hpp             # Project identity
├── src/
│   ├── CMakeLists.txt          # vrp_core, vrp_app, vrp_ga targets
│   ├── vrp/                    # vrp_core implementation — performs no I/O
│   └── app/                    # CLI, reporter, and main()
├── tests/                      # Catch2 suite, one file per component
│   ├── CMakeLists.txt          # Fetches Catch2, registers tests
│   └── SolverTrace.hpp         # Shared oracle for solver and determinism
├── docs/                       # Design notes and the restructure plan
└── README.md
```

The library (`vrp_core`) performs no I/O; all printing lives in `vrp_app`, which
is a separate static library rather than part of `main.cpp` so that argument
parsing and output formatting are testable without spawning a subprocess.

## Not implemented

The following are outside the current model, not defects:

- **Multiple vehicles and capacities** — one vehicle, unlimited capacity.
- **Time windows** — no temporal constraints on visits.
- **Dataset loading** — the 20 coordinates are compiled in; there is no file
  format or `--input` flag. Changing the instance means editing
  `Problem::defaultInstance()` in `src/vrp/Problem.cpp`.
- **A parallel steady-state strategy** — steady-state is inherently serial, and
  its generation loop ignores `--threads` entirely.
- **Local search** — no 2-opt or similar improvement operator runs on offspring.
