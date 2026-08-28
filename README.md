### Genetic Algorithm Route Optimization in C++

A route optimization project that explores evolutionary search through a sequential C++ implementation and an experimental hybrid parallel implementation using **MPI, OpenMP, and Intel TBB**.

The algorithm evolves permutations of customer visits to search for shorter routes that start and end at a depot. The project brings together combinatorial optimization, genetic operators, and parallel programming in a compact codebase.

> **Scope:** The current model uses one vehicle, one depot, and Euclidean distances. It is a Traveling Salesman Problem (TSP) formulation within the broader vehicle-routing domain; vehicle capacities, multiple vehicles, and time windows are not implemented.

## Highlights

- **Permutation-based encoding:** Each candidate route represents an ordering of customer visits.
- **Tournament selection:** Parents are selected by comparing five randomly sampled candidates.
- **Order-preserving crossover:** Offspring combine parent routes while avoiding duplicate customer IDs.
- **Swap mutation:** A configurable probability controls swaps between two route positions.
- **Distance-based fitness:** Routes are evaluated by total travel distance, including the return to the depot.
- **Two evolutionary strategies:** Sequential replacement of one route per iteration and parallel generation of a full offspring population.
- **Progress reporting:** Console output includes best distance, a route, and elapsed execution time.

## Problem Model

Location `0` is the depot. The remaining locations are customers, represented by two-dimensional integer coordinates. A candidate solution visits every customer once and returns to the depot:

```text
Depot → Customer 4 → Customer 1 → … → Customer 7 → Depot
```

The objective is to minimize the sum of Euclidean distances along this tour. Lower fitness values indicate shorter routes. The included dataset contains **20 coordinates: one depot and 19 customers**.

This is a stochastic heuristic: it searches for good routes but does not prove global optimality.

## How the Algorithm Works

1. **Initialize** a population of shuffled customer permutations.
2. **Evaluate** the total distance of each route.
3. **Select** two parents through tournament selection.
4. **Recombine** their customer orderings to create a child route.
5. **Mutate** the child by optionally swapping two positions.
6. **Update** the population and repeat for the configured iterations.

The implementations differ in how they perform recombination and population updates:

| Component | Sequential: `main.cpp` | Parallel prototype: `main_parallel.cpp` |
| --- | --- | --- |
| Population | One population | Population partitioned across MPI ranks |
| Crossover | Copy a prefix from one parent, then append unseen customers from the other | Copy a segment from one parent, then fill empty positions using the other |
| Replacement | Replace the worst route with one child per iteration | Replace each local population with a full set of offspring |
| Parallel mechanisms | None | TBB offspring tasks, OpenMP loops, and MPI reductions |
| Progress output | Every iteration | Every 10 iterations, from rank 0 |

The parallel source is an exploration of combining process-level and thread-level parallelism. It requires correctness fixes before its output or timing can be used for evaluation; see [Current limitations](#current-limitations).

## Getting Started

### Requirements

- A C++ compiler supporting C++11 or later; the commands below use C++17.
- For the parallel prototype: an MPI development environment, OpenMP support, and Intel TBB/oneTBB.

### Clone

```bash
git clone https://github.com/gurelx/VRP-GA.git
cd VRP-GA
```

### Required correction before running the sequential version

In `main.cpp`, change:

```cpp
int numLocations = 20;
```

to:

```cpp
int numLocations = 19;
```

The sequential `populate()` function interprets this value as the number of **customers**, generating IDs from `1` through `numLocations`. The supplied coordinate array has valid indices `0` through `19`, so the original value generates an out-of-bounds access to location `20`.

**Do not apply the same change to the parallel version:** its `populate()` function interprets `numLocations` as the total number of locations, including the depot.

### Build and run the sequential version

On Linux or macOS with a compatible compiler:

```bash
g++ -std=c++17 -O2 main.cpp -o vrp_ga
./vrp_ga
```

On Windows with MinGW-w64 available in PowerShell:

```powershell
g++ -std=c++17 -O2 main.cpp -o vrp_ga.exe
.\vrp_ga.exe
```

Build from source instead of relying on the platform-specific binaries included in the repository.

### Parallel prototype build

For a Linux environment with the required libraries installed, the intended build and launch commands are:

```bash
mpic++ -std=c++17 -O2 -fopenmp main_parallel.cpp -ltbb -o vrp_ga_parallel
mpirun -np 4 ./vrp_ga_parallel
```

These commands are environment-dependent and have not been validated here. The parallel implementation has known correctness issues and should not be treated as a validated solver.

## Configuration

Parameters are edited directly in each source file; there is currently no command-line configuration or dataset loader.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `routesSize` / `total_population_size` | `100000` | Sequential population size / requested total parallel population |
| `generations` | `100` | Number of evolutionary iterations |
| `mutationRate` | `0.2` | Probability of attempting a swap for each child |
| `tournamentSize` | `5` | Number of candidate draws per parent selection |
| `numLocations` | `20` in both sources | Different semantics between versions; see the required sequential correction above |

To use a different dataset, edit the `locations` vector in `main()` and keep the location count consistent with that implementation. The first coordinate must remain the depot. Both files also contain an optional random-location helper, but the active examples use fixed coordinates.

For the parallel prototype, the local population size is computed using integer division. Choose a rank count that divides the total population evenly and leaves each rank with a nonempty population.

## Reading the Output

The programs print progress, followed by a route, its reported distance, and elapsed time. Customer numbers refer to indices in the `locations` vector, and distance is expressed in the coordinate system's units.

The console labels use **“Optimal”**, but they should be read as **“best found”**, not as proof of an optimal solution. Results vary because the search is randomized. Parallel route and distance reporting also needs correction as described below.

## Current Limitations

- **Sequential indexing:** The original configuration includes a customer ID outside the coordinate array. Apply the correction above before running.
- **Parallel shared state:** Several OpenMP loops update shared vectors, random-number state, distance accumulators, or selection state without appropriate synchronization. These operations require redesign before parallel results are reliable.
- **MPI route handling:** A route has 19 customer IDs, but the route reduction sends 20 integers. Element-wise `MPI_MIN` also does not select a valid winning tour, and the resulting buffer is not used to migrate routes.
- **Best-route reporting:** The parallel version associates fitness from the previous population with newly generated routes. It does not reliably transfer the winning route to rank 0, and its last reporting checkpoint precedes the final iteration.
- **Benchmark comparability:** The sequential version creates one child per iteration, while the parallel version creates an entire offspring population. Equal iteration counts therefore do not represent equivalent workloads. No validated speedup or solution-quality benchmark is included.

## Repository Structure

```text
VRP-GA/
├── main.cpp             # Sequential genetic algorithm
├── main_parallel.cpp    # Experimental MPI/OpenMP/TBB implementation
├── main                 # Included compiled binary
├── main_parallel        # Included compiled binary
├── .vscode/
│   └── settings.json    # Editor configuration
├── .gitignore
└── README.md
