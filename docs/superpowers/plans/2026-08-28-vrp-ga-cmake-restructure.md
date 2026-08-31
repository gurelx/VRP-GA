# VRP-GA CMake Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace two flat, defect-ridden `.cpp` files with a CMake project built around one dependency-free core library, where evolution strategy and work scheduling are independent axes selected at run time.

**Architecture:** A static library `vrp_core` owns the problem model, population, genetic operators, two evolution strategies, and two executors (serial and thread pool). Per-work-item RNG seeding makes results bit-identical regardless of thread count, which is how the threaded path is proven correct. A thin `vrp_app` library holds argument parsing and reporting; `vrp_ga` is a trivial `main`.

**Tech Stack:** C++20, CMake 3.24+, Ninja, Catch2 v3 (via FetchContent), standard-library threading only. No MPI, no TBB, no OpenMP.

**Spec:** `docs/superpowers/specs/2026-08-28-vrp-ga-cmake-restructure-design.md`

## Global Constraints

- **Never commit without user verification.** Every task ends by staging changes and showing `git diff --cached`, then stopping. The user says when to commit. This overrides the commit steps in any skill.
- `cmake_minimum_required(VERSION 3.24)`. Available locally: CMake 4.3.2, Ninja, GCC 16.1 MinGW-w64 UCRT.
- C++20 exactly: `CMAKE_CXX_STANDARD 20`, `CXX_STANDARD_REQUIRED ON`, `CXX_EXTENSIONS OFF`.
- `vrp_core` links only `Threads::Threads`. No other dependency may enter it. It performs **no I/O** — no `<iostream>`, no printing.
- Namespace is `vrp` throughout; operators live in `vrp::ops`.
- Public headers go in `include/vrp/`, implementation in `src/vrp/`, app code in `src/app/`, tests in `tests/`.
- Genes are customer indices `1..customerCount`. Index `0` is the depot and never appears in a route.
- Every RNG draw derives from `mixSeed(seed, domain, index)`. No thread-local or per-thread seeding anywhere — that would break cross-thread-count reproducibility.
- Catch2 pinned to `v3.7.1`, declared with `FIND_PACKAGE_ARGS` so an installed copy wins.
- **Sanitizers:** the `asan-ubsan` and `tsan` presets are for Linux/CI. GCC on MinGW-w64 ships no TSan and unreliable ASan; do not treat their failure to configure locally as a task failure. On Windows the determinism suite (Task 9) is the required evidence.

**Build commands used throughout:**

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug --output-on-failure
```

Single test by name: `ctest --preset debug -R "<name>" --output-on-failure`
Single test by Catch2 tag: `./build/debug/tests/vrp_tests "[tag]"` (add `.exe` on Windows).

---

## File Structure

| File | Responsibility |
|---|---|
| `CMakeLists.txt` | Project, options, `find_package(Threads)`, subdirectories |
| `cmake/ProjectWarnings.cmake` | `vrp_warnings` INTERFACE target; sanitizer flags |
| `CMakePresets.json` | `debug`, `release`, `asan-ubsan`, `tsan` |
| `include/vrp/Rng.hpp` | xoshiro256\*\*, splitmix64, `mixSeed`, seed domains |
| `include/vrp/Problem.hpp` | `Point`, locations, distance matrix, `routeDistance` |
| `include/vrp/Executor.hpp` | `Executor` interface, `makeExecutor`, `chunkRange` |
| `include/vrp/Population.hpp` | Flat route storage paired with fitness |
| `include/vrp/Operators.hpp` | `tournamentSelect`, `orderCrossover`, `swapMutate` |
| `include/vrp/Config.hpp` | `GaParams`, `StrategyKind` |
| `include/vrp/Strategy.hpp` | `EvolutionStrategy`, `makeStrategy` |
| `include/vrp/Solver.hpp` | `RunResult`, `ProgressCallback`, generation loop |
| `src/app/Cli.hpp/.cpp` | `Options`, `parseArgs`, `usage` |
| `src/app/Reporter.hpp/.cpp` | Progress and result printing |
| `src/app/main.cpp` | Wire-up only |

---

### Task 1: CMake scaffold, presets, and a green test run

**Files:**
- Create: `CMakeLists.txt`, `cmake/ProjectWarnings.cmake`, `CMakePresets.json`
- Create: `src/CMakeLists.txt`, `tests/CMakeLists.txt`, `tests/test_smoke.cpp`
- Create: `include/vrp/Version.hpp`, `src/vrp/Version.cpp`
- Modify: `.gitignore`
- Delete: `main`, `main_parallel` (committed binaries)

**Interfaces:**
- Consumes: nothing.
- Produces: targets `vrp_warnings`, `vrp_core`, `vrp_tests`; the preset names `debug`/`release`; `vrp::projectName()` returning `std::string_view`.

`vrp_core` needs at least one source file to be a valid STATIC library, so `Version.cpp` exists as a placeholder member that later tasks add to. It is real code, not scaffolding to delete.

- [ ] **Step 1: Delete the committed binaries and ignore build output**

```bash
git rm --cached main main_parallel
rm -f main main_parallel
printf '\n# Build output\nbuild/\n\n# CMake\nCMakeUserPresets.json\n' >> .gitignore
```

- [ ] **Step 2: Write `cmake/ProjectWarnings.cmake`**

```cmake
add_library(vrp_warnings INTERFACE)

if(MSVC)
  target_compile_options(vrp_warnings INTERFACE /W4 /permissive-)
  if(VRP_WARNINGS_AS_ERRORS)
    target_compile_options(vrp_warnings INTERFACE /WX)
  endif()
else()
  target_compile_options(vrp_warnings INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
    -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wunused
    -Woverloaded-virtual -Wdouble-promotion)
  if(VRP_WARNINGS_AS_ERRORS)
    target_compile_options(vrp_warnings INTERFACE -Werror)
  endif()
endif()

# Sanitizers are Linux/CI oriented. GCC on MinGW-w64 has no TSan and
# unreliable ASan; do not expect these presets to configure on Windows.
if(VRP_ENABLE_SANITIZERS AND NOT MSVC)
  target_compile_options(vrp_warnings INTERFACE ${VRP_SANITIZER_FLAGS})
  target_link_options(vrp_warnings INTERFACE ${VRP_SANITIZER_FLAGS})
endif()
```

- [ ] **Step 3: Write the root `CMakeLists.txt`**

Options are declared before `include(ProjectWarnings)` because the module reads them.

```cmake
cmake_minimum_required(VERSION 3.24)
project(vrp_ga VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(VRP_BUILD_TESTS "Build the test suite" ${PROJECT_IS_TOP_LEVEL})
option(VRP_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(VRP_ENABLE_SANITIZERS "Enable sanitizer flags from VRP_SANITIZER_FLAGS" OFF)
set(VRP_SANITIZER_FLAGS "-fsanitize=address,undefined;-fno-omit-frame-pointer"
    CACHE STRING "Sanitizer flags used when VRP_ENABLE_SANITIZERS is ON")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ProjectWarnings)

find_package(Threads REQUIRED)

add_subdirectory(src)

if(VRP_BUILD_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif()
```

- [ ] **Step 4: Write `src/CMakeLists.txt`**

`vrp_app` and `vrp_ga` are added in Task 10; this file grows.

```cmake
add_library(vrp_core STATIC
  vrp/Version.cpp
)
target_include_directories(vrp_core PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(vrp_core PUBLIC Threads::Threads)
target_link_libraries(vrp_core PRIVATE vrp_warnings)
```

- [ ] **Step 5: Write `include/vrp/Version.hpp` and `src/vrp/Version.cpp`**

```cpp
// include/vrp/Version.hpp
#pragma once
#include <string_view>

namespace vrp {
std::string_view projectName() noexcept;
std::string_view projectVersion() noexcept;
}  // namespace vrp
```

```cpp
// src/vrp/Version.cpp
#include "vrp/Version.hpp"

namespace vrp {
std::string_view projectName() noexcept { return "vrp_ga"; }
std::string_view projectVersion() noexcept { return "1.0.0"; }
}  // namespace vrp
```

- [ ] **Step 6: Write `tests/CMakeLists.txt`**

```cmake
include(FetchContent)
FetchContent_Declare(Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.7.1
  FIND_PACKAGE_ARGS 3 NAMES Catch2)
FetchContent_MakeAvailable(Catch2)

list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
include(Catch)

add_executable(vrp_tests
  test_smoke.cpp
)
target_link_libraries(vrp_tests PRIVATE vrp_core Catch2::Catch2WithMain)
target_link_libraries(vrp_tests PRIVATE vrp_warnings)

catch_discover_tests(vrp_tests)
```

- [ ] **Step 7: Write `tests/test_smoke.cpp`**

```cpp
#include <catch2/catch_test_macros.hpp>
#include "vrp/Version.hpp"

TEST_CASE("project identity is reported", "[smoke]") {
    REQUIRE(vrp::projectName() == "vrp_ga");
    REQUIRE(vrp::projectVersion() == "1.0.0");
}
```

- [ ] **Step 8: Write `CMakePresets.json`**

```json
{
  "version": 6,
  "configurePresets": [
    {
      "name": "debug",
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build/debug",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug",
        "VRP_BUILD_TESTS": "ON",
        "VRP_WARNINGS_AS_ERRORS": "ON"
      }
    },
    {
      "name": "release",
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build/release",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release",
        "VRP_BUILD_TESTS": "ON"
      }
    },
    {
      "name": "asan-ubsan",
      "inherits": "debug",
      "binaryDir": "${sourceDir}/build/asan",
      "cacheVariables": {
        "VRP_ENABLE_SANITIZERS": "ON",
        "VRP_SANITIZER_FLAGS": "-fsanitize=address,undefined;-fno-omit-frame-pointer"
      }
    },
    {
      "name": "tsan",
      "inherits": "debug",
      "binaryDir": "${sourceDir}/build/tsan",
      "cacheVariables": {
        "VRP_ENABLE_SANITIZERS": "ON",
        "VRP_SANITIZER_FLAGS": "-fsanitize=thread;-fno-omit-frame-pointer"
      }
    }
  ],
  "buildPresets": [
    { "name": "debug", "configurePreset": "debug" },
    { "name": "release", "configurePreset": "release" },
    { "name": "asan-ubsan", "configurePreset": "asan-ubsan" },
    { "name": "tsan", "configurePreset": "tsan" }
  ],
  "testPresets": [
    {
      "name": "debug",
      "configurePreset": "debug",
      "output": { "outputOnFailure": true }
    },
    {
      "name": "release",
      "configurePreset": "release",
      "output": { "outputOnFailure": true }
    }
  ]
}
```

- [ ] **Step 9: Configure, build, and run**

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug --output-on-failure
```

Expected: configure succeeds (Catch2 downloads on first run), build succeeds with zero warnings, one test passes.

- [ ] **Step 10: Stage and request verification**

```bash
git add -A
git diff --cached --stat
```

Show the user. Do not commit until they approve.

---

### Task 2: Rng — xoshiro256\*\* with per-item seeding

**Files:**
- Create: `include/vrp/Rng.hpp`, `src/vrp/Rng.cpp`
- Create: `tests/test_rng.cpp`
- Modify: `src/CMakeLists.txt`, `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `vrp::Rng`, constructed as `Rng(std::uint64_t seed)`; `Rng::fromState(std::array<std::uint64_t,4>)`
  - `std::uint64_t Rng::next() noexcept`
  - `std::uint32_t Rng::below(std::uint32_t bound) noexcept` — unbiased, `bound >= 1`
  - `double Rng::unit() noexcept` — `[0, 1)`
  - `std::uint64_t vrp::splitmix64(std::uint64_t& state) noexcept`
  - `std::uint64_t vrp::mixSeed(std::uint64_t base, std::uint64_t a, std::uint64_t b) noexcept`
  - `constexpr std::uint64_t vrp::kInitDomain`

> **Note on golden vectors:** the reference values below were derived by hand from the xoshiro256\*\* and splitmix64 reference algorithms. If a golden assertion fails, verify the implementation against the published reference *before* editing the expected value — a correct RNG must not be "fixed" to match a wrong constant.

- [ ] **Step 1: Write the failing test**

`tests/test_rng.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <array>
#include <cstdint>
#include <vector>
#include "vrp/Rng.hpp"

TEST_CASE("splitmix64 matches reference output for seed 0", "[rng]") {
    std::uint64_t state = 0;
    REQUIRE(vrp::splitmix64(state) == 0xE220A8397B1DCDAFULL);
}

TEST_CASE("xoshiro256** matches reference output for state {1,2,3,4}", "[rng]") {
    auto rng = vrp::Rng::fromState({1, 2, 3, 4});
    REQUIRE(rng.next() == 11520ULL);
    REQUIRE(rng.next() == 0ULL);
    REQUIRE(rng.next() == 1509978240ULL);
}

TEST_CASE("below stays within bounds", "[rng]") {
    vrp::Rng rng(12345);
    for (int i = 0; i < 10000; ++i) {
        REQUIRE(rng.below(7) < 7u);
    }
    REQUIRE(rng.below(1) == 0u);
}

TEST_CASE("below is free of modulo bias", "[rng]") {
    // 3 does not divide 2^32, so a naive `% 3` skews the first bucket.
    constexpr std::uint32_t kBound = 3;
    constexpr int kDraws = 300000;
    std::vector<int> counts(kBound, 0);
    vrp::Rng rng(99);
    for (int i = 0; i < kDraws; ++i) {
        counts[rng.below(kBound)]++;
    }
    const double expected = static_cast<double>(kDraws) / kBound;
    for (std::uint32_t b = 0; b < kBound; ++b) {
        REQUIRE(static_cast<double>(counts[b]) > expected * 0.97);
        REQUIRE(static_cast<double>(counts[b]) < expected * 1.03);
    }
}

TEST_CASE("unit lies in [0,1)", "[rng]") {
    vrp::Rng rng(7);
    for (int i = 0; i < 10000; ++i) {
        const double u = rng.unit();
        REQUIRE(u >= 0.0);
        REQUIRE(u < 1.0);
    }
}

TEST_CASE("mixSeed separates seed domains", "[rng]") {
    // Initialising item 5 and producing offspring 5 of generation 0 must not
    // draw the same stream.
    REQUIRE(vrp::mixSeed(42, vrp::kInitDomain, 5) != vrp::mixSeed(42, 0, 5));
    // Neighbouring items and generations must differ.
    REQUIRE(vrp::mixSeed(42, 0, 0) != vrp::mixSeed(42, 0, 1));
    REQUIRE(vrp::mixSeed(42, 0, 0) != vrp::mixSeed(42, 1, 0));
    REQUIRE(vrp::mixSeed(42, 0, 0) != vrp::mixSeed(43, 0, 0));
}
```

- [ ] **Step 2: Run to verify it fails**

Add `test_rng.cpp` to the `vrp_tests` source list in `tests/CMakeLists.txt`, then:

```bash
cmake --build --preset debug
```

Expected: FAIL to compile with `vrp/Rng.hpp: No such file or directory`.

- [ ] **Step 3: Write `include/vrp/Rng.hpp`**

```cpp
#pragma once

#include <array>
#include <cstdint>

namespace vrp {

// Domain tag for initial-population seeding. Chosen outside the range of any
// valid generation index so initialisation and generation 0 never collide.
inline constexpr std::uint64_t kInitDomain = 0xFFFFFFFFFFFFFFFFULL;

std::uint64_t splitmix64(std::uint64_t& state) noexcept;

// Derives an independent seed from (base, a, b). Used as
// mixSeed(seed, kInitDomain, itemIndex) or mixSeed(seed, generation, itemIndex).
std::uint64_t mixSeed(std::uint64_t base, std::uint64_t a, std::uint64_t b) noexcept;

// xoshiro256** — four words of state, cheap to construct, which is what makes
// per-work-item seeding affordable.
class Rng {
public:
    explicit Rng(std::uint64_t seed) noexcept;
    static Rng fromState(std::array<std::uint64_t, 4> state) noexcept;

    std::uint64_t next() noexcept;

    // Unbiased value in [0, bound). Precondition: bound >= 1.
    std::uint32_t below(std::uint32_t bound) noexcept;

    // Uniform double in [0, 1).
    double unit() noexcept;

private:
    Rng() noexcept = default;
    std::array<std::uint64_t, 4> s_{};
};

}  // namespace vrp
```

- [ ] **Step 4: Write `src/vrp/Rng.cpp`**

```cpp
#include "vrp/Rng.hpp"

namespace vrp {
namespace {

constexpr std::uint64_t kGolden = 0x9E3779B97F4A7C15ULL;

std::uint64_t mix(std::uint64_t z) noexcept {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

std::uint64_t rotl(std::uint64_t x, int k) noexcept {
    return (x << k) | (x >> (64 - k));
}

}  // namespace

std::uint64_t splitmix64(std::uint64_t& state) noexcept {
    state += kGolden;
    return mix(state);
}

std::uint64_t mixSeed(std::uint64_t base, std::uint64_t a, std::uint64_t b) noexcept {
    std::uint64_t x = base + kGolden * (a + 1ULL);
    x = mix(x);
    x += kGolden * (b + 1ULL);
    return mix(x);
}

Rng::Rng(std::uint64_t seed) noexcept {
    std::uint64_t state = seed;
    for (auto& word : s_) {
        word = splitmix64(state);
    }
    // xoshiro requires a non-zero state; splitmix64 makes this effectively
    // impossible, but the guard costs nothing.
    if (s_[0] == 0 && s_[1] == 0 && s_[2] == 0 && s_[3] == 0) {
        s_[0] = kGolden;
    }
}

Rng Rng::fromState(std::array<std::uint64_t, 4> state) noexcept {
    Rng rng;
    rng.s_ = state;
    return rng;
}

std::uint64_t Rng::next() noexcept {
    const std::uint64_t result = rotl(s_[1] * 5ULL, 7) * 9ULL;
    const std::uint64_t t = s_[1] << 17;
    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];
    s_[2] ^= t;
    s_[3] = rotl(s_[3], 45);
    return result;
}

std::uint32_t Rng::below(std::uint32_t bound) noexcept {
    // Lemire's nearly-divisionless bounded generation. Rejects only the short
    // tail that would otherwise skew low buckets, unlike `next() % bound`.
    std::uint64_t product =
        static_cast<std::uint64_t>(static_cast<std::uint32_t>(next() >> 32)) * bound;
    auto low = static_cast<std::uint32_t>(product);
    if (low < bound) {
        const std::uint32_t threshold = (0U - bound) % bound;
        while (low < threshold) {
            product =
                static_cast<std::uint64_t>(static_cast<std::uint32_t>(next() >> 32)) * bound;
            low = static_cast<std::uint32_t>(product);
        }
    }
    return static_cast<std::uint32_t>(product >> 32);
}

double Rng::unit() noexcept {
    return static_cast<double>(next() >> 11) * 0x1.0p-53;
}

}  // namespace vrp
```

- [ ] **Step 5: Add `Rng.cpp` to the library**

In `src/CMakeLists.txt`, add `vrp/Rng.cpp` to the `vrp_core` source list.

- [ ] **Step 6: Run tests to verify they pass**

```bash
cmake --build --preset debug
ctest --preset debug --output-on-failure
```

Expected: all `[rng]` tests PASS.

- [ ] **Step 7: Stage and request verification**

```bash
git add -A && git diff --cached --stat
```

---

### Task 3: Problem — locations and precomputed distance matrix

**Files:**
- Create: `include/vrp/Problem.hpp`, `src/vrp/Problem.cpp`
- Create: `tests/test_problem.cpp`
- Modify: `src/CMakeLists.txt`, `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `struct vrp::Point { int x; int y; };`
  - `vrp::Problem`, `explicit Problem(std::vector<Point>)`, `static Problem defaultInstance()`
  - `std::size_t locationCount() const noexcept`, `std::size_t customerCount() const noexcept`
  - `double distance(std::size_t a, std::size_t b) const noexcept`
  - `double routeDistance(std::span<const int> route) const noexcept`

`customerCount()` is `locationCount() - 1`, derived and never configurable. This is what structurally removes the out-of-bounds defect from the original `main.cpp`.

- [ ] **Step 1: Write the failing test**

`tests/test_problem.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <vector>
#include "vrp/Problem.hpp"

using Catch::Matchers::WithinAbs;

TEST_CASE("default instance has 20 locations and 19 customers", "[problem]") {
    const vrp::Problem p = vrp::Problem::defaultInstance();
    REQUIRE(p.locationCount() == 20);
    REQUIRE(p.customerCount() == 19);
}

TEST_CASE("distance matrix is symmetric with a zero diagonal", "[problem]") {
    const vrp::Problem p = vrp::Problem::defaultInstance();
    for (std::size_t a = 0; a < p.locationCount(); ++a) {
        REQUIRE(p.distance(a, a) == 0.0);
        for (std::size_t b = 0; b < p.locationCount(); ++b) {
            REQUIRE(p.distance(a, b) == p.distance(b, a));
        }
    }
}

TEST_CASE("distances agree with a hypot reference", "[problem]") {
    const std::vector<vrp::Point> pts{{0, 0}, {3, 4}, {-6, 8}};
    const vrp::Problem p(pts);
    REQUIRE_THAT(p.distance(0, 1), WithinAbs(5.0, 1e-12));
    REQUIRE_THAT(p.distance(0, 2), WithinAbs(10.0, 1e-12));
    REQUIRE_THAT(p.distance(1, 2), WithinAbs(std::hypot(9.0, 4.0), 1e-12));
}

TEST_CASE("route distance closes the loop through the depot", "[problem]") {
    // Unit square: depot (0,0), then (0,1), (1,1), (1,0). Perimeter is 4.
    const std::vector<vrp::Point> pts{{0, 0}, {0, 1}, {1, 1}, {1, 0}};
    const vrp::Problem p(pts);
    const std::vector<int> route{1, 2, 3};
    REQUIRE_THAT(p.routeDistance(route), WithinAbs(4.0, 1e-12));
}

TEST_CASE("route distance is direction-independent", "[problem]") {
    const std::vector<vrp::Point> pts{{0, 0}, {0, 1}, {1, 1}, {1, 0}};
    const vrp::Problem p(pts);
    const std::vector<int> forward{1, 2, 3};
    const std::vector<int> backward{3, 2, 1};
    REQUIRE_THAT(p.routeDistance(forward),
                 WithinAbs(p.routeDistance(backward), 1e-12));
}

TEST_CASE("an empty route has zero length", "[problem]") {
    const std::vector<vrp::Point> pts{{0, 0}, {5, 5}};
    const vrp::Problem p(pts);
    const std::vector<int> empty;
    REQUIRE(p.routeDistance(empty) == 0.0);
}
```

- [ ] **Step 2: Run to verify it fails**

Add `test_problem.cpp` to `tests/CMakeLists.txt`, then `cmake --build --preset debug`.
Expected: FAIL with `vrp/Problem.hpp: No such file or directory`.

- [ ] **Step 3: Write `include/vrp/Problem.hpp`**

```cpp
#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace vrp {

struct Point {
    int x;
    int y;
};

// Owns the location list and a precomputed distance matrix, so route
// evaluation is array lookups and additions with no sqrt in the hot loop.
class Problem {
public:
    // The 20 coordinates carried over from the original implementation.
    static Problem defaultInstance();

    // locations[0] is the depot. Requires at least one location.
    explicit Problem(std::vector<Point> locations);

    std::size_t locationCount() const noexcept { return locations_.size(); }
    std::size_t customerCount() const noexcept { return locations_.size() - 1; }

    const std::vector<Point>& locations() const noexcept { return locations_; }

    double distance(std::size_t a, std::size_t b) const noexcept {
        return matrix_[a * locations_.size() + b];
    }

    // Depot -> route[0] -> ... -> route[n-1] -> depot.
    double routeDistance(std::span<const int> route) const noexcept;

private:
    std::vector<Point> locations_;
    std::vector<double> matrix_;  // locationCount^2, row-major
};

}  // namespace vrp
```

- [ ] **Step 4: Write `src/vrp/Problem.cpp`**

```cpp
#include "vrp/Problem.hpp"

#include <cmath>
#include <utility>

namespace vrp {

Problem::Problem(std::vector<Point> locations) : locations_(std::move(locations)) {
    const std::size_t n = locations_.size();
    matrix_.resize(n * n);
    for (std::size_t a = 0; a < n; ++a) {
        for (std::size_t b = a; b < n; ++b) {
            const double dx = static_cast<double>(locations_[a].x - locations_[b].x);
            const double dy = static_cast<double>(locations_[a].y - locations_[b].y);
            const double d = std::hypot(dx, dy);
            matrix_[a * n + b] = d;
            matrix_[b * n + a] = d;
        }
    }
}

Problem Problem::defaultInstance() {
    return Problem(std::vector<Point>{
        {0, 0}, {1, 3}, {4, 3}, {6, 1}, {3, 0}, {2, 6}, {5, 5}, {8, 8},
        {9, 4}, {7, 2}, {10, 1}, {12, 3}, {13, 7}, {11, 9}, {6, 9},
        {4, 7}, {2, 8}, {0, 5}, {3, 4}, {7, 6}});
}

double Problem::routeDistance(std::span<const int> route) const noexcept {
    if (route.empty()) {
        return 0.0;
    }
    const std::size_t n = locations_.size();
    double total = matrix_[static_cast<std::size_t>(route.front())];  // depot -> first
    for (std::size_t i = 1; i < route.size(); ++i) {
        total += matrix_[static_cast<std::size_t>(route[i - 1]) * n +
                         static_cast<std::size_t>(route[i])];
    }
    total += matrix_[static_cast<std::size_t>(route.back()) * n];  // last -> depot
    return total;
}

}  // namespace vrp
```

- [ ] **Step 5: Add `Problem.cpp` to the library and run tests**

Add `vrp/Problem.cpp` to `src/CMakeLists.txt`, then:

```bash
cmake --build --preset debug
ctest --preset debug --output-on-failure
```

Expected: all `[problem]` tests PASS.

- [ ] **Step 6: Stage and request verification**

```bash
git add -A && git diff --cached --stat
```

---

### Task 4: Executor — serial and thread pool

**Files:**
- Create: `include/vrp/Executor.hpp`, `src/vrp/Executor.cpp`
- Create: `tests/test_executor.cpp`
- Modify: `src/CMakeLists.txt`, `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `using vrp::ParallelBody = std::function<void(std::size_t begin, std::size_t end)>;`
  - `vrp::Executor` with `virtual std::size_t threadCount() const noexcept` and `virtual void parallelFor(std::size_t n, const ParallelBody&)`
  - `std::unique_ptr<Executor> vrp::makeExecutor(std::size_t threads)` — requires `threads >= 1`
  - `std::pair<std::size_t, std::size_t> vrp::chunkRange(std::size_t n, std::size_t parts, std::size_t index) noexcept`

The body takes a **range**, not an index, so callers allocate scratch buffers once per chunk rather than once per item. Later tasks depend on this.

- [ ] **Step 1: Write the failing test**

`tests/test_executor.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <cstddef>
#include <numeric>
#include <vector>
#include "vrp/Executor.hpp"

namespace {

// Records how many times each index was visited, across any thread count.
std::vector<int> visitCounts(std::size_t n, std::size_t threads) {
    std::vector<int> counts(n, 0);
    auto exec = vrp::makeExecutor(threads);
    exec->parallelFor(n, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            counts[i]++;  // disjoint ranges, so no synchronisation is needed
        }
    });
    return counts;
}

}  // namespace

TEST_CASE("chunkRange partitions exactly and disjointly", "[executor]") {
    for (std::size_t n : {0u, 1u, 2u, 7u, 100u, 1001u}) {
        for (std::size_t parts : {1u, 2u, 3u, 8u, 16u}) {
            std::size_t covered = 0;
            std::size_t previousEnd = 0;
            for (std::size_t i = 0; i < parts; ++i) {
                const auto [begin, end] = vrp::chunkRange(n, parts, i);
                REQUIRE(begin <= end);
                REQUIRE(begin == previousEnd);  // contiguous, no gaps
                REQUIRE(end <= n);
                covered += end - begin;
                previousEnd = end;
            }
            REQUIRE(covered == n);      // full coverage
            REQUIRE(previousEnd == n);  // ends exactly at n
        }
    }
}

TEST_CASE("every index is visited exactly once", "[executor]") {
    for (std::size_t threads : {1u, 2u, 4u, 8u}) {
        for (std::size_t n : {0u, 1u, 3u, 64u, 1000u}) {
            const std::vector<int> counts = visitCounts(n, threads);
            REQUIRE(counts.size() == n);
            for (std::size_t i = 0; i < n; ++i) {
                INFO("threads=" << threads << " n=" << n << " i=" << i);
                REQUIRE(counts[i] == 1);
            }
        }
    }
}

TEST_CASE("n smaller than the thread count is handled", "[executor]") {
    const std::vector<int> counts = visitCounts(3, 8);
    REQUIRE(counts == std::vector<int>{1, 1, 1});
}

TEST_CASE("threads == 1 yields a serial executor", "[executor]") {
    auto exec = vrp::makeExecutor(1);
    REQUIRE(exec->threadCount() == 1);
    std::size_t calls = 0;
    exec->parallelFor(10, [&](std::size_t begin, std::size_t end) {
        calls++;
        REQUIRE(begin == 0);
        REQUIRE(end == 10);
    });
    REQUIRE(calls == 1);
}

TEST_CASE("an empty range invokes the body zero times", "[executor]") {
    for (std::size_t threads : {1u, 4u}) {
        auto exec = vrp::makeExecutor(threads);
        std::atomic<int> calls{0};
        exec->parallelFor(0, [&](std::size_t, std::size_t) { calls++; });
        REQUIRE(calls.load() == 0);
    }
}

TEST_CASE("the pool is reusable across many calls", "[executor]") {
    auto exec = vrp::makeExecutor(4);
    std::vector<int> counts(50, 0);
    for (int round = 0; round < 100; ++round) {
        exec->parallelFor(50, [&](std::size_t begin, std::size_t end) {
            for (std::size_t i = begin; i < end; ++i) {
                counts[i]++;
            }
        });
    }
    for (int c : counts) {
        REQUIRE(c == 100);
    }
}

TEST_CASE("work is accumulated correctly under contention", "[executor]") {
    auto exec = vrp::makeExecutor(8);
    constexpr std::size_t kN = 10000;
    std::vector<std::size_t> values(kN);
    exec->parallelFor(kN, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            values[i] = i * 2;
        }
    });
    const std::size_t sum = std::accumulate(values.begin(), values.end(), std::size_t{0});
    REQUIRE(sum == (kN - 1) * kN);  // 2 * sum(0..kN-1)
}
```

- [ ] **Step 2: Run to verify it fails**

Add `test_executor.cpp` to `tests/CMakeLists.txt`, then `cmake --build --preset debug`.
Expected: FAIL with `vrp/Executor.hpp: No such file or directory`.

- [ ] **Step 3: Write `include/vrp/Executor.hpp`**

```cpp
#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace vrp {

// Receives a half-open range [begin, end), never a single index, so callers can
// hoist scratch allocations out of the per-item loop.
using ParallelBody = std::function<void(std::size_t begin, std::size_t end)>;

// Splits [0, n) into `parts` contiguous, disjoint chunks. Deterministic.
std::pair<std::size_t, std::size_t> chunkRange(std::size_t n, std::size_t parts,
                                               std::size_t index) noexcept;

class Executor {
public:
    virtual ~Executor() = default;
    virtual std::size_t threadCount() const noexcept = 0;
    virtual void parallelFor(std::size_t n, const ParallelBody& body) = 0;
};

class SerialExecutor final : public Executor {
public:
    std::size_t threadCount() const noexcept override { return 1; }
    void parallelFor(std::size_t n, const ParallelBody& body) override;
};

// Persistent workers; the calling thread runs chunk 0 rather than idling.
class ThreadPoolExecutor final : public Executor {
public:
    explicit ThreadPoolExecutor(std::size_t threads);
    ~ThreadPoolExecutor() override;

    ThreadPoolExecutor(const ThreadPoolExecutor&) = delete;
    ThreadPoolExecutor& operator=(const ThreadPoolExecutor&) = delete;

    std::size_t threadCount() const noexcept override { return threads_; }
    void parallelFor(std::size_t n, const ParallelBody& body) override;

private:
    void workerLoop(std::stop_token stop, std::size_t index);

    std::size_t threads_;
    std::mutex mutex_;
    std::condition_variable_any startSignal_;
    std::condition_variable doneSignal_;
    const ParallelBody* body_ = nullptr;
    std::size_t itemCount_ = 0;
    std::uint64_t epoch_ = 0;
    std::size_t outstanding_ = 0;
    std::vector<std::jthread> workers_;  // declared last: destroyed first
};

// Requires threads >= 1. The `--threads 0` sentinel is resolved in the CLI, so
// the core library carries no notion of "auto".
std::unique_ptr<Executor> makeExecutor(std::size_t threads);

}  // namespace vrp
```

- [ ] **Step 4: Write `src/vrp/Executor.cpp`**

```cpp
#include "vrp/Executor.hpp"

#include <algorithm>

namespace vrp {

std::pair<std::size_t, std::size_t> chunkRange(std::size_t n, std::size_t parts,
                                               std::size_t index) noexcept {
    const std::size_t base = n / parts;
    const std::size_t remainder = n % parts;
    const std::size_t begin = index * base + std::min(index, remainder);
    const std::size_t end = begin + base + (index < remainder ? 1u : 0u);
    return {begin, end};
}

void SerialExecutor::parallelFor(std::size_t n, const ParallelBody& body) {
    if (n == 0) {
        return;
    }
    body(0, n);
}

ThreadPoolExecutor::ThreadPoolExecutor(std::size_t threads) : threads_(threads) {
    workers_.reserve(threads_ - 1);
    for (std::size_t i = 1; i < threads_; ++i) {
        workers_.emplace_back([this, i](std::stop_token stop) { workerLoop(stop, i); });
    }
}

ThreadPoolExecutor::~ThreadPoolExecutor() {
    for (auto& worker : workers_) {
        worker.request_stop();
    }
    startSignal_.notify_all();
    // std::jthread destructors join.
}

void ThreadPoolExecutor::parallelFor(std::size_t n, const ParallelBody& body) {
    if (n == 0) {
        return;
    }
    {
        const std::lock_guard lock(mutex_);
        body_ = &body;
        itemCount_ = n;
        outstanding_ = threads_ - 1;
        ++epoch_;
    }
    startSignal_.notify_all();

    const auto [begin, end] = chunkRange(n, threads_, 0);
    if (begin < end) {
        body(begin, end);
    }

    std::unique_lock lock(mutex_);
    doneSignal_.wait(lock, [this] { return outstanding_ == 0; });
    body_ = nullptr;
}

void ThreadPoolExecutor::workerLoop(std::stop_token stop, std::size_t index) {
    std::uint64_t seenEpoch = 0;
    for (;;) {
        std::unique_lock lock(mutex_);
        startSignal_.wait(lock, stop, [this, seenEpoch] { return epoch_ != seenEpoch; });
        if (stop.stop_requested()) {
            return;
        }
        seenEpoch = epoch_;
        const std::size_t n = itemCount_;
        const ParallelBody* body = body_;
        lock.unlock();

        const auto [begin, end] = chunkRange(n, threads_, index);
        if (begin < end) {
            (*body)(begin, end);
        }

        lock.lock();
        const bool last = (--outstanding_ == 0);
        lock.unlock();
        if (last) {
            doneSignal_.notify_one();
        }
    }
}

std::unique_ptr<Executor> makeExecutor(std::size_t threads) {
    if (threads <= 1) {
        return std::make_unique<SerialExecutor>();
    }
    return std::make_unique<ThreadPoolExecutor>(threads);
}

}  // namespace vrp
```

- [ ] **Step 5: Add `Executor.cpp` to the library and run tests**

Add `vrp/Executor.cpp` to `src/CMakeLists.txt`, then:

```bash
cmake --build --preset debug
ctest --preset debug --output-on-failure
```

Expected: all `[executor]` tests PASS. Run the suite three times — a pool with a latent race often passes once.

- [ ] **Step 6: Stage and request verification**

```bash
git add -A && git diff --cached --stat
```

---

### Task 5: Population — flat storage paired with fitness

**Files:**
- Create: `include/vrp/Population.hpp`, `src/vrp/Population.cpp`
- Create: `tests/test_population.cpp`
- Modify: `src/CMakeLists.txt`, `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `vrp::Problem`, `vrp::Executor`, `vrp::Rng`, `vrp::mixSeed`, `vrp::kInitDomain`.
- Produces:
  - `vrp::Population`, default-constructible
  - `Population(std::size_t size, std::size_t routeLength)` — shape only, zero-filled
  - `Population(std::size_t size, const Problem&, std::uint64_t seed, Executor&)` — randomised
  - `std::size_t size() const noexcept`, `std::size_t routeLength() const noexcept`
  - `std::span<const int> route(std::size_t) const noexcept`, `std::span<int> route(std::size_t) noexcept`
  - `double fitness(std::size_t) const noexcept`
  - `void setRoute(std::size_t, std::span<const int>, const Problem&)` — writes and re-evaluates together
  - `void evaluateAll(const Problem&, Executor&)`
  - `std::size_t bestIndex() const noexcept`, `std::size_t worstIndex() const noexcept` — ties resolve to the lowest index
  - `void swap(Population&) noexcept`

Routes and fitness live in one object and are written together. That is what prevents the stale-fitness defect. Tie-breaking on the lowest index is required for determinism.

- [ ] **Step 1: Write the failing test**

`tests/test_population.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <algorithm>
#include <vector>
#include "vrp/Executor.hpp"
#include "vrp/Population.hpp"
#include "vrp/Problem.hpp"

using Catch::Matchers::WithinAbs;

namespace {

bool isPermutationOfCustomers(std::span<const int> route, std::size_t customerCount) {
    if (route.size() != customerCount) {
        return false;
    }
    std::vector<int> sorted(route.begin(), route.end());
    std::sort(sorted.begin(), sorted.end());
    for (std::size_t i = 0; i < customerCount; ++i) {
        if (sorted[i] != static_cast<int>(i + 1)) {
            return false;
        }
    }
    return true;
}

}  // namespace

TEST_CASE("every generated route is a valid permutation", "[population]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    const vrp::Population pop(500, problem, 42, *exec);

    REQUIRE(pop.size() == 500);
    REQUIRE(pop.routeLength() == problem.customerCount());
    for (std::size_t i = 0; i < pop.size(); ++i) {
        INFO("individual " << i);
        REQUIRE(isPermutationOfCustomers(pop.route(i), problem.customerCount()));
    }
}

TEST_CASE("no route contains the depot", "[population]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    const vrp::Population pop(200, problem, 7, *exec);
    for (std::size_t i = 0; i < pop.size(); ++i) {
        for (int gene : pop.route(i)) {
            REQUIRE(gene >= 1);
            REQUIRE(gene <= static_cast<int>(problem.customerCount()));
        }
    }
}

TEST_CASE("stored fitness matches independent recomputation", "[population]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    const vrp::Population pop(200, problem, 99, *exec);
    for (std::size_t i = 0; i < pop.size(); ++i) {
        REQUIRE_THAT(pop.fitness(i),
                     WithinAbs(problem.routeDistance(pop.route(i)), 1e-12));
    }
}

TEST_CASE("population construction is identical across thread counts", "[population]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto serial = vrp::makeExecutor(1);
    auto threaded = vrp::makeExecutor(8);
    const vrp::Population a(1000, problem, 12345, *serial);
    const vrp::Population b(1000, problem, 12345, *threaded);

    for (std::size_t i = 0; i < a.size(); ++i) {
        INFO("individual " << i);
        REQUIRE(std::equal(a.route(i).begin(), a.route(i).end(), b.route(i).begin()));
        REQUIRE(a.fitness(i) == b.fitness(i));  // exact: same ops, same order
    }
}

TEST_CASE("setRoute updates the route and its fitness together", "[population]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    vrp::Population pop(10, problem, 1, *exec);

    std::vector<int> replacement(problem.customerCount());
    for (std::size_t i = 0; i < replacement.size(); ++i) {
        replacement[i] = static_cast<int>(i + 1);
    }
    pop.setRoute(3, replacement, problem);

    REQUIRE(std::equal(pop.route(3).begin(), pop.route(3).end(), replacement.begin()));
    REQUIRE_THAT(pop.fitness(3), WithinAbs(problem.routeDistance(replacement), 1e-12));
}

TEST_CASE("bestIndex and worstIndex break ties toward the lowest index",
          "[population]") {
    // Two locations means every route is the single-customer tour, so all
    // fitness values tie and index 0 must win both queries.
    const vrp::Problem problem(std::vector<vrp::Point>{{0, 0}, {3, 4}});
    auto exec = vrp::makeExecutor(1);
    const vrp::Population pop(5, problem, 1, *exec);
    REQUIRE(pop.bestIndex() == 0);
    REQUIRE(pop.worstIndex() == 0);
}

TEST_CASE("bestIndex and worstIndex find the extremes", "[population]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    const vrp::Population pop(300, problem, 5, *exec);

    const std::size_t best = pop.bestIndex();
    const std::size_t worst = pop.worstIndex();
    for (std::size_t i = 0; i < pop.size(); ++i) {
        REQUIRE(pop.fitness(best) <= pop.fitness(i));
        REQUIRE(pop.fitness(worst) >= pop.fitness(i));
    }
}

TEST_CASE("evaluateAll refreshes every fitness value", "[population]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(4);
    vrp::Population pop(400, problem, 3, *exec);

    // Reverse each route in place, bypassing setRoute so fitness goes stale.
    for (std::size_t i = 0; i < pop.size(); ++i) {
        std::span<int> r = pop.route(i);
        std::reverse(r.begin(), r.end());
    }
    pop.evaluateAll(problem, *exec);

    for (std::size_t i = 0; i < pop.size(); ++i) {
        REQUIRE_THAT(pop.fitness(i),
                     WithinAbs(problem.routeDistance(pop.route(i)), 1e-12));
    }
}

TEST_CASE("the shape constructor allocates without randomising", "[population]") {
    const vrp::Population pop(7, 19);
    REQUIRE(pop.size() == 7);
    REQUIRE(pop.routeLength() == 19);
}

TEST_CASE("swap exchanges contents", "[population]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    vrp::Population a(10, problem, 1, *exec);
    vrp::Population b(10, problem, 2, *exec);

    const std::vector<int> firstOfA(a.route(0).begin(), a.route(0).end());
    const std::vector<int> firstOfB(b.route(0).begin(), b.route(0).end());
    REQUIRE(firstOfA != firstOfB);

    a.swap(b);
    REQUIRE(std::equal(a.route(0).begin(), a.route(0).end(), firstOfB.begin()));
    REQUIRE(std::equal(b.route(0).begin(), b.route(0).end(), firstOfA.begin()));
}
```

- [ ] **Step 2: Run to verify it fails**

Add `test_population.cpp` to `tests/CMakeLists.txt`, then `cmake --build --preset debug`.
Expected: FAIL with `vrp/Population.hpp: No such file or directory`.

- [ ] **Step 3: Write `include/vrp/Population.hpp`**

```cpp
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "vrp/Executor.hpp"
#include "vrp/Problem.hpp"

namespace vrp {

// One contiguous buffer of routes plus a parallel fitness array. Storing them
// together is what keeps fitness from going stale against its route.
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

    std::span<const int> route(std::size_t i) const noexcept {
        return {routes_.data() + i * routeLength_, routeLength_};
    }
    std::span<int> route(std::size_t i) noexcept {
        return {routes_.data() + i * routeLength_, routeLength_};
    }

    double fitness(std::size_t i) const noexcept { return fitness_[i]; }

    // Writes the route and its fitness together. Safe to call concurrently for
    // distinct i: the writes are disjoint.
    void setRoute(std::size_t i, std::span<const int> value, const Problem& problem);

    void evaluateAll(const Problem& problem, Executor& executor);

    std::size_t bestIndex() const noexcept;   // ties -> lowest index
    std::size_t worstIndex() const noexcept;  // ties -> lowest index

    void swap(Population& other) noexcept;

private:
    std::size_t routeLength_ = 0;
    std::vector<int> routes_;      // size * routeLength_
    std::vector<double> fitness_;  // size
};

}  // namespace vrp
```

- [ ] **Step 4: Write `src/vrp/Population.cpp`**

```cpp
#include "vrp/Population.hpp"

#include <algorithm>
#include <utility>

#include "vrp/Rng.hpp"

namespace vrp {

Population::Population(std::size_t size, std::size_t routeLength)
    : routeLength_(routeLength), routes_(size * routeLength), fitness_(size, 0.0) {}

Population::Population(std::size_t size, const Problem& problem, std::uint64_t seed,
                       Executor& executor)
    : routeLength_(problem.customerCount()),
      routes_(size * problem.customerCount()),
      fitness_(size, 0.0) {
    const std::size_t length = routeLength_;
    executor.parallelFor(size, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            int* r = routes_.data() + i * length;
            for (std::size_t j = 0; j < length; ++j) {
                r[j] = static_cast<int>(j + 1);
            }
            // Seeded per item, never per thread, so the shuffle is independent
            // of how the range was chunked.
            Rng rng(mixSeed(seed, kInitDomain, i));
            for (std::size_t j = length; j > 1; --j) {
                const std::uint32_t k = rng.below(static_cast<std::uint32_t>(j));
                std::swap(r[j - 1], r[k]);
            }
            fitness_[i] = problem.routeDistance(std::span<const int>(r, length));
        }
    });
}

void Population::setRoute(std::size_t i, std::span<const int> value,
                          const Problem& problem) {
    int* destination = routes_.data() + i * routeLength_;
    std::copy(value.begin(), value.end(), destination);
    fitness_[i] = problem.routeDistance(std::span<const int>(destination, routeLength_));
}

void Population::evaluateAll(const Problem& problem, Executor& executor) {
    const std::size_t length = routeLength_;
    executor.parallelFor(fitness_.size(), [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            fitness_[i] = problem.routeDistance(
                std::span<const int>(routes_.data() + i * length, length));
        }
    });
}

std::size_t Population::bestIndex() const noexcept {
    std::size_t best = 0;
    for (std::size_t i = 1; i < fitness_.size(); ++i) {
        if (fitness_[i] < fitness_[best]) {  // strict: ties keep the lower index
            best = i;
        }
    }
    return best;
}

std::size_t Population::worstIndex() const noexcept {
    std::size_t worst = 0;
    for (std::size_t i = 1; i < fitness_.size(); ++i) {
        if (fitness_[i] > fitness_[worst]) {  // strict: ties keep the lower index
            worst = i;
        }
    }
    return worst;
}

void Population::swap(Population& other) noexcept {
    std::swap(routeLength_, other.routeLength_);
    routes_.swap(other.routes_);
    fitness_.swap(other.fitness_);
}

}  // namespace vrp
```

- [ ] **Step 5: Add `Population.cpp` to the library and run tests**

```bash
cmake --build --preset debug
ctest --preset debug --output-on-failure
```

Expected: all `[population]` tests PASS.

- [ ] **Step 6: Stage and request verification**

```bash
git add -A && git diff --cached --stat
```

---

### Task 6: Operators — selection, order crossover, swap mutation

**Files:**
- Create: `include/vrp/Operators.hpp`, `src/vrp/Operators.cpp`
- Create: `tests/test_operators.cpp`
- Modify: `src/CMakeLists.txt`, `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `vrp::Population`, `vrp::Rng`.
- Produces, all in namespace `vrp::ops`:
  - `std::size_t tournamentSelect(const Population&, std::size_t tournamentSize, Rng&)`
  - `void orderCrossover(std::span<const int> p1, std::span<const int> p2, std::span<int> child, std::vector<char>& seenScratch, Rng&)`
  - `void swapMutate(std::span<int> route, double rate, Rng&)`

`seenScratch` must be sized `customerCount + 1` because genes are `1..customerCount`. The caller owns it so it can be hoisted out of the per-item loop. This replaces the per-gene `std::find` scan, taking crossover from O(n²) to O(n).

- [ ] **Step 1: Write the failing test**

`tests/test_operators.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>
#include "vrp/Executor.hpp"
#include "vrp/Operators.hpp"
#include "vrp/Population.hpp"
#include "vrp/Problem.hpp"
#include "vrp/Rng.hpp"

namespace {

std::vector<int> identityRoute(std::size_t n) {
    std::vector<int> r(n);
    std::iota(r.begin(), r.end(), 1);
    return r;
}

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

}  // namespace

TEST_CASE("crossover always produces a valid permutation", "[operators]") {
    constexpr std::size_t kN = 19;
    std::vector<int> p1 = identityRoute(kN);
    std::vector<int> p2 = identityRoute(kN);
    std::reverse(p2.begin(), p2.end());

    std::vector<int> child(kN);
    std::vector<char> seen(kN + 1, 0);

    for (std::uint64_t seed = 0; seed < 2000; ++seed) {
        vrp::Rng rng(seed);
        vrp::ops::orderCrossover(p1, p2, child, seen, rng);
        INFO("seed " << seed);
        REQUIRE(isPermutation(child, kN));
    }
}

TEST_CASE("crossover handles shuffled parents", "[operators]") {
    constexpr std::size_t kN = 19;
    std::vector<int> child(kN);
    std::vector<char> seen(kN + 1, 0);

    for (std::uint64_t seed = 0; seed < 500; ++seed) {
        vrp::Rng shuffler(seed + 100000);
        std::vector<int> p1 = identityRoute(kN);
        std::vector<int> p2 = identityRoute(kN);
        for (std::size_t j = kN; j > 1; --j) {
            std::swap(p1[j - 1], p1[shuffler.below(static_cast<std::uint32_t>(j))]);
            std::swap(p2[j - 1], p2[shuffler.below(static_cast<std::uint32_t>(j))]);
        }
        vrp::Rng rng(seed);
        vrp::ops::orderCrossover(p1, p2, child, seen, rng);
        INFO("seed " << seed);
        REQUIRE(isPermutation(child, kN));
    }
}

TEST_CASE("crossover output is contiguous with parent1 somewhere", "[operators]") {
    // The copied segment must survive verbatim. Find it by locating the longest
    // run where child[i] == p1[i]; it must be non-empty for identical parents.
    constexpr std::size_t kN = 19;
    const std::vector<int> p1 = identityRoute(kN);
    const std::vector<int> p2 = identityRoute(kN);
    std::vector<int> child(kN);
    std::vector<char> seen(kN + 1, 0);

    for (std::uint64_t seed = 0; seed < 200; ++seed) {
        vrp::Rng rng(seed);
        vrp::ops::orderCrossover(p1, p2, child, seen, rng);
        INFO("seed " << seed);
        // Identical parents must reproduce the parent exactly.
        REQUIRE(child == p1);
    }
}

TEST_CASE("crossover reuses the scratch buffer safely", "[operators]") {
    // Calling twice with the same scratch must not leak state between calls.
    constexpr std::size_t kN = 19;
    std::vector<int> p1 = identityRoute(kN);
    std::vector<int> p2 = identityRoute(kN);
    std::reverse(p2.begin(), p2.end());
    std::vector<char> seen(kN + 1, 0);

    std::vector<int> first(kN);
    std::vector<int> second(kN);
    vrp::Rng a(555);
    vrp::ops::orderCrossover(p1, p2, first, seen, a);
    vrp::ops::orderCrossover(p1, p2, second, seen, a);
    REQUIRE(isPermutation(first, kN));
    REQUIRE(isPermutation(second, kN));
}

TEST_CASE("mutation rate 0 never alters a route", "[operators]") {
    std::vector<int> route = identityRoute(19);
    const std::vector<int> original = route;
    vrp::Rng rng(1);
    for (int i = 0; i < 5000; ++i) {
        vrp::ops::swapMutate(route, 0.0, rng);
    }
    REQUIRE(route == original);
}

TEST_CASE("mutation rate 1 always swaps two distinct positions", "[operators]") {
    for (std::uint64_t seed = 0; seed < 2000; ++seed) {
        std::vector<int> route = identityRoute(19);
        vrp::Rng rng(seed);
        vrp::ops::swapMutate(route, 1.0, rng);
        INFO("seed " << seed);
        REQUIRE(isPermutation(route, 19));
        // Exactly two positions must differ; a self-swap would leave zero.
        std::size_t differences = 0;
        for (std::size_t i = 0; i < route.size(); ++i) {
            if (route[i] != static_cast<int>(i + 1)) {
                differences++;
            }
        }
        REQUIRE(differences == 2);
    }
}

TEST_CASE("mutation preserves permutations under repeated application",
          "[operators]") {
    std::vector<int> route = identityRoute(19);
    vrp::Rng rng(42);
    for (int i = 0; i < 10000; ++i) {
        vrp::ops::swapMutate(route, 0.5, rng);
        REQUIRE(isPermutation(route, 19));
    }
}

TEST_CASE("mutation leaves a one-element route alone", "[operators]") {
    std::vector<int> route{1};
    vrp::Rng rng(1);
    vrp::ops::swapMutate(route, 1.0, rng);
    REQUIRE(route == std::vector<int>{1});
}

TEST_CASE("tournament selection returns an in-range index", "[operators]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    const vrp::Population pop(100, problem, 11, *exec);
    vrp::Rng rng(3);
    for (int i = 0; i < 5000; ++i) {
        REQUIRE(vrp::ops::tournamentSelect(pop, 5, rng) < pop.size());
    }
}

TEST_CASE("tournament selection favours fitter individuals", "[operators]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    auto exec = vrp::makeExecutor(1);
    const vrp::Population pop(1000, problem, 13, *exec);

    // Mean fitness of tournament winners must beat the population mean.
    double populationMean = 0.0;
    for (std::size_t i = 0; i < pop.size(); ++i) {
        populationMean += pop.fitness(i);
    }
    populationMean /= static_cast<double>(pop.size());

    vrp::Rng rng(17);
    constexpr int kDraws = 5000;
    double winnerMean = 0.0;
    for (int i = 0; i < kDraws; ++i) {
        winnerMean += pop.fitness(vrp::ops::tournamentSelect(pop, 5, rng));
    }
    winnerMean /= kDraws;

    REQUIRE(winnerMean < populationMean);
}
```

- [ ] **Step 2: Run to verify it fails**

Add `test_operators.cpp` to `tests/CMakeLists.txt`, then `cmake --build --preset debug`.
Expected: FAIL with `vrp/Operators.hpp: No such file or directory`.

- [ ] **Step 3: Write `include/vrp/Operators.hpp`**

```cpp
#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "vrp/Population.hpp"
#include "vrp/Rng.hpp"

namespace vrp::ops {

// Draws `tournamentSize` candidates and returns the fittest index.
// A tournamentSize of 0 is treated as 1.
std::size_t tournamentSelect(const Population& population, std::size_t tournamentSize,
                             Rng& rng);

// Order crossover (OX): copy a random segment from p1, then fill the remaining
// positions with the genes of p2 in their original order.
//
// `seenScratch` must have size >= customerCount + 1; genes are 1-based. The
// caller owns it so it can be allocated once per chunk instead of per child.
void orderCrossover(std::span<const int> p1, std::span<const int> p2,
                    std::span<int> child, std::vector<char>& seenScratch, Rng& rng);

// With probability `rate`, swaps two distinct positions. The second index is
// drawn from the remaining range, so the swap is never a silent no-op.
void swapMutate(std::span<int> route, double rate, Rng& rng);

}  // namespace vrp::ops
```

- [ ] **Step 4: Write `src/vrp/Operators.cpp`**

```cpp
#include "vrp/Operators.hpp"

#include <algorithm>
#include <utility>

namespace vrp::ops {

std::size_t tournamentSelect(const Population& population, std::size_t tournamentSize,
                             Rng& rng) {
    const auto populationSize = static_cast<std::uint32_t>(population.size());
    std::size_t best = rng.below(populationSize);
    double bestFitness = population.fitness(best);

    const std::size_t draws = tournamentSize > 0 ? tournamentSize : 1;
    for (std::size_t i = 1; i < draws; ++i) {
        const std::size_t candidate = rng.below(populationSize);
        const double candidateFitness = population.fitness(candidate);
        if (candidateFitness < bestFitness) {
            bestFitness = candidateFitness;
            best = candidate;
        }
    }
    return best;
}

void orderCrossover(std::span<const int> p1, std::span<const int> p2,
                    std::span<int> child, std::vector<char>& seenScratch, Rng& rng) {
    const std::size_t n = p1.size();
    std::fill(seenScratch.begin(), seenScratch.end(), char{0});

    auto low = rng.below(static_cast<std::uint32_t>(n));
    auto high = rng.below(static_cast<std::uint32_t>(n));
    if (low > high) {
        std::swap(low, high);
    }

    for (std::size_t i = low; i <= high; ++i) {
        child[i] = p1[i];
        seenScratch[static_cast<std::size_t>(p1[i])] = 1;
    }

    std::size_t write = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const int gene = p2[i];
        if (seenScratch[static_cast<std::size_t>(gene)] != 0) {
            continue;
        }
        if (write >= low && write <= high) {
            write = static_cast<std::size_t>(high) + 1;  // jump over the segment
        }
        child[write] = gene;
        ++write;
    }
}

void swapMutate(std::span<int> route, double rate, Rng& rng) {
    if (route.size() < 2) {
        return;
    }
    if (rng.unit() >= rate) {
        return;
    }
    const auto n = static_cast<std::uint32_t>(route.size());
    const std::uint32_t first = rng.below(n);
    // Draw from the remaining n-1 slots, then shift past `first`, so the two
    // indices can never coincide.
    const std::uint32_t offset = rng.below(n - 1);
    const std::uint32_t second = offset >= first ? offset + 1 : offset;
    std::swap(route[first], route[second]);
}

}  // namespace vrp::ops
```

- [ ] **Step 5: Add `Operators.cpp` to the library and run tests**

```bash
cmake --build --preset debug
ctest --preset debug --output-on-failure
```

Expected: all `[operators]` tests PASS.

- [ ] **Step 6: Stage and request verification**

```bash
git add -A && git diff --cached --stat
```

---

### Task 7: Config and the two evolution strategies

**Files:**
- Create: `include/vrp/Config.hpp`, `include/vrp/Strategy.hpp`, `src/vrp/Strategy.cpp`
- Create: `tests/test_strategies.cpp`
- Modify: `src/CMakeLists.txt`, `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `vrp::Population`, `vrp::Problem`, `vrp::Executor`, `vrp::ops::*`, `vrp::Rng`, `vrp::mixSeed`.
- Produces:
  - `enum class vrp::StrategyKind { SteadyState, Generational };`
  - `struct vrp::GaParams` with `populationSize=100000`, `generations=100`, `mutationRate=0.2`, `tournamentSize=5`, `eliteCount=1`, `seed=42`
  - `vrp::EvolutionStrategy` with `virtual const char* name() const noexcept` and `virtual void step(std::size_t generation, Population&, const Problem&, const GaParams&, Executor&)`
  - `std::unique_ptr<EvolutionStrategy> vrp::makeStrategy(StrategyKind)`

Every offspring is seeded `mixSeed(params.seed, generation, index)` — independent of chunking, which is what makes the threaded result match the serial one.

- [ ] **Step 1: Write the failing test**

`tests/test_strategies.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <string>
#include <vector>
#include "vrp/Config.hpp"
#include "vrp/Executor.hpp"
#include "vrp/Population.hpp"
#include "vrp/Problem.hpp"
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

}  // namespace

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
```

- [ ] **Step 2: Run to verify it fails**

Add `test_strategies.cpp` to `tests/CMakeLists.txt`, then `cmake --build --preset debug`.
Expected: FAIL with `vrp/Config.hpp: No such file or directory`.

- [ ] **Step 3: Write `include/vrp/Config.hpp`**

```cpp
#pragma once

#include <cstddef>
#include <cstdint>

namespace vrp {

enum class StrategyKind {
    SteadyState,
    Generational,
};

struct GaParams {
    std::size_t populationSize = 100000;
    std::size_t generations = 100;
    double mutationRate = 0.2;
    std::size_t tournamentSize = 5;
    std::size_t eliteCount = 1;  // generational only
    std::uint64_t seed = 42;
};

}  // namespace vrp
```

- [ ] **Step 4: Write `include/vrp/Strategy.hpp`**

```cpp
#pragma once

#include <cstddef>
#include <memory>

#include "vrp/Config.hpp"
#include "vrp/Executor.hpp"
#include "vrp/Population.hpp"
#include "vrp/Problem.hpp"

namespace vrp {

// Advances a population by one generation. Implementations contain no threads;
// they schedule through the Executor they are handed.
class EvolutionStrategy {
public:
    virtual ~EvolutionStrategy() = default;
    virtual const char* name() const noexcept = 0;
    virtual void step(std::size_t generation, Population& population,
                      const Problem& problem, const GaParams& params,
                      Executor& executor) = 0;
};

std::unique_ptr<EvolutionStrategy> makeStrategy(StrategyKind kind);

}  // namespace vrp
```

- [ ] **Step 5: Write `src/vrp/Strategy.cpp`**

```cpp
#include "vrp/Strategy.hpp"

#include <algorithm>
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
        const Population& current = population;
        Rng rng(mixSeed(params.seed, generation, 0));

        const std::size_t first = ops::tournamentSelect(current, params.tournamentSize, rng);
        const std::size_t second = ops::tournamentSelect(current, params.tournamentSize, rng);

        child_.resize(population.routeLength());
        seen_.assign(problem.customerCount() + 1, 0);
        ops::orderCrossover(current.route(first), current.route(second), child_, seen_, rng);
        ops::swapMutate(child_, params.mutationRate, rng);

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
        // though partial_sort is not stable.
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

        const std::size_t offspring = count - elites;
        executor.parallelFor(offspring, [&](std::size_t begin, std::size_t end) {
            // Scratch is per chunk, not per child — this is why the body takes
            // a range instead of an index.
            std::vector<char> seen(problem.customerCount() + 1, 0);
            std::vector<int> child(length);
            for (std::size_t k = begin; k < end; ++k) {
                const std::size_t slot = k + elites;
                // Seeded by slot, so chunking cannot change the outcome.
                Rng rng(mixSeed(params.seed, generation, slot));
                const std::size_t a =
                    ops::tournamentSelect(current, params.tournamentSize, rng);
                const std::size_t b =
                    ops::tournamentSelect(current, params.tournamentSize, rng);
                ops::orderCrossover(current.route(a), current.route(b), child, seen, rng);
                ops::swapMutate(child, params.mutationRate, rng);
                next_.setRoute(slot, child, problem);
            }
        });

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
```

- [ ] **Step 6: Add `Strategy.cpp` to the library and run tests**

```bash
cmake --build --preset debug
ctest --preset debug --output-on-failure
```

Expected: all `[strategy]` tests PASS.

- [ ] **Step 7: Stage and request verification**

```bash
git add -A && git diff --cached --stat
```

---

### Task 8: Solver — the generation loop

**Files:**
- Create: `include/vrp/Solver.hpp`, `src/vrp/Solver.cpp`
- Create: `tests/test_solver.cpp`
- Modify: `src/CMakeLists.txt`, `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `vrp::Problem`, `vrp::GaParams`, `vrp::EvolutionStrategy`, `vrp::Executor`, `vrp::Population`.
- Produces:
  - `struct vrp::RunResult { std::vector<int> bestRoute; double bestDistance; std::size_t generationsRun; double elapsedSeconds; };`
  - `using vrp::ProgressCallback = std::function<void(std::size_t generation, double bestDistance)>;`
  - `vrp::Solver(const Problem&, GaParams, std::unique_ptr<EvolutionStrategy>, std::unique_ptr<Executor>)`
  - `RunResult Solver::run(const ProgressCallback& = {})`

`Solver` performs no I/O; it reports through the callback. The `Problem` reference must outlive the `Solver`.

- [ ] **Step 1: Write the failing test**

`tests/test_solver.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <limits>
#include <vector>
#include "vrp/Config.hpp"
#include "vrp/Executor.hpp"
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

}  // namespace

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
    const vrp::RunResult result = solver.run(
        [&](std::size_t generation, double) { generations.push_back(generation); });

    REQUIRE(generations.size() == params.generations);
    REQUIRE(generations.front() == 0);
    REQUIRE(generations.back() == params.generations - 1);
    REQUIRE(result.generationsRun == params.generations);
}

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

TEST_CASE("zero generations returns the initial best", "[solver]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    vrp::GaParams params = smallParams();
    params.generations = 0;
    vrp::Solver solver(problem, params,
                       vrp::makeStrategy(vrp::StrategyKind::Generational),
                       vrp::makeExecutor(1));
    const vrp::RunResult result = solver.run();
    REQUIRE(result.generationsRun == 0);
    REQUIRE(result.bestRoute.size() == problem.customerCount());
}

TEST_CASE("elapsed time is recorded", "[solver]") {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    vrp::Solver solver(problem, smallParams(),
                       vrp::makeStrategy(vrp::StrategyKind::Generational),
                       vrp::makeExecutor(1));
    const vrp::RunResult result = solver.run();
    REQUIRE(result.elapsedSeconds >= 0.0);
}
```

- [ ] **Step 2: Run to verify it fails**

Add `test_solver.cpp` to `tests/CMakeLists.txt`, then `cmake --build --preset debug`.
Expected: FAIL with `vrp/Solver.hpp: No such file or directory`.

- [ ] **Step 3: Write `include/vrp/Solver.hpp`**

```cpp
#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "vrp/Config.hpp"
#include "vrp/Executor.hpp"
#include "vrp/Problem.hpp"
#include "vrp/Strategy.hpp"

namespace vrp {

struct RunResult {
    std::vector<int> bestRoute;
    double bestDistance = 0.0;
    std::size_t generationsRun = 0;
    double elapsedSeconds = 0.0;
};

using ProgressCallback = std::function<void(std::size_t generation, double bestDistance)>;

// Owns the generation loop. Performs no I/O — reporting goes through the
// callback — so it stays testable. `problem` must outlive the Solver.
class Solver {
public:
    Solver(const Problem& problem, GaParams params,
           std::unique_ptr<EvolutionStrategy> strategy, std::unique_ptr<Executor> executor);

    RunResult run(const ProgressCallback& progress = {});

    const EvolutionStrategy& strategy() const noexcept { return *strategy_; }
    const Executor& executor() const noexcept { return *executor_; }

private:
    const Problem& problem_;
    GaParams params_;
    std::unique_ptr<EvolutionStrategy> strategy_;
    std::unique_ptr<Executor> executor_;
};

}  // namespace vrp
```

- [ ] **Step 4: Write `src/vrp/Solver.cpp`**

```cpp
#include "vrp/Solver.hpp"

#include <chrono>
#include <utility>

#include "vrp/Population.hpp"

namespace vrp {

Solver::Solver(const Problem& problem, GaParams params,
               std::unique_ptr<EvolutionStrategy> strategy,
               std::unique_ptr<Executor> executor)
    : problem_(problem),
      params_(params),
      strategy_(std::move(strategy)),
      executor_(std::move(executor)) {}

RunResult Solver::run(const ProgressCallback& progress) {
    const auto started = std::chrono::steady_clock::now();

    Population population(params_.populationSize, problem_, params_.seed, *executor_);

    for (std::size_t generation = 0; generation < params_.generations; ++generation) {
        strategy_->step(generation, population, problem_, params_, *executor_);
        if (progress) {
            progress(generation, population.fitness(population.bestIndex()));
        }
    }

    const std::size_t best = population.bestIndex();
    RunResult result;
    result.bestRoute.assign(population.route(best).begin(), population.route(best).end());
    result.bestDistance = population.fitness(best);
    result.generationsRun = params_.generations;
    result.elapsedSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return result;
}

}  // namespace vrp
```

- [ ] **Step 5: Add `Solver.cpp` to the library and run tests**

```bash
cmake --build --preset debug
ctest --preset debug --output-on-failure
```

Expected: all `[solver]` tests PASS.

- [ ] **Step 6: Stage and request verification**

```bash
git add -A && git diff --cached --stat
```

---

### Task 9: Determinism suite — the keystone

**Files:**
- Create: `tests/test_determinism.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: everything from Tasks 2–8. Adds no production code.

This is the task that proves the threaded path correct. On Windows it is the *only* such proof, because GCC on MinGW-w64 provides no TSan. If any assertion here fails, the cause is a real ordering dependency in the parallel code — do not weaken the assertion to make it pass.

- [ ] **Step 1: Write the failing test**

`tests/test_determinism.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <vector>
#include "vrp/Config.hpp"
#include "vrp/Executor.hpp"
#include "vrp/Population.hpp"
#include "vrp/Problem.hpp"
#include "vrp/Solver.hpp"
#include "vrp/Strategy.hpp"

namespace {

vrp::RunResult runWith(vrp::StrategyKind kind, std::size_t threads, std::uint64_t seed) {
    static const vrp::Problem problem = vrp::Problem::defaultInstance();
    vrp::GaParams params;
    params.populationSize = 512;
    params.generations = 25;
    params.seed = seed;
    vrp::Solver solver(problem, params, vrp::makeStrategy(kind),
                       vrp::makeExecutor(threads));
    return solver.run();
}

// Snapshots an entire population so two runs can be compared element by element.
struct Snapshot {
    std::vector<int> routes;
    std::vector<double> fitness;
};

Snapshot evolveSnapshot(vrp::StrategyKind kind, std::size_t threads, std::uint64_t seed) {
    const vrp::Problem problem = vrp::Problem::defaultInstance();
    vrp::GaParams params;
    params.populationSize = 512;
    params.generations = 25;
    params.seed = seed;

    auto executor = vrp::makeExecutor(threads);
    auto strategy = vrp::makeStrategy(kind);
    vrp::Population population(params.populationSize, problem, params.seed, *executor);
    for (std::size_t g = 0; g < params.generations; ++g) {
        strategy->step(g, population, problem, params, *executor);
    }

    Snapshot snapshot;
    snapshot.fitness.reserve(population.size());
    snapshot.routes.reserve(population.size() * population.routeLength());
    for (std::size_t i = 0; i < population.size(); ++i) {
        snapshot.fitness.push_back(population.fitness(i));
        for (int gene : population.route(i)) {
            snapshot.routes.push_back(gene);
        }
    }
    return snapshot;
}

}  // namespace

TEST_CASE("the same seed reproduces the same result", "[determinism]") {
    for (auto kind : {vrp::StrategyKind::SteadyState, vrp::StrategyKind::Generational}) {
        const vrp::RunResult first = runWith(kind, 1, 777);
        const vrp::RunResult second = runWith(kind, 1, 777);
        REQUIRE(first.bestRoute == second.bestRoute);
        REQUIRE(first.bestDistance == second.bestDistance);
    }
}

TEST_CASE("different seeds produce different results", "[determinism]") {
    const vrp::RunResult a = runWith(vrp::StrategyKind::Generational, 1, 1);
    const vrp::RunResult b = runWith(vrp::StrategyKind::Generational, 1, 2);
    REQUIRE(a.bestRoute != b.bestRoute);
}

TEST_CASE("results are bit-identical across thread counts", "[determinism]") {
    // The load-bearing test. Per-item seeding means the number of threads must
    // not influence any value. Exact equality is intended, not Approx.
    for (auto kind : {vrp::StrategyKind::SteadyState, vrp::StrategyKind::Generational}) {
        const vrp::RunResult serial = runWith(kind, 1, 31337);
        for (std::size_t threads : {2u, 4u, 8u}) {
            const vrp::RunResult threaded = runWith(kind, threads, 31337);
            INFO("threads = " << threads);
            REQUIRE(threaded.bestRoute == serial.bestRoute);
            REQUIRE(threaded.bestDistance == serial.bestDistance);
        }
    }
}

TEST_CASE("entire populations are bit-identical across thread counts",
          "[determinism]") {
    // Comparing only the best individual could hide a divergence elsewhere.
    for (auto kind : {vrp::StrategyKind::SteadyState, vrp::StrategyKind::Generational}) {
        const Snapshot serial = evolveSnapshot(kind, 1, 8675309);
        for (std::size_t threads : {2u, 4u, 8u}) {
            const Snapshot threaded = evolveSnapshot(kind, threads, 8675309);
            INFO("threads = " << threads);
            REQUIRE(threaded.routes == serial.routes);
            REQUIRE(threaded.fitness == serial.fitness);
        }
    }
}

TEST_CASE("repeated threaded runs are stable", "[determinism]") {
    // A latent race often survives a single run. Repeat to raise the odds of
    // catching one.
    const Snapshot reference = evolveSnapshot(vrp::StrategyKind::Generational, 8, 555);
    for (int attempt = 0; attempt < 10; ++attempt) {
        const Snapshot again = evolveSnapshot(vrp::StrategyKind::Generational, 8, 555);
        INFO("attempt " << attempt);
        REQUIRE(again.routes == reference.routes);
        REQUIRE(again.fitness == reference.fitness);
    }
}
```

- [ ] **Step 2: Run to verify it compiles and passes**

Add `test_determinism.cpp` to `tests/CMakeLists.txt`, then:

```bash
cmake --build --preset debug
ctest --preset debug -R determinism --output-on-failure
```

Expected: PASS. These tests exercise existing code, so unlike earlier tasks there is no red phase — a failure here means Tasks 4–8 have a real defect to fix, not that the test needs writing.

- [ ] **Step 3: Run the whole suite repeatedly**

```bash
for i in 1 2 3 4 5; do ctest --preset debug --output-on-failure || break; done
```

Expected: five consecutive clean runs.

- [ ] **Step 4: Run in release mode too**

Optimisation changes timing and can expose races the debug build hides.

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release --output-on-failure
```

Expected: PASS.

- [ ] **Step 5: Stage and request verification**

```bash
git add -A && git diff --cached --stat
```

---

### Task 10: CLI, reporter, and the executable

**Files:**
- Create: `src/app/Cli.hpp`, `src/app/Cli.cpp`, `src/app/Reporter.hpp`, `src/app/Reporter.cpp`, `src/app/main.cpp`
- Create: `tests/test_cli.cpp`
- Modify: `src/CMakeLists.txt`, `tests/CMakeLists.txt`
- Delete: `main.cpp`, `main_parallel.cpp`

**Interfaces:**
- Consumes: `vrp::GaParams`, `vrp::StrategyKind`, `vrp::Solver`, `vrp::RunResult`.
- Produces, in namespace `vrp::app`:
  - `struct Options { GaParams params; StrategyKind strategy; std::size_t threads; bool quiet; bool helpRequested; };`
  - `struct ParseResult { Options options; bool ok; std::string error; };`
  - `ParseResult parseArgs(std::span<const std::string_view> args)` — excludes `argv[0]`
  - `std::string usage()`
  - `class Reporter` with `void onGeneration(std::size_t, double)` and `void onResult(const RunResult&)`

`parseArgs` resolves `--threads 0` to `hardware_concurrency` and `--seed 0` to a `random_device` draw, so the core library never sees a sentinel.

`Reporter` fixes the "never reports the final generation" defect: it prints when `(generation + 1) % interval == 0` **or** `generation + 1 == generations`.

- [ ] **Step 1: Write the failing test**

`tests/test_cli.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include "app/Cli.hpp"

namespace {

vrp::app::ParseResult parse(std::vector<std::string_view> args) {
    return vrp::app::parseArgs(args);
}

}  // namespace

TEST_CASE("defaults reproduce the original sequential program", "[cli]") {
    const auto result = parse({});
    REQUIRE(result.ok);
    REQUIRE(result.options.strategy == vrp::StrategyKind::SteadyState);
    REQUIRE(result.options.threads == 1);
    REQUIRE(result.options.params.populationSize == 100000);
    REQUIRE(result.options.params.generations == 100);
    REQUIRE(result.options.params.mutationRate == 0.2);
    REQUIRE(result.options.params.tournamentSize == 5);
    REQUIRE(result.options.params.eliteCount == 1);
    REQUIRE(result.options.params.seed == 42);
    REQUIRE_FALSE(result.options.quiet);
}

TEST_CASE("every option is parsed", "[cli]") {
    const auto result = parse({"--strategy", "generational", "--population", "500",
                               "--generations", "7", "--mutation", "0.5",
                               "--tournament", "3", "--elite", "2", "--threads", "4",
                               "--seed", "99", "--quiet"});
    REQUIRE(result.ok);
    REQUIRE(result.options.strategy == vrp::StrategyKind::Generational);
    REQUIRE(result.options.params.populationSize == 500);
    REQUIRE(result.options.params.generations == 7);
    REQUIRE(result.options.params.mutationRate == 0.5);
    REQUIRE(result.options.params.tournamentSize == 3);
    REQUIRE(result.options.params.eliteCount == 2);
    REQUIRE(result.options.threads == 4);
    REQUIRE(result.options.params.seed == 99);
    REQUIRE(result.options.quiet);
}

TEST_CASE("threads 0 resolves to hardware concurrency", "[cli]") {
    const auto result = parse({"--threads", "0"});
    REQUIRE(result.ok);
    const unsigned expected = std::max(1u, std::thread::hardware_concurrency());
    REQUIRE(result.options.threads == expected);
}

TEST_CASE("seed 0 resolves to a nonzero random seed", "[cli]") {
    const auto result = parse({"--seed", "0"});
    REQUIRE(result.ok);
    REQUIRE(result.options.params.seed != 0);
}

TEST_CASE("help is requested and reported", "[cli]") {
    const auto result = parse({"--help"});
    REQUIRE(result.ok);
    REQUIRE(result.options.helpRequested);
    REQUIRE_FALSE(vrp::app::usage().empty());
}

TEST_CASE("unknown flags are rejected", "[cli]") {
    const auto result = parse({"--nonsense"});
    REQUIRE_FALSE(result.ok);
    REQUIRE(result.error.find("--nonsense") != std::string::npos);
}

TEST_CASE("a missing value is rejected", "[cli]") {
    const auto result = parse({"--population"});
    REQUIRE_FALSE(result.ok);
}

TEST_CASE("non-numeric values are rejected", "[cli]") {
    REQUIRE_FALSE(parse({"--population", "abc"}).ok);
    REQUIRE_FALSE(parse({"--mutation", "high"}).ok);
    REQUIRE_FALSE(parse({"--threads", "-1"}).ok);
    REQUIRE_FALSE(parse({"--population", "12x"}).ok);
}

TEST_CASE("out-of-range values are rejected", "[cli]") {
    REQUIRE_FALSE(parse({"--population", "0"}).ok);
    REQUIRE_FALSE(parse({"--tournament", "0"}).ok);
    REQUIRE_FALSE(parse({"--mutation", "1.5"}).ok);
    REQUIRE_FALSE(parse({"--mutation", "-0.1"}).ok);
    REQUIRE_FALSE(parse({"--strategy", "quantum"}).ok);
}

TEST_CASE("elite must be smaller than the population", "[cli]") {
    REQUIRE_FALSE(parse({"--population", "10", "--elite", "10"}).ok);
    REQUIRE_FALSE(parse({"--population", "10", "--elite", "11"}).ok);
    REQUIRE(parse({"--population", "10", "--elite", "9"}).ok);
}

TEST_CASE("mutation accepts its boundary values", "[cli]") {
    REQUIRE(parse({"--mutation", "0"}).ok);
    REQUIRE(parse({"--mutation", "1"}).ok);
}
```

- [ ] **Step 2: Run to verify it fails**

Add `test_cli.cpp` to `tests/CMakeLists.txt`, then `cmake --build --preset debug`.
Expected: FAIL with `app/Cli.hpp: No such file or directory`.

- [ ] **Step 3: Write `src/app/Cli.hpp`**

```cpp
#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <string_view>

#include "vrp/Config.hpp"

namespace vrp::app {

struct Options {
    GaParams params{};
    StrategyKind strategy = StrategyKind::SteadyState;
    std::size_t threads = 1;
    bool quiet = false;
    bool helpRequested = false;
};

struct ParseResult {
    Options options{};
    bool ok = true;
    std::string error;
};

// `args` excludes argv[0]. Resolves the --threads 0 and --seed 0 sentinels here
// so the core library never sees them.
ParseResult parseArgs(std::span<const std::string_view> args);

std::string usage();

}  // namespace vrp::app
```

- [ ] **Step 4: Write `src/app/Cli.cpp`**

```cpp
#include "app/Cli.hpp"

#include <algorithm>
#include <charconv>
#include <random>
#include <thread>
#include <vector>

namespace vrp::app {
namespace {

bool parseSize(std::string_view text, std::size_t& out) {
    if (text.empty()) {
        return false;
    }
    const char* begin = text.data();
    const char* end = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(begin, end, out);
    return ec == std::errc{} && ptr == end;
}

bool parseU64(std::string_view text, std::uint64_t& out) {
    if (text.empty()) {
        return false;
    }
    const char* begin = text.data();
    const char* end = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(begin, end, out);
    return ec == std::errc{} && ptr == end;
}

bool parseDouble(std::string_view text, double& out) {
    if (text.empty()) {
        return false;
    }
    // from_chars for double is available in GCC 11+ and MSVC 19.24+.
    const char* begin = text.data();
    const char* end = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(begin, end, out);
    return ec == std::errc{} && ptr == end;
}

ParseResult fail(std::string message) {
    ParseResult result;
    result.ok = false;
    result.error = std::move(message);
    return result;
}

}  // namespace

std::string usage() {
    return
        "vrp_ga [options]\n"
        "\n"
        "  --strategy    steady-state | generational  (default: steady-state)\n"
        "  --population  N    population size         (default: 100000)\n"
        "  --generations N    iterations              (default: 100)\n"
        "  --mutation    R    swap probability, 0..1  (default: 0.2)\n"
        "  --tournament  N    candidates per parent   (default: 5)\n"
        "  --elite       N    survivors, generational (default: 1)\n"
        "  --threads     N    0 selects all cores     (default: 1)\n"
        "  --seed        N    0 draws a random seed   (default: 42)\n"
        "  --quiet            suppress per-generation output\n"
        "  --help             show this message\n"
        "\n"
        "Note: steady-state replaces one individual per generation, so its\n"
        "generation loop does not parallelise. Use --strategy generational to\n"
        "see a speedup from --threads.\n";
}

ParseResult parseArgs(std::span<const std::string_view> args) {
    ParseResult result;
    Options& options = result.options;

    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view flag = args[i];

        if (flag == "--help" || flag == "-h") {
            options.helpRequested = true;
            return result;
        }
        if (flag == "--quiet") {
            options.quiet = true;
            continue;
        }

        if (i + 1 >= args.size()) {
            return fail(std::string("missing value for ").append(flag));
        }
        const std::string_view value = args[++i];

        if (flag == "--strategy") {
            if (value == "steady-state") {
                options.strategy = StrategyKind::SteadyState;
            } else if (value == "generational") {
                options.strategy = StrategyKind::Generational;
            } else {
                return fail(std::string("unknown strategy: ").append(value));
            }
        } else if (flag == "--population") {
            if (!parseSize(value, options.params.populationSize)) {
                return fail(std::string("invalid --population: ").append(value));
            }
            if (options.params.populationSize == 0) {
                return fail("--population must be at least 1");
            }
        } else if (flag == "--generations") {
            if (!parseSize(value, options.params.generations)) {
                return fail(std::string("invalid --generations: ").append(value));
            }
        } else if (flag == "--mutation") {
            if (!parseDouble(value, options.params.mutationRate)) {
                return fail(std::string("invalid --mutation: ").append(value));
            }
            if (options.params.mutationRate < 0.0 || options.params.mutationRate > 1.0) {
                return fail("--mutation must lie in [0, 1]");
            }
        } else if (flag == "--tournament") {
            if (!parseSize(value, options.params.tournamentSize)) {
                return fail(std::string("invalid --tournament: ").append(value));
            }
            if (options.params.tournamentSize == 0) {
                return fail("--tournament must be at least 1");
            }
        } else if (flag == "--elite") {
            if (!parseSize(value, options.params.eliteCount)) {
                return fail(std::string("invalid --elite: ").append(value));
            }
        } else if (flag == "--threads") {
            if (!parseSize(value, options.threads)) {
                return fail(std::string("invalid --threads: ").append(value));
            }
        } else if (flag == "--seed") {
            if (!parseU64(value, options.params.seed)) {
                return fail(std::string("invalid --seed: ").append(value));
            }
        } else {
            return fail(std::string("unknown option: ").append(flag));
        }
    }

    if (options.params.eliteCount >= options.params.populationSize) {
        return fail("--elite must be smaller than --population");
    }

    if (options.threads == 0) {
        options.threads = std::max(1u, std::thread::hardware_concurrency());
    }
    if (options.params.seed == 0) {
        std::random_device device;
        options.params.seed = (static_cast<std::uint64_t>(device()) << 32) ^ device();
        if (options.params.seed == 0) {
            options.params.seed = 1;
        }
    }

    return result;
}

}  // namespace vrp::app
```

- [ ] **Step 5: Write `src/app/Reporter.hpp`**

```cpp
#pragma once

#include <cstddef>
#include <iosfwd>

#include "vrp/Solver.hpp"

namespace vrp::app {

// Prints progress on a fixed interval and always on the final generation —
// the original code reported every tenth generation and so never printed the
// last one.
class Reporter {
public:
    Reporter(std::ostream& out, std::size_t generations, std::size_t interval, bool quiet);

    void onGeneration(std::size_t generation, double bestDistance) const;
    void onResult(const RunResult& result) const;

private:
    std::ostream& out_;
    std::size_t generations_;
    std::size_t interval_;
    bool quiet_;
};

}  // namespace vrp::app
```

- [ ] **Step 6: Write `src/app/Reporter.cpp`**

```cpp
#include "app/Reporter.hpp"

#include <ostream>

namespace vrp::app {

Reporter::Reporter(std::ostream& out, std::size_t generations, std::size_t interval,
                   bool quiet)
    : out_(out),
      generations_(generations),
      interval_(interval == 0 ? 1 : interval),
      quiet_(quiet) {}

void Reporter::onGeneration(std::size_t generation, double bestDistance) const {
    if (quiet_) {
        return;
    }
    const bool onInterval = (generation + 1) % interval_ == 0;
    const bool isFinal = (generation + 1) == generations_;
    if (!onInterval && !isFinal) {
        return;
    }
    out_ << "Generation " << (generation + 1) << ": best distance = " << bestDistance
         << '\n';
}

void Reporter::onResult(const RunResult& result) const {
    out_ << "\nBest route found: depot -> ";
    for (int location : result.bestRoute) {
        out_ << location << " -> ";
    }
    out_ << "depot\n";
    // "Best found", not proven optimal — this is a stochastic heuristic.
    out_ << "Best distance:  " << result.bestDistance << '\n';
    out_ << "Generations:    " << result.generationsRun << '\n';
    out_ << "Elapsed:        " << result.elapsedSeconds << " s\n";
}

}  // namespace vrp::app
```

- [ ] **Step 7: Write `src/app/main.cpp`**

```cpp
#include <iostream>
#include <string_view>
#include <vector>

#include "app/Cli.hpp"
#include "app/Reporter.hpp"
#include "vrp/Executor.hpp"
#include "vrp/Problem.hpp"
#include "vrp/Solver.hpp"
#include "vrp/Strategy.hpp"

int main(int argc, char** argv) {
    std::vector<std::string_view> args(argv + 1, argv + argc);
    const vrp::app::ParseResult parsed = vrp::app::parseArgs(args);

    if (!parsed.ok) {
        std::cerr << "error: " << parsed.error << "\n\n" << vrp::app::usage();
        return 2;
    }
    if (parsed.options.helpRequested) {
        std::cout << vrp::app::usage();
        return 0;
    }

    const vrp::app::Options& options = parsed.options;
    const vrp::Problem problem = vrp::Problem::defaultInstance();

    vrp::Solver solver(problem, options.params, vrp::makeStrategy(options.strategy),
                       vrp::makeExecutor(options.threads));

    const vrp::app::Reporter reporter(std::cout, options.params.generations, 10,
                                      options.quiet);
    const vrp::RunResult result = solver.run(
        [&](std::size_t generation, double best) {
            reporter.onGeneration(generation, best);
        });
    reporter.onResult(result);

    return 0;
}
```

- [ ] **Step 8: Add the app targets and delete the old sources**

Append to `src/CMakeLists.txt`:

```cmake
add_library(vrp_app STATIC
  app/Cli.cpp
  app/Reporter.cpp
)
target_include_directories(vrp_app PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(vrp_app PUBLIC vrp_core)
target_link_libraries(vrp_app PRIVATE vrp_warnings)

add_executable(vrp_ga app/main.cpp)
target_link_libraries(vrp_ga PRIVATE vrp_app vrp_warnings)
```

In `tests/CMakeLists.txt`, add `vrp_app` to `target_link_libraries(vrp_tests ...)`.

```bash
git rm main.cpp main_parallel.cpp
```

- [ ] **Step 9: Build and run tests**

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug --output-on-failure
```

Expected: all tests PASS, including `[cli]`.

- [ ] **Step 10: Run the program for real**

```bash
cmake --build --preset release
./build/release/src/vrp_ga --help
./build/release/src/vrp_ga --population 5000 --generations 50
./build/release/src/vrp_ga --strategy generational --population 5000 --generations 50 --threads 1
./build/release/src/vrp_ga --strategy generational --population 5000 --generations 50 --threads 8
```

Add `.exe` on Windows. Expected: help prints; runs report generations 10, 20, … **and 50**; the two generational runs report an identical best distance with different elapsed times.

- [ ] **Step 11: Stage and request verification**

```bash
git add -A && git diff --cached --stat
```

---

### Task 11: README rewrite and final verification

**Files:**
- Modify: `README.md`
- Verify: everything

**Interfaces:**
- Consumes: the finished program. Adds no code.

- [ ] **Step 1: Capture real numbers to quote**

Do not invent benchmark figures. Run these and use the actual output:

```bash
./build/release/src/vrp_ga --strategy generational --population 20000 --generations 100 --threads 1 --quiet
./build/release/src/vrp_ga --strategy generational --population 20000 --generations 100 --threads 8 --quiet
```

Record both elapsed times and confirm the best distances match exactly.

- [ ] **Step 2: Rewrite `README.md`**

Rewrite these sections; do not patch around the old content:

- **Title and intro** — drop "MPI, OpenMP, and Intel TBB". The project is now a C++20 genetic algorithm with a std::thread execution backend.
- **Highlights** — permutation encoding, tournament selection, order crossover, swap mutation, two selectable strategies, reproducible results independent of thread count.
- **Requirements** — CMake 3.24+, a C++20 compiler, Ninja optional. Catch2 fetched automatically. No MPI, no TBB, no OpenMP.
- **Build and run** — the preset commands only:

  ```bash
  cmake --preset release
  cmake --build --preset release
  ./build/release/src/vrp_ga --help
  ```

- **Configuration** — a table of the ten CLI flags with defaults, replacing the "edit the source and recompile" instructions.
- **Repository structure** — the new tree.
- **Benchmarking** — same binary, same `--seed`, vary `--threads`; note that identical best distances across thread counts is the expected and verified behaviour. State plainly that steady-state does not parallelise its generation loop.
- **Testing** — `ctest --preset debug --output-on-failure`; mention that sanitizer presets target Linux/CI and that MinGW-w64 provides no TSan.

Remove entirely: the manual `numLocations = 19` correction, the MPI/TBB build recipe, the parallel-prototype comparison table, and the resolved entries under "Current limitations".

Keep: that this is a stochastic heuristic, that "best found" is not "optimal", and that the model is single-depot single-vehicle Euclidean TSP without capacities or time windows.

- [ ] **Step 3: Verify every command in the README actually runs**

Execute each block from the rewritten README verbatim in a clean clone:

```bash
# Use the session scratchpad directory, not /tmp.
CHECKOUT="$SCRATCHPAD/vrp-readme-check"
git clone . "$CHECKOUT" && cd "$CHECKOUT"
cmake --preset release && cmake --build --preset release
./build/release/src/vrp_ga --help
```

Set `SCRATCHPAD` to the session scratchpad path first. Delete the checkout when
the verification passes.

Expected: every documented command succeeds. Fix the README, not your memory of it, if any fail.

- [ ] **Step 4: Full verification sweep**

```bash
cmake --preset debug   && cmake --build --preset debug   && ctest --preset debug   --output-on-failure
cmake --preset release && cmake --build --preset release && ctest --preset release --output-on-failure
```

Expected: zero warnings (the debug preset sets `VRP_WARNINGS_AS_ERRORS=ON`), all tests pass in both configurations.

- [ ] **Step 5: Confirm the repository is clean**

```bash
git status --short
ls main main_parallel main.cpp main_parallel.cpp 2>/dev/null || echo "old files removed"
```

Expected: the four old files are gone; `build/` is ignored.

- [ ] **Step 6: Stage everything and request final verification**

```bash
git add -A
git status --short
git diff --cached --stat
```

Present the full change set to the user. Do not commit until they approve.

---

## Task Dependency Order

```
1 (scaffold)
├── 2 (Rng)
├── 3 (Problem)
└── 4 (Executor)
        └── 5 (Population)   [needs 2, 3, 4]
                └── 6 (Operators)   [needs 5]
                        └── 7 (Strategies)   [needs 6]
                                └── 8 (Solver)   [needs 7]
                                        └── 9 (Determinism)
                                                └── 10 (CLI/app)
                                                        └── 11 (README)
```

Tasks 2, 3, and 4 are independent of each other and may be done in any order after Task 1.
