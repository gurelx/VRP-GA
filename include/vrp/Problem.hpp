#pragma once

// <cassert> in a public header has two costs a consumer inherits whether they
// want them or not, both accepted here deliberately rather than overlooked:
//
//  1. It defines the `assert` MACRO in every translation unit that includes
//     this header, so a consumer with an identifier or member named `assert`
//     will not compile. There is no way to keep the inline bounds checks below
//     and avoid this.
//
//  2. distance() is an inline function whose body VARIES WITH NDEBUG. If two
//     translation units in one program include this header with different
//     NDEBUG states -- an optimised library linked against a debug consumer, or
//     a single target that defines NDEBUG in only some of its sources -- they
//     emit two different definitions of the same inline function and the
//     linker keeps one arbitrarily. That is an ODR violation, not merely a
//     surprise about which checks run. It is latent rather than active: the
//     whole project is built one configuration at a time by the presets, so no
//     mixed-NDEBUG link exists today. Anyone adding a consumer with its own
//     build flags must keep NDEBUG consistent across the link.
//
// Do not "fix" this by making distance() non-inline; the assert-free lookup in
// the hot loop is the point of the precomputed matrix.
#include <cassert>
#include <cstddef>
#include <span>
#include <vector>

namespace vrp {

struct Point {
    int x{};
    int y{};
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

    // Derived from the coordinate list and never configurable, so no caller
    // can state a count that disagrees with the coordinates. The empty guard
    // keeps a moved-from Problem from reporting SIZE_MAX instead of 0.
    std::size_t customerCount() const noexcept {
        return locations_.empty() ? std::size_t{0} : locations_.size() - 1;
    }

    const std::vector<Point>& locations() const noexcept { return locations_; }

    double distance(std::size_t a, std::size_t b) const noexcept {
        // An out-of-range index would silently alias a valid cell of the flat
        // n*n block and return a plausible wrong distance, so fault instead.
        assert(a < locations_.size());
        assert(b < locations_.size());
        return matrix_[a * locations_.size() + b];
    }

    // Depot -> route[0] -> ... -> route[n-1] -> depot.
    double routeDistance(std::span<const int> route) const noexcept;

private:
    std::vector<Point> locations_;
    std::vector<double> matrix_;  // locationCount^2, row-major
};

}  // namespace vrp
