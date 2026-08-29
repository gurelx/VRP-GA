#pragma once

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
