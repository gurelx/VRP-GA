#pragma once

// <cassert> here defines the assert macro for every consumer, and distance()'s
// inline body varies with NDEBUG, so a mixed-NDEBUG link is an ODR violation.
#include <cassert>
#include <cstddef>
#include <span>
#include <vector>

namespace vrp {

struct Point {
    int x{};
    int y{};
};

class Problem {
public:
    static Problem defaultInstance();

    // locations[0] is the depot. Requires at least one location.
    explicit Problem(std::vector<Point> locations);

    std::size_t locationCount() const noexcept { return locations_.size(); }

    // The empty guard keeps a moved-from Problem from reporting SIZE_MAX.
    std::size_t customerCount() const noexcept {
        return locations_.empty() ? std::size_t{0} : locations_.size() - 1;
    }

    const std::vector<Point>& locations() const noexcept { return locations_; }

    double distance(std::size_t a, std::size_t b) const noexcept {
        assert(a < locations_.size());
        assert(b < locations_.size());
        return matrix_[a * locations_.size() + b];
    }

    // Depot -> route[0] -> ... -> route[n-1] -> depot.
    double routeDistance(std::span<const int> route) const noexcept;

private:
    std::vector<Point> locations_;
    std::vector<double> matrix_;
};

}  // namespace vrp
