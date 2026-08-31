#include "vrp/Problem.hpp"

#include <cassert>
#include <cmath>
#include <utility>

namespace vrp {
namespace {

constexpr std::size_t kDepot = 0;

std::size_t toIndex(int gene) noexcept {
    assert(gene >= 0);
    return static_cast<std::size_t>(gene);
}

}  // namespace

Problem::Problem(std::vector<Point> locations) : locations_(std::move(locations)) {
    assert(!locations_.empty() && "Problem requires at least one location (the depot)");
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
    double total = distance(kDepot, toIndex(route.front()));
    for (std::size_t i = 1; i < route.size(); ++i) {
        total += distance(toIndex(route[i - 1]), toIndex(route[i]));
    }
    total += distance(toIndex(route.back()), kDepot);
    return total;
}

}  // namespace vrp
