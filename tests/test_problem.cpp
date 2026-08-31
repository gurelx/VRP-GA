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

TEST_CASE("a depot-only problem has no customers", "[problem]") {
    const std::vector<vrp::Point> pts{{0, 0}};
    const vrp::Problem p(pts);
    REQUIRE(p.locationCount() == 1);
    REQUIRE(p.customerCount() == 0);
    REQUIRE(p.distance(0, 0) == 0.0);
}

TEST_CASE("locations returns the coordinates the problem was built from", "[problem]") {
    const std::vector<vrp::Point> pts{{0, 0}, {3, 0}, {3, 4}};
    const vrp::Problem p(pts);
    REQUIRE(p.locations().size() == 3);
    REQUIRE(p.locations()[0].x == 0);
    REQUIRE(p.locations()[0].y == 0);
    REQUIRE(p.locations()[2].x == 3);
    REQUIRE(p.locations()[2].y == 4);

    const vrp::Problem d = vrp::Problem::defaultInstance();
    REQUIRE(d.locations().size() == d.locationCount());
    REQUIRE(d.locations().front().x == 0);
    REQUIRE(d.locations().front().y == 0);
    REQUIRE(d.locations().back().x == 7);
    REQUIRE(d.locations().back().y == 6);
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

// The unit-square fixture gives every leg length 1, so it cannot tell which
// endpoints an implementation used. This 3-4-5 tour makes all three distinct.
TEST_CASE("route distance sums each leg exactly once on an asymmetric tour", "[problem]") {
    // Depot (0,0) -> (3,0) -> (3,4) -> depot. Legs are 3, 4 and 5.
    const std::vector<vrp::Point> pts{{0, 0}, {3, 0}, {3, 4}};
    const vrp::Problem p(pts);
    const std::vector<int> route{1, 2};
    REQUIRE_THAT(p.routeDistance(route), WithinAbs(12.0, 1e-12));

    const std::vector<int> single{2};
    REQUIRE_THAT(p.routeDistance(single), WithinAbs(10.0, 1e-12));
}

// The 3-4-5 fixture, not the unit square: with equal legs the two directions
// coincide even for a broken cycle.
TEST_CASE("route distance is direction-independent", "[problem]") {
    const std::vector<vrp::Point> pts{{0, 0}, {3, 0}, {3, 4}};
    const vrp::Problem p(pts);
    const std::vector<int> forward{1, 2};
    const std::vector<int> backward{2, 1};
    REQUIRE_THAT(p.routeDistance(forward),
                 WithinAbs(p.routeDistance(backward), 1e-12));
}

TEST_CASE("an empty route has zero length", "[problem]") {
    const std::vector<vrp::Point> pts{{0, 0}, {5, 5}};
    const vrp::Problem p(pts);
    const std::vector<int> empty;
    REQUIRE(p.routeDistance(empty) == 0.0);
}
