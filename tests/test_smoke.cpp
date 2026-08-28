#include <catch2/catch_test_macros.hpp>
#include "vrp/Version.hpp"

TEST_CASE("project identity is reported", "[smoke]") {
    REQUIRE(vrp::projectName() == "vrp_ga");
    REQUIRE(vrp::projectVersion() == "1.0.0");
}
