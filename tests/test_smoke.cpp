#include <catch2/catch_test_macros.hpp>
#include "vrp/Version.hpp"

// Not a tautology: these are compared against values src/CMakeLists.txt injects
// from project(), so bumping the version fails here until this literal follows.
TEST_CASE("project identity is reported", "[smoke]") {
    REQUIRE(vrp::projectName() == "vrp_ga");
    REQUIRE(vrp::projectVersion() == "1.0.0");
}
