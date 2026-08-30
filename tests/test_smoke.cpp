#include <catch2/catch_test_macros.hpp>
#include "vrp/Version.hpp"

// Not a tautology: the strings on the right are compared against values the
// build system injects (src/CMakeLists.txt passes project(NAME/VERSION) to
// Version.cpp as VRP_PROJECT_NAME and VRP_PROJECT_VERSION), so this fails if
// the top-level project() declaration and the shipped identity diverge, or if
// the definitions stop reaching the library at all. Bumping the version in
// CMakeLists.txt is therefore expected to fail here until this literal follows.
TEST_CASE("project identity is reported", "[smoke]") {
    REQUIRE(vrp::projectName() == "vrp_ga");
    REQUIRE(vrp::projectVersion() == "1.0.0");
}
