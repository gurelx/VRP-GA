#include <catch2/catch_test_macros.hpp>

#include <sstream>
#include <string>
#include <vector>

#include "app/Reporter.hpp"
#include "vrp/Solver.hpp"

namespace {

// Drives callbacks the way Solver does: 0-based indices 0 .. generations-1.
std::string reportRun(std::size_t generations, std::size_t interval, bool quiet) {
    std::ostringstream out;
    const vrp::app::Reporter reporter(out, generations, interval, quiet);
    for (std::size_t generation = 0; generation < generations; ++generation) {
        reporter.onGeneration(generation, 1.5);
    }
    return out.str();
}

std::size_t countOf(const std::string& haystack, const std::string& needle) {
    std::size_t count = 0;
    for (std::size_t at = haystack.find(needle); at != std::string::npos;
         at = haystack.find(needle, at + needle.size())) {
        ++count;
    }
    return count;
}

}  // namespace

TEST_CASE("generation numbers are printed 1-based", "[reporter]") {
    const std::string output = reportRun(100, 10, false);
    REQUIRE(output.find("Generation 10: best distance = 1.5\n") != std::string::npos);
    REQUIRE(output.find("Generation 9:") == std::string::npos);
    REQUIRE(output.find("Generation 0:") == std::string::npos);
    REQUIRE(output.find("Generation 100:") != std::string::npos);
    REQUIRE(output.find("Generation 99:") == std::string::npos);
}

TEST_CASE("only interval generations are reported", "[reporter]") {
    const std::string output = reportRun(30, 10, false);
    REQUIRE(countOf(output, "Generation ") == 3);
    REQUIRE(output.find("Generation 10:") != std::string::npos);
    REQUIRE(output.find("Generation 20:") != std::string::npos);
    REQUIRE(output.find("Generation 30:") != std::string::npos);
}

TEST_CASE("the final generation is reported even off the interval", "[reporter]") {
    const std::string output = reportRun(25, 10, false);
    REQUIRE(output.find("Generation 25: best distance = 1.5\n") != std::string::npos);
    REQUIRE(countOf(output, "Generation ") == 3);
}

TEST_CASE("a run shorter than the interval still reports its last generation",
          "[reporter]") {
    const std::string output = reportRun(3, 10, false);
    REQUIRE(countOf(output, "Generation ") == 1);
    REQUIRE(output.find("Generation 3:") != std::string::npos);
}

TEST_CASE("a final generation on the interval is reported exactly once",
          "[reporter]") {
    const std::string output = reportRun(20, 10, false);
    REQUIRE(countOf(output, "Generation 20:") == 1);
    REQUIRE(countOf(output, "Generation ") == 2);
}

TEST_CASE("a zero interval reports every generation instead of dividing by zero",
          "[reporter]") {
    const std::string output = reportRun(4, 0, false);
    REQUIRE(countOf(output, "Generation ") == 4);
    REQUIRE(output.find("Generation 1:") != std::string::npos);
    REQUIRE(output.find("Generation 4:") != std::string::npos);
}

TEST_CASE("quiet suppresses per-generation output", "[reporter]") {
    REQUIRE(reportRun(100, 10, true).empty());
}

TEST_CASE("quiet does not suppress the final result", "[reporter]") {
    vrp::RunResult result;
    result.bestRoute = {3, 1, 2};
    result.bestDistance = 42.5;
    result.generationsRun = 7;
    result.elapsedSeconds = 0.25;

    std::ostringstream out;
    const vrp::app::Reporter reporter(out, 100, 10, true);
    reporter.onResult(result);

    const std::string output = out.str();
    REQUIRE(output.find("depot -> 3 -> 1 -> 2 -> depot") != std::string::npos);
    REQUIRE(output.find("Best distance:  42.5") != std::string::npos);
    REQUIRE(output.find("Generations:    7") != std::string::npos);
    REQUIRE(output.find("Elapsed:        0.25 s") != std::string::npos);
}
