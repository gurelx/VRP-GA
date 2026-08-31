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

// Whole-token match, so "-h" is not satisfied by the "-h" inside "--help".
bool mentionsFlag(const std::string& text, std::string_view flag) {
    const auto boundary = [](char c) { return c == ' ' || c == ',' || c == '\n'; };
    for (std::size_t at = text.find(flag); at != std::string::npos;
         at = text.find(flag, at + 1)) {
        const std::size_t after = at + flag.size();
        const bool leftOk = at == 0 || boundary(text[at - 1]);
        const bool rightOk = after == text.size() || boundary(text[after]);
        if (leftOk && rightOk) {
            return true;
        }
    }
    return false;
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

TEST_CASE("seed 0 draws a fresh nonzero seed on every call", "[cli]") {
    // "Nonzero" alone is satisfied by any constant, including
    // hardware_concurrency() -- exactly what a swapped-sentinel bug produces.
    // The specified behaviour is a draw per call, so two parses must disagree.
    const auto first = parse({"--seed", "0"});
    const auto second = parse({"--seed", "0"});
    REQUIRE(first.ok);
    REQUIRE(second.ok);
    REQUIRE(first.options.params.seed != 0);
    REQUIRE(second.options.params.seed != 0);
    REQUIRE(first.options.params.seed != second.options.params.seed);
}

TEST_CASE("help is requested and reported", "[cli]") {
    const auto result = parse({"--help"});
    REQUIRE(result.ok);
    REQUIRE(result.options.helpRequested);
    REQUIRE_FALSE(vrp::app::usage().empty());
}

// SUBSUMED by "an unknown flag is reported as unknown, not as a missing value";
// kept as the guard on the rejection itself.
TEST_CASE("unknown flags are rejected", "[cli]") {
    const auto result = parse({"--nonsense"});
    REQUIRE_FALSE(result.ok);
    REQUIRE(result.error.find("--nonsense") != std::string::npos);
}

// SUBSUMED by "a missing value on a known flag is reported as a missing value".
TEST_CASE("a missing value is rejected", "[cli]") {
    const auto result = parse({"--population"});
    REQUIRE_FALSE(result.ok);
}

// Only unparseable text: "nan" and "inf" DO parse, and are covered separately.
TEST_CASE("values that do not parse as a number are rejected", "[cli]") {
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

TEST_CASE("an unknown flag is reported as unknown, not as a missing value",
          "[cli]") {
    // Both diagnoses name the flag, so a case that only greps for "--nonsense"
    // passes against either.
    const auto trailing = parse({"--nonsense"});
    REQUIRE_FALSE(trailing.ok);
    REQUIRE(trailing.error.find("unknown option") != std::string::npos);

    const auto midline = parse({"--nonsense", "7", "--population", "10"});
    REQUIRE_FALSE(midline.ok);
    REQUIRE(midline.error.find("unknown option") != std::string::npos);
}

TEST_CASE("a missing value on a known flag is reported as a missing value",
          "[cli]") {
    const auto result = parse({"--population"});
    REQUIRE_FALSE(result.ok);
    REQUIRE(result.error.find("missing value") != std::string::npos);
    REQUIRE(result.error.find("--population") != std::string::npos);
}

TEST_CASE("NaN and infinity are rejected for mutation", "[cli]") {
    // from_chars accepts "nan", so this reaches the range check -- where a
    // naive `rate < 0.0 || rate > 1.0` lets it through, both comparisons being
    // false for NaN, and it goes on to behave as a mutation rate of 1.0.
    REQUIRE_FALSE(parse({"--mutation", "nan"}).ok);
    REQUIRE_FALSE(parse({"--mutation", "-nan"}).ok);
    REQUIRE_FALSE(parse({"--mutation", "NAN"}).ok);
    // Infinity is caught by the ordinary bound; pinned so a rewrite of the
    // check cannot lose it.
    REQUIRE_FALSE(parse({"--mutation", "inf"}).ok);
    REQUIRE_FALSE(parse({"--mutation", "-inf"}).ok);
}

TEST_CASE("usage warns that steady-state does not parallelise", "[cli]") {
    // "parallel" is the common root of every plausible rewording and appears
    // nowhere else in the text.
    const std::string text = vrp::app::usage();
    REQUIRE(text.find("parallel") != std::string::npos);
    REQUIRE(text.find("steady-state") != std::string::npos);
    REQUIRE(text.find("--threads") != std::string::npos);
}

TEST_CASE("usage documents every accepted flag", "[cli]") {
    const std::string text = vrp::app::usage();
    for (const std::string_view flag :
         {"--strategy", "--population", "--generations", "--mutation",
          "--tournament", "--elite", "--threads", "--seed", "--quiet", "--help",
          "-h"}) {
        INFO("flag: " << flag);
        REQUIRE(mentionsFlag(text, flag));
    }
}

TEST_CASE("-h is the short form of --help", "[cli]") {
    const auto result = parse({"-h"});
    REQUIRE(result.ok);
    REQUIRE(result.options.helpRequested);
}

TEST_CASE("the population-0 diagnosis names the population", "[cli]") {
    // --population 0 is also rejected incidentally by `eliteCount >=
    // populationSize`, so the explicit guard looks removable. It is not: if the
    // elite rule ever becomes generational-only, `--population 0 --strategy
    // steady-state` would be accepted and walk into the out-of-bounds read
    // documented in Solver.hpp.
    const auto result = parse({"--population", "0"});
    REQUIRE_FALSE(result.ok);
    REQUIRE(result.error.find("population") != std::string::npos);
    // The discriminating half: the elite diagnosis names the population too, so
    // without this the fallback could masquerade as the guard.
    REQUIRE(result.error.find("elite") == std::string::npos);
}

TEST_CASE("the elite diagnosis names both values", "[cli]") {
    // --elite defaults to 1, so `--population 1` alone trips the elite rule and
    // the message must show the numbers.
    const auto result = parse({"--population", "1"});
    REQUIRE_FALSE(result.ok);
    REQUIRE(result.error.find("--elite (1)") != std::string::npos);
    REQUIRE(result.error.find("--population (1)") != std::string::npos);
}
