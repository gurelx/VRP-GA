#include "app/Cli.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <random>
#include <system_error>
#include <thread>
#include <utility>

namespace vrp::app {
namespace {

// Every flag that takes a value. Consulted before a value is demanded, so that
// an unknown flag in the last position is reported as unknown rather than as a
// missing value -- both messages name the flag, but only one of them sends the
// user looking for the right fix.
constexpr std::array<std::string_view, 8> kValueFlags{
    "--strategy", "--population", "--generations", "--mutation",
    "--tournament", "--elite", "--threads", "--seed",
};

bool parseSize(std::string_view text, std::size_t& out) {
    if (text.empty()) {
        return false;
    }
    const char* begin = text.data();
    const char* end = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(begin, end, out);
    return ec == std::errc{} && ptr == end;
}

bool parseU64(std::string_view text, std::uint64_t& out) {
    if (text.empty()) {
        return false;
    }
    const char* begin = text.data();
    const char* end = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(begin, end, out);
    return ec == std::errc{} && ptr == end;
}

bool parseDouble(std::string_view text, double& out) {
    if (text.empty()) {
        return false;
    }
    // from_chars for double is available in GCC 11+ and MSVC 19.24+.
    const char* begin = text.data();
    const char* end = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(begin, end, out);
    return ec == std::errc{} && ptr == end;
}

ParseResult fail(std::string message) {
    ParseResult result;
    result.ok = false;
    result.error = std::move(message);
    return result;
}

}  // namespace

std::string usage() {
    return
        "vrp_ga [options]\n"
        "\n"
        "  --strategy    steady-state | generational  (default: steady-state)\n"
        "  --population  N    population size         (default: 100000)\n"
        "  --generations N    iterations              (default: 100)\n"
        "  --mutation    R    swap probability, 0..1  (default: 0.2)\n"
        "  --tournament  N    candidates per parent   (default: 5)\n"
        "  --elite       N    survivors, generational (default: 1)\n"
        "  --threads     N    0 selects all cores     (default: 1)\n"
        "  --seed        N    0 draws a random seed   (default: 42)\n"
        "  --quiet            suppress per-generation output\n"
        "  --help, -h         show this message\n"
        "\n"
        "Note: steady-state replaces one individual per generation, so its\n"
        "generation loop does not parallelise. Use --strategy generational to\n"
        "see a speedup from --threads.\n";
}

ParseResult parseArgs(std::span<const std::string_view> args) {
    ParseResult result;
    Options& options = result.options;

    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view flag = args[i];

        if (flag == "--help" || flag == "-h") {
            options.helpRequested = true;
            return result;
        }
        if (flag == "--quiet") {
            options.quiet = true;
            continue;
        }

        if (std::ranges::find(kValueFlags, flag) == kValueFlags.end()) {
            return fail(std::string("unknown option: ").append(flag));
        }
        if (i + 1 >= args.size()) {
            return fail(std::string("missing value for ").append(flag));
        }
        const std::string_view value = args[++i];

        if (flag == "--strategy") {
            if (value == "steady-state") {
                options.strategy = StrategyKind::SteadyState;
            } else if (value == "generational") {
                options.strategy = StrategyKind::Generational;
            } else {
                return fail(std::string("unknown strategy: ").append(value));
            }
        } else if (flag == "--population") {
            if (!parseSize(value, options.params.populationSize)) {
                return fail(std::string("invalid --population: ").append(value));
            }
            // Population::bestIndex() asserts a non-empty population; in a
            // release build the empty case is an out-of-bounds read rather
            // than a diagnosable failure, and this is the only guard.
            if (options.params.populationSize == 0) {
                return fail("--population must be at least 1");
            }
        } else if (flag == "--generations") {
            if (!parseSize(value, options.params.generations)) {
                return fail(std::string("invalid --generations: ").append(value));
            }
        } else if (flag == "--mutation") {
            if (!parseDouble(value, options.params.mutationRate)) {
                return fail(std::string("invalid --mutation: ").append(value));
            }
            // Negated conjunction, not a disjunction of the failing halves:
            // from_chars accepts "nan" under chars_format::general, and NaN
            // makes `nan < 0.0` and `nan > 1.0` BOTH false, so the naive form
            // lets it through. It would then reach ops::swapMutate, whose
            // guard `rng.unit() >= rate` is likewise false for NaN -- the rate
            // would behave as 1.0 and every child would mutate. The CLI is the
            // only validator on that path.
            if (!(options.params.mutationRate >= 0.0 &&
                  options.params.mutationRate <= 1.0)) {
                return fail("--mutation must lie in [0, 1]");
            }
        } else if (flag == "--tournament") {
            if (!parseSize(value, options.params.tournamentSize)) {
                return fail(std::string("invalid --tournament: ").append(value));
            }
            if (options.params.tournamentSize == 0) {
                return fail("--tournament must be at least 1");
            }
        } else if (flag == "--elite") {
            if (!parseSize(value, options.params.eliteCount)) {
                return fail(std::string("invalid --elite: ").append(value));
            }
        } else if (flag == "--threads") {
            if (!parseSize(value, options.threads)) {
                return fail(std::string("invalid --threads: ").append(value));
            }
        } else if (flag == "--seed") {
            if (!parseU64(value, options.params.seed)) {
                return fail(std::string("invalid --seed: ").append(value));
            }
        } else {
            // Unreachable while kValueFlags and this chain agree; reached only
            // if a flag is added to one and not the other.
            return fail(std::string("unhandled option: ").append(flag));
        }
    }

    // Checked after the whole command line, not at --elite, so that the two
    // flags may be given in either order.
    if (options.params.eliteCount >= options.params.populationSize) {
        // Both values are named because --elite defaults to 1: a user who
        // typed only `--population 1` would otherwise be told off for a flag
        // they never used, with no hint of where the other 1 came from.
        return fail("--elite (" + std::to_string(options.params.eliteCount) +
                    ") must be smaller than --population (" +
                    std::to_string(options.params.populationSize) + ")");
    }

    if (options.threads == 0) {
        options.threads = std::max(1u, std::thread::hardware_concurrency());
    }
    if (options.params.seed == 0) {
        std::random_device device;
        options.params.seed = (static_cast<std::uint64_t>(device()) << 32) ^ device();
        if (options.params.seed == 0) {
            options.params.seed = 1;
        }
    }

    return result;
}

}  // namespace vrp::app
