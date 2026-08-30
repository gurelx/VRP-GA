#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <string_view>

#include "vrp/Config.hpp"

namespace vrp::app {

struct Options {
    GaParams params{};
    StrategyKind strategy = StrategyKind::SteadyState;
    std::size_t threads = 1;
    bool quiet = false;
    bool helpRequested = false;
};

struct ParseResult {
    Options options{};
    bool ok = true;
    std::string error;
};

// `args` excludes argv[0]. Resolves the --threads 0 and --seed 0 sentinels here
// so the core library never sees them.
ParseResult parseArgs(std::span<const std::string_view> args);

std::string usage();

}  // namespace vrp::app
