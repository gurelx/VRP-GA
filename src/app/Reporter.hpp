#pragma once

#include <cstddef>
#include <iosfwd>

#include "vrp/Solver.hpp"

namespace vrp::app {

// Prints on a fixed interval and always on the final generation. Kept cheap:
// the progress callback runs inside Solver's timed region, so a chatty reporter
// inflates the runtime it reports.
class Reporter {
public:
    Reporter(std::ostream& out, std::size_t generations, std::size_t interval, bool quiet);

    // `generation` is Solver's 0-based index; the printed number is 1-based.
    void onGeneration(std::size_t generation, double bestDistance) const;
    void onResult(const RunResult& result) const;

private:
    std::ostream& out_;
    std::size_t generations_;
    std::size_t interval_;
    bool quiet_;
};

}  // namespace vrp::app
