#pragma once

#include <cstddef>
#include <iosfwd>

#include "vrp/Solver.hpp"

namespace vrp::app {

// Prints progress on a fixed interval and always on the final generation --
// the original code reported every tenth generation and so never printed the
// last one.
//
// Kept deliberately cheap: Solver::elapsedSeconds is measured with the
// progress callback inside the timed region, so a chatty reporter inflates the
// runtime it is reporting.
class Reporter {
public:
    Reporter(std::ostream& out, std::size_t generations, std::size_t interval, bool quiet);

    // `generation` is the 0-based index Solver reports; the printed number is
    // 1-based, so a 50-generation run ends at "Generation 50".
    void onGeneration(std::size_t generation, double bestDistance) const;
    void onResult(const RunResult& result) const;

private:
    std::ostream& out_;
    std::size_t generations_;
    std::size_t interval_;
    bool quiet_;
};

}  // namespace vrp::app
