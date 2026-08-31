#include "app/Reporter.hpp"

#include <ostream>

namespace vrp::app {

Reporter::Reporter(std::ostream& out, std::size_t generations, std::size_t interval,
                   bool quiet)
    : out_(out),
      generations_(generations),
      // interval == 0 would divide by zero; report every generation instead.
      interval_(interval == 0 ? 1 : interval),
      quiet_(quiet) {}

void Reporter::onGeneration(std::size_t generation, double bestDistance) const {
    if (quiet_) {
        return;
    }
    const bool onInterval = (generation + 1) % interval_ == 0;
    const bool isFinal = (generation + 1) == generations_;
    if (!onInterval && !isFinal) {
        return;
    }
    out_ << "Generation " << (generation + 1) << ": best distance = " << bestDistance
         << '\n';
}

void Reporter::onResult(const RunResult& result) const {
    out_ << "\nBest route found: depot -> ";
    for (int location : result.bestRoute) {
        out_ << location << " -> ";
    }
    out_ << "depot\n";
    out_ << "Best distance:  " << result.bestDistance << '\n';
    out_ << "Generations:    " << result.generationsRun << '\n';
    out_ << "Elapsed:        " << result.elapsedSeconds << " s\n";
}

}  // namespace vrp::app
