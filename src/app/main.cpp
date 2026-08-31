#include <exception>
#include <iostream>
#include <string_view>
#include <vector>

#include "app/Cli.hpp"
#include "app/Reporter.hpp"
#include "vrp/Executor.hpp"
#include "vrp/Problem.hpp"
#include "vrp/Solver.hpp"
#include "vrp/Strategy.hpp"

int main(int argc, char** argv) {
    try {
        const std::vector<std::string_view> args(argv + 1, argv + argc);
        const vrp::app::ParseResult parsed = vrp::app::parseArgs(args);

        if (!parsed.ok) {
            std::cerr << "error: " << parsed.error << "\n\n" << vrp::app::usage();
            return 2;
        }
        if (parsed.options.helpRequested) {
            std::cout << vrp::app::usage();
            return 0;
        }

        const vrp::app::Options& options = parsed.options;
        const vrp::Problem problem = vrp::Problem::defaultInstance();

        vrp::Solver solver(problem, options.params, vrp::makeStrategy(options.strategy),
                           vrp::makeExecutor(options.threads));

        std::cout << "Strategy:       " << solver.strategy().name() << '\n';
        std::cout << "Threads:        " << solver.executor().threadCount() << '\n';
        std::cout << "Seed:           " << options.params.seed << '\n';

        const vrp::app::Reporter reporter(std::cout, options.params.generations, 10,
                                          options.quiet);
        // An absent callback under --quiet, not one that returns early: Solver only
        // computes population.fitness(population.bestIndex()) -- an O(populationSize)
        // scan inside the timed region -- when a callback is installed.
        const vrp::RunResult result =
            options.quiet ? solver.run()
                          : solver.run([&](std::size_t generation, double best) {
                                reporter.onGeneration(generation, best);
                            });
        reporter.onResult(result);

        return 0;
    } catch (const std::exception& e) {
        // Allocation for the population buffers is the reachable case; the CLI
        // cannot pre-validate it against available memory.
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
