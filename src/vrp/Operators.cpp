#include "vrp/Operators.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>

namespace vrp::ops {

std::size_t tournamentSelect(const Population& population, std::size_t tournamentSize,
                             Rng& rng) {
    assert(population.size() > 0 && "tournament selection needs a non-empty population");
    const auto populationSize = static_cast<std::uint32_t>(population.size());
    std::size_t best = rng.below(populationSize);
    double bestFitness = population.fitness(best);

    // The first draw is unconditional, so a size of 0 degenerates to 1 rather
    // than to "no candidate at all".
    const std::size_t draws = tournamentSize > 0 ? tournamentSize : 1;
    for (std::size_t i = 1; i < draws; ++i) {
        const std::size_t candidate = rng.below(populationSize);
        const double candidateFitness = population.fitness(candidate);
        // Fitness is a cost: lower wins.
        if (candidateFitness < bestFitness) {
            bestFitness = candidateFitness;
            best = candidate;
        }
    }
    return best;
}

void orderCrossover(std::span<const int> p1, std::span<const int> p2,
                    std::span<int> child, std::vector<char>& seenScratch, Rng& rng) {
    const std::size_t n = p1.size();
    assert(n > 0 && "order crossover needs a non-empty parent");
    assert(p2.size() == n && "parents must have equal length");
    assert(child.size() == n && "child must match the parents' length");

    // The buffer belongs to the caller and is reused across children, so a
    // stale mask from the previous child would make this one drop genes.
    std::fill(seenScratch.begin(), seenScratch.end(), char{0});

    auto low = rng.below(static_cast<std::uint32_t>(n));
    auto high = rng.below(static_cast<std::uint32_t>(n));
    if (low > high) {
        std::swap(low, high);
    }

    for (std::size_t i = low; i <= high; ++i) {
        const int gene = p1[i];
        assert(gene >= 0 && static_cast<std::size_t>(gene) < seenScratch.size() &&
               "seenScratch must be sized customerCount + 1");
        child[i] = gene;
        seenScratch[static_cast<std::size_t>(gene)] = 1;
    }

    std::size_t write = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const int gene = p2[i];
        assert(gene >= 0 && static_cast<std::size_t>(gene) < seenScratch.size() &&
               "seenScratch must be sized customerCount + 1");
        if (seenScratch[static_cast<std::size_t>(gene)] != 0) {
            continue;
        }
        if (write >= low && write <= high) {
            write = static_cast<std::size_t>(high) + 1;  // jump over the segment
        }
        assert(write < n && "parents must be permutations of the same gene set");
        child[write] = gene;
        ++write;
    }
}

void swapMutate(std::span<int> route, double rate, Rng& rng) {
    if (route.size() < 2) {
        return;
    }
    // `>=` rather than `>`: unit() is in [0, 1), so a rate of 0 must decline on
    // a draw of exactly 0.0, and a rate of 1 must fire on every draw.
    if (rng.unit() >= rate) {
        return;
    }
    const auto n = static_cast<std::uint32_t>(route.size());
    const std::uint32_t first = rng.below(n);
    // Draw from the remaining n-1 slots, then shift past `first`, so the two
    // indices can never coincide and the mutation is never a silent no-op.
    const std::uint32_t offset = rng.below(n - 1);
    const std::uint32_t second = offset >= first ? offset + 1 : offset;
    std::swap(route[first], route[second]);
}

}  // namespace vrp::ops
