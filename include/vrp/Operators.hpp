#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "vrp/Population.hpp"
#include "vrp/Rng.hpp"

namespace vrp::ops {

// Determinism comes from per-item seeding: each work item builds its own Rng
// from its own absolute index, so draws never cross items. Draw counts are
// still part of these operators' interface, because the tests mirror the
// streams to pin them -- swapMutate's varies (0, 1, or 3; see below).

// Returns the fittest of `tournamentSize` drawn indices; 0 is treated as 1.
// Precondition: population.size() > 0.
std::size_t tournamentSelect(const Population& population, std::size_t tournamentSize,
                             Rng& rng);

// Order crossover: copy a random segment from p1, then fill the remaining
// positions from p2 in order. NOT canonical Davis OX -- free slots fill left to
// right from 0 and p2 is scanned from 0, where Davis wraps both from high + 1.
// That is a deliberately different operator; do not "correct" it.
//
// Preconditions: p1, p2 and child share the same non-zero size, and p1 and p2
// are permutations of the same gene set -- otherwise the write cursor runs off
// the end of `child`. `seenScratch` must hold customerCount + 1 entries (genes
// are 1-based); it is cleared on entry, so one buffer can serve a whole chunk.
void orderCrossover(std::span<const int> p1, std::span<const int> p2,
                    std::span<int> child, std::vector<char>& seenScratch, Rng& rng);

// With probability `rate`, swaps two distinct positions -- never a silent
// no-op. Routes shorter than two are left alone and consume no draws at all.
void swapMutate(std::span<int> route, double rate, Rng& rng);

}  // namespace vrp::ops
