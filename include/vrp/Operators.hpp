#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "vrp/Population.hpp"
#include "vrp/Rng.hpp"

namespace vrp::ops {

// RNG stream contract. Task 7 seeds one generator per work item and requires a
// threaded run to reproduce a serial one bit for bit, which makes the number of
// draws each operator takes part of its interface rather than an implementation
// detail. Two facts are easy to get wrong:
//
//  - Rng::below is Lemire's method WITH rejection, so the number of underlying
//    next() calls per below() varies with generator state. It is deterministic
//    -- the same state always rejects the same number of times -- so
//    bit-identity still holds. But nothing downstream may assume one next() per
//    below(), or count raw next() calls to predict a stream position.
//
//  - Draw counts per operator. Most are pinned by a test in
//    tests/test_operators.cpp that runs a mirror generator forward and compares
//    stream positions; the exception is called out below.
//      tournamentSelect  exactly max(tournamentSize, 1) below() calls. Stream-
//                        pinned for tournamentSize in {1, 2, 5, 16} only. The
//                        tournamentSize == 0 case is covered solely by
//                        "tournament selection treats size zero as a single
//                        draw", which compares the RETURNED INDEX against one
//                        mirror draw and never inspects the generator
//                        afterwards. An implementation that returned the right
//                        index and then consumed extra draws would pass it, so
//                        the one-draw count for size 0 is documented here and
//                        asserted nowhere. Nothing in the library reaches it --
//                        the CLI rejects --tournament 0 -- but a caller that
//                        did would not get stream alignment for free.
//      orderCrossover    exactly two below(n) calls, on every path;
//      swapMutate        zero draws when route.size() < 2; one unit() when the
//                        rate declines; one unit() and two below() when it
//                        fires.
//    swapMutate's zero-draw path is the one place a future with heterogeneous
//    route lengths would break stream alignment across a population: today
//    every route is routeLength() long, so the population consumes a uniform
//    number of draws, and that is the only reason the short-route early return
//    is invisible.

// Draws `tournamentSize` candidates and returns the fittest index.
// A tournamentSize of 0 is treated as 1.
//
// Precondition: population.size() > 0 -- there is no index to return
// otherwise, and the first draw would call Rng::below(0).
std::size_t tournamentSelect(const Population& population, std::size_t tournamentSize,
                             Rng& rng);

// Order crossover (OX): copy a random segment from p1, then fill the remaining
// positions with the genes of p2 in their original order.
//
// NOT canonical Davis OX. The free slots are filled left to right from position
// 0 and p2 is scanned from index 0, whereas Davis wraps both from high + 1.
// That is a different operator with different offspring, not a defect here, so
// do not "correct" it without changing the tests that pin this behaviour.
//
// `seenScratch` must have size >= customerCount + 1; genes are 1-based. The
// caller owns it so it can be allocated once per chunk instead of per child,
// and it is cleared on entry -- a buffer reused across children must not carry
// the previous child's mask, which would silently drop genes from this one.
//
// Preconditions:
//  - p1, p2 and child all have the same, non-zero size;
//  - p1 and p2 are permutations of the same gene set. The write cursor stays in
//    bounds only because "genes of p2 outside the segment" and "positions
//    outside the segment" are the same count. Parents drawn from different gene
//    sets (or one holding repeats) leave more genes to place than slots to hold
//    them, and the cursor runs off the end of `child` -- a silent heap write,
//    so it is asserted in debug builds alongside the size checks.
void orderCrossover(std::span<const int> p1, std::span<const int> p2,
                    std::span<int> child, std::vector<char>& seenScratch, Rng& rng);

// With probability `rate`, swaps two distinct positions. The second index is
// drawn from the remaining range, so the swap is never a silent no-op. Routes
// shorter than two elements are left alone and consume no draws at all.
void swapMutate(std::span<int> route, double rate, Rng& rng);

}  // namespace vrp::ops
