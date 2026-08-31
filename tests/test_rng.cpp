#include <catch2/catch_test_macros.hpp>
#include <array>
#include <cstdint>
#include "vrp/Rng.hpp"

TEST_CASE("splitmix64 matches reference output for seed 0", "[rng]") {
    std::uint64_t state = 0;
    REQUIRE(vrp::splitmix64(state) == 0xE220A8397B1DCDAFULL);
}

TEST_CASE("xoshiro256** matches reference output for state {1,2,3,4}", "[rng]") {
    auto rng = vrp::Rng::fromState({1, 2, 3, 4});
    REQUIRE(rng.next() == 11520ULL);
    REQUIRE(rng.next() == 0ULL);
    REQUIRE(rng.next() == 1509978240ULL);
    // The fourth draw is not optional: the rotation amount first reaches the
    // output here, and any amount in 1..63 passes the first three.
    REQUIRE(rng.next() == 1215971899390074240ULL);
}

TEST_CASE("fromState survives the all-zero state", "[rng]") {
    auto rng = vrp::Rng::fromState({0, 0, 0, 0});
    bool sawNonZero = false;
    for (int i = 0; i < 10; ++i) {
        if (rng.next() != 0ULL) {
            sawNonZero = true;
        }
    }
    REQUIRE(sawNonZero);
}

TEST_CASE("the same seed reproduces the same stream", "[rng]") {
    // The parallel design rests on this: per-item seeding means an 8-thread run
    // must reproduce a 1-thread run bit for bit.
    vrp::Rng a(2024);
    vrp::Rng b(2024);
    for (int i = 0; i < 500; ++i) {
        REQUIRE(a.next() == b.next());
    }

    vrp::Rng c(2024);
    vrp::Rng d(2025);
    int agreements = 0;
    for (int i = 0; i < 500; ++i) {
        if (c.next() == d.next()) {
            ++agreements;
        }
    }
    REQUIRE(agreements == 0);
}

TEST_CASE("below stays within bounds", "[rng]") {
    vrp::Rng rng(12345);
    for (int i = 0; i < 10000; ++i) {
        REQUIRE(rng.below(7) < 7u);
    }
    REQUIRE(rng.below(1) == 0u);
}

TEST_CASE("below is free of modulo bias", "[rng]") {
    // kBound must exceed 2^31 or this is vacuous: below that, naive
    // `next() % bound` is skewed too slightly for any sample size to resolve.
    constexpr std::uint32_t kBound = 3u << 30;
    constexpr int kDraws = 300000;
    int lowHalf = 0;
    vrp::Rng rng(99);
    for (int i = 0; i < kDraws; ++i) {
        const std::uint32_t v = rng.below(kBound);
        REQUIRE(v < kBound);
        if (v < kBound / 2) {
            ++lowHalf;
        }
    }
    const double fraction = static_cast<double>(lowHalf) / static_cast<double>(kDraws);
    REQUIRE(fraction > 0.49);
    REQUIRE(fraction < 0.51);
}

TEST_CASE("unit lies in the half-open unit interval", "[rng]") {
    // Name avoids an unbalanced '[': catch_discover_tests would otherwise merge
    // this and every following test case into one bogus CTest entry.
    constexpr int kDraws = 10000;
    vrp::Rng rng(7);
    double lo = 2.0;
    double hi = -1.0;
    double sum = 0.0;
    for (int i = 0; i < kDraws; ++i) {
        const double u = rng.unit();
        REQUIRE(u >= 0.0);
        REQUIRE(u < 1.0);
        if (u < lo) {
            lo = u;
        }
        if (u > hi) {
            hi = u;
        }
        sum += u;
    }
    // Range checks alone are satisfied by a generator confined to [0, 0.5) -- a
    // `>> 12` typo, say -- so assert the interval is actually spanned.
    REQUIRE(hi > 0.99);
    REQUIRE(lo < 0.01);
    const double mean = sum / static_cast<double>(kDraws);
    REQUIRE(mean > 0.49);
    REQUIRE(mean < 0.51);
}

TEST_CASE("mixSeed separates seed domains", "[rng]") {
    REQUIRE(vrp::mixSeed(42, vrp::kInitDomain, 5) != vrp::mixSeed(42, 0, 5));
    REQUIRE(vrp::mixSeed(42, 0, 0) != vrp::mixSeed(42, 0, 1));
    REQUIRE(vrp::mixSeed(42, 0, 0) != vrp::mixSeed(42, 1, 0));
    REQUIRE(vrp::mixSeed(42, 0, 0) != vrp::mixSeed(43, 0, 0));
}
