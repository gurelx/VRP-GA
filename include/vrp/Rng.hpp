#pragma once

#include <array>
#include <cstdint>

namespace vrp {

// Domain tag for initial-population seeding: mixSeed folds in `a + 1`, so this
// is the one domain whose term vanishes and it cannot collide with a generation.
inline constexpr std::uint64_t kInitDomain = 0xFFFFFFFFFFFFFFFFULL;

std::uint64_t splitmix64(std::uint64_t& state) noexcept;

// Derives an independent seed from (base, a, b). Used as
// mixSeed(seed, kInitDomain, itemIndex) or mixSeed(seed, generation, itemIndex).
std::uint64_t mixSeed(std::uint64_t base, std::uint64_t a, std::uint64_t b) noexcept;

// xoshiro256**: cheap to construct, which is what makes per-work-item seeding
// affordable.
class Rng {
public:
    explicit Rng(std::uint64_t seed) noexcept;

    // Adopts raw generator words, except the all-zero state -- xoshiro's fixed
    // point -- which is replaced, so fromState({}) does not round-trip.
    static Rng fromState(std::array<std::uint64_t, 4> state) noexcept;

    std::uint64_t next() noexcept;

    // Unbiased value in [0, bound). Precondition: bound >= 1.
    std::uint32_t below(std::uint32_t bound) noexcept;

    // Uniform double in [0, 1).
    double unit() noexcept;

private:
    Rng() noexcept = default;
    std::array<std::uint64_t, 4> s_{};
};

}  // namespace vrp
