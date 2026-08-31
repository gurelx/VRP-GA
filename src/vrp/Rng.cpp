#include "vrp/Rng.hpp"

#include <bit>

namespace vrp {
namespace {

constexpr std::uint64_t kGolden = 0x9E3779B97F4A7C15ULL;

std::uint64_t mix(std::uint64_t z) noexcept {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// The all-zero state is xoshiro's fixed point: it would emit 0 forever. Every
// construction path must run this.
void ensureNonZero(std::array<std::uint64_t, 4>& s) noexcept {
    if (s[0] == 0 && s[1] == 0 && s[2] == 0 && s[3] == 0) {
        s[0] = kGolden;
    }
}

}  // namespace

std::uint64_t splitmix64(std::uint64_t& state) noexcept {
    state += kGolden;
    return mix(state);
}

std::uint64_t mixSeed(std::uint64_t base, std::uint64_t a, std::uint64_t b) noexcept {
    std::uint64_t x = base + kGolden * (a + 1ULL);
    x = mix(x);
    x += kGolden * (b + 1ULL);
    return mix(x);
}

Rng::Rng(std::uint64_t seed) noexcept {
    std::uint64_t state = seed;
    for (auto& word : s_) {
        word = splitmix64(state);
    }
    ensureNonZero(s_);
}

Rng Rng::fromState(std::array<std::uint64_t, 4> state) noexcept {
    Rng rng;
    rng.s_ = state;
    ensureNonZero(rng.s_);
    return rng;
}

std::uint64_t Rng::next() noexcept {
    const std::uint64_t result = std::rotl(s_[1] * 5ULL, 7) * 9ULL;
    const std::uint64_t t = s_[1] << 17;
    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];
    s_[2] ^= t;
    s_[3] = std::rotl(s_[3], 45);
    return result;
}

std::uint32_t Rng::below(std::uint32_t bound) noexcept {
    // Lemire's nearly-divisionless bounded generation: rejects the short tail
    // that `next() % bound` would instead fold onto the low buckets.
    std::uint64_t product =
        static_cast<std::uint64_t>(static_cast<std::uint32_t>(next() >> 32)) * bound;
    auto low = static_cast<std::uint32_t>(product);
    if (low < bound) {
        const std::uint32_t threshold = (0U - bound) % bound;
        while (low < threshold) {
            product =
                static_cast<std::uint64_t>(static_cast<std::uint32_t>(next() >> 32)) * bound;
            low = static_cast<std::uint32_t>(product);
        }
    }
    return static_cast<std::uint32_t>(product >> 32);
}

double Rng::unit() noexcept {
    return static_cast<double>(next() >> 11) * 0x1.0p-53;
}

}  // namespace vrp
