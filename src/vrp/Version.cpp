#include "vrp/Version.hpp"

// Supplied by the build. Deliberately a hard error rather than a fallback
// literal, which would let the injected value silently stop reaching the binary.
#if !defined(VRP_PROJECT_NAME) || !defined(VRP_PROJECT_VERSION)
#error "VRP_PROJECT_NAME and VRP_PROJECT_VERSION must be defined by the build"
#endif

namespace vrp {
std::string_view projectName() noexcept { return VRP_PROJECT_NAME; }
std::string_view projectVersion() noexcept { return VRP_PROJECT_VERSION; }
}  // namespace vrp
