#include "vrp/Version.hpp"

// Supplied by target_compile_definitions in src/CMakeLists.txt from
// project(... VERSION ...). Deliberately a hard error rather than a fallback
// literal: a default here would let the build system's value silently stop
// reaching the binary, and tests/test_smoke.cpp would go back to comparing a
// literal against a copy of itself.
#if !defined(VRP_PROJECT_NAME) || !defined(VRP_PROJECT_VERSION)
#error "VRP_PROJECT_NAME and VRP_PROJECT_VERSION must be defined by the build"
#endif

namespace vrp {
std::string_view projectName() noexcept { return VRP_PROJECT_NAME; }
std::string_view projectVersion() noexcept { return VRP_PROJECT_VERSION; }
}  // namespace vrp
