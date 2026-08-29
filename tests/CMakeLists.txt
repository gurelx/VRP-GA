include(FetchContent)
FetchContent_Declare(Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.7.1
  FIND_PACKAGE_ARGS 3 NAMES Catch2)
FetchContent_MakeAvailable(Catch2)

list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
include(Catch)

add_executable(vrp_tests
  test_smoke.cpp
  test_rng.cpp
  test_problem.cpp
)
target_link_libraries(vrp_tests PRIVATE vrp_core Catch2::Catch2WithMain)
target_link_libraries(vrp_tests PRIVATE vrp_warnings)

get_filename_component(VRP_COMPILER_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
catch_discover_tests(vrp_tests DL_PATHS "${VRP_COMPILER_DIR}")
