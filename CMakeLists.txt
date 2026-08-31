cmake_minimum_required(VERSION 3.24)
project(vrp_ga VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(VRP_BUILD_TESTS "Build the test suite" ${PROJECT_IS_TOP_LEVEL})
option(VRP_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ProjectWarnings)

find_package(Threads REQUIRED)

add_subdirectory(src)

if(VRP_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
