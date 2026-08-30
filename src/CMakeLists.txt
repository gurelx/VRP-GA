add_library(vrp_core STATIC
  vrp/Version.cpp
  vrp/Rng.cpp
  vrp/Problem.cpp
  vrp/Executor.cpp
  vrp/Population.cpp
  vrp/Operators.cpp
  vrp/Strategy.cpp
  vrp/Solver.cpp
)
target_include_directories(vrp_core PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(vrp_core PUBLIC Threads::Threads)
target_link_libraries(vrp_core PRIVATE vrp_warnings)

# Cli and Reporter live here rather than beside main.cpp so that argument
# parsing is testable without spawning a subprocess; vrp_ga is then a trivial
# main. vrp_core still performs no I/O -- all printing lives in vrp_app.
add_library(vrp_app STATIC
  app/Cli.cpp
  app/Reporter.cpp
)
target_include_directories(vrp_app PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_link_libraries(vrp_app PUBLIC vrp_core)
target_link_libraries(vrp_app PRIVATE vrp_warnings)

add_executable(vrp_ga app/main.cpp)
target_link_libraries(vrp_ga PRIVATE vrp_app vrp_warnings)
