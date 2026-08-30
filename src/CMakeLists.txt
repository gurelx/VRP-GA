add_library(vrp_core STATIC
  vrp/Version.cpp
  vrp/Rng.cpp
  vrp/Problem.cpp
  vrp/Executor.cpp
  vrp/Population.cpp
  vrp/Operators.cpp
  vrp/Strategy.cpp
)
target_include_directories(vrp_core PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(vrp_core PUBLIC Threads::Threads)
target_link_libraries(vrp_core PRIVATE vrp_warnings)
