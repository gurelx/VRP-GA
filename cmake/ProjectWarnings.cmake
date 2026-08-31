add_library(vrp_warnings INTERFACE)

if(MSVC)
  target_compile_options(vrp_warnings INTERFACE /W4 /permissive-)
  if(VRP_WARNINGS_AS_ERRORS)
    target_compile_options(vrp_warnings INTERFACE /WX)
  endif()
else()
  target_compile_options(vrp_warnings INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
    -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wunused
    -Woverloaded-virtual -Wdouble-promotion)
  if(VRP_WARNINGS_AS_ERRORS)
    target_compile_options(vrp_warnings INTERFACE -Werror)
  endif()
endif()

