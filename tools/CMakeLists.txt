# Built alongside the shared object because it exists for the Python binding:
# check_bindings.py runs it and compares its output against what ctypes
# computes from the declarations in plidar.py.
add_executable(plidar_layout_probe layout_probe.c)
target_link_libraries(plidar_layout_probe PRIVATE plidar)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(plidar_layout_probe PRIVATE -Wall -Wextra -Wpedantic)
endif()
