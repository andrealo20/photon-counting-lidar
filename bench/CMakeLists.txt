add_executable(plidar_bench bench.c)
target_link_libraries(plidar_bench PRIVATE plidar)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(plidar_bench PRIVATE -Wall -Wextra)
    # clock_gettime needs the POSIX feature test macro under strict C99.
    target_compile_definitions(plidar_bench PRIVATE _POSIX_C_SOURCE=199309L)
endif()
