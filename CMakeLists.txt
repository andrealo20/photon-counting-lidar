cmake_minimum_required(VERSION 3.16)
project(plidar VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(PLIDAR_BUILD_TESTS "Build unit tests" ON)
option(PLIDAR_WERROR      "Treat warnings as errors" ON)
option(PLIDAR_BENCH       "Build the throughput report" ON)
option(PLIDAR_SHARED      "Also build a shared library for the Python tools" ON)

add_library(plidar
    src/status.c
    src/rng.c
    src/irf.c
    src/sim.c
    src/coates.c
    src/estimate.c
    src/crb.c
)

target_include_directories(plidar PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# erfc, log1p, expm1 and lgamma all live in libm on the platforms this
# builds on. Declaring it on the library rather than on each executable puts
# it after libplidar on every consumer's link line, which is the order a
# static link needs.
include(CheckLibraryExists)
check_library_exists(m cos "" PLIDAR_HAVE_LIBM)
if(PLIDAR_HAVE_LIBM)
    target_link_libraries(plidar PUBLIC m)
endif()

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(plidar PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wstrict-prototypes
        -Wdouble-promotion -Wfloat-equal)
    if(PLIDAR_WERROR)
        target_compile_options(plidar PRIVATE -Werror)
    endif()
endif()

# The sweeps and the figures live in Python, and they drive this same code
# through ctypes rather than a second implementation. One shared object is
# all that takes, and it keeps the numbers in the README traceable to the
# library the tests run against.
if(PLIDAR_SHARED)
    add_library(plidar_shared SHARED
        src/status.c
        src/rng.c
        src/irf.c
        src/sim.c
        src/coates.c
        src/estimate.c
        src/crb.c
    )
    target_include_directories(plidar_shared PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
    set_target_properties(plidar_shared PROPERTIES OUTPUT_NAME plidar
                          POSITION_INDEPENDENT_CODE ON)
    if(PLIDAR_HAVE_LIBM)
        target_link_libraries(plidar_shared PUBLIC m)
    endif()
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(plidar_shared PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

if(PLIDAR_BENCH)
    add_subdirectory(bench)
endif()

if(PLIDAR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
