include(FetchContent)

FetchContent_Declare(
    unity
    GIT_REPOSITORY https://github.com/ThrowTheSwitch/Unity.git
    GIT_TAG        v2.6.0
)
FetchContent_MakeAvailable(unity)

target_compile_definitions(unity PUBLIC UNITY_INCLUDE_DOUBLE UNITY_DOUBLE_PRECISION=1e-12)

set(PLIDAR_TEST_MODULES rng irf coates parity estimate crb)

foreach(mod IN LISTS PLIDAR_TEST_MODULES)
    add_executable(test_${mod} test_${mod}.c)
    target_link_libraries(test_${mod} PRIVATE plidar unity)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test_${mod} PRIVATE -Wall -Wextra)
        target_link_libraries(test_${mod} PRIVATE m)
    endif()
    add_test(NAME ${mod} COMMAND test_${mod})
endforeach()

# The parity test accumulates tens of millions of cycles to get its chi
# squared statistic down to a meaningful confidence, so it gets more room
# than the default.
set_tests_properties(parity PROPERTIES TIMEOUT 600)
