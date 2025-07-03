cmake_minimum_required(VERSION 3.14)

# Add distributed cognitive architecture demo
add_executable(distributed-cognitive-demo
    demo.c
)

target_include_directories(distributed-cognitive-demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(distributed-cognitive-demo
    ggml-base
    ${CMAKE_THREAD_LIBS_INIT}
    m  # Math library for sin/cos functions
)

# Add comprehensive integration tests
add_executable(test-distributed-cognitive-integration
    test-integration.c
)

target_include_directories(test-distributed-cognitive-integration PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(test-distributed-cognitive-integration
    ggml-base
    ${CMAKE_THREAD_LIBS_INIT}
    m
)

# Set output directory
set_target_properties(distributed-cognitive-demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties(test-distributed-cognitive-integration PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Add compile definitions
target_compile_definitions(distributed-cognitive-demo PRIVATE
    _GNU_SOURCE
)

target_compile_definitions(test-distributed-cognitive-integration PRIVATE
    _GNU_SOURCE
)