cmake_minimum_required(VERSION 3.16)
project(tensordiffgrad_torch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{TT_METAL_HOME})
    message(FATAL_ERROR "TT_METAL_HOME env var is required to find tt-metal headers")
endif()

set(TT_METAL_HOME $ENV{TT_METAL_HOME})

find_package(Torch REQUIRED)

add_executable(tensordiffgrad_torch_ops_test
    tests/backend_ops_test.cpp
)

target_compile_definitions(tensordiffgrad_torch_ops_test PRIVATE TENSORDIFF_TEST_TORCH)
target_include_directories(tensordiffgrad_torch_ops_test PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/..
    ${TT_METAL_HOME}/tt_metal/api
)
target_link_libraries(tensordiffgrad_torch_ops_test "${TORCH_LIBRARIES}")

set_property(TARGET tensordiffgrad_torch_ops_test PROPERTY CXX_STANDARD 20)

add_executable(tensordiffgrad_torch_ffn_test
    tests/ffn_test_torch.cpp
)

target_include_directories(tensordiffgrad_torch_ffn_test PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/..
    ${TT_METAL_HOME}/tt_metal/api
)
target_link_libraries(tensordiffgrad_torch_ffn_test "${TORCH_LIBRARIES}")
set_property(TARGET tensordiffgrad_torch_ffn_test PROPERTY CXX_STANDARD 20)
