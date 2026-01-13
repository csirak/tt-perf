cmake_minimum_required(VERSION 3.16)
project(tensordiff_torch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{TT_METAL_HOME})
    message(FATAL_ERROR "TT_METAL_HOME env var is required to find tt-metal headers")
endif()

set(TT_METAL_HOME $ENV{TT_METAL_HOME})

find_package(Torch REQUIRED)

add_executable(tensordiff_torch_compare
    torch_compare.cpp
)

target_include_directories(tensordiff_torch_compare PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/..
    ${TT_METAL_HOME}/tt_metal/api
)

target_link_libraries(tensordiff_torch_compare "${TORCH_LIBRARIES}")

add_executable(tensordiff_torch_random_init
    torch_random_init.cpp
)

target_include_directories(tensordiff_torch_random_init PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/..
    ${TT_METAL_HOME}/tt_metal/api
)
target_link_libraries(tensordiff_torch_random_init "${TORCH_LIBRARIES}")

add_executable(tensordiff_torch_layernorm_init
    torch_layernorm_init.cpp
)

target_include_directories(tensordiff_torch_layernorm_init PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/..
    ${TT_METAL_HOME}/tt_metal/api
)
target_link_libraries(tensordiff_torch_layernorm_init "${TORCH_LIBRARIES}")

add_executable(tensordiff_torch_transformer_layer_init
    torch_transformer_layer_init.cpp
)

target_include_directories(tensordiff_torch_transformer_layer_init PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/..
    ${TT_METAL_HOME}/tt_metal/api
)
target_link_libraries(tensordiff_torch_transformer_layer_init "${TORCH_LIBRARIES}")

add_executable(tensordiff_torch_transformer_layer_backward_init
    torch_transformer_layer_backward_init.cpp
)

target_include_directories(tensordiff_torch_transformer_layer_backward_init PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/..
    ${TT_METAL_HOME}/tt_metal/api
)
target_link_libraries(tensordiff_torch_transformer_layer_backward_init "${TORCH_LIBRARIES}")

add_executable(tensordiff_tensor_ops_test
    tests/tensor_ops_test.cpp
)

target_compile_definitions(tensordiff_tensor_ops_test PRIVATE TENSORDIFF_TEST_TORCH)
target_include_directories(tensordiff_tensor_ops_test PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/..
    ${TT_METAL_HOME}/tt_metal/api
)
target_link_libraries(tensordiff_tensor_ops_test "${TORCH_LIBRARIES}")

add_executable(tensordiff_tensor_dispatch_test
    tests/base_tensor_dispatch_test.cpp
)

target_compile_definitions(tensordiff_tensor_dispatch_test PRIVATE TENSORDIFF_TEST_TORCH)
target_include_directories(tensordiff_tensor_dispatch_test PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/..
    ${TT_METAL_HOME}/tt_metal/api
)
target_link_libraries(tensordiff_tensor_dispatch_test "${TORCH_LIBRARIES}")

# Torch may require this on some systems
set_property(TARGET tensordiff_torch_compare PROPERTY CXX_STANDARD 20)
set_property(TARGET tensordiff_torch_random_init PROPERTY CXX_STANDARD 20)
set_property(TARGET tensordiff_torch_layernorm_init PROPERTY CXX_STANDARD 20)
set_property(TARGET tensordiff_torch_transformer_layer_init PROPERTY CXX_STANDARD 20)
set_property(TARGET tensordiff_torch_transformer_layer_backward_init PROPERTY CXX_STANDARD 20)
set_property(TARGET tensordiff_tensor_ops_test PROPERTY CXX_STANDARD 20)
set_property(TARGET tensordiff_tensor_dispatch_test PROPERTY CXX_STANDARD 20)
