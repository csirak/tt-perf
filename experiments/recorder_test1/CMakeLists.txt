cmake_minimum_required(VERSION 3.16)
project(recorder_test1_torch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{TT_METAL_HOME})
    message(FATAL_ERROR "TT_METAL_HOME env var is required to find tt-metal headers")
endif()

set(TT_METAL_HOME $ENV{TT_METAL_HOME})

find_package(Torch REQUIRED)

add_executable(recorder_test1_torch
    recorder_test1_torch.cpp
)

target_include_directories(recorder_test1_torch PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/../..
    ${TT_METAL_HOME}/tt_metal/api
)

target_link_libraries(recorder_test1_torch "${TORCH_LIBRARIES}")

set_property(TARGET recorder_test1_torch PROPERTY CXX_STANDARD 20)
