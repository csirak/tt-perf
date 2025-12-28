cmake_minimum_required(VERSION 3.22...3.30)
project(simple_add)

add_executable(simple_add)
target_sources(simple_add PRIVATE simple_add.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(simple_add PUBLIC TT::Metalium)
