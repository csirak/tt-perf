cmake_minimum_required(VERSION 3.16)
project(tensordiffgradv2 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find libtorch
find_package(Torch REQUIRED)

# Header-only library
add_library(tensordiffgradv2 INTERFACE)
target_include_directories(tensordiffgradv2 INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries(tensordiffgradv2 INTERFACE ${TORCH_LIBRARIES})

# Test executables
add_executable(torch_ffn_test tests/torch_ffn_test.cpp)
target_link_libraries(torch_ffn_test PRIVATE tensordiffgradv2 ${TORCH_LIBRARIES})
target_compile_options(torch_ffn_test PRIVATE -Wall -Wextra)

# FFN comparison test (Torch backend only)
add_executable(ffn_compare_torch tests/ffn_compare_test.cpp)
target_link_libraries(ffn_compare_torch PRIVATE tensordiffgradv2 ${TORCH_LIBRARIES})
target_compile_options(ffn_compare_torch PRIVATE -Wall -Wextra)

# Copy torch libraries for runtime
if(TORCH_LIBRARIES)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.so*")
endif()
