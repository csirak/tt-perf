// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Tensor I/O utilities for cross-implementation verification.
// Binary format: [ndim:u32][dim0:u32][dim1:u32]...[data:bf16[]]
//
// Compatible with Python tensor_io.py for cross-language comparison.

#pragma once

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/operations/creation.hpp>
#include <tt-metalium/distributed.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>

namespace traced {

using Tensor = tt::tt_metal::Tensor;
using MeshDevice = tt::tt_metal::distributed::MeshDevice;

// Save tensor to binary file
// Tensor is synced and brought to CPU before saving
inline void save_tensor(const Tensor& t, const std::string& path, MeshDevice* dev = nullptr) {
    // Sync if device provided
    if (dev) {
        tt::tt_metal::distributed::Synchronize(dev, std::nullopt);
    }

    // Bring to CPU
    auto cpu_tensor = t.cpu();

    // Get shape
    auto shape = cpu_tensor.logical_shape();
    uint32_t ndim = shape.rank();

    // Get data as bf16 vector
    auto data = cpu_tensor.to_vector<bfloat16>();

    // Create parent directories
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());

    // Write to file
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }

    // Write ndim
    file.write(reinterpret_cast<const char*>(&ndim), sizeof(uint32_t));

    // Write dimensions
    for (uint32_t i = 0; i < ndim; ++i) {
        uint32_t dim = shape[i];
        file.write(reinterpret_cast<const char*>(&dim), sizeof(uint32_t));
    }

    // Write data (bf16 = 2 bytes each)
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(bfloat16));

    file.close();
}

// Load tensor from binary file to device
inline Tensor load_tensor(const std::string& path, MeshDevice& dev) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }

    // Read ndim
    uint32_t ndim;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(uint32_t));

    // Read dimensions
    std::vector<uint32_t> dims(ndim);
    for (uint32_t i = 0; i < ndim; ++i) {
        file.read(reinterpret_cast<char*>(&dims[i]), sizeof(uint32_t));
    }

    // Calculate total elements
    size_t numel = 1;
    for (uint32_t d : dims) {
        numel *= d;
    }

    // Read data
    std::vector<bfloat16> data(numel);
    file.read(reinterpret_cast<char*>(data.data()), numel * sizeof(bfloat16));

    file.close();

    // Create shape
    ttnn::Shape shape(dims);

    // Create tensor layout
    auto tensor_layout = ttnn::TensorLayout(
        ttnn::DataType::BFLOAT16,
        ttnn::PageConfig(ttnn::TILE_LAYOUT),
        tt::tt_metal::MemoryConfig{}
    );

    // Create tensor from vector
    auto tensor = Tensor::from_vector(data, ttnn::TensorSpec(shape, tensor_layout));

    // Move to device
    return tensor.to_device(&dev);
}

// Load tensor for weights that need ROW_MAJOR layout (e.g., embedding weights)
inline Tensor load_tensor_row_major(const std::string& path, MeshDevice& dev) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }

    // Read ndim
    uint32_t ndim;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(uint32_t));

    // Read dimensions
    std::vector<uint32_t> dims(ndim);
    for (uint32_t i = 0; i < ndim; ++i) {
        file.read(reinterpret_cast<char*>(&dims[i]), sizeof(uint32_t));
    }

    // Calculate total elements
    size_t numel = 1;
    for (uint32_t d : dims) {
        numel *= d;
    }

    // Read data
    std::vector<bfloat16> data(numel);
    file.read(reinterpret_cast<char*>(data.data()), numel * sizeof(bfloat16));

    file.close();

    // Create shape
    ttnn::Shape shape(dims);

    // Create tensor layout - ROW_MAJOR for embedding weights
    auto tensor_layout = ttnn::TensorLayout(
        ttnn::DataType::BFLOAT16,
        ttnn::PageConfig(ttnn::ROW_MAJOR_LAYOUT),
        tt::tt_metal::MemoryConfig{}
    );

    // Create tensor from vector
    auto tensor = Tensor::from_vector(data, ttnn::TensorSpec(shape, tensor_layout));

    // Move to device
    return tensor.to_device(&dev);
}

}  // namespace traced
