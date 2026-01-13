// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cpu_tensor.hpp"
#include "dtype.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include <string_view>

namespace tensordiff::core {

constexpr char kMagic[4] = {'T', 'D', 'F', '1'};
constexpr uint32_t kVersion = 1;
constexpr char kOriginMagic[4] = {'O', 'R', 'I', 'G'};
constexpr char kOpMagic[4] = {'O', 'P', 'N', 'M'};

struct TensorWithOrigin {
    CpuTensor tensor;
    std::string origin;
    std::string op_name;
};

inline void save_tensor(const CpuTensor& t,
                        const std::string& path,
                        std::string_view origin = {},
                        std::string_view op_name = {}) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }

    const uint32_t ndim = static_cast<uint32_t>(t.shape().size());
    const uint32_t dtype = static_cast<uint32_t>(t.dtype());
    const size_t byte_count = t.numel() * dtype_size(t.dtype());

    file.write(kMagic, sizeof(kMagic));
    file.write(reinterpret_cast<const char*>(&kVersion), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&dtype), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&ndim), sizeof(uint32_t));
    for (uint32_t d : t.shape()) {
        file.write(reinterpret_cast<const char*>(&d), sizeof(uint32_t));
    }

    file.write(reinterpret_cast<const char*>(t.raw().data()), byte_count);

    if (!origin.empty()) {
        const uint32_t origin_len = static_cast<uint32_t>(origin.size());
        file.write(kOriginMagic, sizeof(kOriginMagic));
        file.write(reinterpret_cast<const char*>(&origin_len), sizeof(uint32_t));
        file.write(origin.data(), origin_len);
    }

    if (!op_name.empty()) {
        const uint32_t op_len = static_cast<uint32_t>(op_name.size());
        file.write(kOpMagic, sizeof(kOpMagic));
        file.write(reinterpret_cast<const char*>(&op_len), sizeof(uint32_t));
        file.write(op_name.data(), op_len);
    }
}

inline TensorWithOrigin load_tensor_with_origin(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }

    auto size = file.tellg();
    if (size <= 0) {
        throw std::runtime_error("Empty tensor file: " + path);
    }
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(static_cast<size_t>(size));
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
    if (!file) {
        throw std::runtime_error("Failed to read tensor file: " + path);
    }

    auto read_u32 = [&](size_t& offset) -> uint32_t {
        if (offset + sizeof(uint32_t) > buffer.size()) {
            throw std::runtime_error("Malformed tensor header");
        }
        uint32_t v = 0;
        std::memcpy(&v, buffer.data() + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        return v;
    };

    size_t offset = 0;
    bool is_tdf1 = buffer.size() >= 4 && std::memcmp(buffer.data(), kMagic, sizeof(kMagic)) == 0;
    std::vector<uint32_t> dims;
    DType dtype = DType::kBF16;

    if (!is_tdf1) {
        // Legacy BF16 format: [ndim][dims][data]
        uint32_t ndim = read_u32(offset);
        dims.resize(ndim);
        for (uint32_t i = 0; i < ndim; ++i) {
            dims[i] = read_u32(offset);
        }
        dtype = DType::kBF16;
    } else {
        offset += sizeof(kMagic);
        uint32_t version = read_u32(offset);
        uint32_t dtype_u32 = read_u32(offset);
        uint32_t ndim = read_u32(offset);

        if (version != kVersion) {
            throw std::runtime_error("Unsupported tensordiff tensor version");
        }

        dims.resize(ndim);
        for (uint32_t i = 0; i < ndim; ++i) {
            dims[i] = read_u32(offset);
        }
        dtype = static_cast<DType>(dtype_u32);
    }

    size_t numel = 1;
    for (uint32_t d : dims) {
        numel *= d;
    }
    const size_t byte_count = numel * dtype_size(dtype);
    if (offset + byte_count > buffer.size()) {
        throw std::runtime_error("Tensor payload truncated");
    }

    CpuTensor t(dims, dtype);
    std::memcpy(t.raw().data(), buffer.data() + offset, byte_count);
    offset += byte_count;

    std::string origin;
    std::string op_name;
    while (offset + 8 <= buffer.size()) {
        const char* tag = reinterpret_cast<const char*>(buffer.data() + offset);
        offset += 4;
        uint32_t len = read_u32(offset);
        if (offset + len > buffer.size()) {
            break;
        }
        if (std::memcmp(tag, kOriginMagic, sizeof(kOriginMagic)) == 0) {
            origin.assign(reinterpret_cast<const char*>(buffer.data() + offset), len);
        } else if (std::memcmp(tag, kOpMagic, sizeof(kOpMagic)) == 0) {
            op_name.assign(reinterpret_cast<const char*>(buffer.data() + offset), len);
        } else {
            break;
        }
        offset += len;
    }

    return TensorWithOrigin{std::move(t), std::move(origin), std::move(op_name)};
}

inline CpuTensor load_tensor(const std::string& path) {
    return load_tensor_with_origin(path).tensor;
}

inline void save_tensor_bf16(const CpuTensor& t, const std::string& path) {
    if (t.dtype() != DType::kBF16 && t.dtype() != DType::kF32) {
        throw std::runtime_error("save_tensor_bf16: unsupported dtype");
    }

    std::filesystem::create_directories(std::filesystem::path(path).parent_path());

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }

    uint32_t ndim = static_cast<uint32_t>(t.shape().size());
    file.write(reinterpret_cast<const char*>(&ndim), sizeof(uint32_t));
    for (uint32_t d : t.shape()) {
        file.write(reinterpret_cast<const char*>(&d), sizeof(uint32_t));
    }

    auto data = t.to_bf16_vector();
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(BFloat16));
}

inline CpuTensor load_tensor_bf16(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }

    uint32_t ndim = 0;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(uint32_t));

    std::vector<uint32_t> dims(ndim);
    for (uint32_t i = 0; i < ndim; ++i) {
        file.read(reinterpret_cast<char*>(&dims[i]), sizeof(uint32_t));
    }

    size_t numel = 1;
    for (uint32_t d : dims) {
        numel *= d;
    }

    std::vector<BFloat16> data(numel);
    file.read(reinterpret_cast<char*>(data.data()), numel * sizeof(BFloat16));

    return CpuTensor::from_bf16(std::move(data), std::move(dims));
}

}  // namespace tensordiff::core
