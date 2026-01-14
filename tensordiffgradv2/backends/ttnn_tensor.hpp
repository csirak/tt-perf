// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../core/tensor.hpp"

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/normalization/softmax/softmax.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <ttnn/operations/data_movement/transpose/transpose.hpp>
#include <ttnn/operations/data_movement/reshape_view/reshape.hpp>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>

// torch/torch.h is optional - only used for from_torch() method
#ifdef TENSORDIFFGRADV2_HAS_TORCH
#include <torch/torch.h>
#endif

#include <cstring>
#include <random>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace tensordiffgradv2 {

using TT_Tensor = tt::tt_metal::Tensor;
using MeshDevice = tt::tt_metal::distributed::MeshDevice;

class TtnnTensor : public Tensor {
public:
    TtnnTensor(TT_Tensor tensor, MeshDevice* device)
        : tensor_(std::move(tensor)), device_(device) {
        if (!device_) {
            throw std::runtime_error("TtnnTensor: device is null");
        }
    }

#ifdef TENSORDIFFGRADV2_HAS_TORCH
    // Factory from torch tensor - allocates on device
    static std::shared_ptr<TtnnTensor> from_torch(const torch::Tensor& t, MeshDevice& dev) {
        // Convert to BF16 on CPU
        auto cpu_bf16 = t.to(torch::kBFloat16).contiguous().cpu();

        // Get shape
        std::vector<uint32_t> dims;
        for (int64_t i = 0; i < cpu_bf16.dim(); ++i) {
            dims.push_back(static_cast<uint32_t>(cpu_bf16.size(i)));
        }

        // Convert to tt bfloat16 vector
        const size_t numel = cpu_bf16.numel();
        std::vector<bfloat16> tt_bf16(numel);
        const auto* src = reinterpret_cast<const uint16_t*>(cpu_bf16.data_ptr());
        for (size_t i = 0; i < numel; ++i) {
            // torch bfloat16 and tt bfloat16 have same bit layout
            tt_bf16[i] = bfloat16::from_bits(src[i]);
        }

        // Create TTNN tensor
        ttnn::Shape tt_shape(dims);
        auto layout = ttnn::TensorLayout(
            ttnn::DataType::BFLOAT16,
            ttnn::PageConfig(ttnn::TILE_LAYOUT),
            tt::tt_metal::MemoryConfig{}
        );
        auto tensor = TT_Tensor::from_vector(tt_bf16, ttnn::TensorSpec(tt_shape, layout));
        return std::make_shared<TtnnTensor>(tensor.to_device(&dev), &dev);
    }
#endif

    // Factory methods - create tensors without torch dependency
    static std::shared_ptr<TtnnTensor> randn(std::vector<int64_t> shape, MeshDevice& dev, unsigned seed = 42) {
        // Compute total elements
        size_t numel = 1;
        std::vector<uint32_t> dims;
        for (auto d : shape) {
            numel *= static_cast<size_t>(d);
            dims.push_back(static_cast<uint32_t>(d));
        }

        // Generate random floats using standard library
        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        std::vector<bfloat16> data(numel);
        for (size_t i = 0; i < numel; ++i) {
            data[i] = bfloat16(dist(gen));
        }

        // Create TTNN tensor
        ttnn::Shape tt_shape(dims);
        auto layout = ttnn::TensorLayout(
            ttnn::DataType::BFLOAT16,
            ttnn::PageConfig(ttnn::TILE_LAYOUT),
            tt::tt_metal::MemoryConfig{}
        );
        auto tensor = TT_Tensor::from_vector(data, ttnn::TensorSpec(tt_shape, layout));
        return std::make_shared<TtnnTensor>(tensor.to_device(&dev), &dev);
    }

    static std::shared_ptr<TtnnTensor> zeros(std::vector<int64_t> shape, MeshDevice& dev) {
        size_t numel = 1;
        std::vector<uint32_t> dims;
        for (auto d : shape) {
            numel *= static_cast<size_t>(d);
            dims.push_back(static_cast<uint32_t>(d));
        }

        std::vector<bfloat16> data(numel, bfloat16(0.0f));

        ttnn::Shape tt_shape(dims);
        auto layout = ttnn::TensorLayout(
            ttnn::DataType::BFLOAT16,
            ttnn::PageConfig(ttnn::TILE_LAYOUT),
            tt::tt_metal::MemoryConfig{}
        );
        auto tensor = TT_Tensor::from_vector(data, ttnn::TensorSpec(tt_shape, layout));
        return std::make_shared<TtnnTensor>(tensor.to_device(&dev), &dev);
    }

    static std::shared_ptr<TtnnTensor> ones(std::vector<int64_t> shape, MeshDevice& dev) {
        size_t numel = 1;
        std::vector<uint32_t> dims;
        for (auto d : shape) {
            numel *= static_cast<size_t>(d);
            dims.push_back(static_cast<uint32_t>(d));
        }

        std::vector<bfloat16> data(numel, bfloat16(1.0f));

        ttnn::Shape tt_shape(dims);
        auto layout = ttnn::TensorLayout(
            ttnn::DataType::BFLOAT16,
            ttnn::PageConfig(ttnn::TILE_LAYOUT),
            tt::tt_metal::MemoryConfig{}
        );
        auto tensor = TT_Tensor::from_vector(data, ttnn::TensorSpec(tt_shape, layout));
        return std::make_shared<TtnnTensor>(tensor.to_device(&dev), &dev);
    }

    // Stable initialization for weight matrices
    // W = randn(shape) / ||W|| * scale * sqrt(fan_in)
    static std::shared_ptr<TtnnTensor> stable_init(
        std::vector<int64_t> shape,
        MeshDevice& dev,
        float scale = 0.5f,
        unsigned seed = 42
    ) {
        // Compute total elements and shape
        size_t numel = 1;
        std::vector<uint32_t> dims;
        for (auto d : shape) {
            numel *= static_cast<size_t>(d);
            dims.push_back(static_cast<uint32_t>(d));
        }

        // Generate random floats
        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        std::vector<float> data_f32(numel);
        float norm_sq = 0.0f;
        for (size_t i = 0; i < numel; ++i) {
            data_f32[i] = dist(gen);
            norm_sq += data_f32[i] * data_f32[i];
        }
        float norm = std::sqrt(norm_sq);

        // Compute fan_in (first half of dims product)
        int64_t fan_in = 1;
        for (size_t i = 0; i < shape.size() / 2 || i < 1; ++i) {
            fan_in *= shape[i];
        }

        // Normalize and convert to bfloat16
        float multiplier = (norm > 0) ? (scale * std::sqrt(static_cast<float>(fan_in)) / norm) : 1.0f;
        std::vector<bfloat16> data(numel);
        for (size_t i = 0; i < numel; ++i) {
            data[i] = bfloat16(data_f32[i] * multiplier);
        }

        // Create TTNN tensor
        ttnn::Shape tt_shape(dims);
        auto layout = ttnn::TensorLayout(
            ttnn::DataType::BFLOAT16,
            ttnn::PageConfig(ttnn::TILE_LAYOUT),
            tt::tt_metal::MemoryConfig{}
        );
        auto tensor = TT_Tensor::from_vector(data, ttnn::TensorSpec(tt_shape, layout));
        return std::make_shared<TtnnTensor>(tensor.to_device(&dev), &dev);
    }

    // Stable initialization for inputs: uniform(-0.5, 0.5)
    static std::shared_ptr<TtnnTensor> stable_input(
        std::vector<int64_t> shape,
        MeshDevice& dev,
        unsigned seed = 42
    ) {
        size_t numel = 1;
        std::vector<uint32_t> dims;
        for (auto d : shape) {
            numel *= static_cast<size_t>(d);
            dims.push_back(static_cast<uint32_t>(d));
        }

        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

        std::vector<bfloat16> data(numel);
        for (size_t i = 0; i < numel; ++i) {
            data[i] = bfloat16(dist(gen));
        }

        ttnn::Shape tt_shape(dims);
        auto layout = ttnn::TensorLayout(
            ttnn::DataType::BFLOAT16,
            ttnn::PageConfig(ttnn::TILE_LAYOUT),
            tt::tt_metal::MemoryConfig{}
        );
        auto tensor = TT_Tensor::from_vector(data, ttnn::TensorSpec(tt_shape, layout));
        return std::make_shared<TtnnTensor>(tensor.to_device(&dev), &dev);
    }

    // Core properties
    std::vector<uint32_t> shape() const override {
        auto s = tensor_.logical_shape();
        std::vector<uint32_t> dims(s.rank());
        for (uint32_t i = 0; i < s.rank(); ++i) {
            dims[i] = s[i];
        }
        return dims;
    }

    DType dtype() const override {
        return DType::kBF16;  // TTNN always BF16
    }

    std::string backend() const override { return "ttnn"; }

    // Access raw tensor and device
    const TT_Tensor& raw() const { return tensor_; }
    TT_Tensor& raw() { return tensor_; }
    MeshDevice* device() const { return device_; }

#ifdef TENSORDIFFGRADV2_HAS_TORCH
    // Convert to torch tensor (pulls to CPU)
    torch::Tensor to_torch() const {
        auto cpu_tensor = tensor_.cpu();
        auto shape_tt = cpu_tensor.logical_shape();

        std::vector<int64_t> dims(shape_tt.rank());
        for (uint32_t i = 0; i < shape_tt.rank(); ++i) {
            dims[i] = shape_tt[i];
        }

        auto data = cpu_tensor.to_vector<bfloat16>();
        auto opts = torch::TensorOptions().dtype(torch::kBFloat16);
        auto out = torch::empty(dims, opts);
        auto* dst = reinterpret_cast<uint16_t*>(out.data_ptr());
        for (size_t i = 0; i < data.size(); ++i) {
            dst[i] = data[i].to_bits();
        }
        return out;
    }
#endif

    // Tensor ops
    std::shared_ptr<Tensor> add(const Tensor& other) const override {
        const auto& o = cast(other);
        auto out = ttnn::add(tensor_, o.tensor_);
        return std::make_shared<TtnnTensor>(std::move(out), device_);
    }

    std::shared_ptr<Tensor> sub(const Tensor& other) const override {
        const auto& o = cast(other);
        auto out = ttnn::subtract(tensor_, o.tensor_);
        return std::make_shared<TtnnTensor>(std::move(out), device_);
    }

    std::shared_ptr<Tensor> mul(const Tensor& other) const override {
        const auto& o = cast(other);
        auto out = ttnn::multiply(tensor_, o.tensor_);
        return std::make_shared<TtnnTensor>(std::move(out), device_);
    }

    std::shared_ptr<Tensor> matmul(const Tensor& other, const MatmulOpts& opts = {}) const override {
        const auto& o = cast(other);

        // Build compute kernel config from opts
        // MathFidelity is in tt::tt_metal namespace (base_types.hpp)
        ttnn::WormholeComputeKernelConfig config;
        config.math_fidelity = static_cast<MathFidelity>(opts.math_fidelity);
        config.fp32_dest_acc_en = opts.fp32_acc;
        config.math_approx_mode = opts.approx;

        auto out = ttnn::matmul(
            tensor_,
            o.tensor_,
            false,  // transpose_a
            false,  // transpose_b
            std::nullopt,  // memory_config
            std::nullopt,  // dtype
            std::nullopt,  // program_config
            std::nullopt,  // activation
            config         // compute_kernel_config
        );
        return std::make_shared<TtnnTensor>(std::move(out), device_);
    }

    std::shared_ptr<Tensor> gelu() const override {
        auto out = ttnn::gelu(tensor_);
        return std::make_shared<TtnnTensor>(std::move(out), device_);
    }

    std::shared_ptr<Tensor> relu() const override {
        auto out = ttnn::relu(tensor_);
        return std::make_shared<TtnnTensor>(std::move(out), device_);
    }

    std::shared_ptr<Tensor> softmax(int dim) const override {
        auto out = ttnn::softmax(tensor_, dim);
        return std::make_shared<TtnnTensor>(std::move(out), device_);
    }

    std::shared_ptr<Tensor> sum() const override {
        // Full reduction
        auto out = ttnn::sum(tensor_);
        return std::make_shared<TtnnTensor>(std::move(out), device_);
    }

    std::shared_ptr<Tensor> mul_scalar(float scalar) const override {
        auto out = ttnn::multiply(tensor_, scalar);
        return std::make_shared<TtnnTensor>(std::move(out), device_);
    }

    std::shared_ptr<Tensor> add_scalar(float scalar) const override {
        auto out = ttnn::add(tensor_, scalar);
        return std::make_shared<TtnnTensor>(std::move(out), device_);
    }

    std::shared_ptr<Tensor> transpose(int dim0, int dim1) const override {
        auto out = ttnn::transpose(tensor_, dim0, dim1);
        return std::make_shared<TtnnTensor>(std::move(out), device_);
    }

    std::shared_ptr<Tensor> reshape(std::vector<int64_t> new_shape) const override {
        ttnn::SmallVector<int32_t> shape_vec;
        for (auto d : new_shape) {
            shape_vec.push_back(static_cast<int32_t>(d));
        }
        auto out = ttnn::reshape(tensor_, shape_vec);
        return std::make_shared<TtnnTensor>(std::move(out), device_);
    }

    std::shared_ptr<Tensor> clone() const override {
        // Create a copy on device
        auto cpu_copy = tensor_.cpu();
        auto out = cpu_copy.to_device(device_);
        return std::make_shared<TtnnTensor>(std::move(out), device_);
    }

    std::shared_ptr<Tensor> zeros_like() const override {
        auto s = shape();
        std::vector<int64_t> dims(s.begin(), s.end());
        return zeros(dims, *device_);
    }

    std::shared_ptr<Tensor> ones_like() const override {
        auto s = shape();
        std::vector<int64_t> dims(s.begin(), s.end());
        return ones(dims, *device_);
    }

protected:
    void save_impl(const std::string& path) const override {
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());

        std::ofstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open file for writing: " + path);
        }

        // Pull to CPU first
        auto cpu_tensor = tensor_.cpu();
        auto shape_tt = cpu_tensor.logical_shape();
        auto data = cpu_tensor.to_vector<bfloat16>();

        // TDF1 header
        constexpr char magic[4] = {'T', 'D', 'F', '1'};
        constexpr uint32_t version = 1;
        const uint32_t dtype_code = static_cast<uint32_t>(DType::kBF16);
        const uint32_t ndim = shape_tt.rank();

        file.write(magic, 4);
        file.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&dtype_code), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&ndim), sizeof(uint32_t));

        // Shape
        for (uint32_t i = 0; i < ndim; ++i) {
            uint32_t d = shape_tt[i];
            file.write(reinterpret_cast<const char*>(&d), sizeof(uint32_t));
        }

        // Data - write as raw bf16 bits
        const size_t nbytes = data.size() * sizeof(uint16_t);
        file.write(reinterpret_cast<const char*>(data.data()), nbytes);

        // Origin trailer
        constexpr char orig_magic[4] = {'O', 'R', 'I', 'G'};
        const std::string origin = "ttnn";
        const uint32_t origin_len = static_cast<uint32_t>(origin.size());
        file.write(orig_magic, 4);
        file.write(reinterpret_cast<const char*>(&origin_len), sizeof(uint32_t));
        file.write(origin.data(), origin_len);
    }

private:
    TT_Tensor tensor_;
    MeshDevice* device_ = nullptr;

    static const TtnnTensor& cast(const Tensor& t) {
        const auto* p = dynamic_cast<const TtnnTensor*>(&t);
        if (!p) {
            throw std::runtime_error("TtnnTensor op expects TtnnTensor");
        }
        return *p;
    }
};

}  // namespace tensordiffgradv2
