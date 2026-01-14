// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../core/tensor.hpp"

#include <torch/torch.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace tensordiffgradv2 {

class TorchTensor : public Tensor {
public:
    explicit TorchTensor(torch::Tensor data) : data_(std::move(data)) {
        // Validate dtype
        if (data_.dtype() != torch::kFloat32 && data_.dtype() != torch::kBFloat16) {
            throw std::runtime_error("TorchTensor: only kFloat32 and kBFloat16 supported");
        }
    }

    // Factory methods - dtype MUST be explicit
    static std::shared_ptr<TorchTensor> zeros(std::vector<int64_t> shape, DType dtype) {
        auto opts = torch::TensorOptions().dtype(to_torch_dtype(dtype)).device(torch::kCPU);
        return std::make_shared<TorchTensor>(torch::zeros(shape, opts));
    }

    static std::shared_ptr<TorchTensor> ones(std::vector<int64_t> shape, DType dtype) {
        auto opts = torch::TensorOptions().dtype(to_torch_dtype(dtype)).device(torch::kCPU);
        return std::make_shared<TorchTensor>(torch::ones(shape, opts));
    }

    static std::shared_ptr<TorchTensor> randn(std::vector<int64_t> shape, DType dtype) {
        auto opts = torch::TensorOptions().dtype(to_torch_dtype(dtype)).device(torch::kCPU);
        return std::make_shared<TorchTensor>(torch::randn(shape, opts));
    }

    static std::shared_ptr<TorchTensor> rand(std::vector<int64_t> shape, DType dtype, float lo = 0.0f, float hi = 1.0f) {
        auto opts = torch::TensorOptions().dtype(to_torch_dtype(dtype)).device(torch::kCPU);
        auto t = torch::rand(shape, opts) * (hi - lo) + lo;
        return std::make_shared<TorchTensor>(t);
    }

    // Stable initialization for weight matrices
    // W = randn(shape) / ||W|| * scale * sqrt(fan_in)
    static std::shared_ptr<TorchTensor> stable_init(
        std::vector<int64_t> shape,
        DType dtype,
        float scale = 0.5f,
        unsigned seed = 42
    ) {
        torch::manual_seed(seed);
        auto opts = torch::TensorOptions().dtype(to_torch_dtype(dtype)).device(torch::kCPU);
        auto W = torch::randn(shape, opts);

        // Compute fan_in (first half of dims product)
        int64_t fan_in = 1;
        for (size_t i = 0; i < shape.size() / 2 || i < 1; ++i) {
            fan_in *= shape[i];
        }

        // Normalize: W / ||W|| * scale * sqrt(fan_in)
        float norm = W.norm().item<float>();
        if (norm > 0) {
            W = W / norm * scale * std::sqrt(static_cast<float>(fan_in));
        }

        return std::make_shared<TorchTensor>(W);
    }

    // Stable initialization for inputs: uniform(-0.5, 0.5)
    static std::shared_ptr<TorchTensor> stable_input(
        std::vector<int64_t> shape,
        DType dtype,
        unsigned seed = 42
    ) {
        torch::manual_seed(seed);
        return rand(shape, dtype, -0.5f, 0.5f);
    }

    // Core properties
    std::vector<uint32_t> shape() const override {
        std::vector<uint32_t> dims;
        dims.reserve(static_cast<size_t>(data_.dim()));
        for (int64_t i = 0; i < data_.dim(); ++i) {
            dims.push_back(static_cast<uint32_t>(data_.size(i)));
        }
        return dims;
    }

    DType dtype() const override {
        if (data_.dtype() == torch::kFloat32) return DType::kF32;
        return DType::kBF16;
    }

    std::string backend() const override { return "torch"; }

    // Access raw tensor
    const torch::Tensor& raw() const { return data_; }
    torch::Tensor& raw() { return data_; }

    // Tensor ops
    std::shared_ptr<Tensor> add(const Tensor& other) const override {
        const auto& o = cast(other);
        return std::make_shared<TorchTensor>(data_ + o.data_);
    }

    std::shared_ptr<Tensor> sub(const Tensor& other) const override {
        const auto& o = cast(other);
        return std::make_shared<TorchTensor>(data_ - o.data_);
    }

    std::shared_ptr<Tensor> mul(const Tensor& other) const override {
        const auto& o = cast(other);
        return std::make_shared<TorchTensor>(data_ * o.data_);
    }

    std::shared_ptr<Tensor> matmul(const Tensor& other, const MatmulOpts& opts = {}) const override {
        // opts ignored for torch backend - torch doesn't have compute kernel config
        (void)opts;
        const auto& o = cast(other);
        return std::make_shared<TorchTensor>(torch::matmul(data_, o.data_));
    }

    std::shared_ptr<Tensor> gelu() const override {
        return std::make_shared<TorchTensor>(torch::gelu(data_));
    }

    std::shared_ptr<Tensor> relu() const override {
        return std::make_shared<TorchTensor>(torch::relu(data_));
    }

    std::shared_ptr<Tensor> softmax(int dim) const override {
        return std::make_shared<TorchTensor>(torch::softmax(data_, dim));
    }

    std::shared_ptr<Tensor> sum() const override {
        return std::make_shared<TorchTensor>(data_.sum());
    }

    std::shared_ptr<Tensor> mul_scalar(float scalar) const override {
        return std::make_shared<TorchTensor>(data_ * scalar);
    }

    std::shared_ptr<Tensor> add_scalar(float scalar) const override {
        return std::make_shared<TorchTensor>(data_ + scalar);
    }

    std::shared_ptr<Tensor> transpose(int dim0, int dim1) const override {
        return std::make_shared<TorchTensor>(data_.transpose(dim0, dim1).contiguous());
    }

    std::shared_ptr<Tensor> reshape(std::vector<int64_t> new_shape) const override {
        return std::make_shared<TorchTensor>(data_.reshape(new_shape));
    }

    std::shared_ptr<Tensor> clone() const override {
        return std::make_shared<TorchTensor>(data_.clone());
    }

    std::shared_ptr<Tensor> zeros_like() const override {
        return std::make_shared<TorchTensor>(torch::zeros_like(data_));
    }

    std::shared_ptr<Tensor> ones_like() const override {
        return std::make_shared<TorchTensor>(torch::ones_like(data_));
    }

protected:
    void save_impl(const std::string& path) const override {
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());

        std::ofstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open file for writing: " + path);
        }

        // TDF1 header
        constexpr char magic[4] = {'T', 'D', 'F', '1'};
        constexpr uint32_t version = 1;
        const uint32_t dtype_code = static_cast<uint32_t>(dtype());
        const uint32_t ndim = static_cast<uint32_t>(data_.dim());

        file.write(magic, 4);
        file.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&dtype_code), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&ndim), sizeof(uint32_t));

        // Shape
        for (int64_t i = 0; i < data_.dim(); ++i) {
            uint32_t d = static_cast<uint32_t>(data_.size(i));
            file.write(reinterpret_cast<const char*>(&d), sizeof(uint32_t));
        }

        // Data - ensure contiguous and on CPU
        auto cpu_data = data_.contiguous().cpu();
        const size_t nbytes = cpu_data.numel() * cpu_data.element_size();
        file.write(reinterpret_cast<const char*>(cpu_data.data_ptr()), nbytes);

        // Origin trailer
        constexpr char orig_magic[4] = {'O', 'R', 'I', 'G'};
        const std::string origin = "torch";
        const uint32_t origin_len = static_cast<uint32_t>(origin.size());
        file.write(orig_magic, 4);
        file.write(reinterpret_cast<const char*>(&origin_len), sizeof(uint32_t));
        file.write(origin.data(), origin_len);
    }

private:
    torch::Tensor data_;

    static torch::Dtype to_torch_dtype(DType dtype) {
        switch (dtype) {
            case DType::kF32: return torch::kFloat32;
            case DType::kBF16: return torch::kBFloat16;
        }
        throw std::runtime_error("Unknown dtype");
    }

    static const TorchTensor& cast(const Tensor& t) {
        const auto* p = dynamic_cast<const TorchTensor*>(&t);
        if (!p) {
            throw std::runtime_error("TorchTensor op expects TorchTensor");
        }
        return *p;
    }
};

}  // namespace tensordiffgradv2
