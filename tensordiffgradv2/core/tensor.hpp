// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "dtype.hpp"

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace tensordiffgradv2 {

// Compute kernel configuration options for matmul
// These map to TTNN WormholeComputeKernelConfig
struct MatmulOpts {
    int math_fidelity = 2;   // 0=LoFi, 2=HiFi2, 3=HiFi3, 4=HiFi4
    bool fp32_acc = false;   // fp32_dest_acc_en
    bool approx = false;     // math_approx_mode

    std::string to_string() const {
        std::ostringstream oss;
        oss << "math_fidelity=" << math_fidelity
            << ",fp32_acc=" << (fp32_acc ? "1" : "0")
            << ",approx=" << (approx ? "1" : "0");
        return oss.str();
    }
};

class Tensor {
public:
    virtual ~Tensor() = default;

    // Core properties
    virtual std::vector<uint32_t> shape() const = 0;
    virtual DType dtype() const = 0;
    virtual std::string backend() const = 0;

    // TDF1 serialization - base handles common logic, backend implements writing
    void save(const std::string& path) const {
        save_impl(path);
    }

    // Tensor ops - return shared_ptr for graph building
    // No logging here - logging happens at autograd/Value layer
    virtual std::shared_ptr<Tensor> add(const Tensor& other) const = 0;
    virtual std::shared_ptr<Tensor> sub(const Tensor& other) const = 0;
    virtual std::shared_ptr<Tensor> mul(const Tensor& other) const = 0;
    virtual std::shared_ptr<Tensor> matmul(const Tensor& other, const MatmulOpts& opts = {}) const = 0;
    virtual std::shared_ptr<Tensor> gelu() const = 0;
    virtual std::shared_ptr<Tensor> relu() const = 0;
    virtual std::shared_ptr<Tensor> softmax(int dim) const = 0;
    virtual std::shared_ptr<Tensor> sum() const = 0;

    // Scalar ops
    virtual std::shared_ptr<Tensor> mul_scalar(float scalar) const = 0;
    virtual std::shared_ptr<Tensor> add_scalar(float scalar) const = 0;

    // Shape ops
    virtual std::shared_ptr<Tensor> transpose(int dim0, int dim1) const = 0;
    virtual std::shared_ptr<Tensor> reshape(std::vector<int64_t> new_shape) const = 0;

    // Clone with same backend/dtype
    virtual std::shared_ptr<Tensor> clone() const = 0;

    // Zeros/ones with same shape/dtype/backend
    virtual std::shared_ptr<Tensor> zeros_like() const = 0;
    virtual std::shared_ptr<Tensor> ones_like() const = 0;

protected:
    // Backend implements TDF1 writing
    virtual void save_impl(const std::string& path) const = 0;
};

}  // namespace tensordiffgradv2
