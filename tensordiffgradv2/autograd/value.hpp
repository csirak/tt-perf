// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../core/tensor.hpp"
#include "../core/op_log.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tensordiffgradv2 {

class Graph;  // forward declaration

class Value {
public:
    // Data and gradient (declared first for init order)
    std::shared_ptr<Tensor> data;
    std::shared_ptr<Tensor> grad;
    std::string name;
    bool requires_grad = false;
    std::function<void()> backward_fn;
    std::vector<Value*> parents;

    Value(Graph* graph, std::shared_ptr<Tensor> data_in, std::string name_in, bool requires_grad_in)
        : data(std::move(data_in))
        , name(std::move(name_in))
        , requires_grad(requires_grad_in)
        , graph_(graph) {}

    // Gradient accumulation
    void zero_grad() {
        grad.reset();
    }

    void accumulate_grad(std::shared_ptr<Tensor> g) {
        if (!requires_grad) return;
        if (!grad) {
            grad = g;
        } else {
            grad = grad->add(*g);
        }
    }

    // ========== Ops as methods ==========
    // Each op: compute result, set backward_fn, log if enabled

    Value* add(Value* other, std::string op_name);
    Value* sub(Value* other, std::string op_name);
    Value* mul(Value* other, std::string op_name);
    Value* matmul(Value* other, std::string op_name, const MatmulOpts& opts = {});
    Value* gelu(std::string op_name);
    Value* relu(std::string op_name);
    Value* softmax(int dim, std::string op_name);
    Value* sum(std::string op_name);
    Value* mul_scalar(float scalar, std::string op_name);
    Value* transpose(int dim0, int dim1, std::string op_name);
    Value* reshape(std::vector<int64_t> new_shape, std::string op_name);

private:
    Graph* graph_;

    // Helper to access graph (defined in graph.hpp to avoid circular include)
    Graph* get_graph() const { return graph_; }

    friend class Graph;
};

}  // namespace tensordiffgradv2
