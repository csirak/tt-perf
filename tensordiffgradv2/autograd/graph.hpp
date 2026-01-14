// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "value.hpp"
#include "../core/op_log.hpp"

#include <algorithm>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tensordiffgradv2 {

class Graph {
public:
    // Enable op logging to a directory
    void enable_logging(const std::filesystem::path& dir) {
        recorder_.enable(dir);
    }

    void disable_logging() {
        recorder_.disable();
    }

    bool logging_enabled() const {
        return recorder_.enabled();
    }

    OpRecorder& recorder() { return recorder_; }

    // Create input node (typically data that flows in)
    Value* input(std::shared_ptr<Tensor> t, std::string name, bool requires_grad = true) {
        nodes_.push_back(std::make_unique<Value>(this, std::move(t), std::move(name), requires_grad));
        return nodes_.back().get();
    }

    // Create parameter node (weights, biases)
    Value* param(std::shared_ptr<Tensor> t, std::string name) {
        return input(std::move(t), std::move(name), true);
    }

    // Create constant node (no gradient)
    Value* constant(std::shared_ptr<Tensor> t, std::string name) {
        return input(std::move(t), std::move(name), false);
    }

    // Internal: create a new node (called by Value ops)
    Value* make_node(std::shared_ptr<Tensor> data, std::string name, bool requires_grad) {
        nodes_.push_back(std::make_unique<Value>(this, std::move(data), std::move(name), requires_grad));
        return nodes_.back().get();
    }

    // Zero all gradients
    void zero_grad() {
        for (auto& node : nodes_) {
            node->zero_grad();
        }
    }

    // Backward pass from loss node
    void backward(Value* loss) {
        if (!loss) return;

        // Build topological order
        std::vector<Value*> topo;
        std::vector<Value*> visited;

        std::function<void(Value*)> build_topo = [&](Value* v) {
            if (!v) return;
            if (std::find(visited.begin(), visited.end(), v) != visited.end()) return;
            visited.push_back(v);
            for (auto* p : v->parents) {
                build_topo(p);
            }
            topo.push_back(v);
        };

        build_topo(loss);

        // Initialize loss gradient to ones
        if (loss->requires_grad) {
            loss->grad = loss->data->ones_like();
        }

        // Backward in reverse topological order
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            if ((*it)->backward_fn) {
                (*it)->backward_fn();
            }
        }
    }

private:
    std::vector<std::unique_ptr<Value>> nodes_;
    OpRecorder recorder_;
};

// ========== Value op implementations ==========
// These are defined here to avoid circular include issues

inline Value* Value::add(Value* other, std::string op_name) {
    auto result = data->add(*other->data);
    auto* out = graph_->make_node(result, op_name, requires_grad || other->requires_grad);
    out->parents = {this, other};

    // Backward: d/da(a+b) = 1, d/db(a+b) = 1
    out->backward_fn = [this, other, out]() {
        if (!out->grad) return;
        if (this->requires_grad) this->accumulate_grad(out->grad);
        if (other->requires_grad) other->accumulate_grad(out->grad);
    };

    // Log if enabled
    if (graph_->logging_enabled()) {
        graph_->recorder().record("add", OpType::Binary, {data.get(), other->data.get()},
                                  {name, other->name}, *result, op_name);
    }

    return out;
}

inline Value* Value::sub(Value* other, std::string op_name) {
    auto result = data->sub(*other->data);
    auto* out = graph_->make_node(result, op_name, requires_grad || other->requires_grad);
    out->parents = {this, other};

    out->backward_fn = [this, other, out]() {
        if (!out->grad) return;
        if (this->requires_grad) this->accumulate_grad(out->grad);
        if (other->requires_grad) other->accumulate_grad(out->grad->mul_scalar(-1.0f));
    };

    if (graph_->logging_enabled()) {
        graph_->recorder().record("sub", OpType::Binary, {data.get(), other->data.get()},
                                  {name, other->name}, *result, op_name);
    }

    return out;
}

inline Value* Value::mul(Value* other, std::string op_name) {
    auto result = data->mul(*other->data);
    auto* out = graph_->make_node(result, op_name, requires_grad || other->requires_grad);
    out->parents = {this, other};

    // Capture data for backward
    auto a_data = data;
    auto b_data = other->data;

    out->backward_fn = [this, other, out, a_data, b_data]() {
        if (!out->grad) return;
        if (this->requires_grad) this->accumulate_grad(out->grad->mul(*b_data));
        if (other->requires_grad) other->accumulate_grad(out->grad->mul(*a_data));
    };

    if (graph_->logging_enabled()) {
        graph_->recorder().record("mul", OpType::Binary, {data.get(), other->data.get()},
                                  {name, other->name}, *result, op_name);
    }

    return out;
}

inline Value* Value::matmul(Value* other, std::string op_name, const MatmulOpts& opts) {
    auto result = data->matmul(*other->data, opts);
    auto* out = graph_->make_node(result, op_name, requires_grad || other->requires_grad);
    out->parents = {this, other};

    auto a_data = data;
    auto b_data = other->data;

    out->backward_fn = [this, other, out, a_data, b_data]() {
        if (!out->grad) return;
        // dA = dOut @ B^T
        if (this->requires_grad) {
            auto b_t = b_data->transpose(-2, -1);
            this->accumulate_grad(out->grad->matmul(*b_t));
        }
        // dB = A^T @ dOut
        if (other->requires_grad) {
            auto a_t = a_data->transpose(-2, -1);
            other->accumulate_grad(a_t->matmul(*out->grad));
        }
    };

    if (graph_->logging_enabled()) {
        // Always log opts - even defaults
        graph_->recorder().record("matmul", OpType::Binary, {data.get(), other->data.get()},
                                  {name, other->name}, *result, op_name, opts.to_string());
    }

    return out;
}

inline Value* Value::gelu(std::string op_name) {
    auto result = data->gelu();
    auto* out = graph_->make_node(result, op_name, requires_grad);
    out->parents = {this};

    auto x_data = data;

    out->backward_fn = [this, out, x_data]() {
        if (!out->grad || !this->requires_grad) return;
        // GELU backward approximation using torch
        // For now, use numerical approximation or rely on torch autograd
        // This is a simplified version - proper impl would compute derivative
        // gelu'(x) = 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        //          + 0.5 * x * sech^2(...) * sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
        // For simplicity, we'll use finite difference or assume torch handles it
        // TODO: implement proper GELU backward
        this->accumulate_grad(out->grad);  // placeholder
    };

    if (graph_->logging_enabled()) {
        graph_->recorder().record("gelu", OpType::Unary, {data.get()}, {name}, *result, op_name);
    }

    return out;
}

inline Value* Value::relu(std::string op_name) {
    auto result = data->relu();
    auto* out = graph_->make_node(result, op_name, requires_grad);
    out->parents = {this};

    auto x_data = data;

    out->backward_fn = [this, out, x_data]() {
        if (!out->grad || !this->requires_grad) return;
        // ReLU backward: grad * (x > 0)
        // Since we don't have comparison ops in Tensor interface,
        // we approximate using the fact that relu(x)/x = 1 when x > 0
        // For now, pass gradient through (will implement proper mask later)
        // TODO: add heaviside/step or gt ops to Tensor for proper backward
        this->accumulate_grad(out->grad);  // placeholder - passes all gradients
    };

    if (graph_->logging_enabled()) {
        graph_->recorder().record("relu", OpType::Unary, {data.get()}, {name}, *result, op_name);
    }

    return out;
}

inline Value* Value::softmax(int dim, std::string op_name) {
    auto result = data->softmax(dim);
    auto* out = graph_->make_node(result, op_name, requires_grad);
    out->parents = {this};

    auto y_data = result;  // softmax output needed for backward

    out->backward_fn = [this, out, y_data, dim]() {
        if (!out->grad || !this->requires_grad) return;
        // softmax backward: dx = y * (dout - sum(dout * y, dim))
        auto dy_y = out->grad->mul(*y_data);
        auto sum_dy_y = dy_y->sum();  // TODO: sum along dim, not total
        // Simplified - full impl needs per-row sum
        auto dx = y_data->mul(*out->grad->sub(*sum_dy_y));
        this->accumulate_grad(dx);
    };

    if (graph_->logging_enabled()) {
        graph_->recorder().record("softmax", OpType::Unary, {data.get()}, {name}, *result, op_name);
    }

    return out;
}

inline Value* Value::sum(std::string op_name) {
    auto result = data->sum();
    auto* out = graph_->make_node(result, op_name, requires_grad);
    out->parents = {this};

    auto orig_data = data;

    out->backward_fn = [this, out, orig_data]() {
        if (!out->grad || !this->requires_grad) return;
        // d/dx sum(x) = ones_like(x) * grad_output
        this->accumulate_grad(orig_data->ones_like());
    };

    if (graph_->logging_enabled()) {
        graph_->recorder().record("sum", OpType::Unary, {data.get()}, {name}, *result, op_name);
    }

    return out;
}

inline Value* Value::mul_scalar(float scalar, std::string op_name) {
    auto result = data->mul_scalar(scalar);
    auto* out = graph_->make_node(result, op_name, requires_grad);
    out->parents = {this};

    out->backward_fn = [this, out, scalar]() {
        if (!out->grad || !this->requires_grad) return;
        this->accumulate_grad(out->grad->mul_scalar(scalar));
    };

    if (graph_->logging_enabled()) {
        graph_->recorder().record("mul_scalar", OpType::Unary, {data.get()}, {name}, *result, op_name,
                                  "scalar=" + std::to_string(scalar));
    }

    return out;
}

inline Value* Value::transpose(int dim0, int dim1, std::string op_name) {
    auto result = data->transpose(dim0, dim1);
    auto* out = graph_->make_node(result, op_name, requires_grad);
    out->parents = {this};

    out->backward_fn = [this, out, dim0, dim1]() {
        if (!out->grad || !this->requires_grad) return;
        // transpose backward is just transpose again
        this->accumulate_grad(out->grad->transpose(dim0, dim1));
    };

    if (graph_->logging_enabled()) {
        graph_->recorder().record("transpose", OpType::Unary, {data.get()}, {name}, *result, op_name);
    }

    return out;
}

inline Value* Value::reshape(std::vector<int64_t> new_shape, std::string op_name) {
    auto result = data->reshape(new_shape);
    auto* out = graph_->make_node(result, op_name, requires_grad);
    out->parents = {this};

    auto orig_shape = data->shape();

    out->backward_fn = [this, out, orig_shape]() {
        if (!out->grad || !this->requires_grad) return;
        std::vector<int64_t> shape_i64(orig_shape.begin(), orig_shape.end());
        this->accumulate_grad(out->grad->reshape(shape_i64));
    };

    if (graph_->logging_enabled()) {
        graph_->recorder().record("reshape", OpType::Unary, {data.get()}, {name}, *result, op_name);
    }

    return out;
}

}  // namespace tensordiffgradv2
