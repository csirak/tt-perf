// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "tensordiff/core/cpu_tensor.hpp"
#include "tensordiff/core/dtype.hpp"

#include <functional>
#include <memory>
#include <vector>

namespace tensordiffgrad::core {

using tensordiff::core::CpuTensor;
using tensordiff::core::DType;

inline CpuTensor zeros_like(const CpuTensor& t) {
    CpuTensor out(t.shape(), DType::kF32);
    auto* dst = reinterpret_cast<float*>(out.raw().data());
    const size_t n = out.numel();
    for (size_t i = 0; i < n; ++i) {
        dst[i] = 0.0f;
    }
    return out;
}

inline CpuTensor add_tensors(const CpuTensor& a, const CpuTensor& b);

struct Value {
    CpuTensor data;
    std::unique_ptr<CpuTensor> grad;
    bool requires_grad = false;
    bool grad_initialized = false;
    std::string name;
    std::function<void()> backward_fn;
    std::vector<Value*> parents;

    Value(CpuTensor d, bool req, std::string n = {})
        : data(std::move(d)), requires_grad(req), name(std::move(n)) {}

    void zero_grad() {
        grad.reset();
        grad_initialized = false;
    }

    void accumulate_grad(const CpuTensor& g) {
        if (!requires_grad) return;
        if (!grad) {
            grad = std::make_unique<CpuTensor>(g);
            grad_initialized = true;
            return;
        }
        if (!grad_initialized) {
            *grad = g;
            grad_initialized = true;
            return;
        }
        *grad = add_tensors(*grad, g);
    }
};

struct Graph {
    std::vector<std::unique_ptr<Value>> nodes;
    std::vector<Value*> topo;
    Value* root = nullptr;

    Value* leaf(CpuTensor data, bool requires_grad, std::string name = {}) {
        nodes.push_back(std::make_unique<Value>(std::move(data), requires_grad, std::move(name)));
        return nodes.back().get();
    }

    Value* node(CpuTensor data, bool requires_grad, std::string name = {}) {
        nodes.push_back(std::make_unique<Value>(std::move(data), requires_grad, std::move(name)));
        return nodes.back().get();
    }

    void zero_grad() {
        for (auto& n : nodes) {
            n->zero_grad();
        }
    }

    void build_topo(Value* r) {
        root = r;
        topo.clear();
        std::vector<Value*> stack;
        std::vector<Value*> visited;

        std::function<void(Value*)> dfs = [&](Value* v) {
            if (!v) return;
            for (auto* vv : visited) {
                if (vv == v) return;
            }
            visited.push_back(v);
            for (auto* p : v->parents) {
                dfs(p);
            }
            topo.push_back(v);
        };

        dfs(root);
    }

    void backward(Value* r) {
        if (!r) return;
        if (topo.empty() || root != r) {
            build_topo(r);
        }
        if (r->requires_grad) {
            r->accumulate_grad(ones_like(r->data));
        }
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            if ((*it)->backward_fn) {
                (*it)->backward_fn();
            }
        }
    }

    static CpuTensor ones_like(const CpuTensor& t);
};

inline CpuTensor Graph::ones_like(const CpuTensor& t) {
    CpuTensor out(t.shape(), DType::kF32);
    auto* dst = reinterpret_cast<float*>(out.raw().data());
    const size_t n = out.numel();
    for (size_t i = 0; i < n; ++i) {
        dst[i] = 1.0f;
    }
    return out;
}

}  // namespace tensordiffgrad::core
