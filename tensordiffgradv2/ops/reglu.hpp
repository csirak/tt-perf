// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// ReGLU: ReLU-Gated Linear Unit composite operation
// ReGLU(x) = ReLU(x @ W_gate + b_gate) * (x @ W_value + b_value)
//
#pragma once

#include "../autograd/graph.hpp"

namespace tensordiffgradv2 {
namespace ops {

// ReGLU forward pass
// Args:
//   graph: The computation graph
//   x: Input activation [B, S, D]
//   w_gate, b_gate: Gate projection weights [D, F], [1, 1, F]
//   w_value, b_value: Value projection weights [D, F], [1, 1, F]
//   w_out, b_out: Output projection weights [F, D], [1, 1, D]
//   name: Name prefix for ops
//
// Returns: Output tensor [B, S, D]
//
// Logs hierarchical ops: composite "reglu" op with sub-ops as children
//
inline Value* reglu_forward(Graph& graph,
                            Value* x,
                            Value* w_gate, Value* b_gate,
                            Value* w_value, Value* b_value,
                            Value* w_out, Value* b_out,
                            const std::string& name) {
    // Start composite op - all sub-ops will have this as parent
    if (graph.logging_enabled()) {
        graph.recorder().push_composite(name);
    }

    // Gate path: ReLU(x @ W_gate + b_gate)
    auto* gate = x->matmul(w_gate, name + "_mm_gate");
    gate = gate->add(b_gate, name + "_bias_gate");
    gate = gate->relu(name + "_relu");

    // Value path: x @ W_value + b_value
    auto* value = x->matmul(w_value, name + "_mm_value");
    value = value->add(b_value, name + "_bias_value");

    // GLU: gate * value (elementwise)
    auto* h = gate->mul(value, name + "_glu");

    // Output projection: h @ W_out + b_out
    h = h->matmul(w_out, name + "_mm_out");
    h = h->add(b_out, name + "_bias_out");

    // End composite op - records it referencing first/last sub-op tensors
    if (graph.logging_enabled()) {
        graph.recorder().pop_composite(name);
    }

    return h;
}

}  // namespace ops
}  // namespace tensordiffgradv2
