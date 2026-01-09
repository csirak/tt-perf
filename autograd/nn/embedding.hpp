#pragma once

#include "common.hpp"
#include <ttnn/operations/embedding/embedding.hpp>
#include <ttnn/operations/embedding_backward/embedding_backward.hpp>

namespace static_autograd {

struct Embedding {
    Tensor weight;      // [vocab, dim] ROW_MAJOR
    Tensor d_weight;    // [vocab, dim] TILE_LAYOUT
    Tensor v_weight;    // [vocab, dim] velocity for momentum

    Tensor out;         // [B, S, dim]
    Tensor d_out;

    uint32_t vocab_size;
    uint32_t dim;
    uint32_t batch;
    uint32_t seq;

    Embedding(uint32_t vocab, uint32_t dim_, uint32_t batch_, uint32_t seq_, float init, MeshDevice& dev)
        : weight(make_embedding_weight(ttnn::Shape({vocab, dim_}), init, dev)),
          d_weight(make_zeros(ttnn::Shape({vocab, dim_}), dev)),
          v_weight(make_zeros(ttnn::Shape({vocab, dim_}), dev)),
          out(make_zeros(ttnn::Shape({batch_, seq_, dim_}), dev)),
          d_out(make_zeros(ttnn::Shape({batch_, seq_, dim_}), dev)),
          vocab_size(vocab),
          dim(dim_),
          batch(batch_),
          seq(seq_) {}

    Value* forward(Graph& g, Value* indices) {
#ifdef TRACY_ENABLE
        ZoneScopedN("Embedding_forward");
#endif
        auto* w = g.leaf(&weight, &d_weight, true);

        out = ttnn::embedding(*indices->data, weight, std::nullopt, ttnn::TILE_LAYOUT);

        auto* v = g.node(&out, &d_out);
        v->parents = {indices, w};
        v->backward_fn = [indices, w, this, v]() {
            if (!v->grad) {
                return;
            }

            auto d_out_reshaped = ttnn::reshape(*v->grad, ttnn::Shape({1, 1, batch * seq, dim}));
            auto d_weight_local = ttnn::embedding_bw(*indices->data, weight, d_out_reshaped,
                                                     ttnn::DataType::BFLOAT16);
            if (w->requires_grad) {
                w->accumulate_grad(d_weight_local);
            }
        };

        return v;
    }

    Tensor* execute_forward(const Tensor& indices) {
        out = ttnn::embedding(indices, weight, std::nullopt, ttnn::TILE_LAYOUT);
        return &out;
    }

    void sgd_step(float lr, float momentum = 0.0f, float weight_decay = 0.0f) {
        auto d_weight_rm = ttnn::untilize(d_weight);
        if (momentum > 0.0f) {
            // v = momentum * v + grad + wd * weight
            auto wd_term = ttnn::multiply(weight, weight_decay);
            v_weight = ttnn::add(
                ttnn::multiply(v_weight, momentum),
                ttnn::add(d_weight_rm, wd_term)
            );
            auto scaled = ttnn::multiply(v_weight, lr, std::nullopt, ttnn::L1_MEMORY_CONFIG);
            weight = ttnn::subtract(weight, scaled);
        } else if (weight_decay > 0.0f) {
            auto update = ttnn::add(d_weight_rm, ttnn::multiply(weight, weight_decay));
            auto scaled = ttnn::multiply(update, lr, std::nullopt, ttnn::L1_MEMORY_CONFIG);
            weight = ttnn::subtract(weight, scaled);
        } else {
            auto scaled = ttnn::multiply(d_weight_rm, lr, std::nullopt, ttnn::L1_MEMORY_CONFIG);
            weight = ttnn::subtract(weight, scaled);
        }
    }
};

} // namespace static_autograd
