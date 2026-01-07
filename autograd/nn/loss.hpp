#pragma once

#include "common.hpp"

namespace static_autograd {

struct LastTokenCrossEntropy {
    Tensor row_mask;      // [B*S, C]
    Tensor row_mask_1d;   // [B*S]
    Tensor one_hot_weight; // [C, C]

    Tensor softmax;       // [B*S, C]
    Tensor one_hot;       // [B*S, C]
    Tensor log_probs;     // [B*S, C]
    Tensor nll;           // [B*S]
    Tensor total;         // [1]
    Tensor loss;          // [1]
    Tensor d_loss;        // [1]

    std::vector<uint32_t> row_indices;
    std::vector<uint32_t> index_buf;

    uint32_t batch;
    uint32_t seq;
    uint32_t classes;
    uint32_t answer_pos;
    MeshDevice* device;

    LastTokenCrossEntropy(uint32_t b, uint32_t s, uint32_t c, uint32_t answer, MeshDevice& dev)
        : row_mask(make_zeros(ttnn::Shape({b * s, c}), dev)),
          row_mask_1d(make_zeros(ttnn::Shape({b * s}), dev)),
          one_hot_weight(make_zeros(ttnn::Shape({c, c}), dev)),
          softmax(make_zeros(ttnn::Shape({b * s, c}), dev)),
          one_hot(make_zeros(ttnn::Shape({b * s, c}), dev)),
          log_probs(make_zeros(ttnn::Shape({b * s, c}), dev)),
          nll(make_zeros(ttnn::Shape({b * s}), dev)),
          total(make_zeros(ttnn::Shape({1}), dev)),
          loss(make_zeros(ttnn::Shape({1}), dev)),
          d_loss(make_zeros(ttnn::Shape({1}), dev)),
          batch(b),
          seq(s),
          classes(c),
          answer_pos(answer),
          device(&dev) {
        row_indices.resize(batch);
        for (uint32_t i = 0; i < batch; ++i) {
            row_indices[i] = i * seq + answer_pos;
        }

        std::vector<bfloat16> mask_data(static_cast<size_t>(batch) * seq * classes, bfloat16(0.0f));
        std::vector<bfloat16> mask_1d_data(static_cast<size_t>(batch) * seq, bfloat16(0.0f));
        for (uint32_t i = 0; i < batch; ++i) {
            uint32_t row = row_indices[i];
            mask_1d_data[row] = bfloat16(1.0f);
            uint32_t row_offset = row * classes;
            for (uint32_t c_idx = 0; c_idx < classes; ++c_idx) {
                mask_data[row_offset + c_idx] = bfloat16(1.0f);
            }
        }

        std::vector<bfloat16> identity(static_cast<size_t>(classes) * classes, bfloat16(0.0f));
        for (uint32_t i = 0; i < classes; ++i) {
            identity[i * classes + i] = bfloat16(1.0f);
        }

        auto tile_layout = ttnn::TensorLayout(
            ttnn::DataType::BFLOAT16,
            ttnn::PageConfig(ttnn::TILE_LAYOUT),
            tt::tt_metal::MemoryConfig{}
        );

        row_mask = Tensor::from_vector(mask_data, ttnn::TensorSpec(ttnn::Shape({batch * seq, classes}), tile_layout))
                       .to_device(device);
        row_mask_1d = Tensor::from_vector(mask_1d_data, ttnn::TensorSpec(ttnn::Shape({batch * seq}), tile_layout))
                          .to_device(device);
        one_hot_weight = Tensor::from_vector(identity, ttnn::TensorSpec(ttnn::Shape({classes, classes}), tile_layout))
                             .to_device(device);

        index_buf.assign(batch * seq, 0);
    }

    Value* build(Graph& g, Value* logits) {
        auto* v = g.node(&loss, &d_loss);
        v->parents = {logits};
        v->backward_fn = [logits, this, v]() {
            if (!v->grad) {
                return;
            }
            auto grad_logits = ttnn::subtract(softmax, one_hot);
            grad_logits = ttnn::multiply(grad_logits, row_mask);
            grad_logits = ttnn::multiply(grad_logits, 1.0f / static_cast<float>(batch));
            grad_logits = ttnn::reshape(grad_logits, ttnn::Shape({batch, seq, classes}));
            logits->accumulate_grad(grad_logits);
        };
        return v;
    }

    void execute_forward(const Tensor& logits, const std::vector<uint32_t>& targets) {
        auto flat_logits = ttnn::reshape(logits, ttnn::Shape({batch * seq, classes}));

        std::fill(index_buf.begin(), index_buf.end(), 0);
        for (uint32_t i = 0; i < batch; ++i) {
            index_buf[row_indices[i]] = targets[i];
        }

        auto indices = make_indices(index_buf, ttnn::Shape({batch * seq}), *device);
        one_hot = ttnn::embedding(indices, one_hot_weight, std::nullopt, ttnn::TILE_LAYOUT);

        softmax = ttnn::softmax(flat_logits, -1);
        log_probs = ttnn::log(softmax);
        auto nll_raw = ttnn::sum(ttnn::multiply(one_hot, log_probs), -1, false);
        nll = ttnn::multiply(nll_raw, -1.0f);
        nll = ttnn::multiply(nll, row_mask_1d);
        total = ttnn::sum(nll, 0, false);
        loss = ttnn::multiply(total, 1.0f / static_cast<float>(batch));
    }

    float get_loss() {
        return static_cast<float>(loss.cpu().to_vector<bfloat16>()[0]);
    }
};

} // namespace static_autograd
