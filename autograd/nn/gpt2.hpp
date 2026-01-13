#pragma once

#include "common.hpp"
#include "adam.hpp"

#include <stdexcept>
#include <type_traits>

namespace static_autograd {

// =============================================================================
// PersistentGPT2BFP8: GPT-2 with BFP8 FFN layers for high throughput
// =============================================================================
template<size_t N>
struct PersistentGPT2BFP8 {
    static_assert(N >= 1, "Must have at least 1 layer");

    std::vector<PersistentTransformerLayerBFP8> layers;
    Linear3D output_proj;

    Tensor input, d_input;
    Tensor target;
    Tensor loss, d_loss, diff;

    Graph graph;
    Value* input_node = nullptr;
    Value* loss_node = nullptr;
    bool built = false;

    uint32_t batch, seq, dim, vocab;
    float lr;

    PersistentGPT2BFP8(uint32_t b, uint32_t s, uint32_t d, uint32_t h, uint32_t ffn_mult,
                       uint32_t v, float learning_rate, MeshDevice& dev)
        : output_proj(b, s, d, v, 0.02f, dev),
          input(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_input(make_zeros(ttnn::Shape({b, s, d}), dev)),
          target(make_zeros(ttnn::Shape({b, s, v}), dev)),
          loss(make_zeros(ttnn::Shape({1, 1}), dev)),
          d_loss(make_zeros(ttnn::Shape({1, 1}), dev)),
          diff(make_zeros(ttnn::Shape({b, s, v}), dev)),
          batch(b), seq(s), dim(d), vocab(v),
          lr(learning_rate)
    {
        layers.reserve(N);
        for (size_t i = 0; i < N; ++i) {
            layers.emplace_back(b, s, d, h, ffn_mult, 0.02f, dev);
        }
    }

    void build() {
        input_node = graph.leaf(&input, &d_input, false);

        Value* h = input_node;
        for (size_t i = 0; i < N; ++i) {
            h = layers[i].forward(graph, h);
        }

        auto* logits = output_proj.forward(graph, h);
        loss_node = mse(graph, logits, &target, &loss, &d_loss, &diff);
        graph.build_topo(loss_node);
        built = true;
    }

    void train_step(MeshDevice* dev) {
        {
#ifdef TRACY_ENABLE
            ZoneScopedN("forward");
#endif
            if (!built) build();
            tt::tt_metal::distributed::Synchronize(dev, std::nullopt);
        }

        {
#ifdef TRACY_ENABLE
            ZoneScopedN("backward");
#endif
            graph.zero_grad();
            graph.backward(loss_node);
            tt::tt_metal::distributed::Synchronize(dev, std::nullopt);
        }

        {
#ifdef TRACY_ENABLE
            ZoneScopedN("sgd_step");
#endif
            for (size_t i = 0; i < N; ++i) {
                layers[i].sgd_step(lr);
            }
            output_proj.sgd_step(lr);
            tt::tt_metal::distributed::Synchronize(dev, std::nullopt);
        }

        built = false;  // Force rebuild next iteration (re-run forward)
    }
};

// =============================================================================
// PersistentGPT2: Full GPT-2 style model with N transformer layers
// Uses PersistentMLP pattern: build graph once, execute many times
// =============================================================================
template<size_t N>


struct PersistentGPT2 {
    static_assert(N >= 1, "Must have at least 1 layer");

    // Transformer layers
    std::vector<PersistentTransformerLayer> layers;

    // Output projection
    Linear3D output_proj;

    // Input/output buffers
    Tensor input;           // [B, S, D]
    Tensor d_input;
    Tensor target;          // [B, S, vocab]

    // Loss buffers
    Tensor loss;
    Tensor d_loss;
    Tensor diff;

    // Persistent graph
    Graph graph;
    Value* input_node = nullptr;
    Value* loss_node = nullptr;
    bool built = false;

    uint32_t batch, seq, dim, vocab;
    float lr;

    PersistentGPT2(uint32_t b, uint32_t s, uint32_t d, uint32_t h, uint32_t ffn_mult,
                   uint32_t v, float learning_rate, MeshDevice& dev)
        : output_proj(b, s, d, v, 0.02f, dev),
          input(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_input(make_zeros(ttnn::Shape({b, s, d}), dev)),
          target(make_zeros(ttnn::Shape({b, s, v}), dev)),
          loss(make_zeros(ttnn::Shape({1, 1}), dev)),
          d_loss(make_zeros(ttnn::Shape({1, 1}), dev)),
          diff(make_zeros(ttnn::Shape({b, s, v}), dev)),
          batch(b), seq(s), dim(d), vocab(v),
          lr(learning_rate)
    {
        layers.reserve(N);
        for (size_t i = 0; i < N; ++i) {
            layers.emplace_back(b, s, d, h, ffn_mult, 0.02f, dev);
        }
    }

    // Build computation graph (call once)
    void build() {
        input_node = graph.leaf(&input, &d_input, false);

        // Forward through transformer layers
        Value* h = input_node;
        for (size_t i = 0; i < N; ++i) {
            h = layers[i].forward(graph, h);
        }

        // Output projection
        auto* logits = output_proj.forward(graph, h);

        // MSE loss
        loss_node = mse(graph, logits, &target, &loss, &d_loss, &diff);
        graph.build_topo(loss_node);
        built = true;
    }

    // Execute forward (overwrites same buffers)
    void execute_forward() {
        // This would need to manually re-execute all ops
        // For now, rely on the graph's backward_fn capturing the right tensors
        // The key insight is that tensors are class members that get overwritten
    }

    // Full training step
    void train_step(MeshDevice* dev) {
        {
#ifdef TRACY_ENABLE
            ZoneScopedN("forward");
#endif
            if (!built) build();
            tt::tt_metal::distributed::Synchronize(dev, std::nullopt);
        }

        {
#ifdef TRACY_ENABLE
            ZoneScopedN("backward");
#endif
            graph.zero_grad();
            graph.backward(loss_node);
            tt::tt_metal::distributed::Synchronize(dev, std::nullopt);
        }

        {
#ifdef TRACY_ENABLE
            ZoneScopedN("sgd_step");
#endif
            for (size_t i = 0; i < N; ++i) {
                layers[i].sgd_step(lr);
            }
            output_proj.sgd_step(lr);
            tt::tt_metal::distributed::Synchronize(dev, std::nullopt);
        }

        built = false;  // Force rebuild next iteration (re-run forward)
    }

    float get_loss(MeshDevice* dev) {
        tt::tt_metal::distributed::Synchronize(dev, std::nullopt);
        return static_cast<float>(loss.cpu().to_vector<bfloat16>()[0]);
    }
};

// =============================================================================
// PersistentGrokGPT2T: GPT-2 with token + position embeddings (BF16)
// Single template covers DyT, LayerNorm, and RMSNorm variants.
// =============================================================================
enum class NormKind { DyT, LayerNorm, RMSNorm };

struct EmptyFinalNorm {};

template <NormKind K>
struct NormTraits;

template <>
struct NormTraits<NormKind::DyT> {
    using Layer = PersistentTransformerLayer;
    using FinalNorm = EmptyFinalNorm;
    static constexpr bool kHasFinal = false;
    static constexpr bool kNeedsEps = false;
};

template <>
struct NormTraits<NormKind::LayerNorm> {
    using Layer = PersistentTransformerLayerLN;
    using FinalNorm = LayerNorm;
    static constexpr bool kHasFinal = true;
    static constexpr bool kNeedsEps = true;
};

template <>
struct NormTraits<NormKind::RMSNorm> {
    using Layer = PersistentTransformerLayerRMS;
    using FinalNorm = RMSNorm;
    static constexpr bool kHasFinal = true;
    static constexpr bool kNeedsEps = true;
};

template <size_t N, NormKind K>
struct PersistentGrokGPT2T {
    static_assert(N >= 1, "Must have at least 1 layer");

    using LayerT = typename NormTraits<K>::Layer;
    using FinalNormT = typename NormTraits<K>::FinalNorm;

    Embedding tok;
    Embedding pos;
    std::vector<LayerT> layers;
    FinalNormT ln_final;
    Linear3D output_proj;

    Tensor token_indices;
    Tensor pos_indices;
    Tensor tok_plus_pos;
    Tensor d_tok_plus_pos;

    Graph graph;
    Value* token_node = nullptr;
    Value* pos_node = nullptr;
    Value* logits_node = nullptr;
    bool built = false;

    uint32_t batch, seq, dim, vocab;
    float lr;
    float ln_eps;
    MeshDevice* device;

    template <NormKind KK = K, std::enable_if_t<NormTraits<KK>::kNeedsEps, int> = 0>
    PersistentGrokGPT2T(uint32_t b, uint32_t s, uint32_t d, uint32_t h, uint32_t ffn_mult,
                        uint32_t v, float learning_rate, float ln_eps_, MeshDevice& dev)
        : tok(v, d, b, s, 1.0f, dev),
          pos(s, d, b, s, 1.0f, dev),
          ln_final(b, s, d, ln_eps_, dev),
          output_proj(b, s, d, v, std::sqrt(1.0f / d), dev),
          token_indices(make_indices(std::vector<uint32_t>(b * s, 0), ttnn::Shape({b, s}), dev)),
          pos_indices(make_indices(build_pos_indices(b, s), ttnn::Shape({b, s}), dev)),
          tok_plus_pos(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_tok_plus_pos(make_zeros(ttnn::Shape({b, s, d}), dev)),
          batch(b), seq(s), dim(d), vocab(v),
          lr(learning_rate),
          ln_eps(ln_eps_),
          device(&dev) {
        layers.reserve(N);
        float layer_init = std::sqrt(1.0f / d);
        for (size_t i = 0; i < N; ++i) {
            layers.emplace_back(b, s, d, h, ffn_mult, layer_init, dev, ln_eps_);
        }
    }

    template <NormKind KK = K, std::enable_if_t<!NormTraits<KK>::kNeedsEps, int> = 0>
    PersistentGrokGPT2T(uint32_t b, uint32_t s, uint32_t d, uint32_t h, uint32_t ffn_mult,
                        uint32_t v, float learning_rate, MeshDevice& dev)
        : tok(v, d, b, s, 1.0f, dev),
          pos(s, d, b, s, 1.0f, dev),
          output_proj(b, s, d, v, std::sqrt(1.0f / d), dev),
          token_indices(make_indices(std::vector<uint32_t>(b * s, 0), ttnn::Shape({b, s}), dev)),
          pos_indices(make_indices(build_pos_indices(b, s), ttnn::Shape({b, s}), dev)),
          tok_plus_pos(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_tok_plus_pos(make_zeros(ttnn::Shape({b, s, d}), dev)),
          batch(b), seq(s), dim(d), vocab(v),
          lr(learning_rate),
          ln_eps(0.0f),
          device(&dev) {
        layers.reserve(N);
        float layer_init = std::sqrt(1.0f / d);
        for (size_t i = 0; i < N; ++i) {
            layers.emplace_back(b, s, d, h, ffn_mult, layer_init, dev);
        }
    }

    static std::vector<uint32_t> build_pos_indices(uint32_t b, uint32_t s) {
        std::vector<uint32_t> indices(static_cast<size_t>(b) * s);
        for (uint32_t i = 0; i < b; ++i) {
            uint32_t base = i * s;
            for (uint32_t j = 0; j < s; ++j) {
                indices[base + j] = j;
            }
        }
        return indices;
    }

    void set_tokens(const std::vector<uint32_t>& tokens) {
        if (tokens.size() != static_cast<size_t>(batch) * seq) {
            throw std::runtime_error("token buffer size mismatch");
        }
        token_indices = make_indices(tokens, ttnn::Shape({batch, seq}), *device);
    }

    void build_graph() {
        token_node = graph.leaf(&token_indices, nullptr, false);
        pos_node = graph.leaf(&pos_indices, nullptr, false);

        auto* tok_v = tok.forward(graph, token_node);
        auto* pos_v = pos.forward(graph, pos_node);
        auto* h = add(graph, tok_v, pos_v, &tok_plus_pos, &d_tok_plus_pos);

        for (size_t i = 0; i < N; ++i) {
            h = layers[i].forward(graph, h);
        }

        if constexpr (NormTraits<K>::kHasFinal) {
            h = ln_final.forward(graph, h);
        }

        logits_node = output_proj.forward(graph, h);
        built = true;
    }

    Tensor* execute_forward() {
        auto* tok_out = tok.execute_forward(token_indices);
        auto* pos_out = pos.execute_forward(pos_indices);
        tok_plus_pos = ttnn::add(*tok_out, *pos_out);

        Tensor* h = &tok_plus_pos;
        for (size_t i = 0; i < N; ++i) {
            h = layers[i].execute_forward(*h);
        }

        if constexpr (NormTraits<K>::kHasFinal) {
            h = ln_final.execute_forward(*h);
        }

        output_proj.out = ttnn::add(ttnn::matmul(*h, output_proj.weight, false, true),
                                    output_proj.bias);
        return &output_proj.out;
    }

    void sgd_step(float lr_, float momentum = 0.0f, float weight_decay = 0.0f) {
        tok.sgd_step(lr_, momentum, weight_decay);
        pos.sgd_step(lr_, momentum, weight_decay);
        for (size_t i = 0; i < N; ++i) {
            layers[i].sgd_step(lr_, momentum, weight_decay);
        }
        if constexpr (NormTraits<K>::kHasFinal) {
            ln_final.sgd_step(lr_, momentum, weight_decay);
        }
        output_proj.sgd_step(lr_, momentum, weight_decay);
    }

    void register_adam(Adam& adam) {
        ADAM_REGISTER_EMBEDDING(adam, tok, *device);
        ADAM_REGISTER_EMBEDDING(adam, pos, *device);
        for (size_t i = 0; i < N; ++i) {
            layers[i].register_adam(adam, *device);
        }
        if constexpr (NormTraits<K>::kHasFinal) {
            ADAM_REGISTER_LAYERNORM(adam, ln_final, *device);
        }
        ADAM_REGISTER_LINEAR3D(adam, output_proj, *device);
    }
};

template <size_t N>
using PersistentGrokGPT2 = PersistentGrokGPT2T<N, NormKind::DyT>;

template <size_t N>
using PersistentGrokGPT2LN = PersistentGrokGPT2T<N, NormKind::LayerNorm>;

template <size_t N>
using PersistentGrokGPT2RMS = PersistentGrokGPT2T<N, NormKind::RMSNorm>;

} // namespace static_autograd
