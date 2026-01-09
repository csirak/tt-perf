#pragma once

#include "common.hpp"
#include "adam.hpp"

#include <stdexcept>

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
// PersistentGrokGPT2: GPT-2 with token + position embeddings (BF16)
// Builds graph once and reuses it with manual forward execution.
// =============================================================================
template<size_t N>
struct PersistentGrokGPT2 {
    static_assert(N >= 1, "Must have at least 1 layer");

    Embedding tok;
    Embedding pos;
    std::vector<PersistentTransformerLayer> layers;
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
    MeshDevice* device;

    PersistentGrokGPT2(uint32_t b, uint32_t s, uint32_t d, uint32_t h, uint32_t ffn_mult,
                       uint32_t v, float learning_rate, MeshDevice& dev)
        // PyTorch-like initialization:
        // - Embedding: N(0, 1) → std=1.0
        // - Linear: kaiming uniform → std ≈ sqrt(1/fan_in)
        : tok(v, d, b, s, 1.0f, dev),  // N(0,1) for embedding
          pos(s, d, b, s, 1.0f, dev),  // N(0,1) for embedding
          output_proj(b, s, d, v, std::sqrt(1.0f / d), dev),  // kaiming for linear
          token_indices(make_indices(std::vector<uint32_t>(b * s, 0), ttnn::Shape({b, s}), dev)),
          pos_indices(make_indices(build_pos_indices(b, s), ttnn::Shape({b, s}), dev)),
          tok_plus_pos(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_tok_plus_pos(make_zeros(ttnn::Shape({b, s, d}), dev)),
          batch(b), seq(s), dim(d), vocab(v),
          lr(learning_rate),
          device(&dev) {
        layers.reserve(N);
        float layer_init = std::sqrt(1.0f / d);  // kaiming for linear layers
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
        output_proj.sgd_step(lr_, momentum, weight_decay);
    }

    // Register all parameters with Adam optimizer
    void register_adam(Adam& adam) {
        ADAM_REGISTER_EMBEDDING(adam, tok, *device);
        ADAM_REGISTER_EMBEDDING(adam, pos, *device);
        for (size_t i = 0; i < N; ++i) {
            layers[i].register_adam(adam, *device);
        }
        ADAM_REGISTER_LINEAR3D(adam, output_proj, *device);
    }
};

// =============================================================================
// PersistentGrokGPT2LN: Grok GPT-2 with LayerNorm (for parity/debug)
// =============================================================================
template <size_t N>
struct PersistentGrokGPT2LN {
    static_assert(N >= 1, "Must have at least 1 layer");

    Embedding tok;
    Embedding pos;
    std::vector<PersistentTransformerLayerLN> layers;
    LayerNorm ln_final;  // Final LayerNorm before output projection
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

    PersistentGrokGPT2LN(uint32_t b, uint32_t s, uint32_t d, uint32_t h, uint32_t ffn_mult,
                         uint32_t v, float learning_rate, float ln_eps_, MeshDevice& dev)
        // PyTorch-like initialization:
        // - Embedding: N(0, 1) → std=1.0
        // - Linear: kaiming uniform → std ≈ sqrt(1/fan_in)
        : tok(v, d, b, s, 1.0f, dev),  // N(0,1) for embedding
          pos(s, d, b, s, 1.0f, dev),  // N(0,1) for embedding
          ln_final(b, s, d, ln_eps_, dev),  // Final LN
          output_proj(b, s, d, v, std::sqrt(1.0f / d), dev),  // kaiming for linear
          token_indices(make_indices(std::vector<uint32_t>(b * s, 0), ttnn::Shape({b, s}), dev)),
          pos_indices(make_indices(PersistentGrokGPT2<N>::build_pos_indices(b, s), ttnn::Shape({b, s}), dev)),
          tok_plus_pos(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_tok_plus_pos(make_zeros(ttnn::Shape({b, s, d}), dev)),
          batch(b), seq(s), dim(d), vocab(v),
          lr(learning_rate),
          ln_eps(ln_eps_),
          device(&dev) {
        layers.reserve(N);
        float layer_init = std::sqrt(1.0f / d);  // kaiming for linear layers
        for (size_t i = 0; i < N; ++i) {
            layers.emplace_back(b, s, d, h, ffn_mult, layer_init, dev, ln_eps);
        }
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

        // Final LayerNorm before output projection
        h = ln_final.forward(graph, h);

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

        // Final LayerNorm before output projection
        h = ln_final.execute_forward(*h);

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
        ln_final.sgd_step(lr_, momentum, weight_decay);
        output_proj.sgd_step(lr_, momentum, weight_decay);
    }

    // Register all parameters with Adam optimizer
    void register_adam(Adam& adam) {
        ADAM_REGISTER_EMBEDDING(adam, tok, *device);
        ADAM_REGISTER_EMBEDDING(adam, pos, *device);
        for (size_t i = 0; i < N; ++i) {
            layers[i].register_adam(adam, *device);
        }
        ADAM_REGISTER_LAYERNORM(adam, ln_final, *device);
        ADAM_REGISTER_LINEAR3D(adam, output_proj, *device);
    }
};

// =============================================================================
// PersistentGPT2MXP: GPT-2 with batched typecast for BFP8 FFN
// Key difference: TypecastCache batches all BFP8→BF16 conversions before backward
// =============================================================================
template<size_t N>


struct PersistentGPT2MXP {
    static_assert(N >= 1, "Must have at least 1 layer");

    std::vector<TransformerLayerMXP> layers;
    Linear3D output_proj;

    // TypecastCache for batched BFP8→BF16 conversion
    TypecastCache ffn_cache;

    Tensor input, d_input;
    Tensor target;
    Tensor loss, d_loss, diff;

    Graph graph;
    Value* input_node = nullptr;
    Value* loss_node = nullptr;
    bool built = false;
    bool cache_graph = std::getenv("MXP_DISABLE_GRAPH_CACHE") == nullptr;
    uint32_t weight_typecast_every = 1;
    uint64_t step = 0;

    uint32_t batch, seq, dim, vocab;
    float lr;
    MeshDevice* device;

    PersistentGPT2MXP(uint32_t b, uint32_t s, uint32_t d, uint32_t h, uint32_t ffn_mult,
                      uint32_t v, float learning_rate, MeshDevice& dev)
        : output_proj(b, s, d, v, 0.02f, dev),
          input(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_input(make_zeros(ttnn::Shape({b, s, d}), dev)),
          target(make_zeros(ttnn::Shape({b, s, v}), dev)),
          loss(make_zeros(ttnn::Shape({1, 1}), dev)),
          d_loss(make_zeros(ttnn::Shape({1, 1}), dev)),
          diff(make_zeros(ttnn::Shape({b, s, v}), dev)),
          batch(b), seq(s), dim(d), vocab(v),
          lr(learning_rate),
          device(&dev)
    {
        layers.reserve(N);
        for (size_t i = 0; i < N; ++i) {
            layers.emplace_back(b, s, d, h, ffn_mult, 0.02f, dev);
        }
    }

    void build_graph() {
#ifdef TRACY_ENABLE
        ZoneScopedN("PersistentGPT2MXP_build");
#endif
        input_node = graph.leaf(&input, &d_input, false);

        Value* h = input_node;
        for (size_t i = 0; i < N; ++i) {
            h = layers[i].forward(graph, h);
        }

        auto* logits = output_proj.forward(graph, h);
        loss_node = mse(graph, logits, &target, &loss, &d_loss, &diff);
        graph.build_topo(loss_node);

        // Register all FFN caches for batched typecast
        ffn_cache.clear();
        for (size_t i = 0; i < N; ++i) {
            layers[i].register_caches(ffn_cache);
        }

        built = true;
    }

    // Execute forward without graph construction (reuses buffers)
    void execute_forward() {
        Tensor* h = &input;
        for (size_t i = 0; i < N; ++i) {
            h = layers[i].execute_forward(*h);
        }

        // Output projection
        output_proj.out = ttnn::add(ttnn::matmul(*h, output_proj.weight, false, true), output_proj.bias);

        // MSE loss
        diff = ttnn::subtract(output_proj.out, target);
        auto sq = ttnn::multiply(diff, diff);
        loss = ttnn::mean(sq, std::nullopt, true);
    }

    void train_step() {
#ifdef TRACY_ENABLE
        ZoneScopedN("PersistentGPT2MXP_train_step");
#endif
        {
#ifdef TRACY_ENABLE
            ZoneScopedN("forward");
#endif
            if (!built || !cache_graph) {
                build_graph();
            }
            if (cache_graph) {
                execute_forward();
            }
            tt::tt_metal::distributed::Synchronize(device, std::nullopt);
        }

        // *** MXP: Batch typecast all FFN activations BEFORE backward ***
        {
#ifdef TRACY_ENABLE
            ZoneScopedN("batch_typecast_ffn");
#endif
            ffn_cache.convert_all();
            tt::tt_metal::distributed::Synchronize(device, std::nullopt);
        }

        // Backward pass (uses pre-casted BF16 tensors)
        {
#ifdef TRACY_ENABLE
            ZoneScopedN("backward");
#endif
            graph.zero_grad();
            graph.backward(loss_node);
            tt::tt_metal::distributed::Synchronize(device, std::nullopt);
        }

        // SGD step
        {
#ifdef TRACY_ENABLE
            ZoneScopedN("sgd_step");
#endif
            for (size_t i = 0; i < N; ++i) {
                layers[i].sgd_step(lr);
            }
            output_proj.sgd_step(lr);
            tt::tt_metal::distributed::Synchronize(device, std::nullopt);
        }

        if (!cache_graph) {
            built = false;  // Rebuild graph each iteration when cache is disabled
        }
    }

    float get_loss() {
        tt::tt_metal::distributed::Synchronize(device, std::nullopt);
        return static_cast<float>(loss.cpu().to_vector<bfloat16>()[0]);
    }
};

// =============================================================================
// PersistentGPT2MXP2: GPT-2 with FULLY batched typecast
// Key difference from MXP: Also batches weight BF16→BFP8 conversion after SGD
//
// Phase flow:
// 1. forward (BFP8 compute)
// 2. batch_typecast_ffn (activation BFP8→BF16 for backward)
// 3. backward (BF16 gradients)
// 4. sgd_step_bf16_only (update BF16 masters, NO typecast)
// 5. batch_typecast_weights (BF16→BFP8 for next forward) <- NEW
// =============================================================================
template<size_t N>


struct PersistentGPT2MXP2 {
    static_assert(N >= 1, "Must have at least 1 layer");

    std::vector<TransformerLayerMXP> layers;
    Linear3D output_proj;

    // TypecastCache for batched BFP8→BF16 conversion (activations before backward)
    TypecastCache ffn_cache;

    // WeightTypecastCache for batched BF16→BFP8 conversion (weights after SGD)
    WeightTypecastCache weight_cache;

    Tensor input, d_input;
    Tensor target;
    Tensor loss, d_loss, diff;

    Graph graph;
    Value* input_node = nullptr;
    Value* loss_node = nullptr;
    bool built = false;
    bool cache_graph = std::getenv("MXP_DISABLE_GRAPH_CACHE") == nullptr;
    uint32_t weight_typecast_every = 1;
    uint64_t step = 0;

    uint32_t batch, seq, dim, vocab;
    float lr;
    MeshDevice* device;

    PersistentGPT2MXP2(uint32_t b, uint32_t s, uint32_t d, uint32_t h, uint32_t ffn_mult,
                       uint32_t v, float learning_rate, MeshDevice& dev)
        : output_proj(b, s, d, v, 0.02f, dev),
          input(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_input(make_zeros(ttnn::Shape({b, s, d}), dev)),
          target(make_zeros(ttnn::Shape({b, s, v}), dev)),
          loss(make_zeros(ttnn::Shape({1, 1}), dev)),
          d_loss(make_zeros(ttnn::Shape({1, 1}), dev)),
          diff(make_zeros(ttnn::Shape({b, s, v}), dev)),
          batch(b), seq(s), dim(d), vocab(v),
          lr(learning_rate),
          device(&dev)
    {
        layers.reserve(N);
        for (size_t i = 0; i < N; ++i) {
            layers.emplace_back(b, s, d, h, ffn_mult, 0.02f, dev);
        }

        if (const char* env = std::getenv("MXP_WEIGHT_TYPECAST_EVERY")) {
            int val = std::atoi(env);
            weight_typecast_every = val > 0 ? static_cast<uint32_t>(val) : 1;
        }
    }

    void build_graph() {
#ifdef TRACY_ENABLE
        ZoneScopedN("PersistentGPT2MXP2_build");
#endif
        input_node = graph.leaf(&input, &d_input, false);

        Value* h = input_node;
        for (size_t i = 0; i < N; ++i) {
            h = layers[i].forward(graph, h);
        }

        auto* logits = output_proj.forward(graph, h);
        loss_node = mse(graph, logits, &target, &loss, &d_loss, &diff);
        graph.build_topo(loss_node);

        // Register all FFN caches for batched typecast (activations)
        ffn_cache.clear();
        for (size_t i = 0; i < N; ++i) {
            layers[i].register_caches(ffn_cache);
        }

        // Register all FFN weight caches for batched typecast (weights)
        weight_cache.clear();
        for (size_t i = 0; i < N; ++i) {
            layers[i].register_weight_caches(weight_cache);
        }

        built = true;
    }

    // Execute forward without graph construction (reuses buffers)
    void execute_forward() {
        Tensor* h = &input;
        for (size_t i = 0; i < N; ++i) {
            h = layers[i].execute_forward(*h);
        }

        // Output projection
        output_proj.out = ttnn::add(ttnn::matmul(*h, output_proj.weight, false, true), output_proj.bias);

        // MSE loss
        diff = ttnn::subtract(output_proj.out, target);
        auto sq = ttnn::multiply(diff, diff);
        loss = ttnn::mean(sq, std::nullopt, true);
    }

    void train_step() {
#ifdef TRACY_ENABLE
        ZoneScopedN("PersistentGPT2MXP2_train_step");
#endif
        // 1. Forward pass (BFP8 compute)
        {
#ifdef TRACY_ENABLE
            ZoneScopedN("forward");
#endif
            if (!built || !cache_graph) {
                build_graph();
            }
            if (cache_graph) {
                execute_forward();
            }
            tt::tt_metal::distributed::Synchronize(device, std::nullopt);
        }

        // 2. Batch typecast all FFN activations BFP8→BF16 BEFORE backward
        {
#ifdef TRACY_ENABLE
            ZoneScopedN("batch_typecast_ffn");
#endif
            ffn_cache.convert_all();
            tt::tt_metal::distributed::Synchronize(device, std::nullopt);
        }

        // 3. Backward pass (uses pre-casted BF16 tensors)
        {
#ifdef TRACY_ENABLE
            ZoneScopedN("backward");
#endif
            graph.zero_grad();
            graph.backward(loss_node);
            tt::tt_metal::distributed::Synchronize(device, std::nullopt);
        }

        // 4. SGD step (BF16 only - no inline typecast)
        {
#ifdef TRACY_ENABLE
            ZoneScopedN("sgd_step_bf16_only");
#endif
            for (size_t i = 0; i < N; ++i) {
                layers[i].sgd_step_bf16_only(lr);
            }
            output_proj.sgd_step(lr);  // output_proj is BF16, normal sgd
            tt::tt_metal::distributed::Synchronize(device, std::nullopt);
        }

        // 5. Batch typecast all FFN weights BF16→BFP8 for next forward
        {
#ifdef TRACY_ENABLE
            ZoneScopedN("batch_typecast_weights");
#endif
            if (weight_typecast_every == 1 || (step % weight_typecast_every) == 0) {
                weight_cache.convert_all();
                tt::tt_metal::distributed::Synchronize(device, std::nullopt);
            }
        }

        if (!cache_graph) {
            built = false;  // Rebuild graph each iteration when cache is disabled
        }

        step += 1;
    }

    float get_loss() {
        tt::tt_metal::distributed::Synchronize(device, std::nullopt);
        return static_cast<float>(loss.cpu().to_vector<bfloat16>()[0]);
    }
};

} // namespace static_autograd
