#pragma once

#include "common.hpp"
#include <ttnn/operations/data_movement/untilize/untilize.hpp>
#include <ttnn/operations/data_movement/tilize/tilize.hpp>
#include <vector>

namespace static_autograd {

// =============================================================================
// Adam/AdamW Optimizer for TTNN
//
// Adam: Adaptive Moment Estimation
// m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
// v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
// m_hat = m_t / (1 - beta1^t)
// v_hat = v_t / (1 - beta2^t)
// w = w - lr * m_hat / (sqrt(v_hat) + eps)
//
// AdamW adds decoupled weight decay: w = w - lr * wd * w
// =============================================================================

struct AdamParam {
    Tensor* weight;      // Pointer to weight tensor (bf16 on device)
    Tensor* grad;        // Pointer to gradient tensor
    Tensor m;            // First moment (mean of gradients) - always TILE_LAYOUT
    Tensor v;            // Second moment (variance of gradients) - always TILE_LAYOUT
    std::vector<float> weight_fp32;  // Master weights in float32 for precision
    bool apply_wd;       // Whether to apply weight decay (false for bias/LN params)
    bool is_row_major;   // True if weight is ROW_MAJOR (needs untilize for final update)

    AdamParam(Tensor* w, Tensor* g, MeshDevice& dev, bool wd = true, bool row_major = false)
        : weight(w), grad(g), apply_wd(wd), is_row_major(row_major) {
        auto shape = w->logical_shape();
        // Always use TILE_LAYOUT for m and v (required by TTNN ops like sqrt)
        m = make_zeros(shape, dev);
        v = make_zeros(shape, dev);

        // Initialize fp32 master weights from bf16 weights
        auto w_bf16 = w->cpu().template to_vector<bfloat16>();
        weight_fp32.resize(w_bf16.size());
        for (size_t i = 0; i < w_bf16.size(); ++i) {
            weight_fp32[i] = static_cast<float>(w_bf16[i]);
        }
    }
};

struct Adam {
    std::vector<AdamParam> params;
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    int step_count;
    MeshDevice* device;

    Adam(float learning_rate = 1e-3f, float b1 = 0.9f, float b2 = 0.999f,
         float epsilon = 1e-8f, float wd = 0.0f)
        : lr(learning_rate), beta1(b1), beta2(b2), eps(epsilon),
          weight_decay(wd), step_count(0), device(nullptr) {}

    // Register a parameter with its gradient
    void add_param(Tensor* weight, Tensor* grad, MeshDevice& dev, bool apply_wd = true, bool row_major = false) {
        if (!device) device = &dev;
        params.emplace_back(weight, grad, dev, apply_wd, row_major);
    }

    // Perform one optimization step
    void step() {
#ifdef TRACY_ENABLE
        ZoneScopedN("Adam_step");
#endif
        step_count++;

        // Bias correction factors
        float bias_correction1 = 1.0f - std::pow(beta1, step_count);
        float bias_correction2 = 1.0f - std::pow(beta2, step_count);

        // Effective learning rate with bias correction
        float step_size = lr * std::sqrt(bias_correction2) / bias_correction1;


        for (auto& p : params) {
#ifdef TRACY_ENABLE
            ZoneScopedN("Adam_update_param");
#endif
            // Gradient is always TILE_LAYOUT (all d_weight tensors are TILE_LAYOUT)
            const Tensor& grad = *p.grad;

            // m = beta1 * m + (1 - beta1) * grad
            auto grad_scaled = ttnn::multiply(grad, 1.0f - beta1);
            auto m_scaled = ttnn::multiply(p.m, beta1);
            p.m = ttnn::add(m_scaled, grad_scaled);

            // v = beta2 * v + (1 - beta2) * grad^2
            auto grad_sq = ttnn::square(grad);
            auto grad_sq_scaled = ttnn::multiply(grad_sq, 1.0f - beta2);
            auto v_scaled = ttnn::multiply(p.v, beta2);
            p.v = ttnn::add(v_scaled, grad_sq_scaled);

            // update = step_size * m / (sqrt(v) + eps)
            auto sqrt_v = ttnn::sqrt(p.v);
            auto denom = ttnn::add(sqrt_v, eps);
            auto ratio = ttnn::divide(p.m, denom);
            auto update_tiled = ttnn::multiply(ratio, step_size);

            // Reshape update to match weight shape if needed (grad may be 4D, weight 2D)
            auto weight_shape = p.weight->logical_shape();
            auto update_shape = update_tiled.logical_shape();
            if (update_shape != weight_shape) {
                update_tiled = ttnn::reshape(update_tiled, weight_shape);
            }

            // For ROW_MAJOR weights (e.g., embedding), untilize the update
            Tensor update = p.is_row_major ? ttnn::untilize(update_tiled) : update_tiled;

            // Get update values on CPU for fp32 master weight update
            auto upd_cpu = update.cpu().template to_vector<bfloat16>();

            // Update fp32 master weights with AdamW formula
            // Using fp32 master weights avoids bf16 precision loss for small updates
            float wd_mult = (weight_decay > 0.0f && p.apply_wd) ? lr * weight_decay : 0.0f;
            for (size_t i = 0; i < p.weight_fp32.size(); ++i) {
                float u = static_cast<float>(upd_cpu[i]);
                p.weight_fp32[i] = p.weight_fp32[i] * (1.0f - wd_mult) - u;
            }

            // Convert fp32 master weights back to bf16 tensor
            std::vector<bfloat16> new_w_bf16(p.weight_fp32.size());
            for (size_t i = 0; i < p.weight_fp32.size(); ++i) {
                new_w_bf16[i] = bfloat16(p.weight_fp32[i]);
            }

            // Create bf16 tensor and copy to device
            auto w_shape = p.weight->logical_shape();
            auto layout = p.is_row_major ? ttnn::ROW_MAJOR_LAYOUT : ttnn::TILE_LAYOUT;
            auto tensor_spec = ttnn::TensorSpec(
                w_shape,
                tt::tt_metal::TensorLayout(ttnn::DataType::BFLOAT16, ttnn::PageConfig(layout), ttnn::types::DRAM_MEMORY_CONFIG)
            );
            *p.weight = ttnn::Tensor::from_vector(std::move(new_w_bf16), tensor_spec, device);

        }
    }

    // Zero all gradients
    void zero_grad() {
#ifdef TRACY_ENABLE
        ZoneScopedN("Adam_zero_grad");
#endif
        for (auto& p : params) {
            *p.grad = ttnn::zeros_like(*p.grad);
        }
    }

    // Reset optimizer state (for fresh training)
    void reset() {
        step_count = 0;
        for (auto& p : params) {
            p.m = ttnn::zeros_like(p.m);
            p.v = ttnn::zeros_like(p.v);
        }
    }

    // Get current step count
    int get_step() const { return step_count; }
};

// =============================================================================
// Helper macros for registering parameters from layers
// =============================================================================

// Register Linear3D parameters
#define ADAM_REGISTER_LINEAR3D(adam, layer, dev) do { \
    (adam).add_param(&(layer).weight, &(layer).d_weight, (dev), true); \
    (adam).add_param(&(layer).bias, &(layer).d_bias, (dev), false); \
} while(0)

// Register Embedding parameters (ROW_MAJOR weight, need untilize for grad)
#define ADAM_REGISTER_EMBEDDING(adam, layer, dev) do { \
    (adam).add_param(&(layer).weight, &(layer).d_weight, (dev), true, true); \
} while(0)

// Register LayerNorm parameters (no weight decay)
#define ADAM_REGISTER_LAYERNORM(adam, layer, dev) do { \
    (adam).add_param(&(layer).gamma, &(layer).d_gamma, (dev), false); \
    (adam).add_param(&(layer).beta, &(layer).d_beta, (dev), false); \
} while(0)

// Register DyT parameters (no weight decay on normalization params)
#define ADAM_REGISTER_DYT(adam, layer, dev) do { \
    (adam).add_param(&(layer).alpha, &(layer).d_alpha, (dev), false); \
    (adam).add_param(&(layer).gamma, &(layer).d_gamma, (dev), false); \
    (adam).add_param(&(layer).beta, &(layer).d_beta, (dev), false); \
} while(0)

} // namespace static_autograd
