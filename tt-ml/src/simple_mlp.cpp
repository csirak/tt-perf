// Simple MLP training test using tt-train's TTML library
// Comparison version: matches our traced autograd config exactly
// - batch=1024, 512->256->128
// - constant 0.01 weight init, zero bias
// - SGD with no momentum
// - TTNN tracing enabled

#include <fmt/format.h>
#include <vector>
#include <chrono>

#include <core/ttnn_all_includes.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/operations/trace.hpp>
#include <tt-metalium/distributed.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/losses.hpp"
#include "ops/linear_op.hpp"
#include "ops/unary_ops.hpp"

using ttml::autograd::TensorPtr;
using MeshDevice = tt::tt_metal::distributed::MeshDevice;
using MeshTraceId = tt::tt_metal::distributed::MeshTraceId;

// Custom Linear layer with constant weight initialization
struct ConstantLinear {
    TensorPtr weight;
    TensorPtr bias;

    ConstantLinear(uint32_t in_features, uint32_t out_features, float init_val, MeshDevice* device) {
        // Create weight with constant value (like our autograd)
        auto weight_shape = ttnn::Shape({1, 1, out_features, in_features});
        auto weight_tensor = ttnn::full(weight_shape, init_val, ttnn::DataType::BFLOAT16,
                                        ttnn::TILE_LAYOUT, *device);
        weight = ttml::autograd::create_tensor(weight_tensor);

        // Create bias with zeros (like our autograd)
        auto bias_shape = ttnn::Shape({1, 1, 1, out_features});
        auto bias_tensor = ttnn::zeros(bias_shape, ttnn::DataType::BFLOAT16,
                                       ttnn::TILE_LAYOUT, *device);
        bias = ttml::autograd::create_tensor(bias_tensor);
    }

    TensorPtr operator()(const TensorPtr& input) {
        return ttml::ops::linear_op(input, weight, bias);
    }

    std::vector<TensorPtr> parameters() {
        return {weight, bias};
    }
};

// Simple 2-layer MLP with constant init
struct SimpleMLP {
    ConstantLinear fc1;
    ConstantLinear fc2;

    SimpleMLP(uint32_t in_f, uint32_t hidden, uint32_t out_f, float init, MeshDevice* dev)
        : fc1(in_f, hidden, init, dev), fc2(hidden, out_f, init, dev) {}

    TensorPtr operator()(const TensorPtr& x) {
        auto h = fc1(x);
        h = ttml::ops::relu(h);
        return fc2(h);
    }

    std::vector<TensorPtr> parameters() {
        auto p1 = fc1.parameters();
        auto p2 = fc2.parameters();
        p1.insert(p1.end(), p2.begin(), p2.end());
        return p1;
    }
};

// Simple SGD step (no momentum to match our autograd)
void sgd_step(std::vector<TensorPtr>& params, float lr) {
    for (auto& p : params) {
        if (p->is_grad_initialized()) {
            auto val = p->get_value();
            auto grad = p->get_grad();
            p->set_value(ttnn::subtract(val, ttnn::multiply(grad, lr)));
        }
    }
}

int main() {
    fmt::print("=== TTML MLP Comparison Test ===\n\n");

    // Configuration - matches our traced autograd exactly
    const uint32_t batch_size = 1024;
    const uint32_t in_features = 512;
    const uint32_t hidden_features = 256;
    const uint32_t out_features = 128;
    const uint32_t num_steps = 5;
    const float learning_rate = 0.01f;
    const float init_val = 0.01f;

    fmt::print("Config:\n");
    fmt::print("  batch_size: {}\n", batch_size);
    fmt::print("  in_features: {}\n", in_features);
    fmt::print("  hidden_features: {}\n", hidden_features);
    fmt::print("  out_features: {}\n", out_features);
    fmt::print("  learning_rate: {}\n", learning_rate);
    fmt::print("  init_val: {}\n", init_val);
    fmt::print("  num_steps: {}\n\n", num_steps);

    // Get device from autograd context
    auto* device = &ttml::autograd::ctx().get_device();
    fmt::print("Device initialized\n\n");

    // Create model with constant weight init
    SimpleMLP model(in_features, hidden_features, out_features, init_val, device);
    auto params = model.parameters();

    // Create static input and target tensors (constant, like our autograd)
    std::vector<float> input_data(batch_size * in_features, 1.0f);
    std::vector<float> target_data(batch_size * out_features, 0.5f);

    auto input = ttml::autograd::create_tensor(
        ttml::core::from_vector(input_data, ttnn::Shape({batch_size, 1, 1, in_features}), device));
    auto target = ttml::autograd::create_tensor(
        ttml::core::from_vector(target_data, ttnn::Shape({batch_size, 1, 1, out_features}), device));

    // Lambda for one training step
    auto train_step = [&]() {
        auto output = model(input);
        auto loss = ttml::ops::mse_loss(output, target);
        loss->backward();
        sgd_step(params, learning_rate);
        ttml::autograd::ctx().reset_graph();
        return loss;
    };

    // =========================================================================
    // Check initial forward pass
    // =========================================================================
    fmt::print("=== Initial Forward Pass (before training) ===\n");
    {
        auto output = model(input);
        auto loss = ttml::ops::mse_loss(output, target);
        auto loss_val = ttml::core::to_vector(loss->get_value())[0];
        fmt::print("Initial loss: {:.6f}\n", loss_val);

        // Print output stats
        auto out_vec = ttml::core::to_vector(output->get_value());
        float sum = 0;
        for (auto v : out_vec) sum += v;
        fmt::print("Output mean: {:.6f} (expected ~5.12 for init=0.01, hidden=256)\n", sum / out_vec.size());
        ttml::autograd::ctx().reset_graph();
    }
    fmt::print("\n");

    // =========================================================================
    // Non-traced execution (warmup + measurement)
    // =========================================================================
    fmt::print("=== Non-traced Execution ===\n");

    // Re-create model for fair measurement
    SimpleMLP model_fresh(in_features, hidden_features, out_features, init_val, device);
    auto params_fresh = model_fresh.parameters();

    auto train_step_fresh = [&]() {
        auto output = model_fresh(input);
        auto loss = ttml::ops::mse_loss(output, target);
        loss->backward();
        sgd_step(params_fresh, learning_rate);
        ttml::autograd::ctx().reset_graph();
        return loss;
    };

    // Warmup (compiles kernels but doesn't affect measurement model)
    for (uint32_t i = 0; i < 2; ++i) {
        train_step();  // Use original model for warmup
    }
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);

    // Measure non-traced
    auto start_nontrace = std::chrono::high_resolution_clock::now();
    TensorPtr loss;
    for (uint32_t step = 0; step < num_steps; ++step) {
        loss = train_step_fresh();
    }
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);
    auto end_nontrace = std::chrono::high_resolution_clock::now();

    double nontrace_ms = std::chrono::duration<double, std::milli>(end_nontrace - start_nontrace).count();
    double nontrace_per_iter = nontrace_ms / num_steps;

    auto loss_nontrace = ttml::core::to_vector(loss->get_value())[0];
    fmt::print("Loss after {} non-traced steps: {:.6f}\n", num_steps, loss_nontrace);
    fmt::print("Non-traced: {:.3f} ms/iter\n\n", nontrace_per_iter);

    // =========================================================================
    // Traced execution
    // =========================================================================
    fmt::print("=== Traced Execution ===\n");

    // Need to re-initialize model for fair comparison
    SimpleMLP model2(in_features, hidden_features, out_features, init_val, device);
    auto params2 = model2.parameters();

    auto train_step2 = [&]() {
        auto output = model2(input);
        auto loss = ttml::ops::mse_loss(output, target);
        loss->backward();
        sgd_step(params2, learning_rate);
        ttml::autograd::ctx().reset_graph();
        return loss;
    };

    // Warmup without trace (load kernels)
    for (uint32_t i = 0; i < 2; ++i) {
        train_step2();
    }
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);

    // Capture trace
    auto trace_id = ttnn::operations::trace::begin_trace_capture(device, std::nullopt);
    loss = train_step2();
    ttnn::operations::trace::end_trace_capture(device, trace_id, std::nullopt);
    fmt::print("Trace captured\n");

    // Warmup trace replay
    for (uint32_t i = 0; i < 2; ++i) {
        ttnn::operations::trace::execute_trace(device, trace_id, std::nullopt, false);
    }
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);

    // Measure traced
    auto start_trace = std::chrono::high_resolution_clock::now();
    for (uint32_t step = 0; step < num_steps; ++step) {
        ttnn::operations::trace::execute_trace(device, trace_id, std::nullopt, false);
    }
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);
    auto end_trace = std::chrono::high_resolution_clock::now();

    double trace_ms = std::chrono::duration<double, std::milli>(end_trace - start_trace).count();
    double trace_per_iter = trace_ms / num_steps;

    // Get final loss (need to sync and read)
    auto loss_trace = ttml::core::to_vector(loss->get_value())[0];
    fmt::print("Loss after {} traced steps: {:.6f}\n", num_steps, loss_trace);
    fmt::print("Traced: {:.3f} ms/iter\n\n", trace_per_iter);

    // Release trace
    ttnn::operations::trace::release_trace(device, trace_id);

    // =========================================================================
    // Summary
    // =========================================================================
    fmt::print("============================================================\n");
    fmt::print("Performance Summary\n");
    fmt::print("------------------------------------------------------------\n");
    fmt::print("  Non-traced: {:.3f} ms/iter\n", nontrace_per_iter);
    fmt::print("  Traced:     {:.3f} ms/iter\n", trace_per_iter);
    fmt::print("  Speedup:    {:.2f}x\n", nontrace_per_iter / trace_per_iter);
    fmt::print("============================================================\n");

    // IMPORTANT: Must explicitly close device to trigger profiler dump.
    ttml::autograd::ctx().close_device();

    return 0;
}
