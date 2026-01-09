// Test: ROW_MAJOR subtract behavior
// Checks if ttnn::subtract corrupts ROW_MAJOR tensors

#include "common.hpp"
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/data_movement/untilize/untilize.hpp>
#include <ttnn/operations/data_movement/tilize/tilize.hpp>
#include <ttnn/operations/rand/rand.hpp>

using namespace test;
using Tensor = ttnn::Tensor;

int main() {
    DeviceGuard guard;
    auto& device = *guard.device;

    // Shape matching embedding: [vocab=128, dim=64]
    ttnn::Shape shape({128, 64});

    fmt::print("=== Test: ROW_MAJOR subtract behavior ===\n\n");

    // Create ROW_MAJOR weight (like embedding)
    auto weight_tiled = ttnn::rand(shape, device, ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT,
                                    ttnn::types::DRAM_MEMORY_CONFIG, -1.0f, 1.0f, 42);
    auto weight = ttnn::untilize(weight_tiled);  // Convert to ROW_MAJOR

    // Create ROW_MAJOR update (like what Adam does after untilize)
    auto update_tiled = ttnn::rand(shape, device, ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT,
                                    ttnn::types::DRAM_MEMORY_CONFIG, -0.01f, 0.01f, 43);
    auto update = ttnn::untilize(update_tiled);  // Convert to ROW_MAJOR

    // Get initial values
    auto weight_cpu = weight.cpu().to_vector<bfloat16>();
    auto update_cpu = update.cpu().to_vector<bfloat16>();

    double weight_norm = 0, update_norm = 0;
    for (auto v : weight_cpu) weight_norm += static_cast<float>(v) * static_cast<float>(v);
    for (auto v : update_cpu) update_norm += static_cast<float>(v) * static_cast<float>(v);
    weight_norm = std::sqrt(weight_norm);
    update_norm = std::sqrt(update_norm);

    fmt::print("Before subtract:\n");
    fmt::print("  weight_norm = {:.6f}\n", weight_norm);
    fmt::print("  update_norm = {:.6f}\n", update_norm);
    fmt::print("  weight[0:5] = [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]\n",
               float(weight_cpu[0]), float(weight_cpu[1]), float(weight_cpu[2]),
               float(weight_cpu[3]), float(weight_cpu[4]));
    fmt::print("  update[0:5] = [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]\n",
               float(update_cpu[0]), float(update_cpu[1]), float(update_cpu[2]),
               float(update_cpu[3]), float(update_cpu[4]));

    // Compute expected result manually
    std::vector<float> expected(weight_cpu.size());
    for (size_t i = 0; i < weight_cpu.size(); ++i) {
        expected[i] = static_cast<float>(weight_cpu[i]) - static_cast<float>(update_cpu[i]);
    }

    // Do the subtract (this is what Adam does)
    auto result = ttnn::subtract(weight, update);

    // Check result
    auto result_cpu = result.cpu().to_vector<bfloat16>();

    double result_norm = 0;
    for (auto v : result_cpu) result_norm += static_cast<float>(v) * static_cast<float>(v);
    result_norm = std::sqrt(result_norm);

    // Compute actual difference (what we measure as weight_change)
    double diff_norm = 0;
    for (size_t i = 0; i < weight_cpu.size(); ++i) {
        float d = static_cast<float>(result_cpu[i]) - static_cast<float>(weight_cpu[i]);
        diff_norm += d * d;
    }
    diff_norm = std::sqrt(diff_norm);

    // Compute error vs expected
    double error = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        float d = static_cast<float>(result_cpu[i]) - expected[i];
        error += d * d;
    }
    error = std::sqrt(error);

    fmt::print("\nAfter subtract:\n");
    fmt::print("  result_norm = {:.6f}\n", result_norm);
    fmt::print("  result[0:5] = [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]\n",
               float(result_cpu[0]), float(result_cpu[1]), float(result_cpu[2]),
               float(result_cpu[3]), float(result_cpu[4]));
    fmt::print("  expected[0:5] = [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]\n",
               expected[0], expected[1], expected[2], expected[3], expected[4]);

    fmt::print("\nKey metrics:\n");
    fmt::print("  update_norm (expected change) = {:.6f}\n", update_norm);
    fmt::print("  diff_norm (actual change)     = {:.6f}\n", diff_norm);
    fmt::print("  error vs expected             = {:.6f}\n", error);
    fmt::print("  ratio (actual/expected)       = {:.4f}\n", diff_norm / update_norm);

    bool pass = std::abs(diff_norm - update_norm) < 0.01 * update_norm;
    fmt::print("\n=== {} ===\n", pass ? "PASS" : "FAIL");

    if (!pass) {
        fmt::print("ERROR: Actual change ({:.6f}) != expected change ({:.6f})\n",
                   diff_norm, update_norm);
        fmt::print("This indicates ROW_MAJOR subtract is corrupting values!\n");
    }

    return pass ? 0 : 1;
}
