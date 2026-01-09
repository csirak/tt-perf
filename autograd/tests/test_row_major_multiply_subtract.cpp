// Test: ROW_MAJOR multiply then subtract
// This matches what Adam does: wd_term = w * scalar, then w - wd_term

#include "common.hpp"
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/data_movement/untilize/untilize.hpp>
#include <ttnn/operations/rand/rand.hpp>

using namespace test;
using Tensor = ttnn::Tensor;

int main() {
    DeviceGuard guard;
    auto& device = *guard.device;

    // Shape matching embedding: [vocab=128, dim=64]
    ttnn::Shape shape({128, 64});

    fmt::print("=== Test: ROW_MAJOR multiply then subtract ===\n\n");

    // Create ROW_MAJOR weight (like embedding)
    auto weight_tiled = ttnn::rand(shape, device, ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT,
                                    ttnn::types::DRAM_MEMORY_CONFIG, -1.0f, 1.0f, 42);
    auto weight = ttnn::untilize(weight_tiled);  // Convert to ROW_MAJOR

    // This is what Adam does: wd_term = w * lr * wd
    float scalar = 0.001f * 0.01f;  // lr * wd = 0.00001
    auto wd_term = ttnn::multiply(weight, scalar);

    // Check layouts
    bool w_is_rm = weight.layout() == ttnn::ROW_MAJOR_LAYOUT;
    bool wd_is_rm = wd_term.layout() == ttnn::ROW_MAJOR_LAYOUT;
    fmt::print("Layouts: weight={}, wd_term={}\n", w_is_rm ? "RM" : "TILE", wd_is_rm ? "RM" : "TILE");

    // Get values
    auto weight_cpu = weight.cpu().to_vector<bfloat16>();
    auto wd_cpu = wd_term.cpu().to_vector<bfloat16>();

    double weight_norm = 0, wd_norm = 0;
    for (auto v : weight_cpu) weight_norm += static_cast<float>(v) * static_cast<float>(v);
    for (auto v : wd_cpu) wd_norm += static_cast<float>(v) * static_cast<float>(v);
    weight_norm = std::sqrt(weight_norm);
    wd_norm = std::sqrt(wd_norm);

    fmt::print("weight_norm = {:.6f}\n", weight_norm);
    fmt::print("wd_term_norm = {:.9f} (expected: {:.9f})\n", wd_norm, weight_norm * scalar);

    // Now do the subtract
    auto result = ttnn::subtract(weight, wd_term);
    auto result_cpu = result.cpu().to_vector<bfloat16>();

    // Compute difference
    double diff_norm = 0;
    for (size_t i = 0; i < weight_cpu.size(); ++i) {
        float d = static_cast<float>(result_cpu[i]) - static_cast<float>(weight_cpu[i]);
        diff_norm += d * d;
    }
    diff_norm = std::sqrt(diff_norm);

    fmt::print("\nSubtract result:\n");
    fmt::print("  diff_norm = {:.9f} (expected ~wd_term_norm = {:.9f})\n", diff_norm, wd_norm);
    fmt::print("  weight[0] = {:.6f}\n", float(weight_cpu[0]));
    fmt::print("  wd_term[0] = {:.9f}\n", float(wd_cpu[0]));
    fmt::print("  result[0] = {:.6f}\n", float(result_cpu[0]));
    fmt::print("  expected[0] = {:.6f}\n", float(weight_cpu[0]) - float(wd_cpu[0]));

    bool pass = std::abs(diff_norm - wd_norm) < 0.1 * wd_norm;  // Allow 10% tolerance
    fmt::print("\n=== {} ===\n", pass ? "PASS" : "FAIL");

    if (!pass) {
        fmt::print("ERROR: diff_norm ({:.9f}) != wd_norm ({:.9f})\n", diff_norm, wd_norm);
        fmt::print("The subtract of wd_term from weight is not working correctly!\n");
    }

    return pass ? 0 : 1;
}
