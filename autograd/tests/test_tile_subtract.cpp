// Test: TILE_LAYOUT subtract behavior
// Verifies that ttnn::subtract works correctly on TILE_LAYOUT tensors

#include "common.hpp"
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/rand/rand.hpp>

using namespace test;
using Tensor = ttnn::Tensor;

int main() {
    DeviceGuard guard;
    auto& device = *guard.device;

    // Shape matching embedding: [128, 64]
    ttnn::Shape shape({128, 64});

    fmt::print("=== Test: TILE_LAYOUT subtract behavior ===\n\n");

    // Create TILE_LAYOUT tensors directly
    auto a = ttnn::rand(shape, device, ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT,
                        ttnn::types::DRAM_MEMORY_CONFIG, -1.0f, 1.0f, 42);
    auto b = ttnn::rand(shape, device, ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT,
                        ttnn::types::DRAM_MEMORY_CONFIG, -0.01f, 0.01f, 43);

    // Verify layouts
    bool a_tile = a.layout() == ttnn::TILE_LAYOUT;
    bool b_tile = b.layout() == ttnn::TILE_LAYOUT;
    fmt::print("Layouts: a={}, b={}\n", a_tile ? "TILE" : "RM", b_tile ? "TILE" : "RM");
    fmt::print("Shapes: a={}, b={}\n", a.logical_shape(), b.logical_shape());

    // Get values
    auto a_cpu = a.cpu().to_vector<bfloat16>();
    auto b_cpu = b.cpu().to_vector<bfloat16>();

    double a_norm = 0, b_norm = 0;
    for (auto v : a_cpu) a_norm += static_cast<float>(v) * static_cast<float>(v);
    for (auto v : b_cpu) b_norm += static_cast<float>(v) * static_cast<float>(v);
    a_norm = std::sqrt(a_norm);
    b_norm = std::sqrt(b_norm);

    fmt::print("a_norm = {:.6f}\n", a_norm);
    fmt::print("b_norm = {:.6f}\n", b_norm);

    // Do the subtract
    auto result = ttnn::subtract(a, b);
    auto result_cpu = result.cpu().to_vector<bfloat16>();

    // Compute actual difference from a
    double diff_norm = 0;
    for (size_t i = 0; i < a_cpu.size(); ++i) {
        float d = static_cast<float>(result_cpu[i]) - static_cast<float>(a_cpu[i]);
        diff_norm += d * d;
    }
    diff_norm = std::sqrt(diff_norm);

    fmt::print("\nSubtract result:\n");
    fmt::print("  diff_norm (actual change) = {:.6f}\n", diff_norm);
    fmt::print("  b_norm (expected change)  = {:.6f}\n", b_norm);
    fmt::print("  ratio = {:.4f}\n", diff_norm / b_norm);

    fmt::print("\nSample values:\n");
    fmt::print("  a[0] = {:.6f}\n", float(a_cpu[0]));
    fmt::print("  b[0] = {:.6f}\n", float(b_cpu[0]));
    fmt::print("  result[0] = {:.6f}\n", float(result_cpu[0]));
    fmt::print("  expected[0] = {:.6f}\n", float(a_cpu[0]) - float(b_cpu[0]));

    bool pass = std::abs(diff_norm - b_norm) < 0.1 * b_norm;
    fmt::print("\n=== {} ===\n", pass ? "PASS" : "FAIL");

    if (!pass) {
        fmt::print("ERROR: diff_norm ({:.6f}) != b_norm ({:.6f})\n", diff_norm, b_norm);
    }

    return pass ? 0 : 1;
}
