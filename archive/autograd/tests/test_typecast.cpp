// Test: Typecast BFP8 <-> BF16 round-trip
// Verifies that ttnn::typecast works correctly for BFP8 training optimization

#include "common.hpp"
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/copy/typecast/typecast.hpp>

using namespace ttnn;
using MeshDevice = tt::tt_metal::distributed::MeshDevice;

int main() {
    fmt::print("=== Test: Typecast BFP8 <-> BF16 ===\n");

    test::DeviceGuard dg;
    auto& device = dg.get();

    // Test 1: BF16 -> BFP8 -> BF16 round-trip with ones
    {
        fmt::print("\nTest 1: Round-trip with ones [32, 64]...\n");
        auto bf16 = ttnn::ones(ttnn::Shape({32, 64}), DataType::BFLOAT16, TILE_LAYOUT, device);

        // Typecast to BFP8
        auto bfp8 = ttnn::typecast(bf16, DataType::BFLOAT8_B);

        // Typecast back to BF16
        auto bf16_back = ttnn::typecast(bfp8, DataType::BFLOAT16);

        // Verify values preserved (within BFP8 precision)
        Finish(device.mesh_command_queue());
        test::assert_all_close(bf16_back, 1.0f, 0.1f);
        fmt::print("  PASS: ones round-trip preserves values\n");
    }

    // Test 2: Larger tensor typical of GPT-2 activations
    {
        fmt::print("\nTest 2: GPT-2 activation shape [32, 1, 256, 512]...\n");
        auto bf16 = ttnn::ones(ttnn::Shape({32, 1, 256, 512}), DataType::BFLOAT16, TILE_LAYOUT, device);

        auto bfp8 = ttnn::typecast(bf16, DataType::BFLOAT8_B);
        auto bf16_back = ttnn::typecast(bfp8, DataType::BFLOAT16);

        Finish(device.mesh_command_queue());
        test::assert_all_close(bf16_back, 1.0f, 0.1f);
        fmt::print("  PASS: large tensor round-trip works\n");
    }

    // Test 3: Verify dtypes
    {
        fmt::print("\nTest 3: Verify dtypes...\n");
        auto bf16 = ttnn::ones(ttnn::Shape({32, 32}), DataType::BFLOAT16, TILE_LAYOUT, device);
        auto bfp8 = ttnn::typecast(bf16, DataType::BFLOAT8_B);

        assert(bf16.dtype() == DataType::BFLOAT16);
        assert(bfp8.dtype() == DataType::BFLOAT8_B);
        fmt::print("  PASS: dtypes correct after typecast\n");
    }

    fmt::print("\n=== All typecast tests passed ===\n");
    return 0;
}
