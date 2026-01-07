#include <ttnn/device.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <tt-metalium/distributed.hpp>
#include <fmt/core.h>

using MeshDevice = tt::tt_metal::distributed::MeshDevice;
using DispatchCoreConfig = tt::tt_metal::DispatchCoreConfig;
using DispatchCoreType = tt::tt_metal::DispatchCoreType;

int main() {
    auto device = MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE, 128*1024*1024, 1,
                                                DispatchCoreConfig{DispatchCoreType::ETH});
    
    auto a = ttnn::zeros(ttnn::Shape({32, 256, 512}), ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device);
    auto b = ttnn::zeros(ttnn::Shape({1, 1, 512}), ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device);
    
    fmt::print("Input dtype: {}\n", static_cast<int>(a.dtype()));
    
    // Test: add with output_dtype
    auto c = ttnn::add(a, b, ttnn::DataType::BFLOAT8_B);
    
    fmt::print("Output dtype: {}\n", static_cast<int>(c.dtype()));
    fmt::print("Expected BFP8: {}\n", static_cast<int>(ttnn::DataType::BFLOAT8_B));
    
    tt::tt_metal::distributed::Finish(device->mesh_command_queue());
    ttnn::close_device(*device);
    return 0;
}
