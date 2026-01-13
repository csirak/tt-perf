// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// TTNN FFN forward test (matmul + bias + GELU + matmul + bias).
// Uses static_autograd ops so TENSORDIFF_OPLOG_DIR produces TDF1 op logs.

#include "value.hpp"
#include "ops.hpp"

#include <ttnn/operations/rand/rand.hpp>
#include <ttnn/distributed/api.hpp>
#include <ttnn/device.hpp>

#include <cstdlib>
#include <iostream>
#include <memory>

using static_autograd::Graph;
using static_autograd::Tensor;
using static_autograd::Value;
using static_autograd::MeshDevice;

static std::shared_ptr<MeshDevice> create_device() {
    bool use_worker = std::getenv("USE_WORKER_DISPATCH") != nullptr;
    return MeshDevice::create_unit_mesh(
        0,
        DEFAULT_L1_SMALL_SIZE,
        0,
        1,
        use_worker ? tt::tt_metal::DispatchCoreConfig{}
                   : tt::tt_metal::DispatchCoreConfig{tt::tt_metal::DispatchCoreType::ETH}
    );
}

int main() {
    auto device = create_device();

    const uint32_t B = 1;
    const uint32_t S = 32;
    const uint32_t D = 64;
    const uint32_t F = 256; // ffn_mult=4

    const float init_range = 0.5f;
    uint32_t seed = 1;

    auto rand_tensor = [&](const ttnn::Shape& shape) {
        return ttnn::rand(shape, *device, ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT,
                          ttnn::types::DRAM_MEMORY_CONFIG, -init_range, init_range, seed++);
    };

    Tensor x = rand_tensor(ttnn::Shape({B, S, D}));
    Tensor w1 = rand_tensor(ttnn::Shape({D, F}));
    Tensor b1 = rand_tensor(ttnn::Shape({1, 1, F}));
    Tensor w2 = rand_tensor(ttnn::Shape({F, D}));
    Tensor b2 = rand_tensor(ttnn::Shape({1, 1, D}));

    Tensor d_x = static_autograd::make_zeros(ttnn::Shape({B, S, D}), *device);
    Tensor d_w1 = static_autograd::make_zeros(ttnn::Shape({D, F}), *device);
    Tensor d_b1 = static_autograd::make_zeros(ttnn::Shape({1, 1, F}), *device);
    Tensor d_w2 = static_autograd::make_zeros(ttnn::Shape({F, D}), *device);
    Tensor d_b2 = static_autograd::make_zeros(ttnn::Shape({1, 1, D}), *device);

    Tensor out1 = static_autograd::make_zeros(ttnn::Shape({B, S, F}), *device);
    Tensor d_out1 = static_autograd::make_zeros(ttnn::Shape({B, S, F}), *device);
    Tensor out2 = static_autograd::make_zeros(ttnn::Shape({B, S, F}), *device);
    Tensor d_out2 = static_autograd::make_zeros(ttnn::Shape({B, S, F}), *device);
    Tensor out3 = static_autograd::make_zeros(ttnn::Shape({B, S, F}), *device);
    Tensor d_out3 = static_autograd::make_zeros(ttnn::Shape({B, S, F}), *device);
    Tensor out4 = static_autograd::make_zeros(ttnn::Shape({B, S, D}), *device);
    Tensor d_out4 = static_autograd::make_zeros(ttnn::Shape({B, S, D}), *device);
    Tensor out5 = static_autograd::make_zeros(ttnn::Shape({B, S, D}), *device);
    Tensor d_out5 = static_autograd::make_zeros(ttnn::Shape({B, S, D}), *device);

    Graph g;
    Value* vx = g.leaf(&x, &d_x, false);
    Value* vw1 = g.leaf(&w1, &d_w1, false);
    Value* vb1 = g.leaf(&b1, &d_b1, false);
    Value* vw2 = g.leaf(&w2, &d_w2, false);
    Value* vb2 = g.leaf(&b2, &d_b2, false);

    auto* h1 = static_autograd::mm(g, vx, vw1, &out1, &d_out1);
    auto* h2 = static_autograd::add(g, h1, vb1, &out2, &d_out2);
    auto* h3 = static_autograd::gelu(g, h2, &out3, &d_out3);
    auto* h4 = static_autograd::mm(g, h3, vw2, &out4, &d_out4);
    auto* y = static_autograd::add(g, h4, vb2, &out5, &d_out5);
    (void)y;

    tt::tt_metal::distributed::Synchronize(device.get(), std::nullopt);

    std::cout << "ffn forward done." << "\n";

    tt::tt_metal::distributed::Finish(device->mesh_command_queue());
    ttnn::close_device(*device);
    return 0;
}
