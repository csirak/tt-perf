// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Test Utilities for TTNN Autograd

#pragma once

#include <ttnn/device.hpp>
#include <ttnn/distributed/api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <cassert>
#include <cmath>
#include <memory>

namespace test {

using MeshDevice = tt::tt_metal::distributed::MeshDevice;
using DispatchCoreConfig = tt::tt_metal::DispatchCoreConfig;

// RAII device guard - opens on construct, closes on destruct
struct DeviceGuard {
    std::shared_ptr<MeshDevice> device;

    // Allocate 128MB for trace region to support tracing
    static constexpr size_t TRACE_REGION_SIZE = 128 * 1024 * 1024;

    // Note: 2 CQs don't help for TTNN ops (compute-bound, not dispatch-bound)
    // The 2-CQ speedup is only seen in minimal dispatch-overhead tests
    DeviceGuard(int num_cqs = 1)
        : device(MeshDevice::create_unit_mesh(
            0,                          // device_id
            DEFAULT_L1_SMALL_SIZE,
            TRACE_REGION_SIZE,
            num_cqs,                    // num_command_queues
            DispatchCoreConfig{}
        )) {}

    ~DeviceGuard() {
        Finish(device->mesh_command_queue());
        ttnn::close_device(*device);
    }

    MeshDevice& get() { return *device; }

    // Non-copyable
    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;
};

// Assert all elements are close to expected value
inline void assert_all_close(const tt::tt_metal::Tensor& t, float expected, float tol = 0.1f) {
    auto host = t.cpu().to_vector<bfloat16>();
    for (size_t i = 0; i < host.size(); ++i) {
        float val = static_cast<float>(host[i]);
        assert(std::abs(val - expected) < tol);
    }
}

// Assert tensor has variance (not all elements are the same)
inline void assert_has_variance(const tt::tt_metal::Tensor& t) {
    auto host = t.cpu().to_vector<bfloat16>();
    if (host.size() < 2) return;

    float first = static_cast<float>(host[0]);
    for (size_t i = 1; i < host.size(); ++i) {
        if (std::abs(static_cast<float>(host[i]) - first) > 0.001f) return;
    }
    assert(false && "tensor has no variance - all elements are the same");
}

// Assert two tensors have the same shape
inline void assert_shape_eq(const tt::tt_metal::Tensor& a, const tt::tt_metal::Tensor& b) {
    assert(a.logical_shape() == b.logical_shape());
}

// Assert tensor shape matches expected dimensions
inline void assert_shape(const tt::tt_metal::Tensor& t, uint32_t dim0, uint32_t dim1) {
    assert(t.logical_shape()[0] == dim0);
    assert(t.logical_shape()[1] == dim1);
}

}  // namespace test
