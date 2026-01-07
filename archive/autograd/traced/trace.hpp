// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Trace Utilities
// Helper class for capturing and replaying TTNN traces.

#pragma once

#include <ttnn/operations/trace.hpp>
#include <tt-metalium/distributed.hpp>
#include <optional>
#include <functional>

namespace traced {

using MeshDevice = tt::tt_metal::distributed::MeshDevice;
using MeshTraceId = tt::tt_metal::distributed::MeshTraceId;

// TraceContext: Manages trace capture and replay lifecycle
// First call to run() captures the trace, subsequent calls replay it.
class TraceContext {
    MeshDevice* device;
    std::optional<MeshTraceId> trace_id;
    bool captured = false;

public:
    explicit TraceContext(MeshDevice* dev) : device(dev) {}

    // Non-copyable
    TraceContext(const TraceContext&) = delete;
    TraceContext& operator=(const TraceContext&) = delete;

    // Movable
    TraceContext(TraceContext&& other) noexcept
        : device(other.device), trace_id(other.trace_id), captured(other.captured) {
        other.trace_id = std::nullopt;
        other.captured = false;
    }

    TraceContext& operator=(TraceContext&& other) noexcept {
        if (this != &other) {
            release();
            device = other.device;
            trace_id = other.trace_id;
            captured = other.captured;
            other.trace_id = std::nullopt;
            other.captured = false;
        }
        return *this;
    }

    ~TraceContext() {
        release();
    }

    // Run function: capture on first call, replay on subsequent calls
    template<typename Fn>
    void run(Fn&& fn) {
        if (!captured) {
            trace_id = ttnn::operations::trace::begin_trace_capture(device, std::nullopt);
            fn();
            ttnn::operations::trace::end_trace_capture(device, *trace_id, std::nullopt);
            captured = true;
        } else {
            ttnn::operations::trace::execute_trace(device, *trace_id, std::nullopt, false);
        }
    }

    // Check if trace has been captured
    bool is_captured() const { return captured; }

    // Get trace ID (for debugging)
    std::optional<MeshTraceId> get_trace_id() const { return trace_id; }

    // Manually release trace
    void release() {
        if (trace_id) {
            ttnn::operations::trace::release_trace(device, *trace_id);
            trace_id = std::nullopt;
            captured = false;
        }
    }

    // Reset and re-capture on next run
    void reset() {
        release();
    }
};

}  // namespace traced
