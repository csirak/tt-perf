// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// TensorGroup: Batched tensor operations for reduced dispatch overhead
//
// Register tensors into groups, then apply operations to entire groups at once.
// This minimizes per-op dispatch overhead by batching similar operations.

#pragma once

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/operations/copy/typecast/typecast.hpp>
#include <ttnn/distributed/api.hpp>

#include <vector>
#include <functional>
#include <cstdlib>

#ifdef TRACY_ENABLE
#include <tracy/Tracy.hpp>
#endif

namespace static_mxp {

using Tensor = tt::tt_metal::Tensor;
using MeshDevice = tt::tt_metal::distributed::MeshDevice;

// TensorGroup: A collection of tensor pointers for batched operations
//
// Usage:
//   TensorGroup bfp8_activations;
//   bfp8_activations.add(&layer1.out);
//   bfp8_activations.add(&layer2.out);
//
//   TensorGroup bf16_cache;
//   bf16_cache.add(&layer1.out_bf16);
//   bf16_cache.add(&layer2.out_bf16);
//
//   // Batch typecast all at once
//   bfp8_activations.typecast_to(bf16_cache, ttnn::DataType::BFLOAT16);
//
struct TensorGroup {
    std::vector<Tensor*> tensors;

    // Add a tensor to the group
    void add(Tensor* t) {
        tensors.push_back(t);
    }

    // Add multiple tensors
    void add(std::initializer_list<Tensor*> list) {
        for (auto* t : list) {
            tensors.push_back(t);
        }
    }

    // Clear all tensors from group
    void clear() {
        tensors.clear();
    }

    // Number of tensors in group
    size_t size() const {
        return tensors.size();
    }

    // Access tensor by index
    Tensor* operator[](size_t i) {
        return tensors[i];
    }

    const Tensor* operator[](size_t i) const {
        return tensors[i];
    }

    // =========================================================================
    // Batched Operations
    // =========================================================================

    // Typecast all tensors in this group to destination group
    // src[i] -> dst[i] with given dtype
    // Queues all typecasts without synchronization (caller should sync after)
    // Uses output tensor form to preserve tensor properties (shape, layout, etc.)
    void typecast_to(TensorGroup& dst, ttnn::DataType dtype) {
#ifdef TRACY_ENABLE
        ZoneScopedN("TensorGroup_typecast_to");
#endif
        if (dst.size() != size()) {
            throw std::runtime_error("TensorGroup size mismatch in typecast_to");
        }
        for (size_t i = 0; i < tensors.size(); ++i) {
            if (tensors[i] && dst.tensors[i]) {
                // Use 4-arg form to write to existing tensor (preserves properties)
                ttnn::typecast(*tensors[i], dtype, std::nullopt, *dst.tensors[i]);
            }
        }
    }

    // Typecast all tensors in-place (creates new tensors, replaces pointers)
    void typecast_inplace(ttnn::DataType dtype) {
#ifdef TRACY_ENABLE
        ZoneScopedN("TensorGroup_typecast_inplace");
#endif
        for (auto* t : tensors) {
            if (t) {
                *t = ttnn::typecast(*t, dtype);
            }
        }
    }

    // Apply a unary function to all tensors, storing results in dst group
    // fn signature: Tensor fn(const Tensor&)
    template<typename Fn>
    void apply(TensorGroup& dst, Fn&& fn) {
#ifdef TRACY_ENABLE
        ZoneScopedN("TensorGroup_apply");
#endif
        if (dst.size() != size()) {
            throw std::runtime_error("TensorGroup size mismatch in apply");
        }
        for (size_t i = 0; i < tensors.size(); ++i) {
            if (tensors[i] && dst.tensors[i]) {
                *dst.tensors[i] = fn(*tensors[i]);
            }
        }
    }

    // Apply a unary function in-place
    template<typename Fn>
    void apply_inplace(Fn&& fn) {
#ifdef TRACY_ENABLE
        ZoneScopedN("TensorGroup_apply_inplace");
#endif
        for (auto* t : tensors) {
            if (t) {
                *t = fn(*t);
            }
        }
    }

    // Apply a binary function: this[i] op other[i] -> dst[i]
    template<typename Fn>
    void apply_binary(TensorGroup& other, TensorGroup& dst, Fn&& fn) {
#ifdef TRACY_ENABLE
        ZoneScopedN("TensorGroup_apply_binary");
#endif
        if (other.size() != size() || dst.size() != size()) {
            throw std::runtime_error("TensorGroup size mismatch in apply_binary");
        }
        for (size_t i = 0; i < tensors.size(); ++i) {
            if (tensors[i] && other.tensors[i] && dst.tensors[i]) {
                *dst.tensors[i] = fn(*tensors[i], *other.tensors[i]);
            }
        }
    }
};

// NamedTensorGroup: TensorGroup with string keys for easier debugging
struct NamedTensorGroup {
    std::vector<std::pair<std::string, Tensor*>> tensors;

    void add(const std::string& name, Tensor* t) {
        tensors.emplace_back(name, t);
    }

    size_t size() const {
        return tensors.size();
    }

    Tensor* get(const std::string& name) {
        for (auto& [n, t] : tensors) {
            if (n == name) return t;
        }
        return nullptr;
    }

    // Typecast all to destination group (must have same names/order)
    void typecast_to(NamedTensorGroup& dst, ttnn::DataType dtype) {
#ifdef TRACY_ENABLE
        ZoneScopedN("NamedTensorGroup_typecast_to");
#endif
        if (dst.size() != size()) {
            throw std::runtime_error("NamedTensorGroup size mismatch");
        }
        static const bool use_out_buffer = std::getenv("MXP_TYPECAST_ALLOC") == nullptr;
        for (size_t i = 0; i < tensors.size(); ++i) {
            if (tensors[i].second && dst.tensors[i].second) {
                if (use_out_buffer) {
                    ttnn::typecast(*tensors[i].second, dtype, std::nullopt, *dst.tensors[i].second);
                } else {
                    *dst.tensors[i].second = ttnn::typecast(*tensors[i].second, dtype);
                }
            }
        }
    }
};

// TypecastCache: Pre-allocated BF16 buffers for BFP8 tensors
// Manages source (BFP8) and destination (BF16) tensor groups together
struct TypecastCache {
    TensorGroup sources;      // BFP8 tensors to convert
    TensorGroup destinations; // Pre-allocated BF16 buffers

    // Register a BFP8 tensor and its corresponding BF16 cache buffer
    void register_pair(Tensor* bfp8_src, Tensor* bf16_dst) {
        sources.add(bfp8_src);
        destinations.add(bf16_dst);
    }

    // Batch typecast all registered BFP8 tensors to their BF16 caches
    void convert_all() {
#ifdef TRACY_ENABLE
        ZoneScopedN("TypecastCache_convert_all");
#endif
        sources.typecast_to(destinations, ttnn::DataType::BFLOAT16);
    }

    // Get the BF16 cache for a given source tensor
    Tensor* get_bf16(Tensor* bfp8_src) {
        for (size_t i = 0; i < sources.size(); ++i) {
            if (sources[i] == bfp8_src) {
                return destinations[i];
            }
        }
        return nullptr;
    }

    size_t size() const {
        return sources.size();
    }

    void clear() {
        sources.clear();
        destinations.clear();
    }
};

// WeightTypecastCache: Batched BF16→BFP8 conversion for weights after SGD
// Converts master weights (BF16) to compute weights (BFP8) for next forward
struct WeightTypecastCache {
    TensorGroup bf16_sources;  // BF16 master weights
    TensorGroup bfp8_dests;    // BFP8 weights for forward

    // Register a weight pair: BF16 master → BFP8 compute
    void register_pair(Tensor* bf16_src, Tensor* bfp8_dst) {
        bf16_sources.add(bf16_src);
        bfp8_dests.add(bfp8_dst);
    }

    // Batch typecast all BF16 masters to BFP8 compute weights
    void convert_all() {
#ifdef TRACY_ENABLE
        ZoneScopedN("WeightTypecastCache_convert_all");
#endif
        bf16_sources.typecast_to(bfp8_dests, ttnn::DataType::BFLOAT8_B);
    }

    size_t size() const {
        return bf16_sources.size();
    }

    void clear() {
        bf16_sources.clear();
        bfp8_dests.clear();
    }
};

} // namespace static_mxp
