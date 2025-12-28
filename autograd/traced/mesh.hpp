// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Mesh Partitioning for Data-Parallel Model Execution (Traced Version)
// Automatically partitions models across available cores on a device.
// Uses pre-allocated buffers for trace compatibility.

#pragma once

#include "nn.hpp"
#include "ops.hpp"
#include <tt-metalium/core_coord.hpp>
#include <vector>
#include <cmath>
#include <memory>

namespace traced {

using ttnn::CoreGrid;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::CoreCoord;

// Configuration for a single partition
struct PartitionConfig {
    CoreGrid grid;           // Grid size for matmul ops (e.g., 2x4)
    CoreRangeSet core_range; // Physical cores for elementwise ops
    uint32_t partition_id;

    PartitionConfig(uint32_t id, CoreGrid g, CoreRangeSet cr)
        : grid(g), core_range(std::move(cr)), partition_id(id) {}
};

// Mesh configuration that computes optimal partitioning
struct MeshConfig {
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t total_cores;
    uint32_t n_partitions;
    uint32_t cores_per_partition;
    std::vector<PartitionConfig> partitions;

    static MeshConfig from_device(MeshDevice& device, uint32_t n_partitions = 0) {
        auto grid_size = device.compute_with_storage_grid_size();
        uint32_t gx = grid_size.x;
        uint32_t gy = grid_size.y;
        uint32_t total = gx * gy;

        if (n_partitions == 0) {
            n_partitions = total / 8;
        }

        return MeshConfig::create(gx, gy, n_partitions);
    }

    static MeshConfig create(uint32_t grid_x, uint32_t grid_y, uint32_t n_partitions) {
        MeshConfig config;
        config.grid_x = grid_x;
        config.grid_y = grid_y;
        config.total_cores = grid_x * grid_y;
        config.n_partitions = n_partitions;
        config.cores_per_partition = config.total_cores / n_partitions;
        config.partitions = compute_partitions(grid_x, grid_y, n_partitions);
        return config;
    }

private:
    static std::vector<PartitionConfig> compute_partitions(
        uint32_t grid_x, uint32_t grid_y, uint32_t n_partitions
    ) {
        std::vector<PartitionConfig> partitions;
        uint32_t cores_per_part = (grid_x * grid_y) / n_partitions;

        uint32_t part_cols = 2;
        uint32_t part_rows = 4;

        if (cores_per_part < 8) {
            part_cols = 2;
            part_rows = cores_per_part / 2;
            if (part_rows < 1) part_rows = 1;
        }

        uint32_t partitions_per_row = grid_x / part_cols;

        for (uint32_t i = 0; i < n_partitions; i++) {
            uint32_t row_idx = i / partitions_per_row;
            uint32_t col_idx = i % partitions_per_row;

            uint32_t start_x = col_idx * part_cols;
            uint32_t start_y = row_idx * part_rows;

            CoreGrid grid(part_cols, part_rows);
            CoreRangeSet core_range;

            if (start_y + part_rows > grid_y) {
                uint32_t remaining_rows = grid_y - start_y;
                uint32_t adjusted_cols = cores_per_part / remaining_rows;
                if (adjusted_cols > grid_x - start_x) adjusted_cols = grid_x - start_x;
                if (adjusted_cols < 1) adjusted_cols = 1;
                if (remaining_rows < 1) remaining_rows = 1;
                grid = CoreGrid(adjusted_cols, remaining_rows);
                core_range = CoreRangeSet({
                    CoreRange({start_x, start_y}, {start_x + adjusted_cols - 1, grid_y - 1})
                });
            } else {
                core_range = CoreRangeSet({
                    CoreRange({start_x, start_y}, {start_x + part_cols - 1, start_y + part_rows - 1})
                });
            }

            partitions.emplace_back(i, grid, std::move(core_range));
        }

        return partitions;
    }
};

// Traced Partitioned Linear - runs on specific cores with pre-allocated buffers
struct TracedPartitionedLinear {
    Tensor weight;
    Tensor bias;
    Tensor d_weight;
    Tensor d_bias;
    CoreGrid grid;
    CoreRangeSet core_range;
    uint32_t in_features;
    uint32_t out_features;

    TracedPartitionedLinear(uint32_t in_f, uint32_t out_f, float init,
                            MeshDevice& device, const PartitionConfig& config)
        : weight(make_full(ttnn::Shape({out_f, in_f}), init, device)),
          bias(make_zeros(ttnn::Shape({1, out_f}), device)),
          d_weight(make_zeros(ttnn::Shape({out_f, in_f}), device)),
          d_bias(make_zeros(ttnn::Shape({1, out_f}), device)),
          grid(config.grid),
          core_range(config.core_range),
          in_features(in_f),
          out_features(out_f) {}

    // Forward: out = x @ weight.T + bias (with CoreGrid constraint)
    void forward(const Tensor& x, Tensor& out) {
        auto mm = ttnn::matmul(x, weight, false, true,
                               std::nullopt, std::nullopt, std::nullopt,
                               std::nullopt, std::nullopt, grid);
        out = ttnn::add(mm, bias);
    }

    // Backward with d_input
    void backward(const Tensor& x, const Tensor& d_out, Tensor& d_input) {
        d_weight = ttnn::matmul(d_out, x, true, false,
                                std::nullopt, std::nullopt, std::nullopt,
                                std::nullopt, std::nullopt, grid);
        d_bias = ttnn::sum(d_out, 0, true);
        d_input = ttnn::matmul(d_out, weight, false, false,
                               std::nullopt, std::nullopt, std::nullopt,
                               std::nullopt, std::nullopt, grid);
    }

    // Backward without d_input (for first layer)
    void backward_no_input(const Tensor& x, const Tensor& d_out) {
        d_weight = ttnn::matmul(d_out, x, true, false,
                                std::nullopt, std::nullopt, std::nullopt,
                                std::nullopt, std::nullopt, grid);
        d_bias = ttnn::sum(d_out, 0, true);
    }

    void sgd_step(float lr) {
        weight = ttnn::subtract(weight, ttnn::multiply(d_weight, lr));
        bias = ttnn::subtract(bias, ttnn::multiply(d_bias, lr));
    }
};

// Traced Partitioned MLP - 2-layer MLP on specific cores
struct TracedPartitionedMLP {
    TracedPartitionedLinear layer1;
    TracedPartitionedLinear layer2;
    CoreRangeSet core_range;

    // Forward buffers
    Tensor h;           // Output of layer1
    Tensor h_relu;      // ReLU(h)
    Tensor relu_mask;   // Mask for backward
    Tensor out;         // Final output

    // Backward buffers
    Tensor d_h_relu;    // Gradient at h_relu
    Tensor d_h;         // Gradient at h

    uint32_t batch;
    uint32_t dim;

    TracedPartitionedMLP(uint32_t b, uint32_t in_dim, uint32_t hidden_dim, uint32_t out_dim,
                         MeshDevice& device, const PartitionConfig& config)
        : layer1(in_dim, hidden_dim, 0.01f, device, config),
          layer2(hidden_dim, out_dim, 0.01f, device, config),
          core_range(config.core_range),
          h(make_zeros(ttnn::Shape({b, hidden_dim}), device)),
          h_relu(make_zeros(ttnn::Shape({b, hidden_dim}), device)),
          relu_mask(make_zeros(ttnn::Shape({b, hidden_dim}), device)),
          out(make_zeros(ttnn::Shape({b, out_dim}), device)),
          d_h_relu(make_zeros(ttnn::Shape({b, hidden_dim}), device)),
          d_h(make_zeros(ttnn::Shape({b, hidden_dim}), device)),
          batch(b),
          dim(hidden_dim) {}

    void forward(const Tensor& x) {
        layer1.forward(x, h);
        // ReLU with core_range constraint
        h_relu = ttnn::relu(h, std::nullopt, std::nullopt, core_range);
        relu_mask = ttnn::gtz(h, std::nullopt, std::nullopt, core_range);
        layer2.forward(h_relu, out);
    }

    void backward(const Tensor& x, const Tensor& d_out) {
        layer2.backward(h_relu, d_out, d_h_relu);
        // ReLU backward
        d_h = ttnn::multiply(d_h_relu, relu_mask);
        layer1.backward_no_input(x, d_h);
    }

    void sgd_step(float lr) {
        layer1.sgd_step(lr);
        layer2.sgd_step(lr);
    }

    const Tensor& output() const { return out; }
};

// TracedMeshModel: Wrapper for N partitioned models
template<typename ModelType>
struct TracedMeshModel {
    std::vector<std::unique_ptr<ModelType>> models;
    MeshConfig config;
    MeshDevice* device;

    // Factory for TracedPartitionedMLP
    static TracedMeshModel<TracedPartitionedMLP> create_mlp(
        MeshDevice& device,
        uint32_t n_partitions,
        uint32_t batch,
        uint32_t in_dim,
        uint32_t hidden_dim,
        uint32_t out_dim
    ) {
        TracedMeshModel<TracedPartitionedMLP> mesh_model;
        mesh_model.config = MeshConfig::from_device(device, n_partitions);
        mesh_model.device = &device;

        for (uint32_t i = 0; i < mesh_model.config.n_partitions; i++) {
            mesh_model.models.push_back(std::make_unique<TracedPartitionedMLP>(
                batch, in_dim, hidden_dim, out_dim, device, mesh_model.config.partitions[i]
            ));
        }

        return mesh_model;
    }

    void forward(const std::vector<Tensor>& inputs) {
        for (size_t i = 0; i < models.size(); i++) {
            models[i]->forward(inputs[i]);
        }
    }

    void backward(const std::vector<Tensor>& inputs, const std::vector<Tensor>& d_outputs) {
        for (size_t i = 0; i < models.size(); i++) {
            models[i]->backward(inputs[i], d_outputs[i]);
        }
    }

    void sgd_step(float lr) {
        for (auto& model : models) {
            model->sgd_step(lr);
        }
    }

    std::vector<Tensor> outputs() const {
        std::vector<Tensor> outs;
        outs.reserve(models.size());
        for (const auto& model : models) {
            outs.push_back(model->output());
        }
        return outs;
    }

    size_t size() const { return models.size(); }

    void print_config() const {
        fmt::print("# TracedMeshModel Configuration\n");
        fmt::print("# Device grid: {}x{} = {} cores\n",
                   config.grid_x, config.grid_y, config.total_cores);
        fmt::print("# Partitions: {}, cores/partition: {}\n",
                   config.n_partitions, config.cores_per_partition);
        for (const auto& p : config.partitions) {
            fmt::print("#   Partition {}: grid={}x{}\n",
                       p.partition_id, p.grid.x, p.grid.y);
        }
    }
};

}  // namespace traced
