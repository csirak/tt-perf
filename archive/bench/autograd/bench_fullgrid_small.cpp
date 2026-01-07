// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Full Grid Benchmark with small batch/dim (for comparison with 7x parallel)
// Config: 7 MLPs using full 56 core grid (no partitioning)

#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <tt-metalium/distributed.hpp>

#include <chrono>
#include <vector>
#include <numeric>
#include <array>

using namespace ttnn;
using namespace tt::tt_metal::distributed;

constexpr int N_WARMUP = 2;
constexpr int N_TIMED = 3;
constexpr int N_PARALLEL = 7;

struct MLPInstance {
    Tensor x, w1, b1, w2, b2, dout;
};

MLPInstance create_mlp(MeshDevice& dev, uint32_t batch, uint32_t dim) {
    return {
        ones(Shape({batch, dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
        ones(Shape({dim, dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
        zeros(Shape({1, dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
        ones(Shape({dim, dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
        zeros(Shape({1, dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
        ones(Shape({batch, dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
    };
}

// Forward + backward using full device grid (no CoreGrid constraint)
void run_mlp_async(const MLPInstance& m) {
    auto mm1 = ttnn::matmul(m.x, m.w1);
    auto h_pre = ttnn::add(mm1, m.b1);
    auto h = ttnn::relu(h_pre);
    auto mm2 = ttnn::matmul(h, m.w2);
    auto out = ttnn::add(mm2, m.b2);

    auto dh = ttnn::matmul(m.dout, m.w2);
    auto relu_mask = ttnn::gtz(h_pre);
    auto dh_pre = ttnn::multiply(dh, relu_mask);
    auto dx = ttnn::matmul(dh_pre, m.w1);
}

int main() {
    auto device = MeshDevice::create_unit_mesh(0);
    MeshCommandQueue& cq = device->mesh_command_queue();

    auto grid_size = device->compute_with_storage_grid_size();
    uint32_t total_cores = grid_size.x * grid_size.y;

    fmt::print("# Full Grid Benchmark (small batch/dim for comparison)\n");
    fmt::print("# Device grid: {}x{} = {} cores\n", grid_size.x, grid_size.y, total_cores);
    fmt::print("# Config: {} MLPs using full {} cores each (no partitioning)\n", N_PARALLEL, total_cores);
    fmt::print("# Warmup: {}, Timed: {}\n", N_WARMUP, N_TIMED);
    fmt::print("#\n");

    // Same size as 7x parallel benchmark
    uint32_t batch = 256;
    uint32_t dim = 512;

    // Create 7 MLPs
    std::array<MLPInstance, N_PARALLEL> mlps;
    for (int i = 0; i < N_PARALLEL; i++) {
        mlps[i] = create_mlp(*device, batch, dim);
    }
    Finish(cq);

    // Warmup
    for (int w = 0; w < N_WARMUP; w++) {
        for (int i = 0; i < N_PARALLEL; i++) {
            run_mlp_async(mlps[i]);
        }
        Finish(cq);
    }

    // Timed runs
    std::vector<double> times_ms(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_PARALLEL; i++) {
            run_mlp_async(mlps[i]);
        }
        Finish(cq);
        auto end = std::chrono::high_resolution_clock::now();
        times_ms[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    double avg_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / N_TIMED;
    double total_samples = N_PARALLEL * batch;
    double throughput = total_samples / avg_ms * 1000.0;
    double throughput_per_core = throughput / total_cores;

    fmt::print("batch,dim,n_parallel,cores_per_mlp,total_cores,avg_ms,throughput_samples_sec,throughput_per_core\n");
    fmt::print("{},{},{},{},{},{:.3f},{:.1f},{:.1f}\n",
               batch, dim, N_PARALLEL, total_cores, total_cores, avg_ms, throughput, throughput_per_core);

    fmt::print("#\n");
    fmt::print("# Individual run times: ");
    for (auto t : times_ms) fmt::print("{:.3f} ", t);
    fmt::print("ms\n");

    device->close();
    return 0;
}
