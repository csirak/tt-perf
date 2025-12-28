// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb0 = tt::CBIndex::c_0;
    constexpr uint32_t cb1 = tt::CBIndex::c_1;
    uint32_t tile_size = get_tile_size(cb0);

    const InterleavedAddrGenFast<true> in0 = {
        .bank_base_address = src0_addr,
        .page_size = tile_size,
        .data_format = DataFormat::Float16_b,
    };
    const InterleavedAddrGenFast<true> in1 = {
        .bank_base_address = src1_addr,
        .page_size = tile_size,
        .data_format = DataFormat::Float16_b,
    };

    cb_reserve_back(cb0, 1);
    noc_async_read_tile(0, in0, get_write_ptr(cb0));
    noc_async_read_barrier();
    cb_push_back(cb0, 1);

    cb_reserve_back(cb1, 1);
    noc_async_read_tile(0, in1, get_write_ptr(cb1));
    noc_async_read_barrier();
    cb_push_back(cb1, 1);
}
