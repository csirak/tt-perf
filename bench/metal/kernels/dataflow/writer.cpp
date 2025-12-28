// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    uint32_t tile_size = get_tile_size(cb_out);

    const InterleavedAddrGenFast<true> dst = {
        .bank_base_address = dst_addr,
        .page_size = tile_size,
        .data_format = DataFormat::Float16_b,
    };

    cb_wait_front(cb_out, 1);
    noc_async_write_tile(0, dst, get_read_ptr(cb_out));
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
}
