// Reader Kernel: DRAM → Circular Buffer (BF16 format)
#include <cstdint>

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    uint32_t tile_size = get_tile_size(cb_in);

    InterleavedAddrGenFast<true> src_gen = {
        .bank_base_address = src_addr,
        .page_size = tile_size,
        .data_format = DataFormat::Float16_b
    };

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_reserve_back(cb_in, 1);
        uint32_t l1_addr = get_write_ptr(cb_in);
        noc_async_read_tile(i, src_gen, l1_addr);
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    }
}
