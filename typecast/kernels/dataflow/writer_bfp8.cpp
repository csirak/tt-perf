// Writer Kernel: Circular Buffer → DRAM (BFP8 format)
#include <cstdint>

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    uint32_t tile_size = get_tile_size(cb_out);

    InterleavedAddrGenFast<true> dst_gen = {
        .bank_base_address = dst_addr,
        .page_size = tile_size,
        .data_format = DataFormat::Bfp8_b
    };

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_out, 1);
        uint32_t l1_addr = get_read_ptr(cb_out);
        noc_async_write_tile(i, dst_gen, l1_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
