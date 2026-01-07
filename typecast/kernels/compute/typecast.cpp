// Compute Kernel: Typecast BFP8 → BF16
// The unpacker reads BFP8, packer writes BF16
// No explicit SFPU typecast needed - hardware handles format conversion
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

// num_tiles passed as compile-time define
#ifndef NUM_TILES
#define NUM_TILES 1
#endif

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t num_tiles = NUM_TILES;

    unary_op_init_common(cb_in, cb_out);
    copy_tile_to_dst_init_short(cb_in);

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_in, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();
        copy_tile(cb_in, 0, 0);  // CB_in[0] → DST[0]
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);    // DST[0] → CB_out (format conversion happens here)
        tile_regs_release();

        cb_pop_front(cb_in, 1);
        cb_push_back(cb_out, 1);
    }
}
}
