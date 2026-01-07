// Compute Kernel: Typecast BF16 → BFP8
// The unpacker reads BF16, packer writes BFP8
// No explicit SFPU typecast needed - hardware handles format conversion
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;   // BF16 input
    constexpr uint32_t cb_out = tt::CBIndex::c_16; // BFP8 output

    unary_op_init_common(cb_in, cb_out);
    copy_tile_to_dst_init_short(cb_in);

    cb_wait_front(cb_in, 1);
    cb_reserve_back(cb_out, 1);

    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);  // CB_in[0] → DST[0]
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, cb_out);    // DST[0] → CB_out (packer converts to BFP8)
    tile_regs_release();

    cb_pop_front(cb_in, 1);
    cb_push_back(cb_out, 1);
}
}
