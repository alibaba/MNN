#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

// Byte-wise copy for RVV: one vle8.v/vse8.v pair per iteration at LMUL=m8, so a
// VLEN=256 board moves up to 256 bytes per iteration. The padding rows in a
// depthwise input tensor are only src_width * pack * bytes long and can be a few
// dozen bytes on the low-resolution stages, which is where a dedicated short
// kernel has the most headroom.
void MNNMemcpyBytes_RVV(void* dst, const void* src, size_t size) {
    uint8_t* d = static_cast<uint8_t*>(dst);
    const uint8_t* s = static_cast<const uint8_t*>(src);
    for (size_t vl; size > 0; size -= vl, s += vl, d += vl) {
        vl = __riscv_vsetvl_e8m8(size);
        vuint8m8_t v = __riscv_vle8_v_u8m8(s, vl);
        __riscv_vse8_v_u8m8(d, v, vl);
    }
}
