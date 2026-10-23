#include <riscv_vector.h>
#include <cstddef>
#include <cstdint>

namespace {
// Widening reduction: vle8 + vwredsum(i8->i16) + vmv.x = 3 vector ops per chunk,
// versus the previous vle8 + vwcvt + vwcvt + vredsum + vmv.x = 5.
//
// e8m2 rather than e8m1: on VLEN=128 that is 32 lanes per iteration instead of
// 16, so the chunk count halves. The loop is charged per chunk rather than per
// byte -- a board measurement had the runtime tracking the chunk count, not the
// byte count -- which is where the remaining gain was. Measured on the kernel in
// isolation: 1.25x on a 27-long row up to 2.09x on a 1152-long one.
//
// The reduction still lands in i16, which stays safe while one chunk cannot sum
// past 32767:
//
//     VLMAX(e8m2) * 128 <= 32767   =>   VLMAX <= 255   =>   VLEN <= 1020
//
// At VLEN=128 a chunk sums to at most 32 * 128 = 4096, an eighth of i16, and no
// shipping core is near VLEN=1020. The scalar `sum` accumulating across chunks
// is int32, so row length is not a constraint either. (e8m4 measures faster
// still, 3.44x, but its per-chunk cost grows, it pulls the VLEN bound down to
// 510, and m8 showed no further gain -- e8m2 is the stopping point.)
int32_t sumInt8(const int8_t* src, int size) {
    const size_t vlmax = __riscv_vsetvlmax_e16m1();
    const vint16m1_t zero = __riscv_vmv_v_x_i16m1(0, vlmax);
    int32_t sum = 0;
    int offset = 0;
    while (offset < size) {
        const size_t vl = __riscv_vsetvl_e8m2(static_cast<size_t>(size - offset));
        const vint8m2_t value8 = __riscv_vle8_v_i8m2(src + offset, vl);
        const vint16m1_t reduced = __riscv_vwredsum_vs_i8m2_i16m1(value8, zero, vl);
        sum += static_cast<int32_t>(__riscv_vmv_x_s_i16m1_i16(reduced));
        offset += static_cast<int>(vl);
    }
    return sum;
}
} // namespace

void MNNConvInt8ComputeWeightKernelSum_RVV(int* kernelSum, int32_t* bias, const int8_t* weight, int kernelNum,
                                           int kernelSize, const float* scale, const float* weightBias,
                                           bool compensateSseOffset) {
    for (int i = 0; i < kernelNum; ++i) {
        const int32_t sum = sumInt8(weight + static_cast<size_t>(i) * kernelSize, kernelSize);
        kernelSum[i] = static_cast<int>(sum + kernelSize * (weightBias[i] / scale[i]));
        if (compensateSseOffset) {
            bias[i] -= 128 * sum;
        }
    }
}
