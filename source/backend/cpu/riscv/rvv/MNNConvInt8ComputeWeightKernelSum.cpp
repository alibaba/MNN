#include <riscv_vector.h>
#include <cstddef>
#include <cstdint>

static int32_t sumInt8(const int8_t* src, int size) {
    int32_t sum = 0;
    int offset = 0;
    while (offset < size) {
        const size_t vl = __riscv_vsetvl_e8m1(static_cast<size_t>(size - offset));
        const vint8m1_t value8 = __riscv_vle8_v_i8m1(src + offset, vl);
        const vint16m2_t value16 = __riscv_vwcvt_x_x_v_i16m2(value8, vl);
        const vint32m4_t value32 = __riscv_vwcvt_x_x_v_i32m4(value16, vl);
        const vint32m1_t zero = __riscv_vmv_s_x_i32m1(0, vl);
        const vint32m1_t reduced = __riscv_vredsum_vs_i32m4_i32m1(value32, zero, vl);
        sum += __riscv_vmv_x_s_i32m1_i32(reduced);
        offset += static_cast<int>(vl);
    }
    return sum;
}

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
