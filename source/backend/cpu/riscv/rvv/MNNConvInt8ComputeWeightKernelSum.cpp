#include <riscv_vector.h>
#include <cstddef>
#include <cstdint>

namespace {
// Scalar fallback. Integer addition is associative and commutative, so the
// result is bit-identical to the vector path.
int32_t sumInt8Scalar(const int8_t* src, int size) {
    int32_t sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += static_cast<int32_t>(src[i]);
    }
    return sum;
}

// Widening reduction: vle8 + vwredsum(i8->i16) + vmv.x = 3 vector ops per chunk,
// versus the previous vle8 + vwcvt + vwcvt + vredsum + vmv.x = 5.
// 32 lanes of i8 sum to at most 4096 in magnitude, which fits in i16, so the
// one-step widening reduction cannot overflow.
int32_t sumInt8Vector(const int8_t* src, int size) {
    const size_t vlmax = __riscv_vsetvlmax_e16m1();
    const vint16m1_t zero = __riscv_vmv_v_x_i16m1(0, vlmax);
    int32_t sum = 0;
    int offset = 0;
    while (offset < size) {
        const size_t vl = __riscv_vsetvl_e8m1(static_cast<size_t>(size - offset));
        const vint8m1_t value8 = __riscv_vle8_v_i8m1(src + offset, vl);
        const vint16m1_t reduced = __riscv_vwredsum_vs_i8m1_i16m1(value8, zero, vl);
        sum += static_cast<int32_t>(__riscv_vmv_x_s_i16m1_i16(reduced));
        offset += static_cast<int>(vl);
    }
    return sum;
}

inline bool useVector(int size) {
    // Below one full vector the vector prologue/latency dominates.
    return size >= static_cast<int>(__riscv_vsetvlmax_e8m1());
}
} // namespace

void MNNConvInt8ComputeWeightKernelSum_RVV(int* kernelSum, int32_t* bias, const int8_t* weight, int kernelNum,
                                           int kernelSize, const float* scale, const float* weightBias,
                                           bool compensateSseOffset) {
    const bool vectorPath = useVector(kernelSize);
    for (int i = 0; i < kernelNum; ++i) {
        const int8_t* row = weight + static_cast<size_t>(i) * kernelSize;
        const int32_t sum = vectorPath ? sumInt8Vector(row, kernelSize) : sumInt8Scalar(row, kernelSize);
        kernelSum[i] = static_cast<int>(sum + kernelSize * (weightBias[i] / scale[i]));
        if (compensateSseOffset) {
            bias[i] -= 128 * sum;
        }
    }
}
