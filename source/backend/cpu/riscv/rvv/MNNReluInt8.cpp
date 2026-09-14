#include <riscv_vector.h>
#include <stdint.h>
#include <stddef.h>
#include <sys/types.h>

// Distinct symbol: the generic scalar MNNReluInt8 stays available in
// CommonOptFunction.cpp and is only overridden through the core function table
// once the runtime RVV probe succeeds.
void MNNReluInt8_RVV(int8_t* dst, const int8_t* src, size_t size, ssize_t zeroPoint) {
    if (zeroPoint < INT8_MIN || zeroPoint > INT8_MAX) {
        for (size_t i = 0; i < size; ++i) {
            dst[i] = src[i] < zeroPoint ? static_cast<int8_t>(zeroPoint) : src[i];
        }
        return;
    }

    const int8_t zero = static_cast<int8_t>(zeroPoint);
    size_t offset = 0;
    while (offset < size) {
        const size_t vl = __riscv_vsetvl_e8m8(size - offset);
        vint8m8_t value = __riscv_vle8_v_i8m8(src + offset, vl);
        value = __riscv_vmax_vx_i8m8(value, zero, vl);
        __riscv_vse8_v_i8m8(dst + offset, value, vl);
        offset += vl;
    }
}
