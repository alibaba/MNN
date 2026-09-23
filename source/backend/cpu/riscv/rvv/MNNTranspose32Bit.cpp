#include <riscv_vector.h>

void MNNTranspose32Bit_RVV(int32_t* dstO, const int32_t* srcO, int32_t* dim) {
    int w = dim[0];
    int h = dim[1];
    int srcStride = dim[2];
    int dstStride = dim[3];
    const ptrdiff_t srcStrideByte = static_cast<ptrdiff_t>(srcStride) * sizeof(int32_t);

    for (int i = 0; i < h; ++i) {
        const int32_t* srcPtr = srcO + i;
        int32_t* dstPtr = dstO + static_cast<ptrdiff_t>(i) * dstStride;

        int j = 0;
        while (j < w) {
            size_t vl = __riscv_vsetvl_e32m8(w - j);
            vint32m8_t data = __riscv_vlse32_v_i32m8(srcPtr, srcStrideByte, vl);
            __riscv_vse32_v_i32m8(dstPtr, data, vl);
            srcPtr += static_cast<ptrdiff_t>(vl) * srcStride;
            dstPtr += vl;
            j += vl;
        }
    }
}
