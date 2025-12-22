#include <riscv_vector.h>

void MNNTranspose32Bit(int32_t* dstO, const int32_t* srcO, int32_t* dim) {
    int w = dim[0];
    int h = dim[1];
    int srcStride = dim[2];
    int dstStride = dim[3];
    ptrdiff_t srcStrideByte = srcStride * sizeof(int32_t);

    for (int i = 0; i < h; ++i) {
        const int32_t* srcPtr = srcO + i;
        int32_t* dstPtr = dstO + i * dstStride;

        int j = 0;
        while (j < w) {
            size_t vl = __riscv_vsetvl_e32m8(w - j);
            vint32m8_t data = __riscv_vlse32_v_i32m8(srcPtr, srcStrideByte, vl);
            __riscv_vse32_v_i32m8(dstPtr, data, vl);
            srcPtr += vl * srcStride; 
            dstPtr += vl;
            j += vl;
        }
    }
}

