#include <riscv_vector.h>

void MNNTranspose16Bit(int16_t* dstO, const int16_t* srcO, int16_t* dim) {
    int w = dim[0];
    int h = dim[1];
    int srcStride = dim[2];
    int dstStride = dim[3];
    ptrdiff_t srcStrideByte = srcStride * sizeof(int16_t);

    for (int i = 0; i < h; ++i) {
        const int16_t* srcPtr = srcO + i;
        int16_t* dstPtr = dstO + i * dstStride;

        int j = 0;
        while (j < w) {
            size_t vl = __riscv_vsetvl_e16m8(w - j);
            vint16m8_t data = __riscv_vlse16_v_i16m8(srcPtr, srcStrideByte, vl);
            __riscv_vse16_v_i16m8(dstPtr, data, vl);
            srcPtr += vl * srcStride; 
            dstPtr += vl;
            j += vl;
        }
    }
}


