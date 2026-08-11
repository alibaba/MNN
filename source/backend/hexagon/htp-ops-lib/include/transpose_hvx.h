#ifndef TRANSPOSE_HVX_H
#define TRANSPOSE_HVX_H

#include <hexagon_types.h>
#include <hexagon_protos.h>

#define HVX_TRANSPOSE_STAGE(stride, rt_bytes)                              \
    do {                                                                   \
        for (int base = 0; base < 64; base += (stride) * 2) {             \
            for (int inner = 0; inner < (stride); ++inner) {              \
                HVX_VectorPair pair =                                      \
                    Q6_W_vdeal_VVR(v[base + inner + (stride)],             \
                                   v[base + inner], (rt_bytes));           \
                v[base + inner] = Q6_V_lo_W(pair);                         \
                v[base + inner + (stride)] = Q6_V_hi_W(pair);             \
            }                                                              \
        }                                                                  \
    } while (0)

static inline void hvx_transpose_64x64(HVX_Vector* v) {
    // hvx.pdf p296-p302: for power-of-two Rt, vshuff/vdeal perform the same
    // 2x2 block transpose on byte groups. Using Rt = 2/4/8/16/32/64 bytes
    // gives the 6-stage butterfly needed for a 64x64 fp16 transpose.
    HVX_TRANSPOSE_STAGE(1, 2);
    HVX_TRANSPOSE_STAGE(2, 4);
    HVX_TRANSPOSE_STAGE(4, 8);
    HVX_TRANSPOSE_STAGE(8, 16);
    HVX_TRANSPOSE_STAGE(16, 32);
    HVX_TRANSPOSE_STAGE(32, 64);
}

static inline void hvx_transpose_64x64_first(HVX_Vector* v, int outputCount) {
    if (outputCount >= 32) {
        hvx_transpose_64x64(v);
        return;
    }
    if (outputCount <= 0) {
        return;
    }
    HVX_TRANSPOSE_STAGE(1, 2);
    HVX_TRANSPOSE_STAGE(2, 4);
    HVX_TRANSPOSE_STAGE(4, 8);
    HVX_TRANSPOSE_STAGE(8, 16);
    HVX_TRANSPOSE_STAGE(16, 32);
    for (int inner = 0; inner < outputCount; ++inner) {
        HVX_VectorPair pair = Q6_W_vdeal_VVR(v[inner + 32], v[inner], 64);
        v[inner] = Q6_V_lo_W(pair);
    }
}

#undef HVX_TRANSPOSE_STAGE

static inline void hvx_transpose_32x32(HVX_Vector* v) { (void)v; }

#endif
