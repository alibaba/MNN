//
//  ConvSimdGroupShader.hpp
//  MNN
//
//  Created by MNN on b'2024/12/30'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#if MNN_METAL_ENABLED

static const char* gBasicConvPrefix = R"metal(
#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;
typedef enum : int {
    None  = 0,
    ReLU  = 1,
    ReLU6 = 2,
} conv_activation_type;

inline ftype2 activate(ftype2 value, conv_activation_type type) {
    switch (type) {
        case ReLU:
            return max(value, (ftype2)0);
        case ReLU6:
            return clamp(value, (ftype2)0, (ftype2)6);
        default: // None
            return value;
    }
}
inline ftype4 activate(ftype4 value, conv_activation_type type) {
    switch (type) {
        case ReLU:
            return max(value, (ftype4)0);
        case ReLU6:
            return clamp(value, (ftype4)0, (ftype4)6);
        default: // None
            return value;
    }
}
struct conv1x1_constants {
    int inputSize;
    int inputDepthQuad;
    int outputWidth;
    int outputHeight;
    int outputSize;
    int outputDepthQuad;
    int outputChannel;
    int batch;
    int blockCount;
    conv_activation_type activation;
    float scaleCoef;
};

namespace MNN {
    typedef struct uchar4x2 {
    private:
        uchar2 v[4];
    public:
        uchar4x2(uchar2 a) {
            v[0] = a; v[1] = a; v[2] = a; v[3] = a;
        }
        uchar4x2(uchar2 a, uchar2 b, uchar2 c, uchar2 d) {
            v[0] = a; v[1] = b; v[2] = c; v[3] = d;
        }

        inline thread uchar2& operator[] (const int index) {
            return v[index];
        }
        inline device uchar2& operator[] (const int index) device {
            return v[index];
        }
        inline threadgroup uchar2& operator[] (const int index) threadgroup {
            return v[index];
        }

        inline const thread uchar2& operator[] (const int index) const {
            return v[index];
        }
        inline const device uchar2& operator[] (const int index) const device {
            return v[index];
        }
        inline const threadgroup uchar2& operator[] (const int index) const threadgroup {
            return v[index];
        }

        inline explicit operator half4x2() const {
            return half4x2( half2(v[0]), half2(v[1]), half2(v[2]), half2(v[3]) );
        }
        inline explicit operator half4x2() const device {
            return half4x2( half2(v[0]), half2(v[1]), half2(v[2]), half2(v[3]) );
        }
        inline explicit operator half4x2() const threadgroup {
            return half4x2( half2(v[0]), half2(v[1]), half2(v[2]), half2(v[3]) );
        }

        inline explicit operator float4x2() const {
            return float4x2( float2(v[0]), float2(v[1]), float2(v[2]), float2(v[3]) );
        }
        inline explicit operator float4x2() const device {
            return float4x2( float2(v[0]), float2(v[1]), float2(v[2]), float2(v[3]) );
        }
        inline explicit operator float4x2() const threadgroup {
            return float4x2( float2(v[0]), float2(v[1]), float2(v[2]), float2(v[3]) );
        }
    } uchar4x2;

    typedef struct char4x4 {
    private:
        char4 v[4];
    public:
        char4x4(char4 a) {
            v[0] = a; v[1] = a; v[2] = a; v[3] = a;
        }
        char4x4(char4 a, char4 b, char4 c, char4 d) {
            v[0] = a; v[1] = b; v[2] = c; v[3] = d;
        }

        inline thread char4& operator[] (const int index) {
            return v[index];
        }
        inline device char4& operator[] (const int index) device {
            return v[index];
        }
        inline threadgroup char4& operator[] (const int index) threadgroup {
            return v[index];
        }

        inline const thread char4& operator[] (const int index) const {
            return v[index];
        }
        inline const device char4& operator[] (const int index) const device {
            return v[index];
        }
        inline const threadgroup char4& operator[] (const int index) const threadgroup {
            return v[index];
        }

        inline explicit operator half4x4() const {
            return half4x4( half4(v[0]), half4(v[1]), half4(v[2]), half4(v[3]) );
        }
        inline explicit operator half4x4() const device {
            return half4x4( half4(v[0]), half4(v[1]), half4(v[2]), half4(v[3]) );
        }
        inline explicit operator half4x4() const threadgroup {
            return half4x4( half4(v[0]), half4(v[1]), half4(v[2]), half4(v[3]) );
        }

        inline explicit operator float4x4() const {
            return float4x4( float4(v[0]), float4(v[1]), float4(v[2]), float4(v[3]) );
        }
        inline explicit operator float4x4() const device {
            return float4x4( float4(v[0]), float4(v[1]), float4(v[2]), float4(v[3]) );
        }
        inline explicit operator float4x4() const threadgroup {
            return float4x4( float4(v[0]), float4(v[1]), float4(v[2]), float4(v[3]) );
        }
    } char4x4;
}

#if MNN_METAL_FLOAT16_STORAGE
typedef simdgroup_half8x8 simdgroup_FTYPE8x8;
#else
typedef simdgroup_float8x8 simdgroup_FTYPE8x8;
#endif

#if MNN_METAL_FLOAT32_COMPUTER
typedef simdgroup_float8x8 simdgroup_FLOAT8x8;
typedef float    FLOAT;
typedef float2   FLOAT2;
typedef float4   FLOAT4;
typedef float4x4 FLOAT4x4;
#else
typedef simdgroup_half8x8 simdgroup_FLOAT8x8;
typedef half    FLOAT;
typedef half2   FLOAT2;
typedef half4   FLOAT4;
typedef half4x4 FLOAT4x4;
#endif

#define SIMD_GROUP_WIDTH 32
#define CONV_UNROLL (4)
#define CONV_UNROLL_L (8)

#define INIT_SIMDGROUP_MATRIX(a, b, d) \
    simdgroup_FTYPE8x8 sga[a];\
    simdgroup_FTYPE8x8 sgb[b];\
    simdgroup_FLOAT8x8 sgd[d];\
    for (int i = 0; i < d; i++){\
        sgd[i] = make_filled_simdgroup_matrix<FLOAT, 8>(0.f);\
    }

#define SIMDGROUP_MATRIX_FMA(a, b) \
    for(int j=0; j<b; j++) {\
        for(int i=0; i<a; i++) {\
            simdgroup_multiply_accumulate(sgd[j*a+i], sga[i], sgb[j], sgd[j*a+i]);\
        }\
    }

#define SIMDGROUP_MATRIX_STORE(ptr, d) \
    for(int i=0; i<d; i++) {\
        simdgroup_store(sgd[i], ptr + 64*i, 8);\
    }
)metal";

static const char* gConv1x1WqSgMatrix = R"metal(
// W_QUANT_2/3 sg_matrix kernels are not implemented; the dispatcher routes
// W2/W3 prefill to outer-dequant + fp GEMM instead. Guard the entire string
// so the kernels don't silently miscompile under W_QUANT_2/3 macros.
#if !defined(W_QUANT_2) && !defined(W_QUANT_3)

kernel void conv1x1_gemm_8x16_wquant_sg(const device ftype2 *in            [[buffer(0)]],
                            device ftype4 *out                 [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                        #ifdef W_QUANT_4
                            const device uchar2 *wt      [[buffer(3)]],
                        #elif defined(W_QUANT_8)
                            const device char4 *wt      [[buffer(3)]],
                        #endif
                            const device ftype4 *biasTerms     [[buffer(4)]],
                            const device ftype4 *dequantScale  [[buffer(5)]],
                            uint3 gid                          [[threadgroup_position_in_grid]],
                            uint                  tiitg[[thread_index_in_threadgroup]],
                            uint                  sgitg[[simdgroup_index_in_threadgroup]]) {
    /*
     // Read:
     ftype 0~63   ---> input: [M8, K8]
     ftype 64~191 ---> input: [K8, N16]
     // Write:
     ftype 0~127 ---> input: [N2, M8, N8]
     */

    threadgroup FLOAT sdata[256] = {0.f};
    INIT_SIMDGROUP_MATRIX(1, 2, 2);

    int rx = gid.x;// M/8
    int uz = gid.y;// N/16

    int kl = tiitg / 16; // 0~1
    int rcl = tiitg % 16; // 0~15

    /** input:
     threadgroup: [M8, K8]
     each thread: K2
     layout: [K/4, M, K4] -> [K/8, K2, M/8, M8, K2, K2]
     index : [0, rcl/8, rx, rcl%8, kl, K2]
     offset: ((0*2+rcl/8) * M + rx * 8 + rcl%8) * 2 + kl
     */
    /** weight:
     threadgroup: [K8, N16]
     each thread: K4
     layout: [N/4, K/4, N4, K4] -> [N/16, N4, K/8, K2, N4, K4]
     index : [uz, rcl/4, 0, kl, rcl%4, K4]
     offset: (((uz * 4 + rcl/4) * K/4 + 0*2+kl) * 4 + rcl%4)
     */
    /** output:
     threadgroup: [M8, N16] -> [N2, M8, N2, N4]
     sdata: [(rcl / 4) / 2, (rcl%4) * 2 + kl, (rcl / 4) % 2]
     each thread: N4
     layout: [N/4, M, N4] -> [N/16, N4, M/8, M4, M2, N4]
     index : [uz, rcl/4, rx, rcl%4, kl, N4]
     offset: ((uz * 4 + rcl/4) * M + (rx * 8 + (rcl%4) * 2 + kl))
     */

    // boundary limit
    int idx_n4 = (4 * uz + rcl / 4) < cst.outputDepthQuad ? (4 * uz + rcl / 4) : (cst.outputDepthQuad - 1);
    int idx_m  = (8 * rx + rcl%8) < cst.inputSize * cst.batch ? (8 * rx + rcl%8) : (cst.inputSize * cst.batch - 1);

    auto xy_wt = wt +  ((idx_n4 * cst.inputDepthQuad + 0*2+kl) * 4 + rcl % 4);// [N/4, K/4, N4, K4]
    auto xy_in0  = in + ((0*2+rcl/8) * cst.inputSize * cst.batch + idx_m) * 2 + kl;// [K/4, M, K2, K2]
    auto xy_out = out + (4 * uz + rcl / 4) * cst.outputSize * cst.batch + (rx * 8 + (rcl%4) * 2 + kl);// [N/4, M, N4]

    int quadsPerBlock = (cst.inputDepthQuad + cst.blockCount - 1) / cst.blockCount;
    for (int bi=0; bi<cst.blockCount; ++bi) {
        // [N/4, cst.blockCount, 2/*scale_bias*/, N4]
        FLOAT4 scale = FLOAT4(dequantScale[2 * (idx_n4 * cst.blockCount + bi) + 0]) / (FLOAT)cst.scaleCoef;
        FLOAT4 dequant_bias = FLOAT4(dequantScale[2 * (idx_n4 * cst.blockCount + bi) + 1]) / (FLOAT)cst.scaleCoef;
        int zmin = bi * quadsPerBlock;
        int zmax = min(zmin + quadsPerBlock, cst.inputDepthQuad);

        for (int z = zmin; z < zmax; z += 2) {
            // [M8, K2, K2, K2]
            ((threadgroup ftype2*)sdata)[((rcl%8) * 2 + (rcl/8)) * 2 + kl] = (*xy_in0);
            xy_in0 += 4 * cst.inputSize * cst.batch;

            #ifdef W_QUANT_4
                uchar2 w_int40 = xy_wt[4 * z]; // [N/4, K/4, N4, K4]
                FLOAT4 w40 = FLOAT4((float)(w_int40[0] >> 4) - 8, (float)(w_int40[0] & 15) - 8, (float)(w_int40[1] >> 4) - 8, (float)(w_int40[1] & 15) - 8);
            #elif defined(W_QUANT_8)
                char4 w_int40 = xy_wt[4 * z]; // [N/4, K/4, N4, K4]
                FLOAT4 w40 = FLOAT4((float)w_int40[0], (float)w_int40[1], (float)w_int40[2], (float)w_int40[3]);
            #endif

            FLOAT4 res = w40 * scale[rcl % 4] + dequant_bias[rcl % 4];
            // [K8, N4, N4]
            ((threadgroup ftype*)sdata)[64 + (kl * 4 + 0) * 16 + rcl] = ftype(res[0]);
            ((threadgroup ftype*)sdata)[64 + (kl * 4 + 1) * 16 + rcl] = ftype(res[1]);
            ((threadgroup ftype*)sdata)[64 + (kl * 4 + 2) * 16 + rcl] = ftype(res[2]);
            ((threadgroup ftype*)sdata)[64 + (kl * 4 + 3) * 16 + rcl] = ftype(res[3]);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_load(sga[0], (const threadgroup ftype*)sdata, 8);
            simdgroup_load(sgb[0], ((const threadgroup ftype*)sdata) + 64, 16);
            simdgroup_load(sgb[1], ((const threadgroup ftype*)sdata) + 72, 16);

            SIMDGROUP_MATRIX_FMA(1, 2);

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    SIMDGROUP_MATRIX_STORE((threadgroup FLOAT*)sdata, 2);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if((rx * 8 + (rcl%4) * 2 + kl) < cst.inputSize * cst.batch) {
        if((4 * uz + rcl / 4) < cst.outputDepthQuad) {
            xy_out[0] =  activate(ftype4(((threadgroup FLOAT4*)sdata)[(((rcl / 4) / 2) * 8 + ((rcl%4) * 2 + kl)) * 2 + (rcl / 4) % 2] + FLOAT4(biasTerms[4 * uz + rcl / 4])), cst.activation);
        }
    }
}

kernel void conv1x1_gemm_16x16_wquant_sg(const device ftype4 *in            [[buffer(0)]],
                            device ftype4 *out                 [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                        #ifdef W_QUANT_4
                            const device uchar2 *wt      [[buffer(3)]],
                        #elif defined(W_QUANT_8)
                            const device char4 *wt      [[buffer(3)]],
                        #endif
                            const device ftype4 *biasTerms     [[buffer(4)]],
                            const device ftype4 *dequantScale  [[buffer(5)]],
                            uint3 gid                          [[threadgroup_position_in_grid]],
                            uint                  tiitg[[thread_index_in_threadgroup]],
                            uint                  sgitg[[simdgroup_index_in_threadgroup]]) {
    /*
     // Read:
     ftype 0~127   ---> input: [M16, K8]
     ftype 128~255 ---> input: [K8, N16]
     // Write:
     ftype 0~255 ---> input: [N2, M2, M8, N8]
     */
    threadgroup FLOAT4 sdata[256] = {0.f};

    INIT_SIMDGROUP_MATRIX(2, 2, 4);

    int rx = gid.x;// M/16
    int uz = gid.y;// N/16

    int kl = tiitg / 16;
    int rcl = tiitg % 16;
//    int kl = tiitg % 2;
//    int rcl = tiitg / 2;

    // boundary limit
    int idx_n4 = (4 * uz + rcl / 4) < cst.outputDepthQuad ? (4 * uz + rcl / 4) : (cst.outputDepthQuad - 1);
    int idx_m  = (16 * rx + rcl) < cst.inputSize * cst.batch ? (16 * rx + rcl) : (cst.inputSize * cst.batch - 1);

    auto xy_wt = wt +  (idx_n4 * cst.inputDepthQuad + 0) * 4 + rcl % 4;// [N/4, K/4, N4, K4]
    auto xy_in0  = in + idx_m + cst.inputSize * cst.batch * kl;// [K/4, M, K4]
    auto xy_out = out + (4 * uz + 2 * kl) * cst.outputSize * cst.batch + idx_m;// [N/4, M, N4]

    int quadsPerBlock = (cst.inputDepthQuad + cst.blockCount - 1) / cst.blockCount;
    for (int bi=0; bi<cst.blockCount; ++bi) {
        // [N/4, cst.blockCount, 2/*scale_bias*/, N4]
        FLOAT4 scale = FLOAT4(dequantScale[2 * (idx_n4 * cst.blockCount + bi) + 0]) / (FLOAT)cst.scaleCoef;
        FLOAT4 dequant_bias = FLOAT4(dequantScale[2 * (idx_n4 * cst.blockCount + bi) + 1]) / (FLOAT)cst.scaleCoef;
        int zmin = bi * quadsPerBlock;
        int zmax = min(zmin + quadsPerBlock, cst.inputDepthQuad);

        for (int z = zmin + kl; z < zmax; z += 8) {
            #pragma unroll(4)
            for(int i = 0; i < 4; i++) {
                ((threadgroup ftype4*)sdata)[64 * i + 2* rcl + kl] = *xy_in0;
                xy_in0 += 2 * cst.inputSize * cst.batch;
            }

            #pragma unroll(4)
            for(int i = 0; i < 4; i++) {

            #ifdef W_QUANT_4
                uchar2 w_int40 = xy_wt[4 * (z + 2*i)]; // [N/4, K/4, N4, K4]
                FLOAT4 w40 = FLOAT4((float)(w_int40[0] >> 4) - 8, (float)(w_int40[0] & 15) - 8, (float)(w_int40[1] >> 4) - 8, (float)(w_int40[1] & 15) - 8);
            #elif defined(W_QUANT_8)
                char4 w_int40 = xy_wt[4 * (z + 2*i)]; // [N/4, K/4, N4, K4]
                FLOAT4 w40 = FLOAT4((float)w_int40[0], (float)w_int40[1], (float)w_int40[2], (float)w_int40[3]);
            #endif

                FLOAT4 res = w40 * scale[rcl % 4] + dequant_bias[rcl % 4];
                ((threadgroup ftype*)sdata)[256 * i + 128 + (kl * 4 + 0) * 16 + rcl] = ftype(res[0]);
                ((threadgroup ftype*)sdata)[256 * i + 128 + (kl * 4 + 1) * 16 + rcl] = ftype(res[1]);
                ((threadgroup ftype*)sdata)[256 * i + 128 + (kl * 4 + 2) * 16 + rcl] = ftype(res[2]);
                ((threadgroup ftype*)sdata)[256 * i + 128 + (kl * 4 + 3) * 16 + rcl] = ftype(res[3]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            #pragma unroll(4)
            for(int i = 0; i < 4; i++) {
                simdgroup_load(sga[0], (const threadgroup ftype*)sdata + 256*i, 8);
                simdgroup_load(sga[1], ((const threadgroup ftype*)sdata) + 64 + 256*i, 8);
                simdgroup_barrier(mem_flags::mem_none);

                simdgroup_load(sgb[0], ((const threadgroup ftype*)sdata) + 128 + 256*i, 16);
                simdgroup_load(sgb[1], ((const threadgroup ftype*)sdata) + 136 + 256*i, 16);

                SIMDGROUP_MATRIX_FMA(2, 2);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    SIMDGROUP_MATRIX_STORE((threadgroup FLOAT*)sdata, 4);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if((16 * rx + rcl) < cst.inputSize * cst.batch) {
        if((4 * uz + 2 * kl) < cst.outputDepthQuad) {
            xy_out[0] =  activate(ftype4(sdata[(kl * 16 + rcl) * 2 + 0] + FLOAT4(biasTerms[4 * uz + 2 * kl + 0])), cst.activation);
        }
        if((4 * uz + 2 * kl + 1) < cst.outputDepthQuad) {
            xy_out[cst.outputSize * cst.batch] = activate(ftype4(sdata[(kl * 16 + rcl) * 2 + 1] + FLOAT4(biasTerms[4 * uz + 2 * kl + 1])), cst.activation);
        }
    }
}

kernel void conv1x1_gemm_32x16_wquant_sg(const device ftype4 *in            [[buffer(0)]],
                            device ftype4 *out                 [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                        #ifdef W_QUANT_4
                            const device uchar2 *wt      [[buffer(3)]],
                        #elif defined(W_QUANT_8)
                            const device char4 *wt      [[buffer(3)]],
                        #endif
                            const device ftype4 *biasTerms     [[buffer(4)]],
                            const device ftype4 *dequantScale  [[buffer(5)]],
                            uint3 gid                          [[threadgroup_position_in_grid]],
                            uint                  tiitg[[thread_index_in_threadgroup]],
                            uint                  sgitg[[simdgroup_index_in_threadgroup]]) {
    /*
     // Read:
     ftype 0~255   ---> input: [M32, K8]
     ftype 256~383 ---> input: [K8, N16]
     // Write:
     ftype 0~511 ---> input: [N2, M4, M8, N8]
     */
    threadgroup FLOAT4 sdata[128] = {0.f};

    INIT_SIMDGROUP_MATRIX(4, 2, 8);

    int rx = gid.x;// M/32
    int uz = gid.y;// N/16

    int kl = tiitg % 2;
    int rcl = tiitg / 2;

    const int size_m = cst.inputSize * cst.batch;

    // boundary limit
    int idx_n4 = (4 * uz + rcl / 4) < cst.outputDepthQuad ? (4 * uz + rcl / 4) : (cst.outputDepthQuad - 1);
    int idx_m0  = (16 * rx + rcl) <  size_m ? (16 * rx + rcl) : (size_m - 1);
    int idx_m1  = (16 * rx + rcl) + size_m / 2 < size_m ? (16 * rx + rcl) + size_m / 2: (size_m - 1);

    auto xy_wt = wt +  (idx_n4 * cst.inputDepthQuad + 0) * 4 + rcl % 4;// [N/4, K/4, N4, K4]
    auto xy_in0  = in + idx_m0 + cst.inputSize * cst.batch * kl;// [K/4, M2, M/2, K4]
    auto xy_in1  = in + idx_m1 + cst.inputSize * cst.batch * kl;// [K/4, M2, M/2, K4]

    auto xy_out0 = out + (4 * uz + 2 * kl) * cst.outputSize * cst.batch + idx_m0;// [N/4, M, N4]
    auto xy_out1 = out + (4 * uz + 2 * kl) * cst.outputSize * cst.batch + idx_m1;// [N/4, M, N4]

    int quadsPerBlock = (cst.inputDepthQuad + cst.blockCount - 1) / cst.blockCount;
    for (int bi=0; bi<cst.blockCount; ++bi) {
        // [N/4, cst.blockCount, 2/*scale_bias*/, N4]
        FLOAT4 scale = FLOAT4(dequantScale[2 * (idx_n4 * cst.blockCount + bi) + 0]) / (FLOAT)cst.scaleCoef;
        FLOAT4 dequant_bias = FLOAT4(dequantScale[2 * (idx_n4 * cst.blockCount + bi) + 1]) / (FLOAT)cst.scaleCoef;
        int zmin = bi * quadsPerBlock;
        int zmax = min(zmin + quadsPerBlock, cst.inputDepthQuad);

        for (int z = zmin + kl; z < zmax; z += 2) {
            ((threadgroup ftype4*)sdata)[2* rcl + kl] = *xy_in0;
            ((threadgroup ftype4*)sdata)[32 + 2* rcl + kl] = *xy_in1;

            #ifdef W_QUANT_4
                uchar2 w_int4 = xy_wt[4*z]; // [N/4, K/4, N4, K4]
                FLOAT4 w4 = FLOAT4((float)(w_int4[0] >> 4) - 8, (float)(w_int4[0] & 15) - 8, (float)(w_int4[1] >> 4) - 8, (float)(w_int4[1] & 15) - 8);
            #elif defined(W_QUANT_8)
                char4 w_int4 = xy_wt[4*z]; // [N/4, K/4, N4, K4]
                FLOAT4 w4 = FLOAT4((float)w_int4[0], (float)w_int4[1], (float)w_int4[2], (float)w_int4[3]);
            #endif

            FLOAT4 res = w4 * scale[rcl % 4] + dequant_bias[rcl % 4];
            //            sdata[32 + 2* rcl + kl] = res;
            ((threadgroup ftype*)sdata)[256 + (kl * 4 + 0) * 16 + rcl] = ftype(res[0]);
            ((threadgroup ftype*)sdata)[256 + (kl * 4 + 1) * 16 + rcl] = ftype(res[1]);
            ((threadgroup ftype*)sdata)[256 + (kl * 4 + 2) * 16 + rcl] = ftype(res[2]);
            ((threadgroup ftype*)sdata)[256 + (kl * 4 + 3) * 16 + rcl] = ftype(res[3]);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_load(sga[0], (const threadgroup ftype*)sdata, 8);
            simdgroup_load(sga[1], ((const threadgroup ftype*)sdata) + 64, 8);
            simdgroup_load(sga[2], ((const threadgroup ftype*)sdata) + 128, 8);
            simdgroup_load(sga[3], ((const threadgroup ftype*)sdata) + 192, 8);

            simdgroup_load(sgb[0], ((const threadgroup ftype*)sdata) + 256, 16);
            simdgroup_load(sgb[1], ((const threadgroup ftype*)sdata) + 264, 16);

            SIMDGROUP_MATRIX_FMA(4, 2);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            xy_in0 += 2 * cst.inputSize * cst.batch;
            xy_in1 += 2 * cst.inputSize * cst.batch;

        }
    }

    SIMDGROUP_MATRIX_STORE((threadgroup FLOAT*)sdata, 8);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if((16 * rx + rcl) < size_m) {
        if((4 * uz + 2 * kl) < cst.outputDepthQuad) {
            xy_out0[0] =  activate(ftype4(sdata[(kl * 32 + rcl) * 2 + 0] + FLOAT4(biasTerms[4 * uz + 2 * kl + 0])), cst.activation);
        }
        if((4 * uz + 2 * kl + 1) < cst.outputDepthQuad) {
            xy_out0[cst.outputSize * cst.batch] = activate(ftype4(sdata[(kl * 32 + rcl) * 2 + 1] + FLOAT4(biasTerms[4 * uz + 2 * kl + 1])), cst.activation);
        }
    }
    if((16 * rx + rcl) + size_m / 2 < size_m) {
        if((4 * uz + 2 * kl) < cst.outputDepthQuad) {
            xy_out1[0] =  activate(ftype4(sdata[(kl * 32 + 16 + rcl) * 2 + 0] + FLOAT4(biasTerms[4 * uz + 2 * kl + 0])), cst.activation);
        }
        if((4 * uz + 2 * kl + 1) < cst.outputDepthQuad) {
            xy_out1[cst.outputSize * cst.batch] = activate(ftype4(sdata[(kl * 32 + 16 + rcl) * 2 + 1] + FLOAT4(biasTerms[4 * uz + 2 * kl + 1])), cst.activation);
        }
    }
}

kernel void conv1x1_gemm_16x32_wquant_sg(const device ftype4 *in            [[buffer(0)]],
                            device ftype4 *out                 [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                        #ifdef W_QUANT_4
                            const device uchar2 *wt      [[buffer(3)]],
                        #elif defined(W_QUANT_8)
                            const device char4 *wt      [[buffer(3)]],
                        #endif
                            const device ftype4 *biasTerms     [[buffer(4)]],
                            const device ftype4 *dequantScale  [[buffer(5)]],
                            uint3 gid                          [[threadgroup_position_in_grid]],
                            uint                  tiitg[[thread_index_in_threadgroup]],
                            uint                  sgitg[[simdgroup_index_in_threadgroup]]) {
    /*
     // Read:
     ftype 0~127   ---> input: [M16, K8]
     ftype 128~383 ---> input: [K8, N32]
     // Write:
     ftype 0~511 ---> input: [N2, N2, M2, M8, N8]
     */
    threadgroup FLOAT4 sdata[128] = {0.f};

    INIT_SIMDGROUP_MATRIX(2, 4, 8);

    int rx = gid.x;// M/16
    int uz = gid.y;// N/32

    int kl = tiitg % 2;
    int rcl = tiitg / 2;

    // boundary limit
    int idx_n40 = (4 * uz + rcl / 4) < cst.outputDepthQuad ? (4 * uz + rcl / 4) : (cst.outputDepthQuad - 1);
    int idx_n41 = (4 * uz + rcl / 4) + cst.outputDepthQuad / 2 < cst.outputDepthQuad ? (4 * uz + rcl / 4) + cst.outputDepthQuad / 2 : (cst.outputDepthQuad - 1);

    int idx_m  = (16 * rx + rcl) < cst.inputSize * cst.batch ? (16 * rx + rcl) : (cst.inputSize * cst.batch - 1);

    auto xy_wt0 = wt +  (idx_n40 * cst.inputDepthQuad + 0) * 4 + (rcl % 4);// [N2, N/8, K/4, N4, K4]
    auto xy_wt1 = wt +  (idx_n41 * cst.inputDepthQuad + 0) * 4 + (rcl % 4);// [N2, N/8, K/4, N4, K4]

    auto xy_in0  = in + idx_m + cst.inputSize * cst.batch * kl;// [K/4, M, K4]
    auto xy_out = out + (4 * uz + 2 * kl) * cst.outputSize * cst.batch + idx_m;// [N2, N/8, M, N4]

    int quadsPerBlock = (cst.inputDepthQuad + cst.blockCount - 1) / cst.blockCount;
    for (int bi=0; bi<cst.blockCount; ++bi) {
        // [N/4, cst.blockCount, 2/*scale_bias*/, N4]
        FLOAT4 scale0 = FLOAT4(dequantScale[2 * (idx_n40 * cst.blockCount + bi) + 0]) / (FLOAT)cst.scaleCoef;
        FLOAT4 dequant_bias0 = FLOAT4(dequantScale[2 * (idx_n40 * cst.blockCount + bi) + 1]) / (FLOAT)cst.scaleCoef;
        FLOAT4 scale1 = FLOAT4(dequantScale[2 * (idx_n41 * cst.blockCount + bi) + 0]) / (FLOAT)cst.scaleCoef;
        FLOAT4 dequant_bias1 = FLOAT4(dequantScale[2 * (idx_n41 * cst.blockCount + bi) + 1]) / (FLOAT)cst.scaleCoef;
        int zmin = bi * quadsPerBlock;
        int zmax = min(zmin + quadsPerBlock, cst.inputDepthQuad);

        for (int z = zmin + kl; z < zmax; z += 2) {
            ((threadgroup ftype4*)sdata)[2* rcl + kl] = *xy_in0;

            {
            #ifdef W_QUANT_4
                uchar2 w_int4 = xy_wt0[4*z]; // [N/4, K/4, N4, K4]
                FLOAT4 w4 = FLOAT4((float)(w_int4[0] >> 4) - 8, (float)(w_int4[0] & 15) - 8, (float)(w_int4[1] >> 4) - 8, (float)(w_int4[1] & 15) - 8);
            #elif defined(W_QUANT_8)
                char4 w_int4 = xy_wt0[4*z]; // [N/4, K/4, N4, K4]
                FLOAT4 w4 = FLOAT4((float)w_int4[0], (float)w_int4[1], (float)w_int4[2], (float)w_int4[3]);
            #endif
                FLOAT4 res = w4 * scale0[rcl % 4] + dequant_bias0[rcl % 4];
                //            sdata[32 + 2* rcl + kl] = res;
                ((threadgroup ftype*)sdata)[128 + (kl * 4 + 0) * 32 + rcl] = ftype(res[0]);
                ((threadgroup ftype*)sdata)[128 + (kl * 4 + 1) * 32 + rcl] = ftype(res[1]);
                ((threadgroup ftype*)sdata)[128 + (kl * 4 + 2) * 32 + rcl] = ftype(res[2]);
                ((threadgroup ftype*)sdata)[128 + (kl * 4 + 3) * 32 + rcl] = ftype(res[3]);
            }
            {
            #ifdef W_QUANT_4
                uchar2 w_int4 = xy_wt1[4*z]; // [N/4, K/4, N4, K4]
                FLOAT4 w4 = FLOAT4((float)(w_int4[0] >> 4) - 8, (float)(w_int4[0] & 15) - 8, (float)(w_int4[1] >> 4) - 8, (float)(w_int4[1] & 15) - 8);
            #elif defined(W_QUANT_8)
                char4 w_int4 = xy_wt1[4*z]; // [N/4, K/4, N4, K4]
                FLOAT4 w4 = FLOAT4((float)w_int4[0], (float)w_int4[1], (float)w_int4[2], (float)w_int4[3]);
            #endif
                FLOAT4 res = w4 * scale1[rcl % 4] + dequant_bias1[rcl % 4];
                //            sdata[32 + 2* rcl + kl] = res;
                ((threadgroup ftype*)sdata)[128 + (kl * 4 + 0) * 32 + 16 + rcl] = ftype(res[0]);
                ((threadgroup ftype*)sdata)[128 + (kl * 4 + 1) * 32 + 16 + rcl] = ftype(res[1]);
                ((threadgroup ftype*)sdata)[128 + (kl * 4 + 2) * 32 + 16 + rcl] = ftype(res[2]);
                ((threadgroup ftype*)sdata)[128 + (kl * 4 + 3) * 32 + 16 + rcl] = ftype(res[3]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_load(sga[0], (const threadgroup ftype*)sdata, 8);
            simdgroup_load(sga[1], ((const threadgroup ftype*)sdata) + 64, 8);

            simdgroup_load(sgb[0], ((const threadgroup ftype*)sdata) + 128, 32);
            simdgroup_load(sgb[1], ((const threadgroup ftype*)sdata) + 136, 32);
            simdgroup_load(sgb[2], ((const threadgroup ftype*)sdata) + 144, 32);
            simdgroup_load(sgb[3], ((const threadgroup ftype*)sdata) + 152, 32);

            SIMDGROUP_MATRIX_FMA(2, 4);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            xy_in0 += 2 * cst.inputSize * cst.batch;

        }
    }

    SIMDGROUP_MATRIX_STORE((threadgroup FLOAT*)sdata, 8);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if((16 * rx + rcl) < cst.inputSize * cst.batch) {
        if(4 * uz + 2 * kl < cst.outputDepthQuad) {
            xy_out[0] =  activate(ftype4(sdata[(kl * 16 + rcl) * 2 + 0] + FLOAT4(biasTerms[4 * uz + 2 * kl])), cst.activation);
        }
        if(4 * uz + 2 * kl + 1 < cst.outputDepthQuad) {
            xy_out[cst.outputSize * cst.batch] = activate(ftype4(sdata[(kl * 16 + rcl) * 2 + 1] + FLOAT4(biasTerms[4 * uz + 2 * kl + 1])), cst.activation);
        }
        if(cst.outputDepthQuad / 2 + 4 * uz + 2 * kl < cst.outputDepthQuad) {
            xy_out[cst.outputDepthQuad / 2 * cst.outputSize * cst.batch] = activate(ftype4(sdata[((kl + 2) * 16 + rcl) * 2 + 0] + FLOAT4(biasTerms[cst.outputDepthQuad / 2 + 4 * uz + 2 * kl])), cst.activation);
        }
        if(cst.outputDepthQuad / 2 + 4 * uz + 2 * kl + 1 < cst.outputDepthQuad) {
            xy_out[(cst.outputDepthQuad / 2 + 1) * cst.outputSize * cst.batch] = activate(ftype4(sdata[((kl + 2) * 16 + rcl) * 2 + 1] + FLOAT4(biasTerms[cst.outputDepthQuad / 2 + 4 * uz + 2 * kl + 1])), cst.activation);
        }
    }
}


kernel void conv1x1_gemm_32x64_wquant_split_k_sg(const device ftype4 *in            [[buffer(0)]],
                            device ftype4 *out                 [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                        #ifdef W_QUANT_4
                            const device uchar2 *wt      [[buffer(3)]],
                        #elif defined(W_QUANT_8)
                            const device char4 *wt      [[buffer(3)]],
                        #endif
                            const device ftype4 *biasTerms     [[buffer(4)]],
                            const device ftype *dequantScale   [[buffer(5)]],
                            uint3 gid                          [[threadgroup_position_in_grid]],
                            uint                  tiitg[[thread_index_in_threadgroup]],
                            uint                  tiisg[[thread_index_in_simdgroup]],
                            uint                  sgitg[[simdgroup_index_in_threadgroup]]) {
    /*
     // Read:
     ftype 0~1023   ---> input: [K4, M32, K8]
     ftype 1024~3071 ---> weight: [K4, K8, N64]
     ftype 3072~3199 ---> scale/offset: [N64, 2]
     // Write:
     ftype 0~2047 ---> input: [M2, N2, N2, N2, M2, M8, N8]
     */

    threadgroup FLOAT4 sdata[768] = {(FLOAT)0.f};

    INIT_SIMDGROUP_MATRIX(2, 4, 8);

    int rx = gid.x;// M/32
    int uz = gid.y;// N/64

    // A:[4, 2, 16]
    int ko = tiitg / 32;// 0~3
    int rcl = tiitg % 32;// 0~31
    int kl = rcl / 16;// 0~1
    int ml = rcl % 16;// 0~15 -> m
    // B:[16, 2, 4]
    int no = tiitg / 8;// 0~15
    int sl = tiitg % 8;// 0~7
    int kwl = sl / 4;// 0~1
    int nl = sl % 4;// 0~3

    /** input:
     threadgroup: [K4, M32, K8] -> [K4, M16, M2, K2, K4]
     index: [ko, ml, M2, kl, K4]
     each thread: M2K4
     layout: [K/4, M, K4] -> [K/32, K4, K2, M/32, M16, M2, K4]
     index : [K/32, ko, kl, rx, ml, M2, K4]
     */
    /** weight:
     threadgroup: [K4, K8, N64] -> [K2, K4, K4, N16, N4]
     index: [kwl, K4, K4, no, nl]
     each thread: K4K4
     layout: [N/4, K/4, N4, K4] -> [N/64, N16, K/32, K2, K4, N4, K4]
     index : [uz, no, K/32, kwl, K4, nl, K4]
     */
    /** scale/offset:
     layout:[N/4, blockCount, 2, N4] -> [N/64, N16, blockCount, 2, N4]
     index : [uz, no, blockCount, 2, nl]
     */
    /** output:
     threadgroup: [M32, N64] -> [M2, N2, N2, N2, M2, M8, N8]
     index [kl, ko/2, ko%2, N2, ml/8, ml%8, N2, N4]

     each thread: M4N4
     layout: [N/4, M, N4] -> [N/64, N4, N4, M/32, M2, M16, N4]
     index : [uz, ko, N4, rx, kl, ml, N4]
     */

    // boundary limit

    int idx_m20  = (rx * 16 + ml) * 2 + 0  < cst.inputSize * cst.batch ? (rx * 16 + ml) * 2 + 0 : (cst.inputSize * cst.batch - 1);
    int idx_m21  = (rx * 16 + ml) * 2 + 1  < cst.inputSize * cst.batch ? (rx * 16 + ml) * 2 + 1 : (cst.inputSize * cst.batch - 1);

    int idx_k4 = 0 * 8 + ko * 2 + kl;
    auto xy_in0  = in + idx_k4 * cst.inputSize * cst.batch + idx_m20;// [K/4, M, K4]
    auto xy_in1  = in + idx_k4 * cst.inputSize * cst.batch + idx_m21;// [K/4, M, K4]

    int idx_wk4 = 0 * 8 + kwl * 4 + 0;
    int idx_n4 = (uz * 16 + no) < cst.outputDepthQuad ? (uz * 16 + no) : (cst.outputDepthQuad - 1);
    auto xy_wt = wt +  (idx_n4 * cst.inputDepthQuad + idx_wk4) * 4 + nl;// [N/4, K/4, N4, K4]

    int idx_sa = (ko * 32 + ml * 2 + 0) * 2 + kl;
    int idx_sb = 1024 + (kwl * 16 + 0) * 64 + no * 4 + nl;
    int quadsPerBlock = (cst.inputDepthQuad + cst.blockCount - 1) / cst.blockCount;
    for (int bi=0; bi<cst.blockCount; ++bi) {
        // [N/4, cst.blockCount, 2/*scale_bias*/, N4]
        FLOAT scale0 = FLOAT(dequantScale[((idx_n4 * cst.blockCount + bi) * 2 + 0) * 4 + nl]) / (FLOAT)cst.scaleCoef;
        FLOAT dequant_bias0 = FLOAT(dequantScale[((idx_n4 * cst.blockCount + bi) * 2 + 1) * 4 + nl]) / (FLOAT)cst.scaleCoef;

        int zmin = bi * quadsPerBlock;
        int zmax = min(zmin + quadsPerBlock, cst.inputDepthQuad);

        for (int z = zmin; z < zmax; z += 8) {


            FLOAT4x4 w_dequant; // K4K4
            {
            #ifdef W_QUANT_4
                #pragma unroll(4)
                for (int i = 0; i < 4; i += 1) {
                    uchar2 w_int4 = xy_wt[(z + i) * 4];
                    w_dequant[i][0] = FLOAT(w_int4[0] >> 4);
                    w_dequant[i][1] = FLOAT(w_int4[0] & 0x000F);
                    w_dequant[i][2] = FLOAT(w_int4[1] >> 4);
                    w_dequant[i][3] = FLOAT(w_int4[1] & 0x000F);
                }
                FLOAT4 val = FLOAT4(dequant_bias0 - 8.0 * scale0);
                w_dequant = w_dequant * scale0 + FLOAT4x4(val, val, val, val);

            #elif defined(W_QUANT_8)
                #pragma unroll(4)
                for (int i = 0; i < 4; ++i) {
                    auto w = xy_wt[(z + i) * 4];
                    FLOAT4 w_fp32 = FLOAT4(FLOAT(w[0]), FLOAT(w[1]), FLOAT(w[2]), FLOAT(w[3]));
                    w_dequant[i] = w_fp32 * scale0 + dequant_bias0;
                }
            #endif
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            #pragma unroll(16)
            for (int i = 0; i < 16; ++i) {
                ((threadgroup ftype*)sdata)[idx_sb + 64*i]  = ftype(w_dequant[i/4][i%4]); // K4K4
            }

            ((threadgroup ftype4*)sdata)[idx_sa]     = (ftype4)*(xy_in0);
            ((threadgroup ftype4*)sdata)[idx_sa + 2] = (ftype4)*(xy_in1);



            threadgroup_barrier(mem_flags::mem_threadgroup);

            /*
            A: [K4, M32, K8] -> [K4, M2, M16, K8]
            index: [ik, sgitg/2, sga[0~1]]

            B: [K4, K8, N64] -> [K4, K8, N2, N32]
            index: [ik, sgitg%2, sgb[0~3]]

            sgitg: compute M2 and N2
            */
            threadgroup ftype * sdata_a = (threadgroup ftype*)sdata + 16*8*(sgitg/2);
            threadgroup ftype * sdata_b = (threadgroup ftype*)sdata + 1024 + 32*(sgitg%2);

            #pragma unroll(4)
            for (short ik = 0; ik < 4; ik++) {
                simdgroup_load(sga[0], (const threadgroup ftype*)sdata_a + 256 * ik, 8);
                simdgroup_load(sga[1], ((const threadgroup ftype*)sdata_a) + 256 * ik + 64, 8);

                simdgroup_load(sgb[0], ((threadgroup ftype*)sdata_b) + 512 * ik + 0,  64);
                simdgroup_load(sgb[1], ((threadgroup ftype*)sdata_b) + 512 * ik + 8,  64);
                simdgroup_load(sgb[2], ((threadgroup ftype*)sdata_b) + 512 * ik + 16, 64);
                simdgroup_load(sgb[3], ((threadgroup ftype*)sdata_b) + 512 * ik + 24, 64);

                simdgroup_barrier(mem_flags::mem_none);
                SIMDGROUP_MATRIX_FMA(2, 4);

                simdgroup_barrier(mem_flags::mem_none);
            }

            xy_in0 += 8 * cst.inputSize * cst.batch;
            xy_in1 += 8 * cst.inputSize * cst.batch;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup FLOAT * sdata_c = (threadgroup FLOAT*)sdata + 512*sgitg;

    SIMDGROUP_MATRIX_STORE((threadgroup FLOAT*)sdata_c, 8);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // layout: [N/4, M, N4] -> [N/64, N4, N4, M/32, M2, M16, N4]
    // index : [uz, ko, N4, rx, kl, ml, N4]
    auto xy_out = out + ((uz * 4 + ko) * 4 + 0) * cst.outputSize * cst.batch + (rx * 2 + kl) * 16 + ml;// [N/4, M, N4]

    // sdata [M2, N2, N2, N2, M2, M8, N8]
    // index [kl, ko/2, ko%2, N2, ml/8, ml%8, N2, N4]
    if((rx * 32 + kl * 16 + ml) < cst.inputSize * cst.batch) {
        if((uz * 4 + ko) * 4 < cst.outputDepthQuad) {
            xy_out[0] =  activate(ftype4(((threadgroup FLOAT4*)sdata)[(((kl * 4 + ko) * 2 + 0) * 16 + ml) * 2] + FLOAT4(biasTerms[(uz * 4 + ko) * 4])), cst.activation);
        }
        if((uz * 4 + ko) * 4 + 1 < cst.outputDepthQuad) {
            xy_out[cst.outputSize * cst.batch] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(((kl * 4 + ko) * 2 + 0) * 16 + ml) * 2 + 1] + FLOAT4(biasTerms[(uz * 4 + ko) * 4 + 1])), cst.activation);
        }
        if((uz * 4 + ko) * 4 + 2 < cst.outputDepthQuad) {
            xy_out[cst.outputSize * cst.batch * 2] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(((kl * 4 + ko) * 2 + 1) * 16 + ml) * 2] + FLOAT4(biasTerms[(uz * 4 + ko) * 4 + 2])), cst.activation);
        }
        if((uz * 4 + ko) * 4 + 3 < cst.outputDepthQuad) {
            xy_out[cst.outputSize * cst.batch * 3] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(((kl * 4 + ko) * 2 + 1) * 16 + ml) * 2 + 1] + FLOAT4(biasTerms[(uz * 4 + ko) * 4 + 3])), cst.activation);
        }
    }
}


#endif // !defined(W_QUANT_2) && !defined(W_QUANT_3)
)metal";

static const char* gConv1x1WfpSgMatrix = R"metal(
#ifdef USE_METAL_TENSOR_OPS
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif


// W_QUANT_2/3 dequant path is implemented in conv1x1_w_dequant (prefill outer-dequant).
// The fused_q4_gemm_stage kernels are guarded by #if defined(W_QUANT_4) || defined(W_QUANT_8)
// and the fp GEMM kernels don't use W_QUANT macros, so W2/W3 compilation is safe.

kernel void conv1x1_w_dequant(
                        #if defined(W_QUANT_2) || defined(W_QUANT_3)
                            const device uchar *wi      [[buffer(0)]],
                        #elif defined(W_QUANT_4)
                            const device uchar2 *wi      [[buffer(0)]],
                        #elif defined(W_QUANT_8)
                            const device char4 *wi      [[buffer(0)]],
                        #else
                            const device ftype4 *wi      [[buffer(0)]],// [N/4, K/4, N4, K4]
                        #endif
                            device ftype4 *wf      [[buffer(1)]],// [N/4, K/16, N4, K4, K4]
                            constant conv1x1_constants& cst    [[buffer(2)]],
                            const device ftype4 *dequantScale  [[buffer(3)]],
                            uint3 gid                          [[thread_position_in_grid]]
) {

    int idx_n = gid.x; // N
    int idx_k16 = gid.y; // K/16

    int idx_n4 = idx_n/4;
    int idx_nl = idx_n%4;
    int idx_k4 = idx_k16 * 4;

    if(idx_n4 >= cst.outputDepthQuad || idx_k4 >= cst.inputDepthQuad) {
        return;
    }

    int quadsPerBlock = (cst.inputDepthQuad + cst.blockCount - 1) / cst.blockCount;


    int bi = idx_k4 / quadsPerBlock;
    // [N/4, cst.blockCount, 2/*scale_bias*/, N4]
    FLOAT scale = FLOAT(((const device ftype *)dequantScale)[((idx_n4 * cst.blockCount + bi) * 2 + 0) * 4 + idx_nl]) / (FLOAT)cst.scaleCoef;
    FLOAT dequant_bias = FLOAT(((const device ftype *)dequantScale)[((idx_n4 * cst.blockCount + bi) * 2 + 1) * 4 + idx_nl]) / (FLOAT)cst.scaleCoef;

#ifdef W_QUANT_3
    auto wt_base = wi + (idx_n4 * cst.inputDepthQuad + idx_k4) * 6;
#elif !defined(W_QUANT_2)
    auto xy_wi = wi + (idx_n4 * cst.inputDepthQuad + idx_k4) * 4 + idx_nl;// [N/4, K/4, N4, K4]
#endif
    auto xy_wf = wf + ((idx_n4 * ((cst.inputDepthQuad+3)/4) + idx_k16) * 4 + idx_nl) * 4;// [N/4, K/4, N4, K4]

    #ifdef W_QUANT_2
    #if W_ALIGN_K16_PROTECT
    // Tail-protected path: 4B-aligned uchar4 load per K/4-quad (vs scattered
    // 1-byte loads), nl byte extracted in-register.
    {
        auto xy_wi4 = (const device uchar4*)wi + idx_n4 * cst.inputDepthQuad + idx_k4;
        for(int k = 0; k < 4; k++) {
            if(idx_k4 + k >= cst.inputDepthQuad) { xy_wf[k] = ftype4(0); continue; }
            uchar b = xy_wi4[k][idx_nl];
            FLOAT4 w4 = FLOAT4((float)((b >> 6) & 3) - 2, (float)((b >> 4) & 3) - 2,
                                (float)((b >> 2) & 3) - 2, (float)( b       & 3) - 2);
            xy_wf[k] = (ftype4)(w4 * scale + dequant_bias);
        }
    }
    #else
    // ic%16==0 here, so the 16B tile (4 K/4-quads x 4 N4 bytes) is 16B-aligned:
    // one uint4 load replaces four scattered byte loads.
    {
        uint4 w16 = ((const device uint4*)wi)[idx_n4 * (cst.inputDepthQuad / 4) + idx_k16];
        int sh = 8 * idx_nl;
        for(int k = 0; k < 4; k++) {
            uchar b = uchar((w16[k] >> sh) & 0xFF);
            FLOAT4 w4 = FLOAT4((float)((b >> 6) & 3) - 2, (float)((b >> 4) & 3) - 2,
                                (float)((b >> 2) & 3) - 2, (float)( b       & 3) - 2);
            xy_wf[k] = (ftype4)(w4 * scale + dequant_bias);
        }
    }
    #endif
    #elif defined(W_QUANT_3)
    for(int k = 0; k < 4; k++) {
        #if W_ALIGN_K16_PROTECT
        if(idx_k4 + k >= cst.inputDepthQuad) { xy_wf[k] = ftype4(0); continue; }
        #endif
        const device uchar* tilePtr = wt_base + 6 * k;
        uchar b = tilePtr[idx_nl];
        uchar h = (idx_nl < 2) ? tilePtr[4] : tilePtr[5];
        uchar hShifted = (idx_nl % 2 == 0) ? (h >> 4) : (h & 0xF);
        FLOAT4 w4 = FLOAT4(
            (float)( ((b >> 6) & 3) | (((hShifted >> 3) & 1) << 2) ) - 4,
            (float)( ((b >> 4) & 3) | (((hShifted >> 2) & 1) << 2) ) - 4,
            (float)( ((b >> 2) & 3) | (((hShifted >> 1) & 1) << 2) ) - 4,
            (float)( ( b       & 3) | (( hShifted       & 1) << 2) ) - 4);
        xy_wf[k] = (ftype4)(w4 * scale + dequant_bias);
    }
    #elif defined(W_QUANT_4)
    for(int k = 0; k < 4; k++) {
        #if W_ALIGN_K16_PROTECT
        {
            if(idx_k4 + k >= cst.inputDepthQuad) {
                xy_wf[k] = ftype4(0);
            } else {
                uchar2 w_int4 = xy_wi[4*k]; // [N/4, K/4, N4, K4]
                FLOAT4 w4 = FLOAT4((float)(w_int4[0] >> 4) - 8, (float)(w_int4[0] & 15) - 8, (float)(w_int4[1] >> 4) - 8, (float)(w_int4[1] & 15) - 8);
                FLOAT4 res = w4 * scale + dequant_bias;
                xy_wf[k] = (ftype4)res;
            }
        }
        #else
        {
            uchar2 w_int4 = xy_wi[4*k]; // [N/4, K/4, N4, K4]
            FLOAT4 w4 = FLOAT4((float)(w_int4[0] >> 4) - 8, (float)(w_int4[0] & 15) - 8, (float)(w_int4[1] >> 4) - 8, (float)(w_int4[1] & 15) - 8);
            FLOAT4 res = w4 * scale + dequant_bias;
            xy_wf[k] = (ftype4)res;
        }
        #endif
    }
    #elif defined(W_QUANT_8)
    for(int k = 0; k < 4; k++) {
        #if W_ALIGN_K16_PROTECT
        {
            if(idx_k4 + k >= cst.inputDepthQuad) {
                xy_wf[k] = ftype4(0);
            } else {
                char4 w_int4 = xy_wi[4*k]; // [N/4, K/4, N4, K4]
                FLOAT4 w4 = FLOAT4((float)w_int4[0], (float)w_int4[1], (float)w_int4[2], (float)w_int4[3]);
                FLOAT4 res = w4 * scale + dequant_bias;
                xy_wf[k] = (ftype4)res;
            }
        }
        #else
        {
            char4 w_int4 = xy_wi[4*k]; // [N/4, K/4, N4, K4]
            FLOAT4 w4 = FLOAT4((float)w_int4[0], (float)w_int4[1], (float)w_int4[2], (float)w_int4[3]);
            FLOAT4 res = w4 * scale + dequant_bias;
            xy_wf[k] = (ftype4)res;
        }
        #endif
    }
    #endif

}

kernel void conv1x1_gemm_32x64_split_k_sg(const device ftype4 *in            [[buffer(0)]],
                            device ftype4 *out                 [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                        #ifdef W_QUANT_4
                            const device MNN::uchar4x2 *wt      [[buffer(3)]],// [N/4, K/16, N4, K4, K4]
                        #elif defined(W_QUANT_8)
                            const device MNN::char4x4 *wt      [[buffer(3)]],// [N/4, K/16, N4, K4, K4]
                        #else
                            const device ftype4x4 *wt      [[buffer(3)]],// [N/4, K/16, N4, K4, K4]
                        #endif
                            const device ftype4 *biasTerms     [[buffer(4)]],
                        #if defined(W_QUANT_4) || defined(W_QUANT_8)
                            const device ftype *dequantScale  [[buffer(5)]],
                        #endif
                            uint3 gid                          [[threadgroup_position_in_grid]],
                            uint                  tiitg[[thread_index_in_threadgroup]],
                            uint                  tiisg[[thread_index_in_simdgroup]],
                            uint                  sgitg[[simdgroup_index_in_threadgroup]]) {

#ifdef USE_METAL_TENSOR_OPS

#ifdef LOOP_K64
    /*
     // Read:
     ftype 0~2047   ---> input: [M32, K64]
     ftype 2048~6015 ---> weight: [N64, K64]
     // Write:
     FLOAT 0~2047 ---> input: [M32, N64]
     */
    threadgroup ftype4 sdata[1536] = {0.f};

    const int K = 64, M = 32, N = 64;
    auto tI = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata, dextents<int32_t, 2>(K, M));//[M, K]
    auto tW = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata + 2048, dextents<int32_t, 2>(K, N));//[N, K]

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(M, N, K, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mmOps;

    auto cT = mmOps.get_destination_cooperative_tensor<decltype(tI), decltype(tW), FLOAT>();

    int rx = gid.x;// M/32
    int uz = gid.y;// N/64

    // A:[16， 8]
    int kl = tiitg / 8;// 0~15
    int ml = tiitg % 8;// 0~7

    // B:[16, 2, 4]
    int no = tiitg / 8;// 0~15
    int sl = tiitg % 8;// 0~7
    int kwl = sl / 4;// 0~1
    int nl = sl % 4;// 0~3

    // C:[32, 4]
    int mlc = tiitg / 4;// 0~31
    int nlc = tiitg % 4;// 0~3
    /** input:
     threadgroup: [M32, K64] -> [M8, M4, K16, K4]
     index: [ml, M4, kl, K4]
     each thread: M4K4
     layout: [K/4, M, K4] -> [K/64, K16, M/32, M8, M4, K4]
     index : [K/64, kl, rx, ml, M4, K4]
     */
    /** weight:
     threadgroup: [N64, K64] -> [N16, N4, K2, K32]
     index: [no, nl, kwl, K32]
     each thread: K2K16
     layout: [N/4, K/16, N4, K4, K4] -> [N/64, N16, K/64, K2, K2, N4, K4, K4]
     index : [uz, no, K/64, kwl, K2, nl, K4, K4]
     */
    /** scale/offset:
     layout:[N/4, blockCount, 2, N4] -> [N/64, N16, blockCount, 2, N4]
     index : [uz, no, blockCount, 2, nl]
     */
    /** output:
     threadgroup: [M32, N64] -> [M32, N4, N16]
     index [mlc, nlc, N16]

     each thread: N16
     layout: [N/4, M, N4] -> [N/64, N4, N4, M/32, M32, N4]
     index : [uz, nlc, N4, rx, mlc, N4]
     */

    // boundary limit
    int idx_m40  = (rx * 8 + ml) * 4 + 0  < cst.inputSize * cst.batch ? (rx * 8 + ml) * 4 + 0 : (cst.inputSize * cst.batch - 1);
    int idx_m41  = (rx * 8 + ml) * 4 + 1  < cst.inputSize * cst.batch ? (rx * 8 + ml) * 4 + 1 : (cst.inputSize * cst.batch - 1);
    int idx_m42  = (rx * 8 + ml) * 4 + 2  < cst.inputSize * cst.batch ? (rx * 8 + ml) * 4 + 2 : (cst.inputSize * cst.batch - 1);
    int idx_m43  = (rx * 8 + ml) * 4 + 3  < cst.inputSize * cst.batch ? (rx * 8 + ml) * 4 + 3 : (cst.inputSize * cst.batch - 1);

    int idx_k4 = 0 * 16 + kl;
    auto xy_in0  = in + idx_k4 * cst.inputSize * cst.batch + idx_m40;// [K/4, M, K4]
    auto xy_in1  = in + idx_k4 * cst.inputSize * cst.batch + idx_m41;// [K/4, M, K4]
    auto xy_in2  = in + idx_k4 * cst.inputSize * cst.batch + idx_m42;// [K/4, M, K4]
    auto xy_in3  = in + idx_k4 * cst.inputSize * cst.batch + idx_m43;// [K/4, M, K4]

    int idx_wk16 = (0 * 2 + kwl) * 2 + 0;

    int idx_n4 = (uz * 16 + no) < cst.outputDepthQuad ? (uz * 16 + no) : (cst.outputDepthQuad - 1);
    auto xy_wt = wt +  (idx_n4 * ((cst.inputDepthQuad+3)/4) + idx_wk16) * 4 + nl;// [N/4, K/16, N4, K4, K4]

    int idx_sa = (ml * 4 + 0) * 16 + kl; // [M8, M4, K16] x [K4]
    int idx_sb = 512 + ((no * 4 + nl) * 2 + kwl) * 8 + 0; // [N16 N4, K2, K8] x [K4]
    int quadsPerBlock = (cst.inputDepthQuad + cst.blockCount - 1) / cst.blockCount;

    for (int bi=0; bi<cst.blockCount; ++bi) {
        int zmin = bi * quadsPerBlock;
        int zmax = min(zmin + quadsPerBlock, cst.inputDepthQuad);

        for (int z = zmin; z < zmax; z += 16) {
            FLOAT4x4 w_dequant_0; // K4K4
            {
                auto w = xy_wt[z];
                w_dequant_0 = FLOAT4x4((FLOAT4)w[0], (FLOAT4)w[1], (FLOAT4)w[2], (FLOAT4)w[3]);
            }
            FLOAT4x4 w_dequant_1; // K4K4
            {
                auto w = xy_wt[z + 4];
                w_dequant_1 = FLOAT4x4((FLOAT4)w[0], (FLOAT4)w[1], (FLOAT4)w[2], (FLOAT4)w[3]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            #pragma unroll(4)
            for (int i = 0; i < 4; ++i) {
                ((threadgroup ftype4*)sdata)[idx_sb + i]  = ftype4(w_dequant_0[i]); // K4K4
            }
            #pragma unroll(4)
            for (int i = 0; i < 4; ++i) {
                ((threadgroup ftype4*)sdata)[idx_sb + 4 + i]  = ftype4(w_dequant_1[i]); // K4K4
            }

            ((threadgroup ftype4*)sdata)[idx_sa]      = (ftype4)*(xy_in0);
            ((threadgroup ftype4*)sdata)[idx_sa + 16] = (ftype4)*(xy_in1);
            ((threadgroup ftype4*)sdata)[idx_sa + 32] = (ftype4)*(xy_in2);
            ((threadgroup ftype4*)sdata)[idx_sa + 48] = (ftype4)*(xy_in3);

            threadgroup_barrier(mem_flags::mem_threadgroup);


            auto sA = tI.slice(0, 0);
            auto sB = tW.slice(0, 0);

            mmOps.run(sA, sB, cT);

            xy_in0 += 16 * cst.inputSize * cst.batch;
            xy_in1 += 16 * cst.inputSize * cst.batch;
            xy_in2 += 16 * cst.inputSize * cst.batch;
            xy_in3 += 16 * cst.inputSize * cst.batch;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto tC = tensor<threadgroup FLOAT, dextents<int32_t, 2>, tensor_inline>((threadgroup FLOAT*)sdata, dextents<int32_t, 2>(N, M)); // [M , N]
    cT.store(tC);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // each thread: N16
    // layout: [N/4, M, N4] -> [N/64, N4, N4, M/32, M32, N4]
    // index : [uz, nlc, N4, rx, mlc, N4]

    auto xy_out = out + ((uz * 4 + nlc) * 4 + 0) * cst.outputSize * cst.batch + (rx * 32 + mlc);// [N/4, M, N4]
    // sdata: [M32, N64] -> [M32, N4, N16]
    // index [mlc, nlc, N16]
    if((rx * 32 + mlc) < cst.inputSize * cst.batch) {
        if((uz * 4 + nlc) * 4 < cst.outputDepthQuad) {
            xy_out[0] =  activate(ftype4(((threadgroup FLOAT4*)sdata)[(mlc * 4 + nlc) * 4 + 0] + FLOAT4(biasTerms[(uz * 4 + nlc) * 4])), cst.activation);
        }
        if((uz * 4 + nlc) * 4 + 1 < cst.outputDepthQuad) {
            xy_out[cst.outputSize * cst.batch] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(mlc * 4 + nlc) * 4 + 1] + FLOAT4(biasTerms[(uz * 4 + nlc) * 4 + 1])), cst.activation);
        }
        if((uz * 4 + nlc) * 4 + 2 < cst.outputDepthQuad) {
            xy_out[cst.outputSize * cst.batch * 2] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(mlc * 4 + nlc) * 4 + 2] + FLOAT4(biasTerms[(uz * 4 + nlc) * 4 + 2])), cst.activation);
        }
        if((uz * 4 + nlc) * 4 + 3 < cst.outputDepthQuad) {
            xy_out[cst.outputSize * cst.batch * 3] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(mlc * 4 + nlc) * 4 + 3] + FLOAT4(biasTerms[(uz * 4 + nlc) * 4 + 3])), cst.activation);
        }
    }
#else
    /*
     // Read:
     ftype 0~1023   ---> input: [M32, K32]
     ftype 1024~3071 ---> weight: [N64, K32]
     // Write:
     FLOAT 0~2047 ---> input: [M32, N64]
     */
    threadgroup FLOAT4 sdata[800] = {0.f};

    const int K = 32, M = 32, N = 64;
    auto tI = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata, dextents<int32_t, 2>(K, M));//[M, K]
    auto tW = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata + 1024, dextents<int32_t, 2>(K, N));//[N, K]

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(M, N, K, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mmOps;

    auto cT = mmOps.get_destination_cooperative_tensor<decltype(tI), decltype(tW), FLOAT>();

    int rx = gid.x;// M/32
    int uz = gid.y;// N/64

    // A:[8， 16]
    int kl = tiitg / 16;// 0~7
    int ml = tiitg % 16;// 0~15

    // B:[16, 4, 2]
    int no = tiitg / 8;// 0~15
    int sl = tiitg % 8;// 0~7
    int nl = sl / 2;// 0~3
    int kwl = sl % 2;// 0~1

    // C:[32, 4]
    int mlc = tiitg / 4;// 0~31
    int nlc = tiitg % 4;// 0~3
    /** input:
     threadgroup: [M32, K32] -> [M16, M2, K8, K4]
     index: [ml, M2, kl, K4]
     each thread: M2K4
     layout: [K/4, M, K4] -> [K/32, K8, M/32, M16, M2, K4]
     index : [K/32, kl, rx, ml, M2, K4]
     */
    /** weight:
     threadgroup: [N64, K32] -> [N16 N4, K2, K16]
     index: [no, nl, kwl, K16]
     each thread: K4K4
     layout: [N/4, K/16, N4, K4, K4] -> [N/64, N16, K/32, K2, N4, K4, K4]
     index : [uz, no, K/32, kwl, nl, K4, K4]
     */
    /** scale/offset:
     layout:[N/4, blockCount, 2, N4] -> [N/64, N16, blockCount, 2, N4]
     index : [uz, no, blockCount, 2, nl]
     */
    /** output:
     threadgroup: [M32, N64] -> [M32, N4, N16]
     index [mlc, nlc, N16]

     each thread: N16
     layout: [N/4, M, N4] -> [N/64, N4, N4, M/32, M32, N4]
     index : [uz, nlc, N4, rx, mlc, N4]
     */

    // boundary limit
    int idx_m20  = (rx * 16 + ml) * 2 + 0  < cst.inputSize * cst.batch ? (rx * 16 + ml) * 2 + 0 : (cst.inputSize * cst.batch - 1);
    int idx_m21  = (rx * 16 + ml) * 2 + 1  < cst.inputSize * cst.batch ? (rx * 16 + ml) * 2 + 1 : (cst.inputSize * cst.batch - 1);

    int idx_k4 = 0 * 8 + kl;
    auto xy_in0  = in + idx_k4 * cst.inputSize * cst.batch + idx_m20;// [K/4, M, K4]
    auto xy_in1  = in + idx_k4 * cst.inputSize * cst.batch + idx_m21;// [K/4, M, K4]

    int idx_wk16 = 0 * 2 + kwl;

    int idx_n4 = (uz * 16 + no) < cst.outputDepthQuad ? (uz * 16 + no) : (cst.outputDepthQuad - 1);
    auto xy_wt = wt +  (idx_n4 * ((cst.inputDepthQuad+3)/4) + idx_wk16) * 4 + nl;// [N/4, K/16, N4, K4, K4]

    int idx_sa = (ml * 2 + 0) * 8 + kl; // [M16, M2, K8] x [K4]
    int idx_sb = 256 + ((no * 4 + nl) * 2 + kwl) * 4 + 0; // [N16 N4, K2, K4] x [K4]
    int quadsPerBlock = (cst.inputDepthQuad + cst.blockCount - 1) / cst.blockCount;

    for (int bi=0; bi<cst.blockCount; ++bi) {
    #if defined(W_QUANT_4) || defined(W_QUANT_8)
        // [N/4, cst.blockCount, 2/*scale_bias*/, N4]
        FLOAT scale0 = FLOAT(dequantScale[((idx_n4 * cst.blockCount + bi) * 2 + 0) * 4 + nl]) / (FLOAT)cst.scaleCoef;
        FLOAT dequant_bias0 = FLOAT(dequantScale[((idx_n4 * cst.blockCount + bi) * 2 + 1) * 4 + nl]) / (FLOAT)cst.scaleCoef;

    #endif
        int zmin = bi * quadsPerBlock;
        int zmax = min(zmin + quadsPerBlock, cst.inputDepthQuad);

        for (int z = zmin; z < zmax; z += 8) {
            FLOAT4x4 w_dequant; // K4K4
            {
            #ifdef W_QUANT_4
                MNN::uchar4x2 w_int4 = xy_wt[z];

                auto temp = FLOAT4(uchar4(w_int4[0][0], w_int4[1][0], w_int4[2][0], w_int4[3][0]) >> 4);
                w_dequant[0][0] = temp[0];
                w_dequant[1][0] = temp[1];
                w_dequant[2][0] = temp[2];
                w_dequant[3][0] = temp[3];
                temp = FLOAT4(uchar4(w_int4[0][0], w_int4[1][0], w_int4[2][0], w_int4[3][0]) & 0x000F);
                w_dequant[0][1] = temp[0];
                w_dequant[1][1] = temp[1];
                w_dequant[2][1] = temp[2];
                w_dequant[3][1] = temp[3];
                temp = FLOAT4(uchar4(w_int4[0][1], w_int4[1][1], w_int4[2][1], w_int4[3][1]) >> 4);
                w_dequant[0][2] = temp[0];
                w_dequant[1][2] = temp[1];
                w_dequant[2][2] = temp[2];
                w_dequant[3][2] = temp[3];
                temp = FLOAT4(uchar4(w_int4[0][1], w_int4[1][1], w_int4[2][1], w_int4[3][1]) & 0x000F);
                w_dequant[0][3] = temp[0];
                w_dequant[1][3] = temp[1];
                w_dequant[2][3] = temp[2];
                w_dequant[3][3] = temp[3];

                FLOAT4 val = FLOAT4(dequant_bias0 - 8.0 * scale0);
                w_dequant = w_dequant * scale0 + FLOAT4x4(val, val, val, val);

            #elif defined(W_QUANT_8)
                auto w = xy_wt[z];
                FLOAT4x4 w_fp32 = FLOAT4x4(FLOAT4(w[0]), FLOAT4(w[1]), FLOAT4(w[2]), FLOAT4(w[3]));
                for (int i = 0; i < 4; ++i) {
                    w_dequant[i] = w_fp32[i] * scale0 + dequant_bias0;
                }
            #else
                auto w = xy_wt[z];
                w_dequant = FLOAT4x4((FLOAT4)w[0], (FLOAT4)w[1], (FLOAT4)w[2], (FLOAT4)w[3]);
            #endif
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            #pragma unroll(4)
            for (int i = 0; i < 4; ++i) {
                ((threadgroup ftype4*)sdata)[idx_sb + i]  = ftype4(w_dequant[i]); // K4K4
            }

            #ifdef MNN_METAL_SRC_PROTECT
            if (idx_k4 + z < cst.inputDepthQuad) {
                ((threadgroup ftype4*)sdata)[idx_sa]     = (ftype4)*(xy_in0);
                ((threadgroup ftype4*)sdata)[idx_sa + 8] = (ftype4)*(xy_in1);
            } else {
                ((threadgroup ftype4*)sdata)[idx_sa]     = (ftype4)(0);
                ((threadgroup ftype4*)sdata)[idx_sa + 8] = (ftype4)(0);
            }
            #else
            ((threadgroup ftype4*)sdata)[idx_sa]     = (ftype4)*(xy_in0);
            ((threadgroup ftype4*)sdata)[idx_sa + 8] = (ftype4)*(xy_in1);
            #endif

            threadgroup_barrier(mem_flags::mem_threadgroup);


            auto sA = tI.slice(0, 0);
            auto sB = tW.slice(0, 0);

            mmOps.run(sA, sB, cT);

            xy_in0 += 8 * cst.inputSize * cst.batch;
            xy_in1 += 8 * cst.inputSize * cst.batch;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto tC = tensor<threadgroup FLOAT, dextents<int32_t, 2>, tensor_inline>((threadgroup FLOAT*)sdata, dextents<int32_t, 2>(N, M)); // [M , N]
    cT.store(tC);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // each thread: N16
    // layout: [N/4, M, N4] -> [N/64, N4, N4, M/32, M32, N4]
    // index : [uz, nlc, N4, rx, mlc, N4]

    auto xy_out = out + ((uz * 4 + nlc) * 4 + 0) * cst.outputSize * cst.batch + (rx * 32 + mlc);// [N/4, M, N4]
    // sdata: [M32, N64] -> [M32, N4, N16]
    // index [mlc, nlc, N16]
    if((rx * 32 + mlc) < cst.inputSize * cst.batch) {
        if((uz * 4 + nlc) * 4 < cst.outputDepthQuad) {
            xy_out[0] =  activate(ftype4(((threadgroup FLOAT4*)sdata)[(mlc * 4 + nlc) * 4 + 0] + FLOAT4(biasTerms[(uz * 4 + nlc) * 4])), cst.activation);
        }
        if((uz * 4 + nlc) * 4 + 1 < cst.outputDepthQuad) {
            xy_out[cst.outputSize * cst.batch] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(mlc * 4 + nlc) * 4 + 1] + FLOAT4(biasTerms[(uz * 4 + nlc) * 4 + 1])), cst.activation);
        }
        if((uz * 4 + nlc) * 4 + 2 < cst.outputDepthQuad) {
            xy_out[cst.outputSize * cst.batch * 2] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(mlc * 4 + nlc) * 4 + 2] + FLOAT4(biasTerms[(uz * 4 + nlc) * 4 + 2])), cst.activation);
        }
        if((uz * 4 + nlc) * 4 + 3 < cst.outputDepthQuad) {
            xy_out[cst.outputSize * cst.batch * 3] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(mlc * 4 + nlc) * 4 + 3] + FLOAT4(biasTerms[(uz * 4 + nlc) * 4 + 3])), cst.activation);
        }
    }
#endif
#else
    /*
     // Read:
     ftype 0~1023   ---> input: [K4, M32, K8]
     ftype 1024~3071 ---> weight: [K4, K8, N64]
     ftype 3072~3199 ---> scale/offset: [N64, 2]
     // Write:
     FLOAT 0~2047 ---> input: [M2, N2, N2, N2, M2, M8, N8]
     */
    threadgroup FLOAT4 sdata[800] = {0.f};

    INIT_SIMDGROUP_MATRIX(2, 4, 8);

    int rx = gid.x;// M/32
    int uz = gid.y;// N/64

    // A:[4, 2, 16]
    int ko = tiitg / 32;// 0~3
    int rcl = tiitg % 32;// 0~31
    int kl = rcl / 16;// 0~1
    int ml = rcl % 16;// 0~15 -> m
    // B:[16, 2, 4]
    int no = tiitg / 8;// 0~15
    int sl = tiitg % 8;// 0~7
    int kwl = sl / 4;// 0~1
    int nl = sl % 4;// 0~3

    /** input:
     threadgroup: [K4, M32, K8] -> [K4, M16, M2, K2, K4]
     index: [ko, ml, M2, kl, K4]
     each thread: M2K4
     layout: [K/4, M, K4] -> [K/32, K4, K2, M/32, M16, M2, K4]
     index : [K/32, ko, kl, rx, ml, M2, K4]
     */
    /** weight:
     threadgroup: [K4, K8, N64] -> [K2, K4, K4, N16, N4]
     index: [kwl, K4, K4, no, nl]
     each thread: K4K4
     layout: [N/4, K/16, N4, K4, K4] -> [N/64, N16, K/32, K2,  N4, K4, K4]
     index : [uz, no, K/32, kwl, nl, K4, K4]
     */
    /** scale/offset:
     layout:[N/4, blockCount, 2, N4] -> [N/64, N16, blockCount, 2, N4]
     index : [uz, no, blockCount, 2, nl]
     */
    /** output:
     threadgroup: [M32, N64] -> [M2, N2, N2, N2, M2, M8, N8]
     index [kl, ko/2, ko%2, N2, ml/8, ml%8, N2, N4]

     each thread: N16
     layout: [N/4, M, N4] -> [N/64, N4, N4, M/32, M2, M16, N4]
     index : [uz, ko, N4, rx, kl, ml, N4]
     */

    // boundary limit
    int idx_m20  = (rx * 16 + ml) * 2 + 0  < cst.inputSize * cst.batch ? (rx * 16 + ml) * 2 + 0 : (cst.inputSize * cst.batch - 1);
    int idx_m21  = (rx * 16 + ml) * 2 + 1  < cst.inputSize * cst.batch ? (rx * 16 + ml) * 2 + 1 : (cst.inputSize * cst.batch - 1);

    int idx_k4 = 0 * 8 + ko * 2 + kl;
    auto xy_in0  = in + idx_k4 * cst.inputSize * cst.batch + idx_m20;// [K/4, M, K4]
    auto xy_in1  = in + idx_k4 * cst.inputSize * cst.batch + idx_m21;// [K/4, M, K4]

    int idx_wk16 = 0 * 2 + kwl;

    int idx_n4 = (uz * 16 + no) < cst.outputDepthQuad ? (uz * 16 + no) : (cst.outputDepthQuad - 1);
    auto xy_wt = wt +  (idx_n4 * ((cst.inputDepthQuad+3)/4) + idx_wk16) * 4 + nl;// [N/4, K/16, N4, K4, K4]

    int idx_sa = (ko * 32 + ml * 2 + 0) * 2 + kl;
    int idx_sb = 1024 + (kwl * 16 + 0) * 64 + no * 4 + nl;
    int quadsPerBlock = (cst.inputDepthQuad + cst.blockCount - 1) / cst.blockCount;

    for (int bi=0; bi<cst.blockCount; ++bi) {
    #if defined(W_QUANT_4) || defined(W_QUANT_8)
        // [N/4, cst.blockCount, 2/*scale_bias*/, N4]
        FLOAT scale0 = FLOAT(dequantScale[((idx_n4 * cst.blockCount + bi) * 2 + 0) * 4 + nl]) / (FLOAT)cst.scaleCoef;
        FLOAT dequant_bias0 = FLOAT(dequantScale[((idx_n4 * cst.blockCount + bi) * 2 + 1) * 4 + nl]) / (FLOAT)cst.scaleCoef;

    #endif
        int zmin = bi * quadsPerBlock;
        int zmax = min(zmin + quadsPerBlock, cst.inputDepthQuad);

        for (int z = zmin; z < zmax; z += 8) {
            FLOAT4x4 w_dequant; // K4K4
            {


            #ifdef W_QUANT_4
                MNN::uchar4x2 w_int4 = xy_wt[z];

                auto temp = FLOAT4(uchar4(w_int4[0][0], w_int4[1][0], w_int4[2][0], w_int4[3][0]) >> 4);
                w_dequant[0][0] = temp[0];
                w_dequant[1][0] = temp[1];
                w_dequant[2][0] = temp[2];
                w_dequant[3][0] = temp[3];
                temp = FLOAT4(uchar4(w_int4[0][0], w_int4[1][0], w_int4[2][0], w_int4[3][0]) & 0x000F);
                w_dequant[0][1] = temp[0];
                w_dequant[1][1] = temp[1];
                w_dequant[2][1] = temp[2];
                w_dequant[3][1] = temp[3];
                temp = FLOAT4(uchar4(w_int4[0][1], w_int4[1][1], w_int4[2][1], w_int4[3][1]) >> 4);
                w_dequant[0][2] = temp[0];
                w_dequant[1][2] = temp[1];
                w_dequant[2][2] = temp[2];
                w_dequant[3][2] = temp[3];
                temp = FLOAT4(uchar4(w_int4[0][1], w_int4[1][1], w_int4[2][1], w_int4[3][1]) & 0x000F);
                w_dequant[0][3] = temp[0];
                w_dequant[1][3] = temp[1];
                w_dequant[2][3] = temp[2];
                w_dequant[3][3] = temp[3];

                FLOAT4 val = FLOAT4(dequant_bias0 - 8.0 * scale0);
                w_dequant = w_dequant * scale0 + FLOAT4x4(val, val, val, val);

            #elif defined(W_QUANT_8)
                auto w = xy_wt[z];
                FLOAT4x4 w_fp32 = FLOAT4x4(FLOAT4(w[0]), FLOAT4(w[1]), FLOAT4(w[2]), FLOAT4(w[3]));
                for (int i = 0; i < 4; ++i) {
                    w_dequant[i] = w_fp32[i] * scale0 + dequant_bias0;
                }
            #else
                auto w = xy_wt[z];
                w_dequant = FLOAT4x4((FLOAT4)w[0], (FLOAT4)w[1], (FLOAT4)w[2], (FLOAT4)w[3]);
            #endif
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            #pragma unroll(16)
            for (int i = 0; i < 16; ++i) {
                ((threadgroup ftype*)sdata)[idx_sb + 64*i]  = ftype(w_dequant[i/4][i%4]); // K4K4
            }

            #ifdef MNN_METAL_SRC_PROTECT
            if (idx_k4 + z < cst.inputDepthQuad) {
                ((threadgroup ftype4*)sdata)[idx_sa]     = (ftype4)*(xy_in0);
                ((threadgroup ftype4*)sdata)[idx_sa + 2] = (ftype4)*(xy_in1);
            } else {
                ((threadgroup ftype4*)sdata)[idx_sa]     = (ftype4)(0);
                ((threadgroup ftype4*)sdata)[idx_sa + 2] = (ftype4)(0);
            }
            #else
            ((threadgroup ftype4*)sdata)[idx_sa]     = (ftype4)*(xy_in0);
            ((threadgroup ftype4*)sdata)[idx_sa + 2] = (ftype4)*(xy_in1);
            #endif

            threadgroup_barrier(mem_flags::mem_threadgroup);

            /*
            A: [K4, M32, K8] -> [K4, M2, M16, K8]
            index: [ik, sgitg/2, sga[0~1]]

            B: [K4, K8, N64] -> [K4, K8, N2, N32]
            index: [ik, sgitg%2, sgb[0~3]]

            sgitg: compute M2 and N2
            */
            threadgroup ftype * sdata_a = (threadgroup ftype*)sdata + 16*8*(sgitg/2);
            threadgroup ftype * sdata_b = (threadgroup ftype*)sdata + 1024 + 32*(sgitg%2);

            #pragma unroll(4)
            for (short ik = 0; ik < 4; ik++) {
                simdgroup_load(sga[0], (const threadgroup ftype*)sdata_a + 256 * ik, 8);
                simdgroup_load(sga[1], ((const threadgroup ftype*)sdata_a) + 256 * ik + 64, 8);

                simdgroup_load(sgb[0], ((threadgroup ftype*)sdata_b) + 512 * ik + 0,  64);
                simdgroup_load(sgb[1], ((threadgroup ftype*)sdata_b) + 512 * ik + 8,  64);
                simdgroup_load(sgb[2], ((threadgroup ftype*)sdata_b) + 512 * ik + 16, 64);
                simdgroup_load(sgb[3], ((threadgroup ftype*)sdata_b) + 512 * ik + 24, 64);

                simdgroup_barrier(mem_flags::mem_none);
                SIMDGROUP_MATRIX_FMA(2, 4);

                simdgroup_barrier(mem_flags::mem_none);
            }

            xy_in0 += 8 * cst.inputSize * cst.batch;
            xy_in1 += 8 * cst.inputSize * cst.batch;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup FLOAT * sdata_c = (threadgroup FLOAT*)sdata + 512*sgitg;

    SIMDGROUP_MATRIX_STORE((threadgroup FLOAT*)sdata_c, 8);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // layout: [N/4, M, N4] -> [N/64, N4, N4, M/32, M2, M16, N4]
    // index : [uz, ko, N4, rx, kl, ml, N4]
    auto xy_out = out + ((uz * 4 + ko) * 4 + 0) * cst.outputSize * cst.batch + (rx * 2 + kl) * 16 + ml;// [N/4, M, N4]

    // sdata [M2, N2, N2, N2, M2, M8, N8]
    // index [kl, ko/2, ko%2, N2, ml/8, ml%8, N2, N4]
    if((rx * 32 + kl * 16 + ml) < cst.inputSize * cst.batch) {
        if((uz * 4 + ko) * 4 < cst.outputDepthQuad) {
            xy_out[0] =  activate(ftype4(((threadgroup FLOAT4*)sdata)[(((kl * 4 + ko) * 2 + 0) * 16 + ml) * 2] + FLOAT4(biasTerms[(uz * 4 + ko) * 4])), cst.activation);
        }
        if((uz * 4 + ko) * 4 + 1 < cst.outputDepthQuad) {
            xy_out[cst.outputSize * cst.batch] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(((kl * 4 + ko) * 2 + 0) * 16 + ml) * 2 + 1] + FLOAT4(biasTerms[(uz * 4 + ko) * 4 + 1])), cst.activation);
        }
        if((uz * 4 + ko) * 4 + 2 < cst.outputDepthQuad) {
            xy_out[cst.outputSize * cst.batch * 2] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(((kl * 4 + ko) * 2 + 1) * 16 + ml) * 2] + FLOAT4(biasTerms[(uz * 4 + ko) * 4 + 2])), cst.activation);
        }
        if((uz * 4 + ko) * 4 + 3 < cst.outputDepthQuad) {
            xy_out[cst.outputSize * cst.batch * 3] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(((kl * 4 + ko) * 2 + 1) * 16 + ml) * 2 + 1] + FLOAT4(biasTerms[(uz * 4 + ko) * 4 + 3])), cst.activation);
        }
    }
#endif
}


kernel void conv1x1_gemm_16x16_sg(const device ftype4 *in            [[buffer(0)]],
                            device ftype4 *out                 [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                            const device ftype4 *wt      [[buffer(3)]],
                            const device ftype4 *biasTerms     [[buffer(4)]],
                            uint3 gid                          [[threadgroup_position_in_grid]],
                            uint                  tiitg[[thread_index_in_threadgroup]],
                            uint                  sgitg[[simdgroup_index_in_threadgroup]]) {
    /*
     // Read:
     ftype 0~127   ---> input: [M16, K8]
     ftype 128~255 ---> input: [K8, N16]
     // Write:
     FLOAT 0~255 ---> input: [N2, M2, M8, N8]
     */
    threadgroup FLOAT4 sdata[64] = {0.f};

    INIT_SIMDGROUP_MATRIX(2, 2, 4);
    int rx = gid.x;// M/16
    int uz = gid.y;// N/16

    int kl = tiitg / 16;
    int rcl = tiitg % 16;

    // boundary limit
    int idx_n4 = (4 * uz + rcl / 4) < cst.outputDepthQuad ? (4 * uz + rcl / 4) : (cst.outputDepthQuad - 1);
    int idx_m  = (16 * rx + rcl) < cst.inputSize * cst.batch ? (16 * rx + rcl) : (cst.inputSize * cst.batch - 1);

    auto xy_wt = wt +  (idx_n4 * cst.inputDepthQuad + 0) * 4 + rcl % 4;// [N/4, K/4, N4, K4]
    auto xy_in0  = in + idx_m + cst.inputSize * cst.batch * kl;// [K/4, M, K4]
    auto xy_out = out + (4 * uz + 2 * kl) * cst.outputSize * cst.batch + idx_m;// [N/4, M, N4]

    for (int z = kl; z < cst.inputDepthQuad; z += 2) {
        ((threadgroup ftype4*)sdata)[2* rcl + kl] = (*xy_in0);
        xy_in0 += 2 * cst.inputSize * cst.batch;

        FLOAT4 w4 = FLOAT4(xy_wt[4 * z]); // [N/4, K/4, N4, K4]
        ((threadgroup ftype*)sdata)[128 + (kl * 4 + 0) * 16 + rcl] = ftype(w4[0]);
        ((threadgroup ftype*)sdata)[128 + (kl * 4 + 1) * 16 + rcl] = ftype(w4[1]);
        ((threadgroup ftype*)sdata)[128 + (kl * 4 + 2) * 16 + rcl] = ftype(w4[2]);
        ((threadgroup ftype*)sdata)[128 + (kl * 4 + 3) * 16 + rcl] = ftype(w4[3]);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_load(sga[0], (const threadgroup ftype*)sdata, 8);
        simdgroup_load(sga[1], ((const threadgroup ftype*)sdata) + 64, 8);
        simdgroup_load(sgb[0], ((const threadgroup ftype*)sdata) + 128, 16);
        simdgroup_load(sgb[1], ((const threadgroup ftype*)sdata) + 136, 16);

        SIMDGROUP_MATRIX_FMA(2, 2);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    SIMDGROUP_MATRIX_STORE((threadgroup FLOAT*)sdata, 4);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if((16 * rx + rcl) < cst.inputSize * cst.batch) {
        if((4 * uz + 2 * kl) < cst.outputDepthQuad) {
            xy_out[0] =  activate(ftype4(sdata[(kl * 16 + rcl) * 2 + 0] + FLOAT4(biasTerms[4 * uz + 2 * kl + 0])), cst.activation);
        }
        if((4 * uz + 2 * kl + 1) < cst.outputDepthQuad) {
            xy_out[cst.outputSize * cst.batch] = activate(ftype4(sdata[(kl * 16 + rcl) * 2 + 1] + FLOAT4(biasTerms[4 * uz + 2 * kl + 1])), cst.activation);
        }
    }
}


kernel void conv1x1_gemm_32x16_sg(const device ftype4 *in            [[buffer(0)]],
                            device ftype4 *out                 [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                            const device ftype4 *wt      [[buffer(3)]],
                            const device ftype4 *biasTerms     [[buffer(4)]],
                            uint3 gid                          [[threadgroup_position_in_grid]],
                            uint                  tiitg[[thread_index_in_threadgroup]],
                            uint                  sgitg[[simdgroup_index_in_threadgroup]]) {
    /*
     // Read:
     ftype 0~255   ---> input: [M32, K8]
     ftype 256~383 ---> input: [K8, N16]
     // Write:
     FLOAT 0~511 ---> input: [N2, M4, M8, N8]
     */
    threadgroup FLOAT4 sdata[128] = {0.f};

    INIT_SIMDGROUP_MATRIX(4, 2, 8);

    int rx = gid.x;// M/32
    int uz = gid.y;// N/16

    int kl = tiitg % 2;
    int rcl = tiitg / 2;

    const int size_m = cst.inputSize * cst.batch;

    // boundary limit
    int idx_n4 = (4 * uz + rcl / 4) < cst.outputDepthQuad ? (4 * uz + rcl / 4) : (cst.outputDepthQuad - 1);
    int idx_m0  = (16 * rx + rcl) <  size_m ? (16 * rx + rcl) : (size_m - 1);
    int idx_m1  = (16 * rx + rcl) + size_m / 2 < size_m ? (16 * rx + rcl) + size_m / 2: (size_m - 1);

    auto xy_wt = wt +  (idx_n4 * cst.inputDepthQuad + 0) * 4 + rcl % 4;// [N/4, K/4, N4, K4]
    auto xy_in0  = in + idx_m0 + cst.inputSize * cst.batch * kl;// [K/4, M2, M/2, K4]
    auto xy_in1  = in + idx_m1 + cst.inputSize * cst.batch * kl;// [K/4, M2, M/2, K4]

    auto xy_out0 = out + (4 * uz + 2 * kl) * cst.outputSize * cst.batch + idx_m0;// [N/4, M, N4]
    auto xy_out1 = out + (4 * uz + 2 * kl) * cst.outputSize * cst.batch + idx_m1;// [N/4, M, N4]

    for (int z = kl; z < cst.inputDepthQuad; z += 2) {
        ((threadgroup ftype4*)sdata)[2* rcl + kl] = *xy_in0;
        ((threadgroup ftype4*)sdata)[32 + 2* rcl + kl] = *xy_in1;

        FLOAT4 w4 = FLOAT4(xy_wt[4*z]); // [N/4, K/4, N4, K4]
        ((threadgroup ftype*)sdata)[256 + (kl * 4 + 0) * 16 + rcl] = ftype(w4[0]);
        ((threadgroup ftype*)sdata)[256 + (kl * 4 + 1) * 16 + rcl] = ftype(w4[1]);
        ((threadgroup ftype*)sdata)[256 + (kl * 4 + 2) * 16 + rcl] = ftype(w4[2]);
        ((threadgroup ftype*)sdata)[256 + (kl * 4 + 3) * 16 + rcl] = ftype(w4[3]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_load(sga[0], (const threadgroup ftype*)sdata, 8);
        simdgroup_load(sga[1], ((const threadgroup ftype*)sdata) + 64, 8);
        simdgroup_load(sga[2], ((const threadgroup ftype*)sdata) + 128, 8);
        simdgroup_load(sga[3], ((const threadgroup ftype*)sdata) + 192, 8);

        simdgroup_load(sgb[0], ((const threadgroup ftype*)sdata) + 256, 16);
        simdgroup_load(sgb[1], ((const threadgroup ftype*)sdata) + 264, 16);

        SIMDGROUP_MATRIX_FMA(4, 2);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        xy_in0 += 2 * cst.inputSize * cst.batch;
        xy_in1 += 2 * cst.inputSize * cst.batch;

    }

    SIMDGROUP_MATRIX_STORE((threadgroup FLOAT*)sdata, 8);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if((16 * rx + rcl) < size_m) {
        if((4 * uz + 2 * kl) < cst.outputDepthQuad) {
            xy_out0[0] =  activate(ftype4(sdata[(kl * 32 + rcl) * 2 + 0] + FLOAT4(biasTerms[4 * uz + 2 * kl + 0])), cst.activation);
        }
        if((4 * uz + 2 * kl + 1) < cst.outputDepthQuad) {
            xy_out0[cst.outputSize * cst.batch] = activate(ftype4(sdata[(kl * 32 + rcl) * 2 + 1] + FLOAT4(biasTerms[4 * uz + 2 * kl + 1])), cst.activation);
        }
    }
    if((16 * rx + rcl) + size_m / 2 < size_m) {
        if((4 * uz + 2 * kl) < cst.outputDepthQuad) {
            xy_out1[0] =  activate(ftype4(sdata[(kl * 32 + 16 + rcl) * 2 + 0] + FLOAT4(biasTerms[4 * uz + 2 * kl + 0])), cst.activation);
        }
        if((4 * uz + 2 * kl + 1) < cst.outputDepthQuad) {
            xy_out1[cst.outputSize * cst.batch] = activate(ftype4(sdata[(kl * 32 + 16 + rcl) * 2 + 1] + FLOAT4(biasTerms[4 * uz + 2 * kl + 1])), cst.activation);
        }
    }
}




//======================================================================
// Fused Q4/Q8 GEMM (tensor-API prefill path)
//
// Unpacks quantized weights in-kernel and applies per-block scale/bias
// from buffer(5) — no fp16 pre-dequant pass or extra device-memory round
// trip for the dequanted weight.
//
// Buffer contract:
//   buffer(0): input        [K/4, M, K4] fp16
//   buffer(1): output       [N/4, M, N4] fp16
//   buffer(2): conv1x1_constants
//   buffer(3): quantized weight (int4 packed as MNN::uchar4x2 OR
//              int8 packed as char4 depending on W_QUANT_4/8 macro)
//   buffer(4): biasTerms    [N/4]
//   buffer(5): dequantScale [N/4, blockCount, 2, N4]
//   buffer(6): unused placeholder (host binds mWeight to satisfy the
//              Metal validation layer)
//
// Weight int4 unpack: follows conv1x1_gemm_32x64_wquant_split_k_sg
// (the in-shader sg_matrix path that actually reads mWeight int4 layout).
// The int4 buffer memory layout is [N/4, K/4-quad, N4-inner, uchar2] where
// each uchar2 holds 4 K-nibbles for one (K/4-quad, N4-inner). Successive
// uchar2 in memory step by N4-inner (+1) or by K/4-quad (+4).
//
// IMPORTANT: mWeight is NOT `[N/4, K/16, N4, K4, K4]` (that's the fp16
// dequanted mTempWeight layout). The W_QUANT_4 branch in the fp16 tensor
// API kernel (conv1x1_gemm_32x64_split_k_sg) uses `MNN::uchar4x2 w = wt[z]`
// which reads 4 successive uchar2 slots — under mWeight's real layout these
// are 4 different N4-inners of the SAME K/4-quad, not 4 K/4-quads for the
// same N4-inner. That's why buffer(3) is always fed mTempWeight (fp16) in
// that kernel; the W_QUANT_4 branch is effectively dead.
//
// For a thread with (nl, kwl) reading K = 4*(kwl*4 + i) + j for i,j ∈ [0..3]:
//   uchar2 w = xy_wt_i4[(zQuad + kwl*4 + i) * 4]  // zQuad is K/4-quad index
//   w[0]>>4 -> K col 0 of row i,  w[0]&0xF -> col 1
//   w[1]>>4 -> K col 2 of row i,  w[1]&0xF -> col 3
// After unpack, apply `w * scale + (bias - 8*scale)` as in the split_k kernel.
//
// scale/bias per quant block:
//   scaleCoef is host-side compensation kept in cst.scaleCoef; the
//   physical dequantScale buffer stores s*coef, so we divide by coef.
//   Layout: [N/4, blockCount, 2, N4] indexed as
//     scale = dequantScale[((idx_n4 * blockCount + bi) * 2 + 0) * 4 + nl]
//     bias  = dequantScale[((idx_n4 * blockCount + bi) * 2 + 1) * 4 + nl]
//
// K_TILE = 8 per iter and quant block spans quadsPerBlock = ceil(ic_4 / blockCount)
// K-quads (each K-quad = 4 K along contiguous K4). This kernel is only
// selected when blockCount divides ic_4 cleanly and quadsPerBlock >= 2 K-quads,
// which holds for all Qwen3 W4-block32 shapes we care about.
//======================================================================
#if defined(W_QUANT_2) || defined(W_QUANT_3) || defined(W_QUANT_4) || defined(W_QUANT_8)
kernel void conv1x1_fused_q4_gemm_stage(
                            const device ftype4 *in            [[buffer(0)]],
                            device ftype4 *out                 [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                        #ifdef W_QUANT_2
                            const device uchar4 *wt_int2       [[buffer(3)]],
                        #elif defined(W_QUANT_3)
                            const device uchar *wt_int3        [[buffer(3)]],
                        #elif defined(W_QUANT_4)
                            const device MNN::uchar4x2 *wt_int4 [[buffer(3)]],
                        #else
                            const device MNN::char4x4  *wt_int8 [[buffer(3)]],
                        #endif
                            const device ftype4 *biasTerms     [[buffer(4)]],
                            const device ftype *dequantScale   [[buffer(5)]],
                            const device ftype4x4 *wt_fp       [[buffer(6)]],
                            uint3 gid                          [[threadgroup_position_in_grid]],
                            uint                  tiitg [[thread_index_in_threadgroup]],
                            uint                  tiisg [[thread_index_in_simdgroup]],
                            uint                  sgitg [[simdgroup_index_in_threadgroup]]) {
#ifdef USE_METAL_TENSOR_OPS
    threadgroup FLOAT4 sdata[800] = {0.f};

    const int K = 32, M = 32, N = 64;
    auto tI = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata, dextents<int32_t, 2>(K, M));            // [M, K]
    auto tW = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata + 1024, dextents<int32_t, 2>(K, N));     // [N, K]

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(M, N, K, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mmOps;

    auto cT = mmOps.get_destination_cooperative_tensor<decltype(tI), decltype(tW), FLOAT>();

    int rx = gid.x; // M/32
    int uz = gid.y; // N/64

    // Thread-role decomposition (same as baseline non-LOOP_K64):
    //   A: [8, 16]  kl=tiitg/16 (0..7)   ml=tiitg%16 (0..15)
    //   B: [16, 4, 2]  no=tiitg/8 (0..15) nl=(tiitg%8)/2 (0..3) kwl=(tiitg%8)%2 (0..1)
    //   C: [32, 4]  mlc=tiitg/4 (0..31) nlc=tiitg%4 (0..3)
    int kl  = tiitg / 16;
    int ml  = tiitg % 16;
    int no  = tiitg / 8;
    int sl  = tiitg % 8;
    int nl  = sl / 2;
    int kwl = sl % 2;
    int mlc = tiitg / 4;
    int nlc = tiitg % 4;

    // Input row-pair mapping (M2K4 per thread)
    int idx_m20 = (rx * 16 + ml) * 2 + 0 < cst.inputSize * cst.batch ? (rx * 16 + ml) * 2 + 0 : (cst.inputSize * cst.batch - 1);
    int idx_m21 = (rx * 16 + ml) * 2 + 1 < cst.inputSize * cst.batch ? (rx * 16 + ml) * 2 + 1 : (cst.inputSize * cst.batch - 1);

    int idx_k4 = 0 * 8 + kl;
    auto xy_in0 = in + idx_k4 * cst.inputSize * cst.batch + idx_m20; // [K/4, M, K4]
    auto xy_in1 = in + idx_k4 * cst.inputSize * cst.batch + idx_m21; // [K/4, M, K4]

    // Weight tile mapping.
    //   fp16 mTempWeight  layout: [N/4, K/16, N4, K4, K4]  (ftype4x4 slots)
    //   int2 mWeight (Q2) layout: [N/4, K/4-quad, N4, uchar]
    //   int3 mWeight (Q3) layout: [N/4, K/4-quad, 6 bytes] (shared tile across N4)
    //   int4 mWeight (Q4) layout: [N/4, K/4-quad, N4, uchar2]
    //   int8 mWeight (Q8) layout: [N/4, K/4-quad, N4, char4]
    int idx_n4   = (uz * 16 + no) < cst.outputDepthQuad ? (uz * 16 + no) : (cst.outputDepthQuad - 1);
    // Reinterpret buffer(3) as the packed-quant element type.
    // W2: uchar4 = 4 bytes per K/4-quad (one per N4 row), thread nl picks byte.
    // W3: uchar*  = 6 bytes per K/4-quad (shared tile), thread accesses tilePtr[nl].
    // W4: uchar2  = 2 bytes × 4 N4 rows per K/4-quad.
    // W8: char4   = 4 bytes × 4 N4 rows per K/4-quad.
#ifdef W_QUANT_2
    auto xy_wt_i2 = ((const device uchar*)wt_int2) + (idx_n4 * cst.inputDepthQuad) * 4 + nl;
#elif defined(W_QUANT_3)
    auto xy_wt_i3 = wt_int3 + idx_n4 * cst.inputDepthQuad * 6;
#elif defined(W_QUANT_4)
    auto xy_wt_i4 = ((const device uchar2*)wt_int4) + (idx_n4 * cst.inputDepthQuad + 0) * 4 + nl;
#else // W_QUANT_8
    auto xy_wt_i8 = ((const device char4*)wt_int8)  + (idx_n4 * cst.inputDepthQuad + 0) * 4 + nl;
#endif

    int idx_sa = (ml * 2 + 0) * 8 + kl;                                  // input write offset in sdata (ftype4 units)
    int idx_sb = 256 + ((no * 4 + nl) * 2 + kwl) * 4 + 0;                // weight write offset (ftype4 units, base = 1024/4 = 256)
    int quadsPerBlock = (cst.inputDepthQuad + cst.blockCount - 1) / cst.blockCount;

    for (int bi = 0; bi < cst.blockCount; ++bi) {
        // Per-block scale/bias (same as split_k_sg branch of gemm_32x64_split_k_sg).
        // Layout of dequantScale: [N/4, blockCount, 2, N4]. `nl` (0..3) picks the
        // N4-inner slot corresponding to this thread's OC row within the (uz*16+no)
        // group. The `/scaleCoef` divides out the host-side compensation.
        FLOAT scale0        = FLOAT(dequantScale[((idx_n4 * cst.blockCount + bi) * 2 + 0) * 4 + nl]) / (FLOAT)cst.scaleCoef;
        FLOAT dequant_bias0 = FLOAT(dequantScale[((idx_n4 * cst.blockCount + bi) * 2 + 1) * 4 + nl]) / (FLOAT)cst.scaleCoef;
        int zmin = bi * quadsPerBlock;
        int zmax = min(zmin + quadsPerBlock, cst.inputDepthQuad);

        for (int z = zmin; z < zmax; z += 8) {
            // ---- Phase 1: weight load (in-kernel quant unpack) ----
            //
            // Outer z counts K/4-quads; one iter per quant block since
            // quadsPerBlock == zmax-zmin K/4-quads and we jump by 8 (=1 block for the
            // Qwen3-0.6B W4-block32 shape). Inner unrolled i=0..3 reads 4
            // K/4-quads for THIS thread's (nl) N4-inner row, corresponding to
            // the kwl-half of the block (K positions 4*(kwl*4+i) .. 4*(kwl*4+i)+3).
            FLOAT4x4 w_dequant;
            {
#ifdef W_QUANT_2
                // W2: each uchar packs 4 x 2-bit values at bit pairs [7:6,5:4,3:2,1:0].
                // Unsigned range [0,3] centered at 1.5 → subtract 2.0 (mirrors conv1x1_w_dequant).
                #pragma unroll(4)
                for (int i = 0; i < 4; ++i) {
                    uchar b = xy_wt_i2[(z + kwl * 4 + i) * 4];
                    w_dequant[i][0] = FLOAT((b >> 6) & 3);
                    w_dequant[i][1] = FLOAT((b >> 4) & 3);
                    w_dequant[i][2] = FLOAT((b >> 2) & 3);
                    w_dequant[i][3] = FLOAT( b       & 3);
                }
                FLOAT4 val = FLOAT4(dequant_bias0 - 2.0 * scale0);
                w_dequant  = w_dequant * scale0 + FLOAT4x4(val, val, val, val);
#elif defined(W_QUANT_3)
                // W3: 6-byte tile per (4 OC, 4 IC) per K/4-quad.
                // Bytes 0..3: low 2 bits per OC row. Bytes 4..5: shared high-bit nibbles
                // (byte4 for OC 0,1; byte5 for OC 2,3; nl%2 picks nibble within byte).
                // Unsigned range [0,7] centered at 3.5 → subtract 4.0.
                #pragma unroll(4)
                for (int i = 0; i < 4; ++i) {
                    const device uchar* tilePtr = xy_wt_i3 + (z + kwl * 4 + i) * 6;
                    uchar lo = tilePtr[nl];
                    uchar h  = (nl < 2) ? tilePtr[4] : tilePtr[5];
                    uchar hShifted = (nl % 2 == 0) ? (h >> 4) : (h & 0xF);
                    w_dequant[i][0] = FLOAT(((lo >> 6) & 3) | (((hShifted >> 3) & 1) << 2));
                    w_dequant[i][1] = FLOAT(((lo >> 4) & 3) | (((hShifted >> 2) & 1) << 2));
                    w_dequant[i][2] = FLOAT(((lo >> 2) & 3) | (((hShifted >> 1) & 1) << 2));
                    w_dequant[i][3] = FLOAT(( lo       & 3) | (( hShifted       & 1) << 2));
                }
                FLOAT4 val = FLOAT4(dequant_bias0 - 4.0 * scale0);
                w_dequant  = w_dequant * scale0 + FLOAT4x4(val, val, val, val);
#elif defined(W_QUANT_4)
                // Mirror conv1x1_gemm_32x64_wquant_split_k_sg's inner unpack:
                //   uchar2 w = xy_wt_i4[k4QuadIndex * 4]
                //   -> w[0]>>4 (K col 0), w[0]&0xF (col 1),
                //      w[1]>>4 (col 2),   w[1]&0xF (col 3)
                // k4QuadIndex for row i = z + kwl*4 + i.
                #pragma unroll(4)
                for (int i = 0; i < 4; ++i) {
                    uchar2 w = xy_wt_i4[(z + kwl * 4 + i) * 4];
                    w_dequant[i][0] = FLOAT(w[0] >> 4);
                    w_dequant[i][1] = FLOAT(w[0] & 0x0F);
                    w_dequant[i][2] = FLOAT(w[1] >> 4);
                    w_dequant[i][3] = FLOAT(w[1] & 0x0F);
                }
                FLOAT4 val = FLOAT4(dequant_bias0 - 8.0 * scale0);
                w_dequant  = w_dequant * scale0 + FLOAT4x4(val, val, val, val);
#else // W_QUANT_8
                // Q8: each char4 gives 4 signed int8 K values for one row.
                // scale/bias directly applied (no -8 offset unlike Q4 which is
                // unsigned nibble minus 8). Follows the split_k Q8 branch.
                #pragma unroll(4)
                for (int i = 0; i < 4; ++i) {
                    char4 w = xy_wt_i8[(z + kwl * 4 + i) * 4];
                    FLOAT4 w4 = FLOAT4(FLOAT(w[0]), FLOAT(w[1]), FLOAT(w[2]), FLOAT(w[3]));
                    w_dequant[i] = w4 * scale0 + dequant_bias0;
                }
#endif
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            #pragma unroll(4)
            for (int i = 0; i < 4; ++i) {
                ((threadgroup ftype4*)sdata)[idx_sb + i] = ftype4(w_dequant[i]);
            }

            // ---- Load input tile (with SRC_PROTECT boundary handling) ----
            #ifdef MNN_METAL_SRC_PROTECT
            if (idx_k4 + z < cst.inputDepthQuad) {
                ((threadgroup ftype4*)sdata)[idx_sa]     = (ftype4)*(xy_in0);
                ((threadgroup ftype4*)sdata)[idx_sa + 8] = (ftype4)*(xy_in1);
            } else {
                ((threadgroup ftype4*)sdata)[idx_sa]     = (ftype4)(0);
                ((threadgroup ftype4*)sdata)[idx_sa + 8] = (ftype4)(0);
            }
            #else
            ((threadgroup ftype4*)sdata)[idx_sa]     = (ftype4)*(xy_in0);
            ((threadgroup ftype4*)sdata)[idx_sa + 8] = (ftype4)*(xy_in1);
            #endif

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ---- Phase 2: tensor API matmul ----
            auto sA = tI.slice(0, 0);
            auto sB = tW.slice(0, 0);
            mmOps.run(sA, sB, cT);

            xy_in0 += 8 * cst.inputSize * cst.batch;
            xy_in1 += 8 * cst.inputSize * cst.batch;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto tC = tensor<threadgroup FLOAT, dextents<int32_t, 2>, tensor_inline>((threadgroup FLOAT*)sdata, dextents<int32_t, 2>(N, M));
    cT.store(tC);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Epilogue: write [N/4, M, N4] layout with bias + activation.
    // sdata after cT.store: [M32, N64] -> [M32, N4, N16]
    // per-thread contribution: N16 at [mlc, nlc, N16]
    auto xy_out = out + ((uz * 4 + nlc) * 4 + 0) * cst.outputSize * cst.batch + (rx * 32 + mlc);
    if ((rx * 32 + mlc) < cst.inputSize * cst.batch) {
        if ((uz * 4 + nlc) * 4 < cst.outputDepthQuad) {
            xy_out[0] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(mlc * 4 + nlc) * 4 + 0] + FLOAT4(biasTerms[(uz * 4 + nlc) * 4])), cst.activation);
        }
        if ((uz * 4 + nlc) * 4 + 1 < cst.outputDepthQuad) {
            xy_out[cst.outputSize * cst.batch] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(mlc * 4 + nlc) * 4 + 1] + FLOAT4(biasTerms[(uz * 4 + nlc) * 4 + 1])), cst.activation);
        }
        if ((uz * 4 + nlc) * 4 + 2 < cst.outputDepthQuad) {
            xy_out[cst.outputSize * cst.batch * 2] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(mlc * 4 + nlc) * 4 + 2] + FLOAT4(biasTerms[(uz * 4 + nlc) * 4 + 2])), cst.activation);
        }
        if ((uz * 4 + nlc) * 4 + 3 < cst.outputDepthQuad) {
            xy_out[cst.outputSize * cst.batch * 3] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(mlc * 4 + nlc) * 4 + 3] + FLOAT4(biasTerms[(uz * 4 + nlc) * 4 + 3])), cst.activation);
        }
    }
#endif // USE_METAL_TENSOR_OPS
}
#endif // W_QUANT_2 || W_QUANT_3 || W_QUANT_4 || W_QUANT_8



// conv1x1_fused_q4_gemm_stage_m8: M8-native tile for decode-verify (M=8), removing the
// 32-row tile's padding waste. Q4-only; same staging as the 32-row kernel, smaller A tile/C store.
#if defined(W_QUANT_4) && defined(FUSED_Q4_REAL_UNPACK)
kernel void conv1x1_fused_q4_gemm_stage_m8(
                            const device ftype4 *in            [[buffer(0)]],
                            device ftype4 *out                 [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                            const device MNN::uchar4x2 *wt_int4 [[buffer(3)]],
                            const device ftype4 *biasTerms     [[buffer(4)]],
                            const device ftype *dequantScale   [[buffer(5)]],
                            const device ftype4x4 *wt_fp       [[buffer(6)]],
                            uint3 gid                          [[threadgroup_position_in_grid]],
                            uint                  tiitg [[thread_index_in_threadgroup]],
                            uint                  tiisg [[thread_index_in_simdgroup]],
                            uint                  sgitg [[simdgroup_index_in_threadgroup]]) {
#ifdef USE_METAL_TENSOR_OPS
    threadgroup FLOAT4 sdata[800] = {0.f};

    const int K = 32, M = 8, N = 64;
    auto tI = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata, dextents<int32_t, 2>(K, M));
    auto tW = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata + 1024, dextents<int32_t, 2>(K, N));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(M, N, K, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mmOps;

    auto cT = mmOps.get_destination_cooperative_tensor<decltype(tI), decltype(tW), FLOAT>();

    int rx = gid.x;
    int uz = gid.y;

    int idx_n4 = (uz * 16 + tiitg / 8) < cst.outputDepthQuad ? (uz * 16 + tiitg / 8) : (cst.outputDepthQuad - 1);
    int quadsPerBlock = (cst.inputDepthQuad + cst.blockCount - 1) / cst.blockCount;

    // A tile: 8 rows x 8 K4-quads, one ftype4 per thread on the first 64 threads.
    int aml = (int)tiitg / 8;
    int akl = (int)tiitg % 8;
    bool aAct = aml < M;
    int am = rx * M + aml;
    int amc = am < cst.inputSize * cst.batch ? am : (cst.inputSize * cst.batch - 1);

    // v2 register prefetch (identical mapping to the 32-row kernel)
    uint2 w2pre;
    {
        const device uint2 *wq = (const device uint2 *)wt_int4;
        int no0 = (int)tiitg / 8;
        int kw0 = ((int)tiitg % 8) / 4;
        int i0  = (int)tiitg % 4;
        int n4_0 = (uz * 16 + no0) < cst.outputDepthQuad ? (uz * 16 + no0) : (cst.outputDepthQuad - 1);
        w2pre = wq[n4_0 * cst.inputDepthQuad + (kw0 * 4 + i0)];
    }
    for (int bi = 0; bi < cst.blockCount; ++bi) {
        int zmin = bi * quadsPerBlock;
        int zmax = min(zmin + quadsPerBlock, cst.inputDepthQuad);

        ftype4 sc4t_h = *(const device ftype4 *)(dequantScale + ((idx_n4 * cst.blockCount + bi) * 2 + 0) * 4);
        ftype4 db4t_h = *(const device ftype4 *)(dequantScale + ((idx_n4 * cst.blockCount + bi) * 2 + 1) * 4);
        FLOAT4 sc4_h = FLOAT4(sc4t_h) / (FLOAT)cst.scaleCoef;
        FLOAT4 db4_h = FLOAT4(db4t_h) / (FLOAT)cst.scaleCoef;

        for (int z = zmin; z < zmax; z += 8) {
            FLOAT4 wb[4];
            {
                const device uint2 *wq = (const device uint2 *)wt_int4;
                int no = (int)tiitg / 8;
                int kw = ((int)tiitg % 8) / 4;
                int i  = (int)tiitg % 4;
                int idx_n4c = (uz * 16 + no) < cst.outputDepthQuad ? (uz * 16 + no) : (cst.outputDepthQuad - 1);
                uint2 w2 = w2pre;
                w2pre = (z + 8 < cst.inputDepthQuad) ? wq[idx_n4c * cst.inputDepthQuad + (z + 8 + kw * 4 + i)] : w2pre;
                uchar4 b0 = as_type<uchar4>(w2.x);
                uchar4 b1 = as_type<uchar4>(w2.y);
                FLOAT4 sc4 = sc4_h, db4 = db4_h;
                FLOAT4 v4  = db4 - 8.0f * sc4;
                #pragma unroll(4)
                for (int j = 0; j < 4; ++j) {
                    uchar wx = (j < 2) ? (j == 0 ? b0.x : b0.z) : (j == 2 ? b1.x : b1.z);
                    uchar wy = (j < 2) ? (j == 0 ? b0.y : b0.w) : (j == 2 ? b1.y : b1.w);
                    FLOAT4 row = FLOAT4(FLOAT(wx >> 4), FLOAT(wx & 0x0F), FLOAT(wy >> 4), FLOAT(wy & 0x0F));
                    wb[j] = row * sc4[j] + v4[j];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            {
                int no = (int)tiitg / 8;
                int kw = ((int)tiitg % 8) / 4;
                #pragma unroll(4)
                for (int j = 0; j < 4; ++j) {
                    ((threadgroup ftype4*)sdata)[256 + ((no * 4 + j) * 2 + kw) * 4 + ((int)tiitg % 4)] = ftype4(wb[j]);
                }
            }
            if (aAct) {
                #ifdef MNN_METAL_SRC_PROTECT
                if (akl + z < cst.inputDepthQuad) {
                    ((threadgroup ftype4*)sdata)[aml * 8 + akl] = (ftype4)in[(z + akl) * cst.inputSize * cst.batch + amc];
                } else {
                    ((threadgroup ftype4*)sdata)[aml * 8 + akl] = (ftype4)(0);
                }
                #else
                ((threadgroup ftype4*)sdata)[aml * 8 + akl] = (ftype4)in[(z + akl) * cst.inputSize * cst.batch + amc];
                #endif
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            auto sA = tI.slice(0, 0);
            auto sB = tW.slice(0, 0);
            mmOps.run(sA, sB, cT);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto tC = tensor<threadgroup FLOAT, dextents<int32_t, 2>, tensor_inline>((threadgroup FLOAT*)sdata, dextents<int32_t, 2>(N, M));
    cT.store(tC);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Epilogue: [N/4, M, N4] with bias + activation. sdata: FLOAT [M8][N64].
    for (int e = (int)tiitg; e < M * 16; e += 128) {
        int m = e / 16, n4l = e % 16;
        int gm = rx * M + m;
        if (gm >= cst.inputSize * cst.batch) continue;
        int n4 = uz * 16 + n4l;
        if (n4 >= cst.outputDepthQuad) continue;
        FLOAT4 v = FLOAT4(((threadgroup FLOAT*)sdata)[m * 64 + n4l * 4 + 0],
                          ((threadgroup FLOAT*)sdata)[m * 64 + n4l * 4 + 1],
                          ((threadgroup FLOAT*)sdata)[m * 64 + n4l * 4 + 2],
                          ((threadgroup FLOAT*)sdata)[m * 64 + n4l * 4 + 3]);
        out[n4 * cst.outputSize * cst.batch + gm] = activate(ftype4(v + FLOAT4(biasTerms[n4])), cst.activation);
    }
#endif // USE_METAL_TENSOR_OPS
}
#endif // W_QUANT_4 && FUSED_Q4_REAL_UNPACK

// conv1x1_fused_q4_gemm_stage_ksplit: K-split x4 across gid.z for TG-starved decode-verify shapes;
// fp32 partials, reduce pass adds bias/activation. NOT bit-exact vs single pass (summation order).
#if defined(W_QUANT_4) && defined(FUSED_Q4_REAL_UNPACK)
#define FUSED_KSPLIT_KS 4
kernel void conv1x1_fused_q4_gemm_stage_ksplit(
                            const device ftype4 *in            [[buffer(0)]],
                            device float4 *partialOut          [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                            const device MNN::uchar4x2 *wt_int4 [[buffer(3)]],
                            const device ftype4 *biasTerms     [[buffer(4)]],
                            const device ftype *dequantScale   [[buffer(5)]],
                            const device ftype4x4 *wt_fp       [[buffer(6)]],
                            uint3 gid                          [[threadgroup_position_in_grid]],
                            uint                  tiitg [[thread_index_in_threadgroup]],
                            uint                  tiisg [[thread_index_in_simdgroup]],
                            uint                  sgitg [[simdgroup_index_in_threadgroup]]) {
#ifdef USE_METAL_TENSOR_OPS
    threadgroup FLOAT4 sdata[800] = {0.f};

    const int K = 32, M = 32, N = 64;
    auto tI = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata, dextents<int32_t, 2>(K, M));
    auto tW = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata + 1024, dextents<int32_t, 2>(K, N));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(M, N, K, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mmOps;

    auto cT = mmOps.get_destination_cooperative_tensor<decltype(tI), decltype(tW), FLOAT>();

    int rx = gid.x;
    int uz = gid.y;
    int ks = gid.z;   // K partition index (0..KS-1)

    int kl  = tiitg / 16;
    int ml  = tiitg % 16;
    int nl  = (tiitg % 8) / 2;
    int mlc = tiitg / 4;
    int nlc = tiitg % 4;

    int idx_m20 = (rx * 16 + ml) * 2 + 0 < cst.inputSize * cst.batch ? (rx * 16 + ml) * 2 + 0 : (cst.inputSize * cst.batch - 1);
    int idx_m21 = (rx * 16 + ml) * 2 + 1 < cst.inputSize * cst.batch ? (rx * 16 + ml) * 2 + 1 : (cst.inputSize * cst.batch - 1);

    int idx_n4   = (uz * 16 + tiitg / 8) < cst.outputDepthQuad ? (uz * 16 + tiitg / 8) : (cst.outputDepthQuad - 1);
    int idx_sa   = (ml * 2 + 0) * 8 + kl;
    int quadsPerBlock = (cst.inputDepthQuad + cst.blockCount - 1) / cst.blockCount;

    // quant-block partition: scale hoist semantics preserved (whole blocks per part)
    int nsub = (cst.blockCount + FUSED_KSPLIT_KS - 1) / FUSED_KSPLIT_KS;
    int bi0  = ks * nsub;
    int bi1  = min(bi0 + nsub, cst.blockCount);
    int zpart_end = min(bi1 * quadsPerBlock, cst.inputDepthQuad);

    int idx_k4 = 0 * 8 + kl;
    auto xy_in0 = in + (bi0 * quadsPerBlock + idx_k4) * cst.inputSize * cst.batch + idx_m20;
    auto xy_in1 = in + (bi0 * quadsPerBlock + idx_k4) * cst.inputSize * cst.batch + idx_m21;

    // v2 register prefetch (first quad of this partition)
    uint2 w2pre;
    {
        const device uint2 *wq = (const device uint2 *)wt_int4;
        int no0 = (int)tiitg / 8;
        int kw0 = ((int)tiitg % 8) / 4;
        int i0  = (int)tiitg % 4;
        int n4_0 = (uz * 16 + no0) < cst.outputDepthQuad ? (uz * 16 + no0) : (cst.outputDepthQuad - 1);
        int z0 = min(bi0 * quadsPerBlock + kw0 * 4 + i0, cst.inputDepthQuad - 1);
        w2pre = wq[n4_0 * cst.inputDepthQuad + z0];
    }
    for (int bi = bi0; bi < bi1; ++bi) {
        int zmin = bi * quadsPerBlock;
        int zmax = min(zmin + quadsPerBlock, cst.inputDepthQuad);

        // scale hoist (identical to single-pass kernel)
        ftype4 sc4t_h = *(const device ftype4 *)(dequantScale + ((idx_n4 * cst.blockCount + bi) * 2 + 0) * 4);
        ftype4 db4t_h = *(const device ftype4 *)(dequantScale + ((idx_n4 * cst.blockCount + bi) * 2 + 1) * 4);
        FLOAT4 sc4_h = FLOAT4(sc4t_h) / (FLOAT)cst.scaleCoef;
        FLOAT4 db4_h = FLOAT4(db4t_h) / (FLOAT)cst.scaleCoef;

        for (int z = zmin; z < zmax; z += 8) {
            FLOAT4 wb[4];
            {
                const device uint2 *wq = (const device uint2 *)wt_int4;
                int no = (int)tiitg / 8;
                int kw = ((int)tiitg % 8) / 4;
                int i  = (int)tiitg % 4;
                int idx_n4c = (uz * 16 + no) < cst.outputDepthQuad ? (uz * 16 + no) : (cst.outputDepthQuad - 1);
                uint2 w2 = w2pre;
                int znext = z + 8 + kw * 4 + i;
                w2pre = (z + 8 < zpart_end) ? wq[idx_n4c * cst.inputDepthQuad + min(znext, cst.inputDepthQuad - 1)] : w2pre;
                uchar4 b0 = as_type<uchar4>(w2.x);
                uchar4 b1 = as_type<uchar4>(w2.y);
                FLOAT4 sc4 = sc4_h, db4 = db4_h;
                FLOAT4 v4  = db4 - 8.0f * sc4;
                #pragma unroll(4)
                for (int j = 0; j < 4; ++j) {
                    uchar wx = (j < 2) ? (j == 0 ? b0.x : b0.z) : (j == 2 ? b1.x : b1.z);
                    uchar wy = (j < 2) ? (j == 0 ? b0.y : b0.w) : (j == 2 ? b1.y : b1.w);
                    FLOAT4 row = FLOAT4(FLOAT(wx >> 4), FLOAT(wx & 0x0F), FLOAT(wy >> 4), FLOAT(wy & 0x0F));
                    wb[j] = row * sc4[j] + v4[j];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            {
                int no = (int)tiitg / 8;
                int kw = ((int)tiitg % 8) / 4;
                #pragma unroll(4)
                for (int j = 0; j < 4; ++j) {
                    ((threadgroup ftype4*)sdata)[256 + ((no * 4 + j) * 2 + kw) * 4 + ((int)tiitg % 4)] = ftype4(wb[j]);
                }
            }
            #ifdef MNN_METAL_SRC_PROTECT
            if (idx_k4 + z < cst.inputDepthQuad) {
                ((threadgroup ftype4*)sdata)[idx_sa]     = (ftype4)*(xy_in0);
                ((threadgroup ftype4*)sdata)[idx_sa + 8] = (ftype4)*(xy_in1);
            } else {
                ((threadgroup ftype4*)sdata)[idx_sa]     = (ftype4)(0);
                ((threadgroup ftype4*)sdata)[idx_sa + 8] = (ftype4)(0);
            }
            #else
            ((threadgroup ftype4*)sdata)[idx_sa]     = (ftype4)*(xy_in0);
            ((threadgroup ftype4*)sdata)[idx_sa + 8] = (ftype4)*(xy_in1);
            #endif

            threadgroup_barrier(mem_flags::mem_threadgroup);

            auto sA = tI.slice(0, 0);
            auto sB = tW.slice(0, 0);
            mmOps.run(sA, sB, cT);

            xy_in0 += 8 * cst.inputSize * cst.batch;
            xy_in1 += 8 * cst.inputSize * cst.batch;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto tC = tensor<threadgroup FLOAT, dextents<int32_t, 2>, tensor_inline>((threadgroup FLOAT*)sdata, dextents<int32_t, 2>(N, M));
    cT.store(tC);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // fp32 partials, no bias/activation (applied in reduce pass)
    int m = rx * 32 + mlc;
    if (m < cst.inputSize * cst.batch) {
        for (int r = 0; r < 4; ++r) {
            int n4 = (uz * 4 + nlc) * 4 + r;
            if (n4 < cst.outputDepthQuad) {
                partialOut[((size_t)ks * cst.outputDepthQuad + n4) * cst.outputSize * cst.batch + m] =
                    float4(((threadgroup FLOAT4*)sdata)[(mlc * 4 + nlc) * 4 + r]);
            }
        }
    }
#endif // USE_METAL_TENSOR_OPS
}


// K-split x4 with the M8-native tile: same partitioning/partials/reduce as _ksplit,
// minus the 32-row tile's padding rows. Routed for TG-starved shapes when area <= 8.
kernel void conv1x1_fused_q4_gemm_stage_ksplit_m8(
                            const device ftype4 *in            [[buffer(0)]],
                            device float4 *partialOut          [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                            const device MNN::uchar4x2 *wt_int4 [[buffer(3)]],
                            const device ftype4 *biasTerms     [[buffer(4)]],
                            const device ftype *dequantScale   [[buffer(5)]],
                            const device ftype4x4 *wt_fp       [[buffer(6)]],
                            uint3 gid                          [[threadgroup_position_in_grid]],
                            uint                  tiitg [[thread_index_in_threadgroup]],
                            uint                  tiisg [[thread_index_in_simdgroup]],
                            uint                  sgitg [[simdgroup_index_in_threadgroup]]) {
#ifdef USE_METAL_TENSOR_OPS
    threadgroup FLOAT4 sdata[800] = {0.f};

    const int K = 32, M = 8, N = 64;
    auto tI = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata, dextents<int32_t, 2>(K, M));
    auto tW = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata + 1024, dextents<int32_t, 2>(K, N));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(M, N, K, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mmOps;

    auto cT = mmOps.get_destination_cooperative_tensor<decltype(tI), decltype(tW), FLOAT>();

    int rx = gid.x;
    int uz = gid.y;
    int ks = gid.z;

    int idx_n4 = (uz * 16 + tiitg / 8) < cst.outputDepthQuad ? (uz * 16 + tiitg / 8) : (cst.outputDepthQuad - 1);
    int quadsPerBlock = (cst.inputDepthQuad + cst.blockCount - 1) / cst.blockCount;

    int nsub = (cst.blockCount + FUSED_KSPLIT_KS - 1) / FUSED_KSPLIT_KS;
    int bi0  = ks * nsub;
    int bi1  = min(bi0 + nsub, cst.blockCount);
    int zpart_end = min(bi1 * quadsPerBlock, cst.inputDepthQuad);

    int aml = (int)tiitg / 8;
    int akl = (int)tiitg % 8;
    bool aAct = aml < M;
    int am = rx * M + aml;
    int amc = am < cst.inputSize * cst.batch ? am : (cst.inputSize * cst.batch - 1);

    uint2 w2pre;
    {
        const device uint2 *wq = (const device uint2 *)wt_int4;
        int no0 = (int)tiitg / 8;
        int kw0 = ((int)tiitg % 8) / 4;
        int i0  = (int)tiitg % 4;
        int n4_0 = (uz * 16 + no0) < cst.outputDepthQuad ? (uz * 16 + no0) : (cst.outputDepthQuad - 1);
        int z0 = min(bi0 * quadsPerBlock + kw0 * 4 + i0, cst.inputDepthQuad - 1);
        w2pre = wq[n4_0 * cst.inputDepthQuad + z0];
    }
    for (int bi = bi0; bi < bi1; ++bi) {
        int zmin = bi * quadsPerBlock;
        int zmax = min(zmin + quadsPerBlock, cst.inputDepthQuad);

        ftype4 sc4t_h = *(const device ftype4 *)(dequantScale + ((idx_n4 * cst.blockCount + bi) * 2 + 0) * 4);
        ftype4 db4t_h = *(const device ftype4 *)(dequantScale + ((idx_n4 * cst.blockCount + bi) * 2 + 1) * 4);
        FLOAT4 sc4_h = FLOAT4(sc4t_h) / (FLOAT)cst.scaleCoef;
        FLOAT4 db4_h = FLOAT4(db4t_h) / (FLOAT)cst.scaleCoef;

        for (int z = zmin; z < zmax; z += 8) {
            FLOAT4 wb[4];
            {
                const device uint2 *wq = (const device uint2 *)wt_int4;
                int no = (int)tiitg / 8;
                int kw = ((int)tiitg % 8) / 4;
                int i  = (int)tiitg % 4;
                int idx_n4c = (uz * 16 + no) < cst.outputDepthQuad ? (uz * 16 + no) : (cst.outputDepthQuad - 1);
                uint2 w2 = w2pre;
                int znext = z + 8 + kw * 4 + i;
                w2pre = (z + 8 < zpart_end) ? wq[idx_n4c * cst.inputDepthQuad + min(znext, cst.inputDepthQuad - 1)] : w2pre;
                uchar4 b0 = as_type<uchar4>(w2.x);
                uchar4 b1 = as_type<uchar4>(w2.y);
                FLOAT4 sc4 = sc4_h, db4 = db4_h;
                FLOAT4 v4  = db4 - 8.0f * sc4;
                #pragma unroll(4)
                for (int j = 0; j < 4; ++j) {
                    uchar wx = (j < 2) ? (j == 0 ? b0.x : b0.z) : (j == 2 ? b1.x : b1.z);
                    uchar wy = (j < 2) ? (j == 0 ? b0.y : b0.w) : (j == 2 ? b1.y : b1.w);
                    FLOAT4 row = FLOAT4(FLOAT(wx >> 4), FLOAT(wx & 0x0F), FLOAT(wy >> 4), FLOAT(wy & 0x0F));
                    wb[j] = row * sc4[j] + v4[j];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            {
                int no = (int)tiitg / 8;
                int kw = ((int)tiitg % 8) / 4;
                #pragma unroll(4)
                for (int j = 0; j < 4; ++j) {
                    ((threadgroup ftype4*)sdata)[256 + ((no * 4 + j) * 2 + kw) * 4 + ((int)tiitg % 4)] = ftype4(wb[j]);
                }
            }
            if (aAct) {
                #ifdef MNN_METAL_SRC_PROTECT
                if (akl + z < cst.inputDepthQuad) {
                    ((threadgroup ftype4*)sdata)[aml * 8 + akl] = (ftype4)in[(z + akl) * cst.inputSize * cst.batch + amc];
                } else {
                    ((threadgroup ftype4*)sdata)[aml * 8 + akl] = (ftype4)(0);
                }
                #else
                ((threadgroup ftype4*)sdata)[aml * 8 + akl] = (ftype4)in[(z + akl) * cst.inputSize * cst.batch + amc];
                #endif
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            auto sA = tI.slice(0, 0);
            auto sB = tW.slice(0, 0);
            mmOps.run(sA, sB, cT);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto tC = tensor<threadgroup FLOAT, dextents<int32_t, 2>, tensor_inline>((threadgroup FLOAT*)sdata, dextents<int32_t, 2>(N, M));
    cT.store(tC);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int e = (int)tiitg; e < M * 16; e += 128) {
        int m = e / 16, n4l = e % 16;
        int gm = rx * M + m;
        if (gm >= cst.inputSize * cst.batch) continue;
        int n4 = uz * 16 + n4l;
        if (n4 >= cst.outputDepthQuad) continue;
        FLOAT4 v = FLOAT4(((threadgroup FLOAT*)sdata)[m * 64 + n4l * 4 + 0],
                          ((threadgroup FLOAT*)sdata)[m * 64 + n4l * 4 + 1],
                          ((threadgroup FLOAT*)sdata)[m * 64 + n4l * 4 + 2],
                          ((threadgroup FLOAT*)sdata)[m * 64 + n4l * 4 + 3]);
        partialOut[((size_t)ks * cst.outputDepthQuad + n4) * cst.outputSize * cst.batch + gm] = float4(v);
    }
#endif // USE_METAL_TENSOR_OPS
}

kernel void conv1x1_fused_q4_ksplit_reduce(
                            const device float4 *partialIn     [[buffer(0)]],
                            device ftype4 *out                 [[buffer(1)]],
                            const device ftype4 *biasTerms     [[buffer(2)]],
                            constant conv1x1_constants& cst    [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    int total = cst.outputDepthQuad * cst.outputSize * cst.batch;
    if ((int)gid >= total) return;
    int n4 = (int)gid / (cst.outputSize * cst.batch);
    float4 s = float4(0.f);
    for (int k = 0; k < FUSED_KSPLIT_KS; ++k) {
        s += partialIn[(size_t)k * total + gid];
    }
    out[gid] = activate(ftype4(FLOAT4(s) + FLOAT4(biasTerms[n4])), cst.activation);
}
#endif // W_QUANT_4 && FUSED_Q4_REAL_UNPACK

)metal";

//======================================================================
// conv1x1_fused_q4_gemm_stage_m64 — M=64 tile variant (P0 M_TILE=64)
//----------------------------------------------------------------------
// Same fused-Q4 GEMM as conv1x1_fused_q4_gemm_stage but with M=64 tile
// (vs baseline M=32). Halves grid.x for prefill, so each threadgroup
// consumes the same K-cost (weight reads = N*K, dequant work) but produces
// 2x the M output — cuts weight-read redundancy across TGs in half.
// This is the M=64, N=64 variant (N unchanged for lower risk).
//
// Threadgroup memory (16 KB max used):
//   sA:  [M=64, K=32] fp16 = 2048 half = 4096 B  (ftype offsets 0..2047)
//   sB:  [N=64, K=32] fp16 = 2048 half = 4096 B  (ftype offsets 2048..4095)
//   cT.store: [N=64, M=64] float = 4096 float = 16384 B (overwrites sA/sB
//                                                        after barrier)
//   Total sdata: 1024 FLOAT4 = 16 KB.
//
// Thread roles (128 threads = 4 simdgroups):
//   Input load:  ml = tiitg%16 (0..15) → 4 M rows (4*ml + 0..3)
//                kl = tiitg/16 (0..7)  → K column (K4 chunk)
//                Each thread: 4 rows × K4 = 16 fp16 (vs 2 rows in M=32).
//   Weight load: no = tiitg/8, nl = (tiitg%8)/2, kwl = (tiitg%8)%2
//                (unchanged from M=32).
//   Epilogue:    mlc = tiitg/4 (0..31), nlc = tiitg%4 (0..3)
//                mm iter 0..1 → m_idx = mm*32+mlc covers M=0..63.
//======================================================================
static const char* gConv1x1WfpSgMatrixM64 = R"metal(
#if defined(W_QUANT_2) || defined(W_QUANT_3) || defined(W_QUANT_4) || defined(W_QUANT_8)
kernel void conv1x1_fused_q4_gemm_stage_m64(
                            const device ftype4 *in            [[buffer(0)]],
                            device ftype4 *out                 [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                        #ifdef W_QUANT_2
                            const device uchar4 *wt_int2       [[buffer(3)]],
                        #elif defined(W_QUANT_3)
                            const device uchar *wt_int3        [[buffer(3)]],
                        #elif defined(W_QUANT_4)
                            const device MNN::uchar4x2 *wt_int4 [[buffer(3)]],
                        #else
                            const device MNN::char4x4  *wt_int8 [[buffer(3)]],
                        #endif
                            const device ftype4 *biasTerms     [[buffer(4)]],
                            const device ftype *dequantScale   [[buffer(5)]],
                            const device ftype4x4 *wt_fp       [[buffer(6)]],
#ifdef FQ4_GATEUP_DUAL
                        #ifdef W_QUANT_2
                            const device uchar4 *wt2_int2      [[buffer(7)]],
                        #elif defined(W_QUANT_3)
                            const device uchar *wt2_int3       [[buffer(7)]],
                        #elif defined(W_QUANT_4)
                            const device MNN::uchar4x2 *wt2_int4 [[buffer(7)]],
                        #else
                            const device MNN::char4x4  *wt2_int8 [[buffer(7)]],
                        #endif
                            const device ftype4 *biasTerms2    [[buffer(8)]],
                            const device ftype *dequantScale2  [[buffer(9)]],
                            constant float &scaleCoef2         [[buffer(10)]],
#elif defined(FQ4_SILU_MUL)
                            const device ftype4 *gateIn        [[buffer(7)]],
#endif
                            uint3 gid                          [[threadgroup_position_in_grid]],
                            uint                  tiitg [[thread_index_in_threadgroup]],
                            uint                  tiisg [[thread_index_in_simdgroup]],
                            uint                  sgitg [[simdgroup_index_in_threadgroup]]) {
#ifdef USE_METAL_TENSOR_OPS
#ifdef FQ4_GATEUP_DUAL
    // The dual halves N and stages two weight tiles, so the operands are
    // unchanged and the store destination halves.
    const int K = 32, M = 64, N = 32, NT = 2;
#else
    const int K = 32, M = 64, N = 64, NT = 1;
#endif
#ifdef FQ4_SMEM_PAD
    // A store address is (4*ml + r) * RS + kl in ftype4 units. With RS = K the
    // bank index is independent of ml, so all 16 ml lanes of a simdgroup hit
    // the same banks (16-way conflict); padding breaks that up. RS must stay a
    // multiple of 8 ftype so every operand row starts 16-byte aligned: at fp16
    // RS = 36 puts row 1 at 72 B, which matmul2d reads as garbage.
    const int RS = 40;                 // padded row stride, ftype
#else
    const int RS = K;                  // == the default tensor row stride
#endif
    // sdata must hold the larger of the staged operands and the cT.store
    // destination. Derived from the tile constants rather than hardcoded: the
    // operand side scales with sizeof(ftype), which differs between precisions,
    // while the destination is always FLOAT.
#ifdef FQ4_DOUBLE_BUF
    // Double-buffered staging keeps two operand sets so the next K8 window's
    // dequant/stores overlap the current window's matmul (one barrier per
    // window instead of two). The second set lives in the 16 KB the cT.store
    // destination already budgets, which is overwritten only after every
    // matmul is done.
    const int SMEM_OPERAND4 = 2 * (M + NT * N) * RS * sizeof(ftype) / sizeof(FLOAT4);
#else
    const int SMEM_OPERAND4 = (M + NT * N) * RS * sizeof(ftype) / sizeof(FLOAT4);
#endif
    const int SMEM_DEST4    = M * N * sizeof(FLOAT) / sizeof(FLOAT4);
    threadgroup FLOAT4 sdata[SMEM_OPERAND4 > SMEM_DEST4 ? SMEM_OPERAND4 : SMEM_DEST4] = {0.f};

    const int RS4 = RS / 4;
    const int SB  = M * RS;            // first sB base, ftype
#ifdef FQ4_DOUBLE_BUF
    const int SBUF  = (M + NT * N) * RS; // one buffer-set footprint, ftype
    const int SBUF4 = SBUF / 4;          // one buffer-set footprint, ftype4
#endif
    auto tI = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata,      dextents<int32_t, 2>(K, M), array<int, 2>{1, RS});
    auto tW = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata + SB, dextents<int32_t, 2>(K, N), array<int, 2>{1, RS});
#ifdef FQ4_GATEUP_DUAL
    // Second weight tile, immediately after the first.
    auto tW2 = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata + SB + N * RS, dextents<int32_t, 2>(K, N), array<int, 2>{1, RS});
#endif

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(M, N, K, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mmOps;

    auto cT = mmOps.get_destination_cooperative_tensor<decltype(tI), decltype(tW), FLOAT>();
#ifdef FQ4_GATEUP_DUAL
    // Halving N keeps the two accumulators at the same 32 floats per thread the
    // single N=64 destination already costs.
    auto cT2 = mmOps.get_destination_cooperative_tensor<decltype(tI), decltype(tW2), FLOAT>();
#endif

    int rx = gid.x; // M/64 index
    int uz = gid.y; // N/64 index

    int kl  = tiitg / 16;
    int ml  = tiitg % 16;
    int no  = tiitg / 8;
    int sl  = tiitg % 8;
    int nl  = sl / 2;
    int kwl = sl % 2;
    int mlc = tiitg / 4;
    int nlc = tiitg % 4;

    const int nQuads = N / 4;         // n4 quads spanned by one weight tile
#ifdef FQ4_GATEUP_DUAL
    // Half the 'no' groups stage this projection's tile, half the peer's. Both
    // halves run identical code on selected pointers, so there is no divergence.
    int mtx = no / nQuads;            // 0 = this projection, 1 = peer
    int non = no % nQuads;
    const device ftype *dqs = (mtx == 0) ? dequantScale : dequantScale2;
    const FLOAT sCoef       = (mtx == 0) ? (FLOAT)cst.scaleCoef : (FLOAT)scaleCoef2;
#else
    const int mtx = 0;
    int non = no;
    const device ftype *dqs = dequantScale;
    const FLOAT sCoef       = (FLOAT)cst.scaleCoef;
#endif

    // Input row mapping: 4 rows per thread (M64 / 16 ml groups = 4 rows/thread).
    int idx_m0 = (rx * 16 + ml) * 4 + 0 < cst.inputSize * cst.batch ? (rx * 16 + ml) * 4 + 0 : (cst.inputSize * cst.batch - 1);
    int idx_m1 = (rx * 16 + ml) * 4 + 1 < cst.inputSize * cst.batch ? (rx * 16 + ml) * 4 + 1 : (cst.inputSize * cst.batch - 1);
    int idx_m2 = (rx * 16 + ml) * 4 + 2 < cst.inputSize * cst.batch ? (rx * 16 + ml) * 4 + 2 : (cst.inputSize * cst.batch - 1);
    int idx_m3 = (rx * 16 + ml) * 4 + 3 < cst.inputSize * cst.batch ? (rx * 16 + ml) * 4 + 3 : (cst.inputSize * cst.batch - 1);

    int idx_k4 = 0 * 8 + kl;
    auto xy_in0 = in + idx_k4 * cst.inputSize * cst.batch + idx_m0;
    auto xy_in1 = in + idx_k4 * cst.inputSize * cst.batch + idx_m1;
    auto xy_in2 = in + idx_k4 * cst.inputSize * cst.batch + idx_m2;
    auto xy_in3 = in + idx_k4 * cst.inputSize * cst.batch + idx_m3;

    int idx_n4   = (uz * nQuads + non) < cst.outputDepthQuad ? (uz * nQuads + non) : (cst.outputDepthQuad - 1);
#ifdef W_QUANT_2
#ifdef FQ4_GATEUP_DUAL
    auto wtBase   = (mtx == 0) ? (const device uchar*)wt_int2 : (const device uchar*)wt2_int2;
#else
    auto wtBase   = (const device uchar*)wt_int2;
#endif
    auto xy_wt_i2 = wtBase + (idx_n4 * cst.inputDepthQuad) * 4 + nl;
#elif defined(W_QUANT_3)
#ifdef FQ4_GATEUP_DUAL
    auto wtBase   = (mtx == 0) ? wt_int3 : wt2_int3;
#else
    auto wtBase   = wt_int3;
#endif
    auto xy_wt_i3 = wtBase + idx_n4 * cst.inputDepthQuad * 6;
#elif defined(W_QUANT_4)
#ifdef FQ4_GATEUP_DUAL
    auto wtBase   = (mtx == 0) ? (const device uchar2*)wt_int4 : (const device uchar2*)wt2_int4;
#else
    auto wtBase   = (const device uchar2*)wt_int4;
#endif
    auto xy_wt_i4 = wtBase + (idx_n4 * cst.inputDepthQuad + 0) * 4 + nl;
#else // W_QUANT_8
#ifdef FQ4_GATEUP_DUAL
    auto wtBase   = (mtx == 0) ? (const device char4*)wt_int8 : (const device char4*)wt2_int8;
#else
    auto wtBase   = (const device char4*)wt_int8;
#endif
    auto xy_wt_i8 = wtBase + (idx_n4 * cst.inputDepthQuad + 0) * 4 + nl;
#endif

    // sA layout (ftype array): row m at K4 group kl sits at m * RS4 + kl in
    // ftype4 units; each thread owns 4 rows m = 4*ml + r.
    // sB tiles follow sA: tile mtx starts at (M + mtx * N) * RS4.
    int idx_sa = (4 * ml) * RS4 + kl;
    int idx_sb = (M + mtx * N) * RS4 + (non * 4 + nl) * RS4 + kwl * 4 + 0;
    int quadsPerBlock = (cst.inputDepthQuad + cst.blockCount - 1) / cst.blockCount;

#ifdef FQ4_DOUBLE_BUF
    int it = 0;                        // global K8-window counter -> buffer phase
#endif
    for (int bi = 0; bi < cst.blockCount; ++bi) {
        FLOAT scale0        = FLOAT(dqs[((idx_n4 * cst.blockCount + bi) * 2 + 0) * 4 + nl]) / sCoef;
        FLOAT dequant_bias0 = FLOAT(dqs[((idx_n4 * cst.blockCount + bi) * 2 + 1) * 4 + nl]) / sCoef;
        int zmin = bi * quadsPerBlock;
        int zmax = min(zmin + quadsPerBlock, cst.inputDepthQuad);

        for (int z = zmin; z < zmax; z += 8) {
#if defined(W_QUANT_4) && defined(FQ4_UNORM)
            // Wide-load B staging (A/B-verified): one contiguous 8B uint2 per
            // thread replaces four 2B uchar2 loads. The uint2 at (n4, k4) packs
            // the 4x4 weight tile as four N rows of four K values, so the four
            // stores land on consecutive N rows at one K4 group.
#ifdef FQ4_DOUBLE_BUF
            const int bufF4 = (it & 1) * SBUF4;
#else
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const int bufF4 = 0;
#endif
            {
                const int k4p = tiitg % 8;
                FLOAT4 sc4 = FLOAT4(*(const device ftype4*)(dqs + ((idx_n4 * cst.blockCount + bi) * 2 + 0) * 4)) / sCoef;
                FLOAT4 db4 = FLOAT4(*(const device ftype4*)(dqs + ((idx_n4 * cst.blockCount + bi) * 2 + 1) * 4)) / sCoef;
                const device uint2 *wq = (const device uint2 *)wtBase;
                uint2 w2 = wq[idx_n4 * cst.inputDepthQuad + z + k4p];
                FLOAT4 hi0 = FLOAT4(unpack_unorm4x8_to_float(w2.x & 0xF0F0F0F0u));
                FLOAT4 lo0 = FLOAT4(unpack_unorm4x8_to_float(w2.x & 0x0F0F0F0Fu));
                FLOAT4 hi1 = FLOAT4(unpack_unorm4x8_to_float(w2.y & 0xF0F0F0F0u));
                FLOAT4 lo1 = FLOAT4(unpack_unorm4x8_to_float(w2.y & 0x0F0F0F0Fu));
                FLOAT4 wd[4];
                wd[0] = FLOAT4(hi0[0], lo0[0], hi0[1], lo0[1]);
                wd[1] = FLOAT4(hi0[2], lo0[2], hi0[3], lo0[3]);
                wd[2] = FLOAT4(hi1[0], lo1[0], hi1[1], lo1[1]);
                wd[3] = FLOAT4(hi1[2], lo1[2], hi1[3], lo1[3]);
                #pragma unroll(4)
                for (int j = 0; j < 4; ++j) {
                    FLOAT4 val = FLOAT4(db4[j] - 8.0 * sc4[j]);
                    FLOAT4 sv  = FLOAT4(sc4[j] * (FLOAT)(255.0 / 16.0), sc4[j] * (FLOAT)255.0,
                                        sc4[j] * (FLOAT)(255.0 / 16.0), sc4[j] * (FLOAT)255.0);
                    wd[j] = wd[j] * sv + val;
                    ((threadgroup ftype4*)sdata)[bufF4 + (M + mtx * N) * RS4 + (non * 4 + j) * RS4 + k4p] = ftype4(wd[j]);
                }
            }
#else
            FLOAT4x4 w_dequant;
            {
#ifdef W_QUANT_2
                #pragma unroll(4)
                for (int i = 0; i < 4; ++i) {
                    uchar b = xy_wt_i2[(z + kwl * 4 + i) * 4];
                    w_dequant[i][0] = FLOAT((b >> 6) & 3);
                    w_dequant[i][1] = FLOAT((b >> 4) & 3);
                    w_dequant[i][2] = FLOAT((b >> 2) & 3);
                    w_dequant[i][3] = FLOAT( b       & 3);
                }
                FLOAT4 val = FLOAT4(dequant_bias0 - 2.0 * scale0);
                w_dequant  = w_dequant * scale0 + FLOAT4x4(val, val, val, val);
#elif defined(W_QUANT_3)
                #pragma unroll(4)
                for (int i = 0; i < 4; ++i) {
                    const device uchar* tilePtr = xy_wt_i3 + (z + kwl * 4 + i) * 6;
                    uchar lo = tilePtr[nl];
                    uchar h  = (nl < 2) ? tilePtr[4] : tilePtr[5];
                    uchar hShifted = (nl % 2 == 0) ? (h >> 4) : (h & 0xF);
                    w_dequant[i][0] = FLOAT(((lo >> 6) & 3) | (((hShifted >> 3) & 1) << 2));
                    w_dequant[i][1] = FLOAT(((lo >> 4) & 3) | (((hShifted >> 2) & 1) << 2));
                    w_dequant[i][2] = FLOAT(((lo >> 2) & 3) | (((hShifted >> 1) & 1) << 2));
                    w_dequant[i][3] = FLOAT(( lo       & 3) | (( hShifted       & 1) << 2));
                }
                FLOAT4 val = FLOAT4(dequant_bias0 - 4.0 * scale0);
                w_dequant  = w_dequant * scale0 + FLOAT4x4(val, val, val, val);
#elif defined(W_QUANT_4)
                #pragma unroll(4)
                for (int i = 0; i < 4; ++i) {
                    uchar2 w = xy_wt_i4[(z + kwl * 4 + i) * 4];
                    w_dequant[i][0] = FLOAT(w[0] >> 4);
                    w_dequant[i][1] = FLOAT(w[0] & 0x0F);
                    w_dequant[i][2] = FLOAT(w[1] >> 4);
                    w_dequant[i][3] = FLOAT(w[1] & 0x0F);
                }
                FLOAT4 val = FLOAT4(dequant_bias0 - 8.0 * scale0);
                w_dequant  = w_dequant * scale0 + FLOAT4x4(val, val, val, val);
#else // W_QUANT_8
                #pragma unroll(4)
                for (int i = 0; i < 4; ++i) {
                    char4 w = xy_wt_i8[(z + kwl * 4 + i) * 4];
                    FLOAT4 w4 = FLOAT4(FLOAT(w[0]), FLOAT(w[1]), FLOAT(w[2]), FLOAT(w[3]));
                    w_dequant[i] = w4 * scale0 + dequant_bias0;
                }
#endif
            }

#ifdef FQ4_DOUBLE_BUF
            // The previous window's matmul reads the other buffer set, so this
            // stage proceeds without waiting for it.
            const int bufF4 = (it & 1) * SBUF4;
#else
            // Single-buffer: order the previous window's matmul before the overwrite.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const int bufF4 = 0;
#endif

            #pragma unroll(4)
            for (int i = 0; i < 4; ++i) {
                ((threadgroup ftype4*)sdata)[bufF4 + idx_sb + i] = ftype4(w_dequant[i]);
            }
#endif

            // Input load: 4 rows per thread.
            #ifdef MNN_METAL_SRC_PROTECT
            if (idx_k4 + z < cst.inputDepthQuad) {
                ((threadgroup ftype4*)sdata)[bufF4 + idx_sa]           = (ftype4)*(xy_in0);
                ((threadgroup ftype4*)sdata)[bufF4 + idx_sa + RS4]     = (ftype4)*(xy_in1);
                ((threadgroup ftype4*)sdata)[bufF4 + idx_sa + RS4 * 2] = (ftype4)*(xy_in2);
                ((threadgroup ftype4*)sdata)[bufF4 + idx_sa + RS4 * 3] = (ftype4)*(xy_in3);
            } else {
                ((threadgroup ftype4*)sdata)[bufF4 + idx_sa]           = (ftype4)(0);
                ((threadgroup ftype4*)sdata)[bufF4 + idx_sa + RS4]     = (ftype4)(0);
                ((threadgroup ftype4*)sdata)[bufF4 + idx_sa + RS4 * 2] = (ftype4)(0);
                ((threadgroup ftype4*)sdata)[bufF4 + idx_sa + RS4 * 3] = (ftype4)(0);
            }
            #else
            ((threadgroup ftype4*)sdata)[bufF4 + idx_sa]           = (ftype4)*(xy_in0);
            ((threadgroup ftype4*)sdata)[bufF4 + idx_sa + RS4]     = (ftype4)*(xy_in1);
            ((threadgroup ftype4*)sdata)[bufF4 + idx_sa + RS4 * 2] = (ftype4)*(xy_in2);
            ((threadgroup ftype4*)sdata)[bufF4 + idx_sa + RS4 * 3] = (ftype4)*(xy_in3);
            #endif

#ifdef FQ4_DOUBLE_BUF
            // Advance inside the stage so the next window's DRAM loads issue
            // while this window's matmul runs.
            xy_in0 += 8 * cst.inputSize * cst.batch;
            xy_in1 += 8 * cst.inputSize * cst.batch;
            xy_in2 += 8 * cst.inputSize * cst.batch;
            xy_in3 += 8 * cst.inputSize * cst.batch;
            ++it;
#endif

            threadgroup_barrier(mem_flags::mem_threadgroup);

#ifdef FQ4_DOUBLE_BUF
            const int bufF = bufF4 * 4; // ftype units
            auto tIb = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata + bufF,      dextents<int32_t, 2>(K, M), array<int, 2>{1, RS});
            auto tWb = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata + bufF + SB, dextents<int32_t, 2>(K, N), array<int, 2>{1, RS});
            auto sA = tIb.slice(0, 0);
            auto sB = tWb.slice(0, 0);
#else
            auto sA = tI.slice(0, 0);
            auto sB = tW.slice(0, 0);
#endif
            mmOps.run(sA, sB, cT);
#ifdef FQ4_GATEUP_DUAL
#ifdef FQ4_DOUBLE_BUF
            auto tW2b = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata + bufF + SB + N * RS, dextents<int32_t, 2>(K, N), array<int, 2>{1, RS});
            auto sB2 = tW2b.slice(0, 0);
#else
            auto sB2 = tW2.slice(0, 0);
#endif
            mmOps.run(sA, sB2, cT2);
#endif

#ifndef FQ4_DOUBLE_BUF
            xy_in0 += 8 * cst.inputSize * cst.batch;
            xy_in1 += 8 * cst.inputSize * cst.batch;
            xy_in2 += 8 * cst.inputSize * cst.batch;
            xy_in3 += 8 * cst.inputSize * cst.batch;
#endif
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto tC = tensor<threadgroup FLOAT, dextents<int32_t, 2>, tensor_inline>((threadgroup FLOAT*)sdata, dextents<int32_t, 2>(N, M));
    const int outStride = cst.outputSize * cst.batch;

#ifdef FQ4_GATEUP_DUAL
    // Both destinations share the one scratch buffer, so the peer (gate) tile is
    // stored first and snapshotted into registers before the self (up) tile
    // overwrites it: 4 FLOAT4 per thread, which the halved N pays for. The gate
    // tensor is then never materialized at all.
    //
    // Per-thread N coverage is nQuads/4 = 2 n4 slots (vs 4 at N=64).
    cT2.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    FLOAT4 gv[4];
    #pragma unroll(2)
    for (int mm = 0; mm < 2; ++mm) {
        int m_idx = mm * 32 + mlc;
        #pragma unroll(2)
        for (int j = 0; j < 2; ++j) {
            int n4 = min(uz * nQuads + nlc * 2 + j, cst.outputDepthQuad - 1);
            gv[mm * 2 + j] = ((threadgroup FLOAT4*)sdata)[m_idx * nQuads + nlc * 2 + j] +
                             FLOAT4(biasTerms2[n4]);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    cT.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int mm = 0; mm < 2; ++mm) {
        int m_idx = mm * 32 + mlc;
        auto xy_out = out + (uz * nQuads + nlc * 2) * outStride + (rx * 64 + m_idx);
        if ((rx * 64 + m_idx) < cst.inputSize * cst.batch) {
            #pragma unroll(2)
            for (int j = 0; j < 2; ++j) {
                if (uz * nQuads + nlc * 2 + j < cst.outputDepthQuad) {
                    FLOAT4 u = FLOAT4(activate(ftype4(((threadgroup FLOAT4*)sdata)[m_idx * nQuads + nlc * 2 + j] +
                                                      FLOAT4(biasTerms[uz * nQuads + nlc * 2 + j])), cst.activation));
                    FLOAT4 g = gv[mm * 2 + j];
                    xy_out[j * outStride] = ftype4(u * (g / (FLOAT(1.0) + exp(-g))));
                }
            }
        }
    }
#else
    cT.store(tC);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Epilogue: same per-thread pattern as M=32 kernel, but with mm=0..1 to cover M=64.
    // Each (mm, mlc, nlc) thread role writes 4 N-inner values at (M=rx*64+m_idx, N4-slot=uz*4+nlc).
    //
    // FQ4_SILU_MUL folds the gate/up MUL_SILU here: the gate projection has
    // already written a tensor with this exact layout, so the extra read is at
    // the address about to be written. That turns three full-tensor passes
    // (write up, read up+gate, write out) into one read plus one write.
    for (int mm = 0; mm < 2; ++mm) {
        int m_idx = mm * 32 + mlc;
        auto xy_out = out + ((uz * 4 + nlc) * 4 + 0) * outStride + (rx * 64 + m_idx);
#ifdef FQ4_SILU_MUL
        auto xy_gate = gateIn + ((uz * 4 + nlc) * 4 + 0) * outStride + (rx * 64 + m_idx);
#endif
        if ((rx * 64 + m_idx) < cst.inputSize * cst.batch) {
            #pragma unroll(4)
            for (int j = 0; j < 4; ++j) {
                if ((uz * 4 + nlc) * 4 + j < cst.outputDepthQuad) {
                    ftype4 v = activate(ftype4(((threadgroup FLOAT4*)sdata)[(m_idx * 4 + nlc) * 4 + j] +
                                               FLOAT4(biasTerms[(uz * 4 + nlc) * 4 + j])), cst.activation);
#ifdef FQ4_SILU_MUL
                    FLOAT4 g = FLOAT4(xy_gate[j * outStride]);
                    v = ftype4(FLOAT4(v) * (g / (FLOAT(1.0) + exp(-g))));
#endif
                    xy_out[j * outStride] = v;
                }
            }
        }
    }
#endif // FQ4_GATEUP_DUAL
#endif // USE_METAL_TENSOR_OPS
}
#endif // W_QUANT_2 || W_QUANT_3 || W_QUANT_4 || W_QUANT_8

//======================================================================
// conv1x1_gemm_64x64_split_k_sg — sg_matrix M=64 tile (fp16 weights,
// outer-dequant path, non-tensor-API devices e.g. M4).
//
// Same [N64, K32] weight tile per iteration as the 32x64 kernel, but each
// threadgroup produces M64 output rows — grid.x halves, so weight DRAM
// traffic across threadgroups halves.
// 4 simdgroups: sgitg -> (M half = sgitg/2, N half = sgitg%2); each SG
// computes an M32 x N32 tile = sgd[16] accumulators.
// Threadgroup memory: loop phase A 4 KB + B 4 KB fp16; epilogue 16 KB
// float C. sdata = 1024 FLOAT4 = 16 KB.
//======================================================================
kernel void conv1x1_gemm_64x64_split_k_sg(const device ftype4 *in            [[buffer(0)]],
                            device ftype4 *out                 [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                            const device ftype4x4 *wt          [[buffer(3)]],// [N/4, K/16, N4, K4, K4] fp16 (pre-dequanted)
                            const device ftype4 *biasTerms     [[buffer(4)]],
                            uint3 gid                          [[threadgroup_position_in_grid]],
                            uint                  tiitg[[thread_index_in_threadgroup]],
                            uint                  tiisg[[thread_index_in_simdgroup]],
                            uint                  sgitg[[simdgroup_index_in_threadgroup]]) {
    /*
     // Read (ftype offsets):
     0~2047    ---> input : [K4, M64, K8]
     2048~4095 ---> weight: [K4, K8, N64]
     // Write:
     FLOAT 0~4095 ---> per-SG [M32, N32] tiles at float offset 1024*sgitg
     */
    threadgroup FLOAT4 sdata[1024] = {0.f};

    INIT_SIMDGROUP_MATRIX(4, 4, 16);

    int rx = gid.x;// M/64
    int uz = gid.y;// N/64

    // A:[4, 2, 16] — ko: K4-chunk of K8 (0~3), kl: K2 (0~1), ml: 0~15 (4 M rows each)
    int ko = tiitg / 32;
    int rcl = tiitg % 32;
    int kl = rcl / 16;
    int ml = rcl % 16;
    // B:[16, 2, 4] — identical to the 32x64 kernel (weight tile unchanged)
    int no = tiitg / 8;
    int sl = tiitg % 8;
    int kwl = sl / 4;
    int nl = sl % 4;

    /** input:
     threadgroup: [K4, M64, K8] -> [K4, M16, M4, K2, K4]
     index: [ko, ml, M4, kl, K4] — each thread M4K4
     layout: [K/4, M, K4] -> [K/32, K4, K2, M/64, M16, M4, K4]
     */
    /** weight:
     threadgroup: [K4, K8, N64] -> [K2, K4, K4, N16, N4]
     index: [kwl, K4, K4, no, nl] — each thread K4K4 (one ftype4x4)
     layout: [N/4, K/16, N4, K4, K4] -> [N/64, N16, K/32, K2, N4, K4, K4]
     */

    // boundary limit
    int mBound = cst.inputSize * cst.batch;
    int idx_m40 = (rx * 16 + ml) * 4 + 0 < mBound ? (rx * 16 + ml) * 4 + 0 : (mBound - 1);
    int idx_m41 = (rx * 16 + ml) * 4 + 1 < mBound ? (rx * 16 + ml) * 4 + 1 : (mBound - 1);
    int idx_m42 = (rx * 16 + ml) * 4 + 2 < mBound ? (rx * 16 + ml) * 4 + 2 : (mBound - 1);
    int idx_m43 = (rx * 16 + ml) * 4 + 3 < mBound ? (rx * 16 + ml) * 4 + 3 : (mBound - 1);

    int idx_k4 = ko * 2 + kl;
    auto xy_in0 = in + idx_k4 * mBound + idx_m40;// [K/4, M, K4]
    auto xy_in1 = in + idx_k4 * mBound + idx_m41;
    auto xy_in2 = in + idx_k4 * mBound + idx_m42;
    auto xy_in3 = in + idx_k4 * mBound + idx_m43;

    int idx_n4 = (uz * 16 + no) < cst.outputDepthQuad ? (uz * 16 + no) : (cst.outputDepthQuad - 1);
    auto xy_wt = wt + (idx_n4 * ((cst.inputDepthQuad + 3) / 4) + kwl) * 4 + nl;// [N/4, K/16, N4, K4, K4]

    int idx_sa = (ko * 64 + ml * 4 + 0) * 2 + kl;                 // ftype4 units, [K4, M64, K2] x K4
    int idx_sb = 2048 + (kwl * 16 + 0) * 64 + no * 4 + nl;        // ftype units, [K2, K16, N64]

    for (int z = 0; z < cst.inputDepthQuad; z += 8) {
        ftype4x4 w_local = xy_wt[z];

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll(16)
        for (int i = 0; i < 16; ++i) {
            ((threadgroup ftype*)sdata)[idx_sb + 64 * i] = w_local[i / 4][i % 4];
        }

        #ifdef MNN_METAL_SRC_PROTECT
        if (idx_k4 + z < cst.inputDepthQuad) {
            ((threadgroup ftype4*)sdata)[idx_sa]     = (ftype4)*(xy_in0);
            ((threadgroup ftype4*)sdata)[idx_sa + 2] = (ftype4)*(xy_in1);
            ((threadgroup ftype4*)sdata)[idx_sa + 4] = (ftype4)*(xy_in2);
            ((threadgroup ftype4*)sdata)[idx_sa + 6] = (ftype4)*(xy_in3);
        } else {
            ((threadgroup ftype4*)sdata)[idx_sa]     = (ftype4)(0);
            ((threadgroup ftype4*)sdata)[idx_sa + 2] = (ftype4)(0);
            ((threadgroup ftype4*)sdata)[idx_sa + 4] = (ftype4)(0);
            ((threadgroup ftype4*)sdata)[idx_sa + 6] = (ftype4)(0);
        }
        #else
        ((threadgroup ftype4*)sdata)[idx_sa]     = (ftype4)*(xy_in0);
        ((threadgroup ftype4*)sdata)[idx_sa + 2] = (ftype4)*(xy_in1);
        ((threadgroup ftype4*)sdata)[idx_sa + 4] = (ftype4)*(xy_in2);
        ((threadgroup ftype4*)sdata)[idx_sa + 6] = (ftype4)*(xy_in3);
        #endif

        threadgroup_barrier(mem_flags::mem_threadgroup);

        /*
        A: [K4, M64, K8] -> per-SG M half at ftype offset 256*(sgitg/2)
        B: [K4, K8, N64] -> per-SG N half at ftype offset 2048 + 32*(sgitg%2)
        */
        threadgroup ftype * sdata_a = (threadgroup ftype*)sdata + 256 * (sgitg / 2);
        threadgroup ftype * sdata_b = (threadgroup ftype*)sdata + 2048 + 32 * (sgitg % 2);

        #pragma unroll(4)
        for (short ik = 0; ik < 4; ik++) {
            simdgroup_load(sga[0], (const threadgroup ftype*)sdata_a + 512 * ik +   0, 8);
            simdgroup_load(sga[1], (const threadgroup ftype*)sdata_a + 512 * ik +  64, 8);
            simdgroup_load(sga[2], (const threadgroup ftype*)sdata_a + 512 * ik + 128, 8);
            simdgroup_load(sga[3], (const threadgroup ftype*)sdata_a + 512 * ik + 192, 8);

            simdgroup_load(sgb[0], (const threadgroup ftype*)sdata_b + 512 * ik +  0, 64);
            simdgroup_load(sgb[1], (const threadgroup ftype*)sdata_b + 512 * ik +  8, 64);
            simdgroup_load(sgb[2], (const threadgroup ftype*)sdata_b + 512 * ik + 16, 64);
            simdgroup_load(sgb[3], (const threadgroup ftype*)sdata_b + 512 * ik + 24, 64);

            simdgroup_barrier(mem_flags::mem_none);
            SIMDGROUP_MATRIX_FMA(4, 4);
            simdgroup_barrier(mem_flags::mem_none);
        }

        xy_in0 += 8 * mBound;
        xy_in1 += 8 * mBound;
        xy_in2 += 8 * mBound;
        xy_in3 += 8 * mBound;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup FLOAT * sdata_c = (threadgroup FLOAT*)sdata + 1024 * sgitg;
    SIMDGROUP_MATRIX_STORE(sdata_c, 16);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Epilogue: 128 threads x 8 stores cover the [M64, N64] tile.
    // mlc: output row within tile (0~63); nh: N32 half (0~1).
    // Source: SG (msg*2 + nh) tile, sgd index j*4+mi, 8x8 row-major ld 8.
    {
        int mlc = tiitg % 64;
        int nh  = tiitg / 64;
        int msg = mlc / 32;
        int mi  = (mlc % 32) / 8;
        int mr  = mlc % 8;
        threadgroup FLOAT * csrc = (threadgroup FLOAT*)sdata + 1024 * (msg * 2 + nh);
        if ((rx * 64 + mlc) < mBound) {
            for (int n4o = 0; n4o < 8; ++n4o) {
                int n4 = nh * 8 + n4o;                  // N4 group within N64 (0~15)
                if ((uz * 16 + n4) < cst.outputDepthQuad) {
                    FLOAT4 val = *((threadgroup FLOAT4*)(csrc + 64 * ((n4o / 2) * 4 + mi) + mr * 8 + (n4o % 2) * 4));
                    out[(uz * 16 + n4) * cst.outputSize * cst.batch + (rx * 64 + mlc)] =
                        activate(ftype4(val + FLOAT4(biasTerms[uz * 16 + n4])), cst.activation);
                }
            }
        }
    }
}
)metal";

static const char* gConv1x1WfpSgReduce = R"metal(
kernel void conv1x1_z4_sg(const device ftype4 *in            [[buffer(0)]],
                         device ftype4 *out                 [[buffer(1)]],
                         constant conv1x1_constants& cst    [[buffer(2)]],
                         const device ftype4x4 *wt          [[buffer(3)]],
                         const device ftype4 *biasTerms     [[buffer(4)]],
                         uint3 gid[[threadgroup_position_in_grid]],
                         uint  tiisg[[thread_index_in_simdgroup]],
                         uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    if ((int)gid.x >= cst.outputSize || (int)gid.y >= cst.outputDepthQuad || (int)gid.z >= cst.batch) return;

    int rx = gid.x;
    int uz = gid.y;
    auto xy_wt = wt + uz * cst.inputDepthQuad;
    auto xy_in0  = in  + (int)gid.z  * cst.inputSize + rx + 0;
    auto xy_out = out + (int)gid.z * cst.outputSize + uz * cst.outputSize * cst.batch + rx;
    auto biasValue = FLOAT4(biasTerms[uz]);
    FLOAT4 result0 = 0;

    for (int z = tiisg; z < cst.inputDepthQuad; z+=SIMD_GROUP_WIDTH) {
        auto xy_in = xy_in0 + z * cst.inputSize * cst.batch;
        auto in40 = *xy_in;
        auto w = xy_wt[z];

        result0 += FLOAT4(in40 * w);
    }
    result0 = simd_sum(result0);

    *xy_out = activate(ftype4(result0 + biasValue), cst.activation);
}
)metal";

static const char* gConv1x1WqSgReduce = R"metal(


template <int AREA_THREAD>
kernel void conv1x1_gemv_g4mx_wquant_sg(const device ftype4 *in            [[buffer(0)]],
                            device ftype4 *out                 [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                        #ifdef W_QUANT_2
                            const device uchar4 *wt             [[buffer(3)]],
                        #elif defined(W_QUANT_3)
                            const device uchar *wt              [[buffer(3)]],
                        #elif defined(W_QUANT_4)
                            const device ushort4 *wt             [[buffer(3)]],
                        #elif defined(W_QUANT_8)
                            const device MNN::char4x4 *wt      [[buffer(3)]],
                        #endif
                            const device ftype4 *biasTerms     [[buffer(4)]],
                            const device ftype4 *dequantScale  [[buffer(5)]],
                            uint3 gid[[threadgroup_position_in_grid]],
                            uint  tiisg[[thread_index_in_simdgroup]],
                            uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    // each threadgroup contain 1 simdgroup
    // each simdgroup compute 8 data
    int uz = gid.x;
    int rx = gid.y * AREA_THREAD;
    auto area_size = cst.outputSize * cst.batch;
    if(uz >= cst.outputDepthQuad || rx >= area_size) {
        return;
    }
#ifdef W_QUANT_3
    auto xy_wt = wt + uz * cst.inputDepthQuad * 6;
#else
    auto xy_wt = wt + uz * cst.inputDepthQuad;
#endif
    auto xy_in0  = in + rx;
    auto xy_out = out + uz * area_size + rx;
    auto biasValue = FLOAT4(biasTerms[uz]);
    FLOAT4 result[AREA_THREAD] = {FLOAT4(0)};
    int quadsPerBlock = (cst.inputDepthQuad + cst.blockCount - 1) / cst.blockCount;

    int lanesPerBlock = min(SIMD_GROUP_WIDTH, quadsPerBlock);
    int blocksInFlight = SIMD_GROUP_WIDTH / lanesPerBlock;
    int laneInBlock = (tiisg) % lanesPerBlock;
    int blockSlot = (tiisg) / lanesPerBlock;


    for (int bi= blockSlot; bi<cst.blockCount; bi += blocksInFlight) {
        FLOAT4 scale = FLOAT4(dequantScale[2 * (uz * cst.blockCount + bi) + 0]) / (FLOAT)cst.scaleCoef;
        FLOAT4 dequant_bias = FLOAT4(dequantScale[2 * (uz * cst.blockCount + bi) + 1]) / (FLOAT)cst.scaleCoef;
        int zmin = bi * quadsPerBlock;
        int zmax = min(zmin + quadsPerBlock, cst.inputDepthQuad);

    #if defined(W_QUANT_2) || defined(W_QUANT_3)
        // Deferred dequantization with power-of-two pre-scaling (see g8 kernel for
        // mask/pre-scale derivation). Each token accumulates its own raw_dot and
        // input_sum; the weight tile is loaded once and shared across all tokens.
        {
            FLOAT4 raw_dot[AREA_THREAD] = {FLOAT4(0)};
            FLOAT input_sum[AREA_THREAD] = {FLOAT(0)};
        #ifdef W_QUANT_2
            for (int z = zmin + laneInBlock; z < zmax; z += lanesPerBlock) {
                uchar4 w_b = xy_wt[z];
                auto base_xy = xy_in0 + z * area_size;
                for (int i = 0; i < AREA_THREAD; i++) {
                #ifdef MNN_METAL_SRC_PROTECT
                    FLOAT4 in40 = (rx + (int)i) < area_size ? (FLOAT4)*(base_xy + i) : (FLOAT4)0;
                #else
                    FLOAT4 in40 = (FLOAT4)*(base_xy + i);
                #endif
                    input_sum[i] += in40[0] + in40[1] + in40[2] + in40[3];
                    FLOAT in_ps0 = in40[0] * FLOAT(1.0/64.0);
                    FLOAT in_ps1 = in40[1] * FLOAT(1.0/16.0);
                    FLOAT in_ps2 = in40[2] * FLOAT(1.0/4.0);
                    raw_dot[i][0] += in_ps0 * FLOAT(w_b[0] & 0xC0) + in_ps1 * FLOAT(w_b[0] & 0x30)
                                   + in_ps2 * FLOAT(w_b[0] & 0x0C) + in40[3] * FLOAT(w_b[0] & 0x03);
                    raw_dot[i][1] += in_ps0 * FLOAT(w_b[1] & 0xC0) + in_ps1 * FLOAT(w_b[1] & 0x30)
                                   + in_ps2 * FLOAT(w_b[1] & 0x0C) + in40[3] * FLOAT(w_b[1] & 0x03);
                    raw_dot[i][2] += in_ps0 * FLOAT(w_b[2] & 0xC0) + in_ps1 * FLOAT(w_b[2] & 0x30)
                                   + in_ps2 * FLOAT(w_b[2] & 0x0C) + in40[3] * FLOAT(w_b[2] & 0x03);
                    raw_dot[i][3] += in_ps0 * FLOAT(w_b[3] & 0xC0) + in_ps1 * FLOAT(w_b[3] & 0x30)
                                   + in_ps2 * FLOAT(w_b[3] & 0x0C) + in40[3] * FLOAT(w_b[3] & 0x03);
                }
            }
            FLOAT4 adjusted_bias = dequant_bias - FLOAT(2.0) * scale;
            for (int i = 0; i < AREA_THREAD; i++) {
                result[i] += raw_dot[i] * scale + input_sum[i] * adjusted_bias;
            }
        #else // W_QUANT_3
            for (int z = zmin + laneInBlock; z < zmax; z += lanesPerBlock) {
                const device ushort* tilePtr = (const device ushort*)(xy_wt + z * 6);
                ushort tw0 = tilePtr[0], tw1 = tilePtr[1], tw2 = tilePtr[2];
                uchar lo0 = uchar(tw0), lo1 = uchar(tw0 >> 8), lo2 = uchar(tw1), lo3 = uchar(tw1 >> 8);
                uchar hb0 = uchar(tw2), hb1 = uchar(tw2 >> 8);
                uchar h0 = hb0 >> 4, h1 = hb0 & 0xF, h2 = hb1 >> 4, h3 = hb1 & 0xF;
                auto base_xy = xy_in0 + z * area_size;
                for (int i = 0; i < AREA_THREAD; i++) {
                #ifdef MNN_METAL_SRC_PROTECT
                    FLOAT4 in40 = (rx + (int)i) < area_size ? (FLOAT4)*(base_xy + i) : (FLOAT4)0;
                #else
                    FLOAT4 in40 = (FLOAT4)*(base_xy + i);
                #endif
                    input_sum[i] += in40[0] + in40[1] + in40[2] + in40[3];
                    FLOAT in_ps0 = in40[0] * FLOAT(1.0/64.0);
                    FLOAT in_ps1 = in40[1] * FLOAT(1.0/16.0);
                    FLOAT in_ps2 = in40[2] * FLOAT(1.0/4.0);
                    FLOAT in_hi0 = in40[0] * FLOAT(0.5);
                    FLOAT in_hi2 = in40[2] * FLOAT(2.0);
                    FLOAT in_hi3 = in40[3] * FLOAT(4.0);
                    raw_dot[i][0] += in_ps0 * FLOAT(lo0 & 0xC0) + in_ps1 * FLOAT(lo0 & 0x30)
                                   + in_ps2 * FLOAT(lo0 & 0x0C) + in40[3] * FLOAT(lo0 & 0x03)
                                   + in_hi0 * FLOAT(h0 & 0x8) + in40[1] * FLOAT(h0 & 0x4)
                                   + in_hi2 * FLOAT(h0 & 0x2) + in_hi3 * FLOAT(h0 & 0x1);
                    raw_dot[i][1] += in_ps0 * FLOAT(lo1 & 0xC0) + in_ps1 * FLOAT(lo1 & 0x30)
                                   + in_ps2 * FLOAT(lo1 & 0x0C) + in40[3] * FLOAT(lo1 & 0x03)
                                   + in_hi0 * FLOAT(h1 & 0x8) + in40[1] * FLOAT(h1 & 0x4)
                                   + in_hi2 * FLOAT(h1 & 0x2) + in_hi3 * FLOAT(h1 & 0x1);
                    raw_dot[i][2] += in_ps0 * FLOAT(lo2 & 0xC0) + in_ps1 * FLOAT(lo2 & 0x30)
                                   + in_ps2 * FLOAT(lo2 & 0x0C) + in40[3] * FLOAT(lo2 & 0x03)
                                   + in_hi0 * FLOAT(h2 & 0x8) + in40[1] * FLOAT(h2 & 0x4)
                                   + in_hi2 * FLOAT(h2 & 0x2) + in_hi3 * FLOAT(h2 & 0x1);
                    raw_dot[i][3] += in_ps0 * FLOAT(lo3 & 0xC0) + in_ps1 * FLOAT(lo3 & 0x30)
                                   + in_ps2 * FLOAT(lo3 & 0x0C) + in40[3] * FLOAT(lo3 & 0x03)
                                   + in_hi0 * FLOAT(h3 & 0x8) + in40[1] * FLOAT(h3 & 0x4)
                                   + in_hi2 * FLOAT(h3 & 0x2) + in_hi3 * FLOAT(h3 & 0x1);
                }
            }
            FLOAT4 adjusted_bias = dequant_bias - FLOAT(4.0) * scale;
            for (int i = 0; i < AREA_THREAD; i++) {
                result[i] += raw_dot[i] * scale + input_sum[i] * adjusted_bias;
            }
        #endif
        }
    #elif defined(W_QUANT_4)
        if constexpr (AREA_THREAD == 1) {
            // Deferred dequantization with ushort pre-scaling (decode only, AREA_THREAD==1):
            // Uses ushort4 native vector + mask instead of custom struct shift/mask.
            FLOAT4 raw_dot = FLOAT4(0);
            FLOAT input_sum = FLOAT(0);

            for (int z = zmin + laneInBlock; z < zmax; z += lanesPerBlock) {
                FLOAT4 in40 = (FLOAT4)*(xy_in0 + z * area_size);
                // Pre-scale input for ushort nibble positions
                FLOAT s0 = in40[1];
                FLOAT s1 = in40[0] * FLOAT(0.0625);         // /16
                FLOAT s2 = in40[3] * FLOAT(0.00390625);     // /256
                FLOAT s3 = in40[2] * FLOAT(0.000244140625); // /4096

                input_sum += in40[0] + in40[1] + in40[2] + in40[3];

                ushort4 w16 = xy_wt[z];

                raw_dot[0] += s0 * FLOAT(w16[0] & 0x000F) + s1 * FLOAT(w16[0] & 0x00F0)
                            + s2 * FLOAT(w16[0] & 0x0F00) + s3 * FLOAT(w16[0] & 0xF000);
                raw_dot[1] += s0 * FLOAT(w16[1] & 0x000F) + s1 * FLOAT(w16[1] & 0x00F0)
                            + s2 * FLOAT(w16[1] & 0x0F00) + s3 * FLOAT(w16[1] & 0xF000);
                raw_dot[2] += s0 * FLOAT(w16[2] & 0x000F) + s1 * FLOAT(w16[2] & 0x00F0)
                            + s2 * FLOAT(w16[2] & 0x0F00) + s3 * FLOAT(w16[2] & 0xF000);
                raw_dot[3] += s0 * FLOAT(w16[3] & 0x000F) + s1 * FLOAT(w16[3] & 0x00F0)
                            + s2 * FLOAT(w16[3] & 0x0F00) + s3 * FLOAT(w16[3] & 0xF000);
            }
            FLOAT4 adjusted_bias = dequant_bias - FLOAT(8.0) * scale;
            result[0] += raw_dot * scale + input_sum * adjusted_bias;
        } else {
            // Original per-element dequantization (prefill, AREA_THREAD > 1)
            for (int z = zmin + laneInBlock; z < zmax; z += lanesPerBlock) {
                ushort4 w16 = xy_wt[z];
                FLOAT4x4 w_dequant;
                for (int i = 0; i < 4; i += 1) {
                    FLOAT4 w4 = FLOAT4((float)((w16[i] >> 4) & 0xF) - 8, (float)(w16[i] & 0xF) - 8, (float)((w16[i] >> 12) & 0xF) - 8, (float)((w16[i] >> 8) & 0xF) - 8);
                    w_dequant[i] = w4 * scale[i] + dequant_bias[i];
                }

                auto base_xy = xy_in0 + z * area_size;
                for(int i = 0; i < AREA_THREAD; i++) {
                    #ifdef MNN_METAL_SRC_PROTECT
                    FLOAT4 in40 = (rx + (int)i) < area_size ? (FLOAT4)*(base_xy + i) : (FLOAT4)0;
                    #else
                    FLOAT4 in40 = (FLOAT4)*(base_xy + i);
                    #endif
                    result[i] += FLOAT4(in40 * w_dequant);
                }
            }
        }
        #elif defined(W_QUANT_8)
        for (int z = zmin + laneInBlock; z < zmax; z += lanesPerBlock) {
            auto w = xy_wt[z];
            FLOAT4x4 w_fp32 = FLOAT4x4(FLOAT4(w[0]), FLOAT4(w[1]), FLOAT4(w[2]), FLOAT4(w[3]));
            FLOAT4x4 w_dequant;
            for (int i = 0; i < 4; ++i) {
                w_dequant[i] = w_fp32[i] * scale[i] + dequant_bias[i];
            }

            auto base_xy = xy_in0 + z * area_size;
            for(int i = 0; i < AREA_THREAD; i++) {
                #ifdef MNN_METAL_SRC_PROTECT
                FLOAT4 in40 = (rx + (int)i) < area_size ? (FLOAT4)*(base_xy + i) : (FLOAT4)0;
                #else
                FLOAT4 in40 = (FLOAT4)*(base_xy + i);
                #endif
                result[i] += FLOAT4(in40 * w_dequant);
            }
        }
        #endif
    }

    for(int i = 0; i < AREA_THREAD; i++) {
        result[i] = simd_sum(result[i]);
    }

    // result store
    for(uint i = 0; i < AREA_THREAD; i++) {
        if (tiisg == i && (rx + (int)i) < area_size) {
            xy_out[i] = activate(ftype4(result[i] + biasValue), cst.activation);
        }
    }
}

typedef decltype(conv1x1_gemv_g4mx_wquant_sg<1>) kernel_type_t;
template [[host_name("conv1x1_gemv_g4m2_wquant_sg")]] kernel kernel_type_t conv1x1_gemv_g4mx_wquant_sg<2>;
template [[host_name("conv1x1_gemv_g4m3_wquant_sg")]] kernel kernel_type_t conv1x1_gemv_g4mx_wquant_sg<3>;
template [[host_name("conv1x1_gemv_g4m4_wquant_sg")]] kernel kernel_type_t conv1x1_gemv_g4mx_wquant_sg<4>;
template [[host_name("conv1x1_gemv_g4m5_wquant_sg")]] kernel kernel_type_t conv1x1_gemv_g4mx_wquant_sg<5>;
template [[host_name("conv1x1_gemv_g4m6_wquant_sg")]] kernel kernel_type_t conv1x1_gemv_g4mx_wquant_sg<6>;
template [[host_name("conv1x1_gemv_g4m7_wquant_sg")]] kernel kernel_type_t conv1x1_gemv_g4mx_wquant_sg<7>;
template [[host_name("conv1x1_gemv_g4m8_wquant_sg")]] kernel kernel_type_t conv1x1_gemv_g4mx_wquant_sg<8>;
template [[host_name("conv1x1_gemv_g4m9_wquant_sg")]] kernel kernel_type_t conv1x1_gemv_g4mx_wquant_sg<9>;
template [[host_name("conv1x1_gemv_g4m10_wquant_sg")]] kernel kernel_type_t conv1x1_gemv_g4mx_wquant_sg<10>;
template [[host_name("conv1x1_gemv_g4m11_wquant_sg")]] kernel kernel_type_t conv1x1_gemv_g4mx_wquant_sg<11>;
template [[host_name("conv1x1_gemv_g4m12_wquant_sg")]] kernel kernel_type_t conv1x1_gemv_g4mx_wquant_sg<12>;
template [[host_name("conv1x1_gemv_g4m13_wquant_sg")]] kernel kernel_type_t conv1x1_gemv_g4mx_wquant_sg<13>;
template [[host_name("conv1x1_gemv_g4m14_wquant_sg")]] kernel kernel_type_t conv1x1_gemv_g4mx_wquant_sg<14>;
template [[host_name("conv1x1_gemv_g4m15_wquant_sg")]] kernel kernel_type_t conv1x1_gemv_g4mx_wquant_sg<15>;
template [[host_name("conv1x1_gemv_g4m16_wquant_sg")]] kernel kernel_type_t conv1x1_gemv_g4mx_wquant_sg<16>;

// 2-simdgroup GEMV kernel: each threadgroup has 2 simdgroups, each independently
// processes one outputDepthQuad (4 OC). Input is shared via L1 cache (no barrier needed).
// Halves the number of dispatched threadgroups vs g4m1 for better GPU occupancy.
// Uses deferred dequantization with ushort pre-scaling trick.
kernel void conv1x1_gemv_g4m1_2sg_wquant_sg(const device ftype4 *in       [[buffer(0)]],
                            device ftype4 *out                             [[buffer(1)]],
                            constant conv1x1_constants& cst                [[buffer(2)]],
                        #ifdef W_QUANT_2
                            const device uchar4 *wt                         [[buffer(3)]],
                        #elif defined(W_QUANT_3)
                            const device uchar *wt                          [[buffer(3)]],
                        #elif defined(W_QUANT_4)
                            const device ushort4 *wt                        [[buffer(3)]],
                        #elif defined(W_QUANT_8)
                            const device MNN::char4x4 *wt                  [[buffer(3)]],
                        #endif
                            const device ftype4 *biasTerms                  [[buffer(4)]],
                            const device ftype4 *dequantScale               [[buffer(5)]],
                        #ifdef GATE_UP_FUSED
                            device ftype4 *out_up                           [[buffer(6)]],
                        #ifdef W_QUANT_2
                            const device uchar4 *wt_up                      [[buffer(7)]],
                        #elif defined(W_QUANT_3)
                            const device uchar *wt_up                       [[buffer(7)]],
                        #elif defined(W_QUANT_4)
                            const device ushort4 *wt_up                     [[buffer(7)]],
                        #elif defined(W_QUANT_8)
                            const device MNN::char4x4 *wt_up               [[buffer(7)]],
                        #endif
                            const device ftype4 *biasTerms_up               [[buffer(8)]],
                            const device ftype4 *dequantScale_up            [[buffer(9)]],
                            constant float *gate_up_seg                     [[buffer(14)]],
                        #endif
                        #ifdef QKV_FUSED
                            device ftype4 *out_k                            [[buffer(6)]],
                        #ifdef W_QUANT_2
                            const device uchar4 *wt_k                       [[buffer(7)]],
                        #elif defined(W_QUANT_3)
                            const device uchar *wt_k                        [[buffer(7)]],
                        #elif defined(W_QUANT_4)
                            const device ushort4 *wt_k                      [[buffer(7)]],
                        #elif defined(W_QUANT_8)
                            const device MNN::char4x4 *wt_k                [[buffer(7)]],
                        #endif
                            const device ftype4 *biasTerms_k                [[buffer(8)]],
                            const device ftype4 *dequantScale_k             [[buffer(9)]],
                            device ftype4 *out_v                            [[buffer(10)]],
                        #ifdef W_QUANT_2
                            const device uchar4 *wt_v                       [[buffer(11)]],
                        #elif defined(W_QUANT_3)
                            const device uchar *wt_v                        [[buffer(11)]],
                        #elif defined(W_QUANT_4)
                            const device ushort4 *wt_v                      [[buffer(11)]],
                        #elif defined(W_QUANT_8)
                            const device MNN::char4x4 *wt_v                [[buffer(11)]],
                        #endif
                            const device ftype4 *biasTerms_v                [[buffer(12)]],
                            const device ftype4 *dequantScale_v             [[buffer(13)]],
                            // qkv_seg: [0..1]=k/v scaleCoef, [2..3]=k/v
                            //          outputDepthQuad, [4..5]=4th projection's
                            //          scaleCoef + outputDepthQuad,
                            //          [6..8]=packed-grid base of projections 1..3,
                            //          [9..12]=merged-output element offset per
                            //          projection (QKV_MERGED_OUT)
                            constant float *qkv_seg                         [[buffer(14)]],
                        #ifdef QKV_FUSED_P4
                            device ftype4 *out_w                            [[buffer(15)]],
                        #ifdef W_QUANT_2
                            const device uchar4 *wt_w                       [[buffer(16)]],
                        #elif defined(W_QUANT_3)
                            const device uchar *wt_w                        [[buffer(16)]],
                        #elif defined(W_QUANT_4)
                            const device ushort4 *wt_w                      [[buffer(16)]],
                        #elif defined(W_QUANT_8)
                            const device MNN::char4x4 *wt_w                [[buffer(16)]],
                        #endif
                            const device ftype4 *biasTerms_w                [[buffer(17)]],
                            const device ftype4 *dequantScale_w             [[buffer(18)]],
                        #endif
                        #endif
                        #ifdef LN_FUSED
                            const device ftype4 *ln_residual_in             [[buffer(20)]],
                            const device float4 *ln_gamma                    [[buffer(21)]],
                            device ftype4 *ln_residual_out                  [[buffer(22)]],
                            constant float *ln_eps                          [[buffer(23)]],
                        #endif
                            uint3 gid[[threadgroup_position_in_grid]],
                            uint  tiisg[[thread_index_in_simdgroup]],
                            uint  sgitg[[simdgroup_index_in_threadgroup]]) {
#ifdef GATE_UP_FUSED
    // gid.z selects: 0 = gate (leader), 1 = up (follower)
    if (gid.z == 1) {
        out = out_up;
        wt = wt_up;
        biasTerms = biasTerms_up;
        dequantScale = dequantScale_up;
    }
#endif
#ifdef QKV_FUSED
    // Unlike GATE_UP_FUSED the projections have different outputChannel, so
    // each follower carries its own outputDepthQuad.
#ifdef QKV_PACKED_GRID
    // Packed grid: the projections' threadgroup ranges are laid end to end
    // along x (grid.z is 1), so members smaller than the largest one no longer
    // launch threadgroups that do nothing but early-return. qkv_seg[6..8] hold
    // the base of projections 1..3; recover (projection, local x) from the flat
    // index. Bases are ascending, so the comparisons cascade.
    int qkv_proj = 0;
    if (int(gid.x) >= int(qkv_seg[6])) qkv_proj = 1;
    if (int(gid.x) >= int(qkv_seg[7])) qkv_proj = 2;
#ifdef QKV_FUSED_P4
    if (int(gid.x) >= int(qkv_seg[8])) qkv_proj = 3;
#endif
    const int qkv_gx = int(gid.x) - (qkv_proj == 0 ? 0 : int(qkv_seg[5 + qkv_proj]));
#else
    // Rectangular grid: z selects the projection, x is sized for the largest.
    const int qkv_proj = int(gid.z);
    const int qkv_gx = int(gid.x);
#endif
    int qkvOutputDepthQuad = cst.outputDepthQuad;
    float qkv_scale_coef = cst.scaleCoef;
    if (qkv_proj == 1) {
#ifndef QKV_MERGED_OUT
        out = out_k;
#endif
        biasTerms = biasTerms_k;
        wt = wt_k;
        dequantScale = dequantScale_k;
        qkv_scale_coef = qkv_seg[0];
        qkvOutputDepthQuad = int(qkv_seg[2]);
    } else if (qkv_proj == 2) {
#ifndef QKV_MERGED_OUT
        out = out_v;
#endif
        biasTerms = biasTerms_v;
        wt = wt_v;
        dequantScale = dequantScale_v;
        qkv_scale_coef = qkv_seg[1];
        qkvOutputDepthQuad = int(qkv_seg[3]);
    }
#ifdef QKV_FUSED_P4
    // 4-projection groups (Qwen3.5 linear-attention layers: qkv/z/b/a share
    // one LN input).
    else if (qkv_proj == 3) {
#ifndef QKV_MERGED_OUT
        out = out_w;
#endif
        biasTerms = biasTerms_w;
        wt = wt_w;
        dequantScale = dequantScale_w;
        qkv_scale_coef = qkv_seg[4];
        qkvOutputDepthQuad = int(qkv_seg[5]);
    }
#endif
    // Merged output: every member's output tensor is laid into one allocation, so
    // the destination is the leader's base (buffer(1)) plus a per-projection
    // element offset -- qkv_seg[9..12], the leader's being 0 -- and out_k/out_v/
    // out_w go unread. The alternative form selects one of N bound output
    // pointers in the branch above instead.
#ifdef QKV_MERGED_OUT
    const int qkv_out_off = int(qkv_seg[9 + qkv_proj]);
#define GEMV_OUT_BASE qkv_out_off
#else
#define GEMV_OUT_BASE 0
#endif
#define GEMV_OUT out
    const int tg_x = qkv_gx;
#else
#define GEMV_OUT out
#define GEMV_OUT_BASE 0
    const int tg_x = int(gid.x);
#endif

    const int area_size = cst.outputSize * cst.batch;

    // Threadgroup geometry, shared by the LN prologue and the uz derivations
    // below. tgQuads = output quads the threadgroup owns, tg_simds = its
    // simdgroup count; the host must dispatch tg_simds * 32 threads.
    //
    // The single-stream shapes are two numbers, not a list of named variants:
    //   GEMV_QUADS_PER_TG  output quads per threadgroup (2, or 4 to hold the
    //                      dual-stream grid's threadgroup count so the LN
    //                      prologue is not re-read by twice as many groups)
    //   GEMV_SPLIT_K       K ranges each quad is split into (1 = unsplit,
    //                      2 = two halves reduced through threadgroup memory)
    // sgitg % tgQuads picks the quad, sgitg / tgQuads the K range. 4x2 was
    // "SPLIT_K_WIDE" and 2x2 was "SPLIT_K_2" -- both split K by 2, so the old
    // names described the threadgroup width while sounding like K depth.
    // 8- and 16-quad TGs lose: the parallelism the larger threadgroups give
    // up costs more than the LN prologue read they amortize.
#if defined(GEMV_2OCQUAD_PER_SG)
    // Two oc quads per simdgroup, carried as two accumulator streams in the same
    // thread: doubles the weight reads in flight with no barrier and shares the
    // input read plus the LN prologue between them. 2 simdgroups, 4 quads/TG.
    //
    // GATE_UP_SILU is the one shape the macro name does not describe: there the
    // second stream is the up matrix at the SAME quad, not a second quad, so the
    // threadgroup owns 2 quads rather than 4.
    //
    // GEMV_2OCQUAD_PER_SG_SPLIT_K keeps the quads either way and pairs up
    // 2 * factor simdgroups over the K range.
#ifdef GEMV_2OCQUAD_PER_SG_SPLIT_K
    constexpr int tg_simds  = 2 * GEMV_2OCQUAD_PER_SG_SPLIT_K;
#else
    constexpr int tg_simds  = 2;
#endif
#ifdef GATE_UP_SILU
    constexpr int tgQuads = 2;
#else
    constexpr int tgQuads = 4;
#endif
#else
#ifdef GEMV_QUADS_PER_TG
    constexpr int tgQuads = GEMV_QUADS_PER_TG;
#else
    constexpr int tgQuads = 2;
#endif
#ifdef GEMV_SPLIT_K
    constexpr int tgSplitK = GEMV_SPLIT_K;
#else
    constexpr int tgSplitK = 1;
#endif
    constexpr int tg_simds = tgQuads * tgSplitK;
#endif

#ifdef LN_FUSED
    // Lowest output quad this threadgroup owns; see the uz derivations below.
    // The test is threadgroup-uniform, so a threadgroup with no work at all can
    // still leave before the barrier. Threadgroups that keep only some of their
    // simdgroups must not: those stragglers are held back until after the
    // reduction (their per-simdgroup check sits below).
    const int tg_first_uz = tg_x * tgQuads;
#ifdef QKV_FUSED
    if (tg_first_uz >= qkvOutputDepthQuad) return;
#else
    if (tg_first_uz >= cst.outputDepthQuad) return;
#endif

    // Hoisted above the per-simdgroup output-range checks below: the reduction
    // barrier has to be reached by every thread in the threadgroup, and a tail
    // threadgroup can have one simdgroup land outside outputDepthQuad.
    //
    // inv_rms is the same scalar for both simdgroups, so sweep the input once
    // with all 64 threads and reduce through threadgroup memory, instead of
    // letting each simdgroup re-read the whole input+residual to recompute it.
    // The host sizes the threadgroup at tg_simds simdgroups (2, or more under
    // GEMV_2OCQUAD_PER_SG_SPLIT_K, or GEMV_QUADS_PER_TG * GEMV_SPLIT_K) and the
    // sweep/reduce below follow that count.
    //
    // Only one threadgroup writes ln_residual_out, to avoid races when several
    // threadgroups process the same input quads; both of its simdgroups take
    // part since they now cover disjoint halves of the input.
    float sq_sum = 0.0f;
#ifdef LN_STAGE
    // The sweep below already touches every input quad once per threadgroup, so
    // keep (in + residual) * gamma there instead of letting the GEMV body re-read
    // both device streams for each quad and K half it owns (4x here). At 256
    // threadgroups per layer that redundancy is a multiple of the weight traffic.
    // inv_rms is not known until the sweep finishes, so it is applied on read.
    // Stored as ftype4: in fp16 mode that is half the threadgroup footprint of
    // the FLOAT4 it replaces (4KB vs 8KB here, one fewer occupancy clamp), and
    // it rounds the GEMM input exactly like the unfused path does (AddRMSNorm
    // writes an fp16 tensor), so the numerics match the separate-dispatch
    // baseline the fusion replaces. LN_STAGE_FP32 restores the float4 staging
    // (host rollback switch).
#ifdef LN_STAGE_FP32
    threadgroup FLOAT4 ln_staged[LN_STAGE_QUADS];
#else
    threadgroup ftype4 ln_staged[LN_STAGE_QUADS];
#endif
#endif
    const bool ln_write_residual = (
#if defined(GATE_UP_FUSED) || defined(QKV_FUSED)
        gid.z == 0 && gid.x == 0
#else
        gid.x == 0
#endif
    );
    for (int z = (int)(sgitg * 32 + tiisg); z < cst.inputDepthQuad; z += 32 * tg_simds) {
        float4 d = (float4)*(in + z * area_size) + (float4)*(ln_residual_in + z * area_size);
        sq_sum += dot(d, d);
#ifdef LN_STAGE
#ifdef LN_STAGE_FP32
        ln_staged[z] = (FLOAT4)d * (FLOAT4)ln_gamma[z];
#else
        ln_staged[z] = (ftype4)((FLOAT4)d * (FLOAT4)ln_gamma[z]);
#endif
#endif
        if (ln_write_residual) {
            ln_residual_out[z * area_size] = (ftype4)d;
        }
    }
    sq_sum = simd_sum(sq_sum);
    threadgroup float ln_sq_partial[tg_simds];
    if (tiisg == 0) {
        ln_sq_partial[sgitg] = sq_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sq_sum = 0.0f;
    for (int s = 0; s < tg_simds; ++s) {
        sq_sum += ln_sq_partial[s];
    }
    const float inv_rms = rsqrt(sq_sum / (float)(cst.inputDepthQuad * 4) + *ln_eps);
#endif

// Shared input load for every quant branch below: LN_FUSED applies the fused
// RMSNorm transform, plain read otherwise. Kernel-local macro (undef'd at the
// end of this kernel) so the six loop bodies don't each carry the #ifdef pair.
#ifdef LN_STAGE
#define GEMV_2SG_LOAD_IN4(z) ((FLOAT4)ln_staged[(z)] * (FLOAT)inv_rms)
#elif defined(LN_FUSED)
#define GEMV_2SG_LOAD_IN4(z) (((FLOAT4)*(xy_in0 + (z) * area_size) + (FLOAT4)*(ln_residual_in + (z) * area_size)) * inv_rms * (FLOAT4)ln_gamma[(z)])
#else
#define GEMV_2SG_LOAD_IN4(z) ((FLOAT4)*(xy_in0 + (z) * area_size))
#endif

#ifdef GEMV_2OCQUAD_PER_SG
    // Dual-stream variant (fused pipelines only): each simdgroup carries TWO
    // independent accumulator streams — doubles in-flight weight reads with no
    // barrier (cf. the split-K tg reduce) and shares the input read + LN
    // prologue across both streams. By default the streams are two adjacent
    // output quads of one matrix.
#ifdef GATE_UP_SILU
    // SwiGLU variant: the two streams drive gate and up for the SAME
    // quad instead of two adjacent quads of one matrix, so the epilogue can
    // emit up * silu(gate) itself and the separate MUL_SILU dispatch -- plus the
    // gate/up round trip through memory -- disappears. Parallelism is untouched:
    // grid.x doubles as grid.z drops from 2 to 1, so both the threadgroup count
    // and the 4 quad-dots each threadgroup issues stay exactly as they were.
#ifdef GEMV_2OCQUAD_PER_SG_SPLIT_K
    // K-split on top of the dual streams: 2 * GEMV_2OCQUAD_PER_SG_SPLIT_K simdgroups
    // instead of 2. Every simdgroup pair covers the threadgroup's two quads,
    // and pair p sweeps quant-block range p of GEMV_2OCQUAD_PER_SG_SPLIT_K; partials are merged
    // through threadgroup memory in the epilogue. The dispatch is what limits
    // these small fused GEMVs (parallelism, not DRAM), so widening the
    // threadgroup multiplies the loads in flight per quad. The host only
    // enables this when outputDepthQuad is even and blockCount divides by the
    // factor, so nothing early-returns before the barrier below.
    const int uz = tg_x * 2 + ((int)sgitg & 1);
    const int ds_sk = (int)sgitg >> 1;
#else
    const int uz = tg_x * 2 + (int)sgitg;
#endif
    if (uz >= cst.outputDepthQuad) return;
    const bool stream1_valid = true;
    const int  uz1 = uz;            // stream 1 is the up matrix at the same quad
    const float coef0 = cst.scaleCoef;
    const float coef1 = gate_up_seg[0];
#else
    const int uz = (tg_x * 2 + (int)sgitg) * 2;
#ifdef QKV_FUSED
    if (uz >= qkvOutputDepthQuad) return;
    const bool stream1_valid = (uz + 1) < qkvOutputDepthQuad;
    float cur_scale_coef = qkv_scale_coef;
#else
    if (uz >= cst.outputDepthQuad) return;
    const bool stream1_valid = (uz + 1) < cst.outputDepthQuad;
    float cur_scale_coef = cst.scaleCoef;
#endif
    #ifdef GATE_UP_FUSED
    if (gid.z == 1) {
        cur_scale_coef = gate_up_seg[0];
    }
    #endif

    // Invalid stream 1 aliases stream 0 (safe reads); its result is discarded.
    const int uz1 = stream1_valid ? (uz + 1) : uz;
    const float coef0 = cur_scale_coef;
    const float coef1 = cur_scale_coef;
#endif
    // Hoist the per-bi scale/dbias divisions out of the loop: coef is
    // loop-invariant, so multiply by its reciprocal instead.
    const float inv_coef0 = 1.0f / coef0;
    const float inv_coef1 = 1.0f / coef1;
#ifdef W_QUANT_3
    // 6-byte (4 OC x 4 IC) tiles: row stride is 6x the ic_4 count.
    const int wt_row_stride = cst.inputDepthQuad * 6;
#else
    const int wt_row_stride = cst.inputDepthQuad;
#endif
    auto xy_wt0 = wt + uz * wt_row_stride;
    auto xy_in0 = in;
    auto biasValue0 = FLOAT4(biasTerms[uz]);
#ifdef GATE_UP_SILU
    auto xy_wt1 = wt_up + uz1 * wt_row_stride;
    auto biasValue1 = FLOAT4(biasTerms_up[uz1]);
    const device ftype4* dqs1 = dequantScale_up;
#else
    auto xy_wt1 = wt + uz1 * wt_row_stride;
    auto biasValue1 = FLOAT4(biasTerms[uz1]);
    const device ftype4* dqs1 = dequantScale;
#endif

#ifdef GEMV_QBLOCK_W16_DS
    // Dual-stream W16: the host picked GEMV_QBLOCK_W16_LANES_PER_BLOCK from the
    // post-split block count (blockCount / GEMV_2OCQUAD_PER_SG_SPLIT_K), the same rule the
    // runtime widening below follows, so compile-time lanes make the uint4 pair
    // loop in the W4 branch match laneInBlock/blockSlot exactly.
    constexpr int quadsPerBlock = GEMV_QBLOCK_W16_QUADS_PER_BLOCK;
    constexpr int lanesPerBlock = GEMV_QBLOCK_W16_LANES_PER_BLOCK;
    constexpr int blocksInFlight = SIMD_GROUP_WIDTH / lanesPerBlock;
    const int laneInBlock = int(tiisg) & (lanesPerBlock - 1);
    const int blockSlot  = int(tiisg) / lanesPerBlock;
    const int ds_blocks_owned = max(cst.blockCount / GEMV_2OCQUAD_PER_SG_SPLIT_K, 1);
#else
    int quadsPerBlock = (cst.inputDepthQuad + cst.blockCount - 1) / cst.blockCount;
    int lanesPerBlock = min(SIMD_GROUP_WIDTH, max(quadsPerBlock / 4, 1));
#ifdef GEMV_2OCQUAD_PER_SG_SPLIT_K
    // A pair now owns only blockCount / GEMV_2OCQUAD_PER_SG_SPLIT_K blocks. Widen the lanes per
    // block so blocksInFlight never exceeds that count -- otherwise the surplus
    // lane groups sit idle instead of shortening each lane's run.
    const int ds_blocks_owned = max(cst.blockCount / GEMV_2OCQUAD_PER_SG_SPLIT_K, 1);
    lanesPerBlock = max(lanesPerBlock, SIMD_GROUP_WIDTH / ds_blocks_owned);
    lanesPerBlock = min(lanesPerBlock, SIMD_GROUP_WIDTH);
#endif
    int blocksInFlight = SIMD_GROUP_WIDTH / lanesPerBlock;
    int laneInBlock = tiisg % lanesPerBlock;
    int blockSlot  = tiisg / lanesPerBlock;
#endif

    FLOAT4 result0 = FLOAT4(0);
    FLOAT4 result1 = FLOAT4(0);

#ifdef GEMV_2OCQUAD_PER_SG_SPLIT_K
    const int ds_bi_begin = ds_sk * ds_blocks_owned;
    const int ds_bi_end   = ds_bi_begin + ds_blocks_owned;
#else
    const int ds_bi_begin = 0;
    const int ds_bi_end   = cst.blockCount;
#endif

    for (int bi = ds_bi_begin + blockSlot; bi < ds_bi_end; bi += blocksInFlight) {
        FLOAT4 scale0 = FLOAT4(dequantScale[2 * (uz * cst.blockCount + bi) + 0]) * (FLOAT)inv_coef0;
        FLOAT4 dbias0 = FLOAT4(dequantScale[2 * (uz * cst.blockCount + bi) + 1]) * (FLOAT)inv_coef0;
        FLOAT4 scale1 = FLOAT4(dqs1[2 * (uz1 * cst.blockCount + bi) + 0]) * (FLOAT)inv_coef1;
        FLOAT4 dbias1 = FLOAT4(dqs1[2 * (uz1 * cst.blockCount + bi) + 1]) * (FLOAT)inv_coef1;
        int zmin = bi * quadsPerBlock;
        int zmax = min(zmin + quadsPerBlock, cst.inputDepthQuad);

    #if defined(W_QUANT_2) || defined(W_QUANT_3)
        // Deferred + pre-scaling dual streams (see g8 kernel for the
        // mask/pre-scale derivation). Trap-A: must precede the W4 branch.
        FLOAT4 raw_dot0 = FLOAT4(0);
        FLOAT4 raw_dot1 = FLOAT4(0);
        FLOAT input_sum = FLOAT(0);
        for (int z = zmin + laneInBlock; z < zmax; z += lanesPerBlock) {
            FLOAT4 in4 = GEMV_2SG_LOAD_IN4(z);
            input_sum += dot(in4, FLOAT4(1.0));

            FLOAT in_ps0 = in4[0] * FLOAT(1.0/64.0);
            FLOAT in_ps1 = in4[1] * FLOAT(1.0/16.0);
            FLOAT in_ps2 = in4[2] * FLOAT(1.0/4.0);
        #ifdef W_QUANT_2
            // Vectorized mask-only unpack: 4 vector FMAs per row instead of 16
            // scalar ones; elementwise identical to the scalar form.
            uchar4 wA = xy_wt0[z];
            uchar4 wB = xy_wt1[z];
            raw_dot0 += in_ps0 * FLOAT4(wA & uchar4(0xC0)) + in_ps1 * FLOAT4(wA & uchar4(0x30))
                      + in_ps2 * FLOAT4(wA & uchar4(0x0C)) + in4[3] * FLOAT4(wA & uchar4(0x03));
            raw_dot1 += in_ps0 * FLOAT4(wB & uchar4(0xC0)) + in_ps1 * FLOAT4(wB & uchar4(0x30))
                      + in_ps2 * FLOAT4(wB & uchar4(0x0C)) + in4[3] * FLOAT4(wB & uchar4(0x03));
        #else
            FLOAT in_hi0 = in4[0] * FLOAT(0.5);
            FLOAT in_hi2 = in4[2] * FLOAT(2.0);
            FLOAT in_hi3 = in4[3] * FLOAT(4.0);
            const device ushort* tA = (const device ushort*)(xy_wt0 + z * 6);
            const device ushort* tB = (const device ushort*)(xy_wt1 + z * 6);
            ushort sa0 = tA[0], sa1 = tA[1], sa2 = tA[2];
            ushort sb0 = tB[0], sb1 = tB[1], sb2 = tB[2];
            // Vectorized: gather the 4 lo bytes / 4 hi nibbles per row into
            // uchar4 and mask as vectors (see the W2 branch above).
            uchar4 loA = uchar4(uchar(sa0), uchar(sa0 >> 8), uchar(sa1), uchar(sa1 >> 8));
            uchar hab0 = uchar(sa2), hab1 = uchar(sa2 >> 8);
            uchar4 hiA = uchar4(hab0 >> 4, hab0 & 0xF, hab1 >> 4, hab1 & 0xF);
            uchar4 loB = uchar4(uchar(sb0), uchar(sb0 >> 8), uchar(sb1), uchar(sb1 >> 8));
            uchar hbb0 = uchar(sb2), hbb1 = uchar(sb2 >> 8);
            uchar4 hiB = uchar4(hbb0 >> 4, hbb0 & 0xF, hbb1 >> 4, hbb1 & 0xF);
            raw_dot0 += in_ps0 * FLOAT4(loA & uchar4(0xC0)) + in_ps1 * FLOAT4(loA & uchar4(0x30))
                      + in_ps2 * FLOAT4(loA & uchar4(0x0C)) + in4[3] * FLOAT4(loA & uchar4(0x03))
                      + in_hi0 * FLOAT4(hiA & uchar4(0x8)) + in4[1] * FLOAT4(hiA & uchar4(0x4))
                      + in_hi2 * FLOAT4(hiA & uchar4(0x2)) + in_hi3 * FLOAT4(hiA & uchar4(0x1));
            raw_dot1 += in_ps0 * FLOAT4(loB & uchar4(0xC0)) + in_ps1 * FLOAT4(loB & uchar4(0x30))
                      + in_ps2 * FLOAT4(loB & uchar4(0x0C)) + in4[3] * FLOAT4(loB & uchar4(0x03))
                      + in_hi0 * FLOAT4(hiB & uchar4(0x8)) + in4[1] * FLOAT4(hiB & uchar4(0x4))
                      + in_hi2 * FLOAT4(hiB & uchar4(0x2)) + in_hi3 * FLOAT4(hiB & uchar4(0x1));
        #endif
        }
        #ifdef W_QUANT_2
        FLOAT4 adj0 = dbias0 - FLOAT(2.0) * scale0;
        FLOAT4 adj1 = dbias1 - FLOAT(2.0) * scale1;
        #else
        FLOAT4 adj0 = dbias0 - FLOAT(4.0) * scale0;
        FLOAT4 adj1 = dbias1 - FLOAT(4.0) * scale1;
        #endif
        result0 += raw_dot0 * scale0 + input_sum * adj0;
        result1 += raw_dot1 * scale1 + input_sum * adj1;
    #elif defined(W_QUANT_4)
    #ifdef GEMV_QBLOCK_W16_DS
        // uint4 (16B) weight reads for the dual-stream body, mirroring the
        // single-stream W16 specialization: the two weight streams (gate/up)
        // share the input quads this lane owns.
        FLOAT4 raw_dot0 = FLOAT4(0);
        FLOAT4 raw_dot1 = FLOAT4(0);
        FLOAT input_sum = FLOAT(0);
        const device uint4* wt16_0 = (const device uint4*)(xy_wt0 + zmin);
        const device uint4* wt16_1 = (const device uint4*)(xy_wt1 + zmin);
        constexpr int uint4PairsPerLane = (quadsPerBlock / 2) / lanesPerBlock;
        const int pair_begin = laneInBlock * uint4PairsPerLane;
        for (int pair = pair_begin; pair < pair_begin + uint4PairsPerLane; ++pair) {
            const uint4 packed0 = wt16_0[pair];
            const uint4 packed1 = wt16_1[pair];
            const ushort4 w00 = ushort4(packed0.x & 0xFFFFu, packed0.x >> 16, packed0.y & 0xFFFFu, packed0.y >> 16);
            const ushort4 w01 = ushort4(packed0.z & 0xFFFFu, packed0.z >> 16, packed0.w & 0xFFFFu, packed0.w >> 16);
            const ushort4 w10 = ushort4(packed1.x & 0xFFFFu, packed1.x >> 16, packed1.y & 0xFFFFu, packed1.y >> 16);
            const ushort4 w11 = ushort4(packed1.z & 0xFFFFu, packed1.z >> 16, packed1.w & 0xFFFFu, packed1.w >> 16);
            const FLOAT4 input0 = GEMV_2SG_LOAD_IN4(zmin + 2 * pair);
            const FLOAT4 input1 = GEMV_2SG_LOAD_IN4(zmin + 2 * pair + 1);
            input_sum += input0[0] + input0[1] + input0[2] + input0[3] +
                         input1[0] + input1[1] + input1[2] + input1[3];
            raw_dot0 += input0[0] * FLOAT(1.0/16.0) * FLOAT4(w00 & ushort4(0x00F0)) +
                        input0[1] * FLOAT4(w00 & ushort4(0x000F)) +
                        input0[2] * FLOAT(1.0/4096.0) * FLOAT4(w00 & ushort4(0xF000)) +
                        input0[3] * FLOAT(1.0/256.0) * FLOAT4(w00 & ushort4(0x0F00)) +
                        input1[0] * FLOAT(1.0/16.0) * FLOAT4(w01 & ushort4(0x00F0)) +
                        input1[1] * FLOAT4(w01 & ushort4(0x000F)) +
                        input1[2] * FLOAT(1.0/4096.0) * FLOAT4(w01 & ushort4(0xF000)) +
                        input1[3] * FLOAT(1.0/256.0) * FLOAT4(w01 & ushort4(0x0F00));
            raw_dot1 += input0[0] * FLOAT(1.0/16.0) * FLOAT4(w10 & ushort4(0x00F0)) +
                        input0[1] * FLOAT4(w10 & ushort4(0x000F)) +
                        input0[2] * FLOAT(1.0/4096.0) * FLOAT4(w10 & ushort4(0xF000)) +
                        input0[3] * FLOAT(1.0/256.0) * FLOAT4(w10 & ushort4(0x0F00)) +
                        input1[0] * FLOAT(1.0/16.0) * FLOAT4(w11 & ushort4(0x00F0)) +
                        input1[1] * FLOAT4(w11 & ushort4(0x000F)) +
                        input1[2] * FLOAT(1.0/4096.0) * FLOAT4(w11 & ushort4(0xF000)) +
                        input1[3] * FLOAT(1.0/256.0) * FLOAT4(w11 & ushort4(0x0F00));
        }
    #else
        FLOAT4 raw_dot0 = FLOAT4(0);
        FLOAT4 raw_dot1 = FLOAT4(0);
        FLOAT input_sum = FLOAT(0);
        for (int z = zmin + laneInBlock; z < zmax; z += lanesPerBlock) {
            FLOAT4 in4 = GEMV_2SG_LOAD_IN4(z);
            input_sum += dot(in4, FLOAT4(1.0));

            FLOAT in_ps0 = in4[0] * FLOAT(1.0/16.0);
            FLOAT in_ps2 = in4[2] * FLOAT(1.0/4096.0);
            FLOAT in_ps3 = in4[3] * FLOAT(1.0/256.0);

            // Vectorized nibble unpack (hot path): 4 vector FMAs per row
            // instead of 16 scalar ones; elementwise identical.
            ushort4 wA = xy_wt0[z];
            ushort4 wB = xy_wt1[z];
            raw_dot0 += in_ps0 * FLOAT4(wA & ushort4(0x00F0)) + in4[1] * FLOAT4(wA & ushort4(0x000F))
                      + in_ps2 * FLOAT4(wA & ushort4(0xF000)) + in_ps3 * FLOAT4(wA & ushort4(0x0F00));
            raw_dot1 += in_ps0 * FLOAT4(wB & ushort4(0x00F0)) + in4[1] * FLOAT4(wB & ushort4(0x000F))
                      + in_ps2 * FLOAT4(wB & ushort4(0xF000)) + in_ps3 * FLOAT4(wB & ushort4(0x0F00));
        }
    #endif
        FLOAT4 adj0 = dbias0 - FLOAT(8.0) * scale0;
        FLOAT4 adj1 = dbias1 - FLOAT(8.0) * scale1;
        result0 += raw_dot0 * scale0 + input_sum * adj0;
        result1 += raw_dot1 * scale1 + input_sum * adj1;
    #elif defined(W_QUANT_8)
        // w[j] holds output lane j's 4 ic weights (transform packs ro*4+ri),
        // so each lane needs dot(in4, w[j]) with lane j's scale/bias. Deferred
        // like the W4 branch: raw dot + input_sum, dequant applied per block.
        // (A per-z `result += in4[i] * (w[i]*scale[i]+bias[i])` loop is the
        // TRANSPOSED product with the wrong scale lane -- W8 regression trap.)
        FLOAT4 raw_dot0 = FLOAT4(0);
        FLOAT4 raw_dot1 = FLOAT4(0);
        FLOAT input_sum = FLOAT(0);
        for (int z = zmin + laneInBlock; z < zmax; z += lanesPerBlock) {
            auto wA = xy_wt0[z];
            auto wB = xy_wt1[z];
            FLOAT4 in4 = GEMV_2SG_LOAD_IN4(z);
            input_sum += dot(in4, FLOAT4(1.0));
            raw_dot0 += FLOAT4(dot(in4, FLOAT4(wA[0])), dot(in4, FLOAT4(wA[1])),
                               dot(in4, FLOAT4(wA[2])), dot(in4, FLOAT4(wA[3])));
            raw_dot1 += FLOAT4(dot(in4, FLOAT4(wB[0])), dot(in4, FLOAT4(wB[1])),
                               dot(in4, FLOAT4(wB[2])), dot(in4, FLOAT4(wB[3])));
        }
        result0 += raw_dot0 * scale0 + input_sum * dbias0;
        result1 += raw_dot1 * scale1 + input_sum * dbias1;
    #endif
    }

    result0 = simd_sum(result0);
    result1 = simd_sum(result1);
#ifdef GEMV_2OCQUAD_PER_SG_SPLIT_K
    threadgroup FLOAT4 ds_partial[2][2 * GEMV_2OCQUAD_PER_SG_SPLIT_K];
    const int ds_slot = (int)sgitg & 1;
    if (tiisg == 0) {
        ds_partial[0][sgitg] = result0;
        ds_partial[1][sgitg] = result1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool ds_write = (ds_sk == 0 && tiisg == 0);
    if (ds_write) {
        result0 = FLOAT4(0);
        result1 = FLOAT4(0);
        for (int s = 0; s < GEMV_2OCQUAD_PER_SG_SPLIT_K; ++s) {
            result0 += ds_partial[0][s * 2 + ds_slot];
            result1 += ds_partial[1][s * 2 + ds_slot];
        }
    }
#else
    const bool ds_write = (tiisg == 0);
#endif
    if (ds_write) {
#ifdef GATE_UP_SILU
        // Matches the op this replaces (MUL_SILU: in0 = up, in1 = gate), applied
        // after each projection's own activation, exactly as the unfused
        // conv -> conv -> binary chain did.
        FLOAT4 gateV = FLOAT4(activate(ftype4(result0 + biasValue0), cst.activation));
        FLOAT4 upV   = FLOAT4(activate(ftype4(result1 + biasValue1), cst.activation));
        GEMV_OUT[GEMV_OUT_BASE + uz * area_size] = ftype4(upV * (gateV / (FLOAT4(1) + exp(-gateV))));
#else
        GEMV_OUT[GEMV_OUT_BASE + uz * area_size] = activate(ftype4(result0 + biasValue0), cst.activation);
        if (stream1_valid) {
            GEMV_OUT[GEMV_OUT_BASE + uz1 * area_size] = activate(ftype4(result1 + biasValue1), cst.activation);
        }
#endif
    }
#else  // !GEMV_2OCQUAD_PER_SG

    // Single stream: each simdgroup owns one output quad and one K range.
#if GEMV_SPLIT_K > 1
    // Split-K: sgitg % tgQuads picks the output quad, sgitg / tgQuads the K
    // range; partials combined via threadgroup memory. Host guarantees the grid is
    // exact (oc % (4 * tgQuads) == 0) so no simdgroup early-returns before the
    // barrier below.
    const int uz = tg_x * tgQuads + ((int)sgitg % tgQuads);
    const int sk_half = (int)sgitg / tgQuads;
#else
    // tgQuads = GEMV_QUADS_PER_TG (2, or 4 for the tg4 decode variant); each
    // simdgroup owns the quad matching its index. Hardcoding 2 here left the
    // upper half of every tg4 output unwritten (stale memory -> garbage decode).
    const int uz = tg_x * tgQuads + (int)sgitg;
#endif
#ifdef QKV_FUSED
    if (uz >= qkvOutputDepthQuad) return;
    float cur_scale_coef = qkv_scale_coef;
#else
    if (uz >= cst.outputDepthQuad) return;
    float cur_scale_coef = cst.scaleCoef;
#endif
    #ifdef GATE_UP_FUSED
    // Gate uses cst.scaleCoef (leader's), up uses its own scaleCoef via gate_up_seg[0].
    // Without this, up's dequant is scaled by gate's coefficient -> systematic bias
    // whenever gate/up weights have different fp16-fit ranges (visible on Qwen3.5-2B
    // as decode drift into repetition / low-quality output).
    if (gid.z == 1) {
        cur_scale_coef = gate_up_seg[0];
    }
    #endif
    // Hoist the per-bi scale/bias divisions out of the loop: the coef is
    // loop-invariant, so multiply by its reciprocal instead.
    const float inv_scale_coef = 1.0f / cur_scale_coef;

#ifdef W_QUANT_3
    auto xy_wt = wt + uz * cst.inputDepthQuad * 6;
#else
    auto xy_wt = wt + uz * cst.inputDepthQuad;
#endif
    auto xy_in0 = in;
    auto biasValue = FLOAT4(biasTerms[uz]);

#ifdef GEMV_QBLOCK_W16
    // Q4 block 32/64/128/256 specialization for decode GEMV. The host supplies
    // the exact C4 quads per quant block (8/16/32/64), so the inner extent is
    // compile-time constant and no tail check is needed.
    //
    // GEMV_QBLOCK_W16_LANES_PER_BLOCK = how many lanes share one quant block. The host
    // picks it per pipeline from the blocks a simdgroup owns (blockCount, or
    // blockCount/2 under split-K) so that all 32 lanes stay fed: too few lanes
    // per block idles lanes on short rows, too many shortens each lane's
    // contiguous run. See chooseQ4W16LanesPerBlock in MetalConvolution1x1.mm.
#ifndef GEMV_QBLOCK_W16_QUADS_PER_BLOCK
#define GEMV_QBLOCK_W16_QUADS_PER_BLOCK 16
#endif
#ifndef GEMV_QBLOCK_W16_LANES_PER_BLOCK
#define GEMV_QBLOCK_W16_LANES_PER_BLOCK 1
#endif
    constexpr int quadsPerBlock = GEMV_QBLOCK_W16_QUADS_PER_BLOCK;
    constexpr int lanesPerBlock = GEMV_QBLOCK_W16_LANES_PER_BLOCK;
    constexpr int blocksInFlight = SIMD_GROUP_WIDTH / lanesPerBlock;
    const int laneInBlock = int(tiisg) & (lanesPerBlock - 1);
    const int blockSlot = int(tiisg) / lanesPerBlock;
#else
    int quadsPerBlock = (cst.inputDepthQuad + cst.blockCount - 1) / cst.blockCount;
    // GEMV inner reduction lane partitioning.
    //
    // GEMV_LANES_PER_BLOCK (split-K pipelines): host-computed, see gemvLanesPerBlock().
    //
    // Fallback (M4/Apple GPU 7-8): min(32, max(quadsPerBlock/4, 1))
    //   For Qwen3-0.6B blockCount=32, inputDepthQuad/32≈8 => lanesPerBlock=2, blocksInFlight=16.
    //   Only 2 lanes participate in inner K reduction; other 30 lanes iterate outer
    //   blocks. Fine on narrower SM (M4) where BW is the bottleneck.
#ifdef GEMV_LANES_PER_BLOCK
    // Every input is known at pipeline build time, and keeping it a literal lets
    // the compiler strength-reduce the %/÷ below instead of carrying a runtime
    // divisor through the hot path.
    constexpr int lanesPerBlock = GEMV_LANES_PER_BLOCK;
    int blocksInFlight = SIMD_GROUP_WIDTH / lanesPerBlock;
    int laneInBlock = tiisg % lanesPerBlock;
    int blockSlot = tiisg / lanesPerBlock;
#else
    int lanesPerBlock = min(SIMD_GROUP_WIDTH, max(quadsPerBlock / 4, 1));
    int blocksInFlight = SIMD_GROUP_WIDTH / lanesPerBlock;
    int laneInBlock = tiisg % lanesPerBlock;
    int blockSlot = tiisg / lanesPerBlock;
#endif
#endif // GEMV_QBLOCK_W16

    FLOAT4 result = FLOAT4(0);

#if GEMV_SPLIT_K > 1
    const int sk_bi_begin = sk_half * (cst.blockCount / 2);
    const int sk_bi_end   = (sk_half == 1) ? cst.blockCount : (cst.blockCount / 2);
    for (int bi = sk_bi_begin + blockSlot; bi < sk_bi_end; bi += blocksInFlight) {
#else
    for (int bi = blockSlot; bi < cst.blockCount; bi += blocksInFlight) {
#endif
        FLOAT4 scale = FLOAT4(dequantScale[2 * (uz * cst.blockCount + bi) + 0]) * (FLOAT)inv_scale_coef;
        FLOAT4 dequant_bias = FLOAT4(dequantScale[2 * (uz * cst.blockCount + bi) + 1]) * (FLOAT)inv_scale_coef;
        int zmin = bi * quadsPerBlock;
        #ifdef GEMV_QBLOCK_W16
        int zmax = zmin + quadsPerBlock;
        #else
        int zmax = min(zmin + quadsPerBlock, cst.inputDepthQuad);
        #endif

    #if defined(W_QUANT_2) || defined(W_QUANT_3)
        // Deferred + pre-scaling (see g8 kernel for the mask/pre-scale
        // derivation). Trap-A: must precede the W4 branch.
        FLOAT4 raw_dot = FLOAT4(0);
        FLOAT input_sum = FLOAT(0);
        for (int z = zmin + laneInBlock; z < zmax; z += lanesPerBlock) {
            FLOAT4 in4 = GEMV_2SG_LOAD_IN4(z);
            input_sum += dot(in4, FLOAT4(1.0));

            FLOAT in_ps0 = in4[0] * FLOAT(1.0/64.0);
            FLOAT in_ps1 = in4[1] * FLOAT(1.0/16.0);
            FLOAT in_ps2 = in4[2] * FLOAT(1.0/4.0);
        #ifdef W_QUANT_2
            // Vectorized mask-only unpack (see the GEMV_2OCQUAD_PER_SG body above).
            uchar4 w_b = xy_wt[z];
            raw_dot += in_ps0 * FLOAT4(w_b & uchar4(0xC0)) + in_ps1 * FLOAT4(w_b & uchar4(0x30))
                     + in_ps2 * FLOAT4(w_b & uchar4(0x0C)) + in4[3] * FLOAT4(w_b & uchar4(0x03));
        #else
            FLOAT in_hi0 = in4[0] * FLOAT(0.5);
            FLOAT in_hi2 = in4[2] * FLOAT(2.0);
            FLOAT in_hi3 = in4[3] * FLOAT(4.0);
            const device ushort* t = (const device ushort*)(xy_wt + z * 6);
            ushort tw0 = t[0], tw1 = t[1], tw2 = t[2];
            // Vectorized lo-byte / hi-nibble unpack (see the GEMV_2OCQUAD_PER_SG body above).
            uchar4 lo = uchar4(uchar(tw0), uchar(tw0 >> 8), uchar(tw1), uchar(tw1 >> 8));
            uchar hb0 = uchar(tw2), hb1 = uchar(tw2 >> 8);
            uchar4 hi = uchar4(hb0 >> 4, hb0 & 0xF, hb1 >> 4, hb1 & 0xF);
            raw_dot += in_ps0 * FLOAT4(lo & uchar4(0xC0)) + in_ps1 * FLOAT4(lo & uchar4(0x30))
                     + in_ps2 * FLOAT4(lo & uchar4(0x0C)) + in4[3] * FLOAT4(lo & uchar4(0x03))
                     + in_hi0 * FLOAT4(hi & uchar4(0x8)) + in4[1] * FLOAT4(hi & uchar4(0x4))
                     + in_hi2 * FLOAT4(hi & uchar4(0x2)) + in_hi3 * FLOAT4(hi & uchar4(0x1));
        #endif
        }
        #ifdef W_QUANT_2
        FLOAT4 adjusted_bias = dequant_bias - FLOAT(2.0) * scale;
        #else
        FLOAT4 adjusted_bias = dequant_bias - FLOAT(4.0) * scale;
        #endif
        result += raw_dot * scale + input_sum * adjusted_bias;
    #elif defined(W_QUANT_4)
        FLOAT4 raw_dot = FLOAT4(0);
        FLOAT input_sum = FLOAT(0);

        // NOTE: Adding `#pragma clang loop unroll_count(N)` here (2x and 4x)
        // is a net negative. The Metal compiler already
        // schedules this compact 4-row loop body optimally; forcing unroll
        // increases register pressure and hurts simdgroup occupancy in the
        // decode (GEMV) regime. Leave the loop unannotated.
#ifdef GEMV_QBLOCK_W16
        // Read weights as uint4 (16 bytes, two C4 quads) and keep the same
        // pre-scaling mask math. The host also enables this branch for eligible
        // LN/QKV/GateUp fused non-GEMV_2OCQUAD_PER_SG pipelines.
        const device uint4* wt16 = (const device uint4*)(xy_wt + zmin);
        const device ftype4* in16 = xy_in0 + zmin * area_size;
        // lanesPerBlock lanes share a block evenly.
        constexpr int uint4PairsPerLane = (quadsPerBlock / 2) / lanesPerBlock;
        const int pair_begin = laneInBlock * uint4PairsPerLane;
        for (int pair = pair_begin; pair < pair_begin + uint4PairsPerLane; ++pair) {
            const uint4 packed_weight = wt16[pair];
            const ushort4 weight0 = ushort4(packed_weight.x & 0xFFFFu, packed_weight.x >> 16,
                                            packed_weight.y & 0xFFFFu, packed_weight.y >> 16);
            const ushort4 weight1 = ushort4(packed_weight.z & 0xFFFFu, packed_weight.z >> 16,
                                            packed_weight.w & 0xFFFFu, packed_weight.w >> 16);
        #ifdef LN_FUSED
            // Fused into the block-input RMSNorm (has_ln): the same
            // raw*inv_rms*gamma transform the legacy loop applies, on the two
            // quads this lane owns. zmin+2*pair / +1 are the absolute quads.
            const int zln0 = zmin + 2 * pair;
        #ifdef LN_STAGE
            const FLOAT4 input0 = GEMV_2SG_LOAD_IN4(zln0);
            const FLOAT4 input1 = GEMV_2SG_LOAD_IN4(zln0 + 1);
        #else
            const FLOAT4 input0 = ((FLOAT4)in16[(2 * pair) * area_size]
                                   + (FLOAT4)*(ln_residual_in + zln0 * area_size))
                                  * inv_rms * (FLOAT4)ln_gamma[zln0];
            const FLOAT4 input1 = ((FLOAT4)in16[(2 * pair + 1) * area_size]
                                   + (FLOAT4)*(ln_residual_in + (zln0 + 1) * area_size))
                                  * inv_rms * (FLOAT4)ln_gamma[zln0 + 1];
        #endif
        #else
            const FLOAT4 input0 = FLOAT4(in16[(2 * pair) * area_size]);
            const FLOAT4 input1 = FLOAT4(in16[(2 * pair + 1) * area_size]);
        #endif
            input_sum += input0[0] + input0[1] + input0[2] + input0[3] +
                         input1[0] + input1[1] + input1[2] + input1[3];
            raw_dot += input0[0] * FLOAT(1.0 / 16.0) * FLOAT4(weight0 & ushort4(0x00F0)) +
                       input0[1] * FLOAT4(weight0 & ushort4(0x000F)) +
                       input0[2] * FLOAT(1.0 / 4096.0) * FLOAT4(weight0 & ushort4(0xF000)) +
                       input0[3] * FLOAT(1.0 / 256.0) * FLOAT4(weight0 & ushort4(0x0F00)) +
                       input1[0] * FLOAT(1.0 / 16.0) * FLOAT4(weight1 & ushort4(0x00F0)) +
                       input1[1] * FLOAT4(weight1 & ushort4(0x000F)) +
                       input1[2] * FLOAT(1.0 / 4096.0) * FLOAT4(weight1 & ushort4(0xF000)) +
                       input1[3] * FLOAT(1.0 / 256.0) * FLOAT4(weight1 & ushort4(0x0F00));
        }
#else
        for (int z = zmin + laneInBlock; z < zmax; z += lanesPerBlock) {
            FLOAT4 in4 = GEMV_2SG_LOAD_IN4(z);
            input_sum += dot(in4, FLOAT4(1.0));

            // Pre-scaling trick: avoid shift by pre-dividing input
            FLOAT in_ps0 = in4[0] * FLOAT(1.0/16.0);    // compensates nibble at bits[4:7] (×16)
            FLOAT in_ps2 = in4[2] * FLOAT(1.0/4096.0);  // compensates nibble at bits[12:15] (×4096)
            FLOAT in_ps3 = in4[3] * FLOAT(1.0/256.0);   // compensates nibble at bits[8:11] (×256)

            // Read weight as ushort4, mask without shift -- vectorized: 4
            // vector FMAs instead of 16 scalar ones, elementwise identical.
            ushort4 w16 = xy_wt[z];
            raw_dot += in_ps0 * FLOAT4(w16 & ushort4(0x00F0)) + in4[1] * FLOAT4(w16 & ushort4(0x000F))
                     + in_ps2 * FLOAT4(w16 & ushort4(0xF000)) + in_ps3 * FLOAT4(w16 & ushort4(0x0F00));
        }
#endif
        FLOAT4 adjusted_bias = dequant_bias - FLOAT(8.0) * scale;
        result += raw_dot * scale + input_sum * adjusted_bias;
    #elif defined(W_QUANT_8)
        // See the GEMV_2OCQUAD_PER_SG W8 branch above: w[j] = output lane j's ic weights, so
        // accumulate dot(in4, w[j]) per lane; the per-z scalar-broadcast form
        // is the transposed product with the wrong scale lane.
        FLOAT4 raw_dot = FLOAT4(0);
        FLOAT input_sum = FLOAT(0);
        for (int z = zmin + laneInBlock; z < zmax; z += lanesPerBlock) {
            auto w = xy_wt[z];
            FLOAT4 in4 = GEMV_2SG_LOAD_IN4(z);
            input_sum += dot(in4, FLOAT4(1.0));
            raw_dot += FLOAT4(dot(in4, FLOAT4(w[0])), dot(in4, FLOAT4(w[1])),
                              dot(in4, FLOAT4(w[2])), dot(in4, FLOAT4(w[3])));
        }
        result += raw_dot * scale + input_sum * dequant_bias;
    #endif
    }

    result = simd_sum(result);

#if GEMV_SPLIT_K > 1
    threadgroup FLOAT4 sk_partial[tgQuads];
    if (sk_half == 1 && tiisg == 0) {
        sk_partial[(int)sgitg % tgQuads] = result;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sk_half == 0 && tiisg == 0) {
        result += sk_partial[(int)sgitg % tgQuads];
        GEMV_OUT[GEMV_OUT_BASE + uz * area_size] = activate(ftype4(result + biasValue), cst.activation);
    }
#else
    if (tiisg == 0) {
        GEMV_OUT[GEMV_OUT_BASE + uz * area_size] = activate(ftype4(result + biasValue), cst.activation);
    }
#endif
#endif // GEMV_2OCQUAD_PER_SG
#undef GEMV_2SG_LOAD_IN4
#undef GEMV_OUT
#undef GEMV_OUT_BASE
}

kernel void conv1x1_gemv_g8_wquant_sg(const device ftype4 *in            [[buffer(0)]],
                            device ftype4 *out                 [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                        #ifdef W_QUANT_2
                            const device uchar4 *wt      [[buffer(3)]],
                        #elif defined(W_QUANT_3)
                            const device uchar *wt      [[buffer(3)]],
                        #elif defined(W_QUANT_4)
                            const device MNN::uchar4x2 *wt      [[buffer(3)]],
                        #elif defined(W_QUANT_8)
                            const device MNN::char4x4 *wt      [[buffer(3)]],
                        #endif
                            const device ftype4 *biasTerms     [[buffer(4)]],
                            const device ftype4 *dequantScale  [[buffer(5)]],
                            uint3 gid[[threadgroup_position_in_grid]],
                            uint3 threadsPerThreadgroup [[threads_per_threadgroup]],
                            uint  tiisg[[thread_index_in_simdgroup]],
                            uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    // each threadgroup contain 2 simdgroup
    // each simdgroup compute 4 data
    int simdgroupOc = 2;
    int simdgroupSize = threadsPerThreadgroup.x / SIMD_GROUP_WIDTH;
    int simdgroupIc = simdgroupSize/simdgroupOc;
    int SIMD_GROUP_WIDTH_4 = int(threadsPerThreadgroup.x) / simdgroupOc;
    int o_sgitg = sgitg % simdgroupOc;
    int i_sgitg = sgitg / simdgroupOc;

    int uz = gid.x * simdgroupOc + o_sgitg;

    int rx = gid.y;
#ifdef W_QUANT_3
    auto xy_wt = wt + uz * cst.inputDepthQuad * 6;
#else
    auto xy_wt = wt + uz * cst.inputDepthQuad;
#endif
    auto xy_in0  = in + rx;
    auto area_size = cst.outputSize * cst.batch;
    auto xy_out = out + uz * area_size + rx;
    FLOAT4 result0 = FLOAT4(0);
    threadgroup FLOAT4 localSum[32];
    if(uz < cst.outputDepthQuad) {
        int quadsPerBlock = (cst.inputDepthQuad + cst.blockCount - 1) / cst.blockCount;
        
        int lanesPerBlock = min(SIMD_GROUP_WIDTH_4, quadsPerBlock);
        int blocksInFlight = SIMD_GROUP_WIDTH_4 / lanesPerBlock;
        int laneInBlock = (tiisg + i_sgitg * SIMD_GROUP_WIDTH) % lanesPerBlock;
        int blockSlot = (tiisg + i_sgitg * SIMD_GROUP_WIDTH) / lanesPerBlock;


        for (int bi= blockSlot; bi<cst.blockCount; bi += blocksInFlight) {
            FLOAT4 scale = FLOAT4(dequantScale[2 * (uz * cst.blockCount + bi) + 0]) / (FLOAT)cst.scaleCoef;
            FLOAT4 dequant_bias = FLOAT4(dequantScale[2 * (uz * cst.blockCount + bi) + 1]) / (FLOAT)cst.scaleCoef;
            int zmin = bi * quadsPerBlock;
            int zmax = min(zmin + quadsPerBlock, cst.inputDepthQuad);
            // The W2/W3 branch must be tested before W4/W8 since the ladder is
            // exclusive; only one W_QUANT_* macro is defined per compilation.
            #if defined(W_QUANT_2) || defined(W_QUANT_3)
            // Deferred dequantization with power-of-two pre-scaling (mask-only
            // extraction, no per-element shifts/subs): accumulate unsigned raw
            // dots and input sums, apply scale/bias once per quant block.
            // W2 packs 4 2-bit values per byte: IC0@[7:6] IC1@[5:4] IC2@[3:2]
            // IC3@[1:0]. W3 adds a high-bit plane in bytes 4..5 of the 6-byte
            // tile (nibble per OC pair, bit3=IC0..bit0=IC3, weight x4).
            {
                FLOAT4 raw_dot = FLOAT4(0);
                FLOAT input_sum = FLOAT(0);
            #ifdef W_QUANT_2
                for (int z = zmin + laneInBlock; z < zmax; z += lanesPerBlock) {
                    FLOAT4 in40 = (FLOAT4)*(xy_in0 + z * area_size);
                    input_sum += in40[0] + in40[1] + in40[2] + in40[3];

                    // Pre-scale so a bare mask recovers the value x input:
                    //   &0xC0 -> x1/64, &0x30 -> x1/16, &0x0C -> x1/4, &0x03 -> x1
                    FLOAT in_ps0 = in40[0] * FLOAT(1.0/64.0);
                    FLOAT in_ps1 = in40[1] * FLOAT(1.0/16.0);
                    FLOAT in_ps2 = in40[2] * FLOAT(1.0/4.0);

                    uchar4 w_b = xy_wt[z];
                    raw_dot[0] += in_ps0 * FLOAT(w_b[0] & 0xC0) + in_ps1 * FLOAT(w_b[0] & 0x30)
                                + in_ps2 * FLOAT(w_b[0] & 0x0C) + in40[3] * FLOAT(w_b[0] & 0x03);
                    raw_dot[1] += in_ps0 * FLOAT(w_b[1] & 0xC0) + in_ps1 * FLOAT(w_b[1] & 0x30)
                                + in_ps2 * FLOAT(w_b[1] & 0x0C) + in40[3] * FLOAT(w_b[1] & 0x03);
                    raw_dot[2] += in_ps0 * FLOAT(w_b[2] & 0xC0) + in_ps1 * FLOAT(w_b[2] & 0x30)
                                + in_ps2 * FLOAT(w_b[2] & 0x0C) + in40[3] * FLOAT(w_b[2] & 0x03);
                    raw_dot[3] += in_ps0 * FLOAT(w_b[3] & 0xC0) + in_ps1 * FLOAT(w_b[3] & 0x30)
                                + in_ps2 * FLOAT(w_b[3] & 0x0C) + in40[3] * FLOAT(w_b[3] & 0x03);
                }
                FLOAT4 adjusted_bias = dequant_bias - FLOAT(2.0) * scale;
            #else
                for (int z = zmin + laneInBlock; z < zmax; z += lanesPerBlock) {
                    FLOAT4 in40 = (FLOAT4)*(xy_in0 + z * area_size);
                    input_sum += in40[0] + in40[1] + in40[2] + in40[3];

                    // lo-plane pre-scales (as W2) plus hi-plane ones (bit weight
                    // x4): &0x8 -> x1/2, &0x4 -> x1, &0x2 -> x2, &0x1 -> x4
                    FLOAT in_ps0 = in40[0] * FLOAT(1.0/64.0);
                    FLOAT in_ps1 = in40[1] * FLOAT(1.0/16.0);
                    FLOAT in_ps2 = in40[2] * FLOAT(1.0/4.0);
                    FLOAT in_hi0 = in40[0] * FLOAT(0.5);
                    FLOAT in_hi2 = in40[2] * FLOAT(2.0);
                    FLOAT in_hi3 = in40[3] * FLOAT(4.0);

                    const device ushort* tilePtr = (const device ushort*)(xy_wt + z * 6);
                    ushort tw0 = tilePtr[0], tw1 = tilePtr[1], tw2 = tilePtr[2];
                    uchar lo0 = uchar(tw0), lo1 = uchar(tw0 >> 8), lo2 = uchar(tw1), lo3 = uchar(tw1 >> 8);
                    uchar hb0 = uchar(tw2), hb1 = uchar(tw2 >> 8);
                    uchar h0 = hb0 >> 4;
                    uchar h1 = hb0 & 0xF;
                    uchar h2 = hb1 >> 4;
                    uchar h3 = hb1 & 0xF;

                    raw_dot[0] += in_ps0 * FLOAT(lo0 & 0xC0) + in_ps1 * FLOAT(lo0 & 0x30)
                                + in_ps2 * FLOAT(lo0 & 0x0C) + in40[3] * FLOAT(lo0 & 0x03)
                                + in_hi0 * FLOAT(h0 & 0x8) + in40[1] * FLOAT(h0 & 0x4)
                                + in_hi2 * FLOAT(h0 & 0x2) + in_hi3 * FLOAT(h0 & 0x1);
                    raw_dot[1] += in_ps0 * FLOAT(lo1 & 0xC0) + in_ps1 * FLOAT(lo1 & 0x30)
                                + in_ps2 * FLOAT(lo1 & 0x0C) + in40[3] * FLOAT(lo1 & 0x03)
                                + in_hi0 * FLOAT(h1 & 0x8) + in40[1] * FLOAT(h1 & 0x4)
                                + in_hi2 * FLOAT(h1 & 0x2) + in_hi3 * FLOAT(h1 & 0x1);
                    raw_dot[2] += in_ps0 * FLOAT(lo2 & 0xC0) + in_ps1 * FLOAT(lo2 & 0x30)
                                + in_ps2 * FLOAT(lo2 & 0x0C) + in40[3] * FLOAT(lo2 & 0x03)
                                + in_hi0 * FLOAT(h2 & 0x8) + in40[1] * FLOAT(h2 & 0x4)
                                + in_hi2 * FLOAT(h2 & 0x2) + in_hi3 * FLOAT(h2 & 0x1);
                    raw_dot[3] += in_ps0 * FLOAT(lo3 & 0xC0) + in_ps1 * FLOAT(lo3 & 0x30)
                                + in_ps2 * FLOAT(lo3 & 0x0C) + in40[3] * FLOAT(lo3 & 0x03)
                                + in_hi0 * FLOAT(h3 & 0x8) + in40[1] * FLOAT(h3 & 0x4)
                                + in_hi2 * FLOAT(h3 & 0x2) + in_hi3 * FLOAT(h3 & 0x1);
                }
                FLOAT4 adjusted_bias = dequant_bias - FLOAT(4.0) * scale;
            #endif
                result0 += raw_dot * scale + input_sum * adjusted_bias;
            }
            #elif defined(W_QUANT_4)
            // Deferred dequantization: accumulate raw dot products and input sums,
            // apply scale/bias once per quant block.
            {
                FLOAT4 raw_dot = FLOAT4(0);
                FLOAT input_sum = FLOAT(0);
                constexpr FLOAT4 ones = FLOAT4(1.0);

                for (int z = zmin + laneInBlock; z < zmax; z += lanesPerBlock) {
                    FLOAT4 in40 = (FLOAT4)*(xy_in0 + z * area_size);
                    MNN::uchar4x2 w_int4 = xy_wt[z];

                    input_sum += dot(in40, ones);

                    FLOAT4 wv0 = FLOAT4(FLOAT(w_int4[0][0] >> 4), FLOAT(w_int4[0][0] & 15),
                                        FLOAT(w_int4[0][1] >> 4), FLOAT(w_int4[0][1] & 15));
                    FLOAT4 wv1 = FLOAT4(FLOAT(w_int4[1][0] >> 4), FLOAT(w_int4[1][0] & 15),
                                        FLOAT(w_int4[1][1] >> 4), FLOAT(w_int4[1][1] & 15));
                    FLOAT4 wv2 = FLOAT4(FLOAT(w_int4[2][0] >> 4), FLOAT(w_int4[2][0] & 15),
                                        FLOAT(w_int4[2][1] >> 4), FLOAT(w_int4[2][1] & 15));
                    FLOAT4 wv3 = FLOAT4(FLOAT(w_int4[3][0] >> 4), FLOAT(w_int4[3][0] & 15),
                                        FLOAT(w_int4[3][1] >> 4), FLOAT(w_int4[3][1] & 15));

                    raw_dot[0] += dot(in40, wv0);
                    raw_dot[1] += dot(in40, wv1);
                    raw_dot[2] += dot(in40, wv2);
                    raw_dot[3] += dot(in40, wv3);
                }
                FLOAT4 adjusted_bias = dequant_bias - FLOAT(8.0) * scale;
                result0 += raw_dot * scale + input_sum * adjusted_bias;
            }
            #else
            // W_QUANT_8
            for (int z = zmin + laneInBlock; z < zmax; z += lanesPerBlock) {
                FLOAT4 in40 = (FLOAT4)*(xy_in0 + z * area_size);

                auto w = xy_wt[z];
                FLOAT4x4 w_fp32 = FLOAT4x4(FLOAT4(w[0]), FLOAT4(w[1]), FLOAT4(w[2]), FLOAT4(w[3]));
                FLOAT4x4 w_dequant;
                for (int i = 0; i < 4; ++i) {
                    w_dequant[i] = w_fp32[i] * scale[i] + dequant_bias[i];
                }

                result0 += FLOAT4(in40 * w_dequant);
            }
            #endif
        }
        FLOAT4 res = simd_sum(result0);

        if (0 == tiisg) {
            localSum[i_sgitg + o_sgitg * simdgroupIc] = res;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(uz < cst.outputDepthQuad) {
        if (i_sgitg == 0 && tiisg == 0) {
            FLOAT4 res = FLOAT4(biasTerms[uz]);
            for (int i=0; i<simdgroupIc; ++i) {
                res += localSum[i + o_sgitg * simdgroupIc];
            }
            xy_out[0] = activate(ftype4(res), cst.activation);
        }
    }
}

kernel void conv1x1_gemv_g16_wquant_sg(const device ftype4 *in            [[buffer(0)]],
                            device ftype4 *out                 [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                        #ifdef W_QUANT_2
                            const device uchar4 *wt            [[buffer(3)]],
                        #elif defined(W_QUANT_3)
                            const device uchar *wt             [[buffer(3)]],
                        #elif defined(W_QUANT_4)
                            const device ushort4 *wt            [[buffer(3)]],
                        #elif defined(W_QUANT_8)
                            const device MNN::char4x4 *wt      [[buffer(3)]],
                        #endif
                            const device ftype4 *biasTerms     [[buffer(4)]],
                            const device ftype4 *dequantScale  [[buffer(5)]],
                            uint3 gid[[threadgroup_position_in_grid]],
                            uint  tiisg[[thread_index_in_simdgroup]],
                            uint  sgitg[[simdgroup_index_in_threadgroup]]) {
#ifdef G16_SPLIT_K
    // 4 simdgroups per threadgroup (128 threads); each SG pair (sgitg >> 1)
    // cooperates on the same two oc_4 rows, splitting the quant-block range
    // in half (sk_half = sgitg & 1). Partials combine via threadgroup memory
    // after the loop. Host guarantees the grid is exact (oc % 16 == 0), so no
    // simdgroup may early-return before the barrier below -- the legacy guard
    // is deliberately absent here.
    const int G16_ROWS = 2;
    const int sk_half = (int)sgitg & 1;
    int uz = G16_ROWS * (gid.x * 2 + (int)(sgitg >> 1));
#else
    // 2 simdgroups per threadgroup; each simdgroup computes 8 output data (2 oc_4).
    const int GEMV_G16_SGS = 2;
    const int G16_ROWS = 2;
    int uz = G16_ROWS * (gid.x * GEMV_G16_SGS + sgitg);
    if(uz >= cst.outputDepthQuad) {
        return;
    }
#endif
    auto area_size = cst.outputSize * cst.batch;
    int rx = gid.y;
#ifdef W_QUANT_3
    auto xy_wt = wt + uz * cst.inputDepthQuad * 6;
#else
    auto xy_wt = wt + uz * cst.inputDepthQuad;
#endif
    auto xy_in0  = in + rx;
    auto xy_out = out + uz * area_size + rx;
    auto biasValue0 = FLOAT4(biasTerms[uz]);
    auto biasValue1 = FLOAT4(biasTerms[uz + 1]);

    FLOAT4 result0 = FLOAT4(0);
    FLOAT4 result1 = FLOAT4(0);


    int quadsPerBlock = (cst.inputDepthQuad + cst.blockCount - 1) / cst.blockCount;

    int lanesPerBlock = min(SIMD_GROUP_WIDTH, quadsPerBlock);
    int blocksInFlight = SIMD_GROUP_WIDTH / lanesPerBlock;
    int laneInBlock = (tiisg) % lanesPerBlock;
    int blockSlot = (tiisg) / lanesPerBlock;

#ifdef G16_SPLIT_K
    const int g16_bi_begin = sk_half * (cst.blockCount / 2);
    const int g16_bi_end   = (sk_half == 1) ? cst.blockCount : (cst.blockCount / 2);
    for (int bi = g16_bi_begin + blockSlot; bi < g16_bi_end; bi += blocksInFlight) {
#else
    for (int bi= blockSlot; bi<cst.blockCount; bi += blocksInFlight) {
#endif
        const int quant_offset = 2 * (uz * cst.blockCount + bi);
        FLOAT4 scale0 = FLOAT4(dequantScale[quant_offset + 0]) / (FLOAT)cst.scaleCoef;
        FLOAT4 dequant_bias0 = FLOAT4(dequantScale[quant_offset + 1]) / (FLOAT)cst.scaleCoef;
        FLOAT4 scale1 = FLOAT4(dequantScale[quant_offset + (cst.blockCount << 1)]) / (FLOAT)cst.scaleCoef;
        FLOAT4 dequant_bias1 = FLOAT4(dequantScale[quant_offset + (cst.blockCount << 1) + 1]) / (FLOAT)cst.scaleCoef;
        int zmin = bi * quadsPerBlock;
        int zmax = min(zmin + quadsPerBlock, cst.inputDepthQuad);

        #if defined(W_QUANT_2) || defined(W_QUANT_3)
        // Deferred + pre-scaling dual streams (mask/pre-scale derivation in
        // the g8 kernel). Trap-A: must precede the W4 branch.
        {
            FLOAT4 raw_dot0 = FLOAT4(0), raw_dot1 = FLOAT4(0);
            FLOAT input_sum = FLOAT(0);

            for (int z = zmin + laneInBlock; z < zmax; z += lanesPerBlock) {
                FLOAT4 in4 = (FLOAT4)*(xy_in0 + z * area_size);
                input_sum += in4[0] + in4[1] + in4[2] + in4[3];

                FLOAT in_ps0 = in4[0] * FLOAT(1.0/64.0);
                FLOAT in_ps1 = in4[1] * FLOAT(1.0/16.0);
                FLOAT in_ps2 = in4[2] * FLOAT(1.0/4.0);
            #ifdef W_QUANT_2
                uchar4 w0 = xy_wt[z];
                uchar4 w1 = xy_wt[cst.inputDepthQuad + z];
                raw_dot0[0] += in_ps0 * FLOAT(w0[0] & 0xC0) + in_ps1 * FLOAT(w0[0] & 0x30)
                             + in_ps2 * FLOAT(w0[0] & 0x0C) + in4[3] * FLOAT(w0[0] & 0x03);
                raw_dot0[1] += in_ps0 * FLOAT(w0[1] & 0xC0) + in_ps1 * FLOAT(w0[1] & 0x30)
                             + in_ps2 * FLOAT(w0[1] & 0x0C) + in4[3] * FLOAT(w0[1] & 0x03);
                raw_dot0[2] += in_ps0 * FLOAT(w0[2] & 0xC0) + in_ps1 * FLOAT(w0[2] & 0x30)
                             + in_ps2 * FLOAT(w0[2] & 0x0C) + in4[3] * FLOAT(w0[2] & 0x03);
                raw_dot0[3] += in_ps0 * FLOAT(w0[3] & 0xC0) + in_ps1 * FLOAT(w0[3] & 0x30)
                             + in_ps2 * FLOAT(w0[3] & 0x0C) + in4[3] * FLOAT(w0[3] & 0x03);
                raw_dot1[0] += in_ps0 * FLOAT(w1[0] & 0xC0) + in_ps1 * FLOAT(w1[0] & 0x30)
                             + in_ps2 * FLOAT(w1[0] & 0x0C) + in4[3] * FLOAT(w1[0] & 0x03);
                raw_dot1[1] += in_ps0 * FLOAT(w1[1] & 0xC0) + in_ps1 * FLOAT(w1[1] & 0x30)
                             + in_ps2 * FLOAT(w1[1] & 0x0C) + in4[3] * FLOAT(w1[1] & 0x03);
                raw_dot1[2] += in_ps0 * FLOAT(w1[2] & 0xC0) + in_ps1 * FLOAT(w1[2] & 0x30)
                             + in_ps2 * FLOAT(w1[2] & 0x0C) + in4[3] * FLOAT(w1[2] & 0x03);
                raw_dot1[3] += in_ps0 * FLOAT(w1[3] & 0xC0) + in_ps1 * FLOAT(w1[3] & 0x30)
                             + in_ps2 * FLOAT(w1[3] & 0x0C) + in4[3] * FLOAT(w1[3] & 0x03);
            #else
                FLOAT in_hi0 = in4[0] * FLOAT(0.5);
                FLOAT in_hi2 = in4[2] * FLOAT(2.0);
                FLOAT in_hi3 = in4[3] * FLOAT(4.0);
                const device ushort* t0 = (const device ushort*)(xy_wt + z * 6);
                const device ushort* t1 = (const device ushort*)(xy_wt + (cst.inputDepthQuad + z) * 6);
                ushort sa0 = t0[0], sa1 = t0[1], sa2 = t0[2];
                ushort sb0 = t1[0], sb1 = t1[1], sb2 = t1[2];
                uchar a0 = uchar(sa0), a1 = uchar(sa0 >> 8), a2 = uchar(sa1), a3 = uchar(sa1 >> 8);
                uchar hab0 = uchar(sa2), hab1 = uchar(sa2 >> 8);
                uchar ha0 = hab0 >> 4, ha1 = hab0 & 0xF, ha2 = hab1 >> 4, ha3 = hab1 & 0xF;
                uchar b0 = uchar(sb0), b1 = uchar(sb0 >> 8), b2 = uchar(sb1), b3 = uchar(sb1 >> 8);
                uchar hbb0 = uchar(sb2), hbb1 = uchar(sb2 >> 8);
                uchar hb0 = hbb0 >> 4, hb1 = hbb0 & 0xF, hb2 = hbb1 >> 4, hb3 = hbb1 & 0xF;
                raw_dot0[0] += in_ps0 * FLOAT(a0 & 0xC0) + in_ps1 * FLOAT(a0 & 0x30)
                             + in_ps2 * FLOAT(a0 & 0x0C) + in4[3] * FLOAT(a0 & 0x03)
                             + in_hi0 * FLOAT(ha0 & 0x8) + in4[1] * FLOAT(ha0 & 0x4)
                             + in_hi2 * FLOAT(ha0 & 0x2) + in_hi3 * FLOAT(ha0 & 0x1);
                raw_dot0[1] += in_ps0 * FLOAT(a1 & 0xC0) + in_ps1 * FLOAT(a1 & 0x30)
                             + in_ps2 * FLOAT(a1 & 0x0C) + in4[3] * FLOAT(a1 & 0x03)
                             + in_hi0 * FLOAT(ha1 & 0x8) + in4[1] * FLOAT(ha1 & 0x4)
                             + in_hi2 * FLOAT(ha1 & 0x2) + in_hi3 * FLOAT(ha1 & 0x1);
                raw_dot0[2] += in_ps0 * FLOAT(a2 & 0xC0) + in_ps1 * FLOAT(a2 & 0x30)
                             + in_ps2 * FLOAT(a2 & 0x0C) + in4[3] * FLOAT(a2 & 0x03)
                             + in_hi0 * FLOAT(ha2 & 0x8) + in4[1] * FLOAT(ha2 & 0x4)
                             + in_hi2 * FLOAT(ha2 & 0x2) + in_hi3 * FLOAT(ha2 & 0x1);
                raw_dot0[3] += in_ps0 * FLOAT(a3 & 0xC0) + in_ps1 * FLOAT(a3 & 0x30)
                             + in_ps2 * FLOAT(a3 & 0x0C) + in4[3] * FLOAT(a3 & 0x03)
                             + in_hi0 * FLOAT(ha3 & 0x8) + in4[1] * FLOAT(ha3 & 0x4)
                             + in_hi2 * FLOAT(ha3 & 0x2) + in_hi3 * FLOAT(ha3 & 0x1);
                raw_dot1[0] += in_ps0 * FLOAT(b0 & 0xC0) + in_ps1 * FLOAT(b0 & 0x30)
                             + in_ps2 * FLOAT(b0 & 0x0C) + in4[3] * FLOAT(b0 & 0x03)
                             + in_hi0 * FLOAT(hb0 & 0x8) + in4[1] * FLOAT(hb0 & 0x4)
                             + in_hi2 * FLOAT(hb0 & 0x2) + in_hi3 * FLOAT(hb0 & 0x1);
                raw_dot1[1] += in_ps0 * FLOAT(b1 & 0xC0) + in_ps1 * FLOAT(b1 & 0x30)
                             + in_ps2 * FLOAT(b1 & 0x0C) + in4[3] * FLOAT(b1 & 0x03)
                             + in_hi0 * FLOAT(hb1 & 0x8) + in4[1] * FLOAT(hb1 & 0x4)
                             + in_hi2 * FLOAT(hb1 & 0x2) + in_hi3 * FLOAT(hb1 & 0x1);
                raw_dot1[2] += in_ps0 * FLOAT(b2 & 0xC0) + in_ps1 * FLOAT(b2 & 0x30)
                             + in_ps2 * FLOAT(b2 & 0x0C) + in4[3] * FLOAT(b2 & 0x03)
                             + in_hi0 * FLOAT(hb2 & 0x8) + in4[1] * FLOAT(hb2 & 0x4)
                             + in_hi2 * FLOAT(hb2 & 0x2) + in_hi3 * FLOAT(hb2 & 0x1);
                raw_dot1[3] += in_ps0 * FLOAT(b3 & 0xC0) + in_ps1 * FLOAT(b3 & 0x30)
                             + in_ps2 * FLOAT(b3 & 0x0C) + in4[3] * FLOAT(b3 & 0x03)
                             + in_hi0 * FLOAT(hb3 & 0x8) + in4[1] * FLOAT(hb3 & 0x4)
                             + in_hi2 * FLOAT(hb3 & 0x2) + in_hi3 * FLOAT(hb3 & 0x1);
            #endif
            }
            #ifdef W_QUANT_2
            FLOAT4 adj0 = dequant_bias0 - FLOAT(2.0) * scale0;
            FLOAT4 adj1 = dequant_bias1 - FLOAT(2.0) * scale1;
            #else
            FLOAT4 adj0 = dequant_bias0 - FLOAT(4.0) * scale0;
            FLOAT4 adj1 = dequant_bias1 - FLOAT(4.0) * scale1;
            #endif
            result0 += raw_dot0 * scale0 + input_sum * adj0;
            result1 += raw_dot1 * scale1 + input_sum * adj1;
        }
        #elif defined(W_QUANT_4)
        // Deferred + pre-scaling nibble extraction (mirrors g4m1_2sg kernel).
        // Read weight as ushort4 (single 128-bit vector load per z instead of
        // 8 bytes via uchar4x2). Extract each nibble with a pure mask (no shift)
        // and compensate the bit-position by pre-scaling the paired input lane.
        // Bias correction (`- 8 * scale`) is folded into adjusted_bias applied
        // outside the inner loop, saving 8 fp16 subs per z-step.
        {
            FLOAT4 raw_dot0 = FLOAT4(0), raw_dot1 = FLOAT4(0);
            FLOAT input_sum = FLOAT(0);

            for (int z = zmin + laneInBlock; z < zmax; z += lanesPerBlock) {
                FLOAT4 in4 = (FLOAT4)*(xy_in0 + z * area_size);
                input_sum += in4[0] + in4[1] + in4[2] + in4[3];

                // Pre-scale the three shifted-nibble slots so the mask alone
                // recovers the original weight magnitude:
                //   raw & 0x000F -> in4[1]              (bits[0:3],  ×1)
                //   raw & 0x00F0 -> in4[0]  * (1/16)    (bits[4:7],  ×16)
                //   raw & 0x0F00 -> in4[3]  * (1/256)   (bits[8:11], ×256)
                //   raw & 0xF000 -> in4[2]  * (1/4096)  (bits[12:15],×4096)
                FLOAT in_ps0 = in4[0] * FLOAT(1.0/16.0);
                FLOAT in_ps2 = in4[2] * FLOAT(1.0/4096.0);
                FLOAT in_ps3 = in4[3] * FLOAT(1.0/256.0);

                // First oc_4 (uz)
                ushort4 w16 = xy_wt[z];
                raw_dot0[0] += in_ps0 * FLOAT(w16[0] & 0x00F0) + in4[1] * FLOAT(w16[0] & 0x000F)
                            + in_ps2 * FLOAT(w16[0] & 0xF000) + in_ps3 * FLOAT(w16[0] & 0x0F00);
                raw_dot0[1] += in_ps0 * FLOAT(w16[1] & 0x00F0) + in4[1] * FLOAT(w16[1] & 0x000F)
                            + in_ps2 * FLOAT(w16[1] & 0xF000) + in_ps3 * FLOAT(w16[1] & 0x0F00);
                raw_dot0[2] += in_ps0 * FLOAT(w16[2] & 0x00F0) + in4[1] * FLOAT(w16[2] & 0x000F)
                            + in_ps2 * FLOAT(w16[2] & 0xF000) + in_ps3 * FLOAT(w16[2] & 0x0F00);
                raw_dot0[3] += in_ps0 * FLOAT(w16[3] & 0x00F0) + in4[1] * FLOAT(w16[3] & 0x000F)
                            + in_ps2 * FLOAT(w16[3] & 0xF000) + in_ps3 * FLOAT(w16[3] & 0x0F00);

                // Second oc_4 (uz+1)
                w16 = xy_wt[cst.inputDepthQuad + z];
                raw_dot1[0] += in_ps0 * FLOAT(w16[0] & 0x00F0) + in4[1] * FLOAT(w16[0] & 0x000F)
                            + in_ps2 * FLOAT(w16[0] & 0xF000) + in_ps3 * FLOAT(w16[0] & 0x0F00);
                raw_dot1[1] += in_ps0 * FLOAT(w16[1] & 0x00F0) + in4[1] * FLOAT(w16[1] & 0x000F)
                            + in_ps2 * FLOAT(w16[1] & 0xF000) + in_ps3 * FLOAT(w16[1] & 0x0F00);
                raw_dot1[2] += in_ps0 * FLOAT(w16[2] & 0x00F0) + in4[1] * FLOAT(w16[2] & 0x000F)
                            + in_ps2 * FLOAT(w16[2] & 0xF000) + in_ps3 * FLOAT(w16[2] & 0x0F00);
                raw_dot1[3] += in_ps0 * FLOAT(w16[3] & 0x00F0) + in4[1] * FLOAT(w16[3] & 0x000F)
                            + in_ps2 * FLOAT(w16[3] & 0xF000) + in_ps3 * FLOAT(w16[3] & 0x0F00);
            }
            FLOAT4 adj0 = dequant_bias0 - FLOAT(8.0) * scale0;
            FLOAT4 adj1 = dequant_bias1 - FLOAT(8.0) * scale1;
            result0 += raw_dot0 * scale0 + input_sum * adj0;
            result1 += raw_dot1 * scale1 + input_sum * adj1;
        }
        #elif defined(W_QUANT_8)
        for (int z = zmin + laneInBlock; z < zmax; z += lanesPerBlock) {
            FLOAT4 in40 = (FLOAT4)*(xy_in0 + z * area_size);

            {
                auto w = xy_wt[z];
                FLOAT4x4 w_fp32 = FLOAT4x4(FLOAT4(w[0]), FLOAT4(w[1]), FLOAT4(w[2]), FLOAT4(w[3]));
                FLOAT4x4 w_dequant;
                for (int i = 0; i < 4; ++i) {
                    w_dequant[i] = w_fp32[i] * scale0[i] + dequant_bias0[i];
                }
                result0 += FLOAT4(in40 * w_dequant);
            }
            {
                auto w = xy_wt[cst.inputDepthQuad + z];
                FLOAT4x4 w_fp32 = FLOAT4x4(FLOAT4(w[0]), FLOAT4(w[1]), FLOAT4(w[2]), FLOAT4(w[3]));
                FLOAT4x4 w_dequant;
                for (int i = 0; i < 4; ++i) {
                    w_dequant[i] = w_fp32[i] * scale1[i] + dequant_bias1[i];
                }
                result1 += FLOAT4(in40 * w_dequant);
            }
        }
        #endif
    }

    FLOAT4 res0 = simd_sum(result0);
    FLOAT4 res1 = simd_sum(result1);

#ifdef G16_SPLIT_K
    // Two rows per SG pair -> two partial slots per pair.
    threadgroup FLOAT4 g16_sk_partial0[2];
    threadgroup FLOAT4 g16_sk_partial1[2];
    const int row_pair = (int)sgitg >> 1;
    if (sk_half == 1 && tiisg == 0) {
        g16_sk_partial0[row_pair] = res0;
        g16_sk_partial1[row_pair] = res1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sk_half == 0 && tiisg == 0) {
        res0 += g16_sk_partial0[row_pair];
        res1 += g16_sk_partial1[row_pair];
        xy_out[0] = activate(ftype4(res0 + biasValue0), cst.activation);
        xy_out[area_size] = activate(ftype4(res1 + biasValue1), cst.activation);
    }
#else
    /* true */
    if (tiisg == 0) {
        xy_out[0] = activate(ftype4(res0 + biasValue0), cst.activation);
        xy_out[area_size] = activate(ftype4(res1 + biasValue1), cst.activation);
    }
#endif
}
)metal";

#endif
