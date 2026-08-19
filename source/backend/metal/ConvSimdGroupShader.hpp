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
    int input_size;
    int input_slice;
    int output_width;
    int output_height;
    int output_size;
    int output_slice;
    int output_channel;
    int batch;
    int block_size;
    conv_activation_type activation;
    float scale_coef;
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

kernel void conv1x1_gemm_8x8_wquant_sg(const device ftype2 *in            [[buffer(0)]],
                            device ftype2 *out                 [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                        #ifdef W_QUANT_4
                            const device uchar *wt      [[buffer(3)]],
                        #elif defined(W_QUANT_8)
                            const device char2 *wt      [[buffer(3)]],
                        #endif
                            const device ftype2 *biasTerms     [[buffer(4)]],
                            const device ftype2 *dequantScale  [[buffer(5)]],
                            uint3 gid                          [[threadgroup_position_in_grid]],
                            uint                  tiitg[[thread_index_in_threadgroup]],
                            uint                  sgitg[[simdgroup_index_in_threadgroup]]) {
    /*
     // Read:
     ftype 0~63   ---> input: [M8, K8]
     ftype 64~127 ---> input: [K8, N8]
     // Write:
     ftype 0~63 ---> input: [M8, N8]
     */

    threadgroup FLOAT sdata[128] = {0.f};
    INIT_SIMDGROUP_MATRIX(1, 1, 1);

    int rx = gid.x;// M/8
    int uz = gid.y;// N/8

    int kl = tiitg / 16; // 0~1
    int rcl = tiitg % 16; // 0~15
    int kr = rcl % 2; // 0~1
    int ml = rcl / 2; // 0 ~ 7
    int nl = ml / 2; // 0 ~ 3
    int nr = ml % 2; // 0 ~ 1


    /** input:
     threadgroup: [M8, K8]
     each thread: K2
     layout: [K/4, M, K4] -> [K/8, K2, M/8, M8, K2, K2]
     index : [0, kr, rx, ml, kl, K2]
     offset: ((0*2+kr) * M + rx * 8 + ml) * 2 + kl
     */
    /** weight:
     threadgroup: [K8, N8]
     each thread: K2
     layout: [N/4, K/4, N4, K2, K2] -> [N/8, N2, K/8, K2, N4, K2, K2]
     index : [uz, nr, 0, kr, nl, kl, K2]
     offset: (((uz * 2 + nr) * K/4 + 0*2+kr) * 4 + nl) * 2 + kl
     */
    /** output:
     threadgroup: [M8, N8] -> [M8, N4, N2]
     sdata: [ml, kr * 2 + kl]
     each thread: N4
     layout: [N/4, M, N4] -> [N/8, N2, M/8, M8, N2, N2]
     index : [uz, kr, rx, ml, kl, N2]
     offset: (((uz * 2 + kr) * M + rx * 8 + ml) * 2 + kl)
     */

    // boundary limit
    int idx_n4 = (uz * 2 + nr) < cst.output_slice ? (uz * 2 + nr) : (cst.output_slice - 1);
    int idx_m  = (rx * 8 + ml) < cst.input_size * cst.batch ? (rx * 8 + ml) : (cst.input_size * cst.batch - 1);

    auto xy_wt = wt +  ((idx_n4 * cst.input_slice + 0*2+kr) * 4 + nl) * 2 + kl;// [N/4, K/4, N4, K4]
    auto xy_in0  = in + ((0*2+kr) * cst.input_size * cst.batch + idx_m) * 2 + kl;// [K/4, M, K2, K2]
    auto xy_out = out + ((uz * 2 + kr) * cst.output_size * cst.batch + rx * 8 + ml) * 2 + kl;// [N/4, M, N4]

    int block = (cst.input_slice + cst.block_size - 1) / cst.block_size;
    for (int bi=0; bi<cst.block_size; ++bi) {
        // [N/4, cst.block_size, 2/*scale_bias*/, N2， N2]
        FLOAT2 scale2 = FLOAT2(dequantScale[(2 * (idx_n4 * cst.block_size + bi) + 0) * 2 + nl / 2]) / (FLOAT)cst.scale_coef;
        FLOAT2 dequant_bias2 = FLOAT2(dequantScale[(2 * (idx_n4 * cst.block_size + bi) + 1) * 2 + nl / 2]) / (FLOAT)cst.scale_coef;
        FLOAT scale = scale2[nl % 2];
        FLOAT dequant_bias = dequant_bias2[nl % 2];
        int zmin = bi * block;
        int zmax = min(zmin + block, cst.input_slice);

        for (int z = zmin; z < zmax; z += 2) {
            // [M8, K2, K2, K2]
            ((threadgroup ftype2*)sdata)[(ml * 2 + kr) * 2 + kl] = (*xy_in0);
            xy_in0 += 4 * cst.input_size * cst.batch;

            #ifdef W_QUANT_4
                uchar w_int40 = xy_wt[8 * z]; // [N/4, K/4, N4, K4]
                FLOAT2 w20 = FLOAT2((float)(w_int40 >> 4) - 8, (float)(w_int40 & 15) - 8);
            #elif defined(W_QUANT_8)
                char2 w_int40 = xy_wt[8 * z]; // [N/4, K/4, N4, K4]
                FLOAT2 w20 = FLOAT2((float)w_int40[0], (float)w_int40[1]);
            #endif

            FLOAT2 res = w20 * scale + dequant_bias;
            // [K8, N4, N2]
            ((threadgroup ftype*)sdata)[64 + (kr * 4 + kl * 2 + 0) * 8 + nr * 4 + nl] = ftype(res[0]);
            ((threadgroup ftype*)sdata)[64 + (kr * 4 + kl * 2 + 1) * 8 + nr * 4 + nl] = ftype(res[1]);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_load(sga[0], (const threadgroup ftype*)sdata, 8);
            simdgroup_load(sgb[0], ((const threadgroup ftype*)sdata) + 64, 8);

            SIMDGROUP_MATRIX_FMA(1, 1);

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    SIMDGROUP_MATRIX_STORE((threadgroup FLOAT*)sdata, 1);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if((rx * 8 + ml) < cst.input_size * cst.batch) {
        if((uz * 2 + kr) < cst.output_slice) {
            xy_out[0] =  activate(ftype2(((threadgroup FLOAT2*)sdata)[ml * 4 + kr * 2 + kl] + FLOAT2(biasTerms[(uz * 2 + kr) * 2 + kl])), cst.activation);
        }
    }
}

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
    int idx_n4 = (4 * uz + rcl / 4) < cst.output_slice ? (4 * uz + rcl / 4) : (cst.output_slice - 1);
    int idx_m  = (8 * rx + rcl%8) < cst.input_size * cst.batch ? (8 * rx + rcl%8) : (cst.input_size * cst.batch - 1);

    auto xy_wt = wt +  ((idx_n4 * cst.input_slice + 0*2+kl) * 4 + rcl % 4);// [N/4, K/4, N4, K4]
    auto xy_in0  = in + ((0*2+rcl/8) * cst.input_size * cst.batch + idx_m) * 2 + kl;// [K/4, M, K2, K2]
    auto xy_out = out + (4 * uz + rcl / 4) * cst.output_size * cst.batch + (rx * 8 + (rcl%4) * 2 + kl);// [N/4, M, N4]

    int block = (cst.input_slice + cst.block_size - 1) / cst.block_size;
    for (int bi=0; bi<cst.block_size; ++bi) {
        // [N/4, cst.block_size, 2/*scale_bias*/, N4]
        FLOAT4 scale = FLOAT4(dequantScale[2 * (idx_n4 * cst.block_size + bi) + 0]) / (FLOAT)cst.scale_coef;
        FLOAT4 dequant_bias = FLOAT4(dequantScale[2 * (idx_n4 * cst.block_size + bi) + 1]) / (FLOAT)cst.scale_coef;
        int zmin = bi * block;
        int zmax = min(zmin + block, cst.input_slice);

        for (int z = zmin; z < zmax; z += 2) {
            // [M8, K2, K2, K2]
            ((threadgroup ftype2*)sdata)[((rcl%8) * 2 + (rcl/8)) * 2 + kl] = (*xy_in0);
            xy_in0 += 4 * cst.input_size * cst.batch;

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

    if((rx * 8 + (rcl%4) * 2 + kl) < cst.input_size * cst.batch) {
        if((4 * uz + rcl / 4) < cst.output_slice) {
            xy_out[0] =  activate(ftype4(((threadgroup FLOAT4*)sdata)[(((rcl / 4) / 2) * 8 + ((rcl%4) * 2 + kl)) * 2 + (rcl / 4) % 2] + FLOAT4(biasTerms[4 * uz + rcl / 4])), cst.activation);
        }
    }
}

kernel void conv1x1_gemm_8x32_wquant_sg(const device ftype2 *in            [[buffer(0)]],
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
     ftype 64~319 ---> input: [K8, N32]
     // Write:
     ftype 0~255 ---> input: [N4, M8, N8]
     */

    threadgroup FLOAT sdata[512] = {0.f};
    INIT_SIMDGROUP_MATRIX(1, 4, 4);

    int rx = gid.x;// M/8
    int uz = gid.y;// N/32

    int kl = tiitg / 16; // 0~1
    int rcl = tiitg % 16; // 0~15
    int kr = rcl % 2; // 0~1
    int ml = rcl / 2; // 0 ~ 7

    /** input:
     threadgroup: [M8, K8]
     each thread: K2
     layout: [K/4, M, K4] -> [K/8, K2, M/8, M8, K2, K2]
     index : [0, kr, rx, ml, kl, K2]
     offset: ((0*2+kr) * M + rx * 8 + ml) * 2 + kl
     */
    /** weight:
     threadgroup: [K8, N32]
     each thread: N2K4
     layout: [N/4, K/4, N4, K4] -> [N/32, N8, K/8, K2, N2, N2, K4]
     index : [uz, ml, 0, kr, kl, N2, K4]
     offset: (((uz * 8 + ml) * K/4 + 0*2+kr) * 4 + kl * 2)
     */
    /** output:
     threadgroup: [M8, N32] -> [N4, M4, M2, N2, N4]
     sdata: [ml/2, kr*2+kl, M2, ml%2, N4]
     each thread: M2N4
     layout: [N/4, M, N4] -> [N/32, N8, M/8, M4, M2, N4]
     index : [uz, ml, rx, kr*2+kl, M2, N4]
     offset: ((uz * 8 + ml) * M + (rx * 8 + (kr*2+kl) * 2 + 0/1))
     */

    // boundary limit
    int idx_n4 = (uz * 8 + ml) < cst.output_slice ? (uz * 8 + ml) : (cst.output_slice - 1);
    int idx_m  = (rx * 8 + ml) < cst.input_size * cst.batch ? (rx * 8 + ml) : (cst.input_size * cst.batch - 1);

    auto xy_wt = wt +  ((idx_n4 * cst.input_slice + 0*2+kr) * 4 + kl * 2);// [N/4, K/4, N4, K4]
    auto xy_in0  = in + ((0*2+kr) * cst.input_size * cst.batch + idx_m) * 2 + kl;// [K/4, M, K2, K2]
    auto xy_out = out + (uz * 8 + ml) * cst.output_size * cst.batch + (rx * 8 + (kr*2+kl) * 2);// [N/4, M, N4]

    int block = (cst.input_slice + cst.block_size - 1) / cst.block_size;
    for (int bi=0; bi<cst.block_size; ++bi) {
        // [N/4, cst.block_size, 2/*scale_bias*/, N4]
        FLOAT4 scale = FLOAT4(dequantScale[2 * (idx_n4 * cst.block_size + bi) + 0]) / (FLOAT)cst.scale_coef;
        FLOAT4 dequant_bias = FLOAT4(dequantScale[2 * (idx_n4 * cst.block_size + bi) + 1]) / (FLOAT)cst.scale_coef;
        int zmin = bi * block;
        int zmax = min(zmin + block, cst.input_slice);

        for (int z = zmin; z < zmax; z += 2) {
            // [M8, K2, K2, K2]
            ((threadgroup ftype2*)sdata)[(ml * 2 + kr) * 2 + kl] = (*xy_in0);
            xy_in0 += 4 * cst.input_size * cst.batch;

            {
                #ifdef W_QUANT_4
                    uchar2 w_int40 = xy_wt[4 * z + 0]; // [N/4, K/4, N4, K4]
                    FLOAT4 w40 = FLOAT4((float)(w_int40[0] >> 4) - 8, (float)(w_int40[0] & 15) - 8, (float)(w_int40[1] >> 4) - 8, (float)(w_int40[1] & 15) - 8);
                #elif defined(W_QUANT_8)
                    char4 w_int40 = xy_wt[4 * z + 0]; // [N/4, K/4, N4, K4]
                    FLOAT4 w40 = FLOAT4((float)w_int40[0], (float)w_int40[1], (float)w_int40[2], (float)w_int40[3]);
                #endif

                FLOAT4 res = w40 * scale[(kl * 2) % 4] + dequant_bias[(kl * 2) % 4];
                // [K8, N4, N4]
                ((threadgroup ftype*)sdata)[64 + (kr * 4 + 0) * 32 + ml * 4 + kl * 2] = ftype(res[0]);
                ((threadgroup ftype*)sdata)[64 + (kr * 4 + 1) * 32 + ml * 4 + kl * 2] = ftype(res[1]);
                ((threadgroup ftype*)sdata)[64 + (kr * 4 + 2) * 32 + ml * 4 + kl * 2] = ftype(res[2]);
                ((threadgroup ftype*)sdata)[64 + (kr * 4 + 3) * 32 + ml * 4 + kl * 2] = ftype(res[3]);
            }

            {
                #ifdef W_QUANT_4
                    uchar2 w_int40 = xy_wt[4 * z + 1]; // [N/4, K/4, N4, K4]
                    FLOAT4 w40 = FLOAT4((float)(w_int40[0] >> 4) - 8, (float)(w_int40[0] & 15) - 8, (float)(w_int40[1] >> 4) - 8, (float)(w_int40[1] & 15) - 8);
                #elif defined(W_QUANT_8)
                    char4 w_int40 = xy_wt[4 * z + 1]; // [N/4, K/4, N4, K4]
                    FLOAT4 w40 = FLOAT4((float)w_int40[0], (float)w_int40[1], (float)w_int40[2], (float)w_int40[3]);
                #endif

                FLOAT4 res = w40 * scale[(kl * 2 + 1) % 4] + dequant_bias[(kl * 2 + 1) % 4];
                // [K8, N4, N4]
                ((threadgroup ftype*)sdata)[64 + (kr * 4 + 0) * 32 + ml * 4 + kl * 2 + 1] = ftype(res[0]);
                ((threadgroup ftype*)sdata)[64 + (kr * 4 + 1) * 32 + ml * 4 + kl * 2 + 1] = ftype(res[1]);
                ((threadgroup ftype*)sdata)[64 + (kr * 4 + 2) * 32 + ml * 4 + kl * 2 + 1] = ftype(res[2]);
                ((threadgroup ftype*)sdata)[64 + (kr * 4 + 3) * 32 + ml * 4 + kl * 2 + 1] = ftype(res[3]);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_load(sga[0], (const threadgroup ftype*)sdata, 8);
            simdgroup_load(sgb[0], ((const threadgroup ftype*)sdata) + 64, 32);
            simdgroup_load(sgb[1], ((const threadgroup ftype*)sdata) + 72, 32);
            simdgroup_load(sgb[2], ((const threadgroup ftype*)sdata) + 80, 32);
            simdgroup_load(sgb[3], ((const threadgroup ftype*)sdata) + 88, 32);

            SIMDGROUP_MATRIX_FMA(1, 4);

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    SIMDGROUP_MATRIX_STORE((threadgroup FLOAT*)sdata, 4);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if((rx * 8 + (kr*2+kl) * 2) < cst.input_size * cst.batch) {
        if((uz * 8 + ml) < cst.output_slice) {
            xy_out[0] =  activate(ftype4(((threadgroup FLOAT4*)sdata)[((((ml/2) * 4 + (kr*2+kl)) * 2) + 0) * 2 + ml%2] + FLOAT4(biasTerms[uz * 8 + ml])), cst.activation);
        }
    }
    if((rx * 8 + (kr*2+kl) * 2 + 1) < cst.input_size * cst.batch) {
        if((uz * 8 + ml) < cst.output_slice) {
            xy_out[1] =  activate(ftype4(((threadgroup FLOAT4*)sdata)[((((ml/2) * 4 + (kr*2+kl)) * 2) + 1) * 2 + ml%2] + FLOAT4(biasTerms[uz * 8 + ml])), cst.activation);
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
    int idx_n4 = (4 * uz + rcl / 4) < cst.output_slice ? (4 * uz + rcl / 4) : (cst.output_slice - 1);
    int idx_m  = (16 * rx + rcl) < cst.input_size * cst.batch ? (16 * rx + rcl) : (cst.input_size * cst.batch - 1);

    auto xy_wt = wt +  (idx_n4 * cst.input_slice + 0) * 4 + rcl % 4;// [N/4, K/4, N4, K4]
    auto xy_in0  = in + idx_m + cst.input_size * cst.batch * kl;// [K/4, M, K4]
    auto xy_out = out + (4 * uz + 2 * kl) * cst.output_size * cst.batch + idx_m;// [N/4, M, N4]

    int block = (cst.input_slice + cst.block_size - 1) / cst.block_size;
    for (int bi=0; bi<cst.block_size; ++bi) {
        // [N/4, cst.block_size, 2/*scale_bias*/, N4]
        FLOAT4 scale = FLOAT4(dequantScale[2 * (idx_n4 * cst.block_size + bi) + 0]) / (FLOAT)cst.scale_coef;
        FLOAT4 dequant_bias = FLOAT4(dequantScale[2 * (idx_n4 * cst.block_size + bi) + 1]) / (FLOAT)cst.scale_coef;
        int zmin = bi * block;
        int zmax = min(zmin + block, cst.input_slice);

        for (int z = zmin + kl; z < zmax; z += 8) {
            #pragma unroll(4)
            for(int i = 0; i < 4; i++) {
                ((threadgroup ftype4*)sdata)[64 * i + 2* rcl + kl] = *xy_in0;
                xy_in0 += 2 * cst.input_size * cst.batch;
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

    if((16 * rx + rcl) < cst.input_size * cst.batch) {
        if((4 * uz + 2 * kl) < cst.output_slice) {
            xy_out[0] =  activate(ftype4(sdata[(kl * 16 + rcl) * 2 + 0] + FLOAT4(biasTerms[4 * uz + 2 * kl + 0])), cst.activation);
        }
        if((4 * uz + 2 * kl + 1) < cst.output_slice) {
            xy_out[cst.output_size * cst.batch] = activate(ftype4(sdata[(kl * 16 + rcl) * 2 + 1] + FLOAT4(biasTerms[4 * uz + 2 * kl + 1])), cst.activation);
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

    const int size_m = cst.input_size * cst.batch;

    // boundary limit
    int idx_n4 = (4 * uz + rcl / 4) < cst.output_slice ? (4 * uz + rcl / 4) : (cst.output_slice - 1);
    int idx_m0  = (16 * rx + rcl) <  size_m ? (16 * rx + rcl) : (size_m - 1);
    int idx_m1  = (16 * rx + rcl) + size_m / 2 < size_m ? (16 * rx + rcl) + size_m / 2: (size_m - 1);

    auto xy_wt = wt +  (idx_n4 * cst.input_slice + 0) * 4 + rcl % 4;// [N/4, K/4, N4, K4]
    auto xy_in0  = in + idx_m0 + cst.input_size * cst.batch * kl;// [K/4, M2, M/2, K4]
    auto xy_in1  = in + idx_m1 + cst.input_size * cst.batch * kl;// [K/4, M2, M/2, K4]

    auto xy_out0 = out + (4 * uz + 2 * kl) * cst.output_size * cst.batch + idx_m0;// [N/4, M, N4]
    auto xy_out1 = out + (4 * uz + 2 * kl) * cst.output_size * cst.batch + idx_m1;// [N/4, M, N4]

    int block = (cst.input_slice + cst.block_size - 1) / cst.block_size;
    for (int bi=0; bi<cst.block_size; ++bi) {
        // [N/4, cst.block_size, 2/*scale_bias*/, N4]
        FLOAT4 scale = FLOAT4(dequantScale[2 * (idx_n4 * cst.block_size + bi) + 0]) / (FLOAT)cst.scale_coef;
        FLOAT4 dequant_bias = FLOAT4(dequantScale[2 * (idx_n4 * cst.block_size + bi) + 1]) / (FLOAT)cst.scale_coef;
        int zmin = bi * block;
        int zmax = min(zmin + block, cst.input_slice);

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

            xy_in0 += 2 * cst.input_size * cst.batch;
            xy_in1 += 2 * cst.input_size * cst.batch;

        }
    }

    SIMDGROUP_MATRIX_STORE((threadgroup FLOAT*)sdata, 8);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if((16 * rx + rcl) < size_m) {
        if((4 * uz + 2 * kl) < cst.output_slice) {
            xy_out0[0] =  activate(ftype4(sdata[(kl * 32 + rcl) * 2 + 0] + FLOAT4(biasTerms[4 * uz + 2 * kl + 0])), cst.activation);
        }
        if((4 * uz + 2 * kl + 1) < cst.output_slice) {
            xy_out0[cst.output_size * cst.batch] = activate(ftype4(sdata[(kl * 32 + rcl) * 2 + 1] + FLOAT4(biasTerms[4 * uz + 2 * kl + 1])), cst.activation);
        }
    }
    if((16 * rx + rcl) + size_m / 2 < size_m) {
        if((4 * uz + 2 * kl) < cst.output_slice) {
            xy_out1[0] =  activate(ftype4(sdata[(kl * 32 + 16 + rcl) * 2 + 0] + FLOAT4(biasTerms[4 * uz + 2 * kl + 0])), cst.activation);
        }
        if((4 * uz + 2 * kl + 1) < cst.output_slice) {
            xy_out1[cst.output_size * cst.batch] = activate(ftype4(sdata[(kl * 32 + 16 + rcl) * 2 + 1] + FLOAT4(biasTerms[4 * uz + 2 * kl + 1])), cst.activation);
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
    int idx_n40 = (4 * uz + rcl / 4) < cst.output_slice ? (4 * uz + rcl / 4) : (cst.output_slice - 1);
    int idx_n41 = (4 * uz + rcl / 4) + cst.output_slice / 2 < cst.output_slice ? (4 * uz + rcl / 4) + cst.output_slice / 2 : (cst.output_slice - 1);

    int idx_m  = (16 * rx + rcl) < cst.input_size * cst.batch ? (16 * rx + rcl) : (cst.input_size * cst.batch - 1);

    auto xy_wt0 = wt +  (idx_n40 * cst.input_slice + 0) * 4 + (rcl % 4);// [N2, N/8, K/4, N4, K4]
    auto xy_wt1 = wt +  (idx_n41 * cst.input_slice + 0) * 4 + (rcl % 4);// [N2, N/8, K/4, N4, K4]

    auto xy_in0  = in + idx_m + cst.input_size * cst.batch * kl;// [K/4, M, K4]
    auto xy_out = out + (4 * uz + 2 * kl) * cst.output_size * cst.batch + idx_m;// [N2, N/8, M, N4]

    int block = (cst.input_slice + cst.block_size - 1) / cst.block_size;
    for (int bi=0; bi<cst.block_size; ++bi) {
        // [N/4, cst.block_size, 2/*scale_bias*/, N4]
        FLOAT4 scale0 = FLOAT4(dequantScale[2 * (idx_n40 * cst.block_size + bi) + 0]) / (FLOAT)cst.scale_coef;
        FLOAT4 dequant_bias0 = FLOAT4(dequantScale[2 * (idx_n40 * cst.block_size + bi) + 1]) / (FLOAT)cst.scale_coef;
        FLOAT4 scale1 = FLOAT4(dequantScale[2 * (idx_n41 * cst.block_size + bi) + 0]) / (FLOAT)cst.scale_coef;
        FLOAT4 dequant_bias1 = FLOAT4(dequantScale[2 * (idx_n41 * cst.block_size + bi) + 1]) / (FLOAT)cst.scale_coef;
        int zmin = bi * block;
        int zmax = min(zmin + block, cst.input_slice);

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

            xy_in0 += 2 * cst.input_size * cst.batch;

        }
    }

    SIMDGROUP_MATRIX_STORE((threadgroup FLOAT*)sdata, 8);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if((16 * rx + rcl) < cst.input_size * cst.batch) {
        if(4 * uz + 2 * kl < cst.output_slice) {
            xy_out[0] =  activate(ftype4(sdata[(kl * 16 + rcl) * 2 + 0] + FLOAT4(biasTerms[4 * uz + 2 * kl])), cst.activation);
        }
        if(4 * uz + 2 * kl + 1 < cst.output_slice) {
            xy_out[cst.output_size * cst.batch] = activate(ftype4(sdata[(kl * 16 + rcl) * 2 + 1] + FLOAT4(biasTerms[4 * uz + 2 * kl + 1])), cst.activation);
        }
        if(cst.output_slice / 2 + 4 * uz + 2 * kl < cst.output_slice) {
            xy_out[cst.output_slice / 2 * cst.output_size * cst.batch] = activate(ftype4(sdata[((kl + 2) * 16 + rcl) * 2 + 0] + FLOAT4(biasTerms[cst.output_slice / 2 + 4 * uz + 2 * kl])), cst.activation);
        }
        if(cst.output_slice / 2 + 4 * uz + 2 * kl + 1 < cst.output_slice) {
            xy_out[(cst.output_slice / 2 + 1) * cst.output_size * cst.batch] = activate(ftype4(sdata[((kl + 2) * 16 + rcl) * 2 + 1] + FLOAT4(biasTerms[cst.output_slice / 2 + 4 * uz + 2 * kl + 1])), cst.activation);
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
     layout:[N/4, block_size, 2, N4] -> [N/64, N16, block_size, 2, N4]
     index : [uz, no, block_size, 2, nl]
     */
    /** output:
     threadgroup: [M32, N64] -> [M2, N2, N2, N2, M2, M8, N8]
     index [kl, ko/2, ko%2, N2, ml/8, ml%8, N2, N4]

     each thread: M4N4
     layout: [N/4, M, N4] -> [N/64, N4, N4, M/32, M2, M16, N4]
     index : [uz, ko, N4, rx, kl, ml, N4]
     */

    // boundary limit

    int idx_m20  = (rx * 16 + ml) * 2 + 0  < cst.input_size * cst.batch ? (rx * 16 + ml) * 2 + 0 : (cst.input_size * cst.batch - 1);
    int idx_m21  = (rx * 16 + ml) * 2 + 1  < cst.input_size * cst.batch ? (rx * 16 + ml) * 2 + 1 : (cst.input_size * cst.batch - 1);

    int idx_k4 = 0 * 8 + ko * 2 + kl;
    auto xy_in0  = in + idx_k4 * cst.input_size * cst.batch + idx_m20;// [K/4, M, K4]
    auto xy_in1  = in + idx_k4 * cst.input_size * cst.batch + idx_m21;// [K/4, M, K4]

    int idx_wk4 = 0 * 8 + kwl * 4 + 0;
    int idx_n4 = (uz * 16 + no) < cst.output_slice ? (uz * 16 + no) : (cst.output_slice - 1);
    auto xy_wt = wt +  (idx_n4 * cst.input_slice + idx_wk4) * 4 + nl;// [N/4, K/4, N4, K4]

    int idx_sa = (ko * 32 + ml * 2 + 0) * 2 + kl;
    int idx_sb = 1024 + (kwl * 16 + 0) * 64 + no * 4 + nl;
    int block = (cst.input_slice + cst.block_size - 1) / cst.block_size;
    for (int bi=0; bi<cst.block_size; ++bi) {
        // [N/4, cst.block_size, 2/*scale_bias*/, N4]
        FLOAT scale0 = FLOAT(dequantScale[((idx_n4 * cst.block_size + bi) * 2 + 0) * 4 + nl]) / (FLOAT)cst.scale_coef;
        FLOAT dequant_bias0 = FLOAT(dequantScale[((idx_n4 * cst.block_size + bi) * 2 + 1) * 4 + nl]) / (FLOAT)cst.scale_coef;

        int zmin = bi * block;
        int zmax = min(zmin + block, cst.input_slice);

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

            xy_in0 += 8 * cst.input_size * cst.batch;
            xy_in1 += 8 * cst.input_size * cst.batch;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup FLOAT * sdata_c = (threadgroup FLOAT*)sdata + 512*sgitg;

    SIMDGROUP_MATRIX_STORE((threadgroup FLOAT*)sdata_c, 8);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // layout: [N/4, M, N4] -> [N/64, N4, N4, M/32, M2, M16, N4]
    // index : [uz, ko, N4, rx, kl, ml, N4]
    auto xy_out = out + ((uz * 4 + ko) * 4 + 0) * cst.output_size * cst.batch + (rx * 2 + kl) * 16 + ml;// [N/4, M, N4]

    // sdata [M2, N2, N2, N2, M2, M8, N8]
    // index [kl, ko/2, ko%2, N2, ml/8, ml%8, N2, N4]
    if((rx * 32 + kl * 16 + ml) < cst.input_size * cst.batch) {
        if((uz * 4 + ko) * 4 < cst.output_slice) {
            xy_out[0] =  activate(ftype4(((threadgroup FLOAT4*)sdata)[(((kl * 4 + ko) * 2 + 0) * 16 + ml) * 2] + FLOAT4(biasTerms[(uz * 4 + ko) * 4])), cst.activation);
        }
        if((uz * 4 + ko) * 4 + 1 < cst.output_slice) {
            xy_out[cst.output_size * cst.batch] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(((kl * 4 + ko) * 2 + 0) * 16 + ml) * 2 + 1] + FLOAT4(biasTerms[(uz * 4 + ko) * 4 + 1])), cst.activation);
        }
        if((uz * 4 + ko) * 4 + 2 < cst.output_slice) {
            xy_out[cst.output_size * cst.batch * 2] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(((kl * 4 + ko) * 2 + 1) * 16 + ml) * 2] + FLOAT4(biasTerms[(uz * 4 + ko) * 4 + 2])), cst.activation);
        }
        if((uz * 4 + ko) * 4 + 3 < cst.output_slice) {
            xy_out[cst.output_size * cst.batch * 3] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(((kl * 4 + ko) * 2 + 1) * 16 + ml) * 2 + 1] + FLOAT4(biasTerms[(uz * 4 + ko) * 4 + 3])), cst.activation);
        }
    }
}


kernel void conv1x1_gemm_32x64_wquant_sg(const device ftype2 *in            [[buffer(0)]],
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
                            uint                  tiisg[[thread_index_in_simdgroup]],
                            uint                  sgitg[[simdgroup_index_in_threadgroup]]) {
    /*
     // Read:
     ftype 0~255   ---> input: [M32, K8]
     ftype 256~767 ---> weight: [K8, N64]
     // Write:
     ftype 0~2047 ---> input: [M2, N2, N2, N2, M2, M8, N8]
     */
    threadgroup FLOAT4 sdata[512] = {0.f};

    INIT_SIMDGROUP_MATRIX(2, 4, 8);

    int rx = gid.x;// M/32
    int uz = gid.y;// N/64

    int kl = tiitg % 2;// 0~1 -> inner K
    int rcl = tiitg / 2;// 0~63
    int ko = rcl % 2;// 0~1 -> outter K
    int ml = rcl / 2;// 0~31 -> m
    int ni = rcl % 4;// 0~3 -> inner N
    int no = rcl / 4;// 0~15 -> outter N

    /** input:
     threadgroup: [M32, K8]
     each thread: K2
     layout: [K/4, M, K4] -> [K/8, K2, M/32, M32, K2, K2]
     index : [K/8, ko, rx, ml, kl, K2]
     */
    /** weight:
     threadgroup: [K8, N64]
     each thread: K4
     layout: [N/4, K/4, N4, K4] -> [N/64, N16, K/8, K2, N4, K4]
     index : [uz, no, K/8, kl, ni, K4]
     */
    /** output:
     threadgroup: [M32, N64]
     each thread: M4N4
     layout: [N/4, M, N4] -> [N/16, N4, M, N4]
     index : [uz*4+(2*ko+kl), N4, idx_m, N4]
     */

    // boundary limit

    int idx_n40 = (uz * 16 + no) < cst.output_slice ? (uz * 16 + no) : (cst.output_slice - 1);
    int idx_m  = (rx * 32 + ml) < cst.input_size * cst.batch ? (rx * 32 + ml) : (cst.input_size * cst.batch - 1);

    auto xy_wt0 = wt +  ((idx_n40 * cst.input_slice / 2 + 0) * 2 + kl) * 4 + ni;// [N/4, K/4, N4, K4]

    auto xy_in0  = in + ((0 * 2 + ko) * cst.input_size * cst.batch + idx_m) * 2 + kl;// [K/4, M, K2, K2]
    auto xy_out = out + ((4 * uz + 2 * ko + kl) * 4 + 0) * cst.output_size * cst.batch + idx_m;// [N2, N/8, M, N4]

    const int idx_sa = ml * 8 + ko * 4 + kl * 2;
    const int idx_sb = 256 + (kl * 4 + 0) * 64 + rcl;
    int block = (cst.input_slice + cst.block_size - 1) / cst.block_size;
    for (int bi=0; bi<cst.block_size; ++bi) {
        // [N/4, cst.block_size, 2/*scale_bias*/, N4]
        FLOAT4 scale0 = FLOAT4(dequantScale[2 * (idx_n40 * cst.block_size + bi) + 0]) / (FLOAT)cst.scale_coef;
        FLOAT4 dequant_bias0 = FLOAT4(dequantScale[2 * (idx_n40 * cst.block_size + bi) + 1]) / (FLOAT)cst.scale_coef;

        int zmin = bi * block;
        int zmax = min(zmin + block, cst.input_slice);

        for (int z = zmin; z < zmax; z += 2) {
            FLOAT2 data = (FLOAT2)*xy_in0;
            ((threadgroup ftype*)sdata)[idx_sa] = ftype(data[0]);
            ((threadgroup ftype*)sdata)[idx_sa + 1] = ftype(data[1]);

            {
            #ifdef W_QUANT_4
                uchar2 w_int4 = xy_wt0[4*z]; // [N/4, K/4, N4, K4]
                FLOAT4 w4 = FLOAT4((float)(w_int4[0] >> 4) - 8, (float)(w_int4[0] & 15) - 8, (float)(w_int4[1] >> 4) - 8, (float)(w_int4[1] & 15) - 8);
            #elif defined(W_QUANT_8)
                char4 w_int4 = xy_wt0[4*z]; // [N/4, K/4, N4, K4]
                FLOAT4 w4 = FLOAT4((float)w_int4[0], (float)w_int4[1], (float)w_int4[2], (float)w_int4[3]);
            #endif

                FLOAT4 res = w4 * scale0[ni] + dequant_bias0[ni];

                ((threadgroup ftype*)sdata)[idx_sb] = ftype(res[0]);
                ((threadgroup ftype*)sdata)[idx_sb + 64]  = ftype(res[1]);
                ((threadgroup ftype*)sdata)[idx_sb + 128] = ftype(res[2]);
                ((threadgroup ftype*)sdata)[idx_sb + 192] = ftype(res[3]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);


            const threadgroup ftype * sdata_a = (const threadgroup ftype*)sdata + 16*8*(sgitg/2);
            const threadgroup ftype * sdata_b = (const threadgroup ftype*)sdata + 32*8 + 32*(sgitg%2);

            simdgroup_load(sga[0], (const threadgroup ftype*)sdata_a, 8);
            simdgroup_load(sga[1], ((const threadgroup ftype*)sdata_a) + 64, 8);

            simdgroup_load(sgb[0], ((const threadgroup ftype*)sdata_b) + 0,  64);
            simdgroup_load(sgb[1], ((const threadgroup ftype*)sdata_b) + 8,  64);
            simdgroup_load(sgb[2], ((const threadgroup ftype*)sdata_b) + 16, 64);
            simdgroup_load(sgb[3], ((const threadgroup ftype*)sdata_b) + 24, 64);

            SIMDGROUP_MATRIX_FMA(2, 4);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            xy_in0 += 4 * cst.input_size * cst.batch;
        }
    }

    threadgroup FLOAT * sdata_c = (threadgroup FLOAT*)sdata + 512*sgitg;

    SIMDGROUP_MATRIX_STORE((threadgroup FLOAT*)sdata_c, 8);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // sdata [M2, N2, N2, N2, M2, M8, N8]
    // index [ml/16, ko, kl, N2, (ml/8)%2, ml%8, N2, N4]
    if((rx * 32 + ml) < cst.input_size * cst.batch) {
        if((4 * uz + 2 * ko + kl) * 4 < cst.output_slice) {
            xy_out[0] =  activate(ftype4(sdata[(((ml/16 * 4 + 2 * ko + kl) * 2 + 0) * 16 + ml % 16) * 2] + FLOAT4(biasTerms[(4 * uz + 2 * ko + kl) * 4])), cst.activation);
        }
        if((4 * uz + 2 * ko + kl) * 4 + 1 < cst.output_slice) {
            xy_out[cst.output_size * cst.batch] = activate(ftype4(sdata[(((ml/16 * 4 + 2 * ko + kl) * 2 + 0) * 16 + ml % 16) * 2 + 1] + FLOAT4(biasTerms[(4 * uz + 2 * ko + kl) * 4 + 1])), cst.activation);
        }
        if((4 * uz + 2 * ko + kl) * 4 + 2 < cst.output_slice) {
            xy_out[cst.output_size * cst.batch * 2] = activate(ftype4(sdata[(((ml/16 * 4 + 2 * ko + kl) * 2 + 1) * 16 + ml % 16) * 2] + FLOAT4(biasTerms[(4 * uz + 2 * ko + kl) * 4 + 2])), cst.activation);
        }
        if((4 * uz + 2 * ko + kl) * 4 + 3 < cst.output_slice) {
            xy_out[cst.output_size * cst.batch * 3] = activate(ftype4(sdata[(((ml/16 * 4 + 2 * ko + kl) * 2 + 1) * 16 + ml % 16) * 2 + 1] + FLOAT4(biasTerms[(4 * uz + 2 * ko + kl) * 4 + 3])), cst.activation);
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

    if(idx_n4 >= cst.output_slice || idx_k4 >= cst.input_slice) {
        return;
    }

    int block = (cst.input_slice + cst.block_size - 1) / cst.block_size;


    int bi = idx_k4 / block;
    // [N/4, cst.block_size, 2/*scale_bias*/, N4]
    FLOAT scale = FLOAT(((const device ftype *)dequantScale)[((idx_n4 * cst.block_size + bi) * 2 + 0) * 4 + idx_nl]) / (FLOAT)cst.scale_coef;
    FLOAT dequant_bias = FLOAT(((const device ftype *)dequantScale)[((idx_n4 * cst.block_size + bi) * 2 + 1) * 4 + idx_nl]) / (FLOAT)cst.scale_coef;

#ifdef W_QUANT_3
    auto wt_base = wi + (idx_n4 * cst.input_slice + idx_k4) * 6;
#elif !defined(W_QUANT_2)
    auto xy_wi = wi + (idx_n4 * cst.input_slice + idx_k4) * 4 + idx_nl;// [N/4, K/4, N4, K4]
#endif
    auto xy_wf = wf + ((idx_n4 * ((cst.input_slice+3)/4) + idx_k16) * 4 + idx_nl) * 4;// [N/4, K/4, N4, K4]

    #ifdef W_QUANT_2
    #if W_ALIGN_K16_PROTECT
    // Tail-protected path: 4B-aligned uchar4 load per K/4-slice (vs scattered
    // 1-byte loads), nl byte extracted in-register.
    {
        auto xy_wi4 = (const device uchar4*)wi + idx_n4 * cst.input_slice + idx_k4;
        for(int k = 0; k < 4; k++) {
            if(idx_k4 + k >= cst.input_slice) { xy_wf[k] = ftype4(0); continue; }
            uchar b = xy_wi4[k][idx_nl];
            FLOAT4 w4 = FLOAT4((float)((b >> 6) & 3) - 2, (float)((b >> 4) & 3) - 2,
                                (float)((b >> 2) & 3) - 2, (float)( b       & 3) - 2);
            xy_wf[k] = (ftype4)(w4 * scale + dequant_bias);
        }
    }
    #else
    // ic%16==0 here, so the 16B tile (4 K/4-slices x 4 N4 bytes) is 16B-aligned:
    // one uint4 load replaces four scattered byte loads.
    {
        uint4 w16 = ((const device uint4*)wi)[idx_n4 * (cst.input_slice / 4) + idx_k16];
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
        if(idx_k4 + k >= cst.input_slice) { xy_wf[k] = ftype4(0); continue; }
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
            if(idx_k4 + k >= cst.input_slice) {
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
            if(idx_k4 + k >= cst.input_slice) {
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
     layout:[N/4, block_size, 2, N4] -> [N/64, N16, block_size, 2, N4]
     index : [uz, no, block_size, 2, nl]
     */
    /** output:
     threadgroup: [M32, N64] -> [M32, N4, N16]
     index [mlc, nlc, N16]

     each thread: N16
     layout: [N/4, M, N4] -> [N/64, N4, N4, M/32, M32, N4]
     index : [uz, nlc, N4, rx, mlc, N4]
     */

    // boundary limit
    int idx_m40  = (rx * 8 + ml) * 4 + 0  < cst.input_size * cst.batch ? (rx * 8 + ml) * 4 + 0 : (cst.input_size * cst.batch - 1);
    int idx_m41  = (rx * 8 + ml) * 4 + 1  < cst.input_size * cst.batch ? (rx * 8 + ml) * 4 + 1 : (cst.input_size * cst.batch - 1);
    int idx_m42  = (rx * 8 + ml) * 4 + 2  < cst.input_size * cst.batch ? (rx * 8 + ml) * 4 + 2 : (cst.input_size * cst.batch - 1);
    int idx_m43  = (rx * 8 + ml) * 4 + 3  < cst.input_size * cst.batch ? (rx * 8 + ml) * 4 + 3 : (cst.input_size * cst.batch - 1);

    int idx_k4 = 0 * 16 + kl;
    auto xy_in0  = in + idx_k4 * cst.input_size * cst.batch + idx_m40;// [K/4, M, K4]
    auto xy_in1  = in + idx_k4 * cst.input_size * cst.batch + idx_m41;// [K/4, M, K4]
    auto xy_in2  = in + idx_k4 * cst.input_size * cst.batch + idx_m42;// [K/4, M, K4]
    auto xy_in3  = in + idx_k4 * cst.input_size * cst.batch + idx_m43;// [K/4, M, K4]

    int idx_wk16 = (0 * 2 + kwl) * 2 + 0;

    int idx_n4 = (uz * 16 + no) < cst.output_slice ? (uz * 16 + no) : (cst.output_slice - 1);
    auto xy_wt = wt +  (idx_n4 * ((cst.input_slice+3)/4) + idx_wk16) * 4 + nl;// [N/4, K/16, N4, K4, K4]

    int idx_sa = (ml * 4 + 0) * 16 + kl; // [M8, M4, K16] x [K4]
    int idx_sb = 512 + ((no * 4 + nl) * 2 + kwl) * 8 + 0; // [N16 N4, K2, K8] x [K4]
    int block = (cst.input_slice + cst.block_size - 1) / cst.block_size;

    for (int bi=0; bi<cst.block_size; ++bi) {
        int zmin = bi * block;
        int zmax = min(zmin + block, cst.input_slice);

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

            xy_in0 += 16 * cst.input_size * cst.batch;
            xy_in1 += 16 * cst.input_size * cst.batch;
            xy_in2 += 16 * cst.input_size * cst.batch;
            xy_in3 += 16 * cst.input_size * cst.batch;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto tC = tensor<threadgroup FLOAT, dextents<int32_t, 2>, tensor_inline>((threadgroup FLOAT*)sdata, dextents<int32_t, 2>(N, M)); // [M , N]
    cT.store(tC);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // each thread: N16
    // layout: [N/4, M, N4] -> [N/64, N4, N4, M/32, M32, N4]
    // index : [uz, nlc, N4, rx, mlc, N4]

    auto xy_out = out + ((uz * 4 + nlc) * 4 + 0) * cst.output_size * cst.batch + (rx * 32 + mlc);// [N/4, M, N4]
    // sdata: [M32, N64] -> [M32, N4, N16]
    // index [mlc, nlc, N16]
    if((rx * 32 + mlc) < cst.input_size * cst.batch) {
        if((uz * 4 + nlc) * 4 < cst.output_slice) {
            xy_out[0] =  activate(ftype4(((threadgroup FLOAT4*)sdata)[(mlc * 4 + nlc) * 4 + 0] + FLOAT4(biasTerms[(uz * 4 + nlc) * 4])), cst.activation);
        }
        if((uz * 4 + nlc) * 4 + 1 < cst.output_slice) {
            xy_out[cst.output_size * cst.batch] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(mlc * 4 + nlc) * 4 + 1] + FLOAT4(biasTerms[(uz * 4 + nlc) * 4 + 1])), cst.activation);
        }
        if((uz * 4 + nlc) * 4 + 2 < cst.output_slice) {
            xy_out[cst.output_size * cst.batch * 2] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(mlc * 4 + nlc) * 4 + 2] + FLOAT4(biasTerms[(uz * 4 + nlc) * 4 + 2])), cst.activation);
        }
        if((uz * 4 + nlc) * 4 + 3 < cst.output_slice) {
            xy_out[cst.output_size * cst.batch * 3] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(mlc * 4 + nlc) * 4 + 3] + FLOAT4(biasTerms[(uz * 4 + nlc) * 4 + 3])), cst.activation);
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
     layout:[N/4, block_size, 2, N4] -> [N/64, N16, block_size, 2, N4]
     index : [uz, no, block_size, 2, nl]
     */
    /** output:
     threadgroup: [M32, N64] -> [M32, N4, N16]
     index [mlc, nlc, N16]

     each thread: N16
     layout: [N/4, M, N4] -> [N/64, N4, N4, M/32, M32, N4]
     index : [uz, nlc, N4, rx, mlc, N4]
     */

    // boundary limit
    int idx_m20  = (rx * 16 + ml) * 2 + 0  < cst.input_size * cst.batch ? (rx * 16 + ml) * 2 + 0 : (cst.input_size * cst.batch - 1);
    int idx_m21  = (rx * 16 + ml) * 2 + 1  < cst.input_size * cst.batch ? (rx * 16 + ml) * 2 + 1 : (cst.input_size * cst.batch - 1);

    int idx_k4 = 0 * 8 + kl;
    auto xy_in0  = in + idx_k4 * cst.input_size * cst.batch + idx_m20;// [K/4, M, K4]
    auto xy_in1  = in + idx_k4 * cst.input_size * cst.batch + idx_m21;// [K/4, M, K4]

    int idx_wk16 = 0 * 2 + kwl;

    int idx_n4 = (uz * 16 + no) < cst.output_slice ? (uz * 16 + no) : (cst.output_slice - 1);
    auto xy_wt = wt +  (idx_n4 * ((cst.input_slice+3)/4) + idx_wk16) * 4 + nl;// [N/4, K/16, N4, K4, K4]

    int idx_sa = (ml * 2 + 0) * 8 + kl; // [M16, M2, K8] x [K4]
    int idx_sb = 256 + ((no * 4 + nl) * 2 + kwl) * 4 + 0; // [N16 N4, K2, K4] x [K4]
    int block = (cst.input_slice + cst.block_size - 1) / cst.block_size;

    for (int bi=0; bi<cst.block_size; ++bi) {
    #if defined(W_QUANT_4) || defined(W_QUANT_8)
        // [N/4, cst.block_size, 2/*scale_bias*/, N4]
        FLOAT scale0 = FLOAT(dequantScale[((idx_n4 * cst.block_size + bi) * 2 + 0) * 4 + nl]) / (FLOAT)cst.scale_coef;
        FLOAT dequant_bias0 = FLOAT(dequantScale[((idx_n4 * cst.block_size + bi) * 2 + 1) * 4 + nl]) / (FLOAT)cst.scale_coef;

    #endif
        int zmin = bi * block;
        int zmax = min(zmin + block, cst.input_slice);

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
            if (idx_k4 + z < cst.input_slice) {
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

            xy_in0 += 8 * cst.input_size * cst.batch;
            xy_in1 += 8 * cst.input_size * cst.batch;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto tC = tensor<threadgroup FLOAT, dextents<int32_t, 2>, tensor_inline>((threadgroup FLOAT*)sdata, dextents<int32_t, 2>(N, M)); // [M , N]
    cT.store(tC);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // each thread: N16
    // layout: [N/4, M, N4] -> [N/64, N4, N4, M/32, M32, N4]
    // index : [uz, nlc, N4, rx, mlc, N4]

    auto xy_out = out + ((uz * 4 + nlc) * 4 + 0) * cst.output_size * cst.batch + (rx * 32 + mlc);// [N/4, M, N4]
    // sdata: [M32, N64] -> [M32, N4, N16]
    // index [mlc, nlc, N16]
    if((rx * 32 + mlc) < cst.input_size * cst.batch) {
        if((uz * 4 + nlc) * 4 < cst.output_slice) {
            xy_out[0] =  activate(ftype4(((threadgroup FLOAT4*)sdata)[(mlc * 4 + nlc) * 4 + 0] + FLOAT4(biasTerms[(uz * 4 + nlc) * 4])), cst.activation);
        }
        if((uz * 4 + nlc) * 4 + 1 < cst.output_slice) {
            xy_out[cst.output_size * cst.batch] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(mlc * 4 + nlc) * 4 + 1] + FLOAT4(biasTerms[(uz * 4 + nlc) * 4 + 1])), cst.activation);
        }
        if((uz * 4 + nlc) * 4 + 2 < cst.output_slice) {
            xy_out[cst.output_size * cst.batch * 2] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(mlc * 4 + nlc) * 4 + 2] + FLOAT4(biasTerms[(uz * 4 + nlc) * 4 + 2])), cst.activation);
        }
        if((uz * 4 + nlc) * 4 + 3 < cst.output_slice) {
            xy_out[cst.output_size * cst.batch * 3] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(mlc * 4 + nlc) * 4 + 3] + FLOAT4(biasTerms[(uz * 4 + nlc) * 4 + 3])), cst.activation);
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
     layout:[N/4, block_size, 2, N4] -> [N/64, N16, block_size, 2, N4]
     index : [uz, no, block_size, 2, nl]
     */
    /** output:
     threadgroup: [M32, N64] -> [M2, N2, N2, N2, M2, M8, N8]
     index [kl, ko/2, ko%2, N2, ml/8, ml%8, N2, N4]

     each thread: N16
     layout: [N/4, M, N4] -> [N/64, N4, N4, M/32, M2, M16, N4]
     index : [uz, ko, N4, rx, kl, ml, N4]
     */

    // boundary limit
    int idx_m20  = (rx * 16 + ml) * 2 + 0  < cst.input_size * cst.batch ? (rx * 16 + ml) * 2 + 0 : (cst.input_size * cst.batch - 1);
    int idx_m21  = (rx * 16 + ml) * 2 + 1  < cst.input_size * cst.batch ? (rx * 16 + ml) * 2 + 1 : (cst.input_size * cst.batch - 1);

    int idx_k4 = 0 * 8 + ko * 2 + kl;
    auto xy_in0  = in + idx_k4 * cst.input_size * cst.batch + idx_m20;// [K/4, M, K4]
    auto xy_in1  = in + idx_k4 * cst.input_size * cst.batch + idx_m21;// [K/4, M, K4]

    int idx_wk16 = 0 * 2 + kwl;

    int idx_n4 = (uz * 16 + no) < cst.output_slice ? (uz * 16 + no) : (cst.output_slice - 1);
    auto xy_wt = wt +  (idx_n4 * ((cst.input_slice+3)/4) + idx_wk16) * 4 + nl;// [N/4, K/16, N4, K4, K4]

    int idx_sa = (ko * 32 + ml * 2 + 0) * 2 + kl;
    int idx_sb = 1024 + (kwl * 16 + 0) * 64 + no * 4 + nl;
    int block = (cst.input_slice + cst.block_size - 1) / cst.block_size;

    for (int bi=0; bi<cst.block_size; ++bi) {
    #if defined(W_QUANT_4) || defined(W_QUANT_8)
        // [N/4, cst.block_size, 2/*scale_bias*/, N4]
        FLOAT scale0 = FLOAT(dequantScale[((idx_n4 * cst.block_size + bi) * 2 + 0) * 4 + nl]) / (FLOAT)cst.scale_coef;
        FLOAT dequant_bias0 = FLOAT(dequantScale[((idx_n4 * cst.block_size + bi) * 2 + 1) * 4 + nl]) / (FLOAT)cst.scale_coef;

    #endif
        int zmin = bi * block;
        int zmax = min(zmin + block, cst.input_slice);

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
            if (idx_k4 + z < cst.input_slice) {
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

            xy_in0 += 8 * cst.input_size * cst.batch;
            xy_in1 += 8 * cst.input_size * cst.batch;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup FLOAT * sdata_c = (threadgroup FLOAT*)sdata + 512*sgitg;

    SIMDGROUP_MATRIX_STORE((threadgroup FLOAT*)sdata_c, 8);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // layout: [N/4, M, N4] -> [N/64, N4, N4, M/32, M2, M16, N4]
    // index : [uz, ko, N4, rx, kl, ml, N4]
    auto xy_out = out + ((uz * 4 + ko) * 4 + 0) * cst.output_size * cst.batch + (rx * 2 + kl) * 16 + ml;// [N/4, M, N4]

    // sdata [M2, N2, N2, N2, M2, M8, N8]
    // index [kl, ko/2, ko%2, N2, ml/8, ml%8, N2, N4]
    if((rx * 32 + kl * 16 + ml) < cst.input_size * cst.batch) {
        if((uz * 4 + ko) * 4 < cst.output_slice) {
            xy_out[0] =  activate(ftype4(((threadgroup FLOAT4*)sdata)[(((kl * 4 + ko) * 2 + 0) * 16 + ml) * 2] + FLOAT4(biasTerms[(uz * 4 + ko) * 4])), cst.activation);
        }
        if((uz * 4 + ko) * 4 + 1 < cst.output_slice) {
            xy_out[cst.output_size * cst.batch] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(((kl * 4 + ko) * 2 + 0) * 16 + ml) * 2 + 1] + FLOAT4(biasTerms[(uz * 4 + ko) * 4 + 1])), cst.activation);
        }
        if((uz * 4 + ko) * 4 + 2 < cst.output_slice) {
            xy_out[cst.output_size * cst.batch * 2] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(((kl * 4 + ko) * 2 + 1) * 16 + ml) * 2] + FLOAT4(biasTerms[(uz * 4 + ko) * 4 + 2])), cst.activation);
        }
        if((uz * 4 + ko) * 4 + 3 < cst.output_slice) {
            xy_out[cst.output_size * cst.batch * 3] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(((kl * 4 + ko) * 2 + 1) * 16 + ml) * 2 + 1] + FLOAT4(biasTerms[(uz * 4 + ko) * 4 + 3])), cst.activation);
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
    int idx_n4 = (4 * uz + rcl / 4) < cst.output_slice ? (4 * uz + rcl / 4) : (cst.output_slice - 1);
    int idx_m  = (16 * rx + rcl) < cst.input_size * cst.batch ? (16 * rx + rcl) : (cst.input_size * cst.batch - 1);

    auto xy_wt = wt +  (idx_n4 * cst.input_slice + 0) * 4 + rcl % 4;// [N/4, K/4, N4, K4]
    auto xy_in0  = in + idx_m + cst.input_size * cst.batch * kl;// [K/4, M, K4]
    auto xy_out = out + (4 * uz + 2 * kl) * cst.output_size * cst.batch + idx_m;// [N/4, M, N4]

    for (int z = kl; z < cst.input_slice; z += 2) {
        ((threadgroup ftype4*)sdata)[2* rcl + kl] = (*xy_in0);
        xy_in0 += 2 * cst.input_size * cst.batch;

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

    if((16 * rx + rcl) < cst.input_size * cst.batch) {
        if((4 * uz + 2 * kl) < cst.output_slice) {
            xy_out[0] =  activate(ftype4(sdata[(kl * 16 + rcl) * 2 + 0] + FLOAT4(biasTerms[4 * uz + 2 * kl + 0])), cst.activation);
        }
        if((4 * uz + 2 * kl + 1) < cst.output_slice) {
            xy_out[cst.output_size * cst.batch] = activate(ftype4(sdata[(kl * 16 + rcl) * 2 + 1] + FLOAT4(biasTerms[4 * uz + 2 * kl + 1])), cst.activation);
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

    const int size_m = cst.input_size * cst.batch;

    // boundary limit
    int idx_n4 = (4 * uz + rcl / 4) < cst.output_slice ? (4 * uz + rcl / 4) : (cst.output_slice - 1);
    int idx_m0  = (16 * rx + rcl) <  size_m ? (16 * rx + rcl) : (size_m - 1);
    int idx_m1  = (16 * rx + rcl) + size_m / 2 < size_m ? (16 * rx + rcl) + size_m / 2: (size_m - 1);

    auto xy_wt = wt +  (idx_n4 * cst.input_slice + 0) * 4 + rcl % 4;// [N/4, K/4, N4, K4]
    auto xy_in0  = in + idx_m0 + cst.input_size * cst.batch * kl;// [K/4, M2, M/2, K4]
    auto xy_in1  = in + idx_m1 + cst.input_size * cst.batch * kl;// [K/4, M2, M/2, K4]

    auto xy_out0 = out + (4 * uz + 2 * kl) * cst.output_size * cst.batch + idx_m0;// [N/4, M, N4]
    auto xy_out1 = out + (4 * uz + 2 * kl) * cst.output_size * cst.batch + idx_m1;// [N/4, M, N4]

    for (int z = kl; z < cst.input_slice; z += 2) {
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

        xy_in0 += 2 * cst.input_size * cst.batch;
        xy_in1 += 2 * cst.input_size * cst.batch;

    }

    SIMDGROUP_MATRIX_STORE((threadgroup FLOAT*)sdata, 8);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if((16 * rx + rcl) < size_m) {
        if((4 * uz + 2 * kl) < cst.output_slice) {
            xy_out0[0] =  activate(ftype4(sdata[(kl * 32 + rcl) * 2 + 0] + FLOAT4(biasTerms[4 * uz + 2 * kl + 0])), cst.activation);
        }
        if((4 * uz + 2 * kl + 1) < cst.output_slice) {
            xy_out0[cst.output_size * cst.batch] = activate(ftype4(sdata[(kl * 32 + rcl) * 2 + 1] + FLOAT4(biasTerms[4 * uz + 2 * kl + 1])), cst.activation);
        }
    }
    if((16 * rx + rcl) + size_m / 2 < size_m) {
        if((4 * uz + 2 * kl) < cst.output_slice) {
            xy_out1[0] =  activate(ftype4(sdata[(kl * 32 + 16 + rcl) * 2 + 0] + FLOAT4(biasTerms[4 * uz + 2 * kl + 0])), cst.activation);
        }
        if((4 * uz + 2 * kl + 1) < cst.output_slice) {
            xy_out1[cst.output_size * cst.batch] = activate(ftype4(sdata[(kl * 32 + 16 + rcl) * 2 + 1] + FLOAT4(biasTerms[4 * uz + 2 * kl + 1])), cst.activation);
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
//   buffer(5): dequantScale [N/4, block_size, 2, N4]
//   buffer(6): unused placeholder (host binds mWeight to satisfy the
//              Metal validation layer)
//
// Weight int4 unpack: follows conv1x1_gemm_32x64_wquant_split_k_sg
// (the in-shader sg_matrix path that actually reads mWeight int4 layout).
// The int4 buffer memory layout is [N/4, K/4-slice, N4-inner, uchar2] where
// each uchar2 holds 4 K-nibbles for one (K/4-slice, N4-inner). Successive
// uchar2 in memory step by N4-inner (+1) or by K/4-slice (+4).
//
// IMPORTANT: mWeight is NOT `[N/4, K/16, N4, K4, K4]` (that's the fp16
// dequanted mTempWeight layout). The W_QUANT_4 branch in the fp16 tensor
// API kernel (conv1x1_gemm_32x64_split_k_sg) uses `MNN::uchar4x2 w = wt[z]`
// which reads 4 successive uchar2 slots — under mWeight's real layout these
// are 4 different N4-inners of the SAME K/4-slice, not 4 K/4-slices for the
// same N4-inner. That's why buffer(3) is always fed mTempWeight (fp16) in
// that kernel; the W_QUANT_4 branch is effectively dead.
//
// For a thread with (nl, kwl) reading K = 4*(kwl*4 + i) + j for i,j ∈ [0..3]:
//   uchar2 w = xy_wt_i4[(z_slice + kwl*4 + i) * 4]  // z_slice is K/4-slice index
//   w[0]>>4 -> K col 0 of row i,  w[0]&0xF -> col 1
//   w[1]>>4 -> K col 2 of row i,  w[1]&0xF -> col 3
// After unpack, apply `w * scale + (bias - 8*scale)` as in the split_k kernel.
//
// scale/bias per quant block:
//   scale_coef is host-side compensation kept in cst.scale_coef; the
//   physical dequantScale buffer stores s*coef, so we divide by coef.
//   Layout: [N/4, block_size, 2, N4] indexed as
//     scale = dequantScale[((idx_n4 * block_size + bi) * 2 + 0) * 4 + nl]
//     bias  = dequantScale[((idx_n4 * block_size + bi) * 2 + 1) * 4 + nl]
//
// K_TILE = 8 per iter and quant block spans block = ceil(ic_4 / block_size)
// K-slices (each K-slice = 4 K along contiguous K4). This kernel is only
// selected when block_size divides ic_4 cleanly and block >= 2 K-slices,
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
    int idx_m20 = (rx * 16 + ml) * 2 + 0 < cst.input_size * cst.batch ? (rx * 16 + ml) * 2 + 0 : (cst.input_size * cst.batch - 1);
    int idx_m21 = (rx * 16 + ml) * 2 + 1 < cst.input_size * cst.batch ? (rx * 16 + ml) * 2 + 1 : (cst.input_size * cst.batch - 1);

    int idx_k4 = 0 * 8 + kl;
    auto xy_in0 = in + idx_k4 * cst.input_size * cst.batch + idx_m20; // [K/4, M, K4]
    auto xy_in1 = in + idx_k4 * cst.input_size * cst.batch + idx_m21; // [K/4, M, K4]

    // Weight tile mapping.
    //   fp16 mTempWeight  layout: [N/4, K/16, N4, K4, K4]  (ftype4x4 slots)
    //   int2 mWeight (Q2) layout: [N/4, K/4-slice, N4, uchar]
    //   int3 mWeight (Q3) layout: [N/4, K/4-slice, 6 bytes] (shared tile across N4)
    //   int4 mWeight (Q4) layout: [N/4, K/4-slice, N4, uchar2]
    //   int8 mWeight (Q8) layout: [N/4, K/4-slice, N4, char4]
    int idx_n4   = (uz * 16 + no) < cst.output_slice ? (uz * 16 + no) : (cst.output_slice - 1);
    // Reinterpret buffer(3) as the packed-quant element type.
    // W2: uchar4 = 4 bytes per K/4-slice (one per N4 row), thread nl picks byte.
    // W3: uchar*  = 6 bytes per K/4-slice (shared tile), thread accesses tilePtr[nl].
    // W4: uchar2  = 2 bytes × 4 N4 rows per K/4-slice.
    // W8: char4   = 4 bytes × 4 N4 rows per K/4-slice.
#ifdef W_QUANT_2
    auto xy_wt_i2 = ((const device uchar*)wt_int2) + (idx_n4 * cst.input_slice) * 4 + nl;
#elif defined(W_QUANT_3)
    auto xy_wt_i3 = wt_int3 + idx_n4 * cst.input_slice * 6;
#elif defined(W_QUANT_4)
    auto xy_wt_i4 = ((const device uchar2*)wt_int4) + (idx_n4 * cst.input_slice + 0) * 4 + nl;
#else // W_QUANT_8
    auto xy_wt_i8 = ((const device char4*)wt_int8)  + (idx_n4 * cst.input_slice + 0) * 4 + nl;
#endif

    int idx_sa = (ml * 2 + 0) * 8 + kl;                                  // input write offset in sdata (ftype4 units)
    int idx_sb = 256 + ((no * 4 + nl) * 2 + kwl) * 4 + 0;                // weight write offset (ftype4 units, base = 1024/4 = 256)
    int block  = (cst.input_slice + cst.block_size - 1) / cst.block_size;

    for (int bi = 0; bi < cst.block_size; ++bi) {
        // Per-block scale/bias (same as split_k_sg branch of gemm_32x64_split_k_sg).
        // Layout of dequantScale: [N/4, block_size, 2, N4]. `nl` (0..3) picks the
        // N4-inner slot corresponding to this thread's OC row within the (uz*16+no)
        // group. The `/scale_coef` divides out the host-side compensation.
        FLOAT scale0        = FLOAT(dequantScale[((idx_n4 * cst.block_size + bi) * 2 + 0) * 4 + nl]) / (FLOAT)cst.scale_coef;
        FLOAT dequant_bias0 = FLOAT(dequantScale[((idx_n4 * cst.block_size + bi) * 2 + 1) * 4 + nl]) / (FLOAT)cst.scale_coef;
        int zmin = bi * block;
        int zmax = min(zmin + block, cst.input_slice);

        for (int z = zmin; z < zmax; z += 8) {
            // ---- Phase 1: weight load (in-kernel quant unpack) ----
            //
            // Outer z counts K/4-slices; one iter per quant block since
            // block == zmax-zmin K/4-slices and we jump by 8 (=1 block for the
            // Qwen3-0.6B W4-block32 shape). Inner unrolled i=0..3 reads 4
            // K/4-slices for THIS thread's (nl) N4-inner row, corresponding to
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
                // W3: 6-byte tile per (4 OC, 4 IC) per K/4-slice.
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
                //   uchar2 w = xy_wt_i4[K4slice_index * 4]
                //   -> w[0]>>4 (K col 0), w[0]&0xF (col 1),
                //      w[1]>>4 (col 2),   w[1]&0xF (col 3)
                // K4slice_index for row i = z + kwl*4 + i.
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
            if (idx_k4 + z < cst.input_slice) {
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

            xy_in0 += 8 * cst.input_size * cst.batch;
            xy_in1 += 8 * cst.input_size * cst.batch;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto tC = tensor<threadgroup FLOAT, dextents<int32_t, 2>, tensor_inline>((threadgroup FLOAT*)sdata, dextents<int32_t, 2>(N, M));
    cT.store(tC);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Epilogue: write [N/4, M, N4] layout with bias + activation.
    // sdata after cT.store: [M32, N64] -> [M32, N4, N16]
    // per-thread contribution: N16 at [mlc, nlc, N16]
    auto xy_out = out + ((uz * 4 + nlc) * 4 + 0) * cst.output_size * cst.batch + (rx * 32 + mlc);
    if ((rx * 32 + mlc) < cst.input_size * cst.batch) {
        if ((uz * 4 + nlc) * 4 < cst.output_slice) {
            xy_out[0] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(mlc * 4 + nlc) * 4 + 0] + FLOAT4(biasTerms[(uz * 4 + nlc) * 4])), cst.activation);
        }
        if ((uz * 4 + nlc) * 4 + 1 < cst.output_slice) {
            xy_out[cst.output_size * cst.batch] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(mlc * 4 + nlc) * 4 + 1] + FLOAT4(biasTerms[(uz * 4 + nlc) * 4 + 1])), cst.activation);
        }
        if ((uz * 4 + nlc) * 4 + 2 < cst.output_slice) {
            xy_out[cst.output_size * cst.batch * 2] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(mlc * 4 + nlc) * 4 + 2] + FLOAT4(biasTerms[(uz * 4 + nlc) * 4 + 2])), cst.activation);
        }
        if ((uz * 4 + nlc) * 4 + 3 < cst.output_slice) {
            xy_out[cst.output_size * cst.batch * 3] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(mlc * 4 + nlc) * 4 + 3] + FLOAT4(biasTerms[(uz * 4 + nlc) * 4 + 3])), cst.activation);
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

    int idx_n4 = (uz * 16 + tiitg / 8) < cst.output_slice ? (uz * 16 + tiitg / 8) : (cst.output_slice - 1);
    int block  = (cst.input_slice + cst.block_size - 1) / cst.block_size;

    // A tile: 8 rows x 8 K4-slices, one ftype4 per thread on the first 64 threads.
    int aml = (int)tiitg / 8;
    int akl = (int)tiitg % 8;
    bool aAct = aml < M;
    int am = rx * M + aml;
    int amc = am < cst.input_size * cst.batch ? am : (cst.input_size * cst.batch - 1);

    // v2 register prefetch (identical mapping to the 32-row kernel)
    uint2 w2pre;
    {
        const device uint2 *wq = (const device uint2 *)wt_int4;
        int no0 = (int)tiitg / 8;
        int kw0 = ((int)tiitg % 8) / 4;
        int i0  = (int)tiitg % 4;
        int n4_0 = (uz * 16 + no0) < cst.output_slice ? (uz * 16 + no0) : (cst.output_slice - 1);
        w2pre = wq[n4_0 * cst.input_slice + (kw0 * 4 + i0)];
    }
    for (int bi = 0; bi < cst.block_size; ++bi) {
        int zmin = bi * block;
        int zmax = min(zmin + block, cst.input_slice);

        ftype4 sc4t_h = *(const device ftype4 *)(dequantScale + ((idx_n4 * cst.block_size + bi) * 2 + 0) * 4);
        ftype4 db4t_h = *(const device ftype4 *)(dequantScale + ((idx_n4 * cst.block_size + bi) * 2 + 1) * 4);
        FLOAT4 sc4_h = FLOAT4(sc4t_h) / (FLOAT)cst.scale_coef;
        FLOAT4 db4_h = FLOAT4(db4t_h) / (FLOAT)cst.scale_coef;

        for (int z = zmin; z < zmax; z += 8) {
            FLOAT4 wb[4];
            {
                const device uint2 *wq = (const device uint2 *)wt_int4;
                int no = (int)tiitg / 8;
                int kw = ((int)tiitg % 8) / 4;
                int i  = (int)tiitg % 4;
                int idx_n4c = (uz * 16 + no) < cst.output_slice ? (uz * 16 + no) : (cst.output_slice - 1);
                uint2 w2 = w2pre;
                w2pre = (z + 8 < cst.input_slice) ? wq[idx_n4c * cst.input_slice + (z + 8 + kw * 4 + i)] : w2pre;
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
                if (akl + z < cst.input_slice) {
                    ((threadgroup ftype4*)sdata)[aml * 8 + akl] = (ftype4)in[(z + akl) * cst.input_size * cst.batch + amc];
                } else {
                    ((threadgroup ftype4*)sdata)[aml * 8 + akl] = (ftype4)(0);
                }
                #else
                ((threadgroup ftype4*)sdata)[aml * 8 + akl] = (ftype4)in[(z + akl) * cst.input_size * cst.batch + amc];
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
        if (gm >= cst.input_size * cst.batch) continue;
        int n4 = uz * 16 + n4l;
        if (n4 >= cst.output_slice) continue;
        FLOAT4 v = FLOAT4(((threadgroup FLOAT*)sdata)[m * 64 + n4l * 4 + 0],
                          ((threadgroup FLOAT*)sdata)[m * 64 + n4l * 4 + 1],
                          ((threadgroup FLOAT*)sdata)[m * 64 + n4l * 4 + 2],
                          ((threadgroup FLOAT*)sdata)[m * 64 + n4l * 4 + 3]);
        out[n4 * cst.output_size * cst.batch + gm] = activate(ftype4(v + FLOAT4(biasTerms[n4])), cst.activation);
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

    int idx_m20 = (rx * 16 + ml) * 2 + 0 < cst.input_size * cst.batch ? (rx * 16 + ml) * 2 + 0 : (cst.input_size * cst.batch - 1);
    int idx_m21 = (rx * 16 + ml) * 2 + 1 < cst.input_size * cst.batch ? (rx * 16 + ml) * 2 + 1 : (cst.input_size * cst.batch - 1);

    int idx_n4   = (uz * 16 + tiitg / 8) < cst.output_slice ? (uz * 16 + tiitg / 8) : (cst.output_slice - 1);
    int idx_sa   = (ml * 2 + 0) * 8 + kl;
    int block    = (cst.input_slice + cst.block_size - 1) / cst.block_size;

    // quant-block partition: scale hoist semantics preserved (whole blocks per part)
    int nsub = (cst.block_size + FUSED_KSPLIT_KS - 1) / FUSED_KSPLIT_KS;
    int bi0  = ks * nsub;
    int bi1  = min(bi0 + nsub, cst.block_size);
    int zpart_end = min(bi1 * block, cst.input_slice);

    int idx_k4 = 0 * 8 + kl;
    auto xy_in0 = in + (bi0 * block + idx_k4) * cst.input_size * cst.batch + idx_m20;
    auto xy_in1 = in + (bi0 * block + idx_k4) * cst.input_size * cst.batch + idx_m21;

    // v2 register prefetch (first slice of this partition)
    uint2 w2pre;
    {
        const device uint2 *wq = (const device uint2 *)wt_int4;
        int no0 = (int)tiitg / 8;
        int kw0 = ((int)tiitg % 8) / 4;
        int i0  = (int)tiitg % 4;
        int n4_0 = (uz * 16 + no0) < cst.output_slice ? (uz * 16 + no0) : (cst.output_slice - 1);
        int z0 = min(bi0 * block + kw0 * 4 + i0, cst.input_slice - 1);
        w2pre = wq[n4_0 * cst.input_slice + z0];
    }
    for (int bi = bi0; bi < bi1; ++bi) {
        int zmin = bi * block;
        int zmax = min(zmin + block, cst.input_slice);

        // scale hoist (identical to single-pass kernel)
        ftype4 sc4t_h = *(const device ftype4 *)(dequantScale + ((idx_n4 * cst.block_size + bi) * 2 + 0) * 4);
        ftype4 db4t_h = *(const device ftype4 *)(dequantScale + ((idx_n4 * cst.block_size + bi) * 2 + 1) * 4);
        FLOAT4 sc4_h = FLOAT4(sc4t_h) / (FLOAT)cst.scale_coef;
        FLOAT4 db4_h = FLOAT4(db4t_h) / (FLOAT)cst.scale_coef;

        for (int z = zmin; z < zmax; z += 8) {
            FLOAT4 wb[4];
            {
                const device uint2 *wq = (const device uint2 *)wt_int4;
                int no = (int)tiitg / 8;
                int kw = ((int)tiitg % 8) / 4;
                int i  = (int)tiitg % 4;
                int idx_n4c = (uz * 16 + no) < cst.output_slice ? (uz * 16 + no) : (cst.output_slice - 1);
                uint2 w2 = w2pre;
                int znext = z + 8 + kw * 4 + i;
                w2pre = (z + 8 < zpart_end) ? wq[idx_n4c * cst.input_slice + min(znext, cst.input_slice - 1)] : w2pre;
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
            if (idx_k4 + z < cst.input_slice) {
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

            xy_in0 += 8 * cst.input_size * cst.batch;
            xy_in1 += 8 * cst.input_size * cst.batch;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto tC = tensor<threadgroup FLOAT, dextents<int32_t, 2>, tensor_inline>((threadgroup FLOAT*)sdata, dextents<int32_t, 2>(N, M));
    cT.store(tC);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // fp32 partials, no bias/activation (applied in reduce pass)
    int m = rx * 32 + mlc;
    if (m < cst.input_size * cst.batch) {
        for (int r = 0; r < 4; ++r) {
            int n4 = (uz * 4 + nlc) * 4 + r;
            if (n4 < cst.output_slice) {
                partialOut[((size_t)ks * cst.output_slice + n4) * cst.output_size * cst.batch + m] =
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

    int idx_n4 = (uz * 16 + tiitg / 8) < cst.output_slice ? (uz * 16 + tiitg / 8) : (cst.output_slice - 1);
    int block  = (cst.input_slice + cst.block_size - 1) / cst.block_size;

    int nsub = (cst.block_size + FUSED_KSPLIT_KS - 1) / FUSED_KSPLIT_KS;
    int bi0  = ks * nsub;
    int bi1  = min(bi0 + nsub, cst.block_size);
    int zpart_end = min(bi1 * block, cst.input_slice);

    int aml = (int)tiitg / 8;
    int akl = (int)tiitg % 8;
    bool aAct = aml < M;
    int am = rx * M + aml;
    int amc = am < cst.input_size * cst.batch ? am : (cst.input_size * cst.batch - 1);

    uint2 w2pre;
    {
        const device uint2 *wq = (const device uint2 *)wt_int4;
        int no0 = (int)tiitg / 8;
        int kw0 = ((int)tiitg % 8) / 4;
        int i0  = (int)tiitg % 4;
        int n4_0 = (uz * 16 + no0) < cst.output_slice ? (uz * 16 + no0) : (cst.output_slice - 1);
        int z0 = min(bi0 * block + kw0 * 4 + i0, cst.input_slice - 1);
        w2pre = wq[n4_0 * cst.input_slice + z0];
    }
    for (int bi = bi0; bi < bi1; ++bi) {
        int zmin = bi * block;
        int zmax = min(zmin + block, cst.input_slice);

        ftype4 sc4t_h = *(const device ftype4 *)(dequantScale + ((idx_n4 * cst.block_size + bi) * 2 + 0) * 4);
        ftype4 db4t_h = *(const device ftype4 *)(dequantScale + ((idx_n4 * cst.block_size + bi) * 2 + 1) * 4);
        FLOAT4 sc4_h = FLOAT4(sc4t_h) / (FLOAT)cst.scale_coef;
        FLOAT4 db4_h = FLOAT4(db4t_h) / (FLOAT)cst.scale_coef;

        for (int z = zmin; z < zmax; z += 8) {
            FLOAT4 wb[4];
            {
                const device uint2 *wq = (const device uint2 *)wt_int4;
                int no = (int)tiitg / 8;
                int kw = ((int)tiitg % 8) / 4;
                int i  = (int)tiitg % 4;
                int idx_n4c = (uz * 16 + no) < cst.output_slice ? (uz * 16 + no) : (cst.output_slice - 1);
                uint2 w2 = w2pre;
                int znext = z + 8 + kw * 4 + i;
                w2pre = (z + 8 < zpart_end) ? wq[idx_n4c * cst.input_slice + min(znext, cst.input_slice - 1)] : w2pre;
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
                if (akl + z < cst.input_slice) {
                    ((threadgroup ftype4*)sdata)[aml * 8 + akl] = (ftype4)in[(z + akl) * cst.input_size * cst.batch + amc];
                } else {
                    ((threadgroup ftype4*)sdata)[aml * 8 + akl] = (ftype4)(0);
                }
                #else
                ((threadgroup ftype4*)sdata)[aml * 8 + akl] = (ftype4)in[(z + akl) * cst.input_size * cst.batch + amc];
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
        if (gm >= cst.input_size * cst.batch) continue;
        int n4 = uz * 16 + n4l;
        if (n4 >= cst.output_slice) continue;
        FLOAT4 v = FLOAT4(((threadgroup FLOAT*)sdata)[m * 64 + n4l * 4 + 0],
                          ((threadgroup FLOAT*)sdata)[m * 64 + n4l * 4 + 1],
                          ((threadgroup FLOAT*)sdata)[m * 64 + n4l * 4 + 2],
                          ((threadgroup FLOAT*)sdata)[m * 64 + n4l * 4 + 3]);
        partialOut[((size_t)ks * cst.output_slice + n4) * cst.output_size * cst.batch + gm] = float4(v);
    }
#endif // USE_METAL_TENSOR_OPS
}

kernel void conv1x1_fused_q4_ksplit_reduce(
                            const device float4 *partialIn     [[buffer(0)]],
                            device ftype4 *out                 [[buffer(1)]],
                            const device ftype4 *biasTerms     [[buffer(2)]],
                            constant conv1x1_constants& cst    [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    int total = cst.output_slice * cst.output_size * cst.batch;
    if ((int)gid >= total) return;
    int n4 = (int)gid / (cst.output_size * cst.batch);
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
//
// Target: Qwen3-4B pp2048 gap vs llama.cpp (which uses M=64, N=128).
// This is the M=64, N=64 variant (N unchanged for lower risk); N=128 can
// be a follow-up if this proves the hypothesis.
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
                            uint3 gid                          [[threadgroup_position_in_grid]],
                            uint                  tiitg [[thread_index_in_threadgroup]],
                            uint                  tiisg [[thread_index_in_simdgroup]],
                            uint                  sgitg [[simdgroup_index_in_threadgroup]]) {
#ifdef USE_METAL_TENSOR_OPS
    // 1024 FLOAT4 = 16 KB. Sized for cT.store (M*N floats = 4096).
    threadgroup FLOAT4 sdata[1024] = {0.f};

    const int K = 32, M = 64, N = 64;
    auto tI = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata, dextents<int32_t, 2>(K, M));            // [M, K]
    auto tW = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata + 2048, dextents<int32_t, 2>(K, N));     // [N, K]

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(M, N, K, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mmOps;

    auto cT = mmOps.get_destination_cooperative_tensor<decltype(tI), decltype(tW), FLOAT>();

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

    // Input row mapping: 4 rows per thread (M64 / 16 ml groups = 4 rows/thread).
    int idx_m0 = (rx * 16 + ml) * 4 + 0 < cst.input_size * cst.batch ? (rx * 16 + ml) * 4 + 0 : (cst.input_size * cst.batch - 1);
    int idx_m1 = (rx * 16 + ml) * 4 + 1 < cst.input_size * cst.batch ? (rx * 16 + ml) * 4 + 1 : (cst.input_size * cst.batch - 1);
    int idx_m2 = (rx * 16 + ml) * 4 + 2 < cst.input_size * cst.batch ? (rx * 16 + ml) * 4 + 2 : (cst.input_size * cst.batch - 1);
    int idx_m3 = (rx * 16 + ml) * 4 + 3 < cst.input_size * cst.batch ? (rx * 16 + ml) * 4 + 3 : (cst.input_size * cst.batch - 1);

    int idx_k4 = 0 * 8 + kl;
    auto xy_in0 = in + idx_k4 * cst.input_size * cst.batch + idx_m0;
    auto xy_in1 = in + idx_k4 * cst.input_size * cst.batch + idx_m1;
    auto xy_in2 = in + idx_k4 * cst.input_size * cst.batch + idx_m2;
    auto xy_in3 = in + idx_k4 * cst.input_size * cst.batch + idx_m3;

    int idx_n4   = (uz * 16 + no) < cst.output_slice ? (uz * 16 + no) : (cst.output_slice - 1);
#ifdef W_QUANT_2
    auto xy_wt_i2 = ((const device uchar*)wt_int2) + (idx_n4 * cst.input_slice) * 4 + nl;
#elif defined(W_QUANT_3)
    auto xy_wt_i3 = wt_int3 + idx_n4 * cst.input_slice * 6;
#elif defined(W_QUANT_4)
    auto xy_wt_i4 = ((const device uchar2*)wt_int4) + (idx_n4 * cst.input_slice + 0) * 4 + nl;
#else // W_QUANT_8
    auto xy_wt_i8 = ((const device char4*)wt_int8)  + (idx_n4 * cst.input_slice + 0) * 4 + nl;
#endif

    // sA layout (ftype array): sdata[m * K + k] with M=64, K=32.
    // Per thread 4 rows m = 4*ml + r (r=0..3), K col group kl (K4 chunk at k = 4*kl).
    // In ftype4 units: row m at K4 kl → offset = m * (K/4) + kl = m * 8 + kl.
    //   r=0: idx = (4*ml + 0) * 8 + kl = 32*ml + kl
    //   r=1: idx = 32*ml + kl + 8
    //   r=2: idx = 32*ml + kl + 16
    //   r=3: idx = 32*ml + kl + 24
    int idx_sa = 32 * ml + kl;
    // sB base in ftype4: sA occupies M*K/4 = 64*32/4 = 512 ftype4.
    int idx_sb = 512 + ((no * 4 + nl) * 2 + kwl) * 4 + 0;
    int block  = (cst.input_slice + cst.block_size - 1) / cst.block_size;

    for (int bi = 0; bi < cst.block_size; ++bi) {
        FLOAT scale0        = FLOAT(dequantScale[((idx_n4 * cst.block_size + bi) * 2 + 0) * 4 + nl]) / (FLOAT)cst.scale_coef;
        FLOAT dequant_bias0 = FLOAT(dequantScale[((idx_n4 * cst.block_size + bi) * 2 + 1) * 4 + nl]) / (FLOAT)cst.scale_coef;
        int zmin = bi * block;
        int zmax = min(zmin + block, cst.input_slice);

        for (int z = zmin; z < zmax; z += 8) {
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

            threadgroup_barrier(mem_flags::mem_threadgroup);

            #pragma unroll(4)
            for (int i = 0; i < 4; ++i) {
                ((threadgroup ftype4*)sdata)[idx_sb + i] = ftype4(w_dequant[i]);
            }

            // Input load: 4 rows per thread.
            #ifdef MNN_METAL_SRC_PROTECT
            if (idx_k4 + z < cst.input_slice) {
                ((threadgroup ftype4*)sdata)[idx_sa]      = (ftype4)*(xy_in0);
                ((threadgroup ftype4*)sdata)[idx_sa + 8]  = (ftype4)*(xy_in1);
                ((threadgroup ftype4*)sdata)[idx_sa + 16] = (ftype4)*(xy_in2);
                ((threadgroup ftype4*)sdata)[idx_sa + 24] = (ftype4)*(xy_in3);
            } else {
                ((threadgroup ftype4*)sdata)[idx_sa]      = (ftype4)(0);
                ((threadgroup ftype4*)sdata)[idx_sa + 8]  = (ftype4)(0);
                ((threadgroup ftype4*)sdata)[idx_sa + 16] = (ftype4)(0);
                ((threadgroup ftype4*)sdata)[idx_sa + 24] = (ftype4)(0);
            }
            #else
            ((threadgroup ftype4*)sdata)[idx_sa]      = (ftype4)*(xy_in0);
            ((threadgroup ftype4*)sdata)[idx_sa + 8]  = (ftype4)*(xy_in1);
            ((threadgroup ftype4*)sdata)[idx_sa + 16] = (ftype4)*(xy_in2);
            ((threadgroup ftype4*)sdata)[idx_sa + 24] = (ftype4)*(xy_in3);
            #endif

            threadgroup_barrier(mem_flags::mem_threadgroup);

            auto sA = tI.slice(0, 0);
            auto sB = tW.slice(0, 0);
            mmOps.run(sA, sB, cT);

            xy_in0 += 8 * cst.input_size * cst.batch;
            xy_in1 += 8 * cst.input_size * cst.batch;
            xy_in2 += 8 * cst.input_size * cst.batch;
            xy_in3 += 8 * cst.input_size * cst.batch;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto tC = tensor<threadgroup FLOAT, dextents<int32_t, 2>, tensor_inline>((threadgroup FLOAT*)sdata, dextents<int32_t, 2>(N, M));
    cT.store(tC);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Epilogue: same per-thread pattern as M=32 kernel, but with mm=0..1 to cover M=64.
    // Each (mm, mlc, nlc) thread role writes 4 N-inner values at (M=rx*64+m_idx, N4-slot=uz*4+nlc).
    for (int mm = 0; mm < 2; ++mm) {
        int m_idx = mm * 32 + mlc;
        auto xy_out = out + ((uz * 4 + nlc) * 4 + 0) * cst.output_size * cst.batch + (rx * 64 + m_idx);
        if ((rx * 64 + m_idx) < cst.input_size * cst.batch) {
            if ((uz * 4 + nlc) * 4 < cst.output_slice) {
                xy_out[0] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(m_idx * 4 + nlc) * 4 + 0] + FLOAT4(biasTerms[(uz * 4 + nlc) * 4])), cst.activation);
            }
            if ((uz * 4 + nlc) * 4 + 1 < cst.output_slice) {
                xy_out[cst.output_size * cst.batch] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(m_idx * 4 + nlc) * 4 + 1] + FLOAT4(biasTerms[(uz * 4 + nlc) * 4 + 1])), cst.activation);
            }
            if ((uz * 4 + nlc) * 4 + 2 < cst.output_slice) {
                xy_out[cst.output_size * cst.batch * 2] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(m_idx * 4 + nlc) * 4 + 2] + FLOAT4(biasTerms[(uz * 4 + nlc) * 4 + 2])), cst.activation);
            }
            if ((uz * 4 + nlc) * 4 + 3 < cst.output_slice) {
                xy_out[cst.output_size * cst.batch * 3] = activate(ftype4(((threadgroup FLOAT4*)sdata)[(m_idx * 4 + nlc) * 4 + 3] + FLOAT4(biasTerms[(uz * 4 + nlc) * 4 + 3])), cst.activation);
            }
        }
    }
#endif // USE_METAL_TENSOR_OPS
}
#endif // W_QUANT_2 || W_QUANT_3 || W_QUANT_4 || W_QUANT_8

//======================================================================
// conv1x1_gemm_64x64_split_k_sg — sg_matrix M=64 tile (fp16 weights,
// outer-dequant path, non-tensor-API devices e.g. M4).
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
    int mBound = cst.input_size * cst.batch;
    int idx_m40 = (rx * 16 + ml) * 4 + 0 < mBound ? (rx * 16 + ml) * 4 + 0 : (mBound - 1);
    int idx_m41 = (rx * 16 + ml) * 4 + 1 < mBound ? (rx * 16 + ml) * 4 + 1 : (mBound - 1);
    int idx_m42 = (rx * 16 + ml) * 4 + 2 < mBound ? (rx * 16 + ml) * 4 + 2 : (mBound - 1);
    int idx_m43 = (rx * 16 + ml) * 4 + 3 < mBound ? (rx * 16 + ml) * 4 + 3 : (mBound - 1);

    int idx_k4 = ko * 2 + kl;
    auto xy_in0 = in + idx_k4 * mBound + idx_m40;// [K/4, M, K4]
    auto xy_in1 = in + idx_k4 * mBound + idx_m41;
    auto xy_in2 = in + idx_k4 * mBound + idx_m42;
    auto xy_in3 = in + idx_k4 * mBound + idx_m43;

    int idx_n4 = (uz * 16 + no) < cst.output_slice ? (uz * 16 + no) : (cst.output_slice - 1);
    auto xy_wt = wt + (idx_n4 * ((cst.input_slice + 3) / 4) + kwl) * 4 + nl;// [N/4, K/16, N4, K4, K4]

    int idx_sa = (ko * 64 + ml * 4 + 0) * 2 + kl;                 // ftype4 units, [K4, M64, K2] x K4
    int idx_sb = 2048 + (kwl * 16 + 0) * 64 + no * 4 + nl;        // ftype units, [K2, K16, N64]

    for (int z = 0; z < cst.input_slice; z += 8) {
        ftype4x4 w_local = xy_wt[z];

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll(16)
        for (int i = 0; i < 16; ++i) {
            ((threadgroup ftype*)sdata)[idx_sb + 64 * i] = w_local[i / 4][i % 4];
        }

        #ifdef MNN_METAL_SRC_PROTECT
        if (idx_k4 + z < cst.input_slice) {
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
                if ((uz * 16 + n4) < cst.output_slice) {
                    FLOAT4 val = *((threadgroup FLOAT4*)(csrc + 64 * ((n4o / 2) * 4 + mi) + mr * 8 + (n4o % 2) * 4));
                    out[(uz * 16 + n4) * cst.output_size * cst.batch + (rx * 64 + mlc)] =
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
    if ((int)gid.x >= cst.output_size || (int)gid.y >= cst.output_slice || (int)gid.z >= cst.batch) return;

    int rx = gid.x;
    int uz = gid.y;
    auto xy_wt = wt + uz * cst.input_slice;
    auto xy_in0  = in  + (int)gid.z  * cst.input_size + rx + 0;
    auto xy_out = out + (int)gid.z * cst.output_size + uz * cst.output_size * cst.batch + rx;
    auto biasValue = FLOAT4(biasTerms[uz]);
    FLOAT4 result0 = 0;

    for (int z = tiisg; z < cst.input_slice; z+=SIMD_GROUP_WIDTH) {
        auto xy_in = xy_in0 + z * cst.input_size * cst.batch;
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
    auto area_size = cst.output_size * cst.batch;
    if(uz >= cst.output_slice || rx >= area_size) {
        return;
    }
#ifdef W_QUANT_3
    auto xy_wt = wt + uz * cst.input_slice * 6;
#else
    auto xy_wt = wt + uz * cst.input_slice;
#endif
    auto xy_in0  = in + rx;
    auto xy_out = out + uz * area_size + rx;
    auto biasValue = FLOAT4(biasTerms[uz]);
    FLOAT4 result[AREA_THREAD] = {FLOAT4(0)};
    int block = (cst.input_slice + cst.block_size - 1) / cst.block_size;

    int middle_step = min(SIMD_GROUP_WIDTH, block);
    int outer_step  = SIMD_GROUP_WIDTH / middle_step;
    int middle_index = (tiisg) % middle_step;
    int outer_index  = (tiisg) / middle_step;


    for (int bi= outer_index; bi<cst.block_size; bi += outer_step) {
        FLOAT4 scale = FLOAT4(dequantScale[2 * (uz * cst.block_size + bi) + 0]) / (FLOAT)cst.scale_coef;
        FLOAT4 dequant_bias = FLOAT4(dequantScale[2 * (uz * cst.block_size + bi) + 1]) / (FLOAT)cst.scale_coef;
        int zmin = bi * block;
        int zmax = min(zmin + block, cst.input_slice);

    #if defined(W_QUANT_2) || defined(W_QUANT_3)
        // Deferred dequantization with power-of-two pre-scaling (see g8 kernel for
        // mask/pre-scale derivation). Each token accumulates its own raw_dot and
        // input_sum; the weight tile is loaded once and shared across all tokens.
        {
            FLOAT4 raw_dot[AREA_THREAD] = {FLOAT4(0)};
            FLOAT input_sum[AREA_THREAD] = {FLOAT(0)};
        #ifdef W_QUANT_2
            for (int z = zmin + middle_index; z < zmax; z += middle_step) {
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
            for (int z = zmin + middle_index; z < zmax; z += middle_step) {
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

            for (int z = zmin + middle_index; z < zmax; z += middle_step) {
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
            for (int z = zmin + middle_index; z < zmax; z += middle_step) {
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
        for (int z = zmin + middle_index; z < zmax; z += middle_step) {
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

// Multi-OC GEMV kernel: each threadgroup processes N_OC4 oc_slices sharing input load.
// This reduces global memory bandwidth by loading input only once for multiple OC groups.
// Designed for decode (area=1) where GEMV is memory-bandwidth bound.
template <int N_OC4>
kernel void conv1x1_gemv_multi_oc_wquant_sg(const device ftype4 *in       [[buffer(0)]],
                            device ftype4 *out                             [[buffer(1)]],
                            constant conv1x1_constants& cst                [[buffer(2)]],
                        #ifdef W_QUANT_4
                            const device MNN::uchar4x2 *wt                 [[buffer(3)]],
                        #elif defined(W_QUANT_8)
                            const device MNN::char4x4 *wt                  [[buffer(3)]],
                        #endif
                            const device ftype4 *biasTerms                 [[buffer(4)]],
                            const device ftype4 *dequantScale              [[buffer(5)]],
                            uint3 gid[[threadgroup_position_in_grid]],
                            uint  tiisg[[thread_index_in_simdgroup]]) {
    const int uz_base = gid.x * N_OC4;  // starting oc_4 index
    const int area_size = cst.output_size * cst.batch;
    if (uz_base >= cst.output_slice) return;

    // Clamp N_OC4 for boundary
    const int n_oc4 = min(N_OC4, cst.output_slice - uz_base);

    auto xy_in0 = in;  // area=1, rx=0
    FLOAT4 result[N_OC4] = {FLOAT4(0)};

    int block = (cst.input_slice + cst.block_size - 1) / cst.block_size;
    int middle_step = min(SIMD_GROUP_WIDTH, block);
    int outer_step  = SIMD_GROUP_WIDTH / middle_step;
    int middle_index = tiisg % middle_step;
    int outer_index  = tiisg / middle_step;

    for (int bi = outer_index; bi < cst.block_size; bi += outer_step) {
        // Load scale/bias for each oc_4
        FLOAT4 scale[N_OC4], adjusted_bias[N_OC4];
        for (int oc = 0; oc < N_OC4; oc++) {
            int uz = uz_base + oc;
            if (uz < cst.output_slice) {
                FLOAT4 s = FLOAT4(dequantScale[2 * (uz * cst.block_size + bi) + 0]) / (FLOAT)cst.scale_coef;
                FLOAT4 db = FLOAT4(dequantScale[2 * (uz * cst.block_size + bi) + 1]) / (FLOAT)cst.scale_coef;
                scale[oc] = s;
                adjusted_bias[oc] = db - FLOAT(8.0) * s;
            }
        }

        int zmin = bi * block;
        int zmax = min(zmin + block, cst.input_slice);

        #ifdef W_QUANT_4
        FLOAT4 raw_dot[N_OC4] = {FLOAT4(0)};
        FLOAT input_sum = FLOAT(0);
        constexpr FLOAT4 ones = FLOAT4(1.0);

        for (int z = zmin + middle_index; z < zmax; z += middle_step) {
            // Load input ONCE - shared across all OC groups
            FLOAT4 in4 = (FLOAT4)*(xy_in0 + z * area_size);
            input_sum += dot(in4, ones);

            // Process each oc_4 with shared input
            for (int oc = 0; oc < N_OC4; oc++) {
                MNN::uchar4x2 w = *(wt + (uz_base + oc) * cst.input_slice + z);
                FLOAT4 wv0 = FLOAT4(FLOAT(w[0][0] >> 4), FLOAT(w[0][0] & 15),
                                    FLOAT(w[0][1] >> 4), FLOAT(w[0][1] & 15));
                FLOAT4 wv1 = FLOAT4(FLOAT(w[1][0] >> 4), FLOAT(w[1][0] & 15),
                                    FLOAT(w[1][1] >> 4), FLOAT(w[1][1] & 15));
                FLOAT4 wv2 = FLOAT4(FLOAT(w[2][0] >> 4), FLOAT(w[2][0] & 15),
                                    FLOAT(w[2][1] >> 4), FLOAT(w[2][1] & 15));
                FLOAT4 wv3 = FLOAT4(FLOAT(w[3][0] >> 4), FLOAT(w[3][0] & 15),
                                    FLOAT(w[3][1] >> 4), FLOAT(w[3][1] & 15));
                raw_dot[oc][0] += dot(in4, wv0);
                raw_dot[oc][1] += dot(in4, wv1);
                raw_dot[oc][2] += dot(in4, wv2);
                raw_dot[oc][3] += dot(in4, wv3);
            }
        }
        // Apply scale and bias once per quant block
        for (int oc = 0; oc < N_OC4; oc++) {
            result[oc] += raw_dot[oc] * scale[oc] + input_sum * adjusted_bias[oc];
        }
        #elif defined(W_QUANT_8)
        for (int z = zmin + middle_index; z < zmax; z += middle_step) {
            FLOAT4 in4 = (FLOAT4)*(xy_in0 + z * area_size);
            for (int oc = 0; oc < N_OC4; oc++) {
                auto w = *(wt + (uz_base + oc) * cst.input_slice + z);
                FLOAT4x4 w_fp32 = FLOAT4x4(FLOAT4(w[0]), FLOAT4(w[1]), FLOAT4(w[2]), FLOAT4(w[3]));
                FLOAT4x4 w_dq;
                FLOAT4 s = scale[oc] + adjusted_bias[oc] + FLOAT(8.0) * scale[oc]; // recover original scale/bias
                FLOAT4 db = adjusted_bias[oc] + FLOAT(8.0) * scale[oc];
                for (int i = 0; i < 4; ++i) {
                    w_dq[i] = w_fp32[i] * scale[oc][i] + db[i];
                }
                result[oc] += FLOAT4(in4 * w_dq);
            }
        }
        #endif
    }

    // Simdgroup reduce
    for (int oc = 0; oc < N_OC4; oc++) {
        result[oc] = simd_sum(result[oc]);
    }

    // Store results
    if (tiisg == 0) {
        for (int oc = 0; oc < N_OC4; oc++) {
            int uz = uz_base + oc;
            if (uz < cst.output_slice) {
                out[uz * area_size] = activate(ftype4(result[oc] + FLOAT4(biasTerms[uz])), cst.activation);
            }
        }
    }
}

typedef decltype(conv1x1_gemv_multi_oc_wquant_sg<2>) multi_oc_kernel_t;
template [[host_name("conv1x1_gemv_multi_oc2_wquant_sg")]] kernel multi_oc_kernel_t conv1x1_gemv_multi_oc_wquant_sg<2>;
template [[host_name("conv1x1_gemv_multi_oc4_wquant_sg")]] kernel multi_oc_kernel_t conv1x1_gemv_multi_oc_wquant_sg<4>;

typedef decltype(conv1x1_gemv_g4mx_wquant_sg<1>) kernel_type_t;
template [[host_name("conv1x1_gemv_g4m1_wquant_sg")]] kernel kernel_type_t conv1x1_gemv_g4mx_wquant_sg<1>;
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

// Fused weight+scale decode GEMV kernel: scale/bias is stored inline before each weight block
// in a single contiguous buffer, eliminating separate scale buffer access.
// Fused buffer layout per OC_slice: [block0: scale(ftype4) | bias(ftype4) | weights(uchar4x2 * block_elems)] [block1: ...] ...
kernel void conv1x1_gemv_fused_wquant_sg(const device ftype4 *in     [[buffer(0)]],
                            device ftype4 *out                        [[buffer(1)]],
                            constant conv1x1_constants& cst           [[buffer(2)]],
                            const device uchar *fused_wt              [[buffer(3)]],
                            const device ftype4 *biasTerms            [[buffer(4)]],
                            uint3 gid[[threadgroup_position_in_grid]],
                            uint  tiisg[[thread_index_in_simdgroup]]) {
    const int uz = gid.x;
    if (uz >= cst.output_slice) return;

    const int area_size = cst.output_size * cst.batch;
    const int block_elems = (cst.input_slice + cst.block_size - 1) / cst.block_size;
    constexpr int ftype4_bytes = sizeof(ftype4);
    const int scale_header = 2 * ftype4_bytes;
    const int block_stride = scale_header + block_elems * 8;
    const FLOAT inv_scale_coef = FLOAT(1.0) / (FLOAT)cst.scale_coef;

    const device uchar *oc_base = fused_wt + uz * cst.block_size * block_stride;

    FLOAT4 result = FLOAT4(0);

    const int middle_step = min(SIMD_GROUP_WIDTH, block_elems);
    const int outer_step  = SIMD_GROUP_WIDTH / middle_step;
    const int middle_index = tiisg % middle_step;
    const int outer_index  = tiisg / middle_step;

    for (int bi = outer_index; bi < cst.block_size; bi += outer_step) {
        const device uchar *bb = oc_base + bi * block_stride;
        FLOAT4 scale = FLOAT4(*(const device ftype4 *)bb) * inv_scale_coef;
        FLOAT4 dbias = FLOAT4(*(const device ftype4 *)(bb + ftype4_bytes)) * inv_scale_coef;
        const device MNN::uchar4x2 *wt_block = (const device MNN::uchar4x2 *)(bb + scale_header);

        const int zmin = bi * block_elems;
        const int zmax = min(zmin + block_elems, cst.input_slice);

        FLOAT4 raw_dot = FLOAT4(0);
        FLOAT in_sum = FLOAT(0);

        for (int z = zmin + middle_index; z < zmax; z += middle_step) {
            FLOAT4 in4 = (FLOAT4)*(in + z * area_size);
            MNN::uchar4x2 w = wt_block[z - zmin];

            in_sum += in4[0] + in4[1] + in4[2] + in4[3];

            raw_dot[0] += in4[0] * FLOAT(w[0][0] >> 4) + in4[1] * FLOAT(w[0][0] & 15)
                        + in4[2] * FLOAT(w[0][1] >> 4) + in4[3] * FLOAT(w[0][1] & 15);
            raw_dot[1] += in4[0] * FLOAT(w[1][0] >> 4) + in4[1] * FLOAT(w[1][0] & 15)
                        + in4[2] * FLOAT(w[1][1] >> 4) + in4[3] * FLOAT(w[1][1] & 15);
            raw_dot[2] += in4[0] * FLOAT(w[2][0] >> 4) + in4[1] * FLOAT(w[2][0] & 15)
                        + in4[2] * FLOAT(w[2][1] >> 4) + in4[3] * FLOAT(w[2][1] & 15);
            raw_dot[3] += in4[0] * FLOAT(w[3][0] >> 4) + in4[1] * FLOAT(w[3][0] & 15)
                        + in4[2] * FLOAT(w[3][1] >> 4) + in4[3] * FLOAT(w[3][1] & 15);
        }

        FLOAT4 adjusted_bias = dbias - FLOAT(8.0) * scale;
        result += raw_dot * scale + in_sum * adjusted_bias;
    }

    result = simd_sum(result);

    if (tiisg == 0) {
        out[uz * area_size] = activate(ftype4(result + FLOAT4(biasTerms[uz])), cst.activation);
    }
}

// 2-simdgroup GEMV kernel: each threadgroup has 2 simdgroups, each independently
// processes one output_slice (4 OC). Input is shared via L1 cache (no barrier needed).
// Halves the number of dispatched threadgroups vs g4m1 for better GPU occupancy.
// Uses deferred dequantization with ushort pre-scaling trick (inspired by llama.cpp).
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
                            // qkv_seg: [0]=k scale_coef, [1]=v scale_coef,
                            //          [2]=k output_slice, [3]=v output_slice
                            //          (QKV_FUSED_P4: [4]=w scale_coef, [5]=w output_slice)
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
    // Compact mode concatenates each projection's exact threadgroup range on
    // grid.x. The legacy rollback uses gid.z plus a rectangular max-size grid.
    uint qkv_projection = gid.z;
    uint qkv_grid_x = gid.x;
#ifdef QKV_COMPACT_GRID
    // Keep this aligned with setupQKVFusion's host-side two-slice TG count.
    const uint qkv_q_groups = (uint(cst.output_slice) + 1) / 2;
    const uint qkv_k_groups = (uint(qkv_seg[2]) + 1) / 2;
    const uint qkv_v_groups = (uint(qkv_seg[3]) + 1) / 2;
    if (qkv_grid_x >= qkv_q_groups) {
        qkv_grid_x -= qkv_q_groups;
        qkv_projection = 1;
        if (qkv_grid_x >= qkv_k_groups) {
            qkv_grid_x -= qkv_k_groups;
            qkv_projection = 2;
#ifdef QKV_FUSED_P4
            if (qkv_grid_x >= qkv_v_groups) {
                qkv_grid_x -= qkv_v_groups;
                qkv_projection = 3;
            }
#endif
        }
    }
#endif
    int qkv_output_slice = cst.output_slice;
    float qkv_scale_coef = cst.scale_coef;
    if (qkv_projection == 1) {
        out = out_k;
        wt = wt_k;
        biasTerms = biasTerms_k;
        dequantScale = dequantScale_k;
        qkv_scale_coef = qkv_seg[0];
        qkv_output_slice = int(qkv_seg[2]);
    } else if (qkv_projection == 2) {
        out = out_v;
        wt = wt_v;
        biasTerms = biasTerms_v;
        dequantScale = dequantScale_v;
        qkv_scale_coef = qkv_seg[1];
        qkv_output_slice = int(qkv_seg[3]);
    }
#ifdef QKV_FUSED_P4
    // 4-projection groups (Qwen3.5 linear-attention layers: qkv/z/b/a share
    // one LN input): projection 3 selects the 4th projection.
    else if (qkv_projection == 3) {
        out = out_w;
        wt = wt_w;
        biasTerms = biasTerms_w;
        dequantScale = dequantScale_w;
        qkv_scale_coef = qkv_seg[4];
        qkv_output_slice = int(qkv_seg[5]);
    }
#endif
#endif

#ifdef ROW_2
    // Dual-row variant (fused pipelines only): each simdgroup handles TWO
    // adjacent output slices with independent accumulator streams — doubles
    // in-flight weight reads with no barrier (cf. SPLIT_K_2's tg reduce) and
    // shares the input read + LN prologue across both rows.
#ifdef QKV_FUSED
    const int uz = (qkv_grid_x * 2 + (int)sgitg) * 2;
    if (uz >= qkv_output_slice) return;
    const bool row1_valid = (uz + 1) < qkv_output_slice;
    float cur_scale_coef = qkv_scale_coef;
#else
    const int uz = (gid.x * 2 + (int)sgitg) * 2;
    if (uz >= cst.output_slice) return;
    const bool row1_valid = (uz + 1) < cst.output_slice;
    float cur_scale_coef = cst.scale_coef;
#endif
    #ifdef GATE_UP_FUSED
    if (gid.z == 1) {
        cur_scale_coef = gate_up_seg[0];
    }
    #endif

    const int area_size = cst.output_size * cst.batch;
    // Invalid second row aliases row0 (safe reads); its result is discarded.
    const int uz1 = row1_valid ? (uz + 1) : uz;
#ifdef W_QUANT_3
    // 6-byte (4 OC x 4 IC) tiles: row stride is 6x the ic_4 count.
    const int wt_row_stride = cst.input_slice * 6;
#else
    const int wt_row_stride = cst.input_slice;
#endif
    auto xy_wt0 = wt + uz * wt_row_stride;
    auto xy_wt1 = wt + uz1 * wt_row_stride;
    auto xy_in0 = in;
    auto biasValue0 = FLOAT4(biasTerms[uz]);
    auto biasValue1 = FLOAT4(biasTerms[uz1]);

#ifdef LN_FUSED
    float sq_sum = 0.0f;
    bool ln_write_residual = (sgitg == 0
    #if defined(QKV_FUSED)
        && qkv_projection == 0 && qkv_grid_x == 0
    #elif defined(GATE_UP_FUSED)
        && gid.z == 0 && gid.x == 0
    #else
        && gid.x == 0
    #endif
    );
    for (int z = tiisg; z < cst.input_slice; z += 32) {
        float4 d = (float4)*(xy_in0 + z * area_size) + (float4)*(ln_residual_in + z * area_size);
        sq_sum += dot(d, d);
        if (ln_write_residual) {
            ln_residual_out[z * area_size] = (ftype4)d;
        }
    }
    sq_sum = simd_sum(sq_sum);
    float inv_rms = rsqrt(sq_sum / (float)(cst.input_slice * 4) + *ln_eps);
#endif

    int block = (cst.input_slice + cst.block_size - 1) / cst.block_size;
#ifdef WIDE_MIDDLE
    int middle_step = min(SIMD_GROUP_WIDTH, max(block, 1));
#else
    int middle_step = min(SIMD_GROUP_WIDTH, max(block / 4, 1));
#endif
    int outer_step  = SIMD_GROUP_WIDTH / middle_step;
    int middle_index = tiisg % middle_step;
    int outer_index  = tiisg / middle_step;

    FLOAT4 result0 = FLOAT4(0);
    FLOAT4 result1 = FLOAT4(0);

    for (int bi = outer_index; bi < cst.block_size; bi += outer_step) {
        FLOAT4 scale0 = FLOAT4(dequantScale[2 * (uz * cst.block_size + bi) + 0]) / (FLOAT)cur_scale_coef;
        FLOAT4 dbias0 = FLOAT4(dequantScale[2 * (uz * cst.block_size + bi) + 1]) / (FLOAT)cur_scale_coef;
        FLOAT4 scale1 = FLOAT4(dequantScale[2 * (uz1 * cst.block_size + bi) + 0]) / (FLOAT)cur_scale_coef;
        FLOAT4 dbias1 = FLOAT4(dequantScale[2 * (uz1 * cst.block_size + bi) + 1]) / (FLOAT)cur_scale_coef;
        int zmin = bi * block;
        int zmax = min(zmin + block, cst.input_slice);

    #if defined(W_QUANT_2) || defined(W_QUANT_3)
        // Deferred + pre-scaling dual-row streams (see g8 kernel for the
        // mask/pre-scale derivation). Trap-A: must precede the W4 branch.
        FLOAT4 raw_dot0 = FLOAT4(0);
        FLOAT4 raw_dot1 = FLOAT4(0);
        FLOAT input_sum = FLOAT(0);
        for (int z = zmin + middle_index; z < zmax; z += middle_step) {
        #ifdef LN_FUSED
            FLOAT4 raw = (FLOAT4)*(xy_in0 + z * area_size) + (FLOAT4)*(ln_residual_in + z * area_size);
            FLOAT4 in4 = raw * inv_rms * (FLOAT4)ln_gamma[z];
        #else
            FLOAT4 in4 = (FLOAT4)*(xy_in0 + z * area_size);
        #endif
            input_sum += in4[0] + in4[1] + in4[2] + in4[3];

            FLOAT in_ps0 = in4[0] * FLOAT(1.0/64.0);
            FLOAT in_ps1 = in4[1] * FLOAT(1.0/16.0);
            FLOAT in_ps2 = in4[2] * FLOAT(1.0/4.0);
        #ifdef W_QUANT_2
            uchar4 wA = xy_wt0[z];
            uchar4 wB = xy_wt1[z];
            raw_dot0[0] += in_ps0 * FLOAT(wA[0] & 0xC0) + in_ps1 * FLOAT(wA[0] & 0x30)
                         + in_ps2 * FLOAT(wA[0] & 0x0C) + in4[3] * FLOAT(wA[0] & 0x03);
            raw_dot0[1] += in_ps0 * FLOAT(wA[1] & 0xC0) + in_ps1 * FLOAT(wA[1] & 0x30)
                         + in_ps2 * FLOAT(wA[1] & 0x0C) + in4[3] * FLOAT(wA[1] & 0x03);
            raw_dot0[2] += in_ps0 * FLOAT(wA[2] & 0xC0) + in_ps1 * FLOAT(wA[2] & 0x30)
                         + in_ps2 * FLOAT(wA[2] & 0x0C) + in4[3] * FLOAT(wA[2] & 0x03);
            raw_dot0[3] += in_ps0 * FLOAT(wA[3] & 0xC0) + in_ps1 * FLOAT(wA[3] & 0x30)
                         + in_ps2 * FLOAT(wA[3] & 0x0C) + in4[3] * FLOAT(wA[3] & 0x03);
            raw_dot1[0] += in_ps0 * FLOAT(wB[0] & 0xC0) + in_ps1 * FLOAT(wB[0] & 0x30)
                         + in_ps2 * FLOAT(wB[0] & 0x0C) + in4[3] * FLOAT(wB[0] & 0x03);
            raw_dot1[1] += in_ps0 * FLOAT(wB[1] & 0xC0) + in_ps1 * FLOAT(wB[1] & 0x30)
                         + in_ps2 * FLOAT(wB[1] & 0x0C) + in4[3] * FLOAT(wB[1] & 0x03);
            raw_dot1[2] += in_ps0 * FLOAT(wB[2] & 0xC0) + in_ps1 * FLOAT(wB[2] & 0x30)
                         + in_ps2 * FLOAT(wB[2] & 0x0C) + in4[3] * FLOAT(wB[2] & 0x03);
            raw_dot1[3] += in_ps0 * FLOAT(wB[3] & 0xC0) + in_ps1 * FLOAT(wB[3] & 0x30)
                         + in_ps2 * FLOAT(wB[3] & 0x0C) + in4[3] * FLOAT(wB[3] & 0x03);
        #else
            FLOAT in_hi0 = in4[0] * FLOAT(0.5);
            FLOAT in_hi2 = in4[2] * FLOAT(2.0);
            FLOAT in_hi3 = in4[3] * FLOAT(4.0);
            const device ushort* tA = (const device ushort*)(xy_wt0 + z * 6);
            const device ushort* tB = (const device ushort*)(xy_wt1 + z * 6);
            ushort sa0 = tA[0], sa1 = tA[1], sa2 = tA[2];
            ushort sb0 = tB[0], sb1 = tB[1], sb2 = tB[2];
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
        FLOAT4 adj0 = dbias0 - FLOAT(2.0) * scale0;
        FLOAT4 adj1 = dbias1 - FLOAT(2.0) * scale1;
        #else
        FLOAT4 adj0 = dbias0 - FLOAT(4.0) * scale0;
        FLOAT4 adj1 = dbias1 - FLOAT(4.0) * scale1;
        #endif
        result0 += raw_dot0 * scale0 + input_sum * adj0;
        result1 += raw_dot1 * scale1 + input_sum * adj1;
    #elif defined(W_QUANT_4)
        FLOAT4 raw_dot0 = FLOAT4(0);
        FLOAT4 raw_dot1 = FLOAT4(0);
        FLOAT input_sum = FLOAT(0);
        for (int z = zmin + middle_index; z < zmax; z += middle_step) {
        #ifdef LN_FUSED
            FLOAT4 raw = (FLOAT4)*(xy_in0 + z * area_size) + (FLOAT4)*(ln_residual_in + z * area_size);
            FLOAT4 in4 = raw * inv_rms * (FLOAT4)ln_gamma[z];
        #else
            FLOAT4 in4 = (FLOAT4)*(xy_in0 + z * area_size);
        #endif
            input_sum += in4[0] + in4[1] + in4[2] + in4[3];

            FLOAT in_ps0 = in4[0] * FLOAT(1.0/16.0);
            FLOAT in_ps2 = in4[2] * FLOAT(1.0/4096.0);
            FLOAT in_ps3 = in4[3] * FLOAT(1.0/256.0);

            ushort4 wA = xy_wt0[z];
            ushort4 wB = xy_wt1[z];
            raw_dot0[0] += in_ps0 * FLOAT(wA[0] & 0x00F0) + in4[1] * FLOAT(wA[0] & 0x000F)
                         + in_ps2 * FLOAT(wA[0] & 0xF000) + in_ps3 * FLOAT(wA[0] & 0x0F00);
            raw_dot0[1] += in_ps0 * FLOAT(wA[1] & 0x00F0) + in4[1] * FLOAT(wA[1] & 0x000F)
                         + in_ps2 * FLOAT(wA[1] & 0xF000) + in_ps3 * FLOAT(wA[1] & 0x0F00);
            raw_dot0[2] += in_ps0 * FLOAT(wA[2] & 0x00F0) + in4[1] * FLOAT(wA[2] & 0x000F)
                         + in_ps2 * FLOAT(wA[2] & 0xF000) + in_ps3 * FLOAT(wA[2] & 0x0F00);
            raw_dot0[3] += in_ps0 * FLOAT(wA[3] & 0x00F0) + in4[1] * FLOAT(wA[3] & 0x000F)
                         + in_ps2 * FLOAT(wA[3] & 0xF000) + in_ps3 * FLOAT(wA[3] & 0x0F00);
            raw_dot1[0] += in_ps0 * FLOAT(wB[0] & 0x00F0) + in4[1] * FLOAT(wB[0] & 0x000F)
                         + in_ps2 * FLOAT(wB[0] & 0xF000) + in_ps3 * FLOAT(wB[0] & 0x0F00);
            raw_dot1[1] += in_ps0 * FLOAT(wB[1] & 0x00F0) + in4[1] * FLOAT(wB[1] & 0x000F)
                         + in_ps2 * FLOAT(wB[1] & 0xF000) + in_ps3 * FLOAT(wB[1] & 0x0F00);
            raw_dot1[2] += in_ps0 * FLOAT(wB[2] & 0x00F0) + in4[1] * FLOAT(wB[2] & 0x000F)
                         + in_ps2 * FLOAT(wB[2] & 0xF000) + in_ps3 * FLOAT(wB[2] & 0x0F00);
            raw_dot1[3] += in_ps0 * FLOAT(wB[3] & 0x00F0) + in4[1] * FLOAT(wB[3] & 0x000F)
                         + in_ps2 * FLOAT(wB[3] & 0xF000) + in_ps3 * FLOAT(wB[3] & 0x0F00);
        }
        FLOAT4 adj0 = dbias0 - FLOAT(8.0) * scale0;
        FLOAT4 adj1 = dbias1 - FLOAT(8.0) * scale1;
        result0 += raw_dot0 * scale0 + input_sum * adj0;
        result1 += raw_dot1 * scale1 + input_sum * adj1;
    #elif defined(W_QUANT_8)
        for (int z = zmin + middle_index; z < zmax; z += middle_step) {
            auto wA = xy_wt0[z];
            auto wB = xy_wt1[z];
            FLOAT4x4 wA_dq;
            FLOAT4x4 wB_dq;
            for (int i = 0; i < 4; ++i) {
                wA_dq[i] = FLOAT4(wA[i]) * scale0[i] + dbias0[i];
                wB_dq[i] = FLOAT4(wB[i]) * scale1[i] + dbias1[i];
            }
        #ifdef LN_FUSED
            FLOAT4 raw = (FLOAT4)*(xy_in0 + z * area_size) + (FLOAT4)*(ln_residual_in + z * area_size);
            FLOAT4 in4 = raw * inv_rms * (FLOAT4)ln_gamma[z];
        #else
            FLOAT4 in4 = (FLOAT4)*(xy_in0 + z * area_size);
        #endif
            result0 += FLOAT4(in4 * wA_dq);
            result1 += FLOAT4(in4 * wB_dq);
        }
    #endif
    }

    result0 = simd_sum(result0);
    result1 = simd_sum(result1);
    if (tiisg == 0) {
        out[uz * area_size] = activate(ftype4(result0 + biasValue0), cst.activation);
        if (row1_valid) {
            out[uz1 * area_size] = activate(ftype4(result1 + biasValue1), cst.activation);
        }
    }
#else  // !ROW_2

    // 2 simdgroups per threadgroup, each handles one output_slice independently
#ifdef SPLIT_K_2
    // 4 simdgroups: sg 0/1 = rows (lower K half), sg 2/3 = same rows (upper K
    // half); partials combined via threadgroup memory. Host guarantees the
    // grid is exact (oc % 8 == 0) so no simdgroup early-returns before the
    // barrier below.
    #ifdef QKV_FUSED
    const int uz = qkv_grid_x * 2 + ((int)sgitg & 1);
    #else
    const int uz = gid.x * 2 + ((int)sgitg & 1);
    #endif
    const int sk_half = (int)sgitg >> 1;
#else
    #ifdef QKV_FUSED
    const int uz = qkv_grid_x * 2 + sgitg;
    #else
    const int uz = gid.x * 2 + sgitg;
    #endif
#endif
#ifdef QKV_FUSED
    if (uz >= qkv_output_slice) return;
    float cur_scale_coef = qkv_scale_coef;
#else
    if (uz >= cst.output_slice) return;
    float cur_scale_coef = cst.scale_coef;
#endif
    #ifdef GATE_UP_FUSED
    // Gate uses cst.scale_coef (leader's), up uses its own scale_coef via gate_up_seg[0].
    // Without this, up's dequant is scaled by gate's coefficient -> systematic bias
    // whenever gate/up weights have different fp16-fit ranges (visible on Qwen3.5-2B
    // as decode drift into repetition / low-quality output).
    if (gid.z == 1) {
        cur_scale_coef = gate_up_seg[0];
    }
    #endif

    const int area_size = cst.output_size * cst.batch;
#ifdef W_QUANT_3
    auto xy_wt = wt + uz * cst.input_slice * 6;
#else
    auto xy_wt = wt + uz * cst.input_slice;
#endif
    auto xy_in0 = in;
    auto biasValue = FLOAT4(biasTerms[uz]);

#ifdef LN_FUSED
    float sq_sum = 0.0f;
    // Only one threadgroup should write ln_residual_out to avoid races
    // when multiple threadgroups process the same input slices.
    bool ln_write_residual = (sgitg == 0
    #if defined(QKV_FUSED)
        && qkv_projection == 0 && qkv_grid_x == 0
    #elif defined(GATE_UP_FUSED)
        && gid.z == 0 && gid.x == 0
    #else
        && gid.x == 0
    #endif
    );
    for (int z = tiisg; z < cst.input_slice; z += 32) {
        float4 d = (float4)*(xy_in0 + z * area_size) + (float4)*(ln_residual_in + z * area_size);
        sq_sum += dot(d, d);
        if (ln_write_residual) {
            ln_residual_out[z * area_size] = (ftype4)d;
        }
    }
    sq_sum = simd_sum(sq_sum);
    float inv_rms = rsqrt(sq_sum / (float)(cst.input_slice * 4) + *ln_eps);
#endif

#ifdef GEMV_QBLOCK_W16
    // Q4 block 32/64/128/256 specialization for decode GEMV. The host supplies
    // the exact C4 slices per quant block (8/16/32/64), so the inner extent is
    // compile-time constant and no tail check is needed.
    //
    // GEMV_QBLOCK_W16_MID = how many lanes share one quant block. The host
    // picks it per pipeline from the blocks a simdgroup owns (block_size, or
    // block_size/2 under split-K) so that all 32 lanes stay fed: too few lanes
    // per block idles lanes on short rows, too many shortens each lane's
    // contiguous run. See chooseQ4W16Mid in MetalConvolution1x1.mm.
#ifndef GEMV_QBLOCK_W16_BLOCK_SLICES
#define GEMV_QBLOCK_W16_BLOCK_SLICES 16
#endif
#ifndef GEMV_QBLOCK_W16_MID
#define GEMV_QBLOCK_W16_MID 1
#endif
    constexpr int block = GEMV_QBLOCK_W16_BLOCK_SLICES;
    constexpr int middle_step = GEMV_QBLOCK_W16_MID;
    constexpr int outer_step = SIMD_GROUP_WIDTH / middle_step;
    const int middle_index = int(tiisg) & (middle_step - 1);
    const int outer_index = int(tiisg) / middle_step;
#else
    int block = (cst.input_slice + cst.block_size - 1) / cst.block_size;
    // GEMV inner reduction lane partitioning.
    //
    // Default (M4/Apple GPU 7-8): min(32, max(block/4, 1))
    //   For Qwen3-0.6B block_size=32, input_slice/32≈8 => middle_step=2, outer_step=16.
    //   Only 2 lanes participate in inner K reduction; other 30 lanes iterate outer
    //   blocks. Fine on narrower SM (M4) where BW is the bottleneck.
    //
    // WIDE_MIDDLE (opt-in via host macro): min(32, block)
    //   Puts more lanes on inner reduction. Beneficial on wider-SM chips (M5+)
    //   where the default under-utilizes SIMD width and BW isn't saturated.
    //
    // Host controls via MTLCompileOptions -> MNN_METAL_GEMV_WIDE_MIDDLE=1.
#ifdef WIDE_MIDDLE
    int middle_step = min(SIMD_GROUP_WIDTH, max(block, 1));
#else
    int middle_step = min(SIMD_GROUP_WIDTH, max(block / 4, 1));
#endif
    int outer_step  = SIMD_GROUP_WIDTH / middle_step;
    int middle_index = tiisg % middle_step;
    int outer_index  = tiisg / middle_step;
#endif

    FLOAT4 result = FLOAT4(0);

#ifdef SPLIT_K_2
    const int sk_bi_begin = sk_half * (cst.block_size / 2);
    const int sk_bi_end   = (sk_half == 1) ? cst.block_size : (cst.block_size / 2);
    for (int bi = sk_bi_begin + outer_index; bi < sk_bi_end; bi += outer_step) {
#else
    for (int bi = outer_index; bi < cst.block_size; bi += outer_step) {
#endif
        FLOAT4 scale = FLOAT4(dequantScale[2 * (uz * cst.block_size + bi) + 0]) / (FLOAT)cur_scale_coef;
        FLOAT4 dequant_bias = FLOAT4(dequantScale[2 * (uz * cst.block_size + bi) + 1]) / (FLOAT)cur_scale_coef;
        int zmin = bi * block;
        #ifdef GEMV_QBLOCK_W16
        int zmax = zmin + block;
        #else
        int zmax = min(zmin + block, cst.input_slice);
        #endif

    #if defined(W_QUANT_2) || defined(W_QUANT_3)
        // Deferred + pre-scaling (see g8 kernel for the mask/pre-scale
        // derivation). Trap-A: must precede the W4 branch.
        FLOAT4 raw_dot = FLOAT4(0);
        FLOAT input_sum = FLOAT(0);
        for (int z = zmin + middle_index; z < zmax; z += middle_step) {
        #ifdef LN_FUSED
            FLOAT4 raw = (FLOAT4)*(xy_in0 + z * area_size) + (FLOAT4)*(ln_residual_in + z * area_size);
            FLOAT4 in4 = raw * inv_rms * (FLOAT4)ln_gamma[z];
        #else
            FLOAT4 in4 = (FLOAT4)*(xy_in0 + z * area_size);
        #endif
            input_sum += in4[0] + in4[1] + in4[2] + in4[3];

            FLOAT in_ps0 = in4[0] * FLOAT(1.0/64.0);
            FLOAT in_ps1 = in4[1] * FLOAT(1.0/16.0);
            FLOAT in_ps2 = in4[2] * FLOAT(1.0/4.0);
        #ifdef W_QUANT_2
            uchar4 w_b = xy_wt[z];
            raw_dot[0] += in_ps0 * FLOAT(w_b[0] & 0xC0) + in_ps1 * FLOAT(w_b[0] & 0x30)
                        + in_ps2 * FLOAT(w_b[0] & 0x0C) + in4[3] * FLOAT(w_b[0] & 0x03);
            raw_dot[1] += in_ps0 * FLOAT(w_b[1] & 0xC0) + in_ps1 * FLOAT(w_b[1] & 0x30)
                        + in_ps2 * FLOAT(w_b[1] & 0x0C) + in4[3] * FLOAT(w_b[1] & 0x03);
            raw_dot[2] += in_ps0 * FLOAT(w_b[2] & 0xC0) + in_ps1 * FLOAT(w_b[2] & 0x30)
                        + in_ps2 * FLOAT(w_b[2] & 0x0C) + in4[3] * FLOAT(w_b[2] & 0x03);
            raw_dot[3] += in_ps0 * FLOAT(w_b[3] & 0xC0) + in_ps1 * FLOAT(w_b[3] & 0x30)
                        + in_ps2 * FLOAT(w_b[3] & 0x0C) + in4[3] * FLOAT(w_b[3] & 0x03);
        #else
            FLOAT in_hi0 = in4[0] * FLOAT(0.5);
            FLOAT in_hi2 = in4[2] * FLOAT(2.0);
            FLOAT in_hi3 = in4[3] * FLOAT(4.0);
            const device ushort* t = (const device ushort*)(xy_wt + z * 6);
            ushort tw0 = t[0], tw1 = t[1], tw2 = t[2];
            uchar lo0 = uchar(tw0), lo1 = uchar(tw0 >> 8), lo2 = uchar(tw1), lo3 = uchar(tw1 >> 8);
            uchar hb0 = uchar(tw2), hb1 = uchar(tw2 >> 8);
            uchar h0 = hb0 >> 4, h1 = hb0 & 0xF, h2 = hb1 >> 4, h3 = hb1 & 0xF;
            raw_dot[0] += in_ps0 * FLOAT(lo0 & 0xC0) + in_ps1 * FLOAT(lo0 & 0x30)
                        + in_ps2 * FLOAT(lo0 & 0x0C) + in4[3] * FLOAT(lo0 & 0x03)
                        + in_hi0 * FLOAT(h0 & 0x8) + in4[1] * FLOAT(h0 & 0x4)
                        + in_hi2 * FLOAT(h0 & 0x2) + in_hi3 * FLOAT(h0 & 0x1);
            raw_dot[1] += in_ps0 * FLOAT(lo1 & 0xC0) + in_ps1 * FLOAT(lo1 & 0x30)
                        + in_ps2 * FLOAT(lo1 & 0x0C) + in4[3] * FLOAT(lo1 & 0x03)
                        + in_hi0 * FLOAT(h1 & 0x8) + in4[1] * FLOAT(h1 & 0x4)
                        + in_hi2 * FLOAT(h1 & 0x2) + in_hi3 * FLOAT(h1 & 0x1);
            raw_dot[2] += in_ps0 * FLOAT(lo2 & 0xC0) + in_ps1 * FLOAT(lo2 & 0x30)
                        + in_ps2 * FLOAT(lo2 & 0x0C) + in4[3] * FLOAT(lo2 & 0x03)
                        + in_hi0 * FLOAT(h2 & 0x8) + in4[1] * FLOAT(h2 & 0x4)
                        + in_hi2 * FLOAT(h2 & 0x2) + in_hi3 * FLOAT(h2 & 0x1);
            raw_dot[3] += in_ps0 * FLOAT(lo3 & 0xC0) + in_ps1 * FLOAT(lo3 & 0x30)
                        + in_ps2 * FLOAT(lo3 & 0x0C) + in4[3] * FLOAT(lo3 & 0x03)
                        + in_hi0 * FLOAT(h3 & 0x8) + in4[1] * FLOAT(h3 & 0x4)
                        + in_hi2 * FLOAT(h3 & 0x2) + in_hi3 * FLOAT(h3 & 0x1);
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

        // NOTE: Adding `#pragma clang loop unroll_count(N)` here was evaluated
        // (2x and 4x) and found to be a net negative on M4 Pro across
        // Qwen3-0.6B/4B/8B: decode regresses 0.4-4% while prefill nets only
        // +1% at best (see prior benchmark notes). The Metal compiler already
        // schedules this compact 4-row loop body optimally; forcing unroll
        // increases register pressure and hurts simdgroup occupancy in the
        // decode (GEMV) regime. Leave the loop unannotated.
#ifdef GEMV_QBLOCK_W16
        // Read weights as uint4 (16 bytes, two C4 slices) and keep the same
        // pre-scaling mask math. The host also enables this branch for eligible
        // LN/QKV/GateUp fused non-ROW_2 pipelines.
        const device uint4* wt16 = (const device uint4*)(xy_wt + zmin);
        const device ftype4* in16 = xy_in0 + zmin * area_size;
        // middle_step lanes share a block evenly.
        constexpr int pair_count = (block / 2) / middle_step;
        const int pair_begin = middle_index * pair_count;
        for (int pair = pair_begin; pair < pair_begin + pair_count; ++pair) {
            const uint4 packed_weight = wt16[pair];
            const ushort4 weight0 = ushort4(packed_weight.x & 0xFFFFu, packed_weight.x >> 16,
                                            packed_weight.y & 0xFFFFu, packed_weight.y >> 16);
            const ushort4 weight1 = ushort4(packed_weight.z & 0xFFFFu, packed_weight.z >> 16,
                                            packed_weight.w & 0xFFFFu, packed_weight.w >> 16);
        #ifdef LN_FUSED
            // Fused into the block-input RMSNorm (has_ln): the same
            // raw*inv_rms*gamma transform the legacy loop applies, on the two
            // slices this lane owns. zmin+2*pair / +1 are the absolute slices.
            const int zln0 = zmin + 2 * pair;
            const FLOAT4 input0 = ((FLOAT4)in16[(2 * pair) * area_size]
                                   + (FLOAT4)*(ln_residual_in + zln0 * area_size))
                                  * inv_rms * (FLOAT4)ln_gamma[zln0];
            const FLOAT4 input1 = ((FLOAT4)in16[(2 * pair + 1) * area_size]
                                   + (FLOAT4)*(ln_residual_in + (zln0 + 1) * area_size))
                                  * inv_rms * (FLOAT4)ln_gamma[zln0 + 1];
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
        for (int z = zmin + middle_index; z < zmax; z += middle_step) {
        #ifdef LN_FUSED
            FLOAT4 raw = (FLOAT4)*(xy_in0 + z * area_size) + (FLOAT4)*(ln_residual_in + z * area_size);
            FLOAT4 in4 = raw * inv_rms * (FLOAT4)ln_gamma[z];
        #else
            FLOAT4 in4 = (FLOAT4)*(xy_in0 + z * area_size);
        #endif
            input_sum += in4[0] + in4[1] + in4[2] + in4[3];

            // Pre-scaling trick: avoid shift by pre-dividing input
            FLOAT in_ps0 = in4[0] * FLOAT(1.0/16.0);    // compensates nibble at bits[4:7] (×16)
            FLOAT in_ps2 = in4[2] * FLOAT(1.0/4096.0);  // compensates nibble at bits[12:15] (×4096)
            FLOAT in_ps3 = in4[3] * FLOAT(1.0/256.0);   // compensates nibble at bits[8:11] (×256)

            // Read weight as ushort4, mask without shift
            ushort4 w16 = xy_wt[z];
            raw_dot[0] += in_ps0 * FLOAT(w16[0] & 0x00F0) + in4[1] * FLOAT(w16[0] & 0x000F)
                        + in_ps2 * FLOAT(w16[0] & 0xF000) + in_ps3 * FLOAT(w16[0] & 0x0F00);
            raw_dot[1] += in_ps0 * FLOAT(w16[1] & 0x00F0) + in4[1] * FLOAT(w16[1] & 0x000F)
                        + in_ps2 * FLOAT(w16[1] & 0xF000) + in_ps3 * FLOAT(w16[1] & 0x0F00);
            raw_dot[2] += in_ps0 * FLOAT(w16[2] & 0x00F0) + in4[1] * FLOAT(w16[2] & 0x000F)
                        + in_ps2 * FLOAT(w16[2] & 0xF000) + in_ps3 * FLOAT(w16[2] & 0x0F00);
            raw_dot[3] += in_ps0 * FLOAT(w16[3] & 0x00F0) + in4[1] * FLOAT(w16[3] & 0x000F)
                        + in_ps2 * FLOAT(w16[3] & 0xF000) + in_ps3 * FLOAT(w16[3] & 0x0F00);
        }
#endif
        FLOAT4 adjusted_bias = dequant_bias - FLOAT(8.0) * scale;
        result += raw_dot * scale + input_sum * adjusted_bias;
    #elif defined(W_QUANT_8)
        // See W_QUANT_4 branch above for the pragma-unroll rationale.
        for (int z = zmin + middle_index; z < zmax; z += middle_step) {
            auto w = xy_wt[z];
            FLOAT4x4 w_fp32 = FLOAT4x4(FLOAT4(w[0]), FLOAT4(w[1]), FLOAT4(w[2]), FLOAT4(w[3]));
            FLOAT4x4 w_dequant;
            for (int i = 0; i < 4; ++i) {
                w_dequant[i] = w_fp32[i] * scale[i] + dequant_bias[i];
            }
        #ifdef LN_FUSED
            FLOAT4 raw = (FLOAT4)*(xy_in0 + z * area_size) + (FLOAT4)*(ln_residual_in + z * area_size);
            FLOAT4 in4 = raw * inv_rms * (FLOAT4)ln_gamma[z];
        #else
            FLOAT4 in4 = (FLOAT4)*(xy_in0 + z * area_size);
        #endif
            result += FLOAT4(in4 * w_dequant);
        }
    #endif
    }

    result = simd_sum(result);

#ifdef SPLIT_K_2
    threadgroup FLOAT4 sk_partial[2];
    if (sk_half == 1 && tiisg == 0) {
        sk_partial[(int)sgitg & 1] = result;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sk_half == 0 && tiisg == 0) {
        result += sk_partial[(int)sgitg & 1];
        out[uz * area_size] = activate(ftype4(result + biasValue), cst.activation);
    }
#else
    if (tiisg == 0) {
        out[uz * area_size] = activate(ftype4(result + biasValue), cst.activation);
    }
#endif
#endif // ROW_2
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
    auto xy_wt = wt + uz * cst.input_slice * 6;
#else
    auto xy_wt = wt + uz * cst.input_slice;
#endif
    auto xy_in0  = in + rx;
    auto area_size = cst.output_size * cst.batch;
    auto xy_out = out + uz * area_size + rx;
    FLOAT4 result0 = FLOAT4(0);
    threadgroup FLOAT4 localSum[32];
    if(uz < cst.output_slice) {
        int block = (cst.input_slice + cst.block_size - 1) / cst.block_size;
        
        int middle_step = min(SIMD_GROUP_WIDTH_4, block);
        int outer_step  = SIMD_GROUP_WIDTH_4 / middle_step;
        int middle_index = (tiisg + i_sgitg * SIMD_GROUP_WIDTH) % middle_step;
        int outer_index  = (tiisg + i_sgitg * SIMD_GROUP_WIDTH) / middle_step;


        for (int bi= outer_index; bi<cst.block_size; bi += outer_step) {
            FLOAT4 scale = FLOAT4(dequantScale[2 * (uz * cst.block_size + bi) + 0]) / (FLOAT)cst.scale_coef;
            FLOAT4 dequant_bias = FLOAT4(dequantScale[2 * (uz * cst.block_size + bi) + 1]) / (FLOAT)cst.scale_coef;
            int zmin = bi * block;
            int zmax = min(zmin + block, cst.input_slice);
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
                for (int z = zmin + middle_index; z < zmax; z += middle_step) {
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
                for (int z = zmin + middle_index; z < zmax; z += middle_step) {
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

                for (int z = zmin + middle_index; z < zmax; z += middle_step) {
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
            for (int z = zmin + middle_index; z < zmax; z += middle_step) {
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
    if(uz < cst.output_slice) {
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
    // 2 simdgroups per threadgroup; each simdgroup computes 8 output data (2 oc_4).
    const int GEMV_G16_SGS = 2;
    const int G16_ROWS = 2;
    int uz = G16_ROWS * (gid.x * GEMV_G16_SGS + sgitg);
    if(uz >= cst.output_slice) {
        return;
    }
    auto area_size = cst.output_size * cst.batch;
    int rx = gid.y;
#ifdef W_QUANT_3
    auto xy_wt = wt + uz * cst.input_slice * 6;
#else
    auto xy_wt = wt + uz * cst.input_slice;
#endif
    auto xy_in0  = in + rx;
    auto xy_out = out + uz * area_size + rx;
    auto biasValue0 = FLOAT4(biasTerms[uz]);
    auto biasValue1 = FLOAT4(biasTerms[uz + 1]);

    FLOAT4 result0 = FLOAT4(0);
    FLOAT4 result1 = FLOAT4(0);


    #ifdef GEMV_QBLOCK_W16_LMHEAD
    // Q4 block 32/64/128/256 specialization (see the g4m1_2sg kernel). Use up
    // to 8 lanes per block; every lane covers an integral number of uint4 pairs.
#ifndef GEMV_QBLOCK_W16_BLOCK_SLICES
#define GEMV_QBLOCK_W16_BLOCK_SLICES 16
#endif
    constexpr int block = GEMV_QBLOCK_W16_BLOCK_SLICES;
    constexpr int middle_step = block < 16 ? block / 2 : 8;
    constexpr int outer_step = SIMD_GROUP_WIDTH / middle_step;
    const int middle_index = int(tiisg) & (middle_step - 1);
    const int outer_index = int(tiisg) / middle_step;
    #else
    int block = (cst.input_slice + cst.block_size - 1) / cst.block_size;

    int middle_step = min(SIMD_GROUP_WIDTH, block);
    int outer_step  = SIMD_GROUP_WIDTH / middle_step;
    int middle_index = (tiisg) % middle_step;
    int outer_index  = (tiisg) / middle_step;
    #endif

    for (int bi= outer_index; bi<cst.block_size; bi += outer_step) {
        const int quant_offset = 2 * (uz * cst.block_size + bi);
        FLOAT4 scale0 = FLOAT4(dequantScale[quant_offset + 0]) / (FLOAT)cst.scale_coef;
        FLOAT4 dequant_bias0 = FLOAT4(dequantScale[quant_offset + 1]) / (FLOAT)cst.scale_coef;
        FLOAT4 scale1 = FLOAT4(dequantScale[quant_offset + (cst.block_size << 1)]) / (FLOAT)cst.scale_coef;
        FLOAT4 dequant_bias1 = FLOAT4(dequantScale[quant_offset + (cst.block_size << 1) + 1]) / (FLOAT)cst.scale_coef;
        int zmin = bi * block;
        #ifdef GEMV_QBLOCK_W16_LMHEAD
        int zmax = zmin + block;
        #else
        int zmax = min(zmin + block, cst.input_slice);
        #endif

        #if defined(W_QUANT_2) || defined(W_QUANT_3)
        // Deferred + pre-scaling dual-row streams (mask/pre-scale derivation in
        // the g8 kernel). Trap-A: must precede the W4 branch.
        {
            FLOAT4 raw_dot0 = FLOAT4(0), raw_dot1 = FLOAT4(0);
            FLOAT input_sum = FLOAT(0);

            for (int z = zmin + middle_index; z < zmax; z += middle_step) {
                FLOAT4 in4 = (FLOAT4)*(xy_in0 + z * area_size);
                input_sum += in4[0] + in4[1] + in4[2] + in4[3];

                FLOAT in_ps0 = in4[0] * FLOAT(1.0/64.0);
                FLOAT in_ps1 = in4[1] * FLOAT(1.0/16.0);
                FLOAT in_ps2 = in4[2] * FLOAT(1.0/4.0);
            #ifdef W_QUANT_2
                uchar4 w0 = xy_wt[z];
                uchar4 w1 = xy_wt[cst.input_slice + z];
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
                const device ushort* t1 = (const device ushort*)(xy_wt + (cst.input_slice + z) * 6);
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

            #ifdef GEMV_QBLOCK_W16_LMHEAD
            // Each iteration covers two slices; both rows use one 16-byte load.
            constexpr int pair_count = (block / 2) / middle_step;
            const int pair_begin = middle_index * pair_count;
            for (int pair = pair_begin; pair < pair_begin + pair_count; ++pair) {
                const int z = zmin + pair * 2;
                const uint4 packed0 = *((const device uint4*)(xy_wt + z));
                const uint4 packed1 = *((const device uint4*)(xy_wt + cst.input_slice + z));
                const ushort4 w00 = ushort4(packed0.x & 0xFFFFu, packed0.x >> 16,
                                            packed0.y & 0xFFFFu, packed0.y >> 16);
                const ushort4 w01 = ushort4(packed0.z & 0xFFFFu, packed0.z >> 16,
                                            packed0.w & 0xFFFFu, packed0.w >> 16);
                const ushort4 w10 = ushort4(packed1.x & 0xFFFFu, packed1.x >> 16,
                                            packed1.y & 0xFFFFu, packed1.y >> 16);
                const ushort4 w11 = ushort4(packed1.z & 0xFFFFu, packed1.z >> 16,
                                            packed1.w & 0xFFFFu, packed1.w >> 16);
                const FLOAT4 in0 = FLOAT4(xy_in0[z * area_size]);
                const FLOAT4 in1 = FLOAT4(xy_in0[(z + 1) * area_size]);
                input_sum += in0[0] + in0[1] + in0[2] + in0[3] +
                             in1[0] + in1[1] + in1[2] + in1[3];
                const FLOAT in0_ps0 = in0[0] * FLOAT(1.0 / 16.0);
                const FLOAT in0_ps2 = in0[2] * FLOAT(1.0 / 4096.0);
                const FLOAT in0_ps3 = in0[3] * FLOAT(1.0 / 256.0);
                const FLOAT in1_ps0 = in1[0] * FLOAT(1.0 / 16.0);
                const FLOAT in1_ps2 = in1[2] * FLOAT(1.0 / 4096.0);
                const FLOAT in1_ps3 = in1[3] * FLOAT(1.0 / 256.0);
                raw_dot0 += in0_ps0 * FLOAT4(w00 & ushort4(0x00F0)) + in0[1] * FLOAT4(w00 & ushort4(0x000F)) +
                            in0_ps2 * FLOAT4(w00 & ushort4(0xF000)) + in0_ps3 * FLOAT4(w00 & ushort4(0x0F00)) +
                            in1_ps0 * FLOAT4(w01 & ushort4(0x00F0)) + in1[1] * FLOAT4(w01 & ushort4(0x000F)) +
                            in1_ps2 * FLOAT4(w01 & ushort4(0xF000)) + in1_ps3 * FLOAT4(w01 & ushort4(0x0F00));
                raw_dot1 += in0_ps0 * FLOAT4(w10 & ushort4(0x00F0)) + in0[1] * FLOAT4(w10 & ushort4(0x000F)) +
                            in0_ps2 * FLOAT4(w10 & ushort4(0xF000)) + in0_ps3 * FLOAT4(w10 & ushort4(0x0F00)) +
                            in1_ps0 * FLOAT4(w11 & ushort4(0x00F0)) + in1[1] * FLOAT4(w11 & ushort4(0x000F)) +
                            in1_ps2 * FLOAT4(w11 & ushort4(0xF000)) + in1_ps3 * FLOAT4(w11 & ushort4(0x0F00));
            }
            #else
            for (int z = zmin + middle_index; z < zmax; z += middle_step) {
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
                w16 = xy_wt[cst.input_slice + z];
                raw_dot1[0] += in_ps0 * FLOAT(w16[0] & 0x00F0) + in4[1] * FLOAT(w16[0] & 0x000F)
                            + in_ps2 * FLOAT(w16[0] & 0xF000) + in_ps3 * FLOAT(w16[0] & 0x0F00);
                raw_dot1[1] += in_ps0 * FLOAT(w16[1] & 0x00F0) + in4[1] * FLOAT(w16[1] & 0x000F)
                            + in_ps2 * FLOAT(w16[1] & 0xF000) + in_ps3 * FLOAT(w16[1] & 0x0F00);
                raw_dot1[2] += in_ps0 * FLOAT(w16[2] & 0x00F0) + in4[1] * FLOAT(w16[2] & 0x000F)
                            + in_ps2 * FLOAT(w16[2] & 0xF000) + in_ps3 * FLOAT(w16[2] & 0x0F00);
                raw_dot1[3] += in_ps0 * FLOAT(w16[3] & 0x00F0) + in4[1] * FLOAT(w16[3] & 0x000F)
                            + in_ps2 * FLOAT(w16[3] & 0xF000) + in_ps3 * FLOAT(w16[3] & 0x0F00);
            }
            #endif
            FLOAT4 adj0 = dequant_bias0 - FLOAT(8.0) * scale0;
            FLOAT4 adj1 = dequant_bias1 - FLOAT(8.0) * scale1;
            result0 += raw_dot0 * scale0 + input_sum * adj0;
            result1 += raw_dot1 * scale1 + input_sum * adj1;
        }
        #elif defined(W_QUANT_8)
        for (int z = zmin + middle_index; z < zmax; z += middle_step) {
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
                auto w = xy_wt[cst.input_slice + z];
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

    /* true */
    if (tiisg == 0) {
        xy_out[0] = activate(ftype4(res0 + biasValue0), cst.activation);
        xy_out[area_size] = activate(ftype4(res1 + biasValue1), cst.activation);
    }
}
)metal";

#endif
