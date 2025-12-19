//
//  ConvSimdGroupShader.hpp
//  MNN
//
//  Created by MNN on b'2024/12/30'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#if MNN_METAL_ENABLED

const char* gBasicConvPrefix = R"metal(
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


const char* gConv1x1WqSgMatrix = R"metal(
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
        FLOAT2 scale = FLOAT2(dequantScale[(2 * (idx_n4 * cst.block_size + bi) + 0) * 2 + nl / 2]) / (FLOAT)cst.scale_coef;
        FLOAT2 dequant_bias = FLOAT2(dequantScale[(2 * (idx_n4 * cst.block_size + bi) + 1) * 2 + nl / 2]) / (FLOAT)cst.scale_coef;
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
                
            FLOAT2 res = w20 * scale[nl % 2] + dequant_bias[nl % 2];
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
                            const device ftype *dequantScale  [[buffer(5)]],
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
)metal";

const char* gConv1x1WfpSgMatrix = R"metal(
#ifdef USE_METAL_TENSOR_OPS
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif

kernel void conv1x1_w_dequant(
                        #ifdef W_QUANT_4
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

    auto xy_wi = wi + (idx_n4 * cst.input_slice + idx_k4) * 4 + idx_nl;// [N/4, K/4, N4, K4]
    auto xy_wf = wf + ((idx_n4 * (cst.input_slice/4) + idx_k16) * 4 + idx_nl) * 4;// [N/4, K/4, N4, K4]

    #ifdef W_QUANT_4
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
    auto xy_wt = wt +  (idx_n4 * (cst.input_slice/4) + idx_wk16) * 4 + nl;// [N/4, K/16, N4, K4, K4]
    
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
    auto xy_wt = wt +  (idx_n4 * (cst.input_slice/4) + idx_wk16) * 4 + nl;// [N/4, K/16, N4, K4, K4]
    
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
            
            ((threadgroup ftype4*)sdata)[idx_sa]     = (ftype4)*(xy_in0);
            ((threadgroup ftype4*)sdata)[idx_sa + 8] = (ftype4)*(xy_in1);

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

    simdgroup_half8x8 sga[2];
    simdgroup_half8x8 sgb[4];
    simdgroup_float8x8 sgd[8];
    for (int i = 0; i < 8; i++){
        sgd[i] = make_filled_simdgroup_matrix<FLOAT, 8>(0.f);
    }

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
    auto xy_wt = wt +  (idx_n4 * (cst.input_slice/4) + idx_wk16) * 4 + nl;// [N/4, K/16, N4, K4, K4]
    
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

)metal";


const char* gConv1x1WfpSgReduce = R"metal(
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

const char* gConv1x1WqSgReduce = R"metal(

template <int AREA_THREAD>
kernel void conv1x1_gemv_g4mx_wquant_sg(const device ftype4 *in            [[buffer(0)]],
                            device ftype4 *out                 [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                        #ifdef W_QUANT_4
                            const device MNN::uchar4x2 *wt      [[buffer(3)]],
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
    auto xy_wt = wt + uz * cst.input_slice;
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
        for (int z = zmin + middle_index; z < zmax; z += middle_step) {
            #ifdef W_QUANT_4
                MNN::uchar4x2 w_int4 = xy_wt[z];

                FLOAT4x4 w_dequant;
                for (int i = 0; i < 4; i += 1) {
                    FLOAT4 w4 = FLOAT4((float)(w_int4[i][0] >> 4) - 8, (float)(w_int4[i][0] & 15) - 8, (float)(w_int4[i][1] >> 4) - 8, (float)(w_int4[i][1] & 15) - 8);
                    FLOAT4 res = w4 * scale[i] + dequant_bias[i];
                    w_dequant[i] = res;
                }
            #elif defined(W_QUANT_8)
                auto w = xy_wt[z];
                FLOAT4x4 w_fp32 = FLOAT4x4(FLOAT4(w[0]), FLOAT4(w[1]), FLOAT4(w[2]), FLOAT4(w[3]));
                FLOAT4x4 w_dequant;
                for (int i = 0; i < 4; ++i) {
                    w_dequant[i] = w_fp32[i] * scale[i] + dequant_bias[i];
                }
            #endif

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

kernel void conv1x1_gemv_g8_wquant_sg(const device ftype4 *in            [[buffer(0)]],
                            device ftype4 *out                 [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                        #ifdef W_QUANT_4
                            const device MNN::uchar4x2 *wt      [[buffer(3)]],
                        #elif defined(W_QUANT_8)
                            const device MNN::char4x4 *wt      [[buffer(3)]],
                        #endif
                            const device ftype4 *biasTerms     [[buffer(4)]],
                            const device ftype4 *dequantScale  [[buffer(5)]],
                            uint3 gid[[threadgroup_position_in_grid]],
                            uint  tiisg[[thread_index_in_simdgroup]],
                            uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    // each threadgroup contain 2 simdgroup
    // each simdgroup compute 4 data
    int uz = gid.x * 2 + sgitg;
    if(uz >= cst.output_slice) {
        return;
    }

    int rx = gid.y;
    auto xy_wt = wt + uz * cst.input_slice;
    auto xy_in0  = in + rx;
    auto area_size = cst.output_size * cst.batch;
    auto xy_out = out + uz * area_size + rx;
    auto biasValue = FLOAT4(biasTerms[uz]);
    FLOAT4 result0 = FLOAT4(0);

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
        for (int z = zmin + middle_index; z < zmax; z += middle_step) {
            FLOAT4 in40 = (FLOAT4)*(xy_in0 + z * area_size);
            
            #ifdef W_QUANT_4
                MNN::uchar4x2 w_int4 = xy_wt[z];

                FLOAT4x4 w_dequant;
                for (int i = 0; i < 4; i += 1) {
                    FLOAT4 w4 = FLOAT4((float)(w_int4[i][0] >> 4) - 8, (float)(w_int4[i][0] & 15) - 8, (float)(w_int4[i][1] >> 4) - 8, (float)(w_int4[i][1] & 15) - 8);
                    FLOAT4 res = w4 * scale[i] + dequant_bias[i];
                    w_dequant[i] = res;
                }
            #elif defined(W_QUANT_8)
                auto w = xy_wt[z];
                FLOAT4x4 w_fp32 = FLOAT4x4(FLOAT4(w[0]), FLOAT4(w[1]), FLOAT4(w[2]), FLOAT4(w[3]));
                FLOAT4x4 w_dequant;
                for (int i = 0; i < 4; ++i) {
                    w_dequant[i] = w_fp32[i] * scale[i] + dequant_bias[i];
                }
            #endif

            result0 += FLOAT4(in40 * w_dequant);
            
        }
    }

    FLOAT4 res = simd_sum(result0);
    /* true */
    if (tiisg == 0) {
        xy_out[0] = activate(ftype4(res + biasValue), cst.activation);
    }
}

kernel void conv1x1_gemv_g16_wquant_sg(const device ftype4 *in            [[buffer(0)]],
                            device ftype4 *out                 [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                        #ifdef W_QUANT_4
                            const device MNN::uchar4x2 *wt      [[buffer(3)]],
                        #elif defined(W_QUANT_8)
                            const device MNN::char4x4 *wt      [[buffer(3)]],
                        #endif
                            const device ftype4 *biasTerms     [[buffer(4)]],
                            const device ftype4 *dequantScale  [[buffer(5)]],
                            uint3 gid[[threadgroup_position_in_grid]],
                            uint  tiisg[[thread_index_in_simdgroup]],
                            uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    // each threadgroup contain 2 simdgroup
    // each simdgroup compute 8 data
    int uz = 2 * (gid.x * 2 + sgitg);
    if(uz >= cst.output_slice) {
        return;
    }
    auto area_size = cst.output_size * cst.batch;
    int rx = gid.y;
    auto xy_wt = wt + uz * cst.input_slice;
    auto xy_in0  = in + rx;
    auto xy_out = out + uz * area_size + rx;
    auto biasValue0 = FLOAT4(biasTerms[uz]);
    auto biasValue1 = FLOAT4(biasTerms[uz + 1]);

    FLOAT4 result0 = FLOAT4(0);
    FLOAT4 result1 = FLOAT4(0);

    int block = (cst.input_slice + cst.block_size - 1) / cst.block_size;
    
    int middle_step = min(SIMD_GROUP_WIDTH, block);
    int outer_step  = SIMD_GROUP_WIDTH / middle_step;
    int middle_index = (tiisg) % middle_step;
    int outer_index  = (tiisg) / middle_step;
    
    for (int bi= outer_index; bi<cst.block_size; bi += outer_step) {
        const int quant_offset = 2 * (uz * cst.block_size + bi);
        FLOAT4 scale0 = FLOAT4(dequantScale[quant_offset + 0]) / (FLOAT)cst.scale_coef;
        FLOAT4 dequant_bias0 = FLOAT4(dequantScale[quant_offset + 1]) / (FLOAT)cst.scale_coef;
        FLOAT4 scale1 = FLOAT4(dequantScale[quant_offset + (cst.block_size << 1)]) / (FLOAT)cst.scale_coef;
        FLOAT4 dequant_bias1 = FLOAT4(dequantScale[quant_offset + (cst.block_size << 1) + 1]) / (FLOAT)cst.scale_coef;
        int zmin = bi * block;
        int zmax = min(zmin + block, cst.input_slice);
        for (int z = zmin + middle_index; z < zmax; z += middle_step) {
            FLOAT4 in40 = (FLOAT4)*(xy_in0 + z * area_size);
            
            #ifdef W_QUANT_4
                MNN::uchar4x2 w_int4 = xy_wt[z];

                FLOAT4x4 w_dequant;
                for (int i = 0; i < 4; i += 1) {
                    FLOAT4 w4 = FLOAT4((float)(w_int4[i][0] >> 4) - 8, (float)(w_int4[i][0] & 15) - 8, (float)(w_int4[i][1] >> 4) - 8, (float)(w_int4[i][1] & 15) - 8);
                    FLOAT4 res = w4 * scale0[i] + dequant_bias0[i];
                    w_dequant[i] = res;
                }
            #elif defined(W_QUANT_8)
                auto w = xy_wt[z];
                FLOAT4x4 w_fp32 = FLOAT4x4(FLOAT4(w[0]), FLOAT4(w[1]), FLOAT4(w[2]), FLOAT4(w[3]));
                FLOAT4x4 w_dequant;
                for (int i = 0; i < 4; ++i) {
                    w_dequant[i] = w_fp32[i] * scale0[i] + dequant_bias0[i];
                }
            #endif

            result0 += FLOAT4(in40 * w_dequant);

            #ifdef W_QUANT_4
                w_int4 = xy_wt[cst.input_slice + z];
                for (int i = 0; i < 4; i += 1) {
                    FLOAT4 w4 = FLOAT4((float)(w_int4[i][0] >> 4) - 8, (float)(w_int4[i][0] & 15) - 8, (float)(w_int4[i][1] >> 4) - 8, (float)(w_int4[i][1] & 15) - 8);
                    FLOAT4 res = w4 * scale1[i] + dequant_bias1[i];
                    w_dequant[i] = res;
                }
            #elif defined(W_QUANT_8)
                w = xy_wt[cst.input_slice + z];
                w_fp32 = FLOAT4x4(FLOAT4(w[0]), FLOAT4(w[1]), FLOAT4(w[2]), FLOAT4(w[3]));
                for (int i = 0; i < 4; ++i) {
                    w_dequant[i] = w_fp32[i] * scale1[i] + dequant_bias1[i];
                }
            #endif

            result1 += FLOAT4(in40 * w_dequant);
            
        }
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

