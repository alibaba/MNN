//
//  ConvSimdGroupShader.hpp
//  MNN
//
//  Created by MNN on b'2024/12/30'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#if MNN_METAL_ENABLED

#include "MetalConv1x1Kernels.hpp"

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
inline ftype activate(ftype value, conv_activation_type type) {
    switch (type) {
        case ReLU:
            return max(value, (ftype)0);
        case ReLU6:
            return clamp(value, (ftype)0, (ftype)6);
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

        inline thread uchar2& operator[] (const int index) thread {
            return v[index];
        }
        inline device uchar2& operator[] (const int index) device {
            return v[index];
        }
        inline threadgroup uchar2& operator[] (const int index) threadgroup {
            return v[index];
        }

        inline const thread uchar2& operator[] (const int index) const thread {
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

        inline thread char4& operator[] (const int index) thread {
            return v[index];
        }
        inline device char4& operator[] (const int index) device {
            return v[index];
        }
        inline threadgroup char4& operator[] (const int index) threadgroup {
            return v[index];
        }

        inline const thread char4& operator[] (const int index) const thread {
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

// MoE shared-expert gate scale folded into the epilogue: the row of token `token`
// is multiplied by rowScale[token].x before it reaches memory. Only the kernels
// that declare the rowScale argument take part. Conv1x1 compiles those kernels
// with CONV_ROW_MUL when a row scale is offered; every other route
// leaves the definition empty and the caller applies the scale itself.
#ifdef CONV_ROW_MUL
#define CONV_ROW_SCALE(value, token) ((value) = ftype4(float4(value) * float(rowScale[token].x)))
#else
#define CONV_ROW_SCALE(value, token)
#endif
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
// Single-pass in-TG split-K simdgroup_matrix GEMM for non-Tensor short prefill (area 6..8).
// Supports W2/W3/W4/W8 and FP16/FP32; each SG accumulates a disjoint K slice, then aliases smem to reduce once.
#if defined(W_QUANT_2) || defined(W_QUANT_3) || defined(W_QUANT_4) || defined(W_QUANT_8)
kernel void conv1x1_gemm_8x16_split_k_wquant_sg(const device ftype4* input [[buffer(0)]],
                                                device ftype4* output [[buffer(1)]],
                                                constant conv1x1_constants& cst [[buffer(2)]],
#ifdef W_QUANT_2
                                                const device uchar4* weights [[buffer(3)]],
#elif defined(W_QUANT_3)
                                                const device uchar* weights [[buffer(3)]],
#elif defined(W_QUANT_4)
                                                const device ushort4* weights [[buffer(3)]],
#else
                                                const device uint4* weights [[buffer(3)]],
#endif
                                                const device ftype4* bias [[buffer(4)]],
                                                const device ftype4* scales [[buffer(5)]],
                                                uint group [[threadgroup_position_in_grid]],
                                                ushort sg [[simdgroup_index_in_threadgroup]],
                                                ushort lane [[thread_index_in_simdgroup]]) {
#if MNN_METAL_FLOAT16_STORAGE
    constexpr int K = 32, N = 16, S = 6;
#else
    constexpr int K = 16, N = 32, S = 8;
#endif
    constexpr int Q = K / 4;
    const int area = cst.inputSize * cst.batch;
    constexpr int tileBytes = S * (8 + N) * Q * sizeof(ftype4);
    constexpr int partBytes = S * 8 * (N / 4) * sizeof(float4);
    threadgroup float4 smem[(tileBytes > partBytes ? tileBytes : partBytes) / sizeof(float4)];
    auto aTile = (threadgroup ftype4 (*)[8][Q])smem;
    auto bTile = (threadgroup ftype4 (*)[N][Q])((threadgroup ftype4*)smem + S * 8 * Q);
    constexpr int lanesPerQuad = 32 / (N / 4);
    const int row = lane / lanesPerQuad, kLane = lane % lanesPerQuad;
    const int oc4 = min(int(group * (N / 4)) + row, cst.outputDepthQuad - 1);
    simdgroup_float8x8 acc[N / 8];
    for (ushort n = 0; n < N / 8; ++n) {
        acc[n] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }
    const int quads = (cst.inputDepthQuad + cst.blockCount - 1) / cst.blockCount;
    const int tiles = (quads + Q - 1) / Q, length = cst.blockCount * tiles;
    const int first = sg * length / S, last = (sg + 1) * length / S;
    for (int block = first / tiles; block * tiles < last; ++block) {
        const int idx = (oc4 * cst.blockCount + block) * 2;
        const float4 scale = float4(scales[idx]) / cst.scaleCoef;
        const float4 dbias = float4(scales[idx + 1]) / cst.scaleCoef;
#ifdef W_QUANT_2
        const float4 offset = dbias - 2.f * scale;
#elif defined(W_QUANT_3)
        const float4 offset = dbias - 4.f * scale;
#elif defined(W_QUANT_4)
        const float4 offset = dbias - 8.f * scale;
#else
        const float4 offset = dbias;
#endif
        const int end = min((block + 1) * quads, cst.inputDepthQuad);
        for (int t = max(first - block * tiles, 0); t < min(tiles, last - block * tiles); ++t) {
            const int k = block * quads + t * Q;
            for (ushort elem = lane; elem < 8 * Q; elem += 32) {
                const int q = k + elem / 8, m = elem % 8;
#ifdef SG8_SPLITK_FULL_K_TILE
                aTile[sg][m][elem / 8] = (m < area) ? input[q * area + m] : ftype4(0);
#else
                aTile[sg][m][elem / 8] = (q < end && m < area) ? input[q * area + m] : ftype4(0);
#endif
            }
            for (ushort l = kLane; l < Q; l += lanesPerQuad) {
#ifdef SG8_SPLITK_FULL_K_TILE
                const int q = k + l;
#else
                const int q = min(k + l, end - 1);
#endif
#ifdef W_QUANT_2
                const uchar4 bytes = weights[oc4 * cst.inputDepthQuad + q];
                for (ushort j = 0; j < 4; ++j) {
                    const uchar b = bytes[j];
                    const float4 codes((b >> 6) & 3, (b >> 4) & 3, (b >> 2) & 3, b & 3);
                    bTile[sg][row * 4 + j][l] = ftype4(codes * scale[j] + offset[j]);
                }
#elif defined(W_QUANT_3)
                const device ushort* tp = (const device ushort*)(weights + (oc4 * cst.inputDepthQuad + q) * 6);
                const ushort t0 = tp[0], t1 = tp[1], t2 = tp[2];
                const uchar los[4] = { uchar(t0 & 0xFF), uchar(t0 >> 8), uchar(t1 & 0xFF), uchar(t1 >> 8) };
                const uchar hs[4]  = { uchar((t2 >> 4) & 0xF), uchar(t2 & 0xF), uchar((t2 >> 12) & 0xF), uchar((t2 >> 8) & 0xF) };
                for (ushort j = 0; j < 4; ++j) {
                    const uchar lo = los[j], h = hs[j];
                    const float4 codes(((lo >> 6) & 3) | (((h >> 3) & 1) << 2),
                                       ((lo >> 4) & 3) | (((h >> 2) & 1) << 2),
                                       ((lo >> 2) & 3) | (((h >> 1) & 1) << 2),
                                       ( lo       & 3) | (( h       & 1) << 2));
                    bTile[sg][row * 4 + j][l] = ftype4(codes * scale[j] + offset[j]);
                }
#elif defined(W_QUANT_4)
                const ushort4 words = weights[oc4 * cst.inputDepthQuad + q];
                for (ushort j = 0; j < 4; ++j) {
                    const ushort w = words[j];
                    const float4 codes((w >> 4) & 15, w & 15, w >> 12, (w >> 8) & 15);
                    bTile[sg][row * 4 + j][l] = ftype4(codes * scale[j] + offset[j]);
                }
#else // W_QUANT_8
                const uint4 words = weights[oc4 * cst.inputDepthQuad + q];
                const char4 c0 = as_type<char4>(words.x), c1 = as_type<char4>(words.y);
                const char4 c2 = as_type<char4>(words.z), c3 = as_type<char4>(words.w);
                bTile[sg][row * 4 + 0][l] = ftype4(float4(c0.x, c0.y, c0.z, c0.w) * scale[0] + offset[0]);
                bTile[sg][row * 4 + 1][l] = ftype4(float4(c1.x, c1.y, c1.z, c1.w) * scale[1] + offset[1]);
                bTile[sg][row * 4 + 2][l] = ftype4(float4(c2.x, c2.y, c2.z, c2.w) * scale[2] + offset[2]);
                bTile[sg][row * 4 + 3][l] = ftype4(float4(c3.x, c3.y, c3.z, c3.w) * scale[3] + offset[3]);
#endif
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (ushort i = 0; i < K; i += 8) {
                simdgroup_FTYPE8x8 a, b;
                simdgroup_load(a, (threadgroup ftype*)aTile[sg] + i, K);
                for (ushort n = 0; n < N / 8; ++n) {
                    simdgroup_load(b, (threadgroup ftype*)bTile[sg] + n * 8 * K + i, K, ulong2(0), true);
                    simdgroup_multiply_accumulate(acc[n], a, b, acc[n]);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto partial = (threadgroup float4 (*)[8][N / 4])smem;
    for (ushort n = 0; n < N / 8; ++n) {
        simdgroup_store(acc[n], (threadgroup float*)partial[sg] + n * 8, N);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort elem = sg * 32 + lane; elem < 8 * (N / 4); elem += S * 32) {
        const int n4 = elem / 8, token = elem % 8, dst = group * (N / 4) + n4;
        if (dst < cst.outputDepthQuad && token < area) {
            float4 total = partial[0][token][n4];
            for (ushort part = 1; part < S; ++part) {
                total += partial[part][token][n4];
            }
            output[dst * area + token] = activate(ftype4(total + float4(bias[dst])), cst.activation);
        }
    }
}
#endif // W_QUANT_2 || W_QUANT_3 || W_QUANT_4 || W_QUANT_8

// W_QUANT_2/3 legacy sg_matrix kernels are not implemented; guard them below.
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
    int m0 = 32 * rx + rcl;
    int m1 = 32 * rx + 16 + rcl;
    int idx_m0  = m0 < size_m ? m0 : (size_m - 1);
    int idx_m1  = m1 < size_m ? m1 : (size_m - 1);

    auto xy_wt = wt + ((long)idx_n4 * cst.inputDepthQuad + 0) * 4 + rcl % 4;// [N/4, K/4, N4, K4]
    auto xy_in0  = in + idx_m0 + (long)cst.inputSize * cst.batch * kl;// [K/4, M, K4]
    auto xy_in1  = in + idx_m1 + (long)cst.inputSize * cst.batch * kl;// [K/4, M, K4]

    auto xy_out0 = out + (long)(4 * uz + 2 * kl) * cst.outputSize * cst.batch + idx_m0;// [N/4, M, N4]
    auto xy_out1 = out + (long)(4 * uz + 2 * kl) * cst.outputSize * cst.batch + idx_m1;// [N/4, M, N4]

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

    if(m0 < size_m) {
        if((4 * uz + 2 * kl) < cst.outputDepthQuad) {
            xy_out0[0] =  activate(ftype4(sdata[(kl * 32 + rcl) * 2 + 0] + FLOAT4(biasTerms[4 * uz + 2 * kl + 0])), cst.activation);
        }
        if((4 * uz + 2 * kl + 1) < cst.outputDepthQuad) {
            xy_out0[cst.outputSize * cst.batch] = activate(ftype4(sdata[(kl * 32 + rcl) * 2 + 1] + FLOAT4(biasTerms[4 * uz + 2 * kl + 1])), cst.activation);
        }
    }
    if(m1 < size_m) {
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
    int n40 = 8 * uz + rcl / 4;
    int n41 = 8 * uz + 4 + rcl / 4;
    int idx_n40 = n40 < cst.outputDepthQuad ? n40 : (cst.outputDepthQuad - 1);
    int idx_n41 = n41 < cst.outputDepthQuad ? n41 : (cst.outputDepthQuad - 1);

    int idx_m  = (16 * rx + rcl) < cst.inputSize * cst.batch ? (16 * rx + rcl) : (cst.inputSize * cst.batch - 1);

    auto xy_wt0 = wt + ((long)idx_n40 * cst.inputDepthQuad + 0) * 4 + (rcl % 4);// [N/4, K/4, N4, K4]
    auto xy_wt1 = wt + ((long)idx_n41 * cst.inputDepthQuad + 0) * 4 + (rcl % 4);// [N/4, K/4, N4, K4]

    auto xy_in0  = in + idx_m + (long)cst.inputSize * cst.batch * kl;// [K/4, M, K4]
    int out_n0 = 8 * uz + 2 * kl;
    int out_n1 = 8 * uz + 4 + 2 * kl;
    auto xy_out0 = out + (long)out_n0 * cst.outputSize * cst.batch + idx_m;// [N/4, M, N4]
    auto xy_out1 = out + (long)out_n1 * cst.outputSize * cst.batch + idx_m;// [N/4, M, N4]

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
        if(out_n0 < cst.outputDepthQuad) {
            xy_out0[0] =  activate(ftype4(sdata[(kl * 16 + rcl) * 2 + 0] + FLOAT4(biasTerms[out_n0])), cst.activation);
        }
        if(out_n0 + 1 < cst.outputDepthQuad) {
            xy_out0[cst.outputSize * cst.batch] = activate(ftype4(sdata[(kl * 16 + rcl) * 2 + 1] + FLOAT4(biasTerms[out_n0 + 1])), cst.activation);
        }
        if(out_n1 < cst.outputDepthQuad) {
            xy_out1[0] = activate(ftype4(sdata[((kl + 2) * 16 + rcl) * 2 + 0] + FLOAT4(biasTerms[out_n1])), cst.activation);
        }
        if(out_n1 + 1 < cst.outputDepthQuad) {
            xy_out1[cst.outputSize * cst.batch] = activate(ftype4(sdata[((kl + 2) * 16 + rcl) * 2 + 1] + FLOAT4(biasTerms[out_n1 + 1])), cst.activation);
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
            #ifdef MNN_METAL_SRC_PROTECT
                if (z + kwl * 4 < cst.inputDepthQuad) {
                    auto w = xy_wt[z];
                    w_dequant = FLOAT4x4((FLOAT4)w[0], (FLOAT4)w[1], (FLOAT4)w[2], (FLOAT4)w[3]);
                } else {
                    w_dequant = FLOAT4x4(FLOAT4(0), FLOAT4(0), FLOAT4(0), FLOAT4(0));
                }
            #else
                auto w = xy_wt[z];
                w_dequant = FLOAT4x4((FLOAT4)w[0], (FLOAT4)w[1], (FLOAT4)w[2], (FLOAT4)w[3]);
            #endif
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
            #ifdef MNN_METAL_SRC_PROTECT
                if (z + kwl * 4 < cst.inputDepthQuad) {
                    auto w = xy_wt[z];
                    w_dequant = FLOAT4x4((FLOAT4)w[0], (FLOAT4)w[1], (FLOAT4)w[2], (FLOAT4)w[3]);
                } else {
                    w_dequant = FLOAT4x4(FLOAT4(0), FLOAT4(0), FLOAT4(0), FLOAT4(0));
                }
            #else
                auto w = xy_wt[z];
                w_dequant = FLOAT4x4((FLOAT4)w[0], (FLOAT4)w[1], (FLOAT4)w[2], (FLOAT4)w[3]);
            #endif
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
    int m0 = 32 * rx + rcl;
    int m1 = 32 * rx + 16 + rcl;
    int idx_m0  = m0 < size_m ? m0 : (size_m - 1);
    int idx_m1  = m1 < size_m ? m1 : (size_m - 1);

    auto xy_wt = wt + ((long)idx_n4 * cst.inputDepthQuad + 0) * 4 + rcl % 4;// [N/4, K/4, N4, K4]
    auto xy_in0  = in + idx_m0 + (long)cst.inputSize * cst.batch * kl;// [K/4, M, K4]
    auto xy_in1  = in + idx_m1 + (long)cst.inputSize * cst.batch * kl;// [K/4, M, K4]

    auto xy_out0 = out + (long)(4 * uz + 2 * kl) * cst.outputSize * cst.batch + idx_m0;// [N/4, M, N4]
    auto xy_out1 = out + (long)(4 * uz + 2 * kl) * cst.outputSize * cst.batch + idx_m1;// [N/4, M, N4]

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

    if(m0 < size_m) {
        if((4 * uz + 2 * kl) < cst.outputDepthQuad) {
            xy_out0[0] =  activate(ftype4(sdata[(kl * 32 + rcl) * 2 + 0] + FLOAT4(biasTerms[4 * uz + 2 * kl + 0])), cst.activation);
        }
        if((4 * uz + 2 * kl + 1) < cst.outputDepthQuad) {
            xy_out0[cst.outputSize * cst.batch] = activate(ftype4(sdata[(kl * 32 + rcl) * 2 + 1] + FLOAT4(biasTerms[4 * uz + 2 * kl + 1])), cst.activation);
        }
    }
    if(m1 < size_m) {
        if((4 * uz + 2 * kl) < cst.outputDepthQuad) {
            xy_out1[0] =  activate(ftype4(sdata[(kl * 32 + 16 + rcl) * 2 + 0] + FLOAT4(biasTerms[4 * uz + 2 * kl + 0])), cst.activation);
        }
        if((4 * uz + 2 * kl + 1) < cst.outputDepthQuad) {
            xy_out1[cst.outputSize * cst.batch] = activate(ftype4(sdata[(kl * 32 + 16 + rcl) * 2 + 1] + FLOAT4(biasTerms[4 * uz + 2 * kl + 1])), cst.activation);
        }
    }
}




#ifndef FUSED_KSPLIT_KS
#define FUSED_KSPLIT_KS 4
#endif
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

)metal";

//======================================================================
// conv1x1_fused_q4_gemm_stage_m64 — M=64 tile variant
//----------------------------------------------------------------------
// Same fused-Q4 GEMM as conv1x1_fused_q4_gemm_stage but with an M=64 tile
// instead of M=32, N unchanged. Halves grid.x for prefill, so each threadgroup
// keeps the same K-cost (weight reads = N*K, dequant work) while producing 2x
// the M output, which halves the weight-read redundancy across threadgroups.
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
#ifndef FQ_TILE_M
#define FQ_TILE_M 64
#endif
#ifndef FQ_TILE_N
#define FQ_TILE_N 64
#endif
#ifndef FQ_TILE_K
#define FQ_TILE_K 32
#endif
#ifndef FQ_SPLIT_K_COUNT
#define FQ_SPLIT_K_COUNT 1
#endif
#ifdef FQ_MULTI_PROJ_OUTPUT
struct FqMultiProjSegment {
    int firstTile;
    int outputQuads;
    float scaleCoef;
};
#endif
kernel void conv1x1_fused_q4_gemm_stage_m64(
                            const device ftype4 *in            [[buffer(0)]],
#if FQ_SPLIT_K_COUNT > 1
                            device float4 *partialOut          [[buffer(1)]],
#else
                            device ftype4 *out                 [[buffer(1)]],
#endif
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
#elif defined(FQ_MULTI_PROJ_OUTPUT)
                            device ftype4 *out1                [[buffer(7)]],
                        #ifdef W_QUANT_2
                            const device uchar4 *w1            [[buffer(8)]],
                        #elif defined(W_QUANT_3)
                            const device uchar *w1             [[buffer(8)]],
                        #elif defined(W_QUANT_4)
                            const device MNN::uchar4x2 *w1     [[buffer(8)]],
                        #else
                            const device MNN::char4x4 *w1      [[buffer(8)]],
                        #endif
                            const device ftype4 *b1            [[buffer(9)]],
                            const device ftype *s1             [[buffer(10)]],
                            device ftype4 *out2                [[buffer(11)]],
                        #ifdef W_QUANT_2
                            const device uchar4 *w2            [[buffer(12)]],
                        #elif defined(W_QUANT_3)
                            const device uchar *w2             [[buffer(12)]],
                        #elif defined(W_QUANT_4)
                            const device MNN::uchar4x2 *w2     [[buffer(12)]],
                        #else
                            const device MNN::char4x4 *w2      [[buffer(12)]],
                        #endif
                            const device ftype4 *b2            [[buffer(13)]],
                            const device ftype *s2             [[buffer(14)]],
                            device ftype4 *out3                [[buffer(15)]],
                        #ifdef W_QUANT_2
                            const device uchar4 *w3            [[buffer(16)]],
                        #elif defined(W_QUANT_3)
                            const device uchar *w3             [[buffer(16)]],
                        #elif defined(W_QUANT_4)
                            const device MNN::uchar4x2 *w3     [[buffer(16)]],
                        #else
                            const device MNN::char4x4 *w3      [[buffer(16)]],
                        #endif
                            const device ftype4 *b3            [[buffer(17)]],
                            const device ftype *s3             [[buffer(18)]],
                            constant FqMultiProjSegment *segments [[buffer(19)]],
#endif
                            uint3 gid                          [[threadgroup_position_in_grid]],
                            uint                  tiitg [[thread_index_in_threadgroup]],
                            uint                  tiisg [[thread_index_in_simdgroup]],
                            uint                  sgitg [[simdgroup_index_in_threadgroup]]) {
#ifdef USE_METAL_TENSOR_OPS
#ifdef FQ4_GATEUP_DUAL
    // The dual halves N and stages two weight tiles, so the operands are
    // unchanged and the store destination halves.
    const int K = FQ_TILE_K, M = FQ_TILE_M, N = 32, NT = 2;
#else
    const int K = FQ_TILE_K, M = FQ_TILE_M, N = FQ_TILE_N, NT = 1;
#endif
    // A store address is row * RS + kl in ftype4 units. Padding breaks 16-way
    // bank conflicts; RS must stay a multiple of 8 ftype for 16B alignment.
    const int RS = K + 8;
#ifdef FQ_DOUBLE_BUF
    const int SMEM_OPERAND4 = 2 * (M + NT * N) * RS * sizeof(ftype) / sizeof(FLOAT4);
#else
    const int SMEM_OPERAND4 = (M + NT * N) * RS * sizeof(ftype) / sizeof(FLOAT4);
#endif
    const int SMEM_DEST4    = M * N * sizeof(FLOAT) / sizeof(FLOAT4);
    threadgroup FLOAT4 sdata[SMEM_OPERAND4 > SMEM_DEST4 ? SMEM_OPERAND4 : SMEM_DEST4] = {0.f};

    const int RS4 = RS / 4;
    const int SB  = M * RS;
#ifdef FQ_DOUBLE_BUF
    const int SBUF  = (M + NT * N) * RS;
    const int SBUF4 = SBUF / 4;
#endif
    auto tI = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata,      dextents<int32_t, 2>(K, M), array<int, 2>{1, RS});
    auto tW = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata + SB, dextents<int32_t, 2>(K, N), array<int, 2>{1, RS});
#ifdef FQ4_GATEUP_DUAL
    auto tW2 = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata + SB + N * RS, dextents<int32_t, 2>(K, N), array<int, 2>{1, RS});
#endif

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(M, N, K, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mmOps;

    auto cT = mmOps.get_destination_cooperative_tensor<decltype(tI), decltype(tW), FLOAT>();
#ifdef FQ4_GATEUP_DUAL
    auto cT2 = mmOps.get_destination_cooperative_tensor<decltype(tI), decltype(tW2), FLOAT>();
#endif

    const int rx = gid.x;
    int uz = gid.y;
    const int area = cst.inputSize * cst.batch;
    const int outStride = cst.outputSize * cst.batch;
    int outputQuads = cst.outputDepthQuad;
#ifdef FQ_MULTI_PROJ_OUTPUT
    const int p = uz >= segments[3].firstTile ? 3 : uz >= segments[2].firstTile ? 2 :
                  uz >= segments[1].firstTile ? 1 : 0;
    uz -= segments[p].firstTile;
    outputQuads = segments[p].outputQuads;
    out = p == 0 ? out : p == 1 ? out1 : p == 2 ? out2 : out3;
#ifdef W_QUANT_2
    wt_int2 = p == 0 ? wt_int2 : p == 1 ? w1 : p == 2 ? w2 : w3;
#elif defined(W_QUANT_3)
    wt_int3 = p == 0 ? wt_int3 : p == 1 ? w1 : p == 2 ? w2 : w3;
#elif defined(W_QUANT_4)
    wt_int4 = p == 0 ? wt_int4 : p == 1 ? w1 : p == 2 ? w2 : w3;
#else
    wt_int8 = p == 0 ? wt_int8 : p == 1 ? w1 : p == 2 ? w2 : w3;
#endif
    biasTerms = p == 0 ? biasTerms : p == 1 ? b1 : p == 2 ? b2 : b3;
    dequantScale = p == 0 ? dequantScale : p == 1 ? s1 : p == 2 ? s2 : s3;
#endif

    const int nQuads = N / 4;
    const int quadsPerBlock = (cst.inputDepthQuad + cst.blockCount - 1) / cst.blockCount;

#if FQ_SPLIT_K_COUNT > 1
    const int ks = gid.z;
    const int nsub = (cst.blockCount + FQ_SPLIT_K_COUNT - 1) / FQ_SPLIT_K_COUNT;
    const int bi0  = ks * nsub;
    const int bi1  = min(bi0 + nsub, cst.blockCount);
    const int zpart_end = min(bi1 * quadsPerBlock, cst.inputDepthQuad);
    if (bi0 >= bi1) {
        for (int e = (int)tiitg; e < M * nQuads; e += 128) {
            int n4l = e / M;
            int m_idx = e % M;
            int gm = rx * M + m_idx;
            if (gm >= area) continue;
            int n4 = uz * nQuads + n4l;
            if (n4 >= outputQuads) continue;
            partialOut[((size_t)ks * cst.outputDepthQuad + n4) * outStride + gm] = float4(0.f);
        }
        return;
    }
#else
    const int bi0 = 0;
    const int bi1 = cst.blockCount;
    const int zpart_end = cst.inputDepthQuad;
#endif

    const int no  = tiitg / 8;
    const int k4p = tiitg % 8;
    const int mlc = tiitg / 4;
    const int nlc = tiitg % 4;

#ifdef FQ4_GATEUP_DUAL
    const int mtx = no / nQuads;
    const int non = no % nQuads;
    const bool bAct = true;
    const device ftype *dqs = (mtx == 0) ? dequantScale : dequantScale2;
    const FLOAT sCoef       = (mtx == 0) ? (FLOAT)cst.scaleCoef : (FLOAT)scaleCoef2;
#else
    const int mtx = 0;
    const int non = no;
    const bool bAct = (N == 64) ? true : (no < nQuads);
    const device ftype *dqs = dequantScale;
#ifdef FQ_MULTI_PROJ_OUTPUT
    const FLOAT sCoef       = (FLOAT)segments[p].scaleCoef;
#else
    const FLOAT sCoef       = (FLOAT)cst.scaleCoef;
#endif
#endif

    const int zStart = bi0 * quadsPerBlock;
#if FQ_TILE_M == 64
    const int kl = tiitg / 16;
    const int ml = tiitg % 16;
    int idx_m0 = (rx * 16 + ml) * 4 + 0 < area ? (rx * 16 + ml) * 4 + 0 : (area - 1);
    int idx_m1 = (rx * 16 + ml) * 4 + 1 < area ? (rx * 16 + ml) * 4 + 1 : (area - 1);
    int idx_m2 = (rx * 16 + ml) * 4 + 2 < area ? (rx * 16 + ml) * 4 + 2 : (area - 1);
    int idx_m3 = (rx * 16 + ml) * 4 + 3 < area ? (rx * 16 + ml) * 4 + 3 : (area - 1);
    auto xy_in0 = in + (zStart + kl) * area + idx_m0;
    auto xy_in1 = in + (zStart + kl) * area + idx_m1;
    auto xy_in2 = in + (zStart + kl) * area + idx_m2;
    auto xy_in3 = in + (zStart + kl) * area + idx_m3;
    const int idx_sa = (4 * ml) * RS4 + kl;
#elif FQ_TILE_M == 32
    const int kl = tiitg / 16;
    const int ml = tiitg % 16;
    int idx_m0 = (rx * 16 + ml) * 2 + 0 < area ? (rx * 16 + ml) * 2 + 0 : (area - 1);
    int idx_m1 = (rx * 16 + ml) * 2 + 1 < area ? (rx * 16 + ml) * 2 + 1 : (area - 1);
    auto xy_in0 = in + (zStart + kl) * area + idx_m0;
    auto xy_in1 = in + (zStart + kl) * area + idx_m1;
    const int idx_sa = (2 * ml) * RS4 + kl;
#elif FQ_TILE_M == 16
    const int kl = tiitg / 16;
    const int ml = tiitg % 16;
    int idx_m0 = (rx * 16 + ml) < area ? (rx * 16 + ml) : (area - 1);
    auto xy_in0 = in + (zStart + kl) * area + idx_m0;
    const int idx_sa = ml * RS4 + kl;
#else // FQ_TILE_M == 8
#if FQ_TILE_K == 64
    const int aml = tiitg / 16;
    const int kl  = tiitg % 16;
    const bool aAct = true;
    int idx_m0 = (rx * 8 + aml) < area ? (rx * 8 + aml) : (area - 1);
#else
    const int aml = tiitg / 8;
    const int kl  = tiitg % 8;
    const bool aAct = (aml < 8);
    int idx_m0 = (rx * 8 + min(aml, 7)) < area ? (rx * 8 + min(aml, 7)) : (area - 1);
#endif
    auto xy_in0 = in + (zStart + kl) * area + idx_m0;
    const int idx_sa = aml * RS4 + kl;
#endif

    const int safeNon = min(non, nQuads - 1);
    const int idx_n4 = (uz * nQuads + safeNon) < outputQuads ? (uz * nQuads + safeNon) : (outputQuads - 1);
#ifdef W_QUANT_2
#ifdef FQ4_GATEUP_DUAL
    auto wtBase = (mtx == 0) ? wt_int2 : wt2_int2;
#else
    auto wtBase = wt_int2;
#endif
#elif defined(W_QUANT_3)
#ifdef FQ4_GATEUP_DUAL
    auto wtBase = (mtx == 0) ? wt_int3 : wt2_int3;
#else
    auto wtBase = wt_int3;
#endif
#elif defined(W_QUANT_4)
#ifdef FQ4_GATEUP_DUAL
    auto wtBase = (mtx == 0) ? (const device uint2*)wt_int4 : (const device uint2*)wt2_int4;
#else
    auto wtBase = (const device uint2*)wt_int4;
#endif
#else // W_QUANT_8
#ifdef FQ4_GATEUP_DUAL
    auto wtBase = (mtx == 0) ? (const device uint4*)wt_int8 : (const device uint4*)wt2_int8;
#else
    auto wtBase = (const device uint4*)wt_int8;
#endif
#endif

#ifdef FQ_DOUBLE_BUF
    int it = 0;
#endif
#ifdef FQ_PREFETCH
#ifdef W_QUANT_4
    uint2 w2_pf = uint2(0u);
    if (bAct) {
        w2_pf = wtBase[idx_n4 * cst.inputDepthQuad + zStart + k4p];
    }
#elif defined(W_QUANT_8)
    uint4 w8_pf = uint4(0u);
    if (bAct) {
        w8_pf = wtBase[idx_n4 * cst.inputDepthQuad + zStart + k4p];
    }
#elif defined(W_QUANT_2)
    uchar4 w2b_pf = uchar4(0);
    if (bAct) {
        w2b_pf = wtBase[idx_n4 * cst.inputDepthQuad + zStart + k4p];
    }
#endif
#endif

    for (int bi = bi0; bi < bi1; ++bi) {
#if defined(W_QUANT_4) && defined(FQ4_HOIST_BLOCK_SCALE)
        FLOAT4 offset4 = FLOAT4(0), scaleHi4 = FLOAT4(0), scaleLo4 = FLOAT4(0);
        if (bAct) {
            const FLOAT4 sc4 = FLOAT4(*(const device ftype4*)(dqs + ((idx_n4 * cst.blockCount + bi) * 2 + 0) * 4)) / sCoef;
            const FLOAT4 db4 = FLOAT4(*(const device ftype4*)(dqs + ((idx_n4 * cst.blockCount + bi) * 2 + 1) * 4)) / sCoef;
            offset4  = db4 - 8.0 * sc4;
            scaleHi4 = sc4 * (FLOAT)(255.0 / 16.0);
            scaleLo4 = sc4 * (FLOAT)255.0;
        }
#elif !defined(W_QUANT_4)
        FLOAT4 sc4 = FLOAT4(0), db4 = FLOAT4(0);
        if (bAct) {
            sc4 = FLOAT4(*(const device ftype4*)(dqs + ((idx_n4 * cst.blockCount + bi) * 2 + 0) * 4)) / sCoef;
            db4 = FLOAT4(*(const device ftype4*)(dqs + ((idx_n4 * cst.blockCount + bi) * 2 + 1) * 4)) / sCoef;
        }
#ifdef W_QUANT_2
        const FLOAT4 offset4 = db4 - 2.0 * sc4;
#elif defined(W_QUANT_3)
        const FLOAT4 offset4 = db4 - 4.0 * sc4;
#endif
#endif
        int zmin = bi * quadsPerBlock;
        int zmax = min(zmin + quadsPerBlock, cst.inputDepthQuad);

        const int kStep = K / 4;
        for (int z = zmin; z < zmax; z += kStep) {
#ifdef FQ_DOUBLE_BUF
            const int bufF4 = (it & 1) * SBUF4;
#else
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const int bufF4 = 0;
#endif
            if (bAct) {
                const int bBase = bufF4 + (M + mtx * N) * RS4 + (non * 4) * RS4 + k4p;
#ifdef W_QUANT_4
#if FQ_TILE_K == 64
                uint2 w2a = wtBase[idx_n4 * cst.inputDepthQuad + z + k4p];
                uint2 w2b = wtBase[idx_n4 * cst.inputDepthQuad + z + 8 + k4p];
                #ifndef FQ4_HOIST_BLOCK_SCALE
                FLOAT4 sc4 = FLOAT4(*(const device ftype4*)(dqs + ((idx_n4 * cst.blockCount + bi) * 2 + 0) * 4)) / sCoef;
                FLOAT4 db4 = FLOAT4(*(const device ftype4*)(dqs + ((idx_n4 * cst.blockCount + bi) * 2 + 1) * 4)) / sCoef;
                #endif
                #pragma unroll(2)
                for (int kh = 0; kh < 2; ++kh) {
                    uint2 w2 = (kh == 0) ? w2a : w2b;
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
                    #ifdef FQ4_HOIST_BLOCK_SCALE
                        const FLOAT4 sv = FLOAT4(scaleHi4[j], scaleLo4[j], scaleHi4[j], scaleLo4[j]);
                        wd[j] = wd[j] * sv + FLOAT4(offset4[j]);
                    #else
                        FLOAT4 val = FLOAT4(db4[j] - 8.0 * sc4[j]);
                        FLOAT4 sv  = FLOAT4(sc4[j] * (FLOAT)(255.0 / 16.0), sc4[j] * (FLOAT)255.0,
                                            sc4[j] * (FLOAT)(255.0 / 16.0), sc4[j] * (FLOAT)255.0);
                        wd[j] = wd[j] * sv + val;
                    #endif
                        ((threadgroup ftype4*)sdata)[bBase + j * RS4 + kh * 8] = ftype4(wd[j]);
                    }
                }
#else
#ifdef FQ_PREFETCH
                uint2 w2 = w2_pf;
                if (z + 8 < zpart_end) {
                    w2_pf = wtBase[idx_n4 * cst.inputDepthQuad + (z + 8) + k4p];
                }
#else
#ifdef MNN_METAL_SRC_PROTECT
                uint2 w2 = wtBase[idx_n4 * cst.inputDepthQuad + min(z + k4p, cst.inputDepthQuad - 1)];
#else
                uint2 w2 = wtBase[idx_n4 * cst.inputDepthQuad + z + k4p];
#endif
#endif
                FLOAT4 hi0 = FLOAT4(unpack_unorm4x8_to_float(w2.x & 0xF0F0F0F0u));
                FLOAT4 lo0 = FLOAT4(unpack_unorm4x8_to_float(w2.x & 0x0F0F0F0Fu));
                FLOAT4 hi1 = FLOAT4(unpack_unorm4x8_to_float(w2.y & 0xF0F0F0F0u));
                FLOAT4 lo1 = FLOAT4(unpack_unorm4x8_to_float(w2.y & 0x0F0F0F0Fu));
                FLOAT4 wd[4];
                wd[0] = FLOAT4(hi0[0], lo0[0], hi0[1], lo0[1]);
                wd[1] = FLOAT4(hi0[2], lo0[2], hi0[3], lo0[3]);
                wd[2] = FLOAT4(hi1[0], lo1[0], hi1[1], lo1[1]);
                wd[3] = FLOAT4(hi1[2], lo1[2], hi1[3], lo1[3]);
#ifndef FQ4_HOIST_BLOCK_SCALE
                FLOAT4 sc4 = FLOAT4(*(const device ftype4*)(dqs + ((idx_n4 * cst.blockCount + bi) * 2 + 0) * 4)) / sCoef;
                FLOAT4 db4 = FLOAT4(*(const device ftype4*)(dqs + ((idx_n4 * cst.blockCount + bi) * 2 + 1) * 4)) / sCoef;
#endif
                #pragma unroll(4)
                for (int j = 0; j < 4; ++j) {
#ifdef FQ4_HOIST_BLOCK_SCALE
                    const FLOAT4 sv = FLOAT4(scaleHi4[j], scaleLo4[j], scaleHi4[j], scaleLo4[j]);
                    wd[j] = wd[j] * sv + FLOAT4(offset4[j]);
#else
                    FLOAT4 val = FLOAT4(db4[j] - 8.0 * sc4[j]);
                    FLOAT4 sv  = FLOAT4(sc4[j] * (FLOAT)(255.0 / 16.0), sc4[j] * (FLOAT)255.0,
                                        sc4[j] * (FLOAT)(255.0 / 16.0), sc4[j] * (FLOAT)255.0);
                    wd[j] = wd[j] * sv + val;
#endif
                    ((threadgroup ftype4*)sdata)[bBase + j * RS4] = ftype4(wd[j]);
                }
#endif // FQ_TILE_K == 64
#elif defined(W_QUANT_8)
#if FQ_TILE_K == 64
                uint4 w8a = wtBase[idx_n4 * cst.inputDepthQuad + z + k4p];
                uint4 w8b = wtBase[idx_n4 * cst.inputDepthQuad + z + 8 + k4p];
                #pragma unroll(2)
                for (int kh = 0; kh < 2; ++kh) {
                    uint4 w8 = (kh == 0) ? w8a : w8b;
                    char4 c0 = as_type<char4>(w8.x);
                    char4 c1 = as_type<char4>(w8.y);
                    char4 c2 = as_type<char4>(w8.z);
                    char4 c3 = as_type<char4>(w8.w);
                    ((threadgroup ftype4*)sdata)[bBase + 0 * RS4 + kh * 8] = ftype4(FLOAT4(FLOAT(c0.x), FLOAT(c0.y), FLOAT(c0.z), FLOAT(c0.w)) * sc4[0] + db4[0]);
                    ((threadgroup ftype4*)sdata)[bBase + 1 * RS4 + kh * 8] = ftype4(FLOAT4(FLOAT(c1.x), FLOAT(c1.y), FLOAT(c1.z), FLOAT(c1.w)) * sc4[1] + db4[1]);
                    ((threadgroup ftype4*)sdata)[bBase + 2 * RS4 + kh * 8] = ftype4(FLOAT4(FLOAT(c2.x), FLOAT(c2.y), FLOAT(c2.z), FLOAT(c2.w)) * sc4[2] + db4[2]);
                    ((threadgroup ftype4*)sdata)[bBase + 3 * RS4 + kh * 8] = ftype4(FLOAT4(FLOAT(c3.x), FLOAT(c3.y), FLOAT(c3.z), FLOAT(c3.w)) * sc4[3] + db4[3]);
                }
#else
#ifdef FQ_PREFETCH
                uint4 w8 = w8_pf;
                if (z + 8 < zpart_end) {
                    w8_pf = wtBase[idx_n4 * cst.inputDepthQuad + (z + 8) + k4p];
                }
#else
#ifdef MNN_METAL_SRC_PROTECT
                uint4 w8 = wtBase[idx_n4 * cst.inputDepthQuad + min(z + k4p, cst.inputDepthQuad - 1)];
#else
                uint4 w8 = wtBase[idx_n4 * cst.inputDepthQuad + z + k4p];
#endif
#endif
                char4 c0 = as_type<char4>(w8.x);
                char4 c1 = as_type<char4>(w8.y);
                char4 c2 = as_type<char4>(w8.z);
                char4 c3 = as_type<char4>(w8.w);
                ((threadgroup ftype4*)sdata)[bBase + 0 * RS4] = ftype4(FLOAT4(FLOAT(c0.x), FLOAT(c0.y), FLOAT(c0.z), FLOAT(c0.w)) * sc4[0] + db4[0]);
                ((threadgroup ftype4*)sdata)[bBase + 1 * RS4] = ftype4(FLOAT4(FLOAT(c1.x), FLOAT(c1.y), FLOAT(c1.z), FLOAT(c1.w)) * sc4[1] + db4[1]);
                ((threadgroup ftype4*)sdata)[bBase + 2 * RS4] = ftype4(FLOAT4(FLOAT(c2.x), FLOAT(c2.y), FLOAT(c2.z), FLOAT(c2.w)) * sc4[2] + db4[2]);
                ((threadgroup ftype4*)sdata)[bBase + 3 * RS4] = ftype4(FLOAT4(FLOAT(c3.x), FLOAT(c3.y), FLOAT(c3.z), FLOAT(c3.w)) * sc4[3] + db4[3]);
#endif
#elif defined(W_QUANT_2)
#if FQ_TILE_K == 64
                uchar4 w2a = wtBase[idx_n4 * cst.inputDepthQuad + z + k4p];
                uchar4 w2b = wtBase[idx_n4 * cst.inputDepthQuad + z + 8 + k4p];
                #pragma unroll(2)
                for (int kh = 0; kh < 2; ++kh) {
                    uchar4 w2c = (kh == 0) ? w2a : w2b;
                    #pragma unroll(4)
                    for (int j = 0; j < 4; ++j) {
                        uchar b = w2c[j];
                        FLOAT4 row = FLOAT4(FLOAT((b >> 6) & 3), FLOAT((b >> 4) & 3), FLOAT((b >> 2) & 3), FLOAT(b & 3));
                        ((threadgroup ftype4*)sdata)[bBase + j * RS4 + kh * 8] = ftype4(row * sc4[j] + offset4[j]);
                    }
                }
#else
#ifdef FQ_PREFETCH
                uchar4 w2b = w2b_pf;
                if (z + 8 < zpart_end) {
                    w2b_pf = wtBase[idx_n4 * cst.inputDepthQuad + (z + 8) + k4p];
                }
#else
#ifdef MNN_METAL_SRC_PROTECT
                uchar4 w2b = wtBase[idx_n4 * cst.inputDepthQuad + min(z + k4p, cst.inputDepthQuad - 1)];
#else
                uchar4 w2b = wtBase[idx_n4 * cst.inputDepthQuad + z + k4p];
#endif
#endif
                #pragma unroll(4)
                for (int j = 0; j < 4; ++j) {
                    uchar b = w2b[j];
                    FLOAT4 row = FLOAT4(FLOAT((b >> 6) & 3), FLOAT((b >> 4) & 3), FLOAT((b >> 2) & 3), FLOAT(b & 3));
                    ((threadgroup ftype4*)sdata)[bBase + j * RS4] = ftype4(row * sc4[j] + offset4[j]);
                }
#endif
#else // W_QUANT_3
#if FQ_TILE_K == 64
                #pragma unroll(2)
                for (int kh = 0; kh < 2; ++kh) {
                    const device ushort *tp = (const device ushort *)(wtBase + (idx_n4 * cst.inputDepthQuad + z + kh * 8 + k4p) * 6);
                    ushort t0 = tp[0], t1 = tp[1], t2 = tp[2];
                    uchar los[4] = { uchar(t0 & 0xFF), uchar(t0 >> 8), uchar(t1 & 0xFF), uchar(t1 >> 8) };
                    uchar hs[4]  = { uchar((t2 >> 4) & 0xF), uchar(t2 & 0xF), uchar((t2 >> 12) & 0xF), uchar((t2 >> 8) & 0xF) };
                    #pragma unroll(4)
                    for (int j = 0; j < 4; ++j) {
                        uchar lo = los[j], h = hs[j];
                        FLOAT4 row = FLOAT4(
                            FLOAT(((lo >> 6) & 3) | (((h >> 3) & 1) << 2)),
                            FLOAT(((lo >> 4) & 3) | (((h >> 2) & 1) << 2)),
                            FLOAT(((lo >> 2) & 3) | (((h >> 1) & 1) << 2)),
                            FLOAT(( lo       & 3) | (( h       & 1) << 2)));
                        ((threadgroup ftype4*)sdata)[bBase + j * RS4 + kh * 8] = ftype4(row * sc4[j] + offset4[j]);
                    }
                }
#else
#ifdef MNN_METAL_SRC_PROTECT
                int kz = min(z + k4p, cst.inputDepthQuad - 1);
#else
                int kz = z + k4p;
#endif
                const device ushort *tp = (const device ushort *)(wtBase + (idx_n4 * cst.inputDepthQuad + kz) * 6);
                ushort t0 = tp[0], t1 = tp[1], t2 = tp[2];
                uchar los[4] = { uchar(t0 & 0xFF), uchar(t0 >> 8), uchar(t1 & 0xFF), uchar(t1 >> 8) };
                uchar hs[4]  = { uchar((t2 >> 4) & 0xF), uchar(t2 & 0xF), uchar((t2 >> 12) & 0xF), uchar((t2 >> 8) & 0xF) };
                #pragma unroll(4)
                for (int j = 0; j < 4; ++j) {
                    uchar lo = los[j], h = hs[j];
                    FLOAT4 row = FLOAT4(
                        FLOAT(((lo >> 6) & 3) | (((h >> 3) & 1) << 2)),
                        FLOAT(((lo >> 4) & 3) | (((h >> 2) & 1) << 2)),
                        FLOAT(((lo >> 2) & 3) | (((h >> 1) & 1) << 2)),
                        FLOAT(( lo       & 3) | (( h       & 1) << 2)));
                    ((threadgroup ftype4*)sdata)[bBase + j * RS4] = ftype4(row * sc4[j] + offset4[j]);
                }
#endif
#endif
            }

            // Input load.
#if FQ_TILE_M == 64
            #ifdef MNN_METAL_SRC_PROTECT
            if (kl + z < cst.inputDepthQuad) {
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
#elif FQ_TILE_M == 32
            #ifdef MNN_METAL_SRC_PROTECT
            if (kl + z < cst.inputDepthQuad) {
                ((threadgroup ftype4*)sdata)[bufF4 + idx_sa]       = (ftype4)*(xy_in0);
                ((threadgroup ftype4*)sdata)[bufF4 + idx_sa + RS4] = (ftype4)*(xy_in1);
            } else {
                ((threadgroup ftype4*)sdata)[bufF4 + idx_sa]       = (ftype4)(0);
                ((threadgroup ftype4*)sdata)[bufF4 + idx_sa + RS4] = (ftype4)(0);
            }
            #else
            ((threadgroup ftype4*)sdata)[bufF4 + idx_sa]       = (ftype4)*(xy_in0);
            ((threadgroup ftype4*)sdata)[bufF4 + idx_sa + RS4] = (ftype4)*(xy_in1);
            #endif
#elif FQ_TILE_M == 16
            #ifdef MNN_METAL_SRC_PROTECT
            if (kl + z < cst.inputDepthQuad) {
                ((threadgroup ftype4*)sdata)[bufF4 + idx_sa] = (ftype4)*(xy_in0);
            } else {
                ((threadgroup ftype4*)sdata)[bufF4 + idx_sa] = (ftype4)(0);
            }
            #else
            ((threadgroup ftype4*)sdata)[bufF4 + idx_sa] = (ftype4)*(xy_in0);
            #endif
#else // FQ_TILE_M == 8
            if (aAct) {
                #ifdef MNN_METAL_SRC_PROTECT
                if (kl + z < cst.inputDepthQuad) {
                    ((threadgroup ftype4*)sdata)[bufF4 + idx_sa] = (ftype4)*(xy_in0);
                } else {
                    ((threadgroup ftype4*)sdata)[bufF4 + idx_sa] = (ftype4)(0);
                }
                #else
                ((threadgroup ftype4*)sdata)[bufF4 + idx_sa] = (ftype4)*(xy_in0);
                #endif
            }
#endif

#ifdef FQ_DOUBLE_BUF
            xy_in0 += kStep * area;
#if FQ_TILE_M >= 32
            xy_in1 += kStep * area;
#endif
#if FQ_TILE_M == 64
            xy_in2 += kStep * area;
            xy_in3 += kStep * area;
#endif
            ++it;
#endif

            threadgroup_barrier(mem_flags::mem_threadgroup);

#ifdef FQ_DOUBLE_BUF
            const int bufF = bufF4 * 4;
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
#ifdef FQ_DOUBLE_BUF
            auto tW2b = tensor<threadgroup ftype, dextents<int32_t, 2>, tensor_inline>((threadgroup ftype*)sdata + bufF + SB + N * RS, dextents<int32_t, 2>(K, N), array<int, 2>{1, RS});
            auto sB2 = tW2b.slice(0, 0);
#else
            auto sB2 = tW2.slice(0, 0);
#endif
            mmOps.run(sA, sB2, cT2);
#endif

#ifndef FQ_DOUBLE_BUF
            xy_in0 += kStep * area;
#if FQ_TILE_M >= 32
            xy_in1 += kStep * area;
#endif
#if FQ_TILE_M == 64
            xy_in2 += kStep * area;
            xy_in3 += kStep * area;
#endif
#endif
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto tC = tensor<threadgroup FLOAT, dextents<int32_t, 2>, tensor_inline>((threadgroup FLOAT*)sdata, dextents<int32_t, 2>(N, M));

#ifdef FQ4_GATEUP_DUAL
    cT2.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);

#if FQ_TILE_M == 64
    FLOAT4 gv[4];
    #pragma unroll(2)
    for (int mm = 0; mm < 2; ++mm) {
        int m_idx = mm * 32 + mlc;
        #pragma unroll(2)
        for (int j = 0; j < 2; ++j) {
            int n4 = min(uz * nQuads + nlc * 2 + j, outputQuads - 1);
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
        if ((rx * 64 + m_idx) < area) {
            #pragma unroll(2)
            for (int j = 0; j < 2; ++j) {
                if (uz * nQuads + nlc * 2 + j < outputQuads) {
                    FLOAT4 u = FLOAT4(activate(ftype4(((threadgroup FLOAT4*)sdata)[m_idx * nQuads + nlc * 2 + j] +
                                                      FLOAT4(biasTerms[uz * nQuads + nlc * 2 + j])), cst.activation));
                    FLOAT4 g = gv[mm * 2 + j];
                    xy_out[j * outStride] = ftype4(u * (g / (FLOAT(1.0) + exp(-g))));
                }
            }
        }
    }
#else
    constexpr int DUAL_ITERS = (FQ_TILE_M * 8 + 127) / 128;
    FLOAT4 gv[DUAL_ITERS];
    for (int k = 0; k < DUAL_ITERS; ++k) {
        int e = k * 128 + (int)tiitg;
        if (e < M * nQuads) {
            int n4l = e / M;
            int m_idx = e % M;
            int n4 = min(uz * nQuads + n4l, outputQuads - 1);
            gv[k] = ((threadgroup FLOAT4*)sdata)[m_idx * nQuads + n4l] + FLOAT4(biasTerms2[n4]);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    cT.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 0; k < DUAL_ITERS; ++k) {
        int e = k * 128 + (int)tiitg;
        if (e < M * nQuads) {
            int n4l = e / M;
            int m_idx = e % M;
            int gm = rx * M + m_idx;
            int n4 = uz * nQuads + n4l;
            if (gm < area && n4 < outputQuads) {
                FLOAT4 u = FLOAT4(activate(ftype4(((threadgroup FLOAT4*)sdata)[m_idx * nQuads + n4l] +
                                                  FLOAT4(biasTerms[n4])), cst.activation));
                FLOAT4 g = gv[k];
                out[(long)n4 * outStride + gm] = ftype4(u * (g / (FLOAT(1.0) + exp(-g))));
            }
        }
    }
#endif
#else
    cT.store(tC);

    threadgroup_barrier(mem_flags::mem_threadgroup);

#if FQ_TILE_M == 64 && FQ_TILE_N == 64 && FQ_SPLIT_K_COUNT == 1
    for (int mm = 0; mm < 2; ++mm) {
        int m_idx = mm * 32 + mlc;
        auto xy_out = out + ((uz * 4 + nlc) * 4 + 0) * outStride + (rx * 64 + m_idx);
#ifdef FQ4_SILU_MUL
        auto xy_gate = gateIn + ((uz * 4 + nlc) * 4 + 0) * outStride + (rx * 64 + m_idx);
#endif
        if ((rx * 64 + m_idx) < area) {
            #pragma unroll(4)
            for (int j = 0; j < 4; ++j) {
                if ((uz * 4 + nlc) * 4 + j < outputQuads) {
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
#else
    for (int e = (int)tiitg; e < M * nQuads; e += 128) {
        int n4l = e / M;
        int m_idx = e % M;
        int gm = rx * M + m_idx;
        if (gm >= area) continue;
        int n4 = uz * nQuads + n4l;
        if (n4 >= outputQuads) continue;
        FLOAT4 v = ((threadgroup FLOAT4*)sdata)[m_idx * nQuads + n4l];
#if FQ_SPLIT_K_COUNT > 1
        partialOut[((size_t)ks * cst.outputDepthQuad + n4) * outStride + gm] = float4(v);
#else
        ftype4 r = activate(ftype4(v + FLOAT4(biasTerms[n4])), cst.activation);
#ifdef FQ4_SILU_MUL
        FLOAT4 g = FLOAT4(gateIn[n4 * outStride + gm]);
        r = ftype4(FLOAT4(r) * (g / (FLOAT(1.0) + exp(-g))));
#endif
        out[n4 * outStride + gm] = r;
#endif
    }
#endif
#endif // FQ4_GATEUP_DUAL
#endif // USE_METAL_TENSOR_OPS
}
#endif // W_QUANT_2 || W_QUANT_3 || W_QUANT_4 || W_QUANT_8

#ifdef USE_METAL_TENSOR_OPS
kernel void conv1x1_pack_input_tensor(const device ftype4* input [[buffer(0)]],
                                     device ftype4* output [[buffer(1)]],
                                     constant conv1x1_constants& cst [[buffer(2)]],
                                     uint gid [[thread_position_in_grid]]) {
    const int area = cst.inputSize * cst.batch;
    if (gid >= area * cst.inputDepthQuad) return;
    const int row = gid / cst.inputDepthQuad;
    const int k4 = gid % cst.inputDepthQuad;
    output[gid] = input[k4 * area + row];
}

kernel void conv1x1_w_dequant_row(
#ifdef W_QUANT_8
                                 const device char4* weight [[buffer(0)]],
#else
                                 const device uchar2* weight [[buffer(0)]],
#endif
                                 device ftype4* output [[buffer(1)]],
                                 constant conv1x1_constants& cst [[buffer(2)]],
                                 const device ftype4* scales [[buffer(3)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    const int n = gid.x;
    const int k4 = gid.y * 4;
    if (n >= cst.outputDepthQuad * 4 || k4 >= cst.inputDepthQuad) return;
    const int quadsPerBlock = (cst.inputDepthQuad + cst.blockCount - 1) / cst.blockCount;
    const int block = k4 / quadsPerBlock;
    const int scaleOffset = (n / 4 * cst.blockCount + block) * 2;
    const float scale = float(scales[scaleOffset][n % 4]) / cst.scaleCoef;
    const float bias = float(scales[scaleOffset + 1][n % 4]) / cst.scaleCoef;
    for (int i = 0; i < 4 && k4 + i < cst.inputDepthQuad; ++i) {
#ifdef W_QUANT_8
        char4 v = weight[(n / 4 * cst.inputDepthQuad + k4 + i) * 4 + n % 4];
        float4 q = float4(v);
#else
        uchar2 v = weight[(n / 4 * cst.inputDepthQuad + k4 + i) * 4 + n % 4];
        float4 q = float4(v.x >> 4, v.x & 15, v.y >> 4, v.y & 15) - 8.0f;
#endif
        output[n * cst.inputDepthQuad + k4 + i] = ftype4(q * scale + bias);
    }
}

// Contiguous device operands let matmul2d schedule the full K reduction without per-tile unpacking.
kernel void conv1x1_gemm_device_tensor(const device ftype* input [[buffer(0)]],
                                      device ftype4* output [[buffer(1)]],
                                      constant conv1x1_constants& cst [[buffer(2)]],
                                      device ftype* weight [[buffer(3)]],
                                      const device ftype4* bias [[buffer(4)]],
                                      uint2 gid [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]]) {
    const int area = cst.inputSize * cst.batch;
    const int ic = cst.inputDepthQuad * 4;
    const int oc = cst.outputDepthQuad * 4;
    auto a = tensor<device ftype, dextents<int32_t, 2>, tensor_inline>(
        (device ftype*)input, dextents<int32_t, 2>(ic, area));
    auto b = tensor<device ftype, dextents<int32_t, 2>, tensor_inline>(
        weight, dextents<int32_t, 2>(ic, oc));
    auto aa = a.slice(0, gid.x * 64);
    auto bb = b.slice(0, gid.y * 64);
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(64, 64, dynamic_extent, false, true, false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
    auto c = mm.get_destination_cooperative_tensor<decltype(aa), decltype(bb), float>();
    mm.run(aa, bb, c);
    threadgroup float4 data[1024];
    auto dest = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(
        (threadgroup float*)data, dextents<int32_t, 2>(64, 64));
    c.store(dest);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int x = tid; x < 1024; x += 128) {
        const int row = gid.x * 64 + x / 16;
        const int n4 = gid.y * 16 + x % 16;
        if (row < area && n4 < cst.outputDepthQuad) {
            output[n4 * area + row] = activate(ftype4(data[x] + float4(bias[n4])), cst.activation);
        }
    }
}
#endif

//======================================================================
// conv1x1_gemm_64x64_split_k_sg — sg_matrix M=64 tile (fp16 weights,
// outer-dequant path, non-tensor-API devices).
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
        #ifdef MNN_METAL_SRC_PROTECT
        ftype4x4 w_local = (z + kwl * 4 < cst.inputDepthQuad)
            ? xy_wt[z] : ftype4x4(ftype4(0), ftype4(0), ftype4(0), ftype4(0));
        #else
        ftype4x4 w_local = xy_wt[z];
        #endif

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
                        #ifdef CONV_ROW_MUL
                            const device ftype4 *rowScale    [[buffer(24)]],
                        #endif
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

    ftype4 value = activate(ftype4(result0 + biasValue), cst.activation);
    CONV_ROW_SCALE(value, (int)gid.z * cst.outputSize + rx);
    *xy_out = value;
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
                        #ifdef GATE_UP_FUSED
                            device ftype4 *out_up               [[buffer(6)]],
                        #ifdef W_QUANT_2
                            const device uchar4 *wt_up          [[buffer(7)]],
                        #elif defined(W_QUANT_3)
                            const device uchar *wt_up           [[buffer(7)]],
                        #elif defined(W_QUANT_4)
                            const device ushort4 *wt_up         [[buffer(7)]],
                        #elif defined(W_QUANT_8)
                            const device MNN::char4x4 *wt_up    [[buffer(7)]],
                        #endif
                            const device ftype4 *biasTerms_up   [[buffer(8)]],
                            const device ftype4 *dequantScale_up [[buffer(9)]],
                            constant float *gate_up_seg         [[buffer(14)]],
                        #endif
                        #ifdef QKV_FUSED
                            device ftype4 *out_k                [[buffer(6)]],
                        #ifdef W_QUANT_2
                            const device uchar4 *wt_k           [[buffer(7)]],
                        #elif defined(W_QUANT_3)
                            const device uchar *wt_k            [[buffer(7)]],
                        #elif defined(W_QUANT_4)
                            const device ushort4 *wt_k          [[buffer(7)]],
                        #elif defined(W_QUANT_8)
                            const device MNN::char4x4 *wt_k     [[buffer(7)]],
                        #endif
                            const device ftype4 *biasTerms_k    [[buffer(8)]],
                            const device ftype4 *dequantScale_k [[buffer(9)]],
                            device ftype4 *out_v                [[buffer(10)]],
                        #ifdef W_QUANT_2
                            const device uchar4 *wt_v           [[buffer(11)]],
                        #elif defined(W_QUANT_3)
                            const device uchar *wt_v            [[buffer(11)]],
                        #elif defined(W_QUANT_4)
                            const device ushort4 *wt_v          [[buffer(11)]],
                        #elif defined(W_QUANT_8)
                            const device MNN::char4x4 *wt_v     [[buffer(11)]],
                        #endif
                            const device ftype4 *biasTerms_v    [[buffer(12)]],
                            const device ftype4 *dequantScale_v [[buffer(13)]],
                            constant float *qkv_seg             [[buffer(14)]],
                        #ifdef QKV_FUSED_4PROJ
                            device ftype4 *out_w                [[buffer(15)]],
                        #ifdef W_QUANT_2
                            const device uchar4 *wt_w           [[buffer(16)]],
                        #elif defined(W_QUANT_3)
                            const device uchar *wt_w            [[buffer(16)]],
                        #elif defined(W_QUANT_4)
                            const device ushort4 *wt_w          [[buffer(16)]],
                        #elif defined(W_QUANT_8)
                            const device MNN::char4x4 *wt_w     [[buffer(16)]],
                        #endif
                            const device ftype4 *biasTerms_w    [[buffer(17)]],
                            const device ftype4 *dequantScale_w [[buffer(18)]],
                        #endif
                        #endif
                        #ifdef CONV_ROW_MUL
                            const device ftype4 *rowScale       [[buffer(24)]],
                        #endif
                            uint3 gid[[threadgroup_position_in_grid]],
                            uint  tiisg[[thread_index_in_simdgroup]],
                            uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    int outputDepthQuad = cst.outputDepthQuad;
    float cur_scale_coef = cst.scaleCoef;
#if defined(GATE_UP_FUSED) && !defined(GATE_UP_SILU)
    if (gid.z == 1) {
        out = out_up;
        wt = wt_up;
        biasTerms = biasTerms_up;
        dequantScale = dequantScale_up;
        cur_scale_coef = gate_up_seg[0];
    }
    const int tg_x = int(gid.x);
#elif defined(QKV_FUSED)
#ifdef QKV_PACKED_GRID
    int qkv_proj = 0;
    if (int(gid.x) >= int(qkv_seg[6])) qkv_proj = 1;
    if (int(gid.x) >= int(qkv_seg[7])) qkv_proj = 2;
#ifdef QKV_FUSED_4PROJ
    if (int(gid.x) >= int(qkv_seg[8])) qkv_proj = 3;
#endif
    const int tg_x = int(gid.x) - (qkv_proj == 0 ? 0 : int(qkv_seg[5 + qkv_proj]));
#else
    const int qkv_proj = int(gid.z);
    const int tg_x = int(gid.x);
#endif
    if (qkv_proj == 1) {
        out = out_k;
        biasTerms = biasTerms_k;
        wt = wt_k;
        dequantScale = dequantScale_k;
        cur_scale_coef = qkv_seg[0];
        outputDepthQuad = int(qkv_seg[2]);
    } else if (qkv_proj == 2) {
        out = out_v;
        biasTerms = biasTerms_v;
        wt = wt_v;
        dequantScale = dequantScale_v;
        cur_scale_coef = qkv_seg[1];
        outputDepthQuad = int(qkv_seg[3]);
    }
#ifdef QKV_FUSED_4PROJ
    else if (qkv_proj == 3) {
        out = out_w;
        biasTerms = biasTerms_w;
        wt = wt_w;
        dequantScale = dequantScale_w;
        cur_scale_coef = qkv_seg[4];
        outputDepthQuad = int(qkv_seg[5]);
    }
#endif
#else
    const int tg_x = int(gid.x);
#endif

    // 2 simdgroups per threadgroup (64 threads), each owns one outputDepthQuad (4 OC).
    int uz = tg_x * 2 + int(sgitg);
    int rx = int(gid.y) * AREA_THREAD;
    auto area_size = cst.outputSize * cst.batch;
    if (uz >= outputDepthQuad || rx >= area_size) {
        return;
    }

#ifdef MNN_METAL_SRC_PROTECT
#define G4MX_LOAD_IN4(z, i) ((rx + (int)(i)) < area_size ? (FLOAT4)*(xy_in0 + (long)(z) * area_size + (i)) : (FLOAT4)0)
#else
#define G4MX_LOAD_IN4(z, i) ((FLOAT4)*(xy_in0 + (long)(z) * area_size + (i)))
#endif
#ifdef W_QUANT_3
    const long wt_row_stride = (long)cst.inputDepthQuad * 6;
#else
    const long wt_row_stride = (long)cst.inputDepthQuad;
#endif
    auto xy_in0  = in + rx;
    auto xy_out = out + (long)uz * area_size + rx;
    int quadsPerBlock = (cst.inputDepthQuad + cst.blockCount - 1) / cst.blockCount;

    int lanesPerBlock = min(SIMD_GROUP_WIDTH, max(quadsPerBlock / 4, 1));
    int blocksInFlight = SIMD_GROUP_WIDTH / lanesPerBlock;
    int laneInBlock = (tiisg) % lanesPerBlock;
    int blockSlot = (tiisg) / lanesPerBlock;
    // Lanes left over by a non-power-of-two lanesPerBlock would repeat a block.
    if (blockSlot >= blocksInFlight) {
        blockSlot = cst.blockCount;
    }

#ifdef GATE_UP_SILU
    auto xy_wt0 = wt + (long)uz * wt_row_stride;
    auto xy_wt1 = wt_up + (long)uz * wt_row_stride;
    auto biasValue0 = FLOAT4(biasTerms[uz]);
    auto biasValue1 = FLOAT4(biasTerms_up[uz]);
    const FLOAT inv_coef0 = FLOAT(1.0) / (FLOAT)cst.scaleCoef;
    const FLOAT inv_coef1 = FLOAT(1.0) / (FLOAT)gate_up_seg[0];
    FLOAT4 result0[AREA_THREAD] = {FLOAT4(0)};
    FLOAT4 result1[AREA_THREAD] = {FLOAT4(0)};

    for (int bi = blockSlot; bi < cst.blockCount; bi += blocksInFlight) {
        FLOAT4 scale0 = FLOAT4(dequantScale[2L * ((long)uz * cst.blockCount + bi) + 0]) * inv_coef0;
        FLOAT4 dbias0 = FLOAT4(dequantScale[2L * ((long)uz * cst.blockCount + bi) + 1]) * inv_coef0;
        FLOAT4 scale1 = FLOAT4(dequantScale_up[2L * ((long)uz * cst.blockCount + bi) + 0]) * inv_coef1;
        FLOAT4 dbias1 = FLOAT4(dequantScale_up[2L * ((long)uz * cst.blockCount + bi) + 1]) * inv_coef1;
        int zmin = bi * quadsPerBlock;
        int zmax = min(zmin + quadsPerBlock, cst.inputDepthQuad);

        FLOAT4 raw_dot0[AREA_THREAD] = {FLOAT4(0)};
        FLOAT4 raw_dot1[AREA_THREAD] = {FLOAT4(0)};
        FLOAT input_sum[AREA_THREAD] = {FLOAT(0)};
    #if defined(W_QUANT_2) || defined(W_QUANT_3)
        for (int z = zmin + laneInBlock; z < zmax; z += lanesPerBlock) {
            auto base_xy = xy_in0 + (long)z * area_size;
            (void)base_xy;
        #ifdef W_QUANT_2
            uchar4 wA = xy_wt0[z];
            uchar4 wB = xy_wt1[z];
            for (int i = 0; i < AREA_THREAD; i++) {
                FLOAT4 in40 = G4MX_LOAD_IN4(z, i);
                input_sum[i] += dot(in40, FLOAT4(1.0));
                FLOAT in_ps0 = in40[0] * FLOAT(1.0/64.0);
                FLOAT in_ps1 = in40[1] * FLOAT(1.0/16.0);
                FLOAT in_ps2 = in40[2] * FLOAT(1.0/4.0);
                raw_dot0[i] += in_ps0 * FLOAT4(wA & uchar4(0xC0)) + in_ps1 * FLOAT4(wA & uchar4(0x30))
                             + in_ps2 * FLOAT4(wA & uchar4(0x0C)) + in40[3] * FLOAT4(wA & uchar4(0x03));
                raw_dot1[i] += in_ps0 * FLOAT4(wB & uchar4(0xC0)) + in_ps1 * FLOAT4(wB & uchar4(0x30))
                             + in_ps2 * FLOAT4(wB & uchar4(0x0C)) + in40[3] * FLOAT4(wB & uchar4(0x03));
            }
        #else // W_QUANT_3
            const device ushort* tA = (const device ushort*)(xy_wt0 + z * 6);
            const device ushort* tB = (const device ushort*)(xy_wt1 + z * 6);
            ushort sa0 = tA[0], sa1 = tA[1], sa2 = tA[2];
            ushort sb0 = tB[0], sb1 = tB[1], sb2 = tB[2];
            uchar4 loA = uchar4(uchar(sa0), uchar(sa0 >> 8), uchar(sa1), uchar(sa1 >> 8));
            uchar hab0 = uchar(sa2), hab1 = uchar(sa2 >> 8);
            uchar4 hiA = uchar4(hab0 >> 4, hab0 & 0xF, hab1 >> 4, hab1 & 0xF);
            uchar4 loB = uchar4(uchar(sb0), uchar(sb0 >> 8), uchar(sb1), uchar(sb1 >> 8));
            uchar hbb0 = uchar(sb2), hbb1 = uchar(sb2 >> 8);
            uchar4 hiB = uchar4(hbb0 >> 4, hbb0 & 0xF, hbb1 >> 4, hbb1 & 0xF);
            for (int i = 0; i < AREA_THREAD; i++) {
                FLOAT4 in40 = G4MX_LOAD_IN4(z, i);
                input_sum[i] += dot(in40, FLOAT4(1.0));
                FLOAT in_ps0 = in40[0] * FLOAT(1.0/64.0);
                FLOAT in_ps1 = in40[1] * FLOAT(1.0/16.0);
                FLOAT in_ps2 = in40[2] * FLOAT(1.0/4.0);
                FLOAT in_hi0 = in40[0] * FLOAT(0.5);
                FLOAT in_hi2 = in40[2] * FLOAT(2.0);
                FLOAT in_hi3 = in40[3] * FLOAT(4.0);
                raw_dot0[i] += in_ps0 * FLOAT4(loA & uchar4(0xC0)) + in_ps1 * FLOAT4(loA & uchar4(0x30))
                             + in_ps2 * FLOAT4(loA & uchar4(0x0C)) + in40[3] * FLOAT4(loA & uchar4(0x03))
                             + in_hi0 * FLOAT4(hiA & uchar4(0x8)) + in40[1] * FLOAT4(hiA & uchar4(0x4))
                             + in_hi2 * FLOAT4(hiA & uchar4(0x2)) + in_hi3 * FLOAT4(hiA & uchar4(0x1));
                raw_dot1[i] += in_ps0 * FLOAT4(loB & uchar4(0xC0)) + in_ps1 * FLOAT4(loB & uchar4(0x30))
                             + in_ps2 * FLOAT4(loB & uchar4(0x0C)) + in40[3] * FLOAT4(loB & uchar4(0x03))
                             + in_hi0 * FLOAT4(hiB & uchar4(0x8)) + in40[1] * FLOAT4(hiB & uchar4(0x4))
                             + in_hi2 * FLOAT4(hiB & uchar4(0x2)) + in_hi3 * FLOAT4(hiB & uchar4(0x1));
            }
        #endif
        }
        #ifdef W_QUANT_2
        FLOAT4 adj0 = dbias0 - FLOAT(2.0) * scale0;
        FLOAT4 adj1 = dbias1 - FLOAT(2.0) * scale1;
        #else
        FLOAT4 adj0 = dbias0 - FLOAT(4.0) * scale0;
        FLOAT4 adj1 = dbias1 - FLOAT(4.0) * scale1;
        #endif
        for (int i = 0; i < AREA_THREAD; i++) {
            result0[i] += raw_dot0[i] * scale0 + input_sum[i] * adj0;
            result1[i] += raw_dot1[i] * scale1 + input_sum[i] * adj1;
        }
    #elif defined(W_QUANT_4)
        for (int z = zmin + laneInBlock; z < zmax; z += lanesPerBlock) {
            ushort4 wA = xy_wt0[z];
            ushort4 wB = xy_wt1[z];
            auto base_xy = xy_in0 + (long)z * area_size;
            (void)base_xy;
            for (int i = 0; i < AREA_THREAD; i++) {
                FLOAT4 in40 = G4MX_LOAD_IN4(z, i);
                input_sum[i] += dot(in40, FLOAT4(1.0));
                FLOAT s0 = in40[0] * FLOAT(0.0625);
                FLOAT s2 = in40[2] * FLOAT(0.000244140625);
                FLOAT s3 = in40[3] * FLOAT(0.00390625);
                raw_dot0[i] += s0 * FLOAT4(wA & ushort4(0x00F0u)) + in40[1] * FLOAT4(wA & ushort4(0x000Fu))
                             + s2 * FLOAT4(wA & ushort4(0xF000u)) + s3 * FLOAT4(wA & ushort4(0x0F00u));
                raw_dot1[i] += s0 * FLOAT4(wB & ushort4(0x00F0u)) + in40[1] * FLOAT4(wB & ushort4(0x000Fu))
                             + s2 * FLOAT4(wB & ushort4(0xF000u)) + s3 * FLOAT4(wB & ushort4(0x0F00u));
            }
        }
        FLOAT4 adj0 = dbias0 - FLOAT(8.0) * scale0;
        FLOAT4 adj1 = dbias1 - FLOAT(8.0) * scale1;
        for (int i = 0; i < AREA_THREAD; i++) {
            result0[i] += raw_dot0[i] * scale0 + input_sum[i] * adj0;
            result1[i] += raw_dot1[i] * scale1 + input_sum[i] * adj1;
        }
    #elif defined(W_QUANT_8)
        for (int z = zmin + laneInBlock; z < zmax; z += lanesPerBlock) {
            auto wA = xy_wt0[z];
            auto wB = xy_wt1[z];
            auto base_xy = xy_in0 + (long)z * area_size;
            (void)base_xy;
            for (int i = 0; i < AREA_THREAD; i++) {
                FLOAT4 in40 = G4MX_LOAD_IN4(z, i);
                input_sum[i] += dot(in40, FLOAT4(1.0));
                raw_dot0[i] += FLOAT4(dot(in40, FLOAT4(wA[0])), dot(in40, FLOAT4(wA[1])),
                                      dot(in40, FLOAT4(wA[2])), dot(in40, FLOAT4(wA[3])));
                raw_dot1[i] += FLOAT4(dot(in40, FLOAT4(wB[0])), dot(in40, FLOAT4(wB[1])),
                                      dot(in40, FLOAT4(wB[2])), dot(in40, FLOAT4(wB[3])));
            }
        }
        for (int i = 0; i < AREA_THREAD; i++) {
            result0[i] += raw_dot0[i] * scale0 + input_sum[i] * dbias0;
            result1[i] += raw_dot1[i] * scale1 + input_sum[i] * dbias1;
        }
    #endif
    }

    for (int i = 0; i < AREA_THREAD; i++) {
        result0[i] = simd_sum(result0[i]);
        result1[i] = simd_sum(result1[i]);
    }

    for (uint i = 0; i < AREA_THREAD; i++) {
        if (tiisg < 4 && (rx + (int)i) < area_size) {
            const int e = (int)tiisg;
            FLOAT gateV = (FLOAT)activate((ftype)(result0[i][e] + biasValue0[e]), cst.activation);
            FLOAT upV   = (FLOAT)activate((ftype)(result1[i][e] + biasValue1[e]), cst.activation);
            const FLOAT silu = gateV * fast::divide(FLOAT(1), FLOAT(1) + fast::exp2(FLOAT(-1.4426950408889634f) * gateV));
            ((device ftype*)(xy_out + i))[e] = (ftype)(upV * silu);
        }
    }
#else // !GATE_UP_SILU
    auto xy_wt = wt + (long)uz * wt_row_stride;
    auto biasValue = FLOAT4(biasTerms[uz]);
    const FLOAT inv_coef = FLOAT(1.0) / (FLOAT)cur_scale_coef;
    FLOAT4 result[AREA_THREAD] = {FLOAT4(0)};

    for (int bi = blockSlot; bi < cst.blockCount; bi += blocksInFlight) {
        FLOAT4 scale = FLOAT4(dequantScale[2L * ((long)uz * cst.blockCount + bi) + 0]) * inv_coef;
        FLOAT4 dequant_bias = FLOAT4(dequantScale[2L * ((long)uz * cst.blockCount + bi) + 1]) * inv_coef;
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
                auto base_xy = xy_in0 + (long)z * area_size;
            (void)base_xy;
                for (int i = 0; i < AREA_THREAD; i++) {
                    FLOAT4 in40 = G4MX_LOAD_IN4(z, i);
                    input_sum[i] += dot(in40, FLOAT4(1.0));
                    FLOAT in_ps0 = in40[0] * FLOAT(1.0/64.0);
                    FLOAT in_ps1 = in40[1] * FLOAT(1.0/16.0);
                    FLOAT in_ps2 = in40[2] * FLOAT(1.0/4.0);
                    raw_dot[i] += in_ps0 * FLOAT4(w_b & uchar4(0xC0)) + in_ps1 * FLOAT4(w_b & uchar4(0x30))
                                + in_ps2 * FLOAT4(w_b & uchar4(0x0C)) + in40[3] * FLOAT4(w_b & uchar4(0x03));
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
                uchar4 lo = uchar4(uchar(tw0), uchar(tw0 >> 8), uchar(tw1), uchar(tw1 >> 8));
                uchar hb0 = uchar(tw2), hb1 = uchar(tw2 >> 8);
                uchar4 hi = uchar4(hb0 >> 4, hb0 & 0xF, hb1 >> 4, hb1 & 0xF);
                auto base_xy = xy_in0 + (long)z * area_size;
            (void)base_xy;
                for (int i = 0; i < AREA_THREAD; i++) {
                    FLOAT4 in40 = G4MX_LOAD_IN4(z, i);
                    input_sum[i] += dot(in40, FLOAT4(1.0));
                    FLOAT in_ps0 = in40[0] * FLOAT(1.0/64.0);
                    FLOAT in_ps1 = in40[1] * FLOAT(1.0/16.0);
                    FLOAT in_ps2 = in40[2] * FLOAT(1.0/4.0);
                    FLOAT in_hi0 = in40[0] * FLOAT(0.5);
                    FLOAT in_hi2 = in40[2] * FLOAT(2.0);
                    FLOAT in_hi3 = in40[3] * FLOAT(4.0);
                    raw_dot[i] += in_ps0 * FLOAT4(lo & uchar4(0xC0)) + in_ps1 * FLOAT4(lo & uchar4(0x30))
                                + in_ps2 * FLOAT4(lo & uchar4(0x0C)) + in40[3] * FLOAT4(lo & uchar4(0x03))
                                + in_hi0 * FLOAT4(hi & uchar4(0x8)) + in40[1] * FLOAT4(hi & uchar4(0x4))
                                + in_hi2 * FLOAT4(hi & uchar4(0x2)) + in_hi3 * FLOAT4(hi & uchar4(0x1));
                }
            }
            FLOAT4 adjusted_bias = dequant_bias - FLOAT(4.0) * scale;
            for (int i = 0; i < AREA_THREAD; i++) {
                result[i] += raw_dot[i] * scale + input_sum[i] * adjusted_bias;
            }
        #endif
        }
    #elif defined(W_QUANT_4)
        if constexpr (AREA_THREAD <= 5) {
            // Deferred dequantization with vectorized nibble mask: eliminates
            // per-z scale/bias FMAs while staying within register budget for M<=5.
            FLOAT4 raw_dot[AREA_THREAD] = {FLOAT4(0)};
            FLOAT input_sum[AREA_THREAD] = {FLOAT(0)};
            for (int z = zmin + laneInBlock; z < zmax; z += lanesPerBlock) {
                ushort4 w16 = xy_wt[z];
                FLOAT4 w0 = FLOAT4(w16 & ushort4(0x00F0u));
                FLOAT4 w1 = FLOAT4(w16 & ushort4(0x000Fu));
                FLOAT4 w2 = FLOAT4(w16 & ushort4(0xF000u));
                FLOAT4 w3 = FLOAT4(w16 & ushort4(0x0F00u));
                auto base_xy = xy_in0 + (long)z * area_size;
            (void)base_xy;
                for (int i = 0; i < AREA_THREAD; i++) {
                    FLOAT4 in40 = G4MX_LOAD_IN4(z, i);
                    input_sum[i] += dot(in40, FLOAT4(1.0));
                    FLOAT s0 = in40[0] * FLOAT(0.0625);
                    FLOAT s2 = in40[2] * FLOAT(0.000244140625);
                    FLOAT s3 = in40[3] * FLOAT(0.00390625);
                    raw_dot[i] += s0 * w0 + in40[1] * w1 + s2 * w2 + s3 * w3;
                }
            }
            FLOAT4 adjusted_bias = dequant_bias - FLOAT(8.0) * scale;
            for (int i = 0; i < AREA_THREAD; i++) {
                result[i] += raw_dot[i] * scale + input_sum[i] * adjusted_bias;
            }
        } else {
            // Keep single result[AREA_THREAD] accumulator to avoid register spilling;
            // hoist -8*scale into adjusted_bias and mask nibbles directly with power-of-two scale.
            FLOAT4 adjusted_bias = dequant_bias - FLOAT(8.0) * scale;
            const FLOAT4 nibble_scale = FLOAT4(FLOAT(0.0625), FLOAT(1.0),
                                               FLOAT(0.000244140625), FLOAT(0.00390625));
            for (int z = zmin + laneInBlock; z < zmax; z += lanesPerBlock) {
                ushort4 w16 = xy_wt[z];
                FLOAT4x4 w_dequant;
                for (int r = 0; r < 4; r += 1) {
                    ushort w = w16[r];
                    FLOAT4 w4 = FLOAT4(FLOAT(w & 0x00F0u), FLOAT(w & 0x000Fu),
                                       FLOAT(w & 0xF000u), FLOAT(w & 0x0F00u));
                    w_dequant[r] = w4 * (scale[r] * nibble_scale) + adjusted_bias[r];
                }

                auto base_xy = xy_in0 + (long)z * area_size;
            (void)base_xy;
                for (int i = 0; i < AREA_THREAD; i++) {
                    FLOAT4 in40 = G4MX_LOAD_IN4(z, i);
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

            auto base_xy = xy_in0 + (long)z * area_size;
            (void)base_xy;
            for (int i = 0; i < AREA_THREAD; i++) {
                FLOAT4 in40 = G4MX_LOAD_IN4(z, i);
                result[i] += FLOAT4(in40 * w_dequant);
            }
        }
    #endif
    }

    for (int i = 0; i < AREA_THREAD; i++) {
        result[i] = simd_sum(result[i]);
    }

    // result store
    for (uint i = 0; i < AREA_THREAD; i++) {
        if (tiisg == i && (rx + (int)i) < area_size) {
            ftype4 value = activate(ftype4(result[i] + biasValue), cst.activation);
            CONV_ROW_SCALE(value, rx + (int)i);
            xy_out[i] = value;
        }
    }
#endif // GATE_UP_SILU
#undef G4MX_LOAD_IN4
}

typedef decltype(conv1x1_gemv_g4mx_wquant_sg<1>) kernel_type_t;
template [[host_name("conv1x1_gemv_g4m2_wquant_sg")]] kernel kernel_type_t conv1x1_gemv_g4mx_wquant_sg<2>;
template [[host_name("conv1x1_gemv_g4m3_wquant_sg")]] kernel kernel_type_t conv1x1_gemv_g4mx_wquant_sg<3>;
template [[host_name("conv1x1_gemv_g4m4_wquant_sg")]] kernel kernel_type_t conv1x1_gemv_g4mx_wquant_sg<4>;
template [[host_name("conv1x1_gemv_g4m5_wquant_sg")]] kernel kernel_type_t conv1x1_gemv_g4mx_wquant_sg<5>;
#if !defined(GATE_UP_FUSED) && !defined(QKV_FUSED)
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
#endif

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
                            //          [6..8]=packed-grid base of projections 1..3
                            constant float *qkv_seg                         [[buffer(14)]],
                        #ifdef QKV_FUSED_4PROJ
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
                        #ifdef CONV_ROW_MUL
                            const device ftype4 *rowScale                    [[buffer(24)]],
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
#ifdef QKV_FUSED_4PROJ
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
    // Write to each member's own tensor; the graph allocator owns its lifetime.
    if (qkv_proj == 1) {
        out = out_k;
        biasTerms = biasTerms_k;
        wt = wt_k;
        dequantScale = dequantScale_k;
        qkv_scale_coef = qkv_seg[0];
        qkvOutputDepthQuad = int(qkv_seg[2]);
    } else if (qkv_proj == 2) {
        out = out_v;
        biasTerms = biasTerms_v;
        wt = wt_v;
        dequantScale = dequantScale_v;
        qkv_scale_coef = qkv_seg[1];
        qkvOutputDepthQuad = int(qkv_seg[3]);
    }
#ifdef QKV_FUSED_4PROJ
    // 4-projection groups (Qwen3.5 linear-attention layers: qkv/z/b/a share
    // one LN input).
    else if (qkv_proj == 3) {
        out = out_w;
        biasTerms = biasTerms_w;
        wt = wt_w;
        dequantScale = dequantScale_w;
        qkv_scale_coef = qkv_seg[4];
        qkvOutputDepthQuad = int(qkv_seg[5]);
    }
#endif
    const int tg_x = qkv_gx;
#else
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
// sgitg % tgQuads picks the quad, sgitg / tgQuads the K range; past 4 quads a
// threadgroup gives up more parallelism than the shared LN prologue repays.
#if defined(GEMV_SG_OC_QUADS_2)
// Two oc quads per simdgroup, carried as two accumulator streams in one thread:
// doubles the weight reads in flight with no barrier and shares the input read and
// the LN prologue. 2 simdgroups, 4 quads/TG -- except under GATE_UP_SILU, whose
// second stream is the up matrix at the SAME quad, so the threadgroup owns 2 quads.
// GEMV_SG_OC_QUADS_2_KSPLIT keeps the quads and pairs 2 * factor simdgroups over K.
#ifdef GEMV_SG_OC_QUADS_2_KSPLIT
    constexpr int tg_simds  = 2 * GEMV_SG_OC_QUADS_2_KSPLIT;
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

    // Hoisted above the per-simdgroup output-range checks below: every thread must
    // reach the reduction barrier, and a tail threadgroup can have one simdgroup land
    // outside outputDepthQuad. inv_rms is one scalar for the whole threadgroup, so its
    // tg_simds * 32 threads sweep the input once and reduce through threadgroup memory
    // rather than each simdgroup re-reading input+residual. Only one threadgroup writes
    // ln_residual_out, since several of them share the same input quads.
    float sq_sum = 0.0f;
#ifdef LN_STAGE
    // The sweep already touches every input quad once per threadgroup, so it keeps
    // (in + residual) * gamma here instead of letting the GEMV body re-read both device
    // streams per quad and K half. inv_rms is only known afterwards and is applied on
    // read. Storing ftype4 rounds the pre-inv_rms value, unlike unfused AddRMSNorm; the
    // RMS reduction and GEMV accumulation stay fp32.
    threadgroup ftype4 ln_staged[LN_STAGE_QUADS];
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
        ln_staged[z] = (ftype4)((FLOAT4)d * (FLOAT4)ln_gamma[z]);
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

#ifdef GEMV_SG_OC_QUADS_2
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
#ifdef GEMV_SG_OC_QUADS_2_KSPLIT
    // K-split on top of the dual streams: 2 * GEMV_SG_OC_QUADS_2_KSPLIT simdgroups
    // instead of 2. Each pair covers the threadgroup's two quads and sweeps one
    // quant-block range, with partials merged in the epilogue. The host only enables
    // this when outputDepthQuad is even and blockCount divides by the factor, so
    // nothing early-returns before the barrier below.
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
    const long wt_row_stride = (long)cst.inputDepthQuad * 6;
#else
    const long wt_row_stride = (long)cst.inputDepthQuad;
#endif
    auto xy_wt0 = wt + (long)uz * wt_row_stride;
    auto xy_in0 = in;
    auto biasValue0 = FLOAT4(biasTerms[uz]);
#ifdef GATE_UP_SILU
    auto xy_wt1 = wt_up + (long)uz1 * wt_row_stride;
    auto biasValue1 = FLOAT4(biasTerms_up[uz1]);
    const device ftype4* dqs1 = dequantScale_up;
#else
    auto xy_wt1 = wt + (long)uz1 * wt_row_stride;
    auto biasValue1 = FLOAT4(biasTerms[uz1]);
    const device ftype4* dqs1 = dequantScale;
#endif

    int quadsPerBlock = (cst.inputDepthQuad + cst.blockCount - 1) / cst.blockCount;
    int lanesPerBlock = min(SIMD_GROUP_WIDTH, max(quadsPerBlock / 4, 1));
#ifdef GEMV_SG_OC_QUADS_2_KSPLIT
    // A pair now owns only blockCount / GEMV_SG_OC_QUADS_2_KSPLIT blocks. Widen the lanes per
    // block so blocksInFlight never exceeds that count -- otherwise the surplus
    // lane groups sit idle instead of shortening each lane's run.
    const int ds_blocks_owned = max(cst.blockCount / GEMV_SG_OC_QUADS_2_KSPLIT, 1);
    lanesPerBlock = max(lanesPerBlock, SIMD_GROUP_WIDTH / ds_blocks_owned);
    lanesPerBlock = min(lanesPerBlock, SIMD_GROUP_WIDTH);
#endif
    int blocksInFlight = SIMD_GROUP_WIDTH / lanesPerBlock;
    int laneInBlock = tiisg % lanesPerBlock;
    int blockSlot  = tiisg / lanesPerBlock;
    // Lanes left over by a non-power-of-two lanesPerBlock would repeat a block.
    if (blockSlot >= blocksInFlight) {
        blockSlot = cst.blockCount;
    }

    FLOAT4 result0 = FLOAT4(0);
    FLOAT4 result1 = FLOAT4(0);

#ifdef GEMV_SG_OC_QUADS_2_KSPLIT
    const int ds_bi_begin = ds_sk * ds_blocks_owned;
    const int ds_bi_end   = ds_bi_begin + ds_blocks_owned;
#else
    const int ds_bi_begin = 0;
    const int ds_bi_end   = cst.blockCount;
#endif

    for (int bi = ds_bi_begin + blockSlot; bi < ds_bi_end; bi += blocksInFlight) {
        FLOAT4 scale0 = FLOAT4(dequantScale[2L * ((long)uz * cst.blockCount + bi) + 0]) * (FLOAT)inv_coef0;
        FLOAT4 dbias0 = FLOAT4(dequantScale[2L * ((long)uz * cst.blockCount + bi) + 1]) * (FLOAT)inv_coef0;
        FLOAT4 scale1 = FLOAT4(dqs1[2L * ((long)uz1 * cst.blockCount + bi) + 0]) * (FLOAT)inv_coef1;
        FLOAT4 dbias1 = FLOAT4(dqs1[2L * ((long)uz1 * cst.blockCount + bi) + 1]) * (FLOAT)inv_coef1;
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
        FLOAT4 adj0 = dbias0 - FLOAT(8.0) * scale0;
        FLOAT4 adj1 = dbias1 - FLOAT(8.0) * scale1;
        result0 += raw_dot0 * scale0 + input_sum * adj0;
        result1 += raw_dot1 * scale1 + input_sum * adj1;
    #elif defined(W_QUANT_8)
        // w[j] holds output lane j's 4 ic weights (transform packs ro*4+ri),
        // so each lane needs dot(in4, w[j]) with lane j's scale/bias. Deferred
        // like the W4 branch: raw dot + input_sum, dequant applied per block.
        // (A per-z `result += in4[i] * (w[i]*scale[i]+bias[i])` loop is the
        // TRANSPOSED product, with the wrong scale lane.)
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
#ifdef GEMV_SG_OC_QUADS_2_KSPLIT
    threadgroup FLOAT4 ds_partial[2][2 * GEMV_SG_OC_QUADS_2_KSPLIT];
    const int ds_slot = (int)sgitg & 1;
    if (tiisg == 0) {
        ds_partial[0][sgitg] = result0;
        ds_partial[1][sgitg] = result1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#ifndef GATE_UP_SILU
    // The silu-mul epilogue below merges the partials element-parallel instead.
    const bool ds_write = (ds_sk == 0 && tiisg == 0);
    if (ds_write) {
        result0 = FLOAT4(0);
        result1 = FLOAT4(0);
        for (int s = 0; s < GEMV_SG_OC_QUADS_2_KSPLIT; ++s) {
            result0 += ds_partial[0][s * 2 + ds_slot];
            result1 += ds_partial[1][s * 2 + ds_slot];
        }
    }
#endif
#else
#ifndef GATE_UP_SILU
    const bool ds_write = (tiisg == 0);
#endif
#endif
#ifdef GATE_UP_SILU
    // Matches the op this replaces (MUL_SILU: in0 = up, in1 = gate), applied after each
    // projection's own activation. Element-parallel: lanes 0-3 own one output element
    // each, so the four exp/div chains do not serialize on lane 0. silu(g) is evaluated
    // as g * (1 / (1 + exp2(-log2e * g))) with fast::exp2 / fast::divide, whose ~2^-11
    // relative error sits below the fp16 store quantization.
#ifdef GEMV_SG_OC_QUADS_2_KSPLIT
    if (ds_sk == 0 && tiisg < 4) {
#else
    if (tiisg < 4) {
#endif
        const int e = (int)tiisg;
        FLOAT gateV;
        FLOAT upV;
#ifdef GEMV_SG_OC_QUADS_2_KSPLIT
        // This lane merges its output element across the split-K partials.
        gateV = FLOAT(0);
        upV = FLOAT(0);
        for (int s = 0; s < GEMV_SG_OC_QUADS_2_KSPLIT; ++s) {
            gateV += ds_partial[0][s * 2 + ds_slot][e];
            upV   += ds_partial[1][s * 2 + ds_slot][e];
        }
        gateV += biasValue0[e];
        upV   += biasValue1[e];
#else
        gateV = result0[e] + biasValue0[e];
        upV   = result1[e] + biasValue1[e];
#endif
        gateV = (FLOAT)activate((ftype)gateV, cst.activation);
        upV   = (FLOAT)activate((ftype)upV, cst.activation);
        const FLOAT silu = gateV * fast::divide(FLOAT(1), FLOAT(1) + fast::exp2(FLOAT(-1.4426950408889634f) * gateV));
        ((device ftype*)(out + uz * area_size))[e] = (ftype)(upV * silu);
    }
#else
    if (ds_write) {
        ftype4 value0 = activate(ftype4(result0 + biasValue0), cst.activation);
        CONV_ROW_SCALE(value0, 0);
        out[uz * area_size] = value0;
        if (stream1_valid) {
            ftype4 value1 = activate(ftype4(result1 + biasValue1), cst.activation);
            CONV_ROW_SCALE(value1, 0);
            out[uz1 * area_size] = value1;
        }
    }
#endif
#else  // !GEMV_SG_OC_QUADS_2

    // Single stream: each simdgroup owns one output quad and one K range.
#if GEMV_SPLIT_K > 1
    // Split-K: sgitg % tgQuads picks the output quad, sgitg / tgQuads the K
    // range; partials combined via threadgroup memory. Host guarantees the grid is
    // exact (oc % (4 * tgQuads) == 0) so no simdgroup early-returns before the
    // barrier below.
    const int uz = tg_x * tgQuads + ((int)sgitg % tgQuads);
    const int sk_half = (int)sgitg / tgQuads;
#else
    // tgQuads = GEMV_QUADS_PER_TG (2, or 4 for the oc16Nsg4SplitK1 decode variant); each
    // simdgroup owns the quad matching its index. Hardcoding 2 here left the
    // upper half of every oc16Nsg4SplitK1 output unwritten (stale memory -> garbage decode).
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
    // whenever gate/up weights have different fp16-fit ranges.
    if (gid.z == 1) {
        cur_scale_coef = gate_up_seg[0];
    }
    #endif
    // Hoist the per-bi scale/bias divisions out of the loop: the coef is
    // loop-invariant, so multiply by its reciprocal instead.
    const float inv_scale_coef = 1.0f / cur_scale_coef;

#ifdef W_QUANT_3
    auto xy_wt = wt + (long)uz * cst.inputDepthQuad * 6;
#else
    auto xy_wt = wt + (long)uz * cst.inputDepthQuad;
#endif
    auto xy_in0 = in;
    auto biasValue = FLOAT4(biasTerms[uz]);

    int quadsPerBlock = (cst.inputDepthQuad + cst.blockCount - 1) / cst.blockCount;
    // GEMV inner reduction lane partitioning.
    //   GEMV_LANES_PER_BLOCK (split-K pipelines): host-computed, see gemvLanesPerBlock().
    //   Fallback: min(32, max(quadsPerBlock/4, 1)) -- a few lanes share a block's inner
    //   K reduction while the rest iterate outer blocks, which suits a bandwidth-bound SM.
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
    // Lanes left over by a non-power-of-two lanesPerBlock would repeat a block.
    if (blockSlot >= blocksInFlight) {
        blockSlot = cst.blockCount;
    }

    FLOAT4 result = FLOAT4(0);

#if GEMV_SPLIT_K > 1
    const int sk_bi_begin = sk_half * (cst.blockCount / 2);
    const int sk_bi_end   = (sk_half == 1) ? cst.blockCount : (cst.blockCount / 2);
    for (int bi = sk_bi_begin + blockSlot; bi < sk_bi_end; bi += blocksInFlight) {
#else
    for (int bi = blockSlot; bi < cst.blockCount; bi += blocksInFlight) {
#endif
        FLOAT4 scale = FLOAT4(dequantScale[2L * ((long)uz * cst.blockCount + bi) + 0]) * (FLOAT)inv_scale_coef;
        FLOAT4 dequant_bias = FLOAT4(dequantScale[2L * ((long)uz * cst.blockCount + bi) + 1]) * (FLOAT)inv_scale_coef;
        int zmin = bi * quadsPerBlock;
        int zmax = min(zmin + quadsPerBlock, cst.inputDepthQuad);

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
            // Vectorized mask-only unpack (see the GEMV_SG_OC_QUADS_2 body above).
            uchar4 w_b = xy_wt[z];
            raw_dot += in_ps0 * FLOAT4(w_b & uchar4(0xC0)) + in_ps1 * FLOAT4(w_b & uchar4(0x30))
                     + in_ps2 * FLOAT4(w_b & uchar4(0x0C)) + in4[3] * FLOAT4(w_b & uchar4(0x03));
        #else
            FLOAT in_hi0 = in4[0] * FLOAT(0.5);
            FLOAT in_hi2 = in4[2] * FLOAT(2.0);
            FLOAT in_hi3 = in4[3] * FLOAT(4.0);
            const device ushort* t = (const device ushort*)(xy_wt + z * 6);
            ushort tw0 = t[0], tw1 = t[1], tw2 = t[2];
            // Vectorized lo-byte / hi-nibble unpack (see the GEMV_SG_OC_QUADS_2 body above).
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
        FLOAT4 adjusted_bias = dequant_bias - FLOAT(8.0) * scale;
        result += raw_dot * scale + input_sum * adjusted_bias;
    #elif defined(W_QUANT_8)
        // See the GEMV_SG_OC_QUADS_2 W8 branch above: w[j] = output lane j's ic weights, so
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
        ftype4 value = activate(ftype4(result + biasValue), cst.activation);
        CONV_ROW_SCALE(value, 0);
        out[uz * area_size] = value;
    }
#else
    if (tiisg == 0) {
        ftype4 value = activate(ftype4(result + biasValue), cst.activation);
        CONV_ROW_SCALE(value, 0);
        out[uz * area_size] = value;
    }
#endif
#endif // GEMV_SG_OC_QUADS_2
#undef GEMV_2SG_LOAD_IN4
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
                        #ifdef CONV_ROW_MUL
                            const device ftype4 *rowScale       [[buffer(24)]],
                        #endif
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
    auto xy_wt = wt + (long)uz * cst.inputDepthQuad * 6;
#else
    auto xy_wt = wt + (long)uz * cst.inputDepthQuad;
#endif
    auto xy_in0  = in + rx;
    auto area_size = cst.outputSize * cst.batch;
    auto xy_out = out + (long)uz * area_size + rx;
    FLOAT4 result0 = FLOAT4(0);
    threadgroup FLOAT4 localSum[32];
    if(uz < cst.outputDepthQuad) {
        int quadsPerBlock = (cst.inputDepthQuad + cst.blockCount - 1) / cst.blockCount;
        
        int lanesPerBlock = min(SIMD_GROUP_WIDTH_4, quadsPerBlock);
        int blocksInFlight = SIMD_GROUP_WIDTH_4 / lanesPerBlock;
        int laneInBlock = (tiisg + i_sgitg * SIMD_GROUP_WIDTH) % lanesPerBlock;
        int blockSlot = (tiisg + i_sgitg * SIMD_GROUP_WIDTH) / lanesPerBlock;
        // Lanes left over by a non-power-of-two lanesPerBlock would repeat a block.
        if (blockSlot >= blocksInFlight) {
            blockSlot = cst.blockCount;
        }


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
            // W_QUANT_8: deferred dequantization (matches g4m1_2sg)
            {
                FLOAT4 raw_dot = FLOAT4(0);
                FLOAT input_sum = FLOAT(0);
                for (int z = zmin + laneInBlock; z < zmax; z += lanesPerBlock) {
                    FLOAT4 in40 = (FLOAT4)*(xy_in0 + z * area_size);
                    auto w = xy_wt[z];
                    input_sum += dot(in40, FLOAT4(1.0));
                    raw_dot += FLOAT4(dot(in40, FLOAT4(w[0])), dot(in40, FLOAT4(w[1])),
                                      dot(in40, FLOAT4(w[2])), dot(in40, FLOAT4(w[3])));
                }
                result0 += raw_dot * scale + input_sum * dequant_bias;
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
            ftype4 value = activate(ftype4(res), cst.activation);
            CONV_ROW_SCALE(value, rx);
            xy_out[0] = value;
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
    if(uz >= cst.outputDepthQuad) {
        return;
    }
    auto area_size = cst.outputSize * cst.batch;
    int rx = gid.y;
#ifdef W_QUANT_3
    auto xy_wt = wt + (long)uz * cst.inputDepthQuad * 6;
#else
    auto xy_wt = wt + (long)uz * cst.inputDepthQuad;
#endif
    auto xy_in0  = in + rx;
    auto xy_out = out + (long)uz * area_size + rx;
    auto biasValue0 = FLOAT4(biasTerms[uz]);
    auto biasValue1 = FLOAT4(biasTerms[uz + 1]);

    FLOAT4 result0 = FLOAT4(0);
    FLOAT4 result1 = FLOAT4(0);


    int quadsPerBlock = (cst.inputDepthQuad + cst.blockCount - 1) / cst.blockCount;

    int lanesPerBlock = min(SIMD_GROUP_WIDTH, quadsPerBlock);
    int blocksInFlight = SIMD_GROUP_WIDTH / lanesPerBlock;
    int laneInBlock = (tiisg) % lanesPerBlock;
    int blockSlot = (tiisg) / lanesPerBlock;
    // Lanes left over by a non-power-of-two lanesPerBlock would repeat a block.
    if (blockSlot >= blocksInFlight) {
        blockSlot = cst.blockCount;
    }

    for (int bi= blockSlot; bi<cst.blockCount; bi += blocksInFlight) {
        const long quant_offset = 2L * ((long)uz * cst.blockCount + bi);
        FLOAT4 scale0 = FLOAT4(dequantScale[quant_offset + 0]) / (FLOAT)cst.scaleCoef;
        FLOAT4 dequant_bias0 = FLOAT4(dequantScale[quant_offset + 1]) / (FLOAT)cst.scaleCoef;
        FLOAT4 scale1 = FLOAT4(dequantScale[quant_offset + ((long)cst.blockCount << 1)]) / (FLOAT)cst.scaleCoef;
        FLOAT4 dequant_bias1 = FLOAT4(dequantScale[quant_offset + ((long)cst.blockCount << 1) + 1]) / (FLOAT)cst.scaleCoef;
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
        {
            FLOAT4 raw_dot0 = FLOAT4(0), raw_dot1 = FLOAT4(0);
            FLOAT input_sum = FLOAT(0);
            for (int z = zmin + laneInBlock; z < zmax; z += lanesPerBlock) {
                FLOAT4 in40 = (FLOAT4)*(xy_in0 + z * area_size);
                auto w0 = xy_wt[z];
                auto w1 = xy_wt[cst.inputDepthQuad + z];
                input_sum += dot(in40, FLOAT4(1.0));
                raw_dot0 += FLOAT4(dot(in40, FLOAT4(w0[0])), dot(in40, FLOAT4(w0[1])),
                                   dot(in40, FLOAT4(w0[2])), dot(in40, FLOAT4(w0[3])));
                raw_dot1 += FLOAT4(dot(in40, FLOAT4(w1[0])), dot(in40, FLOAT4(w1[1])),
                                   dot(in40, FLOAT4(w1[2])), dot(in40, FLOAT4(w1[3])));
            }
            result0 += raw_dot0 * scale0 + input_sum * dequant_bias0;
            result1 += raw_dot1 * scale1 + input_sum * dequant_bias1;
        }
        #endif
    }

    FLOAT4 res0 = simd_sum(result0);
    FLOAT4 res1 = simd_sum(result1);

    if (tiisg == 0) {
        xy_out[0] = activate(ftype4(res0 + biasValue0), cst.activation);
        xy_out[area_size] = activate(ftype4(res1 + biasValue1), cst.activation);
    }
}
)metal";

namespace MNN {

// The production loader and the registry test must see the same assembled text.
static inline std::string conv1x1ShaderSource(PrefillSource source) {
    switch (source) {
        case PrefillSource::WfpMatrix:
            return std::string(gBasicConvPrefix) + gConv1x1WfpSgMatrix + gConv1x1WfpSgMatrixM64;
        case PrefillSource::WqMatrix:
            return std::string(gBasicConvPrefix) + gConv1x1WqSgMatrix;
        case PrefillSource::WqReduce:
            return std::string(gBasicConvPrefix) + gConv1x1WqSgReduce;
        case PrefillSource::None:
        case PrefillSource::Precompiled:
            break;
    }
    return {};
}

} // namespace MNN

#endif
