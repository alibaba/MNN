//
//  MNNSpacemitIme2GemmI8I4Local.cpp
//  MNN
//
//  Spacemit IME2 i8 x i4 GEMM kernels adapted from llama.cpp's
//  ggml-cpu/spacemit/ime2_kernels.cpp for MNN block64 W4 inference.
//
#include <cstddef>
#include <cstdint>
#include <algorithm>

#if defined(MNN_USE_SPACEMIT_IME2)

namespace {

static constexpr bool MNNSpacemitIme2HpRefEnabled() {
    return false;
}

static constexpr bool MNNSpacemitIme2HpM1CenteredEnabled() {
    return false;
}

static constexpr bool MNNSpacemitIme2HpM1NativeHpEnabled() {
    return false;
}

static inline size_t MNNSpacemitIme2Vlenb() {
    size_t value = 0;
    asm volatile("csrr %0, vlenb" : "=r"(value));
    return value;
}

static constexpr bool MNNSpacemitIme2A4RefEnabled() {
    return false;
}

static constexpr bool MNNSpacemitIme2FixedAScaleEnabled() {
    return true;
}

static inline int8_t MNNSpacemitIme2I4High(uint8_t value) {
    const int q = (value >> 4) & 0x0F;
    return static_cast<int8_t>(q >= 8 ? q - 16 : q);
}

template <size_t MB_ROWS>
static void MNNSpacemitIme2GemmI4I4HpRef(size_t blk_len, const uint8_t* quant_a_ptr, const uint8_t* quant_b_data,
                                         const uint8_t* quant_b_zp, float* c_ptr, size_t count_m, size_t count_n,
                                         size_t k_blks, size_t ldc) {
    (void)count_m;
    if (blk_len != 257 || quant_b_zp != nullptr) {
        return;
    }
    constexpr size_t NB_COLS = 32;
    constexpr size_t K_SUBBLOCKS = 8;
    constexpr size_t B_SUBBLOCK_BYTES = sizeof(_Float16) * NB_COLS + 16 * NB_COLS;
    const size_t bSuperBlockStride = B_SUBBLOCK_BYTES * K_SUBBLOCKS;
    const size_t bTileStride = k_blks * bSuperBlockStride;
    const size_t aSubBlockStride = (sizeof(_Float16) + 32) * MB_ROWS;
    const size_t aBlockStride =
        (size_t(256) + size_t(8) * sizeof(_Float16) + size_t(8) * sizeof(_Float16) + sizeof(_Float16)) * MB_ROWS;

    float output[MB_ROWS * NB_COLS];
    for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
        const size_t nbReal = std::min<size_t>(NB_COLS, count_n - ni);
        const uint8_t* bTileBase = quant_b_data + (ni / NB_COLS) * bTileStride;
        const uint8_t* aData = quant_a_ptr;
        for (size_t i = 0; i < MB_ROWS * NB_COLS; ++i) {
            output[i] = 0.0f;
        }

        for (size_t ki = 0; ki < k_blks; ++ki, aData += aBlockStride) {
            const uint8_t* bSuperBlock = bTileBase + ki * bSuperBlockStride;
            const _Float16* aSum = reinterpret_cast<const _Float16*>(aData + aSubBlockStride * K_SUBBLOCKS);
            const _Float16* aScaleAvg =
                reinterpret_cast<const _Float16*>(aData + aBlockStride - sizeof(_Float16) * MB_ROWS);
            const float scaleFactor = static_cast<float>(aScaleAvg[0]);
            for (size_t ksi = 0; ksi < K_SUBBLOCKS; ++ksi) {
                const uint8_t* bBlock = bSuperBlock + ksi * B_SUBBLOCK_BYTES;
                const _Float16* bScale = reinterpret_cast<const _Float16*>(bBlock);
                const uint8_t* bQs = bBlock + sizeof(_Float16) * NB_COLS;
                const _Float16* aScale = reinterpret_cast<const _Float16*>(aData + aSubBlockStride * ksi);
                const uint8_t* aQ = aData + aSubBlockStride * ksi + sizeof(_Float16) * MB_ROWS;
                for (size_t mi = 0; mi < MB_ROWS; ++mi) {
                    const float aScaleValue = static_cast<float>(aScale[mi]) * scaleFactor;
                    const float aSumValue = static_cast<float>(aSum[mi * K_SUBBLOCKS + ksi]);
                    for (size_t ci = 0; ci < NB_COLS; ++ci) {
                        const uint8_t* bCol = bQs + ci * 16;
                        int32_t acc = 0;
                        for (size_t bi = 0; bi < 16; ++bi) {
                            const uint8_t b = bCol[bi];
                            const int b0 = static_cast<int>(b & 0x0F);
                            const int b1 = static_cast<int>((b >> 4) & 0x0F);
                            acc += static_cast<int>(MNNSpacemitIme2I4High(aQ[mi * 32 + 2 * bi])) * b0 +
                                   static_cast<int>(MNNSpacemitIme2I4High(aQ[mi * 32 + 2 * bi + 1])) * b1;
                        }
                        output[ci + mi * NB_COLS] +=
                            (static_cast<float>(acc) + aSumValue) * static_cast<float>(bScale[ci]) * aScaleValue;
                    }
                }
            }
        }

        for (size_t mi = 0; mi < MB_ROWS; ++mi) {
            for (size_t ci = 0; ci < nbReal; ++ci) {
                c_ptr[mi * ldc + ni + ci] = output[mi * NB_COLS + ci];
            }
        }
    }
}

static void MNNSpacemitIme2GemmI4I4HpM4(size_t blk_len, const uint8_t* quant_a_ptr, const uint8_t* quant_b_data,
                                        const uint8_t* quant_b_zp, float* c_ptr, size_t count_m, size_t count_n,
                                        size_t k_blks, size_t ldc) {
    (void)count_m;
    if (blk_len != 257 || quant_b_zp != nullptr) {
        return;
    }
    constexpr size_t NB_COLS = 32;
    constexpr size_t B_SUB_STRIDE = sizeof(_Float16) * NB_COLS + 16 * NB_COLS;
    const size_t B_SUPER_STRIDE = 8 * B_SUB_STRIDE;
    const size_t b_tile_stride = k_blks * B_SUPER_STRIDE;

    for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
        uint8_t* b_tile_base = (uint8_t*)quant_b_data + (ni / NB_COLS) * b_tile_stride;
        uint8_t* a_block = (uint8_t*)quant_a_ptr;
        float* dst_c = c_ptr + ni;

        asm volatile(
            "mv             t5, %[BK]                 \n\t"
            "mv             t6, %[A]                  \n\t"
            "mv             s5, %[B]                  \n\t"
            "vsetvli        t0, x0, e32, m1           \n\t"
            "vxor.vv        v28, v28, v28             \n\t"
            "vxor.vv        v29, v29, v29             \n\t"
            "vxor.vv        v30, v30, v30             \n\t"
            "vxor.vv        v31, v31, v31             \n\t"
            "li             t4, 8                     \n\t"
            "addi           t2, t6, 1088              \n\t"

            ".align 4                                 \n\t"
            "_A4_BLK_LPST%=:                          \n\t"
            "flh            fa1, 64(t2)               \n\t"
            "vsetvli        t0, x0, e32, m1           \n\t"
            "vxor.vv        v18, v30, v30             \n\t"
            "vxor.vv        v19, v31, v31             \n\t"
            "vxor.vv        v20, v30, v30             \n\t"
            "vxor.vv        v21, v31, v31             \n\t"
            "_A4_KsubBLK_LPST%=:                      \n\t"
            "flh            fa0,   0(t6)              \n\t"

            "vsetvli        t0, x0, e16, mf2          \n\t"
            "vle16.v        v8, (s5)                  \n\t"

            "vsetvli        t0, x0, e16, mf2          \n\t"
            "vfmul.vf       v16, v8, fa0              \n\t"

            "flh            ft1, 0(t2)                \n\t"
            "flh            ft2, 16(t2)               \n\t"
            "flh            ft3, 32(t2)               \n\t"
            "flh            ft4, 48(t2)               \n\t"

            "addi           t3, t6, 8                 \n\t"
            "vsetvli        t0, x0, e8, m1            \n\t"
            "vl1r.v         v0, (t3)                  \n\t"
            "addi           t3, s5, 64                \n\t"
            "vl4r.v         v4, (t3)                  \n\t"

            "vsetvli        t0, x0, e16, mf2          \n\t"
            "vfmul.vf       v12, v16, ft1             \n\t"
            "vfmul.vf       v13, v16, ft2             \n\t"
            "vfmul.vf       v24, v16, ft3             \n\t"
            "vfmul.vf       v25, v16, ft4             \n\t"

            "vsetvli        t0, x0, e16, mf2          \n\t"
            "vfwmacc.vf     v28, fa1, v12             \n\t"
            "vfwmacc.vf     v29, fa1, v13             \n\t"
            "vfwmacc.vf     v30, fa1, v24             \n\t"
            "vfwmacc.vf     v31, fa1, v25             \n\t"

            "vsetvli        t0, x0, e8, m1            \n\t"
            "vsrl.vi        v1, v0, 4                 \n\t"
            "vnpack4.vv     v12, v0, v1, 3            \n\t"
            "vpack.vv       v0, v16, v16, 3           \n\t"
            "vupack.vv      v2, v12, v12, 2           \n\t"

            "vsetvli        t0, x0, e32, m1           \n\t"
            "vmadotsu.hp    v18, v3, v4, v0, 0, i4    \n\t"
            "vmadotsu.hp    v19, v3, v5, v0, 1, i4    \n\t"
            "vmadotsu.hp    v20, v3, v6, v0, 2, i4    \n\t"
            "vmadotsu.hp    v21, v3, v7, v0, 3, i4    \n\t"

            "addi           t4, t4, -1                \n\t"
            "addi           t6, t6, 8+128             \n\t"
            "addi           t2, t2, 2                 \n\t"
            "addi           s5, s5, 64+512            \n\t"
            "bgtz           t4, _A4_KsubBLK_LPST%=    \n\t"

            "vsetvli        t0, x0, e16, m1           \n\t"
            "vpack.vv       v8, v18, v19, 1           \n\t"
            "vpack.vv       v12, v20, v21, 1          \n\t"
            "vpack.vv       v26, v8, v12, 2           \n\t"

            "vsetvli        t0, x0, e16, m1           \n\t"
            "vfwmacc.vf     v28, fa1, v26             \n\t"
            "vfwmacc.vf     v30, fa1, v27             \n\t"

            "li             t4, 8                     \n\t"
            "addi           t5, t5, -1                \n\t"
            "addi           t6, t6, 72                \n\t"
            "addi           t2, t6, 1088              \n\t"
            "bgtz           t5, _A4_BLK_LPST%=        \n\t"

            "vsetvli        t0, x0, e32, m1           \n\t"
            "add            t2, %[LDC], %[DST]        \n\t"
            "vse32.v        v28, (%[DST])             \n\t"
            "add            t3, %[LDC], t2            \n\t"
            "vse32.v        v29, (t2)                 \n\t"
            "add            t2, %[LDC], t3            \n\t"
            "vse32.v        v30, (t3)                 \n\t"
            "vse32.v        v31, (t2)                 \n\t"
            : [A] "+r"(a_block), [B] "+r"(b_tile_base)
            : [DST] "r"(dst_c), [LDC] "r"(ldc * 4), [BK] "r"(k_blks)
            : "t0", "t2", "t3", "t4", "t5", "t6", "s5", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v10",
              "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v24", "v25", "v26", "v27",
              "v28", "v29", "v30", "v31", "fa0", "fa1", "ft1", "ft2", "ft3", "ft4", "memory");
    }
}

template <size_t MB_ROWS>
static void MNNSpacemitIme2GemmI8I4HpRef(size_t blk_len, const uint8_t* quant_a_ptr, const uint8_t* quant_b_data,
                                         const uint8_t* quant_b_zp, float* c_ptr, size_t count_m, size_t count_n,
                                         size_t k_blks, size_t ldc) {
    (void)count_m;
    if (blk_len != 256) {
        return;
    }
    constexpr size_t NB_COLS = 32;
    constexpr size_t K_SUBBLOCKS = 8;
    constexpr size_t B_SUBBLOCK_BYTES = sizeof(_Float16) * NB_COLS + 16 * NB_COLS;
    const size_t bSuperBlockStride =
        B_SUBBLOCK_BYTES * K_SUBBLOCKS + (quant_b_zp ? NB_COLS * K_SUBBLOCKS * sizeof(uint8_t) : 0);
    const size_t bTileStride = k_blks * bSuperBlockStride;
    const size_t aSubBlockStride = (sizeof(_Float16) + 32) * MB_ROWS;
    const size_t aBlockStride =
        (size_t(256) + size_t(8) * sizeof(_Float16) + size_t(8) * sizeof(_Float16) + sizeof(_Float16)) * MB_ROWS;

    float output[MB_ROWS * NB_COLS];
    for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
        const size_t nbReal = std::min<size_t>(NB_COLS, count_n - ni);
        const uint8_t* bTileBase = quant_b_data + (ni / NB_COLS) * bTileStride;
        const int8_t* aData = reinterpret_cast<const int8_t*>(quant_a_ptr);
        for (size_t i = 0; i < MB_ROWS * NB_COLS; ++i) {
            output[i] = 0.0f;
        }

        for (size_t ki = 0; ki < k_blks; ++ki, aData += aBlockStride) {
            _Float16 outputF16[MB_ROWS * NB_COLS] = {};
            const uint8_t* bSuperBlock = bTileBase + ki * bSuperBlockStride;
            const uint8_t* bZps = quant_b_zp ? bSuperBlock + B_SUBBLOCK_BYTES * K_SUBBLOCKS : nullptr;
            const _Float16* aSum = reinterpret_cast<const _Float16*>(aData + aSubBlockStride * K_SUBBLOCKS);
            const _Float16* aScaleAvg =
                reinterpret_cast<const _Float16*>(aData + aBlockStride - sizeof(_Float16) * MB_ROWS);
            const _Float16 scaleFactor = aScaleAvg[0];

            for (size_t ksi = 0; ksi < K_SUBBLOCKS; ++ksi) {
                const uint8_t* bBlock = bSuperBlock + ksi * B_SUBBLOCK_BYTES;
                const _Float16* bScale = reinterpret_cast<const _Float16*>(bBlock);
                const uint8_t* bQs = bBlock + sizeof(_Float16) * NB_COLS;
                const _Float16* aScale = reinterpret_cast<const _Float16*>(aData + aSubBlockStride * ksi);
                const int8_t* aQ = aData + aSubBlockStride * ksi + sizeof(_Float16) * MB_ROWS;
                for (size_t mi = 0; mi < MB_ROWS; ++mi) {
                    for (size_t ci = 0; ci < NB_COLS; ++ci) {
                        const uint8_t* bCol = bQs + ci * 16;
                        int16_t acc = 0;
                        for (size_t bi = 0; bi < 16; ++bi) {
                            const uint8_t b = bCol[bi];
                            const int8_t b0 = static_cast<int8_t>(b & 0x0F);
                            const int8_t b1 = static_cast<int8_t>((b >> 4) & 0x0F);
                            acc += static_cast<int16_t>(aQ[mi * 32 + 2 * bi]) * static_cast<int16_t>(b0) +
                                   static_cast<int16_t>(aQ[mi * 32 + 2 * bi + 1]) * static_cast<int16_t>(b1);
                        }
                        outputF16[ci + mi * NB_COLS] += static_cast<_Float16>(acc) * bScale[ci] * aScale[0];
                    }
                }
            }

            for (size_t ksi = 0; ksi < K_SUBBLOCKS; ++ksi) {
                const uint8_t* bBlock = bSuperBlock + ksi * B_SUBBLOCK_BYTES;
                const _Float16* bScale = reinterpret_cast<const _Float16*>(bBlock);
                const uint8_t* bZp = bZps ? bZps + ksi * NB_COLS : nullptr;
                const _Float16* aScale = reinterpret_cast<const _Float16*>(aData + aSubBlockStride * ksi);
                for (size_t mi = 0; mi < MB_ROWS; ++mi) {
                    const _Float16 aSumValue = aSum[mi * K_SUBBLOCKS + ksi];
                    for (size_t ci = 0; ci < NB_COLS; ++ci) {
                        _Float16 aSumBzp = aSumValue;
                        if (bZp != nullptr) {
                            aSumBzp = aSumValue * static_cast<_Float16>(0.125f) * static_cast<_Float16>(bZp[ci]);
                        }
                        output[ci + mi * NB_COLS] += static_cast<float>(aSumBzp * bScale[ci] * aScale[0] * scaleFactor);
                    }
                }
            }

            for (size_t mi = 0; mi < MB_ROWS; ++mi) {
                for (size_t ci = 0; ci < NB_COLS; ++ci) {
                    output[ci + mi * NB_COLS] +=
                        static_cast<float>(outputF16[ci + mi * NB_COLS]) * static_cast<float>(scaleFactor);
                }
            }
        }

        for (size_t mi = 0; mi < MB_ROWS; ++mi) {
            for (size_t ci = 0; ci < nbReal; ++ci) {
                c_ptr[mi * ldc + ni + ci] = output[mi * NB_COLS + ci];
            }
        }
    }
}

static void MNNSpacemitIme2GemmI8I4M1(size_t blk_len, const uint8_t* quant_a_ptr, const uint8_t* quant_b_data,
                                      const uint8_t* quant_b_zp, float* c_ptr, size_t count_m, size_t count_n,
                                      size_t k_blks, size_t ldc) {
    if (quant_b_zp == NULL) {
        for (size_t n = 0; n < count_n; n += 32) {
            size_t nblks = (count_n - n) > 32 ? 32 : count_n - n;
            uint8_t* QuantBDataPtr = (uint8_t*)quant_b_data +       //
                                     n * k_blks * blk_len / 2 +     // b data
                                     n * k_blks * sizeof(_Float16); // scale
            float* CPtr = c_ptr + n;
            size_t cnt = k_blks;

            // A format Version_1 (FP32 SCALE FOR Normal VMADOTins of IME2)
            // A M1K32 int8    256bit
            // Ascale fp32 * 1  32bit
            // || scl*1(fp32) | Asum(int16) | blk0 || scl*1(fp32) | Asum(int16) | blk0 || ...
            // || Element                          || Element                          || ...
            // B format
            // B N8K32 int4    1024bit
            //   4VRF, N32K32, 4096bit
            // Bscale fp16 * N32 512bit;
            // || scl*32..(fp16) | blk0 blk1 ... blk31 || scl*32..(fp16) | blk0 blk1 ... blk31 || ...
            // || Element                              || Element                              || ...
#if 1
            // bias always be nullptr
            __asm__ volatile(

                // t3 = k/32
                "mv           t3, %[BCK]              \n\t"
                "mv           t4, %[NBLKS]            \n\t"
                "mv           s2, %[pA]               \n\t" // s2 = pASCL
                "addi         s3, %[pA], 4+2          \n\t" // s3 = pAData, (pA+AScl+ASum)
                "mv           s4, %[pB]               \n\t" // s4 = pBSCL
                "addi         s5, %[pB], 32*2         \n\t" // s5 = pBdata;
                "mv           s6, %[pC]               \n\t"

                "vsetvli      t0, x0, e32, m1         \n\t"
                "vxor.vv      v2, v0, v0              \n\t" // clear acc

                // ordinary vmadot: vle*6 flw*1 vecIns*21 vmadot*8
                ".align 4                             \n\t"
                "_K_LPST%=:                           \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vl4r.v       v4, (s5)                \n\t" // B Data 4VRF * 8Row * 32
                "addi         s5, s5, 128*4+64        \n\t" // 1024bit

                "vsetvli      t0, x0, e8, mf2         \n\t"
                "vle8.v       v0, (s4)                \n\t" // B Scale 4VRF*8Row*FP16 = 512bit
                "addi         s4, s4, 64+128*4        \n\t"

                "vsetvli      t0, x0, e8, mf4         \n\t"
                "vle8.v       v3, (s3)                \n\t" // A Data M1*K32*int8 = 256bit
                "addi         s3, s3, 32+6            \n\t"

                "flw          f0, (s2)                \n\t" // A Scale fp32
                "lh           t2, 4(s2)               \n\t" // A sum of int16
                "addi         s2, s2, 6+32            \n\t"

                "vsetvli      t0, zero, e8, m1        \n\t"
                "vsrl.vi      v24, v3, 4              \n\t"

                "vnpack4.vv   v8, v3, v3, 3           \n\t" // lo4 of A
                "vnpack4.vv   v10, v24, v24, 3        \n\t" // hi4 of A

                "vsetvli      t0, x0, e32, m1         \n\t"
                "vxor.vv      v16, v16, v16           \n\t"
                "vxor.vv      v18, v16, v16           \n\t"
                "vxor.vv      v20, v16, v16           \n\t"
                "vxor.vv      v22, v16, v16           \n\t"

                "vmadotsu     v16, v10, v4, i4        \n\t" // M0 N0 - N7 INT32(256bit)
                "vmadotsu     v18, v10, v5, i4        \n\t" // M0 N8 - N15
                "vmadotsu     v20, v10, v6, i4        \n\t" // M0 N16 - N23
                "vmadotsu     v22, v10, v7, i4        \n\t" // M0 N24 - N31

                "vsll.vi      v16, v16, 4             \n\t"
                "vsll.vi      v18, v18, 4             \n\t"
                "vsll.vi      v20, v20, 4             \n\t"
                "vsll.vi      v22, v22, 4             \n\t"

                "vmadotu      v16, v8, v4, i4         \n\t"
                "vmadotu      v18, v8, v5, i4         \n\t"
                "vmadotu      v20, v8, v6, i4         \n\t"
                "vmadotu      v22, v8, v7, i4         \n\t"

                "vsetvli      t0, x0, e16, m1         \n\t"
                "vmv.v.i      v28, 8                  \n\t"
                "vpack.vv     v24, v16, v18, 2        \n\t"
                "vpack.vv     v26, v20, v22, 2        \n\t"
                "vpack.vv     v16, v24, v26, 3        \n\t"

                "vwmul.vx     v24, v28, t2            \n\t"
                "vsetvli      t0, x0, e32, m1         \n\t"
                "vadd.vv      v16, v16, v24           \n\t"

                // b_scale fp16 -> fp32
                "vsetvli      t0, x0, e16, mf2        \n\t"
                "vfwcvt.f.f.v v24, v0                 \n\t"
                // mac result i32 -> fp32
                "vsetvli      t0, x0, e32, m1         \n\t"
                "vfcvt.f.x.v  v26, v16                \n\t"
                // a_scale * b_scale;
                "vfmul.vf     v1, v24, f0             \n\t"
                // static_cast<float>(qsum) * a_scale * b_scale;
                "vfmacc.vv    v2, v1, v26             \n\t"

                "addi         t3, t3, -1              \n\t"
                "bgtz         t3, _K_LPST%=           \n\t"
                "_K_LPND%=:                           \n\t"

                //-----------------------------------------
                // STORE Equal 32N-------------------------
                "_ST32%=:                             \n\t"
                "vsetvli      t0, t4, e32, m1         \n\t"
                "vse32.v      v2, (s6)                \n\t" // M0 [N0 : N32]; FP32(1024bit)

                "_FUNC_END%=:                         \n\t"

                :
                : [BCK] "r"(cnt), [NBLKS] "r"(nblks), [pA] "r"(quant_a_ptr), [pB] "r"(QuantBDataPtr), [pC] "r"(CPtr)
                : "cc", "t0", "t2", "t3", "t4", "f0", "s2", "s3", "s4", "s5", "s6");
#else
            __asm__ volatile(

                // t3 = k/32
                "mv           t3, %[BCK]              \n\t"
                "mv           t4, %[NBLKS]            \n\t"
                "vsetvli      t0, x0, e16, m1         \n\t"
                "vmv.v.i      v0, 1                   \n\t" // init the scale
                "mv           s2, %[pA]               \n\t" // s2 = pASCL
                "addi         s3, %[pA], 4+2          \n\t" // s3 = pAData, (pA+AScl+ASum)
                "mv           s4, %[pB]               \n\t" // s4 = pBSCL
                "addi         s5, %[pB], 32*2         \n\t" // s5 = pBdata;
                "mv           s6, %[pC]               \n\t"

                "vsll.vi      v1, v0, 4               \n\t"
                "vxor.vv      v2, v0, v0              \n\t" // clear acc
                "vfcvt.f.x.v  v0, v0                  \n\t"
                "vfcvt.f.x.v  v1, v1                  \n\t"

                // vmadot hp: vle*7 flw*1 vecIns*14 vmadot*8
                ".align 4                             \n\t"
                "_K_LPST%=:                           \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vl4r.v       v4, (s5)                \n\t" // B Data 4VRF * 8Row * 32
                "addi         s5, s5, 128*4+64        \n\t" // 1024bit

                "vsetvli      t0, x0, e8, mf2         \n\t"
                "vle8.v       v30, (s4)               \n\t" // B Scale 4VRF*8Row*FP16 = 512bit
                "addi         s4, s4, 64+128*4        \n\t"

                "vsetvli      t0, x0, e8, mf4         \n\t"
                "vle8.v       v3, (s3)                \n\t" // A Data M1*K32*int8 = 256bit
                "addi         s3, s3, 32+6            \n\t"

                "flw          f0, (s2)                \n\t" // A Scale fp32
                "lh           t2, 4(s2)               \n\t" // A sum of int16
                "addi         s2, s2, 6+32            \n\t"

                "vsetvli      t0, x0, e16, m1         \n\t"
                "vmv.v.i      v28, 8                  \n\t" // Bzp u8 -> u16
                "vsetvli      t0, x0, e8, m1          \n\t"
                "vsrl.vi      v24, v3, 4              \n\t"

                "vsetvli      t0, x0, e16, m1         \n\t"
                "vmul.vx      v26, v28, t2            \n\t" // asum*zp i16*i16
                "vnpack4.vv   v8, v3, v3, 3           \n\t" // lo4 of A
                "vnpack4.vv   v10, v24, v24, 3        \n\t" // hi4 of A

                "vfcvt.f.x.v  v16, v26                \n\t" // zp i16 -> fp16
                "vadd.vi      v18, v16, 0             \n\t"
                "vadd.vi      v20, v16, 0             \n\t"
                "vadd.vi      v22, v16, 0             \n\t"

                "vmadotsu.hp  v16, v10, v4, v1, 0, i4 \n\t" // high 4
                "vmadotsu.hp  v18, v10, v5, v1, 0, i4 \n\t"
                "vmadotsu.hp  v20, v10, v6, v1, 0, i4 \n\t"
                "vmadotsu.hp  v22, v10, v7, v1, 0, i4 \n\t"
                "vmadotu.hp   v16, v8, v4, v0, 0, i4  \n\t" // low 4
                "vmadotu.hp   v18, v8, v5, v0, 0, i4  \n\t"
                "vmadotu.hp   v20, v8, v6, v0, 0, i4  \n\t"
                "vmadotu.hp   v22, v8, v7, v0, 0, i4  \n\t"

                "vpack.vv     v24, v16, v18, 1        \n\t"
                "vpack.vv     v26, v20, v22, 1        \n\t"
                "vpack.vv     v16, v24, v26, 2        \n\t"

                "vsetvli      t0, x0, e16, mf2        \n\t"
                // mac result * b_scale; f16*f16->f32
                "vfwmul.vv     v31, v30, v16          \n\t"

                "vsetvli      t0, x0, e32, m1         \n\t"
                // static_cast<float>(qsum * b_scale) * a_scale;
                "vfmacc.vf    v2, f0, v31             \n\t"

                "addi         t3, t3, -1              \n\t"
                "bgtz         t3, _K_LPST%=           \n\t"
                "_K_LPND%=:                           \n\t"

                //-----------------------------------------
                // STORE Equal 32N-------------------------
                "_ST32%=:                             \n\t"
                "vsetvli      t0, t4, e32, m1         \n\t"
                "vse32.v      v2, (s6)                \n\t" // M0 [N0 : N32]; FP32(1024bit)

                "_FUNC_END%=:                         \n\t"

                :
                : [BCK] "r"(cnt), [NBLKS] "r"(nblks), [pA] "r"(quant_a_ptr), [pB] "r"(QuantBDataPtr), [pC] "r"(CPtr)
                : "cc", "t0", "t2", "t3", "t4", "f0", "s2", "s3", "s4", "s5", "s6");

#endif
        }
    } else {
        for (size_t n = 0; n < count_n; n += 32) {
            size_t nblks = (count_n - n) > 32 ? 32 : count_n - n;
            uint8_t* QuantBDataPtr = (uint8_t*)quant_b_data +       //
                                     n * k_blks * blk_len / 2 +     // b data
                                     n * k_blks * sizeof(uint8_t) + // b zp
                                     n * k_blks * sizeof(_Float16); // scale
            float* CPtr = c_ptr + n;
            size_t cnt = k_blks;

            // A format Version_1 (FP32 SCALE FOR Normal VMADOTins of IME2)
            // A M1K32 int8    256bit
            // Ascale fp32 * 1  32bit
            // || scl*1(fp32) | Asum(int16) | blk0 || scl*1(fp32) | Asum(int16) | blk0 || ...
            // || Element                          || Element                          || ...
            // B format
            // B N8K32 int4    1024bit
            //   4VRF, N32K32, 4096bit
            // Bscale fp16 * N32 512bit;
            // Bzp uint8_t * N32 256bit;
            // || scl*32..(fp16) | zp*32(uint8) | blk0 blk1 ... blk31 || scl*32..(fp16)  ...
            // || Element                                             || Element         ...

            // bias always be nullptr
#if 1
            __asm__ volatile(

                // t3 = k/32
                "mv           t3, %[BCK]              \n\t"
                "mv           t4, %[NBLKS]            \n\t"
                "mv           s2, %[pA]               \n\t" // s2 = pASCL
                "addi         s3, %[pA], 4+2          \n\t" // s3 = pAData, (pA+AScl+ASum)
                "mv           s4, %[pB]               \n\t" // s4 = pBSCL
                "addi         s5, %[pB], 32*3         \n\t" // s5 = pBdata, (pB+BScl+Bzp)
                "mv           s6, %[pC]               \n\t"

                "vsetvli      t0, x0, e32, m1         \n\t"
                "vxor.vv      v2, v0, v0              \n\t" // clear acc

                // ordinary vmadot: vle*6 flw*1 vecIns*21 vmadot*8
                ".align 4                             \n\t"
                "_K_LPST%=:                           \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vl4r.v       v4, (s5)                \n\t" // B Data 4VRF * 8Row * 32
                "addi         s5, s5, 128*4+96        \n\t" // 1024bit

                "vsetvli      t0, x0, e8, mf2         \n\t"
                "vle8.v       v0, (s4)                \n\t" // B Scale 4VRF*8Row*FP16 = 512bit
                "addi         s4, s4, 64              \n\t"

                "vsetvli      t0, x0, e8, mf4         \n\t"
                "vle8.v       v3, (s3)                \n\t" // A Data M1*K32*int8 = 256bit
                "addi         s3, s3, 32+6            \n\t"

                "flw          f0, (s2)                \n\t" // A Scale fp32
                "lh           t2, 4(s2)               \n\t" // A sum of int16
                "addi         s2, s2, 6+32            \n\t"

                "vsetvli      t0, zero, e8, m1        \n\t"
                "vsrl.vi      v24, v3, 4              \n\t"

                "vnpack4.vv   v8, v3, v3, 3           \n\t" // lo4 of A
                "vnpack4.vv   v10, v24, v24, 3        \n\t" // hi4 of A

                "vsetvli      t0, x0, e32, m1         \n\t"
                "vxor.vv      v16, v16, v16           \n\t"
                "vxor.vv      v18, v16, v16           \n\t"
                "vxor.vv      v20, v16, v16           \n\t"
                "vxor.vv      v22, v16, v16           \n\t"

                "vmadotsu     v16, v10, v4, i4        \n\t" // M0 N0 - N7 INT32(256bit)
                "vmadotsu     v18, v10, v5, i4        \n\t" // M0 N8 - N15
                "vmadotsu     v20, v10, v6, i4        \n\t" // M0 N16 - N23
                "vmadotsu     v22, v10, v7, i4        \n\t" // M0 N24 - N31

                "vsll.vi      v16, v16, 4             \n\t"
                "vsll.vi      v18, v18, 4             \n\t"
                "vsll.vi      v20, v20, 4             \n\t"
                "vsll.vi      v22, v22, 4             \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vle8.v       v1, (s4)                \n\t" // Bzp
                "addi         s4, s4, 32+128*4        \n\t"

                "vmadotu      v16, v8, v4, i4         \n\t"
                "vmadotu      v18, v8, v5, i4         \n\t"
                "vmadotu      v20, v8, v6, i4         \n\t"
                "vmadotu      v22, v8, v7, i4         \n\t"

                "vwaddu.vx    v28, v1, x0             \n\t" // uint8 -> uint16
                "vpack.vv     v24, v16, v18, 2        \n\t"
                "vpack.vv     v26, v20, v22, 2        \n\t"
                "vpack.vv     v16, v24, v26, 3        \n\t"

                "vsetvli      t0, x0, e16, m1         \n\t"
                "vwmul.vx     v24, v28, t2            \n\t"
                "vsetvli      t0, x0, e32, m1         \n\t"
                "vadd.vv      v16, v16, v24           \n\t"

                // b_scale fp16 -> fp32
                "vsetvli      t0, x0, e16, mf2        \n\t"
                "vfwcvt.f.f.v v24, v0                 \n\t"
                // mac result i32 -> fp32
                "vsetvli      t0, x0, e32, m1         \n\t"
                "vfcvt.f.x.v  v26, v16                \n\t"
                // a_scale * b_scale;
                "vfmul.vf     v1, v24, f0             \n\t"
                // static_cast<float>(qsum) * a_scale * b_scale;
                "vfmacc.vv    v2, v1, v26             \n\t"

                "addi         t3, t3, -1              \n\t"
                "bgtz         t3, _K_LPST%=           \n\t"
                "_K_LPND%=:                           \n\t"

                //-----------------------------------------
                // STORE Equal 32N-------------------------
                "_ST32%=:                             \n\t"
                "vsetvli      t0, t4, e32, m1         \n\t"
                "vse32.v      v2, (s6)                \n\t" // M0 [N0 : N32]; FP32(1024bit)

                "_FUNC_END%=:                         \n\t"

                :
                : [BCK] "r"(cnt), [NBLKS] "r"(nblks), [pA] "r"(quant_a_ptr), [pB] "r"(QuantBDataPtr), [pC] "r"(CPtr)
                : "cc", "t0", "t2", "t3", "t4", "f0", "s2", "s3", "s4", "s5", "s6");
#else
            __asm__ volatile(

                // t3 = k/32
                "mv           t3, %[BCK]              \n\t"
                "mv           t4, %[NBLKS]            \n\t"
                "vsetvli      t0, x0, e16, m1         \n\t"
                "vmv.v.i      v0, 1                   \n\t" // init the scale
                "mv           s2, %[pA]               \n\t" // s2 = pASCL
                "addi         s3, %[pA], 4+2          \n\t" // s3 = pAData, (pA+AScl+ASum)
                "mv           s4, %[pB]               \n\t" // s4 = pBSCL
                "addi         s5, %[pB], 32*3         \n\t" // s5 = pBdata, (pB+BScl+Bzp)
                "mv           s6, %[pC]               \n\t"

                "vsll.vi      v1, v0, 4               \n\t"
                "vxor.vv      v2, v0, v0              \n\t" // clear acc
                "vfcvt.f.x.v  v0, v0                  \n\t"
                "vfcvt.f.x.v  v1, v1                  \n\t"

                // vmadot hp: vle*6 flw*1 vecIns*14 vmadot*8
                ".align 4                             \n\t"
                "_K_LPST%=:                           \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vl4r.v       v4, (s5)                \n\t" // B Data 4VRF * 8Row * 32
                "addi         s5, s5, 128*4+96        \n\t" // 1024bit

                "vsetvli      t0, x0, e8, mf2         \n\t"
                "vle8.v       v30, (s4)               \n\t" // B Scale 4VRF*8Row*FP16 = 512bit
                "addi         s4, s4, 64              \n\t"

                "vsetvli      t0, x0, e8, mf4         \n\t"
                "vle8.v       v31, (s4)               \n\t" // B zp 32Row*uint8 = 256bit
                "addi         s4, s4, 32+128*4        \n\t"

                "vle8.v       v3, (s3)                \n\t" // A Data M1*K32*int8 = 256bit
                "addi         s3, s3, 32+6            \n\t"

                "flw          f0, (s2)                \n\t" // A Scale fp32
                "lh           t2, 4(s2)               \n\t" // A sum of int16
                "addi         s2, s2, 6+32            \n\t"

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vsrl.vi      v24, v3, 4              \n\t"

                "vsetvli      t0, x0, e16, m1         \n\t"
                "vnpack4.vv   v8, v3, v3, 3           \n\t" // lo4 of A
                "vnpack4.vv   v10, v24, v24, 3        \n\t" // hi4 of A

                "vxor.vv      v16, v16, v16           \n\t"
                "vxor.vv      v18, v16, v16           \n\t"
                "vxor.vv      v20, v16, v16           \n\t"
                "vxor.vv      v22, v16, v16           \n\t"

                "vmadotsu.hp  v16, v10, v4, v1, 0, i4 \n\t" // high 4
                "vmadotsu.hp  v18, v10, v5, v1, 0, i4 \n\t"
                "vmadotsu.hp  v20, v10, v6, v1, 0, i4 \n\t"
                "vmadotsu.hp  v22, v10, v7, v1, 0, i4 \n\t"
                "vmadotu.hp   v16, v8, v4, v0, 0, i4  \n\t" // low 4
                "vmadotu.hp   v18, v8, v5, v0, 0, i4  \n\t"
                "vmadotu.hp   v20, v8, v6, v0, 0, i4  \n\t"
                "vmadotu.hp   v22, v8, v7, v0, 0, i4  \n\t"

                "vsetvli      t0, x0, e8, mf4         \n\t"
                "vwaddu.vx    v28, v31, x0            \n\t" // Bzp u8 -> u16

                "vsetvli      t0, x0, e8, m1          \n\t"
                "vpack.vv     v24, v16, v18, 1        \n\t"
                "vpack.vv     v26, v20, v22, 1        \n\t"
                "vpack.vv     v16, v24, v26, 2        \n\t"

                "vsetvli      t0, x0, e16, mf2        \n\t"
                "vmul.vx      v26, v28, t2            \n\t" // asum*zp i16*i16
                "vfwcvt.f.f.v v22, v30                \n\t" // b_scale fp16 -> fp32
                "vfcvt.f.x.v  v18, v26                \n\t" // zp i16 -> fp16
                "vsetvli      t0, x0, e16, m1         \n\t"
                "vfwadd.vv    v20, v18, v16           \n\t"

                "vsetvli      t0, x0, e32, m1         \n\t"
                // mac result * b_scale; f32*f32->f32
                "vfmul.vv     v31, v22, v20           \n\t"

                "vsetvli      t0, x0, e32, m1         \n\t"
                // static_cast<float>(qsum * b_scale) * a_scale;
                "vfmacc.vf    v2, f0, v31             \n\t"

                "addi         t3, t3, -1              \n\t"
                "bgtz         t3, _K_LPST%=           \n\t"
                "_K_LPND%=:                           \n\t"

                //-----------------------------------------
                // STORE Equal 32N-------------------------
                "_ST32%=:                             \n\t"
                "vsetvli      t0, t4, e32, m1         \n\t"
                "vse32.v      v2, (s6)                \n\t" // M0 [N0 : N32]; FP32(1024bit)

                "_FUNC_END%=:                         \n\t"

                :
                : [BCK] "r"(cnt), [NBLKS] "r"(nblks), [pA] "r"(quant_a_ptr), [pB] "r"(QuantBDataPtr), [pC] "r"(CPtr)
                : "cc", "t0", "t2", "t3", "t4", "f0", "s2", "s3", "s4", "s5", "s6");
#endif
        }
    }
}

static void MNNSpacemitIme2GemmI8I4M4(size_t blk_len, const uint8_t* quant_a_ptr, const uint8_t* quant_b_data,
                                      const uint8_t* quant_b_zp, float* c_ptr, size_t count_m, size_t count_n,
                                      size_t k_blks, size_t ldc) {
    int64_t b_data_stride =
        k_blks * (sizeof(_Float16) + 16 * sizeof(int8_t) + (quant_b_zp != NULL ? sizeof(int8_t) : 0));
    if (quant_b_zp == NULL) {
        for (size_t ni = 0; ni < count_n; ni += 32) {
            uint8_t* b_data = (uint8_t*)quant_b_data + ni * b_data_stride;
            int8_t* a_data = (int8_t*)quant_a_ptr;
            float* dst_c = c_ptr + ni;
#if 0
            asm volatile(
                "li             t1,  8              \n\t"
                "vsetvli        t0, x0, e32, m1     \n\t"
                "vxor.vv        v28, v28, v28       \n\t"
                "vxor.vv        v29, v29, v29       \n\t"
                "vxor.vv        v30, v30, v30       \n\t"
                "vxor.vv        v31, v31, v31       \n\t"
                "mv             t4, %[BK]           \n\t"

                ".align 4                           \n\t"
                "BLK_LOOP%=:                        \n\t"
                // load scale A
                "flw            fa0, (%[A])         \n\t"
                "flw            fa1, 4(%[A])        \n\t"
                "flw            fa2, 8(%[A])        \n\t"
                "flw            fa3, 12(%[A])       \n\t"
                "addi           %[A], %[A], 16      \n\t"

                // load scale B
                "vsetvli        t0, x0, e16, mf2    \n\t"
                "vle16.v        v12, (%[B])         \n\t"
                "addi           %[B], %[B], 64      \n\t"
                "vfwcvt.f.f.v   v14, v12            \n\t"

                "vsetivli       t0, 4, e16, mf2     \n\t"
                "vle16.v        v8, (%[A])          \n\t"  // asum
                "addi           %[A], %[A], 8       \n\t"
                "vwmul.vx       v10, v8, t1         \n\t"  // 8*asum

                "vsetvli        t0, x0, e8, m1      \n\t"
                "vl1r.v         v0, (%[A])          \n\t"
                "addi           %[A], %[A], 128     \n\t"  // 4*32@i8
                "vl4r.v         v4, (%[B])          \n\t"  // 32*32@i4
                "addi           %[B], %[B], 512     \n\t"
                "vsrl.vi        v1, v0, 4           \n\t"
                "vnpack4.vv     v12, v0, v1, 3      \n\t"  // A low  u4
                "vupack.vv      v2, v12, v12, 2     \n\t"

                // init the accumu to asum * zp
                "vsetvli        t0, x0, e32, m1     \n\t"
                "vxor.vv        v16, v16, v16       \n\t"
                "vxor.vv        v18, v16, v16       \n\t"
                "vxor.vv        v20, v16, v16       \n\t"
                "vxor.vv        v22, v16, v16       \n\t"

                // i4 * i4 vmadot
                "vsetvli        t0, x0, e32, m1     \n\t"
                "vmadotsu       v16, v3, v4, i4     \n\t"   // high 4
                "vmadotsu       v18, v3, v5, i4     \n\t"
                "vmadotsu       v20, v3, v6, i4     \n\t"
                "vmadotsu       v22, v3, v7, i4     \n\t"
                "vsll.vi        v16, v16, 4         \n\t"
                "vsll.vi        v18, v18, 4         \n\t"
                "vsll.vi        v20, v20, 4         \n\t"
                "vsll.vi        v22, v22, 4         \n\t"
                "vmadotu        v16, v2, v4, i4     \n\t"   // low 4
                "vmadotu        v18, v2, v5, i4     \n\t"
                "vmadotu        v20, v2, v6, i4     \n\t"
                "vmadotu        v22, v2, v7, i4     \n\t"

                "vpack.vv       v0, v16, v18, 2     \n\t"
                "vpack.vv       v2, v20, v22, 2     \n\t"
                "vpack.vv       v16, v0, v2, 3      \n\t"
                "vpack.vv       v18, v1, v3, 3      \n\t"

                "vrgather.vi    v0, v10, 0          \n\t"
                "vrgather.vi    v1, v10, 1          \n\t"
                "vrgather.vi    v2, v10, 2          \n\t"
                "vrgather.vi    v3, v10, 3          \n\t"

                "vadd.vv        v16, v16, v0        \n\t"
                "vadd.vv        v17, v17, v1        \n\t"
                "vadd.vv        v18, v18, v2        \n\t"
                "vadd.vv        v19, v19, v3        \n\t"

                "vfcvt.f.x.v    v16, v16            \n\t"
                "vfcvt.f.x.v    v17, v17            \n\t"
                "vfcvt.f.x.v    v18, v18            \n\t"
                "vfcvt.f.x.v    v19, v19            \n\t"

                // mul scale
                "vfmul.vv       v16, v16, v14       \n\t"
                "vfmul.vv       v17, v17, v14       \n\t"
                "vfmul.vv       v18, v18, v14       \n\t"
                "vfmul.vv       v19, v19, v14       \n\t"

                "addi           t4, t4, -1          \n\t"
                "vfmacc.vf      v28, fa0, v16       \n\t"
                "vfmacc.vf      v29, fa1, v17       \n\t"
                "vfmacc.vf      v30, fa2, v18       \n\t"
                "vfmacc.vf      v31, fa3, v19       \n\t"

                "bgtz           t4, BLK_LOOP%=      \n\t"

                // save
                "vsetvli        t0, x0, e32, m1     \n\t"
                "add            t2, %[LDC], %[DST]  \n\t"
                "vse32.v        v28, (%[DST])       \n\t"
                "add            t3, %[LDC], t2      \n\t"
                "vse32.v        v29, (t2)           \n\t"
                "add            t2, %[LDC], t3      \n\t"
                "vse32.v        v30, (t3)           \n\t"
                "vse32.v        v31, (t2)           \n\t"
                : [A] "+r"(a_data), [B] "+r"(b_data)
                : [DST] "r"(dst_c), [LDC] "r"(ldc*4), [BK] "r"(k_blks)
                : "t0", "t1", "t2", "t3", "t4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
                  "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
                  "v26", "v27", "v28", "v29", "v30", "v31", "fa0", "fa1", "fa2", "fa3");
#else
            asm volatile(
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vxor.vv        v28, v28, v28           \n\t"
                "vxor.vv        v29, v29, v29           \n\t"
                "vxor.vv        v30, v30, v30           \n\t"
                "vxor.vv        v31, v31, v31           \n\t"
                "vmv.v.i        v0, 1                   \n\t" // init the scale
                "vsll.vi        v1, v0, 4               \n\t"
                "vfcvt.f.x.v    v0, v0                  \n\t"
                "vfcvt.f.x.v    v1, v1                  \n\t"
                "mv             t4, %[BK]               \n\t"

                ".align 4                               \n\t"
                "BLK_LOOP%=:                            \n\t"
                // load scale A
                "flw            fa0, (%[A])             \n\t"
                "flw            fa1, 4(%[A])            \n\t"
                "flw            fa2, 8(%[A])            \n\t"
                "flw            fa3, 12(%[A])           \n\t"
                "addi           %[A], %[A], 16          \n\t"

                // load scale B
                "vsetvli        t0, x0, e16, mf2        \n\t"
                "vle16.v        v12, (%[B])             \n\t"
                "addi           %[B], %[B], 64          \n\t"
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vpack.vv       v14, v12, v12, 3        \n\t"

                "vsetivli       t0, 4, e16, mf2         \n\t"
                "vle16.v        v8, (%[A])              \n\t" // asum
                "addi           %[A], %[A], 8           \n\t"
                "vsll.vi        v8, v8, 3               \n\t" // asum * 8
                "vfcvt.f.x.v    v9, v8                  \n\t"
                "vsetvli        t0, x0, e64, m1         \n\t"
                "vrgather.vi    v10, v9, 0              \n\t"

                "vsetvli        t0, x0, e8, m1          \n\t"
                "vl1r.v         v16, (%[A])             \n\t"
                "addi           %[A], %[A], 128         \n\t" // 4*32@i8
                "vl4r.v         v4, (%[B])              \n\t" // 32*32@i4
                "addi           %[B], %[B], 512         \n\t"
                "vsrl.vi        v17, v16, 4             \n\t"
                "vnpack4.vv     v12, v16, v17, 3        \n\t" // A low  u4
                "vupack.vv      v2, v12, v12, 2         \n\t"

                // init the accumu to asum * zp
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vpack.vv       v16, v10, v10,0         \n\t"
                "vsetvli        t0, x0, e32, m1         \n\t"
                "vpack.vv       v20, v16, v16,0         \n\t"
                "vsetvli        t0, x0, e64, m1         \n\t"
                "vpack.vv       v18, v20, v20, 0        \n\t"
                "vor.vv         v20, v18, v18           \n\t"
                "vor.vv         v21, v18, v18           \n\t"

                // i4 * i4 vmadot
                "vsetvli        t0, x0, e16, m1         \n\t"
                "vmadotsu.hp    v18, v3, v4, v1, 0, i4  \n\t" // high 4
                "vmadotsu.hp    v19, v3, v5, v1, 0, i4  \n\t"
                "vmadotsu.hp    v20, v3, v6, v1, 0, i4  \n\t"
                "vmadotsu.hp    v21, v3, v7, v1, 0, i4  \n\t"
                "vmadotu.hp     v18, v2, v4, v0, 0, i4  \n\t" // low 4
                "vmadotu.hp     v19, v2, v5, v0, 0, i4  \n\t"
                "vmadotu.hp     v20, v2, v6, v0, 0, i4  \n\t"
                "vmadotu.hp     v21, v2, v7, v0, 0, i4  \n\t"

                "vpack.vv       v8, v18, v19, 1         \n\t"
                "vpack.vv       v12, v20, v21, 1        \n\t"
                "vpack.vv       v20, v8, v12, 2         \n\t"

                "vfwmul.vv      v16, v20, v14           \n\t"
                "vfwmul.vv      v18, v21, v14           \n\t"

                "vsetvli        t0, x0, e32, m1         \n\t"

                "addi           t4, t4, -1              \n\t"
                "vfmacc.vf      v28, fa0, v16           \n\t"
                "vfmacc.vf      v29, fa1, v17           \n\t"
                "vfmacc.vf      v30, fa2, v18           \n\t"
                "vfmacc.vf      v31, fa3, v19           \n\t"

                "bgtz           t4, BLK_LOOP%=          \n\t"

                // save
                "vsetvli        t0, x0, e32, m1         \n\t"
                "add            t2, %[LDC], %[DST]      \n\t"
                "vse32.v        v28, (%[DST])           \n\t"
                "add            t3, %[LDC], t2          \n\t"
                "vse32.v        v29, (t2)               \n\t"
                "add            t2, %[LDC], t3          \n\t"
                "vse32.v        v30, (t3)               \n\t"
                "vse32.v        v31, (t2)               \n\t"
                : [A] "+r"(a_data), [B] "+r"(b_data)
                : [DST] "r"(dst_c), [LDC] "r"(ldc * 4), [BK] "r"(k_blks)
                : "t0", "t1", "t2", "t3", "t4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                  "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
                  "v25", "v26", "v27", "v28", "v29", "v30", "v31", "fa0", "fa1", "fa2", "fa3");
#endif
        }
    } else {
        for (size_t ni = 0; ni < count_n; ni += 32) {
            uint8_t* b_data = (uint8_t*)quant_b_data + ni * b_data_stride;
            int8_t* a_data = (int8_t*)quant_a_ptr;
            float* dst_c = c_ptr + ni;

            asm volatile(
                "li             t1,  8          \n\t"
                "vsetvli        t0, x0, e32, m1 \n\t"
                "vxor.vv        v28, v28, v28   \n\t"
                "vxor.vv        v29, v29, v29   \n\t"
                "vxor.vv        v30, v30, v30   \n\t"
                "vxor.vv        v31, v31, v31   \n\t"
                "mv             t4, %[BK]       \n\t"

                ".align 4                        \n\t"
                "BLK_LOOP%=:                     \n\t"
                // load scale A
                "flw            fa0, (%[A])     \n\t"
                "flw            fa1, 4(%[A])    \n\t"
                "flw            fa2, 8(%[A])    \n\t"
                "flw            fa3, 12(%[A])   \n\t"
                "addi           %[A], %[A], 16  \n\t"

                // load scale B
                "vsetvli        t0, x0, e16, mf2\n\t"
                "vle16.v        v12, (%[B])     \n\t"
                "addi           %[B], %[B], 64  \n\t"
                "vfwcvt.f.f.v   v14, v12        \n\t"

                // load zp
                "vsetvli        t0, x0, e8, mf4 \n\t"
                "vle8.v         v8, (%[B])      \n\t"
                "addi           %[B], %[B], 32  \n\t"
                "vwaddu.vx      v10, v8, x0     \n\t"

                // load a sum
                "lh             s1, (%[A])      \n\t"
                "lh             s2, 2(%[A])     \n\t"
                "lh             s3, 4(%[A])     \n\t"
                "lh             s4, 6(%[A])     \n\t"
                "addi           %[A], %[A], 8   \n\t"

                "vsetvli        t0, x0, e8, m1  \n\t"
                "vl1r.v         v0, (%[A])      \n\t"
                "addi           %[A], %[A], 128 \n\t" // 4*32@i8
                "vl4r.v         v4, (%[B])      \n\t" // 32*32@i4
                "addi           %[B], %[B], 512 \n\t"
                "vsrl.vi        v1, v0, 4       \n\t"
                "vnpack4.vv     v12, v0, v1, 3  \n\t" // A low  u4
                "vupack.vv      v2, v12, v12, 2 \n\t"

                // init the accumu to asum * zp
                "vsetvli        t0, x0, e32, m1 \n\t"
                "vxor.vv        v16, v16, v16   \n\t"
                "vxor.vv        v18, v16, v16   \n\t"
                "vxor.vv        v20, v16, v16   \n\t"
                "vxor.vv        v22, v16, v16   \n\t"

                // i4 * i4 vmadot
                "vsetvli        t0, x0, e32, m1 \n\t"
                "vmadotsu       v16, v3, v4, i4 \n\t" // high 4
                "vmadotsu       v18, v3, v5, i4 \n\t"
                "vmadotsu       v20, v3, v6, i4 \n\t"
                "vmadotsu       v22, v3, v7, i4 \n\t"
                "vsll.vi        v16, v16, 4     \n\t"
                "vsll.vi        v18, v18, 4     \n\t"
                "vsll.vi        v20, v20, 4     \n\t"
                "vsll.vi        v22, v22, 4     \n\t"
                "vmadotu        v16, v2, v4, i4 \n\t" // low 4
                "vmadotu        v18, v2, v5, i4 \n\t"
                "vmadotu        v20, v2, v6, i4 \n\t"
                "vmadotu        v22, v2, v7, i4 \n\t"

                "vpack.vv       v0, v16, v18, 2 \n\t"
                "vpack.vv       v2, v20, v22, 2 \n\t"
                "vpack.vv       v16, v0, v2, 3  \n\t"
                "vpack.vv       v18, v1, v3, 3  \n\t"

                "vsetvli        t0, x0, e16, m1 \n\t"
                "vwmul.vx       v0, v10, s1     \n\t"
                "vwmul.vx       v2, v10, s2     \n\t"
                "vwmul.vx       v4, v10, s3     \n\t"
                "vwmul.vx       v6, v10, s4     \n\t"

                "vsetvli        t0, x0, e32, m1 \n\t"
                "vadd.vv        v16, v16, v0    \n\t"
                "vadd.vv        v17, v17, v2    \n\t"
                "vadd.vv        v18, v18, v4    \n\t"
                "vadd.vv        v19, v19, v6    \n\t"

                "vfcvt.f.x.v    v16, v16        \n\t"
                "vfcvt.f.x.v    v17, v17        \n\t"
                "vfcvt.f.x.v    v18, v18        \n\t"
                "vfcvt.f.x.v    v19, v19        \n\t"

                // mul scale
                "vfmul.vv       v16, v16, v14   \n\t"
                "vfmul.vv       v17, v17, v14   \n\t"
                "vfmul.vv       v18, v18, v14   \n\t"
                "vfmul.vv       v19, v19, v14   \n\t"

                "addi           t4, t4, -1      \n\t"
                "vfmacc.vf      v28, fa0, v16   \n\t"
                "vfmacc.vf      v29, fa1, v17   \n\t"
                "vfmacc.vf      v30, fa2, v18   \n\t"
                "vfmacc.vf      v31, fa3, v19   \n\t"

                "bgtz           t4, BLK_LOOP%=  \n\t"

                // save
                "vsetvli        t0, x0, e32, m1 \n\t"
                "add            t2, %[LDC], %[DST]\n\t"
                "vse32.v        v28, (%[DST])   \n\t"
                "add            t3, %[LDC], t2  \n\t"
                "vse32.v        v29, (t2)       \n\t"
                "add            t2, %[LDC], t3  \n\t"
                "vse32.v        v30, (t3)       \n\t"
                "vse32.v        v31, (t2)       \n\t"
                : [A] "+r"(a_data), [B] "+r"(b_data)
                : [DST] "r"(dst_c), [LDC] "r"(ldc * 4), [BK] "r"(k_blks)
                : "t0", "t1", "t2", "t3", "t4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                  "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
                  "v25", "v26", "v27", "v28", "v29", "v30", "v31", "fa0", "fa1", "fa2", "fa3", "s1", "s2", "s3", "s4");
        }
    }
}

static void MNNSpacemitIme2GemmI8I4HpM1AsymPairRef(size_t blk_len, const uint8_t* quant_a_ptr,
                                                   const uint8_t* quant_b_data, const uint8_t* quant_b_zp, float* c_ptr,
                                                   size_t count_m, size_t count_n, size_t k_blks, size_t ldc) {
    (void)count_m;
    (void)ldc;
    if (blk_len != 261 || quant_b_zp != nullptr) {
        return;
    }
    constexpr size_t NB_COLS = 32;
    constexpr size_t A_SUB_STRIDE = sizeof(_Float16) + 32;
    constexpr size_t A_SUPER_STRIDE =
        size_t(256) + size_t(8) * sizeof(_Float16) + size_t(8) * sizeof(_Float16) + sizeof(_Float16);
    constexpr size_t B_SCALE_BYTES = sizeof(_Float16) * NB_COLS;
    constexpr size_t B_Q_BYTES = 16 * NB_COLS;
    constexpr size_t B_PAIR_STRIDE = B_SCALE_BYTES * 2 + B_Q_BYTES * 2;
    constexpr size_t B_SUPER_STRIDE = B_PAIR_STRIDE * 4;
    const size_t b_tile_stride = k_blks * B_SUPER_STRIDE;

    for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
        const size_t nb_real = std::min<size_t>(NB_COLS, count_n - ni);
        const uint8_t* b_tile = quant_b_data + (ni / NB_COLS) * b_tile_stride;
        float output[NB_COLS] = {};
        for (size_t super = 0; super < k_blks; ++super) {
            const uint8_t* a_super = quant_a_ptr + super * A_SUPER_STRIDE;
            const uint8_t* b_super = b_tile + super * B_SUPER_STRIDE;
            const _Float16* a_sums = reinterpret_cast<const _Float16*>(a_super + 8 * A_SUB_STRIDE);
            const float a_average =
                static_cast<float>(*reinterpret_cast<const _Float16*>(a_super + A_SUPER_STRIDE - sizeof(_Float16)));
            // The native vmadot.hp path accumulates all eight K32 dot products of this
            // K256 super-block in FP16, then widens that subtotal into the FP32 output.
            _Float16 dot_output[NB_COLS] = {};
            for (size_t pair = 0; pair < 4; ++pair) {
                const uint8_t* b_pair = b_super + pair * B_PAIR_STRIDE;
                const _Float16* b_scales = reinterpret_cast<const _Float16*>(b_pair);
                const _Float16* b_centered_correction = reinterpret_cast<const _Float16*>(b_pair + B_SCALE_BYTES);
                const uint8_t* b_q[2] = {
                    b_pair + B_SCALE_BYTES * 2,
                    b_pair + B_SCALE_BYTES * 2 + B_Q_BYTES,
                };
                // The two K32 blocks share one block64 correction vector. Form its
                // coefficient in FP32 so large A sums cannot overflow an FP16 product.
                float pair_correction_factor = 0.0f;
                for (size_t sub_in_pair = 0; sub_in_pair < 2; ++sub_in_pair) {
                    const size_t sub = pair * 2 + sub_in_pair;
                    const uint8_t* a_sub = a_super + sub * A_SUB_STRIDE;
                    const _Float16 a_relative_hp = *reinterpret_cast<const _Float16*>(a_sub);
                    const float a_relative = static_cast<float>(a_relative_hp);
                    const int8_t* a_q = reinterpret_cast<const int8_t*>(a_sub + sizeof(_Float16));
                    const float a_sum = static_cast<float>(a_sums[sub]);
                    pair_correction_factor = __builtin_fmaf(a_sum, a_relative, pair_correction_factor);
                    for (size_t col = 0; col < NB_COLS; ++col) {
                        const uint8_t* b_col = b_q[sub_in_pair] + col * 16;
                        int32_t high_dot = 0;
                        int32_t low_dot = 0;
                        for (size_t p = 0; p < 16; ++p) {
                            const uint8_t packed = b_col[p];
                            const uint8_t a0 = static_cast<uint8_t>(a_q[p * 2]);
                            const uint8_t a1 = static_cast<uint8_t>(a_q[p * 2 + 1]);
                            const int32_t a0_high = (a0 & 0x80) != 0 ? static_cast<int32_t>(a0 >> 4) - 16 : a0 >> 4;
                            const int32_t a1_high = (a1 & 0x80) != 0 ? static_cast<int32_t>(a1 >> 4) - 16 : a1 >> 4;
                            high_dot += a0_high * static_cast<int32_t>(packed & 0x0f) +
                                        a1_high * static_cast<int32_t>((packed >> 4) & 0x0f);
                            low_dot += static_cast<int32_t>(a0 & 0x0f) * static_cast<int32_t>(packed & 0x0f) +
                                       static_cast<int32_t>(a1 & 0x0f) * static_cast<int32_t>((packed >> 4) & 0x0f);
                        }
                        const _Float16 scale = static_cast<_Float16>(b_scales[col] * a_relative_hp);
                        const _Float16 high_scale = static_cast<_Float16>(scale * static_cast<_Float16>(16.0f));
                        // vmadot.hp fuses the exact integer subtotal with its FP16 scale and
                        // rounds directly to FP16. FP64 avoids a rare FP32-to-FP16 double round.
                        dot_output[col] = static_cast<_Float16>(__builtin_fma(static_cast<double>(high_dot),
                                                                              static_cast<double>(high_scale),
                                                                              static_cast<double>(dot_output[col])));
                        dot_output[col] = static_cast<_Float16>(__builtin_fma(static_cast<double>(low_dot),
                                                                              static_cast<double>(scale),
                                                                              static_cast<double>(dot_output[col])));
                    }
                }
                pair_correction_factor *= a_average;
                for (size_t col = 0; col < NB_COLS; ++col) {
                    output[col] = __builtin_fmaf(pair_correction_factor, static_cast<float>(b_centered_correction[col]),
                                                 output[col]);
                }
            }
            for (size_t col = 0; col < NB_COLS; ++col) {
                output[col] = __builtin_fmaf(a_average, static_cast<float>(dot_output[col]), output[col]);
            }
        }
        for (size_t col = 0; col < nb_real; ++col) {
            c_ptr[ni + col] = output[col];
        }
    }
}

static void MNNSpacemitIme2GemmI8I4HpM1AsymPair(size_t blk_len, const uint8_t* quant_a_ptr, const uint8_t* quant_b_data,
                                                const uint8_t* quant_b_zp, float* c_ptr, size_t count_m, size_t count_n,
                                                size_t k_blks, size_t ldc) {
    (void)count_m;
    (void)ldc;
    if (blk_len != 261 || quant_b_zp != nullptr) {
        return;
    }
    constexpr size_t NB_COLS = 32;
    constexpr size_t B_PAIR_STRIDE = sizeof(_Float16) * NB_COLS * 2 + 16 * NB_COLS * 2;
    constexpr size_t B_SUPER_STRIDE = B_PAIR_STRIDE * 4;
    const size_t b_tile_stride = k_blks * B_SUPER_STRIDE;
    const _Float16 hp_scale_16 = (_Float16)16.0f;

    for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
        uint8_t* b_data = (uint8_t*)quant_b_data + (ni / NB_COLS) * b_tile_stride;
        uint8_t* a_data = (uint8_t*)quant_a_ptr;
        float* dst_c = c_ptr + ni;

        // B block64 pair:
        //   scale[32] fp16, -weight_bias[32]/8 fp16, q(K32_0)[512], q(K32_1)[512].
        // With A's a_sum=-8*sum(qA), each K32 contributes
        //   da * (dot(qA,qW)*scale + a_sum*(-weight_bias/8)),
        // which is da * dot(qA, qW*scale+weight_bias).
        asm volatile(
            "vsetvli        t0, x0, e16, m1              \n\t"
            "vxor.vv        v31, v31, v31                \n\t"
            "mv             t4, %[BK]                    \n\t"

            ".align 4                                    \n\t"
            "_ASYM_SUPER_LOOP%=:                         \n\t"
            "li             t5, 4                        \n\t"
            "addi           t6, %[A], 272                \n\t"
            "flh            ft1, 288(%[A])               \n\t"
            "fcvt.s.h       ft4, ft1                     \n\t"

            "vsetvli        t0, x0, e16, m1              \n\t"
            "vxor.vv        v16, v16, v16                \n\t"
            "vxor.vv        v17, v17, v17                \n\t"
            "vxor.vv        v18, v18, v18                \n\t"
            "vxor.vv        v19, v19, v19                \n\t"

            ".align 4                                    \n\t"
            "_ASYM_PAIR_LOOP%=:                          \n\t"
            "vsetvli        t0, x0, e16, mf2             \n\t"
            "vle16.v        v8, (%[B])                   \n\t"
            "addi           s5, %[B], 64                 \n\t"
            "vle16.v        v10, (s5)                    \n\t"
            // Keep the shared correction in FP32 across both K32 dot products.
            "vfwcvt.f.f.v   v22, v10                     \n\t"
            "addi           s5, s5, 64                   \n\t"
            "fmv.w.x        fa2, x0                      \n\t"
            "li             t3, 2                        \n\t"

            ".align 4                                    \n\t"
            "_ASYM_SUB_LOOP%=:                           \n\t"
            "flh            fa1, (t6)                    \n\t"
            "addi           t6, t6, 2                    \n\t"
            "flh            ft0, (%[A])                  \n\t"
            "addi           %[A], %[A], 2                \n\t"
            "fcvt.s.h       fa3, fa1                     \n\t"
            "fcvt.s.h       ft2, ft0                     \n\t"
            "fmadd.s        fa2, fa3, ft2, fa2           \n\t"

            "vsetvli        t0, x0, e16, mf2             \n\t"
            "vfmul.vf       v24, v8, ft0                 \n\t"
            "vfmul.vf       v25, v24, %[HP16]            \n\t"

            "vsetvli        t0, x0, e8, m1               \n\t"
            "vpack.vv       v0, v24, v25, 3              \n\t"

            "vsetvli        t0, x0, e8, mf4              \n\t"
            "vle8.v         v3, (%[A])                   \n\t"
            "addi           %[A], %[A], 32               \n\t"
            "vsetvli        t0, x0, e8, m1               \n\t"
            "vl4r.v         v4, (s5)                     \n\t"
            "addi           s5, s5, 512                  \n\t"
            "vsrl.vi        v28, v3, 4                   \n\t"

            "vsetvli        t0, x0, e16, m1              \n\t"
            "vnpack4.vv     v2, v3, v3, 3                \n\t"
            "vnpack4.vv     v3, v28, v28, 3              \n\t"
            "vmadotsu.hp    v16, v3, v4, v0, 4, i4       \n\t"
            "vmadotsu.hp    v17, v3, v5, v0, 5, i4       \n\t"
            "vmadotsu.hp    v18, v3, v6, v0, 6, i4       \n\t"
            "vmadotsu.hp    v19, v3, v7, v0, 7, i4       \n\t"
            "vmadotu.hp     v16, v2, v4, v0, 0, i4       \n\t"
            "vmadotu.hp     v17, v2, v5, v0, 1, i4       \n\t"
            "vmadotu.hp     v18, v2, v6, v0, 2, i4       \n\t"
            "vmadotu.hp     v19, v2, v7, v0, 3, i4       \n\t"

            "addi           t3, t3, -1                   \n\t"
            "bgtz           t3, _ASYM_SUB_LOOP%=         \n\t"
            "fmul.s         fa2, fa2, ft4                \n\t"
            "vsetvli        t0, x0, e32, m1              \n\t"
            "vfmacc.vf      v31, fa2, v22                \n\t"
            "mv             %[B], s5                     \n\t"
            "addi           t5, t5, -1                   \n\t"
            "bgtz           t5, _ASYM_PAIR_LOOP%=        \n\t"

            "vsetvli        t0, x0, e16, m1              \n\t"
            "vpack.vv       v8, v16, v17, 1              \n\t"
            "vpack.vv       v12, v18, v19, 1             \n\t"
            "vpack.vv       v20, v8, v12, 2              \n\t"
            "vsetvli        t0, x0, e16, mf2             \n\t"
            "vfwmacc.vf     v31, ft1, v20                \n\t"

            "addi           %[A], t6, 2                  \n\t"
            "addi           t4, t4, -1                   \n\t"
            "bgtz           t4, _ASYM_SUPER_LOOP%=       \n\t"

            "vsetvli        t0, x0, e32, m1              \n\t"
            "vse32.v        v31, (%[DST])                \n\t"
            : [A] "+r"(a_data), [B] "+r"(b_data)
            : [DST] "r"(dst_c), [BK] "r"(k_blks), [HP16] "f"(hp_scale_16)
            : "cc", "memory", "t0", "t3", "t4", "t5", "t6", "s5", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
              "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
              "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "fa1", "fa2", "fa3", "ft0", "ft1", "ft2", "ft4");
    }
}

static void MNNSpacemitIme2GemmI8I4HpM1(size_t blk_len, const uint8_t* quant_a_ptr, const uint8_t* quant_b_data,
                                        const uint8_t* quant_b_zp, float* c_ptr, size_t count_m, size_t count_n,
                                        size_t k_blks, size_t ldc, bool force_native_hp) {
    (void)count_m;
    (void)ldc;
    if (blk_len != 256 && blk_len != 259) {
        return;
    }
    if ((!force_native_hp && !MNNSpacemitIme2HpM1NativeHpEnabled()) || blk_len != 256 || quant_b_zp != nullptr) {
        constexpr size_t NB_COLS = 32;
        const size_t B_SUPER_STRIDE =
            8 * (sizeof(_Float16) * NB_COLS + 16 * NB_COLS) + (quant_b_zp != nullptr ? 8 * NB_COLS : 0);
        const size_t b_tile_stride = k_blks * B_SUPER_STRIDE;
        const size_t b_zp_skip = quant_b_zp != nullptr ? 8 * NB_COLS : 0;
        const int center_no_zp =
            (blk_len == 259 || MNNSpacemitIme2HpM1CenteredEnabled()) && quant_b_zp == nullptr ? 1 : 0;

        for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
            size_t nblks = (count_n - ni) > NB_COLS ? NB_COLS : count_n - ni;
            uint8_t* a_data = (uint8_t*)quant_a_ptr;
            uint8_t* b_data = (uint8_t*)quant_b_data + (ni / NB_COLS) * b_tile_stride;
            float* dst_c = c_ptr + ni;

            asm volatile(
                "mv             t3, %[BK]               \n\t"
                "mv             t4, %[NBLKS]            \n\t"
                "mv             s2, %[A]                \n\t"
                "mv             s4, %[B]                \n\t"
                "mv             s6, %[DST]              \n\t"

                "vsetvli        t0, x0, e32, m1         \n\t"
                "vxor.vv        v2, v2, v2              \n\t"

                ".align 4                               \n\t"
                "SUPER_LOOP%=:                          \n\t"
                "li             t1, 8                   \n\t"
                "addi           s3, s2, 2               \n\t"
                "addi           s1, s2, 272             \n\t"

                ".align 4                               \n\t"
                "SUB_LOOP%=:                            \n\t"
                "addi           s5, s4, 64              \n\t"

                "vsetvli        t0, x0, e8, m1          \n\t"
                "vl4r.v         v4, (s5)                \n\t"

                "vsetvli        t0, x0, e8, mf2         \n\t"
                "vle8.v         v0, (s4)                \n\t"

                "vsetvli        t0, x0, e8, mf4         \n\t"
                "vle8.v         v3, (s3)                \n\t"

                "vsetvli        t0, zero, e8, m1        \n\t"
                "vsrl.vi        v24, v3, 4              \n\t"

                "vnpack4.vv     v8, v3, v3, 3           \n\t"
                "vnpack4.vv     v10, v24, v24, 3        \n\t"

                "vsetvli        t0, x0, e32, m1         \n\t"
                "vxor.vv        v16, v16, v16           \n\t"
                "vxor.vv        v18, v18, v18           \n\t"
                "vxor.vv        v20, v20, v20           \n\t"
                "vxor.vv        v22, v22, v22           \n\t"

                "vmadotsu       v16, v10, v4, i4        \n\t"
                "vmadotsu       v18, v10, v5, i4        \n\t"
                "vmadotsu       v20, v10, v6, i4        \n\t"
                "vmadotsu       v22, v10, v7, i4        \n\t"

                "vsll.vi        v16, v16, 4             \n\t"
                "vsll.vi        v18, v18, 4             \n\t"
                "vsll.vi        v20, v20, 4             \n\t"
                "vsll.vi        v22, v22, 4             \n\t"

                "vmadotu        v16, v8, v4, i4         \n\t"
                "vmadotu        v18, v8, v5, i4         \n\t"
                "vmadotu        v20, v8, v6, i4         \n\t"
                "vmadotu        v22, v8, v7, i4         \n\t"

                "vsetvli        t0, x0, e16, m1         \n\t"
                "vpack.vv       v24, v16, v18, 2        \n\t"
                "vpack.vv       v26, v20, v22, 2        \n\t"
                "vpack.vv       v16, v24, v26, 3        \n\t"

                "vsetvli        t0, x0, e16, mf2        \n\t"
                "vfwcvt.f.f.v   v24, v0                 \n\t"
                "vsetvli        t0, x0, e32, m1          \n\t"
                "beqz           %[CENTER], _M1_NO_CENTER%= \n\t"
                "lh             t2, (s1)                 \n\t"
                "addi           s1, s1, 2                \n\t"
                "vadd.vx        v16, v16, t2             \n\t"
                "_M1_NO_CENTER%=:                        \n\t"

                "vfcvt.f.x.v    v26, v16                \n\t"
                "vfmacc.vv      v2, v24, v26            \n\t"

                "addi           s3, s3, 34              \n\t"
                "addi           s4, s4, 576             \n\t"
                "addi           t1, t1, -1              \n\t"
                "bgtz           t1, SUB_LOOP%=          \n\t"

                "add            s4, s4, %[ZPSKIP]       \n\t"
                "addi           s2, s2, 290             \n\t"
                "addi           t3, t3, -1              \n\t"
                "bgtz           t3, SUPER_LOOP%=        \n\t"

                "vsetvli        t0, t4, e32, m1         \n\t"
                "vse32.v        v2, (s6)                \n\t"
                :
                : [BK] "r"(k_blks), [NBLKS] "r"(nblks), [A] "r"(a_data), [B] "r"(b_data), [DST] "r"(dst_c),
                  [ZPSKIP] "r"(b_zp_skip), [CENTER] "r"(center_no_zp)
                : "cc", "memory", "t0", "t1", "t2", "t3", "t4", "s1", "s2", "s3", "s4", "s5", "s6", "v0", "v2", "v3",
                  "v4", "v5", "v6", "v7", "v8", "v10", "v16", "v18", "v20", "v22", "v24", "v26");
        }
        return;
    }
    constexpr size_t NB_COLS = 32;
    constexpr size_t B_SUPER_STRIDE = 8 * (sizeof(_Float16) * NB_COLS + 16 * NB_COLS);
    const size_t b_tile_stride = k_blks * B_SUPER_STRIDE;

    for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
        uint8_t* b_data = (uint8_t*)quant_b_data + (ni / NB_COLS) * b_tile_stride;
        int8_t* a_data = (int8_t*)quant_a_ptr;
        float* dst_c = c_ptr + ni;

        asm volatile(
            "vsetvli        t0, x0, e16, m1         \n\t"
            "vxor.vv        v31, v31, v31           \n\t"
            "mv             t4, %[BK]               \n\t"
            "li             t0, 0x4c00              \n\t"
            "fmv.h.x        fa0, t0                 \n\t"

            ".align 4                               \n\t"
            "BLK_LOOP%=:                            \n\t"
            "li             t5, 8                   \n\t"
            "addi           t6, %[A], 288           \n\t"
            "flh            ft1, (t6)               \n\t"
            "addi           t6, %[A], 272           \n\t"

            "vsetvli        t0, x0, e16, m1         \n\t"
            "vxor.vv        v16, v18, v18           \n\t"
            "vxor.vv        v17, v18, v18           \n\t"
            "vxor.vv        v18, v18, v18           \n\t"
            "vxor.vv        v19, v18, v18           \n\t"

            "INNER_BLK_LOOP%=:                      \n\t"
            "flh            fa1, (t6)               \n\t"
            "addi           t6, t6, 2               \n\t"
            "flh            ft0, (%[A])             \n\t"
            "addi           %[A], %[A], 2           \n\t"

            "vsetvli        t0, x0, e8, mf4         \n\t"
            "vle8.v         v3, (%[A])              \n\t"
            "addi           %[A], %[A], 32          \n\t"

            "vsetvli        t0, x0, e16, mf2        \n\t"
            "vle16.v        v8, (%[B])              \n\t"
            "addi           %[B], %[B], 64          \n\t"
            "vl4r.v         v4, (%[B])              \n\t"
            "addi           %[B], %[B], 512         \n\t"
            "vfmul.vf       v8, v8, ft0             \n\t"
            "vfmul.vf       v9, v8, fa0             \n\t"
            "vfmul.vf       v10, v8, fa1            \n\t"
            "vfwmacc.vf     v31, ft1, v10           \n\t"

            "vsetvli        t0, x0, e8, m1          \n\t"
            "vpack.vv       v0, v8, v9, 3           \n\t"
            "vsrl.vi        v28, v3, 4              \n\t"

            "vsetvli        t0, x0, e16, m1         \n\t"
            "vnpack4.vv     v2, v3, v3, 3           \n\t"
            "vnpack4.vv     v3, v28, v28, 3         \n\t"

            "vsetvli        t0, x0, e16, m1         \n\t"
            "vmadotsu.hp    v16, v3, v4, v0, 4, i4  \n\t"
            "vmadotsu.hp    v17, v3, v5, v0, 5, i4  \n\t"
            "vmadotsu.hp    v18, v3, v6, v0, 6, i4  \n\t"
            "vmadotsu.hp    v19, v3, v7, v0, 7, i4  \n\t"
            "vmadotu.hp     v16, v2, v4, v0, 0, i4  \n\t"
            "vmadotu.hp     v17, v2, v5, v0, 1, i4  \n\t"
            "vmadotu.hp     v18, v2, v6, v0, 2, i4  \n\t"
            "vmadotu.hp     v19, v2, v7, v0, 3, i4  \n\t"

            "addi           t5, t5, -1              \n\t"
            "bgtz           t5, INNER_BLK_LOOP%=    \n\t"

            "vpack.vv       v8, v16, v17, 1         \n\t"
            "vpack.vv       v12, v18, v19, 1        \n\t"
            "vpack.vv       v20, v8, v12, 2         \n\t"

            "vsetvli        t0, x0, e16, mf2        \n\t"
            "addi           t4, t4, -1              \n\t"
            "vfwmacc.vf     v31, ft1, v20           \n\t"
            "addi           %[A], t6, 2             \n\t"

            "bgtz           t4, BLK_LOOP%=          \n\t"

            "vsetvli        t0, x0, e32, m1         \n\t"
            "vse32.v        v31, (%[DST])           \n\t"
            : [A] "+r"(a_data), [B] "+r"(b_data)
            : [DST] "r"(dst_c), [BK] "r"(k_blks)
            : "t0", "t1", "t2", "t3", "t4", "t5", "t6", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
              "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24",
              "v25", "v26", "v27", "v28", "v29", "v30", "v31", "fa0", "fa1", "ft0", "ft1", "memory");
    }
}

static void MNNSpacemitIme2GemmI8I4HpM1Residual(size_t blk_len, const uint8_t* quant_a_ptr, const uint8_t* quant_b_data,
                                                const uint8_t* quant_b_zp, float* c_ptr, size_t count_m, size_t count_n,
                                                size_t k_blks, size_t ldc) {
    // Full prefill tiles use the M4 kernel below. The packer stores the final 1-3 rows in the same single-row HP
    // layout used here, so the scheduler can consume them one at a time without repacking or duplicating packed B.
    (void)count_m;
    (void)ldc;
    if (blk_len != 258 || quant_a_ptr == nullptr || quant_b_data == nullptr || quant_b_zp != nullptr ||
        c_ptr == nullptr || count_n == 0 || count_n % 32 != 0 || k_blks == 0) {
        return;
    }
    constexpr size_t NB_COLS = 32;
    constexpr size_t B_SUB_STRIDE = sizeof(_Float16) * NB_COLS + 16 * NB_COLS;
    constexpr size_t B_RESIDUAL_SUB_STRIDE = B_SUB_STRIDE + sizeof(_Float16) * NB_COLS;
    constexpr size_t B_SUPER_STRIDE = 8 * B_RESIDUAL_SUB_STRIDE;
    const size_t b_tile_stride = k_blks * B_SUPER_STRIDE;

    for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
        uint8_t* b_data = (uint8_t*)quant_b_data + (ni / NB_COLS) * b_tile_stride;
        int8_t* a_data = (int8_t*)quant_a_ptr;
        float* dst_c = c_ptr + ni;

        asm volatile(
            "vsetvli        t0, x0, e16, m1         \n\t"
            "vxor.vv        v31, v31, v31           \n\t"
            "mv             t4, %[BK]               \n\t"
            "li             t0, 0x4c00              \n\t"
            "fmv.h.x        fa0, t0                 \n\t"

            ".align 4                               \n\t"
            "_M1R_BLK_LOOP%=:                       \n\t"
            "li             t5, 8                   \n\t"
            "addi           t6, %[A], 288           \n\t"
            "flh            ft1, (t6)               \n\t"
            "addi           t6, %[A], 272           \n\t"

            "vsetvli        t0, x0, e16, m1         \n\t"
            "vxor.vv        v16, v18, v18           \n\t"
            "vxor.vv        v17, v18, v18           \n\t"
            "vxor.vv        v18, v18, v18           \n\t"
            "vxor.vv        v19, v18, v18           \n\t"

            "_M1R_INNER_BLK_LOOP%=:                 \n\t"
            "flh            fa1, (t6)               \n\t"
            "addi           t6, t6, 2               \n\t"
            "flh            ft0, (%[A])             \n\t"
            "addi           %[A], %[A], 2           \n\t"

            "vsetvli        t0, x0, e8, mf4         \n\t"
            "vle8.v         v3, (%[A])              \n\t"
            "addi           %[A], %[A], 32          \n\t"

            "vsetvli        t0, x0, e16, mf2        \n\t"
            "vle16.v        v8, (%[B])              \n\t"
            "addi           %[B], %[B], 64          \n\t"
            "vl4r.v         v4, (%[B])              \n\t"
            "addi           %[B], %[B], 512         \n\t"
            "vle16.v        v12, (%[B])             \n\t"
            "addi           %[B], %[B], 64          \n\t"
            "vfmul.vf       v8, v8, ft0             \n\t"
            "vfmul.vf       v9, v8, fa0             \n\t"
            "vfmul.vf       v10, v8, fa1            \n\t"
            "vfwmacc.vf     v31, ft1, v10           \n\t"
            "vfmul.vf       v10, v12, fa1           \n\t"
            "vfwmacc.vf     v31, ft1, v10           \n\t"

            "vsetvli        t0, x0, e8, m1          \n\t"
            "vpack.vv       v0, v8, v9, 3           \n\t"
            "vsrl.vi        v28, v3, 4              \n\t"

            "vsetvli        t0, x0, e16, m1         \n\t"
            "vnpack4.vv     v2, v3, v3, 3           \n\t"
            "vnpack4.vv     v3, v28, v28, 3         \n\t"
            "vmadotsu.hp    v16, v3, v4, v0, 4, i4  \n\t"
            "vmadotsu.hp    v17, v3, v5, v0, 5, i4  \n\t"
            "vmadotsu.hp    v18, v3, v6, v0, 6, i4  \n\t"
            "vmadotsu.hp    v19, v3, v7, v0, 7, i4  \n\t"
            "vmadotu.hp     v16, v2, v4, v0, 0, i4  \n\t"
            "vmadotu.hp     v17, v2, v5, v0, 1, i4  \n\t"
            "vmadotu.hp     v18, v2, v6, v0, 2, i4  \n\t"
            "vmadotu.hp     v19, v2, v7, v0, 3, i4  \n\t"

            "addi           t5, t5, -1              \n\t"
            "bgtz           t5, _M1R_INNER_BLK_LOOP%= \n\t"
            "vpack.vv       v8, v16, v17, 1         \n\t"
            "vpack.vv       v12, v18, v19, 1        \n\t"
            "vpack.vv       v20, v8, v12, 2         \n\t"

            "vsetvli        t0, x0, e16, mf2        \n\t"
            "addi           t4, t4, -1              \n\t"
            "vfwmacc.vf     v31, ft1, v20           \n\t"
            "addi           %[A], t6, 2             \n\t"
            "bgtz           t4, _M1R_BLK_LOOP%=     \n\t"

            "vsetvli        t0, x0, e32, m1         \n\t"
            "vse32.v        v31, (%[DST])           \n\t"
            : [A] "+r"(a_data), [B] "+r"(b_data)
            : [DST] "r"(dst_c), [BK] "r"(k_blks)
            : "cc", "memory", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22",
              "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "fa0", "fa1", "ft0", "ft1");
    }
}

static void MNNSpacemitIme2GemmI8I4HpM4(size_t blk_len, const uint8_t* quant_a_ptr, const uint8_t* quant_b_data,
                                        const uint8_t* quant_b_zp, float* c_ptr, size_t count_m, size_t count_n,
                                        size_t k_blks, size_t ldc) {
    (void)count_m;
    if (blk_len != 256 && blk_len != 258) {
        return;
    }
    constexpr size_t NB_COLS = 32;
    constexpr size_t B_SUB_STRIDE = sizeof(_Float16) * NB_COLS + 16 * NB_COLS;
    constexpr size_t B_RESIDUAL_SUB_STRIDE = B_SUB_STRIDE + sizeof(_Float16) * NB_COLS;
    const bool fuse_residual = blk_len == 258;
    const size_t B_SUPER_STRIDE =
        fuse_residual ? 8 * B_RESIDUAL_SUB_STRIDE : 8 * B_SUB_STRIDE + (quant_b_zp != nullptr ? 8 * NB_COLS : 0);
    const size_t b_tile_stride = k_blks * B_SUPER_STRIDE;

    if (quant_b_zp != nullptr) {
        if (fuse_residual) {
            return;
        }
        for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
            const size_t nb_real = std::min<size_t>(NB_COLS, count_n - ni);
            if (nb_real != NB_COLS) {
                break;
            }
            uint8_t* b_tile_base = (uint8_t*)quant_b_data + (ni / NB_COLS) * b_tile_stride;
            uint8_t* a_block = (uint8_t*)quant_a_ptr;
            float* dst_c = c_ptr + ni;
            const _Float16 hp_scale_16 = (_Float16)16.0f;
            const _Float16 hp_scale_0125 = (_Float16)0.125f;

            asm volatile(
                "mv             t5, %[BK]                 \n\t"
                "mv             t6, %[A]                  \n\t"
                "mv             s5, %[B]                  \n\t"
                "vsetvli        t0, x0, e32, m1           \n\t"
                "vxor.vv        v28, v28, v28             \n\t"
                "vxor.vv        v29, v29, v29             \n\t"
                "vxor.vv        v30, v30, v30             \n\t"
                "vxor.vv        v31, v31, v31             \n\t"
                "li             t4, 8                     \n\t"
                "li             t1, 4608                  \n\t"
                "addi           t2, t6, 1088              \n\t"
                "add            s6, s5, t1                \n\t"

                ".align 4                                 \n\t"
                "_BLK_LPST%=:                             \n\t"
                "flh            fa1, 64(t2)               \n\t"
                "vsetvli        t0, x0, e32, m1           \n\t"
                "vxor.vv        v18, v30, v30             \n\t"
                "vxor.vv        v19, v31, v31             \n\t"
                "vxor.vv        v20, v30, v30             \n\t"
                "vxor.vv        v21, v31, v31             \n\t"
                "_KsubBLK_LPST%=:                         \n\t"
                "flh            fa0,   0(t6)              \n\t"

                "vsetvli        t0, x0, e16, mf2          \n\t"
                "vle16.v        v12, (s5)                 \n\t"

                "vsetvli        t0, x0, e8, mf4           \n\t"
                "vle8.v         v8, (s6)                  \n\t"

                "fmul.h         fa2, fa0, %[HP16]         \n\t"
                "vfwcvt.f.xu.v  v10, v8                   \n\t"

                "vsetvli        t0, x0, e16, mf2          \n\t"
                "vfmul.vf       v16, v12, fa0             \n\t"
                "vfmul.vf       v17, v12, fa2             \n\t"

                "flh            ft1, 0(t2)                \n\t"
                "flh            ft2, 16(t2)               \n\t"
                "flh            ft3, 32(t2)               \n\t"
                "flh            ft4, 48(t2)               \n\t"

                "fmul.h         ft1, ft1, %[HP0125]       \n\t"
                "fmul.h         ft2, ft2, %[HP0125]       \n\t"
                "fmul.h         ft3, ft3, %[HP0125]       \n\t"
                "fmul.h         ft4, ft4, %[HP0125]       \n\t"

                "addi           t3, t6, 8                 \n\t"
                "vsetvli        t0, x0, e8, m1            \n\t"
                "vl1r.v         v0, (t3)                  \n\t"
                "addi           t3, s5, 64                \n\t"
                "vl4r.v         v4, (t3)                  \n\t"

                "vsetvli        t0, x0, e8, m1            \n\t"
                "vsrl.vi        v1, v0, 4                 \n\t"
                "vnpack4.vv     v12, v0, v1, 3            \n\t"
                "vpack.vv       v0, v17, v16, 3           \n\t"
                "vupack.vv      v2, v12, v12, 2           \n\t"

                "vsetvli        t0, x0, e16, mf2          \n\t"
                "vfmul.vv       v10, v10, v16             \n\t"

                "vsetvli        t0, x0, e16, mf2          \n\t"
                "vfmul.vf       v12, v10, ft1             \n\t"
                "vfmul.vf       v13, v10, ft2             \n\t"
                "vfmul.vf       v24, v10, ft3             \n\t"
                "vfmul.vf       v25, v10, ft4             \n\t"

                "vsetvli        t0, x0, e16, mf2          \n\t"
                "vfwmacc.vf     v28, fa1, v12             \n\t"
                "vfwmacc.vf     v29, fa1, v13             \n\t"
                "vfwmacc.vf     v30, fa1, v24             \n\t"
                "vfwmacc.vf     v31, fa1, v25             \n\t"

                "vsetvli        t0, x0, e32, m1           \n\t"
                "vmadotsu.hp    v18, v3, v4, v0, 0, i4    \n\t"
                "vmadotsu.hp    v19, v3, v5, v0, 1, i4    \n\t"
                "vmadotsu.hp    v20, v3, v6, v0, 2, i4    \n\t"
                "vmadotsu.hp    v21, v3, v7, v0, 3, i4    \n\t"
                "vmadotu.hp     v18, v2, v4, v0, 4, i4    \n\t"
                "vmadotu.hp     v19, v2, v5, v0, 5, i4    \n\t"
                "vmadotu.hp     v20, v2, v6, v0, 6, i4    \n\t"
                "vmadotu.hp     v21, v2, v7, v0, 7, i4    \n\t"

                "addi           t4, t4, -1                \n\t"
                "addi           t6, t6, 8+128             \n\t"
                "addi           t2, t2, 2                 \n\t"
                "addi           s5, s5, 64+512            \n\t"
                "addi           s6, s6, 32                \n\t"
                "bgtz           t4, _KsubBLK_LPST%=       \n\t"

                "vsetvli        t0, x0, e16, m1           \n\t"
                "vpack.vv       v8, v18, v19, 1           \n\t"
                "vpack.vv       v12, v20, v21, 1          \n\t"
                "vpack.vv       v26, v8, v12, 2           \n\t"

                "vsetvli        t0, x0, e16, m1           \n\t"
                "vfwmacc.vf     v28, fa1, v26             \n\t"
                "vfwmacc.vf     v30, fa1, v27             \n\t"

                "li             t4, 8                     \n\t"
                "addi           t5, t5, -1                \n\t"
                "addi           t6, t6, 72                \n\t"
                "mv             s5, s6                    \n\t"
                "addi           t2, t6, 1088              \n\t"
                "add            s6, s5, t1                \n\t"
                "bgtz           t5, _BLK_LPST%=           \n\t"

                "_BLK_LPND%=:                             \n\t"
                "vsetvli        t0, x0, e32, m1           \n\t"
                "add            t2, %[LDC], %[DST]        \n\t"
                "vse32.v        v28, (%[DST])             \n\t"
                "add            t3, %[LDC], t2            \n\t"
                "vse32.v        v29, (t2)                 \n\t"
                "add            t2, %[LDC], t3            \n\t"
                "vse32.v        v30, (t3)                 \n\t"
                "vse32.v        v31, (t2)                 \n\t"
                : [A] "+r"(a_block), [B] "+r"(b_tile_base)
                : [DST] "r"(dst_c), [LDC] "r"(ldc * 4), [BK] "r"(k_blks), [HP16] "f"(hp_scale_16),
                  [HP0125] "f"(hp_scale_0125)
                : "t0", "t1", "t2", "t3", "t4", "t5", "t6", "s5", "s6", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                  "v8", "v10", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v24",
                  "v25", "v26", "v27", "v28", "v29", "v30", "v31", "fa0", "fa1", "fa2", "ft1", "ft2", "ft3", "ft4",
                  "memory");
        }
        return;
    }

    if (fuse_residual) {
        for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
            uint8_t* b_tile_base = (uint8_t*)quant_b_data + (ni / NB_COLS) * b_tile_stride;
            uint8_t* a_block = (uint8_t*)quant_a_ptr;
            float* dst_c = c_ptr + ni;
            const _Float16 hp_scale_16 = (_Float16)16.0f;

            asm volatile(
                "mv             t5, %[BK]                 \n\t"
                "mv             t6, %[A]                  \n\t"
                "mv             s5, %[B]                  \n\t"
                "vsetvli        t0, x0, e32, m1           \n\t"
                "vxor.vv        v28, v28, v28             \n\t"
                "vxor.vv        v29, v29, v29             \n\t"
                "vxor.vv        v30, v30, v30             \n\t"
                "vxor.vv        v31, v31, v31             \n\t"
                "li             t4, 8                     \n\t"
                "addi           t2, t6, 1088              \n\t"

                ".align 4                                 \n\t"
                "_FR_BLK_LPST%=:                          \n\t"
                "flh            fa1, 64(t2)               \n\t"
                "vsetvli        t0, x0, e32, m1           \n\t"
                "vxor.vv        v18, v30, v30             \n\t"
                "vxor.vv        v19, v31, v31             \n\t"
                "vxor.vv        v20, v30, v30             \n\t"
                "vxor.vv        v21, v31, v31             \n\t"
                "_FR_KsubBLK_LPST%=:                      \n\t"
                "flh            fa0,   0(t6)              \n\t"

                "vsetvli        t0, x0, e16, mf2          \n\t"
                "vle16.v        v12, (s5)                 \n\t"

                "fmul.h         fa2, fa0, %[HP16]         \n\t"

                "vsetvli        t0, x0, e16, mf2          \n\t"
                "vfmul.vf       v16, v12, fa0             \n\t"
                "vfmul.vf       v17, v12, fa2             \n\t"

                "flh            ft1, 0(t2)                \n\t"
                "flh            ft2, 16(t2)               \n\t"
                "flh            ft3, 32(t2)               \n\t"
                "flh            ft4, 48(t2)               \n\t"

                "addi           t3, t6, 8                 \n\t"
                "vsetvli        t0, x0, e8, m1            \n\t"
                "vl1r.v         v0, (t3)                  \n\t"
                "addi           t3, s5, 64                \n\t"
                "vl4r.v         v4, (t3)                  \n\t"
                "addi           t3, s5, 576               \n\t"
                "vsetvli        t0, x0, e16, mf2          \n\t"
                "vle16.v        v14, (t3)                 \n\t"

                "vsetvli        t0, x0, e8, m1            \n\t"
                "vsrl.vi        v1, v0, 4                 \n\t"
                "vnpack4.vv     v12, v0, v1, 3            \n\t"
                "vpack.vv       v0, v17, v16, 3           \n\t"
                "vupack.vv      v2, v12, v12, 2           \n\t"

                "vsetvli        t0, x0, e16, mf2          \n\t"
                "vfmul.vf       v12, v16, ft1             \n\t"
                "vfmul.vf       v13, v16, ft2             \n\t"
                "vfmul.vf       v24, v16, ft3             \n\t"
                "vfmul.vf       v25, v16, ft4             \n\t"

                "vsetvli        t0, x0, e16, mf2          \n\t"
                "vfwmacc.vf     v28, fa1, v12             \n\t"
                "vfwmacc.vf     v29, fa1, v13             \n\t"
                "vfwmacc.vf     v30, fa1, v24             \n\t"
                "vfwmacc.vf     v31, fa1, v25             \n\t"

                "vsetvli        t0, x0, e16, mf2          \n\t"
                "vfmul.vf       v12, v14, ft1             \n\t"
                "vfmul.vf       v13, v14, ft2             \n\t"
                "vfmul.vf       v24, v14, ft3             \n\t"
                "vfmul.vf       v25, v14, ft4             \n\t"

                "vsetvli        t0, x0, e16, mf2          \n\t"
                "vfwmacc.vf     v28, fa1, v12             \n\t"
                "vfwmacc.vf     v29, fa1, v13             \n\t"
                "vfwmacc.vf     v30, fa1, v24             \n\t"
                "vfwmacc.vf     v31, fa1, v25             \n\t"

                "vsetvli        t0, x0, e32, m1           \n\t"
                "vmadotsu.hp    v18, v3, v4, v0, 0, i4    \n\t"
                "vmadotsu.hp    v19, v3, v5, v0, 1, i4    \n\t"
                "vmadotsu.hp    v20, v3, v6, v0, 2, i4    \n\t"
                "vmadotsu.hp    v21, v3, v7, v0, 3, i4    \n\t"
                "vmadotu.hp     v18, v2, v4, v0, 4, i4    \n\t"
                "vmadotu.hp     v19, v2, v5, v0, 5, i4    \n\t"
                "vmadotu.hp     v20, v2, v6, v0, 6, i4    \n\t"
                "vmadotu.hp     v21, v2, v7, v0, 7, i4    \n\t"

                "addi           t4, t4, -1                \n\t"
                "addi           t6, t6, 8+128             \n\t"
                "addi           t2, t2, 2                 \n\t"
                "addi           s5, s5, 64+512+64         \n\t"
                "bgtz           t4, _FR_KsubBLK_LPST%=    \n\t"

                "vsetvli        t0, x0, e16, m1           \n\t"
                "vpack.vv       v8, v18, v19, 1           \n\t"
                "vpack.vv       v12, v20, v21, 1          \n\t"
                "vpack.vv       v26, v8, v12, 2           \n\t"

                "vsetvli        t0, x0, e16, m1           \n\t"
                "vfwmacc.vf     v28, fa1, v26             \n\t"
                "vfwmacc.vf     v30, fa1, v27             \n\t"

                "li             t4, 8                     \n\t"
                "addi           t5, t5, -1                \n\t"
                "addi           t6, t6, 72                \n\t"
                "addi           t2, t6, 1088              \n\t"
                "bgtz           t5, _FR_BLK_LPST%=        \n\t"

                "_FR_BLK_LPND%=:                          \n\t"
                "vsetvli        t0, x0, e32, m1           \n\t"
                "add            t2, %[LDC], %[DST]        \n\t"
                "vse32.v        v28, (%[DST])             \n\t"
                "add            t3, %[LDC], t2            \n\t"
                "vse32.v        v29, (t2)                 \n\t"
                "add            t2, %[LDC], t3            \n\t"
                "vse32.v        v30, (t3)                 \n\t"
                "vse32.v        v31, (t2)                 \n\t"
                : [A] "+r"(a_block), [B] "+r"(b_tile_base)
                : [DST] "r"(dst_c), [LDC] "r"(ldc * 4), [BK] "r"(k_blks), [HP16] "f"(hp_scale_16)
                : "t0", "t2", "t3", "t4", "t5", "t6", "s5", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v10",
                  "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v24", "v25", "v26",
                  "v27", "v28", "v29", "v30", "v31", "fa0", "fa1", "fa2", "ft1", "ft2", "ft3", "ft4", "memory");
        }
        return;
    }

    for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
        uint8_t* b_tile_base = (uint8_t*)quant_b_data + (ni / NB_COLS) * b_tile_stride;
        uint8_t* a_block = (uint8_t*)quant_a_ptr;
        float* dst_c = c_ptr + ni;
        const _Float16 hp_scale_16 = (_Float16)16.0f;
        const _Float16 hp_scale_1 = (_Float16)1.0f;

        asm volatile(
            "mv             t5, %[BK]                 \n\t"
            "mv             t6, %[A]                  \n\t"
            "mv             s5, %[B]                  \n\t"
            "vsetvli        t0, x0, e32, m1           \n\t"
            "vxor.vv        v28, v28, v28             \n\t"
            "vxor.vv        v29, v29, v29             \n\t"
            "vxor.vv        v30, v30, v30             \n\t"
            "vxor.vv        v31, v31, v31             \n\t"
            "li             t4, 8                     \n\t"
            "addi           t2, t6, 1088              \n\t"

            ".align 4                                 \n\t"
            "_BLK_LPST%=:                             \n\t"
            "flh            fa1, 64(t2)               \n\t"
            "vsetvli        t0, x0, e32, m1           \n\t"
            "vxor.vv        v18, v30, v30             \n\t"
            "vxor.vv        v19, v31, v31             \n\t"
            "vxor.vv        v20, v30, v30             \n\t"
            "vxor.vv        v21, v31, v31             \n\t"
            "_KsubBLK_LPST%=:                         \n\t"
            "flh            fa0,   0(t6)              \n\t"

            "vsetvli        t0, x0, e16, mf2          \n\t"
            "vle16.v        v12, (s5)                 \n\t"

            "fmul.h         fa2, fa0, %[HP16]         \n\t"

            "vsetvli        t0, x0, e16, mf2          \n\t"
            "vfmul.vf       v16, v12, fa0             \n\t"
            "vfmul.vf       v17, v12, fa2             \n\t"

            "flh            ft1, 0(t2)                \n\t"
            "flh            ft2, 16(t2)               \n\t"
            "flh            ft3, 32(t2)               \n\t"
            "flh            ft4, 48(t2)               \n\t"

            "addi           t3, t6, 8                 \n\t"
            "vsetvli        t0, x0, e8, m1            \n\t"
            "vl1r.v         v0, (t3)                  \n\t"
            "addi           t3, s5, 64                \n\t"
            "vl4r.v         v4, (t3)                  \n\t"

            "vsetvli        t0, x0, e8, m1            \n\t"
            "vsrl.vi        v1, v0, 4                 \n\t"
            "vnpack4.vv     v12, v0, v1, 3            \n\t"
            "vpack.vv       v0, v17, v16, 3           \n\t"
            "vupack.vv      v2, v12, v12, 2           \n\t"

            "vsetvli        t0, x0, e16, mf2          \n\t"
            "vfmul.vf       v12, v16, ft1             \n\t"
            "vfmul.vf       v13, v16, ft2             \n\t"
            "vfmul.vf       v24, v16, ft3             \n\t"
            "vfmul.vf       v25, v16, ft4             \n\t"

            "vsetvli        t0, x0, e16, mf2          \n\t"
            "vfwmacc.vf     v28, fa1, v12             \n\t"
            "vfwmacc.vf     v29, fa1, v13             \n\t"
            "vfwmacc.vf     v30, fa1, v24             \n\t"
            "vfwmacc.vf     v31, fa1, v25             \n\t"

            "vsetvli        t0, x0, e32, m1           \n\t"
            "vmadotsu.hp    v18, v3, v4, v0, 0, i4    \n\t"
            "vmadotsu.hp    v19, v3, v5, v0, 1, i4    \n\t"
            "vmadotsu.hp    v20, v3, v6, v0, 2, i4    \n\t"
            "vmadotsu.hp    v21, v3, v7, v0, 3, i4    \n\t"
            "vmadotu.hp     v18, v2, v4, v0, 4, i4    \n\t"
            "vmadotu.hp     v19, v2, v5, v0, 5, i4    \n\t"
            "vmadotu.hp     v20, v2, v6, v0, 6, i4    \n\t"
            "vmadotu.hp     v21, v2, v7, v0, 7, i4    \n\t"

            "addi           t4, t4, -1                \n\t"
            "addi           t6, t6, 8+128             \n\t"
            "addi           t2, t2, 2                 \n\t"
            "addi           s5, s5, 64+512            \n\t"
            "bgtz           t4, _KsubBLK_LPST%=       \n\t"

            "vsetvli        t0, x0, e16, m1           \n\t"
            "vpack.vv       v8, v18, v19, 1           \n\t"
            "vpack.vv       v12, v20, v21, 1          \n\t"
            "vpack.vv       v26, v8, v12, 2           \n\t"

            "vsetvli        t0, x0, e16, m1           \n\t"
            "vfwmacc.vf     v28, fa1, v26             \n\t"
            "vfwmacc.vf     v30, fa1, v27             \n\t"

            "li             t4, 8                     \n\t"
            "addi           t5, t5, -1                \n\t"
            "addi           t6, t6, 72                \n\t"
            "addi           t2, t6, 1088              \n\t"
            "bgtz           t5, _BLK_LPST%=           \n\t"

            "_BLK_LPND%=:                             \n\t"
            "vsetvli        t0, x0, e32, m1           \n\t"
            "add            t2, %[LDC], %[DST]        \n\t"
            "vse32.v        v28, (%[DST])             \n\t"
            "add            t3, %[LDC], t2            \n\t"
            "vse32.v        v29, (t2)                 \n\t"
            "add            t2, %[LDC], t3            \n\t"
            "vse32.v        v30, (t3)                 \n\t"
            "vse32.v        v31, (t2)                 \n\t"
            : [A] "+r"(a_block), [B] "+r"(b_tile_base)
            : [DST] "r"(dst_c), [LDC] "r"(ldc * 4), [BK] "r"(k_blks), [HP16] "f"(hp_scale_16), [HP1] "f"(hp_scale_1)
            : "t0", "t2", "t3", "t4", "t5", "t6", "s5", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v10",
              "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v24", "v25", "v26", "v27",
              "v28", "v29", "v30", "v31", "fa0", "fa1", "fa2", "ft1", "ft2", "ft3", "ft4", "memory");
    }
}

// blk258 already folds the asymmetric block64 residual into v28-v31. Finish scale/bias/clamp while those
// four row vectors are live, then slide each C4 group directly to [depthQuad][row][channel].
static void MNNSpacemitIme2GemmI8I4HpM4DirectC4(const uint8_t* quant_a_ptr, const uint8_t* quant_b_data, int8_t* dst,
                                                size_t dst_step, size_t count_n, size_t k_blks,
                                                const float* input_scale, const float* bias, float fp32_min,
                                                float fp32_max, bool need_clamp) {
    constexpr size_t NB_COLS = 32;
    constexpr size_t B_SUB_STRIDE = sizeof(_Float16) * NB_COLS + 16 * NB_COLS;
    constexpr size_t B_RESIDUAL_SUB_STRIDE = B_SUB_STRIDE + sizeof(_Float16) * NB_COLS;
    constexpr size_t B_SUPER_STRIDE = 8 * B_RESIDUAL_SUB_STRIDE;
    const size_t b_tile_stride = k_blks * B_SUPER_STRIDE;
    const _Float16 hp_scale_16 = (_Float16)16.0f;

    for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
        uint8_t* b_tile_base = (uint8_t*)quant_b_data + (ni / NB_COLS) * b_tile_stride;
        uint8_t* a_block = (uint8_t*)quant_a_ptr;
        int8_t* direct_dst = dst + (ni / 4) * dst_step;
        const float* bias_tile = bias == nullptr ? nullptr : bias + ni;

        asm volatile(
            "mv             t5, %[BK]                 \n\t"
            "mv             t6, %[A]                  \n\t"
            "mv             s5, %[B]                  \n\t"
            "vsetvli        t0, x0, e32, m1           \n\t"
            "vxor.vv        v28, v28, v28             \n\t"
            "vxor.vv        v29, v29, v29             \n\t"
            "vxor.vv        v30, v30, v30             \n\t"
            "vxor.vv        v31, v31, v31             \n\t"
            "li             t4, 8                     \n\t"
            "addi           t2, t6, 1088              \n\t"

            ".align 4                                 \n\t"
            "_D4_BLK_LPST%=:                          \n\t"
            "flh            fa1, 64(t2)               \n\t"
            "vsetvli        t0, x0, e32, m1           \n\t"
            "vxor.vv        v18, v30, v30             \n\t"
            "vxor.vv        v19, v31, v31             \n\t"
            "vxor.vv        v20, v30, v30             \n\t"
            "vxor.vv        v21, v31, v31             \n\t"
            "_D4_KSUB_LPST%=:                         \n\t"
            "flh            fa0, 0(t6)                \n\t"

            "vsetvli        t0, x0, e16, mf2          \n\t"
            "vle16.v        v12, (s5)                 \n\t"
            "fmul.h         fa2, fa0, %[HP16]         \n\t"
            "vfmul.vf       v16, v12, fa0             \n\t"
            "vfmul.vf       v17, v12, fa2             \n\t"

            "flh            ft1, 0(t2)                \n\t"
            "flh            ft2, 16(t2)               \n\t"
            "flh            ft3, 32(t2)               \n\t"
            "flh            ft4, 48(t2)               \n\t"

            "addi           t3, t6, 8                 \n\t"
            "vsetvli        t0, x0, e8, m1            \n\t"
            "vl1r.v         v0, (t3)                  \n\t"
            "addi           t3, s5, 64                \n\t"
            "vl4r.v         v4, (t3)                  \n\t"
            "addi           t3, s5, 576               \n\t"
            "vsetvli        t0, x0, e16, mf2          \n\t"
            "vle16.v        v14, (t3)                 \n\t"

            "vsetvli        t0, x0, e8, m1            \n\t"
            "vsrl.vi        v1, v0, 4                 \n\t"
            "vnpack4.vv     v12, v0, v1, 3            \n\t"
            "vpack.vv       v0, v17, v16, 3           \n\t"
            "vupack.vv      v2, v12, v12, 2           \n\t"

            "vsetvli        t0, x0, e16, mf2          \n\t"
            "vfmul.vf       v12, v16, ft1             \n\t"
            "vfmul.vf       v13, v16, ft2             \n\t"
            "vfmul.vf       v24, v16, ft3             \n\t"
            "vfmul.vf       v25, v16, ft4             \n\t"
            "vfwmacc.vf     v28, fa1, v12             \n\t"
            "vfwmacc.vf     v29, fa1, v13             \n\t"
            "vfwmacc.vf     v30, fa1, v24             \n\t"
            "vfwmacc.vf     v31, fa1, v25             \n\t"

            "vfmul.vf       v12, v14, ft1             \n\t"
            "vfmul.vf       v13, v14, ft2             \n\t"
            "vfmul.vf       v24, v14, ft3             \n\t"
            "vfmul.vf       v25, v14, ft4             \n\t"
            "vfwmacc.vf     v28, fa1, v12             \n\t"
            "vfwmacc.vf     v29, fa1, v13             \n\t"
            "vfwmacc.vf     v30, fa1, v24             \n\t"
            "vfwmacc.vf     v31, fa1, v25             \n\t"

            "vsetvli        t0, x0, e32, m1           \n\t"
            "vmadotsu.hp    v18, v3, v4, v0, 0, i4    \n\t"
            "vmadotsu.hp    v19, v3, v5, v0, 1, i4    \n\t"
            "vmadotsu.hp    v20, v3, v6, v0, 2, i4    \n\t"
            "vmadotsu.hp    v21, v3, v7, v0, 3, i4    \n\t"
            "vmadotu.hp     v18, v2, v4, v0, 4, i4    \n\t"
            "vmadotu.hp     v19, v2, v5, v0, 5, i4    \n\t"
            "vmadotu.hp     v20, v2, v6, v0, 6, i4    \n\t"
            "vmadotu.hp     v21, v2, v7, v0, 7, i4    \n\t"

            "addi           t4, t4, -1                \n\t"
            "addi           t6, t6, 8+128             \n\t"
            "addi           t2, t2, 2                 \n\t"
            "addi           s5, s5, 64+512+64         \n\t"
            "bgtz           t4, _D4_KSUB_LPST%=       \n\t"

            "vsetvli        t0, x0, e16, m1           \n\t"
            "vpack.vv       v8, v18, v19, 1           \n\t"
            "vpack.vv       v12, v20, v21, 1          \n\t"
            "vpack.vv       v26, v8, v12, 2           \n\t"
            "vfwmacc.vf     v28, fa1, v26             \n\t"
            "vfwmacc.vf     v30, fa1, v27             \n\t"

            "li             t4, 8                     \n\t"
            "addi           t5, t5, -1                \n\t"
            "addi           t6, t6, 72                \n\t"
            "addi           t2, t6, 1088              \n\t"
            "bgtz           t5, _D4_BLK_LPST%=        \n\t"

            "vsetvli        t0, x0, e32, m1           \n\t"
            "flw            ft1, 0(%[ISCALE])         \n\t"
            "flw            ft2, 4(%[ISCALE])         \n\t"
            "flw            ft3, 8(%[ISCALE])         \n\t"
            "flw            ft4, 12(%[ISCALE])        \n\t"
            "vfmul.vf       v28, v28, ft1             \n\t"
            "vfmul.vf       v29, v29, ft2             \n\t"
            "vfmul.vf       v30, v30, ft3             \n\t"
            "vfmul.vf       v31, v31, ft4             \n\t"
            "beqz           %[BIAS], _D4_NO_BIAS%=    \n\t"
            "vle32.v        v16, (%[BIAS])            \n\t"
            "vfadd.vv       v28, v28, v16             \n\t"
            "vfadd.vv       v29, v29, v16             \n\t"
            "vfadd.vv       v30, v30, v16             \n\t"
            "vfadd.vv       v31, v31, v16             \n\t"
            "_D4_NO_BIAS%=:                            \n\t"
            "beqz           %[CLAMP], _D4_NO_CLAMP%=  \n\t"
            "vfmax.vf       v28, v28, %[FMIN]         \n\t"
            "vfmax.vf       v29, v29, %[FMIN]         \n\t"
            "vfmax.vf       v30, v30, %[FMIN]         \n\t"
            "vfmax.vf       v31, v31, %[FMIN]         \n\t"
            "vfmin.vf       v28, v28, %[FMAX]         \n\t"
            "vfmin.vf       v29, v29, %[FMAX]         \n\t"
            "vfmin.vf       v30, v30, %[FMAX]         \n\t"
            "vfmin.vf       v31, v31, %[FMAX]         \n\t"
            "_D4_NO_CLAMP%=:                           \n\t"

            "mv             t2, %[DST]                \n\t"
            "li             t4, 0                     \n\t"
            "li             t5, 8                     \n\t"
            "vsetivli       t0, 4, e32, m1            \n\t"
            "_D4_C4_STORE%=:                           \n\t"
            "vslidedown.vx  v16, v28, t4              \n\t"
            "vslidedown.vx  v17, v29, t4              \n\t"
            "vslidedown.vx  v18, v30, t4              \n\t"
            "vslidedown.vx  v19, v31, t4              \n\t"
            "vse32.v        v16, (t2)                 \n\t"
            "addi           t3, t2, 16                \n\t"
            "vse32.v        v17, (t3)                 \n\t"
            "addi           t3, t3, 16                \n\t"
            "vse32.v        v18, (t3)                 \n\t"
            "addi           t3, t3, 16                \n\t"
            "vse32.v        v19, (t3)                 \n\t"
            "add            t2, t2, %[DSTEP]          \n\t"
            "addi           t4, t4, 4                 \n\t"
            "addi           t5, t5, -1                \n\t"
            "bgtz           t5, _D4_C4_STORE%=        \n\t"
            : [A] "+r"(a_block), [B] "+r"(b_tile_base)
            : [DST] "r"(direct_dst), [DSTEP] "r"(dst_step), [BK] "r"(k_blks), [HP16] "f"(hp_scale_16),
              [ISCALE] "r"(input_scale), [BIAS] "r"(bias_tile), [CLAMP] "r"(need_clamp ? 1 : 0), [FMIN] "f"(fp32_min),
              [FMAX] "f"(fp32_max)
            : "t0", "t2", "t3", "t4", "t5", "t6", "s5", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v10",
              "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v24", "v25", "v26", "v27",
              "v28", "v29", "v30", "v31", "fa0", "fa1", "fa2", "ft1", "ft2", "ft3", "ft4", "memory");
    }
}

static void MNNSpacemitIme2GemmI8I4HpM4FixedA(size_t blk_len, const uint8_t* quant_a_ptr, const uint8_t* quant_b_data,
                                              const uint8_t* quant_b_zp, float* c_ptr, size_t count_m, size_t count_n,
                                              size_t k_blks, size_t ldc) {
    (void)count_m;
    if (blk_len != 260 || quant_b_zp != nullptr) {
        return;
    }
    constexpr size_t NB_COLS = 32;
    constexpr size_t B_SUB_STRIDE = sizeof(_Float16) * NB_COLS + 16 * NB_COLS;
    const size_t B_SUPER_STRIDE = 8 * B_SUB_STRIDE;
    const size_t b_tile_stride = k_blks * B_SUPER_STRIDE;

    for (size_t ni = 0; ni < count_n; ni += NB_COLS) {
        uint8_t* b_tile_base = (uint8_t*)quant_b_data + (ni / NB_COLS) * b_tile_stride;
        uint8_t* a_block = (uint8_t*)quant_a_ptr;
        float* dst_c = c_ptr + ni;
        const _Float16 hp_scale_16 = (_Float16)16.0f;
        const _Float16 hp_scale_1 = (_Float16)1.0f;

        asm volatile(
            "mv             t5, %[BK]                 \n\t"
            "mv             t6, %[A]                  \n\t"
            "mv             s5, %[B]                  \n\t"
            "vsetvli        t0, x0, e32, m1           \n\t"
            "vxor.vv        v28, v28, v28             \n\t"
            "vxor.vv        v29, v29, v29             \n\t"
            "vxor.vv        v30, v30, v30             \n\t"
            "vxor.vv        v31, v31, v31             \n\t"
            "li             t4, 8                     \n\t"
            "addi           t2, t6, 1088              \n\t"

            ".align 4                                 \n\t"
            "_FIXA_BLK_LPST%=:                        \n\t"
            "vsetvli        t0, x0, e32, m1           \n\t"
            "vxor.vv        v18, v30, v30             \n\t"
            "vxor.vv        v19, v31, v31             \n\t"
            "vxor.vv        v20, v30, v30             \n\t"
            "vxor.vv        v21, v31, v31             \n\t"
            "_FIXA_KsubBLK_LPST%=:                    \n\t"

            "vsetvli        t0, x0, e16, mf2          \n\t"
            "vle16.v        v12, (s5)                 \n\t"
            "vfmul.vf       v17, v12, %[HP16]         \n\t"
            "flh            ft1, 0(t2)                \n\t"
            "flh            ft2, 16(t2)               \n\t"
            "flh            ft3, 32(t2)               \n\t"
            "flh            ft4, 48(t2)               \n\t"

            "addi           t3, t6, 8                 \n\t"
            "vsetvli        t0, x0, e8, m1            \n\t"
            "vl1r.v         v0, (t3)                  \n\t"
            "addi           t3, s5, 64                \n\t"
            "vl4r.v         v4, (t3)                  \n\t"

            "vsetvli        t0, x0, e8, m1            \n\t"
            "vsrl.vi        v1, v0, 4                 \n\t"
            "vnpack4.vv     v13, v0, v1, 3            \n\t"
            "vpack.vv       v0, v17, v12, 3           \n\t"
            "vupack.vv      v2, v13, v13, 2           \n\t"

            "vsetvli        t0, x0, e16, mf2          \n\t"
            "vfmul.vf       v13, v12, ft1             \n\t"
            "vfmul.vf       v14, v12, ft2             \n\t"
            "vfmul.vf       v15, v12, ft3             \n\t"
            "vfmul.vf       v16, v12, ft4             \n\t"
            "vfwmacc.vf     v28, %[HP1], v13          \n\t"
            "vfwmacc.vf     v29, %[HP1], v14          \n\t"
            "vfwmacc.vf     v30, %[HP1], v15          \n\t"
            "vfwmacc.vf     v31, %[HP1], v16          \n\t"

            "vsetvli        t0, x0, e32, m1           \n\t"
            "vmadotsu.hp    v18, v3, v4, v0, 0, i4    \n\t"
            "vmadotsu.hp    v19, v3, v5, v0, 1, i4    \n\t"
            "vmadotsu.hp    v20, v3, v6, v0, 2, i4    \n\t"
            "vmadotsu.hp    v21, v3, v7, v0, 3, i4    \n\t"
            "vmadotu.hp     v18, v2, v4, v0, 4, i4    \n\t"
            "vmadotu.hp     v19, v2, v5, v0, 5, i4    \n\t"
            "vmadotu.hp     v20, v2, v6, v0, 6, i4    \n\t"
            "vmadotu.hp     v21, v2, v7, v0, 7, i4    \n\t"

            "addi           t4, t4, -1                \n\t"
            "addi           t6, t6, 8+128             \n\t"
            "addi           t2, t2, 2                 \n\t"
            "addi           s5, s5, 64+512            \n\t"
            "bgtz           t4, _FIXA_KsubBLK_LPST%=  \n\t"

            "vsetvli        t0, x0, e16, m1           \n\t"
            "vpack.vv       v8, v18, v19, 1           \n\t"
            "vpack.vv       v12, v20, v21, 1          \n\t"
            "vpack.vv       v26, v8, v12, 2           \n\t"

            "vsetvli        t0, x0, e16, m1           \n\t"
            "vfwmacc.vf     v28, %[HP1], v26          \n\t"
            "vfwmacc.vf     v30, %[HP1], v27          \n\t"

            "li             t4, 8                     \n\t"
            "addi           t5, t5, -1                \n\t"
            "addi           t6, t6, 72                \n\t"
            "addi           t2, t6, 1088              \n\t"
            "bgtz           t5, _FIXA_BLK_LPST%=      \n\t"

            "vsetvli        t0, x0, e32, m1           \n\t"
            "add            t2, %[LDC], %[DST]        \n\t"
            "vse32.v        v28, (%[DST])             \n\t"
            "add            t3, %[LDC], t2            \n\t"
            "vse32.v        v29, (t2)                 \n\t"
            "add            t2, %[LDC], t3            \n\t"
            "vse32.v        v30, (t3)                 \n\t"
            "vse32.v        v31, (t2)                 \n\t"
            : [A] "+r"(a_block), [B] "+r"(b_tile_base)
            : [DST] "r"(dst_c), [LDC] "r"(ldc * 4), [BK] "r"(k_blks), [HP16] "f"(hp_scale_16), [HP1] "f"(hp_scale_1)
            : "t0", "t2", "t3", "t4", "t5", "t6", "s5", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v10",
              "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v24", "v25", "v26", "v27",
              "v28", "v29", "v30", "v31", "memory");
    }
}

} // namespace

extern "C" __attribute__((aligned(64))) size_t MNNSpacemitIme2GemmI8I4HpM4DirectC4Local(
    size_t blkLen, const uint8_t* quantAPtr, const uint8_t* quantBData, const uint8_t* quantBZp, int8_t* dst,
    size_t dstStep, size_t countM, size_t countN, size_t kBlocks, const float* inputScale, const float* bias,
    float fp32Min, float fp32Max, int needClamp) {
    if (blkLen != 258 || quantAPtr == nullptr || quantBData == nullptr || quantBZp != nullptr || dst == nullptr ||
        countM != 4 || dstStep < countM * 4 * sizeof(float) || countN == 0 || countN % 32 != 0 || kBlocks == 0 ||
        inputScale == nullptr || MNNSpacemitIme2Vlenb() != 128) {
        return 0;
    }
    MNNSpacemitIme2GemmI8I4HpM4DirectC4(quantAPtr, quantBData, dst, dstStep, countN, kBlocks, inputScale, bias, fp32Min,
                                        fp32Max, needClamp != 0);
    return 4;
}

extern "C" __attribute__((aligned(64))) size_t MNNSpacemitIme2GemmI8I4HpM1NativeLocal(
    size_t blkLen, const uint8_t* quantAPtr, const uint8_t* quantBData, const uint8_t* quantBZp, float* cPtr,
    size_t countM, size_t countN, size_t kBlocks, size_t ldc) {
    if ((blkLen != 256 && blkLen != 261) || quantAPtr == nullptr || quantBData == nullptr || quantBZp != nullptr ||
        cPtr == nullptr || countM != 1 || countN == 0 || countN % 32 != 0 || kBlocks == 0) {
        return 0;
    }
    if (blkLen == 261) {
        const bool useRef = MNNSpacemitIme2HpRefEnabled();
        const size_t vlenb = MNNSpacemitIme2Vlenb();
        // The N32 IME2 assembly is intentionally a VLENB=128 AI-core kernel. Keep the scalar
        // oracle usable on every core, but fail closed if a native job was dispatched elsewhere.
        if (!useRef && vlenb != 128) {
            return 0;
        }
        if (useRef) {
            MNNSpacemitIme2GemmI8I4HpM1AsymPairRef(blkLen, quantAPtr, quantBData, nullptr, cPtr, countM, countN,
                                                   kBlocks, ldc);
        } else {
            MNNSpacemitIme2GemmI8I4HpM1AsymPair(blkLen, quantAPtr, quantBData, nullptr, cPtr, countM, countN, kBlocks,
                                                ldc);
        }
    } else {
        MNNSpacemitIme2GemmI8I4HpM1(blkLen, quantAPtr, quantBData, nullptr, cPtr, countM, countN, kBlocks, ldc, true);
    }
    return 1;
}

// Keep the K3 A100 IME2 entry on a stable cache-line boundary across link-layout changes.
extern "C" __attribute__((aligned(64))) size_t MNNSpacemitIme2GemmI8I4Local(size_t blkLen, const uint8_t* quantAPtr,
                                                                            const uint8_t* quantBData,
                                                                            const uint8_t* quantBZp, float* cPtr,
                                                                            size_t countM, size_t countN,
                                                                            size_t kBlocks, size_t ldc) {
    if (blkLen == 257) {
        if (countM >= 4 && quantBZp == nullptr) {
            if (MNNSpacemitIme2A4RefEnabled()) {
                MNNSpacemitIme2GemmI4I4HpRef<4>(blkLen, quantAPtr, quantBData, quantBZp, cPtr, countM, countN, kBlocks,
                                                ldc);
            } else {
                MNNSpacemitIme2GemmI4I4HpM4(blkLen, quantAPtr, quantBData, quantBZp, cPtr, countM, countN, kBlocks,
                                            ldc);
            }
            return 4;
        }
        return 0;
    }
    if (blkLen == 258) {
        if (countM >= 4 && quantBZp == nullptr) {
            MNNSpacemitIme2GemmI8I4HpM4(blkLen, quantAPtr, quantBData, quantBZp, cPtr, countM, countN, kBlocks, ldc);
            return 4;
        }
        if (countM > 0 && quantBZp == nullptr) {
            MNNSpacemitIme2GemmI8I4HpM1Residual(blkLen, quantAPtr, quantBData, quantBZp, cPtr, countM, countN, kBlocks,
                                                ldc);
            return 1;
        }
        return 0;
    }
    if (blkLen == 260) {
        if (countM >= 4 && quantBZp == nullptr && MNNSpacemitIme2FixedAScaleEnabled()) {
            MNNSpacemitIme2GemmI8I4HpM4FixedA(blkLen, quantAPtr, quantBData, quantBZp, cPtr, countM, countN, kBlocks,
                                              ldc);
            return 4;
        }
        return 0;
    }
    if (blkLen == 261) {
        if (quantAPtr == nullptr || quantBData == nullptr || quantBZp != nullptr || cPtr == nullptr || countM != 1 ||
            countN == 0 || countN % 32 != 0 || kBlocks == 0) {
            return 0;
        }
        const bool useRef = MNNSpacemitIme2HpRefEnabled();
        if (!useRef && MNNSpacemitIme2Vlenb() != 128) {
            return 0;
        }
        if (useRef) {
            MNNSpacemitIme2GemmI8I4HpM1AsymPairRef(blkLen, quantAPtr, quantBData, quantBZp, cPtr, countM, countN,
                                                   kBlocks, ldc);
        } else {
            MNNSpacemitIme2GemmI8I4HpM1AsymPair(blkLen, quantAPtr, quantBData, quantBZp, cPtr, countM, countN, kBlocks,
                                                ldc);
        }
        return 1;
    }
    if (blkLen == 256 || blkLen == 259) {
        if (MNNSpacemitIme2HpRefEnabled()) {
            if (blkLen == 259) {
                return 0;
            }
            if (countM >= 4) {
                MNNSpacemitIme2GemmI8I4HpRef<4>(blkLen, quantAPtr, quantBData, quantBZp, cPtr, countM, countN, kBlocks,
                                                ldc);
                return 4;
            }
            MNNSpacemitIme2GemmI8I4HpRef<1>(blkLen, quantAPtr, quantBData, quantBZp, cPtr, countM, countN, kBlocks,
                                            ldc);
            return 1;
        }
        if (countM >= 4) {
            if (blkLen == 259) {
                return 0;
            }
            MNNSpacemitIme2GemmI8I4HpM4(blkLen, quantAPtr, quantBData, quantBZp, cPtr, countM, countN, kBlocks, ldc);
            return 4;
        }
        MNNSpacemitIme2GemmI8I4HpM1(blkLen, quantAPtr, quantBData, quantBZp, cPtr, countM, countN, kBlocks, ldc, false);
        return 1;
    }
    if (countM >= 4) {
        MNNSpacemitIme2GemmI8I4M4(blkLen, quantAPtr, quantBData, quantBZp, cPtr, countM, countN, kBlocks, ldc);
        return 4;
    }
    MNNSpacemitIme2GemmI8I4M1(blkLen, quantAPtr, quantBData, quantBZp, cPtr, countM, countN, kBlocks, ldc);
    return 1;
}

#endif // defined(MNN_USE_SPACEMIT_IME2)
