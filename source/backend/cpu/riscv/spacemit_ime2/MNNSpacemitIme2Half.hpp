//
//  MNNSpacemitIme2Half.hpp
//  MNN
//
//  Created by MNN on 2026/11/18.
//  Copyright © 2026, Alibaba Group Holding Limited
//

#pragma once
#include <cstddef>
#include <cstdint>

namespace MNNIme2 {
// One K64 x N32 tile shared by decode and prefill. Weights are centered signed
// 4-bit values (qW - 8) packed in native IME2 i4 K32 order (two 512-byte panels).
constexpr size_t kSharedBlock64Kernel = 262;
struct Block64Weight {
    // [0..31] = scale.
    _Float16 scales[32];
    // Centered offset: offset + 8 * scale (shared by prefill and decode).
    _Float16 centeredOffset[32];
    // Two K32 panels, each 4 column-groups of 8 cols x 16 packed bytes.
    uint8_t quants[2][512];
};
static_assert(sizeof(Block64Weight) == 1152, "unexpected IME2 block64 layout");
static_assert(offsetof(Block64Weight, quants) == 128 && sizeof(Block64Weight) % 128 == 0,
              "each K32 quant panel must start on a whole-register boundary");
} // namespace MNNIme2

#if defined(__riscv_zvfh)
#include <cstdlib>
#include <memory>
namespace MNNIme2 {
inline bool halfPrefillEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("MNN_CPU_DISABLE_IME2_HALF_PREFILL");
        return !(value && value[0] == '1' && value[1] == 0);
    }();
    return enabled;
}
inline std::unique_ptr<_Float16, decltype(&std::free)> halfWorkspace(size_t count, size_t alignment) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, count * sizeof(_Float16)) != 0) {
        ptr = nullptr;
    }
    return {static_cast<_Float16*>(ptr), &std::free};
}

// RISC-V fflags bits 2 (OF) and 4 (NV) are sticky and per hart. Check the
// actual narrowing operations, including stores of split-K intermediates.
// Merge incoming sticky flags back into the worker's FP environment on exit.
class HalfRangeCheck {
public:
    HalfRangeCheck() { asm volatile("csrrci %0, fflags, 20" : "=r"(mPrevious) :: "memory"); }
    bool failed() const {
        unsigned flags;
        asm volatile("frflags %0" : "=r"(flags) :: "memory");
        return (flags & 20u) != 0;
    }
    ~HalfRangeCheck() { asm volatile("csrs fflags, %0" :: "r"(mPrevious) : "memory"); }
private:
    unsigned mPrevious;
};

// A=[K/8,16,8], B=[K/8,32,8]. A nonzero byte pitch writes FP32 rows;
// zero writes one FP16 tile in the matrix accumulator's native order.
void matmulTile(size_t depth, const _Float16* b, const _Float16* a, void* out, size_t pitch = 0);
} // namespace MNNIme2
// Borrowed immutable panel. The LinearResource retains its cache entry across
// clones and outlives the synchronous TCM dispatch that consumes this pointer.
const uint8_t* MNNIme2GetBlock64Weight(void* context, const int8_t* weight, size_t depth, size_t columns);
// Optional thread-local cache to reuse packed activations across FusedLinear children.
struct MNNIme2ActivationCache {
    const float* src = nullptr;
    size_t rows = 0;
    size_t depth = 0;
    void* data = nullptr;
};
MNNIme2ActivationCache* MNNIme2SwapActivationCache(MNNIme2ActivationCache* next);
size_t MNNIme2HalfProjectionBytes(size_t rows, size_t depth, size_t cols, size_t workers);
// workspace must hold MNNIme2HalfProjectionBytes() bytes for the same shape and worker count.
bool MNNIme2HalfProjection(const float* src, float* dst, const float* bias, const float* clamp, const uint8_t* weight,
                           size_t rows, size_t depth, size_t cols, size_t workers, void* workspace);
bool MNNIme2HalfProjectionFromFp16(const void* srcFp16, float* dst, const float* bias, const float* clamp,
                                   const uint8_t* weight, size_t rows, size_t depth, size_t cols, size_t workers,
                                   void* workspace);
bool MNNIme2HalfFusedGateUp(const float* src, float* dst, const float* biasGate, const float* clampGate,
                            const uint8_t* weightGate, const float* biasUp, const float* clampUp,
                            const uint8_t* weightUp, size_t rows, size_t depth, size_t cols, size_t workers,
                            void* workspace);
const void* MNNIme2HalfFusedGateUpToFp16(const float* src, const float* biasGate, const float* clampGate,
                                         const uint8_t* weightGate, const float* biasUp, const float* clampUp,
                                         const uint8_t* weightUp, size_t rows, size_t depth, size_t cols,
                                         size_t workers, void* workspace);
void MNNIme2ConvertTiledFp16ToC4(const void* srcFp16, float* dst, size_t rows, size_t depth);
bool MNNIme2CausalAttention(const float* q, const float* const* keys, const float* const* values, float* out,
                            size_t len, size_t dim, size_t heads, size_t kv, size_t workers, float scale);
void MNNIme2MaskedSoftmax(float* scores, int valid, int length, float scale);
void MNNIme2OnlineSoftmaxBlock(float* row, int valid, int length, float scale, float& runningMax, float& runningSum,
                               float& alpha);
#endif

bool MNNIme2GatedPrefill(float* state, const float* q, const float* k, float* v, float* projection, float* output,
                         const float* decay, const float* beta, size_t stride, void* scratch, size_t scratchBytes);
bool MNNIme2GatedPrefill64(float* state, const float* q, const float* k, const float* v, float* projK, float* output,
                           const float* decay, const float* beta, size_t stride, void* scratch, size_t scratchBytes,
                           size_t validCount = 64);
