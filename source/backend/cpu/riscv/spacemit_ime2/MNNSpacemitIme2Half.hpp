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
// One K64 x N32 tile. Each byte stores the same K position for columns c and c+16.
constexpr size_t kSharedBlock64Kernel = 262;
struct Block64Weight {
    // Preserve the integer kernel's FP16 conversion and -bias/8 correction.
    uint16_t decodeScale[32];
    uint16_t decodeCorrection[32];
    // Projection rounds the original bias and scale to native half. In particular
    // decodeScale rounds midpoint mantissas up, while native RNE rounds ties to even.
    _Float16 scale[32];
    _Float16 offset[32];
    uint8_t quants[8][128]; // K8 panel, column modulo 16, K modulo 8
};
static_assert(sizeof(Block64Weight) == 1280, "unexpected IME2 block64 layout");
static_assert(offsetof(Block64Weight, quants) == 256 && sizeof(Block64Weight) % 128 == 0,
              "each K8 quant panel must start on a whole-register boundary");
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
size_t MNNIme2HalfProjectionBytes(size_t rows, size_t depth, size_t cols, size_t workers);
// workspace must hold MNNIme2HalfProjectionBytes() bytes for the same shape and worker count.
bool MNNIme2HalfProjection(const float* src, float* dst, const float* bias, const float* clamp, const uint8_t* weight,
                           size_t rows, size_t depth, size_t cols, size_t workers, void* workspace);
bool MNNIme2CausalAttention(const float* q, const float* const* keys, const float* const* values, float* out,
                            size_t len, size_t dim, size_t heads, size_t kv, size_t workers, float scale);
void MNNIme2MaskedSoftmax(float* scores, int valid, int length, float scale);
#endif

bool MNNIme2GatedPrefill(float* state, const float* q, const float* k, float* v, float* projection, float* output,
                         const float* decay, const float* beta, size_t stride, void* scratch, size_t scratchBytes);
