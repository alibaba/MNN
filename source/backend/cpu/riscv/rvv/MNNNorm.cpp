#include "MNNRvvPrefill.hpp"
#include "RVVBackend.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/CPULayerNorm.hpp"
#include "core/Concurrency.h"
#include "core/OpCommonUtils.hpp"
#include "core/TensorUtils.hpp"
#include <riscv_vector.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <math.h>

#ifdef MNN_USE_SPACEMIT_IME2
extern "C" size_t MNNSpacemitIme2RunTcmTasks(size_t, void (*)(size_t, void*, size_t, void*), void*);
extern "C" size_t MNNSpacemitIme2RunDecodeTasks(size_t, void (*)(size_t, void*, size_t, void*), void*);
#endif

void MNNNorm_RVV(float* dst, const float* src, const float* gamma, const float* beta, float epsilon, size_t size,
                 bool RMSNorm) {
    float mean = 0.0f;
    if (!RMSNorm) {
        size_t n = size;
        const float* p = src;
        vfloat32m1_t vAcc = __riscv_vfmv_s_f_f32m1(0.0f, 1);
        while (n > 0) {
            size_t vl = __riscv_vsetvl_e32m4(n);
            vfloat32m4_t v = __riscv_vle32_v_f32m4(p, vl);
            vAcc = __riscv_vfredusum_vs_f32m4_f32m1(v, vAcc, vl);
            p += vl;
            n -= vl;
        }
        mean = __riscv_vfmv_f_s_f32m1_f32(vAcc) / (float)size;
    }

    // Phase 1: square_sum reduction
    size_t n = size;
    const float* srcPtr = src;
    vfloat32m1_t vSqSum = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    if (RMSNorm) {
        while (n > 0) {
            size_t vl = __riscv_vsetvl_e32m4(n);
            vfloat32m4_t vSrc = __riscv_vle32_v_f32m4(srcPtr, vl);
            vfloat32m4_t vSq = __riscv_vfmul_vv_f32m4(vSrc, vSrc, vl);
            vSqSum = __riscv_vfredusum_vs_f32m4_f32m1(vSq, vSqSum, vl);
            srcPtr += vl;
            n -= vl;
        }
    } else {
        while (n > 0) {
            size_t vl = __riscv_vsetvl_e32m4(n);
            vfloat32m4_t vSrc = __riscv_vle32_v_f32m4(srcPtr, vl);
            vfloat32m4_t vDiff = __riscv_vfsub_vf_f32m4(vSrc, mean, vl);
            vfloat32m4_t vSq = __riscv_vfmul_vv_f32m4(vDiff, vDiff, vl);
            vSqSum = __riscv_vfredusum_vs_f32m4_f32m1(vSq, vSqSum, vl);
            srcPtr += vl;
            n -= vl;
        }
    }
    float square_sum = __riscv_vfmv_f_s_f32m1_f32(vSqSum);

    // Phase 2: rsqrt (scalar)
    float variable = 1.0f / sqrtf(square_sum / (float)size + epsilon);

    // Phase 3: normalize + scale
    n = size;
    srcPtr = src;
    float* dstPtr = dst;
    if (RMSNorm && gamma && !beta) {
        const float* gammaPtr = gamma;
        while (n > 0) {
            size_t vl = __riscv_vsetvl_e32m4(n);
            vfloat32m4_t vSrc = __riscv_vle32_v_f32m4(srcPtr, vl);
            vfloat32m4_t vG = __riscv_vle32_v_f32m4(gammaPtr, vl);
            vfloat32m4_t vNorm = __riscv_vfmul_vf_f32m4(vSrc, variable, vl);
            vNorm = __riscv_vfmul_vv_f32m4(vNorm, vG, vl);
            __riscv_vse32_v_f32m4(dstPtr, vNorm, vl);
            srcPtr += vl;
            gammaPtr += vl;
            dstPtr += vl;
            n -= vl;
        }
    } else if (gamma && beta) {
        const float* gammaPtr = gamma;
        const float* betaPtr = beta;
        while (n > 0) {
            size_t vl = __riscv_vsetvl_e32m4(n);
            vfloat32m4_t vSrc = __riscv_vle32_v_f32m4(srcPtr, vl);
            vfloat32m4_t vG = __riscv_vle32_v_f32m4(gammaPtr, vl);
            vfloat32m4_t vB = __riscv_vle32_v_f32m4(betaPtr, vl);
            vfloat32m4_t vNorm = __riscv_vfsub_vf_f32m4(vSrc, mean, vl);
            vNorm = __riscv_vfmul_vf_f32m4(vNorm, variable, vl);
            vNorm = __riscv_vfmadd_vv_f32m4(vNorm, vG, vB, vl);
            __riscv_vse32_v_f32m4(dstPtr, vNorm, vl);
            srcPtr += vl;
            gammaPtr += vl;
            betaPtr += vl;
            dstPtr += vl;
            n -= vl;
        }
    } else if (gamma) {
        const float* gammaPtr = gamma;
        while (n > 0) {
            size_t vl = __riscv_vsetvl_e32m4(n);
            vfloat32m4_t vSrc = __riscv_vle32_v_f32m4(srcPtr, vl);
            vfloat32m4_t vG = __riscv_vle32_v_f32m4(gammaPtr, vl);
            vfloat32m4_t vNorm = __riscv_vfsub_vf_f32m4(vSrc, mean, vl);
            vNorm = __riscv_vfmul_vf_f32m4(vNorm, variable, vl);
            vNorm = __riscv_vfmul_vv_f32m4(vNorm, vG, vl);
            __riscv_vse32_v_f32m4(dstPtr, vNorm, vl);
            srcPtr += vl;
            gammaPtr += vl;
            dstPtr += vl;
            n -= vl;
        }
    } else {
        while (n > 0) {
            size_t vl = __riscv_vsetvl_e32m4(n);
            vfloat32m4_t vSrc = __riscv_vle32_v_f32m4(srcPtr, vl);
            vfloat32m4_t vNorm = __riscv_vfsub_vf_f32m4(vSrc, mean, vl);
            vNorm = __riscv_vfmul_vf_f32m4(vNorm, variable, vl);
            __riscv_vse32_v_f32m4(dstPtr, vNorm, vl);
            srcPtr += vl;
            dstPtr += vl;
            n -= vl;
        }
    }
}

// Each lane accumulates one NC4HW4 lane across channels. Adjacent four-lane
// groups are independent tokens, including when the residual is an output.
void MNNRmsNormC4_RVV(float* out, float* sum, const float* src, const float* residual, const float* gamma,
                      const float* beta, float epsilon, size_t rows, size_t cols, int id, int threads) {
    MNN_ASSERT(threads > 0);
    if (threads <= 0) return;
    const size_t tile = __riscv_vsetvlmax_e32m4() / 4;
    for (size_t row = id * tile; row < rows; row += threads * tile) {
        const size_t count = (rows - row < tile ? rows - row : tile) * 4;
        const auto lane = __riscv_vid_v_u32m4(count);
        auto squared = __riscv_vfmv_v_f_f32m4(0, count);
        for (size_t c = 0; c < cols; c += 4) {
            const size_t offset = c * rows + row * 4;
            auto x = __riscv_vle32_v_f32m4(src + offset, count);
            if (residual) {
                x = __riscv_vfadd_vv_f32m4(x, __riscv_vle32_v_f32m4(residual + offset, count), count);
                __riscv_vse32_v_f32m4(sum + offset, x, count);
            }
            squared = __riscv_vfmacc_vv_f32m4(squared, x, x, count);
        }
        squared = __riscv_vfadd_vv_f32m4(squared, __riscv_vslidedown_vx_f32m4(squared, 1, count), count);
        squared = __riscv_vfadd_vv_f32m4(squared, __riscv_vslidedown_vx_f32m4(squared, 2, count), count);
        auto rowSum = __riscv_vrgather_vv_f32m4(squared, __riscv_vand_vx_u32m4(lane, ~3u, count), count);
        auto denom = __riscv_vfadd_vf_f32m4(__riscv_vfdiv_vf_f32m4(rowSum, float(cols), count), epsilon, count);
        auto inv = __riscv_vfrdiv_vf_f32m4(__riscv_vfsqrt_v_f32m4(denom, count), 1, count);
        auto chan = __riscv_vand_vx_u32m4(lane, 3, count);
        const float* base = residual ? sum : src;
        for (size_t c = 0; c < cols; c += 4) {
            const size_t offset = c * rows + row * 4;
            auto x = __riscv_vfmul_vv_f32m4(__riscv_vle32_v_f32m4(base + offset, count), inv, count);
            if (gamma && beta) {
                auto g = __riscv_vrgather_vv_f32m4(__riscv_vle32_v_f32m4(gamma + c, 4), chan, count);
                auto b = __riscv_vrgather_vv_f32m4(__riscv_vle32_v_f32m4(beta + c, 4), chan, count);
                x = __riscv_vfmacc_vv_f32m4(b, x, g, count);
            }
            __riscv_vse32_v_f32m4(out + offset, x, count);
        }
    }
}

namespace MNN {
void MNNRvvInitPackedNorm(CoreFunctions* core) {
    static auto* const registered = core;
    MNN_ASSERT(registered == core);
    static const auto fallback = core->MNNNormPacked;
    core->MNNNormPacked = [](float* out, float* sum, const float* src, const float* residual, const float* gamma,
                             const float* beta, float epsilon, size_t rows, size_t cols, bool rms, int id, int threads) {
        if (rms && rows >= 16 && cols >= 4 && cols % 4 == 0) {
            MNNRmsNormC4_RVV(out, sum, src, residual, gamma, beta, epsilon, rows, cols, id, threads);
            return;
        }
        fallback(out, sum, src, residual, gamma, beta, epsilon, rows, cols, rms, id, threads);
    };
}

namespace {

template <bool low>
static inline vfloat32m1_t siluVecM1(vfloat32m1_t x, size_t vl) {
    auto z = __riscv_vfmin_vf_f32m1(__riscv_vfmax_vf_f32m1(__riscv_vfneg_v_f32m1(x, vl), -87.f, vl), 87.f, vl);
    auto n = __riscv_vfcvt_x_f_v_i32m1(__riscv_vfmul_vf_f32m1(z, 1.4426950408889634f, vl), vl);
    auto r = __riscv_vfnmsac_vf_f32m1(z, 0.6931471805599453f, __riscv_vfcvt_f_x_v_f32m1(n, vl), vl);
    auto y = __riscv_vfmv_v_f_f32m1(1.f / 720.f, vl);
    y = __riscv_vfmadd_vv_f32m1(y, r, __riscv_vfmv_v_f_f32m1(1.f / 120.f, vl), vl);
    y = __riscv_vfmadd_vv_f32m1(y, r, __riscv_vfmv_v_f_f32m1(1.f / 24.f, vl), vl);
    y = __riscv_vfmadd_vv_f32m1(y, r, __riscv_vfmv_v_f_f32m1(1.f / 6.f, vl), vl);
    y = __riscv_vfmadd_vv_f32m1(y, r, __riscv_vfmv_v_f_f32m1(.5f, vl), vl);
    const auto one = __riscv_vfmv_v_f_f32m1(1.f, vl);
    y = __riscv_vfmadd_vv_f32m1(y, r, one, vl);
    y = __riscv_vfmadd_vv_f32m1(y, r, one, vl);
    auto power = __riscv_vreinterpret_v_i32m1_f32m1(__riscv_vsll_vx_i32m1(__riscv_vadd_vx_i32m1(n, 127, vl), 23, vl));
    auto denom = __riscv_vfmacc_vv_f32m1(one, y, power, vl);
    auto inv = __riscv_vfrec7_v_f32m1(denom, vl);
    const auto two = __riscv_vfmv_v_f_f32m1(2.f, vl);
    inv = __riscv_vfmul_vv_f32m1(inv, __riscv_vfnmsac_vv_f32m1(two, denom, inv, vl), vl);
    if (!low) {
        inv = __riscv_vfmul_vv_f32m1(inv, __riscv_vfnmsac_vv_f32m1(two, denom, inv, vl), vl);
    }
    auto result = __riscv_vfmul_vv_f32m1(x, inv, vl);
    return __riscv_vfmerge_vfm_f32m1(result, -0.0f, __riscv_vmflt_vf_f32m1_b32(x, -87.f, vl), vl);
}

template <bool low>
static inline vfloat32m2_t siluVecM2(vfloat32m2_t x, size_t vl) {
    auto z = __riscv_vfmin_vf_f32m2(__riscv_vfmax_vf_f32m2(__riscv_vfneg_v_f32m2(x, vl), -87.f, vl), 87.f, vl);
    auto n = __riscv_vfcvt_x_f_v_i32m2(__riscv_vfmul_vf_f32m2(z, 1.4426950408889634f, vl), vl);
    auto r = __riscv_vfnmsac_vf_f32m2(z, 0.6931471805599453f, __riscv_vfcvt_f_x_v_f32m2(n, vl), vl);
    auto y = __riscv_vfmv_v_f_f32m2(1.f / 720.f, vl);
    y = __riscv_vfmadd_vv_f32m2(y, r, __riscv_vfmv_v_f_f32m2(1.f / 120.f, vl), vl);
    y = __riscv_vfmadd_vv_f32m2(y, r, __riscv_vfmv_v_f_f32m2(1.f / 24.f, vl), vl);
    y = __riscv_vfmadd_vv_f32m2(y, r, __riscv_vfmv_v_f_f32m2(1.f / 6.f, vl), vl);
    y = __riscv_vfmadd_vv_f32m2(y, r, __riscv_vfmv_v_f_f32m2(.5f, vl), vl);
    const auto one = __riscv_vfmv_v_f_f32m2(1.f, vl);
    y = __riscv_vfmadd_vv_f32m2(y, r, one, vl);
    y = __riscv_vfmadd_vv_f32m2(y, r, one, vl);
    auto power = __riscv_vreinterpret_v_i32m2_f32m2(__riscv_vsll_vx_i32m2(__riscv_vadd_vx_i32m2(n, 127, vl), 23, vl));
    auto denom = __riscv_vfmacc_vv_f32m2(one, y, power, vl);
    auto inv = __riscv_vfrec7_v_f32m2(denom, vl);
    const auto two = __riscv_vfmv_v_f_f32m2(2.f, vl);
    inv = __riscv_vfmul_vv_f32m2(inv, __riscv_vfnmsac_vv_f32m2(two, denom, inv, vl), vl);
    if (!low) {
        inv = __riscv_vfmul_vv_f32m2(inv, __riscv_vfnmsac_vv_f32m2(two, denom, inv, vl), vl);
    }
    auto result = __riscv_vfmul_vv_f32m2(x, inv, vl);
    return __riscv_vfmerge_vfm_f32m2(result, -0.0f, __riscv_vmflt_vf_f32m2_b16(x, -87.f, vl), vl);
}

static inline void replicateQuad32(float* dst32, const float* src4) {
    uint64_t lo, hi;
    ::memcpy(&lo, src4, 8);
    ::memcpy(&hi, src4 + 2, 8);
    uint64_t* out64 = reinterpret_cast<uint64_t*>(dst32);
    for (int i = 0; i < 8; ++i) {
        out64[2 * i] = lo;
        out64[2 * i + 1] = hi;
    }
}

#ifdef MNN_USE_SPACEMIT_IME2
// Stage a 32-token tile in 1-cycle TCM during Pass 1 so Pass 2 avoids re-reading DRAM
// and 8 KB power-of-2 L1d set conflicts, leaving normalized output warm in A100 L2.
struct Ime2RmsNormCtx {
    float* out;
    float* sum;
    const float* src;
    const float* residual;
    const float* gamma;
    const float* beta;
    float epsilon;
    size_t rows;
    size_t cols;
    int workers;
    std::atomic<bool> ok{true};
};

static void ime2RmsNormC4Task(size_t id, void* tcm, size_t bytes, void* user) {
    auto& p = *static_cast<Ime2RmsNormCtx*>(user);
    if (!tcm || __riscv_vlenb() != 128 || bytes < 16384) {
        p.ok.store(false, std::memory_order_relaxed);
        return;
    }
    const size_t fixedBytes = 4096;
    const size_t maxTile = ((bytes - fixedBytes) / (p.cols * sizeof(float))) & ~size_t(3);
    const size_t tile = std::min<size_t>(32, maxTile);
    if (tile < 4) {
        p.ok.store(false, std::memory_order_relaxed);
        return;
    }
    float* tcmBuf = static_cast<float*>(tcm);
    float* scratch = tcmBuf + tile * p.cols;
    float* invBuf = scratch;          // up to 128 floats
    float* gBuf = invBuf + 128;       // 32 floats
    float* bBuf = gBuf + 32;          // 32 floats
    const float invCols = 1.0f / static_cast<float>(p.cols);
    for (size_t row = id * tile; row < p.rows; row += size_t(p.workers) * tile) {
        const size_t tb = std::min(tile, p.rows - row);
        const size_t count = tb * 4;
        auto squared = __riscv_vfmv_v_f_f32m4(0.0f, count);
        for (size_t c = 0; c < p.cols; c += 4) {
            const size_t offset = c * p.rows + row * 4;
            auto x = __riscv_vle32_v_f32m4(p.src + offset, count);
            if (p.residual) {
                x = __riscv_vfadd_vv_f32m4(x, __riscv_vle32_v_f32m4(p.residual + offset, count), count);
                __riscv_vse32_v_f32m4(p.sum + offset, x, count);
            }
            __riscv_vse32_v_f32m4(tcmBuf + (c / 4) * count, x, count);
            squared = __riscv_vfmacc_vv_f32m4(squared, x, x, count);
        }
        __riscv_vse32_v_f32m4(invBuf, squared, count);
        for (size_t r = 0; r < tb; ++r) {
            const float ss = invBuf[r * 4] + invBuf[r * 4 + 1] + invBuf[r * 4 + 2] + invBuf[r * 4 + 3];
            const float inv = 1.0f / std::sqrt(ss * invCols + p.epsilon);
            invBuf[r * 4 + 0] = inv;
            invBuf[r * 4 + 1] = inv;
            invBuf[r * 4 + 2] = inv;
            invBuf[r * 4 + 3] = inv;
        }
        const bool hasGB = p.gamma && p.beta;
        for (size_t c = 0; c < p.cols; c += 4) {
            const float* tSrc = tcmBuf + (c / 4) * count;
            float* dst = p.out + c * p.rows + row * 4;
            if (hasGB) {
                replicateQuad32(gBuf, p.gamma + c);
                replicateQuad32(bBuf, p.beta + c);
                auto vg = __riscv_vle32_v_f32m1(gBuf, 32);
                auto vb = __riscv_vle32_v_f32m1(bBuf, 32);
                for (size_t off = 0; off < count; off += 32) {
                    const size_t vl = std::min<size_t>(32, count - off);
                    auto x = __riscv_vfmul_vv_f32m1(__riscv_vle32_v_f32m1(tSrc + off, vl),
                                                    __riscv_vle32_v_f32m1(invBuf + off, vl), vl);
                    x = __riscv_vfmacc_vv_f32m1(vb, x, vg, vl);
                    __riscv_vse32_v_f32m1(dst + off, x, vl);
                }
            } else {
                auto x = __riscv_vfmul_vv_f32m4(__riscv_vle32_v_f32m4(tSrc, count),
                                                __riscv_vle32_v_f32m4(invBuf, count), count);
                __riscv_vse32_v_f32m4(dst, x, count);
            }
        }
    }
}

static bool runIme2RmsNormC4(float* out, float* sum, const float* src, const float* residual, const float* gamma,
                             const float* beta, float epsilon, size_t rows, size_t cols, int threads) {
    const int workers = std::min<int>(threads, static_cast<int>(UP_DIV(rows, 32)));
    if (workers <= 0 || workers > 8) {
        return false;
    }
    Ime2RmsNormCtx ctx;
    ctx.out = out;
    ctx.sum = sum;
    ctx.src = src;
    ctx.residual = residual;
    ctx.gamma = gamma;
    ctx.beta = beta;
    ctx.epsilon = epsilon;
    ctx.rows = rows;
    ctx.cols = cols;
    ctx.workers = workers;
    return MNNSpacemitIme2RunTcmTasks(workers, ime2RmsNormC4Task, &ctx) == size_t(workers) &&
           ctx.ok.load(std::memory_order_relaxed);
}
#endif

class RvvLayerNorm final : public Execution {
public:
    RvvLayerNorm(std::shared_ptr<CPULayerNorm::Resource> res, Backend* backend)
        : Execution(backend), mResource(std::move(res)), mFallback(mResource, backend) {}

    bool onClone(Backend* bn, const Op*, Execution** dst) override {
        if (!dst) {
            return true;
        }
        *dst = new RvvLayerNorm(mResource, bn);
        return true;
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        mLayoutC4 = TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4;
        return mFallback.onResize(inputs, outputs);
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
#ifdef MNN_USE_SPACEMIT_IME2
        auto bn = static_cast<CPUBackend*>(backend());
        auto core = bn->functions();
        if (mLayoutC4 && core->bytes == 4 && core->pack == 4 && mResource->mRMSNorm && mResource->mGroup <= 1 &&
            CPUBackend::getDataType(inputs[0]) != DataType_DT_INT8 && inputs[0]->getType().bytes() != 1 &&
            bn->threadNumber() <= 8) {
            const int batch = inputs[0]->length(0);
            const int channel = inputs[0]->length(1);
            if (batch >= 32 && channel >= 256 && channel % 4 == 0) {
                const float* gamma = mResource->mIniGammaBeta ? mResource->mGamma->host<float>() : nullptr;
                const float* beta = mResource->mIniGammaBeta ? mResource->mBeta->host<float>() : nullptr;
                const float* input0 = inputs[0]->host<float>();
                float* output0 = outputs[0]->host<float>();
                const float* input1 = (inputs.size() == 2 && outputs.size() == 2) ? inputs[1]->host<float>() : nullptr;
                float* output1 = (inputs.size() == 2 && outputs.size() == 2) ? outputs[1]->host<float>() : nullptr;
                float* normOut = output1 ? output1 : output0;
                float* sumOut = output1 ? output0 : nullptr;
                if (runIme2RmsNormC4(normOut, sumOut, input0, input1, gamma, beta, mResource->mEpsilon, batch, channel,
                                     bn->threadNumber())) {
                    return NO_ERROR;
                }
            }
        }
#endif
        return mFallback.onExecute(inputs, outputs);
    }

private:
    std::shared_ptr<CPULayerNorm::Resource> mResource;
    CPULayerNorm mFallback;
    bool mLayoutC4 = false;
};

#if defined(MNN_SUPPORT_TRANSFORMER_FUSE) && defined(MNN_GATED_RMS_NORM)
#ifdef MNN_USE_SPACEMIT_IME2
struct Ime2GatedRmsCtx {
    const float* x;
    const float* z;
    float* out;
    const float* gamma;
    const float* beta;
    float epsilon;
    int batch;
    int heads;
    int inside;
    int workers;
    bool low;
    std::atomic<bool> ok{true};
};

template <bool low>
static void ime2GatedRmsNormTask(size_t id, void* tcm, size_t bytes, void* user) {
    auto& p = *static_cast<Ime2GatedRmsCtx*>(user);
    const int inside4 = p.inside / 4;
    const size_t gbBytes = (p.gamma && p.beta) ? size_t(inside4) * 64 * sizeof(float) : 0;
    const size_t perTokBytes = size_t(p.heads) * (size_t(p.inside) + 5) * sizeof(float);
    const size_t fixedBytes = gbBytes + 4096;
    if (!tcm || __riscv_vlenb() != 128 || bytes <= fixedBytes + 4 * perTokBytes) {
        p.ok.store(false, std::memory_order_relaxed);
        return;
    }
    const int maxTile = static_cast<int>((bytes - fixedBytes) / perTokBytes) & ~3;
    const int tile = std::min(32, maxTile);
    if (tile < 4) {
        p.ok.store(false, std::memory_order_relaxed);
        return;
    }
    float* ptr = static_cast<float*>(tcm);
    float* gTable = ptr;
    float* bTable = nullptr;
    if (p.gamma && p.beta) {
        bTable = gTable + size_t(inside4) * 32;
        ptr = bTable + size_t(inside4) * 32;
        for (int c4 = 0; c4 < inside4; ++c4) {
            replicateQuad32(gTable + size_t(c4) * 32, p.gamma + c4 * 4);
            replicateQuad32(bTable + size_t(c4) * 32, p.beta + c4 * 4);
        }
    }
    float* xTcm = ptr;
    float* sqTcm = xTcm + size_t(tile) * p.heads * p.inside;
    float* invTcm = sqTcm + size_t(tile) * p.heads * 4;
    float* invH = invTcm + size_t(tile) * p.heads;
    float* xRow = invH + 128;
    const size_t outside = size_t(p.batch) * p.heads;
    const float invInside = 1.0f / static_cast<float>(p.inside);
    const bool hasGB = p.gamma && p.beta;

    for (int b0 = static_cast<int>(id) * tile; b0 < p.batch; b0 += p.workers * tile) {
        const int tb = std::min(tile, p.batch - b0);
        const size_t rowCount = size_t(tb) * p.heads;
        const size_t planeFloats = rowCount * 4;
        // Pass 1: unit-stride load x into TCM and accumulate squared sum per (b, h).
        ::memset(sqTcm, 0, planeFloats * sizeof(float));
        for (int c4 = 0; c4 < inside4; ++c4) {
            const float* xSrc = p.x + size_t(c4) * outside * 4 + size_t(b0) * p.heads * 4;
            float* xDst = xTcm + size_t(c4) * planeFloats;
            for (size_t off = 0; off < planeFloats;) {
                const size_t vl = __riscv_vsetvl_e32m4(planeFloats - off);
                auto vx = __riscv_vle32_v_f32m4(xSrc + off, vl);
                __riscv_vse32_v_f32m4(xDst + off, vx, vl);
                auto sq = __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(sqTcm + off, vl), vx, vx, vl);
                __riscv_vse32_v_f32m4(sqTcm + off, sq, vl);
                off += vl;
            }
        }
        for (size_t r = 0; r < rowCount; ++r) {
            const float ss = sqTcm[r * 4] + sqTcm[r * 4 + 1] + sqTcm[r * 4 + 2] + sqTcm[r * 4 + 3];
            invTcm[r] = 1.0f / std::sqrt(ss * invInside + p.epsilon);
        }
        // Pass 2: unit-stride read z and write out, gathering x from 1-cycle TCM.
        const size_t outFloats = size_t(tb) * 4;
        for (int h = 0; h < p.heads; ++h) {
            for (int b = 0; b < tb; ++b) {
                const float inv = invTcm[size_t(b) * p.heads + h];
                invH[b * 4 + 0] = inv;
                invH[b * 4 + 1] = inv;
                invH[b * 4 + 2] = inv;
                invH[b * 4 + 3] = inv;
            }
            for (int c4 = 0; c4 < inside4; ++c4) {
                const float* tBase = xTcm + size_t(c4) * planeFloats + size_t(h) * 4;
                uint64_t* rd = reinterpret_cast<uint64_t*>(xRow);
                for (int b = 0; b < tb; ++b) {
                    const uint64_t* ts = reinterpret_cast<const uint64_t*>(tBase + size_t(b) * p.heads * 4);
                    rd[2 * b + 0] = ts[0];
                    rd[2 * b + 1] = ts[1];
                }
                const size_t zOff = (size_t(h) * inside4 + c4) * p.batch * 4 + size_t(b0) * 4;
                const float* zPtr = p.z + zOff;
                float* oPtr = p.out + zOff;
                const float* gPtr = hasGB ? gTable + size_t(c4) * 32 : nullptr;
                const float* bPtr = hasGB ? bTable + size_t(c4) * 32 : nullptr;
                for (size_t off = 0; off < outFloats; off += 32) {
                    const size_t vl = std::min<size_t>(32, outFloats - off);
                    auto norm = __riscv_vfmul_vv_f32m1(__riscv_vle32_v_f32m1(xRow + off, vl),
                                                       __riscv_vle32_v_f32m1(invH + off, vl), vl);
                    if (hasGB) {
                        norm = __riscv_vfmacc_vv_f32m1(__riscv_vle32_v_f32m1(bPtr, vl), norm,
                                                       __riscv_vle32_v_f32m1(gPtr, vl), vl);
                    }
                    auto s = siluVecM1<low>(__riscv_vle32_v_f32m1(zPtr + off, vl), vl);
                    __riscv_vse32_v_f32m1(oPtr + off, __riscv_vfmul_vv_f32m1(norm, s, vl), vl);
                }
            }
        }
    }
}

// Consumes head-major [heads, batch, inside] directly from RvvLinearAttention,
// eliminating the intermediate 4 MB NC4HW4 transpose and per-head gather.
template <bool low>
static void ime2GatedRmsNormHeadMajorTask(size_t id, void* tcm, size_t bytes, void* user) {
    auto& p = *static_cast<Ime2GatedRmsCtx*>(user);
    const int inside4 = p.inside / 4;
    const size_t fixedBytes = (size_t(p.inside) * 2 + 256) * sizeof(float);
    const size_t perTokBytes = size_t(p.inside + 4) * sizeof(float);
    if (!tcm || __riscv_vlenb() != 128 || bytes <= fixedBytes + 4 * perTokBytes) {
        p.ok.store(false, std::memory_order_relaxed);
        return;
    }
    const int maxTile = static_cast<int>((bytes - fixedBytes) / perTokBytes) & ~3;
    const int tile = std::min(64, maxTile);
    if (tile < 4) {
        p.ok.store(false, std::memory_order_relaxed);
        return;
    }
    float* gTcm = static_cast<float*>(tcm);
    float* bTcm = gTcm + p.inside;
    float* xTcm = bTcm + p.inside;
    float* c4Tmp = xTcm + size_t(tile) * p.inside;
    const bool hasGB = p.gamma && p.beta;
    if (hasGB) {
        ::memcpy(gTcm, p.gamma, size_t(p.inside) * sizeof(float));
        ::memcpy(bTcm, p.beta, size_t(p.inside) * sizeof(float));
    }
    const float invInside = 1.0f / static_cast<float>(p.inside);
    const size_t numBTiles = UP_DIV(p.batch, tile);
    const size_t totalTasks = size_t(p.heads) * numBTiles;
    auto vZero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    const bool singleVec = (p.inside == 128);
    vfloat32m4_t vgFull = singleVec && hasGB ? __riscv_vle32_v_f32m4(gTcm, 128) : __riscv_vfmv_v_f_f32m4(1.0f, 128);
    vfloat32m4_t vbFull = singleVec && hasGB ? __riscv_vle32_v_f32m4(bTcm, 128) : __riscv_vfmv_v_f_f32m4(0.0f, 128);

    for (size_t taskIdx = id; taskIdx < totalTasks; taskIdx += size_t(p.workers)) {
        const int h = static_cast<int>(taskIdx / numBTiles);
        const int b0 = static_cast<int>(taskIdx % numBTiles) * tile;
        const int tb = std::min(tile, p.batch - b0);
        const float* xHead = p.x + size_t(h) * p.batch * p.inside;
        for (int b = 0; b < tb; ++b) {
            const float* xRow = xHead + size_t(b0 + b) * p.inside;
            float* tRow = xTcm + size_t(b) * p.inside;
            if (singleVec) {
                auto vx = __riscv_vle32_v_f32m4(xRow, 128);
                const float ss = __riscv_vfmv_f_s_f32m1_f32(
                    __riscv_vfredusum_vs_f32m4_f32m1(__riscv_vfmul_vv_f32m4(vx, vx, 128), vZero, 128));
                vx = __riscv_vfmul_vf_f32m4(vx, 1.0f / std::sqrt(ss * invInside + p.epsilon), 128);
                if (hasGB) {
                    vx = __riscv_vfmacc_vv_f32m4(vbFull, vx, vgFull, 128);
                }
                __riscv_vse32_v_f32m4(tRow, vx, 128);
            } else {
                float ss = 0.0f;
                for (int d = 0; d < p.inside;) {
                    const size_t vl = __riscv_vsetvl_e32m4(p.inside - d);
                    auto vx = __riscv_vle32_v_f32m4(xRow + d, vl);
                    ss += __riscv_vfmv_f_s_f32m1_f32(
                        __riscv_vfredusum_vs_f32m4_f32m1(__riscv_vfmul_vv_f32m4(vx, vx, vl), vZero, vl));
                    d += vl;
                }
                const float inv = 1.0f / std::sqrt(ss * invInside + p.epsilon);
                for (int d = 0; d < p.inside;) {
                    const size_t vl = __riscv_vsetvl_e32m4(p.inside - d);
                    auto vx = __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(xRow + d, vl), inv, vl);
                    if (hasGB) {
                        vx = __riscv_vfmacc_vv_f32m4(__riscv_vle32_v_f32m4(bTcm + d, vl), vx,
                                                     __riscv_vle32_v_f32m4(gTcm + d, vl), vl);
                    }
                    __riscv_vse32_v_f32m4(tRow + d, vx, vl);
                    d += vl;
                }
            }
        }
        const size_t outFloats = size_t(tb) * 4;
        for (int c4 = 0; c4 < inside4; ++c4) {
            uint64_t* rd = reinterpret_cast<uint64_t*>(c4Tmp);
            const float* tBase = xTcm + size_t(c4) * 4;
            for (int b = 0; b < tb; ++b) {
                const uint64_t* ts = reinterpret_cast<const uint64_t*>(tBase + size_t(b) * p.inside);
                rd[2 * b + 0] = ts[0];
                rd[2 * b + 1] = ts[1];
            }
            const size_t zOff = (size_t(h) * inside4 + c4) * p.batch * 4 + size_t(b0) * 4;
            const float* zPtr = p.z + zOff;
            float* oPtr = p.out + zOff;
            for (size_t off = 0; off < outFloats;) {
                const size_t vl = __riscv_vsetvl_e32m2(outFloats - off);
                auto vx = __riscv_vle32_v_f32m2(c4Tmp + off, vl);
                auto vz = __riscv_vle32_v_f32m2(zPtr + off, vl);
                auto vs = siluVecM2<low>(vz, vl);
                __riscv_vse32_v_f32m2(oPtr + off, __riscv_vfmul_vv_f32m2(vx, vs, vl), vl);
                off += vl;
            }
        }
    }
}
#endif

// Fused OpType_GatedRMSNorm: out = (RMSNorm(x) * gamma + beta) * silu(z).
// Absorbs the [batch*heads, inside] -> [batch, heads*inside] NC4HW4 repack.
class RvvGatedRMSNorm final : public Execution {
public:
    RvvGatedRMSNorm(std::shared_ptr<CPULayerNorm::Resource> res, Backend* backend)
        : Execution(backend), mResource(std::move(res)) {}

    bool onClone(Backend* bn, const Op*, Execution** dst) override {
        if (!dst) {
            return true;
        }
        if (!bn || bn->type() != MNN_FORWARD_CPU) {
            return false;
        }
        auto core = static_cast<CPUBackend*>(bn)->functions();
        if (!core->supportRVV || core->bytes != 4 || core->pack != 4) {
            return false;
        }
        *dst = new RvvGatedRMSNorm(mResource, bn);
        return true;
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (inputs.size() != 2 || outputs.size() != 1) {
            return INPUT_DATA_ERROR;
        }
        const auto x = inputs[0], z = inputs[1];
        mBatch = z->length(0);
        mInside = x->length(1);
        if (mBatch <= 0 || mInside <= 0 || mInside % 4 != 0 || z->length(1) % mInside != 0) {
            return INPUT_DATA_ERROR;
        }
        mHeads = z->length(1) / mInside;
        if (x->length(0) != mBatch * mHeads) {
            return INPUT_DATA_ERROR;
        }
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        auto cpu = static_cast<CPUBackend*>(backend());
        const float* x = inputs[0]->host<float>();
        const float* z = inputs[1]->host<float>();
        float* out = outputs[0]->host<float>();
        const float* gamma = mResource->mIniGammaBeta ? mResource->mGamma->host<float>() : nullptr;
        const float* beta = mResource->mIniGammaBeta ? mResource->mBeta->host<float>() : nullptr;
        const float eps = mResource->mEpsilon;
        const int batch = mBatch, heads = mHeads, inside = mInside, inside4 = inside / 4;
        const int threads = cpu->threadNumber();
        const bool low = cpu->precisionMode() == BackendConfig::Precision_Low;
        bool headMajor = false;
        if (auto* handoff = MNNRvvGetHandoffContext()) {
            if (handoff->backend == backend()) {
                auto it = handoff->activeLinearAttn.find(inputs[0]);
                if (it != handoff->activeLinearAttn.end()) {
                    const auto info = it->second;
                    handoff->activeLinearAttn.erase(it);
                    if (info.ptr == x && info.batch == 1 && info.length == batch && info.heads == heads &&
                        info.dv == inside) {
                        headMajor = true;
                    }
                }
            }
        }
#ifdef MNN_USE_SPACEMIT_IME2
        if (headMajor && threads <= 8 && inside <= 512) {
            const int workers =
                batch == 1 ? std::min<int>(4, heads) : std::min<int>(threads, heads * UP_DIV(batch, 64));
            Ime2GatedRmsCtx ctx;
            ctx.x = x;
            ctx.z = z;
            ctx.out = out;
            ctx.gamma = gamma;
            ctx.beta = beta;
            ctx.epsilon = eps;
            ctx.batch = batch;
            ctx.heads = heads;
            ctx.inside = inside;
            ctx.workers = workers;
            ctx.low = low;
            auto task = low ? ime2GatedRmsNormHeadMajorTask<true> : ime2GatedRmsNormHeadMajorTask<false>;
            const size_t ran = batch == 1 ? MNNSpacemitIme2RunDecodeTasks(workers, task, &ctx)
                                          : MNNSpacemitIme2RunTcmTasks(workers, task, &ctx);
            if (workers > 0 && ran == size_t(workers) && ctx.ok.load(std::memory_order_relaxed)) {
                return NO_ERROR;
            }
        }
        if (!headMajor && batch >= 32 && threads <= 8 && inside <= 512) {
            const int workers = std::min(threads, UP_DIV(batch, 32));
            Ime2GatedRmsCtx ctx;
            ctx.x = x;
            ctx.z = z;
            ctx.out = out;
            ctx.gamma = gamma;
            ctx.beta = beta;
            ctx.epsilon = eps;
            ctx.batch = batch;
            ctx.heads = heads;
            ctx.inside = inside;
            ctx.workers = workers;
            ctx.low = low;
            auto task = low ? ime2GatedRmsNormTask<true> : ime2GatedRmsNormTask<false>;
            if (MNNSpacemitIme2RunTcmTasks(workers, task, &ctx) == size_t(workers) &&
                ctx.ok.load(std::memory_order_relaxed)) {
                return NO_ERROR;
            }
        }
#endif
        const size_t outside = size_t(batch) * heads;
        const int workers = std::min<int>(threads, static_cast<int>(outside));
        const float invInside = 1.0f / static_cast<float>(inside);
        if (headMajor) {
            MNN_CONCURRENCY_BEGIN(w, workers) {
                const size_t begin = outside * w / workers;
                const size_t end = outside * (w + 1) / workers;
                for (size_t r = begin; r < end; ++r) {
                    const size_t h = r / batch;
                    const size_t b = r % batch;
                    const float* xRow = x + r * inside;
                    float ss = 0.0f;
                    for (int d = 0; d < inside; ++d) {
                        ss += xRow[d] * xRow[d];
                    }
                    const float inv = 1.0f / std::sqrt(ss * invInside + eps);
                    for (int c4 = 0; c4 < inside4; ++c4) {
                        const float* xp = xRow + c4 * 4;
                        const size_t zOff = (h * inside4 + c4) * batch * 4 + b * 4;
                        auto vx = __riscv_vfmul_vf_f32m1(__riscv_vle32_v_f32m1(xp, 4), inv, 4);
                        if (gamma && beta) {
                            vx = __riscv_vfmacc_vv_f32m1(__riscv_vle32_v_f32m1(beta + c4 * 4, 4), vx,
                                                         __riscv_vle32_v_f32m1(gamma + c4 * 4, 4), 4);
                        }
                        auto vz = __riscv_vle32_v_f32m1(z + zOff, 4);
                        auto vs = low ? siluVecM1<true>(vz, 4) : siluVecM1<false>(vz, 4);
                        __riscv_vse32_v_f32m1(out + zOff, __riscv_vfmul_vv_f32m1(vx, vs, 4), 4);
                    }
                }
            }
            MNN_CONCURRENCY_END();
            return NO_ERROR;
        }
        MNN_CONCURRENCY_BEGIN(w, workers) {
            const size_t begin = outside * w / workers;
            const size_t end = outside * (w + 1) / workers;
            for (size_t r = begin; r < end; ++r) {
                float ss = 0.0f;
                for (int c4 = 0; c4 < inside4; ++c4) {
                    const float* xp = x + size_t(c4) * outside * 4 + r * 4;
                    ss += xp[0] * xp[0] + xp[1] * xp[1] + xp[2] * xp[2] + xp[3] * xp[3];
                }
                const float inv = 1.0f / std::sqrt(ss * invInside + eps);
                const size_t b = r / heads;
                const size_t h = r % heads;
                for (int c4 = 0; c4 < inside4; ++c4) {
                    const float* xp = x + size_t(c4) * outside * 4 + r * 4;
                    const size_t zOff = (h * inside4 + c4) * batch * 4 + b * 4;
                    const float* zp = z + zOff;
                    float* op = out + zOff;
                    auto vx = __riscv_vfmul_vf_f32m1(__riscv_vle32_v_f32m1(xp, 4), inv, 4);
                    if (gamma && beta) {
                        vx = __riscv_vfmacc_vv_f32m1(__riscv_vle32_v_f32m1(beta + c4 * 4, 4), vx,
                                                     __riscv_vle32_v_f32m1(gamma + c4 * 4, 4), 4);
                    }
                    auto vz = __riscv_vle32_v_f32m1(zp, 4);
                    auto vs = low ? siluVecM1<true>(vz, 4) : siluVecM1<false>(vz, 4);
                    __riscv_vse32_v_f32m1(op, __riscv_vfmul_vv_f32m1(vx, vs, 4), 4);
                }
            }
        }
        MNN_CONCURRENCY_END();
        return NO_ERROR;
    }

private:
    std::shared_ptr<CPULayerNorm::Resource> mResource;
    int mBatch = 0, mHeads = 0, mInside = 0;
};
#endif

} // namespace

Execution* MNNRvvCreateLayerNorm(CPUBackend* backend, const Op* op) {
    auto res = CPULayerNorm::makeResource(op, backend);
    if (!res) {
        return nullptr;
    }
    return new RvvLayerNorm(std::move(res), backend);
}

Execution* MNNRvvCreateGatedRMSNorm(CPUBackend* backend, const Op* op, const std::vector<Tensor*>& inputs,
                                    const std::vector<Tensor*>& outputs) {
#if defined(MNN_SUPPORT_TRANSFORMER_FUSE) && defined(MNN_GATED_RMS_NORM)
    auto core = backend->functions();
    if (!core->supportRVV || core->bytes != 4 || core->pack != 4) {
        return nullptr;
    }
    const bool externalOk = backend->getRuntime()->hint().useCachedMmap > 1;
    int inside = 0;
    if (!OpCommonUtils::gatedRMSNormOpenCLOk(op, inputs, outputs, nullptr, &inside, nullptr, externalOk) ||
        inside % 4 != 0) {
        return nullptr;
    }
    auto res = CPULayerNorm::makeResource(op, backend);
    if (!res) {
        return nullptr;
    }
    return new RvvGatedRMSNorm(std::move(res), backend);
#else
    (void)backend;
    (void)op;
    (void)inputs;
    (void)outputs;
    return nullptr;
#endif
}
} // namespace MNN
