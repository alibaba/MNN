//
//  CPULinearAttention.cpp
//  MNN
//
//  Created by MNN on 2026/02/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include <limits>
#include <cmath>
#include <vector>
#include <algorithm>
#include "CPULinearAttention.hpp"
#include "CPUBackend.hpp"
#include "core/MNNFileUtils.h"
#include "compute/CommonOptFunction.h"
#include "core/Macro.h"
#include "core/Concurrency.h"
#include "core/BufferAllocator.hpp"
#include "core/TensorUtils.hpp"
#include "core/OpCommonUtils.hpp"
#include "core/BufferAllocator.hpp"
#include "compute/ConvolutionTiledExecutor.hpp"

namespace MNN {

// ─── Byte-aware element access helpers ───
static inline float _readElement(const int8_t* ptr, int index, int bytes) {
#ifdef __aarch64__
    if (bytes == 2) return (float)((const __fp16*)ptr)[index];
#endif
    return ((const float*)ptr)[index];
}

static inline void _writeElement(int8_t* ptr, int index, float val, int bytes) {
#ifdef __aarch64__
    if (bytes == 2) { ((__fp16*)ptr)[index] = (__fp16)val; return; }
#endif
    ((float*)ptr)[index] = val;
}


ErrorCode CPULinearAttention::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto qkv = inputs[0];
    auto convWeight = inputs[3];

    int batch      = qkv->length(0);
    int convDim    = qkv->length(1);    // D (total projection dim)
    int seqLen     = qkv->length(2);    // L
    int kernelSize = convWeight->length(2);
    int convStateSize = kernelSize - 1;

    // ─── Per-type parameters ───
    int convChannels = convDim;
    bool needRecurrentState = false;

    if (mAttentionType == "short_conv") {
        convChannels = mHeadVDim;
    } else if (mAttentionType == "gated_delta_rule") {
        needRecurrentState = true;
    }

    // ─── Persistent state buffers (STATIC): allocate once, shared via onClone ───
    if (mStateCache->mConvState.get() == nullptr) {
        mStateCache->mConvState.reset(Tensor::createDevice<int8_t>({batch * convChannels * convStateSize * mBytes}));
        bool success = backend()->onAcquireBuffer(mStateCache->mConvState.get(), Backend::STATIC);
        if (!success) return OUT_OF_MEMORY;
        ::memset(mStateCache->mConvState->host<int8_t>(), 0, batch * convChannels * convStateSize * mBytes);

        if (needRecurrentState) {
            int H = mNumVHeads, dk = mHeadKDim, dv = mHeadVDim;
            mStateCache->mRecurrentState.reset(Tensor::createDevice<int8_t>({batch * H * dk * dv * mBytes}));
            success = backend()->onAcquireBuffer(mStateCache->mRecurrentState.get(), Backend::STATIC);
            if (!success) return OUT_OF_MEMORY;
            ::memset(mStateCache->mRecurrentState->host<int8_t>(), 0, batch * H * dk * dv * mBytes);
        }
    } else if (seqLen > 1) {
        // Prefill: reset state for new sequence, UNLESS:
        // 1. Loading from prefix cache (PendingRead), or
        // 2. Reusing KV from previous inference (reuse_kv=true, i.e. previous != remove)
        bool loadingFromDisk = (mMeta != nullptr && mMeta->file_flag == KVMeta::PendingRead && mMeta->file_name.size() > 0);
        bool reusingKV = (mMeta != nullptr && mMeta->previous != mMeta->remove);
        if (!loadingFromDisk && !reusingKV) {
            int convStateBytes = batch * convChannels * convStateSize * mBytes;
            ::memset(mStateCache->mConvState->host<int8_t>(), 0, convStateBytes);
            if (mStateCache->mRecurrentState.get() != nullptr) {
                int H = mNumVHeads, dk = mHeadKDim, dv = mHeadVDim;
                ::memset(mStateCache->mRecurrentState->host<int8_t>(), 0, batch * H * dk * dv * mBytes);
            }
        }
    }

    // ─── Temporary buffers (DYNAMIC) ───
    int totalLen = convStateSize + seqLen;
    mConvPadded.reset(Tensor::createDevice<int8_t>({batch * convChannels * totalLen * mBytes}));
    bool success = backend()->onAcquireBuffer(mConvPadded.get(), Backend::DYNAMIC);
    if (!success) return OUT_OF_MEMORY;

    mConvOut.reset(Tensor::createDevice<int8_t>({batch * convChannels * seqLen * mBytes}));
    success = backend()->onAcquireBuffer(mConvOut.get(), Backend::DYNAMIC);
    if (!success) return OUT_OF_MEMORY;

    if (needRecurrentState) {
        int dk = mHeadKDim, dv = mHeadVDim;
        int threadNum = static_cast<CPUBackend*>(backend())->threadNumber();
        int perThread = 2 * dk + 3 * dv; // q_local + k_local + v_local + vpred + delta
        mThreadLocalBuf.reset(Tensor::createDevice<int8_t>({threadNum * perThread * mBytes}));
        success = backend()->onAcquireBuffer(mThreadLocalBuf.get(), Backend::DYNAMIC);
        if (!success) return OUT_OF_MEMORY;

        // Pre-computed decay buffer: exp(gate) for all [B, L, H]
        // Always fp32 — MNNExp requires fp32, decay is a scalar per timestep
        // Use int8_t with explicit byte count to avoid Arm82 backend halving the allocation
        mDecayBuf.reset(Tensor::createDevice<int8_t>({batch * seqLen * mNumVHeads * (int)sizeof(float)}));
        success = backend()->onAcquireBuffer(mDecayBuf.get(), Backend::DYNAMIC);
        if (!success) return OUT_OF_MEMORY;

        backend()->onReleaseBuffer(mDecayBuf.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mThreadLocalBuf.get(), Backend::DYNAMIC);
    }

    // fp16 path: per-thread fp32 temp buffer for Conv1D + SiLu (MNNSiLu requires fp32)
    if (mBytes == 2) {
        int threadNum = static_cast<CPUBackend*>(backend())->threadNumber();
        // Need totalLen floats for padded input + L floats for SiLu output = (totalLen + seqLen) per thread
        mConvFp32Buf.reset(Tensor::createDevice<int8_t>({threadNum * (totalLen + seqLen) * (int)sizeof(float)}));
        success = backend()->onAcquireBuffer(mConvFp32Buf.get(), Backend::DYNAMIC);
        if (!success) return OUT_OF_MEMORY;
        backend()->onReleaseBuffer(mConvFp32Buf.get(), Backend::DYNAMIC);
    }

    backend()->onReleaseBuffer(mConvPadded.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mConvOut.get(), Backend::DYNAMIC);

    return NO_ERROR;
}

void CPULinearAttention::gated_delta_rule_ref(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // Reference implementation (fp32 only, for correctness verification)
    auto qkvTensor    = inputs[0];
    auto gateTensor   = inputs[1];
    auto betaTensor   = inputs[2];
    auto convWTensor  = inputs[3];
    auto outTensor    = outputs[0];

    const float* qkvPtr   = qkvTensor->host<float>();
    const float* gatePtr  = gateTensor->host<float>();
    const float* betaPtr  = betaTensor->host<float>();
    const float* convWPtr = convWTensor->host<float>();
    float* outPtr         = outTensor->host<float>();

    const int B       = qkvTensor->length(0);
    const int D       = qkvTensor->length(1);
    const int L       = qkvTensor->length(2);
    const int H_k     = mNumKHeads;
    const int H_v     = mNumVHeads;
    const int d_k     = mHeadKDim;
    const int d_v     = mHeadVDim;
    const int key_dim = H_k * d_k;
    const int val_dim = H_v * d_v;
    const int K_conv  = convWTensor->length(2);
    const int convStateSize = K_conv - 1;
    const bool useL2Norm = mUseQKL2Norm;
    const int gqa_factor = (H_v > H_k) ? (H_v / H_k) : 1;
    const int H = H_v;

    // Step 1: Depthwise Conv1D + SiLU
    const int totalLen = convStateSize + L;
    std::vector<float> convInput(B * D * totalLen, 0.0f);
    float* convStatePtr = mStateCache->mConvState->host<float>();

    for (int b = 0; b < B; ++b) {
        for (int d = 0; d < D; ++d) {
            float* dst = convInput.data() + b * D * totalLen + d * totalLen;
            const float* stateChannel = convStatePtr + b * D * convStateSize + d * convStateSize;
            ::memcpy(dst, stateChannel, convStateSize * sizeof(float));
            const float* inputChannel = qkvPtr + b * D * L + d * L;
            ::memcpy(dst + convStateSize, inputChannel, L * sizeof(float));
        }
    }

    std::vector<float> convOut(B * D * L, 0.0f);
    for (int b = 0; b < B; ++b) {
        for (int d = 0; d < D; ++d) {
            const float* src    = convInput.data() + b * D * totalLen + d * totalLen;
            const float* weight = convWPtr + d * K_conv;
            float* out          = convOut.data() + b * D * L + d * L;
            for (int l = 0; l < L; ++l) {
                float sum = 0.0f;
                for (int k = 0; k < K_conv; ++k) {
                    sum += src[l + k] * weight[k];
                }
                float sigmoid_val = 1.0f / (1.0f + expf(-sum));
                out[l] = sum * sigmoid_val;
            }
        }
    }

    for (int b = 0; b < B; ++b) {
        for (int d = 0; d < D; ++d) {
            const float* src = convInput.data() + b * D * totalLen + d * totalLen + (totalLen - convStateSize);
            float* dst       = convStatePtr + b * D * convStateSize + d * convStateSize;
            ::memcpy(dst, src, convStateSize * sizeof(float));
        }
    }

    // Step 2: Split Q, K, V
    std::vector<float> Q(B * L * H * d_k, 0.0f);
    std::vector<float> K(B * L * H * d_k, 0.0f);
    std::vector<float> V(B * L * H * d_v, 0.0f);

    for (int b = 0; b < B; ++b) {
        for (int l = 0; l < L; ++l) {
            for (int h = 0; h < H_k; ++h) {
                for (int dk = 0; dk < d_k; ++dk) {
                    int srcChannel = h * d_k + dk;
                    float val = convOut[b * D * L + srcChannel * L + l];
                    for (int r = 0; r < gqa_factor; ++r) {
                        int dstHead = h * gqa_factor + r;
                        Q[(b * L + l) * H * d_k + dstHead * d_k + dk] = val;
                    }
                }
            }
            for (int h = 0; h < H_k; ++h) {
                for (int dk = 0; dk < d_k; ++dk) {
                    int srcChannel = key_dim + h * d_k + dk;
                    float val = convOut[b * D * L + srcChannel * L + l];
                    for (int r = 0; r < gqa_factor; ++r) {
                        int dstHead = h * gqa_factor + r;
                        K[(b * L + l) * H * d_k + dstHead * d_k + dk] = val;
                    }
                }
            }
            for (int h = 0; h < H_v; ++h) {
                for (int dv = 0; dv < d_v; ++dv) {
                    int srcChannel = 2 * key_dim + h * d_v + dv;
                    float val = convOut[b * D * L + srcChannel * L + l];
                    V[(b * L + l) * H * d_v + h * d_v + dv] = val;
                }
            }
        }
    }

    // Step 3: Optional L2 Normalization
    if (useL2Norm) {
        const float eps = 1e-6f;
        for (int i = 0; i < B * L * H; ++i) {
            float* qHead = Q.data() + i * d_k;
            float sumSq = 0.0f;
            for (int dk = 0; dk < d_k; ++dk) sumSq += qHead[dk] * qHead[dk];
            float invNorm = 1.0f / sqrtf(sumSq + eps);
            for (int dk = 0; dk < d_k; ++dk) qHead[dk] *= invNorm;

            float* kHead = K.data() + i * d_k;
            sumSq = 0.0f;
            for (int dk = 0; dk < d_k; ++dk) sumSq += kHead[dk] * kHead[dk];
            invNorm = 1.0f / sqrtf(sumSq + eps);
            for (int dk = 0; dk < d_k; ++dk) kHead[dk] *= invNorm;
        }
    }

    // Step 4: Scale Q
    const float qScale = 1.0f / sqrtf((float)d_k);
    for (int i = 0; i < B * L * H * d_k; ++i) Q[i] *= qScale;

    // Step 5: Gated Delta Rule
    float* rnnStatePtr = mStateCache->mRecurrentState->host<float>();
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < L; ++t) {
            for (int h = 0; h < H; ++h) {
                float* state = rnnStatePtr + (b * H + h) * d_k * d_v;
                const float* q_t = Q.data() + (b * L + t) * H * d_k + h * d_k;
                const float* k_t = K.data() + (b * L + t) * H * d_k + h * d_k;
                const float* v_t = V.data() + (b * L + t) * H * d_v + h * d_v;
                float g_t    = gatePtr[b * L * H + t * H + h];
                float beta_t = betaPtr[b * L * H + t * H + h];

                float decay = expf(g_t);
                for (int i = 0; i < d_k * d_v; ++i) state[i] *= decay;

                std::vector<float> v_pred(d_v, 0.0f);
                for (int dk = 0; dk < d_k; ++dk)
                    for (int dv = 0; dv < d_v; ++dv)
                        v_pred[dv] += state[dk * d_v + dv] * k_t[dk];

                std::vector<float> delta(d_v);
                for (int dv = 0; dv < d_v; ++dv)
                    delta[dv] = beta_t * (v_t[dv] - v_pred[dv]);

                for (int dk = 0; dk < d_k; ++dk)
                    for (int dv = 0; dv < d_v; ++dv)
                        state[dk * d_v + dv] += k_t[dk] * delta[dv];

                float* o_t = outPtr + (b * L + t) * H * d_v + h * d_v;
                for (int dv = 0; dv < d_v; ++dv) {
                    float sum = 0.0f;
                    for (int dk = 0; dk < d_k; ++dk)
                        sum += state[dk * d_v + dv] * q_t[dk];
                    o_t[dv] = sum;
                }
            }
        }
    }
}

ErrorCode CPULinearAttention::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // Load prefix cache from disk (PendingRead)
    if (mMeta != nullptr && mMeta->file_name.size() > 0 && mMeta->file_flag == KVMeta::PendingRead) {
        int layer_index = mMeta->layer_index;
        std::string basePath = MNNFilePathConcat(mPrefixCacheDir, mMeta->file_name) + "_" + std::to_string(layer_index);
        std::string pathk = basePath + ".k";
        std::string pathv = basePath + ".v";
        // Load conv state (.k file)
        auto kfd = MNNOpenFile(pathk.c_str(), MNN_FILE_READ);
        if (kfd != INVALID_FILE) {
            size_t kSize = MNNGetFileSize(kfd);
            if (kSize > 0 && kSize != INVALID_SIZE) {
                void* kMap = MNNMmapFile(kfd, kSize, true);
                if (kMap != nullptr) {
                    ::memcpy(mStateCache->mConvState->host<int8_t>(), kMap, kSize);
                    MNNUnmapFile(kMap, kSize);
                }
            }
            MNNCloseFile(kfd);
        } else {
            MNN_PRINT("CPULinearAttention: Failed to open prefix cache file: %s\n", pathk.c_str());
        }
        // Load recurrent state (.v file)
        auto vfd = MNNOpenFile(pathv.c_str(), MNN_FILE_READ);
        if (vfd != INVALID_FILE) {
            size_t vSize = MNNGetFileSize(vfd);
            if (vSize > 0 && vSize != INVALID_SIZE && mStateCache->mRecurrentState.get() != nullptr) {
                void* vMap = MNNMmapFile(vfd, vSize, true);
                if (vMap != nullptr) {
                    ::memcpy(mStateCache->mRecurrentState->host<int8_t>(), vMap, vSize);
                    MNNUnmapFile(vMap, vSize);
                }
            }
            MNNCloseFile(vfd);
        } else {
            MNN_PRINT("CPULinearAttention: Failed to open prefix cache file: %s\n", pathv.c_str());
        }
        mMeta->layer_index = (layer_index + 1) % mMeta->layer_nums;
    }

    // Normal execution
    if (mAttentionType == "short_conv") {
        short_conv(inputs, outputs);
    } else {
        gated_delta_rule_mnn(inputs, outputs);
    }

    // Save prefix cache to disk (PendingWrite)
    if (mMeta != nullptr && mMeta->file_name.size() > 0 && mMeta->file_flag == KVMeta::PendingWrite) {
        MNNCreateDir(mPrefixCacheDir.c_str());
        int layer_index = mMeta->layer_index;
        std::string basePath = MNNFilePathConcat(mPrefixCacheDir, mMeta->file_name) + "_" + std::to_string(layer_index);
        std::string pathk = basePath + ".k";
        std::string pathv = basePath + ".v";
        // Save conv state (.k file)
        size_t convBytes = mStateCache->mConvState->elementSize();
        auto kfd = MNNCreateFile(pathk.c_str());
        if (kfd != INVALID_FILE) {
            MNNSetFileSize(kfd, convBytes);
            void* kMap = MNNMmapFile(kfd, convBytes);
            if (kMap != nullptr) {
                ::memcpy(kMap, mStateCache->mConvState->host<int8_t>(), convBytes);
                MNNUnmapFile(kMap, convBytes);
            }
            MNNCloseFile(kfd);
        } else {
            MNN_PRINT("CPULinearAttention: Failed to create prefix cache file: %s\n", pathk.c_str());
        }
        // Save recurrent state (.v file) — may be empty for short_conv
        size_t recurrentBytes = (mStateCache->mRecurrentState.get() != nullptr) ? mStateCache->mRecurrentState->elementSize() : 0;
        auto vfd = MNNCreateFile(pathv.c_str());
        if (vfd != INVALID_FILE) {
            if (recurrentBytes > 0) {
                MNNSetFileSize(vfd, recurrentBytes);
                void* vMap = MNNMmapFile(vfd, recurrentBytes);
                if (vMap != nullptr) {
                    ::memcpy(vMap, mStateCache->mRecurrentState->host<int8_t>(), recurrentBytes);
                    MNNUnmapFile(vMap, recurrentBytes);
                }
            }
            MNNCloseFile(vfd);
        } else {
            MNN_PRINT("CPULinearAttention: Failed to create prefix cache file: %s\n", pathv.c_str());
        }
        mMeta->layer_index = (layer_index + 1) % mMeta->layer_nums;
    }

    return NO_ERROR;
}

void CPULinearAttention::gated_delta_rule_mnn(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto qkvTensor    = inputs[0];
    auto gateTensor   = inputs[1];
    auto betaTensor   = inputs[2];
    auto convWTensor  = inputs[3];
    auto outTensor    = outputs[0];

    const int8_t* qkvPtr   = qkvTensor->host<int8_t>();
    const int8_t* gatePtr  = gateTensor->host<int8_t>();
    const int8_t* betaPtr  = betaTensor->host<int8_t>();
    const int8_t* convWPtr = convWTensor->host<int8_t>();
    int8_t* outPtr         = outTensor->host<int8_t>();

    const int B       = qkvTensor->length(0);
    const int D       = qkvTensor->length(1);
    const int L       = qkvTensor->length(2);
    const int H_k     = mNumKHeads;
    const int H_v     = mNumVHeads;
    const int d_k     = mHeadKDim;
    const int d_v     = mHeadVDim;
    const int key_dim = H_k * d_k;
    const int val_dim = H_v * d_v;
    const int K_conv  = convWTensor->length(2);
    const int convStateSize = K_conv - 1;
    const bool useL2Norm = mUseQKL2Norm;
    const int gqa_factor = (H_v > H_k) ? (H_v / H_k) : 1;
    const int H = H_v;
    const int bytes = mBytes;

    // Get pre-allocated buffers
    int8_t* convPadded  = mConvPadded->host<int8_t>();
    int8_t* convOut     = mConvOut->host<int8_t>();
    int8_t* convStatePtr = mStateCache->mConvState->host<int8_t>();

    // ─── Step 1: Depthwise Conv1D + SiLU (multi-threaded across B×D channels) ───
    const int totalLen = convStateSize + L;
    const int totalChannels = B * D;
    int threadNum = static_cast<CPUBackend*>(backend())->threadNumber();

    // fp16 path uses per-thread fp32 temp buffers for vectorized Conv1D + SiLu
    float* convFp32Base = (bytes == 2) ? mConvFp32Buf->host<float>() : nullptr;

    MNN_CONCURRENCY_BEGIN(tId, threadNum) {
        // Per-thread fp32 buffers (only used for fp16 path)
        float* fp32Padded = (bytes == 2) ? convFp32Base + (int)tId * (totalLen + L) : nullptr;
        float* fp32Out    = (bytes == 2) ? fp32Padded + totalLen : nullptr;

        for (int idx = (int)tId; idx < totalChannels; idx += threadNum) {
            int d = idx % D;

            // 1a. Build padded input: cat(convState, qkv) for this channel
            int8_t* padded = convPadded + idx * totalLen * bytes;
            const int8_t* stateChannel = convStatePtr + idx * convStateSize * bytes;
            ::memcpy(padded, stateChannel, convStateSize * bytes);
            const int8_t* inputChannel = qkvPtr + idx * L * bytes;
            ::memcpy(padded + convStateSize * bytes, inputChannel, L * bytes);

            // 1b. Save conv state first (before we overwrite padded)
            const int8_t* newState = padded + (totalLen - convStateSize) * bytes;
            int8_t* dstState = convStatePtr + idx * convStateSize * bytes;
            ::memcpy(dstState, newState, convStateSize * bytes);

            // 1c. Conv1D + SiLU
            const int8_t* weight = convWPtr + d * K_conv * bytes;
            int8_t* out = convOut + idx * L * bytes;

            if (bytes == 2) {
                // fp16 path: convert to fp32, compute Conv1D + SiLu vectorized, convert back
                auto coreFn = static_cast<CPUBackend*>(backend())->functions();
                coreFn->MNNLowpToFp32((const int16_t*)padded, fp32Padded, totalLen);
                float w0 = fp32Padded[0]; // dummy, will read weights separately
#ifdef __aarch64__
                w0 = (float)((__fp16*)weight)[0];
                float w1 = (float)((__fp16*)weight)[1];
                float w2 = (float)((__fp16*)weight)[2];
                float w3 = (float)((__fp16*)weight)[3];
#else
                float w1 = 0, w2 = 0, w3 = 0;
#endif
                if (K_conv == 4) {
                    int l = 0;
                    for (; l + 3 < L; l += 4) {
                        fp32Padded[l]   = fp32Padded[l]*w0 + fp32Padded[l+1]*w1 + fp32Padded[l+2]*w2 + fp32Padded[l+3]*w3;
                        fp32Padded[l+1] = fp32Padded[l+1]*w0 + fp32Padded[l+2]*w1 + fp32Padded[l+3]*w2 + fp32Padded[l+4]*w3;
                        fp32Padded[l+2] = fp32Padded[l+2]*w0 + fp32Padded[l+3]*w1 + fp32Padded[l+4]*w2 + fp32Padded[l+5]*w3;
                        fp32Padded[l+3] = fp32Padded[l+3]*w0 + fp32Padded[l+4]*w1 + fp32Padded[l+5]*w2 + fp32Padded[l+6]*w3;
                    }
                    for (; l < L; ++l) {
                        fp32Padded[l] = fp32Padded[l]*w0 + fp32Padded[l+1]*w1 + fp32Padded[l+2]*w2 + fp32Padded[l+3]*w3;
                    }
                } else {
                    for (int l = 0; l < L; ++l) {
                        float sum = 0.0f;
                        for (int k = 0; k < K_conv; ++k) {
                            float wk = _readElement(weight, k, bytes);
                            sum += fp32Padded[l + k] * wk;
                        }
                        fp32Padded[l] = sum;
                    }
                }
                MNNSiLu(fp32Out, fp32Padded, L);
                coreFn->MNNFp32ToLowp(fp32Out, (int16_t*)out, L);
            } else {
                // fp32 path: direct compute
                float* fPadded = (float*)padded;
                if (K_conv == 4) {
                    float w0 = ((float*)weight)[0], w1 = ((float*)weight)[1];
                    float w2 = ((float*)weight)[2], w3 = ((float*)weight)[3];
                    int l = 0;
                    for (; l + 3 < L; l += 4) {
                        fPadded[l]   = fPadded[l]*w0 + fPadded[l+1]*w1 + fPadded[l+2]*w2 + fPadded[l+3]*w3;
                        fPadded[l+1] = fPadded[l+1]*w0 + fPadded[l+2]*w1 + fPadded[l+3]*w2 + fPadded[l+4]*w3;
                        fPadded[l+2] = fPadded[l+2]*w0 + fPadded[l+3]*w1 + fPadded[l+4]*w2 + fPadded[l+5]*w3;
                        fPadded[l+3] = fPadded[l+3]*w0 + fPadded[l+4]*w1 + fPadded[l+5]*w2 + fPadded[l+6]*w3;
                    }
                    for (; l < L; ++l) {
                        fPadded[l] = fPadded[l]*w0 + fPadded[l+1]*w1 + fPadded[l+2]*w2 + fPadded[l+3]*w3;
                    }
                } else {
                    for (int l = 0; l < L; ++l) {
                        float sum = 0.0f;
                        for (int k = 0; k < K_conv; ++k) sum += fPadded[l + k] * ((float*)weight)[k];
                        fPadded[l] = sum;
                    }
                }
                MNNSiLu((float*)out, fPadded, L);
            }
        }
    }
    MNN_CONCURRENCY_END();

    // ─── Step 1.5: Batch exp(gate) ───
    // Decay buffer is always fp32. Convert gate to fp32 if needed, then MNNExp.
    float* decayPtr = mDecayBuf->host<float>();
    const int gateTotalSize = B * L * H;
    if (bytes == 4) {
        float expOffset[4] = {1.0f, 0.0f, 0.0f, 0.0f};
        MNNExp(decayPtr, (const float*)gatePtr, expOffset, gateTotalSize);
    } else {
        // fp16: compute exp per-element (gate is small: B*L*H)
        for (int i = 0; i < gateTotalSize; ++i) {
            decayPtr[i] = expf(_readElement(gatePtr, i, bytes));
        }
    }

    // ─── Steps 2-5 fused: Split + L2Norm + Scale + Gated Delta Rule ───
    const float qScale = 1.0f / sqrtf((float)d_k);
    auto gcore = static_cast<CPUBackend*>(backend())->functions();
    int8_t* rnnStatePtr = mStateCache->mRecurrentState->host<int8_t>();

    const int totalHeads = B * H;

    int8_t* threadBufBase = mThreadLocalBuf->host<int8_t>();
    const int perThread = 2 * d_k + 3 * d_v;

    MNN_CONCURRENCY_BEGIN(tId, threadNum) {
        int8_t* tBuf = threadBufBase + (int)tId * perThread * bytes;
        // Local buffers in native format (fp16 or fp32)
        int8_t* q_local    = tBuf;
        int8_t* k_local    = tBuf + d_k * bytes;
        int8_t* v_local    = tBuf + 2 * d_k * bytes;
        int8_t* localVPred = tBuf + (2 * d_k + d_v) * bytes;
        int8_t* localDelta = tBuf + (2 * d_k + 2 * d_v) * bytes;

        for (int idx = (int)tId; idx < totalHeads; idx += threadNum) {
            int b = idx / H;
            int h = idx % H;
            int k_head = h / gqa_factor;

            int8_t* state = rnnStatePtr + idx * d_k * d_v * bytes;
            const int8_t* convBase = convOut + b * D * L * bytes;

            // Pre-compute base pointers for this head's channels
            const int8_t* qBase = convBase + k_head * d_k * L * bytes;
            const int8_t* kBase = convBase + (key_dim + k_head * d_k) * L * bytes;
            const int8_t* vBase = convBase + (2 * key_dim + h * d_v) * L * bytes;

            for (int t = 0; t < L; ++t) {
                // ── Step 2: Extract q_t, k_t, v_t from convOut (strided access) ──
                for (int i = 0; i < d_k; ++i) {
                    float qv = _readElement(qBase, i * L + t, bytes);
                    float kv = _readElement(kBase, i * L + t, bytes);
                    _writeElement(q_local, i, qv, bytes);
                    _writeElement(k_local, i, kv, bytes);
                }
                for (int i = 0; i < d_v; ++i) {
                    float vv = _readElement(vBase, i * L + t, bytes);
                    _writeElement(v_local, i, vv, bytes);
                }

                // ── Step 3+4: L2 Normalization + Scale (fused) ──
                if (useL2Norm) {
                    const float eps = 1e-6f;
                    float qSumSq = 0.0f, kSumSq = 0.0f;
                    for (int i = 0; i < d_k; ++i) {
                        float qi = _readElement(q_local, i, bytes);
                        float ki = _readElement(k_local, i, bytes);
                        qSumSq += qi * qi;
                        kSumSq += ki * ki;
                    }
                    float qNormScale = qScale / sqrtf(qSumSq + eps);
                    float kInvNorm = 1.0f / sqrtf(kSumSq + eps);
                    for (int i = 0; i < d_k; ++i) {
                        _writeElement(q_local, i, _readElement(q_local, i, bytes) * qNormScale, bytes);
                        _writeElement(k_local, i, _readElement(k_local, i, bytes) * kInvNorm, bytes);
                    }
                } else {
                    for (int i = 0; i < d_k; ++i) {
                        _writeElement(q_local, i, _readElement(q_local, i, bytes) * qScale, bytes);
                    }
                }

                // ── Step 5: Gated Delta Rule recurrence (optimized 2-pass) ──
                float decay  = decayPtr[b * L * H + t * H + h];
                float beta_t = _readElement(betaPtr, b * L * H + t * H + h, bytes);

                // Pass 1 (read-only): compute S^T@k and S^T@q simultaneously
                int8_t* o_t = outPtr + ((b * L + t) * H * d_v + h * d_v) * bytes;
                gcore->MNNDualMatVec((float*)state, (float*)k_local, (float*)q_local,
                                     (float*)localVPred, (float*)o_t, d_k, d_v);

                // Analytic: vPred = decay * (S^T@k), out = decay * (S^T@q) + dot(k,q) * delta
                float kq = 0.0f;
                for (int i = 0; i < d_k; ++i) {
                    kq += _readElement(k_local, i, bytes) * _readElement(q_local, i, bytes);
                }
                for (int i = 0; i < d_v; ++i) {
                    float vPred_i = decay * _readElement(localVPred, i, bytes);
                    float v_i     = _readElement(v_local, i, bytes);
                    float delta_i = beta_t * (v_i - vPred_i);
                    float out_i   = decay * _readElement(o_t, i, bytes) + kq * delta_i;
                    _writeElement(localDelta, i, delta_i, bytes);
                    _writeElement(o_t, i, out_i, bytes);
                }

                // Pass 2: S = decay*S + k⊗delta
                gcore->MNNDecayRankOneUpdate((float*)state, (float*)k_local,
                                             (float*)localDelta, decay, d_k, d_v);
            } // end timestep
        } // end head
    }
    MNN_CONCURRENCY_END();
}

void CPULinearAttention::short_conv(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto qkvTensor   = inputs[0];
    auto convWTensor = inputs[3];
    auto outTensor   = outputs[0];

    const int8_t* qkvPtr   = qkvTensor->host<int8_t>();
    const int8_t* convWPtr = convWTensor->host<int8_t>();
    int8_t* outPtr         = outTensor->host<int8_t>();

    const int B      = qkvTensor->length(0);
    const int D      = qkvTensor->length(1);   // 3H
    const int L      = qkvTensor->length(2);
    const int H      = D / 3;
    const int K_conv = convWTensor->length(2);
    const int convStateSize = K_conv - 1;
    const int bytes = mBytes;

    int8_t* convPadded   = mConvPadded->host<int8_t>();
    int8_t* convOut      = mConvOut->host<int8_t>();
    int8_t* convStatePtr = mStateCache->mConvState->host<int8_t>();

    int threadNum = static_cast<CPUBackend*>(backend())->threadNumber();
    const int totalLen = convStateSize + L;
    const int totalChannels = B * H;

    MNN_CONCURRENCY_BEGIN(tId, threadNum) {
        for (int idx = (int)tId; idx < totalChannels; idx += threadNum) {
            int b = idx / H;
            int h = idx % H;

            // 1a. Compute Bx = B_[b,h,:] * x_[b,h,:] and build padded input
            int8_t* padded = convPadded + idx * totalLen * bytes;
            const int8_t* stateChannel = convStatePtr + idx * convStateSize * bytes;
            ::memcpy(padded, stateChannel, convStateSize * bytes);

            for (int l = 0; l < L; ++l) {
                float b_val = _readElement(qkvPtr, b * D * L + h * L + l, bytes);
                float x_val = _readElement(qkvPtr, b * D * L + (2 * H + h) * L + l, bytes);
                _writeElement(padded, convStateSize + l, b_val * x_val, bytes);
            }

            // 1b. Depthwise Conv1D (no SiLU)
            int8_t* out = convOut + idx * L * bytes;
            for (int l = 0; l < L; ++l) {
                float sum = 0.0f;
                for (int k = 0; k < K_conv; ++k) {
                    sum += _readElement(padded, l + k, bytes) * _readElement(convWPtr, h * K_conv + k, bytes);
                }
                _writeElement(out, l, sum, bytes);
            }

            // 1c. Update conv state
            const int8_t* newState = padded + (totalLen - convStateSize) * bytes;
            int8_t* dstState = convStatePtr + idx * convStateSize * bytes;
            ::memcpy(dstState, newState, convStateSize * bytes);
        }
    }
    MNN_CONCURRENCY_END();

    // Step 2: y = C_ * conv_out, transpose to output [B, L, 1, H]
    MNN_CONCURRENCY_BEGIN(tId, threadNum) {
        for (int idx = (int)tId; idx < totalChannels; idx += threadNum) {
            int b = idx / H;
            int h = idx % H;

            for (int l = 0; l < L; ++l) {
                float c_val = _readElement(qkvPtr, b * D * L + (H + h) * L + l, bytes);
                float conv_val = _readElement(convOut, idx * L + l, bytes);
                _writeElement(outPtr, (b * L + l) * H + h, c_val * conv_val, bytes);
            }
        }
    }
    MNN_CONCURRENCY_END();
}

bool CPULinearAttention::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto tmp = new CPULinearAttention(bn, op);
    // Share persistent state buffers between prefill and decode Executions
    tmp->mStateCache = mStateCache;
    *dst = tmp;
    return true;
}

CPULinearAttention::CPULinearAttention(Backend *backend, const MNN::Op* op) : Execution(backend) {
    auto param = op->main_as_LinearAttentionParam();
    mAttentionType = param->attn_type()->str();
    mNumKHeads = param->num_k_heads();
    mNumVHeads = param->num_v_heads();
    mHeadKDim = param->head_k_dim();
    mHeadVDim = param->head_v_dim();
    mUseQKL2Norm = param->use_qk_l2norm();
    mBytes = static_cast<CPUBackend*>(backend)->functions()->bytes;
    mStateCache.reset(new StateCache);
    mMeta = (KVMeta*)(backend->getMetaPtr());
    mPrefixCacheDir = static_cast<CPUBackend*>(backend)->getRuntime()->hint().prefixcacheDirPath;
}

CPULinearAttention::~CPULinearAttention() {

}

class CPULinearAttentionCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPULinearAttention(backend, op);
    }
};

REGISTER_CPU_OP_CREATOR_TRANSFORMER(CPULinearAttentionCreator, OpType_LinearAttention);

} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
