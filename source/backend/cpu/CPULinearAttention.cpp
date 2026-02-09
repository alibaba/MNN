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
#include "compute/CommonOptFunction.h"
#include "core/Macro.h"
#include "core/Concurrency.h"
#include "core/BufferAllocator.hpp"
#include "core/TensorUtils.hpp"
#include "core/OpCommonUtils.hpp"
#include "core/BufferAllocator.hpp"
#include "compute/ConvolutionTiledExecutor.hpp"

namespace MNN {


ErrorCode CPULinearAttention::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto qkv = inputs[0];
    auto convWeight = inputs[3];

    int batch      = qkv->length(0);
    int convDim    = qkv->length(1);    // D = 2 * key_dim + value_dim
    int seqLen     = qkv->length(2);    // L
    int kernelSize = convWeight->length(2); // conv kernel size K
    int convStateSize = kernelSize - 1;     // padding history length

    // After GQA expansion, the effective number of heads for the recurrent state is mNumVHeads
    int H  = mNumVHeads;
    int dk = mHeadKDim;
    int dv = mHeadVDim;

    // ─── Persistent state buffers (STATIC): allocate once, shared via onClone ───
    if (mStateCache->mConvState.get() == nullptr) {
        // First time: allocate and zero-initialize
        mStateCache->mConvState.reset(Tensor::createDevice<float>({batch, convDim, convStateSize}));
        bool success = backend()->onAcquireBuffer(mStateCache->mConvState.get(), Backend::STATIC);
        if (!success) return OUT_OF_MEMORY;
        ::memset(mStateCache->mConvState->host<float>(), 0, mStateCache->mConvState->elementSize() * sizeof(float));

        mStateCache->mRecurrentState.reset(Tensor::createDevice<float>({batch, H, dk, dv}));
        success = backend()->onAcquireBuffer(mStateCache->mRecurrentState.get(), Backend::STATIC);
        if (!success) return OUT_OF_MEMORY;
        ::memset(mStateCache->mRecurrentState->host<float>(), 0, mStateCache->mRecurrentState->elementSize() * sizeof(float));
    } else if (seqLen > 1) {
        // Prefill (seqLen > 1): reset state for new sequence
        ::memset(mStateCache->mConvState->host<float>(), 0, mStateCache->mConvState->elementSize() * sizeof(float));
        ::memset(mStateCache->mRecurrentState->host<float>(), 0, mStateCache->mRecurrentState->elementSize() * sizeof(float));
    }
    // Decode (seqLen == 1): keep existing state untouched

    // ─── Temporary buffers (DYNAMIC): re-allocate when seqLen changes ───
    int totalLen = convStateSize + seqLen;
    mConvPadded.reset(Tensor::createDevice<float>({batch * convDim * totalLen}));
    bool success = backend()->onAcquireBuffer(mConvPadded.get(), Backend::DYNAMIC);
    if (!success) return OUT_OF_MEMORY;

    mConvOut.reset(Tensor::createDevice<float>({batch * convDim * seqLen}));
    success = backend()->onAcquireBuffer(mConvOut.get(), Backend::DYNAMIC);
    if (!success) return OUT_OF_MEMORY;

    mTempVPred.reset(Tensor::createDevice<float>({dv}));
    success = backend()->onAcquireBuffer(mTempVPred.get(), Backend::DYNAMIC);
    if (!success) return OUT_OF_MEMORY;

    mTempDelta.reset(Tensor::createDevice<float>({dv}));
    success = backend()->onAcquireBuffer(mTempDelta.get(), Backend::DYNAMIC);
    if (!success) return OUT_OF_MEMORY;

    backend()->onReleaseBuffer(mConvPadded.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mConvOut.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mTempVPred.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mTempDelta.get(), Backend::DYNAMIC);

    return NO_ERROR;
}

void CPULinearAttention::gated_delta_rule_ref(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // ─── Input Tensors ───
    // inputs[0]: qkv         [B, D, L]   - mixed QKV projection output (before conv)
    // inputs[1]: gate        [B, L, H]   - pre-computed log-space decay factor
    // inputs[2]: beta        [B, L, H]   - pre-computed learning rate (after sigmoid)
    // inputs[3]: conv_weight [D, 1, K]   - depthwise conv1d weight
    // outputs[0]: attn_out   [B, L, num_v_heads, head_v_dim]
    //
    // Persistent states (member variables):
    //   mStateCache->mConvState:      [B, D, conv_state_size]  - Conv1D padding history
    //   mStateCache->mRecurrentState: [B, H, d_k, d_v]         - Gated Delta Rule recurrent state S
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

    const int B       = qkvTensor->length(0);   // batch size
    const int D       = qkvTensor->length(1);   // conv_dim = 2*key_dim + value_dim
    const int L       = qkvTensor->length(2);   // sequence length
    const int H_k     = mNumKHeads;              // number of K/Q heads
    const int H_v     = mNumVHeads;              // number of V heads
    const int d_k     = mHeadKDim;               // per-head K dimension
    const int d_v     = mHeadVDim;               // per-head V dimension
    const int key_dim = H_k * d_k;              // total K dimension
    const int val_dim = H_v * d_v;              // total V dimension
    const int K_conv  = convWTensor->length(2);  // conv kernel size
    const int convStateSize = K_conv - 1;        // conv padding history length
    const bool useL2Norm = mUseQKL2Norm;

    // GQA expansion factor
    const int gqa_factor = (H_v > H_k) ? (H_v / H_k) : 1;
    // After GQA expansion, Q and K have H_v heads
    const int H = H_v;

    // ─── Step 1: Depthwise Conv1D + SiLU on qkv (with conv state) ───
    // Python logic:
    //   conv_input = torch.cat([conv_state, mixed_qkv], dim=-1)  # [B, D, convStateSize + L]
    //   mixed_qkv = F.silu(F.conv1d(conv_input, weight, bias=None, padding=0, groups=D))
    //   new_conv_state = conv_input[:, :, -convStateSize:]
    //
    // Build conv_input: concatenate mStateCache->mConvState [B, D, convStateSize] with qkv [B, D, L]
    const int totalLen = convStateSize + L;
    std::vector<float> convInput(B * D * totalLen, 0.0f);
    float* convStatePtr = mStateCache->mConvState->host<float>();

    for (int b = 0; b < B; ++b) {
        for (int d = 0; d < D; ++d) {
            float* dst = convInput.data() + b * D * totalLen + d * totalLen;
            // Copy conv_state history [convStateSize]
            const float* stateChannel = convStatePtr + b * D * convStateSize + d * convStateSize;
            ::memcpy(dst, stateChannel, convStateSize * sizeof(float));
            // Copy current input [L]
            const float* inputChannel = qkvPtr + b * D * L + d * L;
            ::memcpy(dst + convStateSize, inputChannel, L * sizeof(float));
        }
    }

    // Depthwise Conv1D with padding=0, output length = totalLen - K_conv + 1 = L
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
                // SiLU activation: x * sigmoid(x)
                float sigmoid_val = 1.0f / (1.0f + expf(-sum));
                out[l] = sum * sigmoid_val;
            }
        }
    }

    // Update mStateCache->mConvState: new_conv_state = conv_input[:, :, -convStateSize:]
    for (int b = 0; b < B; ++b) {
        for (int d = 0; d < D; ++d) {
            const float* src = convInput.data() + b * D * totalLen + d * totalLen + (totalLen - convStateSize);
            float* dst       = convStatePtr + b * D * convStateSize + d * convStateSize;
            ::memcpy(dst, src, convStateSize * sizeof(float));
        }
    }

    // ─── Step 2: Split Q, K, V and transpose ───
    // convOut layout: [B, D, L] where D = key_dim + key_dim + val_dim
    // After transpose: [B, L, D], then split along last dim:
    //   Q: [B, L, key_dim] -> reshape [B, L, H_k, d_k]
    //   K: [B, L, key_dim] -> reshape [B, L, H_k, d_k]
    //   V: [B, L, val_dim] -> reshape [B, L, H_v, d_v]

    // Allocate Q, K, V in [B, L, H, dim] layout (after GQA expansion)
    std::vector<float> Q(B * L * H * d_k, 0.0f);
    std::vector<float> K(B * L * H * d_k, 0.0f);
    std::vector<float> V(B * L * H * d_v, 0.0f);

    for (int b = 0; b < B; ++b) {
        for (int l = 0; l < L; ++l) {
            // Q part: channels [0, key_dim)
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
            // K part: channels [key_dim, 2*key_dim)
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
            // V part: channels [2*key_dim, 2*key_dim + val_dim)
            for (int h = 0; h < H_v; ++h) {
                for (int dv = 0; dv < d_v; ++dv) {
                    int srcChannel = 2 * key_dim + h * d_v + dv;
                    float val = convOut[b * D * L + srcChannel * L + l];
                    V[(b * L + l) * H * d_v + h * d_v + dv] = val;
                }
            }
        }
    }

    // ─── Step 3: Optional L2 Normalization on Q and K ───
    if (useL2Norm) {
        const float eps = 1e-6f;
        for (int i = 0; i < B * L * H; ++i) {
            // L2 norm Q
            float* qHead = Q.data() + i * d_k;
            float sumSq = 0.0f;
            for (int dk = 0; dk < d_k; ++dk) {
                sumSq += qHead[dk] * qHead[dk];
            }
            float invNorm = 1.0f / sqrtf(sumSq + eps);
            for (int dk = 0; dk < d_k; ++dk) {
                qHead[dk] *= invNorm;
            }
            // L2 norm K
            float* kHead = K.data() + i * d_k;
            sumSq = 0.0f;
            for (int dk = 0; dk < d_k; ++dk) {
                sumSq += kHead[dk] * kHead[dk];
            }
            invNorm = 1.0f / sqrtf(sumSq + eps);
            for (int dk = 0; dk < d_k; ++dk) {
                kHead[dk] *= invNorm;
            }
        }
    }

    // ─── Step 4: Scale Q by 1/sqrt(d_k) ───
    const float qScale = 1.0f / sqrtf((float)d_k);
    for (int i = 0; i < B * L * H * d_k; ++i) {
        Q[i] *= qScale;
    }

    // ─── Step 5: Gated Delta Rule (core recurrence with persistent state) ───
    // Per-step formula (for each batch, each head independently):
    //   S_t  = S_{t-1} * exp(g_t)              // decay old memory
    //   v_pred = S_t^T @ k_t                   // predict value for current key
    //   delta  = beta_t * (v_t - v_pred)       // prediction error * learning rate
    //   S_t    = S_t + k_t @ delta^T           // update memory (outer product)
    //   o_t    = S_t^T @ q_t                   // query the memory
    //
    // Python logic:
    //   initial_state = self.rnn_state  (mStateCache->mRecurrentState, initialized to 0 on first call)
    //   self.rnn_state = last_recurrent_state  (written back after all timesteps)

    // Load recurrent state S from mStateCache->mRecurrentState: [B, H, d_k, d_v]
    float* rnnStatePtr = mStateCache->mRecurrentState->host<float>();

    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < L; ++t) {
            for (int h = 0; h < H; ++h) {
                // State pointer: mStateCache->mRecurrentState layout [B, H, d_k, d_v]
                float* state = rnnStatePtr + (b * H + h) * d_k * d_v;

                // Pointers to current timestep data
                const float* q_t    = Q.data() + (b * L + t) * H * d_k + h * d_k;
                const float* k_t    = K.data() + (b * L + t) * H * d_k + h * d_k;
                const float* v_t    = V.data() + (b * L + t) * H * d_v + h * d_v;
                float g_t           = gatePtr[b * L * H + t * H + h];
                float beta_t        = betaPtr[b * L * H + t * H + h];

                // ── Step 5.1: Decay ── S = S * exp(g_t)
                float decay = expf(g_t);
                for (int i = 0; i < d_k * d_v; ++i) {
                    state[i] *= decay;
                }

                // ── Step 5.2: Read ── v_pred = S^T @ k_t
                std::vector<float> v_pred(d_v, 0.0f);
                for (int dk = 0; dk < d_k; ++dk) {
                    for (int dv = 0; dv < d_v; ++dv) {
                        v_pred[dv] += state[dk * d_v + dv] * k_t[dk];
                    }
                }

                // ── Step 5.3: Delta ── delta = beta_t * (v_t - v_pred)
                std::vector<float> delta(d_v);
                for (int dv = 0; dv < d_v; ++dv) {
                    delta[dv] = beta_t * (v_t[dv] - v_pred[dv]);
                }

                // ── Step 5.4: Write ── S += k_t @ delta^T (outer product)
                for (int dk = 0; dk < d_k; ++dk) {
                    for (int dv = 0; dv < d_v; ++dv) {
                        state[dk * d_v + dv] += k_t[dk] * delta[dv];
                    }
                }

                // ── Step 5.5: Query ── o_t = S^T @ q_t
                float* o_t = outPtr + (b * L + t) * H * d_v + h * d_v;
                for (int dv = 0; dv < d_v; ++dv) {
                    float sum = 0.0f;
                    for (int dk = 0; dk < d_k; ++dk) {
                        sum += state[dk * d_v + dv] * q_t[dk];
                    }
                    o_t[dv] = sum;
                }
            } // end head
        } // end timestep
    } // end batch
    // mStateCache->mRecurrentState is updated in-place (state pointer writes directly to mStateCache->mRecurrentState's buffer)
}

ErrorCode CPULinearAttention::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    gated_delta_rule_mnn(inputs, outputs);
    return NO_ERROR;
}

void CPULinearAttention::gated_delta_rule_mnn(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // ─── Input Tensors (same layout as gated_delta_rule_ref) ───
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

    // Get pre-allocated buffers
    float* convPadded = mConvPadded->host<float>();
    float* convOut    = mConvOut->host<float>();
    float* convStatePtr = mStateCache->mConvState->host<float>();

    // ─── Step 1: Depthwise Conv1D + SiLU (multi-threaded across B×D channels) ───
    // Each channel is independent, so we parallelize across B*D work items.
    // Fused pipeline per channel: build padded → conv1d → update state → SiLU
    const int totalLen = convStateSize + L;
    const int totalChannels = B * D;
    int threadNum = static_cast<CPUBackend*>(backend())->threadNumber();

    MNN_CONCURRENCY_BEGIN(tId, threadNum) {
        for (int idx = (int)tId; idx < totalChannels; idx += threadNum) {
            int b = idx / D;
            int d = idx % D;

            // 1a. Build padded input: cat(convState, qkv) for this channel
            float* padded = convPadded + idx * totalLen;
            const float* stateChannel = convStatePtr + idx * convStateSize;
            ::memcpy(padded, stateChannel, convStateSize * sizeof(float));
            const float* inputChannel = qkvPtr + idx * L;
            ::memcpy(padded + convStateSize, inputChannel, L * sizeof(float));

            // 1b. Conv1D (valid convolution, output length = L)
            const float* weight = convWPtr + d * K_conv;
            float* out = convOut + idx * L;
            for (int l = 0; l < L; ++l) {
                float sum = 0.0f;
                for (int k = 0; k < K_conv; ++k) {
                    sum += padded[l + k] * weight[k];
                }
                out[l] = sum;
            }

            // 1c. Update conv state: keep last (K-1) elements of padded input
            const float* newState = padded + (totalLen - convStateSize);
            float* dstState = convStatePtr + idx * convStateSize;
            ::memcpy(dstState, newState, convStateSize * sizeof(float));

            // 1d. SiLU activation (non-in-place: reuse padded buffer as scratch)
            // NOTE: MNNSiLu CANNOT be called in-place because MNNExp overwrites dst
            // before the final division reads src. Reuse padded[] as scratch (safe
            // because conv state was already saved above, totalLen >= L).
            ::memcpy(padded, out, L * sizeof(float));
            MNNSiLu(out, padded, L);
        }
    }
    MNN_CONCURRENCY_END();

    // ─── Steps 2-5 fused: Split + L2Norm + Scale + Gated Delta Rule ───
    // Multi-threaded across B * H_v (each V-head is independent).
    // Q/K are read from convOut using GQA mapping (no expansion needed).
    // convOut layout: [B, D, L], access: convOut[b*D*L + channel*L + t]
    const float qScale = 1.0f / sqrtf((float)d_k);
    auto gcore = static_cast<CPUBackend*>(backend())->functions();
    float* rnnStatePtr = mStateCache->mRecurrentState->host<float>();

    // MatMulParam for S^T @ vec (constant across all heads)
    MatMulParam matParam;
    matParam.e = 1;
    matParam.l = d_k;
    matParam.h = d_v;
    matParam.numberThread = 1;
    matParam.ATranspose = false;
    matParam.BTranspose = false;

    const int totalHeads = B * H;

    MNN_CONCURRENCY_BEGIN(tId, threadNum) {
        // Per-thread local buffers (allocated once per thread)
        std::vector<float> q_local(d_k);
        std::vector<float> k_local(d_k);
        std::vector<float> v_local(d_v);
        std::vector<float> localVPred(d_v);
        std::vector<float> localDelta(d_v);

        for (int idx = (int)tId; idx < totalHeads; idx += threadNum) {
            int b = idx / H;
            int h = idx % H;                      // V-head index
            int k_head = h / gqa_factor;           // GQA: corresponding K-head

            float* state = rnnStatePtr + idx * d_k * d_v;
            const float* convBase = convOut + b * D * L;

            for (int t = 0; t < L; ++t) {
                // ── Step 2: Extract q_t, k_t, v_t from convOut (transpose on the fly) ──
                for (int i = 0; i < d_k; ++i) {
                    q_local[i] = convBase[(k_head * d_k + i) * L + t];
                    k_local[i] = convBase[(key_dim + k_head * d_k + i) * L + t];
                }
                for (int i = 0; i < d_v; ++i) {
                    v_local[i] = convBase[(2 * key_dim + h * d_v + i) * L + t];
                }

                // ── Step 3: Optional L2 Normalization on q_t and k_t ──
                if (useL2Norm) {
                    const float eps = 1e-6f;
                    float sumSq = 0.0f;
                    for (int i = 0; i < d_k; ++i) sumSq += q_local[i] * q_local[i];
                    float invNorm = 1.0f / sqrtf(sumSq + eps);
                    for (int i = 0; i < d_k; ++i) q_local[i] *= invNorm;

                    sumSq = 0.0f;
                    for (int i = 0; i < d_k; ++i) sumSq += k_local[i] * k_local[i];
                    invNorm = 1.0f / sqrtf(sumSq + eps);
                    for (int i = 0; i < d_k; ++i) k_local[i] *= invNorm;
                }

                // ── Step 4: Scale q_t by 1/sqrt(d_k) ──
                for (int i = 0; i < d_k; ++i) q_local[i] *= qScale;

                // ── Step 5: Gated Delta Rule recurrence ──
                float g_t    = gatePtr[b * L * H + t * H + h];
                float beta_t = betaPtr[b * L * H + t * H + h];

                // 5.1 Decay: S = S * exp(g_t)
                float decay = expf(g_t);
                MNNScaleAndAddBiasScalar(state, state, 0.0f, decay, d_k * d_v);

                // 5.2 Read: v_pred = S^T @ k_t
                gcore->MNNComputeMatMulForE_1(k_local.data(), state, localVPred.data(),
                                               nullptr, &matParam, 0);

                // 5.3 Delta: delta = beta_t * (v_t - v_pred)
                for (int i = 0; i < d_v; ++i) {
                    localDelta[i] = beta_t * (v_local[i] - localVPred[i]);
                }

                // 5.4 Write: S += k_t @ delta^T (outer product)
                for (int di = 0; di < d_k; ++di) {
                    float k_val = k_local[di];
                    for (int dj = 0; dj < d_v; ++dj) {
                        state[di * d_v + dj] += k_val * localDelta[dj];
                    }
                }

                // 5.5 Query: o_t = S^T @ q_t
                float* o_t = outPtr + (b * L + t) * H * d_v + h * d_v;
                gcore->MNNComputeMatMulForE_1(q_local.data(), state, o_t,
                                               nullptr, &matParam, 0);
            } // end timestep
        } // end head
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
    mStateCache.reset(new StateCache);
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