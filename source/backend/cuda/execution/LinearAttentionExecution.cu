#include "LinearAttentionExecution.hpp"
#include "core/TensorUtils.hpp"
#include "MNNCUDADefine.hpp"
#include <cuda_fp16.h>
#include <float.h>

namespace MNN {
namespace CUDA {

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

template<typename T = void>
static inline T* getDevPtr(const Tensor* t) {
    if (!t || t->deviceId() == 0) return nullptr;
    return reinterpret_cast<T*>(t->deviceId());
}

static inline bool isC4Tensor(const Tensor* tensor) {
    return TensorUtils::getDescribe(tensor)->dimensionFormat == MNN_DATA_FORMAT_NC4HW4;
}

static inline void linearAttentionDims(const Tensor* qkv, int& batch, int& convDim, int& seqLen) {
    if (isC4Tensor(qkv)) {
        batch = 1;
        seqLen = qkv->length(0);
        convDim = qkv->length(1);
        return;
    }
    batch = qkv->length(0);
    convDim = qkv->length(1);
    seqLen = qkv->length(2);
}

template <typename T>
__device__ __forceinline__ float read_qkv(const T* input, int b, int d, int l, int D, int L, bool inputC4) {
    const int packedD = ((D + PACK_NUMBER - 1) / PACK_NUMBER) * PACK_NUMBER;
    const int offset = inputC4 ? (b * L + l) * packedD + d : (b * D + d) * L + l;
    return (float)input[offset];
}

template <typename T>
__device__ __forceinline__ float read_token_channel(const T* input, int b, int l, int c, int L, int C,
                                                    bool inputC4) {
    const int packedC = ((C + PACK_NUMBER - 1) / PACK_NUMBER) * PACK_NUMBER;
    const int offset = inputC4 ? (b * L + l) * packedC + c : (b * L + l) * C + c;
    return (float)input[offset];
}

template <typename T>
__device__ __forceinline__ void write_token_channel(T* output, int token, int c, int C, bool outputC4, float value) {
    const int packedC = ((C + PACK_NUMBER - 1) / PACK_NUMBER) * PACK_NUMBER;
    const int offset = outputC4 ? token * packedC + c : token * C + c;
    output[offset] = (T)value;
}

// ============================================================================
// Kernel 1: Depthwise Conv1D + SiLU (fused)
// ============================================================================
template <typename T>
__global__ void conv1d_silu_kernel(const T* __restrict__ qkvInput,   // [B, D, L]
                                   const T* __restrict__ convWeight, // [D, 1, K]
                                   float* __restrict__ convState,    // [B, D, convStateSize]
                                   float* __restrict__ convOutFp32,  // [B, D, L]
                                   int B, int D, int L, int K_conv, int convStateSize, bool inputC4) {
    int channelIdx = blockIdx.x;
    if (channelIdx >= B * D) return;

    int d = channelIdx % D;
    int b = channelIdx / D;
    const T* weight = convWeight + d * K_conv;
    float* outFp32 = convOutFp32 + channelIdx * L;

    extern __shared__ float smem[];
    float* wShared = smem;
    float* padded = smem + K_conv;

    for (int i = threadIdx.x; i < K_conv; i += blockDim.x)
        wShared[i] = (float)weight[i];

    int totalLen = convStateSize + L;
    if (convState != nullptr) {
        float* state = convState + channelIdx * convStateSize;
        for (int i = threadIdx.x; i < convStateSize; i += blockDim.x)
            padded[i] = state[i];
    }
    for (int i = threadIdx.x; i < L; i += blockDim.x)
        padded[convStateSize + i] = read_qkv(qkvInput, b, d, i, D, L, inputC4);
    __syncthreads();

    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < K_conv; ++k)
            sum += padded[l + k] * wShared[k];
        float sigmoid_val = 1.0f / (1.0f + expf(-sum));
        outFp32[l] = sum * sigmoid_val;
    }

    if (convState != nullptr && convStateSize > 0) {
        __syncthreads();
        float* state = convState + channelIdx * convStateSize;
        for (int i = threadIdx.x; i < convStateSize; i += blockDim.x)
            state[i] = padded[totalLen - convStateSize + i];
    }
}

template <typename T>
__global__ void short_conv_kernel(const T* __restrict__ qkvInput, const T* __restrict__ convWeight,
                                  float* __restrict__ convState, float* __restrict__ convOut, int B, int D, int L,
                                  int H, int K, int convStateSize, bool inputC4) {
    const int channelIdx = blockIdx.x;
    if (channelIdx >= B * H)
        return;
    const int b = channelIdx / H;
    const int h = channelIdx % H;
    extern __shared__ float padded[];

    for (int i = threadIdx.x; i < convStateSize; i += blockDim.x) {
        padded[i] = convState[channelIdx * convStateSize + i];
    }
    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        const float bValue = read_qkv(qkvInput, b, h, l, D, L, inputC4);
        const float xValue = read_qkv(qkvInput, b, 2 * H + h, l, D, L, inputC4);
        padded[convStateSize + l] = bValue * xValue;
    }
    __syncthreads();

    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += padded[l + k] * (float)convWeight[h * K + k];
        }
        convOut[channelIdx * L + l] = sum;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < convStateSize; i += blockDim.x) {
        convState[channelIdx * convStateSize + i] = padded[L + i];
    }
}

template <typename T>
__global__ void short_conv_output_kernel(const T* __restrict__ qkvInput, const float* __restrict__ convOut,
                                         T* __restrict__ output, int B, int D, int L, int H, bool inputC4,
                                         bool outputC4) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * L * H;
    if (index >= total)
        return;
    const int h = index % H;
    const int token = index / H;
    const int l = token % L;
    const int b = token / L;
    const float cValue = read_qkv(qkvInput, b, H + h, l, D, L, inputC4);
    write_token_channel(output, token, h, H, outputC4, cValue * convOut[(b * H + h) * L + l]);
}

// ============================================================================
// Transpose kernel: [B, D, L] -> [B, L, D]
// ============================================================================
#define TILE_DIM 32
#define BLOCK_ROWS 8
__global__ void transpose_BDL_to_BLD(
    const float* __restrict__ input, float* __restrict__ output,
    int B, int D, int L
) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    int batchIdx = blockIdx.z;
    const float* in = input + batchIdx * D * L;
    float* out = output + batchIdx * L * D;
    int xBase = blockIdx.x * TILE_DIM;
    int yBase = blockIdx.y * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int d = xBase + threadIdx.y + j;
        int l = yBase + threadIdx.x;
        if (d < D && l < L)
            tile[threadIdx.y + j][threadIdx.x] = in[d * L + l];
    }
    __syncthreads();
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int l = yBase + threadIdx.y + j;
        int d = xBase + threadIdx.x;
        if (l < L && d < D)
            out[l * D + d] = tile[threadIdx.x][threadIdx.y + j];
    }
}

// ============================================================================
// Kernel 2: Gated Delta Rule - Decode (L=1)
// ============================================================================
template<typename T>
__global__ void gated_delta_rule_decode_kernel(
    const float* __restrict__ convOut,
    const T* __restrict__ gateInput,
    const T* __restrict__ betaInput,
    float* __restrict__ recurrentState,
    T* __restrict__ output,
    int B, int H_k, int H_v, int d_k, int d_v,
    int key_dim, int val_dim, int D,
    int gqa_factor, bool useL2Norm, float qScale,
    bool gateC4, bool betaC4, bool outputC4
) {
    int idx = blockIdx.x;
    if (idx >= B * H_v) return;

    int b = idx / H_v;
    int h = idx % H_v;
    int k_head = h / gqa_factor;

    extern __shared__ float shared[];
    float* q_s = shared;
    float* k_s = q_s + d_k;
    float* v_s = k_s + d_k;
    float* vpred_s = v_s + d_v;
    float* delta_s = vpred_s + d_v;

    const float* convBase = convOut + b * D;
    for (int i = threadIdx.x; i < d_k; i += blockDim.x) {
        q_s[i] = convBase[k_head * d_k + i];
        k_s[i] = convBase[key_dim + k_head * d_k + i];
    }
    for (int i = threadIdx.x; i < d_v; i += blockDim.x)
        v_s[i] = convBase[2 * key_dim + h * d_v + i];
    __syncthreads();

    if (useL2Norm) {
        __shared__ float normQ, normK;
        float sumSqQ = 0.0f, sumSqK = 0.0f;
        for (int i = threadIdx.x; i < d_k; i += blockDim.x) {
            sumSqQ += q_s[i] * q_s[i];
            sumSqK += k_s[i] * k_s[i];
        }
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            sumSqQ += __shfl_down_sync(0xffffffff, sumSqQ, offset);
            sumSqK += __shfl_down_sync(0xffffffff, sumSqK, offset);
        }
        __shared__ float warpSumsQ[32], warpSumsK[32];
        int wid = threadIdx.x / warpSize, lid = threadIdx.x % warpSize;
        if (lid == 0) { warpSumsQ[wid] = sumSqQ; warpSumsK[wid] = sumSqK; }
        __syncthreads();
        if (threadIdx.x == 0) {
            int nw = (blockDim.x + warpSize - 1) / warpSize;
            float tQ = 0, tK = 0;
            for (int w = 0; w < nw; w++) { tQ += warpSumsQ[w]; tK += warpSumsK[w]; }
            normQ = 1.0f / sqrtf(tQ + 1e-6f);
            normK = 1.0f / sqrtf(tK + 1e-6f);
        }
        __syncthreads();
        for (int i = threadIdx.x; i < d_k; i += blockDim.x) { q_s[i] *= normQ; k_s[i] *= normK; }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < d_k; i += blockDim.x) q_s[i] *= qScale;
    __syncthreads();

    float decay = expf(read_token_channel(gateInput, b, 0, h, 1, H_v, gateC4));
    float beta_t = read_token_channel(betaInput, b, 0, h, 1, H_v, betaC4);
    float* state = recurrentState + (b * H_v + h) * d_k * d_v;
    int stateSize = d_k * d_v;
    int stateSize4 = stateSize / 4;
    int dv4 = d_v / 4;
    float4* state4 = reinterpret_cast<float4*>(state);

    for (int i = threadIdx.x; i < stateSize4; i += blockDim.x) {
        float4 s = state4[i];
        s.x *= decay; s.y *= decay; s.z *= decay; s.w *= decay;
        state4[i] = s;
    }
    for (int i = stateSize4 * 4 + threadIdx.x; i < stateSize; i += blockDim.x)
        state[i] *= decay;
    __syncthreads();

    for (int j = threadIdx.x; j < d_v; j += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < d_k; i++) sum += state[i * d_v + j] * k_s[i];
        vpred_s[j] = sum;
    }
    __syncthreads();

    for (int j = threadIdx.x; j < d_v; j += blockDim.x)
        delta_s[j] = beta_t * (v_s[j] - vpred_s[j]);
    __syncthreads();

    for (int i = threadIdx.x; i < d_k; i += blockDim.x) {
        float k_val = k_s[i];
        float4* delta4 = reinterpret_cast<float4*>(delta_s);
        float4* row4 = reinterpret_cast<float4*>(state + i * d_v);
        for (int j4 = 0; j4 < dv4; j4++) {
            float4 d4 = delta4[j4], s4 = row4[j4];
            s4.x += k_val * d4.x; s4.y += k_val * d4.y;
            s4.z += k_val * d4.z; s4.w += k_val * d4.w;
            row4[j4] = s4;
        }
        for (int j = dv4 * 4; j < d_v; j++)
            state[i * d_v + j] += k_val * delta_s[j];
    }
    __syncthreads();

    for (int j = threadIdx.x; j < d_v; j += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < d_k; i++) sum += state[i * d_v + j] * q_s[i];
        write_token_channel(output, b * H_v + h, j, d_v, outputC4, sum);
    }
}

// ============================================================================
// Kernel 3: Gated Delta Rule - Prefill (L>1) — REGISTER-TILED STATE
//
// 256 threads = 2 * d_v. Each thread holds d_k/2 state elements in registers.
// Thread t: column j = t % d_v, rows = even (t < d_v) or odd (t >= d_v).
// State access is pure register ops — no shared/global memory for state!
// Only k_s, q_s, v_s, delta_s use shared memory (small vectors).
//
// Requires: d_k <= 128 (so d_k/2 <= 64 register floats per thread).
// ============================================================================
#define MAX_HALF_DK 64

template<typename T>
__global__ __launch_bounds__(256, 1)
void gated_delta_rule_prefill_kernel(
    const float* __restrict__ convOutTransposed,  // [B, L, D]
    const T* __restrict__ gateInput,              // [B, L, H_v]
    const T* __restrict__ betaInput,              // [B, L, H_v]
    float* __restrict__ recurrentState,           // [B, H_v, d_k, d_v]
    T* __restrict__ output,                       // [B, L, H_v, d_v]
    int B, int L, int H_k, int H_v, int d_k, int d_v,
    int key_dim, int val_dim, int D,
    int gqa_factor, bool useL2Norm, float qScale,
    bool gateC4, bool betaC4, bool outputC4
) {
    int idx = blockIdx.x;
    if (idx >= B * H_v) return;

    int b = idx / H_v;
    int h = idx % H_v;
    int k_head = h / gqa_factor;

    const bool stateThread = threadIdx.x < 2 * d_v;
    int myJ = stateThread ? threadIdx.x % d_v : 0;    // my column in state matrix
    int myPart = stateThread ? threadIdx.x / d_v : 0; // 0 = even rows, 1 = odd rows

    // Shared memory: partial[blockDim] + q[dk] + k[dk] + v[dv] + delta[dv]
    extern __shared__ float smem[];
    float* partial_buf = smem;
    float* q_s = partial_buf + blockDim.x;
    float* k_s = q_s + d_k;
    float* v_s = k_s + d_k;
    float* delta_s = v_s + d_v;

    // Load state into registers: thread holds state[myPart+0*2..myPart+63*2][myJ]
    float* globalState = recurrentState + (b * H_v + h) * d_k * d_v;
    float S[MAX_HALF_DK];
    #pragma unroll
    for (int e = 0; e < MAX_HALF_DK; e++) {
        int myI = myPart + e * 2;
        S[e] = (stateThread && myI < d_k) ? globalState[myI * d_v + myJ] : 0.0f;
    }

    const float* convBase = convOutTransposed + b * L * D;

    for (int t = 0; t < L; ++t) {
        // Load q, k, v from transposed layout (coalesced)
        const float* convT = convBase + t * D;
        for (int i = threadIdx.x; i < d_k; i += blockDim.x) {
            q_s[i] = convT[k_head * d_k + i];
            k_s[i] = convT[key_dim + k_head * d_k + i];
        }
        for (int i = threadIdx.x; i < d_v; i += blockDim.x)
            v_s[i] = convT[2 * key_dim + h * d_v + i];
        __syncthreads();

        // L2 normalization
        if (useL2Norm) {
            __shared__ float normQ, normK;
            float sumSqQ = 0.0f, sumSqK = 0.0f;
            for (int i = threadIdx.x; i < d_k; i += blockDim.x) {
                sumSqQ += q_s[i] * q_s[i];
                sumSqK += k_s[i] * k_s[i];
            }
            for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
                sumSqQ += __shfl_down_sync(0xffffffff, sumSqQ, offset);
                sumSqK += __shfl_down_sync(0xffffffff, sumSqK, offset);
            }
            __shared__ float warpSumsQ[8], warpSumsK[8];
            int wid = threadIdx.x / warpSize, lid = threadIdx.x % warpSize;
            if (lid == 0) { warpSumsQ[wid] = sumSqQ; warpSumsK[wid] = sumSqK; }
            __syncthreads();
            if (threadIdx.x == 0) {
                int nw = (blockDim.x + warpSize - 1) / warpSize;
                float tQ = 0, tK = 0;
                for (int w = 0; w < nw; w++) { tQ += warpSumsQ[w]; tK += warpSumsK[w]; }
                normQ = 1.0f / sqrtf(tQ + 1e-6f);
                normK = 1.0f / sqrtf(tK + 1e-6f);
            }
            __syncthreads();
            for (int i = threadIdx.x; i < d_k; i += blockDim.x) { q_s[i] *= normQ; k_s[i] *= normK; }
            __syncthreads();
        }

        for (int i = threadIdx.x; i < d_k; i += blockDim.x) q_s[i] *= qScale;
        __syncthreads();

        float decay = expf(read_token_channel(gateInput, b, t, h, L, H_v, gateC4));
        float beta_t = read_token_channel(betaInput, b, t, h, L, H_v, betaC4);

        // Preload k vector into registers (eliminates shared memory reads in inner loops)
        float vec_reg[MAX_HALF_DK];
        #pragma unroll
        for (int e = 0; e < MAX_HALF_DK; e++) {
            int myI = myPart + e * 2;
            vec_reg[e] = (stateThread && myI < d_k) ? k_s[myI] : 0.0f;
        }

        // 5.1 Decay: pure register ops!
        #pragma unroll
        for (int e = 0; e < MAX_HALF_DK; e++)
            S[e] *= decay;

        // 5.2 Read: v_pred[j] = sum_i S[i][j] * k[i] — all register ops
        float partial_read = 0.0f;
        #pragma unroll
        for (int e = 0; e < MAX_HALF_DK; e++)
            partial_read += S[e] * vec_reg[e];
        partial_buf[threadIdx.x] = partial_read;
        __syncthreads();

        // Combine + delta
        float vpred;
        if (threadIdx.x < d_v)
            vpred = partial_buf[threadIdx.x] + partial_buf[threadIdx.x + d_v];
        if (threadIdx.x < d_v)
            delta_s[threadIdx.x] = beta_t * (v_s[threadIdx.x] - vpred);
        __syncthreads();

        // 5.4 Write: S[i][j] += k[i] * delta[j] — register ops
        if (stateThread) {
            float my_delta = delta_s[myJ];
            #pragma unroll
            for (int e = 0; e < MAX_HALF_DK; e++)
                S[e] += vec_reg[e] * my_delta;
        }

        // Preload q vector (reuse vec_reg)
        #pragma unroll
        for (int e = 0; e < MAX_HALF_DK; e++) {
            int myI = myPart + e * 2;
            vec_reg[e] = (stateThread && myI < d_k) ? q_s[myI] : 0.0f;
        }

        // 5.5 Query: o[j] = sum_i S[i][j] * q[i] — all register ops
        float partial_query = 0.0f;
        #pragma unroll
        for (int e = 0; e < MAX_HALF_DK; e++)
            partial_query += S[e] * vec_reg[e];
        partial_buf[threadIdx.x] = partial_query;
        __syncthreads();

        if (threadIdx.x < d_v) {
            float result = partial_buf[threadIdx.x] + partial_buf[threadIdx.x + d_v];
            const int outputToken = (b * L + t) * H_v + h;
            write_token_channel(output, outputToken, threadIdx.x, d_v, outputC4, result);
        }
        __syncthreads();
    }

    // Store state back to global (once at end)
    #pragma unroll
    for (int e = 0; e < MAX_HALF_DK; e++) {
        int myI = myPart + e * 2;
        if (stateThread && myI < d_k)
            globalState[myI * d_v + myJ] = S[e];
    }
}

// ============================================================================
// CUDALinearAttention Implementation
// ============================================================================

CUDALinearAttention::CUDALinearAttention(Backend* backend, const MNN::Op* op) : Execution(backend) {
    mCudaBackend = static_cast<CUDABackend*>(backend);
    mMeta = (KVMeta*)(backend->getMetaPtr());
    auto param = op->main_as_LinearAttentionParam();
    mAttentionType = param->attn_type()->str();
    mNumKHeads = param->num_k_heads();
    mNumVHeads = param->num_v_heads();
    mHeadKDim = param->head_k_dim();
    mHeadVDim = param->head_v_dim();
    mUseQKL2Norm = param->use_qk_l2norm();
    mPrecision = mCudaBackend->getPrecision();
    mStateCache.reset(new CUDAStateCache);
}

CUDALinearAttention::~CUDALinearAttention() {
}

ErrorCode CUDALinearAttention::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    if (inputs.size() < 4 || outputs.empty())
        return INVALID_VALUE;
    auto qkv = inputs[0];
    auto convWeight = inputs[3];

    int batch = 0, convDim = 0, seqLen = 0;
    linearAttentionDims(qkv, batch, convDim, seqLen);
    int K_conv = convWeight->length(2);
    int convStateSize = K_conv - 1;
    int H = mNumVHeads;
    int dk = mHeadKDim;
    int dv = mHeadVDim;
    const bool shortConv = mAttentionType == "short_conv";
    const bool gatedDelta = mAttentionType == "gated_delta_rule";
    const bool inputC4 = isC4Tensor(qkv);
    const int convChannels = shortConv ? convDim / 3 : convDim;

    if ((!shortConv && !gatedDelta) || batch <= 0 || convDim <= 0 || seqLen <= 0 || K_conv <= 0 || H <= 0 || dk <= 0 ||
        dv <= 0 || (gatedDelta && (dk > 128 || dv > 128)) ||
        (shortConv && (mNumKHeads != 1 || mNumVHeads != 1 || convDim % 3 != 0 || convDim / 3 != dv))) {
        MNN_ERROR("CUDA LinearAttention: invalid type, shape, or head configuration.\n");
        return INVALID_VALUE;
    }
    if (inputC4) {
        const bool outputC4 = isC4Tensor(outputs[0]);
        const bool validPackedShape =
            qkv->dimensions() == 4 && qkv->length(2) == 1 && qkv->length(3) == 1 && outputC4;
        bool validAux = true;
        if (gatedDelta) {
            validAux = isC4Tensor(inputs[1]) && isC4Tensor(inputs[2]);
        }
        if (!validPackedShape || !validAux) {
            MNN_ERROR("CUDA LinearAttention: invalid C4 input/output layout.\n");
            return INVALID_VALUE;
        }
    }

    // Use int32_t to ensure 4 bytes/element in fp16 mode
    const bool needConvStateInit = mStateCache->mConvState.get() == nullptr;
    const bool needRecurrentStateInit = gatedDelta && mStateCache->mRecurrentState.get() == nullptr;
    if (needConvStateInit || needRecurrentStateInit) {
        if (needConvStateInit) {
            int convStateTotal = ALIMAX(batch * convChannels * convStateSize, 1);
            mStateCache->mConvState.reset(Tensor::createDevice<int32_t>({convStateTotal}));
            bool success = backend()->onAcquireBuffer(mStateCache->mConvState.get(), Backend::STATIC);
            if (!success) { MNN_ERROR("LinearAttention: convState STATIC alloc failed\n"); return OUT_OF_MEMORY; }
            cudaMemset(getDevPtr<void>(mStateCache->mConvState.get()), 0, convStateTotal * sizeof(float));
        }
        if (needRecurrentStateInit) {
            int rnnStateTotal = batch * H * dk * dv;
            mStateCache->mRecurrentState.reset(Tensor::createDevice<int32_t>({rnnStateTotal}));
            bool success = backend()->onAcquireBuffer(mStateCache->mRecurrentState.get(), Backend::STATIC);
            if (!success) {
                MNN_ERROR("LinearAttention: recurrentState STATIC alloc failed\n");
                return OUT_OF_MEMORY;
            }
            cudaMemset(getDevPtr<void>(mStateCache->mRecurrentState.get()), 0, rnnStateTotal * sizeof(float));
        }
    } else if (seqLen > 1) {
        // Prefill: reset state for new sequence, UNLESS:
        // 1. Loading from prefix cache (PendingRead), or
        // 2. Reusing KV from previous inference (reuse_kv=true, i.e. previous != remove)
        bool loadingFromDisk = (mMeta != nullptr && mMeta->file_flag == KVMeta::PendingRead && mMeta->file_name.size() > 0);
        bool reusingKV = (mMeta != nullptr && mMeta->previous != mMeta->remove);
        if (!loadingFromDisk && !reusingKV) {
            if (mStateCache->mConvState.get() != nullptr)
                cudaMemset(getDevPtr<void>(mStateCache->mConvState.get()), 0,
                           mStateCache->mConvState->elementSize() * sizeof(float));
            if (mStateCache->mRecurrentState.get() != nullptr)
                cudaMemset(getDevPtr<void>(mStateCache->mRecurrentState.get()), 0,
                           mStateCache->mRecurrentState->elementSize() * sizeof(float));
        }
    }

    int convOutSize = batch * convChannels * seqLen;
    mConvOut.reset(Tensor::createDevice<int32_t>({convOutSize}));
    bool success = backend()->onAcquireBuffer(mConvOut.get(), Backend::DYNAMIC);
    if (!success) { MNN_ERROR("LinearAttention: convOut DYNAMIC alloc failed\n"); return OUT_OF_MEMORY; }

    if (gatedDelta && seqLen > 1) {
        mConvOutTransposed.reset(Tensor::createDevice<int32_t>({batch * convDim * seqLen}));
        success = backend()->onAcquireBuffer(mConvOutTransposed.get(), Backend::DYNAMIC);
        if (!success) return OUT_OF_MEMORY;
        backend()->onReleaseBuffer(mConvOutTransposed.get(), Backend::DYNAMIC);
    }
    backend()->onReleaseBuffer(mConvOut.get(), Backend::DYNAMIC);

    return NO_ERROR;
}

ErrorCode CUDALinearAttention::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // onResize() may be skipped when shapes are unchanged. Ensure state is reset here too.
    int resetBatch = 0, resetDim = 0, seqLen = 0;
    linearAttentionDims(inputs[0], resetBatch, resetDim, seqLen);
    if (seqLen > 1 && mMeta != nullptr && mMeta->previous == mMeta->remove) {
        bool loadingFromDisk = (mMeta->file_flag == KVMeta::PendingRead && mMeta->file_name.size() > 0);
        if (!loadingFromDisk) {
            if (mStateCache->mConvState.get() != nullptr) {
                cudaMemset(getDevPtr<void>(mStateCache->mConvState.get()), 0,
                           mStateCache->mConvState->elementSize() * sizeof(float));
            }
            if (mStateCache->mRecurrentState.get() != nullptr)
                cudaMemset(getDevPtr<void>(mStateCache->mRecurrentState.get()), 0,
                           mStateCache->mRecurrentState->elementSize() * sizeof(float));
        }
    }
    if (mMeta != nullptr && !mMeta->file_name.empty() &&
        (mMeta->file_flag == KVMeta::PendingWrite || mMeta->file_flag == KVMeta::PendingRead) &&
        mMeta->previous == mMeta->remove && mMeta->layer_nums > 0) {
        // Keep the shared prefix index aligned with full-attention layers in hybrid models.
        mMeta->layer_index = (mMeta->layer_index + 1) % mMeta->layer_nums;
    }

    auto qkvTensor = inputs[0];
    auto gateTensor = inputs[1];
    auto betaTensor = inputs[2];
    auto convWTensor = inputs[3];
    auto outTensor = outputs[0];

    int B = 0, D = 0, L = 0;
    linearAttentionDims(qkvTensor, B, D, L);
    int H_k = mNumKHeads;
    int H_v = mNumVHeads;
    int dk = mHeadKDim;
    int dv = mHeadVDim;
    int key_dim = H_k * dk;
    int val_dim = H_v * dv;
    int K_conv = convWTensor->length(2);
    int convStateSize = K_conv - 1;
    int gqa_factor = (H_v > H_k) ? (H_v / H_k) : 1;
    float qScale = 1.0f / sqrtf((float)dk);

    cudaStream_t stream = 0;
    bool useFp16 = (mPrecision == 2);
    const bool inputC4 = isC4Tensor(qkvTensor);
    const bool gateC4 = isC4Tensor(gateTensor);
    const bool betaC4 = isC4Tensor(betaTensor);
    const bool outputC4 = isC4Tensor(outTensor);

    if (mAttentionType == "short_conv") {
        const int shortHeads = D / 3;
        const int blockSize = L == 1 ? 32 : 128;
        const int sharedBytes = (convStateSize + L) * sizeof(float);
        float* convStatePtr =
            mStateCache->mConvState.get() != nullptr ? getDevPtr<float>(mStateCache->mConvState.get()) : nullptr;
        float* convOutPtr = getDevPtr<float>(mConvOut.get());
        if (useFp16) {
            short_conv_kernel<half><<<B * shortHeads, blockSize, sharedBytes, stream>>>(
                getDevPtr<half>(qkvTensor), getDevPtr<half>(convWTensor), convStatePtr, convOutPtr, B, D, L, shortHeads,
                K_conv, convStateSize, inputC4);
        } else {
            short_conv_kernel<float><<<B * shortHeads, blockSize, sharedBytes, stream>>>(
                getDevPtr<float>(qkvTensor), getDevPtr<float>(convWTensor), convStatePtr, convOutPtr, B, D, L,
                shortHeads, K_conv, convStateSize, inputC4);
        }
        checkKernelErrors;

        const int total = B * L * shortHeads;
        const int outputBlock = 256;
        if (useFp16) {
            short_conv_output_kernel<half><<<UP_DIV(total, outputBlock), outputBlock, 0, stream>>>(
                getDevPtr<half>(qkvTensor), convOutPtr, getDevPtr<half>(outTensor), B, D, L, shortHeads, inputC4,
                outputC4);
        } else {
            short_conv_output_kernel<float><<<UP_DIV(total, outputBlock), outputBlock, 0, stream>>>(
                getDevPtr<float>(qkvTensor), convOutPtr, getDevPtr<float>(outTensor), B, D, L, shortHeads, inputC4,
                outputC4);
        }
        checkKernelErrors;
        return NO_ERROR;
    }

    // Step 1: Conv1D + SiLU -> [B, D, L]
    {
        int totalChannels = B * D;
        int smemSize = (K_conv + convStateSize + L) * sizeof(float);
        int blockSize = (L == 1) ? 32 : 128;

        float* convStatePtr = (mStateCache->mConvState.get() != nullptr) ?
                               getDevPtr<float>(mStateCache->mConvState.get()) : nullptr;
        float* convOutPtr = getDevPtr<float>(mConvOut.get());

        if (useFp16) {
            conv1d_silu_kernel<half><<<totalChannels, blockSize, smemSize, stream>>>(
                getDevPtr<half>(qkvTensor), getDevPtr<half>(convWTensor), convStatePtr, convOutPtr, B, D, L, K_conv,
                convStateSize, inputC4);
        } else {
            conv1d_silu_kernel<float><<<totalChannels, blockSize, smemSize, stream>>>(
                getDevPtr<float>(qkvTensor), getDevPtr<float>(convWTensor), convStatePtr, convOutPtr, B, D, L, K_conv,
                convStateSize, inputC4);
        }
        checkKernelErrors;
    }

    // Steps 2-5: Gated Delta Rule
    {
        int totalHeads = B * H_v;
        float* convOutPtr = getDevPtr<float>(mConvOut.get());
        float* rnnStatePtr = getDevPtr<float>(mStateCache->mRecurrentState.get());

        if (L == 1) {
            // Decode: state in global memory
            int smemSize = (2 * dk + 3 * dv) * sizeof(float);
            if (mUseQKL2Norm) smemSize += (32 + 32 + 2) * sizeof(float);
            int blockSize = (max(dk, dv) <= 64) ? 64 : 128;

            if (useFp16) {
                gated_delta_rule_decode_kernel<half><<<totalHeads, blockSize, smemSize, stream>>>(
                    convOutPtr, getDevPtr<half>(gateTensor), getDevPtr<half>(betaTensor),
                    rnnStatePtr, getDevPtr<half>(outTensor),
                    B, H_k, H_v, dk, dv, key_dim, val_dim, D,
                    gqa_factor, mUseQKL2Norm, qScale, gateC4, betaC4, outputC4);
            } else {
                gated_delta_rule_decode_kernel<float><<<totalHeads, blockSize, smemSize, stream>>>(
                    convOutPtr, getDevPtr<float>(gateTensor), getDevPtr<float>(betaTensor),
                    rnnStatePtr, getDevPtr<float>(outTensor),
                    B, H_k, H_v, dk, dv, key_dim, val_dim, D,
                    gqa_factor, mUseQKL2Norm, qScale, gateC4, betaC4, outputC4);
            }
        } else {
            // Prefill: transpose + register-tiled kernel
            float* convOutTransPtr = getDevPtr<float>(mConvOutTransposed.get());
            {
                dim3 block(TILE_DIM, BLOCK_ROWS);
                dim3 grid((D + TILE_DIM - 1) / TILE_DIM, (L + TILE_DIM - 1) / TILE_DIM, B);
                transpose_BDL_to_BLD<<<grid, block, 0, stream>>>(convOutPtr, convOutTransPtr, B, D, L);
            }

            // smem: partial[blockSize] + q[dk] + k[dk] + v[dv] + delta[dv]
            int blockSize = UP_DIV(2 * dv, 32) * 32;
            int smemSize = (blockSize + 2 * dk + 2 * dv) * sizeof(float);
            if (mUseQKL2Norm) smemSize += (8 + 8 + 2) * sizeof(float);

            if (useFp16) {
                gated_delta_rule_prefill_kernel<half><<<totalHeads, blockSize, smemSize, stream>>>(
                    convOutTransPtr, getDevPtr<half>(gateTensor), getDevPtr<half>(betaTensor),
                    rnnStatePtr, getDevPtr<half>(outTensor),
                    B, L, H_k, H_v, dk, dv, key_dim, val_dim, D,
                    gqa_factor, mUseQKL2Norm, qScale, gateC4, betaC4, outputC4);
            } else {
                gated_delta_rule_prefill_kernel<float><<<totalHeads, blockSize, smemSize, stream>>>(
                    convOutTransPtr, getDevPtr<float>(gateTensor), getDevPtr<float>(betaTensor),
                    rnnStatePtr, getDevPtr<float>(outTensor),
                    B, L, H_k, H_v, dk, dv, key_dim, val_dim, D,
                    gqa_factor, mUseQKL2Norm, qScale, gateC4, betaC4, outputC4);
            }
        }
    }

    return NO_ERROR;
}

bool CUDALinearAttention::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) return true;
    auto tmp = new CUDALinearAttention(bn, op);
    tmp->mStateCache = mStateCache;
    *dst = tmp;
    return true;
}

class LinearAttentionCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        auto param = op->main_as_LinearAttentionParam();
        if (param == nullptr || param->attn_type() == nullptr)
            return nullptr;
        const auto type = param->attn_type()->str();
        if (type != "gated_delta_rule" && type != "short_conv")
            return nullptr;
        return new CUDALinearAttention(backend, op);
    }
};

static CUDACreatorRegister<LinearAttentionCreator> __init_linear_attention(OpType_LinearAttention);

#endif // MNN_SUPPORT_TRANSFORMER_FUSE

} // namespace CUDA
} // namespace MNN
