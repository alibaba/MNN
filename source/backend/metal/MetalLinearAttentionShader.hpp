//
//  MetalLinearAttentionShader.hpp
//  MNN
//
//  Created by MNN on 2026/02/12.
//  Copyright 2018, Alibaba Group Holding Limited
//

#if MNN_METAL_ENABLED
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

// Parameter struct shared between CPU and GPU
// Must match the layout in MetalLinearAttention.mm
static const char* gLinearAttnConvSilu = R"metal(
#include <metal_stdlib>
using namespace metal;

#if MNN_METAL_FLOAT16_STORAGE
typedef half ftype;
#else
typedef float ftype;
#endif

struct LinearAttnParam {
    int batch;
    int conv_dim;
    int seq_len;
    int kernel_size;
    int conv_state_size;
    int num_k_heads;
    int num_v_heads;
    int head_k_dim;
    int head_v_dim;
    int key_dim;
    int val_dim;
    int gqa_factor;
    int use_l2norm;
    int qkv_c4;
    int gate_c4;
    int beta_c4;
    int output_c4;
    float q_scale;
    int commit_len;    // lazy-commit: pending tokens to replay before this block
    int pending_seq;   // length of the saved pending block
    int lazy_mode;     // 1: do not persist state after the new block (spec verify)
    // gate/beta chain fold (gate_c4 & 2 / beta_c4 & 2): per-head constants
    // -exp(A_log)[h] and dt_bias[h]; capped at 64 heads.
    float gate_coef[64];
    float gate_bias[64];
};

inline int c4_offset(int token, int channel, int token_count) {
    return ((channel >> 2) * token_count + token) * 4 + (channel & 3);
}

inline int qkv_offset(int b, int d, int l, constant LinearAttnParam& param) {
    if (param.qkv_c4) {
        return c4_offset(b * param.seq_len + l, d, param.batch * param.seq_len);
    }
    return (b * param.conv_dim + d) * param.seq_len + l;
}

// Kernel 1: Depthwise Conv1D + SiLU
// Each thread processes one (batch*channel, seq_pos) element
// Input:  qkv [B, D, L], conv_state [B, D, conv_state_size], conv_weight [D, 1, K]
// Output: conv_out [B, D, L]
// Also updates conv_state in-place
kernel void linear_attn_conv_silu(
    const device ftype* qkv         [[buffer(0)]],
    device ftype* conv_state        [[buffer(1)]],
    const device ftype* conv_weight [[buffer(2)]],
    device ftype* conv_out          [[buffer(3)]],
    constant LinearAttnParam& param [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const int B = param.batch;
    const int D = param.conv_dim;
    const int L = param.seq_len;
    const int K = param.kernel_size;
    const int css = param.conv_state_size; // K - 1

    const int total = B * D * L;
    if ((int)gid >= total) return;

    // Decompose global index -> (batch_chan, seq_pos)
    const int l = gid % L;
    const int bd = gid / L;
    const int b = bd / D;
    const int d = bd % D;

    // Compute valid convolution for position l
    // Padded input = [conv_state[b,d,:], qkv[b,d,:]]
    // conv_state has css elements, qkv has L elements
    // Total padded length = css + L
    // Output at position l: sum over k in [0, K) of padded[l+k] * weight[k]
    //   padded[l+k]: if (l+k) < css -> conv_state[b*D*css + d*css + (l+k)]
    //                else -> qkv[b*D*L + d*L + (l+k - css)]

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        int pos = l + k;  // position in padded input
        float input_val;
        if (pos < css) {
            input_val = (float)conv_state[b * D * css + d * css + pos];
        } else {
            input_val = (float)qkv[qkv_offset(b, d, pos - css, param)];
        }
        sum += input_val * (float)conv_weight[d * K + k];
    }

    // SiLU activation: x * sigmoid(x)
    float sigmoid_val = 1.0f / (1.0f + exp(-sum));
    conv_out[b * D * L + d * L + l] = (ftype)(sum * sigmoid_val);
}

// Decode specialization (L=1): the convolution and state update have the
// same one-thread-per-channel ownership, so keep the old state live until the
// convolution completes and update it before returning. This removes one
// kernel launch plus a second qkv/state traversal per linear-attention layer.
kernel void linear_attn_conv_silu_state_decode(
    const device ftype* qkv         [[buffer(0)]],
    device ftype* conv_state        [[buffer(1)]],
    const device ftype* conv_weight [[buffer(2)]],
    device ftype* conv_out          [[buffer(3)]],
    constant LinearAttnParam& param [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const int D = param.conv_dim;
    const int css = param.conv_state_size;
    const int total = param.batch * D;
    if ((int)gid >= total) return;

    const int b = gid / D;
    const int d = gid % D;
    const int K = css + 1;
    device ftype* state = conv_state + gid * css;
    const float current = (float)qkv[qkv_offset(b, d, 0, param)];

    float sum = current * (float)conv_weight[d * K + css];
    for (int i = 0; i < css; ++i) {
        sum += (float)state[i] * (float)conv_weight[d * K + i];
    }

    if (css > 0) {
        for (int i = 0; i + 1 < css; ++i) {
            state[i] = state[i + 1];
        }
        state[css - 1] = (ftype)current;
    }

    const float sigmoid_val = 1.0f / (1.0f + exp(-sum));
    conv_out[gid] = (ftype)(sum * sigmoid_val);
}

// Kernel 2: Update conv state with last (K-1) elements of padded input.
// One thread owns a complete channel so shifting the state in-place is ordered.
// padded input = [old_conv_state, qkv], total length = css + L
// new conv_state = padded[L .. L+css-1] (last css elements)
// Which maps to: if (L + i) < css -> old_state[L+i], else -> qkv[(L+i) - css]
// Simplified: new_state[i] = padded[L + i], where padded = cat(old_state, qkv)
kernel void linear_attn_conv_state_update(
    const device ftype* qkv         [[buffer(0)]],
    device ftype* conv_state        [[buffer(1)]],
    constant LinearAttnParam& param [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    const int B = param.batch;
    const int D = param.conv_dim;
    const int L = param.seq_len;
    const int css = param.conv_state_size;

    const int total = B * D;
    if ((int)gid >= total) return;

    const int b = gid / D;
    const int d = gid % D;
    device ftype* state = conv_state + gid * css;
    for (int i = 0; i < css; ++i) {
        int pos = L + i;
        if (pos < css) {
            state[i] = state[pos];
        } else {
            state[i] = qkv[qkv_offset(b, d, pos - css, param)];
        }
    }
}

kernel void linear_attn_conv_state_commit(
    const device ftype* pending_raw [[buffer(0)]],
    device ftype* conv_state        [[buffer(1)]],
    constant LinearAttnParam& param [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    const int B = param.batch;
    const int D = param.conv_dim;
    const int css = param.conv_state_size;
    const int commit_len = param.commit_len;
    const int pend_seq = param.pending_seq;
    const int total = B * D;
    if ((int)gid >= total) return;
    const int b = gid / D;
    const int d = gid % D;
    device ftype* state = conv_state + gid * css;
    const device ftype* raw = pending_raw + (b * D + d) * pend_seq;
    for (int i = 0; i < css; ++i) {
        int pos = commit_len + i;
        if (pos < css) {
            state[i] = state[pos];
        } else {
            state[i] = raw[pos - css];
        }
    }
}


kernel void linear_attn_qkvraw_save(
    const device ftype* qkv         [[buffer(0)]],
    device ftype* pending_raw       [[buffer(1)]],
    constant LinearAttnParam& param [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    const int B = param.batch;
    const int D = param.conv_dim;
    const int L = param.seq_len;
    const int total = B * D * L;
    if ((int)gid >= total) return;
    const int t = gid % L;
    const int bd = gid / L;
    const int b = bd / D;
    const int d = bd % D;
    // Write stride is the CURRENT block length; param.pending_seq is the previous block's.
    pending_raw[(b * D + d) * L + t] = qkv[qkv_offset(b, d, t, param)];
}

)metal";


static const char* gLinearAttnQKVPrepSG = R"metal(
#include <metal_stdlib>
using namespace metal;

#if MNN_METAL_FLOAT16_STORAGE
typedef half ftype;
#else
typedef float ftype;
#endif

struct LinearAttnParam {
    int batch;
    int conv_dim;
    int seq_len;
    int kernel_size;
    int conv_state_size;
    int num_k_heads;
    int num_v_heads;
    int head_k_dim;
    int head_v_dim;
    int key_dim;
    int val_dim;
    int gqa_factor;
    int use_l2norm;
    int qkv_c4;
    int gate_c4;
    int beta_c4;
    int output_c4;
    float q_scale;
    int commit_len;    // lazy-commit: pending tokens to replay before this block
    int pending_seq;   // length of the saved pending block
    int lazy_mode;     // 1: do not persist state after the new block (spec verify)
    // gate/beta chain fold (gate_c4 & 2 / beta_c4 & 2): per-head constants
    // -exp(A_log)[h] and dt_bias[h]; capped at 64 heads.
    float gate_coef[64];
    float gate_bias[64];
};

kernel void linear_attn_qkv_prep_sg(
    const device ftype* conv_out         [[buffer(0)]],
    device ftype* q_out                  [[buffer(1)]],
    device ftype* k_out                  [[buffer(2)]],
    device ftype* v_out                  [[buffer(3)]],
    constant LinearAttnParam& param      [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint sgitg  [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]])
{
    const int B = param.batch;
    const int D = param.conv_dim;
    const int L = param.seq_len;
    const int H = param.num_v_heads;
    const int key_dim = param.key_dim;
    const int gqa_factor = param.gqa_factor;
    const int use_l2norm = param.use_l2norm;
    const float q_scale = param.q_scale;

    // 4 simdgroups per threadgroup -> 128 threads / TG.
    // Each simdgroup owns one (b, t, h) element.
    const int idx = (int)(tgpig.x * 4 + sgitg);
    const int total = B * L * H;
    if (idx >= total) return;

    const int b = idx / (L * H);
    const int rem = idx % (L * H);
    const int t = rem / H;
    const int h = rem % H;
    const int k_head = h / gqa_factor;

    const device ftype* conv_base = conv_out + b * D * L;
    device ftype* dst_q = q_out + idx * HEAD_K_DIM;
    device ftype* dst_k = k_out + idx * HEAD_K_DIM;
    device ftype* dst_v = v_out + idx * HEAD_V_DIM;

    // ─── Load Q, K into per-lane registers (single pass over conv_out) ────
    //  lane i owns positions {i, i+32, i+64, ...} of the dk axis.
    float qBuf[SIMD_ITERS_K];
    float kBuf[SIMD_ITERS_K];
    #pragma unroll
    for (int ii = 0; ii < SIMD_ITERS_K; ++ii) {
        int i = lane + ii * 32;
        if (i < HEAD_K_DIM) {
            qBuf[ii] = (float)conv_base[(k_head * HEAD_K_DIM + i) * L + t];
            kBuf[ii] = (float)conv_base[(key_dim + k_head * HEAD_K_DIM + i) * L + t];
        } else {
            qBuf[ii] = 0.0f;
            kBuf[ii] = 0.0f;
        }
    }

    // ─── L2 norm (single simd-wide reduction) + q_scale ────
    float invQ, invK;
    if (use_l2norm) {
        const float eps = 1e-6f;
        float sqQ = 0.0f, sqK = 0.0f;
        #pragma unroll
        for (int ii = 0; ii < SIMD_ITERS_K; ++ii) {
            sqQ += qBuf[ii] * qBuf[ii];
            sqK += kBuf[ii] * kBuf[ii];
        }
        sqQ = simd_sum(sqQ);
        sqK = simd_sum(sqK);
        invQ = rsqrt(sqQ + eps) * q_scale;
        invK = rsqrt(sqK + eps);
    } else {
        invQ = q_scale;
        invK = 1.0f;
    }

    // ─── Write Q, K (coalesced: 32 lanes write 32 consecutive positions) ──
    #pragma unroll
    for (int ii = 0; ii < SIMD_ITERS_K; ++ii) {
        int i = lane + ii * 32;
        if (i < HEAD_K_DIM) {
            dst_q[i] = (ftype)(qBuf[ii] * invQ);
            dst_k[i] = (ftype)(kBuf[ii] * invK);
        }
    }

    // ─── V: direct copy, no normalization ────
    #pragma unroll
    for (int ii = 0; ii < SIMD_ITERS_V; ++ii) {
        int i = lane + ii * 32;
        if (i < HEAD_V_DIM) {
            dst_v[i] = conv_base[(2 * key_dim + h * HEAD_V_DIM + i) * L + t];
        }
    }
}
)metal";


static const char* gLinearAttnGatedDeltaRule = R"metal(
#include <metal_stdlib>
using namespace metal;

#if MNN_METAL_FLOAT16_STORAGE
typedef half ftype;
#else
typedef float ftype;
#endif

struct LinearAttnParam {
    int batch;
    int conv_dim;
    int seq_len;
    int kernel_size;
    int conv_state_size;
    int num_k_heads;
    int num_v_heads;
    int head_k_dim;
    int head_v_dim;
    int key_dim;
    int val_dim;
    int gqa_factor;
    int use_l2norm;
    int qkv_c4;
    int gate_c4;
    int beta_c4;
    int output_c4;
    float q_scale;
    int commit_len;    // lazy-commit: pending tokens to replay before this block
    int pending_seq;   // length of the saved pending block
    int lazy_mode;     // 1: do not persist state after the new block (spec verify)
    // gate/beta chain fold (gate_c4 & 2 / beta_c4 & 2): per-head constants
    // -exp(A_log)[h] and dt_bias[h]; capped at 64 heads.
    float gate_coef[64];
    float gate_bias[64];
};

inline int c4_offset(int token, int channel, int token_count) {
    return ((channel >> 2) * token_count + token) * 4 + (channel & 3);
}

inline int token_channel_offset(int b, int t, int c, int channel, int packed,
                                constant LinearAttnParam& param) {
    if (packed) {
        return c4_offset(b * param.seq_len + t, c, param.batch * param.seq_len);
    }
    return (b * param.seq_len + t) * channel + c;
}

// gate/beta chain fold (bit 2 of gate_c4/beta_c4): replicates the unfused
// elementwise chain op-for-op — Binary ops compute in ftype (half on fp16
// builds), Unary ops in fp32 with the MNNEXP +-87 clamp — with an ftype
// store-rounding after every op, so the folded value is bit-identical to the
// separate-dispatch chain output.
inline float linear_attn_gate_fold(float a, int h, constant LinearAttnParam& p) {
    ftype x = (ftype)a + (ftype)p.gate_bias[h];       // ADD dt_bias  (Binary, half)
    x = (ftype)exp(clamp((float)x, -87.0f, 87.0f));   // EXP          (Unary, fp32)
    x = x + (ftype)1.0f;                              // ADD +1       (Binary, half)
    x = (ftype)log((float)x);                         // LOG          (Unary, fp32)
    x = (ftype)p.gate_coef[h] * x;                    // MUL -exp(A_log)
    return (float)x;
}
inline float linear_attn_beta_fold(float b) {
    return (float)(ftype)(1.0f / (1.0f + exp(clamp(-b, -87.0f, 87.0f))));
}

inline int output_offset(int b, int t, int h, int d, constant LinearAttnParam& param) {
    int token = (b * param.seq_len + t) * param.num_v_heads + h;
    if (param.output_c4) {
        return c4_offset(token, d, param.batch * param.seq_len * param.num_v_heads);
    }
    return token * param.head_v_dim + d;
}

// Kernel 3: Extract Q, K, V and normalize/scale
// Each thread processes one (batch, L, head)
// Avoids fixed-size local arrays to support arbitrary d_k/d_v
kernel void linear_attn_qkv_prep(
    const device ftype* conv_out         [[buffer(0)]],
    device ftype* q_out                  [[buffer(1)]],
    device ftype* k_out                  [[buffer(2)]],
    device ftype* v_out                  [[buffer(3)]],
    constant LinearAttnParam& param      [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const int B = param.batch;
    const int D = param.conv_dim;
    const int L = param.seq_len;
    const int H = param.num_v_heads;
    const int d_k = param.head_k_dim;
    const int d_v = param.head_v_dim;
    const int key_dim = param.key_dim;
    const int gqa_factor = param.gqa_factor;
    const int use_l2norm = param.use_l2norm;
    const float q_scale = param.q_scale;

    const int total = B * L * H;
    if ((int)gid >= total) return;

    const int b = gid / (L * H);
    const int rem = gid % (L * H);
    const int t = rem / H;
    const int h = rem % H;
    const int k_head = h / gqa_factor;

    const device ftype* conv_base = conv_out + b * D * L;
    device ftype* dst_q = q_out + gid * d_k;
    device ftype* dst_k = k_out + gid * d_k;
    device ftype* dst_v = v_out + gid * d_v;

    if (use_l2norm) {
        const float eps = 1e-6f;
        // Pass 1: compute L2 norms for Q and K
        float sumSqQ = 0.0f, sumSqK = 0.0f;
        for (int i = 0; i < d_k; ++i) {
            float q_val = (float)conv_base[(k_head * d_k + i) * L + t];
            float k_val = (float)conv_base[(key_dim + k_head * d_k + i) * L + t];
            sumSqQ += q_val * q_val;
            sumSqK += k_val * k_val;
        }
        float invNormQ = rsqrt(sumSqQ + eps) * q_scale;
        float invNormK = rsqrt(sumSqK + eps);
        // Pass 2: normalize, scale, and write Q/K
        for (int i = 0; i < d_k; ++i) {
            dst_q[i] = (ftype)((float)conv_base[(k_head * d_k + i) * L + t] * invNormQ);
            dst_k[i] = (ftype)((float)conv_base[(key_dim + k_head * d_k + i) * L + t] * invNormK);
        }
    } else {
        // No L2 norm: single pass read, scale Q, write
        for (int i = 0; i < d_k; ++i) {
            dst_q[i] = (ftype)((float)conv_base[(k_head * d_k + i) * L + t] * q_scale);
            dst_k[i] = (ftype)conv_base[(key_dim + k_head * d_k + i) * L + t];
        }
    }
    // V: direct copy
    for (int i = 0; i < d_v; ++i) {
        dst_v[i] = conv_base[(2 * key_dim + h * d_v + i) * L + t];
    }
}

// Kernel 4: Gated Delta Rule (Step 5 Recurrence)
// Each thread processes one (batch, head, j) across all timesteps
kernel void linear_attn_gated_delta_rule(
    const device ftype* q                [[buffer(0)]],
    const device ftype* k                [[buffer(1)]],
    const device ftype* v                [[buffer(2)]],
    const device ftype* gate             [[buffer(3)]],
    const device ftype* beta             [[buffer(4)]],
    device ftype* recurrent_state        [[buffer(5)]],
    device ftype* attn_out               [[buffer(6)]],
    constant LinearAttnParam& param      [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    const int B = param.batch;
    const int L = param.seq_len;
    const int H = param.num_v_heads;
    const int d_k = param.head_k_dim;
    const int d_v = param.head_v_dim;

    const int total = B * H * d_v;
    if ((int)gid >= total) return;

    const int j = gid % d_v;
    const int b_h = gid / d_v;
    const int h = b_h % H;
    const int b = b_h / H;

    // Transposed state layout: [B, H, d_v, d_k]
    // (matches simdgroup-optimized kernel layout)
    device ftype* state = recurrent_state + (b * H + h) * d_v * d_k + j * d_k;

    // Process each timestep sequentially
    for (int t = 0; t < L; ++t) {
        const device ftype* q_t = q + (b * L * H + t * H + h) * d_k;
        const device ftype* k_t = k + (b * L * H + t * H + h) * d_k;
        float v_t_j = (float)v[(b * L * H + t * H + h) * d_v + j];

        float g_t = (float)gate[token_channel_offset(b, t, h, H, param.gate_c4 & 1, param)];
        if (param.gate_c4 & 2) {
            g_t = linear_attn_gate_fold(g_t, h, param);
        }
        float beta_t = (float)beta[token_channel_offset(b, t, h, H, param.beta_c4 & 1, param)];
        if (param.beta_c4 & 2) {
            beta_t = linear_attn_beta_fold(beta_t);
        }

        float decay_val = exp(g_t);

        // 5.1 & 5.2
        float v_pred_j = 0.0f;
        for (int i = 0; i < d_k; ++i) {
            float s_val = (float)state[i] * decay_val;
            state[i] = (ftype)s_val;
            v_pred_j += s_val * (float)k_t[i];
        }

        // 5.3
        float delta_j = beta_t * (v_t_j - v_pred_j);

        // 5.4 & 5.5
        float o_t_j = 0.0f;
        for (int i = 0; i < d_k; ++i) {
            float s_val = (float)state[i] + (float)k_t[i] * delta_j;
            state[i] = (ftype)s_val;
            o_t_j += s_val * (float)q_t[i];
        }

        attn_out[output_offset(b, t, h, j, param)] = (ftype)o_t_j;
    }
}
)metal";

// Non-fused simdgroup-optimized Gated Delta Rule (for prefill, reads pre-arranged Q/K/V)
// Each simdgroup (32 threads) handles one (batch, head, j) element
// State layout: [B, H, d_v, d_k] for coalesced simd access
static const char* gLinearAttnGatedDeltaRuleSG = R"metal(
#include <metal_stdlib>
using namespace metal;

#if MNN_METAL_FLOAT16_STORAGE
typedef half ftype;
typedef half4 ftype4;
#else
typedef float ftype;
typedef float4 ftype4;
#endif

struct LinearAttnParam {
    int batch;
    int conv_dim;
    int seq_len;
    int kernel_size;
    int conv_state_size;
    int num_k_heads;
    int num_v_heads;
    int head_k_dim;
    int head_v_dim;
    int key_dim;
    int val_dim;
    int gqa_factor;
    int use_l2norm;
    int qkv_c4;
    int gate_c4;
    int beta_c4;
    int output_c4;
    float q_scale;
    int commit_len;    // lazy-commit: pending tokens to replay before this block
    int pending_seq;   // length of the saved pending block
    int lazy_mode;     // 1: do not persist state after the new block (spec verify)
    // gate/beta chain fold (gate_c4 & 2 / beta_c4 & 2): per-head constants
    // -exp(A_log)[h] and dt_bias[h]; capped at 64 heads.
    float gate_coef[64];
    float gate_bias[64];
};

inline int c4_offset(int token, int channel, int token_count) {
    return ((channel >> 2) * token_count + token) * 4 + (channel & 3);
}

inline int token_channel_offset(int b, int t, int c, int channel, int packed,
                                constant LinearAttnParam& param) {
    if (packed) {
        return c4_offset(b * param.seq_len + t, c, param.batch * param.seq_len);
    }
    return (b * param.seq_len + t) * channel + c;
}

// gate/beta chain fold (bit 2 of gate_c4/beta_c4): replicates the unfused
// elementwise chain op-for-op — Binary ops compute in ftype (half on fp16
// builds), Unary ops in fp32 with the MNNEXP +-87 clamp — with an ftype
// store-rounding after every op, so the folded value is bit-identical to the
// separate-dispatch chain output.
inline float linear_attn_gate_fold(float a, int h, constant LinearAttnParam& p) {
    ftype x = (ftype)a + (ftype)p.gate_bias[h];       // ADD dt_bias  (Binary, half)
    x = (ftype)exp(clamp((float)x, -87.0f, 87.0f));   // EXP          (Unary, fp32)
    x = x + (ftype)1.0f;                              // ADD +1       (Binary, half)
    x = (ftype)log((float)x);                         // LOG          (Unary, fp32)
    x = (ftype)p.gate_coef[h] * x;                    // MUL -exp(A_log)
    return (float)x;
}
inline float linear_attn_beta_fold(float b) {
    return (float)(ftype)(1.0f / (1.0f + exp(clamp(-b, -87.0f, 87.0f))));
}

inline int output_offset(int b, int t, int h, int d, constant LinearAttnParam& param) {
    int token = (b * param.seq_len + t) * param.num_v_heads + h;
    if (param.output_c4) {
        return c4_offset(token, d, param.batch * param.seq_len * param.num_v_heads);
    }
    return token * param.head_v_dim + d;
}

// SIMD_ITERS is injected as a compile-time macro from C++ side: (d_k + 31) / 32

kernel void linear_attn_gated_delta_rule_sg(
    const device ftype* q                [[buffer(0)]],
    const device ftype* k                [[buffer(1)]],
    const device ftype* v                [[buffer(2)]],
    const device ftype* gate             [[buffer(3)]],
    const device ftype* beta             [[buffer(4)]],
    device ftype* recurrent_state        [[buffer(5)]],
    device ftype* attn_out               [[buffer(6)]],
    constant LinearAttnParam& param      [[buffer(7)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const int B = param.batch;
    const int L = param.seq_len;
    const int H = param.num_v_heads;
    const int d_k = param.head_k_dim;
    const int d_v = param.head_v_dim;

    int idx = tgpig.x * 4 + sgitg;
    const int total = B * H * d_v;
    if (idx >= total) return;

    const int j = idx % d_v;
    const int b_h = idx / d_v;
    const int h = b_h % H;
    const int b = b_h / H;

    // Transposed state: [B, H, d_v, d_k]
    device ftype* state = recurrent_state + (b * H + h) * d_v * d_k + j * d_k;
    const int n_iters = (d_k + 31) / 32;

    // State lives in registers across the whole L loop; flushed once at exit.
    float st_reg[SIMD_ITERS];
    for (int ii = 0; ii < n_iters; ii++) {
        int i = lane + ii * 32;
        st_reg[ii] = (i < d_k) ? (float)state[i] : 0.0f;
    }

    for (int t = 0; t < L; ++t) {
        const int bth = b * L * H + t * H + h;
        const device ftype* q_t = q + bth * d_k;
        const device ftype* k_t = k + bth * d_k;
        float v_t_j = (float)v[bth * d_v + j];
        float g_t = (float)gate[token_channel_offset(b, t, h, H, param.gate_c4 & 1, param)];
        if (param.gate_c4 & 2) {
            g_t = linear_attn_gate_fold(g_t, h, param);
        }
        float decay_val = exp(g_t);
        float beta_t = (float)beta[token_channel_offset(b, t, h, H, param.beta_c4 & 1, param)];
        if (param.beta_c4 & 2) {
            beta_t = linear_attn_beta_fold(beta_t);
        }

        float k_reg[SIMD_ITERS], q_reg[SIMD_ITERS];
        for (int ii = 0; ii < n_iters; ii++) {
            int i = lane + ii * 32;
            if (i < d_k) {
                k_reg[ii] = (float)k_t[i];
                q_reg[ii] = (float)q_t[i];
            }
        }

        float v_pred_j = 0.0f;
        for (int ii = 0; ii < n_iters; ii++) {
            int i = lane + ii * 32;
            if (i < d_k) {
                st_reg[ii] *= decay_val;
                v_pred_j += st_reg[ii] * k_reg[ii];
            }
        }
        v_pred_j = simd_sum(v_pred_j);
        float delta_j = beta_t * (v_t_j - v_pred_j);

        float o_t_j = 0.0f;
        for (int ii = 0; ii < n_iters; ii++) {
            int i = lane + ii * 32;
            if (i < d_k) {
                st_reg[ii] += k_reg[ii] * delta_j;
                o_t_j += st_reg[ii] * q_reg[ii];
            }
        }
        o_t_j = simd_sum(o_t_j);

        if (lane == 0) {
            attn_out[output_offset(b, t, h, j, param)] = (ftype)o_t_j;
        }
    }

    for (int ii = 0; ii < n_iters; ii++) {
        int i = lane + ii * 32;
        if (i < d_k) state[i] = (ftype)st_reg[ii];
    }
}

// dk==128 specialization: each lane owns 4 consecutive elements as one ftype4
// (vectorized 8-byte loads), state held in a float4 register across L.
kernel void linear_attn_gated_delta_rule_sg_v4(
    const device ftype* q                [[buffer(0)]],
    const device ftype* k                [[buffer(1)]],
    const device ftype* v                [[buffer(2)]],
    const device ftype* gate             [[buffer(3)]],
    const device ftype* beta             [[buffer(4)]],
    device ftype* recurrent_state        [[buffer(5)]],
    device ftype* attn_out               [[buffer(6)]],
    constant LinearAttnParam& param      [[buffer(7)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const int B = param.batch;
    const int L = param.seq_len;
    const int H = param.num_v_heads;
    const int d_k = 128;
    const int d_v = param.head_v_dim;

    int idx = tgpig.x * 4 + sgitg;
    const int total = B * H * d_v;
    if (idx >= total) return;

    const int j = idx % d_v;
    const int b_h = idx / d_v;
    const int h = b_h % H;
    const int b = b_h / H;

    // Transposed state: [B, H, d_v, d_k]
    device ftype4* state4 = (device ftype4*)(recurrent_state + (b * H + h) * d_v * d_k + j * d_k);
    float4 st = float4(state4[lane]);

    const device ftype4* q4 = (const device ftype4*)q;
    const device ftype4* k4 = (const device ftype4*)k;
    const int row4 = d_k / 4;

    for (int t = 0; t < L; ++t) {
        const int bth = b * L * H + t * H + h;
        float4 q_t = float4(q4[bth * row4 + (int)lane]);
        float4 k_t = float4(k4[bth * row4 + (int)lane]);
        float v_t_j = (float)v[bth * d_v + j];
        float g_t = (float)gate[token_channel_offset(b, t, h, H, param.gate_c4 & 1, param)];
        if (param.gate_c4 & 2) {
            g_t = linear_attn_gate_fold(g_t, h, param);
        }
        float decay_val = exp(g_t);
        float beta_t = (float)beta[token_channel_offset(b, t, h, H, param.beta_c4 & 1, param)];
        if (param.beta_c4 & 2) {
            beta_t = linear_attn_beta_fold(beta_t);
        }

        st *= decay_val;
        float v_pred_j = simd_sum(dot(st, k_t));
        float delta_j = beta_t * (v_t_j - v_pred_j);

        st += k_t * delta_j;
        float o_t_j = simd_sum(dot(st, q_t));

        if (lane == 0) {
            attn_out[output_offset(b, t, h, j, param)] = (ftype)o_t_j;
        }
    }

    state4[lane] = ftype4(st);
}

// QKV prep + gated delta rule + pending save, preceded by a prologue that commits the
// previous block's accepted prefix; seq_len = 0 selects the prologue alone.
kernel void linear_attn_verify_fused_sg(
    const device ftype* conv_out          [[buffer(0)]],
    const device ftype* gate              [[buffer(1)]],
    const device ftype* beta              [[buffer(2)]],
    device ftype* recurrent_state         [[buffer(3)]],
    device ftype* attn_out                [[buffer(4)]],
    constant LinearAttnParam& param       [[buffer(5)]],
    device ftype* pending_k               [[buffer(6)]],
    device ftype* pending_v               [[buffer(7)]],
    device ftype* pending_gate            [[buffer(8)]],
    device ftype* pending_beta            [[buffer(9)]],
    const device ftype* prev_pending_k    [[buffer(10)]],
    const device ftype* prev_pending_v    [[buffer(11)]],
    const device ftype* prev_pending_gate [[buffer(12)]],
    const device ftype* prev_pending_beta [[buffer(13)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const int B = param.batch;
    const int L = param.seq_len;
    const int H = param.num_v_heads;
    const int d_k = param.head_k_dim;
    const int d_v = param.head_v_dim;
    const int D = param.conv_dim;
    const int key_dim = param.key_dim;
    const int gqa_factor = param.gqa_factor;
    const int use_l2norm = param.use_l2norm;
    const float q_scale = param.q_scale;

    int idx = tgpig.x * 4 + sgitg;
    const int total = B * H * d_v;
    if (idx >= total) return;
    const int j = idx % d_v;
    const int b_h = idx / d_v;
    const int h = b_h % H;
    const int b = b_h / H;
    const int k_head = h / gqa_factor;

    device ftype* state = recurrent_state + (b * H + h) * d_v * d_k + j * d_k;
    const int n_iters = SIMD_ITERS;   // compile-time bound: a runtime one spills s_reg/k_reg/q_reg to local memory

    const int commit_len = param.commit_len;
    float s_reg[SIMD_ITERS];
    for (int ii = 0; ii < n_iters; ii++) {
        int i = lane + ii * 32;
        s_reg[ii] = (i < d_k) ? (float)state[i] : 0.0f;
    }

    // Replay the accepted prefix of the previous pending block. State stays in fp32
    // registers; drift vs the per-step-rounding decode track only wobbles greedy ties.
    const int pend_seq = param.pending_seq;
    for (int t = 0; t < commit_len; ++t) {
        const int bth = b * pend_seq * H + t * H + h;
        const device ftype* k_t = prev_pending_k + bth * d_k;
        float v_t_j = (float)prev_pending_v[bth * d_v + j];
        float decay_val = exp((float)prev_pending_gate[(b * pend_seq + t) * H + h]);
        float beta_t = (float)prev_pending_beta[(b * pend_seq + t) * H + h];
        float k_reg[SIMD_ITERS];
        for (int ii = 0; ii < n_iters; ii++) {
            int i = lane + ii * 32;
            if (i < d_k) {
                k_reg[ii] = (float)k_t[i];
            }
        }
        float v_pred_j = 0.0f;
        for (int ii = 0; ii < n_iters; ii++) {
            int i = lane + ii * 32;
            if (i < d_k) {
                s_reg[ii] *= decay_val;
                v_pred_j += s_reg[ii] * k_reg[ii];
            }
        }
        v_pred_j = simd_sum(v_pred_j);
        float delta_j = beta_t * (v_t_j - v_pred_j);
        for (int ii = 0; ii < n_iters; ii++) {
            int i = lane + ii * 32;
            if (i < d_k) {
                s_reg[ii] += k_reg[ii] * delta_j;
            }
        }
    }
    if (commit_len > 0) {
        for (int ii = 0; ii < n_iters; ii++) {
            int i = lane + ii * 32;
            if (i < d_k) {
                state[i] = (ftype)s_reg[ii];
            }
        }
    }

    // Main loop with inline QKV prep (qkv_prep_sg no longer separate; the prep
    // is fp32 throughout — the q/k rounding to ftype was dropped because the
    // ulp drift against the replay path does not affect acceptance length).
    const device ftype* conv_base = conv_out + b * D * L;
    const int q_row_base = k_head * d_k;
    const int k_row_base = key_dim + k_head * d_k;
    const int v_channel = 2 * key_dim + h * d_v + j;
    for (int t = 0; t < L; ++t) {
        float k_reg[SIMD_ITERS], q_reg[SIMD_ITERS];
        for (int ii = 0; ii < n_iters; ii++) {
            int i = lane + ii * 32;
            if (i < d_k) {
                q_reg[ii] = (float)conv_base[(q_row_base + i) * L + t];
                k_reg[ii] = (float)conv_base[(k_row_base + i) * L + t];
            } else {
                q_reg[ii] = 0.0f;
                k_reg[ii] = 0.0f;
            }
        }
        float inv_q = q_scale;
        float inv_k = 1.0f;
        if (use_l2norm) {
            const float eps = 1e-6f;
            float sq_q = 0.0f, sq_k = 0.0f;
            for (int ii = 0; ii < n_iters; ii++) {
                sq_q += q_reg[ii] * q_reg[ii];
                sq_k += k_reg[ii] * k_reg[ii];
            }
            sq_q = simd_sum(sq_q);
            sq_k = simd_sum(sq_k);
            inv_q = rsqrt(sq_q + eps) * q_scale;
            inv_k = rsqrt(sq_k + eps);
        }
        // q/k stay in fp32 here; stored to the fp16 pending buffer below, the next
        // block's replay reads a rounded copy, but ulp drift does not affect AL.
        for (int ii = 0; ii < n_iters; ii++) {
            q_reg[ii] *= inv_q;
            k_reg[ii] *= inv_k;
        }
        float v_t_j = (float)conv_base[v_channel * L + t];
        const int gb_off = token_channel_offset(b, t, h, H, param.gate_c4 & 1, param);
        const int bb_off = token_channel_offset(b, t, h, H, param.beta_c4 & 1, param);
        float g_t = (float)gate[gb_off];
        if (param.gate_c4 & 2) { g_t = linear_attn_gate_fold(g_t, h, param); }
        float beta_t = (float)beta[bb_off];
        if (param.beta_c4 & 2) { beta_t = linear_attn_beta_fold(beta_t); }
        float decay_val = exp(g_t);

        // Pending save of the NEW block (write side; host ping-pongs buffers).
        // Folded values are stored, matching the replay prologue above which feeds
        // prev_pending_gate straight into exp() and prev_pending_beta in as-is.
        const int src = b * L * H + t * H + h;
        if (j == 0) {
            for (int ii = 0; ii < n_iters; ii++) {
                int i = lane + ii * 32;
                if (i < d_k) {
                    pending_k[src * d_k + i] = (ftype)k_reg[ii];
                }
            }
            if (lane == 0) {
                pending_gate[(b * L + t) * H + h] = (ftype)g_t;
                pending_beta[(b * L + t) * H + h] = (ftype)beta_t;
            }
        }
        if (lane == 0) {
            pending_v[src * d_v + j] = (ftype)v_t_j;
        }

        float v_pred_j = 0.0f;
        for (int ii = 0; ii < n_iters; ii++) {
            int i = lane + ii * 32;
            if (i < d_k) {
                s_reg[ii] *= decay_val;
                v_pred_j += s_reg[ii] * k_reg[ii];
            }
        }
        v_pred_j = simd_sum(v_pred_j);
        float delta_j = beta_t * (v_t_j - v_pred_j);
        float o_t_j = 0.0f;
        for (int ii = 0; ii < n_iters; ii++) {
            int i = lane + ii * 32;
            if (i < d_k) {
                s_reg[ii] += k_reg[ii] * delta_j;
                o_t_j += s_reg[ii] * q_reg[ii];
            }
        }
        o_t_j = simd_sum(o_t_j);
        if (lane == 0) {
            attn_out[output_offset(b, t, h, j, param)] = (ftype)o_t_j;
        }
    }

    if (!param.lazy_mode) {
        for (int ii = 0; ii < n_iters; ii++) {
            int i = lane + ii * 32;
            if (i < d_k) {
                state[i] = (ftype)s_reg[ii];
            }
        }
    }
}

)metal";


static const char* gLinearAttnFusedSG = R"metal(
#include <metal_stdlib>
using namespace metal;

#if MNN_METAL_FLOAT16_STORAGE
typedef half ftype;
#else
typedef float ftype;
#endif

struct LinearAttnParam {
    int batch;
    int conv_dim;
    int seq_len;
    int kernel_size;
    int conv_state_size;
    int num_k_heads;
    int num_v_heads;
    int head_k_dim;
    int head_v_dim;
    int key_dim;
    int val_dim;
    int gqa_factor;
    int use_l2norm;
    int qkv_c4;
    int gate_c4;
    int beta_c4;
    int output_c4;
    float q_scale;
    int commit_len;    // lazy-commit: pending tokens to replay before this block
    int pending_seq;   // length of the saved pending block
    int lazy_mode;     // 1: do not persist state after the new block (spec verify)
    // gate/beta chain fold (gate_c4 & 2 / beta_c4 & 2): per-head constants
    // -exp(A_log)[h] and dt_bias[h]; capped at 64 heads.
    float gate_coef[64];
    float gate_bias[64];
};

inline int c4_offset(int token, int channel, int token_count) {
    return ((channel >> 2) * token_count + token) * 4 + (channel & 3);
}

inline int token_channel_offset(int b, int t, int c, int channel, int packed,
                                constant LinearAttnParam& param) {
    if (packed) {
        return c4_offset(b * param.seq_len + t, c, param.batch * param.seq_len);
    }
    return (b * param.seq_len + t) * channel + c;
}

// gate/beta chain fold (bit 2 of gate_c4/beta_c4): replicates the unfused
// elementwise chain op-for-op — Binary ops compute in ftype (half on fp16
// builds), Unary ops in fp32 with the MNNEXP +-87 clamp — with an ftype
// store-rounding after every op, so the folded value is bit-identical to the
// separate-dispatch chain output.
inline float linear_attn_gate_fold(float a, int h, constant LinearAttnParam& p) {
    ftype x = (ftype)a + (ftype)p.gate_bias[h];       // ADD dt_bias  (Binary, half)
    x = (ftype)exp(clamp((float)x, -87.0f, 87.0f));   // EXP          (Unary, fp32)
    x = x + (ftype)1.0f;                              // ADD +1       (Binary, half)
    x = (ftype)log((float)x);                         // LOG          (Unary, fp32)
    x = (ftype)p.gate_coef[h] * x;                    // MUL -exp(A_log)
    return (float)x;
}
inline float linear_attn_beta_fold(float b) {
    return (float)(ftype)(1.0f / (1.0f + exp(clamp(-b, -87.0f, 87.0f))));
}

inline int output_offset(int b, int t, int h, int d, constant LinearAttnParam& param) {
    int token = (b * param.seq_len + t) * param.num_v_heads + h;
    if (param.output_c4) {
        return c4_offset(token, d, param.batch * param.seq_len * param.num_v_heads);
    }
    return token * param.head_v_dim + d;
}

// SIMD_ITERS is injected as a compile-time macro from C++ side: (d_k + 31) / 32

kernel void linear_attn_fused_sg(
    const device ftype* conv_out         [[buffer(0)]],
    const device ftype* gate             [[buffer(1)]],
    const device ftype* beta             [[buffer(2)]],
    device ftype* recurrent_state        [[buffer(3)]],
    device ftype* attn_out               [[buffer(4)]],
    constant LinearAttnParam& param      [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const int B = param.batch;
    const int L = param.seq_len;
    const int H = param.num_v_heads;
    const int d_k = param.head_k_dim;
    const int d_v = param.head_v_dim;
    const int key_dim = param.key_dim;
    const int gqa_factor = param.gqa_factor;
    const int use_l2norm = param.use_l2norm;
    const float q_scale = param.q_scale;
    const int D = param.conv_dim;

    // 4 simdgroups per threadgroup
    int idx = tgpig.x * 4 + sgitg;
    const int total = B * H * d_v;
    if (idx >= total) return;

    const int j = idx % d_v;
    const int b_h = idx / d_v;
    const int h = b_h % H;
    const int b = b_h / H;
    const int k_head = h / gqa_factor;

    // Transposed state layout: [B, H, d_v, d_k]
    device ftype* state = recurrent_state + (b * H + h) * d_v * d_k + j * d_k;

    const device ftype* conv_base = conv_out + b * D * L;
    const int n_iters = (d_k + 31) / 32;

    for (int t = 0; t < L; ++t) {
        // Read Q, K directly from conv_out [B, D, L]
        float k_reg[SIMD_ITERS];
        float q_reg[SIMD_ITERS];
        for (int ii = 0; ii < n_iters; ii++) {
            int i = lane + ii * 32;
            if (i < d_k) {
                q_reg[ii] = (float)conv_base[(k_head * d_k + i) * L + t];
                k_reg[ii] = (float)conv_base[(key_dim + k_head * d_k + i) * L + t];
            }
        }

        // Inline L2 norm for Q and K using simd_sum
        if (use_l2norm) {
            const float eps = 1e-6f;
            float sq = 0.0f;
            for (int ii = 0; ii < n_iters; ii++)
                if (lane + ii * 32 < d_k) sq += q_reg[ii] * q_reg[ii];
            sq = simd_sum(sq);
            float inv = rsqrt(sq + eps);
            for (int ii = 0; ii < n_iters; ii++)
                if (lane + ii * 32 < d_k) q_reg[ii] *= inv;

            sq = 0.0f;
            for (int ii = 0; ii < n_iters; ii++)
                if (lane + ii * 32 < d_k) sq += k_reg[ii] * k_reg[ii];
            sq = simd_sum(sq);
            inv = rsqrt(sq + eps);
            for (int ii = 0; ii < n_iters; ii++)
                if (lane + ii * 32 < d_k) k_reg[ii] *= inv;
        }

        // Scale Q
        for (int ii = 0; ii < n_iters; ii++)
            if (lane + ii * 32 < d_k) q_reg[ii] *= q_scale;

        // V: channel [2*key_dim + h*d_v + j], position t
        float v_t_j = (float)conv_base[(2 * key_dim + h * d_v + j) * L + t];

        const int bth = b * L * H + t * H + h;
        float g_t = (float)gate[token_channel_offset(b, t, h, H, param.gate_c4 & 1, param)];
        if (param.gate_c4 & 2) {
            g_t = linear_attn_gate_fold(g_t, h, param);
        }
        float decay_val = exp(g_t);
        float beta_t = (float)beta[token_channel_offset(b, t, h, H, param.beta_c4 & 1, param)];
        if (param.beta_c4 & 2) {
            beta_t = linear_attn_beta_fold(beta_t);
        }

        // Step 1: Decay state + compute v_pred
        float v_pred_j = 0.0f;
        for (int ii = 0; ii < n_iters; ii++) {
            int i = lane + ii * 32;
            if (i < d_k) {
                float s_val = (float)state[i] * decay_val;
                state[i] = (ftype)s_val;
                v_pred_j += s_val * k_reg[ii];
            }
        }
        v_pred_j = simd_sum(v_pred_j);

        // Step 2: Compute delta
        float delta_j = beta_t * (v_t_j - v_pred_j);

        // Step 3: Update state + compute output
        float o_t_j = 0.0f;
        for (int ii = 0; ii < n_iters; ii++) {
            int i = lane + ii * 32;
            if (i < d_k) {
                float s_val = (float)state[i] + k_reg[ii] * delta_j;
                state[i] = (ftype)s_val;
                o_t_j += s_val * q_reg[ii];
            }
        }
        o_t_j = simd_sum(o_t_j);

        if (lane == 0) {
            attn_out[output_offset(b, t, h, j, param)] = (ftype)o_t_j;
        }
    }
}
)metal";

static const char* gLinearAttnShortConv = R"metal(
#include <metal_stdlib>
using namespace metal;

#if MNN_METAL_FLOAT16_STORAGE
typedef half ftype;
#else
typedef float ftype;
#endif

struct LinearAttnParam {
    int batch;
    int conv_dim;
    int seq_len;
    int kernel_size;
    int conv_state_size;
    int num_k_heads;
    int num_v_heads;
    int head_k_dim;
    int head_v_dim;
    int key_dim;
    int val_dim;
    int gqa_factor;
    int use_l2norm;
    int qkv_c4;
    int gate_c4;
    int beta_c4;
    int output_c4;
    float q_scale;
    int commit_len;    // lazy-commit: pending tokens to replay before this block
    int pending_seq;   // length of the saved pending block
    int lazy_mode;     // 1: do not persist state after the new block (spec verify)
    // gate/beta chain fold (gate_c4 & 2 / beta_c4 & 2): per-head constants
    // -exp(A_log)[h] and dt_bias[h]; capped at 64 heads.
    float gate_coef[64];
    float gate_bias[64];
};

inline int c4_offset(int token, int channel, int token_count) {
    return ((channel >> 2) * token_count + token) * 4 + (channel & 3);
}

inline int qkv_offset(int b, int d, int l, constant LinearAttnParam& param) {
    if (param.qkv_c4) {
        return c4_offset(b * param.seq_len + l, d, param.batch * param.seq_len);
    }
    return (b * param.conv_dim + d) * param.seq_len + l;
}

inline float short_conv_input(const device ftype* qkv, int b, int h, int l,
                              constant LinearAttnParam& param) {
    int hidden = param.head_v_dim;
    float b_value = (float)qkv[qkv_offset(b, h, l, param)];
    float x_value = (float)qkv[qkv_offset(b, 2 * hidden + h, l, param)];
    return b_value * x_value;
}

inline int output_offset(int b, int l, int h, constant LinearAttnParam& param) {
    int token = b * param.seq_len + l;
    if (param.output_c4) {
        return c4_offset(token, h, param.batch * param.seq_len);
    }
    return token * param.head_v_dim + h;
}

kernel void linear_attn_short_conv_nosilu(
    const device ftype* qkv         [[buffer(0)]],
    device ftype* conv_state        [[buffer(1)]],
    const device ftype* conv_weight [[buffer(2)]],
    device ftype* conv_out          [[buffer(3)]],
    constant LinearAttnParam& param [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    const int B = param.batch;
    const int L = param.seq_len;
    const int K = param.kernel_size;
    const int css = param.conv_state_size;
    const int H = param.head_v_dim;
    const int total = B * H * L;
    if ((int)gid >= total) return;

    const int l = gid % L;
    const int bh = gid / L;
    const int b = bh / H;
    const int h = bh % H;
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        int pos = l + k;
        float value = pos < css ? (float)conv_state[(b * H + h) * css + pos]
                                : short_conv_input(qkv, b, h, pos - css, param);
        sum += value * (float)conv_weight[h * K + k];
    }
    conv_out[(b * H + h) * L + l] = (ftype)sum;
}

kernel void linear_attn_short_conv_state_update(
    const device ftype* qkv         [[buffer(0)]],
    device ftype* conv_state        [[buffer(1)]],
    constant LinearAttnParam& param [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    const int B = param.batch;
    const int L = param.seq_len;
    const int css = param.conv_state_size;
    const int H = param.head_v_dim;
    const int total = B * H * css;
    if ((int)gid >= total) return;

    const int i = gid % css;
    const int bh = gid / css;
    const int b = bh / H;
    const int h = bh % H;
    int pos = L + i;
    ftype value = pos < css ? conv_state[(b * H + h) * css + pos]
                            : (ftype)short_conv_input(qkv, b, h, pos - css, param);
    conv_state[(b * H + h) * css + i] = value;
}

kernel void linear_attn_short_conv_output(
    const device ftype* qkv         [[buffer(0)]],
    const device ftype* conv_out    [[buffer(1)]],
    device ftype* attn_out          [[buffer(2)]],
    constant LinearAttnParam& param [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    const int B = param.batch;
    const int L = param.seq_len;
    const int H = param.head_v_dim;
    const int total = B * H * L;
    if ((int)gid >= total) return;

    const int l = gid % L;
    const int bh = gid / L;
    const int b = bh / H;
    const int h = bh % H;
    float c_value = (float)qkv[qkv_offset(b, H + h, l, param)];
    float conv_value = (float)conv_out[(b * H + h) * L + l];
    attn_out[output_offset(b, l, h, param)] = (ftype)(c_value * conv_value);
}
)metal";


static const char* gLinearAttnFusedSGAlign = R"metal(
#include <metal_stdlib>
using namespace metal;

#if MNN_METAL_FLOAT16_STORAGE
typedef half ftype;
#else
typedef float ftype;
#endif

struct LinearAttnParam {
    int batch;
    int conv_dim;
    int seq_len;
    int kernel_size;
    int conv_state_size;
    int num_k_heads;
    int num_v_heads;
    int head_k_dim;
    int head_v_dim;
    int key_dim;
    int val_dim;
    int gqa_factor;
    int use_l2norm;
    int qkv_c4;
    int gate_c4;
    int beta_c4;
    int output_c4;
    float q_scale;
    int commit_len;    // lazy-commit: pending tokens to replay before this block
    int pending_seq;   // length of the saved pending block
    int lazy_mode;     // 1: do not persist state after the new block (spec verify)
    // gate/beta chain fold (gate_c4 & 2 / beta_c4 & 2): per-head constants
    // -exp(A_log)[h] and dt_bias[h]; capped at 64 heads.
    float gate_coef[64];
    float gate_bias[64];
};

inline int c4_offset_v2(int token, int channel, int token_count) {
    return ((channel >> 2) * token_count + token) * 4 + (channel & 3);
}

inline int token_channel_offset(int b, int t, int c, int channel, int packed,
                                constant LinearAttnParam& param) {
    if (packed) {
        return c4_offset_v2(b * param.seq_len + t, c, param.batch * param.seq_len);
    }
    return (b * param.seq_len + t) * channel + c;
}

// gate/beta chain fold (bit 2 of gate_c4/beta_c4): replicates the unfused
// elementwise chain op-for-op — Binary ops compute in ftype (half on fp16
// builds), Unary ops in fp32 with the MNNEXP +-87 clamp — with an ftype
// store-rounding after every op, so the folded value is bit-identical to the
// separate-dispatch chain output.
inline float linear_attn_gate_fold(float a, int h, constant LinearAttnParam& p) {
    ftype x = (ftype)a + (ftype)p.gate_bias[h];       // ADD dt_bias  (Binary, half)
    x = (ftype)exp(clamp((float)x, -87.0f, 87.0f));   // EXP          (Unary, fp32)
    x = x + (ftype)1.0f;                              // ADD +1       (Binary, half)
    x = (ftype)log((float)x);                         // LOG          (Unary, fp32)
    x = (ftype)p.gate_coef[h] * x;                    // MUL -exp(A_log)
    return (float)x;
}
inline float linear_attn_beta_fold(float b) {
    return (float)(ftype)(1.0f / (1.0f + exp(clamp(-b, -87.0f, 87.0f))));
}

inline int output_offset_v2(int b, int t, int h, int d, constant LinearAttnParam& param) {
    int token = (b * param.seq_len + t) * param.num_v_heads + h;
    if (param.output_c4) {
        return c4_offset_v2(token, d, param.batch * param.seq_len * param.num_v_heads);
    }
    return token * param.head_v_dim + d;
}

// SIMD_ITERS is injected as a compile-time macro from C++ side: (d_k + 31) / 32
// D_K_ALIGNED (0 or 1) is injected from C++ side; when 1, d_k % 32 == 0 and
// per-lane boundary checks against d_k can be dropped.

#ifndef ALIGN_SIMDS_PER_TG
#define ALIGN_SIMDS_PER_TG 4
#endif

// [Opt7] IS_LANE_VALID — 1 when this SIMD_ITERS lane is in-range.
// For the last iteration under D_K_ALIGNED=0 we still need the runtime guard.
#if D_K_ALIGNED
    #define IS_LANE_VALID(ii, lane, d_k) (true)
#else
    #define IS_LANE_VALID(ii, lane, d_k) (((int)(lane) + (ii) * 32) < (d_k))
#endif

kernel void linear_attn_fused_sg_align(
    const device ftype* conv_out         [[buffer(0)]],
    const device ftype* gate             [[buffer(1)]],
    const device ftype* beta             [[buffer(2)]],
    device ftype* recurrent_state        [[buffer(3)]],
    device ftype* attn_out               [[buffer(4)]],
    constant LinearAttnParam& param      [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const int B = param.batch;
    const int L = param.seq_len;
    const int H = param.num_v_heads;
    const int d_k = param.head_k_dim;
    const int d_v = param.head_v_dim;
    const int key_dim = param.key_dim;
    const int gqa_factor = param.gqa_factor;
    const int use_l2norm = param.use_l2norm;
    const float q_scale = param.q_scale;
    const int D = param.conv_dim;

    // Each simdgroup independently handles one (b, h, j).
    const int idx = (int)tgpig.x * ALIGN_SIMDS_PER_TG + (int)sgitg;
    const int total = B * H * d_v;
    if (idx >= total) return;

    const int j         = idx % d_v;
    const int b_h       = idx / d_v;
    const int h         = b_h % H;
    const int b         = b_h / H;
    const int k_head    = h / gqa_factor;

    // Transposed state: [B, H, d_v, d_k]
    device ftype* state = recurrent_state + (b * H + h) * d_v * d_k + j * d_k;

    // [Opt9] Hoist per-(b,h,j) addressing out of the t-loop.
    const device ftype* conv_base = conv_out + b * D * L;
    const int q_row_base   = k_head * d_k;                   // Q  channel base
    const int k_row_base   = key_dim + k_head * d_k;         // K  channel base
    const int v_channel    = 2 * key_dim + h * d_v + j;      // V  channel
    const int bth_base     = b * L * H + h;                  // (b, ., h) part
    const int lane_i32     = (int)lane;
    const int n_iters      = SIMD_ITERS;                     // compile-time
    const int lane_stride  = 32;                             // simd width

    // Precomputed per-iteration channel indices (compile-time expandable).
    int lane_offs[SIMD_ITERS];
    for (int ii = 0; ii < n_iters; ++ii) {
        lane_offs[ii] = lane_i32 + ii * lane_stride;
    }

    // ─── [Opt4][Opt9] Load state once into registers ────────────────────
    float st_reg[SIMD_ITERS];
    for (int ii = 0; ii < n_iters; ++ii) {
#if D_K_ALIGNED
        st_reg[ii] = (float)state[lane_offs[ii]];
#else
        st_reg[ii] = (lane_offs[ii] < d_k) ? (float)state[lane_offs[ii]] : 0.0f;
#endif
    }

    for (int t = 0; t < L; ++t) {
        const int bth = bth_base + t * H;

        // ─── Read Q, K directly from conv_out [B, D, L] ─────────────────
        float k_reg[SIMD_ITERS];
        float q_reg[SIMD_ITERS];
        for (int ii = 0; ii < n_iters; ++ii) {
#if D_K_ALIGNED
            const int i = lane_offs[ii];
            q_reg[ii] = (float)conv_base[(q_row_base + i) * L + t];
            k_reg[ii] = (float)conv_base[(k_row_base + i) * L + t];
#else
            const int i = lane_offs[ii];
            if (i < d_k) {
                q_reg[ii] = (float)conv_base[(q_row_base + i) * L + t];
                k_reg[ii] = (float)conv_base[(k_row_base + i) * L + t];
            } else {
                q_reg[ii] = 0.0f;
                k_reg[ii] = 0.0f;
            }
#endif
        }

        // ─── Inline L2 norm for Q and K using simd_sum ──────────────────
        // [Opt6] q_scale is folded into inv_q, saving a second scale pass.
        float inv_q = q_scale;
        float inv_k = 1.0f;
        if (use_l2norm) {
            const float eps = 1e-6f;
            float sq_q = 0.0f, sq_k = 0.0f;
            for (int ii = 0; ii < n_iters; ++ii) {
                if (IS_LANE_VALID(ii, lane, d_k)) {
                    sq_q += q_reg[ii] * q_reg[ii];
                    sq_k += k_reg[ii] * k_reg[ii];
                }
            }
            sq_q = simd_sum(sq_q);
            sq_k = simd_sum(sq_k);
            inv_q = rsqrt(sq_q + eps) * q_scale;   // fold q_scale here
            inv_k = rsqrt(sq_k + eps);
        }
        for (int ii = 0; ii < n_iters; ++ii) {
            q_reg[ii] *= inv_q;
            k_reg[ii] *= inv_k;
        }

        // ─── Lane-0 device reads + simdgroup broadcast (Opt5/Opt8) ──────
        float v_t_j     = 0.0f;
        float decay_val = 0.0f;
        float beta_t    = 0.0f;
        if (lane == 0) {
            v_t_j     = (float)conv_base[v_channel * L + t];
            float g_t = (float)gate[token_channel_offset(b, t, h, H, param.gate_c4 & 1, param)];
            if (param.gate_c4 & 2) {
                g_t = linear_attn_gate_fold(g_t, h, param);
            }
            decay_val = exp(g_t);
            beta_t    = (float)beta[token_channel_offset(b, t, h, H, param.beta_c4 & 1, param)];
            if (param.beta_c4 & 2) {
                beta_t = linear_attn_beta_fold(beta_t);
            }
        }
        v_t_j     = simd_broadcast_first(v_t_j);
        decay_val = simd_broadcast_first(decay_val);
        beta_t    = simd_broadcast_first(beta_t);

        // ─── [Opt1][Opt4] Decay state in-register + compute v_pred ──────
        float v_pred_j = 0.0f;
        for (int ii = 0; ii < n_iters; ++ii) {
            if (IS_LANE_VALID(ii, lane, d_k)) {
                st_reg[ii] *= decay_val;
                v_pred_j   += st_reg[ii] * k_reg[ii];
            }
        }
        v_pred_j = simd_sum(v_pred_j);

        const float delta_j = beta_t * (v_t_j - v_pred_j);

        // ─── [Opt1][Opt4] Update state in-register + compute output ─────
        float o_t_j = 0.0f;
        for (int ii = 0; ii < n_iters; ++ii) {
            if (IS_LANE_VALID(ii, lane, d_k)) {
                st_reg[ii] += k_reg[ii] * delta_j;
                o_t_j      += st_reg[ii] * q_reg[ii];
            }
        }
        o_t_j = simd_sum(o_t_j);

        if (lane == 0) {
            attn_out[output_offset_v2(b, t, h, j, param)] = (ftype)o_t_j;
        }
    }

    // ─── [Opt4] Write state back to device once ─────────────────────────
    for (int ii = 0; ii < n_iters; ++ii) {
#if D_K_ALIGNED
        state[lane_offs[ii]] = (ftype)st_reg[ii];
#else
        if (lane_offs[ii] < d_k) state[lane_offs[ii]] = (ftype)st_reg[ii];
#endif
    }
}
)metal";


static const char* gLinearAttnFusedSGTG = R"metal(
#include <metal_stdlib>
using namespace metal;

#if MNN_METAL_FLOAT16_STORAGE
typedef half ftype;
#else
typedef float ftype;
#endif

struct LinearAttnParam {
    int batch;
    int conv_dim;
    int seq_len;
    int kernel_size;
    int conv_state_size;
    int num_k_heads;
    int num_v_heads;
    int head_k_dim;
    int head_v_dim;
    int key_dim;
    int val_dim;
    int gqa_factor;
    int use_l2norm;
    int qkv_c4;
    int gate_c4;
    int beta_c4;
    int output_c4;
    float q_scale;
    int commit_len;    // lazy-commit: pending tokens to replay before this block
    int pending_seq;   // length of the saved pending block
    int lazy_mode;     // 1: do not persist state after the new block (spec verify)
    // gate/beta chain fold (gate_c4 & 2 / beta_c4 & 2): per-head constants
    // -exp(A_log)[h] and dt_bias[h]; capped at 64 heads.
    float gate_coef[64];
    float gate_bias[64];
};

inline int c4_offset_v2(int token, int channel, int token_count) {
    return ((channel >> 2) * token_count + token) * 4 + (channel & 3);
}

inline int token_channel_offset(int b, int t, int c, int channel, int packed,
                                constant LinearAttnParam& param) {
    if (packed) {
        return c4_offset_v2(b * param.seq_len + t, c, param.batch * param.seq_len);
    }
    return (b * param.seq_len + t) * channel + c;
}

// gate/beta chain fold (bit 2 of gate_c4/beta_c4): replicates the unfused
// elementwise chain op-for-op — Binary ops compute in ftype (half on fp16
// builds), Unary ops in fp32 with the MNNEXP +-87 clamp — with an ftype
// store-rounding after every op, so the folded value is bit-identical to the
// separate-dispatch chain output.
inline float linear_attn_gate_fold(float a, int h, constant LinearAttnParam& p) {
    ftype x = (ftype)a + (ftype)p.gate_bias[h];       // ADD dt_bias  (Binary, half)
    x = (ftype)exp(clamp((float)x, -87.0f, 87.0f));   // EXP          (Unary, fp32)
    x = x + (ftype)1.0f;                              // ADD +1       (Binary, half)
    x = (ftype)log((float)x);                         // LOG          (Unary, fp32)
    x = (ftype)p.gate_coef[h] * x;                    // MUL -exp(A_log)
    return (float)x;
}
inline float linear_attn_beta_fold(float b) {
    return (float)(ftype)(1.0f / (1.0f + exp(clamp(-b, -87.0f, 87.0f))));
}

inline int output_offset_v2(int b, int t, int h, int d, constant LinearAttnParam& param) {
    int token = (b * param.seq_len + t) * param.num_v_heads + h;
    if (param.output_c4) {
        return c4_offset_v2(token, d, param.batch * param.seq_len * param.num_v_heads);
    }
    return token * param.head_v_dim + d;
}

#if D_K_ALIGNED
    #define IS_LANE_VALID_TG(ii, lane, d_k) (true)
#else
    #define IS_LANE_VALID_TG(ii, lane, d_k) (((int)(lane) + (ii) * 32) < (d_k))
#endif

kernel void linear_attn_fused_sg_tg(
    const device ftype* conv_out         [[buffer(0)]],
    const device ftype* gate             [[buffer(1)]],
    const device ftype* beta             [[buffer(2)]],
    device ftype* recurrent_state        [[buffer(3)]],
    device ftype* attn_out               [[buffer(4)]],
    constant LinearAttnParam& param      [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const int B = param.batch;
    const int L = param.seq_len;
    const int H = param.num_v_heads;
    const int d_k = param.head_k_dim;
    const int d_v = param.head_v_dim;
    const int key_dim = param.key_dim;
    const int gqa_factor = param.gqa_factor;
    const int use_l2norm = param.use_l2norm;
    const float q_scale = param.q_scale;
    const int D = param.conv_dim;

    // Host guarantees d_v % SIMDS_PER_TG == 0, so a TG's SIMDS_PER_TG SGs
    // all fall within the same (b, h) — same k_head, same Q/K row.
    const int idx = (int)tgpig.x * SIMDS_PER_TG + (int)sgitg;
    const int total = B * H * d_v;
    if (idx >= total) return;

    const int j       = idx % d_v;
    const int b_h     = idx / d_v;
    const int h       = b_h % H;
    const int b       = b_h / H;
    const int k_head  = h / gqa_factor;

    // Transposed state: [B, H, d_v, d_k]
    device ftype* state = recurrent_state + (b * H + h) * d_v * d_k + j * d_k;

    // Hoist per-(b, h, j) addressing.
    const device ftype* conv_base = conv_out + b * D * L;
    const int q_row_base = k_head * d_k;
    const int k_row_base = key_dim + k_head * d_k;
    const int v_channel  = 2 * key_dim + h * d_v + j;
    const int bth_base   = b * L * H + h;

    const int TG_THREADS = SIMDS_PER_TG * 32;
    const int tid        = (int)(sgitg * 32 + lane);

    // Shared Q/K row for this (b, h, t) — reused by all SIMDS_PER_TG SGs.
    threadgroup float sh_q[HEAD_K_DIM];
    threadgroup float sh_k[HEAD_K_DIM];
    // Shared per-t scalars (decay_val, beta_t). v_t_j is per-j, kept per-SG.
    threadgroup float sh_scalars[2];

    // ─── Load state once into registers ────────────────────────────────
    float st_reg[SIMD_ITERS];
    for (int ii = 0; ii < SIMD_ITERS; ++ii) {
        int i = (int)lane + ii * 32;
#if D_K_ALIGNED
        st_reg[ii] = (float)state[i];
#else
        st_reg[ii] = (i < d_k) ? (float)state[i] : 0.0f;
#endif
    }

    int bth = bth_base;
    for (int t = 0; t < L; ++t, bth += H) {
        // ─── Cooperative load Q, K into shared memory (128 threads). ───
        for (int i = tid; i < d_k; i += TG_THREADS) {
            sh_q[i] = (float)conv_base[(q_row_base + i) * L + t];
            sh_k[i] = (float)conv_base[(k_row_base + i) * L + t];
        }
        // Scalars (once per TG, per timestep).
        if (tid == 0) {
            float gate_value = (float)gate[token_channel_offset(b, t, h, H, param.gate_c4 & 1, param)];
            if (param.gate_c4 & 2) {
                gate_value = linear_attn_gate_fold(gate_value, h, param);
            }
            sh_scalars[0] = exp(gate_value);
            float beta_raw = (float)beta[token_channel_offset(b, t, h, H, param.beta_c4 & 1, param)];
            sh_scalars[1] = (param.beta_c4 & 2) ? linear_attn_beta_fold(beta_raw) : beta_raw;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ─── L2 norm + q_scale done ONCE by SG 0 on shared Q/K. ────────
        // Other SGs wait on the trailing barrier. This trades some SG
        // idle for eliminating the per-SG redundant reduction of the
        // baseline kernel; net win because HEAD_K_DIM reduction is small.
        if (sgitg == 0) {
            if (use_l2norm) {
                const float eps = 1e-6f;
                float sq_q = 0.0f, sq_k = 0.0f;
                for (int i = (int)lane; i < d_k; i += 32) {
                    float qv = sh_q[i]; float kv = sh_k[i];
                    sq_q += qv * qv; sq_k += kv * kv;
                }
                sq_q = simd_sum(sq_q);
                sq_k = simd_sum(sq_k);
                float inv_q = rsqrt(sq_q + eps) * q_scale;
                float inv_k = rsqrt(sq_k + eps);
                for (int i = (int)lane; i < d_k; i += 32) {
                    sh_q[i] *= inv_q;
                    sh_k[i] *= inv_k;
                }
            } else {
                // Fold q_scale directly onto sh_q; K unchanged.
                for (int i = (int)lane; i < d_k; i += 32) {
                    sh_q[i] *= q_scale;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ─── Load normalized Q, K into per-lane registers. ─────────────
        float q_reg[SIMD_ITERS];
        float k_reg[SIMD_ITERS];
        for (int ii = 0; ii < SIMD_ITERS; ++ii) {
            int i = (int)lane + ii * 32;
#if D_K_ALIGNED
            q_reg[ii] = sh_q[i];
            k_reg[ii] = sh_k[i];
#else
            if (i < d_k) {
                q_reg[ii] = sh_q[i];
                k_reg[ii] = sh_k[i];
            } else {
                q_reg[ii] = 0.0f;
                k_reg[ii] = 0.0f;
            }
#endif
        }

        // ─── Scalars: decay/beta from shared, v_t_j per-SG (lane 0 → bcast).
        const float decay_val = sh_scalars[0];
        const float beta_t    = sh_scalars[1];
        float v_t_j = 0.0f;
        if (lane == 0) {
            v_t_j = (float)conv_base[v_channel * L + t];
        }
        v_t_j = simd_broadcast_first(v_t_j);

        // ─── Decay state in-register + compute v_pred. ─────────────────
        float v_pred_j = 0.0f;
        for (int ii = 0; ii < SIMD_ITERS; ++ii) {
            if (IS_LANE_VALID_TG(ii, lane, d_k)) {
                st_reg[ii] *= decay_val;
                v_pred_j   += st_reg[ii] * k_reg[ii];
            }
        }
        v_pred_j = simd_sum(v_pred_j);

        const float delta_j = beta_t * (v_t_j - v_pred_j);

        // ─── Update state in-register + compute output. ────────────────
        float o_t_j = 0.0f;
        for (int ii = 0; ii < SIMD_ITERS; ++ii) {
            if (IS_LANE_VALID_TG(ii, lane, d_k)) {
                st_reg[ii] += k_reg[ii] * delta_j;
                o_t_j      += st_reg[ii] * q_reg[ii];
            }
        }
        o_t_j = simd_sum(o_t_j);

        if (lane == 0) {
            attn_out[output_offset_v2(b, t, h, j, param)] = (ftype)o_t_j;
        }

        // Keep all SGs in lockstep through the shared-Q/K phase before the
        // compiler schedules state writeback. Although decode has no next
        // timestep, removing this barrier regresses the kernel on Apple GPUs.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ─── Write state back to device once. ──────────────────────────────
    for (int ii = 0; ii < SIMD_ITERS; ++ii) {
        int i = (int)lane + ii * 32;
#if D_K_ALIGNED
        state[i] = (ftype)st_reg[ii];
#else
        if (i < d_k) state[i] = (ftype)st_reg[ii];
#endif
    }
}
)metal";

static const char* gLinearAttnFusedChunkSG = R"metal(
#include <metal_stdlib>
using namespace metal;

#if MNN_METAL_FLOAT16_STORAGE
typedef half ftype;
#else
typedef float ftype;
#endif

struct LinearAttnParam {
    int batch;
    int conv_dim;
    int seq_len;
    int kernel_size;
    int conv_state_size;
    int num_k_heads;
    int num_v_heads;
    int head_k_dim;
    int head_v_dim;
    int key_dim;
    int val_dim;
    int gqa_factor;
    int use_l2norm;
    int qkv_c4;
    int gate_c4;
    int beta_c4;
    int output_c4;
    float q_scale;
    int commit_len;    // lazy-commit: pending tokens to replay before this block
    int pending_seq;   // length of the saved pending block
    int lazy_mode;     // 1: do not persist state after the new block (spec verify)
    // gate/beta chain fold (gate_c4 & 2 / beta_c4 & 2): per-head constants
    // -exp(A_log)[h] and dt_bias[h]; capped at 64 heads.
    float gate_coef[64];
    float gate_bias[64];
};

inline int c4_offset_v2(int token, int channel, int token_count) {
    return ((channel >> 2) * token_count + token) * 4 + (channel & 3);
}

inline int token_channel_offset(int b, int t, int c, int channel, int packed,
                                constant LinearAttnParam& param) {
    if (packed) {
        return c4_offset_v2(b * param.seq_len + t, c, param.batch * param.seq_len);
    }
    return (b * param.seq_len + t) * channel + c;
}

// gate/beta chain fold (bit 2 of gate_c4/beta_c4): replicates the unfused
// elementwise chain op-for-op — Binary ops compute in ftype (half on fp16
// builds), Unary ops in fp32 with the MNNEXP +-87 clamp — with an ftype
// store-rounding after every op, so the folded value is bit-identical to the
// separate-dispatch chain output.
inline float linear_attn_gate_fold(float a, int h, constant LinearAttnParam& p) {
    ftype x = (ftype)a + (ftype)p.gate_bias[h];       // ADD dt_bias  (Binary, half)
    x = (ftype)exp(clamp((float)x, -87.0f, 87.0f));   // EXP          (Unary, fp32)
    x = x + (ftype)1.0f;                              // ADD +1       (Binary, half)
    x = (ftype)log((float)x);                         // LOG          (Unary, fp32)
    x = (ftype)p.gate_coef[h] * x;                    // MUL -exp(A_log)
    return (float)x;
}
inline float linear_attn_beta_fold(float b) {
    return (float)(ftype)(1.0f / (1.0f + exp(clamp(-b, -87.0f, 87.0f))));
}

inline int output_offset_v2(int b, int t, int h, int d, constant LinearAttnParam& param) {
    int token = (b * param.seq_len + t) * param.num_v_heads + h;
    if (param.output_c4) {
        return c4_offset_v2(token, d, param.batch * param.seq_len * param.num_v_heads);
    }
    return token * param.head_v_dim + d;
}

kernel void linear_attn_fused_chunk_sg(
    const device ftype* conv_out         [[buffer(0)]],
    const device ftype* gate             [[buffer(1)]],
    const device ftype* beta             [[buffer(2)]],
    device ftype* recurrent_state        [[buffer(3)]],
    device ftype* attn_out               [[buffer(4)]],
    constant LinearAttnParam& param      [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const int L      = param.seq_len;
    const int H      = param.num_v_heads;
    const int d_k    = param.head_k_dim;
    const int d_v    = param.head_v_dim;
    const int D      = param.conv_dim;
    const int key_dim= param.key_dim;
    const int gqa_factor = param.gqa_factor;
    const int use_l2norm = param.use_l2norm;
    const float q_scale  = param.q_scale;

    // Decode grid position into (b, h, j-block).
    // TGs = B * H * (dv / SIMDS_PER_TG); j-block index within a head = tgpig
    // mod (dv/SIMDS_PER_TG). Each SG then owns j = j_block_base + sgitg.
    const int j_blocks = d_v / SIMDS_PER_TG;              // whole j-blocks per head
    const int idx      = (int)tgpig;
    const int bh       = idx / j_blocks;
    const int j_block  = idx % j_blocks;
    const int b        = bh / H;
    const int h        = bh % H;
    const int k_head   = h / gqa_factor;
    const int j        = j_block * SIMDS_PER_TG + sgitg;  // this SG's dv column

    // ─── State: dk elements per j, hoisted into lane-local registers. ───
    device ftype* state_gm = recurrent_state + (b * H + h) * d_v * d_k + j * d_k;
    float s_reg[SIMD_ITERS];
    for (int ii = 0; ii < SIMD_ITERS; ++ii) {
        int i = lane + ii * 32;
        s_reg[ii] = (i < d_k) ? (float)state_gm[i] : 0.0f;
    }

    // ─── Threadgroup-shared buffers for one chunk. ───
    // Q, K are shared across all SIMDS_PER_TG simdgroups (same for every j in
    // the block). V is per-j but only the SIMDS_PER_TG j-columns owned here.
    threadgroup float sh_q[CHUNK_BT][HEAD_K_DIM];             // scaled+normalized Q
    threadgroup float sh_k[CHUNK_BT][HEAD_K_DIM];             // normalized K
    threadgroup float sh_v[CHUNK_BT][SIMDS_PER_TG];           // per-j V for this TG
    threadgroup float sh_g[CHUNK_BT];
    threadgroup float sh_beta[CHUNK_BT];

    const int TG_THREADS = SIMDS_PER_TG * 32;
    const int tid = (int)(sgitg * 32 + lane);

    const device ftype* conv_base = conv_out + b * D * L;

    // Main chunked timestep loop.
    for (int t0 = 0; t0 < L; t0 += CHUNK_BT) {
        int chunk_len = min(CHUNK_BT, L - t0);

        // ── Cooperative load Q, K, V, gate, beta for this chunk. ──
        int qk_total = chunk_len * d_k;
        for (int idx2 = tid; idx2 < qk_total; idx2 += TG_THREADS) {
            int dt = idx2 / d_k;
            int di = idx2 % d_k;
            int t_abs = t0 + dt;
            sh_q[dt][di] = (float)conv_base[(k_head * d_k + di) * L + t_abs];
            sh_k[dt][di] = (float)conv_base[(key_dim + k_head * d_k + di) * L + t_abs];
        }
        // V: only SIMDS_PER_TG columns (the ones we output), keyed by (dt, sgitg).
        // Each SG loads its own j-column's V across all chunk_len timesteps.
        for (int dt = (int)lane; dt < chunk_len; dt += 32) {
            int t_abs = t0 + dt;
            sh_v[dt][sgitg] = (float)conv_base[(2 * key_dim + h * d_v + j) * L + t_abs];
        }
        for (int dt = tid; dt < chunk_len; dt += TG_THREADS) {
            int bth = b * L * H + (t0 + dt) * H + h;
            int t_abs = t0 + dt;
            float g_raw = (float)gate[token_channel_offset(b, t_abs, h, H, param.gate_c4 & 1, param)];
            if (param.gate_c4 & 2) {
                g_raw = linear_attn_gate_fold(g_raw, h, param);
            }
            sh_g[dt] = g_raw;
            float b_raw = (float)beta[token_channel_offset(b, t_abs, h, H, param.beta_c4 & 1, param)];
            sh_beta[dt] = (param.beta_c4 & 2) ? linear_attn_beta_fold(b_raw) : b_raw;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── L2 norm + q_scale on sh_q / sh_k. Each SG handles a slice of
        //     timesteps (round-robin over sgitg) so the whole TG cooperates. ──
        if (use_l2norm) {
            const float eps = 1e-6f;
            for (int dt = sgitg; dt < chunk_len; dt += SIMDS_PER_TG) {
                // Q norm
                float sq = 0.0f;
                for (int i = lane; i < d_k; i += 32) {
                    float v = sh_q[dt][i]; sq += v * v;
                }
                sq = simd_sum(sq);
                float invQ = rsqrt(sq + eps) * q_scale;
                for (int i = lane; i < d_k; i += 32) sh_q[dt][i] *= invQ;
                // K norm
                sq = 0.0f;
                for (int i = lane; i < d_k; i += 32) {
                    float v = sh_k[dt][i]; sq += v * v;
                }
                sq = simd_sum(sq);
                float invK = rsqrt(sq + eps);
                for (int i = lane; i < d_k; i += 32) sh_k[dt][i] *= invK;
            }
        } else {
            for (int dt = sgitg; dt < chunk_len; dt += SIMDS_PER_TG) {
                for (int i = lane; i < d_k; i += 32) sh_q[dt][i] *= q_scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Sequential delta rule for this SG's j across chunk_len timesteps. ──
        // 32 lanes in the SG cooperatively reduce dk each timestep via simd_sum.
        // State stays register-resident in s_reg[SIMD_ITERS].
        for (int dt = 0; dt < chunk_len; ++dt) {
            float decay  = exp(sh_g[dt]);
            float beta_t = sh_beta[dt];
            float v_t_j  = sh_v[dt][sgitg];

            // Phase 1: decay state, accumulate v_pred = <s, k>
            float v_pred = 0.0f;
            for (int ii = 0; ii < SIMD_ITERS; ++ii) {
                int i = lane + ii * 32;
                if (i < d_k) {
                    s_reg[ii] *= decay;
                    v_pred += s_reg[ii] * sh_k[dt][i];
                }
            }
            v_pred = simd_sum(v_pred);

            float delta = beta_t * (v_t_j - v_pred);

            // Phase 2: state += delta * k, accumulate o = <s, q>
            float o_t_j = 0.0f;
            for (int ii = 0; ii < SIMD_ITERS; ++ii) {
                int i = lane + ii * 32;
                if (i < d_k) {
                    s_reg[ii] += sh_k[dt][i] * delta;
                    o_t_j += s_reg[ii] * sh_q[dt][i];
                }
            }
            o_t_j = simd_sum(o_t_j);

            if (lane == 0) {
                attn_out[output_offset_v2(b, t0 + dt, h, j, param)] = (ftype)o_t_j;
            }
        }
        // No trailing barrier needed: the next iteration's loads only touch
        // sh_q/sh_k/sh_v/sh_g/sh_beta which have been fully consumed here,
        // and the load loop starts with an implicit "any thread may write".
        // But we must ensure no stray thread proceeds to the next load before
        // others finish their reads of the current chunk's sh_* data.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Flush register-resident state back to device memory. ──
    for (int ii = 0; ii < SIMD_ITERS; ++ii) {
        int i = lane + ii * 32;
        if (i < d_k) state_gm[i] = (ftype)s_reg[ii];
    }
}
)metal";


// Chunk-64 Gated Delta Rule prefill without sequence-sized temporary tensors.
//
// The first kernel prepares the two state-independent chunk matrices in
// parallel. The original V values are copied to the not-yet-produced output,
// then conv_out's V region is reused in place:
//
//   V[:, 0:64]   <- T = (I - A)^-1,
//                     A = -strict_lower((K * beta) K^T * decay)
//   V[:, 64:128] <- P =  lower_inclusive(Q K^T * decay)
//
// The recurrent kernel owns a 32-column state slice, walks chunks in order,
// computes v_new = T @ (v*beta - (k*beta*exp(gc))@S) in threadgroup
// memory, then computes output and updates S. Consequently the path retains
// the chunk-parallel dependency reduction without allocating Q/K/V/U/W/P
// scratch tensors. The output buffer only contains transient V values until
// each corresponding final output slice is written.
static const char* gLinearAttnChunk64Inplace = R"metal(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#if MNN_METAL_FLOAT16_STORAGE
typedef half ftype;
#else
typedef float ftype;
#endif

struct LinearAttnParam {
    int batch;
    int conv_dim;
    int seq_len;
    int kernel_size;
    int conv_state_size;
    int num_k_heads;
    int num_v_heads;
    int head_k_dim;
    int head_v_dim;
    int key_dim;
    int val_dim;
    int gqa_factor;
    int use_l2norm;
    int qkv_c4;
    int gate_c4;
    int beta_c4;
    int output_c4;
    float q_scale;
    int commit_len;    // lazy-commit: pending tokens to replay before this block
    int pending_seq;   // length of the saved pending block
    int lazy_mode;     // 1: do not persist state after the new block (spec verify)
    // gate/beta chain fold (gate_c4 & 2 / beta_c4 & 2): per-head constants
    // -exp(A_log)[h] and dt_bias[h]; capped at 64 heads.
    float gate_coef[64];
    float gate_bias[64];
};

inline int ck_c4_offset(int token, int channel, int token_count) {
    return ((channel >> 2) * token_count + token) * 4 + (channel & 3);
}

inline int ck_token_channel_offset(int b, int t, int c, int channel, int packed,
                                   constant LinearAttnParam& param) {
    if (packed) {
        return ck_c4_offset(b * param.seq_len + t, c, param.batch * param.seq_len);
    }
    return (b * param.seq_len + t) * channel + c;
}

// gate/beta chain fold (bit 2 of gate_c4/beta_c4): replicates the unfused
// elementwise chain op-for-op — Binary ops compute in ftype (half on fp16
// builds), Unary ops in fp32 with the MNNEXP +-87 clamp — with an ftype
// store-rounding after every op, so the folded value is bit-identical to the
// separate-dispatch chain output.
inline float linear_attn_gate_fold(float a, int h, constant LinearAttnParam& p) {
    ftype x = (ftype)a + (ftype)p.gate_bias[h];       // ADD dt_bias  (Binary, half)
    x = (ftype)exp(clamp((float)x, -87.0f, 87.0f));   // EXP          (Unary, fp32)
    x = x + (ftype)1.0f;                              // ADD +1       (Binary, half)
    x = (ftype)log((float)x);                         // LOG          (Unary, fp32)
    x = (ftype)p.gate_coef[h] * x;                    // MUL -exp(A_log)
    return (float)x;
}
inline float linear_attn_beta_fold(float b) {
    return (float)(ftype)(1.0f / (1.0f + exp(clamp(-b, -87.0f, 87.0f))));
}

inline int ck_output_offset(int b, int t, int h, int d, constant LinearAttnParam& param) {
    int token = (b * param.seq_len + t) * param.num_v_heads + h;
    if (param.output_c4) {
        return ck_c4_offset(token, d, param.batch * param.seq_len * param.num_v_heads);
    }
    return token * param.head_v_dim + d;
}

#define CK_NSG      4
#define CK_NT       (CK_CHUNK / 16)
#define CK_DK_TILES (CK_DK / 16)
#define CK_A_CHANNEL 0
#define CK_P_CHANNEL CK_CHUNK

// B is supplied as N rows of K values.
#define CK_DESC_TT matmul2d_descriptor(16, 32, 16, false, true, true, \
                                       matmul2d_descriptor::mode::multiply_accumulate)
// B is supplied as K rows of N values.
#define CK_DESC_TF matmul2d_descriptor(16, CK_W, 16, false, false, true, \
                                       matmul2d_descriptor::mode::multiply_accumulate)

// Grid: (ceil(L / 64), B * H), 4 simdgroups per threadgroup.
kernel void linear_attn_chunk64_prep_inplace(
    device ftype* conv_out                  [[buffer(0)]],
    const device ftype* gate               [[buffer(1)]],
    const device ftype* beta               [[buffer(2)]],
    device ftype* attn_out                 [[buffer(3)]],
    constant LinearAttnParam& param        [[buffer(4)]],
    uint3 tgpig  [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const int L = param.seq_len;
    const int H = param.num_v_heads;
    const int D = param.conv_dim;
    const int c = int(tgpig.x);
    const int bh = int(tgpig.y);
    const int b = bh / H;
    const int h = bh % H;
    const int kHead = h / param.gqa_factor;
    const int c0 = c * CK_CHUNK;
    const uint tid = uint(sgitg) * 32u + uint(tiisg);

    device ftype* convBase = conv_out + (long)b * D * L;
    device ftype* scratchBase = convBase + (2 * param.key_dim + h * CK_DV) * L;

    threadgroup float gcTg[CK_CHUNK];
    threadgroup float betaTg[CK_CHUNK];
    threadgroup float qInvTg[CK_CHUNK];
    threadgroup float kInvTg[CK_CHUNK];
    threadgroup float aTg[CK_CHUNK * CK_CHUNK];

    // Preserve V in the output buffer before its conv_out storage is reused.
    for (int e = int(tid); e < CK_CHUNK * CK_DV; e += CK_NSG * 32) {
        int m = e / CK_DV;
        int n = e % CK_DV;
        int token = c0 + m;
        if (token < L) {
            ftype vv = scratchBase[(long)n * L + token];
            attn_out[ck_output_offset(b, token, h, n, param)] = vv;
        }
    }
    // Every original V element must be read before A/P starts overwriting it.
    threadgroup_barrier(mem_flags::mem_device);

    if (tid < CK_CHUNK) {
        int token = c0 + int(tid);
        float gv = 0.0f;
        float bv = 0.0f;
        if (token < L) {
            gv = float(gate[ck_token_channel_offset(b, token, h, H, param.gate_c4 & 1, param)]);
            if (param.gate_c4 & 2) {
                gv = linear_attn_gate_fold(gv, h, param);
            }
            bv = float(beta[ck_token_channel_offset(b, token, h, H, param.beta_c4 & 1, param)]);
            if (param.beta_c4 & 2) {
                bv = linear_attn_beta_fold(bv);
            }
        }
        // Finite pairwise cumsum differences are required by the chunk form.
        // exp(-30) is below fp16's smallest subnormal, preserving state-reset
        // semantics, while the upper clamp protects malformed positive gates.
        gcTg[tid] = clamp(gv, -30.0f, 0.0f);
        betaTg[tid] = bv;
        qInvTg[tid] = 0.0f;
        kInvTg[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float acc = 0.0f;
        for (int i = 0; i < CK_CHUNK; ++i) {
            acc += gcTg[i];
            gcTg[i] = acc;
        }
    }

    if (param.use_l2norm) {
        for (int m = int(sgitg); m < CK_CHUNK; m += CK_NSG) {
            int token = c0 + m;
            float qSum = 0.0f;
            float kSum = 0.0f;
            if (token < L) {
                for (int d = int(tiisg); d < CK_DK; d += 32) {
                    float qv = float(convBase[(kHead * CK_DK + d) * L + token]);
                    float kv = float(convBase[(param.key_dim + kHead * CK_DK + d) * L + token]);
                    qSum += qv * qv;
                    kSum += kv * kv;
                }
            }
            qSum = simd_sum(qSum);
            kSum = simd_sum(kSum);
            if (tiisg == 0 && token < L) {
                qInvTg[m] = rsqrt(qSum + 1.0e-6f) * param.q_scale;
                kInvTg[m] = rsqrt(kSum + 1.0e-6f);
            }
        }
    } else if (tid < CK_CHUNK && c0 + int(tid) < L) {
        qInvTg[tid] = param.q_scale;
        kInvTg[tid] = 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ushort qid = tiisg >> 2;
    const ushort fm = (qid & 4) | ((tiisg >> 1) & 3);
    const ushort fn = ((qid & 2) | (tiisg & 1)) * 4;
    matmul2d<CK_DESC_TT, metal::execution_simdgroup> mm;

    // A = -strict_lower((K * beta) @ K^T * exp(gc_m - gc_n)).
    for (int nt = 0; nt < CK_CHUNK / 32; ++nt) {
        float acc[16];
        for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
        for (int kt = 0; kt < CK_DK_TILES; ++kt) {
            auto ctA = mm.get_left_input_cooperative_tensor<ftype, ftype, float>();
            auto ctB = mm.get_right_input_cooperative_tensor<ftype, ftype, float>();
            auto ctC = mm.get_destination_cooperative_tensor<decltype(ctA), decltype(ctB), float>();
            for (ushort g = 0; g < 2; ++g) {
                int m = int(sgitg) * 16 + int(fm) + int(g) * 8;
                int token = c0 + m;
                for (ushort j = 0; j < 4; ++j) {
                    int d = kt * 16 + int(fn) + int(j);
                    float kv = token < L ? float(convBase[(param.key_dim + kHead * CK_DK + d) * L + token]) : 0.0f;
                    ctA[g * 4 + j] = ftype(kv * kInvTg[m] * betaTg[m]);
                }
            }
            for (ushort g = 0; g < 4; ++g) {
                int n = nt * 32 + int(fm) + (int(g) & 1) * 8 + (int(g) >> 1) * 16;
                int token = c0 + n;
                for (ushort j = 0; j < 4; ++j) {
                    int d = kt * 16 + int(fn) + int(j);
                    float kv = token < L ? float(convBase[(param.key_dim + kHead * CK_DK + d) * L + token]) : 0.0f;
                    ctB[g * 4 + j] = ftype(kv * kInvTg[n]);
                }
            }
            for (int i = 0; i < 16; ++i) ctC[i] = acc[i];
            mm.run(ctA, ctB, ctC);
            for (int i = 0; i < 16; ++i) acc[i] = ctC[i];
        }
        for (ushort i = 0; i < 16; ++i) {
            int m = int(sgitg) * 16 + int(fm) + ((int(i) >> 2) & 1) * 8;
            int n = nt * 32 + int(fn) + (int(i) & 3) + (int(i) >> 3) * 16;
            float av = m > n ? -acc[i] * exp(gcTg[m] - gcTg[n]) : 0.0f;
            aTg[m * CK_CHUNK + n] = av;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Forward substitution produces T - I in place. SG0 lanes own columns
    // {lane, lane+32}; barriers preserve the read-before-write ordering of a
    // row while previously completed rows are consumed.
    if (sgitg == 0) {
        const int j0 = int(tiisg);
        const int j1 = int(tiisg) + 32;
        for (int i = 1; i < CK_CHUNK; ++i) {
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            for (int m = 0; m < i; ++m) {
                float row = aTg[i * CK_CHUNK + m];
                acc0 += row * aTg[m * CK_CHUNK + j0];
                acc1 += row * aTg[m * CK_CHUNK + j1];
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (j0 < i) aTg[i * CK_CHUNK + j0] += acc0;
            if (j1 < i) aTg[i * CK_CHUNK + j1] += acc1;
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = int(tid); e < CK_CHUNK * CK_CHUNK; e += CK_NSG * 32) {
        int m = e / CK_CHUNK;
        int n = e % CK_CHUNK;
        if (c0 + m < L) {
            float tv = aTg[e] + (m == n ? 1.0f : 0.0f);
            scratchBase[(CK_A_CHANNEL + n) * L + c0 + m] = ftype(tv);
        }
    }

    // P = lower_inclusive(Q @ K^T * exp(gc_m - gc_n)).
    for (int nt = 0; nt < CK_CHUNK / 32; ++nt) {
        float acc[16];
        for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
        for (int kt = 0; kt < CK_DK_TILES; ++kt) {
            auto ctA = mm.get_left_input_cooperative_tensor<ftype, ftype, float>();
            auto ctB = mm.get_right_input_cooperative_tensor<ftype, ftype, float>();
            auto ctC = mm.get_destination_cooperative_tensor<decltype(ctA), decltype(ctB), float>();
            for (ushort g = 0; g < 2; ++g) {
                int m = int(sgitg) * 16 + int(fm) + int(g) * 8;
                int token = c0 + m;
                for (ushort j = 0; j < 4; ++j) {
                    int d = kt * 16 + int(fn) + int(j);
                    float qv = token < L ? float(convBase[(kHead * CK_DK + d) * L + token]) : 0.0f;
                    ctA[g * 4 + j] = ftype(qv * qInvTg[m]);
                }
            }
            for (ushort g = 0; g < 4; ++g) {
                int n = nt * 32 + int(fm) + (int(g) & 1) * 8 + (int(g) >> 1) * 16;
                int token = c0 + n;
                for (ushort j = 0; j < 4; ++j) {
                    int d = kt * 16 + int(fn) + int(j);
                    float kv = token < L ? float(convBase[(param.key_dim + kHead * CK_DK + d) * L + token]) : 0.0f;
                    ctB[g * 4 + j] = ftype(kv * kInvTg[n]);
                }
            }
            for (int i = 0; i < 16; ++i) ctC[i] = acc[i];
            mm.run(ctA, ctB, ctC);
            for (int i = 0; i < 16; ++i) acc[i] = ctC[i];
        }
        for (ushort i = 0; i < 16; ++i) {
            int m = int(sgitg) * 16 + int(fm) + ((int(i) >> 2) & 1) * 8;
            int n = nt * 32 + int(fn) + (int(i) & 3) + (int(i) >> 3) * 16;
            float pv = m >= n ? acc[i] * exp(gcTg[m] - gcTg[n]) : 0.0f;
            if (c0 + m < L) {
                scratchBase[(CK_P_CHANNEL + n) * L + c0 + m] = ftype(pv);
            }
        }
    }
}

// Grid: (DV / 32, B * H), 4 simdgroups per threadgroup.
kernel void linear_attn_chunk64_recurrent_inplace(
    const device ftype* conv_out            [[buffer(0)]],
    const device ftype* gate               [[buffer(1)]],
    const device ftype* beta               [[buffer(2)]],
    device ftype* recurrent_state          [[buffer(3)]],
    device ftype* attn_out                 [[buffer(4)]],
    constant LinearAttnParam& param        [[buffer(5)]],
    uint3 tgpig  [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const int L = param.seq_len;
    const int H = param.num_v_heads;
    const int D = param.conv_dim;
    const int chunks = (L + CK_CHUNK - 1) / CK_CHUNK;
    const int ws = int(tgpig.x);
    const int bh = int(tgpig.y);
    const int b = bh / H;
    const int h = bh % H;
    const int kHead = h / param.gqa_factor;
    const int n0 = ws * CK_W;
    const uint tid = uint(sgitg) * 32u + uint(tiisg);

    const device ftype* convBase = conv_out + (long)b * D * L;
    const device ftype* scratchBase = convBase + (2 * param.key_dim + h * CK_DV) * L;

    // The per-tile/per-lane layout is simultaneously a matmul destination and
    // a non-transposed right operand, avoiding state transposes between stages.
    threadgroup float stateTg[CK_DK_TILES][32][16];
    threadgroup float vnTg[CK_CHUNK][CK_W];
    threadgroup float gcTg[CK_CHUNK];
    threadgroup float betaTg[CK_CHUNK];
    threadgroup float qInvTg[CK_CHUNK];
    threadgroup float kInvTg[CK_CHUNK];

    const ushort qid = tiisg >> 2;
    const ushort fm = (qid & 4) | ((tiisg >> 1) & 3);
    const ushort fn = ((qid & 2) | (tiisg & 1)) * 4;

    for (int kt = int(sgitg); kt < CK_DK_TILES; kt += CK_NSG) {
        for (ushort i = 0; i < 16; ++i) {
            int dk = kt * 16 + int(fm) + ((int(i) >> 2) & 1) * 8;
            int n = int(fn) + (int(i) & 3) + (int(i) >> 3) * 16;
            stateTg[kt][tiisg][i] = float(recurrent_state[(long)(bh * CK_DV + n0 + n) * CK_DK + dk]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    matmul2d<CK_DESC_TF, metal::execution_simdgroup> mm;

    for (int c = 0; c < chunks; ++c) {
        const int c0 = c * CK_CHUNK;

        if (tid < CK_CHUNK) {
            int token = c0 + int(tid);
            float gv = 0.0f;
            float bv = 0.0f;
            if (token < L) {
                gv = float(gate[ck_token_channel_offset(b, token, h, H, param.gate_c4 & 1, param)]);
                if (param.gate_c4 & 2) {
                    gv = linear_attn_gate_fold(gv, h, param);
                }
                bv = float(beta[ck_token_channel_offset(b, token, h, H, param.beta_c4 & 1, param)]);
                if (param.beta_c4 & 2) {
                    bv = linear_attn_beta_fold(bv);
                }
            }
            gcTg[tid] = clamp(gv, -30.0f, 0.0f);
            betaTg[tid] = bv;
            qInvTg[tid] = 0.0f;
            kInvTg[tid] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float acc = 0.0f;
            for (int i = 0; i < CK_CHUNK; ++i) {
                acc += gcTg[i];
                gcTg[i] = acc;
            }
        }
        if (param.use_l2norm) {
            for (int m = int(sgitg); m < CK_CHUNK; m += CK_NSG) {
                int token = c0 + m;
                float qSum = 0.0f;
                float kSum = 0.0f;
                if (token < L) {
                    for (int d = int(tiisg); d < CK_DK; d += 32) {
                        float qv = float(convBase[(kHead * CK_DK + d) * L + token]);
                        float kv = float(convBase[(param.key_dim + kHead * CK_DK + d) * L + token]);
                        qSum += qv * qv;
                        kSum += kv * kv;
                    }
                }
                qSum = simd_sum(qSum);
                kSum = simd_sum(kSum);
                if (tiisg == 0 && token < L) {
                    qInvTg[m] = rsqrt(qSum + 1.0e-6f) * param.q_scale;
                    kInvTg[m] = rsqrt(kSum + 1.0e-6f);
                }
            }
        } else if (tid < CK_CHUNK && c0 + int(tid) < L) {
            qInvTg[tid] = param.q_scale;
            kInvTg[tid] = 1.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // tmp = v*beta - (k*beta*exp(gc)) @ state.
        {
            float acc[16];
            for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
            for (int kt = 0; kt < CK_DK_TILES; ++kt) {
                auto ctA = mm.get_left_input_cooperative_tensor<ftype, float, float>();
                auto ctB = mm.get_right_input_cooperative_tensor<ftype, float, float>();
                auto ctC = mm.get_destination_cooperative_tensor<decltype(ctA), decltype(ctB), float>();
                for (ushort g = 0; g < 2; ++g) {
                    int m = int(sgitg) * 16 + int(fm) + int(g) * 8;
                    int token = c0 + m;
                    float scale = betaTg[m] * exp(gcTg[m]) * kInvTg[m];
                    for (ushort j = 0; j < 4; ++j) {
                        int d = kt * 16 + int(fn) + int(j);
                        float kv = token < L ? float(convBase[(param.key_dim + kHead * CK_DK + d) * L + token]) : 0.0f;
                        ctA[g * 4 + j] = ftype(kv * scale);
                    }
                }
                for (ushort i = 0; i < 16; ++i) {
                    ctB[i] = stateTg[kt][tiisg][i];
                    ctC[i] = acc[i];
                }
                mm.run(ctA, ctB, ctC);
                for (int i = 0; i < 16; ++i) acc[i] = ctC[i];
            }
            for (ushort i = 0; i < 16; ++i) {
                int m = int(sgitg) * 16 + int(fm) + ((int(i) >> 2) & 1) * 8;
                int token = c0 + m;
                int n = int(fn) + (int(i) & 3) + (int(i) >> 3) * 16;
                float vv = token < L ? float(attn_out[ck_output_offset(b, token, h, n0 + n, param)]) : 0.0f;
                vnTg[m][n] = vv * betaTg[m] - acc[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // v_new = T @ tmp. All simdgroups retain their result tile in
        // registers until the barrier, then overwrite tmp in place; this
        // avoids a second 64x32 threadgroup buffer.
        float vnewAcc[16];
        for (int i = 0; i < 16; ++i) vnewAcc[i] = 0.0f;
        for (int kt = 0; kt < CK_NT; ++kt) {
            auto ctA = mm.get_left_input_cooperative_tensor<ftype, float, float>();
            auto ctB = mm.get_right_input_cooperative_tensor<ftype, float, float>();
            auto ctC = mm.get_destination_cooperative_tensor<decltype(ctA), decltype(ctB), float>();
            for (ushort g = 0; g < 2; ++g) {
                int m = int(sgitg) * 16 + int(fm) + int(g) * 8;
                for (ushort j = 0; j < 4; ++j) {
                    int tokenK = kt * 16 + int(fn) + int(j);
                    ctA[g * 4 + j] = c0 + m < L
                        ? scratchBase[(CK_A_CHANNEL + tokenK) * L + c0 + m]
                        : ftype(0);
                }
            }
            for (ushort i = 0; i < 16; ++i) {
                int tokenK = kt * 16 + int(fm) + ((int(i) >> 2) & 1) * 8;
                int n = int(fn) + (int(i) & 3) + (int(i) >> 3) * 16;
                ctB[i] = vnTg[tokenK][n];
                ctC[i] = vnewAcc[i];
            }
            mm.run(ctA, ctB, ctC);
            for (int i = 0; i < 16; ++i) vnewAcc[i] = ctC[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ushort i = 0; i < 16; ++i) {
            int m = int(sgitg) * 16 + int(fm) + ((int(i) >> 2) & 1) * 8;
            int n = int(fn) + (int(i) & 3) + (int(i) >> 3) * 16;
            vnTg[m][n] = vnewAcc[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O = (q*exp(gc)) @ state + P @ v_new.
        float outAcc[16];
        for (int i = 0; i < 16; ++i) outAcc[i] = 0.0f;
        for (int kt = 0; kt < CK_DK_TILES; ++kt) {
            auto ctA = mm.get_left_input_cooperative_tensor<ftype, float, float>();
            auto ctB = mm.get_right_input_cooperative_tensor<ftype, float, float>();
            auto ctC = mm.get_destination_cooperative_tensor<decltype(ctA), decltype(ctB), float>();
            for (ushort g = 0; g < 2; ++g) {
                int m = int(sgitg) * 16 + int(fm) + int(g) * 8;
                int token = c0 + m;
                float scale = exp(gcTg[m]) * qInvTg[m];
                for (ushort j = 0; j < 4; ++j) {
                    int d = kt * 16 + int(fn) + int(j);
                    float qv = token < L ? float(convBase[(kHead * CK_DK + d) * L + token]) : 0.0f;
                    ctA[g * 4 + j] = ftype(qv * scale);
                }
            }
            for (ushort i = 0; i < 16; ++i) {
                ctB[i] = stateTg[kt][tiisg][i];
                ctC[i] = outAcc[i];
            }
            mm.run(ctA, ctB, ctC);
            for (int i = 0; i < 16; ++i) outAcc[i] = ctC[i];
        }
        for (int kt = 0; kt < CK_NT; ++kt) {
            auto ctA = mm.get_left_input_cooperative_tensor<ftype, float, float>();
            auto ctB = mm.get_right_input_cooperative_tensor<ftype, float, float>();
            auto ctC = mm.get_destination_cooperative_tensor<decltype(ctA), decltype(ctB), float>();
            for (ushort g = 0; g < 2; ++g) {
                int m = int(sgitg) * 16 + int(fm) + int(g) * 8;
                for (ushort j = 0; j < 4; ++j) {
                    int tokenK = kt * 16 + int(fn) + int(j);
                    ctA[g * 4 + j] = c0 + m < L
                        ? scratchBase[(CK_P_CHANNEL + tokenK) * L + c0 + m]
                        : ftype(0);
                }
            }
            for (ushort i = 0; i < 16; ++i) {
                int tokenK = kt * 16 + int(fm) + ((int(i) >> 2) & 1) * 8;
                int n = int(fn) + (int(i) & 3) + (int(i) >> 3) * 16;
                ctB[i] = vnTg[tokenK][n];
                ctC[i] = outAcc[i];
            }
            mm.run(ctA, ctB, ctC);
            for (int i = 0; i < 16; ++i) outAcc[i] = ctC[i];
        }

        for (ushort i = 0; i < 16; ++i) {
            int m = int(sgitg) * 16 + int(fm) + ((int(i) >> 2) & 1) * 8;
            int token = c0 + m;
            if (token >= L) continue;
            int n = int(fn) + (int(i) & 3) + (int(i) >> 3) * 16;
            attn_out[ck_output_offset(b, token, h, n0 + n, param)] = ftype(outAcc[i]);
        }

        // state = exp(gc_last)*state + k_dec^T @ v_new.
        const float totalDecay = exp(gcTg[CK_CHUNK - 1]);
        for (int mt = int(sgitg); mt < CK_DK_TILES; mt += CK_NSG) {
            float acc[16];
            for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
            for (int kt = 0; kt < CK_NT; ++kt) {
                auto ctA = mm.get_left_input_cooperative_tensor<ftype, float, float>();
                auto ctB = mm.get_right_input_cooperative_tensor<ftype, float, float>();
                auto ctC = mm.get_destination_cooperative_tensor<decltype(ctA), decltype(ctB), float>();
                for (ushort i = 0; i < 8; ++i) {
                    int tokenK = kt * 16 + int(fn) + (int(i) & 3);
                    int token = c0 + tokenK;
                    int dk = mt * 16 + int(fm) + (int(i) >> 2) * 8;
                    float kv = token < L ? float(convBase[(param.key_dim + kHead * CK_DK + dk) * L + token]) : 0.0f;
                    ctA[i] = ftype(kv * kInvTg[tokenK] * exp(gcTg[CK_CHUNK - 1] - gcTg[tokenK]));
                }
                for (ushort i = 0; i < 16; ++i) {
                    int tokenK = kt * 16 + int(fm) + ((int(i) >> 2) & 1) * 8;
                    int n = int(fn) + (int(i) & 3) + (int(i) >> 3) * 16;
                    ctB[i] = vnTg[tokenK][n];
                    ctC[i] = acc[i];
                }
                mm.run(ctA, ctB, ctC);
                for (int i = 0; i < 16; ++i) acc[i] = ctC[i];
            }
            for (ushort i = 0; i < 16; ++i) {
                stateTg[mt][tiisg][i] = totalDecay * stateTg[mt][tiisg][i] + acc[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int kt = int(sgitg); kt < CK_DK_TILES; kt += CK_NSG) {
        for (ushort i = 0; i < 16; ++i) {
            int dk = kt * 16 + int(fm) + ((int(i) >> 2) & 1) * 8;
            int n = int(fn) + (int(i) & 3) + (int(i) >> 3) * 16;
            recurrent_state[(long)(bh * CK_DV + n0 + n) * CK_DK + dk] = ftype(stateTg[kt][tiisg][i]);
        }
    }
}
)metal";

// simdgroup_matrix (8x8 fp32 MMA) port of linear_attn_flash_chunk for devices
// without the Metal tensor API (M4-class Macs, iPhone A-series). Replaces the
// per-timestep scalar fused_chunk_sg path for long prefill: the chunked
// gated-delta-rule formulation turns the sequential recurrence into a handful
// of 16x{16,DV_BLOCK}xHEAD_K_DIM matmuls per chunk, which map directly onto
// simdgroup_float8x8 tiles. Algorithm, buffers and barriers mirror
// linear_attn_flash_chunk step-for-step; only the three tensor-API matmul
// blocks (steps 4, 7, 9, 12) are re-expressed as 8x8 MMA loops.
static const char* gLinearAttnFlashChunkSGMM = R"metal(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#if MNN_METAL_FLOAT16_STORAGE
typedef half ftype;
#else
typedef float ftype;
#endif

struct LinearAttnParam {
    int batch;
    int conv_dim;
    int seq_len;
    int kernel_size;
    int conv_state_size;
    int num_k_heads;
    int num_v_heads;
    int head_k_dim;
    int head_v_dim;
    int key_dim;
    int val_dim;
    int gqa_factor;
    int use_l2norm;
    int qkv_c4;
    int gate_c4;
    int beta_c4;
    int output_c4;
    float q_scale;
    int commit_len;    // lazy-commit: pending tokens to replay before this block
    int pending_seq;   // length of the saved pending block
    int lazy_mode;     // 1: do not persist state after the new block (spec verify)
    // gate/beta chain fold (gate_c4 & 2 / beta_c4 & 2): per-head constants
    // -exp(A_log)[h] and dt_bias[h]; capped at 64 heads.
    float gate_coef[64];
    float gate_bias[64];
};

inline int c4_offset_v2(int token, int channel, int token_count) {
    return ((channel >> 2) * token_count + token) * 4 + (channel & 3);
}

inline int token_channel_offset(int b, int t, int c, int channel, int packed,
                                constant LinearAttnParam& param) {
    if (packed) {
        return c4_offset_v2(b * param.seq_len + t, c, param.batch * param.seq_len);
    }
    return (b * param.seq_len + t) * channel + c;
}

// gate/beta chain fold (bit 2 of gate_c4/beta_c4): replicates the unfused
// elementwise chain op-for-op — Binary ops compute in ftype (half on fp16
// builds), Unary ops in fp32 with the MNNEXP +-87 clamp — with an ftype
// store-rounding after every op, so the folded value is bit-identical to the
// separate-dispatch chain output.
inline float linear_attn_gate_fold(float a, int h, constant LinearAttnParam& p) {
    ftype x = (ftype)a + (ftype)p.gate_bias[h];       // ADD dt_bias  (Binary, half)
    x = (ftype)exp(clamp((float)x, -87.0f, 87.0f));   // EXP          (Unary, fp32)
    x = x + (ftype)1.0f;                              // ADD +1       (Binary, half)
    x = (ftype)log((float)x);                         // LOG          (Unary, fp32)
    x = (ftype)p.gate_coef[h] * x;                    // MUL -exp(A_log)
    return (float)x;
}
inline float linear_attn_beta_fold(float b) {
    return (float)(ftype)(1.0f / (1.0f + exp(clamp(-b, -87.0f, 87.0f))));
}

inline int output_offset_v2(int b, int t, int h, int d, constant LinearAttnParam& param) {
    int token = (b * param.seq_len + t) * param.num_v_heads + h;
    if (param.output_c4) {
        return c4_offset_v2(token, d, param.batch * param.seq_len * param.num_v_heads);
    }
    return token * param.head_v_dim + d;
}

// CHUNK_BT (=16), HEAD_K_DIM, HEAD_V_DIM, DV_BLOCK (=16), SIMDS_PER_TG are
// injected as compile-time macros. CHUNK_BT and DV_BLOCK must be multiples of
// 8; HEAD_K_DIM must be a multiple of 8 and <= 128 (threadgroup memory).

kernel void linear_attn_flash_chunk_sgmm(
    const device ftype* conv_out         [[buffer(0)]],
    const device ftype* gate             [[buffer(1)]],
    const device ftype* beta             [[buffer(2)]],
    device ftype* recurrent_state        [[buffer(3)]],
    device ftype* attn_out               [[buffer(4)]],
    constant LinearAttnParam& param      [[buffer(5)]],
    uint tgpig [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const int L         = param.seq_len;
    const int H         = param.num_v_heads;
    const int d_k       = param.head_k_dim;
    const int d_v       = param.head_v_dim;
    const int D         = param.conv_dim;
    const int key_dim   = param.key_dim;
    const int gqa_factor= param.gqa_factor;
    const int use_l2norm= param.use_l2norm;
    const float q_scale = param.q_scale;

    const int dv_blocks = HEAD_V_DIM / DV_BLOCK;
    const int idx       = (int)tgpig;
    const int bh        = idx / dv_blocks;
    const int dvb       = idx % dv_blocks;
    const int b         = bh / H;
    const int h         = bh % H;
    const int k_head    = h / gqa_factor;
    const int dv_off    = dvb * DV_BLOCK;

    const int TG_THREADS = SIMDS_PER_TG * 32;
    const int tid        = (int)(sgitg * 32 + lane);
    const device ftype* conv_base = conv_out + b * D * L;

    threadgroup float sh_q   [CHUNK_BT][HEAD_K_DIM];
    threadgroup float sh_k   [CHUNK_BT][HEAD_K_DIM];
    threadgroup float sh_state[HEAD_K_DIM][DV_BLOCK]; // in-place accumulate
    threadgroup float sh_attn[CHUNK_BT][CHUNK_BT];
    threadgroup float sh_qkdm[CHUNK_BT][CHUNK_BT];
    threadgroup float sh_v   [CHUNK_BT][DV_BLOCK]; // v_beta / v_new / v_nb
    threadgroup float sh_K_S [CHUNK_BT][DV_BLOCK]; // K@state / v_prime
    threadgroup float sh_out [CHUNK_BT][DV_BLOCK]; // Q·expG @ state / final
    threadgroup float sh_g   [CHUNK_BT];
    threadgroup float sh_beta[CHUNK_BT];
    threadgroup float sh_G   [CHUNK_BT];

    device ftype* state_gm = recurrent_state + (b * H + h) * d_v * d_k;
    // Load state once.
    for (int p = tid; p < HEAD_K_DIM * DV_BLOCK; p += TG_THREADS) {
        int dki = p / DV_BLOCK, dvi = p % DV_BLOCK;
        sh_state[dki][dvi] = (float)state_gm[(dv_off + dvi) * d_k + dki];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int t0 = 0; t0 < L; t0 += CHUNK_BT) {
        int chunk_len = min((int)CHUNK_BT, L - t0);

        // ── 1. Cooperative load Q, K, V, g, β. ──
        for (int p = tid; p < chunk_len * d_k; p += TG_THREADS) {
            int dt = p / d_k, di = p % d_k;
            int t_abs = t0 + dt;
            sh_q[dt][di] = (float)conv_base[(k_head * d_k + di) * L + t_abs];
            sh_k[dt][di] = (float)conv_base[(key_dim + k_head * d_k + di) * L + t_abs];
        }
        for (int p = tid; p < chunk_len * DV_BLOCK; p += TG_THREADS) {
            int dt = p / DV_BLOCK, dvi = p % DV_BLOCK;
            int t_abs = t0 + dt;
            sh_v[dt][dvi] = (float)conv_base[(2 * key_dim + h * d_v + dv_off + dvi) * L + t_abs];
        }
        for (int dt = tid; dt < chunk_len; dt += TG_THREADS) {
            // Clamp gate to [-88, 0]: with fp16 storage the upstream gate may
            // overflow to -inf, which would poison the cumsum in sh_G (see the
            // matching comment in linear_attn_flash_chunk).
            int t_abs = t0 + dt;
            float g_val = (float)gate[token_channel_offset(b, t_abs, h, H, param.gate_c4 & 1, param)];
            if (param.gate_c4 & 2) {
                g_val = linear_attn_gate_fold(g_val, h, param);
            }
            g_val = clamp(g_val, -88.0f, 0.0f);
            sh_g[dt]    = g_val;
            float b_raw = (float)beta[token_channel_offset(b, t_abs, h, H, param.beta_c4 & 1, param)];
            sh_beta[dt] = (param.beta_c4 & 2) ? linear_attn_beta_fold(b_raw) : b_raw;
        }
        for (int dt = chunk_len + tid; dt < CHUNK_BT; dt += TG_THREADS) {
            sh_g[dt] = 0.0f; sh_beta[dt] = 0.0f;
            for (int di = 0; di < HEAD_K_DIM; ++di) { sh_q[dt][di] = 0.0f; sh_k[dt][di] = 0.0f; }
            for (int dvi = 0; dvi < DV_BLOCK; ++dvi) sh_v[dt][dvi] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── 2. L2 norm + q_scale (in-place). ──
        if (use_l2norm) {
            const float eps = 1e-6f;
            for (int dt = sgitg; dt < chunk_len; dt += SIMDS_PER_TG) {
                float sq = 0.0f;
                for (int i = lane; i < d_k; i += 32) { float v = sh_q[dt][i]; sq += v * v; }
                sq = simd_sum(sq);
                float invQ = rsqrt(sq + eps) * q_scale;
                for (int i = lane; i < d_k; i += 32) sh_q[dt][i] *= invQ;
                sq = 0.0f;
                for (int i = lane; i < d_k; i += 32) { float v = sh_k[dt][i]; sq += v * v; }
                sq = simd_sum(sq);
                float invK = rsqrt(sq + eps);
                for (int i = lane; i < d_k; i += 32) sh_k[dt][i] *= invK;
            }
        } else {
            for (int dt = sgitg; dt < chunk_len; dt += SIMDS_PER_TG) {
                for (int i = lane; i < d_k; i += 32) sh_q[dt][i] *= q_scale;
            }
        }
        // ── 3. cumsum(g) → sh_G — FUSED with L2 norm barrier ──
        if (tid == 0) {
            float acc = 0.0f;
            for (int t = 0; t < chunk_len; ++t) { acc += sh_g[t]; sh_G[t] = acc; }
            for (int t = chunk_len; t < CHUNK_BT; ++t) sh_G[t] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── 4. K@K^T → sh_attn,  Q@K^T → sh_qkdm  (8x8 MMA). ──
        // 2x2 tiles per 16x16 output, 8 tiles across both outputs; SGs
        // round-robin over tiles, K-loop over HEAD_K_DIM in 8-wide steps.
        {
            const int kk_tiles = (CHUNK_BT / 8) * (CHUNK_BT / 8); // 4
            for (int t = sgitg; t < 2 * kk_tiles; t += SIMDS_PER_TG) {
                const bool is_qk = t >= kk_tiles;
                const int tt = is_qk ? t - kk_tiles : t;
                const int r8 = (tt / (CHUNK_BT / 8)) * 8;
                const int c8 = (tt % (CHUNK_BT / 8)) * 8;
                simdgroup_float8x8 mC = make_filled_simdgroup_matrix<float, 8>(0.0f);
                for (int kk = 0; kk < HEAD_K_DIM; kk += 8) {
                    simdgroup_float8x8 mA;
                    simdgroup_float8x8 mB;
                    if (is_qk) {
                        simdgroup_load(mA, &sh_q[r8][kk], HEAD_K_DIM);
                    } else {
                        simdgroup_load(mA, &sh_k[r8][kk], HEAD_K_DIM);
                    }
                    simdgroup_load(mB, &sh_k[c8][kk], HEAD_K_DIM, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(mC, mA, mB, mC);
                }
                if (is_qk) {
                    simdgroup_store(mC, &sh_qkdm[r8][c8], CHUNK_BT);
                } else {
                    simdgroup_store(mC, &sh_attn[r8][c8], CHUNK_BT);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── 5. Apply mask + decay + β. Also compute v_beta in sh_v (FUSED). ──
        for (int p = tid; p < CHUNK_BT * CHUNK_BT; p += TG_THREADS) {
            int i = p / CHUNK_BT, jj = p % CHUNK_BT;
            float aval = 0.0f, qval = 0.0f;
            if (i < chunk_len && jj < chunk_len) {
                float decay = exp(sh_G[i] - sh_G[jj]);
                if (i > jj)  aval = -sh_beta[i] * sh_attn[i][jj] * decay;
                if (i >= jj) qval = sh_qkdm[i][jj] * decay;
            }
            sh_attn[i][jj] = aval;
            sh_qkdm[i][jj] = qval;
        }
        for (int p = tid; p < chunk_len * DV_BLOCK; p += TG_THREADS) {
            int dt = p / DV_BLOCK, dvi = p % DV_BLOCK;
            sh_v[dt][dvi] = sh_beta[dt] * sh_v[dt][dvi];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── 6. (I - strict_lower(sh_attn))^{-1} via forward sub (SERIAL, SG 0). ──
        if (sgitg == 0) {
            for (int i = 1; i < chunk_len; ++i) {
                int kcol = (int)lane;
                float new_val = 0.0f;
                if (kcol < i) {
                    float sum = 0.0f;
                    for (int s = kcol + 1; s < i; ++s) sum += sh_attn[i][s] * sh_attn[s][kcol];
                    new_val = sh_attn[i][kcol] + sum;
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                if (kcol < i) sh_attn[i][kcol] = new_val;
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < chunk_len) sh_attn[tid][tid] = 1.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── 7. Combined stage:
        //   * K_S = K @ state              (8x8 MMA)  → sh_K_S
        //   * v_nb = attn @ v_beta         (scalar)   → sh_out (temp)
        //   Disjoint TG-mem regions — one barrier at the end. ──
        {
            const int ks_tiles = (CHUNK_BT / 8) * (DV_BLOCK / 8); // 4
            for (int t = sgitg; t < ks_tiles; t += SIMDS_PER_TG) {
                const int r8 = (t / (DV_BLOCK / 8)) * 8;
                const int c8 = (t % (DV_BLOCK / 8)) * 8;
                simdgroup_float8x8 mC = make_filled_simdgroup_matrix<float, 8>(0.0f);
                for (int kk = 0; kk < HEAD_K_DIM; kk += 8) {
                    simdgroup_float8x8 mA;
                    simdgroup_float8x8 mB;
                    simdgroup_load(mA, &sh_k[r8][kk], HEAD_K_DIM);
                    simdgroup_load(mB, &sh_state[kk][c8], DV_BLOCK);
                    simdgroup_multiply_accumulate(mC, mA, mB, mC);
                }
                simdgroup_store(mC, &sh_K_S[r8][c8], DV_BLOCK);
            }
        }
        for (int p = tid; p < chunk_len * DV_BLOCK; p += TG_THREADS) {
            int t = p / DV_BLOCK, dvi = p % DV_BLOCK;
            float acc = 0.0f;
            for (int s = 0; s <= t; ++s) acc += sh_attn[t][s] * sh_v[s][dvi];
            sh_out[t][dvi] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── 8. Combined stage (same as flash_chunk):
        //   8a. attn_scaled = attn * (β · expG) in-place; 8b. Q *= exp(G[t]);
        //   8c. v_new = v_nb - attn_scaled @ K_S → sh_v. ──
        for (int p = tid; p < CHUNK_BT * CHUNK_BT; p += TG_THREADS) {
            int t = p / CHUNK_BT, s = p % CHUNK_BT;
            sh_attn[t][s] = sh_attn[t][s] * sh_beta[s] * exp(sh_G[s]);
        }
        for (int p = tid; p < chunk_len * d_k; p += TG_THREADS) {
            int dt = p / d_k, di = p % d_k;
            sh_q[dt][di] = sh_q[dt][di] * exp(sh_G[dt]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int p = tid; p < chunk_len * DV_BLOCK; p += TG_THREADS) {
            int t = p / DV_BLOCK, dvi = p % DV_BLOCK;
            float acc = 0.0f;
            for (int s = 0; s < chunk_len; ++s) acc += sh_attn[t][s] * sh_K_S[s][dvi];
            sh_v[t][dvi] = sh_out[t][dvi] - acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── 9. attn_inter = (Q · expG) @ state (8x8 MMA, biggest). ──
        {
            const int qs_tiles = (CHUNK_BT / 8) * (DV_BLOCK / 8); // 4
            for (int t = sgitg; t < qs_tiles; t += SIMDS_PER_TG) {
                const int r8 = (t / (DV_BLOCK / 8)) * 8;
                const int c8 = (t % (DV_BLOCK / 8)) * 8;
                simdgroup_float8x8 mC = make_filled_simdgroup_matrix<float, 8>(0.0f);
                for (int kk = 0; kk < HEAD_K_DIM; kk += 8) {
                    simdgroup_float8x8 mA;
                    simdgroup_float8x8 mB;
                    simdgroup_load(mA, &sh_q[r8][kk], HEAD_K_DIM);
                    simdgroup_load(mB, &sh_state[kk][c8], DV_BLOCK);
                    simdgroup_multiply_accumulate(mC, mA, mB, mC);
                }
                simdgroup_store(mC, &sh_out[r8][c8], DV_BLOCK);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── 10. out += attn_intra @ v_new (scalar, small); write to attn_out. ──
        for (int p = tid; p < chunk_len * DV_BLOCK; p += TG_THREADS) {
            int t = p / DV_BLOCK, dvi = p % DV_BLOCK;
            float acc = sh_out[t][dvi];
            for (int s = 0; s <= t; ++s) acc += sh_qkdm[t][s] * sh_v[s][dvi];
            attn_out[output_offset_v2(b, t0 + t, h, dv_off + dvi, param)] = (ftype)acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── 11. Prepare state update: K_dec = K * exp(G_last - G[t]) in-place;
        //        sh_state *= exp(G_last). ──
        float G_last = sh_G[chunk_len - 1];
        float decay_total = exp(G_last);
        for (int p = tid; p < chunk_len * d_k; p += TG_THREADS) {
            int dt = p / d_k, di = p % d_k;
            sh_k[dt][di] = sh_k[dt][di] * exp(G_last - sh_G[dt]);
        }
        for (int p = tid; p < HEAD_K_DIM * DV_BLOCK; p += TG_THREADS) {
            int dki = p / DV_BLOCK, dvi = p % DV_BLOCK;
            sh_state[dki][dvi] *= decay_total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── 12. State update: sh_state += K_dec^T @ v_new (8x8 MMA,
        //        multiply-accumulate seeded from the decayed state). ──
        {
            const int su_tiles = (HEAD_K_DIM / 8) * (DV_BLOCK / 8);
            for (int t = sgitg; t < su_tiles; t += SIMDS_PER_TG) {
                const int r8 = (t / (DV_BLOCK / 8)) * 8;
                const int c8 = (t % (DV_BLOCK / 8)) * 8;
                simdgroup_float8x8 mC;
                simdgroup_load(mC, &sh_state[r8][c8], DV_BLOCK);
                for (int kk = 0; kk < CHUNK_BT; kk += 8) {
                    simdgroup_float8x8 mA;
                    simdgroup_float8x8 mB;
                    simdgroup_load(mA, &sh_k[kk][r8], HEAD_K_DIM, ulong2(0, 0), true);
                    simdgroup_load(mB, &sh_v[kk][c8], DV_BLOCK);
                    simdgroup_multiply_accumulate(mC, mA, mB, mC);
                }
                simdgroup_store(mC, &sh_state[r8][c8], DV_BLOCK);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Flush state to device once at kernel exit. ──
    for (int p = tid; p < HEAD_K_DIM * DV_BLOCK; p += TG_THREADS) {
        int dki = p / DV_BLOCK, dvi = p % DV_BLOCK;
        state_gm[(dv_off + dvi) * d_k + dki] = (ftype)sh_state[dki][dvi];
    }
}
)metal";

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif /* MNN_METAL_ENABLED */


