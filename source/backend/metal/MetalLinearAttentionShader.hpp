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
    float q_scale;
};

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
            input_val = (float)qkv[b * D * L + d * L + (pos - css)];
        }
        sum += input_val * (float)conv_weight[d * K + k];
    }

    // SiLU activation: x * sigmoid(x)
    float sigmoid_val = 1.0f / (1.0f + exp(-sum));
    conv_out[b * D * L + d * L + l] = (ftype)(sum * sigmoid_val);
}

// Kernel 2: Update conv state with last (K-1) elements of padded input
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

    const int total = B * D * css;
    if ((int)gid >= total) return;

    const int i = gid % css;
    const int bd = gid / css;
    const int b = bd / D;
    const int d = bd % D;

    // new_state[i] = padded[L + i], padded = cat(old_state[css], qkv[L])
    // position in padded = L + i
    // Since L + i >= css (because L >= 1 and i >= 0, and css = K-1, and L+i = L+i),
    // we need: if (L + i) < css -> old_state, else -> qkv[(L+i) - css]
    int pos = L + i;
    ftype val;
    if (pos < css) {
        val = conv_state[b * D * css + d * css + pos];
    } else {
        val = qkv[b * D * L + d * L + (pos - css)];
    }
    // Write to conv_state - note: we need to be careful about reading and writing
    // conv_state simultaneously. Since we write to position i and read from position (L+i),
    // and L >= 1, so (L+i) > i always -> no read-write conflict.
    conv_state[b * D * css + d * css + i] = val;
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
    float q_scale;
};

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

        float g_t    = (float)gate[b * L * H + t * H + h];
        float beta_t = (float)beta[b * L * H + t * H + h];

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

        attn_out[(b * L * H + t * H + h) * d_v + j] = (ftype)o_t_j;
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
    float q_scale;
};

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

    for (int t = 0; t < L; ++t) {
        const int bth = b * L * H + t * H + h;
        const device ftype* q_t = q + bth * d_k;
        const device ftype* k_t = k + bth * d_k;
        float v_t_j = (float)v[bth * d_v + j];
        float decay_val = exp((float)gate[bth]);
        float beta_t = (float)beta[bth];

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
                float s_val = (float)state[i] * decay_val;
                state[i] = (ftype)s_val;
                v_pred_j += s_val * k_reg[ii];
            }
        }
        v_pred_j = simd_sum(v_pred_j);
        float delta_j = beta_t * (v_t_j - v_pred_j);

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
            attn_out[bth * d_v + j] = (ftype)o_t_j;
        }
    }
}
)metal";

// Fused QKV-prep + Gated Delta Rule (for decode, reads directly from conv_out)
// Eliminates intermediate Q, K, V buffers and the qkv_prep kernel launch
// Best for decode (L=1) where conv_out stride = 1 (coalesced)
// Each simdgroup (32 threads) handles one (batch, head, j) element
// State layout: [B, H, d_v, d_k] for coalesced simd access
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
    float q_scale;
};

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
        float decay_val = exp((float)gate[bth]);
        float beta_t = (float)beta[bth];

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
            attn_out[bth * d_v + j] = (ftype)o_t_j;
        }
    }
}
)metal";

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif /* MNN_METAL_ENABLED */
