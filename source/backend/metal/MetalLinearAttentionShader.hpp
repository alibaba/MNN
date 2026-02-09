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

// Kernel 3: Gated Delta Rule (Steps 2-5 fused)
// Each thread processes one (batch, head) pair across all timesteps
// Inputs:
//   conv_out [B, D, L] - output of conv+silu
//   gate [B, L, H] - log-space decay
//   beta [B, L, H] - learning rate
//   recurrent_state [B, H, d_k, d_v] - persistent state S
// Output: attn_out [B, L, H_v, d_v]
kernel void linear_attn_gated_delta_rule(
    const device ftype* conv_out         [[buffer(0)]],
    const device ftype* gate             [[buffer(1)]],
    const device ftype* beta             [[buffer(2)]],
    device ftype* recurrent_state        [[buffer(3)]],
    device ftype* attn_out               [[buffer(4)]],
    constant LinearAttnParam& param      [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    const int B = param.batch;
    const int D = param.conv_dim;
    const int L = param.seq_len;
    const int H_k = param.num_k_heads;
    const int H = param.num_v_heads;
    const int d_k = param.head_k_dim;
    const int d_v = param.head_v_dim;
    const int key_dim = param.key_dim;
    const int gqa_factor = param.gqa_factor;
    const int use_l2norm = param.use_l2norm;
    const float q_scale = param.q_scale;

    const int total_heads = B * H;
    if ((int)gid >= total_heads) return;

    const int b = gid / H;
    const int h = gid % H;          // V-head index
    const int k_head = h / gqa_factor; // GQA: corresponding K-head

    // State pointer: [B, H, d_k, d_v]
    device ftype* state = recurrent_state + (b * H + h) * d_k * d_v;

    // Process each timestep sequentially
    for (int t = 0; t < L; ++t) {
        // Step 2: Extract q_t, k_t, v_t from conv_out (transpose on the fly)
        // conv_out layout: [B, D, L], access: conv_out[b*D*L + channel*L + t]
        float q_local[256]; // max d_k
        float k_local[256];
        float v_local[256]; // max d_v

        const device ftype* conv_base = conv_out + b * D * L;

        for (int i = 0; i < d_k; ++i) {
            q_local[i] = (float)conv_base[(k_head * d_k + i) * L + t];
            k_local[i] = (float)conv_base[(key_dim + k_head * d_k + i) * L + t];
        }
        for (int i = 0; i < d_v; ++i) {
            v_local[i] = (float)conv_base[(2 * key_dim + h * d_v + i) * L + t];
        }

        // Step 3: Optional L2 Normalization on q_t and k_t
        if (use_l2norm) {
            const float eps = 1e-6f;
            float sumSq = 0.0f;
            for (int i = 0; i < d_k; ++i) sumSq += q_local[i] * q_local[i];
            float invNorm = 1.0f / sqrt(sumSq + eps);
            for (int i = 0; i < d_k; ++i) q_local[i] *= invNorm;

            sumSq = 0.0f;
            for (int i = 0; i < d_k; ++i) sumSq += k_local[i] * k_local[i];
            invNorm = 1.0f / sqrt(sumSq + eps);
            for (int i = 0; i < d_k; ++i) k_local[i] *= invNorm;
        }

        // Step 4: Scale q_t by 1/sqrt(d_k)
        for (int i = 0; i < d_k; ++i) q_local[i] *= q_scale;

        // Step 5: Gated Delta Rule recurrence
        float g_t    = (float)gate[b * L * H + t * H + h];
        float beta_t = (float)beta[b * L * H + t * H + h];

        // 5.1 Decay: S = S * exp(g_t)
        float decay_val = exp(g_t);
        for (int i = 0; i < d_k * d_v; ++i) {
            state[i] = (ftype)((float)state[i] * decay_val);
        }

        // 5.2 Read: v_pred = S^T @ k_t
        float v_pred[256]; // max d_v
        for (int j = 0; j < d_v; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < d_k; ++i) {
                sum += (float)state[i * d_v + j] * k_local[i];
            }
            v_pred[j] = sum;
        }

        // 5.3 Delta: delta = beta_t * (v_t - v_pred)
        float delta[256]; // max d_v
        for (int j = 0; j < d_v; ++j) {
            delta[j] = beta_t * (v_local[j] - v_pred[j]);
        }

        // 5.4 Write: S += k_t @ delta^T (outer product)
        for (int i = 0; i < d_k; ++i) {
            float k_val = k_local[i];
            for (int j = 0; j < d_v; ++j) {
                state[i * d_v + j] = (ftype)((float)state[i * d_v + j] + k_val * delta[j]);
            }
        }

        // 5.5 Query: o_t = S^T @ q_t
        device ftype* o_t = attn_out + (b * L + t) * H * d_v + h * d_v;
        for (int j = 0; j < d_v; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < d_k; ++i) {
                sum += (float)state[i * d_v + j] * q_local[i];
            }
            o_t[j] = (ftype)sum;
        }
    } // end timestep
}
)metal";

#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif /* MNN_METAL_ENABLED */
