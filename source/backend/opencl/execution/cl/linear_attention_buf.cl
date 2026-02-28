#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

// Kernel 1: Depthwise Conv1D + SiLU
// Each work-item processes one (batch*channel, seq_pos) element.
// Input:  qkv [B, D, L], conv_state [B, D, conv_state_size], conv_weight [D, 1, K]
// Output: conv_out [B, D, L]
// conv_state is read but NOT updated here (updated by separate kernel).
__kernel void linear_attn_conv_silu(
    __private const int global_dim0,
    __global const FLOAT* qkv,
    __global const FLOAT* conv_state,
    __global const FLOAT* conv_weight,
    __global FLOAT* conv_out,
    __private const int batch,
    __private const int conv_dim,
    __private const int seq_len,
    __private const int kernel_size,
    __private const int conv_state_size)
{
    const int gid = get_global_id(0);
    if (gid >= global_dim0) return;

    const int L = seq_len;
    const int D = conv_dim;
    const int K = kernel_size;
    const int css = conv_state_size;

    // Decompose: gid -> (batch_chan, seq_pos)
    const int l = gid % L;
    const int bd = gid / L;
    const int b = bd / D;
    const int d = bd % D;

    // Compute valid convolution for position l
    // Padded input = [conv_state[b,d,:], qkv[b,d,:]]
    // padded[pos]: if pos < css -> conv_state[b*D*css + d*css + pos]
    //              else -> qkv[b*D*L + d*L + (pos - css)]
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        int pos = l + k;
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
    conv_out[b * D * L + d * L + l] = (FLOAT)(sum * sigmoid_val);
}

// Kernel 2: Update conv state with last (K-1) elements of padded input
// new_state[i] = padded[L + i], where padded = cat(old_state[css], qkv[L])
// Must execute AFTER linear_attn_conv_silu (which reads conv_state).
__kernel void linear_attn_conv_state_update(
    __private const int global_dim0,
    __global const FLOAT* qkv,
    __global FLOAT* conv_state,
    __private const int batch,
    __private const int conv_dim,
    __private const int seq_len,
    __private const int conv_state_size)
{
    const int gid = get_global_id(0);
    if (gid >= global_dim0) return;

    const int L = seq_len;
    const int D = conv_dim;
    const int css = conv_state_size;

    const int i = gid % css;
    const int bd = gid / css;
    const int b = bd / D;
    const int d = bd % D;

    // new_state[i] = padded[L + i]
    // padded = cat(old_state[css], qkv[L])
    // position (L + i) in padded: if (L+i) < css -> old_state, else -> qkv[(L+i) - css]
    int pos = L + i;
    FLOAT val;
    if (pos < css) {
        val = conv_state[b * D * css + d * css + pos];
    } else {
        val = qkv[b * D * L + d * L + (pos - css)];
    }
    // Safe: we write to index i, read from index (L+i) where L >= 1, so (L+i) > i always
    conv_state[b * D * css + d * css + i] = val;
}

// Kernel 3: Gated Delta Rule (Steps 2-5 fused)
// Each work-item processes one (batch, head) pair across all timesteps.
// Uses float32 exclusively for numerical stability of the recurrence.
__kernel void linear_attn_gated_delta_rule(
    __private const int global_dim0,
    __global const FLOAT* conv_out,
    __global const FLOAT* gate,
    __global const FLOAT* beta_in,
    __global FLOAT* recurrent_state,
    __global FLOAT* attn_out,
    __private const int batch,
    __private const int conv_dim,
    __private const int seq_len,
    __private const int num_k_heads,
    __private const int num_v_heads,
    __private const int head_k_dim,
    __private const int head_v_dim,
    __private const int key_dim,
    __private const int val_dim,
    __private const int gqa_factor,
    __private const int use_l2norm,
    __private const float q_scale)
{
    const int gid = get_global_id(0);
    if (gid >= global_dim0) return;

    const int D = conv_dim;
    const int L = seq_len;
    const int H = num_v_heads;
    const int d_k = head_k_dim;
    const int d_v = head_v_dim;

    const int b = gid / H;
    const int h = gid % H;           // V-head index
    const int k_head = h / gqa_factor; // GQA: corresponding K-head

    // State pointer: recurrent_state layout [B, H, d_k, d_v]
    const int state_offset = (b * H + h) * d_k * d_v;

    // Process each timestep sequentially
    for (int t = 0; t < L; ++t) {
        // Step 2: Extract q_t, k_t, v_t from conv_out (transpose on the fly)
        // conv_out layout: [B, D, L], access: conv_out[b*D*L + channel*L + t]
        float q_local[256]; // max d_k
        float k_local[256];
        float v_local[256]; // max d_v

        const int conv_base = b * D * L;

        for (int i = 0; i < d_k; ++i) {
            q_local[i] = (float)conv_out[conv_base + (k_head * d_k + i) * L + t];
            k_local[i] = (float)conv_out[conv_base + (key_dim + k_head * d_k + i) * L + t];
        }
        for (int i = 0; i < d_v; ++i) {
            v_local[i] = (float)conv_out[conv_base + (2 * key_dim + h * d_v + i) * L + t];
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
        float beta_t = (float)beta_in[b * L * H + t * H + h];

        // 5.1 Decay: S = S * exp(g_t)
        float decay_val = exp(g_t);
        for (int i = 0; i < d_k * d_v; ++i) {
            recurrent_state[state_offset + i] = (FLOAT)((float)recurrent_state[state_offset + i] * decay_val);
        }

        // 5.2 Read: v_pred = S^T @ k_t
        float v_pred[256]; // max d_v
        for (int j = 0; j < d_v; ++j) {
            float s = 0.0f;
            for (int i = 0; i < d_k; ++i) {
                s += (float)recurrent_state[state_offset + i * d_v + j] * k_local[i];
            }
            v_pred[j] = s;
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
                recurrent_state[state_offset + i * d_v + j] = (FLOAT)((float)recurrent_state[state_offset + i * d_v + j] + k_val * delta[j]);
            }
        }

        // 5.5 Query: o_t = S^T @ q_t
        const int out_offset = (b * L + t) * H * d_v + h * d_v;
        for (int j = 0; j < d_v; ++j) {
            float s = 0.0f;
            for (int i = 0; i < d_k; ++i) {
                s += (float)recurrent_state[state_offset + i * d_v + j] * q_local[i];
            }
            attn_out[out_offset + j] = (FLOAT)s;
        }
    } // end timestep
}
