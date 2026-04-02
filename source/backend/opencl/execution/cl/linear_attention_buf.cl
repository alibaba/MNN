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
    __private const int d_k,
    __private const int d_v,
    __private const int key_dim,
    __private const int val_dim,
    __private const int gqa_factor,
    __private const float q_scale)
{
    const int dv4 = (d_v + 3) / 4;
    const int total = dv4 * num_v_heads * batch;
    const int gid = get_global_id(1);
    if (gid >= total) return;
    const int x = (gid % dv4) << 2;
    const int bh = gid / dv4;
    const int h = bh % num_v_heads;
    const int b = bh / num_v_heads;
    const int k_head = h / gqa_factor; // GQA: corresponding K-head
    __local float4 __sum[LOCAL_SIZE];
    __local float4 rec_local[K_SIZE];
    __local float key_local[K_SIZE];
    const int lid = get_local_id(0);

    // State pointer: recurrent_state layout [B, H, d_k, d_v]
    const int state_offset = (b * num_v_heads + h) * K_SIZE * d_v + x;
    #ifdef DECODE_PHASE
    const int conv_base = b * conv_dim;
    float g_t    = (float)(gate[b * num_v_heads + h]);
    float4 beta_t = (float4)(beta_in[b * num_v_heads + h]);
    float4 decay_val = (float4)(exp(g_t));
    const int out_offset = b * num_v_heads * d_v + h * d_v;
    float4 s = (float4)(0);
    for (int i = lid; i < K_SIZE; i+=LOCAL_SIZE) {
        float4 rec = convert_float4(vload4(0, recurrent_state + state_offset + i * d_v))* decay_val;
        float key = (float)(conv_out[conv_base + key_dim + k_head * K_SIZE + i]);
        s += rec * (float4)(key);
        rec_local[i] = rec;
        key_local[i] = key;
    }
    __sum[lid] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            __sum[lid] += __sum[lid + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    s = __sum[0];
    float4 v_data = convert_float4(vload4(0, conv_out + conv_base + 2 * key_dim + h * d_v + x));
    float4 delta = beta_t * (v_data - s);
    s = (float4)(0);
    for (int i = lid; i < K_SIZE; i+=LOCAL_SIZE) {
        float4 recurrent_state_data = rec_local[i]  + (float4)(key_local[i]) * delta;
        s += recurrent_state_data * (float4)(conv_out[conv_base + (k_head * K_SIZE + i)]) * (float4)(q_scale);
        vstore4(CONVERT_FLOAT4(recurrent_state_data), 0, recurrent_state + state_offset + i * d_v);
    }
    __sum[lid] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            __sum[lid] += __sum[lid + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    s = __sum[0];
    if(lid == 0){
        vstore4(CONVERT_FLOAT4(s), 0, attn_out + out_offset + x);
    }
    #else
    for (int i = lid; i < K_SIZE; i+=LOCAL_SIZE) {
        rec_local[i] = convert_float4(vload4(0, recurrent_state + state_offset + i * d_v));
    }
    for (int t = 0; t < seq_len; ++t) {
        const int conv_base = b * seq_len * conv_dim;
        float g_t    = (float)(gate[b * seq_len * num_v_heads + t * num_v_heads + h]);
        float4 beta_t = (float4)(beta_in[b * seq_len * num_v_heads + t * num_v_heads + h]);
        float4 decay_val = (float4)(exp(g_t));
        const int out_offset = (b * seq_len + t) * num_v_heads * d_v + h * d_v;
        float4 s = (float4)(0);
        for (int i = lid; i < K_SIZE; i+=LOCAL_SIZE) {
            float4 rec = rec_local[i] * decay_val;
            float key = (float)(conv_out[conv_base + (key_dim + k_head * K_SIZE + i) * seq_len + t]);
            s += rec * (float4)(key);
            key_local[i] = key;
        }
        __sum[lid] = s;
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                __sum[lid] += __sum[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        s = __sum[0];
        float4 v_data;
        v_data.x = (float)(conv_out[conv_base + (2 * key_dim + h * d_v + x) * seq_len + t]);
        v_data.y = (float)(conv_out[conv_base + (2 * key_dim + h * d_v + x + 1) * seq_len + t]);
        v_data.z = (float)(conv_out[conv_base + (2 * key_dim + h * d_v + x + 2) * seq_len + t]);
        v_data.w = (float)(conv_out[conv_base + (2 * key_dim + h * d_v + x + 3) * seq_len + t]);
        float4 delta = beta_t * (v_data - s);
        s = (float4)(0);
        for (int i = lid; i < K_SIZE; i+=LOCAL_SIZE) {
            float4 recurrent_state_data = rec_local[i] * decay_val + (float4)(key_local[i]) * delta;
            s += recurrent_state_data * (float4)(conv_out[conv_base + (k_head * K_SIZE + i) * seq_len + t]) * (float4)(q_scale);
            rec_local[i] = recurrent_state_data;
        }
        __sum[lid] = s;
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                __sum[lid] += __sum[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        s = __sum[0];
        if(lid == 0){
            vstore4(CONVERT_FLOAT4(s), 0, attn_out + out_offset + x);
        }
    }
    for (int i = lid; i < K_SIZE; i+=LOCAL_SIZE) {
        vstore4(CONVERT_FLOAT4(rec_local[i]), 0, recurrent_state + state_offset + i * d_v);
    }
    #endif
}

__kernel void l2_norm(__global const FLOAT* input,
                             __global FLOAT* output,
                             __private const int conv_dim,
                             __private const int head_k_dim,
                             __private const int gqa_factor,
                             __private const int key_dim,
                             __private const int seq_len)
{
    #ifdef USE_VEC
    const int hl = get_global_id(1);
    const int bk = get_global_id(2);
    const int seq_len4 = (seq_len + 3) / 4;
    const int h = hl / seq_len4;
    const int sq = hl % seq_len4;
    const int b = bk / 2;
    const int k = bk % 2;
    const int lid = get_local_id(0);
    const int k_head = h / gqa_factor; // GQA: corresponding K-head
    const int input_offset = (b * conv_dim + k * key_dim + k_head * head_k_dim) * seq_len + (sq << 2);
    __local float4 __sum[K_SIZE];
    float4 sum = 0.0f;
    for(int i = lid; i < K_SIZE; i += K_SIZE){
        float4 in = convert_float4(vload4(0, input + input_offset + i * seq_len));
        sum += in * in;
    }
    __sum[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = K_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            __sum[lid] += __sum[lid + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sum = __sum[0];
    int remain = seq_len - (sq << 2);
    float4 invNorm = (float4)1.0f / sqrt(sum + (float4)1e-6f);
    if(remain == 1){
        for(int i = lid; i < K_SIZE; i += K_SIZE){
            float4 out = convert_float4(vload4(0, input + input_offset + i * seq_len)) * invNorm;
            output[input_offset + i * seq_len] = (FLOAT)out.s0;
        }
    }else if(remain == 2){
        for(int i = lid; i < K_SIZE; i += K_SIZE){
            float4 out = convert_float4(vload4(0, input + input_offset + i * seq_len)) * invNorm;
            vstore2(CONVERT_FLOAT2(out.s01), 0, output + input_offset + i * seq_len);
        }
    }else if(remain == 3){
        for(int i = lid; i < K_SIZE; i += K_SIZE){
            float4 out = convert_float4(vload4(0, input + input_offset + i * seq_len)) * invNorm;
            vstore3(CONVERT_FLOAT3(out.s012), 0, output + input_offset + i * seq_len);
        }
    }else{
        for(int i = lid; i < K_SIZE; i += K_SIZE){
            float4 out = convert_float4(vload4(0, input + input_offset + i * seq_len)) * invNorm;
            vstore4(CONVERT_FLOAT4(out), 0, output + input_offset + i * seq_len);
        }
    }
    #else
    const int h = get_global_id(1);
    const int bk = get_global_id(2);
    const int b = bk / 2;
    const int k = bk % 2;
    const int lid = get_local_id(0);
    const int k_head = h / gqa_factor; // GQA: corresponding K-head
    const int input_offset = b * conv_dim + k * key_dim + k_head * head_k_dim;
    __local float __sum[K_SIZE];
    float sum = 0.0f;
    for(int i = lid; i < K_SIZE; i += K_SIZE){
        float in = (float)(input[input_offset + i]);
        sum += in * in;
    }
    __sum[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = K_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            __sum[lid] += __sum[lid + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sum = __sum[0];
    float invNorm = 1.0f / sqrt(sum + 1e-6f);
    for(int i = lid; i < K_SIZE; i += K_SIZE){
        float out = (float)(input[input_offset + i]) * invNorm;
        output[input_offset + i] = (FLOAT)out;
    }
    #endif
}