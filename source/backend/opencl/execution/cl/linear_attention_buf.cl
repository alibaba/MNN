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

#ifdef CHUNK_PREFILL

// ======================== Chunked Prefill Kernels ========================
// Implements torch_chunk_gated_delta_rule from modeling_qwen3_5.py
// Decomposes sequential recurrence into chunk-parallel operations.
// CHUNK_SIZE must be defined at compile time (e.g. -DCHUNK_SIZE=64).

// Kernel C1: Cumulative sum of gate values within each chunk
// GWS: {num_v_heads, num_chunks, batch}
__kernel void chunk_g_cumsum(
    __global const FLOAT* gate,        // [B, L, H]
    __global float* g_cumsum,           // [B, H, num_chunks, CHUNK_SIZE] (float32 for precision)
    __private const int num_v_heads,
    __private const int seq_len,
    __private const int num_chunks)
{
    const int h = get_global_id(0);
    const int c = get_global_id(1);
    const int b = get_global_id(2);
    if (h >= num_v_heads || c >= num_chunks) return;

    const int H = num_v_heads;
    const int L = seq_len;
    const int C = CHUNK_SIZE;

    float cumsum = 0.0f;
    int out_base = ((b * H + h) * num_chunks + c) * C;
    for (int p = 0; p < C; ++p) {
        int l = c * C + p;
        float g_val = (l < L) ? (float)gate[b * L * H + l * H + h] : 0.0f;
        cumsum += g_val;
        g_cumsum[out_base + p] = cumsum;
    }
}

// Kernel C2: Build Neumann-corrected intra-chunk attention matrix
// attn = I + neumann_correction(-(k_beta @ k^T) * decay_mask)
// GWS: {CHUNK_SIZE, num_v_heads * num_chunks, batch}, LWS: {CHUNK_SIZE, 1, 1}

__kernel void chunk_build_neumann_attn_step0(
    __global const FLOAT* conv_out,    // [B, D, L]
    __global const FLOAT* beta_in,     // [B, L, H]
    __global const float* g_cumsum,    // [B, H, num_chunks, C] (float32)
    __global float* attn_matrix,       // [B, H, num_chunks, C, C] (float32)
    __private const int batch,
    __private const int conv_dim,
    __private const int seq_len,
    __private const int num_v_heads,
    __private const int head_k_dim,
    __private const int key_dim,
    __private const int gqa_factor,
    __private const int num_chunks)
{
    const int rc = get_global_id(0);
    const int hc = get_global_id(1);
    const int b = get_global_id(2);
    if (hc >= num_v_heads * num_chunks || b >= batch || rc >= CHUNK_SIZE * CHUNK_SIZE) return;

    const int col = rc % CHUNK_SIZE;
    const int row = rc / CHUNK_SIZE;
    const int h = hc / num_chunks;
    const int c = hc % num_chunks;


    const int C = CHUNK_SIZE;
    const int dk = head_k_dim;
    const int D = conv_dim;
    const int L = seq_len;
    const int H = num_v_heads;
    const int k_head = h / gqa_factor;

    const int g_base = ((b * H + h) * num_chunks + c) * C;
    const int attn_out_base = ((b * H + h) * num_chunks + c) * C * C;
    const int conv_base = b * D * L;
    
    // Phase 1: Compute strictly-lower-triangular attn
    // attn[i, j] = -(k[i]*beta[i]) dot k[j] * exp(g[i]-g[j])
    float val = 0.0f;
    if (col < row) {
        const int l_i = c * C + row;
        const int l_j = c * C + col;
        if (l_i < L && l_j < L) {
            const float g_i = (float)g_cumsum[g_base + row];
            const float beta_i = (float)beta_in[b * L * H + l_i * H + h];
            const float g_j = (float)g_cumsum[g_base + col];
            const float decay = exp(g_i - g_j);
            float dot = 0.0f;
            for (int d = 0; d < dk; ++d) {
                float k_i_d = (float)conv_out[conv_base + (key_dim + k_head * dk + d) * L + l_i];
                float k_j_d = (float)conv_out[conv_base + (key_dim + k_head * dk + d) * L + l_j];
                dot += k_i_d * k_j_d;
            }
            val = -(beta_i * dot) * decay;
        }
    }
    attn_matrix[attn_out_base + row * C + col] = val;
}

__kernel void chunk_build_neumann_attn_step1(
    __global float* attn_matrix,       // [B, H, num_chunks, C, C] (float32)
    __private const int batch,
    __private const int conv_dim,
    __private const int seq_len,
    __private const int num_v_heads,
    __private const int head_k_dim,
    __private const int key_dim,
    __private const int gqa_factor,
    __private const int num_chunks)
{
    const int col = get_global_id(0);
    const int hc = get_global_id(1);
    const int b = get_global_id(2);
    if (hc >= num_v_heads * num_chunks || b >= batch || col >= CHUNK_SIZE) return;

    const int lid = get_local_id(0);
    const int h = hc / num_chunks;
    const int c = hc % num_chunks;


    const int C = CHUNK_SIZE;
    const int dk = head_k_dim;
    const int D = conv_dim;
    const int L = seq_len;
    const int H = num_v_heads;
    const int k_head = h / gqa_factor;

    const int attn_out_base = ((b * H + h) * num_chunks + c) * C * C;
    
    // Phase 2: Neumann series correction (row by row)
    for (int r = 1; r < C; ++r) {
        float orig = attn_matrix[attn_out_base + r * C + lid];
        float correction = 0.0f;
        if (lid < r) {
            for (int k = 0; k < r; ++k) {
                correction += attn_matrix[attn_out_base + r * C + k] * attn_matrix[attn_out_base + k * C + lid];
            }
        }
        if (lid < r) {
            attn_matrix[attn_out_base + r * C + lid] = orig + correction;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Phase 3: Add identity and write to global (float32)
    attn_matrix[attn_out_base + lid * C + lid] = attn_matrix[attn_out_base + lid * C + lid] + 1.0f;
}

// Kernel C3: v_corrected = attn_matrix @ (V * beta)
// GWS: {UP_DIV(dv, 4), CHUNK_SIZE * num_chunks, B * H}
__kernel void chunk_correct_v(
    __global const float* attn_matrix,  // [B, H, num_chunks, C, C] (float32)
    __global const FLOAT* conv_out,     // [B, D, L]
    __global const FLOAT* beta_in,      // [B, L, H]
    __global const float* g_cumsum,     // [B, H, num_chunks, C] (float32)
    __global float* v_corrected,        // [B, H, num_chunks, C, dv] (float32)
    __global float* k_cumdecay,         // [B, H, num_chunks, C, dk] (float32)
    __private const int global_dim0,
    __private const int global_dim1,
    __private const int global_dim2,
    __private const int conv_dim,
    __private const int seq_len,
    __private const int num_v_heads,
    __private const int head_k_dim,
    __private const int head_v_dim,
    __private const int key_dim,
    __private const int num_chunks)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    if (x >= global_dim0 || y >= global_dim1 || z >= global_dim2) return;

    const int j = x << 2;
    const int c = y / CHUNK_SIZE;
    const int row = y % CHUNK_SIZE;
    const int b = z / num_v_heads;
    const int h = z % num_v_heads;

    const int C = CHUNK_SIZE;
    const int dk = head_k_dim;
    const int dv = head_v_dim;
    const int D = conv_dim;
    const int L = seq_len;
    const int H = num_v_heads;

    const int attn_base = ((b * H + h) * num_chunks + c) * C * C + row * C;
    const int g_base = ((b * H + h) * num_chunks + c) * C;
    const int conv_base = b * D * L;

    float4 result0 = (float4)(0.0f);
    float4 result1 = (float4)(0.0f);
    int l_p = c * C;
    for (int p = 0; p < C && l_p < L; ++p, ++l_p) {
        float a = attn_matrix[attn_base + p];
        float beta_p = (float)beta_in[b * L * H + l_p * H + h];
        float ab = a * beta_p;
        float g_p = g_cumsum[g_base + p];
        float coeff = ab * exp(g_p);
        float4 v_p;
        v_p.x = (float)conv_out[conv_base + (2 * key_dim + h * dv + j) * L + l_p];
        v_p.y = (float)conv_out[conv_base + (2 * key_dim + h * dv + j + 1) * L + l_p];
        v_p.z = (float)conv_out[conv_base + (2 * key_dim + h * dv + j + 2) * L + l_p];
        v_p.w = (float)conv_out[conv_base + (2 * key_dim + h * dv + j + 3) * L + l_p];
        result0 += ab * v_p;
        float4 k_p;
        k_p.x = (float)conv_out[conv_base + (key_dim + h * dk + j) * L + l_p];
        k_p.y = (float)conv_out[conv_base + (key_dim + h * dk + j + 1) * L + l_p];
        k_p.z = (float)conv_out[conv_base + (key_dim + h * dk + j + 2) * L + l_p];
        k_p.w = (float)conv_out[conv_base + (key_dim + h * dk + j + 3) * L + l_p];
        result1 += coeff * k_p;
    }

    int out_idx = ((b * H + h) * num_chunks + c) * C * dv + row * dv + j;
    vstore4((float4)(result0), 0, v_corrected + out_idx);
    vstore4((float4)(result1), 0, k_cumdecay + out_idx);
}

// Kernel C5: Intra-chunk QK attention with decay (lower triangular including diagonal)
// qk[i,j] = (q[i] dot k[j]) * q_scale * exp(g[i]-g[j]) for j <= i
// GWS: {CHUNK_SIZE, CHUNK_SIZE * num_chunks, B * H}
__kernel void chunk_qk_attn(
    __global const FLOAT* conv_out,    // [B, D, L]
    __global const float* g_cumsum,    // [B, H, num_chunks, C] (float32)
    __global float* qk_attn_out,       // [B, H, num_chunks, C, C] (float32, reuses attn_matrix buf)
    __private const int global_dim0,
    __private const int global_dim1,
    __private const int global_dim2,
    __private const int conv_dim,
    __private const int seq_len,
    __private const int num_v_heads,
    __private const int head_k_dim,
    __private const int key_dim,
    __private const int gqa_factor,
    __private const int num_chunks,
    __private const float q_scale)
{
    const int col = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    if (col >= global_dim0 || y >= global_dim1 || z >= global_dim2) return;

    const int c = y / CHUNK_SIZE;
    const int row = y % CHUNK_SIZE;
    const int b = z / num_v_heads;
    const int h = z % num_v_heads;

    const int C = CHUNK_SIZE;
    const int dk = head_k_dim;
    const int D = conv_dim;
    const int L = seq_len;
    const int H = num_v_heads;
    const int k_head = h / gqa_factor;

    float val = 0.0f;
    if (col <= row) {
        const int l_row = c * C + row;
        const int l_col = c * C + col;
        if (l_row < L && l_col < L) {
            const int g_base = ((b * H + h) * num_chunks + c) * C;
            const float decay = exp(g_cumsum[g_base + row] - g_cumsum[g_base + col]);
            const int conv_base = b * D * L;
            float dot = 0.0f;
            for (int d = 0; d < dk; ++d) {
                float q_d = (float)conv_out[conv_base + (k_head * dk + d) * L + l_row];
                float k_d = (float)conv_out[conv_base + (key_dim + k_head * dk + d) * L + l_col];
                dot += q_d * k_d;
            }
            val = dot * q_scale * decay;
        }
    }

    int out_idx = ((b * H + h) * num_chunks + c) * C * C + row * C + col;
    qk_attn_out[out_idx] = val;
}

// Kernel C6: v_new = v_corrected - k_cumdecay @ state (per chunk)
// GWS: {UP_DIV(dv, 4), CHUNK_SIZE, B * H}
__kernel void chunk_compute_vnew(
    __global const float* v_corrected,    // [B, H, num_chunks, C, dv] (float32)
    __global const float* k_cumdecay,     // [B, H, num_chunks, C, dk] (float32)
    __global const FLOAT* recurrent_state,// [B, H, dk, dv]
    __global float* v_new,                // [B, H, C, dv] (float32)
    __private const int global_dim0,
    __private const int global_dim1,
    __private const int global_dim2,
    __private const int head_k_dim,
    __private const int head_v_dim,
    __private const int num_v_heads,
    __private const int num_chunks,
    __private const int chunk_idx)
{
    const int x = get_global_id(0);
    const int p = get_global_id(1);
    const int z = get_global_id(2);
    if (x >= global_dim0 || p >= global_dim1 || z >= global_dim2) return;

    const int j = x << 2;
    const int b = z / num_v_heads;
    const int h = z % num_v_heads;

    const int C = CHUNK_SIZE;
    const int dk = head_k_dim;
    const int dv = head_v_dim;
    const int H = num_v_heads;
    const int c = chunk_idx;

    const int kc_base = ((b * H + h) * num_chunks + c) * C * dk + p * dk;
    const int state_base = (b * H + h) * dk * dv;

    float4 v_prime = (float4)(0.0f);
    for (int d = 0; d < dk; ++d) {
        float kc_d = k_cumdecay[kc_base + d];
        float4 s = convert_float4(vload4(0, recurrent_state + state_base + d * dv + j));
        v_prime += kc_d * s;
    }

    int vc_idx = ((b * H + h) * num_chunks + c) * C * dv + p * dv + j;
    float4 vc = vload4(0, v_corrected + vc_idx);
    float4 vn = vc - v_prime;

    int vn_idx = (b * H + h) * C * dv + p * dv + j;
    vstore4((float4)(vn), 0, v_new + vn_idx);
}

// Kernel C7: Compute output and update recurrent state (per chunk)
// GWS: {dk * UP_DIV(dv, 4), H, batch}
__kernel void chunk_output_state_update(
    __global const FLOAT* conv_out,        // [B, D, L]
    __global const float* qk_attn_matrix,  // [B, H, num_chunks, C, C] (float32)
    __global const float* v_new,           // [B, H, C, dv] (float32)
    __global const float* g_cumsum,        // [B, H, num_chunks, C] (float32)
    __global FLOAT* recurrent_state,       // [B, H, dk, dv]
    __global FLOAT* attn_out,              // [B, L, H*dv]
    __private const int global_dim0,
    __private const int global_dim1,
    __private const int global_dim2,
    __private const int conv_dim,
    __private const int seq_len,
    __private const int num_v_heads,
    __private const int head_k_dim,
    __private const int head_v_dim,
    __private const int key_dim,
    __private const int gqa_factor,
    __private const int num_chunks,
    __private const int chunk_idx,
    __private const float q_scale)
{
    const int x = get_global_id(0);
    const int h = get_global_id(1);
    const int b = get_global_id(2);
    if (x >= global_dim0 || h >= global_dim1 || b >= global_dim2) return;
    
    const int dk = head_k_dim;
    const int dv = head_v_dim;
    const int dv4 = (dv + 3) / 4;
    const int j = (x % dv4) << 2;
    const int d = x / dv4;
    const int C = CHUNK_SIZE;
    const int D = conv_dim;
    const int L = seq_len;
    const int H = num_v_heads;
    const int c = chunk_idx;
    const int k_head = h / gqa_factor;

    const int state_base = (b * H + h) * dk * dv;
    const int g_base = ((b * H + h) * num_chunks + c) * C;
    const int qk_base = ((b * H + h) * num_chunks + c) * C * C;
    const int vn_base = (b * H + h) * C * dv;
    const int conv_base = b * D * L;

    int last_p = min(C, L - c * C);
    if (last_p <= 0) return;

    // Phase 2: Update recurrent state
    // state[d,j] = state[d,j]*exp(g_last) + sum_p k[p,d]*exp(g_last-g[p])*v_new[p,j]
    float g_last = g_cumsum[g_base + last_p - 1];
    float decay_state = exp(g_last);

    float4 s = convert_float4(vload4(0, recurrent_state + state_base + d * dv + j)) * (float4)(decay_state);
    for (int p = 0; p < last_p; ++p) {
        int l_p = c * C + p;
        float g_p = g_cumsum[g_base + p];
        float k_d = (float)conv_out[conv_base + (key_dim + k_head * dk + d) * L + l_p] * exp(g_last - g_p);
        float4 vn = vload4(0, v_new + vn_base + p * dv + j);
        s += k_d * vn;
    }
    vstore4(CONVERT_FLOAT4(s), 0, recurrent_state + state_base + d * dv + j);
}

__kernel void chunk_output(
    __global const FLOAT* conv_out,        // [B, D, L]
    __global const float* qk_attn_matrix,  // [B, H, num_chunks, C, C] (float32)
    __global const float* v_new,           // [B, H, C, dv] (float32)
    __global const float* g_cumsum,        // [B, H, num_chunks, C] (float32)
    __global const FLOAT* recurrent_state, // [B, H, dk, dv]
    __global FLOAT* attn_out,              // [B, L, H*dv]
    __private const int global_dim0,
    __private const int global_dim1,
    __private const int global_dim2,
    __private const int conv_dim,
    __private const int seq_len,
    __private const int num_v_heads,
    __private const int dk,
    __private const int dv,
    __private const int key_dim,
    __private const int gqa_factor,
    __private const int num_chunks,
    __private const int chunk_idx,
    __private const float q_scale)
{
    const int x = get_global_id(0);
    const int h = get_global_id(1);
    const int b = get_global_id(2);
    if (x >= global_dim0 || h >= global_dim1 || b >= global_dim2) return;
    
    const int dv4 = (dv + 3) / 4;
    const int j = (x % dv4) << 2;
    const int p = x / dv4;
    const int C = CHUNK_SIZE;
    const int D = conv_dim;
    const int L = seq_len;
    const int H = num_v_heads;
    const int c = chunk_idx;
    const int k_head = h / gqa_factor;

    const int state_base = (b * H + h) * dk * dv;
    const int g_base = ((b * H + h) * num_chunks + c) * C;
    const int qk_base = ((b * H + h) * num_chunks + c) * C * C;
    const int vn_base = (b * H + h) * C * dv;
    const int conv_base = b * D * L;

    int last_p = min(C, L - c * C);
    if (last_p <= 0 || p >= last_p) return;

    // Phase 1: Compute output for each position p
    int l_p = c * C + p;
    float g_p = g_cumsum[g_base + p];
    float decay_q = exp(g_p) * q_scale;

    // Cross-chunk: q * scale * exp(g) @ state
    float4 inter = (float4)(0.0f);
    for (int d = 0; d < dk; d++) {
        float q_d = (float)conv_out[conv_base + (k_head * dk + d) * L + l_p] * decay_q;
        float4 s = convert_float4(vload4(0, recurrent_state + state_base + d * dv + j));
        inter += q_d * s;
    }
    
    // Intra-chunk: qk_attn @ v_new
    float4 intra = (float4)(0.0f);
    for (int r = 0; r <= p; ++r) {
        float qk = qk_attn_matrix[qk_base + p * C + r];
        float4 vn = vload4(0, v_new + vn_base + r * dv + j);
        intra += qk * vn;
    }

    const int out_offset = (b * L + l_p) * H * dv + h * dv + j;
    vstore4(CONVERT_FLOAT4(inter + intra), 0, attn_out + out_offset);
}

#endif // CHUNK_PREFILL
