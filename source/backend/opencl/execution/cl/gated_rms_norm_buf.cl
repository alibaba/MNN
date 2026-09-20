#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

// OpType_GatedRMSNorm: out = (RMSNorm(x) * gamma + beta) * silu(z).
//
// Layout note (the op absorbs the C4 repacks the exported graph used to carry):
//   x   is [outside, inside]        -> FLOAT4 index (c/4) * outside + h, lane c%4
//   z / out are [batch, heads*inside] -> flat channel f = head * inside + c,
//                                        FLOAT4 index (f/4) * batch + b, lane f%4
// with outside = batch * heads and h = b * heads + head.
//
// When inside % 4 == 0 the x lane and the z/out lane coincide, so the whole row
// is vector work (gated_rms_norm_c4_buf). Otherwise the two sides disagree on
// the lane and the addressing has to go element-wise (gated_rms_norm_buf).

#define SILU(g) ((g) / (1.0f + exp(-(g))))

// Zero the padding lanes of the last channel unit so they cannot enter the sum.
#define MASK_C4_TAIL(value, index, channel_unit, remain)      \
    do {                                                      \
        if ((remain) != 0 && (index) == (channel_unit) - 1) { \
            if ((remain) < 4)                                 \
                (value).w = 0.0f;                             \
            if ((remain) < 3)                                 \
                (value).z = 0.0f;                             \
            if ((remain) < 2)                                 \
                (value).y = 0.0f;                             \
        }                                                     \
    } while (0)

// Sum of squares of row `row` of x, reduced across the workgroup.
#define REDUCE_SQUARE_SUM(sum_local, lid, x, row, outside, channelUnit, channelRemain, acc) \
    do {                                                                                    \
        float4 in_sum = (float4)0;                                                          \
        for (int index = (lid); index < (channelUnit); index += LOCAL_SIZE) {               \
            float4 in = convert_float4((x)[index * (outside) + (row)]);                     \
            MASK_C4_TAIL(in, index, channelUnit, channelRemain);                            \
            in_sum += in * in;                                                              \
        }                                                                                   \
        (sum_local)[lid] = in_sum;                                                          \
        barrier(CLK_LOCAL_MEM_FENCE);                                                       \
        for (int i = LOCAL_SIZE / 2; i > 0; i /= 2) {                                       \
            if ((lid) < i) {                                                                \
                (sum_local)[lid] = (sum_local)[lid] + (sum_local)[(lid) + i];               \
            }                                                                               \
            barrier(CLK_LOCAL_MEM_FENCE);                                                   \
        }                                                                                   \
        (acc) = (sum_local)[0].x + (sum_local)[0].y + (sum_local)[0].z + (sum_local)[0].w;  \
    } while (0)

// inside % 4 == 0: x, z and out share the lane, so everything stays vectorised.
__kernel void gated_rms_norm_c4_buf(__private int global_dim0, __private int global_dim1, __global const FLOAT4* x,
                                    __global const FLOAT4* z, __global FLOAT4* output,
#ifdef GAMMA_BETA
                                    __global const FLOAT4* gamma, __global const FLOAT4* beta,
#endif
                                    __private const int inside, __private const int heads, __private const int batch,
                                    __private const float epsilon) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    float4 local sum_mnn[LOCAL_SIZE];
    if (pos.x >= global_dim0 || pos.y >= global_dim1) {
        return;
    }
    const int lid = get_local_id(0);
    const int outside = global_dim1;
    const int channelUnit = inside >> 2;
    float square_sum_all;
    REDUCE_SQUARE_SUM(sum_mnn, lid, x, pos.y, outside, channelUnit, 0, square_sum_all);
    const float4 scale = (float4)1.0f / (float4)sqrt((float4)(square_sum_all / inside) + (float4)epsilon);

    const int b = pos.y / heads;
    const int head = pos.y - b * heads;
    // f = head * inside + c, and inside % 4 == 0, so the unit offset is exact.
    const int unitBase = (head * inside) >> 2;
    for (int index = lid; index < channelUnit; index += LOCAL_SIZE) {
        float4 in = convert_float4(x[index * outside + pos.y]);
#ifdef GAMMA_BETA
        float4 out = in * scale * convert_float4(gamma[index]) + convert_float4(beta[index]);
#else
        float4 out = in * scale;
#endif
        const int zIdx = (unitBase + index) * batch + b;
        float4 g = convert_float4(z[zIdx]);
        output[zIdx] = CONVERT_FLOAT4(out * SILU(g));
    }
}

// inside % 4 != 0: x carries per-row channel padding that z / out do not, so the
// epilogue addresses both sides element-wise. Correctness path only.
__kernel void gated_rms_norm_buf(__private int global_dim0, __private int global_dim1, __global const FLOAT* x,
                                 __global const FLOAT* z, __global FLOAT* output,
#ifdef GAMMA_BETA
                                 __global const FLOAT* gamma, __global const FLOAT* beta,
#endif
                                 __private const int inside, __private const int heads, __private const int batch,
                                 __private const float epsilon) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    float local sum_mnn[LOCAL_SIZE];
    if (pos.x >= global_dim0 || pos.y >= global_dim1) {
        return;
    }
    const int lid = get_local_id(0);
    const int outside = global_dim1;
    // x element index: unit c/4 of row pos.y, lane c%4. Scalar throughout, so
    // the padding lanes of the last unit are never touched.
    float acc = 0.0f;
    for (int c = lid; c < inside; c += LOCAL_SIZE) {
        float in = (float)x[((c >> 2) * outside + pos.y) * 4 + (c & 3)];
        acc += in * in;
    }
    sum_mnn[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = LOCAL_SIZE / 2; i > 0; i /= 2) {
        if (lid < i) {
            sum_mnn[lid] = sum_mnn[lid] + sum_mnn[lid + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float scale = 1.0f / sqrt(sum_mnn[0] / inside + epsilon);

    const int b = pos.y / heads;
    const int head = pos.y - b * heads;
    for (int c = lid; c < inside; c += LOCAL_SIZE) {
        const int xIdx = ((c >> 2) * outside + pos.y) * 4 + (c & 3);
        float in = (float)x[xIdx];
#ifdef GAMMA_BETA
        float out = in * scale * (float)gamma[c] + (float)beta[c];
#else
        float out = in * scale;
#endif
        const int f = head * inside + c;
        const int zIdx = ((f >> 2) * batch + b) * 4 + (f & 3);
        float g = (float)z[zIdx];
        output[zIdx] = (FLOAT)(out * SILU(g));
    }
}
