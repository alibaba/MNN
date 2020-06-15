#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_2_DIMS \
__private const int global_size_dim0, __private const int global_size_dim1,

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                                             \
if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
return;                                                                                   \
}

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void matmul(GLOBAL_SIZE_2_DIMS __read_only image2d_t input_a, __read_only image2d_t input_b,
                     __write_only image2d_t output_c, __private const int channels,
                     __private const int channel_blocks) {
    const int width_blocks_idx = get_global_id(0);
    const int height_idx       = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(width_blocks_idx, height_idx);
    FLOAT4 a;
    FLOAT4 b0 = 0, b1 = 0, b2 = 0, b3 = 0;

    FLOAT result0 = 0;
    FLOAT result1 = 0;
    FLOAT result2 = 0;
    FLOAT result3 = 0;

    for (short pos = 0; pos < channel_blocks; pos += 1) {
        a = RI_F(input_a, SAMPLER, (int2)(pos, height_idx));

        short remain = (pos + 1) * 4 - channels;

        b0 = RI_F(input_b, SAMPLER, (int2)(width_blocks_idx, pos * 4));
        b1 = RI_F(input_b, SAMPLER, (int2)(width_blocks_idx, pos * 4 + 1));
        b2 = RI_F(input_b, SAMPLER, (int2)(width_blocks_idx, pos * 4 + 2));
        b3 = RI_F(input_b, SAMPLER, (int2)(width_blocks_idx, pos * 4 + 3));

        if (remain == 3) {
            b1 = 0;
            b2 = 0;
            b3 = 0;
        } else if (remain == 2) {
            b2 = 0;
            b3 = 0;
        } else if (remain == 1) {
            b3 = 0;
        }

        FLOAT4 btmp0 = (FLOAT4)(b0.s0, b1.s0, b2.s0, b3.s0);
        FLOAT4 btmp1 = (FLOAT4)(b0.s1, b1.s1, b2.s1, b3.s1);
        FLOAT4 btmp2 = (FLOAT4)(b0.s2, b1.s2, b2.s2, b3.s2);
        FLOAT4 btmp3 = (FLOAT4)(b0.s3, b1.s3, b2.s3, b3.s3);

        result0 += dot(a, btmp0);
        result1 += dot(a, btmp1);
        result2 += dot(a, btmp2);
        result3 += dot(a, btmp3);
    }
    WI_F(output_c, (int2)(width_blocks_idx, height_idx), (FLOAT4)(result0, result1, result2, result3));
}

__kernel void matmul_transB(GLOBAL_SIZE_2_DIMS __read_only image2d_t input_a, __read_only image2d_t input_b,
                     __write_only image2d_t output_c, __private const int channels,
                     __private const int channel_blocks) {
    const int width_blocks_idx = get_global_id(0);
    const int height_idx       = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(width_blocks_idx, height_idx);
    FLOAT4 a;
    FLOAT4 b0 = 0, b1 = 0, b2 = 0, b3 = 0;

    FLOAT result0 = 0;
    FLOAT result1 = 0;
    FLOAT result2 = 0;
    FLOAT result3 = 0;

    for (short pos = 0; pos < channel_blocks; pos += 1) {
        a = RI_F(input_a, SAMPLER, (int2)(pos, height_idx));

        short remain = (pos + 1) * 4 - channels;

        b0 = RI_F(input_b, SAMPLER, (int2)(pos, width_blocks_idx * 4));
        b1 = RI_F(input_b, SAMPLER, (int2)(pos, width_blocks_idx * 4 + 1));
        b2 = RI_F(input_b, SAMPLER, (int2)(pos, width_blocks_idx * 4 + 2));
        b3 = RI_F(input_b, SAMPLER, (int2)(pos, width_blocks_idx * 4 + 3));
        if (remain == 3) {
            a.y = 0;
            a.z = 0;
            a.w = 0;
        } else if (remain == 2) {
            a.z = 0;
            a.w = 0;
        } else if (remain == 1) {
            a.w = 0;
        }

        result0 += dot(a, b0);
        result1 += dot(a, b1);
        result2 += dot(a, b2);
        result3 += dot(a, b3);
    }
    WI_F(output_c, (int2)(width_blocks_idx, height_idx), (FLOAT4)(result0, result1, result2, result3));
}