#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void space_to_batch(GLOBAL_SIZE_3_DIMS __read_only image2d_t uInput, __write_only image2d_t uOutput,
                             __private const int4 inImageSize, __private const int4 outImgSize,
                             __private const int2 padding, __private const int2 blockShape) {

    const int out_c_idx = get_global_id(0);
    const int ou_w_idx = get_global_id(1);
    const int out_hb_idx = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(out_c_idx, ou_w_idx, out_hb_idx);

    const int out_b_idx = out_hb_idx / outImgSize.s1;
    const int out_h_idx = out_hb_idx - mul24(out_b_idx, outImgSize.s1);

    const int r_b_idx = out_b_idx / inImageSize.s3;
    const int in_b_idx = out_b_idx - mul24(r_b_idx, inImageSize.s3);

    const int r_b_w = r_b_idx / blockShape.s1;
    const int in_h_idx = r_b_w + mul24(out_h_idx, blockShape.s0) - padding.s0;
    const int in_w_idx = r_b_idx - mul24(r_b_w, blockShape.s1) + mul24(ou_w_idx, blockShape.s1) - padding.s1;

    const int input_x = select(mul24(out_c_idx, inImageSize.s0) + in_w_idx, -1, in_w_idx < 0 || in_w_idx >= inImageSize.s0);
    const int input_y = select(mul24(in_b_idx, inImageSize.s1) + in_h_idx, -1, in_h_idx < 0 || in_h_idx >= inImageSize.s1);
  
    FLOAT4 value = RI_F(uInput, SAMPLER, (int2)(input_x, input_y));

    WI_F(uOutput, (int2)(mul24(out_c_idx, outImgSize.s0) + ou_w_idx, out_hb_idx), value);
}


