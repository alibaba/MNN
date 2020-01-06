#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

__kernel void batch_to_space(GLOBAL_SIZE_3_DIMS __read_only image2d_t uInput, __write_only image2d_t uOutput,
                             __private const int4 inImageSize, __private const int4 outImgSize,
                             __private const int2 padding, __private const int2 blockShape) {

    const int in_c_idx = get_global_id(0);
    const int in_w_idx = get_global_id(1);
    const int in_hb_idx = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(in_c_idx, in_w_idx, in_hb_idx);

    const int in_b_idx = in_hb_idx / inImageSize.s1;
    const int in_h_idx = in_hb_idx - mul24(in_b_idx, inImageSize.s1);

    const int r_b_idx = in_b_idx / outImgSize.s3;
    const int out_b_idx = in_b_idx - mul24(r_b_idx, outImgSize.s3);

    const int n_h = r_b_idx / blockShape.s1;
    const int mod_h = r_b_idx - mul24(n_h, blockShape.s1);
    
    const int out_h_idx = mad24(in_h_idx, blockShape.s0, n_h - padding.s0);
    const int out_w_idx = mad24(in_w_idx, blockShape.s1, mod_h - padding.s1);

    if (0 <= out_w_idx && out_w_idx < outImgSize.s0 && 0 <= out_h_idx && out_h_idx < outImgSize.s1) {
        FLOAT4 value = RI_F(uInput, SAMPLER, (int2)(mad24(in_c_idx, inImageSize.s0, in_w_idx), in_hb_idx));
        WI_F(uOutput, (int2)(mad24(in_c_idx, outImgSize.s0, out_w_idx), mad24(out_b_idx, outImgSize.s1, out_h_idx)), value);
    }
}
