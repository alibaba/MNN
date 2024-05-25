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
__kernel void select_img(GLOBAL_SIZE_2_DIMS
                            __read_only image2d_t input,
                            __read_only image2d_t input0,
                            __read_only image2d_t input1,
                            __write_only image2d_t output
                            ) {
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(idx, idy);
    int4 select_vec = read_imagei(input, SAMPLER, (int2)(idx, idy));
#ifdef INSIZE1_EUQAL_1
    FLOAT4 in0 = RI_F(input0, SAMPLER, (int2)(0, 0));
    in0 = (FLOAT4)(in0.x);
#else
    FLOAT4 in0 = RI_F(input0, SAMPLER, (int2)(idx, idy));
#endif
    
#ifdef INSIZE2_EUQAL_1
    FLOAT4 in1 = RI_F(input1, SAMPLER, (int2)(0, 0));
    in1 = (FLOAT4)(in1.x);
#else
    FLOAT4 in1 = RI_F(input1, SAMPLER, (int2)(idx, idy));
#endif
    FLOAT4 out = select(in1, in0, select_vec == (int4)1);
    WI_F(output, (int2)(idx, idy), out);
}
