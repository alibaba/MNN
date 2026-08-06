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
// INPUT_TYPE_I4 / RI_DATA / WI_DATA / CONVERT_OUTPUT_I4 are set by the runtime from
// the data tensor dtype (int4/read_imagei for int32 data, float4|half4 otherwise),
// so int32 Select is not corrupted by being read/written as half under fp16 precision.
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
    INPUT_TYPE_I4 in0 = RI_DATA(input0, SAMPLER, (int2)(0, 0));
    in0 = (INPUT_TYPE_I4)(in0.x);
#else
    INPUT_TYPE_I4 in0 = RI_DATA(input0, SAMPLER, (int2)(idx, idy));
#endif

#ifdef INSIZE2_EUQAL_1
    INPUT_TYPE_I4 in1 = RI_DATA(input1, SAMPLER, (int2)(0, 0));
    in1 = (INPUT_TYPE_I4)(in1.x);
#else
    INPUT_TYPE_I4 in1 = RI_DATA(input1, SAMPLER, (int2)(idx, idy));
#endif
    INPUT_TYPE_I4 out;
    out.x = select_vec.x ? in0.x : in1.x;
    out.y = select_vec.y ? in0.y : in1.y;
    out.z = select_vec.z ? in0.z : in1.z;
    out.w = select_vec.w ? in0.w : in1.w;
    WI_DATA(output, (int2)(idx, idy), CONVERT_OUTPUT_I4(out));
}
