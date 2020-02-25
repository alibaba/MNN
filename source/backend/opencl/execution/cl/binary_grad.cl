#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void relu_grad(__read_only image2d_t input0, __read_only image2d_t input1, __write_only image2d_t output) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    const int2 imageDim = get_image_dim(output);
    if (pos.x >= imageDim.x || pos.y >= imageDim.y) {
        return;
    }
    FLOAT4 in0 = RI_F(input0, SAMPLER, pos);
    FLOAT4 in1 = RI_F(input1, SAMPLER, pos);
    FLOAT4 out0 = select(in1, (FLOAT4)0, in0 < (FLOAT4)0);
    WI_F(output, pos, out0);
}

__kernel void relu6_grad(__read_only image2d_t input0, __read_only image2d_t input1, __write_only image2d_t output) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    const int2 imageDim = get_image_dim(output);
    if (pos.x >= imageDim.x || pos.y >= imageDim.y) {
        return;
    }
    FLOAT4 in0 = RI_F(input0, SAMPLER, pos);
    FLOAT4 in1 = RI_F(input1, SAMPLER, pos);
    FLOAT4 out0 = select(in1, (FLOAT4)0, in0 <= (FLOAT4)0 || in0 >= (FLOAT4)6);
    WI_F(output, pos, out0);
}

