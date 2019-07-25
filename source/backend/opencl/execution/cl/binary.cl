#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void binary(__read_only image2d_t input0, __read_only image2d_t input1, __write_only image2d_t output, 
                    int4 shape, int2 whInput1, int4 input1NHWCStep) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    int4 nhwc = (int4)(pos.y/shape.y, pos.y%shape.y, pos.x%shape.z, pos.x/shape.z);
    if (nhwc.x < shape.x && nhwc.w < shape.w) {
        int4 nhwc1 = nhwc * input1NHWCStep;
        int2 pos1 = (int2)(nhwc1.w*whInput1.x+nhwc1.z, nhwc1.x*whInput1.y+nhwc1.y);
        FLOAT4 in0 = RI_F(input0, SAMPLER, pos);
        FLOAT4 in1 = RI_F(input1, SAMPLER, pos1);
        WI_F(output, pos, OPERATOR);
    }
}

__kernel void binary_broadcast(__read_only image2d_t input0, float input1, __write_only image2d_t output, 
                    int4 shape, int2 whInput1, int4 input1NHWCStep) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    int4 nhwc = (int4)(pos.y/shape.y, pos.y%shape.y, pos.x%shape.z, pos.x/shape.z);
    if (nhwc.x < shape.x && nhwc.w < shape.w) {
        int4 nhwc1 = nhwc * input1NHWCStep;
        int2 pos1 = (int2)(nhwc1.w*whInput1.x+nhwc1.z, nhwc1.x*whInput1.y+nhwc1.y);
        FLOAT4 in0 = RI_F(input0, SAMPLER, pos);
        FLOAT4 in1 = (FLOAT4)(input1);
        WI_F(output, pos, OPERATOR);
    }
}