#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#define PI 3.141592653589f
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void binary(__private int global_dim0, __private int global_dim1,
                         __read_only image2d_t input0, __read_only image2d_t input1,
                         __write_only image2d_t output,
                         __private const int4 shape,//[N,H,W,C4]
                         __private const int2 isFull,
                         __private const int activationType) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));//WC4, NH
    
    float4 in0, in1;
    if (pos.x < global_dim0 && pos.y < global_dim1) {

        if(isFull.x == 0) {
            in0 = convert_float4(RI_DATA(input0, SAMPLER, (int2)(0, 0)));
            in0 = (float4)(in0.x, in0.x, in0.x, in0.x);
        } else {
            in0 = convert_float4(RI_DATA(input0, SAMPLER, pos));
        }
        if(isFull.y == 0) {
            in1 = convert_float4(RI_DATA(input1, SAMPLER, (int2)(0, 0)));
            in1 = (float4)(in1.x, in1.x, in1.x, in1.x);
        } else {
            in1 = convert_float4(RI_DATA(input1, SAMPLER, pos));
        }
        
        float4 out = OPERATOR;
        
        if(activationType == 1) {
            out = fmax(out, (float4)0);
        }
        WI_DATA(output, pos, CONVERT_OUTPUT_I4(out));
    }
}

__kernel void binary_prelu(__read_only image2d_t input0, __read_only image2d_t input1, __write_only image2d_t output,
                            int4 shape, int2 whInput1, int4 input1NHWCStep) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    int4 nhwc = (int4)(pos.y/shape.y, pos.y%shape.y, pos.x%shape.z, pos.x/shape.z);
    if (nhwc.x < shape.x && nhwc.w < shape.w) {
            int4 nhwc1 = nhwc * input1NHWCStep;
            int2 pos1 = (int2)(nhwc1.w*whInput1.x+nhwc1.z, nhwc1.x*whInput1.y+nhwc1.y);

            float4 in0 = convert_float4(RI_DATA(input0, SAMPLER, pos));
            float4 in1 = convert_float4(RI_DATA(input1, SAMPLER, pos1));
            OUTPUT_TYPE_I4 out = CONVERT_OUTPUT_I4(OPERATOR);
            WI_DATA(output, pos, out);
        }
}

__kernel void imageCopy(__read_only image2d_t input, __write_only image2d_t output) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    const int2 dim = get_image_dim(input);
    if (pos.x >= dim.x && pos.y >= dim.y) {
        return;
    }
    WI_DATA(output, pos, CONVERT_OUTPUT_I4(RI_DATA(input, SAMPLER, pos)));
}
