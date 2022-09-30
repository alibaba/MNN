#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void binary(__private int global_dim0, __private int global_dim1,
                         __read_only image2d_t input0, __read_only image2d_t input1,
                         __write_only image2d_t output,
                         __private const int4 shape,//[N,H,W,C4]
                         __private const int2 isFull,
                         __private const int activationType) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));//WC4, NH
    
    FLOAT4 in0, in1;
    if (pos.x < global_dim0 && pos.y < global_dim1) {

        if(isFull.x == 0) {
            in0 = RI_F(input0, SAMPLER, (int2)(0, 0));
            in0 = (FLOAT4)(in0.x, in0.x, in0.x, in0.x);
        } else {
            in0 = RI_F(input0, SAMPLER, pos);
        }
        if(isFull.y == 0) {
            in1 = RI_F(input1, SAMPLER, (int2)(0, 0));
            in1 = (FLOAT4)(in1.x, in1.x, in1.x, in1.x);
        } else {
            in1 = RI_F(input1, SAMPLER, pos);
        }
        
        FLOAT4 out = CONVERT_FLOAT4(OPERATOR);
        if(activationType == 1) {
            out = fmax(out, (FLOAT4)0);
        }
        WI_F(output, pos, out);
    }
}

__kernel void binary_prelu(__read_only image2d_t input0, __read_only image2d_t input1, __write_only image2d_t output,
                            int4 shape, int2 whInput1, int4 input1NHWCStep) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    int4 nhwc = (int4)(pos.y/shape.y, pos.y%shape.y, pos.x%shape.z, pos.x/shape.z);
    if (nhwc.x < shape.x && nhwc.w < shape.w) {
            int4 nhwc1 = nhwc * input1NHWCStep;
            int2 pos1 = (int2)(nhwc1.w*whInput1.x+nhwc1.z, nhwc1.x*whInput1.y+nhwc1.y);
            FLOAT4 in0 = RI_F(input0, SAMPLER, pos);
            FLOAT4 in1 = RI_F(input1, SAMPLER, pos1);
            FLOAT4 out = CONVERT_FLOAT4(OPERATOR);
            WI_F(output, pos, out);
        }
}

__kernel void imageCopy(__read_only image2d_t input, __write_only image2d_t output) {
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    const int2 dim = get_image_dim(input);
    if (pos.x >= dim.x && pos.y >= dim.y) {
        return;
    }
    WI_F(output, pos, RI_F(input, SAMPLER, pos));
}
