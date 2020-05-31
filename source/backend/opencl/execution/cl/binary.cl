#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void binary_same_channel_broadcast(__read_only image2d_t input0, __read_only image2d_t input1, __write_only image2d_t output,
                    int4 shape, int2 whInput0, int2 whInput1, int2 whOutput) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    int4 nhwc = (int4)(pos.y/shape.y, pos.y%shape.y, pos.x%shape.z, pos.x/shape.z);
    if (nhwc.x >= shape.x && nhwc.w >= shape.w)
        return;

    FLOAT4 in0, in1;
    int2 pos0, pos1;

    if (whInput0.x == 1 && whInput0.y == 1) {
        pos0 = (int2)(nhwc.w*whInput0.x, 0);
        in0 = RI_F(input0, SAMPLER, pos0);
        pos1 = (int2)(nhwc.w*whOutput.x+nhwc.z, nhwc.x*whOutput.y+nhwc.y);
    } else if (whInput0.x == 1) { // Tensor 0 width length 1
        pos0 = (int2)(nhwc.w*whInput0.x, nhwc.x*whOutput.y+nhwc.y);
        in0 = RI_F(input0, SAMPLER, pos0);
        pos1 = (whInput1.y != 1) ?
            (int2)(nhwc.w*whOutput.x+nhwc.z, nhwc.x*whOutput.y+nhwc.y) :
            (int2)(nhwc.w*whOutput.x+nhwc.z, 0);
    } else if (whInput0.y == 1) { // Tensor 0 height length 1
        pos0 = (int2)(nhwc.w*whOutput.x+nhwc.z, 0);
        in0 = RI_F(input0, SAMPLER, pos0);
        pos1 = (whInput1.x != 1) ?
            (int2)(nhwc.w*whOutput.x+nhwc.z, nhwc.x*whOutput.y+nhwc.y) :
            (int2)(nhwc.w*whInput1.x, nhwc.x*whOutput.y+nhwc.y);
    } 
    in1 = RI_F(input1, SAMPLER, pos1);
    WI_F(output, pos, OPERATOR);
}

__kernel void binary_1toM_channel_broadcast_on_awh(__read_only image2d_t input0, __read_only image2d_t input1, __write_only image2d_t output,
                    int4 shape, int2 whInput0, int2 whInput1, int2 whOutput) {

    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    int4 nhwc = (int4)(pos.y/shape.y, pos.y%shape.y, pos.x%shape.z, pos.x/shape.z);
    if (nhwc.x >= shape.x && nhwc.w >= shape.w)
        return;

    FLOAT4 in0, in1;
    int2 pos0, pos1;
    pos0 = (int2)(nhwc.z, nhwc.y);
    FLOAT4 value = RI_F(input0, SAMPLER, pos0);
    in0 = (FLOAT4)(value.x);

    if (whInput1.x != 1 && whInput1.y == 1) {
        pos1 = (int2)(nhwc.w*whOutput.x+nhwc.z, 0);
    } else if (whInput1.x == 1 && whInput1.y != 1) {
        pos1 = (int2)(nhwc.w*whInput1.x, nhwc.x*whOutput.y+nhwc.y);
    } else if (whInput1.x == 1 && whInput1.y == 1) {
        pos1 = (int2)(nhwc.w*whInput1.x, 0);
    } else {
        pos1 = (int2)(nhwc.w*whOutput.x+nhwc.z, nhwc.x*whOutput.y+nhwc.y);
    }
    in1 = RI_F(input1, SAMPLER, pos1);
    WI_F(output, pos, OPERATOR);
}

__kernel void binary_1toM_channel_broadcast_on_1wh(__read_only image2d_t input0, __read_only image2d_t input1, __write_only image2d_t output,
                    int4 shape, int2 whInput0, int2 whInput1, int2 whOutput) {

    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    int4 nhwc = (int4)(pos.y/shape.y, pos.y%shape.y, pos.x%shape.z, pos.x/shape.z);
    if (nhwc.x >= shape.x && nhwc.w >= shape.w)
        return;

    FLOAT4 in0, in1;
    int2 pos0, pos1;
    if (whInput0.x == 1 && whInput0.y == 1) {
        pos0 = (int2)(0, 0);
        FLOAT4 value = RI_F(input0, SAMPLER, pos0);
        in0 = (FLOAT4)(value.x);
        pos1 = (int2)(nhwc.w*whOutput.x+nhwc.z, nhwc.x*whOutput.y+nhwc.y);
    }
    else if (whInput0.x == 1) { // Tensor 0 width length 1
        pos0 = (int2)(0, nhwc.y);
        FLOAT4 value = RI_F(input0, SAMPLER, pos0);
        in0 = (FLOAT4)(value.x);
        pos1 = (whInput1.y != 1) ?
            (int2)(nhwc.w*whOutput.x+nhwc.z, nhwc.x*whOutput.y+nhwc.y) :
            (int2)(nhwc.w*whOutput.x+nhwc.z, 0);
    } else if (whInput0.y == 1) { // Tensor 0 height length 1
        pos0 = (int2)(nhwc.z, 0);
        FLOAT4 value = RI_F(input0, SAMPLER, pos0);
        in0 = (FLOAT4)(value.x);
        pos1 = (whInput1.x != 1) ?
               (int2)(nhwc.w * whOutput.x + nhwc.z, nhwc.x * whOutput.y + nhwc.y) :
               (int2)(nhwc.w * whInput1.x, nhwc.x * whOutput.y + nhwc.y);
    }
    in1 = RI_F(input1, SAMPLER, pos1);
    WI_F(output, pos, OPERATOR);
}

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

__kernel void binary_value(__read_only image2d_t input0, __read_only image2d_t input1, __write_only image2d_t output,
                    int4 shape, int2 whInput1, int4 input1NHWCStep) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    int4 nhwc = (int4)(pos.y/shape.y, pos.y%shape.y, pos.x%shape.z, pos.x/shape.z);
    if (nhwc.x < shape.x && nhwc.w < shape.w) {
        int4 nhwc1 = nhwc * input1NHWCStep;
        int2 pos1 = (int2)(nhwc1.w*whInput1.x+nhwc1.z, nhwc1.x*whInput1.y+nhwc1.y);
        const FLOAT input1Data = RI_F(input1, SAMPLER, (int2)(0, 0)).x;
        FLOAT4 in0 = RI_F(input0, SAMPLER, pos);
        FLOAT4 in1 = (FLOAT4)(input1Data);
        WI_F(output, pos, OPERATOR);
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
