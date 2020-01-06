#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void softmax_grad(__read_only image2d_t input0, __read_only image2d_t input1, __write_only image2d_t output, int step, int number, int axisOnC4) {
    const int width = get_image_width(output);
    const int number4 = (number + 3) / 4, remain = number4 * 4 - number;
    const int idx = get_global_id(0) * step * number4 + get_global_id(1);
    FLOAT4 sum;
    if (axisOnC4) {
        FLOAT temp = 0;
        for (int i = 0, _idx = idx; i < number4; ++i, _idx += step) {
            int2 pos = (int2)(_idx % width, _idx / width);
            FLOAT4 out = RI_F(input0, SAMPLER, pos) * RI_F(input1, SAMPLER, pos);
            if (i < number4 - 1 || remain == 0) {
                temp = temp + out.x + out.y + out.z + out.w;
            } else if (remain == 1) {
                temp = temp + out.x + out.y + out.z;
            } else if (remain == 2) {
                temp = temp + out.x + out.y;
            } else {
                temp = temp + out.x;
            }
        }
        sum = (FLOAT4)(temp);
    } else {
        sum = 0;
        for (int i = 0, _idx = idx; i < number4; ++i, _idx += step) {
            int2 pos = (int2)(_idx % width, _idx / width);
            FLOAT4 temp = RI_F(input0, SAMPLER, pos) * RI_F(input1, SAMPLER, pos);
            if (i < number4 - 1 || remain == 0) {
                sum = sum + temp;
            } else if (remain == 1) {
                sum.x = sum.x + temp.x;
                sum.y = sum.y + temp.y;
                sum.z = sum.z + temp.z;
            } else if (remain == 2) {
                sum.x = sum.x + temp.x;
                sum.y = sum.y + temp.y;
            } else {
                sum.x = sum.x + temp.x;
            }
        }
    }
    for (int i = 0, _idx = idx; i < number4; ++i, _idx += step) {
        int2 pos = (int2)(_idx % width, _idx / width);
        FLOAT4 out = RI_F(input0, SAMPLER, pos) * (RI_F(input1, SAMPLER, pos) - sum);
        WI_F(output, pos, out);
    }
}

