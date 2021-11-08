#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void conv2d_backprop_filter(__read_only image2d_t input, __read_only image2d_t grad, __global float* output_ptr, int batch, int outputChannel, int inputChannel, int2 inputShape, int2 shape, int2 kernelShape, int2 strides, int2 pads, int2 dilates) {
    const int oc_block = get_global_id(0), ic_block = get_global_id(1);
    if (oc_block * 4 >= outputChannel || ic_block * 4 >= inputChannel) {
        return;
    }
    
    const int h = get_global_id(2) / kernelShape.x, w = get_global_id(2) % kernelShape.x;
    int temp = pads.y - h * dilates.y;
    const int ohStart = ceil((float)temp / strides.y);
    const int ohEnd = floor((float)(inputShape.y - 1 + temp) / strides.y);
    const int ihStart = ohStart * strides.y - temp;
    temp = pads.x - w * dilates.x;
    const int owStart = ceil((float)temp / strides.x);
    const int owEnd = floor((float)(inputShape.x - 1 + temp) / strides.x);
    const int iwStart = owStart * strides.x - temp;
    const int i_offset_0 = ic_block * inputShape.x, g_offset_0 = oc_block * shape.x;
    
    FLOAT4 grad0 = 0, grad1 = 0, grad2 = 0, grad3 = 0;
    for (int b = 0; b < batch; ++b) {
        const int i_offset_1 = b * inputShape.y, g_offset_1 = b * shape.y;
        for (int oh = ohStart, ih = ihStart; oh <= ohEnd; ++oh, ih += strides.y) {
            for (int ow = owStart, iw = iwStart; ow <= owEnd; ++ow, iw += strides.x) {
                FLOAT4 in0 = RI_F(input, SAMPLER, (int2)(i_offset_0 + iw, i_offset_1 + ih));
                FLOAT4 in1 = RI_F(grad, SAMPLER, (int2)(g_offset_0 + ow, g_offset_1 + oh));
                grad0 = mad(in0, (FLOAT4)in1.x, grad0);
                grad1 = mad(in0, (FLOAT4)in1.y, grad1);
                grad2 = mad(in0, (FLOAT4)in1.z, grad2);
                grad3 = mad(in0, (FLOAT4)in1.w, grad3);
            }
        }
    }
    
    // save image kernel into buffer
    {
#define FILL_OUTPUT(grad, index, offset) \
    const int remain_channel_ = inputChannel - ic_block * 4; \
    if (remain_channel_ >= 1) output_ptr[index] = grad.x; \
    if (remain_channel_ >= 2) output_ptr[index + offset] = grad.y; \
    if (remain_channel_ >= 3) output_ptr[index + offset * 2] = grad.z; \
    if (remain_channel_ >= 4) output_ptr[index + offset * 3] = grad.w;
        
        const int remain_channel = outputChannel - oc_block * 4;
        const int kernelSize = kernelShape.x * kernelShape.y;
        int index = (oc_block * inputChannel + ic_block) * 4 * kernelSize + h * kernelShape.x + w;
        if (remain_channel >= 1) {
            FILL_OUTPUT(grad0, index, kernelSize);
        }
        if (remain_channel >= 2) {
            index += kernelSize * inputChannel;
            FILL_OUTPUT(grad1, index, kernelSize);
        }
        if (remain_channel >= 3) {
            index += kernelSize * inputChannel;
            FILL_OUTPUT(grad2, index, kernelSize);
        }
        if (remain_channel >= 4) {
            index += kernelSize * inputChannel;
            FILL_OUTPUT(grad3, index, kernelSize);
        }
#undef FILL_OUTPUT
    }
}
