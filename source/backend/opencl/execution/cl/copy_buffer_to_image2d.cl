#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void copy_buffer_to_image2d(
                                     #ifdef BUFFER_INP_FP32
                                     __global const float4* input,
                                     #else
                                     __global const FLOAT4* input,
                                     #endif
                                     __write_only image2d_t uOutput,
                                     __private const int width, __private const int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        WI_F(uOutput, (int2)(x, y), (FLOAT4)((FLOAT)input[x + y * width].x, (FLOAT)input[x + y * width].y, (FLOAT)input[x + y * width].z, (FLOAT)input[x + y * width].w));
    }
}
