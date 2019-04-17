__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void copy_buffer_to_image2d(__global const float4* input, __write_only image2d_t uOutput,
                                     __private const int width, __private const int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        write_imagef(uOutput, (int2)(x, y), input[x + y * width]);
    }
}
