#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,
#define DEAL_NON_UNIFORM_DIM2(input1, input2)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                     \
    }
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void crop(GLOBAL_SIZE_2_DIMS __read_only image2d_t input, __write_only image2d_t output,
                   __private const int inputH, __private const int intputW, __private const int offsetB,
                   __private const int offsethH, __private const int offsetW, __private const int offsetC4,
                   __private const int outputH, __private const int outputW) {
    int crop_image_width_idx  = get_global_id(0);
    int crop_image_height_idx = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(crop_image_width_idx, crop_image_height_idx);
    const int height = outputH;
    const int width  = outputW;

    const int batch_idx    = crop_image_height_idx / height;
    const int height_idx   = crop_image_height_idx % height;
    const int width_idx    = crop_image_width_idx % width;
    const int channel4_idx = crop_image_width_idx / width;

    const int srcIndexC4 = offsetC4 + channel4_idx;
    const int srcIndexW  = offsetW + width_idx;
    const int srcIndexB  = offsetB + batch_idx;
    const int srcIndexH  = offsethH + height_idx;

    FLOAT4 values =
        RI_F(input, SAMPLER, (int2)(intputW * srcIndexC4 + srcIndexW, inputH * srcIndexB + srcIndexH));
    WI_F(output, (int2)(crop_image_width_idx, crop_image_height_idx), values);
}
