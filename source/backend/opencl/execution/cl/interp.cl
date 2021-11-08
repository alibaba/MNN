#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void interp(GLOBAL_SIZE_3_DIMS __read_only image2d_t input, __write_only image2d_t output,
                     __private const float height_scale, __private const float width_scale,
                     __private const float height_offset, __private const float width_offset,
                     __private const int input_height, __private const int input_width,
                     __private const int out_height) {
    const int output_channel_block_idx      = get_global_id(0);
    const int output_width_block_idx        = get_global_id(1);
    const int output_batch_height_block_idx = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(output_channel_block_idx, output_width_block_idx, output_batch_height_block_idx);
    const int output_channel_block_idxs = global_size_dim0;
    const int output_width              = global_size_dim1;

    const int output_batch_idx  = output_batch_height_block_idx / out_height;
    const int output_height_idx = output_batch_height_block_idx % out_height;

    const float scale_height = output_height_idx * height_scale + height_offset;
    const float scale_width  = output_width_block_idx * width_scale + width_offset;
#define CLAMP(val, min_val, max_val) max(min(val, max_val), min_val)
    const int height_floor   = (int)floor(scale_height);
    const int height_lf      = CLAMP(height_floor, 0, input_height - 1);
    const int height_uf      = CLAMP(height_floor + 1, 0, input_height - 1);
    
    const int width_floor   = (int)floor(scale_width);
    const int width_lf      = CLAMP(width_floor, 0, input_width - 1);
    const int width_uf      = CLAMP(width_floor + 1, 0, input_width - 1);

    const float height_gap = scale_height - height_floor;
    const float width_gap  = scale_width - width_floor;

    const int input_width_offset  = mul24(output_channel_block_idx, input_width);
    const int input_height_offset = mul24(output_batch_idx, input_height);

    float4 top_left =
        read_imagef(input, SAMPLER, (int2)(input_width_offset + width_lf, input_height_offset + height_lf));
    float4 top_right =
        read_imagef(input, SAMPLER, (int2)(input_width_offset + width_uf, input_height_offset + height_lf));
    float4 bottom_left =
        read_imagef(input, SAMPLER, (int2)(input_width_offset + width_lf, input_height_offset + height_uf));
    float4 bottom_right =
        read_imagef(input, SAMPLER, (int2)(input_width_offset + width_uf, input_height_offset + height_uf));
    float4 top    = mad((top_right - top_left), width_gap, top_left);
    float4 bottom = mad((bottom_right - bottom_left), width_gap, bottom_left);
    float4 out    = mad((bottom - top), height_gap, top);

    const int out_image_w = mad24(output_channel_block_idx, output_width, output_width_block_idx);
    const int out_image_h = mad24(output_batch_idx, out_height, output_height_idx);

    write_imagef(output, (int2)(out_image_w, out_image_h), out);
}
