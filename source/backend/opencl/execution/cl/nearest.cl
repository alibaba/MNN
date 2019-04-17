#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void interp(GLOBAL_SIZE_3_DIMS __read_only image2d_t input, __write_only image2d_t output,
                     __private const float height_scale, __private const float width_scale,
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

    const float scale_height = output_height_idx * height_scale;
    const float scale_width  = output_width_block_idx * width_scale;
    const int height_lf      = max(0, (int)floor(scale_height));
    const int width_lf       = max(0, (int)floor(scale_width));

    const int input_width_offset  = mul24(output_channel_block_idx, input_width);
    const int input_height_offset = mul24(output_batch_idx, input_height);

    float4 out =
        read_imagef(input, SAMPLER, (int2)(input_width_offset + width_lf, input_height_offset + height_lf));

    const int out_image_w = mad24(output_channel_block_idx, output_width, output_width_block_idx);
    const int out_image_h = mad24(output_batch_idx, out_height, output_height_idx);

    write_imagef(output, (int2)(out_image_w, out_image_h), out);
}
