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
#ifdef USE_ROUND
    const int height_lf      = min(max(0, (int)floor(scale_height + 0.499f)), input_height - 1);
    const int width_lf       = min(max(0, (int)floor(scale_width + 0.499f)), input_width - 1);
#else
    const int height_lf      = min(max(0, (int)floor(scale_height)), input_height - 1);
    const int width_lf       = min(max(0, (int)floor(scale_width)), input_width - 1);
#endif

    const int input_width_offset  = mul24(output_channel_block_idx, input_width);
    const int input_height_offset = mul24(output_batch_idx, input_height);

    float4 out =
        read_imagef(input, SAMPLER, (int2)(input_width_offset + width_lf, input_height_offset + height_lf));

    const int out_image_w = mad24(output_channel_block_idx, output_width, output_width_block_idx);
    const int out_image_h = mad24(output_batch_idx, out_height, output_height_idx);

    write_imagef(output, (int2)(out_image_w, out_image_h), out);
}

__kernel void interp3D(GLOBAL_SIZE_3_DIMS __read_only image2d_t input, __write_only image2d_t output,
                     __private const float depth_scale, __private const float height_scale, __private const float width_scale,
                     __private const float depth_offset, __private const float height_offset, __private const float width_offset,
                     __private const int input_depth, __private const int input_height, __private const int input_width,
                     __private const int out_depth, __private const int out_height) {
    const int output_channel_block_idx      = get_global_id(0);
    const int output_height_width_block_idx = get_global_id(1);
    const int output_batch_depth_block_idx  = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(output_channel_block_idx, output_height_width_block_idx, output_batch_depth_block_idx);
    const int output_channel_block_idxs = global_size_dim0;
    const int output_tensor_height_width = global_size_dim1;

    const int out_width = output_tensor_height_width / out_height;
    const int output_batch_idx  = output_batch_depth_block_idx / out_depth;
    const int output_depth_idx  = output_batch_depth_block_idx % out_depth;
    const int output_height_idx = output_height_width_block_idx / out_height;
    const int output_width_idx  = output_height_width_block_idx % out_height;

    const float scale_depth  = output_depth_idx * depth_scale + depth_offset;
    const float scale_height = output_height_idx * height_scale + height_offset;
    const float scale_width  = output_width_idx * width_scale + width_offset;
    const int depth_lf       = max(0, (int)floor(scale_depth));
    const int height_lf      = max(0, (int)floor(scale_height));
    const int width_lf       = max(0, (int)floor(scale_width));


    const int input_tensor_width_height = mul24(input_width, input_height);
    const int input_image_width_offset  = mul24(output_channel_block_idx, input_tensor_width_height);
    const int input_image_height_offset = mul24(output_batch_idx, input_depth);

    float4 out = read_imagef(input, SAMPLER,
                             (int2)(input_image_width_offset + input_width * (height_offset + height_lf) + width_lf + width_offset,
                                    input_image_height_offset + depth_lf + depth_offset));

    const int output_image_width_offset  = output_channel_block_idx * output_tensor_height_width;
    const int output_image_height_offset = output_batch_idx * out_depth;

    // TODO: out
    const int out_image_w = output_image_width_offset + output_height_idx * out_width + output_width_idx;
    const int out_image_h = output_image_height_offset + output_batch_idx * out_depth + output_depth_idx;
    write_imagef(output, (int2)(out_image_w, out_image_h), out);
}
