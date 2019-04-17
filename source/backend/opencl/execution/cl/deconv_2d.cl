#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void deconv_2d(GLOBAL_SIZE_3_DIMS __read_only image2d_t input, __read_only image2d_t weights,
                        __read_only image2d_t bias,
                        __write_only image2d_t output,
                        __private const int2 input_shape,
                        __private const int2 output_shape,
                        __private const int2 stride_shape,
                        __private const int2 align_shape,
                        __private const int2 padding_shape, 
                        __private const int2 kernel_shape,
                        __private const int kernel_size,
                        __private const int in_channel_blocks, __private const int out_channel_blocks) {

    const int out_channel_blocks_idx = get_global_id(0);
    const int out_width_idx          = get_global_id(1);
    const int out_batch_height_idx   = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(out_channel_blocks_idx, out_width_idx, out_batch_height_idx);

    float4 out0 = read_imagef(bias, SAMPLER, (int2)(out_channel_blocks_idx, 0));

    const int out_batch_idx  = out_batch_height_idx / output_shape.x;
    const int out_height_idx = out_batch_height_idx % output_shape.x;

    int kernel_start_x = (out_width_idx + align_shape.y) / stride_shape.y;
    int kernel_start_y = max(0, (out_height_idx + align_shape.x) / stride_shape.x);

    int deal_kernel_width  = kernel_shape.y - mad24(kernel_start_x, stride_shape.y, padding_shape.y) + out_width_idx - 1;
    int deal_kernel_height = kernel_shape.x - mad24(kernel_start_y, stride_shape.x, padding_shape.x) + out_height_idx - 1;

    int kernel_x_0, kernel_x_1, kernel_x_2, kernel_x_3, kernel_y;
    float4 in0;
    float4 weights0, weights1, weights2, weights3;
    for (int ic = 0; ic < in_channel_blocks; ic++) {
        kernel_x_0 = ic << 2;
        kernel_x_1 = kernel_x_0 + 1;
        kernel_x_2 = kernel_x_0 + 2;
        kernel_x_3 = kernel_x_0 + 3;
        for (int k_y = deal_kernel_height, idx_h = kernel_start_y; k_y >= 0; k_y -= stride_shape.x, idx_h++) {
            int in_idy      = mad24(out_batch_idx, input_shape.x, idx_h);
            int in_hb_value = select(in_idy, -1, idx_h < 0 || idx_h >= input_shape.x);
            int in_width0   = kernel_start_x;
            for (int k_x = deal_kernel_width; k_x >= 0; k_x -= stride_shape.y) {
                kernel_y = mad24(k_y, kernel_shape.y, k_x);
                kernel_y = mad24(out_channel_blocks_idx, kernel_size, kernel_y);
                weights0 = read_imagef(weights, SAMPLER, (int2)(kernel_x_0, kernel_y));
                weights1 = read_imagef(weights, SAMPLER, (int2)(kernel_x_1, kernel_y));
                weights2 = read_imagef(weights, SAMPLER, (int2)(kernel_x_2, kernel_y));
                weights3 = read_imagef(weights, SAMPLER, (int2)(kernel_x_3, kernel_y));

                int in_idx = mul24(ic, input_shape.y);
                int in_width_value0 = in_width0;                                                           \
                in_width_value0 =                                                                                   \
                    select(in_idx + in_width_value0, -1, (in_width_value0 < 0 || in_width_value0 >= input_shape.y)); \
                in0 = read_imagef(input, SAMPLER, (int2)(in_width_value0, in_hb_value));

                out0 = mad(in0.x, weights0, out0);
                out0 = mad(in0.y, weights1, out0);
                out0 = mad(in0.z, weights2, out0);
                out0 = mad(in0.w, weights3, out0);
                in_width0++;
            }
        }

#ifdef RELU
        out0 = fmax(out0, (float4)0);
#endif

#ifdef RELU6
        out0 = clamp(out0, (float4)0, (float4)6);
#endif

        int out_image_width_idx = mad24(out_channel_blocks_idx, output_shape.y, out_width_idx);
        write_imagef(output, (int2)(out_image_width_idx, out_batch_height_idx), out0);
    }
}
