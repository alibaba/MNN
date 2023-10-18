#define READ_INPUT_IMAGE(i, base)                                                                         \
    int in_width_value##i = in_width##i + base;                                                           \
    in_width_value##i =                                                                                   \
        select(in_idx + in_width_value##i, -1, (in_width_value##i < 0 || in_width_value##i >= input_shape.y)); \
    in##i = read_imagef(input, SAMPLER, (int2)(in_width_value##i, in_hb_value));

#define CALCULATE_OUTPUT(i)                  \
    out##i = mad(in##i.x, weights0, out##i); \
    out##i = mad(in##i.y, weights1, out##i); \
    out##i = mad(in##i.z, weights2, out##i); \
    out##i = mad(in##i.w, weights3, out##i);
#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }
#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;


__kernel void depthwise_deconv2d(GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
                                 __read_only image2d_t weights,
                                 #ifndef NO_BIAS
                                 __read_only image2d_t bias,
                                 #endif
                                 __write_only image2d_t output,
                                 __private const int2 input_shape,
                                 __private const int2 output_shape,
                                 __private const int2 stride_shape,
                                 __private const int2 align_shape,
                                 __private const int2 padding_shape,
                                 __private const int2 kernel_shape, 
                                 __private const int kernel_size, __private const int out_channel_blocks) {
    const int out_channel_blocks_idx = get_global_id(0);
    const int out_width_idx          = get_global_id(1);
    const int out_batch_height_idx   = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(out_channel_blocks_idx, out_width_idx, out_batch_height_idx);
    #ifndef NO_BIAS
    float4 out0 = read_imagef(bias, SAMPLER, (int2)(out_channel_blocks_idx, 0));
    #else
    float4 out0 = (float4)(0.0);
    #endif

    const int out_batch_idx  = out_batch_height_idx / output_shape.x;
    const int out_height_idx = out_batch_height_idx % output_shape.x;

    int kernel_start_x = (out_width_idx + align_shape.y) / stride_shape.y;
    int kernel_start_y = (out_height_idx + align_shape.x) / stride_shape.x;

    int deal_kernel_width  = kernel_shape.y - mad24(kernel_start_x, stride_shape.y, padding_shape.y) + out_width_idx - 1;
    int deal_kernel_height = kernel_shape.x - mad24(kernel_start_y, stride_shape.x, padding_shape.x) + out_height_idx - 1;

    int kernel_image_x;
    float4 in0;
    float4 weight;
    int in_width0;
    int in_idx, in_idy;
    for (int k_y = deal_kernel_height, idx_h = kernel_start_y; k_y >= 0; k_y -= stride_shape.x, idx_h++) {
        in_idy          = mad24(out_batch_idx, input_shape.x, idx_h);
        int in_hb_value = select(in_idy, -1, idx_h < 0 || idx_h >= input_shape.x);
        for (int k_x = deal_kernel_width, in_width_idx = kernel_start_x; k_x >= 0; k_x -= stride_shape.y, in_width_idx++) {
            in_width0 = in_width_idx;

            in_idx = mul24(out_channel_blocks_idx, input_shape.y);
            READ_INPUT_IMAGE(0, 0);

            kernel_image_x = mad24(k_y, kernel_shape.y, k_x);
            weight         = read_imagef(weights, SAMPLER, (int2)(kernel_image_x, out_channel_blocks_idx));
            out0           = mad(in0, weight, out0);
        }

#ifdef RELU
        out0 = fmax(out0, (float4)0);
#endif

#ifdef RELU6
        out0 = clamp(out0, (float4)0, (float4)6);
#endif

        const int output_image_x = mad24(out_channel_blocks_idx, output_shape.y, out_width_idx);
        write_imagef(output, (int2)(output_image_x, out_batch_height_idx), out0);
    }
}
