#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

__kernel void nearest_buf(GLOBAL_SIZE_3_DIMS __global const FLOAT* input,
                      __global FLOAT* output,
                      __private const float height_scale,
                      __private const float width_scale,
                      __private const float height_offset,
                      __private const float width_offset,
                      __private const int input_height,
                      __private const int input_width,
                      __private const int out_height,
                      __private const int out_width,
                      __private const int channelBlocks) {
    const int output_channel_block_idx      = get_global_id(0);
    const int output_width_block_idx        = get_global_id(1);
    const int output_batch_height_block_idx = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(output_channel_block_idx, output_width_block_idx, output_batch_height_block_idx);

    const int output_batch_idx  = output_batch_height_block_idx / out_height;
    const int output_height_idx = output_batch_height_block_idx % out_height;

    const float in_h_idx = output_height_idx * height_scale + height_offset;
    const float in_w_idx = output_width_block_idx * width_scale + width_offset;
#ifdef USE_ROUND
    const int in_h_index      = min(max(0, (int)floor(in_h_idx + 0.499f)), input_height-1);
    const int in_w_index       = min(max(0, (int)floor(in_w_idx + 0.499f)), input_width-1);
#else
    const int in_h_index      = min(max(0, (int)floor(in_h_idx)), input_height-1);
    const int in_w_index       = min(max(0, (int)floor(in_w_idx)), input_width-1);
#endif

    const int inp_offset = ((output_batch_idx * channelBlocks + output_channel_block_idx) * input_height + in_h_index) * input_width + in_w_index;
    FLOAT4 value = vload4(inp_offset, input);

    const int out_offset = ((output_batch_idx * channelBlocks + output_channel_block_idx) * out_height + output_height_idx) * out_width + output_width_block_idx;
    vstore4(value, out_offset, output);
}

__kernel void bilinear_buf(GLOBAL_SIZE_3_DIMS __global const FLOAT* input,
                            __global FLOAT* output,
                            __private const float height_scale,
                            __private const float width_scale,
                            __private const float height_offset,
                            __private const float width_offset,
                            __private const int input_height,
                            __private const int input_width,
                            __private const int out_height,
                            __private const int out_width,
                            __private const int channelBlocks) {
    const int output_channel_block_idx      = get_global_id(0);
    const int output_width_block_idx        = get_global_id(1);
    const int output_batch_height_block_idx = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(output_channel_block_idx, output_width_block_idx, output_batch_height_block_idx);
    
    const int output_batch_idx  = output_batch_height_block_idx / out_height;
    const int output_height_idx = output_batch_height_block_idx % out_height;

    const float in_h_idx = output_height_idx * height_scale + height_offset;
    const float in_w_idx = output_width_block_idx * width_scale + width_offset;
    const int in_h0_index      = min(max(0, (int)floor(in_h_idx)), input_height-1);
    const int in_w0_index      = min(max(0, (int)floor(in_w_idx)), input_width-1);
    const int in_h1_index      = min(max(0, (int)floor(in_h_idx)+1), input_height-1);
    const int in_w1_index      = min(max(0, (int)floor(in_w_idx)+1), input_width-1);
    
    float factor_w = (in_w_idx - (int)floor(in_w_idx));
    float factor_h = (in_h_idx - (int)floor(in_h_idx));
    
    const int inp_offset_base = (output_batch_idx * channelBlocks + output_channel_block_idx) * input_height;
    const int inp_offset_00 = (inp_offset_base + in_h0_index) * input_width + in_w0_index;
    const int inp_offset_01 = (inp_offset_base + in_h0_index) * input_width + in_w1_index;
    const int inp_offset_10 = (inp_offset_base + in_h1_index) * input_width + in_w0_index;
    const int inp_offset_11 = (inp_offset_base + in_h1_index) * input_width + in_w1_index;

    FLOAT4 value_00 = vload4(inp_offset_00, input);
    FLOAT4 value_01 = vload4(inp_offset_01, input);
    FLOAT4 value_10 = vload4(inp_offset_10, input);
    FLOAT4 value_11 = vload4(inp_offset_11, input);

    FLOAT4 value = CONVERT_FLOAT4((float4)((1.0-factor_w)*(1.0-factor_h))*convert_float4(value_00) + (float4)(factor_w*(1.0-factor_h))*convert_float4(value_01) + (float4)((1.0-factor_w)*factor_h)*convert_float4(value_10) + (float4)(factor_w*factor_h)*convert_float4(value_11));
    
    const int out_offset = ((output_batch_idx * channelBlocks + output_channel_block_idx) * out_height + output_height_idx) * out_width + output_width_block_idx;
    
    vstore4(value, out_offset, output);
}

__kernel void nearest3D_buf(GLOBAL_SIZE_3_DIMS __global const FLOAT* input,
        __global FLOAT* output,
        __private const float depth_scale,
        __private const float height_scale,
        __private const float width_scale,
        __private const float depth_offset,
        __private const float height_offset,
        __private const float width_offset,
        __private const int input_depth,
        __private const int input_height,
        __private const int input_width,
        __private const int out_depth,
        __private const int out_height,
        __private const int out_width,
        __private const int channelBlocks) {
    const int output_channel_block_idx      = get_global_id(0);
    const int output_height_width_block_idx = get_global_id(1);
    const int output_batch_depth_block_idx  = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(output_channel_block_idx, output_height_width_block_idx, output_batch_depth_block_idx);


    const int output_batch_idx  = output_batch_depth_block_idx / out_depth;
    const int output_depth_idx  = output_batch_depth_block_idx % out_depth;
    const int output_height_idx = output_height_width_block_idx / out_height;
    const int output_width_idx  = output_height_width_block_idx % out_height;

    const float in_d_idx = output_depth_idx * depth_scale + depth_offset;
    const float in_h_idx = output_height_idx * height_scale + height_offset;
    const float in_w_idx = output_width_idx * width_scale + width_offset;
    const int in_d_index      = min(max(0, (int)floor(in_d_idx)), input_depth-1);
    const int in_h_index      = min(max(0, (int)floor(in_h_idx)), input_height-1);
    const int in_w_index       = min(max(0, (int)floor(in_w_idx)), input_width-1);

    const int inp_offset = (((output_batch_idx * channelBlocks + output_channel_block_idx)
            * input_depth + in_d_index) * input_height + in_h_index) * input_width + in_w_index;

    const int out_offset = (((output_batch_idx * channelBlocks + output_channel_block_idx)
            * out_depth + output_depth_idx) * out_height + output_height_idx) * out_width + output_width_idx;
    FLOAT4 value = vload4(inp_offset, input);
    vstore4(value, out_offset, output);
}