#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,
#define DEAL_NON_UNIFORM_DIM2(input1, input2)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                     \
    }

// convert kernel : from buffer(oihw) to image(ic, oc/4, h, w, oc4)
__kernel void conv2d_filter_buffer_to_nc4hw4_buffer(GLOBAL_SIZE_2_DIMS
                                            __global const FLOAT *input_ptr,
                                            __private const int output_channel,
                                            __private const int2 kernel_shape,
                                            __private const int ic_h_w_size,
                                            __private const int height_width_size,
                                            __global FLOAT *output) {
    int image_width_idx  = get_global_id(0); // ic
    int image_height_idx = get_global_id(1); // oc/4 h w

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int input_channel_4_idx  = image_width_idx;
    const int output_channel_4_idx = (image_height_idx / height_width_size) * 4;
    const int height_width_idx     = image_height_idx % height_width_size;
    const int buffer_height_idx    = height_width_idx / kernel_shape.y;
    const int buffer_width_idx     = height_width_idx % kernel_shape.y;

    const int buffer_offset = output_channel_4_idx * ic_h_w_size + input_channel_4_idx * height_width_size +
                              buffer_height_idx * kernel_shape.y + buffer_width_idx;

    FLOAT4 output_values = 0;
    if (output_channel_4_idx < output_channel) {
        const int remain_channel = output_channel - output_channel_4_idx;
        if (remain_channel >= 4) {
            int offset      = buffer_offset;
            output_values.x = (FLOAT)(*(input_ptr + offset));
            offset          = mad24(1, ic_h_w_size, offset);
            output_values.y = (FLOAT)(*(input_ptr + offset));
            offset += ic_h_w_size;
            output_values.z = (FLOAT)(*(input_ptr + offset));
            offset += ic_h_w_size;
            output_values.w = (FLOAT)(*(input_ptr + offset));
        } else if (remain_channel == 3) {
            int offset      = buffer_offset;
            output_values.x = (FLOAT)(*(input_ptr + offset));
            offset          = mad24(1, ic_h_w_size, offset);
            output_values.y = (FLOAT)(*(input_ptr + offset));
            offset += ic_h_w_size;
            output_values.z = (FLOAT)(*(input_ptr + offset));

        } else if (remain_channel == 2) {
            int offset      = buffer_offset;
            output_values.x = (FLOAT)(*(input_ptr + offset));
            offset          = mad24(1, ic_h_w_size, offset);
            output_values.y = (FLOAT)(*(input_ptr + offset));
        } else if (remain_channel == 1) {
            int offset      = buffer_offset;
            output_values.x = (FLOAT)(*(input_ptr + offset));
        }
    }
    const int out_offset = (image_width_idx*height_width_size*((output_channel+3)/4)+image_height_idx)*4;
    vstore4(output_values, 0, output+out_offset);
}

// convert kernel : from buffer(oihw) to image(oc/4 h w , ic oc4)
__kernel void conv2d_filter_buffer_to_nc4hw4_buffer_floatin(GLOBAL_SIZE_2_DIMS
                                            __global const float *input_ptr,
                                            __private const int output_channel,
                                            __private const int2 kernel_shape,
                                            __private const int ic_h_w_size,
                                            __private const int height_width_size,
                                            __global FLOAT *output) {
    int image_width_idx  = get_global_id(0); // ic
    int image_height_idx = get_global_id(1); // oc/4 h w

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    const int input_channel_4_idx  = image_width_idx;
    const int output_channel_4_idx = (image_height_idx / height_width_size) * 4;
    const int height_width_idx     = image_height_idx % height_width_size;
    const int buffer_height_idx    = height_width_idx / kernel_shape.y;
    const int buffer_width_idx     = height_width_idx % kernel_shape.y;

    const int buffer_offset = output_channel_4_idx * ic_h_w_size + input_channel_4_idx * height_width_size +
                              buffer_height_idx * kernel_shape.y + buffer_width_idx;

    FLOAT4 output_values = 0;
    if (output_channel_4_idx < output_channel) {
        const int remain_channel = output_channel - output_channel_4_idx;
        if (remain_channel >= 4) {
            int offset      = buffer_offset;
            output_values.x = (FLOAT)(*(input_ptr + offset));
            offset          = mad24(1, ic_h_w_size, offset);
            output_values.y = (FLOAT)(*(input_ptr + offset));
            offset += ic_h_w_size;
            output_values.z = (FLOAT)(*(input_ptr + offset));
            offset += ic_h_w_size;
            output_values.w = (FLOAT)(*(input_ptr + offset));
        } else if (remain_channel == 3) {
            int offset      = buffer_offset;
            output_values.x = (FLOAT)(*(input_ptr + offset));
            offset          = mad24(1, ic_h_w_size, offset);
            output_values.y = (FLOAT)(*(input_ptr + offset));
            offset += ic_h_w_size;
            output_values.z = (FLOAT)(*(input_ptr + offset));

        } else if (remain_channel == 2) {
            int offset      = buffer_offset;
            output_values.x = (FLOAT)(*(input_ptr + offset));
            offset          = mad24(1, ic_h_w_size, offset);
            output_values.y = (FLOAT)(*(input_ptr + offset));
        } else if (remain_channel == 1) {
            int offset      = buffer_offset;
            output_values.x = (FLOAT)(*(input_ptr + offset));
        }
    }
    const int out_offset = (image_width_idx*height_width_size*((output_channel+3)/4)+image_height_idx)*4;
    vstore4(output_values, 0, output+out_offset);
}


// convert kernel from buffer(mihw) to image(ic/4, ic4 h w m)
// but now dw only support m == 1
__kernel void dw_filter_buffer_to_nc4hw4_buffer(GLOBAL_SIZE_2_DIMS
                                        __global const FLOAT *input_ptr,
                                        __private const int4 kernel_shape,//[1, Cout, fh, fw]
                                        __private const int height_width_size,
                                        __global FLOAT *output) {
    const int image_width_idx  = get_global_id(0);//fh*fw
    const int image_height_idx = get_global_id(1);//UP_DIV(Cout, 4)

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    FLOAT4 output_values = 0;
    if (kernel_shape.x == 1) {
        const int input_channel_4_idx = image_height_idx * 4;
        const int buffer_height_idx   = image_width_idx / kernel_shape.w;
        const int buffer_width_idx    = image_width_idx % kernel_shape.w;

        const int buffer_offset =
            mad24(mad24(input_channel_4_idx, kernel_shape.z, buffer_height_idx), kernel_shape.w, buffer_width_idx);

        //input [1, Cout,                fh,                 fw]
        //index:[0, input_channel_4_idx, buffer_height_idx,  buffer_width_idx]
        const int remain_channel = kernel_shape.y - input_channel_4_idx;
        if (input_channel_4_idx < kernel_shape.y) {
            if (remain_channel >= 4) {
                int offset      = buffer_offset;
                output_values.x = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.y = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.z = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.w = (FLOAT)(*(input_ptr + offset));
            } else if (remain_channel == 3) {
                int offset      = buffer_offset;
                output_values.x = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.y = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.z = (FLOAT)(*(input_ptr + offset));

            } else if (remain_channel == 2) {
                int offset      = buffer_offset;
                output_values.x = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.y = (FLOAT)(*(input_ptr + offset));
            } else if (remain_channel == 1) {
                int offset      = buffer_offset;
                output_values.x = (FLOAT)(*(input_ptr + offset));
            }
        }
    }

    //output NC4HW4 [1, fw*fh,            1, Cout/4]x oc4
    //index:        [0, image_width_idx,  0, image_height_idx]
    const int out_offset = (image_width_idx*((kernel_shape.y+3)/4)+image_height_idx)*4;
    vstore4(output_values, 0, output+out_offset);
}

__kernel void dw_filter_buffer_to_nc4hw4_buffer_floatin(GLOBAL_SIZE_2_DIMS
                                        __global const float *input_ptr,
                                        __private const int4 kernel_shape,//[1, Cout, fh, fw]
                                        __private const int height_width_size,
                                        __global FLOAT *output) {
    const int image_width_idx  = get_global_id(0);//fh*fw
    const int image_height_idx = get_global_id(1);//UP_DIV(Cout, 4)

    DEAL_NON_UNIFORM_DIM2(image_width_idx, image_height_idx);

    FLOAT4 output_values = 0;
    if (kernel_shape.x == 1) {
        const int input_channel_4_idx = image_height_idx * 4;
        const int buffer_height_idx   = image_width_idx / kernel_shape.w;
        const int buffer_width_idx    = image_width_idx % kernel_shape.w;

        const int buffer_offset =
            mad24(mad24(input_channel_4_idx, kernel_shape.z, buffer_height_idx), kernel_shape.w, buffer_width_idx);

        //input [1, Cout,                fh,                 fw]
        //index:[0, input_channel_4_idx, buffer_height_idx,  buffer_width_idx]
        const int remain_channel = kernel_shape.y - input_channel_4_idx;
        if (input_channel_4_idx < kernel_shape.y) {
            if (remain_channel >= 4) {
                int offset      = buffer_offset;
                output_values.x = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.y = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.z = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.w = (FLOAT)(*(input_ptr + offset));
            } else if (remain_channel == 3) {
                int offset      = buffer_offset;
                output_values.x = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.y = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.z = (FLOAT)(*(input_ptr + offset));

            } else if (remain_channel == 2) {
                int offset      = buffer_offset;
                output_values.x = (FLOAT)(*(input_ptr + offset));
                offset += height_width_size;
                output_values.y = (FLOAT)(*(input_ptr + offset));
            } else if (remain_channel == 1) {
                int offset      = buffer_offset;
                output_values.x = (FLOAT)(*(input_ptr + offset));
            }
        }
    }

    //output NC4HW4 [1, fw*fh,            1, Cout/4]x oc4
    //index:        [0, image_width_idx,  0, image_height_idx]
    const int out_offset = (image_width_idx*((kernel_shape.y+3)/4)+image_height_idx)*4;
    vstore4(output_values, 0, output+out_offset);
}
