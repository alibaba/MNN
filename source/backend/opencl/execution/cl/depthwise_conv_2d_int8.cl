#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

__kernel void depthwise_conv_2d(GLOBAL_SIZE_3_DIMS __global char* input_ptr, __global char* weights_ptr,
                      __global int* bias_ptr,
                      __global char* output_ptr,
                      __global float* scale_ptr,
                      __private const int2 input_shape,
                      __private const int in_channel_blocks,
                      __private const int2 output_shape,
                      __private const int2 weights_shape,
                      __private const int2 stride_shape,
                      __private const int2 padding_shape,
                      __private const int2 dilation_shape,
                      __private const int out_width_blocks,
                      __private const int out_channel_blocks) {

    const int out_c_b_idx = get_global_id(0);
    const int out_w_idx  = get_global_id(1);
    const int out_b_h_idx  = get_global_id(2);

    const int out_h_idx = out_b_h_idx % output_shape.x;
    DEAL_NON_UNIFORM_DIM3(out_c_b_idx, out_w_idx, out_b_h_idx);

    int4 out0 = vload4(out_c_b_idx, bias_ptr);

//deal with width size
    const int width_start = mad24(out_w_idx, stride_shape.y, -padding_shape.y);

//deal with height size
    const int height_start = mad24(out_h_idx, stride_shape.x, -padding_shape.x);

    int4 in0;
    int4 weights0, weights1, weights2, weights3;
    for (int iy = 0; iy < weights_shape.x; iy++) {
        for (int ix = 0; ix < weights_shape.y; ix++) {

            int in_h_idx = height_start + iy;
            int in_w_idx = width_start + ix;

            if(in_h_idx >= 0 && in_h_idx < input_shape.x && in_w_idx >= 0 && in_w_idx < input_shape.y){
                int in_idx = out_c_b_idx*input_shape.x*input_shape.y + in_h_idx * input_shape.y + in_w_idx;
                in0 = convert_int4_sat(vload4(in_idx, (__global char *)input_ptr));

                int weights_idx = iy * weights_shape.y*in_channel_blocks + ix*in_channel_blocks + out_c_b_idx;
                
                weights0 = convert_int4(vload4(weights_idx, (__global char *)weights_ptr));

                out0 = in0*weights0 + out0;

            }
        }
    }

#ifdef RELU
    out0 = max(out0, (int4)0);
#endif
    
    float4 scale = vload4(out_c_b_idx, (__global float*)scale_ptr);
    float4 out0_f = convert_float4_rtp(out0) * scale;

    char4 out0_c = convert_char4_sat(convert_int4_rte(out0_f));

    int out_idx = out_c_b_idx * output_shape.x * output_shape.y + out_h_idx*output_shape.y + out_w_idx;
    vstore4(out0_c, out_idx, (__global char*)output_ptr);
}
