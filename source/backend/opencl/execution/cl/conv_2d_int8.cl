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


__kernel void conv_2d1x1(GLOBAL_SIZE_3_DIMS __global char* input_ptr, __global char* weights_ptr,
                      __global int* bias_ptr,
                      __global char* output_ptr,
                      __global float* scale_ptr,
                      __private const int input_width,
                      __private const int input_size,
                      __private const int in_channel_blocks,
                      __private const int out_channel_blocks) {
    
    const int out_c_b_idx = get_global_id(0);
    const int out_w_idx  = get_global_id(1);
    const int out_h_idx  = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(out_c_b_idx, out_w_idx, out_h_idx);

    const int out_w4_idx = mul24(out_w_idx, 4);

    int4 out0 = vload4(out_c_b_idx, bias_ptr);

    int16 out = {out0, out0, out0, out0};
    int16 in;

    int in_h_w_idx = mad24(out_h_idx, input_width, out_w4_idx);
    int16 weights;
    for (int in_c_b_idx = 0; in_c_b_idx < in_channel_blocks; in_c_b_idx++) {

        __private int in_idx = mad24(in_c_b_idx, input_size, in_h_w_idx)*4;

        in = convert_int16(vload16(0, input_ptr + in_idx));

        int weights_idx = mul24(mad24(in_c_b_idx, out_channel_blocks, out_c_b_idx),16);

        weights = convert_int16(vload16(0, weights_ptr + weights_idx));

        out.s0123 = mad24(in.s0, weights.s0123, out.s0123);
        out.s4567 = mad24(in.s4, weights.s0123, out.s4567);
        out.s89ab = mad24(in.s8, weights.s0123, out.s89ab);
        out.scdef = mad24(in.sc, weights.s0123, out.scdef);

        out.s0123 = mad24(in.s1, weights.s4567, out.s0123);
        out.s4567 = mad24(in.s5, weights.s4567, out.s4567);
        out.s89ab = mad24(in.s9, weights.s4567, out.s89ab);
        out.scdef = mad24(in.sd, weights.s4567, out.scdef);

        out.s0123 = mad24(in.s2, weights.s89ab, out.s0123);
        out.s4567 = mad24(in.s6, weights.s89ab, out.s4567);
        out.s89ab = mad24(in.sa, weights.s89ab, out.s89ab);
        out.scdef = mad24(in.se, weights.s89ab, out.scdef);

        out.s0123 = mad24(in.s3, weights.scdef, out.s0123);
        out.s4567 = mad24(in.s7, weights.scdef, out.s4567);
        out.s89ab = mad24(in.sb, weights.scdef, out.s89ab);
        out.scdef = mad24(in.sf, weights.scdef, out.scdef);

    }

#ifdef RELU

    out = max(out, (int16)0);

#endif
    
    float4 scale = vload4(out_c_b_idx, (__global float*)scale_ptr);
    float16 scale16 = {scale, scale, scale, scale};
    int out_idx = mad24(out_c_b_idx, input_size, in_h_w_idx);

    const int remain = input_width - out_w4_idx;

    if (remain >= 4) {

        float16 out_f = convert_float16_rtp(out) * scale16;

        char16 out_c = convert_char16_sat(convert_int16_rte(out_f));

        vstore16(out_c, 0, output_ptr + out_idx*4);

    } else if (remain == 3) {
        float4 out0_f = convert_float4_rtp(out.s0123) * scale;
        float4 out1_f = convert_float4_rtp(out.s4567) * scale;
        float4 out2_f = convert_float4_rtp(out.s89ab) * scale;

        char4 out0_c = convert_char4_sat(convert_int4_rte(out0_f));
        char4 out1_c = convert_char4_sat(convert_int4_rte(out1_f));
        char4 out2_c = convert_char4_sat(convert_int4_rte(out2_f));

        vstore4(out0_c, out_idx, (__global char*)output_ptr);
        vstore4(out1_c, out_idx+1, (__global char*)output_ptr);
        vstore4(out2_c, out_idx+2, (__global char*)output_ptr);
    } else if (remain == 2) {
        float4 out0_f = convert_float4_rtp(out.s0123) * scale;
        float4 out1_f = convert_float4_rtp(out.s4567) * scale;

        char4 out0_c = convert_char4_sat(convert_int4_rte(out0_f));
        char4 out1_c = convert_char4_sat(convert_int4_rte(out1_f));

        vstore4(out0_c, out_idx, (__global char*)output_ptr);
        vstore4(out1_c, out_idx+1, (__global char*)output_ptr);
        
    } else if (remain == 1) {
        float4 out0_f = convert_float4_rtp(out.s0123) * scale;
        char4 out0_c = convert_char4_sat(convert_int4_rte(out0_f));
        vstore4(out0_c, out_idx, (__global char*)output_ptr);
    }


}

__kernel void conv_2d(GLOBAL_SIZE_3_DIMS __global char* input_ptr, __global char* weights_ptr,
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

    const int out_w4_idx = mul24(out_w_idx, 4);

    int4 out0 = vload4(out_c_b_idx, bias_ptr);
    int16 out = {out0, out0, out0, out0};

//deal with width size
    const int width_start0 = mad24(out_w4_idx, stride_shape.y, -padding_shape.y);
    const int width_start1 = mad24(out_w4_idx+1, stride_shape.y, -padding_shape.y);
    const int width_start2 = mad24(out_w4_idx+2, stride_shape.y, -padding_shape.y);
    const int width_start3 = mad24(out_w4_idx+3, stride_shape.y, -padding_shape.y);

//deal with height size
    const int height_start = mad24(out_h_idx, stride_shape.x, -padding_shape.x);

    int4 in0, in1, in2, in3;
    int16 weights;
    for (int in_c_b_idx = 0; in_c_b_idx < in_channel_blocks; in_c_b_idx++) {
        for (int iy = 0; iy < weights_shape.x; iy++) {
            for (int ix = 0; ix < weights_shape.y; ix++) {

                int in_h_idx = height_start + iy;
                int in_w_idx0 = width_start0 + ix;
                int in_w_idx1 = width_start1 + ix;
                int in_w_idx2 = width_start2 + ix;
                int in_w_idx3 = width_start3 + ix;

                if(in_h_idx >= 0 && in_h_idx < input_shape.x && in_w_idx0 >= 0 && in_w_idx0 < input_shape.y && in_w_idx1 >= 0 && in_w_idx1 < input_shape.y && in_w_idx2 >= 0 && in_w_idx3 < input_shape.y && in_w_idx3 >= 0 && in_w_idx3 < input_shape.y){
                    int in_idx0 = in_c_b_idx*input_shape.x*input_shape.y + in_h_idx * input_shape.y + in_w_idx0;
                    int in_idx1 = in_c_b_idx*input_shape.x*input_shape.y + in_h_idx * input_shape.y + in_w_idx1;
                    int in_idx2 = in_c_b_idx*input_shape.x*input_shape.y + in_h_idx * input_shape.y + in_w_idx2;
                    int in_idx3 = in_c_b_idx*input_shape.x*input_shape.y + in_h_idx * input_shape.y + in_w_idx3;

                    in0 = convert_int4_sat(vload4(in_idx0, (__global char *)input_ptr));
                    in1 = convert_int4_sat(vload4(in_idx1, (__global char *)input_ptr));
                    in2 = convert_int4_sat(vload4(in_idx2, (__global char *)input_ptr));
                    in3 = convert_int4_sat(vload4(in_idx3, (__global char *)input_ptr));

                    int weights_idx = (iy * weights_shape.y*in_channel_blocks*out_channel_blocks +
                                      ix*in_channel_blocks*out_channel_blocks +
                                        in_c_b_idx*out_channel_blocks + out_c_b_idx)*16;

                    weights = convert_int16(vload16(0, weights_ptr + weights_idx));

                    out.s0123 = mad24(in0.x, weights.s0123, out.s0123);
                    out.s4567 = mad24(in1.x, weights.s0123, out.s4567);
                    out.s89ab = mad24(in2.x, weights.s0123, out.s89ab);
                    out.scdef = mad24(in3.x, weights.s0123, out.scdef);

                    out.s0123 = mad24(in0.y, weights.s4567, out.s0123);
                    out.s4567 = mad24(in1.y, weights.s4567, out.s4567);
                    out.s89ab = mad24(in2.y, weights.s4567, out.s89ab);
                    out.scdef = mad24(in3.y, weights.s4567, out.scdef);

                    out.s0123 = mad24(in0.z, weights.s89ab, out.s0123);
                    out.s4567 = mad24(in1.z, weights.s89ab, out.s4567);
                    out.s89ab = mad24(in2.z, weights.s89ab, out.s89ab);
                    out.scdef = mad24(in3.z, weights.s89ab, out.scdef);

                    out.s0123 = mad24(in0.w, weights.scdef, out.s0123);
                    out.s4567 = mad24(in1.w, weights.scdef, out.s4567);
                    out.s89ab = mad24(in2.w, weights.scdef, out.s89ab);
                    out.scdef = mad24(in3.w, weights.scdef, out.scdef);

                }
            }
        }
    }
#ifdef RELU

    out = max(out, (int16)0);

#endif
    
    float4 scale = vload4(out_c_b_idx, (__global float*)scale_ptr);
    float16 scale16 = {scale, scale, scale, scale};

    const int remain = output_shape.y - out_w4_idx;
    int out_idx = out_c_b_idx * output_shape.x * output_shape.y + out_h_idx*output_shape.y + out_w4_idx;
    if (remain >= 4) {

        float16 out_f = convert_float16_rtp(out) * scale16;

        char16 out_c = convert_char16_sat(convert_int16_rte(out_f));

        vstore16(out_c, 0, output_ptr + out_idx*4);

    } else if (remain == 3) {
        float4 out0_f = convert_float4_rtp(out.s0123) * scale;
        float4 out1_f = convert_float4_rtp(out.s4567) * scale;
        float4 out2_f = convert_float4_rtp(out.s89ab) * scale;

        char4 out0_c = convert_char4_sat(convert_int4_rte(out0_f));
        char4 out1_c = convert_char4_sat(convert_int4_rte(out1_f));
        char4 out2_c = convert_char4_sat(convert_int4_rte(out2_f));

        vstore4(out0_c, out_idx, (__global char*)output_ptr);
        vstore4(out1_c, out_idx+1, (__global char*)output_ptr);
        vstore4(out2_c, out_idx+2, (__global char*)output_ptr);
    } else if (remain == 2) {
        float4 out0_f = convert_float4_rtp(out.s0123) * scale;
        float4 out1_f = convert_float4_rtp(out.s4567) * scale;

        char4 out0_c = convert_char4_sat(convert_int4_rte(out0_f));
        char4 out1_c = convert_char4_sat(convert_int4_rte(out1_f));

        vstore4(out0_c, out_idx, (__global char*)output_ptr);
        vstore4(out1_c, out_idx+1, (__global char*)output_ptr);
        
    } else if (remain == 1) {
        float4 out0_f = convert_float4_rtp(out.s0123) * scale;
        char4 out0_c = convert_char4_sat(convert_int4_rte(out0_f));
        vstore4(out0_c, out_idx, (__global char*)output_ptr);
    }
}
