#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define EXP exp
#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;


__kernel void softmax_channel(GLOBAL_SIZE_3_DIMS __read_only image2d_t input, __write_only image2d_t output, __private const int output_channels,
                              __private const int remain_channels, __private const int4 shape // NCHW
                              ) {

    const int width_idx    = get_global_id(0);
    const int batch_height_idx       = get_global_id(1);

    
    if (width_idx < shape.w && batch_height_idx < shape.x*shape.z) {

        FLOAT4 float_max_value = (FLOAT4)-FLT_MAX;
        FLOAT4 input_data;
        for (short i = 0; i < shape.y - 1; ++i) {
            input_data      = RI_F(input, SAMPLER, (int2)(width_idx + i * shape.w, batch_height_idx));
            float_max_value = max(float_max_value, input_data);
        }
        float_max_value.x = max(float_max_value.x, float_max_value.y);
        float_max_value.x = max(float_max_value.x, float_max_value.z);
        float_max_value.x = max(float_max_value.x, float_max_value.w);

        input_data = RI_F(input, SAMPLER, (int2)(width_idx + (shape.y - 1) * shape.w , batch_height_idx));
        if (remain_channels == 0) {
            float_max_value.x = max(float_max_value.x, input_data.x);
            float_max_value.x = max(float_max_value.x, input_data.y);
            float_max_value.x = max(float_max_value.x, input_data.z);
            float_max_value.x = max(float_max_value.x, input_data.w);
        } else if (remain_channels == 1) {
            float_max_value.x = max(float_max_value.x, input_data.z);
            float_max_value.x = max(float_max_value.x, input_data.y);
            float_max_value.x = max(float_max_value.x, input_data.x);
        } else if (remain_channels == 2) {
            float_max_value.x = max(float_max_value.x, input_data.y);
            float_max_value.x = max(float_max_value.x, input_data.x);
        } else if (remain_channels == 3) {
            float_max_value.x = max(float_max_value.x, input_data.x);
        }


        FLOAT4 accum_result       = 0;
        for (short i = 0; i < shape.y - 1; ++i) {
            input_data = RI_F(input, SAMPLER, (int2)(width_idx + i * shape.w, batch_height_idx));
            input_data = EXP(input_data - float_max_value.x);
            accum_result += input_data;
        }
        accum_result.x = accum_result.x + accum_result.y + accum_result.z + accum_result.w;

        input_data = RI_F(input, SAMPLER, (int2)(width_idx + (shape.y - 1) * shape.w, batch_height_idx));
        input_data -= float_max_value.x;
        if (remain_channels == 0) {
            accum_result.x += EXP(input_data.w);
            accum_result.x += EXP(input_data.z);
            accum_result.x += EXP(input_data.y);
            accum_result.x += EXP(input_data.x);
        } else if (remain_channels == 1) {
            accum_result.x += EXP(input_data.z);
            accum_result.x += EXP(input_data.y);
            accum_result.x += EXP(input_data.x);
        } else if (remain_channels == 2) {
            accum_result.x += EXP(input_data.y);
            accum_result.x += EXP(input_data.x);
        } else if (remain_channels == 3) {
            accum_result.x += EXP(input_data.x);
        }
        
        for(int i = 0; i < shape.y; ++i){
            int cur_out_width_pos  = mad24(i, shape.w, width_idx);
            input_data = RI_F(input, SAMPLER, (int2)(cur_out_width_pos, batch_height_idx)) - float_max_value.x;
            input_data = EXP(input_data) / accum_result.x;
            WI_F(output, (int2)(cur_out_width_pos, batch_height_idx), input_data);
        }
    }
}

__kernel void softmax_height(__read_only image2d_t input, __write_only image2d_t output,
                      __private const int4 shape // NCHW
                      ) {
    int wc = get_global_id(0);
    int b = get_global_id(1);
    if (wc < shape.y*shape.w && b < shape.x) {
        /*Compute Max */
        FLOAT4 maxValue = RI_F(input, SAMPLER, (int2)(wc, b*shape.z));
        for (int i=1; i<shape.z; ++i) {
            maxValue = fmax(maxValue, RI_F(input, SAMPLER, (int2)(wc, b*shape.z+i)));
        }
        /*Compute Exp Sum*/
        FLOAT4 sumValue = (FLOAT4)0;
        for (int i=0; i<shape.z; ++i) {
            sumValue += exp(RI_F(input, SAMPLER, (int2)(wc, b*shape.z+i)) - maxValue);
        }
        /*Compute Result */
        for (int i=0; i<shape.z; ++i) {
            FLOAT4 value = exp(RI_F(input, SAMPLER, (int2)(wc, b*shape.z+i)) - maxValue) / sumValue;
            WI_F(output, (int2)(wc, b*shape.z+i), value);
        }
    }
}


__kernel void softmax_width(__read_only image2d_t input, __write_only image2d_t output,
                      __private const int4 shape // NCHW
                      ) {
    int c = get_global_id(0);
    int bh = get_global_id(1);
    if (c < shape.y && bh < shape.x*shape.z) {
        /*Compute Max */
        FLOAT4 maxValue = RI_F(input, SAMPLER, (int2)(c*shape.w, bh));
        for (int i=1; i<shape.w; ++i) {
            maxValue = fmax(maxValue, RI_F(input, SAMPLER, (int2)(c*shape.w+i, bh)));
        }
        /*Compute Exp Sum*/
        FLOAT4 sumValue = (FLOAT4)0;
        for (int i=0; i<shape.w; ++i) {
            sumValue += exp(RI_F(input, SAMPLER, (int2)(c*shape.w+i, bh)) - maxValue);
        }
        /*Compute Result */
        for (int i=0; i<shape.w; ++i) {
            FLOAT4 value = exp(RI_F(input, SAMPLER, (int2)(c*shape.w+i, bh)) - maxValue) / sumValue;
            WI_F(output, (int2)(c*shape.w+i, bh), value);
        }
    }
}
