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


__kernel void softmax_channel(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *input,
                              __global FLOAT *output,
                              __private const int output_channels,
                              __private const int remain_channels,
                              __private const int4 shape) {//NCHW

    const int width_idx    = get_global_id(0);
    const int batch_height_idx       = get_global_id(1);

    if (width_idx < shape.w && batch_height_idx < shape.x*shape.z) {
        const int batch_idx = batch_height_idx / shape.z;
        const int height_idx = batch_height_idx % shape.z;
        const int offset = (((batch_idx*shape.y+0)*shape.z+height_idx)*shape.w+width_idx)*4;

        FLOAT4 float_max_value = (FLOAT4)-FLT_MAX;
        FLOAT4 input_data;
        for (short i = 0; i < shape.y - 1; ++i) {
            input_data      = vload4(i*shape.z*shape.w, input+offset);
            float_max_value = max(float_max_value, input_data);
        }
        
        float_max_value.x = max(float_max_value.x, float_max_value.y);
        float_max_value.x = max(float_max_value.x, float_max_value.z);
        float_max_value.x = max(float_max_value.x, float_max_value.w);

        input_data = vload4((shape.y - 1)*shape.z*shape.w, input+offset);
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
            input_data = vload4(i*shape.z*shape.w, input+offset);;
            input_data = EXP(input_data - float_max_value.x);
            accum_result += input_data;
        }
        accum_result.x = accum_result.x + accum_result.y + accum_result.z + accum_result.w;

        input_data = vload4((shape.y - 1)*shape.z*shape.w, input+offset);
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
            input_data = vload4(i*shape.z*shape.w, input+offset) - float_max_value.x;
            input_data = EXP(input_data) / accum_result.x;
            vstore4(input_data, i*shape.z*shape.w, output+offset);
        }
    }
}


__kernel void softmax_height(__global const FLOAT *input,
                             __global FLOAT *output,
                             __private const int4 shape // NCHW
                             ) {
    int wc = get_global_id(0);
    int b = get_global_id(1);
    
    const int c = wc / shape.w;
    const int w = wc % shape.w;
    const int offset = (((b*shape.y+c)*shape.z+0)*shape.w+w)*4;
    
    if (wc < shape.y*shape.w && b < shape.x) {
        /*Compute Max */
        FLOAT4 maxValue = vload4(0, input+offset);
        for (int i=1; i<shape.z; ++i) {
            maxValue = fmax(maxValue, vload4(i*shape.w, input+offset));
        }
        /*Compute Exp Sum*/
        FLOAT4 sumValue = (FLOAT4)0;
        for (int i=0; i<shape.z; ++i) {
            sumValue += exp(vload4(i*shape.w, input+offset) - maxValue);
        }
        /*Compute Result */
        for (int i=0; i<shape.z; ++i) {
            FLOAT4 value = exp(vload4(i*shape.w, input+offset) - maxValue) / sumValue;
            vstore4(value, i*shape.w, output+offset);
        }
    }    
}


__kernel void softmax_width(__global const FLOAT *input,
                            __global FLOAT *output,
                            __private const int4 shape // NCHW
                            ) {
    int c = get_global_id(0);
    int bh = get_global_id(1);
    
    const int b = bh / shape.z;
    const int h = bh % shape.z;
    const int offset = (((b*shape.y+c)*shape.z+h)*shape.w+0)*4;
    
    if (c < shape.y && bh < shape.x*shape.z) {
        /*Compute Max */
        FLOAT4 maxValue = vload4(0, input+offset);
        for (int i=1; i<shape.w; ++i) {
            maxValue = fmax(maxValue, vload4(i, input+offset));
        }
        /*Compute Exp Sum*/
        FLOAT4 sumValue = (FLOAT4)0;
        for (int i=0; i<shape.w; ++i) {
            sumValue += exp(vload4(i, input+offset) - maxValue);
        }
        /*Compute Result */
        for (int i=0; i<shape.w; ++i) {
            FLOAT4 value = exp(vload4(i, input+offset) - maxValue) / sumValue;
            vstore4(value, i, output+offset);
        }
    }
}
