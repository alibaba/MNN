#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__kernel void layernorm_w_buf(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                        __global const FLOAT * input,
                        __global FLOAT * output,
                        __private const int width,
                        __private const int height,
                        __private const int channel,
#ifdef GAMMA_BETA
                        __global const FLOAT *gamma,
                        __global const FLOAT *beta,
#endif
                        __private float epsilon){
    int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
    float4 local sum[LOCAL_SIZE];
    if (pos.x < global_dim0 && pos.y < global_dim1 && pos.z < global_dim2) {
        const int h = pos.y % height;
        const int c = pos.y / height;
        const int b = pos.z;
        const int lid = get_local_id(0);
        const int channel4 = (channel + 3) / 4;
        const int offset = ((b * channel4 + c) * height + h) * width * 4;

        float4 in_sum = 0;
#ifdef RMSNORM
        float4 mean = 0;
#else
        for(int i = lid; i < width; i+=LOCAL_SIZE){
            float4 in = convert_float4(vload4(i, input + offset));
            in_sum += in;
        }
        sum[lid] = in_sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum[lid] = sum[lid] + sum[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        float4 mean = sum[0] / (float4)width;
#endif
        in_sum = 0;
        for(int i = lid; i < width; i+=LOCAL_SIZE){
            float4 in = convert_float4(vload4(i, input + offset));
            in_sum += (in - mean) * (in - mean);
        }
        sum[lid] = in_sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum[lid] = sum[lid] + sum[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float4 square_sum = sum[0] / (float4)width;
        float4 value = (float4)1.0f / (float4)sqrt(square_sum + (float4)epsilon);
        for(int i = lid; i < width; i+=LOCAL_SIZE){
            float4 in = convert_float4(vload4(i, input + offset));
#ifdef GAMMA_BETA
            float4 out = (in - mean) * value * (float4)gamma[i] + (float4)beta[i];
#else
            float4 out = (in - mean) * value;
#endif
            vstore4(CONVERT_FLOAT4(out), i, output + offset);
        }
    }
}


__kernel void layernorm_hw_buf(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                        __global const FLOAT * input,
                        __global FLOAT * output,
                        __private const int width,
                        __private const int height,
                        __private const int channel,
#ifdef GAMMA_BETA
                        __global const FLOAT *gamma,
                        __global const FLOAT *beta,
#endif
                        __private float epsilon){
    int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
    float4 local sum[LOCAL_SIZE];
    if (pos.x < global_dim0 && pos.y < global_dim1 && pos.z < global_dim2) {
        const int c = pos.y;
        const int b = pos.z;
        const int height_width = height * width;
        const int channel4 = (channel + 3) / 4;
        const int lid = get_local_id(0);
        const int offset = ((b * channel4 + c) * height) * width * 4;

        float4 in_sum = 0;
#ifdef RMSNORM
        float4 mean = 0;
#else
        for(int i = lid; i < height_width; i+=LOCAL_SIZE){
            float4 in = convert_float4(vload4(i, input + offset));
            in_sum += in;
        }
        sum[lid] = in_sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum[lid] = sum[lid] + sum[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        float4 mean = sum[0] / (float4)height_width;
#endif
        in_sum = 0;
        for(int i = lid; i < height_width; i+=LOCAL_SIZE){
            float4 in = convert_float4(vload4(i, input + offset));
            in_sum += (in - mean) * (in - mean);
        }
        sum[lid] = in_sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum[lid] = sum[lid] + sum[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float4 square_sum = sum[0] / (float4)height_width;
        float4 value = (float4)1.0f / (float4)sqrt(square_sum + (float4)epsilon);
        for(int i = lid; i < height_width; i+=LOCAL_SIZE){
            float4 in = convert_float4(vload4(i, input + offset));
#ifdef GAMMA_BETA
            float4 out = (in - mean) * value * (float4)gamma[i] + (float4)beta[i];
#else
            float4 out = (in - mean) * value;
#endif
            vstore4(CONVERT_FLOAT4(out), i, output + offset);
        }
    }
}

__kernel void layernorm_chw_buf(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                        __global const FLOAT * input,
                        __global FLOAT * output,
                        __private const int width,
                        __private const int height,
                        __private const int channel,
#ifdef GAMMA_BETA
                        __global const FLOAT *gamma,
                        __global const FLOAT *beta,
#endif
                        __private float epsilon){
    int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
    float local sum[LOCAL_SIZE];
    if (pos.x < global_dim0 && pos.y < global_dim1 && pos.z < global_dim2) {
        const int b = pos.z;
        const int sum_size = width * height * channel;
        const int reduce_size = width * height;
        const int lid = get_local_id(0);
        const int channel4 = (channel + 3) / 4;
        const int channel_remain = channel - (channel4 - 1) * 4;
        const int offset = ((b * channel4) * height) * width * 4;
        const int wh_offset = height * width * 4;
        
        float4 in_sum = 0;
        float4 in_sum_left = 0;
        float *in_sum_left_ptr = (float*)(&in_sum_left);
#ifdef RMSNORM
        float4 mean = 0;
#else
        for(int c = 0; c < channel4 - 1; ++c){
            for(int i = lid; i < reduce_size; i+=LOCAL_SIZE){
                float4 in = convert_float4(vload4(i, input + offset + c * wh_offset));
                in_sum += in;
            }
        }
        for(int i = lid; i < reduce_size; i+=LOCAL_SIZE){
            float4 in = convert_float4(vload4(i, input + offset + (channel4 - 1) * wh_offset));
            in_sum_left += in;
        }
        in_sum.x = in_sum.x + in_sum.y + in_sum.z + in_sum.w;
        for(int i = 1; i < channel_remain; ++i){
            in_sum_left_ptr[0] += in_sum_left_ptr[i];
        }
        sum[lid] = in_sum.x + in_sum_left.x;
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum[lid] = sum[lid] + sum[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        float4 mean = sum[0] / (float4)sum_size;
#endif
        in_sum = 0;
        in_sum_left = 0;
        for(int c = 0; c < channel4 - 1; ++c){
            for(int i = lid; i < reduce_size; i+=LOCAL_SIZE){
                float4 in = convert_float4(vload4(i, input + offset + c * wh_offset));
                in_sum += (in - mean) * (in - mean);
            }
        }
        
        for(int i = lid; i < reduce_size; i+=LOCAL_SIZE){
            float4 in = convert_float4(vload4(i, input + offset + (channel4 - 1) * wh_offset));
            in_sum_left += (in - mean) * (in - mean);
        }
        
        in_sum.x = in_sum.x + in_sum.y + in_sum.z + in_sum.w;
        for(int i = 1; i < channel_remain; ++i){
            in_sum_left_ptr[0] += in_sum_left_ptr[i];
        }
        
        sum[lid] = in_sum.x + in_sum_left.x;
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum[lid] = sum[lid] + sum[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float4 square_sum = sum[0] / (float4)sum_size;
        float4 value = (float4)1.0f / (float4)sqrt(square_sum + (float4)epsilon);
        for(int c = 0; c < channel4; ++c){
            for(int i = lid; i < reduce_size; i+=LOCAL_SIZE){
                float4 in = convert_float4(vload4(i, input + offset + c * wh_offset));
#ifdef GAMMA_BETA
                float4 out = (in - mean) * value * (float4)gamma[c * reduce_size + i] + (float4)beta[c * reduce_size + i];
#else
                float4 out = (in - mean) * value;
#endif
                vstore4(CONVERT_FLOAT4(out), i, output + offset + c * wh_offset);
            }
        }
    }
}


__kernel void layernorm_plain_buf(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                        __global const FLOAT * input,
                        __global FLOAT * output,
                        __private const int inside,
                        __private const int outside,
#ifdef GAMMA_BETA
                        __global const FLOAT *gamma,
                        __global const FLOAT *beta,
#endif
                        __private float epsilon){
    int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
    COMPUTE_FLOAT local sum[LOCAL_SIZE];
    if (pos.x < global_dim0 && pos.y < global_dim1 && pos.z < global_dim2) {
        const int idx_out = pos.z;
        const int lid = get_local_id(0);
        const int offset = idx_out * inside;
        const int inside_v4 = (inside + 3) >> 2;
        const int inside_remain = inside - ((inside_v4-1) << 2);

        COMPUTE_FLOAT4 in_sum = 0;
        int index = lid;
        for(; index < inside_v4 - 1; index+=LOCAL_SIZE){
            COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(index, input + offset));
            in_sum += in;
        }
        sum[lid] = in_sum.x + in_sum.y + in_sum.z+ in_sum.w;
        
        COMPUTE_FLOAT4 in_left = 0;
        if(index == inside_v4 - 1) {
            in_left = CONVERT_COMPUTE_FLOAT4(vload4(inside_v4 - 1, input + offset));
            sum[lid] = sum[lid] + in_left.x;
            if(inside_remain > 1) {
                sum[lid] = sum[lid] + in_left.y;
            }
            if(inside_remain > 2) {
                sum[lid] = sum[lid] + in_left.z;
            }
            if(inside_remain > 3) {
                sum[lid] = sum[lid] + in_left.w;
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum[lid] = sum[lid] + sum[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        COMPUTE_FLOAT4 mean = sum[0] / (COMPUTE_FLOAT4)inside;

        in_sum = 0;
        index = lid;
        for(; index < inside_v4 - 1; index+=LOCAL_SIZE){
            COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(index, input + offset));
            in_sum += (in - mean) * (in - mean);
        }
        sum[lid] = in_sum.x + in_sum.y + in_sum.z + in_sum.w;
        
        if(index == inside_v4 - 1) {
            COMPUTE_FLOAT4 in_left = CONVERT_COMPUTE_FLOAT4(vload4(inside_v4 - 1, input + offset));
            in_sum = (in_left - mean) * (in_left - mean);
            sum[lid] = sum[lid] + in_sum.x;
            if(inside_remain > 1) {
                sum[lid] = sum[lid] + in_sum.y;
            }
            if(inside_remain > 2) {
                sum[lid] = sum[lid] + in_sum.z;
            }
            if(inside_remain > 3) {
                sum[lid] = sum[lid] + in_sum.w;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum[lid] = sum[lid] + sum[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        COMPUTE_FLOAT4 square_sum = sum[0] / (COMPUTE_FLOAT4)inside;
        COMPUTE_FLOAT4 value = (COMPUTE_FLOAT4)1.0f / (COMPUTE_FLOAT4)sqrt(square_sum + (COMPUTE_FLOAT4)epsilon);

        for(int i = lid; i < inside_v4; i+=LOCAL_SIZE){
            COMPUTE_FLOAT4 in = CONVERT_COMPUTE_FLOAT4(vload4(i, input + offset));
#ifdef GAMMA_BETA
            COMPUTE_FLOAT4 out = (in - mean) * value * CONVERT_COMPUTE_FLOAT4(vload4(i, gamma)) + CONVERT_COMPUTE_FLOAT4(vload4(i, beta));
#else
            COMPUTE_FLOAT4 out = (in - mean) * value;
#endif
            vstore4(CONVERT_FLOAT4(out), i, output + offset);
        }
    }
}
