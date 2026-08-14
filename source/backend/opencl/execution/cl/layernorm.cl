#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
#ifdef LOCAL_SIZE
__kernel void layernorm_w(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                        __read_only image2d_t input,
                        __write_only image2d_t output,
                        __private const int width,
                        __private const int height,
                        __private const int channel,
#ifdef GAMMA_BETA
                        __global const FLOAT *gamma,
                        __global const FLOAT *beta,
#endif
                        __private float epsilon){
    int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
    float4 local sum_mnn[LOCAL_SIZE];
    #ifndef RMSNORM
    float4 local sum_mean_mnn[LOCAL_SIZE];
    #endif
    if (pos.x < global_dim0 && pos.y < global_dim1 && pos.z < global_dim2) {
        const int h = pos.y % height;
        const int c = pos.y / height;
        const int b = pos.z;
        const int lid = get_local_id(0);
        const int bh_offset = mad24(b, height, h);

        float4 in_sum = 0;
#ifdef RMSNORM
        float4 mean = 0;
#else
        for(int i = lid; i < width; i+=LOCAL_SIZE){
            float4 in = convert_float4(RI_F(input, SAMPLER, (int2)(c * width + i, bh_offset)));
            in_sum += in;
        }
        sum_mean_mnn[lid] = in_sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum_mean_mnn[lid] = sum_mean_mnn[lid] + sum_mean_mnn[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        float4 mean = sum_mean_mnn[0] / (float4)width;
#endif
        in_sum = 0;
        for(int i = lid; i < width; i+=LOCAL_SIZE){
            float4 in = convert_float4(RI_F(input, SAMPLER, (int2)(c * width + i, bh_offset)));
            in_sum += (in - mean) * (in - mean);
        }
        sum_mnn[lid] = in_sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum_mnn[lid] = sum_mnn[lid] + sum_mnn[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float4 square_sum = sum_mnn[0] / (float4)width;
        float4 value = (float4)1.0f / (float4)sqrt(square_sum + (float4)epsilon);
        for(int i = lid; i < width; i+=LOCAL_SIZE){
            float4 in = convert_float4(RI_F(input, SAMPLER, (int2)(c * width + i, bh_offset)));
#ifdef GAMMA_BETA
            float4 out = (in - mean) * value * (float4)gamma[i] + (float4)beta[i];
#else
            float4 out = (in - mean) * value;
#endif
            WI_F(output, (int2)(c * width + i, bh_offset), CONVERT_FLOAT4(out));
        }
    }
}


__kernel void layernorm_hw(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                        __read_only image2d_t input,
                        __write_only image2d_t output,
                        __private const int width,
                        __private const int height,
                        __private const int channel,
#ifdef GAMMA_BETA
                        __global const FLOAT *gamma,
                        __global const FLOAT *beta,
#endif
                        __private float epsilon){
    int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
    float4 local sum_mnn[LOCAL_SIZE];
    #ifndef RMSNORM
    float4 local sum_mean_mnn[LOCAL_SIZE];
    #endif
    if (pos.x < global_dim0 && pos.y < global_dim1 && pos.z < global_dim2) {
        const int c = pos.y;
        const int b = pos.z;
        const int height_width = height * width;
        const int lid = get_local_id(0);

        float4 in_sum = 0;
#ifdef RMSNORM
        float4 mean = 0;
#else
        for(int i = lid; i < height_width; i+=LOCAL_SIZE){
            int w = i % width;
            int h = i / width;
            float4 in = convert_float4(RI_F(input, SAMPLER, (int2)(c * width + w, b * height + h)));
            in_sum += in;
        }
        sum_mean_mnn[lid] = in_sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum_mean_mnn[lid] = sum_mean_mnn[lid] + sum_mean_mnn[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        float4 mean = sum_mean_mnn[0] / (float4)height_width;
#endif
        in_sum = 0;
        for(int i = lid; i < height_width; i+=LOCAL_SIZE){
            int w = i % width;
            int h = i / width;
            float4 in = convert_float4(RI_F(input, SAMPLER, (int2)(c * width + w, b * height + h)));
            in_sum += (in - mean) * (in - mean);
        }
        sum_mnn[lid] = in_sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum_mnn[lid] = sum_mnn[lid] + sum_mnn[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float4 square_sum = sum_mnn[0] / (float4)height_width;
        float4 value = (float4)1.0f / (float4)sqrt(square_sum + (float4)epsilon);
        for(int i = lid; i < height_width; i+=LOCAL_SIZE){
            int w = i % width;
            int h = i / width;
            float4 in = convert_float4(RI_F(input, SAMPLER, (int2)(c * width + w, b * height + h)));
#ifdef GAMMA_BETA
            float4 out = (in - mean) * value * (float4)gamma[i] + (float4)beta[i];
#else
            float4 out = (in - mean) * value;
#endif
            WI_F(output, (int2)(c * width + w, b * height + h), CONVERT_FLOAT4(out));
        }
    }
}

__kernel void layernorm_chw(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                        __read_only image2d_t input,
                        __write_only image2d_t output,
                        __private const int width,
                        __private const int height,
                        __private const int channel,
#ifdef GAMMA_BETA
                        __global const FLOAT *gamma,
                        __global const FLOAT *beta,
#endif
                        __private float epsilon){
    int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
    float local sum_mnn[LOCAL_SIZE];
    #ifndef RMSNORM
    float4 local sum_mean_mnn[LOCAL_SIZE];
    #endif
    if (pos.x < global_dim0 && pos.y < global_dim1 && pos.z < global_dim2) {
        const int b = pos.z;
        const int sum_size = width * height * channel;
        const int reduce_size = width * height;
        const int lid = get_local_id(0);
        const int channel4 = (channel + 3) / 4;
        const int channel_remain = channel - (channel4 - 1) * 4;
        
        float4 in_sum = 0;
        float4 in_sum_left = 0;
#ifdef RMSNORM
        float4 mean = 0;
#else
        for(int c = 0; c < channel4 - 1; ++c){
            for(int i = lid; i < reduce_size; i+=LOCAL_SIZE){
                int w = i % width;
                int h = i / width;
                float4 in = convert_float4(RI_F(input, SAMPLER, (int2)(c * width + w, b * height + h)));
                in_sum += in;
            }
        }
        for(int i = lid; i < reduce_size; i+=LOCAL_SIZE){
            int w = i % width;
            int h = i / width;
            float4 in = convert_float4(RI_F(input, SAMPLER, (int2)((channel4 - 1) * width + w, b * height + h)));
            in_sum_left += in;
        }
        in_sum.x = in_sum.x + in_sum.y + in_sum.z + in_sum.w;
        if(channel_remain > 1) { in_sum_left.x += in_sum_left.y; }
        if(channel_remain > 2) { in_sum_left.x += in_sum_left.z; }
        if(channel_remain > 3) { in_sum_left.x += in_sum_left.w; }
        sum_mean_mnn[lid] = in_sum.x + in_sum_left.x;
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum_mean_mnn[lid] = sum_mean_mnn[lid] + sum_mean_mnn[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        float4 mean = sum_mean_mnn[0] / (float4)sum_size;
#endif
        in_sum = 0;
        in_sum_left = 0;
        for(int c = 0; c < channel4 - 1; ++c){
            for(int i = lid; i < reduce_size; i+=LOCAL_SIZE){
                int w = i % width;
                int h = i / width;
                float4 in = convert_float4(RI_F(input, SAMPLER, (int2)(c * width + w, b * height + h)));
                in_sum += (in - mean) * (in - mean);
            }
        }
        
        for(int i = lid; i < reduce_size; i+=LOCAL_SIZE){
            int w = i % width;
            int h = i / width;
            float4 in = convert_float4(RI_F(input, SAMPLER, (int2)((channel4 - 1) * width + w, b * height + h)));
            in_sum_left += (in - mean) * (in - mean);
        }
        
        in_sum.x = in_sum.x + in_sum.y + in_sum.z + in_sum.w;
        if(channel_remain > 1) { in_sum_left.x += in_sum_left.y; }
        if(channel_remain > 2) { in_sum_left.x += in_sum_left.z; }
        if(channel_remain > 3) { in_sum_left.x += in_sum_left.w; }
        
        sum_mnn[lid] = in_sum.x + in_sum_left.x;
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum_mnn[lid] = sum_mnn[lid] + sum_mnn[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float4 square_sum = sum_mnn[0] / (float4)sum_size;
        float4 value = (float4)1.0f / (float4)sqrt(square_sum + (float4)epsilon);
        for(int c = 0; c < channel4; ++c){
            for(int i = lid; i < reduce_size; i+=LOCAL_SIZE){
                int w = i % width;
                int h = i / width;
                float4 in = convert_float4(RI_F(input, SAMPLER, (int2)(c * width + w, b * height + h)));
#ifdef GAMMA_BETA
                // The normalized axis includes channel here, so the 4 packed channels of this C4 block
                // (real channels 4c..4c+3) each have their own gamma/beta. Gather per-channel (identity
                // for channels past `channel`, which are only image padding) instead of broadcasting one.
                float4 gamma4 = (float4)1.0f;
                float4 beta4 = (float4)0.0f;
                const int remain_c = channel - c * 4;
                gamma4.x = gamma[(c * 4) * reduce_size + i];
                beta4.x = beta[(c * 4) * reduce_size + i];
                if(remain_c > 1) {
                    gamma4.y = gamma[(c * 4 + 1) * reduce_size + i];
                    beta4.y = beta[(c * 4 + 1) * reduce_size + i];
                }
                if(remain_c > 2) {
                    gamma4.z = gamma[(c * 4 + 2) * reduce_size + i];
                    beta4.z = beta[(c * 4 + 2) * reduce_size + i];
                }
                if(remain_c > 3) {
                    gamma4.w = gamma[(c * 4 + 3) * reduce_size + i];
                    beta4.w = beta[(c * 4 + 3) * reduce_size + i];
                }
                float4 out = (in - mean) * value * gamma4 + beta4;
#else
                float4 out = (in - mean) * value;
#endif
                WI_F(output, (int2)(c * width + w, b * height + h), CONVERT_FLOAT4(out));
            }
        }
    }
}
#endif
