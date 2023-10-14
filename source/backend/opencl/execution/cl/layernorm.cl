#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

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
    FLOAT4 local sum[LOCAL_SIZE];
    if (pos.x < global_dim0 && pos.y < global_dim1 && pos.z < global_dim2) {
        const int h = pos.y % height;
        const int c = pos.y / height;
        const int b = pos.z;
        const int lid = get_local_id(0);
        const int bh_offset = mad24(b, height, h);

        FLOAT4 in_sum = 0;
        for(int i = lid; i < width; i+=LOCAL_SIZE){
            FLOAT4 in = RI_F(input, SAMPLER, (int2)(c * width + i, bh_offset));
            in_sum += in;
        }
        sum[lid] = in_sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum[lid] = sum[lid] + sum[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        FLOAT4 mean = sum[0] / (FLOAT4)width;
        in_sum = 0;
        for(int i = lid; i < width; i+=LOCAL_SIZE){
            FLOAT4 in = RI_F(input, SAMPLER, (int2)(c * width + i, bh_offset));
            in_sum += (in - mean) * (in - mean);
        }
        sum[lid] = in_sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum[lid] = sum[lid] + sum[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        FLOAT4 square_sum = sum[0] / (FLOAT4)width;
        FLOAT4 value = (FLOAT4)1.0f / (FLOAT4)sqrt(square_sum + (FLOAT4)epsilon);
        for(int i = lid; i < width; i+=LOCAL_SIZE){
            FLOAT4 in = RI_F(input, SAMPLER, (int2)(c * width + i, bh_offset));
#ifdef GAMMA_BETA
            FLOAT4 out = (in - mean) * value * (FLOAT4)gamma[i] + (FLOAT4)beta[i];
#else
            FLOAT4 out = (in - mean) * value;
#endif
            WI_F(output, (int2)(c * width + i, bh_offset), out);
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
    FLOAT4 local sum[LOCAL_SIZE];
    if (pos.x < global_dim0 && pos.y < global_dim1 && pos.z < global_dim2) {
        const int c = pos.y;
        const int b = pos.z;
        const int height_width = height * width;
        const int lid = get_local_id(0);

        FLOAT4 in_sum = 0;
        for(int i = lid; i < height_width; i+=LOCAL_SIZE){
            int w = i % width;
            int h = i / width;
            FLOAT4 in = RI_F(input, SAMPLER, (int2)(c * width + w, b * height + h));
            in_sum += in;
        }
        sum[lid] = in_sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum[lid] = sum[lid] + sum[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        FLOAT4 mean = sum[0] / (FLOAT4)height_width;
        in_sum = 0;
        for(int i = lid; i < height_width; i+=LOCAL_SIZE){
            int w = i % width;
            int h = i / width;
            FLOAT4 in = RI_F(input, SAMPLER, (int2)(c * width + w, b * height + h));
            in_sum += (in - mean) * (in - mean);
        }
        sum[lid] = in_sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum[lid] = sum[lid] + sum[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        FLOAT4 square_sum = sum[0] / (FLOAT4)height_width;
        FLOAT4 value = (FLOAT4)1.0f / (FLOAT4)sqrt(square_sum + (FLOAT4)epsilon);
        for(int i = lid; i < height_width; i+=LOCAL_SIZE){
            int w = i % width;
            int h = i / width;
            FLOAT4 in = RI_F(input, SAMPLER, (int2)(c * width + w, b * height + h));
#ifdef GAMMA_BETA
            FLOAT4 out = (in - mean) * value * (FLOAT4)gamma[i] + (FLOAT4)beta[i];
#else
            FLOAT4 out = (in - mean) * value;
#endif
            WI_F(output, (int2)(c * width + w, b * height + h), out);
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
    FLOAT local sum[LOCAL_SIZE];
    if (pos.x < global_dim0 && pos.y < global_dim1 && pos.z < global_dim2) {
        const int b = pos.z;
        const int sum_size = width * height * channel;
        const int reduce_size = width * height;
        const int lid = get_local_id(0);
        const int channel4 = (channel + 3) / 4;
        const int channel_remain = channel - (channel4 - 1) * 4;
        
        FLOAT4 in_sum = 0;
        FLOAT4 in_sum_left = 0;
        for(int c = 0; c < channel4 - 1; ++c){
            for(int i = lid; i < reduce_size; i+=LOCAL_SIZE){
                int w = i % width;
                int h = i / width;
                FLOAT4 in = RI_F(input, SAMPLER, (int2)(c * width + w, b * height + h));
                in_sum += in;
            }
        }
        for(int i = lid; i < reduce_size; i+=LOCAL_SIZE){
            int w = i % width;
            int h = i / width;
            FLOAT4 in = RI_F(input, SAMPLER, (int2)((channel4 - 1) * width + w, b * height + h));
            in_sum_left += in;
        }
        in_sum.x = in_sum.x + in_sum.y + in_sum.z + in_sum.w;
        FLOAT *in_sum_left_ptr = (FLOAT*)(&in_sum_left);
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
        
        FLOAT4 mean = sum[0] / (FLOAT4)sum_size;
        in_sum = 0;
        in_sum_left = 0;
        for(int c = 0; c < channel4 - 1; ++c){
            for(int i = lid; i < reduce_size; i+=LOCAL_SIZE){
                int w = i % width;
                int h = i / width;
                FLOAT4 in = RI_F(input, SAMPLER, (int2)(c * width + w, b * height + h));
                in_sum += (in - mean) * (in - mean);
            }
        }
        
        for(int i = lid; i < reduce_size; i+=LOCAL_SIZE){
            int w = i % width;
            int h = i / width;
            FLOAT4 in = RI_F(input, SAMPLER, (int2)((channel4 - 1) * width + w, b * height + h));
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
        FLOAT4 square_sum = sum[0] / (FLOAT4)sum_size;
        FLOAT4 value = (FLOAT4)1.0f / (FLOAT4)sqrt(square_sum + (FLOAT4)epsilon);
        for(int c = 0; c < channel4; ++c){
            for(int i = lid; i < reduce_size; i+=LOCAL_SIZE){
                int w = i % width;
                int h = i / width;
                FLOAT4 in = RI_F(input, SAMPLER, (int2)(c * width + w, b * height + h));
#ifdef GAMMA_BETA
                FLOAT4 out = (in - mean) * value * (FLOAT4)gamma[c * reduce_size + i] + (FLOAT4)beta[c * reduce_size + i];
#else
                FLOAT4 out = (in - mean) * value;
#endif
                WI_F(output, (int2)(c * width + w, b * height + h), out);
            }
        }
    }
}
