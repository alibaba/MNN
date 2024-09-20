#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__kernel void layernorm_buf(__private int global_dim0, __private int global_dim1,
                        __global const FLOAT * input,
                        __global FLOAT * output,
                        __private const int inside,
#ifdef GAMMA_BETA
                        __global const FLOAT *gamma,
                        __global const FLOAT *beta,
#endif
                        __private float epsilon){
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
#if LOCAL_SIZE > 1
    float local sum[LOCAL_SIZE];
    if (pos.x < global_dim0 && pos.y < global_dim1) {
        const int lid = get_local_id(0);
        const int offset = pos.y * inside;
        const int inside_v4 = (inside + 3) >> 2;
        #ifdef PACK_LEAVE
        const int loop = inside_v4 - 1;
        const int inside_remain = inside - ((inside_v4-1) << 2);
        #else
        const int loop = inside_v4;
        #endif
        
        float4 in_sum = 0;
        int index = lid;
        #ifdef RMSNORM
        float4 mean = (float4)0;
        #else
        for(; index < loop; index+=LOCAL_SIZE){
            float4 in = convert_float4(vload4(index, input + offset));
            in_sum += in;
        }
        sum[lid] = in_sum.x + in_sum.y + in_sum.z+ in_sum.w;
        
        #ifdef PACK_LEAVE
        if(index == inside_v4 - 1) {
            for(int i = 0; i < inside_remain; ++i)
                float in = input[offset + index * 4 + i];
                sum[lid] = sum[lid] + in;
            }
        }
        #endif
        
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum[lid] = sum[lid] + sum[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        float4 mean = sum[0] / (float4)inside;
        #endif

        in_sum = 0;
        index = lid;
        for(; index < loop; index+=LOCAL_SIZE){
            float4 in = convert_float4(vload4(index, input + offset));
            in_sum += (in - mean) * (in - mean);
        }
        sum[lid] = in_sum.x + in_sum.y + in_sum.z + in_sum.w;
        #ifdef PACK_LEAVE
        if(index == inside_v4 - 1) {
            for(int i = 0; i < inside_remain; ++i)
                float in = input[offset + index * 4 + i];
                in = (in - mean) * (in - mean);
                sum[lid] = sum[lid] + in;
            }
        }
        #endif
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum[lid] = sum[lid] + sum[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float4 square_sum = sum[0] / (float4)inside;
        float4 value = (float4)1.0f / (float4)sqrt(square_sum + (float4)epsilon);
        index = lid;
        for(; index < loop; index+=LOCAL_SIZE){
            float4 in = convert_float4(vload4(index, input + offset));
            #ifdef GAMMA_BETA
            float4 out = (in - mean) * value * convert_float4(vload4(index, gamma)) + convert_float4(vload4(index, beta));
            #else
            float4 out = (in - mean) * value;
            #endif
            vstore4(CONVERT_FLOAT4(out), index, output + offset);
        }
        #ifdef PACK_LEAVE
        if(index == inside_v4 - 1) {
            for(int i = 0; i < inside_remain; ++i){
                float in = input[offset + index * 4 + i];
                #ifdef GAMMA_BETA
                float out = (in - mean.x) * value.x * (float)gamma[index * 4 + i] + (float)beta[index * 4 + i];
                #else
                float out = (in - mean.x) * value.x;
                #endif
                output[offset + index * 4 + i] = out;
            }
        }
        #endif
    }
#else
    if (pos.x < global_dim0 && pos.y < global_dim1) {
        const int offset = pos.y * inside;
        #ifdef RMSNORM
        float mean = 0;
        #else
        float in_sum = 0;
        for(int index = 0; index < inside; index++){
            in_sum += (float)input[offset + index];
        }
        float mean = in_sum / inside;
        #endif

        in_sum = 0;
        for(int index = 0; index < inside; index++){
            float in = (float)input[offset + index];
            in_sum += (in - mean) * (in - mean);
        }
        float square_sum = in_sum / inside;
        float value = 1.0f / sqrt(square_sum + epsilon);
        for(int i = 0; i < inside; ++i){
            float in = input[offset + i];
            #ifdef GAMMA_BETA
            float out = (in - mean) * value * (float)gamma[i] + (float)beta[i];
            #else
            float out = (in - mean) * value;
            #endif
            output[offset + i] = out;
        }
    }

#endif
}
