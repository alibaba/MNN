#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define MASK_C4_TAIL(value, index, channel_unit, remain) \
    do {                                                   \
        if ((remain) != 0 && (index) == (channel_unit)-1) { \
            if ((remain) < 4) (value).w = 0.0f;            \
            if ((remain) < 3) (value).z = 0.0f;            \
            if ((remain) < 2) (value).y = 0.0f;            \
        }                                                  \
    } while (0)

__kernel void layernorm_c4_buf(__private int global_dim0, __private int global_dim1,
                        __global const FLOAT4 * input,
                        __global FLOAT4 * output,
                        __private const int inside,
#ifdef GAMMA_BETA
                        __global const FLOAT4 *gamma,
                        __global const FLOAT4 *beta,
#endif
                        __private float epsilon){
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
#if LOCAL_SIZE > 1
    float4 local sum_mnn[LOCAL_SIZE];
    #ifndef RMSNORM
    float4 local sum_mean_mnn[LOCAL_SIZE];
    #endif
    if (pos.x < global_dim0 && pos.y < global_dim1) {
        const int lid = get_local_id(0);
        const int batch = global_dim1;
        const int channelUnit = (inside + 3) / 4;
        const int channelRemain = inside & 3;

        float4 in_sum = 0;
        int index = lid;
        #ifdef RMSNORM
        float4 mean = (float4)0;
        #else
        for(; index < channelUnit; index+=LOCAL_SIZE){
            int idx = index * batch + pos.y;
            float4 in = convert_float4(input[idx]);
            MASK_C4_TAIL(in, index, channelUnit, channelRemain);
            in_sum += in;
        }
        sum_mean_mnn[lid] = in_sum;

        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum_mean_mnn[lid] = sum_mean_mnn[lid] + sum_mean_mnn[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        float sum_all = sum_mean_mnn[0].x + sum_mean_mnn[0].y + sum_mean_mnn[0].z + sum_mean_mnn[0].w;
        float4 mean = (float4)(sum_all / inside);
        #endif

        in_sum = 0;
        index = lid;
        for(; index < channelUnit; index+=LOCAL_SIZE){
            int idx = index * batch + pos.y;
            float4 in = convert_float4(input[idx]);
            float4 diff = in - mean;
            MASK_C4_TAIL(diff, index, channelUnit, channelRemain);
            in_sum += diff * diff;
        }
        sum_mnn[lid] = in_sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum_mnn[lid] = sum_mnn[lid] + sum_mnn[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float square_sum_all = sum_mnn[0].x + sum_mnn[0].y + sum_mnn[0].z + sum_mnn[0].w;
        float4 square_sum = (float4)(square_sum_all / inside);
        float4 value = (float4)1.0f / (float4)sqrt(square_sum + (float4)epsilon);
        index = lid;
        for(; index < channelUnit; index+=LOCAL_SIZE){
            int idx = index * batch + pos.y;
            float4 in = convert_float4(input[idx]);
            #ifdef GAMMA_BETA
            float4 out = (in - mean) * value * convert_float4(gamma[index]) + convert_float4(beta[index]);
            #else
            float4 out = (in - mean) * value;
            #endif
            MASK_C4_TAIL(out, index, channelUnit, channelRemain);
            output[idx] = CONVERT_FLOAT4(out);
        }
    }
#else
    if (pos.x < global_dim0 && pos.y < global_dim1) {
        const int batch = global_dim1;
        const int channelUnit = (inside + 3) / 4;
        const int channelRemain = inside & 3;

        float4 in_sum = 0;
        #ifdef RMSNORM
        float4 mean = (float4)0;
        #else
        for(int index = 0; index < channelUnit; index++){
            int idx = index * batch + pos.y;
            float4 in = convert_float4(input[idx]);
            MASK_C4_TAIL(in, index, channelUnit, channelRemain);
            in_sum += in;
        }
        float sum_all = in_sum.x + in_sum.y + in_sum.z + in_sum.w;
        float4 mean = (float4)(sum_all / inside);
        #endif

        in_sum = 0;
        for(int index = 0; index < channelUnit; index++){
            int idx = index * batch + pos.y;
            float4 in = convert_float4(input[idx]);
            float4 diff = in - mean;
            MASK_C4_TAIL(diff, index, channelUnit, channelRemain);
            in_sum += diff * diff;
        }
        float square_sum_all = in_sum.x + in_sum.y + in_sum.z + in_sum.w;
        float4 square_sum = (float4)(square_sum_all / inside);
        float4 value = (float4)1.0f / (float4)sqrt(square_sum + (float4)epsilon);
        int idx = pos.x * batch + pos.y;
        float4 in = convert_float4(input[idx]);
        #ifdef GAMMA_BETA
        float4 out = (in - mean) * value * convert_float4(gamma[pos.x]) + convert_float4(beta[pos.x]);
        #else
        float4 out = (in - mean) * value;
        #endif
        MASK_C4_TAIL(out, pos.x, channelUnit, channelRemain);
        output[idx] = CONVERT_FLOAT4(out);
    }
#endif
}

__kernel void binary_add_c4_buf(__private int global_dim0,
                        __global const FLOAT* input0,
                        __global const FLOAT* input1,
                        __global FLOAT* output,
                        __private const int size) {
    const int pos_x = get_global_id(0);
    if (pos_x >= global_dim0) return;
    int offset = pos_x << 2;
#ifdef PACK_LEAVE
    if (offset + 3 >= size) {
        int remain = size - offset;
        for (int i = 0; i < remain; ++i) {
            output[offset + i] = input0[offset + i] + input1[offset + i];
        }
        return;
    }
#endif
    float4 in0 = convert_float4(vload4(0, input0 + offset));
    float4 in1 = convert_float4(vload4(0, input1 + offset));
    float4 out = in0 + in1;
    vstore4(CONVERT_FLOAT4(out), 0, output + offset);
}

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
    float local sum_mnn[LOCAL_SIZE];
    #ifndef RMSNORM
    float local sum_mean_mnn[LOCAL_SIZE];
    #endif
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
        sum_mean_mnn[lid] = in_sum.x + in_sum.y + in_sum.z+ in_sum.w;
        
        #ifdef PACK_LEAVE
        if(index == inside_v4 - 1) {
            for(int i = 0; i < inside_remain; ++i){
                float in = input[offset + index * 4 + i];
                sum_mean_mnn[lid] = sum_mean_mnn[lid] + in;
            }
        }
        #endif
        
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum_mean_mnn[lid] = sum_mean_mnn[lid] + sum_mean_mnn[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        float4 mean = sum_mean_mnn[0] / (float4)inside;
        #endif

        in_sum = 0;
        index = lid;
        for(; index < loop; index+=LOCAL_SIZE){
            float4 in = convert_float4(vload4(index, input + offset));
            in_sum += (in - mean) * (in - mean);
        }
        sum_mnn[lid] = in_sum.x + in_sum.y + in_sum.z + in_sum.w;
        #ifdef PACK_LEAVE
        if(index == inside_v4 - 1) {
            for(int i = 0; i < inside_remain; ++i) {
                float in = input[offset + index * 4 + i];
                in = (in - mean.x) * (in - mean.x);
                sum_mnn[lid] = sum_mnn[lid] + in;
            }
        }
        #endif
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum_mnn[lid] = sum_mnn[lid] + sum_mnn[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float4 square_sum = sum_mnn[0] / (float4)inside;
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
        float in_sum = 0;
        #ifdef RMSNORM
        float mean = 0;
        #else
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