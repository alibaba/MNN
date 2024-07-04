#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__kernel void groupnorm_plain_buf(__private int global_dim0, __private int global_dim1, __private int global_dim2,
#ifdef DOUBLE_INPUTS
                        __global const FLOAT * input0,
                        __global const FLOAT * input1,
#else
                        __global const FLOAT * input,
#endif
                        __global FLOAT * output,
                        __private const int area,
                        __private const int group,
                        __private const int inside,
                        __private const int outside,
#ifdef GAMMA_BETA
                        __global const FLOAT *gamma,
                        __global const FLOAT *beta,
#endif
                        __private float epsilon){
    int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
    float local sum[LOCAL_SIZE];
    if (pos.x < global_dim0 && pos.y < global_dim1 && pos.z < global_dim2) {
        const int idx_out = pos.z;
        const int lid = get_local_id(0);
        const int offset = idx_out * inside;
        const int inside_v4 = (inside + 3) >> 2;
        
#ifdef DOUBLE_INPUTS
        // The product of W and H is a multiple of 4
        #ifdef WH_4
        float4 in_sum = 0;
        int index = lid;
        for(; index < inside_v4; index+=LOCAL_SIZE){
            float4 in0 = convert_float4(vload4(index, input0 + offset));
            in_sum += in0;
            float in1 = input1[idx_out * (inside/area) + index / (area/4)];
            in_sum += (float4)(in1, in1, in1, in1);
        }
        sum[lid] = in_sum.x + in_sum.y + in_sum.z+ in_sum.w;

        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum[lid] = sum[lid] + sum[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        
        float4 mean = sum[0] / (float4)inside;

        in_sum = 0;
        index = lid;
        for(; index < inside_v4; index+=LOCAL_SIZE){
            float4 in0 = convert_float4(vload4(index, input0 + offset));
            float in1 = input1[idx_out * (inside/area) + index / (area/4)];
            in_sum += (in0 + (float4)(in1, in1, in1, in1) - mean) * (in0 + (float4)in1 - mean);
        }
        sum[lid] = in_sum.x + in_sum.y + in_sum.z + in_sum.w;
        
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum[lid] = sum[lid] + sum[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float4 square_sum = sum[0] / (float4)inside;
        float4 value = (float4)1.0f / (float4)sqrt(square_sum + (float4)epsilon);

        for(int i = lid; i < inside_v4; i+=LOCAL_SIZE){
            float4 in0 = convert_float4(vload4(i, input0 + offset));
            float in1 = input1[idx_out * (inside/area) + i / (area/4)];
            float4 out = (in0 + (float4)(in1, in1, in1, in1) - mean) * value;

            #ifdef GAMMA_BETA
            int offset_gamma_beta = (idx_out % group) * inside/area + i / (area/4);
            out = out * (float4)((float)gamma[offset_gamma_beta]) + (float4)((float)beta[offset_gamma_beta]);
            #endif

            #ifdef SWISH
            out = out * native_recip((float4)1+native_exp(convert_float4(-out)));
            #endif
            vstore4(CONVERT_FLOAT4(out), i, output + offset);
        }
        #else
        
        float in_sum = 0;
        int index = lid;
        for(; index < inside; index+=LOCAL_SIZE){
            float in0 = input0[offset + index];
            in_sum += in0;
            float in1 = input1[idx_out * (inside/area) + index / area];
            in_sum += in1;
        }
        sum[lid] = in_sum;

        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum[lid] = sum[lid] + sum[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        
        float mean = sum[0] / inside;

        in_sum = 0;
        index = lid;
        for(; index < inside; index+=LOCAL_SIZE){
            float in0 = input0[offset + index];
            float in1 = input1[idx_out * (inside/area) + index / area];
            in_sum += (in0 + in1 - mean) * (in0 + in1 - mean);
        }
        sum[lid] = in_sum;
        
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
            if (lid < i)
                sum[lid] = sum[lid] + sum[lid + i];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float square_sum = sum[0] / inside;
        float value = 1.0f / sqrt(square_sum + epsilon);

        for(int i = lid; i < inside; i+=LOCAL_SIZE){
            float in0 = input0[offset + i];
            float in1 = input1[idx_out * (inside/area) + i / area];
            float out = (in0 + in1 - mean) * value;

            #ifdef GAMMA_BETA
            int offset_gamma_beta = (idx_out % group) * inside/area + i / area;
            out = out * (float)gamma[offset_gamma_beta] + (float)beta[offset_gamma_beta];
            #endif

            #ifdef SWISH
            out = out * native_recip(1.0+native_exp(-out));
            #endif
            output[offset+i] = (FLOAT)out;
        }
        
        #endif
#else
        const int inside_remain = inside - ((inside_v4-1) << 2);

        float4 in_sum = 0;
        int index = lid;
        for(; index < inside_v4 - 1; index+=LOCAL_SIZE){
            float4 in = convert_float4(vload4(index, input + offset));
            in_sum += in;
        }
        sum[lid] = in_sum.x + in_sum.y + in_sum.z+ in_sum.w;
        
        float4 in_left = 0;
        if(index == inside_v4 - 1) {
            in_left = convert_float4(vload4(inside_v4 - 1, input + offset));
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
        
        float4 mean = sum[0] / (float4)inside;

        in_sum = 0;
        index = lid;
        for(; index < inside_v4 - 1; index+=LOCAL_SIZE){
            float4 in = convert_float4(vload4(index, input + offset));
            in_sum += (in - mean) * (in - mean);
        }
        sum[lid] = in_sum.x + in_sum.y + in_sum.z + in_sum.w;
        
        if(index == inside_v4 - 1) {
            float4 in_left = convert_float4(vload4(inside_v4 - 1, input + offset));
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
        float4 square_sum = sum[0] / (float4)inside;
        float4 value = (float4)1.0f / (float4)sqrt(square_sum + (float4)epsilon);

        // The product of W and H is a multiple of 4
        #ifdef WH_4
        for(int i = lid; i < inside_v4; i+=LOCAL_SIZE){
            float4 in = convert_float4(vload4(i, input + offset));
            float4 out = (in - mean) * value;

            #ifdef GAMMA_BETA
            int offset_gamma_beta = (idx_out % group) * inside/area + i / (area/4);
            out = out * (float4)((float)gamma[offset_gamma_beta]) + (float4)((float)beta[offset_gamma_beta]);
            #endif

            #ifdef SWISH
            out = out * native_recip((float4)1+native_exp(convert_float4(-out)));
            #endif
            vstore4(CONVERT_FLOAT4(out), i, output + offset);
        }
        #else
        for(int i = lid; i < inside; i+=LOCAL_SIZE){
            float in = input[offset+i];
            float out = (in - mean.x) * value.x;

            #ifdef GAMMA_BETA
            int offset_gamma_beta = (idx_out % group) * inside/area + i / area;
            out = out * (float)gamma[offset_gamma_beta] + (float)beta[offset_gamma_beta];
            #endif

            #ifdef SWISH
            out = out * native_recip(1.0+native_exp(-out));
            #endif
            output[offset+i] = (FLOAT)out;
        }
        #endif
#endif
    }
}
