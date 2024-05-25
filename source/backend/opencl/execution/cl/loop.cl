#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#define PI 3.141592653589f
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void batch_matmul(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                         __global FLOAT* output, __global FLOAT* input_A, __global FLOAT* input_B,
#ifdef BIAS
                        __global FLOAT* input_C,
#endif
                        __global int* offset_O, __global int* offset_A, __global int* offset_B,
#ifdef BIAS
                        __global int* offset_C,
#endif
                         __private const int e,
                         __private const int l,
                         __private const int h,
                         __private const int4 offsets,
                         __private const int4 iters,
                         __private const int4 steps) {
    int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
    if (pos.x < global_dim0 && pos.y < global_dim1 && pos.z < global_dim2) {
        pos.x <<= 2;
        pos.y <<= 2;
        int4 index = (int4)(pos.z);
        if (iters.x >= 0) {
            index.x = offset_O[pos.z];
        }
        if (iters.y >= 0) {
            index.y = offset_A[pos.z];
        }
        if (iters.z >= 0) {
            index.z = offset_B[pos.z];
        }
#ifdef BIAS
        if (iters.w >= 0) {
            index.w = offset_C[pos.z];
        }
#endif
        int4 offset = index * steps + offsets;
        
#if TRANSPOSE_A
        __global FLOAT* A_ptr = input_A + offset.y + pos.y;
#else
        __global FLOAT* A_ptr = input_A + offset.y + pos.y * l;
#endif

#if TRANSPOSE_B
        __global FLOAT* B_ptr = input_B + offset.z + pos.x * l;
#else
        __global FLOAT* B_ptr = input_B + offset.z + pos.x;
#endif

#ifdef BIAS
        FLOAT4 value0 = vload4(0, input_C + offset.w + pos.x);
        FLOAT4 value1 = value0;
        FLOAT4 value2 = value0;
        FLOAT4 value3 = value0;
#else
        FLOAT4 value0 = (FLOAT4)0;
        FLOAT4 value1 = (FLOAT4)0;
        FLOAT4 value2 = (FLOAT4)0;
        FLOAT4 value3 = (FLOAT4)0;
#endif

        const int l_pack = (l + 3) >> 2;
        for(int i = 0; i < l_pack - 1; ++i){
            int l_offset = i << 2;
            FLOAT4 value_a0, value_a1, value_a2, value_a3, value_b0, value_b1, value_b2, value_b3;
#if TRANSPOSE_A
            value_a0 = vload4(0, A_ptr + l_offset * e);
            value_a1 = vload4(0, A_ptr + (l_offset + 1) * e);
            value_a2 = vload4(0, A_ptr + (l_offset + 2) * e);
            value_a3 = vload4(0, A_ptr + (l_offset + 3) * e);
#else
            value_a0 = vload4(0, A_ptr + l_offset);
            value_a1 = vload4(0, A_ptr + l_offset + l);
            value_a2 = vload4(0, A_ptr + l_offset + 2 * l);
            value_a3 = vload4(0, A_ptr + l_offset + 3 * l);
#endif

#if TRANSPOSE_B
            FLOAT4 value_tmp0 = vload4(0, B_ptr + l_offset);
            FLOAT4 value_tmp1 = vload4(0, B_ptr + l_offset + l);
            FLOAT4 value_tmp2 = vload4(0, B_ptr + l_offset + 2 * l);
            FLOAT4 value_tmp3 = vload4(0, B_ptr + l_offset + 3 * l);
            value_b0 = (FLOAT4)(value_tmp0.x, value_tmp1.x, value_tmp2.x, value_tmp3.x);
            value_b1 = (FLOAT4)(value_tmp0.y, value_tmp1.y, value_tmp2.y, value_tmp3.y);
            value_b2 = (FLOAT4)(value_tmp0.z, value_tmp1.z, value_tmp2.z, value_tmp3.z);
            value_b3 = (FLOAT4)(value_tmp0.w, value_tmp1.w, value_tmp2.w, value_tmp3.w);
#else
            value_b0 = vload4(0, B_ptr + l_offset * h);
            value_b1 = vload4(0, B_ptr + (l_offset + 1) * h);
            value_b2 = vload4(0, B_ptr + (l_offset + 2) * h);
            value_b3 = vload4(0, B_ptr + (l_offset + 3) * h);
#endif

#ifdef TRANSPOSE_A
            value0 = mad((FLOAT4)value_a0.x, value_b0, value0);
            value0 = mad((FLOAT4)value_a1.x, value_b1, value0);
            value0 = mad((FLOAT4)value_a2.x, value_b2, value0);
            value0 = mad((FLOAT4)value_a3.x, value_b3, value0);
            
            value1 = mad((FLOAT4)value_a0.y, value_b0, value1);
            value1 = mad((FLOAT4)value_a1.y, value_b1, value1);
            value1 = mad((FLOAT4)value_a2.y, value_b2, value1);
            value1 = mad((FLOAT4)value_a3.y, value_b3, value1);
            
            value2 = mad((FLOAT4)value_a0.z, value_b0, value2);
            value2 = mad((FLOAT4)value_a1.z, value_b1, value2);
            value2 = mad((FLOAT4)value_a2.z, value_b2, value2);
            value2 = mad((FLOAT4)value_a3.z, value_b3, value2);
            
            value3 = mad((FLOAT4)value_a0.w, value_b0, value3);
            value3 = mad((FLOAT4)value_a1.w, value_b1, value3);
            value3 = mad((FLOAT4)value_a2.w, value_b2, value3);
            value3 = mad((FLOAT4)value_a3.w, value_b3, value3);
#else
            value0 = mad((FLOAT4)value_a0.x, value_b0, value0);
            value0 = mad((FLOAT4)value_a0.y, value_b1, value0);
            value0 = mad((FLOAT4)value_a0.z, value_b2, value0);
            value0 = mad((FLOAT4)value_a0.w, value_b3, value0);
            
            value1 = mad((FLOAT4)value_a1.x, value_b0, value1);
            value1 = mad((FLOAT4)value_a1.y, value_b1, value1);
            value1 = mad((FLOAT4)value_a1.z, value_b2, value1);
            value1 = mad((FLOAT4)value_a1.w, value_b3, value1);
            
            value2 = mad((FLOAT4)value_a2.x, value_b0, value2);
            value2 = mad((FLOAT4)value_a2.y, value_b1, value2);
            value2 = mad((FLOAT4)value_a2.z, value_b2, value2);
            value2 = mad((FLOAT4)value_a2.w, value_b3, value2);
            
            value3 = mad((FLOAT4)value_a3.x, value_b0, value3);
            value3 = mad((FLOAT4)value_a3.y, value_b1, value3);
            value3 = mad((FLOAT4)value_a3.z, value_b2, value3);
            value3 = mad((FLOAT4)value_a3.w, value_b3, value3);
#endif
        }

        for(int i = ((l_pack - 1) << 2); i < l; ++i){
#if TRANSPOSE_A
            FLOAT4 value_a = vload4(0, A_ptr + i * e);
#else
            FLOAT4 value_a;
            value_a.x = A_ptr[i];
            value_a.y = A_ptr[i + l];
            value_a.z = A_ptr[i + 2 * l];
            value_a.w = A_ptr[i + 3 * l];
#endif

#if TRANSPOSE_B
            FLOAT4 value_b;
            value_b.x = B_ptr[i];
            value_b.y = B_ptr[i + l];
            value_b.z = B_ptr[i + 2 * l];
            value_b.w = B_ptr[i + 3 * l];
#else
            FLOAT4 value_b = vload4(0, B_ptr + i * h);
#endif

            value0 = mad((FLOAT4)value_a.x, value_b, value0);
            value1 = mad((FLOAT4)value_a.y, value_b, value1);
            value2 = mad((FLOAT4)value_a.z, value_b, value2);
            value3 = mad((FLOAT4)value_a.w, value_b, value3);
        }
        
        const int output_offset = offset.x + pos.y * h + pos.x;
#if H_LEAVES == 0
        vstore4(value0, 0, output + output_offset);
        if(pos.y + 1 >= e) return;
        vstore4(value1, 0, output + output_offset + h);
        if(pos.y + 2 >= e) return;
        vstore4(value2, 0, output + output_offset + 2 * h);
        if(pos.y + 3 >= e) return;
        vstore4(value3, 0, output + output_offset + 3 * h);
#else
        if(pos.x + 3 < h){
            vstore4(value0, 0, output + output_offset);
            if(pos.y + 1 >= e) return;
            vstore4(value1, 0, output + output_offset + h);
            if(pos.y + 2 >= e) return;
            vstore4(value2, 0, output + output_offset + 2 * h);
            if(pos.y + 3 >= e) return;
            vstore4(value3, 0, output + output_offset + 3 * h);
        }else{
#if H_LEAVES == 1
            output[output_offset] = value0.x;
            if(pos.y + 1 >= e) return;
            output[output_offset + h] = value1.x;
            if(pos.y + 2 >= e) return;
            output[output_offset + 2 * h] = value2.x;
            if(pos.y + 3 >= e) return;
            output[output_offset + 3 * h] = value3.x;
#elif H_LEAVES == 2
            vstore2((FLOAT2)value0.xy, 0, output + output_offset);
            if(pos.y + 1 >= e) return;
            vstore2((FLOAT2)value1.xy, 0, output + output_offset + h);
            if(pos.y + 2 >= e) return;
            vstore2((FLOAT2)value2.xy, 0, output + output_offset + 2 * h);
            if(pos.y + 3 >= e) return;
            vstore2((FLOAT2)value3.xy, 0, output + output_offset + 3 * h);
#elif H_LEAVES == 3
            vstore3((FLOAT3)value0.xyz, 0, output + output_offset);
            if(pos.y + 1 >= e) return;
            vstore3((FLOAT3)value1.xyz, 0, output + output_offset + h);
            if(pos.y + 2 >= e) return;
            vstore3((FLOAT3)value2.xyz, 0, output + output_offset + 2 * h);
            if(pos.y + 3 >= e) return;
            vstore3((FLOAT3)value3.xyz, 0, output + output_offset + 3 * h);
#endif
        }
#endif
    }
}

__kernel void tile(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                        __read_only image2d_t input,
                        __global OUTPUT_TYPE* output,
                        __private const int width,
                        __private const int height,
                        __private const int channel){
    int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
    if (pos.x < global_dim0 && pos.y < global_dim1 && pos.z < global_dim2) {
        const int w = pos.x % width;
        const int h = pos.x / width;
        const int c = pos.y << 2;

#ifdef MNN_NHWC
        const int c_dst_pitch = 1;
        const int x_dst_pitch = c_dst_pitch * channel;
        const int y_dst_pitch = x_dst_pitch * width;
        const int b_dst_pitch = y_dst_pitch * height;
#else
        const int x_dst_pitch = 1;
        const int y_dst_pitch = x_dst_pitch * width;
        const int c_dst_pitch = y_dst_pitch * height;
        const int b_dst_pitch = c_dst_pitch * channel;
#endif
        __global OUTPUT_TYPE* dst_ptr = output + pos.z * b_dst_pitch + c * c_dst_pitch + h * y_dst_pitch + w * x_dst_pitch;
        
        OUTPUT_TYPE4 value = CONVERT_OUTPUT4(RI_DATA(input, SAMPLER, (int2)(pos.y * width + w, pos.z * height + h)));
        dst_ptr[0] = value.x;
        if(c + 1 >= channel)return;
        dst_ptr[c_dst_pitch] = value.y;
        if(c + 2 >= channel)return;
        dst_ptr[2 * c_dst_pitch] = value.z;
        if(c + 3 >= channel)return;
        dst_ptr[3 * c_dst_pitch] = value.w;
    }
}

__kernel void pack(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                        __global INPUT_TYPE* input,
                        __write_only image2d_t output,
                        __private const int width,
                        __private const int height,
                        __private const int channel){
    int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
    if (pos.x < global_dim0 && pos.y < global_dim1 && pos.z < global_dim2) {
        const int w = pos.x % width;
        const int h = pos.x / width;
        const int c = pos.y << 2;

#ifdef MNN_NHWC
        const int c_src_pitch = 1;
        const int x_src_pitch = c_src_pitch * channel;
        const int y_src_pitch = x_src_pitch * width;
        const int b_src_pitch = y_src_pitch * height;
#else
        const int x_src_pitch = 1;
        const int y_src_pitch = x_src_pitch * width;
        const int c_src_pitch = y_src_pitch * height;
        const int b_src_pitch = c_src_pitch * channel;
#endif
        __global INPUT_TYPE* src_ptr = input + pos.z * b_src_pitch + c * c_src_pitch + h * y_src_pitch + w * x_src_pitch;
        OUTPUT_TYPE_I4 value = (OUTPUT_TYPE_I4)0;
        OUTPUT_TYPE_I *value_ptr = (OUTPUT_TYPE_I*)&value;
        for(int i = 0; i < 4 && (i + c < channel); ++i){
            value_ptr[i] = (OUTPUT_TYPE_I)src_ptr[i * c_src_pitch];
        }
        WI_DATA(output, (int2)(pos.y * width + w, pos.z * height + h), value);
    }
}

__kernel void batch_gather(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                         __global OUTPUT_TYPE* output, __global INPUT_TYPE* input,
                         __global int* offset_dst, __global int* offset_src,
                         __private const int x_size,
                         __private const int4 stride_src,
                         __private const int4 stride_dst,
                         __private const int2 steps,
                         __private const int2 iters,
                         __private const int inputSize) {
    int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
    
    if (pos.x < global_dim0 && pos.y < global_dim1 && pos.z < global_dim2) {
        
        int x = pos.x % x_size;
        int y = pos.x / x_size;

        int2 index = (int2)(pos.z, pos.z);
        if (iters.x >= 0) {
            index.x = offset_dst[pos.z];
        }
        if (iters.y >= 0) {
            index.y = offset_src[pos.z];
        }
        int2 offset = index * steps;
        if(offset.x >= 0){
            if(offset.y >= 0 && offset.y < inputSize){
                output[offset.x + stride_dst.w + x * stride_dst.x + y * stride_dst.y + pos.y * stride_dst.z] = (OUTPUT_TYPE)input[offset.y + stride_src.w + x * stride_src.x + y * stride_src.y + pos.y * stride_src.z];
            }else{
                output[offset.x + stride_dst.w + x * stride_dst.x + y * stride_dst.y + pos.y * stride_dst.z] = (OUTPUT_TYPE)(0);
            }
        }
    }
}

#ifdef LOOP_BINARY_OPERATOR
__kernel void broadcast_binary(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                         __write_only image2d_t output, __read_only image2d_t input0, __read_only image2d_t input1,
                         __private const int8 src0_size, //(batch, channel, height, width)
                         __private const int4 src0C4_size, // nc4hw4
                         __private const int8 src1_size,
                         __private const int4 src1C4_size,
                         __private const int8 dst_size,
                         __private const int dst_width,
                         __private const int dst_height,
                         __private const int dst_channel,
                         __private const int channel_block) {
    int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
    
    if (pos.x < global_dim0 && pos.y < global_dim1 && pos.z < global_dim2) {
        
        const int wo = pos.x;
        const int ho = pos.y;
        const int co = pos.z % channel_block;
        const int no = pos.z / channel_block;
        int co4 = co << 2;
        int4 covec = (int4)(co4 % dst_channel, (co4 + 1) % dst_channel, (co4 + 2) % dst_channel, (co4 + 3) % dst_channel);
        int4 out_offset = ((no * dst_channel + covec) * dst_height + ho) * dst_width + wo;
        int4 w = out_offset % (dst_size.s3 * dst_size.s4); out_offset /= (dst_size.s3 * dst_size.s4);
        int4 h = out_offset % dst_size.s2; out_offset /= dst_size.s2;
        int4 c = out_offset % dst_size.s1; out_offset /= dst_size.s1;
        int4 n = out_offset % dst_size.s0;
        float4 in0, in1;
        float* in0_ptr = (float*)&in0;
        float* in1_ptr = (float*)&in1;
        
        {
            int4 w0 = w % (src0_size.s3 * src0_size.s4);
            int4 h0 = h % src0_size.s2;
            int4 c0 = c % src0_size.s1;
            int4 n0 = n % src0_size.s0;
            int* w0_ptr = (int*)&w0;
            int* h0_ptr = (int*)&h0;
            int* c0_ptr = (int*)&c0;
            int* n0_ptr = (int*)&n0;
            for(int i = 0; i < 4; ++i){
                int c4offset = ((n0_ptr[i] * src0_size.s1 + c0_ptr[i]) * src0_size.s2 + h0_ptr[i]) * src0_size.s3 * src0_size.s4 + w0_ptr[i];
                int wc4 = c4offset % src0C4_size.x; c4offset /= src0C4_size.x;
                int hc4 = c4offset % src0C4_size.y; c4offset /= src0C4_size.y;
                int cc4 = c4offset % src0C4_size.z; c4offset /= src0C4_size.z;
                int nc4 = c4offset % src0C4_size.w;
                int cc4_offset = cc4 / 4;
                int cc4_remain = cc4 % 4;
                float4 tmp = convert_float4(RI_DATA(input0, SAMPLER, (int2)(cc4_offset * src0C4_size.x + wc4, nc4 * src0C4_size.y + hc4)));
                float *tmp_ptr = (float*)&tmp;
                in0_ptr[i] = tmp_ptr[cc4_remain];
            }
        }
        
        {
            int4 w0 = w % (src1_size.s3 * src1_size.s4);
            int4 h0 = h % src1_size.s2;
            int4 c0 = c % src1_size.s1;
            int4 n0 = n % src1_size.s0;
            int* w0_ptr = (int*)&w0;
            int* h0_ptr = (int*)&h0;
            int* c0_ptr = (int*)&c0;
            int* n0_ptr = (int*)&n0;
            for(int i = 0; i < 4; ++i){
                int c4offset = ((n0_ptr[i] * src1_size.s1 + c0_ptr[i]) * src1_size.s2 + h0_ptr[i]) * src1_size.s3 * src1_size.s4 + w0_ptr[i];
                int wc4 = c4offset % src1C4_size.x; c4offset /= src1C4_size.x;
                int hc4 = c4offset % src1C4_size.y; c4offset /= src1C4_size.y;
                int cc4 = c4offset % src1C4_size.z; c4offset /= src1C4_size.z;
                int nc4 = c4offset % src1C4_size.w;
                int cc4_offset = cc4 / 4;
                int cc4_remain = cc4 % 4;
                float4 tmp = convert_float4(RI_DATA(input1, SAMPLER, (int2)(cc4_offset * src1C4_size.x + wc4, nc4 * src1C4_size.y + hc4)));
                float *tmp_ptr = (float*)&tmp;
                in1_ptr[i] = tmp_ptr[cc4_remain];
            }
        }
        
        float4 out = LOOP_BINARY_OPERATOR;
        WI_DATA(output, (int2)(co * dst_width + wo, no * dst_height + ho), CONVERT_OUTPUT_I4(out));
    }
}
#endif

