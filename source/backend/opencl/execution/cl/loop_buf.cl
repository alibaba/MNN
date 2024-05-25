#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#define PI 3.141592653589f
__kernel void tile_buf(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                        __global INPUT_TYPE* input, __global OUTPUT_TYPE* output,
                        __private const int width,
                        __private const int height,
                        __private const int channel){
    int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
    if (pos.x < global_dim0 && pos.y < global_dim1 && pos.z < global_dim2) {
        const int w = pos.x % width;
        const int h = pos.x / width;
        const int c = pos.y << 2;
        const int x_src_pitch = 4;
        const int y_src_pitch = x_src_pitch * width;
        const int c_src_pitch = y_src_pitch * height;
        const int b_src_pitch = c_src_pitch * ((channel + 3) / 4);
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

        OUTPUT_TYPE4 value = CONVERT_OUTPUT4(vload4(0, input + pos.z * b_src_pitch + pos.y * c_src_pitch + h * y_src_pitch + w * x_src_pitch));
        dst_ptr[0] = value.x;
        if(c + 1 >= channel)return;
        dst_ptr[c_dst_pitch] = value.y;
        if(c + 2 >= channel)return;
        dst_ptr[2 * c_dst_pitch] = value.z;
        if(c + 3 >= channel)return;
        dst_ptr[3 * c_dst_pitch] = value.w;
    }
}

__kernel void pack_buf(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                        __global INPUT_TYPE* input, __global OUTPUT_TYPE* output,
                        __private const int width,
                        __private const int height,
                        __private const int channel){
    int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
    if (pos.x < global_dim0 && pos.y < global_dim1 && pos.z < global_dim2) {
        const int w = pos.x % width;
        const int h = pos.x / width;
        const int c = pos.y << 2;
        const int x_dst_pitch = 4;
        const int y_dst_pitch = x_dst_pitch * width;
        const int c_dst_pitch = y_dst_pitch * height;
        const int b_dst_pitch = c_dst_pitch * ((channel + 3) / 4);
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
        OUTPUT_TYPE4 value = (OUTPUT_TYPE4)0;
        OUTPUT_TYPE *value_ptr = (OUTPUT_TYPE*)&value;
        for(int i = 0; i < 4 && (i + c < channel); ++i){
            value_ptr[i] = (OUTPUT_TYPE)src_ptr[i * c_src_pitch];
        }
        vstore4(value, 0, output + pos.z * b_dst_pitch + pos.y * c_dst_pitch + h * y_dst_pitch + w * x_dst_pitch);
    }
}

#ifdef LOOP_BINARY_OPERATOR
__kernel void broadcast_binary_buf(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                         __global OUTPUT_TYPE* output, __global INPUT_TYPE* input0, __global INPUT_TYPE* input1,
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
        const int src0_channel_block = (src0C4_size.z + 3) / 4;
        const int src1_channel_block = (src1C4_size.z + 3) / 4;
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
                in0_ptr[i] = (float)input0[((((nc4 * src0_channel_block) + cc4_offset) * src0C4_size.y + hc4) * src0C4_size.x + wc4) * 4 + cc4_remain];
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
                in1_ptr[i] = (float)input1[((((nc4 * src1_channel_block) + cc4_offset) * src1C4_size.y + hc4) * src1C4_size.x + wc4) * 4 + cc4_remain];
            }
        }
        
        float4 out = LOOP_BINARY_OPERATOR;
        vstore4(CONVERT_OUTPUT4(out), 0, output + ((((no * channel_block) + co) * dst_height + ho) * dst_width + wo) * 4);
    }
}
#endif
