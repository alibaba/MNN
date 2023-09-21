#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
__kernel void tile_buf(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                        __global FLOAT* input, __global FLOAT* output,
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
        __global FLOAT* dst_ptr = output + pos.z * b_dst_pitch + c * c_dst_pitch + h * y_dst_pitch + w * x_dst_pitch;

        FLOAT4 value = vload4(0, input + pos.z * b_src_pitch + pos.y * c_src_pitch + h * y_src_pitch + w * x_src_pitch);
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
                        __global FLOAT* input, __global FLOAT* output,
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
        __global FLOAT* src_ptr = input + pos.z * b_src_pitch + c * c_src_pitch + h * y_src_pitch + w * x_src_pitch;
        FLOAT4 value = (FLOAT4)0;
        FLOAT *value_ptr = (FLOAT*)&value;
        for(int i = 0; i < 4 && (i + c < channel); ++i){
            value_ptr[i] = src_ptr[i * c_src_pitch];
        }
        vstore4(value, 0, output + pos.z * b_dst_pitch + pos.y * c_dst_pitch + h * y_dst_pitch + w * x_dst_pitch);
    }
}

#ifdef LOOP_BINARY_OPERATOR
__kernel void broadcast_binary_buf(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                         __global FLOAT* output, __global FLOAT* input0, __global FLOAT* input1,
                         __private const int4 src0_size, //(width, height, channel, batch)
                         __private const int4 src1_size,
                         __private const int dst_width, __private const int dst_height,
                         __private const int channel_block) {
    int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
    
    if (pos.x < global_dim0 && pos.y < global_dim1 && pos.z < global_dim2) {
        
        const int w = pos.x;
        const int h = pos.y;
        const int c = pos.z % channel_block;
        const int n = pos.z / channel_block;
        const int src0_channel_block = (src0_size.z + 3) / 4;
        const int src1_channel_block = (src1_size.z + 3) / 4;
        
        FLOAT4 in0 = vload4(0, input0 + ((((n * src0_channel_block) + c) * src0_size.y + h) * src0_size.x + w) * 4);
#ifdef BROADCAST_CHANNEL
        const int w1 = w % src1_size.x;
        const int h1 = h % src1_size.y;
        const int n1 = n % src1_size.w;
        const int c1 = c << 2;
        int4 c1_vec = (int4)(c1, c1 + 1, c1 + 2, c1 + 3);
        c1_vec = c1_vec % (int4)(src1_size.z);
        int4 c4_vec = (c1_vec + 3) / 4;
        FLOAT4 in1;
        FLOAT* in1_ptr = (FLOAT*)&in1;
        int* c1_vec_prt = (int*)&c1_vec;
        int* c4_vec_prt = (int*)&c4_vec;
        for(int i = 0; i < 4; ++i){
            int remain = (c4_vec_prt[i] << 2) - c1_vec_prt[i];
            FLOAT4 tmp = vload4(0, input1 + ((((n1 * src1_channel_block) + c4_vec_prt[i]) * src1_size.y + h1) * src1_size.x + w1) * 4);
            FLOAT* tmp_ptr = (FLOAT*)&tmp;
            in1_ptr[i] = tmp_ptr[remain];
        }
#else
        const int w1 = w % src1_size.x;
        const int h1 = h % src1_size.y;
        const int c1 = c;
        const int n1 = n % src1_size.w;
        FLOAT4 in1 = vload4(0, input1 + ((((n1 * src1_channel_block) + c1) * src1_size.y + h1) * src1_size.x + w1) * 4);
#endif
        FLOAT4 out = CONVERT_FLOAT4(LOOP_BINARY_OPERATOR);
        vstore4(out, 0, output + ((((n * channel_block) + c) * dst_height + h) * dst_width + w) * 4);
    }
}
#endif
