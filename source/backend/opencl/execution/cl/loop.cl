#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void batch_matmul(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                         __global FLOAT* output, __global FLOAT* input_A, __global FLOAT* input_B,
#ifdef BIAS
                        __global FLOAT* input_C,
#endif
                        __global FLOAT* offset_O, __global FLOAT* offset_A, __global FLOAT* offset_B,
#ifdef BIAS
                        __global FLOAT* offset_C,
#endif
                         __private const int e,
                         __private const int l,
                         __private const int h,
                         __private const int4 offsets,
                         __private const int4 iters,
                         __private const int4 steps) {
    int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
    
    if (pos.x < global_dim0 && pos.y < global_dim1 && pos.z < global_dim2) {
        int4 index = (int4)(pos.z);
        if (iters.x >= 0) {
            index.x = (int)(offset_O[pos.z]);
        }
        if (iters.y >= 0) {
            index.y = (int)(offset_A[pos.z]);
        }
        if (iters.z >= 0) {
            index.z = (int)(offset_B[pos.z]);
        }
#ifdef BIAS
        if (iters.w >= 0) {
            index.w = (int)(offset_C[pos.z]);
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
        FLOAT value = input_C[offset.w + pos.x];
#else
        FLOAT value = 0;
#endif

        for(int i = 0; i < l; ++i){
#if TRANSPOSE_A
            FLOAT value_a = A_ptr[i * e];
#else
            FLOAT value_a = A_ptr[i];
#endif

#if TRANSPOSE_B
            FLOAT value_b = B_ptr[i];
#else
            FLOAT value_b = B_ptr[i * h];
#endif

            value = mad(value_a, value_b, value);
        }

        output[offset.x + pos.y * h + pos.x] = value;
    }
}

__kernel void tile(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                        __read_only image2d_t input,
                        __global FLOAT* output,
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
        __global FLOAT* dst_ptr = output + pos.z * b_dst_pitch + c * c_dst_pitch + h * y_dst_pitch + w * x_dst_pitch;
        
        FLOAT4 value = RI_F(input, SAMPLER, (int2)(pos.y * width + w, pos.z * height + h));
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
                        __global FLOAT* input,
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
        __global FLOAT* src_ptr = input + pos.z * b_src_pitch + c * c_src_pitch + h * y_src_pitch + w * x_src_pitch;
        FLOAT4 value = (FLOAT4)0;
        FLOAT *value_ptr = (FLOAT*)&value;
        for(int i = 0; i < 4 && (i + c < channel); ++i){
            value_ptr[i] = src_ptr[i * c_src_pitch];
        }
        WI_F(output, (int2)(pos.y * width + w, pos.z * height + h), value);
    }
}

__kernel void batch_gather(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                         __global FLOAT* output, __global FLOAT* input,
                         __global FLOAT* offset_dst, __global FLOAT* offset_src,
                         __private const int x_size,
                         __private const int4 stride_src,
                         __private const int4 stride_dst,
                         __private const int2 steps,
                         __private const int2 iters) {
    int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
    
    if (pos.x < global_dim0 && pos.y < global_dim1 && pos.z < global_dim2) {
        
        int x = pos.x % x_size;
        int y = pos.x / x_size;

        int2 index = (int2)(pos.z, pos.z);
        if (iters.x >= 0) {
            index.x = (int)(offset_dst[pos.z]);
        }
        if (iters.y >= 0) {
            index.y = (int)(offset_src[pos.z]);
        }
        int2 offset = index * steps;
        output[offset.x + stride_dst.w + x * stride_dst.x + y * stride_dst.y + pos.y * stride_dst.z] = input[offset.y + stride_src.w + x * stride_src.x + y * stride_src.y + pos.y * stride_src.z];
    }
}

#ifdef LOOP_BINARY_OPERATOR
__kernel void broadcast_binary(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                         __write_only image2d_t output, __read_only image2d_t input0, __read_only image2d_t input1,
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
        
        FLOAT4 in0 = RI_F(input0, SAMPLER, (int2)(c * src0_size.x + w, n * src0_size.y + h));
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
            FLOAT4 tmp = RI_F(input1, SAMPLER, (int2)(c4_vec_prt[i] * src1_size.x + w1, n1 * src1_size.y + h1));
            FLOAT* tmp_ptr = (FLOAT*)&tmp;
            in1_ptr[i] = tmp_ptr[remain];
        }
#else
        const int w1 = w % src1_size.x;
        const int h1 = h % src1_size.y;
        const int c1 = c;
        const int n1 = n % src1_size.w;
        FLOAT4 in1 = RI_F(input1, SAMPLER, (int2)(c1 * src1_size.x + w1, n1 * src1_size.y + h1));
#endif
        FLOAT4 out = CONVERT_FLOAT4(LOOP_BINARY_OPERATOR);
        WI_F(output, (int2)(c * dst_width + w, n * dst_height + h), out);
    }
}
#endif

