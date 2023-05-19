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

        const int x_dst_pitch = 1;
        const int y_dst_pitch = x_dst_pitch * width;
        const int c_dst_pitch = y_dst_pitch * height;
        const int b_dst_pitch = c_dst_pitch * channel;
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

        const int x_src_pitch = 1;
        const int y_src_pitch = x_src_pitch * width;
        const int c_src_pitch = y_src_pitch * height;
        const int b_src_pitch = c_src_pitch * channel;
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