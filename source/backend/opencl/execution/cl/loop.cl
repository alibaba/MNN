#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#define PI 3.141592653589f
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,


#define DEAL_NON_UNIFORM_DIM2(input1, input2)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                     \
    }
__kernel void set_zero(
                    GLOBAL_SIZE_2_DIMS
                    __global OUTPUT_TYPE *output
                    ) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    DEAL_NON_UNIFORM_DIM2(x, y);
    
    output[y*global_size_dim0 + x] = (OUTPUT_TYPE)(0);
}

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
                         __private const int h,__private const int iter,
                         __private const int4 offsets,
                         __private const int4 iters,
                         __private const int4 steps) {
    int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
    if (pos.x < global_dim0 && pos.y < global_dim1 && pos.z < global_dim2) {
        pos.x <<= 2;
        pos.y <<= 2;
        pos.z += iter;
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
        
#ifdef TRANSPOSE_A
        __global FLOAT* A_ptr = input_A + offset.y + pos.y;
#else
        __global FLOAT* A_ptr = input_A + offset.y + pos.y * l;
#endif

#ifdef TRANSPOSE_B
        __global FLOAT* B_ptr = input_B + offset.z + pos.x * l;
#else
        __global FLOAT* B_ptr = input_B + offset.z + pos.x;
#endif

#if E_LEAVES != 0
        const int valid_y1 = pos.y + 1 < e;
        const int valid_y2 = pos.y + 2 < e;
        const int valid_y3 = pos.y + 3 < e;
#endif
#if H_LEAVES != 0
        const int valid_x1 = pos.x + 1 < h;
        const int valid_x2 = pos.x + 2 < h;
        const int valid_x3 = pos.x + 3 < h;
#endif

#ifdef BIAS
#if H_LEAVES == 0
        COMPUTE_FLOAT4 value0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input_C + offset.w + pos.x));
#else
        COMPUTE_FLOAT4 value0;
        if (pos.x + 3 < h) {
            value0 = CONVERT_COMPUTE_FLOAT4(vload4(0, input_C + offset.w + pos.x));
        } else {
            value0 = CONVERT_COMPUTE_FLOAT4((FLOAT4)(input_C[offset.w + pos.x],
                                                      valid_x1 ? input_C[offset.w + pos.x + 1] : (FLOAT)0,
                                                      valid_x2 ? input_C[offset.w + pos.x + 2] : (FLOAT)0,
                                                      valid_x3 ? input_C[offset.w + pos.x + 3] : (FLOAT)0));
        }
#endif
        COMPUTE_FLOAT4 value1 = value0;
        COMPUTE_FLOAT4 value2 = value0;
        COMPUTE_FLOAT4 value3 = value0;
#else
        COMPUTE_FLOAT4 value0 = (COMPUTE_FLOAT4)0;
        COMPUTE_FLOAT4 value1 = (COMPUTE_FLOAT4)0;
        COMPUTE_FLOAT4 value2 = (COMPUTE_FLOAT4)0;
        COMPUTE_FLOAT4 value3 = (COMPUTE_FLOAT4)0;
#endif

        const int l_pack = (l + 3) >> 2;
        for(int i = 0; i < l_pack - 1; ++i){
            int l_offset = i << 2;
            COMPUTE_FLOAT4 value_a0, value_a1, value_a2, value_a3, value_b0, value_b1, value_b2, value_b3;
#ifdef TRANSPOSE_A
#if E_LEAVES == 0
            value_a0 = CONVERT_COMPUTE_FLOAT4(vload4(0, A_ptr + l_offset * e));
            value_a1 = CONVERT_COMPUTE_FLOAT4(vload4(0, A_ptr + (l_offset + 1) * e));
            value_a2 = CONVERT_COMPUTE_FLOAT4(vload4(0, A_ptr + (l_offset + 2) * e));
            value_a3 = CONVERT_COMPUTE_FLOAT4(vload4(0, A_ptr + (l_offset + 3) * e));
#else
            if (pos.y + 3 < e) {
                value_a0 = CONVERT_COMPUTE_FLOAT4(vload4(0, A_ptr + l_offset * e));
                value_a1 = CONVERT_COMPUTE_FLOAT4(vload4(0, A_ptr + (l_offset + 1) * e));
                value_a2 = CONVERT_COMPUTE_FLOAT4(vload4(0, A_ptr + (l_offset + 2) * e));
                value_a3 = CONVERT_COMPUTE_FLOAT4(vload4(0, A_ptr + (l_offset + 3) * e));
            } else {
                value_a0 = CONVERT_COMPUTE_FLOAT4((FLOAT4)(A_ptr[l_offset * e],
                                                            valid_y1 ? A_ptr[l_offset * e + 1] : (FLOAT)0,
                                                            valid_y2 ? A_ptr[l_offset * e + 2] : (FLOAT)0,
                                                            valid_y3 ? A_ptr[l_offset * e + 3] : (FLOAT)0));
                value_a1 = CONVERT_COMPUTE_FLOAT4((FLOAT4)(A_ptr[(l_offset + 1) * e],
                                                            valid_y1 ? A_ptr[(l_offset + 1) * e + 1] : (FLOAT)0,
                                                            valid_y2 ? A_ptr[(l_offset + 1) * e + 2] : (FLOAT)0,
                                                            valid_y3 ? A_ptr[(l_offset + 1) * e + 3] : (FLOAT)0));
                value_a2 = CONVERT_COMPUTE_FLOAT4((FLOAT4)(A_ptr[(l_offset + 2) * e],
                                                            valid_y1 ? A_ptr[(l_offset + 2) * e + 1] : (FLOAT)0,
                                                            valid_y2 ? A_ptr[(l_offset + 2) * e + 2] : (FLOAT)0,
                                                            valid_y3 ? A_ptr[(l_offset + 2) * e + 3] : (FLOAT)0));
                value_a3 = CONVERT_COMPUTE_FLOAT4((FLOAT4)(A_ptr[(l_offset + 3) * e],
                                                            valid_y1 ? A_ptr[(l_offset + 3) * e + 1] : (FLOAT)0,
                                                            valid_y2 ? A_ptr[(l_offset + 3) * e + 2] : (FLOAT)0,
                                                            valid_y3 ? A_ptr[(l_offset + 3) * e + 3] : (FLOAT)0));
            }
#endif
#else
            value_a0 = CONVERT_COMPUTE_FLOAT4(vload4(0, A_ptr + l_offset));
#if E_LEAVES == 0
            value_a1 = CONVERT_COMPUTE_FLOAT4(vload4(0, A_ptr + l_offset + l));
            value_a2 = CONVERT_COMPUTE_FLOAT4(vload4(0, A_ptr + l_offset + 2 * l));
            value_a3 = CONVERT_COMPUTE_FLOAT4(vload4(0, A_ptr + l_offset + 3 * l));
#else
            value_a1 = valid_y1 ? CONVERT_COMPUTE_FLOAT4(vload4(0, A_ptr + l_offset + l)) : (COMPUTE_FLOAT4)0;
            value_a2 = valid_y2 ? CONVERT_COMPUTE_FLOAT4(vload4(0, A_ptr + l_offset + 2 * l)) : (COMPUTE_FLOAT4)0;
            value_a3 = valid_y3 ? CONVERT_COMPUTE_FLOAT4(vload4(0, A_ptr + l_offset + 3 * l)) : (COMPUTE_FLOAT4)0;
#endif
#endif

#ifdef TRANSPOSE_B
#if H_LEAVES == 0
            FLOAT4 value_tmp0 = vload4(0, B_ptr + l_offset);
            FLOAT4 value_tmp1 = vload4(0, B_ptr + l_offset + l);
            FLOAT4 value_tmp2 = vload4(0, B_ptr + l_offset + 2 * l);
            FLOAT4 value_tmp3 = vload4(0, B_ptr + l_offset + 3 * l);
            value_b0 = CONVERT_COMPUTE_FLOAT4((FLOAT4)(value_tmp0.x, value_tmp1.x, value_tmp2.x, value_tmp3.x));
            value_b1 = CONVERT_COMPUTE_FLOAT4((FLOAT4)(value_tmp0.y, value_tmp1.y, value_tmp2.y, value_tmp3.y));
            value_b2 = CONVERT_COMPUTE_FLOAT4((FLOAT4)(value_tmp0.z, value_tmp1.z, value_tmp2.z, value_tmp3.z));
            value_b3 = CONVERT_COMPUTE_FLOAT4((FLOAT4)(value_tmp0.w, value_tmp1.w, value_tmp2.w, value_tmp3.w));
#else
            if (pos.x + 3 < h) {
                FLOAT4 value_tmp0 = vload4(0, B_ptr + l_offset);
                FLOAT4 value_tmp1 = vload4(0, B_ptr + l_offset + l);
                FLOAT4 value_tmp2 = vload4(0, B_ptr + l_offset + 2 * l);
                FLOAT4 value_tmp3 = vload4(0, B_ptr + l_offset + 3 * l);
                value_b0 = CONVERT_COMPUTE_FLOAT4((FLOAT4)(value_tmp0.x, value_tmp1.x, value_tmp2.x, value_tmp3.x));
                value_b1 = CONVERT_COMPUTE_FLOAT4((FLOAT4)(value_tmp0.y, value_tmp1.y, value_tmp2.y, value_tmp3.y));
                value_b2 = CONVERT_COMPUTE_FLOAT4((FLOAT4)(value_tmp0.z, value_tmp1.z, value_tmp2.z, value_tmp3.z));
                value_b3 = CONVERT_COMPUTE_FLOAT4((FLOAT4)(value_tmp0.w, value_tmp1.w, value_tmp2.w, value_tmp3.w));
            } else {
                value_b0 = CONVERT_COMPUTE_FLOAT4((FLOAT4)(B_ptr[l_offset],
                                                            valid_x1 ? B_ptr[l_offset + l] : (FLOAT)0,
                                                            valid_x2 ? B_ptr[l_offset + 2 * l] : (FLOAT)0,
                                                            valid_x3 ? B_ptr[l_offset + 3 * l] : (FLOAT)0));
                value_b1 = CONVERT_COMPUTE_FLOAT4((FLOAT4)(B_ptr[l_offset + 1],
                                                            valid_x1 ? B_ptr[l_offset + 1 + l] : (FLOAT)0,
                                                            valid_x2 ? B_ptr[l_offset + 1 + 2 * l] : (FLOAT)0,
                                                            valid_x3 ? B_ptr[l_offset + 1 + 3 * l] : (FLOAT)0));
                value_b2 = CONVERT_COMPUTE_FLOAT4((FLOAT4)(B_ptr[l_offset + 2],
                                                            valid_x1 ? B_ptr[l_offset + 2 + l] : (FLOAT)0,
                                                            valid_x2 ? B_ptr[l_offset + 2 + 2 * l] : (FLOAT)0,
                                                            valid_x3 ? B_ptr[l_offset + 2 + 3 * l] : (FLOAT)0));
                value_b3 = CONVERT_COMPUTE_FLOAT4((FLOAT4)(B_ptr[l_offset + 3],
                                                            valid_x1 ? B_ptr[l_offset + 3 + l] : (FLOAT)0,
                                                            valid_x2 ? B_ptr[l_offset + 3 + 2 * l] : (FLOAT)0,
                                                            valid_x3 ? B_ptr[l_offset + 3 + 3 * l] : (FLOAT)0));
            }
#endif
#else
#if H_LEAVES == 0
            value_b0 = CONVERT_COMPUTE_FLOAT4(vload4(0, B_ptr + l_offset * h));
            value_b1 = CONVERT_COMPUTE_FLOAT4(vload4(0, B_ptr + (l_offset + 1) * h));
            value_b2 = CONVERT_COMPUTE_FLOAT4(vload4(0, B_ptr + (l_offset + 2) * h));
            value_b3 = CONVERT_COMPUTE_FLOAT4(vload4(0, B_ptr + (l_offset + 3) * h));
#else
            if (pos.x + 3 < h) {
                value_b0 = CONVERT_COMPUTE_FLOAT4(vload4(0, B_ptr + l_offset * h));
                value_b1 = CONVERT_COMPUTE_FLOAT4(vload4(0, B_ptr + (l_offset + 1) * h));
                value_b2 = CONVERT_COMPUTE_FLOAT4(vload4(0, B_ptr + (l_offset + 2) * h));
                value_b3 = CONVERT_COMPUTE_FLOAT4(vload4(0, B_ptr + (l_offset + 3) * h));
            } else {
                value_b0 = CONVERT_COMPUTE_FLOAT4((FLOAT4)(B_ptr[l_offset * h],
                                                            valid_x1 ? B_ptr[l_offset * h + 1] : (FLOAT)0,
                                                            valid_x2 ? B_ptr[l_offset * h + 2] : (FLOAT)0,
                                                            valid_x3 ? B_ptr[l_offset * h + 3] : (FLOAT)0));
                value_b1 = CONVERT_COMPUTE_FLOAT4((FLOAT4)(B_ptr[(l_offset + 1) * h],
                                                            valid_x1 ? B_ptr[(l_offset + 1) * h + 1] : (FLOAT)0,
                                                            valid_x2 ? B_ptr[(l_offset + 1) * h + 2] : (FLOAT)0,
                                                            valid_x3 ? B_ptr[(l_offset + 1) * h + 3] : (FLOAT)0));
                value_b2 = CONVERT_COMPUTE_FLOAT4((FLOAT4)(B_ptr[(l_offset + 2) * h],
                                                            valid_x1 ? B_ptr[(l_offset + 2) * h + 1] : (FLOAT)0,
                                                            valid_x2 ? B_ptr[(l_offset + 2) * h + 2] : (FLOAT)0,
                                                            valid_x3 ? B_ptr[(l_offset + 2) * h + 3] : (FLOAT)0));
                value_b3 = CONVERT_COMPUTE_FLOAT4((FLOAT4)(B_ptr[(l_offset + 3) * h],
                                                            valid_x1 ? B_ptr[(l_offset + 3) * h + 1] : (FLOAT)0,
                                                            valid_x2 ? B_ptr[(l_offset + 3) * h + 2] : (FLOAT)0,
                                                            valid_x3 ? B_ptr[(l_offset + 3) * h + 3] : (FLOAT)0));
            }
#endif
#endif

#ifdef TRANSPOSE_A
            value0 = mad((COMPUTE_FLOAT4)value_a0.x, value_b0, value0);
            value0 = mad((COMPUTE_FLOAT4)value_a1.x, value_b1, value0);
            value0 = mad((COMPUTE_FLOAT4)value_a2.x, value_b2, value0);
            value0 = mad((COMPUTE_FLOAT4)value_a3.x, value_b3, value0);
            
            value1 = mad((COMPUTE_FLOAT4)value_a0.y, value_b0, value1);
            value1 = mad((COMPUTE_FLOAT4)value_a1.y, value_b1, value1);
            value1 = mad((COMPUTE_FLOAT4)value_a2.y, value_b2, value1);
            value1 = mad((COMPUTE_FLOAT4)value_a3.y, value_b3, value1);
            
            value2 = mad((COMPUTE_FLOAT4)value_a0.z, value_b0, value2);
            value2 = mad((COMPUTE_FLOAT4)value_a1.z, value_b1, value2);
            value2 = mad((COMPUTE_FLOAT4)value_a2.z, value_b2, value2);
            value2 = mad((COMPUTE_FLOAT4)value_a3.z, value_b3, value2);
            
            value3 = mad((COMPUTE_FLOAT4)value_a0.w, value_b0, value3);
            value3 = mad((COMPUTE_FLOAT4)value_a1.w, value_b1, value3);
            value3 = mad((COMPUTE_FLOAT4)value_a2.w, value_b2, value3);
            value3 = mad((COMPUTE_FLOAT4)value_a3.w, value_b3, value3);
#else
            value0 = mad((COMPUTE_FLOAT4)value_a0.x, value_b0, value0);
            value0 = mad((COMPUTE_FLOAT4)value_a0.y, value_b1, value0);
            value0 = mad((COMPUTE_FLOAT4)value_a0.z, value_b2, value0);
            value0 = mad((COMPUTE_FLOAT4)value_a0.w, value_b3, value0);
            
            value1 = mad((COMPUTE_FLOAT4)value_a1.x, value_b0, value1);
            value1 = mad((COMPUTE_FLOAT4)value_a1.y, value_b1, value1);
            value1 = mad((COMPUTE_FLOAT4)value_a1.z, value_b2, value1);
            value1 = mad((COMPUTE_FLOAT4)value_a1.w, value_b3, value1);
            
            value2 = mad((COMPUTE_FLOAT4)value_a2.x, value_b0, value2);
            value2 = mad((COMPUTE_FLOAT4)value_a2.y, value_b1, value2);
            value2 = mad((COMPUTE_FLOAT4)value_a2.z, value_b2, value2);
            value2 = mad((COMPUTE_FLOAT4)value_a2.w, value_b3, value2);
            
            value3 = mad((COMPUTE_FLOAT4)value_a3.x, value_b0, value3);
            value3 = mad((COMPUTE_FLOAT4)value_a3.y, value_b1, value3);
            value3 = mad((COMPUTE_FLOAT4)value_a3.z, value_b2, value3);
            value3 = mad((COMPUTE_FLOAT4)value_a3.w, value_b3, value3);
#endif
        }

        for(int i = ((l_pack - 1) << 2); i < l; ++i){
#ifdef TRANSPOSE_A
#if E_LEAVES == 0
            COMPUTE_FLOAT4 value_a = CONVERT_COMPUTE_FLOAT4(vload4(0, A_ptr + i * e));
#else
            COMPUTE_FLOAT4 value_a;
            if (pos.y + 3 < e) {
                value_a = CONVERT_COMPUTE_FLOAT4(vload4(0, A_ptr + i * e));
            } else {
                value_a = CONVERT_COMPUTE_FLOAT4((FLOAT4)(A_ptr[i * e],
                                                           valid_y1 ? A_ptr[i * e + 1] : (FLOAT)0,
                                                           valid_y2 ? A_ptr[i * e + 2] : (FLOAT)0,
                                                           valid_y3 ? A_ptr[i * e + 3] : (FLOAT)0));
            }
#endif
#else
            COMPUTE_FLOAT4 value_a;
            value_a.x = A_ptr[i];
#if E_LEAVES == 0
            value_a.y = A_ptr[i + l];
            value_a.z = A_ptr[i + 2 * l];
            value_a.w = A_ptr[i + 3 * l];
#else
            value_a.y = valid_y1 ? A_ptr[i + l] : (COMPUTE_FLOAT)0;
            value_a.z = valid_y2 ? A_ptr[i + 2 * l] : (COMPUTE_FLOAT)0;
            value_a.w = valid_y3 ? A_ptr[i + 3 * l] : (COMPUTE_FLOAT)0;
#endif
#endif

#ifdef TRANSPOSE_B
            COMPUTE_FLOAT4 value_b;
            value_b.x = B_ptr[i];
#if H_LEAVES == 0
            value_b.y = B_ptr[i + l];
            value_b.z = B_ptr[i + 2 * l];
            value_b.w = B_ptr[i + 3 * l];
#else
            value_b.y = valid_x1 ? B_ptr[i + l] : (COMPUTE_FLOAT)0;
            value_b.z = valid_x2 ? B_ptr[i + 2 * l] : (COMPUTE_FLOAT)0;
            value_b.w = valid_x3 ? B_ptr[i + 3 * l] : (COMPUTE_FLOAT)0;
#endif
#else
#if H_LEAVES == 0
            COMPUTE_FLOAT4 value_b = CONVERT_COMPUTE_FLOAT4(vload4(0, B_ptr + i * h));
#else
            COMPUTE_FLOAT4 value_b;
            if (pos.x + 3 < h) {
                value_b = CONVERT_COMPUTE_FLOAT4(vload4(0, B_ptr + i * h));
            } else {
                value_b = CONVERT_COMPUTE_FLOAT4((FLOAT4)(B_ptr[i * h],
                                                           valid_x1 ? B_ptr[i * h + 1] : (FLOAT)0,
                                                           valid_x2 ? B_ptr[i * h + 2] : (FLOAT)0,
                                                           valid_x3 ? B_ptr[i * h + 3] : (FLOAT)0));
            }
#endif
#endif

            value0 = mad((COMPUTE_FLOAT4)value_a.x, value_b, value0);
            value1 = mad((COMPUTE_FLOAT4)value_a.y, value_b, value1);
            value2 = mad((COMPUTE_FLOAT4)value_a.z, value_b, value2);
            value3 = mad((COMPUTE_FLOAT4)value_a.w, value_b, value3);
        }
        
        const int output_offset = offset.x + pos.y * h + pos.x;
#if H_LEAVES == 0
        vstore4(CONVERT_FLOAT4(value0), 0, output + output_offset);
        if(pos.y + 1 >= e) return;
        vstore4(CONVERT_FLOAT4(value1), 0, output + output_offset + h);
        if(pos.y + 2 >= e) return;
        vstore4(CONVERT_FLOAT4(value2), 0, output + output_offset + 2 * h);
        if(pos.y + 3 >= e) return;
        vstore4(CONVERT_FLOAT4(value3), 0, output + output_offset + 3 * h);
#else
        if(pos.x + 3 < h){
            vstore4(CONVERT_FLOAT4(value0), 0, output + output_offset);
            if(pos.y + 1 >= e) return;
            vstore4(CONVERT_FLOAT4(value1), 0, output + output_offset + h);
            if(pos.y + 2 >= e) return;
            vstore4(CONVERT_FLOAT4(value2), 0, output + output_offset + 2 * h);
            if(pos.y + 3 >= e) return;
            vstore4(CONVERT_FLOAT4(value3), 0, output + output_offset + 3 * h);
        }else{
#if H_LEAVES == 1
            output[output_offset] = CONVERT_FLOAT(value0.x);
            if(pos.y + 1 >= e) return;
            output[output_offset + h] = CONVERT_FLOAT(value1.x);
            if(pos.y + 2 >= e) return;
            output[output_offset + 2 * h] = CONVERT_FLOAT(value2.x);
            if(pos.y + 3 >= e) return;
            output[output_offset + 3 * h] = CONVERT_FLOAT(value3.x);
#elif H_LEAVES == 2
            vstore2(CONVERT_FLOAT2(value0.xy), 0, output + output_offset);
            if(pos.y + 1 >= e) return;
            vstore2(CONVERT_FLOAT2(value1.xy), 0, output + output_offset + h);
            if(pos.y + 2 >= e) return;
            vstore2(CONVERT_FLOAT2(value2.xy), 0, output + output_offset + 2 * h);
            if(pos.y + 3 >= e) return;
            vstore2(CONVERT_FLOAT2(value3.xy), 0, output + output_offset + 3 * h);
#elif H_LEAVES == 3
            vstore3(CONVERT_FLOAT3(value0.xyz), 0, output + output_offset);
            if(pos.y + 1 >= e) return;
            vstore3(CONVERT_FLOAT3(value1.xyz), 0, output + output_offset + h);
            if(pos.y + 2 >= e) return;
            vstore3(CONVERT_FLOAT3(value2.xyz), 0, output + output_offset + 2 * h);
            if(pos.y + 3 >= e) return;
            vstore3(CONVERT_FLOAT3(value3.xyz), 0, output + output_offset + 3 * h);
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
        const int remain = channel - c;
        value.x = (OUTPUT_TYPE_I)src_ptr[0];
        if(remain > 1) { value.y = (OUTPUT_TYPE_I)src_ptr[c_src_pitch]; }
        if(remain > 2) { value.z = (OUTPUT_TYPE_I)src_ptr[2 * c_src_pitch]; }
        if(remain > 3) { value.w = (OUTPUT_TYPE_I)src_ptr[3 * c_src_pitch]; }
        WI_DATA(output, (int2)(pos.y * width + w, pos.z * height + h), value);
    }
}

#ifndef UNARY_OPERATOR
    #define UNARY_OPERATOR in
#endif
__kernel void batch_gather(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                         __global OUTPUT_TYPE* output, __global INPUT_TYPE* input,
                         #ifdef OFFSET_DST
                         __global int* offset_dst_ptr,
                         #endif
                         #ifdef OFFSET_SRC
                         __global int* offset_src_ptr,
                         #endif
                         __private const int x_size,
                         __private const int iter,
                         __private const int4 stride_src,
                         __private const int4 stride_dst,
                         __private const int2 steps,
                         __private const int inputSize,
                         __private const int outputSize) {
    int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
    
    if (pos.x < global_dim0 && pos.y < global_dim1 && pos.z < global_dim2) {
        
        int x = pos.x % x_size;
        int y = pos.x / x_size;

        pos.z += iter;
        int2 index = (int2)(pos.z, pos.z);
#ifdef OFFSET_DST
        index.x = offset_dst_ptr[pos.z];
#endif
                    
#ifdef OFFSET_SRC
        index.y = offset_src_ptr[pos.z];
#endif
        int2 offset = index * steps;
        int outputIndex = offset.x + stride_dst.w + x * stride_dst.x + y * stride_dst.y + pos.y * stride_dst.z;
        if(outputIndex < outputSize && offset.x >= 0){
            if(offset.y >= 0 && offset.y < inputSize){
                INPUT_TYPE in = input[offset.y + stride_src.w + x * stride_src.x + y * stride_src.y + pos.y * stride_src.z];
                output[outputIndex] = (OUTPUT_TYPE)(UNARY_OPERATOR);
            }else{
                output[outputIndex] = (OUTPUT_TYPE)(0);
            }
        }
    }
}

#ifndef OPERATOR
    #define OPERATOR in0 + in1
#endif
__kernel void loop_binary(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                         __global OUTPUT_TYPE* output, __global INPUT_TYPE* input0, __global INPUT_TYPE* input1,
                         #ifdef OFFSET_DST
                         __global int* offset_dst_ptr,
                         #endif
                         #ifdef OFFSET_SRC0
                         __global int* offset_src0_ptr,
                         #endif
                         #ifdef OFFSET_SRC1
                         __global int* offset_src1_ptr,
                         #endif
                         __private const int input0Stride0,
                         __private const int input0Stride1,
                         __private const int input0Stride2,
                         __private const int input1Stride0,
                         __private const int input1Stride1,
                         __private const int input1Stride2,
                         __private const int outputStride0,
                         __private const int outputStride1,
                         __private const int outputStride2,
                         __private const int iter,
                         __private const int zSize,
                         __private const int4 offsets,
                         __private const int4 steps,
                         __private const int outputSize
                         ) {
                             
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int zn = get_global_id(2);
    
    if (x < global_dim0 && y < global_dim1 && zn < global_dim2) {
        
        int z = zn % zSize;
        int n = zn / zSize;
        n += iter;
        int4 index = (int4)(n, n, n, n);
        #ifdef OFFSET_DST
        index.x = offset_dst_ptr[n];
        #endif
                    
        #ifdef OFFSET_SRC0
        index.y = offset_src0_ptr[n];
        #endif
        
        #ifdef OFFSET_SRC1
        index.z = offset_src1_ptr[n];
        #endif
        
        int4 offset = index * steps + offsets;
        int inputIndex0 = offset.y + z * input0Stride0 + y * input0Stride1 + x * input0Stride2;
        int inputIndex1 = offset.z + z * input1Stride0 + y * input1Stride1 + x * input1Stride2;
        int outputIndex = offset.x + z * outputStride0 + y * outputStride1 + x * outputStride2;
        #ifdef INT_COMPUTE_MOD
        int in0 = (int)input0[inputIndex0];
        int in1 = (int)input1[inputIndex1];
        int out = in0 % in1;
        out = ((out < 0 && in1 > 0) || (out > 0 && in1 < 0)) ? out + in1 : out;
        #else
        float in0 = (float)input0[inputIndex0];
        float in1 = (float)input1[inputIndex1];
        float out = OPERATOR;
        #endif
        if(outputIndex < outputSize){
            output[outputIndex] = (OUTPUT_TYPE)out;
        }
    }
}

__kernel void loop_cumsum(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                         __global OUTPUT_TYPE* output, __global INPUT_TYPE* input0, __global INPUT_TYPE* input1,
                         __private const int input0Stride0,
                         __private const int input0Stride1,
                         __private const int input0Stride2,
                         __private const int input1Stride0,
                         __private const int input1Stride1,
                         __private const int input1Stride2,
                         __private const int outputStride0,
                         __private const int outputStride1,
                         __private const int outputStride2,
                         __private const int loopNumber,
                         __private const int4 offsets,
                         __private const int4 steps,
                         __private const int outputSize
                         ) {
                             
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    
    if (x < global_dim0 && y < global_dim1 && z < global_dim2) {
        
        int inputIndex0 = z * input0Stride0 + y * input0Stride1 + x * input0Stride2;
        int inputIndex1 = z * input1Stride0 + y * input1Stride1 + x * input1Stride2;
        int outputIndex = z * outputStride0 + y * outputStride1 + x * outputStride2;
        
        float in0 = 0;
        if(offsets.z != offsets.y){
            in0 = (float)input0[inputIndex0];
        }
        
        for(int i = 0; i < loopNumber; ++i){
            int4 offset = (int4)i * steps + offsets;
            float in1 = (float)input1[inputIndex1 + offset.z];
            float out = OPERATOR;
        
            if(outputIndex + offset.x < outputSize){
                output[outputIndex + offset.x] = (OUTPUT_TYPE)out;
            }
            in0 = out;
        }
    }
}
