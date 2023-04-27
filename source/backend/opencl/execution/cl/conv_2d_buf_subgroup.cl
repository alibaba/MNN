#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)

#define TYPE_SIZE_half   2
#define TYPE_SIZE_float  4
#define TYPE_SIZE(type) CAT(TYPE_SIZE_, type)

// Creates vector type.
#define MAKE_VECTOR_TYPE_IMPL_1(elem_type)  elem_type
#define MAKE_VECTOR_TYPE_IMPL_2(elem_type)  CAT(elem_type, 2)
#define MAKE_VECTOR_TYPE_IMPL_3(elem_type)  CAT(elem_type, 3)
#define MAKE_VECTOR_TYPE_IMPL_4(elem_type)  CAT(elem_type, 4)
#define MAKE_VECTOR_TYPE_IMPL_8(elem_type)  CAT(elem_type, 8)
#define MAKE_VECTOR_TYPE(elem_type, size)   CAT(MAKE_VECTOR_TYPE_IMPL_, size)(elem_type)

#define AS_TYPE(type, val) CAT(as_, type)(val)

#define GROUP_READ_TYPE_size2 ushort
#define GROUP_READ_TYPE_size4 uint
#define GROUP_READ_TYPE(type_size) CAT(GROUP_READ_TYPE_size, type_size)

#define GROUP_READ_FUNC_size2       intel_sub_group_block_read_us
#define GROUP_READ_FUNC_size4       intel_sub_group_block_read
#define GROUP_READ_FUNC(type_size)  CAT(GROUP_READ_FUNC_size, type_size)

#define GROUP_READN_FUNC_SIZE_DEF(type_size, vector_size)   MAKE_VECTOR_TYPE(GROUP_READ_FUNC(type_size), vector_size)
#define GROUP_READN_FUNC_size1(vector_size)                 GROUP_READN_FUNC_SIZE_DEF(1, vector_size)
#define GROUP_READN_FUNC_size2(vector_size)                 GROUP_READN_FUNC_SIZE_DEF(2, vector_size)
#define GROUP_READN_FUNC_size4(vector_size)                 GROUP_READN_FUNC_SIZE_DEF(4, vector_size)
#define GROUP_READN_FUNC(type_size, vector_size)            CAT(GROUP_READN_FUNC_size, type_size)(vector_size)

#define GROUP_READN_RAW(type_size, vector_size, ptr, offset)                                        \
    GROUP_READN_FUNC(type_size, vector_size)((const __global GROUP_READ_TYPE(type_size)*)(ptr) + (offset))

#define GROUP_READN(type, vector_size, ptr, offset)                                                             \
    AS_TYPE(MAKE_VECTOR_TYPE(type, vector_size), GROUP_READN_RAW(TYPE_SIZE(type), vector_size, ptr, offset))

#define GROUP_READ(ptr, offset)            GROUP_READN(FLOAT, 1, ptr, offset)
#define GROUP_READ2(ptr, offset)           GROUP_READN(FLOAT, 2, ptr, offset)
#define GROUP_READ4(ptr, offset)           GROUP_READN(FLOAT, 4, ptr, offset)
#define GROUP_READ8(ptr, offset)           GROUP_READN(FLOAT, 8, ptr, offset)

#if TYPE_SIZE(FLOAT) == 2
#define AS_INPUT_SRC         CAT(as_, MAKE_VECTOR_TYPE(FLOAT, OUTPUT_X_BLOCK_SIZE))
#define AS_US_SRC            CAT(as_, MAKE_VECTOR_TYPE(ushort, OUTPUT_X_BLOCK_SIZE))
#define GROUP_SHUFFLEN(data, id)  AS_INPUT_SRC(intel_sub_group_shuffle(AS_US_SRC(data), id))
#define GROUP_SHUFFLE(data, id)   AS_TYPE(FLOAT, intel_sub_group_shuffle(as_ushort(data), id))
#else
#define GROUP_SHUFFLEN(data, id)  intel_sub_group_shuffle(data, id)
#define GROUP_SHUFFLE(data, id)   intel_sub_group_shuffle(data, id)
#endif

#define FEATURE_SLICE_SIZE 16
typedef MAKE_VECTOR_TYPE(FLOAT, OUTPUT_X_BLOCK_SIZE) vec;

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__kernel void conv_2d_buf_subgroup_c16(
    __global FLOAT* input,
    __global FLOAT* output,
    __global FLOAT* weights, 
    __global FLOAT* biases
) {
    const int sglid = get_sub_group_local_id();
    const int b = (uint)get_global_id(2);

    const int xy = get_global_id(0);
    const int x = (xy % X_BLOCKS) * OUTPUT_X_BLOCK_SIZE;
    const int y = (xy / X_BLOCKS);

    const int lid1 = (int)get_local_id(1);
    const int feature_per_wg = (int)get_local_size(1) / SLM_DIV_FACTOR;
    const int feature_sub_block = lid1 / feature_per_wg;
    const int feature_block = (int)get_group_id(1);


    const int input_x = x * STRIDE_WIDTH - PADDING_WIDTH;
    const int input_y = y * STRIDE_HEIGHT - PADDING_HEIGHT;

    const uint input_x_pitch = FEATURE_SLICE_SIZE;
    const uint input_y_pitch = input_x_pitch * (INPUT_WIDTH_PAD);
    const uint input_fs_pitch = input_y_pitch * (INPUT_HEIGHT_PAD);
    const uint input_b_pitch = input_fs_pitch * ((INPUT_CHANNEL + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);

    const uint input_offset = b * input_b_pitch +
                              (input_y + PADDING_HEIGHT) * input_y_pitch +
                              (input_x + PADDING_WIDTH) * input_x_pitch;

    const uint output_pack = (OUTPUT_CHANNEL + 4 - 1) / 4;
    const uint output_x_pitch = 4;
    const uint output_y_pitch = output_x_pitch *  OUTPUT_WIDTH;
    const uint output_fs_pitch = output_y_pitch * OUTPUT_HEIGHT;
    const uint output_b_pitch = output_fs_pitch * output_pack;

    const uint output_offset = b * output_b_pitch +
                               feature_block * 4 * output_fs_pitch +
                               y * output_y_pitch +
                               x * output_x_pitch;

    const uint filter_isv_pitch = FEATURE_SLICE_SIZE;
    const uint filter_x_pitch = FEATURE_SLICE_SIZE * FEATURE_SLICE_SIZE;
    const uint filter_y_pitch = filter_x_pitch * FILTER_WIDTH;
    const uint filter_is_pitch = filter_y_pitch * FILTER_HEIGHT;
    const uint filter_os_pitch = filter_is_pitch * ((INPUT_CHANNEL + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);

    const uint filter_offset = feature_block * filter_os_pitch;

#if SLM_DIV_FACTOR == 1
    vec dst = (vec)(GROUP_READ(biases, feature_block * FEATURE_SLICE_SIZE));
#else
    vec dst;

    if (feature_sub_block == 0) {
        dst = (vec)(GROUP_READ(biases, feature_block * FEATURE_SLICE_SIZE));
    } else {
        dst = (vec)0;
    }
#endif 

#if SLM_DIV_FACTOR > 1
    __local vec sum[WORK_GROUP_SIZE];
#endif


#if SLM_DIV_FACTOR > 1
        for (int icb = feature_sub_block * IC_BLOCKS / SLM_DIV_FACTOR; icb < (feature_sub_block + 1) * IC_BLOCKS / SLM_DIV_FACTOR; icb++) {
#else
        for (int icb = 0; icb < IC_BLOCKS; icb++) {
#endif 
            __attribute__((opencl_unroll_hint(FILTER_HEIGHT)))
            for (int kh = 0; kh < FILTER_HEIGHT; kh++) {
                if (input_y + kh * DILATION_HEIGHT < 0 || input_y + kh * DILATION_HEIGHT >= INPUT_HEIGHT)
                    continue;

                FLOAT line_cache[INPUT_LINE_SIZE];

                {
                    int xb = 0;
                    for (; xb + 8 <= INPUT_LINE_SIZE; xb += 8) {
                        FLOAT8 tmp = GROUP_READ8(input, input_offset +
                                                                  icb * input_fs_pitch +
                                                                  kh * DILATION_HEIGHT * input_y_pitch +
                                                                  xb * input_x_pitch);

                        line_cache[xb + 0] = tmp[0];
                        line_cache[xb + 1] = tmp[1];
                        line_cache[xb + 2] = tmp[2];
                        line_cache[xb + 3] = tmp[3];
                        line_cache[xb + 4] = tmp[4];
                        line_cache[xb + 5] = tmp[5];
                        line_cache[xb + 6] = tmp[6];
                        line_cache[xb + 7] = tmp[7];
                    }
                    for (; xb + 4 <= INPUT_LINE_SIZE; xb += 4) {
                        FLOAT4 tmp = GROUP_READ4(input, input_offset +
                                                                  icb * input_fs_pitch +
                                                                  kh * DILATION_HEIGHT * input_y_pitch +
                                                                  xb * input_x_pitch);

                        line_cache[xb + 0] = tmp[0];
                        line_cache[xb + 1] = tmp[1];
                        line_cache[xb + 2] = tmp[2];
                        line_cache[xb + 3] = tmp[3];
                    }
                    for (; xb < INPUT_LINE_SIZE; xb++) {
                        line_cache[xb] = GROUP_READ(input, input_offset +
                                                                 icb * input_fs_pitch +
                                                                 kh * DILATION_HEIGHT * input_y_pitch +
                                                                 xb * input_x_pitch);
                    }
                }

                __attribute__((opencl_unroll_hint(FILTER_WIDTH)))
                for (int kw = 0; kw < FILTER_WIDTH; kw++) {
                    vec src;
                    __attribute__((opencl_unroll_hint(OUTPUT_X_BLOCK_SIZE)))
                    for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++) {
#if FILTER_WIDTH == 1 && DILATION_WIDTH == 1 && STRIDE_WIDTH == 1
                        src[i] = line_cache[i];
#else
                        src[i] = line_cache[kw * DILATION_WIDTH + STRIDE_WIDTH * i];
#endif
                    }
                    FLOAT8 weight0 = GROUP_READ8(weights, filter_offset +
                                                                    icb * filter_is_pitch +
                                                                    kh * filter_y_pitch +
                                                                    kw * filter_x_pitch);
                    FLOAT8 weight1 = GROUP_READ8(weights, filter_offset +
                                                                    icb * filter_is_pitch +
                                                                    kh * filter_y_pitch +
                                                                    kw * filter_x_pitch +
                                                                    8 * filter_isv_pitch);
                    const vec src0  = GROUP_SHUFFLEN(src, 0);
                    const vec src1  = GROUP_SHUFFLEN(src, 1);
                    const vec src2  = GROUP_SHUFFLEN(src, 2);
                    const vec src3  = GROUP_SHUFFLEN(src, 3);
                    const vec src4  = GROUP_SHUFFLEN(src, 4);
                    const vec src5  = GROUP_SHUFFLEN(src, 5);
                    const vec src6  = GROUP_SHUFFLEN(src, 6);
                    const vec src7  = GROUP_SHUFFLEN(src, 7);
                    const vec src8  = GROUP_SHUFFLEN(src, 8);
                    const vec src9  = GROUP_SHUFFLEN(src, 9);
                    const vec src10 = GROUP_SHUFFLEN(src, 10);
                    const vec src11 = GROUP_SHUFFLEN(src, 11);
                    const vec src12 = GROUP_SHUFFLEN(src, 12);
                    const vec src13 = GROUP_SHUFFLEN(src, 13);
                    const vec src14 = GROUP_SHUFFLEN(src, 14);
                    const vec src15 = GROUP_SHUFFLEN(src, 15);

                    dst = mad(weight0.s0, src0,  dst);
                    dst = mad(weight0.s1, src1,  dst);
                    dst = mad(weight0.s2, src2,  dst);
                    dst = mad(weight0.s3, src3,  dst);
                    dst = mad(weight0.s4, src4,  dst);
                    dst = mad(weight0.s5, src5,  dst);
                    dst = mad(weight0.s6, src6,  dst);
                    dst = mad(weight0.s7, src7,  dst);
                    dst = mad(weight1.s0, src8,  dst);
                    dst = mad(weight1.s1, src9,  dst);
                    dst = mad(weight1.s2, src10, dst);
                    dst = mad(weight1.s3, src11, dst);
                    dst = mad(weight1.s4, src12, dst);
                    dst = mad(weight1.s5, src13, dst);
                    dst = mad(weight1.s6, src14, dst);
                    dst = mad(weight1.s7, src15, dst);
                }
            }
        }
        
#if SLM_DIV_FACTOR > 1
    sum[lid1] = dst;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (feature_sub_block == 0) {
        __attribute__((opencl_unroll_hint)) for(int i = 1; i < SLM_DIV_FACTOR; i++)
            dst += sum[lid1 % feature_per_wg + i * feature_per_wg];
#endif

#ifdef RELU
    dst = fmax(dst, (vec)0);
#endif

#ifdef RELU6
    dst = clamp(dst, (vec)0, (vec)6);
#endif

    const uint lid_x = sglid % 4;
    const uint lid_y = sglid / 4;
#if OUTPUT_LEFTOVERS
    if ((feature_block+1)*FEATURE_SLICE_SIZE >= OUTPUT_CHANNEL) {
        for (int i = 0; i < OUTPUT_X_BLOCK_SIZE && (x + i) < OUTPUT_WIDTH; i++) {
            if ((feature_block*FEATURE_SLICE_SIZE + lid_y * 4 < output_pack * 4))
                output[output_offset + lid_y * output_fs_pitch + i * output_x_pitch + lid_x] = dst[i];
        }
    }
    else
#endif  
    {
        for (int i = 0; i < OUTPUT_X_BLOCK_SIZE && (x + i) < OUTPUT_WIDTH; i++) {
            output[output_offset + lid_y * output_fs_pitch + i * output_x_pitch + lid_x] = dst[i];
        }
    }
#if SLM_DIV_FACTOR > 1
    }
#endif
}



__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
__kernel void conv_2d_buf_subgroup_c1(
    __global FLOAT* input,
    __global FLOAT* output,
    __global FLOAT* weights, 
    __global FLOAT* biases
)
{
    const int f_block = get_group_id(1);
    const int lid = get_sub_group_local_id();
    const int b = get_global_id(2);

    const int xy = get_global_id(0);
    const int x = (xy % X_BLOCKS) * OUTPUT_X_BLOCK_SIZE;
    const int y = (xy / X_BLOCKS);

    const int input_x = x * STRIDE_WIDTH - PADDING_WIDTH;
    const int input_y = y * STRIDE_HEIGHT - PADDING_HEIGHT;


    const uint input_x_pitch = 1;
    const uint input_y_pitch = input_x_pitch * INPUT_WIDTH;
    const uint input_f_pitch = input_y_pitch * INPUT_HEIGHT;
    const uint input_b_pitch = input_f_pitch * INPUT_CHANNEL;

    const uint input_offset = b * input_b_pitch +
                              input_y * input_y_pitch +
                              input_x * input_x_pitch;


    const uint output_pack = (OUTPUT_CHANNEL + 4 - 1) / 4;
    const uint output_x_pitch = 4;
    const uint output_y_pitch = output_x_pitch *  OUTPUT_WIDTH;
    const uint output_fs_pitch = output_y_pitch * OUTPUT_HEIGHT;
    const uint output_b_pitch = output_fs_pitch * output_pack;
    
    
    const uint output_offset = b * output_b_pitch +
                               f_block * 4 * output_fs_pitch +
                               y * output_y_pitch +
                               x * output_x_pitch;

   
    const uint filter_isv_pitch = FEATURE_SLICE_SIZE;
    const uint filter_x_pitch = FEATURE_SLICE_SIZE * FEATURE_SLICE_SIZE;
    const uint filter_y_pitch = filter_x_pitch * FILTER_WIDTH;
    const uint filter_is_pitch = filter_y_pitch * FILTER_HEIGHT;
    const uint filter_os_pitch = filter_is_pitch * ((INPUT_CHANNEL + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);

    const uint filter_offset = f_block * filter_os_pitch;

    uint bias_offset = f_block * FEATURE_SLICE_SIZE;
    vec dst = (vec)(GROUP_READ(biases, bias_offset));
    
    FLOAT line_cache[INPUT_CHANNEL * INPUT_BLOCK_SIZE];
    for (int ic = 0; ic < INPUT_CHANNEL; ic++)
    {
        __attribute__((opencl_unroll_hint(INPUT_BLOCK_SIZE)))
        for (int i = 0; i < INPUT_BLOCK_SIZE; i++)
        {
            const int in_elem = i * SUB_GROUP_SIZE + lid;
            const int xb = in_elem % INPUT_LINE_SIZE;
            const int yb = in_elem / INPUT_LINE_SIZE;
            if (input_y + yb >= 0 && input_y + yb < INPUT_HEIGHT &&
                input_x + xb >= 0 && input_x + xb < INPUT_WIDTH)
                line_cache[ic * INPUT_BLOCK_SIZE + i] = input[input_offset +
                                                              ic * input_f_pitch +
                                                              xb * input_x_pitch +
                                                              yb * input_y_pitch];
            else
                line_cache[ic * INPUT_BLOCK_SIZE + i] = 0;
        }
    }

    __attribute__((opencl_unroll_hint(FILTER_HEIGHT)))
    for (int kh = 0; kh < FILTER_HEIGHT; kh++)
    {
        __attribute__((opencl_unroll_hint(FILTER_WIDTH)))
        for (int kw = 0; kw < FILTER_WIDTH; kw++)
        {
            uint offset = filter_offset + kh * filter_y_pitch + kw * filter_x_pitch;

            FLOAT wei[INPUT_CHANNEL];
            __attribute__((opencl_unroll_hint(INPUT_CHANNEL)))
            for (int ic = 0; ic < INPUT_CHANNEL; ic++)
                wei[ic] = GROUP_READ(weights, offset + ic * filter_isv_pitch);

            __attribute__((opencl_unroll_hint(OUTPUT_X_BLOCK_SIZE)))
            for (int i = 0; i < OUTPUT_X_BLOCK_SIZE; i++)
            {
                const uint buf_offset = (kw*DILATION_WIDTH + STRIDE_WIDTH * i + (kh) * INPUT_LINE_SIZE) / SUB_GROUP_SIZE;
                const uint buf_group  = (kw*DILATION_WIDTH + STRIDE_WIDTH * i + (kh) * INPUT_LINE_SIZE) % SUB_GROUP_SIZE;

                __attribute__((opencl_unroll_hint(INPUT_CHANNEL)))
                for (int ic = 0; ic < INPUT_CHANNEL; ic++) {
                    FLOAT src = GROUP_SHUFFLE(line_cache[ic * INPUT_BLOCK_SIZE + buf_offset], buf_group);
                    dst[i] = mad(wei[ic], src, dst[i]);
                }
            }
        }
    }

#ifdef RELU
    dst = fmax(dst, (vec)0);
#endif

#ifdef RELU6
    dst = clamp(dst, (vec)0, (vec)6);
#endif

    const uint lid_x = lid % 4;
    const uint lid_y = lid / 4;
#if OUTPUT_LEFTOVERS
    if ((f_block+1)*FEATURE_SLICE_SIZE >= OUTPUT_CHANNEL) {
        for (int i = 0; i < OUTPUT_X_BLOCK_SIZE && (x + i) < OUTPUT_WIDTH; i++) {
            if ((f_block*FEATURE_SLICE_SIZE + lid_y * 4 < output_pack * 4))
                output[output_offset + lid_y * output_fs_pitch + i * output_x_pitch + lid_x] = dst[i];
        }
    }
    else
#endif  // OUTPUT_LEFTOVERS
    {
        for (int i = 0; i < OUTPUT_X_BLOCK_SIZE && (x + i) < OUTPUT_WIDTH; i++) {
            output[output_offset + lid_y * output_fs_pitch + i * output_x_pitch + lid_x] = dst[i];
        }
    }
}



__kernel void transpose_c16(
    int global_size_dim0,
    int global_size_dim1,
    int global_size_dim2,
    __global FLOAT* input,
    __global FLOAT* output,
    int channel_blocks)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    int c = z % channel_blocks;
    int b = z / channel_blocks;
    int cin = c << 2;
    if(x >= global_size_dim0 || y >= global_size_dim1 || z >= global_size_dim2)
        return;

    // Input offset calculations:
    const uint input_c_pack = ((INPUT_CHANNEL + 4 - 1) / 4);
    const uint input_x_pitch = 4;
    const uint input_y_pitch = input_x_pitch * INPUT_WIDTH;
    const uint input_f_pitch = input_y_pitch * INPUT_HEIGHT;
    const uint input_b_pitch = input_f_pitch * input_c_pack;

    const uint input_offset = b * input_b_pitch +
                              cin * input_f_pitch + 
                              (y - PADDING_HEIGHT) * input_y_pitch +
                              (x - PADDING_WIDTH) * input_x_pitch;

    // Output offset calculations:
    const uint output_x_pitch = FEATURE_SLICE_SIZE;
    const uint output_y_pitch = output_x_pitch * (INPUT_WIDTH_PAD);
    const uint output_f_pitch = output_y_pitch * (INPUT_HEIGHT_PAD);
    const uint output_b_pitch = output_f_pitch * ((INPUT_CHANNEL + FEATURE_SLICE_SIZE - 1) / FEATURE_SLICE_SIZE);

    const uint output_offset = b * output_b_pitch +
                               c * output_f_pitch + 
                               y * output_y_pitch +
                               x * output_x_pitch;

    FLOAT16 out = (FLOAT16)0;
    if(x - PADDING_WIDTH < 0 || x - PADDING_WIDTH >= INPUT_WIDTH || y - PADDING_HEIGHT < 0 || y - PADDING_HEIGHT >= INPUT_HEIGHT){
        vstore16(out, 0, output + output_offset);
        return;
    }

    out.s0123 = (cin >= input_c_pack) ? (FLOAT4)0 : vload4(0, input + input_offset);
    out.s4567 = (cin + 1 >= input_c_pack) ? (FLOAT4)0 : vload4(0, input + input_f_pitch + input_offset);
    out.s89ab = (cin + 2 >= input_c_pack) ? (FLOAT4)0 : vload4(0, input + input_f_pitch * 2 + input_offset);
    out.scdef = (cin + 3 >= input_c_pack) ? (FLOAT4)0 : vload4(0, input + input_f_pitch * 3 + input_offset);

    vstore16(out, 0, output + output_offset);
}


__kernel void transpose_c1(
    int global_size_dim0,
    int global_size_dim1,
    int global_size_dim2,
    __global FLOAT* input,
    __global FLOAT* output,
    int channel_blocks)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    int c = z % channel_blocks;
    int b = z / channel_blocks;
    int cout = c << 2;
    
    if(x >= global_size_dim0 || y >= global_size_dim1 || z >= global_size_dim2)
        return;

    // Input offset calculations:
    const uint input_c_pack = ((INPUT_CHANNEL + 4 - 1) / 4);
    const uint input_x_pitch = 4;
    const uint input_y_pitch = input_x_pitch * INPUT_WIDTH;
    const uint input_f_pitch = input_y_pitch * INPUT_HEIGHT;
    const uint input_b_pitch = input_f_pitch * input_c_pack;

    const uint input_offset = b * input_b_pitch +
                              c * input_f_pitch + 
                              y * input_y_pitch +
                              x * input_x_pitch;

    // Output offset calculations:
    const uint output_x_pitch = 1;
    const uint output_y_pitch = output_x_pitch * INPUT_WIDTH;
    const uint output_f_pitch = output_y_pitch * INPUT_HEIGHT;
    const uint output_b_pitch = output_f_pitch * INPUT_CHANNEL;

    const uint output_offset = b * output_b_pitch +
                               cout * output_f_pitch + 
                               y * output_y_pitch +
                               x * output_x_pitch;

    FLOAT4 out = vload4(0, input + input_offset);
    
    __attribute__((opencl_unroll_hint(4)))
    for(int i = 0; i < 4; ++i){
        if(cout + i >= INPUT_CHANNEL) return;
        output[output_offset + i * output_f_pitch] = out[i];
    }
}
