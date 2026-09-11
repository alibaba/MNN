#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,
#define DEAL_NON_UNIFORM_DIM2(input1, input2)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                     \
    }
#define GLOBAL_SIZE_3_DIMS __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,
#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                     \
    }

#define MNN_DATA_FORMAT_NCHW 0
#define MNN_DATA_FORMAT_NHWC 1
#define MNN_DATA_FORMAT_NC4HW4 2
#define MNN_DATA_FORMAT_C4NHW4 3

// Some drivers reject direct scalar access to __global half* even with cl_khr_fp16
// enabled ("Use vector data load builtin functions instead"), so half element access
// goes through vload_half / vstore_half.
#ifdef INPUT_IS_HALF
#define MNN_LOAD_INPUT(ptr, idx) vload_half((idx), (ptr))
#else
#define MNN_LOAD_INPUT(ptr, idx) (ptr)[idx]
#endif
#ifdef OUTPUT_IS_HALF
#define MNN_STORE_OUTPUT(val, ptr, idx) vstore_half((float)(val), (idx), (ptr))
#else
#define MNN_STORE_OUTPUT(val, ptr, idx) (ptr)[idx] = (val)
#endif
#ifdef INPUT_IS_HALF
#define MNN_LOAD_INPUT4(ptr, elem_off) vload_half4((elem_off) / 4, (ptr))
#else
#define MNN_LOAD_INPUT4(ptr, elem_off) vload4((elem_off) / 4, (ptr))
#endif
#ifdef OUTPUT_IS_HALF
#define MNN_STORE_OUTPUT4(val, ptr, elem_off) vstore_half4(convert_float4(val), (elem_off) / 4, (ptr))
#else
#define MNN_STORE_OUTPUT4(val, ptr, elem_off) vstore4((val), (elem_off) / 4, (ptr))
#endif
#ifdef MNN_SUPPORT_FP16
#define MNN_LOAD_FLOAT(ptr, idx) vload_half((idx), (ptr))
#else
#define MNN_LOAD_FLOAT(ptr, idx) *((ptr) + (idx))
#endif

__kernel void buffer_convert_to_buffer(GLOBAL_SIZE_3_DIMS
                                    __global const INPUT_TYPE *input_ptr,
                                    __private const int4 shape, // N C H W
                                    __global OUTPUT_TYPE *output_ptr
) {

    int wh  = get_global_id(0);
    int c = get_global_id(1);
    int n = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(wh, c, n);
    int w = wh % shape.w;
    int h = wh / shape.w;
    int input_offset, output_offset;
    
#if INPUT_FORMAT == MNN_DATA_FORMAT_NCHW
    input_offset = ((n * shape.y + c) * shape.z + h) * shape.w + w;
#elif INPUT_FORMAT == MNN_DATA_FORMAT_NHWC
    input_offset = ((n * shape.z + h) * shape.w + w) * shape.y + c;
#elif INPUT_FORMAT == MNN_DATA_FORMAT_NC4HW4
    input_offset = ((((c / 4) * shape.x + n) * shape.z + h) * shape.w + w) * 4 + (c % 4);
#endif

#if OUTPUT_FORMAT == MNN_DATA_FORMAT_NCHW
    output_offset = ((n * shape.y + c) * shape.z + h) * shape.w + w;
#elif OUTPUT_FORMAT == MNN_DATA_FORMAT_NHWC
    output_offset = ((n * shape.z + h) * shape.w + w) * shape.y + c;
#elif OUTPUT_FORMAT == MNN_DATA_FORMAT_NC4HW4
    output_offset = ((((c / 4) * shape.x + n) * shape.z + h) * shape.w + w) * 4 + (c % 4);
#endif

    MNN_STORE_OUTPUT(MNN_LOAD_INPUT(input_ptr, input_offset), output_ptr, output_offset);
}

__kernel void buffer_copy_to_buffer(GLOBAL_SIZE_2_DIMS
                                    __global const INPUT_TYPE *input_ptr,
                                    __global OUTPUT_TYPE *output_ptr,
                                    __private const int size // N C H W
) {

    const int x  = get_global_id(0);
    const int y  = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(x, y);
    const int offset = x << 2;
#ifdef PACK_LEAVE
    if(offset + 3 >= size){
        for(int i = 0; i < size - offset; ++i){
            MNN_STORE_OUTPUT(MNN_LOAD_INPUT(input_ptr, offset + i), output_ptr, offset + i);
        }
    } else {
#endif
        MNN_STORE_OUTPUT4(CONVERT_OUTPUT4(MNN_LOAD_INPUT4(input_ptr, offset)), output_ptr, offset);
#ifdef PACK_LEAVE
    }
#endif
}
