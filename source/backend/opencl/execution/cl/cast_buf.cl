#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_2_DIMS \
__private const int global_size_dim0, __private const int global_size_dim1,

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                                                   \
    }

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

__kernel void cast_buf(GLOBAL_SIZE_2_DIMS
                            __global INPUT_TYPE* input,
                            __global OUTPUT_TYPE* output,
                            __private const int size
                            ) {
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(idx, idy);
    const int inp_offset = idx * 4;
#ifdef PACK_LEAVE
    if(inp_offset + 3 >= size){
        int remain = size - inp_offset;
        for(int i = 0; i < remain; ++i){
            #ifdef TO_BOOL
            int value = (int)MNN_LOAD_INPUT(input, inp_offset + i);
            value = value == 0 ? 0 : 1;
            MNN_STORE_OUTPUT(value, output, inp_offset + i);
            #else
            MNN_STORE_OUTPUT(MNN_LOAD_INPUT(input, inp_offset + i), output, inp_offset + i);
            #endif
        }
    }else {
#endif
        #ifdef TO_BOOL
        int4 value = convert_int4(vload4(0, input + inp_offset));
        value = value == (int4)0 ? (int4)0 : (int4)1;
        vstore4(CONVERT_OUTPUT4(value), 0, output + inp_offset);
        #else
        vstore4(CONVERT_OUTPUT4(vload4(0, input + inp_offset)), 0, output + inp_offset);
        #endif
#ifdef PACK_LEAVE
    }
#endif
}
