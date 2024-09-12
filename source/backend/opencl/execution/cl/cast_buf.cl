#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_2_DIMS \
__private const int global_size_dim0, __private const int global_size_dim1,

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                                                   \
    }

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
            int value = (int)input[inp_offset + i];
            value = value == 0 ? 0 : 1;
            output[inp_offset + i] = (OUTPUT_TYPE)value;
            #else
            output[inp_offset + i] = (OUTPUT_TYPE)input[inp_offset + i];
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
