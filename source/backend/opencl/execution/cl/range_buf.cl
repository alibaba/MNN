#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_2_DIMS \
__private const int global_size_dim0, __private const int global_size_dim1,

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                                                   \
    }

__kernel void range_buf(GLOBAL_SIZE_2_DIMS
                            __global const INPUT_TYPE* input0,
                            __global const INPUT_TYPE* input2,
                            __global OUTPUT_TYPE* output,
                            __private const int size
                            ) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(x, y);
                                
    int index = x << 2;
    int4 index4 = (int4)(index, index + 1, index + 2, index + 3);
    INPUT_TYPE start = input0[0];
    INPUT_TYPE step = input2[0];
    OUTPUT_TYPE4 value = (OUTPUT_TYPE4)start + CONVERT_OUTPUT4(index4) * (OUTPUT_TYPE4)step;
#ifdef PACK_LEAVE
    if(index + 3 >= size){
        OUTPUT_TYPE* value_ptr = (OUTPUT_TYPE*)&value;
        for(int i = 0; i < size - index; ++i){
            output[index + i] = value_ptr[i];
        }
    }else{
#endif
        vstore4(value, 0, output + index);
#ifdef PACK_LEAVE
    }
#endif
}
