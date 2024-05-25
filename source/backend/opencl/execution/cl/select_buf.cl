#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_2_DIMS \
__private const int global_size_dim0, __private const int global_size_dim1,

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                                                   \
    }

__kernel void select_buf(GLOBAL_SIZE_2_DIMS
                            __global const int* select,
                            __global const FLOAT* input0,
                            __global const FLOAT* input1,
                            __global FLOAT* output
                            ) {
    const int idx = get_global_id(0);
    const int idy = get_global_id(1);

    DEAL_NON_UNIFORM_DIM2(idx, idy);
    if (select[idx]) {
#ifdef INSIZE1_EUQAL_1
        output[idx] = input0[0];
#else
        output[idx] = input0[idx];
#endif
    } else {
#ifdef INSIZE2_EUQAL_1
        output[idx] = input1[0];
#else
        output[idx] = input1[idx];
#endif
    }
}
