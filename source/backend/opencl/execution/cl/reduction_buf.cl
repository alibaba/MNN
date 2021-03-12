// TODO: use INIT_SCALAR_VALUE, OPERATOR, FINAL_OPERATOR_ON_CHANNEL macro abstract and simplify code
// TODO: support reduce dims include batch
// TODO: support keep_dim=False
// TODO: fix channel reduce result re-pack problem
#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_2_DIMS \
__private const int global_size_dim0, __private const int global_size_dim1,

__kernel void reduct_buf(GLOBAL_SIZE_2_DIMS
                            __global const FLOAT* input,
                            __global FLOAT* output,
                            __private const int batch,
                            __private const int height,
                            __private const int width
                            ) {
    const int batch_idx = get_global_id(0);
    const int width_idx = get_global_id(1);

    const int inp_offset = ((batch_idx * height + 0) * width + width_idx)*4;
    FLOAT num = input[inp_offset];
    for (int h = 1; h < height; h++) {
        FLOAT in = input[inp_offset + h*width*4];
        num = OPERATE;
    }
    
    #ifdef GET_AVG
    num = num / height;
    #endif
    const int out_offset = batch_idx * width + width_idx;
    vstore4((FLOAT4)(num, 0.0, 0.0, 0.0), out_offset, output);
}
