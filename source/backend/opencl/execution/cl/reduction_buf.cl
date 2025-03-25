// TODO: use INIT_SCALAR_VALUE, OPERATOR, FINAL_OPERATOR_ON_CHANNEL macro abstract and simplify code
// TODO: support reduce dims include batch
// TODO: support keep_dim=False
// TODO: fix channel reduce result re-pack problem
#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_2_DIMS \
__private const int global_size_dim0, __private const int global_size_dim1,

#define GLOBAL_SIZE_3_DIMS \
__private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

__kernel void reduct_buf(GLOBAL_SIZE_3_DIMS
                              __global const INPUT_TYPE *input,
                              __global OUTPUT_TYPE *output,
                              __private const int inside,
                              __private const int outside,
                              __private const int dim) {

    const int x = get_global_id(0);
    const int y = get_global_id(1); // inside
    const int z = get_global_id(2); // outside
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    
    INPUT_TYPE out = (INPUT_TYPE)VALUE;
    const int offset = z * dim * inside + y;
    
#if REDUCT_LOCAL_SIZE > 4
    const int lid = get_local_id(0);
    INPUT_TYPE local sum[REDUCT_LOCAL_SIZE];
    for(int i = lid; i < dim; i+=REDUCT_LOCAL_SIZE){
        INPUT_TYPE in = (INPUT_TYPE)input[offset + i * inside];
        out = OPERATE(out, in);
    }
    sum[lid] = out;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = REDUCT_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            sum[lid] = OPERATE(sum[lid], sum[lid + i]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    out = sum[0];
#else
    for(int i = 0; i < dim; ++i){
        INPUT_TYPE in = (INPUT_TYPE)input[offset + i * inside];
        out = OPERATE(out, in);
    }
#endif

#ifdef GET_AVG
    out = out / dim;
#endif
    output[z * inside + y] = (OUTPUT_TYPE)out;
}

__kernel void reduct_v4_buf(GLOBAL_SIZE_3_DIMS
                              __global const INPUT_TYPE *input,
                              __global OUTPUT_TYPE *output,
                              __private const int inside,
                              __private const int outside,
                              __private const int dim) {

    const int x = get_global_id(0);
    const int y = get_global_id(1); // inside
    const int z = get_global_id(2); // outside
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    
    INPUT_TYPE4 out = (INPUT_TYPE4)VALUE;
    const int offset = z * dim * inside + (y << 2);
    
#if REDUCT_LOCAL_SIZE > 4
    const int lid = get_local_id(0);
    INPUT_TYPE4 local sum[REDUCT_LOCAL_SIZE];
    for(int i = lid; i < dim; i+=REDUCT_LOCAL_SIZE){
        INPUT_TYPE4 in = vload4(0, input + offset + i * inside);
        out = OPERATE(out, in);
    }
    sum[lid] = out;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = REDUCT_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            sum[lid] = OPERATE(sum[lid], sum[lid + i]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    out = sum[0];
#else
    for(int i = 0; i < dim; ++i){
        INPUT_TYPE4 in = vload4(0, input + offset + i * inside);
        out = OPERATE(out, in);
    }
#endif

#ifdef GET_AVG
    out = out / (INPUT_TYPE4)dim;
#endif
    vstore4(CONVERT_OUTPUT4(out), 0, output + z * inside + (y << 2));
}
