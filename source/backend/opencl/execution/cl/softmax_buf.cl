#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define EXP exp
#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }


__kernel void softmax_buf(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *input,
                              __global FLOAT *output,
                              __private const int inside,
                              __private const int outside,
                              __private const int dim) {

    const int x = get_global_id(0);
    const int y = get_global_id(1); // inside
    const int z = get_global_id(2); // outside
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    
    const int offset = z * dim * inside + y;
#if SOFTMAX_LOCAL_SIZE >= 4
    int lid = get_local_id(0);
    COMPUTE_FLOAT local sum_mnn[SOFTMAX_LOCAL_SIZE];
    COMPUTE_FLOAT local max_mnn[SOFTMAX_LOCAL_SIZE];

    COMPUTE_FLOAT maxValue = (COMPUTE_FLOAT)-FLT_MAX;
    for (int i = lid; i < dim; i+=SOFTMAX_LOCAL_SIZE) {
        maxValue = fmax(maxValue, (COMPUTE_FLOAT)(input[offset+i*inside]));
    }

    max_mnn[lid] = maxValue;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = SOFTMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            max_mnn[lid] = fmax(max_mnn[lid], max_mnn[lid + i]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    maxValue = max_mnn[0];

    COMPUTE_FLOAT sumValue = (COMPUTE_FLOAT)0;
    for (int i = lid; i < dim; i+=SOFTMAX_LOCAL_SIZE) {
        sumValue += exp((COMPUTE_FLOAT)(input[offset+i*inside]) - maxValue);
    }
    sum_mnn[lid] = sumValue;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = SOFTMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            sum_mnn[lid] = sum_mnn[lid] + sum_mnn[lid + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sumValue = sum_mnn[0];
    for(int i = lid; i < dim; i+=SOFTMAX_LOCAL_SIZE){
        output[offset + i * inside] = (FLOAT)exp((COMPUTE_FLOAT)(input[offset + i * inside]) - maxValue) / sumValue;
    }
#else
    COMPUTE_FLOAT maxValue = (COMPUTE_FLOAT)-FLT_MAX;
    for (int i = 0; i < dim; i++) {
        maxValue = fmax(maxValue, (COMPUTE_FLOAT)(input[offset+i*inside]));
    }

    COMPUTE_FLOAT sumValue = (COMPUTE_FLOAT)0;
    for (int i = 0; i < dim; i++) {
        sumValue += exp((COMPUTE_FLOAT)(input[offset+i*inside]) - maxValue);
    }
    for(int i = 0; i < dim; i++){
        output[offset + i * inside] = (FLOAT)exp((COMPUTE_FLOAT)(input[offset+i*inside]) - maxValue) / sumValue;
    }
#endif
}

__kernel void softmax_v4_buf(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *input,
                              __global FLOAT *output,
                              __private const int inside,
                              __private const int outside,
                              __private const int dim) {

    const int x = get_global_id(0);
    const int y = get_global_id(1); // inside
    const int z = get_global_id(2); // outside
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    
    const int offset = z * dim * inside + (y << 2);
#if SOFTMAX_LOCAL_SIZE >= 4
    int lid = get_local_id(0);
    COMPUTE_FLOAT4 local sum_mnn[SOFTMAX_LOCAL_SIZE];
    COMPUTE_FLOAT4 local max_mnn[SOFTMAX_LOCAL_SIZE];

    COMPUTE_FLOAT4 maxValue = (COMPUTE_FLOAT4)-FLT_MAX;
    for (int i = lid; i < dim; i+=SOFTMAX_LOCAL_SIZE) {
        maxValue = fmax(maxValue, CONVERT_COMPUTE_FLOAT4(vload4(0, input+offset+i*inside)));
    }

    max_mnn[lid] = maxValue;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = SOFTMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            max_mnn[lid] = fmax(max_mnn[lid], max_mnn[lid + i]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    maxValue = max_mnn[0];

    COMPUTE_FLOAT4 sumValue = (COMPUTE_FLOAT4)0;
    for (int i = lid; i < dim; i+=SOFTMAX_LOCAL_SIZE) {
        sumValue += exp(CONVERT_COMPUTE_FLOAT4(vload4(0, input+offset+i*inside)) - maxValue);
    }
    sum_mnn[lid] = sumValue;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = SOFTMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            sum_mnn[lid] = sum_mnn[lid] + sum_mnn[lid + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sumValue = sum_mnn[0];
    for(int i = lid; i < dim; i+=SOFTMAX_LOCAL_SIZE){
        vstore4(CONVERT_FLOAT4(exp(CONVERT_COMPUTE_FLOAT4(vload4(0, input+offset+i*inside)) - maxValue) / sumValue), 0, output+offset+i*inside);
    }
#else
    COMPUTE_FLOAT4 maxValue = (COMPUTE_FLOAT4)-FLT_MAX;
    for (int i = 0; i < dim; i++) {
        maxValue = fmax(maxValue, CONVERT_COMPUTE_FLOAT4(vload4(0, input+offset+i*inside)));
    }

    COMPUTE_FLOAT4 sumValue = (COMPUTE_FLOAT4)0;
    for (int i = 0; i < dim; i++) {
        sumValue += exp(CONVERT_COMPUTE_FLOAT4(vload4(0, input+offset+i*inside)) - maxValue);
    }
    for(int i = 0; i < dim; i++){
        vstore4(CONVERT_FLOAT4(exp(CONVERT_COMPUTE_FLOAT4(vload4(0, input+offset+i*inside)) - maxValue) / sumValue), 0, output+offset+i*inside);
    }
#endif
}
