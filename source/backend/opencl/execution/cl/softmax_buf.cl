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


__kernel void softmax_in1_buf(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *input,
                              __global FLOAT *output,
                              __private const int inside,
                              __private const int outside,
                              __private const int dim) {

    const int x = get_global_id(0);
    const int y = get_global_id(1); // inside = 1
    const int z = get_global_id(2); // outside
    DEAL_NON_UNIFORM_DIM3(x, y, z);
    
    const int offset = z * dim + y;
    const int dim4 = (dim + 3) / 4;
    const int loop_end = max(0, dim4 - 1);
#if SOFTMAX_LOCAL_SIZE >= 4
    int lid = get_local_id(0);
    COMPUTE_FLOAT local sum[SOFTMAX_LOCAL_SIZE];

    // compute maxvalue
    COMPUTE_FLOAT4 maxValue = (COMPUTE_FLOAT4)-FLT_MAX;
    for (int i = lid; i < loop_end; i+=SOFTMAX_LOCAL_SIZE) {
        maxValue = fmax(maxValue, CONVERT_COMPUTE_FLOAT4(vload4(i, input+offset)));
    }

    sum[lid] = fmax(fmax(fmax(maxValue.x, maxValue.y), maxValue.z), maxValue.w);
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = SOFTMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            sum[lid] = fmax(sum[lid], sum[lid + i]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    maxValue.x = sum[0];
    for(int i = loop_end << 2; i < dim; ++i){
        maxValue.x = fmax(maxValue.x, (COMPUTE_FLOAT)(input[offset+i]));
    }

    // compute sumvalue
    COMPUTE_FLOAT4 sumValue = (COMPUTE_FLOAT4)0;
    for (int i = lid; i < loop_end; i+=SOFTMAX_LOCAL_SIZE) {
        sumValue += exp(CONVERT_COMPUTE_FLOAT4(vload4(i, input+offset)) - (COMPUTE_FLOAT4)maxValue.x);
    }
    sum[lid] = sumValue.x + sumValue.y + sumValue.z + sumValue.w;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = SOFTMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            sum[lid] = sum[lid] + sum[lid + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sumValue.x = sum[0];
    for(int i = loop_end << 2; i < dim; ++i){
        sumValue.x += exp((COMPUTE_FLOAT)(input[offset+i]) - maxValue.x);
    }
    
    // store result
    for(int i = lid; i < loop_end; i+=SOFTMAX_LOCAL_SIZE){
        vstore4(CONVERT_FLOAT4(exp(CONVERT_COMPUTE_FLOAT4(vload4(i, input+offset)) - (COMPUTE_FLOAT4)maxValue.x) / (COMPUTE_FLOAT4)sumValue.x), 0, output + offset + i * 4);
    }
    for(int i = loop_end << 2; i < dim; ++i){
        output[offset + i] = (FLOAT)exp((COMPUTE_FLOAT)(input[offset + i]) - maxValue.x) / sumValue.x;
    }
#else
    // compute maxvalue
    COMPUTE_FLOAT4 maxValue = (COMPUTE_FLOAT4)-FLT_MAX;
    for (int i = 0; i < loop_end; i++) {
        maxValue = fmax(maxValue, CONVERT_COMPUTE_FLOAT4(vload4(i, input+offset)));
    }
    maxValue.x = fmax(fmax(fmax(maxValue.x, maxValue.y), maxValue.z), maxValue.w);
    for(int i = loop_end << 2; i < dim; ++i){
        maxValue.x = fmax(maxValue.x, (COMPUTE_FLOAT)(input[offset+i]));
    }
    
    // compute sumvalue
    COMPUTE_FLOAT4 sumValue = (COMPUTE_FLOAT4)0;
    for (int i = 0; i < loop_end; i++) {
        sumValue += exp(CONVERT_COMPUTE_FLOAT4(vload4(i, input+offset)) - (COMPUTE_FLOAT4)maxValue.x);
    }
    sumValue.x = sumValue.x + sumValue.y + sumValue.z + sumValue.w;
    for(int i = loop_end << 2; i < dim; ++i){
        sumValue.x += exp((COMPUTE_FLOAT)(input[offset+i]) - maxValue.x);
    }
    
    // store result
    for(int i = 0; i < loop_end; i++){
        vstore4(CONVERT_FLOAT4(exp(CONVERT_COMPUTE_FLOAT4(vload4(i, input+offset)) - (COMPUTE_FLOAT4)maxValue.x) / (COMPUTE_FLOAT4)sumValue.x), 0, output + offset + i * 4);
    }
    for(int i = loop_end << 2; i < dim; ++i){
        output[offset + i] = (FLOAT)exp((COMPUTE_FLOAT)(input[offset + i]) - maxValue.x) / sumValue.x;
    }
#endif
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
    COMPUTE_FLOAT local sum[SOFTMAX_LOCAL_SIZE];

    COMPUTE_FLOAT maxValue = (COMPUTE_FLOAT)-FLT_MAX;
    for (int i = lid; i < dim; i+=SOFTMAX_LOCAL_SIZE) {
        maxValue = fmax(maxValue, (COMPUTE_FLOAT)(input[offset+i*inside]));
    }

    sum[lid] = maxValue;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = SOFTMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            sum[lid] = fmax(sum[lid], sum[lid + i]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    maxValue = sum[0];

    COMPUTE_FLOAT sumValue = (COMPUTE_FLOAT)0;
    for (int i = lid; i < dim; i+=SOFTMAX_LOCAL_SIZE) {
        sumValue += exp((COMPUTE_FLOAT)(input[offset+i*inside]) - maxValue);
    }
    sum[lid] = sumValue;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = SOFTMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            sum[lid] = sum[lid] + sum[lid + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sumValue = sum[0];
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
    COMPUTE_FLOAT4 local sum[SOFTMAX_LOCAL_SIZE];

    COMPUTE_FLOAT4 maxValue = (COMPUTE_FLOAT4)-FLT_MAX;
    for (int i = lid; i < dim; i+=SOFTMAX_LOCAL_SIZE) {
        maxValue = fmax(maxValue, CONVERT_COMPUTE_FLOAT4(vload4(0, input+offset+i*inside)));
    }

    sum[lid] = maxValue;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = SOFTMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            sum[lid] = fmax(sum[lid], sum[lid + i]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    maxValue = sum[0];

    COMPUTE_FLOAT4 sumValue = (COMPUTE_FLOAT4)0;
    for (int i = lid; i < dim; i+=SOFTMAX_LOCAL_SIZE) {
        sumValue += exp(CONVERT_COMPUTE_FLOAT4(vload4(0, input+offset+i*inside)) - maxValue);
    }
    sum[lid] = sumValue;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = SOFTMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            sum[lid] = sum[lid] + sum[lid + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sumValue = sum[0];
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
