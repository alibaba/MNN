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


__kernel void softmax_channel(GLOBAL_SIZE_3_DIMS
                              __global const FLOAT *input,
                              __global FLOAT *output,
                              __private const int remain_channels,
                              __private const int4 shape) {//NCHW

    const int x = get_global_id(0);
    const int w = get_global_id(1);
    const int bh = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(x, w, bh);
    
    const int batch_idx = bh / shape.z;
    const int height_idx = bh % shape.z;
    const int offset = (((batch_idx*shape.y+0)*shape.z+height_idx)*shape.w+w)*4;
#if SOFTMAX_LOCAL_SIZE >= 4
    int lid = get_local_id(0);
    FLOAT4 local sum[SOFTMAX_LOCAL_SIZE];

    FLOAT4 maxValue = (FLOAT4)-FLT_MAX;
    for (int i = lid; i < shape.y - 1; i+=SOFTMAX_LOCAL_SIZE) {
        maxValue = fmax(maxValue, vload4(i*shape.z*shape.w, input+offset));
    }

    sum[lid] = maxValue;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = SOFTMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            sum[lid] = fmax(sum[lid], sum[lid + i]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    maxValue = sum[0];

    maxValue.x = fmax(maxValue.x, maxValue.y);
    maxValue.x = fmax(maxValue.x, maxValue.z);
    maxValue.x = fmax(maxValue.x, maxValue.w);

    FLOAT4 input_data = vload4((shape.y - 1) *shape.z*shape.w, input+offset);
    if (remain_channels == 0) {
        maxValue.x = fmax(maxValue.x, input_data.x);
        maxValue.x = fmax(maxValue.x, input_data.y);
        maxValue.x = fmax(maxValue.x, input_data.z);
        maxValue.x = fmax(maxValue.x, input_data.w);
    } else if (remain_channels == 1) {
        maxValue.x = fmax(maxValue.x, input_data.z);
        maxValue.x = fmax(maxValue.x, input_data.y);
        maxValue.x = fmax(maxValue.x, input_data.x);
    } else if (remain_channels == 2) {
        maxValue.x = fmax(maxValue.x, input_data.y);
        maxValue.x = fmax(maxValue.x, input_data.x);
    } else if (remain_channels == 3) {
        maxValue.x = fmax(maxValue.x, input_data.x);
    }

    FLOAT4 sumValue = (FLOAT4)0;
    for (int i = lid; i < shape.y - 1; i+=SOFTMAX_LOCAL_SIZE) {
        sumValue += exp(vload4(i*shape.z*shape.w, input+offset) - (FLOAT4)maxValue.x);
    }
    sum[lid] = sumValue;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = SOFTMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            sum[lid] = sum[lid] + sum[lid + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sumValue = sum[0];
    sumValue.x = sumValue.x + sumValue.y + sumValue.z + sumValue.w;
    
    
    input_data -= maxValue.x;
    if (remain_channels == 0) {
        sumValue.x += exp(input_data.w);
        sumValue.x += exp(input_data.z);
        sumValue.x += exp(input_data.y);
        sumValue.x += exp(input_data.x);
    } else if (remain_channels == 1) {
        sumValue.x += exp(input_data.z);
        sumValue.x += exp(input_data.y);
        sumValue.x += exp(input_data.x);
    } else if (remain_channels == 2) {
        sumValue.x += exp(input_data.y);
        sumValue.x += exp(input_data.x);
    } else if (remain_channels == 3) {
        sumValue.x += exp(input_data.x);
    }
    for(int i = lid; i < shape.y; i+=SOFTMAX_LOCAL_SIZE){
        FLOAT4 value = exp(vload4(i*shape.z*shape.w, input+offset) - maxValue.x) / sumValue.x;
        vstore4(value, i*shape.z*shape.w, output+offset);
    }
#else
    FLOAT4 maxValue = (FLOAT4)-FLT_MAX;
    for (int i = 0; i < shape.y - 1; i++) {
        maxValue = fmax(maxValue, vload4(i*shape.z*shape.w, input+offset));
    }
    
    maxValue.x = fmax(maxValue.x, maxValue.y);
    maxValue.x = fmax(maxValue.x, maxValue.z);
    maxValue.x = fmax(maxValue.x, maxValue.w);

    FLOAT4 input_data = vload4((shape.y - 1) *shape.z*shape.w, input+offset);
    if (remain_channels == 0) {
        maxValue.x = fmax(maxValue.x, input_data.x);
        maxValue.x = fmax(maxValue.x, input_data.y);
        maxValue.x = fmax(maxValue.x, input_data.z);
        maxValue.x = fmax(maxValue.x, input_data.w);
    } else if (remain_channels == 1) {
        maxValue.x = fmax(maxValue.x, input_data.z);
        maxValue.x = fmax(maxValue.x, input_data.y);
        maxValue.x = fmax(maxValue.x, input_data.x);
    } else if (remain_channels == 2) {
        maxValue.x = fmax(maxValue.x, input_data.y);
        maxValue.x = fmax(maxValue.x, input_data.x);
    } else if (remain_channels == 3) {
        maxValue.x = fmax(maxValue.x, input_data.x);
    }

    FLOAT4 sumValue = (FLOAT4)0;
    for (int i = 0; i < shape.y - 1; i++) {
        sumValue += exp(vload4(i*shape.z*shape.w, input+offset) - (FLOAT4)maxValue.x);
    }
    sumValue.x = sumValue.x + sumValue.y + sumValue.z + sumValue.w;
    input_data -= maxValue.x;
    if (remain_channels == 0) {
        sumValue.x += exp(input_data.w);
        sumValue.x += exp(input_data.z);
        sumValue.x += exp(input_data.y);
        sumValue.x += exp(input_data.x);
    } else if (remain_channels == 1) {
        sumValue.x += exp(input_data.z);
        sumValue.x += exp(input_data.y);
        sumValue.x += exp(input_data.x);
    } else if (remain_channels == 2) {
        sumValue.x += exp(input_data.y);
        sumValue.x += exp(input_data.x);
    } else if (remain_channels == 3) {
        sumValue.x += exp(input_data.x);
    }
    for(int i = 0; i < shape.y; i++){
        FLOAT4 value = exp(vload4(i*shape.z*shape.w, input+offset) - maxValue.x) / sumValue.x;
        vstore4(value, i*shape.z*shape.w, output+offset);
    }
#endif
}


__kernel void softmax_height(GLOBAL_SIZE_3_DIMS
                             __global const FLOAT *input,
                             __global FLOAT *output,
                             __private const int remain_channels,
                             __private const int4 shape // NCHW
                             ) {
    const int x = get_global_id(0);
    const int wc = get_global_id(1);
    const int b = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(x, wc, b);
    
    const int c = wc / shape.w;
    const int w = wc % shape.w;
    const int offset = (((b*shape.y+c)*shape.z+0)*shape.w+w)*4;
#if SOFTMAX_LOCAL_SIZE >= 4
    int lid = get_local_id(0);
    FLOAT4 local sum[SOFTMAX_LOCAL_SIZE];
    
    /*Compute Max */
    FLOAT4 maxValue = (FLOAT4)(-FLT_MAX);
    for (int i=lid; i<shape.z; i+=SOFTMAX_LOCAL_SIZE) {
        maxValue = fmax(maxValue, vload4(i*shape.w, input+offset));
    }
    sum[lid] = maxValue;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = SOFTMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            sum[lid] = fmax(sum[lid], sum[lid + i]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    maxValue = sum[0];
    
    /*Compute Exp Sum*/
    FLOAT4 sumValue = (FLOAT4)0;
    for (int i=lid; i<shape.z; i+=SOFTMAX_LOCAL_SIZE) {
        sumValue += exp(vload4(i*shape.w, input+offset) - maxValue);
    }
    sum[lid] = sumValue;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = SOFTMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            sum[lid] = sum[lid] + sum[lid + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sumValue = sum[0];

    /*Compute Result */
    for (int i=lid; i<shape.z; i+=SOFTMAX_LOCAL_SIZE) {
        FLOAT4 value = exp(vload4(i*shape.w, input+offset) - maxValue) / sumValue;
        vstore4(value, i*shape.w, output+offset);
    }
#else
    /*Compute Max */
    FLOAT4 maxValue = (FLOAT4)(-FLT_MAX);
    for (int i=0; i<shape.z; i++) {
        maxValue = fmax(maxValue, vload4(i*shape.w, input+offset));
    }
    
    /*Compute Exp Sum*/
    FLOAT4 sumValue = (FLOAT4)0;
    for (int i=0; i<shape.z; i++) {
        sumValue += exp(vload4(i*shape.w, input+offset) - maxValue);
    }

    /*Compute Result */
    for (int i=0; i<shape.z; i++) {
        FLOAT4 value = exp(vload4(i*shape.w, input+offset) - maxValue) / sumValue;
        vstore4(value, i*shape.w, output+offset);
    }
#endif
}


__kernel void softmax_width(GLOBAL_SIZE_3_DIMS
                            __global const FLOAT *input,
                            __global FLOAT *output,
                            __private const int remain_channels,
                            __private const int4 shape // NCHW
                            ) {
    const int x = get_global_id(0);
    const int c = get_global_id(1);
    const int bh = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(x, c, bh);
    const int b = bh / shape.z;
    const int h = bh % shape.z;
    const int offset = (((b*shape.y+c)*shape.z+h)*shape.w+0)*4;
#if SOFTMAX_LOCAL_SIZE >= 4
    int lid = get_local_id(0);
    FLOAT4 local sum[SOFTMAX_LOCAL_SIZE];
    
    /*Compute Max */
    FLOAT4 maxValue = (FLOAT4)(-FLT_MAX);
    for (int i=lid; i<shape.w; i+=SOFTMAX_LOCAL_SIZE) {
        maxValue = fmax(maxValue, vload4(i, input+offset));
    }
    sum[lid] = maxValue;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = SOFTMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            sum[lid] = fmax(sum[lid], sum[lid + i]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    maxValue = sum[0];
    
    /*Compute Exp Sum*/
    FLOAT4 sumValue = (FLOAT4)0;
    for (int i=lid; i<shape.w; i+=SOFTMAX_LOCAL_SIZE) {
        sumValue += exp(vload4(i, input+offset) - maxValue);
    }
    sum[lid] = sumValue;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = SOFTMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            sum[lid] = sum[lid] + sum[lid + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sumValue = sum[0];
    
    /*Compute Result */
    for (int i=lid; i<shape.w; i+=SOFTMAX_LOCAL_SIZE) {
        FLOAT4 value = exp(vload4(i, input+offset) - maxValue) / sumValue;
        vstore4(value, i, output+offset);
    }
#else
    /*Compute Max */
    FLOAT4 maxValue = (FLOAT4)(-FLT_MAX);
    for (int i=0; i<shape.w; i++) {
        maxValue = fmax(maxValue, vload4(i, input+offset));
    }
    /*Compute Exp Sum*/
    FLOAT4 sumValue = (FLOAT4)0;
    for (int i=0; i<shape.w; i++) {
        sumValue += exp(vload4(i, input+offset) - maxValue);
    }
    
    /*Compute Result */
    for (int i=0; i<shape.w; i++) {
        FLOAT4 value = exp(vload4(i, input+offset) - maxValue) / sumValue;
        vstore4(value, i, output+offset);
    }
#endif
}
