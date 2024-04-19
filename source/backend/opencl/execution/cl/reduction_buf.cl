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

__kernel void reduct_width_buf(GLOBAL_SIZE_3_DIMS
                            __global const INPUT_TYPE* input,
                            __global OUTPUT_TYPE* output,
                            __private const int inputWidth,
                            __private const int inputHeight,
                            __private const int inputChannel,
                            __private const int inputBatch,
                            __private const int inputChannelBlock,
                            __private const int oututWidth,
                            __private const int outputHeight,
                            __private const int outputChannel,
                            __private const int outputChannelBlock
                            ) {
    const int width_idx = get_global_id(0);
    const int height_idx = get_global_id(1);
    const int batch_channel_idx = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(width_idx, height_idx, batch_channel_idx);
                                
    const int batch_idx = batch_channel_idx / outputChannelBlock;
    const int channel_idx = batch_channel_idx % outputChannelBlock;
    const int offset = ((((batch_idx * inputChannelBlock) + channel_idx) * inputHeight + height_idx) * inputWidth + 0)*4;
    const int outputOffset = ((((batch_idx * outputChannelBlock) + channel_idx) * outputHeight + height_idx) * oututWidth + 0)*4;
    INPUT_TYPE4 out = (INPUT_TYPE4)VALUE;
    
#if LOCAL_SIZE > 0
    const int lid = get_local_id(0);
    INPUT_TYPE4 local sum[LOCAL_SIZE];
    for(int i = lid; i < inputWidth; i+=LOCAL_SIZE){
        INPUT_TYPE4 in = vload4(i, input + offset);
        out = OPERATE(out, in);
    }
    sum[lid] = out;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            sum[lid] = OPERATE(sum[lid], sum[lid + i]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    out = sum[0];
#else
    for(int i = 0; i < inputWidth; ++i){
        INPUT_TYPE4 in = vload4(i, input + offset);
        out = OPERATE(out, in);
    }
#endif

#ifdef GET_AVG
    out = out / inputWidth;
#endif
    vstore4(CONVERT_OUTPUT4(out), 0, output + outputOffset);
}


__kernel void reduct_height_buf(GLOBAL_SIZE_3_DIMS
                            __global const INPUT_TYPE* input,
                            __global OUTPUT_TYPE* output,
                            __private const int inputWidth,
                            __private const int inputHeight,
                            __private const int inputChannel,
                            __private const int inputBatch,
                            __private const int inputChannelBlock,
                            __private const int oututWidth,
                            __private const int outputHeight,
                            __private const int outputChannel,
                            __private const int outputChannelBlock
                            ) {
#if LOCAL_SIZE > 0
    const int width_local_idx = get_global_id(0);
    const int height_idx = get_global_id(1);
    const int batch_channel_idx = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(width_local_idx, height_idx, batch_channel_idx);
    
    const int width_idx = get_group_id(0);
    const int batch_idx = batch_channel_idx / outputChannelBlock;
    const int channel_idx = batch_channel_idx % outputChannelBlock;
    
    const int offset = ((((batch_idx * inputChannelBlock) + channel_idx) * inputHeight + 0) * inputWidth + width_idx)*4;
    const int outputOffset = ((((batch_idx * outputChannelBlock) + channel_idx) * outputHeight + 0) * oututWidth + width_idx)*4;
    const int lid = get_local_id(0);
    INPUT_TYPE4 local sum[LOCAL_SIZE];
    INPUT_TYPE4 out = (INPUT_TYPE4)VALUE;
    for(int i = lid; i < inputHeight; i+=LOCAL_SIZE){
        INPUT_TYPE4 in = vload4(i * inputWidth, input + offset);
        out = OPERATE(out, in);
    }
    sum[lid] = out;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            sum[lid] = OPERATE(sum[lid], sum[lid + i]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    out = sum[0];
#else

    const int width_idx = get_global_id(0);
    const int height_idx = get_global_id(1);
    const int batch_channel_idx = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(width_idx, height_idx, batch_channel_idx);
    
    const int batch_idx = batch_channel_idx / outputChannelBlock;
    const int channel_idx = batch_channel_idx % outputChannelBlock;
    
    const int offset = ((((batch_idx * inputChannelBlock) + channel_idx) * inputHeight + 0) * inputWidth + width_idx)*4;
    const int outputOffset = ((((batch_idx * outputChannelBlock) + channel_idx) * outputHeight + 0) * oututWidth + width_idx)*4;
    INPUT_TYPE4 out = (INPUT_TYPE4)VALUE;
    for(int i = 0; i < inputHeight; ++i){
        INPUT_TYPE4 in = vload4(i * inputWidth, input + offset);
        out = OPERATE(out, in);
    }
#endif
    
#ifdef GET_AVG
    out = out / inputHeight;
#endif
    vstore4(CONVERT_OUTPUT4(out), 0, output + outputOffset);
}

__kernel void reduct_channel_buf(GLOBAL_SIZE_3_DIMS
                            __global const INPUT_TYPE* input,
                            __global OUTPUT_TYPE* output,
                            __private const int inputWidth,
                            __private const int inputHeight,
                            __private const int inputChannel,
                            __private const int inputBatch,
                            __private const int inputChannelBlock,
                            __private const int oututWidth,
                            __private const int outputHeight,
                            __private const int outputChannel,
                            __private const int outputChannelBlock
                            ) {
#if LOCAL_SIZE > 0
    const int width_local_idx = get_global_id(0);
    const int height_idx = get_global_id(1);
    const int batch_idx = get_global_id(2);
    
    DEAL_NON_UNIFORM_DIM3(width_local_idx, height_idx, batch_idx);
    const int width_idx = get_group_id(0);
    
    const int offset = ((((batch_idx * inputChannelBlock) + 0) * inputHeight + height_idx) * inputWidth + width_idx)*4;
    const int outputOffset = ((((batch_idx * outputChannelBlock) + 0) * outputHeight + height_idx) * oututWidth + width_idx)*4;
    int remain = inputChannel - (inputChannelBlock - 1) * 4;
    const int lid = get_local_id(0);
    INPUT_TYPE local sum[LOCAL_SIZE];
    INPUT_TYPE4 out = (INPUT_TYPE4)VALUE;
    INPUT_TYPE4 in;
    INPUT_TYPE *inPtr = (INPUT_TYPE*)&in;
    for(int i = lid; i < inputChannelBlock - 1; i += LOCAL_SIZE){
        in = vload4(i * inputWidth * inputHeight, input + offset);
        out = OPERATE(out, in);
    }
    out.x = OPERATE(out.x, out.y);
    out.x = OPERATE(out.x, out.z);
    out.x = OPERATE(out.x, out.w);
    sum[lid] = out.x;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            sum[lid] = OPERATE(sum[lid], sum[lid + i]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    out.x = sum[0];
    in = vload4((inputChannelBlock - 1) * inputWidth * inputHeight, input + offset);
    for(int j = 0; j < remain; ++j){
        out.x = OPERATE(out.x, inPtr[j]);
    }
#ifdef GET_AVG
    out.x = out.x / inputChannel;
#endif
    output[outputOffset] = (OUTPUT_TYPE)out.x;
    
#else
    const int width_idx = get_global_id(0);
    const int height_idx = get_global_id(1);
    const int batch_idx = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(width_idx, height_idx, batch_idx);
                                
    const int offset = ((((batch_idx * inputChannelBlock) + 0) * inputHeight + height_idx) * inputWidth + width_idx)*4;
    const int outputOffset = ((((batch_idx * outputChannelBlock) + 0) * outputHeight + height_idx) * oututWidth + width_idx)*4;
    int remain = inputChannel - (inputChannelBlock - 1) * 4;
    
    INPUT_TYPE out = (INPUT_TYPE)VALUE;
    INPUT_TYPE4 in;
    INPUT_TYPE *inPtr = (INPUT_TYPE*)&in;
    for(int i = 0; i < inputChannelBlock - 1; ++i){
        in = vload4(i * inputWidth * inputHeight, input + offset);
        for(int j = 0; j < 4; ++j){
            out = OPERATE(out, inPtr[j]);
        }
    }
    in = vload4((inputChannelBlock - 1) * inputWidth * inputHeight, input + offset);
    for(int j = 0; j < remain; ++j){
        out = OPERATE(out, inPtr[j]);
    }
#ifdef GET_AVG
    out = out / inputChannel;
#endif
    output[outputOffset] = (OUTPUT_TYPE)out;
#endif
}

__kernel void reduct_channel_dim1_buf(GLOBAL_SIZE_3_DIMS
                            __global const INPUT_TYPE* input,
                            __global OUTPUT_TYPE* output,
                            __private const int inputWidth,
                            __private const int inputHeight,
                            __private const int inputChannel,
                            __private const int inputBatch,
                            __private const int inputChannelBlock,
                            __private const int oututWidth,
                            __private const int outputHeight,
                            __private const int outputChannel,
                            __private const int outputChannelBlock
                            ) {
#if LOCAL_SIZE > 0
    const int width_local_idx = get_global_id(0);
    const int height_idx = get_global_id(1);
    const int batch_idx = get_global_id(2);
    
    DEAL_NON_UNIFORM_DIM3(width_local_idx, height_idx, batch_idx);
    const int width_idx = get_group_id(0);
    
    const int offset = ((((batch_idx * inputChannelBlock) + 0) * inputHeight + height_idx) * inputWidth + width_idx)*4;
    const int outputOffset = ((batch_idx * outputHeight + height_idx) * oututWidth + width_idx);
    int remain = inputChannel - (inputChannelBlock - 1) * 4;
    const int lid = get_local_id(0);
    INPUT_TYPE local sum[LOCAL_SIZE];
    INPUT_TYPE4 out = (INPUT_TYPE4)VALUE;
    INPUT_TYPE4 in;
    INPUT_TYPE *inPtr = (INPUT_TYPE*)&in;
    for(int i = lid; i < inputChannelBlock - 1; i += LOCAL_SIZE){
        in = vload4(i * inputWidth * inputHeight, input + offset);
        out = OPERATE(out, in);
    }
    out.x = OPERATE(out.x, out.y);
    out.x = OPERATE(out.x, out.z);
    out.x = OPERATE(out.x, out.w);
    sum[lid] = out.x;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            sum[lid] = OPERATE(sum[lid], sum[lid + i]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    out.x = sum[0];
    in = vload4((inputChannelBlock - 1) * inputWidth * inputHeight, input + offset);
    for(int j = 0; j < remain; ++j){
        out.x = OPERATE(out.x, inPtr[j]);
    }
#ifdef GET_AVG
    out.x = out.x / inputChannel;
#endif
    output[outputOffset] = (OUTPUT_TYPE)out.x;
    
#else
    const int width_idx = get_global_id(0);
    const int height_idx = get_global_id(1);
    const int batch_idx = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(width_idx, height_idx, batch_idx);
    const int offset = ((((batch_idx * inputChannelBlock) + 0) * inputHeight + height_idx) * inputWidth + width_idx)*4;
    const int outputOffset = ((batch_idx * outputHeight + height_idx) * oututWidth + width_idx);
    int remain = inputChannel - (inputChannelBlock - 1) * 4;
    INPUT_TYPE out = (INPUT_TYPE)VALUE;
    INPUT_TYPE4 in;
    INPUT_TYPE *inPtr = (INPUT_TYPE*)&in;
    for(int i = 0; i < inputChannelBlock - 1; ++i){
        in = vload4(i * inputWidth * inputHeight, input + offset);
        for(int j = 0; j < 4; ++j){
            out = OPERATE(out, inPtr[j]);
        }
    }
    in = vload4((inputChannelBlock - 1) * inputWidth * inputHeight, input + offset);
    for(int j = 0; j < remain; ++j){
        out = OPERATE(out, inPtr[j]);
    }
#ifdef GET_AVG
    out = out / inputChannel;
#endif
    output[outputOffset] = (OUTPUT_TYPE)out;
#endif
}


__kernel void reduct_batch_buf(GLOBAL_SIZE_3_DIMS
                            __global const INPUT_TYPE* input,
                            __global OUTPUT_TYPE* output,
                            __private const int inputWidth,
                            __private const int inputHeight,
                            __private const int inputChannel,
                            __private const int inputBatch,
                            __private const int inputChannelBlock,
                            __private const int oututWidth,
                            __private const int outputHeight,
                            __private const int outputChannel,
                            __private const int outputChannelBlock
                            ) {
#if LOCAL_SIZE > 0
    const int width_local_idx = get_global_id(0);
    const int height_idx = get_global_id(1);
    const int channel_idx = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(width_local_idx, height_idx, channel_idx);
    const int width_idx = get_group_id(0);
                            
    const int offset = ((((0 * inputChannelBlock) + channel_idx) * inputHeight + height_idx) * inputWidth + width_idx)*4;
    const int outputOffset = ((((0 * outputChannelBlock) + channel_idx) * outputHeight + height_idx) * oututWidth + width_idx)*4;
    int batchOffset = inputChannelBlock * inputHeight * inputWidth;
    const int lid = get_local_id(0);
    INPUT_TYPE4 local sum[LOCAL_SIZE];
    INPUT_TYPE4 out = (INPUT_TYPE4)VALUE;
    for(int i = lid; i < inputBatch; i+=LOCAL_SIZE){
        INPUT_TYPE4 in = vload4(i * batchOffset, input + offset);
        out = OPERATE(out, in);
    }
    sum[lid] = out;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i)
            sum[lid] = OPERATE(sum[lid], sum[lid + i]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    out = sum[0];
#ifdef GET_AVG
    out = out / inputBatch;
#endif
    vstore4(CONVERT_OUTPUT4(out), 0, output + outputOffset);
#else
    const int width_idx = get_global_id(0);
    const int height_idx = get_global_id(1);
    const int channel_idx = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(width_idx, height_idx, channel_idx);
                                
    const int offset = ((((0 * inputChannelBlock) + channel_idx) * inputHeight + height_idx) * inputWidth + width_idx)*4;
    const int outputOffset = ((((0 * outputChannelBlock) + channel_idx) * outputHeight + height_idx) * oututWidth + width_idx)*4;
    int batchOffset = inputChannelBlock * inputHeight * inputWidth;
    INPUT_TYPE4 out = (INPUT_TYPE4)VALUE;
    for(int i = 0; i < inputBatch; ++i){
        INPUT_TYPE4 in = vload4(i * batchOffset, input + offset);
        out = OPERATE(out, in);
    }
#ifdef GET_AVG
    out = out / inputBatch;
#endif
    vstore4(CONVERT_OUTPUT4(out), 0, output + outputOffset);
#endif
}
