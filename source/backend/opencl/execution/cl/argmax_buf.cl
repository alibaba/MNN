#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_3_DIMS \
__private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

#define ARGMAX_SELECT(a, b, c, d)    \
    a.x = b.x < c.x ? d : a.x;     \
    a.y = b.y < c.y ? d : a.y;     \
    a.z = b.z < c.z ? d : a.z;     \
    a.w = b.w < c.w ? d : a.w;     \

#define ARGMIN_SELECT(a, b, c, d)    \
    a.x = b.x > c.x ? d : a.x;     \
    a.y = b.y > c.y ? d : a.y;     \
    a.z = b.z > c.z ? d : a.z;     \
    a.w = b.w > c.w ? d : a.w;     \

__kernel void argmax_width_buf(GLOBAL_SIZE_3_DIMS
                            __global const FLOAT* input,
                            __global FLOAT* output,
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
    int4 index = 0;
    FLOAT4 maxValue = vload4(0, input + offset);
    for(int i = 1; i < inputWidth; ++i){
        FLOAT4 value = vload4(i, input + offset);
#ifdef ARGMAX
        ARGMAX_SELECT(index, maxValue, value, i);
        maxValue = fmax(maxValue, value);
#else
        ARGMIN_SELECT(index, maxValue, value, i);
        maxValue = fmin(maxValue, value);
#endif
    }
    vstore4(CONVERT_FLOAT4(index), 0, output + outputOffset);
}


__kernel void argmax_height_buf(GLOBAL_SIZE_3_DIMS
                            __global const FLOAT* input,
                            __global FLOAT* output,
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
                                
    const int offset = ((((batch_idx * inputChannelBlock) + channel_idx) * inputHeight + 0) * inputWidth + width_idx)*4;
    const int outputOffset = ((((batch_idx * outputChannelBlock) + channel_idx) * outputHeight + 0) * oututWidth + width_idx)*4;
    int4 index = 0;
    FLOAT4 maxValue = vload4(0, input + offset);
    for(int i = 1; i < inputHeight; ++i){
        FLOAT4 value = vload4(i * inputWidth, input + offset);
#ifdef ARGMAX
        ARGMAX_SELECT(index, maxValue, value, i);
        maxValue = fmax(maxValue, value);
#else
        ARGMIN_SELECT(index, maxValue, value, i);
        maxValue = fmin(maxValue, value);
#endif
    }
    vstore4(CONVERT_FLOAT4(index), 0, output + outputOffset);
}

__kernel void argmax_channel_buf(GLOBAL_SIZE_3_DIMS
                            __global const FLOAT* input,
                            __global FLOAT* output,
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
    const int batch_idx = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(width_idx, height_idx, batch_idx);
                                
    const int offset = ((((batch_idx * inputChannelBlock) + 0) * inputHeight + height_idx) * inputWidth + width_idx)*4;
    const int outputOffset = ((((batch_idx * outputChannelBlock) + 0) * outputHeight + height_idx) * oututWidth + width_idx)*4;
    int index = 0;
    int remain = inputChannel - (inputChannelBlock - 1) * 4;
#ifdef ARGMAX
    FLOAT maxValue = (FLOAT)-FLT_MAX;
#else
    FLOAT maxValue = (FLOAT)FLT_MAX;
#endif
    FLOAT4 value;
    FLOAT *valuePtr = (FLOAT*)&value;
    for(int i = 0; i < inputChannelBlock - 1; ++i){
        value = vload4(i * inputWidth * inputHeight, input + offset);
        for(int j = 0; j < 4; ++j){
#ifdef ARGMAX
            if(maxValue < valuePtr[j]){
                index = i * 4 + j;
                maxValue = valuePtr[j];
            }
#else
            if(maxValue > valuePtr[j]){
                index = i * 4 + j;
                maxValue = valuePtr[j];
            }
#endif
        }
    }
    value = vload4((inputChannelBlock - 1) * inputWidth * inputHeight, input + offset);
    for(int j = 0; j < remain; ++j){
#ifdef ARGMAX
            if(maxValue < valuePtr[j]){
                index = (inputChannelBlock - 1) * 4 + j;
                maxValue = valuePtr[j];
            }
#else
            if(maxValue > valuePtr[j]){
                index = (inputChannelBlock - 1) * 4 + j;
                maxValue = valuePtr[j];
            }
#endif
    }
    output[outputOffset] = (FLOAT)index;
}

__kernel void argmax_channel_dim1_buf(GLOBAL_SIZE_3_DIMS
                            __global const FLOAT* input,
                            __global FLOAT* output,
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
    const int batch_idx = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(width_idx, height_idx, batch_idx);
                                
    const int offset = ((((batch_idx * inputChannelBlock) + 0) * inputHeight + height_idx) * inputWidth + width_idx)*4;
    const int outputOffset = ((batch_idx * outputHeight + height_idx) * oututWidth + width_idx);
    int index = 0;
    int remain = inputChannel - (inputChannelBlock - 1) * 4;
#ifdef ARGMAX
    FLOAT maxValue = (FLOAT)-FLT_MAX;
#else
    FLOAT maxValue = (FLOAT)FLT_MAX;
#endif
    FLOAT4 value;
    FLOAT *valuePtr = (FLOAT*)&value;
    for(int i = 0; i < inputChannelBlock - 1; ++i){
        value = vload4(i * inputWidth * inputHeight, input + offset);
        for(int j = 0; j < 4; ++j){
#ifdef ARGMAX
            if(maxValue < valuePtr[j]){
                index = i * 4 + j;
                maxValue = valuePtr[j];
            }
#else
            if(maxValue > valuePtr[j]){
                index = i * 4 + j;
                maxValue = valuePtr[j];
            }
#endif
        }
    }
    value = vload4((inputChannelBlock - 1) * inputWidth * inputHeight, input + offset);
    for(int j = 0; j < remain; ++j){
#ifdef ARGMAX
            if(maxValue < valuePtr[j]){
                index = (inputChannelBlock - 1) * 4 + j;
                maxValue = valuePtr[j];
            }
#else
            if(maxValue > valuePtr[j]){
                index = (inputChannelBlock - 1) * 4 + j;
                maxValue = valuePtr[j];
            }
#endif
    }
    output[outputOffset] = (FLOAT)index;
}


__kernel void argmax_batch_buf(GLOBAL_SIZE_3_DIMS
                            __global const FLOAT* input,
                            __global FLOAT* output,
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
    const int channel_idx = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(width_idx, height_idx, channel_idx);
                                
    const int offset = ((((0 * inputChannelBlock) + channel_idx) * inputHeight + height_idx) * inputWidth + width_idx)*4;
    const int outputOffset = ((((0 * outputChannelBlock) + channel_idx) * outputHeight + height_idx) * oututWidth + width_idx)*4;
    int4 index = 0;
    int batchOffset = inputChannelBlock * inputHeight * inputWidth;
    FLOAT4 maxValue = vload4(0, input + offset);
    for(int i = 1; i < inputBatch; ++i){
        FLOAT4 value = vload4(i * batchOffset, input + offset);
#ifdef ARGMAX
        ARGMAX_SELECT(index, maxValue, value, i);
        maxValue = fmax(maxValue, value);
#else
        ARGMIN_SELECT(index, maxValue, value, i);
        maxValue = fmin(maxValue, value);
#endif
    }
    vstore4(CONVERT_FLOAT4(index), 0, output + outputOffset);
}
