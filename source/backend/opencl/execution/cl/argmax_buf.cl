#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_3_DIMS \
__private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

#define ARGMAX_SELECT(A, B, C, D)          \
    if(A.x < B.x){ A.x = B.x; C.x = D; }    \
    if(A.y < B.y){ A.y = B.y; C.y = D; }    \
    if(A.z < B.z){ A.z = B.z; C.z = D; }    \
    if(A.w < B.w){ A.w = B.w; C.w = D; }    

#define ARGMIN_SELECT(A, B, C, D)    \
    if(A.x > B.x){ A.x = B.x; C.x = D; }    \
    if(A.y > B.y){ A.y = B.y; C.y = D; }    \
    if(A.z > B.z){ A.z = B.z; C.z = D; }    \
    if(A.w > B.w){ A.w = B.w; C.w = D; }    

__kernel void argmax_width_buf(GLOBAL_SIZE_3_DIMS
                            __global const FLOAT* input,
                            __global int* output,
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
    const int x = get_global_id(0);
    const int height_idx = get_global_id(1);
    const int batch_channel_idx = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(x, height_idx, batch_channel_idx);
                                
    const int batch_idx = batch_channel_idx / outputChannelBlock;
    const int channel_idx = batch_channel_idx % outputChannelBlock;
                                
    const int offset = ((((batch_idx * inputChannelBlock) + channel_idx) * inputHeight + height_idx) * inputWidth + 0)*4;
    const int outputOffset = ((((batch_idx * outputChannelBlock) + channel_idx) * outputHeight + height_idx) * oututWidth + 0)*4;
    int4 index = 0;
#ifdef ARGMAX
    FLOAT4 maxValue = (FLOAT4)-FLT_MAX;
#else
    FLOAT4 maxValue = (FLOAT4)FLT_MAX;
#endif
#if ARGMAX_LOCAL_SIZE >= 4
    int lid = get_local_id(0);
    FLOAT4 local reduce[ARGMAX_LOCAL_SIZE];
    int4 local index_reduce[ARGMAX_LOCAL_SIZE];
    
    for (int i=lid; i < inputWidth; i+=ARGMAX_LOCAL_SIZE) {
        FLOAT4 value = vload4(i, input + offset);
#ifdef ARGMAX
        ARGMAX_SELECT(maxValue, value, index, i);
#else
        ARGMIN_SELECT(maxValue, value, index, i);
#endif
    }
    reduce[lid] = maxValue;
    index_reduce[lid] = index;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = ARGMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i){
#ifdef ARGMAX
            if(reduce[lid].x < reduce[lid + i].x){reduce[lid].x = reduce[lid + i].x; index_reduce[lid].x = index_reduce[lid + i].x;}
            if(reduce[lid].y < reduce[lid + i].y){reduce[lid].y = reduce[lid + i].y; index_reduce[lid].y = index_reduce[lid + i].y;}
            if(reduce[lid].z < reduce[lid + i].z){reduce[lid].z = reduce[lid + i].z; index_reduce[lid].z = index_reduce[lid + i].z;}
            if(reduce[lid].w < reduce[lid + i].w){reduce[lid].w = reduce[lid + i].w; index_reduce[lid].w = index_reduce[lid + i].w;}
#else
            if(reduce[lid].x > reduce[lid + i].x){reduce[lid].x = reduce[lid + i].x; index_reduce[lid].x = index_reduce[lid + i].x;}
            if(reduce[lid].y > reduce[lid + i].y){reduce[lid].y = reduce[lid + i].y; index_reduce[lid].y = index_reduce[lid + i].y;}
            if(reduce[lid].z > reduce[lid + i].z){reduce[lid].z = reduce[lid + i].z; index_reduce[lid].z = index_reduce[lid + i].z;}
            if(reduce[lid].w > reduce[lid + i].w){reduce[lid].w = reduce[lid + i].w; index_reduce[lid].w = index_reduce[lid + i].w;}
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid == 0){
        vstore4(index_reduce[0], 0, output + outputOffset);
    }
#else
    for(int i = 0; i < inputWidth; ++i){
        FLOAT4 value = vload4(i, input + offset);
#ifdef ARGMAX
        ARGMAX_SELECT(maxValue, value, index, i);
#else
        ARGMIN_SELECT(maxValue, value, index, i);
#endif
    }
    vstore4(index, 0, output + outputOffset);
#endif
}


__kernel void argmax_height_buf(GLOBAL_SIZE_3_DIMS
                            __global const FLOAT* input,
                            __global int* output,
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
    const int x = get_global_id(0);
    const int width_idx = get_global_id(1);
    const int batch_channel_idx = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(x, width_idx, batch_channel_idx);
                                
    const int batch_idx = batch_channel_idx / outputChannelBlock;
    const int channel_idx = batch_channel_idx % outputChannelBlock;
                                
    const int offset = ((((batch_idx * inputChannelBlock) + channel_idx) * inputHeight + 0) * inputWidth + width_idx)*4;
    const int outputOffset = ((((batch_idx * outputChannelBlock) + channel_idx) * outputHeight + 0) * oututWidth + width_idx)*4;
    int4 index = 0;
#ifdef ARGMAX
    FLOAT4 maxValue = (FLOAT4)-FLT_MAX;
#else
    FLOAT4 maxValue = (FLOAT4)FLT_MAX;
#endif
#if ARGMAX_LOCAL_SIZE >= 4
    int lid = get_local_id(0);
    FLOAT4 local reduce[ARGMAX_LOCAL_SIZE];
    int4 local index_reduce[ARGMAX_LOCAL_SIZE];
    
    for (int i=lid; i < inputHeight; i+=ARGMAX_LOCAL_SIZE) {
        FLOAT4 value = vload4(i * inputWidth, input + offset);
#ifdef ARGMAX
        ARGMAX_SELECT(maxValue, value, index, i);
#else
        ARGMIN_SELECT(maxValue, value, index, i);
#endif
    }
    reduce[lid] = maxValue;
    index_reduce[lid] = index;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = ARGMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i){
#ifdef ARGMAX
            if(reduce[lid].x < reduce[lid + i].x){reduce[lid].x = reduce[lid + i].x; index_reduce[lid].x = index_reduce[lid + i].x;}
            if(reduce[lid].y < reduce[lid + i].y){reduce[lid].y = reduce[lid + i].y; index_reduce[lid].y = index_reduce[lid + i].y;}
            if(reduce[lid].z < reduce[lid + i].z){reduce[lid].z = reduce[lid + i].z; index_reduce[lid].z = index_reduce[lid + i].z;}
            if(reduce[lid].w < reduce[lid + i].w){reduce[lid].w = reduce[lid + i].w; index_reduce[lid].w = index_reduce[lid + i].w;}
#else
            if(reduce[lid].x > reduce[lid + i].x){reduce[lid].x = reduce[lid + i].x; index_reduce[lid].x = index_reduce[lid + i].x;}
            if(reduce[lid].y > reduce[lid + i].y){reduce[lid].y = reduce[lid + i].y; index_reduce[lid].y = index_reduce[lid + i].y;}
            if(reduce[lid].z > reduce[lid + i].z){reduce[lid].z = reduce[lid + i].z; index_reduce[lid].z = index_reduce[lid + i].z;}
            if(reduce[lid].w > reduce[lid + i].w){reduce[lid].w = reduce[lid + i].w; index_reduce[lid].w = index_reduce[lid + i].w;}
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid == 0){
        vstore4(index_reduce[0], 0, output + outputOffset);
    }
#else
    for(int i = 0; i < inputHeight; ++i){
        FLOAT4 value = vload4(i * inputWidth, input + offset);
#ifdef ARGMAX
        ARGMAX_SELECT(maxValue, value, index, i);
#else
        ARGMIN_SELECT(maxValue, value, index, i);
#endif
    }
    vstore4(index, 0, output + outputOffset);
#endif
}

__kernel void argmax_channel_buf(GLOBAL_SIZE_3_DIMS
                            __global const FLOAT* input,
                            __global int* output,
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
    const int x = get_global_id(0);
    const int wh = get_global_id(1);
    const int batch_idx = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(x, wh, batch_idx);
                                
    const int width_idx = wh % oututWidth;
    const int height_idx = wh / oututWidth;
    const int offset = ((((batch_idx * inputChannelBlock) + 0) * inputHeight + height_idx) * inputWidth + width_idx)*4;
#ifdef ARGMAX_CHANNEL_DIM1
    const int outputOffset = ((batch_idx * outputHeight + height_idx) * oututWidth + width_idx);
#else
    const int outputOffset = ((((batch_idx * outputChannelBlock) + 0) * outputHeight + height_idx) * oututWidth + width_idx)*4;
#endif
    int remain = inputChannel - (inputChannelBlock - 1) * 4;
#ifdef ARGMAX
    FLOAT maxValue = (FLOAT)-FLT_MAX;
#else
    FLOAT maxValue = (FLOAT)FLT_MAX;
#endif
    int index = 0;
    FLOAT4 value;
    FLOAT *valuePtr = (FLOAT*)&value;
#if ARGMAX_LOCAL_SIZE >= 4
    int lid = get_local_id(0);
    FLOAT local reduce[ARGMAX_LOCAL_SIZE];
    int local index_reduce[ARGMAX_LOCAL_SIZE];
    
    for (int i=lid; i < inputChannelBlock - 1; i+=ARGMAX_LOCAL_SIZE) {
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
    reduce[lid] = maxValue;
    index_reduce[lid] = index;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = ARGMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i){
#ifdef ARGMAX
            if(reduce[lid] < reduce[lid + i]){reduce[lid] = reduce[lid + i]; index_reduce[lid] = index_reduce[lid + i];}
#else
            if(reduce[lid] > reduce[lid + i]){reduce[lid] = reduce[lid + i]; index_reduce[lid] = index_reduce[lid + i];}
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid == 0){
        maxValue = reduce[lid];
        index = index_reduce[lid];
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
        output[outputOffset] = index;
    }
#else
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
    output[outputOffset] = index;
#endif
}

__kernel void argmax_batch_buf(GLOBAL_SIZE_3_DIMS
                            __global const FLOAT* input,
                            __global int* output,
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
    const int x = get_global_id(0);
    const int wh = get_global_id(1);
    const int channel_idx = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(x, wh, channel_idx);
    
    const int width_idx = wh % oututWidth;
    const int height_idx = wh / oututWidth;
    const int offset = ((((0 * inputChannelBlock) + channel_idx) * inputHeight + height_idx) * inputWidth + width_idx)*4;
    const int outputOffset = ((((0 * outputChannelBlock) + channel_idx) * outputHeight + height_idx) * oututWidth + width_idx)*4;
    int4 index = 0;
    int batchOffset = inputChannelBlock * inputHeight * inputWidth;
#ifdef ARGMAX
    FLOAT4 maxValue = (FLOAT4)-FLT_MAX;
#else
    FLOAT4 maxValue = (FLOAT4)FLT_MAX;
#endif
#if ARGMAX_LOCAL_SIZE >= 4
    int lid = get_local_id(0);
    FLOAT4 local reduce[ARGMAX_LOCAL_SIZE];
    int4 local index_reduce[ARGMAX_LOCAL_SIZE];
    
    for (int i=lid; i < inputBatch; i+=ARGMAX_LOCAL_SIZE) {
        FLOAT4 value = vload4(i * batchOffset, input + offset);
#ifdef ARGMAX
        ARGMAX_SELECT(maxValue, value, index, i);
#else
        ARGMIN_SELECT(maxValue, value, index, i);
#endif
    }
    reduce[lid] = maxValue;
    index_reduce[lid] = index;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = ARGMAX_LOCAL_SIZE/2; i > 0; i /= 2){
        if (lid < i){
#ifdef ARGMAX
            if(reduce[lid].x < reduce[lid + i].x){reduce[lid].x = reduce[lid + i].x; index_reduce[lid].x = index_reduce[lid + i].x;}
            if(reduce[lid].y < reduce[lid + i].y){reduce[lid].y = reduce[lid + i].y; index_reduce[lid].y = index_reduce[lid + i].y;}
            if(reduce[lid].z < reduce[lid + i].z){reduce[lid].z = reduce[lid + i].z; index_reduce[lid].z = index_reduce[lid + i].z;}
            if(reduce[lid].w < reduce[lid + i].w){reduce[lid].w = reduce[lid + i].w; index_reduce[lid].w = index_reduce[lid + i].w;}
#else
            if(reduce[lid].x > reduce[lid + i].x){reduce[lid].x = reduce[lid + i].x; index_reduce[lid].x = index_reduce[lid + i].x;}
            if(reduce[lid].y > reduce[lid + i].y){reduce[lid].y = reduce[lid + i].y; index_reduce[lid].y = index_reduce[lid + i].y;}
            if(reduce[lid].z > reduce[lid + i].z){reduce[lid].z = reduce[lid + i].z; index_reduce[lid].z = index_reduce[lid + i].z;}
            if(reduce[lid].w > reduce[lid + i].w){reduce[lid].w = reduce[lid + i].w; index_reduce[lid].w = index_reduce[lid + i].w;}
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid == 0){
        vstore4(index_reduce[0], 0, output + outputOffset);
    }
#else
    for(int i = 0; i < inputBatch; ++i){
        FLOAT4 value = vload4(i * batchOffset, input + offset);
#ifdef ARGMAX
        ARGMAX_SELECT(maxValue, value, index, i);
#else
        ARGMIN_SELECT(maxValue, value, index, i);
#endif
    }
    vstore4(index, 0, output + outputOffset);
#endif
}
