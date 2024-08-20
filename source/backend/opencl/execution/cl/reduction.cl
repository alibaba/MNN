// TODO: use INIT_SCALAR_VALUE, OPERATOR, FINAL_OPERATOR_ON_CHANNEL macro abstract and simplify code
// TODO: support reduce dims include batch
// TODO: support keep_dim=False
// TODO: fix channel reduce result re-pack problem
#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define GLOBAL_SIZE_2_DIMS \
__private const int global_size_dim0, __private const int global_size_dim1,

#define GLOBAL_SIZE_3_DIMS \
__private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }


__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void reduct_width(GLOBAL_SIZE_3_DIMS
                            __read_only image2d_t input,
                            __write_only image2d_t output,
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
    const int bh = batch_idx*inputHeight+height_idx;
    const int wc = channel_idx*inputWidth;
    INPUT_TYPE_I4 out = (INPUT_TYPE_I4)VALUE;
    
#if LOCAL_SIZE > 0
    const int lid = get_local_id(0);
    INPUT_TYPE_I4 local sum[LOCAL_SIZE];
    for(int i = lid; i < inputWidth; i+=LOCAL_SIZE){
        INPUT_TYPE_I4 in = RI_DATA(input, SAMPLER, (int2)(wc+i, bh));
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
        INPUT_TYPE_I4 in = RI_DATA(input, SAMPLER, (int2)(wc+i, bh));
        out = OPERATE(out, in);
    }
#endif

#ifdef GET_AVG
    out = out / inputWidth;
#endif
    WI_DATA(output, (int2)(channel_idx, bh), CONVERT_OUTPUT_I4(out));
}


__kernel void reduct_height(GLOBAL_SIZE_3_DIMS
                            __read_only image2d_t input,
                            __write_only image2d_t output,
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
    
    const int bh = batch_idx*inputHeight;
    const int wc = channel_idx*inputWidth+width_idx;
    const int lid = get_local_id(0);
    INPUT_TYPE_I4 local sum[LOCAL_SIZE];
    INPUT_TYPE_I4 out = (INPUT_TYPE_I4)VALUE;
    for(int i = lid; i < inputHeight; i+=LOCAL_SIZE){
        INPUT_TYPE_I4 in = RI_DATA(input, SAMPLER, (int2)(wc, bh+i));
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
    
    const int bh = batch_idx*inputHeight;
    const int wc = channel_idx*inputWidth+width_idx;
    INPUT_TYPE_I4 out = (INPUT_TYPE_I4)VALUE;
    for(int i = 0; i < inputHeight; ++i){
        INPUT_TYPE_I4 in = RI_DATA(input, SAMPLER, (int2)(wc, bh+i));
        out = OPERATE(out, in);
    }
#endif
    
#ifdef GET_AVG
    out = out / inputHeight;
#endif
    WI_DATA(output, (int2)(wc, batch_idx), CONVERT_OUTPUT_I4(out));
}

__kernel void reduct_channel(GLOBAL_SIZE_3_DIMS
                            __read_only image2d_t input,
                            __write_only image2d_t output,
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
    
    const int bh = batch_idx*inputHeight+height_idx;
    const int wc = width_idx;
    int remain = inputChannel - (inputChannelBlock - 1) * 4;
    const int lid = get_local_id(0);
    INPUT_TYPE_I local sum[LOCAL_SIZE];
    INPUT_TYPE_I4 out = (INPUT_TYPE_I4)VALUE;
    INPUT_TYPE_I4 in;
    INPUT_TYPE_I *inPtr = (INPUT_TYPE_I*)&in;
    for(int i = lid; i < inputChannelBlock - 1; i += LOCAL_SIZE){
        in = RI_DATA(input, SAMPLER, (int2)(i*inputWidth+wc, bh));
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
    in = RI_DATA(input, SAMPLER, (int2)((inputChannelBlock - 1)*inputWidth+wc, bh));
    for(int j = 0; j < remain; ++j){
        out.x = OPERATE(out.x, inPtr[j]);
    }
#ifdef GET_AVG
    out.x = out.x / inputChannel;
#endif
    WI_DATA(output, (int2)(wc, bh), (OUTPUT_TYPE_I4)(out.x, 0, 0, 0));
    
#else
    const int width_idx = get_global_id(0);
    const int height_idx = get_global_id(1);
    const int batch_idx = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(width_idx, height_idx, batch_idx);
    
    const int bh = batch_idx*inputHeight+height_idx;
    const int wc = width_idx;
    int remain = inputChannel - (inputChannelBlock - 1) * 4;
    
    INPUT_TYPE_I out = (INPUT_TYPE_I)VALUE;
    INPUT_TYPE_I4 in;
    INPUT_TYPE_I *inPtr = (INPUT_TYPE_I*)&in;
    
    for(int i = 0; i < inputChannelBlock - 1; ++i){
        in = RI_DATA(input, SAMPLER, (int2)(i*inputWidth+wc, bh));
        for(int j = 0; j < 4; ++j){
            out = OPERATE(out, inPtr[j]);
        }
    }
    in = RI_DATA(input, SAMPLER, (int2)((inputChannelBlock - 1)*inputWidth+wc, bh));
    for(int j = 0; j < remain; ++j){
        out = OPERATE(out, inPtr[j]);
    }
#ifdef GET_AVG
    out = out / inputChannel;
#endif
    WI_DATA(output, (int2)(wc, bh), (OUTPUT_TYPE_I4)(out, 0, 0, 0));
#endif
}

__kernel void reduct_batch(GLOBAL_SIZE_3_DIMS
                            __read_only image2d_t input,
                            __write_only image2d_t output,
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
                            
    const int bh = height_idx;
    const int wc = channel_idx*inputWidth+width_idx;
    int batchOffset = inputChannelBlock * inputHeight * inputWidth;
    const int lid = get_local_id(0);
    INPUT_TYPE_I4 local sum[LOCAL_SIZE];
    INPUT_TYPE_I4 out = (INPUT_TYPE_I4)VALUE;
    for(int i = lid; i < inputBatch; i+=LOCAL_SIZE){
        INPUT_TYPE_I4 in = RI_DATA(input, SAMPLER, (int2)(wc, i*inputHeight+bh));
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
    WI_DATA(output, (int2)(wc, bh), CONVERT_OUTPUT_I4(out));
#else
    const int width_idx = get_global_id(0);
    const int height_idx = get_global_id(1);
    const int channel_idx = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(width_idx, height_idx, channel_idx);
    
    const int bh = height_idx;
    const int wc = channel_idx*inputWidth+width_idx;
    int batchOffset = inputChannelBlock * inputHeight * inputWidth;
    INPUT_TYPE_I4 out = (INPUT_TYPE_I4)VALUE;
    for(int i = 0; i < inputBatch; ++i){
        INPUT_TYPE_I4 in = RI_DATA(input, SAMPLER, (int2)(wc, i*inputHeight+bh));
        out = OPERATE(out, in);
    }
#ifdef GET_AVG
    out = out / inputBatch;
#endif
    WI_DATA(output, (int2)(wc, bh), CONVERT_OUTPUT_I4(out));
#endif
}

