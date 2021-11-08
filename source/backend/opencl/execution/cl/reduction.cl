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

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;


__kernel void reduct_general_mean(GLOBAL_SIZE_2_DIMS
                            __read_only image2d_t input,
                            __write_only image2d_t output,
                            __private const int batch,
                            __private const int height,
                            __private const int width
                            ) {
    const int batch_idx = get_global_id(0);
    const int width_idx = get_global_id(1);

    FLOAT sum = 0;
    for (int h = 0; h < height; h++) {
        FLOAT4 in = RI_F(input, SAMPLER, (int2)(width_idx, batch_idx*height+h));
        sum = sum + in.x;
    }
    WI_F(output, (int2)(width_idx, batch_idx), (FLOAT4)(sum/height, 0.0, 0.0, 0.0));
}
__kernel void reduct_general_sum(GLOBAL_SIZE_2_DIMS
                            __read_only image2d_t input,
                            __write_only image2d_t output,
                            __private const int batch,
                            __private const int height,
                            __private const int width
                            ) {
    const int batch_idx = get_global_id(0);
    const int width_idx = get_global_id(1);

    FLOAT sum = 0;
    for (int h = 0; h < height; h++) {
        FLOAT4 in = RI_F(input, SAMPLER, (int2)(width_idx, batch_idx*height+h));
        sum = sum + in.x;
    }
    WI_F(output, (int2)(width_idx, batch_idx), (FLOAT4)(sum, 0.0, 0.0, 0.0));
}

__kernel void reduct_general_max(GLOBAL_SIZE_2_DIMS
                            __read_only image2d_t input,
                            __write_only image2d_t output,
                            __private const int batch,
                            __private const int height,
                            __private const int width
                            ) {
    const int batch_idx = get_global_id(0);
    const int width_idx = get_global_id(1);

    FLOAT sum = -MAXFLOAT;
    for (int h = 0; h < height; h++) {
        FLOAT4 in = RI_F(input, SAMPLER, (int2)(width_idx, batch_idx*height+h));
        sum = max(sum, in.x);
    }
    WI_F(output, (int2)(width_idx, batch_idx), (FLOAT4)(sum, 0.0, 0.0, 0.0));
}

__kernel void reduct_general_min(GLOBAL_SIZE_2_DIMS
                            __read_only image2d_t input,
                            __write_only image2d_t output,
                            __private const int batch,
                            __private const int height,
                            __private const int width
                            ) {
    const int batch_idx = get_global_id(0);
    const int width_idx = get_global_id(1);

    FLOAT sum = MAXFLOAT;
    for (int h = 0; h < height; h++) {
        FLOAT4 in = RI_F(input, SAMPLER, (int2)(width_idx, batch_idx*height+h));
        sum = min(sum, in.x);
    }
    WI_F(output, (int2)(width_idx, batch_idx), (FLOAT4)(sum, 0.0, 0.0, 0.0));
}

__kernel void reduct_general_mul(GLOBAL_SIZE_2_DIMS
                            __read_only image2d_t input,
                            __write_only image2d_t output,
                            __private const int batch,
                            __private const int height,
                            __private const int width
                            ) {
    const int batch_idx = get_global_id(0);
    const int width_idx = get_global_id(1);

    FLOAT sum = 1.0;
    for (int h = 0; h < height; h++) {
        FLOAT4 in = RI_F(input, SAMPLER, (int2)(width_idx, batch_idx*height+h));
        sum = sum * in.x;
    }
    WI_F(output, (int2)(width_idx, batch_idx), (FLOAT4)(sum, 0.0, 0.0, 0.0));
}

__kernel void reduct_general_mean_local(GLOBAL_SIZE_2_DIMS
                            __read_only image2d_t input,
                            __write_only image2d_t output,
                            __private const int batch,
                            __private const int height,
                            __private const int width
                            ) {
    const int batch_idx = get_global_id(1);
    const int width_idx = get_global_id(2);
    
    const int idx = get_local_id(0);
    FLOAT local sum[256];
    sum[idx] = 0.0;
    const int reduce_num = get_local_size(0);

    for (int h = idx; h < height; h+=reduce_num) {
        FLOAT4 in = RI_F(input, SAMPLER, (int2)(width_idx, batch_idx*height+h));
        sum[idx] = sum[idx] + in.x;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = reduce_num/2; i > 0; i /= 2){
        if (idx < i)
            sum[idx] = sum[idx] + sum[idx + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (idx == 0) {
        WI_F(output, (int2)(width_idx, batch_idx), (FLOAT4)(sum[0]/height, 0.0, 0.0, 0.0));
    }
}
__kernel void reduct_general_sum_local(GLOBAL_SIZE_2_DIMS
                            __read_only image2d_t input,
                            __write_only image2d_t output,
                            __private const int batch,
                            __private const int height,
                            __private const int width
                            ) {
    const int batch_idx = get_global_id(1);
    const int width_idx = get_global_id(2);

    const int idx = get_local_id(0);
    FLOAT local sum[256];
    sum[idx] = 0.0;

    const int reduce_num = get_local_size(0);

    for (int h = idx; h < height; h+=reduce_num) {
        FLOAT4 in = RI_F(input, SAMPLER, (int2)(width_idx, batch_idx*height+h));
        sum[idx] = sum[idx] + in.x;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = reduce_num/2; i > 0; i /= 2){
        if (idx < i)
            sum[idx] = sum[idx] + sum[idx + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (idx == 0) {
        WI_F(output, (int2)(width_idx, batch_idx), (FLOAT4)(sum[0], 0.0, 0.0, 0.0));
    }
}

__kernel void reduct_general_max_local(GLOBAL_SIZE_2_DIMS
                            __read_only image2d_t input,
                            __write_only image2d_t output,
                            __private const int batch,
                            __private const int height,
                            __private const int width
                            ) {
    const int batch_idx = get_global_id(1);
    const int width_idx = get_global_id(2);

    const int idx = get_local_id(0);
    FLOAT local sum[256];
    sum[idx] = -MAXFLOAT;
    const int reduce_num = get_local_size(0);

    for (int h = idx; h < height; h+=reduce_num) {
        FLOAT4 in = RI_F(input, SAMPLER, (int2)(width_idx, batch_idx*height+h));
        sum[idx] = max(sum[idx], in.x);
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = reduce_num/2; i > 0; i /= 2){
        if (idx < i)
            sum[idx] = max(sum[idx], sum[idx + i]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (idx == 0) {
        WI_F(output, (int2)(width_idx, batch_idx), (FLOAT4)(sum[0], 0.0, 0.0, 0.0));
    }
    
}

__kernel void reduct_general_min_local(GLOBAL_SIZE_2_DIMS
                            __read_only image2d_t input,
                            __write_only image2d_t output,
                            __private const int batch,
                            __private const int height,
                            __private const int width
                            ) {
    const int batch_idx = get_global_id(1);
    const int width_idx = get_global_id(2);
    
    const int idx = get_local_id(0);
    FLOAT local sum[256];
    sum[idx] = MAXFLOAT;

    const int reduce_num = get_local_size(0);

    for (int h = idx; h < height; h+=reduce_num) {
        FLOAT4 in = RI_F(input, SAMPLER, (int2)(width_idx, batch_idx*height+h));
        sum[idx] = min(sum[idx], in.x);
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = reduce_num/2; i > 0; i /= 2){
        if (idx < i)
            sum[idx] = min(sum[idx], sum[idx + i]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (idx == 0) {
        WI_F(output, (int2)(width_idx, batch_idx), (FLOAT4)(sum[0], 0.0, 0.0, 0.0));
    }
}

__kernel void reduct_general_mul_local(GLOBAL_SIZE_2_DIMS
                            __read_only image2d_t input,
                            __write_only image2d_t output,
                            __private const int batch,
                            __private const int height,
                            __private const int width
                            ) {
    const int batch_idx = get_global_id(1);
    const int width_idx = get_global_id(2);

    const int idx = get_local_id(0);
    FLOAT local sum[256];
    sum[idx] = 1.0;

    const int reduce_num = get_local_size(0);

    for (int h = idx; h < height; h+=reduce_num) {
        FLOAT4 in = RI_F(input, SAMPLER, (int2)(width_idx, batch_idx*height+h));
        sum[idx] = sum[idx] * in.x;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = reduce_num/2; i > 0; i /= 2){
        if (idx < i)
            sum[idx] = sum[idx] * sum[idx + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (idx == 0) {
        WI_F(output, (int2)(width_idx, batch_idx), (FLOAT4)(sum[0], 0.0, 0.0, 0.0));
    }
}

