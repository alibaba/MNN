#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#define READ_INPUT_IMAGE(i, base)                                                                         \
    int in_width_value##i = in_width##i + base;                                                           \
    in_width_value##i =                                                                                   \
        select(in_idx + in_width_value##i, -1, (in_width_value##i < 0 || in_width_value##i >= input_shape.y)); \
    in##i = RI_F(input, SAMPLER, (int2)(in_width_value##i, in_hb_value));

#define CALCULATE_OUTPUT(i)                  \
    out##i = mad(in##i.x, weights0, out##i); \
    out##i = mad(in##i.y, weights1, out##i); \
    out##i = mad(in##i.z, weights2, out##i); \
    out##i = mad(in##i.w, weights3, out##i);    

#define CALCULATE_OUTPUT_WEIGHTS4(i, j)                  \
    out##i = mad(in##j.x, weights4, out##i); \
    out##i = mad(in##j.y, weights5, out##i); \
    out##i = mad(in##j.z, weights6, out##i); \
    out##i = mad(in##j.w, weights7, out##i);

#define CALCULATE_OUTPUT_OPT(i)                  \
    out##i = mad(in_sm##i[local_idx].x, weights0, out##i); \
    out##i = mad(in_sm##i[local_idx].y, weights1, out##i); \
    out##i = mad(in_sm##i[local_idx].z, weights2, out##i); \
    out##i = mad(in_sm##i[local_idx].w, weights3, out##i);   

#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                     \
    }

#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

#define UNIT 4
#define MOD_NUM 15

#ifdef INPUT_CHANNEL_LEAVE
    #define PADZEROSVEC(k, channel, data0, data1, data2, data3) \
        data0 = (k << 2) < channel ? data0 : 0; \
        data1 = (k << 2) + 1 < channel ? data1 : 0; \
        data2 = (k << 2) + 2 < channel ? data2 : 0; \
        data3 = (k << 2) + 3 < channel ? data3 : 0;
#else
    #define PADZEROSVEC(k, channel, data0, data1, data2, data3)
#endif

__kernel
#if SET_ATTRIBUTE
__attribute__((work_group_size_hint(16, 16, 1)))
#endif
void conv_2d_1x1_mali(GLOBAL_SIZE_2_DIMS __private const int out_w_blocks, __read_only image2d_t input,
                          #ifdef BUFFER_INP_FP32
                          __global const float *kernel_ptr,
                          __global const float *bias_ptr,
                          #else
                          __global const FLOAT *kernel_ptr,
                          __global const FLOAT *bias_ptr,
                          #endif
                          __write_only image2d_t output,
                          __private const int in_c_block, __private const int out_h,
                          __private const int out_w) {

    const int out_c_w_idx = get_global_id(0); //c/4 w
    const int out_b_h_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(out_c_w_idx, out_b_h_idx);

    const int out_c_idx = out_c_w_idx / out_w_blocks;
    const int out_w_idx = out_c_w_idx % out_w_blocks;

    const int out_w4_idx = mul24(out_w_idx, 4);

    #ifdef BUFFER_INP_FP32
    FLOAT4 out0 = CONVERT_FLOAT4(vload4(out_c_idx, (__global float *)bias_ptr));
    #else
    FLOAT4 out0 = vload4(out_c_idx, (__global FLOAT *)bias_ptr);
    #endif
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    FLOAT4 weights0;
    FLOAT4 weights1;
    FLOAT4 weights2;
    FLOAT4 weights3;

    FLOAT4 in0; 
    FLOAT4 in1; 
    FLOAT4 in2;
    FLOAT4 in3; 

    FLOAT16 weight16;

    const int intput_width_idx0 = out_w4_idx;
    const int intput_width_idx1 = out_w4_idx + 1;
    const int intput_width_idx2 = out_w4_idx + 2;
    const int intput_width_idx3 = out_w4_idx + 3;

    for (int in_channel_block_idx = 0; in_channel_block_idx < in_c_block; ++in_channel_block_idx) {
        int input_width_base  = mul24(in_channel_block_idx, out_w);

        int offset = mad24(out_c_idx, in_c_block, in_channel_block_idx)*4;
        in0 = RI_F(input, SAMPLER, (int2)(input_width_base + intput_width_idx0, out_b_h_idx));
        in1 = RI_F(input, SAMPLER, (int2)(input_width_base + intput_width_idx1, out_b_h_idx));
        in2 = RI_F(input, SAMPLER, (int2)(input_width_base + intput_width_idx2, out_b_h_idx));
        in3 = RI_F(input, SAMPLER, (int2)(input_width_base + intput_width_idx3, out_b_h_idx));

        #ifdef BUFFER_INP_FP32
        weights0 = CONVERT_FLOAT4(vload4(offset, (__global float *)kernel_ptr));
        weights1 = CONVERT_FLOAT4(vload4(offset + 1, (__global float *)kernel_ptr));
        weights2 = CONVERT_FLOAT4(vload4(offset + 2, (__global float *)kernel_ptr));
        weights3 = CONVERT_FLOAT4(vload4(offset + 3, (__global float *)kernel_ptr));
        #else
        weights0 = vload4(offset, (__global FLOAT *)kernel_ptr);
        weights1 = vload4(offset + 1, (__global FLOAT *)kernel_ptr);
        weights2 = vload4(offset + 2, (__global FLOAT *)kernel_ptr);
        weights3 = vload4(offset + 3, (__global FLOAT *)kernel_ptr);
        #endif
        
        out0.x += dot(weights0, in0);
        out0.y += dot(weights1, in0);
        out0.z += dot(weights2, in0);
        out0.w += dot(weights3, in0);

        out1.x += dot(weights0, in1);
        out1.y += dot(weights1, in1);
        out1.z += dot(weights2, in1);
        out1.w += dot(weights3, in1);

        out2.x += dot(weights0, in2);
        out2.y += dot(weights1, in2);
        out2.z += dot(weights2, in2);
        out2.w += dot(weights3, in2);

        out3.x += dot(weights0, in3);
        out3.y += dot(weights1, in3);
        out3.z += dot(weights2, in3);
        out3.w += dot(weights3, in3);

    }

#ifdef RELU
    out0 = fmax(out0, (FLOAT4)0);
    out1 = fmax(out1, (FLOAT4)0);
    out2 = fmax(out2, (FLOAT4)0);
    out3 = fmax(out3, (FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (FLOAT4)0, (FLOAT4)6);
    out1 = clamp(out1, (FLOAT4)0, (FLOAT4)6);
    out2 = clamp(out2, (FLOAT4)0, (FLOAT4)6);
    out3 = clamp(out3, (FLOAT4)0, (FLOAT4)6);
#endif

    const int out_x_base = out_c_idx*out_w;

    const int remain = out_w - out_w4_idx;
    int output_idx   = out_x_base + out_w4_idx;
    
    if (remain >= 4) {
        WI_F(output, (int2)(output_idx, out_b_h_idx), out0);
        WI_F(output, (int2)(output_idx + 1, out_b_h_idx), out1);
        WI_F(output, (int2)(output_idx + 2, out_b_h_idx), out2);
        WI_F(output, (int2)(output_idx + 3, out_b_h_idx), out3);
    } else if (remain == 3) {
        WI_F(output, (int2)(output_idx, out_b_h_idx), out0);
        WI_F(output, (int2)(output_idx + 1, out_b_h_idx), out1);
        WI_F(output, (int2)(output_idx + 2, out_b_h_idx), out2);
    } else if (remain == 2) {
        WI_F(output, (int2)(output_idx, out_b_h_idx), out0);
        WI_F(output, (int2)(output_idx + 1, out_b_h_idx), out1);
    } else if (remain == 1) {
        WI_F(output, (int2)(output_idx, out_b_h_idx), out0);
    }

}

__kernel
#if SET_ATTRIBUTE
__attribute__((work_group_size_hint(16, 16, 1)))
#endif
void conv_2d_1x1(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                          __global const char *kernel_ptr,
                          __global const float *dequantScaleOffset,
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
                          __global const uchar *kernel_ptr,
                          __global const float *dequantScaleOffset,
#elif (defined USE_BUFFER)
                          __global const FLOAT *weights,
#else
                          __read_only image2d_t weights,
#endif
                          __read_only image2d_t bias,
                          __write_only image2d_t output,
                          __private const int2 input_shape,
                          __private const int in_channel_block, __private const int2 output_shape,
                          __private const int2 stride_shape,
                          __private const int output_width_4,
                          __private const int out_channel_blocks
#if (defined USE_LOW_BIT_WEIGHT_INT8) || (defined USE_LOW_BIT_WEIGHT_INT4)
                         ,__private const int blockDim
                         ,__private const int inChannel
#endif
) {

    const int output_channel_width_idx = get_global_id(0);
    const int output_batch_height_idx  = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(output_channel_width_idx, output_batch_height_idx);

    const int output_channel_block_idx = output_channel_width_idx / output_width_4;
    const int output_width_block_idx   = output_channel_width_idx % output_width_4;

#if (defined USE_LOW_BIT_WEIGHT_INT4)
    int weight_ic_offset = output_channel_block_idx * 8;
    int weight_oc_offset = out_channel_blocks * 8;
#else
    int weight_ic_offset = output_channel_block_idx * 16;
    int weight_oc_offset = out_channel_blocks * 16;
#endif

    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(output_channel_block_idx, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

#ifdef MNN_CONV_S1D1
    int intput_width_idx0 = output_width_block_idx << 2;
    int intput_width_idx1 = intput_width_idx0 + 1;
    int intput_width_idx2 = intput_width_idx0 + 2;
    int intput_width_idx3 = intput_width_idx0 + 3;
#else
    int intput_width_idx0 = mul24(output_width_block_idx, stride_shape.y*4);
    int intput_width_idx1 = intput_width_idx0 + stride_shape.y;
    int intput_width_idx2 = intput_width_idx1 + stride_shape.y;
    int intput_width_idx3 = intput_width_idx2 + stride_shape.y;

    intput_width_idx0 = select(intput_width_idx0, INT_MIN, intput_width_idx0 >= input_shape.y);
    intput_width_idx1 = select(intput_width_idx1, INT_MIN, intput_width_idx1 >= input_shape.y);
    intput_width_idx2 = select(intput_width_idx2, INT_MIN, intput_width_idx2 >= input_shape.y);
    intput_width_idx3 = select(intput_width_idx3, INT_MIN, intput_width_idx3 >= input_shape.y);
#endif

    int batch_index            = output_batch_height_idx / output_shape.x;
    int input_height_block_idx = mul24((output_batch_height_idx % output_shape.x), stride_shape.x) + batch_index * input_shape.x;

    FLOAT4 in0;
    FLOAT4 in1;
    FLOAT4 in2;
    FLOAT4 in3;
    FLOAT4 weights0;
    FLOAT4 weights1;
    FLOAT4 weights2;
    FLOAT4 weights3;
    int weight_offset = output_channel_block_idx * in_channel_block * 4 * 4;

    for (int in_channel_block_idx = 0; in_channel_block_idx < in_channel_block; ++in_channel_block_idx) {
#if (defined USE_LOW_BIT_WEIGHT_INT8) || (defined USE_LOW_BIT_WEIGHT_INT4)
        int kindex = (in_channel_block_idx * 4) / blockDim * out_channel_blocks * 8;
        COMPUTE_FLOAT8 ScaleOffset0 = CONVERT_COMPUTE_FLOAT8(vload8(output_channel_block_idx, dequantScaleOffset + kindex));
        COMPUTE_FLOAT4 scale0 = (COMPUTE_FLOAT4)(ScaleOffset0.s0, ScaleOffset0.s2, ScaleOffset0.s4, ScaleOffset0.s6);
        COMPUTE_FLOAT4 offset0 = (COMPUTE_FLOAT4)(ScaleOffset0.s1, ScaleOffset0.s3, ScaleOffset0.s5, ScaleOffset0.s7);
#endif
        int input_width_base  = in_channel_block_idx * input_shape.y;
        int weights_width_base = in_channel_block_idx << 2;
        
#if (defined USE_LOW_BIT_WEIGHT_INT8)
        FLOAT16 weights = CONVERT_FLOAT16(vload16(0, kernel_ptr + weight_ic_offset + in_channel_block_idx * weight_oc_offset));
        FLOAT4 weights0 = CONVERT_FLOAT4(weights.s0123) * scale0 + offset0;
        FLOAT4 weights1 = CONVERT_FLOAT4(weights.s4567) * scale0 + offset0;
        FLOAT4 weights2 = CONVERT_FLOAT4(weights.s89ab) * scale0 + offset0;
        FLOAT4 weights3 = CONVERT_FLOAT4(weights.scdef) * scale0 + offset0;
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
        uchar8 charWeightsInt4 = vload8(0, kernel_ptr + weight_ic_offset + in_channel_block_idx * weight_oc_offset);
        char4 charWeights0 = (char4)(0, 0, 0, 0);
        char4 charWeights1 = (char4)(0, 0, 0, 0);
        char4 charWeights2 = (char4)(0, 0, 0, 0);
        char4 charWeights3 = (char4)(0, 0, 0, 0);
        charWeights0.x = (charWeightsInt4.s0 >> 4) - 8;
        charWeights0.y = (charWeightsInt4.s0 & MOD_NUM) - 8;
        charWeights0.z = (charWeightsInt4.s1 >> 4) - 8;
        charWeights0.w = (charWeightsInt4.s1 & MOD_NUM) - 8;
        charWeights1.x = (charWeightsInt4.s2 >> 4) - 8;
        charWeights1.y = (charWeightsInt4.s2 & MOD_NUM) - 8;
        charWeights1.z = (charWeightsInt4.s3 >> 4) - 8;
        charWeights1.w = (charWeightsInt4.s3 & MOD_NUM)- 8;
        charWeights2.x = (charWeightsInt4.s4 >> 4) - 8;
        charWeights2.y = (charWeightsInt4.s4 & MOD_NUM) - 8;
        charWeights2.z = (charWeightsInt4.s5 >> 4) - 8;
        charWeights2.w = (charWeightsInt4.s5 & MOD_NUM) - 8;
        charWeights3.x = (charWeightsInt4.s6 >> 4) - 8;
        charWeights3.y = (charWeightsInt4.s6 & MOD_NUM) - 8;
        charWeights3.z = (charWeightsInt4.s7 >> 4) - 8;
        charWeights3.w = (charWeightsInt4.s7 & MOD_NUM) - 8;
        weights0 = mad(CONVERT_FLOAT4(charWeights0), scale0, offset0);
        weights1 = mad(CONVERT_FLOAT4(charWeights1), scale0, offset0);
        weights2 = mad(CONVERT_FLOAT4(charWeights2), scale0, offset0);
        weights3 = mad(CONVERT_FLOAT4(charWeights3), scale0, offset0);
#elif (defined USE_BUFFER)
        weights0 = vload4(weights_width_base, weights + weight_offset);
        weights1 = vload4(weights_width_base + 1, weights + weight_offset);
        weights2 = vload4(weights_width_base + 2, weights + weight_offset);
        weights3 = vload4(weights_width_base + 3, weights + weight_offset);
#else
        weights0 = RI_F(weights, SAMPLER, (int2)(weights_width_base + 0, output_channel_block_idx));
        weights1 = RI_F(weights, SAMPLER, (int2)(weights_width_base + 1, output_channel_block_idx));
        weights2 = RI_F(weights, SAMPLER, (int2)(weights_width_base + 2, output_channel_block_idx));
        weights3 = RI_F(weights, SAMPLER, (int2)(weights_width_base + 3, output_channel_block_idx));
#endif
        PADZEROSVEC(in_channel_block_idx, inChannel, weights0, weights1, weights2, weights3);
        in0 = RI_F(input, SAMPLER, (int2)(input_width_base + intput_width_idx0, input_height_block_idx));
        in1 = RI_F(input, SAMPLER, (int2)(input_width_base + intput_width_idx1, input_height_block_idx));
        in2 = RI_F(input, SAMPLER, (int2)(input_width_base + intput_width_idx2, input_height_block_idx));
        in3 = RI_F(input, SAMPLER, (int2)(input_width_base + intput_width_idx3, input_height_block_idx));

        CALCULATE_OUTPUT(0);
        CALCULATE_OUTPUT(1);
        CALCULATE_OUTPUT(2);
        CALCULATE_OUTPUT(3);
    }

#ifdef RELU
    out0 = fmax(out0, (FLOAT4)0);
    out1 = fmax(out1, (FLOAT4)0);
    out2 = fmax(out2, (FLOAT4)0);
    out3 = fmax(out3, (FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (FLOAT4)0, (FLOAT4)6);
    out1 = clamp(out1, (FLOAT4)0, (FLOAT4)6);
    out2 = clamp(out2, (FLOAT4)0, (FLOAT4)6);
    out3 = clamp(out3, (FLOAT4)0, (FLOAT4)6);
#endif

    const int out_x_base = mul24(output_channel_block_idx, output_shape.y);
    int out_x_idx        = output_width_block_idx << 2;

    const int remain = output_shape.y - out_x_idx;
    int output_idx   = out_x_base + out_x_idx;
    if (remain >= 4) {
        WI_F(output, (int2)(output_idx, output_batch_height_idx), out0);
        WI_F(output, (int2)(output_idx + 1, output_batch_height_idx), out1);
        WI_F(output, (int2)(output_idx + 2, output_batch_height_idx), out2);
        WI_F(output, (int2)(output_idx + 3, output_batch_height_idx), out3);
    } else if (remain == 3) {
        WI_F(output, (int2)(output_idx, output_batch_height_idx), out0);
        WI_F(output, (int2)(output_idx + 1, output_batch_height_idx), out1);
        WI_F(output, (int2)(output_idx + 2, output_batch_height_idx), out2);
    } else if (remain == 2) {
        WI_F(output, (int2)(output_idx, output_batch_height_idx), out0);
        WI_F(output, (int2)(output_idx + 1, output_batch_height_idx), out1);
    } else if (remain == 1) {
        WI_F(output, (int2)(output_idx, output_batch_height_idx), out0);
    }
}

__kernel
#if SET_ATTRIBUTE
__attribute__((work_group_size_hint(16, 16, 1)))
#endif
void conv_2d_1x1_c8h1w4(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                          __global const char *kernel_ptr,
                          __global const float *dequantScaleOffset,
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
                          __global const uchar *kernel_ptr,
                          __global const float *dequantScaleOffset,
#elif (defined USE_BUFFER)
                          __global const FLOAT *weights,
#else
                          __read_only image2d_t weights,
#endif
                          __read_only image2d_t bias,
                          __write_only image2d_t output,
                          __private const int2 input_shape,
                          __private const int in_channel_block, __private const int2 output_shape,
                          __private const int2 stride_shape,
                          __private const int output_width_4,
                          __private const int out_channel_blocks
#if (defined USE_LOW_BIT_WEIGHT_INT8) || (defined USE_LOW_BIT_WEIGHT_INT4)
                         ,__private const int blockDim
                         ,__private const int inChannel
#endif
) {

    const int output_channel_width_idx = get_global_id(0);
    const int output_batch_height_idx  = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(output_channel_width_idx, output_batch_height_idx);

    const int output_channel_block_idx = output_channel_width_idx / output_width_4;
    const int output_width_block_idx   = output_channel_width_idx % output_width_4;
    const int output_channel_idx = output_channel_block_idx << 1;

#if (defined USE_LOW_BIT_WEIGHT_INT4)
    int weight_ic_offset = output_channel_block_idx * 16;
    int weight_oc_offset = out_channel_blocks * 8;
#else
    int weight_ic_offset = output_channel_block_idx * 32;
    int weight_oc_offset = out_channel_blocks * 16;
#endif
    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(output_channel_idx, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;
    
    FLOAT4 out4 = RI_F(bias, SAMPLER, (int2)(output_channel_idx + 1, 0));
    FLOAT4 out5 = out4;
    FLOAT4 out6 = out4;
    FLOAT4 out7 = out4;

#ifdef MNN_CONV_S1D1
    int intput_width_idx0 = output_width_block_idx << 2;
    int intput_width_idx1 = intput_width_idx0 + 1;
    int intput_width_idx2 = intput_width_idx0 + 2;
    int intput_width_idx3 = intput_width_idx0 + 3;
#else
    int intput_width_idx0 = mul24(output_width_block_idx, stride_shape.y*4);
    int intput_width_idx1 = intput_width_idx0 + stride_shape.y;
    int intput_width_idx2 = intput_width_idx1 + stride_shape.y;
    int intput_width_idx3 = intput_width_idx2 + stride_shape.y;

    intput_width_idx0 = select(intput_width_idx0, INT_MIN, intput_width_idx0 >= input_shape.y);
    intput_width_idx1 = select(intput_width_idx1, INT_MIN, intput_width_idx1 >= input_shape.y);
    intput_width_idx2 = select(intput_width_idx2, INT_MIN, intput_width_idx2 >= input_shape.y);
    intput_width_idx3 = select(intput_width_idx3, INT_MIN, intput_width_idx3 >= input_shape.y);
#endif

    int batch_index            = output_batch_height_idx / output_shape.x;
    int input_height_block_idx = mul24((output_batch_height_idx % output_shape.x), stride_shape.x) + batch_index * input_shape.x;

    FLOAT4 in0;
    FLOAT4 in1;
    FLOAT4 in2;
    FLOAT4 in3;
    FLOAT4 weights0;
    FLOAT4 weights1;
    FLOAT4 weights2;
    FLOAT4 weights3;
    FLOAT4 weights4;
    FLOAT4 weights5;
    FLOAT4 weights6;
    FLOAT4 weights7;
    int weight_offset = output_channel_idx * in_channel_block * 4 * 4;
    int weight_offset1 = weight_offset + in_channel_block * 4 * 4;

    for (int in_channel_block_idx = 0; in_channel_block_idx < in_channel_block; ++in_channel_block_idx) {
#if (defined USE_LOW_BIT_WEIGHT_INT8) || (defined USE_LOW_BIT_WEIGHT_INT4)
        int kindex = (in_channel_block_idx * 4) / blockDim * out_channel_blocks * 8;
        COMPUTE_FLOAT8 ScaleOffset0 = CONVERT_COMPUTE_FLOAT8(vload8(output_channel_idx, dequantScaleOffset + kindex));
        COMPUTE_FLOAT4 scale0 = (COMPUTE_FLOAT4)(ScaleOffset0.s0, ScaleOffset0.s2, ScaleOffset0.s4, ScaleOffset0.s6);
        COMPUTE_FLOAT4 offset0 = (COMPUTE_FLOAT4)(ScaleOffset0.s1, ScaleOffset0.s3, ScaleOffset0.s5, ScaleOffset0.s7);
        COMPUTE_FLOAT8 ScaleOffset1 = CONVERT_COMPUTE_FLOAT8(vload8(output_channel_idx + 1, dequantScaleOffset + kindex));
        COMPUTE_FLOAT4 scale1 = (COMPUTE_FLOAT4)(ScaleOffset1.s0, ScaleOffset1.s2, ScaleOffset1.s4, ScaleOffset1.s6);
        COMPUTE_FLOAT4 offset1 = (COMPUTE_FLOAT4)(ScaleOffset1.s1, ScaleOffset1.s3, ScaleOffset1.s5, ScaleOffset1.s7);
#endif
        
        int input_width_base  = in_channel_block_idx * input_shape.y;
        int weights_width_base = in_channel_block_idx << 2;
        in0 = RI_F(input, SAMPLER, (int2)(input_width_base + intput_width_idx0, input_height_block_idx));
        in1 = RI_F(input, SAMPLER, (int2)(input_width_base + intput_width_idx1, input_height_block_idx));
        in2 = RI_F(input, SAMPLER, (int2)(input_width_base + intput_width_idx2, input_height_block_idx));
        in3 = RI_F(input, SAMPLER, (int2)(input_width_base + intput_width_idx3, input_height_block_idx));

#if (defined USE_LOW_BIT_WEIGHT_INT8)
        FLOAT16 weightsInt80 = CONVERT_FLOAT16(vload16(0, kernel_ptr + weight_ic_offset + in_channel_block_idx * weight_oc_offset));
        FLOAT16 weightsInt81 = CONVERT_FLOAT16(vload16(0, kernel_ptr + 16 + weight_ic_offset + in_channel_block_idx * weight_oc_offset));
        FLOAT4 weights0 = CONVERT_FLOAT4(weightsInt80.s0123) * scale0 + offset0;
        FLOAT4 weights1 = CONVERT_FLOAT4(weightsInt80.s4567) * scale0 + offset0;
        FLOAT4 weights2 = CONVERT_FLOAT4(weightsInt80.s89ab) * scale0 + offset0;
        FLOAT4 weights3 = CONVERT_FLOAT4(weightsInt80.scdef) * scale0 + offset0;
        FLOAT4 weights4 = CONVERT_FLOAT4(weightsInt81.s0123) * scale1 + offset1;
        FLOAT4 weights5 = CONVERT_FLOAT4(weightsInt81.s4567) * scale1 + offset1;
        FLOAT4 weights6 = CONVERT_FLOAT4(weightsInt81.s89ab) * scale1 + offset1;
        FLOAT4 weights7 = CONVERT_FLOAT4(weightsInt81.scdef) * scale1 + offset1;
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
        uchar16 charWeightsInt4 = vload16(0, kernel_ptr + weight_ic_offset + in_channel_block_idx * weight_oc_offset);
        char4 charWeights0 = (char4)(0, 0, 0, 0);
        char4 charWeights1 = (char4)(0, 0, 0, 0);
        char4 charWeights2 = (char4)(0, 0, 0, 0);
        char4 charWeights3 = (char4)(0, 0, 0, 0);
        char4 charWeights4 = (char4)(0, 0, 0, 0);
        char4 charWeights5 = (char4)(0, 0, 0, 0);
        char4 charWeights6 = (char4)(0, 0, 0, 0);
        char4 charWeights7 = (char4)(0, 0, 0, 0);
        charWeights0.x = (charWeightsInt4.s0 >> 4) - 8;
        charWeights0.y = (charWeightsInt4.s0 & MOD_NUM) - 8;
        charWeights0.z = (charWeightsInt4.s1 >> 4) - 8;
        charWeights0.w = (charWeightsInt4.s1 & MOD_NUM) - 8;
        charWeights1.x = (charWeightsInt4.s2 >> 4) - 8;
        charWeights1.y = (charWeightsInt4.s2 & MOD_NUM) - 8;
        charWeights1.z = (charWeightsInt4.s3 >> 4) - 8;
        charWeights1.w = (charWeightsInt4.s3 & MOD_NUM) - 8;
        charWeights2.x = (charWeightsInt4.s4 >> 4) - 8;
        charWeights2.y = (charWeightsInt4.s4 & MOD_NUM) - 8;
        charWeights2.z = (charWeightsInt4.s5 >> 4) - 8;
        charWeights2.w = (charWeightsInt4.s5 & MOD_NUM) - 8;
        charWeights3.x = (charWeightsInt4.s6 >> 4) - 8;
        charWeights3.y = (charWeightsInt4.s6 & MOD_NUM) - 8;
        charWeights3.z = (charWeightsInt4.s7 >> 4) - 8;
        charWeights3.w = (charWeightsInt4.s7 & MOD_NUM) - 8;
        charWeights4.x = (charWeightsInt4.s8 >> 4) - 8;
        charWeights4.y = (charWeightsInt4.s8 & MOD_NUM) - 8;
        charWeights4.z = (charWeightsInt4.s9 >> 4) - 8;
        charWeights4.w = (charWeightsInt4.s9 & MOD_NUM) - 8;
        charWeights5.x = (charWeightsInt4.sa >> 4) - 8;
        charWeights5.y = (charWeightsInt4.sa & MOD_NUM) - 8;
        charWeights5.z = (charWeightsInt4.sb >> 4) - 8;
        charWeights5.w = (charWeightsInt4.sb & MOD_NUM) - 8;
        charWeights6.x = (charWeightsInt4.sc >> 4) - 8;
        charWeights6.y = (charWeightsInt4.sc & MOD_NUM) - 8;
        charWeights6.z = (charWeightsInt4.sd >> 4) - 8;
        charWeights6.w = (charWeightsInt4.sd & MOD_NUM) - 8;
        charWeights7.x = (charWeightsInt4.se >> 4) - 8;
        charWeights7.y = (charWeightsInt4.se & MOD_NUM) - 8;
        charWeights7.z = (charWeightsInt4.sf >> 4) - 8;
        charWeights7.w = (charWeightsInt4.sf & MOD_NUM) - 8;
        weights0 = mad(CONVERT_FLOAT4(charWeights0), scale0, offset0);
        weights1 = mad(CONVERT_FLOAT4(charWeights1), scale0, offset0);
        weights2 = mad(CONVERT_FLOAT4(charWeights2), scale0, offset0);
        weights3 = mad(CONVERT_FLOAT4(charWeights3), scale0, offset0);
        weights4 = mad(CONVERT_FLOAT4(charWeights4), scale1, offset1);
        weights5 = mad(CONVERT_FLOAT4(charWeights5), scale1, offset1);
        weights6 = mad(CONVERT_FLOAT4(charWeights6), scale1, offset1);
        weights7 = mad(CONVERT_FLOAT4(charWeights7), scale1, offset1);
#elif (defined USE_BUFFER)
        weights0 = vload4(weights_width_base, weights + weight_offset);
        weights1 = vload4(weights_width_base + 1, weights + weight_offset);
        weights2 = vload4(weights_width_base + 2, weights + weight_offset);
        weights3 = vload4(weights_width_base + 3, weights + weight_offset);

        weights4 = vload4(weights_width_base, weights + weight_offset1);
        weights5 = vload4(weights_width_base + 1, weights + weight_offset1);
        weights6 = vload4(weights_width_base + 2, weights + weight_offset1);
        weights7 = vload4(weights_width_base + 3, weights + weight_offset1);
#else
        weights0 = RI_F(weights, SAMPLER, (int2)(weights_width_base + 0, output_channel_idx));
        weights1 = RI_F(weights, SAMPLER, (int2)(weights_width_base + 1, output_channel_idx));
        weights2 = RI_F(weights, SAMPLER, (int2)(weights_width_base + 2, output_channel_idx));
        weights3 = RI_F(weights, SAMPLER, (int2)(weights_width_base + 3, output_channel_idx));
        
        weights4 = RI_F(weights, SAMPLER, (int2)(weights_width_base + 0, output_channel_idx + 1));
        weights5 = RI_F(weights, SAMPLER, (int2)(weights_width_base + 1, output_channel_idx + 1));
        weights6 = RI_F(weights, SAMPLER, (int2)(weights_width_base + 2, output_channel_idx + 1));
        weights7 = RI_F(weights, SAMPLER, (int2)(weights_width_base + 3, output_channel_idx + 1));
#endif
        PADZEROSVEC(in_channel_block_idx, inChannel, weights0, weights1, weights2, weights3);
        PADZEROSVEC(in_channel_block_idx, inChannel, weights4, weights5, weights6, weights7);

        CALCULATE_OUTPUT(0);
        CALCULATE_OUTPUT(1);
        CALCULATE_OUTPUT(2);
        CALCULATE_OUTPUT(3);
        
        CALCULATE_OUTPUT_WEIGHTS4(4, 0);
        CALCULATE_OUTPUT_WEIGHTS4(5, 1);
        CALCULATE_OUTPUT_WEIGHTS4(6, 2);
        CALCULATE_OUTPUT_WEIGHTS4(7, 3);
    }

#ifdef RELU
    out0 = fmax(out0, (FLOAT4)0);
    out1 = fmax(out1, (FLOAT4)0);
    out2 = fmax(out2, (FLOAT4)0);
    out3 = fmax(out3, (FLOAT4)0);
    out4 = fmax(out4, (FLOAT4)0);
    out5 = fmax(out5, (FLOAT4)0);
    out6 = fmax(out6, (FLOAT4)0);
    out7 = fmax(out7, (FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (FLOAT4)0, (FLOAT4)6);
    out1 = clamp(out1, (FLOAT4)0, (FLOAT4)6);
    out2 = clamp(out2, (FLOAT4)0, (FLOAT4)6);
    out3 = clamp(out3, (FLOAT4)0, (FLOAT4)6);
    out4 = clamp(out4, (FLOAT4)0, (FLOAT4)6);
    out5 = clamp(out5, (FLOAT4)0, (FLOAT4)6);
    out6 = clamp(out6, (FLOAT4)0, (FLOAT4)6);
    out7 = clamp(out7, (FLOAT4)0, (FLOAT4)6);
#endif

    const int out_x_base = mul24(output_channel_idx, output_shape.y);
    int out_x_idx        = output_width_block_idx << 2;

    const int remain = output_shape.y - out_x_idx;
    int output_idx   = out_x_base + out_x_idx;
    if (remain >= 4) {
        WI_F(output, (int2)(output_idx, output_batch_height_idx), out0);
        WI_F(output, (int2)(output_idx + 1, output_batch_height_idx), out1);
        WI_F(output, (int2)(output_idx + 2, output_batch_height_idx), out2);
        WI_F(output, (int2)(output_idx + 3, output_batch_height_idx), out3);
    } else if (remain == 3) {
        WI_F(output, (int2)(output_idx, output_batch_height_idx), out0);
        WI_F(output, (int2)(output_idx + 1, output_batch_height_idx), out1);
        WI_F(output, (int2)(output_idx + 2, output_batch_height_idx), out2);
    } else if (remain == 2) {
        WI_F(output, (int2)(output_idx, output_batch_height_idx), out0);
        WI_F(output, (int2)(output_idx + 1, output_batch_height_idx), out1);
    } else if (remain == 1) {
        WI_F(output, (int2)(output_idx, output_batch_height_idx), out0);
    }
    
    if(output_channel_idx + 1 >= out_channel_blocks)
        return;
    output_idx += output_shape.y;
    if (remain >= 4) {
        WI_F(output, (int2)(output_idx, output_batch_height_idx), out4);
        WI_F(output, (int2)(output_idx + 1, output_batch_height_idx), out5);
        WI_F(output, (int2)(output_idx + 2, output_batch_height_idx), out6);
        WI_F(output, (int2)(output_idx + 3, output_batch_height_idx), out7);
    } else if (remain == 3) {
        WI_F(output, (int2)(output_idx, output_batch_height_idx), out4);
        WI_F(output, (int2)(output_idx + 1, output_batch_height_idx), out5);
        WI_F(output, (int2)(output_idx + 2, output_batch_height_idx), out6);
    } else if (remain == 2) {
        WI_F(output, (int2)(output_idx, output_batch_height_idx), out4);
        WI_F(output, (int2)(output_idx + 1, output_batch_height_idx), out5);
    } else if (remain == 1) {
        WI_F(output, (int2)(output_idx, output_batch_height_idx), out4);
    }
}

__kernel
#if SET_ATTRIBUTE
__attribute__((work_group_size_hint(16, 16, 1)))
#endif
void conv_2d_c4h1w4(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                      __global const char *kernel_ptr,
                      __global const float *dequantScaleOffset,
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
                      __global const uchar *kernel_ptr,
                      __global const float *dequantScaleOffset,
#elif (defined USE_BUFFER)
                      __global const FLOAT *weights,
#else
                      __read_only image2d_t weights,
#endif
#ifdef BIAS
                      __read_only image2d_t bias,
#endif
                      __write_only image2d_t output,
                      __private const int2 input_shape,
                      __private const int in_channel_block_length,
                      __private const int2 output_shape,
                      __private const int2 weights_shape,
                      __private const int2 stride_shape,
                      __private const int2 padding_shape,
                      __private const int2 dilation_shape,
                      __private const int out_width_blocks,
                      __private const int out_channel_blocks,
                      __private const int out_height_blocks
#if (defined USE_LOW_BIT_WEIGHT_INT8) || (defined USE_LOW_BIT_WEIGHT_INT4)
                     ,__private const int blockDim
                     ,__private const int inChannel
#endif
) {

    const int output_channel_width_idx = get_global_id(0);
    const int output_batch_height_idx  = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(output_channel_width_idx, output_batch_height_idx);

    const int out_channel_block_idx = output_channel_width_idx / out_width_blocks;
    const int out_height_block_idx   = output_channel_width_idx % out_width_blocks;

#ifdef BIAS
    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(out_channel_block_idx, 0));
#else
    FLOAT4 out0 = (FLOAT4)0;
#endif
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    int in_width0          = mad24(out_height_block_idx, stride_shape.y<<2, -padding_shape.y);
    int in_width1          = in_width0 + stride_shape.y;
    int in_width2          = in_width0 + stride_shape.y * 2;
    int in_width3          = in_width0 + stride_shape.y * 3;
    
#ifdef MNN_CONV_S1D1
    const int height_start = mad24((output_batch_height_idx % output_shape.x), 1, -padding_shape.x);
    const int kh_start = select(0, (-height_start), height_start < 0);
    int in_height_start    = kh_start + height_start;
    int in_height_end      = min(weights_shape.x + height_start, input_shape.x);

    const int batch_idx          = mul24((output_batch_height_idx / output_shape.x), input_shape.x);
    const int weights_h_idx = mul24(out_channel_block_idx, mul24(weights_shape.y, weights_shape.x)) + mul24(select(0, (-height_start), height_start < 0), weights_shape.y);
#else
    const int height_start = mad24((output_batch_height_idx % output_shape.x), stride_shape.x, -padding_shape.x);
    const int kh_start = select(0, (-height_start + dilation_shape.x - 1) / dilation_shape.x, height_start < 0);
    int in_height_start    = mad24(kh_start, dilation_shape.x, height_start);
    int in_height_end      = min(mad24(weights_shape.x, dilation_shape.x, height_start), input_shape.x);

    const int batch_idx          = mul24((output_batch_height_idx / output_shape.x), input_shape.x);
    const int weights_h_idx = mul24(out_channel_block_idx, mul24(weights_shape.y, weights_shape.x)) + mul24(select(0, (-height_start + dilation_shape.x - 1) / dilation_shape.x, height_start < 0), weights_shape.y);
#endif

#if (defined USE_LOW_BIT_WEIGHT_INT8) || (defined USE_LOW_BIT_WEIGHT_INT4) || (defined USE_BUFFER)
    const int weight_oc_offset = out_channel_blocks * weights_shape.x * weights_shape.y * 4;
#endif

    FLOAT4 in0, in1, in2, in3;
    FLOAT4 weights0, weights1, weights2, weights3;
    for (int in_channel_block_idx = 0; in_channel_block_idx < in_channel_block_length; ++in_channel_block_idx) {
        
#if (defined USE_LOW_BIT_WEIGHT_INT8) || (defined USE_LOW_BIT_WEIGHT_INT4)
        int kindex = (in_channel_block_idx * 4) / blockDim * out_channel_blocks * 8;
        COMPUTE_FLOAT8 ScaleOffset0 = CONVERT_COMPUTE_FLOAT8(vload8(out_channel_block_idx, dequantScaleOffset + kindex));
        COMPUTE_FLOAT4 scale0 = (COMPUTE_FLOAT4)(ScaleOffset0.s0, ScaleOffset0.s2, ScaleOffset0.s4, ScaleOffset0.s6);
        COMPUTE_FLOAT4 offset0 = (COMPUTE_FLOAT4)(ScaleOffset0.s1, ScaleOffset0.s3, ScaleOffset0.s5, ScaleOffset0.s7);
#endif
        
        const int in_idx = mul24(in_channel_block_idx, input_shape.y);
#if (defined USE_LOW_BIT_WEIGHT_INT8) || (defined USE_LOW_BIT_WEIGHT_INT4) || (defined USE_BUFFER)
        int weight_offset = ((((4*in_channel_block_idx+0)* out_channel_blocks + out_channel_block_idx) *weights_shape.x + kh_start)*weights_shape.y + 0) * 4;
#else
        int weights_x_idx = in_channel_block_idx << 2;
        int weights_y_idx = weights_h_idx;
#endif
        for (int iy = in_height_start; iy < in_height_end; iy += dilation_shape.x) {
            int in_hb_value = iy + batch_idx;
#ifdef MNN_CONV_S1D1
            {
                READ_INPUT_IMAGE(0, 0);
                READ_INPUT_IMAGE(1, 0);
                READ_INPUT_IMAGE(2, 0);
                READ_INPUT_IMAGE(3, 0);
                
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                char4 charWeight0 = vload4(0, kernel_ptr+weight_offset);
                char4 charWeight1 = vload4(0, kernel_ptr+weight_offset+weight_oc_offset);
                char4 charWeight2 = vload4(0, kernel_ptr+weight_offset+weight_oc_offset*2);
                char4 charWeight3 = vload4(0, kernel_ptr+weight_offset+weight_oc_offset*3);
                weights0 = mad(CONVERT_FLOAT4(charWeight0), scale0, offset0);
                weights1 = mad(CONVERT_FLOAT4(charWeight1), scale0, offset0);
                weights2 = mad(CONVERT_FLOAT4(charWeight2), scale0, offset0);
                weights3 = mad(CONVERT_FLOAT4(charWeight3), scale0, offset0);
                weight_offset += 4;
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
                uchar2 charWeightInt40 = vload2(0, kernel_ptr+weight_offset/2);
                uchar2 charWeightInt41 = vload2(0, kernel_ptr+weight_offset/2+weight_oc_offset/2);
                uchar2 charWeightInt42 = vload2(0, kernel_ptr+weight_offset/2+weight_oc_offset*2/2);
                uchar2 charWeightInt43 = vload2(0, kernel_ptr+weight_offset/2+weight_oc_offset*3/2);
                char4 charWeight0 = (char4)(0, 0, 0, 0);
                char4 charWeight1 = (char4)(0, 0, 0, 0);
                char4 charWeight2 = (char4)(0, 0, 0, 0);
                char4 charWeight3 = (char4)(0, 0, 0, 0);
                charWeight0.x = (charWeightInt40.s0 >> 4) - 8;
                charWeight0.y = (charWeightInt40.s0 & MOD_NUM) - 8;
                charWeight0.z = (charWeightInt40.s1 >> 4) - 8;
                charWeight0.w = (charWeightInt40.s1 & MOD_NUM) - 8;
                charWeight1.x = (charWeightInt41.s0 >> 4) - 8;
                charWeight1.y = (charWeightInt41.s0 & MOD_NUM) - 8;
                charWeight1.z = (charWeightInt41.s1 >> 4) - 8;
                charWeight1.w = (charWeightInt41.s1 & MOD_NUM) - 8;
                charWeight2.x = (charWeightInt42.s0 >> 4) - 8;
                charWeight2.y = (charWeightInt42.s0 & MOD_NUM) - 8;
                charWeight2.z = (charWeightInt42.s1 >> 4) - 8;
                charWeight2.w = (charWeightInt42.s1 & MOD_NUM) - 8;
                charWeight3.x = (charWeightInt43.s0 >> 4) - 8;
                charWeight3.y = (charWeightInt43.s0 & MOD_NUM) - 8;
                charWeight3.z = (charWeightInt43.s1 >> 4) - 8;
                charWeight3.w = (charWeightInt43.s1 & MOD_NUM) - 8;
                weights0 = mad(CONVERT_FLOAT4(charWeight0), scale0, offset0);
                weights1 = mad(CONVERT_FLOAT4(charWeight1), scale0, offset0);
                weights2 = mad(CONVERT_FLOAT4(charWeight2), scale0, offset0);
                weights3 = mad(CONVERT_FLOAT4(charWeight3), scale0, offset0);
                weight_offset += 4;
#elif (defined USE_BUFFER)
                weights0 = vload4(0, weights+weight_offset);
                weights1 = vload4(0, weights+weight_offset+weight_oc_offset);
                weights2 = vload4(0, weights+weight_offset+weight_oc_offset*2);
                weights3 = vload4(0, weights+weight_offset+weight_oc_offset*3);
                weight_offset += 4;
#else
                weights0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 0, weights_y_idx));
                weights1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx));
                weights2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx));
                weights3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx++));
#endif
                PADZEROSVEC(in_channel_block_idx, inChannel, weights0, weights1, weights2, weights3);
                CALCULATE_OUTPUT(0);
                CALCULATE_OUTPUT(1);
                CALCULATE_OUTPUT(2);
                CALCULATE_OUTPUT(3);
            }
            for (int w = 1; w < weights_shape.y; w++){
                in0 = in1;
                in1 = in2;
                in2 = in3;
                READ_INPUT_IMAGE(3, w);
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                char4 charWeight0 = vload4(0, kernel_ptr+weight_offset);
                char4 charWeight1 = vload4(0, kernel_ptr+weight_offset+weight_oc_offset);
                char4 charWeight2 = vload4(0, kernel_ptr+weight_offset+weight_oc_offset*2);
                char4 charWeight3 = vload4(0, kernel_ptr+weight_offset+weight_oc_offset*3);
                weights0 = mad(CONVERT_FLOAT4(charWeight0), scale0, offset0);
                weights1 = mad(CONVERT_FLOAT4(charWeight1), scale0, offset0);
                weights2 = mad(CONVERT_FLOAT4(charWeight2), scale0, offset0);
                weights3 = mad(CONVERT_FLOAT4(charWeight3), scale0, offset0);
                weight_offset += 4;
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
                uchar2 charWeightInt40 = vload2(0, kernel_ptr+weight_offset/2);
                uchar2 charWeightInt41 = vload2(0, kernel_ptr+weight_offset/2+weight_oc_offset/2);
                uchar2 charWeightInt42 = vload2(0, kernel_ptr+weight_offset/2+weight_oc_offset*2/2);
                uchar2 charWeightInt43 = vload2(0, kernel_ptr+weight_offset/2+weight_oc_offset*3/2);
                char4 charWeight0 = (char4)(0, 0, 0, 0);
                char4 charWeight1 = (char4)(0, 0, 0, 0);
                char4 charWeight2 = (char4)(0, 0, 0, 0);
                char4 charWeight3 = (char4)(0, 0, 0, 0);
                charWeight0.x = (charWeightInt40.s0 >> 4) - 8;
                charWeight0.y = (charWeightInt40.s0 & MOD_NUM) - 8;
                charWeight0.z = (charWeightInt40.s1 >> 4) - 8;
                charWeight0.w = (charWeightInt40.s1 & MOD_NUM) - 8;
                charWeight1.x = (charWeightInt41.s0 >> 4) - 8;
                charWeight1.y = (charWeightInt41.s0 & MOD_NUM) - 8;
                charWeight1.z = (charWeightInt41.s1 >> 4) - 8;
                charWeight1.w = (charWeightInt41.s1 & MOD_NUM) - 8;
                charWeight2.x = (charWeightInt42.s0 >> 4) - 8;
                charWeight2.y = (charWeightInt42.s0 & MOD_NUM) - 8;
                charWeight2.z = (charWeightInt42.s1 >> 4) - 8;
                charWeight2.w = (charWeightInt42.s1 & MOD_NUM) - 8;
                charWeight3.x = (charWeightInt43.s0 >> 4) - 8;
                charWeight3.y = (charWeightInt43.s0 & MOD_NUM) - 8;
                charWeight3.z = (charWeightInt43.s1 >> 4) - 8;
                charWeight3.w = (charWeightInt43.s1 & MOD_NUM) - 8;
                weights0 = mad(CONVERT_FLOAT4(charWeight0), scale0, offset0);
                weights1 = mad(CONVERT_FLOAT4(charWeight1), scale0, offset0);
                weights2 = mad(CONVERT_FLOAT4(charWeight2), scale0, offset0);
                weights3 = mad(CONVERT_FLOAT4(charWeight3), scale0, offset0);
                weight_offset += 4;
#elif (defined USE_BUFFER)
                weights0 = vload4(0, weights+weight_offset);
                weights1 = vload4(0, weights+weight_offset+weight_oc_offset);
                weights2 = vload4(0, weights+weight_offset+weight_oc_offset*2);
                weights3 = vload4(0, weights+weight_offset+weight_oc_offset*3);
                weight_offset += 4;
#else
                weights0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 0, weights_y_idx));
                weights1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx));
                weights2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx));
                weights3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx++));
#endif
                PADZEROSVEC(in_channel_block_idx, inChannel, weights0, weights1, weights2, weights3);
                CALCULATE_OUTPUT(0);
                CALCULATE_OUTPUT(1);
                CALCULATE_OUTPUT(2);
                CALCULATE_OUTPUT(3);
            }
#else
            for (int w = 0; w < weights_shape.y; w++) {
                int input_width_base = mul24(w, dilation_shape.y);
                READ_INPUT_IMAGE(0, input_width_base);
                READ_INPUT_IMAGE(1, input_width_base);
                READ_INPUT_IMAGE(2, input_width_base);
                READ_INPUT_IMAGE(3, input_width_base);
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                char4 charWeight0 = vload4(0, kernel_ptr+weight_offset);
                char4 charWeight1 = vload4(0, kernel_ptr+weight_offset+weight_oc_offset);
                char4 charWeight2 = vload4(0, kernel_ptr+weight_offset+weight_oc_offset*2);
                char4 charWeight3 = vload4(0, kernel_ptr+weight_offset+weight_oc_offset*3);
                weights0 = mad(CONVERT_FLOAT4(charWeight0), scale0, offset0);
                weights1 = mad(CONVERT_FLOAT4(charWeight1), scale0, offset0);
                weights2 = mad(CONVERT_FLOAT4(charWeight2), scale0, offset0);
                weights3 = mad(CONVERT_FLOAT4(charWeight3), scale0, offset0);
                weight_offset += 4;
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
                uchar2 charWeightInt40 = vload2(0, kernel_ptr+weight_offset/2);
                uchar2 charWeightInt41 = vload2(0, kernel_ptr+weight_offset/2+weight_oc_offset/2);
                uchar2 charWeightInt42 = vload2(0, kernel_ptr+weight_offset/2+weight_oc_offset*2/2);
                uchar2 charWeightInt43 = vload2(0, kernel_ptr+weight_offset/2+weight_oc_offset*3/2);
                char4 charWeight0 = (char4)(0, 0, 0, 0);
                char4 charWeight1 = (char4)(0, 0, 0, 0);
                char4 charWeight2 = (char4)(0, 0, 0, 0);
                char4 charWeight3 = (char4)(0, 0, 0, 0);
                charWeight0.x = (charWeightInt40.s0 >> 4) - 8;
                charWeight0.y = (charWeightInt40.s0 & MOD_NUM) - 8;
                charWeight0.z = (charWeightInt40.s1 >> 4) - 8;
                charWeight0.w = (charWeightInt40.s1 & MOD_NUM) - 8;
                charWeight1.x = (charWeightInt41.s0 >> 4) - 8;
                charWeight1.y = (charWeightInt41.s0 & MOD_NUM) - 8;
                charWeight1.z = (charWeightInt41.s1 >> 4) - 8;
                charWeight1.w = (charWeightInt41.s1 & MOD_NUM) - 8;
                charWeight2.x = (charWeightInt42.s0 >> 4) - 8;
                charWeight2.y = (charWeightInt42.s0 & MOD_NUM) - 8;
                charWeight2.z = (charWeightInt42.s1 >> 4) - 8;
                charWeight2.w = (charWeightInt42.s1 & MOD_NUM) - 8;
                charWeight3.x = (charWeightInt43.s0 >> 4) - 8;
                charWeight3.y = (charWeightInt43.s0 & MOD_NUM) - 8;
                charWeight3.z = (charWeightInt43.s1 >> 4) - 8;
                charWeight3.w = (charWeightInt43.s1 & MOD_NUM) - 8;
                weights0 = mad(CONVERT_FLOAT4(charWeight0), scale0, offset0);
                weights1 = mad(CONVERT_FLOAT4(charWeight1), scale0, offset0);
                weights2 = mad(CONVERT_FLOAT4(charWeight2), scale0, offset0);
                weights3 = mad(CONVERT_FLOAT4(charWeight3), scale0, offset0);
                weight_offset += 4;
#elif (defined USE_BUFFER)
                weights0 = vload4(0, weights+weight_offset);
                weights1 = vload4(0, weights+weight_offset+weight_oc_offset);
                weights2 = vload4(0, weights+weight_offset+weight_oc_offset*2);
                weights3 = vload4(0, weights+weight_offset+weight_oc_offset*3);
                weight_offset += 4;
#else
                weights0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 0, weights_y_idx)); 
                weights1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx)); 
                weights2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx)); 
                weights3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx++));
#endif
                PADZEROSVEC(in_channel_block_idx, inChannel, weights0, weights1, weights2, weights3);
                CALCULATE_OUTPUT(0);
                CALCULATE_OUTPUT(1);
                CALCULATE_OUTPUT(2);
                CALCULATE_OUTPUT(3);
            }
#endif
        }
    }

#ifdef RELU
    out0 = fmax(out0, (FLOAT4)0);
    out1 = fmax(out1, (FLOAT4)0);
    out2 = fmax(out2, (FLOAT4)0);
    out3 = fmax(out3, (FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (FLOAT4)0, (FLOAT4)6);
    out1 = clamp(out1, (FLOAT4)0, (FLOAT4)6);
    out2 = clamp(out2, (FLOAT4)0, (FLOAT4)6);
    out3 = clamp(out3, (FLOAT4)0, (FLOAT4)6);
#endif

    const int out_x_base = mul24(out_channel_block_idx, output_shape.y);
    int out_x_idx        = out_height_block_idx << 2;

    const int remain = output_shape.y - out_x_idx;
    int output_idx   = out_x_base + out_x_idx;
    if (remain >= 4) {
        WI_F(output, (int2)(output_idx, output_batch_height_idx), out0);
        WI_F(output, (int2)(output_idx + 1, output_batch_height_idx), out1);
        WI_F(output, (int2)(output_idx + 2, output_batch_height_idx), out2);
        WI_F(output, (int2)(output_idx + 3, output_batch_height_idx), out3);
    } else if (remain == 3) {
        WI_F(output, (int2)(output_idx, output_batch_height_idx), out0);
        WI_F(output, (int2)(output_idx + 1, output_batch_height_idx), out1);
        WI_F(output, (int2)(output_idx + 2, output_batch_height_idx), out2);
    } else if (remain == 2) {
        WI_F(output, (int2)(output_idx, output_batch_height_idx), out0);
        WI_F(output, (int2)(output_idx + 1, output_batch_height_idx), out1);
    } else if (remain == 1) {
        WI_F(output, (int2)(output_idx, output_batch_height_idx), out0);
    }
}

__kernel
#if SET_ATTRIBUTE
__attribute__((work_group_size_hint(16, 16, 1)))
#endif
void conv_2d_c8h4w1(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                      __global const char *kernel_ptr,
                      __global const float *dequantScaleOffset,
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
                      __global const uchar *kernel_ptr,
                      __global const float *dequantScaleOffset,
#elif (defined USE_BUFFER)
                      __global const FLOAT *weights,
#else
                      __read_only image2d_t weights,
#endif
#ifdef BIAS
                      __read_only image2d_t bias,
#endif
                      __write_only image2d_t output,
                      __private const int2 input_shape,
                      __private const int in_channel_block_length,
                      __private const int2 output_shape,
                      __private const int2 weights_shape,
                      __private const int2 stride_shape,
                      __private const int2 padding_shape,
                      __private const int2 dilation_shape,
                      __private const int out_width_blocks,
                      __private const int out_channel_blocks,
                      __private const int out_height_blocks
#if (defined USE_LOW_BIT_WEIGHT_INT8) || (defined USE_LOW_BIT_WEIGHT_INT4)
                     ,__private const int blockDim
                     ,__private const int inChannel
#endif
) {

    const int output_channel_width_idx = get_global_id(0);
    const int output_batch_height_idx  = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(output_channel_width_idx, output_batch_height_idx);

    const int out_channel_block_idx = (output_channel_width_idx / out_width_blocks) << 1;
    const int out_width_block_idx   = output_channel_width_idx % out_width_blocks;
    const int out_height_block_idx   = (output_batch_height_idx % out_height_blocks);
    const int out_batch_block_idx   = output_batch_height_idx / out_height_blocks;

#ifdef BIAS
    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(out_channel_block_idx, 0));
    FLOAT4 out4 = RI_F(bias, SAMPLER, (int2)(out_channel_block_idx + 1, 0));
#else
    FLOAT4 out0 = (FLOAT4)0;
    FLOAT4 out4 = (FLOAT4)0;
#endif
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;
    FLOAT4 out5 = out4;
    FLOAT4 out6 = out4;
    FLOAT4 out7 = out4;

#if (defined USE_LOW_BIT_WEIGHT_INT8) || (defined USE_LOW_BIT_WEIGHT_INT4) || (defined USE_BUFFER)
    const int weight_oc_offset = weights_shape.x * weights_shape.y * 4;
    const int weight_ic_offset = out_channel_blocks * weight_oc_offset;
#endif

    int in_width0          = mad24(out_width_block_idx, stride_shape.y, -padding_shape.y);
    int in_height0         = mad24(out_height_block_idx, stride_shape.x<<2, -padding_shape.x);
    int in_height1         = in_height0 + stride_shape.x;
    int in_height2         = in_height1 + stride_shape.x;
    int in_height3         = in_height2 + stride_shape.x;
    int weight_size        = mul24(weights_shape.y, weights_shape.x);
    
    const int weights_h_idx = mul24(out_channel_block_idx, weight_size);
    const int batch_idx = mul24(out_batch_block_idx, input_shape.x);
    
    FLOAT4 in0, in1, in2, in3;
    FLOAT4 weights0, weights1, weights2, weights3, weights4, weights5, weights6, weights7;
    for (int in_channel_block_idx = 0; in_channel_block_idx < in_channel_block_length; ++in_channel_block_idx) {
#if (defined USE_LOW_BIT_WEIGHT_INT8) || (defined USE_LOW_BIT_WEIGHT_INT4)
        int kindex = (in_channel_block_idx * 4) / blockDim * out_channel_blocks * 8;
        COMPUTE_FLOAT8 ScaleOffset0 = CONVERT_COMPUTE_FLOAT8(vload8(out_channel_block_idx, dequantScaleOffset + kindex));
        COMPUTE_FLOAT4 scale0 = (COMPUTE_FLOAT4)(ScaleOffset0.s0, ScaleOffset0.s2, ScaleOffset0.s4, ScaleOffset0.s6);
        COMPUTE_FLOAT4 offset0 = (COMPUTE_FLOAT4)(ScaleOffset0.s1, ScaleOffset0.s3, ScaleOffset0.s5, ScaleOffset0.s7);
        COMPUTE_FLOAT8 ScaleOffset1 = CONVERT_COMPUTE_FLOAT8(vload8(out_channel_block_idx + 1, dequantScaleOffset + kindex));
        COMPUTE_FLOAT4 scale1 = (COMPUTE_FLOAT4)(ScaleOffset1.s0, ScaleOffset1.s2, ScaleOffset1.s4, ScaleOffset1.s6);
        COMPUTE_FLOAT4 offset1 = (COMPUTE_FLOAT4)(ScaleOffset1.s1, ScaleOffset1.s3, ScaleOffset1.s5, ScaleOffset1.s7);
                                                        
#endif
        const int in_idx = mul24(in_channel_block_idx, input_shape.y);
#if (defined USE_LOW_BIT_WEIGHT_INT8) || (defined USE_LOW_BIT_WEIGHT_INT4) || (defined USE_BUFFER)
        int weight_offset = ((((4*in_channel_block_idx+0)* out_channel_blocks + out_channel_block_idx) *weights_shape.x + 0)*weights_shape.y + 0) * 4;
#else
        int weights_x_idx = in_channel_block_idx << 2;
        int weights_y_idx = weights_h_idx;
#endif
        for (int iy = 0; iy < weights_shape.x * dilation_shape.x; iy += dilation_shape.x) {
            int h0 =  select(in_height0 + iy + batch_idx, -1, (in_height0 + iy < 0 || in_height0 + iy  >= input_shape.x));
            int h1 =  select(in_height1 + iy + batch_idx, -1, (in_height1 + iy < 0 || in_height1 + iy  >= input_shape.x));
            int h2 =  select(in_height2 + iy + batch_idx, -1, (in_height2 + iy < 0 || in_height2 + iy  >= input_shape.x));
            int h3 =  select(in_height3 + iy + batch_idx, -1, (in_height3 + iy < 0 || in_height3 + iy  >= input_shape.x));
            for (int ix = 0; ix < weights_shape.y * dilation_shape.y; ix += dilation_shape.y) {
                int w0 =  select(in_width0 + ix + in_idx, -1, (in_width0 + ix < 0 || in_width0 + ix  >= input_shape.y));
                
                in0 = RI_F(input, SAMPLER, (int2)(w0, h0));
                in1 = RI_F(input, SAMPLER, (int2)(w0, h1));
                in2 = RI_F(input, SAMPLER, (int2)(w0, h2));
                in3 = RI_F(input, SAMPLER, (int2)(w0, h3));

#if (defined USE_LOW_BIT_WEIGHT_INT8)
                char4 charWeight0 = vload4(0, kernel_ptr+weight_offset);
                char4 charWeight1 = vload4(0, kernel_ptr+weight_offset+weight_ic_offset);
                char4 charWeight2 = vload4(0, kernel_ptr+weight_offset+weight_ic_offset*2);
                char4 charWeight3 = vload4(0, kernel_ptr+weight_offset+weight_ic_offset*3);
                weights0 = mad(CONVERT_FLOAT4(charWeight0), scale0, offset0);
                weights1 = mad(CONVERT_FLOAT4(charWeight1), scale0, offset0);
                weights2 = mad(CONVERT_FLOAT4(charWeight2), scale0, offset0);
                weights3 = mad(CONVERT_FLOAT4(charWeight3), scale0, offset0);
                charWeight0 = vload4(0, kernel_ptr+weight_offset+weight_oc_offset);
                charWeight1 = vload4(0, kernel_ptr+weight_offset+weight_oc_offset+weight_ic_offset);
                charWeight2 = vload4(0, kernel_ptr+weight_offset+weight_oc_offset+weight_ic_offset*2);
                charWeight3 = vload4(0, kernel_ptr+weight_offset+weight_oc_offset+weight_ic_offset*3);
                weights4 = mad(CONVERT_FLOAT4(charWeight0), scale1, offset1);
                weights5 = mad(CONVERT_FLOAT4(charWeight1), scale1, offset1);
                weights6 = mad(CONVERT_FLOAT4(charWeight2), scale1, offset1);
                weights7 = mad(CONVERT_FLOAT4(charWeight3), scale1, offset1);
                weight_offset += 4;
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
                uchar2 charWeightInt40 = vload2(0, kernel_ptr+weight_offset/2);
                uchar2 charWeightInt41 = vload2(0, kernel_ptr+weight_offset/2+weight_ic_offset/2);
                uchar2 charWeightInt42 = vload2(0, kernel_ptr+weight_offset/2+weight_ic_offset*2/2);
                uchar2 charWeightInt43 = vload2(0, kernel_ptr+weight_offset/2+weight_ic_offset*3/2);
                char4 charWeight0 = (char4)(0, 0, 0, 0);
                char4 charWeight1 = (char4)(0, 0, 0, 0);
                char4 charWeight2 = (char4)(0, 0, 0, 0);
                char4 charWeight3 = (char4)(0, 0, 0, 0);
                charWeight0.x = (charWeightInt40.s0 >> 4) - 8;
                charWeight0.y = (charWeightInt40.s0 & MOD_NUM) - 8;
                charWeight0.z = (charWeightInt40.s1 >> 4) - 8;
                charWeight0.w = (charWeightInt40.s1 & MOD_NUM) - 8;
                charWeight1.x = (charWeightInt41.s0 >> 4) - 8;
                charWeight1.y = (charWeightInt41.s0 & MOD_NUM) - 8;
                charWeight1.z = (charWeightInt41.s1 >> 4) - 8;
                charWeight1.w = (charWeightInt41.s1 & MOD_NUM) - 8;
                charWeight2.x = (charWeightInt42.s0 >> 4) - 8;
                charWeight2.y = (charWeightInt42.s0 & MOD_NUM) - 8;
                charWeight2.z = (charWeightInt42.s1 >> 4) - 8;
                charWeight2.w = (charWeightInt42.s1 & MOD_NUM)- 8;
                charWeight3.x = (charWeightInt43.s0 >> 4) - 8;
                charWeight3.y = (charWeightInt43.s0 & MOD_NUM) - 8;
                charWeight3.z = (charWeightInt43.s1 >> 4) - 8;
                charWeight3.w = (charWeightInt43.s1 & MOD_NUM) - 8;
                weights0 = mad(CONVERT_FLOAT4(charWeight0), scale0, offset0);
                weights1 = mad(CONVERT_FLOAT4(charWeight1), scale0, offset0);
                weights2 = mad(CONVERT_FLOAT4(charWeight2), scale0, offset0);
                weights3 = mad(CONVERT_FLOAT4(charWeight3), scale0, offset0);
                charWeightInt40 = vload2(0, kernel_ptr+weight_offset/2+weight_oc_offset/2);
                charWeightInt41 = vload2(0, kernel_ptr+weight_offset/2+weight_oc_offset/2+weight_ic_offset/2);
                charWeightInt42 = vload2(0, kernel_ptr+weight_offset/2+weight_oc_offset/2+weight_ic_offset*2/2);
                charWeightInt43 = vload2(0, kernel_ptr+weight_offset/2+weight_oc_offset/2+weight_ic_offset*3/2);
                charWeight0 = (char4)(0, 0, 0, 0);
                charWeight1 = (char4)(0, 0, 0, 0);
                charWeight2 = (char4)(0, 0, 0, 0);
                charWeight3 = (char4)(0, 0, 0, 0);
                charWeight0.x = (charWeightInt40.s0 >> 4) - 8;
                charWeight0.y = (charWeightInt40.s0 & MOD_NUM) - 8;
                charWeight0.z = (charWeightInt40.s1 >> 4) - 8;
                charWeight0.w = (charWeightInt40.s1 & MOD_NUM) - 8;
                charWeight1.x = (charWeightInt41.s0 >> 4) - 8;
                charWeight1.y = (charWeightInt41.s0 & MOD_NUM) - 8;
                charWeight1.z = (charWeightInt41.s1 >> 4) - 8;
                charWeight1.w = (charWeightInt41.s1 & MOD_NUM)- 8;
                charWeight2.x = (charWeightInt42.s0 >> 4) - 8;
                charWeight2.y = (charWeightInt42.s0 & MOD_NUM) - 8;
                charWeight2.z = (charWeightInt42.s1 >> 4) - 8;
                charWeight2.w = (charWeightInt42.s1 & MOD_NUM)- 8;
                charWeight3.x = (charWeightInt43.s0 >> 4) - 8;
                charWeight3.y = (charWeightInt43.s0 & MOD_NUM) - 8;
                charWeight3.z = (charWeightInt43.s1 >> 4) - 8;
                charWeight3.w = (charWeightInt43.s1 & MOD_NUM) - 8;
                weights4 = mad(CONVERT_FLOAT4(charWeight0), scale1, offset1);
                weights5 = mad(CONVERT_FLOAT4(charWeight1), scale1, offset1);
                weights6 = mad(CONVERT_FLOAT4(charWeight2), scale1, offset1);
                weights7 = mad(CONVERT_FLOAT4(charWeight3), scale1, offset1);
                weight_offset += 4;
#elif (defined USE_BUFFER)
                weights0 = vload4(0, weights+weight_offset);
                weights1 = vload4(0, weights+weight_offset+weight_ic_offset);
                weights2 = vload4(0, weights+weight_offset+weight_ic_offset*2);
                weights3 = vload4(0, weights+weight_offset+weight_ic_offset*3);
                weights4 = vload4(0, weights+weight_offset + weight_oc_offset);
                weights5 = vload4(0, weights+weight_offset+weight_ic_offset + weight_oc_offset);
                weights6 = vload4(0, weights+weight_offset+weight_ic_offset*2 + weight_oc_offset);
                weights7 = vload4(0, weights+weight_offset+weight_ic_offset*3 + weight_oc_offset);
                weight_offset += 4;
#else
                weights0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 0, weights_y_idx));
                weights1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx));
                weights2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx));
                weights3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx));
                weights4 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 0, weight_size + weights_y_idx));
                weights5 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weight_size + weights_y_idx));
                weights6 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weight_size + weights_y_idx));
                weights7 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weight_size + weights_y_idx++));
#endif
                PADZEROSVEC(in_channel_block_idx, inChannel, weights0, weights1, weights2, weights3);
                PADZEROSVEC(in_channel_block_idx, inChannel, weights4, weights5, weights6, weights7);
                
                CALCULATE_OUTPUT(0);
                CALCULATE_OUTPUT(1);
                CALCULATE_OUTPUT(2);
                CALCULATE_OUTPUT(3);
                CALCULATE_OUTPUT_WEIGHTS4(4, 0);
                CALCULATE_OUTPUT_WEIGHTS4(5, 1);
                CALCULATE_OUTPUT_WEIGHTS4(6, 2);
                CALCULATE_OUTPUT_WEIGHTS4(7, 3);
            }
        }
    }

#ifdef RELU
    out0 = fmax(out0, (FLOAT4)0);
    out1 = fmax(out1, (FLOAT4)0);
    out2 = fmax(out2, (FLOAT4)0);
    out3 = fmax(out3, (FLOAT4)0);
    out4 = fmax(out4, (FLOAT4)0);
    out5 = fmax(out5, (FLOAT4)0);
    out6 = fmax(out6, (FLOAT4)0);
    out7 = fmax(out7, (FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (FLOAT4)0, (FLOAT4)6);
    out1 = clamp(out1, (FLOAT4)0, (FLOAT4)6);
    out2 = clamp(out2, (FLOAT4)0, (FLOAT4)6);
    out3 = clamp(out3, (FLOAT4)0, (FLOAT4)6);
    out4 = clamp(out4, (FLOAT4)0, (FLOAT4)6);
    out5 = clamp(out5, (FLOAT4)0, (FLOAT4)6);
    out6 = clamp(out6, (FLOAT4)0, (FLOAT4)6);
    out7 = clamp(out7, (FLOAT4)0, (FLOAT4)6);
#endif

    const int out_x_base = mul24(out_channel_block_idx, output_shape.y);
    const int out_y_base = mul24(out_batch_block_idx, output_shape.x);
    int out_x_idx        = out_width_block_idx;
    int out_y_idx        = out_height_block_idx << 2;

    const int remain_y = output_shape.x - out_y_idx;
    int output_idx   = out_x_base + out_x_idx;
    int output_idy   = out_y_base + out_y_idx;
    
    if(remain_y >= 4){
        WI_F(output, (int2)(output_idx, output_idy), out0);
        WI_F(output, (int2)(output_idx, output_idy + 1), out1);
        WI_F(output, (int2)(output_idx, output_idy + 2), out2);
        WI_F(output, (int2)(output_idx, output_idy + 3), out3);
    }else if(remain_y == 3){
        WI_F(output, (int2)(output_idx, output_idy), out0);
        WI_F(output, (int2)(output_idx, output_idy + 1), out1);
        WI_F(output, (int2)(output_idx, output_idy + 2), out2);
    }else if(remain_y == 2){
        WI_F(output, (int2)(output_idx, output_idy), out0);
        WI_F(output, (int2)(output_idx, output_idy + 1), out1);
    }else if(remain_y == 1){
        WI_F(output, (int2)(output_idx, output_idy), out0);
    }
    
    if(out_channel_block_idx + 1 >= out_channel_blocks) {
        return;
    }
    output_idx   += output_shape.y;
    if(remain_y >= 4){
        WI_F(output, (int2)(output_idx, output_idy), out4);
        WI_F(output, (int2)(output_idx, output_idy + 1), out5);
        WI_F(output, (int2)(output_idx, output_idy + 2), out6);
        WI_F(output, (int2)(output_idx, output_idy + 3), out7);
    }else if(remain_y == 3){
        WI_F(output, (int2)(output_idx, output_idy), out4);
        WI_F(output, (int2)(output_idx, output_idy + 1), out5);
        WI_F(output, (int2)(output_idx, output_idy + 2), out6);
    }else if(remain_y == 2){
        WI_F(output, (int2)(output_idx, output_idy), out4);
        WI_F(output, (int2)(output_idx, output_idy + 1), out5);
    }else if(remain_y == 1){
        WI_F(output, (int2)(output_idx, output_idy), out4);
    }
}

__kernel
#if SET_ATTRIBUTE
__attribute__((work_group_size_hint(16, 16, 1)))
#endif
void conv_2d_c4h4w1(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                      __global const char *kernel_ptr,
                      __global const float *dequantScaleOffset,
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
                      __global const uchar *kernel_ptr,
                      __global const float *dequantScaleOffset,
#elif (defined USE_BUFFER)
                      __global const FLOAT *weights,
#else
                      __read_only image2d_t weights,
#endif
#ifdef BIAS
                      __read_only image2d_t bias,
#endif
                      __write_only image2d_t output,
                      __private const int2 input_shape,
                      __private const int in_channel_block_length,
                      __private const int2 output_shape,
                      __private const int2 weights_shape,
                      __private const int2 stride_shape,
                      __private const int2 padding_shape,
                      __private const int2 dilation_shape,
                      __private const int out_width_blocks,
                      __private const int out_channel_blocks,
                      __private const int out_height_blocks
#if (defined USE_LOW_BIT_WEIGHT_INT8) || (defined USE_LOW_BIT_WEIGHT_INT4)
                     ,__private const int blockDim
                     ,__private const int inChannel
#endif
) {

    const int output_channel_width_idx = get_global_id(0);
    const int output_batch_height_idx  = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(output_channel_width_idx, output_batch_height_idx);

    const int out_channel_block_idx = output_channel_width_idx / out_width_blocks;
    const int out_width_block_idx   = output_channel_width_idx % out_width_blocks;
    const int out_height_block_idx   = (output_batch_height_idx % out_height_blocks);
    const int out_batch_block_idx   = output_batch_height_idx / out_height_blocks;

#ifdef BIAS
    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(out_channel_block_idx, 0));
#else
    FLOAT4 out0 = (FLOAT4)0;
#endif
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    int in_width0          = mad24(out_width_block_idx, stride_shape.y, -padding_shape.y);
    int in_height0         = mad24(out_height_block_idx, stride_shape.x<<2, -padding_shape.x);
    int in_height1         = in_height0 + stride_shape.x;
    int in_height2         = in_height1 + stride_shape.x;
    int in_height3         = in_height2 + stride_shape.x;
    int weight_size        = mul24(weights_shape.y, weights_shape.x);
    
    const int weights_h_idx = mul24(out_channel_block_idx, weight_size);
    const int batch_idx = mul24(out_batch_block_idx, input_shape.x);
    
    FLOAT4 in0, in1, in2, in3;
    FLOAT4 weights0, weights1, weights2, weights3;
#if (defined USE_LOW_BIT_WEIGHT_INT8) || (defined USE_LOW_BIT_WEIGHT_INT4) || (defined USE_BUFFER)
    const int weight_oc_offset = out_channel_blocks * weights_shape.x * weights_shape.y * 4;
#endif
    for (int in_channel_block_idx = 0; in_channel_block_idx < in_channel_block_length; ++in_channel_block_idx) {
#if (defined USE_LOW_BIT_WEIGHT_INT8) || (defined USE_LOW_BIT_WEIGHT_INT4)
        int kindex = (in_channel_block_idx * 4) / blockDim * out_channel_blocks * 8;
        COMPUTE_FLOAT8 ScaleOffset0 = CONVERT_COMPUTE_FLOAT8(vload8(out_channel_block_idx, dequantScaleOffset + kindex));
        COMPUTE_FLOAT4 scale0 = (COMPUTE_FLOAT4)(ScaleOffset0.s0, ScaleOffset0.s2, ScaleOffset0.s4, ScaleOffset0.s6);
        COMPUTE_FLOAT4 offset0 = (COMPUTE_FLOAT4)(ScaleOffset0.s1, ScaleOffset0.s3, ScaleOffset0.s5, ScaleOffset0.s7);
#endif
        const int in_idx = mul24(in_channel_block_idx, input_shape.y);
#if (defined USE_LOW_BIT_WEIGHT_INT8) || (defined USE_LOW_BIT_WEIGHT_INT4) || (defined USE_BUFFER)
        int weight_offset = ((((4*in_channel_block_idx+0)* out_channel_blocks + out_channel_block_idx) *weights_shape.x + 0)*weights_shape.y + 0) * 4;
#else
        int weights_x_idx = in_channel_block_idx << 2;
        int weights_y_idx = weights_h_idx;
#endif
        for (int iy = 0; iy < weights_shape.x * dilation_shape.x; iy += dilation_shape.x) {
            int h0 =  select(in_height0 + iy + batch_idx, -1, (in_height0 + iy < 0 || in_height0 + iy  >= input_shape.x));
            int h1 =  select(in_height1 + iy + batch_idx, -1, (in_height1 + iy < 0 || in_height1 + iy  >= input_shape.x));
            int h2 =  select(in_height2 + iy + batch_idx, -1, (in_height2 + iy < 0 || in_height2 + iy  >= input_shape.x));
            int h3 =  select(in_height3 + iy + batch_idx, -1, (in_height3 + iy < 0 || in_height3 + iy  >= input_shape.x));
            for (int ix = 0; ix < weights_shape.y * dilation_shape.y; ix += dilation_shape.y) {
                int w0 =  select(in_width0 + ix + in_idx, -1, (in_width0 + ix < 0 || in_width0 + ix  >= input_shape.y));
                
                in0 = RI_F(input, SAMPLER, (int2)(w0, h0));
                in1 = RI_F(input, SAMPLER, (int2)(w0, h1));
                in2 = RI_F(input, SAMPLER, (int2)(w0, h2));
                in3 = RI_F(input, SAMPLER, (int2)(w0, h3));
                
#if (defined USE_LOW_BIT_WEIGHT_INT8)
                char4 charWeight0 = vload4(0, kernel_ptr+weight_offset);
                char4 charWeight1 = vload4(0, kernel_ptr+weight_offset+weight_oc_offset);
                char4 charWeight2 = vload4(0, kernel_ptr+weight_offset+weight_oc_offset*2);
                char4 charWeight3 = vload4(0, kernel_ptr+weight_offset+weight_oc_offset*3);
                weights0 = mad(CONVERT_FLOAT4(charWeight0), scale0, offset0);
                weights1 = mad(CONVERT_FLOAT4(charWeight1), scale0, offset0);
                weights2 = mad(CONVERT_FLOAT4(charWeight2), scale0, offset0);
                weights3 = mad(CONVERT_FLOAT4(charWeight3), scale0, offset0);
                weight_offset += 4;
#elif (defined USE_LOW_BIT_WEIGHT_INT4)
                uchar2 charWeightInt40 = vload2(0, kernel_ptr+weight_offset/2);
                uchar2 charWeightInt41 = vload2(0, kernel_ptr+weight_offset/2+weight_oc_offset/2);
                uchar2 charWeightInt42 = vload2(0, kernel_ptr+weight_offset/2+weight_oc_offset*2/2);
                uchar2 charWeightInt43 = vload2(0, kernel_ptr+weight_offset/2+weight_oc_offset*3/2);
                char4 charWeight0 = (char4)(0, 0, 0, 0);
                char4 charWeight1 = (char4)(0, 0, 0, 0);
                char4 charWeight2 = (char4)(0, 0, 0, 0);
                char4 charWeight3 = (char4)(0, 0, 0, 0);
                charWeight0.x = (charWeightInt40.s0 >> 4) - 8;
                charWeight0.y = (charWeightInt40.s0 & MOD_NUM) - 8;
                charWeight0.z = (charWeightInt40.s1 >> 4) - 8;
                charWeight0.w = (charWeightInt40.s1 & MOD_NUM) - 8;
                charWeight1.x = (charWeightInt41.s0 >> 4) - 8;
                charWeight1.y = (charWeightInt41.s0 & MOD_NUM) - 8;
                charWeight1.z = (charWeightInt41.s1 >> 4) - 8;
                charWeight1.w = (charWeightInt41.s1 & MOD_NUM) - 8;
                charWeight2.x = (charWeightInt42.s0 >> 4) - 8;
                charWeight2.y = (charWeightInt42.s0 & MOD_NUM) - 8;
                charWeight2.z = (charWeightInt42.s1 >> 4) - 8;
                charWeight2.w = (charWeightInt42.s1 & MOD_NUM) - 8;
                charWeight3.x = (charWeightInt43.s0 >> 4) - 8;
                charWeight3.y = (charWeightInt43.s0 & MOD_NUM) - 8;
                charWeight3.z = (charWeightInt43.s1 >> 4) - 8;
                charWeight3.w = (charWeightInt43.s1 & MOD_NUM) - 8;
                weights0 = mad(CONVERT_FLOAT4(charWeight0), scale0, offset0);
                weights1 = mad(CONVERT_FLOAT4(charWeight1), scale0, offset0);
                weights2 = mad(CONVERT_FLOAT4(charWeight2), scale0, offset0);
                weights3 = mad(CONVERT_FLOAT4(charWeight3), scale0, offset0);
                weight_offset += 4;
#elif (defined USE_BUFFER)
                weights0 = vload4(0, weights+weight_offset);
                weights1 = vload4(0, weights+weight_offset+weight_oc_offset);
                weights2 = vload4(0, weights+weight_offset+weight_oc_offset*2);
                weights3 = vload4(0, weights+weight_offset+weight_oc_offset*3);
                weight_offset += 4;
#else
                weights0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 0, weights_y_idx));
                weights1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx));
                weights2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx));
                weights3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx++));
#endif
                PADZEROSVEC(in_channel_block_idx, inChannel, weights0, weights1, weights2, weights3);

                CALCULATE_OUTPUT(0);
                CALCULATE_OUTPUT(1);
                CALCULATE_OUTPUT(2);
                CALCULATE_OUTPUT(3);
            }
        }
    }

#ifdef RELU
    out0 = fmax(out0, (FLOAT4)0);
    out1 = fmax(out1, (FLOAT4)0);
    out2 = fmax(out2, (FLOAT4)0);
    out3 = fmax(out3, (FLOAT4)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (FLOAT4)0, (FLOAT4)6);
    out1 = clamp(out1, (FLOAT4)0, (FLOAT4)6);
    out2 = clamp(out2, (FLOAT4)0, (FLOAT4)6);
    out3 = clamp(out3, (FLOAT4)0, (FLOAT4)6);
#endif

    const int out_x_base = mul24(out_channel_block_idx, output_shape.y);
    const int out_y_base = mul24(out_batch_block_idx, output_shape.x);
    int out_x_idx        = out_width_block_idx;
    int out_y_idx        = out_height_block_idx << 2;

    const int remain_y = output_shape.x - out_y_idx;
    int output_idx   = out_x_base + out_x_idx;
    int output_idy   = out_y_base + out_y_idx;

    if(remain_y >= 4){
        WI_F(output, (int2)(output_idx, output_idy), out0);
        WI_F(output, (int2)(output_idx, output_idy + 1), out1);
        WI_F(output, (int2)(output_idx, output_idy + 2), out2);
        WI_F(output, (int2)(output_idx, output_idy + 3), out3);
    }else if(remain_y == 3){
        WI_F(output, (int2)(output_idx, output_idy), out0);
        WI_F(output, (int2)(output_idx, output_idy + 1), out1);
        WI_F(output, (int2)(output_idx, output_idy + 2), out2);
    }else if(remain_y == 2){
        WI_F(output, (int2)(output_idx, output_idy), out0);
        WI_F(output, (int2)(output_idx, output_idy + 1), out1);
    }else{
        WI_F(output, (int2)(output_idx, output_idy), out0);
    }
}
