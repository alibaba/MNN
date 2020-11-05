#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#define READ_INPUT_IMAGE(i, base)                                                                         \
    int inOffset##i = inWidthOffset##i + base;                                                           \
    inOffset##i =                                                                                   \
        select(inCurIdx + inOffset##i, -1, (inOffset##i < 0 || inOffset##i >= inputShape.y)); \
    inValue##i = RI_F(input, SAMPLER, (int2)(inOffset##i, inHeightIdx));

#define CALCULATE_OUTPUT(i)                  \
    outValue##i = mad(inValue##i.x, weights0, outValue##i); \
    outValue##i = mad(inValue##i.y, weights1, outValue##i); \
    outValue##i = mad(inValue##i.z, weights2, outValue##i); \
    outValue##i = mad(inValue##i.w, weights3, outValue##i);

#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                     \
    }

__kernel
#if SET_ATTRIBUTE
__attribute__((work_group_size_hint(16, 16, 1)))
#endif
void depthwise_conv2d_s1(GLOBAL_SIZE_2_DIMS __read_only image2d_t input, __read_only image2d_t filter,
                                  #ifndef NO_BIAS
                                  __read_only image2d_t bias,
                                  #endif
                                  __write_only image2d_t output,
                                  __private const int2 inputShape,
                                  __private const int inChannelBlocks, 
                                  __private const int2 outputShape,
                                  __private const int2 filterShape,
                                  __private const int2 paddingShape) {

    const int outChannelWidthIdx = get_global_id(0);
    const int outHeightBlockIdx     = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(outChannelWidthIdx, outHeightBlockIdx);
    int ow4              = (outputShape.y + 3) / 4;
    const int outChannelBlockIdx = outChannelWidthIdx / ow4;
    const int outWidthBlockidx   = outChannelWidthIdx % ow4;

    const int inChannelBlockIdx = outChannelBlockIdx;

    #ifndef NO_BIAS
    FLOAT4 outValue0 = RI_F(bias, SAMPLER, (int2)(outChannelBlockIdx, 0));
    #else
    FLOAT4 outValue0 = (FLOAT4)(0.0f);
    #endif
    FLOAT4 outValue1 = outValue0;
    FLOAT4 outValue2 = outValue0;
    FLOAT4 outValue3 = outValue0;

    const int outWidthBlockidx4 = outWidthBlockidx << 2;
    const int inWidthOffset0             = outWidthBlockidx4 - paddingShape.y;
    const int inWidthOffset1             = inWidthOffset0 + 1;
    const int inWidthOffset2             = inWidthOffset0 + 2;
    const int inWidthOffset3             = inWidthOffset0 + 3;

    int heightIdx            = outHeightBlockIdx % outputShape.x - paddingShape.x;
    const int outBatchIdx = mul24((outHeightBlockIdx / outputShape.x), inputShape.x);
    const int inCurIdx = mul24(inChannelBlockIdx, inputShape.y);

    const int inWidthIdx0 = select(inCurIdx + inWidthOffset0, -1, (inWidthOffset0 < 0 || inWidthOffset0 >= inputShape.y));
    const int inWidthIdx1 = select(inCurIdx + inWidthOffset1, -1, (inWidthOffset1 < 0 || inWidthOffset1 >= inputShape.y));
    const int inWidthIdx2 = select(inCurIdx + inWidthOffset2, -1, (inWidthOffset2 < 0 || inWidthOffset2 >= inputShape.y));

    FLOAT4 inValue0, inValue1, inValue2, inValue3;
    for (int kh = 0; kh < filterShape.x; kh++) {
        int inHeightIdx = select(heightIdx + outBatchIdx, -1, (heightIdx < 0 || heightIdx >= inputShape.x));
        heightIdx++;
        inValue1       = RI_F(input, SAMPLER, (int2)(inWidthIdx0, inHeightIdx));
        inValue2       = RI_F(input, SAMPLER, (int2)(inWidthIdx1, inHeightIdx));
        inValue3       = RI_F(input, SAMPLER, (int2)(inWidthIdx2, inHeightIdx));
        for (int kw = 0; kw < filterShape.y; kw++) {

            int filterIdx   = mad24(kh, filterShape.y, kw);
            inValue0 = inValue1;
            inValue1 = inValue2;
            inValue2 = inValue3;

            int inWidthIdx = inWidthOffset3 + kw;
            inWidthIdx = select(inCurIdx + inWidthIdx, -1, (inWidthIdx < 0 || inWidthIdx >= inputShape.y));
            inValue3  = RI_F(input, SAMPLER, (int2)(inWidthIdx, inHeightIdx));

            FLOAT4 weights = RI_F(filter, SAMPLER, (int2)(filterIdx, inChannelBlockIdx));

            outValue0 = mad(inValue0, weights, outValue0);
            outValue1 = mad(inValue1, weights, outValue1);
            outValue2 = mad(inValue2, weights, outValue2);
            outValue3 = mad(inValue3, weights, outValue3);
        }
    }

#ifdef RELU
    outValue0 = fmax(outValue0, (FLOAT4)0);
    outValue1 = fmax(outValue1, (FLOAT4)0);
    outValue2 = fmax(outValue2, (FLOAT4)0);
    outValue3 = fmax(outValue3, (FLOAT4)0);
#endif

#ifdef RELU6
    outValue0 = clamp(outValue0, (FLOAT4)0, (FLOAT4)6);
    outValue1 = clamp(outValue1, (FLOAT4)0, (FLOAT4)6);
    outValue2 = clamp(outValue2, (FLOAT4)0, (FLOAT4)6);
    outValue3 = clamp(outValue3, (FLOAT4)0, (FLOAT4)6);
#endif

    const int remain     = outputShape.y - outWidthBlockidx4;
    int outWidthIdx       = mul24(outChannelBlockIdx, outputShape.y) + outWidthBlockidx4;
    if (remain >= 4) {
        WI_F(output, (int2)(outWidthIdx, outHeightBlockIdx), outValue0);
        WI_F(output, (int2)(outWidthIdx + 1, outHeightBlockIdx), outValue1);
        WI_F(output, (int2)(outWidthIdx + 2, outHeightBlockIdx), outValue2);
        WI_F(output, (int2)(outWidthIdx + 3, outHeightBlockIdx), outValue3);
    } else if (remain == 3) {
        WI_F(output, (int2)(outWidthIdx, outHeightBlockIdx), outValue0);
        WI_F(output, (int2)(outWidthIdx + 1, outHeightBlockIdx), outValue1);
        WI_F(output, (int2)(outWidthIdx + 2, outHeightBlockIdx), outValue2);
    } else if (remain == 2) {
        WI_F(output, (int2)(outWidthIdx, outHeightBlockIdx), outValue0);
        WI_F(output, (int2)(outWidthIdx + 1, outHeightBlockIdx), outValue1);
    } else if (remain == 1) {
        WI_F(output, (int2)(outWidthIdx, outHeightBlockIdx), outValue0);
    }
}

__kernel
#if SET_ATTRIBUTE
__attribute__((work_group_size_hint(16, 16, 1)))
#endif
void depthwise_conv2d(GLOBAL_SIZE_2_DIMS __read_only image2d_t input, __read_only image2d_t filter,
                               #ifndef NO_BIAS
                               __read_only image2d_t bias,
                               #endif
                               __write_only image2d_t output,
                               __private const int2 inputShape,
                               __private const int inChannelBlocks, __private const int2 outputShape,
                               __private const int2 filterShape,
                               __private const int2 paddingShape,
                               __private const int2 dilationShape,
                               __private const int2 strideShape) {

    const int outChannelWidthIdx = get_global_id(0);
    const int outHeightIdx     = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(outChannelWidthIdx, outHeightIdx);

    int ow4              = (outputShape.y + 3) / 4;
    const int outChannelBlockIdx = outChannelWidthIdx / ow4;
    const int outWidthBlockidx   = outChannelWidthIdx % ow4;

    const int inChannelBlockIdx = outChannelBlockIdx;

    #ifndef NO_BIAS
    FLOAT4 outValue0 = RI_F(bias, SAMPLER, (int2)(outChannelBlockIdx, 0));
    #else
    FLOAT4 outValue0 = (FLOAT4)(0.0f);
    #endif
    FLOAT4 outValue1 = outValue0;
    FLOAT4 outValue2 = outValue0;
    FLOAT4 outValue3 = outValue0;

    const int inWidthOffset0  = mad24(outWidthBlockidx, strideShape.y << 2, -paddingShape.y);
    const int inWidthOffset1  = inWidthOffset0 + strideShape.y;
    const int inWidthOffset2  = inWidthOffset1 + strideShape.y;
    const int inWidthOffset3  = inWidthOffset2 + strideShape.y;
    int heightIdx = mad24(outHeightIdx % outputShape.x, strideShape.x, -paddingShape.x);

    const int outBatchIdx = mul24((outHeightIdx / outputShape.x), inputShape.x);

    const int inCurIdx = mul24(inChannelBlockIdx, inputShape.y);
    for (int kh = 0; kh < filterShape.x; kh++) {
        int inHeightIdx = select(heightIdx + outBatchIdx, -1, (heightIdx < 0 || heightIdx >= inputShape.x));
        heightIdx += dilationShape.x;
        for (int kw = 0; kw < filterShape.y; kw++) {
            int filterIdx = mad24(kh, filterShape.y, kw);
            FLOAT4 inValue0, inValue1, inValue2, inValue3;
            int inWidthIdx = mul24(kw, dilationShape.y);

            READ_INPUT_IMAGE(0, inWidthIdx);
            READ_INPUT_IMAGE(1, inWidthIdx);
            READ_INPUT_IMAGE(2, inWidthIdx);
            READ_INPUT_IMAGE(3, inWidthIdx);

            FLOAT4 weights = RI_F(filter, SAMPLER, (int2)(filterIdx, inChannelBlockIdx));

            outValue0 = mad(inValue0, weights, outValue0);
            outValue1 = mad(inValue1, weights, outValue1);
            outValue2 = mad(inValue2, weights, outValue2);
            outValue3 = mad(inValue3, weights, outValue3);
        }
    }

#ifdef RELU
    outValue0 = fmax(outValue0, (FLOAT4)0);
    outValue1 = fmax(outValue1, (FLOAT4)0);
    outValue2 = fmax(outValue2, (FLOAT4)0);
    outValue3 = fmax(outValue3, (FLOAT4)0);
#endif

#ifdef RELU6
    outValue0 = clamp(outValue0, (FLOAT4)0, (FLOAT4)6);
    outValue1 = clamp(outValue1, (FLOAT4)0, (FLOAT4)6);
    outValue2 = clamp(outValue2, (FLOAT4)0, (FLOAT4)6);
    outValue3 = clamp(outValue3, (FLOAT4)0, (FLOAT4)6);
#endif

    const int outWidthBlockidx4        = outWidthBlockidx << 2;
    const int remain = outputShape.y - outWidthBlockidx4;
    int outWidthIdx   = mul24(outChannelBlockIdx, outputShape.y) + outWidthBlockidx4;
    if (remain >= 4) {
        WI_F(output, (int2)(outWidthIdx, outHeightIdx), outValue0);
        WI_F(output, (int2)(outWidthIdx + 1, outHeightIdx), outValue1);
        WI_F(output, (int2)(outWidthIdx + 2, outHeightIdx), outValue2);
        WI_F(output, (int2)(outWidthIdx + 3, outHeightIdx), outValue3);
    } else if (remain == 3) {
        WI_F(output, (int2)(outWidthIdx, outHeightIdx), outValue0);
        WI_F(output, (int2)(outWidthIdx + 1, outHeightIdx), outValue1);
        WI_F(output, (int2)(outWidthIdx + 2, outHeightIdx), outValue2);
    } else if (remain == 2) {
        WI_F(output, (int2)(outWidthIdx, outHeightIdx), outValue0);
        WI_F(output, (int2)(outWidthIdx + 1, outHeightIdx), outValue1);
    } else if (remain == 1) {
        WI_F(output, (int2)(outWidthIdx, outHeightIdx), outValue0);
    }
}
