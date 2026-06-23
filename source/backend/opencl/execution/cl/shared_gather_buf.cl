#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,
#define DEAL_NON_UNIFORM_DIM2(input1, input2) \
    if ((input1) >= global_size_dim0 || (input2) >= global_size_dim1) { \
        return; \
    }

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void shared_gather_quant_buffer(
    GLOBAL_SIZE_2_DIMS
    __global OUTPUT_TYPE* output,
#ifdef USE_LOW_BIT_WEIGHT_INT8
    __global const char* weight,
#elif defined(USE_LOW_BIT_WEIGHT_INT4)
    __global const uchar* weight,
#else
    __global const FLOAT* weight,
#endif
    __global const int* indices,
    __global const FLOAT* dequantScaleOffset,
    __private const int ic,
    __private const int oc,
    __private const int blockSize,
    __private const float coef
) {
    const int select_idx = get_global_id(0);
    const int k4 = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(select_idx, k4);

    const int base_ic = k4 << 2;
    if (base_ic >= ic) {
        return;
    }

    const int ocIndex = indices[select_idx];
    if (ocIndex < 0 || ocIndex >= oc) {
        return;
    }

    const int icC4 = (ic + 3) >> 2;
    const int out_c_idx = ocIndex >> 2;
    const int oc_in4 = ocIndex & 3;
    const int ocBlock = ocIndex >> 3;
    const int oc_in8 = ocIndex & 7;
    const int dstChannelC4 = ((oc + 3) >> 2) << 2;
    const int tileIndex = ocBlock * icC4 + k4;

#ifdef USE_LOW_BIT_WEIGHT_INT8
    const int weightTileStride = 32;
    const int weightBase = tileIndex * weightTileStride;
#elif defined(USE_LOW_BIT_WEIGHT_INT4)
    const int weightTileStride = 16;
    const int weightBase = tileIndex * weightTileStride;
#else
    const int weightTileStride = 0;
    const int weightBase = 0;
#endif

    const int outBase = select_idx * ic + base_ic;
    COMPUTE_FLOAT4 out4 = (COMPUTE_FLOAT4)(0, 0, 0, 0);

    for (int i = 0; i < 4; ++i) {
        const int icIndex = base_ic + i;
        if (icIndex >= ic) {
            break;
        }

        const int blockIndex = icIndex / blockSize;
        const int channelIndex = (out_c_idx << 2) + oc_in4;
        int scaleIndex = blockIndex * dstChannelC4 + channelIndex;

#ifdef ASYMMETRIC
        scaleIndex = scaleIndex * 2;
        FLOAT sRaw = dequantScaleOffset[scaleIndex + 0];
        FLOAT bRaw = dequantScaleOffset[scaleIndex + 1];
        COMPUTE_FLOAT scale = (COMPUTE_FLOAT)(convert_float(sRaw) / coef);
        COMPUTE_FLOAT offset = (COMPUTE_FLOAT)(convert_float(bRaw) / coef);
#else
        FLOAT sRaw = dequantScaleOffset[scaleIndex];
        COMPUTE_FLOAT scale = (COMPUTE_FLOAT)(convert_float(sRaw) / coef);
        COMPUTE_FLOAT offset = (COMPUTE_FLOAT)0;
#endif

        COMPUTE_FLOAT wVal = (COMPUTE_FLOAT)0;
#ifdef USE_LOW_BIT_WEIGHT_INT8
        const int byteIndex = weightBase + i * 8 + oc_in8;
        char qw = weight[byteIndex];
        wVal = (COMPUTE_FLOAT)qw;
#elif defined(USE_LOW_BIT_WEIGHT_INT4)
        const int byteIndex = weightBase + i * 4 + (oc_in8 >> 1);
        uchar packed = weight[byteIndex];
        int nibble = (oc_in8 & 1) == 0 ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
#ifdef ASYMMETRIC
        wVal = (COMPUTE_FLOAT)nibble;
#else
        wVal = (COMPUTE_FLOAT)((int)nibble - 8);
#endif
#else
        const int byteIndex = weightBase + i * 8 + oc_in8;
        wVal = (COMPUTE_FLOAT)weight[byteIndex];
#endif

        COMPUTE_FLOAT v = mad(wVal, scale, offset);
        if (i == 0) {
            out4.s0 = v;
        } else if (i == 1) {
            out4.s1 = v;
        } else if (i == 2) {
            out4.s2 = v;
        } else {
            out4.s3 = v;
        }
    }

    OUTPUT_TYPE4 outVec = CONVERT_OUTPUT4(out4);
    if (base_ic + 3 < ic) {
        vstore4(outVec, 0, output + outBase);
    } else {
        OUTPUT_TYPE* outPtr = (OUTPUT_TYPE*)(&outVec);
        const int remain = ic - base_ic;
        for (int i = 0; i < remain; ++i) {
            output[outBase + i] = outPtr[i];
        }
    }
}

__kernel void shared_gather_quant_image(
    GLOBAL_SIZE_2_DIMS
    __global OUTPUT_TYPE* output,
    __read_only image2d_t weight,
    __global const int* indices,
    __global const FLOAT* dequantScaleOffset,
    __private const int ic,
    __private const int oc,
    __private const int blockSize,
    __private const float coef
) {
    const int select_idx = get_global_id(0);
    const int k4 = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(select_idx, k4);

    const int base_ic = k4 << 2;
    if (base_ic >= ic) {
        return;
    }

    const int ocIndex = indices[select_idx];
    if (ocIndex < 0 || ocIndex >= oc) {
        return;
    }

    const int out_c_idx = ocIndex >> 2;
    const int oc_in4 = ocIndex & 3;
    const int ocBlock = ocIndex >> 3;
    const int oc_in8 = ocIndex & 7;
    const int dstChannelC4 = ((oc + 3) >> 2) << 2;
    const int outBase = select_idx * ic + base_ic;
    COMPUTE_FLOAT4 out4 = (COMPUTE_FLOAT4)(0, 0, 0, 0);

#ifdef USE_LOW_BIT_WEIGHT_INT4
    const uchar16 weightBytes = as_uchar16(read_imagei(weight, SAMPLER, (int2)(k4, ocBlock)));
#endif

    for (int i = 0; i < 4; ++i) {
        const int icIndex = base_ic + i;
        if (icIndex >= ic) {
            break;
        }

        const int blockIndex = icIndex / blockSize;
        const int channelIndex = (out_c_idx << 2) + oc_in4;
        int scaleIndex = blockIndex * dstChannelC4 + channelIndex;

#ifdef ASYMMETRIC
        scaleIndex = scaleIndex * 2;
        FLOAT sRaw = dequantScaleOffset[scaleIndex + 0];
        FLOAT bRaw = dequantScaleOffset[scaleIndex + 1];
        COMPUTE_FLOAT scale = (COMPUTE_FLOAT)(convert_float(sRaw) / coef);
        COMPUTE_FLOAT offset = (COMPUTE_FLOAT)(convert_float(bRaw) / coef);
#else
        FLOAT sRaw = dequantScaleOffset[scaleIndex];
        COMPUTE_FLOAT scale = (COMPUTE_FLOAT)(convert_float(sRaw) / coef);
        COMPUTE_FLOAT offset = (COMPUTE_FLOAT)0;
#endif

        COMPUTE_FLOAT wVal = (COMPUTE_FLOAT)0;
#ifdef USE_LOW_BIT_WEIGHT_INT8
        const int imageX = (k4 << 1) + (i >> 1);
        const char16 weightBytes = as_char16(read_imagei(weight, SAMPLER, (int2)(imageX, ocBlock)));
        char qw = weightBytes[(i & 1) * 8 + oc_in8];
        wVal = (COMPUTE_FLOAT)qw;
#elif defined(USE_LOW_BIT_WEIGHT_INT4)
        uchar packed = weightBytes[i * 4 + (oc_in8 >> 1)];
        int nibble = (oc_in8 & 1) == 0 ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
#ifdef ASYMMETRIC
        wVal = (COMPUTE_FLOAT)nibble;
#else
        wVal = (COMPUTE_FLOAT)((int)nibble - 8);
#endif
#else
        const int imageX = (k4 << 1) + (i >> 1);
        const char16 weightBytes = as_char16(read_imagei(weight, SAMPLER, (int2)(imageX, ocBlock)));
        wVal = (COMPUTE_FLOAT)weightBytes[(i & 1) * 8 + oc_in8];
#endif

        COMPUTE_FLOAT v = mad(wVal, scale, offset);
        if (i == 0) {
            out4.s0 = v;
        } else if (i == 1) {
            out4.s1 = v;
        } else if (i == 2) {
            out4.s2 = v;
        } else {
            out4.s3 = v;
        }
    }

    OUTPUT_TYPE4 outVec = CONVERT_OUTPUT4(out4);
    if (base_ic + 3 < ic) {
        vstore4(outVec, 0, output + outBase);
    } else {
        OUTPUT_TYPE* outPtr = (OUTPUT_TYPE*)(&outVec);
        const int remain = ic - base_ic;
        for (int i = 0; i < remain; ++i) {
            output[outBase + i] = outPtr[i];
        }
    }
}
