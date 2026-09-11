#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_DIM2 \
    __private int global_size_dim0, __private int global_size_dim1,

#define UNIFORM_BOUNDRY_CHECK(index0, index1) \
    if(index0 >= global_size_dim0 || index1 >= global_size_dim1) { \
        return; \
    }

#define UCHAR4_TO_CHAR8(a, c) \
    a.s0 = (c.s0 >> 4); a.s1 = (c.s0 & 15); a.s2 = (c.s1 >> 4); a.s3 = (c.s1 & 15); a.s4 = (c.s2 >> 4); a.s5 = (c.s2 & 15); a.s6 = (c.s3 >> 4); a.s7 = (c.s3 & 15);

// 2bit unpack: 2 packed bytes (b0,b1) → 8 chars, unsigned [0,3].
// Origin offset is folded into the dequant `offset` (or asym bias) by the caller.
#define UCHAR2_TO_CHAR8_W2(a, b0, b1) \
    a.s0 = (char)((b0 >> 6) & 3); \
    a.s1 = (char)((b0 >> 4) & 3); \
    a.s2 = (char)((b0 >> 2) & 3); \
    a.s3 = (char)((b0 >> 0) & 3); \
    a.s4 = (char)((b1 >> 6) & 3); \
    a.s5 = (char)((b1 >> 4) & 3); \
    a.s6 = (char)((b1 >> 2) & 3); \
    a.s7 = (char)((b1 >> 0) & 3);

// 3bit unpack: 2 low bytes + 1 high byte → 8 chars, unsigned [0,7].
// Origin offset is folded into the dequant `offset` (or asym bias) by the caller.
#define UCHAR3_TO_CHAR8_W3(a, b0, b1, h) \
    a.s0 = (char)(((b0 >> 6) & 3) | (((h >> 7) & 1) << 2)); \
    a.s1 = (char)(((b0 >> 4) & 3) | (((h >> 6) & 1) << 2)); \
    a.s2 = (char)(((b0 >> 2) & 3) | (((h >> 5) & 1) << 2)); \
    a.s3 = (char)(((b0 >> 0) & 3) | (((h >> 4) & 1) << 2)); \
    a.s4 = (char)(((b1 >> 6) & 3) | (((h >> 3) & 1) << 2)); \
    a.s5 = (char)(((b1 >> 4) & 3) | (((h >> 2) & 1) << 2)); \
    a.s6 = (char)(((b1 >> 2) & 3) | (((h >> 1) & 1) << 2)); \
    a.s7 = (char)(((b1 >> 0) & 3) | (((h >> 0) & 1) << 2));

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void inverse_quant_weight(GLOBAL_SIZE_DIM2
    #ifdef USE_IMAGE
    __read_only image2d_t weight,
    #else
    #if QUANT_BIT == 8
    __global const char *weight,
    #else
    __global const uchar *weight,
    #endif
    #endif
    __global const FLOAT *dequantScaleOffset,
    __global FLOAT* output,
    __private const int inputChannel,
    __private const int inputChannel4Align,
    __private const int outputChannelAlign,
    __private const int outputChannel4Align,
    __private const int blockDim,
    __private const float coef){
    const int x = get_global_id(0); //ic
    const int y = get_global_id(1); //oc

    UNIFORM_BOUNDRY_CHECK(x, y);
    
#if QUANT_BIT == 2 || QUANT_BIT == 3 || QUANT_BIT == 4
    const int ic = x << 2;
    const int oc = y << 3;
    const int output_offset = ic * outputChannelAlign + oc;

    #ifdef ASYMMETRIC
    COMPUTE_FLOAT8 scale, offset;
    {
        COMPUTE_FLOAT16 ScaleOffset = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + ((ic / blockDim) * outputChannel4Align + oc) * 2)) / coef);
        scale = ScaleOffset.s02468ace;
        offset = ScaleOffset.s13579bdf;
    }
    #else
    COMPUTE_FLOAT8 scale = CONVERT_COMPUTE_FLOAT8(convert_float8(vload8(0, dequantScaleOffset + (ic / blockDim) * outputChannel4Align + oc)) / coef);
    #if QUANT_BIT == 2
    COMPUTE_FLOAT8 offset = (COMPUTE_FLOAT8)(-2) * scale;
    #elif QUANT_BIT == 3
    COMPUTE_FLOAT8 offset = (COMPUTE_FLOAT8)(-4) * scale;
    #else
    COMPUTE_FLOAT8 offset = (COMPUTE_FLOAT8)(-8) * scale;
    #endif
    #endif
    COMPUTE_FLOAT8 weights0, weights1, weights2, weights3;
    {
#if QUANT_BIT == 4
        #ifdef USE_IMAGE
        uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(x, y)));
        #else
        uchar16 charWeightsInt40 = vload16(x, weight + y * inputChannel4Align * 4);
        #endif
        char8 charWeights0;
        UCHAR4_TO_CHAR8(charWeights0, charWeightsInt40.s0123);
        weights0 = CONVERT_COMPUTE_FLOAT8(charWeights0) * scale + offset;
        UCHAR4_TO_CHAR8(charWeights0, charWeightsInt40.s4567);
        weights1 = ic + 1 >= inputChannel ? 0 : CONVERT_COMPUTE_FLOAT8(charWeights0) * scale + offset;
        UCHAR4_TO_CHAR8(charWeights0, charWeightsInt40.s89ab);
        weights2 = ic + 2 >= inputChannel ? 0 : CONVERT_COMPUTE_FLOAT8(charWeights0) * scale + offset;
        UCHAR4_TO_CHAR8(charWeights0, charWeightsInt40.scdef);
        weights3 = ic + 3 >= inputChannel ? 0 : CONVERT_COMPUTE_FLOAT8(charWeights0) * scale + offset;
#elif QUANT_BIT == 2
        // 2bit storage: 8 bytes per (4IC × 8OC) tile, weight_offset row stride = inputChannel4Align * 2
        uchar8 charWeightsInt20 = vload8(x, weight + y * inputChannel4Align * 2);
        char8 charWeights0;
        UCHAR2_TO_CHAR8_W2(charWeights0, charWeightsInt20.s0, charWeightsInt20.s1);
        weights0 = CONVERT_COMPUTE_FLOAT8(charWeights0) * scale + offset;
        UCHAR2_TO_CHAR8_W2(charWeights0, charWeightsInt20.s2, charWeightsInt20.s3);
        weights1 = ic + 1 >= inputChannel ? 0 : CONVERT_COMPUTE_FLOAT8(charWeights0) * scale + offset;
        UCHAR2_TO_CHAR8_W2(charWeights0, charWeightsInt20.s4, charWeightsInt20.s5);
        weights2 = ic + 2 >= inputChannel ? 0 : CONVERT_COMPUTE_FLOAT8(charWeights0) * scale + offset;
        UCHAR2_TO_CHAR8_W2(charWeights0, charWeightsInt20.s6, charWeightsInt20.s7);
        weights3 = ic + 3 >= inputChannel ? 0 : CONVERT_COMPUTE_FLOAT8(charWeights0) * scale + offset;
#elif QUANT_BIT == 3
        // 3bit storage: 12 bytes per (4IC × 8OC) tile, row stride = inputChannel4Align * 3
        const int base = y * inputChannel4Align * 3 + x * 12;
        uchar8 charWeightsInt30Lo = vload8(0, weight + base);
        uchar4 charWeightsInt30Hi = vload4(0, weight + base + 8);
        char8 charWeights0;
        UCHAR3_TO_CHAR8_W3(charWeights0, charWeightsInt30Lo.s0, charWeightsInt30Lo.s1, charWeightsInt30Hi.s0);
        weights0 = CONVERT_COMPUTE_FLOAT8(charWeights0) * scale + offset;
        UCHAR3_TO_CHAR8_W3(charWeights0, charWeightsInt30Lo.s2, charWeightsInt30Lo.s3, charWeightsInt30Hi.s1);
        weights1 = ic + 1 >= inputChannel ? 0 : CONVERT_COMPUTE_FLOAT8(charWeights0) * scale + offset;
        UCHAR3_TO_CHAR8_W3(charWeights0, charWeightsInt30Lo.s4, charWeightsInt30Lo.s5, charWeightsInt30Hi.s2);
        weights2 = ic + 2 >= inputChannel ? 0 : CONVERT_COMPUTE_FLOAT8(charWeights0) * scale + offset;
        UCHAR3_TO_CHAR8_W3(charWeights0, charWeightsInt30Lo.s6, charWeightsInt30Lo.s7, charWeightsInt30Hi.s3);
        weights3 = ic + 3 >= inputChannel ? 0 : CONVERT_COMPUTE_FLOAT8(charWeights0) * scale + offset;
#endif
    }
    vstore8(CONVERT_FLOAT8(weights0), 0, output+output_offset);
    vstore8(CONVERT_FLOAT8(weights1), 0, output+output_offset+outputChannelAlign);
    vstore8(CONVERT_FLOAT8(weights2), 0, output+output_offset+2*outputChannelAlign);
    vstore8(CONVERT_FLOAT8(weights3), 0, output+output_offset+3*outputChannelAlign);
#elif QUANT_BIT == 8
    const int ic = x << 1;
    const int oc = y << 3;
    const int output_offset = ic * outputChannelAlign + oc;
    
    #ifdef ASYMMETRIC
    COMPUTE_FLOAT8 scale, offset;
    {
        COMPUTE_FLOAT16 ScaleOffset = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + ((ic / blockDim) * outputChannel4Align + oc) * 2)) / coef);
        scale = ScaleOffset.s02468ace;
        offset = ScaleOffset.s13579bdf;
    }
    #else
    COMPUTE_FLOAT8 scale = CONVERT_COMPUTE_FLOAT8(convert_float8(vload8(0, dequantScaleOffset + (ic / blockDim) * outputChannel4Align + oc)) / coef);
    #endif
    COMPUTE_FLOAT8 weights0, weights1;
    {
        #ifdef USE_IMAGE
        COMPUTE_FLOAT16 wei = CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight, SAMPLER, (int2)(x, y))));
        #else
        COMPUTE_FLOAT16 wei = CONVERT_COMPUTE_FLOAT16(vload16(x, weight + y * inputChannel4Align * 8));
        #endif
        #ifdef ASYMMETRIC
        weights0 = wei.s01234567 * scale + offset;
        weights1 = ic + 1 >= inputChannel ? 0 : wei.s89abcdef * scale + offset;
        #else
        weights0 = wei.s01234567 * scale;
        weights1 = ic + 1 >= inputChannel ? 0 : wei.s89abcdef * scale;
        #endif
    }
    vstore8(CONVERT_FLOAT8(weights0), 0, output+output_offset);
    vstore8(CONVERT_FLOAT8(weights1), 0, output+output_offset+outputChannelAlign);
    #endif
}
