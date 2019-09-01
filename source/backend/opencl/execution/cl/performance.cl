
#define MAD_V4(x, y)  \
    x = mad(y, x, y); \
    y = mad(x, y, x); \
    x = mad(y, x, y); \
    y = mad(x, y, x);
#define MAD_V16(x, y) \
    MAD_V4(x, y);     \
    MAD_V4(x, y);     \
    MAD_V4(x, y);     \
    MAD_V4(x, y);
#define MAD_V64(x, y) \
    MAD_V16(x, y);    \
    MAD_V16(x, y);    \
    MAD_V16(x, y);    \
    MAD_V16(x, y);
#define MAD_V128(x, y) \
    MAD_V64(x, y);     \
    MAD_V64(x, y);     \
    MAD_V64(x, y);     \
    MAD_V64(x, y);
#define MAD_V256(x, y) \
    MAD_V128(x, y);    \
    MAD_V128(x, y);    \
    MAD_V128(x, y);    \
    MAD_V128(x, y);

#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__kernel void float_precision(__global float* output_ptr, float mul_value) {
    float mul_x = mul_value;
    float mul_y = (float)get_local_id(0);

    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);

    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    output_ptr[get_global_id(0)] = mul_y;
}

__kernel void half4_precision(__global half* output_ptr, float mul_value) {
    half mul    = (half)mul_value;
    half4 mul_x = (half4)(mul);
    half4 mul_y = (half4)get_local_id(0);

    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);
    MAD_V256(mul_x, mul_y);

    output_ptr[get_global_id(0)] = (mul_y.S0) + (mul_y.S1) + (mul_y.S2) + (mul_y.S3);
}
