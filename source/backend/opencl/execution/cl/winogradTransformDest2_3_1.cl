#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void winogradTransformDest(__read_only image2d_t uInput, // 0
                                    __read_only image2d_t uBias, __write_only image2d_t uOutput,
                                    __private const int unitWidth, // 3
                                    __private const int unitHeight, __private const int dstWidth,
                                    __private const int dstHeight, // 6
                                    __private const int dstChannelC4,__private const int batchOffset) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    if (pos.x < unitWidth*unitHeight && pos.y < dstChannelC4) {
        int unitWidth_idx = pos.x % unitWidth;
        int unitHeight_idx = pos.x / unitWidth;
        int srcY       = pos.y * unitHeight + unitHeight_idx;
        FLOAT4 bias    = RI_F(uBias, SAMPLER, (int2)(pos.y, 0));

        {
            int oyStart = unitHeight_idx * 2;
            int oxStart = unitWidth_idx * 2;
            FLOAT4 S00  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 0, srcY));
            FLOAT4 S10  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 1, srcY));
            FLOAT4 S20  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 2, srcY));
            FLOAT4 S30  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 3, srcY));
            FLOAT4 S01  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 4, srcY));
            FLOAT4 S11  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 5, srcY));
            FLOAT4 S21  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 6, srcY));
            FLOAT4 S31  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 7, srcY));
            FLOAT4 S02  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 8, srcY));
            FLOAT4 S12  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 9, srcY));
            FLOAT4 S22  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 10, srcY));
            FLOAT4 S32  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 11, srcY));
            FLOAT4 S03  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 12, srcY));
            FLOAT4 S13  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 13, srcY));
            FLOAT4 S23  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 14, srcY));
            FLOAT4 S33  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 15, srcY));
            FLOAT4 m00  = +S00 + S01 + S02;
            FLOAT4 m10  = +S10 + S11 + S12;
            FLOAT4 m20  = +S20 + S21 + S22;
            FLOAT4 m30  = +S30 + S31 + S32;
            FLOAT4 m01  = +S01 - S02 + S03;
            FLOAT4 m11  = +S11 - S12 + S13;
            FLOAT4 m21  = +S21 - S22 + S23;
            FLOAT4 m31  = +S31 - S32 + S33;
            {
                int ox = oxStart + 0;
                int oy = oyStart + 0;
                if (ox < dstWidth && oy < dstHeight) {
                    int imageOx = ox + pos.y * dstWidth;
                    int imageOy = oy + batchOffset * dstHeight;
                    FLOAT4 res  = bias + m00 + m10 + m20;
#ifdef RELU
                    res = max(res, (FLOAT4)(0));
#endif
#ifdef RELU6
                    res = clamp(res, (FLOAT4)(0), (FLOAT4)(6));
#endif
                    WI_F(uOutput, (int2)(imageOx, imageOy), res);
                }
            }
            {
                int ox = oxStart + 1;
                int oy = oyStart + 0;
                if (ox < dstWidth && oy < dstHeight) {
                    int imageOx = ox + pos.y * dstWidth;
                    int imageOy = oy + batchOffset * dstHeight;
                    FLOAT4 res  = bias + m10 - m20 + m30;
#ifdef RELU
                    res = max(res, (FLOAT4)(0));
#endif
#ifdef RELU6
                    res = clamp(res, (FLOAT4)(0), (FLOAT4)(6));
#endif
                    WI_F(uOutput, (int2)(imageOx, imageOy), res);
                }
            }
            {
                int ox = oxStart + 0;
                int oy = oyStart + 1;
                if (ox < dstWidth && oy < dstHeight) {
                    int imageOx = ox + pos.y * dstWidth;
                    int imageOy = oy + batchOffset * dstHeight;
                    FLOAT4 res  = bias + m01 + m11 + m21;
#ifdef RELU
                    res = max(res, (FLOAT4)(0));
#endif
#ifdef RELU6
                    res = clamp(res, (FLOAT4)(0), (FLOAT4)(6));
#endif
                    WI_F(uOutput, (int2)(imageOx, imageOy), res);
                }
            }
            {
                int ox = oxStart + 1;
                int oy = oyStart + 1;
                if (ox < dstWidth && oy < dstHeight) {
                    int imageOx = ox + pos.y * dstWidth;
                    int imageOy = oy + batchOffset * dstHeight;
                    FLOAT4 res  = bias + m11 - m21 + m31;
#ifdef RELU
                    res = max(res, (FLOAT4)(0));
#endif
#ifdef RELU6
                    res = clamp(res, (FLOAT4)(0), (FLOAT4)(6));
#endif
                    WI_F(uOutput, (int2)(imageOx, imageOy), res);
                }
            }
        }
    }
}
