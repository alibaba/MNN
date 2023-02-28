#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void winogradTransformDest(__read_only image2d_t uInput, // 0
                                    __read_only image2d_t uBias, __write_only image2d_t uOutput,
                                    __private const int unitWidth, // 3
                                    __private const int unitHeight, __private const int dstWidth,
                                    __private const int dstHeight, // 6
                                    __private const int dstChannelC4, __private const int batchOffset) {
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
            FLOAT4 S40  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 4, srcY));
            FLOAT4 S50  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 5, srcY));
            FLOAT4 S01  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 6, srcY));
            FLOAT4 S11  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 7, srcY));
            FLOAT4 S21  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 8, srcY));
            FLOAT4 S31  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 9, srcY));
            FLOAT4 S41  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 10, srcY));
            FLOAT4 S51  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 11, srcY));
            FLOAT4 S02  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 12, srcY));
            FLOAT4 S12  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 13, srcY));
            FLOAT4 S22  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 14, srcY));
            FLOAT4 S32  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 15, srcY));
            FLOAT4 S42  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 16, srcY));
            FLOAT4 S52  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 17, srcY));
            FLOAT4 S03  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 18, srcY));
            FLOAT4 S13  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 19, srcY));
            FLOAT4 S23  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 20, srcY));
            FLOAT4 S33  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 21, srcY));
            FLOAT4 S43  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 22, srcY));
            FLOAT4 S53  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 23, srcY));
            FLOAT4 S04  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 24, srcY));
            FLOAT4 S14  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 25, srcY));
            FLOAT4 S24  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 26, srcY));
            FLOAT4 S34  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 27, srcY));
            FLOAT4 S44  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 28, srcY));
            FLOAT4 S54  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 29, srcY));
            FLOAT4 S05  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 30, srcY));
            FLOAT4 S15  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 31, srcY));
            FLOAT4 S25  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 32, srcY));
            FLOAT4 S35  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 33, srcY));
            FLOAT4 S45  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 34, srcY));
            FLOAT4 S55  = RI_F(uInput, SAMPLER, (int2)(unitWidth_idx + unitWidth * 35, srcY));
            FLOAT4 m00  = +S00 + S01 + S02 + S03 + S04;
            FLOAT4 m10  = +S10 + S11 + S12 + S13 + S14;
            FLOAT4 m20  = +S20 + S21 + S22 + S23 + S24;
            FLOAT4 m30  = +S30 + S31 + S32 + S33 + S34;
            FLOAT4 m40  = +S40 + S41 + S42 + S43 + S44;
            FLOAT4 m50  = +S50 + S51 + S52 + S53 + S54;
            FLOAT4 m01  = +S01 - S02 + (FLOAT)2.0 * S03 - (FLOAT)2.0 * S04 + S05;
            FLOAT4 m11  = +S11 - S12 + (FLOAT)2.0 * S13 - (FLOAT)2.0 * S14 + S15;
            FLOAT4 m21  = +S21 - S22 + (FLOAT)2.0 * S23 - (FLOAT)2.0 * S24 + S25;
            FLOAT4 m31  = +S31 - S32 + (FLOAT)2.0 * S33 - (FLOAT)2.0 * S34 + S35;
            FLOAT4 m41  = +S41 - S42 + (FLOAT)2.0 * S43 - (FLOAT)2.0 * S44 + S45;
            FLOAT4 m51  = +S51 - S52 + (FLOAT)2.0 * S53 - (FLOAT)2.0 * S54 + S55;
            {
                int ox = oxStart + 0;
                int oy = oyStart + 0;
                if (ox < dstWidth && oy < dstHeight) {
                    int imageOx = ox + pos.y * dstWidth;
                    int imageOy = oy + batchOffset * dstHeight;
                    FLOAT4 res  = bias + m00 + m10 + m20 + m30 + m40;
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
                    FLOAT4 res  = bias + m10 - m20 + (FLOAT)2.0 * m30 - (FLOAT)2.0 * m40 + m50;
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
                    FLOAT4 res  = bias + m01 + m11 + m21 + m31 + m41;
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
                    FLOAT4 res  = bias + m11 - m21 + (FLOAT4)2.0 * m31 - (FLOAT4)2.0 * m41 + m51;
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
