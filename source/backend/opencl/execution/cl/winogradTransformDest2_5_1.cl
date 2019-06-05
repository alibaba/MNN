#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void winogradTransformDest(__read_only image2d_t uInput, // 0
                                    __read_only image2d_t uBias, __write_only image2d_t uOutput,
                                    __private const int unitWidth, // 3
                                    __private const int unitHeight, __private const int dstWidth,
                                    __private const int dstHeight, // 6
                                    __private const int dstChannelC4, __private const int offsetX,
                                    __private const int offsetY, __private const int batchOffset) {
    int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
    if (pos.x < unitWidth && pos.y < unitHeight) {
        int2 realPos   = (int2)(pos.x + offsetX, pos.y + offsetY);
        int srcWidth   = (unitWidth * unitHeight + 3) / 4;
        int dstXOrigin = unitWidth * pos.y + pos.x;
        int dstX       = dstXOrigin / 4;
        int dstY       = 4 * pos.z + dstXOrigin % 4;
        int oz         = pos.z % dstChannelC4;
        FLOAT4 bias    = RI_F(uBias, SAMPLER, (int2)(oz, 0));
        int batchIndex = pos.z / dstChannelC4;

        batchIndex = batchOffset;
        {
            int oyStart = realPos.y * 2;
            int oxStart = realPos.x * 2;
            FLOAT4 S00  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 0, dstY));
            FLOAT4 S10  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 1, dstY));
            FLOAT4 S20  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 2, dstY));
            FLOAT4 S30  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 3, dstY));
            FLOAT4 S40  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 4, dstY));
            FLOAT4 S50  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 5, dstY));
            FLOAT4 S01  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 6, dstY));
            FLOAT4 S11  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 7, dstY));
            FLOAT4 S21  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 8, dstY));
            FLOAT4 S31  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 9, dstY));
            FLOAT4 S41  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 10, dstY));
            FLOAT4 S51  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 11, dstY));
            FLOAT4 S02  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 12, dstY));
            FLOAT4 S12  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 13, dstY));
            FLOAT4 S22  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 14, dstY));
            FLOAT4 S32  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 15, dstY));
            FLOAT4 S42  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 16, dstY));
            FLOAT4 S52  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 17, dstY));
            FLOAT4 S03  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 18, dstY));
            FLOAT4 S13  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 19, dstY));
            FLOAT4 S23  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 20, dstY));
            FLOAT4 S33  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 21, dstY));
            FLOAT4 S43  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 22, dstY));
            FLOAT4 S53  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 23, dstY));
            FLOAT4 S04  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 24, dstY));
            FLOAT4 S14  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 25, dstY));
            FLOAT4 S24  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 26, dstY));
            FLOAT4 S34  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 27, dstY));
            FLOAT4 S44  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 28, dstY));
            FLOAT4 S54  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 29, dstY));
            FLOAT4 S05  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 30, dstY));
            FLOAT4 S15  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 31, dstY));
            FLOAT4 S25  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 32, dstY));
            FLOAT4 S35  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 33, dstY));
            FLOAT4 S45  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 34, dstY));
            FLOAT4 S55  = RI_F(uInput, SAMPLER, (int2)(dstX + srcWidth * 35, dstY));
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
                    int imageOx = ox + oz * dstWidth;
                    int imageOy = oy + batchIndex * dstHeight;
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
                    int imageOx = ox + oz * dstWidth;
                    int imageOy = oy + batchIndex * dstHeight;
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
                    int imageOx = ox + oz * dstWidth;
                    int imageOy = oy + batchIndex * dstHeight;
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
                    int imageOx = ox + oz * dstWidth;
                    int imageOy = oy + batchIndex * dstHeight;
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
