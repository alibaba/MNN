#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void winogradTransformSource(__read_only image2d_t uInput, // 0
                                      __write_only image2d_t uOutput, __private const int unitWidth,
                                      __private const int unitHeight, // 3
                                      __private const int padX, __private const int padY,
                                      __private const int srcWidth, // 6
                                      __private const int srcHeight, __private const int srcChannelC4,
                                      __private const int batchOffset) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1)); 
    if (pos.x < unitWidth*unitHeight && pos.y < srcChannelC4) {
        int unitWidth_idx = pos.x % unitWidth;
        int unitHeight_idx = pos.x / unitWidth;
        int dstX       = mad24(pos.y, unitWidth, unitWidth_idx);

        {
            int sxStart = (unitWidth_idx) * 2 - padX;
            int syStart = (unitHeight_idx) * 2 - padY;
            FLOAT4 S00;
            FLOAT4 S10;
            FLOAT4 S20;
            FLOAT4 S30;
            FLOAT4 S40;
            FLOAT4 S50;
            FLOAT4 S01;
            FLOAT4 S11;
            FLOAT4 S21;
            FLOAT4 S31;
            FLOAT4 S41;
            FLOAT4 S51;
            FLOAT4 S02;
            FLOAT4 S12;
            FLOAT4 S22;
            FLOAT4 S32;
            FLOAT4 S42;
            FLOAT4 S52;
            FLOAT4 S03;
            FLOAT4 S13;
            FLOAT4 S23;
            FLOAT4 S33;
            FLOAT4 S43;
            FLOAT4 S53;
            FLOAT4 S04;
            FLOAT4 S14;
            FLOAT4 S24;
            FLOAT4 S34;
            FLOAT4 S44;
            FLOAT4 S54;
            FLOAT4 S05;
            FLOAT4 S15;
            FLOAT4 S25;
            FLOAT4 S35;
            FLOAT4 S45;
            FLOAT4 S55;
            {
                int sx      = 0 + sxStart;
                int sy      = 0 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S00         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 1 + sxStart;
                int sy      = 0 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S10         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 2 + sxStart;
                int sy      = 0 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S20         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 3 + sxStart;
                int sy      = 0 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S30         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 4 + sxStart;
                int sy      = 0 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S40         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 5 + sxStart;
                int sy      = 0 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S50         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 0 + sxStart;
                int sy      = 1 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S01         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 1 + sxStart;
                int sy      = 1 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S11         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 2 + sxStart;
                int sy      = 1 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S21         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 3 + sxStart;
                int sy      = 1 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S31         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 4 + sxStart;
                int sy      = 1 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S41         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 5 + sxStart;
                int sy      = 1 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S51         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 0 + sxStart;
                int sy      = 2 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S02         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 1 + sxStart;
                int sy      = 2 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S12         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 2 + sxStart;
                int sy      = 2 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S22         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 3 + sxStart;
                int sy      = 2 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S32         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 4 + sxStart;
                int sy      = 2 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S42         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 5 + sxStart;
                int sy      = 2 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S52         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 0 + sxStart;
                int sy      = 3 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S03         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 1 + sxStart;
                int sy      = 3 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S13         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 2 + sxStart;
                int sy      = 3 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S23         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 3 + sxStart;
                int sy      = 3 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S33         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 4 + sxStart;
                int sy      = 3 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S43         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 5 + sxStart;
                int sy      = 3 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S53         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 0 + sxStart;
                int sy      = 4 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S04         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 1 + sxStart;
                int sy      = 4 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S14         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 2 + sxStart;
                int sy      = 4 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S24         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 3 + sxStart;
                int sy      = 4 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S34         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 4 + sxStart;
                int sy      = 4 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S44         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 5 + sxStart;
                int sy      = 4 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S54         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 0 + sxStart;
                int sy      = 5 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S05         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 1 + sxStart;
                int sy      = 5 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S15         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 2 + sxStart;
                int sy      = 5 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S25         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 3 + sxStart;
                int sy      = 5 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S35         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 4 + sxStart;
                int sy      = 5 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S45         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 5 + sxStart;
                int sy      = 5 + syStart;
                int imageSx = select(sx + pos.y * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchOffset * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S55         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            FLOAT4 m00 = +S00 - (FLOAT)1.25 * S02 + (FLOAT)0.25 * S04;
            FLOAT4 m10 = +S10 - (FLOAT)1.25 * S12 + (FLOAT)0.25 * S14;
            FLOAT4 m20 = +S20 - (FLOAT)1.25 * S22 + (FLOAT)0.25 * S24;
            FLOAT4 m30 = +S30 - (FLOAT)1.25 * S32 + (FLOAT)0.25 * S34;
            FLOAT4 m40 = +S40 - (FLOAT)1.25 * S42 + (FLOAT)0.25 * S44;
            FLOAT4 m50 = +S50 - (FLOAT)1.25 * S52 + (FLOAT)0.25 * S54;
            FLOAT4 m01 = +(FLOAT)0.666667 * S01 + (FLOAT)0.666667 * S02 - (FLOAT)0.166667 * S03 - (FLOAT)0.166667 * S04;
            FLOAT4 m11 = +(FLOAT)0.666667 * S11 + (FLOAT)0.666667 * S12 - (FLOAT)0.166667 * S13 - (FLOAT)0.166667 * S14;
            FLOAT4 m21 = +(FLOAT)0.666667 * S21 + (FLOAT)0.666667 * S22 - (FLOAT)0.166667 * S23 - (FLOAT)0.166667 * S24;
            FLOAT4 m31 = +(FLOAT)0.666667 * S31 + (FLOAT)0.666667 * S32 - (FLOAT)0.166667 * S33 - (FLOAT)0.166667 * S34;
            FLOAT4 m41 = +(FLOAT)0.666667 * S41 + (FLOAT)0.666667 * S42 - (FLOAT)0.166667 * S43 - (FLOAT)0.166667 * S44;
            FLOAT4 m51 = +(FLOAT)0.666667 * S51 + (FLOAT)0.666667 * S52 - (FLOAT)0.166667 * S53 - (FLOAT)0.166667 * S54;
            FLOAT4 m02 = -(FLOAT)0.666667 * S01 + (FLOAT)0.666667 * S02 + (FLOAT)0.166667 * S03 - (FLOAT)0.166667 * S04;
            FLOAT4 m12 = -(FLOAT)0.666667 * S11 + (FLOAT)0.666667 * S12 + (FLOAT)0.166667 * S13 - (FLOAT)0.166667 * S14;
            FLOAT4 m22 = -(FLOAT)0.666667 * S21 + (FLOAT)0.666667 * S22 + (FLOAT)0.166667 * S23 - (FLOAT)0.166667 * S24;
            FLOAT4 m32 = -(FLOAT)0.666667 * S31 + (FLOAT)0.666667 * S32 + (FLOAT)0.166667 * S33 - (FLOAT)0.166667 * S34;
            FLOAT4 m42 = -(FLOAT)0.666667 * S41 + (FLOAT)0.666667 * S42 + (FLOAT)0.166667 * S43 - (FLOAT)0.166667 * S44;
            FLOAT4 m52 = -(FLOAT)0.666667 * S51 + (FLOAT)0.666667 * S52 + (FLOAT)0.166667 * S53 - (FLOAT)0.166667 * S54;
            FLOAT4 m03 =
                -(FLOAT)0.0833333 * S01 - (FLOAT)0.0416667 * S02 + (FLOAT)0.0833333 * S03 + (FLOAT)0.0416667 * S04;
            FLOAT4 m13 =
                -(FLOAT)0.0833333 * S11 - (FLOAT)0.0416667 * S12 + (FLOAT)0.0833333 * S13 + (FLOAT)0.0416667 * S14;
            FLOAT4 m23 =
                -(FLOAT)0.0833333 * S21 - (FLOAT)0.0416667 * S22 + (FLOAT)0.0833333 * S23 + (FLOAT)0.0416667 * S24;
            FLOAT4 m33 =
                -(FLOAT)0.0833333 * S31 - (FLOAT)0.0416667 * S32 + (FLOAT)0.0833333 * S33 + (FLOAT)0.0416667 * S34;
            FLOAT4 m43 =
                -(FLOAT)0.0833333 * S41 - (FLOAT)0.0416667 * S42 + (FLOAT)0.0833333 * S43 + (FLOAT)0.0416667 * S44;
            FLOAT4 m53 =
                -(FLOAT)0.0833333 * S51 - (FLOAT)0.0416667 * S52 + (FLOAT)0.0833333 * S53 + (FLOAT)0.0416667 * S54;
            FLOAT4 m04 =
                +(FLOAT)0.0833333 * S01 - (FLOAT)0.0416667 * S02 - (FLOAT)0.0833333 * S03 + (FLOAT)0.0416667 * S04;
            FLOAT4 m14 =
                +(FLOAT)0.0833333 * S11 - (FLOAT)0.0416667 * S12 - (FLOAT)0.0833333 * S13 + (FLOAT)0.0416667 * S14;
            FLOAT4 m24 =
                +(FLOAT)0.0833333 * S21 - (FLOAT)0.0416667 * S22 - (FLOAT)0.0833333 * S23 + (FLOAT)0.0416667 * S24;
            FLOAT4 m34 =
                +(FLOAT)0.0833333 * S31 - (FLOAT)0.0416667 * S32 - (FLOAT)0.0833333 * S33 + (FLOAT)0.0416667 * S34;
            FLOAT4 m44 =
                +(FLOAT)0.0833333 * S41 - (FLOAT)0.0416667 * S42 - (FLOAT)0.0833333 * S43 + (FLOAT)0.0416667 * S44;
            FLOAT4 m54 =
                +(FLOAT)0.0833333 * S51 - (FLOAT)0.0416667 * S52 - (FLOAT)0.0833333 * S53 + (FLOAT)0.0416667 * S54;
            FLOAT4 m05 = +(FLOAT)4.0 * S01 - (FLOAT)5.0 * S03 + S05;
            FLOAT4 m15 = +(FLOAT)4.0 * S11 - (FLOAT)5.0 * S13 + S15;
            FLOAT4 m25 = +(FLOAT)4.0 * S21 - (FLOAT)5.0 * S23 + S25;
            FLOAT4 m35 = +(FLOAT)4.0 * S31 - (FLOAT)5.0 * S33 + S35;
            FLOAT4 m45 = +(FLOAT)4.0 * S41 - (FLOAT)5.0 * S43 + S45;
            FLOAT4 m55 = +(FLOAT)4.0 * S51 - (FLOAT)5.0 * S53 + S55;
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 0), +m00 - (FLOAT)1.25 * m20 + (FLOAT)0.25 * m40);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 1),
                 +(FLOAT)0.666667 * m10 + (FLOAT)0.666667 * m20 - (FLOAT)0.166667 * m30 - (FLOAT)0.166667 * m40);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 2),
                 -(FLOAT)0.666667 * m10 + (FLOAT)0.666667 * m20 + (FLOAT)0.166667 * m30 - (FLOAT)0.166667 * m40);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 3),
                 -(FLOAT)0.0833333 * m10 - (FLOAT)0.0416667 * m20 + (FLOAT)0.0833333 * m30 + (FLOAT)0.0416667 * m40);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 4),
                 +(FLOAT)0.0833333 * m10 - (FLOAT)0.0416667 * m20 - (FLOAT)0.0833333 * m30 + (FLOAT)0.0416667 * m40);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 5), +(FLOAT)4.0 * m10 - (FLOAT)5.0 * m30 + m50);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 6), +m01 - (FLOAT)1.25 * m21 + (FLOAT)0.25 * m41);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 7),
                 +(FLOAT)0.666667 * m11 + (FLOAT)0.666667 * m21 - (FLOAT)0.166667 * m31 - (FLOAT)0.166667 * m41);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 8),
                 -(FLOAT)0.666667 * m11 + (FLOAT)0.666667 * m21 + (FLOAT)0.166667 * m31 - (FLOAT)0.166667 * m41);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 9),
                 -(FLOAT)0.0833333 * m11 - (FLOAT)0.0416667 * m21 + (FLOAT)0.0833333 * m31 + (FLOAT)0.0416667 * m41);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 10),
                 +(FLOAT)0.0833333 * m11 - (FLOAT)0.0416667 * m21 - (FLOAT)0.0833333 * m31 + (FLOAT)0.0416667 * m41);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 11), +(FLOAT)4.0 * m11 - (FLOAT)5.0 * m31 + m51);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 12), +m02 - (FLOAT)1.25 * m22 + (FLOAT)0.25 * m42);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 13),
                 +(FLOAT)0.666667 * m12 + (FLOAT)0.666667 * m22 - (FLOAT)0.166667 * m32 - (FLOAT)0.166667 * m42);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 14),
                 -(FLOAT)0.666667 * m12 + (FLOAT)0.666667 * m22 + (FLOAT)0.166667 * m32 - (FLOAT)0.166667 * m42);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 15),
                 -(FLOAT)0.0833333 * m12 - (FLOAT)0.0416667 * m22 + (FLOAT)0.0833333 * m32 + (FLOAT)0.0416667 * m42);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 16),
                 +(FLOAT)0.0833333 * m12 - (FLOAT)0.0416667 * m22 - (FLOAT)0.0833333 * m32 + (FLOAT)0.0416667 * m42);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 17), +(FLOAT)4.0 * m12 - (FLOAT)5.0 * m32 + m52);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 18), +m03 - (FLOAT)1.25 * m23 + (FLOAT)0.25 * m43);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 19),
                 +(FLOAT)0.666667 * m13 + (FLOAT)0.666667 * m23 - (FLOAT)0.166667 * m33 - (FLOAT)0.166667 * m43);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 20),
                 -(FLOAT)0.666667 * m13 + (FLOAT)0.666667 * m23 + (FLOAT)0.166667 * m33 - (FLOAT)0.166667 * m43);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 21),
                 -(FLOAT)0.0833333 * m13 - (FLOAT)0.0416667 * m23 + (FLOAT)0.0833333 * m33 + (FLOAT)0.0416667 * m43);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 22),
                 +(FLOAT)0.0833333 * m13 - (FLOAT)0.0416667 * m23 - (FLOAT)0.0833333 * m33 + (FLOAT)0.0416667 * m43);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 23), +(FLOAT)4.0 * m13 - (FLOAT)5.0 * m33 + m53);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 24), +m04 - (FLOAT)1.25 * m24 + (FLOAT)0.25 * m44);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 25),
                 +(FLOAT)0.666667 * m14 + (FLOAT)0.666667 * m24 - (FLOAT)0.166667 * m34 - (FLOAT)0.166667 * m44);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 26),
                 -(FLOAT)0.666667 * m14 + (FLOAT)0.666667 * m24 + (FLOAT)0.166667 * m34 - (FLOAT)0.166667 * m44);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 27),
                 -(FLOAT)0.0833333 * m14 - (FLOAT)0.0416667 * m24 + (FLOAT)0.0833333 * m34 + (FLOAT)0.0416667 * m44);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 28),
                 +(FLOAT)0.0833333 * m14 - (FLOAT)0.0416667 * m24 - (FLOAT)0.0833333 * m34 + (FLOAT)0.0416667 * m44);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 29), +(FLOAT)4.0 * m14 - (FLOAT)5.0 * m34 + m54);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 30), +m05 - (FLOAT)1.25 * m25 + (FLOAT)0.25 * m45);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 31),
                 +(FLOAT)0.666667 * m15 + (FLOAT)0.666667 * m25 - (FLOAT)0.166667 * m35 - (FLOAT)0.166667 * m45);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 32),
                 -(FLOAT)0.666667 * m15 + (FLOAT)0.666667 * m25 + (FLOAT)0.166667 * m35 - (FLOAT)0.166667 * m45);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 33),
                 -(FLOAT)0.0833333 * m15 - (FLOAT)0.0416667 * m25 + (FLOAT)0.0833333 * m35 + (FLOAT)0.0416667 * m45);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 34),
                 +(FLOAT)0.0833333 * m15 - (FLOAT)0.0416667 * m25 - (FLOAT)0.0833333 * m35 + (FLOAT)0.0416667 * m45);
            WI_F(uOutput, (int2)(dstX, unitHeight_idx + unitHeight * 35), +(FLOAT)4.0 * m15 - (FLOAT)5.0 * m35 + m55);
        }
    }
}
