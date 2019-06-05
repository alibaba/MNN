#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void winogradTransformSource(__read_only image2d_t uInput, // 0
                                      __write_only image2d_t uOutput, __private const int unitWidth,
                                      __private const int unitHeight, // 3
                                      __private const int padX, __private const int padY,
                                      __private const int srcWidth, // 6
                                      __private const int srcHeight, __private const int srcChannelC4,
                                      __private const int offsetX, // 9
                                      __private const int offsetY, __private const int batchOffset) {
    int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
    if (pos.x < unitWidth && pos.y < unitHeight) {
        int2 realPos   = (int2)(pos.x + offsetX, pos.y + offsetY);
        int dstXOrigin = pos.z;
        int batchIndex = pos.z / srcChannelC4;
        int srcZ       = pos.z % srcChannelC4;
        int dstYOrigin = unitWidth * pos.y + pos.x;
        int dstHeight  = (unitWidth * unitHeight + 3) / 4;
        int dstY       = dstYOrigin / 4;
        int dstX       = dstYOrigin % 4 + 4 * dstXOrigin;

        batchIndex = batchOffset;
        {
            int sxStart = (realPos.x) * 2 - padX;
            int syStart = (realPos.y) * 2 - padY;
            FLOAT4 S00;
            FLOAT4 S10;
            FLOAT4 S20;
            FLOAT4 S30;
            FLOAT4 S01;
            FLOAT4 S11;
            FLOAT4 S21;
            FLOAT4 S31;
            FLOAT4 S02;
            FLOAT4 S12;
            FLOAT4 S22;
            FLOAT4 S32;
            FLOAT4 S03;
            FLOAT4 S13;
            FLOAT4 S23;
            FLOAT4 S33;
            {
                int sx      = 0 + sxStart;
                int sy      = 0 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S00         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 1 + sxStart;
                int sy      = 0 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S10         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 2 + sxStart;
                int sy      = 0 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S20         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 3 + sxStart;
                int sy      = 0 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S30         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 0 + sxStart;
                int sy      = 1 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S01         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 1 + sxStart;
                int sy      = 1 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S11         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 2 + sxStart;
                int sy      = 1 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S21         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 3 + sxStart;
                int sy      = 1 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S31         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 0 + sxStart;
                int sy      = 2 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S02         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 1 + sxStart;
                int sy      = 2 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S12         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 2 + sxStart;
                int sy      = 2 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S22         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 3 + sxStart;
                int sy      = 2 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S32         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 0 + sxStart;
                int sy      = 3 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S03         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 1 + sxStart;
                int sy      = 3 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S13         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 2 + sxStart;
                int sy      = 3 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S23         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            {
                int sx      = 3 + sxStart;
                int sy      = 3 + syStart;
                int imageSx = select(sx + srcZ * srcWidth, -1, sx < 0 || sx >= srcWidth);
                int imageSy = select(batchIndex * srcHeight + sy, -1, sy < 0 || sy >= srcHeight);
                S33         = RI_F(uInput, SAMPLER, (int2)(imageSx, imageSy));
            }
            FLOAT4 m00 = +S00 - S02;
            FLOAT4 m10 = +S10 - S12;
            FLOAT4 m20 = +S20 - S22;
            FLOAT4 m30 = +S30 - S32;
            FLOAT4 m01 = +(FLOAT)0.5 * S01 + (FLOAT)0.5 * S02;
            FLOAT4 m11 = +(FLOAT)0.5 * S11 + (FLOAT)0.5 * S12;
            FLOAT4 m21 = +(FLOAT)0.5 * S21 + (FLOAT)0.5 * S22;
            FLOAT4 m31 = +(FLOAT)0.5 * S31 + (FLOAT)0.5 * S32;
            FLOAT4 m02 = -(FLOAT)0.5 * S01 + (FLOAT)0.5 * S02;
            FLOAT4 m12 = -(FLOAT)0.5 * S11 + (FLOAT)0.5 * S12;
            FLOAT4 m22 = -(FLOAT)0.5 * S21 + (FLOAT)0.5 * S22;
            FLOAT4 m32 = -(FLOAT)0.5 * S31 + (FLOAT)0.5 * S32;
            FLOAT4 m03 = -S01 + S03;
            FLOAT4 m13 = -S11 + S13;
            FLOAT4 m23 = -S21 + S23;
            FLOAT4 m33 = -S31 + S33;
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 0), +m00 - m20);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 1), +(FLOAT)0.5 * m10 + (FLOAT)0.5 * m20);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 2), -(FLOAT)0.5 * m10 + (FLOAT)0.5 * m20);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 3), -m10 + m30);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 4), +m01 - m21);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 5), +(FLOAT)0.5 * m11 + (FLOAT)0.5 * m21);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 6), -(FLOAT)0.5 * m11 + (FLOAT)0.5 * m21);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 7), -m11 + m31);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 8), +m02 - m22);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 9), +(FLOAT)0.5 * m12 + (FLOAT)0.5 * m22);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 10), -(FLOAT)0.5 * m12 + (FLOAT)0.5 * m22);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 11), -m12 + m32);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 12), +m03 - m23);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 13), +(FLOAT)0.5 * m13 + (FLOAT)0.5 * m23);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 14), -(FLOAT)0.5 * m13 + (FLOAT)0.5 * m23);
            WI_F(uOutput, (int2)(dstX, dstY + dstHeight * 15), -m13 + m33);
        }
    }
}
