#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_DIM2 \
    __private int global_size_dim0, __private int global_size_dim1,

#define UNIFORM_BOUNDRY_CHECK(index0, index1) \
    if(index0 >= global_size_dim0 || index1 >= global_size_dim1) { \
        return; \
    }

// [dstChannel, srcChannel, 3, 3] -> [4x4, srcChannelPad, dstChannelpad] (N, Kpad, Npad)
__kernel void winoTransWeightBuf2_3_1(GLOBAL_SIZE_DIM2
                              __global const float* input, // 0
                              __global FLOAT* output,
                              __private const int srcChannel, // 3
                              __private const int dstChannel,
                              __private const int srcChannelPad, // 6
                              __private const int dstChannelPad
) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    UNIFORM_BOUNDRY_CHECK(pos.x, pos.y);
    
    const int src_c = pos.x;
    const int dst_c = pos.y;
    
    const int out_offset = (0 * srcChannelPad + src_c) * dstChannelPad + dst_c;
    const int out_offset_add = srcChannelPad * dstChannelPad;
    if(src_c >= srcChannel || dst_c >= dstChannel) {
        for(int i = 0; i < 16; i++) {
            output[out_offset + i * out_offset_add] = (FLOAT)0;
        }
        return;
    }
    
    const int in_offset = (dst_c * srcChannel + src_c) * 9;
    FLOAT8 in = CONVERT_FLOAT8(vload8(0, input + in_offset));
    FLOAT in8 = input[in_offset+8];
    
    FLOAT GB_00 = in.s0;
    FLOAT GB_01 = in.s1;
    FLOAT GB_02 = in.s2;
    FLOAT GB_10 = in.s0 + in.s3 + in.s6;
    FLOAT GB_11 = in.s1 + in.s4 + in.s7;
    FLOAT GB_12 = in.s2 + in.s5 + in8;
    FLOAT GB_20 = in.s0 - in.s3 + in.s6;
    FLOAT GB_21 = in.s1 - in.s4 + in.s7;
    FLOAT GB_22 = in.s2 - in.s5 + in8;
    FLOAT GB_30 = in.s6;
    FLOAT GB_31 = in.s7;
    FLOAT GB_32 = in8;
    
    FLOAT GBGT_00 = GB_00;
    FLOAT GBGT_01 = GB_00 + GB_01  + GB_02;
    FLOAT GBGT_02 = GB_00 - GB_01  + GB_02;
    FLOAT GBGT_03 = GB_02;
    
    FLOAT GBGT_10 = GB_10;
    FLOAT GBGT_11 = GB_10 + GB_11  + GB_12;
    FLOAT GBGT_12 = GB_10 - GB_11  + GB_12;
    FLOAT GBGT_13 = GB_12;
    
    FLOAT GBGT_20 = GB_20;
    FLOAT GBGT_21 = GB_20 + GB_21  + GB_22;
    FLOAT GBGT_22 = GB_20 - GB_21  + GB_22;
    FLOAT GBGT_23 = GB_22;
    
    FLOAT GBGT_30 = GB_30;
    FLOAT GBGT_31 = GB_30 + GB_31  + GB_32;
    FLOAT GBGT_32 = GB_30 - GB_31  + GB_32;
    FLOAT GBGT_33 = GB_32;

    output[out_offset + 0 * out_offset_add] = GBGT_00;
    output[out_offset + 1 * out_offset_add] = GBGT_01;
    output[out_offset + 2 * out_offset_add] = GBGT_02;
    output[out_offset + 3 * out_offset_add] = GBGT_03;
    output[out_offset + 4 * out_offset_add] = GBGT_10;
    output[out_offset + 5 * out_offset_add] = GBGT_11;
    output[out_offset + 6 * out_offset_add] = GBGT_12;
    output[out_offset + 7 * out_offset_add] = GBGT_13;
    output[out_offset + 8 * out_offset_add] = GBGT_20;
    output[out_offset + 9 * out_offset_add] = GBGT_21;
    output[out_offset + 10 * out_offset_add] = GBGT_22;
    output[out_offset + 11 * out_offset_add] = GBGT_23;
    output[out_offset + 12 * out_offset_add] = GBGT_30;
    output[out_offset + 13 * out_offset_add] = GBGT_31;
    output[out_offset + 14 * out_offset_add] = GBGT_32;
    output[out_offset + 15 * out_offset_add] = GBGT_33;
}

__kernel void winoTransSrcBuf2_3_1(GLOBAL_SIZE_DIM2
                                      __global const FLOAT* uInput, // 0
                                      __global FLOAT* uOutput, __private const int unitWidth,
                                      __private const int unitHeight, // 3
                                      __private const int padX, __private const int padY,
                                      __private const int srcWidth, // 6
                                      __private const int srcHeight, __private const int srcChannelC4,
                                      __private const int dstHeightPad, __private const int srcChannelPad,
                                      __private const int batchOffset) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1)); 
    UNIFORM_BOUNDRY_CHECK(pos.x, pos.y);
    
    if(pos.x >= unitWidth * unitHeight || pos.y >= srcChannelC4) {
        return;
    }
    int unitWidth_idx = pos.x % unitWidth;
    int unitHeight_idx = pos.x / unitWidth;
    int2 realPos   = (int2)(unitWidth_idx, unitHeight_idx);
    int dstXOrigin = pos.y;
    int batchIndex = pos.y / srcChannelC4;
    int srcZ       = pos.y % srcChannelC4;
    int dstYOrigin = unitWidth * unitHeight_idx + unitWidth_idx;

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
        
        int inp_offset = (((batchIndex * srcChannelC4 + srcZ) * srcHeight + syStart) * srcWidth + sxStart) * 4;
        {
            int sx      = 0 + sxStart;
            int sy      = 0 + syStart;
            
            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S00         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset);
        }
        {
            int sx      = 1 + sxStart;
            int sy      = 0 + syStart;
            
            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S10         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+4);
        }
        {
            int sx      = 2 + sxStart;
            int sy      = 0 + syStart;
            
            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S20         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+8);
        }
        {
            int sx      = 3 + sxStart;
            int sy      = 0 + syStart;

            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S30         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+12);
        }
        {
            int sx      = 0 + sxStart;
            int sy      = 1 + syStart;

            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S01         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+4*srcWidth);
        }
        {
            int sx      = 1 + sxStart;
            int sy      = 1 + syStart;

            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S11         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+4*srcWidth+4);
        }
        {
            int sx      = 2 + sxStart;
            int sy      = 1 + syStart;

            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S21         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+4*srcWidth+8);
        }
        {
            int sx      = 3 + sxStart;
            int sy      = 1 + syStart;

            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S31         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+4*srcWidth+12);
        }
        {
            int sx      = 0 + sxStart;
            int sy      = 2 + syStart;

            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S02         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+8*srcWidth);
        }
        {
            int sx      = 1 + sxStart;
            int sy      = 2 + syStart;

            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S12         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+8*srcWidth+4);
        }
        {
            int sx      = 2 + sxStart;
            int sy      = 2 + syStart;

            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S22         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+8*srcWidth+8);
        }
        {
            int sx      = 3 + sxStart;
            int sy      = 2 + syStart;

            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S32         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+8*srcWidth+12);
        }
        {
            int sx      = 0 + sxStart;
            int sy      = 3 + syStart;

            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S03         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+12*srcWidth);
        }
        {
            int sx      = 1 + sxStart;
            int sy      = 3 + syStart;

            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S13         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+12*srcWidth+4);
        }
        {
            int sx      = 2 + sxStart;
            int sy      = 3 + syStart;

            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S23         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+12*srcWidth+8);
        }
        {
            int sx      = 3 + sxStart;
            int sy      = 3 + syStart;

            bool outBound = (sx < 0 || sx >= srcWidth || sy < 0 || sy >= srcHeight);
            S33         = outBound ? (FLOAT4)(0) : vload4(0, uInput+inp_offset+12*srcWidth+12);
        }
        FLOAT4 m00 = +S00 - S02;
        FLOAT4 m10 = +S10 - S12;
        FLOAT4 m20 = +S20 - S22;
        FLOAT4 m30 = +S30 - S32;
        FLOAT4 m01 = +(FLOAT)0.5f * S01 + (FLOAT)0.5f * S02;
        FLOAT4 m11 = +(FLOAT)0.5f * S11 + (FLOAT)0.5f * S12;
        FLOAT4 m21 = +(FLOAT)0.5f * S21 + (FLOAT)0.5f * S22;
        FLOAT4 m31 = +(FLOAT)0.5f * S31 + (FLOAT)0.5f * S32;
        FLOAT4 m02 = -(FLOAT)0.5f * S01 + (FLOAT)0.5f * S02;
        FLOAT4 m12 = -(FLOAT)0.5f * S11 + (FLOAT)0.5f * S12;
        FLOAT4 m22 = -(FLOAT)0.5f * S21 + (FLOAT)0.5f * S22;
        FLOAT4 m32 = -(FLOAT)0.5f * S31 + (FLOAT)0.5f * S32;
        FLOAT4 m03 = -S01 + S03;
        FLOAT4 m13 = -S11 + S13;
        FLOAT4 m23 = -S21 + S23;
        FLOAT4 m33 = -S31 + S33;
        
        //NC4HW4 [alpha*alpha, srcChannelPad, dstHeightPad]
        //index: [0,           dstXOrigin,   dstY,      dstYOrigin % 4]

        int out_offset = (0*srcChannelPad + 4*dstXOrigin) * dstHeightPad + dstYOrigin;
        int batch_offset = srcChannelPad*dstHeightPad;
        
        FLOAT4 res = (+m00 - m20);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;

        out_offset += batch_offset;
        res = (+(FLOAT)0.5f * m10 + (FLOAT)0.5f * m20);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;
        
        out_offset += batch_offset;
        res = (-(FLOAT)0.5f * m10 + (FLOAT)0.5f * m20);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;
        
        out_offset += batch_offset;
        res = (-m10 + m30);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;
        
        
        out_offset += batch_offset;
        res = (+m01 - m21);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;
        
        out_offset += batch_offset;
        res = (+(FLOAT)0.5f * m11 + (FLOAT)0.5f * m21);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;
        
        out_offset += batch_offset;
        res = (-(FLOAT)0.5f * m11 + (FLOAT)0.5f * m21);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;
        
        out_offset += batch_offset;
        res = (-m11 + m31);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;
        
        out_offset += batch_offset;
        res = (+m02 - m22);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;
        
        out_offset += batch_offset;
        res = (+(FLOAT)0.5f * m12 + (FLOAT)0.5f * m22);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;
        
        out_offset += batch_offset;
        res = (-(FLOAT)0.5f * m12 + (FLOAT)0.5f * m22);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;
        
        out_offset += batch_offset;
        res = (-m12 + m32);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;
        
        out_offset += batch_offset;
        res = (+m03 - m23);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;
        
        out_offset += batch_offset;
        res = (+(FLOAT)0.5f * m13 + (FLOAT)0.5f * m23);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;
        
        out_offset += batch_offset;
        res = (-(FLOAT)0.5f * m13 + (FLOAT)0.5f * m23);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;
        
        out_offset += batch_offset;
        res = (-m13 + m33);
        uOutput[out_offset] = res.x;
        uOutput[out_offset + dstHeightPad] = res.y;
        uOutput[out_offset + dstHeightPad + dstHeightPad] = res.z;
        uOutput[out_offset + dstHeightPad + dstHeightPad + dstHeightPad] = res.w;
    }
}


__kernel void winoTransDstBuf2_3_1(GLOBAL_SIZE_DIM2
                                    __global const FLOAT* uInput,
                                    __global const FLOAT* uBias,
                                    __global FLOAT* uOutput,
                                    __private const int unitWidth, //wUnit
                                    __private const int unitHeight, //hUnit
                                    __private const int dstWidth,
                                    __private const int dstHeight,
                                    __private const int dstChannelC4,
                                    __private const int srcWidthPad,
                                    __private const int dstChannelPad,
                                    __private const int batchOffset) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    UNIFORM_BOUNDRY_CHECK(pos.x, pos.y);

    int unitWidth_idx = pos.x % unitWidth;
    int unitHeight_idx = pos.x / unitWidth;
    int2 realPos   = (int2)(unitWidth_idx, unitHeight_idx);
    int dstXOrigin = unitWidth * unitHeight_idx + unitWidth_idx;
    int oz         = pos.y % dstChannelC4;
    
    FLOAT4 bias    = vload4(0, uBias+oz*4);
    int batchIndex = pos.y / dstChannelC4;

    batchIndex = batchOffset;
    {
        int oyStart = realPos.y * 2;
        int oxStart = realPos.x * 2;
        
        // [alpha2, srcWidthPad, dstChannelPad]
        //index: [0, dstXOrigin, 4*oz]

        const int inp_offset = (0 * srcWidthPad + dstXOrigin) * dstChannelPad + 4*oz;
        const int b_offset = dstChannelPad*srcWidthPad;

        FLOAT4 S00  = vload4(0, uInput+inp_offset+b_offset*0);
        FLOAT4 S10  = vload4(0, uInput+inp_offset+b_offset*1);
        FLOAT4 S20  = vload4(0, uInput+inp_offset+b_offset*2);
        FLOAT4 S30  = vload4(0, uInput+inp_offset+b_offset*3);
        FLOAT4 S01  = vload4(0, uInput+inp_offset+b_offset*4);
        FLOAT4 S11  = vload4(0, uInput+inp_offset+b_offset*5);
        FLOAT4 S21  = vload4(0, uInput+inp_offset+b_offset*6);
        FLOAT4 S31  = vload4(0, uInput+inp_offset+b_offset*7);
        FLOAT4 S02  = vload4(0, uInput+inp_offset+b_offset*8);
        FLOAT4 S12  = vload4(0, uInput+inp_offset+b_offset*9);
        FLOAT4 S22  = vload4(0, uInput+inp_offset+b_offset*10);
        FLOAT4 S32  = vload4(0, uInput+inp_offset+b_offset*11);
        FLOAT4 S03  = vload4(0, uInput+inp_offset+b_offset*12);
        FLOAT4 S13  = vload4(0, uInput+inp_offset+b_offset*13);
        FLOAT4 S23  = vload4(0, uInput+inp_offset+b_offset*14);
        FLOAT4 S33  = vload4(0, uInput+inp_offset+b_offset*15);

        FLOAT4 m00  = +S00 + S01 + S02;
        FLOAT4 m10  = +S10 + S11 + S12;
        FLOAT4 m20  = +S20 + S21 + S22;
        FLOAT4 m30  = +S30 + S31 + S32;
        FLOAT4 m01  = +S01 - S02 + S03;
        FLOAT4 m11  = +S11 - S12 + S13;
        FLOAT4 m21  = +S21 - S22 + S23;
        FLOAT4 m31  = +S31 - S32 + S33;
        
        //NC4HW4 [batch, dstChannelC4, dstHeight, dstWidth]
        //index: [batchIndex, oz,      oyStart,   oxStart]
        int out_offset = (((batchIndex * dstChannelC4+ oz) * dstHeight + oyStart) * dstWidth + oxStart)*4;
        {
            int ox = oxStart + 0;
            int oy = oyStart + 0;
            if (ox < dstWidth && oy < dstHeight) {
                FLOAT4 res  = bias + m00 + m10 + m20;
#ifdef RELU
                res = max(res, (FLOAT4)(0));
#endif
#ifdef RELU6
                res = clamp(res, (FLOAT4)(0), (FLOAT4)(6));
#endif
                vstore4(res, 0, uOutput+out_offset);
            }
        }
        {
            int ox = oxStart + 1;
            int oy = oyStart + 0;
            if (ox < dstWidth && oy < dstHeight) {
                FLOAT4 res  = bias + m10 - m20 + m30;
#ifdef RELU
                res = max(res, (FLOAT4)(0));
#endif
#ifdef RELU6
                res = clamp(res, (FLOAT4)(0), (FLOAT4)(6));
#endif
                vstore4(res, 0, uOutput+out_offset+4);
            }
        }
        {
            int ox = oxStart + 0;
            int oy = oyStart + 1;
            if (ox < dstWidth && oy < dstHeight) {
                FLOAT4 res  = bias + m01 + m11 + m21;
#ifdef RELU
                res = max(res, (FLOAT4)(0));
#endif
#ifdef RELU6
                res = clamp(res, (FLOAT4)(0), (FLOAT4)(6));
#endif
                vstore4(res, 0, uOutput+out_offset+4*dstWidth);
            }
        }
        {
            int ox = oxStart + 1;
            int oy = oyStart + 1;
            if (ox < dstWidth && oy < dstHeight) {
                FLOAT4 res  = bias + m11 - m21 + m31;
#ifdef RELU
                res = max(res, (FLOAT4)(0));
#endif
#ifdef RELU6
                res = clamp(res, (FLOAT4)(0), (FLOAT4)(6));
#endif
                vstore4(res, 0, uOutput+out_offset+4*dstWidth+4);
            }
        }
    }
}
