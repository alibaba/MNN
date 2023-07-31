#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_DIM2 \
    __private int global_size_dim0, __private int global_size_dim1,

#define UNIFORM_BOUNDRY_CHECK(index0, index1) \
    if(index0 >= global_size_dim0 || index1 >= global_size_dim1) { \
        return; \
    }
    
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void winoTransSrcBuf2_3_1_c16_c16(GLOBAL_SIZE_DIM2
                                      __global const FLOAT* uInput, // 0
                                      __global FLOAT* uOutput, __private const int unitWidth,
                                      __private const int unitHeight, // 3
                                      __private const int padX, __private const int padY,
                                      __private const int srcWidth, // 6
                                      __private const int srcHeight, __private const int srcChannelC4, __private const int srcChannelC16, __private const int dstHeight,
                                      __private const int batchOffset,
                                      __private const int input_pad_left, __private const int input_pad_right) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1)); 
    UNIFORM_BOUNDRY_CHECK(pos.x, pos.y);

    const int unitWidth_idx = pos.x % unitWidth;
    const int unitHeight_idx = pos.x / unitWidth;
    const int sglid = get_sub_group_local_id();
    const int pos_y = get_group_id(1);
    int2 realPos   = (int2)(unitWidth_idx, unitHeight_idx);
    int src_pitch = srcWidth + input_pad_left + input_pad_right;
    
    {
        int sxStart = (realPos.x) * 2 - padX;
        int syStart = (realPos.y) * 2 - padY;
        FLOAT4 S[4];
        
        int inp_offset = (((batchOffset * srcChannelC16 + pos_y) * srcHeight + syStart) * src_pitch + sxStart + input_pad_left) * 16;
        for(int i = 0; i < 4; ++i){
            int sy = i + syStart;
            if(sy < 0 || sy >= srcHeight){
                S[i] = (FLOAT4)0;
            }else{
#ifdef MNN_SUPPORT_FP16
                S[i] = as_half4(intel_sub_group_block_read_us4((__global ushort*)(uInput + inp_offset)));
#else
                S[i] = as_float4(intel_sub_group_block_read4((__global uint*)(uInput + inp_offset)));
#endif
            }
            inp_offset += 16*src_pitch;
        }
        FLOAT m00 = +S[0].s0 - S[2].s0;
        FLOAT m10 = +S[0].s1 - S[2].s1;
        FLOAT m20 = +S[0].s2 - S[2].s2;
        FLOAT m30 = +S[0].s3 - S[2].s3;
        FLOAT m01 = +(FLOAT)0.5f * S[1].s0 + (FLOAT)0.5f * S[2].s0;
        FLOAT m11 = +(FLOAT)0.5f * S[1].s1 + (FLOAT)0.5f * S[2].s1;
        FLOAT m21 = +(FLOAT)0.5f * S[1].s2 + (FLOAT)0.5f * S[2].s2;
        FLOAT m31 = +(FLOAT)0.5f * S[1].s3 + (FLOAT)0.5f * S[2].s3;
        FLOAT m02 = -(FLOAT)0.5f * S[1].s0 + (FLOAT)0.5f * S[2].s0;
        FLOAT m12 = -(FLOAT)0.5f * S[1].s1 + (FLOAT)0.5f * S[2].s1;
        FLOAT m22 = -(FLOAT)0.5f * S[1].s2 + (FLOAT)0.5f * S[2].s2;
        FLOAT m32 = -(FLOAT)0.5f * S[1].s3 + (FLOAT)0.5f * S[2].s3;
        FLOAT m03 = -S[1].s0 + S[3].s0;
        FLOAT m13 = -S[1].s1 + S[3].s1;
        FLOAT m23 = -S[1].s2 + S[3].s2;
        FLOAT m33 = -S[1].s3 + S[3].s3;
        
        //NC4HW4 [alpha*alpha, srcChannelC16, dstHeight, 16]
        //index: [0,           pos.y / 16,   pos.x,      0]
        int out_offset = (pos_y * dstHeight + pos.x) * 16 + sglid;
        int batch_offset = srcChannelC16*dstHeight*16;
        uOutput[out_offset+0*batch_offset]  = +m00 - m20;
        uOutput[out_offset+1*batch_offset]  = +(FLOAT)0.5f * m10 + (FLOAT)0.5f * m20;
        uOutput[out_offset+2*batch_offset]  = -(FLOAT)0.5f * m10 + (FLOAT)0.5f * m20;
        uOutput[out_offset+3*batch_offset]  = -m10 + m30;
        uOutput[out_offset+4*batch_offset]  = +m01 - m21;
        uOutput[out_offset+5*batch_offset]  = +(FLOAT)0.5f * m11 + (FLOAT)0.5f * m21;
        uOutput[out_offset+6*batch_offset]  = -(FLOAT)0.5f * m11 + (FLOAT)0.5f * m21;
        uOutput[out_offset+7*batch_offset]  = -m11 + m31;
        uOutput[out_offset+8*batch_offset]  = +m02 - m22;
        uOutput[out_offset+9*batch_offset]  = +(FLOAT)0.5f * m12 + (FLOAT)0.5f * m22;
        uOutput[out_offset+10*batch_offset] = -(FLOAT)0.5f * m12 + (FLOAT)0.5f * m22;
        uOutput[out_offset+11*batch_offset] = -m12 + m32;
        uOutput[out_offset+12*batch_offset] = +m03 - m23;
        uOutput[out_offset+13*batch_offset] = +(FLOAT)0.5f * m13 + (FLOAT)0.5f * m23;
        uOutput[out_offset+14*batch_offset] = -(FLOAT)0.5f * m13 + (FLOAT)0.5f * m23;
        uOutput[out_offset+15*batch_offset] = -m13 + m33;
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void winoTransDstBuf2_3_1_c16_c16(GLOBAL_SIZE_DIM2
                                    __global const FLOAT* uInput,
                                    __global const FLOAT* uBias,
                                    __global FLOAT* uOutput,
                                    __private const int unitWidth, //wUnit
                                    __private const int unitHeight, //hUnit
                                    __private const int dstWidth,
                                    __private const int dstHeight,
                                    __private const int dstChannelC4,__private const int dstChannelC16,__private const int srcWidth,
                                    __private const int batchOffset,
                                    __private const int output_pad_left, __private const int output_pad_right) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    UNIFORM_BOUNDRY_CHECK(pos.x, pos.y);

    const int unitWidth_idx = pos.x % unitWidth;
    const int unitHeight_idx = pos.x / unitWidth; 
    const int sglid = get_sub_group_local_id();
    const int pos_y = get_group_id(1);
    int2 realPos   = (int2)(unitWidth_idx, unitHeight_idx);
    
    FLOAT bias    = uBias[pos.y];

    {
        int oyStart = realPos.y * 2;
        int oxStart = realPos.x * 2;
        
        //NC4HW4 [alpha2, dstChannelC16, wUnit*hUnit, 16]
        //index: [0,        pos.y/4,      pos.x, pos.y%4]
        const int inp_offset = (pos_y * srcWidth + pos.x) * 16 + sglid;
        const int ic_offset = 16*srcWidth*dstChannelC16;
        FLOAT S00  = uInput[inp_offset+ic_offset*0];
        FLOAT S10  = uInput[inp_offset+ic_offset*1];
        FLOAT S20  = uInput[inp_offset+ic_offset*2];
        FLOAT S30  = uInput[inp_offset+ic_offset*3];
        FLOAT S01  = uInput[inp_offset+ic_offset*4];
        FLOAT S11  = uInput[inp_offset+ic_offset*5];
        FLOAT S21  = uInput[inp_offset+ic_offset*6];
        FLOAT S31  = uInput[inp_offset+ic_offset*7];
        FLOAT S02  = uInput[inp_offset+ic_offset*8];
        FLOAT S12  = uInput[inp_offset+ic_offset*9];
        FLOAT S22  = uInput[inp_offset+ic_offset*10];
        FLOAT S32  = uInput[inp_offset+ic_offset*11];
        FLOAT S03  = uInput[inp_offset+ic_offset*12];
        FLOAT S13  = uInput[inp_offset+ic_offset*13];
        FLOAT S23  = uInput[inp_offset+ic_offset*14];
        FLOAT S33  = uInput[inp_offset+ic_offset*15];

        FLOAT m00  = +S00 + S01 + S02;
        FLOAT m10  = +S10 + S11 + S12;
        FLOAT m20  = +S20 + S21 + S22;
        FLOAT m30  = +S30 + S31 + S32;
        FLOAT m01  = +S01 - S02 + S03;
        FLOAT m11  = +S11 - S12 + S13;
        FLOAT m21  = +S21 - S22 + S23;
        FLOAT m31  = +S31 - S32 + S33;
        
        //NC4HW4 [batch, dstChannelC4, dstHeight, dstWidth]
        //index: [batchOffset, pos.y,      oyStart,   oxStart]
        int dst_pitch = dstWidth + output_pad_left + output_pad_right;
        int out_offset = (((batchOffset * dstChannelC16+ pos_y) * dstHeight + oyStart) * dst_pitch + oxStart + output_pad_left)*16 + sglid;
        {
            FLOAT2 res  = (FLOAT2)(bias + m00 + m10 + m20, bias + m10 - m20 + m30);
#ifdef RELU
            res = max(res, (FLOAT2)0);
#endif
#ifdef RELU6
            res = clamp(res, (FLOAT2)0, (FLOAT2)6);
#endif

#if OUTPUT_LEFTOVERS
            uOutput[out_offset] = res.x;
            if(oxStart + 1<  dstWidth){
                uOutput[out_offset + 16] = res.y;
            }
#else
#ifdef MNN_SUPPORT_FP16
            intel_sub_group_block_write_us2((__global ushort*)(uOutput + out_offset), as_ushort2(res));
#else
            intel_sub_group_block_write2((__global uint*)(uOutput + out_offset), as_uint2(res));
#endif
#endif //OUTPUT_LEFTOVERS
        }
        {
            int oy = oyStart + 1;
            if (oy < dstHeight) {
                FLOAT2 res  = (FLOAT2)(bias + m01 + m11 + m21, bias + m11 - m21 + m31);
#ifdef RELU
                res = max(res, (FLOAT2)0);
#endif
#ifdef RELU6
                res = clamp(res, (FLOAT2)0, (FLOAT2)6);
#endif

#if OUTPUT_LEFTOVERS
                uOutput[out_offset+16*dst_pitch] = res.x;
                if(oxStart + 1<  dstWidth){
                    uOutput[out_offset + 16 + 16*dst_pitch] = res.y;
                }
#else
#ifdef MNN_SUPPORT_FP16
                intel_sub_group_block_write_us2((__global ushort*)(uOutput + out_offset+16*dst_pitch), as_ushort2(res));
#else
                intel_sub_group_block_write2((__global uint*)(uOutput + out_offset+16*dst_pitch), as_uint2(res));
#endif
#endif //OUTPUT_LEFTOVERS
            }
        }

        if(unitWidth_idx == 0){
            int pad_offset = (((batchOffset * dstChannelC16+ pos_y) * dstHeight + oyStart) * dst_pitch)*16 + sglid;
            for(int i = 0; i < output_pad_left; ++i){
                uOutput[pad_offset + i * 16] = 0;
                uOutput[pad_offset + (i + dst_pitch) * 16] = 0;
            }
        }
        if(unitWidth_idx == unitWidth - 1){
            int pad_offset = (((batchOffset * dstChannelC16+ pos_y) * dstHeight + oyStart) * dst_pitch + output_pad_left + dstWidth)*16 + sglid;
            for(int i = 0; i < output_pad_right; ++i){
                uOutput[pad_offset + i * 16] = 0;
                uOutput[pad_offset + (i + dst_pitch) * 16] = 0;
            }
        }
    }
}


__kernel void winoTransSrcBuf2_3_1_c4_c16(GLOBAL_SIZE_DIM2
                                      __global const FLOAT* uInput, // 0
                                      __global FLOAT* uOutput, __private const int unitWidth,
                                      __private const int unitHeight, // 3
                                      __private const int padX, __private const int padY,
                                      __private const int srcWidth, // 6
                                      __private const int srcHeight, __private const int srcChannelC4, __private const int srcChannelC16, __private const int dstHeight,
                                      __private const int batchOffset,
                                      __private const int input_pad_left, __private const int input_pad_right) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1)); 
    UNIFORM_BOUNDRY_CHECK(pos.x, pos.y);

    int unitWidth_idx = pos.x % unitWidth;
    int unitHeight_idx = pos.x / unitWidth;
    int2 realPos   = (int2)(unitWidth_idx, unitHeight_idx);
    
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
        
        int inp_offset = (((batchOffset * srcChannelC4 + pos.y) * srcHeight + syStart) * srcWidth + sxStart) * 4;
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
        
        //NC4HW4 [alpha*alpha, srcChannelC16, dstHeight, 16]
        //index: [0,           pos.y / 4,   pos.x,      pos.y % 4]
        int out_offset = ((pos.y / 4) * dstHeight + pos.x) * 16 + (pos.y % 4) * 4;
        int batch_offset = srcChannelC16*dstHeight*16;
        vstore4(+m00 - m20,                             0, uOutput+out_offset+0*batch_offset);
        vstore4(+(FLOAT)0.5f * m10 + (FLOAT)0.5f * m20, 0, uOutput+out_offset+1*batch_offset);
        vstore4(-(FLOAT)0.5f * m10 + (FLOAT)0.5f * m20, 0, uOutput+out_offset+2*batch_offset);
        vstore4(-m10 + m30,                             0, uOutput+out_offset+3*batch_offset);
        vstore4(+m01 - m21,                             0, uOutput+out_offset+4*batch_offset);
        vstore4(+(FLOAT)0.5f * m11 + (FLOAT)0.5f * m21, 0, uOutput+out_offset+5*batch_offset);
        vstore4(-(FLOAT)0.5f * m11 + (FLOAT)0.5f * m21, 0, uOutput+out_offset+6*batch_offset);
        vstore4(-m11 + m31,                             0, uOutput+out_offset+7*batch_offset);
        vstore4(+m02 - m22,                             0, uOutput+out_offset+8*batch_offset);
        vstore4(+(FLOAT)0.5f * m12 + (FLOAT)0.5f * m22, 0, uOutput+out_offset+9*batch_offset);
        vstore4(-(FLOAT)0.5f * m12 + (FLOAT)0.5f * m22, 0, uOutput+out_offset+10*batch_offset);
        vstore4(-m12 + m32,                             0, uOutput+out_offset+11*batch_offset);
        vstore4(+m03 - m23,                             0, uOutput+out_offset+12*batch_offset);
        vstore4(+(FLOAT)0.5f * m13 + (FLOAT)0.5f * m23, 0, uOutput+out_offset+13*batch_offset);
        vstore4(-(FLOAT)0.5f * m13 + (FLOAT)0.5f * m23, 0, uOutput+out_offset+14*batch_offset);
        vstore4(-m13 + m33,                             0, uOutput+out_offset+15*batch_offset);
    }
}


__kernel void winoTransDstBuf2_3_1_c16_c4(GLOBAL_SIZE_DIM2
                                    __global const FLOAT* uInput,
                                    __global const FLOAT* uBias,
                                    __global FLOAT* uOutput,
                                    __private const int unitWidth, //wUnit
                                    __private const int unitHeight, //hUnit
                                    __private const int dstWidth,
                                    __private const int dstHeight,
                                    __private const int dstChannelC4,__private const int dstChannelC16,__private const int srcWidth,
                                    __private const int batchOffset,
                                    __private const int output_pad_left, __private const int output_pad_right) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    UNIFORM_BOUNDRY_CHECK(pos.x, pos.y);

    int unitWidth_idx = pos.x % unitWidth;
    int unitHeight_idx = pos.x / unitWidth;
    int2 realPos   = (int2)(unitWidth_idx, unitHeight_idx);
    
    FLOAT4 bias    = vload4(0, uBias+pos.y*4);

    {
        int oyStart = realPos.y * 2;
        int oxStart = realPos.x * 2;
        
        //NC4HW4 [alpha2, dstChannelC16, wUnit*hUnit, 16]
        //index: [0,        pos.y/4,      pos.x, pos.y%4]
        const int inp_offset = ((pos.y / 4) * srcWidth + pos.x) * 16 + (pos.y % 4) * 4;
        const int ic_offset = 16*srcWidth*dstChannelC16;
        FLOAT4 S00  = vload4(0, uInput+inp_offset+ic_offset*0);
        FLOAT4 S10  = vload4(0, uInput+inp_offset+ic_offset*1);
        FLOAT4 S20  = vload4(0, uInput+inp_offset+ic_offset*2);
        FLOAT4 S30  = vload4(0, uInput+inp_offset+ic_offset*3);
        FLOAT4 S01  = vload4(0, uInput+inp_offset+ic_offset*4);
        FLOAT4 S11  = vload4(0, uInput+inp_offset+ic_offset*5);
        FLOAT4 S21  = vload4(0, uInput+inp_offset+ic_offset*6);
        FLOAT4 S31  = vload4(0, uInput+inp_offset+ic_offset*7);
        FLOAT4 S02  = vload4(0, uInput+inp_offset+ic_offset*8);
        FLOAT4 S12  = vload4(0, uInput+inp_offset+ic_offset*9);
        FLOAT4 S22  = vload4(0, uInput+inp_offset+ic_offset*10);
        FLOAT4 S32  = vload4(0, uInput+inp_offset+ic_offset*11);
        FLOAT4 S03  = vload4(0, uInput+inp_offset+ic_offset*12);
        FLOAT4 S13  = vload4(0, uInput+inp_offset+ic_offset*13);
        FLOAT4 S23  = vload4(0, uInput+inp_offset+ic_offset*14);
        FLOAT4 S33  = vload4(0, uInput+inp_offset+ic_offset*15);

        FLOAT4 m00  = +S00 + S01 + S02;
        FLOAT4 m10  = +S10 + S11 + S12;
        FLOAT4 m20  = +S20 + S21 + S22;
        FLOAT4 m30  = +S30 + S31 + S32;
        FLOAT4 m01  = +S01 - S02 + S03;
        FLOAT4 m11  = +S11 - S12 + S13;
        FLOAT4 m21  = +S21 - S22 + S23;
        FLOAT4 m31  = +S31 - S32 + S33;
        
        //NC4HW4 [batch, dstChannelC4, dstHeight, dstWidth]
        //index: [batchOffset, pos.y,      oyStart,   oxStart]
        int out_offset = (((batchOffset * dstChannelC4+ pos.y) * dstHeight + oyStart) * dstWidth + oxStart)*4;
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


__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemm_buf_intel(__global const FLOAT* input0,
                             __global const FLOAT* input1,
                             __global FLOAT* output,
                             __private const int width,//ROUND_UP(wUnit*hUnit, 8)
                             __private const int height,//dstChannelC16
                             __private const int srcChannelC16,
                             __private const int alpha2) {
    int3 pos = (int3)(get_global_id(0), get_group_id(1), get_global_id(2));
    const int sglid = get_sub_group_local_id();
    const int pos_x = pos.x << 3;
    const int pos_y = pos.y;

    FLOAT8 o = (FLOAT8)(0);
    const int kernel_base  = mul24(mul24(mad24(pos.z, height, pos_y), srcChannelC16), 256);
    const int inp_base = mul24(mad24(mul24(pos.z, srcChannelC16), width, pos_x), 16);
     
     for(int k = 0; k < srcChannelC16; ++k){
     
#ifdef MNN_SUPPORT_FP16
        FLOAT8 wei0 = as_half8(intel_sub_group_block_read_us8((__global ushort*)(input1 + kernel_base + k * 256)));
        FLOAT8 wei1 = as_half8(intel_sub_group_block_read_us8((__global ushort*)(input1 + kernel_base + k * 256 + 8 * 16)));

        FLOAT8 s = as_half8(intel_sub_group_block_read_us8((__global ushort*)(input0 + inp_base + k * width * 16)));
        o = mad(wei0.s0, as_half8(intel_sub_group_shuffle(as_ushort8(s),  0)), o);
        o = mad(wei0.s1, as_half8(intel_sub_group_shuffle(as_ushort8(s),  1)), o);
        o = mad(wei0.s2, as_half8(intel_sub_group_shuffle(as_ushort8(s),  2)), o);
        o = mad(wei0.s3, as_half8(intel_sub_group_shuffle(as_ushort8(s),  3)), o);
        o = mad(wei0.s4, as_half8(intel_sub_group_shuffle(as_ushort8(s),  4)), o);
        o = mad(wei0.s5, as_half8(intel_sub_group_shuffle(as_ushort8(s),  5)), o);
        o = mad(wei0.s6, as_half8(intel_sub_group_shuffle(as_ushort8(s),  6)), o);
        o = mad(wei0.s7, as_half8(intel_sub_group_shuffle(as_ushort8(s),  7)), o);
        o = mad(wei1.s0, as_half8(intel_sub_group_shuffle(as_ushort8(s),  8)), o);
        o = mad(wei1.s1, as_half8(intel_sub_group_shuffle(as_ushort8(s),  9)), o);
        o = mad(wei1.s2, as_half8(intel_sub_group_shuffle(as_ushort8(s), 10)), o);
        o = mad(wei1.s3, as_half8(intel_sub_group_shuffle(as_ushort8(s), 11)), o);
        o = mad(wei1.s4, as_half8(intel_sub_group_shuffle(as_ushort8(s), 12)), o);
        o = mad(wei1.s5, as_half8(intel_sub_group_shuffle(as_ushort8(s), 13)), o);
        o = mad(wei1.s6, as_half8(intel_sub_group_shuffle(as_ushort8(s), 14)), o);
        o = mad(wei1.s7, as_half8(intel_sub_group_shuffle(as_ushort8(s), 15)), o);

#else        
        FLOAT8 wei0 = as_float8(intel_sub_group_block_read8((__global uint*)(input1 + kernel_base + k * 256)));
        FLOAT8 wei1 = as_float8(intel_sub_group_block_read8((__global uint*)(input1 + kernel_base + k * 256 + 8 * 16)));

        FLOAT8 s = as_float8(intel_sub_group_block_read8((__global uint*)(input0 + inp_base + k * width * 16)));
        o = mad(wei0.s0, intel_sub_group_shuffle(s,  0), o);
        o = mad(wei0.s1, intel_sub_group_shuffle(s,  1), o);
        o = mad(wei0.s2, intel_sub_group_shuffle(s,  2), o);
        o = mad(wei0.s3, intel_sub_group_shuffle(s,  3), o);
        o = mad(wei0.s4, intel_sub_group_shuffle(s,  4), o);
        o = mad(wei0.s5, intel_sub_group_shuffle(s,  5), o);
        o = mad(wei0.s6, intel_sub_group_shuffle(s,  6), o);
        o = mad(wei0.s7, intel_sub_group_shuffle(s,  7), o);
        o = mad(wei1.s0, intel_sub_group_shuffle(s,  8), o);
        o = mad(wei1.s1, intel_sub_group_shuffle(s,  9), o);
        o = mad(wei1.s2, intel_sub_group_shuffle(s, 10), o);
        o = mad(wei1.s3, intel_sub_group_shuffle(s, 11), o);
        o = mad(wei1.s4, intel_sub_group_shuffle(s, 12), o);
        o = mad(wei1.s5, intel_sub_group_shuffle(s, 13), o);
        o = mad(wei1.s6, intel_sub_group_shuffle(s, 14), o);
        o = mad(wei1.s7, intel_sub_group_shuffle(s, 15), o);
#endif 
     }

    int out_offset = mul24(mad24(mad24(pos.z, height, pos_y), width, pos_x), 16);
#ifdef MNN_SUPPORT_FP16
    intel_sub_group_block_write_us8((__global ushort*)(output + out_offset), as_ushort8(o));
#else
    intel_sub_group_block_write8((__global uint*)(output + out_offset), as_uint8(o));
#endif
}
