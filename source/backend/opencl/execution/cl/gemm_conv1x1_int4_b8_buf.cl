#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_DIM2 \
    __private int global_size_dim0, __private int global_size_dim1,

#define UNIFORM_BOUNDRY_CHECK(index0, index1) \
    if(index0 >= global_size_dim0 || index1 >= global_size_dim1) { \
        return; \
    }
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#ifdef USE_IMAGE1D_INPUT
#ifdef MNN_SUPPORT_FP16
#define RI_F(i, coord) read_imageh(i, coord)
#else
#define RI_F(i, coord) read_imagef(i, coord)
#endif
#define LOAD_INPUT4(offset) CONVERT_COMPUTE_FLOAT4(RI_F(input, (offset) >> 2))
#define LOAD_INPUT16(offset) (COMPUTE_FLOAT16)(LOAD_INPUT4(offset), LOAD_INPUT4((offset) + 4), LOAD_INPUT4((offset) + 8), LOAD_INPUT4((offset) + 12))
#else
#define LOAD_INPUT4(offset) CONVERT_COMPUTE_FLOAT4(vload4(0, input + (offset)))
#define LOAD_INPUT16(offset) CONVERT_COMPUTE_FLOAT16(vload16(0, input + (offset)))
#endif
#define UCHAR4_TO_FLOAT8(b, scale, offset) \
    wei.s0 = CONVERT_FLOAT((b.s0 >> 4)); \
    wei.s1 = CONVERT_FLOAT((b.s0 & 15)); \
    wei.s2 = CONVERT_FLOAT((b.s1 >> 4)); \
    wei.s3 = CONVERT_FLOAT((b.s1 & 15)); \
    wei.s4 = CONVERT_FLOAT((b.s2 >> 4)); \
    wei.s5 = CONVERT_FLOAT((b.s2 & 15)); \
    wei.s6 = CONVERT_FLOAT((b.s3 >> 4)); \
    wei.s7 = CONVERT_FLOAT((b.s3 & 15)); \
    wei = wei * scale + offset;
__kernel void gemm_b8_c8_int4_buf(GLOBAL_SIZE_DIM2
#ifdef USE_IMAGE1D_INPUT
                        __read_only image1d_buffer_t input,
#else
                        __global const FLOAT* input,
#endif
#ifdef USE_IMAGE
                        __read_only image2d_t weight,
#else
                        __global const uchar *weight,
#endif
                        __global const FLOAT *dequantScaleOffset,
                        __global const FLOAT *bias,
                        __global FLOAT* output,
                        __private const int bhw,
                        __private const int dstChannelAlign,
                        __private const int srcChannelAlign,
                        __private const int blockNum,
                        __private const int blockDim,
                        __private const float coef) {
    const int x = get_global_id(0); //b/8
    const int y = get_global_id(1); //c/8

    UNIFORM_BOUNDRY_CHECK(x, y);
    
    const int out_b_idx = x << 3;  // 8 batches per work-item
    const int out_c_idx = y << 1;  // 8 channels (2 * float4)

    COMPUTE_FLOAT8 out0 = CONVERT_COMPUTE_FLOAT8(vload8(0, bias + (out_c_idx << 2)));
    COMPUTE_FLOAT8 out1 = out0;
    COMPUTE_FLOAT8 out2 = out0;
    COMPUTE_FLOAT8 out3 = out0;
    COMPUTE_FLOAT8 out4 = out0;
    COMPUTE_FLOAT8 out5 = out0;
    COMPUTE_FLOAT8 out6 = out0;
    COMPUTE_FLOAT8 out7 = out0;
    
    const int bhw4 = bhw << 2;
    const int input_offset = out_b_idx * 4;
    int out_offset = out_c_idx * bhw4 + out_b_idx * 4;
#ifndef USE_IMAGE
    const int weight_offset = y * srcChannelAlign * 4;
#endif
    const int loop = (blockDim + 4 - 1) / 4;
#if INPUT_CHANNEL_LEAVES_NUM != 0
    const int loop_end = max(loop - 1, 0);
#else
    const int loop_end = loop;
#endif

#if INPUT_BATCH_LEAVES_NUM != 0
    if(out_b_idx + 7 >= bhw){
        // Tail path: handle remaining 1-7 batches
        for (int i = 0; i < blockNum; i++){
            #ifdef ASYMMETRIC
            COMPUTE_FLOAT8 scale, offset;
            {
                COMPUTE_FLOAT16 scaleOffset = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + (out_c_idx << 3) + i * dstChannelAlign * 2)) / coef);
                scale = scaleOffset.s02468ace;
                offset = scaleOffset.s13579bdf;
            }
            #else
            COMPUTE_FLOAT8 scale = CONVERT_COMPUTE_FLOAT8(convert_float8(vload8(0, dequantScaleOffset + (out_c_idx << 2) + i * dstChannelAlign)) / coef);
            COMPUTE_FLOAT8 offset = (COMPUTE_FLOAT8)(-8) * scale;
            #endif
            for (int j = 0; j < loop_end; j++) {
                int k = i * loop + j;
                COMPUTE_FLOAT8 wei;
                #ifdef USE_IMAGE
                uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(k, y)));
                #else
                uchar16 charWeightsInt40 = vload16(k, weight + weight_offset);
                #endif
                
#if INPUT_BATCH_LEAVES_NUM >= 4
                // First 4 batches are always valid
                COMPUTE_FLOAT16 inA = LOAD_INPUT16(input_offset + k * bhw4);
                {
                    UCHAR4_TO_FLOAT8(charWeightsInt40.s0123, scale, offset);
                    out0 = mad((COMPUTE_FLOAT8)inA.s0, wei, out0);
                    out1 = mad((COMPUTE_FLOAT8)inA.s4, wei, out1);
                    out2 = mad((COMPUTE_FLOAT8)inA.s8, wei, out2);
                    out3 = mad((COMPUTE_FLOAT8)inA.sc, wei, out3);
                }
                {
                    UCHAR4_TO_FLOAT8(charWeightsInt40.s4567, scale, offset);
                    out0 = mad((COMPUTE_FLOAT8)inA.s1, wei, out0);
                    out1 = mad((COMPUTE_FLOAT8)inA.s5, wei, out1);
                    out2 = mad((COMPUTE_FLOAT8)inA.s9, wei, out2);
                    out3 = mad((COMPUTE_FLOAT8)inA.sd, wei, out3);
                }
                {
                    UCHAR4_TO_FLOAT8(charWeightsInt40.s89ab, scale, offset);
                    out0 = mad((COMPUTE_FLOAT8)inA.s2, wei, out0);
                    out1 = mad((COMPUTE_FLOAT8)inA.s6, wei, out1);
                    out2 = mad((COMPUTE_FLOAT8)inA.sa, wei, out2);
                    out3 = mad((COMPUTE_FLOAT8)inA.se, wei, out3);
                }
                {
                    UCHAR4_TO_FLOAT8(charWeightsInt40.scdef, scale, offset);
                    out0 = mad((COMPUTE_FLOAT8)inA.s3, wei, out0);
                    out1 = mad((COMPUTE_FLOAT8)inA.s7, wei, out1);
                    out2 = mad((COMPUTE_FLOAT8)inA.sb, wei, out2);
                    out3 = mad((COMPUTE_FLOAT8)inA.sf, wei, out3);
                }
                // Remaining batches (4..6)
                {
                    COMPUTE_FLOAT16 inB = LOAD_INPUT16(input_offset + 16 + k * bhw4);
                    {
                        UCHAR4_TO_FLOAT8(charWeightsInt40.s0123, scale, offset);
                        out4 = mad((COMPUTE_FLOAT8)inB.s0, wei, out4);
                        #if INPUT_BATCH_LEAVES_NUM >= 6
                        out5 = mad((COMPUTE_FLOAT8)inB.s4, wei, out5);
                        #endif
                        #if INPUT_BATCH_LEAVES_NUM >= 7
                        out6 = mad((COMPUTE_FLOAT8)inB.s8, wei, out6);
                        #endif
                    }
                    {
                        UCHAR4_TO_FLOAT8(charWeightsInt40.s4567, scale, offset);
                        out4 = mad((COMPUTE_FLOAT8)inB.s1, wei, out4);
                        #if INPUT_BATCH_LEAVES_NUM >= 6
                        out5 = mad((COMPUTE_FLOAT8)inB.s5, wei, out5);
                        #endif
                        #if INPUT_BATCH_LEAVES_NUM >= 7
                        out6 = mad((COMPUTE_FLOAT8)inB.s9, wei, out6);
                        #endif
                    }
                    {
                        UCHAR4_TO_FLOAT8(charWeightsInt40.s89ab, scale, offset);
                        out4 = mad((COMPUTE_FLOAT8)inB.s2, wei, out4);
                        #if INPUT_BATCH_LEAVES_NUM >= 6
                        out5 = mad((COMPUTE_FLOAT8)inB.s6, wei, out5);
                        #endif
                        #if INPUT_BATCH_LEAVES_NUM >= 7
                        out6 = mad((COMPUTE_FLOAT8)inB.sa, wei, out6);
                        #endif
                    }
                    {
                        UCHAR4_TO_FLOAT8(charWeightsInt40.scdef, scale, offset);
                        out4 = mad((COMPUTE_FLOAT8)inB.s3, wei, out4);
                        #if INPUT_BATCH_LEAVES_NUM >= 6
                        out5 = mad((COMPUTE_FLOAT8)inB.s7, wei, out5);
                        #endif
                        #if INPUT_BATCH_LEAVES_NUM >= 7
                        out6 = mad((COMPUTE_FLOAT8)inB.sb, wei, out6);
                        #endif
                    }
                }
#else
                // INPUT_BATCH_LEAVES_NUM < 4: only 1-3 valid batches
                COMPUTE_FLOAT4 in0 = LOAD_INPUT4(input_offset + k * bhw4);
                #if INPUT_BATCH_LEAVES_NUM >= 2
                COMPUTE_FLOAT4 in1 = LOAD_INPUT4(input_offset + k * bhw4 + 4);
                #endif
                #if INPUT_BATCH_LEAVES_NUM >= 3
                COMPUTE_FLOAT4 in2 = LOAD_INPUT4(input_offset + k * bhw4 + 8);
                #endif
                {
                    UCHAR4_TO_FLOAT8(charWeightsInt40.s0123, scale, offset);
                    out0 = mad((COMPUTE_FLOAT8)in0.s0, wei, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((COMPUTE_FLOAT8)in1.s0, wei, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((COMPUTE_FLOAT8)in2.s0, wei, out2);
                    #endif
                }
                {
                    UCHAR4_TO_FLOAT8(charWeightsInt40.s4567, scale, offset);
                    out0 = mad((COMPUTE_FLOAT8)in0.s1, wei, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((COMPUTE_FLOAT8)in1.s1, wei, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((COMPUTE_FLOAT8)in2.s1, wei, out2);
                    #endif
                }
                {
                    UCHAR4_TO_FLOAT8(charWeightsInt40.s89ab, scale, offset);
                    out0 = mad((COMPUTE_FLOAT8)in0.s2, wei, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((COMPUTE_FLOAT8)in1.s2, wei, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((COMPUTE_FLOAT8)in2.s2, wei, out2);
                    #endif
                }
                {
                    UCHAR4_TO_FLOAT8(charWeightsInt40.scdef, scale, offset);
                    out0 = mad((COMPUTE_FLOAT8)in0.s3, wei, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((COMPUTE_FLOAT8)in1.s3, wei, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((COMPUTE_FLOAT8)in2.s3, wei, out2);
                    #endif
                }
#endif
            }
            #if INPUT_CHANNEL_LEAVES_NUM != 0
            {
                int k = i * loop + loop_end;
                COMPUTE_FLOAT8 wei;
                #ifdef USE_IMAGE
                uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(k, y)));
                #else
                uchar16 charWeightsInt40 = vload16(k, weight + weight_offset);
                #endif
#if INPUT_BATCH_LEAVES_NUM >= 4
                COMPUTE_FLOAT16 inA = LOAD_INPUT16(input_offset + k * bhw4);
                {
                    UCHAR4_TO_FLOAT8(charWeightsInt40.s0123, scale, offset);
                    out0 = mad((COMPUTE_FLOAT8)inA.s0, wei, out0);
                    out1 = mad((COMPUTE_FLOAT8)inA.s4, wei, out1);
                    out2 = mad((COMPUTE_FLOAT8)inA.s8, wei, out2);
                    out3 = mad((COMPUTE_FLOAT8)inA.sc, wei, out3);
                }
                #if INPUT_CHANNEL_LEAVES_NUM >= 2
                {
                    UCHAR4_TO_FLOAT8(charWeightsInt40.s4567, scale, offset);
                    out0 = mad((COMPUTE_FLOAT8)inA.s1, wei, out0);
                    out1 = mad((COMPUTE_FLOAT8)inA.s5, wei, out1);
                    out2 = mad((COMPUTE_FLOAT8)inA.s9, wei, out2);
                    out3 = mad((COMPUTE_FLOAT8)inA.sd, wei, out3);
                }
                #endif
                #if INPUT_CHANNEL_LEAVES_NUM >= 3
                {
                    UCHAR4_TO_FLOAT8(charWeightsInt40.s89ab, scale, offset);
                    out0 = mad((COMPUTE_FLOAT8)inA.s2, wei, out0);
                    out1 = mad((COMPUTE_FLOAT8)inA.s6, wei, out1);
                    out2 = mad((COMPUTE_FLOAT8)inA.sa, wei, out2);
                    out3 = mad((COMPUTE_FLOAT8)inA.se, wei, out3);
                }
                #endif
                {
                    COMPUTE_FLOAT16 inB = LOAD_INPUT16(input_offset + 16 + k * bhw4);
                    {
                        UCHAR4_TO_FLOAT8(charWeightsInt40.s0123, scale, offset);
                        out4 = mad((COMPUTE_FLOAT8)inB.s0, wei, out4);
                        #if INPUT_BATCH_LEAVES_NUM >= 6
                        out5 = mad((COMPUTE_FLOAT8)inB.s4, wei, out5);
                        #endif
                        #if INPUT_BATCH_LEAVES_NUM >= 7
                        out6 = mad((COMPUTE_FLOAT8)inB.s8, wei, out6);
                        #endif
                    }
                    #if INPUT_CHANNEL_LEAVES_NUM >= 2
                    {
                        UCHAR4_TO_FLOAT8(charWeightsInt40.s4567, scale, offset);
                        out4 = mad((COMPUTE_FLOAT8)inB.s1, wei, out4);
                        #if INPUT_BATCH_LEAVES_NUM >= 6
                        out5 = mad((COMPUTE_FLOAT8)inB.s5, wei, out5);
                        #endif
                        #if INPUT_BATCH_LEAVES_NUM >= 7
                        out6 = mad((COMPUTE_FLOAT8)inB.s9, wei, out6);
                        #endif
                    }
                    #endif
                    #if INPUT_CHANNEL_LEAVES_NUM >= 3
                    {
                        UCHAR4_TO_FLOAT8(charWeightsInt40.s89ab, scale, offset);
                        out4 = mad((COMPUTE_FLOAT8)inB.s2, wei, out4);
                        #if INPUT_BATCH_LEAVES_NUM >= 6
                        out5 = mad((COMPUTE_FLOAT8)inB.s6, wei, out5);
                        #endif
                        #if INPUT_BATCH_LEAVES_NUM >= 7
                        out6 = mad((COMPUTE_FLOAT8)inB.sa, wei, out6);
                        #endif
                    }
                    #endif
                }
#else
                COMPUTE_FLOAT4 in0 = LOAD_INPUT4(input_offset + k * bhw4);
                #if INPUT_BATCH_LEAVES_NUM >= 2
                COMPUTE_FLOAT4 in1 = LOAD_INPUT4(input_offset + k * bhw4 + 4);
                #endif
                #if INPUT_BATCH_LEAVES_NUM >= 3
                COMPUTE_FLOAT4 in2 = LOAD_INPUT4(input_offset + k * bhw4 + 8);
                #endif
                {
                    UCHAR4_TO_FLOAT8(charWeightsInt40.s0123, scale, offset);
                    out0 = mad((COMPUTE_FLOAT8)in0.s0, wei, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((COMPUTE_FLOAT8)in1.s0, wei, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((COMPUTE_FLOAT8)in2.s0, wei, out2);
                    #endif
                }
                #if INPUT_CHANNEL_LEAVES_NUM >= 2
                {
                    UCHAR4_TO_FLOAT8(charWeightsInt40.s4567, scale, offset);
                    out0 = mad((COMPUTE_FLOAT8)in0.s1, wei, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((COMPUTE_FLOAT8)in1.s1, wei, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((COMPUTE_FLOAT8)in2.s1, wei, out2);
                    #endif
                }
                #endif
                #if INPUT_CHANNEL_LEAVES_NUM >= 3
                {
                    UCHAR4_TO_FLOAT8(charWeightsInt40.s89ab, scale, offset);
                    out0 = mad((COMPUTE_FLOAT8)in0.s2, wei, out0);
                    #if INPUT_BATCH_LEAVES_NUM >= 2
                    out1 = mad((COMPUTE_FLOAT8)in1.s2, wei, out1);
                    #endif
                    #if INPUT_BATCH_LEAVES_NUM >= 3
                    out2 = mad((COMPUTE_FLOAT8)in2.s2, wei, out2);
                    #endif
                }
                #endif
#endif
            }
            #endif
        }
    } else {
#endif
    for (int i = 0; i < blockNum; i++){
        #ifdef ASYMMETRIC
        COMPUTE_FLOAT8 scale, offset;
        {
            COMPUTE_FLOAT16 scaleOffset = CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0, dequantScaleOffset + (out_c_idx << 3) + i * dstChannelAlign * 2)) / coef);
            scale = scaleOffset.s02468ace;
            offset = scaleOffset.s13579bdf;
        }
        #else
        COMPUTE_FLOAT8 scale = CONVERT_COMPUTE_FLOAT8(convert_float8(vload8(0, dequantScaleOffset + (out_c_idx << 2) + i * dstChannelAlign)) / coef);
        COMPUTE_FLOAT8 offset = (COMPUTE_FLOAT8)(-8) * scale;
        #endif
        for (int j = 0; j < loop_end; j++) {
            int k = i * loop + j;
            COMPUTE_FLOAT8 wei;
            
            #ifdef USE_IMAGE
            uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(k, y)));
            #else
            uchar16 charWeightsInt40 = vload16(k, weight + weight_offset);
            #endif
            
            COMPUTE_FLOAT16 inA = LOAD_INPUT16(input_offset + k * bhw4);
            COMPUTE_FLOAT16 inB = LOAD_INPUT16(input_offset + 16 + k * bhw4);
            
            {
                UCHAR4_TO_FLOAT8(charWeightsInt40.s0123, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)inA.s0, wei, out0);
                out1 = mad((COMPUTE_FLOAT8)inA.s4, wei, out1);
                out2 = mad((COMPUTE_FLOAT8)inA.s8, wei, out2);
                out3 = mad((COMPUTE_FLOAT8)inA.sc, wei, out3);
                out4 = mad((COMPUTE_FLOAT8)inB.s0, wei, out4);
                out5 = mad((COMPUTE_FLOAT8)inB.s4, wei, out5);
                out6 = mad((COMPUTE_FLOAT8)inB.s8, wei, out6);
                out7 = mad((COMPUTE_FLOAT8)inB.sc, wei, out7);
            }
            {
                UCHAR4_TO_FLOAT8(charWeightsInt40.s4567, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)inA.s1, wei, out0);
                out1 = mad((COMPUTE_FLOAT8)inA.s5, wei, out1);
                out2 = mad((COMPUTE_FLOAT8)inA.s9, wei, out2);
                out3 = mad((COMPUTE_FLOAT8)inA.sd, wei, out3);
                out4 = mad((COMPUTE_FLOAT8)inB.s1, wei, out4);
                out5 = mad((COMPUTE_FLOAT8)inB.s5, wei, out5);
                out6 = mad((COMPUTE_FLOAT8)inB.s9, wei, out6);
                out7 = mad((COMPUTE_FLOAT8)inB.sd, wei, out7);
            }
            {
                UCHAR4_TO_FLOAT8(charWeightsInt40.s89ab, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)inA.s2, wei, out0);
                out1 = mad((COMPUTE_FLOAT8)inA.s6, wei, out1);
                out2 = mad((COMPUTE_FLOAT8)inA.sa, wei, out2);
                out3 = mad((COMPUTE_FLOAT8)inA.se, wei, out3);
                out4 = mad((COMPUTE_FLOAT8)inB.s2, wei, out4);
                out5 = mad((COMPUTE_FLOAT8)inB.s6, wei, out5);
                out6 = mad((COMPUTE_FLOAT8)inB.sa, wei, out6);
                out7 = mad((COMPUTE_FLOAT8)inB.se, wei, out7);
            }
            {
                UCHAR4_TO_FLOAT8(charWeightsInt40.scdef, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)inA.s3, wei, out0);
                out1 = mad((COMPUTE_FLOAT8)inA.s7, wei, out1);
                out2 = mad((COMPUTE_FLOAT8)inA.sb, wei, out2);
                out3 = mad((COMPUTE_FLOAT8)inA.sf, wei, out3);
                out4 = mad((COMPUTE_FLOAT8)inB.s3, wei, out4);
                out5 = mad((COMPUTE_FLOAT8)inB.s7, wei, out5);
                out6 = mad((COMPUTE_FLOAT8)inB.sb, wei, out6);
                out7 = mad((COMPUTE_FLOAT8)inB.sf, wei, out7);
            }
        }
        #if INPUT_CHANNEL_LEAVES_NUM != 0
        {
            int k = i * loop + loop_end;
            COMPUTE_FLOAT8 wei;
            #ifdef USE_IMAGE
            uchar16 charWeightsInt40 = as_uchar16(read_imagei(weight, SAMPLER, (int2)(k, y)));
            #else
            uchar16 charWeightsInt40 = vload16(k, weight + weight_offset);
            #endif
            COMPUTE_FLOAT16 inA = LOAD_INPUT16(input_offset + k * bhw4);
            COMPUTE_FLOAT16 inB = LOAD_INPUT16(input_offset + 16 + k * bhw4);
            {
                UCHAR4_TO_FLOAT8(charWeightsInt40.s0123, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)inA.s0, wei, out0);
                out1 = mad((COMPUTE_FLOAT8)inA.s4, wei, out1);
                out2 = mad((COMPUTE_FLOAT8)inA.s8, wei, out2);
                out3 = mad((COMPUTE_FLOAT8)inA.sc, wei, out3);
                out4 = mad((COMPUTE_FLOAT8)inB.s0, wei, out4);
                out5 = mad((COMPUTE_FLOAT8)inB.s4, wei, out5);
                out6 = mad((COMPUTE_FLOAT8)inB.s8, wei, out6);
                out7 = mad((COMPUTE_FLOAT8)inB.sc, wei, out7);
            }
            #if INPUT_CHANNEL_LEAVES_NUM >= 2
            {
                UCHAR4_TO_FLOAT8(charWeightsInt40.s4567, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)inA.s1, wei, out0);
                out1 = mad((COMPUTE_FLOAT8)inA.s5, wei, out1);
                out2 = mad((COMPUTE_FLOAT8)inA.s9, wei, out2);
                out3 = mad((COMPUTE_FLOAT8)inA.sd, wei, out3);
                out4 = mad((COMPUTE_FLOAT8)inB.s1, wei, out4);
                out5 = mad((COMPUTE_FLOAT8)inB.s5, wei, out5);
                out6 = mad((COMPUTE_FLOAT8)inB.s9, wei, out6);
                out7 = mad((COMPUTE_FLOAT8)inB.sd, wei, out7);
            }
            #endif
            #if INPUT_CHANNEL_LEAVES_NUM >= 3
            {
                UCHAR4_TO_FLOAT8(charWeightsInt40.s89ab, scale, offset);
                out0 = mad((COMPUTE_FLOAT8)inA.s2, wei, out0);
                out1 = mad((COMPUTE_FLOAT8)inA.s6, wei, out1);
                out2 = mad((COMPUTE_FLOAT8)inA.sa, wei, out2);
                out3 = mad((COMPUTE_FLOAT8)inA.se, wei, out3);
                out4 = mad((COMPUTE_FLOAT8)inB.s2, wei, out4);
                out5 = mad((COMPUTE_FLOAT8)inB.s6, wei, out5);
                out6 = mad((COMPUTE_FLOAT8)inB.sa, wei, out6);
                out7 = mad((COMPUTE_FLOAT8)inB.se, wei, out7);
            }
            #endif
        }
        #endif
    }
#if INPUT_BATCH_LEAVES_NUM != 0
    }
#endif

#ifdef RELU
    out0 = fmax(out0, (COMPUTE_FLOAT8)0);
    out1 = fmax(out1, (COMPUTE_FLOAT8)0);
    out2 = fmax(out2, (COMPUTE_FLOAT8)0);
    out3 = fmax(out3, (COMPUTE_FLOAT8)0);
    out4 = fmax(out4, (COMPUTE_FLOAT8)0);
    out5 = fmax(out5, (COMPUTE_FLOAT8)0);
    out6 = fmax(out6, (COMPUTE_FLOAT8)0);
    out7 = fmax(out7, (COMPUTE_FLOAT8)0);
#endif

#ifdef RELU6
    out0 = clamp(out0, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
    out1 = clamp(out1, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
    out2 = clamp(out2, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
    out3 = clamp(out3, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
    out4 = clamp(out4, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
    out5 = clamp(out5, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
    out6 = clamp(out6, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
    out7 = clamp(out7, (COMPUTE_FLOAT8)0, (COMPUTE_FLOAT8)6);
#endif

#if INPUT_BATCH_LEAVES_NUM != 0
    if(out_b_idx + 7 >= bhw){
#if INPUT_BATCH_LEAVES_NUM >= 4
        // First 4 batches are always valid
        vstore16(CONVERT_FLOAT16((COMPUTE_FLOAT16)(out0.s0123, out1.s0123, out2.s0123, out3.s0123)), 0, output+out_offset);
        if((out_c_idx << 2) + 4 < dstChannelAlign){
            vstore16(CONVERT_FLOAT16((COMPUTE_FLOAT16)(out0.s4567, out1.s4567, out2.s4567, out3.s4567)), 0, output+out_offset+bhw4);
        }
        // Write remaining valid batches (4..6)
        #if INPUT_BATCH_LEAVES_NUM == 4
        // No remaining batches
        #elif INPUT_BATCH_LEAVES_NUM == 5
        {
            int off2 = out_offset + 16;
            vstore4(CONVERT_FLOAT4(out4.s0123), 0, output+off2);
            if((out_c_idx << 2) + 4 < dstChannelAlign){
                vstore4(CONVERT_FLOAT4(out4.s4567), 0, output+off2+bhw4);
            }
        }
        #elif INPUT_BATCH_LEAVES_NUM == 6
        {
            int off2 = out_offset + 16;
            vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out4.s0123, out5.s0123)), 0, output+off2);
            if((out_c_idx << 2) + 4 < dstChannelAlign){
                vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out4.s4567, out5.s4567)), 0, output+off2+bhw4);
            }
        }
        #elif INPUT_BATCH_LEAVES_NUM == 7
        {
            int off2 = out_offset + 16;
            vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out4.s0123, out5.s0123)), 0, output+off2);
            vstore4(CONVERT_FLOAT4(out6.s0123), 0, output+off2+8);
            if((out_c_idx << 2) + 4 < dstChannelAlign){
                vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out4.s4567, out5.s4567)), 0, output+off2+bhw4);
                vstore4(CONVERT_FLOAT4(out6.s4567), 0, output+off2+bhw4+8);
            }
        }
        #endif
#elif INPUT_BATCH_LEAVES_NUM == 3
        vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0.s0123, out1.s0123)), 0, output+out_offset);
        vstore4(CONVERT_FLOAT4(out2.s0123), 0, output+out_offset+8);
        if((out_c_idx << 2) + 4 < dstChannelAlign){
            vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0.s4567, out1.s4567)), 0, output+out_offset+bhw4);
            vstore4(CONVERT_FLOAT4(out2.s4567), 0, output+out_offset+bhw4+8);
        }
#elif INPUT_BATCH_LEAVES_NUM == 2
        vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0.s0123, out1.s0123)), 0, output+out_offset);
        if((out_c_idx << 2) + 4 < dstChannelAlign){
            vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0.s4567, out1.s4567)), 0, output+out_offset+bhw4);
        }
#elif INPUT_BATCH_LEAVES_NUM == 1
        vstore4(CONVERT_FLOAT4(out0.s0123), 0, output+out_offset);
        if((out_c_idx << 2) + 4 < dstChannelAlign){
            vstore4(CONVERT_FLOAT4(out0.s4567), 0, output+out_offset+bhw4);
        }
#endif
    } else {
#endif
        // Write all 8 batches
        vstore16(CONVERT_FLOAT16((COMPUTE_FLOAT16)(out0.s0123, out1.s0123, out2.s0123, out3.s0123)), 0, output+out_offset);
        vstore16(CONVERT_FLOAT16((COMPUTE_FLOAT16)(out4.s0123, out5.s0123, out6.s0123, out7.s0123)), 0, output+out_offset+16);
        if((out_c_idx << 2) + 4 < dstChannelAlign){
            vstore16(CONVERT_FLOAT16((COMPUTE_FLOAT16)(out0.s4567, out1.s4567, out2.s4567, out3.s4567)), 0, output+out_offset+bhw4);
            vstore16(CONVERT_FLOAT16((COMPUTE_FLOAT16)(out4.s4567, out5.s4567, out6.s4567, out7.s4567)), 0, output+out_offset+bhw4+16);
        }
#if INPUT_BATCH_LEAVES_NUM != 0
    }
#endif
}
