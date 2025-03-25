#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#define PI 3.141592653589f
#ifndef WGSW
    #define WGSW 32 // work-group handle size W dimension
#endif
#ifndef WGSC
    #define WGSC 32 // work-group handle size C dimension
#endif
#ifndef WGSH
    #define WGSH 32 // work-group handle size H dimension
#endif
#ifndef TSW
    #define TSW 8 // thread handle size W dimension
#endif
#ifndef TSC
    #define TSC 8 // thread handle size C dimension
#endif
#ifndef TSH
    #define TSH 8 // thread handle size H dimension
#endif

// [C4 N H 1 4] -> [N H C 1]
__kernel void tile_trans_3d_buf(__global INPUT_TYPE* input,
                        __global OUTPUT_TYPE* output,
                        __private const int widthPad,
                        __private const int heightPad,
                        __private const int channelPad,
                        __private const int batch,
                        __private const int width,
                        __private const int height,
                        __private const int channel
) {
    int b = get_global_id(2);
    
    const int lidc = get_local_id(0);
    const int lidh = get_local_id(1);
    // group id
    const int c = get_group_id(0) * WGSC;
    const int h = get_group_id(1) * WGSH;

    int jc = lidc;
    int ih = lidh;
    
    __local INPUT_TYPE4 localData[WGSH][WGSC/4];//h64c64
    
    #pragma unroll
    for(int i = 0; i < TSH; i++) {
        #pragma unroll
        for(int j = 0; j < TSC / 4; j++) {
            int offset_h = i * WGSH / TSH + ih;
            int offset_c = j * WGSC / TSC + jc ;
            // [TSH, WGSH / TSH]   [TSC / 4, WGSC / TSC, 4]
            localData[offset_h][offset_c] = (h + offset_h >= height || c + 4 * offset_c >= channel) ? (INPUT_TYPE4)0 : vload4(0, input + ((b + (c/4+offset_c)*batch) * height + (h+offset_h)) * 4);
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // C offset: [WGSC / TSC, TSC / 4]
    // H offset: [WGSH / TSH, TSH]
    int oc_base = jc * TSC / 4;
    int oh_base = ih * TSH;

    //#pragma unroll
    for(int i = 0; i < TSH; i++) {
        int oh = oh_base + i;

        //#pragma unroll
        for(int j = 0; j < TSC / 4; j++) {
            int oc = oc_base + j;
            
            OUTPUT_TYPE4 value =  CONVERT_OUTPUT4(localData[oh][oc]);

            vstore4(value, 0, output + ((b * heightPad + h + oh) * channelPad + c + 4 * oc));
        }
    }
}
// [C4 N H W 4] -> [N C W H]
__kernel void tile_trans_4d_buf(__global INPUT_TYPE* input,
                        __global OUTPUT_TYPE* output,
                        __private const int widthPad,
                        __private const int heightPad,
                        __private const int channelPad,
                        __private const int batch,
                        __private const int width,
                        __private const int height,
                        __private const int channel
) {
    int bc = get_global_id(2);
    int b  = bc % batch;
    int c4 = bc / batch;
    int c = c4 << 2;
    
    const int lidw = get_local_id(0);
    const int lidh = get_local_id(1);
    // group id
    const int w = get_group_id(0) * WGSW;
    const int h = get_group_id(1) * WGSH;

    int jw = lidw;
    int ih = lidh;
    
    __local INPUT_TYPE4 localData[WGSH][WGSW];//w32h32c4
    
    #pragma unroll
    for(int i = 0; i < TSH; i++) {
        #pragma unroll
        for(int j = 0; j < TSW; j++) {
            int offset_h = h + ih + i * WGSH/TSH;
            int offset_w = w + jw + j * WGSW/TSW;
            localData[ih + i * WGSH / TSH][jw + j * WGSW/TSW] = (offset_h >= height || offset_w >= width) ? (INPUT_TYPE4)0 : vload4(0, input + (((b + c4 * batch) * height + offset_h) * width + offset_w) * 4);
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // c4w32h32
    int oh = ih * TSH >> 4;
    int mh = ih & (16 / TSH - 1);
    // TSW offset: [TSH / 4, TSW / 4, 16 / TSH]
    int ow_base = jw * TSW;
    int oh_offset = oh << 4;

    //#pragma unroll
    for(int i = 0; i < TSH / 4; i++) {
        //#pragma unroll
        for(int j = 0; j < TSW / 4; j++) {
            
            // c4
            OUTPUT_TYPE16 value;
            int ow = ow_base + (((i * TSW / 4) + j) * (16 / TSH) + mh);
            
            value.s0 =  localData[0+oh_offset][ow].s0;
            value.s1 =  localData[1+oh_offset][ow].s0;
            value.s2 =  localData[2+oh_offset][ow].s0;
            value.s3 =  localData[3+oh_offset][ow].s0;
            value.s4 =  localData[4+oh_offset][ow].s0;
            value.s5 =  localData[5+oh_offset][ow].s0;
            value.s6 =  localData[6+oh_offset][ow].s0;
            value.s7 =  localData[7+oh_offset][ow].s0;
            value.s8 =  localData[8+oh_offset][ow].s0;
            value.s9 =  localData[9+oh_offset][ow].s0;
            value.sa = localData[10+oh_offset][ow].s0;
            value.sb = localData[11+oh_offset][ow].s0;
            value.sc = localData[12+oh_offset][ow].s0;
            value.sd = localData[13+oh_offset][ow].s0;
            value.se = localData[14+oh_offset][ow].s0;
            value.sf = localData[15+oh_offset][ow].s0;
            vstore16(value, 0, output + (((b * channelPad + c + 0) * widthPad + w + ow) * heightPad + h + oh_offset));
            
            if(c + 1 < channel) {
                value.s0 =  localData[0+oh_offset][ow].s1;
                value.s1 =  localData[1+oh_offset][ow].s1;
                value.s2 =  localData[2+oh_offset][ow].s1;
                value.s3 =  localData[3+oh_offset][ow].s1;
                value.s4 =  localData[4+oh_offset][ow].s1;
                value.s5 =  localData[5+oh_offset][ow].s1;
                value.s6 =  localData[6+oh_offset][ow].s1;
                value.s7 =  localData[7+oh_offset][ow].s1;
                value.s8 =  localData[8+oh_offset][ow].s1;
                value.s9 =  localData[9+oh_offset][ow].s1;
                value.sa = localData[10+oh_offset][ow].s1;
                value.sb = localData[11+oh_offset][ow].s1;
                value.sc = localData[12+oh_offset][ow].s1;
                value.sd = localData[13+oh_offset][ow].s1;
                value.se = localData[14+oh_offset][ow].s1;
                value.sf = localData[15+oh_offset][ow].s1;
                vstore16(value, 0, output + (((b * channelPad + c + 1) * widthPad + w + ow) * heightPad + h + oh_offset));
            }
            
            if(c + 2 < channel) {
                value.s0 =  localData[0+oh_offset][ow].s2;
                value.s1 =  localData[1+oh_offset][ow].s2;
                value.s2 =  localData[2+oh_offset][ow].s2;
                value.s3 =  localData[3+oh_offset][ow].s2;
                value.s4 =  localData[4+oh_offset][ow].s2;
                value.s5 =  localData[5+oh_offset][ow].s2;
                value.s6 =  localData[6+oh_offset][ow].s2;
                value.s7 =  localData[7+oh_offset][ow].s2;
                value.s8 =  localData[8+oh_offset][ow].s2;
                value.s9 =  localData[9+oh_offset][ow].s2;
                value.sa = localData[10+oh_offset][ow].s2;
                value.sb = localData[11+oh_offset][ow].s2;
                value.sc = localData[12+oh_offset][ow].s2;
                value.sd = localData[13+oh_offset][ow].s2;
                value.se = localData[14+oh_offset][ow].s2;
                value.sf = localData[15+oh_offset][ow].s2;
                vstore16(value, 0, output + (((b * channelPad + c + 2) * widthPad + w + ow) * heightPad + h + oh_offset));
            }
            
            if(c + 3 < channel) {
                value.s0 =  localData[0+oh_offset][ow].s3;
                value.s1 =  localData[1+oh_offset][ow].s3;
                value.s2 =  localData[2+oh_offset][ow].s3;
                value.s3 =  localData[3+oh_offset][ow].s3;
                value.s4 =  localData[4+oh_offset][ow].s3;
                value.s5 =  localData[5+oh_offset][ow].s3;
                value.s6 =  localData[6+oh_offset][ow].s3;
                value.s7 =  localData[7+oh_offset][ow].s3;
                value.s8 =  localData[8+oh_offset][ow].s3;
                value.s9 =  localData[9+oh_offset][ow].s3;
                value.sa = localData[10+oh_offset][ow].s3;
                value.sb = localData[11+oh_offset][ow].s3;
                value.sc = localData[12+oh_offset][ow].s3;
                value.sd = localData[13+oh_offset][ow].s3;
                value.se = localData[14+oh_offset][ow].s3;
                value.sf = localData[15+oh_offset][ow].s3;
                vstore16(value, 0, output + (((b * channelPad + c + 3) * widthPad + w + ow) * heightPad + h + oh_offset));
            }
        }
    }
}

__kernel void tile_buf(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                        __global INPUT_TYPE* input, __global OUTPUT_TYPE* output,
                        __private const int widthPad,
                        __private const int heightPad,
                        __private const int channelPad,
                        __private const int batch,
                        __private const int width,
                        __private const int height,
                        __private const int channel){
    int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
    if (pos.x < global_dim0 && pos.y < global_dim1 && pos.z < global_dim2) {
        const int b = pos.z % batch;
        const int w = pos.x;
        const int h = pos.y;
        const int c_4 = pos.z / batch;
        
        const int c = c_4 << 2;
        const int x_src_pitch = 4;
        const int y_src_pitch = x_src_pitch * width;
        const int b_src_pitch = y_src_pitch * height;
        const int c_src_pitch = b_src_pitch * batch;
        
        bool outBound = (w >= width || h >= height || c >= channel);
#ifdef MNN_NHWC
    #if defined(DIMENSION_3) && defined(TRANSPOSE)
        // [N, W, H, 1]
        const int c_dst_pitch = 1;
        const int y_dst_pitch = c_dst_pitch * channelPad;
        const int x_dst_pitch = y_dst_pitch * heightPad;
        const int b_dst_pitch = x_dst_pitch * widthPad;
        OUTPUT_TYPE4 value = outBound ? (OUTPUT_TYPE4)0 : CONVERT_OUTPUT4(vload4(0, input + b * b_src_pitch + c_4 * c_src_pitch + h * y_src_pitch + w * x_src_pitch));
    #elif defined(DIMENSION_4) && defined(TRANSPOSE)
        // [N, H, C, W]
        const int x_dst_pitch = 1;
        const int c_dst_pitch = x_dst_pitch * widthPad;
        const int y_dst_pitch = c_dst_pitch * channelPad;
        const int b_dst_pitch = y_dst_pitch * heightPad;
        OUTPUT_TYPE4 value = outBound ? (OUTPUT_TYPE4)0 : CONVERT_OUTPUT4(vload4(0, input + b * b_src_pitch + c_4 * c_src_pitch + h * y_src_pitch + w * x_src_pitch));
    #elif defined(DIMENSION_3)
        // [N, H, W, 1]
        const int c_dst_pitch = 1;
        const int x_dst_pitch = c_dst_pitch * channelPad;
        const int y_dst_pitch = x_dst_pitch * widthPad;
        const int b_dst_pitch = y_dst_pitch * heightPad;
        OUTPUT_TYPE4 value = outBound ? (OUTPUT_TYPE4)0 : CONVERT_OUTPUT4(vload4(0, input + b * b_src_pitch + c_4 * c_src_pitch + h * y_src_pitch + w * x_src_pitch));
    #else
        // [N, H, W, C]
        const int c_dst_pitch = 1;
        const int x_dst_pitch = c_dst_pitch * channelPad;
        const int y_dst_pitch = x_dst_pitch * widthPad;
        const int b_dst_pitch = y_dst_pitch * heightPad;
        OUTPUT_TYPE4 value = outBound ? (OUTPUT_TYPE4)0 : CONVERT_OUTPUT4(vload4(0, input + b * b_src_pitch + c_4 * c_src_pitch + h * y_src_pitch + w * x_src_pitch));
    #endif
#else
    #if defined(DIMENSION_3) && defined(TRANSPOSE)
        // [N, H, C, 1]
        const int x_dst_pitch = 1;
        const int c_dst_pitch = x_dst_pitch * widthPad;
        const int y_dst_pitch = c_dst_pitch * channelPad;
        const int b_dst_pitch = y_dst_pitch * heightPad;
        OUTPUT_TYPE4 value = outBound ? (OUTPUT_TYPE4)0 : CONVERT_OUTPUT4(vload4(0, input + b * b_src_pitch + c_4 * c_src_pitch + h * y_src_pitch + w * x_src_pitch));
        
    #elif defined(DIMENSION_4) && defined(TRANSPOSE)
        // [N, C, W, H]
        const int y_dst_pitch = 1;
        const int x_dst_pitch = y_dst_pitch * heightPad;
        const int c_dst_pitch = x_dst_pitch * widthPad;
        const int b_dst_pitch = c_dst_pitch * channelPad;
        OUTPUT_TYPE4 value = outBound ? (OUTPUT_TYPE4)0 : CONVERT_OUTPUT4(vload4(0, input + b * b_src_pitch + c_4 * c_src_pitch + h * y_src_pitch + w * x_src_pitch));
    #elif defined(DIMENSION_3)
        // [N, C, H, 1]
        const int x_dst_pitch = 1;
        const int y_dst_pitch = x_dst_pitch * widthPad;
        const int c_dst_pitch = y_dst_pitch * heightPad;
        const int b_dst_pitch = c_dst_pitch * channelPad;
        OUTPUT_TYPE4 value = outBound ? (OUTPUT_TYPE4)0 : CONVERT_OUTPUT4(vload4(0, input + b * b_src_pitch + c_4 * c_src_pitch + h * y_src_pitch + w * x_src_pitch));
    #else
        // [N, C, H, W]
        const int x_dst_pitch = 1;
        const int y_dst_pitch = x_dst_pitch * widthPad;
        const int c_dst_pitch = y_dst_pitch * heightPad;
        const int b_dst_pitch = c_dst_pitch * channelPad;
        OUTPUT_TYPE4 value = outBound ? (OUTPUT_TYPE4)0 : CONVERT_OUTPUT4(vload4(0, input + b * b_src_pitch + c_4 * c_src_pitch + h * y_src_pitch + w * x_src_pitch));
    #endif
#endif

        __global OUTPUT_TYPE* dst_ptr = output + b * b_dst_pitch + c * c_dst_pitch + h * y_dst_pitch + w * x_dst_pitch;

        dst_ptr[0] = value.x;
        if(c + 1 >= channel)return;
        dst_ptr[c_dst_pitch] = value.y;
        if(c + 2 >= channel)return;
        dst_ptr[2 * c_dst_pitch] = value.z;
        if(c + 3 >= channel)return;
        dst_ptr[3 * c_dst_pitch] = value.w;
    }
}

__kernel void pack_buf(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                        __global INPUT_TYPE* input, __global OUTPUT_TYPE* output,
                        __private const int widthPad,
                        __private const int heightPad,
                        __private const int channelPad,
                        __private const int batch,
                        __private const int width,
                        __private const int height,
                        __private const int channel){
    int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
    if (pos.x < global_dim0 && pos.y < global_dim1 && pos.z < global_dim2) {
        
        const int b = pos.z % batch;
        const int w = pos.x;
        const int h = pos.y;
        const int c_4 = pos.z / batch;
        
        const int c = c_4 << 2;
        if(w >= width || h >= height || c >= channel) {
            return;
        }
        const int x_dst_pitch = 4;
        const int y_dst_pitch = x_dst_pitch * width;
        const int c_dst_pitch = y_dst_pitch * height;
        const int b_dst_pitch = c_dst_pitch * ((channel + 3) / 4);
#ifdef MNN_NHWC
    #if defined(TRANSPOSE) && defined(DIMENSION_3)
        // [N, W, H, 1]
        const int c_src_pitch = 1;
        const int y_src_pitch = c_src_pitch;
        const int x_src_pitch = y_src_pitch * heightPad;
        const int b_src_pitch = x_src_pitch * widthPad;
    #elif defined(TRANSPOSE) && defined(DIMENSION_4)
        // [N, H, C, W]
        const int x_src_pitch = 1;
        const int c_src_pitch = x_src_pitch * widthPad;
        const int y_src_pitch = c_src_pitch * channelPad;
        const int b_src_pitch = y_src_pitch * heightPad;
    #else
        // [N, H, W, C]
        const int c_src_pitch = 1;
        const int x_src_pitch = c_src_pitch * channelPad;
        const int y_src_pitch = x_src_pitch * widthPad;
        const int b_src_pitch = y_src_pitch * heightPad;
    #endif
#else
    #if defined(TRANSPOSE) && defined(DIMENSION_3)
        // dst:[N, C, H, 1] -> src:[N, H, C, 1]
        const int x_src_pitch = 1;
        const int c_src_pitch = x_src_pitch * widthPad;
        const int y_src_pitch = c_src_pitch * channelPad;
        const int b_src_pitch = y_src_pitch * heightPad;
    #elif defined(TRANSPOSE) && defined(DIMENSION_4)
        // dst:[N, C, H, W] -> src:[N, C, W, H]
        const int y_src_pitch = 1;
        const int x_src_pitch = y_src_pitch * heightPad;
        const int c_src_pitch = x_src_pitch * widthPad;
        const int b_src_pitch = c_src_pitch * channelPad;
    #else
        // [N, C, H, W]
        const int x_src_pitch = 1;
        const int y_src_pitch = x_src_pitch * widthPad;
        const int c_src_pitch = y_src_pitch * heightPad;
        const int b_src_pitch = c_src_pitch * channelPad;
    #endif
#endif
        __global INPUT_TYPE* src_ptr = input + b * b_src_pitch + c * c_src_pitch + h * y_src_pitch + w * x_src_pitch;
        OUTPUT_TYPE4 value = (OUTPUT_TYPE4)0;
        OUTPUT_TYPE *value_ptr = (OUTPUT_TYPE*)&value;
        for(int i = 0; i < 4 && (i + c < channel); ++i){
            value_ptr[i] = (OUTPUT_TYPE)src_ptr[i * c_src_pitch];
        }
        vstore4(value, 0, output + b * b_dst_pitch + c_4 * c_dst_pitch + h * y_dst_pitch + w * x_dst_pitch);
    }
}

#ifdef LOOP_BINARY_OPERATOR
__kernel void loop_binary_buf(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                         __global OUTPUT_TYPE* output, __global INPUT_TYPE* input0, __global INPUT_TYPE* input1,
                         __private const int input0Stride0,
                         __private const int input0Stride1,
                         __private const int input0Stride2,
                         __private const int input1Stride0,
                         __private const int input1Stride1,
                         __private const int input1Stride2,
                         __private const int outputStride0,
                         __private const int outputStride1,
                         __private const int outputStride2
                         ) {
                             
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    
    if (x < global_dim0 && y < global_dim1 && z < global_dim2) {
        
        int inputIndex0 = z * input0Stride0 + y * input0Stride1 + x * input0Stride2;
        int inputIndex1 = z * input1Stride0 + y * input1Stride1 + x * input1Stride2;
        int outputIndex = z * outputStride0 + y * outputStride1 + x * outputStride2;
        float in0 = (float)input0[inputIndex0];
        float in1 = (float)input1[inputIndex1];
        float out = LOOP_BINARY_OPERATOR;
        output[outputIndex] = (OUTPUT_TYPE)out;
    }
}
#endif
