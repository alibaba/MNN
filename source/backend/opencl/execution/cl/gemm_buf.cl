#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_DIM2 \
    __private int global_size_dim0, __private int global_size_dim1,

#define UNIFORM_BOUNDRY_CHECK(index0, index1) \
    if(index0 >= global_size_dim0 || index1 >= global_size_dim1) { \
        return; \
    }

__kernel void gemm_buf(GLOBAL_SIZE_DIM2
                        __global const FLOAT* input0,
                        __global const FLOAT* input1,
                        __global FLOAT* output,
                        __private const int width,//UP_DIV(wUnit*hUnit,4)
                        __private const int height,//dstChannelC4
                        __private const int srcChannelC4,
                        __private const int alpha2) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    UNIFORM_BOUNDRY_CHECK(pos.x, pos.y);

    const int pos_x = pos.x % width;
    const int pos_y = pos.x / width;
    const int pos_z = pos.y;

    FLOAT16 o = (FLOAT16)0;
    
    int kenerlY   = mad24(pos_z, height, pos_y);
    int srcY      = mad24(pos_z, width, pos_x);

    for (int k = 0; k < srcChannelC4; ++k) {
        __private int index = mul24(k, 4);
        
        //NHWC  [1, 1, alpha2*height, srcChannelC4*4] x 4
        //index:[0, 0, pos_z*width+pos_y,    index+0]
        //int inp1_offset = (((k * (alpha2*height) + kenerlY) * (srcChannelC4*4) + index)*4 + 0)*4;
        
        FLOAT16 k_v16 = vload16(kenerlY*(srcChannelC4) + k, input1);
        
        //NC4HW4 [alpha*alpha, srcChannelC4, width, 4] x 4
        //index: [pos_z,       k,            pos_x, 0]
        
        FLOAT16 s = vload16(((pos_z*srcChannelC4 + k) * width + pos_x), input0);

        o = mad((FLOAT16)((FLOAT4)s.s0, (FLOAT4)s.s4, (FLOAT4)s.s8, (FLOAT4)s.sc), (FLOAT16)(k_v16.s0123, k_v16.s0123, k_v16.s0123, k_v16.s0123), o);
        o = mad((FLOAT16)((FLOAT4)s.s1, (FLOAT4)s.s5, (FLOAT4)s.s9, (FLOAT4)s.sd), (FLOAT16)(k_v16.s4567, k_v16.s4567, k_v16.s4567, k_v16.s4567), o);
        o = mad((FLOAT16)((FLOAT4)s.s2, (FLOAT4)s.s6, (FLOAT4)s.sa, (FLOAT4)s.se), (FLOAT16)(k_v16.s89ab, k_v16.s89ab, k_v16.s89ab, k_v16.s89ab), o);
        o = mad((FLOAT16)((FLOAT4)s.s3, (FLOAT4)s.s7, (FLOAT4)s.sb, (FLOAT4)s.sf), (FLOAT16)(k_v16.scdef, k_v16.scdef, k_v16.scdef, k_v16.scdef), o);
    }

    __private int out_y_idx = mul24(pos_y, 4);
    //NC4HW4 [dstChannelC4, alpha2, 4, UP_DIV(wUnit*hUnit,4)] x 4

    //index: [pos_y,  pos_z,  0, pos_x]
    int out_offset = (((pos_y * alpha2 + pos_z) * 4 + 0) * width + pos_x) * 4;

    vstore4(o.s0123, 0, output+out_offset);
    vstore4(o.s4567, 0, output+out_offset+4*width);
    vstore4(o.s89ab, 0, output+out_offset+8*width);
    vstore4(o.scdef, 0, output+out_offset+12*width);
}



__kernel void gemm_buf2(GLOBAL_SIZE_DIM2
                        __global const FLOAT* input0,
                        __global const FLOAT* input1,
                        __global FLOAT* output,
                        __private const int width,//UP_DIV(wUnit*hUnit,8)
                        __private const int height,//dstChannelC4
                        __private const int srcChannelC4,
                        __private const int alpha2) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    UNIFORM_BOUNDRY_CHECK(pos.x, pos.y);

    const int width_block = (width+1) >> 1;
    const int pos_x = (pos.x % width_block) << 1;
    const int pos_y = pos.x / width_block;
    const int pos_z = pos.y;

    FLOAT16 o0 = (FLOAT16)0;
    FLOAT16 o1 = (FLOAT16)0;

    const int kenerlY   = mad24(pos_z, height, pos_y);
    const int kernel_base = mul24(kenerlY, srcChannelC4);
    const int inp_base = (pos_z*srcChannelC4 + 0) * width + pos_x;
    
    for (int k = 0; k < srcChannelC4; ++k) {
        //NHWC  [1, 1, alpha2*height, srcChannelC4*4] x 4
        //index:[0, 0, pos_z*width+pos_y,    index+0]
        //int inp1_offset = (((k * (alpha2*height) + kenerlY) * (srcChannelC4*4) + index)*4 + 0)*4;
        
        FLOAT16 k_v16 = vload16(kernel_base + k, input1);
        
        //NC4HW4 [alpha*alpha, srcChannelC4, width, 4] x 4
        //index: [pos_z,       k,            pos_x, 0]
        
        const int inp_offset = mad24(k, width, inp_base);
        FLOAT16 s = vload16(inp_offset, input0);

        o0 = mad((FLOAT16)((FLOAT4)s.s0, (FLOAT4)s.s4, (FLOAT4)s.s8, (FLOAT4)s.sc), (FLOAT16)(k_v16.s0123, k_v16.s0123, k_v16.s0123, k_v16.s0123), o0);
        o0 = mad((FLOAT16)((FLOAT4)s.s1, (FLOAT4)s.s5, (FLOAT4)s.s9, (FLOAT4)s.sd), (FLOAT16)(k_v16.s4567, k_v16.s4567, k_v16.s4567, k_v16.s4567), o0);
        o0 = mad((FLOAT16)((FLOAT4)s.s2, (FLOAT4)s.s6, (FLOAT4)s.sa, (FLOAT4)s.se), (FLOAT16)(k_v16.s89ab, k_v16.s89ab, k_v16.s89ab, k_v16.s89ab), o0);
        o0 = mad((FLOAT16)((FLOAT4)s.s3, (FLOAT4)s.s7, (FLOAT4)s.sb, (FLOAT4)s.sf), (FLOAT16)(k_v16.scdef, k_v16.scdef, k_v16.scdef, k_v16.scdef), o0);
        
        s = vload16(inp_offset + 1, input0);
        o1 = mad((FLOAT16)((FLOAT4)s.s0, (FLOAT4)s.s4, (FLOAT4)s.s8, (FLOAT4)s.sc), (FLOAT16)(k_v16.s0123, k_v16.s0123, k_v16.s0123, k_v16.s0123), o1);
        o1 = mad((FLOAT16)((FLOAT4)s.s1, (FLOAT4)s.s5, (FLOAT4)s.s9, (FLOAT4)s.sd), (FLOAT16)(k_v16.s4567, k_v16.s4567, k_v16.s4567, k_v16.s4567), o1);
        o1 = mad((FLOAT16)((FLOAT4)s.s2, (FLOAT4)s.s6, (FLOAT4)s.sa, (FLOAT4)s.se), (FLOAT16)(k_v16.s89ab, k_v16.s89ab, k_v16.s89ab, k_v16.s89ab), o1);
        o1 = mad((FLOAT16)((FLOAT4)s.s3, (FLOAT4)s.s7, (FLOAT4)s.sb, (FLOAT4)s.sf), (FLOAT16)(k_v16.scdef, k_v16.scdef, k_v16.scdef, k_v16.scdef), o1);
    }

    __private int out_y_idx = mul24(pos_y, 4);
    //NC4HW4 [dstChannelC4, alpha2, 4, UP_DIV(wUnit*hUnit,4)] x 4

    //index: [pos_y,  pos_z,  0, pos_x]
    int out_offset = (((pos_y * alpha2 + pos_z) * 4 + 0) * width + pos_x) * 4;

    vstore4(o0.s0123, 0, output+out_offset);
    vstore4(o0.s4567, 0, output+out_offset+4*width);
    vstore4(o0.s89ab, 0, output+out_offset+8*width);
    vstore4(o0.scdef, 0, output+out_offset+12*width);
    
    if(pos_x + 1 >= width) return;
    vstore4(o1.s0123, 1, output+out_offset);
    vstore4(o1.s4567, 1, output+out_offset+4*width);
    vstore4(o1.s89ab, 1, output+out_offset+8*width);
    vstore4(o1.scdef, 1, output+out_offset+12*width);
}

