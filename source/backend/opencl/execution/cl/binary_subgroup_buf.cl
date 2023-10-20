#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__kernel void binary_buf_c4_c4_c4(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                         __global FLOAT* input0, __global FLOAT* input1, __global FLOAT* output,
                         __private const int4 shape,//[N,H,W,C]
                         __private const int2 isFull,
                         __private const int activationType,
                        __private const int input0_pad_left, __private const int input0_pad_right,
                        __private const int input1_pad_left, __private const int input1_pad_right,
                        __private const int output_pad_left, __private const int output_pad_right) {
    if (get_global_id(0) >= global_dim0 || get_global_id(1) >= global_dim1 || get_global_id(2) >= global_dim2) 
        return;
    const int channel4 = (shape.w + 3) / 4;
    const int w_idx = get_global_id(0) % shape.z;
    const int h_idx = get_global_id(0) / shape.z;
    const int batch_idx = get_global_id(2);
    const int channel_idx = get_global_id(1);

    const int offset = (((batch_idx*channel4+channel_idx)*shape.y+h_idx)*shape.z+w_idx) * 4;
    
#ifdef OPENCL_INPUT_INT
    FLOAT4 in0 = CONVERT_FLOAT4(convert_int4(vload4(0, input0 + offset*isFull.x)));
    FLOAT4 in1 = CONVERT_FLOAT4(convert_int4(vload4(0, input1 + offset*isFull.y)));
#else
    FLOAT4 in0 = vload4(0, input0 + offset*isFull.x);
    FLOAT4 in1 = vload4(0, input1 + offset*isFull.y);
#endif
    if(isFull.x == 0) {
        in0 = (FLOAT4)(in0.x, in0.x, in0.x, in0.x);
    }
    if(isFull.y == 0) {
        in1 = (FLOAT4)(in1.x, in1.x, in1.x, in1.x);
    }
#ifdef OPENCL_INPUT_INT
    FLOAT4 out = CONVERT_FLOAT4(convert_int4(OPERATOR));
#else
    FLOAT4 out = CONVERT_FLOAT4(OPERATOR);
#endif
    if(activationType == 1) {
        out = fmax(out, (FLOAT4)0);
    }
    vstore4(out, 0, output + offset);
}

__kernel void binary_buf_c4_c4_c16(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                         __global FLOAT* input0, __global FLOAT* input1, __global FLOAT* output,
                         __private const int4 shape,//[N,H,W,C]
                         __private const int2 isFull,
                         __private const int activationType,
                        __private const int input0_pad_left, __private const int input0_pad_right,
                        __private const int input1_pad_left, __private const int input1_pad_right,
                        __private const int output_pad_left, __private const int output_pad_right) {
    if (get_global_id(0) >= global_dim0 || get_global_id(1) >= global_dim1 || get_global_id(2) >= global_dim2) 
        return;
    const int channel4 = (shape.w + 3) / 4;
    const int channel16 = (shape.w + 15) / 16;
    const int w_idx = get_global_id(0) % shape.z;
    const int h_idx = get_global_id(0) / shape.z;
    const int batch_idx = get_global_id(2);
    const int channel_idx = get_global_id(1);
    const int dst_width = shape.z + output_pad_left + output_pad_right;
    const int channe_out_idx = channel_idx >> 2;

    const int offset = (((batch_idx*channel4+channel_idx)*shape.y+h_idx)*shape.z+w_idx) * 4;
    const int dst_offset =  (((batch_idx*channel16+channe_out_idx)*shape.y+h_idx)*dst_width+w_idx+output_pad_left) * 16 + (channel_idx % 4) * 4;
    
#ifdef OPENCL_INPUT_INT
    FLOAT4 in0 = CONVERT_FLOAT4(convert_int4(vload4(0, input0 + offset*isFull.x)));
    FLOAT4 in1 = CONVERT_FLOAT4(convert_int4(vload4(0, input1 + offset*isFull.y)));
#else
    FLOAT4 in0 = vload4(0, input0 + offset*isFull.x);
    FLOAT4 in1 = vload4(0, input1 + offset*isFull.y);
#endif
    if(isFull.x == 0) {
        in0 = (FLOAT4)(in0.x, in0.x, in0.x, in0.x);
    }
    if(isFull.y == 0) {
        in1 = (FLOAT4)(in1.x, in1.x, in1.x, in1.x);
    }
#ifdef OPENCL_INPUT_INT
    FLOAT4 out = CONVERT_FLOAT4(convert_int4(OPERATOR));
#else
    FLOAT4 out = CONVERT_FLOAT4(OPERATOR);
#endif
    if(activationType == 1) {
        out = fmax(out, (FLOAT4)0);
    }
    vstore4(out, 0, output + dst_offset);
    if(w_idx == 0){
        int pad_offset = (((batch_idx*channel16+channe_out_idx)*shape.y+h_idx)*dst_width) * 16 + (channel_idx % 4) * 4;
        for(int i = 0; i < output_pad_left; ++i){
            vstore4((FLOAT4)0, 0, output + pad_offset + i * 16);
        }
        pad_offset += (shape.z + output_pad_left) * 16;
        for(int i = 0; i < output_pad_right; ++i){
            vstore4((FLOAT4)0, 0, output + pad_offset + i * 16);
        }
    }
}

__kernel void binary_buf_c4_c16_c4(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                         __global FLOAT* input0, __global FLOAT* input1, __global FLOAT* output,
                         __private const int4 shape,//[N,H,W,C]
                         __private const int2 isFull,
                         __private const int activationType,
                        __private const int input0_pad_left, __private const int input0_pad_right,
                        __private const int input1_pad_left, __private const int input1_pad_right,
                        __private const int output_pad_left, __private const int output_pad_right) {
    if (get_global_id(0) >= global_dim0 || get_global_id(1) >= global_dim1 || get_global_id(2) >= global_dim2) 
        return;
    const int channel4 = (shape.w + 3) / 4;
    const int channel16 = (shape.w + 15) / 16;
    const int w_idx = get_global_id(0) % shape.z;
    const int h_idx = get_global_id(0) / shape.z;
    const int batch_idx = get_global_id(2);
    const int channel_idx = get_global_id(1);
    const int src_width = shape.z + input1_pad_left + input1_pad_right;
    const int channe_out_idx = channel_idx >> 2;

    const int offset0 = (((batch_idx*channel4+channel_idx)*shape.y+h_idx)*shape.z+w_idx) * 4;
    const int offset1 = (((batch_idx*channel16+channe_out_idx)*shape.y+h_idx)*src_width+w_idx+input1_pad_left) * 16 + (channel_idx % 4) * 4;

#ifdef OPENCL_INPUT_INT
    FLOAT4 in0 = CONVERT_FLOAT4(convert_int4(vload4(0, input0 + offset0*isFull.x)));
    FLOAT4 in1 = CONVERT_FLOAT4(convert_int4(vload4(0, input1 + offset1*isFull.y)));
#else
    FLOAT4 in0 = vload4(0, input0 + offset0*isFull.x);
    FLOAT4 in1 = vload4(0, input1 + offset1*isFull.y);
#endif
    if(isFull.x == 0) {
        in0 = (FLOAT4)(in0.x, in0.x, in0.x, in0.x);
    }
    if(isFull.y == 0) {
        in1 = (FLOAT4)(in1.x, in1.x, in1.x, in1.x);
    }
#ifdef OPENCL_INPUT_INT
    FLOAT4 out = CONVERT_FLOAT4(convert_int4(OPERATOR));
#else
    FLOAT4 out = CONVERT_FLOAT4(OPERATOR);
#endif
    if(activationType == 1) {
        out = fmax(out, (FLOAT4)0);
    }
    vstore4(out, 0, output + offset0);
}

__kernel void binary_buf_c16_c4_c4(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                         __global FLOAT* input0, __global FLOAT* input1, __global FLOAT* output,
                         __private const int4 shape,//[N,H,W,C]
                         __private const int2 isFull,
                         __private const int activationType,
                        __private const int input0_pad_left, __private const int input0_pad_right,
                        __private const int input1_pad_left, __private const int input1_pad_right,
                        __private const int output_pad_left, __private const int output_pad_right) {
    if (get_global_id(0) >= global_dim0 || get_global_id(1) >= global_dim1 || get_global_id(2) >= global_dim2) 
        return;
    const int channel4 = (shape.w + 3) / 4;
    const int channel16 = (shape.w + 15) / 16;
    const int w_idx = get_global_id(0) % shape.z;
    const int h_idx = get_global_id(0) / shape.z;
    const int batch_idx = get_global_id(2);
    const int channel_idx = get_global_id(1);
    const int src_width = shape.z + input0_pad_left + input0_pad_right;
    const int channe_out_idx = channel_idx >> 2;

    const int offset1 = (((batch_idx*channel4+channel_idx)*shape.y+h_idx)*shape.z+w_idx) * 4;
    const int offset0 = (((batch_idx*channel16+channe_out_idx)*shape.y+h_idx)*src_width+w_idx+input0_pad_left) * 16 + (channel_idx % 4) * 4;
    
#ifdef OPENCL_INPUT_INT
    FLOAT4 in0 = CONVERT_FLOAT4(convert_int4(vload4(0, input0 + offset0*isFull.x)));
    FLOAT4 in1 = CONVERT_FLOAT4(convert_int4(vload4(0, input1 + offset1*isFull.y)));
#else
    FLOAT4 in0 = vload4(0, input0 + offset0*isFull.x);
    FLOAT4 in1 = vload4(0, input1 + offset1*isFull.y);
#endif
    if(isFull.x == 0) {
        in0 = (FLOAT4)(in0.x, in0.x, in0.x, in0.x);
    }
    if(isFull.y == 0) {
        in1 = (FLOAT4)(in1.x, in1.x, in1.x, in1.x);
    }
#ifdef OPENCL_INPUT_INT
    FLOAT4 out = CONVERT_FLOAT4(convert_int4(OPERATOR));
#else
    FLOAT4 out = CONVERT_FLOAT4(OPERATOR);
#endif
    if(activationType == 1) {
        out = fmax(out, (FLOAT4)0);
    }
    vstore4(out, 0, output + offset1);
}

__kernel void binary_buf_c4_c16_c16(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                         __global FLOAT* input0, __global FLOAT* input1, __global FLOAT* output,
                         __private const int4 shape,//[N,H,W,C]
                         __private const int2 isFull,
                         __private const int activationType,
                        __private const int input0_pad_left, __private const int input0_pad_right,
                        __private const int input1_pad_left, __private const int input1_pad_right,
                        __private const int output_pad_left, __private const int output_pad_right) {
    if (get_global_id(0) >= global_dim0 || get_global_id(1) >= global_dim1 || get_global_id(2) >= global_dim2) 
        return;
    const int channel4 = (shape.w + 3) / 4;
    const int channel16 = (shape.w + 15) / 16;
    const int w_idx = get_global_id(0) % shape.z;
    const int h_idx = get_global_id(0) / shape.z;
    const int batch_idx = get_global_id(2);
    const int channel_idx = get_global_id(1);
    const int src_width = shape.z + input1_pad_left + input1_pad_right;
    const int dst_width = shape.z + output_pad_left + output_pad_right;
    const int channe_out_idx = channel_idx >> 2;

    const int offset0 = (((batch_idx*channel4+channel_idx)*shape.y+h_idx)*shape.z+w_idx) * 4;
    const int offset1 = (((batch_idx*channel16+channe_out_idx)*shape.y+h_idx)*src_width+w_idx+input1_pad_left) * 16 + (channel_idx % 4) * 4;
    const int dst_offset =  (((batch_idx*channel16+channe_out_idx)*shape.y+h_idx)*dst_width+w_idx+output_pad_left) * 16 + (channel_idx % 4) * 4;
    
#ifdef OPENCL_INPUT_INT
    FLOAT4 in0 = CONVERT_FLOAT4(convert_int4(vload4(0, input0 + offset0*isFull.x)));
    FLOAT4 in1 = CONVERT_FLOAT4(convert_int4(vload4(0, input1 + offset1*isFull.y)));
#else
    FLOAT4 in0 = vload4(0, input0 + offset0*isFull.x);
    FLOAT4 in1 = vload4(0, input1 + offset1*isFull.y);
#endif
    if(isFull.x == 0) {
        in0 = (FLOAT4)(in0.x, in0.x, in0.x, in0.x);
    }
    if(isFull.y == 0) {
        in1 = (FLOAT4)(in1.x, in1.x, in1.x, in1.x);
    }
#ifdef OPENCL_INPUT_INT
    FLOAT4 out = CONVERT_FLOAT4(convert_int4(OPERATOR));
#else
    FLOAT4 out = CONVERT_FLOAT4(OPERATOR);
#endif
    if(activationType == 1) {
        out = fmax(out, (FLOAT4)0);
    }
    vstore4(out, 0, output + dst_offset);
    if(w_idx == 0){
        int pad_offset = (((batch_idx*channel16+channe_out_idx)*shape.y+h_idx)*dst_width) * 16 + (channel_idx % 4) * 4;
        for(int i = 0; i < output_pad_left; ++i){
            vstore4((FLOAT4)0, 0, output + pad_offset + i * 16);
        }
        pad_offset += (shape.z + output_pad_left) * 16;
        for(int i = 0; i < output_pad_right; ++i){
            vstore4((FLOAT4)0, 0, output + pad_offset + i * 16);
        }
    }
}

__kernel void binary_buf_c16_c4_c16(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                         __global FLOAT* input0, __global FLOAT* input1, __global FLOAT* output,
                         __private const int4 shape,//[N,H,W,C]
                         __private const int2 isFull,
                         __private const int activationType,
                        __private const int input0_pad_left, __private const int input0_pad_right,
                        __private const int input1_pad_left, __private const int input1_pad_right,
                        __private const int output_pad_left, __private const int output_pad_right) {
    if (get_global_id(0) >= global_dim0 || get_global_id(1) >= global_dim1 || get_global_id(2) >= global_dim2) 
        return;
    const int channel4 = (shape.w + 3) / 4;
    const int channel16 = (shape.w + 15) / 16;
    const int w_idx = get_global_id(0) % shape.z;
    const int h_idx = get_global_id(0) / shape.z;
    const int batch_idx = get_global_id(2);
    const int channel_idx = get_global_id(1);
    const int src_width = shape.z + input0_pad_left + input0_pad_right;
    const int dst_width = shape.z + output_pad_left + output_pad_right;
    const int channe_out_idx = channel_idx >> 2;

    const int offset1 = (((batch_idx*channel4+channel_idx)*shape.y+h_idx)*shape.z+w_idx) * 4;
    const int offset0 = (((batch_idx*channel16+channe_out_idx)*shape.y+h_idx)*src_width+w_idx+input0_pad_left) * 16 + (channel_idx % 4) * 4;
    const int dst_offset =  (((batch_idx*channel16+channe_out_idx)*shape.y+h_idx)*dst_width+w_idx+output_pad_left) * 16 + (channel_idx % 4) * 4;
   
#ifdef OPENCL_INPUT_INT
    FLOAT4 in0 = CONVERT_FLOAT4(convert_int4(vload4(0, input0 + offset0*isFull.x)));
    FLOAT4 in1 = CONVERT_FLOAT4(convert_int4(vload4(0, input1 + offset1*isFull.y)));
#else
    FLOAT4 in0 = vload4(0, input0 + offset0*isFull.x);
    FLOAT4 in1 = vload4(0, input1 + offset1*isFull.y);
#endif
    if(isFull.x == 0) {
        in0 = (FLOAT4)(in0.x, in0.x, in0.x, in0.x);
    }
    if(isFull.y == 0) {
        in1 = (FLOAT4)(in1.x, in1.x, in1.x, in1.x);
    }
#ifdef OPENCL_INPUT_INT
    FLOAT4 out = CONVERT_FLOAT4(convert_int4(OPERATOR));
#else
    FLOAT4 out = CONVERT_FLOAT4(OPERATOR);
#endif
    if(activationType == 1) {
        out = fmax(out, (FLOAT4)0);
    }
    vstore4(out, 0, output + dst_offset);
    if(w_idx == 0){
        int pad_offset = (((batch_idx*channel16+channe_out_idx)*shape.y+h_idx)*dst_width) * 16 + (channel_idx % 4) * 4;
        for(int i = 0; i < output_pad_left; ++i){
            vstore4((FLOAT4)0, 0, output + pad_offset + i * 16);
        }
        pad_offset += (shape.z + output_pad_left) * 16;
        for(int i = 0; i < output_pad_right; ++i){
            vstore4((FLOAT4)0, 0, output + pad_offset + i * 16);
        }
    }
}



__kernel void prelu_buf_c4_c4(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                         __global FLOAT* input0, __global FLOAT* input1, __global FLOAT* output,
                         __private const int4 shape,//[N,H,W,C]
                         __private const int input0_pad_left, __private const int input0_pad_right,
                        __private const int output_pad_left, __private const int output_pad_right
                         ) {
    if (get_global_id(0) >= global_dim0 || get_global_id(1) >= global_dim1 || get_global_id(2) >= global_dim2) 
        return;
    const int channel4 = (shape.w + 3) / 4;
    const int w_idx = get_global_id(0) % shape.z;
    const int h_idx = get_global_id(0) / shape.z;
    const int batch_idx = get_global_id(2);
    const int channel_idx = get_global_id(1);
    
    const int offset0 = (((batch_idx*channel4+channel_idx)*shape.y+h_idx)*shape.z+w_idx) * 4;
    const int offset1 = channel_idx * 4;
#ifdef OPENCL_INPUT_INT
    FLOAT4 in0 = CONVERT_FLOAT4(convert_int4(vload4(0, input0 + offset0)));
    FLOAT4 in1 = CONVERT_FLOAT4(convert_int4(vload4(0, input1 + offset1)));
    FLOAT4 out = CONVERT_FLOAT4(convert_int4(OPERATOR));
#else
    FLOAT4 in0 = vload4(0, input0 + offset0);
    FLOAT4 in1 = vload4(0, input1 + offset1);
    FLOAT4 out = CONVERT_FLOAT4(OPERATOR);
#endif
    vstore4(out, 0, output + offset0);
}

__kernel void prelu_buf_c4_c16(__private int global_dim0, __private int global_dim1,__private int global_dim2,
                         __global FLOAT* input0, __global FLOAT* input1, __global FLOAT* output,
                         __private const int4 shape,//[N,H,W,C]
                         __private const int input0_pad_left, __private const int input0_pad_right,
                        __private const int output_pad_left, __private const int output_pad_right
                         ) {
    if (get_global_id(0) >= global_dim0 || get_global_id(1) >= global_dim1 || get_global_id(2) >= global_dim2) 
        return;
    const int channel4 = (shape.w + 3) / 4;
    const int channel16 = (shape.w + 15) / 16;
    const int w_idx = get_global_id(0) % shape.z;
    const int h_idx = get_global_id(0) / shape.z;
    const int batch_idx = get_global_id(2);
    const int channel_idx = get_global_id(1);
    const int dst_width = shape.z + output_pad_left + output_pad_right;
    const int channe_out_idx = channel_idx >> 2;
    
    const int offset0 = (((batch_idx*channel4+channel_idx)*shape.y+h_idx)*shape.z+w_idx) * 4;
    const int offset1 = channel_idx * 4;
    const int offset =  (((batch_idx*channel16+channe_out_idx)*shape.y+h_idx)*dst_width+w_idx+output_pad_left) * 16 + (channel_idx % 4) * 4;

#ifdef OPENCL_INPUT_INT
    FLOAT4 in0 = CONVERT_FLOAT4(convert_int4(vload4(0, input0 + offset0)));
    FLOAT4 in1 = CONVERT_FLOAT4(convert_int4(vload4(0, input1 + offset1)));
    FLOAT4 out = CONVERT_FLOAT4(convert_int4(OPERATOR));
#else
    FLOAT4 in0 = vload4(0, input0 + offset0);
    FLOAT4 in1 = vload4(0, input1 + offset1);
    FLOAT4 out = CONVERT_FLOAT4(OPERATOR);
#endif
    vstore4(out, 0, output + offset);
    if(w_idx == 0){
        int pad_offset = (((batch_idx*channel16+channe_out_idx)*shape.y+h_idx)*dst_width) * 16 + (channel_idx % 4) * 4;
        for(int i = 0; i < output_pad_left; ++i){
            vstore4((FLOAT4)0, 0, output + pad_offset + i * 16);
        }
        pad_offset += (shape.z + output_pad_left) * 16;
        for(int i = 0; i < output_pad_right; ++i){
            vstore4((FLOAT4)0, 0, output + pad_offset + i * 16);
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prelu_buf_c16_c16(__private int global_dim0, __private int global_dim1,__private int global_dim2,
                         __global FLOAT* input0, __global FLOAT* input1, __global FLOAT* output,
                         __private const int4 shape,//[N,H,W,C]
                        __private const int input0_pad_left, __private const int input0_pad_right,
                        __private const int output_pad_left, __private const int output_pad_right) {
    const int channel16 = (shape.w + 15) / 16;
    const int width_pack = (shape.z + 3) / 4;
    const int w_idx = (get_global_id(0) % width_pack) << 2;
    const int h_idx = get_global_id(0) / width_pack;
    const int batch_idx = get_global_id(2);
    const int channel_idx = get_group_id(1);
    const int sglid = get_sub_group_local_id();
    const int src_width = shape.z + input0_pad_left + input0_pad_right;
    const int dst_width = shape.z + output_pad_left + output_pad_right;

    const int offset0 = (((batch_idx*channel16+channel_idx)*shape.y+h_idx)*src_width+w_idx+input0_pad_left) * 16;
    const int offset1 = channel_idx * 16;
    const int offset =  (((batch_idx*channel16+channel_idx)*shape.y+h_idx)*dst_width+w_idx+output_pad_left) * 16;

#ifdef OPENCL_INPUT_INT
#ifdef MNN_SUPPORT_FP16
    FLOAT4 in0 = CONVERT_FLOAT4(convert_int4(as_half4(intel_sub_group_block_read_us4((__global ushort*)(input0 + offset0)))));
    FLOAT4 in1 = (FLOAT4)(convert_int(as_half(intel_sub_group_block_read_us((__global ushort*)(input1 + offset1)))));
#else
    FLOAT4 in0 = CONVERT_FLOAT4(convert_int4(as_float4(intel_sub_group_block_read4((__global uint*)(input0 + offset0)))));
    FLOAT4 in1 = (FLOAT4)(convert_int(as_float(intel_sub_group_block_read((__global uint*)(input1 + offset1)))));
#endif
    FLOAT4 out = CONVERT_FLOAT4(convert_int4(OPERATOR));
#else
#ifdef MNN_SUPPORT_FP16
    FLOAT4 in0 = as_half4(intel_sub_group_block_read_us4((__global ushort*)(input0 + offset0)));
    FLOAT4 in1 = (FLOAT4)(as_half(intel_sub_group_block_read_us((__global ushort*)(input1 + offset1))));
#else
    FLOAT4 in0 = as_float4(intel_sub_group_block_read4((__global uint*)(input0 + offset0)));
    FLOAT4 in1 = (FLOAT4)(as_float(intel_sub_group_block_read((__global uint*)(input1 + offset1))));
#endif
    
    FLOAT4 out = CONVERT_FLOAT4(OPERATOR);
#endif
    {
        if (w_idx + 4 > shape.z) {
            for (int i = 0; i < shape.z % 4; i++) {
                output[offset + i * 16 + sglid] = out[i];
            }
        }else{
#ifdef MNN_SUPPORT_FP16
            intel_sub_group_block_write_us4((__global ushort*)(output + offset), as_ushort4(out));
#else
            intel_sub_group_block_write4((__global uint*)(output + offset), as_uint4(out));
#endif
        }
    }
    if(w_idx == 0){
        int pad_offset = (((batch_idx*channel16+channel_idx)*shape.y+h_idx)*dst_width) * 16 + sglid;
        for(int i = 0; i < output_pad_left; ++i){
            output[pad_offset + i * 16] = 0;
        }
        pad_offset += (shape.z + output_pad_left) * 16;
        for(int i = 0; i < output_pad_right; ++i){
            output[pad_offset + i * 16] = 0;
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prelu_buf_c16_c4(__private int global_dim0, __private int global_dim1,__private int global_dim2,
                         __global FLOAT* input0, __global FLOAT* input1, __global FLOAT* output,
                         __private const int4 shape,//[N,H,W,C]
                        __private const int input0_pad_left, __private const int input0_pad_right,
                        __private const int output_pad_left, __private const int output_pad_right) {
    const int channel4 = (shape.w + 3) / 4;
    const int channel16 = (shape.w + 15) / 16;
    const int width_pack = (shape.z + 3) / 4;
    const int w_idx = (get_global_id(0) % width_pack) << 2;
    const int h_idx = get_global_id(0) / width_pack;
    const int batch_idx = get_global_id(2);
    const int channel_idx = get_group_id(1);
    const int sglid = get_sub_group_local_id();
    const int src_width = shape.z + input0_pad_left + input0_pad_right;
    const int width_height = shape.z * shape.y * 4;

    const int offset0 = (((batch_idx*channel16+channel_idx)*shape.y+h_idx)*src_width+w_idx+input0_pad_left) * 16;
    const int offset1 = channel_idx * 16;
    const int offset =  (((batch_idx*channel4+(channel_idx<<2))*shape.y+h_idx)*shape.z+w_idx) * 4;

#ifdef OPENCL_INPUT_INT
#ifdef MNN_SUPPORT_FP16
    FLOAT4 in0 = CONVERT_FLOAT4(convert_int4(as_half4(intel_sub_group_block_read_us4((__global ushort*)(input0 + offset0)))));
    FLOAT4 in1 = (FLOAT4)(convert_int(as_half(intel_sub_group_block_read_us((__global ushort*)(input1 + offset1)))));
#else
    FLOAT4 in0 = CONVERT_FLOAT4(convert_int4(as_float4(intel_sub_group_block_read4((__global uint*)(input0 + offset0)))));
    FLOAT4 in1 = (FLOAT4)(convert_int(as_float(intel_sub_group_block_read((__global uint*)(input1 + offset1)))));
#endif
    FLOAT4 out = CONVERT_FLOAT4(convert_int4(OPERATOR));
#else
#ifdef MNN_SUPPORT_FP16
    FLOAT4 in0 = as_half4(intel_sub_group_block_read_us4((__global ushort*)(input0 + offset0)));
    FLOAT4 in1 = (FLOAT4)(as_half(intel_sub_group_block_read_us((__global ushort*)(input1 + offset1))));
#else
    FLOAT4 in0 = as_float4(intel_sub_group_block_read4((__global uint*)(input0 + offset0)));
    FLOAT4 in1 = (FLOAT4)(as_float(intel_sub_group_block_read((__global uint*)(input1 + offset1))));
#endif
    
    FLOAT4 out = CONVERT_FLOAT4(OPERATOR);
#endif

    const int lid_x = sglid % 4;
    const int lid_y = sglid / 4;
    int block_size = w_idx + 4 > shape.z ? (shape.z % 4) : 4;
    for (int i = 0; i < block_size; i++) {
        output[offset + i * 4 + lid_y * width_height + lid_x] = out[i];
    }
}



__attribute__((intel_reqd_sub_group_size(16)))
__kernel void binary_buf_c16_c16_c16(__private int global_dim0, __private int global_dim1,__private int global_dim2,
                         __global FLOAT* input0, __global FLOAT* input1, __global FLOAT* output,
                         __private const int4 shape,//[N,H,W,C4]
                         __private const int2 isFull,
                         __private const int activationType,
                        __private const int input0_pad_left, __private const int input0_pad_right,
                        __private const int input1_pad_left, __private const int input1_pad_right,
                        __private const int output_pad_left, __private const int output_pad_right) {
    const int channel16 = (shape.w + 15) / 16;
    const int width_pack = (shape.z + 3) / 4;
    const int w_idx = (get_global_id(0) % width_pack) << 2;
    const int h_idx = get_global_id(0) / width_pack;
    const int batch_idx = get_global_id(2);
    const int channel_idx = get_group_id(1);
    const int sglid = get_sub_group_local_id();
    const int src0_width = shape.z + input0_pad_left + input0_pad_right;
    const int src1_width = shape.z + input1_pad_left + input1_pad_right;
    const int dst_width = shape.z + output_pad_left + output_pad_right;

    const int offset0 = (((batch_idx*channel16+channel_idx)*shape.y+h_idx)*src0_width+w_idx+input0_pad_left) * 16;
    const int offset1 = (((batch_idx*channel16+channel_idx)*shape.y+h_idx)*src1_width+w_idx+input1_pad_left) * 16;
    const int offset =  (((batch_idx*channel16+channel_idx)*shape.y+h_idx)*dst_width+w_idx+output_pad_left) * 16;

#ifdef OPENCL_INPUT_INT
#ifdef MNN_SUPPORT_FP16
    FLOAT4 in0 = isFull.x ? CONVERT_FLOAT4(convert_int4(as_half4(intel_sub_group_block_read_us4((__global ushort*)(input0 + offset0))))) : (FLOAT4)(convert_int(input0[0]));
    FLOAT4 in1 = isFull.y ? CONVERT_FLOAT4(convert_int4(as_half4(intel_sub_group_block_read_us4((__global ushort*)(input1 + offset1))))) : (FLOAT4)(convert_int(input1[0]));
#else
    FLOAT4 in0 = isFull.x ? CONVERT_FLOAT4(convert_int4(as_float4(intel_sub_group_block_read4((__global uint*)(input0 + offset0))))) : (FLOAT4)(convert_int(input0[0]));
    FLOAT4 in1 = isFull.y ? CONVERT_FLOAT4(convert_int4(as_float4(intel_sub_group_block_read4((__global uint*)(input1 + offset1))))) : (FLOAT4)(convert_int(input1[0]));
#endif
    FLOAT4 out = CONVERT_FLOAT4(convert_int4(OPERATOR));
#else
#ifdef MNN_SUPPORT_FP16
    FLOAT4 in0 = isFull.x ? as_half4(intel_sub_group_block_read_us4((__global ushort*)(input0 + offset0))) : (FLOAT4)(input0[0]);
    FLOAT4 in1 = isFull.y ? as_half4(intel_sub_group_block_read_us4((__global ushort*)(input1 + offset1))) : (FLOAT4)(input1[0]);
#else
    FLOAT4 in0 = isFull.x ? as_float4(intel_sub_group_block_read4((__global uint*)(input0 + offset0))) : (FLOAT4)(input0[0]);
    FLOAT4 in1 = isFull.y ? as_float4(intel_sub_group_block_read4((__global uint*)(input1 + offset1))) : (FLOAT4)(input1[0]);
#endif
    
    FLOAT4 out = CONVERT_FLOAT4(OPERATOR);
#endif
    if(activationType == 1) {
        out = fmax(out, (FLOAT4)0);
    }

    {
        if (w_idx + 4 > shape.z) {
            for (int i = 0; i < shape.z % 4; i++) {
                output[offset + i * 16 + sglid] = out[i];
            }
        }else{
#ifdef MNN_SUPPORT_FP16
            intel_sub_group_block_write_us4((__global ushort*)(output + offset), as_ushort4(out));
#else
            intel_sub_group_block_write4((__global uint*)(output + offset), as_uint4(out));
#endif
        }
    }
    if(w_idx == 0){
        int pad_offset = (((batch_idx*channel16+channel_idx)*shape.y+h_idx)*dst_width) * 16 + sglid;
        for(int i = 0; i < output_pad_left; ++i){
            output[pad_offset + i * 16] = 0;
        }
        pad_offset += (shape.z + output_pad_left) * 16;
        for(int i = 0; i < output_pad_right; ++i){
            output[pad_offset + i * 16] = 0;
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void binary_buf_c16_c16_c4(__private int global_dim0, __private int global_dim1,__private int global_dim2,
                         __global FLOAT* input0, __global FLOAT* input1, __global FLOAT* output,
                         __private const int4 shape,//[N,H,W,C4]
                         __private const int2 isFull,
                         __private const int activationType,
                        __private const int input0_pad_left, __private const int input0_pad_right,
                        __private const int input1_pad_left, __private const int input1_pad_right,
                        __private const int output_pad_left, __private const int output_pad_right) {
    const int channel16 = (shape.w + 15) / 16;
    const int channel4 = (shape.w + 3) / 4;
    const int width_pack = (shape.z + 3) / 4;
    const int w_idx = (get_global_id(0) % width_pack) << 2;
    const int h_idx = get_global_id(0) / width_pack;
    const int batch_idx = get_global_id(2);
    const int channel_idx = get_group_id(1);
    const int sglid = get_sub_group_local_id();
    const int src0_width = shape.z + input0_pad_left + input0_pad_right;
    const int src1_width = shape.z + input1_pad_left + input1_pad_right;
    const int width_height = shape.z * shape.y * 4;

    const int offset0 = (((batch_idx*channel16+channel_idx)*shape.y+h_idx)*src0_width+w_idx+input0_pad_left) * 16;
    const int offset1 = (((batch_idx*channel16+channel_idx)*shape.y+h_idx)*src1_width+w_idx+input1_pad_left) * 16;
    const int offset =  (((batch_idx*channel4+(channel_idx << 2))*shape.y+h_idx)*shape.z+w_idx) * 4;

#ifdef OPENCL_INPUT_INT
#ifdef MNN_SUPPORT_FP16
    FLOAT4 in0 = isFull.x ? CONVERT_FLOAT4(convert_int4(as_half4(intel_sub_group_block_read_us4((__global ushort*)(input0 + offset0))))) : (FLOAT4)(convert_int(input0[0]));
    FLOAT4 in1 = isFull.y ? CONVERT_FLOAT4(convert_int4(as_half4(intel_sub_group_block_read_us4((__global ushort*)(input1 + offset1))))) : (FLOAT4)(convert_int(input1[0]));
#else
    FLOAT4 in0 = isFull.x ? CONVERT_FLOAT4(convert_int4(as_float4(intel_sub_group_block_read4((__global uint*)(input0 + offset0))))) : (FLOAT4)(convert_int(input0[0]));
    FLOAT4 in1 = isFull.y ? CONVERT_FLOAT4(convert_int4(as_float4(intel_sub_group_block_read4((__global uint*)(input1 + offset1))))) : (FLOAT4)(convert_int(input1[0]));
#endif
    FLOAT4 out = CONVERT_FLOAT4(convert_int4(OPERATOR));
#else
#ifdef MNN_SUPPORT_FP16
    FLOAT4 in0 = isFull.x ? as_half4(intel_sub_group_block_read_us4((__global ushort*)(input0 + offset0))) : (FLOAT4)(input0[0]);
    FLOAT4 in1 = isFull.y ? as_half4(intel_sub_group_block_read_us4((__global ushort*)(input1 + offset1))) : (FLOAT4)(input1[0]);
#else
    FLOAT4 in0 = isFull.x ? as_float4(intel_sub_group_block_read4((__global uint*)(input0 + offset0))) : (FLOAT4)(input0[0]);
    FLOAT4 in1 = isFull.y ? as_float4(intel_sub_group_block_read4((__global uint*)(input1 + offset1))) : (FLOAT4)(input1[0]);
#endif
    
    FLOAT4 out = CONVERT_FLOAT4(OPERATOR);
#endif
    if(activationType == 1) {
        out = fmax(out, (FLOAT4)0);
    }

    const int lid_x = sglid % 4;
    const int lid_y = sglid / 4;
    int block_size = w_idx + 4 > shape.z ? (shape.z % 4) : 4;
    for (int i = 0; i < block_size; i++) {
        output[offset + i * 4 + lid_y * width_height + lid_x] = out[i];
    }
}
