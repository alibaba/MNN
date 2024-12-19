#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_3_DIMS __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,
#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                     \
    }

#define MNN_DATA_FORMAT_NCHW 0
#define MNN_DATA_FORMAT_NHWC 1
#define MNN_DATA_FORMAT_NC4HW4 2
#define MNN_DATA_FORMAT_C4NHW4 3

#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)
#define OUTPUT_TYPE2 CAT(OUTPUT_TYPE, 2)
#define OUTPUT_TYPE3 CAT(OUTPUT_TYPE, 3)
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#ifdef SHARED_TO_CL
__kernel void gl_to_cl(GLOBAL_SIZE_3_DIMS
                                    __global uchar *input_ptr,
                                    #ifdef USE_IMAGE
                                    __write_only image2d_t output_ptr,
                                    #else
                                    __global OUTPUT_TYPE *output_ptr,
                                    #endif
                                    __private const int4 shape // N C H W
) {

    int wblock  = get_global_id(0);
    int cblock = get_global_id(1);
    int nh = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(wblock, cblock, nh);
    const int w = wblock << 2;
    const int h = nh % shape.z;
    const int c = cblock << 2;
    const int n = nh / shape.z;
    
    int idx = c * shape.w + w;    // c/4*w
    int idy = nh;    // n*h
    const int offset = idy * shape.w * 4;
    OUTPUT_TYPE4 in0 = CONVERT_OUTPUT4(vload4(idx, input_ptr + offset));
    OUTPUT_TYPE4 in1 = CONVERT_OUTPUT4(vload4(idx + 1, input_ptr + offset));
    OUTPUT_TYPE4 in2 = CONVERT_OUTPUT4(vload4(idx + 2, input_ptr + offset));
    OUTPUT_TYPE4 in3 = CONVERT_OUTPUT4(vload4(idx + 3, input_ptr + offset));

#ifdef USE_IMAGE
    WI_DATA(output_ptr, (int2)(idx, idy), in0);
    if(w + 1 >= shape.w) return;
    WI_DATA(output_ptr, (int2)(idx+1, idy), in1);
    if(w + 2 >= shape.w) return;
    WI_DATA(output_ptr, (int2)(idx+2, idy), in2);
    if(w + 3 >= shape.w) return;
    WI_DATA(output_ptr, (int2)(idx+3, idy), in3);
#else
    #if OUTPUT_FORMAT == MNN_DATA_FORMAT_NCHW
    int output_offset = ((n * shape.y + c) * shape.z + h) * shape.w + w;
    int stride = shape.z * shape.w;
    int remain = shape.w - w;
    if(remain >= 4){
        vstore4((OUTPUT_TYPE4)(in0.x, in1.x, in2.x, in3.x), 0, output_ptr + output_offset);
        if(c + 1 >= shape.y) return;
        vstore4((OUTPUT_TYPE4)(in0.y, in1.y, in2.y, in3.y), 0, output_ptr + output_offset + stride);
        if(c + 2 >= shape.y) return;
        vstore4((OUTPUT_TYPE4)(in0.z, in1.z, in2.z, in3.z), 0, output_ptr + output_offset + stride + stride);
        if(c + 3 >= shape.y) return;
        vstore4((OUTPUT_TYPE4)(in0.w, in1.w, in2.w, in3.w), 0, output_ptr + output_offset + stride + stride + stride);
    } else if(remain == 3){
        vstore3((OUTPUT_TYPE3)(in0.x, in1.x, in2.x), 0, output_ptr + output_offset);
        if(c + 1 >= shape.y) return;
        vstore3((OUTPUT_TYPE3)(in0.y, in1.y, in2.y), 0, output_ptr + output_offset + stride);
        if(c + 2 >= shape.y) return;
        vstore3((OUTPUT_TYPE3)(in0.z, in1.z, in2.z), 0, output_ptr + output_offset + stride + stride);
        if(c + 3 >= shape.y) return;
        vstore3((OUTPUT_TYPE3)(in0.w, in1.w, in2.w), 0, output_ptr + output_offset + stride + stride + stride);
    } else if(remain == 2){
        vstore2((OUTPUT_TYPE2)(in0.x, in1.x), 0, output_ptr + output_offset);
        if(c + 1 >= shape.y) return;
        vstore2((OUTPUT_TYPE2)(in0.y, in1.y), 0, output_ptr + output_offset + stride);
        if(c + 2 >= shape.y) return;
        vstore2((OUTPUT_TYPE2)(in0.z, in1.z), 0, output_ptr + output_offset + stride + stride);
        if(c + 3 >= shape.y) return;
        vstore2((OUTPUT_TYPE2)(in0.w, in1.w), 0, output_ptr + output_offset + stride + stride + stride);
    }else if(remain == 1){
        output_ptr[output_offset] = in0.x;
        if(c + 1 >= shape.y) return;
        output_ptr[output_offset + stride] = in0.y;
        if(c + 2 >= shape.y) return;
        output_ptr[output_offset + stride + stride] = in0.z;
        if(c + 3 >= shape.y) return;
        output_ptr[output_offset + stride + stride + stride] = in0.w;
    }
    #elif OUTPUT_FORMAT == MNN_DATA_FORMAT_NHWC
    int output_offset = ((n * shape.z + h) * shape.w + w) * shape.y + c;
    int remain = shape.y - c;
    if(remain >= 4){
        vstore4(CONVERT_OUTPUT4(in0), 0, output_ptr + output_offset);
        if(w + 1 >= shape.w) return;
        vstore4(CONVERT_OUTPUT4(in1), 0, output_ptr + output_offset + shape.y);
        if(w + 2 >= shape.w) return;
        vstore4(CONVERT_OUTPUT4(in2), 0, output_ptr + output_offset + shape.y + shape.y);
        if(w + 3 >= shape.w) return;
        vstore4(CONVERT_OUTPUT4(in3), 0, output_ptr + output_offset + shape.y + shape.y + shape.y);
    } else if(remain == 3){
        vstore3((OUTPUT_TYPE3)(in0.x, in0.y, in0.z), 0, output_ptr + output_offset);
        if(w + 1 >= shape.w) return;
        vstore3((OUTPUT_TYPE3)(in1.x, in1.y, in1.z), 0, output_ptr + output_offset + shape.y);
        if(w + 2 >= shape.w) return;
        vstore3((OUTPUT_TYPE3)(in2.x, in2.y, in2.z), 0, output_ptr + output_offset + shape.y + shape.y);
        if(w + 3 >= shape.w) return;
        vstore3((OUTPUT_TYPE3)(in3.x, in3.y, in3.z), 0, output_ptr + output_offset + shape.y + shape.y + shape.y);
    } else if(remain == 2){
        vstore2((OUTPUT_TYPE2)(in0.x, in0.y), 0, output_ptr + output_offset);
        if(w + 1 >= shape.w) return;
        vstore2((OUTPUT_TYPE2)(in1.x, in1.y), 0, output_ptr + output_offset + shape.y);
        if(w + 2 >= shape.w) return;
        vstore2((OUTPUT_TYPE2)(in2.x, in2.y), 0, output_ptr + output_offset + shape.y + shape.y);
        if(w + 3 >= shape.w) return;
        vstore2((OUTPUT_TYPE2)(in3.x, in3.y), 0, output_ptr + output_offset + shape.y + shape.y + shape.y);
    }else if(remain == 1){
        output_ptr[output_offset] = in0.x;
        if(w + 1 >= shape.w) return;
        output_ptr[output_offset + shape.y] = in1.x;
        if(w + 2 >= shape.w) return;
        output_ptr[output_offset + shape.y + shape.y] = in1.x;
        if(w + 3 >= shape.w) return;
        output_ptr[output_offset + shape.y + shape.y + shape.y] = in1.x;
    }
    #elif OUTPUT_FORMAT == MNN_DATA_FORMAT_NC4HW4
    int output_offset = (((cblock * shape.x + n) * shape.z + h) * shape.w + w) * 4;
    vstore4(in0, 0, output_ptr + output_offset);
    if(w + 1 >= shape.w) return;
    vstore4(in1, 0, output_ptr + output_offset + 4);
    if(w + 2 >= shape.w) return;
    vstore4(in2, 0, output_ptr + output_offset + 8);
    if(w + 3 >= shape.w) return;
    vstore4(in3, 0, output_ptr + output_offset + 12);
    #endif
#endif
}
#endif

#ifdef CL_TO_SHARED
__kernel void cl_to_gl(GLOBAL_SIZE_3_DIMS
                                    #ifdef USE_IMAGE
                                    __read_only image2d_t input_ptr,
                                    #else
                                    __global INPUT_TYPE *input_ptr,
                                    #endif
                                    __global uchar *output_ptr,
                                    __private const int4 shape // N C H W
) {

    int wblock  = get_global_id(0);
    int cblock = get_global_id(1);
    int nh = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(wblock, cblock, nh);
    const int w = wblock << 2;
    const int h = nh % shape.z;
    const int c = cblock << 2;
    const int n = nh / shape.z;
    
    int idx = c * shape.w + w;    // c/4*w
    int idy = nh;    // n*h
#ifdef USE_IMAGE
    INPUT_TYPE4 in0 = RI_DATA(input_ptr, SAMPLER, (int2)(idx, idy));
    INPUT_TYPE4 in1 = RI_DATA(input_ptr, SAMPLER, (int2)(idx+1, idy));
    INPUT_TYPE4 in2 = RI_DATA(input_ptr, SAMPLER, (int2)(idx+2, idy));
    INPUT_TYPE4 in3 = RI_DATA(input_ptr, SAMPLER, (int2)(idx+3, idy));
#else
    #if INPUT_FORMAT == MNN_DATA_FORMAT_NCHW
    int input_offset = ((n * shape.y + c) * shape.z + h) * shape.w + w;
    int stride = shape.z * shape.w;
    INPUT_TYPE4 tmp0, tmp1, tmp2, tmp3;
    tmp0 = vload4(0, input_ptr + input_offset);
    tmp1 = vload4(0, input_ptr + input_offset + stride);
    tmp2 = vload4(0, input_ptr + input_offset + stride + stride);
    tmp3 = vload4(0, input_ptr + input_offset + stride + stride + stride);
    INPUT_TYPE4 in0 = (INPUT_TYPE4)(tmp0.x, tmp1.x, tmp2.x, tmp3.x);
    INPUT_TYPE4 in1 = (INPUT_TYPE4)(tmp0.y, tmp1.y, tmp2.y, tmp3.y);
    INPUT_TYPE4 in2 = (INPUT_TYPE4)(tmp0.z, tmp1.z, tmp2.z, tmp3.z);
    INPUT_TYPE4 in3 = (INPUT_TYPE4)(tmp0.w, tmp1.w, tmp2.w, tmp3.w);
    #elif INPUT_FORMAT == MNN_DATA_FORMAT_NHWC
    int input_offset = ((n * shape.z + h) * shape.w + w) * shape.y + c;
    INPUT_TYPE4 in0 = vload4(0, input_ptr + input_offset);
    INPUT_TYPE4 in1 = vload4(0, input_ptr + input_offset + shape.y);
    INPUT_TYPE4 in2 = vload4(0, input_ptr + input_offset + shape.y + shape.y);
    INPUT_TYPE4 in3 = vload4(0, input_ptr + input_offset + shape.y + shape.y + shape.y);
    #elif INPUT_FORMAT == MNN_DATA_FORMAT_NC4HW4
    int input_offset = (((cblock * shape.x + n) * shape.z + h) * shape.w + w) * 4;
    INPUT_TYPE4 in0 = vload4(0, input_ptr + input_offset);
    INPUT_TYPE4 in1 = vload4(0, input_ptr + input_offset + 4);
    INPUT_TYPE4 in2 = vload4(0, input_ptr + input_offset + 8);
    INPUT_TYPE4 in3 = vload4(0, input_ptr + input_offset + 12);
    #endif
#endif
    const int offset = idy * shape.w * 4;
    vstore4(convert_uchar4(in0), idx, output_ptr + offset);
    if(w + 1 >= shape.w) return;
    vstore4(convert_uchar4(in1), idx+1, output_ptr + offset);
    if(w + 2 >= shape.w) return;
    vstore4(convert_uchar4(in2), idx+2, output_ptr + offset);
    if(w + 3 >= shape.w) return;
    vstore4(convert_uchar4(in3), idx+3, output_ptr + offset);
}
#endif
