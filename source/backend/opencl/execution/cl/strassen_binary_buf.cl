#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__kernel void binary_cfunction_buf(__private int global_dim0, __private int global_dim1,
                         __global FLOAT* input0,
                         __private const int offsetC,
                         __private const int strideC,
                         __global FLOAT* input1, __global FLOAT* output,
                         __private const int width,//[offsetA, offsetB, offsetC, 0]
                         __private const int height//[strideA, strideB, strideC, 0]
) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));// [X/16, Y]
    
    if (pos.x < global_dim0 && pos.y < global_dim1) {
        int offset_11 = offsetC + pos.x * 8 + pos.y * strideC;
        int offset_12 = offset_11 + width;
        int offset_21 = offset_11 + strideC * height;
        int offset_22 = offset_21 + width;

        FLOAT8 in_11 = vload8(0, input0 + offset_11);
        FLOAT8 in_12 = vload8(0, input0 + offset_12);
        FLOAT8 in_21 = vload8(0, input0 + offset_21);
        FLOAT8 in_22 = vload8(0, input0 + offset_22);
        FLOAT8 in_cx = vload8(0, input1 + pos.x * 8 + pos.y * width);

        in_12 = in_12 + in_cx;
        in_21 = in_12 + in_21;
        in_12 = in_22 + in_12;
        in_22 = in_22 + in_21;
        in_12 = in_11 + in_12;

        vstore8(in_21, 0, output + offset_21);
        vstore8(in_22, 0, output + offset_22);
        vstore8(in_12, 0, output + offset_12);
    }
}

#ifndef OPERATOR
#define OPERATOR in0+in1
#endif

__kernel void binary_function_buf(__private int global_dim0, __private int global_dim1,
                         __global FLOAT* input0, __global FLOAT* input1, __global FLOAT* output,
                         __private const int4 baseOffsets,//[offsetA, offsetB, offsetC, 0]
                         __private const int4 strides//[strideA, strideB, strideC, 0]
) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));// [X/16, Y]
    
    if (pos.x < global_dim0 && pos.y < global_dim1) {
        const int baseOffsetA = baseOffsets.x;
        const int baseOffsetB = baseOffsets.y;
        const int baseOffsetC = baseOffsets.z;
        const int strideA = strides.x;
        const int strideB = strides.y;
        const int strideC = strides.z;
        
        
        int offsetA = pos.x * 8 + pos.y * VEC_H * strideA + baseOffsetA;
        int offsetB = pos.x * 8 + pos.y * VEC_H * strideB + baseOffsetB;
        int offsetC = pos.x * 8 + pos.y * VEC_H * strideC + baseOffsetC;

        {
            FLOAT8 in0 = vload8(0, input0 + offsetA);
            FLOAT8 in1 = vload8(0, input1 + offsetB);
            FLOAT8 out = OPERATOR;
            vstore8(out, 0, output + offsetC);
        }
        #if VEC_H >= 2
        {
            offsetA += strideA;
            offsetB += strideB;
            offsetC += strideC;
            FLOAT8 in0 = vload8(0, input0 + offsetA);
            FLOAT8 in1 = vload8(0, input1 + offsetB);
            FLOAT8 out = OPERATOR;
            vstore8(out, 0, output + offsetC);
        }
        #endif
        #if VEC_H == 4
        {
            offsetA += strideA;
            offsetB += strideB;
            offsetC += strideC;
            FLOAT8 in0 = vload8(0, input0 + offsetA);
            FLOAT8 in1 = vload8(0, input1 + offsetB);
            FLOAT8 out = OPERATOR;
            vstore8(out, 0, output + offsetC);
        }
        {
            offsetA += strideA;
            offsetB += strideB;
            offsetC += strideC;
            FLOAT8 in0 = vload8(0, input0 + offsetA);
            FLOAT8 in1 = vload8(0, input1 + offsetB);
            FLOAT8 out = OPERATOR;
            vstore8(out, 0, output + offsetC);
        }
        #endif
    }
}
