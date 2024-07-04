#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__kernel void splitgelu_buf(__private int global_dim0, __private int global_dim1, __private int global_dim2,
                        __global const FLOAT * input,
                        #ifdef DOUBLE_INPUTS
                        __global const FLOAT * input1,
                        #endif
                        __global FLOAT * output,
                        __private const int4 shape
){
    int3 pos = (int3)(get_global_id(0), get_global_id(1), get_global_id(2));
    if (pos.x < global_dim0 && pos.y < global_dim1 && pos.z < global_dim2) {
        const int b   = pos.x;
        const int c_4 = pos.y;

// The product of W and H is a multiple of 4
#ifdef WH_4
        const int hw_4  = pos.z;

        const int channel_4 = (shape.y + 3) >> 2;
        const int area_4 = (shape.z + 3) >> 2;
        const int in_offset = ((b * channel_4 + c_4) * area_4 * 2 + hw_4) * 16;
        const int out_offset = ((b * channel_4 + c_4) * area_4 + hw_4) * 16;

        float16 valueL = convert_float16(vload16(0, input + in_offset));
        float16 valueR = convert_float16(vload16(area_4, input + in_offset));

        #ifdef DOUBLE_INPUTS
        float4 valueConstL = convert_float4(vload4(hw, input1));
        float4 valueConstR = convert_float4(vload4(area_4+hw, input1));
        valueL += (float16)((float4)valueConstL.x, (float4)valueConstL.y, (float4)valueConstL.z, (float4)valueConstL.w);
        valueR += (float16)((float4)valueConstR.x, (float4)valueConstR.y, (float4)valueConstR.z, (float4)valueConstR.w);
        #endif
        float16 out = (erf(valueR * (float16)0.7071067932881648) + (float16)1.0) * valueR * (float16)0.5;
        out *= valueL;
        vstore16(CONVERT_FLOAT16(out), 0, output + out_offset);
#else
        const int hw  = pos.z;
        
        const int channel_4 = (shape.y + 3) >> 2;
        const int in_offset = ((b * channel_4 + c_4) * shape.z * 2 + hw) * 4;
        const int out_offset = ((b * channel_4 + c_4) * shape.z + hw) * 4;
        
        float4 valueL = convert_float4(vload4(0, input + in_offset));
        float4 valueR = convert_float4(vload4(shape.z, input + in_offset));

        #ifdef DOUBLE_INPUTS
        float valueConstL = input1[hw];
        float valueConstR = input1[shape.z+hw];
        valueL += (float4)valueConstL;
        valueR += (float4)valueConstR;
        #endif
        float4 out = (erf(valueR * (float4)0.7071067932881648) + (float4)1.0) * valueR * (float4)0.5;
        out *= valueL;
        vstore4(CONVERT_FLOAT4(out), 0, output + out_offset);
#endif
    }
}
