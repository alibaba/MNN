#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#define PI 3.141592653589f

__kernel void binary_buf(__private int global_dim0, __private int global_dim1,
                         __global INPUT_TYPE* input0, __global INPUT_TYPE* input1, __global OUTPUT_TYPE* output,
                         __private const int4 shape,//[N,H,W,C4]
                         __private const int2 isFull,
                         __private const int activationType) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));//NC4, HW
    
    if (pos.x < global_dim0 && pos.y < global_dim1) {
    #ifdef WH_PACK4
        int offset = pos.x * (shape.y*shape.z/4) + pos.y;
        #ifdef A_SINGLE
        float data0 = input0[0];
        float16 in0_16 = (float16)data0;
        #else
        float16 in0_16 = convert_float16(vload16(offset, input0));
        #endif
        
        #ifdef B_SINGLE
        float data1 = input1[0];
        float16 in1_16 = (float16)data1;
        #else
        float16 in1_16 = convert_float16(vload16(offset, input1));
        #endif
        
        float16 out;
        float4 in0 = in0_16.s0123;
        float4 in1 = in1_16.s0123;
        out.s0123 = OPERATOR;
        
        in0 = in0_16.s4567;
        in1 = in1_16.s4567;
        out.s4567 = OPERATOR;
        
        in0 = in0_16.s89ab;
        in1 = in1_16.s89ab;
        out.s89ab = OPERATOR;
        
        in0 = in0_16.scdef;
        in1 = in1_16.scdef;
        out.scdef = OPERATOR;
        
        if(activationType == 1) {
            out = fmax(out, (float16)0);
        }
        vstore16(CONVERT_OUTPUT16(out), offset, output);
    #else
        int offset = pos.x * (shape.y*shape.z) + pos.y;
        #ifdef A_SINGLE
        float data0 = input0[0];
        float4 in0 = (float4)(data0, data0, data0, data0);
        #else
        float4 in0 = convert_float4(vload4(offset, input0));
        #endif
        
        #ifdef B_SINGLE
        float data1 = input1[0];
        float4 in1 = (float4)(data1, data1, data1, data1);
        #else
        float4 in1 = convert_float4(vload4(offset, input1));
        #endif
        
        float4 out = OPERATOR;
        
        if(activationType == 1) {
            out = fmax(out, (float4)0);
        }
        vstore4(CONVERT_OUTPUT4(out), offset, output);
    #endif
    }
}


__kernel void prelu_buf(__private int global_dim0, __private int global_dim1,
                         __global INPUT_TYPE* input0, __global INPUT_TYPE* input1, __global OUTPUT_TYPE* output,
                         __private const int4 shape//[N,H,W,C4]
                         ) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));//NC4, HW
    
    if (pos.x < global_dim0 && pos.y < global_dim1) {
        int offset = pos.x * (shape.y*shape.z) + pos.y;
        float4 in0 = convert_float4(vload4(offset, input0));
        float4 in1 = convert_float4(vload4(pos.x % shape.w, input1));
        float4 out = OPERATOR;
        vstore4(CONVERT_OUTPUT4(out), offset, output);
    }
}
