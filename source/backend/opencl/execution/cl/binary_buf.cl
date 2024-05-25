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
        int offset = pos.x * (shape.y*shape.z) + pos.y;
        
        float4 in0 = convert_float4(vload4(offset*isFull.x, input0));
        float4 in1 = convert_float4(vload4(offset*isFull.y, input1));
        if(isFull.x == 0) {
            in0 = (float4)(in0.x, in0.x, in0.x, in0.x);
        }
        if(isFull.y == 0) {
            in1 = (float4)(in1.x, in1.x, in1.x, in1.x);
        }
        
        float4 out = OPERATOR;
        
        if(activationType == 1) {
            out = fmax(out, (float4)0);
        }
        vstore4(CONVERT_OUTPUT4(out), offset, output);
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
