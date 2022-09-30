#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__kernel void binary_buf(__private int global_dim0, __private int global_dim1,
                         __global FLOAT* input0, __global FLOAT* input1, __global FLOAT* output,
                         __private const int4 shape,//[N,H,W,C4]
                         __private const int2 isFull,
                         __private const int activationType) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));//NC4, HW
    
    if (pos.x < global_dim0 && pos.y < global_dim1) {
        int offset = pos.x * (shape.y*shape.z) + pos.y;
        FLOAT4 in0 = vload4(offset*isFull.x, input0);
        FLOAT4 in1 = vload4(offset*isFull.y, input1);
        if(isFull.x == 0) {
            in0 = (FLOAT4)(in0.x, in0.x, in0.x, in0.x);
        }
        if(isFull.y == 0) {
            in1 = (FLOAT4)(in1.x, in1.x, in1.x, in1.x);
        }
        FLOAT4 out = CONVERT_FLOAT4(OPERATOR);
        if(activationType == 1) {
            out = fmax(out, (FLOAT4)0);
        }
        vstore4(out, offset, output);
    }
}


__kernel void prelu_buf(__private int global_dim0, __private int global_dim1,
                         __global FLOAT* input0, __global FLOAT* input1, __global FLOAT* output,
                         __private const int4 shape//[N,H,W,C4]
                         ) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));//NC4, HW
    
    if (pos.x < global_dim0 && pos.y < global_dim1) {
        int offset = pos.x * (shape.y*shape.z) + pos.y;
        FLOAT4 in0 = vload4(offset, input0);
        FLOAT4 in1 = vload4(pos.x % shape.w, input1);
        FLOAT4 out = CONVERT_FLOAT4(OPERATOR);
        vstore4(out, offset, output);
    }
}
