#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#define PI 3.141592653589f

__kernel void binary_buf(__private int global_dim0, __private int global_dim1,
                         __global INPUT_TYPE* input0, __global INPUT_TYPE* input1, __global OUTPUT_TYPE* output,
                         __private const int size,
                         __private const int activationType) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));//NCHW, 1
    
    if (pos.x < global_dim0 && pos.y < global_dim1) {
        int offset = pos.x << 2;
#ifdef PACK_LEAVE
        if(offset + 3 >= size){
            int remain = size - offset;
            float4 in0, in1;
            float* in0_ptr = (float*)&in0;
            float* in1_ptr = (float*)&in1;
            
            for(int i = 0; i < remain; ++i){
                #ifdef A_SINGLE
                in0_ptr[i] = (float)input0[0];
                #else
                in0_ptr[i] = (float)input0[offset + i];
                #endif
        
                #ifdef B_SINGLE
                in1_ptr[i] = (float)input1[0];
                #else
                in1_ptr[i] = (float)input1[offset + i];
                #endif
            }
            float4 out = OPERATOR;
            if(activationType == 1) {
                out = fmax(out, (float4)0);
            }
            float* out_ptr = (float*)&out;
            for(int i = 0; i < remain; ++i){
                output[offset + i] = (OUTPUT_TYPE)out_ptr[i];
            }
        }else {
#endif
            #ifdef A_SINGLE
            float data0 = input0[0];
            float4 in0 = (float4)(data0, data0, data0, data0);
            #else
            float4 in0 = convert_float4(vload4(0, input0 + offset));
            #endif
        
            #ifdef B_SINGLE
            float data1 = input1[0];
            float4 in1 = (float4)(data1, data1, data1, data1);
            #else
            float4 in1 = convert_float4(vload4(0, input1 + offset));
            #endif
            
            float4 out = OPERATOR;
        
            if(activationType == 1) {
                out = fmax(out, (float4)0);
            }
            vstore4(CONVERT_OUTPUT4(out), 0, output + offset);
#ifdef PACK_LEAVE
        }
#endif
    }
}


__kernel void prelu_buf(__private int global_dim0, __private int global_dim1,
                         __global INPUT_TYPE* input0, __global INPUT_TYPE* input1, __global OUTPUT_TYPE* output,
                         __private const int4 shape
                         ) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));//NC4, HW
                                 
    if (pos.x < global_dim0 && pos.y < global_dim1) {
        int b = pos.x / shape.w;
        int c = pos.x % shape.w;
        int offset = (b + c * shape.x) * (shape.y*shape.z) + pos.y;
        float4 in0 = convert_float4(vload4(offset, input0));
        float4 in1 = convert_float4(vload4(pos.x % shape.w, input1));
        float4 out = OPERATOR;
        vstore4(CONVERT_OUTPUT4(out), offset, output);
    }
}
