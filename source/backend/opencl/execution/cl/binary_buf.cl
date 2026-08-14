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
            #ifdef INT_COMPUTE_MOD
            int4 in0 = (int4)0, in1 = (int4)1;
            #ifdef A_SINGLE
            in0 = (int4)((int)input0[0]);
            #else
            in0.x = (int)input0[offset];
            if(remain > 1) { in0.y = (int)input0[offset + 1]; }
            if(remain > 2) { in0.z = (int)input0[offset + 2]; }
            if(remain > 3) { in0.w = (int)input0[offset + 3]; }
            #endif
            #ifdef B_SINGLE
            in1 = (int4)((int)input1[0]);
            #else
            in1.x = (int)input1[offset];
            if(remain > 1) { in1.y = (int)input1[offset + 1]; }
            if(remain > 2) { in1.z = (int)input1[offset + 2]; }
            if(remain > 3) { in1.w = (int)input1[offset + 3]; }
            #endif
            int4 out = in0 % in1;
            out = ((out < (int4)0 && in1 > (int4)0) || (out > (int4)0 && in1 < (int4)0)) ? out + in1 : out;
            if(activationType == 1) {
                out = out > 0 ? out : 0;
            }
            output[offset] = (OUTPUT_TYPE)out.x;
            if(remain > 1) { output[offset + 1] = (OUTPUT_TYPE)out.y; }
            if(remain > 2) { output[offset + 2] = (OUTPUT_TYPE)out.z; }
            if(remain > 3) { output[offset + 3] = (OUTPUT_TYPE)out.w; }
            #else
            float4 in0 = (float4)0, in1 = (float4)0;
            #ifdef A_SINGLE
            in0 = (float4)((float)input0[0]);
            #else
            in0.x = (float)input0[offset];
            if(remain > 1) { in0.y = (float)input0[offset + 1]; }
            if(remain > 2) { in0.z = (float)input0[offset + 2]; }
            if(remain > 3) { in0.w = (float)input0[offset + 3]; }
            #endif
            #ifdef B_SINGLE
            in1 = (float4)((float)input1[0]);
            #else
            in1.x = (float)input1[offset];
            if(remain > 1) { in1.y = (float)input1[offset + 1]; }
            if(remain > 2) { in1.z = (float)input1[offset + 2]; }
            if(remain > 3) { in1.w = (float)input1[offset + 3]; }
            #endif
            float4 out = OPERATOR;
            if(activationType == 1) {
                out = fmax(out, (float4)0);
            }
            output[offset] = (OUTPUT_TYPE)out.x;
            if(remain > 1) { output[offset + 1] = (OUTPUT_TYPE)out.y; }
            if(remain > 2) { output[offset + 2] = (OUTPUT_TYPE)out.z; }
            if(remain > 3) { output[offset + 3] = (OUTPUT_TYPE)out.w; }
            #endif
        }else {
#endif
        #ifdef INT_COMPUTE_MOD
            #ifdef A_SINGLE
            int data0 = input0[0];
            int4 in0 = (int4)(data0, data0, data0, data0);
            #else
            int4 in0 = convert_int4(vload4(0, input0 + offset));
            #endif
        
            #ifdef B_SINGLE
            int data1 = input1[0];
            int4 in1 = (int4)(data1, data1, data1, data1);
            #else
            int4 in1 = convert_int4(vload4(0, input1 + offset));
            #endif
            
            int4 out = in0 % in1;
            out = ((out < (int4)0 && in1 > (int4)0) || (out > (int4)0 && in1 < (int4)0)) ? out + in1 : out;
        
            if(activationType == 1) {
                out = out > 0 ? out : 1;
            }
            vstore4(CONVERT_OUTPUT4(out), 0, output + offset);
        #else
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
        #endif
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
