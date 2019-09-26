#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void gemm(__read_only image2d_t uInput, __read_only image2d_t uKernel, __write_only image2d_t uOutput,
                   __private const int width, __private const int height, __private const int multiLength, __private const int alpha2) {
    
    int2 pos = (int2)(get_global_id(0), get_global_id(1)); 
    if (pos.x < width*height && pos.y < alpha2) {
        
        const int pos_x = pos.x % width;
        const int pos_y = pos.x / width;
        const int pos_z = pos.y;

        FLOAT4 o0 = (FLOAT4)(0);
        FLOAT4 o1 = (FLOAT4)(0);
        FLOAT4 o2 = (FLOAT4)(0);
        FLOAT4 o3 = (FLOAT4)(0);
        int kenerlY   = mad24(pos_z, height, pos_y);
        int srcY      = mad24(pos_z, width, pos_x);

        for (int k = 0; k < multiLength; ++k) {
            __private int index = mul24(k, 4);
            FLOAT4 k0 = RI_F(uKernel, SAMPLER, (int2)(index, kenerlY));
            FLOAT4 k1 = RI_F(uKernel, SAMPLER, (int2)(index+1, kenerlY));
            FLOAT4 k2 = RI_F(uKernel, SAMPLER, (int2)(index+2, kenerlY));
            FLOAT4 k3 = RI_F(uKernel, SAMPLER, (int2)(index+3, kenerlY));

            FLOAT4 s0 = RI_F(uInput, SAMPLER, (int2)(index, srcY));
            FLOAT4 s1 = RI_F(uInput, SAMPLER, (int2)(index+1, srcY));
            FLOAT4 s2 = RI_F(uInput, SAMPLER, (int2)(index+2, srcY));
            FLOAT4 s3 = RI_F(uInput, SAMPLER, (int2)(index+3, srcY));

            o0 = mad(s0.x, k0, o0);
            o0 = mad(s0.y, k1, o0);
            o0 = mad(s0.z, k2, o0);
            o0 = mad(s0.w, k3, o0);

            o1 = mad(s1.x, k0, o1);
            o1 = mad(s1.y, k1, o1);
            o1 = mad(s1.z, k2, o1);
            o1 = mad(s1.w, k3, o1);

            o2 = mad(s2.x, k0, o2);
            o2 = mad(s2.y, k1, o2);
            o2 = mad(s2.z, k2, o2);
            o2 = mad(s2.w, k3, o2);

            o3 = mad(s3.x, k0, o3);
            o3 = mad(s3.y, k1, o3);
            o3 = mad(s3.z, k2, o3);
            o3 = mad(s3.w, k3, o3);
        }

        __private int out_y_idx = mul24(pos_y, 4);
        WI_F(uOutput, (int2)(srcY, out_y_idx), o0);
        WI_F(uOutput, (int2)(srcY, out_y_idx + 1), o1);
        WI_F(uOutput, (int2)(srcY, out_y_idx + 2), o2);
        WI_F(uOutput, (int2)(srcY, out_y_idx + 3), o3);
    }
}
