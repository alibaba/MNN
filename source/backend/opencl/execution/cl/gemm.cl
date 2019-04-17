#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void gemm(__read_only image2d_t uInput, __read_only image2d_t uKernel, __write_only image2d_t uOutput,
                   __private const int width, __private const int height, __private const int multiLength) {
    const int pos_x = get_global_id(0);
    const int pos_y = get_global_id(1);
    const int pos_z = get_global_id(2);
    if (pos_x < width && pos_y < height) {
        FLOAT4 o0 = (FLOAT4)(0);
        FLOAT4 o1 = (FLOAT4)(0);
        FLOAT4 o2 = (FLOAT4)(0);
        FLOAT4 o3 = (FLOAT4)(0);
        int kenerlY   = pos_y + pos_z * height;
        int srcY      = pos_x + pos_z * width;

        for (int k = 0; k < multiLength; ++k) {
            int x0        = 4 * k + 0;
            int x1        = 4 * k + 1;
            int x2        = 4 * k + 2;
            int x3        = 4 * k + 3;
            FLOAT4 k0 = RI_F(uKernel, SAMPLER, (int2)(x0, kenerlY));
            FLOAT4 k1 = RI_F(uKernel, SAMPLER, (int2)(x1, kenerlY));
            FLOAT4 k2 = RI_F(uKernel, SAMPLER, (int2)(x2, kenerlY));
            FLOAT4 k3 = RI_F(uKernel, SAMPLER, (int2)(x3, kenerlY));

            FLOAT4 s0 = RI_F(uInput, SAMPLER, (int2)(x0, srcY));
            FLOAT4 s1 = RI_F(uInput, SAMPLER, (int2)(x1, srcY));
            FLOAT4 s2 = RI_F(uInput, SAMPLER, (int2)(x2, srcY));
            FLOAT4 s3 = RI_F(uInput, SAMPLER, (int2)(x3, srcY));

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

        WI_F(uOutput, (int2)(srcY, 4 * pos_y + 0), o0);
        WI_F(uOutput, (int2)(srcY, 4 * pos_y + 1), o1);
        WI_F(uOutput, (int2)(srcY, 4 * pos_y + 2), o2);
        WI_F(uOutput, (int2)(srcY, 4 * pos_y + 3), o3);
    }
}
