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

__kernel void gemmWinograd(__read_only image2d_t uInput, __read_only image2d_t uKernel, __write_only image2d_t uOutput,
                   __private const int unitWidth, __private const int unitHeight, __private const int dstChannelC4, __private const int multiLength, __private const int alpha2) {
    
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    const int unitWidth4 = (unitWidth + 3) / 4;
    if (pos.x < unitWidth4 * unitHeight && pos.y < alpha2 * dstChannelC4) {
        
        const int pos_x = pos.x % unitWidth4;
        const int pos_y = pos.x / unitWidth4;
        const int pos_z = pos.y % dstChannelC4;
        const int pos_w = pos.y / dstChannelC4;

        FLOAT4 o0 = (FLOAT4)(0);
        FLOAT4 o1 = (FLOAT4)(0);
        FLOAT4 o2 = (FLOAT4)(0);
        FLOAT4 o3 = (FLOAT4)(0);
        int srcY = mad24(pos_w, unitHeight, pos_y);
        int srcX = pos_x << 2;

        for (int k = 0; k < multiLength; ++k) {
            __private int index = mul24(k, 4);
            __private int x_offset = mul24(k, unitWidth);
            FLOAT4 k0 = RI_F(uKernel, SAMPLER, (int2)(index, pos.y));
            FLOAT4 k1 = RI_F(uKernel, SAMPLER, (int2)(index + 1, pos.y));
            FLOAT4 k2 = RI_F(uKernel, SAMPLER, (int2)(index + 2, pos.y));
            FLOAT4 k3 = RI_F(uKernel, SAMPLER, (int2)(index + 3, pos.y));

            FLOAT4 s0 = RI_F(uInput, SAMPLER, (int2)(srcX + x_offset, srcY));
            FLOAT4 s1 = RI_F(uInput, SAMPLER, (int2)(srcX + x_offset + 1, srcY));
            FLOAT4 s2 = RI_F(uInput, SAMPLER, (int2)(srcX + x_offset + 2, srcY));
            FLOAT4 s3 = RI_F(uInput, SAMPLER, (int2)(srcX + x_offset + 3, srcY));

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

        __private int out_y_idx = mad24(pos_z, unitHeight, pos_y);
        __private int out_x_idx = mad24(pos_w, unitWidth, srcX);
        const int remain = unitWidth - srcX;
        if(remain >= 4){
            WI_F(uOutput, (int2)(out_x_idx, out_y_idx), o0);
            WI_F(uOutput, (int2)(out_x_idx + 1, out_y_idx), o1);
            WI_F(uOutput, (int2)(out_x_idx + 2, out_y_idx), o2);
            WI_F(uOutput, (int2)(out_x_idx + 3, out_y_idx), o3);
        }else if(remain == 3){
            WI_F(uOutput, (int2)(out_x_idx, out_y_idx), o0);
            WI_F(uOutput, (int2)(out_x_idx + 1, out_y_idx), o1);
            WI_F(uOutput, (int2)(out_x_idx + 2, out_y_idx), o2);
        }else if(remain == 2){
            WI_F(uOutput, (int2)(out_x_idx, out_y_idx), o0);
            WI_F(uOutput, (int2)(out_x_idx + 1, out_y_idx), o1);
        }else if(remain == 1){
            WI_F(uOutput, (int2)(out_x_idx, out_y_idx), o0);
        }
    }
}


__kernel void gemmWinogradW2(__read_only image2d_t uInput, __read_only image2d_t uKernel, __write_only image2d_t uOutput,
                   __private const int unitWidth, __private const int unitHeight, __private const int dstChannelC4, __private const int multiLength, __private const int alpha2) {
    
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    const int unitWidth8 = (unitWidth + 7) / 8;
    if (pos.x < unitWidth8 * unitHeight && pos.y < alpha2 * dstChannelC4) {
        
        const int pos_x = pos.x % unitWidth8;
        const int pos_y = pos.x / unitWidth8;
        const int pos_z = pos.y % dstChannelC4;
        const int pos_w = pos.y / dstChannelC4;

        FLOAT4 o0 = (FLOAT4)(0);
        FLOAT4 o1 = (FLOAT4)(0);
        FLOAT4 o2 = (FLOAT4)(0);
        FLOAT4 o3 = (FLOAT4)(0);
        FLOAT4 o4 = (FLOAT4)(0);
        FLOAT4 o5 = (FLOAT4)(0);
        FLOAT4 o6 = (FLOAT4)(0);
        FLOAT4 o7 = (FLOAT4)(0);
        int srcY = mad24(pos_w, unitHeight, pos_y);
        int srcX = pos_x << 3;

        for (int k = 0; k < multiLength; ++k) {
            __private int index = mul24(k, 4);
            __private int x_offset = mul24(k, unitWidth);
            FLOAT4 k0 = RI_F(uKernel, SAMPLER, (int2)(index, pos.y));
            FLOAT4 k1 = RI_F(uKernel, SAMPLER, (int2)(index + 1, pos.y));
            FLOAT4 k2 = RI_F(uKernel, SAMPLER, (int2)(index + 2, pos.y));
            FLOAT4 k3 = RI_F(uKernel, SAMPLER, (int2)(index + 3, pos.y));

            FLOAT4 s0 = RI_F(uInput, SAMPLER, (int2)(srcX + x_offset, srcY));
            FLOAT4 s1 = RI_F(uInput, SAMPLER, (int2)(srcX + x_offset + 1, srcY));
            FLOAT4 s2 = RI_F(uInput, SAMPLER, (int2)(srcX + x_offset + 2, srcY));
            FLOAT4 s3 = RI_F(uInput, SAMPLER, (int2)(srcX + x_offset + 3, srcY));
            FLOAT4 s4 = RI_F(uInput, SAMPLER, (int2)(srcX + x_offset + 4, srcY));
            FLOAT4 s5 = RI_F(uInput, SAMPLER, (int2)(srcX + x_offset + 5, srcY));
            FLOAT4 s6 = RI_F(uInput, SAMPLER, (int2)(srcX + x_offset + 6, srcY));
            FLOAT4 s7 = RI_F(uInput, SAMPLER, (int2)(srcX + x_offset + 7, srcY));

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
            
            o4 = mad(s4.x, k0, o4);
            o4 = mad(s4.y, k1, o4);
            o4 = mad(s4.z, k2, o4);
            o4 = mad(s4.w, k3, o4);

            o5 = mad(s5.x, k0, o5);
            o5 = mad(s5.y, k1, o5);
            o5 = mad(s5.z, k2, o5);
            o5 = mad(s5.w, k3, o5);

            o6 = mad(s6.x, k0, o6);
            o6 = mad(s6.y, k1, o6);
            o6 = mad(s6.z, k2, o6);
            o6 = mad(s6.w, k3, o6);

            o7 = mad(s7.x, k0, o7);
            o7 = mad(s7.y, k1, o7);
            o7 = mad(s7.z, k2, o7);
            o7 = mad(s7.w, k3, o7);
        }

        __private int out_y_idx = mad24(pos_z, unitHeight, pos_y);
        __private int out_x_idx = mad24(pos_w, unitWidth, srcX);
        const int remain = unitWidth - srcX;
        if(remain >= 8){
            WI_F(uOutput, (int2)(out_x_idx, out_y_idx), o0);
            WI_F(uOutput, (int2)(out_x_idx + 1, out_y_idx), o1);
            WI_F(uOutput, (int2)(out_x_idx + 2, out_y_idx), o2);
            WI_F(uOutput, (int2)(out_x_idx + 3, out_y_idx), o3);
            WI_F(uOutput, (int2)(out_x_idx + 4, out_y_idx), o4);
            WI_F(uOutput, (int2)(out_x_idx + 5, out_y_idx), o5);
            WI_F(uOutput, (int2)(out_x_idx + 6, out_y_idx), o6);
            WI_F(uOutput, (int2)(out_x_idx + 7, out_y_idx), o7);
        }else if(remain == 7){
            WI_F(uOutput, (int2)(out_x_idx, out_y_idx), o0);
            WI_F(uOutput, (int2)(out_x_idx + 1, out_y_idx), o1);
            WI_F(uOutput, (int2)(out_x_idx + 2, out_y_idx), o2);
            WI_F(uOutput, (int2)(out_x_idx + 3, out_y_idx), o3);
            WI_F(uOutput, (int2)(out_x_idx + 4, out_y_idx), o4);
            WI_F(uOutput, (int2)(out_x_idx + 5, out_y_idx), o5);
            WI_F(uOutput, (int2)(out_x_idx + 6, out_y_idx), o6);
        }else if(remain == 6){
            WI_F(uOutput, (int2)(out_x_idx, out_y_idx), o0);
            WI_F(uOutput, (int2)(out_x_idx + 1, out_y_idx), o1);
            WI_F(uOutput, (int2)(out_x_idx + 2, out_y_idx), o2);
            WI_F(uOutput, (int2)(out_x_idx + 3, out_y_idx), o3);
            WI_F(uOutput, (int2)(out_x_idx + 4, out_y_idx), o4);
            WI_F(uOutput, (int2)(out_x_idx + 5, out_y_idx), o5);
        }else if(remain == 5){
            WI_F(uOutput, (int2)(out_x_idx, out_y_idx), o0);
            WI_F(uOutput, (int2)(out_x_idx + 1, out_y_idx), o1);
            WI_F(uOutput, (int2)(out_x_idx + 2, out_y_idx), o2);
            WI_F(uOutput, (int2)(out_x_idx + 3, out_y_idx), o3);
            WI_F(uOutput, (int2)(out_x_idx + 4, out_y_idx), o4);
        }else if(remain == 4){
            WI_F(uOutput, (int2)(out_x_idx, out_y_idx), o0);
            WI_F(uOutput, (int2)(out_x_idx + 1, out_y_idx), o1);
            WI_F(uOutput, (int2)(out_x_idx + 2, out_y_idx), o2);
            WI_F(uOutput, (int2)(out_x_idx + 3, out_y_idx), o3);
        }else if(remain == 3){
            WI_F(uOutput, (int2)(out_x_idx, out_y_idx), o0);
            WI_F(uOutput, (int2)(out_x_idx + 1, out_y_idx), o1);
            WI_F(uOutput, (int2)(out_x_idx + 2, out_y_idx), o2);
        }else if(remain == 2){
            WI_F(uOutput, (int2)(out_x_idx, out_y_idx), o0);
            WI_F(uOutput, (int2)(out_x_idx + 1, out_y_idx), o1);
        }else if(remain == 1){
            WI_F(uOutput, (int2)(out_x_idx, out_y_idx), o0);
        }
    }
}
