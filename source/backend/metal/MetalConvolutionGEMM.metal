//
//  MetalConvolutionGEMM.metal
//  MNN
//
//  Created by MNN on 2019/02/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalConvolutionActivation.metal"

using namespace metal;
using namespace MNN;

struct conv_im2col_cst {
    int input_width;
    int input_height;
    int input_size;
    int input_slice;
    int output_width;
    int output_height;
    int output_size;
    int output_slice;
    int batch;
    
    int kernel_x;
    int kernel_y;
    int kernel_size;
    int stride_x;
    int stride_y;
    int pad_x;
    int pad_y;
    int dilation_x;
    int dilation_y;
    conv_activation_type activation;
};

kernel void conv_im2col(const device ftype4 *im         [[buffer(0)]],
                        device ftype4 *cols             [[buffer(1)]],
                        constant conv_im2col_cst& cst   [[buffer(2)]],
                        uint3 gid                       [[thread_position_in_grid]]) {
    auto z = gid.z % cst.input_slice;
    auto b = gid.z / cst.input_slice;
    if ((int)gid.x < cst.output_width && (int)gid.y < cst.output_height && (int)b < cst.batch) {
        int offset_x = gid.x * cst.stride_x - cst.pad_x;
        int offset_y = gid.y * cst.stride_y - cst.pad_y;
        int index = b * cst.output_size + gid.y * cst.output_width + gid.x;
        int cols_y = index / 4;
        int cols_x = index % 4 + z * cst.kernel_size * 4;
        
        auto xy_cols = cols + cols_y * cst.kernel_size * cst.input_slice * 4 + cols_x;
        auto xy_im   = im + b * cst.input_size * cst.input_slice + z * cst.input_size;
        for (int ky = 0, src_y = offset_y; ky < cst.kernel_y; ky++, src_y += cst.dilation_y) {
            for (int kx = 0, src_x = offset_x; kx < cst.kernel_x; kx++, src_x += cst.dilation_x) {
                auto pad = src_x < 0 || src_y < 0 || src_x >= cst.input_width || src_y >= cst.input_height;
                xy_cols[(ky * cst.kernel_x + kx) * 4] = pad ? 0 : xy_im[src_y * cst.input_width + src_x];
            }
        }
    }
}

kernel void qntconv_im2col(const device ftype4 *im          [[buffer(0)]],
                           device char4 *cols               [[buffer(1)]],
                           constant conv_im2col_cst& cst    [[buffer(2)]],
                           constant float& scale            [[buffer(3)]],
                           constant int2& range             [[buffer(4)]],
                           uint3 gid                        [[thread_position_in_grid]]) {
    auto z = gid.z % cst.input_slice;
    auto b = gid.z / cst.input_slice;
    if ((int)gid.x < cst.output_width && (int)gid.y < cst.output_height && (int)b < cst.batch) {
        int offset_x = gid.x * cst.stride_x - cst.pad_x;
        int offset_y = gid.y * cst.stride_y - cst.pad_y;
        int index = b * cst.output_size + gid.y * cst.output_width + gid.x;
        int cols_y = index / 4;
        int cols_x = index % 4 + z * cst.kernel_size * 4;
        
        auto xy_cols = cols + cols_y * cst.kernel_size * cst.input_slice * 4 + cols_x;
        auto xy_im   = im + b * cst.input_size * cst.input_slice + z * cst.input_size;
        for (int ky = 0, src_y = offset_y; ky < cst.kernel_y; ky++, src_y += cst.dilation_y) {
            for (int kx = 0, src_x = offset_x; kx < cst.kernel_x; kx++, src_x += cst.dilation_x) {
                auto pad = src_x < 0 || src_y < 0 || src_x >= cst.input_width || src_y >= cst.input_height;
                xy_cols[(ky * cst.kernel_x + kx) * 4] = pad ? 0 : char4(clamp(int4(round(float4(xy_im[src_y * cst.input_width + src_x]) * scale)), range.x, range.y));
            }
        }
    }
}

kernel void conv_col2im(const device ftype4 *cols       [[buffer(0)]],
                        device ftype4 *im               [[buffer(1)]],
                        const device ftype4 *biasTerms  [[buffer(2)]],
                        constant conv_im2col_cst& cst   [[buffer(3)]],
                        uint3 gid                       [[thread_position_in_grid]]) {
    auto z = gid.z % cst.output_slice;
    auto b = gid.z / cst.output_slice;
    if ((int)gid.x < cst.output_width && (int)gid.y < cst.output_height && (int)b < cst.batch) {
        int index = b * cst.output_size + gid.y * cst.output_width + gid.x;
        auto src_x = index / 4;
        auto src_y = index % 4 + z * 4;
        auto src_y_stride = UP_DIV(cst.output_size * cst.batch, 4);
        
        auto v = cols[(int)src_y * src_y_stride + (int)src_x] + biasTerms[(int)z];
        im[(int)gid.z * cst.output_size + (int)gid.y * cst.output_width + (int)gid.x] = activate(v, cst.activation);
    }
}

kernel void qntconv_col2im(const device float4 *cols        [[buffer(0)]],
                           device ftype4 *im                [[buffer(1)]],
                           const device ftype4 *biasTerms   [[buffer(2)]],
                           constant conv_im2col_cst& cst    [[buffer(3)]],
                           const device float4 *alpha       [[buffer(4)]],
                           uint3 gid                        [[thread_position_in_grid]]) {
    auto z = gid.z % cst.output_slice;
    auto b = gid.z / cst.output_slice;
    if ((int)gid.x < cst.output_width && (int)gid.y < cst.output_height && (int)b < cst.batch) {
        int index = b * cst.output_size + gid.y * cst.output_width + gid.x;
        auto src_x = index / 4;
        auto src_y = index % 4 + z * 4;
        auto src_y_stride = UP_DIV(cst.output_size * cst.batch, 4);
        
        auto a4 = alpha[(int)z];
        auto b4 = float4(biasTerms[(int)z]);
        auto v = ftype4(float4(cols[(int)src_y * src_y_stride + (int)src_x]) * a4 + b4);
        im[(int)gid.z * cst.output_size + (int)gid.y * cst.output_width + (int)gid.x] = activate(v, cst.activation);
    }
}

struct matmul4x4_const {
    int output_width;
    int output_height;
    int multi_length;
    int group;
};

template <typename IType, typename OType>
static inline void matmul4x4_template(const device IType *in,
                                      device OType *out,
                                      const device IType *kt,
                                      constant matmul4x4_const &cst,
                                      uint3 gid) {
    if ((int)gid.x < cst.output_width && (int)gid.y < cst.output_height) {
        auto ky = (int)gid.y + (int)gid.z * cst.output_height;
        auto iy = (int)gid.x + (int)gid.z * cst.output_width;
        auto off_in  = in  + iy * cst.multi_length;
        auto off_wt  = kt  + ky * cst.multi_length;
        auto off_out = out + iy + 4 * (int)gid.y * cst.output_width * cst.group;
        
        float4 result0 = 0, result1 = 0, result2 = 0, result3 = 0;
        for (int k = 0; k < cst.multi_length; ++k) {
            auto w4x4 = float4x4(off_wt[k]);
            auto i4x4 = float4x4(off_in[k]);
            result0 += w4x4 * i4x4[0];
            result1 += w4x4 * i4x4[1];
            result2 += w4x4 * i4x4[2];
            result3 += w4x4 * i4x4[3];
        }
        *off_out = OType(result0); off_out += cst.output_width * cst.group;
        *off_out = OType(result1); off_out += cst.output_width * cst.group;
        *off_out = OType(result2); off_out += cst.output_width * cst.group;
        *off_out = OType(result3);
    }
}

kernel void matmul4x4(const device ftype4x4 *in     [[buffer(0)]],
                      device ftype4 *out            [[buffer(1)]],
                      const device ftype4x4 *kt     [[buffer(2)]],
                      constant matmul4x4_const &cst [[buffer(3)]],
                      uint3 gid                     [[thread_position_in_grid]]) {
    matmul4x4_template<ftype4x4, ftype4>(in, out, kt, cst, gid);
}

kernel void qntmatmul4x4(const device char4x4 *in       [[buffer(0)]],
                         device float4 *out             [[buffer(1)]],
                         const device char4x4 *kt       [[buffer(2)]],
                         constant matmul4x4_const &cst  [[buffer(3)]],
                         uint3 gid                      [[thread_position_in_grid]]) {
    matmul4x4_template<char4x4, float4>(in, out, kt, cst, gid);
}
