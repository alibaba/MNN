//
//  MetalConvolutionWinograd.metal
//  MNN
//
//  Created by MNN on 2019/02/01.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <metal_stdlib>
#include "MetalConvolutionActivation.metal"

using namespace metal;

struct winograd_constants {
    int4 input_shape;
    int4 output_shape;
    int pad_x;
    int pad_y;
    int unit_width;
    int unit_height;
    int unit;
    conv_activation_type activation;
};

static inline ftype4 get_input(const device ftype4 *input, int x, int y, constant winograd_constants &cst) {
    return x < cst.input_shape.x && y < cst.input_shape.y && x >= 0 && y >= 0 ? input[x + y * cst.input_shape.x] : 0;
}

kernel void winograd_transform_source2_3_1(const device ftype4 *in          [[buffer(0)]],
                                           device ftype4 *out               [[buffer(1)]],
                                           constant winograd_constants &cst [[buffer(2)]],
                                           uint3 gid                        [[thread_position_in_grid]]) {
    auto pos = int3(gid);
    if (pos.x < cst.unit_width && pos.y < cst.unit_height) {
        int ix = pos.x * cst.unit - cst.pad_x;
        int iy = pos.y * cst.unit - cst.pad_y;

        auto z_in = in + pos.z * cst.input_shape.x * cst.input_shape.y;
        auto S00 = get_input(z_in, ix + 0, iy + 0, cst);
        auto S10 = get_input(z_in, ix + 1, iy + 0, cst);
        auto S20 = get_input(z_in, ix + 2, iy + 0, cst);
        auto S30 = get_input(z_in, ix + 3, iy + 0, cst);
        auto S01 = get_input(z_in, ix + 0, iy + 1, cst);
        auto S11 = get_input(z_in, ix + 1, iy + 1, cst);
        auto S21 = get_input(z_in, ix + 2, iy + 1, cst);
        auto S31 = get_input(z_in, ix + 3, iy + 1, cst);
        auto S02 = get_input(z_in, ix + 0, iy + 2, cst);
        auto S12 = get_input(z_in, ix + 1, iy + 2, cst);
        auto S22 = get_input(z_in, ix + 2, iy + 2, cst);
        auto S32 = get_input(z_in, ix + 3, iy + 2, cst);
        auto S03 = get_input(z_in, ix + 0, iy + 3, cst);
        auto S13 = get_input(z_in, ix + 1, iy + 3, cst);
        auto S23 = get_input(z_in, ix + 2, iy + 3, cst);
        auto S33 = get_input(z_in, ix + 3, iy + 3, cst);

        auto m00 = +S00 - S02;
        auto m10 = +S10 - S12;
        auto m20 = +S20 - S22;
        auto m30 = +S30 - S32;
        auto m01 = +0.5 * S01 + 0.5 * S02;
        auto m11 = +0.5 * S11 + 0.5 * S12;
        auto m21 = +0.5 * S21 + 0.5 * S22;
        auto m31 = +0.5 * S31 + 0.5 * S32;
        auto m02 = -0.5 * S01 + 0.5 * S02;
        auto m12 = -0.5 * S11 + 0.5 * S12;
        auto m22 = -0.5 * S21 + 0.5 * S22;
        auto m32 = -0.5 * S31 + 0.5 * S32;
        auto m03 = -S01 + S03;
        auto m13 = -S11 + S13;
        auto m23 = -S21 + S23;
        auto m33 = -S31 + S33;

        int dst_x_origin = pos.z;
        int dst_y_origin = cst.unit_width * pos.y + pos.x;
        int dst_y_stride = cst.input_shape.z * 4;
        int dst_y        = dst_y_origin / 4;
        int dst_x        = dst_y_origin % 4 + 4 * dst_x_origin;
        int src_height   = UP_DIV(cst.unit_width * cst.unit_height, 4);
        int stride       = src_height * dst_y_stride;
        auto xy_out = out + dst_y * dst_y_stride + dst_x;
                          *xy_out =  +m00 - m20;
        xy_out += stride; *xy_out =  +0.5 * m10 + 0.5 * m20;
        xy_out += stride; *xy_out =  -0.5 * m10 + 0.5 * m20;
        xy_out += stride; *xy_out =  -m10 + m30;
        xy_out += stride; *xy_out =  +m01 - m21;
        xy_out += stride; *xy_out =  +0.5 * m11 + 0.5 * m21;
        xy_out += stride; *xy_out =  -0.5 * m11 + 0.5 * m21;
        xy_out += stride; *xy_out =  -m11 + m31;
        xy_out += stride; *xy_out =  +m02 - m22;
        xy_out += stride; *xy_out=  +0.5 * m12 + 0.5 * m22;
        xy_out += stride; *xy_out =  -0.5 * m12 + 0.5 * m22;
        xy_out += stride; *xy_out =  -m12 + m32;
        xy_out += stride; *xy_out =  +m03 - m23;
        xy_out += stride; *xy_out =  +0.5 * m13 + 0.5 * m23;
        xy_out += stride; *xy_out =  -0.5 * m13 + 0.5 * m23;
        xy_out += stride; *xy_out =  -m13 + m33;
    }
}

static inline void set_output(constant winograd_constants &cst, device ftype4 *output, int x, int y, ftype4 value) {
    output[y * cst.output_shape.x + x] = activate(value, cst.activation);
}

kernel void winograd_transform_dest2_3_1(const device ftype4 *in            [[buffer(0)]],
                                         const device ftype4 *biasTerms     [[buffer(1)]],
                                         device ftype4 *out                 [[buffer(2)]],
                                         constant winograd_constants &cst   [[buffer(3)]],
                                         uint3 gid                          [[thread_position_in_grid]]) {
    auto pos = int3(gid);
    if (pos.x < cst.unit_width && pos.y < cst.unit_height) {
        int dst_w        = UP_DIV(cst.unit_width * cst.unit_height, 4);
        int dst_x_origin = cst.unit_width * pos.y + pos.x;
        int dst_x        = dst_x_origin / 4;
        int dst_y        = 4 * pos.z + dst_x_origin % 4;
        int dst_y_stride = dst_w * 16;
        auto xy_in = in + dst_y * dst_y_stride + dst_x;

        auto S00 = *xy_in; xy_in += dst_w;
        auto S10 = *xy_in; xy_in += dst_w;
        auto S20 = *xy_in; xy_in += dst_w;
        auto S30 = *xy_in; xy_in += dst_w;
        auto S01 = *xy_in; xy_in += dst_w;
        auto S11 = *xy_in; xy_in += dst_w;
        auto S21 = *xy_in; xy_in += dst_w;
        auto S31 = *xy_in; xy_in += dst_w;
        auto S02 = *xy_in; xy_in += dst_w;
        auto S12 = *xy_in; xy_in += dst_w;
        auto S22 = *xy_in; xy_in += dst_w;
        auto S32 = *xy_in; xy_in += dst_w;
        auto S03 = *xy_in; xy_in += dst_w;
        auto S13 = *xy_in; xy_in += dst_w;
        auto S23 = *xy_in; xy_in += dst_w;
        auto S33 = *xy_in;

        auto m00 = +S00 + S01 + S02;
        auto m10 = +S10 + S11 + S12;
        auto m20 = +S20 + S21 + S22;
        auto m30 = +S30 + S31 + S32;
        auto m01 = +S01 - S02 + S03;
        auto m11 = +S11 - S12 + S13;
        auto m21 = +S21 - S22 + S23;
        auto m31 = +S31 - S32 + S33;

        // write output
        auto b4 = biasTerms[int(pos.z)];
        int oy = pos.y * cst.unit;
        int ox = pos.x * cst.unit;
        auto z_out = out + pos.z * cst.output_shape.x * cst.output_shape.y;
        
        /* if true */ {
            set_output(cst, z_out, ox + 0, oy + 0, b4 + m00 + m10 + m20);
        }
        if (ox + 1 < cst.output_shape.x) {
            set_output(cst, z_out, ox + 1, oy + 0, b4 + m10 - m20 + m30);
        }
        if (oy + 1 < cst.output_shape.y) {
            set_output(cst, z_out, ox + 0, oy + 1, b4 + m01 + m11 + m21);
        }
        if (ox + 1 < cst.output_shape.x && oy + 1 < cst.output_shape.y) {
            set_output(cst, z_out, ox + 1, oy + 1, b4 + m11 - m21 + m31);
        }
    }
}
