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

kernel void winograd_transform_source2_5_1(const device ftype4 *in          [[buffer(0)]],
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
        auto S40 = get_input(z_in, ix + 4, iy + 0, cst);
        auto S50 = get_input(z_in, ix + 5, iy + 0, cst);
        auto S01 = get_input(z_in, ix + 0, iy + 1, cst);
        auto S11 = get_input(z_in, ix + 1, iy + 1, cst);
        auto S21 = get_input(z_in, ix + 2, iy + 1, cst);
        auto S31 = get_input(z_in, ix + 3, iy + 1, cst);
        auto S41 = get_input(z_in, ix + 4, iy + 1, cst);
        auto S51 = get_input(z_in, ix + 5, iy + 1, cst);
        auto S02 = get_input(z_in, ix + 0, iy + 2, cst);
        auto S12 = get_input(z_in, ix + 1, iy + 2, cst);
        auto S22 = get_input(z_in, ix + 2, iy + 2, cst);
        auto S32 = get_input(z_in, ix + 3, iy + 2, cst);
        auto S42 = get_input(z_in, ix + 4, iy + 2, cst);
        auto S52 = get_input(z_in, ix + 5, iy + 2, cst);
        auto S03 = get_input(z_in, ix + 0, iy + 3, cst);
        auto S13 = get_input(z_in, ix + 1, iy + 3, cst);
        auto S23 = get_input(z_in, ix + 2, iy + 3, cst);
        auto S33 = get_input(z_in, ix + 3, iy + 3, cst);
        auto S43 = get_input(z_in, ix + 4, iy + 3, cst);
        auto S53 = get_input(z_in, ix + 5, iy + 3, cst);
        auto S04 = get_input(z_in, ix + 0, iy + 4, cst);
        auto S14 = get_input(z_in, ix + 1, iy + 4, cst);
        auto S24 = get_input(z_in, ix + 2, iy + 4, cst);
        auto S34 = get_input(z_in, ix + 3, iy + 4, cst);
        auto S44 = get_input(z_in, ix + 4, iy + 4, cst);
        auto S54 = get_input(z_in, ix + 5, iy + 4, cst);
        auto S05 = get_input(z_in, ix + 0, iy + 5, cst);
        auto S15 = get_input(z_in, ix + 1, iy + 5, cst);
        auto S25 = get_input(z_in, ix + 2, iy + 5, cst);
        auto S35 = get_input(z_in, ix + 3, iy + 5, cst);
        auto S45 = get_input(z_in, ix + 4, iy + 5, cst);
        auto S55 = get_input(z_in, ix + 5, iy + 5, cst);

        auto m00 = +S00 - 1.25 * S02 + 0.25 * S04;
        auto m10 = +S10 - 1.25 * S12 + 0.25 * S14;
        auto m20 = +S20 - 1.25 * S22 + 0.25 * S24;
        auto m30 = +S30 - 1.25 * S32 + 0.25 * S34;
        auto m40 = +S40 - 1.25 * S42 + 0.25 * S44;
        auto m50 = +S50 - 1.25 * S52 + 0.25 * S54;
        auto m01 = +0.666667 * S01 + 0.666667 * S02 - 0.166667 * S03 - 0.166667 * S04;
        auto m11 = +0.666667 * S11 + 0.666667 * S12 - 0.166667 * S13 - 0.166667 * S14;
        auto m21 = +0.666667 * S21 + 0.666667 * S22 - 0.166667 * S23 - 0.166667 * S24;
        auto m31 = +0.666667 * S31 + 0.666667 * S32 - 0.166667 * S33 - 0.166667 * S34;
        auto m41 = +0.666667 * S41 + 0.666667 * S42 - 0.166667 * S43 - 0.166667 * S44;
        auto m51 = +0.666667 * S51 + 0.666667 * S52 - 0.166667 * S53 - 0.166667 * S54;
        auto m02 = -0.666667 * S01 + 0.666667 * S02 + 0.166667 * S03 - 0.166667 * S04;
        auto m12 = -0.666667 * S11 + 0.666667 * S12 + 0.166667 * S13 - 0.166667 * S14;
        auto m22 = -0.666667 * S21 + 0.666667 * S22 + 0.166667 * S23 - 0.166667 * S24;
        auto m32 = -0.666667 * S31 + 0.666667 * S32 + 0.166667 * S33 - 0.166667 * S34;
        auto m42 = -0.666667 * S41 + 0.666667 * S42 + 0.166667 * S43 - 0.166667 * S44;
        auto m52 = -0.666667 * S51 + 0.666667 * S52 + 0.166667 * S53 - 0.166667 * S54;
        auto m03 = -0.0833333 * S01 - 0.0416667 * S02 + 0.0833333 * S03 + 0.0416667 * S04;
        auto m13 = -0.0833333 * S11 - 0.0416667 * S12 + 0.0833333 * S13 + 0.0416667 * S14;
        auto m23 = -0.0833333 * S21 - 0.0416667 * S22 + 0.0833333 * S23 + 0.0416667 * S24;
        auto m33 = -0.0833333 * S31 - 0.0416667 * S32 + 0.0833333 * S33 + 0.0416667 * S34;
        auto m43 = -0.0833333 * S41 - 0.0416667 * S42 + 0.0833333 * S43 + 0.0416667 * S44;
        auto m53 = -0.0833333 * S51 - 0.0416667 * S52 + 0.0833333 * S53 + 0.0416667 * S54;
        auto m04 = +0.0833333 * S01 - 0.0416667 * S02 - 0.0833333 * S03 + 0.0416667 * S04;
        auto m14 = +0.0833333 * S11 - 0.0416667 * S12 - 0.0833333 * S13 + 0.0416667 * S14;
        auto m24 = +0.0833333 * S21 - 0.0416667 * S22 - 0.0833333 * S23 + 0.0416667 * S24;
        auto m34 = +0.0833333 * S31 - 0.0416667 * S32 - 0.0833333 * S33 + 0.0416667 * S34;
        auto m44 = +0.0833333 * S41 - 0.0416667 * S42 - 0.0833333 * S43 + 0.0416667 * S44;
        auto m54 = +0.0833333 * S51 - 0.0416667 * S52 - 0.0833333 * S53 + 0.0416667 * S54;
        auto m05 = +4.0 * S01 - 5.0 * S03 + S05;
        auto m15 = +4.0 * S11 - 5.0 * S13 + S15;
        auto m25 = +4.0 * S21 - 5.0 * S23 + S25;
        auto m35 = +4.0 * S31 - 5.0 * S33 + S35;
        auto m45 = +4.0 * S41 - 5.0 * S43 + S45;
        auto m55 = +4.0 * S51 - 5.0 * S53 + S55;

        int dst_x_origin = pos.z;
        int dst_y_origin = cst.unit_width * pos.y + pos.x;
        int dst_y_stride = cst.input_shape.z * 4;
        int dst_y        = dst_y_origin / 4;
        int dst_x        = dst_y_origin % 4 + 4 * dst_x_origin;
        int src_height   = UP_DIV(cst.unit_width * cst.unit_height, 4);
        int stride       = src_height * dst_y_stride;
        auto xy_out = out + dst_y * dst_y_stride + dst_x;
                          *xy_out = +m00 - 1.25 * m20 + 0.25 * m40;
        xy_out += stride; *xy_out = +0.666667 * m10 + 0.666667 * m20 - 0.166667 * m30 - 0.166667 * m40;
        xy_out += stride; *xy_out = -0.666667 * m10 + 0.666667 * m20 + 0.166667 * m30 - 0.166667 * m40;
        xy_out += stride; *xy_out = -0.0833333 * m10 - 0.0416667 * m20 + 0.0833333 * m30 + 0.0416667 * m40;
        xy_out += stride; *xy_out = +0.0833333 * m10 - 0.0416667 * m20 - 0.0833333 * m30 + 0.0416667 * m40;
        xy_out += stride; *xy_out = +4.0 * m10 - 5.0 * m30 + m50;
        xy_out += stride; *xy_out = +m01 - 1.25 * m21 + 0.25 * m41;
        xy_out += stride; *xy_out = +0.666667 * m11 + 0.666667 * m21 - 0.166667 * m31 - 0.166667 * m41;
        xy_out += stride; *xy_out = -0.666667 * m11 + 0.666667 * m21 + 0.166667 * m31 - 0.166667 * m41;
        xy_out += stride; *xy_out = -0.0833333 * m11 - 0.0416667 * m21 + 0.0833333 * m31 + 0.0416667 * m41;
        xy_out += stride; *xy_out = +0.0833333 * m11 - 0.0416667 * m21 - 0.0833333 * m31 + 0.0416667 * m41;
        xy_out += stride; *xy_out = +4.0 * m11 - 5.0 * m31 + m51;
        xy_out += stride; *xy_out = +m02 - 1.25 * m22 + 0.25 * m42;
        xy_out += stride; *xy_out = +0.666667 * m12 + 0.666667 * m22 - 0.166667 * m32 - 0.166667 * m42;
        xy_out += stride; *xy_out = -0.666667 * m12 + 0.666667 * m22 + 0.166667 * m32 - 0.166667 * m42;
        xy_out += stride; *xy_out = -0.0833333 * m12 - 0.0416667 * m22 + 0.0833333 * m32 + 0.0416667 * m42;
        xy_out += stride; *xy_out = +0.0833333 * m12 - 0.0416667 * m22 - 0.0833333 * m32 + 0.0416667 * m42;
        xy_out += stride; *xy_out = +4.0 * m12 - 5.0 * m32 + m52;
        xy_out += stride; *xy_out = +m03 - 1.25 * m23 + 0.25 * m43;
        xy_out += stride; *xy_out = +0.666667 * m13 + 0.666667 * m23 - 0.166667 * m33 - 0.166667 * m43;
        xy_out += stride; *xy_out = -0.666667 * m13 + 0.666667 * m23 + 0.166667 * m33 - 0.166667 * m43;
        xy_out += stride; *xy_out = -0.0833333 * m13 - 0.0416667 * m23 + 0.0833333 * m33 + 0.0416667 * m43;
        xy_out += stride; *xy_out = +0.0833333 * m13 - 0.0416667 * m23 - 0.0833333 * m33 + 0.0416667 * m43;
        xy_out += stride; *xy_out = +4.0 * m13 - 5.0 * m33 + m53;
        xy_out += stride; *xy_out = +m04 - 1.25 * m24 + 0.25 * m44;
        xy_out += stride; *xy_out = +0.666667 * m14 + 0.666667 * m24 - 0.166667 * m34 - 0.166667 * m44;
        xy_out += stride; *xy_out = -0.666667 * m14 + 0.666667 * m24 + 0.166667 * m34 - 0.166667 * m44;
        xy_out += stride; *xy_out = -0.0833333 * m14 - 0.0416667 * m24 + 0.0833333 * m34 + 0.0416667 * m44;
        xy_out += stride; *xy_out = +0.0833333 * m14 - 0.0416667 * m24 - 0.0833333 * m34 + 0.0416667 * m44;
        xy_out += stride; *xy_out = +4.0 * m14 - 5.0 * m34 + m54;
        xy_out += stride; *xy_out = +m05 - 1.25 * m25 + 0.25 * m45;
        xy_out += stride; *xy_out = +0.666667 * m15 + 0.666667 * m25 - 0.166667 * m35 - 0.166667 * m45;
        xy_out += stride; *xy_out = -0.666667 * m15 + 0.666667 * m25 + 0.166667 * m35 - 0.166667 * m45;
        xy_out += stride; *xy_out = -0.0833333 * m15 - 0.0416667 * m25 + 0.0833333 * m35 + 0.0416667 * m45;
        xy_out += stride; *xy_out = +0.0833333 * m15 - 0.0416667 * m25 - 0.0833333 * m35 + 0.0416667 * m45;
        xy_out += stride; *xy_out = +4.0 * m15 - 5.0 * m35 + m55;
    }
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

kernel void winograd_transform_dest2_5_1(const device ftype4 *in            [[buffer(0)]],
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
        int dst_y_stride = dst_w * 36;
        auto xy_in = in + dst_y * dst_y_stride + dst_x;

        auto S00 = *xy_in; xy_in += dst_w;
        auto S10 = *xy_in; xy_in += dst_w;
        auto S20 = *xy_in; xy_in += dst_w;
        auto S30 = *xy_in; xy_in += dst_w;
        auto S40 = *xy_in; xy_in += dst_w;
        auto S50 = *xy_in; xy_in += dst_w;
        auto S01 = *xy_in; xy_in += dst_w;
        auto S11 = *xy_in; xy_in += dst_w;
        auto S21 = *xy_in; xy_in += dst_w;
        auto S31 = *xy_in; xy_in += dst_w;
        auto S41 = *xy_in; xy_in += dst_w;
        auto S51 = *xy_in; xy_in += dst_w;
        auto S02 = *xy_in; xy_in += dst_w;
        auto S12 = *xy_in; xy_in += dst_w;
        auto S22 = *xy_in; xy_in += dst_w;
        auto S32 = *xy_in; xy_in += dst_w;
        auto S42 = *xy_in; xy_in += dst_w;
        auto S52 = *xy_in; xy_in += dst_w;
        auto S03 = *xy_in; xy_in += dst_w;
        auto S13 = *xy_in; xy_in += dst_w;
        auto S23 = *xy_in; xy_in += dst_w;
        auto S33 = *xy_in; xy_in += dst_w;
        auto S43 = *xy_in; xy_in += dst_w;
        auto S53 = *xy_in; xy_in += dst_w;
        auto S04 = *xy_in; xy_in += dst_w;
        auto S14 = *xy_in; xy_in += dst_w;
        auto S24 = *xy_in; xy_in += dst_w;
        auto S34 = *xy_in; xy_in += dst_w;
        auto S44 = *xy_in; xy_in += dst_w;
        auto S54 = *xy_in; xy_in += dst_w;
        auto S05 = *xy_in; xy_in += dst_w;
        auto S15 = *xy_in; xy_in += dst_w;
        auto S25 = *xy_in; xy_in += dst_w;
        auto S35 = *xy_in; xy_in += dst_w;
        auto S45 = *xy_in; xy_in += dst_w;
        auto S55 = *xy_in;

        auto m00 = +S00 + S01 + S02 + S03 + S04;
        auto m10 = +S10 + S11 + S12 + S13 + S14;
        auto m20 = +S20 + S21 + S22 + S23 + S24;
        auto m30 = +S30 + S31 + S32 + S33 + S34;
        auto m40 = +S40 + S41 + S42 + S43 + S44;
        auto m50 = +S50 + S51 + S52 + S53 + S54;
        auto m01 = +S01 - S02 + 2.0 * S03 - 2.0 * S04 + S05;
        auto m11 = +S11 - S12 + 2.0 * S13 - 2.0 * S14 + S15;
        auto m21 = +S21 - S22 + 2.0 * S23 - 2.0 * S24 + S25;
        auto m31 = +S31 - S32 + 2.0 * S33 - 2.0 * S34 + S35;
        auto m41 = +S41 - S42 + 2.0 * S43 - 2.0 * S44 + S45;
        auto m51 = +S51 - S52 + 2.0 * S53 - 2.0 * S54 + S55;

        // write output
        auto b4 = biasTerms[int(pos.z)];
        int oy = pos.y * cst.unit;
        int ox = pos.x * cst.unit;
        auto z_out = out + pos.z * cst.output_shape.x * cst.output_shape.y;
        
        /* if true */ {
            set_output(cst, z_out, ox + 0, oy + 0, b4 + m00 + m10 + m20 + m30 + m40);
        }
        if (ox + 1 < cst.output_shape.x) {
            set_output(cst, z_out, ox + 1, oy + 0, b4 + m10 - m20 + 2.0 * m30 - 2.0 * m40 + m50);
        }
        if (oy + 1 < cst.output_shape.y) {
            set_output(cst, z_out, ox + 0, oy + 1, b4 + m01 + m11 + m21 + m31 + m41);
        }
        if (ox + 1 < cst.output_shape.x && oy + 1 < cst.output_shape.y) {
            set_output(cst, z_out, ox + 1, oy + 1, b4 + m11 - m21 + 2.0 * m31 - 2.0 * m41 + m51);
        }
    }
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
