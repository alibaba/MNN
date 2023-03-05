#define CONV_UNROLL (4)
#define CONV_UNROLL_L (8)

struct conv1x1_constants {
    int input_size;
    int input_slice;
    int output_width;
    int output_height;
    int output_size;
    int output_slice;
    int output_channel;
    int batch;
    conv_activation_type activation;
};

kernel void conv1x1_w1h1(const device ftype4 *in         [[buffer(0)]],
                    device ftype4 *out              [[buffer(1)]],
                    constant conv1x1_constants& cst [[buffer(2)]],
                    const device ftype4x4 *wt       [[buffer(3)]],
                    const device ftype4 *biasTerms  [[buffer(4)]],
                    uint3 gid                       [[thread_position_in_grid]]) {
    if ((int)gid.x >= cst.output_width || (int)gid.y >= cst.output_height || (int)gid.z >= cst.batch * cst.output_slice) return;
    int idx_w = gid.x;
    int idx_h = gid.y;
    int idx_c = gid.z % cst.output_slice;
    int idx_b = gid.z / cst.output_slice;

    auto xy_wt = wt + idx_c * cst.input_slice;
    auto xy_in0  = in  + (int)idx_b * cst.input_slice * cst.input_size + idx_h * cst.output_width + idx_w;

    auto xy_out = out + (int)idx_b * cst.output_slice * cst.output_size + idx_c * cst.output_size + idx_h * cst.output_width + idx_w;
    auto biasValue = FLOAT4(biasTerms[idx_c]);
    FLOAT4 result0 = biasValue;

    for (auto z = 0; z < cst.input_slice; z++) {
        auto in40 = xy_in0[0];
        auto w = xy_wt[z];

        result0 += FLOAT4(in40 * w);
        xy_in0 += cst.input_size;
    }

    /* true            */ *xy_out = activate(ftype4(result0), cst.activation);
}

kernel void conv1x1_g1z4(const device ftype4 *in            [[buffer(0)]],
                         device ftype4 *out                 [[buffer(1)]],
                         constant conv1x1_constants& cst    [[buffer(2)]],
                         const device ftype4x4 *wt          [[buffer(3)]],
                         const device ftype4 *biasTerms     [[buffer(4)]],
                         uint3 gid                          [[thread_position_in_grid]]) {
    if ((int)gid.x * CONV_UNROLL >= cst.output_size || (int)gid.y >= cst.output_slice || (int)gid.z >= cst.batch) return;
    
    int rx = gid.x * CONV_UNROLL;
    int uz = gid.y;
    auto xy_wt = wt + uz * cst.input_slice;
    auto xy_in0  = in  + (int)gid.z * cst.input_slice  * cst.input_size + rx + 0;
    auto xy_out = out + (int)gid.z * cst.output_slice * cst.output_size + uz * cst.output_size + rx;
    auto biasValue = FLOAT4(biasTerms[uz]);
    FLOAT4 result0 = biasValue, result1 = biasValue, result2 = biasValue, result3 = biasValue;
    int computeSize = min(cst.output_size - rx, CONV_UNROLL);

    for (auto z = 0; z < cst.input_slice; z++) {
        auto in40 = *xy_in0;
        auto in41 = *(xy_in0 + 1);
        auto in42 = *(xy_in0 + 2);
        auto in43 = *(xy_in0 + 3);
        auto w = xy_wt[z];
        
        result0 += FLOAT4(in40 * w);
        result1 += FLOAT4(in41 * w);
        result2 += FLOAT4(in42 * w);
        result3 += FLOAT4(in43 * w);
        xy_in0 += cst.input_size;
    }
    
    /* true                               */ *xy_out = activate(ftype4(result0), cst.activation);
    if (computeSize > 1) {xy_out[1] = activate(ftype4(result1), cst.activation); }
    if (computeSize > 2) {xy_out[2] = activate(ftype4(result2), cst.activation); }
    if (computeSize > 3) {xy_out[3] = activate(ftype4(result3), cst.activation); }
}


kernel void conv1x1_g1z8(const device ftype4 *in            [[buffer(0)]],
                         device ftype4 *out                 [[buffer(1)]],
                         constant conv1x1_constants& cst    [[buffer(2)]],
                         const device ftype4x4 *wt          [[buffer(3)]],
                         const device ftype4 *biasTerms     [[buffer(4)]],
                         uint3 gid                          [[thread_position_in_grid]]) {
    if ((int)gid.x * CONV_UNROLL_L >= cst.output_size || (int)gid.y >= cst.output_slice || (int)gid.z >= cst.batch) return;

    int rx = gid.x * CONV_UNROLL_L;
    int uz = gid.y;
    auto xy_wt = wt + uz * cst.input_slice;
    auto xy_in0  = in  + (int)gid.z * cst.input_slice  * cst.input_size + rx + 0;

    auto xy_out = out + (int)gid.z * cst.output_slice * cst.output_size + uz * cst.output_size + rx;
    auto biasValue = FLOAT4(biasTerms[uz]);
    FLOAT4 result0 = biasValue, result1 = biasValue, result2 = biasValue, result3 = biasValue;
    FLOAT4 result4 = biasValue, result5 = biasValue, result6 = biasValue, result7 = biasValue;

    int computeSize = min(cst.output_size - rx, CONV_UNROLL_L);
    for (auto z = 0; z < cst.input_slice; z++) {
            auto in40 = xy_in0[0];
            auto in41 = xy_in0[1];
            auto in42 = xy_in0[2];
            auto in43 = xy_in0[3];
            auto in44 = xy_in0[4];
            auto in45 = xy_in0[5];
            auto in46 = xy_in0[6];
            auto in47 = xy_in0[7];

            auto w = xy_wt[z];

            result0 += FLOAT4(in40 * w);
            result1 += FLOAT4(in41 * w);
            result2 += FLOAT4(in42 * w);
            result3 += FLOAT4(in43 * w);
            result4 += FLOAT4(in44 * w);
            result5 += FLOAT4(in45 * w);
            result6 += FLOAT4(in46 * w);
            result7 += FLOAT4(in47 * w);
            xy_in0 += cst.input_size;
    }

    /* true                               */ *xy_out = activate(ftype4(result0), cst.activation);
    if (computeSize > 1) {xy_out[1] = activate(ftype4(result1), cst.activation); }
    if (computeSize > 2) {xy_out[2] = activate(ftype4(result2), cst.activation); }
    if (computeSize > 3) {xy_out[3] = activate(ftype4(result3), cst.activation); }
    if (computeSize > 4) {xy_out[4] = activate(ftype4(result4), cst.activation); }
    if (computeSize > 5) {xy_out[5] = activate(ftype4(result5), cst.activation); }
    if (computeSize > 6) {xy_out[6] = activate(ftype4(result6), cst.activation); }
    if (computeSize > 7) {xy_out[7] = activate(ftype4(result7), cst.activation); }
}


kernel void conv1x1_w4h2(const device ftype4 *in            [[buffer(0)]],
                         device ftype4 *out                 [[buffer(1)]],
                         constant conv1x1_constants& cst    [[buffer(2)]],
                         const device ftype4x4 *wt          [[buffer(3)]],
                         const device ftype4 *biasTerms     [[buffer(4)]],
                         uint3 gid                          [[thread_position_in_grid]]) {
    if ((int)gid.x * 4 >= cst.output_width || (int)gid.y * 2 >= cst.output_height || (int)gid.z >= cst.batch * cst.output_slice) return;

    int idx_w = gid.x << 2;
    int idx_h = gid.y << 1;
    int idx_c = gid.z % cst.output_slice;
    int idx_b = gid.z / cst.output_slice;

    auto xy_wt = wt + idx_c * cst.input_slice;
    auto xy_in0  = in  + (int)idx_b * cst.input_slice * cst.input_size + idx_h * cst.output_width + idx_w;

    auto xy_out = out + (int)idx_b * cst.output_slice * cst.output_size + idx_c * cst.output_size + idx_h * cst.output_width + idx_w;
    auto biasValue = FLOAT4(biasTerms[idx_c]);
    FLOAT4 result0 = biasValue, result1 = biasValue, result2 = biasValue, result3 = biasValue;
    FLOAT4 result4 = biasValue, result5 = biasValue, result6 = biasValue, result7 = biasValue;

    for (auto z = 0; z < cst.input_slice; z++) {
        auto in40 = xy_in0[0];
        auto in41 = xy_in0[1];
        auto in42 = xy_in0[2];
        auto in43 = xy_in0[3];
        auto in44 = xy_in0[cst.output_width+0];
        auto in45 = xy_in0[cst.output_width+1];
        auto in46 = xy_in0[cst.output_width+2];
        auto in47 = xy_in0[cst.output_width+3];

        auto w = xy_wt[z];

        result0 += FLOAT4(in40 * w);
        result1 += FLOAT4(in41 * w);
        result2 += FLOAT4(in42 * w);
        result3 += FLOAT4(in43 * w);
        result4 += FLOAT4(in44 * w);
        result5 += FLOAT4(in45 * w);
        result6 += FLOAT4(in46 * w);
        result7 += FLOAT4(in47 * w);
        xy_in0 += cst.input_size;
    }

    int widthSize = min(cst.output_width - idx_w, 4);
    /* true            */ *xy_out = activate(ftype4(result0), cst.activation);
    if (widthSize > 1) {xy_out[1] = activate(ftype4(result1), cst.activation); }
    if (widthSize > 2) {xy_out[2] = activate(ftype4(result2), cst.activation); }
    if (widthSize > 3) {xy_out[3] = activate(ftype4(result3), cst.activation); }
    
    int heightSize = min(cst.output_height - idx_h, 2);
    if(heightSize > 1) {
        /* true         */ {xy_out[cst.output_width+0] = activate(ftype4(result4), cst.activation); }
        if (widthSize > 1) {xy_out[cst.output_width+1] = activate(ftype4(result5), cst.activation); }
        if (widthSize > 2) {xy_out[cst.output_width+2] = activate(ftype4(result6), cst.activation); }
        if (widthSize > 3) {xy_out[cst.output_width+3] = activate(ftype4(result7), cst.activation); }
    }
}


kernel void conv1x1_w4h4(const device ftype4 *in            [[buffer(0)]],
                         device ftype4 *out                 [[buffer(1)]],
                         constant conv1x1_constants& cst    [[buffer(2)]],
                         const device ftype4x4 *wt          [[buffer(3)]],
                         const device ftype4 *biasTerms     [[buffer(4)]],
                         uint3 gid                          [[thread_position_in_grid]]) {
    if ((int)gid.x * 4 >= cst.output_width || (int)gid.y * 4 >= cst.output_height || (int)gid.z >= cst.batch * cst.output_slice) return;

    int idx_w = gid.x << 2;
    int idx_h = gid.y << 2;
    int idx_c = gid.z % cst.output_slice;
    int idx_b = gid.z / cst.output_slice;

    auto xy_wt = wt + idx_c * cst.input_slice;
    auto xy_in0  = in  + (int)idx_b * cst.input_slice * cst.input_size + idx_h * cst.output_width + idx_w;

    auto xy_out = out + (int)idx_b * cst.output_slice * cst.output_size + idx_c * cst.output_size + idx_h * cst.output_width + idx_w;
    auto biasValue = FLOAT4(biasTerms[idx_c]);
    FLOAT4 result00 = biasValue, result01 = biasValue, result02 = biasValue, result03 = biasValue;
    FLOAT4 result10 = biasValue, result11 = biasValue, result12 = biasValue, result13 = biasValue;
    FLOAT4 result20 = biasValue, result21 = biasValue, result22 = biasValue, result23 = biasValue;
    FLOAT4 result30 = biasValue, result31 = biasValue, result32 = biasValue, result33 = biasValue;

    for (auto z = 0; z < cst.input_slice; z++) {
        auto in00 = xy_in0[0];
        auto in01 = xy_in0[1];
        auto in02 = xy_in0[2];
        auto in03 = xy_in0[3];
        auto in10 = xy_in0[cst.output_width+0];
        auto in11 = xy_in0[cst.output_width+1];
        auto in12 = xy_in0[cst.output_width+2];
        auto in13 = xy_in0[cst.output_width+3];
        
        auto in20 = xy_in0[cst.output_width+cst.output_width+0];
        auto in21 = xy_in0[cst.output_width+cst.output_width+1];
        auto in22 = xy_in0[cst.output_width+cst.output_width+2];
        auto in23 = xy_in0[cst.output_width+cst.output_width+3];
        auto in30 = xy_in0[cst.output_width+cst.output_width+cst.output_width+0];
        auto in31 = xy_in0[cst.output_width+cst.output_width+cst.output_width+1];
        auto in32 = xy_in0[cst.output_width+cst.output_width+cst.output_width+2];
        auto in33 = xy_in0[cst.output_width+cst.output_width+cst.output_width+3];


        auto w = xy_wt[z];

        result00 += FLOAT4(in00 * w);
        result01 += FLOAT4(in01 * w);
        result02 += FLOAT4(in02 * w);
        result03 += FLOAT4(in03 * w);
        result10 += FLOAT4(in10 * w);
        result11 += FLOAT4(in11 * w);
        result12 += FLOAT4(in12 * w);
        result13 += FLOAT4(in13 * w);
        
        result20 += FLOAT4(in20 * w);
        result21 += FLOAT4(in21 * w);
        result22 += FLOAT4(in22 * w);
        result23 += FLOAT4(in23 * w);
        result30 += FLOAT4(in30 * w);
        result31 += FLOAT4(in31 * w);
        result32 += FLOAT4(in32 * w);
        result33 += FLOAT4(in33 * w);
        
        xy_in0 += cst.input_size;
    }

    int widthSize = min(cst.output_width - idx_w, 4);
    /* true            */ *xy_out = activate(ftype4(result00), cst.activation);
    if (widthSize > 1) {xy_out[1] = activate(ftype4(result01), cst.activation); }
    if (widthSize > 2) {xy_out[2] = activate(ftype4(result02), cst.activation); }
    if (widthSize > 3) {xy_out[3] = activate(ftype4(result03), cst.activation); }
    
    int heightSize = min(cst.output_height - idx_h, 4);
    if(heightSize > 1) {
        /* true         */ {xy_out[cst.output_width+0] = activate(ftype4(result10), cst.activation); }
        if (widthSize > 1) {xy_out[cst.output_width+1] = activate(ftype4(result11), cst.activation); }
        if (widthSize > 2) {xy_out[cst.output_width+2] = activate(ftype4(result12), cst.activation); }
        if (widthSize > 3) {xy_out[cst.output_width+3] = activate(ftype4(result13), cst.activation); }
    }
    if(heightSize > 2) {
        /* true         */ {xy_out[cst.output_width+cst.output_width+0] = activate(ftype4(result20), cst.activation); }
        if (widthSize > 1) {xy_out[cst.output_width+cst.output_width+1] = activate(ftype4(result21), cst.activation); }
        if (widthSize > 2) {xy_out[cst.output_width+cst.output_width+2] = activate(ftype4(result22), cst.activation); }
        if (widthSize > 3) {xy_out[cst.output_width+cst.output_width+3] = activate(ftype4(result23), cst.activation); }
    }
    if(heightSize > 3) {
        /* true         */ {xy_out[cst.output_width+cst.output_width+cst.output_width+0] = activate(ftype4(result30), cst.activation); }
        if (widthSize > 1) {xy_out[cst.output_width+cst.output_width+cst.output_width+1] = activate(ftype4(result31), cst.activation); }
        if (widthSize > 2) {xy_out[cst.output_width+cst.output_width+cst.output_width+2] = activate(ftype4(result32), cst.activation); }
        if (widthSize > 3) {xy_out[cst.output_width+cst.output_width+cst.output_width+3] = activate(ftype4(result33), cst.activation); }
    }
}


kernel void conv1x1_w2c2(const device ftype4 *in            [[buffer(0)]],
                         device ftype4 *out                 [[buffer(1)]],
                         constant conv1x1_constants& cst    [[buffer(2)]],
                         const device ftype4x4 *wt          [[buffer(3)]],
                         const device ftype4 *biasTerms     [[buffer(4)]],
                         uint3 gid                          [[thread_position_in_grid]]) {
    if ((int)gid.x * 2 >= cst.output_width || (int)gid.y >= cst.output_height) return;

    int channel_pack = (cst.output_channel + 7) >> 3;
    int idx_w = gid.x << 1;
    int idx_h = gid.y;
    int idx_c = (gid.z % channel_pack) << 1;
    int idx_b = gid.z / channel_pack;
    
    if(idx_b >=  cst.batch || idx_c >= cst.output_slice) return;
    auto xy_wt = wt + idx_c * cst.input_slice;
    auto xy_in0  = in  + (int)idx_b * cst.input_slice * cst.input_size + idx_h * cst.output_width + idx_w;

    auto xy_out = out + (int)idx_b * cst.output_slice * cst.output_size + idx_c * cst.output_size + idx_h * cst.output_width + idx_w;
    auto biasValue0 = FLOAT4(biasTerms[idx_c]);
    auto biasValue1 = FLOAT4(biasTerms[idx_c+1]);

    FLOAT4 result0 = biasValue0, result1 = biasValue0;
    FLOAT4 result4 = biasValue1, result5 = biasValue1;

    for (auto z = 0; z < cst.input_slice; z++) {
        auto in40 = xy_in0[0];
        auto in41 = xy_in0[1];

        auto w0 = xy_wt[z];
        auto w1 = xy_wt[cst.input_slice+z];

        result0 += FLOAT4(in40 * w0);
        result1 += FLOAT4(in41 * w0);
        result4 += FLOAT4(in40 * w1);
        result5 += FLOAT4(in41 * w1);
        xy_in0 += cst.input_size;
    }

    int widthSize = min(cst.output_width - idx_w, 2);
    /* true            */ *xy_out = activate(ftype4(result0), cst.activation);
    if (widthSize > 1) {xy_out[1] = activate(ftype4(result1), cst.activation); }
    
    int channelSize = min(cst.output_slice - idx_c, 2);
    if(channelSize > 1) {
        /* true         */ {xy_out[cst.output_size+0] = activate(ftype4(result4), cst.activation); }
        if (widthSize > 1) {xy_out[cst.output_size+1] = activate(ftype4(result5), cst.activation); }
    }
}


kernel void conv1x1_w2h2(const device ftype4 *in            [[buffer(0)]],
                         device ftype4 *out                 [[buffer(1)]],
                         constant conv1x1_constants& cst    [[buffer(2)]],
                         const device ftype4x4 *wt          [[buffer(3)]],
                         const device ftype4 *biasTerms     [[buffer(4)]],
                         uint3 gid                          [[thread_position_in_grid]]) {
    if ((int)gid.x * 2 >= cst.output_width || (int)gid.y * 2 >= cst.output_height || (int)gid.z >= cst.batch * cst.output_slice) return;

    int idx_w = gid.x << 1;
    int idx_h = gid.y << 1;
    int idx_c = gid.z % cst.output_slice;
    int idx_b = gid.z / cst.output_slice;

    auto xy_wt = wt + idx_c * cst.input_slice;
    auto xy_in0  = in  + (int)idx_b * cst.input_slice * cst.input_size + idx_h * cst.output_width + idx_w;

    auto xy_out = out + (int)idx_b * cst.output_slice * cst.output_size + idx_c * cst.output_size + idx_h * cst.output_width + idx_w;
    auto biasValue = FLOAT4(biasTerms[idx_c]);
    FLOAT4 result0 = biasValue, result1 = biasValue;
    FLOAT4 result4 = biasValue, result5 = biasValue;

    for (auto z = 0; z < cst.input_slice; z++) {
        auto in40 = xy_in0[0];
        auto in41 = xy_in0[1];
        auto in44 = xy_in0[cst.output_width+0];
        auto in45 = xy_in0[cst.output_width+1];

        auto w = xy_wt[z];

        result0 += FLOAT4(in40 * w);
        result1 += FLOAT4(in41 * w);
        result4 += FLOAT4(in44 * w);
        result5 += FLOAT4(in45 * w);
        xy_in0 += cst.input_size;
    }

    int widthSize = min(cst.output_width - idx_w, 2);
    /* true            */ *xy_out = activate(ftype4(result0), cst.activation);
    if (widthSize > 1) {xy_out[1] = activate(ftype4(result1), cst.activation); }
    
    int heightSize = min(cst.output_height - idx_h, 2);
    if(heightSize > 1) {
        /* true         */ {xy_out[cst.output_width+0] = activate(ftype4(result4), cst.activation); }
        if (widthSize > 1) {xy_out[cst.output_width+1] = activate(ftype4(result5), cst.activation); }
    }
}


kernel void conv1x1_w2h2c2(const device ftype4 *in            [[buffer(0)]],
                         device ftype4 *out                 [[buffer(1)]],
                         constant conv1x1_constants& cst    [[buffer(2)]],
                         const device ftype4x4 *wt          [[buffer(3)]],
                         const device ftype4 *biasTerms     [[buffer(4)]],
                         uint3 gid                          [[thread_position_in_grid]]) {
    if ((int)gid.x * 2 >= cst.output_width || (int)gid.y * 2 >= cst.output_height) return;

    int channel_pack = (cst.output_channel + 7) >> 3;
    int idx_w = gid.x << 1;
    int idx_h = gid.y << 1;
    int idx_c = (gid.z % channel_pack) << 1;
    int idx_b = gid.z / channel_pack;

    if(idx_b >=  cst.batch || idx_c >= cst.output_slice) return;
    auto xy_wt = wt + idx_c * cst.input_slice;
    auto xy_in0  = in  + (int)idx_b * cst.input_slice * cst.input_size + idx_h * cst.output_width + idx_w;

    auto xy_out = out + (int)idx_b * cst.output_slice * cst.output_size + idx_c * cst.output_size + idx_h * cst.output_width + idx_w;
    auto biasValue0 = FLOAT4(biasTerms[idx_c]);
    auto biasValue1 = FLOAT4(biasTerms[idx_c+1]);

    FLOAT4 result0 = biasValue0, result1 = biasValue0;
    FLOAT4 result4 = biasValue0, result5 = biasValue0;
    FLOAT4 result2 = biasValue1, result3 = biasValue1;
    FLOAT4 result6 = biasValue1, result7 = biasValue1;
    for (auto z = 0; z < cst.input_slice; z++) {
        auto in40 = xy_in0[0];
        auto in41 = xy_in0[1];
        auto in44 = xy_in0[cst.output_width+0];
        auto in45 = xy_in0[cst.output_width+1];

        auto w0 = xy_wt[z];
        auto w1 = xy_wt[cst.input_slice+z];

        result0 += FLOAT4(in40 * w0);
        result1 += FLOAT4(in41 * w0);
        result4 += FLOAT4(in44 * w0);
        result5 += FLOAT4(in45 * w0);
        result2 += FLOAT4(in40 * w1);
        result3 += FLOAT4(in41 * w1);
        result6 += FLOAT4(in44 * w1);
        result7 += FLOAT4(in45 * w1);
        xy_in0 += cst.input_size;
    }

    int widthSize = min(cst.output_width - idx_w, 2);
    /* true            */ *xy_out = activate(ftype4(result0), cst.activation);
    if (widthSize > 1) {xy_out[1] = activate(ftype4(result1), cst.activation); }
    
    int heightSize = min(cst.output_height - idx_h, 2);
    if(heightSize > 1) {
        /* true         */ {xy_out[cst.output_width+0] = activate(ftype4(result4), cst.activation); }
        if (widthSize > 1) {xy_out[cst.output_width+1] = activate(ftype4(result5), cst.activation); }
    }
    
    int channelSize = min(cst.output_slice - idx_c, 2);
    if(channelSize > 1) {
        /* true         */  xy_out[cst.output_size]   = activate(ftype4(result2), cst.activation);
        if (widthSize > 1) {xy_out[cst.output_size+1] = activate(ftype4(result3), cst.activation); }
        
        if(heightSize > 1) {
            /* true         */ {xy_out[cst.output_size+cst.output_width+0] = activate(ftype4(result6), cst.activation); }
            if (widthSize > 1) {xy_out[cst.output_size+cst.output_width+1] = activate(ftype4(result7), cst.activation); }
        }
    }
}
