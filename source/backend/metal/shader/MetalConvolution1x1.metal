#define CONV_UNROLL (4)
#define CONV_UNROLL_L (8)

struct conv1x1_constants {
    int inputSize;
    int inputDepthQuad;
    int outputWidth;
    int outputHeight;
    int outputSize;
    int outputDepthQuad;
    int outputChannel;
    int batch;
    int blockCount;
    conv_activation_type activation;
    float scaleCoef;
};

kernel void conv1x1_g1z4(const device ftype4 *in            [[buffer(0)]],
                         device ftype4 *out                 [[buffer(1)]],
                         constant conv1x1_constants& cst    [[buffer(2)]],
                         const device ftype4x4 *wt          [[buffer(3)]],
                         const device ftype4 *biasTerms     [[buffer(4)]],
                         uint3 gid                          [[thread_position_in_grid]]) {
    if ((int)gid.x * CONV_UNROLL >= cst.outputSize || (int)gid.y >= cst.outputDepthQuad || (int)gid.z >= cst.batch) return;
    
    int rx = gid.x * CONV_UNROLL;
    int uz = gid.y;
    auto xy_wt = wt + uz * cst.inputDepthQuad;
    auto xy_in0  = in  + (int)gid.z  * cst.inputSize + rx + 0;
    auto xy_out = out + (int)gid.z * cst.outputSize + uz * cst.outputSize * cst.batch + rx;
    auto biasValue = FLOAT4(biasTerms[uz]);
    FLOAT4 result0 = biasValue, result1 = biasValue, result2 = biasValue, result3 = biasValue;
    int computeSize = min(cst.outputSize - rx, CONV_UNROLL);

    for (auto z = 0; z < cst.inputDepthQuad; z++) {
        auto in40 = *xy_in0;
        auto in41 = *(xy_in0 + 1);
        auto in42 = *(xy_in0 + 2);
        auto in43 = *(xy_in0 + 3);
        auto w = xy_wt[z];
        
        result0 += FLOAT4(in40 * w);
        result1 += FLOAT4(in41 * w);
        result2 += FLOAT4(in42 * w);
        result3 += FLOAT4(in43 * w);
        xy_in0 += cst.inputSize * cst.batch;
    }
    
    /* true                               */ *xy_out = activate(ftype4(result0), cst.activation);
    if (computeSize > 1) {xy_out[1] = activate(ftype4(result1), cst.activation); }
    if (computeSize > 2) {xy_out[2] = activate(ftype4(result2), cst.activation); }
    if (computeSize > 3) {xy_out[3] = activate(ftype4(result3), cst.activation); }
}

kernel void conv1x1_g1z4_w8(const device ftype4 *in            [[buffer(0)]],
                            device ftype4 *out                 [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                            const device MNN::char4x4 *wt      [[buffer(3)]],
                            const device ftype4 *biasTerms     [[buffer(4)]],
                            const device ftype4 *dequantScale  [[buffer(5)]],
                            uint3 gid                          [[thread_position_in_grid]]) {
    if ((int)gid.x * CONV_UNROLL >= cst.outputSize || (int)gid.y >= cst.outputDepthQuad || (int)gid.z >= cst.batch) return;

    int rx = gid.x * CONV_UNROLL;
    int uz = gid.y;
    auto xy_wt = wt + uz * cst.inputDepthQuad;
    auto xy_in0  = in  + (int)gid.z  * cst.inputSize + rx + 0;
    auto xy_out = out + (int)gid.z * cst.outputSize + uz * cst.outputSize * cst.batch + rx;
    auto biasValue = FLOAT4(biasTerms[uz]);
    FLOAT4 result0 = biasValue, result1 = biasValue, result2 = biasValue, result3 = biasValue;
    int computeSize = min(cst.outputSize - rx, CONV_UNROLL);
    int quadsPerBlock = (cst.inputDepthQuad + cst.blockCount - 1) / cst.blockCount;
    for (int bi=0; bi<cst.blockCount; ++bi) {
        FLOAT4 bs0 = FLOAT4(dequantScale[2 * (uz * cst.blockCount + bi) + 0]) / (FLOAT)cst.scaleCoef;
        FLOAT4 bs1 = FLOAT4(dequantScale[2 * (uz * cst.blockCount + bi) + 1]) / (FLOAT)cst.scaleCoef;
        FLOAT4 scale = bs0;
        FLOAT4 dequant_bias = bs1;
        int zmin = bi * quadsPerBlock;
        int zmax = min(zmin + quadsPerBlock, cst.inputDepthQuad);
        for (int z = zmin; z < zmax; z++) {
            auto in40 = (FLOAT4)*xy_in0;
            auto in41 = computeSize > 1 ? (FLOAT4)*(xy_in0 + 1) : (FLOAT4)0.0;
            auto in42 = computeSize > 2 ? (FLOAT4)*(xy_in0 + 2) : (FLOAT4)0.0;
            auto in43 = computeSize > 3 ? (FLOAT4)*(xy_in0 + 3) : (FLOAT4)0.0;
            auto w = xy_wt[z];
            FLOAT4x4 w_fp32 = FLOAT4x4(FLOAT4(w[0]), FLOAT4(w[1]), FLOAT4(w[2]), FLOAT4(w[3]));
            FLOAT4x4 w_dequant;
            for (int i = 0; i < 4; ++i) {
                w_dequant[i] = w_fp32[i] * scale[i] + dequant_bias[i];
            }
            result0 += FLOAT4(in40 * w_dequant);
            result1 += FLOAT4(in41 * w_dequant);
            result2 += FLOAT4(in42 * w_dequant);
            result3 += FLOAT4(in43 * w_dequant);
            xy_in0 += cst.inputSize * cst.batch;
        }
    }
    /* true */ 
    xy_out[0] = activate(ftype4(result0), cst.activation);
    if (computeSize > 1) {xy_out[1] = activate(ftype4(result1), cst.activation); }
    if (computeSize > 2) {xy_out[2] = activate(ftype4(result2), cst.activation); }
    if (computeSize > 3) {xy_out[3] = activate(ftype4(result3), cst.activation); }
}


kernel void conv1x1_g1z4_w4(const device ftype4 *in            [[buffer(0)]],
                            device ftype4 *out                 [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                            const device MNN::uchar4x2 *wt      [[buffer(3)]],
                            const device ftype4 *biasTerms     [[buffer(4)]],
                            const device ftype4 *dequantScale  [[buffer(5)]],
                            uint3 gid                          [[thread_position_in_grid]]) {
    if ((int)gid.x * CONV_UNROLL >= cst.outputSize || (int)gid.y >= cst.outputDepthQuad || (int)gid.z >= cst.batch) return;

    int rx = gid.x * CONV_UNROLL;
    int uz = gid.y;
    auto xy_wt = wt + uz * cst.inputDepthQuad;
    auto xy_in0  = in  + (int)gid.z  * cst.inputSize + rx + 0;
    auto xy_out = out + (int)gid.z * cst.outputSize + uz * cst.outputSize * cst.batch + rx;
    auto biasValue = FLOAT4(biasTerms[uz]);
    FLOAT4 result0 = biasValue, result1 = biasValue, result2 = biasValue, result3 = biasValue;
    int computeSize = min(cst.outputSize - rx, CONV_UNROLL);
    int quadsPerBlock = (cst.inputDepthQuad + cst.blockCount - 1) / cst.blockCount;
    for (int bi=0; bi<cst.blockCount; ++bi) {
        FLOAT4 scale = FLOAT4(dequantScale[2 * (uz * cst.blockCount + bi) + 0]) / (FLOAT)cst.scaleCoef;
        FLOAT4 dequant_bias = FLOAT4(dequantScale[2 * (uz * cst.blockCount + bi) + 1]) / (FLOAT)cst.scaleCoef;
        int zmin = bi * quadsPerBlock;
        int zmax = min(zmin + quadsPerBlock, cst.inputDepthQuad);
        for (int z = zmin; z < zmax; z++) {
            auto in40 = (FLOAT4)*xy_in0;
            auto in41 = (FLOAT4)*(xy_in0 + 1);
            auto in42 = (FLOAT4)*(xy_in0 + 2);
            auto in43 = (FLOAT4)*(xy_in0 + 3);
            MNN::uchar4x2 w_int4 = xy_wt[z];
            // MNN::char4x4  w_int8(char4(0));
            /* weight int4->float */
            //FLOAT4x4 w_fp32 = FLOAT4x4(FLOAT4(w[0]), FLOAT4(w[1]), FLOAT4(w[2]), FLOAT4(w[3]));
            FLOAT4x4 w_dequant;
            for (int i = 0; i < 4; ++i) {
                // ftype4 w4 = ftype4(w_fp32[i]);
                FLOAT4 w4 = FLOAT4((float)(w_int4[i][0] >> 4) - 8, (float)(w_int4[i][0] & 15) - 8, (float)(w_int4[i][1] >> 4) - 8, (float)(w_int4[i][1] & 15) - 8);
                FLOAT4 res = w4 * scale[i] + dequant_bias[i];
                w_dequant[i] = res;
            }

            result0 += FLOAT4(in40 * w_dequant);
            result1 += FLOAT4(in41 * w_dequant);
            result2 += FLOAT4(in42 * w_dequant);
            result3 += FLOAT4(in43 * w_dequant);
            xy_in0 += cst.inputSize * cst.batch;
        }
    }
    
    /* true */ 
    xy_out[0] = activate(ftype4(result0), cst.activation);
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
    if ((int)gid.x * CONV_UNROLL_L >= cst.outputSize || (int)gid.y >= cst.outputDepthQuad || (int)gid.z >= cst.batch) return;

    int rx = gid.x * CONV_UNROLL_L;
    int uz = gid.y;
    auto xy_wt = wt + uz * cst.inputDepthQuad;
    auto xy_in0  = in  + (int)gid.z  * cst.inputSize + rx + 0;

    auto xy_out = out + (int)gid.z * cst.outputSize + uz * cst.batch * cst.outputSize + rx;
    auto biasValue = FLOAT4(biasTerms[uz]);
    FLOAT4 result0 = biasValue, result1 = biasValue, result2 = biasValue, result3 = biasValue;
    FLOAT4 result4 = biasValue, result5 = biasValue, result6 = biasValue, result7 = biasValue;

    int computeSize = min(cst.outputSize - rx, CONV_UNROLL_L);
    for (auto z = 0; z < cst.inputDepthQuad; z++) {
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
            xy_in0 += cst.inputSize * cst.batch;
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

kernel void conv1x1_w4h4(const device ftype4 *in            [[buffer(0)]],
                         device ftype4 *out                 [[buffer(1)]],
                         constant conv1x1_constants& cst    [[buffer(2)]],
                         const device ftype4x4 *wt          [[buffer(3)]],
                         const device ftype4 *biasTerms     [[buffer(4)]],
                         uint3 gid                          [[thread_position_in_grid]]) {
    if ((int)gid.x * 16 >= cst.outputWidth || (int)gid.y >= cst.batch * cst.outputDepthQuad) return;

    int idx_w = gid.x << 4;
    int idx_h = 0;
    int idx_c = gid.y / cst.batch;
    int idx_b = gid.y % cst.batch;

    auto xy_wt = wt + idx_c * cst.inputDepthQuad;
    auto xy_in0  = in  + (int)idx_b * cst.inputSize + idx_h * cst.outputWidth + idx_w;

    auto xy_out = out + (int)idx_b * cst.outputSize + idx_c * cst.outputSize * cst.batch + idx_h * cst.outputWidth + idx_w;
    auto biasValue = FLOAT4(biasTerms[idx_c]);
    FLOAT4 result00 = biasValue, result01 = biasValue, result02 = biasValue, result03 = biasValue;
    FLOAT4 result10 = biasValue, result11 = biasValue, result12 = biasValue, result13 = biasValue;
    FLOAT4 result20 = biasValue, result21 = biasValue, result22 = biasValue, result23 = biasValue;
    FLOAT4 result30 = biasValue, result31 = biasValue, result32 = biasValue, result33 = biasValue;

    for (auto z = 0; z < cst.inputDepthQuad; z++) {
        auto in00 = xy_in0[0];
        auto in01 = xy_in0[1];
        auto in02 = xy_in0[2];
        auto in03 = xy_in0[3];
        auto in10 = xy_in0[4];
        auto in11 = xy_in0[5];
        auto in12 = xy_in0[6];
        auto in13 = xy_in0[7];
        
        auto in20 = xy_in0[8];
        auto in21 = xy_in0[9];
        auto in22 = xy_in0[10];
        auto in23 = xy_in0[11];
        auto in30 = xy_in0[12];
        auto in31 = xy_in0[13];
        auto in32 = xy_in0[14];
        auto in33 = xy_in0[15];


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
        
        xy_in0 += cst.inputSize * cst.batch;
    }

    int widthSize = min(cst.outputWidth - idx_w, 16);
    /* true            */ *xy_out = activate(ftype4(result00), cst.activation);
    if (widthSize > 1) {xy_out[1] = activate(ftype4(result01), cst.activation); }
    if (widthSize > 2) {xy_out[2] = activate(ftype4(result02), cst.activation); }
    if (widthSize > 3) {xy_out[3] = activate(ftype4(result03), cst.activation); }
    if (widthSize > 4) {xy_out[4] = activate(ftype4(result10), cst.activation); }
    if (widthSize > 5) {xy_out[5] = activate(ftype4(result11), cst.activation); }
    if (widthSize > 6) {xy_out[6] = activate(ftype4(result12), cst.activation); }
    if (widthSize > 7) {xy_out[7] = activate(ftype4(result13), cst.activation); }
    if (widthSize > 8) {xy_out[8] = activate(ftype4(result20), cst.activation); }
    if (widthSize > 9) {xy_out[9] = activate(ftype4(result21), cst.activation); }
    if (widthSize > 10) {xy_out[10] = activate(ftype4(result22), cst.activation); }
    if (widthSize > 11) {xy_out[11] = activate(ftype4(result23), cst.activation); }
    if (widthSize > 12) {xy_out[12] = activate(ftype4(result30), cst.activation); }
    if (widthSize > 13) {xy_out[13] = activate(ftype4(result31), cst.activation); }
    if (widthSize > 14) {xy_out[14] = activate(ftype4(result32), cst.activation); }
    if (widthSize > 15) {xy_out[15] = activate(ftype4(result33), cst.activation); }
}


kernel void conv1x1_w2c2(const device ftype4 *in            [[buffer(0)]],
                         device ftype4 *out                 [[buffer(1)]],
                         constant conv1x1_constants& cst    [[buffer(2)]],
                         const device ftype4x4 *wt          [[buffer(3)]],
                         const device ftype4 *biasTerms     [[buffer(4)]],
                         uint3 gid                          [[thread_position_in_grid]]) {
    if ((int)gid.x * 2 >= cst.outputWidth || (int)gid.y * 2 >= cst.batch * cst.outputDepthQuad) return;

    int channel_pack = (cst.outputChannel + 7) >> 3;
    int idx_w = gid.x << 1;
    int idx_h = 0;
    int idx_c = (gid.y % channel_pack) << 1;
    int idx_b = gid.y / channel_pack;
    
    if(idx_b >=  cst.batch || idx_c >= cst.outputDepthQuad) return;
    auto xy_wt = wt + idx_c * cst.inputDepthQuad;
    auto xy_in0  = in  + (int)idx_b * cst.inputSize + idx_h * cst.outputWidth + idx_w;

    auto xy_out = out + (int)idx_b * cst.outputSize + idx_c * cst.outputSize * cst.batch + idx_h * cst.outputWidth + idx_w;
    auto biasValue0 = FLOAT4(biasTerms[idx_c]);
    auto biasValue1 = FLOAT4(biasTerms[idx_c+1]);

    FLOAT4 result0 = biasValue0, result1 = biasValue0;
    FLOAT4 result4 = biasValue1, result5 = biasValue1;

    for (auto z = 0; z < cst.inputDepthQuad; z++) {
        auto in40 = xy_in0[0];
        auto in41 = xy_in0[1];

        auto w0 = xy_wt[z];
        auto w1 = xy_wt[cst.inputDepthQuad+z];

        result0 += FLOAT4(in40 * w0);
        result1 += FLOAT4(in41 * w0);
        result4 += FLOAT4(in40 * w1);
        result5 += FLOAT4(in41 * w1);
        xy_in0 += cst.inputSize * cst.batch;
    }

    int widthSize = min(cst.outputWidth - idx_w, 2);
    /* true            */ *xy_out = activate(ftype4(result0), cst.activation);
    if (widthSize > 1) {xy_out[1] = activate(ftype4(result1), cst.activation); }
    
    int channelSize = min(cst.outputDepthQuad - idx_c, 2);
    if(channelSize > 1) {
        /* true         */ {xy_out[cst.outputSize * cst.batch +0] = activate(ftype4(result4), cst.activation); }
        if (widthSize > 1) {xy_out[cst.outputSize * cst.batch +1] = activate(ftype4(result5), cst.activation); }
    }
}

kernel void conv1x1_w4c2(const device ftype4 *in            [[buffer(0)]],
                         device ftype4 *out                 [[buffer(1)]],
                         constant conv1x1_constants& cst    [[buffer(2)]],
                         const device ftype4x4 *wt          [[buffer(3)]],
                         const device ftype4 *biasTerms     [[buffer(4)]],
                         uint3 gid                          [[thread_position_in_grid]]) {
    if ((int)gid.x * 4 >= cst.outputWidth || (int)gid.y * 2 >= cst.batch * cst.outputDepthQuad) return;

    int channel_pack = (cst.outputChannel + 7) >> 3;
    int idx_w = gid.x << 2;
    int idx_h = 0;
    int idx_c = (gid.y % channel_pack) << 1;
    int idx_b = gid.y / channel_pack;

    if(idx_b >=  cst.batch || idx_c >= cst.outputDepthQuad) return;
    auto xy_wt = wt + idx_c * cst.inputDepthQuad;
    auto xy_in0  = in  + (int)idx_b * cst.inputSize + idx_h * cst.outputWidth + idx_w;

    auto xy_out = out + (int)idx_b * cst.outputSize + idx_c * cst.outputSize * cst.batch + idx_h * cst.outputWidth + idx_w;
    auto biasValue0 = FLOAT4(biasTerms[idx_c]);
    auto biasValue1 = FLOAT4(biasTerms[idx_c+1]);

    FLOAT4 result0 = biasValue0, result1 = biasValue0;
    FLOAT4 result4 = biasValue0, result5 = biasValue0;
    FLOAT4 result2 = biasValue1, result3 = biasValue1;
    FLOAT4 result6 = biasValue1, result7 = biasValue1;
    for (auto z = 0; z < cst.inputDepthQuad; z++) {
        auto in40 = xy_in0[0];
        auto in41 = xy_in0[1];
        auto in44 = xy_in0[2];
        auto in45 = xy_in0[3];

        auto w0 = xy_wt[z];
        auto w1 = xy_wt[cst.inputDepthQuad+z];

        result0 += FLOAT4(in40 * w0);
        result1 += FLOAT4(in41 * w0);
        result4 += FLOAT4(in44 * w0);
        result5 += FLOAT4(in45 * w0);
        result2 += FLOAT4(in40 * w1);
        result3 += FLOAT4(in41 * w1);
        result6 += FLOAT4(in44 * w1);
        result7 += FLOAT4(in45 * w1);
        xy_in0 += cst.inputSize * cst.batch;
    }

    int widthSize = min(cst.outputWidth - idx_w, 4);
    /* true            */ *xy_out = activate(ftype4(result0), cst.activation);
    if (widthSize > 1) {xy_out[1] = activate(ftype4(result1), cst.activation); }
    if (widthSize > 2) {xy_out[2] = activate(ftype4(result4), cst.activation); }
    if (widthSize > 3) {xy_out[3] = activate(ftype4(result5), cst.activation); }
        
    int channelSize = min(cst.outputDepthQuad - idx_c, 2);
    if(channelSize > 1) {
        /* true         */  xy_out[cst.outputSize * cst.batch]   = activate(ftype4(result2), cst.activation);
        if (widthSize > 1) {xy_out[cst.outputSize * cst.batch +1] = activate(ftype4(result3), cst.activation); }
        if (widthSize > 2) {xy_out[cst.outputSize * cst.batch +2] = activate(ftype4(result6), cst.activation); }
        if (widthSize > 3) {xy_out[cst.outputSize * cst.batch +3] = activate(ftype4(result7), cst.activation); }
    }
}
