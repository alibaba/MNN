#define CONV_UNROLL (4)
#define CONV_UNROLL_L (8)

#define INIT_SIMDGROUP_MATRIX(a, b, d) \
    simdgroup_T8x8 sga[a];\
    simdgroup_T8x8 sgb[b];\
    simdgroup_T8x8 sgd[d];\
    for (int i = 0; i < d; i++){\
        sgd[i] = make_filled_simdgroup_matrix<FLOAT, 8>(0.f);\
    }

#define SIMDGROUP_MATRIX_FMA(a, b) \
    for(int j=0; j<b; j++) {\
        for(int i=0; i<a; i++) {\
            simdgroup_multiply_accumulate(sgd[j*a+i], sga[i], sgb[j], sgd[j*a+i]);\
        }\
    }
    
#define SIMDGROUP_MATRIX_STORE(ptr, d) \
    for(int i=0; i<d; i++) {\
        simdgroup_store(sgd[i], ptr + 64*i, 8);\
    }

struct conv1x1_constants {
    int input_size;
    int input_slice;
    int output_width;
    int output_height;
    int output_size;
    int output_slice;
    int output_channel;
    int batch;
    int block_size;
    conv_activation_type activation;
    float scale_coef;
};

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
    auto xy_in0  = in  + (int)gid.z  * cst.input_size + rx + 0;
    auto xy_out = out + (int)gid.z * cst.output_size + uz * cst.output_size * cst.batch + rx;
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
        xy_in0 += cst.input_size * cst.batch;
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
    if ((int)gid.x * CONV_UNROLL >= cst.output_size || (int)gid.y >= cst.output_slice || (int)gid.z >= cst.batch) return;

    int rx = gid.x * CONV_UNROLL;
    int uz = gid.y;
    auto xy_wt = wt + uz * cst.input_slice;
    auto xy_in0  = in  + (int)gid.z  * cst.input_size + rx + 0;
    auto xy_out = out + (int)gid.z * cst.output_size + uz * cst.output_size * cst.batch + rx;
    auto biasValue = FLOAT4(biasTerms[uz]);
    FLOAT4 result0 = biasValue, result1 = biasValue, result2 = biasValue, result3 = biasValue;
    int computeSize = min(cst.output_size - rx, CONV_UNROLL);
    int block = (cst.input_slice + cst.block_size - 1) / cst.block_size;
    for (int bi=0; bi<cst.block_size; ++bi) {
        FLOAT4 bs0 = FLOAT4(dequantScale[2 * (uz * cst.block_size + bi) + 0]) / (FLOAT)cst.scale_coef;
        FLOAT4 bs1 = FLOAT4(dequantScale[2 * (uz * cst.block_size + bi) + 1]) / (FLOAT)cst.scale_coef;
        FLOAT4 scale = bs0;
        FLOAT4 dequant_bias = bs1;
        int zmin = bi * block;
        int zmax = min(zmin + block, cst.input_slice);
        for (int z = zmin; z < zmax; z++) {
            auto in40 = (FLOAT4)*xy_in0;
            auto in41 = (FLOAT4)*(xy_in0 + 1);
            auto in42 = (FLOAT4)*(xy_in0 + 2);
            auto in43 = (FLOAT4)*(xy_in0 + 3);
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
            xy_in0 += cst.input_size * cst.batch;
        }
    }
    /* true */ 
    xy_out[0] = activate(ftype4(result0), cst.activation);
    if (computeSize > 1) {xy_out[1] = activate(ftype4(result1), cst.activation); }
    if (computeSize > 2) {xy_out[2] = activate(ftype4(result2), cst.activation); }
    if (computeSize > 3) {xy_out[3] = activate(ftype4(result3), cst.activation); }
}


kernel void conv1x1_gemm_16x16_w4(const device ftype4 *in            [[buffer(0)]],
                            device ftype4 *out                 [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                            const device uchar2 *wt      [[buffer(3)]],
                            const device ftype4 *biasTerms     [[buffer(4)]],
                            const device ftype4 *dequantScale  [[buffer(5)]],
                            uint3 gid                          [[threadgroup_position_in_grid]],
                            uint                  tiitg[[thread_index_in_threadgroup]],
                            uint                  sgitg[[simdgroup_index_in_threadgroup]]) {
    /*
     // Read:
     ftype 0~127   ---> input: [M16, K8]
     ftype 128~255 ---> input: [K8, N16]
     // Write:
     ftype 0~255 ---> input: [N2, M2, M8, N8]
     */
    threadgroup FLOAT4 sdata[256] = {0.f};
    
    INIT_SIMDGROUP_MATRIX(2, 2, 4);
    
    int rx = gid.x;// M/16
    int uz = gid.y;// N/16
    
    int kl = tiitg / 16;
    int rcl = tiitg % 16;
//    int kl = tiitg % 2;
//    int rcl = tiitg / 2;
    
    // boundary limit
    int idx_n4 = (4 * uz + rcl / 4) < cst.output_slice ? (4 * uz + rcl / 4) : (cst.output_slice - 1);
    int idx_m  = (16 * rx + rcl) < cst.input_size * cst.batch ? (16 * rx + rcl) : (cst.input_size * cst.batch - 1);
    
    auto xy_wt = wt +  (idx_n4 * cst.input_slice + 0) * 4 + rcl % 4;// [N/4, K/4, N4, K4]
    auto xy_in0  = in + idx_m + cst.input_size * cst.batch * kl;// [K/4, M, K4]
    auto xy_out = out + (4 * uz + 2 * kl) * cst.output_size * cst.batch + idx_m;// [N/4, M, N4]
    
    int block = (cst.input_slice + cst.block_size - 1) / cst.block_size;
    for (int bi=0; bi<cst.block_size; ++bi) {
        // [N/4, cst.block_size, 2/*scale_bias*/, N4]
        FLOAT4 scale = FLOAT4(dequantScale[2 * (idx_n4 * cst.block_size + bi) + 0]) / (FLOAT)cst.scale_coef;
        FLOAT4 dequant_bias = FLOAT4(dequantScale[2 * (idx_n4 * cst.block_size + bi) + 1]) / (FLOAT)cst.scale_coef;
        int zmin = bi * block;
        int zmax = min(zmin + block, cst.input_slice);

        for (int z = zmin + kl; z < zmax; z += 8) {
            #pragma unroll(4)
            for(int i = 0; i < 4; i++) {
                sdata[64 * i + 2* rcl + kl] = (FLOAT4)*xy_in0;
                xy_in0 += 2 * cst.input_size * cst.batch;
            }
            
            #pragma unroll(4)
            for(int i = 0; i < 4; i++) {
                uchar2 w_int40 = xy_wt[4 * (z + 2*i)]; // [N/4, K/4, N4, K4]
                FLOAT4 w40 = FLOAT4((float)(w_int40[0] >> 4) - 8, (float)(w_int40[0] & 15) - 8, (float)(w_int40[1] >> 4) - 8, (float)(w_int40[1] & 15) - 8);
                
                FLOAT4 res = w40 * scale[rcl % 4] + dequant_bias[rcl % 4];
                ((threadgroup FLOAT*)sdata)[256 * i + 128 + (kl * 4 + 0) * 16 + rcl] = res[0];
                ((threadgroup FLOAT*)sdata)[256 * i + 128 + (kl * 4 + 1) * 16 + rcl] = res[1];
                ((threadgroup FLOAT*)sdata)[256 * i + 128 + (kl * 4 + 2) * 16 + rcl] = res[2];
                ((threadgroup FLOAT*)sdata)[256 * i + 128 + (kl * 4 + 3) * 16 + rcl] = res[3];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            #pragma unroll(4)
            for(int i = 0; i < 4; i++) {
                simdgroup_load(sga[0], (const threadgroup FLOAT*)sdata + 256*i, 8);
                simdgroup_load(sga[1], ((const threadgroup FLOAT*)sdata) + 64 + 256*i, 8);
                simdgroup_barrier(mem_flags::mem_none);
                
                simdgroup_load(sgb[0], ((const threadgroup FLOAT*)sdata) + 128 + 256*i, 16);
                simdgroup_load(sgb[1], ((const threadgroup FLOAT*)sdata) + 136 + 256*i, 16);
                
                SIMDGROUP_MATRIX_FMA(2, 2);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    
    SIMDGROUP_MATRIX_STORE((threadgroup FLOAT*)sdata, 4);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if((16 * rx + rcl) < cst.input_size * cst.batch) {
        if((4 * uz + 2 * kl) < cst.output_slice) {
            xy_out[0] =  activate(ftype4(sdata[(kl * 16 + rcl) * 2 + 0] + FLOAT4(biasTerms[4 * uz + 2 * kl + 0])), cst.activation);
        }
        if((4 * uz + 2 * kl + 1) < cst.output_slice) {
            xy_out[cst.output_size * cst.batch] = activate(ftype4(sdata[(kl * 16 + rcl) * 2 + 1] + FLOAT4(biasTerms[4 * uz + 2 * kl + 1])), cst.activation);
        }
    }
}

kernel void conv1x1_gemm_32x16_w4(const device ftype4 *in            [[buffer(0)]],
                            device ftype4 *out                 [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                            const device uchar2 *wt      [[buffer(3)]],
                            const device ftype4 *biasTerms     [[buffer(4)]],
                            const device ftype4 *dequantScale  [[buffer(5)]],
                            uint3 gid                          [[threadgroup_position_in_grid]],
                            uint                  tiitg[[thread_index_in_threadgroup]],
                            uint                  sgitg[[simdgroup_index_in_threadgroup]]) {
    /*
     // Read:
     ftype 0~255   ---> input: [M32, K8]
     ftype 256~383 ---> input: [K8, N16]
     // Write:
     ftype 0~511 ---> input: [N2, M4, M8, N8]
     */
    threadgroup FLOAT4 sdata[128] = {0.f};
    
    INIT_SIMDGROUP_MATRIX(4, 2, 8);
    
    int rx = gid.x;// M/32
    int uz = gid.y;// N/16
    
    int kl = tiitg % 2;
    int rcl = tiitg / 2;
    
    const int size_m = cst.input_size * cst.batch;
    
    // boundary limit
    int idx_n4 = (4 * uz + rcl / 4) < cst.output_slice ? (4 * uz + rcl / 4) : (cst.output_slice - 1);
    int idx_m0  = (16 * rx + rcl) <  size_m ? (16 * rx + rcl) : (size_m - 1);
    int idx_m1  = (16 * rx + rcl) + size_m / 2 < size_m ? (16 * rx + rcl) + size_m / 2: (size_m - 1);

    auto xy_wt = wt +  (idx_n4 * cst.input_slice + 0) * 4 + rcl % 4;// [N/4, K/4, N4, K4]
    auto xy_in0  = in + idx_m0 + cst.input_size * cst.batch * kl;// [K/4, M2, M/2, K4]
    auto xy_in1  = in + idx_m1 + cst.input_size * cst.batch * kl;// [K/4, M2, M/2, K4]

    auto xy_out0 = out + (4 * uz + 2 * kl) * cst.output_size * cst.batch + idx_m0;// [N/4, M, N4]
    auto xy_out1 = out + (4 * uz + 2 * kl) * cst.output_size * cst.batch + idx_m1;// [N/4, M, N4]

    int block = (cst.input_slice + cst.block_size - 1) / cst.block_size;
    for (int bi=0; bi<cst.block_size; ++bi) {
        // [N/4, cst.block_size, 2/*scale_bias*/, N4]
        FLOAT4 scale = FLOAT4(dequantScale[2 * (idx_n4 * cst.block_size + bi) + 0]) / (FLOAT)cst.scale_coef;
        FLOAT4 dequant_bias = FLOAT4(dequantScale[2 * (idx_n4 * cst.block_size + bi) + 1]) / (FLOAT)cst.scale_coef;
        int zmin = bi * block;
        int zmax = min(zmin + block, cst.input_slice);

        for (int z = zmin + kl; z < zmax; z += 2) {
            sdata[2* rcl + kl] = (FLOAT4)*xy_in0;
            sdata[32 + 2* rcl + kl] = (FLOAT4)*xy_in1;

            uchar2 w_int4 = xy_wt[4*z]; // [N/4, K/4, N4, K4]
            FLOAT4 w4 = FLOAT4((float)(w_int4[0] >> 4) - 8, (float)(w_int4[0] & 15) - 8, (float)(w_int4[1] >> 4) - 8, (float)(w_int4[1] & 15) - 8);
            FLOAT4 res = w4 * scale[rcl % 4] + dequant_bias[rcl % 4];
            //            sdata[32 + 2* rcl + kl] = res;
            ((threadgroup FLOAT*)sdata)[256 + (kl * 4 + 0) * 16 + rcl] = res[0];
            ((threadgroup FLOAT*)sdata)[256 + (kl * 4 + 1) * 16 + rcl] = res[1];
            ((threadgroup FLOAT*)sdata)[256 + (kl * 4 + 2) * 16 + rcl] = res[2];
            ((threadgroup FLOAT*)sdata)[256 + (kl * 4 + 3) * 16 + rcl] = res[3];
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            simdgroup_load(sga[0], (const threadgroup FLOAT*)sdata, 8);
            simdgroup_load(sga[1], ((const threadgroup FLOAT*)sdata) + 64, 8);
            simdgroup_load(sga[2], ((const threadgroup FLOAT*)sdata) + 128, 8);
            simdgroup_load(sga[3], ((const threadgroup FLOAT*)sdata) + 192, 8);
            
            //            simdgroup_load(sgb[0], (const threadgroup FLOAT*)sdata + 128, 8, 0, true);
            //            simdgroup_load(sgb[1], ((const threadgroup FLOAT*)sdata) + 192, 8, 0, true);
            simdgroup_load(sgb[0], ((const threadgroup FLOAT*)sdata) + 256, 16);
            simdgroup_load(sgb[1], ((const threadgroup FLOAT*)sdata) + 264, 16);
            
            SIMDGROUP_MATRIX_FMA(4, 2);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            xy_in0 += 2 * cst.input_size * cst.batch;
            xy_in1 += 2 * cst.input_size * cst.batch;

        }
    }
    
    SIMDGROUP_MATRIX_STORE((threadgroup FLOAT*)sdata, 8);
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if((16 * rx + rcl) < size_m) {
        if((4 * uz + 2 * kl) < cst.output_slice) {
            xy_out0[0] =  activate(ftype4(sdata[(kl * 32 + rcl) * 2 + 0] + FLOAT4(biasTerms[4 * uz + 2 * kl + 0])), cst.activation);
        }
        if((4 * uz + 2 * kl + 1) < cst.output_slice) {
            xy_out0[cst.output_size * cst.batch] = activate(ftype4(sdata[(kl * 32 + rcl) * 2 + 1] + FLOAT4(biasTerms[4 * uz + 2 * kl + 1])), cst.activation);
        }
    }
    if((16 * rx + rcl) + size_m / 2 < size_m) {
        if((4 * uz + 2 * kl) < cst.output_slice) {
            xy_out1[0] =  activate(ftype4(sdata[(kl * 32 + 16 + rcl) * 2 + 0] + FLOAT4(biasTerms[4 * uz + 2 * kl + 0])), cst.activation);
        }
        if((4 * uz + 2 * kl + 1) < cst.output_slice) {
            xy_out1[cst.output_size * cst.batch] = activate(ftype4(sdata[(kl * 32 + 16 + rcl) * 2 + 1] + FLOAT4(biasTerms[4 * uz + 2 * kl + 1])), cst.activation);
        }
    }
}

kernel void conv1x1_gemm_16x32_w4(const device ftype4 *in            [[buffer(0)]],
                            device ftype4 *out                 [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                            const device uchar2 *wt      [[buffer(3)]],
                            const device ftype4 *biasTerms     [[buffer(4)]],
                            const device ftype4 *dequantScale  [[buffer(5)]],
                            uint3 gid                          [[threadgroup_position_in_grid]],
                            uint                  tiitg[[thread_index_in_threadgroup]],
                            uint                  sgitg[[simdgroup_index_in_threadgroup]]) {
    /*
     // Read:
     ftype 0~127   ---> input: [M16, K8]
     ftype 128~383 ---> input: [K8, N32]
     // Write:
     ftype 0~511 ---> input: [N2, N2, M2, M8, N8]
     */
    threadgroup FLOAT4 sdata[128] = {0.f};
    
    INIT_SIMDGROUP_MATRIX(2, 4, 8);
    
    int rx = gid.x;// M/16
    int uz = gid.y;// N/32
    
    int kl = tiitg % 2;
    int rcl = tiitg / 2;
    
    // boundary limit
    int idx_n40 = (4 * uz + rcl / 4) < cst.output_slice ? (4 * uz + rcl / 4) : (cst.output_slice - 1);
    int idx_n41 = (4 * uz + rcl / 4) + cst.output_slice / 2 < cst.output_slice ? (4 * uz + rcl / 4) + cst.output_slice / 2 : (cst.output_slice - 1);
    
    int idx_m  = (16 * rx + rcl) < cst.input_size * cst.batch ? (16 * rx + rcl) : (cst.input_size * cst.batch - 1);
    
    auto xy_wt0 = wt +  (idx_n40 * cst.input_slice + 0) * 4 + (rcl % 4);// [N2, N/8, K/4, N4, K4]
    auto xy_wt1 = wt +  (idx_n41 * cst.input_slice + 0) * 4 + (rcl % 4);// [N2, N/8, K/4, N4, K4]

    auto xy_in0  = in + idx_m + cst.input_size * cst.batch * kl;// [K/4, M, K4]
    auto xy_out = out + (4 * uz + 2 * kl) * cst.output_size * cst.batch + idx_m;// [N2, N/8, M, N4]
    
    int block = (cst.input_slice + cst.block_size - 1) / cst.block_size;
    for (int bi=0; bi<cst.block_size; ++bi) {
        // [N/4, cst.block_size, 2/*scale_bias*/, N4]
        FLOAT4 scale0 = FLOAT4(dequantScale[2 * (idx_n40 * cst.block_size + bi) + 0]) / (FLOAT)cst.scale_coef;
        FLOAT4 dequant_bias0 = FLOAT4(dequantScale[2 * (idx_n40 * cst.block_size + bi) + 1]) / (FLOAT)cst.scale_coef;
        FLOAT4 scale1 = FLOAT4(dequantScale[2 * (idx_n41 * cst.block_size + bi) + 0]) / (FLOAT)cst.scale_coef;
        FLOAT4 dequant_bias1 = FLOAT4(dequantScale[2 * (idx_n41 * cst.block_size + bi) + 1]) / (FLOAT)cst.scale_coef;
        int zmin = bi * block;
        int zmax = min(zmin + block, cst.input_slice);

        for (int z = zmin + kl; z < zmax; z += 2) {
            sdata[2* rcl + kl] = (FLOAT4)*xy_in0;
            
            {
                uchar2 w_int4 = xy_wt0[4*z]; // [N/4, K/4, N4, K4]
                FLOAT4 w4 = FLOAT4((float)(w_int4[0] >> 4) - 8, (float)(w_int4[0] & 15) - 8, (float)(w_int4[1] >> 4) - 8, (float)(w_int4[1] & 15) - 8);
                FLOAT4 res = w4 * scale0[rcl % 4] + dequant_bias0[rcl % 4];
                //            sdata[32 + 2* rcl + kl] = res;
                ((threadgroup FLOAT*)sdata)[128 + (kl * 4 + 0) * 32 + rcl] = res[0];
                ((threadgroup FLOAT*)sdata)[128 + (kl * 4 + 1) * 32 + rcl] = res[1];
                ((threadgroup FLOAT*)sdata)[128 + (kl * 4 + 2) * 32 + rcl] = res[2];
                ((threadgroup FLOAT*)sdata)[128 + (kl * 4 + 3) * 32 + rcl] = res[3];
            }
            {
                uchar2 w_int4 = xy_wt1[4*z]; // [N/4, K/4, N4, K4]
                FLOAT4 w4 = FLOAT4((float)(w_int4[0] >> 4) - 8, (float)(w_int4[0] & 15) - 8, (float)(w_int4[1] >> 4) - 8, (float)(w_int4[1] & 15) - 8);
                FLOAT4 res = w4 * scale1[rcl % 4] + dequant_bias1[rcl % 4];
                //            sdata[32 + 2* rcl + kl] = res;
                ((threadgroup FLOAT*)sdata)[128 + (kl * 4 + 0) * 32 + 16 + rcl] = res[0];
                ((threadgroup FLOAT*)sdata)[128 + (kl * 4 + 1) * 32 + 16 + rcl] = res[1];
                ((threadgroup FLOAT*)sdata)[128 + (kl * 4 + 2) * 32 + 16 + rcl] = res[2];
                ((threadgroup FLOAT*)sdata)[128 + (kl * 4 + 3) * 32 + 16 + rcl] = res[3];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_load(sga[0], (const threadgroup FLOAT*)sdata, 8);
            simdgroup_load(sga[1], ((const threadgroup FLOAT*)sdata) + 64, 8);
            
            simdgroup_load(sgb[0], ((const threadgroup FLOAT*)sdata) + 128, 32);
            simdgroup_load(sgb[1], ((const threadgroup FLOAT*)sdata) + 136, 32);
            simdgroup_load(sgb[2], ((const threadgroup FLOAT*)sdata) + 144, 32);
            simdgroup_load(sgb[3], ((const threadgroup FLOAT*)sdata) + 152, 32);
            
            SIMDGROUP_MATRIX_FMA(2, 4);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            xy_in0 += 2 * cst.input_size * cst.batch;

        }
    }
    
    SIMDGROUP_MATRIX_STORE((threadgroup FLOAT*)sdata, 8);
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if((16 * rx + rcl) < cst.input_size * cst.batch) {
        if(4 * uz + 2 * kl < cst.output_slice) {
            xy_out[0] =  activate(ftype4(sdata[(kl * 16 + rcl) * 2 + 0] + FLOAT4(biasTerms[4 * uz + 2 * kl])), cst.activation);
        }
        if(4 * uz + 2 * kl + 1 < cst.output_slice) {
            xy_out[cst.output_size * cst.batch] = activate(ftype4(sdata[(kl * 16 + rcl) * 2 + 1] + FLOAT4(biasTerms[4 * uz + 2 * kl + 1])), cst.activation);
        }
        if(cst.output_slice / 2 + 4 * uz + 2 * kl < cst.output_slice) {
            xy_out[cst.output_slice / 2 * cst.output_size * cst.batch] = activate(ftype4(sdata[((kl + 2) * 16 + rcl) * 2 + 0] + FLOAT4(biasTerms[cst.output_slice / 2 + 4 * uz + 2 * kl])), cst.activation);
        }
        if(cst.output_slice / 2 + 4 * uz + 2 * kl + 1 < cst.output_slice) {
            xy_out[(cst.output_slice / 2 + 1) * cst.output_size * cst.batch] = activate(ftype4(sdata[((kl + 2) * 16 + rcl) * 2 + 1] + FLOAT4(biasTerms[cst.output_slice / 2 + 4 * uz + 2 * kl + 1])), cst.activation);
        }
    }
}


kernel void conv1x1_gemm_32x64_w4(const device ftype2 *in            [[buffer(0)]],
                            device ftype4 *out                 [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                            const device uchar2 *wt      [[buffer(3)]],
                            const device ftype4 *biasTerms     [[buffer(4)]],
                            const device ftype4 *dequantScale  [[buffer(5)]],
                            uint3 gid                          [[threadgroup_position_in_grid]],
                            uint                  tiitg[[thread_index_in_threadgroup]],
                            uint                  tiisg[[thread_index_in_simdgroup]],
                            uint                  sgitg[[simdgroup_index_in_threadgroup]]) {
    /*
     // Read:
     ftype 0~255   ---> input: [M32, K8]
     ftype 256~767 ---> weight: [K8, N64]
     // Write:
     ftype 0~2047 ---> input: [M2, N2, N2, N2, M2, M8, N8]
     */
    threadgroup FLOAT4 sdata[512] = {0.f};
    
    INIT_SIMDGROUP_MATRIX(2, 4, 8);
    
    int rx = gid.x;// M/32
    int uz = gid.y;// N/64
    
    int kl = tiitg % 2;// 0~1 -> inner K
    int rcl = tiitg / 2;// 0~63
    int ko = rcl % 2;// 0~1 -> outter K
    int ml = rcl / 2;// 0~31 -> m
    int ni = rcl % 4;// 0~3 -> inner N
    int no = rcl / 4;// 0~15 -> outter N
    
    /** input:
     threadgroup: [M32, K8]
     each thread: K2
     layout: [K/4, M, K4] -> [K/8, K2, M/32, M32, K2, K2]
     index : [K/8, ko, rx, ml, kl, K2]
     */
    /** weight:
     threadgroup: [K8, N64]
     each thread: K4
     layout: [N/4, K/4, N4, K4] -> [N/64, N16, K/8, K2, N4, K4]
     index : [uz, no, K/8, kl, ni, K4]
     */
    /** output:
     threadgroup: [M32, N64]
     each thread: M4N4
     layout: [N/4, M, N4] -> [N/16, N4, M, N4]
     index : [uz*4+(2*ko+kl), N4, idx_m, N4]
     */
    
    // boundary limit

    int idx_n40 = (uz * 16 + no) < cst.output_slice ? (uz * 16 + no) : (cst.output_slice - 1);
    int idx_m  = (rx * 32 + ml) < cst.input_size * cst.batch ? (rx * 32 + ml) : (cst.input_size * cst.batch - 1);
    
    auto xy_wt0 = wt +  ((idx_n40 * cst.input_slice / 2 + 0) * 2 + kl) * 4 + ni;// [N/4, K/4, N4, K4]

    auto xy_in0  = in + ((0 * 2 + ko) * cst.input_size * cst.batch + idx_m) * 2 + kl;// [K/4, M, K2, K2]
    auto xy_out = out + ((4 * uz + 2 * ko + kl) * 4 + 0) * cst.output_size * cst.batch + idx_m;// [N2, N/8, M, N4]
    
    const int idx_sa = ml * 8 + ko * 4 + kl * 2;
    const int idx_sb = 256 + (kl * 4 + 0) * 64 + rcl;
    int block = (cst.input_slice + cst.block_size - 1) / cst.block_size;
    for (int bi=0; bi<cst.block_size; ++bi) {
        // [N/4, cst.block_size, 2/*scale_bias*/, N4]
        FLOAT4 scale0 = FLOAT4(dequantScale[2 * (idx_n40 * cst.block_size + bi) + 0]) / (FLOAT)cst.scale_coef;
        FLOAT4 dequant_bias0 = FLOAT4(dequantScale[2 * (idx_n40 * cst.block_size + bi) + 1]) / (FLOAT)cst.scale_coef;

        int zmin = bi * block;
        int zmax = min(zmin + block, cst.input_slice);
        
        for (int z = zmin; z < zmax; z += 2) {
            FLOAT2 data = (FLOAT2)*xy_in0;
            ((threadgroup FLOAT*)sdata)[idx_sa] = data[0];
            ((threadgroup FLOAT*)sdata)[idx_sa + 1] = data[1];
            
            {
                uchar2 w_int4 = xy_wt0[4*z]; // [N/4, K/4, N4, K4]
                FLOAT4 w4 = FLOAT4((float)(w_int4[0] >> 4) - 8, (float)(w_int4[0] & 15) - 8, (float)(w_int4[1] >> 4) - 8, (float)(w_int4[1] & 15) - 8);
                FLOAT4 res = w4 * scale0[ni] + dequant_bias0[ni];
                //            sdata[32 + 2* rcl + kl] = res;
                ((threadgroup FLOAT*)sdata)[idx_sb] = res[0];
                ((threadgroup FLOAT*)sdata)[idx_sb + 64] = res[1];
                ((threadgroup FLOAT*)sdata)[idx_sb + 128] = res[2];
                ((threadgroup FLOAT*)sdata)[idx_sb + 192] = res[3];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            
            const threadgroup FLOAT * sdata_a = (const threadgroup FLOAT*)sdata + 16*8*(sgitg/2);
            const threadgroup FLOAT * sdata_b = (const threadgroup FLOAT*)sdata + 32*8 + 32*(sgitg%2);

            simdgroup_load(sga[0], (const threadgroup FLOAT*)sdata_a, 8);
            simdgroup_load(sga[1], ((const threadgroup FLOAT*)sdata_a) + 64, 8);
            
            simdgroup_load(sgb[0], ((const threadgroup FLOAT*)sdata_b) + 0,  64);
            simdgroup_load(sgb[1], ((const threadgroup FLOAT*)sdata_b) + 8,  64);
            simdgroup_load(sgb[2], ((const threadgroup FLOAT*)sdata_b) + 16, 64);
            simdgroup_load(sgb[3], ((const threadgroup FLOAT*)sdata_b) + 24, 64);
            
            SIMDGROUP_MATRIX_FMA(2, 4);
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            xy_in0 += 4 * cst.input_size * cst.batch;
        }
    }
        
    threadgroup FLOAT * sdata_c = (threadgroup FLOAT*)sdata + 512*sgitg;
    
    SIMDGROUP_MATRIX_STORE((threadgroup FLOAT*)sdata_c, 8);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // sdata [M2, N2, N2, N2, M2, M8, N8]
    // index [ml/16, ko, kl, N2, (ml/8)%2, ml%8, N2, N4]
    if((rx * 32 + ml) < cst.input_size * cst.batch) {
        if((4 * uz + 2 * ko + kl) * 4 < cst.output_slice) {
            xy_out[0] =  activate(ftype4(sdata[(((ml/16 * 4 + 2 * ko + kl) * 2 + 0) * 16 + ml % 16) * 2] + FLOAT4(biasTerms[(4 * uz + 2 * ko + kl) * 4])), cst.activation);
        }
        if((4 * uz + 2 * ko + kl) * 4 + 1 < cst.output_slice) {
            xy_out[cst.output_size * cst.batch] = activate(ftype4(sdata[(((ml/16 * 4 + 2 * ko + kl) * 2 + 0) * 16 + ml % 16) * 2 + 1] + FLOAT4(biasTerms[(4 * uz + 2 * ko + kl) * 4 + 1])), cst.activation);
        }
        if((4 * uz + 2 * ko + kl) * 4 + 2 < cst.output_slice) {
            xy_out[cst.output_size * cst.batch * 2] = activate(ftype4(sdata[(((ml/16 * 4 + 2 * ko + kl) * 2 + 1) * 16 + ml % 16) * 2] + FLOAT4(biasTerms[(4 * uz + 2 * ko + kl) * 4 + 2])), cst.activation);
        }
        if((4 * uz + 2 * ko + kl) * 4 + 3 < cst.output_slice) {
            xy_out[cst.output_size * cst.batch * 3] = activate(ftype4(sdata[(((ml/16 * 4 + 2 * ko + kl) * 2 + 1) * 16 + ml % 16) * 2 + 1] + FLOAT4(biasTerms[(4 * uz + 2 * ko + kl) * 4 + 3])), cst.activation);
        }
    }
}

kernel void conv1x1_g1z4_w4(const device ftype4 *in            [[buffer(0)]],
                            device ftype4 *out                 [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                            const device MNN::uchar4x2 *wt      [[buffer(3)]],
                            const device ftype4 *biasTerms     [[buffer(4)]],
                            const device ftype4 *dequantScale  [[buffer(5)]],
                            uint3 gid                          [[thread_position_in_grid]]) {
    if ((int)gid.x * CONV_UNROLL >= cst.output_size || (int)gid.y >= cst.output_slice || (int)gid.z >= cst.batch) return;

    int rx = gid.x * CONV_UNROLL;
    int uz = gid.y;
    auto xy_wt = wt + uz * cst.input_slice;
    auto xy_in0  = in  + (int)gid.z  * cst.input_size + rx + 0;
    auto xy_out = out + (int)gid.z * cst.output_size + uz * cst.output_size * cst.batch + rx;
    auto biasValue = FLOAT4(biasTerms[uz]);
    FLOAT4 result0 = biasValue, result1 = biasValue, result2 = biasValue, result3 = biasValue;
    int computeSize = min(cst.output_size - rx, CONV_UNROLL);
    int block = (cst.input_slice + cst.block_size - 1) / cst.block_size;
    for (int bi=0; bi<cst.block_size; ++bi) {
        FLOAT4 scale = FLOAT4(dequantScale[2 * (uz * cst.block_size + bi) + 0]) / (FLOAT)cst.scale_coef;
        FLOAT4 dequant_bias = FLOAT4(dequantScale[2 * (uz * cst.block_size + bi) + 1]) / (FLOAT)cst.scale_coef;
        int zmin = bi * block;
        int zmax = min(zmin + block, cst.input_slice);
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
            xy_in0 += cst.input_size * cst.batch;
        }
    }
    
    /* true */ 
    xy_out[0] = activate(ftype4(result0), cst.activation);
    if (computeSize > 1) {xy_out[1] = activate(ftype4(result1), cst.activation); }
    if (computeSize > 2) {xy_out[2] = activate(ftype4(result2), cst.activation); }
    if (computeSize > 3) {xy_out[3] = activate(ftype4(result3), cst.activation); }
}

kernel void conv1x1_gemv_g8_w4(const device ftype4 *in            [[buffer(0)]],
                            device ftype4 *out                 [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                            const device MNN::uchar4x2 *wt      [[buffer(3)]],
                            const device ftype4 *biasTerms     [[buffer(4)]],
                            const device ftype4 *dequantScale  [[buffer(5)]],
                            uint3 gid[[threadgroup_position_in_grid]],
                            uint  tiisg[[thread_index_in_simdgroup]],
                            uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    // each threadgroup contain 2 simdgroup
    // each simdgroup compute 4 data
    int uz = gid.x * 2 + sgitg;
    if(uz >= cst.output_slice) {
        return;
    }

    int rx = gid.y;
    auto xy_wt = wt + uz * cst.input_slice;
    auto xy_in0  = in  + (int)gid.z  * cst.input_size + rx + 0;
    auto xy_out = out + (int)gid.z * cst.output_size + uz * cst.output_size * cst.batch + rx;
    auto biasValue = FLOAT4(biasTerms[uz]);
    FLOAT4 result0 = FLOAT4(0);

    int block = (cst.input_slice + cst.block_size - 1) / cst.block_size;
    
    int middle_step = min(SIMD_GROUP_WIDTH, block);
    int outer_step  = SIMD_GROUP_WIDTH / middle_step;
    int middle_index = (tiisg) % middle_step;
    int outer_index  = (tiisg) / middle_step;
    
    for (int bi= outer_index; bi<cst.block_size; bi += outer_step) {
        FLOAT4 scale = FLOAT4(dequantScale[2 * (uz * cst.block_size + bi) + 0]) / (FLOAT)cst.scale_coef;
        FLOAT4 dequant_bias = FLOAT4(dequantScale[2 * (uz * cst.block_size + bi) + 1]) / (FLOAT)cst.scale_coef;
        int zmin = bi * block;
        int zmax = min(zmin + block, cst.input_slice);
        for (int z = zmin + middle_index; z < zmax; z += middle_step) {
            FLOAT4 in40 = (FLOAT4)*(xy_in0 + z);
            
            MNN::uchar4x2 w_int4 = xy_wt[z];

            FLOAT4x4 w_dequant;
            for (int i = 0; i < 4; i += 1) {
                FLOAT4 w4 = FLOAT4((float)(w_int4[i][0] >> 4) - 8, (float)(w_int4[i][0] & 15) - 8, (float)(w_int4[i][1] >> 4) - 8, (float)(w_int4[i][1] & 15) - 8);
                FLOAT4 res = w4 * scale[i] + dequant_bias[i];
                w_dequant[i] = res;
            }

            result0 += FLOAT4(in40 * w_dequant);
            
        }
    }

    FLOAT4 res = simd_sum(result0);
    /* true */
    if (tiisg == 0) {
        xy_out[0] = activate(ftype4(res + biasValue), cst.activation);
    }
}




kernel void conv1x1_gemv_g16_w4(const device ftype4 *in            [[buffer(0)]],
                            device ftype4 *out                 [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                            const device MNN::uchar4x2 *wt      [[buffer(3)]],
                            const device ftype4 *biasTerms     [[buffer(4)]],
                            const device ftype4 *dequantScale  [[buffer(5)]],
                            uint3 gid[[threadgroup_position_in_grid]],
                            uint  tiisg[[thread_index_in_simdgroup]],
                            uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    // each threadgroup contain 2 simdgroup
    // each simdgroup compute 8 data
    int uz = 2 * (gid.x * 2 + sgitg);
    if(uz >= cst.output_slice) {
        return;
    }
    
    auto xy_wt = wt + uz * cst.input_slice;
    auto xy_in0  = in;
    auto xy_out = out + (int)gid.z * cst.output_size + uz;
    auto biasValue0 = FLOAT4(biasTerms[uz]);
    auto biasValue1 = FLOAT4(biasTerms[uz + 1]);

    FLOAT4 result0 = FLOAT4(0);
    FLOAT4 result1 = FLOAT4(0);

    int block = (cst.input_slice + cst.block_size - 1) / cst.block_size;
    
    int middle_step = min(SIMD_GROUP_WIDTH, block);
    int outer_step  = SIMD_GROUP_WIDTH / middle_step;
    int middle_index = (tiisg) % middle_step;
    int outer_index  = (tiisg) / middle_step;
    
    for (int bi= outer_index; bi<cst.block_size; bi += outer_step) {
        const int quant_offset = 2 * (uz * cst.block_size + bi);
        FLOAT4 scale0 = FLOAT4(dequantScale[quant_offset + 0]) / (FLOAT)cst.scale_coef;
        FLOAT4 dequant_bias0 = FLOAT4(dequantScale[quant_offset + 1]) / (FLOAT)cst.scale_coef;
        FLOAT4 scale1 = FLOAT4(dequantScale[quant_offset + (cst.block_size << 1)]) / (FLOAT)cst.scale_coef;
        FLOAT4 dequant_bias1 = FLOAT4(dequantScale[quant_offset + (cst.block_size << 1) + 1]) / (FLOAT)cst.scale_coef;
        int zmin = bi * block;
        int zmax = min(zmin + block, cst.input_slice);
        for (int z = zmin + middle_index; z < zmax; z += middle_step) {
            FLOAT4 in40 = (FLOAT4)*(xy_in0 + z);
            
            MNN::uchar4x2 w_int4 = xy_wt[z];

            FLOAT4x4 w_dequant;
            for (int i = 0; i < 4; i += 1) {
                FLOAT4 w4 = FLOAT4((float)(w_int4[i][0] >> 4) - 8, (float)(w_int4[i][0] & 15) - 8, (float)(w_int4[i][1] >> 4) - 8, (float)(w_int4[i][1] & 15) - 8);
                FLOAT4 res = w4 * scale0[i] + dequant_bias0[i];
                w_dequant[i] = res;
            }
            result0 += FLOAT4(in40 * w_dequant);

            w_int4 = xy_wt[cst.input_slice + z];
            for (int i = 0; i < 4; i += 1) {
                FLOAT4 w4 = FLOAT4((float)(w_int4[i][0] >> 4) - 8, (float)(w_int4[i][0] & 15) - 8, (float)(w_int4[i][1] >> 4) - 8, (float)(w_int4[i][1] & 15) - 8);
                FLOAT4 res = w4 * scale1[i] + dequant_bias1[i];
                w_dequant[i] = res;
            }
            
            result1 += FLOAT4(in40 * w_dequant);
            
        }
    }

    FLOAT4 res0 = simd_sum(result0);
    FLOAT4 res1 = simd_sum(result1);

    /* true */
    if (tiisg == 0) {
        xy_out[0] = activate(ftype4(res0 + biasValue0), cst.activation);
        xy_out[1] = activate(ftype4(res1 + biasValue1), cst.activation);

    }
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
    auto xy_in0  = in  + (int)gid.z  * cst.input_size + rx + 0;

    auto xy_out = out + (int)gid.z * cst.output_size + uz * cst.batch * cst.output_size + rx;
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
            xy_in0 += cst.input_size * cst.batch;
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
    if ((int)gid.x * 16 >= cst.output_width || (int)gid.y >= cst.batch * cst.output_slice) return;

    int idx_w = gid.x << 4;
    int idx_h = 0;
    int idx_c = gid.y / cst.batch;
    int idx_b = gid.y % cst.batch;

    auto xy_wt = wt + idx_c * cst.input_slice;
    auto xy_in0  = in  + (int)idx_b * cst.input_size + idx_h * cst.output_width + idx_w;

    auto xy_out = out + (int)idx_b * cst.output_size + idx_c * cst.output_size * cst.batch + idx_h * cst.output_width + idx_w;
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
        
        xy_in0 += cst.input_size * cst.batch;
    }

    int widthSize = min(cst.output_width - idx_w, 16);
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
    if ((int)gid.x * 2 >= cst.output_width || (int)gid.y * 2 >= cst.batch * cst.output_slice) return;

    int channel_pack = (cst.output_channel + 7) >> 3;
    int idx_w = gid.x << 1;
    int idx_h = 0;
    int idx_c = (gid.y % channel_pack) << 1;
    int idx_b = gid.y / channel_pack;
    
    if(idx_b >=  cst.batch || idx_c >= cst.output_slice) return;
    auto xy_wt = wt + idx_c * cst.input_slice;
    auto xy_in0  = in  + (int)idx_b * cst.input_size + idx_h * cst.output_width + idx_w;

    auto xy_out = out + (int)idx_b * cst.output_size + idx_c * cst.output_size * cst.batch + idx_h * cst.output_width + idx_w;
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
        xy_in0 += cst.input_size * cst.batch;
    }

    int widthSize = min(cst.output_width - idx_w, 2);
    /* true            */ *xy_out = activate(ftype4(result0), cst.activation);
    if (widthSize > 1) {xy_out[1] = activate(ftype4(result1), cst.activation); }
    
    int channelSize = min(cst.output_slice - idx_c, 2);
    if(channelSize > 1) {
        /* true         */ {xy_out[cst.output_size * cst.batch +0] = activate(ftype4(result4), cst.activation); }
        if (widthSize > 1) {xy_out[cst.output_size * cst.batch +1] = activate(ftype4(result5), cst.activation); }
    }
}

kernel void conv1x1_w4c2(const device ftype4 *in            [[buffer(0)]],
                         device ftype4 *out                 [[buffer(1)]],
                         constant conv1x1_constants& cst    [[buffer(2)]],
                         const device ftype4x4 *wt          [[buffer(3)]],
                         const device ftype4 *biasTerms     [[buffer(4)]],
                         uint3 gid                          [[thread_position_in_grid]]) {
    if ((int)gid.x * 4 >= cst.output_width || (int)gid.y * 2 >= cst.batch * cst.output_slice) return;

    int channel_pack = (cst.output_channel + 7) >> 3;
    int idx_w = gid.x << 2;
    int idx_h = 0;
    int idx_c = (gid.y % channel_pack) << 1;
    int idx_b = gid.y / channel_pack;

    if(idx_b >=  cst.batch || idx_c >= cst.output_slice) return;
    auto xy_wt = wt + idx_c * cst.input_slice;
    auto xy_in0  = in  + (int)idx_b * cst.input_size + idx_h * cst.output_width + idx_w;

    auto xy_out = out + (int)idx_b * cst.output_size + idx_c * cst.output_size * cst.batch + idx_h * cst.output_width + idx_w;
    auto biasValue0 = FLOAT4(biasTerms[idx_c]);
    auto biasValue1 = FLOAT4(biasTerms[idx_c+1]);

    FLOAT4 result0 = biasValue0, result1 = biasValue0;
    FLOAT4 result4 = biasValue0, result5 = biasValue0;
    FLOAT4 result2 = biasValue1, result3 = biasValue1;
    FLOAT4 result6 = biasValue1, result7 = biasValue1;
    for (auto z = 0; z < cst.input_slice; z++) {
        auto in40 = xy_in0[0];
        auto in41 = xy_in0[1];
        auto in44 = xy_in0[2];
        auto in45 = xy_in0[3];

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
        xy_in0 += cst.input_size * cst.batch;
    }

    int widthSize = min(cst.output_width - idx_w, 4);
    /* true            */ *xy_out = activate(ftype4(result0), cst.activation);
    if (widthSize > 1) {xy_out[1] = activate(ftype4(result1), cst.activation); }
    if (widthSize > 2) {xy_out[2] = activate(ftype4(result4), cst.activation); }
    if (widthSize > 3) {xy_out[3] = activate(ftype4(result5), cst.activation); }
        
    int channelSize = min(cst.output_slice - idx_c, 2);
    if(channelSize > 1) {
        /* true         */  xy_out[cst.output_size * cst.batch]   = activate(ftype4(result2), cst.activation);
        if (widthSize > 1) {xy_out[cst.output_size * cst.batch +1] = activate(ftype4(result3), cst.activation); }
        if (widthSize > 2) {xy_out[cst.output_size * cst.batch +2] = activate(ftype4(result6), cst.activation); }
        if (widthSize > 3) {xy_out[cst.output_size * cst.batch +3] = activate(ftype4(result7), cst.activation); }
    }
}

kernel void conv1x1_gemm_16x16(const device ftype4 *in            [[buffer(0)]],
                            device ftype4 *out                 [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                            const device ftype4 *wt      [[buffer(3)]],
                            const device ftype4 *biasTerms     [[buffer(4)]],
                            uint3 gid                          [[threadgroup_position_in_grid]],
                            uint                  tiitg[[thread_index_in_threadgroup]],
                            uint                  sgitg[[simdgroup_index_in_threadgroup]]) {
    /*
     // Read:
     ftype 0~127   ---> input: [M16, K8]
     ftype 128~255 ---> input: [K8, N16]
     // Write:
     ftype 0~255 ---> input: [N2, M2, M8, N8]
     */
    threadgroup FLOAT4 sdata[64] = {0.f};
    
    INIT_SIMDGROUP_MATRIX(2, 2, 4);
    int rx = gid.x;// M/16
    int uz = gid.y;// N/16
    
    int kl = tiitg / 16;
    int rcl = tiitg % 16;
    
    // boundary limit
    int idx_n4 = (4 * uz + rcl / 4) < cst.output_slice ? (4 * uz + rcl / 4) : (cst.output_slice - 1);
    int idx_m  = (16 * rx + rcl) < cst.input_size * cst.batch ? (16 * rx + rcl) : (cst.input_size * cst.batch - 1);
    
    auto xy_wt = wt +  (idx_n4 * cst.input_slice + 0) * 4 + rcl % 4;// [N/4, K/4, N4, K4]
    auto xy_in0  = in + idx_m + cst.input_size * cst.batch * kl;// [K/4, M, K4]
    auto xy_out = out + (4 * uz + 2 * kl) * cst.output_size * cst.batch + idx_m;// [N/4, M, N4]

    for (int z = kl; z < cst.input_slice; z += 2) {
        sdata[2* rcl + kl] = FLOAT4(*xy_in0);
        xy_in0 += 2 * cst.input_size * cst.batch;

        FLOAT4 w4 = FLOAT4(xy_wt[4 * z]); // [N/4, K/4, N4, K4]
        ((threadgroup FLOAT*)sdata)[128 + (kl * 4 + 0) * 16 + rcl] = w4[0];
        ((threadgroup FLOAT*)sdata)[128 + (kl * 4 + 1) * 16 + rcl] = w4[1];
        ((threadgroup FLOAT*)sdata)[128 + (kl * 4 + 2) * 16 + rcl] = w4[2];
        ((threadgroup FLOAT*)sdata)[128 + (kl * 4 + 3) * 16 + rcl] = w4[3];

        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        simdgroup_load(sga[0], (const threadgroup FLOAT*)sdata, 8);
        simdgroup_load(sga[1], ((const threadgroup FLOAT*)sdata) + 64, 8);
        simdgroup_load(sgb[0], ((const threadgroup FLOAT*)sdata) + 128, 16);
        simdgroup_load(sgb[1], ((const threadgroup FLOAT*)sdata) + 136, 16);
        
        SIMDGROUP_MATRIX_FMA(2, 2);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    SIMDGROUP_MATRIX_STORE((threadgroup FLOAT*)sdata, 4);
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if((16 * rx + rcl) < cst.input_size * cst.batch) {
        if((4 * uz + 2 * kl) < cst.output_slice) {
            xy_out[0] =  activate(ftype4(sdata[(kl * 16 + rcl) * 2 + 0] + FLOAT4(biasTerms[4 * uz + 2 * kl + 0])), cst.activation);
        }
        if((4 * uz + 2 * kl + 1) < cst.output_slice) {
            xy_out[cst.output_size * cst.batch] = activate(ftype4(sdata[(kl * 16 + rcl) * 2 + 1] + FLOAT4(biasTerms[4 * uz + 2 * kl + 1])), cst.activation);
        }
    }
}


kernel void conv1x1_gemm_32x16(const device ftype4 *in            [[buffer(0)]],
                            device ftype4 *out                 [[buffer(1)]],
                            constant conv1x1_constants& cst    [[buffer(2)]],
                            const device ftype4 *wt      [[buffer(3)]],
                            const device ftype4 *biasTerms     [[buffer(4)]],
                            uint3 gid                          [[threadgroup_position_in_grid]],
                            uint                  tiitg[[thread_index_in_threadgroup]],
                            uint                  sgitg[[simdgroup_index_in_threadgroup]]) {
    /*
     // Read:
     ftype 0~255   ---> input: [M32, K8]
     ftype 256~383 ---> input: [K8, N16]
     // Write:
     ftype 0~511 ---> input: [N2, M4, M8, N8]
     */
    threadgroup FLOAT4 sdata[128] = {0.f};
    
    INIT_SIMDGROUP_MATRIX(4, 2, 8);
    
    int rx = gid.x;// M/32
    int uz = gid.y;// N/16
    
    int kl = tiitg % 2;
    int rcl = tiitg / 2;
    
    const int size_m = cst.input_size * cst.batch;
    
    // boundary limit
    int idx_n4 = (4 * uz + rcl / 4) < cst.output_slice ? (4 * uz + rcl / 4) : (cst.output_slice - 1);
    int idx_m0  = (16 * rx + rcl) <  size_m ? (16 * rx + rcl) : (size_m - 1);
    int idx_m1  = (16 * rx + rcl) + size_m / 2 < size_m ? (16 * rx + rcl) + size_m / 2: (size_m - 1);

    auto xy_wt = wt +  (idx_n4 * cst.input_slice + 0) * 4 + rcl % 4;// [N/4, K/4, N4, K4]
    auto xy_in0  = in + idx_m0 + cst.input_size * cst.batch * kl;// [K/4, M2, M/2, K4]
    auto xy_in1  = in + idx_m1 + cst.input_size * cst.batch * kl;// [K/4, M2, M/2, K4]

    auto xy_out0 = out + (4 * uz + 2 * kl) * cst.output_size * cst.batch + idx_m0;// [N/4, M, N4]
    auto xy_out1 = out + (4 * uz + 2 * kl) * cst.output_size * cst.batch + idx_m1;// [N/4, M, N4]
    
    for (int z = kl; z < cst.input_slice; z += 2) {
        sdata[2* rcl + kl] = (FLOAT4)*xy_in0;
        sdata[32 + 2* rcl + kl] = (FLOAT4)*xy_in1;

        FLOAT4 w4 = FLOAT4(xy_wt[4*z]); // [N/4, K/4, N4, K4]
        ((threadgroup FLOAT*)sdata)[256 + (kl * 4 + 0) * 16 + rcl] = w4[0];
        ((threadgroup FLOAT*)sdata)[256 + (kl * 4 + 1) * 16 + rcl] = w4[1];
        ((threadgroup FLOAT*)sdata)[256 + (kl * 4 + 2) * 16 + rcl] = w4[2];
        ((threadgroup FLOAT*)sdata)[256 + (kl * 4 + 3) * 16 + rcl] = w4[3];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        simdgroup_load(sga[0], (const threadgroup FLOAT*)sdata, 8);
        simdgroup_load(sga[1], ((const threadgroup FLOAT*)sdata) + 64, 8);
        simdgroup_load(sga[2], ((const threadgroup FLOAT*)sdata) + 128, 8);
        simdgroup_load(sga[3], ((const threadgroup FLOAT*)sdata) + 192, 8);
        
        simdgroup_load(sgb[0], ((const threadgroup FLOAT*)sdata) + 256, 16);
        simdgroup_load(sgb[1], ((const threadgroup FLOAT*)sdata) + 264, 16);
        
        SIMDGROUP_MATRIX_FMA(4, 2);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        xy_in0 += 2 * cst.input_size * cst.batch;
        xy_in1 += 2 * cst.input_size * cst.batch;

    }

    SIMDGROUP_MATRIX_STORE((threadgroup FLOAT*)sdata, 8);
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if((16 * rx + rcl) < size_m) {
        if((4 * uz + 2 * kl) < cst.output_slice) {
            xy_out0[0] =  activate(ftype4(sdata[(kl * 32 + rcl) * 2 + 0] + FLOAT4(biasTerms[4 * uz + 2 * kl + 0])), cst.activation);
        }
        if((4 * uz + 2 * kl + 1) < cst.output_slice) {
            xy_out0[cst.output_size * cst.batch] = activate(ftype4(sdata[(kl * 32 + rcl) * 2 + 1] + FLOAT4(biasTerms[4 * uz + 2 * kl + 1])), cst.activation);
        }
    }
    if((16 * rx + rcl) + size_m / 2 < size_m) {
        if((4 * uz + 2 * kl) < cst.output_slice) {
            xy_out1[0] =  activate(ftype4(sdata[(kl * 32 + 16 + rcl) * 2 + 0] + FLOAT4(biasTerms[4 * uz + 2 * kl + 0])), cst.activation);
        }
        if((4 * uz + 2 * kl + 1) < cst.output_slice) {
            xy_out1[cst.output_size * cst.batch] = activate(ftype4(sdata[(kl * 32 + 16 + rcl) * 2 + 1] + FLOAT4(biasTerms[4 * uz + 2 * kl + 1])), cst.activation);
        }
    }
}
