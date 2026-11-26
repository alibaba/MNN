//
//  GemmInt8W4.inl
//  MNN
//
//  Copyright © 2018, Alibaba Group Holding Limited
//
// W4A8 int8 GEMM shared by the AVX2 and AVX-VNNI builds. The includer defines MNN_GEMMINT8_W4_NAME and
// MNN_GEMMINT8_W4_DOT(acc, s, w), which adds the u8 x s8 dot product of each 4-byte group to the int32 lanes.

#ifndef LOAD_INT4_TO_INT8
#define LOAD_INT4_TO_INT8 \
auto w_int4 = _mm_loadu_si128((__m128i const*)weight_sz);\
auto w_0 = _mm_and_si128(mask, _mm_srli_epi16(w_int4, 4));\
auto w_1 = _mm_and_si128(mask, w_int4);
#endif

void MNN_GEMMINT8_W4_NAME(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realDst) {
    MNN_ASSERT(post->useInt8==0);
    const auto dst_step_tmp = dst_step / sizeof(int8_t);
    auto zero128 = _mm256_set1_ps(0.0f);
    auto minValue = _mm256_set1_ps(post->minValue);
    auto maxValue = _mm256_set1_ps(post->maxValue);
    auto offset = _mm256_set1_epi32(128);
    __m256 fp32min, fp32max;
    if (post->fp32minmax) {
        fp32min = _mm256_set1_ps((post->fp32minmax)[0]);
        fp32max = _mm256_set1_ps((post->fp32minmax)[1]);
    }
    const float* biasPtr = nullptr;
    int inputBlockNum = 1;
    if (post->biasFloat) {
        biasPtr = post->biasFloat;
    }
    auto accumbuff = post->accumBuffer;
    auto blockNum = post->blockNum;
    if (post->inputBias) {
        inputBlockNum = blockNum;
    }

    int weight_step_Y = (GEMMINT8_AVX2_L * GEMMINT8_AVX2_H) / 2;
    int weight_step_Z = src_depth_quad * weight_step_Y + 2 * sizeof(float)* GEMMINT8_AVX2_H;
    const __m128i mask = _mm_set1_epi8(0xf);

    auto srcKernelSumPtr = post->srcKernelSum;
    __m256 kernelSum0, kernelSum1, kernelSum2, kernelSum3;
    auto neg128_f   = _mm256_set1_ps(-128.f);
    __m256 extrascale0 = _mm256_setzero_ps();
    __m256 extrascale1 = _mm256_setzero_ps();
    __m256 extrascale2 = _mm256_setzero_ps();
    __m256 extrascale3 = _mm256_setzero_ps();
    __m256 extrabias0 = _mm256_setzero_ps();
    __m256 extrabias1 = _mm256_setzero_ps();
    __m256 extrabias2 = _mm256_setzero_ps();
    __m256 extrabias3 = _mm256_setzero_ps();
    if (post->inputScale) {
        if (GEMMINT8_AVX2_E == realDst) {
            extrascale0 = _mm256_set1_ps(post->inputScale[0]);
            extrascale1 = _mm256_set1_ps(post->inputScale[1]);
            extrascale2 = _mm256_set1_ps(post->inputScale[2]);
            extrascale3 = _mm256_set1_ps(post->inputScale[3]);
        } else {
            extrascale0 = _mm256_set1_ps(post->inputScale[0]);
            if (realDst > 1) {
                extrascale1 = _mm256_set1_ps(post->inputScale[1]);
            }
            if (realDst > 2) {
                extrascale2 = _mm256_set1_ps(post->inputScale[2]);
            }
        }
    }
    auto oneValue = _mm256_set1_epi16(1);
    __m256 bias0, bias1, bias2, bias3;
    // weight&scale&bias: [oc/hp, blocknum, weight_step_Z]
    // weight_step_Z: [(kx*ky), ic/lp/blocknum, hp, lp] + [hp] + [hp]
    // input: [blocknum, blockLu, EP, LP]
    if (GEMMINT8_AVX2_E == realDst) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            auto dst_x           = dst + dz * dst_step_tmp;
            auto accum_x       = accumbuff;

            for (int bk = 0; bk < blockNum; ++bk) {
                // block's weight&scale&bias
                const auto weight_dz = weight + dz * blockNum * weight_step_Z + bk *  weight_step_Z;
                const float* scale_dz = reinterpret_cast<const float*>(weight_dz + src_depth_quad * weight_step_Y);
                const auto weightBias_dz = scale_dz + GEMMINT8_AVX2_H;
                // block's input
                const auto src_x = src + bk * src_depth_quad * GEMMINT8_AVX2_L * realDst;
                kernelSum0 = _mm256_set1_ps((post->srcKernelSum + bk * realDst)[0]);
                kernelSum1 = _mm256_set1_ps((post->srcKernelSum + bk * realDst)[1]);
                kernelSum2 = _mm256_set1_ps((post->srcKernelSum + bk * realDst)[2]);
                kernelSum3 = _mm256_set1_ps((post->srcKernelSum + bk * realDst)[3]);

                __m256i D00 = _mm256_set1_epi32(0);
                __m256i D01 = _mm256_set1_epi32(0);
                __m256i D02 = _mm256_set1_epi32(0);
                __m256i D03 = _mm256_set1_epi32(0);
                for (int sz = 0; sz < src_depth_quad; ++sz) {
                    const auto weight_sz = weight_dz + sz * weight_step_Y;
                    const auto src_z     = src_x + sz * GEMMINT8_AVX2_L * realDst;
                    LOAD_INT4_TO_INT8;
                    auto w0 = _mm256_insertf128_si256(_mm256_castsi128_si256(w_0), w_1, 1);
                    auto s0 = _mm256_castps_si256(_mm256_broadcast_ss((float*)src_z + 0));
                    auto s1 = _mm256_castps_si256(_mm256_broadcast_ss((float*)src_z + 1));
                    auto s2 = _mm256_castps_si256(_mm256_broadcast_ss((float*)src_z + 2));
                    auto s3 = _mm256_castps_si256(_mm256_broadcast_ss((float*)src_z + 3));

                    D00 = MNN_GEMMINT8_W4_DOT(D00, s0, w0);
                    D01 = MNN_GEMMINT8_W4_DOT(D01, s1, w0);
                    D02 = MNN_GEMMINT8_W4_DOT(D02, s2, w0);
                    D03 = MNN_GEMMINT8_W4_DOT(D03, s3, w0);
                }
                auto D0 = D00;
                auto D1 = D01;
                auto D2 = D02;
                auto D3 = D03;
                auto scaleValue = _mm256_loadu_ps(scale_dz);
                auto weightBiasValue = _mm256_loadu_ps((float*)weightBias_dz);

                auto f0 = _mm256_cvtepi32_ps(D0);
                auto f1 = _mm256_cvtepi32_ps(D1);
                auto f2 = _mm256_cvtepi32_ps(D2);
                auto f3 = _mm256_cvtepi32_ps(D3);
                // x_kernelSum x w_quantZero
                auto xy0_0 = _mm256_mul_ps(kernelSum0, weightBiasValue); // x dimemsion first
                auto xy0_1 = _mm256_mul_ps(kernelSum1, weightBiasValue); // ..second
                auto xy0_2 = _mm256_mul_ps(kernelSum2, weightBiasValue); // .. third
                auto xy0_3 = _mm256_mul_ps(kernelSum3, weightBiasValue); // ..fourth
                f0 = _mm256_mul_ps(f0, scaleValue);
                f1 = _mm256_mul_ps(f1, scaleValue);
                f2 = _mm256_mul_ps(f2, scaleValue);
                f3 = _mm256_mul_ps(f3, scaleValue);
                if (post->inputScale) {
                    if (post->inputBias) {
                        extrascale0 = _mm256_set1_ps((post->inputScale + bk * realDst)[0]);
                        extrascale1 = _mm256_set1_ps((post->inputScale + bk * realDst)[1]);
                        extrascale2 = _mm256_set1_ps((post->inputScale + bk * realDst)[2]);
                        extrascale3 = _mm256_set1_ps((post->inputScale + bk * realDst)[3]);
                    }
                    f0 = _mm256_mul_ps(f0, extrascale0);
                    f1 = _mm256_mul_ps(f1, extrascale1);
                    f2 = _mm256_mul_ps(f2, extrascale2);
                    f3 = _mm256_mul_ps(f3, extrascale3);
                    if ((post->useInt8 == 0) && post->weightKernelSum && (post->inputBias || (bk == blockNum - 1))) {
                        if (post->inputBias) {
                            auto wsumDz = post->weightKernelSum + dz * (blockNum * GEMMINT8_AVX2_H) + bk * GEMMINT8_AVX2_H;
                            auto wsum = _mm256_loadu_ps(wsumDz);
                            extrabias0 = _mm256_set1_ps((post->inputBias + bk * realDst)[0]);
                            extrabias1 = _mm256_set1_ps((post->inputBias + bk * realDst)[1]);
                            extrabias2 = _mm256_set1_ps((post->inputBias + bk * realDst)[2]);
                            extrabias3 = _mm256_set1_ps((post->inputBias + bk * realDst)[3]);
                            bias0 = _mm256_mul_ps(extrabias0, wsum);
                            bias1 = _mm256_mul_ps(extrabias1, wsum);
                            bias2 = _mm256_mul_ps(extrabias2, wsum);
                            bias3 = _mm256_mul_ps(extrabias3, wsum);
                        } else if (bk == blockNum - 1) { // if input not block quant, only accum once!
                            auto wsumDz = post->weightKernelSum + dz * GEMMINT8_AVX2_H;
                            auto wsum = _mm256_loadu_ps(wsumDz);
                            bias0 = _mm256_mul_ps(_mm256_mul_ps(extrascale0, neg128_f), wsum);
                            bias1 = _mm256_mul_ps(_mm256_mul_ps(extrascale1, neg128_f), wsum);
                            bias2 = _mm256_mul_ps(_mm256_mul_ps(extrascale2, neg128_f), wsum);
                            bias3 = _mm256_mul_ps(_mm256_mul_ps(extrascale3, neg128_f), wsum);
                        }
                        f0 = _mm256_add_ps(f0, bias0);
                        f1 = _mm256_add_ps(f1, bias1);
                        f2 = _mm256_add_ps(f2, bias2);
                        f3 = _mm256_add_ps(f3, bias3);
                    }
                }
                f0 = _mm256_add_ps(f0, xy0_0);
                f1 = _mm256_add_ps(f1, xy0_1);
                f2 = _mm256_add_ps(f2, xy0_2);
                f3 = _mm256_add_ps(f3, xy0_3);

                if (bk > 0) {
                    auto dstv0 = _mm256_loadu_ps(((float*)accum_x) + 0 * AVX2_PACKINT8);
                    auto dstv1 = _mm256_loadu_ps(((float*)accum_x) + 1 * AVX2_PACKINT8);
                    auto dstv2 = _mm256_loadu_ps(((float*)accum_x) + 2 * AVX2_PACKINT8);
                    auto dstv3 = _mm256_loadu_ps(((float*)accum_x) + 3 * AVX2_PACKINT8);
                    f0 = _mm256_add_ps(f0, dstv0);
                    f1 = _mm256_add_ps(f1, dstv1);
                    f2 = _mm256_add_ps(f2, dstv2);
                    f3 = _mm256_add_ps(f3, dstv3);
                }
                if (bk == blockNum - 1) {
                    if (biasPtr) {
                        const auto bias_dz   = biasPtr + dz * AVX2_PACKINT8;
                        auto biasValue       = _mm256_loadu_ps(bias_dz);
                        f0 = _mm256_add_ps(f0, biasValue);
                        f1 = _mm256_add_ps(f1, biasValue);
                        f2 = _mm256_add_ps(f2, biasValue);
                        f3 = _mm256_add_ps(f3, biasValue);
                    }
                    if (post->fp32minmax) {
                        f0 = _mm256_min_ps(f0, fp32max);
                        f1 = _mm256_min_ps(f1, fp32max);
                        f2 = _mm256_min_ps(f2, fp32max);
                        f3 = _mm256_min_ps(f3, fp32max);
                        f0 = _mm256_max_ps(f0, fp32min);
                        f1 = _mm256_max_ps(f1, fp32min);
                        f2 = _mm256_max_ps(f2, fp32min);
                        f3 = _mm256_max_ps(f3, fp32min);
                    }
                    _mm256_storeu_ps(((float*)dst_x) + 0 * AVX2_PACKINT8, f0);
                    _mm256_storeu_ps(((float*)dst_x) + 1 * AVX2_PACKINT8, f1);
                    _mm256_storeu_ps(((float*)dst_x) + 2 * AVX2_PACKINT8, f2);
                    _mm256_storeu_ps(((float*)dst_x) + 3 * AVX2_PACKINT8, f3);
                } else {
                    _mm256_storeu_ps(((float*)accum_x) + 0 * AVX2_PACKINT8, f0);
                    _mm256_storeu_ps(((float*)accum_x) + 1 * AVX2_PACKINT8, f1);
                    _mm256_storeu_ps(((float*)accum_x) + 2 * AVX2_PACKINT8, f2);
                    _mm256_storeu_ps(((float*)accum_x) + 3 * AVX2_PACKINT8, f3);
                }
            }
        }
        return;
    }
    if (3 == realDst) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            auto dst_x           = dst + dz * dst_step_tmp;
            auto accum_x       = accumbuff;

            for (int bk = 0; bk < blockNum; ++bk) {
                // block's weight&scale&bias
                const auto weight_dz = weight + dz * blockNum * weight_step_Z + bk *  weight_step_Z;
                const float* scale_dz = reinterpret_cast<const float*>(weight_dz + src_depth_quad * weight_step_Y);
                const auto weightBias_dz = scale_dz + GEMMINT8_AVX2_H;
                // block's input
                const auto src_x = src + bk * src_depth_quad * GEMMINT8_AVX2_L * realDst;
                kernelSum0 = _mm256_set1_ps((post->srcKernelSum + bk * realDst)[0]);
                kernelSum1 = _mm256_set1_ps((post->srcKernelSum + bk * realDst)[1]);
                kernelSum2 = _mm256_set1_ps((post->srcKernelSum + bk * realDst)[2]);

                __m256i D00 = _mm256_set1_epi32(0);
                __m256i D01 = _mm256_set1_epi32(0);
                __m256i D02 = _mm256_set1_epi32(0);
                for (int sz = 0; sz < src_depth_quad; ++sz) {
                    const auto weight_sz = weight_dz + sz * weight_step_Y;
                    const auto src_z     = src_x + sz * GEMMINT8_AVX2_L * realDst;
                    LOAD_INT4_TO_INT8;

                    auto w0 = _mm256_insertf128_si256(_mm256_castsi128_si256(w_0), w_1, 1);
                    auto s0 = _mm256_castps_si256(_mm256_broadcast_ss((float*)src_z + 0));
                    auto s1 = _mm256_castps_si256(_mm256_broadcast_ss((float*)src_z + 1));
                    auto s2 = _mm256_castps_si256(_mm256_broadcast_ss((float*)src_z + 2));

                    D00 = MNN_GEMMINT8_W4_DOT(D00, s0, w0);
                    D01 = MNN_GEMMINT8_W4_DOT(D01, s1, w0);
                    D02 = MNN_GEMMINT8_W4_DOT(D02, s2, w0);
                }
                auto D0 = D00;
                auto D1 = D01;
                auto D2 = D02;
                auto scaleValue = _mm256_loadu_ps(scale_dz);
                auto weightBiasValue = _mm256_loadu_ps((float*)weightBias_dz);

                auto f0 = _mm256_cvtepi32_ps(D0);
                auto f1 = _mm256_cvtepi32_ps(D1);
                auto f2 = _mm256_cvtepi32_ps(D2);
                // x_kernelSum x w_quantZero
                auto xy0_0 = _mm256_mul_ps(kernelSum0, weightBiasValue); // x dimemsion first
                auto xy0_1 = _mm256_mul_ps(kernelSum1, weightBiasValue); // ..second
                auto xy0_2 = _mm256_mul_ps(kernelSum2, weightBiasValue); // .. third
                f0 = _mm256_mul_ps(f0, scaleValue);
                f1 = _mm256_mul_ps(f1, scaleValue);
                f2 = _mm256_mul_ps(f2, scaleValue);
                if (post->inputScale) {
                    if (post->inputBias) {
                        extrascale0 = _mm256_set1_ps((post->inputScale + bk * realDst)[0]);
                        extrascale1 = _mm256_set1_ps((post->inputScale + bk * realDst)[1]);
                        extrascale2 = _mm256_set1_ps((post->inputScale + bk * realDst)[2]);
                    }
                    f0 = _mm256_mul_ps(f0, extrascale0);
                    f1 = _mm256_mul_ps(f1, extrascale1);
                    f2 = _mm256_mul_ps(f2, extrascale2);
                    if ((post->useInt8 == 0) && post->weightKernelSum && (post->inputBias || (bk == blockNum - 1))) {
                        if (post->inputBias) {
                            auto wsumDz = post->weightKernelSum + dz * (blockNum * GEMMINT8_AVX2_H) + bk * GEMMINT8_AVX2_H;
                            auto wsum = _mm256_loadu_ps(wsumDz);
                            extrabias0 = _mm256_set1_ps((post->inputBias + bk * realDst)[0]);
                            extrabias1 = _mm256_set1_ps((post->inputBias + bk * realDst)[1]);
                            extrabias2 = _mm256_set1_ps((post->inputBias + bk * realDst)[2]);
                            bias0 = _mm256_mul_ps(extrabias0, wsum);
                            bias1 = _mm256_mul_ps(extrabias1, wsum);
                            bias2 = _mm256_mul_ps(extrabias2, wsum);
                        } else if (bk == blockNum - 1) { // if input not block quant, only accum once!
                            auto wsumDz = post->weightKernelSum + dz *GEMMINT8_AVX2_H;
                            auto wsum = _mm256_loadu_ps(wsumDz);
                            bias0 = _mm256_mul_ps(_mm256_mul_ps(extrascale0, neg128_f), wsum);
                            bias1 = _mm256_mul_ps(_mm256_mul_ps(extrascale1, neg128_f), wsum);
                            bias2 = _mm256_mul_ps(_mm256_mul_ps(extrascale2, neg128_f), wsum);
                        }
                        f0 = _mm256_add_ps(f0, bias0);
                        f1 = _mm256_add_ps(f1, bias1);
                        f2 = _mm256_add_ps(f2, bias2);
                    }
                }
                f0 = _mm256_add_ps(f0, xy0_0);
                f1 = _mm256_add_ps(f1, xy0_1);
                f2 = _mm256_add_ps(f2, xy0_2);

                if (bk > 0) {
                    auto dstv0 = _mm256_loadu_ps(((float*)accum_x) + 0 * AVX2_PACKINT8);
                    auto dstv1 = _mm256_loadu_ps(((float*)accum_x) + 1 * AVX2_PACKINT8);
                    auto dstv2 = _mm256_loadu_ps(((float*)accum_x) + 2 * AVX2_PACKINT8);
                    f0 = _mm256_add_ps(f0, dstv0);
                    f1 = _mm256_add_ps(f1, dstv1);
                    f2 = _mm256_add_ps(f2, dstv2);
                }
                if (bk == blockNum - 1) {
                    if (nullptr != biasPtr) {
                        const auto bias_dz   = biasPtr + dz * AVX2_PACKINT8;
                        auto biasValue       = _mm256_loadu_ps(bias_dz);
                        f0 = _mm256_add_ps(f0, biasValue);
                        f1 = _mm256_add_ps(f1, biasValue);
                        f2 = _mm256_add_ps(f2, biasValue);
                    }
                    if (post->fp32minmax) {
                        f0 = _mm256_min_ps(f0, fp32max);
                        f1 = _mm256_min_ps(f1, fp32max);
                        f2 = _mm256_min_ps(f2, fp32max);
                        f0 = _mm256_max_ps(f0, fp32min);
                        f1 = _mm256_max_ps(f1, fp32min);
                        f2 = _mm256_max_ps(f2, fp32min);
                    }
                    _mm256_storeu_ps(((float*)dst_x) + 0 * AVX2_PACKINT8, f0);
                    _mm256_storeu_ps(((float*)dst_x) + 1 * AVX2_PACKINT8, f1);
                    _mm256_storeu_ps(((float*)dst_x) + 2 * AVX2_PACKINT8, f2);
                } else {
                    _mm256_storeu_ps(((float*)accum_x) + 0 * AVX2_PACKINT8, f0);
                    _mm256_storeu_ps(((float*)accum_x) + 1 * AVX2_PACKINT8, f1);
                    _mm256_storeu_ps(((float*)accum_x) + 2 * AVX2_PACKINT8, f2);
                }
            }
        }
        return;
    }
    if (2 == realDst) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            auto dst_x           = dst + dz * dst_step_tmp;
            auto accum_x       = accumbuff;

            for (int bk = 0; bk < blockNum; ++bk) {
                // block's weight&scale&bias
                const auto weight_dz = weight + dz * blockNum * weight_step_Z + bk *  weight_step_Z;
                const float* scale_dz = reinterpret_cast<const float*>(weight_dz + src_depth_quad * weight_step_Y);
                const auto weightBias_dz = scale_dz + GEMMINT8_AVX2_H;
                // block's input
                const auto src_x = src + bk * src_depth_quad * GEMMINT8_AVX2_L * realDst;
                kernelSum0 = _mm256_set1_ps((post->srcKernelSum + bk * realDst)[0]);
                kernelSum1 = _mm256_set1_ps((post->srcKernelSum + bk * realDst)[1]);

                __m256i D00 = _mm256_set1_epi32(0);
                __m256i D01 = _mm256_set1_epi32(0);

                for (int sz = 0; sz < src_depth_quad; ++sz) {
                    const auto weight_sz = weight_dz + sz * weight_step_Y;
                    const auto src_z     = src_x + sz * GEMMINT8_AVX2_L * realDst;
                    LOAD_INT4_TO_INT8;
                    auto w0 = _mm256_insertf128_si256(_mm256_castsi128_si256(w_0), w_1, 1);
                    auto s0 = _mm256_castps_si256(_mm256_broadcast_ss((float*)src_z + 0));
                    auto s1 = _mm256_castps_si256(_mm256_broadcast_ss((float*)src_z + 1));

                    D00 = MNN_GEMMINT8_W4_DOT(D00, s0, w0);
                    D01 = MNN_GEMMINT8_W4_DOT(D01, s1, w0);
                }
                auto D0 = D00;
                auto D1 = D01;
                auto scaleValue = _mm256_loadu_ps(scale_dz);
                auto weightBiasValue = _mm256_loadu_ps((float*)weightBias_dz);

                auto f0 = _mm256_cvtepi32_ps(D0);
                auto f1 = _mm256_cvtepi32_ps(D1);
                // x_kernelSum x w_quantZero
                auto xy0_0 = _mm256_mul_ps(kernelSum0, weightBiasValue); // x dimemsion first
                auto xy0_1 = _mm256_mul_ps(kernelSum1, weightBiasValue); // ..second
                f0 = _mm256_mul_ps(f0, scaleValue);
                f1 = _mm256_mul_ps(f1, scaleValue);
                if (post->inputScale) {
                    if (post->inputBias) {
                        extrascale0 = _mm256_set1_ps((post->inputScale + bk * realDst)[0]);
                        extrascale1 = _mm256_set1_ps((post->inputScale + bk * realDst)[1]);
                    }
                    f0 = _mm256_mul_ps(f0, extrascale0);
                    f1 = _mm256_mul_ps(f1, extrascale1);
                    if ((post->useInt8 == 0) && post->weightKernelSum && (post->inputBias || (bk == blockNum - 1))) {
                        if (post->inputBias) {
                            auto wsumDz = post->weightKernelSum + dz * (blockNum * GEMMINT8_AVX2_H) + bk * GEMMINT8_AVX2_H;
                            auto wsum = _mm256_loadu_ps(wsumDz);
                            extrabias0 = _mm256_set1_ps((post->inputBias + bk * realDst)[0]);
                            extrabias1 = _mm256_set1_ps((post->inputBias + bk * realDst)[1]);
                            bias0 = _mm256_mul_ps(extrabias0, wsum);
                            bias1 = _mm256_mul_ps(extrabias1, wsum);
                        } else if (bk == blockNum - 1) { // if input not block quant, only accum once!
                            auto wsumDz = post->weightKernelSum + dz * GEMMINT8_AVX2_H;
                            auto wsum = _mm256_loadu_ps(wsumDz);
                            bias0 = _mm256_mul_ps(_mm256_mul_ps(extrascale0, neg128_f), wsum);
                            bias1 = _mm256_mul_ps(_mm256_mul_ps(extrascale1, neg128_f), wsum);
                        }
                        f0 = _mm256_add_ps(f0, bias0);
                        f1 = _mm256_add_ps(f1, bias1);
                    }
                }
                f0 = _mm256_add_ps(f0, xy0_0);
                f1 = _mm256_add_ps(f1, xy0_1);

                if (bk > 0) {
                    auto dstv0 = _mm256_loadu_ps(((float*)accum_x) + 0 * AVX2_PACKINT8);
                    auto dstv1 = _mm256_loadu_ps(((float*)accum_x) + 1 * AVX2_PACKINT8);
                    f0         = _mm256_add_ps(f0, dstv0);
                    f1         = _mm256_add_ps(f1, dstv1);
                }
                if (bk == blockNum - 1) {
                    if (nullptr != biasPtr) {
                        const auto bias_dz   = biasPtr + dz * AVX2_PACKINT8;
                        auto biasValue       = _mm256_loadu_ps(bias_dz);
                        f0 = _mm256_add_ps(f0, biasValue);
                        f1 = _mm256_add_ps(f1, biasValue);
                    }
                    if (post->fp32minmax) {
                        f0 = _mm256_min_ps(f0, fp32max);
                        f1 = _mm256_min_ps(f1, fp32max);
                        f0 = _mm256_max_ps(f0, fp32min);
                        f1 = _mm256_max_ps(f1, fp32min);
                    }
                        _mm256_storeu_ps(((float*)dst_x) + 0 * AVX2_PACKINT8, f0);
                        _mm256_storeu_ps(((float*)dst_x) + 1 * AVX2_PACKINT8, f1);
                } else {
                    _mm256_storeu_ps(((float*)accum_x) + 0 * AVX2_PACKINT8, f0);
                    _mm256_storeu_ps(((float*)accum_x) + 1 * AVX2_PACKINT8, f1);
                }
            }
        }
        return;
    }
    if (1 == realDst) {
        for (int dz = 0; dz < dst_depth_quad; ++dz) {
            auto dst_x           = dst + dz * dst_step_tmp;
            auto accum_x       = accumbuff;

            for (int bk = 0; bk < blockNum; ++bk) {
                // block's weight&scale&bias
                const auto weight_dz = weight + dz * blockNum * weight_step_Z + bk *  weight_step_Z;
                const float* scale_dz = reinterpret_cast<const float*>(weight_dz + src_depth_quad * weight_step_Y);
                const auto weightBias_dz = scale_dz + GEMMINT8_AVX2_H;
                // block's input
                const auto src_x = src + bk * src_depth_quad * GEMMINT8_AVX2_L * realDst;
                // source kernel sum
                kernelSum0 = _mm256_set1_ps((post->srcKernelSum + bk * realDst)[0]);

                __m256i D00 = _mm256_set1_epi32(0);

                for (int sz = 0; sz < src_depth_quad; ++sz) {
                    const auto weight_sz = weight_dz + sz * weight_step_Y;
                    const auto src_z     = src_x + sz * GEMMINT8_AVX2_L * realDst;
                    LOAD_INT4_TO_INT8;
                    auto w0 = _mm256_insertf128_si256(_mm256_castsi128_si256(w_0), w_1, 1);
                    auto s0 = _mm256_castps_si256(_mm256_broadcast_ss((float*)src_z + 0));

                    D00 = MNN_GEMMINT8_W4_DOT(D00, s0, w0);
                }
                auto D0 = D00;
                auto scaleValue = _mm256_loadu_ps(scale_dz);
                auto weightBiasValue = _mm256_loadu_ps((float*)weightBias_dz);

                auto f0 = _mm256_cvtepi32_ps(D0);
                // x_kernelSum x w_quantZero
                auto xy0_0 = _mm256_mul_ps(kernelSum0, weightBiasValue); // x dimemsion first
                f0 = _mm256_mul_ps(f0, scaleValue);
                if (post->inputScale) {
                    if (post->inputBias) {
                        extrascale0 = _mm256_set1_ps((post->inputScale + bk * realDst)[0]);
                    }
                    f0 = _mm256_mul_ps(f0, extrascale0);
                    if ((post->useInt8 == 0) && post->weightKernelSum && (post->inputBias || (bk == blockNum - 1))) {
                        if (post->inputBias) {
                            auto wsumDz = post->weightKernelSum + dz * (blockNum * GEMMINT8_AVX2_H) + bk * GEMMINT8_AVX2_H;
                            auto wsum = _mm256_loadu_ps(wsumDz);
                            extrabias0 = _mm256_set1_ps((post->inputBias + bk * realDst)[0]);
                            bias0 = _mm256_mul_ps(extrabias0, wsum);
                        } else if (bk == blockNum - 1) { // if input not block quant, only accum once!
                            auto wsumDz = post->weightKernelSum + dz * GEMMINT8_AVX2_H;
                            auto wsum = _mm256_loadu_ps(wsumDz);
                            bias0 = _mm256_mul_ps(_mm256_mul_ps(extrascale0, neg128_f), wsum);
                        }
                        f0 = _mm256_add_ps(f0, bias0);
                    }
                }
                f0 = _mm256_add_ps(f0, xy0_0);

                if (bk > 0) {
                    auto dstv = _mm256_loadu_ps(((float*)accum_x));
                    f0        = _mm256_add_ps(f0, dstv);
                }
                if (bk == 0) {
                    if (biasPtr) {
                        const auto bias_dz   = biasPtr + dz * AVX2_PACKINT8;
                        auto biasValue       = _mm256_loadu_ps(bias_dz);
                        f0 = _mm256_add_ps(f0, biasValue);
                    }
                }
                if (bk == blockNum - 1) {
                    if (post->fp32minmax) {
                        f0 = _mm256_min_ps(f0, fp32max);
                        f0 = _mm256_max_ps(f0, fp32min);
                    }
                    _mm256_storeu_ps(((float*)dst_x), f0);

                } else {
                    _mm256_storeu_ps(((float*)accum_x) , f0);
                }
            }
        }
        return;
    }

}
