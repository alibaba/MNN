#define GEMMINT8_AVX512_H GEMMINT8_AVX512_H_NOVNNI

#define AVX512_BROADCAST_INT32(src) _mm512_castps_si512(_mm512_broadcastss_ps(_mm_load_ss(src)))
#define SCALE_BIAS_VEC(N) \
            auto d##N = _mm512_add_epi32(D##N, biasValue);\
            auto f##N = _mm512_cvtepi32_ps(d##N);\
            f##N = _mm512_mul_ps(f##N, scaleValue);

#define POSTTREAT(N, O) \
                f##N = _mm512_min_ps(f##N, maxValue);\
                f##N = _mm512_max_ps(f##N, minValue);\
                auto m##N = _mm512_cmp_ps_mask(f##N, zero512, 1);\
                auto b##N = _mm512_mask_blend_ps(m##N, plus, minus);\
                f##N = _mm512_add_ps(f##N, b##N);\
                d##N = _mm512_cvtps_epi32(_mm512_roundscale_ps(f##N, 3));\
                auto hd##N = _mm512_cvtsepi32_epi16(d##N); hd##N = _mm256_add_epi16(hd##N, offset);\
                auto h0##N = _mm256_extracti128_si256(hd##N, 0);\
                auto h1##N = _mm256_extracti128_si256(hd##N, 1);\
                h0##N = _mm_packus_epi16(h0##N, h1##N);\
                _mm_storeu_si128((__m128i*)dst_x + O, h0##N);


// GemmInt8 with NO VNNI
void MATMULCOREFUNC_NAME(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step, size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realDst) {
    const auto dst_step_tmp = dst_step / sizeof(int8_t);
    auto zero512 = _mm512_set1_ps(0.0f);
    auto minValue = _mm512_set1_ps(post->minValue);
    auto maxValue = _mm512_set1_ps(post->maxValue);
    auto plus = _mm512_set1_ps(0.5f);
    auto minus = _mm512_set1_ps(-0.5f);
    auto offset = _mm256_set1_epi16(128);
    int dzUnit = GEMMINT8_AVX512_H / PACK_UNIT;
    int dzU = dst_depth_quad / dzUnit;
    int dzR = dst_depth_quad % dzUnit;
    auto one = _mm512_set1_epi16(1);
    if (realDst == GEMMINT8_AVX512_E) {
        for (int dz = 0; dz < dzU; ++dz) {
            auto weight_dz = weight + dz * src_depth_quad * (GEMMINT8_AVX512_L * GEMMINT8_AVX512_H);
            auto bias_dz = (int32_t*)post->bias + dz * PACK_UNIT * dzUnit;
            float* scale_dz = (float*)post->scale + dz * PACK_UNIT * dzUnit;
            auto dst_z = dst + dz * dst_step_tmp * dzUnit;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m512i D0 = _mm512_set1_epi32(0);
            __m512i D1 = _mm512_set1_epi32(0);
            __m512i D2 = _mm512_set1_epi32(0);
            __m512i D3 = _mm512_set1_epi32(0);

            __m512i D4 = _mm512_set1_epi32(0);
            __m512i D5 = _mm512_set1_epi32(0);
            __m512i D6 = _mm512_set1_epi32(0);
            __m512i D7 = _mm512_set1_epi32(0);

            __m512i D8 = _mm512_set1_epi32(0);
            __m512i D9 = _mm512_set1_epi32(0);
            __m512i D10 = _mm512_set1_epi32(0);
            __m512i D11 = _mm512_set1_epi32(0);

            __m512i D12 = _mm512_set1_epi32(0);
            __m512i D13 = _mm512_set1_epi32(0);
            __m512i D14 = _mm512_set1_epi32(0);
            __m512i D15 = _mm512_set1_epi32(0);


            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMMINT8_AVX512_L * GEMMINT8_AVX512_H) * sz;
                const auto src_z     = (const float*)(src_x + sz * GEMMINT8_AVX512_E * GEMMINT8_AVX512_L);
                auto w0 = _mm512_loadu_si512(weight_sz);
                auto w1 = _mm512_loadu_si512(weight_sz + 1 * PACK_UNIT * GEMMINT8_AVX512_E);
                auto w2 = _mm512_loadu_si512(weight_sz + 2 * PACK_UNIT * GEMMINT8_AVX512_E);
                auto w3 = _mm512_loadu_si512(weight_sz + 3 * PACK_UNIT * GEMMINT8_AVX512_E);

                auto s0 = AVX512_BROADCAST_INT32(src_z + 0);
                auto s1 = AVX512_BROADCAST_INT32(src_z + 1);
                auto s2 = AVX512_BROADCAST_INT32(src_z + 2);
                auto s3 = AVX512_BROADCAST_INT32(src_z + 3);

                D0 = mnn_mm512_dpbusds_epi32(D0, s0, w0);
                D1 = mnn_mm512_dpbusds_epi32(D1, s1, w0);
                D2 = mnn_mm512_dpbusds_epi32(D2, s2, w0);
                D3 = mnn_mm512_dpbusds_epi32(D3, s3, w0);

                D4 = mnn_mm512_dpbusds_epi32(D4, s0, w1);
                D5 = mnn_mm512_dpbusds_epi32(D5, s1, w1);
                D6 = mnn_mm512_dpbusds_epi32(D6, s2, w1);
                D7 = mnn_mm512_dpbusds_epi32(D7, s3, w1);

                D8 = mnn_mm512_dpbusds_epi32(D8, s0, w2);
                D9 = mnn_mm512_dpbusds_epi32(D9, s1, w2);
                D10 = mnn_mm512_dpbusds_epi32(D10, s2, w2);
                D11 = mnn_mm512_dpbusds_epi32(D11, s3, w2);

                D12 = mnn_mm512_dpbusds_epi32(D12, s0, w3);
                D13 = mnn_mm512_dpbusds_epi32(D13, s1, w3);
                D14 = mnn_mm512_dpbusds_epi32(D14, s2, w3);
                D15 = mnn_mm512_dpbusds_epi32(D15, s3, w3);
            }

            auto biasValue = _mm512_loadu_si512(bias_dz);
            auto scaleValue = _mm512_loadu_ps(scale_dz);

            SCALE_BIAS_VEC(0);
            SCALE_BIAS_VEC(1);
            SCALE_BIAS_VEC(2);
            SCALE_BIAS_VEC(3);

            biasValue = _mm512_loadu_si512(bias_dz + 1 * PACK_UNIT);
            scaleValue = _mm512_loadu_ps(scale_dz + 1 * PACK_UNIT);
            SCALE_BIAS_VEC(4);
            SCALE_BIAS_VEC(5);
            SCALE_BIAS_VEC(6);
            SCALE_BIAS_VEC(7);

            biasValue = _mm512_loadu_si512(bias_dz + 2 * PACK_UNIT);
            scaleValue = _mm512_loadu_ps(scale_dz + 2 * PACK_UNIT);
            SCALE_BIAS_VEC(8);
            SCALE_BIAS_VEC(9);
            SCALE_BIAS_VEC(10);
            SCALE_BIAS_VEC(11);

            biasValue = _mm512_loadu_si512(bias_dz + 3 * PACK_UNIT);
            scaleValue = _mm512_loadu_ps(scale_dz + 3 * PACK_UNIT);
            SCALE_BIAS_VEC(12);
            SCALE_BIAS_VEC(13);
            SCALE_BIAS_VEC(14);
            SCALE_BIAS_VEC(15);

            if (post->useInt8 == 0) {
                _mm512_storeu_ps(((float*)dst_x), f0);
                _mm512_storeu_ps(((float*)dst_x) + 16, f1);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 2, f2);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 3, f3);
                dst_x += dst_step_tmp;
                _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f4);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 1, f5);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 2, f6);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 3, f7);
                dst_x += dst_step_tmp;
                _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f8);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 1, f9);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 2, f10);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 3, f11);
                dst_x += dst_step_tmp;
                _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f12);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 1, f13);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 2, f14);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 3, f15);
            } else {
                POSTTREAT(0, 0);
                POSTTREAT(1, 1);
                POSTTREAT(2, 2);
                POSTTREAT(3, 3);
                dst_x += dst_step_tmp;

                POSTTREAT(4, 0);
                POSTTREAT(5, 1);
                POSTTREAT(6, 2);
                POSTTREAT(7, 3);
                dst_x += dst_step_tmp;

                POSTTREAT(8, 0);
                POSTTREAT(9, 1);
                POSTTREAT(10, 2);
                POSTTREAT(11, 3);
                dst_x += dst_step_tmp;

                POSTTREAT(12, 0);
                POSTTREAT(13, 1);
                POSTTREAT(14, 2);
                POSTTREAT(15, 3);
            }
        }
        auto weight_dz = weight + dzU * src_depth_quad * (GEMMINT8_AVX512_L * GEMMINT8_AVX512_H);
        auto bias_dz = (int32_t*)post->bias + dzU * PACK_UNIT * dzUnit;
        float* scale_dz = (float*)post->scale + dzU * PACK_UNIT * dzUnit;

        auto dst_z = dst + dzU * dst_step_tmp * dzUnit;
        const auto src_x   = src;
        auto dst_x         = dst_z;
        for (int i=0; i<dzR; ++i) {
            __m512i D0 = _mm512_set1_epi32(0);
            __m512i D1 = _mm512_set1_epi32(0);
            __m512i D2 = _mm512_set1_epi32(0);
            __m512i D3 = _mm512_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMMINT8_AVX512_L * GEMMINT8_AVX512_H) * sz;
                const auto src_z     = (const float*)(src_x + sz * GEMMINT8_AVX512_E * GEMMINT8_AVX512_L);
                auto w0 = _mm512_loadu_si512(weight_sz);

                auto s0 = AVX512_BROADCAST_INT32(src_z + 0);
                auto s1 = AVX512_BROADCAST_INT32(src_z + 1);
                auto s2 = AVX512_BROADCAST_INT32(src_z + 2);
                auto s3 = AVX512_BROADCAST_INT32(src_z + 3);

                D0 = mnn_mm512_dpbusds_epi32(D0, s0, w0);
                D1 = mnn_mm512_dpbusds_epi32(D1, s1, w0);
                D2 = mnn_mm512_dpbusds_epi32(D2, s2, w0);
                D3 = mnn_mm512_dpbusds_epi32(D3, s3, w0);
            }

            auto biasValue = _mm512_loadu_si512(bias_dz);
            auto scaleValue = _mm512_loadu_ps(scale_dz);

            SCALE_BIAS_VEC(0);
            SCALE_BIAS_VEC(1);
            SCALE_BIAS_VEC(2);
            SCALE_BIAS_VEC(3);

            if (post->useInt8 == 0) {
                _mm512_storeu_ps(((float*)dst_x), f0);
                _mm512_storeu_ps(((float*)dst_x) + 16, f1);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 2, f2);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 3, f3);
            } else {
                POSTTREAT(0, 0);
                POSTTREAT(1, 1);
                POSTTREAT(2, 2);
                POSTTREAT(3, 3);
            }
            dst_x += dst_step_tmp;
            scale_dz += PACK_UNIT;
            bias_dz += PACK_UNIT;
            weight_dz += PACK_UNIT * GEMMINT8_AVX512_E;
        }
        return;
    }
    // e = 3
    if (realDst == 3) {
        for (int dz = 0; dz < dzU; ++dz) {
            auto weight_dz = weight + dz * src_depth_quad * (GEMMINT8_AVX512_L * GEMMINT8_AVX512_H);
            auto bias_dz = (int32_t*)post->bias + dz * PACK_UNIT * dzUnit;
            float* scale_dz = (float*)post->scale + dz * PACK_UNIT * dzUnit;
            auto dst_z = dst + dz * dst_step_tmp * dzUnit;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m512i D0 = _mm512_set1_epi32(0);
            __m512i D1 = _mm512_set1_epi32(0);
            __m512i D2 = _mm512_set1_epi32(0);

            __m512i D4 = _mm512_set1_epi32(0);
            __m512i D5 = _mm512_set1_epi32(0);
            __m512i D6 = _mm512_set1_epi32(0);

            __m512i D8 = _mm512_set1_epi32(0);
            __m512i D9 = _mm512_set1_epi32(0);
            __m512i D10 = _mm512_set1_epi32(0);

            __m512i D12 = _mm512_set1_epi32(0);
            __m512i D13 = _mm512_set1_epi32(0);
            __m512i D14 = _mm512_set1_epi32(0);


            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMMINT8_AVX512_L * GEMMINT8_AVX512_H) * sz;
                const auto src_z     = (const float*)(src_x + sz * GEMMINT8_AVX512_E * GEMMINT8_AVX512_L);
                auto w0 = _mm512_loadu_si512(weight_sz);
                auto w1 = _mm512_loadu_si512(weight_sz + 1 * PACK_UNIT * GEMMINT8_AVX512_E);
                auto w2 = _mm512_loadu_si512(weight_sz + 2 * PACK_UNIT * GEMMINT8_AVX512_E);
                auto w3 = _mm512_loadu_si512(weight_sz + 3 * PACK_UNIT * GEMMINT8_AVX512_E);

                auto s0 = AVX512_BROADCAST_INT32(src_z + 0);
                auto s1 = AVX512_BROADCAST_INT32(src_z + 1);
                auto s2 = AVX512_BROADCAST_INT32(src_z + 2);

                D0 = mnn_mm512_dpbusds_epi32(D0, s0, w0);
                D1 = mnn_mm512_dpbusds_epi32(D1, s1, w0);
                D2 = mnn_mm512_dpbusds_epi32(D2, s2, w0);

                D4 = mnn_mm512_dpbusds_epi32(D4, s0, w1);
                D5 = mnn_mm512_dpbusds_epi32(D5, s1, w1);
                D6 = mnn_mm512_dpbusds_epi32(D6, s2, w1);

                D8 = mnn_mm512_dpbusds_epi32(D8, s0, w2);
                D9 = mnn_mm512_dpbusds_epi32(D9, s1, w2);
                D10 = mnn_mm512_dpbusds_epi32(D10, s2, w2);

                D12 = mnn_mm512_dpbusds_epi32(D12, s0, w3);
                D13 = mnn_mm512_dpbusds_epi32(D13, s1, w3);
                D14 = mnn_mm512_dpbusds_epi32(D14, s2, w3);
            }

            auto biasValue = _mm512_loadu_si512(bias_dz);
            auto scaleValue = _mm512_loadu_ps(scale_dz);

            SCALE_BIAS_VEC(0);
            SCALE_BIAS_VEC(1);
            SCALE_BIAS_VEC(2);

            biasValue = _mm512_loadu_si512(bias_dz + 1 * PACK_UNIT);
            scaleValue = _mm512_loadu_ps(scale_dz + 1 * PACK_UNIT);
            SCALE_BIAS_VEC(4);
            SCALE_BIAS_VEC(5);
            SCALE_BIAS_VEC(6);

            biasValue = _mm512_loadu_si512(bias_dz + 2 * PACK_UNIT);
            scaleValue = _mm512_loadu_ps(scale_dz + 2 * PACK_UNIT);
            SCALE_BIAS_VEC(8);
            SCALE_BIAS_VEC(9);
            SCALE_BIAS_VEC(10);

            biasValue = _mm512_loadu_si512(bias_dz + 3 * PACK_UNIT);
            scaleValue = _mm512_loadu_ps(scale_dz + 3 * PACK_UNIT);
            SCALE_BIAS_VEC(12);
            SCALE_BIAS_VEC(13);
            SCALE_BIAS_VEC(14);

            if (post->useInt8 == 0) {
                _mm512_storeu_ps(((float*)dst_x), f0);
                _mm512_storeu_ps(((float*)dst_x) + 16, f1);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 2, f2);
                dst_x += dst_step_tmp;
                _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f4);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 1, f5);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 2, f6);
                dst_x += dst_step_tmp;
                _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f8);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 1, f9);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 2, f10);
                dst_x += dst_step_tmp;
                _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f12);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 1, f13);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 2, f14);
            } else {
                POSTTREAT(0, 0);
                POSTTREAT(1, 1);
                POSTTREAT(2, 2);
                dst_x += dst_step_tmp;

                POSTTREAT(4, 0);
                POSTTREAT(5, 1);
                POSTTREAT(6, 2);
                dst_x += dst_step_tmp;

                POSTTREAT(8, 0);
                POSTTREAT(9, 1);
                POSTTREAT(10, 2);
                dst_x += dst_step_tmp;

                POSTTREAT(12, 0);
                POSTTREAT(13, 1);
                POSTTREAT(14, 2);
            }
        }
        auto weight_dz = weight + dzU * src_depth_quad * (GEMMINT8_AVX512_L * GEMMINT8_AVX512_H);
        auto bias_dz = (int32_t*)post->bias + dzU * PACK_UNIT * dzUnit;
        float* scale_dz = (float*)post->scale + dzU * PACK_UNIT * dzUnit;

        auto dst_z = dst + dzU * dst_step_tmp * dzUnit;
        const auto src_x   = src;
        auto dst_x         = dst_z;
        for (int i=0; i<dzR; ++i) {
            __m512i D0 = _mm512_set1_epi32(0);
            __m512i D1 = _mm512_set1_epi32(0);
            __m512i D2 = _mm512_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMMINT8_AVX512_L * GEMMINT8_AVX512_H) * sz;
                const auto src_z     = (const float*)(src_x + sz * GEMMINT8_AVX512_E * GEMMINT8_AVX512_L);
                auto w0 = _mm512_loadu_si512(weight_sz);

                auto s0 = AVX512_BROADCAST_INT32(src_z + 0);
                auto s1 = AVX512_BROADCAST_INT32(src_z + 1);
                auto s2 = AVX512_BROADCAST_INT32(src_z + 2);

                D0 = mnn_mm512_dpbusds_epi32(D0, s0, w0);
                D1 = mnn_mm512_dpbusds_epi32(D1, s1, w0);
                D2 = mnn_mm512_dpbusds_epi32(D2, s2, w0);
            }

            auto biasValue = _mm512_loadu_si512(bias_dz);
            auto scaleValue = _mm512_loadu_ps(scale_dz);

            SCALE_BIAS_VEC(0);
            SCALE_BIAS_VEC(1);
            SCALE_BIAS_VEC(2);

            if (post->useInt8 == 0) {
                _mm512_storeu_ps(((float*)dst_x), f0);
                _mm512_storeu_ps(((float*)dst_x) + 16, f1);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 2, f2);
            } else {
                POSTTREAT(0, 0);
                POSTTREAT(1, 1);
                POSTTREAT(2, 2);
            }
            dst_x += dst_step_tmp;
            scale_dz += PACK_UNIT;
            bias_dz += PACK_UNIT;
            weight_dz += PACK_UNIT * GEMMINT8_AVX512_E;
        }
        return;
    }
    // e = 2
    if (realDst == 2) {
        for (int dz = 0; dz < dzU; ++dz) {
            auto weight_dz = weight + dz * src_depth_quad * (GEMMINT8_AVX512_L * GEMMINT8_AVX512_H);
            auto bias_dz = (int32_t*)post->bias + dz * PACK_UNIT * dzUnit;
            float* scale_dz = (float*)post->scale + dz * PACK_UNIT * dzUnit;
            auto dst_z = dst + dz * dst_step_tmp * dzUnit;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m512i D0 = _mm512_set1_epi32(0);
            __m512i D1 = _mm512_set1_epi32(0);

            __m512i D4 = _mm512_set1_epi32(0);
            __m512i D5 = _mm512_set1_epi32(0);

            __m512i D8 = _mm512_set1_epi32(0);
            __m512i D9 = _mm512_set1_epi32(0);

            __m512i D12 = _mm512_set1_epi32(0);
            __m512i D13 = _mm512_set1_epi32(0);


            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMMINT8_AVX512_L * GEMMINT8_AVX512_H) * sz;
                const auto src_z     = (const float*)(src_x + sz * GEMMINT8_AVX512_E * GEMMINT8_AVX512_L);
                auto w0 = _mm512_loadu_si512(weight_sz);
                auto w1 = _mm512_loadu_si512(weight_sz + 1 * PACK_UNIT * GEMMINT8_AVX512_E);
                auto w2 = _mm512_loadu_si512(weight_sz + 2 * PACK_UNIT * GEMMINT8_AVX512_E);
                auto w3 = _mm512_loadu_si512(weight_sz + 3 * PACK_UNIT * GEMMINT8_AVX512_E);

                auto s0 = AVX512_BROADCAST_INT32(src_z + 0);
                auto s1 = AVX512_BROADCAST_INT32(src_z + 1);

                D0 = mnn_mm512_dpbusds_epi32(D0, s0, w0);
                D1 = mnn_mm512_dpbusds_epi32(D1, s1, w0);

                D4 = mnn_mm512_dpbusds_epi32(D4, s0, w1);
                D5 = mnn_mm512_dpbusds_epi32(D5, s1, w1);

                D8 = mnn_mm512_dpbusds_epi32(D8, s0, w2);
                D9 = mnn_mm512_dpbusds_epi32(D9, s1, w2);

                D12 = mnn_mm512_dpbusds_epi32(D12, s0, w3);
                D13 = mnn_mm512_dpbusds_epi32(D13, s1, w3);
            }

            auto biasValue = _mm512_loadu_si512(bias_dz);
            auto scaleValue = _mm512_loadu_ps(scale_dz);

            SCALE_BIAS_VEC(0);
            SCALE_BIAS_VEC(1);

            biasValue = _mm512_loadu_si512(bias_dz + 1 * PACK_UNIT);
            scaleValue = _mm512_loadu_ps(scale_dz + 1 * PACK_UNIT);
            SCALE_BIAS_VEC(4);
            SCALE_BIAS_VEC(5);

            biasValue = _mm512_loadu_si512(bias_dz + 2 * PACK_UNIT);
            scaleValue = _mm512_loadu_ps(scale_dz + 2 * PACK_UNIT);
            SCALE_BIAS_VEC(8);
            SCALE_BIAS_VEC(9);

            biasValue = _mm512_loadu_si512(bias_dz + 3 * PACK_UNIT);
            scaleValue = _mm512_loadu_ps(scale_dz + 3 * PACK_UNIT);
            SCALE_BIAS_VEC(12);
            SCALE_BIAS_VEC(13);

            if (post->useInt8 == 0) {
                _mm512_storeu_ps(((float*)dst_x), f0);
                _mm512_storeu_ps(((float*)dst_x) + 16, f1);
                dst_x += dst_step_tmp;
                _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f4);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 1, f5);
                dst_x += dst_step_tmp;
                _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f8);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 1, f9);
                dst_x += dst_step_tmp;
                _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f12);
                _mm512_storeu_ps(((float*)dst_x) + 16 * 1, f13);
            } else {
                POSTTREAT(0, 0);
                POSTTREAT(1, 1);
                dst_x += dst_step_tmp;

                POSTTREAT(4, 0);
                POSTTREAT(5, 1);
                dst_x += dst_step_tmp;

                POSTTREAT(8, 0);
                POSTTREAT(9, 1);
                dst_x += dst_step_tmp;

                POSTTREAT(12, 0);
                POSTTREAT(13, 1);
            }
        }
        auto weight_dz = weight + dzU * src_depth_quad * (GEMMINT8_AVX512_L * GEMMINT8_AVX512_H);
        auto bias_dz = (int32_t*)post->bias + dzU * PACK_UNIT * dzUnit;
        float* scale_dz = (float*)post->scale + dzU * PACK_UNIT * dzUnit;

        auto dst_z = dst + dzU * dst_step_tmp * dzUnit;
        const auto src_x   = src;
        auto dst_x         = dst_z;
        for (int i=0; i<dzR; ++i) {
            __m512i D0 = _mm512_set1_epi32(0);
            __m512i D1 = _mm512_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMMINT8_AVX512_L * GEMMINT8_AVX512_H) * sz;
                const auto src_z     = (const float*)(src_x + sz * GEMMINT8_AVX512_E * GEMMINT8_AVX512_L);
                auto w0 = _mm512_loadu_si512(weight_sz);

                auto s0 = AVX512_BROADCAST_INT32(src_z + 0);
                auto s1 = AVX512_BROADCAST_INT32(src_z + 1);

                D0 = mnn_mm512_dpbusds_epi32(D0, s0, w0);
                D1 = mnn_mm512_dpbusds_epi32(D1, s1, w0);
            }

            auto biasValue = _mm512_loadu_si512(bias_dz);
            auto scaleValue = _mm512_loadu_ps(scale_dz);

            SCALE_BIAS_VEC(0);
            SCALE_BIAS_VEC(1);

            if (post->useInt8 == 0) {
                _mm512_storeu_ps(((float*)dst_x), f0);
                _mm512_storeu_ps(((float*)dst_x) + 16, f1);
            } else {
                POSTTREAT(0, 0);
                POSTTREAT(1, 1);
            }
            dst_x += dst_step_tmp;
            scale_dz += PACK_UNIT;
            bias_dz += PACK_UNIT;
            weight_dz += PACK_UNIT * GEMMINT8_AVX512_E;
        }
        return;
    }
    if (realDst == 1) {
        for (int dz = 0; dz < dzU; ++dz) {
            auto weight_dz = weight + dz * src_depth_quad * (GEMMINT8_AVX512_L * GEMMINT8_AVX512_H);
            auto bias_dz = (int32_t*)post->bias + dz * PACK_UNIT * dzUnit;
            float* scale_dz = (float*)post->scale + dz * PACK_UNIT * dzUnit;
            auto dst_z = dst + dz * dst_step_tmp * dzUnit;
            const auto src_x   = src;
            auto dst_x         = dst_z;
            __m512i D0 = _mm512_set1_epi32(0);

            __m512i D4 = _mm512_set1_epi32(0);

            __m512i D8 = _mm512_set1_epi32(0);

            __m512i D12 = _mm512_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMMINT8_AVX512_L * GEMMINT8_AVX512_H) * sz;
                const auto src_z     = (const float*)(src_x + sz * GEMMINT8_AVX512_E * GEMMINT8_AVX512_L);
                auto w0 = _mm512_loadu_si512(weight_sz);
                auto w1 = _mm512_loadu_si512(weight_sz + 1 * PACK_UNIT * GEMMINT8_AVX512_E);
                auto w2 = _mm512_loadu_si512(weight_sz + 2 * PACK_UNIT * GEMMINT8_AVX512_E);
                auto w3 = _mm512_loadu_si512(weight_sz + 3 * PACK_UNIT * GEMMINT8_AVX512_E);

                auto s0 = AVX512_BROADCAST_INT32(src_z + 0);

                D0 = mnn_mm512_dpbusds_epi32(D0, s0, w0);

                D4 = mnn_mm512_dpbusds_epi32(D4, s0, w1);

                D8 = mnn_mm512_dpbusds_epi32(D8, s0, w2);

                D12 = mnn_mm512_dpbusds_epi32(D12, s0, w3);
            }

            auto biasValue = _mm512_loadu_si512(bias_dz);
            auto scaleValue = _mm512_loadu_ps(scale_dz);

            SCALE_BIAS_VEC(0);

            biasValue = _mm512_loadu_si512(bias_dz + 1 * PACK_UNIT);
            scaleValue = _mm512_loadu_ps(scale_dz + 1 * PACK_UNIT);
            SCALE_BIAS_VEC(4);

            biasValue = _mm512_loadu_si512(bias_dz + 2 * PACK_UNIT);
            scaleValue = _mm512_loadu_ps(scale_dz + 2 * PACK_UNIT);
            SCALE_BIAS_VEC(8);

            biasValue = _mm512_loadu_si512(bias_dz + 3 * PACK_UNIT);
            scaleValue = _mm512_loadu_ps(scale_dz + 3 * PACK_UNIT);
            SCALE_BIAS_VEC(12);

            if (post->useInt8 == 0) {
                _mm512_storeu_ps(((float*)dst_x), f0);
                dst_x += dst_step_tmp;
                _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f4);
                dst_x += dst_step_tmp;
                _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f8);
                dst_x += dst_step_tmp;
                _mm512_storeu_ps(((float*)dst_x) + 16 * 0, f12);
            } else {
                POSTTREAT(0, 0);
                dst_x += dst_step_tmp;

                POSTTREAT(4, 0);
                dst_x += dst_step_tmp;

                POSTTREAT(8, 0);
                dst_x += dst_step_tmp;

                POSTTREAT(12, 0);
            }
        }
        auto weight_dz = weight + dzU * src_depth_quad * (GEMMINT8_AVX512_L * GEMMINT8_AVX512_H);
        auto bias_dz = (int32_t*)post->bias + dzU * PACK_UNIT * dzUnit;
        float* scale_dz = (float*)post->scale + dzU * PACK_UNIT * dzUnit;

        auto dst_z = dst + dzU * dst_step_tmp * dzUnit;
        const auto src_x   = src;
        auto dst_x         = dst_z;
        for (int i=0; i<dzR; ++i) {
            __m512i D0 = _mm512_set1_epi32(0);

            for (int sz = 0; sz < src_depth_quad; ++sz) {
                const auto weight_sz = weight_dz + (GEMMINT8_AVX512_L * GEMMINT8_AVX512_H) * sz;
                const auto src_z     = (const float*)(src_x + sz * GEMMINT8_AVX512_E * GEMMINT8_AVX512_L);
                auto w0 = _mm512_loadu_si512(weight_sz);

                auto s0 = AVX512_BROADCAST_INT32(src_z + 0);

                D0 = mnn_mm512_dpbusds_epi32(D0, s0, w0);
            }

            auto biasValue = _mm512_loadu_si512(bias_dz);
            auto scaleValue = _mm512_loadu_ps(scale_dz);

            SCALE_BIAS_VEC(0);

            if (post->useInt8 == 0) {
                _mm512_storeu_ps(((float*)dst_x), f0);
            } else {
                POSTTREAT(0, 0);
            }
            dst_x += dst_step_tmp;
            scale_dz += PACK_UNIT;
            bias_dz += PACK_UNIT;
            weight_dz += PACK_UNIT * GEMMINT8_AVX512_E;
        }
        return;
    }
}