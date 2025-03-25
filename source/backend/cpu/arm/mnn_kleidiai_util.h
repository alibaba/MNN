//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <arm_neon.h>
#include <assert.h>
#include <cfloat>
#include <stdint.h>
#include <string.h>
#include <vector>

#include "kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.h"
#include "kai_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon.h"
#include "kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod.h"
#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm.h"
#include "kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa.h"
#include "kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot.h"

#include "kai_common.h"

namespace MNN {
    class KleidiAIUtil {
    public:
    struct rhsPackParamCommon {
        int8_t mLhsZeroPoint = 1;
        uint8_t mRhsZeroPoint = 8;
    };

    static void transferNCHWToNC4HW4(float* src, float* dst, size_t rowNum, size_t rowSize);
    static void transferNCHWToNC4HW4(__fp16* src, __fp16* dst, size_t rowNum, size_t rowSize);
    static void transferNC4HW4ToNCHW(float* src, float* dst, size_t rowNum, size_t rowSize);
    static void transferNC4HW4ToNCHW(__fp16* src, __fp16* dst, size_t rowNum, size_t rowSize);

    /// Rhs pack functions for matmul_clamp_f32_qai8dxp_qsi4cxp.
    static void packQsi4cxps16s0Qs4cxs0s1(
        size_t num_groups,   //
        size_t n,            //
        size_t k,            //
        size_t nr,           //
        size_t kr,           //
        size_t sr,           //
        const uint8_t* rhs,  //
        const float* bias,   //
        const float* scale,  //
        void* rhs_packed,    //
        size_t extra_bytes,  //
        const struct KleidiAIUtil::rhsPackParamCommon* paramsCommon);  //

    static void packQsi4cxps16s0Qs4cx(
        size_t num_groups,   //
        size_t n,            //
        size_t k,            //
        size_t nr,           //
        size_t kr,           //
        size_t sr,           //
        const uint8_t* rhs,  //
        const float* bias,   //
        const float* scale,  //
        void* rhs_packed,    //
        size_t extra_bytes,  //
        const struct KleidiAIUtil::rhsPackParamCommon* paramsCommon);  //

    static void packQsi4cxps1s0Qsu4cxs0s1(
        size_t num_groups,   //
        size_t n,            //
        size_t k,            //
        size_t nr,           //
        size_t kr,           //
        size_t sr,           //
        const uint8_t* rhs,  //
        const float* bias,   //
        const float* scale,  //
        void* rhs_packed,    //
        size_t extra_bytes,  //
        const struct KleidiAIUtil::rhsPackParamCommon* paramsCommon);
};
}