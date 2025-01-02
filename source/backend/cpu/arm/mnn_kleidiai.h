//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <MNN/ErrorCode.hpp>
#include <core/Backend.hpp>
#include <core/Execution.hpp>
#include <core/TensorUtils.hpp>
#include <backend/cpu/CPUBackend.hpp>
#include <backend/cpu/CPURuntime.hpp>

#include <arm_neon.h>
#include <assert.h>
#include <cfloat>
#include <stdint.h>
#include <string.h>
#include <vector>

#include "kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.h"
#include "kai_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod.h"
#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm.h"

#include "kai_common.h"

namespace MNN {
    class KleidiAI {
    public:
        static KleidiAI &getInstance(bool bAsymmetric, bool acthalf, bool blockwise) {
            if(!instance) {
                instance = new KleidiAI(bAsymmetric, acthalf, blockwise);
            }
            return *instance;
        }

        static KleidiAI &getInstance() {
            if(!instance) {
                instance = new KleidiAI;
            }
            return *instance;
        }

        ~KleidiAI() {}

        typedef struct KaiInfo {
            bool kaiEnable = false;
            bool asymmetric = false; //Asymmetric quantized model.
            bool acthalf = false; // activation half precision.
            bool blockwise = false; // weight quant using block wise.
            bool dot = false; //CPU support sdot.
            bool i8mm = false; //CPU support i8mm.
        } KaiInfo;

        //Kai util
        void packNCHWToNC4HW4(float* data, size_t rowNum, size_t rowSize);
        void packNC4HW4ToNCHW(float* data, size_t rowNum, size_t rowSize);

        //Set info
        void setEnable(bool enable);
        void setModelAsymmetric(bool bAsymmetric);

        //Check
        bool canAccelerate() {
            return (mKaiInfo.kaiEnable && mKaiInfo.dot && mKaiInfo.i8mm &&
                    !mKaiInfo.asymmetric && !mKaiInfo.acthalf && !mKaiInfo.blockwise);
        }

        //Get info
        size_t getMr(size_t m = 1) { return (m == 1) ? mKaiMrDotprod : mKaiMrI8mm; }
        size_t getNr() { return mKaiNr; }
        size_t getKr() { return mKaiKr; }
        size_t getSr() { return mKaiSr; }
        size_t getMStep(size_t m = 1) { return (m == 1) ? mKaiMstepDotprod : mKaiMstepI8mm; }
        size_t getNStep() { return mKaiNStep; }
        size_t getVecNumPerThread(size_t totalVec, size_t totalThread, size_t minStep) { return kai_roundup((totalVec + totalThread - 1) / totalThread, minStep); }

        //Lhs
        size_t getLhsQuantedPackedSize(size_t m, size_t k);
        size_t getLhsQuantedPackedOffset(size_t m, size_t mIdx, size_t k);
        void runLhsQuantPack(size_t m, size_t k, size_t mr, const void* lhs, void* lhsQuantedPacked);

        //Rhs
        size_t getRhsPackedSize(size_t n, size_t k);
        size_t getRhsPackedOffset(size_t nIdx, size_t k);
        void runRhsPack(size_t n, size_t k, const void* rhs, const void* scale, const void *bias, void* rhsPacked, bool packedInt4 = false);

        //Dst
        size_t getDstOffset(size_t mIdx, size_t nIdx, size_t n) { return (nIdx * sizeof(float)) + mIdx * (n * sizeof(float)); }

        //Matmul
        void runMatmul(size_t m, size_t n, size_t k, const void* lhsPacked, const void* rhsPacked, size_t dst_stride, void* dst);

    private:
        KleidiAI(bool bAsymmetric = false, bool acthalf = false, bool blockwise = false) {
            const MNNCPUInfo& gCPUInfo = *MNNGetCPUInfo();
            mKaiInfo.dot = gCPUInfo.dot;
            mKaiInfo.i8mm = gCPUInfo.i8mm;
            mKaiInfo.kaiEnable = true;
            mKaiInfo.asymmetric = bAsymmetric;
            mKaiInfo.acthalf = acthalf;
            mKaiInfo.blockwise = blockwise;

            if(canAccelerate()) {
                MNN_PRINT("\nKleidiAI is running!\n");
            }
        }

        static KleidiAI *instance;
        KaiInfo mKaiInfo;

        const size_t mKaiMstepDotprod = 1;
        const size_t mKaiMstepI8mm = 8;
        const size_t mKaiNStep = 4;

        const size_t mKaiMrDotprod = 1;
        const size_t mKaiMrI8mm = 4;
        const size_t mKaiNr = 4;
        const size_t mKaiKr = 16;
        const size_t mKaiSr = 2;
    };
}