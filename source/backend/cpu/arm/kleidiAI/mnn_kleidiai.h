//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <MNN/ErrorCode.hpp>
#include "core/Backend.hpp"
#include "core/Execution.hpp"
#include "core/TensorUtils.hpp"
#include "core/ConvolutionCommon.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/CPURuntime.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"

#include "mnn_kleidiai_util.h"

namespace MNN {
    class KleidiAI {
    public:
        //Define some necessary data structures.
        enum class accelType {
            /*
            QI4: Int4 quantified;
            ASYM/SYM: Asymmetric/symmetric;
            CHNLQT/BLKQT: Per channel quantified/Per block quantified;
            FP16: FP16 output.
            */
            QI4_ASYM_FP32_CHNLQT,
            QI4_ASYM_FP32_BLKQT,
            QI4_ASYM_FP16_CHNLQT,
            QI4_ASYM_FP16_BLKQT,
            QI4_SYM_FP32_CHNLQT,
            QI4_SYM_FP32_BLKQT,
            QI4_SYM_FP16_CHNLQT,
            QI4_SYM_FP16_BLKQT,
            NOT_SUPPORT
        };

        typedef struct modelInfo {
            bool mQi4; //Int4 quant.
            bool mAsymmetric; //Asymmetric quantized model.
            bool mFp16; //fp16 or fp32.
            size_t mBlockSize; //0: Per channel quant; others: Per block quant.

            modelInfo(bool qi4 = false, bool asymmetric = false, bool fp16 = false, size_t blockSize = 0) {
                mQi4 = qi4;
                mAsymmetric = asymmetric;
                mFp16 = fp16;
                mBlockSize = blockSize;
            }

            bool operator<(const modelInfo& rhs) const {
                if(mQi4 != rhs.mQi4) {
                    return mQi4 < rhs.mQi4;
                }

                if(mAsymmetric != rhs.mAsymmetric) {
                    return mAsymmetric < rhs.mAsymmetric;
                }

                if(mFp16 != rhs.mFp16) {
                    return mFp16 < rhs.mFp16;
                }

                bool lhsPerChannel = mBlockSize == 0 ? true : false;
                bool rhsPerChannel = rhs.mBlockSize == 0 ? true : false;
                return lhsPerChannel < rhsPerChannel;
            }

            bool support() const {
                return mQi4 == true && mBlockSize % 32 == 0;
            }

            void print() const {
                MNN_PRINT("\nKleidiAI loaded model info: qi4 = %s, asymmetric = %s, fp16 = %s, blockSize = %ld\n",
                          mQi4 ? "TRUE" : "FALSE",
                          mAsymmetric ? "TRUE" : "FALSE",
                          mFp16 ? "TRUE" : "FALSE",
                          mBlockSize);
            }
        } modelInfo;

        typedef struct CPUInfo {
            bool mDot = false;
            bool mI8mm = false;
            bool mSme2 = false;

            void operator=(const MNNCPUInfo& MNNInfo) {
                mDot = MNNInfo.dot;
                mI8mm = MNNInfo.i8mm;
                mSme2 = MNNInfo.sme2;
            }

            bool support() const {
                return mDot && (mI8mm || mSme2);
            }
        } CPUInfo;

        typedef struct kleidiaiInfo {
            size_t mKaiMstepGemv;
            size_t mKaiMstepGemm;
            size_t mKaiNStep;

            size_t mKaiMrGemv;
            size_t mKaiMrGemm;
            size_t mKaiNr;
            size_t mKaiKr;
            size_t mKaiSr;

            kleidiaiInfo() {
                mKaiMstepGemv = 0;
                mKaiMstepGemm = 0;
                mKaiNStep = 0;

                mKaiMrGemv = 0;
                mKaiMrGemm = 0;
                mKaiNr = 0;
                mKaiKr = 0;
                mKaiSr = 0;
            }

            void init(bool sme2) {
                if(sme2) {
                    mKaiMstepGemv = 1;
                    mKaiMstepGemm = kai_get_m_step_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa();
                    mKaiNStep = kai_get_n_step_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa();

                    mKaiMrGemv = 1;
                    mKaiMrGemm = kai_get_mr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa();
                    mKaiNr = kai_get_nr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa();
                    mKaiKr = 4;
                    mKaiSr = 1;
                } else {
                    mKaiMstepGemv = 1;
                    mKaiMstepGemm = 8;
                    mKaiNStep = 4;

                    mKaiMrGemv = 1;
                    mKaiMrGemm = 4;
                    mKaiNr = 4;
                    mKaiKr = 16;
                    mKaiSr = 2;
                }
            }
        } kleidiaiInfo;

        //Public static members.
        static bool mKaiInitialized;
        static accelType mAccelType;
        static CPUInfo mCPUInfo;
        static modelInfo mModelInfo;
        static kleidiaiInfo mKleidiaiInfo;
        static const std::map<modelInfo, accelType> mModelInfoMap;

        //Get instance.
        static KleidiAI &getInstance(const modelInfo& modelInfo, const MNNCPUInfo& gCPUInfo);
        static KleidiAI &getInstance();

        ~KleidiAI() {}

        //Check
        static bool canAccelerate() { return mKaiInitialized && mAccelType != accelType::NOT_SUPPORT && mCPUInfo.support() && mModelInfo.support(); }

        //Get info
        static size_t getMr(size_t m = 1) { return (m == 1) ? mKleidiaiInfo.mKaiMrGemv : mKleidiaiInfo.mKaiMrGemm; }
        static size_t getNr() { return mKleidiaiInfo.mKaiNr; }
        static size_t getKr() { return mKleidiaiInfo.mKaiKr; }
        static size_t getSr() { return mKleidiaiInfo.mKaiSr; }
        static size_t getMStep(size_t m = 1) { return (m == 1) ? mKleidiaiInfo.mKaiMstepGemv : mKleidiaiInfo.mKaiMstepGemm; }
        static size_t getNStep() { return mKleidiaiInfo.mKaiNStep; }
        size_t getVecNumPerThread(size_t totalVec, size_t totalThread, size_t minStep) { return kai_roundup((totalVec + totalThread - 1) / totalThread, minStep); }

        //Lhs
        size_t getLhsQuantedPackedSize(size_t m, size_t k);
        size_t getLhsQuantedPackedOffset(size_t m, size_t mIdx, size_t k);
        void runLhsQuantPack(size_t m, size_t k, size_t mr, const void* lhs, void* lhsQuantedPacked);

        //Rhs
        size_t getRhsPackedSize(size_t n, size_t k);
        size_t getRhsPackedOffset(size_t nIdx, size_t k);
        void runRhsPack(size_t n, size_t k, const void* rhs, const void* scale, const void* zeroPoint, const void *bias, void* rhsPacked, bool packedQ4);

        //Dst
        size_t getDstOffset(size_t mIdx, size_t nIdx, size_t n, size_t elementSize) { return (nIdx * elementSize) + mIdx * (n * elementSize); }

        //Matmul
        void runMatmul(size_t m, size_t n, size_t k, const void* lhsPacked, const void* rhsPacked, size_t dst_stride, void* dst);

    private:
        KleidiAI() {}

        static KleidiAI *mKaiInstance;
    };
}