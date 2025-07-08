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

#define FLT16_MAX 65504.0f
#define FLT16_MIN -65504.0f

namespace MNN {
    class KleidiAI {
    public:
        // ===================================================================
        // Enum definition

        enum class AccelType {
            /*
            ASYM/SYM: Asymmetric/symmetric;
            CHNLQT/BLKQT: channel wise/block wise;
            */
            QINT = 0,
            QI4_ASYM_CHNLQT_F32 = QINT,
            QI4_ASYM_CHNLQT_F16,
            QI4_ASYM_BLKQT_F32,
            QI4_ASYM_BLKQT_F16,
            QI4_SYM_CHNLQT_F32,
            QI4_SYM_BLKQT,
            QI8_ASYM_CHNLQT,
            QI8_ASYM_BLKQT,
            QI8_SYM_CHNLQT,
            QI8_SYM_BLKQT,
            QINT_END = QI8_SYM_BLKQT,

            FLOAT,
            FP16 = FLOAT,
            FP32,
            BF16,
            FLOAT_END = BF16,

            ACC_TYPE_NUMBER,
            ACC_TYPE_ERROR = ACC_TYPE_NUMBER
        };

        // ===================================================================
        // Some necessary data structures
        typedef struct KernelParam {
            size_t mKaiMstepGemv = 0;
            size_t mKaiMstepGemm = 0;
            size_t mKaiNStep = 0;

            size_t mKaiMrGemv = 0;
            size_t mKaiMrGemm = 0;
            size_t mKaiNr = 0;
            size_t mKaiKr = 0;
            size_t mKaiSr = 0;
        } KernelParam;

        typedef struct KernelInfo {
            bool mKernelSupport = false;
            KernelParam mKernelParam;
        } KernelInfo;

        typedef struct StaticInfo {
            bool mDot = false;
            bool mI8mm = false;
            bool mSme2 = false;

            KernelInfo mKernelInfo[(size_t)AccelType::ACC_TYPE_NUMBER];
        } StaticInfo;


        typedef struct QIntInfo {
            size_t mBits;
            bool mAsymmetric; //Asymmetric quantized model.
            size_t mBlockSize; //0: Per channel quant; others: Per block quant.
            size_t mBytes; //4: float32; 2: float16.

            QIntInfo(size_t bits = 4, bool asymmetric = false, size_t blockSize = 0, size_t bytes = 0) {
                mBits = bits;
                mAsymmetric = asymmetric;
                mBlockSize = blockSize;
                mBytes = bytes;
            }

            bool operator<(const QIntInfo& rhs) const {
                if(mBits != rhs.mBits) {
                    return mBits < rhs.mBits;
                }

                if(mAsymmetric != rhs.mAsymmetric) {
                    return mAsymmetric < rhs.mAsymmetric;
                }

                if(mBytes != rhs.mBytes) {
                    return mBytes < rhs.mBytes;
                }

                bool lhsPerChannel = mBlockSize == 0 ? true : false;
                bool rhsPerChannel = rhs.mBlockSize == 0 ? true : false;
                return lhsPerChannel < rhsPerChannel;
            }
        } QIntInfo;

        // ===================================================================

        //Public static members.
        static bool mKaiInitialized;

        //Get instance.
        static KleidiAI &getInstance(const MNNCPUInfo& gCPUInfo);
        static KleidiAI &getInstance();
        static void initKernelInfo();

        ~KleidiAI() {}

        void printInfo(AccelType type);

        //Check and set
        bool canAccelerate();
        bool canAccelerate(AccelType type);
        bool canAccelerate(AccelType type, const Convolution2DCommon *common);
        bool isLoaded(AccelType type);
        void setLoaded(AccelType type) { mLoaded[(size_t)type] = true; }

        //Get info
        static AccelType getQIntAccelType(size_t bits, bool bAsymmetric, size_t blockSize, size_t bytes);
        size_t getMr(AccelType type, size_t m = 1);
        size_t getNr(AccelType type);
        size_t getKr(AccelType type);
        size_t getSr(AccelType type);
        size_t getMStep(AccelType type, size_t m = 1);
        size_t getNStep(AccelType type);
        size_t getVecNumPerThread(size_t totalVec, size_t totalThread, size_t minStep);
        //Get Static info
        bool bSupportSme2() { return mStaticInfo.mSme2; }

        //Lhs
        size_t getLhsPackedSize(AccelType type, size_t m, size_t k);
        size_t getLhsQuantedPackedSize(AccelType type, size_t m, size_t k, size_t bl);
        size_t getLhsQuantedPackedOffset(AccelType type, size_t m, size_t mIdx, size_t k, size_t bl);
        void runLhsPack(AccelType type, size_t m, size_t k, size_t mIdx, const void* lhs, size_t lhsStride, void* lhsPacked);
        void runLhsQuantPack(AccelType type, size_t m, size_t k, size_t bl, size_t mr, const void* lhs, void* lhsQuantedPacked);

        //Rhs
        size_t getRhsPackedSize(AccelType type, size_t n, size_t k, size_t bl);
        size_t getRhsPackedOffset(AccelType type, size_t nIdx, size_t k, size_t bl);
        void runRhsPack(AccelType type, size_t numGroups, size_t n, size_t k, size_t bl, size_t rhsStride,
                        const void* rhs, const void* scale, const void* zeroPoint, const void* bias,
                        void* rhsPacked);

        //Dst
        size_t getDstOffset(size_t mIdx, size_t nIdx, size_t n, size_t elementSize) { return (nIdx * elementSize) + mIdx * (n * elementSize); }

        //Matmul
        void runMatmul(AccelType type, size_t m, size_t n, size_t k, size_t bl,
                       const void* lhsPacked, const void* rhsPacked, void* dst,
                       size_t dstStrideRow, size_t dstStrideCol,
                       const float scalarMax, const float scalarMin);

    private:
        KleidiAI() {}

        static KleidiAI *mKaiInstance;
        //Static info, never change after construct.
        static StaticInfo mStaticInfo;
        //Status, will change while pipeline is running. 
        bool mLoaded[(size_t)AccelType::ACC_TYPE_NUMBER] = { false };
        bool mLinear = false; //All pipeline format has been set as NCHW.
    };

    // ===================================================================
    // Inline functions
    inline bool KleidiAI::canAccelerate() {
        for(size_t type = 0; type < (size_t)AccelType::ACC_TYPE_NUMBER; type++) {
            if(mStaticInfo.mKernelInfo[(size_t)type].mKernelSupport && isLoaded(static_cast<AccelType>(type))) {
                return true;
            }
        }
        return false;
    }

    inline bool KleidiAI::canAccelerate(AccelType type) {
        if(type >= AccelType::ACC_TYPE_ERROR) {
            return false;
        }
        return mStaticInfo.mKernelInfo[(size_t)type].mKernelSupport;
    }

    inline bool KleidiAI::canAccelerate(AccelType type, const Convolution2DCommon* common) {
        if(type >= AccelType::ACC_TYPE_ERROR) {
            return false;
        }
        if(common->group() != 1) {
            return false;
        }
        if(type == AccelType::QI4_ASYM_CHNLQT_F32|| type == AccelType::QI4_ASYM_CHNLQT_F16 || type == AccelType::QI8_ASYM_CHNLQT) {
            if(common->inputCount() % 32 != 0) {
                return false;
            }
        }
        if(common->kernelX() == 1 && common->kernelY() == 1
            && common->padX() == 0 && common->padY() == 0
            && common->strideX() == 1 && common->strideY() == 1
            && common->dilateX() == 1 && common->dilateY() == 1) {
            return mStaticInfo.mKernelInfo[(size_t)type].mKernelSupport;
        }
        return false;
    }

    inline bool KleidiAI::isLoaded(AccelType type) {
        MNN_ASSERT(type < AccelType::ACC_TYPE_NUMBER);
        return mLoaded[(size_t)type];
    }

    inline size_t KleidiAI::getMr(AccelType type, size_t m) {
        KernelParam *pParam = &mStaticInfo.mKernelInfo[(size_t)type].mKernelParam;
        return (m == 1) ? pParam->mKaiMrGemv : pParam->mKaiMrGemm;
    }

    inline size_t KleidiAI::getNr(AccelType type) {
        KernelParam *pParam = &mStaticInfo.mKernelInfo[(size_t)type].mKernelParam;
        return pParam->mKaiNr;
    }

    inline size_t KleidiAI::getKr(AccelType type) {
        KernelParam *pParam = &mStaticInfo.mKernelInfo[(size_t)type].mKernelParam;
        return pParam->mKaiKr;
    }

    inline size_t KleidiAI::getSr(AccelType type) {
        KernelParam *pParam = &mStaticInfo.mKernelInfo[(size_t)type].mKernelParam;
        return pParam->mKaiSr;
    }

    inline size_t KleidiAI::getMStep(AccelType type, size_t m) {
        KernelParam *pParam = &mStaticInfo.mKernelInfo[(size_t)type].mKernelParam;
        return (m == 1) ? pParam->mKaiMstepGemv : pParam->mKaiMstepGemm;
    }

    inline size_t KleidiAI::getNStep(AccelType type) {
        KernelParam *pParam = &mStaticInfo.mKernelInfo[(size_t)type].mKernelParam;
        return pParam->mKaiNStep;
    }

    inline size_t KleidiAI::getVecNumPerThread(size_t totalVec, size_t totalThread, size_t minStep) {
        return kai_roundup((totalVec + totalThread - 1) / totalThread, minStep);
    }
}