//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if defined(__aarch64__)

#include "mnn_kleidiai.h"

using namespace MNN;

#define FLT16_MAX 65504.0f
#define FLT16_MIN -65504.0f

bool KleidiAI::mKaiInitialized = false;
KleidiAI *KleidiAI::mKaiInstance = NULL;
KleidiAI::StaticInfo KleidiAI::mStaticInfo;

//Get instance.
KleidiAI& KleidiAI::getInstance(const MNNCPUInfo& gCPUInfo, bool bFP16, bool bBF16) {
    if(!mKaiInstance) {
        mKaiInstance = new KleidiAI;
        mKaiInitialized = true;

        mStaticInfo.mFP16 = bFP16;
        mStaticInfo.mBF16 = bBF16;
        mStaticInfo.mDot = gCPUInfo.dot;
        mStaticInfo.mI8mm = gCPUInfo.i8mm;
        mStaticInfo.mSme2 = gCPUInfo.sme2;

        initKernelInfo();
    }
    return *mKaiInstance;
}

KleidiAI& KleidiAI::getInstance() {
    if(!mKaiInstance) {
        MNN_ASSERT(0); //Should never happen.
    }
    return *mKaiInstance;
}

//Print
void KleidiAI::printInfo(AccelType type) {
    if(type == AccelType::ACC_TYPE_ERROR) {
        return;
    }

    static const char * const names[] = {
        "QI4_ASYM_CHNLQT",
        "QI4_ASYM_BLKQT",
        "QI4_SYM_CHNLQT",
        "QI4_SYM_BLKQT",
        "QI8_ASYM_CHNLQT",
        "QI8_ASYM_BLKQT",
        "QI8_SYM_CHNLQT",
        "QI8_SYM_BLKQT",
        "FP16",
        "FP32",
        "BF16",
    };

    KernelInfo *pInfo = &mStaticInfo.mKernelInfo[(size_t)type];
    if(pInfo->mKernelSupport) {
        MNN_PRINT("\nKleidiAI is running! AccelType is %s. ", names[(size_t)type]);
    } else {
        MNN_PRINT("\nKleidiAI cannot accelerate! AccelType is %s. ", names[(size_t)type]);
    }

    if(mStaticInfo.mFP16) {
        MNN_PRINT("Data type is FP16.\n");
    } else if(mStaticInfo.mBF16) {
        MNN_PRINT("Data type is BF16.\n");
    } else {
        MNN_PRINT("Data type is FP32.\n");
    }
}

//Init
void KleidiAI::initKernelInfo() {
    for(size_t type = 0; type < static_cast<size_t>(AccelType::ACC_TYPE_NUMBER); type++) {
        KernelInfo *pInfo = &mStaticInfo.mKernelInfo[type];
        bool bSupport = false;

        switch(static_cast<AccelType>(type)) {
        case AccelType::QI4_SYM_CHNLQT:
        {
            if(!mStaticInfo.mFP16 && !mStaticInfo.mBF16) { //Currently only support FP32.
                KernelParam *pParam = &pInfo->mKernelParam;
                if(mStaticInfo.mSme2) {
                    bSupport = true;
                    pParam->mKaiMstepGemv = 1;
                    pParam->mKaiMstepGemm = kai_get_m_step_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa();
                    pParam->mKaiNStep = kai_get_n_step_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa();
                    pParam->mKaiMrGemv = 1;
                    pParam->mKaiMrGemm = kai_get_mr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa();
                    pParam->mKaiNr = kai_get_nr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa();
                    pParam->mKaiKr = 4;
                    pParam->mKaiSr = 1;
                } else if(mStaticInfo.mDot && mStaticInfo.mI8mm) {
                    bSupport = true;
                    pParam->mKaiMstepGemv = 1;
                    pParam->mKaiMstepGemm = 8;
                    pParam->mKaiNStep = 4;
                    pParam->mKaiMrGemv = 1;
                    pParam->mKaiMrGemm = 4;
                    pParam->mKaiNr = 4;
                    pParam->mKaiKr = 16;
                    pParam->mKaiSr = 2;
                } else {
                    bSupport = false;
                }
            }
            break;
        }
        case AccelType::QI4_ASYM_CHNLQT:
        case AccelType::QI4_ASYM_BLKQT:
        case AccelType::QI4_SYM_BLKQT:
        case AccelType::QI8_ASYM_CHNLQT:
        case AccelType::QI8_ASYM_BLKQT:
        case AccelType::QI8_SYM_CHNLQT:
        case AccelType::QI8_SYM_BLKQT:
        case AccelType::FP16:
        case AccelType::FP32:
        case AccelType::BF16:
            break;
        default:
            MNN_ASSERT(0);
            break;
        }

        pInfo->mKernelSupport = bSupport;
    }
}

//Get Info
KleidiAI::AccelType KleidiAI::getQIntAccelType(size_t bits, bool bAsymmetric, size_t blockSize) {
    static std::map<KleidiAI::QIntInfo, KleidiAI::AccelType> infoMap = {
        {KleidiAI::QIntInfo(4, true,   0),  KleidiAI::AccelType::QI4_ASYM_CHNLQT},
        {KleidiAI::QIntInfo(4, true,  -1),  KleidiAI::AccelType::QI4_ASYM_BLKQT},
        {KleidiAI::QIntInfo(4, false,  0),  KleidiAI::AccelType::QI4_SYM_CHNLQT},
        {KleidiAI::QIntInfo(4, false, -1),  KleidiAI::AccelType::QI4_SYM_BLKQT},
        {KleidiAI::QIntInfo(8, true,   0),  KleidiAI::AccelType::QI8_ASYM_CHNLQT},
        {KleidiAI::QIntInfo(8, true,  -1),  KleidiAI::AccelType::QI8_ASYM_BLKQT},
        {KleidiAI::QIntInfo(8, false,  0),  KleidiAI::AccelType::QI8_SYM_CHNLQT},
        {KleidiAI::QIntInfo(8, false, -1),  KleidiAI::AccelType::QI8_SYM_BLKQT},
    };

    QIntInfo info(bits, bAsymmetric, blockSize);
    auto it = infoMap.find(info);
    if(it != infoMap.end()) {
        return it->second;
    } else {
        return AccelType::ACC_TYPE_ERROR;
    }
}

//Lhs
size_t KleidiAI::getLhsQuantedPackedSize(AccelType type, size_t m, size_t k, size_t bl) {
    MNN_ASSERT(type >= AccelType::QINT && type <= AccelType::QINT_END);
    KAI_UNUSED(bl);

    switch(type) {
    case AccelType::QI4_SYM_CHNLQT:
        return kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m, k, getMr(type, m), getKr(type), getSr(type));
    default:
        MNN_ASSERT(0);
    }

    return 0;
}

size_t KleidiAI::getLhsQuantedPackedOffset(AccelType type, size_t m, size_t mIdx, size_t k, size_t bl) {
    MNN_ASSERT(type >= AccelType::QINT && type <= AccelType::QINT_END);
    KAI_UNUSED(bl);

    if(mIdx == 0) {
        return 0;
    }

    switch(type) {
    case AccelType::QI4_SYM_CHNLQT:
        return kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(mIdx, k, getMr(type, m), getKr(type), getSr(type));
    default:
        MNN_ASSERT(0);
    }

    return 0;
}

void KleidiAI::runLhsPack(AccelType type, size_t m, size_t k, size_t mIdx, const void* lhs, size_t lhsStride, void* lhsPacked)
{
    MNN_ASSERT(type >= AccelType::FLOAT && type <= AccelType::FLOAT_END);
    //For float ukernels, Not support yet.
}

void KleidiAI::runLhsQuantPack(AccelType type, size_t m, size_t k, size_t bl, size_t mr, const void* lhs, void* lhsQuantedPacked) {
    MNN_ASSERT(type >= AccelType::QINT && type <= AccelType::QINT_END);
    KAI_UNUSED(bl);

    switch(type) {
    case AccelType::QI4_SYM_CHNLQT:
        kai_run_lhs_quant_pack_qai8dxp_f32(m, k, mr, getKr(type), getSr(type), 0, (const float *)lhs, k * sizeof(float), lhsQuantedPacked);
        break;
    default:
        MNN_ASSERT(0);
    }
}

//Rhs
size_t KleidiAI::getRhsPackedSize(AccelType type, size_t n, size_t k, size_t bl) {
    KAI_UNUSED(bl);
    
    switch(type) {
    case AccelType::QI4_SYM_CHNLQT:
        if(mStaticInfo.mSme2) {
            return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon(n, k, getNr(type), getKr(type), getSr(type));
        } else {
            return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(n, k, getNr(type), getKr(type), getSr(type));
        }
    default:
        MNN_ASSERT(0);
        return 0;
    }
}

size_t KleidiAI::getRhsPackedOffset(AccelType type, size_t nIdx, size_t k, size_t bl) {
    KAI_UNUSED(bl);

    if(nIdx == 0) {
        return 0;
    }

    switch(type) {
    case AccelType::QI4_SYM_CHNLQT:
        if(mStaticInfo.mSme2) {
            return kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon(nIdx, k, getNr(type), getKr(type), getSr(type));
        } else {
            return kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(nIdx, k, getNr(type), getKr(type), getSr(type));
        }
    default:
        MNN_ASSERT(0);
        return 0;
    }
}

void KleidiAI::runRhsPack(AccelType type, size_t numGroups, size_t n, size_t k, size_t bl, size_t rhsStride,
                          const void* rhs, const void* scale, const void* zeroPoint, const void* bias,
                          void* rhsPacked, bool packedQ4) {
    KAI_UNUSED(bl);

    switch(type) {
    case AccelType::QI4_SYM_CHNLQT:
    {
        KleidiAIUtil::rhsPackParamCommon paramCommon;
        if(mStaticInfo.mSme2) {
            if(packedQ4) {
                KleidiAIUtil::packQsi4cxps1s0Qsu4cxs0s1(numGroups, n, k, getNr(type), getKr(type), getSr(type),
                                                        (const uint8_t *)rhs, (const float *)bias, (const float *)scale,
                                                        rhsPacked, 0, &paramCommon);
            } else {
                MNN_ASSERT(0);
            }
        } else {
            if(packedQ4) {
                KleidiAIUtil::packQsi4cxps16s0Qs4cxs0s1(numGroups, n, k, getNr(type), getKr(type), getSr(type),
                                                        (const uint8_t *)rhs, (const float *)bias, (const float *)scale,
                                                        rhsPacked, 0, &paramCommon);
            } else {
                KleidiAIUtil::packQsi4cxps16s0Qs4cx(numGroups, n, k, getNr(type), getKr(type), getSr(type),
                                                    (const uint8_t *)rhs, (const float *)bias, (const float *)scale,
                                                    rhsPacked, 0, &paramCommon);
            }
        }
        break;
    }
    default:
        MNN_ASSERT(0);
    }
}

//Matmul
void KleidiAI::runMatmul(AccelType type, size_t m, size_t n, size_t k, size_t bl,
                         const void* lhsPacked, const void* rhsPacked, void* dst,
                         size_t dstStrideRow, size_t dstStrideCol,
                         const float scalarMax, const float scalarMin) {
    KAI_UNUSED(bl);

    switch(type) {
    case AccelType::QI4_SYM_CHNLQT:
    {
        if(mStaticInfo.mSme2) {
            if(m == 1) {
                kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot(m, n, k,
                                                                                 (const void *)lhsPacked, (const void *)rhsPacked, (float *)dst,
                                                                                 dstStrideRow, dstStrideCol, scalarMin, scalarMax);
            } else {
                kai_run_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa(m, n, k,
                                                                                     (const void *)lhsPacked, (const void *)rhsPacked, (float *)dst,
                                                                                     dstStrideRow, dstStrideCol, scalarMin, scalarMax);
            }
        } else {
            if(m == 1) {
                kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod(m, n, k,
                                                                                   (const void *)lhsPacked, (const void *)rhsPacked, (float *)dst,
                                                                                   dstStrideRow, dstStrideCol, scalarMin, scalarMax);
            } else {
                kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm(m, n, k,
                                                                                (const void *)lhsPacked, (const void *)rhsPacked, (float *)dst,
                                                                                dstStrideRow, dstStrideCol, scalarMin, scalarMax);
            }
        }

        break;
    }
    default:
        MNN_ASSERT(0);
    }
}

#endif // defined(__aarch64__)