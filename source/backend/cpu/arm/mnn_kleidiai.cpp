//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if defined(__aarch64__)

#include "mnn_kleidiai.h"

using namespace MNN;

bool KleidiAI::mKaiInitialized = false;
KleidiAI *KleidiAI::mKaiInstance = NULL;
KleidiAI::StaticInfo KleidiAI::mStaticInfo;

//Get instance.
KleidiAI& KleidiAI::getInstance(const MNNCPUInfo& gCPUInfo) {
    if(!mKaiInstance) {
        mKaiInstance = new KleidiAI;
        mKaiInitialized = true;

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
        "QI4_ASYM_CHNLQT_F32",
        "QI4_ASYM_CHNLQT_F16",
        "QI4_ASYM_BLKQT_F32",
        "QI4_ASYM_BLKQT_F16",
        "QI4_SYM_CHNLQT_F32",
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
        MNN_PRINT("\nKleidiAI is running! AccelType is %s.\n", names[(size_t)type]);
    } else {
        MNN_PRINT("\nKleidiAI cannot accelerate! AccelType is %s.\n", names[(size_t)type]);
    }

}

//Init
void KleidiAI::initKernelInfo() {
    for (size_t type = 0; type < static_cast<size_t>(AccelType::ACC_TYPE_NUMBER); type++) {
        KernelInfo *pInfo = &mStaticInfo.mKernelInfo[type];
        KernelParam *pParam = &pInfo->mKernelParam;
        bool bSupport = false;

        switch(static_cast<AccelType>(type)) {
        case AccelType::QI4_SYM_CHNLQT_F32:
        {
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
            break;
        }
        case AccelType::QI4_ASYM_CHNLQT_F32:
        case AccelType::QI4_ASYM_CHNLQT_F16:
        case AccelType::QI4_ASYM_BLKQT_F32:
        case AccelType::QI4_ASYM_BLKQT_F16:
        {
            if(mStaticInfo.mDot && mStaticInfo.mI8mm) {
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
            break;
        }
        case AccelType::QI4_SYM_BLKQT:
        case AccelType::QI8_ASYM_CHNLQT:
        case AccelType::QI8_ASYM_BLKQT:
        case AccelType::QI8_SYM_CHNLQT:
        case AccelType::QI8_SYM_BLKQT:
            break;
        case AccelType::FP16:
        {
            if (mStaticInfo.mSme2) {
                bSupport = true;
                pParam->mKaiMstepGemm = kai_get_m_step_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa();
                pParam->mKaiMrGemm = kai_get_mr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa();
                pParam->mKaiNStep = kai_get_n_step_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa();
                pParam->mKaiNr = kai_get_nr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa();
                pParam->mKaiKr = kai_get_kr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa();
                pParam->mKaiSr = kai_get_sr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa();
            } else {
                bSupport = false;
            }
            break;
        }
        case AccelType::FP32:
        {
            if (mStaticInfo.mSme2) {
                bSupport = true;
                pParam->mKaiMstepGemm = kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
                pParam->mKaiMrGemm = kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
                pParam->mKaiNStep = kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
                pParam->mKaiNr = kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
                pParam->mKaiKr = kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
                pParam->mKaiSr = kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
            } else {
                bSupport = false;
            }
            break;
        }
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
KleidiAI::AccelType KleidiAI::getQIntAccelType(size_t bits, bool bAsymmetric, size_t blockSize, size_t bytes) {
    static std::map<KleidiAI::QIntInfo, KleidiAI::AccelType> infoMap = {
        {KleidiAI::QIntInfo(4, true,   0, 4),  KleidiAI::AccelType::QI4_ASYM_CHNLQT_F32},
        {KleidiAI::QIntInfo(4, true,  -1, 4),  KleidiAI::AccelType::QI4_ASYM_BLKQT_F32},
        {KleidiAI::QIntInfo(4, false,  0, 4),  KleidiAI::AccelType::QI4_SYM_CHNLQT_F32},
        {KleidiAI::QIntInfo(4, true,   0, 2),  KleidiAI::AccelType::QI4_ASYM_CHNLQT_F16},
        {KleidiAI::QIntInfo(4, true,  -1, 2),  KleidiAI::AccelType::QI4_ASYM_BLKQT_F16},
        {KleidiAI::QIntInfo(4, false, -1, -1),  KleidiAI::AccelType::QI4_SYM_BLKQT},
        {KleidiAI::QIntInfo(8, true,   0, -1),  KleidiAI::AccelType::QI8_ASYM_CHNLQT},
        {KleidiAI::QIntInfo(8, true,  -1, -1),  KleidiAI::AccelType::QI8_ASYM_BLKQT},
        {KleidiAI::QIntInfo(8, false,  0, -1),  KleidiAI::AccelType::QI8_SYM_CHNLQT},
        {KleidiAI::QIntInfo(8, false, -1, -1),  KleidiAI::AccelType::QI8_SYM_BLKQT},
    };

    QIntInfo info(bits, bAsymmetric, blockSize, bytes);
    auto it = infoMap.find(info);
    if(it != infoMap.end()) {
        return it->second;
    } else {
        return AccelType::ACC_TYPE_ERROR;
    }
}

//Lhs
size_t KleidiAI::getLhsPackedSize(AccelType type, size_t m, size_t k) {
    MNN_ASSERT(type >= AccelType::FLOAT && type <= AccelType::FLOAT_END);

    switch(type) {
    case AccelType::FP16:
        return kai_get_lhs_packed_size_lhs_pack_x16p2vlx2_x16_sme(m, k, getMr(type, m), getKr(type), getSr(type));
    case AccelType::FP32:
        return kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme(m, k, getMr(type, m), getKr(type), getSr(type));
    default:
        MNN_ASSERT(0);
    }
    return 0;
}

size_t KleidiAI::getLhsQuantedPackedSize(AccelType type, size_t m, size_t k, size_t bl) {
    MNN_ASSERT(type >= AccelType::QINT && type <= AccelType::QINT_END);

    switch(type) {
    case AccelType::QI4_SYM_CHNLQT_F32:
        return kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m, k, getMr(type, m), getKr(type), getSr(type));
    case AccelType::QI4_ASYM_CHNLQT_F32:
        bl = k;
    case AccelType::QI4_ASYM_BLKQT_F32:
        return kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32pscalef32_f32_neon(m, k, bl, getMr(type, m), getKr(type), getSr(type));
    case AccelType::QI4_ASYM_CHNLQT_F16:
        bl = k;
    case AccelType::QI4_ASYM_BLKQT_F16:
        return kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32pscalef32_f16_neon(m, k, bl, getMr(type, m), getKr(type), getSr(type));
    default:
        MNN_ASSERT(0);
    }

    return 0;
}

size_t KleidiAI::getLhsQuantedPackedOffset(AccelType type, size_t m, size_t mIdx, size_t k, size_t bl) {
    MNN_ASSERT(type >= AccelType::QINT && type <= AccelType::QINT_END);

    if(mIdx == 0) {
        return 0;
    }

    switch(type) {
    case AccelType::QI4_SYM_CHNLQT_F32:
        return kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(mIdx, k, getMr(type, m), getKr(type), getSr(type));
    case AccelType::QI4_ASYM_CHNLQT_F32:
        bl = k;
    case AccelType::QI4_ASYM_BLKQT_F32:
        return kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32pscalef32_f32_neon(mIdx, k, bl, getMr(type, m), getKr(type), getSr(type));
    case AccelType::QI4_ASYM_CHNLQT_F16:
        bl = k;
    case AccelType::QI4_ASYM_BLKQT_F16:
        return kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32pscalef32_f16_neon(mIdx, k, bl, getMr(type, m), getKr(type), getSr(type));
    default:
        MNN_ASSERT(0);
    }

    return 0;
}

void KleidiAI::runLhsPack(AccelType type, size_t m, size_t k, size_t mIdx, const void* lhs, size_t lhsStride, void* lhsPacked)
{
    MNN_ASSERT(type >= AccelType::FLOAT && type <= AccelType::FLOAT_END);
    KAI_UNUSED(mIdx);

    switch(type) {
    case AccelType::FP16:
        kai_run_lhs_pack_x16p2vlx2_x16_sme(m, k, getMr(type, m), getKr(type), getSr(type), 0, lhs, lhsStride, lhsPacked);
        break;
    case AccelType::FP32:
        kai_run_lhs_pack_f32p2vlx1_f32_sme(m, k, getMr(type, m), getKr(type), getSr(type), 0, lhs, lhsStride, lhsPacked);
        break;
    default:
        MNN_ASSERT(0);
    }
}

void KleidiAI::runLhsQuantPack(AccelType type, size_t m, size_t k, size_t bl, size_t mr, const void* lhs, void* lhsQuantedPacked) {
    MNN_ASSERT(type >= AccelType::QINT && type <= AccelType::QINT_END);

    switch(type) {
    case AccelType::QI4_SYM_CHNLQT_F32:
        kai_run_lhs_quant_pack_qai8dxp_f32(m, k, mr, getKr(type), getSr(type), 0, (const float *)lhs, k * sizeof(float), lhsQuantedPacked);
        break;
    case AccelType::QI4_ASYM_CHNLQT_F32:
        bl = k;
    case AccelType::QI4_ASYM_BLKQT_F32:
        kai_run_lhs_quant_pack_qsi8d32pscalef32_f32_neon(m, k, bl, mr, getKr(type), getSr(type), 0, (const float *)lhs, k * sizeof(float), lhsQuantedPacked);
        break;
    case AccelType::QI4_ASYM_CHNLQT_F16:
        bl = k;
    case AccelType::QI4_ASYM_BLKQT_F16:
        kai_run_lhs_quant_pack_qsi8d32pscalef32_f16_neon(m, k, bl, mr, getKr(type), getSr(type), 0, (const __fp16 *)lhs, k * sizeof(__fp16), lhsQuantedPacked);
        break;
    default:
        MNN_ASSERT(0);
    }
}

//Rhs
size_t KleidiAI::getRhsPackedSize(AccelType type, size_t n, size_t k, size_t bl) {
    switch(type) {
    case AccelType::QI4_SYM_CHNLQT_F32:
        if(mStaticInfo.mSme2) {
            return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon(n, k, getNr(type), getKr(type), getSr(type));
        } else {
            return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(n, k, getNr(type), getKr(type), getSr(type));
        }
    case AccelType::QI4_ASYM_CHNLQT_F32:
    case AccelType::QI4_ASYM_CHNLQT_F16:
        bl = k;
    case AccelType::QI4_ASYM_BLKQT_F32:
    case AccelType::QI4_ASYM_BLKQT_F16:
        return kai_get_rhs_packed_size_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon(n, k, getNr(type), getKr(type), bl);
    case AccelType::FP16:
        return kai_get_rhs_packed_size_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme(n, k);
    case AccelType::FP32:
        return kai_get_rhs_packed_size_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(n, k);
    default:
        MNN_ASSERT(0);
        return 0;
    }
}

size_t KleidiAI::getRhsPackedOffset(AccelType type, size_t nIdx, size_t k, size_t bl) {
    if(nIdx == 0) {
        return 0;
    }

    switch(type) {
    case AccelType::QI4_SYM_CHNLQT_F32:
        if(mStaticInfo.mSme2) {
            return kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon(nIdx, k, getNr(type), getKr(type), getSr(type));
        } else {
            return kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(nIdx, k, getNr(type), getKr(type), getSr(type));
        }
    case AccelType::QI4_ASYM_CHNLQT_F32:
    case AccelType::QI4_ASYM_CHNLQT_F16:
        bl = k;
    case AccelType::QI4_ASYM_BLKQT_F32:
    case AccelType::QI4_ASYM_BLKQT_F16:
        return kai_get_rhs_packed_offset_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon(nIdx, k, getNr(type), getKr(type), bl);
    default:
        MNN_ASSERT(0);
        return 0;
    }
}

void KleidiAI::runRhsPack(AccelType type, size_t numGroups, size_t n, size_t k, size_t bl, size_t rhsStride,
                          const void* rhs, const void* scale, const void* zeroPoint, const void* bias,
                          void* rhsPacked) {
    switch(type) {
    case AccelType::QI4_SYM_CHNLQT_F32:
    {
        KleidiAIUtil::rhsPackParamCommon paramCommon;
        if(mStaticInfo.mSme2) {
            KleidiAIUtil::packQsi4cxps1s0Qsu4cxs0s1(numGroups, n, k, getNr(type), getKr(type), getSr(type),
                                                        (const uint8_t *)rhs, (const float *)bias, (const float *)scale,
                                                        rhsPacked, 0, &paramCommon);
        } else {
            KleidiAIUtil::packQsi4cxps16s0Qs4cxs0s1(numGroups, n, k, getNr(type), getKr(type), getSr(type),
                                                        (const uint8_t *)rhs, (const float *)bias, (const float *)scale,
                                                        rhsPacked, 0, &paramCommon);
        }
        break;
    }
    case AccelType::QI4_ASYM_CHNLQT_F32:
    case AccelType::QI4_ASYM_CHNLQT_F16:
        bl = k;
    case AccelType::QI4_ASYM_BLKQT_F32:
    case AccelType::QI4_ASYM_BLKQT_F16:
        struct kai_rhs_pack_nxk_qai4c32p_params params;
        params.lhs_zero_point = 1;
        params.rhs_zero_point = 8;
        kai_run_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon(numGroups, n, k, getNr(type), getKr(type), getSr(type), bl,
                                                    (const uint8_t *)rhs, zeroPoint, bias, scale,
                                                    rhsPacked, 0, &params);
        break;
    case AccelType::FP16:
        kai_run_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme(numGroups, n, k, getNr(type), getKr(type), getSr(type),
                                                    rhsStride, rhs, bias, scale, rhsPacked, 0, nullptr);
        break;
    case AccelType::FP32:
        kai_run_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme(numGroups, n, k, getNr(type), getKr(type), getSr(type),
                                                          rhsStride, rhs, bias, scale, rhsPacked, 0, nullptr);
        break;
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

    switch (type) {
    case AccelType::QI4_SYM_CHNLQT_F32:
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
    case AccelType::QI4_ASYM_CHNLQT_F32:
        bl = k;
    case AccelType::QI4_ASYM_BLKQT_F32:
        if(m == 1) {
            kai_run_matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod(m, n, k, bl,
                                                                            (const void *)lhsPacked, (const void *)rhsPacked, (float *)dst,
                                                                            dstStrideRow, dstStrideCol, scalarMin, scalarMax);
        } else {
            kai_run_matmul_clamp_f32_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm(m, n, k, bl,
                                                                        (const void *)lhsPacked, (const void *)rhsPacked, (float *)dst,
                                                                        dstStrideRow, dstStrideCol, scalarMin, scalarMax);
        }
        break;
    case AccelType::QI4_ASYM_CHNLQT_F16:
        bl = k;
    case AccelType::QI4_ASYM_BLKQT_F16:
        if(m == 1) {
            kai_run_matmul_clamp_f16_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod(m, n, k, bl,
                                                                            (const void *)lhsPacked, (const void *)rhsPacked, (float *)dst,
                                                                            dstStrideRow, dstStrideCol, scalarMin, scalarMax);
        } else {
            kai_run_matmul_clamp_f16_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm(m, n, k, bl,
                                                                        (const void *)lhsPacked, (const void *)rhsPacked, (float *)dst,
                                                                        dstStrideRow, dstStrideCol, scalarMin, scalarMax);
        }
        break;
    case AccelType::FP16:
    {
        if (m == 1) {
            kai_run_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot(m, n, k, lhsPacked, k * sizeof(__fp16), rhsPacked, dst, dstStrideRow, dstStrideCol, scalarMin, scalarMax);
        } else {
            kai_run_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa(m, n, k, lhsPacked, rhsPacked, dst, dstStrideRow , dstStrideCol, scalarMin, scalarMax);
        }
        break;
    }
    case AccelType::FP32:
    {
        if (m == 1) {
            kai_run_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla(m, n, k, lhsPacked, k * sizeof(float), rhsPacked, dst, dstStrideRow, dstStrideCol, scalarMin, scalarMax);
        } else {
            kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(m, n, k, lhsPacked, rhsPacked, dst, dstStrideRow, dstStrideCol, scalarMin, scalarMax);
        }
        break;
    }
    default:
        MNN_ASSERT(0);
    }
}

#endif // defined(__aarch64__)