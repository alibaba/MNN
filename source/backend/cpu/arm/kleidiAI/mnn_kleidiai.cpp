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

KleidiAI *KleidiAI::mKaiInstance = NULL;
bool KleidiAI::mKaiInitialized = false;
KleidiAI::modelInfo KleidiAI::mModelInfo;
KleidiAI::accelType KleidiAI::mAccelType = KleidiAI::accelType::NOT_SUPPORT;
KleidiAI::CPUInfo KleidiAI::mCPUInfo;
KleidiAI::kleidiaiInfo KleidiAI::mKleidiaiInfo;

const std::map<KleidiAI::modelInfo, KleidiAI::accelType> KleidiAI::mModelInfoMap = {
                        /*qi4,  asym,  fp16,   blkSize*/
    {KleidiAI::modelInfo(true,  false, false,  0),       KleidiAI::accelType::QI4_SYM_FP32_CHNLQT},
    //TODO: KleidiAI support.
    // {KleidiAI::modelInfo(true,  true,  false,  0),       KleidiAI::accelType::QI4_ASYM_FP32_CHNLQT},
    // {KleidiAI::modelInfo(true,  true,  false, -1),       KleidiAI::accelType::QI4_ASYM_FP32_BLKQT},
    // {KleidiAI::modelInfo(true,  true,  true,   0),       KleidiAI::accelType::QI4_ASYM_FP16_CHNLQT},
    // {KleidiAI::modelInfo(true,  true,  true,  -1),       KleidiAI::accelType::QI4_ASYM_FP16_BLKQT},
    // {KleidiAI::modelInfo(true,  false, false, -1),       KleidiAI::accelType::QI4_SYM_FP32_BLKQT},
    // {KleidiAI::modelInfo(true,  false, true,   0),       KleidiAI::accelType::QI4_SYM_FP16_CHNLQT},
    // {KleidiAI::modelInfo(true,  false, true,  -1),       KleidiAI::accelType::QI4_SYM_FP16_BLKQT},
};

//Get instance.
KleidiAI& KleidiAI::getInstance(const modelInfo& modelInfo, const MNNCPUInfo& gCPUInfo) {
    if(!mKaiInstance) {
        //Set mKaiInitialized and construct.
        mKaiInstance = new KleidiAI;
        mKaiInitialized = true;

        //Set model info.
        mModelInfo = modelInfo;
        //Set mAccelType.
        auto it = mModelInfoMap.find(mModelInfo);
        if(it != mModelInfoMap.end()) {
            mAccelType = it->second;
        } else {
            mAccelType = accelType::NOT_SUPPORT;
        }
        mModelInfo.print();

        //Set CPU info
        mCPUInfo = gCPUInfo;
        
        if(canAccelerate()) {
            MNN_PRINT("KleidiAI is running!\n");
            //Init Kleidi info related to model type.
            mKleidiaiInfo.init(mCPUInfo.mSme2);
        } else {
            MNN_PRINT("KleidiAI cannot accelerate!\n");
        }
    }
    return *mKaiInstance;
}

KleidiAI& KleidiAI::getInstance() {
    if(!mKaiInstance) {
        MNN_ASSERT(0); //Should never happen.
    }
    return *mKaiInstance;
}

//Lhs
size_t KleidiAI::getLhsQuantedPackedSize(size_t m, size_t k) {
    switch(mAccelType) {
        case accelType::QI4_SYM_FP32_CHNLQT:
            return kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m, k, getMr(m), getKr(), getSr());
        default:
            MNN_ASSERT(0);
            return 0;
    }
}

size_t KleidiAI::getLhsQuantedPackedOffset(size_t m, size_t mIdx, size_t k) {
    if(mIdx == 0) {
        return 0;
    }

    switch(mAccelType) {
        case accelType::QI4_SYM_FP32_CHNLQT:
            return kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(mIdx, k, getMr(m), getKr(), getSr());
        default:
            MNN_ASSERT(0);
            return 0;
    }
}

void KleidiAI::runLhsQuantPack(size_t m, size_t k, size_t mr, const void* lhs, void* lhsQuantedPacked) {
    void (*pack)(size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start,
                 const float* lhs, size_t lhs_stride, void* lhs_packed) = NULL;

    switch(mAccelType) {
        case accelType::QI4_SYM_FP32_CHNLQT:
            pack = kai_run_lhs_quant_pack_qai8dxp_f32;
            break;
        default:
            MNN_ASSERT(0);
            break;
    }

    if(pack) {
        pack(m, k, mr, getKr(), getSr(), 0, (const float *)lhs, k * sizeof(float), lhsQuantedPacked);
    }
}

//Rhs
size_t KleidiAI::getRhsPackedSize(size_t n, size_t k) {
    switch(mAccelType) {
        case accelType::QI4_SYM_FP32_CHNLQT:
            if(mCPUInfo.mSme2) {
                return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxpo_qsu4cxs1s0(n, k, getNr(), getKr(), getSr());
            } else {
                return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(n, k, getNr(), getKr(), getSr());
            }
        default:
            MNN_ASSERT(0);
            return 0;
    }
}

size_t KleidiAI::getRhsPackedOffset(size_t nIdx, size_t k) {
    if(nIdx == 0) {
        return 0;
    }

    switch(mAccelType) {
        case accelType::QI4_SYM_FP32_CHNLQT:
            if(mCPUInfo.mSme2) {
                return kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxpo_qsu4cxs1s0(nIdx, k, getNr(), getKr(), getSr());
            } else {
                return kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(nIdx, k, getNr(), getKr(), getSr());
            }
        default:
            MNN_ASSERT(0);
            return 0;
    }
}

void KleidiAI::runRhsPack(size_t n, size_t k, const void* rhs, const void* scale, const void* zeroPoint, const void* bias, void* rhsPacked, bool packedQ4) {
    void (*packSymChnl)(size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, const uint8_t* rhs, const float* bias,
                         const float* scale, void* rhs_packed, size_t extra_bytes,
                         const struct KleidiAIUtil::rhsPackParamCommon* params) = NULL;

    switch(mAccelType) {
    case accelType::QI4_SYM_FP32_CHNLQT:
    {
        if(mCPUInfo.mSme2) {
            if(packedQ4) {
                packSymChnl = KleidiAIUtil::packQsi4cxpoQsu4cxs0s1;
            } else {
                MNN_ASSERT(0);
            }
        } else {
            if(packedQ4) {
                packSymChnl = KleidiAIUtil::packQsi4cxps16s0Qs4cxs0s1;
            } else {
                packSymChnl = KleidiAIUtil::packQsi4cxps16s0Qs4cx;
            }
        }
        break;
    }
    default:
        MNN_ASSERT(0);
    }

    KleidiAIUtil::rhsPackParamCommon paramCommon;
    if(packSymChnl) {
        packSymChnl(1, n, k, getNr(), getKr(), getSr(),
                    (const uint8_t *)rhs, (const float *)bias, (const float *)scale,
                    rhsPacked, 0, &paramCommon);
    }
}

//Matmul
void KleidiAI::runMatmul(size_t m, size_t n, size_t k, const void* lhsPacked, const void* rhsPacked, size_t dst_stride, void* dst) {
    void (*runChnlQuantMatmul)(size_t m, size_t n, size_t k, const void* lhs_packed, const void* rhs_packed, float* dst,
                               size_t dst_stride_row, size_t dst_stride_col, float scalar_min, float scalar_max) = NULL;

    const float scalar_max = mModelInfo.mFp16 ? FLT16_MAX : FLT_MAX;
    const float scalar_min = -scalar_max;

    switch(mAccelType) {
    case accelType::QI4_SYM_FP32_CHNLQT: 
        if(m == 1) {
            if(mCPUInfo.mSme2) {
                runChnlQuantMatmul = kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4cxpo4vlx4_1x4vl_sme2_sdot;
            } else {
                runChnlQuantMatmul = kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod;
            }
        } else {
            if(mCPUInfo.mSme2) {
                runChnlQuantMatmul = kai_run_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa;
            } else {
                runChnlQuantMatmul = kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm;
            }
        }
        break;
    default:
        MNN_ASSERT(0);
    }

    if(runChnlQuantMatmul) {
        runChnlQuantMatmul(m, n, k, (const void *)lhsPacked, (const void *)rhsPacked, (float *)dst, dst_stride, sizeof(float), scalar_min, scalar_max);
    }
}

#endif // defined(__aarch64__)