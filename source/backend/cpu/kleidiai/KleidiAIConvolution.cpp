//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#ifdef MNN_KLEIDIAI_ENABLED
#include "KleidiAIConvolution.hpp"
#include <arm_neon.h>
#include <string.h>
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/CPURuntime.hpp"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"
#include "backend/cpu/CPUTensorConvert.hpp"

// KleidiAI micro-kernel headers (fp16 / fp32 SME2 matmul + packing).
#include "kai_common.h"
#include "kai_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme.h"
#include "kai_lhs_pack_f32p2vlx1_f32_sme.h"
#include "kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h"
#include "kai_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla.h"
#include "kai_lhs_pack_x16p2vlx2_x16_sme.h"
#include "kai_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme.h"
#include "kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa.h"
#include "kai_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot.h"

namespace MNN {

// ===================================================================
// Static gating + per-instance kernel parameter resolution / ukernel dispatch
// (moved out of the former KleidiAI class).

bool KleidiAIConvolution::isSupported(bool bFP16) {
    // Float matmul ukernels are only available on SME2.
    (void)bFP16;
    return MNNGetCPUInfo()->sme2;
}

void KleidiAIConvolution::configKernel() {
    mSme2 = MNNGetCPUInfo()->sme2;
    if (!mSme2) {
        return;
    }
    KernelParam& p = mParam;
    Ukernel& u = mUkernel;
    switch (mKernelType) {
        case KernelType::FP16:
            p.mKaiMstepGemm = kai_get_m_step_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa();
            p.mKaiMrGemm    = kai_get_mr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa();
            p.mKaiNStep     = kai_get_n_step_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa();
            p.mKaiNr        = kai_get_nr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa();
            p.mKaiKr        = kai_get_kr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa();
            p.mKaiSr        = kai_get_sr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa();
            mElementSize    = sizeof(__fp16);
            u.rhsPackedSize = kai_get_rhs_packed_size_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme;
            u.runRhsPack    = kai_run_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme;
            u.lhsPackedSize = kai_get_lhs_packed_size_lhs_pack_x16p2vlx2_x16_sme;
            u.runLhsPack    = kai_run_lhs_pack_x16p2vlx2_x16_sme;
            u.matmulGemm    = kai_run_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa;
            u.matmulGemv    = kai_run_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot;
            break;
        case KernelType::FP32:
            p.mKaiMstepGemm = kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
            p.mKaiMrGemm    = kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
            p.mKaiNStep     = kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
            p.mKaiNr        = kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
            p.mKaiKr        = kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
            p.mKaiSr        = kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
            mElementSize    = sizeof(float);
            u.rhsPackedSize = kai_get_rhs_packed_size_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme;
            u.runRhsPack    = kai_run_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme;
            u.lhsPackedSize = kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme;
            u.runLhsPack    = kai_run_lhs_pack_f32p2vlx1_f32_sme;
            u.matmulGemm    = kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa;
            u.matmulGemv    = kai_run_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla;
            break;
        default:
            break;
    }
}

size_t KleidiAIConvolution::getRhsPackedSize(size_t n, size_t k) const {
    return mUkernel.rhsPackedSize(n, k);
}

void KleidiAIConvolution::runRhsPack(size_t numGroups, size_t n, size_t k, size_t rhsStride,
                                     const void* rhs, const void* scale, const void* bias, void* rhsPacked) const {
    mUkernel.runRhsPack(numGroups, n, k, getNr(), getKr(), getSr(),
                        rhsStride, rhs, bias, scale, rhsPacked, 0, nullptr);
}

size_t KleidiAIConvolution::getLhsPackedSize(size_t m, size_t k) const {
    return mUkernel.lhsPackedSize(m, k, getMr(m), getKr(), getSr());
}

void KleidiAIConvolution::runLhsPack(size_t m, size_t k, const void* lhs, size_t lhsStride, void* lhsPacked) const {
    mUkernel.runLhsPack(m, k, getMr(m), getKr(), getSr(), 0, lhs, lhsStride, lhsPacked);
}

void KleidiAIConvolution::runMatmul(size_t m, size_t n, size_t k,
                                    const void* lhsPacked, const void* rhsPacked, void* dst,
                                    size_t dstStrideRow, size_t dstStrideCol,
                                    const float scalarMax, const float scalarMin) const {
    if (m == 1) {
        // GEMV path takes the (un-packed) lhs stride in bytes as an extra argument.
        mUkernel.matmulGemv(m, n, k, lhsPacked, k * mElementSize, rhsPacked, dst,
                            dstStrideRow, dstStrideCol, scalarMin, scalarMax);
    } else {
        mUkernel.matmulGemm(m, n, k, lhsPacked, rhsPacked, dst,
                            dstStrideRow, dstStrideCol, scalarMin, scalarMax);
    }
}


KleidiAIConvolution::KleidiAIConvolution(const Convolution2DCommon *common, Backend *b, const float *originWeight,
                                        size_t originWeightSize, const float *bias, size_t biasSize)
    : CPUConvolution(common, b) {

        auto outputCount = (int)biasSize;
        auto core = static_cast<CPUBackend*>(b)->functions();
        mResource.reset(new CPUConvolution::Resource);
        mResource->backend = b;
        auto mSrcCount   = (int)originWeightSize / outputCount;
        if (!mResource->copyBiasAlign(bias, (int)biasSize)) {
            MNN_ERROR("Not Enough Memory\n");
            mValid = false;
            return;
        }
        if (b->getRuntime()->hint().useCachedMmap > 1) {
            return;
        }

        if (core->bytes == 2) {
            AutoRelease<Tensor> tempTensor(Tensor::createDevice<float>({outputCount * mSrcCount}));
            mValid = b->onAcquireBuffer(tempTensor.get(), Backend::STATIC);
            if (!mValid) {
                MNN_ERROR("Not Enough Memory\n");
                return;
            }
            core->MNNFp32ToLowp(originWeight, tempTensor->host<int16_t>(), outputCount * mSrcCount);

            mKernelType = KernelType::FP16;
            configKernel();
            AutoRelease<Tensor> tempBiasTensor(Tensor::createDevice<float>({outputCount}));
            mValid = b->onAcquireBuffer(tempBiasTensor.get(), Backend::STATIC);
            if (!mValid) {
                b->onReleaseBuffer(tempTensor.get(), Backend::STATIC);
                MNN_ERROR("Not Enough Memory\n");
                return;
            }
            core->MNNFp32ToLowp(bias, tempBiasTensor->host<int16_t>(), outputCount);

            int packedSize = getRhsPackedSize(outputCount, mSrcCount);
            //Alloc packed weight tensor.
            mResource->mWeight.reset(Tensor::createDevice<int8_t>({packedSize}));
            bool success = b->onAcquireBuffer(mResource->mWeight.get(), Backend::STATIC);
            if (!success) {
                b->onReleaseBuffer(tempBiasTensor.get(), Backend::STATIC);
                b->onReleaseBuffer(tempTensor.get(), Backend::STATIC);
                MNN_ERROR("Out of static memory!\n");
                return;
            }

            //Run rhs pack.
            runRhsPack(1, outputCount, mSrcCount, mSrcCount * sizeof(__fp16),
                       tempTensor->host<void>(), nullptr, tempBiasTensor->host<void>(),
                       mResource->mWeight->host<void>());
            b->onReleaseBuffer(tempBiasTensor.get(), Backend::STATIC);
            b->onReleaseBuffer(tempTensor.get(), Backend::STATIC);
        } else {
            mKernelType = KernelType::FP32;
            configKernel();
            int packedSize = getRhsPackedSize(outputCount, mSrcCount);
            //Alloc packed weight tensor.
            mResource->mWeight.reset(Tensor::createDevice<int8_t>(std::vector<int>{packedSize}));
            mValid  = b->onAcquireBuffer(mResource->mWeight.get(), Backend::STATIC);
            if (!mValid) {
                MNN_ERROR("Out of static memory!\n");
                return;
            }

            //Run rhs pack.
            runRhsPack(1, outputCount, mSrcCount, mSrcCount * sizeof(float),
                       originWeight, nullptr, bias, mResource->mWeight->host<void>());
        }

}

KleidiAIConvolution::KleidiAIConvolution(std::shared_ptr<CPUConvolution::Resource> resource, const Convolution2DCommon *common, Backend* b) : CPUConvolution(common, b) {
    mResource = resource;
}

KleidiAIConvolution::~KleidiAIConvolution() {
    // Do nothing
}

bool KleidiAIConvolution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    auto exe = new KleidiAIConvolution(mResource, op->main_as_Convolution2D()->common(), bn);
    exe->mKernelType = this->mKernelType;
    exe->configKernel();
    *dst = exe;
    return true;
}

ErrorCode KleidiAIConvolution::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    CPUConvolution::onResize(inputs, outputs);
    auto core = static_cast<CPUBackend*>(backend())->functions();
    int bytes = core->bytes;
    auto input       = inputs[0];
    auto output      = outputs[0];
    auto inputDes       = TensorUtils::getDescribe(inputs[0]);
    auto outputDes      = TensorUtils::getDescribe(outputs[0]);
    auto ic = input->channel();
    auto oc = output->channel();
    auto batch       = input->batch();
    auto b = backend();

    auto inputOriginFmt = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
    auto outputOriginFmt = TensorUtils::getDescribe(outputs[0])->dimensionFormat;
    halide_type_t dataType = core->bytes == 2 ? halide_type_of<int16_t>() : halide_type_of<float>();
    if(inputOriginFmt != MNN_DATA_FORMAT_NHWC){
        mInputConvertBuffer.reset(Tensor::createDevice(std::vector<int>{input->batch(), input->height(), input->width(), input->channel()}, dataType, Tensor::DimensionType::TENSORFLOW));
        mValid = b->onAcquireBuffer(mInputConvertBuffer.get(), Backend::DYNAMIC);
        if (!mValid) {
            MNN_ERROR("Out of dynamic memory!\n");
            return OUT_OF_MEMORY;
        }
    }
    if (outputOriginFmt != MNN_DATA_FORMAT_NHWC){
        mOutputConvertBuffer.reset(Tensor::createDevice(std::vector<int>{output->batch(), output->height(), output->width(), output->channel()}, dataType, Tensor::DimensionType::TENSORFLOW));
        mValid = b->onAcquireBuffer(mOutputConvertBuffer.get(), Backend::DYNAMIC);
        if (!mValid) {
            MNN_ERROR("Out of dynamic memory!\n");
            return OUT_OF_MEMORY;
        }
    }

    auto m = batch * input->width() * input->height();
    if (m != 1) {
        int packedSize = getLhsPackedSize(m, ic);

        mInputResource.reset(Tensor::createDevice<float>({packedSize}));
        bool success = backend()->onAcquireBuffer(mInputResource.get(), Backend::DYNAMIC);
        if (!success) {
            MNN_ERROR("Out of dynamic memory!\n");
            return OUT_OF_MEMORY;
        }

        b->onReleaseBuffer(mInputResource.get(), Backend::DYNAMIC);
    }

    if(inputOriginFmt != MNN_DATA_FORMAT_NHWC){
        b->onReleaseBuffer(mInputConvertBuffer.get(), Backend::DYNAMIC);
    }
    if (outputOriginFmt != MNN_DATA_FORMAT_NHWC){
        b->onReleaseBuffer(mOutputConvertBuffer.get(), Backend::DYNAMIC);
    }

    mPostParameters = getPostParameters();
    return NO_ERROR;
}

ErrorCode KleidiAIConvolution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    auto core = static_cast<CPUBackend*>(backend())->functions();
    auto inputPtr = input->host<uint8_t>();
    auto weightPtr = mResource->mWeight->host<uint8_t>();
    int threadNum = static_cast<CPUBackend*>(backend())->threadNumber();

    const size_t m = input->batch() * input->width() * input->height(); //lhs vector number.
    const size_t n = output->channel(); //rhs vector number.
    const size_t k = input->channel(); //vector size.
    auto dst = output->host<uint8_t>();
    halide_type_t dataType = core->bytes == 2 ? halide_type_of<int16_t>() : halide_type_of<float>();
    size_t elementSize = core->bytes;
    auto b = backend();

    auto inputDes = TensorUtils::getDescribe(inputs[0]);
    if(inputDes->dimensionFormat != MNN_DATA_FORMAT_NHWC){
        MNN_CONCURRENCY_BEGIN(tId, threadNum) {
            CPUTensorConverter::convert(input, mInputConvertBuffer.get(), core, tId, threadNum);
        };
        MNN_CONCURRENCY_END();
        inputPtr = mInputConvertBuffer->host<uint8_t>();
    }
    auto lhsPacked = inputPtr;
    if(m != 1) {
        lhsPacked = mInputResource->host<uint8_t>();
        runLhsPack(m, k, inputPtr, k * elementSize, lhsPacked);
    }

    auto outputDes = TensorUtils::getDescribe(outputs[0]);
    auto outputPtr = output->host<uint8_t>();
    if(outputDes->dimensionFormat != MNN_DATA_FORMAT_NHWC){
        outputPtr = mOutputConvertBuffer->host<uint8_t>();
    }

    runMatmul(m, n, k, lhsPacked, weightPtr, outputPtr, n * elementSize, elementSize, mPostParameters[3], mPostParameters[2]);

    if(outputDes->dimensionFormat != MNN_DATA_FORMAT_NHWC){
        MNN_CONCURRENCY_BEGIN(tId, threadNum) {
            CPUTensorConverter::convert(mOutputConvertBuffer.get(), output, core, tId, threadNum);
        };
        MNN_CONCURRENCY_END();
    }

    return NO_ERROR;
}
} // namespace MNN
#endif //MNN_KLEIDIAI_ENABLED
