//
//  ConvolutionHybrid.cpp
//  MNN
//
//  Created by MNN on 2023/10/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvolutionHybrid.hpp"
#include <string.h>
#include "core/BufferAllocator.hpp"
#include "backend/cpu/CPUBackend.hpp"
#include "core/Concurrency.h"
#include "ConvOpt.h"
#include "core/Macro.h"
#include "CommonOptFunction.h"
#include "core/TensorUtils.hpp"
#include <math.h>
#include "backend/cpu/compute/DenseConvolutionTiledExecutor.hpp"

namespace MNN {

bool ConvolutionHybrid::initQuantizeResource(std::shared_ptr<ConvolutionCommon::Int8Common> int8Info, std::shared_ptr<CPUConvolution::Resource> resource, int hU, int hP, int lU, int lP, int outputCount, int srcChannel, int kernelSize, int bytes) {
    int weightLength = hU * lU * hP * lP;
    resource->mWeight.reset(Tensor::createDevice<uint8_t>(
        {weightLength}));
    auto res = resource->backend->onAcquireBuffer(resource->mWeight.get(), Backend::STATIC);
    if (!res) {
        return false;
    }
    resource->mDequantize.bits = 8;
    resource->hU = hU;
    resource->lU = lU;
    resource->hP = hP;
    resource->lP = lP;

    // Save scale bias
    resource->mDequantize.mScaleBias.reset(MNN::Tensor::createDevice<float>({hU * hP * 2}));
    res = resource->backend->onAcquireBuffer(resource->mDequantize.mScaleBias.get(), Backend::STATIC);
    if (!res) {
        return false;
    }
    auto alphaPtr = resource->mDequantize.mScaleBias->host<float>();
    auto biasPtr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(alphaPtr) + hU * hP * bytes);
    ::memset(alphaPtr, 0, 2 * hU * hP * bytes);
    int h = int8Info->alpha.size();
    if (int8Info->canUseInt4 && int8Info->asymmetric) {
        // int4 to uint4, -8 offset merge to bias
        for (int i = 0; i < h/2; ++i) {
            int8Info->alpha.get()[2 * i] -= 8 * int8Info->alpha.get()[2 * i + 1];
        }
    }
    if (bytes == 2) {
        auto core = static_cast<CPUBackend*>(resource->backend)->functions();
        if (int8Info->asymmetric) {
            std::unique_ptr<int16_t[]> tmp(new int16_t[h]);
            core->MNNFp32ToLowp(int8Info->alpha.get(), tmp.get(), h);
            for (int i=0; i< h/2; ++i) {
                reinterpret_cast<int16_t*>(alphaPtr)[i] = tmp[2 * i + 1];
                reinterpret_cast<int16_t*>(biasPtr)[i] = tmp[2 * i];
            }
        } else {
            core->MNNFp32ToLowp(int8Info->alpha.get(), reinterpret_cast<int16_t*>(alphaPtr), h);
            if (int8Info->canUseInt4) {
                for (int i = 0; i < h; ++i) {
                    int8Info->alpha.get()[i] *= -8.0;
                }
                core->MNNFp32ToLowp(int8Info->alpha.get(), reinterpret_cast<int16_t*>(biasPtr), h);
            }
        }
    } else {
        if (int8Info->asymmetric) {
            h = h / 2;
            for (int i=0; i<h; ++i) {
                alphaPtr[i] = int8Info->alpha.get()[2 * i + 1];
                biasPtr[i] = int8Info->alpha.get()[2 * i];
            }
        } else {
            for (int i=0; i<h; ++i) {
                alphaPtr[i] = int8Info->alpha.get()[i];
                if (int8Info->canUseInt4) {
                    biasPtr[i] = -8.0 * int8Info->alpha.get()[i];
                } else {
                    biasPtr[i] = 0.f;
                }
            }
        }
    }
    std::vector<int8_t> data(weightLength, 0);
    auto srcWInt8 = int8Info->weight.get();
    if (hP * hU != outputCount || lP * lU != srcChannel) {
        int packedic = lU * lP;
        for (int i = 0; i < outputCount; ++i) {
            for (int j = 0; j < srcChannel; ++j) {
                int destIdx = i * packedic + j;
                int srcIdx = i * srcChannel + j;
                data[destIdx] = srcWInt8[srcIdx];
            }
        }
        srcWInt8 = data.data();
    }
    if (int8Info->canUseInt4) {
        MNN_ASSERT(weightLength % 2 == 0);
        weightLength = UP_DIV(weightLength, 2);
        resource->mDequantize.bits = 4;

        auto srcPtr = int8Info->weight.get();
        auto dstPtr = resource->mWeight->host<uint8_t>();
        // oc, ic -> oc/hP, ic/lP, hP, lP
        if (hP == 8 && lP == 8) {
            for (int i = 0; i < hU; i++) {
                for (int j = 0; j < lU; j++) {
                    for (int k = 0; k < 2; k++) {
                        for (int n = 0; n < 16; n++) {
                            int hp_idx = n / 8;
                            int lp_idx = n % 8;
                            int s0 = srcWInt8[(i * hP + k * 4 + hp_idx) * lP *lU + (j * lP + lp_idx)];
                            int s1 = srcWInt8[(i * hP + k * 4 + hp_idx + 2) * lP * lU + (j * lP + lp_idx)];
                            int d = (s0 + 8) * 16 + (s1 + 8);
                            dstPtr[(i * lU * lP * hP + j * hP * lP + k * 32) / 2 + n] = (uint8_t)d;
                        }
                    }
                }
            }
        } else {
            for (int i = 0; i < hU; i++) {
                for (int j = 0; j < lU; j++) {
                    for (int k = 0; k < hP; k++) {
                        for (int l = 0; l < lP; l+=2) {
                            int s0 = srcWInt8[(i * hP + k) * lP * lU + (j * lP + l)];
                            int s1 = srcWInt8[(i * hP + k) * lP * lU + (j * lP + l + 1)];
                            int d = (s0 + 8) * 16 + (s1 + 8);
                            dstPtr[(i * lU * lP * hP + j * hP * lP + k * lP + l) / 2] = d;
                        }
                    }
                }
            }
        }
    } else {
        // Reorder weight for int8
        auto dstWInt8 = resource->mWeight->host<int8_t>();
        // oc, ic -> oc/hP, ic/lP, hP, lP
        for (int i = 0; i < hU; i++) {
            for (int j = 0; j < lU; j++) {
                for (int k = 0; k < hP; k++) {
                    for (int l = 0; l < lP; l++) {
                        dstWInt8[i * lU * lP * hP + j * hP * lP + k * lP + l] = srcWInt8[(i * hP + k) * lP * lU + (j * lP + l)];
                    }
                }
            }
        }
    }
    return true;
}

ConvolutionHybrid::ConvolutionHybrid(const Convolution2DCommon *common, Backend *b, const float *originWeight,
                                     size_t originWeightSize, const float *bias, size_t biasSize, std::shared_ptr<ConvolutionCommon::Int8Common> quantInfo)
    : CPUConvolution(common, b) {
    mResource.reset(new CPUConvolution::Resource);
    mResource->backend = b;
    if (!mResource->copyBiasAlign(bias, (int)biasSize)) {
        MNN_ERROR("Not Enough Memory\n");
        mValid = false;
        return;
    }
    MNN_ASSERT(nullptr != quantInfo.get());
    originWeightSize = quantInfo->weight.size();
    auto outputCount = (int)biasSize;
    int inputCount = (int)originWeightSize / (int)biasSize * common->kernelX() * common->kernelY();
    auto core = static_cast<CPUBackend*>(b)->functions();
    auto int8_core = static_cast<CPUBackend*>(backend())->int8Functions();
    int unit     = core->pack;
    int ePack, lPack, hPack;
    core->MNNGetMatMulPackMode(&ePack, &lPack, &hPack);
    // printf("ePack, lPack, hPack = %d, %d, %d\n", ePack, lPack, hPack);
    // printf("UNIT, SRC_UNIT, DST_XUNIT = %d, %d, %d\n", UNIT, SRC_UNIT, DST_XUNIT);
    hPack = unit;
    lPack = unit;
    // [oc, ic] => [oc/unit, ic/src_unit, unit, src_unit]
    if (unit == 4 && core->supportI8mm) { // Low Memory: use fp32 and smmla.
        hPack = 8;
        lPack = 8;
    }
    auto hU = UP_DIV(outputCount, hPack);
    auto lU = UP_DIV(inputCount, lPack);
    ConvolutionHybrid::initQuantizeResource(quantInfo, mResource, hU, hPack, lU, lPack, outputCount, (int)originWeightSize / (int)biasSize, common->kernelX() * common->kernelY(), core->bytes);
}

ConvolutionHybrid::ConvolutionHybrid(std::shared_ptr<CPUConvolution::Resource> resource, const Convolution2DCommon *common, Backend* b) : CPUConvolution(common, b) {
    mResource = resource;
}

ConvolutionHybrid::~ConvolutionHybrid() {
    // Do nothing
}

bool ConvolutionHybrid::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new ConvolutionHybrid(mResource, op->main_as_Convolution2D()->common(), bn);
    return true;
}

ErrorCode ConvolutionHybrid::allocTensor(Tensor* tensor, size_t size) {
    tensor->buffer().type          = halide_type_of<int8_t>();
    tensor->buffer().dimensions    = 1;
    tensor->buffer().dim[0].extent = size;
    bool success = backend()->onAcquireBuffer(tensor, Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }
    return NO_ERROR;
}

ErrorCode ConvolutionHybrid::allocDynamicQuantInfo(int thread, int batch, int ic, int oc, int bytes) {
    // absmax: thread * batch * bytes
    // sum: thread * batch * sizeof(int)
    // dequant_scale: batch * bytes
    // quant_scale: batch * bytes
    allocTensor(&mQuantInfo.quant_info, (thread + 2) * batch * bytes + thread * batch * sizeof(int));
    if (ANeedToPack8) {
        int ic8 = UP_DIV(ic, 8) * 8;
        int oc8 = UP_DIV(oc, 8) * 8;
        mInputTemp.reset(Tensor::createDevice<int8_t>({batch, 1, 1, ic8}));
        mOutputTemp.reset(Tensor::createDevice<float>({batch, 1, 1, oc8}));
        bool allocSucc = backend()->onAcquireBuffer(mInputTemp.get(), Backend::DYNAMIC);
        allocSucc      = allocSucc && backend()->onAcquireBuffer(mOutputTemp.get(), Backend::DYNAMIC);
        if (!allocSucc) {
            return OUT_OF_MEMORY;
        }
        allocTensor(&mQuantInfo.quant_buffer, batch * ic8);
        backend()->onReleaseBuffer(mInputTemp.get(), Backend::DYNAMIC);
        backend()->onReleaseBuffer(mOutputTemp.get(), Backend::DYNAMIC);
    } else {
        allocTensor(&mQuantInfo.quant_buffer, batch * ic);
    }
    backend()->onReleaseBuffer(&mQuantInfo.quant_info, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mQuantInfo.quant_buffer, Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode ConvolutionHybrid::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    CPUConvolution::onResize(inputs, outputs);
    auto input  = inputs[0];
    auto output = outputs[0];
    auto core = static_cast<CPUBackend*>(backend())->functions();
    auto int8_core = static_cast<CPUBackend*>(backend())->int8Functions();
    auto inputPtr = input->host<float>();
    auto outputPtr = output->host<float>();
    auto weightPtr = mResource->mWeight->host<float>();
    auto biasPtr = mResource->mBias->host<float>();
    auto batch = output->batch() * output->height() * output->width();
    int ic = input->channel();
    int oc = output->channel();
    int bytes    = core->bytes;
    int unit     = core->pack;
    int eP, lP, hP;
    core->MNNGetMatMulPackMode(&eP, &lP, &hP);
    int UNIT, SRC_UNIT, DST_XUNIT;
    int8_core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    hP = unit;
    lP = unit;
    int tileC     = std::max(unit, hP);
    LowMemoryGemmFuncWithInt8Weight gemmKernel;
    gemmKernel = core->MNNGemmHybridInt8;
    float weightBytes = 1;
    if (mResource->mDequantize.bits == 4) {
        weightBytes = 0.5;
        gemmKernel = core->MNNGemmHybridInt4;
    }

    const uint8_t* dequantAlpha = mResource->mDequantize.mScaleBias->host<uint8_t>();;
    const uint8_t* dequantBias = dequantAlpha + mResource->hU * mResource->hP * bytes;;
    int threadNumber = ((CPUBackend *)backend())->threadNumber();
    auto oC4      = UP_DIV(oc, tileC);
    int iC4        = UP_DIV(ic, unit);
    if (iC4 < threadNumber || oC4 < threadNumber) {
        threadNumber = std::min(oC4, iC4);
    }
    int tileCount = UP_DIV(oC4, threadNumber);
    int iTileCount = UP_DIV(iC4, threadNumber);
    if (unit == 4 && core->supportI8mm) { // Low Memory: use fp32 and smmla.
       ANeedToPack8 = true;
    }
    int8_t order[32] = {0, 1, 2, 3, 12, 13, 14, 15, 16, 17, 18, 19, 28, 29, 30, 31, 8, 9, 10, 11, 4, 5, 6, 7, 24, 25, 26, 27, 20, 21, 22, 23};
    allocDynamicQuantInfo(threadNumber, batch, ic, oc, bytes);
    mDynamicQuant = [=]() {
        auto maxPtr = mQuantInfo.quant_info.host<uint8_t>();
        auto sumPtr = maxPtr + threadNumber * batch * bytes;
        auto dequantPtr = sumPtr + threadNumber * batch * sizeof(int);
        auto quantPtr = dequantPtr + batch * bytes;
        // compute sum and absmax
        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            int workCount = iTileCount;
            if (tId == threadNumber - 1) {
                workCount = iC4 - tId * iTileCount;
            }
            int icIndex = tId * iTileCount;
            auto input_ptr = reinterpret_cast<const float*>(input->host<uint8_t>() + icIndex * batch * unit * bytes);
            auto max_ptr = reinterpret_cast<float*>(maxPtr + tId * batch * bytes);
            core->MNNAbsMax(input_ptr, max_ptr, workCount, batch, unit);
        }
        MNN_CONCURRENCY_END();
        // compute scale
        core->MNNQuantScale((float*)maxPtr, (float*)quantPtr, (float*)dequantPtr, threadNumber, batch);
        // quant
        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            int workCount = iTileCount;
            if (tId == threadNumber - 1) {
                workCount = iC4 - tId * iTileCount;
            }
            int icIndex = tId * iTileCount;
            auto input_ptr = reinterpret_cast<float*>(input->host<uint8_t>() + icIndex * batch * unit * bytes);
            auto quant_ptr = mQuantInfo.quant_buffer.host<int8_t>() + icIndex * batch * unit;
            auto scale_ptr = reinterpret_cast<float*>(quantPtr);
            auto sum_ptr = reinterpret_cast<float*>(sumPtr + tId * batch * sizeof(int));
            core->MNNDynamicQuant(input_ptr, quant_ptr, scale_ptr, sum_ptr, workCount, batch, unit);
        }
        MNN_CONCURRENCY_END();
        // compute quant sum
        core->MNNQuantSum((float*)sumPtr, (float*)dequantPtr, threadNumber, batch);
    };
    mFunction.first = threadNumber;
    mFunction.second = [=](int tId){
        int workCount = tileCount;
        if (tId == threadNumber - 1) {
            workCount = oC4 - tId * tileCount;
        }
        int unit_ = unit;
        int tileCount_ = tileCount;
        if (ANeedToPack8) {
            int oC8      = UP_DIV(oc, 8);
            tileCount_ = UP_DIV(oC8, threadNumber);
            workCount = tileCount_;
            if (tId == threadNumber - 1) {
                workCount = oC8 - tId * tileCount_;
            }
            unit_ = 8;
        }

        int ocIndex = tId * tileCount_ * unit_;
        const float* finput_ptr = input->host<float>();
        const int8_t* input_ptr = mQuantInfo.quant_buffer.host<int8_t>();
        const int8_t* input_ptr_tmp = mQuantInfo.quant_buffer.host<int8_t>();
        auto weight_ptr = mResource->mWeight->host<int8_t>() + static_cast<int>(ocIndex * ic * weightBytes);
        auto output_ptr = reinterpret_cast<float*>(outputs[0]->host<uint8_t>() + ocIndex * batch * bytes);
        if (ANeedToPack8 && batch > 1) {
            input_ptr = mInputTemp->host<int8_t>();
            output_ptr = reinterpret_cast<float*>(mOutputTemp->host<uint8_t>() + ocIndex * batch * bytes);
        }
        auto bias_ptr = reinterpret_cast<const float*>(mResource->mBias->host<uint8_t>() + ocIndex * bytes);
        auto alpha_ptr =  reinterpret_cast<const float*>(dequantAlpha + ocIndex * bytes);
        auto zero_ptr =  reinterpret_cast<const float*>(dequantBias + ocIndex * bytes);
        const uint8_t* max_ptr = mQuantInfo.quant_info.host<uint8_t>();
        const float* sums_ptr = reinterpret_cast<const float*>(max_ptr + threadNumber * batch * bytes);
        const float* scale_ptr = reinterpret_cast<const float*>(max_ptr + threadNumber * batch * (bytes + sizeof(int)));
        size_t dst_depth_quad = workCount;
        size_t src_depth_quad = UP_DIV(ic, unit_);
        size_t dst_step       = batch * unit_ * bytes;
        size_t realSize       = batch;
        const float* param[6];
        param[0] = alpha_ptr;
        param[1] = zero_ptr;
        param[2] = bias_ptr;
        param[3] = sums_ptr;
        param[4] = scale_ptr;
        param[5] = (float*)order;
        gemmKernel(output_ptr, input_ptr, weight_ptr, src_depth_quad, dst_step, dst_depth_quad, realSize, param);
    };
    return NO_ERROR;
}

ErrorCode ConvolutionHybrid::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mDynamicQuant();
    if (ANeedToPack8 && inputs[0]->batch() > 1) {
        auto core = static_cast<CPUBackend*>(backend())->functions();
        auto plane_in = inputs[0]->width() * inputs[0]->height() * inputs[0]->batch();
        auto plane_out = outputs[0]->width() * outputs[0]->height() * outputs[0]->batch();
        auto depth = UP_DIV(inputs[0]->channel(), core->pack);
        auto output_depth = UP_DIV(outputs[0]->channel(), core->pack);
        int areaOffset[2] = {plane_out, plane_out};
        MNNPackInt8C2Origin(mInputTemp.get()->host<float>(), mQuantInfo.quant_buffer.host<float>(), plane_in, depth, plane_in);
        MNN_CONCURRENCY_BEGIN(tId, mFunction.first) {
            mFunction.second((int)tId);
        }
        MNN_CONCURRENCY_END();
        MNNUnpackC2Float(outputs[0]->host<float>(), mOutputTemp.get()->host<float>(), plane_out, output_depth, areaOffset, core->pack);
        return NO_ERROR;
    }

    MNN_CONCURRENCY_BEGIN(tId, mFunction.first) {
        mFunction.second((int)tId);
    }
    MNN_CONCURRENCY_END();
    return NO_ERROR;
}
} // namespace MNN
