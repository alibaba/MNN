//
//  Arm82Convolution1x1.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Arm82Convolution1x1.hpp"
#include "Arm82Backend.hpp"
#include "Concurrency.h"
#include "MNN_generated.h"
#include "Macro.h"
#define SRC_Z_UNIT 4
#define DST_Z_UNIT 8
#define DST_X_UNIT 24

/*Use int16_t to replace fp16*/

extern "C" {
void MNNFloat32ToFloat16C4(int16_t* dst, const float* src, size_t sizeQuad);
void MNNFloat16ToFloat32C4(float* dst, const int16_t* src, size_t sizeQuad, size_t depth, size_t dstStride);

void MNNFloat16_Unit_4_MatMul(int16_t* dst, const int16_t* src, const int16_t* weight, size_t icUnit, size_t ocUnit,
                              size_t ocStep);
void MNNFloat16_Common_4_MatMul(int16_t* dst, const int16_t* src, const int16_t* weight, size_t icUnit, size_t ocUnit,
                                size_t ocStep, size_t width);

void MNNFloat16C8ToC4AddBias(int16_t* dst, const int16_t* src, const int16_t* bias, size_t size, size_t ocUnit);
void MNNFloat16C8ToC4AddBiasRelu(int16_t* dst, const int16_t* src, const int16_t* bias, size_t size, size_t ocUnit);
void MNNFloat16C8ToC4AddBiasRelu6(int16_t* dst, const int16_t* src, const int16_t* bias, size_t size, size_t ocUnit);
}

#ifndef __aarch64__
void MNNFloat32ToFloat16C4(int16_t* dst, const float* src, size_t sizeQuad) {
}
void MNNFloat16ToFloat32C4(float* dst, const int16_t* src, size_t sizeQuad, size_t depth, size_t dstStride) {
}

void MNNFloat16_Unit_4_MatMul(int16_t* dst, const int16_t* src, const int16_t* weight, size_t icUnit, size_t ocUnit,
                              size_t ocStep) {
}
void MNNFloat16_Common_4_MatMul(int16_t* dst, const int16_t* src, const int16_t* weight, size_t icUnit, size_t ocUnit,
                                size_t ocStep, size_t width) {
}

void MNNFloat16C8ToC4AddBias(int16_t* dst, const int16_t* src, const int16_t* bias, size_t size, size_t ocUnit) {
}
void MNNFloat16C8ToC4AddBiasRelu(int16_t* dst, const int16_t* src, const int16_t* bias, size_t size, size_t ocUnit) {
}
void MNNFloat16C8ToC4AddBiasRelu6(int16_t* dst, const int16_t* src, const int16_t* bias, size_t size, size_t ocUnit) {
}

#endif

namespace MNN {
bool Arm82Convolution1x1::support(const Op* op) {
    if (nullptr != op->main_as_Convolution2D()->quanParameter()) {
        // TODO: Currently Don't support idst quan
        return false;
    }
    // TODO: Current Only Support 1x1
    auto convOp = op->main_as_Convolution2D()->common();
    if (1 == convOp->kernelX() && 1 == convOp->kernelY() && 0 == convOp->padX() && 0 == convOp->padY() &&
        1 == convOp->strideX() && 1 == convOp->strideY()) {
        return true;
    }
    return false;
}

Arm82Convolution1x1::Arm82Convolution1x1(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                         const Op* op, Backend* bn)
    : Execution(bn) {
    mIm2ColParamter = new CPUConvolution::Im2ColParameter;
    mConvOp         = op->main_as_Convolution2D();
    auto input      = inputs[0];
    auto output     = outputs[0];
    auto common     = mConvOp->common();
    auto kw         = common->kernelX();
    auto kh         = common->kernelY();
    int ic          = input->channel();
    int oc          = output->channel();
    int kernelCount = kw * kh;
    mWeight.reset(Tensor::create<int16_t>(
        std::vector<int>{UP_DIV(oc, DST_Z_UNIT), UP_DIV(ic, SRC_Z_UNIT), kernelCount, DST_Z_UNIT * SRC_Z_UNIT}));
    std::shared_ptr<Tensor> tempTensor(Tensor::create<float>(std::vector<int>{mWeight->elementSize()}));
    ::memset(tempTensor->host<float>(), 0, tempTensor->size());
    auto tempDst      = tempTensor->host<float>();
    int cur           = 0;
    auto sourceWeight = mConvOp->weight()->data();
    for (int oz = 0; oz < oc; ++oz) {
        auto dstOz = tempDst + (oz / DST_Z_UNIT) * mWeight->stride(0) + oz % DST_Z_UNIT;
        for (int sz = 0; sz < ic; ++sz) {
            auto dstSz = dstOz + (sz / SRC_Z_UNIT) * mWeight->stride(1) + (sz % SRC_Z_UNIT) * DST_Z_UNIT;
            for (int kc = 0; kc < kernelCount; ++kc) {
                dstSz[kc * SRC_Z_UNIT * DST_Z_UNIT] = sourceWeight[cur++];
            }
        }
    }
    MNNFloat32ToFloat16C4(mWeight->host<int16_t>(), tempDst, tempTensor->elementSize() / 4);

    mBias.reset(Tensor::create<int16_t>(std::vector<int>{UP_DIV(oc, DST_Z_UNIT) * DST_Z_UNIT}));
    tempTensor.reset(Tensor::create<float>(mBias->shape()));
    ::memset(tempTensor->host<float>(), 0, tempTensor->size());
    ::memcpy(tempTensor->host<float>(), mConvOp->bias()->data(), oc * sizeof(float));
    MNNFloat32ToFloat16C4(mBias->host<int16_t>(), tempTensor->host<float>(), tempTensor->elementSize() / 4);
}
Arm82Convolution1x1::~Arm82Convolution1x1() {
    delete mIm2ColParamter;
}

ErrorCode Arm82Convolution1x1::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    auto ow     = output->width();
    auto oh     = output->height();
    auto iw     = input->width();
    auto ih     = input->height();

    auto common  = mConvOp->common();
    auto strideX = common->strideX();
    auto strideY = common->strideY();
    auto kw      = common->kernelX();
    auto kh      = common->kernelY();

    if (common->padMode() == PadMode::PadMode_VALID) {
        mIm2ColParamter->padX = ((ow - 1) * strideX + kw - iw + 1) / 2;
        mIm2ColParamter->padY = ((oh - 1) * strideY + kh - ih + 1) / 2;
    } else {
        mIm2ColParamter->padX = ((ow - 1) * strideX + kw - iw) / 2;
        mIm2ColParamter->padY = ((oh - 1) * strideY + kh - ih) / 2;
    }

    mIm2ColParamter->iw = iw;
    mIm2ColParamter->ih = ih;
    mIm2ColParamter->ow = ow;
    mIm2ColParamter->oh = oh;
    int numberThread    = ((Arm82Backend*)backend())->numberThread();
    mTempInput.reset(Tensor::createDevice<int16_t>(input->shape(), Tensor::CAFFE_C4));
    mTempCol.reset(Tensor::createDevice<int16_t>(
        std::vector<int>{numberThread, DST_X_UNIT, UP_DIV(input->channel(), SRC_Z_UNIT), SRC_Z_UNIT, kw * kh}));
    mTempDst.reset(Tensor::createDevice<int16_t>(
        std::vector<int>{numberThread, DST_X_UNIT, UP_DIV(output->channel(), DST_Z_UNIT), DST_Z_UNIT}));
    mTempDstC4.reset(Tensor::createDevice<int16_t>(
        std::vector<int>{numberThread, DST_X_UNIT, UP_DIV(output->channel(), DST_Z_UNIT), DST_Z_UNIT}));
    backend()->onAcquireBuffer(mTempInput.get(), Backend::DYNAMIC);
    backend()->onAcquireBuffer(mTempDst.get(), Backend::DYNAMIC);
    backend()->onAcquireBuffer(mTempDstC4.get(), Backend::DYNAMIC);
    backend()->onAcquireBuffer(mTempCol.get(), Backend::DYNAMIC);

    backend()->onReleaseBuffer(mTempInput.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mTempDstC4.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mTempDst.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mTempCol.get(), Backend::DYNAMIC);

    return NO_ERROR;
}

ErrorCode Arm82Convolution1x1::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];

    MNNFloat32ToFloat16C4(mTempInput->host<int16_t>(), input->host<float>(), mTempInput->elementSize() / 4);

    int plane         = output->width() * output->height();
    int tileCount     = UP_DIV(plane, DST_X_UNIT);
    int ocUnit        = UP_DIV(output->channel(), DST_Z_UNIT);
    int scUnit        = UP_DIV(input->channel(), SRC_Z_UNIT);
    auto dstOrigin    = output->host<float>();
    auto srcOrigin    = mTempInput->host<int16_t>();
    auto weight       = mWeight->host<int16_t>();
    auto bias         = mBias->host<int16_t>();
    int ocDiv4        = UP_DIV(output->channel(), 4);
    auto srcPlane     = input->width() * input->height();
    int numThread     = std::min(tileCount, (((Arm82Backend*)backend())->numberThread()));
    auto postFunction = MNNFloat16C8ToC4AddBias;
    if (mConvOp->common()->relu()) {
        postFunction = MNNFloat16C8ToC4AddBiasRelu;
    } else if (mConvOp->common()->relu6()) {
        postFunction = MNNFloat16C8ToC4AddBiasRelu6;
    }
    MNN_CONCURRENCY_BEGIN(tId, numThread) {
        auto tempDst    = mTempDst->host<int16_t>() + tId * mTempDst->stride(0);
        auto tempSource = mTempCol->host<int16_t>() + tId * mTempCol->stride(0);
        auto tempDst32  = mTempDstC4->host<int16_t>() + tId * mTempDstC4->stride(0);
        for (int t = (int)tId; t < tileCount; t += numThread) {
            auto start    = t * DST_X_UNIT;
            auto end      = std::min(start + DST_X_UNIT, plane);
            auto count    = end - start;
            auto srcStart = srcOrigin + start * SRC_Z_UNIT;

            // Im2Col
            for (int sz = 0; sz < scUnit; ++sz) {
                auto srcSz = srcStart + sz * srcPlane * SRC_Z_UNIT;
                auto dstSz = tempSource + sz * count * SRC_Z_UNIT;
                ::memcpy(dstSz, srcSz, count * SRC_Z_UNIT * sizeof(int16_t));
            }

            // MatrixMul
            if (DST_X_UNIT == count) {
                MNNFloat16_Unit_4_MatMul(tempDst, tempSource, weight, scUnit, ocUnit,
                                         count * DST_Z_UNIT * sizeof(int16_t));
            } else {
                MNNFloat16_Common_4_MatMul(tempDst, tempSource, weight, scUnit, ocUnit,
                                           count * DST_Z_UNIT * sizeof(int16_t), count);
            }

            // PostTreat
            postFunction(tempDst32, tempDst, bias, count, ocUnit);
            auto dstStart = dstOrigin + start * SRC_Z_UNIT;
            MNNFloat16ToFloat32C4(dstStart, tempDst32, count, ocDiv4, plane * SRC_Z_UNIT * sizeof(float));
        }
    }
    MNN_CONCURRENCY_END();

    return NO_ERROR;
}

} // namespace MNN
