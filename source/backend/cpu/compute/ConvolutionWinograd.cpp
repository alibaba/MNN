//
//  ConvolutionWinograd.cpp
//  MNN
//
//  Created by MNN on 2018/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/compute/ConvolutionWinograd.hpp"
#include <math.h>
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#include "backend/cpu/compute/ConvOpt.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "math/WingoradGenerater.hpp"
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif
#define CONVOLUTION_WINOGRAD_MAX_UNIT 8
#define CONVOLUTION_WINOGRAD_MIN_UNIT 2
using namespace MNN::Math;

//#define MNN_WINOGRAD_PRINT_REDUCE_RATE
//#define MNN_WINO_TRANFORM_TEST_CLOSE
namespace MNN {
ConvolutionWinograd::ConvolutionWinograd(const Convolution2DCommon *convOp, const Tensor *input, const Tensor *output,
                                         Backend *b, const float *originWeight, size_t originWeightSize,
                                         const float *bias, size_t biasSize, int unit)
    : MNN::CPUConvolution(convOp, b) {
    mResource.reset(new Resource);
    mResource->backend = b;
    mResource->mBias.reset(Tensor::createDevice<float>({ALIGN_UP4((int)biasSize)}));
    mValid = backend()->onAcquireBuffer(mResource->mBias.get(), Backend::STATIC);
    if (!mValid) {
        return;
    }

    ::memset(mResource->mBias->host<float>(), 0, mResource->mBias->size());
    ::memcpy(mResource->mBias->host<float>(), bias, biasSize * sizeof(float));
    mTempBuffer.buffer().type         = halide_type_of<float>();
    mTransformMidBuffer.buffer().type = halide_type_of<float>();
    MNN_ASSERT(mCommon->kernelX() == mCommon->kernelY());

    int threadNumber = ((CPUBackend *)backend())->threadNumber();

    auto kernelSize = mCommon->kernelY();
    WinogradGenerater generator(unit, kernelSize, 1, true);

    int alpha        = unit + kernelSize - 1;
    int alpha2       = alpha * alpha;
    mSourceTransform = WinogradFunction::chooseSourceTransform(alpha, alpha);
    mDestTransform   = WinogradFunction::chooseDestTransform(alpha, unit);

    int srcCount                       = input->channel();
    int outputCount                    = output->channel();
    auto ic4 = UP_DIV(srcCount, 4);
    auto oc4 = UP_DIV(outputCount, 4);
    int ePack, hPack, lPack;
    MNNGetMatMulPackMode(&ePack, &lPack, &hPack);
    if (hPack % 4 != 0) {
        auto hDiv = MNNGetC4DivNumber(hPack);
        mCacheBuffer.buffer().dimensions = 2;
        mCacheBuffer.buffer().dim[0].extent = threadNumber;
        mCacheBuffer.buffer().dim[1].extent = hDiv * ePack * 4 + ePack * 4 * oc4;
        TensorUtils::setLinearLayout(&mCacheBuffer);
    } else {
        mCacheBuffer.buffer().dimensions = 0;
    }

    mTempBuffer.buffer().dim[0].extent = threadNumber;
    mTempBuffer.buffer().dim[1].extent = ePack;
    mTempBuffer.buffer().dim[2].extent = ic4 + oc4;
    mTempBuffer.buffer().dim[3].extent = 4 * alpha2;
    TensorUtils::setLinearLayout(&mTempBuffer);

    mTransformMidBuffer.buffer().dim[0].extent = threadNumber;
    mTransformMidBuffer.buffer().dim[1].extent = 2;
    mTransformMidBuffer.buffer().dim[2].extent = alpha2;
    mTransformMidBuffer.buffer().dim[3].extent = 4;
    TensorUtils::setLinearLayout(&mTransformMidBuffer);

    mGemmMidBuffer.buffer().dim[0].extent = threadNumber;
    mGemmMidBuffer.buffer().dim[1].extent = ePack * ic4 * 4;
    mGemmMidBuffer.buffer().dimensions = 2;
    TensorUtils::setLinearLayout(&mGemmMidBuffer);
    mA = generator.A();
    mB = generator.B();
    

    // Transform Kernel
    auto G = generator.G();
    std::shared_ptr<Tensor> sourceWeight(Tensor::create<float>(
        std::vector<int>{outputCount, srcCount, kernelSize, kernelSize}, (void *)originWeight, Tensor::CAFFE));
    mResource->mWeight = generator.allocTransformWeight(sourceWeight.get(), 1, hPack, false);
    mValid  = backend()->onAcquireBuffer(mResource->mWeight.get(), Backend::STATIC);
    if (!mValid) {
        return;
    }
    generator.transformWeight(mResource->mWeight.get(), sourceWeight.get());
}
ConvolutionWinograd::~ConvolutionWinograd() {
    // Do nothing
}
bool ConvolutionWinograd::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    auto dstExe = new ConvolutionWinograd(mResource, op->main_as_Convolution2D()->common(), bn);
    dstExe->mA = mA;
    dstExe->mB = mB;
    TensorUtils::copyShape(&mCacheBuffer, &(dstExe->mCacheBuffer), true);
    TensorUtils::copyShape(&mTempBuffer, &(dstExe->mTempBuffer), true);
    TensorUtils::copyShape(&mTransformMidBuffer, &(dstExe->mTransformMidBuffer), true);
    TensorUtils::copyShape(&mGemmMidBuffer, &(dstExe->mGemmMidBuffer), true);
    dstExe->mSourceTransform = mSourceTransform;
    dstExe->mDestTransform = mDestTransform;
    *dst = dstExe;
    return true;
}

ErrorCode ConvolutionWinograd::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input   = inputs[0];
    auto output  = outputs[0];
    auto dstUnit = mA->length(1);
    auto srcUnit = mA->length(0);
    int ePack, lPack, hPack;
    MNNGetMatMulPackMode(&ePack, &lPack, &hPack);

    auto srcUnit2 = srcUnit * srcUnit;

    int ow   = output->width();
    int oh   = output->height();
    int iw   = input->width();
    int ih   = input->height();
    int ic_4 = UP_DIV(input->channel(), 4);
    int dc_4 = UP_DIV(output->channel(), 4);
    // MNN_PRINT("%d, %d\n", srcUnit, dstUnit);

    int padY = mPadY;
    int padX = mPadX;

    auto wUnit = UP_DIV(ow, dstUnit);
    auto hUnit = UP_DIV(oh, dstUnit);

    auto totalCount   = wUnit * hUnit;
    auto postFunction = mPostFunction;
    // MNN_PRINT("ow=%d, oh=%d\n", ow, oh);
    int threadNumber = std::max(((CPUBackend *)backend())->threadNumber(), 1);
    int tileCount    = UP_DIV(totalCount, ePack);
    int eRemain = totalCount % ePack;
    threadNumber     = std::min(threadNumber, tileCount);
    std::vector<size_t> parameters(6);
    parameters[0] = eRemain * sizeof(float);
    parameters[1] = input->channel();
    parameters[2] = output->channel();
    parameters[3] = ePack * 4 * sizeof(float);
    parameters[4] = 0;
    parameters[5] = 0;

    std::vector<size_t> parametersRemain = parameters;
    parametersRemain[3] = eRemain * 4 * sizeof(float);


    for (int batchIndex = 0; batchIndex < input->batch(); ++batchIndex) {
        auto srcOrigin = input->host<float>() + batchIndex * input->stride(0);
        auto dstOrigin = output->host<float>() + batchIndex * output->stride(0);

        auto weight    = mResource->mWeight->host<float>();
        auto bias      = mResource->mBias->host<float>();
        auto tFunction = [&](int tId) {
            auto _srcOrigin = mTempBuffer.host<float>() + tId * mTempBuffer.stride(0);
            auto gemmBuffer = mGemmMidBuffer.host<float>() + tId * mGemmMidBuffer.stride(0);
            auto cache = mCacheBuffer.host<float>() + tId * mCacheBuffer.stride(0);
            auto midBuffer0 = mTransformMidBuffer.host<float>() + tId * mTransformMidBuffer.stride(0);
            auto midBuffer1 =
                mTransformMidBuffer.host<float>() + tId * mTransformMidBuffer.stride(0) + mTransformMidBuffer.stride(1);
            for (int tIndex = (int)tId; tIndex < tileCount; tIndex += threadNumber) {
                int xIndex  = (int)tIndex * ePack;
                int xReamin = totalCount - xIndex;
                int xC      = xReamin > ePack ? ePack : xReamin;

                /*Source Transform Begin*/
#ifndef MNN_WINO_TRANFORM_TEST_CLOSE
                {
                    int sourceZStep = iw * ih * 4;
                    int dstZStep    = xC * 4;
                    int unitStep    = ic_4 * xC * 4;
                    int oyBegin = xIndex / wUnit;
                    int oxBegin = xIndex % wUnit;
                    int oyEnd = (xIndex + xC-1) / wUnit;
                    int remain = xC;
                    auto dstS = _srcOrigin;
                    for (int hIndex=oyBegin; hIndex <= oyEnd; ++hIndex) {
                        int step = std::min(wUnit - oxBegin, remain);
                        int srcY  = hIndex * dstUnit - padY;
                        int ey    = ALIMIN(srcY + srcUnit, ih) - srcY;
                        int sy    = ALIMAX(0, srcY) - srcY;
                        for (int si=0; si<step; ++si) {
                            auto wIndex = si + oxBegin;
                            int srcX  = wIndex * dstUnit - padX;
                            int sx    = ALIMAX(0, srcX) - srcX;
                            int ex    = ALIMIN(srcX + srcUnit, iw) - srcX;
                            int count = 4 * (ex - sx);
                            auto dst_x = dstS + 4 * si;
                            auto srcStart = srcOrigin + (srcX + srcY * iw) * 4;
                            if (ex - sx == srcUnit && ey - sy == srcUnit) {
                                for (int z = 0; z < ic_4; ++z) {
                                    auto srcZ = srcStart + z * sourceZStep;
                                    // Transform
                                    for (int i = 0; i < srcUnit; ++i) {
                                        mSourceTransform(srcZ + 4 * i * iw, midBuffer1 + 4 * i, 4, 4 * srcUnit);
                                    }
                                    auto dstZ = dst_x + z * dstZStep;
                                    for (int i = 0; i < srcUnit; ++i) {
                                        mSourceTransform(midBuffer1 + 4 * i * srcUnit, dstZ + i * unitStep, 4,
                                                         unitStep * srcUnit);
                                    }
                                }
                            } else {
                                for (int z = 0; z < ic_4; ++z) {
                                    // Extract
                                    auto srcZ = srcStart + z * sourceZStep;
                                    ::memset(midBuffer0, 0, mTransformMidBuffer.stride(1) * sizeof(float));
                                    if (count > 0) {
                                        for (int yy = sy; yy < ey; ++yy) {
                                            auto dst_yy = midBuffer0 + yy * srcUnit * 4 + sx * 4;
                                            auto src_yy = srcZ + 4 * iw * yy + sx * 4;
                                            ::memcpy(dst_yy, src_yy, count * sizeof(float));
                                        }
                                    }
                                    // Transform
                                    for (int i = 0; i < srcUnit; ++i) {
                                        mSourceTransform(midBuffer0 + 4 * i * srcUnit, midBuffer1 + 4 * i, 4, 4 * srcUnit);
                                    }
                                    auto dstZ = dst_x + z * dstZStep;
                                    for (int i = 0; i < srcUnit; ++i) {
                                        mSourceTransform(midBuffer1 + 4 * i * srcUnit, dstZ + i * unitStep, 4,
                                                         unitStep * srcUnit);
                                    }
                                }
                            }
                        }
                        oxBegin = 0;
                        remain -= step;
                        dstS += 4 * step;
                    }
                }
                /*Source Transform End*/
#endif
                // Multi
                auto _dstOrigin = _srcOrigin + xC * srcUnit2 * ic_4 * 4;

                if (xC == ePack) {
                    for (int i = 0; i < srcUnit2; ++i) {
                        MNNPackC4ForMatMul_A(gemmBuffer, _srcOrigin + i * ic_4 * 4 * xC, ePack, ic_4 * 4, ePack);
                        MNNPackedMatMul(_dstOrigin + i * dc_4 * 4 * xC, gemmBuffer, weight + i * mResource->mWeight->stride(0), parameters.data(), cache, nullptr, nullptr);
                    }
                } else {
                    for (int i = 0; i < srcUnit2; ++i) {
                        MNNPackC4ForMatMul_A(gemmBuffer, _srcOrigin + i * ic_4 * 4 * xC, xC, ic_4 * 4, xC);
                        MNNPackedMatMulRemain(_dstOrigin + i * dc_4 * 4 * xC, gemmBuffer, weight + i * mResource->mWeight->stride(0), xC, parametersRemain.data(), cache, nullptr, nullptr);
                    }
                }
#ifndef MNN_WINO_TRANFORM_TEST_CLOSE
                /* Dest Transform And Post Treat Begin */
                {
                    int dstZStep = ow * oh * 4;
                    int srcZStep = xC * 4;
                    int unitStep = dc_4 * xC * 4;
                    int oyBegin = xIndex / wUnit;
                    int oxBegin = xIndex % wUnit;
                    int oyEnd = (xIndex + xC-1) / wUnit;
                    int remain = xC;
                    auto dstS = _dstOrigin;
                    for (int hIndex=oyBegin; hIndex <= oyEnd; ++hIndex) {
                        int step = std::min(wUnit - oxBegin, remain);
                        int dstY = hIndex * dstUnit;
                        int ey = ALIMIN(dstY + dstUnit, oh) - dstY;
                        for (int si=0; si<step; ++si) {
                            auto wIndex = si + oxBegin;
                            auto srcXi = dstS + 4 * si;
                            int dstX = wIndex * dstUnit;
                            auto dstStart = dstOrigin + 4 * (dstX + dstY * ow);
                            int ex = ALIMIN(dstX + dstUnit, ow) - dstX;

                            int count = ex * 4;
                            if (ex == dstUnit) {
                                for (int z = 0; z < dc_4; ++z) {
                                    auto dstZAddr = dstStart + z * dstZStep;
                                    auto srcZ     = srcXi + z * srcZStep;
                                    // Transform
                                    for (int i = 0; i < srcUnit; ++i) {
                                        mDestTransform(srcZ + i * unitStep, midBuffer0 + i * dstUnit * 4,
                                                       srcUnit * unitStep, 4);
                                    }
                                    for (int i = 0; i < ey; ++i) {
                                        auto dstAddr = dstZAddr + i * 4 * ow;
                                        mDestTransform(midBuffer0 + i * 4, dstAddr, 4 * dstUnit, 4);
                                    }
                                }
                            } else {
                                for (int z = 0; z < dc_4; ++z) {
                                    auto dstZAddr = dstStart + z * dstZStep;
                                    auto srcZ     = srcXi + z * srcZStep;
                                    // Transform
                                    for (int i = 0; i < srcUnit; ++i) {
                                        mDestTransform(srcZ + i * unitStep, midBuffer0 + i * dstUnit * 4,
                                                       srcUnit * unitStep, 4);
                                    }
                                    for (int i = 0; i < ey; ++i) {
                                        mDestTransform(midBuffer0 + i * 4, midBuffer1 + i * dstUnit * 4, 4 * dstUnit, 4);
                                    }
                                    for (int yy = 0; yy < ey; ++yy) {
                                        auto dstYAddr = dstZAddr + yy * 4 * ow;
                                        auto srcYAddr = midBuffer1 + yy * 4 * dstUnit;
                                        ::memcpy(dstYAddr, srcYAddr, count * sizeof(float));
                                    }
                                }
                            }
                        }
                        oxBegin = 0;
                        remain -= step;
                        dstS += 4 * step;
                    }
                }
#endif
                /*Dest Transform And Post Treat End*/
            }
        };

        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            tFunction((int)tId);
        }
        MNN_CONCURRENCY_END();

        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            for (int dy=(int)tId; dy < dc_4; dy += threadNumber) {
                postFunction(dstOrigin + 4 * ow * oh * dy, bias + 4* dy, ow * oh, 1);
            }
        }
        MNN_CONCURRENCY_END();
    }

    return NO_ERROR;
}

int ConvolutionWinograd::bestWinogradUnit(const Convolution2DCommon *common, const Tensor *inputTensor,
                                          const Tensor *outputTensor, int threadNumber) {
    int ow      = outputTensor->width();
    int oh      = outputTensor->height();
    int oc      = outputTensor->channel();
    int ePack, hPack, lPack;
    MNNGetMatMulPackMode(&ePack, &lPack, &hPack);
    int unit2   = UP_DIV(ow * oh, ePack * threadNumber);
    int maxUnit = (int)::sqrtf((float)unit2);
    maxUnit     = std::min(maxUnit, CONVOLUTION_WINOGRAD_MAX_UNIT);
    maxUnit     = std::max(maxUnit, CONVOLUTION_WINOGRAD_MIN_UNIT);

    int ic           = inputTensor->channel();
    auto kernelSize  = common->kernelY();
    int unit         = 0;
    float maxRate    = 0.0f;
    float originCost = (float)ow * oh * (float)ic * oc * kernelSize * kernelSize;
    static std::set<int> supportSu{4, 6, 8};
    for (int u = CONVOLUTION_WINOGRAD_MIN_UNIT; u <= maxUnit; ++u) {
        auto sui = u + kernelSize - 1;
        auto su = (float)sui;
        if (supportSu.find(sui) == supportSu.end()) {
            continue;
        }
        if (nullptr == WinogradFunction::chooseDestTransform((int)su, u)) {
            continue;
        }
        /*Let F(6,3) be choosed when it can speed up from F(2,3) than 0.6*/
        float penalty = (su * su) / (float)(kernelSize * kernelSize) * 0.12f;
        float winogradCost =
            (2 * su * su * ic + su * su * ic * oc + (su + u) * u * oc) * (UP_DIV(ow, u) * UP_DIV(oh, u));
        float reduceRate = originCost / winogradCost - penalty;
        // MNN_PRINT("ow=%d, oh=%d, %f, %f, winograd unit:%d\n", ow, oh, winogradCost, reduceRate, u);
        if (reduceRate > maxRate) {
            maxRate = reduceRate;
            unit    = u;
        }
    }
    if (maxRate < 1.0f) {
        return 0;
    }
    return unit;
}

bool ConvolutionWinograd::canUseWinograd(const Convolution2DCommon *common) {
    if (common->kernelY() != common->kernelX() || common->kernelY() <= 1) {
        return false;
    }
    if (common->dilateX() != 1 || common->dilateY() != 1) {
        return false;
    }
    if (common->strideX() != 1 || common->strideY() != 1) {
        return false;
    }
    return true;
}

ErrorCode ConvolutionWinograd::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    CPUConvolution::onResize(inputs, outputs);
    // FUNC_PRINT(mA->length(1));
    bool success = backend()->onAcquireBuffer(&mTempBuffer, Backend::DYNAMIC);
    success      = success && backend()->onAcquireBuffer(&mGemmMidBuffer, Backend::DYNAMIC);
    success      = success && (backend()->onAcquireBuffer(&mTransformMidBuffer, Backend::DYNAMIC));
    if (mCacheBuffer.buffer().dimensions > 0) {
        success      = success && backend()->onAcquireBuffer(&mCacheBuffer, Backend::DYNAMIC);
    }
    backend()->onReleaseBuffer(&mTempBuffer, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mTransformMidBuffer, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mGemmMidBuffer, Backend::DYNAMIC);
    if (mCacheBuffer.buffer().dimensions > 0) {
        backend()->onReleaseBuffer(&mCacheBuffer, Backend::DYNAMIC);
    }
    if (!success) {
        return OUT_OF_MEMORY;
    }
    return NO_ERROR;
}
} // namespace MNN
