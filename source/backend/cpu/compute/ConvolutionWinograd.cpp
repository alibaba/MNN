//
//  ConvolutionWinograd.cpp
//  MNN
//
//  Created by MNN on 2018/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "ConvolutionWinograd.hpp"
#include <math.h>
#include "CommonOptFunction.h"
#include "Concurrency.h"
#include "ConvOpt.h"
#include "Macro.h"
#include "TensorUtils.hpp"
#include "WingoradGenerater.hpp"
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif
#define CONVOLUTION_WINOGRAD_MAX_UNIT 6
#define CONVOLUTION_WINOGRAD_MIN_UNIT 2
using namespace MNN::Math;

//#define MNN_WINOGRAD_PRINT_REDUCE_RATE

namespace MNN {
ConvolutionWinograd::ConvolutionWinograd(const Convolution2DCommon *convOp, const Tensor *input, const Tensor *output,
                                         Backend *b, const float *originWeight, size_t originWeightSize,
                                         const float *bias, size_t biasSize, int unit)
    : MNN::CPUConvolution(convOp, b) {
    mBias.reset(Tensor::createDevice<float>({ALIGN_UP4((int)biasSize)}));
    mValid = backend()->onAcquireBuffer(mBias.get(), Backend::STATIC);
    if (!mValid) {
        return;
    }

    ::memset(mBias->host<float>(), 0, mBias->size());
    ::memcpy(mBias->host<float>(), bias, biasSize * sizeof(float));
    mTempBuffer.buffer().type         = halide_type_of<float>();
    mTransformMidBuffer.buffer().type = halide_type_of<float>();
    MNN_ASSERT(mCommon->kernelX() == mCommon->kernelY());

    int threadNumber = ((CPUBackend *)backend())->threadNumber();

    auto kernelSize = mCommon->kernelY();
    WinogradGenerater generator(unit, kernelSize);

    int alpha        = unit + kernelSize - 1;
    int alpha2       = alpha * alpha;
    mSourceTransform = WinogradFunction::chooseSourceTransform(alpha, alpha);
    mDestTransform   = WinogradFunction::chooseDestTransform(alpha, unit);

    int srcCount                       = input->channel();
    int outputCount                    = output->channel();
    mTempBuffer.buffer().dim[0].extent = threadNumber;
    mTempBuffer.buffer().dim[1].extent = CONVOLUTION_TILED_NUMBER;
    mTempBuffer.buffer().dim[2].extent = UP_DIV(srcCount, 4) + UP_DIV(outputCount, 4);
    mTempBuffer.buffer().dim[3].extent = 4 * alpha2;
    TensorUtils::setLinearLayout(&mTempBuffer);

    mTransformMidBuffer.buffer().dim[0].extent = threadNumber;
    mTransformMidBuffer.buffer().dim[1].extent = 2;
    mTransformMidBuffer.buffer().dim[2].extent = alpha2;
    mTransformMidBuffer.buffer().dim[3].extent = 4;
    TensorUtils::setLinearLayout(&mTransformMidBuffer);

    mA = generator.A();
    mB = generator.B();

    // Transform Kernel
    auto G = generator.G();
    std::shared_ptr<Tensor> sourceWeight(Tensor::create<float>(
        std::vector<int>{outputCount, srcCount, kernelSize, kernelSize}, (void *)originWeight, Tensor::CAFFE));
    mWeight = generator.allocTransformWeight(sourceWeight.get(), 4, 4, false);
    mValid  = backend()->onAcquireBuffer(mWeight.get(), Backend::STATIC);
    if (!mValid) {
        return;
    }
    generator.transformWeight(mWeight.get(), sourceWeight.get());
}
ConvolutionWinograd::~ConvolutionWinograd() {
    if (nullptr != mBias) {
        backend()->onReleaseBuffer(mBias.get(), Backend::STATIC);
    }
    if (nullptr != mWeight) {
        backend()->onReleaseBuffer(mWeight.get(), Backend::STATIC);
    }
}
ErrorCode ConvolutionWinograd::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input   = inputs[0];
    auto output  = outputs[0];
    auto dstUnit = mA->length(1);
    auto srcUnit = mA->length(0);

    auto srcUnit2 = srcUnit * srcUnit;
    auto dstUnit2 = dstUnit * dstUnit;

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
    int tileCount    = UP_DIV(totalCount, CONVOLUTION_TILED_NUMBER);
    threadNumber     = std::min(threadNumber, tileCount);

    for (int batchIndex = 0; batchIndex < input->batch(); ++batchIndex) {
        auto srcOrigin = input->host<float>() + batchIndex * input->stride(0);
        auto dstOrigin = output->host<float>() + batchIndex * output->stride(0);

        auto weight    = mWeight->host<float>();
        auto bias      = mBias->host<float>();
        auto tFunction = [&](int tId) {
            auto _srcOrigin = mTempBuffer.host<float>() + tId * mTempBuffer.stride(0);
            auto midBuffer0 = mTransformMidBuffer.host<float>() + tId * mTransformMidBuffer.stride(0);
            auto midBuffer1 =
                mTransformMidBuffer.host<float>() + tId * mTransformMidBuffer.stride(0) + mTransformMidBuffer.stride(1);
            for (int tIndex = (int)tId; tIndex < tileCount; tIndex += threadNumber) {
                int xIndex  = (int)tIndex * CONVOLUTION_TILED_NUMBER;
                int xReamin = totalCount - xIndex;
                int xC      = xReamin > CONVOLUTION_TILED_NUMBER ? CONVOLUTION_TILED_NUMBER : xReamin;

                /*Source Transform Begin*/
                {
                    int sourceZStep = iw * ih * 4;
                    int dstZStep    = xC * 4;
                    int unitStep    = ic_4 * xC * 4;
                    for (int xi = 0; xi < xC; ++xi) {
                        auto index = xIndex + xi;

                        int wIndex = index % wUnit;
                        int hIndex = index / wUnit;

                        int srcX  = wIndex * dstUnit - padX;
                        int srcY  = hIndex * dstUnit - padY;
                        int sy    = ALIMAX(0, srcY) - srcY;
                        int ey    = ALIMIN(srcY + srcUnit, ih) - srcY;
                        int sx    = ALIMAX(0, srcX) - srcX;
                        int ex    = ALIMIN(srcX + srcUnit, iw) - srcX;
                        int count = 4 * (ex - sx);

                        auto dst_x = _srcOrigin + 4 * xi;

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
                }
                /*Source Transform End*/

                // Multi
                auto _dstOrigin = _srcOrigin + xC * srcUnit2 * ic_4 * 4;

                if (xC == CONVOLUTION_TILED_NUMBER) {
                    for (int i = 0; i < srcUnit2; ++i) {
                        MNNGemmFloatUnit_4(_dstOrigin + i * dc_4 * 4 * xC, _srcOrigin + i * ic_4 * 4 * xC,
                                           weight + i * 16 * ic_4 * dc_4, ic_4, xC * 4, dc_4, 0);
                    }
                } else {
                    for (int i = 0; i < srcUnit2; ++i) {
                        MNNGemmFloatCommon_4(_dstOrigin + i * dc_4 * 4 * xC, _srcOrigin + i * ic_4 * 4 * xC,
                                             weight + i * 16 * ic_4 * dc_4, ic_4, xC * 4, dc_4, xC, 0);
                    }
                }

                /* Dest Transform And Post Treat Begin */
                {
                    int dstZStep = ow * oh * 4;
                    int srcZStep = xC * 4;
                    int unitStep = dc_4 * xC * 4;
                    for (int xi = 0; xi < xC; ++xi) {
                        auto index = xIndex + xi;
                        auto srcXi = _dstOrigin + 4 * xi;

                        int wIndex = index % wUnit;
                        int hIndex = index / wUnit;

                        int dstX = wIndex * dstUnit;
                        int dstY = hIndex * dstUnit;

                        auto dstStart = dstOrigin + 4 * (dstX + dstY * ow);

                        int ey = ALIMIN(dstY + dstUnit, oh) - dstY;
                        int ex = ALIMIN(dstX + dstUnit, ow) - dstX;

                        int count = ex * 4;
                        if (ex == dstUnit) {
                            for (int z = 0; z < dc_4; ++z) {
                                auto dstZAddr = dstStart + z * dstZStep;
                                auto srcZ     = srcXi + z * srcZStep;
                                auto biasZ    = bias + 4 * z;
                                // Transform
                                for (int i = 0; i < srcUnit; ++i) {
                                    mDestTransform(srcZ + i * unitStep, midBuffer0 + i * dstUnit * 4,
                                                   srcUnit * unitStep, 4);
                                }
                                for (int i = 0; i < ey; ++i) {
                                    auto dstAddr = dstZAddr + i * 4 * ow;
                                    mDestTransform(midBuffer0 + i * 4, dstAddr, 4 * dstUnit, 4);
                                    postFunction(dstAddr, biasZ, dstUnit, 1);
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
                                // PostTreat
                                postFunction(midBuffer1, bias + 4 * z, dstUnit2, 1);

                                for (int yy = 0; yy < ey; ++yy) {
                                    auto dstYAddr = dstZAddr + yy * 4 * ow;
                                    auto srcYAddr = midBuffer1 + yy * 4 * dstUnit;
                                    ::memcpy(dstYAddr, srcYAddr, count * sizeof(float));
                                }
                            }
                        }
                    }
                }
                /*Dest Transform And Post Treat End*/
            }
        };

        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            tFunction((int)tId);
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
    int unit2   = UP_DIV(ow * oh, CONVOLUTION_TILED_NUMBER * threadNumber);
    int maxUnit = (int)::sqrtf((float)unit2);
    maxUnit     = std::min(maxUnit, CONVOLUTION_WINOGRAD_MAX_UNIT);
    maxUnit     = std::max(maxUnit, CONVOLUTION_WINOGRAD_MIN_UNIT);

    int ic           = inputTensor->channel();
    auto kernelSize  = common->kernelY();
    int unit         = CONVOLUTION_WINOGRAD_MIN_UNIT;
    float maxRate    = 0.0f;
    float originCost = (float)ow * oh * (float)ic * oc * kernelSize * kernelSize;
    static std::set<int> supportSu{4, 8};
    for (int u = CONVOLUTION_WINOGRAD_MIN_UNIT; u <= maxUnit; ++u) {
        float su = (float)(u + kernelSize - 1);
        if (supportSu.find(su) == supportSu.end()) {
            continue;
        }
        if (nullptr == WinogradFunction::chooseDestTransform((int)su, u)) {
            continue;
        }
        /*Let F(6,3) be choosed when it can speed up from F(2,3) than 0.6*/
        float penalty = (su * su) / (float)(kernelSize * kernelSize) * 0.12f;
        float winogradCost =
            (2 * su * su * su * ic + su * su * ic * oc + 2 * su * u * u * oc) * (UP_DIV(ow, u) * UP_DIV(oh, u));
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
    success      = success && (backend()->onAcquireBuffer(&mTransformMidBuffer, Backend::DYNAMIC));
    backend()->onReleaseBuffer(&mTempBuffer, Backend::DYNAMIC);
    backend()->onReleaseBuffer(&mTransformMidBuffer, Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }
    return NO_ERROR;
}
} // namespace MNN
