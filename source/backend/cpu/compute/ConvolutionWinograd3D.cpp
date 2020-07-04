//
//  ConvolutionWinograd3D.cpp
//  MNN
//
//  Created by MNN on 2018/09/23.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/compute/ConvolutionWinograd3D.hpp"
#include "backend/cpu/CPUBackend.hpp"
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

namespace MNN {
ConvolutionWinograd3D::ConvolutionWinograd3D(const Convolution3DCommon *convOp, const Tensor *input, const Tensor *output,
                                             Backend *b, const float *originWeight, size_t originWeightSize,
                                             const float *bias, size_t biasSize, int unit) : Execution(b), mUnit(unit) {
    for (int32_t kernel: *(convOp->kernels())) {
        mKernels.push_back(kernel);
    }
    MNN_ASSERT(mKernels[1] == mKernels[2]);
    mPadMode = convOp->padMode();
    if (mPadMode != PadMode_SAME) {
        for (int32_t pad: *(convOp->pads())) {
            mPads.push_back(pad);
        }
    }
    mPostFunction = CPUConvolution3D::getPostFunction(convOp);

    const int inputChannel = convOp->inputCount(), outputChannel = convOp->outputCount();
    const int kernelDepth = mKernels[0], kernelSize = mKernels[1], alpha = unit + kernelSize - 1, alpha2 = alpha * alpha;
    mAlpha = alpha;

    mSourceTransform = WinogradFunction::chooseSourceTransform(alpha, alpha);
    mDestTransform   = WinogradFunction::chooseDestTransform(alpha, unit);

    mWeight.reset(Tensor::createDevice<float>({ALIGN_UP4(inputChannel) * ALIGN_UP4(outputChannel) * kernelDepth * alpha2}));
    mBias.reset(Tensor::createDevice<float>({ALIGN_UP4((int)biasSize)}));
    bool valid = b->onAcquireBuffer(mWeight.get(), Backend::STATIC);
    valid = valid && b->onAcquireBuffer(mBias.get(), Backend::STATIC);
    if (!valid) {
        return;
    }

    memset(mBias->host<float>(), 0, mBias->size());
    memcpy(mBias->host<float>(), bias, biasSize * sizeof(float));

    WinogradGenerater generator(unit, kernelSize);

    const int srcDepthStep = inputChannel * outputChannel * kernelSize * kernelSize;
    const int dstDepthStep = ALIGN_UP4(inputChannel) * ALIGN_UP4(outputChannel) * alpha2;
    std::shared_ptr<Tensor> srcWeight, transWeight;
    for (int d = 0; d < kernelDepth; ++d) {
        srcWeight.reset(Tensor::create<float>({outputChannel, inputChannel, kernelSize, kernelSize}, (void*)(originWeight + d * srcDepthStep)));
        transWeight.reset(Tensor::create<float>({alpha2, UP_DIV(outputChannel, 4), UP_DIV(inputChannel, 4), 4, 4},
                                                (void*)(mWeight->host<float>() + d * dstDepthStep)));
        generator.transformWeight(transWeight.get(), srcWeight.get());
    }
}
ConvolutionWinograd3D::~ConvolutionWinograd3D() {
    if (nullptr != mBias) {
        backend()->onReleaseBuffer(mBias.get(), Backend::STATIC);
    }
    if (nullptr != mWeight) {
        backend()->onReleaseBuffer(mWeight.get(), Backend::STATIC);
    }
}

ErrorCode ConvolutionWinograd3D::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    const int oc = output->length(1), od = output->length(2);
    const int ic = input->length(1), id = input->length(2);
    const int threadNumber = ((CPUBackend*)backend())->threadNumber();
    const int alpha2 = mAlpha * mAlpha;
    auto CONVOLUTION_TILED_NUMBER = MNNGetConvolutionTileNumber();

    if (mPadMode == PadMode_SAME) {
        mPads.clear();
        for (int i = 0; i < 3; ++i) {
            int inputNeeded = output->length(i + 2) - 1 + mKernels[i];
            mPads.push_back((inputNeeded - input->length(i + 2)) / 2);
        }
    }

    mSourceBuffer.reset(Tensor::createDevice<float>({threadNumber, id, alpha2, UP_DIV(ic, 4), CONVOLUTION_TILED_NUMBER, 4}));
    mDestBuffer.reset(Tensor::createDevice<float>({threadNumber, od + 1, alpha2, UP_DIV(oc, 4), CONVOLUTION_TILED_NUMBER, 4}));
    mTempBuffer.reset(Tensor::createDevice<float>({threadNumber, 2, alpha2, 4}));

    bool succ = backend()->onAcquireBuffer(mSourceBuffer.get(), Backend::DYNAMIC);
    succ = succ && backend()->onAcquireBuffer(mDestBuffer.get(), Backend::DYNAMIC);
    succ = succ && backend()->onAcquireBuffer(mTempBuffer.get(), Backend::DYNAMIC);
    if (!succ) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(mSourceBuffer.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mDestBuffer.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mTempBuffer.get(), Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode ConvolutionWinograd3D::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input   = inputs[0];
    auto output  = outputs[0];
    auto CONVOLUTION_TILED_NUMBER = MNNGetConvolutionTileNumber();

    const int dstUnit = mUnit, srcUnit = mAlpha, srcUnit2 = srcUnit * srcUnit;
    const int outputWidth = output->length(4), outputHeight = output->length(3), outputDepth = output->length(2);
    const int inputWidth = input->length(4), inputHeight = input->length(3), inputDepth = input->length(2);
    const int wUnit = UP_DIV(outputWidth, dstUnit), hUnit = UP_DIV(outputHeight, dstUnit);
    const int ic_4 = UP_DIV(input->length(1), 4), dc_4 = UP_DIV(output->length(1), 4);
    const int padY = mPads[1], padX = mPads[2], padDepth = mPads[0], kernelDepth = mKernels[0];
    const int totalCount = wUnit * hUnit, tileCount = UP_DIV(totalCount, CONVOLUTION_TILED_NUMBER);

    auto postFunction = mPostFunction;
    const int threadNumber = std::max(((CPUBackend *)backend())->threadNumber(), 1);

    auto sourceTransformFunc = [=](int xIndex, int xC, const float* srcOrigin, float* dstOrigin, float* midBuffer0, float* midBuffer1) {
        int sourceZStep = inputDepth * inputWidth * inputHeight * 4;
        int dstZStep    = xC * 4;
        int unitStep    = ic_4 * xC * 4;
        for (int xi = 0; xi < xC; ++xi) {
            const int index = xIndex + xi, wIndex = index % wUnit, hIndex = index / wUnit;
            const int srcX = wIndex * dstUnit - padX, srcY = hIndex * dstUnit - padY;
            const int sx = ALIMAX(0, srcX) - srcX, ex = ALIMIN(srcX + srcUnit, inputWidth) - srcX;
            const int sy = ALIMAX(0, srcY) - srcY, ey = ALIMIN(srcY + srcUnit, inputHeight) - srcY;
            const int count = 4 * (ex - sx);

            auto dst_x = dstOrigin + 4 * xi;

            auto srcStart = srcOrigin + (srcX + srcY * inputWidth) * 4;
            if (ey - sy < srcUnit) {
                memset(midBuffer1, 0, srcUnit2 * 4 * sizeof(float));
            }
            if (ex - sx == srcUnit) {
                for (int z = 0; z < ic_4; ++z) {
                    auto srcZ = srcStart + z * sourceZStep;
                    auto dstZ = dst_x + z * dstZStep;
                    for (int d = 0; d < inputDepth; ++d) {
                        auto src_depth = srcZ + d * inputWidth * inputHeight * 4;
                        auto dst_depth = dstZ + d * srcUnit2 * ic_4 * xC * 4;
                        // Transform
                        for (int i = sy; i < ey; ++i) {
                            mSourceTransform(src_depth + 4 * i * inputWidth, midBuffer1 + 4 * i, 4, 4 * srcUnit);
                        }
                        for (int i = 0; i < srcUnit; ++i) {
                            mSourceTransform(midBuffer1 + 4 * i * srcUnit, dst_depth + i * unitStep, 4,
                                             unitStep * srcUnit);
                        }
                    }
                }
            } else {
                memset(midBuffer0, 0, srcUnit2 * 4 * sizeof(float));
                for (int z = 0; z < ic_4; ++z) {
                    // Extract
                    auto srcZ = srcStart + z * sourceZStep;
                    auto dstZ = dst_x + z * dstZStep;
                    for (int d = 0; d < inputDepth; ++d) {
                        auto src_depth = srcZ + d * inputWidth * inputHeight * 4;
                        auto dst_depth = dstZ + d * srcUnit2 * ic_4 * xC * 4;
                        if (count > 0) {
                            for (int yy = sy; yy < ey; ++yy) {
                                auto dst_yy = midBuffer0 + yy * srcUnit * 4 + sx * 4;
                                auto src_yy = src_depth + 4 * inputWidth * yy + sx * 4;
                                memcpy(dst_yy, src_yy, count * sizeof(float));
                            }
                        }
                        // Transform
                        for (int i = sy; i < ey; ++i) {
                            mSourceTransform(midBuffer0 + 4 * i * srcUnit, midBuffer1 + 4 * i, 4, 4 * srcUnit);
                        }
                        for (int i = 0; i < srcUnit; ++i) {
                            mSourceTransform(midBuffer1 + 4 * i * srcUnit, dst_depth + i * unitStep, 4,
                                             unitStep * srcUnit);
                        }
                    }
                }
            }
        }
    };

    auto destTransformFunc = [=](int xIndex, int xC, const float* srcOrigin, float* dstOrigin, float* midBuffer0, float* midBuffer1) {
        int dstZStep = outputDepth * outputHeight * outputWidth * 4;
        int srcZStep = xC * 4;
        int unitStep = dc_4 * xC * 4;
        for (int xi = 0; xi < xC; ++xi) {
            const int index = xIndex + xi, wIndex = index % wUnit, hIndex = index / wUnit;
            auto srcXi = srcOrigin + 4 * xi;

            const int dstX = wIndex * dstUnit, dstY = hIndex * dstUnit;
            auto dstStart = dstOrigin + 4 * (dstX + dstY * outputWidth);

            const int ey = ALIMIN(dstY + dstUnit, outputHeight) - dstY;
            const int ex = ALIMIN(dstX + dstUnit, outputWidth) - dstX;

            const int count = ex * 4;
            if (ex == dstUnit) {
                for (int z = 0; z < dc_4; ++z) {
                    auto dstZAddr = dstStart + z * dstZStep;
                    auto srcZ     = srcXi + z * srcZStep;
                    for (int d = 0; d < outputDepth; ++d) {
                        auto dst_depth = dstZAddr + d * outputHeight * outputWidth * 4;
                        auto src_depth = srcZ + d * srcUnit2 * dc_4 * xC * 4;
                        for (int i = 0; i < srcUnit; ++i) {
                            mDestTransform(src_depth + i * unitStep, midBuffer0 + i * dstUnit * 4,
                                           srcUnit * unitStep, 4);
                        }
                        for (int i = 0; i < ey; ++i) {
                            auto dstAddr = dst_depth + i * 4 * outputWidth;
                            mDestTransform(midBuffer0 + i * 4, dstAddr, 4 * dstUnit, 4);
                        }
                    }
                }
            } else {
                for (int z = 0; z < dc_4; ++z) {
                    auto dstZAddr = dstStart + z * dstZStep;
                    auto srcZ     = srcXi + z * srcZStep;
                    for (int d = 0; d < outputDepth; ++d) {
                        auto dst_depth = dstZAddr + d * outputHeight * outputWidth * 4;
                        auto src_depth = srcZ + d * srcUnit2 * dc_4 * xC * 4;
                        for (int i = 0; i < srcUnit; ++i) {
                            mDestTransform(src_depth + i * unitStep, midBuffer0 + i * dstUnit * 4,
                                           srcUnit * unitStep, 4);
                        }
                        for (int i = 0; i < ey; ++i) {
                            mDestTransform(midBuffer0 + i * 4, midBuffer1 + i * dstUnit * 4, 4 * dstUnit, 4);
                        }

                        for (int yy = 0; yy < ey; ++yy) {
                            auto dstYAddr = dst_depth + yy * 4 * outputWidth;
                            auto srcYAddr = midBuffer1 + yy * 4 * dstUnit;
                            memcpy(dstYAddr, srcYAddr, count * sizeof(float));
                        }
                    }
                }
            }
        }
    };

    auto gemmFunc = [=](int xC, int start, int end, const float* srcOrigin, const float* weight, float* dstOrigin) {
        float* tempDst = dstOrigin + outputDepth * srcUnit2 * dc_4 * xC * 4;
        const int element = (end - start) * dc_4 * xC * 4, offset = start * dc_4 * xC * 4;
        for (int od = 0; od < outputDepth; ++od) {
            bool add = false;
            float* _dstOrigin = dstOrigin + (od * srcUnit2 + start) * dc_4 * xC * 4;
            const int srcD = od - padDepth, kdStart = -ALIMIN(srcD, 0), kdEnd = kernelDepth - ALIMAX(srcD + kernelDepth - inputDepth, 0);
            for (int kd = kdStart; kd < kdEnd; ++kd) {
                const float* _srcOrigin = srcOrigin + (kd + srcD) * srcUnit2 * ic_4 * xC * 4;
                const float* _weight = weight + kd * srcUnit2 * dc_4 * ic_4 * 16;
                for (int i = start; i < end; ++i) {
                    if (xC == CONVOLUTION_TILED_NUMBER) {
                        MNNGemmFloatUnit_4(tempDst + i * dc_4 * xC * 4, _srcOrigin + i * ic_4 * 4 * xC,
                                           _weight + i * 16 * ic_4 * dc_4, ic_4, xC * 4, dc_4, 0);
                    } else {
                        MNNGemmFloatCommon_4(tempDst + i * dc_4 * xC * 4, _srcOrigin + i * ic_4 * 4 * xC,
                                             _weight + (i * dc_4) * ic_4 * 16, ic_4, xC * 4, dc_4, xC, 0);
                    }
                }
                if (add) {
                    MNNMatrixAdd(_dstOrigin, _dstOrigin, tempDst + offset, element / 4, 0, 0, 0, 1);
                } else {
                    memcpy(_dstOrigin, tempDst + offset, element * sizeof(float));
                }
                add = true;
            }
        }
    };

    auto gemmConcurrencyFunc = [=, &gemmFunc](int xC, const float* _srcOrigin, const float* weight, float* _dstOrigin) {
        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            const int step = UP_DIV(srcUnit2, threadNumber);
            gemmFunc(xC, tId * step, ALIMIN((tId + 1) * step, srcUnit2), _srcOrigin, weight, _dstOrigin);
        }
        MNN_CONCURRENCY_END()
    };

    auto tFunction = [&](const int tId, const int tileStart, const int tileStep, const int tileEnd, const float* srcOrigin, float* dstOrigin) {
        auto _srcOrigin = mSourceBuffer->host<float>() + tId * mSourceBuffer->stride(0);
        auto _dstOrigin = mDestBuffer->host<float>() + tId * mDestBuffer->stride(0);
        auto midBuffer0 = mTempBuffer->host<float>() + tId * mTempBuffer->stride(0);
        auto midBuffer1 = midBuffer0 + mTempBuffer->stride(1);
        for (int tIndex = (int)tId; tIndex < tileCount; tIndex += threadNumber) {
            int xIndex  = (int)tIndex * CONVOLUTION_TILED_NUMBER;
            int xReamin = totalCount - xIndex;
            int xC      = xReamin > CONVOLUTION_TILED_NUMBER ? CONVOLUTION_TILED_NUMBER : xReamin;

            sourceTransformFunc(xIndex, xC, srcOrigin, _srcOrigin, midBuffer0, midBuffer1);

            if (threadNumber != tileStep) {
                gemmConcurrencyFunc(xC, _srcOrigin, mWeight->host<float>(), _dstOrigin);
            } else {
                gemmFunc(xC, 0, srcUnit2, _srcOrigin, mWeight->host<float>(), _dstOrigin);
            }

            destTransformFunc(xIndex, xC, _dstOrigin, dstOrigin, midBuffer0, midBuffer1);
        }
    };

    for (int batchIndex = 0; batchIndex < input->batch(); ++batchIndex) {
        auto srcOrigin = input->host<float>() + batchIndex * input->stride(0);
        auto dstOrigin = output->host<float>() + batchIndex * output->stride(0);

        if (tileCount >= threadNumber) {
            MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
                tFunction((int)tId, (int)tId, threadNumber, tileCount / threadNumber * threadNumber, srcOrigin, dstOrigin);
            }
            MNN_CONCURRENCY_END();
        }

        if (tileCount % threadNumber != 0) {
            tFunction(0, tileCount / threadNumber * threadNumber, 1, tileCount, srcOrigin, dstOrigin);
        }

        MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
            int channelStep = UP_DIV(dc_4, threadNumber);
            int channelStart = channelStep * tId, channelNum = ALIMIN(channelStep * (tId + 1), dc_4) - channelStart;
            if (channelNum > 0) {
                postFunction(dstOrigin + channelStart * outputHeight * outputWidth * outputDepth * 4, mBias->host<float>() + 4 * channelStart, outputWidth * outputHeight * outputDepth, channelNum);
            }
        }
        MNN_CONCURRENCY_END();
    }

    return NO_ERROR;
}

int ConvolutionWinograd3D::bestWinogradUnit(const Convolution3DCommon *common, const Tensor *inputTensor,
                                          const Tensor *outputTensor, int threadNumber) {
    const int ow = outputTensor->length(4), oh = outputTensor->length(3), oc = outputTensor->length(1);
    auto CONVOLUTION_TILED_NUMBER = MNNGetConvolutionTileNumber();

    int unit2   = UP_DIV(ow * oh, CONVOLUTION_TILED_NUMBER * threadNumber);
    int maxUnit = (int)::sqrtf((float)unit2);
    maxUnit     = std::min(maxUnit, CONVOLUTION_WINOGRAD_MAX_UNIT);
    maxUnit     = std::max(maxUnit, CONVOLUTION_WINOGRAD_MIN_UNIT);

    int ic           = inputTensor->channel();
    auto kernelSize  = (*common->kernels())[1];
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

bool ConvolutionWinograd3D::canUseWinograd(const Convolution3DCommon *common) {
    std::vector<int> kernels;
    for (int kernel: *(common->kernels())) {
        if (kernel <= 1) {
            return false;
        }
        kernels.push_back(kernel);
    }
    if (kernels[1] != kernels[2]) {
        return false;
    }
    for (int dialate: *(common->dilates())) {
        if (dialate != 1) {
            return false;
        }
    }
    for (int stride: *(common->strides())) {
        if (stride != 1) {
            return false;
        }
    }
    return true;
}
} // namespace MNN
