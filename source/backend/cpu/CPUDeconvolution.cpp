//
//  CPUDeconvolution.cpp
//  MNN
//
//  Created by MNN on 2018/07/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUDeconvolution.hpp"
#include "BufferAllocator.hpp"
#include "CPUBackend.hpp"
#include "Concurrency.h"
#include "Macro.h"
#include "Matrix.hpp"
#include "TensorUtils.hpp"
#include "Vec4.hpp"
#include "compute/CommonOptFunction.h"
#include "compute/ConvOpt.h"
#include "compute/DeconvolutionWithStride.hpp"
//#define MNN_OPEN_TIME_TRACE
#include "AutoTime.hpp"

using namespace MNN::Math;
namespace MNN {
CPUDeconvolutionBasic::CPUDeconvolutionBasic(const Tensor* input, const Op* convOp, Backend* b)
    : CPUConvolution(convOp->main_as_Convolution2D()->common(), b) {
    mSrcCount = input->channel();
}

ErrorCode CPUDeconvolutionBasic::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    if (mCommon->padMode() == PadMode_SAME) {
        const int outputWidth  = output->width();
        const int outputHeight = output->height();

        const int outputWidthPadded  = (input->width() - 1) * mCommon->strideX() + mCommon->kernelX();
        const int outputHeightPadded = (input->height() - 1) * mCommon->strideY() + mCommon->kernelY();

        const int padNeededWidth  = outputWidthPadded - outputWidth;
        const int padNeededHeight = outputHeightPadded - outputHeight;

        mPadX = padNeededWidth / 2;
        mPadY = padNeededHeight / 2;
        return NO_ERROR;
    }
    mPadX = mCommon->padX();
    mPadY = mCommon->padY();

    return NO_ERROR;
}

CPUDeconvolutionCommon::CPUDeconvolutionCommon(const Tensor* input, const Op* convOp, Backend* b)
    : CPUDeconvolutionBasic(input, convOp, b) {
    auto conv2D     = convOp->main_as_Convolution2D();
    int outputCount = mCommon->outputCount();
    mBias.reset(Tensor::createDevice<float>(std::vector<int>{ALIGN_UP4(outputCount)}));
    bool success = b->onAcquireBuffer(mBias.get(), Backend::STATIC);
    if (!success) {
        mValid = false;
        return;
    }
    ::memset(mBias->host<float>(), 0, mBias->size());
    ::memcpy(mBias->host<float>(), conv2D->bias()->data(), conv2D->bias()->size() * sizeof(float));
}

CPUDeconvolutionCommon::~CPUDeconvolutionCommon() {
    backend()->onReleaseBuffer(mBias.get(), Backend::STATIC);
}

static void _transformWeight(const float* tempWeight, float* dest, int outputCount, int srcCount, int fh, int fw,
                             float* cache) {
    AUTOTIME;
    int srcCountD4 = UP_DIV(srcCount, 4);
    // c, n, h, w -> c/4, n, h, w, 4
    MNNPackC4(dest, tempWeight, fw * fh * outputCount, srcCount);
    // Permute: c/4, n, h, w, 4 -> n, h, w, c/4, 4
    auto outside = fw * fh * outputCount;
    for (int oc = 0; oc < outside; ++oc) {
        auto srcOc = dest + oc * 4;
        auto dstOc = cache + oc * 4 * srcCountD4;
        for (int ic = 0; ic < srcCountD4; ++ic) {
            auto srcIc = srcOc + ic * 4 * outside;
            auto dstIc = dstOc + ic * 4;
            Vec4::save(dstIc, Vec4::load(srcIc));
        }
    }
    // n, h, w, c/4, 4 -> n/4, c/4, h, w, 4, 4
    MNNPackC4(dest, cache, srcCountD4 * fw * fh * 4, outputCount);
}

CPUDeconvolution::CPUDeconvolution(const Tensor* input, const Op* convOp, Backend* backend)
    : MNN::CPUDeconvolutionCommon(input, convOp, backend) {
    auto layer              = convOp->main_as_Convolution2D()->common();
    const float* tempWeight = convOp->main_as_Convolution2D()->weight()->data();
    int fw                  = layer->kernelX();
    int fh                  = layer->kernelY();
    int srcCount            = mSrcCount;
    int alignedWeightSize   = ALIGN_UP4(layer->outputCount()) * ALIGN_UP4(srcCount) * fw * fh;
    mWeight.reset(Tensor::createDevice<float>(std::vector<int>{alignedWeightSize}));
    std::shared_ptr<Tensor> cache(Tensor::createDevice<float>({alignedWeightSize}));
    bool success = backend->onAcquireBuffer(mWeight.get(), Backend::STATIC) &&
                   backend->onAcquireBuffer(cache.get(), Backend::STATIC);
    if (!success) {
        mValid = false;
        return;
    }
    float* dest = mWeight->host<float>();
    MNN_ASSERT(nullptr != dest);
    int outputCount = layer->outputCount();
    _transformWeight(tempWeight, dest, outputCount, srcCount, fh, fw, cache->host<float>());
    backend->onReleaseBuffer(cache.get(), Backend::STATIC);
    mOrigin.reset(new CPUDeconvolutionOrigin(input, convOp, backend));
}

CPUDeconvolution::~CPUDeconvolution() {
    backend()->onReleaseBuffer(mWeight.get(), Backend::STATIC);
}

ErrorCode CPUDeconvolutionMultiInput::onExecute(const std::vector<Tensor*>& inputs,
                                                const std::vector<Tensor*>& outputs) {
    auto outputCount = outputs[0]->channel();
    auto srcCount    = inputs[0]->channel();
    auto fw          = inputs[1]->width();
    auto fh          = inputs[1]->height();
    _transformWeight(inputs[1]->host<float>(), mWeight->host<float>(), outputCount, srcCount, fh, fw,
                     mCacheWeight->host<float>());
    ::memset(mBias->host<float>(), 0, mBias->size());
    ::memcpy(mBias->host<float>(), inputs[2]->host<float>(), inputs[2]->size());
    return mOrigin->onExecute(mTempInputs, outputs);
}
ErrorCode CPUDeconvolutionMultiInput::onResize(const std::vector<Tensor*>& inputs,
                                               const std::vector<Tensor*>& outputs) {
    auto outputCount      = outputs[0]->channel();
    auto srcCount         = inputs[0]->channel();
    auto fw               = inputs[1]->width();
    auto fh               = inputs[1]->height();
    int alignedWeightSize = ALIGN_UP4(outputCount) * ALIGN_UP4(srcCount) * fw * fh;
    mWeight.reset(Tensor::createDevice<float>({alignedWeightSize}));
    mCacheWeight.reset(Tensor::createDevice<float>({alignedWeightSize}));
    mBias.reset(Tensor::createDevice<float>({ALIGN_UP4(outputCount)}));
    mTempInputs = {inputs[0], mWeight.get(), mBias.get()};
    backend()->onAcquireBuffer(mWeight.get(), Backend::DYNAMIC);
    backend()->onAcquireBuffer(mCacheWeight.get(), Backend::DYNAMIC);
    backend()->onAcquireBuffer(mBias.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mCacheWeight.get(), Backend::DYNAMIC);
    auto error = mOrigin->onResize(mTempInputs, outputs);
    backend()->onReleaseBuffer(mWeight.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mBias.get(), Backend::DYNAMIC);
    return error;
}

ErrorCode CPUDeconvolutionOrigin::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    CPUDeconvolutionBasic::onResize(inputs, outputs);
    auto input  = inputs[0];
    auto output = outputs[0];
    auto oc     = output->channel();
    if (ALIGN_UP4(oc) != inputs[2]->length(0)) {
        return INPUT_DATA_ERROR;
    }
    auto weightAddr = inputs[1]->host<float>();

    auto ocC4       = UP_DIV(output->channel(), 4);
    auto icC4       = UP_DIV(input->channel(), 4);
    auto kw         = mCommon->kernelX();
    auto kh         = mCommon->kernelY();
    auto dilateX    = mCommon->dilateX();
    auto dilateY    = mCommon->dilateY();
    auto strideX    = mCommon->strideX();
    auto strideY    = mCommon->strideY();
    auto padX       = mCommon->padX();
    auto padY       = mCommon->padY();
    auto width      = input->width();
    auto height     = input->height();
    auto src_height = output->height();
    auto src_width  = output->width();

    auto kernelCount = ocC4 * mCommon->kernelX() * mCommon->kernelY();
    mComputors.clear();
    auto allocator     = ((CPUBackend*)backend())->getBufferAllocator();
    int number         = std::min(kernelCount, ((CPUBackend*)backend())->threadNumber());
    auto kcUnit        = UP_DIV(kernelCount, number);
    auto plane         = width * height;
    const int maxDepth = 5;
    std::shared_ptr<Tensor> tempColTotalBuffer(Tensor::createDevice<float>({kernelCount, plane, 4}));
    auto res = backend()->onAcquireBuffer(tempColTotalBuffer.get(), Backend::DYNAMIC);
    if (!res) {
        return OUT_OF_MEMORY;
    }
    auto colBufferPtr = tempColTotalBuffer->host<float>();
    auto biasPtr      = inputs[2]->host<float>();

    for (int b = 0; b < input->batch(); ++b) {
        auto inputPtr  = input->host<float>() + b * input->stride(0);
        auto outputPtr = output->host<float>() + b * output->stride(0);
        allocator->barrierBegin();
        std::shared_ptr<char> __defer(nullptr, [allocator](void* c) { allocator->barrierEnd(); });
        Unit unit;

        for (int i = 0; i < number; ++i) {
            auto zStart = i * kcUnit;
            auto zEnd   = std::min(zStart + kcUnit, kernelCount);
            if (zEnd <= zStart) {
                continue;
            }
            auto kcSize = zEnd - zStart;
            allocator->beginGroup();
            std::shared_ptr<char> __deferGroup(nullptr, [allocator](void* c) { allocator->endGroup(); });
            std::shared_ptr<Tensor> tempColBuffer(
                Tensor::create<float>({kcSize, plane, 4}, colBufferPtr + tempColTotalBuffer->stride(0) * zStart));
            std::shared_ptr<Tensor> tempWeightBuffer(
                Tensor::create<float>({kcSize, icC4, 16}, weightAddr + icC4 * 16 * zStart));
            std::shared_ptr<StrassenMatrixComputor> computor(new StrassenMatrixComputor(backend(), maxDepth));
            std::shared_ptr<Tensor> tempInputBuffer(
                Tensor::create<float>({icC4, input->width() * input->height(), 4}, inputPtr));
            auto errorCode = computor->onEncode({tempInputBuffer.get(), tempWeightBuffer.get()}, {tempColBuffer.get()});
            if (NO_ERROR != errorCode) {
                return errorCode;
            }
            unit.matrixMulti.emplace_back(computor);
        }
        auto threadNumber = ((CPUBackend*)backend())->threadNumber();
        unit.postFunction = std::make_pair(
            threadNumber, [colBufferPtr, outputPtr, ocC4, width, height, kh, kw, padY, padX, dilateY, dilateX, strideY,
                           strideX, threadNumber, src_width, src_height, plane, biasPtr, this](int tId) {
                for (int z = (tId); z < ocC4; z += threadNumber) {
                    auto dstZ = outputPtr + z * src_height * src_width * 4;
                    auto srcZ = colBufferPtr + kw * kh * 4 * plane * z;
                    ::memset(dstZ, 0, 4 * src_width * src_height * sizeof(float));

                    for (int oy = 0; oy < height; ++oy) {
                        for (int ox = 0; ox < width; ++ox) {
                            int srcStartX = ox * strideX - padX;
                            int srcStartY = oy * strideY - padY;

                            int sfy = ALIMAX(0, (UP_DIV(-srcStartY, dilateY)));
                            int efy = ALIMIN(kh, UP_DIV(src_height - srcStartY, dilateY));

                            int sfx = ALIMAX(0, (UP_DIV(-srcStartX, dilateX)));
                            int efx = ALIMIN(kw, UP_DIV(src_width - srcStartX, dilateX));

                            auto dstStart = dstZ + srcStartX * 4 + srcStartY * src_width * 4;
                            auto srcStart = srcZ + 4 * (ox + oy * width);

                            for (int fy = sfy; fy < efy; ++fy) {
                                auto dstY = dstStart + fy * 4 * dilateY * src_width;
                                auto srcY = srcStart + fy * kw * plane * 4;
                                for (int fx = sfx; fx < efx; ++fx) {
                                    auto dstX = dstY + fx * dilateX * 4;
                                    auto srcX = srcY + fx * plane * 4;
                                    Vec4::save(dstX, Vec4::load(dstX) + Vec4::load(srcX));
                                }
                            }
                        }
                    }
                    mPostFunction(dstZ, biasPtr + 4 * z, src_height * src_width, 1);
                }
            });
        mComputors.emplace_back(unit);
    }
    backend()->onReleaseBuffer(tempColTotalBuffer.get(), Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode CPUDeconvolutionOrigin::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    for (auto& unit : mComputors) {
        MNN_CONCURRENCY_BEGIN(tId, unit.matrixMulti.size()) {
            unit.matrixMulti[tId]->onExecute();
        }
        MNN_CONCURRENCY_END();
        MNN_CONCURRENCY_BEGIN(tId, unit.postFunction.first) {
            unit.postFunction.second((int)tId);
        }
        MNN_CONCURRENCY_END();
    }
    return NO_ERROR;
}
class CPUDeconvolutionCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const {
        if (inputs.size() == 3) {
            return new CPUDeconvolutionMultiInput(inputs[0], op, backend);
        }
        auto convOp = op->main_as_Convolution2D();
        auto common = convOp->common();
        if (common->strideY() > 1 || common->strideX() > 1) {
            if (common->dilateX() == 1 && common->dilateY() == 1) {
                return new DeconvolutionWithStride(inputs[0], op, backend);
            }
        }
        return new CPUDeconvolution(inputs[0], op, backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUDeconvolutionCreator, OpType_Deconvolution);
} // namespace MNN
