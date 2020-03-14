//
//  CPUConv2DBackPropFilter.cpp
//  MNN
//
//  Created by MNN on 2019/4/18.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUConv2DBackPropFilter.hpp"
#include "backend/cpu/CPUMatMul.hpp"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include "math/Vec4.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/BufferAllocator.hpp"
using namespace MNN::Math;
namespace MNN {
CPUConv2DBackPropFilter::CPUConv2DBackPropFilter(const Convolution2DCommon *convOp, Backend *bn)
    : CPUConvolution(convOp, bn) {
    mStrideX = mCommon->strideX();
    mStrideY = mCommon->strideY();
    mDilateX = mCommon->dilateX();
    mDilateY = mCommon->dilateY();
}

ErrorCode CPUConv2DBackPropFilter::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input        = inputs[0];
    auto outputDiff   = inputs[1];
    auto kw = mCommon->kernelX();
    auto kh = mCommon->kernelY();
    auto batch = outputDiff->batch();
    auto width = outputDiff->width();
    auto height = outputDiff->height();
    auto channel = outputDiff->channel();
    auto ic = input->channel();
    mMatMul.reset();

    // Compute Pad
    CPUConvolution::onResize({input}, {outputDiff});

    mFunctions.clear();
    std::shared_ptr<Tensor> tempInput;
    tempInput.reset(Tensor::createDevice<float>({
        input->batch(),
        input->height(),
        input->width(),
        input->channel(),
    }, Tensor::TENSORFLOW));
    auto res = backend()->onAcquireBuffer(tempInput.get(), Backend::DYNAMIC);
    if (!res) {
        return OUT_OF_MEMORY;
    }
    auto backend = (CPUBackend*)Execution::backend();
    auto threadNumber = backend->threadNumber();
    mFunctions.emplace_back(std::make_pair(threadNumber, [tempInput, input, threadNumber](int tId) {
        for (int batchIndex=tId; batchIndex < tempInput->batch(); batchIndex+=threadNumber) {
            auto src = input->host<float>() + batchIndex * input->stride(0);
            auto dst = tempInput->host<float>() + batchIndex * tempInput->stride(0);
            MNNTensorConvertNC4HW4ToNHWC(dst, src, input->width()*input->height(), input->channel());
        }
    }));
    std::shared_ptr<Tensor> colBuffer(Tensor::createDevice<float>({
        batch * width * height,
        kw * kh * ic
    }));
    res = backend->onAcquireBuffer(colBuffer.get(), Backend::DYNAMIC);
    if (!res) {
        return OUT_OF_MEMORY;
    }
    mFunctions.emplace_back(std::make_pair(threadNumber, [this, colBuffer, tempInput, batch, width, height, ic, kw, kh, threadNumber](int tId) {
        auto colAddr = colBuffer->host<float>();
        auto srcAddr = tempInput->host<float>();
        auto ih = tempInput->height();
        auto iw = tempInput->width();
        for (int n=tId; n<batch; n+=threadNumber) {
            auto srcBatch = srcAddr + tempInput->stride(0) * n;
            auto dstBatch = colAddr + n * kw * kh * ic * width * height;
            ::memset(dstBatch, 0, kw * kh * ic * width * height * sizeof(float));
            for (int y=0; y<height; ++y) {
                for (int x=0; x<width; ++x) {
                    auto dstX = dstBatch + (x+y*width)*kw*kh*ic;
                    for (int ky=0; ky<kh; ++ky) {
                        auto sy = y * mStrideY - mPadY + ky*mDilateY;
                        if (sy < 0 || sy >= ih) {
                            continue;
                        }
                        for (int kx=0; kx<kw; ++kx) {
                            auto sx = x * mStrideX - mPadX + kx*mDilateX;
                            if (sx < 0 || sx >= iw) {
                                continue;
                            }
                            auto dst = dstX + kx + ky*kw;
                            auto src = srcBatch + (sy * iw + sx) * ic;
                            for (int sz=0; sz<ic; ++sz) {
                                dst[kw*kh*sz] = src[sz];
                            }
                        }
                    }
                }
            }
        }
    }));
    backend->onReleaseBuffer(tempInput.get(), Backend::DYNAMIC);
    std::shared_ptr<Tensor> tempDest(Tensor::createDevice<float>({
        batch,
        height,
        width,
        channel
    }, Tensor::TENSORFLOW));
    res = backend->onAcquireBuffer(tempDest.get(), Backend::DYNAMIC);
    if (!res) {
        return OUT_OF_MEMORY;
    }
    mFunctions.emplace_back(std::make_pair(threadNumber, [tempDest, outputDiff, threadNumber] (int tId) {
        for (int batchIndex=tId; batchIndex < outputDiff->batch(); batchIndex+=threadNumber) {
            auto src = outputDiff->host<float>() + batchIndex * outputDiff->stride(0);
            auto dst = tempDest->host<float>() + batchIndex * tempDest->stride(0);
            MNNTensorConvertNC4HW4ToNHWC(dst, src, outputDiff->width()*outputDiff->height(), outputDiff->channel());
        }
    }));
    mMatMul.reset(new CPUMatMul(backend, true, false, true));
    std::shared_ptr<Tensor> tempDestWrap(Tensor::create<float>({
        batch * height * width,
        channel
    }, tempDest->host<float>()));
    std::shared_ptr<Tensor> tempWeight(Tensor::create<float>({
        channel,
        ic * kw * kh
    }, outputs[0]->host<float>()));
    auto code = mMatMul->onResize({tempDestWrap.get(), colBuffer.get()}, {tempWeight.get()});
    if (NO_ERROR != code) {
        return OUT_OF_MEMORY;
    }
    backend->onReleaseBuffer(tempDest.get(), Backend::DYNAMIC);
    backend->onReleaseBuffer(colBuffer.get(), Backend::DYNAMIC);
    return NO_ERROR;
}
ErrorCode CPUConv2DBackPropFilter::onExecute(const std::vector<Tensor *> &inputs,
                                             const std::vector<Tensor *> &outputs) {
    for (auto& f : mFunctions) {
        MNN_CONCURRENCY_BEGIN(tId, f.first) {
            f.second(tId);
        }
        MNN_CONCURRENCY_END();
    }
    if (nullptr != mMatMul) {
        mMatMul->onExecute({}, {});
    }
    return NO_ERROR;
}

class CPUConv2DBackPropFilterDepthwise : public CPUConv2DBackPropFilter {
public:
    CPUConv2DBackPropFilterDepthwise(const Convolution2DCommon *common, Backend *bn)
        : CPUConv2DBackPropFilter(common, bn) {
    }
    virtual ~CPUConv2DBackPropFilterDepthwise() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        auto originWeight = outputs[0];
        auto input        = inputs[0];
        auto outputDiff   = inputs[1];
        auto kw = mCommon->kernelX();
        auto kh = mCommon->kernelY();
        auto batch = outputDiff->batch();
        auto width = outputDiff->width();
        auto height = outputDiff->height();
        auto channel = outputDiff->channel();
        auto iw = input->width();
        auto ih = input->height();

        // Compute Pad
        CPUConvolution::onResize({input}, {outputDiff});
        auto channelC4 = UP_DIV(channel, 4);;
        auto threadNumber = std::min(((CPUBackend*)backend())->threadNumber(), channelC4);
        std::shared_ptr<Tensor> tempWeight(Tensor::createDevice<float>({threadNumber, kw*kh, 4}));
        auto res = backend()->onAcquireBuffer(tempWeight.get(), Backend::DYNAMIC);
        if (!res) {
            return OUT_OF_MEMORY;
        }
        backend()->onReleaseBuffer(tempWeight.get(), Backend::DYNAMIC);
        mFunctions.emplace_back(std::make_pair(threadNumber, [this, tempWeight, channelC4, outputDiff, input, originWeight, threadNumber, batch, kw, kh, width, height, iw, ih, channel](int tId) {
            auto tempDst = tempWeight->host<float>();
            for (int z=tId; z<channelC4; z+=threadNumber) {
                auto lastDst = originWeight->host<float>() + kw * kh * z * 4;
                auto inputZ = input->host<float>() + z * iw * ih * 4;
                auto diffZ = outputDiff->host<float>() + z * width * height * 4;
                for (int ky=0; ky<kh; ++ky) {
                    for (int kx=0; kx<kw; ++kx) {
                        Vec4 weightDiff(0.0f);
                        for (int b=0; b<batch; ++b) {
                            auto inputB = inputZ + b * input->stride(0);
                            auto diffB = diffZ + b * outputDiff->stride(0);
                            for (int y=0; y<height; ++y) {
                                auto sy = y*mStrideY - mPadY + mDilateY*ky;
                                if (sy < 0 || sy >= ih) {
                                    continue;
                                }
                                auto diffY = diffB + y * width * 4;
                                auto inputY = inputB + sy * iw * 4;
                                for (int x=0; x<width; ++x) {
                                    auto sx = x*mStrideX - mPadX + mDilateX*kx;
                                    if (sx >= 0 && sx < iw) {
                                        weightDiff = weightDiff + Vec4::load(diffY + 4*x) * Vec4::load(inputY + 4*sx);
                                    }
                                }
                            }
                        }
                        Vec4::save(tempDst + 4*(kx+ky*kw), weightDiff);
                    }
                }
                int packSize = 4;
                if (z == channelC4 -1) {
                    packSize = channel - z * 4;
                }
                MNNUnpackC4(lastDst, tempDst, kw*kh, packSize);
            }
        }));
        return NO_ERROR;
    }
};
class CPUConv2DBackPropFilterCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto conv2DCommon = op->main_as_Convolution2D()->common();
        if (inputs[0]->channel() == inputs[1]->channel() && inputs[1]->channel() == conv2DCommon->group()) {
            return new CPUConv2DBackPropFilterDepthwise(conv2DCommon, backend);
        }
        return new CPUConv2DBackPropFilter(conv2DCommon, backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUConv2DBackPropFilterCreator, OpType_Conv2DBackPropFilter);

} // namespace MNN
