//
//  CPUPoolInt8.cpp
//  MNN
//
//  Created by MNN on 2019/06/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUPoolInt8.hpp"
#include "core/Macro.h"
#include <math.h>
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif
#include "compute/Int8FunctionsOpt.h"
#include "core/Concurrency.h"
#include "backend/cpu/compute/CommonOptFunction.h"

namespace MNN {

static void poolingAvgNC16HW16Int8(void poolfunc(int8_t*, int8_t*, size_t, size_t, size_t, size_t, size_t, ssize_t, ssize_t), const Tensor *src, Tensor *dst,
                                   int stridesx, int stridesy, int kernelx, int kernely, int paddingx, int paddingy)
{
    const int inputHeight = src->height();
    const int inputWidth = src->width();
    const int outputHeight = dst->height();
    const int outputWidth = dst->width();
    const int channel = dst->channel();
    const int batchsize = src->batch();

    const auto srcPtr = src->host<int8_t>();
    auto dstPtr       = dst->host<int8_t>();
    int pack = 16;
    int thred0 = UP_DIV(paddingx, stridesx);
    int thred1 = inputWidth + paddingx - kernelx;
    thred1 = UP_DIV(thred1, stridesx);     // ix + kernelx >= inputWidth;
    // int factor = static_cast<int>((1 << 24)/(kernelx * kernely));

    const int channel_ = UP_DIV(channel, pack);
    for (int oc = 0; oc < channel_; ++oc) {
        for(int ob = 0; ob < batchsize; ++ob) {
            for (int oy = 0; oy < outputHeight; ++oy) {
                int iy = oy * stridesy - paddingy;
                const int kernely_ = std::min(iy + kernely, inputHeight) - std::max(iy, 0);
                iy = std::max(iy, 0);
                int ox = 0;
                for (ox = 0; ox < thred0; ++ox) { // ix < 0;
                    int ix = ox * stridesx - paddingx;
                    const int kernelx_ = std::min(ix + kernelx, inputWidth) - std::max(ix, 0);
                    ix = std::max(ix, 0);

                    int mul = static_cast<int>((1 << 24)/(kernelx_ * kernely_));

                    const int indexOutput = pack* (ox + outputWidth * (oy + outputHeight * (ob + batchsize * oc)));
                    const int indexInput = pack * (ix + inputWidth * (iy + inputHeight * (ob + batchsize * oc)));
                    int8_t* dstCur = dstPtr + indexOutput;
                    int8_t* srcCur = srcPtr + indexInput;

                    poolfunc(dstCur, srcCur, 1, inputWidth, kernelx_, kernely_, stridesx, paddingx, mul);

                } // ix < 0;

                // ix > 0 && ix + kernelx < inputWidth;
                if (thred1 - thred0 > 0) {
                    int ix = ox * stridesx - paddingx;
                    const int kernelx_ = std::min(ix + kernelx, inputWidth) - std::max(ix, 0);
                    ix = std::max(ix, 0);
                    int mul = static_cast<int>((1 << 24)/(kernelx_ * kernely_));

                    const int indexOutput = pack * (ox + outputWidth * (oy + outputHeight * (ob + batchsize * oc)));    
                    const int indexInput = pack * (ix + inputWidth * (iy + inputHeight * (ob + batchsize * oc)));

                    int8_t* dstCur = dstPtr + indexOutput;
                    int8_t* srcCur = srcPtr + indexInput;

                    poolfunc(dstCur, srcCur, thred1 - thred0, inputWidth, kernelx_, kernely_, stridesx, 0, mul);
                }

                for (ox = thred1; ox < outputWidth; ++ox) { // ix + kernelx > inputWidth;
                    int ix = ox * stridesx - paddingx;
                    const int kernelx_ = std::min(ix + kernelx, inputWidth) - std::max(ix, 0);
                    ix = std::max(ix, 0);

                    int mul = static_cast<int>((1 << 24)/(kernelx_ * kernely_));

                    const int indexOutput = pack* (ox + outputWidth * (oy + outputHeight * (ob + batchsize * oc)));
                    const int indexInput = pack * (ix + inputWidth * (iy + inputHeight * (ob + batchsize * oc)));
                    int8_t* dstCur = dstPtr + indexOutput;
                    int8_t* srcCur = srcPtr + indexInput;

                    poolfunc(dstCur, srcCur, 1, inputWidth, kernelx_, kernely_, stridesx, paddingx, mul);

                } 
            }
        }
    }
}

static void poolingMaxNC16HW16Int8(void poolfunc(int8_t*, int8_t*, size_t, size_t, size_t, size_t, size_t), const Tensor *src, Tensor *dst, int stridesx, int stridesy, int kernelx, int kernely, int paddingx, int paddingy)
{
    const int inputHeight = src->height();
    const int inputWidth = src->width();
    const int outputHeight = dst->height();
    const int outputWidth = dst->width();
    const int channel = dst->channel();
    const int batchsize = src->batch();
    int pack = 16;
    int thred0 = UP_DIV(paddingx, stridesx);
    int thred1 = inputWidth + paddingx - kernelx;
    thred1 = UP_DIV(thred1, stridesx);     // ix + kernelx >= inputWidth;

    const auto srcPtr = src->host<int8_t>();
    auto dstPtr       = dst->host<int8_t>();

    const int channel16 = UP_DIV(channel, pack);
    for (int oc = 0; oc < channel16; ++oc){
        for(int ob = 0; ob < batchsize; ++ob){
            for (int oy = 0; oy < outputHeight; ++oy) {
                
                int iy = oy * stridesy - paddingy;
                const int kernely_ = std::min(iy + kernely, inputHeight) - std::max(iy, 0);
                iy = std::max(iy, 0);
                int ox = 0;
                for (ox = 0; ox < thred0; ++ox) { // ix < 0;
                    int ix = ox * stridesx - paddingx;
                    const int kernelx_ = std::min(ix + kernelx, inputWidth) - std::max(ix, 0);
                    ix = std::max(ix, 0);

                    const int indexOutput = pack* (ox + outputWidth * (oy + outputHeight * (ob + batchsize * oc)));
                    const int indexInput = pack * (ix + inputWidth * (iy + inputHeight * (ob + batchsize * oc)));
                    int8_t* dstCur = dstPtr + indexOutput;
                    int8_t* srcCur = srcPtr + indexInput;

                    poolfunc(dstCur, srcCur, 1, inputWidth, kernelx_, kernely_, stridesx);

                } // ix < 0;

                // ix > 0 && ix + kernelx < inputWidth;
                if (thred1 - thred0 > 0) {
                    int ix = ox * stridesx - paddingx;
                    const int kernelx_ = std::min(ix + kernelx, inputWidth) - std::max(ix, 0);
                    ix = std::max(ix, 0);

                    const int indexOutput = pack * (ox + outputWidth * (oy + outputHeight * (ob + batchsize * oc)));    
                    const int indexInput = pack * (ix + inputWidth * (iy + inputHeight * (ob + batchsize * oc)));

                    int8_t* dstCur = dstPtr + indexOutput;
                    int8_t* srcCur = srcPtr + indexInput;

                    poolfunc(dstCur, srcCur, thred1 - thred0, inputWidth, kernelx_, kernely_, stridesx);
                }

                for (ox = thred1; ox < outputWidth; ++ox) { // ix + kernelx > inputWidth;
                    int ix = ox * stridesx - paddingx;
                    const int kernelx_ = std::min(ix + kernelx, inputWidth) - std::max(ix, 0);
                    ix = std::max(ix, 0);

                    const int indexOutput = pack* (ox + outputWidth * (oy + outputHeight * (ob + batchsize * oc)));
                    const int indexInput = pack * (ix + inputWidth * (iy + inputHeight * (ob + batchsize * oc)));
                    int8_t* dstCur = dstPtr + indexOutput;
                    int8_t* srcCur = srcPtr + indexInput;

                    poolfunc(dstCur, srcCur, 1, inputWidth, kernelx_, kernely_, stridesx);

                }
            }
        }
    }
}

CPUPoolInt8::CPUPoolInt8(Backend *backend, const Pool *parameter) : Execution(backend), mParameter(parameter) {
}

ErrorCode CPUPoolInt8::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    const auto input = inputs[0];
    auto output      = outputs[0];

    auto core = static_cast<CPUBackend*>(backend())->int8Functions();

    int strideWidth  = mParameter->strideX();
    int strideHeight = mParameter->strideY();
    int padWidth     = mParameter->padX();
    int padHeight    = mParameter->padY();
    int kernelWidth  = mParameter->kernelX();
    int kernelHeight = mParameter->kernelY();

    const int inputWidth   = input->width();
    const int inputHeight  = input->height();
    const int outputWidth  = output->width();
    const int outputHeight = output->height();

    kernelWidth  = std::min(kernelWidth, inputWidth);
    kernelHeight = std::min(kernelHeight, inputHeight);
    if (mParameter->isGlobal()) {
        kernelWidth  = inputWidth;
        kernelHeight = inputHeight;
        strideWidth  = inputWidth;
        strideHeight = inputHeight;
        padWidth     = 0;
        padHeight    = 0;
    }
    if (mParameter->padType() == PoolPadType_SAME) {
        int padNeededWidth  = (outputWidth - 1) * strideWidth + kernelWidth - inputWidth;
        int padNeededHeight = (outputHeight - 1) * strideHeight + kernelHeight - inputHeight;
        padWidth            = padNeededWidth > 0 ? padNeededWidth / 2 : 0;
        padHeight           = padNeededHeight > 0 ? padNeededHeight / 2 : 0;
    }

    const int channel = input->channel();
    
    mThreadFunction = [=](const Tensor *src, Tensor *dst) {
        poolingMaxNC16HW16Int8(core->MNNMaxPoolInt8, src, dst, strideWidth, strideHeight, kernelWidth, kernelHeight, padWidth, padHeight);
    };
    if (mParameter->type() == MNN::PoolType_AVEPOOL) {
        mThreadFunction = [=](const Tensor *src, Tensor *dst) {
            poolingAvgNC16HW16Int8(core->MNNAvgPoolInt8, src, dst, strideWidth, strideHeight, kernelWidth, kernelHeight, padWidth, padHeight);
        };
    }

    mInputTemp.reset(Tensor::createDevice<int8_t>({input->batch(), inputHeight, inputWidth, UP_DIV(channel, 16) * 16}));
    mOutputTemp.reset(Tensor::createDevice<int8_t>({output->batch(), outputHeight, outputWidth, UP_DIV(channel, 16) * 16}));

    bool allocSucc = backend()->onAcquireBuffer(mInputTemp.get(), Backend::DYNAMIC);
    allocSucc      = allocSucc && backend()->onAcquireBuffer(mOutputTemp.get(), Backend::DYNAMIC);
    if (!allocSucc) {
        return OUT_OF_MEMORY;
    }

    backend()->onReleaseBuffer(mInputTemp.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mOutputTemp.get(), Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode CPUPoolInt8::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input  = inputs[0];
    auto output = outputs[0];
    auto channel_input = input->channel();
    auto plane_in = input->width() * input->height() * input->batch();
    auto plane_out = output->width() * output->height() * output->batch();
    auto core = static_cast<CPUBackend*>(backend())->functions();
    auto depth = UP_DIV(channel_input, core->pack);
    
    if (core->pack == 8) {
        MNNPackC2Origin(mInputTemp.get()->host<double>(), input->host<double>(), plane_in, depth, plane_in);
        mThreadFunction(mInputTemp.get(), mOutputTemp.get());
        MNNUnpackC2Origin(output->host<double>(), mOutputTemp.get()->host<double>(), plane_out, depth, plane_out);
    }
    else if (core->pack == 4) {
        MNNPackC4Origin(mInputTemp.get()->host<float>(), input->host<float>(), plane_in, depth, plane_in);
        mThreadFunction(mInputTemp.get(), mOutputTemp.get());
        MNNUnpackC4Origin(output->host<float>(), mOutputTemp.get()->host<float>(), plane_out, depth, plane_out);
    }
    else if (core->pack == 16) {
        mThreadFunction(input, output);
    }
    return NO_ERROR;
}

class CPUPoolInt8Creator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        return new CPUPoolInt8(backend, op->main_as_Pool());
    }
};

REGISTER_CPU_OP_CREATOR(CPUPoolInt8Creator, OpType_PoolInt8);

} // namespace MNN
