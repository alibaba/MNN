//
//  CPUPool.cpp
//  MNN
//
//  Created by MNN on 2018/07/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "backend/cpu/CPUBackend.hpp"
#include "core/Concurrency.h"
#include "backend/cpu/CPUPool.hpp"
#include "compute/CommonOptFunction.h"
#include "math/Vec.hpp"
#include "core/TensorUtils.hpp"

using Vec4 = MNN::Math::Vec<float, 4>;
using Vec16 = MNN::Math::Vec<int8_t, 16>;

namespace MNN {

class CPUPool : public Execution {
public:
    CPUPool(Backend *b, const Pool *parameter, void* func, int bytes, bool returnRedice) : MNN::Execution(b), mParameter(parameter) {
        if(returnRedice){
            mComputeRedice = (decltype(mComputeRedice))func;
        }else{
            mCompute = (decltype(mCompute))func;
        }
        mBytes = bytes;
    }
    virtual ~CPUPool() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        auto layer       = mParameter;
        int strideWidth  = layer->strideX();
        int strideHeight = layer->strideY();
        int padWidth     = layer->padX();
        int padHeight    = layer->padY();
        auto core   = static_cast<CPUBackend*>(backend())->functions();
        MNN_ASSERT(DataType_DT_INT8 != TensorUtils::getDescribe(inputs[0])->type);
        
        // edit const if global
        auto input       = inputs[0];
        auto output      = outputs[0];
        int kernelWidth  = layer->kernelX();
        int kernelHeight = layer->kernelY();
        if (layer->isGlobal()) {
            kernelWidth  = input->width();
            kernelHeight = input->height();
            strideWidth  = input->width();
            strideHeight = input->height();
            padWidth     = 0;
            padHeight    = 0;
        }
        if (layer->padType() == PoolPadType_SAME) {
            int padNeededWidth  = (output->width() - 1) * strideWidth + kernelWidth - input->width();
            int padNeededHeight = (output->height() - 1) * strideHeight + kernelHeight - input->height();
            padWidth            = padNeededWidth > 0 ? padNeededWidth / 2 : 0;
            padHeight           = padNeededHeight > 0 ? padNeededHeight / 2 : 0;
        } else if (layer->padType() == PoolPadType_VALID) {
            padWidth = padHeight = 0;
        }
        auto totalDepth        = input->batch() * UP_DIV(input->channel(), core->pack);
        auto inputPlaneStride  = core->pack * input->width() * input->height();
        auto outputPlaneStride = core->pack * output->width() * output->height();
        int threadNumber       = ((CPUBackend *)backend())->threadNumber();
        auto padType           = layer->padType();
        auto countType         = layer->countType();
        if (layer->pads() != nullptr && padType == PoolPadType_CAFFE) {
            padType = PoolPadType_VALID;
        }
        if(outputs.size() == 2){
            mFunction = std::make_pair(threadNumber, [=](int tId) {
                for (int channel = (int)tId; channel < totalDepth; channel += threadNumber) {
                    auto inputData         = input->host<uint8_t>();
                    auto outputData        = output->host<uint8_t>();
                    auto rediceData        = outputs[1]->host<uint8_t>();
                    // run
                    mComputeRedice(inputData + channel * inputPlaneStride * mBytes, input->width(), input->height(),
                             outputData + outputPlaneStride * channel * mBytes, output->width(), output->height(), kernelWidth,
                             kernelHeight, strideWidth, strideHeight, padWidth, padHeight, padType, countType, rediceData + outputPlaneStride * channel * mBytes);
                }
            });
        }else{
            mFunction = std::make_pair(threadNumber, [=](int tId) {
                for (int channel = (int)tId; channel < totalDepth; channel += threadNumber) {
                    auto inputData         = input->host<uint8_t>();
                    auto outputData        = output->host<uint8_t>();
                    // run
                    mCompute(inputData + channel * inputPlaneStride * mBytes, input->width(), input->height(),
                             outputData + outputPlaneStride * channel * mBytes, output->width(), output->height(), kernelWidth,
                             kernelHeight, strideWidth, strideHeight, padWidth, padHeight, padType, countType);
                }
            });
        }
        return NO_ERROR;
    }
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        MNN_CONCURRENCY_BEGIN(tId, mFunction.first) {
            mFunction.second((int)tId);
        }
        MNN_CONCURRENCY_END();
        return NO_ERROR;
    }

private:
    const Pool *mParameter;
    void(*mCompute)(const void* channelInput, int inputWidth, int inputHeight, void *channelOutput,
                           int outputWidth, int outputHeight, int kernelWidth, int kernelHeight, int strideWidth,
                           int strideHeight, int padWidth, int padHeight, int padType, int countType);
    void(*mComputeRedice)(const void* channelInput, int inputWidth, int inputHeight, void *channelOutput,
                           int outputWidth, int outputHeight, int kernelWidth, int kernelHeight, int strideWidth,
                           int strideHeight, int padWidth, int padHeight, int padType, int countType, void *rediceOutput);
    std::pair<int, std::function<void(int)> > mFunction;
    int mBytes;
};
class CPUPoolCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        void* func = nullptr;
        bool returnRedice = false;
        if (inputs[0]->getType() == halide_type_of<int8_t>()) {
            if (op->main_as_Pool()->type() == PoolType_AVEPOOL) {
                func = (void*)(poolingAvg<int8_t, Vec16, 4>);
            } else {
                func = (void*)(poolingMax<int8_t, Vec16, 4, -128>);
            }
            return new CPUPool(backend, op->main_as_Pool(), func, 1, returnRedice);
        }
        auto core = static_cast<CPUBackend*>(backend)->functions();
        if (op->main_as_Pool()->type() == PoolType_AVEPOOL) {
            func = (void*)(core->MNNPoolingAvg);
        } else {
            func = (void*)(core->MNNPoolingMax);
            if(outputs.size() == 2){
                func = (void*)(core->MNNPoolingMaxWithRedice);
                returnRedice = true;
            }
        }
        return new CPUPool(backend, op->main_as_Pool(), func, core->bytes, returnRedice);
    }
};

REGISTER_CPU_OP_CREATOR(CPUPoolCreator, OpType_Pooling);

} // namespace MNN
