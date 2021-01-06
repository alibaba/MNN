//
//  ConvolutionFloatFactory.cpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/compute/ConvolutionFloatFactory.h"
#include "backend/cpu/CPUConvolutionDepthwise.hpp"
#include "backend/cpu/compute/ConvOpt.h"
#include "backend/cpu/compute/Convolution1x1Strassen.hpp"
#include "backend/cpu/compute/ConvolutionGroup.hpp"
#include "backend/cpu/compute/ConvolutionIntFactory.hpp"
#include "backend/cpu/compute/ConvolutionTiledExecutor.hpp"
#include "backend/cpu/compute/ConvolutionWinograd.hpp"
#include "core/Macro.h"
#include "backend/cpu/OneDNNConvolution.hpp"

namespace MNN {

static Execution* _createUnit(const Tensor* input, const Tensor* output, Backend* backend,
                              const Convolution2DCommon* common, const float* originWeight, size_t originWeightSize,
                              const float* bias, size_t biasSize) {
#ifdef MNN_USE_ONEDNN
    return OneDNN::createConvolution(common, backend, originWeight, originWeightSize, bias, biasSize);
#endif
    auto layer   = common;
    bool fastWay = layer->kernelY() == 1 && layer->kernelX() == 1;
    if (fastWay) {
        return new Convolution1x1Strassen(common, backend, originWeight, originWeightSize, bias, biasSize);
    }
    if (!ConvolutionWinograd::canUseWinograd(common)) {
        return new ConvolutionTiledExecutor(common, backend, originWeight, originWeightSize, bias, biasSize);
    }
    auto cpuBackend = (CPUBackend*)backend;
    if (cpuBackend->memoryMode() == BackendConfig::Memory_Low) {
        return new ConvolutionTiledExecutor(common, backend, originWeight, originWeightSize, bias, biasSize);
    }
    auto unit = ConvolutionWinograd::bestWinogradUnit(common, input, output, cpuBackend->threadNumber());
    if (unit <= 1) {
        return new ConvolutionTiledExecutor(common, backend, originWeight, originWeightSize, bias, biasSize);
    }
    return new ConvolutionWinograd(common, input, output, backend, originWeight, originWeightSize, bias, biasSize,
                                   unit);
}

Execution* ConvolutionFloatFactory::create(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                           const MNN::Op* op, Backend* backend) {
    auto conv2d = op->main_as_Convolution2D();
    if (inputs.empty()) {
        // Create Default Inputs and Outputs
        std::shared_ptr<Tensor> tempInput;
        std::shared_ptr<Tensor> tempOutput;
        auto common = conv2d->common();
        int ow = 2, oh = 2;
        int iw = (common->kernelX() - 1) * common->dilateX() + common->strideX() * (ow - 1) + 1;
        int ih = (common->kernelY() - 1) * common->dilateY() + common->strideY() * (oh - 1) + 1;
        tempInput.reset(Tensor::createDevice<float>({1, conv2d->common()->inputCount(), ih, iw}, Tensor::CAFFE_C4));
        tempOutput.reset(Tensor::createDevice<float>({1, conv2d->common()->outputCount(), oh, ow}, Tensor::CAFFE_C4));
        return create({tempInput.get()}, {tempOutput.get()}, op, backend);
    }
    const float* originWeight = nullptr;
    size_t originWeightSize   = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    if (nullptr != conv2d->quanParameter()) {
        quanCommon = ConvolutionCommon::load(conv2d->quanParameter());
        if (nullptr == quanCommon) {
            MNN_ERROR("Memory not Enough, can't extract IDST Convolution: %s \n", op->name()->c_str());
            return nullptr;
        }
        if (quanCommon->weightFloat.get() == nullptr) {
            return ConvolutionIntFactory::create(inputs[0], outputs[0], op, backend, quanCommon.get());
        }
        // Back to float
        originWeight     = quanCommon->weightFloat.get();
        originWeightSize = quanCommon->weightFloat.size();
    } else if (nullptr == conv2d->weight() || nullptr == conv2d->bias()) {
        MNN_ERROR("%s has no weight or bias. The model may be benchmark model, please revert the weight/bias firstly\n", op->name()->c_str());
        return nullptr;
    }
    auto common = conv2d->common();
    if (nullptr == originWeight) {
        originWeight     = op->main_as_Convolution2D()->weight()->data();
        originWeightSize = op->main_as_Convolution2D()->weight()->size();
    }

    if (1 == common->group()) {
        return _createUnit(inputs[0], outputs[0], backend, common, originWeight, originWeightSize,
                           conv2d->bias()->data(), conv2d->bias()->size());
    }
    // Split
    std::vector<std::shared_ptr<Execution>> subConvolution;
    auto group            = common->group();
    auto groupOutputCount = common->outputCount() / group;
    auto groupWeightSize  = originWeightSize / group;
    std::shared_ptr<Tensor> emptyInput(Tensor::createDevice<float>(inputs[0]->shape(), Tensor::CAFFE));
    std::shared_ptr<Tensor> emptyOutput(Tensor::createDevice<float>(outputs[0]->shape(), Tensor::CAFFE));
    emptyInput->setLength(1, inputs[0]->channel() / group);
    emptyOutput->setLength(1, outputs[0]->channel() / group);
    for (int i = 0; i < group; ++i) {
        auto newConvolution =
            _createUnit(emptyInput.get(), emptyOutput.get(), backend, common, originWeight + groupWeightSize * i,
                        groupWeightSize, conv2d->bias()->data() + groupOutputCount * i, groupOutputCount);
        subConvolution.push_back(std::shared_ptr<Execution>(newConvolution));
    }
    return new ConvolutionGroup(backend, subConvolution);
}
} // namespace MNN
