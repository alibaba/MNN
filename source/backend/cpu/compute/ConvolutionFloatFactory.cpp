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

#include "backend/cpu/compute/ConvolutionWinogradBridge.hpp"
#include "backend/cpu/compute/DenseConvolutionTiledExecutor.hpp"
#ifdef MNN_USE_SPARSE_COMPUTE
#include "backend/cpu/compute/SparseConvolutionTiledExecutor.hpp"
#endif
#include "core/Macro.h"
#include "core/OpCommonUtils.hpp"
#include "backend/cpu/OneDNNConvolution.hpp"

namespace MNN {

static Execution* _createUnit(const Tensor* input, const Tensor* output, Backend* backend,
                              const Convolution2D* conv2d, const float* originWeight, size_t originWeightSize,
                              const float* bias, size_t biasSize) {
    auto common = conv2d->common();
#ifdef MNN_USE_ONEDNN
    return OneDNN::createConvolution(common, backend, originWeight, originWeightSize, bias, biasSize);
#endif

#ifdef MNN_USE_SPARSE_COMPUTE

    auto core = static_cast<CPUBackend*>(backend)->functions();
    int bytes = core->bytes;
#ifdef MNN_USE_SSE
    const bool onlySSENotAVX = core->pack == 4; // no backend of only sse without avx2 or avx512
#else
    const bool onlySSENotAVX = false;
#endif
    if (!onlySSENotAVX && bytes == 4 && conv2d->sparseParameter()) {
        if (SparseConvolutionTiledExecutor::shouldUseSparseConvolution(originWeightSize, conv2d->sparseParameter())) {
            return new SparseConvolutionTiledExecutor(common, backend, originWeight, originWeightSize,
                                                      conv2d->sparseParameter(), bias, biasSize);
        }
    }

#endif
    bool fastWay = common->kernelY() == 1 && common->kernelX() == 1
        && output->width() == input->width() && output->height() == input->height()
        && common->strideX() == 1 && common->strideY() == 1;
    if (fastWay) {
        return new Convolution1x1Strassen(common, backend, originWeight, originWeightSize, bias, biasSize);
    }
    if (!ConvolutionWinogradBridge::canUseWinograd(common)) {
        return new DenseConvolutionTiledExecutor(common, backend, originWeight, originWeightSize, bias, biasSize);
    }
    auto cpuBackend = (CPUBackend*)backend;
    if (cpuBackend->memoryMode() == BackendConfig::Memory_Low) {
        return new DenseConvolutionTiledExecutor(common, backend, originWeight, originWeightSize, bias, biasSize);
    }
    PerfConfig convPerfconfig = DenseConvolutionTiledExecutor::bestTileConvolutionConfig(common, input, output, cpuBackend->threadNumber(), backend);
    auto winogradConfig = ConvolutionWinogradBridge::bestWinogradUnit(common, input, output, cpuBackend->threadNumber(), backend, convPerfconfig);
    if (winogradConfig.unit <= 1) {
        return new DenseConvolutionTiledExecutor(common, backend, originWeight, originWeightSize, bias, biasSize);
    }
    return ConvolutionWinogradBridge::createWinogradImpl(common, input, output, backend, originWeight, originWeightSize, bias, biasSize,
                                   winogradConfig);
}

Execution* ConvolutionFloatFactory::create(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                           const MNN::Op* op, Backend* backend) {
    auto conv2d = op->main_as_Convolution2D();
    if (inputs.size() > 1) {
        // Multi Input
        return new ConvolutionTiledExecutorMultiInput(conv2d->common(), backend);
    }
    const float* originWeight = nullptr;
    const float* originBias   = nullptr;
    int originWeightSize   = 0;
    int originBiasSize     = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    std::unique_ptr<Tensor> externalWeightTensor, externalBiasTensor;
    if (nullptr != conv2d->quanParameter()) {
        quanCommon = ConvolutionCommon::load(conv2d->quanParameter());
        if (nullptr == quanCommon) {
            MNN_ERROR("Memory not Enough, can't extract IDST Convolution: %s \n", op->name()->c_str());
            return nullptr;
        }

        if (quanCommon->weightFloat.get() == nullptr) {
            if (backend->type() != MNN_FORWARD_CPU) {
                // From BF16
                return nullptr;
            }
            return ConvolutionIntFactory::create(inputs[0], outputs[0], op, backend, quanCommon.get());
        }
        // Back to float
        originWeight     = quanCommon->weightFloat.get();
        originWeightSize = quanCommon->weightFloat.size();
    } else if (USE_EXTERNAL_DATA(conv2d)) {
        bool res = OpCommonUtils::loadConvData(backend, op, externalWeightTensor, externalBiasTensor, originWeightSize, originBiasSize);
        if (!res) {
            MNN_ERROR("%s load external weight or bias failed.", op->name()->c_str());
            return nullptr;
        }
        originWeight = externalWeightTensor->host<float>();
        originBias = externalBiasTensor->host<float>();
    } else if (nullptr == conv2d->weight() || nullptr == conv2d->bias()) {
        MNN_ERROR("%s has no weight or bias. The model may be benchmark model, please revert the weight/bias firstly\n", op->name()->c_str());
        return nullptr;
    }
    auto common = conv2d->common();
    if (nullptr == originWeight) {
        originWeight     = op->main_as_Convolution2D()->weight()->data();
        originWeightSize = op->main_as_Convolution2D()->weight()->size();
    }
    if (nullptr == originBias) {
        originBias     = op->main_as_Convolution2D()->bias()->data();
        originBiasSize = op->main_as_Convolution2D()->bias()->size();
    }

    int group            = common->group();
    if (common->inputCount() != inputs[0]->channel() && common->inputCount() > 0) {
        group = inputs[0]->channel()/ conv2d->common()->inputCount();
    }
    MNN_ASSERT(group > 0);
    if (1 == group) {
        return _createUnit(inputs[0], outputs[0], backend, conv2d, originWeight, originWeightSize,
                           originBias, originBiasSize);
    }
    // TODO: Use Geometry to split
    // Split
    std::vector<std::shared_ptr<Execution>> subConvolution;
    auto groupOutputCount = common->outputCount() / group;
    auto groupWeightSize  = originWeightSize / group;
    std::shared_ptr<Tensor> emptyInput(Tensor::createDevice<float>(inputs[0]->shape(), Tensor::CAFFE_C4));
    std::shared_ptr<Tensor> emptyOutput(Tensor::createDevice<float>(outputs[0]->shape(), Tensor::CAFFE_C4));
    emptyInput->setLength(1, inputs[0]->channel() / group);
    emptyOutput->setLength(1, outputs[0]->channel() / group);
    for (int i = 0; i < group; ++i) {
        auto newConvolution =
            _createUnit(emptyInput.get(), emptyOutput.get(), backend, conv2d, originWeight + groupWeightSize * i,
                        groupWeightSize, conv2d->bias()->data() + groupOutputCount * i, groupOutputCount);
        subConvolution.push_back(std::shared_ptr<Execution>(newConvolution));
    }
    return new ConvolutionGroup(backend, subConvolution);
}
} // namespace MNN
