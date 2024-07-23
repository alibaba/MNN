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
#include "backend/cpu/compute/ConvInt8TiledExecutor.hpp"

namespace MNN {

static Execution* _createUnit(const Tensor* input, const Tensor* output, Backend* backend,
                              const Convolution2D* conv2d, const float* originWeight, size_t originWeightSize, const float* bias, size_t biasSize, std::shared_ptr<ConvolutionCommon::Int8Common> weightQuantInfo, bool supportSparse, bool lowMemory) {
    auto cpuBackend = (CPUBackend*)backend;
    auto common = conv2d->common();
#ifdef MNN_USE_ONEDNN
    return OneDNN::createConvolution(common, backend, originWeight, originWeightSize, bias, biasSize);
#endif

#ifdef MNN_USE_SPARSE_COMPUTE
    if (conv2d->sparseParameter() && nullptr != weightQuantInfo.get()) {
        if (supportSparse && weightQuantInfo->quan->index() != nullptr) {
            return new SparseConvolutionTiledExecutor(common, backend, weightQuantInfo->quan,
                                                      conv2d->sparseParameter(), bias, biasSize);
        }
    }
#endif
    bool fastWay = common->kernelY() == 1 && common->kernelX() == 1
        && output->width() == input->width() && output->height() == input->height()
        && common->strideX() == 1 && common->strideY() == 1;

    if (lowMemory && nullptr != weightQuantInfo.get() && originWeightSize == 0) {
        if (cpuBackend->memoryMode() == BackendConfig::Memory_Low) {
            auto core = static_cast<CPUBackend*>(backend)->functions();
            auto resourceInt8 = CPUConvolution::makeResourceInt8(backend, conv2d, core->pack);
            return new DenseConvInt8TiledExecutor(backend, conv2d, resourceInt8, true);
        } else {
            return new DenseConvolutionTiledExecutor(common, backend, originWeight, originWeightSize, bias, biasSize, weightQuantInfo);
        }
    }
    if (fastWay && cpuBackend->functions()->matmulBytes == 0) {
        return new Convolution1x1Strassen(common, backend, originWeight, originWeightSize, bias, biasSize, weightQuantInfo);
    }
    if (originWeightSize == 0) {
        return new DenseConvolutionTiledExecutor(common, backend, originWeight, originWeightSize, bias, biasSize, weightQuantInfo);
    }
    if (!ConvolutionWinogradBridge::canUseWinograd(common)) {
        return new DenseConvolutionTiledExecutor(common, backend, originWeight, originWeightSize, bias, biasSize, nullptr);
    }
    PerfConfig convPerfconfig = DenseConvolutionTiledExecutor::bestTileConvolutionConfig(common, input, output, cpuBackend->threadNumber(), backend);
    auto winogradConfig = ConvolutionWinogradBridge::bestWinogradUnit(common, input, output, cpuBackend->threadNumber(), backend, convPerfconfig);
    if (winogradConfig.unit <= 1) {
        return new DenseConvolutionTiledExecutor(common, backend, originWeight, originWeightSize, bias, biasSize, nullptr);
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
#ifdef MNN_LOW_MEMORY
    bool lowMemory = static_cast<CPUBackend*>(backend)->memoryMode() != BackendConfig::Memory_High && static_cast<CPUBackend*>(backend)->functions()->MNNPackedMatMul_int8 != nullptr;
#else
    bool lowMemory = false;
#endif
    const float* originWeight = nullptr;
    const float* originBias   = nullptr;
    int originWeightSize   = 0;
    int originBiasSize     = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    std::unique_ptr<Tensor> externalWeightTensor, externalBiasTensor;
    bool supportSparse = false;
    auto core = static_cast<CPUBackend*>(backend)->functions();
    int bytes = core->bytes;
#ifdef MNN_USE_SPARSE_COMPUTE
#ifdef MNN_USE_SSE
    const bool onlySSENotAVX = core->pack == 4; // no backend of only sse without avx2 or avx512
#else
    const bool onlySSENotAVX = false;
#endif
    supportSparse = !onlySSENotAVX && bytes == 4;
#endif
    if (nullptr != conv2d->quanParameter()) {
        bool forceFloat = false;
        if (!supportSparse && conv2d->quanParameter()->index() != nullptr) {
            // The weight is storage as float sparse, but the backend don't support sparse compute, expand it
            forceFloat = true;
        }
        quanCommon = ConvolutionCommon::load(conv2d, backend, forceFloat, lowMemory);
        if (nullptr == quanCommon) {
            MNN_ERROR("Memory not Enough, can't extract IDST Convolution: %s \n", op->name()->c_str());
            return nullptr;
        }

        if (conv2d->quanParameter()->has_scaleInt()) {
            if (bytes < 4) {
                // From BF16 / FP16
                return nullptr;
            }
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
    if (nullptr == originWeight && nullptr != op->main_as_Convolution2D()->weight()) {
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
                           originBias, originBiasSize, quanCommon, supportSparse, lowMemory);
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
                        groupWeightSize, conv2d->bias()->data() + groupOutputCount * i, groupOutputCount, quanCommon, supportSparse, lowMemory);
        subConvolution.push_back(std::shared_ptr<Execution>(newConvolution));
    }
    return new ConvolutionGroup(backend, subConvolution);
}
} // namespace MNN
