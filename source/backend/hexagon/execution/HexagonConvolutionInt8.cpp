#include "HexagonConvolutionInt8.hpp"

#include "HexagonBackend.hpp"
#include "HexagonQuantization.hpp"
#include "HexagonRuntime.hpp"
#include "core/ConvolutionCommon.hpp"
#include "core/TensorUtils.hpp"
#include "dsp/int8_cnn.h"
#include "htp_command.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace MNN {
namespace {

bool validRuntimeTensors(const std::vector<Tensor*>& inputs,
                         const std::vector<Tensor*>& outputs,
                         int inputChannels, int outputChannels) {
    if (inputs.size() != 1 || outputs.size() != 1 ||
        inputs[0] == nullptr || outputs[0] == nullptr ||
        HexagonBackend::getBytes(inputs[0]) != 1 ||
        HexagonBackend::getBytes(outputs[0]) != 1 ||
        inputs[0]->dimensions() != 4 || outputs[0]->dimensions() != 4 ||
        inputs[0]->channel() != inputChannels ||
        outputs[0]->channel() != outputChannels ||
        inputs[0]->batch() != outputs[0]->batch() ||
        inputs[0]->height() != outputs[0]->height() ||
        inputs[0]->width() != outputs[0]->width()) {
        return false;
    }
    const auto inputDescribe = TensorUtils::getDescribe(inputs[0]);
    const auto outputDescribe = TensorUtils::getDescribe(outputs[0]);
    return inputDescribe != nullptr && outputDescribe != nullptr &&
           inputDescribe->quantAttr != nullptr &&
           outputDescribe->quantAttr != nullptr &&
           inputDescribe->quantAttr->type == DataType_DT_INT8 &&
           outputDescribe->quantAttr->type == DataType_DT_INT8 &&
           inputDescribe->quantAttr->zero == 0.0f &&
           outputDescribe->quantAttr->zero == 0.0f &&
           std::isfinite(inputDescribe->quantAttr->scale) &&
           inputDescribe->quantAttr->scale > 0.0f &&
           std::isfinite(outputDescribe->quantAttr->scale) &&
           outputDescribe->quantAttr->scale > 0.0f;
}

} // namespace

HexagonConvolutionInt8::Resource::~Resource() {
    if (allocator == nullptr) {
        return;
    }
    if (weight.first != nullptr) {
        allocator->free(weight);
    }
    if (bias.first != nullptr) {
        allocator->free(bias);
    }
    if (requant.first != nullptr) {
        allocator->free(requant);
    }
}

HexagonConvolutionInt8::HexagonConvolutionInt8(
    Backend* backend, std::shared_ptr<Resource> resource)
    : HexagonExecution(backend), mResource(std::move(resource)) {
}

bool HexagonConvolutionInt8::validTensorContract(
    const Op* op, const std::vector<Tensor*>& inputs,
    const std::vector<Tensor*>& outputs, int channelPack) {
    return channelPack > 0 && channelPack % 32 == 0 &&
           supportsW8A8Conv1x1(op, inputs, outputs) &&
           HexagonBackend::getBytes(inputs[0]) == 1 &&
           HexagonBackend::getBytes(outputs[0]) == 1;
}

ErrorCode HexagonConvolutionInt8::onBuildCmd(
    const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
    std::vector<HexagonCommand>& dst) {
    const auto runtime =
        static_cast<const HexagonRuntime*>(backend()->getRuntime());
    if (runtime == nullptr ||
        (runtime->info().capabilities & DSP_CAP_CONV1X1_W8A8) == 0 ||
        mResource == nullptr ||
        runtime->info().vectorSize != mResource->channelPack ||
        !validRuntimeTensors(inputs, outputs, mResource->inputChannels,
                             mResource->outputChannels)) {
        mValid = false;
        return NOT_SUPPORT;
    }

    DSPConv2DInt8Param params = {};
    params.batch = inputs[0]->batch();
    params.inputHeight = inputs[0]->height();
    params.inputWidth = inputs[0]->width();
    params.inputChannels = inputs[0]->channel();
    params.outputHeight = outputs[0]->height();
    params.outputWidth = outputs[0]->width();
    params.outputChannels = outputs[0]->channel();
    params.kernelY = 1;
    params.kernelX = 1;
    params.strideY = 1;
    params.strideX = 1;
    params.dilateY = 1;
    params.dilateX = 1;
    params.groups = 1;
    params.channelPack = mResource->channelPack;
    params.relu = mResource->relu;
    params.relu6 = mResource->relu6;

    const auto input = HexagonBackend::getDevicePtr(inputs[0]);
    const auto output = HexagonBackend::getDevicePtr(outputs[0]);
    const auto weight = HexagonBackend::getDevicePtr(mResource->weight);
    const auto bias = HexagonBackend::getDevicePtr(mResource->bias);
    const auto requant = HexagonBackend::getDevicePtr(mResource->requant);
    auto hexagonBackend = static_cast<HexagonBackend*>(backend());
    const size_t inputBytes = hexagonBackend->getSize(inputs[0]);
    const size_t outputBytes = hexagonBackend->getSize(outputs[0]);
    if (input.first < 0 || output.first < 0 || weight.first < 0 ||
        bias.first < 0 || requant.first < 0 ||
        inputBytes > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        outputBytes > static_cast<size_t>(std::numeric_limits<int>::max())) {
        mValid = false;
        return NOT_SUPPORT;
    }

    dst.emplace_back();
    dst.back().build(
        static_cast<HexagonBackend*>(backend()), DSP_OP_CONV2D_W8A8,
        &params, sizeof(params), {input, weight, bias, requant}, {output},
        inputs, outputs,
        {static_cast<int>(inputBytes),
         mResource->weightBytes, mResource->biasBytes,
         mResource->requantBytes},
        {static_cast<int>(outputBytes)});
    return NO_ERROR;
}

bool HexagonConvolutionInt8::onClone(Backend* backend, const Op* op,
                                     Execution** dst) {
    if (!mValid || mResource == nullptr) {
        return false;
    }
    if (dst == nullptr) {
        return true;
    }
    *dst = new HexagonConvolutionInt8(backend, mResource);
    return true;
}

HexagonConvolutionInt8* HexagonConvolutionInt8::create(
    Backend* backend, const Op* op, const std::vector<Tensor*>& inputs,
    const std::vector<Tensor*>& outputs) {
    if (backend == nullptr) {
        return nullptr;
    }
    const auto runtime =
        static_cast<const HexagonRuntime*>(backend->getRuntime());
    if (runtime == nullptr ||
        (runtime->info().capabilities & DSP_CAP_CONV1X1_W8A8) == 0 ||
        !validTensorContract(op, inputs, outputs,
                             runtime->info().vectorSize)) {
        return nullptr;
    }

    const auto conv = op->main_as_Convolution2D();
    const int inputChannels = inputs[0]->channel();
    const int outputChannels = outputs[0]->channel();
    auto quant = ConvolutionCommon::load(
        op, backend, false, true, nullptr);
    if (quant == nullptr || quant->asymmetric ||
        quant->weight.get() == nullptr ||
        static_cast<size_t>(quant->weight.size()) <
            static_cast<size_t>(inputChannels) * outputChannels ||
        quant->alphaSize != outputChannels) {
        return nullptr;
    }
    const float* weightScale = quant->getAlphaFloat();
    if (weightScale == nullptr) {
        return nullptr;
    }

    std::vector<int8_t> packedWeight;
    if (!packW8A8Conv1x1Weight(
            packedWeight, quant->weight.get(), quant->weight.size(),
            inputChannels, outputChannels)) {
        return nullptr;
    }

    const float* bias =
        conv->bias() == nullptr ? nullptr : conv->bias()->data();
    const int biasCount =
        conv->bias() == nullptr ? 0 : conv->bias()->size();
    const auto inputAttr =
        TensorUtils::getDescribe(inputs[0])->quantAttr;
    const auto outputAttr =
        TensorUtils::getDescribe(outputs[0])->quantAttr;
    std::vector<int32_t> biasInt32;
    std::vector<DSPRequantParam> requant;
    if (!buildW8A8Conv1x1PostParams(
            biasInt32, requant, bias, biasCount, weightScale,
            quant->alphaSize, outputChannels, inputAttr->scale,
            outputAttr->scale, conv->common()->relu(),
            conv->common()->relu6(),
            static_cast<int32_t>(std::nearbyint(outputAttr->min)),
            static_cast<int32_t>(std::nearbyint(outputAttr->max)))) {
        return nullptr;
    }

    if (packedWeight.size() >
            static_cast<size_t>(std::numeric_limits<int>::max()) ||
        biasInt32.size() >
            static_cast<size_t>(std::numeric_limits<int>::max()) /
                sizeof(int32_t) ||
        requant.size() >
            static_cast<size_t>(std::numeric_limits<int>::max()) /
                sizeof(DSPRequantParam)) {
        return nullptr;
    }

    auto hexagonBackend = static_cast<HexagonBackend*>(backend);
    auto allocator = hexagonBackend->getAllocator(2);
    if (allocator == nullptr) {
        return nullptr;
    }
    std::shared_ptr<Resource> resource(new Resource);
    resource->allocator = allocator;
    resource->inputChannels = inputChannels;
    resource->outputChannels = outputChannels;
    resource->weightBytes = static_cast<int>(packedWeight.size());
    resource->biasBytes =
        static_cast<int>(biasInt32.size() * sizeof(int32_t));
    resource->requantBytes =
        static_cast<int>(requant.size() * sizeof(DSPRequantParam));
    resource->channelPack = runtime->info().vectorSize;
    resource->relu = conv->common()->relu() ? 1 : 0;
    resource->relu6 = conv->common()->relu6() ? 1 : 0;
    resource->weight = allocator->alloc(packedWeight.size());
    resource->bias =
        allocator->alloc(biasInt32.size() * sizeof(int32_t));
    resource->requant =
        allocator->alloc(requant.size() * sizeof(DSPRequantParam));
    if (resource->weight.first == nullptr ||
        resource->bias.first == nullptr ||
        resource->requant.first == nullptr) {
        return nullptr;
    }

    std::memcpy(HexagonBackend::getPtr(resource->weight),
                packedWeight.data(), packedWeight.size());
    std::memcpy(HexagonBackend::getPtr(resource->bias),
                biasInt32.data(), biasInt32.size() * sizeof(int32_t));
    std::memcpy(HexagonBackend::getPtr(resource->requant),
                requant.data(),
                requant.size() * sizeof(DSPRequantParam));
    hexagonBackend->markHostInput(
        resource->weight, static_cast<int>(packedWeight.size()));
    hexagonBackend->markHostInput(
        resource->bias,
        static_cast<int>(biasInt32.size() * sizeof(int32_t)));
    hexagonBackend->markHostInput(
        resource->requant,
        static_cast<int>(requant.size() * sizeof(DSPRequantParam)));
    return new HexagonConvolutionInt8(backend, std::move(resource));
}

} // namespace MNN
