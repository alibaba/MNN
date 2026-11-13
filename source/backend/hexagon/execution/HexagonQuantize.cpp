#include "HexagonQuantize.hpp"

#include "HexagonBackend.hpp"
#include "HexagonRuntime.hpp"
#include "MNN_generated.h"
#include "core/TensorUtils.hpp"
#include "dsp/int8_cnn.h"
#include "htp_command.h"

#include <cmath>
#include <cstdint>
#include <limits>

namespace MNN {
namespace {

bool readQuantInfo(const Tensor* tensor, float* scale, int32_t* zero) {
    if (tensor == nullptr || scale == nullptr || zero == nullptr ||
        TensorUtils::getDescribe(tensor)->quantAttr == nullptr) {
        return false;
    }
    const auto info = TensorUtils::getQuantInfo(tensor);
    if (info.size() < 2 || !std::isfinite(info[0]) || info[0] <= 0.0f ||
        !std::isfinite(info[1]) || info[1] != 0.0f) {
        return false;
    }
    *scale = info[0];
    *zero = 0;
    return true;
}

bool supportsQuantBoundary(const HexagonRuntime* runtime) {
    return runtime != nullptr &&
           (runtime->info().capabilities & DSP_CAP_CONV1X1_W8A8) != 0;
}

bool validTensors(const std::vector<Tensor*>& inputs,
                  const std::vector<Tensor*>& outputs, bool quantize,
                  const HexagonBackend* backend, int32_t* size) {
    if (backend == nullptr || size == nullptr || inputs.size() != 1 ||
        outputs.size() != 1 || inputs[0] == nullptr ||
        outputs[0] == nullptr ||
        !HexagonQuantize::validTensorContract(inputs[0], outputs[0],
                                              quantize)) {
        return false;
    }
    const size_t inputSize = backend->getElementSize(inputs[0]);
    const size_t outputSize = backend->getElementSize(outputs[0]);
    if (inputSize != outputSize ||
        inputSize > static_cast<size_t>(
                        std::numeric_limits<int32_t>::max())) {
        return false;
    }
    *size = static_cast<int32_t>(inputSize);
    return true;
}

} // namespace

HexagonQuantize::HexagonQuantize(Backend* backend, bool quantize, float scale,
                                 int32_t zero)
    : HexagonExecution(backend),
      mQuantize(quantize),
      mScale(scale),
      mZero(zero) {
}

bool HexagonQuantize::validTensorContract(const Tensor* input,
                                          const Tensor* output,
                                          bool quantize) {
    if (input == nullptr || output == nullptr ||
        input->getType().code != halide_type_float ||
        output->getType().code != halide_type_float) {
        return false;
    }
    const auto inputDescribe = TensorUtils::getDescribe(input);
    const auto outputDescribe = TensorUtils::getDescribe(output);
    if (quantize) {
        return !inputDescribe->applyQuant && outputDescribe->applyQuant &&
               outputDescribe->quantAttr != nullptr &&
               HexagonBackend::getBytes(input) == 2 &&
               HexagonBackend::getBytes(output) == 1;
    }
    return inputDescribe->applyQuant &&
           inputDescribe->quantAttr != nullptr &&
           !outputDescribe->applyQuant &&
           HexagonBackend::getBytes(input) == 1 &&
           HexagonBackend::getBytes(output) == 2;
}

HexagonQuantize* HexagonQuantize::create(
    Backend* backend, const Op* op, const std::vector<Tensor*>& inputs,
    const std::vector<Tensor*>& outputs) {
    if (backend == nullptr || op == nullptr ||
        (op->type() != OpType_FloatToInt8 &&
         op->type() != OpType_Int8ToFloat) ||
        HexagonRuntime::getDstFunctions() == nullptr) {
        return nullptr;
    }
    const auto runtime =
        static_cast<const HexagonRuntime*>(backend->getRuntime());
    if (!supportsQuantBoundary(runtime)) {
        return nullptr;
    }
    const bool quantize = op->type() == OpType_FloatToInt8;
    int32_t size = 0;
    if (!validTensors(inputs, outputs, quantize,
                      static_cast<HexagonBackend*>(backend), &size)) {
        return nullptr;
    }
    float scale = 0.0f;
    int32_t zero = 0;
    if (!readQuantInfo(quantize ? outputs[0] : inputs[0], &scale, &zero)) {
        return nullptr;
    }
    return new HexagonQuantize(backend, quantize, scale, zero);
}

ErrorCode HexagonQuantize::onBuildCmd(
    const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
    std::vector<HexagonCommand>& dst) {
    auto hexBackend = static_cast<HexagonBackend*>(backend());
    const auto runtime =
        static_cast<const HexagonRuntime*>(hexBackend->getRuntime());
    int32_t size = 0;
    if (!supportsQuantBoundary(runtime) ||
        !validTensors(inputs, outputs, mQuantize, hexBackend, &size)) {
        return NOT_SUPPORT;
    }
    float scale = 0.0f;
    int32_t zero = 0;
    if (!readQuantInfo(mQuantize ? outputs[0] : inputs[0], &scale, &zero) ||
        scale != mScale || zero != mZero) {
        return NOT_SUPPORT;
    }
    const auto srcDev = HexagonBackend::getDevicePtr(inputs[0]);
    const auto dstDev = HexagonBackend::getDevicePtr(outputs[0]);
    const size_t srcBytes = hexBackend->getSize(inputs[0]);
    const size_t dstBytes = hexBackend->getSize(outputs[0]);
    if (srcDev.first <= 0 || dstDev.first <= 0 ||
        srcBytes > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        dstBytes > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return NOT_SUPPORT;
    }

    const auto tensor = mQuantize ? inputs[0] : outputs[0];
    DSPQuantizeParam params = {
        size,
        HexagonBackend::getBytes(tensor),
        scale,
        zero,
        tensor->batch(),
        tensor->channel(),
        tensor->height() * tensor->width(),
        runtime->info().vectorSize,
    };
    dst.emplace_back();
    dst.back().build(
        hexBackend,
        mQuantize ? DSP_OP_QUANTIZE_I8 : DSP_OP_DEQUANTIZE_I8,
        &params, sizeof(params), {srcDev}, {dstDev}, inputs, outputs,
        {static_cast<int>(srcBytes)}, {static_cast<int>(dstBytes)});
    return NO_ERROR;
}

bool HexagonQuantize::onClone(Backend* backend, const Op* op,
                              Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (dst == nullptr) {
        return true;
    }
    *dst = new HexagonQuantize(backend, mQuantize, mScale, mZero);
    return true;
}

} // namespace MNN
