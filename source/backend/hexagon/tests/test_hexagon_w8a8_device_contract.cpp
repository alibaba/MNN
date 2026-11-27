#include "HexagonBackend.hpp"
#include "HexagonCommand.hpp"
#include "HexagonExecutionFactory.hpp"
#include "HexagonQuantization.hpp"
#include "HexagonRuntime.hpp"
#include "MNN_generated.h"
#include "core/TensorUtils.hpp"

#include <MNN/Interpreter.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

#define CHECK(condition)                                                                            \
    do {                                                                                            \
        if (!(condition)) {                                                                         \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);                        \
            return false;                                                                           \
        }                                                                                           \
    } while (0)

constexpr int kInputChannels = 5;
constexpr int kMiddleChannels = 7;
constexpr int kOutputChannels = 3;
constexpr int kArea = 2;
constexpr float kInputScale = 0.25f;
constexpr float kMiddleScale = 0.5f;
constexpr float kOutputScale = 1.0f;
constexpr float kWeightScale1 = 0.125f;
constexpr float kWeightScale2 = 0.25f;

std::unique_ptr<MNN::TensorDescribeT> makeQuantDescribe(int index, float scale) {
    std::unique_ptr<MNN::TensorDescribeT> describe(new MNN::TensorDescribeT);
    describe->index = index;
    describe->quantInfo.reset(new MNN::TensorQuantInfoT);
    describe->quantInfo->scale = scale;
    describe->quantInfo->zero = 0.0f;
    describe->quantInfo->min = -127.0f;
    describe->quantInfo->max = 127.0f;
    describe->quantInfo->type = MNN::DataType_DT_INT8;
    return describe;
}

std::unique_ptr<MNN::OpT> makeConv(const std::string& name, int inputIndex,
                                   int outputIndex, int inputChannels,
                                   int outputChannels,
                                   const std::vector<int8_t>& weight,
                                   const std::vector<float>& bias,
                                   float weightScale) {
    std::unique_ptr<MNN::OpT> op(new MNN::OpT);
    op->name = name;
    op->type = MNN::OpType_Convolution;
    op->main.type = MNN::OpParameter_Convolution2D;
    op->main.value = new MNN::Convolution2DT;
    op->inputIndexes = {inputIndex};
    op->outputIndexes = {outputIndex};
    op->defaultDimentionFormat = MNN::MNN_DATA_FORMAT_NC4HW4;

    auto conv = op->main.AsConvolution2D();
    conv->common.reset(new MNN::Convolution2DCommonT);
    conv->common->inputCount = inputChannels;
    conv->common->outputCount = outputChannels;
    conv->common->kernelX = 1;
    conv->common->kernelY = 1;
    conv->common->strideX = 1;
    conv->common->strideY = 1;
    conv->common->dilateX = 1;
    conv->common->dilateY = 1;
    conv->common->group = 1;
    conv->common->padMode = MNN::PadMode_CAFFE;
    conv->bias = bias;
    conv->quanParameter.reset(new MNN::IDSTQuanT);
    conv->quanParameter->buffer.assign(weight.begin(), weight.end());
    conv->quanParameter->alpha.assign(outputChannels, weightScale);
    conv->quanParameter->type = 4;
    conv->quanParameter->quantScale = 1.0f;
    conv->quanParameter->aMaxOrBits = 8;
    conv->quanParameter->aMin = -127;
    conv->quanParameter->readType = 0;
    return op;
}

std::unique_ptr<MNN::NetT> makeChainedModel(
    std::vector<int8_t>& weight1, std::vector<float>& bias1,
    std::vector<int8_t>& weight2, std::vector<float>& bias2) {
    weight1.resize(kMiddleChannels * kInputChannels);
    bias1.resize(kMiddleChannels);
    for (int oc = 0; oc < kMiddleChannels; ++oc) {
        bias1[oc] = static_cast<float>((oc % 3) - 1) *
                    kInputScale * kWeightScale1;
        for (int ic = 0; ic < kInputChannels; ++ic) {
            weight1[static_cast<size_t>(oc) * kInputChannels + ic] =
                static_cast<int8_t>((oc * 3 + ic * 2) % 5 - 2);
        }
    }
    weight2.resize(kOutputChannels * kMiddleChannels);
    bias2.resize(kOutputChannels);
    for (int oc = 0; oc < kOutputChannels; ++oc) {
        bias2[oc] = static_cast<float>(oc - 1) *
                    kMiddleScale * kWeightScale2;
        for (int ic = 0; ic < kMiddleChannels; ++ic) {
            weight2[static_cast<size_t>(oc) * kMiddleChannels + ic] =
                static_cast<int8_t>((oc * 5 + ic * 3) % 7 - 3);
        }
    }

    std::unique_ptr<MNN::NetT> net(new MNN::NetT);
    net->tensorName = {"input", "middle", "output"};
    net->tensorNumber = 3;
    net->outputName = {"output"};
    net->usage = MNN::Usage_INFERENCE;
    net->sourceType = MNN::NetSource_TENSORFLOW;
    net->extraTensorDescribe.emplace_back(makeQuantDescribe(0, kInputScale));
    net->extraTensorDescribe.emplace_back(makeQuantDescribe(1, kMiddleScale));
    net->extraTensorDescribe.emplace_back(makeQuantDescribe(2, kOutputScale));

    std::unique_ptr<MNN::OpT> input(new MNN::OpT);
    input->name = "input";
    input->type = MNN::OpType_Input;
    input->main.type = MNN::OpParameter_Input;
    input->main.value = new MNN::InputT;
    input->main.AsInput()->dims = {1, kInputChannels, 1, kArea};
    input->main.AsInput()->dtype = MNN::DataType_DT_FLOAT;
    input->main.AsInput()->dformat = MNN::MNN_DATA_FORMAT_NC4HW4;
    input->outputIndexes = {0};
    net->oplists.emplace_back(std::move(input));
    net->oplists.emplace_back(makeConv(
        "w8a8_conv1", 0, 1, kInputChannels, kMiddleChannels,
        weight1, bias1, kWeightScale1));
    net->oplists.emplace_back(makeConv(
        "w8a8_conv2", 1, 2, kMiddleChannels, kOutputChannels,
        weight2, bias2, kWeightScale2));
    return net;
}

bool testMultiBatchNc64Layout() {
    constexpr int batch = 2;
    constexpr int inputChannels = 65;
    constexpr int outputChannels = 67;
    constexpr int area = 3;
    constexpr float scale = 1.0f;
    std::vector<int8_t> weight(inputChannels * outputChannels);
    for (int oc = 0; oc < outputChannels; ++oc) {
        for (int ic = 0; ic < inputChannels; ++ic) {
            weight[static_cast<size_t>(oc) * inputChannels + ic] =
                static_cast<int8_t>((oc * 7 + ic * 3) % 5 - 2);
        }
    }
    std::vector<float> bias(outputChannels, 0.0f);

    std::unique_ptr<MNN::NetT> net(new MNN::NetT);
    net->tensorName = {"input", "output"};
    net->tensorNumber = 2;
    net->outputName = {"output"};
    net->usage = MNN::Usage_INFERENCE;
    net->sourceType = MNN::NetSource_TENSORFLOW;
    net->extraTensorDescribe.emplace_back(makeQuantDescribe(0, scale));
    net->extraTensorDescribe.emplace_back(makeQuantDescribe(1, scale));
    std::unique_ptr<MNN::OpT> inputOp(new MNN::OpT);
    inputOp->name = "input";
    inputOp->type = MNN::OpType_Input;
    inputOp->main.type = MNN::OpParameter_Input;
    inputOp->main.value = new MNN::InputT;
    inputOp->main.AsInput()->dims =
        {batch, inputChannels, 1, area};
    inputOp->main.AsInput()->dtype = MNN::DataType_DT_FLOAT;
    inputOp->main.AsInput()->dformat = MNN::MNN_DATA_FORMAT_NC4HW4;
    inputOp->outputIndexes = {0};
    net->oplists.emplace_back(std::move(inputOp));
    net->oplists.emplace_back(makeConv(
        "multi_batch_conv", 0, 1, inputChannels, outputChannels,
        weight, bias, scale));

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(MNN::Net::Pack(builder, net.get()));
    std::shared_ptr<MNN::Interpreter> interpreter(
        MNN::Interpreter::createFromBuffer(builder.GetBufferPointer(),
                                           builder.GetSize()),
        MNN::Interpreter::destroy);
    CHECK(interpreter != nullptr);
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_HEXAGON;
    config.backupType = MNN_FORWARD_HEXAGON;
    auto session = interpreter->createSession(config);
    CHECK(session != nullptr);

    auto input = interpreter->getSessionInput(session, "input");
    CHECK(input != nullptr);
    MNN::Tensor hostInput(input, MNN::Tensor::CAFFE);
    MNN::TensorUtils::getDescribe(&hostInput)->dimensionFormat =
        MNN::MNN_DATA_FORMAT_NCHW;
    std::vector<int8_t> inputQ(batch * inputChannels * area);
    for (int n = 0; n < batch; ++n) {
        for (int ic = 0; ic < inputChannels; ++ic) {
            for (int a = 0; a < area; ++a) {
                const int index = (n * inputChannels + ic) * area + a;
                inputQ[index] = static_cast<int8_t>(
                    (n * 11 + ic * 5 + a * 3) % 9 - 4);
                hostInput.host<float>()[index] = inputQ[index];
            }
        }
    }
    CHECK(input->copyFromHostTensor(&hostInput));

    int convCallbacks = 0;
    CHECK(interpreter->runSessionWithCallBackInfo(
              session,
              [](const std::vector<MNN::Tensor*>&,
                 const MNN::OperatorInfo*) { return true; },
              [&](const std::vector<MNN::Tensor*>& outputs,
                  const MNN::OperatorInfo* info) {
                  if (info->name() == "multi_batch_conv") {
                      CHECK(outputs.size() == 1);
                      CHECK(interpreter->getBackend(session, outputs[0])->type() ==
                            MNN_FORWARD_HEXAGON);
                      ++convCallbacks;
                  }
                  return true;
              }) == MNN::NO_ERROR);
    CHECK(convCallbacks == 1);

    auto output = interpreter->getSessionOutput(session, "output");
    CHECK(output != nullptr);
    MNN::Tensor hostOutput(output, MNN::Tensor::CAFFE);
    MNN::TensorUtils::getDescribe(&hostOutput)->dimensionFormat =
        MNN::MNN_DATA_FORMAT_NCHW;
    CHECK(output->copyToHostTensor(&hostOutput));
    auto outputBackend = interpreter->getBackend(session, output);
    CHECK(outputBackend != nullptr);
    auto outputRuntime = static_cast<const MNN::HexagonRuntime*>(
        const_cast<MNN::Backend*>(outputBackend)->getRuntime());
    const int copyStatus = outputRuntime->consumeCommandStatus();
    if (copyStatus != 0) {
        std::printf("output copy status=%d\n", copyStatus);
        return false;
    }
    for (int n = 0; n < batch; ++n) {
        for (int oc = 0; oc < outputChannels; ++oc) {
            for (int a = 0; a < area; ++a) {
                int32_t accumulator = 0;
                for (int ic = 0; ic < inputChannels; ++ic) {
                    accumulator +=
                        inputQ[(n * inputChannels + ic) * area + a] *
                        weight[static_cast<size_t>(oc) * inputChannels + ic];
                }
                const float expected = static_cast<float>(
                    std::max(-127, std::min(127, accumulator)));
                const float actual = hostOutput.host<float>()[
                    (n * outputChannels + oc) * area + a];
                if (actual != expected) {
                    std::printf(
                        "multi-batch mismatch n=%d oc=%d area=%d "
                        "actual=%f expected=%f\n",
                        n, oc, a, actual, expected);
                    return false;
                }
            }
        }
    }
    return true;
}

bool testQuantizedConvDoesNotFallBackToFp16() {
    std::vector<int8_t> weight(kMiddleChannels * kInputChannels, 1);
    std::vector<float> bias(kMiddleChannels, 0.0f);
    auto op = makeConv("no_fp16_fallback", 0, 1, kInputChannels,
                       kMiddleChannels, weight, bias, kWeightScale1);
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(MNN::Op::Pack(builder, op.get()));
    const auto packedOp =
        flatbuffers::GetRoot<MNN::Op>(builder.GetBufferPointer());

    std::unique_ptr<MNN::Tensor> input(
        MNN::Tensor::createDevice<float>({1, kInputChannels, 1, kArea}));
    std::unique_ptr<MNN::Tensor> output(
        MNN::Tensor::createDevice<float>({1, kMiddleChannels, 1, kArea}));
    auto setQuant = [](MNN::Tensor* tensor, float scale) {
        auto describe = MNN::TensorUtils::getDescribe(tensor);
        describe->dimensionFormat = MNN::MNN_DATA_FORMAT_NC4HW4;
        describe->quantAttr.reset(new MNN::QuantAttr);
        describe->quantAttr->scale = scale;
        describe->quantAttr->zero = 0.0f;
        describe->quantAttr->min = -127.0f;
        describe->quantAttr->max = 127.0f;
        describe->quantAttr->type = MNN::DataType_DT_INT8;
        describe->applyQuant = true;
    };
    setQuant(input.get(), kInputScale);
    setQuant(output.get(), kMiddleScale);

    MNN::Backend::Info info;
    info.type = MNN_FORWARD_HEXAGON;
    MNN::HexagonRuntime runtime(info);
    CHECK(runtime.isAvailable());
    std::unique_ptr<MNN::Backend> backend(runtime.onCreate());
    CHECK(backend != nullptr);
    auto& runtimeInfo =
        const_cast<MNN::HexagonInfo&>(runtime.info());
    const uint32_t capabilities = runtimeInfo.capabilities;
    runtimeInfo.capabilities &= ~DSP_CAP_CONV1X1_W8A8;
    std::unique_ptr<MNN::Execution> execution(
        MNN::HexagonExecutionFactory::create(
            packedOp, {input.get()}, {output.get()}, backend.get()));
    runtimeInfo.capabilities = capabilities;
    CHECK(execution == nullptr);
    return true;
}

bool testInvalidHostQuantizationFailsNextRun() {
    MNN::Backend::Info info;
    info.type = MNN_FORWARD_HEXAGON;
    MNN::HexagonRuntime runtime(info);
    CHECK(runtime.isAvailable());
    std::unique_ptr<MNN::Backend> backend(runtime.onCreate());
    CHECK(backend != nullptr);
    std::unique_ptr<MNN::Tensor> device(
        MNN::Tensor::createDevice<float>({1, 4, 1, 1}));
    auto describe = MNN::TensorUtils::getDescribe(device.get());
    describe->dimensionFormat = MNN::MNN_DATA_FORMAT_NC4HW4;
    describe->quantAttr.reset(new MNN::QuantAttr);
    describe->quantAttr->scale = 0.25f;
    describe->quantAttr->zero = 0.0f;
    describe->quantAttr->min = -127.0f;
    describe->quantAttr->max = 127.0f;
    describe->quantAttr->type = MNN::DataType_DT_INT8;
    describe->applyQuant = true;
    CHECK(backend->onAcquireBuffer(device.get(), MNN::Backend::STATIC));

    MNN::Tensor hostInput(device.get(), MNN::Tensor::CAFFE);
    MNN::TensorUtils::getDescribe(&hostInput)->dimensionFormat =
        MNN::MNN_DATA_FORMAT_NCHW;
    std::fill(hostInput.host<float>(),
              hostInput.host<float>() + hostInput.elementSize(), 0.0f);
    hostInput.host<float>()[0] =
        std::numeric_limits<float>::quiet_NaN();
    backend->onCopyBuffer(&hostInput, device.get());
    backend->onExecuteBegin();
    CHECK(runtime.pCurrentStatus == MNN::INVALID_VALUE);
    CHECK(backend->onReleaseBuffer(device.get(), MNN::Backend::STATIC));
    return true;
}

bool testDspRejectsUndersizedConvBuffer() {
    MNN::Backend::Info info;
    info.type = MNN_FORWARD_HEXAGON;
    MNN::HexagonRuntime runtime(info);
    CHECK(runtime.isAvailable());
    std::unique_ptr<MNN::Backend> backend(runtime.onCreate());
    CHECK(backend != nullptr);
    auto hexagonBackend = static_cast<MNN::HexagonBackend*>(backend.get());
    auto allocator = hexagonBackend->getAllocator(2);
    CHECK(allocator != nullptr);

    constexpr int kInputBytes = 64;
    constexpr int kWeightBytes = 1024;
    constexpr int kBiasBytes = 32 * sizeof(int32_t);
    constexpr int kRequantBytes = 32 * sizeof(DSPRequantParam);
    constexpr int kOutputBytes = 64;
    const auto input = allocator->alloc(kInputBytes);
    const auto weight = allocator->alloc(kWeightBytes);
    const auto bias = allocator->alloc(kBiasBytes);
    const auto requant = allocator->alloc(kRequantBytes);
    const auto output = allocator->alloc(kOutputBytes);
    CHECK(input.first != nullptr && weight.first != nullptr &&
          bias.first != nullptr && requant.first != nullptr &&
          output.first != nullptr);

    DSPConv2DInt8Param params = {};
    params.batch = 1;
    params.inputHeight = 1;
    params.inputWidth = 1;
    params.inputChannels = 1;
    params.outputHeight = 1;
    params.outputWidth = 1;
    params.outputChannels = 1;
    params.kernelY = 1;
    params.kernelX = 1;
    params.strideY = 1;
    params.strideX = 1;
    params.dilateY = 1;
    params.dilateX = 1;
    params.groups = 1;
    params.channelPack = 64;

    MNN::HexagonCommand command;
    command.build(
        hexagonBackend, DSP_OP_CONV2D_W8A8, &params, sizeof(params),
        {MNN::HexagonBackend::getDevicePtr(input),
         MNN::HexagonBackend::getDevicePtr(weight),
         MNN::HexagonBackend::getDevicePtr(bias),
         MNN::HexagonBackend::getDevicePtr(requant)},
        {MNN::HexagonBackend::getDevicePtr(output)}, {}, {},
        {kInputBytes, kWeightBytes - 1, kBiasBytes, kRequantBytes},
        {kOutputBytes});
    CHECK(command.execute() == 0);
    const int flushStatus = hexagonBackend->flushCommand();
    CHECK(flushStatus != 0);
    CHECK(runtime.consumeCommandStatus() == flushStatus);

    allocator->free(input);
    allocator->free(weight);
    allocator->free(bias);
    allocator->free(requant);
    allocator->free(output);
    return true;
}

bool testChainedConv() {
    std::vector<int8_t> weight1;
    std::vector<float> bias1;
    std::vector<int8_t> weight2;
    std::vector<float> bias2;
    auto net = makeChainedModel(weight1, bias1, weight2, bias2);
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(MNN::Net::Pack(builder, net.get()));

    std::shared_ptr<MNN::Interpreter> interpreter(
        MNN::Interpreter::createFromBuffer(builder.GetBufferPointer(),
                                           builder.GetSize()),
        MNN::Interpreter::destroy);
    CHECK(interpreter != nullptr);
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_HEXAGON;
    config.backupType = MNN_FORWARD_HEXAGON;
    auto session = interpreter->createSession(config);
    CHECK(session != nullptr);

    auto input = interpreter->getSessionInput(session, "input");
    CHECK(input != nullptr);
    MNN::Tensor hostInput(input, MNN::Tensor::CAFFE);
    MNN::TensorUtils::getDescribe(&hostInput)->dimensionFormat =
        MNN::MNN_DATA_FORMAT_NCHW;
    std::vector<int8_t> inputQ(kInputChannels * kArea);
    for (int ic = 0; ic < kInputChannels; ++ic) {
        for (int area = 0; area < kArea; ++area) {
            const int index = ic * kArea + area;
            inputQ[index] = static_cast<int8_t>(
                (ic * 3 + area * 2) % 7 - 3);
            hostInput.host<float>()[index] =
                inputQ[index] * kInputScale;
        }
    }
    CHECK(input->copyFromHostTensor(&hostInput));

    int convCallbacks = 0;
    CHECK(interpreter->runSessionWithCallBackInfo(
              session,
              [](const std::vector<MNN::Tensor*>&,
                 const MNN::OperatorInfo*) {
                  return true;
              },
              [&](const std::vector<MNN::Tensor*>& outputs,
                  const MNN::OperatorInfo* info) {
                  if (info->name() == "w8a8_conv1" ||
                      info->name() == "w8a8_conv2") {
                      CHECK(outputs.size() == 1);
                      CHECK(interpreter->getBackend(session, outputs[0])->type() ==
                            MNN_FORWARD_HEXAGON);
                      ++convCallbacks;
                  }
                  return true;
              }) == MNN::NO_ERROR);
    CHECK(convCallbacks == 2);

    std::vector<int32_t> biasInt1;
    std::vector<int32_t> biasInt2;
    std::vector<DSPRequantParam> requant1;
    std::vector<DSPRequantParam> requant2;
    std::vector<float> weightScale1(kMiddleChannels, kWeightScale1);
    std::vector<float> weightScale2(kOutputChannels, kWeightScale2);
    CHECK(MNN::buildW8A8Conv1x1PostParams(
        biasInt1, requant1, bias1.data(), bias1.size(),
        weightScale1.data(), weightScale1.size(), kMiddleChannels,
        kInputScale, kMiddleScale, false, false));
    CHECK(MNN::buildW8A8Conv1x1PostParams(
        biasInt2, requant2, bias2.data(), bias2.size(),
        weightScale2.data(), weightScale2.size(), kOutputChannels,
        kMiddleScale, kOutputScale, false, false));

    std::vector<int8_t> middleQ(kMiddleChannels * kArea);
    for (int oc = 0; oc < kMiddleChannels; ++oc) {
        for (int area = 0; area < kArea; ++area) {
            int32_t accumulator = biasInt1[oc];
            for (int ic = 0; ic < kInputChannels; ++ic) {
                accumulator += inputQ[ic * kArea + area] *
                               weight1[static_cast<size_t>(oc) *
                                       kInputChannels + ic];
            }
            middleQ[oc * kArea + area] =
                MNN::requantizeReference(accumulator, requant1[oc]);
        }
    }

    auto output = interpreter->getSessionOutput(session, "output");
    CHECK(output != nullptr);
    MNN::Tensor hostOutput(output, MNN::Tensor::CAFFE);
    MNN::TensorUtils::getDescribe(&hostOutput)->dimensionFormat =
        MNN::MNN_DATA_FORMAT_NCHW;
    CHECK(output->copyToHostTensor(&hostOutput));
    auto chainedBackend = interpreter->getBackend(session, output);
    CHECK(chainedBackend != nullptr);
    auto chainedRuntime = static_cast<const MNN::HexagonRuntime*>(
        const_cast<MNN::Backend*>(chainedBackend)->getRuntime());
    const int chainedStatus = chainedRuntime->consumeCommandStatus();
    if (chainedStatus != 0) {
        std::printf("chained output copy status=%d\n", chainedStatus);
        return false;
    }
    for (int oc = 0; oc < kOutputChannels; ++oc) {
        for (int area = 0; area < kArea; ++area) {
            int32_t accumulator = biasInt2[oc];
            for (int ic = 0; ic < kMiddleChannels; ++ic) {
                accumulator += middleQ[ic * kArea + area] *
                               weight2[static_cast<size_t>(oc) *
                                       kMiddleChannels + ic];
            }
            const float expected =
                MNN::requantizeReference(accumulator, requant2[oc]) *
                kOutputScale;
            const float actual =
                hostOutput.host<float>()[oc * kArea + area];
            if (std::fabs(actual - expected) > kOutputScale) {
                std::printf(
                    "chained mismatch oc=%d area=%d actual=%f expected=%f\n",
                    oc, area, actual, expected);
                return false;
            }
        }
    }
    return true;
}

bool testInjectedDspFailure() {
    MNN::Backend::Info info;
    info.type = MNN_FORWARD_HEXAGON;
    MNN::HexagonRuntime runtime(info);
    CHECK(runtime.isAvailable());
    std::unique_ptr<MNN::Backend> backend(runtime.onCreate());
    CHECK(backend != nullptr);
    auto hexagonBackend = static_cast<MNN::HexagonBackend*>(backend.get());

    MNN::HexagonCommand command;
    command.build(hexagonBackend, 98, nullptr, 0, {}, {});
    CHECK(command.execute() == 0);
    CHECK(hexagonBackend->flushCommand() == 98);
    CHECK(runtime.consumeCommandStatus() == 98);
    CHECK(runtime.consumeCommandStatus() == 0);
    return true;
}

bool testHostNhwcQuantizedCopy() {
    MNN::Backend::Info info;
    info.type = MNN_FORWARD_HEXAGON;
    MNN::HexagonRuntime runtime(info);
    CHECK(runtime.isAvailable());
    std::unique_ptr<MNN::Backend> backend(runtime.onCreate());
    std::unique_ptr<MNN::Tensor> device(MNN::Tensor::createDevice<float>({1, 3, 1, 2}));
    auto describe = MNN::TensorUtils::getDescribe(device.get());
    describe->dimensionFormat = MNN::MNN_DATA_FORMAT_NC4HW4;
    describe->quantAttr.reset(new MNN::QuantAttr);
    describe->quantAttr->scale = 1.0f;
    describe->quantAttr->zero = 0.0f;
    describe->quantAttr->min = -127.0f;
    describe->quantAttr->max = 127.0f;
    describe->quantAttr->type = MNN::DataType_DT_INT8;
    describe->applyQuant = true;
    CHECK(backend->onAcquireBuffer(device.get(), MNN::Backend::STATIC));
    MNN::Tensor hostInput(device.get(), MNN::Tensor::TENSORFLOW);
    const float values[] = {1, 2, 3, 4, 5, 6};
    std::memcpy(hostInput.host<float>(), values, sizeof(values));
    backend->onCopyBuffer(&hostInput, device.get());
    const auto packed = reinterpret_cast<const int8_t*>(MNN::HexagonBackend::getPtr(device.get()));
    const int pack = runtime.info().vectorSize;
    CHECK(packed[0] == 1 && packed[1] == 2 && packed[2] == 3);
    CHECK(packed[pack] == 4 && packed[pack + 1] == 5 && packed[pack + 2] == 6);
    MNN::Tensor hostOutput(device.get(), MNN::Tensor::TENSORFLOW);
    backend->onCopyBuffer(device.get(), &hostOutput);
    CHECK(runtime.consumeCommandStatus() == 0);
    CHECK(std::memcmp(hostOutput.host<float>(), values, sizeof(values)) == 0);
    CHECK(backend->onReleaseBuffer(device.get(), MNN::Backend::STATIC));
    return true;
}

bool testDspRequantTiesAndAccumulatorOverflow() {
    MNN::Backend::Info info;
    info.type = MNN_FORWARD_HEXAGON;
    MNN::HexagonRuntime runtime(info);
    CHECK(runtime.isAvailable());
    std::unique_ptr<MNN::Backend> backend(runtime.onCreate());
    auto hexBackend = static_cast<MNN::HexagonBackend*>(backend.get());
    CHECK(hexBackend != nullptr);
    auto allocator = hexBackend->getAllocator(2);
    const int sizes[] = {64, 1024, 128, 32 * static_cast<int>(sizeof(DSPRequantParam)), 64};
    MNN::MemChunk buffers[5];
    for (int i = 0; i < 5; ++i) {
        buffers[i] = allocator->alloc(sizes[i]);
        CHECK(buffers[i].first != nullptr);
        std::memset(MNN::HexagonBackend::getPtr(buffers[i]), 0, sizes[i]);
    }
    DSPConv2DInt8Param p = {};
    p.batch = p.inputHeight = p.inputWidth = p.inputChannels = 1;
    p.outputHeight = p.outputWidth = 1;
    p.outputChannels = 32;
    p.kernelY = p.kernelX = p.strideY = p.strideX = 1;
    p.dilateY = p.dilateX = p.groups = 1;
    p.channelPack = 64;
    auto biases = reinterpret_cast<int32_t*>(MNN::HexagonBackend::getPtr(buffers[2]));
    auto post = reinterpret_cast<DSPRequantParam*>(MNN::HexagonBackend::getPtr(buffers[3]));
    const double scales[] = {0.5, 0.75, 0.25, 0.375};
    bool passed = true;
    for (int mode = 0; mode < 2; ++mode) {
        std::vector<int8_t> rawWeight(32, mode == 0 ? 0 : 127);
        std::vector<int8_t> packedWeight;
        CHECK(MNN::packW8A8Conv1x1Weight(packedWeight, rawWeight.data(), 32, 1, 32));
        std::memcpy(MNN::HexagonBackend::getPtr(buffers[1]), packedWeight.data(), packedWeight.size());
        MNN::HexagonBackend::getPtr(buffers[0])[0] = mode == 0 ? 0 : 127;
        for (int lane = 0; lane < 32; ++lane) {
            biases[lane] = mode == 0 ? lane - 16 : std::numeric_limits<int32_t>::max() - 1;
            post[lane] = MNN::makeRequantMultiplier(mode == 0 ? scales[lane % 4] : 0.5);
        }
        for (int i = 0; i < 4; ++i) {
            hexBackend->markHostInput(buffers[i], sizes[i]);
        }
        hexBackend->markHostOutput(buffers[4], sizes[4]);
        MNN::HexagonCommand command;
        command.build(hexBackend, DSP_OP_CONV2D_W8A8, &p, sizeof(p),
                      {MNN::HexagonBackend::getDevicePtr(buffers[0]),
                       MNN::HexagonBackend::getDevicePtr(buffers[1]),
                       MNN::HexagonBackend::getDevicePtr(buffers[2]),
                       MNN::HexagonBackend::getDevicePtr(buffers[3])},
                      {MNN::HexagonBackend::getDevicePtr(buffers[4])}, {}, {},
                      {sizes[0], sizes[1], sizes[2], sizes[3]}, {sizes[4]});
        CHECK(command.execute() == 0);
        CHECK(hexBackend->flushCommand() == 0);
        const auto output = reinterpret_cast<const int8_t*>(MNN::HexagonBackend::getPtr(buffers[4]));
        for (int lane = 0; lane < 32; ++lane) {
            const int expected = mode == 0 ? MNN::requantizeReference(biases[lane], post[lane]) : 127;
            if (output[lane] != expected) {
                std::printf("DSP numeric mismatch mode=%d lane=%d actual=%d expected=%d\n",
                            mode, lane, output[lane], expected);
                passed = false;
            }
        }
    }
    for (auto& buffer : buffers) {
        allocator->free(buffer);
    }
    CHECK(passed);
    return true;
}

} // namespace

int main() {
    if (!testHostNhwcQuantizedCopy() ||
        !testDspRequantTiesAndAccumulatorOverflow() ||
        !testChainedConv() || !testMultiBatchNc64Layout() ||
        !testQuantizedConvDoesNotFallBackToFp16() ||
        !testInvalidHostQuantizationFailsNextRun() ||
        !testDspRejectsUndersizedConvBuffer() ||
        !testInjectedDspFailure()) {
        return 1;
    }
    std::printf("PASS: Hexagon W8A8 device contract\n");
    return 0;
}
