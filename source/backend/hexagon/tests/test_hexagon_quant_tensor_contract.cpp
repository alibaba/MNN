#include "HexagonBackend.hpp"
#include "HexagonConvolutionInt8.hpp"
#include "HexagonQuantization.hpp"
#include "HexagonQuantize.hpp"
#include "HexagonRuntime.hpp"
#include "core/TensorUtils.hpp"

#include <cstdio>
#include <flatbuffers/flatbuffers.h>
#include <memory>
#include <vector>

namespace {

#define CHECK(condition)                                                                            \
    do {                                                                                            \
        if (!(condition)) {                                                                         \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);                        \
            return false;                                                                           \
        }                                                                                           \
    } while (0)

std::unique_ptr<MNN::Tensor> makeQuantTensor(const std::vector<int>& shape,
                                             float scale, float zero) {
    std::unique_ptr<MNN::Tensor> tensor(
        MNN::Tensor::createDevice<float>(shape));
    auto describe = MNN::TensorUtils::getDescribe(tensor.get());
    describe->dimensionFormat = MNN::MNN_DATA_FORMAT_NC4HW4;
    describe->quantAttr.reset(new MNN::QuantAttr);
    describe->quantAttr->scale = scale;
    describe->quantAttr->zero = zero;
    describe->quantAttr->min = -127.0f;
    describe->quantAttr->max = 127.0f;
    describe->quantAttr->type = MNN::DataType_DT_INT8;
    describe->applyQuant = true;
    return tensor;
}

const MNN::Op* makeConvOp(flatbuffers::FlatBufferBuilder& builder, int kernel,
                          int stride, int readType) {
    MNN::OpT op;
    op.type = MNN::OpType_Convolution;
    op.main.type = MNN::OpParameter_Convolution2D;
    op.main.value = new MNN::Convolution2DT;
    auto conv = op.main.AsConvolution2D();
    conv->common.reset(new MNN::Convolution2DCommonT);
    conv->common->inputCount = 35;
    conv->common->outputCount = 37;
    conv->common->kernelX = kernel;
    conv->common->kernelY = kernel;
    conv->common->strideX = stride;
    conv->common->strideY = stride;
    conv->common->dilateX = 1;
    conv->common->dilateY = 1;
    conv->common->group = 1;
    conv->common->padMode = MNN::PadMode_CAFFE;
    conv->quanParameter.reset(new MNN::IDSTQuanT);
    conv->quanParameter->type = 1;
    conv->quanParameter->aMaxOrBits = 8;
    conv->quanParameter->readType = readType;
    const auto offset = MNN::Op::Pack(builder, &op);
    builder.Finish(offset);
    return flatbuffers::GetRoot<MNN::Op>(builder.GetBufferPointer());
}

bool testPhysicalStorageBytes() {
    std::unique_ptr<MNN::Tensor> tensor(
        MNN::Tensor::createDevice<float>({1, 3, 2, 2}));
    CHECK(MNN::HexagonBackend::getBytes(tensor.get()) == 2);

    auto describe = MNN::TensorUtils::getDescribe(tensor.get());
    describe->quantAttr.reset(new MNN::QuantAttr);
    describe->quantAttr->scale = 0.125f;
    describe->applyQuant = true;
    CHECK(MNN::HexagonBackend::getBytes(tensor.get()) == 1);

    describe->applyQuant = false;
    CHECK(MNN::HexagonBackend::getBytes(tensor.get()) == 2);
    return true;
}

bool testSupportedConvContract() {
    auto input = makeQuantTensor({1, 35, 2, 3}, 0.25f, 0.0f);
    auto output = makeQuantTensor({1, 37, 2, 3}, 0.125f, 0.0f);

    flatbuffers::FlatBufferBuilder supportedBuilder;
    const auto supported = makeConvOp(supportedBuilder, 1, 1, 0);
    CHECK(MNN::supportsW8A8Conv1x1(
        supported, {input.get()}, {output.get()}));
    CHECK(MNN::HexagonConvolutionInt8::validTensorContract(
        supported, {input.get()}, {output.get()}, 64));
    CHECK(!MNN::HexagonConvolutionInt8::validTensorContract(
        supported, {input.get()}, {output.get()}, 0));

    flatbuffers::FlatBufferBuilder kernelBuilder;
    CHECK(!MNN::supportsW8A8Conv1x1(
        makeConvOp(kernelBuilder, 3, 1, 0),
        {input.get()}, {output.get()}));

    flatbuffers::FlatBufferBuilder strideBuilder;
    CHECK(!MNN::supportsW8A8Conv1x1(
        makeConvOp(strideBuilder, 1, 2, 0),
        {input.get()}, {output.get()}));

    MNN::TensorUtils::getDescribe(input.get())->quantAttr->zero = 1.0f;
    CHECK(!MNN::supportsW8A8Conv1x1(
        supported, {input.get()}, {output.get()}));
    CHECK(!MNN::HexagonConvolutionInt8::validTensorContract(
        supported, {input.get()}, {output.get()}, 64));
    return true;
}

bool testQuantPropagationAndBoundaryContract() {
    auto input = makeQuantTensor({1, 35, 2, 3}, 0.25f, 0.0f);
    auto output = makeQuantTensor({1, 37, 2, 3}, 0.125f, 0.0f);
    flatbuffers::FlatBufferBuilder builder;
    const auto op = makeConvOp(builder, 1, 1, 0);

    MNN::HexagonRuntimeCreator creator;
    CHECK(creator.onSetQuantInfo(op, {input.get()}, {output.get()}));
    CHECK(MNN::TensorUtils::getDescribe(output.get())->applyQuant);

    std::unique_ptr<MNN::Tensor> fp16(
        MNN::Tensor::createDevice<float>({1, 3, 2, 2}));
    auto fp16Describe = MNN::TensorUtils::getDescribe(fp16.get());
    fp16Describe->dimensionFormat = MNN::MNN_DATA_FORMAT_NC4HW4;
    CHECK(MNN::HexagonQuantize::validTensorContract(
        fp16.get(), output.get(), true));
    CHECK(MNN::HexagonQuantize::validTensorContract(
        output.get(), fp16.get(), false));

    MNN::TensorUtils::getDescribe(output.get())->applyQuant = false;
    CHECK(!MNN::HexagonQuantize::validTensorContract(
        fp16.get(), output.get(), true));
    return true;
}

} // namespace

int main() {
    if (!testPhysicalStorageBytes() || !testSupportedConvContract() ||
        !testQuantPropagationAndBoundaryContract()) {
        return 1;
    }
    std::printf("PASS: Hexagon quantized tensor contract\n");
    return 0;
}
