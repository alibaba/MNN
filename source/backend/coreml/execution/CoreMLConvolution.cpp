//
//  CoreMLConvolution.cpp
//  MNN
//
//  Created by MNN on 2021/03/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <float.h>
#include "core/ConvolutionCommon.hpp"
#include "CoreMLConvolution.hpp"
#include "backend/arm82/Arm82OptFunc.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <sys/stat.h>

namespace MNN {

static int reductionBlockSize(const Convolution2DCommon* common) {
    const int area = common->kernelX() * common->kernelY();
    // Larger blocks can make Core ML move SAME/VALID graphs to CPU.
    const bool explicit3x3 = common->kernelX() == 3 && common->kernelY() == 3 && common->padMode() == PadMode_CAFFE;
    const int limit = explicit3x3 ? 512 : 256;
    return std::max(8, (limit / area / 8) * 8);
}

static std::string weightKey(const Op* op) {
    // Geometry-generated children have different Op addresses across modules.
    // Match their immutable contents and external-file revision, not layer names.
    CC_SHA256_CTX hash;
    CC_SHA256_Init(&hash);
    if (op->externalPath()) {
        struct stat file;
        if (::stat(op->externalPath()->c_str(), &file) != 0) {
            return "";
        }
        const uint64_t revision[] = {static_cast<uint64_t>(file.st_dev),
                                     static_cast<uint64_t>(file.st_ino),
                                     static_cast<uint64_t>(file.st_size),
                                     static_cast<uint64_t>(file.st_mtimespec.tv_sec),
                                     static_cast<uint64_t>(file.st_mtimespec.tv_nsec),
                                     static_cast<uint64_t>(file.st_ctimespec.tv_sec),
                                     static_cast<uint64_t>(file.st_ctimespec.tv_nsec)};
        CC_SHA256_Update(&hash, revision, sizeof(revision));
    }
    std::unique_ptr<OpT> canonical(op->UnPack());
    canonical->name.clear();
    canonical->inputIndexes.clear();
    canonical->outputIndexes.clear();
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(Op::Pack(builder, canonical.get()));
    CC_SHA256_Update(&hash, builder.GetBufferPointer(), builder.GetSize());
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, &hash);
    return std::string(reinterpret_cast<const char*>(digest), sizeof(digest));
}

static void dequantizeBlock(const int8_t* src, float* dst, size_t start, size_t size, bool int4, float scale,
                            float offset) {
    if (!int4) {
        for (size_t i = 0; i < size; ++i) {
            dst[i] = src[start + i] * scale + offset;
        }
        return;
    }
    size_t i = 0;
    if ((start & 1) && size) {
        dst[i++] = ((static_cast<uint8_t>(src[start / 2]) & 15) - 8) * scale + offset;
        ++start;
    }
    auto packed = reinterpret_cast<const uint8_t*>(src) + start / 2;
    auto out = dst + i;
    size_t pairs = (size - i) / 2;
    // Decode adjacent nibbles together so the compiler can vectorize the loop.
    for (size_t j = 0; j < pairs; ++j) {
        auto value = packed[j];
        out[2 * j] = ((value >> 4) - 8) * scale + offset;
        out[2 * j + 1] = ((value & 15) - 8) * scale + offset;
    }
    if (i + 2 * pairs < size) {
        out[2 * pairs] = ((packed[pairs] >> 4) - 8) * scale + offset;
    }
}

static Tensor* loadQuantizedWeight(CoreMLBackend* backend, const Op* op, const std::vector<int>& shape,
                                   const std::string& name) {
    auto param = op->main_as_Convolution2D()->quanParameter();
    if (!param || param->type() != 1 || param->index() || param->has_scaleInt()) {
        return nullptr;
    }
    auto quant = ConvolutionCommon::load(op, backend, false, true, nullptr, true);
    if (!quant || !quant->weight.get()) {
        return nullptr;
    }
    size_t count = 1;
    for (int dim : shape) {
        count *= dim;
    }
    if (quant->canUseInt4 ? quant->weight.size() != UP_DIV(count, 2)
                          : (quant->originBits != 8 || quant->weight.size() != count)) {
        return nullptr;
    }
    const int groups = quant->alphaSize / (quant->asymmetric ? 2 : 1);
    if (groups <= 0 || count % groups) {
        return nullptr;
    }
    auto alpha = quant->getAlphaFloat();
    if (!alpha) {
        return nullptr;
    }
    quant->alphaHalf.release();
    const size_t blockSize = count / groups;
    std::shared_ptr<CoreMLWeightSource> source(new CoreMLWeightSource);
    source->elements = count;
    source->storageBytes =
        quant->weight.size() + quant->alpha.size() * sizeof(float) + quant->alphaHalf.size() * sizeof(int16_t);
    // Decode a bounded block directly into its final FP16 storage. A full FP32
    // projection (especially the vocabulary head) need not exist even briefly.
    source->decode = [quant, blockSize, groups](int16_t* dst) {
        auto alpha = quant->getAlphaFloat();
        float values[1024];
        for (int group = 0; group < groups; ++group) {
            float scale = alpha[quant->asymmetric ? 2 * group + 1 : group];
            float offset = quant->asymmetric ? alpha[2 * group] : 0.0f;
            for (size_t j = 0; j < blockSize; j += 1024) {
                size_t size = std::min<size_t>(1024, blockSize - j);
                size_t start = group * blockSize + j;
                dequantizeBlock(quant->weight.get(), values, start, size, quant->canUseInt4, scale, offset);
                MNNQuantizeFP16(values, dst + start, size);
            }
        }
    };
    auto tensor = backend->allocateWeight(name, shape, backend->lowMemory() ? source : nullptr);
    if (!tensor) {
        return nullptr;
    }
    if (!backend->lowMemory()) {
        source->decode(tensor->host<int16_t>());
    }
    return tensor;
}

static inline std::string padTypeConvert(PadMode padMod) {
    std::string padType = "valid";
    switch (padMod) {
        case PadMode_SAME:
            padType = "same";
            break;
        case PadMode_VALID:
            padType = "valid";
            break;
        case PadMode_CAFFE:
            padType = "custom";
            break;
        default:
            padType = "custom";
            break;
    }
    return padType;
}

CoreMLConvolution::CoreMLConvolution(MNN::Backend* b, const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                     const std::vector<MNN::Tensor*>& outputs)
    : CoreMLCommonExecution(b, op) {
    mIsDeconv = op->type() == OpType_Deconvolution || op->type() == OpType_DeconvolutionDepthwise;
    initConv2d(inputs, outputs);
}

void CoreMLConvolution::initConv2d(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto conv2D = mOp->main_as_Convolution2D();
    auto common = conv2D->common();
    auto kernelX = common->kernelX();
    auto kernelY = common->kernelY();
    auto inputCount = inputs[0]->channel();
    auto outputCount = common->outputCount();
    auto group = common->group();
    if (group <= 0 || inputCount % group || outputCount % group) {
        mValid = false;
        return;
    }
    mIsLinear = mCoreMLBackend->lowMemory() && !mIsDeconv && kernelY == 1 && kernelX == 1 && group == 1 &&
                common->strideX() == 1 && common->strideY() == 1 && common->padX() == 0 && common->padY() == 0;
    std::vector<int> weightShape = {outputCount, inputCount / group, kernelY, kernelX};
    if (mIsDeconv) {
        weightShape = {inputCount, outputCount / group, kernelY, kernelX};
    }
    if (mIsLinear) {
        weightShape = {outputCount, inputCount};
    }
    if (inputs.size() > 1) {
        if (inputs.size() < 3) {
            std::vector<float> zeros(outputCount, 0.0f);
            mBias = buildBlob(zeros.data(), zeros.size() * sizeof(float),
                              CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT32, {outputCount}, "bias");
        }
        return;
    }
    mWeight = mCoreMLBackend->existingWeight(mName + "_weight");
    mBias = mCoreMLBackend->existingWeight(mName + "_bias");
    if ((!mWeight.empty() || !mCoreMLBackend->existingWeight(mName + "_partial_weight_0").empty()) && !mBias.empty()) {
        return;
    }
    const auto key = weightKey(mOp);
    if (mCoreMLBackend->restoreWeights(key)) {
        mWeight = mCoreMLBackend->existingWeight(mName + "_weight");
        mBias = mCoreMLBackend->existingWeight(mName + "_bias");
        return;
    }
    // load weight and bias
    if (loadQuantizedWeight(mCoreMLBackend, mOp, weightShape, mName + "_weight")) {
        mWeight = mName + "_weight";
    } else if (nullptr != conv2D->quanParameter()) {
        auto quanCommon = ConvolutionCommon::load(mOp, backend(), true);
        if (!quanCommon || !quanCommon->weightFloat.get()) {
            mValid = false;
            return;
        }
        auto weightPtr = quanCommon->weightFloat.get();
        auto weightSize = quanCommon->weightFloat.size();
        mWeight = buildBlob(weightPtr, weightSize * sizeof(float), CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT32,
                            weightShape, "weight");
    } else {
        if (!conv2D->weight()) {
            mValid = false;
            return;
        }
        auto weightSize = conv2D->weight()->size();
        auto weightPtr = conv2D->weight()->data();
        mWeight = buildBlob(weightPtr, weightSize * sizeof(float), CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT32,
                            weightShape, "weight");
    }
    // load bias
    {
        std::vector<float> bias(outputCount, 0.0f);
        if (conv2D->bias() && conv2D->bias()->size() == outputCount) {
            std::copy(conv2D->bias()->begin(), conv2D->bias()->end(), bias.begin());
        }
        mBias = buildBlob(bias.data(), bias.size() * sizeof(float), CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT32,
                          {outputCount}, "bias");
    }
    if (!mCoreMLBackend->lowMemory() && !mIsLinear && !mIsDeconv && group == 1 && inputs[0]->dimensions() == 4 &&
        inputCount * kernelY * kernelX > 512) {
        auto weight = mCoreMLBackend->weightTensor(mWeight);
        if (!weight || !weight->host<int16_t>()) {
            mValid = false;
            return;
        }
        int block = reductionBlockSize(common);
        size_t kernel = kernelY * kernelX;
        for (int begin = 0; begin < inputCount; begin += block) {
            int count = std::min(block, inputCount - begin);
            std::vector<int16_t> values(outputCount * count * kernel);
            for (int oc = 0; oc < outputCount; ++oc) {
                ::memcpy(values.data() + oc * count * kernel,
                         weight->host<int16_t>() + (oc * inputCount + begin) * kernel,
                         count * kernel * sizeof(int16_t));
            }
            buildBlob(values.data(), values.size() * sizeof(int16_t),
                      CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT16, {outputCount, count, kernelY, kernelX},
                      "partial_weight_" + std::to_string(begin));
        }
        mCoreMLBackend->releaseWeight(mWeight);
        mWeight.clear();
    }
    if (mValid && !mBias.empty()) {
        mCoreMLBackend->cacheWeights(key);
    }
}

void CoreMLConvolution::addOutput(Operation builder, const std::vector<Tensor*>& outputs) {
    auto common = mOp->main_as_Convolution2D()->common();
    if (!common->relu() && !common->relu6()) {
        builder.output(outputs);
        return;
    }
    auto value = builder.output(outputs[0]->shape(), "pre_activation");
    auto activation = operation(common->relu6() ? "clip" : "relu");
    activation.input("x", value);
    if (common->relu6()) {
        activation.scalar("alpha", 0.0f, "relu_min").scalar("beta", 6.0f, "relu_max");
    }
    activation.output(outputs);
}

ErrorCode CoreMLConvolution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto inputWidth = inputs[0]->width();
    auto inputHeight = inputs[0]->height();
    auto outputWidth = outputs[0]->width();
    auto outputHeight = outputs[0]->height();
    auto batch = inputs[0]->batch();
    auto inputCount = inputs[0]->channel();
    auto conv2D = mOp->main_as_Convolution2D();
    auto common = conv2D->common();
    auto kernelX = common->kernelX();
    auto kernelY = common->kernelY();
    auto outputCount = common->outputCount();
    auto strideX = common->strideX();
    auto strideY = common->strideY();
    auto dilateX = common->dilateX();
    auto dilateY = common->dilateY();
    auto padX = common->padX();
    auto padY = common->padY();
    auto padMod = common->padMode();
    auto group = common->group();
    auto padType = mIsDeconv ? std::string("custom") : padTypeConvert(padMod);
    if (kernelX == 1) {
        dilateX = 1;
    }
    if (kernelY == 1) {
        dilateY = 1;
    }
    auto isConv1d = inputs[0]->shape().size() == 3;
    if (inputs.size() > 1) {
        mWeight = mCoreMLBackend->getTensorName(inputs[1]);
        if (mIsLinear || inputs[1]->dimensions() == 3) {
            auto shape = mIsLinear ? std::vector<int>{outputCount, inputCount} : inputs[1]->shape();
            if (!mIsLinear) {
                shape.push_back(1);
            }
            mWeight =
                operation("reshape").input("x", mWeight).shape("shape", shape, "weight_shape").output(shape, "weight");
        }
        if (inputs.size() > 2) {
            mBias = mCoreMLBackend->getTensorName(inputs[2]);
        }
    }
    if (mIsLinear) {
        // Flatten tokens/spatial positions into the matrix row dimension so
        // Core ML sees one GEMM rather than a batch of single-row products.
        const auto shape = inputs[0]->shape();
        std::vector<int> perm = shape.size() == 3 ? std::vector<int>{0, 2, 1} : std::vector<int>{0, 2, 3, 1};
        std::vector<int> channelLast;
        for (auto axis : perm) {
            channelLast.push_back(shape[axis]);
        }
        auto x = operation("transpose")
                     .input("x", inputs[0])
                     .array<int>("perm", perm, "pre_perm")
                     .output(channelLast, "pre_transpose");
        int rows = 1;
        for (size_t i = 0; i + 1 < channelLast.size(); ++i) {
            rows *= channelLast[i];
        }
        std::vector<int> matrixShape = {rows, inputCount};
        x = operation("reshape")
                .input("x", x)
                .shape("shape", matrixShape, "matrix_shape")
                .output(matrixShape, "matrix_input");
        channelLast.back() = outputCount;
        matrixShape.back() = outputCount;
        auto y = operation("matmul")
                     .input("x", x)
                     .input("y", mWeight)
                     .scalar("transpose_x", false)
                     .scalar("transpose_y", true)
                     .output(matrixShape, "matmul");
        y = operation("add").input("x", y).input("y", mBias).output(matrixShape, "bias_add");
        y = operation("reshape")
                .input("x", y)
                .shape("shape", channelLast, "output_shape")
                .output(channelLast, "spatial_output");
        auto post = operation("transpose");
        post.input("x", y);
        auto postPerm = shape.size() == 3 ? std::vector<int>{0, 2, 1} : std::vector<int>{0, 3, 1, 2};
        post.array<int>("perm", postPerm, "post_perm");
        addOutput(std::move(post), outputs);
        return NO_ERROR;
    }
    // Bound fp16 reduction length and add bias after the balanced partial sums.
    // A long fp16 convolution can lose small products when accumulating into bias.
    if (!mIsDeconv && group == 1 && !isConv1d && inputCount * kernelY * kernelX > 512) {
        int block = reductionBlockSize(common);
        auto x = mCoreMLBackend->getTensorName(inputs[0]);
        auto shape = std::vector<int>{batch, outputCount, outputHeight, outputWidth};
        std::vector<std::string> partials;
        auto pads = ConvolutionCommon::convolutionPad(inputs[0], outputs[0], common);
        int bottom = pads.second, right = pads.first;
        if (padMod == PadMode_CAFFE && common->pads() && common->pads()->size() == 4) {
            bottom = common->pads()->Get(2);
            right = common->pads()->Get(3);
        }
        for (int begin = 0; begin < inputCount; begin += block) {
            int count = std::min(block, inputCount - begin);
            std::string tag = std::to_string(begin);
            auto slice = [&](const std::string& value, const std::vector<int>& size, const std::string& suffix) {
                return operation("slice_by_size")
                    .input("x", value)
                    .array<int>("begin", {0, begin, 0, 0}, suffix + "_begin_" + tag)
                    .array("size", size, suffix + "_size_" + tag)
                    .output(size, suffix + tag);
            };
            auto input = slice(x, {batch, count, inputHeight, inputWidth}, "input");
            auto weight = mCoreMLBackend->existingWeight(mName + "_partial_weight_" + tag);
            if (weight.empty()) {
                weight = slice(mWeight, {outputCount, count, kernelY, kernelX}, "weight");
            }
            partials.push_back(operation("conv")
                                   .input("x", input)
                                   .input("weight", weight)
                                   .scalar("groups", 1, "partial_groups_" + tag)
                                   .scalar("pad_type", padType, "partial_pad_type_" + tag)
                                   .array<int>("strides", {strideY, strideX}, "partial_strides_" + tag)
                                   .array<int>("dilations", {dilateY, dilateX}, "partial_dilations_" + tag)
                                   .array<int>("pad", {pads.second, bottom, pads.first, right}, "partial_pad_" + tag)
                                   .output(shape, "partial_" + tag));
        }
        int serial = 0;
        while (partials.size() > 1) {
            std::vector<std::string> next;
            for (int i = 0; i < partials.size(); i += 2) {
                if (i + 1 == partials.size()) {
                    next.push_back(partials[i]);
                    continue;
                }
                next.push_back(operation("add")
                                   .input("x", partials[i])
                                   .input("y", partials[i + 1])
                                   .output(shape, "sum_" + std::to_string(serial++)));
            }
            partials = std::move(next);
        }
        auto bias = operation("reshape")
                        .input("x", mBias)
                        .shape("shape", {1, outputCount, 1, 1}, "partial_bias_shape")
                        .output({1, outputCount, 1, 1}, "partial_bias");
        auto add = operation("add");
        add.input("x", partials[0]).input("y", bias);
        addOutput(std::move(add), outputs);
        return NO_ERROR;
    }
    std::pair<int, int> pads;
    if (mIsDeconv) {
        pads = ConvolutionCommon::convolutionTransposePad(inputs[0], outputs[0], common);
    } else {
        pads = ConvolutionCommon::convolutionPad(inputs[0], outputs[0], common);
    }
    int top = pads.second;
    int left = pads.first;
    int bottom = pads.second;
    int right = pads.first;
    if (padMod == PadMode_CAFFE && common->pads() && common->pads()->size() == 4) {
        bottom = common->pads()->Get(2);
        right = common->pads()->Get(3);
    }
    auto builder = operation(mIsDeconv ? "conv_transpose" : "conv");
    if (mIsDeconv) {
        builder.array<int>("output_shape", {batch, outputCount, outputHeight, outputWidth});
    }
    if (isConv1d) {
        // unsqueeze to 2d
        builder.input("x", operation("expand_dims")
                               .input("x", inputs[0])
                               .array<int>("axes", {3})
                               .output({batch, inputCount, inputHeight, 1}, "expand"));
    } else {
        builder.input("x", inputs[0]);
    }
    builder.input("weight", mWeight).input("bias", mBias).scalar("groups", group).scalar("pad_type", padType);
    builder.array<int>("strides", {strideY, strideX})
        .array<int>("dilations", {dilateY, dilateX})
        .array<int>("pad", {top, bottom, left, right});
    if (isConv1d) {
        auto conv2d = builder.output({batch, outputCount, outputHeight, outputWidth}, "conv2d");
        auto squeeze = operation("squeeze");
        squeeze.input("x", conv2d).array<int>("axes", {3});
        addOutput(std::move(squeeze), outputs);
    } else {
        addOutput(std::move(builder), outputs);
    }
    return NO_ERROR;
}

CoreMLConvolution3D::CoreMLConvolution3D(MNN::Backend* b, const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                         const std::vector<MNN::Tensor*>& outputs)
    : CoreMLCommonExecution(b, op) {
    auto conv3D = mOp->main_as_Convolution3D();
    auto common = conv3D->common();
    auto kernels = common->kernels();
    auto inputCount = common->inputCount();
    auto outputCount = common->outputCount();
    auto kernelZ = kernels->Get(0);
    auto kernelY = kernels->Get(1);
    auto kernelX = kernels->Get(2);
    auto group = common->group();
    if (group <= 0 || inputCount % group || outputCount % group) {
        mValid = false;
        return;
    }
    std::vector<int> weightShape = {outputCount, inputCount / group, kernelZ, kernelY, kernelX};
    auto weightSize = conv3D->weight()->size();
    auto weightPtr = conv3D->weight()->data();
    mWeight = buildBlob(weightPtr, weightSize * sizeof(float), CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT32,
                        weightShape, "weight");
    auto biasSize = conv3D->bias()->size();
    auto biasPtr = conv3D->bias()->data();
    mBias = buildBlob(biasPtr, biasSize * sizeof(float), CORE_ML__SPECIFICATION__MILSPEC__DATA_TYPE__FLOAT32,
                      {outputCount}, "bias");
}

ErrorCode CoreMLConvolution3D::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const Tensor* input = inputs[0];
    const Tensor* output = outputs[0];
    auto conv3D = mOp->main_as_Convolution3D();
    auto common = conv3D->common();
    auto dilates = common->dilates();
    auto strides = common->strides();
    auto kernels = common->kernels();
    auto pads = common->pads();
    auto padMod = common->padMode();
    auto group = common->group();
    auto inputCount = common->inputCount();
    auto outputCount = common->outputCount();
    auto padType = padTypeConvert(padMod);
    // padType = "valid"; // conv3d
    auto strideZ = strides->Get(0);
    auto strideY = strides->Get(1);
    auto strideX = strides->Get(2);
    auto dilateZ = dilates->Get(0);
    auto dilateY = dilates->Get(1);
    auto dilateX = dilates->Get(2);
    auto padZ = pads->Get(0);
    auto padY = pads->Get(1);
    auto padX = pads->Get(2);
    int endZ = pads->size() == 6 ? pads->Get(3) : padZ;
    int endY = pads->size() == 6 ? pads->Get(4) : padY;
    int endX = pads->size() == 6 ? pads->Get(5) : padX;
    auto builder = operation("conv");
    builder.input("x", inputs[0])
        .input("weight", mWeight)
        .input("bias", mBias)
        .scalar("groups", group)
        .scalar("pad_type", padType)
        .array<int>("strides", {strideZ, strideY, strideX})
        .array<int>("dilations", {dilateZ, dilateY, dilateX})
        .array<int>("pad", {padZ, endZ, padY, endY, padX, endX});
    if (common->relu() || common->relu6()) {
        auto value = builder.output(outputs[0]->shape(), "pre_activation");
        auto activation = operation(common->relu6() ? "clip" : "relu");
        activation.input("x", value);
        if (common->relu6()) {
            activation.scalar("alpha", 0.0f, "relu_min").scalar("beta", 6.0f, "relu_max");
        }
        activation.output(outputs);
    } else {
        builder.output(outputs);
    }
    return NO_ERROR;
}

REGISTER_COREML_OP_CREATOR(CoreMLConvolution, OpType_Convolution);
REGISTER_COREML_OP_CREATOR(CoreMLConvolution, OpType_ConvolutionDepthwise);
REGISTER_COREML_OP_CREATOR(CoreMLConvolution, OpType_Deconvolution);
REGISTER_COREML_OP_CREATOR(CoreMLConvolution, OpType_DeconvolutionDepthwise);
REGISTER_COREML_OP_CREATOR(CoreMLConvolution3D, OpType_Convolution3D);
} // namespace MNN
