//
//  IndexedLinearCommon.cpp
//  MNN
//
//  Created by MNN on 2026/09/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

#include "core/IndexedLinearCommon.hpp"
#include "core/FileLoader.hpp"
#include <climits>
#include <cstring>

namespace MNN {

bool IndexedLinearCommon::dimensions(const Op* op, int& experts, int& inputChannels, int& outputChannels) {
    if (!op || op->main_type() != OpParameter_IndexedLinearParam) {
        return false;
    }
    auto param = op->main_as_IndexedLinearParam();
    if (!param || !param->linears() || param->linears()->size() == 0 ||
        (param->inputMode() != 0 && param->inputMode() != 1) ||
        (param->fusedActivation() != -1 && param->fusedActivation() != UnaryOpOperation_SILU)) {
        return false;
    }
    experts = param->linears()->size();
    inputChannels = outputChannels = 0;
    for (auto expert : *param->linears()) {
        auto c = expert ? expert->common() : nullptr;
        if (!c || c->inputCount() <= 0 || c->outputCount() <= 0 || c->kernelX() != 1 || c->kernelY() != 1 ||
            c->group() != 1 || c->relu() || c->relu6()) {
            return false;
        }
        if (inputChannels && (inputChannels != c->inputCount() || outputChannels != c->outputCount())) {
            return false;
        }
        inputChannels = c->inputCount();
        outputChannels = c->outputCount();
    }
    if (param->fusedActivation() != -1 && outputChannels % 2) {
        return false;
    }
    return true;
}

namespace {
std::shared_ptr<ConvolutionCommon::Int8Common> loadProjection(const Op* op, const Convolution2D* conv,
                                                              bool forceFloat) {
    const int inputChannels = conv->common()->inputCount(), outputChannels = conv->common()->outputCount();
    if ((int64_t)inputChannels * outputChannels > INT_MAX) {
        return nullptr;
    }
    if (!conv->quanParameter() && conv->weight() && conv->weight()->size() == inputChannels * outputChannels) {
        std::shared_ptr<ConvolutionCommon::Int8Common> result(new ConvolutionCommon::Int8Common);
        result->quan = nullptr;
        result->asymmetric = false;
        result->alphaSize = 0;
        result->weightFloat.reset(inputChannels * outputChannels);
        if (!result->weightFloat.get()) {
            return nullptr;
        }
        ::memcpy(result->weightFloat.get(), conv->weight()->data(), result->weightFloat.size() * sizeof(float));
        return result;
    }
    std::unique_ptr<Convolution2DT> param(conv->UnPack());
    if (!param->quanParameter) {
        if (param->external.size() < 3 || !op->externalPath()) {
            return nullptr;
        }
        param->quanParameter.reset(new IDSTQuanT);
        param->quanParameter->type = 8; // Existing external FP32 weight representation.
    }
    flatbuffers::FlatBufferBuilder builder;
    auto weight = Convolution2D::Pack(builder, param.get());
    auto path = op->externalPath() ? builder.CreateString(op->externalPath()->str())
                                   : flatbuffers::Offset<flatbuffers::String>();
    OpBuilder opBuilder(builder);
    opBuilder.add_type(OpType_Convolution);
    opBuilder.add_main_type(OpParameter_Convolution2D);
    opBuilder.add_main(weight.Union());
    opBuilder.add_externalPath(path);
    builder.Finish(opBuilder.Finish());
    // Indexed banks own their static resources. Read the source file even when the
    // ordinary convolution backend uses an architecture-specific mmap cache.
    auto result = ConvolutionCommon::load(flatbuffers::GetRoot<Op>(builder.GetBufferPointer()), nullptr, forceFloat,
                                          !forceFloat, nullptr, true);
    if (result && result->weightFloat.get() && result->weightFloat.size() != (int64_t)inputChannels * outputChannels) {
        MNN_ERROR("IndexedLinear: invalid floating weight length\n");
        return nullptr;
    }
    if (result) {
        result->quan = conv->quanParameter();
    }
    return result;
}

bool loadProjectionBias(const Op* op, const Convolution2D* conv, std::vector<float>& bias) {
    const int outputChannels = conv->common()->outputCount();
    bias.assign(outputChannels, 0.0f);
    if (conv->bias() && conv->bias()->size()) {
        if (conv->bias()->size() != outputChannels) {
            return false;
        }
        ::memcpy(bias.data(), conv->bias()->data(), outputChannels * sizeof(float));
        return true;
    }
    auto external = conv->external();
    const int biasIndex = conv->quanParameter() ? 3 : 2;
    if (!external || external->size() <= biasIndex || external->Get(biasIndex) == 0) {
        return true;
    }
    if (!op->externalPath() || external->Get(biasIndex) != (int64_t)outputChannels * sizeof(float)) {
        return false;
    }
    int64_t offset = external->Get(0);
    for (int i = 1; i < biasIndex; ++i) {
        auto size = external->Get(i);
        if (offset < 0 || size < 0 || offset > INT64_MAX - size) {
            return false;
        }
        offset += size;
    }
    FileLoader loader(op->externalPath()->c_str(), true);
    return offset >= 0 && loader.valid() && loader.offset(offset) == 0 &&
           loader.read(reinterpret_cast<char*>(bias.data()), outputChannels * sizeof(float));
}
} // namespace

std::shared_ptr<ConvolutionCommon::Int8Common> IndexedLinearCommon::loadLinear(const Op* op, int expert,
                                                                               Backend* backend, bool forceFloat) {
    int count, inputChannels, outputChannels;
    if (!dimensions(op, count, inputChannels, outputChannels) || expert < 0 || expert >= count ||
        (int64_t)inputChannels * outputChannels > INT_MAX) {
        return nullptr;
    }
    auto param = op->main_as_IndexedLinearParam();
    auto gate = loadProjection(op, param->linears()->Get(expert), forceFloat);
    if (!gate || param->fusedActivation() == -1) {
        return gate;
    }
    if (gate->canUseInt2 || gate->canUseInt3) {
        return nullptr;
    }
    const int channels = outputChannels / 2;
    const int sourceOutputs = outputChannels;
    const bool floating = gate->weightFloat.get() != nullptr;
    const int components = gate->asymmetric ? 2 : 1;
    const int gateComponents = components;
    const int groups = floating ? 0 : gate->alphaSize / sourceOutputs / gateComponents;
    const bool halfAlpha = !floating && gate->alphaIsFp16 && gate->alphaHalf.get() && !gate->alpha.get();
    if ((int64_t)outputChannels * groups * components > INT_MAX)
        return nullptr;
    if (!floating &&
        (groups <= 0 || inputChannels % groups || gate->alphaSize != sourceOutputs * groups * gateComponents ||
         (!halfAlpha && !gate->getAlphaFloat()))) {
        return nullptr;
    }
    std::shared_ptr<ConvolutionCommon::Int8Common> result(new ConvolutionCommon::Int8Common);
    result->quan = gate->quan;
    result->asymmetric = components == 2;
    result->canUseInt4 = gate->canUseInt4;
    result->alphaSize = outputChannels * groups * components;
    const int elements = inputChannels * outputChannels;
    const int bits = result->canUseInt4 ? 4 : 8;
    if (floating) {
        result->weightFloat.reset(elements);
        if (!result->weightFloat.get())
            return nullptr;
    } else {
        const int64_t sourceBytes = ((int64_t)sourceOutputs * inputChannels * bits + 7) / 8;
        if (!gate->weight.get() || gate->weight.size() < sourceBytes)
            return nullptr;
        result->weight.reset(elements * (int64_t)bits / 8);
        result->alphaIsFp16 = halfAlpha;
        if (halfAlpha) {
            result->alphaHalf.reset(result->alphaSize);
        } else {
            result->alpha.reset(result->alphaSize);
        }
        if (!result->weight.get() || (halfAlpha ? !result->alphaHalf.get() : !result->alpha.get()))
            return nullptr;
        ::memset(result->weight.get(), 0, result->weight.size());
    }
    // Pair the merged gate/up output rows after decoding the serialized linear.
    // This is a lossless reorder of quantized integers and their original scales,
    // not a new quantization. Pairs stay inside every backend's output tile.
    for (int channel = 0; channel < channels; ++channel) {
        for (int projection = 0; projection < 2; ++projection) {
            const auto& source = gate;
            const int sourceChannel = channel + projection * channels;
            const int targetChannel = channel * 2 + projection;
            if (floating) {
                ::memcpy(result->weightFloat.get() + targetChannel * inputChannels,
                         source->weightFloat.get() + sourceChannel * inputChannels, inputChannels * sizeof(float));
                continue;
            }
            for (int input = 0; input < inputChannels; ++input) {
                const int src = sourceChannel * inputChannels + input, dst = targetChannel * inputChannels + input;
                if (bits == 4) {
                    const auto packed = reinterpret_cast<const uint8_t*>(source->weight.get());
                    const int value = src % 2 ? packed[src / 2] & 15 : packed[src / 2] >> 4;
                    auto target = reinterpret_cast<uint8_t*>(result->weight.get());
                    target[dst / 2] |= dst % 2 ? value : value << 4;
                } else {
                    result->weight.get()[dst] = source->weight.get()[src];
                }
            }
            const int sourceComponents = source->asymmetric ? 2 : 1;
            const auto alpha = halfAlpha ? nullptr : source->getAlphaFloat();
            for (int block = 0; block < groups; ++block) {
                const int src = (sourceChannel * groups + block) * sourceComponents;
                const int dst = (targetChannel * groups + block) * components;
                if (halfAlpha) {
                    result->alphaHalf.get()[dst + components - 1] = source->alphaHalf.get()[src + sourceComponents - 1];
                    if (components == 2)
                        result->alphaHalf.get()[dst] = source->asymmetric ? source->alphaHalf.get()[src] : 0;
                } else {
                    result->alpha.get()[dst + components - 1] = alpha[src + sourceComponents - 1];
                    if (components == 2)
                        result->alpha.get()[dst] = source->asymmetric ? alpha[src] : 0.0f;
                }
            }
        }
    }
    return result;
}

bool IndexedLinearCommon::loadBias(const Op* op, int expert, std::vector<float>& bias) {
    int count, inputs, outputs;
    if (!dimensions(op, count, inputs, outputs) || expert < 0 || expert >= count)
        return false;
    auto param = op->main_as_IndexedLinearParam();
    std::vector<float> gate;
    if (!loadProjectionBias(op, param->linears()->Get(expert), gate))
        return false;
    if (param->fusedActivation() == -1) {
        bias = std::move(gate);
        return true;
    }
    bias.resize(outputs);
    for (int channel = 0; channel < outputs / 2; ++channel) {
        bias[channel * 2] = gate[channel];
        bias[channel * 2 + 1] = gate[channel + outputs / 2];
    }
    return true;
}

} // namespace MNN

#endif // MNN_SUPPORT_TRANSFORMER_FUSE
