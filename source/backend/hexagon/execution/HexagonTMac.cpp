#include "HexagonTMac.hpp"

#include <algorithm>
#include <cstring>

#include "HexagonBackend.hpp"
#include "core/Macro.h"
#include "htp_command.h"

namespace MNN {

static bool hasNonZeroBias(const float* bias, int size) {
    if (bias == nullptr || size <= 0) {
        return false;
    }
    for (int i = 0; i < size; ++i) {
        if (bias[i] != 0.0f) {
            return true;
        }
    }
    return false;
}

static void reorderTMacWeightForHvx(uint8_t* dst, const uint8_t* src, int ic, int oc, int scaleBlockNum) {
    const int ocPack = 128;
    const int blockSize = ic / scaleBlockNum;
    const int weightBlockBytes = blockSize / 8;
    const int weightBlockCount = scaleBlockNum * weightBlockBytes;
    const int ocP = UP_DIV(oc, ocPack);
    ::memset(dst, 0, (size_t)ocP * weightBlockCount * 128);
    for (int oz = 0; oz < ocP; ++oz) {
        uint8_t* dstOz = dst + (size_t)oz * weightBlockCount * 128;
        for (int oy = 0; oy < ocPack; ++oy) {
            const int o = oz * ocPack + oy;
            if (o >= oc) {
                continue;
            }
            const uint8_t* srcO = src + (size_t)o * weightBlockCount;
            const int lane = oy < 64 ? oy * 2 : (oy - 64) * 2 + 1;
            for (int k = 0; k < weightBlockCount; ++k) {
                dstOz[(size_t)k * 128 + lane] = srcO[k];
            }
        }
    }
}

static void packTMacScaleForHvx(float* dst, const float* src, int oc, int scaleBlockNum, bool asymmetric) {
    const int ocPack = 128;
    const int vecLanes = 32;
    const int vectorsPerPack = 8;
    const int ocP = UP_DIV(oc, ocPack);
    ::memset(dst, 0, (size_t)ocP * scaleBlockNum * vectorsPerPack * vecLanes * sizeof(float));
    for (int oz = 0; oz < ocP; ++oz) {
        for (int block = 0; block < scaleBlockNum; ++block) {
            float* pack = dst + ((size_t)oz * scaleBlockNum + block) * vectorsPerPack * vecLanes;
            for (int lane = 0; lane < vecLanes; ++lane) {
                const int ocIndex[4] = {oz * ocPack + lane * 2, oz * ocPack + lane * 2 + 1, oz * ocPack + 64 + lane * 2,
                                        oz * ocPack + 64 + lane * 2 + 1};
                for (int part = 0; part < 4; ++part) {
                    if (ocIndex[part] >= oc) {
                        continue;
                    }
                    float alpha = 0.0f;
                    float offsetFactor = 0.0f;
                    const int scaleIndex = ocIndex[part] * scaleBlockNum + block;
                    if (asymmetric) {
                        const float minValue = src[2 * scaleIndex + 0];
                        alpha = src[2 * scaleIndex + 1];
                        offsetFactor = minValue - alpha;
                    } else {
                        alpha = src[scaleIndex];
                        offsetFactor = -alpha;
                    }
                    pack[(part * 2 + 0) * vecLanes + lane] = alpha;
                    pack[(part * 2 + 1) * vecLanes + lane] = offsetFactor;
                }
            }
        }
    }
}

HexagonTMac::Resource::~Resource() {
    if (allocator == nullptr) {
        return;
    }
    if (weight.first != nullptr) {
        allocator->free(weight);
    }
    if (scale.first != nullptr) {
        allocator->free(scale);
    }
    if (bias.first != nullptr) {
        allocator->free(bias);
    }
}

HexagonTMac::HexagonTMac(Backend* backend, std::shared_ptr<Resource> res, const Op* op)
    : HexagonExecution(backend), mResource(std::move(res)) {
    auto conv2d = op != nullptr ? op->main_as_Convolution2D() : nullptr;
    auto common = conv2d != nullptr ? conv2d->common() : nullptr;
    if (common != nullptr) {
        mRelu = common->relu() ? 1 : 0;
        mRelu6 = common->relu6() ? 1 : 0;
    }
}

HexagonTMac* HexagonTMac::create(Backend* backend, const Op* op, const std::vector<Tensor*>& inputs,
                                 const std::vector<Tensor*>& outputs) {
    auto conv2d = op != nullptr ? op->main_as_Convolution2D() : nullptr;
    if (conv2d == nullptr || conv2d->common() == nullptr || conv2d->quanParameter() == nullptr) {
        return nullptr;
    }
    auto common = conv2d->common();
    if (common->kernelY() != 1 || common->kernelX() != 1 || common->strideX() != 1 || common->strideY() != 1 ||
        common->relu() || common->relu6()) {
        return nullptr;
    }
    if (inputs.size() != 1 || outputs.size() != 1 || inputs[0] == nullptr || outputs[0] == nullptr) {
        return nullptr;
    }
    if (HexagonBackend::getBytes(inputs[0]) != 2 || HexagonBackend::getBytes(outputs[0]) != 2) {
        return nullptr;
    }
    int ic = common->inputCount();
    const int oc = common->outputCount();
    if (ic <= 0 || oc <= 0 || (ic % 32) != 0 || (oc % 32) != 0) {
        return nullptr;
    }

    auto quanCommon = ConvolutionCommon::load(op, backend, false, true, nullptr);
    if (!quanCommon || quanCommon->originBits != 1 || quanCommon->alpha.get() == nullptr ||
        quanCommon->weight.get() == nullptr) {
        return nullptr;
    }
    const int alphaSize = quanCommon->alpha.size();
    const int alphaUnit = quanCommon->asymmetric ? 2 : 1;
    if (alphaSize < oc * alphaUnit || alphaSize % (oc * alphaUnit) != 0) {
        return nullptr;
    }
    const int scaleBlockNum = alphaSize / (oc * alphaUnit);
    if (scaleBlockNum != 1 || (ic % scaleBlockNum) != 0) {
        return nullptr;
    }
    const int blockSize = ic / scaleBlockNum;
    if ((blockSize & 7) != 0) {
        return nullptr;
    }
    const size_t rawWeightSize = (size_t)oc * scaleBlockNum * (blockSize / 8);
    if ((size_t)quanCommon->weight.size() < rawWeightSize) {
        return nullptr;
    }

    auto hexBackend = static_cast<HexagonBackend*>(backend);
    auto bufferAlloc = hexBackend->getAllocator(2);
    std::shared_ptr<Resource> res(new Resource);
    res->allocator = bufferAlloc;
    res->inputChannels = ic;
    res->outputChannels = oc;
    res->scaleBlockNum = scaleBlockNum;
    res->asymmetric = quanCommon->asymmetric ? 1 : 0;

    const int packedOcP = UP_DIV(oc, 128);
    const int packedWeightBlockCount = scaleBlockNum * (blockSize / 8);
    const size_t packedWeightSize = (size_t)packedOcP * packedWeightBlockCount * 128;
    const size_t rawScaleSize = (size_t)alphaSize * sizeof(float);
    const size_t packedScaleSize = (size_t)packedOcP * scaleBlockNum * 8 * 32 * sizeof(float);
    res->weight = bufferAlloc->alloc(packedWeightSize);
    res->scale = bufferAlloc->alloc(rawScaleSize + packedScaleSize);
    if (res->weight.first == nullptr || res->scale.first == nullptr) {
        return nullptr;
    }
    reorderTMacWeightForHvx(HexagonBackend::getPtr(res->weight),
                            reinterpret_cast<const uint8_t*>(quanCommon->weight.get()), ic, oc, scaleBlockNum);
    auto scalePtr = reinterpret_cast<float*>(HexagonBackend::getPtr(res->scale));
    ::memcpy(scalePtr, quanCommon->alpha.get(), rawScaleSize);
    packTMacScaleForHvx(scalePtr + alphaSize, quanCommon->alpha.get(), oc, scaleBlockNum, quanCommon->asymmetric);
    hexBackend->markHostInput(res->weight, (int)packedWeightSize);
    hexBackend->markHostInput(res->scale, (int)(rawScaleSize + packedScaleSize));

    const float* originBias = nullptr;
    int originBiasSize = 0;
    if (conv2d->bias() != nullptr) {
        originBias = conv2d->bias()->data();
        originBiasSize = conv2d->bias()->size();
    }
    res->hasBias = hasNonZeroBias(originBias, std::min(oc, originBiasSize));
    if (res->hasBias) {
        const int biasRound = UP_DIV(oc, 64) * 64;
        const int biasSize = biasRound * (int)sizeof(int16_t) + 64;
        res->bias = bufferAlloc->alloc((size_t)biasSize);
        if (res->bias.first == nullptr) {
            return nullptr;
        }
        auto biasPtr = HexagonBackend::getPtr(res->bias);
        ::memset(biasPtr, 0, (size_t)biasSize);
        HexagonBackend::fp32ToFp16(originBias, reinterpret_cast<int16_t*>(biasPtr), std::min(oc, originBiasSize));
        hexBackend->markHostInput(res->bias, biasSize);
    }

    return new HexagonTMac(backend, res, op);
}

ErrorCode HexagonTMac::onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                  std::vector<HexagonCommand>& dst) {
    if (inputs.size() != 1 || outputs.size() != 1 || mResource == nullptr) {
        return NOT_SUPPORT;
    }
    auto input = inputs[0];
    auto output = outputs[0];
    if (input == nullptr || output == nullptr) {
        return INPUT_DATA_ERROR;
    }
    const int area = output->batch() * (output->dimensions() > 2 ? output->height() : 1) *
                     (output->dimensions() > 3 ? output->width() : 1);
    if (area <= 0 || HexagonBackend::getBytes(input) != 2 || HexagonBackend::getBytes(output) != 2) {
        return NOT_SUPPORT;
    }

    auto inputDev = HexagonBackend::getDevicePtr(input);
    auto weightDev = HexagonBackend::getDevicePtr(mResource->weight);
    auto scaleDev = HexagonBackend::getDevicePtr(mResource->scale);
    std::pair<int, int> biasDev = {-1, 0};
    if (mResource->hasBias) {
        biasDev = HexagonBackend::getDevicePtr(mResource->bias);
    }
    auto outputDev = HexagonBackend::getDevicePtr(output);

    int params[] = {area,
                    mResource->inputChannels,
                    mResource->outputChannels,
                    mResource->scaleBlockNum,
                    mResource->asymmetric,
                    mRelu,
                    mRelu6,
                    (int32_t)static_cast<HexagonBackend*>(backend())->getSize(output)};
    std::vector<std::pair<int, int>> inputFds = {inputDev, weightDev, scaleDev, biasDev};
    std::vector<std::pair<int, int>> outputFds = {outputDev};

    dst.emplace_back();
    dst.back().build(static_cast<HexagonBackend*>(backend()), DSP_OP_TMAC_A16W1, params, sizeof(params), inputFds,
                     outputFds, inputs, outputs);
    return NO_ERROR;
}

bool HexagonTMac::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (dst == nullptr) {
        return true;
    }
    auto exe = new HexagonTMac(bn, mResource, op);
    exe->mRelu = mRelu;
    exe->mRelu6 = mRelu6;
    *dst = exe;
    return true;
}

} // namespace MNN
