#include "HexagonDeconvolution.hpp"

#include <algorithm>
#include <climits>
#include <cstring>
#include <vector>
#include "HexagonBackend.hpp"
#include "HexagonRuntime.hpp"
#include "MNN_generated.h"
#include "backend/hexagon/htp-ops-lib/include/dsp/ops.h"
#include "htp_command.h"

namespace MNN {
static_assert(sizeof(ConvolutionCommon::Im2ColParameter) == sizeof(Im2ColParameter), "Im2ColParameter layout mismatch");

namespace {

struct TileShape {
    int mp = 1;
    int np = 1;
};

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

static TileShape chooseIm2ColTileShape(int totalMp, int totalNp, int kAlign, int availSize) {
    int maxSum = availSize / (64 * kAlign);
    maxSum = std::max(maxSum, 2);
    TileShape best;
    int64_t bestCost = INT64_MAX;
    int bestChunkPairs = INT_MAX;
    int bestTileArea = 0;
    const int maxMp = std::min(totalMp, maxSum - 1);
    for (int candMp = 1; candMp <= maxMp; ++candMp) {
        const int maxNp = std::min(totalNp, maxSum - candMp);
        for (int candNp = 1; candNp <= maxNp; ++candNp) {
            const int oxChunks = UP_DIV(totalMp, candMp);
            const int oyChunks = UP_DIV(totalNp, candNp);
            const int64_t activationOuterCost = (int64_t)totalMp + (int64_t)oxChunks * totalNp;
            const int64_t weightOuterCost = (int64_t)oyChunks * totalMp + (int64_t)totalNp;
            const int64_t cost = std::min(activationOuterCost, weightOuterCost);
            const int chunkPairs = oxChunks * oyChunks;
            const int tileArea = candMp * candNp;
            if (cost < bestCost ||
                (cost == bestCost && chunkPairs < bestChunkPairs) ||
                (cost == bestCost && chunkPairs == bestChunkPairs && tileArea > bestTileArea)) {
                bestCost = cost;
                bestChunkPairs = chunkPairs;
                bestTileArea = tileArea;
                best.mp = candMp;
                best.np = candNp;
            }
        }
    }
    return best;
}

static void reorderFp16WeightForHmx(int16_t* dst, const int16_t* src, int ic, int oc, int kernelX, int kernelY) {
    constexpr int icPack = 32;
    constexpr int ocPack = 32;
    const int icP = UP_DIV(ic, icPack);
    const int ocP = UP_DIV(oc, ocPack);
    const int kp = kernelY * kernelX * icP;
    constexpr int packs = icPack * ocPack;
    const size_t reorderedSize = (size_t)ocP * kp * packs;
    if (icP * icPack != ic || ocP * ocPack != oc) {
        ::memset(dst, 0, reorderedSize * sizeof(int16_t));
    }
    for (int oz = 0; oz < ocP; ++oz) {
        for (int kk = 0; kk < kp; ++kk) {
            const int kernelIndex = kk / icP;
            const int iz = kk % icP;
            const int ky = kernelIndex / kernelX;
            const int kx = kernelIndex % kernelX;
            const size_t blockBase = ((size_t)oz * kp + kk) * packs;
            for (int oy = 0; oy < ocPack; ++oy) {
                const int o = oz * ocPack + oy;
                if (o >= oc) {
                    continue;
                }
                for (int ix = 0; ix < icPack; ++ix) {
                    const int i = iz * icPack + ix;
                    if (i >= ic) {
                        continue;
                    }
                    const size_t srcIndex = (((size_t)o * ic + i) * kernelY + ky) * kernelX + kx;
                    const int ixPair = ix / 2;
                    const int ixRem = ix & 1;
                    const size_t dstIndex = blockBase + (size_t)ixPair * 64 + oy * 2 + ixRem;
                    dst[dstIndex] = src[srcIndex];
                }
            }
        }
    }
}

static void transformDeconvWeightToConv(float* dst, const float* src, int inputChannels, int outputChannels,
                                        int kernelY, int kernelX) {
    for (int ic = 0; ic < inputChannels; ++ic) {
        for (int oc = 0; oc < outputChannels; ++oc) {
            for (int ky = 0; ky < kernelY; ++ky) {
                const int convKy = kernelY - 1 - ky;
                for (int kx = 0; kx < kernelX; ++kx) {
                    const int convKx = kernelX - 1 - kx;
                    const size_t srcIndex = (((size_t)ic * outputChannels + oc) * kernelY + ky) * kernelX + kx;
                    const size_t dstIndex = (((size_t)oc * inputChannels + ic) * kernelY + convKy) * kernelX + convKx;
                    dst[dstIndex] = src[srcIndex];
                }
            }
        }
    }
}

static void setDeconvAsConvIm2ColParameter(ConvolutionCommon::Im2ColParameter& param,
                                           const Convolution2DCommon* common, Tensor* input, Tensor* output,
                                           int convPadX, int convPadY, int pack) {
    ::memset(&param, 0, sizeof(param));
    param.dilateX = 1;
    param.dilateY = 1;
    param.strideX = 1;
    param.strideY = 1;
    param.icDiv4 = UP_DIV(input->channel(), pack);
    param.kernelX = common->kernelX();
    param.kernelY = common->kernelY();
    param.padX = convPadX;
    param.padY = convPadY;
    param.ih = input->height();
    param.iw = input->width();
    param.oh = output->height();
    param.ow = output->width();
    param.srcZStep = input->stride(1) * pack * input->batch();
    param.srcYStep = input->stride(2) * pack;
    param.packCUnit = pack;
    param.ic = input->channel();
    param.icup4 = UP_DIV(input->channel(), 32) * 32;

}

} // namespace

HexagonDeconvolution::Resource::~Resource() {
    if (allocator != nullptr) {
        if (weight.first != nullptr) {
            allocator->free(weight);
        }
        if (bias.first != nullptr) {
            allocator->free(bias);
        }
    }
}

HexagonDeconvolution::HexagonDeconvolution(Backend* backend, std::shared_ptr<Resource> res, const Op* op)
    : HexagonExecution(backend), mResource(std::move(res)), mOp(op) {
}

HexagonDeconvolution* HexagonDeconvolution::create(Backend* backend, const Op* op,
                                                   const std::vector<Tensor*>& inputs,
                                                   const std::vector<Tensor*>& outputs) {
    if (op == nullptr || op->type() != OpType_Deconvolution || inputs.size() != 1 || outputs.size() != 1 ||
        HexagonRuntime::getDstFunctions() == nullptr) {
        return nullptr;
    }
    if (inputs[0] == nullptr || outputs[0] == nullptr || inputs[0]->dimensions() != 4 || outputs[0]->dimensions() != 4) {
        return nullptr;
    }
    auto conv2d = op->main_as_Convolution2D();
    if (conv2d == nullptr || conv2d->common() == nullptr) {
        return nullptr;
    }
    auto common = conv2d->common();
    if (common->group() != 1 || common->strideX() != 1 || common->strideY() != 1 ||
        common->dilateX() != 1 || common->dilateY() != 1) {
        return nullptr;
    }
    if (!((common->kernelX() == 1 && common->kernelY() == 4) ||
          (common->kernelX() == 3 && common->kernelY() == 3))) {
        return nullptr;
    }
    int inputChannels = common->inputCount();
    int outputChannels = common->outputCount();
    const int kernelX = common->kernelX();
    const int kernelY = common->kernelY();
    if (inputChannels <= 0) {
        inputChannels = inputs[0]->channel();
    }
    if (inputChannels <= 0 || outputChannels <= 0 || kernelX <= 0 || kernelY <= 0) {
        return nullptr;
    }

    const float* originWeight = nullptr;
    int originWeightSize = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    ConvolutionCommon::getConvParameters(&quanCommon, backend, op, &originWeight, &originWeightSize);
    const size_t expectedWeightSize = (size_t)inputChannels * outputChannels * kernelY * kernelX;
    if (originWeight == nullptr || (size_t)originWeightSize < expectedWeightSize) {
        return nullptr;
    }

    auto allocator = static_cast<HexagonBackend*>(backend)->getAllocator(2);
    std::shared_ptr<Resource> res(new Resource);
    res->allocator = allocator;
    res->inputChannels = inputChannels;
    res->outputChannels = outputChannels;
    res->kernelX = kernelX;
    res->kernelY = kernelY;

    const float* originBias = nullptr;
    int originBiasSize = 0;
    if (conv2d->bias() != nullptr) {
        originBias = conv2d->bias()->data();
        originBiasSize = conv2d->bias()->size();
    }
    constexpr int pack = 32;
    const int ocP = UP_DIV(outputChannels, pack);
    res->hasBias = hasNonZeroBias(originBias, std::min(outputChannels, originBiasSize));
    if (res->hasBias) {
        const int biasSize = ocP * pack * (int)sizeof(int16_t) + 64;
        res->bias = allocator->alloc((size_t)biasSize);
        if (res->bias.first == nullptr) {
            return nullptr;
        }
        auto biasPtr = HexagonBackend::getPtr(res->bias);
        ::memset(biasPtr, 0, (size_t)biasSize);
        HexagonBackend::fp32ToFp16(originBias, (int16_t*)biasPtr, std::min(outputChannels, originBiasSize));
        static_cast<HexagonBackend*>(backend)->markHostInput(res->bias, biasSize);
    }

    const int icP = UP_DIV(inputChannels, pack);
    const size_t reorderedWeightSize = (size_t)ocP * icP * kernelY * kernelX * pack * pack;
    res->weight = allocator->alloc(reorderedWeightSize * sizeof(int16_t));
    if (res->weight.first == nullptr) {
        return nullptr;
    }
    std::vector<float> convWeight(expectedWeightSize);
    transformDeconvWeightToConv(convWeight.data(), originWeight, inputChannels, outputChannels, kernelY, kernelX);
    std::vector<int16_t> tempWeight(expectedWeightSize);
    HexagonBackend::fp32ToFp16(convWeight.data(), tempWeight.data(), tempWeight.size());
    reorderFp16WeightForHmx((int16_t*)HexagonBackend::getPtr(res->weight), tempWeight.data(),
                            inputChannels, outputChannels, kernelX, kernelY);
    static_cast<HexagonBackend*>(backend)->markHostInput(res->weight, (int)(reorderedWeightSize * sizeof(int16_t)));

    return new HexagonDeconvolution(backend, res, op);
}

ErrorCode HexagonDeconvolution::onBuildCmd(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                           std::vector<HexagonCommand>& dst) {
    if (inputs.size() != 1 || outputs.size() != 1 || mResource == nullptr || mOp == nullptr) {
        return NOT_SUPPORT;
    }
    auto input = inputs[0];
    auto output = outputs[0];
    if (input == nullptr || output == nullptr || input->dimensions() != 4 || output->dimensions() != 4) {
        return NOT_SUPPORT;
    }
    if (input->getType().code != halide_type_float || output->getType().code != halide_type_float ||
        HexagonBackend::getBytes(input) != 2 || HexagonBackend::getBytes(output) != 2) {
        return NOT_SUPPORT;
    }
    auto conv2d = mOp->main_as_Convolution2D();
    if (conv2d == nullptr || conv2d->common() == nullptr) {
        return NOT_SUPPORT;
    }
    auto common = conv2d->common();
    auto deconvPad = ConvolutionCommon::convolutionTransposePad(input, output, common);
    const int convPadX = common->kernelX() - 1 - deconvPad.first;
    const int convPadY = common->kernelY() - 1 - deconvPad.second;
    if (convPadX < 0 || convPadY < 0) {
        return NOT_SUPPORT;
    }

    const auto runtime = static_cast<const HexagonRuntime*>(backend()->getRuntime());
    int vtcmSize = runtime->info().vtcmSize;
    if (vtcmSize <= 0) {
        vtcmSize = 4 * 1024 * 1024;
    }
    const int batch = output->length(0);
    const int oc = output->length(1);
    const int oh = output->height();
    const int ow = output->width();
    const int ic = input->length(1);
    const int area = batch * oh * ow;
    const int k = common->kernelY() * common->kernelX() * UP_DIV(ic, 32) * 32;
    const int kAlign = UP_DIV(k, 32) * 32;
    const int totalMp = UP_DIV(area, 32);
    const int totalNp = UP_DIV(oc, 32);
    const int availSize = vtcmSize - 4 * 1024 - 256;
    TileShape tile = chooseIm2ColTileShape(totalMp, totalNp, kAlign, availSize);

    HmxIm2ColConvParam params{};
    setDeconvAsConvIm2ColParameter(mParam, common, input, output, convPadX, convPadY, 64);
    mParam.kernelCountUnit = common->kernelX() * common->kernelY() * UP_DIV(ic, 32);
    mParam.ic = UP_DIV(ic, 32) * 32;
    mParam.icup4 = UP_DIV(ic, 32) * 32;
    ::memcpy(&params.im2col, &mParam, sizeof(mParam));
    params.oc = oc;
    params.mp = tile.mp;
    params.np = tile.np;
    params.relu = common->relu() ? 1 : 0;
    params.relu6 = common->relu6() ? 1 : 0;
    params.batch = batch;
    params.outputBytes = (int32_t)static_cast<HexagonBackend*>(backend())->getSize(output);

    auto inputDev = HexagonBackend::getDevicePtr(input);
    auto outputDev = HexagonBackend::getDevicePtr(output);
    auto weightDev = HexagonBackend::getDevicePtr(mResource->weight);
    std::pair<int, int> biasDev = {-1, 0};
    if (mResource->hasBias) {
        biasDev = HexagonBackend::getDevicePtr(mResource->bias);
    }
    if (inputDev.first <= 0 || outputDev.first <= 0 || weightDev.first <= 0) {
        return NOT_SUPPORT;
    }
    std::vector<std::pair<int, int>> inputFds = {inputDev, weightDev, biasDev};
    std::vector<std::pair<int, int>> outputFds = {outputDev};
    dst.emplace_back();
    dst.back().build(static_cast<HexagonBackend*>(backend()), DSP_OP_IM2COL_CONVOLUTION_FP16,
                     &params, sizeof(params), inputFds, outputFds, inputs, outputs);
    return NO_ERROR;
}

bool HexagonDeconvolution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (dst == nullptr) {
        return true;
    }
    *dst = new HexagonDeconvolution(bn, mResource, op);
    return true;
}

} // namespace MNN
