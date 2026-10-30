//
//  ConvolutionFloatFactory.cpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/compute/ConvolutionFloatFactory.h"
#include "backend/cpu/CPUConvolutionDepthwise.hpp"
#include "backend/cpu/CPURuntime.hpp"
#include "backend/cpu/compute/ConvOpt.h"
#include "backend/cpu/compute/Convolution1x1Strassen.hpp"
#include "backend/cpu/compute/ConvolutionGroup.hpp"
#include "backend/cpu/compute/ConvolutionSingleTap.hpp"
#include "backend/cpu/compute/ConvolutionIntFactory.hpp"

#include "backend/cpu/compute/ConvolutionWinogradBridge.hpp"
#include "backend/cpu/compute/DenseConvolutionTiledExecutor.hpp"
#ifdef MNN_USE_SPARSE_COMPUTE
#include "backend/cpu/compute/SparseConvolutionTiledExecutor.hpp"
#endif
#include "core/Macro.h"
#include "core/OpCommonUtils.hpp"
#include "core/ConvolutionCommon.hpp"
#include "core/Concurrency.h"
#include "core/TensorUtils.hpp"
#include <memory>
#include "backend/cpu/OneDNNConvolution.hpp"
#include "backend/cpu/compute/CPUExtension.hpp"
#include "backend/cpu/compute/ConvInt8TiledExecutor.hpp"

#ifdef MNN_KLEIDIAI_ENABLED
#include "backend/cpu/kleidiai/KleidiAIConvInt8.hpp"
#include "backend/cpu/kleidiai/KleidiAIConvolution.hpp"
#include "backend/cpu/kleidiai/KleidiAIDenseConvolution.hpp"
#endif // MNN_KLEIDIAI_ENABLED

namespace MNN {

#ifdef MNN_KLEIDIAI_ENABLED
#ifdef MNN_LOW_MEMORY
// Try to build a KleidiAI int4 dynamic-quant convolution. Returns nullptr when unsupported.
static Execution* _createKleidiAIConvInt8(Backend* backend, const Op* op,
                                          std::shared_ptr<ConvolutionCommon::Int8Common> weightQuantInfo) {
    if (!weightQuantInfo->canUseInt4) {
        return nullptr;
    }
    auto common = op->main_as_Convolution2D()->common();
    auto core = static_cast<CPUBackend*>(backend)->functions();
    int oc = common->outputCount();
    int ic = common->inputCount();

    int dequantCnt = weightQuantInfo->alphaSize;
    if (weightQuantInfo->asymmetric) {
        dequantCnt /= 2;
    }
    int blockNum = dequantCnt / oc;
    size_t blkSize = blockNum == 1 ? 0 : ic / blockNum;

    auto kernelType = KleidiAIConvInt8::getKernelType(4, weightQuantInfo->asymmetric, blkSize, core->bytes);
    if (!KleidiAIConvInt8::isSupported(kernelType, common)) {
        return nullptr;
    }
    auto execution = new KleidiAIConvInt8(backend, op, weightQuantInfo, true, kernelType, blockNum);
    if (!execution->valid()) {
        delete execution;
        return nullptr;
    }
    return execution;
}
#endif // MNN_LOW_MEMORY

static Execution* _createKleidiAIUnit(const Tensor* input, const Tensor* output, Backend* backend, const Op* op,
                                      const float* originWeight, size_t originWeightSize, const float* bias,
                                      size_t biasSize, std::shared_ptr<ConvolutionCommon::Int8Common> weightQuantInfo,
                                      bool supportSparse, bool lowMemory) {
    auto cpuBackend = (CPUBackend*)backend;
    auto common = op->main_as_Convolution2D()->common();
    bool memoryLow = cpuBackend->memoryMode() == BackendConfig::Memory_Low;

    auto createDense = [&]() -> Execution* {
        if (MNNGetCPUInfo()->sme2 && !weightQuantInfo) {
            return new KleidiAIDenseConvolution(common, backend, originWeight, originWeightSize, bias, biasSize,
                                                weightQuantInfo);
        }
        return nullptr;
    };

#ifdef MNN_LOW_MEMORY
    // Quantized weight is only handled in Memory_Low mode; otherwise it is unsupported here.
    if (lowMemory && nullptr != weightQuantInfo.get() && originWeightSize == 0) {
        return memoryLow ? _createKleidiAIConvInt8(backend, op, weightQuantInfo) : nullptr;
    }
#else
    if (memoryLow) {
        return createDense();
    }
#endif

    bool fastWay = common->kernelY() == 1 && common->kernelX() == 1 && output->width() == input->width() &&
                   output->height() == input->height() && common->strideX() == 1 && common->strideY() == 1;

    // Original impl has a strassen path here (built without MNN_REDUCE_SIZE); KleidiAI need not care about it.
    if (fastWay && cpuBackend->functions()->matmulBytes == 0) {
        bool bFP16 = cpuBackend->functions()->bytes == 2;
        if (KleidiAIConvolution::isSupported(bFP16)) {
            return new KleidiAIConvolution(common, backend, originWeight, originWeightSize, bias, biasSize);
        }
    }

    return createDense();
}
#endif // MNN_KLEIDIAI_ENABLED

static Execution* _createUnit(const Tensor* input, const Tensor* output, Backend* backend, const Op* op,
                              const float* originWeight, size_t originWeightSize, const float* bias, size_t biasSize,
                              std::shared_ptr<ConvolutionCommon::Int8Common> weightQuantInfo, bool supportSparse,
                              bool lowMemory) {
    auto cpuBackend = (CPUBackend*)backend;
    auto conv2d = op->main_as_Convolution2D();
    auto common = conv2d->common();
#ifdef MNN_USE_ONEDNN
    return OneDNN::createConvolution(common, backend, originWeight, originWeightSize, bias, biasSize);
#endif

#ifdef MNN_USE_SPARSE_COMPUTE
    if (conv2d->sparseParameter() && nullptr != weightQuantInfo.get()) {
        if (supportSparse && weightQuantInfo->quan->index() != nullptr) {
            return new SparseConvolutionTiledExecutor(common, backend, weightQuantInfo->quan, conv2d->sparseParameter(),
                                                      bias, biasSize);
        }
    }
#endif

#ifdef MNN_KLEIDIAI_ENABLED
    if (cpuBackend->getRuntime()->hint().enableKleidiAI) {
        auto execution = _createKleidiAIUnit(input, output, backend, op, originWeight, originWeightSize, bias, biasSize,
                                             weightQuantInfo, supportSparse, lowMemory);

        if (execution) {
            return execution;
        }
    }
#endif // MNN_KLEIDIAI_ENABLED

    bool fastWay = common->kernelY() == 1 && common->kernelX() == 1 && output->width() == input->width() &&
                   output->height() == input->height() && common->strideX() == 1 && common->strideY() == 1;
#ifdef MNN_LOW_MEMORY
    if (lowMemory && nullptr != weightQuantInfo.get() && originWeightSize == 0) {
        if (cpuBackend->memoryMode() == BackendConfig::Memory_Low) {
            auto extension = cpuBackend->functions()->extension;
            if (extension != nullptr && extension->createInt8GemmExecution != nullptr) {
                auto execution = extension->createInt8GemmExecution(backend, op, weightQuantInfo, true);
                if (execution != nullptr) {
                    return execution;
                }
            }
            return new DenseConvInt8TiledExecutor(backend, op, weightQuantInfo, true);
        } else {
            return new DenseConvolutionTiledExecutor(common, backend, originWeight, originWeightSize, bias, biasSize,
                                                     weightQuantInfo);
        }
    }
#else
    if (cpuBackend->memoryMode() == BackendConfig::Memory_Low) {
        return new DenseConvolutionTiledExecutor(common, backend, originWeight, originWeightSize, bias, biasSize,
                                                 weightQuantInfo);
    }
#endif

#ifndef MNN_REDUCE_SIZE
    if (fastWay && cpuBackend->functions()->matmulBytes == 0) {
        return new Convolution1x1Strassen(common, backend, originWeight, originWeightSize, bias, biasSize);
    }
#endif

    if (cpuBackend->getRuntime()->hint().winogradMemoryUsed == 0 ||
        (!ConvolutionWinogradBridge::canUseWinograd(common))) {
        return new DenseConvolutionTiledExecutor(common, backend, originWeight, originWeightSize, bias, biasSize,
                                                 nullptr);
    }
    PerfConfig convPerfconfig = DenseConvolutionTiledExecutor::bestTileConvolutionConfig(
        common, input, output, cpuBackend->threadNumber(), backend);
    auto winogradConfig = ConvolutionWinogradBridge::bestWinogradUnit(common, input, output, cpuBackend->threadNumber(),
                                                                      backend, convPerfconfig);
    if (winogradConfig.unit <= 1) {
        return new DenseConvolutionTiledExecutor(common, backend, originWeight, originWeightSize, bias, biasSize,
                                                 nullptr);
    }
    return ConvolutionWinogradBridge::createWinogradImpl(common, input, output, backend, originWeight, originWeightSize,
                                                         bias, biasSize, winogradConfig);
}

// Builds a 1x1 convolution unit from an [outputCount][inputCount] weight
// matrix with common's stride. The returned builder owns the synthesized op
// the unit keeps pointers into, so it must outlive the unit.
static std::pair<std::unique_ptr<flatbuffers::FlatBufferBuilder>, Execution*> _createOneByOne(
    const Convolution2DCommon* common, const Tensor* input, const Tensor* output, Backend* backend, int inputCount,
    int outputCount, const float* weight, const float* bias, int biasSize, bool keepActivation) {
    std::unique_ptr<Convolution2DT> conv(new Convolution2DT);
    conv->common.reset(new Convolution2DCommonT);
    auto convCommon         = conv->common.get();
    convCommon->kernelX     = 1;
    convCommon->kernelY     = 1;
    convCommon->strideX     = common->strideX();
    convCommon->strideY     = common->strideY();
    convCommon->padMode     = PadMode_CAFFE;
    convCommon->outputCount = outputCount;
    convCommon->inputCount  = inputCount;
    convCommon->relu        = keepActivation && common->relu();
    convCommon->relu6       = keepActivation && common->relu6();
    conv->weight.assign(weight, weight + (size_t)outputCount * inputCount);
    conv->bias.assign(bias, bias + biasSize);
    OpT opT;
    opT.type       = OpType_Convolution;
    opT.main.type  = OpParameter_Convolution2D;
    opT.main.value = conv.release();
    std::unique_ptr<flatbuffers::FlatBufferBuilder> builder(new flatbuffers::FlatBufferBuilder);
    builder->Finish(Op::Pack(*builder, &opT));
    auto op      = flatbuffers::GetRoot<Op>(builder->GetBufferPointer());
    auto opW     = op->main_as_Convolution2D()->weight();
    auto opB     = op->main_as_Convolution2D()->bias();
    Execution* unit = _createUnit(input, output, backend, op, opW->data(), opW->size(), opB->data(), opB->size(), nullptr,
                                  false, false);
    return std::make_pair(std::move(builder), unit);
}

// Compute each kernel tap as a 1x1 convolution over the region where its
// input is in bounds, avoiding multiplications by padding zeros. Accumulate
// the results, then apply bias and activation once.
//
// Select between this path and full convolution on each resize.
class ConvolutionTapRegions : public CPUConvolution {
public:
    // Fraction of tap x output products whose input lands inside the image.
    static float liveFraction(const Convolution2DCommon* common, const Tensor* input, const Tensor* output) {
        const auto pad = ConvolutionCommon::convolutionPad(input, output, common);
        auto axis = [](int kernel, int dilate, int stride, int pad, int inSize, int outSize) {
            int live = 0;
            for (int t = 0; t < kernel; ++t) {
                const int offset = t * dilate - pad;
                for (int o = 0; o < outSize; ++o) {
                    const int i = o * stride + offset;
                    live += (i >= 0 && i < inSize) ? 1 : 0;
                }
            }
            return (float)live / (float)(kernel * outSize);
        };
        return axis(common->kernelX(), common->dilateX(), common->strideX(), pad.first, input->width(), output->width()) *
               axis(common->kernelY(), common->dilateY(), common->strideY(), pad.second, input->height(), output->height());
    }
    // Small channel counts cannot amortize the per-tap copies and accumulation.
    static bool supported(const Convolution2DCommon* common, int inputCount, int outputCount) {
        return inputCount >= 64 && outputCount >= 64 && common->strideX() == 1 && common->strideY() == 1 &&
               (common->dilateX() > 1 || common->dilateY() > 1) && common->kernelX() * common->kernelY() > 1;
    }
    static Execution* create(const Op* op, Backend* backend, const Tensor* input, const Tensor* output,
                             int inputCount, int outputCount, const float* weight, const float* bias, int biasSize) {
        const auto common = op->main_as_Convolution2D()->common();
        std::unique_ptr<ConvolutionTapRegions> exe(new ConvolutionTapRegions(common, backend));
        const int kernelSize = common->kernelX() * common->kernelY();
        std::vector<float> tapWeight((size_t)outputCount * inputCount);
        std::vector<float> zeroBias(outputCount, 0.0f);
        for (int t = 0; t < kernelSize; ++t) {
            for (size_t o = 0; o < tapWeight.size(); ++o) {
                tapWeight[o] = weight[o * kernelSize + t];
            }
            auto unit = _createOneByOne(common, input, output, backend, inputCount, outputCount, tapWeight.data(),
                                        zeroBias.data(), outputCount, false);
            std::unique_ptr<Execution> unitOwner(unit.second);
            if (!unitOwner || !unitOwner->valid()) {
                return nullptr;
            }
            Tap tap;
            tap.ky = t / common->kernelX();
            tap.kx = t % common->kernelX();
            tap.op = std::move(unit.first);
            tap.unit = std::move(unitOwner);
            tap.input.reset(new Tensor(4, Tensor::CAFFE_C4));
            tap.output.reset(new Tensor(4, Tensor::CAFFE_C4));
            exe->mTaps.push_back(std::move(tap));
        }
        // Pack the full kernel last so its scratch buffer cannot back a tap's weights.
        exe->mFull.reset(_createUnit(input, output, backend, op, weight, outputCount * inputCount * kernelSize,
                                    bias, biasSize, nullptr, false, false));
        if (!exe->mFull || !exe->mFull->valid()) {
            return nullptr;
        }
        auto core = static_cast<CPUBackend*>(backend)->functions();
        std::vector<float> fullBias(outputCount, 0.0f);
        ::memcpy(fullBias.data(), bias, ALIMIN(biasSize, outputCount) * sizeof(float));
        exe->mBias.resize((size_t)UP_DIV(outputCount, core->pack) * core->pack * core->bytes);
        Resource::copyBias((float*)exe->mBias.data(), fullBias.data(), outputCount, backend);
        return exe.release();
    }

    bool onClone(Backend* backend, const Op* op, Execution** dst) override {
        if (!mFull->onClone(backend, op, nullptr)) {
            return false;
        }
        for (const auto& tap : mTaps) {
            const auto tapOp = flatbuffers::GetRoot<Op>(tap.op->GetBufferPointer());
            if (!tap.unit->onClone(backend, tapOp, nullptr)) {
                return false;
            }
        }
        if (nullptr == dst) {
            return true;
        }
        std::unique_ptr<ConvolutionTapRegions> exe(
            new ConvolutionTapRegions(op->main_as_Convolution2D()->common(), backend));
        Execution* full = nullptr;
        const bool fullCloned = mFull->onClone(backend, op, &full);
        exe->mFull.reset(full);
        if (!fullCloned || !full || !full->valid()) {
            return false;
        }
        for (const auto& tap : mTaps) {
            const auto tapOp = flatbuffers::GetRoot<Op>(tap.op->GetBufferPointer());
            Execution* unit = nullptr;
            const bool unitCloned = tap.unit->onClone(backend, tapOp, &unit);
            std::unique_ptr<Execution> unitOwner(unit);
            if (!unitCloned || !unit || !unit->valid()) {
                return false;
            }
            Tap cloned;
            cloned.ky = tap.ky;
            cloned.kx = tap.kx;
            cloned.op = tap.op;
            cloned.unit = std::move(unitOwner);
            cloned.input.reset(new Tensor(4, Tensor::CAFFE_C4));
            cloned.output.reset(new Tensor(4, Tensor::CAFFE_C4));
            exe->mTaps.push_back(std::move(cloned));
        }
        exe->mBias = mBias;
        *dst = exe.release();
        return true;
    }

    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        mUseRegions = inputs[0]->batch() == outputs[0]->batch() &&
                      liveFraction(mCommon, inputs[0], outputs[0]) < 0.75f;
        if (!mUseRegions) {
            return mFull->onResize(inputs, outputs);
        }
        CPUConvolution::onResize(inputs, outputs);
        // Read the activation here, not in onExecute: mCommon points into the
        // model buffer, which Interpreter::releaseModel frees once sessions
        // exist.
        mPostParameters = getPostParameters();
        auto input  = inputs[0];
        auto output = outputs[0];
        const int ih = input->height(), iw = input->width();
        const int oh = output->height(), ow = output->width();
        mDirect = -1;
        for (int i = 0; i < (int)mTaps.size(); ++i) {
            auto& tap = mTaps[i];
            tap.oy = tap.ky * mCommon->dilateY() - mPadY;
            tap.ox = tap.kx * mCommon->dilateX() - mPadX;
            tap.y0 = ALIMAX(0, -tap.oy);
            tap.y1 = ALIMIN(oh, ih - tap.oy);
            tap.x0 = ALIMAX(0, -tap.ox);
            tap.x1 = ALIMIN(ow, iw - tap.ox);
            tap.active = tap.y0 < tap.y1 && tap.x0 < tap.x1;
            if (mDirect < 0 && tap.oy == 0 && tap.ox == 0 && ih == oh && iw == ow) {
                mDirect = i;
            }
        }
        for (int i = 0; i < (int)mTaps.size(); ++i) {
            auto& tap = mTaps[i];
            if (!tap.active) {
                continue;
            }
            ErrorCode code = NO_ERROR;
            if (i == mDirect) {
                code = tap.unit->onResize(inputs, outputs);
            } else {
                _setShape(tap.input.get(), input->batch(), input->channel(), tap.y1 - tap.y0, tap.x1 - tap.x0);
                _setShape(tap.output.get(), output->batch(), output->channel(), tap.y1 - tap.y0, tap.x1 - tap.x0);
                bool res = backend()->onAcquireBuffer(tap.input.get(), Backend::DYNAMIC);
                res      = res && backend()->onAcquireBuffer(tap.output.get(), Backend::DYNAMIC);
                if (!res) {
                    return OUT_OF_MEMORY;
                }
                code = tap.unit->onResize({tap.input.get()}, {tap.output.get()});
                // Taps run one after another, so their buffers may share memory.
                backend()->onReleaseBuffer(tap.input.get(), Backend::DYNAMIC);
                backend()->onReleaseBuffer(tap.output.get(), Backend::DYNAMIC);
            }
            if (NO_ERROR != code) {
                return code;
            }
        }
        return NO_ERROR;
    }

    ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override {
        if (!mUseRegions) {
            return mFull->onExecute(inputs, outputs);
        }
        auto cpuBackend   = static_cast<CPUBackend*>(backend());
        auto core         = cpuBackend->functions();
        const int pack    = core->pack;
        const int bytes   = core->bytes;
        const int threads = cpuBackend->threadNumber();
        auto input        = inputs[0];
        auto output       = outputs[0];
        const int batch   = input->batch();
        const int ih = input->height(), iw = input->width();
        const int oh = output->height(), ow = output->width();
        const int inPlanes  = UP_DIV(input->channel(), pack) * batch;
        const int outPlanes = UP_DIV(output->channel(), pack) * batch;
        auto src = input->host<uint8_t>();
        auto dst = output->host<uint8_t>();

        if (mDirect >= 0) {
            auto code = mTaps[mDirect].unit->onExecute(inputs, outputs);
            if (NO_ERROR != code) {
                return code;
            }
        } else {
            ::memset(dst, 0, (size_t)outPlanes * oh * ow * pack * bytes);
        }
        for (int i = 0; i < (int)mTaps.size(); ++i) {
            auto& tap = mTaps[i];
            if (!tap.active || i == mDirect) {
                continue;
            }
            const int h = tap.y1 - tap.y0, w = tap.x1 - tap.x0;
            auto tapIn  = tap.input->host<uint8_t>();
            auto tapOut = tap.output->host<uint8_t>();
            MNN_CONCURRENCY_BEGIN(tId, threads) {
                for (int p = (int)tId; p < inPlanes; p += threads) {
                    for (int y = 0; y < h; ++y) {
                        ::memcpy(tapIn + ((size_t)(p * h + y) * w) * pack * bytes,
                                 src + ((size_t)(p * ih + y + tap.y0 + tap.oy) * iw + tap.x0 + tap.ox) * pack * bytes,
                                 (size_t)w * pack * bytes);
                    }
                }
            }
            MNN_CONCURRENCY_END();
            auto code = tap.unit->onExecute({tap.input.get()}, {tap.output.get()});
            if (NO_ERROR != code) {
                return code;
            }
            MNN_CONCURRENCY_BEGIN(tId, threads) {
                for (int p = (int)tId; p < outPlanes; p += threads) {
                    for (int y = 0; y < h; ++y) {
                        auto row = (float*)(dst + ((size_t)(p * oh + y + tap.y0) * ow + tap.x0) * pack * bytes);
                        core->MNNMatrixAdd(row, row, (const float*)(tapOut + ((size_t)(p * h + y) * w) * pack * bytes), w, 0,
                                           0, 0, 1);
                    }
                }
            }
            MNN_CONCURRENCY_END();
        }
        const int outDepth     = UP_DIV(output->channel(), pack);
        const size_t planeSize = (size_t)batch * oh * ow;
        MNN_CONCURRENCY_BEGIN(tId, threads) {
            for (int z = (int)tId; z < outDepth; z += threads) {
                auto plane = (float*)(dst + z * planeSize * pack * bytes);
                core->MNNAxByClampBroadcastUnit(plane, plane, (const float*)(mBias.data() + (size_t)z * pack * bytes),
                                                planeSize, 0, 0, 1, mPostParameters.data());
            }
        }
        MNN_CONCURRENCY_END();
        return NO_ERROR;
    }

private:
    struct Tap {
        int ky = 0, kx = 0;
        // Owns the 1x1 op `unit` points into; declared first, destroyed last.
        std::shared_ptr<flatbuffers::FlatBufferBuilder> op;
        std::shared_ptr<Execution> unit;
        std::shared_ptr<Tensor> input, output;
        int oy = 0, ox = 0, y0 = 0, y1 = 0, x0 = 0, x1 = 0;
        bool active = false;
    };
    ConvolutionTapRegions(const Convolution2DCommon* common, Backend* backend) : CPUConvolution(common, backend) {
    }
    static void _setShape(Tensor* tensor, int batch, int channel, int height, int width) {
        auto& buffer         = tensor->buffer();
        buffer.dimensions    = 4;
        buffer.dim[0].extent = batch;
        buffer.dim[1].extent = channel;
        buffer.dim[2].extent = height;
        buffer.dim[3].extent = width;
        TensorUtils::getDescribe(tensor)->dimensionFormat = MNN_DATA_FORMAT_NC4HW4;
        TensorUtils::setLinearLayout(tensor);
    }
    std::unique_ptr<Execution> mFull;
    bool mUseRegions = false;
    std::vector<Tap> mTaps;
    std::vector<uint8_t> mBias;
    std::vector<float> mPostParameters;
    int mDirect = -1;
};

Execution* ConvolutionFloatFactory::create(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                           const MNN::Op* op, Backend* backend) {
    auto conv2d = op->main_as_Convolution2D();
    if (inputs.size() > 1) {
        // Multi Input
        return new ConvolutionTiledExecutorMultiInput(conv2d->common(), backend);
    }
#ifdef MNN_LOW_MEMORY
    bool lowMemory = static_cast<CPUBackend*>(backend)->memoryMode() == BackendConfig::Memory_Low;
    if (lowMemory && static_cast<CPUBackend*>(backend)->functions()->bytes == 2 &&
        static_cast<CPUBackend*>(backend)->int8Functions()->MNNGemmInt8AddBiasScale_Unit_FP16 == nullptr) {
        // Fall back to fp32
        return nullptr;
    }
#else
    bool lowMemory = false;
#endif

    const float* originWeight = nullptr;
    const float* originBias = nullptr;
    int originWeightSize = 0;
    int originBiasSize = 0;
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
        quanCommon = ConvolutionCommon::load(op, backend, forceFloat, lowMemory);
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
        originWeight = quanCommon->weightFloat.get();
        originWeightSize = quanCommon->weightFloat.size();
    } else if (nullptr == conv2d->weight() || nullptr == conv2d->bias()) {
        MNN_ERROR("%s has no weight or bias. The model may be benchmark model, please revert the weight/bias firstly\n",
                  op->name()->c_str());
        return nullptr;
    }
    auto common = conv2d->common();
    if (nullptr == originWeight && nullptr != op->main_as_Convolution2D()->weight()) {
        originWeight = op->main_as_Convolution2D()->weight()->data();
        originWeightSize = op->main_as_Convolution2D()->weight()->size();
    }
    if (nullptr == originBias && op->main_as_Convolution2D()->bias()) {
        originBias = op->main_as_Convolution2D()->bias()->data();
        originBiasSize = op->main_as_Convolution2D()->bias()->size();
    }

    int group = common->group();
    if (common->inputCount() != inputs[0]->channel() && common->inputCount() > 0) {
        group = inputs[0]->channel() / conv2d->common()->inputCount();
    }
    MNN_ASSERT(group > 0);
    if (group <= 0) {
        MNN_ERROR("Convolution %s has invalid group:%d\n", op->name()->c_str(), group);
        return nullptr;
    }
    if (1 == group) {
        const int kernelSize  = common->kernelX() * common->kernelY();
        const int inputCount  = inputs[0]->channel();
        const int outputCount = common->outputCount();
        const bool plainFloat = nullptr == quanCommon.get() && !lowMemory && nullptr != originWeight && kernelSize > 1 &&
                                originWeightSize == outputCount * inputCount * kernelSize;
        std::shared_ptr<flatbuffers::FlatBufferBuilder> builder;
        std::unique_ptr<Execution> reduced;
        const int ky = common->kernelY() / 2, kx = common->kernelX() / 2;
        if (nullptr == quanCommon.get() && !lowMemory && nullptr != originWeight && kernelSize > 1 &&
            originWeightSize == outputCount * inputCount * kernelSize &&
            common->padMode() == PadMode_SAME && common->kernelX() % 2 == 1 && common->kernelY() % 2 == 1 &&
            (common->dilateX() > 1 || common->dilateY() > 1)) {
            std::unique_ptr<Convolution2DT> reducedConv(new Convolution2DT);
            reducedConv->common.reset(new Convolution2DCommonT);
            auto reducedCommon         = reducedConv->common.get();
            reducedCommon->kernelX     = 1;
            reducedCommon->kernelY     = 1;
            reducedCommon->strideX     = common->strideX();
            reducedCommon->strideY     = common->strideY();
            reducedCommon->padMode     = PadMode_CAFFE;
            reducedCommon->outputCount = outputCount;
            reducedCommon->inputCount  = inputCount;
            reducedCommon->relu        = common->relu();
            reducedCommon->relu6       = common->relu6();
            const int tap = ky * common->kernelX() + kx;
            std::vector<float> tapWeight((size_t)outputCount * inputCount);
            for (size_t o = 0; o < tapWeight.size(); ++o) {
                tapWeight[o] = originWeight[o * kernelSize + tap];
            }
            OpT reducedOpT;
            reducedOpT.type       = OpType_Convolution;
            reducedOpT.main.type  = OpParameter_Convolution2D;
            reducedOpT.main.value = reducedConv.release();
            builder.reset(new flatbuffers::FlatBufferBuilder);
            builder->Finish(Op::Pack(*builder, &reducedOpT));
            auto reducedOp  = flatbuffers::GetRoot<Op>(builder->GetBufferPointer());
            // Factory selection must use the reduced output shape, including during weight rearrangement.
            std::unique_ptr<Tensor> reducedOutput(Tensor::createDevice<float>(outputs[0]->shape(), Tensor::CAFFE_C4));
            reducedOutput->setLength(2, (inputs[0]->height() - 1) / common->strideY() + 1);
            reducedOutput->setLength(3, (inputs[0]->width() - 1) / common->strideX() + 1);
            reduced.reset(_createUnit(inputs[0], reducedOutput.get(), backend, reducedOp,
                                      tapWeight.data(), tapWeight.size(), originBias,
                                      originBiasSize, nullptr, false, false));
            if (!reduced || !reduced->valid()) {
                return reduced.release();
            }
        }
        // Build the reduced kernel first so it cannot retain part of the full kernel's packing scratch buffer.
        Execution* full = nullptr;
        if (plainFloat && core->bytes == 4 && ConvolutionTapRegions::supported(common, inputCount, outputCount)) {
            full = ConvolutionTapRegions::create(op, backend, inputs[0], outputs[0], inputCount, outputCount,
                                                 originWeight, originBias, originBiasSize);
        } else {
            full = _createUnit(inputs[0], outputs[0], backend, op, originWeight, originWeightSize,
                               originBias, originBiasSize, quanCommon, supportSparse, lowMemory);
        }
        if (reduced) {
            return new ConvolutionSingleTap(backend, common, ky, kx, std::shared_ptr<Execution>(full),
                                          std::move(builder), std::move(reduced));
        }
        return full;
    }
    // TODO: Use Geometry to split
    // Split
    std::vector<std::shared_ptr<Execution>> subConvolution;
    auto groupOutputCount = common->outputCount() / group;
    auto groupWeightSize = originWeightSize / group;
    std::shared_ptr<Tensor> emptyInput(Tensor::createDevice<float>(inputs[0]->shape(), Tensor::CAFFE_C4));
    std::shared_ptr<Tensor> emptyOutput(Tensor::createDevice<float>(outputs[0]->shape(), Tensor::CAFFE_C4));
    emptyInput->setLength(1, inputs[0]->channel() / group);
    emptyOutput->setLength(1, outputs[0]->channel() / group);
    for (int i = 0; i < group; ++i) {
        auto newConvolution = _createUnit(
            emptyInput.get(), emptyOutput.get(), backend, op, originWeight + groupWeightSize * i, groupWeightSize,
            conv2d->bias()->data() + groupOutputCount * i, groupOutputCount, quanCommon, supportSparse, lowMemory);
        subConvolution.push_back(std::shared_ptr<Execution>(newConvolution));
    }
    return new ConvolutionGroup(backend, subConvolution);
}
} // namespace MNN
