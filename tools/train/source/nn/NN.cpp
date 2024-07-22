//
//  NN.cpp
//  MNN
//
//  Created by MNN on 2019/11/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <MNN/expr/ExecutorScope.hpp>
#include "NN.hpp"
#include "Distributions.hpp"
#include "module/PipelineModule.hpp"
#include "module/WhileModule.hpp"
#include "module/IfModule.hpp"
#include "module/NMSModule.hpp"
#include "Initializer.hpp"
#include "MNN_generated.h"
#include "RandomGenerator.hpp"
#include "core/Macro.h"
#include "math/WingoradGenerater.hpp"
#include "core/WinogradInt8Attr.hpp"
#include <string>

using namespace MNN::Express;
namespace MNN {
namespace Express {
static VARP _activate(VARP x, NN::ActivationFunctionType type) {
    switch (type) {
        case NN::None:
            return x;
        case NN::Relu:
            return _Relu(x);
        case NN::Relu6:
            return _Relu6(x);
        default:
            break;
    }
    return nullptr;
}

class DropoutModule : public Module {
public:
    DropoutModule(const float dropRatio) {
        mDropRatio = dropRatio;
        setType("Dropout");
    }

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override {
        Express::VARP x = inputs[0];

        if (getIsTraining()) {
            float scale  = 1. / (1. - mDropRatio);
            auto mask    = _Input(x->getInfo()->dim, x->getInfo()->order, x->getInfo()->type);
            auto maskPtr = mask->writeMap<float>();
            auto eltSize = x->getInfo()->size;
            Distributions::uniform(eltSize, 0, 1, maskPtr, RandomGenerator::generator());
            for (int i = 0; i < eltSize; i++) {
                maskPtr[i] = maskPtr[i] < mDropRatio ? 0.0f : scale;
            }
            x = x * mask;
        }

        return {x};
    }

private:
    DropoutModule() = default;

    Module* clone(CloneContext* ctx) const override {
        DropoutModule* module(new DropoutModule);
        module->mDropRatio = mDropRatio;
        return this->cloneBaseTo(ctx, module);
    }

    float mDropRatio;
};

class BatchNormModule : public Module {
public:
    BatchNormModule(EXPRP expr, const float m = 0.99) {
        MNN_ASSERT(expr->get() != nullptr);
        MNN_ASSERT(expr->get()->type() == OpType_BatchNorm);
        auto bnPa = expr->get()->main_as_BatchNorm();
        auto& inputs = expr->inputs();
        int dims = 4;
        if (!inputs.empty()) {
            auto info = inputs[0]->getInfo();
            if (nullptr != info) {
                dims = info->dim.size();
            }
        }
        mEps = bnPa->epsilon();
        mMomentum = m;
        mChannels = bnPa->channels();
        std::vector<int> statShape;
        std::vector<int> reductionDims;
        int channels = mChannels;
        if (dims == 2) {
            statShape      = {1, channels};
            mReductionDims = {0};
        }
        if (dims == 3) {
            statShape      = {1, channels, 1};
            mReductionDims = {0, 2};
        }
        if (dims == 4) {
            statShape      = {1, channels, 1, 1};
            mReductionDims = {0, 2, 3};
        }
        MNN_ASSERT(bnPa->biasData()->size() == mChannels);
        mBias = _TrainableParam(bnPa->biasData()->data(), statShape, NCHW);
        MNN_ASSERT(bnPa->slopeData()->size() == mChannels);
        mScale = _TrainableParam(bnPa->slopeData()->data(), statShape, NCHW);
        MNN_ASSERT(bnPa->meanData()->size() == mChannels);
        mRunningMean = _Const(bnPa->meanData()->data(), statShape, NCHW);
        MNN_ASSERT(bnPa->meanData()->size() == mChannels);
        mRunningVariance = _Const(bnPa->varData()->data(), statShape, NCHW);
        addParameter(mScale);
        addParameter(mBias);
        mRunningVariancePos = addParameter(mRunningVariance);
        mRunningMeanPos = addParameter(mRunningMean);

        setType("BatchNorm");
    }
    BatchNormModule(const int channels, const int dims = 4, const float m = 0.99, const float e = 1e-5) {
        mMomentum = m;
        mEps      = e;
        mChannels = channels;

        std::vector<int> statShape;
        std::vector<int> reductionDims;
        if (dims == 2) {
            statShape      = {1, channels};
            mReductionDims = {0};
        }
        if (dims == 3) {
            statShape      = {1, channels, 1};
            mReductionDims = {0, 2};
        }
        if (dims == 4) {
            statShape      = {1, channels, 1, 1};
            mReductionDims = {0, 2, 3};
        }

        mScale           = _TrainableParam(1.0f, statShape, NCHW);
        mBias            = _TrainableParam(0.0f, statShape, NCHW);
        mRunningMean     = _Const(0.0f, statShape, NCHW);
        mRunningVariance = _Const(0.0f, statShape, NCHW);

        addParameter(mScale);
        addParameter(mBias);
        mRunningVariancePos = addParameter(mRunningVariance);
        mRunningMeanPos = addParameter(mRunningMean);
        setType("BatchNorm");
    }

    VARP runningMean() {
        return mRunningMean;
    }

    VARP runningVariance() {
        return mRunningVariance;
    }

    VARP scale() {
        return mScale;
    }

    VARP bias() {
        return mBias;
    }

    float eps() {
        return mEps;
    }

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override {
        Express::VARP x = inputs[0];
        auto dimFormat = x->getInfo()->order;
        VARP outputData = nullptr;
        if (getIsTraining()) {
            if (dimFormat == NC4HW4 || dimFormat == NHWC) {
                x = _Convert(x, NCHW);
            }
            MNN_ASSERT(x->getInfo()->dim[1] == mChannels);
            auto sampleMean     = _ReduceMean(x, mReductionDims, true); // mean for each channel in the batch
            auto xSub = x - sampleMean;
            auto sampleVar      = _ReduceMean(_Square(xSub), mReductionDims,
                                         true); // variance for each channel in the batch
            auto rSampleStd     = _Reciprocal(_Sqrt(sampleVar + _Const(mEps)));
            auto normalizedData = xSub * rSampleStd;
            outputData          = normalizedData * mScale + mBias;

            mRunningMean = _Const(mMomentum) * mRunningMean + _Const(1 - mMomentum) * sampleMean;
            mRunningVariance = _Const(mMomentum) * mRunningVariance + _Const(1 - mMomentum) * sampleVar;
            outputData->setName(name());
            outputData = _Convert(outputData, dimFormat);
            setParameter(mRunningMean, mRunningMeanPos);
            setParameter(mRunningVariance, mRunningVariancePos);
            return {outputData};
        }
        auto rStd  = _Const(1.0f) / _Sqrt(mRunningVariance + _Const(mEps));
        auto alpha = rStd * mScale;
        auto beta  = mBias - mRunningMean * rStd * mScale;
        //outputData = (_Convert(x, NCHW) * alpha) + beta;
        alpha.fix(VARP::CONSTANT);
        beta.fix(VARP::CONSTANT);
        //FUNC_PRINT_ALL(alpha->readMap<float>()[0], f);
        x = _Convert(x, NC4HW4);
        std::vector<float> scale(alpha->getInfo()->size);
        std::vector<float> bias(beta->getInfo()->size);
        ::memcpy(scale.data(), alpha->readMap<float>(), scale.size() * sizeof(float));
        ::memcpy(bias.data(), beta->readMap<float>(), bias.size() * sizeof(float));
        outputData = _Scale(x, mChannels, std::move(scale), std::move(bias));
        outputData->setName(name());
        outputData = _Convert(outputData, dimFormat);
        return {outputData};
    }

private:
    BatchNormModule() = default;

    Module* clone(CloneContext* ctx) const override {
        BatchNormModule* module(new BatchNormModule);
        module->mMomentum = mMomentum;
        module->mEps = mEps;
        module->mScale = ctx->getOrClone(mScale);
        module->mBias = ctx->getOrClone(mBias);
        module->mRunningMean = ctx->getOrClone(mRunningMean);
        module->mRunningVariance = ctx->getOrClone(mRunningVariance);
        module->mRunningMeanPos = mRunningMeanPos;
        module->mRunningVariancePos = mRunningVariancePos;
        module->mChannels = mChannels;
        module->mReductionDims = mReductionDims;
        return this->cloneBaseTo(ctx, module);
    }

    float mMomentum       = 0.99;
    float mEps            = 1e-5;
    VARP mScale           = nullptr;
    VARP mBias            = nullptr;
    VARP mRunningMean     = nullptr;
    VARP mRunningVariance = nullptr;
    int mRunningMeanPos = -1;
    int mRunningVariancePos = -1;
    int mChannels;
    std::vector<int> mReductionDims;
};

void NN::ConvOption::reset(int size) {
    stride     = std::vector<int>(size, 1);
    channel    = std::vector<int>(size, 0);
    kernelSize = std::vector<int>(size, 1);
    dilate     = std::vector<int>(size, 1);
    padMode    = VALID;
    pads       = std::vector<int>(size, 0);
    depthwise  = false;
    fusedActivationFunction = None;
}
class ConvModule : public Module {
public:
    ConvModule(const NN::ConvParameters& parameters) {
        mParameter = parameters;
        if (nullptr != mParameter.bias) {
            addParameter(mParameter.bias);
        }
        if (nullptr != mParameter.weight) {
            addParameter(mParameter.weight);
        }
        setName(parameters.name);
        setType("Conv");
    }
    NN::ConvParameters& convParameters() {
        return mParameter;
    }
    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override {
        auto input = inputs[0];
        auto& option = mParameter.option;
        if (getIsTraining()) {
            auto tempOutput = _Conv(mParameter.weight, mParameter.bias, _Convert(input, NC4HW4), option.padMode, option.stride, option.dilate, mParameter.group, mParameter.option.pads);
            tempOutput->setName(name());
            tempOutput = _activate(tempOutput, option.fusedActivationFunction);
            return {tempOutput};
        }
        bool relu = option.fusedActivationFunction == NN::Relu;
        bool relu6 = option.fusedActivationFunction == NN::Relu6;
        std::vector<float> weight;
        std::vector<float> bias;
        {
            auto weightInfo = mParameter.weight->getInfo();
            weight.resize(weightInfo->size);
            ::memcpy(weight.data(), mParameter.weight->readMap<float>(), weight.size() * sizeof(float));
        }
        {
            bias.resize(mParameter.option.channel[1]);
            if (nullptr != mParameter.bias) {
                ::memcpy(bias.data(), mParameter.bias->readMap<float>(), bias.size() * sizeof(float));
            } else {
                ::memset(bias.data(), 0, bias.size() * sizeof(float));
            }
        }
        auto tempOutput = _Conv(std::move(weight), std::move(bias), _Convert(input, NC4HW4), option.channel, option.kernelSize, option.padMode, option.stride, option.dilate, mParameter.group, mParameter.option.pads, relu, relu6);
        tempOutput->setName(name());
        return {tempOutput};
    }

private:
    ConvModule() = default;

    Module* clone(CloneContext* ctx) const override {
        ConvModule* module(new ConvModule);
        module->mParameter = mParameter;
        module->mParameter.weight = ctx->getOrClone(mParameter.weight);
        module->mParameter.bias = ctx->getOrClone(mParameter.bias);
        return this->cloneBaseTo(ctx, module);
    }

    NN::ConvParameters mParameter;
};
static std::tuple<VARP, VARP, int> _initParameters(const NN::ConvOption& option, bool hasBias,
                                                   std::shared_ptr<Initializer> weightInit,
                                                   std::shared_ptr<Initializer> biasInit) {
    std::tuple<VARP, VARP, int> defaultRes;
    if (nullptr == weightInit) {
        weightInit.reset(Initializer::xavier());
    }
    if (nullptr == biasInit) {
        biasInit.reset(Initializer::constValue(0.0f));
    }
    VARP weight;
    int group = 1;
    if (option.depthwise) {
        if (option.channel[1] != option.channel[0]) {
            MNN_ERROR("Can't support not the same channel for convolution depthwise\n");
            return defaultRes;
        }
        weight = weightInit->createConstVar({option.channel[0], 1, option.kernelSize[1], option.kernelSize[0]}, NCHW);
        weight.fix(VARP::TRAINABLE);
        group  = option.channel[0];
    } else {
        weight = weightInit->createConstVar(
            {option.channel[1], option.channel[0], option.kernelSize[1], option.kernelSize[0]}, NCHW);
        weight.fix(VARP::TRAINABLE);
    }
    VARP bias;
    if (hasBias) {
        bias = biasInit->createConstVar({option.channel[1]}, NCHW);
        bias.fix(VARP::TRAINABLE);
    }
    return std::make_tuple(weight, bias, group);
}

Module* NN::ConvTranspose(const ConvOption& option, bool hasBias,
                                          std::shared_ptr<Initializer> weightInit,
                                          std::shared_ptr<Initializer> biasInit) {
    VARP input  = _Input({1, option.channel[0], -1, -1}, NC4HW4);
    auto tuple  = _initParameters(option, hasBias, weightInit, biasInit);
    auto weight = std::get<0>(tuple);
    if (nullptr == weight) {
        return nullptr;
    }
    if (!option.depthwise) {
        weight = _Transpose(weight, {1, 0, 2, 3});
        weight.fix(VARP::TRAINABLE);
    }
    auto bias  = std::get<1>(tuple);
    auto group = std::get<2>(tuple);
    if (nullptr != bias) {
        auto tempOutput = _Deconv(weight, bias, input, option.padMode, option.stride, option.dilate, group);
        tempOutput = _activate(tempOutput, option.fusedActivationFunction);
        return NN::extract({input}, {tempOutput}, true);
    }
    auto tempOutput = _Deconv(weight, nullptr, input, option.padMode, option.stride, option.dilate, group);
    tempOutput = _activate(tempOutput, option.fusedActivationFunction);
    return NN::extract({input}, {tempOutput}, true);
}
Module* NN::Conv(const ConvOption& option, bool hasBias, std::shared_ptr<Initializer> weightInit,
                                 std::shared_ptr<Initializer> biasInit) {
    auto tuple  = _initParameters(option, hasBias, weightInit, biasInit);
    ConvParameters parameters;
    parameters.weight = std::get<0>(tuple);
    if (nullptr == parameters.weight) {
        return nullptr;
    }
    parameters.bias  = std::get<1>(tuple);
    parameters.group = std::get<2>(tuple);
    parameters.option = option;
    return new ConvModule(parameters);
}

Module* NN::Linear(int l, int t, bool hasBias, std::shared_ptr<Initializer> weightInit,
                                   std::shared_ptr<Initializer> biasInit) {
    if (nullptr == weightInit) {
        weightInit.reset(Initializer::xavier());
    }
    if (nullptr == biasInit) {
        biasInit.reset(Initializer::constValue(0.0f));
    }
    auto weight = weightInit->createConstVar({t, l}, NCHW);
    weight.fix(VARP::TRAINABLE);
    // Save lazy mode
    auto lazyEval = ExecutorScope::Current()->lazyEval;
    auto lazyMode = ExecutorScope::Current()->getLazyMode();
    ExecutorScope::Current()->lazyEval = true;
    ExecutorScope::Current()->setLazyComputeMode(Executor::LAZY_FULL);
    auto input  = _Input({l}, NCHW);
    auto output = _MatMul(input, weight, false, true);
    if (!hasBias) {
        return NN::extract({input}, {output}, true);
    }
    auto bias = biasInit->createConstVar({1, t}, NCHW);
    bias.fix(VARP::TRAINABLE);
    output    = _Add(output, bias);
    auto module = NN::extract({input}, {output}, true);
    module->setType("Linear");
    // Revert lazy mode
    ExecutorScope::Current()->lazyEval = lazyEval;
    ExecutorScope::Current()->setLazyComputeMode(lazyMode);

    return module;
}

Module* NN::Dropout(const float dropRatio) {
    return new DropoutModule(dropRatio);
}

Module* NN::BatchNorm(const int channels, const int dims, const float m, const float e) {
    return new BatchNormModule(channels, dims, m, e);
}

NN::ConvParameters NN::Utils::ExtractConvolution(EXPRP source) {
    ConvParameters _default;
    if (source->get() == nullptr) {
        return _default;
    }
    if (source->get()->type() != OpType_Convolution && source->get()->type() != OpType_ConvolutionDepthwise) {
        return _default;
    }
    auto conv2D = source->get()->main_as_Convolution2D();
    NN::ConvOption option;
    option.kernelSize = {conv2D->common()->kernelX(), conv2D->common()->kernelY()};
    option.stride     = {conv2D->common()->strideX(), conv2D->common()->strideY()};
    if (nullptr != conv2D->common()->pads()) {
        option.pads.resize(conv2D->common()->pads()->size());
        for (int i=0; i<option.pads.size(); ++i) {
            option.pads[i] = conv2D->common()->pads()->data()[i];
        }
    } else {
        option.pads       = {conv2D->common()->padX(), conv2D->common()->padY()};
    }
    switch (conv2D->common()->padMode()) {
        case MNN::PadMode_SAME:
            option.padMode = SAME;
            break;
        case MNN::PadMode_VALID:
            option.padMode = VALID;
            break;
        case MNN::PadMode_CAFFE:
            option.padMode = CAFFE;
            break;
        default:
            break;
    }
    option.dilate    = {conv2D->common()->dilateX(), conv2D->common()->dilateY()};
    option.depthwise = source->get()->type() == OpType_ConvolutionDepthwise;
    auto inputCount = conv2D->common()->inputCount();
    if (0 == inputCount) {
        auto inputInfo = source->inputs()[0]->getInfo();
        if (nullptr != inputInfo) {
            if (NHWC == inputInfo->order) {
                inputCount = source->inputs()[0]->getInfo()->dim[3];
            } else {
                inputCount = source->inputs()[0]->getInfo()->dim[1];
            }
        } else {
            if (nullptr == conv2D->weight()) {
                MNN_ERROR("Can't extract convolution\n");
                return _default;
            }
            auto weightCount = conv2D->weight()->size();
            if (option.depthwise) {
                inputCount = conv2D->common()->outputCount();
            } else {
                inputCount = weightCount / conv2D->common()->kernelX() / conv2D->common()->kernelY() / conv2D->common()->outputCount();
            }
        }
    }
    option.channel   = {inputCount, conv2D->common()->outputCount()};
    int group        = 1;
    if (option.depthwise) {
        group = conv2D->common()->outputCount();
    }
    VARP weight;
    auto inputs = source->inputs();
    if (inputs.size() > 1) {
        weight = inputs[1];
    }
    VARP bias;
    if (inputs.size() > 2) {
        bias = inputs[2];
    }
    if (inputs.size() < 2) {
        // Extract Weight And Bias from Conv2D
        if (conv2D->weight() == nullptr || conv2D->bias() == nullptr) {
            return _default;
        }
        bias = _TrainableParam(conv2D->bias()->data(), {option.channel[1]}, NCHW);
        weight = _TrainableParam(conv2D->weight()->data(), {option.channel[1], option.channel[0] / group, option.kernelSize[1], option.kernelSize[0]}, NCHW);
    }
    _default.option = std::move(option);
    _default.weight = std::move(weight);
    _default.bias = std::move(bias);
    _default.group = group;
    if (conv2D->common()->relu()) {
        _default.option.fusedActivationFunction = NN::Relu;
    }
    if (conv2D->common()->relu6()) {
        _default.option.fusedActivationFunction = NN::Relu6;
    }
    _default.name = source->name();
    return _default;
}

Module* NN::Conv(const ConvParameters& parameter) {
    return new ConvModule(parameter);
}

Module* NN::Utils::ExtractNotRunableOp(Express::EXPRP expr, const std::map<std::string, SubGraph>& subgraphs) {
    if (nullptr == expr->get()) {
        return nullptr;
    }
    if (expr->get()->type() == OpType_BatchNorm) {
        return new BatchNormModule(expr);
    }
    if (expr->get()->type() == OpType_Dropout) {
        return new DropoutModule(0.3f);
    }
    return nullptr;
}

class ConvBNReluFusedModule : public Module {
public:
    ConvBNReluFusedModule(std::vector<std::shared_ptr<Module> > modules,
                          NN::FeatureScaleStatMethod featureScaleStatMethod,
                          NN::ScaleUpdateMethod scaleUpdateMethod, const int bits, bool winograd = false) {
        MNN_ASSERT(modules.size() >= 1);
        MNN_ASSERT(modules[0]->type() == "Conv");

        if (modules.size() == 3) {
            MNN_ASSERT(modules[1]->type() == "BatchNorm");
            MNN_ASSERT(modules[2]->type() == "ReLU" || modules[2]->type() == "ReLU6");
        }

        for (int i = 0; i < modules.size(); i++) {
            auto type = modules[i]->type();
            if (type == "Conv") {
                mConvParameter = std::static_pointer_cast<ConvModule>(modules[i])->convParameters();
                mOption = mConvParameter.option;
                mGroup = mConvParameter.group;
                mWeight = mConvParameter.weight;
                mBias = mConvParameter.bias;
                if (nullptr != mWeight) {
                    addParameter(mWeight);
                }
                if (nullptr != mBias) {
                    addParameter(mBias);
                }
                if (winograd && mOption.kernelSize[0] > 1 && mOption.kernelSize[1] > 1
                    && mOption.stride[0] == 1 && mOption.stride[1] == 1
                    && mOption.dilate[0] == 1 && mOption.dilate[1] == 1 && mGroup == 1) {
                    mWinogradAttr.reset(new WinogradInt8Attr);
                    mWinogradTransInputMaxPos = addParameter(mWinogradTransInputMax);
                    mWinogradTransInputMinPos = addParameter(mWinogradTransInputMin);
                    mWinogradTransWeightScalePos = addParameter(mWinogradTransInputMax);
                }
                setName(mConvParameter.name);
                modules[i] = nullptr;
            } else if (type == "BatchNorm") {
                mBatchNorm = modules[i];
                registerModel({mBatchNorm});
            } else if (type == "ReLU") {
                mActivation = NN::Relu;
                modules[i] = nullptr;
            } else if (type == "ReLU6") {
                mActivation = NN::Relu6;
                modules[i] = nullptr;
            } else {
                MNN_ASSERT(false);
            }
        }

        if (mOption.fusedActivationFunction == NN::Relu || mOption.fusedActivationFunction == NN::Relu6) {
            mActivation = mOption.fusedActivationFunction;
        }

        mFeatureScaleStatMethod = NN::PerTensor;
        mScaleUpdateMethod = scaleUpdateMethod;

        mBits = bits;
        mLimit = (float)(1 << (bits - 1)) - 1.0f;
        mLimitScale = _Scalar<float>(1.0f / mLimit);
        mWeightClampValue = _Scalar<float>(mLimit);
        // mInputClampValue = _Scalar<float>(mLimit);
        // mOutputClampValue = _Scalar<float>(mLimit);

        // lower bits only apply to weights
        mInputClampValue = _Scalar<float>((float)(1 << (8 - 1)) - 1.0f);
        mOutputClampValue = _Scalar<float>((float)(1 << (8 - 1)) - 1.0f);

        mInputMinPos = addParameter(mInputMin);
        mInputMaxPos = addParameter(mInputMax);
        mOutputMinPos = addParameter(mOutputMin);
        mOutputMaxPos = addParameter(mOutputMax);

        setType("ConvBNReluFused");
    }

    std::pair<VARP, VARP> computeScaleAndZeroPoint(VARP min, VARP max, VARP clampVar) {
        MNN_ASSERT((!(min == nullptr)));
        MNN_ASSERT((!(max == nullptr)));

        min = _Minimum(_Scalar<float>(0.0f), min);
        max = _Maximum(_Scalar<float>(0.0f), max);

        auto scale = (max - min) / (_Scalar(2.0f) * clampVar);
        auto zeroPoint = _Round((_Scalar(0.0f) - min) / scale - clampVar);

        return std::make_pair(scale, zeroPoint);
    }

    std::vector<VARP> fakeQuantFeatureWithMinMax(VARP x, VARP useMin, VARP useMax, VARP clampVar, INTS axis = {}) {
        auto originFormat = x->getInfo()->order;
        auto tempX        = x;
        if (originFormat == NC4HW4) {
            tempX = _Convert(tempX, NCHW);
        }
        auto originX = tempX;
        VARP min, max;

        bool keepDims = false;
        if (axis.size() > 0) {
            // PerChannel for winograd
            keepDims = true;
        }

        min = _ReduceMin(tempX, axis, keepDims);
        max = _ReduceMax(tempX, axis, keepDims);

        VARP scale, zeroPoint;
        VARP nudgeMin, nudgeMax;

        if (!(useMin == nullptr)) {
            MNN_ASSERT(!(useMax == nullptr));
            auto scaleAndZeroPoint = computeScaleAndZeroPoint(useMin, useMax, clampVar);
            scale = scaleAndZeroPoint.first;
            zeroPoint = scaleAndZeroPoint.second;
        } else {
            auto scaleAndZeroPoint = computeScaleAndZeroPoint(min, max, clampVar);
            scale = scaleAndZeroPoint.first;
            zeroPoint = scaleAndZeroPoint.second;
        }

        float limit = clampVar->readMap<float>()[0];
        nudgeMin = (_Scalar<float>(-limit) - zeroPoint) * scale;
        nudgeMax = (_Scalar<float>(limit) - zeroPoint) * scale;

        nudgeMin = _Minimum(_Scalar<float>(0.0f), nudgeMin);
        nudgeMax = _Maximum(_Scalar<float>(0.0f), nudgeMax);

        auto quantX = clamp(_Round(tempX / scale + zeroPoint), clampVar);
        tempX = scale * (quantX - zeroPoint);
        // Break the grad by use cast
        tempX = _Cast<float>(tempX);
        // Move grad from tempX to originX
        tempX = _Convert(tempX + _ZeroGrad(originX), originFormat);

        return {tempX, nudgeMin, nudgeMax};
    }

    VARP clamp(VARP x, VARP clampVar) {
        return _Maximum(_Minimum(x, clampVar), _Negative(clampVar));
    }

    VARP updateParameter(VARP originValue, VARP newValue) const {
        if (nullptr == originValue) {
            return newValue;
        }
        auto ptr = originValue->readMap<float>();
        if (ptr[0] == -100.0f) {
            return newValue;
        }
        switch (mScaleUpdateMethod) {
            case NN::MovingAverage:
                return originValue * _Scalar<float>(mMomentum) + newValue * _Scalar<float>(1.0f-mMomentum);
            case NN::Maximum:
                return _Maximum(originValue, newValue);
            default:
                break;
        }
        MNN_ASSERT(false);
        return nullptr;
    }
    bool bestWinogradUnit(const VARP x, int* unitH = nullptr, int* unitW = nullptr) {
        if (x->getInfo() == nullptr) {
            return false;
        }
        int kernelW = mOption.kernelSize[0], kernelH = mOption.kernelSize[1], padW = mOption.pads[0], padH = mOption.pads[1];
        int outH = x->getInfo()->dim[2] + 2 * padH - kernelH + 1, outW = x->getInfo()->dim[3] + 2 * padW - kernelW + 1;
        int inChannel = mOption.channel[0], outChannel = mOption.channel[1];
        int threadNumber = 1, ePack = 12;
        int unit2   = UP_DIV(outH * outW, ePack * threadNumber);
        int maxUnit = (int)::sqrtf((float)unit2);
        const int MAX_UNIT = 4, MIN_UNIT = 2;
        maxUnit = std::max(std::min(maxUnit, MAX_UNIT), MIN_UNIT);

        auto units = std::pair<int, int>({0, 0});
        float maxRate = 2.0f, originCost = outH * outW * inChannel * outChannel * kernelH * kernelW;
        std::set<int> supportSu{4, 6};
        for (int uh = MIN_UNIT; uh <= maxUnit; ++uh) {
            for (int uw = MIN_UNIT; uw <= maxUnit; ++uw) {
                auto alphaH = uh + kernelH - 1, alphaW = uw + kernelW - 1;
                if (supportSu.find(alphaH) == supportSu.end() || supportSu.find(alphaW) == supportSu.end()) {
                    continue;
                }
                float winogradCost =
                    (2 * alphaH * alphaW * inChannel + alphaH * alphaW * inChannel * outChannel + (alphaH * alphaW + uh * alphaW) * outChannel) * (UP_DIV(outW, uw) * UP_DIV(outH, uh));
                float reduceRate = originCost / winogradCost;
                if (reduceRate > maxRate) {
                    maxRate = reduceRate;
                    units = std::pair<int, int>({uh, uw});
                }
            }
        }
        if (units.first == 0 || units.second == 0) {
            return false;
        }
        if (unitH != nullptr && unitW != nullptr) {
            *unitH = units.first;
            *unitW = units.second;
        }
        return true;
    }
    VARP _winogradConv(const VARP x, const VARP weight) {
        auto inDims = x->getInfo()->dim;
        int batch = inDims[0], inH = inDims[2], inW = inDims[3];
        int inChannel = mOption.channel[0], outChannel = mOption.channel[1];
        int kernelW = mOption.kernelSize[0], kernelH = mOption.kernelSize[1], padW = mOption.pads[0], padH = mOption.pads[1];
        int outH = inH + 2 * padH - kernelH + 1, outW = inW + 2 * padW - kernelW + 1;
        int unitH, unitW;
        bestWinogradUnit(x, &unitH, &unitW);
        if (mWinogradAttr->attrs.empty()) {
            mWinogradAttr->add(0, 0, kernelH, kernelW, unitH, unitW);
        }
        if (unitH != mWinogradAttr->attrs[0].unitY || unitW != mWinogradAttr->attrs[0].unitX) {
            MNN_ERROR("Winograd Conv not support variable input shape\n");
            return nullptr;
        }
        int alphaH = unitH + kernelH - 1, alphaW = unitW + kernelW - 1;
        int unitNumH = UP_DIV(outH, unitH), unitNumW = UP_DIV(outW, unitW);
        int needH = unitNumH * unitH + kernelH - 1, needW = unitNumW * unitW + kernelW - 1;
        int paddings[] = {0, 0, 0, 0, padH, needH - inH - padH, padW, needW - inW - padW};
        auto xx = _Pad(x, _Const(paddings, {8}, NCHW, halide_type_of<int32_t>()));
        // [ic * alphaH * alphaW, N * h_unit_num * w_unit_num]
        xx = _Im2Col(xx, {alphaW, alphaH}, {1, 1}, {0, 0}, {unitW, unitH});
        // [N * h_unit_num * w_unit_num, ic, alphaH, alphaW]
        xx = _Transpose(_Reshape(xx, {inChannel, alphaH, alphaW, -1}), {3, 0, 1, 2});
        // Must be the same as ConvInt8Winograd.cpp
        Math::WinogradGenerater genH(unitH, kernelH,  1, true), genW(unitW, kernelW,  1, true);
        auto srcTransH = _Const(genH.B()->host<void>(), {alphaH, alphaH}, NCHW);
        auto srcTransW = _Const(genW.B()->host<void>(), {alphaW, alphaW}, NCHW);
        xx = _MatMul(_MatMul(_Transpose(srcTransH, {1, 0}), xx), srcTransW);
        // [alphaH * alphaW, ic, N * h_unit_num * w_unit_num]
        xx = _Reshape(_Transpose(xx, {2, 3, 1, 0}), {alphaH * alphaW, inChannel, -1});

        auto inputPair = fakeQuantFeatureWithMinMax(xx, nullptr, nullptr, mInputClampValue, {1, 2, 3});
        mWinogradTransInputMin = updateParameter(mWinogradTransInputMin, inputPair[1]);
        mWinogradTransInputMax = updateParameter(mWinogradTransInputMax, inputPair[2]);
        setParameter(mWinogradTransInputMin, mWinogradTransInputMinPos);
        setParameter(mWinogradTransInputMax, mWinogradTransInputMaxPos);

        auto wTransH = _Const(genH.G()->host<void>(), {alphaH, kernelH}, NCHW);
        auto wTransW = _Const(genW.G()->host<void>(), {alphaW, kernelW}, NCHW);
        // [oc, ic, alphaH, alphaW]
        auto ww = _MatMul(_MatMul(wTransH, weight), _Transpose(wTransW, {1, 0}));
        // [alphaH * alphaW, oc, ic]
        ww = _Transpose(_Reshape(ww, {outChannel, inChannel, -1}), {2, 0, 1});
        auto wwInfo = ww->getInfo();

        // simulate weight quant
        auto weightScale = _Maximum(_ReduceMax(_Abs(ww), {2}, true), _Scalar<float>(1E-6)) * _Reciprocal(mWeightClampValue);
//        ww = clamp(_Round(ww * _Reciprocal(weightScale)), mWeightClampValue) * weightScale;
        setParameter(weightScale, mWinogradTransWeightScalePos);

        // [alphaH * alphaW, oc, N * h_unit_num * w_unit_num]
        auto yy = _MatMul(ww, xx);
        // [oc, N * h_unit_num * w_unit_num, alphaH, alphaW]
        yy = _Reshape(_Transpose(yy, {1, 2, 0}), {outChannel, -1, alphaH, alphaW});
        auto dstTransH = _Const(genH.A()->host<void>(), {alphaH, unitH}, NCHW);
        auto dstTransW = _Const(genW.A()->host<void>(), {alphaW, unitW}, NCHW);
        // [oc, N * h_unit_num * w_unit_num, unitH, unitW]
        yy = _MatMul(_MatMul(_Transpose(dstTransH, {1, 0}), yy), dstTransW);
        // [N, oc, h_unit_num * unitH, w_unit_num * unitW]
        yy = _Reshape(_Transpose(_Reshape(yy, {outChannel, batch, unitNumH, unitNumW, unitH, unitW}), {1, 0, 2, 4, 3, 5}), {batch, outChannel, unitNumH * unitH, unitNumW * unitW});
        int sliceStartData[] = {0, 0, 0, 0}, sliceEndData[] = {-1, -1, outH, outW};
        yy = _Slice(yy, _Const(sliceStartData, {4}, NCHW), _Const(sliceEndData, {4}, NCHW));
        // TODO: add operator!= to VARP
        if (!(mBias == nullptr)) {
            yy = yy + _Reshape(mBias, {1, -1, 1, 1});
        }
        return yy;
    }

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override {
        VARP res;
        if (getIsTraining()) {
            auto x = _Convert(inputs[0], NCHW);
            // simulate weight quant
            auto weightScale = _Maximum(_ReduceMax(_Abs(mWeight), {1, 2, 3}, true), _Scalar<float>(1E-6)) * _Reciprocal(mWeightClampValue);
            auto weightTemp = clamp(_Round(mWeight * _Reciprocal(weightScale)), mWeightClampValue) * weightScale;
            weightTemp = weightTemp + _ZeroGrad(mWeight);

            // simulate input quant to get original input scale
            auto inputPair = fakeQuantFeatureWithMinMax(x, nullptr, nullptr, mInputClampValue);
            mInputMin = updateParameter(mInputMin, inputPair[1]);
            mInputMax = updateParameter(mInputMax, inputPair[2]);
            setParameter(mInputMin, mInputMinPos);
            setParameter(mInputMax, mInputMaxPos);

            // simulate output quant to get original output scale
            if (mWinogradAttr != nullptr && bestWinogradUnit(x)) {
                res = _winogradConv(x, weightTemp);
#ifdef MNN_WINOGRAD_DEBUG
                VARP res2 = _Conv(weightTemp, mBias, _Convert(inputPair[0], NC4HW4), mOption.padMode, mOption.stride,
                            mOption.dilate, mGroup, mOption.pads);
                auto diff = res2 - res;
                diff = diff * diff;
                FUNC_PRINT_ALL(_ReduceMax(diff)->readMap<float>()[0], f);
#endif
            } else {
                res = _Conv(weightTemp, mBias, _Convert(inputPair[0], NC4HW4), mOption.padMode, mOption.stride,
                            mOption.dilate, mGroup, mOption.pads);
            }
            res->setName(name());

            if (mBatchNorm) {
                res = mBatchNorm->forward(res);
            }

            res = _activate(res, mActivation);

            auto outputPair = fakeQuantFeatureWithMinMax(res, nullptr, nullptr, mOutputClampValue);
            mOutputMin = updateParameter(mOutputMin, outputPair[1]);
            mOutputMax = updateParameter(mOutputMax, outputPair[2]);
            setParameter(mOutputMin, mOutputMinPos);
            setParameter(mOutputMax, mOutputMaxPos);

            res = outputPair[0];
        } else {
            if (nullptr == mInputMin) {
                // Initial for test
                // simulate weight quant
                auto weightScale = _Maximum(_ReduceMax(_Abs(mWeight), {1, 2, 3}, true), _Scalar<float>(1E-6)) * _Reciprocal(mWeightClampValue);
                auto weightTemp = clamp(_Round(mWeight * _Reciprocal(weightScale)), mWeightClampValue) * weightScale;

                auto x = _Convert(inputs[0], NCHW);

                auto inputPair = fakeQuantFeatureWithMinMax(x, nullptr, nullptr, mInputClampValue);
                mInputMin = updateParameter(mInputMin, inputPair[1]);
                mInputMax = updateParameter(mInputMax, inputPair[2]);
                setParameter(mInputMin, mInputMinPos);
                setParameter(mInputMax, mInputMaxPos);

                VARP simuRes;
                if (mWinogradAttr != nullptr && bestWinogradUnit(x)) {
                    simuRes = _winogradConv(x, weightTemp);
                } else {
                    simuRes = _Conv(weightTemp, mBias, _Convert(inputPair[0], NC4HW4), mOption.padMode, mOption.stride,
                                     mOption.dilate, mGroup, mOption.pads);
                }
                if (mBatchNorm) {
                    simuRes = mBatchNorm->forward(simuRes);
                }
                simuRes = _activate(simuRes, mActivation);

                Variable::prepareCompute({simuRes});

                auto outputPair = fakeQuantFeatureWithMinMax(simuRes, nullptr, nullptr, mOutputClampValue);
                mOutputMin = updateParameter(mOutputMin, outputPair[1]);
                mOutputMax = updateParameter(mOutputMax, outputPair[2]);
                setParameter(mOutputMin, mOutputMinPos);
                setParameter(mOutputMax, mOutputMaxPos);
            }

            // fold bn to conv weights and bias
            VARP fusedWeights = mWeight;
            VARP fusedBias = mBias;
            fusedBias = _Reshape(fusedBias, {static_cast<int>(fusedBias->getInfo()->size), 1, 1, 1});
            if (mBatchNorm) {
                auto bn = std::static_pointer_cast<BatchNormModule>(mBatchNorm);
                auto bnMean = bn->runningMean();
                auto bnVar = bn->runningVariance();
                auto bnScale = bn->scale();
                auto bnBias = bn->bias();
                auto bnEps = bn->eps();
                MNN_ASSERT(bnMean->getInfo()->dim.size() == 4);

                auto rStd  = _Const(1.0f) / _Sqrt(bnVar + _Const(bnEps));
                auto alpha = rStd * bnScale;
                auto beta  = bnBias - bnMean * rStd * bnScale;

                alpha = _Reshape(alpha, {static_cast<int>(alpha->getInfo()->size), 1, 1, 1});
                beta = _Reshape(beta, {static_cast<int>(beta->getInfo()->size), 1, 1, 1});

                fusedWeights = alpha * fusedWeights;
                fusedBias = alpha * fusedBias + beta;
            }

            auto x = _Convert(inputs[0], NC4HW4);

            int8_t inputZeroPoint, outputZeroPoint;
            {
                VARP channelScale, zeroPoint;
                auto scaleAndZeroPoint = computeScaleAndZeroPoint(mInputMin, mInputMax, mInputClampValue);
                mInputScale = scaleAndZeroPoint.first;
                mInputZeroPoint = scaleAndZeroPoint.second;

                // always PerTensor
                channelScale = _Reciprocal(mInputScale);
                zeroPoint = _Cast<int8_t>(mInputZeroPoint);

                inputZeroPoint = zeroPoint->readMap<int8_t>()[0];

                x = _FloatToInt8(x, channelScale, -int8_t(mInputClampValue->readMap<float>()[0]), int8_t(mInputClampValue->readMap<float>()[0]), inputZeroPoint);
            }
            {
                VARP channelScale, zeroPoint;
                auto scaleAndZeroPoint = computeScaleAndZeroPoint(mOutputMin, mOutputMax, mOutputClampValue);
                mOutputScale = scaleAndZeroPoint.first;
                mOutputZeroPoint = scaleAndZeroPoint.second;

                // always PerTensor
                channelScale = mOutputScale;
                zeroPoint = _Cast<int8_t>(mOutputZeroPoint);

                outputZeroPoint = zeroPoint->readMap<int8_t>()[0];
            }

            std::vector<int8_t> weight;
            std::vector<float> bias;
            std::vector<float> weightScaleVector;
            {
                VARP weightScale, quanWeight, convScale;
                // auto newWeight = fusedWeights * mInputScale;
                weightScale = _Maximum(_ReduceMax(_Abs(fusedWeights), {1, 2, 3}, true), _Scalar<float>(1E-6)) * mLimitScale;
                quanWeight  = _Cast<int8_t>(clamp(_Round(fusedWeights * _Reciprocal(weightScale)), mWeightClampValue));
                convScale   = _Reciprocal(mOutputScale) * weightScale * mInputScale;
                Variable::prepareCompute({quanWeight, convScale});

                // // reference for how to get quantized bias
                // auto remains = _ReduceSum(_Cast<int32_t>(mInputZeroPoint) * _Cast<int32_t>(quanWeight), {1, 2, 3}, true);
                // MNN_ASSERT((mOutputZeroPoint->getInfo()->dim.size() == 0) && (mOutputZeroPoint->getInfo()->size == 1)); // only support per-tensor, per-channel is removed.
                // auto outputZeroPointFused = _Cast<int32_t>(_Cast<float>(mOutputZeroPoint) * _Reciprocal(convScale));
                // auto quanBias = _Cast<int32_t>(fusedBias * _Reciprocal(weightScale * mInputScale)) - remains + outputZeroPointFused;

                {
                    auto info = quanWeight->getInfo();
                    weight.resize(info->size);
                    auto ptr = quanWeight->readMap<int8_t>();
                    ::memcpy(weight.data(), ptr, weight.size() * sizeof(int8_t));
                }
                {
                    auto biasinfo = fusedBias->getInfo();
                    bias.resize(biasinfo->size);
                    auto ptr = fusedBias->readMap<float>();
                    ::memcpy(bias.data(), ptr, bias.size() * sizeof(float));

                    auto info = weightScale->getInfo();
                    weightScaleVector.resize(info->size);
                    MNN_ASSERT(weightScaleVector.size() == bias.size());
                    auto ptrScale = weightScale->readMap<float>();
                    ::memcpy(weightScaleVector.data(), ptrScale, weightScaleVector.size() * sizeof(float));
                }
            }
            bool relu = mActivation == NN::None ? false : true;
            res = _Conv(std::move(weight), std::move(bias), std::move(weightScaleVector), _Convert(x, NC4HW4), mOption.channel,
                        mOption.kernelSize, mOption.padMode, mOption.stride, mOption.dilate, mGroup, mOption.pads, relu,
                        mInputScale->readMap<float>()[0], mOutputScale->readMap<float>()[0],
                        inputZeroPoint, outputZeroPoint,
                        -int8_t(mOutputClampValue->readMap<float>()[0]), int8_t(mOutputClampValue->readMap<float>()[0]), mWeightClampValue->readMap<float>()[0], mAccumulateToInt16);
            if (mWinogradAttr != nullptr && !mWinogradAttr->attrs.empty()) {
                auto scaleAndZeroPoint = computeScaleAndZeroPoint(mWinogradTransInputMin, mWinogradTransInputMax, mInputClampValue);
                auto inputScaleVar = scaleAndZeroPoint.first;
                auto inputZeroPointVar = scaleAndZeroPoint.second;
                auto weightScaleVar = parameters()[mWinogradTransWeightScalePos];

                // Winograd Transformed input scale
                auto inputScaleInfo = inputScaleVar->getInfo();
                auto inputScaleData = inputScaleVar->readMap<float>();
                if (inputScaleInfo == nullptr || inputScaleData == nullptr) {
                    MNN_ERROR("Error for WinogradConvModule, trans input scale not ready\n");
                    return {};
                }
                std::vector<float> inputScales(inputScaleData, inputScaleData + inputScaleInfo->size);
                // Winograd Transformed input zero point
                inputZeroPointVar = _Cast<int32_t>(inputZeroPointVar);
                auto inputZeroPointInfo = inputZeroPointVar->getInfo();
                auto inputZeroPointData = inputZeroPointVar->readMap<int32_t>();
                if (inputZeroPointInfo == nullptr || inputZeroPointData == nullptr) {
                    MNN_ERROR("Error for WinogradConvModule, trans input zero point not ready\n");
                    return {};
                }
                std::vector<int32_t> inputZeroPoints(inputZeroPointData, inputZeroPointData + inputZeroPointInfo->size);
                // Winograd Transformed weight scale
                auto weightScaleInfo = weightScaleVar->getInfo();
                auto weightScaleData = weightScaleVar->readMap<float>();
                if (weightScaleInfo == nullptr || weightScaleData == nullptr) {
                    MNN_ERROR("Error for WinogradConvModule, trans input scale not ready\n");
                    return {};
                }
                std::vector<float> weightScales(weightScaleData, weightScaleData + weightScaleInfo->size);

                mWinogradAttr->attrs[0].inputScales = inputScales;
                mWinogradAttr->attrs[0].inputZeroPoints = inputZeroPoints;
                mWinogradAttr->attrs[0].weightScales = weightScales;

                res = mWinogradAttr->turnToWinogradConv(res);
            }
            res->setName(name());

            // always PerTensor
            res  = _Int8ToFloat(res, mOutputScale, outputZeroPoint);
        }

        return {res};
    }

private:
    ConvBNReluFusedModule() = default;

    Module* clone(CloneContext* ctx) const override {
        ConvBNReluFusedModule* module(new ConvBNReluFusedModule);
        module->mConvParameter = mConvParameter;
        module->mConvParameter.weight = ctx->getOrClone(mConvParameter.weight);
        module->mConvParameter.bias = ctx->getOrClone(mConvParameter.bias);
        module->mOption = mOption;
        module->mGroup = mGroup;
        module->mWeight = ctx->getOrClone(mWeight);
        module->mBias = ctx->getOrClone(mBias);
        module->mActivation = mActivation;
        module->mBits = mBits;
        module->mLimit = mLimit;
        module->mLimitScale = ctx->getOrClone(mLimitScale);
        module->mWeightClampValue = ctx->getOrClone(mWeightClampValue);
        module->mInputScale = ctx->getOrClone(mInputScale);
        module->mOutputScale = ctx->getOrClone(mOutputScale);
        module->mInputMin = ctx->getOrClone(mInputMin);
        module->mInputMax = ctx->getOrClone(mInputMax);
        module->mOutputMin = ctx->getOrClone(mOutputMin);
        module->mOutputMax = ctx->getOrClone(mOutputMax);
        module->mInputZeroPoint = ctx->getOrClone(mInputZeroPoint);
        module->mOutputZeroPoint = ctx->getOrClone(mOutputZeroPoint);
        module->mInputMinPos = mInputMinPos;
        module->mInputMaxPos = mInputMaxPos;
        module->mOutputMinPos = mOutputMinPos;
        module->mOutputMaxPos = mOutputMaxPos;
        module->mInputClampValue = ctx->getOrClone(mInputClampValue);
        module->mOutputClampValue = ctx->getOrClone(mOutputClampValue);
        module->mMomentum = mMomentum;
        module->mFeatureScaleStatMethod = mFeatureScaleStatMethod;
        module->mScaleUpdateMethod = mScaleUpdateMethod;
        if (mBatchNorm) {
            module->mBatchNorm.reset(mBatchNorm->clone(ctx));
            module->registerModel({module->mBatchNorm});
        }
        module->mWinogradAttr = mWinogradAttr;
        module->mWinogradTransInputMin = ctx->getOrClone(mWinogradTransInputMin);
        module->mWinogradTransInputMax = ctx->getOrClone(mWinogradTransInputMax);
        module->mWinogradTransInputMinPos = mWinogradTransInputMinPos;
        module->mWinogradTransInputMaxPos = mWinogradTransInputMaxPos;
        module->mWinogradTransWeightScalePos = mWinogradTransWeightScalePos;
        return this->cloneBaseTo(ctx, module);
    }

    NN::ConvParameters mConvParameter;
    NN::ConvOption mOption;
    int mGroup;
    VARP mWeight;
    VARP mBias;
    NN::ActivationFunctionType mActivation = NN::ActivationFunctionType::None;
    std::shared_ptr<Module> mBatchNorm = nullptr;
    int mBits;
    float mLimit;
    VARP mLimitScale;
    Express::VARP mWeightClampValue;
    VARP mInputScale = nullptr;
    VARP mOutputScale = nullptr;
    VARP mInputMin = nullptr;
    VARP mInputMax = nullptr;
    VARP mOutputMin = nullptr;
    VARP mOutputMax = nullptr;
    VARP mInputZeroPoint = nullptr;
    VARP mOutputZeroPoint = nullptr;
    int mInputMinPos = -1;
    int mInputMaxPos = -1;
    int mOutputMinPos = -1;
    int mOutputMaxPos = -1;
    VARP mInputClampValue;
    VARP mOutputClampValue;
    float mMomentum = 0.99f;
    NN::FeatureScaleStatMethod mFeatureScaleStatMethod;
    NN::ScaleUpdateMethod mScaleUpdateMethod;
    bool mAccumulateToInt16 = false;
    std::shared_ptr<WinogradInt8Attr> mWinogradAttr;
    VARP mWinogradTransInputMin = _Const(-100.f);
    VARP mWinogradTransInputMax = _Const(-100.f);
    int mWinogradTransInputMinPos = -1;
    int mWinogradTransInputMaxPos = -1;
    int mWinogradTransWeightScalePos = -1;
};

Module* NN::ConvBNReluFused(std::vector<std::shared_ptr<Module> > modules,
                                            NN::FeatureScaleStatMethod featureScaleStatMethod,
                                            NN::ScaleUpdateMethod scaleUpdateMethod, const int bits, bool winograd) {
    return new ConvBNReluFusedModule(modules, featureScaleStatMethod, scaleUpdateMethod, bits, winograd);
}

Module* NN::ConvInt8(const ConvOption& option, int bits, bool hasBias,
                                     std::shared_ptr<Initializer> weightInit, std::shared_ptr<Initializer> biasInit, NN::FeatureScaleStatMethod featureMethod, NN::ScaleUpdateMethod method) {
    std::shared_ptr<Module> conv(NN::Conv(option));
    return new ConvBNReluFusedModule({conv}, featureMethod, method, bits);
}
Module* NN::ConvInt8(const ConvParameters& para, int bits, NN::FeatureScaleStatMethod featureMethod, NN::ScaleUpdateMethod method) {
    std::shared_ptr<Module> conv(NN::Conv(para));
    return new ConvBNReluFusedModule({conv}, featureMethod, method, bits);
}

bool NN::turnQuantize(Module* module, const int bits, NN::FeatureScaleStatMethod featureScaleStatMethod, NN::ScaleUpdateMethod scaleUpdateMethod, bool winogradOpt) {
    if (nullptr == module || module->type() != PIPELINE_MODULE) {
        MNN_ERROR("Invalide module for quantized\n");
        return false;
    }
    auto pipModule = static_cast<PipelineModule*>(module);
    std::vector<int> needEraseIndices;
    for (int i = 0; i < pipModule->mSubModules.size(); i++) {
        auto& m = pipModule->mSubModules[i];
        auto& theModule = std::get<0>(m);
        auto moduleType = theModule->type();
        //auto& inputIndices = std::get<1>(m);
        auto& outputIndices = std::get<2>(m);

        if (moduleType == "Conv" && i < pipModule->mSubModules.size() - 1) {
            auto& p1 = pipModule->mSubModules[i+1];
            auto p1Module = std::get<0>(p1);
            auto& p1ModuleType = p1Module->type();
            auto& p1InputIndices = std::get<1>(p1);
            auto& p1OutputIndices = std::get<2>(p1);

            auto convOutputCount = pipModule->countOutputReference(outputIndices);
            bool convSingleOutputReference = ((outputIndices.size() == 1) && (convOutputCount[0] == 1));

            // only conv
            if ((!convSingleOutputReference) || (p1ModuleType == "Conv") ||
                    (p1ModuleType != "BatchNorm" && p1ModuleType != "ReLU" && p1ModuleType != "ReLU6")) {
                theModule.reset(NN::ConvBNReluFused({theModule}, featureScaleStatMethod, scaleUpdateMethod, bits, winogradOpt));
                pipModule->registerModel({theModule});
                continue;
            }
            // conv + bn + ?
            if (p1ModuleType == "BatchNorm") {
                bool convBnConnected = ((convSingleOutputReference) && (p1InputIndices.size() == 1) && (p1InputIndices[0] == outputIndices[0]));
                if (!convBnConnected) {
                    theModule.reset(NN::ConvBNReluFused({theModule}, featureScaleStatMethod, scaleUpdateMethod, bits, winogradOpt));
                    pipModule->registerModel({theModule});
                    continue;
                }

                // last conv + bn
                if (i == pipModule->mSubModules.size() - 2) {
                    theModule.reset(NN::ConvBNReluFused({theModule, p1Module}, featureScaleStatMethod, scaleUpdateMethod, bits, winogradOpt));
                    pipModule->registerModel({theModule});
                    outputIndices = p1OutputIndices;
                    needEraseIndices.emplace_back(i + 1);
                    continue;
                }
                // maybe there is a relu or relu6 after conv + bn
                auto& p2 = pipModule->mSubModules[i+2];
                auto& p2Module = std::get<0>(p2);
                auto p2ModuleType = p2Module->type();
                auto& p2InputIndices = std::get<1>(p2);
                auto& p2OutputIndices = std::get<2>(p2);

                auto bnOutputCount = pipModule->countOutputReference(p1OutputIndices);
                bool bnSingleOutputReference = ((p1OutputIndices.size() == 1) && (bnOutputCount[0] == 1));

                // only conv + bn
                if ((!bnSingleOutputReference) || (p2ModuleType != "ReLU" && p2ModuleType != "ReLU6")) {
                    theModule.reset(NN::ConvBNReluFused({theModule, p1Module}, featureScaleStatMethod, scaleUpdateMethod, bits, winogradOpt));
                    pipModule->registerModel({theModule});
                    outputIndices = p1OutputIndices;
                    needEraseIndices.emplace_back(i + 1);
                    continue;
                } else { // conv + bn + relu or conv + bn + relu6
                    bool convBnReluConnected = ((bnSingleOutputReference) && (p2InputIndices.size() == 1) && (p2InputIndices[0] == p1OutputIndices[0]));
                    bool isPrelu = false;
                    if (p2ModuleType == "ReLU") {
                        auto p2Op = ((ExprModule*)p2Module.get())->getExpr()->get();
                        float slope = p2Op->main_as_Relu()->slope();
                        isPrelu = std::abs(slope) > 1e-6;
                    }
                    if (!convBnReluConnected || isPrelu) {
                        theModule.reset(NN::ConvBNReluFused({theModule, p1Module}, featureScaleStatMethod, scaleUpdateMethod, bits, winogradOpt));
                        pipModule->registerModel({theModule});
                        outputIndices = p1OutputIndices;
                        needEraseIndices.emplace_back(i + 1);
                        continue;
                    }

                    theModule.reset(NN::ConvBNReluFused({theModule, p1Module, p2Module}, featureScaleStatMethod, scaleUpdateMethod, bits, winogradOpt));
                    pipModule->registerModel({theModule});
                    outputIndices = p2OutputIndices;
                    needEraseIndices.emplace_back(i + 1);
                    needEraseIndices.emplace_back(i + 2);
                    continue;
                }
            }
            // conv + relu or conv + relu6
            if (p1ModuleType == "ReLU" || p1ModuleType == "ReLU6") {
                bool convReluConnected = ((convSingleOutputReference) && (p1InputIndices.size() == 1) && (p1InputIndices[0] == outputIndices[0]));
                bool isPrelu = false;
                if (p1ModuleType == "ReLU") {
                    auto p1Op = ((ExprModule*)p1Module.get())->getExpr()->get();
                    float slope = p1Op->main_as_Relu()->slope();
                    isPrelu = std::abs(slope) > 1e-6;
                }
                if (!convReluConnected || isPrelu) {
                    theModule.reset(NN::ConvBNReluFused({theModule}, featureScaleStatMethod, scaleUpdateMethod, bits));
                    pipModule->registerModel({theModule});
                    continue;
                }

                theModule.reset(NN::ConvBNReluFused({theModule, p1Module}, featureScaleStatMethod, scaleUpdateMethod, bits));
                pipModule->registerModel({theModule});
                outputIndices = p1OutputIndices;
                needEraseIndices.emplace_back(i + 1);
                continue;
            }
        }

        if (i == pipModule->mSubModules.size() - 1 && moduleType == "Conv") {
            theModule.reset(NN::ConvBNReluFused({theModule}, featureScaleStatMethod, scaleUpdateMethod, bits));
            pipModule->registerModel({theModule});
        }
    }

    // erase useless submodules
    const int eraseSize = needEraseIndices.size();
    int alreadyErasedCount = 0;
    for (int i = 0; i < eraseSize; i++) {
        auto position = needEraseIndices[i] - alreadyErasedCount;
        auto type = std::get<0>(pipModule->mSubModules[position])->type();
        MNN_ASSERT(type == "BatchNorm" || type == "ReLU" || type == "ReLU6");
        pipModule->mSubModules.erase(pipModule->mSubModules.begin() + position);
        alreadyErasedCount++;
    }
    return true;
}

Module* NN::extract(std::vector<Express::VARP> inputs, std::vector<Express::VARP> outputs, bool fortrain, const std::map<std::string, SubGraph>& subGraph) {
    std::function<std::pair<std::vector<int>, std::shared_ptr<Module>>(EXPRP)> transformFunction;
    if (fortrain) {
        transformFunction =
        [&subGraph](EXPRP source) {
            if (source->get() == nullptr) {
                return std::make_pair(std::vector<int>{}, std::shared_ptr<Module>(nullptr));
            }
            std::shared_ptr<Module> m(NN::Utils::ExtractNotRunableOp(source, subGraph));
            if (nullptr != m) {
                m->setName(source->name());
                return std::make_pair(std::vector<int>{}, m);
            }
            auto convExtracted = NN::Utils::ExtractConvolution(source);
            if (convExtracted.weight == nullptr) {
                return std::make_pair(std::vector<int>{}, std::shared_ptr<Module>(nullptr));
            }
            std::shared_ptr<Module> module(NN::Conv(convExtracted));
            module->setName(source->name());
            return std::make_pair(std::vector<int>{0}, module);
        };
    } else {
        transformFunction = [&subGraph](EXPRP source) {
            if (source->get() == nullptr) {
                return std::make_pair(std::vector<int>{}, std::shared_ptr<Module>(nullptr));
            }
            std::shared_ptr<Module> m(NN::Utils::ExtractNotRunableOp(source, subGraph));
            if (nullptr != m) {
                m->setName(source->name());
                return std::make_pair(std::vector<int>{}, m);
            }
            return std::make_pair(std::vector<int>{}, std::shared_ptr<Module>(nullptr));
        };
    }
    return new PipelineModule(inputs, outputs, transformFunction);
}

} // namespace Express
} // namespace MNN
