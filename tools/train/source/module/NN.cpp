//
//  NN.cpp
//  MNN
//
//  Created by MNN on 2019/11/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "NN.hpp"
#include "Distributions.hpp"
#include "FixModule.hpp"
#include "Initializer.hpp"
#include "MNN_generated.h"
#include "RandomGenerator.hpp"
#include "core/Macro.h"
#include <string>

using namespace MNN::Express;
namespace MNN {
namespace Train {
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
    float mDropRatio;
};

class BatchNormModule : public Module {
public:
    BatchNormModule(EXPRP expr, const float m = 0.99) {
        MNN_ASSERT(expr->get() != nullptr);
        MNN_ASSERT(expr->get()->type() == OpType_BatchNorm);
        auto bnPa = expr->get()->main_as_BatchNorm();
        mEps = bnPa->epsilon();
        mMomentum = m;
        mChannels = bnPa->channels();
        MNN_ASSERT(bnPa->biasData()->size() == mChannels);
        mBias = _TrainableParam(bnPa->biasData()->data(), {1, mChannels, 1, 1}, NCHW);
        MNN_ASSERT(bnPa->slopeData()->size() == mChannels);
        mScale = _TrainableParam(bnPa->slopeData()->data(), {1, mChannels, 1, 1}, NCHW);
        MNN_ASSERT(bnPa->meanData()->size() == mChannels);
        mRunningMean = _Const(bnPa->meanData()->data(), {1, mChannels, 1, 1}, NCHW);
        MNN_ASSERT(bnPa->meanData()->size() == mChannels);
        mRunningVariance = _Const(bnPa->varData()->data(), {1, mChannels, 1, 1}, NCHW);
        addParameter(mScale);
        addParameter(mBias);
        addParameter(mRunningVariance);
        addParameter(mRunningMean);
        mReductionDims = {0, 2, 3};
        setType("BatchNorm");
    }
    BatchNormModule(const int channels, const int dims = 4, const float m = 0.99, const float e = 1e-5) {
        mMomentum = m;
        mEps      = e;
        mChannels = channels;

        MNN_ASSERT((dims == 2) || (dims == 4));

        std::vector<int> statShape;
        std::vector<int> reductionDims;
        if (dims == 2) {
            statShape      = {1, channels};
            mReductionDims = {0};
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
        addParameter(mRunningVariance);
        addParameter(mRunningMean);
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
            Variable::prepareCompute({inputs[0], outputData, mRunningMean, mRunningVariance});
            mRunningMean.fix(Express::VARP::CONSTANT);
            mRunningVariance.fix(Express::VARP::CONSTANT);
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
    float mMomentum       = 0.99;
    float mEps            = 1e-5;
    VARP mScale           = nullptr;
    VARP mBias            = nullptr;
    VARP mRunningMean     = nullptr;
    VARP mRunningVariance = nullptr;
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
    VARP input  = _Input({1, option.channel[0], 1, 1}, NC4HW4);
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
        return new FixModule({tempOutput}, {weight, bias}, {{input, NC4HW4}});
    }
    auto tempOutput = _Deconv(weight, nullptr, input, option.padMode, option.stride, option.dilate, group);
    tempOutput = _activate(tempOutput, option.fusedActivationFunction);
    return new FixModule({tempOutput}, {weight}, {{input, NC4HW4}});
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
    auto input  = _Input({l}, NCHW);
    auto output = _MatMul(input, weight, false, true);
    if (!hasBias) {
        return new FixModule({output}, {weight}, {{input, NCHW}});
    }
    auto bias = biasInit->createConstVar({1, t}, NCHW);
    bias.fix(VARP::TRAINABLE);
    output    = _Add(output, bias);
    auto module = new FixModule({output}, {weight, bias}, {{input, NCHW}});
    module->setType("Linear");
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
    option.channel   = {conv2D->common()->inputCount(), conv2D->common()->outputCount()};
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

static int _clamp(int c, int maxValue, int minValue) {
    if (c > maxValue) {
        return maxValue;
    }
    if (c < minValue) {
        return minValue;
    }
    return c;
}
class ConvOctaveModule : public Module {
public:
    ConvOctaveModule(const NN::ConvOption& option, VARP weight, VARP bias, int group, float inFactor, float outFactor)
        : mOption(option) {
        auto inputCountC4  = UP_DIV(option.channel[0], 4);
        auto outputCountC4 = UP_DIV(option.channel[1], 4);
        MNN_ASSERT(inputCountC4 > 1 && outputCountC4 > 1);
        MNN_ASSERT(nullptr != bias);
        auto iC0 = (int)((float)inputCountC4 * inFactor);
        iC0      = _clamp(iC0, inputCountC4 - 1, 1);

        auto oC0 = (int)((float)outputCountC4 * outFactor);
        oC0      = _clamp(oC0, outputCountC4 - 1, 1);

        iC0         = iC0 * 4;
        auto iC1    = option.channel[0] - iC0;
        oC0         = oC0 * 4;
        auto oC1    = option.channel[1] - oC0;
        mSplitInput = {iC0, iC1};

        MNN_PRINT("Octave: %d, %d -> %d - %d, %d-%d\n", option.channel[0], option.channel[1], iC0, iC1, oC0, oC1);
        auto splitBias = _Split(bias * _Scalar<float>(0.5f), {oC0, oC1}, 0);
        mLBias         = splitBias[0];
        mHBias         = splitBias[1];
        mLBias.fix(VARP::TRAINABLE);
        mHBias.fix(VARP::TRAINABLE);

        auto splitWeight = _Split(weight, {oC0, oC1}, 0);
        auto lw          = _Split(splitWeight[0], {iC0, iC1}, 1);
        auto hw          = _Split(splitWeight[1], {iC0, iC1}, 1);
        mLLW             = lw[0];
        mLHW             = lw[1];
        mHLW             = hw[0];
        mHHW             = hw[1];

        mLLW.fix(VARP::TRAINABLE);
        mLHW.fix(VARP::TRAINABLE);
        mHLW.fix(VARP::TRAINABLE);
        mHHW.fix(VARP::TRAINABLE);
        mGroup = group;
        addParameter(mLBias);
        addParameter(mHBias);
        addParameter(mLLW);
        addParameter(mLHW);
        addParameter(mHHW);
        addParameter(mHLW);
        setType("ConvOctave");
    }
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override {
        auto input      = _Convert(inputs[0], NC4HW4);
        auto inputSplit = _Split(input, mSplitInput, 1);
        auto XL         = inputSplit[0];
        auto XH         = inputSplit[1];
        if (input->getInfo()->dim[3] < 2) {
            auto L2L = _Conv(mLLW, mLBias, XL, mOption.padMode, mOption.stride, mOption.dilate, mGroup);
            auto L2H = _Conv(mHLW, mHBias, XL, mOption.padMode, mOption.stride, mOption.dilate, mGroup);
            auto H2L = _Conv(mLHW, mLBias, XH, mOption.padMode, mOption.stride, mOption.dilate, mGroup);
            auto H2H = _Conv(mHHW, mHBias, XH, mOption.padMode, mOption.stride, mOption.dilate, mGroup);
            auto L   = L2L + H2L;
            auto H   = H2H + L2H;
            return {_Concat({L, H}, 1)};
        }
        XL        = _AvePool(XL, {2, 2}, {2, 2});
        auto info = XL->getInfo();
        auto L2L  = _Conv(mLLW, mLBias, XL, mOption.padMode, mOption.stride, mOption.dilate, mGroup);
        auto L2H  = _Conv(mHLW, mHBias, XL, mOption.padMode, mOption.stride, mOption.dilate, mGroup);
        auto H2L =
            _Conv(mLHW, mLBias, _AvePool(XH, {2, 2}, {2, 2}), mOption.padMode, mOption.stride, mOption.dilate, mGroup);
        auto H2H      = _Conv(mHHW, mHBias, XH, mOption.padMode, mOption.stride, mOption.dilate, mGroup);
        auto L        = L2L + H2L;
        auto H        = H2H;
        auto dstShape = H->getInfo()->dim; // NCHW
        { H = H2H + _Interp({L2H}, 0.0f, 0.0f, dstShape[3], dstShape[2], 1, true); }
        auto res = _Concat({_Interp({L}, 0.0f, 0.0f, dstShape[3], dstShape[2], 1, true), H}, 1);
        info     = res->getInfo();
        MNN_ASSERT(nullptr != info);
        return {_activate(res, mOption.fusedActivationFunction)};
    }

private:
    const NN::ConvOption mOption;
    VARP mLLW;
    VARP mLHW;
    VARP mHLW;
    VARP mHHW;
    VARP mLBias;
    VARP mHBias;

    std::vector<int> mSplitInput;
    int mGroup;
};

Module* NN::Conv(const ConvParameters& parameter) {
    return new ConvModule(parameter);
}

Module* NN::ConvOctave(const ConvParameters& parameters,
                                       float inFactor, float outFactor) {
    auto module = new ConvOctaveModule(parameters.option, parameters.weight, parameters.bias, parameters.group, inFactor, outFactor);
    module->setName(parameters.name);
    return module;
}
Module* NN::Utils::ExtractNotRunableOp(Express::EXPRP expr) {
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
                          NN::ScaleUpdateMethod scaleUpdateMethod, const int bits) {
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

        mFeatureScaleStatMethod = featureScaleStatMethod;
        mScaleUpdateMethod = scaleUpdateMethod;

        auto limit = (float)(1 << (bits - 1)) - 1.0f;
        mLimitScale = _Scalar<float>(1.0f / limit);
        mClampValue = _Scalar<float>(limit);

        setType("ConvBNReluFused");
    }

    std::pair<VARP, VARP> fakeQuantFeature(VARP x, VARP useScale = nullptr) {
        auto originFormat = x->getInfo()->order;
        auto tempX        = x;
        if (originFormat == NC4HW4) {
            tempX = _Convert(tempX, NCHW);
        }
        auto originX = tempX;
        VARP scale;
        if (mFeatureScaleStatMethod == NN::PerTensor) {
            scale = _Maximum(_ReduceMax(_Abs(tempX)), _Scalar<float>(0.0001f)) * mLimitScale;
        } else {
            auto originSize = originX->getInfo()->size;
            auto batch = originX->getInfo()->dim[0];
            auto channel = originX->getInfo()->dim[1];
            if (originSize / batch / channel < 10) {
                // Too small data
                //MNN_PRINT("%d - %d - %d\n", originSize, batch, channel);
                std::vector<int> dims = {1, channel, 1, 1};
                auto dimVar = _Const(dims.data(), {4}, NCHW, halide_type_of<int32_t>());
                auto singleScale = _Maximum(_ReduceMax(_Abs(tempX)), _Scalar<float>(0.0001f)) * mLimitScale;
                scale = _Fill(dimVar, singleScale);
            } else {
                //MNN_PRINT("%d - %d - %d\n", originSize, batch, channel);
                scale = _Maximum(_ReduceMax(_Abs(tempX), {0, 2, 3}, true), _Scalar<float>(0.0001f)) * mLimitScale;
            }
        }
        scale.fix(VARP::CONSTANT);
        if (useScale == nullptr) {
            tempX = _Round(tempX * _Reciprocal(scale)) * scale;
        } else {
            tempX = _Round(tempX * _Reciprocal(useScale)) * useScale;
        }
        tempX = _Convert(tempX + _ZeroGrad(originX), originFormat);
        return std::make_pair(tempX, scale);
    }

    VARP clamp(VARP x) {
        return _Maximum(_Minimum(x, mClampValue), _Negative(mClampValue));
    }

    VARP updateScale(VARP originValue, VARP newValue) const {
        if (nullptr == originValue) {
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

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override {
        VARP res;
        if (getIsTraining()) {
            Variable::prepareCompute({inputs[0]});
            auto x = _Convert(inputs[0], NCHW);
            // simulate weight quant
            auto weightScale = _Maximum(_ReduceMax(_Abs(mWeight), {1, 2, 3}, true), _Scalar<float>(1E-6)) * mLimitScale;
            weightScale.fix(VARP::CONSTANT);
            auto weightTemp = _Round(mWeight * _Reciprocal(weightScale)) * weightScale;
            weightTemp = weightTemp + _ZeroGrad(mWeight);

            // simulate input quant to get original input scale
            auto inputPair  = fakeQuantFeature(x);
            mInputScale = updateScale(mInputScale, inputPair.second);
            mInputScale.fix(VARP::CONSTANT);

            // simulate output quant to get original output scale
            res = _Conv(weightTemp, mBias, _Convert(inputPair.first, NC4HW4), mOption.padMode, mOption.stride,
                        mOption.dilate, mGroup, mOption.pads);
            res->setName(name());
            auto conv = res;

            if (mBatchNorm) {
                res = mBatchNorm->forward(res);
            }

            res = _activate(res, mActivation);

            Variable::prepareCompute({conv, res});
            auto outputPair = fakeQuantFeature(res);
            mOutputScale = updateScale(mOutputScale, outputPair.second);
            mOutputScale.fix(VARP::CONSTANT);
            res = outputPair.first;
        } else {
            if (nullptr == mInputScale) {
                // Initial for test
                // simulate weight quant
                auto weightScale = _Maximum(_ReduceMax(_Abs(mWeight), {1, 2, 3}, true), _Scalar<float>(1E-6)) * mLimitScale;
                weightScale.fix(VARP::CONSTANT);
                auto weightTemp = _Round(mWeight * _Reciprocal(weightScale)) * weightScale;

                auto x = _Convert(inputs[0], NCHW);
                auto inputPair  = fakeQuantFeature(x);
                mInputScale     = inputPair.second;
                inputPair.first.fix(VARP::CONSTANT);

                auto simuRes = _Conv(weightTemp, mBias, _Convert(inputPair.first, NC4HW4), mOption.padMode, mOption.stride,
                                     mOption.dilate, mGroup, mOption.pads);
                if (mBatchNorm) {
                    simuRes = mBatchNorm->forward(simuRes);
                }
                simuRes = _activate(simuRes, mActivation);

                Variable::prepareCompute({simuRes});
                auto outputPair = fakeQuantFeature(simuRes);
                mOutputScale    = outputPair.second;
                outputPair.first.fix(VARP::CONSTANT);
            }

            // fold bn to conv weights and bias
            VARP fusedWeights = mWeight;
            VARP fusedBias = mBias;
            fusedBias = _Reshape(fusedBias, {fusedBias->getInfo()->size, 1, 1, 1});
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

                alpha = _Reshape(alpha, {alpha->getInfo()->size, 1, 1, 1});
                beta = _Reshape(beta, {beta->getInfo()->size, 1, 1, 1});
                alpha.fix(VARP::CONSTANT);
                beta.fix(VARP::CONSTANT);

                fusedWeights = alpha * fusedWeights;
                fusedBias = alpha * fusedBias + beta;
                fusedWeights.fix(VARP::CONSTANT);
                fusedBias.fix(VARP::CONSTANT);
            }

            auto x = _Convert(inputs[0], NC4HW4);
            {
                std::vector<int> dims = {x->getInfo()->dim[1]};
                auto dimVar = _Const(dims.data(), {1}, NCHW, halide_type_of<int32_t>());
                VARP channelScale;
                if (mFeatureScaleStatMethod == NN::PerTensor) {
                    channelScale = _Reciprocal(_Fill(dimVar, mInputScale));
                } else {
                    channelScale = _Reciprocal(mInputScale);
                }
                x = _FloatToInt8(x, channelScale, -127, 127);// TODO add clamp
            }

            std::vector<int8_t> weight;
            std::vector<int32_t> bias;
            std::vector<float> scale;
            {
                VARP weightScale, quanWeight, convScale;
                if (mOption.depthwise) {
                    auto newWeight = fusedWeights * _Reshape(mInputScale, {-1, 1, 1, 1});
                    weightScale = _Maximum(_ReduceMax(_Abs(newWeight), {1, 2, 3}, true), _Scalar<float>(1E-6)) * mLimitScale;
                    quanWeight  = _Cast<int8_t>(_Round(newWeight * _Reciprocal(weightScale)));
                    convScale   = _Reshape(_Reciprocal(mOutputScale), {-1, 1, 1, 1}) * weightScale;
                } else {
                    auto newWeight = fusedWeights * mInputScale;
                    weightScale = _Maximum(_ReduceMax(_Abs(newWeight), {1, 2, 3}, true), _Scalar<float>(1E-6)) * mLimitScale;
                    quanWeight  = _Cast<int8_t>(_Round(newWeight * _Reciprocal(weightScale)));
                    convScale   = _Reshape(_Reciprocal(mOutputScale), {-1, 1, 1, 1}) * weightScale;
                }
                auto quanBias    = _Cast<int32_t>(fusedBias * _Reciprocal(weightScale));
                Variable::prepareCompute({quanBias, quanWeight, convScale});
                {
                    auto info = quanWeight->getInfo();
                    weight.resize(info->size);
                    auto ptr = quanWeight->readMap<int8_t>();
                    ::memcpy(weight.data(), ptr, weight.size() * sizeof(int8_t));
                }
                {
                    auto biasinfo = quanBias->getInfo();
                    bias.resize(biasinfo->size);
                    auto ptr = quanBias->readMap<int32_t>();
                    ::memcpy(bias.data(), ptr, bias.size() * sizeof(int32_t));
                    auto info = convScale->getInfo();
                    scale.resize(info->size);
                    MNN_ASSERT(scale.size() == bias.size());
                    auto ptrScale = convScale->readMap<float>();
                    ::memcpy(scale.data(), ptrScale, scale.size() * sizeof(float));
                }
            }
            bool relu = mActivation == NN::None ? false : true;
            res = _Conv(std::move(weight), std::move(bias), std::move(scale), _Convert(x, NC4HW4), mOption.channel,
                        mOption.kernelSize, mOption.padMode, mOption.stride, mOption.dilate, mGroup, mOption.pads, relu);
            res->setName(name());
            {
                std::vector<int> dims = {res->getInfo()->dim[1]};
                auto dimVar = _Const(dims.data(), {1}, NCHW, halide_type_of<int32_t>());
                VARP channelScale;
                if (mFeatureScaleStatMethod == NN::PerTensor) {
                    channelScale = _Fill(dimVar, mOutputScale);
                } else {
                    channelScale = mOutputScale;
                }
                res  = _Int8ToFloat(res, channelScale);
            }
        }

        return {res};
    }

private:
    NN::ConvParameters mConvParameter;
    NN::ConvOption mOption;
    int mGroup;
    VARP mWeight;
    VARP mBias;
    NN::ActivationFunctionType mActivation = NN::ActivationFunctionType::None;
    std::shared_ptr<Module> mBatchNorm = nullptr;
    VARP mLimitScale;
    VARP mInputScale = nullptr;
    VARP mOutputScale = nullptr;
    VARP mClampValue;
    float mMomentum = 0.99f;
    NN::FeatureScaleStatMethod mFeatureScaleStatMethod;
    NN::ScaleUpdateMethod mScaleUpdateMethod;
};

Module* NN::ConvBNReluFused(std::vector<std::shared_ptr<Module> > modules,
                                            NN::FeatureScaleStatMethod featureScaleStatMethod,
                                            NN::ScaleUpdateMethod scaleUpdateMethod, const int bits) {
    return new ConvBNReluFusedModule(modules, featureScaleStatMethod, scaleUpdateMethod, bits);
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

} // namespace Train
} // namespace MNN
