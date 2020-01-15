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

using namespace MNN::Express;
namespace MNN {
namespace Train {
class DropoutModule : public Module {
public:
    DropoutModule(const float dropRatio) {
        mDropRatio = dropRatio;
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
    BatchNormModule(const int channels, const int dims = 4, const float m = 0.999, const float e = 1e-5) {
        mMomentum = m;
        mEps      = e;
        mChannels = channels;

        MNN_ASSERT((dims == 2) || (dims == 4));

        std::vector<int> statShape;
        std::vector<int> reductionDims;
        if (dims == 2) {
            statShape      = {channels};
            mReductionDims = {0};
        }
        if (dims == 4) {
            statShape      = {channels, 1, 1};
            mReductionDims = {0, 2, 3};
        }

        mScale           = _Const(1.0f, statShape, NCHW);
        mBias            = _Const(0.0f, statShape, NCHW);
        mRunningMean     = _Const(0.0f, statShape, NCHW);
        mRunningVariance = _Const(0.0f, statShape, NCHW);

        addParameter(mScale);
        addParameter(mBias);
    }

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override {
        Express::VARP x = inputs[0];

        MNN_ASSERT(x->getInfo()->dim[1] == mChannels);

        auto dimFormat = x->getInfo()->order;
        if (dimFormat == NC4HW4 || dimFormat == NHWC) {
            x = _Convert(x, NCHW);
        }

        VARP outputData = nullptr;

        if (getIsTraining()) {
            Variable::prepareCompute({x});
            auto sampleMean     = _ReduceMean(x, mReductionDims, true); // mean for each channel in the batch
            auto sampleVar      = _ReduceMean(_Square(_Subtract(x, sampleMean)), mReductionDims,
                                         true); // variance for each channel in the batch
            auto rSampleStd     = _Const(1.0f) / _Sqrt(sampleVar + _Const(mEps));
            auto normalizedData = _Subtract(x, sampleMean) * rSampleStd;
            outputData          = normalizedData * mScale + mBias;

            mRunningMean = _Const(mMomentum) * mRunningMean + _Const(1 - mMomentum) * sampleMean;
            mRunningMean.fix(Express::VARP::CONST);
            mRunningVariance = _Const(mMomentum) * mRunningVariance + _Const(1 - mMomentum) * sampleVar;
            mRunningVariance.fix(Express::VARP::CONST);
        } else {
            auto rStd  = _Const(1.0f) / _Sqrt(mRunningVariance + _Const(mEps));
            auto alpha = rStd * mScale;
            auto beta  = mBias - mRunningMean * rStd * mScale;
            alpha.fix(VARP::CONST);
            beta.fix(VARP::CONST);
            outputData = x * alpha + beta;
        }

        if (dimFormat != NCHW) {
            outputData = _Convert(outputData, dimFormat);
        }

        return {outputData};
    }

private:
    float mMomentum       = 0.999;
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
}

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
        weight = weightInit->createConstVar({1, option.channel[0], option.kernelSize[1], option.kernelSize[0]}, NCHW);
        group  = option.channel[0];
    } else {
        weight = weightInit->createConstVar(
            {option.channel[1], option.channel[0], option.kernelSize[1], option.kernelSize[0]}, NCHW);
    }
    VARP bias;
    if (hasBias) {
        bias = biasInit->createConstVar({option.channel[1]}, NCHW);
    }
    return std::make_tuple(weight, bias, group);
}

std::shared_ptr<Module> NN::ConvTranspose(const ConvOption& option, bool hasBias,
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
        return std::shared_ptr<Module>(new FixModule({tempOutput}, {weight, bias}, {{input, NC4HW4}}));
    }
    auto tempOutput = _Deconv(weight, nullptr, input, option.padMode, option.stride, option.dilate, group);
    return std::shared_ptr<Module>(new FixModule({tempOutput}, {weight}, {{input, NC4HW4}}));
}

std::shared_ptr<Module> NN::Conv(const ConvOption& option, bool hasBias, std::shared_ptr<Initializer> weightInit,
                                 std::shared_ptr<Initializer> biasInit) {
    VARP input  = _Input({1, option.channel[0], 1, 1}, NC4HW4);
    auto tuple  = _initParameters(option, hasBias, weightInit, biasInit);
    auto weight = std::get<0>(tuple);
    if (nullptr == weight) {
        return nullptr;
    }
    auto bias  = std::get<1>(tuple);
    auto group = std::get<2>(tuple);
    if (nullptr != bias) {
        auto tempOutput = _Conv(weight, bias, input, option.padMode, option.stride, option.dilate, group);
        return std::shared_ptr<Module>(new FixModule({tempOutput}, {weight, bias}, {{input, NC4HW4}}));
    }
    auto tempOutput = _Conv(weight, nullptr, input, option.padMode, option.stride, option.dilate, group);
    return std::shared_ptr<Module>(new FixModule({tempOutput}, {weight}, {{input, NC4HW4}}));
}

std::shared_ptr<Module> NN::Linear(int l, int t, bool hasBias, std::shared_ptr<Initializer> weightInit,
                                   std::shared_ptr<Initializer> biasInit) {
    if (nullptr == weightInit) {
        weightInit.reset(Initializer::xavier());
    }
    if (nullptr == biasInit) {
        biasInit.reset(Initializer::constValue(0.0f));
    }
    auto weight = weightInit->createConstVar({t, l}, NCHW);
    auto input  = _Input({l}, NCHW);
    auto output = _MatMul(input, weight, false, true);
    if (!hasBias) {
        return std::shared_ptr<Module>(new FixModule({output}, {weight}, {{input, NCHW}}));
    }
    auto bias = biasInit->createConstVar({1, t}, NCHW);
    output    = _Add(output, bias);
    return std::shared_ptr<Module>(new FixModule({output}, {weight, bias}, {{input, NCHW}}));
}

std::shared_ptr<Module> NN::Dropout(const float dropRatio) {
    return std::shared_ptr<Module>(new DropoutModule(dropRatio));
}

std::shared_ptr<Module> NN::BatchNorm(const int channels, const int dims, const float m, const float e) {
    return std::shared_ptr<Module>(new BatchNormModule(channels, dims, m, e));
}

class ConvInt8Module : public Module {
public:
    ConvInt8Module(const NN::ConvOption& option, VARP weight, VARP bias, int group, int bits) : mOption(option) {
        MNN_ASSERT(bits <= 8 && bits > 1);
        auto limit = (float)(1 << (bits - 1)) - 1.0f;
        mWeight    = weight;
        mBias      = bias;
        mGroup     = group;
        if (nullptr != mBias) {
            addParameter(mBias);
        }
        mLimitScale = _Scalar<float>(1.0f / limit);
        mClampValue = _Scalar<float>(limit);
        addParameter(mWeight);
    }

    std::pair<VARP, VARP> fakeQuantFeature(VARP x) {
        auto originFormat = x->getInfo()->order;
        auto tempX        = x;
        if (originFormat == NC4HW4) {
            tempX = _Convert(tempX, NCHW);
        }
        auto originX = tempX;
        auto scale   = _Maximum(_ReduceMax(_Abs(tempX)), _Scalar<float>(0.000000001f)) * mLimitScale;
        scale.fix(VARP::CONST);
        tempX = _Round(tempX * _Reciprocal(scale)) * scale;
        tempX = _Convert(tempX + _ZeroGrad(originX), originFormat);
        return std::make_pair(tempX, scale);
    }
    VARP clamp(VARP x) {
        return _Maximum(_Minimum(x, mClampValue), _Negative(mClampValue));
    }
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override {
        VARP res;
        auto x = _Convert(inputs[0], NCHW);
        Variable::prepareCompute({x});
        if (getIsTraining()) {
            auto weightScale = _ReduceMax(_Abs(mWeight), {1, 2, 3}, true) * mLimitScale;
            weightScale.fix(VARP::CONST);
            // FUNC_PRINT_ALL(weightScale->readMap<float>()[0], f);
            auto weightTemp = _Round(mWeight * _Reciprocal(weightScale)) * weightScale;
            weightTemp      = weightTemp + _ZeroGrad(mWeight);
            auto inputPair  = fakeQuantFeature(x);
            res = _Conv(weightTemp, mBias, _Convert(inputPair.first, NC4HW4), mOption.padMode, mOption.stride,
                        mOption.dilate, mGroup);
            Variable::prepareCompute({res});
            auto outputPair = fakeQuantFeature(res);
            res             = outputPair.first;
            mInputScale     = inputPair.second;
            mOutputScale    = outputPair.second;
        } else {
            x = _Round(x * _Reciprocal(mInputScale));
            x = _Cast<int8_t>(clamp(x));
            std::vector<int8_t> weight;
            std::vector<int32_t> bias;
            std::vector<float> scale;
            {
                auto weightScale = _ReduceMax(_Abs(mWeight), {1, 2, 3}, true) * mLimitScale;
                auto quanWeight  = _Cast<int8_t>(_Round(mWeight * _Reciprocal(weightScale)));
                auto quanTemp    = _Cast<int8_t>(_Round(mWeight * _Reciprocal(weightScale)));
                auto convScale   = mInputScale * _Reciprocal(mOutputScale) * weightScale;
                auto quanBias    = _Cast<int32_t>(mBias * _Reciprocal(mInputScale * weightScale));
                Variable::prepareCompute({quanBias, quanWeight, convScale});
                {
                    auto info = quanWeight->getInfo();
                    weight.resize(info->size);
                    auto ptr = quanWeight->readMap<int8_t>();
                    ::memcpy(weight.data(), ptr, weight.size() * sizeof(int8_t));
                }
                {
                    auto info = quanBias->getInfo();
                    bias.resize(info->size);
                    auto ptr = quanBias->readMap<int32_t>();
                    ::memcpy(bias.data(), ptr, bias.size() * sizeof(int32_t));
                }
                {
                    auto info = convScale->getInfo();
                    scale.resize(info->size);
                    auto ptr = convScale->readMap<float>();
                    ::memcpy(scale.data(), ptr, scale.size() * sizeof(float));
                }
            }
            res = _Conv(std::move(weight), std::move(bias), std::move(scale), _Convert(x, NC4HW4), mOption.channel,
                        mOption.kernelSize, mOption.padMode, mOption.stride, mOption.dilate, mGroup, mOption.pads);
            res = _Cast<float>(_Convert(res, NCHW)) * mOutputScale;
            res = _Convert(res, NC4HW4);
        }
        return {res};
    }

private:
    const NN::ConvOption mOption;
    VARP mWeight;
    VARP mBias;
    int mGroup;
    VARP mLimitScale;
    VARP mInputScale;
    VARP mOutputScale;
    VARP mClampValue;
};

std::shared_ptr<Module> NN::ConvInt8(const ConvOption& option, int bits, bool hasBias,
                                     std::shared_ptr<Initializer> weightInit, std::shared_ptr<Initializer> biasInit) {
    auto tuple = _initParameters(option, hasBias, weightInit, biasInit);
    return std::shared_ptr<Module>(
        new ConvInt8Module(option, std::get<0>(tuple), std::get<1>(tuple), std::get<2>(tuple), bits));
}
std::shared_ptr<Module> NN::ConvInt8(const ConvOption& option, VARP weight, VARP bias, int group, int bits) {
    return std::shared_ptr<Module>(new ConvInt8Module(option, weight, bias, group, bits));
}

std::tuple<NN::ConvOption, VARP, VARP, int> NN::Utils::ExtractConvolution(EXPRP source) {
    std::tuple<NN::ConvOption, VARP, VARP, int> _default;
    if (source->get() == nullptr) {
        return _default;
    }
    if (source->get()->type() != OpType_Convolution && source->get()->type() != OpType_ConvolutionDepthwise) {
        return _default;
    }
    auto inputs = source->inputs();
    if (inputs.size() < 2) {
        // TODO Support Extract Single Convolution
        return _default;
    }
    auto conv2D = source->get()->main_as_Convolution2D();
    NN::ConvOption option;
    option.kernelSize = {conv2D->common()->kernelX(), conv2D->common()->kernelY()};
    option.stride     = {conv2D->common()->strideX(), conv2D->common()->strideY()};
    option.pads       = {conv2D->common()->padX(), conv2D->common()->padY()};
    switch (conv2D->common()->padMode()) {
        case MNN::PadMode_SAME:
            option.padMode = SAME;
            break;
        case MNN::PadMode_VALID:
            option.padMode = VALID;
            break;
        default:
            break;
    }
    option.dilate    = {conv2D->common()->dilateX(), conv2D->common()->dilateY()};
    option.depthwise = source->get()->type() == OpType_ConvolutionDepthwise;
    option.channel   = {conv2D->common()->inputCount(), conv2D->common()->outputCount()};
    int group        = 1;
    if (source->get()->type() == OpType_ConvolutionDepthwise) {
        group = conv2D->common()->outputCount();
    }
    VARP weight = inputs[1];
    VARP bias;
    if (inputs.size() > 2) {
        bias = inputs[2];
    }
    return std::make_tuple(option, weight, bias, group);
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
        mLBias.fix(VARP::CONST);
        mHBias.fix(VARP::CONST);

        auto splitWeight = _Split(weight, {oC0, oC1}, 0);
        auto lw          = _Split(splitWeight[0], {iC0, iC1}, 1);
        auto hw          = _Split(splitWeight[1], {iC0, iC1}, 1);
        mLLW             = lw[0];
        mLHW             = lw[1];
        mHLW             = hw[0];
        mHHW             = hw[1];

        mLLW.fix(VARP::CONST);
        mLHW.fix(VARP::CONST);
        mHLW.fix(VARP::CONST);
        mHHW.fix(VARP::CONST);
        mGroup = group;
        addParameter(mLBias);
        addParameter(mHBias);
        addParameter(mLLW);
        addParameter(mLHW);
        addParameter(mHHW);
        addParameter(mHLW);
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
        return {res};
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
std::shared_ptr<Module> NN::Conv(const ConvOption& option, Express::VARP weight, Express::VARP bias, int group) {
    VARP input = _Input({1, option.channel[0], 1, 1}, NC4HW4);
    if (nullptr == weight) {
        return nullptr;
    }
    if (nullptr != bias) {
        auto tempOutput = _Conv(weight, bias, input, option.padMode, option.stride, option.dilate, group);
        return std::shared_ptr<Module>(new FixModule({tempOutput}, {weight, bias}, {{input, NC4HW4}}));
    }
    auto tempOutput = _Conv(weight, nullptr, input, option.padMode, option.stride, option.dilate, group);
    return std::shared_ptr<Module>(new FixModule({tempOutput}, {weight}, {{input, NC4HW4}}));
}

std::shared_ptr<Module> NN::ConvOctave(const ConvOption& option, Express::VARP weight, Express::VARP bias, int group,
                                       float inFactor, float outFactor) {
    return std::shared_ptr<Module>(new ConvOctaveModule(option, weight, bias, group, inFactor, outFactor));
}

} // namespace Train
} // namespace MNN
