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
#include "RandomGenerator.hpp"

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
        mEps = e;
        mChannels = channels;

        MNN_ASSERT((dims == 2) || (dims == 4));

        std::vector<int> statShape;
        std::vector<int> reductionDims;
        if (dims == 2) {
            statShape = {channels};
            mReductionDims = {0};
        }
        if (dims == 4) {
            statShape = {channels, 1 , 1};
            mReductionDims = {0, 2, 3};
        }

        mScale = _Const(1.0f, statShape, NCHW);
        mBias = _Const(0.0f, statShape, NCHW);
        mRunningMean = _Const(0.0f, statShape, NCHW);
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
            auto sampleMean = _ReduceMean(x, mReductionDims, true); // mean for each channel in the batch
            auto sampleVar = _ReduceMean(_Square(_Subtract(x, sampleMean)), mReductionDims, true); // variance for each channel in the batch
            auto rSampleStd = _Const(1.0f) / _Sqrt(sampleVar + _Const(mEps));
            auto normalizedData = _Subtract(x, sampleMean) * rSampleStd;
            outputData = normalizedData * mScale + mBias;

            mRunningMean = _Const(mMomentum) * mRunningMean + _Const(1 - mMomentum) * sampleMean;
            mRunningMean.fix(Express::VARP::CONST);
            mRunningVariance = _Const(mMomentum) * mRunningVariance + _Const(1 - mMomentum) * sampleVar;
            mRunningVariance.fix(Express::VARP::CONST);
        } else {
            auto rStd = _Const(1.0f) / _Sqrt(mRunningVariance + _Const(mEps));
            auto normalizedData = _Subtract(x, mRunningMean) * rStd;
            outputData = normalizedData * mScale + mBias;
        }

        if (dimFormat != NCHW) {
            outputData = _Convert(outputData, dimFormat);
        }

        return {outputData};
    }

private:
    float mMomentum = 0.999;
    float mEps = 1e-5;
    VARP mScale = nullptr;
    VARP mBias = nullptr;
    VARP mRunningMean = nullptr;
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

} // namespace Train
} // namespace MNN
