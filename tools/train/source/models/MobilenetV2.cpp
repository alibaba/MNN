//
//  MobilenetV2.cpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MobilenetV2.hpp"

namespace MNN {
namespace Train {
namespace Model {

_ConvBnRelu::_ConvBnRelu(std::vector<int> inputOutputChannels, int kernelSize, int stride, bool depthwise) {
    int inputChannels = inputOutputChannels[0], outputChannels = inputOutputChannels[1];

    NN::ConvOption convOption;
    convOption.kernelSize = {kernelSize, kernelSize};
    convOption.channel    = {inputChannels, outputChannels};
    convOption.padMode    = Express::SAME;
    convOption.stride     = {stride, stride};
    convOption.depthwise  = depthwise;
    conv                  = NN::Conv(convOption, false, std::shared_ptr<Initializer>(Initializer::MSRA()));

    bn = NN::BatchNorm(outputChannels);

    registerModel({conv, bn});
}

std::vector<Express::VARP> _ConvBnRelu::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];

    x = conv->forward(x);
    x = bn->forward(x);
    x = _Relu6(x);

    return {x};
}

_BottleNeck::_BottleNeck(std::vector<int> inputOutputChannels, int stride, int expandRatio) {
    int inputChannels = inputOutputChannels[0], outputChannels = inputOutputChannels[1];
    int expandChannels = inputChannels * expandRatio;

    if (stride == 1 && inputChannels == outputChannels) {
        useShortcut = true;
    }

    if (expandRatio != 1) {
        layers.emplace_back(ConvBnRelu({inputChannels, expandChannels}, 1));
    }

    layers.emplace_back(ConvBnRelu({expandChannels, expandChannels}, 3, stride, true));

    NN::ConvOption convOption;
    convOption.kernelSize = {1, 1};
    convOption.channel    = {expandChannels, outputChannels};
    convOption.padMode    = Express::SAME;
    convOption.stride     = {1, 1};
    convOption.depthwise  = false;
    layers.emplace_back(NN::Conv(convOption, false, std::shared_ptr<Initializer>(Initializer::MSRA())));

    layers.emplace_back(NN::BatchNorm(outputChannels));

    registerModel(layers);
}

std::vector<Express::VARP> _BottleNeck::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];

    for (int i = 0; i < layers.size(); i++) {
        x = layers[i]->forward(x);
    }

    if (useShortcut) {
        x = x + inputs[0];
    }

    return {x};
}

MobilenetV2::MobilenetV2(int numClasses, float widthMult, int divisor) {
    int inputChannels = 32;
    int lastChannels  = 1280;

    std::vector<std::vector<int> > invertedResidualSetting;
    invertedResidualSetting.push_back({1, 16, 1, 1});
    invertedResidualSetting.push_back({6, 24, 2, 2});
    invertedResidualSetting.push_back({6, 32, 3, 2});
    invertedResidualSetting.push_back({6, 64, 4, 2});
    invertedResidualSetting.push_back({6, 96, 3, 1});
    invertedResidualSetting.push_back({6, 160, 3, 2});
    invertedResidualSetting.push_back({6, 320, 1, 1});

    inputChannels = makeDivisible(inputChannels * widthMult, divisor);
    lastChannels  = makeDivisible(lastChannels * std::max(1.0f, widthMult), divisor);

    firstConv = ConvBnRelu({3, inputChannels}, 3, 2);

    for (int i = 0; i < invertedResidualSetting.size(); i++) {
        std::vector<int> setting = invertedResidualSetting[i];
        int t                    = setting[0];
        int c                    = setting[1];
        int n                    = setting[2];
        int s                    = setting[3];

        int outputChannels = makeDivisible(c * widthMult, divisor);

        for (int j = 0; j < n; j++) {
            int stride = 1;
            if (j == 0) {
                stride = s;
            }

            bottleNeckBlocks.emplace_back(BottleNeck({inputChannels, outputChannels}, stride, t));
            inputChannels = outputChannels;
        }
    }

    lastConv = ConvBnRelu({inputChannels, lastChannels}, 1);

    dropout = NN::Dropout(0.1);
    fc      = NN::Linear(lastChannels, numClasses, true, std::shared_ptr<Initializer>(Initializer::MSRA()));

    registerModel({firstConv, lastConv, dropout, fc});
    registerModel(bottleNeckBlocks);
}

std::vector<Express::VARP> MobilenetV2::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];

    x = firstConv->forward(x);

    for (int i = 0; i < bottleNeckBlocks.size(); i++) {
        x = bottleNeckBlocks[i]->forward(x);
    }

    x = lastConv->forward(x);

    // global avg pooling
    x = _AvePool(x, {-1, -1});

    x = _Convert(x, NCHW);
    x = _Reshape(x, {0, -1});

    x = dropout->forward(x);
    x = fc->forward(x);

    x = _Softmax(x, 1);
    return {x};
}

} // namespace Model
} // namespace Train
} // namespace MNN
