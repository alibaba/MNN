//
//  MobilenetV1.cpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "MobilenetV1.hpp"
#include "Initializer.hpp"
using namespace MNN::Express;

namespace MNN {
namespace Train {
namespace Model {
class _ConvBlock : public Module {
public:
    _ConvBlock(std::vector<int> inputOutputChannels, int stride);

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::shared_ptr<Module> conv3x3;
    std::shared_ptr<Module> bn1;
    std::shared_ptr<Module> conv1x1;
    std::shared_ptr<Module> bn2;
};

std::shared_ptr<Module> ConvBlock(std::vector<int> inputOutputChannels, int stride) {
    return std::shared_ptr<Module>(new _ConvBlock(inputOutputChannels, stride));
}

_ConvBlock::_ConvBlock(std::vector<int> inputOutputChannels, int stride) {
    int inputChannels = inputOutputChannels[0], outputChannels = inputOutputChannels[1];

    NN::ConvOption convOption;
    convOption.kernelSize = {3, 3};
    convOption.channel    = {inputChannels, inputChannels};
    convOption.padMode    = Express::SAME;
    convOption.stride     = {stride, stride};
    convOption.depthwise  = true;
    conv3x3.reset(NN::Conv(convOption, false, std::shared_ptr<Initializer>(Initializer::MSRA())));

    bn1.reset(NN::BatchNorm(inputChannels));

    convOption.reset();
    convOption.kernelSize = {1, 1};
    convOption.channel    = {inputChannels, outputChannels};
    convOption.padMode    = Express::SAME;
    convOption.stride     = {1, 1};
    convOption.depthwise  = false;
    conv1x1.reset(NN::Conv(convOption, false, std::shared_ptr<Initializer>(Initializer::MSRA())));

    bn2.reset(NN::BatchNorm(outputChannels));

    registerModel({conv3x3, bn1, conv1x1, bn2});
}

std::vector<Express::VARP> _ConvBlock::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];

    x = conv3x3->forward(x);
    x = bn1->forward(x);
    x = _Relu6(x);
    x = conv1x1->forward(x);
    x = bn2->forward(x);
    x = _Relu6(x);

    return {x};
}

MobilenetV1::MobilenetV1(int numClasses, float widthMult, int divisor) {
    NN::ConvOption convOption;
    convOption.kernelSize = {3, 3};
    int outputChannels    = makeDivisible(32 * widthMult, divisor);
    convOption.channel    = {3, outputChannels};
    convOption.padMode    = Express::SAME;
    convOption.stride     = {2, 2};
    conv1.reset(NN::Conv(convOption, false, std::shared_ptr<Initializer>(Initializer::MSRA())));

    bn1.reset(NN::BatchNorm(outputChannels));

    std::vector<std::vector<int> > convSettings;
    convSettings.push_back({64, 1});
    convSettings.push_back({128, 2});
    convSettings.push_back({256, 2});
    convSettings.push_back({512, 6});
    convSettings.push_back({1024, 2});

    int inputChannels = outputChannels;
    for (int i = 0; i < convSettings.size(); i++) {
        auto setting   = convSettings[i];
        outputChannels = setting[0];
        int times      = setting[1];
        outputChannels = makeDivisible(outputChannels * widthMult, divisor);

        for (int j = 0; j < times; j++) {
            int stride = 1;
            if (times > 1 && j == 0) {
                stride = 2;
            }

            convBlocks.emplace_back(ConvBlock({inputChannels, outputChannels}, stride));
            inputChannels = outputChannels;
        }
    }

    dropout.reset(NN::Dropout(0.1));
    fc.reset(NN::Linear(1024, numClasses, true, std::shared_ptr<Initializer>(Initializer::MSRA())));

    registerModel({conv1, bn1, dropout, fc});
    registerModel(convBlocks);
}

std::vector<Express::VARP> MobilenetV1::onForward(const std::vector<Express::VARP> &inputs) {
    using namespace Express;
    VARP x = inputs[0];

    x = conv1->forward(x);
    x = bn1->forward(x);
    x = _Relu6(x);

    for (int i = 0; i < convBlocks.size(); i++) {
        x = convBlocks[i]->forward(x);
    }

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
