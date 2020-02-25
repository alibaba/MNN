//
//  MobilenetV2.hpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MobilenetV2_hpp
#define MobilenetV2_hpp

#include <Initializer.hpp>
#include <vector>
#include "MobilenetUtils.hpp"
#include "Module.hpp"
#include "NN.hpp"

namespace MNN {
namespace Train {
namespace Model {

class _ConvBnRelu : public Module {
public:
    _ConvBnRelu(std::vector<int> inputOutputChannels, int kernelSize = 3, int stride = 1, bool depthwise = false);

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::shared_ptr<Module> conv;
    std::shared_ptr<Module> bn;
};

std::shared_ptr<Module> ConvBnRelu(std::vector<int> inputOutputChannels, int kernelSize = 3, int stride = 1,
                                   bool depthwise = false) {
    return std::shared_ptr<Module>(new _ConvBnRelu(inputOutputChannels, kernelSize, stride, depthwise));
}

class _BottleNeck : public Module {
public:
    _BottleNeck(std::vector<int> inputOutputChannels, int stride, int expandRatio);

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::vector<std::shared_ptr<Module> > layers;
    bool useShortcut = false;
};

std::shared_ptr<Module> BottleNeck(std::vector<int> inputOutputChannels, int stride, int expandRatio) {
    return std::shared_ptr<Module>(new _BottleNeck(inputOutputChannels, stride, expandRatio));
}

class MobilenetV2 : public Module {
public:
    MobilenetV2(int numClasses = 1000, float widthMult = 1.0f, int divisor = 8);

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::shared_ptr<Module> firstConv;
    std::vector<std::shared_ptr<Module> > bottleNeckBlocks;
    std::shared_ptr<Module> lastConv;
    std::shared_ptr<Module> dropout;
    std::shared_ptr<Module> fc;
};

} // namespace Model
} // namespace Train
} // namespace MNN

#endif // MobilenetV2_hpp
