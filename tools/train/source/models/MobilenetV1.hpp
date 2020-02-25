//
//  MobilenetV1.hpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MobilenetV1_hpp
#define MobilenetV1_hpp

#include <vector>
#include "Initializer.hpp"
#include "MobilenetUtils.hpp"
#include "Module.hpp"
#include "NN.hpp"

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

class MNN_PUBLIC MobilenetV1 : public Module {
public:
    MobilenetV1(int numClasses = 1000, float widthMult = 1.0f, int divisor = 8);

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::shared_ptr<Module> conv1;
    std::shared_ptr<Module> bn1;
    std::vector<std::shared_ptr<Module> > convBlocks;
    std::shared_ptr<Module> dropout;
    std::shared_ptr<Module> fc;
};

} // namespace Model
} // namespace Train
} // namespace MNN

#endif // MobilenetV1_hpp
