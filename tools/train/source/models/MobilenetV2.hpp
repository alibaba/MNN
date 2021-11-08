//
//  MobilenetV2.hpp
//  MNN
//
//  Created by MNN on 2020/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MobilenetV2_hpp
#define MobilenetV2_hpp

#include "Initializer.hpp"
#include <vector>
#include "MobilenetUtils.hpp"
#include <MNN/expr/Module.hpp>
#include "NN.hpp"
#include <algorithm>

namespace MNN {
namespace Train {
namespace Model {

class MNN_PUBLIC MobilenetV2 : public Express::Module {
public:
    // use tensorflow numClasses = 1001, which label 0 means outlier of the original 1000 classes
    // so you maybe need to add 1 to your true labels, if you are testing with ImageNet dataset
    MobilenetV2(int numClasses = 1001, float widthMult = 1.0f, int divisor = 8);

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::shared_ptr<Express::Module> firstConv;
    std::vector<std::shared_ptr<Express::Module> > bottleNeckBlocks;
    std::shared_ptr<Express::Module> lastConv;
    std::shared_ptr<Express::Module> dropout;
    std::shared_ptr<Express::Module> fc;
};

} // namespace Model
} // namespace Train
} // namespace MNN

#endif // MobilenetV2_hpp
