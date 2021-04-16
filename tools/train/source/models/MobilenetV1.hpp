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
#include "MobilenetUtils.hpp"
#include <MNN/expr/Module.hpp>
#include "NN.hpp"

namespace MNN {
namespace Train {
namespace Model {

class MNN_PUBLIC MobilenetV1 : public Express::Module {
public:
    // use tensorflow numClasses = 1001, which label 0 means outlier of the original 1000 classes
    // so you maybe need to add 1 to your true labels, if you are testing with ImageNet dataset
    MobilenetV1(int numClasses = 1001, float widthMult = 1.0f, int divisor = 8);

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    std::shared_ptr<Express::Module> conv1;
    std::shared_ptr<Express::Module> bn1;
    std::vector<std::shared_ptr<Express::Module> > convBlocks;
    std::shared_ptr<Express::Module> dropout;
    std::shared_ptr<Express::Module> fc;
};

} // namespace Model
} // namespace Train
} // namespace MNN

#endif // MobilenetV1_hpp
