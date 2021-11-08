//
//  Lenet.hpp
//  MNN
//
//  Created by MNN on 2020/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef LenetModels_hpp
#define LenetModels_hpp

#include <MNN/expr/Module.hpp>

namespace MNN {
namespace Train {
namespace Model {

class MNN_PUBLIC Lenet : public Express::Module {
public:
    Lenet();

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override;

    std::shared_ptr<Express::Module> conv1;
    std::shared_ptr<Express::Module> conv2;
    std::shared_ptr<Express::Module> ip1;
    std::shared_ptr<Express::Module> ip2;
    std::shared_ptr<Express::Module> dropout;
};

} // namespace Model
} // namespace Train
} // namespace MNN

#endif // LenetModels_hpp
