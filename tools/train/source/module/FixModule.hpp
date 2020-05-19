//
//  FixModule.hpp
//  MNN
//
//  Created by MNN on 2019/12/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef FixModule_hpp
#define FixModule_hpp
#include "Module.hpp"
namespace MNN {
namespace Train {

class FixModule : public Module {
public:
    FixModule(std::vector<Express::VARP> output, std::vector<Express::VARP> parameters,
              std::vector<std::pair<Express::VARP, Express::Dimensionformat>> inputs);
    virtual ~FixModule() = default;
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override;
    virtual void onClearCache() override;
private:
    std::vector<std::pair<Express::VARP, Express::Dimensionformat>> mInputs;
    std::vector<Express::VARP> mOutput;
};
} // namespace Train
} // namespace MNN

#endif
