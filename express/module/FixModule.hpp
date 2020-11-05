//
//  FixModule.hpp
//  MNN
//
//  Created by MNN on 2019/12/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef FixModule_hpp
#define FixModule_hpp
#include <MNN/expr/Module.hpp>
namespace MNN {
namespace Express {

class FixModule : public Module {
public:
    FixModule(std::vector<Express::VARP> output, std::vector<Express::VARP> parameters,
              std::vector<std::pair<Express::VARP, Express::Dimensionformat>> inputs);
    virtual ~FixModule() = default;
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override;
    virtual void onClearCache() override;
private:
    FixModule() = default;

    Module* clone(CloneContext* ctx) const override;

    std::vector<std::pair<Express::VARP, Express::Dimensionformat>> mInputs;
    std::vector<Express::VARP> mOutput;
};
} // namespace Express
} // namespace MNN

#endif
