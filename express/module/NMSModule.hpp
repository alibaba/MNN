//
//  NMSModule.hpp
//  MNN
//
//  Created by MNN on b'2020/09/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef NMSModule_hpp
#define NMSModule_hpp
#include <MNN/expr/Module.hpp>
#include "core/Schedule.hpp"
namespace MNN {
namespace Express {
class NMSModule : public Module {
public:
    virtual ~ NMSModule() {
        // Do nothing
    }
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override;
    static NMSModule* create(const Op* op);

private:
    NMSModule(){}

    Module* clone(CloneContext* ctx) const override;
};
}
}
#endif
