//
//  MoEModule.hpp
//  MNN
//
//  Created by MNN on 2025/05/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MoEModule_hpp
#define MoEModule_hpp

#include <MNN/expr/Module.hpp>
#include "core/Schedule.hpp"
namespace MNN {
namespace Express {
class MoEModule : public Module {
public:
    virtual ~MoEModule() {} // Do nothing
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override;
    static MoEModule* create(const Op* op, const std::map<std::string, SubGraph>& subGraph, std::shared_ptr<Executor::RuntimeManager> rtmgr, const Module::Config& config);
private:
    MoEModule(){}
    Module* clone(CloneContext* ctx) const override;
    int mNumExperts = 128, mTopK = 8;
    std::vector<std::shared_ptr<Module>> mExperts;
    std::vector<VARP> mHiddenStatesList;
};
}
}

#endif /* MoEModule_hpp */
