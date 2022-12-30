//
//  WhileModule.hpp
//  MNN
//
//  Created by MNN on b'2020/09/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef WhileModule_hpp
#define WhileModule_hpp
#include <MNN/expr/Module.hpp>
#include "core/Schedule.hpp"
namespace MNN {
namespace Express {
class WhileModule : public Module {
public:
    virtual ~ WhileModule() {
        // Do nothing
    }
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override;
    static WhileModule* create(const Op* op, const std::map<std::string, SubGraph>& subGraph);

    struct Info {
        int mCondInputNumber = 0;
        int mBodyInputNumber = 0;
        int mOutputNumber;

        // First mCondInputs' index, Second: inputs's index
        std::vector<std::pair<int, int>> mInputForCond;

        // First mBodyInputs' index, Second: inputs's index
        std::vector<std::pair<int, int>> mInputForBody;
        std::vector<std::pair<int, int>> mOutputFromBody;
        std::vector<std::pair<int, int>> mOutputFromBodyInput;
        std::vector<int> mOutputFromInput;
        std::vector<std::pair<int, int>> mUpdateForCond;
        std::vector<std::pair<int, int>> mUpdateForBody;

        std::vector<std::pair<int, int>> mCondUpdateForCond;
        std::vector<std::pair<int, int>> mCondUpdateForBody;
    };
private:
    WhileModule(){}

    Module* clone(CloneContext* ctx) const override;
    std::shared_ptr<Info> mInfo;


    std::shared_ptr<Module> mCond;
    std::shared_ptr<Module> mBody;
};
}
}
#endif
