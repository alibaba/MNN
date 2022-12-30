//
//  PipelineModule.hpp
//  MNN
//
//  Created by MNN on 2020/01/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef PipelineModule_hpp
#define PipelineModule_hpp
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include "core/AutoStorage.h"
#include "utils/InitNet.hpp"

namespace MNN {
struct Net;
}

namespace MNN {
namespace Express {
#define PIPELINE_MODULE "_pipeline_module__"

class ExprModule : public Module {
public:
    ExprModule(EXPRP expr);
    virtual std::vector<VARP> onForward(const std::vector<VARP>& inputs) override;
    const std::vector<int>& inputIndexes() const {
        return mInputIndexes;
    }
    const EXPRP getExpr() {
        return mExpr;
    }

private:
    Module* clone(CloneContext* ctx) const override;
    EXPRP mExpr;
    std::vector<VARP> mInputs;
    std::vector<int> mInputIndexes;
};

class PipelineModule : public Module {
public:
    typedef std::function<std::pair<std::vector<int>, std::shared_ptr<Module>>(Express::EXPRP)> Transformer;
    MNN_PUBLIC static Module* load(const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const uint8_t* buffer, size_t length, std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtMgr, const Module::Config* config = nullptr);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override;
    virtual void onClearCache() override;
    MNN_PUBLIC std::vector<int> countOutputReference(std::vector<int> outputIndices);

    MNN_PUBLIC PipelineModule(std::vector<Express::VARP> inputs, std::vector<Express::VARP> outputs,
                   const Transformer& transformFunction = {});
private:
    static Module* load(const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, std::shared_ptr<BufferStorage> bufferStorage, const std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtMgr, const Module::Config* config, std::map<std::string, SubGraph>& subGraphMap);
    static void _createSubGraph(const MNN::Net* net, std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtMgr, const Module::Config* config, std::map<std::string, SubGraph>& subGraphMap);

    PipelineModule(){}

    Module* clone(CloneContext* ctx) const override;

    std::vector<std::tuple<std::shared_ptr<Module>, std::vector<int>, std::vector<int>>> mSubModules;
    int mStackSize = 0;
    int mInputSize = 0;
    std::vector<int> mOutputIndex;
    friend class NN;
    std::vector<VARP> mInitVars;
    std::shared_ptr<Schedule::ScheduleInfo> mSharedConst;
};
} // namespace Express
} // namespace MNN

#endif
