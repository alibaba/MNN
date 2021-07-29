//
//  StaticModule.hpp
//  MNN
//
//  Created by MNN on b'2020/09/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef StaticModule_hpp
#define StaticModule_hpp

#include <set>
#include <MNN/expr/Module.hpp>
#include "core/Schedule.hpp"

namespace MNN {
class Session;
class Backend;
namespace Express {
struct BufferStorage;
class StaticModule : public Module {
public:
    StaticModule(const void* buffer, size_t length, const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const Module::Config& config, bool copyOutput, std::shared_ptr<Schedule::ScheduleInfo> sharedConst);
    virtual ~ StaticModule();
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override;
    void setReusedTensors(std::set<int> reused);

private:
    StaticModule() = default;

    Module* clone(CloneContext* ctx) const override;
    void resizeTensor(Tensor* tensor, const std::vector<int>& dims);

    struct Resource {
        std::vector<std::string> mInputs;
        std::vector<std::string> mOutputs;
        bool mShapeFix;
        int mOutputNumbers;
        // First: outputIndex, Second: outputTensor Index
        std::vector<int> mOutputFromTensor;
        // First: outputIndex, Second: input var index
        std::vector<std::pair<int, int>> mOutputFromInput;
        // the outputs will be used as inputs
        std::set<int> mReusedTensors;
        bool mUseContentInputs = false;
        std::shared_ptr<BufferStorage> mNetStorage;
        bool mCopyOutput = false;
        ScheduleConfig mConfig;
        std::shared_ptr<Schedule::ScheduleInfo> mSharedConst;
    };
    std::shared_ptr<Session> mSession;
    std::vector<Tensor*> mInputTensors;
    std::vector<Tensor*> mOutputTensors;
    std::vector<std::shared_ptr<Tensor>> mOutputTensorsWrap;

    std::shared_ptr<Backend> mResourceBackend, mBackupResourceBackend;
    std::shared_ptr<Resource> mResource;
};
}
}
#endif
