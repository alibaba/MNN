//
//  StaticModule.hpp
//  MNN
//
//  Created by MNN on b'2020/09/10'.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef StaticModule_hpp
#define StaticModule_hpp

#include <MNN/expr/Module.hpp>
#include "core/Schedule.hpp"
#include "core/Session.hpp"

namespace MNN {
class Session;
class Backend;
struct BufferStorage;
namespace Express {
class StaticModule : public Module {
public:
    StaticModule(const void* buffer, size_t length, const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtMgr, const Module::Config& config, bool copyOutput, std::shared_ptr<Schedule::ScheduleInfo> sharedConst);
    virtual ~ StaticModule();
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override;
private:
    StaticModule() = default;

    Module* clone(CloneContext* ctx) const override;
    struct Resource {
        std::vector<std::string> mInputs;
        std::vector<std::string> mOutputs;
        int mOutputNumbers;
        // First: outputIndex, Second: outputTensor Index
        std::vector<int> mOutputFromTensor;
        // First: outputIndex, Second: input var index
        std::vector<std::pair<int, int>> mOutputFromInput;
        bool mUseContentInputs = false;
        std::shared_ptr<BufferStorage> mNetStorage;
        ScheduleConfig mConfig;
        std::shared_ptr<Schedule::ScheduleInfo> mSharedConst;
        Session::ModeGroup mModes;
    };
    std::shared_ptr<Session> mSession;
    std::vector<Tensor*> mInputTensors;
    std::vector<Tensor*> mOutputTensors;
    std::shared_ptr<Backend> mResourceBackend, mBackupResourceBackend;
    std::shared_ptr<Resource> mResource;
};
}
}
#endif
