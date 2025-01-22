//
//  StaticModule.hpp
//  MNN
//
//  Created by MNN on 2020/09/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef StaticModule_hpp
#define StaticModule_hpp

#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include "core/Schedule.hpp"
#include "core/Session.hpp"

namespace MNN {
class Session;
class Backend;
struct BufferStorage;
namespace Express {
class StaticModule : public Module {
public:
    StaticModule(std::vector<int> inputs, std::vector<int> outputs, std::vector<std::shared_ptr<BufferStorage>>&& buffer, Schedule::ScheduleInfo&& scheduleInfo, std::shared_ptr<Schedule::ScheduleInfo> sharedConst, Session::ModeGroup&& mode, std::shared_ptr<Executor::RuntimeManager> rt, const Module::Config& config);
    virtual ~ StaticModule();
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override;
    virtual void onClearCache() override;
    virtual int onOptimize(Interpreter::SessionMode stage) override;
    const Session* getSession() const { return mSession.get(); }

private:
    ErrorCode _resize(const std::vector<Express::VARP>& inputs);
    ErrorCode _execute();

    StaticModule() = default;
    void resetInputOutputs();

    Module* clone(CloneContext* ctx) const override;
    struct Resource {
        std::vector<int> mInputs;
        std::vector<int> mOutputs;
        int mOutputNumbers;
        Backend::Info mBnInfo;
        BackendConfig mBnConfig;
        // First: outputIndex, Second: outputTensor Index
        std::vector<int> mOutputFromTensor;
        // First: outputIndex, Second: input var index
        std::vector<std::pair<int, int>> mOutputFromInput;
        bool mUseContentInputs = false;
        std::shared_ptr<Schedule::ScheduleInfo> mSharedConst;
        Session::ModeGroup mModes;
        std::vector<std::shared_ptr<BufferStorage>> mBuffer;
        std::vector<bool> mInputNeedCPU;
    };
    std::shared_ptr<Session> mSession;
    std::vector<Tensor*> mInputTensors;
    std::vector<std::pair<Tensor*, MNNForwardType>> mPrevInputTensor;
    std::vector<Tensor*> mOutputTensors;
    std::shared_ptr<Resource> mResource;
    bool mShapeInferSeperate = false;
    std::vector<MNN::Express::VARP> mOutputVars;
    std::shared_ptr<MNN::Express::Executor::RuntimeManager> mRuntimeManager;
};
}
}
#endif
