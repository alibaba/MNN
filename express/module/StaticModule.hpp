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
namespace MNN {
class Session;
class Backend;
namespace Express {
struct NetStorage;
class StaticModule : public Module {
public:
    StaticModule(const void* buffer, size_t length, const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const Module::Config& config);
    virtual ~ StaticModule();
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override;
    void setReusedTensors(std::set<int> reused);

private:
    StaticModule() = default;

    Module* clone(CloneContext* ctx) const override;

    std::vector<std::string> mInputs;
    std::vector<std::string> mOutputs;

    std::shared_ptr<Session> mSession;
    std::vector<Tensor*> mInputTensors;
    std::vector<Tensor*> mOutputTensors;
    bool mShapeFix;
    int mOutputNumbers;

    // First: outputIndex, Second: outputTensor Index
    std::vector<int> mOutputFromTensor;
    // First: outputIndex, Second: input var index
    std::vector<std::pair<int, int>> mOutputFromInput;
    void resizeTensor(Tensor* tensor, const std::vector<int>& dims);
    // the outputs will be used as inputs
    std::set<int> mReusedTensors;
    std::shared_ptr<Backend> mResourceBackend;
    std::shared_ptr<NetStorage> mNetStorage;
};
}
}
#endif
