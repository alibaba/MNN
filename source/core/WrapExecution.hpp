//
//  WrapExecution.hpp
//  MNN
//
//  Created by MNN on 2018/09/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef WrapExecution_hpp
#define WrapExecution_hpp

#include <stdio.h>
#include <memory>
#include "core/Backend.hpp"
#include "core/Execution.hpp"
#include "core/Macro.h"
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/Int8FunctionsOpt.h"

namespace MNN {

/** execution wrapper. hiding cross-backend tensor converting. */
class MNN_PUBLIC WrapExecution : public Execution {
public:
    /**
     * @brief initializer.
     * @param CPUBackend    CPU backend.
     * @param execution     execution to be wrapped.
     */
    WrapExecution(Backend *CPUBackend, std::shared_ptr<Execution> execution, bool isStatic = true);
    /**
     * @brief deinitializer.
     */
    virtual ~WrapExecution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

    static bool needWrap(const Tensor* input, Backend* current);
    static Tensor* copyConstCache(Tensor* tensor, Backend* curBackend, std::map<Tensor*, std::shared_ptr<Tensor>>& cache);
private:
    Tensor *_getCopyTensor(Tensor *input, Tensor* outsideInput);
    Backend *mCPUBackend;
    std::shared_ptr<Execution> mExecution;
    std::vector<Tensor *> mWrapInputTensors;
    std::shared_ptr<Tensor> mWrapForRaster;
    std::map<Tensor *, std::tuple<Backend *, Backend *, std::shared_ptr<Tensor>>> mInputMaps;
    bool mStatic;
};

/** execution cast wrapper. insert tensor cast dynamic. */
class CastWrapExecution : public Execution {
public:
    CastWrapExecution(Backend* backend, DataType runT, const Op* op, Execution* exe)
                    : Execution(backend), mRunType(runT), mType(op->type()), mExecution(exe) {}
    CastWrapExecution(const CPUBackend::Creator* creator, const Op* op, Backend* backend,
                      const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, DataType runT);
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
private:
    OpType mType;
    const CPUBackend::Creator* mCreator;
    DataType mRunType;
    std::shared_ptr<Execution> mExecution;
    Tensor* mRasterInput;
    std::vector<Tensor*> mWrapInputs, mInputs;
    std::unique_ptr<Tensor> mRasterInputTensor;
    std::vector<std::unique_ptr<Tensor>> mWrapInputTensor;
    std::map<const Tensor*, const Tensor*> mCasts;
};
} // namespace MNN

#endif /* WrapExecution_hpp */
