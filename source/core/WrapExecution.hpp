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
private:
    Tensor *_getCopyTensor(Tensor *input);
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
    CastWrapExecution(Backend* backend, halide_type_t runT, const Op* op, Execution* exe)
                    : Execution(backend), runType(runT), mType(op->type()), mExecution(exe) {}
    CastWrapExecution(const CPUBackend::Creator* creator, const Op* op, Backend* backend,
                      const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs, halide_type_t runT);
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
private:
    OpType mType;
    const CPUBackend::Creator* mCreator;
    halide_type_t runType;
    std::shared_ptr<Execution> mExecution;
    Tensor* mRasterInput;
    std::vector<Tensor*> mWrapInputs, mInputs;
    std::unique_ptr<Tensor> mRasterInputTensor;
    std::vector<std::unique_ptr<Tensor>> mWrapInputTensor;
    std::map<const Tensor*, const Tensor*> mCasts;
    std::map<const Tensor*, std::vector<float>> mScales;
    bool firstResize = true;
};
class CheckNANExecution : public Execution {
public:
    CheckNANExecution(Execution* exe);
    virtual ~CheckNANExecution();
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
private:
    Execution* mExecution;
};
} // namespace MNN

#endif /* WrapExecution_hpp */
