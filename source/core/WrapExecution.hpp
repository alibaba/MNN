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

private:
    Tensor *_getCopyTensor(Tensor *input);
    Backend *mCPUBackend;
    std::shared_ptr<Execution> mExecution;
    std::vector<Tensor *> mWrapInputTensors;
    std::shared_ptr<Tensor> mWrapForRaster;
    std::map<Tensor *, std::tuple<Backend *, Backend *, std::shared_ptr<Tensor>>> mInputMaps;
    bool mStatic;
};
} // namespace MNN

#endif /* WrapExecution_hpp */
