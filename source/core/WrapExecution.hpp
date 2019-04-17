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
#include "Backend.hpp"
#include "Execution.hpp"
#include "Macro.h"

namespace MNN {

/** execution wrapper. hiding cross-backend tensor converting. */
class WrapExecution : public Execution {
public:
    /**
     * @brief initializer.
     * @param CPUBackend    CPU backend.
     * @param execution     execution to be wrapped.
     */
    WrapExecution(Backend *CPUBackend, std::shared_ptr<Execution> execution);
    /**
     * @brief deinitializer.
     */
    virtual ~WrapExecution() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    Backend *mCPUBackend;
    std::shared_ptr<Execution> mExecution;
    std::vector<Tensor *> mWrapInputTensors;
    std::vector<std::tuple<Backend *, Backend *, Tensor *, std::shared_ptr<Tensor>>> mInputMaps;
};
} // namespace MNN

#endif /* WrapExecution_hpp */
