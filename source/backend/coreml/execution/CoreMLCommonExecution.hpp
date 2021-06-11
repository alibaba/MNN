//
//  CoreMLCommonExecution.hpp
//  MNN
//
//  Created by MNN on 2021/03/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_COREMLCOMMONEXECUTION_HPP
#define MNN_COREMLCOMMONEXECUTION_HPP
#include "core/Execution.hpp"
#include "CoreMLBackend.hpp"
#include <memory>

namespace MNN {

class CoreMLCommonExecution : public Execution {
public:
    CoreMLCommonExecution(Backend *backend, const Op *op);
    virtual ~CoreMLCommonExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
protected:
    void initLayer();
    void setLayerInputsAndOutputs(CoreML__Specification__NeuralNetworkLayer* layer, std::vector<std::string>&& inputs, std::vector<std::string>&& outputs);
    CoreML__Specification__NeuralNetworkLayer* mLayer_ = nullptr;
    CoreMLBackend* mCoreMLBackend;
    const Op* mOp;
};

} // namespace MNN
#endif // MNN_COREMLCOMMONEXECUTION_HPP
