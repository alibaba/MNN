//
//  TRTCommonExecution.hpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef TRTCommonExecution_hpp
#define TRTCommonExecution_hpp
#include "TRTBackend.hpp"
#include "core/Execution.hpp"

using namespace std;
namespace MNN {

class TRTCommonExecution : public Execution {
public:
    TRTCommonExecution(Backend *backend, const Op *op);
    virtual ~TRTCommonExecution() = default;

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

protected:
    TRTBackend *mTrtBackend;
    const Op *mOp;
    std::vector<Tensor *> mInputs;
    std::vector<Tensor *> mOutputs;

    virtual std::vector<ITensor *> onEncode(const std::vector<ITensor *> &inputs) = 0;
};

} // namespace MNN
#endif /* TRTCommonExecution_hpp */
