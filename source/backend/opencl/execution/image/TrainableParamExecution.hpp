//
//  TrainableParamExecution.hpp
//  MNN
//
//  Created by MNN on 2019/10/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef TrainableParamExecution_hpp
#define TrainableParamExecution_hpp

#include "CommonExecution.hpp"
namespace MNN {
namespace OpenCL {

class TrainableParamExecution : public CommonExecution {
public:
    TrainableParamExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~TrainableParamExecution();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    const MNN::Op *mOp;
    bool mInitialized;
};

} // namespace OpenCL
} // namespace MNN
#endif /* TrainableParamExecution_hpp */
