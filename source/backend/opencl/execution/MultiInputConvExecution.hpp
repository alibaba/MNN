//
//  MultiInputConvExecution.hpp
//  MNN
//
//  Created by MNN on 2019/10/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MultiInputConvExecution_hpp
#define MultiInputConvExecution_hpp

#include "backend/opencl/execution/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class MultiInputConvExecution : public CommonExecution {
public:
    MultiInputConvExecution(const MNN::Op *op, Backend *backend);
    virtual ~MultiInputConvExecution();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    PadMode mPadMode;
    std::vector<int> mStrides{1, 1};
    std::vector<int> mPaddings{0, 0};
    std::vector<int> mDilations{1, 1};
    std::shared_ptr<Tensor> mFilter;

};
}
}

#endif /* MultiInputConvExecution_hpp */
