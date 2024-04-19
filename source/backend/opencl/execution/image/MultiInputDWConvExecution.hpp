//
//  MultiInputDWConvExecution.hpp
//  MNN
//
//  Created by MNN on 2019/10/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MultiInputDWConvExecution_hpp
#define MultiInputDWConvExecution_hpp

#include "CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class MultiInputDWConvExecution : public CommonExecution {
public:
    MultiInputDWConvExecution(const MNN::Op *op, Backend *backend);
    virtual ~MultiInputDWConvExecution();

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    PadMode mPadMode;
    std::vector<int> mStrides{1, 1};
    std::vector<int> mPaddings{0, 0};
    std::vector<int> mDilations{1, 1};
    std::shared_ptr<Tensor> mFilter;
    bool isRelu = false;
    bool isRelu6 = false;

};
}
}

#endif /* MultiInputDWConvExecution_hpp */
