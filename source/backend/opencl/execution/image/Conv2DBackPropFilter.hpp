//
//  Conv2DBackPropFilter.hpp
//  MNN
//
//  Created by MNN on 2019/10/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Conv2DBackPropFilter_hpp
#define Conv2DBackPropFilter_hpp

#include "backend/opencl/execution/image/CommonExecution.hpp"
#include "backend/opencl/core/runtime/OpenCLRuntime.hpp"

namespace MNN {
namespace OpenCL {

class Conv2DBackPropFilter : public CommonExecution {
public:
    Conv2DBackPropFilter(const MNN::Op *op, Backend *backend);
    virtual ~Conv2DBackPropFilter();

    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::vector<int> mStrides;
    std::vector<int> mPaddings;
    std::vector<int> mDilations;
    std::vector<int> mKernels;
};
}
}

#endif /* Conv2DBackPropFilter_hpp */
