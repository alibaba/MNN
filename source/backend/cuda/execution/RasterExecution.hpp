//
//  RasterExecution.hpp
//  MNN
//
//  Created by MNN on 2020/07/30.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef RasterExecution_hpp
#define RasterExecution_hpp
#include <map>
#include <memory>
#include <vector>
#include "backend/cuda/core/CUDABackend.hpp"
#include "core/Execution.hpp"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace CUDA {
class RasterExecution : public Execution {
public:
    RasterExecution(Backend *backend);
    virtual ~RasterExecution();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::vector<std::pair<void *, Tensor::InsideDescribe::Region *>> mTempInputCopy;
    bool mNeedZero = false;
    std::pair<bool, int> mFuseRaster;

    void *mOffset;
    std::shared_ptr<Tensor> offsetTensor;
};
} // namespace CUDA
} // namespace MNN

#endif
