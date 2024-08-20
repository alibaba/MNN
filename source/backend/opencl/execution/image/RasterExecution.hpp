//
//  RasterExecution.hpp
//  MNN
//
//  Created by MNN on 2020/05/12.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef RaterExecution_hpp
#define RaterExecution_hpp
#include "CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class RasterExecution : public CommonExecution {
public:
    RasterExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~RasterExecution() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    
private:
    bool CanCombine(const std::vector<Tensor *> &outputs);
    std::map<Tensor*, cl::Buffer *> mTempInput;
    cl::Buffer *mTempOutput;
    OpenCLBackend *mOpenCLBackend;
    bool mNeedZero = false;
    bool mFast = false;

};

} // namespace OpenCL
} // namespace MNN

#endif
