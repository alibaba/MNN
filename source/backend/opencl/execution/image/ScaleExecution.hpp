//
//  ScaleExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef ScaleExecution_hpp
#define ScaleExecution_hpp

#include "CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class ScaleExecution : public CommonExecution {
public:
    ScaleExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~ScaleExecution();

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mScale;
    std::shared_ptr<Tensor> mBias;
    uint32_t mMaxWorkGroupSize;
    OpenCLBackend *mOpenCLBackend;
    bool mHasBias = false;
};

} // namespace OpenCL
} // namespace MNN
#endif /* ScaleExecution_hpp */
