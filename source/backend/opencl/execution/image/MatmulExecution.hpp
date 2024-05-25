//
//  MatmulExecution.hpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MatMulExecution_hpp
#define MatMulExecution_hpp

#include "CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class MatMulExecution : public CommonExecution {
public:
    MatMulExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend, bool transposeA, bool transposeB);
    virtual ~MatMulExecution() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    bool mTransposeA;
    bool mTransposeB;
    uint32_t mMaxWorkGroupSize;
    std::vector<int> mInput0Shape;
    std::vector<int> mInput1Shape;
    bool mAreadySetArg;
    OpenCLBackend *mOpenCLBackend;
};

} // namespace OpenCL
} // namespace MNN

#endif
