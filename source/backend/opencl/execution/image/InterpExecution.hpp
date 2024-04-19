//
//  InterpExecution.hpp
//  MNN
//
//  Created by MNN on 2019/02/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef InterpExecution_hpp
#define InterpExecution_hpp

#include "CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class InterpExecution : public CommonExecution {
public:
    InterpExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~InterpExecution() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;


private:
    std::vector<uint32_t> mLWS{0, 0, 0, 0};
    std::vector<uint32_t> mGWS{0, 0, 0, 0};
    uint32_t mMaxWorkGroupSize;
    OpenCLBackend *mOpenCLBackend;
    float mCordTransform[4];
};

} // namespace OpenCL
} // namespace MNN
#endif /* InterpExecution_hpp */
