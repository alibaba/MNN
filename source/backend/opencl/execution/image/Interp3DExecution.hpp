//
//  Interp3DExecution.hpp
//  MNN
//
//  Created by MNN on 2019/02/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef Interp3DExecution_hpp
#define Interp3DExecution_hpp

#include "CommonExecution.hpp"

namespace MNN {
namespace OpenCL {

class Interp3DExecution : public CommonExecution {
public:
    Interp3DExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~Interp3DExecution() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;


private:
    std::vector<uint32_t> mLWS{0, 0, 0, 0};
    std::vector<uint32_t> mGWS{0, 0, 0, 0};
    uint32_t mMaxWorkGroupSize;
    OpenCLBackend *mOpenCLBackend;
    float mCordTransform[6];
};

} // namespace OpenCL
} // namespace MNN
#endif /* InterpExecution_hpp */
