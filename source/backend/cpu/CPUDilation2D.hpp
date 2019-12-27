//
//  CPUDilation2D.hpp
//  MNN
//
//  Created by MNN on 2019/11/29.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUDilation2D_hpp
#define CPUDilation2D_hpp

#include <array>
#include "core/Execution.hpp"
#include "MNN_generated.h"

namespace MNN {
class CPUDilation2D : public Execution {
public:
    CPUDilation2D(Backend *b, const MNN::Op *op);
    virtual ~CPUDilation2D();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mWeight;
    std::array<int, 2> mKernelSize;
    std::array<int, 2> mStrides;
    std::array<int, 2> mDilations;
    std::array<int, 2> mPads;
    PadMode mPadMode;
};

} // namespace MNN

#endif /* CPUDilation2D_hpp */
