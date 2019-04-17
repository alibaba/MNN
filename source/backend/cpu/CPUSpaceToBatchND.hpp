//
//  CPUSpaceToBatchND.hpp
//  MNN
//
//  Created by MNN on 2018/12/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUSpaceToBatchND_hpp
#define CPUSpaceToBatchND_hpp

#include "Execution.hpp"

namespace MNN {

class CPUSpaceToBatchND : public Execution {
public:
    CPUSpaceToBatchND(const Op *op, Backend *bn);
    virtual ~CPUSpaceToBatchND() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int mPadTop;
    int mPadLeft;
    int mBlockShapeHeight;
    int mBlockShapeWidth;
};

} // namespace MNN

#endif /* CPUSpaceToBatchND_hpp */
