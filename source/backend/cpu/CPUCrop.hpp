//
//  CPUCrop.hpp
//  MNN
//
//  Created by MNN on 2018/10/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUCrop_hpp
#define CPUCrop_hpp

#include "Execution.hpp"

namespace MNN {
class CPUCrop : public Execution {
public:
    CPUCrop(Backend* b, const MNN::Op* op);
    virtual ~CPUCrop() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

private:
    int mAxis = 2;
    std::vector<int> mOffsets;

    static void cropCopy(const Tensor* inputTensor, Tensor* outputTensor, const std::vector<int>& offsets);
};
} // namespace MNN

#endif /* CPUCrop_hpp */
