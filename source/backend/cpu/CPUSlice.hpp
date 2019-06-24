//
//  CPUSlice.hpp
//  MNN
//
//  Created by MNN on 2018/07/19.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUSlice_hpp
#define CPUSlice_hpp

#include "Execution.hpp"

namespace MNN {
class CPUSlice : public Execution {
public:
    CPUSlice(Backend *b, int axis);
    virtual ~CPUSlice() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int mAxis;
    std::shared_ptr<Tensor> mTempInput;
};

} // namespace MNN

#endif /* CPUSlice_hpp */
