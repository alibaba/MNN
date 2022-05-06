//
//  CPUDet.hpp
//  MNN
//
//  Created by MNN on 2018/08/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUDet_hpp
#define CPUDet_hpp

#include <MNN/Tensor.hpp>
#include "core/Execution.hpp"

namespace MNN {
class CPUDet : public Execution {
public:
    CPUDet(Backend *bn) : Execution(bn) { }
    virtual ~CPUDet() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mTempMat, mTempRowPtrs;
};

} // namespace MNN
#endif /* CPUDet_hpp */
