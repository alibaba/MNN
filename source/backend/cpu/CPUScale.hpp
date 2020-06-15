//
//  CPUScale.hpp
//  MNN
//
//  Created by MNN on 2018/08/07.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUScale_hpp
#define CPUScale_hpp

#include <MNN/Tensor.hpp>
#include "core/Execution.hpp"

namespace MNN {
class CPUScale : public Execution {
public:
    CPUScale(const Op *op, Backend *bn);
    virtual ~CPUScale();
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mScaleBias;
};

} // namespace MNN
#endif /* CPUScale_hpp */
