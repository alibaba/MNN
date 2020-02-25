//
//  CPUMatrixBandPart.hpp
//  MNN
//
//  Created by MNN on 2019/09/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef CPUMatrixBandPart_hpp
#define CPUMatrixBandPart_hpp

#include "backend/cpu/CPUBackend.hpp"
namespace MNN {

class CPUMatrixBandPart : public Execution {
public:
    CPUMatrixBandPart(Backend *backend) : Execution(backend) {
        // Do nothing
    }
    virtual ~CPUMatrixBandPart() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mMask;
};
} // namespace MNN

#endif // CPUMatrixBandPart_hpp
