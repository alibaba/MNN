//
//  CPUPoolGrad.hpp
//  MNN
//
//  Created by jiangxiaotang on 2019/4/19.
//  Copyright © 2019 Alibaba. All rights reserved.
//

#ifndef CPUPoolGrad_hpp
#define CPUPoolGrad_hpp

#include "CPUBackend.hpp"

namespace MNN {
class CPUCommonPoolGrad : public Execution {
public:
    virtual ~ CPUCommonPoolGrad() = default;
    CPUCommonPoolGrad(Backend *b, const Pool *parameter) : Execution(b) {
        mStrideX = parameter->strideX();
        mStrideY = parameter->strideY();
        mKernelX = parameter->kernelX();
        mKernelY = parameter->kernelY();
        mGlobal  = parameter->isGlobal();
    }
    
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        if (mGlobal) {
            mKernelX = inputs[0]->width();
            mKernelY = inputs[0]->height();
        }
        return NO_ERROR;
    }

protected:
    int mStrideX;
    int mStrideY;
    int mKernelX;
    int mKernelY;
    bool mGlobal;
};
} // namespace MNN
#endif /* CPUPoolGrad_hpp */
